"""Hybrid retrieval on Qdrant: BM25 sparse (exact tool names) + dense embeddings (meaning), fused with RRF,
then a cross-encoder rerank. All models are open-source (Hugging Face) and run locally via fastembed/ONNX.

Why hybrid: Resume2Vec (2025) found Software Testing is the one domain where keyword ATS beats dense
embeddings — tool names like "Playwright" are hard lexical entities. So we keep BM25 in the loop.
"""

import uuid
from dataclasses import dataclass
from functools import cached_property, lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from app.config import Settings, get_settings
from app.rag.corpus import Doc


@dataclass
class Hit:
    id: str
    score: float
    payload: dict


class KnowledgeStore:
    def __init__(self, s: Settings):
        self.s = s
        url = s.qdrant_url
        self.client = QdrantClient(location=":memory:") if url == ":memory:" else QdrantClient(url=url)

    @cached_property
    def dense(self) -> TextEmbedding:
        return TextEmbedding(self.s.embed_dense_model)

    @cached_property
    def sparse(self) -> SparseTextEmbedding:
        return SparseTextEmbedding(self.s.embed_sparse_model)

    @cached_property
    def reranker(self) -> TextCrossEncoder:
        return TextCrossEncoder(self.s.rerank_model)

    def index(self, collection: str, docs: list[Doc]) -> None:
        texts = [d.text for d in docs]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(
            collection,
            vectors_config={"dense": models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE)},
            sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        self.client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, d.id)),
                    vector={
                        "dense": dv.tolist(),
                        "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                    },
                    payload={"doc_id": d.id, "text": d.text, **d.meta},
                )
                for d, dv, sv in zip(docs, dense, sparse)
            ],
        )

    def hybrid_search(self, collection: str, query: str, limit: int, where: models.Filter | None = None) -> list[Hit]:
        dv = next(iter(self.dense.query_embed(query))).tolist()
        sv = next(iter(self.sparse.query_embed(query)))
        res = self.client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=dv, using="dense", limit=limit * 2, filter=where),
                models.Prefetch(
                    query=models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                    using="bm25",
                    limit=limit * 2,
                    filter=where,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return [Hit(id=p.payload["doc_id"], score=p.score, payload=p.payload) for p in res.points]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.reranker.rerank(query, texts)]


@lru_cache
def get_store() -> KnowledgeStore:
    return KnowledgeStore(get_settings())
