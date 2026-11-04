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
        self.degraded = False  # True = Qdrant was unreachable and QDRANT_FALLBACK_MEMORY put an empty index in its place
        url = s.qdrant_url
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
            return
        try:
            probe = QdrantClient(url=url, timeout=1.5)
            probe.get_collections()
            self.client = probe
        except Exception as e:
            if not s.qdrant_fallback_memory:
                raise RuntimeError(f"Qdrant at {url} is not reachable ({type(e).__name__}). Start it with "
                                   "`docker compose up -d qdrant`, or set QDRANT_FALLBACK_MEMORY=true to run without "
                                   "search (an empty index).") from e
            import logging
            logging.getLogger(__name__).warning("Qdrant at %s offline; QDRANT_FALLBACK_MEMORY: empty in-memory index, "
                                                "search finds nothing", url)
            self.client, self.degraded = QdrantClient(location=":memory:"), True

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
        self.index_items(collection, [(d.id, d.text, {"doc_id": d.id, "text": d.text, **d.meta}) for d in docs])

    def index_items(self, collection: str, items: list[tuple[str, str, dict]]) -> None:
        """(key, text to embed, payload) → one point each. CV chunks carry payload doc_id = cv_id + chunk_id."""
        self.upsert_items(collection, items, recreate=True)

    def upsert_items(self, collection: str, items: list[tuple[str, str, dict]], recreate: bool = False) -> None:
        """Add or replace points; `recreate` drops the collection first (full re-index)."""
        texts = [t for _, t, _ in items]
        dense = list(self.dense.embed(texts))
        sparse = list(self.sparse.embed(texts))
        if recreate and self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config={"dense": models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        self.client.upsert(
            collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
                    vector={
                        "dense": dv.tolist(),
                        "bm25": models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                    },
                    payload=payload,
                )
                for (key, _, payload), dv, sv in zip(items, dense, sparse)
            ],
        )

    def hybrid_search(self, collection: str, query: str, limit: int, where: models.Filter | None = None,
                      mode: str = "hybrid") -> list[Hit]:
        """mode: hybrid (BM25 + dense, fused by RRF) | bm25 | dense. The single modes are the B0/B1 baselines."""
        if not self.client.collection_exists(collection):
            if self.degraded:  # running without Qdrant on purpose: no index, no hits
                return []
            raise RuntimeError(f"Qdrant has no collection '{collection}': index the data first "
                               "(`cd backend && uv run python -m app.rag.ingest`).")
        dense = models.Prefetch(query=next(iter(self.dense.query_embed(query))).tolist(), using="dense",
                                limit=limit * 2, filter=where)
        sv = next(iter(self.sparse.query_embed(query)))
        bm25 = models.Prefetch(query=models.SparseVector(indices=sv.indices.tolist(), values=sv.values.tolist()),
                               using="bm25", limit=limit * 2, filter=where)
        if mode == "hybrid":
            res = self.client.query_points(collection, prefetch=[dense, bm25],
                                           query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit)
        else:
            p = dense if mode == "dense" else bm25
            res = self.client.query_points(collection, query=p.query, using=p.using, query_filter=where, limit=limit)
        return [Hit(id=p.payload["doc_id"], score=p.score, payload=p.payload) for p in res.points]

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [float(x) for x in self.reranker.rerank(query, texts)]

    def rerank_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [float(x) for x in self.reranker.rerank_pairs(pairs)] if pairs else []


@lru_cache
def get_store() -> KnowledgeStore:
    return KnowledgeStore(get_settings())
