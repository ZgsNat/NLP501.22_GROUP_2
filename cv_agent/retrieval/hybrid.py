"""Hybrid retrieval: BM25 + dense fused by Reciprocal Rank Fusion (FR-13, config B2)
and the JD -> query builder with optional ESCO query expansion (FR-14, Loop 2)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from cv_agent.ingest.taxonomy import expand
from cv_agent.retrieval.bm25 import BM25Index
from cv_agent.retrieval.dense import DenseIndex, Encoder
from cv_agent.schema import CV, JD


def rrf(rankings: Iterable[Sequence[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def cv_index_text(cv: CV) -> str:
    """Normalised view: masked text + canonical skill names (so aliases match)."""
    canon = " ".join(sorted({e.value for e in cv.entities if e.type != "years"}))
    return f"{cv.masked_text}\n{canon}"


def build_query(jd: JD, expansion: Sequence[str] = ()) -> str:
    must_terms = [v for r in jd.must_haves for v in r.values]
    nice_terms = [v for r in jd.nice_to_haves for v in r.values]
    # repeat must-have terms to up-weight them in BM25
    return " ".join([jd.title] + must_terms * 2 + nice_terms + list(expansion))


def expansion_terms(jd: JD) -> List[str]:
    return expand([v for r in jd.requirements for v in r.values])


class HybridRetriever:
    def __init__(self, cvs: Sequence[CV], encoder: Encoder, mode: str = "hybrid", rrf_k: int = 60) -> None:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"retrieval mode must be bm25|dense|hybrid, got {mode}")
        self.mode, self.rrf_k = mode, rrf_k
        docs = [(cv.cv_id, cv_index_text(cv)) for cv in cvs]
        self.bm25 = BM25Index().fit(docs) if mode in {"bm25", "hybrid"} else None
        self.dense = DenseIndex(encoder).fit(docs) if mode in {"dense", "hybrid"} else None

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        n_all = len(self.bm25.ids if self.bm25 else self.dense.ids)
        if self.mode == "bm25":
            return self.bm25.search(query, top_k)
        if self.mode == "dense":
            return self.dense.search(query, top_k)
        lists = [self.bm25.search(query, n_all), self.dense.search(query, n_all)]
        return rrf(lists, self.rrf_k)[:top_k]
