"""Cross-encoder rerank top-k -> top-n (FR-15, config B4).

``cross_encoder`` uses sentence-transformers CrossEncoder (e.g. BAAI/bge-reranker-base).
``lexical`` is a dependency-free fallback scoring requirement coverage; report it as such.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Protocol, Sequence, Tuple

from cv_agent.llm.context import cv_digest, jd_digest
from cv_agent.ranking.scoring import requirement_match
from cv_agent.schema import CV, JD

log = logging.getLogger(__name__)


class Reranker(Protocol):
    name: str

    def score(self, jd: JD, cvs: Sequence[CV]) -> List[float]: ...


class LexicalReranker:
    name = "lexical"

    def score(self, jd: JD, cvs: Sequence[CV]) -> List[float]:
        out = []
        for cv in cvs:
            num = den = 0.0
            for req in jd.requirements:
                w = 2.0 if req.must else 1.0
                sat, _ = requirement_match(cv, req)
                num += w * sat
                den += w
            out.append((num / den if den else 0.0) * (0.5 + 0.5 * cv.parse_quality))
        return out


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_chars: int = 1200) -> None:
        from sentence_transformers import CrossEncoder  # optional heavy dependency
        self.model = CrossEncoder(model_name, max_length=512)
        self.name, self.max_chars = f"ce:{model_name}", max_chars

    def score(self, jd: JD, cvs: Sequence[CV]) -> List[float]:
        # 512-token window: raw JD + raw CV truncated the CV away, so pass the requirement list and the
        # CV's evidence lines (same digests the LLM sees)
        query = jd_digest(jd)
        pairs = [(query, cv_digest(cv, self.max_chars)) for cv in cvs]
        logits = self.model.predict(pairs, batch_size=8)
        return [1 / (1 + math.exp(-float(x))) for x in logits]


def build_reranker(cfg: Dict) -> Reranker:
    if cfg.get("reranker", "lexical") == "cross_encoder":
        try:
            return CrossEncoderReranker(cfg.get("model", "BAAI/bge-reranker-base"))
        except Exception as exc:
            log.warning("Cross-encoder unavailable (%s); falling back to lexical reranker", exc)
    return LexicalReranker()


def rerank(reranker: Reranker, jd: JD, cvs: Sequence[CV], top_n: int) -> List[Tuple[str, float]]:
    scores = reranker.score(jd, cvs)
    ranked = sorted(zip([c.cv_id for c in cvs], scores), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:top_n]
