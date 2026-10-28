"""NLI entailment check for verify_evidence (FR-22).

``hf`` uses an MNLI cross-encoder (e.g. cross-encoder/nli-deberta-v3-base or a
multilingual XNLI model). ``lexical`` is a dependency-free fallback: every skill
entity and number mentioned in the claim must appear in the premise.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, Protocol

from cv_agent.ingest.extractor import extract_entities
from cv_agent.retrieval.bm25 import tokenize

log = logging.getLogger(__name__)


class NLIModel(Protocol):
    name: str

    def entails(self, premise: str, hypothesis: str) -> float: ...


class LexicalNLI:
    name = "lexical"
    _NUM = re.compile(r"\d+(?:[.,]\d+)?")
    # claim boilerplate that never needs to appear in the CV span
    _GENERIC = {"experience", "kinh", "nghiệm", "has", "have", "strong", "hands", "on", "with", "using", "candidate",
                "ứng", "viên", "có", "skill", "skills", "knowledge", "proficient", "good", "solid", "years", "year", "năm",
                "đáp", "yêu", "cầu", "meets", "meet", "requirement", "requirements", "satisfies", "shows", "demonstrates",
                "the", "is", "in", "of", "tools", "depth", "languages", "test_types", "cicd", "certification",
                "foreign_lang", "level", "worked", "works", "used", "uses"}

    def entails(self, premise: str, hypothesis: str) -> float:
        p_ents = {e.value for e in extract_entities(premise)}
        h_ents = {e.value for e in extract_entities(hypothesis) if e.type != "years"}
        if h_ents and not h_ents <= p_ents:
            return 0.0
        h_nums = set(self._NUM.findall(hypothesis))
        if h_nums and not h_nums <= set(self._NUM.findall(premise)):
            return 0.0
        if h_ents:
            return 1.0
        h_tok = {t for t in tokenize(hypothesis) if t not in self._GENERIC}
        if not h_tok:
            return 0.5
        return len(h_tok & set(tokenize(premise))) / len(h_tok)


class HFNLI:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base") -> None:
        from sentence_transformers import CrossEncoder  # optional heavy dependency
        self.model = CrossEncoder(model_name)
        self.name = f"hf:{model_name}"
        labels = getattr(self.model.config, "id2label", {}) or {}
        self.entail_idx = next((int(i) for i, l in labels.items() if "entail" in str(l).lower()), 1)

    def entails(self, premise: str, hypothesis: str) -> float:
        import numpy as np
        logits = np.asarray(self.model.predict([(premise, hypothesis)]))[0]
        probs = np.exp(logits - logits.max())
        return float(probs[self.entail_idx] / probs.sum())


def build_nli(cfg: Dict) -> NLIModel:
    if cfg.get("nli", "lexical") == "hf":
        try:
            return HFNLI(cfg.get("nli_model", "cross-encoder/nli-deberta-v3-base"))
        except Exception as exc:
            log.warning("HF NLI unavailable (%s); falling back to lexical NLI", exc)
    return LexicalNLI()
