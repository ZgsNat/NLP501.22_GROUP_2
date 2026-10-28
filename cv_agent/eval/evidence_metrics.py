"""Faithfulness metrics for H4: unsupported-claim rate, evidence precision,
ERASER-style comprehensiveness / sufficiency (re-scoring with the criteria scorer)."""
from __future__ import annotations

import re
from typing import Dict, List

from cv_agent.ingest.pipeline import ingest_text
from cv_agent.ranking.scoring import score_candidate
from cv_agent.schema import CV, JD, Claim, CriteriaProfile
from cv_agent.verify.evidence import _norm, check_claim
from cv_agent.verify.nli import LexicalNLI

_ORACLE_NLI = LexicalNLI()


def unsupported_claim_rate(claims: Dict[str, List[Claim]], cvs: Dict[str, CV]) -> float:
    """Share of claims SHOWN to HR (status != no_evidence/removed) that a strict auditor rejects."""
    shown = [c for cl in claims.values() for c in cl if c.status not in {"no_evidence", "removed"}]
    if not shown:
        return float("nan")
    bad = sum(1 for c in shown if not check_claim(c, cvs[c.cv_id].masked_text, _ORACLE_NLI, fuzzy_threshold=0.95)[0])
    return bad / len(shown)


def evidence_precision(claims: Dict[str, List[Claim]], rationale: Dict[str, List[str]]) -> float:
    """Share of supported claim spans that overlap an annotated rationale span."""
    supported = [c for cl in claims.values() for c in cl if c.status in {"supported", "unverified"}]
    if not supported:
        return float("nan")
    hit = 0
    for c in supported:
        gold = [_norm(r) for r in rationale.get(c.cv_id, [])]
        span = _norm(c.span_text)
        hit += bool(span) and any(span in g or g in span for g in gold)
    return hit / len(supported)


def _rescore(text: str, cv_id: str, jd: JD, profile: CriteriaProfile) -> float:
    return score_candidate(ingest_text(text, cv_id), jd, profile).potential_score


def eraser_scores(cv: CV, claims: List[Claim], jd: JD, profile: CriteriaProfile) -> Dict[str, float]:
    """comprehensiveness = s(full) - s(full minus rationale); sufficiency = s(full) - s(rationale only)."""
    spans = [c.span_text for c in claims if c.status in {"supported", "unverified"} and c.span_text]
    if not spans:
        return {"comprehensiveness": float("nan"), "sufficiency": float("nan")}
    full = _rescore(cv.masked_text, cv.cv_id, jd, profile)
    removed = cv.masked_text
    for s in spans:
        removed = re.sub(re.escape(s), " ", removed, flags=re.I)
    only = "\n".join(spans)
    return {"comprehensiveness": full - _rescore(removed, cv.cv_id, jd, profile),
            "sufficiency": full - _rescore(only, cv.cv_id, jd, profile)}
