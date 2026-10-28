"""Criteria scoring (FR-16, FR-17, config B3): S(c,k) in [0,1] per axis and
Score = 1[H(c)] x sum_k w(k) * S(c,k). Knock-outs are recorded, never deleted (NFR-06)."""
from __future__ import annotations

from typing import Dict, List, Tuple

from cv_agent.ingest.taxonomy import SKILLS
from cv_agent.schema import AXES, CV, JD, CriteriaProfile, Requirement, ScoredCandidate

VALUE_AXES = {"tools": "tool", "languages": "language", "test_types": "test_type", "cicd": "cicd",
              "certification": "certification", "foreign_lang": "foreign_lang"}
RELATED_CREDIT = 0.5      # partial credit when CV has a related (not identical) skill
MUST_WEIGHT, NICE_WEIGHT = 2.0, 1.0
DEPTH_SHORTFALL_CREDIT = 0.75
DEFAULT_YEARS_SATURATION = 5.0
DEFAULT_TOOLS_SATURATION = 4


def _cv_values(cv: CV, axis: str) -> Dict[str, str]:
    """canonical value -> first span text in CV."""
    etype = VALUE_AXES[axis]
    out: Dict[str, str] = {}
    for e in cv.entities:
        if e.type == etype and e.value not in out:
            out[e.value] = e.span.text
    return out


def requirement_match(cv: CV, req: Requirement) -> Tuple[float, List[str]]:
    """Return (satisfaction in [0,1], evidence spans)."""
    if req.axis == "experience":
        years = cv.years_experience
        span = [e.span.text for e in cv.entities if e.type == "years"][:1]
        return (min(1.0, years / req.min_level) if req.min_level else 1.0), span
    if req.axis == "depth":
        level = cv.depth.level if cv.depth else 1
        spans = [s.text for s in (cv.depth.evidence if cv.depth else [])][:2]
        if level >= req.min_level:
            return 1.0, spans
        return DEPTH_SHORTFALL_CREDIT * level / req.min_level, spans
    have = _cv_values(cv, req.axis)
    hits = [have[v] for v in req.values if v in have]
    if hits:
        return 1.0, hits[:3]
    related = [have[r] for v in req.values for r in (SKILLS[v].related if v in SKILLS else ()) if r in have]
    return (RELATED_CREDIT, related[:2]) if related else (0.0, [])


def must_have_ok(cv: CV, jd: JD) -> Tuple[bool, List[str]]:
    reasons = []
    for req in jd.must_haves:
        sat, _ = requirement_match(cv, req)
        if sat < 1.0:
            want = " / ".join(req.values) if req.values else f"≥ {req.min_level:g}"
            reasons.append(f"Thiếu must-have [{req.axis}]: {want}")
    return not reasons, reasons


def _default_axis_score(cv: CV, axis: str) -> Tuple[float, List[str]]:
    """S(c,k) when the JD says nothing about axis k: generic strength signal."""
    if axis == "experience":
        return min(1.0, cv.years_experience / DEFAULT_YEARS_SATURATION), []
    if axis == "depth":
        level = cv.depth.level if cv.depth else 1
        return level / 4.0, [s.text for s in (cv.depth.evidence if cv.depth else [])][:2]
    have = _cv_values(cv, axis)
    if axis == "tools":
        return min(1.0, len(have) / DEFAULT_TOOLS_SATURATION), list(have.values())[:3]
    return (1.0 if have else 0.0), list(have.values())[:2]


def axis_scores(cv: CV, jd: JD) -> Tuple[Dict[str, float], Dict[str, List[str]]]:
    scores: Dict[str, float] = {}
    evidence: Dict[str, List[str]] = {}
    for axis in AXES:
        reqs = [r for r in jd.requirements if r.axis == axis]
        if not reqs:
            scores[axis], evidence[axis] = _default_axis_score(cv, axis)
            continue
        num = den = 0.0
        spans: List[str] = []
        for req in reqs:
            w = MUST_WEIGHT if req.must else NICE_WEIGHT
            sat, ev = requirement_match(cv, req)
            num += w * sat
            den += w
            spans.extend(s for s in ev if s not in spans)
        scores[axis] = num / den if den else 0.0
        evidence[axis] = spans[:4]
    return scores, evidence


def score_candidate(cv: CV, jd: JD, profile: CriteriaProfile, retrieval_score: float = 0.0,
                    rerank_score: float = 0.0) -> ScoredCandidate:
    s, evidence = axis_scores(cv, jd)
    ok, reasons = must_have_ok(cv, jd)
    potential = {axis: profile.weights.get(axis, 0.0) * s[axis] for axis in AXES}
    indicator = 1.0 if ok else 0.0
    contributions = {axis: indicator * v for axis, v in potential.items()}
    return ScoredCandidate(
        cv_id=cv.cv_id, retrieval_score=retrieval_score, rerank_score=rerank_score,
        axis_scores={k: round(v, 6) for k, v in s.items()}, contributions=contributions,
        must_have_ok=ok, knockout_reasons=reasons, score=sum(contributions.values()),
        potential_score=sum(potential.values()), evidence=evidence,
    )


def rank(scored: List[ScoredCandidate]) -> List[ScoredCandidate]:
    """Score desc; ties broken by potential score (orders knocked-out candidates by how close they
    come, e.g. when nobody passes the must-haves), then cross-encoder, retrieval, id (deterministic)."""
    return sorted(scored, key=lambda c: (-round(c.score, 9), -round(c.potential_score, 9), -c.rerank_score,
                                         -c.retrieval_score, c.cv_id))
