"""Explanation card L1–L6 for the top-N (FR-27) + contrastive / counterfactual (FR-28).

L1 Must-have / knock-out    L2 Score decomposition (sum == score)
L3 Evidence spans (verified claims + LLM call ids)
L4 Contrastive: why this candidate ranks above the next one
L5 Counterfactual: score change if the candidate had a missing skill
L6 Data quality & process notes (parse confidence, flags, arbiter, masking)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

from cv_agent.ingest.extractor import DEPTH_LABELS
from cv_agent.ranking.scoring import VALUE_AXES, score_candidate
from cv_agent.schema import AXES, CV, JD, Claim, CriteriaProfile, Entity, ScoredCandidate, Span

L2_TOLERANCE = 1e-9


@dataclass
class ExplanationCard:
    rank: int
    cv_id: str
    score: float
    L1: Dict[str, Any] = field(default_factory=dict)
    L2: List[Dict[str, Any]] = field(default_factory=list)
    L3: List[Dict[str, Any]] = field(default_factory=list)
    L4: Dict[str, Any] = field(default_factory=dict)
    L5: List[Dict[str, Any]] = field(default_factory=list)
    L6: Dict[str, Any] = field(default_factory=dict)


def decomposition(c: ScoredCandidate, profile: CriteriaProfile) -> List[Dict[str, Any]]:
    rows = [{"axis": a, "weight": round(profile.weights.get(a, 0.0), 4), "S": round(c.axis_scores.get(a, 0.0), 4),
             "contribution": c.contributions.get(a, 0.0), "evidence": c.evidence.get(a, [])} for a in AXES]
    total = sum(r["contribution"] for r in rows)
    if abs(total - c.score) > L2_TOLERANCE:
        raise AssertionError(f"L2 decomposition {total} != score {c.score} for {c.cv_id}")
    return rows


def contrastive(a: ScoredCandidate, b: Optional[ScoredCandidate]) -> Dict[str, Any]:
    if b is None:
        return {}
    # both knocked out => contributions are all 0 and the order comes from potential_score: explain that instead
    both_out = not a.must_have_ok and not b.must_have_ok
    per_axis = (lambda c: c.axis_scores) if both_out else (lambda c: c.contributions)
    diffs = sorted(((ax, per_axis(a).get(ax, 0) - per_axis(b).get(ax, 0)) for ax in AXES), key=lambda t: -t[1])
    return {"vs": b.cv_id, "delta_score": round(a.score - b.score, 4),
            "basis": "axis_scores (both knocked out)" if both_out else "contributions",
            "delta_potential": round(a.potential_score - b.potential_score, 4),
            "advantages": [{"axis": ax, "delta": round(d, 4)} for ax, d in diffs if d > 1e-6][:3],
            "disadvantages": [{"axis": ax, "delta": round(d, 4)} for ax, d in diffs[::-1] if d < -1e-6][:2],
            "arbiter": [n for n in a.arbiter_notes if b.cv_id in n]}


def counterfactuals(cv: CV, jd: JD, profile: CriteriaProfile, current: ScoredCandidate,
                    limit: int = 3) -> List[Dict[str, Any]]:
    """Add one missing required value to the CV and recompute the score."""
    have = {(e.type, e.value) for e in cv.entities}
    out = []
    for req in sorted(jd.requirements, key=lambda r: (not r.must, -profile.weights.get(r.axis, 0))):
        etype = VALUE_AXES.get(req.axis)
        if not etype or not req.values or any((etype, v) in have for v in req.values):
            continue
        value = req.values[0]
        fake = Entity(etype, value, Span(0, 0, value))
        alt = score_candidate(replace(cv, entities=cv.entities + [fake]), jd, profile)
        out.append({"if_has": value, "axis": req.axis, "must": req.must,
                    "new_score": round(alt.score, 4), "delta": round(alt.score - current.score, 4),
                    "passes_must_have": alt.must_have_ok})
        if len(out) >= limit:
            break
    return out


def build_card(rank: int, c: ScoredCandidate, nxt: Optional[ScoredCandidate], cv: CV, jd: JD,
               profile: CriteriaProfile, claims: List[Claim], llm_log: Dict[str, Dict]) -> ExplanationCard:
    depth = cv.depth.level if cv.depth else 1
    return ExplanationCard(
        rank=rank, cv_id=c.cv_id, score=round(c.score, 4),
        L1={"must_have_ok": c.must_have_ok, "knockout_reasons": c.knockout_reasons,
            "note": "Không tự động loại — HR quyết định" if not c.must_have_ok else ""},
        L2=decomposition(c, profile),
        L3=[{"claim": cl.text, "span": cl.span_text, "status": cl.status, "axis": cl.axis,
             "llm_call_id": cl.llm_call_id, "prompt_version": llm_log.get(cl.llm_call_id or "", {}).get("prompt_version")}
            for cl in claims],
        L4=contrastive(c, nxt),
        L5=counterfactuals(cv, jd, profile, c),
        L6={"parse_quality": cv.parse_quality, "flags": cv.flags, "language": cv.language,
            "automation_depth": f"{depth} – {DEPTH_LABELS.get(depth, '')}",
            "depth_inferred": bool(cv.depth and not cv.depth.evidence),
            "years_experience": cv.years_experience, "pii_masked": cv.pii_found,
            "arbiter_notes": c.arbiter_notes, "potential_score_without_knockout": round(c.potential_score, 4),
            "unverified_claims": sum(1 for cl in claims if cl.status != "supported")},
    )


def build_cards(ranking: List[ScoredCandidate], cvs: Dict[str, CV], jd: JD, profile: CriteriaProfile,
                claims: Dict[str, List[Claim]], llm_records: List[Dict], top_n: int = 10) -> List[ExplanationCard]:
    log_by_id = {r["call_id"]: r for r in llm_records}
    top = ranking[:top_n]
    return [build_card(i + 1, c, top[i + 1] if i + 1 < len(top) else None, cvs[c.cv_id], jd, profile,
                       claims.get(c.cv_id, []), log_by_id) for i, c in enumerate(top)]
