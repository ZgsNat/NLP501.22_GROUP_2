"""Shortlist critic: checks the goal the agent works towards and says how to fix what is missing.

The checks are deterministic (auditable, reproducible); the *planner* decides how to act on them.
A goal that cannot be reached any more (e.g. not enough must-have candidates in the whole corpus after
every search) is reported as ``unreachable`` instead of blocking the run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

LOW_PARSE_QUALITY = 0.5


@dataclass(frozen=True)
class Finding:
    goal: str
    ok: bool
    detail: str
    suggestion: Optional[Dict[str, Any]] = None   # {"action": ..., "args": {...}} the planner may take
    hard: bool = True                              # soft findings are advice and never block "finish"
    unreachable: bool = False


@dataclass(frozen=True)
class CriticReport:
    passed: bool
    findings: Tuple[Finding, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "findings": [asdict(f) for f in self.findings]}


def shortlist_signature(state: Dict[str, Any], top_n: int) -> str:
    """Who is in the shortlist (order-free: swapping two shortlisted CVs needs no new claims)."""
    return "|".join(sorted(c.cv_id for c in state.get("ranking", [])[:top_n]))


def open_near_ties(state: Dict[str, Any], max_rank: int, max_delta: float) -> List[Tuple[str, str]]:
    """Adjacent pairs in the shortlist zone that are close, both pass the must-haves, and were not compared."""
    ranking = state.get("ranking", [])
    done = {tuple(p) for p in state.get("compared_pairs", [])}
    moved = set(state.get("swapped", []))  # a swapped candidate is not judged again (moves one place at most)
    out = []
    for a, b in zip(ranking[:max_rank], ranking[1:max_rank + 1]):
        if a.must_have_ok and b.must_have_ok and abs(a.score - b.score) < max_delta \
                and (a.cv_id, b.cv_id) not in done and (b.cv_id, a.cv_id) not in done \
                and a.cv_id not in moved and b.cv_id not in moved:
            out.append((a.cv_id, b.cv_id))
    return out


def critique(state: Dict[str, Any], cfg: Dict[str, Any], searches_left: int, llm_usable: bool = True,
             compares_used: Optional[int] = None) -> CriticReport:
    """``searches_left``: searches the agent may still run (0 when search is blocked). ``llm_usable``: LLM tools can
    still run (not in fallback, judge enabled). Goals that need them become ``unreachable`` otherwise, so the
    critic never demands an action no tool can take. ``compares_used``: pair judgements over the whole run."""
    goals, top_n = cfg["agent"]["goals"], cfg["report"]["top_n"]
    shortlist = state.get("ranking", [])[:top_n]
    findings: List[Finding] = []

    # 1. enough candidates that pass every must-have
    need = goals["min_must_pass_shortlist"]
    have = sum(c.must_have_ok for c in shortlist)
    in_pool = state.get("must_pass_count", 0)
    if have >= min(need, top_n):
        findings.append(Finding("must_pass_shortlist", True, f"{have}/{top_n} ứng viên đạt mọi must-have"))
    elif searches_left > 0:
        findings.append(Finding("must_pass_shortlist", False,
                                f"chỉ {have} ứng viên đạt must-have (cần {need}); toàn pool có {in_pool}",
                                {"action": "search", "args": {"extra_terms": []}}))
    else:
        findings.append(Finding("must_pass_shortlist", False,
                                f"chỉ {have} ứng viên đạt must-have (cần {need}), tìm thêm không giúp được — "
                                f"báo cho HR thay vì hạ tiêu chí", unreachable=True))

    # 2. claims of the current shortlist went through verify_evidence
    if goals.get("require_verified_claims") and cfg["report"].get("claims", True) and cfg["criteria"]["enabled"]:
        sig = shortlist_signature(state, top_n)
        if state.get("verified_for") == sig:
            stats = state.get("verify_stats") or {}
            findings.append(Finding("claims_verified", True,
                                    f"đã verify shortlist hiện tại ({stats.get('supported', 0)} claim có bằng chứng, "
                                    f"{stats.get('no_evidence', 0)} gắn cờ No evidence)"))
        elif llm_usable:
            findings.append(Finding("claims_verified", False, "shortlist hiện tại chưa được sinh + verify claim",
                                    {"action": "verify_claims", "args": {}}))
        else:
            findings.append(Finding("claims_verified", False, "không verify được: LLM tắt hoặc đang fallback",
                                    unreachable=True))

    # 3. close calls inside the shortlist were looked at by the LLM judge
    if goals.get("resolve_near_ties"):
        ties = open_near_ties(state, cfg["judge"].get("max_rank") or top_n, cfg["agent"]["compare_max_delta"])
        if not ties:
            findings.append(Finding("near_ties_resolved", True, "không còn cặp sát điểm chưa so sánh"))
        elif (len(state.get("compared_pairs", [])) if compares_used is None else compares_used)                 >= cfg["agent"]["max_compares"]:
            findings.append(Finding("near_ties_resolved", False, f"còn {len(ties)} cặp sát điểm nhưng đã hết lượt so sánh",
                                    unreachable=True))
        elif not llm_usable:
            findings.append(Finding("near_ties_resolved", False, f"{len(ties)} cặp sát điểm nhưng LLM không dùng được",
                                    unreachable=True))
        else:
            a, b = ties[0]
            findings.append(Finding("near_ties_resolved", False, f"{len(ties)} cặp sát điểm chưa so sánh, ví dụ {a}/{b}",
                                    {"action": "compare", "args": {"a": a, "b": b}}))

    # 4. advice only: low-quality parses in the shortlist deserve a look (never blocks finishing)
    inspected = set(state.get("inspected", []))
    shaky = [c.cv_id for c in shortlist if state["cvs"][c.cv_id].parse_quality < LOW_PARSE_QUALITY
             and c.cv_id not in inspected]
    if shaky:
        findings.append(Finding("parse_quality", False, f"CV parse kém trong shortlist: {shaky}",
                                {"action": "inspect_cv", "args": {"cv_id": shaky[0]}}, hard=False))

    passed = all(f.ok or f.unreachable for f in findings if f.hard)
    return CriticReport(passed, tuple(findings))
