"""Tools the agent can call. Each tool has a guardrail (``unavailable``) and argument validation, so neither
the LLM planner nor a bug can skip the HR checkpoint, override the must-have filter, leave the shortlist
zone, or spend LLM budget after a fallback.

Every ``run`` returns ``(state update, observation)``; the observation is what the planner sees next.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from cv_agent.agent.critic import critique, open_near_ties, shortlist_signature
from cv_agent.arbiter.arbiter import _judge_pair
from cv_agent.ingest.taxonomy import normalize
from cv_agent.llm.context import cv_digest
from cv_agent.orchestration import nodes
from cv_agent.orchestration.nodes import LLM_FAILURES
from cv_agent.orchestration.state import PipelineState, Runtime
from cv_agent.retrieval.hybrid import expansion_terms
from cv_agent.verify.claims import generate_claims
from cv_agent.verify.evidence import verify_claims

log = logging.getLogger(__name__)
Update = Dict[str, Any]
INSPECT_WINDOW = 20      # inspect_cv only inside the reranked head
MAX_EXTRA_TERMS = 5


class ToolError(ValueError):
    """Invalid arguments for a tool (the planner falls back to the rules)."""


# ------------------------------------------------------------------ helpers

def searches_left(rt: Runtime, state: PipelineState) -> int:
    qe = rt.cfg["retrieval"]["query_expansion"]
    allowed = 1 + (qe["max_rounds"] if qe["enabled"] else 0)
    return max(0, allowed - state.get("searches", 0))


def llm_allowed(rt: Runtime, state: PipelineState) -> bool:
    return not state.get("fallback") and rt.cfg["judge"]["mode"] != "none"


def _ranked(state: PipelineState) -> bool:
    return bool(state.get("ranking")) and not state.get("ranking_stale", True)


def compares_used(state: PipelineState) -> int:
    """LLM pair judgements over the whole run (``compared_pairs`` is reset by a new search, this is not)."""
    return state.get("compares", len(state.get("compared_pairs", [])))


def must_pass_needed(rt: Runtime) -> int:
    """Pool size of must-have passers after which another search cannot help the shortlist goal."""
    qe = rt.cfg["retrieval"]["query_expansion"]
    need = min(rt.cfg["agent"]["goals"]["min_must_pass_shortlist"], rt.cfg["report"]["top_n"])
    return max(need, qe["min_must_pass"]) if qe["enabled"] else need


def _shortlist_ids(rt: Runtime, state: PipelineState) -> List[str]:
    return [c.cv_id for c in state.get("ranking", [])[: rt.cfg["report"]["top_n"]]]


# ------------------------------------------------------------------ ask_hr (Loop 1)

def ask_hr_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if state.get("profile") is not None and state["profile"].approved:
        return "HR đã duyệt tiêu chí"
    if not state.get("questions"):
        return "JD không còn điểm mơ hồ"
    if state.get("clarify_rounds", 0) >= rt.cfg["criteria"]["clarification"]["max_rounds"]:
        return "đã hỏi đủ số vòng"
    return None


def ask_hr(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    update = nodes.clarify_node(rt, state)
    # only answers from a real HR callback are remembered; simulated answers (CLI, tests) are not HR input
    if rt.memory is not None and rt.hr_answer_fn is not None:
        try:
            rt.memory.remember_answers(state["jd"], update["hr_answers"])
        except (OSError, ValueError) as exc:
            log.warning("agent memory not updated: %s", exc)
    return update, f"HR trả lời {update['hr_answers']}; còn {len(update['questions'])} câu hỏi"


# ------------------------------------------------------------------ search (Stage 3 + Loop 2)

def search_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if ask_hr_unavailable(rt, state) is None:  # Loop 1 first: never search with open JD ambiguities
        return "còn câu hỏi làm rõ JD chưa hỏi HR"
    if searches_left(rt, state) <= 0:
        return "đã dùng hết số lần tìm"
    if state.get("candidate_pool") and state.get("must_pass_count", 0) >= must_pass_needed(rt):
        return "pool đã đủ ứng viên đạt must-have"
    used = set(state.get("expansion", []))
    if state.get("candidate_pool") and not [t for t in expansion_terms(state["jd"]) if t not in used]:
        return "không còn thuật ngữ mở rộng mới"
    return None


def _clean_terms(raw: Any) -> Tuple[List[str], List[str]]:
    """Only taxonomy skills are accepted as extra query terms (no free text from the LLM reaches retrieval)."""
    if raw in (None, ""):
        return [], []
    if not isinstance(raw, list):
        raise ToolError("extra_terms must be a list of skill names")
    kept, dropped = [], []
    for term in raw[:MAX_EXTRA_TERMS * 2]:
        skill = normalize(str(term))
        (kept if skill and skill.name not in kept else dropped).append(skill.name if skill else str(term))
    return kept[:MAX_EXTRA_TERMS], dropped


def search(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    kept, dropped = _clean_terms(args.get("extra_terms"))  # validate before bothering HR
    update: Update = {}
    if not state["profile"].approved:  # HR checkpoint 1 (FR-10) always precedes the first search
        update = nodes.approve_node(rt, state)
    current = list(state.get("expansion", []))
    if not kept and state.get("searches", 0) > 0:  # a repeated search without terms: taxonomy expansion
        kept = [t for t in expansion_terms(state["jd"]) if t not in current][:MAX_EXTRA_TERMS]
    expansion = current + [t for t in kept if t not in current]
    if state.get("candidate_pool") and expansion == current:  # same query again would return the same pool
        raise ToolError(f"no new taxonomy terms; already used: {current}")
    merged = {**state, **update, "expansion": expansion}
    # a new pool means a new ranking: earlier pairwise verdicts no longer apply to it
    update = {**update, **nodes.retrieve_node(rt, merged), "expansion": expansion,
              "searches": state.get("searches", 0) + 1, "ranking_stale": True, "compared_pairs": [], "swapped": []}
    note = f"; bỏ thuật ngữ ngoài taxonomy {dropped}" if dropped else ""
    return update, (f"pool={len(update['candidate_pool'])}, đạt must-have={update['must_pass_count']}, "
                    f"mở rộng={expansion or '-'}{note}")


# ------------------------------------------------------------------ rank (Stage 4)

def rank_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if not state.get("candidate_pool"):
        return "chưa tìm kiếm"
    return None if state.get("ranking_stale", True) else "ranking đã cập nhật"


def rank_tool(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    update = {**nodes.rank_node(rt, state), "ranking_stale": False}
    top = update["ranking"][: rt.cfg["report"]["top_n"]]
    return update, ("top: " + ", ".join(f"{c.cv_id}={c.score:.3f}{'' if c.must_have_ok else '(KO)'}" for c in top)
                    + f"; knock-out={len(update['knocked_out'])}")


# ------------------------------------------------------------------ inspect_cv (read-only)

def inspect_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if not _ranked(state):
        return "chưa có ranking"
    if len(state.get("inspected", [])) >= rt.cfg["agent"]["max_inspects"]:
        return "đã xem đủ số CV cho phép"
    return None


def inspect_cv(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    cv_id = str(args.get("cv_id", ""))
    head = {c.cv_id: c for c in state["ranking"][:INSPECT_WINDOW]}
    if cv_id not in head:
        raise ToolError(f"cv_id {cv_id!r} is not in the top {INSPECT_WINDOW}")
    if cv_id in state.get("inspected", []):
        raise ToolError(f"{cv_id} was already inspected")
    c, cv = head[cv_id], state["cvs"][cv_id]
    contrib = sorted(c.contributions.items(), key=lambda kv: -kv[1])[:3]
    obs = (f"{cv_id}: score={c.score:.3f}, must-have={'đạt' if c.must_have_ok else c.knockout_reasons}, "
           f"đóng góp={[(k, round(v, 3)) for k, v in contrib]}, parse={cv.parse_quality}, flags={cv.flags}, "
           f"bằng chứng: {cv_digest(cv, 300)!r}")
    return {"inspected": list(state.get("inspected", [])) + [cv_id]}, obs


# ------------------------------------------------------------------ compare (Stage 5, one pair)

def compare_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if not _ranked(state):
        return "chưa có ranking"
    if not llm_allowed(rt, state):
        return "LLM judge tắt hoặc đang fallback"
    if compares_used(state) >= rt.cfg["agent"]["max_compares"]:
        return "đã so sánh đủ số cặp cho phép"  # each comparison = `passes` LLM calls
    if not open_near_ties(state, _max_rank(rt), rt.cfg["agent"]["compare_max_delta"]):
        return "không còn cặp sát điểm chưa so sánh"
    return None


def _max_rank(rt: Runtime) -> int:
    return rt.cfg["judge"].get("max_rank") or rt.cfg["report"]["top_n"]


def compare(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    a_id, b_id = str(args.get("a", "")), str(args.get("b", ""))
    allowed = open_near_ties(state, _max_rank(rt), rt.cfg["agent"]["compare_max_delta"])
    pair = next((p for p in allowed if set(p) == {a_id, b_id}), None)
    if pair is None:  # guardrail: adjacent, both pass must-haves, close score, shortlist zone, not yet compared
        raise ToolError(f"({a_id}, {b_id}) is not an open near-tie pair; allowed: {allowed[:5]}")
    ranking = list(state["ranking"])
    i = next(k for k, c in enumerate(ranking) if c.cv_id == pair[0])
    a, b = ranking[i], ranking[i + 1]
    profile = None if rt.cfg["criteria"].get("generic_prompt") else state["profile"]
    compared = list(state.get("compared_pairs", [])) + [list(pair)]
    stats = dict(state.get("judge_stats") or {"pairs": 0, "swaps": 0, "consistent_pairs": 0})
    passes = rt.cfg["judge"]["passes"] + rt.cfg["judge"]["passes"] % 2
    try:
        margin, _ids, valid = _judge_pair(rt.gateway, state["jd"], profile, a, b, state["cvs"], passes,
                                          rt.cfg["llm"].get("cv_chars", 700))
    except LLM_FAILURES as exc:
        return {"fallback": True, "compared_pairs": compared, "compares": compares_used(state) + 1}, f"LLM lỗi ({type(exc).__name__}) -> fallback"
    stats = {**stats, "pairs": stats["pairs"] + 1, "consistent_pairs": stats["consistent_pairs"]
             + int(valid == passes and abs(margin) == passes)}
    update: Update = {"compared_pairs": compared, "compares": compares_used(state) + 1, "judge_stats": stats, "budget": rt.gateway.budget.snapshot()}
    if margin < 0:
        note = f"Agent/Arbiter: xếp trên {a.cv_id} (Δ={a.score - b.score:+.3f}, thắng {-margin}/{passes} lượt)"
        ranking[i], ranking[i + 1] = replace(b, arbiter_notes=b.arbiter_notes + [note]), a
        # like the fixed arbiter: a swapped candidate moves at most one place (no drift via later pairs)
        update = {**update, "ranking": ranking, "judge_stats": {**stats, "swaps": stats["swaps"] + 1},
                  "swapped": list(state.get("swapped", [])) + [a.cv_id, b.cv_id]}
        return update, f"LLM chọn {b.cv_id} hơn {a.cv_id} ({-margin}/{passes} phiếu) -> đổi chỗ"
    ranking[i] = replace(a, arbiter_notes=a.arbiter_notes + [f"Agent/Arbiter giữ thứ tự trên {b.cv_id}"])
    return {**update, "ranking": ranking}, f"LLM giữ {a.cv_id} trên {b.cv_id} (margin {margin}/{passes})"


# ------------------------------------------------------------------ verify_claims (Stage 6 + Loop 4)

def verify_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if not _ranked(state):
        return "chưa có ranking"
    if not rt.cfg["report"].get("claims", True) or not rt.cfg["criteria"]["enabled"]:
        return "cấu hình không sinh claim"
    if not llm_allowed(rt, state):
        return "LLM tắt hoặc đang fallback"
    if state.get("verified_for") == shortlist_signature(state, rt.cfg["report"]["top_n"]):
        return "shortlist hiện tại đã verify"
    return None


def verify_tool(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    """Generate + verify claims only for shortlisted CVs that have none yet (re-ranking needs no regeneration)."""
    cvs, jd, top_n = state["cvs"], state["jd"], rt.cfg["report"]["top_n"]
    cv_chars = rt.cfg["llm"].get("claims_cv_chars", 1000)
    shortlist = state["ranking"][:top_n]
    by_id = {c.cv_id: c for c in shortlist}
    todo = [c for c in shortlist if c.cv_id not in state.get("claims", {})]
    fresh: Dict[str, list] = {}
    fallback = False
    for c in todo:
        try:
            fresh[c.cv_id] = generate_claims(rt.gateway, jd, cvs[c.cv_id], c, cv_chars=cv_chars)
        except LLM_FAILURES as exc:
            fallback = True
            log.warning("agent claims stopped (%s): %s", type(exc).__name__, exc)
            break
    vcfg = rt.cfg["verify"]
    stats = dict(state.get("verify_stats") or {})
    if vcfg["enabled"] and fresh:
        def regen(cv_id: str, failed, feedback, attempt):
            return generate_claims(rt.gateway, jd, cvs[cv_id], by_id[cv_id], failed, attempt, cv_chars)

        fresh, new_stats = verify_claims(fresh, {k: cvs[k].masked_text for k in fresh}, rt.nli,
                                         None if fallback else regen, vcfg["max_rounds"], vcfg["on_fail"],
                                         vcfg["fuzzy_threshold"])
        stats = {k: stats.get(k, 0) + new_stats.get(k, 0) for k in set(stats) | set(new_stats)}
    elif fresh:
        fresh = {k: [replace(cl, status="unverified") for cl in v] for k, v in fresh.items()}
    update: Update = {"claims": {**state.get("claims", {}), **fresh}, "verify_stats": stats, "fallback": fallback
                      or bool(state.get("fallback")), "budget": rt.gateway.budget.snapshot()}
    if not fallback:
        update["verified_for"] = shortlist_signature(state, top_n)
    return update, (f"sinh claim cho {len(fresh)} CV mới; verify: {stats}" + (" ; LLM lỗi -> fallback" if fallback else ""))


# ------------------------------------------------------------------ critic (runs after every tool) / finish

def run_critic(rt: Runtime, state: PipelineState) -> Update:
    """Cheap and deterministic, so it runs after every tool instead of costing the planner a step."""
    if not _ranked(state):
        return {"critic": None}
    can_search = searches_left(rt, state) if search_unavailable(rt, state) is None else 0
    return {"critic": critique(state, rt.cfg, can_search, llm_allowed(rt, state), compares_used(state)).to_dict()}


def finish_unavailable(rt: Runtime, state: PipelineState) -> Optional[str]:
    if not state.get("candidate_pool") and searches_left(rt, state) == 0:
        return None  # nothing was found and nothing is left to try: report the empty result
    if not _ranked(state):
        return "chưa có ranking"
    if state.get("fallback"):
        return None  # LLM unavailable: report what we have (FR-26)
    if (state.get("critic") or {}).get("passed"):
        return None
    return "critic chưa xác nhận đạt mục tiêu"


def finish(rt: Runtime, state: PipelineState, args: Dict[str, Any]) -> Tuple[Update, str]:
    return {"finished": True}, "kết thúc, lập báo cáo"


# ------------------------------------------------------------------ registry

class Tool:
    def __init__(self, name: str, description: str, args_hint: str,
                 unavailable: Callable[[Runtime, PipelineState], Optional[str]],
                 run: Callable[[Runtime, PipelineState, Dict[str, Any]], Tuple[Update, str]]) -> None:
        self.name, self.description, self.args_hint = name, description, args_hint
        self.unavailable, self.run = unavailable, run


TOOLS: Dict[str, Tool] = {t.name: t for t in (
    Tool("ask_hr", "ask HR the open clarification questions about the JD", "{}", ask_hr_unavailable, ask_hr),
    Tool("search", "retrieve candidates; optional extra skill terms widen the query (taxonomy skills only)",
         '{"extra_terms": ["skill", ...]}', search_unavailable, search),
    Tool("rank", "score the candidate pool against the Criteria Profile", "{}", rank_unavailable, rank_tool),
    Tool("inspect_cv", "read one CV's score breakdown and evidence lines (read-only)", '{"cv_id": "..."}',
         inspect_unavailable, inspect_cv),
    Tool("compare", "LLM judges an adjacent near-tie pair in the shortlist zone, both orders; may swap them",
         '{"a": "cv_id", "b": "cv_id"}', compare_unavailable, compare),
    Tool("verify_claims", "generate evidence claims for the shortlist and verify them against the CV text", "{}",
         verify_unavailable, verify_tool),
    Tool("finish", "stop and write the report (only when the critic passed)", "{}",
         finish_unavailable, finish),
)}


def available_tools(rt: Runtime, state: PipelineState) -> Dict[str, Tool]:
    return {name: t for name, t in TOOLS.items() if t.unavailable(rt, state) is None}
