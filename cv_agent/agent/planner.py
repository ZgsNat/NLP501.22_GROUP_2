"""Planners pick the next tool. ``LLMPlanner`` lets the LLM decide; ``RulePlanner`` is the deterministic
policy (same order as the fixed P6 pipeline, then the critic's suggestions). The rules are also the safety
net: an invalid, unavailable or failed LLM decision is replaced by the rule decision and logged.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from cv_agent.agent.critic import open_near_ties, shortlist_signature
from cv_agent.agent.tools import TOOLS, available_tools, compares_used, searches_left
from cv_agent.llm.context import jd_digest
from cv_agent.orchestration.nodes import LLM_FAILURES
from cv_agent.orchestration.state import PipelineState, Runtime
from cv_agent.retrieval.hybrid import expansion_terms

AGENT_PROMPT_VERSION = "agent-v1"
HISTORY_STEPS = 5
OBS_CHARS = 220

AGENT_SYSTEM = (
    "You are the planning agent of a CV screening system for Automation Test roles. You decide the NEXT tool "
    "call only. Work towards the goal with as few LLM-costly steps (compare, verify_claims) as needed. Never "
    "guess personal attributes; you only see masked data and scores. Answer with JSON only."
)


@dataclass(frozen=True)
class Decision:
    action: str
    args: Dict[str, Any]
    reason: str
    source: str  # llm | rules | rules(fallback: ...) | budget


# ------------------------------------------------------------------ rule policy

def rule_decision(rt: Runtime, state: PipelineState) -> Decision:
    avail = available_tools(rt, state)
    cfg = rt.cfg

    def pick(action: str, reason: str, **args: Any) -> Decision:
        return Decision(action, args, reason, "rules")

    if "ask_hr" in avail:
        return pick("ask_hr", "JD còn điểm mơ hồ")
    if not state.get("candidate_pool"):
        if "search" in avail:
            return pick("search", "chưa có ứng viên")
        return Decision("finish", {}, "không tìm được ứng viên nào", "rules(forced)")
    qe = cfg["retrieval"]["query_expansion"]
    if "search" in avail and qe["enabled"] and state.get("must_pass_count", 0) < qe["min_must_pass"] \
            and [t for t in expansion_terms(state["jd"]) if t not in state.get("expansion", [])]:
        return pick("search", "ít ứng viên đạt must-have -> mở rộng truy vấn")
    if "rank" in avail:
        return pick("rank", "ranking chưa cập nhật")
    if "compare" in avail:  # same near-tie rule as the fixed pipeline first
        ties = open_near_ties(state, cfg["judge"].get("max_rank") or cfg["report"]["top_n"],
                              cfg["judge"]["near_tie_threshold"])
        if ties:
            return pick("compare", "cặp hoà điểm trong shortlist", a=ties[0][0], b=ties[0][1])
    if "verify_claims" in avail:
        return pick("verify_claims", "shortlist chưa verify")
    for f in (state.get("critic") or {}).get("findings", []):
        sug = f.get("suggestion")
        if not f["ok"] and f.get("hard", True) and sug and sug["action"] in avail:
            return pick(sug["action"], f"critic: {f['goal']}", **sug.get("args", {}))
    if "finish" in avail:
        return pick("finish", "đạt mục tiêu")
    for f in (state.get("critic") or {}).get("findings", []):  # soft advice once hard goals are done
        sug = f.get("suggestion")
        if not f["ok"] and sug and sug["action"] in avail:
            return pick(sug["action"], f"critic (lưu ý): {f['goal']}", **sug.get("args", {}))
    return Decision("finish", {}, "không còn hành động hợp lệ", "rules(forced)")


# ------------------------------------------------------------------ LLM planner

def state_summary(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    cfg, top_n = rt.cfg, rt.cfg["report"]["top_n"]
    ranking = state.get("ranking", [])
    top = []
    for c in ranking[:top_n + 1]:  # compact rows: [cv_id, score, passes must-haves]
        top.append([c.cv_id, round(c.score, 3), c.must_have_ok])
    b = rt.gateway.budget
    critic = state.get("critic") or {}
    return {
        "step": state.get("agent_step", 0), "max_steps": cfg["agent"]["max_steps"],
        "llm_calls_left": max(0, b.max_calls - b.calls),
        "compares_left": max(0, cfg["agent"]["max_compares"] - compares_used(state)),
        "hr_approved": bool(state.get("profile") and state["profile"].approved),
        "open_hr_questions": [q.text for q in state.get("questions", [])][:3],
        "searches_left": searches_left(rt, state) if "search" in available_tools(rt, state) else 0, "pool_size": len(state.get("candidate_pool", [])),
        "must_pass_in_pool": state.get("must_pass_count"), "query_expansion": state.get("expansion", []),
        "ranking_up_to_date": bool(ranking) and not state.get("ranking_stale", True),
        "shortlist": top,
        "open_near_ties": open_near_ties(state, cfg["judge"].get("max_rank") or top_n,
                                         cfg["agent"]["compare_max_delta"])[:5],
        "shortlist_claims_verified": state.get("verified_for") == shortlist_signature(state, top_n),
        "critic_passed": critic.get("passed"),
        "critic_open_goals": [f"{f['goal']}: {f['detail']}" for f in critic.get("findings", []) if not f["ok"]],
        "llm_fallback": bool(state.get("fallback")),
    }


def planner_prompt(rt: Runtime, state: PipelineState) -> str:
    avail = available_tools(rt, state)
    tools = "\n".join(f"- {t.name} {t.args_hint}: {t.description}" for t in avail.values())
    history = "\n".join(f"{h['step']}. {h['action']} {json.dumps(h['args'], ensure_ascii=False)} -> "
                        f"{h['observation'][:OBS_CHARS]}" for h in state.get("agent_log", [])[-HISTORY_STEPS:])
    goal = (f"GOAL: a ranked shortlist of the best {rt.cfg['report']['top_n']} candidates; at least "
            f"{rt.cfg['agent']['goals']['min_must_pass_shortlist']} pass every MUST requirement; shortlisted claims "
            f"verified; close calls in the shortlist judged. The critic re-checks the goal after every tool; 'finish' "
            f"is only possible once it passed. Unreachable goals do not block.")
    return (f"{goal}\n\nJOB:\n{jd_digest(state['jd'])}\n\nSTATE:\n"
            f"{json.dumps(state_summary(rt, state), ensure_ascii=False, default=str)}\n\n"
            f"HISTORY:\n{history or '(none)'}\n\nAVAILABLE TOOLS:\n{tools}\n\n"
            'Choose ONE tool. Return JSON only: {"action": "<tool name>", "args": {...}, '
            '"reason": "<max 15 words>"}')


def _action_name(raw: Any) -> str:
    """Small LLMs write 'finish {}' or 'compare(a, b)': keep the leading tool name."""
    name = str(raw or "").strip().lower()
    for sep in (" ", "(", "{", ":"):
        name = name.split(sep, 1)[0]
    return name


def _validated(data: Dict[str, Any], avail: Dict[str, Any]) -> Optional[str]:
    if not isinstance(data, dict):
        return "output is not an object"
    if data.get("action") not in avail:
        return f"action {data.get('action')!r} not available"
    if not isinstance(data.get("args", {}), dict):
        return "args is not an object"
    return None


def llm_decision(rt: Runtime, state: PipelineState) -> Decision:
    rule = rule_decision(rt, state)
    avail = available_tools(rt, state)
    try:
        data, _call_id = rt.gateway.call("planner", AGENT_PROMPT_VERSION, AGENT_SYSTEM, planner_prompt(rt, state),
                                         {"rule_choice": asdict(rule)}, meta={"step": state.get("agent_step", 0)})
    except LLM_FAILURES as exc:
        return Decision(rule.action, rule.args, rule.reason, f"rules(fallback: {type(exc).__name__})")
    if isinstance(data, dict):
        data = {**data, "action": _action_name(data.get("action"))}
    problem = _validated(data, avail)
    if problem:
        return Decision(rule.action, rule.args, rule.reason, f"rules(fallback: {problem})")
    return Decision(data["action"], dict(data.get("args") or {}), str(data.get("reason", ""))[:200], "llm")


def decide(rt: Runtime, state: PipelineState) -> Decision:
    acfg = rt.cfg["agent"]
    if state.get("agent_step", 0) >= acfg["max_steps"]:  # step budget spent: finish with what we have
        if not state.get("candidate_pool") and searches_left(rt, state) > 0:  # minimum: one search, one ranking
            return Decision("search", {}, "hết số bước nhưng chưa có ứng viên", "budget")
        if not state.get("candidate_pool"):
            return Decision("finish", {}, "hết số bước, không có ứng viên", "budget")
        if not state.get("ranking") or state.get("ranking_stale", True):
            return Decision("rank", {}, "hết số bước, xếp hạng lần cuối", "budget")
        return Decision("finish", {}, f"hết {acfg['max_steps']} bước", "budget")
    if acfg["planner"] == "llm" and not state.get("fallback"):
        return llm_decision(rt, state)
    return rule_decision(rt, state)


def tool_names() -> List[str]:
    return list(TOOLS)
