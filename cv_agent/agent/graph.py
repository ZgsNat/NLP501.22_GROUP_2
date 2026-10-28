"""Agent graph: criteria (+ memory) -> [plan -> act]* -> report.

The planner chooses one tool per step; ``act`` runs it behind its guardrail. An invalid tool call is replaced
by the rule decision, so a weak or misbehaving LLM can slow the agent down but never break the run.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, replace
from functools import partial
from typing import Any, Dict

from cv_agent.agent.planner import Decision, decide, rule_decision
from cv_agent.agent.tools import TOOLS, ToolError, run_critic
from cv_agent.criteria.clarify import apply_answers
from cv_agent.orchestration import nodes
from cv_agent.orchestration.state import PipelineState, Runtime

log = logging.getLogger(__name__)


def agent_criteria_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    """Stage 2 plus long-term memory: reuse HR answers for this JD and HR-calibrated weights."""
    update = nodes.criteria_node(rt, state)
    notes = []
    if rt.memory is not None:
        try:
            known = rt.memory.recall_answers(state["jd"])
        except ValueError as exc:  # optional feature: a broken memory file must not stop the search
            known = {}
            notes.append(f"bỏ qua bộ nhớ hỏng ({exc})")
        # stale entries (an option index the question no longer has) are ignored
        reuse = {q.qid: known[q.qid] for q in update["questions"]
                 if q.qid in known and 0 <= known[q.qid] < len(q.options)}
        if reuse:
            update = {**update, "jd": apply_answers(state["jd"], update["questions"], reuse),
                      "hr_answers": reuse, "questions": [q for q in update["questions"] if q.qid not in reuse]}
            notes.append(f"dùng lại {len(reuse)} câu trả lời HR đã lưu cho JD này")
        if rt.cfg["agent"]["memory"].get("apply_weights"):
            try:
                weights = rt.memory.recall_weights(update["profile"].name)
            except ValueError:
                weights = None
            if weights:
                update = {**update, "profile": replace(update["profile"], weights=weights)}
                notes.append(f"áp dụng trọng số HR đã hiệu chỉnh cho profile '{update['profile'].name}'")
    msg = "memory: " + "; ".join(notes) if notes else "memory: không có gì để dùng lại"
    return {**update, "agent_step": 0, "agent_log": [], "ranking_stale": True,
            "planner_stats": {}, "trace": update["trace"] + [msg]}


def plan_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    step = state.get("agent_step", 0) + 1
    decision = decide(rt, {**state, "agent_step": step - 1})
    return {"agent_step": step, "pending": asdict(decision)}


def _run(rt: Runtime, state: PipelineState, decision: Decision):
    tool = TOOLS[decision.action]
    if decision.source not in ("budget", "rules(forced)"):  # guardrail re-checked at execution time
        why = tool.unavailable(rt, state)
        if why:
            raise ToolError(f"{decision.action} unavailable: {why}")
    return tool.run(rt, state, decision.args)


def act_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    decision = Decision(**state["pending"])
    try:
        update, obs = _run(rt, state, decision)
    except ToolError as exc:  # invalid LLM tool call -> rule decision instead
        rule = rule_decision(rt, state)
        decision = Decision(rule.action, rule.args, rule.reason, f"rules(fallback: {exc})")
        try:
            update, obs = _run(rt, state, decision)
        except ToolError as exc2:  # rules disagree with a guardrail (a bug): stop safely with what we have
            log.error("agent: rule decision %s rejected (%s); finishing", rule.action, exc2)
            decision = Decision("finish", {}, f"dừng an toàn: {exc2}", "rules(forced)")
            update, obs = _run(rt, state, decision)
    update = {**update, **run_critic(rt, {**state, **update})}
    stats = dict(state.get("planner_stats", {}))
    # llm | rules | budget | fallback (an LLM decision that had to be replaced by the rules)
    key = "fallback" if decision.source.startswith("rules(fallback") else decision.source.split("(")[0]
    stats[key] = stats.get(key, 0) + 1
    entry = {"step": state["agent_step"], "action": decision.action, "args": decision.args,
             "reason": decision.reason, "source": decision.source, "observation": obs}
    msg = f"agent bước {entry['step']} [{decision.source}] {decision.action} {decision.args or ''} – {decision.reason} → {obs}"
    log.info(msg)
    return {**update, "agent_log": list(state.get("agent_log", [])) + [entry], "planner_stats": stats,
            "trace": list(update.get("trace", state.get("trace", []))) + [msg]}


def after_act(state: PipelineState) -> str:
    return "report" if state.get("finished") else "plan"


def add_agent_nodes(g, rt: Runtime, end: str) -> None:
    g.add_node("criteria", partial(agent_criteria_node, rt))
    g.add_node("plan", partial(plan_node, rt))
    g.add_node("act", partial(act_node, rt))
    g.add_node("report", partial(nodes.report_node, rt))
    g.set_entry_point("criteria")
    g.add_edge("criteria", "plan")
    g.add_edge("plan", "act")
    g.add_conditional_edges("act", after_act, {"plan": "plan", "report": "report"})
    g.add_edge("report", end)
