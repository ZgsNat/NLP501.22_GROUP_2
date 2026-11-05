from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge(left: dict | None, right: dict | None) -> dict:
    """Reducer: shallow-merge dict updates instead of replacing (lets nodes update one key of loops/assessments).
    A None value deletes its key (P7's screen drops assessments of CVs that left the shortlist)."""
    return {k: v for k, v in {**(left or {}), **(right or {})}.items() if v is not None}


def add_counts(left: dict | None, right: dict | None) -> dict:
    out = dict(left or {})
    for k, v in (right or {}).items():
        out[k] = out.get(k, 0) + v
    return out


class AgentState(TypedDict, total=False):
    # input
    jd_text: str
    company_profile: str
    models: dict[str, str]  # {"reasoning": "...", "fast": "..."} — persisted so resumes use the same brain
    lang: str  # "en" | "vi": the language of every LLM text HR reads (prompts.localized); quotes stay verbatim
    ablate: list[str]  # evaluation only: switch off parts ("clarify", "expand", "verify", "arbiter", "profile")

    # criteria
    requirements: dict
    needs_clarification: bool
    clarifications: Annotated[dict[str, str], merge]  # question -> HR answer
    weights: dict[str, float]
    policy_context: list[dict]

    # retrieval
    query_terms: list[str]
    substitutes: Annotated[dict[str, list[str]], merge]  # must-have -> accepted related skills (after expansion)
    declined: Annotated[dict[str, list[str]], merge]  # must-have -> related skills HR rejected (never re-proposed)
    fallbacks: Annotated[dict[str, list[str]], merge]  # must-have -> substitutes accepted by L2 (not stated in the JD)
    candidates: list[dict]
    knocked_out: dict[str, str]  # cv_id -> reason (H(c) = 0)
    screened: dict[str, str]  # cv_id -> why it was not assessed (not retrieved / not shortlisted)
    hr_rejected: Annotated[dict[str, str], merge]  # cv_id -> HR's reason (L5 / record_decisions): out of the ranking
    recovered: Annotated[dict[str, list[dict]], merge]  # cv_id -> facts the investigator verified (L6)
    investigated: Annotated[dict[str, str], merge]  # cv_id -> outcome of its investigation (never twice)

    # judgement
    assessments: Annotated[dict[str, dict], merge]
    retry_ids: list[str]
    ranking: list[dict]
    report: dict
    hr_action: str

    loops: Annotated[dict[str, int], add_counts]

    # P7 orchestrator only (agent/orchestrator.py)
    goal: str  # what HR asked for, in their words
    messages: Annotated[list[AnyMessage], add_messages]  # the orchestrator's conversation: HR, its tool calls, results
    calls: list[dict]  # tool calls the guard approved, waiting to run
    turn_start: int  # loops["agent"] when the current HR turn began (the step budget is per turn)
    turn_tokens: int  # loops["agent_tokens"] when the current HR turn began (so is the token budget)
    seen: dict[str, str]  # "tool:args:version" → observation, so a repeated call costs nothing
    shortlist: list[str]  # what the agent last presented to HR
    reply: str  # what the agent last said to HR
    responded: bool  # the agent answered HR this step: hand the turn to HR
    halt: str  # why the harness stopped the agent this turn (budget, repeated errors); code answers HR instead
    llm_order: list[str]  # rank_shortlist's controlled LLM order of the shortlist (agent/llm_rank.py); [] = rule order
    llm_reasons: dict[str, str]  # cv_id → the LLM's one-sentence reason for its place (shown to HR)
    goal_spec: dict  # agent/goal.py GoalSpec: HR's goal as constraints, each quoting HR's words
    goal_feasibility: dict  # what the pool can deliver against goal_spec (0 tokens)
    concepts: dict  # slug → {description, terms, evidence: cv_id → graded quote}: open-vocabulary criteria


def public_view(update: dict[str, Any]) -> dict[str, Any]:
    """What the UI gets: drop internal / bulky keys."""
    out = {k: v for k, v in update.items() if k not in {"jd_text", "models", "ablate", "messages", "calls", "seen"}}
    if "messages" in update:  # the agent's tool calls, as plain dicts the UI can show
        out["tool_calls"] = [{"name": c["name"], "args": c["args"]} for m in update["messages"]
                             for c in getattr(m, "tool_calls", None) or []]
    return out
