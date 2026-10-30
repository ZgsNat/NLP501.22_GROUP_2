"""The StateGraph. Read this file top to bottom to understand the agent: nodes are the work, the routers below are
the *agentic* part — every cycle is a decision point with a hard limit from Settings (the harness)."""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.state import AgentState
from app.config import get_settings


def after_ingest(state: AgentState) -> str:
    return "clarify" if state.get("needs_clarification") else "search"  # L1


def after_search(state: AgentState) -> str:
    s = get_settings()
    covered = sum(c["cv_id"] not in state["knocked_out"] for c in state["candidates"])
    scarce = covered < s.min_coverage and state.get("loops", {}).get("expand", 0) < s.max_expansions \
        and "expand" not in state.get("ablate", [])
    return "expand_query" if scarce else "rerank"  # L2


def after_verify(state: AgentState) -> str:
    return "assess" if state.get("retry_ids") else "score"  # L4 (retry budget enforced inside verify_evidence)


def after_review(state: AgentState) -> str:
    return "score" if state.get("hr_action") == "adjust" else END  # L5


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(AgentState)
    for name in ("ingest", "clarify", "search", "investigate", "expand_query", "rerank", "assess", "verify_evidence",
                 "score", "arbitrate", "report", "hr_review"):
        g.add_node(name, getattr(nodes, name))

    g.add_edge(START, "ingest")
    g.add_conditional_edges("ingest", after_ingest, ["clarify", "search"])
    g.add_edge("clarify", "ingest")
    g.add_edge("search", "investigate")  # L6: tool-using agent, only when a knock-out looks like a parse failure
    g.add_conditional_edges("investigate", after_search, ["expand_query", "rerank"])
    g.add_edge("expand_query", "search")
    g.add_edge("rerank", "assess")
    g.add_edge("assess", "verify_evidence")
    g.add_conditional_edges("verify_evidence", after_verify, ["assess", "score"])
    g.add_edge("score", "arbitrate")  # L3 lives inside arbitrate: LLM only on near-ties
    g.add_edge("arbitrate", "report")
    g.add_edge("report", "hr_review")
    g.add_conditional_edges("hr_review", after_review, ["score", END])
    return g.compile(checkpointer=checkpointer)
