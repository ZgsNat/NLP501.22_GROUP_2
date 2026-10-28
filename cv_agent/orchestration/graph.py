"""StateGraph wiring (FR-24): 7 stages, conditional loops for clarification
(Loop 1) and query expansion (Loop 2); the verify loop (Loop 4) runs inside
``evidence_node``. Uses LangGraph when installed, else a minimal compatible runner.
"""
from __future__ import annotations

import logging
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cv_agent.llm.backends import build_backend
from cv_agent.llm.gateway import Budget, LLMGateway
from cv_agent.orchestration import nodes
from cv_agent.orchestration.state import PipelineState, Runtime
from cv_agent.ranking.reranker import build_reranker
from cv_agent.retrieval.dense import build_encoder
from cv_agent.schema import CV, JD
from cv_agent.verify.nli import build_nli

log = logging.getLogger(__name__)
END = "__end__"
MAX_STEPS = 50


class SimpleGraph:
    """Tiny subset of LangGraph's StateGraph API used when langgraph is not installed."""

    def __init__(self, _state_type: Any) -> None:
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, str] = {}
        self.cond: Dict[str, tuple] = {}
        self.entry: Optional[str] = None

    def add_node(self, name: str, fn: Callable) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, fn: Callable, mapping: Dict[str, str]) -> None:
        self.cond[src] = (fn, mapping)

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def compile(self) -> "SimpleGraph":
        return self

    def invoke(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        current, steps = self.entry, 0
        limit = (config or {}).get("recursion_limit", MAX_STEPS)
        state = dict(state)
        while current and current != END:
            steps += 1
            if steps > limit:
                raise RuntimeError("Graph exceeded max steps (possible infinite loop)")
            state.update(self.nodes[current](state) or {})
            if current in self.cond:
                fn, mapping = self.cond[current]
                current = mapping[fn(state)]
            else:
                current = self.edges.get(current, END)
        return state


def _graph_cls():
    try:
        from langgraph.graph import StateGraph
        return StateGraph, "langgraph"
    except ImportError:
        return SimpleGraph, "simple"


def build_graph(rt: Runtime, prefer_langgraph: bool = True):
    cls, kind = _graph_cls() if prefer_langgraph else (SimpleGraph, "simple")
    g = cls(PipelineState)
    if rt.cfg.get("agent", {}).get("enabled"):  # LLM planner chooses the tools (cv_agent.agent)
        from cv_agent.agent.graph import add_agent_nodes
        add_agent_nodes(g, rt, END)
        return g.compile(), f"{kind}+agent[{rt.cfg['agent']['planner']}]"
    g.add_node("criteria", partial(nodes.criteria_node, rt))
    g.add_node("clarify", partial(nodes.clarify_node, rt))
    g.add_node("approve", partial(nodes.approve_node, rt))
    g.add_node("retrieve", partial(nodes.retrieve_node, rt))
    g.add_node("expand", partial(nodes.expand_node, rt))
    g.add_node("rank", partial(nodes.rank_node, rt))
    g.add_node("judge", partial(nodes.judge_node, rt))
    g.add_node("evidence", partial(nodes.evidence_node, rt))
    g.add_node("report", partial(nodes.report_node, rt))
    g.set_entry_point("criteria")
    g.add_conditional_edges("criteria", partial(nodes.need_clarification, rt), {"clarify": "clarify", "approve": "approve"})
    g.add_conditional_edges("clarify", partial(nodes.need_clarification, rt), {"clarify": "clarify", "approve": "approve"})
    g.add_edge("approve", "retrieve")
    g.add_conditional_edges("retrieve", partial(nodes.need_expansion, rt), {"expand": "expand", "rank": "rank"})
    g.add_edge("expand", "retrieve")
    g.add_edge("rank", "judge")
    g.add_edge("judge", "evidence")
    g.add_edge("evidence", "report")
    g.add_edge("report", END)
    log.debug("graph backend: %s", kind)
    return g.compile(), kind


def build_runtime(cfg: Dict[str, Any], run_dir: Optional[Path] = None, hr_answer_fn=None, approve_fn=None,
                  shared: Optional[Runtime] = None) -> Runtime:
    """Create services. ``shared`` lets experiment loops reuse encoder/reranker/NLI/indices across JDs."""
    run_id = uuid.uuid4().hex[:8]
    b = cfg["budget"]
    gateway = LLMGateway(build_backend(cfg["llm"]), log_path=(run_dir / "llm_calls.jsonl") if run_dir else None,
                         cache_dir=Path(cfg["paths"]["cache_dir"]) if cfg["paths"].get("cache_dir") else None,
                         budget=Budget(b["max_tokens"], b["max_calls"], b["timeout_s"]), run_id=run_id)
    memory = _memory(cfg)
    if shared is not None:
        return Runtime(cfg, gateway, shared.encoder, shared.reranker, shared.nli, hr_answer_fn, approve_fn,
                       shared.retriever_cache, memory)
    return Runtime(cfg, gateway, build_encoder(cfg["retrieval"]["dense"]), build_reranker(cfg["rerank"]),
                   build_nli(cfg["verify"]), hr_answer_fn, approve_fn, memory=memory)


def _memory(cfg: Dict[str, Any]):
    agent = cfg.get("agent", {})
    if not (agent.get("enabled") and agent.get("memory", {}).get("enabled")):
        return None
    from cv_agent.agent.memory import AgentMemory
    from cv_agent.config import ROOT
    path = Path(agent["memory"]["path"])
    return AgentMemory(path if path.is_absolute() else ROOT / path)


def recursion_limit(cfg: Dict[str, Any]) -> int:
    """Fixed graph: MAX_STEPS. Agent: plan + act per step, plus criteria/report and slack."""
    agent = cfg.get("agent", {})
    return max(MAX_STEPS, 2 * agent.get("max_steps", 0) + 10) if agent.get("enabled") else MAX_STEPS


def run_pipeline(rt: Runtime, jd: JD, cvs: List[CV], profile=None) -> PipelineState:
    graph, kind = build_graph(rt)
    init: PipelineState = {"run_id": rt.gateway.run_id, "jd": jd, "cvs": {c.cv_id: c for c in cvs},
                           "trace": [f"graph={kind}, experiment={rt.cfg.get('experiment')}, llm={rt.gateway.backend.name}"]}
    if profile is not None:
        init["profile"] = profile
    started = time.perf_counter()
    out = graph.invoke(init, {"recursion_limit": recursion_limit(rt.cfg)})
    out = dict(out)
    out["latency_s"] = round(time.perf_counter() - started, 3)
    out["llm_stats"] = rt.gateway.stats()
    return out
