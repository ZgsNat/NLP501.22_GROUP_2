"""Shared state read/written by every stage (FR-24) + the runtime services nodes use."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

from cv_agent.criteria.clarify import Question
from cv_agent.llm.gateway import LLMGateway
from cv_agent.report.explain import ExplanationCard
from cv_agent.schema import CV, JD, Claim, CriteriaProfile, ScoredCandidate


class PipelineState(TypedDict, total=False):
    run_id: str
    # criteria
    jd: JD
    profile: CriteriaProfile
    questions: List[Question]
    hr_answers: Dict[str, int]
    clarify_rounds: int
    # candidate pool
    cvs: Dict[str, CV]
    candidate_pool: List[Tuple[str, float]]
    expansion: List[str]
    qe_rounds: int
    must_pass_count: int
    ranking: List[ScoredCandidate]
    knocked_out: List[Dict[str, Any]]
    judge_stats: Dict[str, Any]
    # evidence
    claims: Dict[str, List[Claim]]
    verify_stats: Dict[str, int]
    # budget / control
    budget: Dict[str, Any]
    fallback: bool
    trace: List[str]
    cards: List[ExplanationCard]
    # agent mode (cv_agent.agent): planner loop bookkeeping
    agent_step: int
    agent_log: List[Dict[str, Any]]
    pending: Dict[str, Any]
    searches: int
    ranking_stale: bool
    compared_pairs: List[List[str]]
    compares: int
    swapped: List[str]
    inspected: List[str]
    verified_for: str
    critic: Dict[str, Any]
    planner_stats: Dict[str, int]
    finished: bool


HRAnswerFn = Callable[[JD, List[Question]], Dict[str, int]]
ApproveFn = Callable[[JD, CriteriaProfile], bool]


@dataclass
class Runtime:
    """Non-serialisable services shared by nodes (kept out of the graph state)."""
    cfg: Dict[str, Any]
    gateway: LLMGateway
    encoder: Any
    reranker: Any
    nli: Any
    hr_answer_fn: Optional[HRAnswerFn] = None
    approve_fn: Optional[ApproveFn] = None
    retriever_cache: Dict[Any, Any] = field(default_factory=dict)
    memory: Any = None   # cv_agent.agent.memory.AgentMemory when agent.memory.enabled
