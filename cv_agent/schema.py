"""Canonical CV / JD / Criteria schema shared by every stage.

Every module reads and writes these dataclasses only, so stages can be
developed and tested in isolation (see requirement: "chốt canonical schema tuần 1").
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# The 8 scoring axes ("8 trục") used by the Criteria Profile and the scorer.
AXES: tuple = (
    "tools",          # automation tools / frameworks (Selenium, Playwright, Appium...)
    "depth",          # automation depth 1-4 (rubric)
    "languages",      # programming languages (Java, Python, JS/TS, C#...)
    "test_types",     # API, UI, mobile, performance, security...
    "cicd",           # CI/CD & DevOps (Jenkins, GitLab CI, Docker...)
    "experience",     # years of experience
    "certification",  # ISTQB & similar
    "foreign_lang",   # English / Japanese level
)

ENTITY_TYPES: tuple = ("tool", "language", "test_type", "cicd", "years", "certification", "foreign_lang")


@dataclass(frozen=True)
class Span:
    """A character span inside the (masked) CV text."""
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Entity:
    type: str            # one of ENTITY_TYPES
    value: str           # normalised value, e.g. "selenium"
    span: Span
    esco_uri: Optional[str] = None


@dataclass(frozen=True)
class DepthAssessment:
    level: int           # 1..4
    evidence: List[Span] = field(default_factory=list)
    rationale: str = ""


@dataclass
class CV:
    cv_id: str
    raw_text: str                       # original text; never leaves ingest stage
    masked_text: str                    # PII-masked text used by all downstream stages
    entities: List[Entity] = field(default_factory=list)
    depth: Optional[DepthAssessment] = None
    years_experience: float = 0.0
    language: str = "en"                # detected language of CV
    parse_quality: float = 1.0          # 0..1; low => flagged, never dropped (FR-05)
    flags: List[str] = field(default_factory=list)
    pii_found: Dict[str, int] = field(default_factory=dict)
    source_path: Optional[str] = None

    def values(self, entity_type: str) -> List[str]:
        return sorted({e.value for e in self.entities if e.type == entity_type})

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialisable view WITHOUT raw text (privacy, NFR-02)."""
        d = asdict(self)
        d.pop("raw_text", None)
        return d


@dataclass(frozen=True)
class Requirement:
    axis: str                 # one of AXES
    values: tuple             # accepted values (any-of), e.g. ("selenium", "playwright")
    min_level: float = 0.0    # for numeric axes: min years / min depth
    must: bool = False        # True => part of H(c) knock-out
    text: str = ""            # original phrase in JD


@dataclass
class JD:
    jd_id: str
    title: str
    text: str
    company_type: str = "default"
    requirements: List[Requirement] = field(default_factory=list)

    @property
    def must_haves(self) -> List[Requirement]:
        return [r for r in self.requirements if r.must]

    @property
    def nice_to_haves(self) -> List[Requirement]:
        return [r for r in self.requirements if not r.must]


@dataclass
class CriteriaProfile:
    name: str
    weights: Dict[str, float]           # axis -> w(c,k); must sum to 1
    description: str = ""
    approved: bool = False              # HR checkpoint 1 (FR-10)

    def validate(self) -> None:
        unknown = set(self.weights) - set(AXES)
        if unknown:
            raise ValueError(f"Unknown axes in profile {self.name}: {sorted(unknown)}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Profile {self.name}: weights sum to {total:.4f}, expected 1.0")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError(f"Profile {self.name}: negative weight")


@dataclass
class Claim:
    """A statement produced by the LLM (or rule) about a candidate, with its evidence span."""
    cv_id: str
    text: str
    span_text: str
    axis: str = ""
    verified: Optional[bool] = None
    status: str = "pending"    # pending | supported | no_evidence | removed
    attempts: int = 0
    llm_call_id: Optional[str] = None


@dataclass
class ScoredCandidate:
    cv_id: str
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    axis_scores: Dict[str, float] = field(default_factory=dict)      # S(c,k)
    contributions: Dict[str, float] = field(default_factory=dict)    # w*S per axis (L2)
    must_have_ok: bool = True
    knockout_reasons: List[str] = field(default_factory=list)
    score: float = 0.0                 # 1[H] * sum(w*S); equals sum(contributions)
    potential_score: float = 0.0       # sum(w*S) ignoring knock-out (for counterfactual / HR review)
    evidence: Dict[str, List[str]] = field(default_factory=dict)     # axis -> CV spans used for S(c,k)
    arbiter_notes: List[str] = field(default_factory=list)
