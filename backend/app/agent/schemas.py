"""Structured outputs the LLM must return. Field descriptions are part of the prompt — keep them precise."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

CRITERIA = (
    "tools",
    "automation_depth",
    "programming_language",
    "test_types",
    "cicd_infra",
    "experience_level",
    "istqb",
    "domain_language",
)


class JDRequirements(BaseModel):
    title: str
    must_have: list[str] = Field(
        description="Tools/frameworks/languages explicitly REQUIRED, lowercase canonical names (e.g. 'selenium', 'java'). "
        "Empty if the JD names none."
    )
    nice_to_have: list[str] = Field(description="Tools/skills described as a plus or preferred, lowercase.")
    min_years: int | None = Field(description="Minimum years of experience if stated as a number, else null.")
    depth_level: int | None = Field(
        description="Required automation depth 1-4 (1 run scripts, 2 write scripts, 3 build framework, 4 test architecture), "
        "null if unclear."
    )
    languages: list[str] = Field(description="Programming languages required or preferred, lowercase.")
    test_types: list[str] = Field(description="Subset of: web, mobile, api, performance.")
    ambiguities: list[str] = Field(
        description="Concrete questions for HR about requirements too vague to rank fairly (e.g. 'How many years is "
        "several?'). Empty list if the JD is specific enough. Never ask about anything already answered by HR."
    )
    # Parsed from the JD text by app.nlp.jd, never asked of the LLM (kept out of the JSON schema it sees).
    foreign_languages: SkipJsonSchema[list[str]] = []
    domains: SkipJsonSchema[list[str]] = []


class CriterionScore(BaseModel):
    score: int = Field(ge=0, le=100, description="0-100 for this criterion against the JD")
    rationale: str = Field(description="One short sentence, at most 25 words")
    evidence: list[str] = Field(
        description="1-3 short quotes copied VERBATIM from the CV that justify the score. Empty if there is no evidence."
    )


RULE_CRITERIA = ("tools", "istqb", "experience_level")  # scored deterministically in scoring.py from the CV profile


class DepthFact(BaseModel):
    level: int = Field(ge=1, le=4, description="Highest automation depth PROVEN by the CV: 1 runs/maintains existing "
                       "scripts, 2 writes new scripts, 3 designs/builds a framework, 4 test architecture/platform for teams")
    rationale: str = Field(description="One short sentence, at most 25 words")
    evidence: list[str] = Field(description="1-3 short quotes copied VERBATIM from the CV proving that level")


class CandidateAssessment(BaseModel):
    """Facts where facts exist (depth → scored by code; years are parsed from dates, not asked); calibrated
    judgement only where they don't."""

    automation_depth: DepthFact
    programming_language: CriterionScore
    test_types: CriterionScore
    cicd_infra: CriterionScore
    domain_language: CriterionScore


class CandidateExplanation(BaseModel):
    cv_id: str
    explanation: str = Field(description="2-3 sentences: why this rank, citing the verified evidence")
    strengths: list[str]
    gaps: list[str] = Field(description="Gaps against the JD the HR should probe in interview")


class RankingReport(BaseModel):
    summary: str = Field(description="3-4 sentences for HR: overall pool quality, key trade-offs, recommended next step")
    candidates: list[CandidateExplanation]


class WeightChange(BaseModel):
    criterion: Literal[CRITERIA]  # closed set: the LLM can only point at criteria that exist
    direction: Literal["more", "less", "ignore"]


class WeightInstruction(BaseModel):
    """HR's plain-language re-weighting ("certification matters more for this client") as typed changes."""

    changes: list[WeightChange] = Field(description="One entry per criterion the instruction is about; empty if none")
