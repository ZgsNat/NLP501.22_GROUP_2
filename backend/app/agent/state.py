from typing import Annotated, Any, TypedDict


def merge(left: dict | None, right: dict | None) -> dict:
    """Reducer: shallow-merge dict updates instead of replacing (lets nodes update one key of loops/assessments)."""
    return {**(left or {}), **(right or {})}


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
    recovered: Annotated[dict[str, list[dict]], merge]  # cv_id -> facts the investigator verified (L6)
    investigated: Annotated[dict[str, str], merge]  # cv_id -> outcome of its investigation (never twice)

    # judgement
    assessments: Annotated[dict[str, dict], merge]
    retry_ids: list[str]
    ranking: list[dict]
    report: dict
    hr_action: str

    loops: Annotated[dict[str, int], add_counts]


def public_view(update: dict[str, Any]) -> dict[str, Any]:
    """What the UI gets: drop internal / bulky keys."""
    return {k: v for k, v in update.items() if k not in {"jd_text", "models", "ablate"}}
