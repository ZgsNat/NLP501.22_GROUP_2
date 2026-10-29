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

    # criteria
    requirements: dict
    needs_clarification: bool
    clarifications: Annotated[dict[str, str], merge]  # question -> HR answer
    weights: dict[str, float]
    policy_context: list[dict]

    # retrieval
    query_terms: list[str]
    substitutes: Annotated[dict[str, list[str]], merge]  # must-have -> accepted related skills (after expansion)
    hits: list[dict]  # raw hybrid-search results (internal)
    candidates: list[dict]
    knocked_out: dict[str, str]  # cv_id -> reason (H(c) = 0)

    # judgement
    assessments: Annotated[dict[str, dict], merge]
    retry_ids: list[str]
    ranking: list[dict]
    report: dict
    hr_action: str

    loops: Annotated[dict[str, int], add_counts]


def public_view(update: dict[str, Any]) -> dict[str, Any]:
    """What the UI gets: drop internal / bulky keys."""
    return {k: v for k, v in update.items() if k not in {"hits", "jd_text", "models"}}
