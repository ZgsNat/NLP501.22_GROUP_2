"""Versioned prompts. Bump the version whenever wording changes (logged with each call, FR-31).

v1.1: inputs are compact digests (``cv_agent.llm.context``) and outputs are short
JSON objects — on CPU, generated tokens dominate latency.
v1.2: claims limited to at most 4 short ones about MUST requirements.
v1.3: claims cite numbered CV lines; repair rounds only re-ask the rejected claims.
"""
from __future__ import annotations

import json
from typing import Any, Dict

PROMPT_VERSION = "v1.3"

SYSTEM = (
    "You are a senior QA Lead helping HR shortlist Automation Test candidates. "
    "Judge ONLY from the masked CV text provided. Never guess personal attributes. "
    "Every claim MUST be backed by evidence cited from the CV. Answer with JSON only."
)


def _criteria_block(criteria: Dict[str, Any]) -> str:
    return json.dumps(criteria, ensure_ascii=False, separators=(",", ":"))


def pairwise(jd_text: str, criteria: Dict[str, Any], cv_a: str, cv_b: str) -> str:
    crit = f"CRITERIA WEIGHTS: {_criteria_block(criteria)}\n\n" if criteria else ""
    return (
        f"JOB:\n{jd_text}\n\n{crit}CANDIDATE A:\n{cv_a}\n\nCANDIDATE B:\n{cv_b}\n\n"
        "Which candidate fits the job better? Return JSON only: "
        '{"winner": "A" or "B", "span": "<verbatim quote from the winner, max 15 words>"}'
    )


def listwise(jd_text: str, criteria: Dict[str, Any], candidates: Dict[str, str]) -> str:
    blocks = "\n\n".join(f"[{cid}]\n{text}" for cid, text in candidates.items())
    crit = f"CRITERIA WEIGHTS: {_criteria_block(criteria)}\n\n" if criteria else ""
    return (
        f"JOB:\n{jd_text}\n\n{crit}CANDIDATES:\n{blocks}\n\n"
        'Rank ALL candidates from best to worst fit. Return JSON only: {"ranking": ["<id>", ...]}'
    )


def claims(jd_text: str, numbered_cv: str, rejected: str = "") -> str:
    """Claims cite evidence by CV line number; the cited line becomes the span (verbatim by construction)."""
    out_format = ('Return JSON only: {"claims": [{"axis": "<axis>", "claim": "<max 8 words>", '
                  '"line": <number of the CV line that states it>}]}')
    if rejected:  # Loop 4: only repair the rejected claims, do not regenerate the whole set
        return (
            f"JOB:\n{jd_text}\n\nCV LINES:\n{numbered_cv}\n\nTHESE CLAIMS WERE REJECTED (cited line does not state "
            f"them):\n{rejected}\n\nFor each rejected claim, either cite the CV line that literally states it, or drop "
            f"it. Return [] if none can be supported. {out_format}"
        )
    return (
        f"JOB:\n{jd_text}\n\nCV LINES:\n{numbered_cv}\n\n"
        "List up to 4 MUST requirements of the job that this CV clearly satisfies. Skip any requirement the CV "
        f"does not mention — never write a claim about missing skills. {out_format}"
    )
