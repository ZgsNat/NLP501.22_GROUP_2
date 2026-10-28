"""LLM claim generation: every claim must carry a verbatim CV span (FR-21).

The CV digest is shown with numbered lines and the LLM cites a line number; the
cited line becomes the claim's span, so "span not in CV" cannot come from a
paraphrase — verify_evidence then only has to check entailment (NLI).
"""
from __future__ import annotations

import logging
from typing import List, Sequence

from cv_agent.llm import prompts
from cv_agent.llm.context import cv_digest_lines, jd_digest, numbered
from cv_agent.llm.gateway import LLMGateway
from cv_agent.ranking.scoring import requirement_match
from cv_agent.schema import CV, JD, Claim, ScoredCandidate

log = logging.getLogger(__name__)
DEFAULT_CV_CHARS = 1000


def _context(jd: JD, cv: CV, scored: ScoredCandidate, attempt: int) -> dict:
    """Structured hints for the heuristic backend (ignored by real LLMs)."""
    facts, missing = [], []
    for req in jd.requirements:
        sat, spans = requirement_match(cv, req)
        if sat >= 1.0 and spans:
            facts.append({"axis": req.axis, "claim": f"Đáp ứng yêu cầu {req.axis}: {spans[0]}", "span": spans[0]})
        elif req.values:
            missing.append({"axis": req.axis, "value": req.values[0]})
    return {"facts": facts, "missing": missing, "attempt": attempt, "cv_id": cv.cv_id}


def _span_for(item: dict, lines: List[str]) -> str:
    try:
        idx = int(item.get("line"))
    except (TypeError, ValueError):
        idx = 0
    if 1 <= idx <= len(lines):
        return lines[idx - 1]
    return str(item.get("span", "")).strip()  # no valid line: fall back to a quoted span, if any


def generate_claims(gw: LLMGateway, jd: JD, cv: CV, scored: ScoredCandidate, rejected: Sequence[Claim] = (),
                    attempt: int = 0, cv_chars: int = DEFAULT_CV_CHARS) -> List[Claim]:
    lines = cv_digest_lines(cv, cv_chars) or [cv.masked_text[:cv_chars]]
    rejected_txt = "\n".join(f"- {c.text} (cited: {c.span_text[:80]!r})" for c in rejected)
    user = prompts.claims(jd_digest(jd), numbered(lines), rejected_txt)
    data, call_id = gw.call("claims", prompts.PROMPT_VERSION, prompts.SYSTEM, user,
                            _context(jd, cv, scored, attempt), meta={"cv_id": cv.cv_id, "attempt": attempt})
    out: List[Claim] = []
    for item in data.get("claims", []) or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("claim", "")).strip()
        if text:
            out.append(Claim(cv_id=cv.cv_id, text=text, span_text=_span_for(item, lines),
                             axis=str(item.get("axis", "")), attempts=attempt, llm_call_id=call_id))
    return out
