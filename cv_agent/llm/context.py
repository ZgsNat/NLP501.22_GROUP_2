"""Compact LLM context (NFR-04 latency on CPU laptops).

``cv_digest`` keeps only the masked-CV lines that carry evidence (skill entities,
years, automation-depth cues), verbatim and in their original order, so spans
the LLM quotes from the digest still verify against the full CV.
``jd_digest`` reduces a JD to its title and parsed must-have / nice-to-have list.
"""
from __future__ import annotations

import bisect
from typing import List

from cv_agent.ingest.extractor import _RANGE_RE
from cv_agent.schema import CV, JD

_PLACEHOLDER_ONLY = {"[NAME]", "[EMAIL]", "[PHONE]", "[ADDRESS]", "[DATE]", "[DOB]", "[GENDER]", "[AGE]", "[PHOTO]"}


def _evidence_offsets(cv: CV) -> List[int]:
    offsets = [e.span.start for e in cv.entities]
    offsets += [m.start() for m in _RANGE_RE.finditer(cv.masked_text)]  # employment periods
    if cv.depth:
        offsets += [s.start for s in cv.depth.evidence]
    return sorted(offsets)


def cv_digest(cv: CV, max_chars: int = 700) -> str:
    """Evidence lines of the masked CV, original order, capped at ``max_chars``."""
    lines = cv_digest_lines(cv, max_chars)
    return "\n".join(lines) if lines else cv.masked_text[:max_chars]


def numbered(lines: List[str]) -> str:
    """``[1] first line`` … so the LLM can cite evidence by line number."""
    return "\n".join(f"[{i}] {line}" for i, line in enumerate(lines, start=1))


def cv_digest_lines(cv: CV, max_chars: int = 700) -> List[str]:
    text = cv.masked_text
    lines = text.splitlines(keepends=True)
    starts, pos = [], 0
    for line in lines:
        starts.append(pos)
        pos += len(line)
    keep = {bisect.bisect_right(starts, off) - 1 for off in _evidence_offsets(cv)}
    out, used = [], 0
    for idx in sorted(keep):
        line = lines[idx].strip()
        if not line or line in _PLACEHOLDER_ONLY or line.split(":")[-1].strip() in _PLACEHOLDER_ONLY:
            continue
        if used + len(line) + 1 > max_chars:
            break
        out.append(line)
        used += len(line) + 1
    return out


def jd_digest(jd: JD, max_chars: int = 600) -> str:
    def fmt(r) -> str:
        if r.values:
            return f"{r.axis}: {' / '.join(r.values)}"
        return f"{r.axis} >= {r.min_level:g}"

    if not jd.requirements:  # nothing parsed: the raw text is the only signal
        return jd.text[:max_chars]
    parts = [jd.title]
    if jd.must_haves:
        parts.append("MUST: " + "; ".join(fmt(r) for r in jd.must_haves))
    if jd.nice_to_haves:
        parts.append("NICE: " + "; ".join(fmt(r) for r in jd.nice_to_haves))
    digest = "\n".join(parts)
    return digest if len(digest) <= max_chars else digest[:max_chars]
