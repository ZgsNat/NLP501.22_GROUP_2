"""Data-quality flags (ported from the `coding` branch, ingest/extractor.parse_quality and ingest/pipeline.mark_duplicates).
A flag never removes a CV: it tells HR that the parse, and so the score, deserves a second look."""

import re

from app.nlp.pipeline import CVProfile


def flags(p: CVProfile) -> list[str]:
    words = len(p.text.split())
    out = []
    if words < 120:
        out.append("short_text")
    if sum(ch.isalpha() for ch in p.text) / max(len(p.text), 1) < 0.45:
        out.append("garbled_text")
    if len(p.skills) < 3:
        out.append("few_skills")
    if not any(s.kind in ("experience", "projects") for s in p.sections):
        out.append("no_experience_section")
    if p.years_total is None:
        out.append("no_dated_roles")
    return out


def _shingles(text: str, k: int = 5) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 0))}


def duplicates(profiles: list[CVProfile], threshold: float = 0.85) -> dict[str, str]:
    """cv_id → the earlier CV it near-duplicates (Jaccard over 5-word shingles). Flagged for HR, not removed."""
    sig = {p.cv_id: _shingles(p.text) for p in profiles}
    out: dict[str, str] = {}
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            sa, sb = sig[a.cv_id], sig[b.cv_id]
            if b.cv_id not in out and sa and sb and len(sa & sb) / len(sa | sb) >= threshold:
                out[b.cv_id] = a.cv_id
    return out
