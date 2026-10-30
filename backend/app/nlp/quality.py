"""Data-quality flags (ported from the `coding` branch, ingest/extractor.parse_quality and ingest/pipeline.mark_duplicates).
A flag never removes a CV: it tells HR that the parse, and so the score, deserves a second look."""

import re

from app.nlp.pipeline import CVProfile


# Symbols inside words ("Aut%%ati#n"): OCR / encoding noise. Letter ratio alone misses it (the noise is sparse).
NOISY = re.compile(r"\w[%~#@^*]+\w|[%~#@^]{2,}")


def noisy_rate(text: str) -> float:
    toks = [t for t in text.split() if "[REDACTED" not in t]
    return sum(bool(NOISY.search(t)) for t in toks) / max(len(toks), 1)


def flags(p: CVProfile) -> list[str]:
    """Thresholds measured on the coding set (docs/EXTRACTION_RESULTS.md): garbled CVs have ≥ 55% noisy tokens, clean
    ones ≤ 3%; template CVs have a median of 68 words, so "short" means almost no content, not "concise"."""
    words = len(p.text.split())
    out = []
    if words < 50:
        out.append("short_text")
    if sum(ch.isalpha() for ch in p.text) / max(len(p.text), 1) < 0.45 or noisy_rate(p.text) > 0.15:
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
