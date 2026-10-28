"""Stage 1 — Ingest: parse -> mask PII -> extract -> depth -> quality flags."""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from cv_agent.ingest.extractor import assess_depth, detect_language, estimate_years, extract_entities, parse_quality
from cv_agent.ingest.parsers import SUPPORTED, read_text
from cv_agent.ingest.pii import mask_pii
from cv_agent.schema import CV

log = logging.getLogger(__name__)


def ingest_text(raw_text: str, cv_id: Optional[str] = None, source_path: Optional[str] = None,
                n_images: int = 0) -> CV:
    raw_text = unicodedata.normalize("NFC", raw_text)  # PDFs often yield decomposed Vietnamese
    masked = mask_pii(raw_text)
    text = masked.text
    entities = extract_entities(text)
    has_tool = any(e.type == "tool" for e in entities)
    quality, flags = parse_quality(text, len(entities))
    pii_counts = dict(masked.counts)
    if n_images:
        pii_counts["PHOTO"] = pii_counts.get("PHOTO", 0) + n_images
    return CV(
        cv_id=cv_id or "cv_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:10],
        raw_text=raw_text,
        masked_text=text,
        entities=entities,
        depth=assess_depth(text, has_tool),
        years_experience=estimate_years(text),
        language=detect_language(text),
        parse_quality=quality,
        flags=flags,
        pii_found=pii_counts,
        source_path=source_path,
    )


def ingest_file(path: Path, cv_id: Optional[str] = None) -> CV:
    raw, n_images = read_text(path)
    return ingest_text(raw, cv_id or path.stem, str(path), n_images)


def ingest_dir(directory: Path) -> List[CV]:
    """Ingest every supported file; unreadable files are logged and flagged, not dropped silently."""
    cvs: List[CV] = []
    for path in sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in SUPPORTED):
        try:
            cvs.append(ingest_file(path))
        except Exception as exc:  # keep going, but make the failure visible
            log.error("Failed to ingest %s: %s", path, exc)
            cvs.append(CV(cv_id=path.stem, raw_text="", masked_text="", parse_quality=0.0,
                          flags=["parse_failed", f"error:{type(exc).__name__}"], source_path=str(path)))
    return mark_duplicates(cvs)


# ---------------------------------------------------------------- duplicates (FR-06)

def _shingles(text: str, k: int = 5) -> set:
    words = re.findall(r"\w+", text.lower())
    if not words:  # empty / failed parses are never "duplicates" of each other
        return set()
    return {" ".join(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))}


def mark_duplicates(cvs: Iterable[CV], threshold: float = 0.85) -> List[CV]:
    """Flag near-duplicate CVs (Jaccard over 5-word shingles). Duplicates are flagged for HR, not removed."""
    cvs = list(cvs)
    sigs: Dict[str, set] = {cv.cv_id: _shingles(cv.masked_text) for cv in cvs}
    for i, a in enumerate(cvs):
        for b in cvs[i + 1:]:
            sa, sb = sigs[a.cv_id], sigs[b.cv_id]
            if sa and sb and len(sa & sb) / len(sa | sb) >= threshold:
                b.flags.append(f"duplicate_of:{a.cv_id}")
    return cvs
