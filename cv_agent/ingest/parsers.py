"""Read CV files (PDF, DOCX, TXT) into plain text (FR-01)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


class UnsupportedFormat(ValueError):
    pass


def read_text(path: Path) -> Tuple[str, int]:
    """Return (text, n_images). Images are counted so the PII report can say a photo was dropped."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFormat(f"{path.name}: unsupported format {suffix}; expected one of {sorted(SUPPORTED)}")
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace"), 0
    if suffix == ".pdf":
        return _read_pdf(path)
    return _read_docx(path)


def _read_pdf(path: Path) -> Tuple[str, int]:
    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError("PyMuPDF is required for PDF: pip install pymupdf") from exc
    with fitz.open(path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)
        n_images = sum(len(page.get_images()) for page in doc)
    return text, n_images


def _read_docx(path: Path) -> Tuple[str, int]:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("python-docx is required for DOCX: pip install python-docx") from exc
    document = docx.Document(str(path))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            lines.append(" | ".join(cell.text.strip() for cell in row.cells))
    n_images = sum(1 for rel in document.part.rels.values() if "image" in rel.reltype)
    return "\n".join(lines), n_images
