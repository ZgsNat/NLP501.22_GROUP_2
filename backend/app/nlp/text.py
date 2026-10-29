"""Stage 1–2 of the CV pipeline: read any file format into text, then normalise it so every later stage (regexes,
BM25 tokens, grounding offsets) sees one canonical form.

Real CVs arrive as PDF/DOCX exports with decomposed Vietnamese diacritics, ligatures, odd bullets and hard-wrapped
lines. None of that is meaning, all of it breaks exact matching, so it is removed here once."""

import io
import re
import unicodedata
from pathlib import Path

BULLET = re.compile(r"^[ \t]*(?:[•●▪◦■►▸✓✔∙·]|[*+\-–—](?=\s))[ \t]*", re.M)  # "*x*" emphasis is not a bullet
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
INVISIBLE = re.compile("[\\u200b-\\u200d\\ufeff\\u00ad]")  # zero-width chars and soft hyphens
# Letters that exist in Vietnamese but not in English (base vowels + all tone-marked forms after NFC).
VI_CHARS = re.compile(r"[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]", re.I)
VI_STOPWORDS = {"và", "của", "các", "trong", "với", "được", "cho", "những", "tại", "năm", "làm", "việc", "kiểm", "thử"}


def read_file(name: str, data: bytes) -> str:
    """Bytes of an uploaded CV → raw text. Supports .md/.txt, .pdf (text layer) and .docx (paragraphs + tables)."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(data))
        lines = [p.text for p in doc.paragraphs]
        lines += [" | ".join(c.text for c in row.cells) for t in doc.tables for row in t.rows]
        return "\n".join(lines)
    if ext in {".md", ".txt", ""}:
        return data.decode("utf-8-sig", errors="replace")
    raise ValueError(f"Unsupported CV format: {ext} (use .pdf, .docx, .md or .txt)")


def normalize(text: str) -> str:
    """Canonical text: NFC Unicode, plain spaces, '- ' bullets, de-hyphenated and re-joined hard-wrapped lines."""
    text = unicodedata.normalize("NFC", text)  # "ê" + combining acute (PDF/Word) → "ế": one code point, one token
    text = INVISIBLE.sub("", text)
    for lig, plain in LIGATURES.items():
        text = text.replace(lig, plain)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace(" ", " ")
    text = BULLET.sub("- ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "automa-\ntion" → "automation" (PDF line-break hyphenation)
    lines: list[str] = []
    for raw in text.split("\n"):
        line = re.sub(r" {2,}", " ", raw).strip()
        # A hard-wrapped continuation starts lowercase and is not a new bullet → it belongs to the previous line.
        if line and lines and lines[-1] and line[0].islower() and not lines[-1].startswith("#"):
            lines[-1] += " " + line
        else:
            lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def detect_language(text: str) -> str:
    """'vi' | 'mixed' | 'en' from the share of words that are unmistakably Vietnamese (no model, no dependency)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    if not words:
        return "en"
    vi = sum(bool(VI_CHARS.search(w)) or w in VI_STOPWORDS for w in words) / len(words)
    return "vi" if vi >= 0.35 else "mixed" if vi >= 0.06 else "en"
