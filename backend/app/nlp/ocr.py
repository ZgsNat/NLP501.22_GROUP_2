"""Stage 1b: OCR for CVs without a text layer (scanned PDFs, photos), with PaddleOCR (PP-OCRv5).

Optional: `uv sync --group ocr` (PaddlePaddle, about 1 GB; models download once to ~/.paddlex). Without it, PDFs
are read from their text layer only and image CVs are rejected. PaddleOCR has no Vietnamese model: lang="vi" uses
its latin model, which drops letters with stacked diacritics ("NGHIỆM" → "NGHIM"). docs/OCR_RESULTS.md measures what
that costs downstream."""

import io
import re
from functools import lru_cache

MIN_CHARS = 20  # a PDF page with fewer extractable characters is a scan
DPI = 200  # scanned pages are rendered at this resolution before OCR


def available() -> bool:
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def engine():
    from paddleocr import PaddleOCR

    # Fixed detection size: with the default (no downscaling) the detector found no text at all on a clean 200-dpi
    # A4 scan, yet read the same page cropped or at 150 dpi. Long side 1600 px keeps 10 pt text legible.
    return PaddleOCR(lang="vi", use_doc_orientation_classify=False, use_doc_unwarping=False,
                     use_textline_orientation=False, text_det_limit_type="max", text_det_limit_side_len=1600)


def lines(texts: list[str], boxes: list[list[float]]) -> str:
    """OCR text boxes [x0, y0, x1, y1] → text in reading order. Boxes whose vertical centres are within half a line
    height share a line (left to right); a vertical gap taller than a line becomes a blank line, so sections stay
    apart for the segmenter."""
    rows: list[dict] = []
    for t, (x0, y0, x1, y1) in sorted(zip(texts, boxes), key=lambda tb: ((tb[1][1] + tb[1][3]) / 2, tb[1][0])):
        c, h = (y0 + y1) / 2, y1 - y0
        if rows and abs(c - rows[-1]["c"]) < max(h, rows[-1]["h"]) / 2:
            rows[-1]["items"].append((x0, t))
            rows[-1]["bottom"] = max(rows[-1]["bottom"], y1)
        else:
            rows.append({"c": c, "h": h, "top": y0, "bottom": y1, "items": [(x0, t)]})
    out = []
    for i, r in enumerate(rows):
        if i and r["top"] - rows[i - 1]["bottom"] > max(r["h"], rows[i - 1]["h"]):
            out.append("")
        out.append(" ".join(t for _, t in sorted(r["items"])))
    return "\n".join(out)


# An "l" touching a capital is how the latin model usually reads a capital I: "WebdriverlO", "XCUlTest", "GitLab Cl".
CAPITAL_I = re.compile(r"\b\w*(?:[A-Z]l|l[A-Z])\w*\b")


@lru_cache(maxsize=1)
def tool_words() -> frozenset[str]:
    from app.nlp.taxonomy import aliases, skills

    return frozenset(w for k in skills() for a in aliases(k) for w in re.findall(r"\w+", a.lower()))


def fix_capital_i(text: str) -> str:
    """Undo the I → l confusion, but only where the corrected word is part of a known tool name (measured: it caused
    every tool the OCR path missed, docs/OCR_RESULTS.md)."""
    def fix(m: re.Match) -> str:
        w = m[0]
        cand = re.sub(r"(?<=[A-Z])l|l(?=[A-Z])", "I", w)
        return cand if cand.lower() in tool_words() and w.lower() not in tool_words() else w

    return CAPITAL_I.sub(fix, text)


def image_text(img) -> str:
    """A PIL image of one page → its text."""
    import numpy as np

    res = engine().predict(np.asarray(img.convert("RGB"))[:, :, ::-1])[0]  # PaddleOCR expects BGR, like OpenCV
    return fix_capital_i(lines(list(res["rec_texts"]), [list(map(float, b)) for b in res["rec_boxes"]]))


def pdf_page(data: bytes, index: int):
    """Page `index` of a PDF, rendered as a PIL image."""
    import pypdfium2 as pdfium

    return pdfium.PdfDocument(io.BytesIO(data))[index].render(scale=DPI / 72).to_pil()
