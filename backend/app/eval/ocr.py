"""Scanned CVs: what reading a CV through OCR (app.nlp.ocr, PaddleOCR) costs the NLP pipeline, and whether masking
still holds.

Clean CVs of the `coding` set (stratified English / Vietnamese) are rendered as image-only PDFs in two conditions,
then read back through the real upload path (`read_file` → no text layer → OCR → `process`):

    scan          flat-bed scan: slight tilt, blur and noise, JPEG 75
    phone photo   tilted, low contrast, blurrier, noisier, JPEG 45

Gold is the generator's latent facts (as in app.eval.extraction); the same CVs read as text are the baseline, so every
difference is the cost of OCR. Masking is checked against what OCR actually produced: a label OCR misspells
("Gii tính") is a label the redactor may no longer recognise.

    uv run --group ocr python -m app.eval.ocr [--n 30]     → docs/OCR_RESULTS.md
    ... --heldout                                          30 other CVs, never looked at → docs/OCR_RESULTS_heldout.md
    python -m app.eval.ocr rescore                         re-score the saved OCR text after a pipeline change
"""

import argparse
import io
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.config import ROOT, get_settings
from app.eval.bench import CODING_DIR
from app.eval.extraction import prf, to_key
from app.nlp import ocr
from app.nlp.pipeline import process
from app.nlp.text import read_file

DOC = ROOT / "docs" / "OCR_RESULTS.md"
FONTS = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
CONDITIONS = {"scan": dict(angle=0.4, blur=0.4, noise=6, contrast=1.0, quality=75),
              "phone photo": dict(angle=1.5, blur=0.9, noise=14, contrast=0.75, quality=45)}
GENDER_LINE = re.compile(r"^\s*(?:gender|sex|g\w{0,2}i\s*t\w{0,2}nh)\s*[:：]\s*\S", re.I | re.M)  # label survived


def render(text: str, seed: int, angle: float, blur: float, noise: float, contrast: float, quality: int) -> bytes:
    """CV text → an image-only PDF (A4 at 150 dpi, Arial 24 px), degraded like a scan or a phone photo."""
    font = next(ImageFont.truetype(f, 24) for f in FONTS if os.path.exists(f))
    W, H, M, step = 1240, 1754, 100, 36
    rng = np.random.default_rng(seed)
    pages, y = [Image.new("L", (W, H), 255)], M
    for para in text.splitlines():
        words, line = para.split(), ""
        for w in words + [None]:  # greedy wrap to the text width
            if w is not None and font.getlength(f"{line} {w}".strip()) <= W - 2 * M:
                line = f"{line} {w}".strip()
                continue
            if y > H - M:
                pages.append(Image.new("L", (W, H), 255))
                y = M
            ImageDraw.Draw(pages[-1]).text((M, y), line, font=font, fill=0)
            y += step
            line = w or ""
            if w is None:
                break
    out = []
    for img in pages:
        img = img.rotate(rng.uniform(-angle, angle), resample=Image.BICUBIC, fillcolor=255)
        img = ImageEnhance.Contrast(img).enhance(contrast).filter(ImageFilter.GaussianBlur(blur))
        img = Image.fromarray((np.asarray(img, float) + rng.normal(0, noise, (H, W))).clip(0, 255).astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        out.append(Image.open(buf))
    pdf = io.BytesIO()
    out[0].save(pdf, "PDF", resolution=150, save_all=True, append_images=out[1:])
    return pdf.getvalue()


def cer(ref: str, hyp: str) -> float:
    """Character error rate: edit distance / reference length, whitespace collapsed."""
    a, b = " ".join(ref.split()), " ".join(hyp.split())
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def leaks(raw: str, read: str, masked: str) -> dict[str, bool]:
    """Which PII the masked text still shows. `read` is the text as OCR saw it, so a misread name still counts."""
    low, digits = masked.lower(), re.sub(r"\D", "", masked)
    name = next(ln.strip() for ln in read.splitlines() if ln.strip())
    phone = re.search(r"\b0\d{9}\b", raw)
    born = re.search(r"\b\d{2}/\d{2}/\d{4}\b", raw)
    return {"name": name.lower() in low, "email": "example.com" in low,
            "phone": bool(phone) and phone[0] in digits, "birth date": bool(born) and born[0] in masked,
            "gender": bool(GENDER_LINE.search(masked)), "address": "nguyen trai" in low}


def picked(n: int, heldout: bool = False) -> list[dict]:
    """n/2 English and n/2 Vietnamese clean CVs; `heldout` = the next n/2 of each, disjoint from the default set the
    fixes were developed on."""
    latent = [x for x in json.loads((CODING_DIR / "latent.json").read_text()) if not x["low_quality"]]
    rng = np.random.default_rng(0)
    return [x for lang in ("en", "vi") for x in
            rng.permutation(sorted((x for x in latent if x["lang_cv"] == lang), key=lambda x: x["cv_id"]))
            [n // 2 * heldout: n // 2 * (1 + heldout)]]


def read(n: int, heldout: bool = False) -> dict:
    """The slow part: render every picked CV in each condition and read it back through OCR."""
    out = {}
    for x in picked(n, heldout):
        raw = (CODING_DIR / "cvs" / f"{x['cv_id']}.txt").read_text(encoding="utf-8")
        out[x["cv_id"]] = {}
        for cond, kw in CONDITIONS.items():
            pdf = render(raw, int(x["cv_id"].split("_")[1]), **kw)
            t0 = time.perf_counter()
            out[x["cv_id"]][cond] = {"text": read_file(f"{x['cv_id']}.pdf", pdf), "seconds": time.perf_counter() - t0}
        print(x["cv_id"], x["lang_cv"], " · ".join(f"{c} CER {cer(raw, v['text']):.3f}"
                                                  for c, v in out[x["cv_id"]].items()), flush=True)
    return out


def score(texts: dict) -> dict:
    """The fast part: the current NLP pipeline over the saved OCR text (re-run after a pipeline change, no OCR)."""
    latent = {x["cv_id"]: x for x in json.loads((CODING_DIR / "latent.json").read_text())}
    vocab = {k for x in latent.values() if not x["low_quality"] for t in x["tools"] + x["langs"] + x["cicd"] if (k := to_key(t))}
    rows = []
    for cid, conds in texts.items():
        x, raw = latent[cid], (CODING_DIR / "cvs" / f"{cid}.txt").read_text(encoding="utf-8")
        gold = {k for t in x["tools"] + x["langs"] + x["cicd"] if (k := to_key(t))}
        for cond, v in {"text (no OCR)": {"text": raw, "seconds": 0.0}, **conds}.items():
            v = {**v, "text": ocr.fix_capital_i(v["text"]) if cond != "text (no OCR)" else raw}  # idempotent
            p = process(cid, v["text"])
            pred = {k for k in p.skills if k in vocab}
            rows.append({"cv": cid, "lang": x["lang_cv"], "cond": cond, "seconds": v["seconds"], "cer": cer(raw, v["text"]),
                         "tp": len(pred & gold), "fp": len(pred - gold), "fn": len(gold - pred), "missed": sorted(gold - pred),
                         "years_err": None if p.years_total is None else abs(p.years_total - x["years"]),
                         "experience": any(s.kind in ("experience", "projects") for s in p.sections),
                         "leaks": leaks(raw, v["text"], p.text), "text": v["text"] if cond != "text (no OCR)" else ""})
    return {"n": len(texts), "texts": texts, "rows": rows}


def render_doc(res: dict) -> str:
    rows = res["rows"]
    lines = ["# OCR: scanned CVs through the NLP pipeline (generated by `python -m app.eval.ocr`; do not edit by hand)", "",
             f"{res['n']} clean CVs of the `coding` set (half English, half Vietnamese) rendered as image-only PDFs and read "
             "back through `read_file` → PaddleOCR PP-OCRv5 (`lang=\"vi\"`, the latin model) → `process`. Gold: the "
             "generator's latent facts. *text (no OCR)* is the same CVs read as text: the difference is the cost of OCR.", "",
             "| Condition | slice | CER | tool P / R / F1 | years within 1 y | experience section found | s / page |",
             "|---|---|---|---|---|---|---|"]
    for cond in ["text (no OCR)", *CONDITIONS]:
        for lang in ("all", "en", "vi"):
            rs = [r for r in rows if r["cond"] == cond and lang in ("all", r["lang"])]
            tp, fp, fn = (sum(r[k] for r in rs) for k in ("tp", "fp", "fn"))
            yrs = [r["years_err"] for r in rs if r["years_err"] is not None]
            lines.append(f"| {cond if lang == 'all' else ''} | {lang} | {sum(r['cer'] for r in rs) / len(rs):.3f} | "
                         + " / ".join(f"{v:.3f}" for v in prf(tp, fp, fn))
                         + f" | {sum(e <= 1 for e in yrs)}/{len(rs)} | {sum(r['experience'] for r in rs)}/{len(rs)} | "
                         + (f"{sum(r['seconds'] for r in rs) / len(rs):.1f} |" if cond != "text (no OCR)" else "— |"))
    fields = list(rows[0]["leaks"])
    lines += ["", "## Masking after OCR (PII still visible in the text that is indexed and shown to the LLM)", "",
              "| Condition | slice | " + " | ".join(fields) + " |", "|---|---|" + "---|" * len(fields)]
    for cond in ["text (no OCR)", *CONDITIONS]:
        for lang in ("en", "vi"):
            rs = [r for r in rows if r["cond"] == cond and r["lang"] == lang]
            lines.append(f"| {cond if lang == 'en' else ''} | {lang} | "
                         + " | ".join(f"{sum(r['leaks'][f] for r in rs)}/{len(rs)}" for f in fields) + " |")
    if before := res.get("before"):
        lines += ["", "## Before → after the three fixes", "",
                  "The first run found three defects, all in how our pipeline reads OCR output, none in the benchmark data:",
                  "capital I read as \"l\" in tool names (`ocr.fix_capital_i`), Vietnamese headings with dropped letters "
                  "(`segment.loose_heading`) and PII labels with dropped letters (`redact.loose_kind`). Same OCR text, re-scored:", "",
                  "| Condition | tool recall | Vietnamese CVs with an experience section | Vietnamese address leaks | Vietnamese gender leaks |",
                  "|---|---|---|---|---|"]
        for cond in CONDITIONS:
            cells = []
            for rs in ([r for r in before if r["cond"] == cond], [r for r in rows if r["cond"] == cond]):
                vi = [r for r in rs if r["lang"] == "vi"]
                cells.append((prf(*(sum(r[k] for r in rs) for k in ("tp", "fp", "fn")))[1], sum(r["experience"] for r in vi),
                              sum(r["leaks"]["address"] for r in vi), sum(r["leaks"]["gender"] for r in vi), len(vi)))
            (r0, e0, a0, g0, n0), (r1, e1, a1, g1, _) = cells
            lines.append(f"| {cond} | {r0:.3f} → {r1:.3f} | {e0}/{n0} → {e1}/{n0} | {a0}/{n0} → {a1}/{n0} | {g0}/{n0} → {g1}/{n0} |")
    ex = next((r for r in rows if r["cond"] == "scan" and r["lang"] == "vi"), None)
    if ex:
        lines += ["", f"What the latin model reads on a Vietnamese scan ({ex['cv']}, first 160 characters):", "",
                  "```", ex["text"][:160], "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("mode", nargs="?", choices=["rescore"], help="re-score the saved OCR text, no OCR")
    ap.add_argument("--heldout", action="store_true", help="CVs disjoint from the set the fixes were developed on")
    a = ap.parse_args()
    path = get_settings().runtime_dir / "eval" / ("ocr_heldout.json" if a.heldout else "ocr.json")
    if a.heldout:
        DOC = DOC.with_name("OCR_RESULTS_heldout.md")
    if a.mode == "rescore":
        old = json.loads(path.read_text())
        res = {**score(old["texts"]), "before": old.get("before") or old["rows"]}  # the first scoring stays the baseline
    elif not ocr.available():
        raise SystemExit("OCR is not installed: uv sync --group ocr")
    else:
        res = score(read(a.n, a.heldout))
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    DOC.write_text(render_doc(res))
    print(DOC.read_text())
