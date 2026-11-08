"""E33: controlled noise on CVs with independent gold, so the extraction numbers can be reproduced and stress-tested.

The gold is not ours: the `coding` branch's generator (cv_agent/data/synthetic.py, seed 42) samples each CV's facts
first (tools, years, depth, name, ...) and only then renders the text; `latent.json` keeps those facts. This tool
never writes new facts. It takes the rendered CVs, applies one kind of noise at a set rate with a fixed seed, writes the
noisy copies to disk (so every input can be opened), runs the same NLP pipeline the app uses, and scores against the
unchanged gold. Level 0 ("original") reproduces docs/EXTRACTION_RESULTS.md; the other rows show how fast each stage
degrades.

Noise kinds (applied to the 193 clean CVs; the generator's own 7 noisy CVs stay in "original"):
    char      a share of characters replaced by '#@%~' (the generator's own model of a badly parsed PDF)
    ocr       OCR-style confusions in a share of words (I→l, l→1, O→0, rn→m, cl→d, dropped diacritics)
    accents   a share of words typed without Vietnamese diacritics
    layout    a share of line breaks lost, as when a PDF flattens columns
    truncate  the end of the CV cut off

Every noisy CV is also rendered as a text-layer PDF (fpdf2, A4, Arial) and read back through the app's own
`read_file` (pypdf) → `process`, so the second table is the real upload path, line wrapping and all. The third table
re-ranks the 6 `coding` JDs with B0 BM25 and B3 8-axis ⊕ CE [hybrid] (rules + cross-encoder, no LLM; JD requirements
from the cache) on the PDF-read CVs, scored with nDCG@10 against the coding gold.

    python -m app.eval.noise_robustness            → docs/NOISE_ROBUSTNESS_RESULTS.md (+ .runtime/bench/coding_noise/<kind>_<level>/{cvs,pdf}/)
    python -m app.eval.noise_robustness --no-rank  → skip the ranking table (minutes per condition)
"""

import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from app.config import ROOT, get_settings
from app.eval.bench import CODING_DIR
from app.eval.extraction import prf, to_key
from app.nlp import quality
from app.nlp.pipeline import process
from app.nlp.text import read_file

OUT = get_settings().runtime_dir / "bench" / "coding_noise"
DOC = ROOT / "docs" / "NOISE_ROBUSTNESS_RESULTS.md"
SEED = 42
SWEEP = {"char": (0.02, 0.05, 0.10, 0.25), "ocr": (0.05, 0.15, 0.30), "accents": (0.5, 1.0),
         "layout": (0.25, 0.5, 1.0), "truncate": (0.25, 0.5)}
RANK_CONDITIONS = (("original", 0.0), ("char", 0.05), ("char", 0.10), ("ocr", 0.15), ("accents", 1.0),
                   ("layout", 1.0), ("truncate", 0.25))
RANK_SYSTEMS = ("B0 BM25", "B3 8-axis ⊕ CE [hybrid]")
FONTS = ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf")
NOISE_CHARS = "#@%~"
OCR_SWAPS = (("rn", "m"), ("cl", "d"), ("I", "l"), ("l", "1"), ("O", "0"), ("o", "0"), ("S", "5"), ("B", "8"))
WORD = re.compile(r"\S+")


def strip_accents(word: str) -> str:
    word = unicodedata.normalize("NFD", word.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(ch for ch in word if not unicodedata.combining(ch)))


def _ocr_word(word: str, rng: random.Random) -> str:
    hits = [(a, b) for a, b in OCR_SWAPS if a in word]
    if hits:
        a, b = rng.choice(hits)
        return word.replace(a, b, 1)
    return strip_accents(word)


def perturb(text: str, kind: str, level: float, rng: random.Random) -> str:
    """One kind of noise at one rate. The same (text, kind, level, seed) always gives the same output."""
    if kind == "original" or level <= 0:
        return text
    if kind == "char":
        return "".join(rng.choice(NOISE_CHARS) if not ch.isspace() and rng.random() < level else ch for ch in text)
    if kind == "ocr":
        return WORD.sub(lambda m: _ocr_word(m.group(0), rng) if rng.random() < level else m.group(0), text)
    if kind == "accents":
        return WORD.sub(lambda m: strip_accents(m.group(0)) if rng.random() < level else m.group(0), text)
    if kind == "layout":
        return re.sub(r"\n", lambda m: " " if rng.random() < level else "\n", text)
    if kind == "truncate":
        return text[: int(len(text) * (1 - level))]
    raise ValueError(f"unknown noise kind {kind!r}")


def gold_tools(x: dict) -> set[str]:
    return {k for t in x["tools"] + x["langs"] + x["cicd"] if (k := to_key(t))}


def score(latent: list[dict], texts: dict[str, str], vocab: set[str]) -> dict:
    """Tool P/R/F1, years MAE, depth accuracy, PII leaks and the share flagged for review, against the generator's facts."""
    tp = fp = fn = 0
    years_err, depth_hit, flagged, leaks, fields = [], 0, 0, 0, 0
    for x in latent:
        raw = texts[x["cv_id"]]
        p = process(x["cv_id"], raw)
        gold, pred = gold_tools(x), {k for k in p.skills if k in vocab}
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
        if p.years_total is not None:
            years_err.append(abs(p.years_total - x["years"]))
        depth_hit += p.depth_cue == x["depth"]
        flagged += bool(quality.flags(p))
        for value in (x["name"], {"M": "Male", "F": "Female"}[x["gender"]]):
            if value.lower() in raw.lower():
                fields += 1
                leaks += value.lower() in p.text.lower()
    p_, r_, f_ = prf(tp, fp, fn)
    n = len(latent)
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "p": p_, "r": r_, "f1": f_,
            "years_mae": sum(years_err) / len(years_err) if years_err else None, "years_n": len(years_err),
            "years_within_1": sum(e <= 1 for e in years_err), "depth_acc": depth_hit / n,
            "flagged": flagged / n, "leaks": leaks, "pii_fields": fields}


def conditions() -> list[tuple[str, float]]:
    return [("original", 0.0)] + [(k, lv) for k, levels in SWEEP.items() for lv in levels]


def font() -> Path:
    path = next((Path(f) for f in FONTS if Path(f).exists()), None)
    if path is None:
        raise SystemExit(f"no Unicode TTF font for the PDFs; tried {FONTS}")
    return path


def render_pdf(text: str) -> bytes:
    """CV text → a text-layer A4 PDF, the way a CV exported from Word looks to the parser (wrapped lines, pages)."""
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 18)
    pdf.add_page()
    pdf.add_font("cv", fname=str(font()))
    pdf.set_font("cv", size=10.5)
    for line in text.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 5.2, line, new_x="LMARGIN", new_y="NEXT", wrapmode="CHAR")
        else:
            pdf.ln(3)
    return bytes(pdf.output())


def noisy_texts(latent: list[dict], raw: dict[str, str], kind: str, level: float, every_cv: bool = False) -> dict[str, str]:
    """The noisy copy of each CV in the condition's pool. `every_cv` (ranking) keeps the whole pool and leaves the CVs
    the noise does not apply to unchanged."""
    out = {}
    for x in latent:
        applies = kind == "original" or (not x["low_quality"] and (kind != "accents" or x["lang_cv"] == "vi"))
        if not applies and not every_cv:
            continue
        rng = random.Random(f"{SEED}:{kind}:{level}:{x['cv_id']}")
        out[x["cv_id"]] = perturb(raw[x["cv_id"]], kind, level, rng) if applies else raw[x["cv_id"]]
    return out


def via_pdf(texts: dict[str, str], folder: Path) -> dict[str, str]:
    """Render each text as a PDF under `folder`, then read it back with the app's own reader."""
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for cv, t in texts.items():
        path, txt = folder / f"{cv}.pdf", folder.parent / "cvs" / f"{cv}.txt"
        # reuse a PDF rendered from this exact text (same seed → same text), otherwise render it
        if path.exists() and txt.exists() and txt.read_text(encoding="utf-8") == t:
            data = path.read_bytes()
        else:
            data = render_pdf(t)
            path.write_bytes(data)
        out[cv] = read_file(f"{cv}.pdf", data)
    return out


def rank(texts: dict[str, str]) -> dict[str, float]:
    """B0 and B3 on the 6 coding JDs with these CV texts; mean nDCG@10."""
    from app.eval.bench import Bench, load
    from app.eval.run import retrieval_systems, score_system
    from app.rag.corpus import Doc

    b = load("coding")
    noisy = Bench(b.name, [Doc(d.id, texts.get(d.id, d.text), d.meta) for d in b.cvs], b.jds, b.gold, b.traps)
    systems = retrieval_systems(get_settings().embed_dense_model, noisy, set(RANK_SYSTEMS))
    out = {}
    for name, per_jd in systems.items():
        s = score_system(per_jd, b.gold, b.traps)
        out[name] = sum(s["ndcg@10"]) / len(s["ndcg@10"])
    return out


def run(write: bool = True, ranking: bool = True, sample: int = 0) -> dict:
    latent = json.loads((CODING_DIR / "latent.json").read_text(encoding="utf-8"))
    if sample:  # a seeded random subset of CVs, for a quick run; the ranking pool stays whole
        keep = set(random.Random(SEED).sample([x["cv_id"] for x in latent], sample))
        scored = [x for x in latent if x["cv_id"] in keep]
    else:
        scored = latent
    vocab = {k for x in latent for k in gold_tools(x)}
    raw = {x["cv_id"]: (CODING_DIR / "cvs" / f"{x['cv_id']}.txt").read_text(encoding="utf-8") for x in latent}
    by_id = {x["cv_id"]: x for x in latent}
    text_rows, pdf_rows, rank_rows = [], [], []
    for kind, level in conditions():
        texts = noisy_texts(scored, raw, kind, level)
        pool = [by_id[c] for c in texts]
        base = OUT / f"{kind}_{level:g}"
        if write:
            (base / "cvs").mkdir(parents=True, exist_ok=True)
            for cv, t in texts.items():
                (base / "cvs" / f"{cv}.txt").write_text(t, encoding="utf-8")
        text_rows.append({"kind": kind, "level": level, **score(pool, texts, vocab)})
        pdf_rows.append({"kind": kind, "level": level, **score(pool, via_pdf(texts, base / "pdf"), vocab)})
        print(f"{kind} {level:g}: text F1 {text_rows[-1]['f1']:.3f} · PDF F1 {pdf_rows[-1]['f1']:.3f}", flush=True)
    if ranking:
        for kind, level in (RANK_CONDITIONS[:2] if sample else RANK_CONDITIONS):
            texts = noisy_texts(latent, raw, kind, level, every_cv=True)
            rank_rows.append({"kind": kind, "level": level,
                              **rank(via_pdf(texts, OUT / f"{kind}_{level:g}" / "pdf_pool"))})
            print(f"rank {kind} {level:g}: {rank_rows[-1]}", flush=True)
    return {"text": text_rows, "pdf": pdf_rows, "rank": rank_rows}


def _table(rows: list[dict]) -> list[str]:
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    out = ["| noise | level | CVs | tool P / R / F1 | TP / FP / FN | years found | years within 1 y (of all CVs) | "
           "years MAE (where found) | depth acc | flagged for review | PII leaks (name, gender) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['kind']} | {r['level']:g} | {r['n']} | {f3(r['p'])} / {f3(r['r'])} / **{f3(r['f1'])}** | "
                   f"{r['tp']} / {r['fp']} / {r['fn']} | {r['years_n']}/{r['n']} | {r['years_within_1'] / r['n']:.0%} | "
                   f"{f3(r['years_mae'])} | {r['depth_acc']:.1%} | {r['flagged']:.0%} | {r['leaks']}/{r['pii_fields']} |")
    return out


def report(res: dict) -> str:
    out = ["# Noise robustness of the extraction (generated by `python -m app.eval.noise_robustness`; do not edit by hand)", "",
           "Gold: `.runtime/bench/coding/latent.json`, the facts the `coding` branch's generator sampled before rendering "
           "each CV (seed 42); this tool never changes them. Noisy inputs: `.runtime/bench/coding_noise/<kind>_<level>/cvs/` "
           "(text) and `…/pdf/` (the same text as a PDF). "
           f"Seed {SEED} per (kind, level, CV). `original` = the generator's 200 CVs as rendered (its 7 noisy CVs included) "
           "and must match `docs/EXTRACTION_RESULTS.md`; the noise rows use the 193 clean CVs (`accents`: the clean "
           "Vietnamese ones). `ocr` here is simulated character confusion; the real PaddleOCR run is `docs/OCR_RESULTS.md`.", "",
           *([f"**Quick run:** a seeded random sample of {res['sample']} of the 200 CVs (seed {SEED}), and the ranking "
              "table for `original` and `char 0.05` only. A full run omits `--sample`.", ""] if res.get("sample") else []),
           "## 1. Text in, straight to the NLP pipeline", "", *_table(res["text"]), "",
           "## 2. PDF in, through the app's upload path (`read_file` → pypdf → `process`)", "",
           "Each noisy CV rendered as a text-layer A4 PDF (fpdf2, Arial 10.5 pt, long lines wrapped) and read back.", "",
           *_table(res["pdf"]), ""]
    if res["rank"]:
        names = list(RANK_SYSTEMS)
        base = res["rank"][0]
        out += ["## 3. Ranking on the PDF-read CVs: coding benchmark, 6 JDs, mean nDCG@10", "",
                "All 200 CVs in the pool; the noise applies as in the tables above and leaves the other CVs unchanged. "
                "Systems without LLM judgement; JD requirements from `.runtime/eval/requirements/coding/`. PDFs: "
                "`…/<kind>_<level>/pdf_pool/`.", "",
                "| noise | level | " + " | ".join(f"{n} | Δ vs original" for n in names) + " |",
                "|---|---|" + "---|---|" * len(names)]
        for r in res["rank"]:
            cells = " | ".join(f"**{r[n]:.3f}** | {r[n] - base[n]:+.3f}" for n in names)
            out.append(f"| {r['kind']} | {r['level']:g} | {cells} |")
        out.append("")
    out += ["## Reading the tables", "",
            "- *years MAE* is over the CVs where a date range was still found, so it can fall as noise rises (only the "
            "easy CVs remain); read it with *years found* and *years within 1 y (of all CVs)*.",
            "- *flagged for review* is the X10 quality gate: the share of CVs sent to HR or the investigator (N12) "
            "instead of being scored silently.",
            "- A PII leak counts a name or gender value from the original CV that is still in the text the pipeline "
            "indexes after redaction.",
            "- A PDF wraps long lines, so `layout` noise (lost line breaks) comes back with breaks at wrap points, as a "
            "real exported CV would.", ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    sample = int(argv[argv.index("--sample") + 1]) if "--sample" in argv else 0
    res = run(write="--no-write" not in argv, ranking="--no-rank" not in argv, sample=sample)
    res["sample"] = sample
    DOC.write_text(report(res), encoding="utf-8")
    print(f"→ {DOC} · noisy CVs in {OUT}")
    r = res["text"][0]
    print(f"original: F1 {r['f1']:.3f}, years MAE {r['years_mae']:.2f}, depth {r['depth_acc']:.1%}, leaks {r['leaks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
