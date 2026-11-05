"""RQ1: how well the deterministic NLP pipeline turns raw text into facts, and how well redaction masks PII.

Gold comes from the `coding` branch's generator (Đán), never from our own code:
    CVs   .runtime/bench/coding/latent.json   the latent facts each synthetic CV was rendered from
    JDs   .runtime/bench/coding/jds.yaml      `gold_spec`, the rubric its gold labels were written with
Our pipeline was not developed on this set, so the numbers are closer to held-out than anything measured on data/.

    python -m app.eval.extraction            → docs/EXTRACTION_RESULTS.md
"""

import json
import re
import sys
import unicodedata
from collections import defaultdict

import yaml

from app.agent import investigate as inv
from app.config import ROOT
from app.eval.bench import CODING_DIR
from app.nlp import quality
from app.nlp.jd import FOREIGN
from app.nlp.pipeline import process
from app.nlp.taxonomy import aliases, group, skills

OUT = ROOT / "docs" / "EXTRACTION_RESULTS.md"
REQ = ROOT / ".runtime" / "eval" / "requirements" / "coding"


def to_key(name: str) -> str | None:
    """Generator surface name → taxonomy key ('Rest Assured' → 'restassured'); None = not in our gazetteer."""
    low = name.lower()
    return next((k for k in skills() if low in aliases(k) or low.replace(" ", "") == k.replace("_", "")), None)


def strip_accents(text: str) -> str:
    """The same CV typed without Vietnamese diacritics (case kept): "Xây dựng" → "Xay dung", "Đà Nẵng" → "Da Nang"."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return unicodedata.normalize("NFC", "".join(ch for ch in text if not unicodedata.combining(ch)))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def cv_extraction() -> dict:
    latent = json.loads((CODING_DIR / "latent.json").read_text(encoding="utf-8"))
    vocab = {k for x in latent for t in x["tools"] + x["langs"] + x["cicd"] if (k := to_key(t))}
    counts = defaultdict(lambda: [0, 0, 0])  # slice → tp, fp, fn
    gaps: dict[str, int] = defaultdict(int)
    years_err, depth_hit, depth_n, istqb_hit, fl_hit, leaks, n = [], 0, 0, 0, 0, defaultdict(int), defaultdict(int)
    flagged = defaultdict(int)  # quality flags (X10) on low-quality vs clean CVs
    plain = defaultdict(int)  # Vietnamese CVs again with the diacritics stripped: depth and years robustness
    for x in latent:
        raw = (CODING_DIR / "cvs" / f"{x['cv_id']}.txt").read_text(encoding="utf-8")
        p = process(x["cv_id"], raw)
        gold = set()
        for t in x["tools"] + x["langs"] + x["cicd"]:
            k = to_key(t)
            gold.add(k) if k else gaps.__setitem__(t, gaps[t] + 1)
        pred = {k for k in p.skills if k in vocab}
        slices = ["all", x["lang_cv"], "low_quality" if x["low_quality"] else "clean"]
        if x["low_quality"]:  # what N12's code path can recover: every near-miss line proposed, then verified
            near = inv.near_misses(p, sorted(vocab - pred))
            recovered = {k for k, line in near.items() if inv.verify(p, k, line, {k})[0]}
            c = counts["low_quality + N12 search (upper bound)"]
            c[0] += len((pred | recovered) & gold)
            c[1] += len((pred | recovered) - gold)
            c[2] += len(gold - pred - recovered)
        for sl in slices:
            c = counts[sl]
            c[0] += len(pred & gold)
            c[1] += len(pred - gold)
            c[2] += len(gold - pred)
        if x["lang_cv"] == "vi" and not x["low_quality"]:
            q = process(x["cv_id"], strip_accents(raw))
            plain["n"] += 1
            plain["depth_accented"] += p.depth_cue == x["depth"]
            plain["depth_plain"] += q.depth_cue == x["depth"]
            plain["years_accented"] += p.years_total is not None and abs(p.years_total - x["years"]) <= 1
            plain["years_plain"] += q.years_total is not None and abs(q.years_total - x["years"]) <= 1
            plain["tools_same"] += {k for k in q.skills if k in vocab} == pred
        flagged["low_quality" if x["low_quality"] else "clean"] += bool(quality.flags(p))
        flagged["n_low" if x["low_quality"] else "n_clean"] += 1
        if p.years_total is not None:
            years_err.append(abs(p.years_total - x["years"]))
        depth_n += 1
        depth_hit += p.depth_cue == x["depth"]
        istqb_hit += ("istqb" in p.skills) == x["istqb"]
        fl = {f for f in ("english", "japanese") if re.search(FOREIGN[f], p.text.lower())}
        fl_hit += fl == {f for f in ("english", "japanese") if x[f]}
        # redaction: a field leaks if its value is in the raw CV and still in the text the pipeline indexes
        for field, value in (("name", x["name"]), ("gender", {"M": "Male", "F": "Female"}[x["gender"]])):
            if value.lower() in raw.lower():
                n[field] += 1
                leaks[field] += value.lower() in p.text.lower()
        for field, needle in (("email", "@example.com"), ("phone", "Phone:"), ("birth_date", "Date of birth")):
            if needle.lower() in raw.lower():
                n[field] += 1
                line = next(ln for ln in raw.splitlines() if needle.lower() in ln.lower())
                value = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
                leaks[field] += bool(value) and value.lower() in p.text.lower()
    return {"n": len(latent), "counts": dict(counts), "gaps": dict(sorted(gaps.items(), key=lambda kv: -kv[1])),
            "years_mae": sum(years_err) / len(years_err) if years_err else None, "years_n": len(years_err),
            "years_within_1": sum(e <= 1 for e in years_err), "depth_acc": depth_hit / depth_n,
            "istqb_acc": istqb_hit / len(latent), "foreign_acc": fl_hit / len(latent),
            "leaks": dict(leaks), "fields": dict(n), "flagged": dict(flagged), "plain": dict(plain)}


def jd_extraction() -> dict:
    """Field-level agreement of the three JD parsers (Settings.jd_parser) with `gold_spec`, on the cached requirements
    the ladder used (python -m app.eval.run retrieval --bench coding writes them)."""
    jds = {j["id"]: j for j in yaml.safe_load((CODING_DIR / "jds.yaml").read_text(encoding="utf-8")) if j.get("gold_spec")}
    out = {}
    for mode in ("llm", "rules", "hybrid"):
        c = defaultdict(lambda: [0, 0, 0])
        exact = defaultdict(int)
        for jid, j in jds.items():
            path = REQ / mode / f"{jid}.json"
            if not path.exists():
                continue
            r, subs, g = json.loads(path.read_text(encoding="utf-8"))["req"], json.loads(path.read_text(encoding="utf-8"))["subs"], j["gold_spec"]
            tool = lambda ks: {k for k in ks if k and group(k) not in ("language", "certification")}  # noqa: E731
            must = tool(r["must_have"] + [s for m in r["must_have"] for s in subs.get(m, [])])
            every = must | tool(r["nice_to_have"])
            g_must = tool(map(to_key, g["must"].get("tools_any", [])))
            g_all = g_must | tool(map(to_key, g.get("nice", {}).get("tools", [])))
            langs = {k for k in r["languages"] + r["must_have"] + r["nice_to_have"] if group(k) == "language"}
            g_langs = {to_key(x) or x for x in g.get("nice", {}).get("langs", []) + g["must"].get("langs", [])}
            for field, pred, gold in (("must tools", must, g_must), ("all tools", every, g_all), ("languages", langs, g_langs)):
                c[field][0] += len(pred & gold)
                c[field][1] += len(pred - gold)
                c[field][2] += len(gold - pred)
            exact["min years"] += r["min_years"] == g["must"].get("years")
            exact["depth"] += r["depth_level"] == g["must"].get("depth")
            exact["istqb"] += ("istqb" in r["must_have"] + r["nice_to_have"]) == bool(g.get("nice", {}).get("istqb"))
            fl = set(r.get("foreign_languages", []))
            exact["japanese"] += ("japanese" in fl) == bool(g.get("nice", {}).get("japanese") or g["must"].get("japanese"))
            exact["n"] += 1
        out[mode] = {"prf": {k: prf(*v) for k, v in c.items()}, "exact": dict(exact)}
    return out


def _plain_rows(pl: dict) -> list[str]:
    if not pl.get("n"):
        return []
    n = pl["n"]
    return [f"Vietnamese CVs typed without diacritics ({n} clean vi CVs, `strip_accents`): automation depth "
            f"{pl['depth_accented'] / n:.1%} accented → **{pl['depth_plain'] / n:.1%}** without diacritics; years within 1 "
            f"year {pl['years_accented']}/{n} → {pl['years_plain']}/{n}; same tool set {pl['tools_same']}/{n}.", ""]


def render(cv: dict, jd: dict) -> str:
    f = lambda t: f"{t:.3f}"  # noqa: E731
    lines = ["# Extraction and redaction results (generated by `python -m app.eval.extraction`; do not edit by hand)", "",
             f"Gold: the `coding` branch generator's latent facts for {cv['n']} synthetic CVs and its `gold_spec` for "
             "its JDs. Our pipeline was first measured on this set in v3.1 and the defects it exposed were fixed, so these "
             "numbers are optimistic for this set. Tools are compared as taxonomy keys over the "
             "generator's vocabulary (a closed-vocabulary evaluation).", "",
             "## CV → facts (deterministic NLP pipeline, no LLM)", "",
             "Garbled CVs are left to the investigator (N12) by design; its row is an upper bound: every near-miss line "
             "its search finds is proposed (the LLM's job) and kept only if `investigate.verify` accepts it.", "",
             "| Slice | tool precision | tool recall | tool F1 | TP / FP / FN |", "|---|---|---|---|---|"]
    for sl in ("all", "en", "vi", "clean", "low_quality", "low_quality + N12 search (upper bound)"):
        if sl in cv["counts"]:
            tp, fp, fn = cv["counts"][sl]
            p, r, f1 = prf(tp, fp, fn)
            lines.append(f"| {sl} | {f(p)} | {f(r)} | {f(f1)} | {tp} / {fp} / {fn} |")
    lines += ["", "| Fact | result |", "|---|---|",
              f"| years of experience (dated roles) | MAE {cv['years_mae']:.2f} y over {cv['years_n']} CVs with dates; "
              f"{cv['years_within_1']}/{cv['years_n']} within 1 year |",
              f"| automation depth L0–L4 (verb cue) | accuracy {cv['depth_acc']:.1%} |",
              f"| ISTQB present | accuracy {cv['istqb_acc']:.1%} |",
              f"| foreign languages (English, Japanese) | exact set {cv['foreign_acc']:.1%} |",
              f"| quality flag (X10) raised | {cv['flagged'].get('low_quality', 0)}/{cv['flagged'].get('n_low', 0)} "
              f"low-quality CVs, {cv['flagged'].get('clean', 0)}/{cv['flagged'].get('n_clean', 0)} clean CVs |", "",
              *_plain_rows(cv.get("plain", {})),
              "Gazetteer gaps (generator tools our taxonomy has no key for, with CV counts): "
              + (", ".join(f"{k} ({v})" for k, v in cv["gaps"].items()) or "none"), "",
              "## Redaction (masking before anything is indexed or shown to an LLM)", "",
              "| Field | CVs containing it | still visible after redaction | leak rate |", "|---|---|---|---|"]
    for field, n in cv["fields"].items():
        lk = cv["leaks"].get(field, 0)
        lines.append(f"| {field} | {n} | {lk} | {lk / n:.1%} |")
    lines += ["", "## JD → requirements (three parsers, Settings.jd_parser)", "",
              "| Parser | must tools P / R / F1 | all tools P / R / F1 | languages P / R / F1 | min years | depth | ISTQB | Japanese |",
              "|---|---|---|---|---|---|---|---|"]
    for mode, v in jd.items():
        e, prfs = v["exact"], v["prf"]
        if not e:
            continue
        cell = lambda k: " / ".join(map(f, prfs.get(k, (0, 0, 0))))  # noqa: E731
        lines.append(f"| {mode} | {cell('must tools')} | {cell('all tools')} | {cell('languages')} | "
                     + " | ".join(f"{e[k]}/{e['n']}" for k in ("min years", "depth", "istqb", "japanese")) + " |")
    lines += ["", "`gold_spec` is the coding branch's own reading of each JD (for example it files \"Java or Python\" under "
              "nice-to-have where the JD lists it under Requirements), so JD numbers measure agreement with that "
              "reading, not absolute correctness."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = render(cv_extraction(), jd_extraction())
    OUT.write_text(text, encoding="utf-8")
    print(text if "-q" not in sys.argv else f"→ {OUT}")
