"""Djinni: a real-data layer (anonymised public CVs and JDs, MIT; Drushchak & Romanyshyn, UNLP 2024).

8 English "QA Automation" JDs × 250 real CVs, prepared on the `coding` branch (data/djinni, pinned below). Real CVs stay
out of data/ (repo rule), so they are copied from git into .runtime/bench/djinni, like the coding benchmark. Only
labels are committed (data/eval/djinni_*.json).

Labels follow TREC pooling: per JD, the union of four rankers' top 20 plus 10 random CVs, 424 pairs in all
(labeling/pool.json). Metrics are condensed: each JD's ranking is scored over its judged CVs only.

    python -m app.eval.djinni fetch                          copy the pinned set into .runtime/bench/djinni
    python -m app.eval.djinni silver --tag a --model claude_code:claude-opus-5-5     LLM silver labels (not human)
    python -m app.eval.djinni silver --tag b --model claude_code:claude-sonnet-5-5
    python -m app.eval.djinni import labels_A.xlsx labels_B.xlsx [--adjudicated FILE.xlsx]
                                                             team's xlsx → data/eval/human_djinni_*.json
    python -m app.eval.djinni score                          → docs/DJINNI_LABELS.md (κ human–human, human–silver)
    python -m app.eval.djinni review                         → .runtime/bench/djinni/labeling/review.html: label in a
                                                             browser, blind or with Opus's grade shown; its exported
                                                             JSON goes to `import` like the xlsx files;
                                                             ?blind=1 locks blind mode, &subset=1 = 80 pairs (subset())

Gold (`gold()`): the adjudicated human labels once two annotators who pass the quality check (`qc_problems`) finished every pair and no pair is left ≥ 2 grades
apart; until then the mean of the two silver annotators. Which one is in use is stated in every report.
"""

import argparse
import asyncio
import hashlib
import html
import json
import random
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

from app.config import ROOT, get_settings
from app.eval import metrics
from app.rag.corpus import Doc

REF = "a0a4c1a8aa205cea97e0d343f85ce440ee7ea6b5"  # origin/coding: "Djinni gold set for human labelling"
DIR = get_settings().runtime_dir / "bench" / "djinni"
LABELS = ROOT / "data" / "eval"
DOC = ROOT / "docs" / "DJINNI_LABELS.md"
BATCH = 20  # CVs per silver-label call
QC_MIN_KAPPA = 0.4  # an annotator must reach "moderate" weighted κ with at least one other annotator (Landis & Koch)
SUBSET_PER_JD = 10  # ?subset=1 on the page: 80 pairs, enough for a human–human κ
MIN_READ_SECONDS = 20  # ?blind=1: grading unlocks this long after a new pair opens
QC_MIN_SECONDS = 10.0  # blind labels: median seconds between grades; below this nobody read a JD and a CV


# ── data ──────────────────────────────────────────────────────────────────────────────────────────────
def fetch(ref: str = REF) -> None:
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True).stdout  # noqa: E731
    files = git("ls-tree", "-r", "--name-only", ref, "data/djinni").decode().split("\n")
    for f in filter(None, files):
        if f.endswith((".txt", "jds.yaml", "meta.json", "pool.json", ".xlsx")):
            out = DIR / Path(f).relative_to("data/djinni")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(git("show", f"{ref}:{f}"))
    print(f"djinni → {DIR} ({len(list((DIR / 'cvs').glob('*.txt')))} CVs)")


def _need() -> None:
    if not (DIR / "jds.yaml").exists():
        raise SystemExit("djinni set missing: run `python -m app.eval.djinni fetch` first")


def cvs() -> list[Doc]:
    _need()
    return [Doc(p.stem, p.read_text(encoding="utf-8"), {"headline": p.read_text(encoding="utf-8").split("\n", 1)[0]})
            for p in sorted((DIR / "cvs").glob("*.txt"))]


def jds() -> list[Doc]:
    _need()
    return [Doc(j["id"], j["text"], {"title": j.get("title", ""), "company_profile": "default"})
            for j in yaml.safe_load((DIR / "jds.yaml").read_text(encoding="utf-8"))]


def pairs() -> dict[str, list[str]]:
    _need()
    return json.loads((DIR / "labeling" / "pool.json").read_text())["pool"]


# ── labels ────────────────────────────────────────────────────────────────────────────────────────────
def _load(path: Path) -> dict[tuple[str, str], int]:
    return {(x["jd"], x["cv"]): int(x["grade"]) for x in json.loads(path.read_text(encoding="utf-8"))["labels"]}


def silver_files() -> list[Path]:
    return sorted(LABELS.glob("djinni_silver_*.json"))


def human_files() -> list[Path]:
    """One file per human annotator. The adjudicated set and the aggregate gold (app.eval.djinni_gold) are derived
    from these, not annotators."""
    return sorted(p for p in LABELS.glob("human_djinni_*.json")
                  if p.stem not in ("human_djinni_adjudicated", "human_djinni_gold"))


def median_seconds(labels: list[dict]) -> float | None:
    """Median gap between consecutive grades (the page stamps `at` on every click); None without timestamps."""
    from datetime import datetime
    from statistics import median

    ts = sorted(datetime.fromisoformat(x["at"].replace("Z", "+00:00")) for x in labels if x.get("at"))
    return median((b - a).total_seconds() for a, b in zip(ts, ts[1:])) if len(ts) > 2 else None


def qc_problems(labels: list[dict], others: dict[str, dict[tuple[str, str], int]]) -> list[str]:
    """Why an annotator's labels cannot be gold ([] = usable). κ floor for everyone; the speed floor only for blind
    labels (accepting a shown model grade is fast, and review mode is already reported as anchored on the model)."""
    mine = {(x["jd"], x["cv"]): int(x["grade"]) for x in labels}
    problems, best = [], None
    for other in others.values():
        common = sorted(set(mine) & set(other))
        if len(common) >= 20:
            k = metrics.weighted_kappa([mine[c] for c in common], [other[c] for c in common])
            best = k if best is None else max(best, k)
    if best is not None and best < QC_MIN_KAPPA:
        problems.append(f"best κ with any other annotator {best:.2f} < {QC_MIN_KAPPA}")
    secs = median_seconds(labels)
    blind = not any(x.get("saw_model") for x in labels)
    if blind and secs is not None and secs < QC_MIN_SECONDS:
        problems.append(f"median {secs:.1f} s per pair < {QC_MIN_SECONDS:.0f} s")
    return problems


def _raw(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["labels"]


def qc_report() -> dict[Path, list[str]]:
    """Every human label file → its QC problems, judged against all other annotators (human, silver, Opus)."""
    named = {p: _load(p) for p in human_files() + silver_files() + ([MODEL_GOLD] if MODEL_GOLD.exists() else [])}
    return {p: qc_problems(_raw(p), {q: v for q, v in named.items() if q != p}) for p in human_files()}


def qualified_humans() -> list[Path]:
    return [p for p, problems in qc_report().items() if not problems]


def adjudicate(a: dict, b: dict, override: dict | None = None) -> tuple[dict, list[tuple[str, str]]]:
    """Coding-branch rule: equal → keep; 1 apart → the lower grade; ≥ 2 apart → a conflict, unless `override` (the
    pair both annotators settled together) gives the grade. → (gold, unresolved conflicts)."""
    override, out, conflicts = override or {}, {}, []
    for k in sorted(set(a) & set(b)):
        if k in override:
            out[k] = override[k]
        elif abs(a[k] - b[k]) <= 1:
            out[k] = min(a[k], b[k])
        else:
            conflicts.append(k)
    return out, conflicts


def gold() -> tuple[dict[str, dict[str, float]], str]:
    """→ (JD → CV → grade over the judged pairs, which labels these are)."""
    want = {(j, c) for j, cs in pairs().items() for c in cs}
    hum = [_load(p) for p in qualified_humans()]
    adj = LABELS / "human_djinni_adjudicated.json"
    if len(hum) >= 2 and all(want <= set(h) for h in hum[:2]):
        g, conflicts = adjudicate(hum[0], hum[1], _load(adj) if adj.exists() else None)
        if not conflicts:
            return _nest({k: float(v) for k, v in g.items()}), "human (2 annotators, adjudicated)"
    sil = [_load(p) for p in silver_files()]
    if not sil:
        raise SystemExit("no djinni labels yet: run `python -m app.eval.djinni silver` first")
    mean = {k: sum(s[k] for s in sil) / len(sil) for k in want if all(k in s for s in sil)}
    return _nest(mean), f"silver (mean of {len(sil)} LLM annotators; not human)"


def human_gold() -> tuple[dict[str, dict[str, float]], str] | None:
    """The best complete human label set, or None: two annotators adjudicated, else one annotator who labelled every
    pair. The description says whether that person saw the model's grade (review mode anchors on it)."""
    want = {(j, c) for j, cs in pairs().items() for c in cs}
    g, source = gold()
    if source.startswith("human"):
        return g, source
    for p in qualified_humans():
        d = json.loads(p.read_text(encoding="utf-8"))
        labels = {(x["jd"], x["cv"]): x for x in d["labels"]}
        if want <= set(labels):
            seen = sum(bool(labels[k].get("saw_model")) for k in want)
            how = "blind" if not seen else f"reviewing Opus 5.5's grade on {seen}/{len(want)} pairs"
            return _nest({k: float(labels[k]["grade"]) for k in want}), f"human (1 annotator, {how})"
    return None


def _nest(flat: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for (j, c), v in flat.items():
        out.setdefault(j, {})[c] = v
    return out


# ── the team's xlsx ───────────────────────────────────────────────────────────────────────────────────
def _col(ref: str) -> str:
    return re.match(r"[A-Z]+", ref).group(0)


def read_xlsx(path: Path, sheet: str = "Nhãn") -> list[dict[str, str]]:
    """Rows of one sheet as {header: value}, stdlib only. Handles inline strings (as generated) and shared strings
    (what Excel writes on save)."""
    z = zipfile.ZipFile(path)
    wb = z.read("xl/workbook.xml").decode("utf-8")
    rid = re.search(rf'<sheet [^>]*name="{re.escape(html.escape(sheet))}"[^>]*r:id="(\w+)"', wb)
    if not rid:
        raise ValueError(f"{path.name}: no sheet {sheet!r}")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    target = re.search(rf'<Relationship [^>]*Id="{rid.group(1)}"[^>]*Target="([^"]+)"', rels) or \
        re.search(rf'<Relationship [^>]*Target="([^"]+)"[^>]*Id="{rid.group(1)}"', rels)
    name = target.group(1).lstrip("/")
    name = name if name.startswith("xl/") else f"xl/{name}"
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = z.read("xl/sharedStrings.xml").decode("utf-8")
        shared = [html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))
                  for si in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    rows = []
    for row in re.findall(r"<row[^>]*>(.*?)</row>", z.read(name).decode("utf-8"), re.S):
        cells = {}
        for attrs, body in re.findall(r"<c ([^>]*?)(?:/>|>(.*?)</c>)", row, re.S):
            ref, kind = re.search(r'r="(\w+)"', attrs).group(1), re.search(r't="(\w+)"', attrs)
            v = re.search(r"<v>(.*?)</v>", body or "", re.S)
            if kind and kind.group(1) == "s" and v:
                val = shared[int(v.group(1))]
            elif kind and kind.group(1) == "inlineStr":
                val = html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", body, re.S)))
            else:
                val = html.unescape(v.group(1)) if v else ""
            cells[_col(ref)] = val
        rows.append(cells)
    header = rows[0] if rows else {}
    return [{header.get(c, c): v for c, v in r.items()} for r in rows[1:]]


def labels_from_xlsx(path: Path) -> list[dict]:
    out = []
    for r in read_xlsx(path):
        grade = str(r.get("Nhãn (0–3)", "")).strip()
        if grade in ("", "None"):
            continue
        g = int(float(grade))
        if not 0 <= g <= 3:
            raise ValueError(f"{path.name}: grade {grade!r} for {r.get('jd_id')} / {r.get('cv_id')} is not 0–3")
        out.append({"jd": r["jd_id"], "cv": r["cv_id"], "grade": g, "note": r.get("Ghi chú", "") or ""})
    return out


REVIEW_PAGE = Path(__file__).with_name("djinni_review.html")
MODEL_GOLD = LABELS / "djinni_gold_opus.json"  # Opus 5.5, 8 CVs a call: the suggestion the review page shows


def subset(per_jd: int = SUBSET_PER_JD) -> list[tuple[str, str]]:
    """A stratified sample for a careful blind read: per JD, pairs ordered by the silver mean (ties by a hash) and
    taken at even steps, so every grade level is in it without the annotator being told which is which."""
    sil = [_load(p) for p in silver_files()]
    out = []
    for jd, cv_ids in sorted(pairs().items()):
        mean = lambda c: sum(s.get((jd, c), 0) for s in sil) / max(len(sil), 1)  # noqa: E731
        ranked = sorted(cv_ids, key=lambda c: (mean(c), hashlib.sha1(f"{jd}|{c}".encode()).hexdigest()))
        k = min(per_jd, len(ranked))
        steps = sorted({round(i * (len(ranked) - 1) / max(k - 1, 1)) for i in range(k)})
        out += [(jd, ranked[i]) for i in steps]
    return out


def review(out: Path | None = None) -> Path:
    """One self-contained page to label every pooled pair: blind (no model grade shown) or review (Opus's grade and
    evidence shown). Real CVs are inside, so it is written under .runtime, never to docs/ or data/."""
    out = out or DIR / "labeling" / "review.html"
    rnd, flat = random.Random(13), [{"jd": j, "cv": c} for j, cs in sorted(pairs().items()) for c in cs]
    rnd.shuffle(flat)  # no JD block order and no system order to anchor on
    model = ({f"{x['jd']}|{x['cv']}": {"grade": x["grade"], "note": x.get("note", "")}
              for x in json.loads(MODEL_GOLD.read_text(encoding="utf-8"))["labels"]} if MODEL_GOLD.exists() else {})
    from scripts.llm_label import RUBRIC

    data = {"pool": REF, "pairs": flat, "rubric": RUBRIC, "model": model, "min_read": MIN_READ_SECONDS,
            "subset": [f"{j}|{c}" for j, c in subset()],
            "jds": {d.id: {"title": d.meta["title"], "text": d.text} for d in jds()},
            "cvs": {d.id: d.text for d in cvs() if any(p["cv"] == d.id for p in flat)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # a CV can never close the <script>
    out.write_text(REVIEW_PAGE.read_text(encoding="utf-8").replace("__DATA__", blob), encoding="utf-8")
    return out


def labels_from_page(path: Path) -> tuple[str, list[dict]]:
    """→ (annotator, labels) from the JSON the review page exports; grades checked like the xlsx path."""
    d = json.loads(path.read_text(encoding="utf-8"))
    for x in d["labels"]:
        if int(x["grade"]) not in range(4):
            raise ValueError(f"{path.name}: grade {x['grade']!r} for {x['jd']} / {x['cv']} is not 0–3")
    return d["annotator"], [{"jd": x["jd"], "cv": x["cv"], "grade": int(x["grade"]), "note": x.get("note", ""),
                             "saw_model": bool(x.get("saw_model")), **({"at": x["at"]} if x.get("at") else {})}
                            for x in d["labels"]]


def slug(name: str) -> str:
    """A file-safe ASCII name: "Đán" → "dan" (no diacritics in committed file names)."""
    from app.nlp.text import fold

    return re.sub(r"[^a-z0-9]+", "_", fold(name)).strip("_")


def import_labels(files: list[Path], adjudicated: Path | None = None) -> None:
    want = {(j, c) for j, cs in pairs().items() for c in cs}
    for f in files:
        if f.suffix.lower() == ".json":  # from the review page
            annotator, labels = labels_from_page(f)
            who = slug(annotator.removeprefix("human ")) or f.stem
            seen = sum(x["saw_model"] for x in labels)
            (LABELS / f"human_djinni_{who}.json").write_text(json.dumps(
                {"annotator": f"human {who}" + (f" (saw the model's grade on {seen} pairs)" if seen else ""),
                 "source": f.name, "labels": labels}, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{f.name}: {len(labels)}/{len(want)} pairs labelled ({seen} in review mode) → human_djinni_{who}.json")
            continue
        labels = labels_from_xlsx(f)
        who = slug(f.stem.removeprefix("labels_")) or f.stem
        (LABELS / f"human_djinni_{who}.json").write_text(json.dumps(
            {"annotator": f"human {who}", "source": f.name, "labels": labels}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print(f"{f.name}: {len(labels)}/{len(want)} pairs labelled → human_djinni_{who}.json")
    if adjudicated:
        labels = labels_from_xlsx(adjudicated)
        (LABELS / "human_djinni_adjudicated.json").write_text(json.dumps(
            {"annotator": "adjudicated", "source": adjudicated.name, "labels": labels}, indent=1, ensure_ascii=False),
            encoding="utf-8")
        print(f"{adjudicated.name}: {len(labels)} settled pairs")


# ── silver labels (an LLM, labelled as such) ──────────────────────────────────────────────────────────
async def silver(spec: str, tag: str, out: Path | None = None, batch: int = BATCH) -> None:
    """The same blind rubric as scripts/llm_label.py; resumable (pairs already graded are kept). `out` writes
    elsewhere than djinni_silver_<tag>.json, so an extra model annotator never changes the frozen v2 silver mean."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.nlp.injection import neutralize
    from app.providers.llm import create_llm
    from scripts.llm_label import RUBRIC, Grades

    out_path = out or LABELS / f"djinni_silver_{tag}.json"
    done = json.loads(out_path.read_text(encoding="utf-8"))["labels"] if out_path.exists() else []
    have = {(x["jd"], x["cv"]) for x in done}
    text, jd_text = {d.id: d.text for d in cvs()}, {d.id: d.text for d in jds()}
    model, rnd = create_llm(spec).with_structured_output(Grades), random.Random(7)
    for jd, cv_ids in sorted(pairs().items()):
        todo = rnd.sample([c for c in cv_ids if (jd, c) not in have], len([c for c in cv_ids if (jd, c) not in have]))
        for i in range(0, len(todo), batch):
            shown = todo[i:i + batch]
            body = "\n\n".join(f"=== {c} ===\n{neutralize(text[c])[0]}" for c in shown)
            got: Grades = await model.ainvoke([SystemMessage(RUBRIC), HumanMessage(
                f"Job description:\n{jd_text[jd]}\n\nGrade each of these {len(shown)} CVs:\n\n{body}")])
            by = {g.cv.strip(): g for g in got.grades if g.cv.strip() in shown}
            done += [{"jd": jd, "cv": c, "grade": by[c].grade, "note": by[c].note} for c in shown if c in by]
            out_path.write_text(json.dumps({"annotator": f"silver {tag} ({spec.split(':', 1)[-1]}, LLM, blind, rubric; "
                                            "not human)", "labels": done}, indent=1, ensure_ascii=False),
                                encoding="utf-8")
            print(f"{jd}: graded {len(by)}/{len(shown)}", flush=True)
    print(f"→ {out_path} ({len(done)} pairs)")


# ── report ────────────────────────────────────────────────────────────────────────────────────────────
def kappa_rows(named: dict[str, dict[tuple[str, str], int]]) -> list[str]:
    rows, names = [], list(named)
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            common = sorted(set(named[n1]) & set(named[n2]))
            if not common:
                continue
            xs, ys = [named[n1][k] for k in common], [named[n2][k] for k in common]
            rows.append(f"| {n1} vs {n2} | {len(common)} | {sum(x == y for x, y in zip(xs, ys)) / len(common):.1%} | "
                        f"{metrics.weighted_kappa(xs, ys):.3f} |")
    return rows


def score() -> str:
    want = {(j, c) for j, cs in pairs().items() for c in cs}
    humans = {json.loads(p.read_text(encoding="utf-8"))["annotator"]: _load(p) for p in human_files()}
    silvers = {json.loads(p.read_text(encoding="utf-8"))["annotator"]: _load(p) for p in silver_files()}
    if len(silvers) >= 2:
        s = list(silvers.values())
        silvers["silver mean (rounded down)"] = {k: int(sum(x[k] for x in s) / len(s)) for k in set.intersection(
            *(set(x) for x in s))}
    if MODEL_GOLD.exists():  # a model annotator too: reported, never mixed into the silver mean or the gold
        silvers["Opus 5.5 close read (8 CVs a call; LLM, not human)"] = _load(MODEL_GOLD)
    _, source = gold()
    lines = ["# Djinni: human labels vs silver (generated by `python -m app.eval.djinni score`; do not edit)", "",
             f"Real anonymised CVs (Djinni, MIT), {len(pairs())} JDs, {len(want)} pooled pairs. Gold in use: **{source}**.",
             "Quadratic-weighted Cohen's κ on the pairs both sides graded. Decision rule (EVAL.md §2): κ(human, silver) "
             "≥ 0.8 keeps silver as the main set; ≥ 0.6 as a secondary set; below that the human labels replace it.", "",
             "## Progress", "", "| Annotator | pairs labelled |", "|---|---|"]
    lines += [f"| {n} | {len(set(x) & want)}/{len(want)} |" for n, x in {**humans, **silvers}.items()]
    if not humans:
        lines += ["", "No human labels yet: fill `labels_A.xlsx` / `labels_B.xlsx` (team members who did not write "
                  "the criteria) and run `python -m app.eval.djinni import`."]
    if humans:
        lines += ["", "## Annotator quality check", "",
                  f"Gold uses only annotators who pass: weighted κ ≥ {QC_MIN_KAPPA} with at least one other annotator, "
                  f"and, for blind labels, a median of ≥ {QC_MIN_SECONDS:.0f} s per pair (the page stamps every grade). "
                  "Review-mode labels are fast by nature (accepting a shown grade) and count as anchored on the model.",
                  "", "| Annotator | mode | median s / pair | notes written | verdict |", "|---|---|---|---|---|"]
        for p, problems in qc_report().items():
            raw = _raw(p)
            secs = median_seconds(raw)
            mode = "review (saw Opus's grade)" if any(x.get("saw_model") for x in raw) else "blind"
            lines.append(f"| {json.loads(p.read_text(encoding='utf-8'))['annotator']} | {mode} | "
                         f"{'n/a' if secs is None else f'{secs:.1f}'} | {sum(bool(x.get('note')) for x in raw)} | "
                         f"{'✅ usable' if not problems else '❌ excluded: ' + '; '.join(problems)} |")
    good = [_load(p) for p in qualified_humans()]
    if len(good) >= 2:
        adj = LABELS / "human_djinni_adjudicated.json"
        _, conflicts = adjudicate(good[0], good[1], _load(adj) if adj.exists() else None)
        lines += ["", f"Pairs ≥ 2 grades apart still to settle together: **{len(conflicts)}**"]
        lines += [f"- {j} / {c}" for j, c in conflicts[:50]]
    elif len(humans) >= 2:
        lines += ["", f"Adjudicated human gold (v2.1) needs two annotators who pass the check; {len(good)} do."]
    lines += ["", "## Agreement", "", "| Pair | pairs | exact agreement | weighted κ |", "|---|---|---|---|"]
    lines += kappa_rows({**humans, **silvers}) or ["| (needs two annotators) | | | |"]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(DOC)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    s = sub.add_parser("silver")
    s.add_argument("--tag", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--out", type=Path, help="write here instead of data/eval/djinni_silver_<tag>.json")
    s.add_argument("--batch", type=int, default=BATCH, help="CVs per call (fewer = a closer read)")
    i = sub.add_parser("import")
    i.add_argument("files", nargs="+", type=Path)
    i.add_argument("--adjudicated", type=Path)
    sub.add_parser("score")
    sub.add_parser("review")
    a = ap.parse_args()
    if a.cmd == "review":
        print(f"open in a browser: {review()}")
        return
    if a.cmd == "fetch":
        fetch()
    elif a.cmd == "silver":
        asyncio.run(silver(a.model, a.tag, a.out, a.batch))
    elif a.cmd == "import":
        import_labels(a.files, a.adjudicated)
        print(score())
    else:
        print(score())


if __name__ == "__main__":
    sys.exit(main())
