"""Benchmarks the ladder runs on.

    ours    data/: 50 hand-written CVs, 12 JDs, silver labels from 2 independent annotators (docs/EVAL.md)
    coding  the `coding` branch's synthetic set (Đán): 200 template CVs, mostly Vietnamese, 6 JDs, gold from its own
            annotation rubric. Our code was never developed on it, so it is the closer thing to a held-out test.
    djinni  public data (lecturer feedback 3): the Djinni Recruitment Dataset (Drushchak & Romanyshyn, UNLP 2024, MIT),
            8 English "QA Automation" JDs × 250 anonymised real profiles, TREC-style pooled pairs (424) graded 0–3 by
            two human annotators in labeling/labels_{A,B}.xlsx. Every JD is test: nothing is tuned on it. Metrics are
            judged-only (each JD ranks its labelled pairs), as pooling requires.

The coding and djinni sets are copied from git into .runtime/bench/<name> (`python -m app.eval.bench fetch [djinni]`);
djinni's real CVs stay out of data/ (repo rule). Its CVs carry fake
names, phones and birth dates to exercise redaction, and the repo rule keeps PII-shaped data out of data/."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import ROOT, get_settings
from app.rag.corpus import Doc, load_dir

CODING_DIR = get_settings().runtime_dir / "bench" / "coding"
CODING_PROFILE = {"banking": "bank", "outsourcing": "outsourcing", "startup": "startup"}
OUR_TRAPS = frozenset({"CV-004", "CV-005", "CV-006", "CV-034", "CV-043", "CV-044", "CV-045", "CV-046"})


@dataclass
class Bench:
    name: str
    cvs: list[Doc]
    jds: list[Doc]
    gold: dict[str, dict[str, float]]
    traps: frozenset[str]


def load(name: str = "ours") -> Bench:
    if name == "ours":
        from app.eval.run import gold
        return Bench("ours", list(load_dir("cvs")), list(load_dir("jds")), gold(), OUR_TRAPS)
    if name == "djinni":
        return load_djinni()
    if not (CODING_DIR / "gold.json").exists():
        raise SystemExit("coding benchmark missing: run `python -m app.eval.bench fetch` first")
    g = json.loads((CODING_DIR / "gold.json").read_text())
    labelled = set.intersection(*(set(v["relevance"]) for v in g.values()))
    cvs = [Doc(p.stem, p.read_text(encoding="utf-8"), {"headline": ""})
           for p in sorted((CODING_DIR / "cvs").glob("*.txt")) if p.stem in labelled]
    jds = [Doc(j["id"], j["text"], {"title": j.get("title", ""),
                                    "company_profile": CODING_PROFILE.get(j.get("company_type"), "default")})
           for j in yaml.safe_load((CODING_DIR / "jds.yaml").read_text()) if j["id"] in g]
    return Bench("coding", cvs, jds, {j: {c: float(x) for c, x in v["relevance"].items()} for j, v in g.items()},
                 frozenset())


CORPUS_DIR = get_settings().runtime_dir / "bench" / "coding_corpus"
DJINNI_DIR = get_settings().runtime_dir / "bench" / "djinni"
ANNOTATORS = ("A", "B")
# The coding branch scripts HR's clarification answers as option indices (cv_agent/criteria/clarify.py); our agent
# takes words, so each index is written out with the same meaning.
HR_OPTIONS = {
    "seniority_years": ["At least 3 years of experience is required.", "At least 5 years of experience is required.",
                        "At least 7 years of experience is required.", "Years of experience are not required."],
    "junior_vs_years": ["Lower it to at least 1 year of experience.", "Keep the stated years.",
                        "Drop the years requirement."],
    "english_level": ["English for reading documents is a plus, not required.",
                      "English for client communication is required.", "English is not needed."],
    "tool_family": ["Must-have: one of Selenium, Playwright or Cypress (web automation).",
                    "Must-have: one of Appium, Espresso or XCUITest (mobile).",
                    "Must-have: Postman or Rest Assured (API).", "No specific tool is required."],
    "tools_all_or_any": ["One of the listed tools is enough.", "All listed tools are required.",
                         "Treat the listed tools as nice-to-have."],
}


def read_labels(path: Path) -> dict[tuple[str, str], int]:
    """One annotator's sheet "Nhãn" (columns jd_id, cv_id, "Nhãn (0–3)") → {(jd, cv): grade}; blank rows are skipped."""
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("reading the label sheets needs openpyxl: `uv run --with openpyxl python -m ...`") from None
    rows = list(openpyxl.load_workbook(path, read_only=True, data_only=True)["Nhãn"].iter_rows(values_only=True))
    head = [str(h or "") for h in rows[0]]
    jd, cv, lab = head.index("jd_id"), head.index("cv_id"), next(i for i, h in enumerate(head) if h.startswith("Nhãn"))
    return {(r[jd], r[cv]): grade(r[lab]) for r in rows[1:] if r[jd] and grade(r[lab]) is not None}


def grade(x) -> int | None:
    """A 0–3 grade from a sheet cell ("2", 2, "2 - phù hợp"), else None."""
    try:
        g = int(str(x).strip()[:1])
    except (ValueError, TypeError):
        return None
    return g if 0 <= g <= 3 else None


def djinni_labels(d: Path = DJINNI_DIR) -> dict[str, dict[tuple[str, str], int]]:
    return {a: read_labels(p) for a in ANNOTATORS if (p := d / "labeling" / f"labels_{a}.xlsx").exists()}


def agreement(labels: dict[str, dict[tuple[str, str], int]]) -> dict:
    """Pairs both annotators graded → exact agreement and quadratic-weighted κ (EVAL.md decision rule: ≥ 0.6)."""
    from app.eval.metrics import weighted_kappa

    if len(labels) < 2:
        return {"pairs": 0}
    a, b = (labels[x] for x in ANNOTATORS)
    both = sorted(set(a) & set(b))
    if not both:
        return {"pairs": 0}
    xa, xb = [a[k] for k in both], [b[k] for k in both]
    return {"pairs": len(both), "exact": sum(x == y for x, y in zip(xa, xb)) / len(both), "kappa": weighted_kappa(xa, xb)}


def djinni_gold(labels: dict[str, dict[tuple[str, str], int]]) -> dict[str, dict[str, float]]:
    """Gold = the mean grade of the annotators who graded the pair. Unjudged pairs are absent (judged-only metrics)."""
    sums: dict[tuple[str, str], list[int]] = {}
    for lab in labels.values():
        for k, g in lab.items():
            sums.setdefault(k, []).append(g)
    gold: dict[str, dict[str, float]] = {}
    for (jd, cv), gs in sums.items():
        gold.setdefault(jd, {})[cv] = sum(gs) / len(gs)
    return gold


def load_djinni() -> Bench:
    if not (DJINNI_DIR / "jds.yaml").exists():
        raise SystemExit("djinni benchmark missing: run `python -m app.eval.bench fetch djinni` first")
    gold = djinni_gold(djinni_labels())
    if not gold:
        raise SystemExit(f"djinni has no labels yet: fill {DJINNI_DIR}/labeling/labels_A.xlsx and labels_B.xlsx "
                         "(two annotators, independently; see the sheet 'Hướng dẫn')")
    cvs = [Doc(p.stem, p.read_text(encoding="utf-8"), {"headline": ""}) for p in sorted((DJINNI_DIR / "cvs").glob("*.txt"))
           if any(p.stem in v for v in gold.values())]
    jds = [Doc(j["id"], j["text"], {"title": j.get("title", ""),
                                    "company_profile": CODING_PROFILE.get(j.get("company_type"), "default")})
           for j in yaml.safe_load((DJINNI_DIR / "jds.yaml").read_text()) if j["id"] in gold]
    return Bench("djinni", cvs, jds, gold, frozenset())


def corpus_dir(name: str) -> Path:
    return CORPUS_DIR if name == "coding" else get_settings().runtime_dir / "bench" / f"{name}_corpus"


def materialize(name: str = "coding") -> Path:
    """Lay a benchmark out like data/ so the whole agent can run on it: DATA_DIR=<this dir>."""
    import shutil

    b = load(name)
    out = corpus_dir(name)
    meta = {j["id"]: j for j in yaml.safe_load((CODING_DIR / "jds.yaml").read_text())} if name == "coding" else {}
    for sub in ("cvs", "jds"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for d in b.cvs:
        (out / "cvs" / f"{d.id}.md").write_text(f"---\nid: {d.id}\nheadline: ''\n---\n{d.text}\n", encoding="utf-8")
    for d in b.jds:
        answers = " ".join(HR_OPTIONS[q][i] for q, i in (meta.get(d.id, {}).get("hr_answers") or {}).items() if q in HR_OPTIONS)
        front = {"id": d.id, "title": d.meta["title"], "company_profile": d.meta["company_profile"]}
        if answers:
            front["hr_answer"] = answers
        (out / "jds" / f"{d.id}.md").write_text(
            f"---\n{yaml.safe_dump(front, allow_unicode=True)}---\n{d.text}\n", encoding="utf-8")
    data = ROOT / "data"
    for f in ("criteria_profiles.yaml", "skills_taxonomy.yaml"):
        shutil.copy(data / f, out / f)
    shutil.copytree(data / "policies", out / "policies", dirs_exist_ok=True)
    return out


def git(*a: str) -> bytes:
    return subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True).stdout


def fetch_djinni(ref: str = "origin/coding") -> None:
    """Copy data/djinni (JDs, 250 profiles, label sheets, licence) into .runtime/bench/djinni. Label sheets already
    there are kept: they may hold this machine's annotations; a newer copy from the branch goes next to them."""
    for f in filter(None, git("ls-tree", "-r", "--name-only", ref, "data/djinni").decode().split("\n")):
        out = DJINNI_DIR / Path(f).relative_to("data/djinni")
        if out.suffix == ".xlsx" and out.exists():
            out = out.with_name(out.stem + ".branch.xlsx")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(git("show", f"{ref}:{f}"))
    print(f"djinni benchmark → {DJINNI_DIR} ({len(list((DJINNI_DIR / 'cvs').glob('*.txt')))} profiles)")


def fetch(ref: str = "origin/coding") -> None:
    """Copy the coding branch's synthetic CVs, JDs and gold into .runtime/bench/coding (read-only on that branch)."""
    files = git("ls-tree", "-r", "--name-only", ref, "data/synthetic", "data/jds/jds.yaml").decode().split("\n")
    for f in filter(None, files):
        if f.endswith((".txt", "gold.json", "latent.json", "jds.yaml")):
            out = CODING_DIR / ("cvs/" + Path(f).name if f.endswith(".txt") else Path(f).name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(git("show", f"{ref}:{f}"))
    print(f"coding benchmark → {CODING_DIR} ({len(list((CODING_DIR / 'cvs').glob('*.txt')))} CVs)")


if __name__ == "__main__":
    if sys.argv[1:3] == ["fetch", "djinni"]:
        fetch_djinni(*sys.argv[3:4])
    elif sys.argv[1:2] == ["fetch"]:
        fetch(*sys.argv[2:3])
    elif sys.argv[1:2] == ["materialize"]:
        print(f"DATA_DIR={materialize(*sys.argv[2:3])}")
    elif sys.argv[1:2] == ["agreement"]:
        print(agreement(djinni_labels()))
    else:
        print(__doc__)
