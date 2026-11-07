"""Benchmarks the ladder runs on.

    ours    data/: 50 hand-written CVs, 12 JDs, silver labels from 2 independent annotators (docs/EVAL.md)
    coding  the `coding` branch's synthetic set (Đán): 200 template CVs, mostly Vietnamese, 6 JDs, gold from its own
            annotation rubric. Our code was never developed on it, so it is the closer thing to a held-out test.
    djinni  250 real anonymised profiles, 8 real JDs and 424 dual-human judgements; evaluation is judged-only.

The coding set is copied from git into .runtime/bench/coding (`python -m app.eval.bench fetch`). Its CVs carry fake
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
DJINNI_DIR = get_settings().runtime_dir / "bench" / "djinni"
DJINNI_HOLDOUT_DIR = get_settings().runtime_dir / "bench" / "djinni_holdout"
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
        gold_path = ROOT / "data" / "eval" / "human_djinni_gold.json"
        if not gold_path.exists() or not (DJINNI_DIR / "jds.yaml").exists():
            raise SystemExit("Djinni benchmark missing: need .runtime/bench/djinni and human_djinni_gold.json")
        payload = json.loads(gold_path.read_text(encoding="utf-8"))
        gold: dict[str, dict[str, float]] = {}
        for row in payload["labels"]:
            gold.setdefault(row["jd"], {})[row["cv"]] = float(row["grade"])
        cvs = [Doc(p.stem, p.read_text(encoding="utf-8"), {"headline": ""})
               for p in sorted((DJINNI_DIR / "cvs").glob("*.txt"))]
        jds = [Doc(j["id"], j["text"], {"title": j.get("title", ""), "company_profile": "default"})
               for j in yaml.safe_load((DJINNI_DIR / "jds.yaml").read_text(encoding="utf-8"))]
        if {j.id for j in jds} != set(gold):
            raise ValueError("Djinni JD text and Aggregate Human Gold cover different JDs")
        return Bench("djinni", cvs, jds, gold, frozenset())
    if name == "djinni_holdout":
        from app.eval.djinni_holdout import GOLD as holdout_gold
        from app.eval.djinni_holdout import verify as verify_holdout

        if not holdout_gold.exists():
            raise SystemExit("Djinni unseen holdout is locked: two blind human label files must pass "
                             "`python -m app.eval.djinni_holdout import-labels A.json B.json` before any system run")
        bad = verify_holdout()
        if bad:
            raise SystemExit("Djinni unseen holdout integrity failed: " + ", ".join(bad))
        payload = json.loads(holdout_gold.read_text(encoding="utf-8"))
        gold: dict[str, dict[str, float]] = {}
        for row in payload["labels"]:
            gold.setdefault(row["jd"], {})[row["cv"]] = float(row["grade"])
        cvs = [Doc(p.stem, p.read_text(encoding="utf-8"), {"headline": ""})
               for p in sorted((DJINNI_HOLDOUT_DIR / "cvs").glob("*.txt"))]
        jds = [Doc(j["id"], j["text"], {"title": j.get("title", ""), "company_profile": "default"})
               for j in yaml.safe_load((DJINNI_HOLDOUT_DIR / "jds.yaml").read_text(encoding="utf-8"))]
        return Bench("djinni_holdout", cvs, jds, gold, frozenset())
    if name != "coding":
        raise ValueError(f"unknown benchmark {name!r}")
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
DJINNI_CORPUS_DIR = get_settings().runtime_dir / "bench" / "djinni_corpus"
DJINNI_HOLDOUT_CORPUS_DIR = get_settings().runtime_dir / "bench" / "djinni_holdout_corpus"
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


def materialize() -> Path:
    """Lay the coding benchmark out like data/ so the whole agent can run on it: DATA_DIR=<this dir>."""
    import shutil

    b = load("coding")
    meta = {j["id"]: j for j in yaml.safe_load((CODING_DIR / "jds.yaml").read_text())}
    for sub in ("cvs", "jds"):
        (CORPUS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for d in b.cvs:
        (CORPUS_DIR / "cvs" / f"{d.id}.md").write_text(f"---\nid: {d.id}\nheadline: ''\n---\n{d.text}\n", encoding="utf-8")
    for d in b.jds:
        answers = " ".join(HR_OPTIONS[q][i] for q, i in (meta[d.id].get("hr_answers") or {}).items() if q in HR_OPTIONS)
        front = {"id": d.id, "title": d.meta["title"], "company_profile": d.meta["company_profile"]}
        if answers:
            front["hr_answer"] = answers
        (CORPUS_DIR / "jds" / f"{d.id}.md").write_text(
            f"---\n{yaml.safe_dump(front, allow_unicode=True)}---\n{d.text}\n", encoding="utf-8")
    data = ROOT / "data"
    for f in ("criteria_profiles.yaml", "skills_taxonomy.yaml"):
        shutil.copy(data / f, CORPUS_DIR / f)
    shutil.copytree(data / "policies", CORPUS_DIR / "policies", dirs_exist_ok=True)
    return CORPUS_DIR


def materialize_djinni() -> Path:
    """Lay the real Djinni pool out like data/ so the unchanged P6 graph can run on it."""
    import shutil

    b = load("djinni")
    for sub in ("cvs", "jds"):
        (DJINNI_CORPUS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for d in b.cvs:
        (DJINNI_CORPUS_DIR / "cvs" / f"{d.id}.md").write_text(
            f"---\nid: {d.id}\nheadline: ''\n---\n{d.text}\n", encoding="utf-8")
    for d in b.jds:
        front = {"id": d.id, "title": d.meta["title"], "company_profile": "default"}
        (DJINNI_CORPUS_DIR / "jds" / f"{d.id}.md").write_text(
            f"---\n{yaml.safe_dump(front, allow_unicode=True)}---\n{d.text}\n", encoding="utf-8")
    data = ROOT / "data"
    for name in ("criteria_profiles.yaml", "skills_taxonomy.yaml"):
        shutil.copy(data / name, DJINNI_CORPUS_DIR / name)
    shutil.copytree(data / "policies", DJINNI_CORPUS_DIR / "policies", dirs_exist_ok=True)
    return DJINNI_CORPUS_DIR


def materialize_djinni_holdout() -> Path:
    """Materialize only after the double-human Gold gate has opened."""
    import shutil

    b = load("djinni_holdout")  # fail-closed while the holdout is unseen/unlabelled
    for sub in ("cvs", "jds"):
        (DJINNI_HOLDOUT_CORPUS_DIR / sub).mkdir(parents=True, exist_ok=True)
    for d in b.cvs:
        (DJINNI_HOLDOUT_CORPUS_DIR / "cvs" / f"{d.id}.md").write_text(
            f"---\nid: {d.id}\nheadline: ''\n---\n{d.text}\n", encoding="utf-8")
    for d in b.jds:
        front = {"id": d.id, "title": d.meta["title"], "company_profile": "default"}
        (DJINNI_HOLDOUT_CORPUS_DIR / "jds" / f"{d.id}.md").write_text(
            f"---\n{yaml.safe_dump(front, allow_unicode=True)}---\n{d.text}\n", encoding="utf-8")
    data = ROOT / "data"
    for name in ("criteria_profiles.yaml", "skills_taxonomy.yaml"):
        shutil.copy(data / name, DJINNI_HOLDOUT_CORPUS_DIR / name)
    shutil.copytree(data / "policies", DJINNI_HOLDOUT_CORPUS_DIR / "policies", dirs_exist_ok=True)
    return DJINNI_HOLDOUT_CORPUS_DIR


def fetch(ref: str = "origin/coding") -> None:
    """Copy the coding branch's synthetic CVs, JDs and gold into .runtime/bench/coding (read-only on that branch)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True).stdout  # noqa: E731
    files = git("ls-tree", "-r", "--name-only", ref, "data/synthetic", "data/jds/jds.yaml").decode().split("\n")
    for f in filter(None, files):
        if f.endswith((".txt", "gold.json", "latent.json", "jds.yaml")):
            out = CODING_DIR / ("cvs/" + Path(f).name if f.endswith(".txt") else Path(f).name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(git("show", f"{ref}:{f}"))
    print(f"coding benchmark → {CODING_DIR} ({len(list((CODING_DIR / 'cvs').glob('*.txt')))} CVs)")


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(*sys.argv[2:3])
    elif sys.argv[1:2] == ["materialize"]:
        target = sys.argv[2:3]
        path = materialize_djinni() if target == ["djinni"] else (
            materialize_djinni_holdout() if target == ["djinni_holdout"] else materialize())
        print(f"DATA_DIR={path}")
    else:
        print(__doc__)
