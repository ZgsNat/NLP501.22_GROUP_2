"""Benchmarks the ladder runs on.

    ours    data/: 50 hand-written CVs, 12 JDs, silver labels from 2 independent annotators (docs/EVAL.md)
    coding  the `coding` branch's synthetic set (Đán): 200 template CVs, mostly Vietnamese, 6 JDs, gold from its own
            annotation rubric. Our code was never developed on it, so it is the closer thing to a held-out test.

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


def fetch(ref: str = "origin/coding") -> None:
    """Copy the coding branch's synthetic CVs, JDs and gold into .runtime/bench/coding (read-only on that branch)."""
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, check=True, capture_output=True).stdout  # noqa: E731
    files = git("ls-tree", "-r", "--name-only", ref, "data/synthetic", "data/jds/jds.yaml").decode().split("\n")
    for f in filter(None, files):
        if f.endswith((".txt", "gold.json", "jds.yaml")):
            out = CODING_DIR / ("cvs/" + Path(f).name if f.endswith(".txt") else Path(f).name)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(git("show", f"{ref}:{f}"))
    print(f"coding benchmark → {CODING_DIR} ({len(list((CODING_DIR / 'cvs').glob('*.txt')))} CVs)")


if __name__ == "__main__":
    fetch(*sys.argv[2:3]) if sys.argv[1:2] == ["fetch"] else print(__doc__)
