"""NLP501-bench v1: one frozen benchmark that mixes both data sources (docs/BENCHMARK.md).

    ours     12 JDs × 50 CVs, silver labels (data/eval/annotator_*.yaml)
    coding   6 JDs × 200 CVs from the `coding` branch at a pinned commit, gold from its rubric
Each JD ranks the CVs labelled for it (its own source's pool). JDs are split once into dev (tuning allowed) and test
(locked); the official number is the mean over the test JDs. The manifest pins the split, the source commit and a
SHA-256 of every input file, and `verify` fails if anything changed.

    python -m app.eval.benchmark freeze                 write data/eval/benchmark_v1.json (once; a change = v2)
    python -m app.eval.benchmark verify                 check every input against the manifest
    python -m app.eval.benchmark dan-core --src PATH    rank with the coding branch's own deterministic core
                                                         (PATH = a checkout of the pinned commit)
    python -m app.eval.benchmark report                 → docs/BENCHMARK_RESULTS.md
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

from app.config import ROOT
from app.eval import metrics
from app.eval.bench import CODING_DIR, load
from app.eval.run import result_path, save, score_system

MANIFEST = ROOT / "data" / "eval" / "benchmark_v1.json"
DOC = ROOT / "docs" / "BENCHMARK_RESULTS.md"
CODING_COMMIT = "6991c3d427de7a4eb5bea95bf2c4ce32395968ef"  # origin/coding when the team agreed on v1
DAN = "Đán core (coding branch)"
# Head-to-head: the three versions of the project, then the merged agent and its fixed-pipeline twin.
SYSTEMS = {"B0 BM25": "keyword baseline, whole CV",
           "B2 hybrid": "BM25 + dense (RRF), whole CV",
           "R structural + CE": "requirement-wise retrieval over structural chunks + cross-encoder",
           DAN: "the coding branch's deterministic ranker, unchanged",
           "B3 rules ⊕ CE": "our v2 branch: 4 rule criteria fused with the cross-encoder",
           "B3 8-axis ⊕ CE [hybrid]": "merged v3.1, deterministic: 8 rule criteria, hybrid JD parser",
           "B4": "merged v3.1, fixed pipeline: the agent with every loop off",
           "P6": "merged v3.1, full agent"}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _inputs() -> list[Path]:
    files = sorted((ROOT / "data" / "cvs").glob("*.md")) + sorted((ROOT / "data" / "jds").glob("*.md"))
    files += sorted((ROOT / "data" / "eval").glob("annotator_*.yaml"))
    files += [CODING_DIR / "gold.json", CODING_DIR / "jds.yaml"] + sorted((CODING_DIR / "cvs").glob("*.txt"))
    return files


def split_ours(seed: int = 501) -> dict[str, list[str]]:
    """Stratified by company type: one test JD per type, then two more at random; the rest is dev."""
    rnd, by = random.Random(seed), {}
    for jd in load("ours").jds:
        by.setdefault(jd.meta.get("company_profile", "default"), []).append(jd.id)
    test = [rnd.choice(sorted(ids)) for _, ids in sorted(by.items())]
    test += rnd.sample(sorted(j for ids in by.values() for j in ids if j not in test), 2)
    return {"test": sorted(test), "dev": sorted(j for ids in by.values() for j in ids if j not in test)}


def freeze() -> dict:
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} exists: v1 is frozen. A changed benchmark is a new version (benchmark_v2.json).")
    gold = json.loads((CODING_DIR / "gold.json").read_text())
    ours = split_ours()
    m = {"version": "NLP501-bench v1", "coding_commit": CODING_COMMIT, "seed": 501,
         "splits": {"test": {"ours": ours["test"], "coding": sorted(j for j, v in gold.items() if v["split"] == "test")},
                    "dev": {"ours": ours["dev"], "coding": sorted(j for j, v in gold.items() if v["split"] != "test")}},
         "labels": {"ours": "silver: mean of two LLM annotators, 0–3 (to be replaced by human-adjudicated labels = v1.1)",
                    "coding": "gold from the coding branch's annotation rubric, 0–3"},
         "sha256": {str(p.relative_to(ROOT)): _sha(p) for p in _inputs()}}
    MANIFEST.write_text(json.dumps(m, indent=1, ensure_ascii=False))
    return m


def verify() -> list[str]:
    """Inputs that differ from the manifest (empty = the benchmark is intact)."""
    m = json.loads(MANIFEST.read_text())
    bad = [p for p, h in m["sha256"].items() if not (ROOT / p).exists() or _sha(ROOT / p) != h]
    extra = [str(p.relative_to(ROOT)) for p in _inputs() if str(p.relative_to(ROOT)) not in m["sha256"]]
    return bad + [f"{p} (not in manifest)" for p in extra]


def dan_core(src: Path) -> None:
    """Rank every benchmark JD with the coding branch's own deterministic core (cv_agent.ranking), unchanged."""
    sys.path.insert(0, str(src))
    from cv_agent.criteria.jd_parser import parse_jd
    from cv_agent.criteria.profiles import list_profiles
    from cv_agent.ingest.pipeline import ingest_text
    from cv_agent.ranking.scoring import rank, score_candidate

    profiles = list_profiles()
    name = {"bank": "Ngân hàng", "startup": "Startup", "outsourcing": "Outsourcing", "default": "Mặc định"}
    for bench in ("ours", "coding"):
        b = load(bench)
        cvs = [ingest_text(d.text, d.id) for d in b.cvs]
        meta = {j["id"]: j for j in yaml.safe_load((CODING_DIR / "jds.yaml").read_text())} if bench == "coding" else {}
        per_jd = {}
        for jd in b.jds:
            company = meta.get(jd.id, {}).get("company_type") or jd.meta.get("company_profile", "default")
            profile = meta.get(jd.id, {}).get("profile") or name.get(jd.meta.get("company_profile", "default"), "Mặc định")
            parsed = parse_jd(jd.text, jd.id, jd.meta.get("title", ""), company)
            per_jd[jd.id] = [c.cv_id for c in rank([score_candidate(cv, parsed, profiles[profile]) for cv in cvs])]
        save(DAN, per_jd, bench)
        print(f"{DAN}: {bench} {len(per_jd)} JDs")


def _scores(system: str) -> dict[str, dict[str, float]]:
    """Per-JD metrics of a system over both sources (only the JDs it has results for)."""
    out = {}
    for bench in ("ours", "coding"):
        path = result_path(system, bench)
        if not path.exists():
            continue
        b = load(bench)
        per_jd = json.loads(path.read_text())["jds"]
        sc = score_system(per_jd, b.gold, b.traps)
        for i, jd in enumerate(sc["jds"]):
            out[jd] = {k: sc[k][i] for k in ("ndcg@10", "p@5", "r@20", "mrr", "tau")}
    return out


def report() -> str:
    m = json.loads(MANIFEST.read_text())
    bad = verify()
    scores = {s: _scores(s) for s in SYSTEMS}
    lines = ["# NLP501-bench v1 results (generated by `python -m app.eval.benchmark report`; do not edit by hand)", "",
             f"Protocol: docs/BENCHMARK.md. Manifest: data/eval/benchmark_v1.json "
             f"({'all inputs match' if not bad else f'⚠️ {len(bad)} inputs changed: ' + ', '.join(bad[:5])}). "
             "Each JD ranks the CVs labelled for it; mean over JDs with a 95% bootstrap CI; paired one-sided bootstrap "
             "p for “first > second”. LLM: Claude Sonnet 5.5 / Haiku 5.5; PYTHONHASHSEED=0.", ""]
    for split in ("test", "dev"):
        ids = {src: m["splits"][split][src] for src in ("ours", "coding")}
        every = ids["ours"] + ids["coding"]
        lines += [f"## {split.capitalize()} split: {len(ids['ours'])} JDs from ours-silver + {len(ids['coding'])} from coding-rubric"
                  + (" (official)" if split == "test" else " (tuning allowed)"), "",
                  "| System | what it is | JDs | nDCG@10 (all) | ours-silver | coding-rubric | P@5 | R@20 | τ |", "|---|---|---|---|---|---|---|---|---|"]
        rows = {}
        for s, what in SYSTEMS.items():
            have = [j for j in every if j in scores[s]]
            if not have:
                lines.append(f"| {s} | {what} | 0 | not run | | | | | |")
                continue
            mean = lambda k, js: sum(scores[s][j][k] for j in js) / len(js) if js else float("nan")  # noqa: E731
            nd = [scores[s][j]["ndcg@10"] for j in have]
            rows[s] = dict(zip(have, nd))
            part = lambda src: f"{mean('ndcg@10', [j for j in ids[src] if j in scores[s]]):.3f}" if any(j in scores[s] for j in ids[src]) else "—"  # noqa: E731
            lines.append(f"| {s} | {what} | {len(have)} | " + "{:.3f} [{:.3f}, {:.3f}]".format(*metrics.bootstrap_ci(nd))
                         + f" | {part('ours')} | {part('coding')} | {mean('p@5', have):.3f} | {mean('r@20', have):.3f} "
                         f"| {mean('tau', have):.3f} |")
        lines += ["", "| First | second | JDs in common | ΔnDCG@10 | p |", "|---|---|---|---|---|"]
        for a, b in (("B3 8-axis ⊕ CE [hybrid]", DAN), ("B3 8-axis ⊕ CE [hybrid]", "B3 rules ⊕ CE"),
                     ("P6", DAN), ("P6", "B4")):
            common = [j for j in every if j in rows.get(a, {}) and j in rows.get(b, {})]
            if common:
                xs, ys = [rows[a][j] for j in common], [rows[b][j] for j in common]
                lines.append(f"| {a} | {b} | {len(common)} | {sum(xs) / len(xs) - sum(ys) / len(ys):+.3f} | "
                             f"{metrics.paired_bootstrap_p(xs, ys):.3f} |")
        lines += ["", "Per JD nDCG@10:", "", "| System | " + " | ".join(every) + " |", "|---|" + "---|" * len(every)]
        for s in rows:
            lines.append(f"| {s} | " + " | ".join(f"{rows[s][j]:.2f}" if j in rows[s] else "—" for j in every) + " |")
        lines.append("")
    DOC.write_text("\n".join(lines))
    return DOC.read_text()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["freeze", "verify", "dan-core", "report"])
    ap.add_argument("--src", type=Path, help="dan-core: a checkout of the coding branch at the pinned commit")
    a = ap.parse_args()
    if a.what == "freeze":
        print(json.dumps(freeze()["splits"], indent=1))
    elif a.what == "verify":
        print("\n".join(verify()) or "benchmark intact")
    elif a.what == "dan-core":
        dan_core(a.src)
    else:
        print(report())
