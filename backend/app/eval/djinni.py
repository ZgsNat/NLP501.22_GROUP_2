"""Djinni-QA human gold and external-validity pilot.

The real texts stay under ``.runtime/bench/djinni``.  Only human judgements and
their reproducible aggregate are committed.  Two independent 0--3 judgements
are combined by their arithmetic mean; disagreement is retained as an interval
and a confidence flag instead of being hidden by rounding or ``min``.

    python -m app.eval.djinni check
    python -m app.eval.djinni gold
    python -m app.eval.djinni report
    python -m app.eval.djinni all
    python -m app.eval.djinni freeze-production
    python -m app.eval.djinni production-report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from app.config import ROOT, get_settings
from app.eval import metrics

LABELS = ROOT / "data" / "eval"
BENCH = get_settings().runtime_dir / "bench" / "djinni"
DAN = LABELS / "human_djinni_dan.json"
VU = LABELS / "human_djinni_vu.json"
GOLD = LABELS / "human_djinni_gold.json"
REPORT = ROOT / "docs" / "EXTERNAL_VALIDITY.md"
PRODUCTION_MANIFEST = LABELS / "djinni_production_v1.json"
PRODUCTION_REPORT = ROOT / "docs" / "DJINNI_PRODUCTION_RESULTS.md"
PRODUCTION_RESULTS = get_settings().runtime_dir / "eval" / "results_djinni"
PRODUCTION_SYSTEMS = ("B0 BM25", "B1 dense", "B2 hybrid", "B3 8-axis ⊕ CE [hybrid]", "P6")


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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def production_manifest() -> dict[str, Any]:
    """Protocol and inputs frozen before any production-system result is inspected."""
    return {
        "benchmark": "djinni-human",
        "version": "production-v1",
        "frozen_before_run": True,
        "tuning": "none; one controlled run; failures are reported, not tuned away",
        "systems": list(PRODUCTION_SYSTEMS),
        "runtime": {
            "llm_by_component": {
                "B3_jd_parser": "ollama:qwen3:1.7b",
                "P6_agent": "google:gemini-3.8-flash",
            },
            "jd_parser": "hybrid",
            "dense_model": "BAAI/bge-small-en-v1.5",
            "rerank_model": "Xenova/ms-marco-MiniLM-L-6-v2",
            "onnx_batch_sizes": {"embedding": 16, "rerank": 16},
            "python": "3.12 (Linux container, dependencies from frozen uv.lock)",
        },
        "protocol_amendments": [
            {
                "when": "before any P6 result existed",
                "change": "P6 provider changed from local Ollama to Google Gemini because local inference was operationally infeasible; Gemini 3.8 Flash was the strongest user-listed model that passed the pre-run quota smoke test (Pro/Argon had zero quota)",
                "performance_observed_before_change": False,
                "scope": "P6 only; B0-B3 configurations and completed outputs unchanged",
            }
        ],
        "evaluation": {
            "ranking": "judged-only nDCG@10 on unrounded Aggregate Human Gold",
            "binary": "P@5 relevant iff both Vũ and Đán graded >=2",
            "uncertainty": "JD bootstrap CI plus min/max label-sensitivity bounds",
            "unjudged": "ignored, never mapped to grade 0",
        },
        "inputs": {
            str(GOLD.relative_to(ROOT)): _sha(GOLD),
            ".runtime/bench/djinni/jds.yaml": _sha(BENCH / "jds.yaml"),
            ".runtime/bench/djinni/cvs/*": _tree_sha(list((BENCH / "cvs").glob("*.txt"))),
        },
    }


def freeze_production() -> dict[str, Any]:
    current = production_manifest()
    if PRODUCTION_MANIFEST.exists():
        frozen = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
        if frozen != current:
            raise ValueError("Djinni production inputs/protocol changed after freeze")
        return frozen
    PRODUCTION_MANIFEST.write_text(json.dumps(current, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return current


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path.name}: labels must be a non-empty list")
    pairs: list[tuple[str, str]] = []
    for i, row in enumerate(labels):
        if not all(k in row for k in ("jd", "cv", "grade")):
            raise ValueError(f"{path.name}: label {i} misses jd/cv/grade")
        if not isinstance(row["grade"], int) or not 0 <= row["grade"] <= 3:
            raise ValueError(f"{path.name}: label {i} has invalid grade {row['grade']!r}")
        if row.get("saw_model") is not False:
            raise ValueError(f"{path.name}: label {i} is not recorded as blind")
        pairs.append((row["jd"], row["cv"]))
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"{path.name}: duplicate (JD, CV) pairs")
    if payload.get("saw_model") != 0:
        raise ValueError(f"{path.name}: top-level saw_model must be 0 after provenance correction")
    return payload


def _by_pair(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(x["jd"], x["cv"]): x for x in payload["labels"]}


def agreement(vu: dict[str, Any], dan: dict[str, Any]) -> dict[str, Any]:
    if vu.get("pool") != dan.get("pool"):
        raise ValueError("Vũ and Đán label different pools")
    vm, dm = _by_pair(vu), _by_pair(dan)
    if set(vm) != set(dm):
        raise ValueError(f"pair sets differ: Vũ={len(vm)}, Đán={len(dm)}, common={len(set(vm) & set(dm))}")
    keys = sorted(vm)
    vg, dg = [vm[k]["grade"] for k in keys], [dm[k]["grade"] for k in keys]
    delta = [abs(a - b) for a, b in zip(vg, dg)]
    per_jd: dict[str, dict[str, float | int]] = {}
    for jd in sorted({j for j, _ in keys}):
        subset = [k for k in keys if k[0] == jd]
        a, b = [vm[k]["grade"] for k in subset], [dm[k]["grade"] for k in subset]
        per_jd[jd] = {
            "pairs": len(subset),
            "exact": sum(x == y for x, y in zip(a, b)) / len(subset),
            "weighted_kappa": metrics.weighted_kappa(a, b),
        }
    return {
        "pairs": len(keys),
        "exact": sum(x == y for x, y in zip(vg, dg)) / len(keys),
        "weighted_kappa": metrics.weighted_kappa(vg, dg),
        "within_one": sum(x <= 1 for x in delta) / len(delta),
        "mean_absolute_difference": sum(delta) / len(delta),
        "bias_vu_minus_dan": sum(a - b for a, b in zip(vg, dg)) / len(keys),
        "confidence_counts": {
            "high": sum(x == 0 for x in delta),
            "medium": sum(x == 1 for x in delta),
            "low": sum(x >= 2 for x in delta),
        },
        "per_jd": per_jd,
    }


def build_gold() -> dict[str, Any]:
    vu, dan = _load(VU), _load(DAN)
    stats = agreement(vu, dan)
    vm, dm = _by_pair(vu), _by_pair(dan)
    rows = []
    for jd, cv in sorted(vm):
        vg, dg = vm[(jd, cv)]["grade"], dm[(jd, cv)]["grade"]
        gap = abs(vg - dg)
        rows.append({
            "jd": jd,
            "cv": cv,
            "grade": (vg + dg) / 2,
            "vu_grade": vg,
            "dan_grade": dg,
            "lower": min(vg, dg),
            "upper": max(vg, dg),
            "relevant": vg >= 2 and dg >= 2,
            "confidence": "high" if gap == 0 else ("medium" if gap == 1 else "low"),
            "vu_note": vm[(jd, cv)].get("note", ""),
        })
    return {
        "dataset": "djinni-qa",
        "version": "aggregate-human-gold-v1",
        "pool": vu["pool"],
        "method": "dual-annotator arithmetic mean; no rounding",
        "adjudicated": False,
        "annotators": [vu.get("annotator", "Vũ"), dan.get("annotator", "Đán")],
        "binary_relevance": "vu_grade >= 2 and dan_grade >= 2",
        "source_files": {VU.name: _sha(VU), DAN.name: _sha(DAN)},
        "agreement": stats,
        "labels": rows,
    }


def write_gold() -> dict[str, Any]:
    gold = build_gold()
    GOLD.write_text(json.dumps(gold, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return gold


TOKEN = re.compile(r"[a-z0-9+#.]{2,}")


def _tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def _bm25_rank(query: str, docs: dict[str, str]) -> list[str]:
    terms = Counter(_tokens(query))
    doc_terms = {k: Counter(_tokens(v)) for k, v in docs.items()}
    lengths = {k: sum(v.values()) for k, v in doc_terms.items()}
    avgdl = sum(lengths.values()) / len(lengths)
    df = Counter(t for counts in doc_terms.values() for t in counts)
    n, k1, b = len(docs), 1.2, 0.75
    scores: dict[str, float] = {}
    for cid, counts in doc_terms.items():
        score = 0.0
        for term, qf in terms.items():
            if not (tf := counts.get(term, 0)):
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += qf * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * lengths[cid] / avgdl))
        scores[cid] = score
    return sorted(docs, key=lambda c: (-scores[c], c))


def _pilot_rankings(jds: list[dict[str, Any]], cvs: dict[str, str], meta: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Cheap, reproducible pilot baselines.  These are deliberately not called B1/B2/B3/P6/P7."""
    keyword = meta.get("primary_keyword", {})
    out = {"Candidate keyword prior": {}, "B0 BM25": {}, "Criteria heuristic pilot": {}}
    for jd in jds:
        jid = jd["id"]
        out["Candidate keyword prior"][jid] = sorted(
            cvs, key=lambda c: (-{"QA Automation": 2, "QA": 1}.get(keyword.get(c, ""), 0), c)
        )
        out["B0 BM25"][jid] = _bm25_rank(jd["text"], cvs)
        stack = set(_tokens(str(jd.get("stack", ""))))
        title = set(_tokens(str(jd.get("title", ""))))
        scores = {}
        for cid, text in cvs.items():
            ts = set(_tokens(text))
            scores[cid] = (
                45 * (keyword.get(cid) == "QA Automation")
                + 20 * (keyword.get(cid) == "QA")
                + 8 * len(stack & ts)
                + 2 * len(title & ts)
                + 5 * bool({"framework", "selenium", "playwright", "cypress", "appium"} & ts)
            )
        out["Criteria heuristic pilot"][jid] = sorted(cvs, key=lambda c: (-scores[c], c))
    return out


def evaluate(gold: dict[str, Any]) -> dict[str, Any]:
    for required in (BENCH / "jds.yaml", BENCH / "meta.json", BENCH / "cvs"):
        if not required.exists():
            raise FileNotFoundError(f"Djinni runtime data missing: {required}")
    jds = yaml.safe_load((BENCH / "jds.yaml").read_text(encoding="utf-8"))
    cvs = {p.stem: p.read_text(encoding="utf-8") for p in sorted((BENCH / "cvs").glob("*.txt"))}
    meta = json.loads((BENCH / "meta.json").read_text(encoding="utf-8"))
    rankings = _pilot_rankings(jds, cvs, meta)
    rows = gold["labels"]
    variants = {
        "lower": defaultdict(dict),
        "mean": defaultdict(dict),
        "upper": defaultdict(dict),
    }
    relevant: dict[str, dict[str, bool]] = defaultdict(dict)
    for x in rows:
        for key in variants:
            variants[key][x["jd"]][x["cv"]] = float(x["grade"] if key == "mean" else x[key])
        relevant[x["jd"]][x["cv"]] = bool(x["relevant"])
    results: dict[str, Any] = {}
    for system, per_jd in rankings.items():
        measured = {"ndcg": {k: [] for k in variants}, "p5": [], "tau": []}
        for jd in sorted(variants["mean"]):
            order = [c for c in per_jd[jd] if c in variants["mean"][jd]]
            for key, grades in variants.items():
                measured["ndcg"][key].append(metrics.ndcg(order, grades[jd], 10))
            measured["p5"].append(sum(relevant[jd][c] for c in order[:5]) / 5)
            measured["tau"].append(metrics.tau_vs_gold(order, variants["mean"][jd]))
        results[system] = {
            "jds": len(measured["p5"]),
            "ndcg10_lower": metrics.bootstrap_ci(measured["ndcg"]["lower"]),
            "ndcg10": metrics.bootstrap_ci(measured["ndcg"]["mean"]),
            "ndcg10_upper": metrics.bootstrap_ci(measured["ndcg"]["upper"]),
            "p5": metrics.bootstrap_ci(measured["p5"]),
            "tau": metrics.bootstrap_ci(measured["tau"]),
        }
    return {"systems": results, "n_jds": len(jds), "n_cvs": len(cvs)}


def _result_path(name: str) -> Path:
    return PRODUCTION_RESULTS / f"{name.replace(' ', '_').replace('+', 'plus').replace(':', '-')}.json"


def evaluate_production(gold: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score persisted production orders without loading a model or treating unjudged CVs as zero."""
    freeze_production()
    gold = gold or build_gold()
    variants = {key: defaultdict(dict) for key in ("lower", "mean", "upper")}
    relevant: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in gold["labels"]:
        variants["lower"][row["jd"]][row["cv"]] = float(row["lower"])
        variants["mean"][row["jd"]][row["cv"]] = float(row["grade"])
        variants["upper"][row["jd"]][row["cv"]] = float(row["upper"])
        relevant[row["jd"]][row["cv"]] = bool(row["relevant"])
    out: dict[str, Any] = {"manifest": _sha(PRODUCTION_MANIFEST), "systems": {}, "missing": []}
    per_system_values: dict[str, list[float]] = {}
    for system in PRODUCTION_SYSTEMS:
        path = _result_path(system)
        if not path.exists():
            out["missing"].append(system)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_jd = payload["jds"]
        values = {"lower": [], "mean": [], "upper": [], "p5": [], "tau": []}
        orders = {}
        for jd in sorted(variants["mean"]):
            raw = per_jd.get(jd)
            if raw is None:
                raise ValueError(f"{system}: missing JD {jd}")
            raw_order = raw if isinstance(raw, list) else raw["ranking"]
            order = [cv for cv in raw_order if cv in variants["mean"][jd]]
            if len(order) != len(set(order)) or set(order) != set(variants["mean"][jd]):
                raise ValueError(f"{system}/{jd}: production order must cover every judged CV exactly once")
            orders[jd] = order
            for key in ("lower", "mean", "upper"):
                values[key].append(metrics.ndcg(order, variants[key][jd], 10))
            values["p5"].append(sum(relevant[jd][cv] for cv in order[:5]) / 5)
            values["tau"].append(metrics.tau_vs_gold(order, variants["mean"][jd]))
        per_system_values[system] = values["mean"]
        out["systems"][system] = {
            "orders_sha256": hashlib.sha256(json.dumps(orders, sort_keys=True).encode()).hexdigest(),
            "ndcg10_lower": metrics.bootstrap_ci(values["lower"]),
            "ndcg10": metrics.bootstrap_ci(values["mean"]),
            "ndcg10_upper": metrics.bootstrap_ci(values["upper"]),
            "p5_conservative": metrics.bootstrap_ci(values["p5"]),
            "tau": metrics.bootstrap_ci(values["tau"]),
            "per_jd_ndcg10": dict(zip(sorted(variants["mean"]), values["mean"])),
        }
    if "B0 BM25" in per_system_values:
        base = per_system_values["B0 BM25"]
        for name, values in per_system_values.items():
            out["systems"][name]["delta_vs_b0"] = sum(values) / len(values) - sum(base) / len(base)
            out["systems"][name]["p_vs_b0"] = metrics.paired_bootstrap_p(values, base)
    return out


def render_production(result: dict[str, Any]) -> str:
    lines = [
        "# Djinni-QA production ladder on Aggregate Human Gold", "",
        "Frozen protocol: `data/eval/djinni_production_v1.json`. No tuning after the freeze. Rankings are filtered "
        "to judged pairs before scoring; unjudged profiles are ignored, never assigned zero relevance.", "",
        "| System | nDCG@10 on min grades | nDCG@10 Aggregate Gold | nDCG@10 on max grades | P@5 both≥2 | τ-b | Δ vs B0 | p(first>B0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"  # noqa: E731
    for name in PRODUCTION_SYSTEMS:
        if name not in result["systems"]:
            lines.append(f"| {name} | pending | pending | pending | pending | pending | — | — |")
            continue
        row = result["systems"][name]
        lines.append(f"| {name} | {fmt(row['ndcg10_lower'])} | **{fmt(row['ndcg10'])}** | "
                     f"{fmt(row['ndcg10_upper'])} | {fmt(row['p5_conservative'])} | {fmt(row['tau'])} | "
                     f"{row.get('delta_vs_b0', 0):+.3f} | {row.get('p_vs_b0', 1):.3f} |")
    lines += ["", "The min/max-grade columns are label-sensitivity analyses, not mathematical lower/upper bounds on nDCG: changing grades also changes the ideal DCG denominator.", "",
              "Per-JD orders are persisted under `.runtime/eval/results_djinni/`; the report stores an order hash "
              "for auditability. Missing systems remain `pending` and are never substituted with pilot heuristics.",
              ""]
    if "P6" in result["systems"]:
        lines += ["P6 status: Gemini 3.8 Flash completed successfully across all 8 production JDs.", ""]
    else:
        lines += ["P6 status: two Gemini 3.8 Flash attempts stopped before the first completed JD after HTTP 503 "
                  "high-demand responses. No partial score was promoted and B0–B3 were not rerun.", ""]
    return "\n".join(lines)


def write_production_report() -> dict[str, Any]:
    result = evaluate_production()
    PRODUCTION_REPORT.write_text(render_production(result), encoding="utf-8")
    return result


def render(gold: dict[str, Any], result: dict[str, Any]) -> str:
    a, c = gold["agreement"], gold["agreement"]["confidence_counts"]
    lines = [
        "# External validity: Djinni-QA human-labelled pilot",
        "",
        "Generated by `python -m app.eval.djinni all`; do not edit measured values by hand.",
        "",
        "## Data and label protocol",
        "",
        f"- Real, anonymised Djinni data: **{result['n_jds']} JDs**, **{result['n_cvs']} CVs**; TREC-style judged pool: **{a['pairs']} (JD, CV) pairs**.",
        "- Vũ and Đán graded the same pool independently on the 0–3 rubric without system grades.",
        "- Aggregate Human Gold = arithmetic mean of the two grades, retaining half-steps; it is aggregate, not adjudicated.",
        "- Binary relevance for P@5 is conservative: both annotators must grade the pair at least 2.",
        "- Real texts remain in `.runtime/`; only judgements and their aggregate are committed.",
        "",
        "## Human agreement",
        "",
        "| pairs | exact | within ±1 | quadratic-weighted κ | mean absolute difference | Vũ−Đán bias |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {a['pairs']} | {a['exact']:.1%} | {a['within_one']:.1%} | **{a['weighted_kappa']:.3f}** | {a['mean_absolute_difference']:.3f} | {a['bias_vu_minus_dan']:+.3f} |",
        "",
        f"Confidence retained per judgement: **{c['high']} high** (same grade), **{c['medium']} medium** (one-grade gap), **{c['low']} low** (gap ≥2).",
        "",
        "## Measured pilot baselines",
        "",
        "Mean and 95% bootstrap CI over 8 JDs. Evaluation is judged-only because the 424-pair pool was formed from B0–B3 top results plus random controls. The min/max-grade columns are sensitivity analyses, not mathematical nDCG bounds; the headline Gold is the unrounded annotator mean.",
        "",
        "| System | JDs | nDCG@10 on min grades | nDCG@10 Aggregate Gold | nDCG@10 on max grades | P@5 (both ≥2) | τ-b |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda x: f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"  # noqa: E731
    for name, row in result["systems"].items():
        lines.append(f"| {name} | {row['jds']} | {fmt(row['ndcg10_lower'])} | **{fmt(row['ndcg10'])}** | "
                     f"{fmt(row['ndcg10_upper'])} | {fmt(row['p5'])} | {fmt(row['tau'])} |")
    lines += [
        "",
        "## What this resolves—and what it does not",
        "",
        "- **Resolved:** the project now has human relevance judgements on real recruitment text, a versioned Aggregate Human Gold artifact, agreement statistics, and a reproducible external pilot.",
        "- **Production B0–B3 are now measured separately:** see `DJINNI_PRODUCTION_RESULTS.md`; P6 remains pending after provider HTTP 503 failures and has no reported score.",
        "- **A virgin holdout is locked, not yet measured:** six disjoint JDs and 40 disjoint CVs are frozen in `djinni_holdout_v1.json`; the fail-closed gate requires two blind human label files before any run.",
        "- **Pooling limitation:** unjudged pairs are ignored, not treated as grade 0. Pooling was built from B0–B3, so it is suitable for a pilot but may miss candidates uniquely promoted by a new system.",
        "- **Sample size:** eight JDs give wide intervals. Report per-source results separately from the 10-JD synthetic test split.",
    ]
    return "\n".join(lines) + "\n"


def run(write_report: bool = True) -> dict[str, Any]:
    gold = write_gold()
    result = evaluate(gold)
    if write_report:
        REPORT.write_text(render(gold, result), encoding="utf-8")
    return {"gold": gold, "evaluation": result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?",
                        choices=["check", "gold", "report", "all", "freeze-production", "production-report"],
                        default="all")
    args = parser.parse_args()
    if args.command == "check":
        g = build_gold()
        print(json.dumps(g["agreement"], ensure_ascii=False, indent=2))
    elif args.command == "gold":
        g = write_gold()
        print(f"{len(g['labels'])} labels -> {GOLD}")
    elif args.command == "freeze-production":
        print(json.dumps(freeze_production(), ensure_ascii=False, indent=2))
    elif args.command == "production-report":
        print(render_production(write_production_report()))
    else:
        out = run(write_report=True)
        print(f"{len(out['gold']['labels'])} labels -> {GOLD}")
        print(f"report -> {REPORT}")


if __name__ == "__main__":
    main()
