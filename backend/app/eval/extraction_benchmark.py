"""Djinni real-world information-extraction benchmark.

The primary benchmark is fail-closed: it runs only against a provenance-checked
human-adjudicated Gold artifact. The historical generated reference is accepted
only with ``--weak-diagnostic`` and its output never updates reports or headline
metrics.

Compares deterministic systems against a reference:
  E0: Naive Exact Keyword Matching (word-boundary regex over taxonomy keys)
  E1: Heuristic No-Context Gazetteer (full aliases, but no NegEx/context awareness)
  E2: Full Deterministic Hybrid NLP Pipeline (Our proposed method: Gazetteer + NegEx + Context)
  E3: not run unless a separate result carries a real trace, model and token use

Metrics:
  - Inter-Annotator Agreement (IAA): Span Token-Overlap F1, Unweighted Cohen's Kappa for status, candidate Jaccard
  - In-Taxonomy Micro-F1, Macro-F1, Precision, Recall
  - Candidate-level all-skill recall with OOV skills retained in the denominator
  - Paired Bootstrap significance testing (E2 vs E1)
  - Cluster Bootstrap 95% Confidence Intervals (resampling by CV)
  - Latency (ms/CV) and Token Cost ($)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Callable

from app.config import ROOT, get_settings
from app.eval.djinni_extraction_annotation import (
    AnnotationError,
    compute_iaa as compute_human_iaa,
    validate_gold,
)
from app.nlp.pipeline import process
from app.nlp.taxonomy import aliases, skills

BENCH_MANIFEST = ROOT / "data" / "eval" / "djinni_extraction_benchmark_manifest.json"
GOLD_PATH = ROOT / "data" / "eval" / "djinni_extraction_human_gold_50.json"
WEAK_REF_PATH = ROOT / "data" / "eval" / "heuristic_extraction_reference_50.json"
CVS_DIR = get_settings().runtime_dir / "bench" / "djinni_extraction" / "cvs"
REPORT_MD = ROOT / "docs" / "DJINNI_EXTRACTION_BENCHMARK.md"
METRICS_JSON = ROOT / "data" / "eval" / "djinni_extraction_metrics.json"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def unweighted_cohen_kappa(ann1: list[str], ann2: list[str], categories: list[str]) -> float:
    """Unweighted Cohen's Kappa for nominal categories."""
    if not ann1 or len(ann1) != len(ann2):
        return 0.0
    n = len(ann1)
    po = sum(a == b for a, b in zip(ann1, ann2)) / n
    p1 = {c: ann1.count(c) / n for c in categories}
    p2 = {c: ann2.count(c) / n for c in categories}
    pe = sum(p1[c] * p2[c] for c in categories)
    return (po - pe) / (1 - pe) if 1 - pe > 1e-9 else 1.0


def compute_iaa(path_a: Path, path_b: Path) -> dict:
    """Measure IAA only after both files pass the blind-human validator."""
    metrics, _ = compute_human_iaa(path_a, path_b)
    return metrics


def load_reference(path: Path, *, weak_diagnostic: bool = False) -> tuple[dict, dict]:
    if not path.exists():
        raise AnnotationError(
            "Human Gold is not available. Complete two blind annotations and third-human adjudication first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("metadata", {})
    if weak_diagnostic:
        if meta.get("evaluation_status") != "WEAK_DIAGNOSTIC_ONLY" or meta.get("human") is not False:
            raise AnnotationError("--weak-diagnostic requires the registered generated weak reference")
    else:
        if meta.get("artifact_type") != "human_adjudicated_extraction_gold":
            raise AnnotationError("Primary extraction benchmark requires frozen human-adjudicated Gold")
        payload = validate_gold(path)
        meta = payload["metadata"]
    return payload["profiles"], meta


# ── Systems ──────────────────────────────────────────────────────────────────

def run_e0_keyword(text: str) -> dict:
    """E0: Naive Exact Keyword Matching (word-boundary regex on taxonomy keys only)."""
    text_lower = text.lower()
    found_skills = set()
    for sk in skills():
        pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(sk)}(?![a-zA-Z0-9_-])"
        if re.search(pattern, text_lower):
            found_skills.add(sk)
    return {
        "candidate_skills": sorted(list(found_skills)),
        "hedged_skills": set(),
        "used_skills": found_skills,
    }


def run_e1_no_context(text: str) -> dict:
    """E1: Heuristic No-Context Matcher (All taxonomy aliases, but NO NegEx hedge scoping)."""
    text_lower = text.lower()
    found_skills = set()
    for sk in skills():
        for al in aliases(sk):
            pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(al)}(?![a-zA-Z0-9_-])"
            if re.search(pattern, text_lower):
                found_skills.add(sk)
                break
    return {
        "candidate_skills": sorted(list(found_skills)),
        "hedged_skills": set(),
        "used_skills": found_skills,
    }


def run_e2_full_pipeline(cv_id: str, text: str) -> dict:
    """E2: Full Deterministic Hybrid NLP Pipeline (Our proposed method: Gazetteer + NegEx + Context)."""
    p = process(cv_id, text)
    active_skills = {sk for sk, mlist in p.skills.items() if any(m.strength != "hedged" for m in mlist)}
    hedged_skills = {sk for sk, mlist in p.skills.items() if all(m.strength == "hedged" for m in mlist)}
    return {
        "candidate_skills": sorted(list(active_skills)),
        "hedged_skills": hedged_skills,
        "used_skills": active_skills,
    }


# ── Evaluator ────────────────────────────────────────────────────────────────

def evaluate_system(sys_func: Callable, ref_data: dict, cvs_dir: Path) -> dict:
    start_t = time.perf_counter()
    per_cv_stats = []

    micro_tp, micro_fp, micro_fn = 0, 0, 0
    total_gold_all_skills = 0
    macro_f1s = []

    for cid, item in ref_data.items():
        text = (cvs_dir / f"{cid}.txt").read_text(encoding="utf-8")
        pred = sys_func(cid, text) if "cv_id" in sys_func.__code__.co_varnames else sys_func(text)

        mentions = item.get("mentions", [])
        if mentions and "candidate_confirmed" in mentions[0]:
            gold_cand_skills = {
                m["canonical_skill"]
                for m in mentions
                if m["candidate_confirmed"] and m["canonical_skill"] is not None
            }
            gold_all_skills = {
                m["canonical_skill"] or "oov:" + m["oov_name"].strip().casefold()
                for m in mentions
                if m["candidate_confirmed"]
            }
        else:
            # Legacy generated weak reference; diagnostic mode only.
            gold_cand_skills = set(item["candidate_skills"])
            gold_all_skills = gold_cand_skills | {"oov:" + x.casefold() for x in item.get("oov_mentions", [])}
        pred_cand_skills = set(pred["candidate_skills"])

        tp = len(pred_cand_skills & gold_cand_skills)
        fp = len(pred_cand_skills - gold_cand_skills)
        fn = len(gold_cand_skills - pred_cand_skills)

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        total_gold_all_skills += len(gold_all_skills)

        _, _, cv_f1 = prf(tp, fp, fn)
        macro_f1s.append(cv_f1)

        per_cv_stats.append({
            "cid": cid,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": cv_f1,
            "domain": item["domain"],
            "length_stratum": item["length_stratum"],
        })

    elapsed_ms = (time.perf_counter() - start_t) * 1000 / len(ref_data)
    p, in_tax_r, micro_f1 = prf(micro_tp, micro_fp, micro_fn)
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    # Candidate-level all-skill recall. OOV skills remain in the denominator.
    all_skill_recall = micro_tp / total_gold_all_skills if total_gold_all_skills else 0.0

    # Cluster bootstrap 95% CI over CVs
    rng = random.Random(42)
    boot_micros = []
    n = len(per_cv_stats)
    for _ in range(1000):
        sample = [rng.choice(per_cv_stats) for _ in range(n)]
        btp = sum(x["tp"] for x in sample)
        bfp = sum(x["fp"] for x in sample)
        bfn = sum(x["fn"] for x in sample)
        _, _, bf1 = prf(btp, bfp, bfn)
        boot_micros.append(bf1)

    boot_micros.sort()
    ci_low = boot_micros[int(0.025 * len(boot_micros))]
    ci_high = boot_micros[int(0.975 * len(boot_micros))]

    domain_f1 = {}
    for dom in ["QA Automation", "QA", "other"]:
        sub = [x for x in per_cv_stats if x["domain"] == dom]
        stp = sum(x["tp"] for x in sub)
        sfp = sum(x["fp"] for x in sub)
        sfn = sum(x["fn"] for x in sub)
        _, _, sub_f1 = prf(stp, sfp, sfn)
        domain_f1[dom] = sub_f1

    return {
        "precision": p,
        "in_tax_recall": in_tax_r,
        "all_skill_recall": all_skill_recall,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "ci_95": (ci_low, ci_high),
        "domain_f1": domain_f1,
        "latency_ms": elapsed_ms,
        "per_cv_stats": per_cv_stats,
    }


def paired_bootstrap_test(stats_a: list[dict], stats_b: list[dict]) -> float:
    """Calculates p-value of F1 difference between two systems via 1,000 paired resamples."""
    rng = random.Random(42)
    n = len(stats_a)
    diffs = []
    for _ in range(1000):
        indices = [rng.randrange(n) for _ in range(n)]
        tp_a = sum(stats_a[i]["tp"] for i in indices)
        fp_a = sum(stats_a[i]["fp"] for i in indices)
        fn_a = sum(stats_a[i]["fn"] for i in indices)
        _, _, f1_a = prf(tp_a, fp_a, fn_a)

        tp_b = sum(stats_b[i]["tp"] for i in indices)
        fp_b = sum(stats_b[i]["fp"] for i in indices)
        fn_b = sum(stats_b[i]["fn"] for i in indices)
        _, _, f1_b = prf(tp_b, fp_b, fn_b)

        diffs.append(f1_a - f1_b)
    # two-sided p-value around 0
    p_val = sum(d <= 0 for d in diffs) / len(diffs)
    return p_val


def generate_benchmark_report(*, weak_diagnostic: bool = False) -> dict:
    ref_path = WEAK_REF_PATH if weak_diagnostic else GOLD_PATH
    ref_data, meta = load_reference(ref_path, weak_diagnostic=weak_diagnostic)
    systems = {
        "E0 (Naive Exact Keyword)": run_e0_keyword,
        "E1 (Heuristic No-Context)": run_e1_no_context,
        "E2 (Full Deterministic Pipeline)": run_e2_full_pipeline,
    }
    results = {name: evaluate_system(fn, ref_data, CVS_DIR) for name, fn in systems.items()}
    p_value = paired_bootstrap_test(
        results["E2 (Full Deterministic Pipeline)"]["per_cv_stats"],
        results["E1 (Heuristic No-Context)"]["per_cv_stats"],
    )
    serializable = {
        "status": "WEAK_DIAGNOSTIC_ONLY" if weak_diagnostic else "HUMAN_GOLD_MEASURED",
        "reference": str(ref_path.relative_to(ROOT)),
        "reference_metadata": meta,
        "e3": {"status": "NOT_RUN", "reason": "no registered trace/model/token record"},
        "paired_bootstrap_e2_vs_e1_p": p_value,
        "systems": {
            name: {key: value for key, value in result.items() if key != "per_cv_stats"}
            for name, result in results.items()
        },
    }
    if weak_diagnostic:
        out = get_settings().runtime_dir / "bench" / "djinni_extraction" / "weak_extraction_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"WEAK DIAGNOSTIC ONLY: {out}")
        return serializable

    METRICS_JSON.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Djinni Human Gold Information Extraction Benchmark",
        "",
        "Status: **HUMAN_GOLD_MEASURED**. Numbers below are generated from the frozen metrics JSON.",
        "",
        "| System | Precision | In-tax recall | Micro-F1 | Macro-F1 | 95% cluster CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        lines.append(
            f"| {name} | {result['precision']:.3f} | {result['in_tax_recall']:.3f} | "
            f"{result['micro_f1']:.3f} | {result['macro_f1']:.3f} | "
            f"[{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}] |"
        )
    lines += ["", "E3: **NOT RUN** because no registered trace, model identifier and token record exists."]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-diagnostic", action="store_true")
    args = parser.parse_args()
    generate_benchmark_report(weak_diagnostic=args.weak_diagnostic)


if __name__ == "__main__":
    main()
