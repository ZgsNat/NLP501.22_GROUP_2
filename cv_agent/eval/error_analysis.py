"""Per-JD error analysis of a finished experiment run (post hoc, never used to choose configurations).

A mean over 6 JDs hides where a method fails. This module reports, from ``results.csv``:

* every configuration's score per JD;
* near-duplicate job descriptions (word 3-gram Jaccard): the same posting re-published for several cities
  counts several times in a plain mean, so the report also gives the mean with each duplicate group collapsed
  into one JD (``dedup``) and the effective number of JDs;
* means for duplicate postings versus distinct ones, plus the number of different QA signals per JD (a JD with
  few signals passed the QA filter on loose phrases such as "test cases").

Groups come from the JD text only, never from the results; the report is still exploratory and is marked
``post_hoc`` so it is not read as a pre-registered hypothesis.
"""
from __future__ import annotations

import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

STRONG_QA_SIGNALS = 3        # distinct QA phrases a JD needs to count as a clear test-automation role
DUPLICATE_JACCARD = 0.8      # word 3-gram overlap above which two postings are the same job
SHINGLE = 3
CONFIGS = ("B0", "B1", "B2", "B3", "B4", "P6")
METRICS = ("ndcg@10", "unsupported_rate")


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def per_jd(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for r in rows:
        value = _num(r.get(metric))
        if not math.isnan(value):
            table.setdefault(str(r["jd_id"]), {})[r["config"]] = value
    return table


def _shingles(text: str) -> Set[tuple]:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return set()  # a JD without text is never a duplicate of another
    return {tuple(words[i:i + SHINGLE]) for i in range(max(len(words) - SHINGLE + 1, 1))}


def near_duplicates(texts: Dict[str, str], threshold: float = DUPLICATE_JACCARD) -> List[List[str]]:
    """Connected groups of texts whose pairwise shingle Jaccard reaches ``threshold``."""
    sh = {k: _shingles(v) for k, v in texts.items()}
    parent = {k: k for k in texts}

    def root(k: str) -> str:
        while parent[k] != k:
            k = parent[k]
        return k

    for a, b in combinations(sorted(texts), 2):
        union = sh[a] | sh[b]
        if union and len(sh[a] & sh[b]) / len(union) >= threshold:
            parent[root(b)] = root(a)
    groups: Dict[str, List[str]] = {}
    for k in sorted(texts):
        groups.setdefault(root(k), []).append(k)
    return sorted(g for g in groups.values() if len(g) > 1)


def signal_groups(signals: Dict[str, Sequence[str]], strong: int = STRONG_QA_SIGNALS) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"strong": [], "weak": []}
    for jd, found in sorted(signals.items()):
        out["strong" if len(set(found)) >= strong else "weak"].append(jd)
    return out


def group_means(table: Dict[str, Dict[str, float]], groups: Dict[str, List[str]],
                configs: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, jds in groups.items():
        out[name] = {"n_jds": len(jds)}
        for cfg in configs:
            vals = [table[j][cfg] for j in jds if cfg in table.get(j, {})]
            out[name][cfg] = sum(vals) / len(vals) if vals else float("nan")
    return out


def dedup_means(table: Dict[str, Dict[str, float]], dups: List[List[str]],
                configs: Sequence[str]) -> Dict[str, float]:
    """Mean over JDs where each duplicate group first collapses into the mean of its members."""
    grouped = {jd for g in dups for jd in g}
    units = [[jd] for jd in sorted(table) if jd not in grouped] + [list(g) for g in dups]
    out: Dict[str, float] = {"n_jds": len(units)}
    for cfg in configs:
        unit_means = []
        for unit in units:
            vals = [table[j][cfg] for j in unit if cfg in table.get(j, {})]
            if vals:
                unit_means.append(sum(vals) / len(vals))
        out[cfg] = sum(unit_means) / len(unit_means) if unit_means else float("nan")
    return out


def analyse_run(run_dir: Path, specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Write ``error_analysis.json`` next to ``results.csv`` and return it."""
    with open(run_dir / "results.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    tables = {m: per_jd(rows, m) for m in METRICS}
    configs = [c for c in CONFIGS if any(r["config"] == c for r in rows)]
    by_id = {s["id"]: s for s in specs if s["id"] in tables["ndcg@10"]}
    dups = near_duplicates({jd: s.get("text") or "" for jd, s in by_id.items()})
    in_dup = {jd for g in dups for jd in g}
    groups = {"duplicate_postings": sorted(in_dup), "distinct": sorted(set(tables["ndcg@10"]) - in_dup)}
    report = {
        "post_hoc": True,
        "groups_rule": f"duplicate_postings = word {SHINGLE}-gram Jaccard >= {DUPLICATE_JACCARD} with another JD",
        "near_duplicates": dups,
        "effective_n_jds": len(groups["distinct"]) + len(dups),
        "groups": group_means(tables["ndcg@10"], groups, configs),
        "dedup": dedup_means(tables["ndcg@10"], dups, configs),
        "group_members": groups,
        "unsupported_by_group": group_means(tables["unsupported_rate"], groups, [c for c in configs if c == "P6"]),
        "per_jd": {jd: {"title": by_id.get(jd, {}).get("title", ""),
                        "qa_signals": len(set(by_id.get(jd, {}).get("qa_signals") or [])),
                        "qa_signal_list": sorted(set(by_id.get(jd, {}).get("qa_signals") or [])),
                        "ndcg@10": tables["ndcg@10"][jd], "unsupported_rate": tables["unsupported_rate"].get(jd, {})}
                   for jd in sorted(tables["ndcg@10"])},
    }
    (run_dir / "error_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report
