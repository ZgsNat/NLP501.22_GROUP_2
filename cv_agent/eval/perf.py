"""Latency tuning tools (NFR-04): near-tie threshold sweep on dev, and p50/p95 benchmark.

* ``tune_threshold`` — runs P6 with only the arbiter (no claims / verify) for each
  candidate threshold on the dev split; recommends the cheapest threshold whose
  mean nDCG@10 is within ``tolerance`` of the best one.
* ``bench`` — runs the full P6 with the LLM cache DISABLED (real latency) and
  reports p50/p95 per JD and per LLM task, calls, tokens, fallbacks and the
  order-consistency rate of pairwise judgements.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

from cv_agent.config import ROOT, deep_merge, load_config
from cv_agent.criteria.clarify import simulated_hr
from cv_agent.criteria.profiles import list_profiles
from cv_agent.eval.experiments import git_commit, jd_from_spec, ranking_metrics, specs_for
from cv_agent.ingest.pipeline import ingest_dir
from cv_agent.orchestration.graph import build_runtime, run_pipeline
from cv_agent.schema import CV

log = logging.getLogger(__name__)
EXACT_TIE = 1e-9   # threshold that only sends exact ties to the LLM


def percentile(xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q in 0..100); NaN for empty input."""
    data = sorted(xs)
    if not data:
        return float("nan")
    k = max(1, math.ceil(q / 100 * len(data)))
    return data[k - 1]


def _run_jds(cfg: Dict[str, Any], specs: List[Dict], cvs: List[CV], gold: Dict, run_dir: Optional[Path] = None):
    profiles = list_profiles()
    shared, rows = None, []
    for spec in specs:
        rt = build_runtime(cfg, (run_dir / spec["id"]) if run_dir else None, shared=shared,
                           hr_answer_fn=lambda _jd, qs, s=spec: simulated_hr(qs, s.get("hr_answers")))
        shared = shared or rt
        started = time.perf_counter()
        out = run_pipeline(rt, jd_from_spec(spec), cvs, profiles.get(spec.get("profile", "")))
        wall = time.perf_counter() - started
        ranking = [c.cv_id for c in out["ranking"]]
        g = gold.get(spec["id"], {})
        quality = ranking_metrics(ranking, g) if g else {"ndcg@10": float("nan"), "kendall_tau": float("nan")}
        js = out.get("judge_stats", {})
        rows.append({
            "jd_id": spec["id"], "wall_s": round(wall, 2), "fallback": bool(out.get("fallback")),
            "ndcg@10": quality["ndcg@10"], "kendall_tau": quality["kendall_tau"],
            "llm_calls": out["llm_stats"]["calls"], "tokens": out["llm_stats"]["used_tokens"],
            "judge_pairs": js.get("pairs", 0), "consistent_pairs": js.get("consistent_pairs", 0),
            "swaps": js.get("swaps", 0), "interrupted": js.get("interrupted"),
            "call_latency": [(r["task"], r["latency_s"], r["prompt_tokens"], r["completion_tokens"])
                             for r in rt.gateway.records],
        })
    return rows


def _load(data_dir: Path, split: str, jd_ids: Optional[Sequence[str]] = None):
    gold = json.loads((data_dir / "gold.json").read_text(encoding="utf-8"))
    specs = [s for s in specs_for(data_dir) if (jd_ids and s["id"] in jd_ids)
             or (not jd_ids and (split == "all" or s.get("split") == split))]
    return specs, ingest_dir(data_dir / "cvs"), gold


def tune_threshold(thresholds: Sequence[float], data_dir: Path, split: str = "dev", tolerance: float = 0.01,
                   overrides: Optional[Dict] = None) -> Dict[str, Any]:
    specs, cvs, gold = _load(data_dir, split)
    if not specs:
        raise ValueError(f"No JD in split {split!r}")
    table = []
    for t in thresholds:
        cfg = load_config("P6", overrides=deep_merge(overrides or {}, {
            "judge": {"near_tie_threshold": t}, "report": {"claims": False}, "verify": {"enabled": False}}))
        rows = _run_jds(cfg, specs, cvs, gold)
        table.append({"threshold": t, "ndcg@10": mean(r["ndcg@10"] for r in rows),
                      "kendall_tau": mean(r["kendall_tau"] for r in rows),
                      "judge_pairs": mean(r["judge_pairs"] for r in rows),
                      "wall_s": mean(r["wall_s"] for r in rows),
                      "fallbacks": sum(r["fallback"] for r in rows)})
    best = max(r["ndcg@10"] for r in table)
    ok = [r for r in table if r["ndcg@10"] >= best - tolerance and r["fallbacks"] == 0] or table
    chosen = min(ok, key=lambda r: (r["judge_pairs"], r["threshold"]))
    return {"split": split, "jds": [s["id"] for s in specs], "tolerance": tolerance, "table": table,
            "recommended_threshold": chosen["threshold"]}


def _summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task: Dict[str, List[float]] = {}
    out_tok: Dict[str, List[int]] = {}
    for r in rows:
        for task, lat, _p, c in r["call_latency"]:
            by_task.setdefault(task, []).append(lat)
            out_tok.setdefault(task, []).append(c)
    walls = [r["wall_s"] for r in rows]
    pairs = sum(r["judge_pairs"] for r in rows)
    return {
        "n_jds": len(rows), "wall_p50_s": percentile(walls, 50), "wall_p95_s": percentile(walls, 95),
        "wall_max_s": max(walls) if walls else float("nan"),
        "calls_per_jd": mean(r["llm_calls"] for r in rows), "tokens_per_jd": mean(r["tokens"] for r in rows),
        "fallbacks": sum(r["fallback"] for r in rows), "ndcg@10": mean(r["ndcg@10"] for r in rows),
        "kendall_tau": mean(r["kendall_tau"] for r in rows),
        "order_consistency": (sum(r["consistent_pairs"] for r in rows) / pairs) if pairs else float("nan"),
        "per_task": {t: {"n": len(v), "p50_s": percentile(v, 50), "p95_s": percentile(v, 95),
                         "mean_output_tokens": mean(out_tok[t])} for t, v in by_task.items()},
    }


def bench(models: Sequence[str], data_dir: Path, split: str = "all", jd_ids: Optional[Sequence[str]] = None,
          experiment: str = "P6", overrides: Optional[Dict] = None,
          out_root: Path = ROOT / "runs" / "bench") -> Path:
    specs, cvs, gold = _load(data_dir, split, jd_ids)
    run_dir = out_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {"commit": git_commit(), "experiment": experiment, "jds": [s["id"] for s in specs],
                              "n_cvs": len(cvs), "models": {}}
    for model in models:
        cfg = load_config(experiment, overrides=deep_merge(overrides or {}, {
            "llm": {"backend": "ollama", "model": model}, "paths": {"cache_dir": None}}))
        warm = build_runtime(cfg)  # load the model once so cold-start is not counted per JD
        try:
            warm.gateway.backend.complete_json("pairwise", "Return JSON.", '{"winner": "A"}', {})
        except Exception as exc:  # report, but still try the benchmark
            log.warning("Warm-up for %s failed: %s", model, exc)
        rows = _run_jds(cfg, specs, cvs, gold, run_dir / model.replace(":", "_"))
        report["models"][model] = {"summary": _summarise(rows), "per_jd": [
            {k: v for k, v in r.items() if k != "call_latency"} for r in rows], "config": {
            "near_tie_threshold": cfg["judge"]["near_tie_threshold"], "max_rank": cfg["judge"].get("max_rank"),
            "cv_chars": cfg["llm"].get("cv_chars"), "max_output_tokens": cfg["llm"].get("max_output_tokens"),
            "budget": cfg["budget"]}}
        (run_dir / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
    return run_dir
