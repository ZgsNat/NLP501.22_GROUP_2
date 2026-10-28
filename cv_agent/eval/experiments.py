"""Run the experiment configurations on the Gold Set and test hypotheses H0–H6 (D5)."""
from __future__ import annotations

import csv
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

import yaml

from cv_agent.config import ROOT, load_config
from cv_agent.criteria.clarify import simulated_hr
from cv_agent.criteria.jd_parser import parse_jd
from cv_agent.criteria.profiles import list_profiles
from cv_agent.eval.evidence_metrics import eraser_scores, evidence_precision, unsupported_claim_rate
from cv_agent.eval.fairness import impact_ratio
from cv_agent.eval.metrics import cohen_kappa, graded_tau, kendall_tau, mrr_at_k, ndcg_at_k, paired_bootstrap, \
    recall_at_k, wilcoxon_signed_rank
from cv_agent.ingest.pipeline import ingest_dir
from cv_agent.orchestration.graph import build_runtime, run_pipeline
from cv_agent.schema import CV, JD

log = logging.getLogger(__name__)
ALL_CONFIGS = ["B0", "B1", "B2", "B3", "B4", "LLM-all", "P6", "A1", "A2", "P6-noclar", "AG", "AG-rules"]
H3_EPSILON = 0.02   # should be re-chosen on the dev split
H5_TOP = 20
H5_MIN_IMPACT_RATIO = 0.8  # 4/5 rule, checked per JD
H7_MAX_PLANNER_FALLBACK = 0.2  # share of LLM planner decisions the rules had to replace
JUDGE_TASKS = ("pairwise", "listwise")


def load_jd_specs(path: Path = ROOT / "data" / "jds" / "jds.yaml") -> List[Dict[str, Any]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def specs_for(data_dir: Path) -> List[Dict[str, Any]]:
    """JDs shipped with a data folder (``<data>/jds.yaml``, e.g. public datasets) or the default JD set."""
    local = Path(data_dir) / "jds.yaml"
    return load_jd_specs(local) if local.exists() else load_jd_specs()


def jd_from_spec(spec: Dict[str, Any]) -> JD:
    return parse_jd(spec["text"], spec["id"], spec.get("title", ""), spec.get("company_type", "default"))


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=ROOT, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-git"


def _nan_mean(xs: Sequence[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return mean(xs) if xs else float("nan")


def ranking_metrics(ranking: List[str], gold: Dict[str, Any]) -> Dict[str, float]:
    """Recall on the full ranking; with incomplete labels (``judged_only``) MRR / nDCG are computed on the
    condensed list of labelled CVs and τ only over pairs with different grades (ties inside a grade are not
    an error). Fully labelled data (synthetic Gold Set) keeps τ against the HR order."""
    rel = gold["relevance"]
    if gold.get("judged_only"):
        cond = [cid for cid in ranking if cid in rel]
        tau = graded_tau(cond, rel)
    else:
        cond, tau = ranking, kendall_tau(ranking, gold["hr_ranking"])
    return {"recall@100": recall_at_k(ranking, rel, 100), "recall@10": recall_at_k(ranking, rel, 10),
            "mrr@10": mrr_at_k(cond, rel, 10), "ndcg@10": ndcg_at_k(cond, rel, 10), "kendall_tau": tau}


def evaluate_run(out: Dict[str, Any], gold: Dict[str, Any], meta: Dict[str, Any], cvs: Dict[str, CV],
                 jd: JD, with_eraser: bool = True) -> Dict[str, float]:
    ranking = [c.cv_id for c in out["ranking"]]
    claims = out.get("claims", {})
    row = {**ranking_metrics(ranking, gold),
           "llm_calls": out["llm_stats"]["calls"], "tokens": out["llm_stats"]["used_tokens"],
           # judge-only cost (arbiter / LLM-all) for H3, excluding claim generation
           "judge_calls": sum(out["llm_stats"]["by_task"].get(t, 0) for t in JUDGE_TASKS),
           "judge_tokens": sum(out["llm_stats"]["tokens_by_task"].get(t, 0) for t in JUDGE_TASKS),
           "latency_s": out["latency_s"], "fallback": int(bool(out.get("fallback"))),
           **agent_metrics(out),
           "unsupported_rate": unsupported_claim_rate(claims, cvs) if claims else float("nan"),
           "evidence_precision": evidence_precision(claims, meta.get("rationale", {})) if claims else float("nan")}
    groups = {cid: g["gender"] for cid, g in meta.get("groups", {}).items() if cid in cvs}
    row["impact_ratio"] = impact_ratio(ranking[:H5_TOP], groups)["impact_ratio"] if groups else float("nan")
    if with_eraser and claims and "profile" in out:
        er = [eraser_scores(cvs[cid], cl, jd, out["profile"]) for cid, cl in claims.items()]
        row["comprehensiveness"] = _nan_mean([e["comprehensiveness"] for e in er])
        row["sufficiency"] = _nan_mean([e["sufficiency"] for e in er])
    return row


def agent_metrics(out: Dict[str, Any]) -> Dict[str, float]:
    """Agent configs: steps, LLM planner decisions and how many the rules had to replace; NaN otherwise."""
    log_ = out.get("agent_log")
    if not log_:
        return {"agent_steps": float("nan"), "planner_fallback_rate": float("nan"), "goal_met": float("nan")}
    stats = out.get("planner_stats", {})
    asked = stats.get("llm", 0) + stats.get("fallback", 0)
    return {"agent_steps": len(log_), "planner_fallback_rate": stats.get("fallback", 0) / asked if asked else 0.0,
            "goal_met": float(bool((out.get("critic") or {}).get("passed")))}


def component_names(rt) -> Dict[str, str]:
    """Components actually loaded (a missing model falls back to lexical/hashing; a config may not use all)."""
    return {"encoder": rt.encoder.name, "reranker": rt.reranker.name, "nli": rt.nli.name}


def run_config(name: str, specs: List[Dict], cvs: List[CV], gold: Dict, meta: Dict, run_dir: Path,
               overrides: Optional[Dict] = None) -> List[Dict[str, Any]]:
    exp = "P6" if name == "P6-noclar" else name
    extra = {"criteria": {"clarification": {"enabled": False}}} if name == "P6-noclar" else {}
    cfg = load_config(exp, overrides={**extra, **(overrides or {})})
    by_id = {c.cv_id: c for c in cvs}
    shared = None
    rows = []
    profiles = list_profiles()
    for spec in specs:
        jd = jd_from_spec(spec)
        rt = build_runtime(cfg, run_dir / name / spec["id"], shared=shared,
                           hr_answer_fn=lambda _jd, qs, s=spec: simulated_hr(qs, s.get("hr_answers")))
        shared = shared or rt
        profile = None if cfg["criteria"]["profile"] == "Uniform" else profiles.get(spec.get("profile", ""))
        out = run_pipeline(rt, jd, cvs, profile)
        row = {"config": name, "jd_id": spec["id"], "split": spec.get("split", "test"),
               "ambiguous": int(bool(spec.get("ambiguous"))), **component_names(rt),
               **evaluate_run(out, gold[spec["id"]], meta, by_id, out["jd"])}
        rows.append(row)
        log.info("%s %s ndcg@10=%.3f tau=%.3f calls=%d", name, spec["id"], row["ndcg@10"], row["kendall_tau"],
                 row["llm_calls"])
    return rows


def _col(rows: List[Dict], cfg: str, metric: str, ambiguous: Optional[bool] = None) -> List[float]:
    sel = [r for r in rows if r["config"] == cfg and (ambiguous is None or bool(r["ambiguous"]) == ambiguous)]
    return [float(r[metric]) for r in sorted(sel, key=lambda r: r["jd_id"])]


def _cmp(rows, a, b, metric, ambiguous=None) -> Dict[str, Any]:
    x, y = _col(rows, a, metric, ambiguous), _col(rows, b, metric, ambiguous)
    if not x or not y:
        return {"missing": True}
    return {"a": a, "b": b, "metric": metric, "mean_a": _nan_mean(x), "mean_b": _nan_mean(y),
            **paired_bootstrap(x, y), "wilcoxon": wilcoxon_signed_rank(x, y)}


def _sig_pos(c: Dict) -> bool:
    return not c.get("missing") and c["ci_low"] > 0


def evaluate_hypotheses(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    h: Dict[str, Any] = {}
    h1 = {m: [_cmp(rows, "B2", "B0", m), _cmp(rows, "B2", "B1", m)] for m in ("recall@100", "mrr@10")}
    h1_ok = all(not c.get("missing") for cs in h1.values() for c in cs) and all(
        cs[0]["mean_a"] >= max(cs[0]["mean_b"], cs[1]["mean_b"]) - 1e-12 and (_sig_pos(cs[0]) or _sig_pos(cs[1]))
        for cs in h1.values())
    h["H1"] = {"accepted": h1_ok, "details": h1}
    c2 = _cmp(rows, "P6", "A2", "kendall_tau")
    h["H2"] = {"accepted": _sig_pos(c2), "details": c2}
    c3 = _cmp(rows, "P6", "LLM-all", "ndcg@10")
    calls = _cmp(rows, "P6", "LLM-all", "judge_calls")
    h["H3"] = {"accepted": (not c3.get("missing")) and c3["mean_diff"] >= -H3_EPSILON and calls["mean_diff"] < 0,
               "epsilon": H3_EPSILON, "details": {"ndcg@10": c3, "llm_calls": calls,
                                                  "judge_tokens": _cmp(rows, "P6", "LLM-all", "judge_tokens")}}
    c4 = _cmp(rows, "P6", "A1", "unsupported_rate")
    h["H4"] = {"accepted": (not c4.get("missing")) and c4["mean_diff"] < 0, "details": c4}
    h["H5"] = _h5(rows)
    c6 = _cmp(rows, "P6", "P6-noclar", "kendall_tau", ambiguous=True)
    h["H6"] = {"accepted": (not c6.get("missing")) and c6["mean_diff"] > 0, "details": c6}
    h["H7"] = _h7(rows)
    c0 = _cmp(rows, "P6", "B4", "ndcg@10")
    f0 = _cmp(rows, "P6", "B4", "unsupported_rate")
    over_budget = sum(r["fallback"] for r in rows if r["config"] == "P6")
    h["H0"] = {"accepted": (not c0.get("missing")) and (c0["mean_diff"] > 0 or f0.get("mean_diff", 0) < 0)
               and over_budget == 0, "details": {"ndcg@10": c0, "unsupported_rate": f0, "p6_fallbacks": over_budget}}
    return h


def _h7(rows: List[Dict]) -> Dict[str, Any]:
    """H7: an LLM planner (AG) is at least as good as the rule planner (AG-rules) with the same tools, goals and
    budget — nDCG@10 not lower than ε, unsupported claims not higher — and the rules rarely have to step in."""
    nd, un = _cmp(rows, "AG", "AG-rules", "ndcg@10"), _cmp(rows, "AG", "AG-rules", "unsupported_rate")
    if nd.get("missing"):
        return {"accepted": False, "details": {"missing": True}}
    fb = _nan_mean(_col(rows, "AG", "planner_fallback_rate"))
    ok = nd["mean_diff"] >= -H3_EPSILON and (un.get("missing") or not un["mean_diff"] > 0)         and not fb > H7_MAX_PLANNER_FALLBACK
    return {"accepted": bool(ok), "planner_fallback_rate": fb, "details": {
        "ndcg@10": nd, "unsupported_rate": un, "llm_calls": _cmp(rows, "AG", "AG-rules", "llm_calls"),
        "agent_steps": _cmp(rows, "AG", "AG-rules", "agent_steps"), "goal_met": _cmp(rows, "AG", "AG-rules", "goal_met")}}


def _h5(rows: List[Dict]) -> Dict[str, Any]:
    """4/5 rule must hold for EVERY JD (a good mean can hide a failing JD)."""
    per_jd = {r["jd_id"]: float(r["impact_ratio"]) for r in rows if r["config"] == "P6"}
    valid = {j: v for j, v in per_jd.items() if not math.isnan(v)}
    failing = sorted(j for j, v in valid.items() if v < H5_MIN_IMPACT_RATIO)
    return {"accepted": bool(valid) and not failing, "impact_ratio_by_jd": per_jd, "failing_jds": failing,
            "impact_ratio_min": min(valid.values()) if valid else float("nan"),
            "impact_ratio_mean": _nan_mean(list(valid.values())),
            "note": "Δrank khi đổi tên: chạy `python -m cv_agent audit`"}


def kappa_report(gold: Dict[str, Any]) -> Dict[str, float]:
    return {jd: cohen_kappa(g["annotator_a"], g["annotator_b"]) for jd, g in gold.items() if "annotator_b" in g}


def run_all(configs: Sequence[str], data_dir: Path, split: str = "all", out_root: Path = ROOT / "runs" / "experiments",
            overrides: Optional[Dict] = None) -> Path:
    gold = json.loads((data_dir / "gold.json").read_text(encoding="utf-8"))
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    specs = [s for s in specs_for(data_dir) if s["id"] in gold and (split == "all" or s.get("split") == split)]
    cvs = ingest_dir(data_dir / "cvs")
    run_dir = out_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for name in configs:
        rows.extend(run_config(name, specs, cvs, gold, meta, run_dir, overrides))
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("config", "jd_id"), k))
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    p6_cfg = load_config("P6", overrides=overrides)
    summary = {"commit": git_commit(), "split": split, "configs": list(configs), "n_jds": len(specs),
               "n_cvs": len(cvs), "synthetic": meta.get("synthetic", False),
               "llm_backend": p6_cfg["llm"],
               "components_requested": {"dense_encoder": p6_cfg["retrieval"]["dense"]["encoder"],
                                        "reranker": p6_cfg["rerank"]["reranker"], "nli": p6_cfg["verify"]["nli"]},
               "components_loaded": {c: sorted({r[k] for r in rows if r["config"] == c for k in ("encoder", "reranker", "nli")})
                                   for c in configs},
               "cohen_kappa": kappa_report(gold), "hypotheses": evaluate_hypotheses(rows),
               "means": {c: {m: _nan_mean(_col(rows, c, m)) for m in ("recall@100", "mrr@10", "ndcg@10", "kendall_tau",
                                                                      "llm_calls", "judge_calls", "tokens", "unsupported_rate",
                                                                      "evidence_precision", "impact_ratio", "latency_s", "agent_steps",
                                                                      "planner_fallback_rate", "goal_met")}
                         for c in configs}}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    return run_dir
