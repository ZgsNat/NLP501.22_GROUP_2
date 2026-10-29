"""Command line entry point: ``python -m cv_agent <command>``."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from cv_agent.config import ROOT, deep_merge, load_config


def _new_run_dir(cfg: dict) -> Path:
    run_dir = ROOT / cfg["paths"]["runs_dir"] / f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg['experiment']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_run(out: dict, cfg: dict, run_dir: Path) -> Path:
    payload = {"run_id": out["run_id"], "experiment": cfg["experiment"], "jd_id": out["jd"].jd_id,
               "profile": asdict(out["profile"]) if out.get("profile") else None,
               "ranking": [asdict(c) for c in out["ranking"]], "knocked_out": out.get("knocked_out", []),
               "cards": [asdict(c) for c in out.get("cards", [])], "trace": out.get("trace", []),
               "verify_stats": out.get("verify_stats"), "llm_stats": out.get("llm_stats"),
               "fallback": out.get("fallback", False), "latency_s": out.get("latency_s"), "config": cfg}
    (run_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return run_dir


def cmd_generate(args) -> int:
    from cv_agent.data.synthetic import generate
    from cv_agent.eval.experiments import load_jd_specs
    specs = {s["id"]: {**s["gold_spec"], "ambiguous": s.get("ambiguous", False), "split": s.get("split", "test")}
             for s in load_jd_specs()}
    print(generate(Path(args.out), specs, n=args.n, seed=args.seed))
    return 0


def cmd_rank(args) -> int:
    from cv_agent.criteria.jd_parser import parse_jd
    from cv_agent.criteria.profiles import list_profiles
    from cv_agent.eval.experiments import jd_from_spec, load_jd_specs, specs_for
    from cv_agent.ingest.pipeline import ingest_dir
    from cv_agent.orchestration.graph import build_runtime, run_pipeline

    cfg = load_config(args.config, overrides=_llm_overrides(args))
    specs = {s["id"]: s for s in load_jd_specs()} | {s["id"]: s for s in specs_for(Path(args.cvs).parent)}
    if args.jd in specs:
        jd, profile_name = jd_from_spec(specs[args.jd]), specs[args.jd].get("profile")
    else:
        jd, profile_name = parse_jd(Path(args.jd).read_text(encoding="utf-8"), Path(args.jd).stem), None
    profile_name = args.profile or profile_name
    profile = list_profiles().get(profile_name) if profile_name and cfg["criteria"]["profile"] != "Uniform" else None
    cvs = ingest_dir(Path(args.cvs))
    run_dir = _new_run_dir(cfg)  # llm_calls.jsonl (FR-31) + result.json
    out = run_pipeline(build_runtime(cfg, run_dir), jd, cvs, profile)
    _save_run(out, cfg, run_dir)
    for line in out["trace"]:
        print("·", line)
    print(f"\nTop {args.top} – {jd.title} [{cfg['experiment']}]")
    for i, c in enumerate(out["ranking"][: args.top], 1):
        ko = "" if c.must_have_ok else "  (thiếu must-have)"
        print(f"{i:>2}. {c.cv_id:<14} score={c.score:.4f}{ko}")
    print(f"\nLLM: {out['llm_stats']}\nSaved: {run_dir / 'result.json'}")
    return 0


def cmd_export_cvs(args) -> int:
    """FR-01: write each ingested CV as canonical JSON (masked text only, no raw text)."""
    from cv_agent.ingest.pipeline import ingest_dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cvs = ingest_dir(Path(args.cvs))
    for cv in cvs:
        (out_dir / f"{cv.cv_id}.json").write_text(
            json.dumps(cv.to_public_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"Exported {len(cvs)} canonical CV(s) to {out_dir}")
    return 0


def cmd_fetch_datasets(args) -> int:
    """Download public datasets (Resume–Job Fit, SkillSpan) into the data/synthetic layout."""
    from cv_agent.data.public import fetch_resume_job_fit, fetch_skillspan
    raw = Path(args.raw)
    meta = fetch_resume_job_fit(Path(args.out), raw, args.min_qa_signals)
    print(f"Resume–Job Fit: {meta['n_jds']} JD QA/Automation, {meta['n_cvs']} CV, {meta['n_pairs']} cặp có nhãn "
          f"-> {args.out}  (licence: {meta['license']}; chỉ dùng cho nghiên cứu, không phân phối lại)")
    for path in fetch_skillspan(raw):
        print(f"SkillSpan: {path}")
    return 0


def cmd_eval_spans(args) -> int:
    """Span-level P/R/F1 of the entity extractor on SkillSpan (FR-03)."""
    from cv_agent.data.public import fetch_skillspan
    from cv_agent.eval.span_metrics import load_skillspan, score_sentences, taxonomy_predictor
    paths = [p for p in fetch_skillspan(Path(args.raw)) if p.name.endswith(f"{args.split}.json")]
    predict = taxonomy_predictor
    if args.hybrid and not args.tagger:
        print("--hybrid cần --tagger <thư mục tagger>.")
        return 2
    if args.tagger:
        from cv_agent.training.skill_tagger import load_predictor, union_predictor
        tagger = load_predictor(Path(args.tagger))
        predict = union_predictor(tagger, taxonomy_predictor) if args.hybrid else tagger
    res = score_sentences((s for p in paths for s in load_skillspan(p)), predict=predict)
    print(json.dumps({"split": args.split, **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}},
                     ensure_ascii=False, indent=1))
    return 0


def cmd_train_tagger(args) -> int:
    """Fine-tune the open-vocabulary skill tagger on SkillSpan and compare it with the taxonomy extractor."""
    from cv_agent.training.skill_tagger import DEFAULT_BASE, TaggerConfig, run
    base = args.base_model or DEFAULT_BASE
    cfg = TaggerConfig(base_model=base, out_dir=Path(args.out_dir) / f"{base.split('/')[-1]}-skillspan-tagger",
                       epochs=args.epochs, batch_size=args.batch_size, max_len=args.max_len, seed=args.seed,
                       limit=args.limit)
    report = run(cfg, Path(args.raw))
    for name, res in report["test"].items():
        print(f"{name}: test P/R/F1 exact {res['precision_exact']:.3f} / {res['recall_exact']:.3f} / "
              f"{res['f1_exact']:.3f}, overlap F1 {res['f1_overlap']:.3f}")
    return 0


def cmd_experiments(args) -> int:
    from cv_agent.eval.experiments import ALL_CONFIGS, run_all
    configs = ALL_CONFIGS if args.configs == "all" else args.configs.split(",")
    overrides = deep_merge(deep_merge(_llm_overrides(args), NEURAL_OVERRIDES if args.neural else {}),
                           {"rerank": {"mode": args.rerank_mode}} if args.rerank_mode else {})
    overrides = deep_merge(overrides, model_overrides(args))
    run_dir = run_all(configs, Path(args.data), args.split, overrides=overrides or None)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"Results: {run_dir}")
    for h, v in sorted(summary["hypotheses"].items()):
        print(f"  {h}: {'CHẤP NHẬN' if v['accepted'] else 'BÁC BỎ'}")
    return 0


def cmd_audit(args) -> int:
    from cv_agent.criteria.profiles import list_profiles
    from cv_agent.eval.experiments import jd_from_spec, specs_for
    from cv_agent.eval.fairness import name_swap_delta
    from cv_agent.ingest.pipeline import ingest_text
    from cv_agent.orchestration.graph import build_runtime, run_pipeline

    data = Path(args.data)
    meta = json.loads((data / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("names"):
        print(f"Audit đổi tên cần meta.json có 'names' (tên + giới tính từng CV); {data} không có -> bỏ qua.")
        return 2
    specs = {s["id"]: s for s in specs_for(data)}
    if args.jd not in specs:
        print(f"Không có JD '{args.jd}' trong {data}. Có: {', '.join(sorted(specs))}")
        return 2
    spec = specs[args.jd]
    cfg = load_config(args.config, overrides={"llm": {"backend": args.llm}} if args.llm else None)
    profile = list_profiles().get(spec.get("profile"))
    raw = {p.stem: p.read_text(encoding="utf-8") for p in sorted((data / "cvs").glob("*.txt"))
           if p.stem in meta["names"]}

    def rank_fn(texts):
        cvs = [ingest_text(t, cid) for cid, t in texts.items()]
        out = run_pipeline(build_runtime(cfg), jd_from_spec(spec), cvs, profile)
        return [c.cv_id for c in out["ranking"]]

    res = name_swap_delta(rank_fn, raw, meta["names"])
    print(json.dumps({"jd": args.jd, "config": args.config, **res}, ensure_ascii=False, indent=1))
    return 0


# FR-12/15/22 with pretrained models instead of the dependency-free fallbacks (needs requirements-ml.txt)
NEURAL_OVERRIDES = {"retrieval": {"dense": {"encoder": "sentence_transformers"}},
                    "rerank": {"reranker": "cross_encoder"}, "verify": {"nli": "hf"}}


def _llm_overrides(args) -> dict:
    llm = {}
    if getattr(args, "llm", None):
        llm["backend"] = args.llm
    if getattr(args, "model", None):
        llm["model"] = args.model
    return {"llm": llm} if llm else {}


def model_overrides(args) -> dict:
    """--dense-model / --rerank-model: e.g. a fine-tuned model folder from notebooks/finetune_colab.ipynb."""
    out: dict = {}  # a model path only makes sense for the neural component, so it switches that one on
    if getattr(args, "dense_model", None):
        out["retrieval"] = {"dense": {"encoder": "sentence_transformers", "model": args.dense_model}}
    if getattr(args, "rerank_model", None):
        out["rerank"] = {"reranker": "cross_encoder", "model": args.rerank_model}
    return out


def cmd_finetune(args) -> int:
    from cv_agent.training import finetune as ft
    if args.target == "both" and args.base_model:
        print("--base-model chỉ dùng với --target dense hoặc --target rerank (hai mô hình gốc khác nhau).")
        return 2
    try:
        seeds = [int(s) for s in args.seeds.split(",")]
    except ValueError:
        print(f"--seeds phải là danh sách số nguyên, ví dụ 42,7,13 (nhận được {args.seeds!r}).")
        return 2
    if len(set(seeds)) != len(seeds):
        print(f"--seeds bị trùng ({args.seeds}): mỗi seed chỉ chạy một lần, nếu không độ lệch chuẩn sẽ sai.")
        return 2
    targets = ["dense", "rerank"] if args.target == "both" else [args.target]
    reports = []
    for target in targets:
        base = args.base_model or ft.DEFAULT_BASE[target]
        for seed in seeds:
            out = Path(args.out_dir) / ft.model_dir_name(base, seed)
            cfg = ft.TrainConfig(target=target, base_model=base, out_dir=out, epochs=args.epochs,
                                 batch_size=args.batch_size, max_len=args.max_len, seed=seed, limit=args.limit)
            report = ft.run(cfg, Path(args.raw))
            reports.append(report)
            print(f"{target} seed {seed}: {out} | nDCG@10 trước -> sau: " + ", ".join(
                f"{k} {report['before'][k]['ndcg@10']:.3f} -> {report['after'][k]['ndcg@10']:.3f}"
                for k in report["after"]))
    if len(seeds) > 1:
        print(json.dumps(ft.seed_summary(reports), indent=1))
    return 0


def cmd_error_analysis(args) -> int:
    """Per-JD breakdown, near-duplicate JDs and de-duplicated means of a finished run (post hoc)."""
    from cv_agent.eval.error_analysis import analyse_run
    from cv_agent.eval.experiments import specs_for
    report = analyse_run(Path(args.run), specs_for(Path(args.data)))
    print(f"{args.run}: JD trùng lặp {report['near_duplicates'] or 'không có'}; "
          f"số JD hiệu dụng {report['effective_n_jds']}")
    for name, means in {**report["groups"], "dedup": report["dedup"]}.items():
        print(f"  {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in means.items() if k != "n_jds"))
    return 0


def cmd_rescore(args) -> int:
    from cv_agent.eval.experiments import rescore
    failed = 0
    for run in args.runs:
        try:
            summary = rescore(Path(run))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            failed += 1
            print(f"{run}: bỏ qua, không đọc được kết quả ({exc})")
            continue
        print(f"{run}: " + ", ".join(f"{h} {'CHẤP NHẬN' if v['accepted'] else 'BÁC BỎ'}"
                                      for h, v in sorted(summary["hypotheses"].items())))
    return 1 if failed else 0


def cmd_tune(args) -> int:
    from cv_agent.eval.perf import EXACT_TIE, tune_threshold
    thresholds = [EXACT_TIE if t == "tie" else float(t) for t in args.thresholds.split(",")]
    res = tune_threshold(thresholds, Path(args.data), args.split, args.tolerance, _llm_overrides(args))
    out = ROOT / "runs" / "tune" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'threshold':>10} {'nDCG@10':>8} {'tau':>7} {'pairs/JD':>9} {'wall_s':>8} fallback")
    for r in res["table"]:
        print(f"{r['threshold']:>10.3g} {r['ndcg@10']:>8.4f} {r['kendall_tau']:>7.4f} {r['judge_pairs']:>9.1f} "
              f"{r['wall_s']:>8.1f} {r['fallbacks']}")
    print(f"Đề xuất near_tie_threshold = {res['recommended_threshold']:g} (split={res['split']}, JD={res['jds']})")
    print(f"Saved: {out}")
    return 0


def cmd_bench(args) -> int:
    from cv_agent.eval.perf import bench
    jd_ids = args.jds.split(",") if args.jds else None
    overrides = {"judge": {"near_tie_threshold": args.threshold}} if args.threshold is not None else {}
    run_dir = bench(args.models.split(","), Path(args.data), args.split, jd_ids, args.config, overrides)
    report = json.loads((run_dir / "bench.json").read_text(encoding="utf-8"))
    for model, rep in report["models"].items():
        s = rep["summary"]
        print(f"{model}: JD p50={s['wall_p50_s']:.0f}s p95={s['wall_p95_s']:.0f}s | calls/JD={s['calls_per_jd']:.1f} "
              f"| fallback={s['fallbacks']} | nDCG@10={s['ndcg@10']:.3f} | nhất quán đảo thứ tự={s['order_consistency']:.2f}")
        for task, t in s["per_task"].items():
            print(f"    {task:<9} n={t['n']:<3} p50={t['p50_s']:.1f}s p95={t['p95_s']:.1f}s out_tokens~{t['mean_output_tokens']:.0f}")
    print(f"Saved: {run_dir / 'bench.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cv_agent", description="Agentic CV search for Automation Test roles")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate-data", help="generate synthetic CVs + gold labels")
    g.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=42)
    g.set_defaults(fn=cmd_generate)
    r = sub.add_parser("rank", help="rank CVs for one JD")
    r.add_argument("--jd", default="fpt_automation_test", help="JD id from data/jds/jds.yaml or a text file")
    r.add_argument("--cvs", default=str(ROOT / "data" / "synthetic" / "cvs"))
    r.add_argument("--config", default="P6")
    r.add_argument("--profile", default=None)
    r.add_argument("--llm", default=None, help="override llm backend: ollama | heuristic | openai_compat")
    r.add_argument("--model", default=None, help="override llm model, e.g. qwen2.5:3b for CPU-only laptops")
    r.add_argument("--top", type=int, default=10)
    r.set_defaults(fn=cmd_rank)
    x = sub.add_parser("export-cvs", help="ingest a CV folder and save canonical JSON (FR-01)")
    x.add_argument("--cvs", default=str(ROOT / "data" / "synthetic" / "cvs"))
    x.add_argument("--out", default=str(ROOT / "runs" / "canonical_cvs"))
    x.set_defaults(fn=cmd_export_cvs)
    f = sub.add_parser("fetch-datasets", help="download public datasets (Resume–Job Fit QA subset, SkillSpan)")
    f.add_argument("--out", default=str(ROOT / "data" / "public"))
    f.add_argument("--raw", default=str(ROOT / "data" / "public" / "raw"))
    f.add_argument("--min-qa-signals", type=int, default=2, help="distinct QA terms a JD needs to be kept")
    f.set_defaults(fn=cmd_fetch_datasets)
    sp = sub.add_parser("eval-spans", help="span P/R/F1 of the entity extractor on SkillSpan (FR-03)")
    sp.add_argument("--raw", default=str(ROOT / "data" / "public" / "raw"))
    sp.add_argument("--split", default="test", choices=["dev", "test"])
    sp.add_argument("--tagger", default=None, help="folder of a tagger from train-tagger (default: taxonomy extractor)")
    sp.add_argument("--hybrid", action="store_true", help="with --tagger: union of tagger and taxonomy spans")
    sp.set_defaults(fn=cmd_eval_spans)
    tg = sub.add_parser("train-tagger", help="fine-tune a BIO skill/knowledge tagger on SkillSpan (FR-03)")
    tg.add_argument("--base-model", default=None, help="default: jjzha/jobbert-base-cased")
    tg.add_argument("--out-dir", default=str(ROOT / "models"))
    tg.add_argument("--raw", default=str(ROOT / "data" / "public" / "raw"))
    tg.add_argument("--epochs", type=int, default=3)
    tg.add_argument("--batch-size", type=int, default=16)
    tg.add_argument("--max-len", type=int, default=128)
    tg.add_argument("--seed", type=int, default=42)
    tg.add_argument("--limit", type=int, default=None, help="smoke run: cap sentences per split")
    tg.set_defaults(fn=cmd_train_tagger)
    e = sub.add_parser("experiments", help="run configurations and test H0-H6")
    e.add_argument("--configs", default="all")
    e.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    e.add_argument("--split", default="all", choices=["all", "dev", "test"])
    e.add_argument("--llm", default=None)
    e.add_argument("--model", default=None, help="override llm model, e.g. qwen2.5:3b")
    e.add_argument("--neural", action="store_true", help="pretrained dense encoder, cross-encoder reranker, NLI")
    e.add_argument("--rerank-mode", default=None, choices=["filter", "within_criteria"],
                   help="filter = FR-15 top_n cut before criteria; within_criteria = reranker only breaks ties")
    e.add_argument("--dense-model", default=None, help="dense encoder name or local folder (fine-tuned model)")
    e.add_argument("--rerank-model", default=None, help="cross-encoder name or local folder (fine-tuned model)")
    e.set_defaults(fn=cmd_experiments)
    f = sub.add_parser("finetune", help="ConFit-style fine-tuning on public pairs (GPU/Colab recommended)")
    f.add_argument("--target", default="both", choices=["dense", "rerank", "both"])
    f.add_argument("--base-model", default=None, help="default: multilingual-e5-base / bge-reranker-base")
    f.add_argument("--out-dir", default=str(ROOT / "models"))
    f.add_argument("--raw", default=str(ROOT / "data" / "public" / "raw"))
    f.add_argument("--epochs", type=int, default=1)
    f.add_argument("--batch-size", type=int, default=16)
    f.add_argument("--max-len", type=int, default=256, help="tokens per text; 512 is better on a GPU")
    f.add_argument("--limit", type=int, default=None, help="smoke run: cap training examples and eval JDs")
    f.add_argument("--seeds", default="42", help="comma-separated seeds; several give mean ± sd of the gain")
    f.set_defaults(fn=cmd_finetune)
    ea = sub.add_parser("error-analysis", help="per-JD breakdown and duplicate JDs of a finished run (post hoc)")
    ea.add_argument("run", help="runs/experiments/<timestamp> folder")
    ea.add_argument("--data", default=str(ROOT / "data" / "public"), help="data folder of that run (for JD texts)")
    ea.set_defaults(fn=cmd_error_analysis)
    rs = sub.add_parser("rescore", help="re-test H0-H7 of finished runs from results.csv (no model re-run)")
    rs.add_argument("runs", nargs="+", help="runs/experiments/<timestamp> folders")
    rs.set_defaults(fn=cmd_rescore)
    a = sub.add_parser("audit", help="fairness name-swap audit (H5)")
    a.add_argument("--jd", default="fpt_automation_test")
    a.add_argument("--config", default="B3")
    a.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    a.add_argument("--llm", default=None)
    a.set_defaults(fn=cmd_audit)
    t = sub.add_parser("tune", help="sweep the near-tie threshold on the dev split (arbiter only)")
    t.add_argument("--thresholds", default="tie,0.005,0.01,0.02,0.05", help="'tie' = exact ties only")
    t.add_argument("--split", default="dev", choices=["all", "dev", "test"])
    t.add_argument("--tolerance", type=float, default=0.01, help="allowed nDCG@10 drop vs the best threshold")
    t.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    t.add_argument("--llm", default=None)
    t.add_argument("--model", default=None)
    t.set_defaults(fn=cmd_tune)
    b = sub.add_parser("bench", help="p50/p95 latency of a configuration with Ollama, LLM cache disabled")
    b.add_argument("--models", default="qwen2.5:7b,qwen2.5:3b")
    b.add_argument("--config", default="P6")
    b.add_argument("--jds", default=None, help="comma-separated JD ids (default: --split)")
    b.add_argument("--split", default="all", choices=["all", "dev", "test"])
    b.add_argument("--threshold", type=float, default=None, help="override judge.near_tie_threshold")
    b.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    b.set_defaults(fn=cmd_bench)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Vietnamese output on Windows consoles / redirects
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    return args.fn(args)
