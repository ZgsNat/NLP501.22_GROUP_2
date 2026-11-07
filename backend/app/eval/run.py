"""Evaluation harness: every system ranks all 50 CVs for all 12 JDs and is scored against silver labels.

    cd backend && uv run python -m app.eval.run retrieval            # B0–B3 + chunking ablation (minutes)
    cd backend && uv run python -m app.eval.run retrieval --dense intfloat/multilingual-e5-large
    cd backend && uv run python -m app.eval.run agent --systems B4,P6,A1,A2   # LLM systems (slow; LLM cache on)
    cd backend && uv run python -m app.eval.run report               # → docs/EVAL_RESULTS.md
    cd backend && uv run python -m app.eval.bench fetch && uv run python -m app.eval.run retrieval --bench coding

Ladder (CLAUDE.md): B0 BM25 → B1 dense → B2 hybrid → B3 +criteria (rules, no LLM judgement) → B4 fixed LLM pipeline
→ P6 full agent. Ablations: A1 = P6 − verify_evidence, A2 = P6 − company profile. Chunking: whole / window /
structural with requirement-wise retrieval, with and without the cross-encoder.
Labels: data/eval/annotator_{a,b}.yaml (0–3); gold = their mean. Results: .runtime/eval/results/<system>.json"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import observability
from app.agent import nodes
from app.agent.graph import build_graph
from app.agent.scoring import CORE4, knockout_reason, rule_total
from app.config import ROOT, Settings, get_settings
from app.eval import metrics
from app.eval.bench import OUR_TRAPS, Bench
from app.eval.bench import load as load_bench
from app.nlp.pipeline import process
from app.providers import cache
from app.providers.llm import create_llm, default_spec
from app.rag import retrieve
from app.rag.corpus import load_dir, load_yaml
from app.rag.store import KnowledgeStore

OUT = get_settings().runtime_dir / "eval"
LABELS = ROOT / "data" / "eval"
HR_ANSWERS = {"JD-003": "Must-have: Selenium with Java. At least 3 years. English is required for client calls."}
TRAPS = sorted(OUR_TRAPS)  # keyword stuffing, hedges, title inflation
AGENT_SYSTEMS = {"B4": ["clarify", "expand", "verify", "arbiter", "investigate"], "P6": [], "A1": ["verify"],
                 "A2": ["profile"], "A3": ["arbiter"], "A4": ["investigate"]}


# ── labels ────────────────────────────────────────────────────────────────────────────────────────────
def annotators() -> dict[str, dict[str, dict[str, int]]]:
    return {p.stem.removeprefix("annotator_"): yaml.safe_load(p.read_text()) for p in sorted(LABELS.glob("annotator_*.yaml"))}


def gold() -> dict[str, dict[str, float]]:
    ann = list(annotators().values())
    return {jd: {cv: sum(a[jd][cv] for a in ann) / len(ann) for cv in ann[0][jd]} for jd in ann[0]}


# ── shared inputs ─────────────────────────────────────────────────────────────────────────────────────
async def requirements(jd, mode: str = "llm", bench: str = "ours") -> dict:
    """JD → requirements with the agent's own extraction (Settings.jd_parser modes), cached on disk so every system
    sees the same input. The LLM answers come from the LLM cache, so recomputing is free once warm."""
    path = OUT / "requirements" / bench / mode / f"{jd.id}.json"
    if path.exists():
        return json.loads(path.read_text())
    req, subs, _ = await nodes.extract_requirements(create_llm(default_spec("reasoning")), jd.text, mode)
    out = {"req": req.model_dump(), "subs": subs,
           "terms": list(dict.fromkeys(req.must_have + req.nice_to_have + req.languages))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def full_order(ranked: list[str], pool: list[str]) -> list[str]:
    return list(dict.fromkeys(ranked + pool))  # unretrieved CVs go last, in a fixed order


# ── retrieval-level systems (no LLM judgement) ────────────────────────────────────────────────────────
def b3_systems(store: KnowledgeStore, pool: list[str], prof: dict, r: dict, w: dict) -> dict[str, list[str]]:
    """B3 family on one JD: hedge-aware knock-out + rule criteria (CORE4 = v2, or all 8), alone or fused with the
    cross-encoder over the same requirement-wise retrieval the agent uses. No LLM judgement."""
    req, subs = r["req"], r["subs"]
    qs = retrieve.queries(req, r["terms"], subs)
    hits = retrieve.rank_cvs(store, "chunks_structural", qs, limit=max(200, 2 * len(pool)))
    ce = retrieve.rerank_cvs(store, qs, [h for h in hits if h.cv_id in prof], prof)
    ko = {c for c in pool if knockout_reason(prof[c], req["must_have"], subs)}
    out = {}
    for crit_name, crit in (("core4", CORE4), ("8-axis", None)):
        rule = {c: rule_total(prof[c], req, subs, w, criteria=crit) for c in pool}
        out[crit_name] = sorted(pool, key=lambda c: (c not in ko, rule[c], ce.get(c, -99)), reverse=True)
        # ⊕: what the agent's rerank node orders by — eligible CVs by the mean of min-max normalised CE and rule
        # scores, then knocked-out ones by rule score
        ok = [c for c in ce if c not in ko]
        norm = lambda xs: {k: (v - min(xs.values())) / ((max(xs.values()) - min(xs.values())) or 1) for k, v in xs.items()}  # noqa: E731
        ce_n, ru_n = (norm({c: ce[c] for c in ok}), norm({c: rule[c] for c in ok})) if ok else ({}, {})
        fused = sorted(ok, key=lambda c: ce_n[c] + ru_n[c], reverse=True)
        out[f"{crit_name} ⊕ CE"] = fused + sorted((c for c in pool if c not in ok), key=lambda c: (c not in ko, rule[c]),
                                                  reverse=True)
    return out


def retrieval_systems(dense_model: str, b: Bench, only: set[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Run retrieval systems, optionally restricted to exact result names.

    ``only`` is primarily for the frozen external-validity run: loading the three
    chunking ablations and every cross-encoder variant at once needlessly raises
    peak RAM, while each requested system is still computed by the same code path.
    An omitted/empty set preserves the historical all-systems run.
    """
    only = only or set()
    wanted = lambda name: not only or name in only  # noqa: E731
    base_names = {"B0 BM25", "B1 dense", "B2 hybrid"}
    need_whole = any(wanted(name) for name in base_names)
    need_b3 = not only or any(name.startswith("B3 ") for name in only)
    need_ablation = not only
    s = Settings(qdrant_url=":memory:", embed_dense_model=dense_model)
    store = KnowledgeStore(s)
    pool = [d.id for d in b.cvs]
    styles = ("structural", "window", "whole") if need_ablation else (("structural",) if need_b3 else ())
    profiles = {st: {d.id: process(d.id, d.text, d.meta.get("headline", ""), st) for d in b.cvs} for st in styles}
    if need_whole:
        store.index(retrieve.WHOLE, b.cvs)
    for st, ps in profiles.items():
        store.index_items(f"chunks_{st}", retrieve.chunk_items(list(ps.values())))
    weights_of = load_yaml("criteria_profiles.yaml")["profiles"]

    out: dict[str, dict[str, list[str]]] = {}
    for jd in b.jds:
        add = lambda name, ids: out.setdefault(name, {}).__setitem__(jd.id, full_order(ids, pool))  # noqa: E731
        for mode, name in (("bm25", "B0 BM25"), ("dense", "B1 dense"), ("hybrid", "B2 hybrid")):
            if wanted(name):
                add(name, [h.id for h in store.hybrid_search(retrieve.WHOLE, jd.text, limit=len(pool), mode=mode)])
        if need_ablation:
            r = asyncio.run(requirements(jd, "llm", b.name))
            req, subs = r["req"], r["subs"]
            qs = retrieve.queries(req, r["terms"], subs)
            for st, weighted in (("whole", False), ("window", False), ("structural", False), ("structural", True)):
                hits = retrieve.rank_cvs(store, f"chunks_{st}", qs, limit=max(200, 2 * len(pool)), section_weighted=weighted)
                name = f"R {st}" + (" (section-weighted)" if weighted else "")
                add(name, [h.cv_id for h in hits])
                ce = retrieve.rerank_cvs(store, qs, hits, profiles[st])
                add(f"{name} + CE", sorted(ce, key=ce.get, reverse=True))
        if need_b3:
            w = weights_of[jd.meta.get("company_profile", "default")]["weights"]
            modes = ("llm", "rules", "hybrid") if need_ablation else tuple(
                mode for mode in ("llm", "rules", "hybrid") if any(name.endswith(f"[{mode}]") for name in only)
            )
            for mode in modes:
                rm = asyncio.run(requirements(jd, mode, b.name))
                for variant, ids in b3_systems(store, pool, profiles["structural"], rm, w).items():
                    if (mode, variant) == ("llm", "core4"):
                        name = "B3 +criteria (rules)"
                    elif (mode, variant) == ("llm", "core4 ⊕ CE"):
                        name = "B3 rules ⊕ CE"
                    else:
                        name = f"B3 {variant} [{mode}]"
                    if wanted(name):
                        add(name, ids)
    return out


# ── agent systems (in-process graph, scripted HR) ─────────────────────────────────────────────────────
async def run_agent(jd, ablate: list[str], model: str | None = None) -> dict:
    graph = build_graph(InMemorySaver())
    usage = observability.UsageCounter()
    cfg = {"configurable": {"thread_id": f"eval-{jd.id}-{time.time()}"}, "callbacks": [usage]}
    inp = {"jd_text": jd.text, "company_profile": jd.meta.get("company_profile", "default"), "ablate": ablate}
    if model:  # one "<provider>:<model>" for both roles, persisted in state so resumes use the same brain
        inp["models"] = {"reasoning": model, "fast": model}
    first_verify, t0, events = None, time.perf_counter(), []
    while True:
        async for mode, chunk in graph.astream(inp, cfg, stream_mode=["custom"]):
            events.append(chunk)
            if chunk.get("node") == "verify_evidence" and "grounded" in chunk.get("message", "") and not first_verify:
                first_verify = chunk["message"]
        state = await graph.aget_state(cfg)
        if not state.next:
            break
        pending = state.tasks[0].interrupts[0].value
        if pending["kind"] == "clarification":
            answer = jd.meta.get("hr_answer") or HR_ANSWERS.get(jd.id, "Use the JD as written; be strict on the named tools.")
            inp = Command(resume={"answers": {q: answer for q in pending["payload"]["questions"]}})
        elif pending["kind"] == "substitutes":
            inp = Command(resume={"accept": {p["must_have"]: p["related"] for p in pending["payload"]["proposals"]}})
        else:
            inp = Command(resume={"action": "approve"})
    v = state.values
    ev = [e for a in v.get("assessments", {}).values() for c in a.get("llm_review", {}).values() for e in c["evidence"]]
    u = usage.summary()
    return {"ranking": [r["cv_id"] for r in v["ranking"]], "seconds": round(time.perf_counter() - t0, 1),
            "calls": u["calls"], "tokens": u["total_tokens"], "loops": {k: n for k, n in v.get("loops", {}).items() if n},
            "first_verify": first_verify, "evidence": len(ev),
            "grounded": sum(bool(e["grounded"]) for e in ev), "supported": sum(bool(e["supported"]) for e in ev),
            "tiers": {r["cv_id"]: r["tier"] for r in v["ranking"] if r.get("tier")}, "quality": quality(events)}


def quality(events: list[dict]) -> dict:
    """Q-metrics of one run from the structured metrics the nodes emit (docs/AI_OUTPUT_CRITERIA.md), plus the path
    the graph took: the sequence of nodes, which differs per JD when the agent's decisions differ."""
    m = lambda node: [e["metrics"] for e in events if e.get("node") == node and e.get("metrics")]  # noqa: E731
    msgs = [e.get("message", "") for e in events if e.get("type") == "log"]
    path = [e["node"] for e in events if e.get("type") == "node_start"]
    verify, rep, arb, inv = m("verify_evidence"), m("report"), m("arbitrate"), m("investigate")
    return {"path": path, "verify_first": verify[0] if verify else None, "verify_final": verify[-1] if verify else None,
            "report_first": rep[0] if rep else None, "report_final": rep[-1] if rep else None,
            "arbiter_pairs": len(arb), "arbiter_agree": sum(x["agree"] for x in arb),
            "investigate": {k: sum(x.get(k, 0) for x in inv) for k in ("investigated", "recovered", "calls", "invalid", "facts")},
            "invalid_outputs": sum("LLM output invalid" in x for x in msgs),
            "gate_fallbacks": sum("fallback to default" in x for x in msgs)}


def results_dir(bench: str = "ours") -> Path:
    return OUT / ("results" if bench == "ours" else f"results_{bench}")


def result_path(name: str, bench: str = "ours") -> Path:
    return results_dir(bench) / f"{name.replace(' ', '_').replace('+', 'plus').replace(':', '-')}.json"


def save(name: str, per_jd: dict, bench: str = "ours") -> None:
    results_dir(bench).mkdir(parents=True, exist_ok=True)
    result_path(name, bench).write_text(json.dumps({"system": name, "jds": per_jd}, indent=1))


# ── report ────────────────────────────────────────────────────────────────────────────────────────────
def score_system(per_jd: dict, g: dict, traps: frozenset[str] = OUR_TRAPS) -> dict:
    jds = sorted(per_jd)
    order = lambda jd: per_jd[jd] if isinstance(per_jd[jd], list) else per_jd[jd]["ranking"]  # noqa: E731
    m = {"ndcg@10": [metrics.ndcg(order(j), g[j], 10) for j in jds],
         "ndcg@5": [metrics.ndcg(order(j), g[j], 5) for j in jds],
         "p@5": [metrics.precision(order(j), g[j], 5) for j in jds],
         "r@20": [metrics.recall(order(j), g[j], 20) for j in jds],
         "mrr": [metrics.mrr(order(j), g[j]) for j in jds],
         "tau": [metrics.tau_vs_gold(order(j), g[j]) for j in jds],
         "traps@10": [sum(c in traps and g[j].get(c, 0) < 2 for c in order(j)[:10]) for j in jds]}
    return {"jds": jds, **m}


def report(bench: str = "ours") -> str:
    b = load_bench(bench)
    g = b.gold
    results = {json.loads(p.read_text())["system"]: json.loads(p.read_text())["jds"]
               for p in sorted(results_dir(bench).glob("*.json"))}
    lines = ["# Evaluation results (generated by `python -m app.eval.run report`; do not edit by hand)", ""]
    if bench == "ours":
        ann = annotators()
        a, bb = ann["a"], ann["b"]
        flat_a = [a[j][c] for j in a for c in a[j]]
        flat_b = [bb[j][c] for j in a for c in a[j]]
        lines += [f"Silver labels: 2 independent LLM annotators (Claude Opus, Claude Sonnet), {len(flat_a)} (JD, CV) "
                  f"pairs, graded 0–3. Agreement: exact {sum(x == y for x, y in zip(flat_a, flat_b)) / len(flat_a):.1%}, "
                  f"quadratic-weighted κ = {metrics.weighted_kappa(flat_a, flat_b):.3f}. Gold = mean of the two. "
                  "These are **silver** labels until human annotators check them (docs/EVAL.md). The second "
                  "benchmark is in [EVAL_RESULTS_coding.md](EVAL_RESULTS_coding.md).", ""]
    else:
        lines += [f"Benchmark `{bench}`: the `coding` branch's synthetic set ({len(b.cvs)} template CVs, {len(b.jds)} "
                  "JDs, gold from its own annotation rubric; see app/eval/bench.py). Our pipeline was not developed "
                  "on it. Its gold follows rules similar to its own scoring, which favours that style of ranker.", ""]
    lines += [f"Mean over JDs with a 95% bootstrap CI over JDs. τ = Kendall's τ-b between the system order and gold "
              f"grades over all {len(b.cvs)} CVs. traps@10 = keyword-stuffed / hedged / title-inflated CVs (gold < 2) "
              "in the top 10, summed over JDs.", "",
              "R@20 = share of relevant CVs (gold ≥ 2) in the top 20; MRR = 1 / rank of the first relevant CV.", "",
              "| System | JDs | nDCG@10 | nDCG@5 | P@5 | R@20 | MRR | τ vs gold | traps@10 |",
              "|---|---|---|---|---|---|---|---|---|"]
    rows = dict(results)
    if bench == "ours":
        rows = {"Annotator A order vs gold (ceiling-ish)": {j: sorted(a[j], key=lambda c: (-a[j][c], c)) for j in a},
                **results}
    scored = {}
    for name, per_jd in rows.items():
        sc = scored[name] = score_system(per_jd, g, b.traps)
        ci = lambda k: "{:.3f} [{:.3f}, {:.3f}]".format(*metrics.bootstrap_ci(sc[k]))  # noqa: E731
        lines.append(f"| {name} | {len(sc['jds'])} | {ci('ndcg@10')} | {ci('ndcg@5')} | {ci('p@5')} | {ci('r@20')} "
                     f"| {ci('mrr')} | {ci('tau')} "
                     f"| {sum(sc['traps@10'])} |")
    pairs = [("P6", "B3 rules ⊕ CE"), ("B3 rules ⊕ CE", "R structural + CE"), ("R structural (section-weighted)", "R structural"), ("R structural (section-weighted) + CE", "R whole + CE"),
             ("R structural + CE", "R whole + CE"), ("R structural + CE", "R window + CE"),
             ("R structural", "R whole"), ("R structural + CE", "B2 hybrid"), ("B3 +criteria (rules)", "R structural + CE"),
             ("P6", "B4"), ("P6", "B3 +criteria (rules)"), ("P6", "A1"), ("P6", "A2"), ("P6", "A3"),
             # v3 merge (docs/EVAL.md §8): each against the v2 deterministic baseline
             ("B3 core4 ⊕ CE [rules]", "B3 rules ⊕ CE"), ("B3 core4 ⊕ CE [hybrid]", "B3 rules ⊕ CE"),
             ("B3 8-axis ⊕ CE [llm]", "B3 rules ⊕ CE"), ("B3 8-axis ⊕ CE [rules]", "B3 rules ⊕ CE"),
             ("B3 8-axis ⊕ CE [hybrid]", "B3 rules ⊕ CE"), ("B3 8-axis [hybrid]", "B3 +criteria (rules)")]
    lines += ["", "Paired comparisons (same JDs): one-sided bootstrap p for “first > second” on nDCG@10.", "",
              "| First | Second | ΔnDCG@10 | p | Δτ |", "|---|---|---|---|---|"]
    for x, y in pairs:
        if x in scored and y in scored:
            common = [i for i, j in enumerate(scored[x]["jds"]) if j in scored[y]["jds"]]
            jy = {j: i for i, j in enumerate(scored[y]["jds"])}
            xs = [scored[x]["ndcg@10"][i] for i in common]
            ys = [scored[y]["ndcg@10"][jy[scored[x]["jds"][i]]] for i in common]
            dt = sum(scored[x]["tau"][i] - scored[y]["tau"][jy[scored[x]["jds"][i]]] for i in common) / len(common)
            lines.append(f"| {x} | {y} | {sum(xs) / len(xs) - sum(ys) / len(ys):+.3f} | "
                         f"{metrics.paired_bootstrap_p(xs, ys):.3f} | {dt:+.3f} |")
    agent = {k: v for k, v in results.items() if k.split(" [")[0] in AGENT_SYSTEMS}
    if agent:
        lines += ["", "Agent systems: cost and faithfulness (sum or mean over JDs).", "",
                  "| System | mean s/JD | LLM calls | tokens | evidence quotes | grounded | supported "
                  "| strong tier: gold ≥ 2 | weak tier: gold < 2 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, per_jd in agent.items():
            v = list(per_jd.values())
            ev = sum(x["evidence"] for x in v) or 1
            tiers = [(t, g[j][c]) for j, x in per_jd.items() for c, t in x["tiers"].items()]
            hit = lambda tier, ok: (lambda xs: f"{sum(map(ok, xs))}/{len(xs)}")([gg for t, gg in tiers if t == tier])  # noqa: E731
            lines.append(f"| {name} | {sum(x['seconds'] for x in v) / len(v):.0f} | {sum(x['calls'] for x in v)} | "
                         f"{sum(x['tokens'] for x in v):,} | {ev} | {sum(x['grounded'] for x in v) / ev:.1%} | "
                         f"{sum(x['supported'] for x in v) / ev:.1%} | {hit('strong', lambda x: x >= 2)} "
                         f"| {hit('weak', lambda x: x < 2)} |")
    lines += scorecard(agent)
    if bench == "ours":
        todo = {j: {c: {"a": a[j][c], "b": bb[j][c]} for c in a[j] if a[j][c] != bb[j][c]} for j in a}
        (OUT / "to_adjudicate.yaml").write_text(yaml.safe_dump(todo, sort_keys=True))
        lines += ["", f"Annotator disagreements to adjudicate by humans: {sum(map(len, todo.values()))} pairs "
                      "(.runtime/eval/to_adjudicate.yaml): " + ", ".join(f"{j} {len(v)}" for j, v in sorted(todo.items()))]
    lines += ["", "Per JD nDCG@10:", "", "| System | " + " | ".join(sorted(g)) + " |", "|---|" + "---|" * len(g)]
    for name, sc in scored.items():
        by = dict(zip(sc["jds"], sc["ndcg@10"]))
        lines.append(f"| {name} | " + " | ".join(f"{by[j]:.2f}" if j in by else "—" for j in sorted(g)) + " |")
    return "\n".join(lines) + "\n"


def scorecard(agent: dict[str, dict]) -> list[str]:
    """AI-output scorecard (docs/AI_OUTPUT_CRITERIA.md): each Q-criterion with its measured value per agent system.
    first → final = before and after the graph's own controls acted (retry, drop, flag)."""
    rows = {k: [x["quality"] for x in v.values() if x.get("quality")] for k, v in agent.items()}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        return []
    pct = lambda a, b: f"{a / b:.1%}" if b else "—"  # noqa: E731
    tot = lambda qs, part, key: sum((q[part] or {}).get(key, 0) for q in qs)  # noqa: E731
    out = ["", "AI-output scorecard (Q-ids from docs/AI_OUTPUT_CRITERIA.md; first → final = before → after the "
               "graph's own controls acted).", "",
           "| System | runs | Q2 grounded quotes | Q3 supported | Q4 rationales contradicting the CV | Q5 protected words "
           "| Q6 report candidates with unsupported claims | Q7 arbiter verdicts stable under swap | Q1 invalid outputs "
           "| Q8 gate fallbacks | latency p50 / p95 s | distinct paths |", "|---|" + "---|" * 11]
    for name, qs in rows.items():
        secs = [x["seconds"] for x in agent[name].values() if x.get("quality")]
        q2 = " → ".join(pct(tot(qs, p, "grounded"), tot(qs, p, "evidence")) for p in ("verify_first", "verify_final"))
        q4 = " → ".join(pct(tot(qs, p, "claims_bad"), tot(qs, p, "claims")) for p in ("verify_first", "verify_final"))
        q6 = " → ".join(pct(tot(qs, p, "claims_bad"), tot(qs, p, "claims")) for p in ("report_first", "report_final"))
        out.append(f"| {name} | {len(qs)} | {q2} | {pct(tot(qs, 'verify_final', 'supported'), tot(qs, 'verify_final', 'evidence'))} "
                   f"| {q4} | {sum(tot([q], 'verify_first', 'protected') for q in qs)} found, 0 kept | {q6} "
                   f"| {pct(sum(q['arbiter_agree'] for q in qs), sum(q['arbiter_pairs'] for q in qs))} "
                   f"| {sum(q['invalid_outputs'] for q in qs)} | {sum(q['gate_fallbacks'] for q in qs)} "
                   f"| {metrics.percentile(secs, 50):.0f} / {metrics.percentile(secs, 95):.0f} "
                   f"| {len({tuple(q['path']) for q in qs})} of {len(qs)} |")
    return out


async def run_agents(systems: list[str], jds: list, model: str | None = None, bench: str = "ours") -> None:
    """One event loop for the whole run: cached LLM clients hold connections bound to the loop that created them.
    With `model`, results are saved as "<system> [<model>]" next to the default-model runs."""
    for system in systems:
        name = f"{system} [{model.split(':', 1)[1]}]" if model else system
        path = result_path(name, bench)
        per_jd = json.loads(path.read_text())["jds"] if path.exists() else {}
        for jd in jds:
            if jd.id in per_jd:
                continue  # resumable: a crash or Ctrl-C keeps finished JDs
            result = await run_agent(jd, AGENT_SYSTEMS[system], model)
            from app.providers import claude_code
            if claude_code.QUOTA_HIT:  # the run degraded silently (CVs → needs review): never save it
                raise SystemExit(f"Subscription limit reached during {name} {jd.id}: not saved. Rerun the same "
                                 "command after the limit resets; finished JDs are kept.")
            per_jd[jd.id] = result
            save(name, per_jd, bench)
            r = per_jd[jd.id]
            print(f"{name} {jd.id}: {r['seconds']}s {r['tokens']} tok, top5 {r['ranking'][:5]}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["retrieval", "agent", "report"])
    ap.add_argument("--dense", default=get_settings().embed_dense_model)
    ap.add_argument("--systems", default="P6,B4,A1,A2,A3,A4")
    ap.add_argument("--only", default="", help="retrieval only: comma-separated exact system names")
    ap.add_argument("--jds", default="")
    ap.add_argument("--bench", default="ours", choices=["ours", "coding", "djinni", "djinni_holdout"])
    ap.add_argument("--model", help='agent only: "<provider>:<model>" for both roles, e.g. claude_code:claude-sonnet-5-5')
    a = ap.parse_args()
    cache.enable(get_settings().runtime_dir / "llm_cache.sqlite")
    if a.what == "retrieval":
        suffix = "" if a.dense == get_settings().embed_dense_model else f" [{a.dense.split('/')[-1]}]"
        only = {name.strip() for name in a.only.split(",") if name.strip()}
        for name, per_jd in retrieval_systems(a.dense, load_bench(a.bench), only).items():
            save(name + suffix, per_jd, a.bench)
            print(f"saved {name + suffix}")
    elif a.what == "agent":
        expected = {"ours": ROOT / "data", "coding": get_settings().runtime_dir / "bench" / "coding_corpus",
                    "djinni": get_settings().runtime_dir / "bench" / "djinni_corpus",
                    "djinni_holdout": get_settings().runtime_dir / "bench" / "djinni_holdout_corpus"}[a.bench]
        if get_settings().data_dir.resolve() != expected.resolve():
            raise SystemExit(f"--bench {a.bench} needs DATA_DIR={expected}; materialize external corpora with "
                             f"`python -m app.eval.bench materialize {a.bench}`")
        asyncio.run(run_agents(a.systems.split(","), [d for d in load_dir("jds") if not a.jds or d.id in a.jds.split(",")],
                               a.model, a.bench))
    else:
        text = report(a.bench)
        Path(ROOT / "docs" / ("EVAL_RESULTS.md" if a.bench == "ours" else f"EVAL_RESULTS_{a.bench}.md")).write_text(text)
        print(text)


if __name__ == "__main__":
    main()
