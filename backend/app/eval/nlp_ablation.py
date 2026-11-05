"""E19: how much each NLP stage is worth in nDCG@10 (NLP501-bench v1 splits).

    cd backend && uv run python -m app.eval.nlp_ablation          rank every variant (slow) → docs/NLP_ABLATION_RESULTS.md
    cd backend && uv run python -m app.eval.nlp_ablation report   re-render from the saved rankings

The system is the deterministic ranker the agent screens with (B3 8-axis ⊕ CE, hybrid JD parser: knock-out H + 8 rule
criteria fused with the cross-encoder, zero LLM judgement). Each variant switches one CV-side NLP stage off by
rewriting the parsed CVProfile, so retrieval, the index and the JD are identical across variants and the only thing
that moves is what that stage contributed. Production code is not touched.

    − hedge scope (NegEx)    a hedged mention ("familiar with Java") counts as a real one (X7 hedge scope)
    − evidence strength      a tool only listed in Skills counts like one used in a role (X7 used/listed)
    − depth cue              no L1–L4 from the candidate's action verbs (X7 depth_cue)
    − dated years            years from "N years" claims only, not from parsed role dates (X5 date grammar)
    − all four               keyword presence only: what is left of the rules without the NLP pipeline
"""

import asyncio
import json
import sys
from collections.abc import Callable
from dataclasses import replace

from app.config import ROOT, Settings
from app.eval import metrics
from app.eval.bench import Bench, load
from app.eval.benchmark import MANIFEST
from app.eval.run import b3_systems, full_order, requirements, result_path, save
from app.nlp import extract
from app.nlp.pipeline import CVProfile, process
from app.rag import retrieve
from app.rag.corpus import load_yaml
from app.rag.store import KnowledgeStore

DOC = ROOT / "docs" / "NLP_ABLATION_RESULTS.md"
SYSTEM = "8-axis ⊕ CE"
ROLE_SECTIONS = ("experience", "projects")


def no_hedge(p: CVProfile) -> CVProfile:
    """Hedged mentions get the strength they would have without the cue."""
    def fix(m: extract.Mention) -> extract.Mention:
        if m.strength != "hedged":
            return m
        return replace(m, strength="used" if m.section in ROLE_SECTIONS else "listed")
    return replace(p, skills={k: [fix(m) for m in ms] for k, ms in p.skills.items()})


def no_evidence(p: CVProfile) -> CVProfile:
    """Listed counts as used: the Skills section and real work are no longer told apart."""
    return replace(p, skills={k: [replace(m, strength="used") if m.strength == "listed" else m for m in ms]
                              for k, ms in p.skills.items()})


def no_depth(p: CVProfile) -> CVProfile:
    return replace(p, depth_cue=0, depth_cue_line=None)


def no_dates(p: CVProfile) -> CVProfile:
    """Years as a keyword ATS reads them: the largest "N years" claim, or unknown."""
    total, auto, src = extract.years(p.text, [], p.chunks, p.skills)
    return replace(p, years_total=total, years_automation=auto, years_source=src)


def all_off(p: CVProfile) -> CVProfile:
    return no_dates(no_depth(no_evidence(no_hedge(p))))


VARIANTS: dict[str, Callable[[CVProfile], CVProfile]] = {
    "full pipeline": lambda p: p,
    "− hedge scope (NegEx)": no_hedge,
    "− evidence strength (used/listed)": no_evidence,
    "− depth cue (L1–L4)": no_depth,
    "− dated years": no_dates,
    "− all four (keyword presence)": all_off,
}


def rank_bench(b: Bench) -> dict[str, dict[str, list[str]]]:
    """variant → JD → full ranking of the bench's pool."""
    store = KnowledgeStore(Settings(qdrant_url=":memory:"))
    pool = [d.id for d in b.cvs]
    prof = {d.id: process(d.id, d.text, d.meta.get("headline", "")) for d in b.cvs}
    store.index_items("chunks_structural", retrieve.chunk_items(list(prof.values())))
    weights = load_yaml("criteria_profiles.yaml")["profiles"]
    out: dict[str, dict[str, list[str]]] = {v: {} for v in VARIANTS}
    for jd in b.jds:
        r = asyncio.run(requirements(jd, "hybrid", b.name))
        w = weights[jd.meta.get("company_profile", "default")]["weights"]
        # The cross-encoder reads only chunks, which no variant changes: score it once per JD, not once per variant
        qs = retrieve.queries(r["req"], r["terms"], r["subs"])
        hits = retrieve.rank_cvs(store, "chunks_structural", qs, limit=max(200, 2 * len(pool)))
        ce = retrieve.rerank_cvs(store, qs, [h for h in hits if h.cv_id in prof], prof)
        for name, fn in VARIANTS.items():
            variant = {c: fn(p) for c, p in prof.items()}
            out[name][jd.id] = full_order(b3_systems(store, pool, variant, r, w, ce=ce)[SYSTEM], pool)
        print(f"{b.name} {jd.id}: {len(VARIANTS)} variants", flush=True)
    return out


def per_jd_ndcg(rankings: dict[str, dict[str, list[str]]], gold: dict[str, dict[str, float]]
                ) -> dict[str, dict[str, float]]:
    return {v: {jd: metrics.ndcg(ids, gold[jd], 10) for jd, ids in per.items() if jd in gold}
            for v, per in rankings.items()}


def traps_at_10(rankings: dict[str, dict[str, list[str]]], gold: dict[str, dict[str, float]], traps: frozenset[str]
                ) -> dict[str, float]:
    """variant → mean number of trap CVs (keyword stuffing, hedges, title inflation) not relevant to the JD in the top 10:
    what the hedge scope and evidence strength exist to keep out, and nDCG barely sees."""
    return {v: sum(sum(c in traps and gold[j].get(c, 0) < 2 for c in ids[:10]) for j, ids in per.items()) / len(per)
            for v, per in rankings.items()}


def table(scores: dict[str, dict[str, float]], jds: list[str], traps: dict[str, float] | None = None) -> list[str]:
    full = [scores["full pipeline"][j] for j in jds]
    trap = (lambda v: f" | {traps[v]:.2f}") if traps else (lambda v: "")  # noqa: E731
    lines = ["| Variant | nDCG@10 [95% CI] | Δ vs full | p (full > variant)" + (" | traps@10 (ours)" if traps else "")
             + " |", "|---|---|---|---" + ("|---" if traps else "") + "|"]
    for v, per in scores.items():
        xs = [per[j] for j in jds]
        mean, lo, hi = metrics.bootstrap_ci(xs)
        if v == "full pipeline":
            lines.append(f"| **{v}** | **{mean:.3f}** [{lo:.3f}, {hi:.3f}] | — | —{trap(v)} |")
            continue
        delta = mean - sum(full) / len(full)
        lines.append(f"| {v} | {mean:.3f} [{lo:.3f}, {hi:.3f}] | {delta:+.3f} | "
                     f"{metrics.paired_bootstrap_p(full, xs):.3f}{trap(v)} |")
    return lines


def result_name(variant: str) -> str:
    """Saved-results name of a variant ("/" would make a sub-directory of the results folder)."""
    return f"E19 {variant}".replace("/", "-")


def rank_all() -> None:
    """The slow part: rank every variant on both sources and keep the rankings (results dir, `E19 <variant>`)."""
    for name in ("ours", "coding"):
        for v, per in rank_bench(load(name)).items():
            save(result_name(v), per, name)


def report() -> str:
    m = json.loads(MANIFEST.read_text())
    scores: dict[str, dict[str, float]] = {v: {} for v in VARIANTS}
    traps: dict[str, float] = {}
    for name in ("ours", "coding"):
        b = load(name)
        rankings = {v: json.loads(result_path(result_name(v), name).read_text())["jds"] for v in VARIANTS}
        for v, per in per_jd_ndcg(rankings, b.gold).items():
            scores[v].update(per)
        if b.traps:
            traps = traps_at_10(rankings, b.gold, b.traps)
    lines = ["# NLP-stage ablation (generated by `python -m app.eval.nlp_ablation`; do not edit by hand)", "",
             "System: B3 8-axis ⊕ CE with the hybrid JD parser (deterministic, zero LLM judgement). Each variant "
             "switches one CV-side NLP stage off by rewriting the parsed profile; retrieval, index and JD are "
             "identical. Labels and splits: NLP501-bench v1 (data/eval/benchmark_v1.json). Mean nDCG@10 over JDs, "
             "95% bootstrap CI, paired one-sided bootstrap p. PYTHONHASHSEED=0.", ""]
    for split in ("test", "dev"):
        jds = m["splits"][split]["ours"] + m["splits"][split]["coding"]
        lines += [f"## {split.capitalize()} split ({len(jds)} JDs)", "", *table(scores, jds), ""]
    every = sorted(scores["full pipeline"])
    lines += [f"## All {len(every)} JDs", "", *table(scores, every, traps), "",
              "traps@10: mean number of the 8 trap CVs of ours (keyword stuffing, hedged skills, inflated titles) that "
              "sit in a JD's top 10 without being relevant to it, over ours' 12 JDs. Lower is better.", "",
              "Per JD nDCG@10:", "", "| Variant | " + " | ".join(every) + " |", "|---|" + "---|" * len(every)]
    lines += [f"| {v} | " + " | ".join(f"{scores[v][j]:.2f}" for j in every) + " |" for v in scores]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(DOC)


if __name__ == "__main__":
    if sys.argv[1:2] != ["report"]:  # `report` re-renders from the saved rankings
        rank_all()
    print(report())
