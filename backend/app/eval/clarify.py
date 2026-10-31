"""H6, controlled: does asking HR recover the ranking a vague JD loses? (v7 slide 26: "τ improves on vague JDs when
the clarification step is on")

Only 3 benchmark JDs are naturally vague, too few to conclude. So every JD of NLP501-bench v1 gets a *vague twin*: the
same text with every tool name replaced by its category ("a browser automation framework", "a CI server") and every
year count by "several years". The twin keeps the original JD's labels: they encode what the hiring manager wanted,
which is exactly what HR knows and the vague text no longer says. Three conditions, all ranked by the agent's own
deterministic scorer (8 rule criteria ⊕ cross-encoder, the hybrid JD parser):

    clear      the original JD                                   (upper bound)
    vague      the twin, nobody asks                              (what a fixed pipeline gets)
    asked      the twin + HR's answer, merged exactly as the agent does in N1 ("HR clarifications (authoritative)")

and the L1 decision itself: does the agent ask on the twin, and stay quiet on the clear JD?

The HR answer is scripted from the original JD's requirements, so this measures the mechanism (detect → ask →
integrate), not how well a real HR person answers.

    python -m app.eval.clarify  → docs/CLARIFY_RESULTS.md (LLM: JD extraction and the gate; cached)
"""

import asyncio
import json
import re

from app.agent.nodes import ask_hr_gate, extract_requirements
from app.agent.schemas import JDRequirements
from app.config import ROOT, Settings, get_settings
from app.eval import metrics
from app.eval.bench import load
from app.eval.run import b3_systems, requirements
from app.nlp.pipeline import process
from app.nlp.taxonomy import aliases, group, skills
from app.providers import cache
from app.providers.llm import create_llm, default_spec
from app.rag import retrieve
from app.rag.corpus import load_yaml
from app.rag.store import KnowledgeStore

DOC = ROOT / "docs" / "CLARIFY_RESULTS.md"
CATEGORY = {"web_e2e": "a browser automation framework", "mobile": "a mobile automation framework",
            "api": "an API testing tool", "performance": "a load-testing tool", "test_framework": "a test runner",
            "bdd": "a BDD tool", "cicd": "a CI server", "infra": "container tooling", "language": "a programming language",
            "certification": "a testing certification", "design_pattern": "good test design",
            "desktop": "a UI automation tool", "test_management": "a test management tool"}
YEARS = re.compile(r"\b\d+\s*\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)\b", re.I)
YEARS_VI = re.compile(r"\b\d+\s*\+?\s*năm\b", re.I)


def vague(text: str) -> str:
    """The JD with tool names replaced by their category and year counts by "several years"."""
    names = sorted({a: group(k) for k in skills() for a in aliases(k) if len(a) >= 3 and a != "requests"}.items(),
                   key=lambda kv: -len(kv[0]))
    rx = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(a) for a, _ in names) + r")(?![A-Za-z0-9])", re.I)
    by = {a.lower(): g for a, g in names}
    out = rx.sub(lambda m: CATEGORY.get(by.get(m.group(1).lower(), ""), "the right tools"), text)
    return YEARS_VI.sub("nhiều năm", YEARS.sub("several years", out))


def hr_answer(req: dict, subs: dict) -> str:
    """What HR, who knows the original requirements, answers."""
    must = "; ".join(" or ".join([m, *subs.get(m, [])]) for m in req["must_have"]) or "no specific tool"
    years = f" At least {req['min_years']} years of experience." if req.get("min_years") else ""
    return f"Must-have: {must}.{years}"


async def run() -> dict:
    cache.enable(get_settings().runtime_dir / "llm_cache.sqlite")
    reasoning, fast = create_llm(default_spec("reasoning")), create_llm(default_spec("fast"))
    weights = load_yaml("criteria_profiles.yaml")["profiles"]
    rows = []
    for bench in ("ours", "coding"):
        b = load(bench)
        store = KnowledgeStore(Settings(qdrant_url=":memory:"))
        prof = {d.id: process(d.id, d.text, d.meta.get("headline", "")) for d in b.cvs}
        store.index_items("chunks_structural", retrieve.chunk_items(list(prof.values())))
        pool = list(prof)
        for jd in b.jds:
            w = weights[jd.meta.get("company_profile", "default")]["weights"]
            clear = await requirements(jd, "hybrid", bench)
            twin = vague(jd.text)
            req_v, subs_v, _ = await extract_requirements(reasoning, twin, "hybrid")
            ask_v, why_v = await ask_hr_gate(fast, twin, req_v.model_copy(deep=True))
            ask_c, _ = await ask_hr_gate(fast, jd.text, JDRequirements(**clear["req"]))
            question = (req_v.ambiguities or ["Which tools/frameworks are must-have for this role?"])[0]
            answered = (f"{twin}\n\nHR clarifications (authoritative):\n- Q: {question}\n"
                        f"  A: {hr_answer(clear['req'], clear['subs'])}")
            req_a, subs_a, _ = await extract_requirements(reasoning, answered, "hybrid")
            g = b.gold[jd.id]
            row = {"bench": bench, "jd": jd.id, "asks_on_vague": ask_v, "asks_on_clear": ask_c, "why": why_v or "",
                   "must_vague": req_v.must_have, "must_asked": req_a.must_have, "must_clear": clear["req"]["must_have"]}
            for cond, req, subs in (("clear", clear["req"], clear["subs"]), ("vague", req_v.model_dump(), subs_v),
                                    ("asked", req_a.model_dump(), subs_a)):
                r = {"req": req, "subs": subs, "terms": list(dict.fromkeys(req["must_have"] + req["nice_to_have"]
                                                                          + req["languages"]))}
                order = b3_systems(store, pool, prof, r, w)["8-axis ⊕ CE"]
                row[cond] = {"ndcg@10": metrics.ndcg(order, g, 10), "tau": metrics.tau_vs_gold(order, g)}
            rows.append(row)
            print(f"{jd.id}: ask vague {ask_v} / clear {ask_c} · nDCG clear {row['clear']['ndcg@10']:.3f} "
                  f"vague {row['vague']['ndcg@10']:.3f} asked {row['asked']['ndcg@10']:.3f}", flush=True)
    return {"rows": rows}


def render(res: dict) -> str:
    rows = res["rows"]
    mean = lambda c, k: sum(r[c][k] for r in rows) / len(rows)  # noqa: E731
    p = lambda k: metrics.paired_bootstrap_p([r["asked"][k] for r in rows], [r["vague"][k] for r in rows])  # noqa: E731
    asked_v, asked_c = sum(r["asks_on_vague"] for r in rows), sum(r["asks_on_clear"] for r in rows)
    real = lambda k: sum(r["asked" if r["asks_on_vague"] else "vague"][k] for r in rows) / len(rows)  # noqa: E731
    lines = ["# H6, controlled: asking HR on vague JDs (generated by `python -m app.eval.clarify`; do not edit by hand)", "",
             f"Every NLP501-bench v1 JD ({len(rows)}) gets a vague twin: tool names → their category, year counts → "
             "\"several years\"; the original labels stay (they encode the hiring intent). Ranked by the agent's "
             "deterministic scorer (8 rule criteria ⊕ CE, hybrid JD parser). The HR answer is scripted from the original "
             "requirements: this tests the mechanism (detect → ask → integrate), not a real HR conversation.", "",
             f"**L1 decision:** the agent asks HR on {asked_v}/{len(rows)} vague twins and on {asked_c}/{len(rows)} "
             "original JDs (3 of which are vague by design).", "",
             "| Condition | nDCG@10 (mean) | Kendall τ (mean) |", "|---|---|---|",
             f"| clear JD (upper bound) | {mean('clear', 'ndcg@10'):.3f} | {mean('clear', 'tau'):.3f} |",
             f"| vague twin, nobody asks | {mean('vague', 'ndcg@10'):.3f} | {mean('vague', 'tau'):.3f} |",
             f"| vague twin, HR asked and answered | {mean('asked', 'ndcg@10'):.3f} | {mean('asked', 'tau'):.3f} |",
             f"| **the agent as it runs** (HR's answer only where the gate asked) | {real('ndcg@10'):.3f} | {real('tau'):.3f} |", "",
             f"Where the gate did not ask ({len(rows) - asked_v} twins), the twin still contained a test-type phrase such as "
             "\"API testing\" that the rule counts as a concrete tool, so the rule did not fire and the LLM gate chose to "
             "proceed. Asking can also lower a score: the original requirements are stricter than the vague ones.", "",
             f"Asked vs not asked, paired over {len(rows)} JDs: ΔnDCG@10 "
             f"{mean('asked', 'ndcg@10') - mean('vague', 'ndcg@10'):+.3f} (p = {p('ndcg@10'):.3f}), "
             f"Δτ {mean('asked', 'tau') - mean('vague', 'tau'):+.3f} (p = {p('tau'):.3f}). Share of the vague loss "
             "recovered: " + (f"{(mean('asked', 'ndcg@10') - mean('vague', 'ndcg@10')) / max(mean('clear', 'ndcg@10') - mean('vague', 'ndcg@10'), 1e-9):.0%}"
                              if mean('clear', 'ndcg@10') > mean('vague', 'ndcg@10') else "n/a") + ".", "",
             "| JD | asks (vague / clear) | must-haves read from the twin → after HR | nDCG@10 clear / vague / asked | τ clear / vague / asked |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['jd']} | {'yes' if r['asks_on_vague'] else 'no'} / {'yes' if r['asks_on_clear'] else 'no'} "
                     f"| {r['must_vague'] or '—'} → {r['must_asked'] or '—'} "
                     f"| {r['clear']['ndcg@10']:.2f} / {r['vague']['ndcg@10']:.2f} / {r['asked']['ndcg@10']:.2f} "
                     f"| {r['clear']['tau']:.2f} / {r['vague']['tau']:.2f} / {r['asked']['tau']:.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys
    path = get_settings().runtime_dir / "eval" / "clarify.json"
    if sys.argv[1:2] == ["report"]:  # re-render from the saved run, no LLM calls
        DOC.write_text(render(json.loads(path.read_text())))
        raise SystemExit(print(DOC.read_text()[:2400]))
    result = asyncio.run(run())
    (get_settings().runtime_dir / "eval" / "clarify.json").write_text(json.dumps(result, indent=1, default=str))
    DOC.write_text(render(result))
    print(DOC.read_text()[:2000])
