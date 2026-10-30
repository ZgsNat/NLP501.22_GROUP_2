"""Q8: does the agent decide right? Planted scenarios with a known correct decision, run on a real model.

    L1 ambiguity gate   18 JDs (both benchmarks). Gold: ask HR exactly where the benchmark scripts an HR answer
                        (JD-003, ambiguous_senior, ambiguous_junior), proceed elsewhere.
    L6 investigator     knock-outs the parser caused, and knock-outs it must keep:
                        + garbled   OCR-noise CVs of the coding set, must-haves the generator says they have
                        + typo      one of our CVs with its main tool misspelt everywhere ("Selenuim")
                        − garbled   OCR-noise CVs, must-haves the generator says they lack
                        − hedged    a misspelt tool that the CV only says it is learning
                        Gold: recover exactly the skills the CV has; never recover a skill it lacks.

    python -m app.eval.scenarios [--model claude_code:claude-sonnet-5-5] [--only investigator]
        → docs/SCENARIO_RESULTS.md, or docs/SCENARIO_RESULTS_<model>.md with --model (LLM cache on)
"""

import argparse
import asyncio
import json
import re
import time

import yaml

from app.agent import investigate as inv
from app.agent.nodes import ask_hr_gate, extract_requirements
from app.agent.scoring import knockout_reason
from app.config import ROOT, get_settings
from app.eval.bench import CODING_DIR, load
from app.eval.extraction import to_key
from app.nlp.pipeline import process
from app.providers import cache
from app.providers.llm import create_llm, default_spec

OUT = ROOT / "docs" / "SCENARIO_RESULTS.md"
ASK = {"JD-003", "ambiguous_senior", "ambiguous_junior"}
TYPOS = {"selenium": "Selenuim", "playwright": "Playwrigth", "cypress": "Cyprees", "appium": "Apium", "postman": "Postmann"}


async def gate(reasoning, fast) -> list[dict]:
    out = []
    for b in ("ours", "coding"):
        for jd in load(b).jds:
            req, _, _ = await extract_requirements(reasoning, jd.text, "llm")
            ask, why = await ask_hr_gate(fast, jd.text, req)
            out.append({"jd": jd.id, "gold": jd.id in ASK, "ask": ask, "why": why or "no open questions"})
    return out


def investigator_cases() -> list[dict]:
    """(profile, missing must-haves, accepted substitutes, skills the CV truly has among them, kind)."""
    cases = []
    latent = {x["cv_id"]: x for x in json.loads((CODING_DIR / "latent.json").read_text())}
    for jd in yaml.safe_load((CODING_DIR / "jds.yaml").read_text()):
        r = json.loads((ROOT / ".runtime/eval/requirements/coding/llm" / f"{jd['id']}.json").read_text())
        must, subs = r["req"]["must_have"], r["subs"]
        for cid, x in latent.items():
            if not x["low_quality"]:
                continue
            p = process(cid, (CODING_DIR / "cvs" / f"{cid}.txt").read_text(encoding="utf-8"))
            missing = [m for m in must if knockout_reason(p, [m], subs)]
            if not missing:
                continue
            has = {k for t in x["tools"] + x["langs"] + x["cicd"] + x["test_types"] if (k := to_key(t))}  # "api testing"
            truth = {t for m in missing for t in [m, *subs.get(m, [])] if t in has}
            out = any(not ({m, *subs.get(m, [])} & has) for m in missing)  # a must-have the CV truly lacks
            cases.append({"id": f"{cid}×{jd['id']}", "p": p, "missing": missing, "subs": subs, "truth": truth,
                          "kind": "+ garbled" if truth else "− garbled", "stays_out": out})
    used = {}
    for d in load("ours").cvs:
        p = process(d.id, d.text)
        for tool, typo in TYPOS.items():
            if tool not in used and (b := p.best(tool)) and b.strength == "used":
                used[tool] = d
    for tool, d in used.items():
        text = re.sub(tool, TYPOS[tool], d.text, flags=re.I)
        if not knockout_reason(process(d.id, text), [tool], {}):
            continue  # another alias still proves the tool ("Newman" for Postman): the typo caused no knock-out
        cases.append({"id": f"{d.id}:{TYPOS[tool]}", "p": process(d.id, text), "missing": [tool], "subs": {},
                      "truth": {tool}, "kind": "+ typo", "stays_out": False})
    for d in [d for d in load("ours").cvs if not process(d.id, d.text).best("selenium")][:3]:
        text = d.text.rstrip() + "\n- Currently learning Selenuim through an online course."
        cases.append({"id": f"{d.id}:learning", "p": process(d.id, text), "missing": ["selenium"], "subs": {},
                      "truth": set(), "kind": "− hedged", "stays_out": True})
    return cases


async def investigator(model, per_kind: int | None = None) -> list[dict]:
    s, out, seen = get_settings(), [], {}
    for c in investigator_cases():
        seen[c["kind"]] = seen.get(c["kind"], 0) + 1
        if per_kind and seen[c["kind"]] > per_kind:
            continue  # a capped run (a subscription model) takes the first cases of each kind
        why = inv.triggers(c["p"], c["missing"], c["subs"])
        if why is None:  # the deterministic trigger is part of the agent: a case it skips is reported, not run
            out.append({**{k: c[k] for k in ("id", "kind", "missing", "stays_out")}, "truth": sorted(c["truth"]), "triggered": False,
                        "found": [], "calls": 0, "invalid": False, "kept_out": True})
            continue
        res = await inv.investigate_cv(model, c["p"], "Automation Test Engineer", c["missing"], c["subs"], why,
                                       s.max_investigation_steps)
        p = inv.with_facts(c["p"], res["facts"])
        out.append({"id": c["id"], "kind": c["kind"], "missing": c["missing"], "truth": sorted(c["truth"]), "stays_out": c["stays_out"],
                    "triggered": True, "found": sorted(f["skill"] for f in res["facts"]), "calls": res["calls"],
                    "invalid": res["invalid"], "kept_out": knockout_reason(p, c["missing"], c["subs"]) is not None,
                    "trace": [f"{t['tool']}({t['arg'][:30]}) → {t['observation'].splitlines()[0][:70] if t['observation'] else ''}"
                              for t in res["trace"]]})
    return out


def render(spec: str, g: list[dict], iv: list[dict], secs: float) -> str:
    ok = sum(x["ask"] == x["gold"] for x in g) if g else 0
    tp = sum(x["ask"] and x["gold"] for x in g)
    lines = ["# Agent decision scenarios (generated by `python -m app.eval.scenarios`; do not edit by hand)", "",
             f"Model: `{spec}` · {secs:.0f} s · LLM cache on. Q8 in docs/AI_OUTPUT_CRITERIA.md.", ""]
    if g:
        lines += ["## L1 ambiguity gate: pause for HR or proceed?", "",
                  f"Accuracy {ok}/{len(g)} ({ok / len(g):.0%}); asked HR on {tp}/{sum(x['gold'] for x in g)} JDs that "
                  f"need it, and on {sum(x['ask'] and not x['gold'] for x in g)}/{sum(not x['gold'] for x in g)} that do not.",
                  "", "| JD | should ask | asked | why |", "|---|---|---|---|"]
        lines += [f"| {x['jd']} | {'yes' if x['gold'] else 'no'} | {'yes' if x['ask'] else 'no'} | {x['why'][:110]} |"
                  for x in g]
    lines += ["", "## L6 investigator: recover what the parser missed, never what the CV lacks", "",
              "| Case kind | cases | triggered | correct final status | skills recovered (true / false) | mean LLM calls | invalid outputs |",
              "|---|---|---|---|---|---|---|"]
    for kind in ("+ garbled", "+ typo", "− garbled", "− hedged"):
        xs = [x for x in iv if x["kind"] == kind]
        if not xs:
            continue
        correct = sum(x["kept_out"] == x["stays_out"] for x in xs)
        true_f = sum(len(set(x["found"]) & set(x["truth"])) for x in xs)
        false_f = sum(len(set(x["found"]) - set(x["truth"])) for x in xs)
        run = [x for x in xs if x["triggered"]]
        lines.append(f"| {kind} | {len(xs)} | {len(run)} | {correct}/{len(xs)} | {true_f} / {false_f} | "
                     f"{sum(x['calls'] for x in run) / max(len(run), 1):.1f} | {sum(x['invalid'] for x in xs)} |")
    want = sum(len(x["truth"]) for x in iv if x["kind"][0] == "+")
    got = sum(len(set(x["found"]) & set(x["truth"])) for x in iv)
    wrong = sum(len(set(x["found"]) - set(x["truth"])) for x in iv)
    lines += ["", f"Recovery recall {got}/{want}; false recoveries {wrong} (the target is 0: code verifies every proposal). "
              "Correct final status = the CV is back in the pool exactly when, with the true facts, it meets every must-have "
              "(a + case can still rightly stay out when another must-have is truly missing).", "",
              "Traces (first 12 cases):", ""]
    for x in [x for x in iv if x.get("trace")][:12]:
        lines.append(f"- **{x['id']}** ({x['kind']}, missing {x['missing']}, truth {x['truth']}): " + " · ".join(x["trace"]))
    return "\n".join(lines) + "\n"


async def main(spec: str | None, only: str | None, per_kind: int | None = None) -> None:
    cache.enable(get_settings().runtime_dir / "llm_cache.sqlite")
    reasoning = create_llm(spec or default_spec("reasoning"))
    fast = create_llm(spec or default_spec("fast"))
    t0 = time.perf_counter()
    g = await gate(reasoning, fast) if only != "investigator" else []
    iv = await investigator(reasoning, per_kind) if only != "gate" else []
    out = OUT.with_name(f"SCENARIO_RESULTS_{spec.split(':', 1)[1]}.md") if spec else OUT
    out.write_text(render(spec or default_spec("reasoning"), g, iv, time.perf_counter() - t0))
    print(out.read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help='"<provider>:<model>" for both roles (default: the configured roles)')
    ap.add_argument("--only", choices=["gate", "investigator"])
    ap.add_argument("--per-kind", type=int, help="cap investigator cases per kind (cost control)")
    a = ap.parse_args()
    asyncio.run(main(a.model, a.only, a.per_kind))
