"""Release gate: drive the real HTTP API through every sample JD, play HR, and check the v1 criteria.

    uv run uvicorn app.main:app --app-dir backend        # in another terminal
    cd backend && uv run python -m scripts.e2e            # all JDs → prints a markdown table, exits 1 on failure
    cd backend && uv run python -m scripts.e2e JD-002     # one JD
"""

import json
import re
import statistics
import sys
import time

import httpx

from app.agent.scoring import foreign_ids, tiebreak_key

API = "http://localhost:8000/api"
HR_ANSWERS = {  # what a recruiter would answer per JD when the agent asks (L1)
    "JD-003": "Must-have: Selenium with Java. At least 3 years. English is required for client calls.",
}
HR_ADJUST = {"JD-001": {"istqb": 0.25}}  # one re-weight to exercise L5, then approve


def run(c: httpx.Client, jd: dict) -> dict:
    res = {"jd": jd["id"], "error": None, "logs": [], "interrupts": []}
    t0 = time.perf_counter()
    adjusted = False

    def stream(url: str, body: dict) -> dict | None:
        pending = None
        with c.stream("POST", url, json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                ev = json.loads(line)
                if ev["type"] == "run_started":
                    res["run_id"] = ev["run_id"]
                elif ev["type"] == "log":
                    res["logs"].append(ev)
                elif ev["type"] == "interrupt":
                    pending = ev
                    res["interrupts"].append(ev["kind"])
                elif ev["type"] == "done":
                    res["usage"] = ev["usage"]
                elif ev["type"] == "error":
                    res["error"] = ev["message"]
        return pending

    pending = stream(f"{API}/runs", {"jd_text": jd["text"], "company_profile": jd["company_profile"]})
    while pending and not res["error"]:
        if pending["kind"] == "substitutes":  # HR accepts every proposed related tool
            body = {"accept": {p["must_have"]: p["related"] for p in pending["payload"]["proposals"]}}
        elif pending["kind"] == "clarification":
            answer = HR_ANSWERS.get(jd["id"], "Use the JD as written; be strict on the named tools.")
            body = {"answers": {q: answer for q in pending["payload"]["questions"]}}
        elif jd["id"] in HR_ADJUST and not adjusted:
            adjusted = True
            body = {"action": "adjust", "weights": {**pending["payload"]["weights"], **HR_ADJUST[jd["id"]]}}
        else:
            body = {"action": "approve"}
        pending = stream(f"{API}/runs/{res['run_id']}/resume", body)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    if "run_id" in res:
        snap = c.get(f"{API}/runs/{res['run_id']}").json()
        res["values"], res["next"] = snap["values"], snap["next"]
    return res


def metrics(r: dict) -> dict:
    v = r.get("values", {})
    ranked = [x for x in v.get("ranking", []) if x["status"] == "ranked"]
    totals = [x["total"] for x in ranked]
    first_verify = next((ln["message"] for ln in r["logs"] if ln["node"] == "verify_evidence"
                         and "grounded" in ln["message"]), "")
    m = re.search(r"(\d+)/(\d+)", first_verify)
    loops = v.get("loops", {})
    rerank = {c["cv_id"]: c.get("rerank_score") for c in v.get("candidates", [])}
    keys = [tiebreak_key(x, v.get("assessments", {}).get(x["cv_id"]), rerank.get(x["cv_id"])) for x in ranked[:5]]
    return {
        "jd": r["jd"], "ok": not r["error"] and r.get("next") == [] and bool(v.get("report")),
        "seconds": r["seconds"], "loops": {k: n for k, n in loops.items() if n},
        "shortlist": len(ranked), "knocked_out": sum(x["status"] == "knocked_out" for x in v.get("ranking", [])),
        "every_cv_has_reason": all(x["status"] == "ranked" or x["reason"] for x in v.get("ranking", [])),
        "grounded_first_pass": round(100 * int(m[1]) / int(m[2]), 1) if m and int(m[2]) else None,
        "spread": round(statistics.pstdev(totals), 3) if len(totals) > 1 else 0.0,
        "decisive": len(set(keys)) == len(keys),
        "report_grounded": not foreign_ids(v.get("report") or {}, [x["cv_id"] for x in ranked[:5]]),
        "top": f"{totals[0]:.2f}–{totals[min(4, len(totals) - 1)]:.2f}" if totals else "—",
        "llm_calls": r.get("usage", {}).get("calls"), "tokens": r.get("usage", {}).get("total_tokens"),
        "by_node": {k: n["input_tokens"] + n["output_tokens"] for k, n in r.get("usage", {}).get("by_node", {}).items()},
        "error": r["error"],
    }


def main(only: list[str]) -> int:
    c = httpx.Client(timeout=None)
    jds = [j for j in c.get(f"{API}/config").json()["sample_jds"] if not only or j["id"] in only]
    rows = []
    for jd in jds:
        print(f"▶ {jd['id']} …", flush=True)
        rows.append(metrics(run(c, jd)))
        print(f"  {json.dumps(rows[-1])}", flush=True)

    fired = {k for r in rows for k in r["loops"]}
    gate = {
        "all runs completed": all(r["ok"] for r in rows),
        "all 5 loops fired": not only and {"clarify", "expand", "verify", "arbitrate", "review"} <= fired,
        # v1 gate = HR gets a usable order. Absolute calibration (spread) is reported, not gated: it needs gold labels.
        "top-5 ranking decisive (no unresolved ties)": all(r["decisive"] for r in rows),
        "token usage recorded": all(r["tokens"] for r in rows),
        "report mentions only ranked candidates": all(r["report_grounded"] for r in rows),
        "every CV has a status and a reason": all(r["every_cv_has_reason"] for r in rows),
    }
    print("\n| JD | ok | time (s) | loops fired | shortlist / KO | grounded 1st pass | spread | top-5 | LLM calls | tokens |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['jd']} | {'✅' if r['ok'] else '❌'} | {r['seconds']} | {r['loops']} | {r['shortlist']} / {r['knocked_out']} "
              f"| {r['grounded_first_pass']}% | {r['spread']} | {r['top']} | {r['llm_calls']} | {r['tokens']} |")
    print("\nRelease gate:")
    for k, passed in gate.items():
        print(f"  {'✅' if passed else '❌'} {k}")
    json.dump({"rows": rows, "gate": gate}, open("../.runtime/e2e-results.json", "w"), indent=1)
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
