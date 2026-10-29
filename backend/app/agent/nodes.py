"""Graph nodes. Each node: reads state, does ONE job, returns a partial update, and narrates its decisions
through `emit()` so the UI (and Langfuse) can show *why* the agent did what it did."""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache, wraps

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from langgraph.types import interrupt
from qdrant_client import models

from app.agent import prompts
from app.agent.schemas import CRITERIA, CandidateAssessment, JDRequirements, RankingReport
from app.agent.scoring import canonicalize, is_grounded, knockout_reason, mentions, related, weighted_total
from app.agent.state import AgentState
from app.config import get_settings
from app.providers.decider import decide
from app.providers.llm import create_llm, default_spec
from app.rag.corpus import get_doc, load_yaml
from app.rag.store import get_store

Node = Callable[[AgentState], Awaitable[dict]]


def emit(node: str, message: str, level: str = "info") -> None:
    get_stream_writer()({"type": "log", "node": node, "level": level, "message": message})


def traced(name: str):
    """Wrap a node with start/end events + timing (the UI's live graph is driven by these)."""

    def deco(fn: Node) -> Node:
        @wraps(fn)
        async def wrapper(state: AgentState) -> dict:
            write = get_stream_writer()
            write({"type": "node_start", "node": name})
            t0 = time.perf_counter()
            out = await fn(state)
            write({"type": "node_end", "node": name, "duration_ms": int((time.perf_counter() - t0) * 1000)})
            return out

        return wrapper

    return deco


@lru_cache
def _llm(spec: str) -> BaseChatModel:
    return create_llm(spec)


def llm(state: AgentState, role: str) -> BaseChatModel:
    return _llm(state.get("models", {}).get(role) or default_spec(role))


def cv_text(cv_id: str) -> str:
    return get_doc("cvs", cv_id).text


# ── 1. Ingest: JD → structured criteria (+ policy RAG + company weights) ──────────────────────────────
@traced("ingest")
async def ingest(state: AgentState) -> dict:
    s = get_settings()
    jd = state["jd_text"]
    if answers := state.get("clarifications"):
        jd += "\n\nHR clarifications (authoritative):\n" + "\n".join(f"- Q: {q}\n  A: {a}" for q, a in answers.items())

    req: JDRequirements = await llm(state, "reasoning").with_structured_output(JDRequirements).ainvoke(
        [SystemMessage(prompts.INGEST), HumanMessage(jd)]
    )
    # Harness boundary: LLM free text never drives deterministic logic directly — canonicalize to taxonomy keys.
    hard, soft, subs = canonicalize(req.must_have)
    _, nice, _ = canonicalize(req.nice_to_have)
    if hard != req.must_have:
        emit("ingest", f"Canonicalized must-have {req.must_have} → {hard}" + (f", soft {soft}" if soft else ""), "warn")
    req.must_have, req.nice_to_have = hard, list(dict.fromkeys(soft + nice))
    emit("ingest", f"Must-have: {req.must_have or '—'} · min years: {req.min_years} · depth: L{req.depth_level}")

    profile = state.get("company_profile", "default")
    where = models.Filter(must=[models.FieldCondition(key="company_profile", match=models.MatchAny(any=[profile, "all"]))])
    hits = get_store().hybrid_search("policies", jd, limit=3, where=where)
    emit("ingest", f"RAG: retrieved policies {[h.id for h in hits]} for profile '{profile}'")

    needs = False
    if state.get("loops", {}).get("clarify", 0) < s.max_clarifications:
        if not req.must_have:
            # Rule before model: with zero must-have tools there is no knock-out and no fair ranking — always ask.
            req.ambiguities = ["Which tools/frameworks are must-have for this role?", *req.ambiguities][:4]
            needs = True
            emit("ingest", "Rule: no concrete must-have tools → ask HR (no LLM needed)", "loop")
        elif req.ambiguities:
            # System-One gate: is it worth interrupting a human?
            d = await decide(
                llm(state, "fast"),
                "Do these open questions block a fair ranking enough to pause and ask HR?",
                ["proceed", "ask_hr"],
                f"JD:\n{jd}\n\nOpen questions:\n" + "\n".join(req.ambiguities),
            )
            needs = d.choice == "ask_hr"
            emit("ingest", f"Ambiguity gate → {d.choice} (p={d.confidence:.2f}): {d.reason}", "loop" if needs else "info")

    return {
        "requirements": req.model_dump(),
        "needs_clarification": needs,
        "weights": state.get("weights") or load_yaml("criteria_profiles.yaml")["profiles"][profile]["weights"],
        "policy_context": [{"source": h.id, "text": h.payload["text"]} for h in hits],
        "query_terms": state.get("query_terms") or list(dict.fromkeys(req.must_have + req.nice_to_have + req.languages)),
        "substitutes": subs,
    }


# ── L1: Human-in-the-loop clarification ───────────────────────────────────────────────────────────────
@traced("clarify")
async def clarify(state: AgentState) -> dict:
    # NOTE: on resume LangGraph re-runs this node from the top, so nothing expensive may happen before interrupt().
    questions = state["requirements"]["ambiguities"]
    answer = interrupt({"kind": "clarification", "payload": {"questions": questions}})
    emit("clarify", f"HR answered {len(answer['answers'])} question(s) → re-extracting criteria", "loop")
    return {"clarifications": answer["answers"], "loops": {"clarify": 1}}


# ── 2. Adaptive hybrid search + coverage check ───────────────────────────────────────────────────────
@traced("search")
async def search(state: AgentState) -> dict:
    s = get_settings()
    req, subs = state["requirements"], state.get("substitutes", {})
    query = f"{req['title']}. " + ", ".join(state["query_terms"])
    hits = get_store().hybrid_search("cvs", query, limit=s.search_limit)

    candidates, knocked = [], {}
    for h in hits:
        text = h.payload["text"]
        if reason := knockout_reason(text, req["must_have"], subs):
            knocked[h.id] = reason
        candidates.append(
            {
                "cv_id": h.id,
                "headline": h.payload.get("headline", ""),
                "retrieval_score": round(h.score, 4),
                "rerank_score": None,
                "matched_must_have": [m for m in req["must_have"] if mentions(text, m)],
            }
        )
    covered = len(candidates) - len(knocked)
    emit("search", f"Hybrid BM25+dense → {len(candidates)} CVs; {covered} cover all must-haves, {len(knocked)} knocked out")
    return {"candidates": candidates, "knocked_out": knocked}


# ── L2: query expansion via the ESCO-lite taxonomy (deterministic, zero tokens) ───────────────────────
@traced("expand_query")
async def expand_query(state: AgentState) -> dict:
    req, subs = state["requirements"], dict(state.get("substitutes", {}))
    candidates = state["candidates"]
    new_terms = []
    for m in req["must_have"]:
        have = sum(m in c["matched_must_have"] for c in candidates)
        extra = [r for r in related(m) if r not in subs.get(m, [])]
        if have < get_settings().min_coverage and extra:
            subs[m] = subs.get(m, []) + extra
            new_terms += extra
            emit("expand_query", f"'{m}' is scarce ({have} CVs) → accept related skills {extra}", "loop")
    return {
        "substitutes": subs,
        "query_terms": state["query_terms"] + [t for t in new_terms if t not in state["query_terms"]],
        "loops": {"expand": 1},
    }


# ── 3. Cross-encoder rerank → shortlist (cheap model decides who gets the expensive LLM) ──────────────
@traced("rerank")
async def rerank(state: AgentState) -> dict:
    s = get_settings()
    req = state["requirements"]
    query = f"{req['title']}. Required: {', '.join(req['must_have'] + req['languages'])}"
    eligible = [c for c in state["candidates"] if c["cv_id"] not in state["knocked_out"]]
    scores = get_store().rerank(query, [cv_text(c["cv_id"]) for c in eligible]) if eligible else []
    for c, sc in zip(eligible, scores):
        c["rerank_score"] = round(sc, 4)
    eligible.sort(key=lambda c: c["rerank_score"], reverse=True)
    shortlist = eligible[: s.shortlist_size]
    emit("rerank", f"Cross-encoder kept top {len(shortlist)}/{len(eligible)} for LLM assessment "
                   f"(saves {max(len(eligible) - len(shortlist), 0)} LLM calls)")
    knocked = [c for c in state["candidates"] if c["cv_id"] in state["knocked_out"]][:5]
    return {"candidates": shortlist + knocked, "retry_ids": [c["cv_id"] for c in shortlist]}


# ── 4. LLM assessment per candidate (concurrent) ──────────────────────────────────────────────────────
async def _assess_one(state: AgentState, cv_id: str, sem: asyncio.Semaphore) -> dict | None:
    """Failure isolation: one candidate's broken LLM output must not crash the run."""
    prev = state.get("assessments", {}).get(cv_id)
    for attempt in range(2):
        try:
            return await _assess_call(state, cv_id, prev, sem)
        except Exception as e:  # malformed / truncated structured output, provider hiccup
            emit("assess", f"{cv_id}: LLM output invalid ({type(e).__name__}), attempt {attempt + 1}/2", "warn")
    # keep the previous (partly ungrounded) assessment and stop retrying it; otherwise flag for human review
    return {**prev, "attempts": 99} if prev else None


async def _assess_call(state: AgentState, cv_id: str, prev: dict | None, sem: asyncio.Semaphore) -> dict:
    policy = "\n\n".join(p["text"] for p in state.get("policy_context", []))
    msgs = [
        SystemMessage(prompts.ASSESS + "\n\nCompany hiring policy:\n" + policy),
        HumanMessage(f"JD requirements:\n{json.dumps(state['requirements'], indent=1)}\n\nCV {cv_id}:\n{cv_text(cv_id)}"),
    ]
    if prev:
        bad = [e["quote"] for c in prev["criteria"].values() for e in c["evidence"] if e["grounded"] is False]
        msgs.append(HumanMessage(prompts.ASSESS_RETRY.format(bad="\n".join(f"- {b}" for b in bad))))
    async with sem:
        out: CandidateAssessment = await llm(state, "reasoning").with_structured_output(CandidateAssessment).ainvoke(msgs)
    return {
        "criteria": {
            k: {**v, "evidence": [{"quote": q, "grounded": None} for q in v["evidence"]]}
            for k, v in out.model_dump().items()
        },
        "knocked_out": False,
        "knockout_reason": None,
        "attempts": (prev or {}).get("attempts", 0) + 1,
    }


@traced("assess")
async def assess(state: AgentState) -> dict:
    ids = state["retry_ids"]
    sem = asyncio.Semaphore(get_settings().llm_concurrency)
    results = dict(zip(ids, await asyncio.gather(*(_assess_one(state, i, sem) for i in ids))))
    ok = {k: v for k, v in results.items() if v}
    failed = {k: "Assessment failed (invalid LLM output) — needs manual review" for k, v in results.items() if not v}
    emit("assess", f"LLM scored {len(ok)} CV(s) on 8 criteria with evidence quotes"
                   + (f"; {len(failed)} flagged for manual review" if failed else ""))
    out = {"assessments": ok, "retry_ids": list(ok)}
    if failed:
        out["knocked_out"] = {**state["knocked_out"], **failed}
    return out


# ── L4: verify_evidence — every quote must exist in the CV (anti-hallucination) ───────────────────────
@traced("verify_evidence")
async def verify_evidence(state: AgentState) -> dict:
    s = get_settings()
    updated, retry, total, bad_total = {}, [], 0, 0
    for cv_id in state["retry_ids"]:
        a = state["assessments"][cv_id]
        text = cv_text(cv_id)
        bad = 0
        for crit in a["criteria"].values():
            for e in crit["evidence"]:
                e["grounded"] = is_grounded(e["quote"], text)
                bad += not e["grounded"]
                total += 1
        bad_total += bad
        if bad and a["attempts"] <= s.max_verify_retries:
            retry.append(cv_id)
        elif bad:  # out of retries: claims without grounded evidence lose their credit
            for crit in a["criteria"].values():
                if crit["evidence"] and not any(e["grounded"] for e in crit["evidence"]):
                    crit["score"] = min(crit["score"], 20)
                    crit["rationale"] += " [capped: evidence not found in CV]"
        updated[cv_id] = a
    grounded_pct = 100 * (total - bad_total) / total if total else 100
    emit("verify_evidence", f"{total - bad_total}/{total} quotes grounded ({grounded_pct:.1f}%)")
    if retry:
        emit("verify_evidence", f"Hallucinated quotes in {retry} → sending back to assess", "loop")
    return {"assessments": updated, "retry_ids": retry, "loops": {"verify": 1 if retry else 0}}


# ── 5. Score: Score = 1[H(c)] × Σ w·S  (deterministic) ────────────────────────────────────────────────
@traced("score")
async def score(state: AgentState) -> dict:
    heads = {c["cv_id"]: c["headline"] for c in state["candidates"]}
    rows = [
        {"cv_id": cid, "headline": heads.get(cid, ""), "knocked_out": False, "arbiter_note": None,
         "total": weighted_total(a, state["weights"], False)}
        for cid, a in state["assessments"].items() if cid in heads
    ]
    rows += [
        {"cv_id": cid, "headline": heads[cid], "knocked_out": True, "arbiter_note": reason, "total": 0.0}
        for cid, reason in state["knocked_out"].items() if cid in heads
    ]
    rows.sort(key=lambda r: r["total"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    emit("score", "Weighted totals: " + ", ".join(f"{r['cv_id']}={r['total']:.2f}" for r in rows[:5]))
    return {"ranking": rows}


# ── L3: selective arbiter — only spend LLM on near-ties ───────────────────────────────────────────────
@traced("arbitrate")
async def arbitrate(state: AgentState) -> dict:
    s = get_settings()
    rows = [dict(r) for r in state["ranking"]]
    calls = 0
    for i in range(min(5, len(rows)) - 1):
        a, b = rows[i], rows[i + 1]
        if a["knocked_out"] or b["knocked_out"] or a["total"] - b["total"] >= s.near_tie_delta:
            continue

        def evidence(cid: str) -> str:
            crit = state["assessments"][cid]["criteria"]
            return "\n".join(f"{k} {v['score']}: " + "; ".join(e["quote"] for e in v["evidence"] if e["grounded"])
                             for k, v in crit.items())

        d = await decide(
            llm(state, "reasoning"),
            prompts.ARBITER,
            [a["cv_id"], b["cv_id"]],
            f"JD: {json.dumps(state['requirements'])}\n\n{a['cv_id']}:\n{evidence(a['cv_id'])}\n\n"
            f"{b['cv_id']}:\n{evidence(b['cv_id'])}",
        )
        calls += 1
        winner = d.choice
        note = f"Near-tie (Δ={a['total'] - b['total']:.3f}) resolved by arbiter: {winner} — {d.reason}"
        if winner == b["cv_id"]:
            rows[i], rows[i + 1] = b, a
        rows[i]["arbiter_note"] = note
        rows[i + 1]["arbiter_note"] = f"Placed below {winner} by arbiter (Δ={a['total'] - b['total']:.3f})"
        emit("arbitrate", note, "loop")
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    if not calls:
        emit("arbitrate", "No near-ties in the top 5 → arbiter skipped (0 LLM calls)")
    return {"ranking": rows, "loops": {"arbitrate": calls}}


# ── 6. Explainable report ─────────────────────────────────────────────────────────────────────────────
@traced("report")
async def report(state: AgentState) -> dict:
    top = [r for r in state["ranking"] if not r["knocked_out"]][:5]
    if not top:  # don't pay an LLM to describe an empty pool
        emit("report", "No candidate passed the knock-out → deterministic report", "warn")
        return {"report": {"summary": "No candidate met the must-have requirements. Consider relaxing must-haves "
                                      "or accepting related skills.", "candidates": []}}
    facts = []
    for r in top:
        crit = state["assessments"][r["cv_id"]]["criteria"]
        facts.append({
            "cv_id": r["cv_id"], "rank": r["rank"], "total": r["total"], "arbiter_note": r["arbiter_note"],
            "criteria": {k: {"score": v["score"], "rationale": v["rationale"],
                             "verified_evidence": [e["quote"] for e in v["evidence"] if e["grounded"]]}
                         for k, v in crit.items()},
        })
    out: RankingReport = await llm(state, "reasoning").with_structured_output(RankingReport).ainvoke(
        [SystemMessage(prompts.REPORT),
         HumanMessage(f"JD: {json.dumps(state['requirements'])}\nWeights: {state['weights']}\n\n"
                      f"Ranked candidates:\n{json.dumps(facts, indent=1)}")]
    )
    emit("report", f"Report ready for top {len(top)} candidates")
    return {"report": out.model_dump()}


# ── L5: HR review — approve, or re-weight and re-score (no new LLM assessment needed) ─────────────────
@traced("hr_review")
async def hr_review(state: AgentState) -> dict:
    answer = interrupt({"kind": "review", "payload": {"ranking": state["ranking"], "weights": state["weights"]}})
    if answer.get("action") != "adjust":
        emit("hr_review", "HR approved the ranking")
        return {"hr_action": "approve"}
    w = {k: max(float(answer["weights"].get(k, 0)), 0.0) for k in CRITERIA}
    total = sum(w.values()) or 1.0
    w = {k: round(v / total, 4) for k, v in w.items()}
    emit("hr_review", f"HR re-weighted criteria → re-scoring with {w}", "loop")
    return {"hr_action": "adjust", "weights": w, "loops": {"review": 1}}
