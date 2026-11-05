"""Graph nodes. Each node: reads state, does ONE job, returns a partial update, and narrates its decisions
through `emit()` so the UI (and Langfuse) can show *why* the agent did what it did."""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from functools import lru_cache, wraps

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_config, get_stream_writer
from langgraph.types import interrupt
from qdrant_client import models

from app.agent import checks, prompts
from app.agent import investigate as inv
from app.agent.schemas import (
    CRITERIA,
    CandidateAssessment,
    HrTexts,
    JDRequirements,
    RankingReport,
    WeightInstruction,
)
from app.agent.scoring import (
    apply_instruction,
    calibrate_weights,
    canonicalize,
    check_evidence,
    contrastive,
    counterfactuals,
    decomposition,
    depth_score,
    fallback_uses,
    foreign_ids,
    knockout_reason,
    preferences,
    related,
    rule_criteria,
    rule_total,
    tiebreak_key,
    weighted_total,
)
from app.agent.state import AgentState
from app.config import get_settings
from app.nlp.jd import parse as parse_jd
from app.nlp import quality
from app.nlp.injection import sanitize
from app.nlp.taxonomy import group
from app.nlp.pipeline import profile
from app.providers.decider import decide
from app.providers.llm import create_llm, default_spec
from app.rag import retrieve
from app.rag.corpus import get_doc, load_dir, load_yaml
from app.rag.store import get_store

NEEDS_REVIEW = "Assessment failed (invalid LLM output) — needs manual review"

Node = Callable[[AgentState], Awaitable[dict]]


def emit(node: str, message: str, level: str = "info", **metrics) -> None:
    """Narrate a decision. `metrics` (optional) carry the numbers behind it for the eval scorecard (Q-ids)."""
    get_stream_writer()({"type": "log", "node": node, "level": level, "message": message, **({"metrics": metrics} if metrics else {})})


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


async def for_hr(state: AgentState, texts: list[str]) -> list[str]:
    """LLM free text HR reads, in the UI language (Q18). English or nothing to do: unchanged. A translation that
    drops an item, a number or a tool is not used: HR then reads the English original."""
    if state.get("lang") != "vi" or not texts:
        return texts
    try:
        out: HrTexts = await llm(state, "fast").with_structured_output(HrTexts).ainvoke(
            [SystemMessage(prompts.TO_VIETNAMESE), HumanMessage(json.dumps(texts, ensure_ascii=False))])
    except Exception as e:  # a translation never costs the run
        emit("ingest", f"Translation for HR failed ({type(e).__name__}) → English kept", "warn")
        return texts
    if issues := checks.translation_issues(texts, out.items):
        emit("ingest", f"Translation for HR rejected {issues} → English kept", "warn")
        return texts
    return out.items


def prof(state: AgentState, cv_id: str):
    """The parsed CV plus any facts the investigator verified (N12): what every rule below reads."""
    return inv.with_facts(profile(cv_id), state.get("recovered", {}).get(cv_id, []))


def off(state: AgentState, part: str) -> bool:
    """Ablation switch (evaluation only): is this part of the agent turned off for this run?"""
    return part in state.get("ablate", [])


async def extract_requirements(model: BaseChatModel, jd: str, mode: str | None = None
                               ) -> tuple[JDRequirements, dict[str, list[str]], list[str]]:
    """JD → requirements on taxonomy keys, JD-stated alternatives, and the raw must-have phrases.
    mode (Settings.jd_parser): rules = the deterministic parser (app.nlp.jd) only, no LLM; llm = the v2 LLM extraction,
    canonicalized; hybrid = the parser wins, the LLM adds title and ambiguities and fills fields the parser left empty."""
    mode = mode or get_settings().jd_parser
    parsed = parse_jd(jd)
    if mode == "rules":
        req = JDRequirements(title=parsed["title"], must_have=parsed["must_have"], nice_to_have=parsed["nice_to_have"],
                             min_years=parsed["min_years"], depth_level=parsed["depth_level"],
                             languages=parsed["languages"], test_types=parsed["test_types"], ambiguities=[],
                             foreign_languages=parsed["foreign_languages"], domains=parsed["domains"])
        return req, parsed["substitutes"], parsed["must_have"]
    req: JDRequirements = await model.with_structured_output(JDRequirements).ainvoke(
        # Never localized: its title and lists feed retrieval and scoring, and the UI language must not move a rank
        [SystemMessage(prompts.INGEST), HumanMessage(jd)]
    )
    # Harness boundary: LLM free text never drives deterministic logic directly — canonicalize to taxonomy keys.
    raw = req.must_have
    hard, soft, subs = canonicalize(req.must_have)
    n_hard, n_soft, n_subs = canonicalize(req.nice_to_have)  # every tool named as a plus counts, "or" or not
    nice = [k for k in dict.fromkeys(soft + n_hard + n_soft + [s for v in n_subs.values() for s in v]) if k not in hard]
    req.must_have, req.nice_to_have = hard, nice
    # The JD's own duties, read with the CV depth lexicon, set a floor: "Design the shared framework" is at least L3
    # whatever the LLM wrote (an 8B model often answers L1 for any "Automation Test Engineer").
    req.depth_level = max(req.depth_level or 0, parsed["depth_level"] or 0) or None
    req.foreign_languages, req.domains = parsed["foreign_languages"], parsed["domains"]
    if mode == "hybrid" and parsed["must_have"]:  # a structured JD: the parser is literal and reproducible
        req.must_have, subs = parsed["must_have"], parsed["substitutes"]
        req.nice_to_have = parsed["nice_to_have"] or req.nice_to_have
        req.min_years = parsed["min_years"] or req.min_years
        req.languages = parsed["languages"] or req.languages
        req.test_types = parsed["test_types"] or req.test_types
    return req, subs, raw


# ── 1. Ingest: JD → structured criteria (+ policy RAG + company weights) ──────────────────────────────
@traced("ingest")
async def ingest(state: AgentState) -> dict:
    s = get_settings()
    jd, planted = sanitize(state["jd_text"], "job description")  # a pasted JD is untrusted text too (Q16)
    if planted:
        emit("ingest", f"Removed {planted} instruction-like sentence(s) from the JD before any LLM saw it", "warn")
    if answers := state.get("clarifications"):
        jd += "\n\nHR clarifications (authoritative):\n" + "\n".join(f"- Q: {q}\n  A: {a}" for q, a in answers.items())

    req, subs, raw = await extract_requirements(llm(state, "reasoning"), jd)
    req.ambiguities = await for_hr(state, req.ambiguities)  # HR reads these questions; the requirements stay English
    if raw != req.must_have:
        emit("ingest", f"Canonicalized must-have {raw} → {req.must_have}", "warn")
    emit("ingest", f"Must-have: {req.must_have or '—'} · min years: {req.min_years} · depth: L{req.depth_level}")

    company = "default" if off(state, "profile") else state.get("company_profile", "default")
    where = models.Filter(must=[models.FieldCondition(key="company_profile", match=models.MatchAny(any=[company, "all"]))])
    hits = get_store().hybrid_search("policies", jd, limit=3, where=where)
    emit("ingest", f"RAG: retrieved policies {[h.id for h in hits]} for profile '{company}'")

    needs = False
    if state.get("loops", {}).get("clarify", 0) < s.max_clarifications and not off(state, "clarify"):
        needs, why = await ask_hr_gate(llm(state, "fast"), jd, req)
        if why:
            emit("ingest", why, "loop" if needs else "info")

    return {
        "requirements": req.model_dump(),
        "needs_clarification": needs,
        "weights": state.get("weights") or load_yaml("criteria_profiles.yaml")["profiles"][company]["weights"],
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


# ── 2. Requirement-wise hybrid search over CV chunks + knock-out screening ────────────────────────────
@traced("search")
async def search(state: AgentState) -> dict:
    s = get_settings()
    req, subs = state["requirements"], state.get("substitutes", {})
    qs = retrieve.queries(req, state["query_terms"], subs)
    pool = [d.id for d in load_dir("cvs")]
    # The index may hold more CVs than the pool (a smaller DATA_DIR, removed uploads): only the pool is ranked.
    in_pool = set(pool)
    hits = [h for h in retrieve.rank_cvs(get_store(), retrieve.CHUNKS, qs) if h.cv_id in in_pool]
    found = {h.cv_id for h in hits}
    order = hits[: s.search_limit]
    screened = {h.cv_id: f"Not retrieved: relevance rank {i} of {len(hits)} is below the search limit ({s.search_limit})"
                for i, h in enumerate(hits, 1) if h not in order}
    screened |= {cid: "Not retrieved: no passage matched any requirement" for cid in pool if cid not in found}

    knocked = {cid: r for cid in pool if (r := knockout_reason(prof(state, cid), req["must_have"], subs))}  # whole pool: cheap
    added = []
    if s.rule_recall:  # rules are free, so they screen the whole pool too: retrieval misses facts, rules don't
        rule = {cid: rule_total(prof(state, cid), req, subs, state["weights"], state.get("fallbacks", {}))
                for cid in pool if cid not in knocked}
        have, by_id = {h.cv_id for h in order}, {h.cv_id: h for h in hits}
        added = [cid for cid in sorted(rule, key=lambda c: (-rule[c], c))[: s.search_limit] if cid not in have]
        order += [by_id.get(cid) or retrieve.CVHit(cid, 0.0, {}) for cid in added]
    candidates = []
    for h in order:
        p = prof(state, h.cv_id)
        candidates.append({
            "cv_id": h.cv_id, "headline": p.headline, "lang": p.lang,
            "retrieval_score": round(h.score, 4), "rerank_score": None,
            "matched_must_have": [m for m in req["must_have"] if (b := p.best(m)) and b.strength != "hedged"],
            "evidence": {label: list(v) for label, v in h.best.items()},  # query → [chunk_id, score]
        })
    screened = {cid: r for cid, r in screened.items() if cid not in knocked and cid not in added}
    hedged = sum("hedged" in r for r in knocked.values())
    covered = sum(c["cv_id"] not in knocked for c in candidates)
    emit("search", f"{len(qs)} requirement queries over {len(pool)} CVs (structural chunks, BM25+dense) → "
                   f"{len(candidates)} candidates ({len(added)} added by rule score that retrieval missed), "
                   f"{covered} of them cover all must-haves; knock-out screened the whole "
                   f"pool: {len(knocked)} out" + (f" ({hedged} only had hedged mentions)" if hedged else ""))
    return {"candidates": candidates, "knocked_out": knocked, "screened": screened}


JUNIOR = re.compile(r"\b(?:junior|fresher|intern|entry[- ]level|graduate|thực tập|mới tốt nghiệp)\b", re.I)
SENIOR = re.compile(r"\b(?:senior|lead|principal|staff|architect|head|trưởng nhóm)\b", re.I)

# Vague-JD detection (Literature Review §1.1 Mức 4, §1.4): Automation Test JDs that leave out specifics the system
# needs for knock-out and scoring. Each criterion has three states: clear (skip), vague/missing (ask).
EXPERIENCE_CUE = re.compile(r"\b(?:experienc\w*|years?|kinh\s*nghiệm|năm)\b", re.I)
# Groups whose tools already reveal the kind of testing, so asking "which type?" would be redundant (fix B).
TYPE_GROUPS = {"web_e2e": "web", "mobile": "mobile", "api": "api", "performance": "performance"}
# A foreign language the JD mentions but hedges between must-have and nice-to-have (JD-003: "a plus, or maybe
# required depending on the client"). "a plus"/"preferred" alone is a clear nice-to-have, so it is NOT a hedge.
FOREIGN_WORDS = re.compile(r"\benglish\b|tiếng\s*anh|toeic|ielts|toefl|\bjapanese\b|tiếng\s*nhật|jlpt", re.I)
LANG_HEDGE = re.compile(r"\b(?:maybe|may\s+be|might|depend(?:s|ing)?|or\s+(?:maybe\s+)?required)\b|tùy|có\s*thể", re.I)
MANY_MUST_HAVES = 6  # a "must-have" list longer than this cannot realistically be all knock-outs (fix D)
# Taxonomy keys that name a test TYPE ("API testing"), not a concrete tool/framework. They reveal the kind of
# testing (so vagueness_questions needn't ask "which type?"), but they are NOT a named knock-out tool: a JD whose
# only must-have is an umbrella phrase still hasn't named a required tool, so L1 must ask for one (fix #1; see the
# false-negative noted in docs/CLARIFY_RESULTS.md where a vague twin collapsed to ['api_testing'] and nobody asked).
TYPE_UMBRELLA = {"api_testing"}


def concrete_must_have(must_have: list[str]) -> list[str]:
    """must-have entries that actually name a tool/framework — a test-type umbrella ("api testing") is not one."""
    return [m for m in must_have if m not in TYPE_UMBRELLA]


def seniority_conflict(title: str, min_years: int | None) -> str | None:
    """A JD that contradicts itself ("Junior" but 4+ years) cannot be ranked fairly without asking which one holds."""
    if min_years is not None and min_years >= 3 and JUNIOR.search(title or ""):
        return f"The title says junior but the JD asks for {min_years}+ years: which one applies?"
    if min_years is not None and min_years <= 1 and SENIOR.search(title or ""):
        return f"The title says senior but the JD asks for only {min_years} year(s): which one applies?"
    return None


def _foreign_language_hedged(jd: str) -> bool:
    """True if a JD line mentions a foreign language AND hedges whether it is required — the must/nice status
    changes the outsourcing weight a lot (Literature Review §1.4, Bảng 4: English ≥ B2 is worth 0.20 there)."""
    return any(FOREIGN_WORDS.search(ln) and LANG_HEDGE.search(ln) for ln in jd.splitlines())


def vagueness_questions(jd: str, req: JDRequirements) -> list[str]:
    """Detect domain-specific vagueness/omissions in an Automation Test JD and generate targeted questions for HR.

    Literature Review §1.1 Mức 4: depth L1–L4 is invisible from a tool name alone (ii), and QA years ≠ automation
    years (iv); §1.4: foreign language and test type carry company-specific weight. These are *soft* questions: on
    their own they don't force asking, but they enrich the single clarification round.
    """
    qs: list[str] = []
    # Years — vague ("several years") or omitted, but only when the JD signals experience matters at all
    if req.min_years is None and EXPERIENCE_CUE.search(jd):
        qs.append("How many years of test automation experience are required (not total QA)?")
    # Depth L1–L4 — a tool name ("Selenium") doesn't say run vs write scripts vs build a framework
    if req.depth_level is None and req.must_have:
        qs.append("What level of automation ownership is expected: writing test scripts on an existing framework, "
                  "or building/maintaining the framework and CI integration?")
    # Test type — ask only when NEITHER the duties NOR the must-have tools reveal it (fix B)
    tool_types = {TYPE_GROUPS[g] for m in req.must_have if (g := group(m)) in TYPE_GROUPS}
    if not req.test_types and not tool_types:
        qs.append("Which types of testing: web UI, mobile, API, performance, or a combination?")
    # Foreign language present but hedged between required and preferred (fix C)
    if req.foreign_languages and _foreign_language_hedged(jd):
        names = ", ".join(lang.title() for lang in req.foreign_languages)
        qs.append(f"Is {names} a hard requirement or just preferred?")
    # Laundry list — too many must-haves to all be genuine knock-outs (fix D)
    if len(req.must_have) >= MANY_MUST_HAVES:
        qs.append(f"The JD lists {len(req.must_have)} must-have tools; which are truly required vs nice-to-have?")
    # No programming language and no must-have tools that imply one
    if not req.languages and not req.must_have:
        qs.append("Which programming language does the team use for test automation?")
    return qs


async def ask_hr_gate(model: BaseChatModel, jd: str, req: JDRequirements) -> tuple[bool, str | None]:
    """L1 decision: pause for HR or proceed? (mutates req.ambiguities when it asks). → (ask, narration)."""
    soft = vagueness_questions(jd, req)

    def enrich(lead: list[str]) -> None:
        """Order the one clarification round (max_clarifications=1, so HR is asked at most once): rule questions
        first (lead), then the domain-specific soft ones, then the LLM's own — deduplicated, capped at 4 (fixes A, F)."""
        ordered: list[str] = []
        for q in [*lead, *soft, *req.ambiguities]:
            if q.lower() not in {x.lower() for x in ordered}:
                ordered.append(q)
        req.ambiguities = ordered[:4]

    if not concrete_must_have(req.must_have):
        # Rule before model: with no named tool (or only a test-type umbrella like "API testing") there is no
        # knock-out and no fair ranking — always ask.
        enrich(["Which specific tools or frameworks are must-have for this role?"])
        return True, "Rule: no concrete must-have tools → ask HR (no LLM needed)"
    if conflict := seniority_conflict(req.title, req.min_years):
        enrich([conflict])
        return True, f"Rule: {conflict} → ask HR (no LLM needed)"

    enrich([])  # soft questions + the LLM's own ambiguities
    if not req.ambiguities:
        return False, None
    # System-One gate: is it worth interrupting a human?
    d = await decide(model, "Do these open questions block a fair ranking enough to pause and ask HR?",
                     ["proceed", "ask_hr"], f"JD:\n{jd}\n\nOpen questions:\n" + "\n".join(req.ambiguities))
    return d.choice == "ask_hr", f"Ambiguity gate → {d.choice} (p={d.confidence:.2f}): {d.reason}"


# ── L6: investigate — a tool-using agent for knock-outs the parser itself may have caused ──────────────
def investigable(state: AgentState) -> list[tuple[str, list[str], str]]:
    """Knock-outs that look like parse failures, closest to passing first: (cv_id, missing must-haves, why)."""
    req, subs, done = state["requirements"], state.get("substitutes", {}), state.get("investigated", {})
    todo = []
    for cid, reason in state["knocked_out"].items():
        if cid in done or reason == NEEDS_REVIEW:
            continue
        p = prof(state, cid)
        missing = [m for m in req["must_have"] if knockout_reason(p, [m], subs)]
        if missing and (why := inv.triggers(p, missing, subs)):
            todo.append((len(missing), cid, missing, why))
    return [t[1:] for t in sorted(todo)]


@traced("investigate")
async def investigate(state: AgentState) -> dict:
    return {} if off(state, "investigate") else await investigate_some(state)


async def investigate_some(state: AgentState, only: list[str] | None = None) -> dict:
    """The L6 body; `only` limits it to the CVs the P7 orchestrator chose (still only where a trigger fires)."""
    s = get_settings()
    req, subs = state["requirements"], state.get("substitutes", {})
    todo = [t for t in investigable(state) if only is None or t[0] in only][: s.max_investigations]
    if not todo:
        emit("investigate", "No knock-out looks like a parsing failure (no garbled text, no misspelt must-have) → "
                            "investigator not needed (0 LLM calls)", investigated=0, recovered=0, calls=0, invalid=0)
        return {}
    for cid, missing, why in todo:
        emit("investigate", f"{cid}: knocked out for {missing}, but {why} → investigating "
                            f"(≤ {s.max_investigation_steps} tool calls)", "loop")
    sem = asyncio.Semaphore(s.llm_concurrency)

    async def one(cid: str, missing: list[str], why: str) -> dict:
        async with sem:
            return await inv.investigate_cv(llm(state, "reasoning"), prof(state, cid), req.get("title", ""), missing,
                                            subs, why, s.max_investigation_steps)

    results = await asyncio.gather(*(one(*t) for t in todo))
    knocked, screened, cands = dict(state["knocked_out"]), dict(state.get("screened", {})), list(state["candidates"])
    recovered, outcome = {}, {}
    for (cid, missing, why), res in zip(todo, results):
        for t in res["trace"]:
            emit("investigate", f"{cid} · {t['tool']}({t['arg'][:40]}) — {t['thought']}"
                                + (f" → {t['observation'].splitlines()[0][:120]}" if t["observation"] else ""))
        recovered[cid] = res["facts"]
        p = inv.with_facts(profile(cid), state.get("recovered", {}).get(cid, []) + res["facts"])
        found = [f["skill"] for f in res["facts"]]
        if (r := knockout_reason(p, req["must_have"], subs)) is None:
            knocked.pop(cid, None)
            screened.pop(cid, None)
            if cid not in {c["cv_id"] for c in cands}:
                cands.append({"cv_id": cid, "headline": p.headline, "lang": p.lang, "retrieval_score": 0.0,
                              "rerank_score": None, "evidence": {},
                              "matched_must_have": [m for m in req["must_have"] if (b := p.best(m)) and b.strength != "hedged"]})
            outcome[cid] = f"recovered {found} from the CV text (verified quotes) → back in the pool"
        else:
            knocked[cid] = f"{r} (investigated{': found ' + str(found) if found else ''}; knock-out confirmed)"
            outcome[cid] = "knock-out confirmed" + (f"; found {found}" if found else "")
        emit("investigate", f"{cid}: {outcome[cid]}" + (" [invalid model output: stopped]" if res["invalid"] else ""),
             "loop" if found else "info")
    n_rec = sum("recovered" in o for o in outcome.values())
    emit("investigate", f"Investigated {len(todo)} knock-out(s) in {sum(r['calls'] for r in results)} LLM calls: "
                        f"{n_rec} back in the pool, {len(todo) - n_rec} confirmed",
         investigated=len(todo), recovered=n_rec, calls=sum(r["calls"] for r in results),
         invalid=sum(r["invalid"] for r in results), facts=sum(len(r["facts"]) for r in results))
    return {"recovered": recovered, "investigated": outcome, "knocked_out": knocked, "screened": screened,
            "candidates": cands, "loops": {"investigate": len(todo)}}


# ── L2: query expansion via the taxonomy — HR approves substitutes (zero tokens) ─────────────────────
def substitute_proposals(state: AgentState) -> list[dict]:
    """Must-haves too few candidates cover, with related skills HR has not yet accepted or declined."""
    req, subs, declined = state["requirements"], state.get("substitutes", {}), state.get("declined", {})
    proposals = []
    for m in req["must_have"]:
        have = sum(m in c["matched_must_have"] for c in state["candidates"])
        extra = [r for r in related(m) if r not in subs.get(m, []) and r not in declined.get(m, [])]
        if have < get_settings().min_coverage and extra:
            proposals.append({"must_have": m, "covered": have, "related": extra})
    return proposals


@traced("expand_query")
async def expand_query(state: AgentState) -> dict:
    # Cheap and deterministic, so it is safe to recompute when LangGraph re-runs this node on resume.
    req, subs = state["requirements"], state.get("substitutes", {})
    proposals = substitute_proposals(state)
    if not proposals:
        return {"loops": {"expand": 1}}
    if get_settings().ask_hr_substitutes:
        answer = interrupt({"kind": "substitutes", "payload": {"proposals": proposals}})
        wanted = answer.get("accept", {})
    else:
        wanted = {p["must_have"]: p["related"] for p in proposals}
    accept, reject = {}, {}
    for p in proposals:  # canonicalize the human answer too: only proposed skills can be accepted
        ok = [r for r in wanted.get(p["must_have"], []) if r in p["related"]]
        accept[p["must_have"]] = subs.get(p["must_have"], []) + ok
        reject[p["must_have"]] = [r for r in p["related"] if r not in ok]
        emit("expand_query", f"'{p['must_have']}' is scarce ({p['covered']} CVs) → accepted substitutes {ok or '—'}"
                             + (f", declined {reject[p['must_have']]}" if reject[p["must_have"]] else ""), "loop")
    new_terms = [t for v in accept.values() for t in v if t not in state["query_terms"]]
    fallbacks = {m: [r for r in v if r not in subs.get(m, [])] for m, v in accept.items()}
    return {"substitutes": accept, "declined": reject, "fallbacks": fallbacks, "query_terms": state["query_terms"] + new_terms,
            "loops": {"expand": 1}}


# ── 3. Cross-encoder rerank on evidence passages → shortlist ──────────────────────────────────────────
@traced("rerank")
async def rerank(state: AgentState) -> dict:
    s = get_settings()
    req = state["requirements"]
    qs = retrieve.queries(req, state["query_terms"], state.get("substitutes", {}))
    eligible = [dict(c) for c in state["candidates"] if c["cv_id"] not in state["knocked_out"]]
    hits = [retrieve.CVHit(c["cv_id"], c["retrieval_score"], {k: tuple(v) for k, v in c["evidence"].items()})
            for c in eligible]
    scores = retrieve.rerank_cvs(get_store(), qs, hits, {c["cv_id"]: prof(state, c["cv_id"]) for c in eligible})
    rules = {c["cv_id"]: rule_total(prof(state, c["cv_id"]), req, state.get("substitutes", {}), state["weights"],
                                    state.get("fallbacks", {}))
             for c in eligible}
    for c in eligible:
        c["rerank_score"], c["rule_score"] = scores.get(c["cv_id"]), rules[c["cv_id"]]
    # Shortlist on two zero-token signals: passage relevance (cross-encoder) and the parsed-CV rule score, each
    # min-max normalised. B3 vs "R structural + CE" showed the rule score orders candidates better (τ); CE finds them.
    norm = lambda xs: {k: (v - min(xs.values())) / ((max(xs.values()) - min(xs.values())) or 1) for k, v in xs.items()}  # noqa: E731
    ce_n, ru_n = norm(scores) if scores else {}, norm(rules) if rules else {}
    eligible.sort(key=lambda c: (ce_n.get(c["cv_id"], 0) + ru_n.get(c["cv_id"], 0)) / 2, reverse=True)
    shortlist = eligible[: s.shortlist_size]
    screened = dict(state.get("screened", {}))
    for i, c in enumerate(eligible[s.shortlist_size:], s.shortlist_size + 1):
        missing = [m for m in req["must_have"] if m not in c["matched_must_have"]]
        screened[c["cv_id"]] = (f"Not shortlisted: rank {i} of {len(eligible)} on passage relevance + rule score"
                                f" (shortlist = top {s.shortlist_size})" + (f"; via substitutes for {missing}" if missing else ""))
    emit("rerank", f"Cross-encoder scored {len(eligible)} CVs on their evidence passages per requirement, fused with "
                   f"the rule score; kept top "
                   f"{len(shortlist)} for LLM assessment (saves {max(len(eligible) - len(shortlist), 0)} LLM calls)")
    rest = [c for c in state["candidates"] if c["cv_id"] not in {e["cv_id"] for e in eligible}]
    return {"candidates": eligible + rest, "retry_ids": [c["cv_id"] for c in shortlist], "screened": screened}


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
    req, p = state["requirements"], prof(state, cv_id)
    policy = "\n\n".join(x["text"] for x in state.get("policy_context", []))
    msgs = [
        SystemMessage(prompts.localized(prompts.ASSESS, state.get("lang")) + "\n\nCompany hiring policy:\n" + policy),
        HumanMessage(f"JD requirements:\n{json.dumps(req, indent=1)}\n\n{p.facts_for_llm()}\n\n"
                     f"CV {cv_id} (each passage is labelled with the section the parser found it in):\n{p.for_llm()}"),
    ]
    if prev and prev.get("llm_review"):  # tell the model exactly what failed: since v3 its quotes live in llm_review, not in the rule criteria
        review = prev.get("llm_review", {})
        bad = [f'quote not in the CV: "{e["quote"]}"' for c in review.values() for e in c["evidence"] if e["grounded"] is False]
        bad += [f"{k}: {i}" for k, issues in prev.get("checks", {}).items() for i in issues]
        msgs.append(HumanMessage(prompts.ASSESS_RETRY.format(bad="\n".join(f"- {b}" for b in bad))))
    async with sem:
        out: CandidateAssessment = await llm(state, "reasoning").with_structured_output(CandidateAssessment).ainvoke(msgs)
    # Rules before models: all 8 criteria come from the parsed CV at zero tokens. The LLM's judgement is kept as a
    # verified review that explains the candidate; it sets scores only when Settings.llm_in_score (the v2 design,
    # which ranked 0.033 nDCG@10 below its own deterministic input, docs/EVAL.md §7).
    rules = rule_criteria(p, req, state.get("substitutes", {}), state.get("fallbacks", {}))
    facts = out.model_dump()
    depth = facts.pop("automation_depth")
    facts["automation_depth"] = {"score": depth_score(depth["level"], req.get("depth_level")),
                                 "rationale": f"L{depth['level']}: {depth['rationale']}", "evidence": depth["evidence"]}
    if depth["level"] >= p.depth_cue + 2:
        emit("assess", f"{cv_id}: LLM says L{depth['level']} but the CV's own action verbs show at most "
                       f"L{p.depth_cue} — verify_evidence will check where the proof comes from", "warn")
    llm_part = {k: {**v, "evidence": [{"quote": q, "grounded": None, "supported": None, "section": None}
                                      for q in v["evidence"]]} for k, v in facts.items()}
    in_score = get_settings().llm_in_score
    return {
        "criteria": {k: (llm_part if in_score and k in llm_part else rules)[k] for k in CRITERIA},
        "llm_review": llm_part,
        "facts": {"depth_level": depth["level"], "depth_cue": p.depth_cue, "years": p.years_automation or 0.0,
                  "years_total": p.years_total},
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
    failed = {k: NEEDS_REVIEW for k, v in results.items() if not v}
    emit("assess", f"LLM reviewed {len(ok)} CV(s): depth fact + 4 criteria with evidence quotes; scores: "
                   + ("LLM for those 5, rules for 3" if get_settings().llm_in_score else "all 8 by rule")
                   + (f"; {len(failed)} flagged for manual review" if failed else ""))
    out = {"assessments": ok, "retry_ids": list(ok)}
    if failed:
        out["knocked_out"] = {**state["knocked_out"], **failed}
    return out


# ── L4: verify_evidence — quotes must exist (grounded) and prove the criterion (supported) ──────────────
@traced("verify_evidence")
async def verify_evidence(state: AgentState) -> dict:
    s = get_settings()
    act = not off(state, "verify")  # ablation A1: still measure, but never retry or cap
    updated, retry, total, bad_total, unsupported, claims, claims_bad, prot = {}, [], 0, 0, 0, 0, 0, 0
    for cv_id in state["retry_ids"]:
        a, p = state["assessments"][cv_id], prof(state, cv_id)
        bad = 0
        for name, crit in a["llm_review"].items():
            crit["evidence"] = [check_evidence(name, e["quote"], p) for e in crit["evidence"]]
            bad += sum(not e["grounded"] for e in crit["evidence"])
            unsupported += sum(e["grounded"] and not e["supported"] for e in crit["evidence"])
            total += len(crit["evidence"])
            prot += len(checks.protected(crit["rationale"]))
            crit["rationale"] = checks.strip_protected(crit["rationale"])  # Q5: removed at once, never retried
        a["checks"] = checks.review_issues(a["llm_review"], p)  # Q4: claims that contradict the parsed CV
        claims, claims_bad = claims + len(a["llm_review"]), claims_bad + len(a["checks"])
        bad_total += bad
        if act and (bad or a["checks"]) and a["attempts"] <= s.max_verify_retries:
            retry.append(cv_id)
        elif act:
            for k, issues in a["checks"].items():  # out of retries: keep the text, label what is not supported
                a["llm_review"][k]["rationale"] += f" [unverified: {'; '.join(issues)}]"
            for crit in a["llm_review"].values():  # shared with criteria when llm_in_score
                if crit["evidence"] and not any(e["grounded"] for e in crit["evidence"]):
                    crit["score"] = min(crit["score"], 20)  # out of retries: claims without evidence lose credit
                    crit["rationale"] += " [capped: evidence not found in CV]"
                elif crit["evidence"] and not any(e["supported"] for e in crit["evidence"]) and crit["score"] > 50:
                    crit["score"] = 50  # rubric: "only skills-section listings" = 50
                    crit["rationale"] += " [capped: listed, not demonstrated]"
        updated[cv_id] = a
    grounded = total - bad_total
    emit("verify_evidence", f"{grounded}/{total} LLM quotes grounded ({100 * grounded / total if total else 100:.1f}%), "
                            f"{grounded - unsupported}/{total} also support their criterion (right section, right entity); "
                            f"{claims_bad}/{claims} rationales contradict the parsed CV; {prot} protected-attribute word(s) removed",
         evidence=total, grounded=grounded, supported=grounded - unsupported, claims=claims, claims_bad=claims_bad,
         protected=prot)
    if retry:
        emit("verify_evidence", f"Hallucinated quotes or contradicted claims in {retry} → sending back to assess", "loop")
    return {"assessments": updated, "retry_ids": retry, "loops": {"verify": 1 if retry else 0}}


# ── 5. Score: Score = 1[H(c)] × Σ w·S (deterministic) — and a status + reason for EVERY CV in the pool ────
def tier(total: float) -> str:
    s = get_settings()
    return "strong" if total >= s.tier_strong else "possible" if total >= s.tier_possible else "weak"


@traced("score")
async def score(state: AgentState) -> dict:
    cands = {c["cv_id"]: c for c in state["candidates"]}
    head = lambda cid: cands[cid]["headline"] if cid in cands else get_doc("cvs", cid).meta.get("headline", "")  # noqa: E731
    row = lambda cid, status, total=0.0, reason=None: {  # noqa: E731
        "cv_id": cid, "headline": head(cid), "status": status, "knocked_out": status == "knocked_out",
        "total": total, "tier": tier(total) if status == "ranked" else None, "reason": reason, "arbiter_note": None,
        "card": None, "flags": []}
    ranked = [row(cid, "ranked", weighted_total(a, state["weights"], False))
              for cid, a in state["assessments"].items() if cid in cands]
    rerank = {c["cv_id"]: c.get("rerank_score") for c in state["candidates"]}
    req, subs = state["requirements"], state.get("substitutes", {})
    for r in ranked:
        if via := fallback_uses(prof(state, r["cv_id"]), req["must_have"], subs, state.get("fallbacks", {})):
            r["reason"] = f"Ranked after direct matches: meets must-haves only via substitutes ({', '.join(via)})"
    fused = {c["cv_id"]: i for i, c in enumerate(state["candidates"])}  # rerank node left them in fused order
    by_total = get_settings().rank_by == "total"
    ranked.sort(key=lambda r: (r["reason"] is None,
                               tiebreak_key(r, state["assessments"].get(r["cv_id"]), rerank.get(r["cv_id"])) if by_total
                               else (-fused.get(r["cv_id"], len(fused)),)), reverse=True)
    if lo := state.get("llm_order"):  # P7 rank_shortlist: the controlled LLM order holds until the pool is re-screened
        ranked.sort(key=lambda r: (r["reason"] is not None, lo.index(r["cv_id"]) if r["cv_id"] in lo else len(lo)))
    # The tail is ordered too (HR browses it, τ measures it): what the LLM never assessed is ordered by the
    # zero-token rule score. Order of groups: reranked → not retrieved → needs review → knocked out.
    w = state["weights"]
    pre = lambda cid: rule_total(prof(state, cid), req, subs, w, state.get("fallbacks", {}))  # noqa: E731
    review = [row(cid, "needs_review", reason=r) for cid, r in state["knocked_out"].items() if r == NEEDS_REVIEW]
    knocked = sorted((row(cid, "knocked_out", reason=r) for cid, r in state["knocked_out"].items() if r != NEEDS_REVIEW),
                     key=lambda r: pre(r["cv_id"]), reverse=True)
    rest = [row(cid, "not_shortlisted", reason=r) for cid, r in state.get("screened", {}).items()
            if cid not in state["assessments"] and cid not in state["knocked_out"]]
    reranked = sorted((r for r in rest if rerank.get(r["cv_id"]) is not None), key=lambda r: fused[r["cv_id"]])
    unretrieved = sorted((r for r in rest if rerank.get(r["cv_id"]) is None), key=lambda r: pre(r["cv_id"]), reverse=True)
    screened = reranked + unretrieved
    rows = ranked + screened + review + knocked
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    fb = state.get("fallbacks", {})
    for r in ranked:  # explanation cards: they explain the rule score, so the decomposition always adds up
        r["card"] = {"decomposition": decomposition(state["assessments"][r["cv_id"]]["criteria"], w),
                     "what_if": counterfactuals(prof(state, r["cv_id"]), req, subs, w, fb)}
    for r in knocked:  # how close was it: the score with every missing must-have present
        r["card"] = {"what_if": counterfactuals(prof(state, r["cv_id"]), req, subs, w, fb, limit=1)}
    _versus_next(rows, state)
    dup = quality.duplicates([prof(state, r["cv_id"]) for r in rows])
    for r in rows:
        implied = sorted({m.skill for ms in prof(state, r["cv_id"]).skills.values() for m in ms if m.strength == "implied"})
        r["flags"] = quality.flags(prof(state, r["cv_id"])) + ([f"duplicate_of:{dup[r['cv_id']]}"] if r["cv_id"] in dup else []) \
            + ([f"implied_evidence:{','.join(implied)} (described, not named: verify in interview)"] if implied else [])
    if flagged := [r["cv_id"] for r in rows if r["flags"]]:
        emit("score", f"Data-quality flags on {flagged} (short, garbled, undated or duplicate CVs): check before trusting "
                      "their score", "warn")
    totals = [r["total"] for r in ranked]
    if ties := len(totals) - len(set(totals)):
        emit("score", f"{ties} exact tie(s) ordered by " + ("proven depth → years → relevance" if by_total else
                      "passage relevance fused with the rule score") + " (0 LLM tokens)", "warn")
    emit("score", "Weighted totals: " + ", ".join(f"{r['cv_id']}={r['total']:.2f} ({r['tier']})" for r in ranked[:5])
                  + f" · every CV has a status: {len(ranked)} ranked, {len(screened)} not shortlisted, "
                    f"{len(review)} for review, {len(knocked)} knocked out")
    return {"ranking": rows}


def _versus_next(rows: list[dict], state: AgentState) -> None:
    """Card L4: why each ranked candidate sits above the next ranked one (recomputed after arbiter swaps)."""
    ranked = [r for r in rows if r["status"] == "ranked"]
    crit = lambda cid: state["assessments"][cid]["criteria"]  # noqa: E731
    for a, b in zip(ranked, ranked[1:] + [None]):
        a["card"]["versus_next"] = contrastive(crit(a["cv_id"]), crit(b["cv_id"]), state["weights"], b["cv_id"]) if b else None


# ── L3: selective arbiter — only spend LLM on near-ties ───────────────────────────────────────────────
async def judge_pair(state: AgentState, a: str, b: str) -> list:
    """The arbiter's two verdicts on a pair. Each pair is judged in both orders (ported from the `coding` branch's
    arbiter): an LLM that picks whoever is shown first is position-biased, so only a verdict that survives the swap
    may change the rule order."""
    def evidence(cid: str) -> str:
        x = state["assessments"][cid]
        rules = [f"{k} {v['score']}: " + "; ".join(e["quote"] for e in v["evidence"]) for k, v in x["criteria"].items()]
        review = [f"{k} (review): {v['rationale']} " + "; ".join(e["quote"] for e in v["evidence"] if e["grounded"])
                  for k, v in x.get("llm_review", {}).items()]
        return "\n".join(rules + review)

    return [await decide(llm(state, "reasoning"), prompts.localized(prompts.ARBITER, state.get("lang")), [first, second],
                         f"JD: {json.dumps(state['requirements'])}\n\n{first}:\n{evidence(first)}"
                         f"\n\n{second}:\n{evidence(second)}") for first, second in ((a, b), (b, a))]


@traced("arbitrate")
async def arbitrate(state: AgentState) -> dict:
    s = get_settings()
    rows = [dict(r) for r in state["ranking"]]
    calls, judged = 0, set()
    if off(state, "arbiter"):
        return {}
    for i in range(min(5, len(rows)) - 1):
        a, b = rows[i], rows[i + 1]
        # |Δ|: with fused ordering a neighbour can have the higher total, and a negative gap is not a tie (a trace
        # showed the arbiter lifting a 0.77 candidate over a 0.92 one: beyond its mandate of near-ties only)
        if a["status"] != "ranked" or b["status"] != "ranked" or abs(a["total"] - b["total"]) >= s.near_tie_delta:
            continue
        if calls + 2 > s.max_arbiter_calls or a["cv_id"] in judged:
            continue  # each candidate is judged at most once → no cascades of LLM overrides
        judged |= {a["cv_id"], b["cv_id"]}
        verdicts = await judge_pair(state, a["cv_id"], b["cv_id"])
        calls += 2
        delta = a["total"] - b["total"]
        if verdicts[0].choice == verdicts[1].choice:
            winner = verdicts[0].choice
            note = f"Near-tie (Δ={delta:.3f}) resolved by arbiter in both orders: {winner} — {verdicts[0].reason}"
            if winner == b["cv_id"]:
                rows[i], rows[i + 1] = b, a
            rows[i]["arbiter_note"] = note
            rows[i + 1]["arbiter_note"] = f"Placed below {winner} by arbiter, consistent in both orders (Δ={delta:.3f})"
        else:
            note = (f"Near-tie {a['cv_id']} / {b['cv_id']} (Δ={delta:.3f}): the verdict flipped when the order was "
                    "swapped (position bias) → rule order kept")
            a["arbiter_note"] = b["arbiter_note"] = note
        emit("arbitrate", note, "loop", agree=verdicts[0].choice == verdicts[1].choice)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    _versus_next(rows, state)
    if not calls:
        emit("arbitrate", "No near-ties in the top 5 → arbiter skipped (0 LLM calls)")
    return {"ranking": rows, "loops": {"arbitrate": calls}}


# ── 6. Explainable report ─────────────────────────────────────────────────────────────────────────────
@traced("report")
async def report(state: AgentState) -> dict:
    top = [r for r in state["ranking"] if r["status"] == "ranked"][:5]
    if not top:  # don't pay an LLM to describe an empty pool
        emit("report", "No candidate passed the knock-out → deterministic report", "warn")
        return {"report": {"summary": "No candidate met the must-have requirements. Consider relaxing must-haves "
                                      "or accepting related skills.", "candidates": []}}
    top_ids = sorted(r["cv_id"] for r in top)
    prev = state.get("report") or {}
    if prev.get("top_ids") == top_ids:  # L5 re-weight only reordered the same people → explanations still hold
        emit("report", "Top-5 unchanged after re-scoring → report reused (0 LLM tokens)")
        return {}
    facts = []
    for r in top:
        a = state["assessments"][r["cv_id"]]
        facts.append({
            "cv_id": r["cv_id"], "rank": r["rank"], "total": r["total"], "arbiter_note": r["arbiter_note"],
            "criteria": {k: {"score": v["score"], "rationale": v["rationale"],
                             "verified_evidence": [e["quote"] for e in v["evidence"] if e["grounded"]]}
                         for k, v in a["criteria"].items()},
            "review": {k: {"rationale": v["rationale"], "verified_evidence": [e["quote"] for e in v["evidence"] if e["grounded"]]}
                       for k, v in a.get("llm_review", {}).items()},
            "what_if": (r.get("card") or {}).get("what_if"),
        })
    msgs = [SystemMessage(prompts.localized(prompts.REPORT, state.get("lang"))),
            HumanMessage(f"JD: {json.dumps(state['requirements'])}\nWeights: {state['weights']}\n\n"
                         f"Ranked candidates:\n{json.dumps(facts, indent=1)}")]
    profiles = {i: prof(state, i) for i in top_ids}
    for attempt in range(2):
        out = (await llm(state, "reasoning").with_structured_output(RankingReport).ainvoke(msgs)).model_dump()
        if bad := foreign_ids(out, top_ids):  # zero-token guard: the report may only discuss ranked people
            emit("report", f"Report mentions CVs not in the ranking {bad} → regenerating", "loop")
            msgs.append(HumanMessage(prompts.REPORT_RETRY.format(bad=bad, allowed=top_ids)))
            continue
        issues = checks.report_issues(out, profiles)  # Q6: every claim must be backed by that candidate's CV
        if not issues:
            emit("report", f"Report ready for top {len(top)} candidates; every claim checked against the CVs",
                 claims=len(out["candidates"]), claims_bad=0)
            return {"report": {**out, "top_ids": top_ids, "checks": {}}}
        if attempt == 0:
            emit("report", f"Report claims not supported by the CVs {issues} → regenerating", "loop",
                 claims=len(out["candidates"]), claims_bad=len(issues))
            msgs.append(HumanMessage(prompts.REPORT_CLAIMS_RETRY.format(
                bad="\n".join(f"- {k}: {i}" for k, v in issues.items() for i in v))))
            continue
        emit("report", f"Unsupported claims remain {issues} → dropped those sentences, kept the rest", "warn",
             claims=len(out["candidates"]), claims_bad=len(issues))
        return {"report": {**checks.sanitize_report(out, profiles), "top_ids": top_ids, "checks": issues}}
    emit("report", "Report still ungrounded → deterministic summary", "warn")
    return {"report": {"summary": "Ranked by weighted criteria; see per-candidate scores and verified evidence. "
                                  f"Top candidates: {', '.join(r['cv_id'] for r in top)}.",
                       "candidates": [], "top_ids": top_ids}}


# ── L5: HR review — approve, or re-weight and re-score (no new LLM assessment needed) ─────────────────
def _decide(state: AgentState, decisions: list[dict]) -> dict:
    """HR approves / rejects / overrides candidates, each with a reason (ported from the `coding` branch,
    hitl/audit.py). Every decision is appended to an audit log; the weights that best explain them (pairwise
    logistic regression on criterion scores, shrunk to the current profile) re-score the pool."""
    ok = [d for d in decisions if d.get("action") in ("approve", "reject", "override") and (d.get("reason") or "").strip()]
    run_id = (get_config().get("configurable") or {}).get("thread_id", "")
    path = get_settings().runtime_dir / "hr_audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for d in ok:
            fh.write(json.dumps({"run_id": run_id, "ts": time.time(), **d}) + "\n")
    scores = {cid: {k: c["score"] / 100 for k, c in a["criteria"].items()} for cid, a in state["assessments"].items()}
    pairs = preferences(ok, [r["cv_id"] for r in state["ranking"]])
    w = calibrate_weights(pairs, scores, state["weights"])
    moved = {k: round(w[k] - state["weights"][k], 3) for k in w if abs(w[k] - state["weights"][k]) >= 0.01}
    emit("hr_review", f"HR recorded {len(ok)} decision(s) → {len(pairs)} preference pair(s) → weights moved "
                      f"{moved or 'not at all'} (audit: .runtime/hr_audit.jsonl)", "loop")
    return {"hr_action": "adjust", "weights": w, "loops": {"review": 1}}



@traced("hr_review")
async def hr_review(state: AgentState) -> dict:
    left = get_settings().max_reviews - state.get("loops", {}).get("review", 0)
    answer = interrupt({"kind": "review", "payload": {"ranking": state["ranking"], "weights": state["weights"],
                                                       "adjustments_left": left}})
    if answer.get("action") == "decide" and left > 0:
        return _decide(state, answer.get("decisions", []))
    if answer.get("action") != "adjust" or left <= 0:
        emit("hr_review", "HR approved the ranking" if answer.get("action") != "adjust"
             else f"Re-weight budget used up ({get_settings().max_reviews}) → ranking is final", "info")
        return {"hr_action": "approve"}
    w = state["weights"]
    if "weights" in answer:
        w = {k: max(float(answer["weights"].get(k, 0)), 0.0) for k in CRITERIA}
        total = sum(w.values()) or 1.0
        w = {k: round(v / total, 4) for k, v in w.items()}
    if text := (answer.get("instruction") or "").strip():
        # HR speaks in words; the LLM may only pick criteria + directions from closed sets (canonicalized at the boundary)
        out: WeightInstruction = await llm(state, "fast").with_structured_output(WeightInstruction).ainvoke(
            [SystemMessage(prompts.REWEIGHT), HumanMessage(f"Criteria: {list(CRITERIA)}\nCurrent weights: {w}\n\n{text}")])
        changes = [c.model_dump() for c in out.changes]
        w = apply_instruction(w, changes)
        emit("hr_review", f"HR said “{text}” → {', '.join(f'{c['criterion']} {c['direction']}' for c in changes) or 'no change'}",
             "loop")
    emit("hr_review", f"HR re-weighted criteria → re-scoring with {w} (0 new assessments)", "loop")
    return {"hr_action": "adjust", "weights": w, "loops": {"review": 1}}
