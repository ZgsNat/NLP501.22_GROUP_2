"""The P7 tool layer: what the orchestrator (agent/orchestrator.py) can do. Each tool is a typed wrapper over code that
already exists in nodes.py, scoring.py and investigate.py, so P6 and P7 share every rule, check and prompt.

A tool has:
  - Args: a pydantic model, both the schema the LLM sees and the validator the guard runs on each call;
  - check(state, args): None, or the reason the harness refuses the call (preconditions, budgets, HR authority);
  - run(state, args): (observation for the LLM, state update);
  - kind: "read" (no state change, so a repeat is answered from cache), "act" (changes state), "hr" (pauses for HR
    with interrupt(): runs alone in its own node, and does nothing expensive before the interrupt), "final" (the
    answer to HR, which ends the agent's turn).
Observations are compact JSON: the orchestrator sees ids, scores and flags, not whole CVs, so its context stays small.
Tools that change the pool (update_requirements, set_weights, investigate, ...) re-screen at once and return the new
screen, which saves the orchestrator a step."""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.agent import checks, nodes
from app.agent import investigate as inv
from app.agent.schemas import WeightChange
from app.agent.scoring import apply_instruction, contrastive, rule_criteria
from app.agent.state import AgentState, add_counts, merge
from app.config import get_settings
from app.nlp.extract import CREDIT, best
from app.nlp.taxonomy import key as tkey
from app.nlp.taxonomy import mentions
from app.nlp.taxonomy import skills
from app.rag import retrieve
from app.rag.corpus import get_doc, load_dir
from app.rag.store import get_store

MERGED = {"clarifications", "substitutes", "declined", "fallbacks", "recovered", "investigated", "assessments"}
CV_ID = r"\b(?:CV|UP)-\d{3}\b"


class Run:
    """Runs node functions one after another on a local copy of the state, the way LangGraph would apply their
    updates (merge reducers, summed loop counters), and collects one update with the same effect."""

    def __init__(self, state: AgentState):
        self.state, self.update = dict(state), {}

    def add(self, update: dict | None) -> None:
        for k, v in (update or {}).items():
            if k in MERGED:
                self.state[k], self.update[k] = merge(self.state.get(k), v), {**self.update.get(k, {}), **v}
            elif k == "loops":
                self.state[k], self.update[k] = add_counts(self.state.get(k), v), add_counts(self.update.get(k), v)
            else:
                self.state[k] = self.update[k] = v


# ── shared steps ──────────────────────────────────────────────────────────────────────────────────────
def _basis(state: AgentState) -> str:
    """What an LLM review was written against: a new requirement set makes it stale; new weights do not."""
    return json.dumps([state["requirements"], state.get("substitutes", {})], sort_keys=True)


def rule_assessment(state: AgentState, cv_id: str, prev: dict | None) -> dict:
    """All 8 criteria by rule (0 tokens), keeping an earlier LLM review if it was written for these requirements."""
    p, keep = nodes.prof(state, cv_id), None
    if prev and prev.get("llm_review") and prev.get("basis") == _basis(state):
        keep = prev
    crit = rule_criteria(p, state["requirements"], state.get("substitutes", {}), state.get("fallbacks", {}))
    if keep and get_settings().llm_in_score:
        crit = keep["criteria"]
    facts = {"depth_level": p.depth_cue, "depth_cue": p.depth_cue, "years": p.years_automation or 0.0,
             "years_total": p.years_total}
    return {"criteria": crit, "llm_review": keep["llm_review"] if keep else {}, "checks": keep.get("checks", {}) if keep else {},
            "facts": keep["facts"] if keep else facts, "knocked_out": False, "knockout_reason": None,
            "attempts": keep["attempts"] if keep else 0, "basis": _basis(state)}


def ranked(state: AgentState) -> list[dict]:
    return [r for r in state.get("ranking", []) if r["status"] == "ranked"]


def explained(state: AgentState, cv_id: str) -> bool:
    return bool(state.get("assessments", {}).get(cv_id, {}).get("llm_review"))


def screen_view(state: AgentState) -> dict:
    s, rows = get_settings(), state["ranking"]
    top = ranked(state)
    count = lambda status: sum(r["status"] == status for r in rows)  # noqa: E731
    return {
        "pool": len(rows), "ranked": len(top), "not_shortlisted": count("not_shortlisted"),
        "knocked_out": count("knocked_out"), "needs_review": count("needs_review"),
        "top": [{"cv_id": r["cv_id"], "rank": r["rank"], "total": r["total"], "tier": r["tier"],
                 "explained": explained(state, r["cv_id"]), **({"note": r["reason"]} if r["reason"] else {}),
                 **({"flags": r["flags"][:2]} if r["flags"] else {})} for r in top[:10]],
        "near_ties": [[a["cv_id"], b["cv_id"], round(a["total"] - b["total"], 3)] for a, b in zip(top[:5], top[1:5])
                      if abs(a["total"] - b["total"]) < s.near_tie_delta],
        "investigable": [{"cv_id": c, "missing": m, "why": w} for c, m, w in nodes.investigable(state)[:6]],
        "scarce_must_haves": nodes.substitute_proposals(state),
    }


async def screen_into(r: Run) -> dict:
    """search → rerank → rule assessment of the shortlist → score: the P6 pipeline up to the ranking, 0 LLM tokens."""
    r.add({"llm_order": []})  # a new screen invalidates an LLM order of the old shortlist
    r.add(await nodes.search(r.state))
    r.add(await nodes.rerank(r.state))
    short, old = r.state["retry_ids"], r.state.get("assessments", {})
    r.add({"assessments": {**{c: None for c in old if c not in short},
                           **{c: rule_assessment(r.state, c, old.get(c)) for c in short}}, "retry_ids": []})
    r.add(await nodes.score(r.state))
    return screen_view(r.state)


# Words HR uses for each criterion: a weight change must point at one of them in HR's own text.
CRITERION_WORDS = {
    "tools": r"tool|framework|stack|skill", "automation_depth": r"depth|architect|framework|ownership|design|senior",
    "programming_language": r"language|java|python|typescript|javascript|c#|coding|program",
    "test_types": r"api|mobile|web|performance|load|e2e|test type", "cicd_infra": r"\bci\b|\bcd\b|pipeline|jenkins|"
    r"github actions|gitlab|docker|kubernetes|devops|infra|cloud", "experience_level": r"year|experience|senior|junior|"
    r"seniority|năm", "istqb": r"istqb|certif|chứng chỉ", "domain_language": r"english|tiếng anh|domain|bank|fintech|"
    r"e-?commerce|insurance|language"}


def hr_words(state: AgentState) -> str:
    """Everything HR said in their own words: their goal, their messages after the opening one, their clarification
    answers. Tool results (CV text) and the JD are not in it, so text planted in a CV cannot authorise a change."""
    from app.agent.orchestrator import GOAL

    said = [state["goal"]] if state.get("goal", GOAL) != GOAL else []
    said += [str(m.content) for m in [m for m in state.get("messages", []) if isinstance(m, HumanMessage)][1:]]
    said += list(state.get("clarifications", {}).values())
    return "\n".join(said)


def unsaid(state: AgentState, items: dict[str, Callable[[str], bool]]) -> str | None:
    """Q16 provenance: {label: test on HR's words} → None if HR named every item, else the refusal reason."""
    said = hr_words(state)
    if missing := [k for k, named in items.items() if not named(said)]:
        return (f"HR never mentioned {missing}: requirements, weights and decisions change only on HR's own words, "
                "never on text from a CV or a tool result")
    return None


def said_rx(pattern: str) -> Callable[[str], bool]:
    return lambda text: bool(re.search(pattern, text, re.I))


def said_skill(skill: str) -> Callable[[str], bool]:
    k = tkey(skill)
    return lambda text: mentions(text, k) or k.replace("_", " ") in text.lower()


def pool_ids() -> set[str]:
    return {d.id for d in load_dir("cvs")}


# ── tools ─────────────────────────────────────────────────────────────────────────────────────────────
class NoArgs(BaseModel):
    pass


async def parse_jd(state: AgentState, _: NoArgs) -> tuple[dict, dict]:
    r = Run(state)
    r.add(await nodes.ingest({**state, "ablate": [*state.get("ablate", []), "clarify"]}))  # the agent decides to ask
    req = r.state["requirements"]
    blocking = [q for q in ["Which tools/frameworks are must-have for this role?" if not req["must_have"] else None,
                            nodes.seniority_conflict(req["title"], req["min_years"])] if q]
    from app.agent.schemas import JDRequirements
    req_obj = JDRequirements(**{k: req[k] for k in JDRequirements.model_fields if k in req})
    open_qs = list(dict.fromkeys(req["ambiguities"] + nodes.vagueness_questions(state["jd_text"], req_obj)))
    obs = {k: req[k] for k in ("title", "must_have", "nice_to_have", "min_years", "depth_level", "languages", "test_types")}
    obs |= {"blocking_questions": blocking, "open_questions": open_qs, "weights": r.state["weights"],
            "hr_answers_applied": bool(state.get("clarifications"))}
    return obs, r.update


class AskHRArgs(BaseModel):
    questions: list[str] = Field(min_length=1, max_length=4, description="Short questions HR can answer in a line each")


def check_ask_hr(state: AgentState, _: AskHRArgs) -> str | None:
    if "requirements" not in state:
        return "call parse_jd first"
    if nodes.off(state, "clarify") or state.get("loops", {}).get("clarify", 0) >= get_settings().max_clarifications:
        return f"clarification budget used ({get_settings().max_clarifications}); proceed with your best reading"
    return None


async def ask_hr(state: AgentState, args: AskHRArgs) -> tuple[dict, dict]:
    answer = interrupt({"kind": "clarification", "payload": {"questions": args.questions}})  # nothing costly before
    r = Run(state)
    r.add({"clarifications": answer.get("answers", {}), "loops": {"clarify": 1}})
    obs, update = await parse_jd(r.state, NoArgs())  # re-read the JD with the answers: saves a step
    r.add(update)
    return {"hr_answers": answer.get("answers", {}), "requirements_now": obs}, r.update


def check_screen(state: AgentState, _: NoArgs) -> str | None:
    return None if "requirements" in state else "call parse_jd first"


async def screen(state: AgentState, _: NoArgs) -> tuple[dict, dict]:
    r = Run(state)
    return await screen_into(r), r.update


def needs_ranking(state: AgentState) -> str | None:
    return None if state.get("ranking") else "call screen first"


def check_substitutes(state: AgentState, _: NoArgs) -> str | None:
    s = get_settings()
    if why := needs_ranking(state):
        return why
    if nodes.off(state, "expand") or state.get("loops", {}).get("expand", 0) >= s.max_expansions:
        return f"expansion budget used ({s.max_expansions})"
    return None if nodes.substitute_proposals(state) else "no must-have is scarce, or HR already decided on every related skill"


async def propose_substitutes(state: AgentState, _: NoArgs) -> tuple[dict, dict]:
    r = Run(state)
    r.add(await nodes.expand_query(state))  # interrupts for HR; the proposals before it are cheap and deterministic
    return {"accepted": r.state.get("fallbacks", {}), "declined": r.state.get("declined", {}),
            "screen": await screen_into(r)}, r.update


class CVsArgs(BaseModel):
    cv_ids: list[str] = Field(min_length=1, description="CV ids, e.g. ['CV-012']")


def check_investigate(state: AgentState, args: CVsArgs) -> str | None:
    if why := needs_ranking(state):
        return why
    ok = {c for c, _, _ in nodes.investigable(state)}
    if bad := [c for c in args.cv_ids if c not in ok]:
        return f"{bad} show no sign of a parsing failure (or were investigated already); only {sorted(ok)[:6]} qualify"
    n = get_settings().max_investigations
    return None if len(args.cv_ids) <= n else f"at most {n} CVs per call"


async def investigate(state: AgentState, args: CVsArgs) -> tuple[dict, dict]:
    r = Run(state)
    r.add(await nodes.investigate_some(state, only=args.cv_ids))
    outcome = {c: r.state.get("investigated", {}).get(c) for c in args.cv_ids}
    obs = {"outcomes": outcome}
    if any("recovered" in (o or "") for o in outcome.values()):
        obs["screen"] = await screen_into(r)  # recovered CVs pass the knock-out now: rank them
    return obs, r.update


def check_explain(state: AgentState, args: CVsArgs) -> str | None:
    if why := needs_ranking(state):
        return why
    if bad := [c for c in args.cv_ids if c not in state.get("assessments", {})]:
        return f"{bad} are not on the shortlist (only ranked CVs can be explained)"
    if done := [c for c in args.cv_ids if explained(state, c)]:
        return f"{done} are explained already; use get_candidate to read their review"
    n = get_settings().max_explain
    return None if len(args.cv_ids) <= n else f"at most {n} CVs per call"


async def explain(state: AgentState, args: CVsArgs) -> tuple[dict, dict]:
    """LLM review with evidence quotes, then verify_evidence (L4) until the quotes hold or the retries run out."""
    r = Run(state)
    r.add({"retry_ids": args.cv_ids})
    for _ in range(get_settings().max_verify_retries + 1):
        r.add(await nodes.assess(r.state))
        r.add(await nodes.verify_evidence(r.state))
        if not r.state["retry_ids"]:
            break
    r.add({"assessments": {c: {**r.state["assessments"][c], "basis": _basis(r.state)} for c in args.cv_ids
                           if c in r.state["assessments"]}, "retry_ids": []})
    r.add(await nodes.score(r.state))  # cards and tie-breaks read the review's depth fact
    obs = {}
    for c in args.cv_ids:
        a = r.state["assessments"].get(c, {})
        rev = a.get("llm_review", {})
        obs[c] = {"review": {k: v["rationale"][:220] for k, v in rev.items()},
                  "verified_quotes": sum(e["grounded"] and e["supported"] for v in rev.values() for e in v["evidence"]),
                  **({"unverified": a["checks"]} if a.get("checks") else {})}
    return obs, r.update


class PairArgs(BaseModel):
    cv_a: str
    cv_b: str


def check_compare(state: AgentState, args: PairArgs) -> str | None:
    if why := needs_ranking(state):
        return why
    if bad := [c for c in (args.cv_a, args.cv_b) if c not in state.get("assessments", {})]:
        return f"{bad} are not ranked; use get_candidate for their status and reason"
    return None if args.cv_a != args.cv_b else "compare two different CVs"


async def compare(state: AgentState, args: PairArgs) -> tuple[dict, dict]:
    """Why one sits above the other (0 tokens); on an adjacent near-tie, also the arbiter (L3, both orders)."""
    s, rows = get_settings(), [dict(r) for r in state["ranking"]]
    rank = {r["cv_id"]: i for i, r in enumerate(rows)}
    hi, lo = sorted((args.cv_a, args.cv_b), key=rank.get)
    crit = lambda c: state["assessments"][c]["criteria"]  # noqa: E731
    delta = rows[rank[hi]]["total"] - rows[rank[lo]]["total"]
    obs = {"higher": hi, "lower": lo, "delta_total": round(delta, 4),
           "why": contrastive(crit(hi), crit(lo), state["weights"], lo)}
    order = [r["cv_id"] for r in ranked(state)]
    adjacent = abs(order.index(hi) - order.index(lo)) == 1
    calls = state.get("loops", {}).get("arbitrate", 0)
    if not adjacent or abs(delta) >= s.near_tie_delta:
        obs["arbiter"] = "not needed: not an adjacent near-tie, the rule order stands"
        return obs, {}
    if nodes.off(state, "arbiter") or calls + 2 > s.max_arbiter_calls:
        obs["arbiter"] = f"arbiter budget used ({s.max_arbiter_calls} calls): the rule order stands"
        return obs, {}
    v = await nodes.judge_pair(state, hi, lo)
    if v[0].choice == v[1].choice:
        note = f"Near-tie (Δ={delta:.3f}) resolved by arbiter in both orders: {v[0].choice} — {v[0].reason}"
        if v[0].choice == lo:
            rows[rank[hi]], rows[rank[lo]] = rows[rank[lo]], rows[rank[hi]]
    else:
        note = f"Near-tie {hi} / {lo} (Δ={delta:.3f}): the verdict flipped when the order was swapped → rule order kept"
    for c in (hi, lo):
        next(r for r in rows if r["cv_id"] == c)["arbiter_note"] = note
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    nodes._versus_next(rows, state)
    nodes.emit("arbitrate", note, "loop", agree=v[0].choice == v[1].choice)
    obs["arbiter"] = note
    return obs, {"ranking": rows, "loops": {"arbitrate": 2}}


def check_rank(state: AgentState, _: NoArgs) -> str | None:
    s = get_settings()
    if why := needs_ranking(state):
        return why
    if state.get("loops", {}).get("llm_rank", 0) >= s.max_llm_ranks:
        return f"LLM ranking budget used ({s.max_llm_ranks} per run)"
    return None if len(ranked(state)) >= 2 else "fewer than two ranked candidates: nothing to order"


async def rank_shortlist(state: AgentState, _: NoArgs) -> tuple[dict, dict]:
    """The top Settings.llm_rank_size of the screened order (rule-assessed if not yet), reordered by the LLM under the
    controls of agent/llm_rank.py; rule scores and knock-outs stay; explanations stay valid."""
    from app.agent import llm_rank
    from app.nlp.injection import sanitize

    s, r = get_settings(), Run(state)
    top = [x["cv_id"] for x in state["ranking"] if x["status"] in ("ranked", "not_shortlisted")][: s.llm_rank_size]
    r.add({"assessments": {c: rule_assessment(state, c, None) for c in top if c not in state.get("assessments", {})}})
    res = await llm_rank.rank(nodes.llm(state, "reasoning"), sanitize(state["jd_text"], "job description")[0],
                              state["requirements"], top, {c: nodes.prof(state, c) for c in top})
    r.add({"llm_order": res["order"]})
    r.add(await nodes.score(r.state))
    nodes.emit("rank_shortlist", f"LLM ranked the top {len(top)} in both input orders (agreement@5 "
                                 f"{res['agreement@5']:.0%}) → {res['order'][:5]}", "loop", agreement=res["agreement@5"])
    return {"order": res["order"], "agreement@5": res["agreement@5"],
            "reasons": {c: res["reasons"].get(c, "")[:160] for c in res["order"][:10]},
            "screen": screen_view(r.state)}, {**r.update, "loops": {**r.update.get("loops", {}), "llm_rank": 1}}


class CandidateArgs(BaseModel):
    cv_id: str
    section: str = Field("", description="Optional: also return one CV section verbatim (experience, skills, ...)")


def check_candidate(state: AgentState, args: CandidateArgs) -> str | None:
    return None if get_doc("cvs", args.cv_id) else f"no CV {args.cv_id} in the pool"


async def get_candidate(state: AgentState, args: CandidateArgs) -> tuple[dict, dict]:
    p = nodes.prof(state, args.cv_id)
    row = next((r for r in state.get("ranking", []) if r["cv_id"] == args.cv_id), None)
    by = {s: sorted(k for k, ms in p.skills.items() if best(ms).strength == s) for s in CREDIT}
    obs = {"cv_id": args.cv_id, "headline": p.headline, "years_total": p.years_total,
           "years_automation": p.years_automation, "depth_cue": f"L{p.depth_cue}: {(p.depth_cue_line or '')[:120]}",
           "skills": by}
    if row:
        card = row.get("card") or {}
        obs |= {"status": row["status"], "rank": row["rank"], "total": row["total"], "reason": row["reason"],
                "flags": row["flags"], "arbiter_note": row["arbiter_note"],
                "top_contributions": sorted(card.get("decomposition", []), key=lambda d: -d["contribution"])[:4],
                "what_if": card.get("what_if")}
    if rev := state.get("assessments", {}).get(args.cv_id, {}).get("llm_review"):
        obs["review"] = {k: v["rationale"][:220] for k, v in rev.items()}
    if o := state.get("investigated", {}).get(args.cv_id):
        obs["investigated"] = o
    if args.section:
        obs["section_text"] = inv.read_section(p, args.section)
    return obs, {}


class SearchArgs(BaseModel):
    query: str = Field(description="Free text, e.g. 'banking domain' or 'performance testing with k6'")


async def search_cvs(state: AgentState, args: SearchArgs) -> tuple[dict, dict]:
    rows = {r["cv_id"]: r for r in state.get("ranking", [])}
    out, seen = [], set()
    for h in get_store().hybrid_search(retrieve.CHUNKS, args.query, limit=40):
        cid = h.payload.get("doc_id")
        if cid in seen or cid not in pool_ids():
            continue
        seen.add(cid)
        r = rows.get(cid, {})
        out.append({"cv_id": cid, "section": h.payload.get("section"), "passage": h.payload.get("text", "")[:160],
                    **({"status": r["status"], "rank": r["rank"]} if r else {})})
        if len(out) == 8:
            break
    return {"query": args.query, "hits": out}, {}


class RequirementArgs(BaseModel):
    add_must_have: list[str] = []
    remove_must_have: list[str] = []
    add_nice_to_have: list[str] = []
    min_years: int | None = None


def check_requirements(state: AgentState, args: RequirementArgs) -> str | None:
    if "requirements" not in state:
        return "call parse_jd first"
    if bad := [t for t in args.add_must_have + args.add_nice_to_have if tkey(t) not in skills()]:
        return f"{bad} are not skills in the taxonomy; use canonical names like 'selenium', 'java', 'jenkins'"
    if not (args.add_must_have or args.remove_must_have or args.add_nice_to_have or args.min_years is not None):
        return "nothing to change"
    items = {t: said_skill(t) for t in args.add_must_have + args.remove_must_have + args.add_nice_to_have}
    if args.min_years is not None:
        items["years of experience"] = said_rx(rf"\b{args.min_years}\b|year|năm")
    return unsaid(state, items)


async def update_requirements(state: AgentState, args: RequirementArgs) -> tuple[dict, dict]:
    req = dict(state["requirements"])
    drop = {tkey(t) for t in args.remove_must_have}
    req["must_have"] = list(dict.fromkeys([m for m in req["must_have"] if m not in drop] + [tkey(t) for t in args.add_must_have]))
    req["nice_to_have"] = list(dict.fromkeys([n for n in req["nice_to_have"] if n not in req["must_have"]]
                                             + [tkey(t) for t in args.add_nice_to_have if tkey(t) not in req["must_have"]]))
    if args.min_years is not None:
        req["min_years"] = args.min_years
    nodes.emit("update_requirements", f"HR changed the requirements → must-have {req['must_have']}, "
                                      f"min years {req['min_years']}", "loop")
    r = Run(state)
    r.add({"requirements": req, "query_terms": list(dict.fromkeys(req["must_have"] + req["nice_to_have"] + req["languages"]))})
    return {"must_have": req["must_have"], "nice_to_have": req["nice_to_have"], "min_years": req["min_years"],
            "screen": await screen_into(r)}, r.update


class WeightArgs(BaseModel):
    changes: list[WeightChange] = Field(min_length=1, description="certification = istqb, years/seniority = "
                                        "experience_level, CI/CD = cicd_infra, English = domain_language")


def check_weights(state: AgentState, args: WeightArgs) -> str | None:
    return needs_ranking(state) or unsaid(state, {c.criterion: said_rx(CRITERION_WORDS[c.criterion]) for c in args.changes})


async def set_weights(state: AgentState, args: WeightArgs) -> tuple[dict, dict]:
    w = apply_instruction(state["weights"], [c.model_dump() for c in args.changes])
    nodes.emit("set_weights", f"HR re-weighted: {', '.join(f'{c.criterion} {c.direction}' for c in args.changes)} → {w}", "loop")
    r = Run(state)
    r.add({"weights": w})
    return {"weights": w, "screen": await screen_into(r)}, r.update


class Decision(BaseModel):
    cv_id: str
    action: Literal["approve", "reject", "override"]
    reason: str = Field(min_length=3, description="HR's reason, in their words")
    new_rank: int | None = Field(None, description="override only: the rank HR wants")


class DecisionArgs(BaseModel):
    decisions: list[Decision] = Field(min_length=1)


def check_decisions(state: AgentState, args: DecisionArgs) -> str | None:
    if why := needs_ranking(state):
        return why
    if why := unsaid(state, {d.cv_id: said_rx(re.escape(d.cv_id)) for d in args.decisions}):
        return why
    rows = {r["cv_id"] for r in state["ranking"]}
    return None if not (bad := [d.cv_id for d in args.decisions if d.cv_id not in rows]) else f"{bad} are not in the pool"


async def record_decisions(state: AgentState, args: DecisionArgs) -> tuple[dict, dict]:
    """HR's decisions → audit log → weights that best explain them (pairwise logistic regression) → re-screen."""
    update = nodes._decide(state, [d.model_dump() for d in args.decisions])
    r = Run(state)
    r.add({"weights": update["weights"]})
    return {"weights": update["weights"], "screen": await screen_into(r)}, r.update


class RespondArgs(BaseModel):
    message: str = Field(max_length=2500, description="What you tell HR, at most about 300 words: findings, trade-offs, "
                         "what to probe next")
    shortlist: list[str] = Field([], max_length=10, description="CV ids you recommend, best first; empty if none")


def check_respond(state: AgentState, args: RespondArgs) -> str | None:
    """Q15: what reaches HR is checked like any LLM output: real ids, verified candidates, supported claims, no
    protected attributes."""
    top = {r["cv_id"] for r in ranked(state)}
    if bad := [c for c in args.shortlist if c not in top]:
        return f"{bad} are not ranked candidates; shortlist only from the ranking"
    if not nodes.off(state, "verify") and (todo := [c for c in args.shortlist if not explained(state, c)]):
        return f"explain {todo} before recommending them (their evidence is not verified yet)"
    if words := checks.protected(args.message):
        return f"the message mentions protected attributes {words}; remove them"
    if bad := sorted(set(re.findall(CV_ID, args.message)) - pool_ids()):
        return f"the message mentions CVs that do not exist: {bad}"
    issues = []
    for sentence in checks.SENTENCE.split(args.message):
        ids = set(re.findall(CV_ID, sentence))
        if len(ids) == 1 and (cid := ids.pop()) in pool_ids():
            issues += [f"{cid}: {i}" for i in checks.claim_issues(sentence, nodes.prof(state, cid))]
    return f"claims not supported by the CVs: {issues}; rewrite them" if issues else None


async def respond_to_hr(state: AgentState, args: RespondArgs) -> tuple[dict, dict]:
    cands = []
    for c in args.shortlist:
        rev = state["assessments"][c].get("llm_review", {})
        row = next(r for r in state["ranking"] if r["cv_id"] == c)
        gaps = [f"no {', '.join(w['if_used'])} in a role" for w in (row.get("card") or {}).get("what_if") or []]
        cands.append({"cv_id": c, "explanation": " ".join(v["rationale"] for v in rev.values())[:600],
                      "strengths": [k for k, v in rev.items() if v.get("score", 0) >= 75], "gaps": gaps})
    report = {"summary": args.message, "candidates": cands, "top_ids": sorted(args.shortlist), "checks": {}}
    return {"delivered": True}, {"reply": args.message, "shortlist": args.shortlist, "report": report}


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    kind: Literal["read", "act", "hr", "final"]
    run: Callable[[AgentState, Any], Awaitable[tuple[dict, dict]]]
    check: Callable[[AgentState, Any], str | None] = lambda state, args: None


TOOLS = {t.name: t for t in [
    Tool("parse_jd", "Read the JD (with any HR answers) into requirements, company weights and open questions. "
         "Blocking questions make a fair ranking impossible without HR.", NoArgs, "act", parse_jd),
    Tool("ask_hr", "Pause and ask HR. Only when the answer changes who is ranked (blocking or material questions). "
         "The JD is re-read with the answers.", AskHRArgs, "hr", ask_hr, check_ask_hr),
    Tool("screen", "Rank the whole pool against the requirements: hybrid retrieval + rule scores + knock-out, 0 LLM "
         "tokens. Returns the top 10, near-ties, knock-outs that look like parse failures, and scarce must-haves.",
         NoArgs, "act", screen, check_screen),
    Tool("investigate", "Send a tool-using sub-agent to re-read knocked-out CVs that screen marked investigable "
         "(garbled or misspelt text). Code verifies every fact it finds; recovered CVs are re-ranked.",
         CVsArgs, "act", investigate, check_investigate),
    Tool("propose_substitutes", "When screen reports scarce must-haves: ask HR to accept related skills, then "
         "re-screen.", NoArgs, "hr", propose_substitutes, check_substitutes),
    Tool("explain", "LLM review of ranked candidates with evidence quotes that code verifies against the CV (one "
         "reasoning call each). Required before recommending a candidate.", CVsArgs, "act", explain, check_explain),
    Tool("rank_shortlist", "Reorder the top of the screened pool by LLM judgement of each CV's evidence, checked: every "
         "candidate is read, two passes in opposite orders are averaged, rule scores and knock-outs stay. Use after "
         "screen, before explaining your shortlist.", NoArgs, "act", rank_shortlist, check_rank),
    Tool("compare", "Why one ranked candidate sits above another, criterion by criterion (0 tokens). On an adjacent "
         "near-tie it also asks the arbiter in both orders, which may swap them.", PairArgs, "act", compare, check_compare),
    Tool("get_candidate", "Facts about one CV: parsed skills, years, depth, status, score breakdown, review; "
         "optionally one section verbatim.", CandidateArgs, "read", get_candidate, check_candidate),
    Tool("search_cvs", "Free-text search over all CV passages (BM25 + dense), for questions screen does not answer.",
         SearchArgs, "read", search_cvs),
    Tool("update_requirements", "Change must-haves, nice-to-haves or minimum years when HR asks, then re-screen.",
         RequirementArgs, "act", update_requirements, check_requirements),
    Tool("set_weights", "Make criteria matter more, less or not at all when HR asks, then re-screen.",
         WeightArgs, "act", set_weights, check_weights),
    Tool("record_decisions", "Record HR's approve/reject/override decisions with reasons; weights are re-fitted "
         "to them and the pool re-screened.", DecisionArgs, "act", record_decisions, check_decisions),
    Tool("respond_to_hr", "Answer HR and end your turn: a message and, when you recommend candidates, a shortlist "
         "of explained, ranked CVs.", RespondArgs, "final", respond_to_hr, check_respond),
]}


def schemas() -> list[dict]:
    """OpenAI-format tool definitions (accepted by every LangChain chat model's bind_tools)."""
    out = []
    for t in TOOLS.values():
        spec = convert_to_openai_tool(t.args)
        spec["function"] |= {"name": t.name, "description": t.description}
        out.append(spec)
    return out
