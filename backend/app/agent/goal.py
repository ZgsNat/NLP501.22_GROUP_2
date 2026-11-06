"""Goal-driven control for P7 (agentic, but controlled): HR's goal as a typed spec the agent writes, and the code that
plans against it, scores it and checks the result. Zero LLM tokens in this module.

Why: an HR goal today is rarely "N CVs with tools X, Y". It mixes hard and soft constraints, concepts no taxonomy
holds ("AI in automation"), and can be impossible on the pool at hand. The agent must understand the goal, find out
whether it can be met, say so and confirm before acting, then check that what it hands over meets the goal, and
declare what it could not meet. The LLM writes the spec and the concept terms; code holds them to HR's words
(provenance), counts what the pool can deliver, scores bonuses and audits the shortlist.

    GoalSpec        count + constraints (must / nice / bonus / exclude) on taxonomy skills or HR-named concepts; each
                    constraint carries the HR words it comes from (Q19)
    concept         an open-vocabulary criterion: terms the LLM proposes, evidence that code finds and grades in the
                    CV (used in a role > listed > hedged), never a score the LLM sets
    feasibility     per-constraint and joint counts over the whole pool, the bottleneck, what relaxing it would give
    goal_report     the shortlist against the spec: unmet items the answer to HR must declare (Q20)
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.scoring import knockout_reason
from app.config import get_settings
from app.nlp.extract import CREDIT, FOLDED
from app.nlp.segment import ENTRY_SECTIONS
from app.nlp.taxonomy import key as tkey
from app.nlp.taxonomy import skills
from app.nlp.text import fold

Kind = Literal["must", "nice", "bonus", "exclude"]


class Constraint(BaseModel):
    kind: Kind = Field(description="must = knock-out; nice = counts in the tools criterion; bonus = extra points on "
                       "top of the weighted score; exclude = never shortlist")
    target: str = Field(description="a taxonomy skill ('selenium', 'playwright') or a concept named with define_concept "
                        "('ai_in_testing')")
    strength: Literal["normal", "strong"] = Field("normal", description="bonus only: 'strong' for a big plus")
    hr_quote: str = Field(min_length=2, description="HR's own words this constraint comes from, copied verbatim")


class GoalSpec(BaseModel):
    count: int | None = Field(None, ge=1, le=20, description="how many candidates HR wants, if they said")
    constraints: list[Constraint] = Field(default_factory=list, max_length=12)
    open_questions: list[str] = Field(default_factory=list, max_length=4,
                                      description="what you cannot decide from HR's words (e.g. what 'AI' means)")
    accept_shortfall: str | None = Field(None, description="HR's words accepting fewer candidates than asked, if "
                                         "they said so after seeing the feasibility")


# ── provenance ────────────────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", fold(s)).strip(" .,;:\"'“”")


def quoted(quote: str, hr_text: str) -> bool:
    """Q19: the quote is HR's words (case, accents and spacing ignored)."""
    q = _norm(quote)
    return bool(q) and re.search(rf"(?<![a-z0-9]){re.escape(q)}(?![a-z0-9])", _norm(hr_text)) is not None


def target_kind(target: str, concepts: dict) -> str:
    if tkey(target) in skills():
        return "skill"
    return "concept" if slug(target) in concepts else "unknown"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", fold(name)).strip("_")


# ── concepts: open-vocabulary criteria, graded by code ────────────────────────────────────────────────
def term_rx(terms: list[str]) -> re.Pattern:
    alts = sorted({re.escape(_norm(t)) for t in terms if _norm(t)}, key=len, reverse=True)
    return re.compile(rf"(?<![a-z0-9])(?:{'|'.join(alts)})(?![a-z0-9])")


def concept_evidence(p, terms: list[str]) -> dict | None:
    """The CV's best line naming a concept term: used (experience / projects) > listed (elsewhere) > hedged."""
    rx, best = term_rx(terms), None
    order = {"used": 3, "listed": 2, "hedged": 1}
    for c in p.chunks:
        for line in c.text.split("\n"):
            m = rx.search(_norm(line))
            if not m:
                continue
            strength = ("hedged" if FOLDED["hedge"].search(fold(line))  # "đang học" and "dang hoc" alike
                        else "used" if c.section in ENTRY_SECTIONS else "listed")
            if best is None or order[strength] > order[best["strength"]]:
                best = {"strength": strength, "term": m.group(0), "section": c.section, "quote": line.strip()[:200]}
    return best


def concept_score(ev: dict | None) -> float:
    return CREDIT[ev["strength"]] if ev else 0.0


def meets(p, c: Constraint, concepts: dict, subs: dict) -> bool:
    """Does the CV satisfy a must / exclude target? Skills by the knock-out rule, concepts by non-hedged evidence."""
    if target_kind(c.target, concepts) == "skill":
        return knockout_reason(p, [tkey(c.target)], subs) is None
    ev = concepts.get(slug(c.target), {}).get("evidence", {}).get(p.cv_id)
    return bool(ev) and ev["strength"] != "hedged"


# ── feasibility: what the pool can deliver, before any ranking ────────────────────────────────────────
def feasibility(spec: GoalSpec, profiles: dict, concepts: dict, subs: dict, jd_musts: list[str] = ()) -> dict:
    """jd_musts: the JD's own must-haves that HR's goal did not touch. They knock out too, so they count here, marked
    "(JD)": a goal that looks feasible on HR's constraints alone may not be once the JD's are applied."""
    musts = [c for c in spec.constraints if c.kind == "must"]
    named = {tkey(c.target) for c in spec.constraints}
    musts += [Constraint(kind="must", target=m, hr_quote="(JD)") for m in jd_musts if m not in named]
    excl = [c for c in spec.constraints if c.kind == "exclude"]
    unknown = sorted({c.target for c in spec.constraints if target_kind(c.target, concepts) == "unknown"})
    known = [c for c in musts if c.target not in unknown]
    ok = {cv: [c.target for c in known if meets(p, c, concepts, subs)] for cv, p in profiles.items()}
    out = {cv for cv, p in profiles.items() if any(meets(p, c, concepts, subs) for c in excl if c.target not in unknown)}
    alive = lambda drop=None: [cv for cv in profiles if cv not in out  # noqa: E731
                               and all(c.target in ok[cv] for c in known if c.target != drop)]
    joint = alive()
    label = lambda c: f"{c.target} (JD)" if c.hr_quote == "(JD)" else c.target  # noqa: E731
    per = {label(c): sum(c.target in ok[cv] for cv in profiles) for c in known}
    want = spec.count or 1
    relax = {label(c): len(alive(c.target)) for c in known} if len(joint) < want else {}
    bottleneck = min(per, key=per.get) if per else None
    status = "unknown_targets" if unknown else "ok" if len(joint) >= want else "infeasible"
    return {"status": status, "requested": spec.count, "pool": len(profiles), "per_must": per,
            "meet_all_musts": len(joint), "bottleneck": bottleneck, "if_relaxed": relax, "unknown_targets": unknown,
            "excluded": len(out)}


def blocking_questions(spec: GoalSpec, f: dict, concepts: dict, can_ask: bool = True) -> list[str]:
    """The open questions that must go to HR before ranking. They block only when the answer can change who meets
    HR's goal: the goal is not feasible as stated, or it rests on a concept whose terms the LLM proposed. A feasible
    goal on skills alone is met as HR wrote it; its open questions are assumptions the answer states, not reasons to
    stop and ask (E26 `feasible`: "is GitLab CI fine instead of Jenkins?" changed nothing HR asked for). Once HR can
    no longer be asked (clarification budget used), nothing blocks: what is left is stated as an assumption (E26
    round 5: holding the answer for a question the harness forbids asking ran two runs into the token budget)."""
    if not spec.open_questions or not can_ask:
        return []
    on_concept = any(target_kind(c.target, concepts) == "concept" for c in spec.constraints)
    return list(spec.open_questions) if f.get("status") != "ok" or on_concept else []


# ── scoring: bonuses on top of the weighted total, deterministic and shown on the card ────────────────
def bonus(p, spec: GoalSpec | None, concepts: dict) -> dict[str, float]:
    """{target: points} for nice/bonus constraints on concepts and bonus constraints on skills."""
    if not spec:
        return {}
    s, out = get_settings(), {}
    for c in spec.constraints:
        if c.kind not in ("bonus", "nice"):
            continue
        kind = target_kind(c.target, concepts)
        if kind == "skill" and c.kind == "bonus":
            m = p.best(c.target)
            credit = CREDIT[m.strength] if m else 0.0
        elif kind == "concept":
            credit = concept_score(concepts[slug(c.target)]["evidence"].get(p.cv_id))
        else:
            continue  # a nice-to-have skill already counts in the tools criterion
        size = s.goal_bonus_strong if c.strength == "strong" else s.goal_bonus
        if credit:
            out[c.target] = round(size * credit, 4)
    return out


def goal_filter(p, spec: GoalSpec | None, concepts: dict, subs: dict) -> str | None:
    """Why a CV may not be shortlisted under HR's goal (concept must-haves and excludes; skill musts are already
    knock-outs in the requirements), or None."""
    if not spec:
        return None
    for c in spec.constraints:
        kind = target_kind(c.target, concepts)
        if kind == "unknown":
            continue
        if c.kind == "exclude" and meets(p, c, concepts, subs):
            return f"HR excluded '{c.target}'"
        if c.kind == "must" and kind == "concept" and not meets(p, c, concepts, subs):
            return f"HR must-have '{c.target}': no verified mention in a role or skills list"
    return None


# ── the shortlist against the goal ────────────────────────────────────────────────────────────────────
def goal_report(spec: GoalSpec | None, shortlist: list[str], profiles: dict, concepts: dict, subs: dict,
                feasibility: dict | None = None, can_ask: bool = True) -> dict:
    """What HR asked vs what the shortlist delivers. `unmet` lists what the answer must declare; `assumed` the open
    questions that did not block (shown to HR as assumptions)."""
    if not spec:
        return {"unmet": [], "items": [], "assumed": []}
    items, unmet = [], []
    if spec.count and len(shortlist) < spec.count:
        unmet.append(f"count: {len(shortlist)} of the {spec.count} candidates HR asked for")
    for c in spec.constraints:
        if c.kind in ("must", "exclude"):
            bad = [cv for cv in shortlist if meets(profiles[cv], c, concepts, subs) != (c.kind == "must")]
            items.append({"constraint": f"{c.kind} {c.target}", "met_by": len(shortlist) - len(bad), "of": len(shortlist)})
            if bad:
                unmet.append(f"{c.kind} {c.target}: not met by {bad}")
        else:
            has = [cv for cv in shortlist if bonus(profiles[cv], GoalSpec(constraints=[c]), concepts)
                   or (c.kind == "nice" and profiles[cv].best(c.target))]
            items.append({"constraint": f"{c.kind} {c.target}", "met_by": len(has), "of": len(shortlist)})
            if shortlist and not has:
                unmet.append(f"{c.kind} {c.target}: no shortlisted candidate shows it")
    blocking = blocking_questions(spec, feasibility or {}, concepts, can_ask)
    if blocking:
        unmet.append(f"open questions not settled with HR: {blocking}")
    return {"unmet": unmet, "items": items, "assumed": [q for q in spec.open_questions if q not in blocking]}
