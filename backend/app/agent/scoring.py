"""Deterministic, LLM-free parts of the judgement: skill matching, knock-out H(c), the weighted score,
and evidence grounding. Kept pure so they are unit-testable and cost zero tokens."""

import json
import re
from difflib import SequenceMatcher

from app.nlp.extract import CREDIT
from app.nlp.pipeline import CVProfile
from app.nlp.segment import ENTRY_SECTIONS
from app.nlp.taxonomy import aliases, group, mentions, related, skills

__all__ = ["aliases", "mentions", "related"]  # re-exported: the taxonomy moved to app.nlp.taxonomy


AND = re.compile(r"\s+(?:and|và)\s+|;\s*")
OR = re.compile(r"\s(?:or|hoặc)\s|/")
SOFT = re.compile(r"\b(?:or similar|or equivalent|e\.g\.|such as|a plus)\b|tương tự|tương đương")


def canonicalize(phrases: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Map free-text requirement phrases onto taxonomy keys → (hard, soft, substitutes). Each and-clause is read on
    its own, in English and Vietnamese, so one phrase can hold several "one of" groups.

    "playwright and typescript"                          → hard [playwright, typescript]
    "selenium or cypress"                                → hard [selenium], substitutes {selenium: [cypress]}
    "github actions or similar" / "jenkins … tương tự"   → soft [github_actions]  (never a knock-out)
    "(java, typescript or python) and ci (gitlab ci or jenkins)" → hard [java, gitlab_ci], two substitute groups
    """
    keys = list(skills())
    hard, soft, subs = [], [], {}
    for clause in (c for phrase in phrases for c in AND.split(phrase.lower())):
        found = [k for k in keys if mentions(clause, k)]
        found.sort(key=lambda k: min(clause.find(a) for a in aliases(k) if a in clause))
        if not found:
            continue
        if SOFT.search(clause):
            soft += found
        elif OR.search(clause):
            hard.append(found[0])
            subs[found[0]] = [*subs.get(found[0], []), *found[1:]]
        else:
            hard += found
    dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return dedup(hard), [s for s in dedup(soft) if s not in hard], {m: dedup(v) for m, v in subs.items()}


def _usable(p: CVProfile, skill: str):
    """Best mention of a skill unless it is only hedged ("familiar with", "currently learning")."""
    m = p.best(skill)
    return m if m and m.strength != "hedged" else None


def knockout_reason(p: CVProfile, must_have: list[str], substitutes: dict[str, list[str]]) -> str | None:
    """H(c): a must-have is missing (or only hedged) and none of its accepted substitutes is usable either."""
    reasons = []
    for m in must_have:
        if _usable(p, m) or any(_usable(p, s) for s in substitutes.get(m, [])):
            continue
        hedged = p.best(m)
        reasons.append(f"only hedged mention of {m}: “{hedged.line[:90]}”" if hedged else f"missing must-have: {m}")
    return ("; ".join(reasons)).capitalize() if reasons else None


def fallback_uses(p: CVProfile, must_have: list[str], substitutes: dict[str, list[str]],
                  fallbacks: dict[str, list[str]]) -> list[str]:
    """Must-haves this CV meets only through an L2 fallback ("selenium for robot_framework"). A JD's own "X or Y" is an
    equal alternative; a fallback HR accepted because the pool was scarce widens the pool but never outranks a direct
    match (score orders these after everyone who meets the must-haves directly)."""
    out = []
    for m in must_have:
        stated = [s for s in substitutes.get(m, []) if s not in fallbacks.get(m, [])]
        if _usable(p, m) or any(_usable(p, s) for s in stated):
            continue
        if used := next((s for s in fallbacks.get(m, []) if _usable(p, s)), None):
            out.append(f"{used} for {m}")
    return out


def _criterion(score: int, rationale: str, quotes: list[str], section: str | None = None) -> dict:
    return {"score": score, "rationale": rationale + " [rule-based]",
            "evidence": [{"quote": q, "grounded": True, "supported": True, "section": section}
                         for q in dict.fromkeys(quotes)][:3]}


def tools_score(p: CVProfile, must: list[str], nice: list[str], subs: dict[str, list[str]],
                fallbacks: dict[str, list[str]] | None = None) -> dict:
    """70% must-have coverage + 30% nice-to-have coverage. Credit by evidence strength: used in a role 1.0,
    only listed 0.7, hedged 0.2. An alternative the JD itself names ("Java, TypeScript or Python") counts in full;
    an L2 fallback HR accepted only because the pool was scarce gets 0.6 of that."""
    fallbacks = fallbacks or {}
    credit, quotes, notes = [], [], []
    for m in must:
        direct = p.best(m)
        options = [(CREDIT[direct.strength], direct, None)] if direct else []
        for s in subs.get(m, []):
            if b := _usable(p, s):
                options.append(((0.6 if s in fallbacks.get(m, []) else 1.0) * CREDIT[b.strength], b, b.skill))
        c, mention, via = max(options, key=lambda o: o[0], default=(0.0, None, None))
        credit.append(c)
        if mention:
            quotes.append(mention.line)
        if via:
            notes.append(f"{via} for {m}")
        elif not mention:
            notes.append(f"missing {m}")
        elif mention.strength != "used":
            notes.append(f"{m} {mention.strength} only")
    nice_hits = [n for n in nice if _usable(p, n)]
    parts = [(w, c) for w, c in ((0.7, sum(credit) / len(credit) if credit else None),
                                 (0.3, sum(CREDIT[_usable(p, n).strength] for n in nice_hits) / len(nice) if nice else None))
             if c is not None]
    score = round(100 * sum(w * c for w, c in parts) / sum(w for w, _ in parts)) if parts else 50
    quotes += [_usable(p, n).line for n in nice_hits[:2]]
    rationale = f"Must-have demonstrated {sum(c == 1.0 for c in credit)}/{len(must)}"
    rationale += f" ({'; '.join(notes)})" if notes else ""
    rationale += f"; nice-to-have {len(nice_hits)}/{len(nice)}" if nice else ""
    return _criterion(score, rationale, quotes)


def experience_rule(p: CVProfile, min_years: int | None) -> dict:
    """Years come from parsed role dates (overlaps merged, gaps excluded), not from an LLM's arithmetic."""
    if p.years_automation is None:
        return _criterion(0, "No dated roles and no years stated", [])
    score = experience_score(p.years_automation, min_years)
    return _criterion(score, f"{p.years_automation:g} years in automation roles ({p.years_total:g} total)",
                      p.years_source[:3], "experience")


def rule_total(p: CVProfile, req: dict, subs: dict[str, list[str]], weights: dict[str, float],
               fallbacks: dict[str, list[str]] | None = None) -> float:
    """Zero-token pre-score from the parsed CV: tools, ISTQB, years and the verb depth cue, weighted like the final
    score (renormalised over these four). Orders what the LLM never sees and helps choose who it sees (B3)."""
    crit = {"tools": tools_score(p, req["must_have"], req["nice_to_have"], subs, fallbacks)["score"],
            "istqb": istqb_score(p.text)["score"],
            "experience_level": experience_rule(p, req.get("min_years"))["score"],
            "automation_depth": depth_score(max(p.depth_cue, 1), req.get("depth_level"))}
    total_w = sum(weights.get(k, 0.0) for k in crit) or 1.0
    return round(sum(weights.get(k, 0.0) * v for k, v in crit.items()) / total_w / 100, 4)


def istqb_score(text: str) -> dict:
    """Advanced = 100, Foundation = 75, expired = 40, none = 0 — the certification level is real signal."""
    lines = [ln.strip().lstrip("-*• ") for ln in text.splitlines() if mentions(ln, "istqb")]
    if not lines:
        return _criterion(0, "No ISTQB certification mentioned", [])
    joined = " ".join(lines).lower()
    if "expired" in joined:
        score, label = 40, "expired"
    elif re.search(r"\badvanced\b|\bexpert\b", joined):
        score, label = 100, "Advanced/Expert"
    else:
        score, label = 75, "Foundation"
    return _criterion(score, f"ISTQB {label}", lines)


def foreign_ids(report: dict, allowed: list[str]) -> list[str]:
    """CV ids the report talks about that are not in the ranking it was given (copied from examples, invented, ...)."""
    text = json.dumps(report)
    return sorted({m for m in re.findall(r"\b(?:CV|UP)-\d{3}\b", text) if m not in allowed})


def tiebreak_key(row: dict, assessment: dict | None, rerank: float | None) -> tuple:
    """Rank order: total, then proven depth, then years, then cross-encoder relevance — deterministic, 0 tokens."""
    facts = (assessment or {}).get("facts", {})
    return (row["total"], facts.get("depth_level", 0), facts.get("years", 0.0), rerank or 0.0)


def depth_score(level: int, required: int | None) -> int:
    """L1–L4 → 25/50/75/100, minus 15 per level below what the JD requires."""
    return max(0, 25 * level - (15 * (required - level) if required and level < required else 0))


def experience_score(years: float, min_years: int | None) -> int:
    """Below the minimum scales up to 60; at the minimum 70; reaches 100 at twice the minimum (or 8y if none)."""
    target = min_years or 4
    if years < target:
        return round(60 * years / target)
    return min(100, round(70 + 30 * (years - target) / target))


def apply_instruction(weights: dict[str, float], changes: list[dict]) -> dict[str, float]:
    """more ×1.5 (a zero weight starts at 0.05), less ×0.6, ignore → 0; then renormalise to sum 1."""
    w = dict(weights)
    for c in changes:
        k = c["criterion"]
        if c["direction"] == "more":
            w[k] = max(w.get(k, 0.0), 0.05) * 1.5
        elif c["direction"] == "less":
            w[k] = w.get(k, 0.0) * 0.6
        else:
            w[k] = 0.0
    total = sum(w.values()) or 1.0
    return {k: round(v / total, 4) for k, v in w.items()}


def weighted_total(assessment: dict, weights: dict[str, float], knocked_out: bool) -> float:
    """Score(CV, JD) = 1[H(c)] × Σ_k w(c,k) · S(c,k), normalised to 0–1."""
    if knocked_out:
        return 0.0
    return round(sum(w * assessment["criteria"][k]["score"] / 100 for k, w in weights.items()), 4)


def _norm(s: str) -> str:
    s = re.sub(r"[*_`#>\"“”‘’]", "", s.lower())
    return re.sub(r"\s+", " ", s.replace("–", "-").replace("—", "-")).strip(" .-")


def is_grounded(quote: str, text: str, threshold: float = 0.85) -> bool:
    """A quote is grounded if (almost) all of it appears as one contiguous span of the CV."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    m = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
    return m.size / len(q) >= threshold


# Faithfulness has two levels. *Grounded*: the quote exists in the CV. *Supported*: it also comes from a section that
# can prove the criterion and names the right kind of thing. "Selenium, Java, Jenkins" copied from the skills list
# is grounded, but it does not support an automation-depth or CI claim — that is "listed, not demonstrated".
_TYPES = r"\b(?:api|rest|graphql|mobile|android|ios|web|ui|browser|performance|load|stress|e2e|end-to-end|contract|" \
         r"integration|regression|accessibility|security|smoke)\b"
_CI = r"pipeline|\bci\b|\bcd\b|ci/cd|grid|parallel|container|nightly|merge|gate|build|runner|deploy"
_LANG = r"\b(?:scala|kotlin|swift|golang|go|ruby|php|c\+\+|groovy|bash|shell|sql)\b"
SUPPORT: dict[str, tuple[tuple[str, ...] | None, tuple[str, ...], str]] = {
    # criterion: (sections allowed, taxonomy groups that count as the right entity, regex that also counts)
    "automation_depth": (ENTRY_SECTIONS, (), ""),
    "test_types": (ENTRY_SECTIONS, ("api", "mobile", "web_e2e", "performance"), _TYPES),
    "cicd_infra": (ENTRY_SECTIONS, ("cicd", "infra"), _CI),
    "programming_language": (None, ("language",), _LANG),
}
TAG = re.compile(r"^\s*\[[a-z]+\]\s*")


def locate(quote: str, p: CVProfile) -> tuple[bool, str | None]:
    """→ (grounded, section of the chunk that contains it; None if it spans chunks)."""
    q = TAG.sub("", quote)
    for c in p.chunks:
        if is_grounded(q, c.text):
            return True, c.section
    return is_grounded(q, p.text), None


def check_evidence(criterion: str, quote: str, p: CVProfile) -> dict:
    grounded, section = locate(quote, p)
    supported = grounded
    if grounded and criterion in SUPPORT:
        sections, groups, rx = SUPPORT[criterion]
        in_place = sections is None or section is None or section in sections
        low = quote.lower()
        names = not (groups or rx) or bool(rx and re.search(rx, low)) or any(
            group(k) in groups and mentions(low, k) for k in skills())
        supported = in_place and names
    return {"quote": TAG.sub("", quote), "grounded": grounded, "supported": supported, "section": section}
