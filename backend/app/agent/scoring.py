"""Deterministic, LLM-free parts of the judgement: skill matching, knock-out H(c), the weighted score,
and evidence grounding. Kept pure so they are unit-testable and cost zero tokens."""

import json
import re
from difflib import SequenceMatcher

from app.nlp.extract import CREDIT
from app.nlp.jd import DOMAINS, FOREIGN, canonicalize
from app.nlp.pipeline import CVProfile
from app.nlp.segment import ENTRY_SECTIONS
from app.nlp.taxonomy import aliases, group, mentions, related, skills
from app.nlp.taxonomy import key as tkey

__all__ = ["aliases", "canonicalize", "mentions", "related"]  # re-exported from app.nlp (moved in v2/v3)


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


CORE4 = ("tools", "istqb", "experience_level", "automation_depth")  # the v2 rule pre-score (B3)
TEST_TYPE_GROUPS = {"web": "web_e2e", "ui": "web_e2e", "e2e": "web_e2e", "mobile": "mobile", "api": "api",
                    "contract": "api", "performance": "performance", "load": "performance"}


def _best_in(p: CVProfile, keys) -> tuple[float, str | None]:
    """Best evidence credit (used 1.0 / listed 0.7 / hedged 0.2) among `keys`, and that mention's line."""
    ms = [m for k in keys if (m := p.best(k))]
    m = max(ms, key=lambda m: CREDIT[m.strength], default=None)
    return (CREDIT[m.strength], m.line) if m else (0.0, None)


def _group_keys(groups: set[str]) -> list[str]:
    return [k for k in skills() if group(k) in groups]


def _units(req: dict, subs: dict[str, list[str]], groups: set[str]) -> list[list[str]]:
    """Requirements touching `groups`, one unit per requirement: a must-have with its JD alternatives is ONE unit
    ("GitLab CI, GitHub Actions or Jenkins"), each nice-to-have is its own unit."""
    units = [[tkey(m), *(tkey(s) for s in subs.get(m, []))] for m in req["must_have"]]
    units += [[tkey(n)] for n in (*req["nice_to_have"], *req.get("languages", []))]
    seen, out = set(), []
    for u in units:
        u = [k for k in dict.fromkeys(u) if group(k) in groups]
        if u and not seen.issuperset(u):
            seen.update(u)
            out.append(u)
    return out


def depth_rule(p: CVProfile, required: int | None) -> dict:
    level = max(p.depth_cue, 1)
    return _criterion(depth_score(level, required), f"L{level} from the candidate's own action verbs"
                      + (f" (JD asks L{required})" if required else ""),
                      [p.depth_cue_line] if p.depth_cue_line else [], "experience")


def language_rule(p: CVProfile, req: dict, subs: dict[str, list[str]]) -> dict:
    """Any one of the JD's programming languages is enough ("Java, TypeScript or Python")."""
    wanted = [k for u in _units(req, subs, {"language"}) for k in u]
    credit, line = _best_in(p, wanted or _group_keys({"language"}))
    return _criterion(round(100 * credit), (f"best of {wanted}" if wanted else "no language named by the JD; best "
                                            "language shown") + f": credit {credit:g}", [line] if line else [])


def test_types_rule(p: CVProfile, req: dict, subs: dict[str, list[str]]) -> dict:
    """Mean over the test types the JD needs (from its stated types and the groups of its tools)."""
    types = {TEST_TYPE_GROUPS[t] for t in (x.lower() for x in req.get("test_types", [])) if t in TEST_TYPE_GROUPS}
    types |= {group(u[0]) for u in _units(req, subs, set(TEST_TYPE_GROUPS.values()))}
    if not types:
        credit, line = _best_in(p, _group_keys(set(TEST_TYPE_GROUPS.values())))
        return _criterion(round(100 * credit), "no test type named by the JD; best shown", [line] if line else [])
    per = {t: _best_in(p, _group_keys({t})) for t in sorted(types)}
    return _criterion(round(100 * sum(c for c, _ in per.values()) / len(per)),
                      "; ".join(f"{t} {c:g}" for t, (c, _) in per.items()), [ln for _, ln in per.values() if ln])


def cicd_rule(p: CVProfile, req: dict, subs: dict[str, list[str]]) -> dict:
    """Each CI/infra tool the JD names; another CI server transfers 60% (Jenkins → GitLab CI is a short step)."""
    units = _units(req, subs, {"cicd", "infra"})
    ci_any, ci_line = _best_in(p, _group_keys({"cicd"}))
    if not units:
        infra, infra_line = _best_in(p, _group_keys({"infra"}))
        return _criterion(round(100 * (0.8 * ci_any + 0.2 * infra)), "no CI/infra named by the JD; best shown",
                          [x for x in (ci_line, infra_line) if x])
    per = {}
    for u in units:
        c, line = _best_in(p, u)
        if group(u[0]) == "cicd" and 0.6 * ci_any > c:
            c, line = 0.6 * ci_any, ci_line
        per[" or ".join(u)] = (c, line)
    return _criterion(round(100 * sum(c for c, _ in per.values()) / len(per)),
                      "; ".join(f"{k} {c:g}" for k, (c, _) in per.items()), [ln for _, ln in per.values() if ln])


def domain_language_rule(p: CVProfile, req: dict) -> dict:
    """Foreign languages and business domains the JD names (parsed from the JD text, see app.nlp.jd)."""
    low, parts, quotes = p.text.lower(), [], []
    for lang in req.get("foreign_languages", []):
        hit = re.search(FOREIGN[lang], low)
        parts.append((lang, 1.0 if hit else 0.0))
        quotes += [ln.strip() for ln in p.text.splitlines() if hit and re.search(FOREIGN[lang], ln.lower())][:1]
    for dom in req.get("domains", []):
        work = [c for c in p.chunks if c.section in ENTRY_SECTIONS and re.search(DOMAINS[dom], c.text.lower())]
        parts.append((dom, 1.0 if work else 0.5 if re.search(DOMAINS[dom], low) else 0.0))
        quotes += [ln.strip() for c in work[:1] for ln in c.text.splitlines() if re.search(DOMAINS[dom], ln.lower())][:1]
    if not parts:
        return _criterion(50, "JD names no foreign language or domain: neutral", [])
    return _criterion(round(100 * sum(c for _, c in parts) / len(parts)),
                      "; ".join(f"{k} {c:g}" for k, c in parts), quotes)


def rule_criteria(p: CVProfile, req: dict, subs: dict[str, list[str]],
                  fallbacks: dict[str, list[str]] | None = None) -> dict[str, dict]:
    """All 8 criteria from the parsed CV, zero tokens: Score = 1[H(c)] × Σ w·S is fully deterministic and every S has
    the CV line it came from. The LLM explains and arbitrates; it does not set these numbers."""
    return {"tools": tools_score(p, req["must_have"], req["nice_to_have"], subs, fallbacks),
            "automation_depth": depth_rule(p, req.get("depth_level")),
            "programming_language": language_rule(p, req, subs),
            "test_types": test_types_rule(p, req, subs),
            "cicd_infra": cicd_rule(p, req, subs),
            "experience_level": experience_rule(p, req.get("min_years")),
            "istqb": istqb_score(p.text),
            "domain_language": domain_language_rule(p, req)}


def rule_total(p: CVProfile, req: dict, subs: dict[str, list[str]], weights: dict[str, float],
               fallbacks: dict[str, list[str]] | None = None, criteria: tuple[str, ...] | None = None) -> float:
    """Zero-token score from the parsed CV, weighted like the final score and renormalised over `criteria` (all 8 by
    default; CORE4 is the v2 pre-score). Orders the pool and chooses who the LLM explains."""
    crit = {k: v["score"] for k, v in rule_criteria(p, req, subs, fallbacks).items() if not criteria or k in criteria}
    total_w = sum(weights.get(k, 0.0) for k in crit) or 1.0
    return round(sum(weights.get(k, 0.0) * v for k, v in crit.items()) / total_w / 100, 4)


# ── Explanation cards (ported from the `coding` branch, report/explain.py: L2 decomposition, L4 contrastive,
# L5 counterfactual). Deterministic and zero tokens: they explain the rule score, so they always add up.
def decomposition(criteria: dict[str, dict], weights: dict[str, float]) -> list[dict]:
    """Per criterion: weight × S = contribution. The contributions sum to the weighted total."""
    return [{"criterion": k, "weight": round(w, 4), "score": criteria[k]["score"],
             "contribution": round(w * criteria[k]["score"] / 100, 4)} for k, w in weights.items() if k in criteria]


def contrastive(mine: dict[str, dict], theirs: dict[str, dict], weights: dict[str, float], other_id: str) -> dict:
    """Why this candidate sits above the next one: the criteria with the largest weighted differences."""
    d = sorted(((k, w * (mine[k]["score"] - theirs[k]["score"]) / 100) for k, w in weights.items()
                if k in mine and k in theirs), key=lambda t: -t[1])
    return {"vs": other_id, "advantages": [{"criterion": k, "delta": round(x, 4)} for k, x in d if x > 1e-4][:3],
            "disadvantages": [{"criterion": k, "delta": round(x, 4)} for k, x in d[::-1] if x < -1e-4][:2]}


def counterfactuals(p: CVProfile, req: dict, subs: dict[str, list[str]], weights: dict[str, float],
                    fallbacks: dict[str, list[str]] | None = None, limit: int = 3) -> list[dict]:
    """"If the CV showed X in a role": for each required tool it lacks (must-haves first), the knock-out and score
    it would get. Tells HR what to probe in interview and how close a knocked-out candidate was."""
    from dataclasses import replace as dc_replace

    from app.nlp.extract import Mention

    lacks = lambda k: not (_usable(p, k) or any(_usable(p, s) for s in subs.get(k, [])))  # noqa: E731
    missing = [m for m in req["must_have"] if lacks(m)]
    base = 0.0 if missing else rule_total(p, req, subs, weights, fallbacks)
    # knocked out → one scenario with every missing must-have; otherwise one per missing tool, must-haves first
    scenarios = [missing] if missing else [[n] for n in req["nice_to_have"] if lacks(n)]
    out = []
    for keys in scenarios[:limit]:
        q = dc_replace(p, skills={**p.skills, **{tkey(k): [Mention(tkey(k), "", "experience", "used", "")] for k in keys}})
        ko = knockout_reason(q, req["must_have"], subs) is not None
        total = 0.0 if ko else rule_total(q, req, subs, weights, fallbacks)
        out.append({"if_used": keys, "must_have": bool(missing), "passes_knockout": not ko, "total": round(total, 4),
                    "delta": round(total - base, 4)})
    return out


# ── HR decisions → weights (ported from the `coding` branch, hitl/audit.py, FR-29/30) ─────────────────
def preferences(decisions: list[dict], ranking: list[str]) -> list[tuple[str, str]]:
    """(preferred, other) pairs: every approved CV over every rejected one; an override above the CVs it passes."""
    approved = [d["cv_id"] for d in decisions if d["action"] == "approve"]
    rejected = [d["cv_id"] for d in decisions if d["action"] == "reject"]
    pairs = [(a, r) for a in approved for r in rejected]
    order = list(ranking)
    for d in decisions:
        if d["action"] == "override" and d["cv_id"] in order and d.get("new_rank"):
            old, new = order.index(d["cv_id"]), min(d["new_rank"] - 1, len(order) - 1)
            pairs += [(d["cv_id"], o) for o in order[new:old]] + [(o, d["cv_id"]) for o in order[old + 1:new + 1]]
    return pairs


def calibrate_weights(pairs: list[tuple[str, str]], scores: dict[str, dict[str, float]], prior: dict[str, float],
                      l2: float = 1.0, lr: float = 0.5, epochs: int = 500) -> dict[str, float]:
    """Pairwise logistic regression on ΔS (criterion scores 0–1), shrunk towards the current weights: the weights
    that best explain HR's choices without drifting far from the company profile. Σw = 1, w ≥ 0."""
    import numpy as np

    keys = list(prior)
    x = np.array([[scores[a].get(k, 0) - scores[b].get(k, 0) for k in keys] for a, b in pairs
                  if a in scores and b in scores], dtype=float)
    w0 = np.array([prior[k] for k in keys], dtype=float)
    if not len(x):
        return dict(prior)
    w = w0.copy()
    for _ in range(epochs):
        pr = 1 / (1 + np.exp(-(x @ w)))
        w -= lr * (-(x.T @ (1 - pr)) / len(x) + l2 * (w - w0))
    w = np.clip(w, 0, None)
    return {k: round(float(v / (w.sum() or 1)), 4) for k, v in zip(keys, w)}


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
