"""Deterministic, LLM-free parts of the judgement: skill matching, knock-out H(c), the weighted score,
and evidence grounding. Kept pure so they are unit-testable and cost zero tokens."""

import json
import re
from difflib import SequenceMatcher

from app.rag.corpus import load_yaml


def _entry(skill: str) -> dict:
    return load_yaml("skills_taxonomy.yaml")["skills"].get(re.sub(r"[\s\-/]+", "_", skill.lower().strip()), {})


def aliases(skill: str) -> list[str]:
    entry = _entry(skill)
    return list({skill.lower(), *(a.lower() for a in entry.get("aliases", []))})


def related(skill: str) -> list[str]:
    return [r.lower() for r in _entry(skill).get("related", [])]


def mentions(text: str, skill: str) -> bool:
    low = text.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", low) for a in aliases(skill))


def canonicalize(phrases: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Map free-text requirement phrases onto taxonomy keys → (hard, soft, substitutes).

    "playwright and typescript"       → hard [playwright, typescript]
    "selenium or cypress"             → hard [selenium], substitutes {selenium: [cypress]}
    "github actions or similar"       → soft [github_actions]  (never a knock-out)
    """
    keys = list(load_yaml("skills_taxonomy.yaml")["skills"])
    hard, soft, subs = [], [], {}
    for phrase in phrases:
        low = phrase.lower()
        found = [k for k in keys if mentions(low, k)]
        found.sort(key=lambda k: min(low.find(a) for a in aliases(k) if a in low))
        if not found:
            continue
        if re.search(r"\b(or similar|or equivalent|e\.g\.|such as|a plus)\b", low):
            soft += found
        elif " or " in low or "/" in low:
            hard.append(found[0])
            subs[found[0]] = found[1:]
        else:
            hard += found
    dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return dedup(hard), [s for s in dedup(soft) if s not in hard], subs


def knockout_reason(text: str, must_have: list[str], substitutes: dict[str, list[str]]) -> str | None:
    """H(c): a must-have is missing and none of its accepted substitutes is present either."""
    missing = [
        m for m in must_have if not mentions(text, m) and not any(mentions(text, s) for s in substitutes.get(m, []))
    ]
    return f"Missing must-have: {', '.join(missing)}" if missing else None


def line_with(text: str, skill: str) -> str | None:
    """The CV line that mentions a skill — evidence that is grounded by construction."""
    return next((ln.strip().lstrip("-*• ").strip() for ln in text.splitlines() if mentions(ln, skill)), None)


def _criterion(score: int, rationale: str, quotes: list[str]) -> dict:
    return {"score": score, "rationale": rationale + " [rule-based]",
            "evidence": [{"quote": q, "grounded": True} for q in dict.fromkeys(quotes)][:3]}


def tools_score(text: str, must: list[str], nice: list[str], subs: dict[str, list[str]]) -> dict:
    """70% must-have coverage (substitute = 0.6 credit) + 30% nice-to-have coverage. No LLM needed to count tools."""
    credit, quotes, missing, via = [], [], [], []
    for m in must:
        if ln := line_with(text, m):
            credit.append(1.0)
            quotes.append(ln)
        elif sub := next((s for s in subs.get(m, []) if mentions(text, s)), None):
            credit.append(0.6)
            via.append(f"{sub} for {m}")
            quotes.append(line_with(text, sub))
        else:
            credit.append(0.0)
            missing.append(m)
    nice_hits = [n for n in nice if mentions(text, n)]
    parts = [(w, c) for w, c in ((0.7, sum(credit) / len(credit) if credit else None),
                                 (0.3, len(nice_hits) / len(nice) if nice else None)) if c is not None]
    score = round(100 * sum(w * c for w, c in parts) / sum(w for w, _ in parts)) if parts else 50
    quotes += [ln for n in nice_hits[:2] if (ln := line_with(text, n))]
    rationale = f"Must-have {credit.count(1.0)}/{len(must)} direct"
    rationale += f" (substitutes: {', '.join(via)})" if via else ""
    rationale += f", missing {missing}" if missing else ""
    rationale += f"; nice-to-have {len(nice_hits)}/{len(nice)}" if nice else ""
    return _criterion(score, rationale, quotes)


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
    return sorted({m for m in re.findall(r"CV-\d{3}", text) if m not in allowed})


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
