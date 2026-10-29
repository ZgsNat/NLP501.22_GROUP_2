"""Deterministic, LLM-free parts of the judgement: skill matching, knock-out H(c), the weighted score,
and evidence grounding. Kept pure so they are unit-testable and cost zero tokens."""

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
