"""The skills gazetteer (data/skills_taxonomy.yaml): canonical keys, surface aliases and related skills.
Dictionary-based entity recognition is the right tool here: tool names are a closed, curated vocabulary."""

import re
from functools import lru_cache

from app.rag.corpus import load_yaml


def skills() -> dict[str, dict]:
    return load_yaml("skills_taxonomy.yaml")["skills"]


def key(skill: str) -> str:
    """'GitHub Actions' / 'github-actions' → 'github_actions' (the taxonomy key)."""
    return re.sub(r"[\s\-/]+", "_", skill.lower().strip())


def _entry(skill: str) -> dict:
    return skills().get(key(skill), {})


def aliases(skill: str) -> list[str]:
    entry = _entry(skill)
    return list({skill.lower(), *(a.lower() for a in entry.get("aliases", []))})


def related(skill: str) -> list[str]:
    return [r.lower() for r in _entry(skill).get("related", [])]


def group(skill: str) -> str:
    return _entry(skill).get("group", "")


def automation_evidence(skill: str) -> bool:
    """Does using this skill show automation work? Defect trackers and test management (Jira, HP ALM) are tools every
    manual tester uses, so naming them must not raise a CV's automation depth or years."""
    return group(skill) != "test_management"


@lru_cache(maxsize=512)
def alias_regex(skill: str) -> re.Pattern:
    """Longest alias first, alphanumeric boundaries (so 'java' never matches inside 'javascript'). `not_after` words
    in the taxonomy veto a match ("change requests" is not the requests library)."""
    alts = sorted(aliases(skill), key=len, reverse=True)
    veto = "".join(f"(?<!{re.escape(w)} )" for w in _entry(skill).get("not_after", []))
    return re.compile(r"(?<![a-z0-9])" + veto + "(?:" + "|".join(map(re.escape, alts)) + r")(?![a-z0-9])")


def mentions(text: str, skill: str) -> bool:
    return bool(alias_regex(skill).search(text.lower()))
