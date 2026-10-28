"""Skill taxonomy for the Automation Test domain (ESCO-lite).

Each canonical skill maps to aliases (surface forms found in CVs/JDs), an
ESCO-style concept id and "related" skills used by Query Expansion (FR-14).

NOTE: ids use a local ``esco-lite:`` namespace. Replace/extend with the real
ESCO v1.1 export (``load_esco_csv``) when the team has downloaded it; new
tools missing in ESCO (Playwright, k6, ...) stay here as manual additions.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Skill:
    name: str
    type: str                  # entity type (tool, language, test_type, cicd, certification, foreign_lang)
    aliases: Tuple[str, ...]
    related: Tuple[str, ...] = ()
    esco_uri: Optional[str] = None


def _s(name: str, type_: str, aliases: Iterable[str] = (), related: Iterable[str] = ()) -> Skill:
    return Skill(name, type_, tuple({name, *aliases}), tuple(related), f"esco-lite:{type_}/{name}")


_SKILLS: List[Skill] = [
    # --- automation tools / frameworks
    _s("selenium", "tool", ["selenium webdriver", "webdriver", "selenium grid"], ["playwright", "cypress", "webdriverio"]),
    _s("playwright", "tool", [], ["selenium", "cypress"]),
    _s("cypress", "tool", [], ["playwright", "selenium"]),
    _s("webdriverio", "tool", ["wdio"], ["selenium"]),
    _s("appium", "tool", [], ["espresso", "xcuitest"]),
    _s("espresso", "tool", [], ["appium"]),
    _s("xcuitest", "tool", [], ["appium"]),
    _s("robot framework", "tool", ["robotframework"], ["selenium"]),
    _s("katalon", "tool", ["katalon studio"], ["selenium"]),
    _s("testng", "tool", [], ["junit"]),
    _s("junit", "tool", ["junit5"], ["testng"]),
    _s("pytest", "tool", [], ["unittest"]),
    _s("cucumber", "tool", ["bdd", "gherkin", "specflow"], ["testng"]),
    _s("postman", "tool", ["newman"], ["rest assured"]),
    _s("rest assured", "tool", ["restassured", "rest-assured"], ["postman"]),
    _s("jmeter", "tool", ["apache jmeter"], ["k6", "gatling", "locust"]),
    _s("k6", "tool", [], ["jmeter", "gatling"]),
    _s("gatling", "tool", [], ["jmeter", "k6"]),
    _s("locust", "tool", [], ["jmeter"]),
    _s("jira", "tool", ["xray", "zephyr"], ["testrail"]),
    _s("testrail", "tool", [], ["jira"]),
    _s("uft", "tool", ["qtp", "micro focus uft"], ["selenium"]),
    _s("tosca", "tool", ["tricentis tosca"], ["selenium"]),
    # --- programming languages
    _s("java", "language", [], ["kotlin"]),
    _s("python", "language", [], []),
    _s("javascript", "language", ["js", "node.js", "nodejs"], ["typescript"]),
    _s("typescript", "language", ["ts"], ["javascript"]),
    _s("c#", "language", ["csharp", ".net", "dotnet"], []),
    _s("kotlin", "language", [], ["java"]),
    _s("sql", "language", ["mysql", "postgresql", "oracle sql"], []),
    # --- test types
    _s("ui testing", "test_type", ["ui test", "web testing", "web automation", "gui testing", "e2e", "end-to-end"], ["api testing"]),
    _s("api testing", "test_type", ["api test", "rest api testing", "api automation", "web service testing"], ["ui testing"]),
    _s("mobile testing", "test_type", ["mobile automation", "mobile test", "android testing", "ios testing"], ["ui testing"]),
    _s("performance testing", "test_type", ["load testing", "stress testing", "performance test"], []),
    _s("security testing", "test_type", ["penetration testing", "owasp"], []),
    _s("database testing", "test_type", ["db testing", "data testing", "etl testing"], []),
    _s("manual testing", "test_type", ["manual test", "kiểm thử thủ công", "test case design"], []),
    # --- CI/CD & devops
    _s("jenkins", "cicd", [], ["gitlab ci", "github actions"]),
    _s("gitlab ci", "cicd", ["gitlab-ci", "gitlab ci/cd"], ["jenkins", "github actions"]),
    _s("github actions", "cicd", [], ["jenkins", "gitlab ci"]),
    _s("azure devops", "cicd", ["azure pipelines", "tfs"], ["jenkins"]),
    _s("docker", "cicd", [], ["kubernetes"]),
    _s("kubernetes", "cicd", ["k8s"], ["docker"]),
    _s("git", "cicd", ["github", "gitlab", "bitbucket"], []),
    # --- certifications
    _s("istqb", "certification", ["istqb ctfl", "ctfl", "istqb foundation", "istqb advanced", "ctal"], []),
    _s("aws certified", "certification", ["aws certification"], []),
    # --- foreign languages
    _s("english", "foreign_lang", ["tiếng anh", "toeic", "ielts"], []),
    _s("japanese", "foreign_lang", ["tiếng nhật", "jlpt", "n1", "n2", "n3"], []),
]

SKILLS: Dict[str, Skill] = {s.name: s for s in _SKILLS}


def _alias_index(skills: Iterable[Skill]) -> List[Tuple[re.Pattern, Skill, str]]:
    index = []
    for skill in skills:
        for alias in skill.aliases:
            # word-ish boundaries that also work for "c#", ".net", "node.js"
            pattern = re.compile(r"(?<![\w#.])" + re.escape(alias) + r"(?![\w#])", re.IGNORECASE)
            index.append((pattern, skill, alias))
    # longer aliases first so "selenium webdriver" wins over "selenium"
    index.sort(key=lambda t: -len(t[2]))
    return index


ALIAS_INDEX = _alias_index(_SKILLS)


def normalize(term: str) -> Optional[Skill]:
    """Map a surface form to its canonical skill (FR-05)."""
    t = term.strip().lower()
    for skill in _SKILLS:
        if t in (a.lower() for a in skill.aliases):
            return skill
    return None


def expand(values: Iterable[str]) -> List[str]:
    """Return related skills for query expansion (FR-14), excluding the inputs."""
    seen = set(values)
    out: List[str] = []
    for v in values:
        skill = SKILLS.get(v)
        for r in (skill.related if skill else ()):
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def load_esco_csv(path: Path) -> int:
    """Attach real ESCO URIs from an ESCO v1.1 ``skills_en.csv`` export.

    Matches on preferredLabel / altLabels; returns number of skills updated.
    """
    updated = 0
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"conceptUri", "preferredLabel"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: not an ESCO skills export, missing columns {sorted(missing)}")
        for row in reader:
            labels = [row.get("preferredLabel", "")] + (row.get("altLabels", "") or "").split("\n")
            for label in labels:
                skill = normalize(label)
                current = SKILLS.get(skill.name) if skill else None
                if current and (current.esco_uri or "").startswith("esco-lite:"):  # first ESCO match wins
                    SKILLS[skill.name] = Skill(skill.name, skill.type, skill.aliases, skill.related, row["conceptUri"])
                    updated += 1
                    break
    return updated
