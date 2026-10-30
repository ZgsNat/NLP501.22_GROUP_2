import asyncio

import pytest

from app.agent import checks
from app.nlp.pipeline import process

CV = """## Experience
### Senior QA — Bank (2019–2023)
- Built a Selenium + Java framework with Page Object Model.
- Ran the suite nightly on Jenkins.
## Skills
Selenium, Java, Jenkins, Postman
"""
P = process("CV-C", CV)


def test_claims_must_match_the_parsed_cv():
    assert checks.claim_issues("Built a Selenium framework in Java over 4 years.", P) == []
    issues = checks.claim_issues("Strong Cypress and Playwright work across 9 years.", P)
    assert {"names cypress but the CV never mentions it", "names playwright but the CV never mentions it"} <= set(issues)
    assert any(i.startswith("claims 9 years") for i in issues)
    # statements of absence and quotes of the JD are not claims about the candidate
    assert checks.claim_issues("No Cypress evidence; the JD asks for 5 years of Playwright.", P) == []
    # without dated roles there is no fact to contradict
    assert checks.overstated_years("12 years of testing.", process("CV-U", "Skills: Selenium")) is None


def test_protected_attributes_are_flagged_and_stripped_without_false_alarms():
    text = "Strong Selenium work. She is a 28-year-old married woman. Tests single sign-on, race conditions and page loads."
    assert checks.protected(text) == ["married", "woman", "year-old"]
    assert checks.strip_protected(text) == "Strong Selenium work. Tests single sign-on, race conditions and page loads."
    assert checks.protected("Giới tính: nữ giới, 30 tuổi") == ["giới tính", "nữ giới", "tuổi"]


def test_report_claims_are_checked_per_candidate_and_sanitized():
    rep = {"summary": "Pool is strong.", "candidates": [{
        "cv_id": "CV-C", "explanation": "Built a Selenium framework. Also an expert in Cypress.",
        "strengths": ["Java", "JMeter load testing"], "gaps": ["No Playwright"]}]}
    issues = checks.report_issues(rep, {"CV-C": P})
    assert set(issues) == {"CV-C"} and len(issues["CV-C"]) == 2  # cypress and jmeter; the gap is fine
    c = checks.sanitize_report(rep, {"CV-C": P})["candidates"][0]
    assert c["explanation"] == "Built a Selenium framework." and c["strengths"] == ["Java"] and c["gaps"] == ["No Playwright"]


def test_assess_retry_prompt_lists_what_failed(monkeypatch):
    """Regression (v3): the retry prompt read the rule criteria, whose evidence is always grounded, so after a
    hallucination the LLM was asked to try again without being told what was wrong."""
    from app.agent import nodes

    seen = []

    class Stop(Exception):
        pass

    class Fake:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, msgs):
            seen.append(msgs)
            raise Stop

    monkeypatch.setattr(nodes, "llm", lambda state, role: Fake())
    monkeypatch.setattr(nodes, "profile", lambda cv_id: P)
    prev = {"criteria": {}, "checks": {"test_types": ["names cypress but the CV never mentions it"]},
            "llm_review": {"cicd_infra": {"rationale": "", "evidence": [{"quote": "Deployed on Kubernetes", "grounded": False}]}}}
    with pytest.raises(Stop):
        asyncio.run(nodes._assess_call({"requirements": {}, "policy_context": []}, "CV-C", prev, asyncio.Semaphore(1)))
    retry = seen[0][-1].content
    assert "Deployed on Kubernetes" in retry and "names cypress" in retry
