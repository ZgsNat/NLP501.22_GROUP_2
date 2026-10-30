import pytest

from app.agent.schemas import CRITERIA
from app.agent.scoring import (
    rule_criteria,
    rule_total,
    apply_instruction,
    canonicalize,
    check_evidence,
    depth_score,
    experience_score,
    foreign_ids,
    is_grounded,
    istqb_score,
    knockout_reason,
    mentions,
    tiebreak_key,
    tools_score,
    weighted_total,
)
from app.nlp.pipeline import process


def cv(body: str):
    return process("CV-T", body)

CV = "Built a Playwright + TypeScript framework with Page Object Model; ran it in GitHub Actions on every PR."


def test_mentions_uses_word_boundaries_and_aliases():
    assert mentions(CV, "playwright")
    assert mentions(CV, "github actions")
    assert not mentions("I like javascript", "java")


@pytest.mark.parametrize(
    "phrases, hard, soft, subs",
    [
        (["Hands-on experience with Playwright and TypeScript"], ["playwright", "typescript"], [], {}),
        (["Selenium or Cypress"], ["selenium"], [], {"selenium": ["cypress"]}),
        (["CI pipelines (GitHub Actions or similar)"], [], ["github_actions"], {}),
        (["3+ years in test automation"], [], [], {}),
        (["Strength in Java, TypeScript or Python and in CI/CD (GitLab CI, GitHub Actions or Jenkins)"],
         ["java", "gitlab_ci"], [], {"java": ["typescript", "python"], "gitlab_ci": ["github_actions", "jenkins"]}),
        (["Kinh nghiệm RestAssured hoặc Postman và sử dụng Jenkins hoặc công cụ CI tương tự"],
         ["restassured"], ["jenkins"], {"restassured": ["postman"]}),
    ],
)
def test_canonicalize(phrases, hard, soft, subs):
    assert canonicalize(phrases) == (hard, soft, subs)


def test_knockout_respects_substitutes_and_hedges():
    cypress_cv = cv("## Experience\n- Wrote Cypress tests in TypeScript.")
    assert knockout_reason(cypress_cv, ["playwright"], {}) == "Missing must-have: playwright"
    assert knockout_reason(cypress_cv, ["playwright"], {"playwright": ["cypress"]}) is None
    hedged = cv("## Skills\nFamiliar with Appium (not used in production).")
    assert knockout_reason(hedged, ["appium"], {}).startswith("Only hedged mention of appium")


def test_weighted_total_and_knockout():
    a = {"criteria": {"tools": {"score": 100}, "istqb": {"score": 50}}}
    assert weighted_total(a, {"tools": 0.8, "istqb": 0.2}, False) == 0.9
    assert weighted_total(a, {"tools": 0.8, "istqb": 0.2}, True) == 0.0


def test_grounding_accepts_verbatim_and_formatting_noise_rejects_fabrication():
    assert is_grounded("built a playwright + typescript framework", CV)
    assert is_grounded("**Built a Playwright + TypeScript framework**", CV)
    assert not is_grounded("Led a team of 12 SDETs at a bank", CV)



def test_tools_score_credits_evidence_strength_and_substitutes():
    p = cv("## Experience\n- Built Cypress suites in TypeScript\n- Ran them in GitHub Actions\n## Skills\nJava")
    full = tools_score(p, ["cypress", "typescript"], ["github actions"], {})
    assert full["score"] == 100 and all(e["grounded"] for e in full["evidence"])
    stated = tools_score(p, ["playwright"], [], {"playwright": ["cypress"]})  # the JD said "Playwright or Cypress"
    assert stated["score"] == 100
    sub = tools_score(p, ["playwright"], [], {"playwright": ["cypress"]}, fallbacks={"playwright": ["cypress"]})
    assert sub["score"] == 60 and "cypress for playwright" in sub["rationale"]  # L2 fallback: partial credit
    listed = tools_score(p, ["java"], [], {})
    assert listed["score"] == 70 and "java listed only" in listed["rationale"]  # a skills-list claim is not work


def test_evidence_must_be_grounded_and_supported():
    p = cv("## Experience\n- Designed a Selenium framework with Page Object Model\n## Skills\nSelenium, Java, Jenkins")
    assert check_evidence("automation_depth", "Designed a Selenium framework", p)["supported"]
    listed = check_evidence("automation_depth", "Selenium, Java, Jenkins", p)
    assert listed["grounded"] and not listed["supported"] and listed["section"] == "skills"
    assert check_evidence("programming_language", "Selenium, Java, Jenkins", p)["supported"]  # any section, names a language
    assert not check_evidence("cicd_infra", "Led a team of 12 SDETs", p)["grounded"]


def test_istqb_score():
    assert istqb_score("ISTQB Foundation Level, 2021")["score"] == 75
    assert istqb_score("ISTQB Foundation; ISTQB Advanced Test Automation Engineer")["score"] == 100
    assert istqb_score("ISTQB Foundation (expired)")["score"] == 40
    assert istqb_score("No certificates")["score"] == 0


def test_depth_and_experience_scores_are_monotonic_and_penalise_gaps():
    assert [depth_score(lv, None) for lv in (1, 2, 3, 4)] == [25, 50, 75, 100]
    assert depth_score(2, 3) == 35  # one level below a required L3
    assert experience_score(2, 4) < experience_score(4, 4) == 70 < experience_score(6, 4) < experience_score(8, 4) == 100
    assert experience_score(12, 4) == 100


def test_tiebreak_prefers_depth_then_years_then_relevance():
    row = {"total": 0.96}
    l4 = tiebreak_key(row, {"facts": {"depth_level": 4, "years": 5}}, 0.1)
    l3_senior = tiebreak_key(row, {"facts": {"depth_level": 3, "years": 12}}, 0.9)
    l3_junior = tiebreak_key(row, {"facts": {"depth_level": 3, "years": 6}}, 0.9)
    assert l4 > l3_senior > l3_junior
    assert tiebreak_key({"total": 0.97}, None, None) > l4  # the score always wins


def test_foreign_ids_catches_ids_outside_the_ranking():
    report = {"summary": "CV-002 and CV-003 are the strongest", "candidates": [{"cv_id": "CV-018"}]}
    assert foreign_ids(report, ["CV-018", "CV-002"]) == ["CV-003"]
    assert foreign_ids(report, ["CV-002", "CV-003", "CV-018"]) == []


def test_apply_instruction_renormalises():
    w = {"tools": 0.5, "istqb": 0.0, "domain_language": 0.5}
    out = apply_instruction(w, [{"criterion": "istqb", "direction": "more"}, {"criterion": "tools", "direction": "less"}])
    assert sum(out.values()) == pytest.approx(1.0, abs=1e-3)
    assert out["istqb"] > 0 and out["tools"] < out["domain_language"]
    assert apply_instruction(w, [{"criterion": "tools", "direction": "ignore"}])["tools"] == 0.0


def test_rule_total_prefers_demonstrated_work_over_claims():

    req = {"must_have": ["selenium"], "nice_to_have": [], "min_years": 3, "depth_level": 3}
    w = {"tools": 0.4, "automation_depth": 0.3, "experience_level": 0.2, "istqb": 0.1, "domain_language": 0.5}
    doer = cv("## Experience\n### QA — Bank (2018–2024)\n- Designed a Selenium framework with Page Object Model")
    claimer = cv("## Experience\n### QA — Bank (2018–2024)\n- Executed manual regression\n## Skills\nSelenium")
    assert 0 <= rule_total(claimer, req, {}, w) < rule_total(doer, req, {}, w) <= 1


def test_fallback_uses_separates_jd_alternatives_from_l2_fallbacks():
    from app.agent.scoring import fallback_uses

    p = cv("## Experience\n- Built Selenium suites in Python\n- Ran them in GitLab CI")
    subs = {"robot_framework": ["selenium"], "jenkins": ["gitlab_ci"]}  # JD said "Jenkins or GitLab CI"
    assert fallback_uses(p, ["robot_framework", "jenkins"], subs, {"robot_framework": ["selenium"]}) == \
        ["selenium for robot_framework"]
    assert fallback_uses(p, ["jenkins"], subs, {}) == []


def test_rule_criteria_score_all_eight_from_the_profile():
    p = cv("## Experience\n### SDET — Bank (2019–2024)\n- Designed a Playwright + TypeScript framework for payment APIs\n"
           "- Ran suites in GitLab CI on Docker runners\n## Skills\nJava, Postman\nEnglish (IELTS 7.0)")
    req = {"must_have": ["playwright", "java"], "nice_to_have": ["docker"], "languages": [], "test_types": ["api"],
           "min_years": 3, "depth_level": 3, "foreign_languages": ["english"], "domains": ["banking"]}
    subs = {"java": ["typescript", "python"], "playwright": []}
    r = rule_criteria(p, req, subs)
    assert set(r) == set(CRITERIA) and all(v["rationale"].endswith("[rule-based]") for v in r.values())
    assert r["programming_language"]["score"] == 100  # TypeScript used: one of the JD's languages is enough
    assert r["automation_depth"]["score"] == 75  # L3 from "Designed … framework"
    assert r["domain_language"]["score"] == 100  # English named, banking in an experience entry
    assert 0 < rule_total(p, req, subs, {k: 1 / 8 for k in CRITERIA}) <= 1


def test_cicd_counts_a_one_of_group_once_and_transfers_between_ci_servers():
    p = cv("## Experience\n### QA — Co (2020–2023)\n- Ran the regression suite in Jenkins on every merge")
    req = {"must_have": ["gitlab_ci"], "nice_to_have": [], "languages": [], "test_types": []}
    one_of = rule_criteria(p, req, {"gitlab_ci": ["github_actions", "jenkins"]})["cicd_infra"]
    assert one_of["score"] == 100  # Jenkins satisfies "GitLab CI, GitHub Actions or Jenkins" in full
    other_ci = rule_criteria(p, req, {})["cicd_infra"]
    assert other_ci["score"] == 60  # only GitLab CI asked: another CI server transfers 60%


def test_test_types_average_over_what_the_jd_needs():
    p = cv("## Experience\n### QA — Co (2020–2023)\n- Wrote API tests with RestAssured")
    req = {"must_have": [], "nice_to_have": [], "languages": [], "test_types": ["api", "mobile"]}
    assert rule_criteria(p, req, {})["test_types"]["score"] == 50  # api proven, mobile missing


def test_card_decomposition_adds_up_and_what_if_explains_a_knock_out():
    from app.agent.scoring import counterfactuals, decomposition
    p = cv("## Experience\n### QA — Co (2019–2024)\n- Designed a Cypress framework in TypeScript\n- Ran it in GitHub Actions")
    req = {"must_have": ["cypress", "java"], "nice_to_have": ["docker"], "languages": [], "test_types": []}
    w = {k: 1 / 8 for k in CRITERIA}
    crit = rule_criteria(p, req, {})
    assert abs(sum(d["contribution"] for d in decomposition(crit, w)) - rule_total(p, req, {}, w)) < 1e-3
    (what_if,) = counterfactuals(p, req, {}, w)  # knocked out for java → one scenario with every missing must-have
    assert what_if["if_used"] == ["java"] and what_if["passes_knockout"] and what_if["total"] > 0


def test_hr_preferences_calibrate_weights_towards_what_hr_chose():
    from app.agent.scoring import calibrate_weights, preferences
    pairs = preferences([{"cv_id": "A", "action": "approve"}, {"cv_id": "B", "action": "reject"}], ["B", "A"])
    assert pairs == [("A", "B")]
    scores = {"A": {"istqb": 1.0, "tools": 0.5}, "B": {"istqb": 0.0, "tools": 0.9}}
    w = calibrate_weights(pairs, scores, {"istqb": 0.5, "tools": 0.5})
    assert w["istqb"] > 0.5 > w["tools"] and abs(sum(w.values()) - 1) < 1e-3
    assert preferences([{"cv_id": "C", "action": "override", "new_rank": 1}], ["A", "B", "C"]) == [("C", "A"), ("C", "B")]
