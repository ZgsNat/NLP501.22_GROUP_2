import pytest

from app.agent.scoring import (
    canonicalize,
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
    ],
)
def test_canonicalize(phrases, hard, soft, subs):
    assert canonicalize(phrases) == (hard, soft, subs)


def test_knockout_respects_substitutes():
    cypress_cv = "Wrote Cypress tests in TypeScript."
    assert knockout_reason(cypress_cv, ["playwright"], {}) == "Missing must-have: playwright"
    assert knockout_reason(cypress_cv, ["playwright"], {"playwright": ["cypress"]}) is None


def test_weighted_total_and_knockout():
    a = {"criteria": {"tools": {"score": 100}, "istqb": {"score": 50}}}
    assert weighted_total(a, {"tools": 0.8, "istqb": 0.2}, False) == 0.9
    assert weighted_total(a, {"tools": 0.8, "istqb": 0.2}, True) == 0.0


def test_grounding_accepts_verbatim_and_formatting_noise_rejects_fabrication():
    assert is_grounded("built a playwright + typescript framework", CV)
    assert is_grounded("**Built a Playwright + TypeScript framework**", CV)
    assert not is_grounded("Led a team of 12 SDETs at a bank", CV)



def test_tools_score_counts_must_nice_and_substitutes():
    cv = "- Built Cypress suites in TypeScript\n- Ran them in GitHub Actions"
    full = tools_score(cv, ["cypress", "typescript"], ["github actions"], {})
    assert full["score"] == 100 and all(e["grounded"] for e in full["evidence"])
    sub = tools_score(cv, ["playwright"], [], {"playwright": ["cypress"]})
    assert sub["score"] == 60 and "cypress for playwright" in sub["rationale"]


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
