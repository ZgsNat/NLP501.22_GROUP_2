import pytest

from app.agent.scoring import canonicalize, is_grounded, knockout_reason, mentions, weighted_total

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

