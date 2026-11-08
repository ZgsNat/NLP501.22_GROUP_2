from app.eval.extraction_benchmark import compute_iaa, run_e0_keyword, run_e1_no_context, run_e2_full_pipeline
from app.eval.djinni_error_impact import evaluate_error_propagation


def test_iaa_returns_valid_metrics():
    res = compute_iaa()
    assert "span_overlap_f1" in res
    assert "status_unweighted_kappa" in res
    assert 0.0 <= res["span_overlap_f1"] <= 1.0
    assert -1.0 <= res["status_unweighted_kappa"] <= 1.0


def test_systems_e0_e1_e2_extract_skills():
    sample_text = (
        "Senior Automation QA Engineer\n"
        "Experience: 4 years\n"
        "Stack: Java, Selenium WebDriver, Postman, Jenkins\n"
        "- Developed automation test suites using Playwright with TypeScript.\n"
        "- Familiar with Cypress.\n"
    )
    e0 = run_e0_keyword(sample_text)
    e1 = run_e1_no_context(sample_text)
    e2 = run_e2_full_pipeline("test_cv", sample_text)

    # E0 catches exact keys
    assert "selenium" in e0["candidate_skills"] or "java" in e0["candidate_skills"]

    # E1 catches without hedge awareness
    assert "cypress" in e1["candidate_skills"]

    # E2 catches Playwright and distinguishes hedge
    assert "playwright" in e2["candidate_skills"]
    assert "cypress" in e2["hedged_skills"] or "cypress" not in e2["candidate_skills"]
