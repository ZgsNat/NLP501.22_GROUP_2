import pytest

from app.eval.djinni_extraction_annotation import AnnotationError
from app.eval.extraction_benchmark import (
    GOLD_PATH,
    WEAK_REF_PATH,
    generate_benchmark_report,
    load_reference,
    run_e0_keyword,
    run_e1_no_context,
    run_e2_full_pipeline,
)
from app.eval.djinni_error_impact import evaluate_error_propagation


def test_generated_reference_cannot_enter_primary_benchmark():
    with pytest.raises(AnnotationError, match="human-adjudicated Gold"):
        load_reference(WEAK_REF_PATH)


def test_generated_reference_is_explicitly_weak_only():
    profiles, metadata = load_reference(WEAK_REF_PATH, weak_diagnostic=True)
    assert profiles
    assert metadata["human"] is False
    assert metadata["evaluation_status"] == "WEAK_DIAGNOSTIC_ONLY"


def test_primary_benchmark_fails_closed_while_human_gold_is_missing():
    assert not GOLD_PATH.exists()
    with pytest.raises(AnnotationError, match="Human Gold is not available"):
        generate_benchmark_report()


def test_error_impact_fails_closed_while_human_gold_is_missing():
    with pytest.raises(AnnotationError, match="Human Gold is not available"):
        evaluate_error_propagation()


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
