from datetime import date

from cv_agent.criteria.clarify import apply_answers, detect_issues, simulated_hr
from cv_agent.criteria.jd_parser import parse_jd
from cv_agent.criteria.profiles import list_profiles, normalize_weights
from cv_agent.ingest.extractor import estimate_years
from cv_agent.ingest.pipeline import ingest_text, mark_duplicates
from tests.conftest import SENIOR_CV


def test_entities_have_spans_pointing_into_masked_text(senior_cv):
    for e in senior_cv.entities:
        assert senior_cv.masked_text[e.span.start:e.span.end] == e.span.text
    assert {"selenium", "rest assured", "postman"} <= set(senior_cv.values("tool"))
    assert {"java", "python"} <= set(senior_cv.values("language"))
    assert "istqb" in senior_cv.values("certification")


def test_depth_and_years(senior_cv, manual_cv):
    assert senior_cv.depth.level == 3 and senior_cv.depth.evidence
    assert senior_cv.years_experience == 6
    assert manual_cv.depth.level == 1


def test_years_from_ranges_when_not_stated():
    text = "Company A (01/2018 - 12/2019)\nCompany B (2020 - Present)"
    assert abs(estimate_years(text, today=date(2026, 1, 1)) - 8.0) <= 0.1


def test_low_quality_cv_is_flagged_not_dropped():
    cv = ingest_text("@@## ~~ %% selenium", "bad")
    assert cv.parse_quality < 0.7 and "low_confidence_parse" in cv.flags


def test_duplicate_detection():
    a, b = ingest_text(SENIOR_CV, "a"), ingest_text(SENIOR_CV + "\n", "b")
    out = mark_duplicates([a, b])
    assert "duplicate_of:a" in out[1].flags


def test_jd_must_and_nice(jd):
    must = {(r.axis, r.values) for r in jd.must_haves}
    assert ("tools", ("selenium", "playwright", "appium")) in must
    assert ("languages", ("java", "python")) in must
    assert any(r.axis == "experience" and r.min_level == 2 for r in jd.must_haves)
    assert {r.axis for r in jd.nice_to_haves} >= {"certification", "cicd"}


def test_profiles_sum_to_one_and_four_templates():
    profiles = list_profiles()
    assert {"Mặc định", "Outsourcing", "Ngân hàng", "Startup"} <= set(profiles)
    for p in profiles.values():
        assert abs(sum(p.weights.values()) - 1) < 1e-9


def test_normalize_weights_is_pure():
    w = {"tools": 2.0, "depth": 2.0}
    out = normalize_weights(w)
    assert w == {"tools": 2.0, "depth": 2.0} and abs(sum(out.values()) - 1) < 1e-9 and out["tools"] == 0.5


def test_clarification_detects_vague_senior_and_applies_answer():
    jd = parse_jd("Senior QA Automation\nRequirements\n- Strong automation tools, etc.\n- Good English communication", "amb")
    qs = detect_issues(jd)
    ids = {q.qid for q in qs}
    assert {"seniority_years", "english_level", "tool_family"} <= ids
    new_jd = apply_answers(jd, qs, simulated_hr(qs, {"seniority_years": 1}))
    assert any(r.axis == "experience" and r.min_level == 5 and r.must for r in new_jd.requirements)
    assert jd.requirements != new_jd.requirements  # original untouched


def test_clarification_detects_junior_conflict():
    jd = parse_jd("Junior Tester\nRequirements\n- At least 4 years of experience in testing\n- Selenium", "j")
    assert "junior_vs_years" in {q.qid for q in detect_issues(jd)}
