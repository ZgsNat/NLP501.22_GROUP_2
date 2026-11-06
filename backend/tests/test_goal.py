"""Goal-driven control (agent/goal.py): provenance, open-vocabulary concepts graded by code, feasibility, bonuses and
the goal check: everything the agent plans with is computed without an LLM."""

from app.agent import goal
from app.agent.goal import Constraint, GoalSpec
from app.nlp.pipeline import process

AI_TERMS = ["copilot", "chatgpt", "llm", "self-healing", "healenium"]


def cv(cv_id, experience, skills=""):
    text = f"# QA Automation Engineer\n\n## Experience\n- {experience}\n\n## Skills\n{skills}\n"
    return process(cv_id, text)


POOL = {
    "A": cv("A", "Built a Selenium framework in Java with self-healing locators (Healenium) for 4 years", "Selenium, Java"),
    "B": cv("B", "Wrote Selenium tests in Python for checkout flows", "Selenium, Python, ChatGPT"),
    "C": cv("C", "Wrote Playwright tests in TypeScript", "Playwright, currently learning Copilot"),
    "D": cv("D", "Wrote Selenium tests in Java", "Selenium, Java"),
}
CONCEPTS = {"ai_in_testing": {"terms": AI_TERMS, "evidence": {c: e for c, p in POOL.items()
                                                                if (e := goal.concept_evidence(p, AI_TERMS))}}}
HR = ("5 CV, trong đó yêu cầu selenium là bắt buộc, playwright là có nhưng có thể dùng các tiêu chí khác để đánh giá, "
      "AI là bắt buộc, nếu biết xử dụng AI trong automation thì là điểm cộng lớn")


def test_a_constraint_must_quote_hr_word_for_word_ignoring_case_and_accents():
    assert goal.quoted("selenium là bắt buộc", HR) and goal.quoted("SELENIUM LA BAT BUOC", HR)
    assert goal.quoted("AI", HR) and not goal.quoted("ai", "the team maintains it")  # a word, not a substring
    assert not goal.quoted("Cypress là bắt buộc", HR)


def test_concept_evidence_is_graded_by_where_and_how_the_cv_says_it():
    ev = CONCEPTS["ai_in_testing"]["evidence"]
    assert ev["A"]["strength"] == "used"  # in a role
    assert ev["B"]["strength"] == "listed"  # skills list only
    assert ev["C"]["strength"] == "hedged"  # "currently learning"
    assert "D" not in ev
    for line in ("Đang học Copilot cho test tự động", "Dang hoc Copilot cho test tu dong"):  # typed with or without accents
        assert goal.concept_evidence(cv("V", "Wrote Selenium tests", line), AI_TERMS)["strength"] == "hedged", line


def test_feasibility_counts_the_pool_names_the_bottleneck_and_what_relaxing_gives():
    spec = GoalSpec(count=3, constraints=[Constraint(kind="must", target="selenium", hr_quote="selenium"),
                                          Constraint(kind="must", target="ai_in_testing", hr_quote="AI")])
    f = goal.feasibility(spec, POOL, CONCEPTS, {})
    assert f["per_must"] == {"selenium": 3, "ai_in_testing": 2} and f["meet_all_musts"] == 2
    assert f["status"] == "infeasible" and f["bottleneck"] == "ai_in_testing"
    assert f["if_relaxed"] == {"selenium": 2, "ai_in_testing": 3}
    unknown = GoalSpec(constraints=[Constraint(kind="must", target="quantum", hr_quote="quantum")])
    assert goal.feasibility(unknown, POOL, CONCEPTS, {})["status"] == "unknown_targets"


def test_a_big_plus_scores_by_evidence_credit_and_a_hedge_counts_little():
    spec = GoalSpec(constraints=[Constraint(kind="bonus", target="ai_in_testing", strength="strong", hr_quote="AI")])
    pts = {c: goal.bonus(p, spec, CONCEPTS).get("ai_in_testing", 0) for c, p in POOL.items()}
    assert pts["A"] > pts["B"] > pts["C"] > pts["D"] == 0


def test_concept_must_and_exclude_filter_with_a_reason():
    spec = GoalSpec(constraints=[Constraint(kind="must", target="ai_in_testing", hr_quote="AI"),
                                 Constraint(kind="exclude", target="python", hr_quote="python")])
    assert goal.goal_filter(POOL["A"], spec, CONCEPTS, {}) is None
    assert "excluded" in goal.goal_filter(POOL["B"], spec, CONCEPTS, {})
    assert "must-have 'ai_in_testing'" in goal.goal_filter(POOL["C"], spec, CONCEPTS, {})  # hedged is not enough


def test_goal_report_lists_what_the_answer_must_declare():
    spec = GoalSpec(count=5, constraints=[Constraint(kind="must", target="selenium", hr_quote="selenium"),
                                          Constraint(kind="bonus", target="ai_in_testing", strength="strong", hr_quote="AI")])
    rep = goal.goal_report(spec, ["A", "D"], POOL, CONCEPTS, {})
    assert any(u.startswith("count: 2 of the 5") for u in rep["unmet"])
    assert {"constraint": "bonus ai_in_testing", "met_by": 1, "of": 2} in rep["items"]
    assert not goal.goal_report(spec, ["D"], POOL, CONCEPTS, {})["items"][1]["met_by"]


def test_the_jds_own_must_haves_count_in_feasibility_and_are_labelled():
    spec = GoalSpec(count=2, constraints=[Constraint(kind="must", target="selenium", hr_quote="selenium")])
    assert goal.feasibility(spec, POOL, CONCEPTS, {})["meet_all_musts"] == 3
    f = goal.feasibility(spec, POOL, CONCEPTS, {}, jd_musts=["java"])
    assert f["per_must"] == {"selenium": 3, "java (JD)": 2} and f["meet_all_musts"] == 2  # A and D: Selenium + Java


def test_open_questions_block_only_when_the_answer_can_change_who_meets_the_goal():
    skills = GoalSpec(count=2, constraints=[Constraint(kind="must", target="selenium", hr_quote="selenium là bắt buộc")],
                      open_questions=["Is GitLab CI fine instead of Jenkins?"])
    concept = skills.model_copy(update={"constraints": [*skills.constraints, Constraint(
        kind="bonus", target="ai_in_testing", hr_quote="AI trong automation")]})
    assert goal.blocking_questions(skills, {"status": "ok"}, CONCEPTS) == []  # met as written: an assumption
    assert goal.blocking_questions(skills, {"status": "infeasible"}, CONCEPTS) == skills.open_questions
    assert goal.blocking_questions(concept, {"status": "ok"}, CONCEPTS) == skills.open_questions  # terms to confirm
    rep = goal.goal_report(skills, ["A", "B"], POOL, CONCEPTS, {}, {"status": "ok"})
    assert rep["unmet"] == [] and rep["assumed"] == skills.open_questions
    assert goal.goal_report(skills, ["A", "B"], POOL, CONCEPTS, {})["unmet"]  # no feasibility known: still blocks
