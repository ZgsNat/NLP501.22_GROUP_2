"""E26: the goal suite's cases parse and its behaviour checks judge a run the way the docstring says."""

from app.eval.agentic import Trace
from app.eval.goals import cases, checks


def test_every_case_has_a_goal_an_hr_answer_and_expectations():
    ids = [c["id"] for c in cases()]
    assert len(ids) == len(set(ids)) >= 5
    assert all(c["goal"] and c["hr"] and c["expect"] and c["jd"].startswith("JD-") for c in cases())


def test_a_hidden_shortfall_and_a_skipped_question_fail_their_checks():
    case = next(c for c in cases() if c["id"] == "ai_goal")
    tr = Trace("JD-001", {"shortlist": ["CV-001", "CV-003"], "goal_spec": {"count": 5}, "report": {"goal": {"limits": []}},
                          "loops": {}})
    tr.tools = [(0, "parse_jd", {}), (0, "formalize_goal", {}), (0, "screen", {}), (0, "ask_hr", {})]
    res = checks(case, tr)
    assert res["formalized the goal"] and res["asked HR before ranking"] is False
    assert res["declared the shortfall"] is False and res["defined a concept"] is False
    assert res["recommended only CVs with HR's must-haves"]  # CV-001 and CV-003 both name Selenium


def test_pass_rates_count_each_check_over_repeated_runs():
    from app.eval.goals import pass_rates

    case = {"id": "feasible"}
    rows = [(case, None, {"asked HR": True, "within budget": True}),
            (case, None, {"asked HR": False, "within budget": None})]
    assert pass_rates(rows) == {"feasible": {"asked HR": (1, 2), "within budget": (1, 1)}}
