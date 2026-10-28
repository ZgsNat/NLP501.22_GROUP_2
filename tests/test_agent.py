from dataclasses import replace
from pathlib import Path

import pytest

from cv_agent.agent.critic import critique
from cv_agent.agent.memory import AgentMemory
from cv_agent.agent.planner import state_summary
from cv_agent.agent.tools import ToolError, _clean_terms, compare
from cv_agent.config import ROOT, load_config
from cv_agent.eval.experiments import jd_from_spec, load_jd_specs
from cv_agent.ingest.pipeline import ingest_text
from cv_agent.llm.backends import HeuristicBackend
from cv_agent.llm.gateway import Budget, LLMGateway
from cv_agent.orchestration.graph import build_runtime, run_pipeline

N_CVS = 40


@pytest.fixture(scope="module")
def corpus():
    files = sorted((ROOT / "data" / "synthetic" / "cvs").glob("syn_[0-9]*.txt"))[:N_CVS]
    return [ingest_text(f.read_text(encoding="utf-8"), f.stem) for f in files]


@pytest.fixture(scope="module")
def fpt():
    return jd_from_spec({s["id"]: s for s in load_jd_specs()}["fpt_automation_test"])


def _cfg(name="AG", **agent):
    over = {"llm": {"backend": "heuristic"}, "paths": {"cache_dir": None}}
    if agent:
        over["agent"] = agent
    return load_config(name, overrides=over)


def _run(cfg, jd, cvs, backend=None, budget=None, approve_fn=None, hr_answer_fn=None):
    rt = build_runtime(cfg, approve_fn=approve_fn, hr_answer_fn=hr_answer_fn)
    if backend is not None or budget is not None:
        rt = replace(rt, gateway=LLMGateway(backend or HeuristicBackend(), budget=budget or Budget()))
    return run_pipeline(rt, jd, cvs)


class ScriptedPlanner(HeuristicBackend):
    """Heuristic LLM, except the planner always answers with a fixed (possibly invalid) decision."""

    def __init__(self, decision):
        super().__init__()
        self.decision = decision

    def _planner(self, ctx):
        return dict(self.decision)


def test_llm_planner_runs_to_goal_and_logs_every_decision(corpus, fpt):
    out = _run(_cfg(), fpt, corpus)
    log = out["agent_log"]
    assert log[-1]["action"] == "finish" and out["critic"]["passed"]
    assert {e["source"] for e in log} == {"llm"} and all(e["observation"] for e in log)
    actions = [e["action"] for e in log]
    assert actions[0] == "search" and {"rank", "verify_claims"} <= set(actions)
    assert len(out["cards"]) == 10 and out["llm_stats"]["by_task"]["planner"] == len(log)


def test_rule_planner_matches_llm_planner_that_follows_rules(corpus, fpt):
    rules = _run(_cfg("AG-rules"), fpt, corpus)
    llm = _run(_cfg(), fpt, corpus)
    assert [c.cv_id for c in rules["ranking"][:10]] == [c.cv_id for c in llm["ranking"][:10]]
    assert "planner" not in rules["llm_stats"]["by_task"]


@pytest.mark.parametrize("decision", [
    {"action": "finish", "args": {}},                               # not available before the goal is met
    {"action": "compare", "args": {"a": "syn_000", "b": "nope"}},  # unavailable early / invalid pair later
    {"action": "delete_all_cvs", "args": {}},                       # not a tool
    {"action": "search", "args": "not a dict"},
])
def test_invalid_llm_decisions_are_replaced_by_rules(corpus, fpt, decision):
    out = _run(_cfg(), fpt, corpus, backend=ScriptedPlanner(decision))
    assert out["agent_log"][-1]["action"] == "finish" and out["cards"]
    # the scripted answer is accepted only where it is valid (e.g. "finish" once the goal is met)
    assert out["planner_stats"].get("fallback", 0) >= 1 and out["planner_stats"].get("llm", 0) <= 1
    assert all(e["source"].startswith("rules(fallback") or e["action"] == decision["action"]
               for e in out["agent_log"])


def test_search_terms_outside_taxonomy_are_dropped():
    kept, dropped = _clean_terms(["Selenium WebDriver", "rm -rf /", "playwright", "selenium"])
    assert kept == ["selenium", "playwright"] and "rm -rf /" in dropped
    with pytest.raises(ToolError):
        _clean_terms("selenium")


def test_compare_refuses_pairs_outside_the_guardrail(corpus, fpt):
    out = _run(_cfg("AG-rules"), fpt, corpus)
    rt = build_runtime(_cfg())
    far = (out["ranking"][0].cv_id, out["ranking"][15].cv_id)
    with pytest.raises(ToolError):
        compare(rt, out, {"a": far[0], "b": far[1]})


def test_hr_checkpoint_cannot_be_skipped(corpus, fpt):
    with pytest.raises(PermissionError):
        _run(_cfg(), fpt, corpus, approve_fn=lambda jd, profile: False)


def test_step_budget_forces_finish_with_a_fresh_ranking(corpus, fpt):
    out = _run(_cfg(max_steps=3), fpt, corpus)
    assert len(out["agent_log"]) <= 5 and out["agent_log"][-1]["action"] == "finish"
    assert out["agent_log"][-1]["source"] == "budget" and not out.get("ranking_stale")


def test_llm_budget_exhaustion_falls_back_and_still_reports(corpus, fpt):
    out = _run(_cfg(), fpt, corpus, budget=Budget(max_calls=4))
    assert out["fallback"] and out["agent_log"][-1]["action"] == "finish" and out["cards"]


def test_critic_marks_unreachable_goal_instead_of_blocking(corpus, fpt):
    cfg = _cfg(goals={"min_must_pass_shortlist": 10, "require_verified_claims": False, "resolve_near_ties": False})
    out = _run(cfg, fpt, corpus[:8])
    report = critique(out, cfg, searches_left=0)
    finding = next(f for f in report.findings if f.goal == "must_pass_shortlist")
    assert finding.unreachable and report.passed


def test_planner_summary_holds_ids_and_scores_only(corpus, fpt):
    out = _run(_cfg("AG-rules"), fpt, corpus)
    summary = str(state_summary(build_runtime(_cfg()), out))
    assert "[EMAIL]" not in summary and "Nguyen" not in summary and "syn_" in summary


def test_memory_answers_and_weights_roundtrip(tmp_path, fpt):
    mem = AgentMemory(tmp_path / "m.json")
    mem.remember_answers(fpt, {"seniority_years": 1})
    mem.remember_answers(fpt, {"other": 0})
    assert mem.recall_answers(fpt) == {"seniority_years": 1, "other": 0}
    weights = {"tools": 0.5, "depth": 0.5}
    mem.remember_weights("Outsourcing", weights, n_pairs=12)
    assert mem.recall_weights("Outsourcing") == weights and mem.recall_weights("Startup") is None
    with pytest.raises(ValueError):
        mem.remember_weights("Outsourcing", {"tools": 0.7}, 1)
    (tmp_path / "bad.json").write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError):
        AgentMemory(tmp_path / "bad.json").recall_answers(fpt)


def test_agent_reuses_remembered_hr_answers(corpus, tmp_path):
    spec = {s["id"]: s for s in load_jd_specs()}["ambiguous_senior"]
    jd = jd_from_spec(spec)
    cfg = _cfg(memory={"enabled": True, "path": str(tmp_path / "mem.json"), "apply_weights": True})
    hr = lambda _jd, qs: {q.qid: q.recommended for q in qs}  # noqa: E731 (a real HR callback)
    first = _run(cfg, jd, corpus, hr_answer_fn=hr)
    asked = [e for e in first["agent_log"] if e["action"] == "ask_hr"]
    assert asked, "the ambiguous JD should make the agent ask HR"
    second = _run(cfg, jd, corpus)
    assert not [e for e in second["agent_log"] if e["action"] == "ask_hr"]
    assert any("dùng lại" in t for t in second["trace"])


def test_agent_applies_remembered_weights(corpus, fpt, tmp_path):
    mem_path = tmp_path / "mem.json"
    weights = {"tools": 1.0}
    AgentMemory(mem_path).remember_weights("Outsourcing", weights, 5)
    cfg = _cfg(memory={"enabled": True, "path": str(mem_path), "apply_weights": True})
    rt = build_runtime(cfg)
    from cv_agent.criteria.profiles import list_profiles
    out = run_pipeline(rt, fpt, corpus, list_profiles()["Outsourcing"])
    assert out["profile"].weights == weights


def test_experiment_configs_disable_memory():
    for name in ("AG", "AG-rules"):
        assert load_config(name)["agent"]["memory"]["enabled"] is False
    assert Path(load_config("P6")["agent"]["memory"]["path"]).name == "agent_memory.json"


def test_simulated_answers_are_not_stored_as_hr_input(corpus, tmp_path):
    jd = jd_from_spec({s["id"]: s for s in load_jd_specs()}["ambiguous_senior"])
    mem = tmp_path / "mem.json"
    cfg = _cfg(memory={"enabled": True, "path": str(mem), "apply_weights": True})
    _run(cfg, jd, corpus)  # no hr_answer_fn -> simulated answers
    assert not mem.exists() or AgentMemory(mem).recall_answers(jd) == {}


def test_corrupt_memory_file_does_not_stop_the_agent(corpus, fpt, tmp_path):
    mem = tmp_path / "mem.json"
    mem.write_text("{broken", encoding="utf-8")
    out = _run(_cfg(memory={"enabled": True, "path": str(mem), "apply_weights": True}), fpt, corpus)
    assert out["agent_log"][-1]["action"] == "finish" and any("bộ nhớ hỏng" in t for t in out["trace"])


def test_agent_without_llm_judge_finishes_with_unreachable_goals(corpus, fpt):
    out = _run(load_config("AG", overrides={"llm": {"backend": "heuristic"}, "paths": {"cache_dir": None},
                                           "judge": {"mode": "none"}}), fpt, corpus)
    assert out["agent_log"][-1]["action"] == "finish"
    assert {f["goal"] for f in out["critic"]["findings"] if f.get("unreachable")} >= {"claims_verified"}


def test_empty_corpus_finishes_with_empty_report(fpt):
    out = _run(_cfg(), fpt, [])
    assert out["agent_log"][-1]["action"] == "finish" and not out.get("ranking")


def test_swapped_candidate_is_not_judged_again():
    from cv_agent.agent.critic import open_near_ties
    from cv_agent.schema import ScoredCandidate
    ranking = [ScoredCandidate(c, score=0.5, must_have_ok=True) for c in "pabc"]
    state = {"ranking": ranking, "compared_pairs": [["a", "b"]], "swapped": ["a", "b"]}
    assert open_near_ties(state, 10, 0.05) == []  # (p,a), (a,b), (b,c) all touch a swapped candidate
    assert open_near_ties({**state, "swapped": []}, 10, 0.05) == [("p", "a"), ("b", "c")]


def test_search_is_blocked_while_hr_questions_are_open(corpus):
    from cv_agent.agent.tools import search_unavailable
    jd = jd_from_spec({s["id"]: s for s in load_jd_specs()}["ambiguous_senior"])
    rt = build_runtime(_cfg())
    from cv_agent.agent.graph import agent_criteria_node
    state = {"jd": jd, "cvs": {c.cv_id: c for c in corpus}, "trace": []}
    state = {**state, **agent_criteria_node(rt, state)}
    assert state["questions"] and search_unavailable(rt, state)


@pytest.mark.parametrize("raw,name", [("finish {}", "finish"), ("compare(a, b)", "compare"), (" Rank ", "rank"),
                                      (None, "")])
def test_action_names_are_normalised(raw, name):
    from cv_agent.agent.planner import _action_name
    assert _action_name(raw) == name


def test_compare_budget_caps_llm_judgements(corpus, fpt):
    out = _run(_cfg(max_compares=2), fpt, corpus)
    assert len([e for e in out["agent_log"] if e["action"] == "compare"]) <= 2
    assert out["agent_log"][-1]["action"] == "finish" and out["critic"]["passed"]


# ------------------------------------------------------------------ budgets that span the whole run

@pytest.fixture(scope="module")
def ranked_fpt(corpus, fpt):
    return _run(_cfg("AG-rules"), fpt, corpus)


def test_compare_budget_is_not_refilled_by_a_new_search(ranked_fpt):
    from cv_agent.agent.tools import compare_unavailable
    cfg = _cfg()
    rt = build_runtime(cfg)
    state = {**ranked_fpt, "compared_pairs": [], "swapped": [], "compares": cfg["agent"]["max_compares"]}
    assert compare_unavailable(rt, state) == "đã so sánh đủ số cặp cho phép"
    assert state_summary(rt, state)["compares_left"] == 0


def test_search_does_not_reset_the_compare_count(ranked_fpt):
    from cv_agent.agent.tools import search
    rt = build_runtime(_cfg())
    state = {**ranked_fpt, "must_pass_count": 0, "searches": 1, "compares": 3}
    update, _obs = search(rt, state, {"extra_terms": ["appium"]})
    assert update["compared_pairs"] == [] and "compares" not in update


def test_search_is_blocked_once_the_pool_has_enough_must_pass_candidates(ranked_fpt):
    from cv_agent.agent.tools import search_unavailable
    rt = build_runtime(_cfg())
    state = {**ranked_fpt, "searches": 1, "expansion": []}
    assert search_unavailable(rt, {**state, "must_pass_count": 50}) == "pool đã đủ ứng viên đạt must-have"
    assert search_unavailable(rt, {**state, "must_pass_count": 0}) is None


def test_search_without_new_terms_is_refused(ranked_fpt, fpt):
    from cv_agent.agent.tools import search
    from cv_agent.retrieval.hybrid import expansion_terms
    rt = build_runtime(_cfg())
    used = list(expansion_terms(fpt)) + ["selenium"]
    state = {**ranked_fpt, "must_pass_count": 0, "searches": 1, "expansion": used}
    with pytest.raises(ToolError):
        search(rt, state, {"extra_terms": ["selenium"]})
    with pytest.raises(ToolError):
        search(rt, state, {})
    from cv_agent.agent.tools import run_critic, search_unavailable
    assert search_unavailable(rt, state) == "không còn thuật ngữ mở rộng mới"
    finding = next(f for f in run_critic(rt, state)["critic"]["findings"] if f["goal"] == "must_pass_shortlist")
    assert finding["ok"] or finding["unreachable"]


def test_critic_does_not_suggest_a_blocked_search(ranked_fpt):
    from dataclasses import replace as dc_replace
    from cv_agent.agent.tools import run_critic
    rt = build_runtime(_cfg())
    ko = [dc_replace(c, must_have_ok=False) for c in ranked_fpt["ranking"]]
    state = {**ranked_fpt, "ranking": ko, "searches": 1, "must_pass_count": 50}
    finding = next(f for f in run_critic(rt, state)["critic"]["findings"] if f["goal"] == "must_pass_shortlist")
    assert finding["unreachable"] and finding["suggestion"] is None
