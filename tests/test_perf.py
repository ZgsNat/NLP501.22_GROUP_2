import pytest

from cv_agent.arbiter.arbiter import selective_arbiter
from cv_agent.eval.perf import EXACT_TIE, percentile, tune_threshold
from cv_agent.ingest.pipeline import ingest_text
from cv_agent.llm.backends import HeuristicBackend
from cv_agent.llm.context import cv_digest, jd_digest
from cv_agent.llm.gateway import Budget, LLMGateway
from cv_agent.schema import ScoredCandidate
from tests.conftest import SENIOR_CV


def test_cv_digest_keeps_verbatim_evidence_lines_only(senior_cv):
    d = cv_digest(senior_cv)
    for line in d.splitlines():
        assert line in senior_cv.masked_text
    assert "Selenium WebDriver" in d and "6 years of experience" in d and "2019 - Present" in d
    assert "[EMAIL]" not in d and "[NAME]" not in d


def test_cv_digest_respects_budget(senior_cv):
    assert len(cv_digest(senior_cv, max_chars=120)) <= 120


def test_jd_digest_lists_must_and_nice(jd):
    d = jd_digest(jd)
    assert "MUST:" in d and "selenium / playwright / appium" in d and "NICE:" in d


def test_arbiter_max_rank_limits_pairs(jd, profile):
    cvs = {f"c{i}": ingest_text(SENIOR_CV, f"c{i}") for i in range(6)}
    ranking = [ScoredCandidate(f"c{i}", score=0.5) for i in range(6)]  # all exact ties
    res = selective_arbiter(LLMGateway(HeuristicBackend()), jd, profile, ranking, cvs, EXACT_TIE, 2, max_rank=2)
    assert len(res.pairs_checked) == 2 and res.consistent_pairs == 0  # heuristic picks first position on ties


def test_arbiter_interrupted_keeps_partial_decisions(jd, profile):
    cvs = {c: ingest_text(SENIOR_CV, c) for c in ("a", "b", "c", "d")}
    ranking = [ScoredCandidate("a", score=0.50), ScoredCandidate("b", score=0.52),
               ScoredCandidate("c", score=0.30), ScoredCandidate("d", score=0.31)]
    gw = LLMGateway(HeuristicBackend(), budget=Budget(max_calls=2))
    res = selective_arbiter(gw, jd, profile, ranking, cvs, 0.05, 2)
    assert [c.cv_id for c in res.ranking[:2]] == ["b", "a"] and res.interrupted


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 50) != percentile([], 50)  # NaN


def test_tune_threshold_offline(tmp_path, monkeypatch):
    from cv_agent.data.synthetic import generate
    from cv_agent.eval.experiments import load_jd_specs
    specs = {s["id"]: {**s["gold_spec"], "split": s.get("split", "test")} for s in load_jd_specs()}
    generate(tmp_path, specs, n=30, seed=3)
    monkeypatch.chdir(tmp_path)
    res = tune_threshold([EXACT_TIE, 0.05], tmp_path, "dev", overrides={"llm": {"backend": "heuristic"}})
    assert res["recommended_threshold"] in (EXACT_TIE, 0.05)
    pairs = [r["judge_pairs"] for r in res["table"]]
    assert pairs[0] <= pairs[1]


def test_claims_cite_line_numbers_become_verbatim_spans(senior_cv, jd):
    from cv_agent.llm.context import cv_digest_lines
    from cv_agent.verify.claims import _span_for
    lines = cv_digest_lines(senior_cv, 1000)
    assert _span_for({"line": 1}, lines) == lines[0] and lines[0] in senior_cv.masked_text
    assert _span_for({"line": 99, "span": "x"}, lines) == "x"
    assert _span_for({"line": "abc"}, lines) == ""


def test_repair_prompt_only_lists_rejected_claims():
    from cv_agent.llm import prompts
    p = prompts.claims("JOB", "[1] Selenium", "- Uses Appium (cited: 'Selenium')")
    assert "REJECTED" in p and "Uses Appium" in p
