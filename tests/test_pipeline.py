from pathlib import Path

import pytest

from cv_agent.arbiter.arbiter import find_near_ties, selective_arbiter
from cv_agent.config import load_config
from cv_agent.eval.metrics import cohen_kappa, kendall_tau, mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from cv_agent.hitl.audit import AuditLog, Decision, calibrate_weights, preferences_from_decisions
from cv_agent.ingest.pipeline import ingest_text
from cv_agent.llm.backends import HeuristicBackend
from cv_agent.llm.gateway import Budget, BudgetExceeded, LLMGateway, PIILeakError
from cv_agent.orchestration.graph import build_runtime, run_pipeline
from cv_agent.ranking.scoring import score_candidate
from cv_agent.schema import ScoredCandidate
from tests.conftest import MANUAL_CV, SENIOR_CV

HEUR = {"llm": {"backend": "heuristic"}, "paths": {"cache_dir": None}}


def _cvs():
    variants = [SENIOR_CV, MANUAL_CV, SENIOR_CV.replace("Selenium WebDriver", "Playwright"),
                SENIOR_CV.replace("6 years", "3 years").replace("ISTQB CTFL", ""), MANUAL_CV.replace("1 years", "4 years")]
    return [ingest_text(t, f"cv{i}") for i, t in enumerate(variants)]


@pytest.mark.parametrize("exp", ["B0", "B1", "B2", "B3", "B4", "LLM-all", "P6", "A1", "A2"])
def test_every_configuration_runs_end_to_end(exp, jd):
    cfg = load_config(exp, overrides=HEUR)
    out = run_pipeline(build_runtime(cfg), jd, _cvs())
    assert len(out["ranking"]) == 5
    if cfg["criteria"]["enabled"]:
        assert out["cards"] and out["ranking"][0].must_have_ok
    if exp in {"B0", "B1", "B2", "B3"}:
        assert out["llm_stats"]["calls"] == 0
    if exp == "A1":
        assert all(c.status == "unverified" for cl in out["claims"].values() for c in cl)


def test_p6_verified_claims_only_supported_or_flagged(jd):
    out = run_pipeline(build_runtime(load_config("P6", overrides=HEUR)), jd, _cvs())
    statuses = {c.status for cl in out["claims"].values() for c in cl}
    assert statuses <= {"supported", "no_evidence"}


def test_budget_exceeded_triggers_fallback(jd):
    cfg = load_config("P6", overrides={**HEUR, "budget": {"max_calls": 1}})
    out = run_pipeline(build_runtime(cfg), jd, _cvs())
    assert out["fallback"] is True and out["cards"]


def test_gateway_refuses_unmasked_pii():
    gw = LLMGateway(HeuristicBackend())
    with pytest.raises(PIILeakError):
        gw.call("listwise", "v", "sys", "contact a@b.com", {"scores": {}})


def test_gateway_logs_every_call(tmp_path: Path):
    gw = LLMGateway(HeuristicBackend(), log_path=tmp_path / "calls.jsonl", budget=Budget(max_calls=2))
    gw.call("listwise", "v1", "sys", "user", {"scores": {"a": 1}})
    rec = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert '"prompt_version": "v1"' in rec and '"latency_s"' in rec
    gw.call("listwise", "v1", "sys", "user2", {"scores": {"a": 1}})
    with pytest.raises(BudgetExceeded):
        gw.call("listwise", "v1", "sys", "user3", {"scores": {"a": 1}})


def test_arbiter_swaps_only_near_ties_with_both_orders(jd, profile):
    cvs = {c.cv_id: c for c in _cvs()}
    ranking = [ScoredCandidate("cv1", score=0.50), ScoredCandidate("cv0", score=0.52), ScoredCandidate("cv2", score=0.10)]
    assert find_near_ties(ranking, 0.05) == [0]
    gw = LLMGateway(HeuristicBackend())
    res = selective_arbiter(gw, jd, profile, ranking, cvs, threshold=0.05, passes=2)
    assert [c.cv_id for c in res.ranking] == ["cv0", "cv1", "cv2"]
    assert res.pairs_checked == [("cv1", "cv0")] and len(res.call_ids) == 2


def test_metrics():
    rel = {"a": 3, "b": 0, "c": 2}
    assert recall_at_k(["a", "b"], rel, 2) == 0.5
    assert mrr_at_k(["b", "c"], rel) == 0.5
    assert ndcg_at_k(["a", "c", "b"], rel) == pytest.approx(1.0)
    assert kendall_tau(["a", "c"], ["a", "c"]) == pytest.approx(1.0)
    assert kendall_tau(["c", "a"], ["a", "c"]) == pytest.approx(-1.0)
    assert cohen_kappa({"x": 1, "y": 2}, {"x": 1, "y": 2}) == 1.0
    assert paired_bootstrap([1, 1, 1], [0, 0, 0])["ci_low"] == 1.0


def test_hitl_requires_reason_and_calibrates(tmp_path, jd, profile):
    log = AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(ValueError):
        log.record(Decision("r", "j", "cv0", "approve", reason=" "))
    log.record(Decision("r", "j", "cv1", "override", reason="manual tester rất mạnh", new_rank=1))
    scored = {c.cv_id: score_candidate(c, jd, profile) for c in _cvs()}
    pairs = preferences_from_decisions(log.load("j"), ["cv0", "cv2", "cv1"])
    assert pairs == [("cv1", "cv0"), ("cv1", "cv2")]
    w = calibrate_weights(pairs, scored, profile.weights)
    assert abs(sum(w.values()) - 1) < 1e-9
