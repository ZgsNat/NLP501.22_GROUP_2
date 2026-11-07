import json
from collections import Counter

import pytest

from app.eval import djinni_gold as djinni
from app.eval import bench as bench_module
from app.eval.bench import load


def test_djinni_sources_cover_the_same_blind_pool():
    vu, dan = djinni._load(djinni.VU), djinni._load(djinni.DAN)
    stats = djinni.agreement(vu, dan)

    assert stats["pairs"] == 424
    assert stats["exact"] == pytest.approx(284 / 424)
    assert stats["weighted_kappa"] == pytest.approx(0.6964640018875821)
    assert stats["confidence_counts"] == {"high": 284, "medium": 111, "low": 29}


def test_aggregate_gold_keeps_half_steps_and_uncertainty():
    gold = djinni.build_gold()
    labels = gold["labels"]

    assert gold["method"] == "dual-annotator arithmetic mean; no rounding"
    assert gold["adjudicated"] is False
    assert len(labels) == len({(x["jd"], x["cv"]) for x in labels}) == 424
    assert Counter(x["grade"] for x in labels) == {
        0.0: 118, 0.5: 65, 1.0: 81, 1.5: 34, 2.0: 79, 2.5: 20, 3.0: 27
    }
    assert sum(x["relevant"] for x in labels) == 118


def test_committed_gold_is_exactly_reproducible_from_sources():
    committed = json.loads(djinni.GOLD.read_text(encoding="utf-8"))
    rebuilt = djinni.build_gold()
    assert committed["labels"] == rebuilt["labels"]
    assert committed["source_files"] == rebuilt["source_files"]
    for key in ("pairs", "exact", "weighted_kappa", "within_one", "mean_absolute_difference",
                "bias_vu_minus_dan", "confidence_counts"):
        assert committed["agreement"][key] == pytest.approx(rebuilt["agreement"][key])
    for jd, row in committed["agreement"]["per_jd"].items():
        assert row == pytest.approx(rebuilt["agreement"]["per_jd"][jd])


def test_djinni_is_a_real_benchmark_adapter():
    bench = load("djinni")
    assert bench.name == "djinni"
    assert len(bench.jds) == 8 and len(bench.cvs) == 250
    assert sum(len(rows) for rows in bench.gold.values()) == 424


def test_production_scoring_ignores_unjudged_profiles(monkeypatch, tmp_path):
    monkeypatch.setattr(djinni, "PRODUCTION_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(djinni, "PRODUCTION_RESULTS", tmp_path / "results")
    djinni.PRODUCTION_RESULTS.mkdir()
    result_path = djinni._result_path("B0 BM25")
    result_path.write_text(json.dumps({"system": "B0 BM25", "jds": {"j": ["unjudged", "good", "bad"]}}))
    gold = {"labels": [
        {"jd": "j", "cv": "good", "grade": 3.0, "lower": 3, "upper": 3, "relevant": True},
        {"jd": "j", "cv": "bad", "grade": 0.0, "lower": 0, "upper": 0, "relevant": False},
    ]}

    result = djinni.evaluate_production(gold)

    assert result["systems"]["B0 BM25"]["ndcg10"][0] == pytest.approx(1.0)
    assert result["systems"]["B0 BM25"]["p5_conservative"][0] == pytest.approx(0.2)


def test_production_report_keeps_missing_p6_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(djinni, "PRODUCTION_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(djinni, "PRODUCTION_RESULTS", tmp_path / "results")
    djinni.PRODUCTION_RESULTS.mkdir()
    result = djinni.evaluate_production({"labels": [
        {"jd": "j", "cv": "c", "grade": 2.0, "lower": 2, "upper": 2, "relevant": True},
    ]})

    assert "P6" in result["missing"]
    assert "| P6 | pending |" in djinni.render_production(result)


def test_unseen_holdout_fails_closed_without_human_gold(monkeypatch, tmp_path):
    from app.eval import djinni_holdout

    monkeypatch.setattr(djinni_holdout, "GOLD", tmp_path / "missing-gold.json")
    with pytest.raises(SystemExit, match="holdout is locked"):
        bench_module.load("djinni_holdout")
