import json
import math
from pathlib import Path

import pytest

from cv_agent.training import finetune as ft

QA_JD = "Senior QA Automation Engineer\nWe need Selenium and Python test automation."
DEV_JD = "Backend Developer\nBuild REST services in Go."
DATA_JD = "Data Analyst\nSQL, dashboards, Power BI."

ROWS = [
    ("Resume QA1: Selenium", QA_JD, 2),
    ("Resume QA2: chef", QA_JD, 0),
    ("Resume QA1: Selenium", DEV_JD, 0),   # resume also judged for a QA JD -> must not be trained on
    ("Resume G1: Go, Kafka", DEV_JD, 2),
    ("Resume G2: Go", DEV_JD, 2),
    ("Resume N1: nurse", DEV_JD, 0),
    ("Resume P1: Java", DEV_JD, 1),
    ("Resume D1: SQL", DATA_JD, 2),
    ("Resume D2: painter", DATA_JD, 0),
]


def test_leakage_safe_rows_drop_qa_jds_and_every_resume_judged_for_one():
    safe = ft.leakage_safe_rows(ROWS)
    assert all(jd != QA_JD for _, jd, _ in safe)
    assert all(not r.startswith("Resume QA") for r, _, _ in safe)
    assert len(safe) == 6


def test_split_by_jd_is_disjoint_and_deterministic():
    rows = [(f"r{i}", f"JD number {i % 10}", i % 3) for i in range(60)]
    train, dev = ft.split_by_jd(rows, dev_share=0.3)
    assert {jd for _, jd, _ in train}.isdisjoint({jd for _, jd, _ in dev})
    assert len(train) + len(dev) == len(rows)
    assert len({jd for _, jd, _ in dev}) == 3
    _, dev_reversed = ft.split_by_jd(list(reversed(rows)), dev_share=0.3)
    assert {jd for _, jd, _ in dev_reversed} == {jd for _, jd, _ in dev}  # input order does not matter


def test_split_by_jd_keeps_at_least_one_dev_jd():
    rows = [("r1", "A", 2), ("r2", "B", 0)]
    train, dev = ft.split_by_jd(rows, dev_share=0.01)
    assert len({jd for _, jd, _ in dev}) == 1 and len(train) == 1


def test_triplets_pair_each_good_fit_with_a_no_fit_of_the_same_jd():
    trips = ft.triplets(ft.leakage_safe_rows(ROWS))
    assert ("" not in {t[1] for t in trips})
    by_jd = {}
    for jd, pos, neg in trips:
        by_jd.setdefault(jd, []).append((pos, neg))
    assert sorted(p for p, _ in by_jd[DEV_JD]) == ["Resume G1: Go, Kafka", "Resume G2: Go"]
    assert {n for _, n in by_jd[DEV_JD]} == {"Resume N1: nurse"}
    assert by_jd[DATA_JD] == [("Resume D1: SQL", "Resume D2: painter")]


def test_triplets_skip_jds_without_a_negative():
    assert ft.triplets([("good", "JD", 2), ("maybe", "JD", 1)]) == []


def test_cross_pairs_scale_labels_to_unit_interval():
    pairs = ft.cross_pairs([("a", "JD", 0), ("b", "JD", 1), ("c", "JD", 2)])
    assert pairs == [("JD", "a", 0.0), ("JD", "b", 0.5), ("JD", "c", 1.0)]


def test_ndcg_by_jd_is_one_for_a_perfect_scorer_and_skips_unrankable_jds():
    rows = ft.leakage_safe_rows(ROWS) + [("lonely", "Only one resume", 2)]
    rel = {r: lab for r, _, lab in rows}
    perfect = ft.ndcg_by_jd(lambda jd, resumes: [rel[r] for r in resumes], rows)
    worst = ft.ndcg_by_jd(lambda jd, resumes: [-rel[r] for r in resumes], rows)
    assert perfect["ndcg@10"] == pytest.approx(1.0)
    assert worst["ndcg@10"] < perfect["ndcg@10"]
    assert perfect["n_jds"] == 2


def test_ndcg_by_jd_is_nan_without_rankable_jds():
    assert math.isnan(ft.ndcg_by_jd(lambda jd, r: [0.0] * len(r), [("a", "JD", 2)])["ndcg@10"])


def test_e5_prefixes_only_for_e5_models():
    assert ft.with_prefix(["x"], "query", "intfloat/multilingual-e5-base") == ["query: x"]
    assert ft.with_prefix(["x"], "passage", "models/e5-base-rjf-ft") == ["passage: x"]
    assert ft.with_prefix(["x"], "query", "BAAI/bge-m3") == ["x"]


def test_training_config_rejects_bad_values():
    with pytest.raises(ValueError):
        ft.TrainConfig(target="dense", base_model="m", out_dir=Path("o"), epochs=0)
    with pytest.raises(ValueError):
        ft.TrainConfig(target="sparse", base_model="m", out_dir=Path("o"))


def test_colab_notebook_is_in_sync_with_the_training_sources():
    from scripts.build_colab_notebook import NOTEBOOK, build_notebook
    assert json.loads(NOTEBOOK.read_text(encoding="utf-8")) == build_notebook(), \
        "run: python scripts/build_colab_notebook.py"


def test_experiments_cli_accepts_fine_tuned_model_paths():
    from cv_agent.cli import build_parser, model_overrides
    args = build_parser().parse_args(["experiments", "--neural", "--dense-model", "models/e5-ft",
                                      "--rerank-model", "models/ce-ft"])
    assert model_overrides(args) == {
        "retrieval": {"dense": {"encoder": "sentence_transformers", "model": "models/e5-ft"}},
        "rerank": {"reranker": "cross_encoder", "model": "models/ce-ft"}}
    assert model_overrides(build_parser().parse_args(["experiments"])) == {}


def test_run_writes_metrics_next_to_the_model_without_touching_eval_data(tmp_path, monkeypatch):
    rows = {"train": ft.leakage_safe_rows(ROWS), "dev": [("a", "JD1", 2), ("b", "JD1", 0)],
            "test": [("c", "JD2", 2), ("d", "JD2", 0), ("e", "JD3", 1), ("f", "JD3", 0)]}
    seen = {}

    def fake_train(cfg, train_rows, eval_sets):
        seen.update(train=train_rows, eval=eval_sets)
        return {"n_train_examples": 1, "before": {k: {"ndcg@10": 0.5} for k in eval_sets},
                "after": {k: {"ndcg@10": 0.6} for k in eval_sets}}

    monkeypatch.setattr(ft, "load_rows", lambda raw_dir: rows)
    monkeypatch.setattr(ft, "train_dense", fake_train)
    monkeypatch.setattr(ft, "_device", lambda: "cpu")
    cfg = ft.TrainConfig(target="dense", base_model="intfloat/multilingual-e5-base", out_dir=tmp_path / "m", limit=1)
    report = ft.run(cfg, tmp_path / "raw")
    saved = json.loads((tmp_path / "m" / "finetune_metrics.json").read_text(encoding="utf-8"))
    assert saved["after"]["test"]["ndcg@10"] == 0.6 and saved["data"]["test"] == {"pairs": 4, "jds": 2}
    assert report["config"]["out_dir"] == str(tmp_path / "m")
    assert seen["train"] == rows["train"]
    assert len({jd for _, jd, _ in seen["eval"]["test"]}) == 1  # limit=1 keeps one eval JD


def test_split_files_drops_resumes_judged_for_a_qa_jd_in_the_other_file():
    train_csv = [("Shared resume", DEV_JD, 2), ("Other", DEV_JD, 0), ("D1", DATA_JD, 2), ("D2", DATA_JD, 0)]
    test_csv = [("Shared resume", QA_JD, 2), ("T1", "Nurse\nCare for patients.", 2), ("T2", "Nurse\nCare for patients.", 0)]
    rows = ft.split_files(train_csv, test_csv)
    every = rows["train"] + rows["dev"] + rows["test"]
    assert "Shared resume" not in {r for r, _, _ in every}
    assert all(not ft.is_qa_jd(jd) for _, jd, _ in every)
    assert {jd for _, jd, _ in rows["test"]} == {"Nurse\nCare for patients."}


def test_split_files_keeps_test_jds_unseen_in_training():
    train_csv = [("a", DEV_JD, 2), ("b", DEV_JD, 0), ("c", DATA_JD, 2), ("d", DATA_JD, 0)]
    test_csv = [("e", DEV_JD, 2), ("f", DEV_JD, 0)]
    assert ft.split_files(train_csv, test_csv)["test"] == []


def test_e5_prefix_rule_looks_at_the_model_folder_name_only():
    assert ft.with_prefix(["x"], "query", "runs/e5-experiments/bge-m3-ft") == ["x"]


def test_model_flags_switch_on_the_neural_component_they_replace():
    from cv_agent.cli import build_parser, model_overrides
    args = build_parser().parse_args(["experiments", "--dense-model", "models/e5-ft"])
    assert model_overrides(args) == {"retrieval": {"dense": {"encoder": "sentence_transformers",
                                                             "model": "models/e5-ft"}}}


def test_finetune_both_targets_refuses_a_single_base_model(capsys):
    from cv_agent.cli import main
    assert main(["finetune", "--target", "both", "--base-model", "x/y"]) == 2
