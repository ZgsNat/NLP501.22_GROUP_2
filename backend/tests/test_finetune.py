import os
from types import SimpleNamespace

from app.config import get_settings
from app.eval import finetune, token_cost
from app.eval.run import settings_for
from app.rag.retrieve import CVHit, Query


def _chunks(*ids):
    return {i: SimpleNamespace(embed_text=f"text {i}") for i in ids}


def test_jd_query_joins_the_requirement_queries():
    qs = [Query("role", "QA Engineer.", 1.0), Query("must:selenium", "selenium webdriver", 2.0)]
    assert finetune.jd_query(qs) == "QA Engineer. | selenium webdriver"


def test_evidence_is_the_best_passage_per_query_without_repeats():
    hit = CVHit("CV-1", 0.9, {"role": ("c2", 0.5), "must:x": ("c1", 0.8), "nice": ("c2", 0.3)})
    assert finetune.evidence(hit, _chunks("c1", "c2", "c3")) == "text c2\ntext c1"


def test_evidence_falls_back_to_the_first_chunks_when_not_retrieved():
    assert finetune.evidence(None, _chunks("a", "b", "c", "d")) == "text a\ntext b\ntext c"


def test_training_examples_come_from_dev_jds_only(monkeypatch):
    monkeypatch.setattr(finetune, "dev_jds", lambda: {"ours": ["J-dev"]})
    gold = {"J-dev": {"CV-1": 3.0, "CV-2": 0.0}, "J-test": {"CV-1": 2.0}}
    inputs = {j: {c: (f"q {j}", f"e {c}") for c in ("CV-1", "CV-2")} for j in gold}
    p = SimpleNamespace(b=SimpleNamespace(gold=gold), inputs=inputs)
    assert sorted(finetune.examples({"ours": p})) == [("q J-dev", "e CV-1", 1.0), ("q J-dev", "e CV-2", 0.0)]


def test_token_saving_is_a_pooled_ratio_over_common_jds():
    sys_ = {"J1": {"tokens": 100}, "J2": {"tokens": 300}, "J3": {"tokens": 999}}
    ref = {"J1": {"tokens": 400}, "J2": {"tokens": 600}}
    s, common = token_cost.saving(sys_, ref)
    assert common == ["J1", "J2"] and abs(s - 0.6) < 1e-9  # 1 − 400/1000


def test_e_every_overrides_apply_only_inside_its_run():
    base = get_settings().shortlist_size
    with settings_for("E-every") as s:
        assert (s.shortlist_size, s.near_tie_delta, s.max_arbiter_calls) == (1000, 1.0, 8)
    assert get_settings().shortlist_size == base and "SHORTLIST_SIZE" not in os.environ
    with settings_for("P6") as s:
        assert s.shortlist_size == base


def test_export_round_trips_through_the_colab_reader(tmp_path, monkeypatch):
    from scripts.colab_train_ce import read_jsonl

    monkeypatch.setattr(finetune, "dev_jds", lambda: {"ours": ["J-dev"]})
    data = [("q1", "kiểm thử tự động", 1.0), ("q2", "e2", 0.0)]
    rows, meta = read_jsonl(finetune.export(data, tmp_path / "x.jsonl"))
    assert rows == data and meta["dev_jds"] == {"ours": ["J-dev"]}


def test_install_unpacks_a_nested_zip_and_keeps_the_previous_model(tmp_path):
    import zipfile

    archive = tmp_path / "ce-ft-dev.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for f in ("config.json", "train.json", "model.safetensors", "tokenizer_config.json"):
            z.writestr(f"content/ce-ft-dev/{f}", "{}")
    out = tmp_path / "models" / "ce-ft-dev"
    out.mkdir(parents=True)
    (out / "old.txt").write_text("previous")
    finetune.install(archive, out)
    assert sorted(p.name for p in out.iterdir()) == ["config.json", "model.safetensors", "tokenizer_config.json",
                                                    "train.json"]
    assert (out.with_name("ce-ft-dev.prev") / "old.txt").read_text() == "previous"


def test_install_rejects_a_zip_without_a_model(tmp_path):
    import zipfile

    import pytest

    archive = tmp_path / "x.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("notes.txt", "hi")
    with pytest.raises(SystemExit, match="no config.json"):
        finetune.install(archive, tmp_path / "m")


def test_held_out_is_a_stable_half_of_each_pool():
    ours = [f"CV-{i:03d}" for i in range(1, 51)]
    coding = [f"syn_{i:03d}" for i in range(200)]
    for pool in (ours, coding):
        share = sum(map(finetune.held_out, pool)) / len(pool)
        assert 0.35 < share < 0.65
    assert [finetune.held_out(c) for c in ours] == [finetune.held_out(c) for c in ours]  # no randomness


def test_cv_disjoint_training_never_sees_a_held_out_cv(monkeypatch):
    monkeypatch.setattr(finetune, "dev_jds", lambda: {"ours": ["J"]})
    cvs = [f"CV-{i:03d}" for i in range(1, 51)]
    gold = {"J": {c: 1.0 for c in cvs}}
    inputs = {"J": {c: ("q", c) for c in cvs}}
    p = SimpleNamespace(b=SimpleNamespace(gold=gold), inputs=inputs)
    kept = {e for _, e, _ in finetune.examples({"ours": p}, cv_disjoint=True)}
    assert kept and not any(finetune.held_out(c) for c in kept)
    assert len(finetune.examples({"ours": p})) == 50
