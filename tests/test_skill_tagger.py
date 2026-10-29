import json
import math
from pathlib import Path

import pytest

from cv_agent.eval import span_scoring as ss
from cv_agent.eval.span_metrics import score_sentences
from cv_agent.training import skill_tagger as st

SENT = {"tokens": ["Strong", "Python", "and", "team", "communication", "skills"],
        "tags_skill": ["O", "O", "O", "B", "I", "O"],
        "tags_knowledge": ["O", "B", "O", "O", "O", "O"]}


def test_merge_layers_types_each_span_and_prefers_skill_on_overlap():
    assert st.merge_layers(SENT["tags_skill"], SENT["tags_knowledge"]) == [
        "O", "B-KNOWLEDGE", "O", "B-SKILL", "I-SKILL", "O"]
    # knowledge span overlapping a skill span: the skill wins, and the knowledge remainder restarts with B-
    assert st.merge_layers(["B", "I", "O"], ["O", "B", "I"]) == ["B-SKILL", "I-SKILL", "B-KNOWLEDGE"]


def test_merge_layers_repairs_an_orphan_inside_tag():
    assert st.merge_layers(["O", "I", "I"], ["O", "O", "O"]) == ["O", "B-SKILL", "I-SKILL"]


def test_align_labels_keeps_the_first_subword_of_each_word():
    word_ids = [None, 0, 1, 1, 2, None]
    assert st.align_labels(word_ids, ["O", "B-SKILL", "I-SKILL"]) == [
        -100, st.LABEL2ID["O"], st.LABEL2ID["B-SKILL"], -100, st.LABEL2ID["I-SKILL"], -100]


def test_word_labels_read_the_first_subword_prediction():
    word_ids = [None, 0, 1, 1, 2, None]
    preds = [0, 0, 1, 0, 2, 0]
    assert st.word_labels(word_ids, preds, n_words=3) == ["O", "B-SKILL", "I-SKILL"]


def test_word_labels_pad_words_cut_by_truncation_with_outside():
    assert st.word_labels([None, 0, None], [0, 1, 0], n_words=3) == ["B-SKILL", "O", "O"]


def test_typed_spans_do_not_merge_adjacent_spans_of_different_types():
    tokens = ["Python", "Django", "teamwork"]
    labels = ["B-KNOWLEDGE", "I-SKILL", "B-SKILL"]
    text, spans = st.label_spans(tokens, labels)
    assert sorted(text[s:e] for s, e in spans) == ["Django", "Python", "teamwork"]


def test_score_spans_counts_exact_and_overlap_matches():
    res = ss.score_spans([([(0, 5), (10, 12)], [(0, 5), (9, 12), (20, 25)])])
    assert res["precision_exact"] == pytest.approx(0.5) and res["recall_exact"] == pytest.approx(1 / 3)
    assert res["precision_overlap"] == pytest.approx(1.0) and res["recall_overlap"] == pytest.approx(2 / 3)
    assert math.isnan(ss.score_spans([([], [])])["precision_exact"])


def test_score_sentences_accepts_any_span_predictor():
    gold_perfect = lambda text, tokens, sent: ss.bio_spans(tokens, sent["tags_skill"])[1] + \
        ss.bio_spans(tokens, sent["tags_knowledge"])[1]  # noqa: E731
    res = score_sentences([SENT], predict=gold_perfect)
    assert res["f1_exact"] == pytest.approx(1.0)


def test_span_f1_from_label_ids_ignores_padding():
    gold = [[-100, 0, 1, 2, -100]]
    assert st.span_f1_from_ids([[0, 0, 1, 2, 0]], gold) == pytest.approx(1.0)
    assert st.span_f1_from_ids([[0, 0, 0, 0, 0]], gold) == 0.0


def test_tagger_config_validates():
    with pytest.raises(ValueError):
        st.TaggerConfig(base_model="m", out_dir=Path("o"), epochs=0)
    with pytest.raises(ValueError):
        st.TaggerConfig(base_model="m", out_dir=Path("o"), max_len=8)


def test_hybrid_predictor_unions_model_and_taxonomy_spans_without_duplicates():
    model = lambda text, tokens, sent: [(0, 6), (10, 14)]  # noqa: E731
    taxonomy = lambda text, tokens, sent: [(0, 6), (20, 28)]  # noqa: E731
    assert st.union_predictor(model, taxonomy)("", [], {}) == [(0, 6), (10, 14), (20, 28)]


def test_run_writes_metrics_and_keeps_test_out_of_model_selection(tmp_path, monkeypatch):
    splits = {"train": [SENT] * 3, "dev": [SENT], "test": [SENT, SENT]}
    seen = {}

    def fake_fit(cfg, train, dev):
        seen.update(train=len(train), dev=len(dev))
        return object(), {"best_dev_span_f1": 0.5, "epochs_run": 1}

    monkeypatch.setattr(st, "load_splits", lambda raw: splits)
    monkeypatch.setattr(st, "fit", fake_fit)
    monkeypatch.setattr(st, "model_predictor", lambda model, cfg: lambda text, tokens, sent: [])
    monkeypatch.setattr(st, "_device", lambda: "cpu")
    report = st.run(st.TaggerConfig(base_model="b/m", out_dir=tmp_path / "t"), tmp_path)
    saved = json.loads((tmp_path / "t" / "tagger_metrics.json").read_text(encoding="utf-8"))
    assert seen == {"train": 3, "dev": 1}
    assert saved["data"] == {"train": 3, "dev": 1, "test": 2}
    assert set(saved["test"]) >= {"tagger", "taxonomy", "hybrid"} and report["fit"]["best_dev_span_f1"] == 0.5


def test_train_tagger_cli_names_the_output_after_the_base_model(tmp_path, monkeypatch, capsys):
    from cv_agent.cli import main
    seen = {}

    def fake_run(cfg, raw):
        seen["cfg"] = cfg
        res = {"precision_exact": 0.5, "recall_exact": 0.4, "f1_exact": 0.44, "f1_overlap": 0.6}
        return {"test": {"tagger": res}}

    monkeypatch.setattr(st, "run", fake_run)
    assert main(["train-tagger", "--out-dir", str(tmp_path), "--epochs", "1", "--limit", "5"]) == 0
    assert seen["cfg"].out_dir.name == "jobbert-base-cased-skillspan-tagger" and seen["cfg"].limit == 5
    assert "tagger: test P/R/F1 exact 0.500" in capsys.readouterr().out


def test_warmup_argument_follows_the_installed_transformers_api():
    class Old:
        def __init__(self, warmup_ratio=0.0, warmup_steps=0):
            pass

    class New:
        def __init__(self, warmup_steps=0):
            pass

    assert st._warmup(Old, 0.1) == {"warmup_ratio": 0.1}
    assert st._warmup(New, 0.1) == {"warmup_steps": 0.1}


def test_hybrid_flag_needs_a_tagger(capsys):
    from cv_agent.cli import main
    assert main(["eval-spans", "--hybrid"]) == 2
