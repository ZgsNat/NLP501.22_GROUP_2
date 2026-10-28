import csv
import json

import pytest
import yaml

from cv_agent.data import public
from cv_agent.eval.experiments import evaluate_run, specs_for
from cv_agent.eval.span_metrics import bio_spans, score_sentences
from cv_agent.schema import ScoredCandidate

QA_JD = "Senior QA Automation Engineer\nWe need Selenium and Python test automation."
DEV_JD = "Backend Developer\nBuild REST services in Go."


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["resume_text", "job_description_text", "label"])
        w.writeheader()
        w.writerows(rows)


def _pairs(tmp_path):
    rows = [{"resume_text": "Resume A: Selenium, Java, 5 years", "job_description_text": QA_JD, "label": "Good Fit"},
            {"resume_text": "Resume B: manual testing", "job_description_text": QA_JD, "label": "Potential Fit"},
            {"resume_text": "Resume C: chef", "job_description_text": QA_JD, "label": "No Fit"},
            {"resume_text": "Resume A: Selenium, Java, 5 years", "job_description_text": DEV_JD, "label": "No Fit"},
            {"resume_text": "Resume D", "job_description_text": QA_JD, "label": "unknown"}]
    path = tmp_path / "train.csv"
    _write_csv(path, rows)
    return public.read_pairs([path])


def test_read_pairs_maps_labels_and_skips_unknown(tmp_path):
    pairs = _pairs(tmp_path)
    assert len(pairs) == 4 and {lab for _, _, lab in pairs} == {0, 1, 2}


def test_build_keeps_only_qa_jds_with_judged_only_gold(tmp_path):
    meta = public.build_resume_job_fit(_pairs(tmp_path), tmp_path / "out")
    out = tmp_path / "out"
    gold = json.loads((out / "gold.json").read_text(encoding="utf-8"))
    specs = yaml.safe_load((out / "jds.yaml").read_text(encoding="utf-8"))
    assert meta["n_jds"] == 1 and meta["n_cvs"] == 3 and not meta["synthetic"]
    (jd_id, g), = gold.items()
    assert g["judged_only"] and sorted(g["relevance"].values()) == [0, 1, 2]
    assert g["hr_ranking"][0] == max(g["relevance"], key=g["relevance"].get)
    assert specs[0]["id"] == jd_id and specs[0]["title"] == "Senior QA Automation Engineer"
    assert len(list((out / "cvs").glob("*.txt"))) == 3


def test_loose_single_mention_is_not_a_qa_jd(tmp_path):
    loose = "Senior Business Analyst. Write test cases with the QA lead."
    assert public.qa_signals(loose) == ["test cases"]
    with pytest.raises(ValueError):
        public.build_resume_job_fit([("r", loose, 2)], tmp_path)


def test_rebuild_removes_stale_cvs(tmp_path):
    (tmp_path / "cvs").mkdir()
    (tmp_path / "cvs" / "rjf_old.txt").write_text("x", encoding="utf-8")
    public.build_resume_job_fit([("r", QA_JD, 2)], tmp_path)
    assert not (tmp_path / "cvs" / "rjf_old.txt").exists()


def test_build_without_qa_jd_raises(tmp_path):
    with pytest.raises(ValueError):
        public.build_resume_job_fit([("r", DEV_JD, 1)], tmp_path)


def test_download_reuses_existing_file(tmp_path, monkeypatch):
    cached = tmp_path / "a__b__train.csv"
    cached.write_text("x", encoding="utf-8")
    monkeypatch.setattr(public.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert public.download("a/b", "train.csv", tmp_path) == cached


def test_specs_for_prefers_data_folder(tmp_path):
    (tmp_path / "jds.yaml").write_text(yaml.safe_dump([{"id": "x", "text": "QA"}]), encoding="utf-8")
    assert [s["id"] for s in specs_for(tmp_path)] == ["x"]
    assert "fpt_automation_test" in {s["id"] for s in specs_for(tmp_path / "missing")}


def test_judged_only_evaluates_on_labelled_cvs(jd):
    out = {"ranking": [ScoredCandidate("unlabelled"), ScoredCandidate("good"), ScoredCandidate("bad")],
           "llm_stats": {"calls": 0, "used_tokens": 0, "by_task": {}, "tokens_by_task": {}}, "latency_s": 0}
    gold = {"relevance": {"good": 2, "bad": 0}, "hr_ranking": ["good", "bad"]}
    assert evaluate_run(out, {**gold, "judged_only": True}, {}, {}, jd)["ndcg@10"] == 1.0
    assert evaluate_run(out, gold, {}, {}, jd)["ndcg@10"] < 1.0


def test_bio_spans_char_offsets():
    text, spans = bio_spans(["Use", "Selenium", "WebDriver", "and", "Java"], ["O", "B", "I", "O", "B"])
    assert [text[s:e] for s, e in spans] == ["Selenium WebDriver", "Java"]


def test_score_sentences_exact_overlap_and_taxonomy_recall():
    sent = {"tokens": ["Experience", "with", "Selenium", "WebDriver", "and", "team", "leadership"],
            "tags_skill": ["O", "O", "O", "O", "O", "B", "I"],
            "tags_knowledge": ["O", "O", "B", "I", "O", "O", "O"]}
    res = score_sentences([sent])
    assert res["n_gold_spans"] == 2 and res["n_gold_in_taxonomy"] == 1
    assert res["precision_exact"] == 1.0 and res["recall_exact"] == 0.5 and res["recall_in_taxonomy"] == 1.0


def test_graded_tau_ignores_order_inside_a_grade():
    from cv_agent.eval.metrics import graded_tau
    rel = {"a": 2, "b": 2, "c": 0}
    assert graded_tau(["b", "a", "c"], rel) == 1.0 and graded_tau(["c", "a", "b"], rel) == -1.0


def test_recall_uses_full_ranking_with_judged_only():
    from cv_agent.eval.experiments import ranking_metrics
    gold = {"relevance": {"good": 2, "bad": 0}, "hr_ranking": ["good", "bad"], "judged_only": True}
    m = ranking_metrics([f"u{i}" for i in range(12)] + ["good", "bad"], gold)
    assert m["recall@10"] == 0.0 and m["ndcg@10"] == 1.0 and m["kendall_tau"] == 1.0


def test_skill_and_knowledge_layers_are_not_merged():
    sent = {"tokens": ["Java", "Selenium", "tests"], "tags_skill": ["B", "O", "O"],
            "tags_knowledge": ["O", "I", "I"]}
    assert score_sentences([sent])["n_gold_spans"] == 2  # "Java" and the orphan-started "Selenium tests"


def test_audit_refuses_data_without_names(tmp_path, capsys):
    from cv_agent.cli import main
    (tmp_path / "meta.json").write_text('{"synthetic": false}', encoding="utf-8")
    assert main(["audit", "--data", str(tmp_path)]) == 2
    assert "names" in capsys.readouterr().out
