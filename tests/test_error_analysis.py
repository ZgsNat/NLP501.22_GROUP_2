import json

import pytest

from cv_agent.eval import error_analysis as ea

ROWS = [
    {"config": "B0", "jd_id": "a", "ndcg@10": 0.4, "unsupported_rate": float("nan")},
    {"config": "B3", "jd_id": "a", "ndcg@10": 0.9, "unsupported_rate": float("nan")},
    {"config": "B0", "jd_id": "b", "ndcg@10": 0.6, "unsupported_rate": float("nan")},
    {"config": "B3", "jd_id": "b", "ndcg@10": 0.3, "unsupported_rate": float("nan")},
    {"config": "P6", "jd_id": "b", "ndcg@10": 0.3, "unsupported_rate": 0.5},
]


def test_per_jd_table_keeps_every_config_of_every_jd():
    table = ea.per_jd(ROWS, "ndcg@10")
    assert table == {"a": {"B0": 0.4, "B3": 0.9}, "b": {"B0": 0.6, "B3": 0.3, "P6": 0.3}}


def test_near_duplicates_group_postings_that_differ_only_in_the_city():
    base = "Software Developer {city}. A developer with Ansible playbooks who understands automation testing " \
           "and writes test cases for infrastructure code in a six month contract role."
    texts = {"x": base.format(city="Dallas"), "y": base.format(city="Cleveland"),
             "z": "Senior QA engineer: Selenium, Cucumber BDD, test plans for banking applications."}
    assert ea.near_duplicates(texts, threshold=0.7) == [["x", "y"]]


def test_near_duplicates_return_nothing_for_distinct_texts():
    assert ea.near_duplicates({"a": "one two three four", "b": "five six seven eight"}) == []


def test_group_means_average_each_config_over_the_group():
    means = ea.group_means(ea.per_jd(ROWS, "ndcg@10"), {"g": ["a", "b"]}, ["B0", "B3"])
    assert means["g"]["B0"] == pytest.approx(0.5) and means["g"]["B3"] == pytest.approx(0.6)
    assert means["g"]["n_jds"] == 2


def test_signal_groups_split_jds_by_the_number_of_distinct_qa_signals():
    signals = {"a": ["selenium", "cucumber", "test plan"], "b": ["test cases", "automation testing"]}
    assert ea.signal_groups(signals, strong=3) == {"strong": ["a"], "weak": ["b"]}


def test_dedup_means_count_each_duplicate_group_once():
    table = {"a": {"B3": 0.9}, "b": {"B3": 0.3}, "c": {"B3": 0.3}}
    res = ea.dedup_means(table, [["b", "c"]], ["B3"])
    assert res["n_jds"] == 2 and res["B3"] == pytest.approx(0.6)  # plain mean would be 0.5


def test_analyse_run_writes_a_post_hoc_report(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.csv").write_text("config,jd_id,ndcg@10,unsupported_rate\nB0,a,0.4,nan\nB3,a,0.9,nan\n"
                                     "B0,b,0.6,nan\nB3,b,0.3,nan\nB0,c,0.6,nan\nB3,c,0.3,nan\n", encoding="utf-8")
    body = "Ansible playbooks automation testing test cases " + " ".join(f"duty{i}" for i in range(60))
    ansible = "Software Developer {}. " + body
    specs = [{"id": "a", "title": "QA", "text": "Selenium Cucumber test plans", "qa_signals": ["s1", "s2", "s3"]},
             {"id": "b", "title": "Dev", "text": ansible.format("Dallas"), "qa_signals": ["s1", "s2"]},
             {"id": "c", "title": "Dev", "text": ansible.format("Cleveland"), "qa_signals": ["s1", "s2"]}]
    report = ea.analyse_run(run, specs)
    saved = json.loads((run / "error_analysis.json").read_text(encoding="utf-8"))
    assert saved["post_hoc"] is True and saved == json.loads(json.dumps(report))
    assert saved["near_duplicates"] == [["b", "c"]] and saved["effective_n_jds"] == 2
    assert saved["groups"]["distinct"]["B3"] == pytest.approx(0.9)
    assert saved["groups"]["duplicate_postings"]["B3"] == pytest.approx(0.3)
    assert saved["dedup"]["B3"] == pytest.approx(0.6)
    assert saved["per_jd"]["a"]["title"] == "QA" and saved["per_jd"]["a"]["qa_signals"] == 3


def test_jds_without_text_are_never_duplicates():
    assert ea.near_duplicates({"a": "", "b": "   ", "c": "real text here"}) == []
