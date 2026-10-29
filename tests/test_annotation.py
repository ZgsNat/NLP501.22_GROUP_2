import json

import pytest

from cv_agent.eval import annotation as an


def test_pool_is_the_union_of_top_k_plus_a_random_sample_and_hides_the_system_order():
    rankings = {"B0": ["a", "b", "c", "d"], "B1": ["b", "e", "a", "f"]}
    universe = list("abcdefghij")
    pooled = an.pool(rankings, universe, depth=2, n_random=3, seed=1)
    assert {"a", "b", "e"} <= set(pooled) and len(pooled) == 6 and len(set(pooled)) == 6
    assert pooled != sorted(pooled, key=lambda c: rankings["B0"].index(c) if c in rankings["B0"] else 99)
    assert an.pool(rankings, universe, depth=2, n_random=3, seed=1) == pooled


def test_pool_random_sample_is_capped_by_what_is_left():
    assert len(an.pool({"B0": ["a", "b"]}, ["a", "b", "c"], depth=2, n_random=5, seed=0)) == 3


def test_weighted_kappa_is_one_for_identical_and_penalises_far_disagreements_more():
    a = {"x": 0, "y": 1, "z": 2, "w": 3}
    assert an.weighted_kappa(a, dict(a)) == pytest.approx(1.0)
    near = an.weighted_kappa(a, {"x": 0, "y": 1, "z": 3, "w": 3})
    far = an.weighted_kappa(a, {"x": 3, "y": 1, "z": 2, "w": 0})
    assert far < near < 1.0


def test_weighted_kappa_is_nan_without_common_items():
    assert an.weighted_kappa({"x": 1}, {"y": 1}) != an.weighted_kappa({"x": 1}, {"y": 1})


def test_sheet_round_trip_reads_back_the_grades(tmp_path):
    items = [("jd1", "cv1"), ("jd1", "cv2"), ("jd2", "cv1")]
    path = tmp_path / "A.xlsx"
    an.write_sheet(path, "A", items, {"jd1": ("QA Automation", "JD one"), "jd2": ("SDET", "JD two")},
                   {"cv1": "Selenium", "cv2": "Cypress"}, seed=0)
    from openpyxl import load_workbook
    wb = load_workbook(path)
    ws = wb[an.LABEL_SHEET]
    for row in ws.iter_rows(min_row=2):
        row[an.GRADE_COL].value = 2 if row[1].value == "jd1" else 0
    wb.save(path)
    labels, missing = an.read_sheet(path)
    assert missing == [] and labels == {("jd1", "cv1"): 2, ("jd1", "cv2"): 2, ("jd2", "cv1"): 0}


def test_read_sheet_reports_blank_and_invalid_grades(tmp_path):
    path = tmp_path / "B.xlsx"
    an.write_sheet(path, "B", [("j", "c1"), ("j", "c2")], {"j": ("t", "x")}, {"c1": "a", "c2": "b"}, seed=0)
    from openpyxl import load_workbook
    wb = load_workbook(path)
    rows = list(wb[an.LABEL_SHEET].iter_rows(min_row=2))
    rows[0][an.GRADE_COL].value = 7
    wb.save(path)
    labels, missing = an.read_sheet(path)
    assert labels == {} and len(missing) == 2


def test_merge_takes_agreement_and_the_lower_grade_for_a_one_step_disagreement():
    a = {("j", "c1"): 3, ("j", "c2"): 2}
    b = {("j", "c1"): 3, ("j", "c2"): 1}
    gold, conflicts = an.merge(a, b, splits={"j": "test"})
    assert conflicts == []
    assert gold["j"]["relevance"] == {"c1": 3, "c2": 1}
    assert gold["j"]["annotator_a"] == {"c1": 3, "c2": 2} and gold["j"]["judged_only"] is True
    assert gold["j"]["hr_ranking"] == ["c1", "c2"]


def test_merge_requires_adjudication_for_far_disagreements():
    a, b = {("j", "c1"): 3}, {("j", "c1"): 0}
    gold, conflicts = an.merge(a, b, splits={"j": "test"})
    assert conflicts == [("j", "c1", 3, 0)] and gold == {}
    gold, conflicts = an.merge(a, b, splits={"j": "test"}, adjudicated={("j", "c1"): 2})
    assert conflicts == [] and gold["j"]["relevance"] == {"c1": 2}


def test_merge_ignores_items_only_one_annotator_labelled():
    gold, _ = an.merge({("j", "c1"): 2, ("j", "c2"): 1}, {("j", "c1"): 2}, splits={"j": "test"})
    assert gold["j"]["relevance"] == {"c1": 2}


def test_agreement_report_gives_kappa_per_jd_and_overall():
    a = {("j1", "c1"): 0, ("j1", "c2"): 3, ("j2", "c1"): 1, ("j2", "c2"): 2}
    rep = an.agreement_report(a, dict(a))
    assert rep["overall"]["weighted_kappa"] == pytest.approx(1.0) and set(rep["per_jd"]) == {"j1", "j2"}
    assert json.loads(json.dumps(rep)) == rep


def test_write_sheet_drops_control_characters_excel_rejects(tmp_path):
    path = tmp_path / "C.xlsx"
    an.write_sheet(path, "C", [("j", "c1")], {"j": ("t", "x\x0b")}, {"c1": "Selenium\x0f, HTTP\x01"}, seed=0)
    from openpyxl import load_workbook
    row = next(load_workbook(path)[an.LABEL_SHEET].iter_rows(min_row=2, values_only=True))
    assert row[4] == "Selenium, HTTP"


def _filled(path, grades):
    from openpyxl import load_workbook
    wb = load_workbook(path)
    for row in wb[an.LABEL_SHEET].iter_rows(min_row=2):
        row[an.GRADE_COL].value = grades[row[2].value]
    wb.save(path)


def _labelling_dir(tmp_path):
    import yaml
    data = tmp_path / "data"
    data.mkdir()
    (data / "jds.yaml").write_text(yaml.safe_dump([{"id": "j", "title": "QA", "text": "x", "split": "test"}]),
                                   encoding="utf-8")
    items = [("j", "c1"), ("j", "c2")]
    for name in "AB":
        an.write_sheet(tmp_path / f"{name}.xlsx", name, items, {"j": ("QA", "x")}, {"c1": "a", "c2": "b"})
    return data


def test_import_labels_cli_writes_gold_and_agreement(tmp_path):
    from cv_agent.cli import main
    data = _labelling_dir(tmp_path)
    _filled(tmp_path / "A.xlsx", {"c1": 3, "c2": 1})
    _filled(tmp_path / "B.xlsx", {"c1": 2, "c2": 1})
    assert main(["import-labels", str(tmp_path / "A.xlsx"), str(tmp_path / "B.xlsx"), "--data", str(data)]) == 0
    gold = json.loads((data / "gold.json").read_text(encoding="utf-8"))
    assert gold["j"]["relevance"] == {"c1": 2, "c2": 1}
    assert "weighted_kappa" in json.loads((data / "agreement.json").read_text(encoding="utf-8"))["overall"]


def test_import_labels_cli_stops_on_far_disagreements(tmp_path):
    from cv_agent.cli import main
    data = _labelling_dir(tmp_path)
    _filled(tmp_path / "A.xlsx", {"c1": 3, "c2": 1})
    _filled(tmp_path / "B.xlsx", {"c1": 0, "c2": 1})
    assert main(["import-labels", str(tmp_path / "A.xlsx"), str(tmp_path / "B.xlsx"), "--data", str(data)]) == 3
    assert (data / "conflicts.csv").exists() and not (data / "gold.json").exists()
