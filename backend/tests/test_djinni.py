import zipfile

import pytest

from app.eval import djinni
from app.eval.run import score_system

WORKBOOK = ('<workbook xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            '<sheet name="H&#432;&#7899;ng d&#7851;n" sheetId="1" r:id="rId1"/>'
            '<sheet name="Nhãn" sheetId="3" r:id="rId3"/></sheets></workbook>')
RELS = ('<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId3" Target="worksheets/sheet3.xml"/></Relationships>')


def _xlsx(path, rows_xml: str, shared: list[str] | None = None) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", WORKBOOK)
        z.writestr("xl/_rels/workbook.xml.rels", RELS)
        z.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData/></worksheet>")
        z.writestr("xl/worksheets/sheet3.xml", f"<worksheet><sheetData>{rows_xml}</sheetData></worksheet>")
        if shared is not None:
            z.writestr("xl/sharedStrings.xml", "<sst>" + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")


def _inline(ref: str, text: str) -> str:
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


HEADER = ('<row r="1">' + _inline("A1", "STT") + _inline("B1", "jd_id") + _inline("C1", "cv_id")
          + _inline("F1", "Nhãn (0–3)") + _inline("G1", "Ghi chú") + "</row>")


def test_read_xlsx_inline_strings_as_generated(tmp_path):
    p = tmp_path / "labels_A.xlsx"
    _xlsx(p, HEADER + '<row r="2"><c r="A2"><v>1</v></c>' + _inline("B2", "J1") + _inline("C2", "C1")
          + '<c r="F2"><v>2</v></c>' + _inline("G2", "used it in a role") + "</row>"
          + '<row r="3"><c r="A3"><v>2</v></c>' + _inline("B3", "J1") + _inline("C3", "C2") + "</row>")
    assert djinni.labels_from_xlsx(p) == [{"jd": "J1", "cv": "C1", "grade": 2, "note": "used it in a role"}]


def test_read_xlsx_shared_strings_as_excel_saves_them(tmp_path):
    p = tmp_path / "labels_B.xlsx"
    shared = ["STT", "jd_id", "cv_id", "Nhãn (0–3)", "J1", "C9"]
    row1 = '<row r="1">' + "".join(f'<c r="{c}1" t="s"><v>{i}</v></c>' for c, i in zip("ABCF", range(4))) + "</row>"
    row2 = '<row r="2"><c r="A2"><v>1</v></c><c r="B2" t="s"><v>4</v></c><c r="C2" t="s"><v>5</v></c>' \
           '<c r="F2"><v>3</v></c></row>'
    _xlsx(p, row1 + row2, shared)
    assert djinni.labels_from_xlsx(p) == [{"jd": "J1", "cv": "C9", "grade": 3, "note": ""}]


def test_a_grade_outside_0_to_3_is_rejected(tmp_path):
    p = tmp_path / "labels_A.xlsx"
    _xlsx(p, HEADER + '<row r="2">' + _inline("B2", "J1") + _inline("C2", "C1") + '<c r="F2"><v>5</v></c></row>')
    with pytest.raises(ValueError, match="not 0–3"):
        djinni.labels_from_xlsx(p)


def test_adjudicate_keeps_equal_takes_lower_and_flags_far_pairs():
    a = {("J", "x"): 2, ("J", "y"): 3, ("J", "z"): 0}
    b = {("J", "x"): 2, ("J", "y"): 2, ("J", "z"): 3}
    g, conflicts = djinni.adjudicate(a, b)
    assert g == {("J", "x"): 2, ("J", "y"): 2} and conflicts == [("J", "z")]
    g, conflicts = djinni.adjudicate(a, b, {("J", "z"): 1})
    assert g[("J", "z")] == 1 and conflicts == []


def test_kappa_rows_compare_every_pair_of_annotators():
    x = {("J", str(i)): i % 4 for i in range(8)}
    rows = djinni.kappa_rows({"h1": x, "h2": dict(x)})
    assert rows == ["| h1 vs h2 | 8 | 100.0% | 1.000 |"]


def test_scoring_is_condensed_to_judged_cvs():
    # u1/u2 were never judged: a pooled benchmark must not count them as irrelevant
    per_jd = {"J": ["u1", "u2", "a", "b"]}
    assert score_system(per_jd, {"J": {"a": 3.0, "b": 0.0}}, frozenset())["ndcg@10"] == [1.0]


def test_page_labels_keep_whether_the_model_grade_was_seen(tmp_path):
    import json

    p = tmp_path / "human_djinni_dan.json"
    p.write_text(json.dumps({"annotator": "human Dan", "labels": [
        {"jd": "J", "cv": "a", "grade": 2, "note": "API tests in a role", "saw_model": True},
        {"jd": "J", "cv": "b", "grade": 0}]}), encoding="utf-8")
    who, labels = djinni.labels_from_page(p)
    assert who == "human Dan" and [x["saw_model"] for x in labels] == [True, False]


def test_page_rejects_a_grade_outside_0_to_3(tmp_path):
    import json

    p = tmp_path / "x.json"
    p.write_text(json.dumps({"annotator": "human X", "labels": [{"jd": "J", "cv": "a", "grade": 4}]}))
    with pytest.raises(ValueError, match="not 0–3"):
        djinni.labels_from_page(p)


def test_review_page_embeds_every_pair_and_a_cv_cannot_close_the_script(tmp_path, monkeypatch):
    from app.rag.corpus import Doc

    monkeypatch.setattr(djinni, "pairs", lambda: {"J1": ["c1", "c2"]})
    monkeypatch.setattr(djinni, "jds", lambda: [Doc("J1", "QA Automation JD", {"title": "QA"})])
    monkeypatch.setattr(djinni, "cvs", lambda: [Doc("c1", "Selenium </script><b>x", {}), Doc("c2", "Cypress", {}),
                                               Doc("c9", "not pooled", {})])
    monkeypatch.setattr(djinni, "MODEL_GOLD", tmp_path / "none.json")
    html = djinni.review(tmp_path / "review.html").read_text(encoding="utf-8")
    assert "__DATA__" not in html and html.count("</script>") == 1  # only the page's own closing tag
    assert '"c1"' in html and '"c2"' in html and "not pooled" not in html


def test_slug_is_ascii_for_vietnamese_names():
    assert djinni.slug("Đán") == "dan" and djinni.slug("Trần Việt Hoa") == "tran_viet_hoa"


def test_human_gold_uses_one_complete_annotator_and_says_how(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(djinni, "pairs", lambda: {"J": ["a", "b"]})
    monkeypatch.setattr(djinni, "LABELS", tmp_path)
    (tmp_path / "djinni_silver_a.json").write_text(json.dumps({"annotator": "s", "labels": [
        {"jd": "J", "cv": "a", "grade": 3}, {"jd": "J", "cv": "b", "grade": 0}]}))
    assert djinni.human_gold() is None  # silver only
    (tmp_path / "human_djinni_dan.json").write_text(json.dumps({"annotator": "human dan", "labels": [
        {"jd": "J", "cv": "a", "grade": 2, "saw_model": True}, {"jd": "J", "cv": "b", "grade": 1, "saw_model": True}]}))
    g, source = djinni.human_gold()
    assert g == {"J": {"a": 2.0, "b": 1.0}} and "1 annotator" in source and "2/2" in source
    assert djinni.gold()[1].startswith("silver")  # one person is not the frozen gold: that needs two, adjudicated


def _stamped(grades, seconds, saw_model=False):
    from datetime import datetime, timedelta

    t0 = datetime(2026, 11, 5)
    return [{"jd": "J", "cv": f"c{i}", "grade": g, "saw_model": saw_model,
             "at": (t0 + timedelta(seconds=seconds * i)).isoformat() + "Z"} for i, g in enumerate(grades)]


def test_qc_rejects_a_blind_annotator_who_agrees_with_nobody_and_grades_in_seconds():
    truth = [0, 1, 2, 3] * 10
    other = {"silver": {("J", f"c{i}"): g for i, g in enumerate(truth)}}
    noise = [3, 2, 1, 0] * 10  # the reverse of everyone else
    problems = djinni.qc_problems(_stamped(noise, 4), other)
    assert any("κ" in p for p in problems) and any("s per pair" in p for p in problems)
    assert djinni.qc_problems(_stamped(truth, 30), other) == []


def test_qc_speed_floor_applies_to_blind_labels_only():
    truth = [0, 1, 2, 3] * 10
    other = {"silver": {("J", f"c{i}"): g for i, g in enumerate(truth)}}
    assert djinni.qc_problems(_stamped(truth, 1, saw_model=True), other) == []  # accepting a shown grade is fast
    assert djinni.median_seconds([{"jd": "J", "cv": "a", "grade": 1}]) is None  # no timestamps: no speed check


def test_gold_ignores_a_second_annotator_who_fails_qc(tmp_path, monkeypatch):
    import json

    cvs = [f"c{i}" for i in range(40)]
    truth = [0, 1, 2, 3] * 10
    monkeypatch.setattr(djinni, "pairs", lambda: {"J": cvs})
    monkeypatch.setattr(djinni, "LABELS", tmp_path)
    monkeypatch.setattr(djinni, "MODEL_GOLD", tmp_path / "none.json")
    (tmp_path / "djinni_silver_a.json").write_text(json.dumps({"annotator": "s", "labels": _stamped(truth, 0)}))
    (tmp_path / "human_djinni_a.json").write_text(json.dumps({"annotator": "human a", "labels": _stamped(truth, 30)}))
    (tmp_path / "human_djinni_b.json").write_text(json.dumps({"annotator": "human b",
                                                              "labels": _stamped([3, 2, 1, 0] * 10, 4)}))
    assert djinni.qualified_humans() == [tmp_path / "human_djinni_a.json"]
    assert djinni.gold()[1].startswith("silver")  # one usable human is not adjudicated gold
    assert "1 annotator, blind" in djinni.human_gold()[1]
