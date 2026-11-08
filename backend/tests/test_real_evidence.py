from app.eval import real_evidence as re_


def test_status_valid_when_no_structural_flag():
    assert re_.status([]) == re_.OK
    assert re_.status(["few_skills"]) == re_.OK  # an accountant names no test tool: not a parse defect


def test_status_partial_when_layout_hides_structure():
    assert re_.status(["no_experience_section", "few_skills"]) == f"{re_.PART}: no_experience_section"


def test_status_invalid_text_wins_over_partial_layout():
    assert re_.status(["no_dated_roles", "garbled_text"]) == f"{re_.BAD}: garbled_text"


def test_row_records_derived_facts_only(monkeypatch, tmp_path):
    cv = tmp_path / "cv.txt"
    cv.write_text("QA Engineer\n\nEXPERIENCE\nAutomation Engineer, Acme (Jan 2019 - Dec 2022)\n"
                  "- Built a Selenium framework in Java with TestNG for 300 UI tests.\n"
                  "- Wrote API tests with Postman and RestAssured, run in Jenkins.\n\n"
                  "SKILLS\nSelenium, Java, TestNG, Postman, Jenkins, Git, SQL\n", encoding="utf-8")
    monkeypatch.setattr(re_, "rel", lambda p: p.name)
    r = re_.row("test", cv)
    assert set(r) == set(re_.COLUMNS)
    assert "selenium(used)" in r["tools"] and r["experience"] and r["years"] != ""
    assert "Acme" not in " ".join(str(v) for v in r.values())  # no CV text leaves the machine
