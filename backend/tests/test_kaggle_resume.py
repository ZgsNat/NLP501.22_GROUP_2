from app.eval.kaggle_resume import report, automation_tools, token_f1
from app.nlp.pipeline import process


def test_token_f1_measures_how_much_text_the_reader_recovered():
    assert token_f1("Selenium Java tests", "selenium java tests") == 1.0
    assert token_f1("", "anything") == 0.0
    assert 0.0 < token_f1("selenium java", "selenium java jenkins docker") < 1.0


def test_testing_tools_ignore_languages_ci_and_hedged_mentions():
    p = process("KR-1", "## Experience\n### QA — Shop (2020–2022)\n- Wrote Selenium tests in Java on Jenkins.\n"
                        "- Familiar with Cypress.\n")
    assert automation_tools(p) == ["selenium"]  # java, jenkins are not test tools; cypress is only hedged


def test_report_counts_false_positives_outside_it(monkeypatch, tmp_path):
    from app.eval import kaggle_resume

    monkeypatch.setattr(kaggle_resume, "DOC", tmp_path / "r.md")
    base = dict(words=200, f1=0.95, sections=4, has_experience=True, dated=True, skills=3, redacted=1, flags=[])
    rows = [{**base, "id": "1", "category": "INFORMATION-TECHNOLOGY", "tools": ["selenium"],
             "passes": {"selenium + java": True, "playwright / cypress": False}},
            {**base, "id": "2", "category": "CHEF", "tools": ["requests"],
             "passes": {"selenium + java": False, "playwright / cypress": False}},
            {"id": "3", "category": "CHEF", "error": "PdfReadError"}]
    text = (tmp_path / "r.md").read_text(encoding="utf-8") if report(rows) else ""
    assert "PDFs read without error | 2/3" in text
    assert "| all other trades | 1 | 100.0% | 0.0% | 0.0% |" in text
    assert "requests ×1" in text
