import json

import pandas as pd
import pytest
import yaml

from cv_agent.data import djinni

JD_BODY = ("We build logistics software. Responsibilities: design and maintain the test automation framework, "
           "write API and UI tests, integrate them into CI. Requirements: 3+ years in QA automation, "
           "{stack}, Git, Jenkins, REST API testing, English upper-intermediate. ") * 4


def jobs(n=12):
    stacks = ["Java and Selenium", "Python and pytest", "TypeScript and Playwright", "C# and .NET NUnit"]
    exps = ["1y", "2y", "3y", "5y"]
    rows = [{"Position": f"QA Automation Engineer {i}", "Long Description": JD_BODY.format(stack=stacks[i % 4]) + f" ref{i}",
             "Company Name": f"Company {i}", "Exp Years": exps[i % 4], "Primary Keyword": "QA Automation",
             "English Level": "upper", "Published": "2023-01-01", "Long Description_lang": "en", "id": i}
            for i in range(n)]
    rows.append({**rows[0], "Primary Keyword": "Java", "id": 100})          # not a QA automation job
    rows.append({**rows[1], "Long Description_lang": "uk", "id": 101})     # not English
    rows.append({**rows[2], "Long Description": "too short", "id": 102})   # below the length filter
    return pd.DataFrame(rows)


def cands():
    rows = []
    for i in range(40):
        kw = "QA Automation" if i < 20 else "QA" if i < 30 else "Java"
        cv = (f"Selenium, Java, TestNG, Jenkins. Built a page object framework for web tests, project {i}."
              if kw != "Java" else f"Spring Boot, unit tests with JUnit, Kafka, microservices, service {i}.") * 3
        rows.append({"Position": f"Engineer {i}", "Moreinfo": "", "Looking For": "remote", "Highlights": "",
                     "Primary Keyword": kw, "English Level": "upper", "Experience Years": 3.0, "CV": cv,
                     "CV_lang": "en", "id": f"c{i}"})
    rows.append({**rows[0], "CV_lang": "uk", "id": "c_uk"})
    return pd.DataFrame(rows)


def test_stack_of_picks_the_dominant_language_family():
    assert djinni.stack_of("Java, Selenium, TestNG; some JavaScript") == "java"
    assert djinni.stack_of("pytest and Python requests") == "python"
    assert djinni.stack_of("Playwright with TypeScript") == "javascript"
    assert djinni.stack_of("NUnit, C#, SpecFlow") == "csharp"
    assert djinni.stack_of("manual testing only") == "other"


def test_select_jds_keeps_english_qa_automation_jobs_one_per_company_and_spreads_stacks():
    sel = djinni.select_jds(jobs(), n=4, seed=1)
    assert len(sel) == 4
    assert set(sel["Primary Keyword"]) == {"QA Automation"} and set(sel["Long Description_lang"]) == {"en"}
    assert sel["Company Name"].is_unique
    assert len({djinni.stack_of(t) for t in sel["Long Description"]}) == 4
    assert not set(sel["id"]) & {100, 101, 102}


def test_select_jds_is_deterministic_for_a_seed():
    a, b = djinni.select_jds(jobs(), n=4, seed=7), djinni.select_jds(jobs(), n=4, seed=7)
    assert list(a["id"]) == list(b["id"])


def test_select_candidates_mixes_automation_manual_and_hard_negatives():
    sel = djinni.select_candidates(cands(), n_auto=5, n_manual=3, n_other=2, seed=3)
    assert list(sel["Primary Keyword"].value_counts().sort_index().items()) == [("Java", 2), ("QA", 3),
                                                                               ("QA Automation", 5)]
    assert "c_uk" not in set(sel["id"])


def test_select_candidates_rejects_a_pool_larger_than_the_data():
    with pytest.raises(ValueError):
        djinni.select_candidates(cands(), n_auto=50, n_manual=1, n_other=1, seed=0)


def test_cv_text_puts_position_experience_and_english_level_first():
    row = cands().iloc[0]
    text = djinni.cv_text(row)
    assert text.splitlines()[0] == "Engineer 0"
    assert "3 năm kinh nghiệm" in text and "English: upper" in text and "Selenium" in text


def test_build_writes_cvs_jds_and_meta_without_gold(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    jobs().to_parquet(raw / djinni.JOB_FILE)
    cands().to_parquet(raw / djinni.CAND_FILE)
    meta = djinni.build(raw, tmp_path / "out", n_jds=4, n_auto=5, n_manual=3, n_other=2, seed=0)
    out = tmp_path / "out"
    assert len(list((out / "cvs").glob("dj_*.txt"))) == 10
    specs = yaml.safe_load((out / "jds.yaml").read_text(encoding="utf-8"))
    assert len(specs) == 4 and all(s["id"].startswith("dj_jd_") and s["text"] for s in specs)
    saved = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(meta)) and saved["license"].startswith("MIT")
    assert saved["synthetic"] is False and not (out / "gold.json").exists()
