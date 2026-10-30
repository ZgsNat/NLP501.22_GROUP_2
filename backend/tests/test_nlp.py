import io
import unicodedata

import pytest

from app.nlp import chunking, extract, segment
from app.nlp.pipeline import process
from app.nlp.redact import redact
from app.nlp.text import detect_language, normalize, read_file

CV = """# Senior QA

PROFESSIONAL PROFILE
Automation engineer with 6 years in banking.

Work Experience:
Senior Automation Engineer — Bank
Jan 2021 – Dec 2023
• Designed a Selenium + Java framework with Page Object Model.
• Ran tests on Jenkins.
Freelance QA (part-time) | 06/2022 - 12/2022
+ Wrote Cypress tests for a shop.
2019/01–2020/12 (career break)

Kỹ năng
Selenium, Java (basic), familiar with Appium and Docker, but Jenkins daily.
"""


def test_normalize_unicode_bullets_and_wrapped_lines():
    decomposed = unicodedata.normalize("NFD", "Kiểm thử")
    assert normalize(decomposed) == "Kiểm thử"  # NFC: one code point per letter
    assert normalize("• one\n● two\n+ three\n*bold*") == "- one\n- two\n- three\n*bold*"
    assert normalize("built an automa-\ntion suite\nthat runs nightly") == "built an automation suite that runs nightly"


def test_detect_language():
    assert detect_language("Designed a Selenium framework for the payments team.") == "en"
    assert detect_language("Thiết kế framework Selenium cho dự án ngân hàng và viết kịch bản kiểm thử.") == "vi"
    assert detect_language("Senior QA. Kinh nghiệm làm việc with Selenium, Java, TestNG and Jenkins pipelines.") == "mixed"


def test_redact_contacts_and_protected_attributes_but_not_dates():
    raw = ("Họ và tên: Nguyen Van A\nEmail: a.b@example.com | 0912 345 678 | linkedin.com/in/someone\n"
           "Ngày sinh: 01/01/1995\nGender: Male\nMarital status: single\n- Senior QA, Bank (2019/05–2021/08)\n"
           "- Cut a 3,200-case regression from 9 hours to 55 minutes")
    text, counts = redact(raw)
    for leaked in ("Nguyen", "example.com", "0912", "linkedin", "1995", "Male", "single"):
        assert leaked not in text
    assert {"name", "birth_date", "gender", "marital_status", "phone", "url"} <= set(counts)
    assert "2019/05–2021/08" in text and "3,200-case" in text


@pytest.mark.parametrize("line, kind", [
    ("## Experience", "experience"), ("WORK EXPERIENCE", "experience"), ("Employment History:", "experience"),
    ("Kinh nghiệm làm việc", "experience"), ("Summary / Tóm tắt", "summary"), ("**Key Projects**", "projects"),
    ("Chứng chỉ", "certifications"), ("## Hobbies and stuff", "other"),
    ("### Senior QA — Bank (2021–2025)", None), ("Built a Selenium framework", None), ("Skills are important to me", None),
])
def test_heading_kind(line, kind):
    assert segment.heading_kind(line) == kind


@pytest.mark.parametrize("line, months", [
    ("Senior QA — Bank (2021–2025)", 48), ("Jan 2020 – Dec 2020", 12), ("03/2021 - 06/2021", 4),
    ("2019/05–2021/08", 28), ("Tháng 6/2019 – Tháng 2/2021", 21), ("2018 to 2020", 24),
    ("Built 3,200 tests in 9 hours", None), ("2000-2500 test cases", None),
])
def test_date_range(line, months):
    rng = segment.date_range(line)
    assert (rng[1] - rng[0] if rng else None) == months


def test_open_ended_ranges_end_today():
    from datetime import date

    today = date(2026, 1, 15)
    assert segment.date_range("03/2021 - nay", today) == (2021 * 12 + 2, 2026 * 12 + 1)
    assert segment.date_range("Since 2022", today) == (2022 * 12, 2026 * 12 + 1)


def test_merged_months_does_not_double_count_overlaps():
    assert segment.merged_months([(0, 36), (17, 23), (40, 46)]) == 42


def test_pipeline_sections_entries_years_and_gap():
    p = process("CV-T", CV, "Senior QA")
    assert [s.kind for s in p.sections] == ["summary", "experience", "skills"]  # "# Senior QA" is a title, no body
    titles = [e.title for e in p.entries]
    assert titles[0] == "Senior Automation Engineer — Bank" and titles[1].startswith("Freelance QA")
    assert p.years_total == 3.0  # 2021–2023 (36 months) + overlapping freelance; the career break is not a job
    assert all(c.text == p.text[c.start:c.end] for c in p.chunks)  # offsets are exact


def test_hedge_scope_follows_negex():
    p = process("CV-T", CV)
    strength = {k: extract.best(ms).strength for k, ms in p.skills.items()}
    assert strength["selenium"] == "used"  # used in a role beats the skills-list mention
    assert strength["appium"] == strength["docker"] == "hedged"  # "familiar with X and Y" — forward scope
    assert strength["jenkins"] == "used"  # "but" ends the hedge scope; also used in a role
    assert [m.strength for m in p.skills["java"]] == ["used", "hedged"]  # "Java (basic)" — backward scope, own segment


@pytest.mark.parametrize("line, hedged", [
    ("Đã tham gia khóa học Selenium 2 tuần do công ty tổ chức", True),
    ("Đang tự học: Selenium, Java, Jenkins", True),
    ("A colleague demoed Appium to us and I followed a tutorial", True),
    ("Our developers run Cypress tests in the pipeline; I read the results", True),
    ("Cypress: following a beginner course; no hands-on experience yet", True),
    ("Wrote Selenium tests covering basic flows", False),
    ("Designed the Cypress framework for the team", False),
])
def test_hedge_cues(line, hedged):
    import re as _re

    low = line.lower()
    m = _re.search(r"selenium|appium|cypress", low)
    assert extract.is_hedged(low, m.start(), m.end()) == hedged


@pytest.mark.parametrize("bullet, level", [
    ("Defined the test automation strategy for all teams", 4),
    ("Designed a Selenium + Java framework with Page Object Model", 3),
    ("I designed and built the browser test harness from scratch", 3),
    ("Wrote 150 Cypress tests in JavaScript", 2),
    ("Executed the existing automated regression suite on Jenkins", 1),  # "automated" is an adjective here
    ("Our developers run Cypress tests in the pipeline", 0),  # someone else's action
    ("Wrote and executed 120 manual test cases", 0),
    ("Familiar with Selenium; wrote a few scripts in training", 0),  # hedged line
    ("Viết khoảng 320 kịch bản Selenium bằng Java", 2),
    ("Xây dựng framework Appium dùng chung cho Android và iOS", 3),
])
def test_depth_cue(bullet, level):
    p = process("CV-T", f"## Experience\n### QA — Co (2020–2022)\n- {bullet}")
    assert p.depth_cue == level


def test_jd_depth_reads_duties_not_nice_to_haves():
    jd = ("## About the role\nWe define the test strategy for all teams.\n## Responsibilities\n"
          "- Design and evolve the shared automation framework.\n- Write API tests with RestAssured.\n"
          "## Requirements\nNice to have:\n- Defined a test automation strategy across several teams.")
    assert extract.jd_depth(jd) == (3, "Design and evolve the shared automation framework.")
    assert extract.jd_depth("Ưu tiên:\n- Xây dựng framework kiểm thử") == (0, None)


def test_chunking_strategies_cover_the_text():
    p = process("CV-T", CV, "Senior QA")
    exp = [c for c in p.chunks if c.section == "experience"]
    assert len(exp) == 3 and "experience: Senior Automation Engineer — Bank" in exp[0].embed_text
    win = chunking.window("CV-T", p.text, size=20, overlap=5)
    assert len(win) > 1 and win[0].text.split()[-5:] == win[1].text.split()[:5]
    assert chunking.whole("CV-T", p.text)[0].text == p.text


def test_read_docx():
    from docx import Document

    doc = Document()
    doc.add_paragraph("WORK EXPERIENCE")
    doc.add_paragraph("• Built a Playwright framework")
    buf = io.BytesIO()
    doc.save(buf)
    text = normalize(read_file("cv.docx", buf.getvalue()))
    assert text == "WORK EXPERIENCE\n- Built a Playwright framework"
    with pytest.raises(ValueError):
        read_file("cv.png", b"")


def test_jd_parser_reads_sections_cues_and_one_of_groups():
    from app.nlp.jd import parse
    jd = ("# QA Automation Engineer\n## About the role\nA retail bank building mobile banking apps.\n"
          "## Responsibilities\n- Build API test suites with RestAssured and run them in CI.\n"
          "## Requirements\nMust have:\n- 3+ years in test automation.\n- Selenium WebDriver.\n- Java, Python or TypeScript.\n"
          "- CI experience (Jenkins or GitLab CI).\nNice to have:\n- Docker and Kubernetes.\n- ISTQB is a plus.\n"
          "## Benefits\n- Health insurance and a MacBook.")
    r = parse(jd)
    assert r["must_have"] == ["selenium", "java", "jenkins"]
    assert r["substitutes"] == {"java": ["python", "typescript"], "jenkins": ["gitlab_ci"]}
    assert {"docker", "kubernetes", "istqb", "restassured"} <= set(r["nice_to_have"])
    assert r["min_years"] == 3 and r["depth_level"] == 2 and r["test_types"] == ["api"]
    assert r["domains"] == ["banking"]  # "health insurance" sits under Benefits and is ignored


def test_jd_parser_vietnamese():
    from app.nlp.jd import parse
    r = parse("## Yêu cầu\n- Tối thiểu 2 năm kinh nghiệm kiểm thử tự động\n- Thành thạo Selenium và Java\n"
              "- Kiểm thử API (RestAssured hoặc Postman)\nƯu tiên:\n- Biết Docker, tiếng Anh tốt")
    assert r["must_have"] == ["selenium", "java", "restassured"] and r["substitutes"] == {"restassured": ["postman"]}
    assert r["nice_to_have"] == ["docker"] and r["min_years"] == 2 and r["foreign_languages"] == ["english"]


def test_quality_flags_and_duplicates():
    from app.nlp import quality
    short = process("CV-S", "Tester. Selenium.")
    assert {"short_text", "few_skills", "no_dated_roles"} <= set(quality.flags(short))
    a = process("CV-A", CV, "Senior QA")
    b = process("CV-B", CV + "\nHobbies: chess.", "Senior QA")
    assert quality.duplicates([a, b]) == {"CV-B": "CV-A"}
    assert not {"few_skills", "no_dated_roles", "no_experience_section"} & set(quality.flags(a))
