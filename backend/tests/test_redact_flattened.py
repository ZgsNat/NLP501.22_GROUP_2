"""Redaction when a PDF lost its line breaks: labels and the header name sit mid-line (E33 `layout` noise)."""
from app.nlp.redact import redact

FLAT_VI = ("Bùi Thu Thảo Automation Test Engineer Email: thao66@example.com Điện thoại: 0928134815 Giới tính: Nữ "
           "Ngày sinh: 23/09/1993  TÓM TẮT 1 năm kinh nghiệm kiểm thử phần mềm.  KINH NGHIỆM Company E (2025 - nay)")
FLAT_EN = ("Nguyen Van Hung QA Automation Engineer Email: hung79@example.com Phone: 0949827566 Gender: Male "
           "Date of birth: 04/09/2001  SUMMARY 3 years of test automation.  EXPERIENCE Company B (2023 - Present)")


def test_flattened_vietnamese_header_leaks_nothing():
    out, counts = redact(FLAT_VI)
    for value in ("Bùi Thu Thảo", "Nữ", "23/09/1993", "0928134815", "thao66@example.com"):
        assert value not in out
    assert {"name", "gender", "birth_date"} <= set(counts)
    assert "Automation Test Engineer" in out and "TÓM TẮT" in out and "Company E (2025 - nay)" in out


def test_flattened_english_header_leaks_nothing():
    out, counts = redact(FLAT_EN)
    for value in ("Nguyen Van Hung", "Male", "04/09/2001"):
        assert value not in out
    assert "QA Automation Engineer" in out and "SUMMARY 3 years" in out


def test_inline_labels_do_not_eat_ordinary_prose():
    text = "Led the payment project. Project name: Payment Gateway. Tested the address: field validation rules."
    out, _ = redact(text)
    assert "Project name: Payment Gateway" in out  # "name" alone is too ambiguous to redact mid-line


def test_name_merged_with_a_short_title_line():
    out, _ = redact("Lê Đức Phúc QA/QC Engineer\nEmail: phuc31@example.com\n")
    assert "Lê Đức Phúc" not in out and "QA/QC Engineer" in out


def test_capitalised_job_titles_are_not_taken_for_names():
    for title in ("Java Software Engineer", "Full Stack Web Developer", "Middle Manual QA Engineer",
                  "CUSTOMER SERVICE MANAGER Summary", "VETERAN ADVOCATE INTERN Professional",
                  "Quality Architect (Test manager)", "Professional SummaryHighly motivated tester"):
        out, counts = redact(f"{title}\n4 năm kinh nghiệm · English: upper intermediate\n")
        assert title in out and "name" not in counts, title


def test_inline_labels_in_real_cv_layouts_are_redacted():
    out, counts = redact("Personal Details: Full Name: Hassan Aly Marital Status: Married. Gender: Female "
                         "Date of Birth: 23.06.1975 Nationality: French")
    for value in ("Hassan Aly", "Married", "Female", "23.06.1975", "French"):
        assert value not in out
    assert {"name", "marital_status", "gender", "birth_date", "nationality"} <= set(counts)


def test_normal_layout_is_unchanged():
    text = "Bùi Thu Thảo\nAutomation Test Engineer\nGiới tính: Nữ\n\nKINH NGHIỆM\nCompany E (2025 - nay)\n"
    out, _ = redact(text)
    assert out.startswith("[REDACTED:name]\nAutomation Test Engineer\n[REDACTED:gender]\n")
