import pytest

from cv_agent.ingest.pii import contains_pii, mask_pii
from tests.conftest import SENIOR_CV


def test_masks_all_pii_fields_in_cv():
    res = mask_pii(SENIOR_CV)
    for leaked in ("Nguyen Van Hung", "hung.nv@example.com", "0912 345 678", "12/05/1992", "Male", "Le Loi"):
        assert leaked not in res.text
    assert {"NAME", "EMAIL", "PHONE", "DOB", "GENDER", "ADDRESS"} <= set(res.counts)


def test_keeps_skills_and_employment_dates():
    res = mask_pii(SENIOR_CV)
    assert "Selenium WebDriver" in res.text
    assert "2019 - Present" in res.text
    assert "6 years of experience" in res.text


@pytest.mark.parametrize("phone", ["0912345678", "+84 912 345 678", "84-912-345-678", "0912.345.678", "0387654321"])
def test_vietnamese_phone_formats(phone):
    assert "[PHONE]" in mask_pii(f"Liên hệ: {phone}").text


def test_vietnamese_labels():
    text = "Họ và tên: Lê Thị Mai\nGiới tính: Nữ\nNgày sinh: 01/02/1995\nĐịa chỉ: 5 Đường Láng, Hà Nội"
    out = mask_pii(text).text
    assert "Lê Thị Mai" not in out and "Nữ" not in out and "01/02/1995" not in out and "Đường Láng" not in out


def test_name_masked_everywhere():
    text = "Pham Minh Khoa\nAutomation Tester\nKhoa built the framework."
    assert "Khoa" not in mask_pii(text).text


def test_age_and_profile_url():
    out = mask_pii("28 years old. linkedin.com/in/someone").text
    assert "28 years old" not in out and "linkedin" not in out


def test_contains_pii_guard():
    assert contains_pii("mail me at a@b.com")
    assert not contains_pii("[EMAIL] [PHONE] Selenium")
