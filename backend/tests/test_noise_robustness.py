import random

import pytest

from app.eval import noise_robustness as nr

TEXT = "Họ tên: Lê Văn Nam\nAutomation Engineer\nBuilt a Selenium framework in Java with TestNG.\nSkills: Postman, Jenkins\n"


def rng():
    return random.Random("seed")


def test_level_zero_and_original_leave_the_text_unchanged():
    assert nr.perturb(TEXT, "char", 0.0, rng()) == TEXT
    assert nr.perturb(TEXT, "original", 0.5, rng()) == TEXT


@pytest.mark.parametrize("kind", ["char", "ocr", "accents", "layout", "truncate"])
def test_noise_is_reproducible_from_the_seed(kind):
    assert nr.perturb(TEXT, kind, 0.5, rng()) == nr.perturb(TEXT, kind, 0.5, rng())


def test_char_noise_replaces_only_non_space_characters():
    out = nr.perturb(TEXT, "char", 1.0, rng())
    assert len(out) == len(TEXT)
    assert all(c in nr.NOISE_CHARS for c in out if not c.isspace())
    assert [i for i, c in enumerate(out) if c.isspace()] == [i for i, c in enumerate(TEXT) if c.isspace()]


def test_accents_strips_vietnamese_diacritics():
    assert nr.perturb(TEXT, "accents", 1.0, rng()).startswith("Ho ten: Le Van Nam")


def test_layout_and_truncate():
    assert "\n" not in nr.perturb(TEXT, "layout", 1.0, rng())
    assert nr.perturb(TEXT, "truncate", 0.5, rng()) == TEXT[: len(TEXT) // 2]


def test_pdf_round_trip_through_the_app_reader_keeps_vietnamese_text():
    pytest.importorskip("fpdf")
    from app.nlp.text import read_file

    back = read_file("cv.pdf", nr.render_pdf(TEXT))
    assert "Lê Văn Nam" in back and "Selenium framework" in back


def test_noisy_texts_keep_the_ranking_pool_whole():
    latent = [{"cv_id": "a", "low_quality": False, "lang_cv": "en"},
              {"cv_id": "b", "low_quality": True, "lang_cv": "en"},
              {"cv_id": "c", "low_quality": False, "lang_cv": "vi"}]
    raw = {"a": "Java Selenium", "b": "Ja#a", "c": "Kiểm thử"}
    assert set(nr.noisy_texts(latent, raw, "accents", 1.0)) == {"c"}
    pool = nr.noisy_texts(latent, raw, "accents", 1.0, every_cv=True)
    assert pool == {"a": "Java Selenium", "b": "Ja#a", "c": "Kiem thu"}


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        nr.perturb(TEXT, "shuffle", 0.5, rng())
