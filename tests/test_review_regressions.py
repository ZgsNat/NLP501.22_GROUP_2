"""Regression tests for defects found in code review."""
import unicodedata

import pytest

from cv_agent.arbiter.arbiter import selective_arbiter
from cv_agent.config import load_config
from cv_agent.criteria.jd_parser import parse_jd
from cv_agent.ingest.extractor import estimate_years
from cv_agent.ingest.pii import contains_pii, mask_pii
from cv_agent.ingest.pipeline import ingest_text, mark_duplicates
from cv_agent.llm.backends import HeuristicBackend, LLMError
from cv_agent.llm.gateway import LLMGateway
from cv_agent.orchestration.graph import build_runtime, run_pipeline
from cv_agent.schema import CV, Claim, ScoredCandidate
from cv_agent.verify.evidence import verify_claims
from cv_agent.verify.nli import LexicalNLI
from tests.conftest import MANUAL_CV, SENIOR_CV

HEUR = {"llm": {"backend": "heuristic"}, "paths": {"cache_dir": None}}


def test_skill_header_is_not_treated_as_name():
    out = mask_pii("Selenium Developer\nJava Python Selenium").text
    assert out.count("Selenium") == 2


def test_common_name_token_does_not_erase_english():
    out = mask_pii("Nguyễn Tuấn Anh\nTiếng Anh: IELTS 7.0, làm việc tại Việt Nam").text
    assert "Nguyễn Tuấn Anh" not in out and "Tiếng Anh" in out and "Việt Nam" in out


def test_empty_label_does_not_swallow_next_line():
    out = mask_pii("Name:\nSkills: Java, Selenium").text
    assert "Skills: Java, Selenium" in out


@pytest.mark.parametrize("rng", ["15/03/2020 — 20/05/2022", "15/03/2020 to 20/05/2022", "15/03/2020 đến 20/05/2022"])
def test_employment_date_ranges_are_kept(rng):
    out = mask_pii(f"Company A ({rng})").text
    assert "[DATE]" not in out and "2020" in out


@pytest.mark.parametrize("phone", ["024 3825 1234", "(+84) 912 345 678", "028.3823.4567"])
def test_more_phone_formats_fully_masked(phone):
    out = mask_pii(f"Tel {phone} end").text
    assert out == "Tel [PHONE] end" and not contains_pii(out)


def test_nfd_vietnamese_is_normalised():
    cv = ingest_text(unicodedata.normalize("NFD", "Kiểm thử tự động với Selenium, 3 năm kinh nghiệm"), "x")
    assert "selenium" in cv.values("tool") and cv.years_experience == 3


def test_year_range_uses_lower_bound():
    jd = parse_jd("Requirements\n- 3-5 years of experience in automation testing", "j")
    assert [r.min_level for r in jd.requirements if r.axis == "experience"] == [3.0]


def test_jd_with_recruiter_contact_does_not_crash():
    jd = parse_jd("Requirements\n- Selenium\n- At least 2 years of experience\nSend CV to hr@company.vn or 0912345678", "j")
    assert not contains_pii(jd.text)
    out = run_pipeline(build_runtime(load_config("P6", overrides=HEUR)), jd,
                       [ingest_text(SENIOR_CV, "a"), ingest_text(MANUAL_CV, "b")])
    assert out["cards"]


class _DownBackend(HeuristicBackend):
    def complete_json(self, *a, **k):
        raise LLMError("ollama unreachable")


def test_llm_outage_falls_back_instead_of_crashing(jd, monkeypatch):
    import cv_agent.orchestration.graph as graph
    monkeypatch.setattr(graph, "build_backend", lambda cfg: _DownBackend())
    cvs = [ingest_text(SENIOR_CV, "a"), ingest_text(SENIOR_CV.replace("ISTQB CTFL", ""), "b")]
    out = run_pipeline(build_runtime(load_config("P6", overrides=HEUR)), jd, cvs)
    assert out["fallback"] is True and out["cards"]


def test_arbiter_skips_knocked_out_and_uses_even_passes(jd, profile):
    cvs = {c: ingest_text(SENIOR_CV, c) for c in ("p", "q", "k1", "k2")}
    ranking = [ScoredCandidate("p", score=0.50), ScoredCandidate("q", score=0.52),
               ScoredCandidate("k1", score=0.0, must_have_ok=False), ScoredCandidate("k2", score=0.0, must_have_ok=False)]
    res = selective_arbiter(LLMGateway(HeuristicBackend()), jd, profile, ranking, cvs, threshold=0.05, passes=3)
    assert res.pairs_checked == [("p", "q")] and len(res.call_ids) == 4


def test_failed_regeneration_flags_instead_of_silently_dropping():
    def regen(*_):
        raise RuntimeError("budget")

    final, stats = verify_claims({"c": [Claim("c", "Uses Playwright", "expert in Playwright")]}, {"c": "Selenium"},
                                 LexicalNLI(), regen)
    assert [c.status for c in final["c"]] == ["no_evidence"] and stats["no_evidence"] == 1


def test_failed_parses_are_not_duplicates():
    out = mark_duplicates([CV("a", "", ""), CV("b", "", "")])
    assert not any(f.startswith("duplicate_of") for c in out for f in c.flags)


def test_years_unaffected_by_masking_of_ranges():
    cv = ingest_text("Company A (15/03/2020 — 20/05/2022)", "y")
    assert estimate_years(cv.masked_text) > 2
