"""EN/VI: the UI language reaches every prompt whose output HR reads; English prompts are unchanged."""
import pytest
from pydantic import ValidationError

from app.agent import prompts
from app.main import RunRequest


def test_english_prompts_are_unchanged():
    for p in (prompts.ASSESS, prompts.REPORT, prompts.ARBITER, prompts.AGENT, prompts.LLM_RANK, prompts.INGEST):
        assert prompts.localized(p, "en") == p
        assert prompts.localized(p, None) == p


def test_vietnamese_adds_the_language_rule_and_keeps_quotes_verbatim():
    out = prompts.localized(prompts.ASSESS, "vi")
    assert out.startswith(prompts.ASSESS) and out.endswith(prompts.VIETNAMESE)
    assert "Vietnamese" in out and "verbatim" in prompts.VIETNAMESE  # evidence grounding needs the CV's own words


def test_run_request_defaults_to_english_and_rejects_other_languages():
    assert RunRequest(jd_text="QA").lang == "en"
    assert RunRequest(jd_text="QA", lang="vi").lang == "vi"
    with pytest.raises(ValidationError):
        RunRequest(jd_text="QA", lang="fr")


def test_translation_keeps_count_numbers_and_tools():
    from app.agent.checks import translation_issues

    src = ["How many years of Playwright are required?", "Is 5+ years of Java needed?"]
    good = ["Yêu cầu bao nhiêu năm Playwright?", "Có cần 5+ năm Java không?"]
    assert translation_issues(src, good) == []
    assert translation_issues(src, good[:1]) == ["1 items for 2"]
    assert translation_issues(src, [good[0], "Có cần 3+ năm Java không?"])[0].startswith("numbers changed")
    assert translation_issues(src, ["Yêu cầu bao nhiêu năm Cypress?", good[1]])[0].startswith("tools changed")
    assert translation_issues(src, ["", good[1]])[0].startswith("empty translation")


class _Fake:
    """A model whose structured output is fixed, to drive nodes.for_hr without an LLM."""
    def __init__(self, items=None, error=None):
        self.items, self.error = items, error

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, msgs):
        if self.error:
            raise self.error
        from app.agent.schemas import HrTexts
        return HrTexts(items=self.items)


async def _for_hr(monkeypatch, lang, fake, texts):
    from app.agent import nodes

    monkeypatch.setattr(nodes, "llm", lambda state, role: fake)
    monkeypatch.setattr(nodes, "emit", lambda *a, **k: None)
    return await nodes.for_hr({"lang": lang}, texts)


async def test_for_hr_translates_only_in_vietnamese(monkeypatch):
    src = ["Is Playwright a must-have?"]
    vi = ["Playwright có phải must-have (bắt buộc) không?"]
    assert await _for_hr(monkeypatch, "en", _Fake(vi), src) == src
    assert await _for_hr(monkeypatch, "vi", _Fake(vi), src) == vi
    assert await _for_hr(monkeypatch, "vi", _Fake(vi), []) == []


async def test_for_hr_keeps_english_when_the_translation_fails(monkeypatch):
    src = ["Is Playwright a must-have?"]
    assert await _for_hr(monkeypatch, "vi", _Fake(["Cypress có bắt buộc không?"]), src) == src  # tool changed
    assert await _for_hr(monkeypatch, "vi", _Fake(error=RuntimeError("down")), src) == src


def test_requirement_extraction_is_never_localized():
    import inspect

    from app.agent import nodes

    src = inspect.getsource(nodes.extract_requirements)
    assert "prompts.localized(" not in src and "lang" not in inspect.signature(nodes.extract_requirements).parameters
