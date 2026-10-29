import pytest

from app.config import Settings
from app.providers.llm import PROVIDERS, create_llm, default_spec, get_provider, parse_spec


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)  # ignore the developer's .env


def test_all_providers_registered():
    assert set(PROVIDERS) == {"openai", "anthropic", "google", "ollama", "gateway"}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google", "ollama", "gateway"])
def test_llm_provider_switches_both_roles(provider):
    s = settings(llm_provider=provider)
    p = get_provider(provider, s)
    assert default_spec("reasoning", s) == f"{provider}:{p.default_model('reasoning')}"
    assert default_spec("fast", s) == f"{provider}:{p.default_model('fast')}"


def test_explicit_role_spec_overrides_provider():
    s = settings(llm_provider="anthropic", llm_fast="ollama:qwen3:8b")
    assert default_spec("reasoning", s) == "anthropic:claude-sonnet-5-5"
    assert default_spec("fast", s) == "ollama:qwen3:8b"


def test_unconfigured_provider_fails_fast():
    with pytest.raises(ValueError, match="not configured"):
        create_llm("openai:gpt-5", settings())


def test_configured_provider_builds_model():
    llm = create_llm("anthropic:claude-haiku-5-5", settings(anthropic_api_key="sk-test"))
    assert type(llm).__name__ == "ChatAnthropic"


def test_parse_spec_keeps_colons_in_model_name():
    assert parse_spec("ollama:qwen3:8b") == ("ollama", "qwen3:8b")
    with pytest.raises(ValueError):
        parse_spec("nope:model")
