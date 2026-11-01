"""LLM provider layer: interface + registry + factory.

Pick the active provider in .env with LLM_PROVIDER=openai|anthropic|google|ollama|gateway|claude_code; each provider brings its
own default "reasoning" and "fast" model (also overridable in .env). Nodes only ever call `create_llm(spec)` /
`default_spec(role)` — they never import a vendor SDK.

To add a provider: subclass LLMProvider, decorate with @register. Nothing else changes.
"""

import shutil
from abc import ABC, abstractmethod
from typing import Literal

from langchain_core.language_models import BaseChatModel

from app.config import Settings, get_settings

Role = Literal["reasoning", "fast"]


class LLMProvider(ABC):
    """One vendor/endpoint. A model is addressed as "<provider>:<model>"."""

    name: str

    def __init__(self, s: Settings):
        self.s = s

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def create(self, model: str) -> BaseChatModel: ...

    @abstractmethod
    def default_model(self, role: Role) -> str: ...


PROVIDERS: dict[str, type[LLMProvider]] = {}


def register(cls: type[LLMProvider]) -> type[LLMProvider]:
    PROVIDERS[cls.name] = cls
    return cls


@register
class OpenAIProvider(LLMProvider):
    name = "openai"

    def is_configured(self) -> bool:
        return bool(self.s.openai_api_key)

    def create(self, model: str) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=self.s.openai_api_key, temperature=0)

    def default_model(self, role: Role) -> str:
        return self.s.openai_model if role == "reasoning" else self.s.openai_fast_model


@register
class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def is_configured(self) -> bool:
        return bool(self.s.anthropic_api_key)

    def create(self, model: str) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=self.s.anthropic_api_key, temperature=0, max_tokens=4096)

    def default_model(self, role: Role) -> str:
        return self.s.anthropic_model if role == "reasoning" else self.s.anthropic_fast_model


@register
class GoogleProvider(LLMProvider):
    name = "google"

    def is_configured(self) -> bool:
        return bool(self.s.google_api_key)

    def create(self, model: str) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=self.s.google_api_key, temperature=0)

    def default_model(self, role: Role) -> str:
        return self.s.google_model if role == "reasoning" else self.s.google_fast_model


@register
class OllamaProvider(LLMProvider):
    name = "ollama"

    def is_configured(self) -> bool:
        return bool(self.s.ollama_base_url)

    def create(self, model: str) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        # reasoning=False turns off qwen3-style <think> output: we want structured answers, not monologue.
        # num_predict + repeat_penalty bound small models that otherwise loop ("The candidate has experience with...")
        return ChatOllama(model=model, base_url=self.s.ollama_base_url, temperature=0, reasoning=False,
                          num_ctx=8192, num_predict=2048, repeat_penalty=1.15)

    def default_model(self, role: Role) -> str:
        return self.s.ollama_model if role == "reasoning" else self.s.ollama_fast_model


@register
class GatewayProvider(LLMProvider):
    """Any OpenAI-compatible endpoint (third-party key that proxies Claude / GPT / Gemini / ...)."""

    name = "gateway"

    def is_configured(self) -> bool:
        return bool(self.s.gateway_base_url and self.s.gateway_api_key)

    def create(self, model: str) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, base_url=self.s.gateway_base_url, api_key=self.s.gateway_api_key, temperature=0)

    def default_model(self, role: Role) -> str:
        return self.s.gateway_model if role == "reasoning" else self.s.gateway_fast_model


@register
class ClaudeCodeProvider(LLMProvider):
    """Claude via the local `claude` CLI on the logged-in subscription: no key, dev only (see claude_code.py)."""

    name = "claude_code"

    def is_configured(self) -> bool:
        return shutil.which("claude") is not None

    def create(self, model: str) -> BaseChatModel:
        from app.providers.claude_code import ChatClaudeCode

        return ChatClaudeCode(model=model, effort=self.s.claude_code_effort, workdir=self.s.runtime_dir / "claude_code")

    def default_model(self, role: Role) -> str:
        return self.s.anthropic_model if role == "reasoning" else self.s.anthropic_fast_model


def get_provider(name: str, s: Settings | None = None) -> LLMProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}; available: {list(PROVIDERS)}")
    return PROVIDERS[name](s or get_settings())


def is_configured(name: str, s: Settings | None = None) -> bool:
    return get_provider(name, s).is_configured()


def parse_spec(spec: str) -> tuple[str, str]:
    provider, sep, model = spec.partition(":")
    if not sep or provider not in PROVIDERS or not model:
        raise ValueError(f"Bad model spec {spec!r}; expected '<provider>:<model>' with provider in {list(PROVIDERS)}")
    return provider, model


def default_spec(role: Role, s: Settings | None = None) -> str:
    """LLM_REASONING / LLM_FAST win if set; otherwise LLM_PROVIDER + that provider's default model for the role."""
    s = s or get_settings()
    explicit = s.llm_reasoning if role == "reasoning" else s.llm_fast
    return explicit or f"{s.llm_provider}:{get_provider(s.llm_provider, s).default_model(role)}"


def create_llm(spec: str, s: Settings | None = None) -> BaseChatModel:
    name, model = parse_spec(spec)
    provider = get_provider(name, s)
    if not provider.is_configured():
        raise ValueError(f"Provider {name!r} is not configured — set its key in .env")
    return provider.create(model)
