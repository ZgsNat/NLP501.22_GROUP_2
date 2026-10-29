from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Active provider; each role uses that provider's model below unless LLM_REASONING / LLM_FAST pin a full spec.
    llm_provider: str = "ollama"
    llm_reasoning: str = ""  # optional "<provider>:<model>" override, e.g. mix providers per role
    llm_fast: str = ""

    openai_api_key: str = ""
    openai_model: str = "gpt-5"
    openai_fast_model: str = "gpt-5-mini"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_fast_model: str = "claude-haiku-5-5"

    google_api_key: str = ""
    google_model: str = "gemini-2.5-pro"
    google_fast_model: str = "gemini-2.5-flash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_fast_model: str = "qwen3:8b"

    gateway_base_url: str = ""
    gateway_api_key: str = ""
    gateway_model: str = "claude-sonnet-5-5"
    gateway_fast_model: str = "claude-haiku-5-5"

    qdrant_url: str = "http://localhost:6333"  # ":memory:" for tests
    embed_dense_model: str = "BAAI/bge-small-en-v1.5"
    embed_sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    langfuse_base_url: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    data_dir: Path = ROOT / "data"
    runtime_dir: Path = ROOT / ".runtime"

    # Agent knobs (the "harness" limits that keep loops bounded and cost predictable)
    search_limit: int = 30
    min_coverage: int = 5  # candidates that must cover all must-haves before we stop expanding
    max_expansions: int = 2
    max_clarifications: int = 1
    max_verify_retries: int = 2
    shortlist_size: int = 8  # how many reranked CVs get the expensive LLM assessment
    near_tie_delta: float = 0.05
    max_arbiter_calls: int = 3
    llm_concurrency: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
