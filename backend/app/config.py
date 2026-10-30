from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    search_limit: int = 40  # CVs carried from retrieval into screening; the rest get "not retrieved" as reason
    min_coverage: int = 5  # candidates that must cover all must-haves before we stop expanding
    max_expansions: int = 2
    max_clarifications: int = 1
    max_verify_retries: int = 2
    shortlist_size: int = 8  # how many reranked CVs get the expensive LLM assessment
    # JD → requirements: llm = LLM extraction canonicalized, plus the parser's depth floor, languages and domains;
    # rules = deterministic parser only (0 tokens); hybrid = parser first, LLM fills gaps. llm won on both benchmarks
    # (docs/EVAL.md §8: rules −0.006 / −0.005, hybrid −0.009 / −0.019 nDCG@10 vs llm with 8 rule criteria).
    jd_parser: Literal["llm", "rules", "hybrid"] = "llm"
    near_tie_delta: float = 0.05
    max_arbiter_calls: int = 6  # LLM calls; each near-tie pair is judged in both orders (2 calls), so 3 pairs
    # v3: the LLM explains and arbitrates; the 8 rule criteria set Score. True = v2 (LLM scores 5 of 8 criteria).
    llm_in_score: bool = False
    # Order of the shortlist: "fused" = mean of normalised cross-encoder and rule score (B3 ⊕), "total" = Σ w·S only
    rank_by: Literal["fused", "total"] = "fused"
    llm_concurrency: int = 4
    max_investigations: int = 4  # L6: knocked-out CVs the investigator may examine per search round
    max_investigation_steps: int = 6  # L6: tool calls per investigated CV
    max_reviews: int = 5  # L5: HR re-weight rounds before the ranking is final
    ask_hr_substitutes: bool = True  # L2 proposes related tools and HR approves (False = accept silently, v1)
    tier_strong: float = 0.75  # HR-facing tiers on the weighted total; provisional until calibrated on gold labels
    tier_possible: float = 0.55
    llm_cache: bool = False  # cache LLM responses in .runtime/llm_cache.sqlite (reproducible, cheap evaluation)


@lru_cache
def get_settings() -> Settings:
    return Settings()
