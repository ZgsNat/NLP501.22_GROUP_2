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

    claude_code_effort: str = ""  # claude_code only: --effort for the CLI (low | medium | high | max); "" = CLI default

    # System One decision models (Jev-style: typed answers with calibrated probabilities, no text), served at
    # POST {base}/v1/systemone by TypeSafe (Jev, with a key) or locally by Ollama >= 0.35 (nimble, tev1).
    decider: Literal["llm", "systemone"] = "llm"  # backend of providers/decider.decide (the typed gates)
    system_one_base_url: str = "http://localhost:11434"
    system_one_model: str = "nimble"
    system_one_api_key: str = ""
    # Q16 layer 1b: ask the System One model about the CV sentences the lexicon let through; remove those at or above
    # the threshold. docs/INJECTION_RESULTS.md (nimble, 150 clean CVs): 10/10 paraphrased attacks at every threshold;
    # false alarms 2 / 1 / 0 sentences of 2016 at 0.5 / 0.7 / 0.9 (career-objective lines). Off by default only
    # because it needs a running System One server (a 9.3 GB local model); turn it on where one runs.
    injection_model: bool = False
    injection_threshold: float = 0.9

    gateway_base_url: str = ""
    gateway_api_key: str = ""
    gateway_model: str = "claude-sonnet-5-5"
    gateway_fast_model: str = "claude-haiku-5-5"

    # Woku Shop API (OpenAI-compatible proxy for GPT / Claude / Gemini / DeepSeek)
    woku_base_url: str = "https://llm.wokushop.com/v1"
    woku_api_key: str = ""
    woku_model: str = "claude-sonnet-4-6"
    woku_fast_model: str = "claude-sonnet-4-6"

    qdrant_url: str = "http://localhost:6333"  # ":memory:" for tests
    # Qdrant unreachable: False (default) = a clear error; True = run on an empty in-memory index, so the CV library
    # and parsing work without Docker but search finds nothing (and /api/health says so).
    qdrant_fallback_memory: bool = False
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
    # Second recall channel: the top `search_limit` CVs by rule score (zero tokens) also enter screening. Retrieval
    # alone carried only 20% of a 200-CV pool and lost CVs that belong in the top 10 (docs/BENCHMARK_RESULTS.md).
    rule_recall: bool = True
    min_coverage: int = 5  # candidates that must cover all must-haves before we stop expanding
    max_expansions: int = 2
    max_clarifications: int = 1
    max_verify_retries: int = 2
    shortlist_size: int = 8  # how many reranked CVs get the expensive LLM assessment
    # JD → requirements: llm = LLM extraction canonicalized, plus the parser's depth floor, languages and domains;
    # rules = deterministic parser only (0 tokens); hybrid = parser first, LLM fills gaps. The best mode depends on the
    # model: with qwen3:8b llm won (0.884 / 0.824 nDCG@10); with Claude Sonnet llm fell to 0.842 / 0.795 because the
    # stronger model turns alternatives and preferences into hard must-haves, while hybrid held at 0.881 / 0.792.
    # Hybrid is stable across models, so it is the default (docs/EVAL.md §8 rule; docs/FINAL_REPORT.md §5.2).
    jd_parser: Literal["llm", "rules", "hybrid"] = "hybrid"
    near_tie_delta: float = 0.05
    max_arbiter_calls: int = 6  # LLM calls; each near-tie pair is judged in both orders (2 calls), so 3 pairs
    # v3: the LLM explains and arbitrates; the 8 rule criteria set Score. True = v2 (LLM scores 5 of 8 criteria).
    llm_in_score: bool = False
    # Order of the shortlist: "fused" = mean of normalised cross-encoder and rule score (B3 ⊕), "total" = Σ w·S only
    rank_by: Literal["fused", "total"] = "fused"
    llm_concurrency: int = 4
    max_investigations: int = 4  # L6: knocked-out CVs the investigator may examine per search round
    max_investigation_steps: int = 6  # L6: tool calls per investigated CV
    # L6 may accept a must-have a strong CV describes without naming ("harness on the WebDriver protocol") as implied
    # evidence: skill-specific cue, experience line, not hedged; credit 0.5 and an HR flag. Not adopted (EVAL.md §8):
    # on the benchmark dev split it rescued CV-047 twice, both correct by the labels, but below the top 10, so
    # nDCG@10 was unchanged (0.867 -> 0.867); the rule needs a gain on one source. Off by default.
    implied_evidence: bool = False
    max_reviews: int = 5  # L5: HR re-weight rounds before the ranking is final
    ask_hr_substitutes: bool = True  # L2 proposes related tools and HR approves (False = accept silently, v1)
    tier_strong: float = 0.75  # HR-facing tiers on the weighted total; provisional until calibrated on gold labels
    tier_possible: float = 0.55
    llm_cache: bool = False  # cache LLM responses in .runtime/llm_cache.sqlite (reproducible, cheap evaluation)

    # P7 orchestrator (agent/orchestrator.py, loop L7): an LLM picks the next tool; these bound what it may spend.
    agent_role: Literal["fast", "reasoning"] = "fast"  # who orchestrates; judgement inside tools stays on reasoning
    max_agent_steps: int = 12  # orchestrator decisions per HR turn; then code answers HR with the current ranking
    # Orchestrator tokens (input + output) per HR turn, same fallback. Each decision re-reads the conversation, so a
    # first turn with explain + three arbiter calls used ~110k on claude_code (Sonnet 5.5, its CLI adds ~1.4k a call).
    max_agent_tokens: int = 150_000
    # HR messages per run (HR-in-the-loop, L5 in agent mode). Kept small on purpose: the core result is the first
    # answer (goal → shortlist + evidence); follow-ups are HR feedback, not a general chatbot, and each one re-reads
    # the whole conversation (demo A: 87k tokens for one turn on JD-003 vs 454k for five on JD-002).
    max_hr_turns: int = 3
    max_explain: int = 10  # CVs one explain call may assess (each is one reasoning-model call)
    llm_rank_size: int = 15  # controlled LLM ranking (agent/llm_rank.py): shortlist size it may reorder (2 calls)
    max_llm_ranks: int = 2  # LLM ranking calls per HR turn in P7 (each is two passes)
    max_single_agent_steps: int = 30  # S1 baseline (eval/single_agent.py): one LLM, tools, no harness; its only limit


@lru_cache
def get_settings() -> Settings:
    return Settings()
