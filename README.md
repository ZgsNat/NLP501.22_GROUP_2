# NLP501.22 · Group 2: Agentic CV–JD Ranking for Automation Test Engineers

> Master's NLP final project (FSB). An **agentic AI system** that matches and ranks Automation Test Engineer CVs against a job description, **explains every decision with evidence quoted from the CV**, and keeps **HR in the loop**.

| Member | Part |
|---|---|
| Hoàng Xuân Sơn | 01: Problem statement, 8-criteria scoring, company profiles |
| Nguyễn Văn Vũ | 02: Literature review (31 papers) |
| Nguyễn Đình Đán | 02: Agentic AI in HR & research gaps G1–G5 |
| Trần Việt Hoa | 03–04: Datasets & metrics |
| Nguyễn Thái Thịnh | 05–06: Architecture & experiments (repo maintainer) |

The literature-review deck is `slide_present_literature_review_group_2.html` (Vietnamese). **Its numbers (86.8 nDCG@10, −78% tokens, 97.6% grounded, τ = 0.81) are hypotheses and targets, not measured results.** This repository exists to measure them.

---

## 1. The problem

Screening Automation Test candidates is a 4-stage funnel (10 000 CVs → 200 → 20 → top 3), and it has three specific difficulties:

- **Tool names are hard lexical entities.** Resume2Vec (2025) found Software Testing is the *one* domain where keyword ATS beats dense embeddings. "Playwright" isn't a synonym of anything, so pure semantic search misses it.
- **Depth matters more than keywords.** A candidate who *runs* Selenium scripts (L1) is not one who *architects* a test platform (L4).
- **Every company weighs criteria differently.** A bank values ISTQB and stability; a mobile startup values Playwright and speed.

### Scoring model
```
Score(CV, JD) = 1[H(c)] × Σ_k  w(c,k) · S(c,k)
```
- `H(c)`: hard knock-out. A must-have tool is missing (and no accepted substitute is present), so the score is 0.
- `w(c,k)`: weights per company type over 8 criteria, summing to 1.0. See [`data/criteria_profiles.yaml`](data/criteria_profiles.yaml).
- `S(c,k)`: 0–100 per criterion, every score backed by **verbatim evidence** from the CV.

The 8 criteria are: tools/frameworks · automation depth (L1–L4) · programming language · test types (web/mobile/API) · CI/CD & infrastructure · years & seniority · ISTQB · domain & English.

---

## 2. Architecture

```
            ┌──────────── L1: ask HR to clarify (interrupt) ───────────┐
            ▼                                                          │
 JD ──▶  ingest ──▶ clarify? ──▶ search ──▶ rerank ──▶ assess ──▶ verify_evidence ──▶ score ──▶ arbitrate ──▶ report ──▶ hr_review ──▶ END
          │ RAG        ▲           │  ▲                  ▲               │                ▲          L3: LLM only                     │
          │ policies   │           ▼  │ L2: expand       └── L4: hallucinated quote ──────┘          on near-ties                     │
          ▼            │       expand_query (taxonomy)                                    └──────── L5: HR re-weights ───────────────┘
   criteria + weights
```

| Layer | Tech | Where |
|---|---|---|
| Agent orchestration | **LangGraph** StateGraph: 11 nodes, 5 bounded loops, interrupts, SQLite checkpoints | [`backend/app/agent/`](backend/app/agent) |
| LLM brain | **Provider factory**: OpenAI, Anthropic, Google Gemini, Ollama (local), any OpenAI-compatible gateway | [`backend/app/providers/llm.py`](backend/app/providers/llm.py) |
| Retrieval | **Qdrant** hybrid search: BM25 sparse + `bge-small-en-v1.5` dense, fused with RRF, then a `ms-marco-MiniLM` cross-encoder. All open-source HF models via fastembed/ONNX | [`backend/app/rag/`](backend/app/rag) |
| API | **FastAPI**, NDJSON streaming of node events | [`backend/app/main.py`](backend/app/main.py), contract in [`docs/API.md`](docs/API.md) |
| Observability | **Langfuse v4** (self-hosted); one session per run, one trace per resume | [`backend/app/observability.py`](backend/app/observability.py) |
| UI | **Vite + React + TypeScript**: live graph, timeline, HR panels, evidence viewer, radar chart | [`frontend/`](frontend) |

### The five loops (the "agentic" part)
| Loop | Trigger | What happens | Limit |
|---|---|---|---|
| **L1** Clarify | JD has no concrete must-have tools (rule), or a fast model judges ambiguities blocking | Graph **pauses**, HR answers, criteria are re-extracted | `max_clarifications` |
| **L2** Expand | Fewer than `min_coverage` CVs satisfy all must-haves | Accept related skills from the ESCO-lite taxonomy (Playwright → Cypress/Selenium), search again | `max_expansions` |
| **L3** Arbiter | Two adjacent top-5 candidates within Δ < 0.05 | One LLM pairwise decision with verified evidence. Otherwise no LLM call at all | top 5 only |
| **L4** Verify | An evidence quote doesn't exist in the CV text | That candidate is re-assessed with feedback; after the budget, unsupported scores are capped | `max_verify_retries` |
| **L5** Review | HR re-weights criteria | Deterministic re-score, **no new LLM assessment** | — |

### Design principles (harness engineering)
1. **Bounded autonomy.** Every cycle has a limit in [`config.py`](backend/app/config.py).
2. **Cheap before expensive.** BM25/dense (ms) → cross-encoder (ms) → LLM on the shortlist only → arbiter LLM on near-ties only.
3. **LLM output never drives deterministic logic raw.** Requirements are canonicalized onto the taxonomy ([`scoring.canonicalize`](backend/app/agent/scoring.py)), and decisions are typed `Literal` choices ([`decider.py`](backend/app/providers/decider.py), inspired by System-One models such as TypeSafe's Jev).
4. **Verify, don't trust.** Grounding is checked deterministically, at zero tokens.
5. **Rules before models.** For example, "no must-have tools → always ask HR" runs before any LLM gate.
6. **Isolate failures.** One candidate's malformed LLM output flags that candidate for manual review and never crashes the run.
7. **Explainability by construction.** Every node narrates its decisions (`emit()`), which drives the UI timeline, and Langfuse holds the full trace.

---

## 3. Quick start

**Prerequisites:** Docker Desktop, [uv](https://docs.astral.sh/uv/), Node ≥ 20. For the local model: [Ollama](https://ollama.com) with `ollama pull qwen3:8b`.

```bash
# 1. config
cp .env.example .env                       # pick LLM_PROVIDER and add that provider's key

# 2. infrastructure
docker compose up -d                       # Qdrant :6333
docker compose --profile observability up -d   # + Langfuse :3000 (login = LF_ADMIN_EMAIL / LF_ADMIN_PASSWORD)

# 3. backend
uv sync
cd backend && uv run python -m app.rag.ingest && cd ..   # index data/ into Qdrant
uv run uvicorn app.main:app --app-dir backend --reload   # API :8000

# 4. frontend
cd frontend && npm install && npm run dev                # UI http://localhost:5173
```
- **No backend?** Open `http://localhost:5173/?mock=1` for a replay of a recorded run.
- **Terminal only?** `cd backend && uv run python -m scripts.smoke JD-002` runs the whole graph and answers HR interrupts automatically.
- **Tests** (no LLM or Docker needed): `uv run pytest`.
- **Stop & free RAM:** `docker compose --profile observability down`.

---

## 4. Switching the LLM provider

One line in `.env` switches the whole agent:

```env
LLM_PROVIDER=anthropic          # openai | anthropic | google | ollama | gateway
ANTHROPIC_API_KEY=sk-ant-...
```

Each provider has a default model per **role**: `reasoning` (assessment, arbiter, report) and `fast` (typed gate decisions). Change them per provider:

```env
OPENAI_MODEL=gpt-5              OPENAI_FAST_MODEL=gpt-5-mini
ANTHROPIC_MODEL=claude-sonnet-5-5   ANTHROPIC_FAST_MODEL=claude-haiku-5-5
GOOGLE_MODEL=gemini-2.5-pro     GOOGLE_FAST_MODEL=gemini-2.5-flash
OLLAMA_MODEL=qwen3:8b           OLLAMA_FAST_MODEL=qwen3:8b
GATEWAY_BASE_URL=https://...    GATEWAY_MODEL=...   # any OpenAI-compatible proxy
```

To **mix providers**, pin a role to a full spec (this wins over `LLM_PROVIDER`):

```env
LLM_REASONING=anthropic:claude-sonnet-5-5
LLM_FAST=ollama:qwen3:8b        # free local gates, paid judgement
```

You can also override models **per run** in the UI, or in the API body: `{"models": {"reasoning": "openai:gpt-5"}}`.

**Adding a provider** means one class in [`providers/llm.py`](backend/app/providers/llm.py):

```python
@register
class MistralProvider(LLMProvider):
    name = "mistral"
    def is_configured(self): return bool(self.s.mistral_api_key)
    def create(self, model): return ChatMistralAI(model=model, api_key=self.s.mistral_api_key)
    def default_model(self, role): return "mistral-large-latest" if role == "reasoning" else "mistral-small-latest"
```

Nodes never import a vendor SDK; they call `create_llm(spec)` / `default_spec(role)` only.

---

## 5. Repository layout

```
backend/
  app/
    agent/        graph.py (read first) · nodes.py · state.py · scoring.py (pure, tested) · schemas.py · prompts.py
    providers/    llm.py (interface + registry + factory) · decider.py (typed System-One decisions)
    rag/          corpus.py · store.py (Qdrant hybrid + rerank) · ingest.py
    main.py       FastAPI + NDJSON streaming + HR resume
    config.py     all settings & loop limits
  scripts/smoke.py   end-to-end CLI run
  tests/             pytest (scoring, routers, providers)
data/
  cvs/ (30 synthetic anonymised CVs) · jds/ (4 JDs) · policies/ (5 hiring policies for RAG)
  criteria_profiles.yaml · skills_taxonomy.yaml (ESCO-lite, 37 skills)
frontend/          Vite + React + TS
docs/              API.md (contract) · WALKTHROUGH.md (guided code reading) · claude-settings.proposed.json
docker-compose.yml Qdrant + optional Langfuse stack
CLAUDE.md          context for Claude Code
.claude/agents/    agent-engineer (Opus) · fullstack-dev (Sonnet) · git-steward (Haiku)
```

### Sample data
All CVs are **synthetic and anonymised** (no names, ages, genders or contacts). They include deliberate edge cases:
- keyword-stuffed manual testers;
- a near-tie pair of Playwright seniors;
- only 3 Playwright CVs (which triggers L2);
- mobile-only and performance-only specialists;
- one CV of impressive but evidence-free claims.

| JD | Profile | Exercises |
|---|---|---|
| JD-001 | Bank | Selenium + Java, ISTQB; the standard path |
| JD-002 | Startup | Playwright + TypeScript; **L2 expansion**, L3 near-ties |
| JD-003 | Outsourcing | Deliberately vague; **L1 HR clarification** |
| JD-004 | Default | Performance & API (JMeter/k6) |

---

## 6. Experiments (plan)

| Config | Description |
|---|---|
| B0 | BM25 only |
| B1 | Dense only |
| B2 | Hybrid (BM25 + dense, RRF) |
| B3 | B2 + criteria profile scoring |
| B4 | Fixed linear pipeline (same retriever, cross-encoder, criteria, no loops) |
| **P6** | Full agentic StateGraph |
| A1 | P6 − `verify_evidence` (faithfulness ablation) |
| A2 | P6 − company profile (generic prompt) |

**Metrics:** nDCG@10, Kendall's τ against HR ranking, faithfulness (% grounded quotes), p50 latency, cost/JD (tokens from Langfuse). Use a group split to avoid leakage, and a blind gold set.

### Status
- ✅ Prototype runs end to end (CLI, HTTP + UI, Langfuse) on local `qwen3:8b`, about 3 min per JD.
- ✅ All five loops observed firing in real runs.
- ⏳ Evaluation harness (gold labels, B0–B4/P6 ladder, ablations): next milestone.
- ⚠️ Known issues: score saturation with the 8B model (most candidates score 0.90–1.00); arbiter can cascade within a near-tie cluster; token usage is not yet recorded for Ollama generations. Details are in [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).

---

## 7. Contributing
- Python uses **uv** (never pip); the frontend uses **npm**. Docker Desktop.
- Branch from `main`, open a PR. Conventional Commits (`feat:`, `fix:`, `docs:` …).
- Never commit `.env`, real CVs or personal data.
- New deterministic logic needs a test; new loops need a limit in `Settings`.
