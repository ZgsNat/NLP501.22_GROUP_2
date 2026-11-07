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

HR also brings goals no JD states: *"5 CVs, Selenium a must, Playwright nice, AI used in automation a big plus."* The
system must turn that into checkable constraints, find out whether the pool can meet them before ranking, ask HR when
it cannot, and say which parts of the goal its shortlist does not meet. Full problem statement:
[`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) §1.

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

Two layers ([`docs/agentic-architecture-workflow.md`](docs/agentic-architecture-workflow.md)):

- **P7, the agent (API default).** An LLM works towards HR's goal by choosing among 17 typed tools
  ([`agent/tools.py`](backend/app/agent/tools.py)): `formalize_goal`, `check_goal`, `define_concept`, `screen`,
  `ask_hr`, `explain`, `compare`, `respond_to_hr` … A code harness (`guard`) checks every call: schema, per-tool
  preconditions, HR authority, budgets. HR's goal becomes a typed spec whose every constraint quotes HR; feasibility is
  counted before ranking; the answer must declare what it could not meet
  ([`agent/goal.py`](backend/app/agent/goal.py), [`agent/orchestrator.py`](backend/app/agent/orchestrator.py)).
- **P6, the fixed graph** below, which P7 runs as its `screen` tool and which stays as the baseline. Rules set every
  score in both; the LLM explains, asks and judges true near-ties.

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
| Agent orchestration | **LangGraph**: P7 orchestrator (LLM ⇄ guard, 17 typed tools, multi-turn with HR) over the P6 StateGraph (12 nodes, 6 bounded loops), interrupts, SQLite checkpoints | [`backend/app/agent/`](backend/app/agent) |
| CV language processing | **Deterministic NLP pipeline**: PDF/DOCX parsing, Unicode normalisation, PII and protected-attribute redaction, EN/VN section and date parsing, structural chunking with contextual headers, gazetteer skill NER with NegEx-style hedge scoping, clause-initial depth cues | [`backend/app/nlp/`](backend/app/nlp), explained in [`docs/NLP_PIPELINE.md`](docs/NLP_PIPELINE.md) |
| Evaluation | NLP501-bench v1 / v1.1 / v2 (B0 → P6 ladder, ablations A1–A4), human labels, agent behaviour suites (E15 HR tasks, E26 complex goals); nDCG, Kendall's τ, bootstrap CIs | [`backend/app/eval/`](backend/app/eval), [`docs/EVAL.md`](docs/EVAL.md) → [`docs/EVAL_RESULTS.md`](docs/EVAL_RESULTS.md) |
| LLM brain | **Provider factory**: OpenAI, Anthropic, Google Gemini, Ollama (local), any OpenAI-compatible gateway, the local Claude Code CLI (`claude_code`, dev) | [`backend/app/providers/llm.py`](backend/app/providers/llm.py) |
| Retrieval | **Qdrant** hybrid search over **CV chunks**, one query per requirement (BM25 sparse + `bge-small-en-v1.5` dense, RRF), max-pooled per CV, then a `ms-marco-MiniLM` cross-encoder on (requirement, evidence passage) pairs. All open-source HF models via fastembed/ONNX | [`backend/app/rag/`](backend/app/rag) |
| API | **FastAPI**, NDJSON streaming of node events | [`backend/app/main.py`](backend/app/main.py), contract in [`docs/API.md`](docs/API.md) |
| Observability | **Langfuse v4** (self-hosted); one session per run, one trace per resume | [`backend/app/observability.py`](backend/app/observability.py) |
| UI | **Vite + React + TypeScript**: live graph, timeline, HR panels, evidence viewer, radar chart | [`frontend/`](frontend) |

> 🗺️ **Interactive architecture map:** open [`docs/architecture.html`](docs/architecture.html) in a browser. Every component has an ID (e.g. `N6` = assess node, `H1` = loop budgets); click any box to see its file, state reads/writes and knobs.

### The six loops of P6
| Loop | Trigger | What happens | Limit |
|---|---|---|---|
| **L1** Clarify | JD has no concrete must-have tools (rule), or a fast model judges ambiguities blocking | Graph **pauses**, HR answers, criteria are re-extracted | `max_clarifications` |
| **L2** Expand | Fewer than `min_coverage` CVs satisfy all must-haves | Propose related skills from the ESCO-lite taxonomy (Playwright → Cypress/Selenium); the graph **pauses for HR to accept or decline**; declined tools are never proposed again | `max_expansions` |
| **L3** Arbiter | Two adjacent top-5 candidates within Δ < 0.05 | One LLM pairwise decision with verified evidence. Otherwise no LLM call at all. Each candidate is judged at most once (no cascades) | `max_arbiter_calls` |
| **L4** Verify | An evidence quote doesn't exist in the CV (ungrounded) | That candidate is re-assessed with feedback; after the budget, ungrounded criteria are capped at 20. Quotes that exist but come from the wrong section or name the wrong thing (*listed, not demonstrated*) cap the criterion at 50 | `max_verify_retries` |
| **L5** Review | HR re-weights criteria with sliders **or in plain words** ("certification matters more") | Deterministic re-score, **no new LLM assessment**; the report is reused if the top 5 are the same people | `max_reviews` |
| **L6** Investigate | A knock-out looks like a parse failure (garbled text) | A tool-using LLM loop reads and searches the CV; code verifies every proposed fact before it counts | `max_investigations` |

### Design principles (harness engineering)
1. **Bounded autonomy.** Every cycle has a limit in [`config.py`](backend/app/config.py).
2. **Cheap before expensive.** BM25/dense (ms) → cross-encoder (ms) → LLM on the shortlist only → arbiter LLM on near-ties only.
3. **LLM output never drives deterministic logic raw.** Requirements are canonicalized onto the taxonomy ([`scoring.canonicalize`](backend/app/agent/scoring.py)), and decisions are typed `Literal` choices ([`decider.py`](backend/app/providers/decider.py), inspired by System-One models such as TypeSafe's Jev).
4. **Verify, don't trust.** Grounding is checked deterministically, at zero tokens.
5. **Rules before models.** For example, "no must-have tools → always ask HR" runs before any LLM gate.
6. **Isolate failures.** One candidate's malformed LLM output flags that candidate for manual review and never crashes the run.
7. **Extract facts, compute scores.** Small LLMs calibrate numbers badly (everyone gets 95) but extract facts well. Years come from **parsed role dates**; tools are credited by **evidence strength** (used in a role > listed > hedged); `istqb` is a rule. The LLM reports the depth level L1–L4 (code turns it into a score) and scores only 4 of 8 criteria.
8. **Structure before semantics.** A CV is parsed into sections and entries before anything else, so the system knows a skills-list claim from demonstrated work, and every CV, ranked or not, carries a status and a reason.
9. **Measure the cost.** A provider-agnostic `UsageCounter` callback records LLM calls and tokens per graph node for every run (`done` event, `GET /runs/{id}`, UI sidebar). This is what any "% tokens saved" claim must be based on.
10. **Explainability by construction.** Every node narrates its decisions (`emit()`), which drives the UI timeline, and Langfuse holds the full trace.

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
- **Release gate:** with the API running, `cd backend && uv run python -m scripts.e2e` drives all 12 JDs through the HTTP API, plays HR (clarify, one re-weight, approve) and prints the metrics table plus pass/fail.
- **Share a run:** `http://localhost:5173/?run=<run_id>` opens a finished or paused run.
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
    nlp/          text.py · redact.py · segment.py · chunking.py · extract.py · taxonomy.py · pipeline.py (CVProfile)
    rag/          corpus.py · store.py (Qdrant hybrid + rerank) · retrieve.py (requirement-wise) · ingest.py
    eval/         metrics.py (nDCG, τ-b, κ, bootstrap) · run.py (systems ladder + report)
    main.py       FastAPI + NDJSON streaming + HR resume
    config.py     all settings & loop limits
  scripts/smoke.py   end-to-end CLI run · scripts/e2e.py release gate over HTTP
  tests/             pytest (NLP pipeline, scoring, metrics, API upload, routers, providers)
data/
  cvs/ (50 synthetic anonymised CVs, 6 Vietnamese/mixed) · jds/ (12 JDs, 1 in Vietnamese) · policies/ (5 hiring policies)
  eval/ (silver relevance labels from 2 independent annotators, 600 pairs)
  criteria_profiles.yaml · skills_taxonomy.yaml (ESCO-lite, 37 skills)
frontend/          Vite + React + TS
docs/              architecture.html (component map) · NLP_PIPELINE.md · EVAL.md · EVAL_RESULTS.md (generated)
                   API.md (contract) · WALKTHROUGH.md · RELEASE_V1.md · RELEASE_V2.md
docker-compose.yml Qdrant + optional Langfuse stack
CLAUDE.md          context for Claude Code
.claude/agents/    agent-engineer (Opus) · fullstack-dev (Sonnet) · git-steward (Haiku)
```

### Sample data
All 50 CVs are **synthetic and anonymised** (no names, ages, genders or contacts). They include deliberate edge cases:
- keyword-stuffed manual testers;
- a near-tie pair of Playwright seniors;
- only 3 Playwright CVs (which triggers L2);
- mobile-only and performance-only specialists;
- a CV of impressive but evidence-free claims.

CV-031–050 (v2) stress the NLP pipeline:
- 6 Vietnamese or mixed-language CVs;
- 6 long CVs over 900 words, with overlapping roles and a career gap;
- hedging traps ("currently learning", "not used in production");
- title inflation ("Automation Architect" with L1 bullets);
- strong CVs written in plain language.

| JD | Profile | Exercises |
|---|---|---|
| JD-001 | Bank | Selenium + Java, ISTQB; the standard path |
| JD-002 | Startup | Playwright + TypeScript; **L2 expansion**, L3 near-ties |
| JD-003 | Outsourcing | Deliberately vague; **L1 HR clarification** |
| JD-004 | Default | Performance & API (JMeter or k6) |
| JD-005–009 | Mixed | Playwright, mobile Appium, junior Cypress, API RestAssured, Robot Framework |
| JD-010 | Bank | **Written in Vietnamese** |
| JD-011 | Default | Test Automation Architect (L4), "Java, TypeScript or Python" |
| JD-012 | Default | Performance (k6 or JMeter) |

`data/eval/` holds the relevance labels: 600 (JD, CV) pairs graded 0–3 by two independent annotators ([`docs/EVAL.md`](docs/EVAL.md)).

---

## 6. Experiments and status

**Read first:**
- [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md): the final report;
- [`NLP501_FinalSlide_v7.html`](NLP501_FinalSlide_v7.html): the deck;
- [`docs/BENCHMARK.md`](docs/BENCHMARK.md): **NLP501-bench v1**, the team's one frozen benchmark. It holds 18 JDs from
  both datasets, 10 of them locked for test, with every input pinned by checksum.

**Official numbers:** `cd backend && uv run python -m app.eval.benchmark report` →
[`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md).

| Test split, nDCG@10 | 10 JDs | ours-silver (6) | coding-rubric (4) |
|---|---|---|---|
| B0 BM25 | 0.695 | 0.800 | 0.536 |
| Structural chunks + cross-encoder | 0.715 | 0.813 | 0.568 |
| Rubric ranker (8-axis rules, before our NLP and retrieval) | 0.810 | 0.773 | **0.866** |
| Our v2 (4 rule criteria ⊕ CE) | 0.748 | 0.801 | 0.669 |
| v3.1, deterministic (B3) | 0.841 | **0.888** | 0.770 |
| v3.1, fixed pipeline (B4) | 0.839 | 0.877 | 0.781 |
| **v3.1, full agent (P6)** | **0.851** | 0.882 | 0.805 |

Re-scored with the 120 blind human pairs (v1.1, [`docs/BENCHMARK_V1_1_RESULTS.md`](docs/BENCHMARK_V1_1_RESULTS.md)),
the order of the systems is the same.

**What the numbers say:**
- **v3.1 is the best version measured:** +0.093 over our v2 (p = 0.009) and +0.041 over the rubric ranker (p = 0.12,
  not yet significant).
- **Rules score, so the agent and its fixed twin rank almost alike** (+0.012, p = 0.064; mostly the one vague JD,
  where the agent asks HR). The agent's value is control:
  - 8 distinct paths over 18 JDs against 1;
  - contradicted LLM claims cut from 2.1% to 0.6%;
  - 100% grounded quotes;
  - an arbiter limited to true near-ties.
- **The benchmark exposed two faults, both fixed on dev and re-reported on test:**
  - a recall funnel sized for 50 CVs (the agent on the coding set went from 0.60 to 0.77);
  - an arbiter acting beyond its mandate.

The other results each have their own generated file:
- AI-output criteria: [`docs/AI_OUTPUT_CRITERIA.md`](docs/AI_OUTPUT_CRITERIA.md)
- extraction and redaction: [`docs/EXTRACTION_RESULTS.md`](docs/EXTRACTION_RESULTS.md)
- fairness: [`docs/ROBUSTNESS_RESULTS.md`](docs/ROBUSTNESS_RESULTS.md)
- real text: [`docs/REALTEXT_RESULTS.md`](docs/REALTEXT_RESULTS.md)
- decision scenarios: [`docs/SCENARIO_RESULTS.md`](docs/SCENARIO_RESULTS.md)
- errors: [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md)

### Beyond the benchmark
- **Human labels.** Hoa and Phúc graded 120 pairs blind, κ 0.965 with each other; against the silver mean κ 0.695 /
  0.661, so silver is the secondary set ([`docs/HUMAN_LABELS.md`](docs/HUMAN_LABELS.md)).
- **Real text with human labels.** Djinni-QA: 8 real JDs × 250 anonymised CVs, 424 judged pairs with two human grades
- **Real text with human labels.** Djinni-QA: 8 real JDs × 250 anonymised CVs, 424 judged pairs with two human grades
  (κ 0.696). BM25 0.422 → 8-axis rules 0.636 → P6 full agent **0.643** nDCG@10 (P@5 0.725, +0.221 over BM25, p < 0.001)
  ([`docs/EXTERNAL_VALIDITY.md`](docs/EXTERNAL_VALIDITY.md), [`docs/DJINNI_PRODUCTION_RESULTS.md`](docs/DJINNI_PRODUCTION_RESULTS.md)).
  ([`docs/EXTERNAL_VALIDITY.md`](docs/EXTERNAL_VALIDITY.md), [`docs/DJINNI_PRODUCTION_RESULTS.md`](docs/DJINNI_PRODUCTION_RESULTS.md)).
  A 6-JD × 40-CV holdout is locked until two blind label files exist ([`docs/DJINNI_HOLDOUT.md`](docs/DJINNI_HOLDOUT.md)).
- **The agent (P7), live on Claude.** HR tasks: 40/42 behaviour checks ([`docs/AGENTIC_RESULTS.md`](docs/AGENTIC_RESULTS.md));
  complex HR goals: 81/83 over three runs per goal ([`docs/GOAL_RESULTS.md`](docs/GOAL_RESULTS.md)).
- **Circularity firewall.** An LLM ordering graded by labels from the same model family is not a ranking result, so
  `rank_shortlist` is off by default ([`docs/CIRCULARITY_AUDIT.md`](docs/CIRCULARITY_AUDIT.md)).
- **Open:** consented real CVs ([`docs/labeling/CONSENT.md`](docs/labeling/CONSENT.md)); the Djinni holdout labels;
  a held-out prompt-injection set.

## 7. Contributing
- Python uses **uv** (never pip); the frontend uses **npm**. Docker Desktop.
- Branch from `main`, open a PR. Conventional Commits (`feat:`, `fix:`, `docs:` …).
- Never commit `.env`, real CVs or personal data.
- New deterministic logic needs a test; new loops need a limit in `Settings`.
