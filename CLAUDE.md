# NLP501 Group 2: Agentic CV–JD Ranking for Automation Test roles

Master's NLP final project (FSB). Topic: **an agentic AI system that matches and ranks Automation Test Engineer CVs against job descriptions, with explanations.** Five members: Sơn (problem & criteria), Vũ (literature, 31 papers), Đán (agentic in HR & gaps G1–G5), Hoa (datasets & metrics), Thịnh (architecture & experiments; maintains this repo). Write code and docs for the whole team, not only the agent part.

`slide_present_literature_review_group_2.html` (Vietnamese) is the literature-review deck. **Its numbers (86.8 nDCG@10, −78% tokens, 97.6% grounded, τ=0.81) are targets/hypotheses, not measured results.** Never cite them as results. Producing real measurements is the goal of this repo.

## Core concepts
- Scoring: `Score(CV,JD) = 1[H(c)] × Σ_k w(c,k)·S(c,k)`. H = hard knock-out on must-have tools; w = per-company-type weights over 8 criteria (`data/criteria_profiles.yaml`); S = 0–100 per criterion. Since v3 all 8 S come from rules over the parsed CV (`scoring.rule_criteria`); the LLM explains (`llm_review`) and arbitrates near-ties in both orders, it does not set S (`LLM_IN_SCORE`).
- Automation depth L1–L4 (run → write → framework → test architecture). See `data/policies/POL-depth-scale.md`.
- Hybrid retrieval is mandatory: BM25 catches exact tool names (Resume2Vec: testing is the one domain where keyword ATS beats dense), dense catches meaning, then a cross-encoder reranks.
- Experiment ladder: B0 BM25 → B1 dense → B2 hybrid → B3 +criteria → B4 fixed linear pipeline → **P6 full agent**. Ablations: A1 = P6 − verify_evidence, A2 = P6 − company profile, A3 = P6 − arbiter, A4 = P6 − investigator. Metrics: nDCG@10 (official: NLP501-bench v1 test split), P@5, R@20, MRR, Kendall's τ, the AI-output scorecard (Q1–Q13), LLM calls/tokens.

## Architecture
- `backend/app/agent/graph.py`: LangGraph StateGraph. **Read it first.** 12 nodes, 6 bounded loops: L1 HR clarification (interrupt), L2 query expansion (taxonomy; HR approves substitutes via interrupt), L3 selective arbiter (LLM only on near-ties, |Δ| < near_tie_delta, both orders), L4 verify_evidence (quotes grounded and supported, claims consistent with the CV), L5 HR review/re-weight (interrupt), L6 investigate (N12, `agent/investigate.py`: tool-using LLM loop for knock-outs that look like parse failures; code verifies every fact). Search screens by retrieval and by rule score (`rule_recall`). Every CV ends with a status + reason.
- `backend/app/agent/orchestrator.py` (v4, **P7**, API mode `agent`, the default): the agentic version. An LLM picks the next tool from `agent/tools.py` (14 typed tools wrapping the nodes below: `screen` = the P6 pipeline up to the ranking at 0 tokens, `rank_shortlist` (controlled LLM ranking, `agent/llm_rank.py`), `explain`, `investigate`, `compare`, `ask_hr`, `respond_to_hr`, ...); `guard` is the harness (schema, per-tool preconditions, HR authority, Q14/Q15, cache, budgets `max_agent_steps`/`max_agent_tokens`/`max_hr_turns`); `hr_turn` makes it a multi-turn conversation with HR. `graph.py` stays as the fixed P6 baseline. `claude_code` emulates tool calling via `--json-schema`.
- `backend/app/agent/nodes.py`: one job per node; narrate decisions with `emit()`. `scoring.py`: pure, deterministic and tested (rule criteria, explanation cards, HR weight calibration).
- `backend/app/providers/llm.py`: `LLMProvider` interface + `@register` registry + factory. Switch the whole agent with `LLM_PROVIDER` in `.env` (openai, anthropic, google, ollama, gateway, claude_code = the local `claude` CLI on a subscription login, dev only); `LLM_REASONING`/`LLM_FAST` pin a role to a full `"<provider>:<model>"`. Roles: `reasoning` (judgement) and `fast` (typed gate decisions via `providers/decider.py`).
- `backend/app/nlp/`: the CV pipeline (read_file, with OCR for scans in `ocr.py` → normalize → redact → sections/dates → structural chunks → skill mentions with NegEx-style hedges, years, depth cue → `CVProfile`). Everything downstream reads `pipeline.profile(cv_id)`, never raw text. `jd.py` parses JDs without an LLM (`JD_PARSER`), `quality.py` flags bad parses and duplicates. See `docs/NLP_PIPELINE.md`.
- `backend/app/rag/`: Qdrant hybrid store (fastembed: bge-small dense + BM25 sparse + MiniLM cross-encoder); `retrieve.py` = one query per requirement over chunks, max-pooled per CV. `data/` holds the synthetic, anonymised corpus; `data/eval/` the silver labels.
- `backend/app/eval/`: the B0 → P6 ladder, chunking ablation and A1–A4 against labels, on two benchmarks (`bench.py`: `ours`, and the `coding` branch's synthetic set via `--bench coding`; the agent runs on it via `bench materialize` + `DATA_DIR`), combined into the frozen **NLP501-bench v1** (`benchmark.py`, `docs/BENCHMARK.md`); extraction, robustness, real-text, scenario, error and human-label evaluations alongside; `docs/EVAL.md` (protocol, §8 = v3 decision rule) → `docs/EVAL_RESULTS*.md` (generated, never hand-edit).
- `backend/app/main.py`: FastAPI; runs stream as NDJSON; contract in `docs/API.md`. Checkpoints in `.runtime/checkpoints.sqlite`.
- `frontend/`: Vite + React + TS live view of the graph (`?mock=1` works without the backend).
- Langfuse (self-hosted) traces every run; one session per run_id.

## Component IDs
`docs/architecture.html` is the architecture map. Every component has a stable ID (N1–N12 nodes, R1–R4 routers, L1–L6 loops, S state keys, H harness controls, C0–C8 criteria, X1–X10 NLP stages, E1–E14 evaluation, P/D/O/T/A/U/K/I). When the user names an ID ("N6 is too slow", "tighten H1 for L2"), look it up in the `REG` table at the bottom of that file for its file › symbol. When you add, rename or remove a component, update `REG` in the same change.

## Commands
```bash
uv sync                                                  # Python deps (never pip)
docker compose up -d                                     # Qdrant; add --profile observability for Langfuse :3000
cd backend && uv run python -m app.rag.ingest            # (re)index data/ into Qdrant
cd backend && uv run python -m scripts.smoke JD-002      # end-to-end in terminal, auto-answers HR
uv run pytest                                            # tests (no LLM, no Docker needed)
cd backend && uv run python -m app.eval.run retrieval    # B0–B3 + chunking ablation (minutes); `agent` = LLM systems; `report`
# eval numbers in docs/ are produced with PYTHONHASHSEED=0, LLM_PROVIDER=claude_code and REFERENCE_DATE=2026-10-30
# ("Present" in a CV resolves to it, so years and ranks do not drift with the calendar); query text no longer depends on
# the hash seed (taxonomy.aliases keeps the declared order)
cd backend && uv run python -m app.eval.benchmark report    # THE official table: NLP501-bench v1 (docs/BENCHMARK.md); `verify` checks the frozen inputs
cd backend && uv run python -m app.eval.benchmark_v2 report # v2 = v1 + 8 real Djinni JDs in test (18 test JDs); `freeze`, `verify`
cd backend && uv run python -m app.eval.djinni fetch     # real-data layer to .runtime; `silver`, `import labels_A.xlsx labels_B.xlsx`, `score` (κ)
cd backend && uv run python -m app.eval.nlp_ablation     # E19: nDCG cost of switching off hedge scope / evidence / depth cue / dated years
cd backend && uv run --group ml python -m app.eval.finetune train   # E20: cross-encoder fine-tuned on dev JDs; `score` = test + Djinni
cd backend && uv run python -m app.eval.token_cost       # E21: tokens of B4 / P6 vs E-every (LLM on every candidate), after `run agent --systems B4,P6,E-every`
cd backend && uv run python -m app.eval.kaggle_resume    # E23: pipeline on 2,484 real résumé PDFs (reading, structure, tool false positives); `fetch cv2.zip` first
cd backend && uv run python -m app.eval.extraction       # RQ1: CV/JD extraction F1 + redaction leaks vs the coding branch's generator truth
cd backend && uv run python -m app.eval.agentic          # E15: P7 on HR tasks (behaviour checks, refusals, tokens, nDCG vs P6; dev JDs)
cd backend && uv run python -m app.eval.llm_rank         # E18: controlled LLM ranking of the shortlist, dev JDs, both benchmarks
cd backend && uv run python -m app.eval.single_agent     # S1: one agent on its own vs P6 (dev JDs; --inject JD-002,JD-003)
cd backend && uv run python -m app.eval.injection        # Q16 layer 1b: System One model (ollama pull nimble) vs paraphrased injection
cd backend && uv run python -m app.eval.robustness       # Q11: counterfactual fairness (name/gender/age/marital twins) + format robustness
cd backend && uv run python -m app.eval.clarify          # H6: vague twins of all 18 JDs, asking HR vs not (`report` re-renders)
cd backend && uv run --group ocr python -m app.eval.ocr  # scanned CVs: PaddleOCR (optional group `ocr`, ~1 GB) → pipeline; `rescore` = no OCR
uv run uvicorn app.main:app --app-dir backend --reload   # API :8000
cd frontend && npm run dev                               # UI :5173
```

## Rules
- Package manager is **uv** (Python) and **npm** (frontend). Docker Desktop, not Colima.
- LLM output never drives deterministic logic raw: canonicalize at the boundary (see `scoring.canonicalize`).
- Every LLM output has a criterion, a check and an action on failure (`docs/AI_OUTPUT_CRITERIA.md`, Q1–Q13; checks in `agent/checks.py`). A new LLM output needs a Q row too.
- Every loop has a limit in `config.Settings`. New loops need one too.
- Code that runs before `interrupt()` re-runs on resume, so keep it cheap and idempotent.
- No PII, protected attributes or real CVs in `data/`. Secrets live only in `.env` (gitignored).
- Keep it lean: no speculative abstractions. Add a test for any new deterministic logic.
- A change to retrieval, chunking, extraction, prompts or scoring is not "better" until `app.eval.run` says so; report negative results too. Tune on the benchmark's dev JDs only; report test once (docs/BENCHMARK.md).
