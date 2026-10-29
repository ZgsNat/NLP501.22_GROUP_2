# Morning walkthrough: read the agent in 45 minutes

The goal isn't "it works". The goal is that **you can explain every decision the agent makes and where you control it.**

## 0. Start it (5 min)
```bash
docker compose --profile observability up -d     # Qdrant :6333 + Langfuse :3000 (login: LF_ADMIN_* in .env)
cd backend && uv run python -m app.rag.ingest    # index 30 CVs + 5 policies (idempotent)
uv run uvicorn app.main:app --app-dir backend --reload    # API :8000 (from repo root)
cd frontend && npm run dev                                 # UI http://localhost:5173  (?mock=1 = no backend)
```
Want cloud models? Put a key in `.env`, then type e.g. `anthropic:claude-sonnet-5-5` in the UI's "reasoning" box. That's the provider factory at work: no code change.

## 1. Read in this order
| # | File | The question it answers |
|---|---|---|
| 1 | `backend/app/agent/graph.py` | What can happen? The 5 routers *are* the agentic behaviour. |
| 2 | `backend/app/agent/state.py` | What does the agent remember? Note the **reducers** (`merge`, `add_counts`). They're why nodes can return partial updates. |
| 3 | `backend/app/agent/nodes.py` | How is each step done? Look for `emit()`: every decision is narrated. |
| 4 | `backend/app/agent/scoring.py` | What is deliberately **not** an LLM? Knock-out, scoring, grounding and canonicalization are all pure and tested. |
| 5 | `backend/app/providers/llm.py` + `decider.py` | How is the "brain" swappable, and what is a System-One decision? |
| 6 | `backend/app/main.py` | How do streaming + interrupts + checkpoints become an HTTP API? |

## 2. Harness ideas to notice
1. **Bounded autonomy.** Every cycle has a limit in `config.Settings` (`max_clarifications`, `max_expansions`, `max_verify_retries`). An agent without limits is a cost incident waiting to happen.
2. **Canonicalize at the boundary.** The first real run failed: qwen3:8b put whole sentences in `must_have`, and the knock-out rejected all 30 CVs. The fix wasn't a better prompt. It was `scoring.canonicalize()`, which maps free text onto the taxonomy. *LLM output never drives deterministic logic raw.*
3. **Cheap before expensive.** BM25 + dense (ms) → cross-encoder (ms) → LLM only on the shortlist → arbiter LLM only on near-ties. That's the Selective Arbiter idea from the slide, and where the token savings will come from.
4. **Verify, don't trust.** `verify_evidence` checks every quote against the CV text with zero tokens. Ungrounded quotes go back to the LLM with feedback (L4); after the retry budget runs out, unsupported scores are capped.
5. **Interrupts re-run the node.** On resume, LangGraph re-executes the interrupted node from the top. That's why `clarify` and `hr_review` do nothing expensive before `interrupt()`, and why the ambiguity *decision* lives in `ingest`.
6. **Typed decisions (Jev idea).** `decide()` returns a `Literal` choice + confidence, never prose to parse. Today it's backed by an LLM with structured output. Swap in Jev or a small local classifier later without touching nodes.
7. **Bound the generation, isolate the failure.** The second real run hit a degenerate repetition loop: qwen3:8b wrote ~400 KB of *"The candidate has experience with…"* inside one `rationale` field. The JSON never closed, the parse failed, and the **whole run crashed** after ~10 minutes. There were two fixes: `num_predict` + `repeat_penalty` in the Ollama factory, plus a length budget in the schema field description (bound the generation); and `_assess_one` retries once and then flags that one candidate as "needs manual review" instead of raising (isolate the failure). Note that the candidate is *flagged*, not silently dropped. Silently dropping candidates would be a fairness bug.
8. **Observability.** Open Langfuse → Sessions. One session per run, one trace per resume. Compare what the UI timeline says with what the LLM actually saw.

## 3. Try these experiments
- JD-002 (Playwright): watch **L2** fire. Only 3 CVs mention Playwright, so it accepts Cypress/Selenium as substitutes. Was that the right call? Should HR be asked instead?
- JD-003 (ambiguous): watch the **L1** gate decide whether to interrupt you.
- In the review step, set `istqb` to 0.4 and re-score: **L5** with zero new LLM assessments.
- Run the same JD with `ollama:qwen3:8b` vs a cloud model. Compare the grounded %, latency and ranking. That's the start of your experiment table.

## 4. Found in tonight's real runs (open questions for you)
- **Score saturation:** qwen3:8b gives almost everyone 0.90–1.00, so the weighted score barely separates candidates. Is that the model or the prompt? Try a cloud model on the same JD and compare. That comparison is your first experiment.
- **Arbiter cascades:** in JD-003, CV-002 (highest score, but a Playwright/TS profile for a Selenium/Java JD) was moved from #1 to #5 by four consecutive near-tie decisions. The arbiter was arguably *right*, correcting an over-score. But should one LLM be allowed to override the scores that much? Possible fixes: cap moves per candidate, or rerank a near-tie cluster in one listwise call.
- **Token accounting:** Langfuse has every trace (sessions = run_id), but no token usage for the Ollama generations. You need real token counts before you can claim "−78% tokens".
- **JD-003 lesson:** the LLM read "several years" as 3 and flagged no ambiguity, so the run went ahead with **zero** must-have tools. The fix is a deterministic rule ("no must-have tools → always ask HR") that runs *before* the LLM gate.

## 5. What's NOT built yet (next sessions)
- **Evaluation harness:** gold labels, nDCG@10, Kendall's τ, faithfulness, the B0→B4→P6 ladder and A1/A2 ablations. This turns slide targets into real numbers.
- PII scrubbing on ingest (the corpus is synthetic and already anonymised).
- Applying `docs/claude-settings.proposed.json` (your decision, see below), and installing ponytail together.

## 6. Claude setup in this repo
- `CLAUDE.md` is the team context, loaded every session. Keep it short; every line costs tokens on every turn.
- `.claude/agents/`:
  - `agent-engineer` (Opus): graph and agent design, where reasoning quality matters.
  - `fullstack-dev` (Sonnet): well-specified work like API, UI, data and tests. *This is your Sonnet answer: the middle tier for volume work where the spec is clear.*
  - `git-steward` (Haiku): status, diffs, commit plans. Cheap and read-mostly.
- How this prototype was built: Opus wrote the contract (`docs/API.md`) first, then two Sonnet subagents built the data and the frontend **in parallel** against it, while Opus built the agent core. A contract first is what makes parallel subagents safe.
- `docs/claude-settings.proposed.json`: an allowlist so you stop approving `git status`. Claude was (correctly) blocked from granting itself permissions. Review it and copy it to `.claude/settings.json` if you agree.
