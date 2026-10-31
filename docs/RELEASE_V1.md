# v1.0: Agentic CV Ranking prototype

**Status:** release gate passed on 2026-10-29, on a local model only (`ollama:qwen3:8b`, MacBook Pro M5 Pro, 24 GB).
**Branch:** `feature/thinh-agentic-architecture`

## What v1 is
An end-to-end agentic system that ranks Automation Test CVs against a JD and explains every decision with evidence quoted from the CV:
- FastAPI + LangGraph backend
- live React UI
- Qdrant hybrid retrieval
- self-hosted Langfuse tracing
- HR in the loop

It runs **fully locally at zero API cost**. Switching to OpenAI, Claude, Gemini or a gateway is one line in `.env`: `LLM_PROVIDER=...`.

## How to try it (10 min)
```bash
cp .env.example .env
docker compose --profile observability up -d     # Qdrant + Langfuse
ollama pull qwen3:8b
uv sync && cd backend && uv run python -m app.rag.ingest && cd ..
uv run uvicorn app.main:app --app-dir backend    # API :8000
cd frontend && npm install && npm run dev        # UI http://localhost:5173
```
Pick JD-003 in the UI to watch the agent stop and ask you (HR) a question. No time to set up? Use `http://localhost:5173/?mock=1`.

## Release gate results (measured, not targets)
`cd backend && uv run python -m scripts.e2e` drives all 4 sample JDs through the real HTTP API and plays HR (answers clarifications, re-weights once, approves).

| JD | Scenario | Time (s) | Loops fired | Shortlist / knocked out | Quotes grounded on 1st pass | Top-5 score range | LLM calls | Tokens |
|---|---|---|---|---|---|---|---|---|
| JD-001 | Bank, Selenium + Java, HR re-weights | 262 | L4 ×2, L3 ×4, L5 ×1 | 8 / 5 | 92.3% | 0.90–0.81 | 23 | 53,005 |
| JD-002 | Startup, Playwright (scarce) | 127 | L2 ×1, L3 ×2 | 5 / 5 | 100% | 0.89–0.79 | 9 | 21,436 |
| JD-003 | Outsourcing, deliberately vague | 321 | L1 ×1, L4 ×2, L3 ×2 | 8 / 5 | 93.0% | 0.99–0.95 | 16 | 34,530 |
| JD-004 | Performance + API | 240 | L4 ×2, L3 ×1 | 6 / 5 | 89.1% | 0.85–0.66 | 13 | 25,931 |

Gate: ✅ all runs completed · ✅ all 5 loops fired · ✅ top-5 ranking decisive (no unresolved ties) · ✅ token usage recorded · ✅ reports mention only ranked candidates.

How to read these numbers:
- **Grounding:** 89–100% of LLM evidence quotes exist verbatim in the CV on the first try. The L4 loop catches the rest, so the final reports are 100% grounded.
- **Where tokens go:** ~70% of tokens are spent in `assess`, the per-candidate LLM scoring. `ingest` costs ~500 tokens. The arbiter costs ~1–3k because it only runs on near-ties.
- **Latency:** 2–5 min per JD on a local 8B model, and it varies with machine load (same calls/tokens, up to 60% slower when the laptop was busy). Cloud models will be much faster.
- **These are not the slide targets.** nDCG@10, Kendall's τ and the B0 → P6 ladder need gold labels (next phase).

## What changed from the first prototype (lessons worth presenting)
Each fix came from a real failure observed in a run:
1. **Canonicalize at the boundary.** The LLM wrote sentences as "must-have tools" and all 30 CVs were rejected. Requirements are now mapped onto a skills taxonomy.
2. **Bound generation, isolate failures.** A 400 KB repetition loop crashed a run. Output is now capped, and a bad candidate is flagged for manual review instead of crashing the run.
3. **Rules before models.** A vague JD with zero must-have tools went unnoticed by the LLM. A rule now always asks HR.
4. **Extract facts, compute scores.** The 8B model gave everyone ~0.95. Now `tools` and `istqb` are rules, depth (L1–L4) and years are extracted facts scored by code, and exact ties are ordered by depth → years → relevance.
5. **Bound the arbiter.** One candidate was pushed from #1 to #5 by four consecutive LLM calls. Each candidate is now judged at most once.
6. **Few-shot examples leak.** The 8B model copied a CV id from a prompt example into a real report. A zero-token guard now checks that reports only mention ranked candidates.
7. **Measure the cost.** Tokens per node are recorded for every run and persisted with its narration log, and `?run=<id>` links are shareable.

## Known limitations (honest list)
- **Absolute calibration is unvalidated.** On an easy JD a strong pool scores 0.95–0.99. The order is decisive, but whether 0.97 *means* "better than 0.95" needs HR gold labels.
- **Synthetic data only:** 30 CVs, 4 JDs, English.
- Local 8B model quality: summaries are serviceable, not polished.
- In-process state: one API worker; usage and logs live in `.runtime/`.

## Next phase: evaluation (proposal for the team)
| Work | Owner (suggested) |
|---|---|
| Gold set: HR-style relevance labels for 4+ JDs (graded 0–3), group split | Hoa (datasets & metrics) |
| Evaluation harness: nDCG@10, Kendall's τ, faithfulness; B0 → B4 → P6 ladder + A1/A2 ablations | Thịnh |
| Check the 8 criteria and company weights against the literature; more JDs per company type | Sơn |
| Map v1 design choices to the papers (ConFit, Resume2Vec, RankGPT, bias) for the report | Vũ, Đán |
| Same gate with a cloud model (`LLM_PROVIDER=anthropic` / `openai`), comparing quality, latency and cost | anyone with a key |
