# API contract (backend ⇄ frontend)

Base URL: `http://localhost:8000/api`. All bodies are JSON. Streaming endpoints return
**NDJSON** (`application/x-ndjson`): one JSON event per line.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | `{ "status": "ok" }` |
| GET  | `/config` | Providers, default models, company profiles, criteria, sample JDs |
| POST | `/runs` | Start a run → NDJSON event stream |
| POST | `/runs/{run_id}/resume` | Answer an HR interrupt → NDJSON event stream |
| GET  | `/runs/{run_id}` | Snapshot of the current graph state + pending interrupt |
| GET  | `/cvs/{cv_id}` | `{ id, headline, text }` (anonymised CV) |

### `GET /config`
```json
{
  "providers": [{ "name": "openai", "configured": true }, { "name": "ollama", "configured": true }],
  "defaults": { "reasoning": "ollama:qwen3:8b", "fast": "ollama:qwen3:8b" },
  "company_profiles": [{ "key": "default", "label": "Default (balanced)" }],
  "criteria": [{ "key": "tools", "label": "Tools / frameworks (must-have)" }],
  "sample_jds": [{ "id": "JD-001", "title": "...", "company_profile": "bank", "text": "..." }],
  "langfuse_url": "http://localhost:3000" 
}
```
`langfuse_url` is `null` when tracing is disabled.

### `POST /runs`
```json
{ "jd_text": "...", "company_profile": "bank", "models": { "reasoning": "openai:gpt-5", "fast": "ollama:qwen3:8b" } }
```
`models` is optional; each value is `"<provider>:<model>"`.

### `POST /runs/{run_id}/resume`
Body depends on the pending interrupt `kind`:
- `clarification` → `{ "answers": { "<question>": "<answer>" } }`
- `review` → `{ "action": "approve" }` or `{ "action": "adjust", "weights": { "<criterion>": 0.3 } }`

## Stream events

Each line is one of:

```jsonc
{ "type": "run_started", "run_id": "…" }
{ "type": "node_start", "node": "search" }
{ "type": "log", "node": "search", "level": "info" | "loop" | "warn", "message": "…" }
{ "type": "update", "node": "search", "update": { /* partial state, see below */ } }
{ "type": "node_end", "node": "search", "duration_ms": 412 }
{ "type": "interrupt", "kind": "clarification", "payload": { "questions": ["…"] } }
{ "type": "interrupt", "kind": "review", "payload": { "ranking": [/*…*/], "weights": {/*…*/} } }
{ "type": "done", "report": { /*…*/ } }
{ "type": "error", "message": "…" }
```
`level: "loop"` marks a reflection loop firing (the interesting agentic moments).

## Graph nodes (fixed names) and loops

```
ingest → clarify ─┬─(ambiguous, L1: ask HR)──→ ingest
                  └→ search ─┬─(coverage too low, L2)→ expand_query → search
                             └→ rerank → assess → verify_evidence ─┬─(ungrounded, L4)→ assess
                                                                    └→ score → arbitrate(L3 near-ties) → report → hr_review ─┬─(adjust, L5)→ score
                                                                                                                             └─(approve)→ END
```
Node names: `ingest, clarify, search, expand_query, rerank, assess, verify_evidence, score, arbitrate, report, hr_review`.

## State shapes (keys that may appear in `update`)

```ts
type Criterion = "tools" | "automation_depth" | "programming_language" | "test_types"
               | "cicd_infra" | "experience_level" | "istqb" | "domain_language";

requirements: {
  title: string; must_have: string[]; nice_to_have: string[];
  min_years: number | null; depth_level: number | null;   // depth 1–4
  languages: string[]; test_types: string[]; ambiguities: string[];
}
weights: Record<Criterion, number>                  // sums to 1.0
policy_context: { source: string; text: string }[]  // RAG snippets
query_terms: string[]
candidates: { cv_id: string; headline: string; retrieval_score: number;
              rerank_score: number | null; matched_must_have: string[] }[]
assessments: Record<cv_id, {
  criteria: Record<Criterion, { score: number /*0–100*/; rationale: string;
                                evidence: { quote: string; grounded: boolean | null }[] }>;
  knocked_out: boolean; knockout_reason: string | null; attempts: number;
}>
ranking: { rank: number; cv_id: string; headline: string; total: number /*0–1*/;
           knocked_out: boolean; arbiter_note: string | null }[]
report: { summary: string;
          candidates: { cv_id: string; explanation: string; strengths: string[]; gaps: string[] }[] }
loops: { clarify: number; expand: number; verify: number; arbitrate: number; review: number }
```

### `GET /runs/{run_id}`
```json
{ "run_id": "…", "values": { /* full state, same shapes */ }, "next": ["hr_review"],
  "interrupt": { "kind": "review", "payload": { } } | null }
```
