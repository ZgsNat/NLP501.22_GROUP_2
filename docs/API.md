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
| GET  | `/cvs` | List the CV pool: `[{ id, headline, lang, uploaded }]` |
| POST | `/cvs` | Upload a CV (multipart) → `CVProfileView` (parse preview) |
| GET  | `/cvs/{cv_id}` | `{ id, headline, text, profile }` (processed, redacted CV + its parse preview) |

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
- `substitutes` → `{ "accept": { "<must_have>": ["<related tool>", ...] } }`. An empty list rejects substitutes for that must-have;
  a must-have missing from `accept` is treated as rejected.
- `review` → one of:
  - `{ "action": "approve" }`
  - `{ "action": "adjust", "weights": { "<criterion>": 0.3 } }` and/or `"instruction": "certification matters more"`
  - `{ "action": "decide", "decisions": [{ "cv_id": "CV-018", "action": "approve" | "reject" | "override", "reason": "…", "new_rank": 1 }] }`
    (v3). Decisions without a reason are ignored. Each one is appended to `.runtime/hr_audit.jsonl`. The weights that best explain the decisions (pairwise logistic regression on criterion scores, shrunk towards the current weights) re-score the pool. This costs one re-score round.

### `GET /cvs`
```json
[{ "id": "CV-014", "headline": "Senior Automation Test Engineer, 6 yrs, Selenium/Java", "lang": "en", "uploaded": false }]
```
`lang` is `"en" | "vi" | "mixed"`. `uploaded` is `true` for CVs added through `POST /cvs` (as opposed to the seeded corpus).

### `POST /cvs`
`multipart/form-data` with one field `file` (`.pdf`, `.docx`, `.md`, `.txt`, `.png` or `.jpg`; scanned PDF pages and photos
are read by OCR when the optional `ocr` dependency group is installed). On success: **201** with a `CVProfileView`
(see below), and the CV joins the pool, so it is returned by `GET /cvs` and `GET /cvs/{id}` and is searchable in later runs.
**400** `{ "detail": "…" }` on an unsupported format or when no text can be extracted (empty file, or a scan/photo without the `ocr` group).

### `GET /cvs/{cv_id}`
```json
{ "id": "CV-014", "headline": "…", "text": "…processed, redacted text…", "profile": { /* CVProfileView */ } }
```
`text` is the text after the NLP pipeline (PII replaced by placeholders such as `[EMAIL]`), i.e. what retrieval, the assessor and
`verify_evidence` see. 404 when the id is unknown.

### `CVProfileView`
The output of the CV NLP pipeline, shown in the UI as the "Parse preview".
```ts
{
  id: string; headline: string;
  lang: "en" | "vi" | "mixed";
  redactions: Record<string, number>;          // PII removed, by kind: { email: 1, phone: 1, birth_date: 1 }
  sections: { kind: string; heading: string }[];   // in document order; kind e.g. summary, experience, projects, skills, education
  chunks: { id: string; section: string; header: string; text: string }[];
                                               // header = contextual header prepended when indexing (id | headline | section)
  skills: { skill: string; strength: "used" | "listed" | "hedged"; section: string; line: string }[];
                                               // used = shown in described work; listed = only in a skills list;
                                               // hedged = "familiar with…", "learning…"; line = the CV line that proves it
  years_total: number | null;
  years_automation: number | null;
  depth_cue: number;                           // automation depth cue 0–4 (0 = none), see POL-depth-scale
  depth_cue_line: string | null;               // the CV line the cue was taken from
}
```

## Stream events

Each line is one of:

```jsonc
{ "type": "run_started", "run_id": "…", "trace_url": "http://localhost:3000/…" | null }
{ "type": "node_start", "node": "search" }
{ "type": "log", "node": "search", "level": "info" | "loop" | "warn", "message": "…" }
{ "type": "update", "node": "search", "update": { /* partial state, see below */ } }
{ "type": "node_end", "node": "search", "duration_ms": 412 }
{ "type": "interrupt", "kind": "clarification", "payload": { "questions": ["…"] } }
{ "type": "interrupt", "kind": "substitutes", "payload": { "proposals": [{ "must_have": "playwright", "covered": 3, "related": ["cypress", "selenium", "webdriverio"] }] } }
{ "type": "interrupt", "kind": "review", "payload": { "ranking": [/*…*/], "weights": {/*…*/} } }
{ "type": "done", "report": { /*…*/ }, "usage": Usage }
{ "type": "error", "message": "…" }
```
`level: "loop"` marks a reflection loop firing (the interesting agentic moments).

### Interrupt kinds
| kind | raised by | payload | resume body |
|---|---|---|---|
| `clarification` | `clarify` (L1) | `{ questions: string[] }` | `{ answers: { question: answer } }` |
| `substitutes` | `expand_query` (L2) | `{ proposals: { must_have: string; covered: number; related: string[] }[] }` | `{ accept: { must_have: string[] } }` |
| `review` | `hr_review` (L5) | `{ ranking: RankRow[]; weights; adjustments_left }` | `{ action: "approve" }`, `{ action: "adjust", weights?, instruction? }` or `{ action: "decide", decisions }` |

`substitutes`: when coverage of a must-have is too low (`covered` CVs show it), L2 no longer widens the query silently; it
proposes related tools (taxonomy) and HR picks which may stand in for the must-have. The client shows `expand_query` as
"waiting for HR" while it is pending. On resume the node re-runs with the accepted tools added to `query_terms`.

## Graph nodes (fixed names) and loops

```
ingest → clarify ─┬─(ambiguous, L1: ask HR)──→ ingest
                  └→ search ─┬─(coverage too low, L2)→ expand_query ⏸ HR picks substitutes → search
                             └→ rerank → assess → verify_evidence ─┬─(ungrounded, L4)→ assess
                                                                    └→ score → arbitrate(L3 near-ties) → report → hr_review ─┬─(adjust, L5)→ score
                                                                                                                             └─(approve)→ END
```
Node names: `ingest, clarify, search, expand_query, rerank, assess, verify_evidence, score, arbitrate, report, hr_review`.
Three nodes can pause the graph for HR (interrupt): `clarify` (`clarification`, L1), `expand_query` (`substitutes`, L2) and
`hr_review` (`review`, L5).

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
  criteria: Record<Criterion, { score: number /*0–100*/; rationale: string; evidence: Evidence[] }>;  // v3: all 8 by rule
  llm_review: Partial<Record<Criterion, { score: number; rationale: string; evidence: Evidence[] }>>;
      // v3: the LLM's judgement of depth, language, test types, CI/infra, domain. Verified by verify_evidence and shown,
      // but it does not set the score unless LLM_IN_SCORE=true (the v2 behaviour).
  knocked_out: boolean; knockout_reason: string | null; attempts: number;
  facts: { depth_level: number | null; years: number | null;       // facts the scorer used
           depth_cue: number | null; years_automation: number | null };
}>
Evidence = { quote: string;
             grounded: boolean | null;   // quote found in the CV text (null = not checked yet)
             supported: boolean | null;  // the section it came from demonstrates the claim (null = not checked yet)
             section: string | null }    // where the quote was found: experience, projects, skills, summary, …
// grounded && !supported = "listed, not demonstrated" (e.g. depth evidence quoted from a skills list);
// !grounded = hallucinated quote (L4 sends it back to assess).

ranking: RankRow[]    // EVERY CV in the pool, in this order: ranked (by rank), not_shortlisted, needs_review, knocked_out
RankRow = { rank: number; cv_id: string; headline: string; total: number /*0–1*/;
            knocked_out: boolean; arbiter_note: string | null;
            status: "ranked" | "needs_review" | "not_shortlisted" | "knocked_out";
            tier: "strong" | "possible" | "weak" | null;   // only set when status = "ranked"
            reason: string | null;                         // why the CV is not ranked; null for ranked
            card: Card | null;                             // v3: deterministic explanation (ranked + knocked_out rows)
            flags: string[] }                              // v3: short_text, garbled_text, few_skills, no_experience_section,
                                                           //     no_dated_roles, duplicate_of:<cv_id>
Card = { decomposition?: { criterion; weight; score /*0–100*/; contribution /*weight·score/100; sums to total*/ }[];
         versus_next?: { vs: cv_id; advantages: { criterion; delta }[]; disadvantages: { criterion; delta }[] } | null;
         what_if?: { if_used: string[]; must_have: boolean; passes_knockout: boolean; total: number; delta: number }[] }
// what_if on a knocked-out row = one scenario with every missing must-have used in a role.
// reason examples:
//   not_shortlisted  "Rank 11 after rerank; best evidence matched: selenium, java; missing: jenkins"
//   knocked_out      "Missing must-have: playwright"
//   knocked_out      "Only hedged mention of appium: 'familiar with Appium (not used in production)'"
// knocked_out (bool) stays for compatibility and is true exactly when status = "knocked_out".
report: { summary: string;
          candidates: { cv_id: string; explanation: string; strengths: string[]; gaps: string[] }[] }
loops: { clarify: number; expand: number; verify: number; arbitrate: number; review: number }
```

### `GET /runs/{run_id}`
```json
{ "run_id": "…", "values": { /* full state, same shapes */ }, "next": ["hr_review"],
  "interrupt": { "kind": "review", "payload": { } } | null, "usage": Usage,
  "logs": [{ "type": "log", "node": "search", "level": "loop", "message": "…" }] }
```
Usage and the narration log persist in `.runtime/runs/<run_id>/`, so `http://localhost:5173/?run=<run_id>` restores a run
with its full "why" even after a server restart. `run_id` must match `[A-Za-z0-9_-]{1,64}` (400 otherwise).

### `Usage` (LLM cost accounting, summed over the start stream and every resume)
```ts
{ calls: number; input_tokens: number; output_tokens: number; total_tokens: number;
  by_node: Record<NodeName, { calls: number; input_tokens: number; output_tokens: number }> }
```
