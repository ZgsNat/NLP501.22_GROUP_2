# v2.0: NLP pipeline and measured evaluation

**Status:** evaluated on 2026-10-30 with a local model only (`ollama:qwen3:8b`, MacBook Pro M5 Pro, 24 GB).
**Branch:** `feature/thinh-agentic-architecture`

## Why v2
Feedback on v1 was that the NLP had to be handled professionally. v1 embedded each CV as one vector and gave the whole text to the LLM. It also had no gold labels, so no claim could be measured. v2 does two things:
1. **Processes CV and JD text properly.** A deterministic pipeline recovers the structure of a CV before anything else reads it ([NLP_PIPELINE.md](NLP_PIPELINE.md)).
2. **Measures every claim.** There are 600 graded (JD, CV) pairs and a ladder of 18 systems with bootstrap CIs and paired tests ([EVAL.md](EVAL.md) → [EVAL_RESULTS.md](EVAL_RESULTS.md)).

## What is new
| Area | v1 | v2 |
|---|---|---|
| Input | Markdown only | PDF, DOCX and MD/TXT upload (`POST /api/cvs`); only redacted text is stored |
| Text processing | Raw text | NFC normalisation, de-hyphenation, PII and protected-attribute redaction (EN + VN), language detection |
| Structure | None | Bilingual sections, role and project entries, date grammar (EN/VN), merged intervals, career gaps excluded |
| Chunking | One vector per CV (11 of 50 CVs truncated at 512 tokens) | Structural chunks with a contextual header. Window and whole kept for the ablation |
| Skills | Substring match | Gazetteer NER over the taxonomy; evidence strength used / listed / hedged; NegEx-style hedge scope in EN + VN |
| Facts | LLM estimated years and depth | Years computed from parsed dates; depth cue from clause-initial verbs; the LLM's depth is cross-checked |
| JD requirements | LLM output, lightly mapped | Canonicalized per and-clause in EN + VN; every nice-to-have kept; JD alternatives credited in full; deterministic depth floor |
| Retrieval | One query per JD, whole CV | One query per requirement over chunks, max-pooled per CV; cross-encoder on (requirement, passage) pairs |
| Knock-out | Missing tool | A hedged-only mention does not count; runs over the whole pool |
| Faithfulness | Quote exists | Quote exists (*grounded*) and comes from the right section and names the right entity (*supported*) |
| Output | Shortlist | Every CV gets a status (ranked / not_shortlisted / needs_review / knocked_out) and a reason. Candidates who meet must-haves only via HR-accepted fallbacks rank after direct matches |
| HR | Weight sliders | Plain-language re-weighting ("certification matters more"), approval of substitutes (L2), a re-review budget |
| Data | 30 CVs, 4 JDs, English | 50 CVs (6 VN or mixed, 6 long, hedging and title-inflation traps), 12 JDs (1 in VN), silver labels |
| Tests | 32 | 93 (46 for the NLP pipeline) |

## Results (12 JDs, silver labels, mean with 95% CI)
| System | nDCG@10 | τ vs gold | traps@10 |
|---|---|---|---|
| B2 hybrid (the v1 retriever) | 0.824 [0.755, 0.888] | 0.36 | 9 |
| R structural + cross-encoder | 0.842 [0.786, 0.901] | 0.45 | 1 |
| **B3 rules ⊕ CE (no LLM)** | **0.871** [0.831, 0.913] | 0.54 | 0 |
| P6 full agent | 0.838 [0.796, 0.884] | 0.54 | 0 |

P6 faithfulness: 99.2% of evidence quotes grounded and 89.9% supported. P6 cost: 199 s, ~13 LLM calls and ~37k tokens per JD on the local model.

## Release gate (HTTP API, all 12 JDs)
`cd backend && uv run python -m scripts.e2e` drives every sample JD through the real HTTP API and plays HR: it answers JD-003's question, accepts substitutes, re-weights JD-001 once and approves.

| JD | Time (s) | Loops fired | Shortlist / knocked out | Quotes grounded on 1st pass | Top-5 score range | LLM calls | Tokens |
|---|---|---|---|---|---|---|---|
| JD-001 | 214 | L4 ×1, L3 ×4, L5 ×1 | 8 / 35 | 97.9% | 0.99–0.93 | 15 | 40,196 |
| JD-002 | 145 | L3 ×2 | 5 / 45 | 100% | 0.94–0.85 | 9 | 25,891 |
| JD-003 | 194 | L1 ×1, L4 ×1, L3 ×2 | 8 / 31 | 98.7% | 0.97–0.96 | 14 | 34,521 |
| JD-004 | 163 | L4 ×1, L3 ×1 | 5 / 45 | 92.7% | 0.92–0.69 | 11 | 27,044 |
| JD-005 | 225 | L4 ×2, L3 ×2 | 6 / 44 | 96.0% | 0.94–0.88 | 14 | 43,468 |
| JD-006 | 164 | L4 ×1, L3 ×2 | 7 / 43 | 98.5% | 0.91–0.81 | 12 | 30,766 |
| JD-007 | 210 | L4 ×1, L3 ×2 | 8 / 40 | 97.5% | 0.95–0.89 | 13 | 37,653 |
| JD-008 | 207 | L3 ×2 | 8 / 14 | 100% | 0.94–0.89 | 12 | 35,239 |
| JD-009 | 263 | L2 ×1, L4 ×2, L3 ×2 | 8 / 37 | 92.2% | 0.90–0.90 | 15 | 46,670 |
| JD-010 (VN) | 215 | L3 ×2 | 8 / 35 | 100% | 0.99–0.90 | 12 | 35,841 |
| JD-011 | 364 | L4 ×2, L3 ×2 | 8 / 12 | 83.3% | 0.97–0.88 | 20 | 58,817 |
| JD-012 | 179 | L4 ×1, L3 ×1 | 7 / 43 | 98.2% | 0.92–0.68 | 11 | 29,111 |

Gate: ✅ all 12 runs completed · ✅ all 5 loops fired · ✅ top 5 decisive (no unresolved ties) · ✅ token usage recorded · ✅ reports mention only ranked candidates · ✅ every one of the 50 CVs has a status and a reason, for every JD.

Note: "Knocked out" counts the whole pool of 50, because v2 screens every CV and v1 screened only retrieved ones. JD-002 no longer triggers L2, because the v2 corpus has more Playwright CVs. L2 now fires on JD-009 (Robot Framework).

## Lessons worth presenting
1. **Measure before you believe.** We expected chunking to be the big win. It made no measurable difference (p ≈ 0.7). The cross-encoder (+0.065) and parsed structure (traps 9 → 0, τ 0.36 → 0.54) were what mattered.
2. **Report negative results.** Down-weighting skills sections looked principled but cost 0.04 nDCG@10. The switch stays in the code, off, with the reason next to it.
3. **A small local LLM is the weak link, not the pipeline.** With qwen3:8b the agent ranks slightly below its own deterministic input. The LLM's value today is the explanations and verified evidence HR reads, not the order.
4. **Read the cached intermediate outputs.** The worst JD (0.58) was traced to our own code, not the model: nice-to-haves were silently dropped, Vietnamese "hoặc" was not read as "or", and two "one of" groups were merged. Fixing them gave +0.021 retrieval nDCG@10 (p = 0.003) and +0.17 on that JD.
5. **Encode HR logic, not tuned weights.** An alternative HR accepted only because the pool was scarce (Selenium for Robot Framework) must not outrank a direct match. That rule removed a mobile lead from #1 on the Robot Framework JD.
6. **Verify loops buy trust, not rank.** Verification raises grounded quotes from 96.1% to 99.2% and costs about 0.01 nDCG@10.

## Known limitations
- **Silver labels.** Two Claude models graded the pairs. Their errors are correlated, and a Claude assessor would be favoured. The 120-pair blind human sample contains all 94 disagreements plus 26 agreeing controls.
- **Post-hoc fusion.** B3⊕ was designed after seeing v2 retrieval results. It needs new JDs to count as a held-out result.
- **Small sample.** With 12 JDs, only differences of about 0.03 nDCG@10 or more can be detected.
- **Synthetic CVs** spelled with the taxonomy's names. Real CVs will be noisier, and scanned PDFs are rejected (no OCR).
- **Tier calibration.** 84% of "strong" candidates have gold ≥ 2, but almost everyone shortlisted is "strong".
- **The JD depth floor** reads "quality gates" as L4, per our policy. That cost 0.12 on JD-001, where annotators preferred L3 framework builders.

## Next steps (suggested owners)
| Work | Owner |
|---|---|
| Grade the 120-pair blind sample (all 94 disagreements + 26 agreeing controls); report human–silver κ | Hoa + one teammate |
| The same ladder with a frontier model for `reasoning`. Use a non-Claude family too (label bias) | Thịnh |
| 4–6 new JDs as a held-out test of B3⊕ and the JD depth floor | Sơn |
| Map the findings (cross-encoder, structure, LLM-as-judge limits) to the 31 papers for the report | Vũ, Đán |
| A multilingual embedder (`--dense intfloat/multilingual-e5-large`) for the Vietnamese CVs | anyone |
