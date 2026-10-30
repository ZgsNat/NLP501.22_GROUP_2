# Evaluation protocol

This document is how we measure the system and how to reproduce the numbers. The numbers themselves are generated into [EVAL_RESULTS.md](EVAL_RESULTS.md), and the findings are summarised at the end. The slide-deck figures (86.8 nDCG@10 and so on) are **targets** from the literature, never results.

## 1. Data

| | Count | Notes |
|---|---|---|
| JDs | 12 | 4 company types; JD-003 is deliberately vague; JD-010 is written in Vietnamese |
| CVs | 50 | Synthetic and anonymised. CV-031–050 were added in v2 to stress the NLP pipeline (below) |
| (JD, CV) pairs | 600 | Every system ranks all 50 CVs for every JD |

What CV-031–050 add:
- 6 Vietnamese or mixed-language CVs
- 6 long CVs over 900 words, including one with overlapping roles and one with a career gap
- 3 hedging traps ("currently learning", "2-week training", "not used in production")
- 2 title-inflation traps ("Automation Architect" with L1 bullets)
- 2 strong CVs in plain language (few tool names)
- varied headings, bullets and date formats

`TRAPS` in `app/eval/run.py` also includes the three v1 keyword-stuffed manual testers (CV-004–006).

## 2. Labels (E1)

- **Scale.** 0 = not relevant (core must-have absent, or only claimed or hedged); 1 = weak or adjacent; 2 = good partial fit, would phone-screen; 3 = strong fit with demonstrated work, would interview. Annotators judge evidence, not keywords, and ignore anything that is not job-related.
- **Annotators.** Two independent annotators graded all 600 pairs: Claude Opus and Claude Sonnet, each in a fresh context. Each saw **only** the JD and CV text: no authoring notes, no code, no system output. Files: `data/eval/annotator_a.yaml` and `annotator_b.yaml`.
- **Agreement.** Exact agreement is 84.3% and quadratic-weighted Cohen's κ = 0.919. No pair differs by more than one grade.
- **Gold.** The mean of the two annotators, so half grades occur.
- **Status: SILVER.** Both annotators are LLMs from one model family, so their errors are correlated and κ is optimistic. Before the final report, two humans (Hoa plus one teammate) should:
  1. re-grade a stratified sample of 120 pairs blind;
  2. report human–human and human–silver κ;
  3. adjudicate the 94 pairs where A and B differ (`python -m app.eval.run report` writes them to `.runtime/eval/to_adjudicate.yaml`).

  If human–silver κ ≥ 0.8, keep the silver set as the main test set.

## 3. Systems: the ladder

| System | What it is | Uses an LLM for |
|---|---|---|
| B0 BM25 | Whole CV vs raw JD text, sparse only | — |
| B1 dense | Whole CV vs raw JD text, `bge-small-en-v1.5` | — |
| B2 hybrid | B0 + B1 fused with RRF (the v1 retrieval core) | — |
| R {whole, window, structural} (+ CE) | **Chunking ablation:** requirement-wise retrieval (D9) over each chunking, with and without the cross-encoder | JD requirement extraction (cached, shared) |
| B3 +criteria (rules) | R structural + CE, then hedge-aware knock-out and rule criteria (tools, ISTQB, years from dates, depth from the verb cue) | Requirement extraction only |
| B4 fixed pipeline | The agent with **every loop off**: no clarification, no expansion, no verification retries, no arbiter | Requirements, assessment, report |
| **P6 full agent** | Everything on. Scripted HR answers the JD-003 question, accepts proposed substitutes and approves. | All roles |
| A1 = P6 − verify | Evidence is measured but never retried or capped | |
| A2 = P6 − profile | Default weights and policies instead of the company type | |
| A3 = P6 − arbiter | No LLM tie-breaks; near-ties keep the deterministic tie-break order | |

All systems use the same corpus, the same extracted requirements (LLM extraction + deterministic canonicalization, H2 and X7 `jd_depth`; cached per JD in `.runtime/eval/requirements/`) and the same local model (`ollama:qwen3:8b`). Every system orders all 50 CVs: ranked, then not shortlisted, then flagged for review, then knocked out. That makes τ defined over the full pool.

## 4. Metrics (E2, `app/eval/metrics.py`, unit-tested)

- **nDCG@10 and nDCG@5**, with gain 2^g − 1 on the graded gold. This is the primary metric, because HR reads the top of the list.
- **P@5:** the share of the top 5 with gold ≥ 2.
- **Kendall's τ-b** between the system's order and the gold grades over all 50 CVs, which handles ties. It is the full-order view and says whether the whole pool is sorted sensibly.
- **traps@10:** trap CVs with gold < 2 in the top 10, summed over JDs.
- **Faithfulness, for LLM systems:** the % of LLM evidence quotes that are *grounded* (exist in the CV) and *supported* (right section and right entity), as defined in [NLP_PIPELINE.md](NLP_PIPELINE.md).
- **Cost:** seconds per JD, LLM calls and tokens.
- **Statistics:**
  - Each metric is the mean over JDs, with a 95% percentile-bootstrap CI that resamples JDs, the unit we sample.
  - System pairs are compared with a one-sided paired bootstrap on the same JDs.
  - With 12 JDs, only large differences can be significant, and the CIs show it.
- **Reference row:** annotator A's order scored against the gold. It is not a true ceiling (A is part of the gold), but it shows how close to "human" a system is.

## 5. Reproduce

```bash
docker compose up -d && cd backend && uv run python -m app.rag.ingest
uv run python -m app.eval.run retrieval      # B0–B3 + chunking ablation, ~5 min
uv run python -m app.eval.run agent          # P6, B4, A1, A2, A3 on qwen3:8b; ~1–2 h locally with a cold cache
uv run python -m app.eval.run report         # writes docs/EVAL_RESULTS.md
```

LLM responses are cached in `.runtime/llm_cache.sqlite` (H17). Rerunning `agent` or `report` reproduces the same numbers, and ablations that share calls with P6 cost little. Delete the cache to measure run-to-run variance.

## 6. Threats to validity

- **Synthetic CVs.** The team wrote the taxonomy and the CVs, so tool names follow the taxonomy's spelling. Real CVs will be noisier: misspellings, unusual layouts, scanned PDFs.
- **Silver labels.** See §2.
- **Small JD sample** (12). Report CIs, not just means; per-JD tables show where systems differ.
- **One local 8B model.** Assessment quality is lower than a frontier model, and absolute scores are not calibrated. The next run is the same ladder with a cloud model via `LLM_PROVIDER`.
- **Scripted HR.** The automated HR accepts every substitute and answers one clarification. A real recruiter may decide differently.

## 7. Findings (v2, 2026-10-30)

All numbers are from [EVAL_RESULTS.md](EVAL_RESULTS.md): 12 JDs, silver labels, local `qwen3:8b`. "p" is the one-sided paired bootstrap over JDs.

1. **Dense retrieval alone is the weakest system.** B1 dense scores nDCG@10 0.741 with 14 traps in the top 10, against 0.794 for B0 BM25 and 0.824 for B2 hybrid. This matches Resume2Vec: in software testing, tool names are lexical entities.
2. **The cross-encoder is the biggest single retrieval gain.** R structural goes from 0.777 to 0.842 with the cross-encoder (+0.065). Removing truncation and scoring (requirement, passage) pairs matters more than how we chunk.
3. **The chunking strategy alone makes no measurable difference** on this corpus. Structural vs whole, both + CE: −0.009, p = 0.70. Structural vs window: +0.002. We report this as a null result.
4. **Structure pays off downstream.** Hedge-aware knock-out and rules over parsed facts take traps@10 from 9 (B2) to 0. They also raise τ over the whole pool from 0.36 to 0.54, so the order of the full pool becomes sensible, not just the top.
5. **The best ranker uses no LLM.** B3 rules ⊕ CE scores 0.871 [0.831, 0.913], τ 0.542, 0 traps.
   - Caveat: the fusion (the mean of the normalised CE and rule scores) was designed after seeing the v2 retrieval results. It is not a held-out result.
6. **The local 8B LLM does not improve the ranking.**
   - P6 scores 0.838, 0.033 below its own LLM-free input (B3⊕). P6 beats B3⊕ in only 3.6% of bootstrap resamples, and τ is the same.
   - With this model, the LLM layer adds explanations and verified evidence, not ordering quality.
   - Whether a frontier model changes this is the next experiment: `LLM_REASONING=<provider>:<model>`, same ladder.
7. **Loops and the company profile have no measurable effect on ranking.** P6 ≈ B4 (−0.004), P6 = A3 (no arbiter, +0.000) and P6 ≈ A2 (default profile, −0.004). With 12 JDs, only differences of about 0.03 or more could be detected.
8. **The verify loop is what makes the evidence trustworthy.**
   - Grounded quotes: 96.1% without the loop (B4), 99.2% with it (P6). Supported quotes: 86.9% → 89.9%.
   - It costs a little ranking: A1 (verify measured but not enforced) is +0.011 nDCG@10 above P6, p = 0.007 for A1 > P6. Capping unproven criteria sometimes demotes a good candidate whose quote the 8B mangled.
9. **JD-side canonicalization** ([NLP_PIPELINE.md](NLP_PIPELINE.md), "The JD side"):
   - Fixing the dropped nice-to-haves, Vietnamese "hoặc", "one of" groups and the depth floor improved requirement-wise retrieval + CE by +0.021 (p = 0.003).
   - On the hardest JD (JD-011, architect), every agent system gained +0.17.
   - B3⊕ stayed flat (−0.005) because JD-001 lost 0.12: our depth policy reads "quality gates" as L4.
10. **Negative result kept:** down-weighting skills and summary passages cut traps at retrieval but cost 0.040 nDCG@10 (p = 0.977 for "better"). It stays off.
11. **Cost (cold cache):** P6 takes 199 s/JD on an M5 Pro, about 13 LLM calls and 37k tokens per JD. Other systems' seconds are lower partly because they reuse P6's cached responses, so compare their calls and tokens, not their time.
12. **Tier calibration:** 63 of 75 "strong" candidates (84%) have gold ≥ 2. Almost every shortlisted candidate is "strong", so the tiers add little beyond the rank.

## 8. v3 merge with the `coding` branch: decision rule (written before running)

v3 ports components from the `coding` branch (Đán) into this pipeline. Two benchmarks are used:
- **`ours`:** this file. Our code was developed on it, so it plays the role of a dev set.
- **`coding`:** his synthetic set, 200 CVs × 6 JDs (`app/eval/bench.py`). Our code was never tuned on it, so it is the closer thing to a held-out test. Its gold comes from rules similar to his scoring, which favours that style of ranker.

The baseline is the v2 deterministic ranker, `B3 rules ⊕ CE`: LLM requirements and 4 rule criteria, fused with the cross-encoder.

**Rule.** A ported component becomes the default only if both conditions hold:
1. Its mean nDCG@10 is **not lower than the baseline by more than 0.01 on either benchmark**.
2. It is **higher on at least one** benchmark.

Ties go to the simpler or zero-token option. With 6–12 JDs, CIs will overlap, so the rule uses point estimates and the report shows the CIs and paired p-values.

**Candidates:**

| Component | Origin | System name |
|---|---|---|
| Deterministic JD parser (`jd_parser = rules`) | `coding` `criteria/jd_parser.py` | `B3 core4 ⊕ CE [rules]` |
| Parser first, LLM fills gaps (`hybrid`) | merge | `B3 core4 ⊕ CE [hybrid]` |
| All 8 criteria by rule from the CVProfile | `coding` `ranking/scoring.py` (8 axes) on our hedge-aware profile | `B3 8-axis ⊕ CE [llm \| rules \| hybrid]` |

**Agent-level changes**, measured with the agent ladder on `ours` (the only benchmark with an indexed corpus for the graph):
- the LLM explains but does not set scores;
- the arbiter judges each pair in both orders;
- explanation cards.

Expected effect: P6 moves to B3⊕ level, the 0.033 gap in §7.6.
