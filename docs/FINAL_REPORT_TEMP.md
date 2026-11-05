# Final report — working draft (team view)

Status snapshot, 2026-11-04. Purpose: one page so every member sees what the system does, what is measured, what is
still open, and where to contribute. Every number below is copied from a generated `docs/*_RESULTS.md` file. The
literature-review deck's numbers (86.8 nDCG@10, −78% tokens, 97.6% grounded, τ = 0.81) are **targets**, compared at
the end; never cite them as results.

The structure follows the six parts of the course brief.

---

## 01 · Problem, goal, scope, assumptions (owner: Sơn)

**Problem.** An HR recruiter has one job description (JD) for an Automation Test role and a pool of CVs. They need a
ranked shortlist that follows *their company type's* criteria, with a reason per candidate that can be checked
against the CV, and they keep the final decision.

**Scoring model.** `Score(CV, JD) = 1[H(c)] × Σ_k w(c,k) · S(c,k)`
- `H`: hard knock-out on must-have tools (with HR-approved substitutes).
- `w`: per-company-type weights over 8 criteria (`data/criteria_profiles.yaml`).
- `S`: 0–100 per criterion, computed **by rules** over the parsed CV (`scoring.rule_criteria`). The LLM explains,
  arbitrates near-ties and may reorder the shortlist; it never sets `S`.
- Automation depth L1–L4 (run → write → framework → test architecture), `data/policies/POL-depth-scale.md`.

**Scope.** Automation Test roles only; CVs in English and Vietnamese; synthetic, anonymised data; the system
recommends, it does not hire.

**Assumptions.** Labels are silver (LLM-made) and rubric-based; human labels are not in yet (see §Open work).

---

## 02 · Literature and gaps (owners: Vũ, Đán)

31 papers, graded A/B/C by evidence; five gaps G1–G5. How the system answers them:

| Finding in the literature | Design decision here |
|---|---|
| Keyword ATS beats dense retrieval on testing CVs (Resume2Vec) | Hybrid retrieval is mandatory: BM25 + dense + cross-encoder |
| LLM rankers are strong but costly and position-biased | LLM only on near-ties and on a bounded shortlist, always in both orders |
| LLM explanations hallucinate | Every quote is checked against the CV (`verify_evidence`, Q2–Q4) |
| Screening models leak protected attributes | PII masking before anything scores; counterfactual twins test it |
| Agents without control drift | A harness (guard) with schema, preconditions, HR authority and budgets |

---

## 03 · Datasets (owner: Hoa)

- **ours**: 18 JDs and a bilingual synthetic CV pool, silver labels by Claude annotators.
- **rubric set**: synthetic JDs and CVs with rubric-derived labels.
- Combined and frozen as **NLP501-bench v1** (`docs/BENCHMARK.md`): **8 dev JDs** (tuning allowed) and **10 test JDs**
  (reported once). Split by group to prevent leakage.
- Why these sources and not others: `docs/data-select-strategy.md`.

---

## 04 · Evaluation method (owners: Hoa, Thịnh)

| Aspect | Metric | Module |
|---|---|---|
| Ranking | nDCG@10 (official, test split), P@5, R@20, MRR, Kendall τ; 95% bootstrap CI, paired one-sided p | `app.eval.benchmark` |
| NLP extraction | tool P/R/F1 against generator truth; redaction leaks | `app.eval.extraction` |
| Scans | CER, tool F1 after OCR | `app.eval.ocr` |
| LLM outputs | scorecard Q1–Q17 (`docs/AI_OUTPUT_CRITERIA.md`) | `app.eval.run agent` |
| Fairness | counterfactual twins (name, gender, age, marital status) | `app.eval.robustness` |
| Safety | prompt-injection recall and false alarms | `app.eval.injection` |
| HR-in-the-loop | vague JD twins: ask HR vs not | `app.eval.clarify` |
| Agentic behaviour | P7 on HR tasks: behaviour checks, refusals, tokens | `app.eval.agentic` |
| Cost | LLM calls, tokens, latency | all of the above |

Decision rule (`docs/EVAL.md` §8): a change ships only if it is not lower by more than 0.01 on either benchmark and
higher on one.

---

## 05 · Prior results → design (owner: Hoa)

The seven findings in the deck map to the decisions in the §02 table. One finding is **not** implemented yet:
contrastive CV–JD representation learning (ConFit v2); see §Open work.

---

## 06 · Proposed method (owner: Thịnh)

### Input
1. A JD (text or file).
2. HR's goal in their own words, e.g. *"5 people strong in Playwright, banking preferred"*.
3. A CV pool (PDF, DOCX, scanned images).

### Processing

```
CV ─ NLP pipeline (backend/app/nlp, docs/NLP_PIPELINE.md) ───────────────────────────▶ CVProfile (JSON)
     read file / OCR → normalize → redact PII → neutralize injection → sections + dates
     → structural chunks → skill mentions with negation hedges, years, depth cue L1–L4

JD ─ rule parser, no LLM (nlp/jd.py) ─▶ must-have / nice-to-have / substitutes / years / domain / languages

P7 agent (agent/orchestrator.py): the LLM picks the next tool; the guard checks every call
  parse_jd → [ask_hr if the JD is vague] → screen (hybrid search → knock-out H → 8 rule criteria, 0 tokens)
  → [investigate CVs that look like parse failures] → [rank_shortlist: controlled LLM ranking, two passes]
  → explain + verify_evidence → [compare near-ties, both orders] → respond_to_hr ⇄ HR replies / re-weights / approves
```

`[ ]` marks steps the agent may skip. P6 (`agent/graph.py`) is the same work as a fixed graph, kept as the baseline.

### Output
- A proposal of N candidates in order. Each one has a score, a tier, a reason with **verified quotes**, strengths,
  and what to probe in interview.
- Every other CV gets a status and a reason (knocked out for missing X, not shortlisted, ...).
- The trace: tools run, guard refusals, tokens (Langfuse session per run).

### Where it is agentic
The LLM chooses the next step among 14 typed tools (`agent/tools.py`) toward HR's goal. It can skip steps, ask HR,
or investigate a suspicious CV. The harness keeps it safe: schema, per-tool preconditions, HR authority (only what HR
said can change requirements), budgets (`max_agent_steps`, `max_agent_tokens`, `max_hr_turns`). Argument from the
business problem: `docs/why-agentic.md`. Architecture: `docs/agentic-architecture-workflow.md`,
`docs/architecture.html`.

### Where it is NLP
CV/JD extraction (X1–X11), bilingual cue lexicons, NegEx-style negation, hybrid retrieval with a cross-encoder,
prompt-injection detection, quote grounding.

---

## Results so far (measured)

| Area | Result | Verdict |
|---|---|---|
| Ranking, official test (10 JDs) | BM25 0.695 · hybrid 0.681 · 8-axis rules ⊕ CE 0.837 · B4 fixed pipeline 0.848 · **P6 0.847** (P@5 0.88, τ 0.51) | ✅ clearly above baselines. ⚠ P6 ≈ B4 (Δ −0.000): the loops add little nDCG |
| P6 vs the rubric ranker | +0.037, p = 0.22 | ⚠ not significant with 10 JDs |
| Tool extraction | F1 0.978 (EN 0.984, VI 0.969) | ✅. ❌ low-quality CVs: F1 0.19 |
| Scanned CVs (OCR) | tool F1 0.997, CER 0.031 | ✅ |
| Evidence grounding | 100% quotes found in the CV; 92–94% supported | ✅ |
| Fairness twins | 0/600 CV×JD scores moved for name, gender, age, marital twins | ✅ but this shows masking works, not that the model is unbiased |
| Prompt injection | 10/10 attacks caught, 0/150 clean CVs flagged (threshold 0.9) | ⚠ small attack set, no held-out set yet |
| Asking HR | vague JD 0.626 → 0.756 when HR answers; agent as it runs 0.704 | ✅ the clarification loop has measurable value |
| LLM shortlist ranking | ours 0.871 → 0.955; rubric set 0.861 → 0.859 | ⚠ gain only on LLM-made labels: possibly circular |
| Single agent on its own (S1) | 0.942 vs P6 0.867 on dev, but 126k–230k tokens per JD and ranks CVs it never read | ⚠ negative result, reported as such |
| Cost | B4 0.93M tokens / 12 dev JDs; ablations A1–A4 1.32–1.44M | measured |
| **v2 test (18 JDs, + 8 real Djinni JDs)** (`BENCHMARK_V2_RESULTS.md`) | BM25 0.582 · hybrid 0.618 · 8-axis ⊕ CE 0.780 · **B4 0.787** · P6 0.782; Djinni alone: BM25 0.441 → B4 0.723, P6 0.696 | ✅ the ladder holds on real CVs; CI narrower (P6 ±0.054). ⚠ P6 ≈ B4 again (−0.005, p = 0.65) |
| NLP-stage ablation, E19 (`NLP_ABLATION_RESULTS.md`) | 18 JDs: switching off hedge scope, used/listed, depth cue or dated years moves nDCG by +0.001 to +0.013; all four off +0.021 (p = 0.986) | ❌ negative on nDCG. What the hedge scope does buy: traps@10 0.00 with it, 0.17 without (trap CVs in the top 10, ours) |
| Fine-tuned cross-encoder, E20 (`FINETUNE_RESULTS.md`, trained on Colab) | v1 test: 8-axis ⊕ CE-ft 0.877 vs 0.831 (+0.046, p = 0.003); Djinni: 0.692 vs 0.719 with the base model (−0.027); v2 all 18: +0.015, p = 0.17 | ⚠ helps on synthetic data only, does not transfer to real CVs. CV-disjoint check (G5, test JDs over held-out CVs): fused CE-ft⊥ 0.850 = leaked CE-ft 0.850 < base CE0 0.867; alone CE-ft⊥ +0.074 over CE0 (p = 0.057). The fused gain does not hold; what holds is the new CE input (CE0 fused 0.867 vs current 0.844) |
| Token cost of selectivity, E21 (`TOKEN_COST_RESULTS.md`) | vs E-every (LLM on every eligible candidate), 8 dev JDs: P6 −47.4%, B4 −66.9% tokens; E-every +0.015 nDCG over P6, p = 0.34 | ✅ selectivity halves tokens at no significant accuracy cost; ⚠ below the −78% target |
| Djinni labels (`DJINNI_LABELS.md`) | 424 pooled pairs, Opus vs Sonnet κ = 0.778. **Human labels (Đán, all 424 pairs, review mode: Opus's grade shown):** changed 73/424 Opus grades (82.8% kept, mostly 0 → 1); κ(human, silver mean) = **0.820** | ✅ κ ≥ 0.8: silver stays the main Djinni set (EVAL §2). ⚠ one annotator who saw the model's grade, so κ is an upper bound; a second, blind annotator is still needed for adjudicated gold (v2.1) |
| Djinni scored with the human labels (`BENCHMARK_V2_RESULTS.md`, v2.1 preview) | every system 0.05-0.09 lower than on silver; system order kept (Kendall τ 0.86): B3 rules ⊕ CE 0.646, B4 0.637, 8-axis ⊕ CE 0.623, P6 0.602, BM25 0.445 | ✅ conclusions hold under human labels (B4 ≥ P6; the ladder over BM25 / hybrid). ⚠ under human labels the 4-criteria ranker edges the 8-criteria one on real CVs |
| Real résumé PDFs, E23 (`KAGGLE_RESUME_RESULTS.md`) | 2,484 PDFs, 24 trades: 100% read, token F1 1.000 vs the dataset text; v4.1 fixes: dated roles 41.9% → 97.7%, "testing tool" false positives in other trades 17.2% → 1.6%, knock-out false passes 0.3% → 0.0% | ✅ found and fixed three faults synthetic data never showed; ranking on ours / coding / Djinni unchanged (±0.002) |
| Reproducibility (v4.1) | query text depended on PYTHONHASHSEED (aliases from a set): two runs of JD-002 swapped near-ties. Fixed; v1 test B3 8-axis ⊕ CE 0.831 → 0.841, dev Δ ≤ 0.005. "Present" now pinned by REFERENCE_DATE in evaluations (extraction MAE had drifted 0.79 → 0.89 with no code change) | ✅ the same JD now gives the same ranking on every run |

### Against the deck's targets
| Target in the deck | Measured | |
|---|---|---|
| nDCG@10 0.868 | 0.847 (test) | close, not reached |
| 97.6% grounded | 100% | reached |
| Kendall τ 0.81 | ~0.51 | far below |
| −78% tokens | −47.4% (P6), −66.9% (B4) vs the LLM on every candidate (E21) | not reached |

Absolute numbers are not comparable with papers (different data). What is comparable is the direction: hybrid beats
keyword-only and dense-only, and the LLM adds on top at a cost. Both match the literature.

---

## Open work — where to contribute

| # | Item | Why it matters | Suggested owner | Entry point |
|---|---|---|---|---|
| 1 | **Second, blind human annotator on Djinni** | One review-mode set is in (κ 0.820 vs silver). A blind second annotator makes the adjudicated gold (v2.1) and a human–human κ: `python -m app.eval.djinni review`, blind mode, then `djinni import` | another member (label) + Hoa (analysis) | `docs/DJINNI_LABELS.md` |
| 2 | Learned component: done as a fine-tuned cross-encoder (E20); contrastive bi-encoder still open | E20 gains on synthetic data, not on Djinni | Vũ / Đán | `app.eval.finetune`, `notebooks/finetune_ce_colab.ipynb` |
| 3 | ~~Public dataset layer~~ done: Djinni in NLP501-bench v2 | | Hoa | `docs/BENCHMARK.md` §v2 |
| 4 | Learning from HR feedback across sessions | Lecturer feedback | Sơn / Thịnh | `scoring` weight calibration |
| 5 | Live E15 run of P7 | End-to-end agentic evidence | Thịnh | `app.eval.agentic` |
| 6 | Held-out injection attack set | Current 10/10 is on a small set | anyone | `app.eval.injection` |
| 7 | ~~Token-saving comparison~~ done (E21): −47% (P6) / −67% (B4) | | Thịnh | `app.eval.token_cost` |
| 8 | Depth cues without diacritics | Unaccented Vietnamese CVs miss L1–L4 | anyone | `nlp/extract.py` |
| 9 | Low-quality CV extraction (F1 0.19) | Weakest NLP slice | anyone | `app.eval.extraction` |

Rules when contributing: tune on dev JDs only, report test once; a change is "better" only when the eval says so,
and negative results are reported too; every new LLM output needs a Q row; generated `*_RESULTS.md` files are never
edited by hand.
