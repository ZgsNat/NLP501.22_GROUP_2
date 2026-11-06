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

**Assumptions.** Ranking labels are silver (LLM-made) and rubric-based. Two humans graded 120 pairs of `ours` blind
and one human graded Djinni while seeing the model's grade (`docs/DATA_EVAL_SUMMARY.md`); no human-adjudicated gold
replaces silver yet.

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
  parse_jd → [HR wrote a goal: formalize_goal (each constraint quotes HR) → define_concept for what no taxonomy
  holds → feasibility over the pool → ask_hr if it is infeasible or unclear] → [ask_hr if the JD is vague] → screen (hybrid search → knock-out H → 8 rule criteria, 0 tokens)
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
The LLM chooses the next step among 17 typed tools (`agent/tools.py`) toward HR's goal. It can skip steps, ask HR,
or investigate a suspicious CV. A complex goal ("5 CVs, Selenium a must, Playwright nice, AI a must, AI in
automation a big plus") becomes a typed plan: code checks that every constraint quotes HR (Q19), counts what the
pool can deliver before ranking, and the agent must go back to HR when the goal cannot be met and declare in its
answer what it could not deliver (Q20, H31/H32; `agent/goal.py`). The harness keeps it safe: schema, per-tool preconditions, HR authority (only what HR
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
| Ranking, official test (10 JDs) | BM25 0.695 · hybrid 0.681 · 8-axis rules ⊕ CE 0.841 · B4 fixed pipeline 0.839 · **P6 0.851** (P@5 0.88, τ 0.51); re-run after v4.1 (31 Oct table: 0.837 / 0.848 / 0.847) | ✅ clearly above baselines. ⚠ P6 ≈ B4 (+0.012, p = 0.064): the gap is the vague `ambiguous_senior`, where the agent asks HR (0.68 vs 0.58) |
| P6 vs the rubric ranker | +0.041, p = 0.12 | ⚠ not significant with 10 JDs |
| Tool extraction | F1 0.978 (EN 0.984, VI 0.969) | ✅. ❌ low-quality CVs: F1 0.19 |
| Scanned CVs (OCR) | tool F1 0.997, CER 0.031 | ✅ |
| Evidence grounding | 100% quotes found in the CV; 92–94% supported | ✅ |
| Fairness twins | 0/600 CV×JD scores moved for name, gender, age, marital twins | ✅ but this shows masking works, not that the model is unbiased |
| Prompt injection | 10/10 attacks caught, 0/150 clean CVs flagged (threshold 0.9) | ⚠ small attack set, no held-out set yet |
| Asking HR | vague JD 0.626 → 0.756 when HR answers; agent as it runs 0.704 | ✅ the clarification loop has measurable value |
| LLM shortlist ranking | ours 0.871 → 0.955; rubric set 0.861 → 0.859 | ⚠ gain only on LLM-made labels: possibly circular |
| Single agent on its own (S1) | 0.942 vs P6 0.867 on dev, but 126k–230k tokens per JD and ranks CVs it never read | ⚠ negative result, reported as such |
| P7 agent on HR tasks, E15 (`AGENTIC_RESULTS.md`, live, Claude via claude_code) | **40/42 behaviour checks** over 11 runs (6 dev JDs + 5 HR conversations, latest run after the 2026-11-06 harness fixes): screens before answering, explains every recommended CV, asks HR on the vague JD (L1), adds Docker as a must-have when told, records a rejection and drops the CV, refuses an age/gender filter (guard Q5). Misses: 'why' was answered from rank_shortlist's reasons without a compare (check kept as written); 'require' took Docker through formalize_goal, a path the check now accepts. The run before missed JD-008 (empty shortlist) and JD-012 (needless question); both have a guard now and passed. nDCG@10 on dev: P7 0.953 vs P6 0.867; 113k–265k tokens per run | ✅ measured agentic behaviour within the harness. ⚠ the nDCG gain comes from `rank_shortlist`, whose gain was only on LLM-made labels (E18): possibly circular. Found and fixed on the way: a rejected #1 stayed #1 (now `rejected_by_hr`), and the claude_code workdir inside the repo let the developer's own Claude memory into the system under evaluation (now outside any repository) |
| P7 on complex HR goals, E26 (`GOAL_RESULTS.md`, live, Claude via claude_code; 5 goals on data/cvs + 6 goal-suite CVs that mention AI at different levels) | **81/83 behaviour checks over 3 runs per goal** (round 5, cache off; round 4 27/28 on one run): formalizes the goal quoting HR, defines 'AI in automation' as a concept graded by code, goes back to HR when the goal cannot be met (Cypress + Appium: 1 CV of 5), recommends only CVs with HR's must-haves, declares the shortfall. The unneeded question on the feasible goal is gone (15/15 after the harness fix). Misses: 2 token-budget stops, traced to a harness contradiction (open questions held after HR could no longer be asked), fixed. Re-run of those two goals (`GOAL_RESULTS_round5b.md`): `exclude` 12/12; `vague_ai` 0/3 within budget on the same tool path (~150k orchestrator tokens, the edge of the budget), and the fallback now recommends the candidates `explain` verified instead of nobody. 88k–248k tokens per goal | ✅ goal-driven control measured with a pass rate. ⚠ scripted HR |
| Cost | B4 0.93M tokens / 12 dev JDs; ablations A1–A4 1.32–1.44M | measured |
| **v2 test (18 JDs, + 8 real Djinni JDs)** (`BENCHMARK_V2_RESULTS.md`) | BM25 0.582 · hybrid 0.618 · 8-axis ⊕ CE 0.780 · **B4 0.787** · P6 0.782; Djinni alone: BM25 0.441 → B4 0.723, P6 0.696 | ✅ the ladder holds on real CVs; CI narrower (P6 ±0.054). ⚠ P6 ≈ B4 again (−0.005, p = 0.65) |
| NLP-stage ablation, E19 (`NLP_ABLATION_RESULTS.md`) | 18 JDs: switching off hedge scope, used/listed, depth cue or dated years moves nDCG by +0.001 to +0.013; all four off +0.021 (p = 0.986) | ❌ negative on nDCG. What the hedge scope does buy: traps@10 0.00 with it, 0.17 without (trap CVs in the top 10, ours) |
| Fine-tuned cross-encoder, E20 (`FINETUNE_RESULTS.md`, trained on Colab) | v1 test: 8-axis ⊕ CE-ft 0.877 vs 0.831 (+0.046, p = 0.003); Djinni: 0.692 vs 0.719 with the base model (−0.027); v2 all 18: +0.015, p = 0.17 | ⚠ helps on synthetic data only, does not transfer to real CVs. CV-disjoint check (G5, test JDs over held-out CVs): fused CE-ft⊥ 0.850 = leaked CE-ft 0.850 < base CE0 0.867; alone CE-ft⊥ +0.074 over CE0 (p = 0.057). The fused gain does not hold; what holds is the new CE input (CE0 fused 0.867 vs current 0.844) |
| Token cost of selectivity, E21 (`TOKEN_COST_RESULTS.md`) | vs E-every (LLM on every eligible candidate), 8 dev JDs: P6 −47.4%, B4 −66.9% tokens; E-every +0.015 nDCG over P6, p = 0.34 | ✅ selectivity halves tokens at no significant accuracy cost; ⚠ below the −78% target |
| Djinni labels (`DJINNI_LABELS.md`) | 424 pooled pairs, Opus vs Sonnet κ = 0.778. **Human labels (Đán, all 424 pairs, review mode: Opus's grade shown):** changed 73/424 Opus grades (82.8% kept, mostly 0 → 1); κ(human, silver mean) = **0.820** | ✅ κ ≥ 0.8: silver stays the main Djinni set (EVAL §2). ⚠ one annotator who saw the model's grade, so κ is an upper bound; a second, blind annotator is still needed for adjudicated gold (v2.1) |
| Djinni scored with the human labels (`BENCHMARK_V2_RESULTS.md`, v2.1 preview) | every system 0.05-0.09 lower than on silver; system order kept (Kendall τ 0.86): B3 rules ⊕ CE 0.646, B4 0.637, 8-axis ⊕ CE 0.623, P6 0.602, BM25 0.445 | ✅ conclusions hold under human labels (B4 ≥ P6; the ladder over BM25 / hybrid). ⚠ under human labels the 4-criteria ranker edges the 8-criteria one on real CVs |
| Annotator quality check (`DJINNI_LABELS.md`) | Human labels count towards gold only with weighted κ ≥ 0.4 against another annotator and, when blind, ≥ 10 s per pair (the page stamps every grade). A first blind attempt failed it and was set aside. Vũ's blind 80-pair subset (29.5 s per pair, 51 notes) failed it too: κ 0.164 with Đán, 0.17–0.26 with the LLM annotators. Đán's review-mode labels: median 1.4 s per pair, i.e. mostly Opus's grade accepted | ⚠ the only human set is anchored on Opus, so the κ 0.820 and the human-label preview are weak human validation, not independent evidence; no adjudicated v2.1 gold. A careful blind reader disagreeing with everyone says the Djinni rubric needs one adjudication session before more labelling |
| Human labels on `ours` (`HUMAN_LABELS.md`, Hoa + Phúc, blind, 120 pairs = all 94 Opus/Sonnet disagreements + 26 controls) | human–human κ **0.965**; vs Opus 0.87–0.88, vs Sonnet 0.65–0.69; vs the silver mean 0.66–0.70 rounded half up, 0.87–0.88 rounded down. Controls: silver grade on 20/26. Disagreements: Opus stricter on 90/94; humans side with Opus 65 / 74, Sonnet 25 / 16 | ✅ first blind human check of the silver labels: right where the two models agree. Blind, confirmed by the team. ⚠ where they differ the gold mean is ~½ grade too generous; the sample is disagreement-heavy, so κ is not over all 600 pairs. Next: re-score v1 with these 120 pairs as adjudicated gold |
| v1.1: v1 re-scored with those 120 human pairs, E27 (`BENCHMARK_V1_1_RESULTS.md`, no LLM) | Order of the 8 systems identical on test and dev (Kendall τ 1.00); test nDCG@10 moves −0.002 to +0.007; P6 0.851 → 0.853, B3 8-axis 0.841 → 0.841, rubric ranker 0.810 → 0.817 | ✅ the silver labels hold for comparing systems. ⚠ the stored rankings were re-run after v4.1 and no longer match the 31 Oct v1 report (P6 0.847 → 0.851, B4 0.848 → 0.839; P6 − B4 now +0.012, p = 0.064) |
| Real résumé PDFs, E23 (`KAGGLE_RESUME_RESULTS.md`) | 2,484 PDFs, 24 trades: 100% read, token F1 1.000 vs the dataset text; v4.1 fixes: dated roles 41.9% → 97.7%, "testing tool" false positives in other trades 17.2% → 1.6%, knock-out false passes 0.3% → 0.0% | ✅ found and fixed three faults synthetic data never showed; ranking on ours / coding / Djinni unchanged (±0.002) |
| Reproducibility (v4.1) | query text depended on PYTHONHASHSEED (aliases from a set): two runs of JD-002 swapped near-ties. Fixed; v1 test B3 8-axis ⊕ CE 0.831 → 0.841, dev Δ ≤ 0.005. "Present" now pinned by REFERENCE_DATE in evaluations (extraction MAE had drifted 0.79 → 0.89 with no code change) | ✅ the same JD now gives the same ranking on every run |

### Against the deck's targets
| Target in the deck | Measured | |
|---|---|---|
| nDCG@10 0.868 | 0.851 (test) | close, not reached |
| 97.6% grounded | 100% | reached |
| Kendall τ 0.81 | ~0.51 | far below |
| −78% tokens | −47.4% (P6), −66.9% (B4) vs the LLM on every candidate (E21) | not reached |

Absolute numbers are not comparable with papers (different data). What is comparable is the direction: hybrid beats
keyword-only and dense-only, and the LLM adds on top at a cost. Both match the literature.

---

## Open work — where to contribute

| # | Item | Why it matters | Suggested owner | Entry point |
|---|---|---|---|---|
| 1 | **Djinni human gold**: Vũ's blind 80-pair run is in but fails the check (κ 0.164 with Đán) | Before more labelling: one session where Đán and Vũ go through ~10 of their disagreements and fix how the rubric reads short real profiles; then re-grade the subset blind | Đán + Vũ (label), Hoa (analysis) | `docs/DJINNI_LABELS.md`, `docs/DATA_EVAL_SUMMARY.md` |
| 1b | ~~Re-score v1 with the 120 human-graded pairs~~ done (E27, `BENCHMARK_V1_1_RESULTS.md`): system order identical on test and dev (τ 1.00), test nDCG moves ≤ 0.007; the rubric ranker gains most, so the deterministic lead narrows +0.031 → +0.024 | | Hoa | `app.eval.benchmark_v1_1` |
| 2 | Learned component: done as a fine-tuned cross-encoder (E20); contrastive bi-encoder still open | E20 gains on synthetic data, not on Djinni | Vũ / Đán | `app.eval.finetune`, `notebooks/finetune_ce_colab.ipynb` |
| 3 | ~~Public dataset layer~~ done: Djinni in NLP501-bench v2 | | Hoa | `docs/BENCHMARK.md` §v2 |
| 4 | ~~Learning from HR feedback across sessions~~ built (H30, off by default) and measured (E24): with a simulated, perfectly consistent HR, learned weights vs static Δ −0.008 nDCG@10 (p = 0.78) | Negative result: re-weighting from HR decisions does not raise nDCG; HR rejections are kept as hard decisions instead (`rejected_by_hr`) | Sơn / Thịnh | `agent/hr_memory.py`, `HR_LEARNING_RESULTS.md` |
| 5 | ~~Live E15 run of P7~~ done: 40/42 checks (`AGENTIC_RESULTS.md`). Guard done: a first answer that names ranked CVs must put them in the shortlist; a follow-up or a budget stop keeps the shortlist HR already has. E26 already reports a pass rate per check over 3 runs; E15 does not yet | | Thịnh | `app.eval.agentic` |
| 6 | Held-out injection attack set | Current 10/10 is on a small set | anyone | `app.eval.injection` |
| 7 | ~~Token-saving comparison~~ done (E21): −47% (P6) / −67% (B4) | | Thịnh | `app.eval.token_cost` |
| 8 | ~~Depth cues without diacritics~~ done (v4.2): unaccented Vietnamese CVs 0.0% → 87.5% depth accuracy, = accented; no ranking change | | anyone | `nlp/extract.py`, `EXTRACTION_RESULTS.md` |
| 9 | Low-quality CV extraction (parser F1 0.157) | Garbled CVs are N12's job by design. Measured (v4.2): even if the LLM proposed every near-miss N12's search finds, `verify` keeps 11/45 gold tools: F1 0.157 → 0.367 at most. The generator replaces ~40% of letters; loosening the similarity bar costs precision on clean CVs. Next step would be a live N12 run on these 7 CVs | anyone | `app.eval.extraction`, `agent/investigate.py` |

Rules when contributing: tune on dev JDs only, report test once; a change is "better" only when the eval says so,
and negative results are reported too; every new LLM output needs a Q row; generated `*_RESULTS.md` files are never
edited by hand.
