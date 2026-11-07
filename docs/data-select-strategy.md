# Data selection strategy for the final evaluation

**Status (2026-11-07):** the human audit and 8-JD Djinni-QA Human Gold are measured. The frozen production
B0–B3 rerun is complete; P6 is pending after provider HTTP 503 failures. A new 6-JD × 40-CV unseen holdout is
hash-locked and fail-closed while its two blind human annotations are pending.
**Owner of the numbers:** datasets & metrics (Hoa), with architecture & experiments (Thịnh).

## 1. Why this document exists

At the final presentation (deck v7) the lecturer made two points:

- The team built and labelled its own 250 CVs. That alone did not convince the audience.
- **The system is built for HR in general, not for one company, so an existing public dataset can be used for the
  experiments.**

The team's counter-point: the project is deliberately scoped to **Automation Test roles**. The skill taxonomy, the
L1–L4 depth scale and the 8 criteria are all about test automation. A general dataset would mostly measure roles the
system was never designed to read, and dilute the result. Only a pipeline meant for every HR niche would need one.

Both points are partly right. This document settles which data each final number comes from, and why.

## 2. Start from what each number must prove

| Claim in the final report | Evidence it needs |
|---|---|
| R1 · The ranking is good *for automation-test hiring* | in-domain JDs and CVs, graded relevance |
| R2 · The result is not an artefact of our own text | real public data the team did not author; label provenance disclosed |
| R3 · The pipeline reads real CVs, not only clean synthetic ones | real CV text |
| R4 · The agent behaves well (asks, investigates, refuses, stays in budget) | scripted HR tasks (E15); no relevance labels needed |

No single dataset proves all four. So the evaluation uses layers, each tied to a claim.

## 3. What exists (updated 2026-11-07)

| Dataset | Size | Real? | Labels, and who made them | Automation-test share | Licence | Covers |
|---|---|---|---|---|---|---|
| **NLP501-bench v1** | 250 CVs, 18 JDs | synthetic | 0–3. 50 CVs: two LLM annotators plus a 120-pair blind double-human audit; 200 CVs: generator rubric | 100% | team | R1 ✅ · R2 ❌ |
| [resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit) (the most-used public set; the 93k "merged" sets build on it) | 8,000 pairs, 351 JDs, 643 resumes | real postings | Good / Potential / No Fit. **Provenance undocumented** (no dataset card) | **3 JDs (74 pairs), 32 resumes** | not stated | too few in-domain pairs to rank |
| [Djinni recruitment dataset](https://aclanthology.org/2024.unlp-1.2) (`lang-uk/recruitment-dataset-*`, UNLP 2024) | 210,250 candidate profiles + job postings | **real, anonymised** | no pair labels; every profile and job has a self-declared `Primary Keyword` | **5,756 "QA Automation" + 25,089 "QA" profiles**; jobs carry the same field | MIT | R2 ✅ · R3 ✅ (weak labels, §4) |
| datasetmaster/resumes (already used, E11) | 217 real CVs | real | none | general developer roles | MIT | R3 (done: `REALTEXT_RESULTS.md`) |
| PJB (2026), JobSearch-XS (2026) | large / 30 queries | real | expert / silver | Chinese market / general | CC BY / ? | no: language and domain |

How the share column was measured:
- We downloaded the dataset and searched its text for `SDET`, `test automation`, `QA automation`, Selenium, Cypress,
  Playwright and Appium.
- A broader search ("QA", "quality assurance", "software tester") finds 15 of 351 JDs.

**So the dilution concern is real and measurable.** The standard public set has 3 usable JDs for this task. Ranking on
all 351 would mostly test accountant and sales roles, which the taxonomy does not cover by design.

**The lecturer's concern remains relevant but is now measured.** The official 250-CV benchmark is synthetic. Hoa and
Phúc have audited its hardest 120 pairs, while Djinni-QA supplies real public text with separate human judgements.

## 4. The strategy: three layers, each for one claim

### Layer 1 · Official number (R1): NLP501-bench v1, test split, as frozen

It is the only data with graded labels for this domain, and it is already frozen with a dev/test split
(`docs/BENCHMARK.md`).

**Measured audit:** two team members graded a blind sample of 120 pairs (`docs/labeling/`) containing all 94 A/B
disagreements plus 26 agreeing controls. Human–human κ is 0.965; human–silver κ is 0.695 and 0.661 using the
pre-existing half-up rule. Therefore:
- κ ≥ 0.8: the silver labels stay the main test set.
- 0.6 ≤ κ < 0.8: they become a secondary set.
- κ < 0.6: the human labels replace them.

### Layer 2 · External check on real data (R2 + R3): Djinni-QA human-labelled pilot

- **Measured data:** 8 real QA Automation JDs, a shared pool of 250 real profiles, and 424 TREC-pooled pairs (49–59
  judged candidates per JD). Sampling seed 42 and the pool recipe are stored with the runtime dataset.
- **Human judgements:** Vũ and Đán independently grade every pooled pair 0–3. Exact agreement is 67.0%, κ = 0.696.
- **Aggregate Human Gold:** unrounded mean of the two grades; each row keeps both grades, `min`/`max` bounds and
  high/medium/low confidence. For binary P@5, both people must grade the candidate at least 2.
- **Measured now:** the pilot plus a controlled production B0–B3 run. Production nDCG@10 is 0.422, 0.464,
  0.482 and 0.636 respectively; B3 is +0.214 over B0. Min/max-grade results are sensitivity analyses, not bounds.
- **Still pending:** P6/P7. Gemini returned HTTP 503 before one JD completed; no partial score is reported.
- **Storage:** real text stays in `.runtime/`; only human judgements and their Aggregate Gold are committed.

### Layer 3 · Genuinely unseen, human-labelled holdout (locked)

- Fixed seed 5012026; 6 new QA Automation JDs and 40 new profiles, disjoint by source ID and normalized text from
  the measured pilot; 240 pairs per annotator.
- Two different blind shuffle orders; no system may materialize or evaluate the benchmark before both human files
  pass full-coverage, provenance, κ ≥ 0.6 and required adjudication checks.
- Current state is `UNLABELLED_LOCKED`, so it repairs the process leak but does not yet provide a result.

### Layer 4 · No headline number on general datasets

The report states the composition numbers above as the reason. This still answers the lecturer: **we do use existing
public data, at the slice that matches our scope.**

### What the 250 CVs keep doing

Their planted cases cannot be found in public data:
- keyword stuffing, hedged skills, title inflation;
- OCR noise, Vietnamese CVs, vague JDs.

They feed the trap metric, robustness and fairness twins (Q11), extraction F1, and the agent tasks (E15).

## 5. What goes into the final report

| Table | Data | Metric |
|---|---|---|
| Main ranking result | NLP501-bench v1 **test** + completed human audit | nDCG@10 with 95% CI, P@5, R@20, τ; κ separately |
| External validity (measured pilot) | Djinni-QA Aggregate Human Gold | nDCG@10 sensitivity bounds, conservative P@5 |
| External validity (production) | Same frozen Djinni pool | B0–B3; P6 explicitly pending after HTTP 503 |
| Leakage repair (locked) | 6 new JDs × 40 new CVs | no metric until two blind human label files open the gate |
| Agentic behaviour | E15 HR tasks (dev JDs) | checks passed, refusals, decisions, tokens |
| Real text, robustness, fairness | E11, Q11 twins, extraction | as already reported |

## 6. Next steps

1. Freeze `human_djinni_gold.json` and do not tune on its measured pilot results.
2. Preserve the completed B0–B3 production orders; resume P6 only under its frozen provider/model and report it separately.
3. Have two humans independently label all 240 frozen holdout pairs, then import labels to open the fail-closed gate.
4. Keep LLM-rank, S1 and injection results supplemental until a controlled official rerun incorporates them.

Sources: [resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit),
[Djinni candidate profiles](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english),
[Djinni dataset paper (UNLP 2024)](https://aclanthology.org/2024.unlp-1.2),
[PJB (arXiv 2603.17386)](https://arxiv.org/abs/2603.17386),
[ConFit (arXiv 2401.16349)](https://arxiv.org/pdf/2401.16349).
