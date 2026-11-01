# Data selection strategy for the final evaluation

**Status:** direction agreed in discussion on 2026-11-01. Proposed to the team; nothing in layers 2–3 is measured yet.
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
| R2 · The result is not an artefact of our own data | data and labels **the team did not make** |
| R3 · The pipeline reads real CVs, not only clean synthetic ones | real CV text |
| R4 · The agent behaves well (asks, investigates, refuses, stays in budget) | scripted HR tasks (E15); no relevance labels needed |

No single dataset proves all four. So the evaluation uses layers, each tied to a claim.

## 3. What exists (checked on 2026-11-01)

| Dataset | Size | Real? | Labels, and who made them | Automation-test share | Licence | Covers |
|---|---|---|---|---|---|---|
| **NLP501-bench v1** | 250 CVs, 18 JDs | synthetic | 0–3. 50 CVs: two LLM annotators (Opus, Sonnet). 200 CVs: a rubric written by the person who generated them. **No human labels yet** | 100% | team | R1 ✅ · R2 ❌ |
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

**The lecturer's concern is also real.** All 250 CVs are synthetic, and no human has checked their labels. A reviewer
can fairly ask whether the system just learned our own data.

## 4. The strategy: three layers, each for one claim

### Layer 1 · Official number (R1): NLP501-bench v1, test split, as frozen

It is the only data with graded labels for this domain, and it is already frozen with a dev/test split
(`docs/BENCHMARK.md`).

**Fix its weakness first:** two team members grade a stratified sample of about 120 pairs blind (`docs/labeling/`),
and we report κ(human, silver). This is the planned v1.1 (BENCHMARK.md rule 7), and the cheapest answer to "who says
these candidates are relevant?"
- κ ≥ 0.8: the silver labels stay the main test set.
- 0.6 ≤ κ < 0.8: they become a secondary set.
- κ < 0.6: the human labels replace them.

### Layer 2 · External check on real data (R2 + R3): "Djinni-QA", weak labels the team did not write

- **JDs:** about 12 Djinni postings with `Primary Keyword = QA Automation`, sampled with a fixed seed.
- **Pool per JD:** about 200 real profiles: 100 QA Automation, 50 QA (mostly manual), 50 other engineering roles.
- **Grades, from the candidates' own fields:**
  - 2 = QA Automation with at least the JD's years;
  - 1 = QA Automation with fewer years, or QA;
  - 0 = other roles.
- **Report:** B0–B3, P6 and P7 screen; nDCG@10 and P@5 over all JDs. No tuning on this set.
- **What it shows:** whether the system separates automation testers from the hardest neighbours, manual QA and
  developers who list Selenium. The labels do not come from the team or from any LLM.
- **What it does not show:** fine ordering among strong automation testers. That stays with layer 1.
- **Storage:** keep the data in `.runtime/` like E11. Repo rule: no real CVs in `data/`.

### Layer 3 · No headline number on general datasets

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
| Main ranking result | NLP501-bench v1 **test** (+ κ with human labels when ready) | nDCG@10 with 95% CI, P@5, R@20, τ |
| External validity | Djinni-QA (weak labels) | nDCG@10, P@5 for B0–B3, P6, P7 |
| Agentic behaviour | E15 HR tasks (dev JDs) | checks passed, refusals, decisions, tokens |
| Real text, robustness, fairness | E11, Q11 twins, extraction | as already reported |

## 6. Next steps

1. Human labelling of about 120 stratified pairs → κ → benchmark v1.1 (layer 1).
2. `app/eval/djinni.py`: fetch, sample with a fixed seed, weak labels, run the ladder (layer 2).
3. Run E15 (`python -m app.eval.agentic`) for the agentic table.
4. Report test once, after all tuning on dev (BENCHMARK.md rule 4).

Sources: [resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit),
[Djinni candidate profiles](https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english),
[Djinni dataset paper (UNLP 2024)](https://aclanthology.org/2024.unlp-1.2),
[PJB (arXiv 2603.17386)](https://arxiv.org/abs/2603.17386),
[ConFit (arXiv 2401.16349)](https://arxiv.org/pdf/2401.16349).
