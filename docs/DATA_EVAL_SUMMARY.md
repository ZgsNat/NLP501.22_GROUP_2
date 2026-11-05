# Data evaluation summary: benchmark integrity and human labels

Status on 2026-11-06. Every number here comes from a generated file; regenerate those, not this page:

| Layer | Command | Generated file |
|---|---|---|
| Frozen benchmark integrity | `cd backend && uv run python -m app.eval.benchmark verify` | — (prints PASS / the changed files) |
| Human labels on `ours` (silver check) | `cd backend && uv run python -m app.eval.human score` | [HUMAN_LABELS.md](HUMAN_LABELS.md) |
| Human labels on Djinni (real CVs) | `cd backend && uv run python -m app.eval.djinni score` | [DJINNI_LABELS.md](DJINNI_LABELS.md) |
| Rankings on v2 (incl. Djinni) | `cd backend && uv run python -m app.eval.benchmark_v2 report` | [BENCHMARK_V2_RESULTS.md](BENCHMARK_V2_RESULTS.md) |

## 1. Human labels on the synthetic benchmark (`ours`)

Hoa and Phúc graded the same 120 pairs **blind** (`docs/labeling/`, sample `v2-all-disagreements-seed13`: all 94 pairs
where the two silver annotators disagree, plus 26 agreeing controls). They never saw a silver grade or a ranking.

- **Human–human:** 90.8% exact, weighted κ **0.965**.
- **Human vs each silver annotator:** Opus (A) κ 0.869 / 0.879; Sonnet (B) κ 0.687 / 0.651.
- **Human vs the silver mean depends on rounding.** 94 of 120 pairs are disagreements, where the mean is x.5:
  rounded half up κ 0.695 / 0.661, rounded down κ 0.870 / 0.879. Neither number alone is the answer.
- **Without rounding:**
  - On the 26 agreeing controls, both humans give the silver grade on 20/26.
  - On the 94 disagreements:
    - Opus is the stricter annotator on 90/94.
    - The humans side with Opus on 65 (Hoa) and 74 (Phúc), and with Sonnet on 25 and 16.
    - Their grade lies between the two on 90/94.

**Reading.** The silver set is right where Opus and Sonnet agree. Where they differ, the gold (their mean) is about
half a grade too generous, and humans side with the stricter annotator.

Under EVAL.md §2, κ against the half-up mean (0.66–0.70) would make silver a secondary set, and κ against the
rounded-down mean or against Opus (≥ 0.87) would keep it as the main set. The sample is built from disagreements, so
neither κ estimates agreement over all 600 pairs.

**To confirm with the annotators before the report calls this independent:**
- `data/eval/labeling_reference/pairs_reference.json` holds the silver grades and the agree/disagree status of all
  120 pairs, in the same folder as the labelling guideline (`GUIDELINES.md`). The labels are blind only if neither
  annotator opened it while grading.
- The guideline names the trap CVs (CV-004–006), which primes the graders towards the rubric.
- This page, unlike the Djinni page, does not stamp a time per grade, so independence and reading time cannot be
  checked from the files. κ 0.965 between two humans (109/120 identical) is high enough to be worth asking about.

**Open:** re-score v1 with the 120 human-graded pairs in place of the silver mean (an adjudicated v1.1 gold) and check
whether the system order holds.

## 2. Human labels on Djinni (real CVs)

| Annotator | Pairs | Mode | Median s / pair | Quality check |
|---|---|---|---|---|
| Đán | 424 | review (saw Opus's grade) | 1.4 | ✅ usable; κ vs silver mean 0.820 |
| Vũ | 80 (stratified subset) | blind | 29.5 (51 notes) | ❌ excluded: best κ with any other annotator 0.25 < 0.4 |
| an earlier blind attempt | — | blind | — | ❌ excluded, set aside |

- Đán vs Vũ on their 80 shared pairs: 32.5% exact, κ 0.164.
- Vũ also disagrees with both LLM annotators: κ 0.17 vs Opus, 0.26 vs Sonnet.
- Vũ's labels are read carefully (30 s per pair, notes). The low agreement points to a different reading of the
  rubric on real profiles, or to profiles too short to grade, not to careless work. That is worth one adjudication
  session before anyone labels more.

**What this means for the report:**
- The Djinni gold in use is still the **silver** set.
- The only human set that passes the check (Đán's) was made while seeing Opus's grade, so it is weak independent
  evidence.
- There is no adjudicated human gold (v2.1) yet.
- Djinni rankings are the ones in BENCHMARK_V2_RESULTS.md (P6 0.696, B4 0.723 on silver; 0.602 / 0.637 on Đán's
  labels). A system's rank on Djinni is not computed anywhere else.

## 3. Frozen benchmark

`app.eval.benchmark verify` checks every input of NLP501-bench v1 against its SHA-256 manifest
(`data/eval/benchmark_v1.json`). The human labels above add files to `data/eval/` and do not touch the frozen inputs.
