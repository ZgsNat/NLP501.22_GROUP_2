# Djinni-QA unseen holdout protocol

**Current status (2026-11-07): `UNLABELLED_LOCKED`.** Preparation and hash verification pass: 6 new JDs, 40 new
CVs, and 240 pairs per annotator. Source IDs and normalized texts are disjoint from the measured 8-JD/250-CV pilot.
The evaluation gate was tested and refuses access before Human Gold exists.

This holdout repairs the fact that the synthetic test JDs and the first
Djinni-QA pilot had already been viewed before the final evaluation was
locked.  It is a **new test set**, not another development set.

## Frozen design

- Source: the original MIT-licensed Djinni Recruitment Dataset parquet files.
- Fixed seed: `5012026`.
- Six new `QA Automation` JDs, with distinct source IDs from the measured
  8-JD pilot and a best-effort spread across Java, Python, JavaScript and C#.
- One shared pool of 40 new profiles: 24 QA Automation, 10 QA and 6 nearby
  developer/DevOps negatives.  All 40 source IDs are outside the measured
  250-profile pool.
- 240 JD–CV pairs per annotator.  Two people label independently and blind on
  the 0–3 relevance scale.  Annotators A and B receive different random order.
- No source keyword, system score, ranking, prior label or other annotator's
  answer is shown in the labeling interface.
- Real text and the two HTML annotation kits stay in `.runtime/`; Git stores
  only the input/source hashes and protocol manifest.

The immutable manifest is `data/eval/djinni_holdout_v1.json`.  Re-running
`prepare` after that file exists fails rather than silently resampling v1.

## Commands

Prepare once from the two downloaded official parquet files:

```bash
cd backend
uv run --with pyarrow python -m app.eval.djinni_holdout prepare
uv run python -m app.eval.djinni_holdout verify
```

The blind kits are written to:

```text
.runtime/holdout/labeling/annotator_A.html
.runtime/holdout/labeling/annotator_B.html
```

After both humans export complete JSON files:

```bash
uv run python -m app.eval.djinni_holdout import-labels A.json B.json
```

Gold is created only if both files cover exactly the frozen 240 pairs,
explicitly record `human=true`, `independent=true` and
`saw_system_output=false`, match the frozen manifest, and achieve quadratic
weighted κ ≥ 0.6.  Every disagreement of two or more grades requires an
independent adjudication file before Gold can be written.

## Fail-closed evaluation gate

`python -m app.eval.run ... --bench djinni_holdout` and corpus materialization
refuse to run while `human_djinni_holdout_gold.json` does not exist or any
frozen input hash differs.  Therefore the group cannot inspect a system result,
tune on it and later call the same data “unseen”.

Once Gold opens the gate, run every pre-registered system once with the same
model/configuration used for the production Djinni evaluation.  Report the
holdout as a separate confirmation table; never merge it into dev or the
already viewed 8-JD pilot.
