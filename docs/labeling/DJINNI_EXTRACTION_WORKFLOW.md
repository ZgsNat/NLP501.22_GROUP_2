# Djinni Extraction Human Gold Workflow

Current status: **HUMAN_ANNOTATION_PENDING**.

All commands run from `backend/` with the project environment active.

1. Verify the frozen split before distributing any file:

   `python -c "from app.eval.djinni_extraction_sampling import verify_frozen_manifest; print(verify_frozen_manifest())"`

2. Build the two prediction-free kits:

   `python -m app.eval.djinni_extraction_annotation prepare-kits`

   The files appear under `.runtime/bench/djinni_extraction/labeling/`. Give kit A and kit B to two distinct people. They work independently and must not see system output, weak labels, scores, or each other's export.

3. Validate each export:

   `python -m app.eval.djinni_extraction_annotation verify human_extraction_a.json --role A`

   `python -m app.eval.djinni_extraction_annotation verify human_extraction_b.json --role B`

4. Measure pre-adjudication IAA and create the conflict-only kit:

   `python -m app.eval.djinni_extraction_annotation iaa human_extraction_a.json human_extraction_b.json`

5. Give the generated conflict HTML to a third person. After that person exports every decision, freeze Gold:

   `python -m app.eval.djinni_extraction_annotation build-gold human_extraction_a.json human_extraction_b.json human_extraction_adjudication.json`

6. Only after step 5 succeeds, run:

   `python -m app.eval.extraction_benchmark`

   `python -m app.eval.djinni_error_impact`

The builder refuses to overwrite an existing Human Gold file. Archive the Gold and its SHA-256 before any new benchmark version. The `--weak-diagnostic` option writes only to `.runtime/` and cannot update official reports.
