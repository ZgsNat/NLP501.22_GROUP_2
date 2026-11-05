# Human Validation of Silver Labels (NLP501-bench v1.1)

**Sample**: 120 pairs (stratified seed 13, incorporating **all 94 silver disagreements**).
**Annotators**: Hoa Reviewer, Phuc.
**Decision Rule (EVAL.md §2)**: Quadratic-weighted Cohen's κ with silver:
- κ ≥ 0.8: Silver set retained as primary test ground truth.
- 0.6 ≤ κ < 0.8: Silver set kept as secondary test set; human labels serve as verified validation benchmark.
- κ < 0.6: Human labels replace silver set entirely.

## Agreement & Validation Matrix

| Evaluation Pair | Pairs | Exact Agreement | Quadratic Weighted κ | Interpretation |
|---|:---:|:---:|:---:|---|
| Hoa Reviewer vs Phuc | 120 | 90.8% | **0.965** | Near-perfect human consensus |
| Hoa Reviewer vs annotator A (Opus) | 120 | 70.8% | **0.869** | Strong agreement |
| Hoa Reviewer vs annotator B (Sonnet) | 120 | 37.5% | **0.687** | Moderate / secondary set threshold |
| Hoa Reviewer vs silver gold (mean, rounded) | 120 | 76.7% | **0.852** | Strong agreement |
| Phuc vs annotator A (Opus) | 120 | 78.3% | **0.879** | Strong agreement |
| Phuc vs annotator B (Sonnet) | 120 | 30.0% | **0.651** | Moderate / secondary set threshold |
| Phuc vs silver gold (mean, rounded) | 120 | 82.5% | **0.855** | Strong agreement |

## Disagreements Adjudication Coverage

| Human Annotator | Disagreements Covered | Total Disagreements in Corpus | Status |
|---|:---:|:---:|:---:|
| Hoa Reviewer | 94 | 94 | **100% Fully Adjudicated** |
| Phuc | 94 | 94 | **100% Fully Adjudicated** |
