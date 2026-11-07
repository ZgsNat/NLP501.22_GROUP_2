# Djinni-QA production ladder on Aggregate Human Gold

Frozen protocol: `data/eval/djinni_production_v1.json`. No tuning after the freeze. Rankings are filtered to judged pairs before scoring; unjudged profiles are ignored, never assigned zero relevance.

| System | nDCG@10 on min grades | nDCG@10 Aggregate Gold | nDCG@10 on max grades | P@5 both≥2 | τ-b | Δ vs B0 | p(first>B0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 BM25 | 0.360 [0.264, 0.452] | **0.422 [0.320, 0.515]** | 0.475 [0.359, 0.590] | 0.325 [0.175, 0.500] | 0.170 [0.077, 0.261] | +0.000 | 1.000 |
| B1 dense | 0.437 [0.363, 0.523] | **0.464 [0.399, 0.540]** | 0.468 [0.389, 0.544] | 0.475 [0.300, 0.675] | 0.172 [0.119, 0.226] | +0.042 | 0.157 |
| B2 hybrid | 0.441 [0.346, 0.536] | **0.482 [0.399, 0.572]** | 0.506 [0.405, 0.609] | 0.525 [0.350, 0.700] | 0.192 [0.101, 0.279] | +0.060 | 0.019 |
| B3 8-axis ⊕ CE [hybrid] | 0.612 [0.514, 0.717] | **0.636 [0.554, 0.725]** | 0.605 [0.541, 0.675] | 0.650 [0.450, 0.850] | 0.400 [0.329, 0.460] | +0.214 | 0.000 |
| P6 | pending | pending | pending | pending | pending | — | — |

The min/max-grade columns are label-sensitivity analyses, not mathematical lower/upper bounds on nDCG: changing grades also changes the ideal DCG denominator.

Per-JD orders are persisted under `.runtime/eval/results_djinni/`; the report stores an order hash for auditability. Missing systems remain `pending` and are never substituted with pilot heuristics.

P6 status: two Gemini 3.8 Flash attempts stopped before the first completed JD after HTTP 503 high-demand responses. No partial score was promoted and B0–B3 were not rerun.
