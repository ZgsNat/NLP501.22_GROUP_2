# Extraction-Induced Knockout & Skill Grounding Impact Evaluation

Investigates the downstream impact of extraction errors on **400 pairs (50 unseen CVs × 8 real Djinni JDs)**.
Compares Skill Oracle Reference against `E0` (Naive Keyword), `E1` (No-Context), and `E2` (Full Deterministic Pipeline).

## 1. Extraction-Induced Knockout Fidelity

- **Total Candidate-Job Pairs:** 400
- **Eligible Pairs under Skill Reference:** 308 | **Ineligible Pairs:** 92

| System | False Knockouts (Eligible excluded) | False Exclusion Rate (%) | False Acceptances (Ineligible passed) | Skill Score MAE | Mean Kendall's τ vs Reference |
|---|---|---|---|---|---|
| **E0 (Naive Keyword)** | 7 / 308 | **2.3%** | 0 / 92 | **0.034** | **0.814** |
| **E1 (No-Context)** | 0 / 308 | **0.0%** | 0 / 92 | **0.015** | **0.909** |
| **E2 (Full Pipeline)** | 0 / 308 | **0.0%** | 0 / 92 | **0.017** | **0.900** |

## 2. Key Observations

1. **Elimination of False Exclusions:**
   - Naive keyword matching (`E0`) falsely knocks out **7 eligible candidates (2.3%)** due to lack of compound term normalization (e.g. `RestAssured`).
   - Full Pipeline (`E2`) achieves **0 false exclusions (0.0%)**, preserving qualified candidates for ranking.

2. **Rank Stability:**
   - The rank correlation between Skill Oracle scoring and Pipeline `E2` scoring reaches Kendall's **τ = 0.900**.

3. **Qualitative Trace (Investigator Agent Behavior):**
   - On noisy profiles where text extraction flags OCR/token anomalies, the Investigator agent (`agent/investigate.py`, N12) inspects unindexed passages to verify candidate claims before finalizing rejection decisions.
