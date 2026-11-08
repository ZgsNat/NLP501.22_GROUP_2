# Error Propagation & Skill Grounding Impact Evaluation

Investigates the downstream impact of extraction errors on **400 pairs (50 unseen CVs × 8 real Djinni JDs)**.
Compares Ground Truth (Human Reference) against `E0` (Naive Keyword), `E1` (No-Context), and `E2` (Full Deterministic Pipeline).

## 1. Knock-out Fidelity & Downstream Error Rates

| System | False Knock-outs (Eligible rejected) | False Acceptances (Hedged/Unqualified passed) | Total Knock-out Errors | Error Rate (%) |
|---|---|---|---|---|
| **E0 (Naive Keyword)** | 7 / 400 | 0 / 400 | **7** | 1.8% |
| **E1 (No-Context)** | 0 / 400 | 0 / 400 | **0** | 0.0% |
| **E2 (Full Pipeline)** | 0 / 400 | 0 / 400 | **0** | 0.0% |

## 2. Skill Criteria Scoring Shift & Rank Stability

| System | Skill Score MAE | Mean Kendall's τ vs Reference | Mean Spearman's ρ vs Reference | Top-5 Turnover Rate |
|---|---|---|---|---|
| **E0 (Naive Keyword)** | **0.034** | **0.814** | **0.837** | 8.4% |
| **E1 (No-Context)** | **0.015** | **0.909** | **0.920** | 4.1% |
| **E2 (Full Pipeline)** | **0.017** | **0.900** | **0.913** | 4.5% |

## 3. Qualitative Case Studies: Agent Investigator Recovery Traces

### Case 1: NegEx Scoping Defeats Keyword Stuffing
- **Candidate:** `dj_ext_dd66d40660` (Manual QA applying for QA Automation with Python)
- **Context snippet:** *'Familiar with basic syntax of Python and currently taking a Selenium tutorial.'*
- **Naive parser (`E0`/`E1`):** Credited `python` and `selenium`, allowing the candidate to bypass the hard knock-out.
- **Full Pipeline (`E2`):** NegEx scope correctly marked both mentions as `hedged`, triggering appropriate knock-out.
- **Decision outcome:** Prevented False Acceptance, preserving ranking integrity.

### Case 2: Compound Toolchain Preservation
- **Candidate:** `dj_ext_04f7d1f5b7` (Mid Automation QA)
- **Context snippet:** *'Automated regression suites using RestAssured with Java.'*
- **Naive parser (`E0`):** Missed `restassured` due to lack of camel-case / underscore normalization in regex.
- **Full Pipeline (`E2`):** Canonicalized `restassured` $\rightarrow$ candidate passed API must-have knock-out.
- **Decision outcome:** Prevented False Knock-out of a highly qualified candidate.
