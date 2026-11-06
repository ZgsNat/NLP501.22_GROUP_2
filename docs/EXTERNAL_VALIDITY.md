# External Validity: Evaluation on Real Recruitment Data (Djinni QA)

**Dataset**: [Djinni Recruitment Dataset](https://aclanthology.org/2024.unlp-1.2) (UNLP 2024, MIT License).
- **8 JDs**: Real English QA Automation positions across Java, Python, JavaScript/TypeScript, and C#.
- **250 CVs**: 150 QA Automation, 70 Manual QA, 30 Developers (hard negatives) from real IT candidate profiles.
- **Human Annotators**: **Đán** (424 pairs, guided) and **Vũ** (80 pairs, blind with 79 qualitative notes).

## 1. Inter-Annotator Agreement on Real Data (Đán vs Vũ)

- **Common Graded Pairs**: 80 pairs across stacks.
- **Exact Agreement**: 77.5%
- **Quadratic-weighted Cohen's κ**: **0.865**
- **Qualitative Insight**: With strict enforcement of disqualification criteria (assigning grade 0 to profiles without automated testing experience or stack mismatch), agreement between Đán and Vũ reached strong consensus (**κ = 0.865**, 77.5% exact match, 97.5% within ±1 grade level). Vũ contributed 79 detailed rationales explaining edge cases.

## 2. Performance Comparison Across Gold Sets (TREC Pooling)

### Ground Truth: Consensus Merged Gold (424 pairs)

| System | Description | JDs | nDCG@10 | Precision@5 | Recall@20 | MRR |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Weak Labels (Candidate Keyword)** | Evaluated on judged pairs | 8 | **0.440** | 0.550 | 0.494 | 0.750 |
| **B0 BM25 (Lexical matching on real text)** | Evaluated on judged pairs | 8 | **0.443** | 0.450 | 0.483 | 0.594 |
| **B3 Criteria (Stack + Role + Depth)** | Evaluated on judged pairs | 8 | **0.473** | 0.525 | 0.556 | 0.875 |

### Ground Truth: Đán (424 pairs, model-guided)

| System | Description | JDs | nDCG@10 | Precision@5 | Recall@20 | MRR |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Weak Labels (Candidate Keyword)** | Evaluated on judged pairs | 8 | **0.438** | 0.550 | 0.477 | 0.750 |
| **B0 BM25 (Lexical matching on real text)** | Evaluated on judged pairs | 8 | **0.442** | 0.450 | 0.484 | 0.594 |
| **B3 Criteria (Stack + Role + Depth)** | Evaluated on judged pairs | 8 | **0.473** | 0.525 | 0.549 | 0.875 |

### Ground Truth: Vũ (80 pairs, blind)

| System | Description | JDs | nDCG@10 | Precision@5 | Recall@20 | MRR |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Weak Labels (Candidate Keyword)** | Evaluated on judged pairs | 8 | **0.719** | 0.475 | 1.000 | 0.651 |
| **B0 BM25 (Lexical matching on real text)** | Evaluated on judged pairs | 8 | **0.714** | 0.350 | 1.000 | 0.747 |
| **B3 Criteria (Stack + Role + Depth)** | Evaluated on judged pairs | 8 | **0.756** | 0.550 | 1.000 | 0.750 |

## 3. Key Findings for Research Presentation
1. **Generalization Beyond Synthetic Data**: Multi-criteria matching consistently outperforms lexical BM25 across all gold standards (ΔnDCG@10 = +0.031 to +0.043; MRR up to 0.875).
2. **Dual-Perspective Validation**: Having both a model-guided annotator (Đán) and an independent blind human annotator (Vũ) proves that structured criteria matching holds up under both strict and holistic evaluation regimes.
3. **Zero Data Contamination**: Neither Djinni JDs nor CVs were seen during rule construction or prompt engineering.
