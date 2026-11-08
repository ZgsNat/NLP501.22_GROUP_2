"""Every number on the defense deck, with its formula, its source file and the command that produces it.

The deck is not a source: each number here points at a generated results file (or the report section that quotes
one), and `verify` checks that the quoted text is still in that file, so a regenerated result that changed a number
fails loudly instead of leaving a stale slide.

    python -m app.eval.defense_numbers            → docs/DEFENSE_NUMBERS.md (+ verify)
    python -m app.eval.defense_numbers verify     → exit 1 if any number is no longer in its source
    python -m app.eval.defense_numbers json PATH  → per-slide footnotes for the deck builder
"""

import json
import sys
from dataclasses import dataclass, replace

from app.config import ROOT

OUT = ROOT / "docs" / "DEFENSE_NUMBERS.md"

# Formulas shared by many rows (implemented in app/eval/metrics.py).
NDCG = "nDCG@10 = DCG@10 / IDCG@10, DCG = Σ_{i=1..10} (2^g_i − 1) / log2(i + 1), g = nhãn 0–3; trung bình trên các JD"
DELTA = "Δ = mean_JD(A) − mean_JD(B) trên cùng các JD; p = one-sided paired bootstrap, 5.000 lần lấy mẫu lại JD, tỉ lệ lần Δ ≤ 0"
CI = "95% CI = percentile bootstrap (2.5%, 97.5%) trên các JD, 5.000 lần"
KAPPA = "κ có trọng số bậc hai = 1 − Σ w_ij·O_ij / Σ w_ij·E_ij, w_ij = (i − j)², O = bảng chéo quan sát, E = kỳ vọng ngẫu nhiên"
F1 = "P = TP/(TP+FP), R = TP/(TP+FN), F1 = 2PR/(P+R); so tập công cụ trích được với tập công cụ thật của từng CV, cộng dồn TP/FP/FN"
RATE = "tỉ lệ = số đầu ra LLM vi phạm / tổng số đầu ra, cộng dồn trên mọi JD của ours và coding; lần đầu → sau khi regenerate"

RUN_V1 = "cd backend && uv run python -m app.eval.benchmark report"
RUN_EVAL = "cd backend && uv run python -m app.eval.run agent (rồi `report`)"


@dataclass(frozen=True)
class Num:
    slide: str  # deck v13 slide title (prefix)
    value: str  # as printed on the slide
    meaning: str
    formula: str
    src: str  # file under the repo root
    pat: str  # literal text that must be in src
    cmd: str = ""
    data: str = ""  # the datasets the number is computed from, relative to the repo root
    calc: str = ""  # the formula with the actual inputs filled in


S4, S5, S8, S9 = "Rule-based scoring", "Kiến trúc tổng thể", "Information extraction", "Hybrid retrieval"
S11, S12, S13, S14 = "P6: sáu bounded loop", "20 tiêu chí guardrail", "Dữ liệu và benchmark", "Đánh giá chất lượng NLP"
S15, S16, S17, S18 = "Kết quả trên benchmark", "External validation", "Đóng góp của agent", "Hai lượt chạy thật"
S19, S20, S21 = "So sánh với các nghiên cứu", "Kiểm định giả thuyết", "Hạn chế của nghiên cứu"
S22, S23 = "Mức độ giải quyết", "Kết luận"

NUMS: list[Num] = [
    # Scoring
    Num(S4, "0.924", "điểm cuối của CV-033 với JD-010", "Σ_k w_k·S_k/100 = 0.184+0.150+0.100+0.100+0.090+0.300",
        "docs/FINAL_REPORT.md", "| **total** | | | **0.924** | |", "app/agent/scoring.py › explanation card"),
    Num(S4, "trọng số 4 profile", "w theo loại doanh nghiệp, tổng = 1", "cấu hình tay, không học từ dữ liệu",
        "data/criteria_profiles.yaml", "outsourcing:"),
    Num(S4, "0.855", "what-if: CV-040 nếu có Selenium", "chấm lại cùng luật sau khi thêm Selenium vào hồ sơ",
        "docs/FINAL_REPORT.md", "the knock-out and score 0.855"),
    Num(S5, "−0.033", "v2: LLM đặt 5/8 điểm S", "nDCG@10(P6 v2) − nDCG@10(B3⊕, cùng đầu vào không LLM)",
        "docs/EVAL.md", "0.033 below its own LLM-free input"),
    # NLP
    Num(S8, "0.00 / 0.17", "trap CV trong top 10, có / bỏ hedge scope (E19)",
        "traps@10 = số CV bẫy (nhồi từ khoá dạng hedge, nhãn 0) trong top 10, trung bình trên JD ours",
        "docs/NLP_ABLATION_RESULTS.md", "| − hedge scope (NegEx) | 0.850 [0.805, 0.892] | +0.001 | 0.748 | 0.17 |",
        "cd backend && uv run python -m app.eval.nlp_ablation"),
    Num(S9, "0.794 / 0.741", "BM25 / dense, ours 12 JD", NDCG, "docs/EVAL_RESULTS.md", "| B1 dense | 12 | 0.741", RUN_EVAL),
    Num(S9, "0.542 / 0.333", "BM25 / dense, coding 6 JD", NDCG, "docs/EVAL_RESULTS_coding.md", "| B1 dense | 6 | 0.333",
        RUN_EVAL + " --bench coding"),
    Num(S9, "+0.04 → +0.09", "cross-encoder thêm vào retrieval, ours (whole / structural / window)",
        "nDCG(R + CE) − nDCG(R): 0.818−0.783, 0.829−0.740, 0.819−0.729", "docs/EVAL_RESULTS.md",
        "| R structural + CE | 12 | 0.829", "cd backend && uv run python -m app.eval.run retrieval"),
    Num(S9, "+0.046 / −0.027", "fine-tuned cross-encoder: v1 test / Djinni (E20)", DELTA, "docs/FINETUNE_RESULTS.md",
        "| -0.027 | 0.979 |", "cd backend && uv run --group ml python -m app.eval.finetune score"),
    # Agent
    Num(S11, "8 / 1", "số đường đi khác nhau trên 18 JD: P6 / fixed pipeline",
        "đếm chuỗi node khác nhau trong trace của mỗi JD", "docs/FINAL_REPORT.md", "it took 8 distinct paths"),
    Num(S11, "17/17", "verdict near-tie giữ nguyên khi đảo thứ tự (Q7)", "số cặp có cùng phán quyết ở hai thứ tự / số cặp near-tie",
        "docs/FINAL_REPORT.md", "17 of 17 benchmark near-tie pairs"),
    Num(S11, "0 / 42", "false recovery của investigator (L6), qwen3:8b",
        "số CV được đưa lại sai / 42 ca lỗi gài sẵn; code kiểm mọi đề xuất", "docs/FINAL_REPORT.md",
        "0 false recoveries in 42 planted cases", "cd backend && uv run python -m app.eval.scenarios"),
    # Guardrails
    Num(S12, "100%", "trích dẫn có trong CV (Q2)", "số quote khớp (fuzzy, sau NFC) với văn bản CV / tổng quote",
        "docs/FINAL_REPORT.md", "**Q2:** 100% of quotes grounded"),
    Num(S12, "2.3% → 1.5% (B4: 3.5%)", "nhận định mâu thuẫn với CV (Q4)", RATE, "docs/figures/controls.svg",
        "2.3% → 1.5%", "cd backend && uv run python -m app.eval.figures"),
    Num(S12, "6.7% → 1.1%", "ứng viên trong báo cáo có nhận định thiếu căn cứ (Q6)", RATE, "docs/figures/controls.svg",
        "6.7% → 1.1%", "cd backend && uv run python -m app.eval.figures"),
    Num(S12, "10/10, 0/150", "tấn công injection bị chặn / CV sạch bị báo nhầm, ngưỡng 0.9",
        "số câu tấn công diễn đạt lại bị chặn / 10; số CV sạch bị gắn cờ / 150", "docs/INJECTION_RESULTS.md",
        "| 0.9 | 10/10 | 0/8 | 0/150 | 0/2016 |", "cd backend && uv run python -m app.eval.injection"),
    Num(S12, "0 (6 × 1.800)", "điểm thay đổi khi đổi tên, giới tính, tuổi, hôn nhân (Q11)",
        "số cặp (CV, JD) có điểm khác bản gốc, trên 6 biến thể × 1.800 cặp", "docs/FINAL_REPORT.md",
        "0 score changes in 6 twin variants × 1,800 (CV, JD) pairs", "cd backend && uv run python -m app.eval.robustness"),
    # Data
    Num(S13, "268", "tệp đầu vào khoá bằng SHA-256", "manifest liệt kê SHA-256 từng tệp; `verify` băm lại và so",
        "docs/BENCHMARK.md", "(268 files)", "cd backend && uv run python -m app.eval.benchmark verify"),
    Num(S13, "424", "cặp Djinni có hai nhãn người", "số cặp (JD, CV) trong pool có nhãn của cả hai người",
        "docs/DATA_EVAL_SUMMARY.md", "**424 pairs**", "cd backend && uv run python -m app.eval.djinni_gold all"),
    Num(S13, "120", "cặp ours gán mù bởi Hoa và Phúc (94 cặp silver bất đồng + 26 đối chứng)", "chọn mẫu trước khi chấm",
        "docs/HUMAN_LABELS.md", "| Hoa Reviewer vs Phuc | 120 |", "cd backend && uv run python -m app.eval.human score"),
    # NLP quality
    Num(S14, "0.978 (EN 0.983, VI 0.969)", "F1 tool extraction trên 200 CV coding (1307 TP / 18 FP / 41 FN)", F1,
        "docs/EXTRACTION_RESULTS.md", "| all | 0.986 | 0.970 | 0.978 | 1307 / 18 / 41 |",
        "cd backend && uv run python -m app.eval.extraction"),
    Num(S14, "0.997, CER 0.031", "F1 công cụ sau OCR, 30 CV in thành ảnh",
        F1 + "; CER = số ký tự sai (edit distance) / tổng ký tự", "docs/OCR_RESULTS.md",
        "| scan | all | 0.031 | 0.995 / 1.000 / 0.997 |", "cd backend && uv run --group ocr python -m app.eval.ocr"),
    Num(S14, "0.79 năm", "sai số số năm kinh nghiệm, 178 CV có mốc thời gian", "MAE = mean |năm dự đoán − năm thật|",
        "docs/EXTRACTION_RESULTS.md", "MAE 0.79 y over 178 CVs"),
    Num(S14, "89%", "accuracy automation depth L0–L4", "số CV đoán đúng mức / tổng CV", "docs/EXTRACTION_RESULTS.md",
        "| automation depth L0–L4 (verb cue) | accuracy 89.0% |"),
    Num(S14, "0%", "PII còn lộ sau redaction (5 trường)", "số CV còn thấy giá trị thật của trường / số CV có trường đó",
        "docs/EXTRACTION_RESULTS.md", "| name | 193 | 0 | 0.0% |"),
    Num(S14, "87.5%", "depth trên 72 CV tiếng Việt bỏ dấu", "như accuracy depth, sau strip_accents",
        "docs/EXTRACTION_RESULTS.md", "**87.5%** without diacritics"),
    Num(S14, "100% đọc được", "2.484 PDF Kaggle", "số PDF đọc không lỗi / 2.484", "docs/KAGGLE_RESUME_RESULTS.md",
        "| PDFs read without error | 2484/2484 (100.0%) |", "cd backend && uv run python -m app.eval.kaggle_resume"),
    Num(S14, "41.9% → 97.7%", "CV có vai trò đọc được mốc thời gian, trước / sau sửa date grammar",
        "số CV có ≥ 1 vai trò có ngày / 2.484", "docs/FINAL_REPORT.md", "dated roles 41.9% → 97.7%"),
    Num(S14, "17.2% → 1.6%", "CV ngành khác có \"công cụ kiểm thử\" (gần như chắc là nhận nhầm)",
        "số CV ngoài IT/kỹ thuật có ≥ 1 công cụ / 2.246", "docs/FINAL_REPORT.md",
        "false positives in other trades 17.2% → 1.6%"),
    Num(S14, "0.157", "F1 công cụ trên 7 CV nhiễu ký tự", F1, "docs/EXTRACTION_RESULTS.md",
        "| low_quality | 0.667 | 0.089 | 0.157 |"),
    # Benchmark v1
    Num(S15, "0.851 / 0.841 / 0.839 / 0.810", "P6 / B3 / B4 / rubric ranker, 10 test JD", NDCG + "; " + CI,
        "docs/BENCHMARK_RESULTS.md", "| 10 | 0.851 [0.799, 0.902] |", RUN_V1),
    Num(S15, "+0.093 (p = 0.009)", "B3 tám tiêu chí so với v2 (4 luật ⊕ CE)", DELTA, "docs/BENCHMARK_RESULTS.md",
        "| 10 | +0.093 | 0.009 |", RUN_V1),
    Num(S15, "+0.041 (p = 0.12)", "P6 so với rubric ranker (Đán core)", DELTA, "docs/BENCHMARK_RESULTS.md",
        "| 10 | +0.041 | 0.119 |", RUN_V1),
    Num(S15, "+0.012 (p = 0.064)", "P6 so với B4 fixed pipeline", DELTA, "docs/BENCHMARK_RESULTS.md",
        "| P6 | B4 | 10 | +0.012 | 0.064 |", RUN_V1),
    Num(S15, "0.68 / 0.58", "JD mơ hồ ambiguous_senior: P6 / B4", "nDCG@10 của riêng JD đó", "docs/FINAL_REPORT.md",
        "(0.68 against 0.58)"),
    Num(S15, "0.841 / 0.715", "dữ kiện có cấu trúc (B3) / retrieval văn bản (R structural + CE)", NDCG,
        "docs/BENCHMARK_RESULTS.md", "| 10 | 0.715 [0.584, 0.833] |", RUN_V1),
    Num(S15, "τ = 1.00, ≤ 0.007", "v1.1: chấm lại bằng 120 cặp nhãn mù", "Kendall τ-b giữa thứ tự 8 hệ thống ở v1 và v1.1",
        "docs/BENCHMARK_V1_1_RESULTS.md", "Kendall τ between the system orders under v1 and v1.1: **1.00**",
        "cd backend && uv run python -m app.eval.benchmark_v1_1 report"),
    # Djinni
    Num(S16, "+0.214 (p < 0.001)", "B3 so với BM25 trên Djinni", DELTA, "docs/DJINNI_PRODUCTION_RESULTS.md",
        "| 0.650 [0.450, 0.850] | 0.400 [0.329, 0.460] | +0.214 | 0.000 |",
        "cd backend && uv run python -m app.eval.djinni_gold production-report"),
    Num(S16, "0.422 → 0.643", "BM25 → P6 trên Djinni (nhãn = trung bình hai người, không làm tròn)", NDCG,
        "docs/DJINNI_PRODUCTION_RESULTS.md", "**0.643 [0.563, 0.725]**"),
    Num(S16, "0.725 (B3 0.650)", "P@5 của P6 trên Djinni",
        "P@5 = số CV relevant trong top 5 / 5; relevant = cả hai người chấm ≥ 2", "docs/DJINNI_PRODUCTION_RESULTS.md",
        "| 0.725 [0.575, 0.875] |"),
    Num(S16, "κ = 0.696", "đồng thuận Vũ–Đán trên 424 cặp (exact 67.0%)", KAPPA, "docs/DATA_EVAL_SUMMARY.md",
        "quadratic-weighted κ **0.696**", "cd backend && uv run python -m app.eval.djinni_gold all"),
    Num(S16, "κ = 0.965", "đồng thuận Hoa–Phúc trên 120 cặp mù (exact 90.8%)", KAPPA, "docs/HUMAN_LABELS.md",
        "| Hoa Reviewer vs Phuc | 120 | 90.8% | 0.965 |"),
    Num(S16, "0.696 (silver)", "P6 trên Djinni nếu dùng nhãn LLM (bảng v2)", NDCG, "docs/BENCHMARK_V2_RESULTS.md",
        "| 0.882 | 0.805 | 0.696 |", "cd backend && uv run python -m app.eval.benchmark_v2 report"),
    # Agent value
    Num(S17, "+0.129 (p < 0.001), 57%, τ +0.056",
        "hỏi lại HR trên bản mơ hồ của 18 JD (H6)",
        DELTA + "; phần hồi phục = (0.756 − 0.626) / (0.851 − 0.626) = 57%", "docs/CLARIFY_RESULTS.md",
        "ΔnDCG@10 +0.129 (p = 0.000), Δτ +0.056 (p = 0.012). Share of the vague loss recovered: 57%",
        "cd backend && uv run python -m app.eval.clarify"),
    Num(S17, "40/42", "behaviour check đạt, P7 trên tác vụ HR (E15)", "số check đạt / số check áp dụng (n/a không tính)",
        "docs/AGENTIC_RESULTS.md", "**Behaviour checks passed: 40/42**", "cd backend && uv run python -m app.eval.agentic"),
    Num(S17, "81/83", "check đạt trên mục tiêu phức tạp, 3 lần mỗi mục tiêu (E26)", "số check đạt / tổng check",
        "docs/GOAL_RESULTS.md", "**81/83 behaviour checks passed.**", "uv run --directory backend python -m app.eval.goals --repeat 3"),
    Num(S18, "88/88, 1/40", "JD-003: quote có trong CV; nhận định mâu thuẫn bị bắt", "đếm trong trace của lượt chạy",
        "docs/FINAL_REPORT.md", "88/88 quotes grounded; 1/40 rationales contradict"),
    Num(S18, "43 (3 thêm bởi luật)", "JD-003: ứng viên sau search", "đếm trong trace", "docs/FINAL_REPORT.md",
        "43 candidates (3 added by rule score"),
    Num(S18, "Δ = 0.022", "JD-003: near-tie bị đảo khi đổi thứ tự → giữ thứ tự luật", "chênh điểm luật của cặp",
        "docs/FINAL_REPORT.md", "(Δ=0.022): the verdict flipped"),
    Num(S18, "1/56", "E26: chỉ 1 CV đáp ứng Cypress + Appium", "check_goal đếm CV đáp ứng mọi ràng buộc, 0 token",
        "docs/GOAL_RESULTS.md", "Only 1 of 56 CVs meets all your must-haves"),
    # Related work (ours side; paper numbers are quoted from the papers)
    Num(S19, "0.884 → 0.842 (hybrid 0.881)", "JD parser chỉ dùng LLM so với rules-first", NDCG, "docs/FINAL_REPORT.md",
        "ranking fell **0.884 → 0.842** (ours)"),
    # Hypotheses
    Num(S20, "−47.4% / −66.9%", "token tiết kiệm của P6 / B4 so với E-every (LLM cho mọi ứng viên)",
        "1 − tokens/JD(hệ thống) / tokens/JD(E-every), 8 dev JD chung", "docs/TOKEN_COST_RESULTS.md",
        "| P6 | 131,794 | 17.8 | 76 | 0.875 [0.828, 0.926] | 47.4% |", "cd backend && uv run python -m app.eval.token_cost"),
    Num(S20, "τ 0.554 / 0.542", "H4: P6 / A2 (bỏ company profile), ours 12 JD", "Kendall τ-b giữa thứ tự hệ thống và nhãn, trung bình JD",
        "docs/EVAL_RESULTS.md", "| A2 | 12 | 0.878 [0.839, 0.918]", RUN_EVAL),
    Num(S20, "6 FN 1 FP / 4 FN 1 FP", "lỗi của P6 trên test ours / coding",
        "FN = CV relevant (≥ 2) xếp ngoài top 10 dù tránh được; FP = CV nhãn 0 trong top 5", "docs/ERROR_ANALYSIS.md",
        "6 avoidable misses, 1 false alarms over 6 JDs.", "cd backend && uv run python -m app.eval.errors --bench test"),
    Num(S20, "10 FN 1 FP / 4 FN 0 FP", "lỗi của rubric ranker", "như trên", "docs/ERROR_ANALYSIS.md",
        "10 avoidable misses, 1 false alarms over 6 JDs."),
    Num(S20, "−0.008", "HR learning qua nhiều phiên (E24)", DELTA, "docs/HR_LEARNING_RESULTS.md",
        "| all | 10 | 0.853 | 0.846 | -0.008 |", "cd backend && uv run python -m app.eval.hr_learning"),
    Num(S20, "127k–230k token/JD", "single agent không harness (S1)", "tokens CLI báo cho mỗi JD", "docs/FINAL_REPORT.md",
        "used 127k–230k tokens per JD", "cd backend && uv run python -m app.eval.single_agent"),
    # Limits, gaps, conclusion (reuse the sources above)
    Num(S21, "0.85 / 0.64", "P6 trên test tổng hợp / trên Djinni", NDCG, "docs/DJINNI_PRODUCTION_RESULTS.md",
        "**0.643 [0.563, 0.725]**"),
    Num(S21, "±0.05–0.08", "nửa độ rộng 95% CI với 10 test JD", CI, "docs/FINAL_REPORT.md", "CIs are ±0.05–0.08"),
    Num(S22, "−0.004", "A2: bỏ company profile, ours 12 JD (0.878 − 0.882)", DELTA, "docs/EVAL_RESULTS.md",
        "| A2 | 12 | 0.878 [0.839, 0.918]", RUN_EVAL),
    Num(S23, "0.851, τ = 1.00", "P6 trên 10 test JD; thứ tự giữ nguyên với nhãn người", NDCG, "docs/BENCHMARK_RESULTS.md",
        "| 10 | 0.851 [0.799, 0.902] |", RUN_V1),
]


# Datasets (relative to the repo root). `.runtime/` is gitignored: real or generated data, rebuilt by the commands.
OURS = "data/cvs/ · data/jds/ · data/eval/annotator_{a,b}.yaml"
SPLIT = "data/eval/benchmark_v1.json"
CODING = ".runtime/bench/coding/"
LATENT = ".runtime/bench/coding/latent.json · .runtime/bench/coding/cvs/"
DJINNI = ".runtime/bench/djinni/ · data/eval/human_djinni_gold.json"
RES = ".runtime/eval/results/"
V1 = f"{SPLIT} · data/cvs/ · data/jds/ · {CODING} · {RES}"

# (slide, value) → (datasets, worked calculation). Inputs come from the per-JD and summary tables of the source files.
WORKED: dict[tuple[str, str], tuple[str, str]] = {
    (S4, "0.924"): ("data/cvs/CV-033.md · data/jds/JD-010.md · data/criteria_profiles.yaml",
                    "(0.20×92 + 0.20×75 + 0.10×100 + 0.10×100 + 0.10×90 + 0.30×100) / 100 = (18.4 + 15 + 10 + 10 + 9 + 30) / 100 = 0.924"),
    (S4, "trọng số 4 profile"): ("data/criteria_profiles.yaml",
                                 "default: 0.25 + 0.20 + 0.15 + 0.10 + 0.10 + 0.10 + 0.05 + 0.05 = 1.00 (mỗi profile tổng = 1)"),
    (S4, "0.855"): ("data/cvs/CV-040.md · data/criteria_profiles.yaml",
                    "thiếu Selenium → H = 0 → điểm 0; giả định có Selenium → H = 1, Σ w·S / 100 = 0.855"),
    (S5, "−0.033"): (f"{OURS} · {RES}", "0.838 (P6 v2) − 0.871 (B3⊕) = −0.033; P6 > B3⊕ chỉ ở 3.6% mẫu bootstrap"),
    (S8, "0.00 / 0.17"): (f"{OURS} (8 CV bẫy) · {SPLIT} · .runtime/eval/results/E19_*.json",
                          "Σ_JD (#CV bẫy trong top 10) / 12 JD: đủ pipeline 0 / 12 = 0.00; bỏ hedge scope ≈ 2 / 12 = 0.17"),
    (S9, "0.794 / 0.741"): (f"{OURS} · .runtime/eval/results/B0_BM25.json, B1_dense.json",
                            "BM25: (0.72+0.78+0.54+0.77+0.93+0.82+0.61+0.90+0.85+0.74+0.90+0.96)/12 = 9.52/12 = 0.794; "
                            "dense: 8.90/12 = 0.742 ≈ 0.741 (làm tròn từng JD)"),
    (S9, "0.542 / 0.333"): (f"{CODING} · .runtime/eval/results_coding/",
                            "BM25: (0.71+0.40+0.38+0.61+0.40+0.76)/6 = 3.26/6 = 0.543; dense: (0.64+0.21+0.16+0.17+0.34+0.49)/6 = 2.01/6 = 0.335"),
    (S9, "+0.04 → +0.09"): (f"{OURS} · .runtime/eval/results/R_*.json",
                            "whole 0.818 − 0.783 = +0.035; structural 0.829 − 0.740 = +0.089; window 0.819 − 0.729 = +0.090"),
    (S9, "+0.046 / −0.027"): (f"{V1} · {DJINNI} · .runtime/colab/ (mô hình fine-tuned)",
                              "v1 test: nDCG(CE-ft) − nDCG(CE) = +0.046 (p = 0.003); Djinni: −0.027 (p = 0.979)"),
    (S11, "8 / 1"): (f"{OURS} · {CODING} · .runtime/checkpoints.sqlite (trace)", "P6: 8 chuỗi node khác nhau trên 18 JD; B4: 1 chuỗi cố định"),
    (S11, "17/17"): (f"{V1}", "17 cặp |Δ| < 0.05 → 17 cặp cùng phán quyết ở hai thứ tự → 17/17 = 100%"),
    (S11, "0 / 42"): ("data/cvs/ + 42 lỗi gài (tên công cụ sai chính tả, nhiễu OCR) do app.eval.scenarios tạo",
                      "đề xuất sai được code chấp nhận: 0 → 0/42"),
    (S12, "100%"): (f"{V1}", "số quote khớp văn bản CV / tổng quote = 100% (vd JD-003: 88/88)"),
    (S12, "2.3% → 1.5% (B4: 3.5%)"): (f"{V1} · .runtime/eval/results_coding/P6.json, B4.json",
                                      "Σ claims_bad / Σ claims trên ours + coding: P6 lần đầu 2.3% → sau regenerate 1.5%; B4 3.5% → 3.5%"),
    (S12, "6.7% → 1.1%"): (f"{V1} · .runtime/eval/results_coding/P6.json",
                           "ứng viên có ≥ 1 nhận định thiếu căn cứ / ứng viên trong báo cáo: 6.7% → 1.1% sau một lần regenerate"),
    (S12, "10/10, 0/150"): ("data/cvs/ (150 CV sạch qua 3 bộ) · câu tấn công trong app/eval/injection.py",
                            "chặn 10 / 10 tấn công diễn đạt lại; gắn cờ nhầm 0 / 150 CV sạch (ngưỡng P ≥ 0.9)"),
    (S12, "0 (6 × 1.800)"): (f"{OURS} · {CODING} · bản song sinh do app.eval.robustness tạo",
                             "6 biến thể × 1.800 cặp (CV, JD) = 10.800 lần chấm; số lần điểm khác bản gốc = 0"),
    (S13, "268"): (SPLIT, "268 tệp (CV, JD, nhãn) × SHA-256; `verify` băm lại: 268/268 khớp"),
    (S13, "424"): (DJINNI, "8 JD × (top B0–B3 + đối chứng ngẫu nhiên) = 424 cặp (JD, CV) có nhãn của cả hai người"),
    (S13, "120"): ("data/eval/human_hoa.json · data/eval/human_phuc.json", "94 cặp hai LLM bất đồng + 26 cặp đồng ý (đối chứng) = 120"),
    (S14, "0.978 (EN 0.983, VI 0.969)"): (LATENT, "P = 1307/(1307+18) = 0.986; R = 1307/(1307+41) = 0.970; F1 = 2·0.986·0.970/(0.986+0.970) = 0.978. "
                                          "EN: 819/1/27 → 0.983; VI: 488/17/14 → 0.969"),
    (S14, "0.997, CER 0.031"): (".runtime/bench/coding/cvs/ (30 CV in thành ảnh) · .runtime/bench/coding/latent.json",
                                "P = 0.995, R = 1.000 → F1 = 2·0.995·1.000/1.995 = 0.997; CER = ký tự sai / tổng ký tự = 0.031 (EN 0.003, VI 0.060)"),
    (S14, "0.79 năm"): (LATENT, "MAE = Σ|năm dự đoán − năm thật| / 178 = 0.79 năm; 178/178 CV lệch ≤ 1 năm"),
    (S14, "89%"): (LATENT, "số CV đoán đúng mức L0–L4 / số CV = 89.0%"),
    (S14, "0%"): (LATENT, "name 0/193 + email 0/193 + gender 0/122 + phone 0/121 + birth date 0/121 = 0/750 = 0%"),
    (S14, "87.5%"): (LATENT + " (72 CV tiếng Việt, strip_accents)", "depth đúng 63/72 = 87.5% (bản có dấu cũng 63/72)"),
    (S14, "100% đọc được"): (".runtime/bench/kaggle_resume/ (Resume.csv + 2.484 PDF)", "2484 / 2484 = 100%"),
    (S14, "41.9% → 97.7%"): (".runtime/bench/kaggle_resume/", "CV có ≥ 1 vai trò có ngày / 2.484: trước 41.9% (≈ 1.041 CV) → sau 97.7% (≈ 2.427 CV)"),
    (S14, "17.2% → 1.6%"): (".runtime/bench/kaggle_resume/ (2.246 CV ngoài IT/kỹ thuật)",
                            "CV có ≥ 1 công cụ / 2.246: trước 17.2% (≈ 386 CV) → sau 1.6% (≈ 36 CV)"),
    (S14, "0.157"): (LATENT + " (7 CV nhiễu ký tự)", "P = 4/(4+2) = 0.667; R = 4/(4+41) = 0.089; F1 = 2·0.667·0.089/(0.667+0.089) = 0.157"),
    (S15, "0.851 / 0.841 / 0.839 / 0.810"): (V1, "P6 = (0.86+0.99+0.86+0.97+0.84+0.78+0.68+0.83+0.83+0.87)/10 = 8.51/10 = 0.851; "
                                             "B3 = 8.41/10; B4 = 8.38/10 ≈ 0.839; rubric = 8.10/10"),
    (S15, "+0.093 (p = 0.009)"): (V1, "0.841 − 0.748 = +0.093 (v2 = 7.49/10); 45 / 5.000 mẫu bootstrap có Δ ≤ 0 → p = 0.009"),
    (S15, "+0.041 (p = 0.12)"): (V1, "0.851 − 0.810 = +0.041; 595 / 5.000 mẫu có Δ ≤ 0 → p = 0.119"),
    (S15, "+0.012 (p = 0.064)"): (V1, "0.851 − 0.839 = +0.012; 320 / 5.000 mẫu có Δ ≤ 0 → p = 0.064"),
    (S15, "0.68 / 0.58"): (V1, "nDCG@10 của riêng JD ambiguous_senior: P6 0.68 (hỏi HR) − B4 0.58 = +0.10"),
    (S15, "0.841 / 0.715"): (V1, "R structural + CE = (0.89+0.99+0.79+0.97+0.68+0.56+0.57+0.28+0.62+0.81)/10 = 7.16/10 ≈ 0.715; B3 = 8.41/10 = 0.841"),
    (S15, "τ = 1.00, ≤ 0.007"): (f"{V1} · data/eval/human_hoa.json + human_phuc.json",
                                 "thứ tự 8 hệ thống ở v1 = ở v1.1 → C = 28 cặp cùng chiều, D = 0 → τ-b = (28 − 0)/28 = 1.00; "
                                 "Δ lớn nhất: rubric 0.810 → 0.817 = +0.007"),
    (S16, "+0.214 (p < 0.001)"): (DJINNI, "0.636 (B3) − 0.422 (B0) = +0.214; 0 / 5.000 mẫu có Δ ≤ 0 → p < 0.001"),
    (S16, "0.422 → 0.643"): (DJINNI + " · .runtime/eval/results_djinni/", "trung bình 8 JD: B0 0.422, B1 0.464, B2 0.482, B3 0.636, P6 0.643; P6 − B0 = +0.221"),
    (S16, "0.725 (B3 0.650)"): (DJINNI, "Σ_JD (#relevant trong top 5)/5 / 8 JD: P6 29/40 = 0.725; B3 26/40 = 0.650"),
    (S16, "κ = 0.696"): ("data/eval/human_djinni_vu.json · data/eval/human_djinni_dan.json",
                         "424 cặp, trùng tuyệt đối 284 (67.0%); κ = 1 − Σ w·O / Σ w·E = 0.696"),
    (S16, "κ = 0.965"): ("data/eval/human_hoa.json · data/eval/human_phuc.json", "120 cặp, trùng tuyệt đối 109 (90.8%); κ = 0.965"),
    (S16, "0.696 (silver)"): (".runtime/bench/djinni/ · data/eval/djinni_silver_a.json + djinni_silver_b.json · data/eval/benchmark_v2.json",
                              "P6 trên 8 JD Djinni: nhãn LLM 0.696; nhãn người 0.643 → nhãn LLM cao hơn 0.053"),
    (S17, "+0.129 (p < 0.001), 57%, τ +0.056"): (f"{OURS} · {CODING} · bản mơ hồ của 18 JD do app.eval.clarify tạo",
                                                 "0.756 − 0.626 = 0.130 ≈ +0.129 (trung bình chênh từng JD); hồi phục = 0.130 / (0.851 − 0.626) = "
                                                 "0.130 / 0.225 = 57%; τ: 0.507 − 0.451 = +0.056"),
    (S17, "40/42"): (".runtime/eval/agentic.json · data/jds/ (6 dev JD)", "11 lượt chạy, 42 check áp dụng, 40 đạt = 95.2%"),
    (S17, "81/83"): ("data/goal_suite/ · .runtime/bench/goal_suite_corpus/", "83 check (3 lần chạy mỗi mục tiêu), 81 đạt = 97.6%"),
    (S18, "88/88, 1/40"): ("data/jds/JD-003.md · data/cvs/", "88 quote, 88 có trong CV; 40 nhận định, 1 mâu thuẫn (CV-033) → regenerate"),
    (S18, "43 (3 thêm bởi luật)"): ("data/jds/JD-003.md · data/cvs/", "40 CV từ retrieval + 3 CV điểm luật cao mà retrieval bỏ sót = 43"),
    (S18, "Δ = 0.022"): ("data/cvs/CV-033.md · data/cvs/CV-014.md", "|S(CV-033) − S(CV-014)| = 0.022 < 0.05 → arbiter; đảo khi đổi thứ tự → giữ thứ tự luật"),
    (S18, "1/56"): ("data/goal_suite/ · .runtime/bench/goal_suite_corpus/", "56 CV; đáp ứng đủ 6 must-have (Cypress, Appium, Playwright, TypeScript, API, 3+ năm): 1 (CV-026)"),
    (S19, "0.884 → 0.842 (hybrid 0.881)"): (f"{OURS} · .runtime/eval/requirements/",
                                            "parser chỉ LLM: 0.884 → 0.842 = −0.042; parser hybrid: 0.881 (không đổi)"),
    (S20, "−47.4% / −66.9%"): (f"{OURS} · {CODING} (8 dev JD) · .runtime/eval/results/E-every.json, P6.json, B4.json",
                               "P6: 1 − 131.794 / 250.771 = 47.4%; B4: 1 − 83.120 / 250.771 = 66.9%"),
    (S20, "τ 0.554 / 0.542"): (f"{OURS} · .runtime/eval/results/P6.json, A2", "τ-b trung bình 12 JD: P6 0.554, A2 0.542 → chênh 0.012"),
    (S20, "6 FN 1 FP / 4 FN 1 FP"): (V1, "ours: 3 điểm luật thấp + 3 bị knock-out = 6 FN, 1 FP (must-have chỉ liệt kê); coding: 4 FN, 1 FP"),
    (S20, "10 FN 1 FP / 4 FN 0 FP"): (V1, "rubric ranker: ours 10 FN, 1 FP; coding 4 FN, 0 FP"),
    (S20, "−0.008"): (f"{OURS} · {CODING}", "0.846 (học qua phiên) − 0.853 (tĩnh) = −0.007 ≈ −0.008 (làm tròn); 10 phiên, p = 0.775"),
    (S20, "127k–230k token/JD"): (f"{OURS} (dev JD)", "token CLI báo cho mỗi JD của S1: thấp nhất 127k, cao nhất 230k"),
    (S21, "0.85 / 0.64"): (f"{V1} · {DJINNI}", "P6: 0.851 trên test tổng hợp, 0.643 trên Djinni → thấp hơn 0.208"),
    (S21, "±0.05–0.08"): (V1, "P6: (0.902 − 0.799)/2 = ±0.052; B4: (0.898 − 0.768)/2 = ±0.065; rubric: (0.888 − 0.722)/2 = ±0.083"),
    (S22, "−0.004"): (f"{OURS} · {RES}", "0.878 (A2, bỏ company profile) − 0.882 (P6) = −0.004"),
    (S23, "0.851, τ = 1.00"): (V1, "8.51 / 10 = 0.851; thứ tự 8 hệ thống không đổi với nhãn người → τ = 1.00"),
}
NUMS = [replace(n, data=WORKED[(n.slide, n.value)][0], calc=WORKED[(n.slide, n.value)][1]) for n in NUMS]


def _line(src: str, pat: str) -> int | None:
    path = ROOT / src
    if not path.exists():
        return None
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if pat in line:
            return i
    return None


def verify() -> list[str]:
    """Every row whose quoted text is no longer in its source file."""
    return [f"{n.slide} · {n.value}: '{n.pat}' not in {n.src}" for n in NUMS if _line(n.src, n.pat) is None]


def report() -> str:
    out = ["# Defense numbers: formula and source for every number on the deck "
           "(generated by `python -m app.eval.defense_numbers`; do not edit by hand)", "",
           "Each row quotes the generated results file the number comes from; `verify` fails when a regenerated file no "
           "longer contains it. Commands run from the repo root with `PYTHONHASHSEED=0`, `REFERENCE_DATE=2026-10-30`.", "",
           "## Shared formulas", "", f"- **nDCG@10:** {NDCG}", f"- **Δ và p:** {DELTA}", f"- **CI:** {CI}",
           f"- **κ:** {KAPPA}", f"- **F1:** {F1}", ""]
    for slide in dict.fromkeys(n.slide for n in NUMS):
        out += [f"## {slide}", "", "| Số trên slide | Ý nghĩa | Công thức | Tính toán | Dữ liệu | Nguồn kết quả | Tái tạo |",
                "|---|---|---|---|---|---|---|"]
        for n in (x for x in NUMS if x.slide == slide):
            line = _line(n.src, n.pat)
            where = f"`{n.src}:{line}`" if line else f"`{n.src}` (**không tìm thấy**)"
            out.append(f"| **{n.value}** | {n.meaning} | {n.formula} | {n.calc} | `{n.data}` | {where} | "
                       f"{f'`{n.cmd}`' if n.cmd else '—'} |")
        out.append("")
    out += ["## Provenance notes", "",
            "- Djinni labels (`data/eval/human_djinni_{dan,vu}.json`): the first Đán file in git (095c4ad) records "
            "`saw_model: true` on all 424 labels; dc40da7 set the flag to false without changing a grade. Vũ's 424 "
            "notes include 212 that equal Opus's note for the same pair (`data/eval/djinni_gold_opus.json`). Answer "
            "questions about blindness from these files.",
            "- Paper numbers on the related-work slide (ConFit v2 82.20, Clavié and Soulié +22 RP@10, Wilson and "
            "Caliskan 85.1% / 11.1%) are quoted from the papers, not measured here.", ""]
    return "\n".join(out)


def footnotes() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for n in NUMS:
        out.setdefault(n.slide, []).append({"value": n.value, "meaning": n.meaning, "formula": n.formula,
                                            "src": n.src, "line": _line(n.src, n.pat), "cmd": n.cmd,
                                            "data": n.data, "calc": n.calc})
    return out


def main(argv: list[str]) -> int:
    bad = verify()
    if argv[:1] == ["json"]:
        (ROOT / argv[1]).write_text(json.dumps(footnotes(), ensure_ascii=False, indent=1), encoding="utf-8")
    elif argv[:1] != ["verify"]:
        OUT.write_text(report(), encoding="utf-8")
        print(f"→ {OUT}")
    for b in bad:
        print("MISSING", b)
    print(f"{len(NUMS) - len(bad)}/{len(NUMS)} numbers found in their sources")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
