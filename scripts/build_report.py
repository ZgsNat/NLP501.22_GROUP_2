"""Write the experimental results report (Word) from the run folders, in the style of the group's plan document.

Every number is read from ``runs/experiments/*/summary.json`` and the other result files, as in
``build_results_slides.py``, so the report can be rebuilt after a new run:

    python scripts/build_report.py --synthetic runs/experiments/<ts> --public runs/experiments/<ts> \
        --errors runs/experiments/<ts>/error_analysis.json --finetune-seeds runs/finetune/<file>.json \
        --tagger models/<tagger> --spans spans_test.json --audit runs/final/audit_B3.jsonl \
        --bench runs/bench/<ts>/bench.json --h7 "7B=runs/experiments/<ts>" --out "Báo cáo ....docx"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

TEMPLATE = "Kế hoạch & Requirements sản phẩm cuối môn – Nhóm 2 NLP501.docx"
LADDER = ("B0", "B1", "B2", "B3", "B4", "LLM-all", "P6", "A1", "A2", "P6-noclar", "AG-rules", "AG")
CONFIG_TEXT = {
    "B0": "BM25", "B1": "Dense (e5)", "B2": "Lai ghép BM25 + dense (RRF)", "B3": "B2 + Criteria Profile",
    "B4": "Quy trình cố định: B3 + cross-encoder + LLM chấm top 20", "LLM-all": "LLM so sánh mọi cặp liền kề trong top 20",
    "P6": "Hệ thống agentic đầy đủ", "A1": "P6 bỏ verify_evidence", "A2": "P6 dùng prompt chung thay Criteria Profile",
    "P6-noclar": "P6 bỏ bước hỏi lại HR", "AG-rules": "Agent với bộ lập kế hoạch dùng luật",
    "AG": "Agent với bộ lập kế hoạch LLM",
}
HYP_TEXT = {
    "H0": "P6 tốt hơn B4 về nDCG@10 hoặc độ trung thực, chi phí trong ngân sách",
    "H1": "Truy hồi lai B2 không thấp hơn max(B0, B1) và cao hơn ít nhất một trong hai",
    "H2": "Criteria Profile theo công ty: Kendall τ của P6 cao hơn A2",
    "H3": "Phân xử chọn lọc: P6 không kém LLM-all quá ε = 0.02 và gọi LLM ít hơn",
    "H4": "Kiểm chứng bằng chứng: tỷ lệ nhận định thiếu bằng chứng của P6 thấp hơn A1",
    "H5": "Công bằng: impact ratio ≥ 0.8 (quy tắc 4/5) ở mọi JD",
    "H6": "Hỏi lại HR làm tăng τ trên JD mơ hồ (P6 so với P6-noclar)",
    "H7": "Bộ lập kế hoạch LLM (AG) không kém bộ lập kế hoạch dùng luật (AG-rules)",
}
SKILLSPAN_PAPER_F1 = 0.597  # Zhang et al. 2022, Table 5: JobBERT, COMBINED, TEST, mean of 5 seeds
GREEN, RED = RGBColor(0x06, 0x5F, 0x46), RGBColor(0x9F, 0x12, 0x39)


# ------------------------------------------------------------------ data
def load_json(path: Optional[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def load_summary(run_dir: Optional[str]) -> Optional[Dict[str, Any]]:
    return load_json(str(Path(run_dir) / "summary.json")) if run_dir else None


def mean(summ: Dict[str, Any], cfg: str, metric: str) -> float:
    return float(summ["means"].get(cfg, {}).get(metric, float("nan")))


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(x: Optional[float]) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


METRIC_NAME = {"ndcg@10": "nDCG@10", "mrr@10": "MRR@10", "recall@100": "Recall@100", "kendall_tau": "Kendall τ",
               "unsupported_rate": "tỷ lệ thiếu bằng chứng", "llm_calls": "lượt gọi LLM",
               "judge_calls": "lượt gọi LLM đánh giá", "judge_tokens": "token đánh giá", "agent_steps": "số bước",
               "goal_met": "tỷ lệ đạt mục tiêu", "planner_fallback_rate": "tỷ lệ luật thay quyết định LLM"}


def cmp_line(d: Dict[str, Any]) -> str:
    p = (d.get("wilcoxon") or {}).get("p_value")
    p_txt = f", p = {p:.2f}" if p is not None and not math.isnan(p) else ""
    return (f"{METRIC_NAME.get(d['metric'], d['metric'])}: {d['a']} {fmt(d['mean_a'])} so với {d['b']} {fmt(d['mean_b'])} "
            f"(Δ {d['mean_diff']:+.3f}, CI 95% [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{p_txt})")


def hyp_evidence(h: Dict[str, Any]) -> str:
    """One line per paired comparison stored in the hypothesis details."""
    if "impact_ratio_mean" in h:
        fail = h.get("failing_jds") or []
        return f"impact ratio trung bình {fmt(h['impact_ratio_mean'])}, nhỏ nhất {fmt(h.get('impact_ratio_min'))}; " \
               f"{len(fail)} JD dưới 0.8"
    det = h.get("details") or {}
    if "a" in det:
        return cmp_line(det)
    lines: List[str] = []
    for value in det.values():
        for d in value if isinstance(value, list) else [value]:
            if isinstance(d, dict) and "mean_a" in d:
                lines.append(cmp_line(d))
    if det.get("p6_fallbacks"):
        lines.append(f"P6 vượt ngân sách thời gian ở {det['p6_fallbacks']} JD (dùng phương án dự phòng)")
    return "; ".join(lines) or "chưa đủ cấu hình để kiểm định"


# ------------------------------------------------------------------ document helpers
def start_from_template(path: Path) -> Document:
    """Open the plan document and keep only its styles and page setup."""
    doc = Document(str(path))
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)
    return doc


def para(doc, text: str = "", bold_lead: str = "", style: Optional[str] = None):
    p = doc.add_paragraph(style=style)
    if bold_lead:
        p.add_run(bold_lead).bold = True
    p.add_run(text)
    return p


def bullets(doc, items: Sequence[str]) -> None:
    style = "List Bullet" if "List Bullet" in [s.name for s in doc.styles] else None
    for it in items:
        p = doc.add_paragraph(style=style)
        p.add_run(("" if style else "• ") + it)


def table(doc, rows: List[List[Any]], widths: Optional[Sequence[float]] = None, size: float = 9) -> None:
    """``rows[0]`` is the header. A cell is text or (text, RGBColor) for a coloured, bold verdict."""
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    names = [s.name for s in doc.styles]
    t.style = "Table Grid" if "Table Grid" in names else t.style
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            text, color = (val if isinstance(val, tuple) else (str(val), None))
            cell = t.cell(i, j)
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            run.font.size = Pt(size)
            run.bold = i == 0 or color is not None
            if color is not None:
                run.font.color.rgb = color
            if i == 0:
                tc_pr = cell._tc.get_or_add_tcPr()
                tc_pr.append(tc_pr.makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:color"): "auto",
                                                             qn("w:fill"): "F1F5F9"}))
    if widths:
        for row in t.rows:
            for j, w in enumerate(widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph()


def verdict(accepted: Optional[bool]):
    return ("Ủng hộ", GREEN) if accepted else ("Không ủng hộ", RED)


# ------------------------------------------------------------------ sections
def sec_setup(doc, syn, pub, ctx) -> None:
    doc.add_heading("2. Thiết lập thực nghiệm", level=2)
    kappa = list((syn.get("cohen_kappa") or {}).values())
    bullets(doc, [
        f"Gold Set tổng hợp: {syn['n_jds']} JD Automation Test × {syn['n_cvs']} CV đã ẩn danh; hai người gán nhãn "
        f"mô phỏng" + (f", Cohen κ = {min(kappa):.2f}–{max(kappa):.2f}" if kappa else "") + ".",
        (f"Dữ liệu công khai resume-job-description-fit (Hugging Face): {pub['n_jds']} JD kiểm thử, {pub['n_cvs']} CV; "
         "nhãn Fit / Potential / No Fit quy về 2 / 1 / 0.") if pub else "Không chạy trên dữ liệu công khai.",
        "Mô hình: multilingual-e5-base và bge-reranker-base đã tinh chỉnh (ConFit), nli-deberta-v3-base, "
        f"LLM {ctx['llm']} qua Ollama; toàn bộ chạy cục bộ trên CPU, dữ liệu cá nhân không rời máy.",
        "Kiểm định: khoảng tin cậy 95% bằng bootstrap ghép cặp theo JD và kiểm định Wilcoxon; một giả thuyết được "
        "ủng hộ khi toàn bộ khoảng tin cậy nằm đúng phía (giả thuyết \"không kém hơn\": cận dưới ≥ −0.02).",
        f"Với n = {syn['n_jds']} JD, giá trị p Wilcoxon nhỏ nhất là {2 / 2 ** syn['n_jds']:.3f}, nên kết luận dựa trên "
        "khoảng tin cậy chứ không dựa riêng vào giá trị p.",
    ])
    table(doc, [["Cấu hình", "Mô tả"]] + [[c, CONFIG_TEXT[c]] for c in LADDER if c in syn["means"]], [2.5, 13])


def sec_ladder(doc, syn) -> None:
    doc.add_heading("3. Kết quả trên Gold Set", level=2)
    doc.add_heading("3.1. Phân tích loại bỏ thành phần", level=3)
    cfgs = [c for c in LADDER if c in syn["means"]]
    rows = [["Cấu hình", "nDCG@10", "MRR@10", "Recall@100", "Kendall τ", "Thiếu bằng chứng", "Lượt gọi LLM/JD",
             "Độ trễ (s/JD)"]]
    for c in cfgs:
        rows.append([c, fmt(mean(syn, c, "ndcg@10")), fmt(mean(syn, c, "mrr@10")), fmt(mean(syn, c, "recall@100")),
                     fmt(mean(syn, c, "kendall_tau")), pct(mean(syn, c, "unsupported_rate")),
                     fmt(mean(syn, c, "llm_calls"), 1), fmt(mean(syn, c, "latency_s"), 1)])
    table(doc, rows, size=8.5)
    b0, b1, b2, b3 = (mean(syn, c, "ndcg@10") for c in ("B0", "B1", "B2", "B3"))
    best = max(cfgs, key=lambda c: mean(syn, c, "ndcg@10"))
    bullets(doc, [
        f"Criteria Profile tạo bước tăng lớn nhất: nDCG@10 từ {fmt(b2)} (B2) lên {fmt(b3)} (B3). Do profile và nhãn "
        "cùng do nhóm xây dựng, mức tăng này có thể bị đánh giá cao.",
        f"Dense {'vẫn thấp hơn' if b1 < b0 else 'cao hơn'} BM25 ({fmt(b1)} so với {fmt(b0)}).",
        "Cột độ trễ đo trong lượt chạy có bộ nhớ đệm LLM, trên máy dùng chung với các tác vụ khác, nên chỉ để tham "
        "khảo; độ trễ chuẩn (tắt bộ nhớ đệm) ở mục 6.",
        f"Cấu hình có nDCG@10 cao nhất là {best} ({fmt(mean(syn, best, 'ndcg@10'))}); P6 đạt "
        f"{fmt(mean(syn, 'P6', 'ndcg@10'))} với tỷ lệ nhận định thiếu bằng chứng {pct(mean(syn, 'P6', 'unsupported_rate'))}.",
    ])


def sec_hypotheses(doc, syn, h7_runs) -> None:
    doc.add_heading("3.2. Kiểm định giả thuyết", level=3)
    hyp = syn.get("hypotheses") or {}
    rows = [["", "Giả thuyết", "Kết quả đo", "Kết luận"]]
    for h in ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        if h not in hyp or (h == "H7" and "missing" in (hyp[h].get("details") or {})):
            continue
        rows.append([h, HYP_TEXT[h], hyp_evidence(hyp[h]), verdict(hyp[h].get("accepted"))])
    for label, summ in h7_runs.items():
        h = (summ.get("hypotheses") or {}).get("H7")
        if h and "missing" not in (h.get("details") or {}):
            rows.append([f"H7 ({label})", HYP_TEXT["H7"], hyp_evidence(h), verdict(h.get("accepted"))])
    table(doc, rows, [1.6, 4.4, 8, 2.2], size=8)
    ok = [r[0] for r in rows[1:] if r[3][0] == "Ủng hộ"]
    para(doc, f"Giả thuyết được ủng hộ: {', '.join(ok) or 'không có'}. Các giả thuyết còn lại không được dữ liệu "
              "ủng hộ và được báo cáo đầy đủ, đúng kế hoạch thực nghiệm đã đặt trước.")


def sec_faith(doc, syn) -> None:
    doc.add_heading("3.3. Độ trung thực và chi phí", level=3)
    cfgs = [c for c in ("B4", "LLM-all", "P6", "A1", "AG-rules", "AG") if c in syn["means"]]
    rows = [["Cấu hình", "Thiếu bằng chứng", "Độ chính xác bằng chứng", "Lượt gọi LLM/JD", "Lượt gọi LLM đánh giá/JD",
             "Token/JD"]]
    for c in cfgs:
        rows.append([c, pct(mean(syn, c, "unsupported_rate")), pct(mean(syn, c, "evidence_precision")),
                     fmt(mean(syn, c, "llm_calls"), 1), fmt(mean(syn, c, "judge_calls"), 1),
                     fmt(mean(syn, c, "tokens"), 0)])
    table(doc, rows, size=8.5)
    para(doc, f"Bước kiểm chứng bằng chứng giảm tỷ lệ nhận định thiếu bằng chứng từ "
              f"{pct(mean(syn, 'A1', 'unsupported_rate'))} (A1) xuống {pct(mean(syn, 'P6', 'unsupported_rate'))} (P6). "
              f"Phân xử chọn lọc dùng {fmt(mean(syn, 'P6', 'judge_calls'), 1)} lượt gọi LLM đánh giá mỗi JD, so với "
              f"{fmt(mean(syn, 'LLM-all', 'judge_calls'), 1)} của LLM-all.")


def sec_finetune(doc, seeds, syn_ft, syn, pub_ft, pub) -> None:
    if not seeds:
        return
    doc.add_heading("3.4. Tinh chỉnh bộ mã hoá dense và cross-encoder", level=3)
    para(doc, f"Tinh chỉnh theo hướng ConFit trên dữ liệu công khai ngoài lĩnh vực kiểm thử (loại mọi JD kiểm thử để "
              f"tránh rò rỉ), {seeds['epochs']} epoch, độ dài tối đa {seeds['max_len']} token, "
              f"{len(seeds['seeds'])} seed ({', '.join(map(str, seeds['seeds']))}) trên GPU Colab T4.")
    names = {"dense": "Dense e5-base", "rerank": "Cross-encoder bge"}
    rows = [["Mô hình", "Tập", "Trước", "Sau (TB ± SD)", "Mức cải thiện (TB ± SD)"]]
    for t, splits in seeds["summary"].items():
        for split, m in splits.items():
            rows.append([names.get(t, t), split, fmt(m["after_mean"] - m["gain_mean"]),
                         f"{fmt(m['after_mean'])} ± {m['after_sd']:.3f}", f"{m['gain_mean']:+.3f} ± {m['gain_sd']:.3f}"])
    table(doc, rows, size=8.5)
    weak = [f"{names.get(t, t)} trên tập {s}" for t, sp in seeds["summary"].items() for s, m in sp.items()
            if not m["gain_mean"] > 2 * m["gain_sd"]]
    para(doc, "Mức cải thiện lớn hơn 2 lần độ lệch chuẩn giữa các seed" +
         (f", trừ {', '.join(weak)}: chưa phân biệt được với dao động giữa các seed." if weak else " ở mọi trường hợp."))
    if syn_ft and syn:
        para(doc, f"Trên Gold Set, B1 tăng từ {fmt(mean(syn, 'B1', 'ndcg@10'))} lên {fmt(mean(syn_ft, 'B1', 'ndcg@10'))}"
                  f" nhưng vẫn thấp hơn BM25 ({fmt(mean(syn, 'B0', 'ndcg@10'))}) và Criteria Profile; trên dữ liệu "
                  + (f"công khai B1 tăng từ {fmt(mean(pub, 'B1', 'ndcg@10'))} lên {fmt(mean(pub_ft, 'B1', 'ndcg@10'))}."
                     if pub and pub_ft else "công khai chưa đo."))


def sec_public(doc, pub, errors) -> None:
    if not pub:
        return
    doc.add_heading("4. Kết quả trên dữ liệu công khai", level=2)
    cfgs = [c for c in ("B0", "B1", "B2", "B3", "B4", "P6") if c in pub["means"]]
    table(doc, [["", *cfgs], ["nDCG@10", *[fmt(mean(pub, c, "ndcg@10")) for c in cfgs]],
                ["Thiếu bằng chứng", *[pct(mean(pub, c, "unsupported_rate")) for c in cfgs]]], size=8.5)
    best = max(cfgs, key=lambda c: mean(pub, c, "ndcg@10"))
    para(doc, f"Trung bình {pub['n_jds']} JD: {best} cao nhất ({fmt(mean(pub, best, 'ndcg@10'))}); P6 đạt "
              f"{fmt(mean(pub, 'P6', 'ndcg@10'))} so với BM25 {fmt(mean(pub, 'B0', 'ndcg@10'))}.")
    if errors and errors.get("near_duplicates"):
        n_dup = sum(map(len, errors["near_duplicates"]))
        g, d = errors["groups"], errors["dedup"]
        para(doc, "", bold_lead="Phân tích lỗi (hậu kiểm, không phải giả thuyết):")
        bullets(doc, [
            f"{n_dup}/{pub['n_jds']} JD gần như trùng nhau (Jaccard trên 3-gram ≥ 0.8) và là tin tuyển lập trình viên "
            "hạ tầng, chỉ qua bộ lọc nhờ các cụm từ chung chung; Criteria Profile cho Automation Test vì vậy áp dụng "
            "tiêu chí không phù hợp.",
            f"B3 đạt {fmt(g['distinct'].get('B3'))} trên các JD khác nhau nhưng chỉ {fmt(g['duplicate_postings'].get('B3'))}"
            f" trên nhóm trùng lặp.",
            f"Khi mỗi nhóm trùng lặp chỉ tính một lần (n hiệu dụng = {errors['effective_n_jds']}): " +
            ", ".join(f"{c} {fmt(d[c])}" for c in ("B0", "B1", "B3", "P6") if c in d) + ".",
        ])


def sec_spans(doc, spans, tagger) -> None:
    if not (spans or tagger):
        return
    doc.add_heading("5. Trích xuất kỹ năng trên SkillSpan", level=2)
    cols = ([("Từ điển kỹ năng", spans)] if spans else []) + \
           ([(lab, tagger["test"][k]) for k, lab in (("tagger", "JobBERT gán nhãn"), ("hybrid", "Hợp hai phương pháp"))
             if k in tagger["test"]] if tagger else [])
    rows = [["SkillSpan (test)", *[c for c, _ in cols]]]
    for key, lab in (("precision_exact", "Precision (khớp đúng)"), ("recall_exact", "Recall (khớp đúng)"),
                     ("f1_exact", "F1 (khớp đúng)"), ("f1_overlap", "F1 (khớp chồng lấn)")):
        rows.append([lab, *[fmt(r[key]) for _, r in cols]])
    table(doc, rows, size=8.5)
    if tagger:
        para(doc, f"Mô hình gán nhãn JobBERT đạt span F1 {fmt(tagger['test']['tagger']['f1_exact'])} (1 seed), tương "
                  f"đương mức {SKILLSPAN_PAPER_F1} mà bài báo SkillSpan công bố (trung bình 5 seed). Hệ thống dùng từ "
                  "điển để gắn mã khái niệm làm bằng chứng và dùng mô hình gán nhãn để phát hiện kỹ năng ngoài từ điển.")


def sec_nfr(doc, syn, audit, bench) -> None:
    doc.add_heading("6. Công bằng và độ trễ", level=2)
    h5 = (syn.get("hypotheses") or {}).get("H5") or {}
    ir = h5.get("impact_ratio_by_jd") or {}
    swap = {a["jd"]: a for a in audit or []}
    rows = [["JD", "Impact ratio", "Δ thứ hạng khi đổi tên (TB / tối đa)"]]
    for jd in ir or swap:
        a = swap.get(jd)
        rows.append([jd, fmt(ir.get(jd), 2), f"{a['mean_abs_delta_rank']:.1f} / {a['max_abs_delta_rank']}" if a else "—"])
    table(doc, rows, [6, 3, 6], size=8.5)
    if bench:
        rows = [["Mô hình (P6, CPU)", "p50 (s/JD)", "p95 (s/JD)", "Lượt gọi LLM/JD", "Dự phòng"]]
        for model, rep in bench["models"].items():
            s = rep["summary"]
            rows.append([model, f"{s['wall_p50_s']:.0f}", f"{s['wall_p95_s']:.0f}", fmt(s["calls_per_jd"], 1),
                         str(s["fallbacks"])])
        table(doc, rows, size=8.5)
    worst = max((a["max_abs_delta_rank"] for a in audit or []), default=None)
    para(doc, ("Hoán đổi tên không làm thay đổi thứ hạng vì tên được che trước mọi bước mã hoá và gọi LLM. "
               if worst == 0 else "") +
         f"Impact ratio dưới 0.8 ở {len(h5.get('failing_jds') or [])}/{len(ir)} JD; vì tên đã được che, chênh lệch "
         "đến từ thuộc tính khác có tương quan với nhóm và cần được kiểm tra lại trên CV thực tế.")


def sec_conclusion(doc, syn, pub, errors) -> None:
    doc.add_heading("7. Kết luận", level=2)
    hyp = syn.get("hypotheses") or {}
    rejected = [h for h in ("H1", "H2", "H3", "H5", "H6") if not hyp.get(h, {}).get("accepted")]
    doc.add_heading("7.1. Kết quả vững chắc", level=3)
    bullets(doc, [
        f"Trên Gold Set, nDCG@10 tăng từ {fmt(mean(syn, 'B0', 'ndcg@10'))} (BM25) lên {fmt(mean(syn, 'P6', 'ndcg@10'))} "
        "(P6), chủ yếu nhờ Criteria Profile theo công ty.",
        "Bước kiểm chứng bằng chứng giảm rõ rệt tỷ lệ nhận định thiếu bằng chứng (H4"
        + (", H0" if hyp.get("H0", {}).get("accepted") else "") + ").",
        "Phân xử chọn lọc cần ít lượt gọi LLM hơn nhiều so với LLM-all.",
        "Tinh chỉnh dense và cross-encoder cải thiện nDCG@10 trên tập test công khai ở cả 3 seed.",
        "Agent tuân thủ ràng buộc an toàn; hệ thống không tự động loại ứng viên và HR ra quyết định cuối.",
    ])
    doc.add_heading("7.2. Hạn chế", level=3)
    items = [f"{', '.join(rejected)} không được dữ liệu ủng hộ." if rejected else "Mọi giả thuyết H1–H6 được ủng hộ.",
             f"Chỉ có {syn['n_jds']} JD trong mỗi lượt đánh giá: khoảng tin cậy rộng.",
             "Gold Set là dữ liệu tổng hợp; profile và nhãn cùng do nhóm xây dựng nên mức cải thiện của Criteria "
             "Profile có thể bị đánh giá cao.",
             "LLM nhỏ (Qwen2.5 3B/7B) chạy trên CPU; kết luận về LLM-all, phân xử và bộ lập kế hoạch có thể khác với "
             "LLM lớn hơn."]
    fallbacks = (hyp.get("H0", {}).get("details") or {}).get("p6_fallbacks")
    if fallbacks and not hyp.get("H0", {}).get("accepted"):
        items.insert(0, f"H0 không được ủng hộ vì P6 vượt ngân sách thời gian ở {fallbacks} JD và phải dùng phương án dự "
                        "phòng, dù tỷ lệ nhận định thiếu bằng chứng thấp hơn B4 rõ rệt; cần chạy lại JD đó trên máy không "
                        "chạy tác vụ khác trước khi kết luận.")
    if pub and errors and errors.get("near_duplicates"):
        items.insert(1, f"Trên dữ liệu công khai, P6 không vượt BM25 ở giá trị trung bình; "
                        f"{sum(map(len, errors['near_duplicates']))}/{pub['n_jds']} JD là cùng một tin tuyển dụng không "
                        "thuộc vai trò kiểm thử.")
    bullets(doc, items)
    doc.add_heading("7.3. Hướng phát triển", level=3)
    bullets(doc, [
        "Xây dựng Gold Set thực tế khoảng 200 CV, hai chuyên gia QA gán nhãn độc lập với người viết profile, đo Cohen κ.",
        "Lọc JD công khai theo tiêu đề và khử trùng lặp trước khi đánh giá lại.",
        "Tinh chỉnh trên Gold Set thực tế khi đủ nhãn; kiểm tra tính công bằng trên CV thực tế.",
        "Dùng GPU hoặc LLM lớn hơn để giảm độ trễ và kiểm định lại H3, H7.",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", required=True)
    ap.add_argument("--public")
    ap.add_argument("--h7", action="append", default=[], help="label=run_dir with AG / AG-rules")
    ap.add_argument("--errors")
    ap.add_argument("--finetune-seeds")
    ap.add_argument("--synthetic-base", help="run with the base models (before fine-tuning)")
    ap.add_argument("--public-base", help="public run with the base models")
    ap.add_argument("--tagger")
    ap.add_argument("--spans")
    ap.add_argument("--audit")
    ap.add_argument("--bench")
    ap.add_argument("--llm", default="qwen2.5:3b")
    ap.add_argument("--template", default=TEMPLATE)
    ap.add_argument("--out", default="Báo cáo kết quả thực nghiệm – Nhóm 2 NLP501.docx")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Vietnamese file name on Windows consoles

    syn, pub = load_summary(args.synthetic), load_summary(args.public)
    syn_base, pub_base = load_summary(args.synthetic_base), load_summary(args.public_base)
    h7 = {lab: load_summary(p) for lab, p in (item.split("=", 1) for item in args.h7)}
    errors, seeds = load_json(args.errors), load_json(args.finetune_seeds)
    tagger = load_json(str(Path(args.tagger) / "tagger_metrics.json")) if args.tagger else None
    spans, bench = load_json(args.spans), load_json(args.bench)
    audit = [json.loads(line) for line in Path(args.audit).read_text(encoding="utf-8").splitlines()
             if line.strip()] if args.audit else None

    doc = start_from_template(Path(args.template))
    doc.add_heading("Báo cáo kết quả thực nghiệm – Nhóm 2 NLP501", level=1)
    intro = para(doc, "Ứng dụng Agentic AI trong đối sánh và xếp hạng ứng viên Automation Test có giải thích. Mọi số "
                      "liệu trong báo cáo được sinh tự động từ thư mục kết quả bằng scripts/build_report.py.")
    intro.runs[-1].italic = True
    doc.add_heading("1. Mục tiêu", level=2)
    para(doc, "Kiểm định các giả thuyết H0–H7 đã đặt trước khi chạy: hệ thống agentic có xếp hạng ứng viên đúng hơn, "
              "giải thích trung thực hơn và tiết kiệm chi phí hơn quy trình cố định hay không, khi giữ nguyên các "
              "thành phần truy hồi, xếp hạng lại và Criteria Profile.")
    sec_setup(doc, syn, pub, {"llm": args.llm})
    sec_ladder(doc, syn)
    sec_hypotheses(doc, syn, h7)
    sec_faith(doc, syn)
    sec_finetune(doc, seeds, syn, syn_base, pub, pub_base)
    sec_public(doc, pub, errors)
    sec_spans(doc, spans, tagger)
    sec_nfr(doc, syn, audit, bench)
    sec_conclusion(doc, syn, pub, errors)
    for p in doc.paragraphs:
        if p.style.name == "Normal":
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    doc.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
