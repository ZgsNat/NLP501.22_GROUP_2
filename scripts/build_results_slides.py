"""Insert the experimental-results section into Slide_Group2_NLP501-2.pptx.

Every number on the new slides is read from ``runs/experiments/*/summary.json`` (and the SkillSpan span report),
so the section can be rebuilt after a new run. Prior-work numbers are the ones already quoted on slides 16-18 and 26.

    python scripts/build_results_slides.py --synthetic runs/experiments/<ts> --public runs/experiments/<ts> \
        --h7 3b=runs/experiments/<ts> --h7 7b=... --spans spans_test.json --out Slide_Group2_NLP501-3.pptx
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

FONT = "Arial"
PX = 13.333 / 1600  # the original slides are 1600x900 screenshots; convert their pixel grid to inches

INK, MUTED, LINE = RGBColor(0x11, 0x18, 0x27), RGBColor(0x6B, 0x72, 0x80), RGBColor(0xE5, 0xE7, 0xEB)
PAGE, WHITE, SOFT = RGBColor(0xF3, 0xF4, 0xF6), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xF9, 0xFA, 0xFB)
BLUE, BLUE_BG, BLUE_LINE = RGBColor(0x1D, 0x4E, 0xD8), RGBColor(0xEF, 0xF6, 0xFF), RGBColor(0xBF, 0xDB, 0xFE)
GREEN, GREEN_BG = RGBColor(0x06, 0x5F, 0x46), RGBColor(0xD1, 0xFA, 0xE5)
RED, RED_BG = RGBColor(0x9F, 0x12, 0x39), RGBColor(0xFF, 0xE4, 0xE6)
AMBER, AMBER_BG = RGBColor(0x92, 0x40, 0x0E), RGBColor(0xFE, 0xF3, 0xC7)
VIOLET, VIOLET_BG = RGBColor(0x5B, 0x21, 0xB6), RGBColor(0xED, 0xE9, 0xFE)
GREY_BAR, ACCENT_BAR = RGBColor(0xCB, 0xD5, 0xE1), RGBColor(0x10, 0xB9, 0x81)

CHIPS = {"section": (RED, RED_BG), "blue": (BLUE, BLUE_BG), "green": (GREEN, GREEN_BG), "amber": (AMBER, AMBER_BG),
         "violet": (VIOLET, VIOLET_BG)}
SECTION = "07 · KẾT QUẢ THỰC NGHIỆM"
NFR04_MAX_S = 5 * 60  # "vài phút một JD trên laptop" (NFR-04), read as p95 <= 5 minutes


# ------------------------------------------------------------------ data

def load_summary(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def mean(summary: Dict[str, Any], config: str, metric: str) -> float:
    return float(summary["means"].get(config, {}).get(metric, float("nan")))


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def pct(x: Optional[float], nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{nd}f}%"


def cmp_text(d: Dict[str, Any], nd: int = 3) -> str:
    """'0.833 vs 0.835 (Δ −0.002, CI [−0.03, 0.02], p=1.00)'."""
    w = d.get("wilcoxon") or {}
    p = w.get("p_value")
    p_txt = f", p={p:.2f}" if p is not None and not math.isnan(p) else ""
    return (f"{fmt(d['mean_a'], nd)} vs {fmt(d['mean_b'], nd)} (Δ {d['mean_diff']:+.{nd}f}, "
            f"CI [{d['ci_low']:+.{nd}f}, {d['ci_high']:+.{nd}f}]{p_txt})")


# ------------------------------------------------------------------ navigation

# page -> (question shown in the 'TIẾP THEO' bar, link to the next slide); filled by main() from the slide order
NAV: Dict[int, Tuple[str, str]] = {}


def nav(page: int) -> Tuple[str, str]:
    return NAV[page]


def set_nav(first_new: int, plan: Sequence[Tuple[str, str]]) -> None:
    """``plan``: (short label, question leading to it) per new slide; the closing Q&A slide follows the last one."""
    targets = list(plan[1:]) + [("Q&A", "Cảm ơn và hỏi đáp.")]
    NAV.clear()
    for page, (label, question) in enumerate(targets, 1):
        NAV[page] = (question, f"Slide {first_new + page}: {label} →")


# ------------------------------------------------------------------ drawing primitives

def box(slide, x, y, w, h, fill=WHITE, line=LINE, radius: Optional[float] = 0.06, name: str = ""):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(0.75)
    shape.shadow.inherit = False
    if name:
        shape.name = name
    shape.text_frame.text = ""
    return shape


def text(slide, x, y, w, h, runs, size=11, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name: str = "", spacing: float = 2):
    """``runs``: a string, or a list of paragraphs, each a string or a list of (text, {bold,color,size}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paragraphs = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        for chunk, opts in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = chunk
            f = r.font
            f.name = FONT
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", False)
            f.color.rgb = opts.get("color", color)
    return tb


def chip(slide, x, y, label, kind="section", size=8.5):
    fg, bg = CHIPS[kind]
    w = 0.2 + len(label) * size * 0.0105
    s = box(slide, x, y, w, 0.24, fill=bg, line=None, radius=0.5)
    tf = s.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name, r.font.size, r.font.bold, r.font.color.rgb = FONT, Pt(size), True, fg
    return x + w + 0.08


def frame(prs, title: str, chips: Sequence[Tuple[str, str]], page: int, total: int, next_text: str,
          next_link: str):
    """Header strip, page background, card, chips, title and the 'TIẾP THEO' bar of the original deck."""
    slide = prs.slides.add_slide(blank_layout(prs))
    box(slide, 0, 0, 13.333, 7.5, fill=PAGE, line=None, radius=None, name="Page")
    box(slide, 0, 0, 13.333, 0.6, fill=WHITE, line=LINE, radius=None, name="Header")
    badge = box(slide, 0.2, 0.12, 0.93, 0.36, fill=INK, line=None, radius=0.15, name="Group badge")
    text(slide, 0.28, 0.15, 0.85, 0.3, [[("NLP501 ·", {"bold": True})], [("NHÓM 2", {"bold": True})]], size=7,
         color=WHITE, spacing=0)
    del badge
    text(slide, 1.23, 0.1, 6.3, 0.5,
         [[("Ứng dụng Agentic AI trong Đối sánh & Xếp hạng Ứng viên Automation Test có Giải thích",
            {"bold": True, "size": 10})],
          [("Chương trình Cao học CNTT · Báo cáo Đề tài Cuối kỳ", {"color": MUTED, "size": 7})]], spacing=1)
    nav = box(slide, 10.55, 0.13, 1.95, 0.34, fill=BLUE_BG, line=BLUE_LINE, radius=0.2, name="Section tab")
    nav.text_frame.margin_top = nav.text_frame.margin_bottom = 0
    text(slide, 10.6, 0.16, 1.85, 0.3, [[("07. Kết quả thực nghiệm", {"bold": True})]], size=8.5, color=BLUE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 12.6, 0.18, 0.6, 0.3, "Cả nhóm", size=7.5, color=MUTED)

    box(slide, 185 * PX, 0.82, 1230 * PX, 6.08, fill=WHITE, line=LINE, radius=0.025, name="Card")
    x = 213 * PX
    for label, kind in chips:
        x = chip(slide, x, 1.03, label, kind)
    text(slide, 213 * PX, 1.33, 10.0, 0.5, title, size=19, bold=True, name="Title")

    bar = box(slide, 213 * PX, 6.32, 1173 * PX, 0.36, fill=BLUE_BG, line=BLUE_LINE, radius=0.2, name="Next bar")
    del bar
    text(slide, 213 * PX + 0.12, 6.39, 6.8, 0.25, [[("TIẾP THEO: ", {"bold": True}), (next_text, {})]], size=9.5,
         anchor=MSO_ANCHOR.MIDDLE)
    text(slide, 213 * PX + 6.9, 6.39, 2.75, 0.25, [[(next_link, {"bold": True})]], size=9.5, color=BLUE,
         align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)

    text(slide, 0.2, 7.18, 2.5, 0.25, [[(f"07 · {page} / {total}", {"bold": True})]], size=10)
    track = box(slide, 1.0, 7.27, 1.85, 0.06, fill=LINE, line=None, radius=0.5)
    del track
    box(slide, 1.0, 7.27, 1.85 * page / total, 0.06, fill=BLUE, line=None, radius=0.5)
    return slide


def blank_layout(prs):
    return next(l for l in prs.slide_layouts if l.name.lower() == "blank")


def table(slide, x, y, w, rows: List[List[Any]], col_w: Sequence[float], size=9.5, header_color=MUTED,
          row_h=0.3, name: str = "", highlight: Optional[int] = None):
    """``rows[0]`` is the header. A cell is a string or (text, {color,bold}). ``highlight``: row index tinted green."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w),
                                   Inches(row_h * len(rows)))
    if name:
        shape.name = name
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    total = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Emu(int(Inches(w) * cw / total))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, cell_val in enumerate(row):
            cell = tbl.cell(i, j)
            val, opts = (cell_val if isinstance(cell_val, tuple) else (str(cell_val), {}))
            cell.fill.solid()
            cell.fill.fore_color.rgb = SOFT if i == 0 else (GREEN_BG if i == highlight else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.07)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = val
            r.font.name = FONT
            r.font.size = Pt(opts.get("size", size))
            r.font.bold = opts.get("bold", i == 0)
            r.font.color.rgb = opts.get("color", header_color if i == 0 else INK)
            p.alignment = opts.get("align", PP_ALIGN.LEFT)
    _thin_borders(tbl)
    return shape


def _thin_borders(tbl) -> None:
    from pptx.oxml.ns import qn
    for row in tbl.rows:
        for cell in row.cells:
            tc_pr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                for old in tc_pr.findall(qn(tag)):
                    tc_pr.remove(old)
                ln = tc_pr.makeelement(qn(tag), {"w": "6350"})
                fill = ln.makeelement(qn("a:solidFill"), {})
                clr = fill.makeelement(qn("a:srgbClr"), {"val": "E5E7EB"})
                fill.append(clr)
                ln.append(fill)
                tc_pr.append(ln)


def note(slide, x, y, w, h, label, body, kind="blue", size=9.5):
    fg, bg = CHIPS[kind]
    box(slide, x, y, w, h, fill=bg, line=None, radius=0.08)
    text(slide, x + 0.12, y + 0.08, w - 0.24, h - 0.16, [[(label + " ", {"bold": True, "color": fg}), (body, {})]],
         size=size, anchor=MSO_ANCHOR.MIDDLE)


def bar_chart(slide, x, y, w, h, categories, series: Dict[str, List[float]], colors: Sequence[RGBColor],
              title: str, number_format="0.000", max_val: Optional[float] = None, highlight: Optional[int] = None,
              legend: bool = False):
    data = CategoryChartData()
    data.categories = categories
    for name, values in series.items():
        data.add_series(name, [None if (v is None or math.isnan(v)) else v for v in values])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    chart = gf.chart
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    tp = chart.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size, tp.runs[0].font.bold, tp.runs[0].font.name = Pt(10), True, FONT
    tp.runs[0].font.color.rgb = INK
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size, chart.legend.font.name = Pt(8.5), FONT
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size, va.tick_labels.font.name = Pt(8), FONT
    va.tick_labels.font.color.rgb = MUTED
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = number_format, False
    va.minimum_scale = 0
    if max_val:
        va.maximum_scale = max_val
    ca = chart.category_axis
    ca.tick_labels.font.size, ca.tick_labels.font.name = Pt(8.5), FONT
    ca.tick_labels.font.color.rgb = INK
    ca.format.line.color.rgb = LINE
    plot = chart.plots[0]
    plot.gap_width = 60
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = number_format, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.name = Pt(8), FONT
    dl.font.color.rgb = INK
    for s, color in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = color
    if highlight is not None and len(plot.series) == 1:
        pt = plot.series[0].points[highlight]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = ACCENT_BAR
    return gf


# ------------------------------------------------------------------ slides

def slide_setup(prs, syn, pub, spans, page, total):
    s = frame(prs, "Thiết Lập Thực Nghiệm: Dữ Liệu, Mô Hình Và Cách Kiểm Định",
              [(SECTION, "section"), ("CHẠY THẬT · LOCAL CPU · KHÔNG GỬI PII RA NGOÀI", "green")], page, total, *nav(page))
    kappa = list((syn.get("cohen_kappa") or {}).values())
    cards = [
        ("Gold Set tổng hợp (test chính)", "blue",
         [f"{syn['n_jds']} JD Automation Test × {syn['n_cvs']} CV ẩn danh",
          f"2 annotator mô phỏng, Cohen κ = {min(kappa):.2f}–{max(kappa):.2f}" if kappa else "nhãn 0–3 theo JD",
          "Đo trên toàn bộ ranking (judged)"]),
        ("Dữ liệu công khai", "violet",
         [f"HF resume-job-description-fit: {pub['n_jds']} JD QA/Automation, {pub['n_cvs']} CV" if pub else
          "HF resume-job-description-fit",
          "Nhãn Fit/Potential/No Fit → 2/1/0, chỉ cặp đã gán",
          f"SkillSpan test (CC-BY-4.0): {spans['n_gold_spans']} span" if spans else "SkillSpan (CC-BY-4.0)"]),
        ("Mô hình (chạy local)", "green",
         ["Dense: multilingual-e5-base", "Cross-encoder: bge-reranker-base", "NLI: nli-deberta-v3-base",
          "LLM: Qwen2.5 3B / 7B qua Ollama, CPU"]),
        ("Kiểm định thống kê", "amber",
         ["Paired bootstrap 95% CI theo JD", "Wilcoxon signed-rank + effect size r",
          "Hypothesis đặt trước khi chạy (slide 26)", "Cache LLM: chạy lại cho cùng kết quả"]),
    ]
    cw, gap, x0, y0 = 2.33, 0.2, 213 * PX, 1.95
    for i, (title, kind, lines) in enumerate(cards):
        x = x0 + i * (cw + gap)
        box(s, x, y0, cw, 2.55, fill=WHITE, line=LINE, radius=0.05)
        chip(s, x + 0.15, y0 + 0.17, f"0{i + 1}", kind)
        text(s, x + 0.15, y0 + 0.52, cw - 0.3, 0.35, title, size=11.5, bold=True)
        text(s, x + 0.15, y0 + 0.95, cw - 0.3, 1.5, [[("• " + ln, {})] for ln in lines], size=9.5, spacing=4)
    note(s, x0, 4.7, 4.75, 1.35, "Cấu hình so sánh:",
         "B0 BM25 · B1 dense · B2 hybrid RRF · B3 +Criteria Profile · B4 fixed pipeline (+cross-encoder, "
         "arbiter, claim) · LLM-all (LLM xếp lại toàn top-20) · P6 full agentic · A1 bỏ verify · "
         "A2 prompt chung · P6-noclar bỏ hỏi HR.", "blue")
    note(s, x0 + 4.95, 4.7, 4.82, 1.35, "Lưu ý khi đọc:",
         "n = 6 JD nên Wilcoxon không thể cho p < 0.031; mọi kết luận dựa trên CI95 bootstrap, không dựa trên "
         "trung bình. Gold Set tổng hợp và Criteria Profile cùng do nhóm viết, nên gain của B3 có thể lạc quan "
         "(slide dữ liệu công khai kiểm tra lại điều này).", "amber", size=9)


def slide_ladder(prs, syn, page, total):
    s = frame(prs, "Ablation B0 → P6: Criteria Profile Tạo Bước Nhảy Lớn Nhất",
              [(SECTION, "section"), ("nDCG@10 · MRR@10 · 6 JD", "blue")], page, total, *nav(page))
    order = [c for c in ("B0", "B1", "B2", "B3", "B4", "LLM-all", "P6", "A1", "A2", "P6-noclar") if c in syn["means"]]
    nd = [mean(syn, c, "ndcg@10") for c in order]
    bar_chart(s, 213 * PX, 1.9, 6.6, 4.25, order, {"nDCG@10": nd}, [GREY_BAR],
              f"nDCG@10 theo cấu hình (trung bình {syn['n_jds']} JD)", max_val=1.0,
              highlight=order.index("P6") if "P6" in order else None)
    b0, b1, b3 = mean(syn, "B0", "ndcg@10"), mean(syn, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    b2, p6 = mean(syn, "B2", "ndcg@10"), mean(syn, "P6", "ndcg@10")
    x = 213 * PX + 6.8
    w = 1173 * PX - 6.8
    items = [
        ("green", "Criteria Profile (B3)", f"nDCG@10 {fmt(b2)} → {fmt(b3)} so với hybrid B2: trọng số theo "
                                           f"công ty + knock-out là nguồn gain chính. Profile và nhãn cùng do "
                                           f"nhóm viết nên mức gain này có thể lạc quan."),
        ("amber", "Dense zero-shot (B1)", f"{fmt(b1)} < BM25 {fmt(b0)}: CV kỹ thuật nhiều tên tool, e5 chưa "
                                          f"fine-tune đánh mất tín hiệu từ khoá."),
        ("blue", "Full pipeline P6", f"nDCG@10 {fmt(p6)}, MRR@10 {fmt(mean(syn, 'P6', 'mrr@10'))}: LLM chỉ "
                                     f"chạm near-tie nên giữ chất lượng của B3/B4."),
    ]
    y = 1.95
    for kind, head, body in items:
        note(s, x, y, w, 1.3, head + ":", body, kind, size=9.5)
        y += 1.42


def _h_rows(syn) -> List[List[Any]]:
    hyp = syn.get("hypotheses") or {}

    def verdict(h):
        if h not in hyp:
            return ("chưa chạy", {"color": MUTED})
        return ("CHẤP NHẬN", {"color": GREEN, "bold": True}) if hyp[h].get("accepted") else \
            ("BÁC BỎ", {"color": RED, "bold": True})

    def det(h, key=None):
        d = (hyp.get(h) or {}).get("details") or {}
        return d.get(key) if key else d

    rows = [["", "Hypothesis (đặt trước)", "Kết quả đo (a vs b, Δ, 95% CI, p)", "Kết luận"]]
    h1_parts = []
    for metric in ("recall@100", "mrr@10"):
        pairs = det("H1", metric)
        if isinstance(pairs, list) and pairs:
            h1_parts.append(f"{metric} " + "; ".join(f"{d['a']} vs {d['b']} {fmt(d['mean_a'])} vs {fmt(d['mean_b'])}"
                                                      for d in pairs))
    rows.append(["H1", "Hybrid B2 không thấp hơn max(B0, B1) và cao hơn ít nhất một bên",
                 "\n".join(h1_parts) or "—", verdict("H1")])
    d = det("H2")
    rows.append(["H2", "Criteria Profile theo công ty: τ của P6 > A2 (prompt chung)",
                 "Kendall τ " + (cmp_text(d) if d else "—"), verdict("H2")])
    d = det("H3", "ndcg@10")
    c = det("H3", "llm_calls")
    rows.append(["H3", "Selective Arbiter: P6 không kém LLM-all (cận dưới CI95 của Δ nDCG@10 ≥ −0.02), ít LLM call hơn",
                 ("nDCG@10 " + cmp_text(d) if d else "—") + ("\njudge calls " + cmp_text(c, 1) if c else ""),
                 verdict("H3")])
    d = det("H4")
    rows.append(["H4", "verify_evidence: tỷ lệ claim không căn cứ P6 < A1",
                 "unsupported " + (cmp_text(d) if d else "—"), verdict("H4")])
    h5 = hyp.get("H5") or {}
    fail = h5.get("failing_jds")
    rows.append(["H5", "Fairness: impact ratio ≥ 0.8 (luật 4/5) ở mọi JD",
                 f"impact ratio trung bình {fmt(h5.get('impact_ratio_mean'))}"
                 + (f"; JD dưới 0.8: {len(fail)}" if fail is not None else ""), verdict("H5")])
    d = det("H6")
    rows.append(["H6", "Hỏi HR cải thiện τ trên JD mơ hồ (P6 vs P6-noclar)",
                 "Kendall τ " + (cmp_text(d) if d else "—"), verdict("H6")])
    d, u = det("H0", "ndcg@10"), det("H0", "unsupported_rate")
    rows.append(["H0", "P6 tốt hơn B4 ở nDCG@10 hoặc faithfulness, cost trong budget",
                 ("nDCG@10 " + cmp_text(d) if d else "—") + ("\nunsupported " + cmp_text(u) if u else ""),
                 verdict("H0")])
    return rows


def slide_hypotheses(prs, syn, page, total):
    s = frame(prs, "Kiểm Chứng H0–H6: Báo Cáo Cả Hypothesis Bị Bác Bỏ",
              [(SECTION, "section"), ("PAIRED BOOTSTRAP 95% CI + WILCOXON", "amber")], page, total, *nav(page))
    rows = _h_rows(syn)
    table(s, 213 * PX, 1.8, 1173 * PX, rows, [0.45, 3.4, 4.6, 1.2], size=8.5, row_h=0.47)
    text(s, 213 * PX, 1.8 + 0.47 * len(rows) + 0.12, 1173 * PX, 0.3,
         f"Quy tắc quyết định: CHẤP NHẬN khi cả CI95 bootstrap nằm đúng phía (không kém hơn: cận dưới ≥ −ε), không "
         f"xét trung bình đơn lẻ. Với n = {syn['n_jds']} JD, p Wilcoxon nhỏ nhất có thể là "
         f"{2 / 2 ** syn['n_jds']:.3f}.", size=8.5, color=MUTED)


def slide_prior(prs, syn, pub, ctx, page, total):
    s = frame(prs, "Đặt Cạnh Prior Work: Cùng Hướng Ở Đâu, Khác Ở Đâu",
              [(SECTION, "section"), ("SO HƯỚNG, KHÔNG SO TUYỆT ĐỐI", "amber")], page, total, *nav(page))
    b0, b1, b3 = mean(syn, "B0", "ndcg@10"), mean(syn, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    b4, p6, la = mean(syn, "B4", "ndcg@10"), mean(syn, "P6", "ndcg@10"), mean(syn, "LLM-all", "ndcg@10")
    calls_p6, calls_la = mean(syn, "P6", "judge_calls"), mean(syn, "LLM-all", "judge_calls")
    un_p6, un_a1 = mean(syn, "P6", "unsupported_rate"), mean(syn, "A1", "unsupported_rate")
    ir = (syn.get("hypotheses") or {}).get("H5", {}).get("impact_ratio_mean")
    saving = 1 - calls_p6 / calls_la if calls_la and not math.isnan(calls_la) and calls_la > 0 else float("nan")
    g, r = {"color": GREEN, "bold": True}, {"color": RED, "bold": True}
    h3_ok = (syn.get("hypotheses") or {}).get("H3", {}).get("accepted")
    rows = [
        ["Công trình (slide)", "Kết quả công bố", "Nhóm 2 đo được (tương ứng)", "Nhận xét"],
        ["ConFit v2 · Yu et al. 2025 (16)", "AliYun: BM25 40.56 > E5 raw 29.66 nDCG@10; Recruiting ngược lại",
         (f"BM25 {fmt(b0)} > e5 zero-shot {fmt(b1)}", {}), ("Cùng hướng: giá trị BM25 phụ thuộc domain → hybrid", g)],
        confit_row(syn, pub, ctx, g, r),
        ["RankGPT · Sun et al. 2023 (17)", "LLM listwise re-rank: +2.7 nDCG@10 (TREC)",
         f"LLM-all {fmt(la)} vs B4 {fmt(b4)} (Qwen 3B)",
         ("LLM nhỏ re-rank toàn bộ không thêm gain", r) if la <= b4 + 0.005 else
         (f"Cùng chiều nhưng nhỏ (+{la - b4:.3f}), chưa có ý nghĩa với n = {syn['n_jds']}", {"color": AMBER, "bold": True})],
        ["FrugalGPT · Chen et al. 2023 (17)", "Cascade giảm tới 98% chi phí, chất lượng ngang GPT-4",
         f"P6 {fmt(calls_p6, 1)} vs LLM-all {fmt(calls_la, 1)} judge call/JD ({pct(saving, 0)} ít hơn); "
         f"nDCG@10 {fmt(p6)} vs {fmt(la)}",
         ("Cùng hướng: ít call, không kém LLM-all (CI95)", g) if h3_ok else
         ("Một phần: ít call hơn rõ, nhưng CI95 chưa loại trừ mức kém hơn ε", {"color": AMBER, "bold": True})],
        ["Gan et al. 2024; Lo et al. 2025 (18)", "Agent: F1 87.73%, nhanh 11×; không đo nDCG, không verify",
         f"Đo bằng ranking metric + faithfulness: unsupported claim P6 {pct(un_p6)} vs A1 {pct(un_a1)}",
         "Lấp gap đo lường mà slide 18 nêu"],
        ["Wilson & Caliskan 2024 (18)", "Embedding retrieval ưu tiên tên White 85.1%, nữ 11.1%",
         f"PII masking trước encode{audit_text(ctx.get('audit'))}; impact ratio trung bình {fmt(ir)}",
         ("Đạt luật 4/5", g) if ir and ir >= 0.8 else ("Chưa đạt 4/5 ở mọi JD → cần audit thêm", r)],
    ]
    tagger = ctx.get("tagger")
    if tagger:
        tf1 = tagger["test"]["tagger"]["f1_exact"]
        rows.append(["SkillSpan · Zhang et al. 2022", f"JobBERT span F1 {SKILLSPAN_PAPER_F1 * 100:.1f} (test, 5 seed)",
                     f"JobBERT tagger span F1 {fmt(tf1)} (1 seed)" +
                     (f"; từ điển {fmt(ctx['spans']['f1_exact'])}" if ctx.get("spans") else ""),
                     ("Gần với bài báo (1 seed so với trung bình 5 seed)", g)
                     if abs(tf1 - SKILLSPAN_PAPER_F1) <= 0.03 else
                     (f"Lệch {tf1 - SKILLSPAN_PAPER_F1:+.3f} so với bài báo", {"color": AMBER, "bold": True})])
    if pub:
        pb0, pp6 = mean(pub, "B0", "ndcg@10"), mean(pub, "P6", "ndcg@10")
        rows.append(["resume-job-description-fit (HF, công khai)", "Bộ nhãn Fit / Potential / No Fit, chưa có "
                     "baseline ranking công bố", f"nDCG@10 BM25 {fmt(pb0)} → P6 {fmt(pp6)} trên {pub['n_jds']} JD",
                     ("Cùng hướng ngoài dữ liệu tổng hợp", g) if pp6 >= pb0 else
                     (f"Không lặp lại: P6 < BM25; tốt nhất là {best_config(pub)}", r)])
    table(s, 213 * PX, 1.8, 1173 * PX, rows, [2.2, 2.9, 3.0, 2.3], size=8.5, row_h=0.5 if len(rows) <= 8 else 0.44)


def confit_row(syn, pub, ctx, g, r) -> List[Any]:
    b0, b1, b3 = mean(syn, "B0", "ndcg@10"), mean(syn, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    syn_ft = ctx.get("syn_ft")
    if not syn_ft:
        return ["ConFit v2 (16)", "Fine-tune bi-encoder: +41.6 nDCG@10 so với BM25 (82.20)",
                f"Không fine-tune; Criteria Profile: {fmt(b0)} → {fmt(b3)} (+{b3 - b0:.3f})",
                "Gain đến từ tri thức tiêu chí thay vì label lớn"]
    b1_ft = mean(syn_ft, "B1", "ndcg@10")
    pub_ft = ctx.get("pub_ft")
    pub_txt, pub_beats = "", False
    if pub and pub_ft:
        pb0, pb1, pb1_ft = (mean(pub, "B0", "ndcg@10"), mean(pub, "B1", "ndcg@10"), mean(pub_ft, "B1", "ndcg@10"))
        pub_txt, pub_beats = f"; công khai {fmt(pb1)} → {fmt(pb1_ft)} vs BM25 {fmt(pb0)}", pb1_ft > pb0
    # ConFit's claim is "fine-tuned dense beats BM25", so the verdict compares B1-ft with B0, not only with B1
    if b1_ft > b0:
        verdict = ("Cùng hướng: dense fine-tune vượt BM25", g)
    elif pub_beats:
        verdict = ("Một phần: vượt BM25 trên dữ liệu công khai, chưa vượt trên Gold Set", {"color": AMBER,
                                                                                          "bold": True})
    elif b1_ft > b1 + 0.005:
        verdict = (f"Một phần: fine-tune giúp dense (+{b1_ft - b1:.3f}) nhưng chưa vượt BM25", {"color": AMBER,
                                                                                                "bold": True})
    else:
        verdict = ("Không lặp lại: fine-tune trên JD khác ngành không giúp", r)
    return ["ConFit v2 (16)", "Fine-tune bi-encoder: +41.6 nDCG@10 so với BM25 (82.20)",
            f"e5 fine-tune (cặp không phải QA), B1 Gold Set {fmt(b1)} → {fmt(b1_ft)} vs BM25 {fmt(b0)}"
            f"{pub_txt}; Criteria Profile {fmt(b3)}", verdict]


def audit_text(audit: Optional[List[Dict[str, Any]]]) -> str:
    if not audit:
        return ""
    worst = max(a["max_abs_delta_rank"] for a in audit)
    return f"; đổi tên: Δrank tối đa {worst} trên {len(audit)} JD"


def best_config(summ, metric="ndcg@10"):
    name = max(summ["means"], key=lambda c: mean(summ, c, metric) or 0)
    return f"{name} {fmt(mean(summ, name, metric))}"


def slide_faith(prs, syn, page, total):
    s = frame(prs, "Faithfulness & Chi Phí: verify_evidence Cắt Claim Bịa, Arbiter Giữ Cost Thấp",
              [(SECTION, "section"), ("H0 · H3 · H4", "green")], page, total, *nav(page))
    cfgs = [c for c in ("B4", "LLM-all", "P6", "A1") if c in syn["means"]]
    bar_chart(s, 213 * PX, 1.9, 4.75, 4.25, cfgs, {"unsupported": [mean(syn, c, "unsupported_rate") for c in cfgs]},
              [RGBColor(0xF4, 0x3F, 0x5E)], "Tỷ lệ claim không có bằng chứng", number_format="0%",
              highlight=cfgs.index("P6") if "P6" in cfgs else None)
    bar_chart(s, 213 * PX + 4.9, 1.9, 4.85, 4.25, cfgs,
              {"LLM calls/JD": [mean(syn, c, "llm_calls") for c in cfgs],
               "judge calls/JD": [mean(syn, c, "judge_calls") for c in cfgs]},
              [GREY_BAR, BLUE], "Lời gọi LLM mỗi JD", number_format="0.0", legend=True)


def slide_agent(prs, h7: Dict[str, Dict[str, Any]], page, total):
    s = frame(prs, "Agentic AI (H7): Planner LLM Hợp Lệ Hơn Ở 7B, Nhưng Chưa Rẻ Hơn Luật",
              [(SECTION, "section"), ("H7 · AG vs AG-rules", "violet")], page, total, *nav(page))
    rows = [["Lượt chạy", "Cấu hình", "nDCG@10", "Claim không căn cứ", "LLM call/JD", "Bước agent",
             "Luật phải thay LLM", "Đạt mục tiêu", "H7"]]
    for label, summ in h7.items():
        hyp = (summ.get("hypotheses") or {}).get("H7", {})
        for cfg in ("AG-rules", "AG"):
            if cfg not in summ["means"]:
                continue
            m = summ["means"][cfg]
            goal = m.get("goal_met")
            fb = m.get("planner_fallback_rate") if cfg == "AG" else None
            verdict = (("CHẤP NHẬN", {"color": GREEN, "bold": True}) if hyp.get("accepted")
                       else ("BÁC BỎ", {"color": RED, "bold": True})) if cfg == "AG" else ""
            rows.append([label if cfg == "AG-rules" else "", cfg, fmt(m.get("ndcg@10")), pct(m.get("unsupported_rate")),
                         fmt(m.get("llm_calls"), 1), fmt(m.get("agent_steps"), 1), pct(fb, 0) if fb is not None
                         else "—", f"{round(goal * summ['n_jds'])}/{summ['n_jds']}" if goal is not None else "—",
                         verdict])
    table(s, 213 * PX, 1.85, 1173 * PX, rows, [2.1, 1.0, 0.85, 1.25, 1.0, 0.9, 1.25, 1.0, 1.0], size=9,
          row_h=0.36)
    y = 2.05 + 0.36 * len(rows)
    note(s, 213 * PX, y, 4.8, min(1.5, 6.15 - y), "Điều đã kiểm chứng:",
         "planner LLM chỉ chọn trong công cụ có guardrail; HR checkpoint, knock-out và ngân sách không bị vượt. "
         "Với 7B, luật hiếm khi phải thay quyết định của LLM.", "green")
    note(s, 213 * PX + 4.97, y, 4.8, min(1.5, 6.15 - y), "Điều chưa đạt:",
         "trên dữ liệu tổng hợp, quyết định 'đúng' đã được luật mã hoá sẵn, nên AG tốn thêm LLM call mà không "
         "tăng nDCG. Cần Gold Set thật để thấy lợi ích của tự lập kế hoạch.", "amber")


SKILLSPAN_PAPER_F1 = 0.597  # Zhang et al. 2022, Table 5: JobBERT, COMBINED, TEST, mean of 5 seeds


def _span_rows(spans, tagger) -> List[List[Any]]:
    cols = [("Từ điển taxonomy", spans)]
    if tagger:
        cols += [(label, tagger["test"][key]) for key, label in (("tagger", "JobBERT tagger"),
                                                                ("hybrid", "Hai bộ hợp lại"))
                 if key in tagger["test"]]
    rows = [["SkillSpan test"] + [c for c, _ in cols]]
    for key, label in (("precision_exact", "Precision (chính xác)"), ("recall_exact", "Recall (chính xác)"),
                       ("f1_exact", "F1 (chính xác)"), ("f1_overlap", "F1 (chồng lấn)")):
        best = max((r[key] for _, r in cols if not math.isnan(r[key])), default=float("nan"))
        win = len(cols) > 1
        rows.append([label] + [(fmt(r[key]), {"bold": win and r[key] == best,
                                              "color": GREEN if win and r[key] == best else INK})
                               for _, r in cols])
    return rows


def slide_public(prs, pub, spans, page, total, ctx=None):
    s = frame(prs, "Dữ Liệu Công Khai & Trích Xuất Skill: Kiểm Tra Ngoài Bộ Dữ Liệu Tự Tạo",
              [(SECTION, "section"), ("HF resume-job-description-fit · SkillSpan", "violet")], page, total, *nav(page))
    x0 = 213 * PX
    if pub:
        order = [c for c in ("B0", "B1", "B2", "B3", "B4", "P6") if c in pub["means"]]
        bar_chart(s, x0, 1.9, 5.3, 2.95, order, {"nDCG@10": [mean(pub, c, "ndcg@10") for c in order]}, [GREY_BAR],
                  f"Công khai: nDCG@10 ({pub['n_jds']} JD, {pub['n_cvs']} CV)", max_val=1.1,
                  highlight=order.index("P6") if "P6" in order else None)
        note(s, x0, 4.95, 5.3, 1.2, "Trung bình 6 JD không lặp lại:",
             f"{best_config(pub)} tốt nhất, P6 {fmt(mean(pub, 'P6', 'ndcg@10'))} thấp hơn BM25 "
             f"{fmt(mean(pub, 'B0', 'ndcg@10'))}." + (" Slide sau tách theo JD để tìm nguyên nhân."
                                                      if (ctx or {}).get("errors") else ""), "amber", size=9)
    tagger = (ctx or {}).get("tagger")
    if spans:
        rows = _span_rows(spans, tagger)
        table(s, x0 + 5.5, 1.9, 4.28, rows, [1.45] + [0.94] * (len(rows[0]) - 1), size=9, row_h=0.32)
        if tagger:
            t = tagger["test"]["tagger"]
            body = (f"từ điển chỉ biết thuật ngữ Automation Test nên recall {fmt(spans['recall_exact'])}. Tagger "
                    f"JobBERT học trên SkillSpan train ({tagger['data']['train']} câu, chọn epoch theo dev) đạt span "
                    f"F1 {fmt(t['f1_exact'])}; bài báo SkillSpan báo cáo {SKILLSPAN_PAPER_F1:.3f} (JobBERT, 5 seed). "
                    "Hệ thống dùng từ điển để gắn concept ID làm bằng chứng, tagger để không bỏ sót skill ngoài "
                    "từ điển.")
        else:
            body = (f"bộ trích xuất là từ điển ESCO/taxonomy nên precision cao ({pct(spans['precision_exact'], 0)}) "
                    "nhưng recall thấp: SkillSpan gán cả kỹ năng mềm và span dài ngoài taxonomy Automation Test.")
        note(s, x0 + 5.5, 3.65, 4.28, 2.5, "Đọc kết quả:", body, "violet", size=9)


DUPLICATE_JACCARD = 0.8  # same threshold as cv_agent.eval.error_analysis


def _jd_label(jd: str, dup_of: Dict[str, int]) -> str:
    return f"…{jd[-6:]}" + (f"  (tin trùng #{dup_of[jd]})" if jd in dup_of else "")


def slide_errors(prs, pub, ctx, page, total):
    ea = ctx["errors"]
    dups = ea["near_duplicates"]
    dup_of = {jd: i + 1 for i, grp in enumerate(dups) for jd in grp}
    n_dup = sum(len(grp) for grp in dups)
    d, g = ea["dedup"], ea["groups"]
    dup_signals = sorted({sig for grp in dups for jd in grp for sig in ea["per_jd"][jd]["qa_signal_list"]})
    what = "Một Tin Đăng Lặp Lại" if len(dups) == 1 else f"{len(dups)} Tin Đăng Lặp Lại"
    s = frame(prs, f"Phân Tích Lỗi: {n_dup}/{pub['n_jds']} JD Công Khai Là {what}, Chỉ {len(dup_signals)} "
                   "Tín Hiệu QA",
              [(SECTION, "section"), ("POST HOC · KHÔNG PHẢI HYPOTHESIS", "amber")], page, total, *nav(page))
    x0 = 213 * PX
    cfgs = [c for c in ("B0", "B1", "B3", "P6") if c in d]
    rows = [["JD công khai", "Tín hiệu QA"] + cfgs + ["Claim không căn cứ P6"]]
    for jd, info in sorted(ea["per_jd"].items(), key=lambda kv: (kv[0] in dup_of, kv[0])):
        vals = info["ndcg@10"]
        best = max(vals.get(c, 0) for c in cfgs)
        rows.append([_jd_label(jd, dup_of), str(info["qa_signals"])] +
                    [(fmt(vals.get(c)), {"bold": vals.get(c) == best, "color": GREEN if vals.get(c) == best else INK})
                     for c in cfgs] + [pct(info["unsupported_rate"].get("P6"), 0)])
    text(s, x0, 1.82, 5.4, 0.3, "nDCG@10 theo từng JD (mô hình gốc)", size=10, bold=True)
    table(s, x0, 2.12, 5.4, rows, [1.75, 0.65] + [0.55] * len(cfgs) + [0.8], size=8.5, row_h=0.31)
    y = 2.3 + 0.31 * len(rows)
    quoted = ", ".join(f"'{sig}'" for sig in dup_signals)
    note(s, x0, y, 5.4, 6.12 - y, "Nguyên nhân:",
         f"{n_dup} JD trùng nhau (độ trùng 3-gram ≥ {DUPLICATE_JACCARD}), lọt bộ lọc QA chỉ nhờ {quoted}: đọc văn "
         f"bản thì đây là tin tuyển lập trình viên hạ tầng, không phải Automation Test. Criteria Profile áp tiêu chí "
         f"sai vai trò: B3 {fmt(g['distinct']['B3'])} trên JD khác nhau, {fmt(g['duplicate_postings']['B3'])} trên "
         f"tin trùng, và tin này bị tính {n_dup} lần trong trung bình.", "amber", size=9)
    groups = [("JD khác nhau", g["distinct"]), (f"Tin trùng ({n_dup})", g["duplicate_postings"]),
              (f"Bỏ trùng (n = {ea['effective_n_jds']})", d)]
    bar_chart(s, x0 + 5.6, 1.85, 4.18, 2.75, [lab for lab, _ in groups],
              {c: [m[c] for _, m in groups] for c in cfgs}, [GREY_BAR, BLUE, GREEN, VIOLET][:len(cfgs)],
              "nDCG@10 theo nhóm JD", number_format="0.00", max_val=1.1, legend=True)
    best_d, worst_d = max(cfgs, key=lambda c: d[c]), min(cfgs, key=lambda c: d[c])
    note(s, x0 + 5.6, 4.72, 4.18, 1.4, "Đọc cẩn thận:",
         f"khi mỗi tin trùng chỉ tính một lần, {best_d} cao nhất ({fmt(d[best_d])}), {worst_d} thấp nhất "
         f"({fmt(d[worst_d])}). Phân tích làm sau khi thấy kết quả nên chỉ là gợi ý; tập đánh giá giữ nguyên, "
         "bước tiếp theo là lọc JD theo tiêu đề và bỏ tin trùng trước khi chạy lại.", "blue", size=9)


def slide_conclusion(prs, syn, pub, ctx, page, total):
    s = frame(prs, "Kết Luận Thực Nghiệm: Điều Đứng Vững, Điều Chưa Và Hướng Tiếp",
              [(SECTION, "section"), ("THREATS TO VALIDITY", "amber")], page, total, *nav(page))
    p6, b0 = mean(syn, "P6", "ndcg@10"), mean(syn, "B0", "ndcg@10")
    stats = [(fmt(p6), "nDCG@10 của P6"), (f"n = {syn['n_jds']}", "JD mỗi lần chạy"),
             ("~200", "CV thật cần gán nhãn")]
    cols = [
        ("green", "Đứng vững", [
            f"Gold Set: nDCG@10 BM25 {fmt(b0)} → P6 {fmt(p6)} nhờ Criteria Profile",
            "verify_evidence giảm claim không căn cứ (H4/H0)",
            "Arbiter chọn lọc ít judge call hơn LLM-all (CI95 < 0)",
            "Agent có guardrail: không vượt checkpoint HR",
            *([f"Tagger JobBERT: span F1 {fmt(ctx['tagger']['test']['tagger']['f1_exact'])} trên SkillSpan "
               f"(bài báo {SKILLSPAN_PAPER_F1})"] if (ctx.get("tagger") or {}).get("test", {}).get("tagger") else []),
            "Đổi tên ứng viên: Δrank = 0 (PII che trước encode)" if worst_swap(ctx) == 0 else
            "PII che trước mọi bước encode / gọi LLM"]),
        ("amber", "Chưa đạt / cần thận trọng", [
            (f"Công khai: P6 < BM25; {sum(map(len, ctx['errors']['near_duplicates']))}/{pub['n_jds']} JD là "
             "tin trùng, sai vai trò" if ctx.get("errors") and ctx["errors"]["near_duplicates"] else
             "Dữ liệu công khai: P6 không hơn BM25") if pub and mean(pub, "P6", "ndcg@10") <
            mean(pub, "B0", "ndcg@10") else "Dense zero-shot kém BM25 trên Gold Set",
            ", ".join(h for h in ("H1", "H2", "H3", "H4", "H5", "H6")
                      if not syn["hypotheses"].get(h, {}).get("accepted")) + " bị bác bỏ: báo cáo như finding",
            "LLM planner chưa thắng luật về chi phí",
            finetune_bullet(syn, ctx),
            "Profile và nhãn Gold Set cùng do nhóm viết: gain B3 có thể lạc quan",
            f"n = {syn['n_jds']} JD: CI rộng, p Wilcoxon ≥ {2 / 2 ** syn['n_jds']:.3f}"]),
        ("blue", "Hướng tiếp theo", [
            "Gold Set thật ~200 CV, 2 QA Lead gán nhãn độc lập với người viết profile, đo Cohen κ",
            "Fine-tune trên Gold Set thật: notebook Colab có sẵn" if ctx.get("ft") else
            "Fine-tune bi-encoder (hướng ConFit) khi đủ label",
            "Lọc JD theo tiêu đề, bỏ tin trùng; audit fairness trên dữ liệu thật",
            "GPU / LLM 7B+ để giảm latency"]),
    ]
    cw, x0 = 3.13, 213 * PX
    for i, (kind, head, items) in enumerate(cols):
        x = x0 + i * (cw + 0.2)
        fg, bg = CHIPS[kind]
        box(s, x, 1.95, cw, 4.15, fill=bg, line=None, radius=0.05)
        text(s, x + 0.2, 2.1, cw - 0.4, 0.4, head, size=13, bold=True, color=fg)
        big, small = stats[i]
        text(s, x + 0.2, 2.55, cw - 0.4, 0.6, big, size=30, bold=True, color=fg)
        text(s, x + 0.2, 3.18, cw - 0.4, 0.3, small, size=9.5, color=MUTED)
        text(s, x + 0.2, 3.6, cw - 0.4, 2.45, [[("• " + it, {})] for it in items], size=10, spacing=5)


def worst_swap(ctx) -> Optional[int]:
    return max((a["max_abs_delta_rank"] for a in ctx.get("audit") or []), default=None)


def finetune_bullet(syn, ctx) -> str:
    syn_ft = ctx.get("syn_ft")
    if not syn_ft:
        return "Dense zero-shot kém BM25, chưa fine-tune"
    b1, b1_ft, b3 = mean(syn, "B1", "ndcg@10"), mean(syn_ft, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    tail = "vẫn sau Criteria Profile" if b1_ft < b3 else "ngang/hơn Criteria Profile"
    return f"Fine-tune dense: B1 {fmt(b1)} → {fmt(b1_ft)}, {tail}"


def _ft_table_rows(ft: Dict[str, Dict[str, Any]], seeds: Optional[Dict[str, Any]] = None) -> List[List[Any]]:
    """Before → after per split; with a multi-seed summary, after is the mean ± sd over seeds."""
    rows = [["Mô hình", "Ví dụ train", "nDCG@10 dev", "nDCG@10 test"]]
    names = {"dense": "Dense e5-base", "rerank": "Cross-encoder bge"}
    summary = (seeds or {}).get("summary", {})
    for target, rep in ft.items():
        cells = []
        for split in ("dev", "test"):
            before, after = rep["before"][split]["ndcg@10"], rep["after"][split]["ndcg@10"]
            m = summary.get(target, {}).get(split)
            if m:
                before, after = m["after_mean"] - m["gain_mean"], m["after_mean"]
            style = {"color": GREEN if after > before else RED, "bold": True}
            spread = f" ± {m['after_sd']:.3f}" if m else ""
            cells.append((f"{fmt(before)} → {fmt(after)}{spread}", style))
        unit = "triplet" if target == "dense" else "cặp"
        rows.append([names.get(target, target), f"{rep['n_train_examples']} {unit}", *cells])
    return rows


def _seed_reading(seeds: Dict[str, Any]) -> str:
    """Which gains stand out from the seed-to-seed spread (gain > 2 sd) and which do not."""
    names = {"dense": "dense", "rerank": "cross-encoder"}
    weak = [f"{names.get(t, t)} {split} (+{fmt(m['gain_mean'])} ± {m['gain_sd']:.3f})"
            for t, splits in seeds["summary"].items() for split, m in splits.items()
            if not m["gain_mean"] > 2 * m["gain_sd"]]
    return ("Mọi gain lớn hơn 2 lần độ lệch chuẩn." if not weak else
            "Gain lớn hơn 2 lần độ lệch chuẩn, trừ " + ", ".join(weak) + ": chưa phân biệt được với nhiễu seed.")


def slide_finetune(prs, syn, pub, ctx, page, total):
    ft, syn_ft, pub_ft = ctx["ft"], ctx.get("syn_ft"), ctx.get("pub_ft")
    b1 = mean(syn, "B1", "ndcg@10")
    b1_ft = mean(syn_ft, "B1", "ndcg@10") if syn_ft else float("nan")
    title = (f"Fine-tune Kiểu ConFit Trên Colab: Dense B1 {fmt(b1)} → {fmt(b1_ft)} Trên Gold Set" if syn_ft else
             "Fine-tune Kiểu ConFit Trên Colab: Dense Encoder Và Cross-encoder")
    s = frame(prs, title,
              [(SECTION, "section"), ("COLAB T4 · notebooks/finetune_colab.ipynb", "violet")], page, total, *nav(page))
    x0 = 213 * PX
    text(s, x0, 1.88, 4.6, 0.3, "Trên cặp công khai không phải QA (trước → sau fine-tune)", size=10, bold=True)
    seeds = ctx.get("ft_seeds")
    table(s, x0, 2.18, 4.6, _ft_table_rows(ft, seeds), [1.3, 0.8, 1.25, 1.25], size=9, row_h=0.34)
    any_rep = next(iter(ft.values()))
    d = any_rep["data"]
    note(s, x0, 3.4, 4.6, 1.25, "Chống rò rỉ:",
         f"bỏ mọi JD QA (chính là JD đánh giá) và mọi CV từng gán cho chúng. Train {d['train']['jds']} JD / "
         f"dev {d['dev']['jds']} JD chia theo JD; test {d['test']['jds']} JD lấy từ test.csv, JD chưa gặp. "
         "Dữ liệu tải thẳng từ Hugging Face: không CV nào rời máy nhóm.", "violet", size=9)
    note(s, x0, 4.8, 4.6, 1.35, "Cách làm:",
         f"triplet (JD, Good Fit, No Fit) + contrastive loss (in-batch negative) cho e5; BCE nhãn 0 / 0.5 / 1 cho "
         f"cross-encoder; {any_rep['config']['epochs']} epoch, max_len {any_rep['config']['max_len']}, " +
         (f"{len(seeds['seeds'])} seed ({', '.join(map(str, seeds['seeds']))}) trên Colab T4, bảng là trung bình ± "
          f"độ lệch chuẩn. {_seed_reading(seeds)} Biểu đồ dùng mô hình seed {any_rep['config']['seed']}."
          if seeds else
          f"1 seed ({any_rep['config']['seed']}), chưa có độ lệch chuẩn. Notebook sinh từ cùng module, mặc định "
          "chạy 3 seed để báo cáo trung bình ± độ lệch chuẩn."), "blue", size=9)
    cfgs = [c for c in ("B1", "B2", "B3", "P6") if syn_ft and c in syn_ft["means"]]
    if cfgs:
        bar_chart(s, x0 + 4.8, 1.88, 4.98, 2.15, cfgs,
                  {"Mô hình gốc": [mean(syn, c, "ndcg@10") for c in cfgs],
                   "Fine-tune": [mean(syn_ft, c, "ndcg@10") for c in cfgs]},
                  [GREY_BAR, VIOLET], "Gold Set tổng hợp: nDCG@10", max_val=1.1, legend=True)
    pcfgs = [c for c in ("B1", "B2", "B3", "P6") if pub and pub_ft and c in pub_ft["means"]]
    if pcfgs:
        bar_chart(s, x0 + 4.8, 4.05, 4.98, 2.1, pcfgs,
                  {"Mô hình gốc": [mean(pub, c, "ndcg@10") for c in pcfgs],
                   "Fine-tune": [mean(pub_ft, c, "ndcg@10") for c in pcfgs]},
                  [GREY_BAR, VIOLET], "Công khai (JD QA): nDCG@10", max_val=1.1, legend=True)


def slide_nfr(prs, syn, ctx, page, total):
    audit, bench = ctx.get("audit") or [], ctx.get("bench")
    worst = worst_swap(ctx)
    swap_txt = "Đổi Tên Không Đổi Thứ Hạng" if worst == 0 else "Audit Đổi Tên Và Luật 4/5"
    p50 = min((m["summary"]["wall_p50_s"] for m in bench["models"].values()), default=None) if bench else None
    lat_txt = f", ~{p50 / 60:.0f} Phút/JD Trên CPU" if p50 else ""
    s = frame(prs, f"Fairness & Latency: {swap_txt}{lat_txt}",
              [(SECTION, "section"), ("NFR-04 · NFR-06 · H5", "green")], page, total, *nav(page))
    x0 = 213 * PX
    ir = (syn.get("hypotheses") or {}).get("H5", {}).get("impact_ratio_by_jd") or {}
    swap = {a["jd"]: a for a in audit}
    rows = [["JD (Gold Set)", "Impact ratio (≥ 0.8)", "Δrank đổi tên (TB / max)"]]
    for jd in ir or swap:
        r = ir.get(jd)
        a = swap.get(jd)
        rows.append([jd, (fmt(r, 2), {"color": GREEN if r is not None and r >= 0.8 else RED, "bold": True}),
                     f"{a['mean_abs_delta_rank']:.1f} / {a['max_abs_delta_rank']}" if a else "—"])
    text(s, x0, 1.88, 4.9, 0.3, "Fairness: luật 4/5 và audit đổi tên (B3, 200 CV/JD)", size=10, bold=True)
    table(s, x0, 2.18, 4.9, rows, [2.1, 1.35, 1.45], size=9, row_h=0.3)
    y = 2.35 + 0.3 * len(rows)
    failing = [jd for jd, v in ir.items() if v < 0.8]
    swap_msg = ("đổi tên nam/nữ, Việt/nước ngoài không làm đổi thứ hạng vì tên bị che trước mọi bước encode và "
                "gọi LLM. " if worst == 0 else
                f"đổi tên làm thứ hạng lệch tối đa {worst} bậc: cần tìm chỗ tên lọt qua bước che PII. " if audit else "")
    ir_msg = (f"Impact ratio < 0.8 ở {len(failing)}/{len(ir)} JD: tên đã bị che nên nguồn lệch là thuộc tính khác "
              "tương quan với nhóm (ví dụ số năm kinh nghiệm); cần audit lại trên CV thật." if failing else
              "Impact ratio ≥ 0.8 ở mọi JD.")
    note(s, x0, y, 4.9, min(1.45, 6.1 - y), "Đọc kết quả:", swap_msg + ir_msg, "amber" if failing else "green",
         size=9)
    if bench:
        brow = [["Model (P6, CPU)", "p50 / JD", "p95 / JD", "LLM call", "Fallback"]]
        for model, rep in bench["models"].items():
            sm = rep["summary"]
            brow.append([model, f"{sm['wall_p50_s']:.0f}s", f"{sm['wall_p95_s']:.0f}s", fmt(sm["calls_per_jd"], 1),
                         str(sm["fallbacks"])])
        text(s, x0 + 5.1, 1.88, 4.68, 0.3, "Latency: bench P6, tắt cache LLM, Ollama trên CPU, không GPU", size=10,
             bold=True)
        table(s, x0 + 5.1, 2.18, 4.68, brow, [1.4, 0.8, 0.8, 0.85, 0.8], size=9, row_h=0.32)
        fast = min(bench["models"].items(), key=lambda kv: kv[1]["summary"]["wall_p50_s"])
        y = 2.35 + 0.32 * len(brow)
        fsum = fast[1]["summary"]
        ok = fsum["wall_p95_s"] <= NFR04_MAX_S and fsum["fallbacks"] == 0
        fb = "không lần nào fallback" if fsum["fallbacks"] == 0 else f"{fsum['fallbacks']} lần fallback"
        note(s, x0 + 5.1, y, 4.68, 1.45, "NFR-04 đạt:" if ok else "NFR-04 chưa đạt:",
             f"{fast[0]} xếp một JD trong ~{fsum['wall_p50_s'] / 60:.0f} phút (p50), p95 "
             f"~{fsum['wall_p95_s'] / 60:.0f} phút trên laptop không GPU, {fb}. Arbiter chỉ xét cặp có thể đổi "
             "top 10; prompt gọn theo dòng bằng chứng. GPU (Colab) rút ngắn cả fine-tune lẫn suy luận.",
             "green" if ok else "amber", size=9)


SPEAKER_NOTES = {
    "setup": "Giới thiệu dữ liệu, mô hình và cách kiểm định. Nhấn mạnh: mọi thứ chạy local, không gửi PII ra ngoài; "
             "hypothesis đặt trước khi chạy; n = 6 JD nên đọc CI, đừng đọc p-value một mình.",
    "ladder": "Bước nhảy lớn nhất là Criteria Profile (B2 → B3). Dense zero-shot kém BM25 trên CV kỹ thuật. "
              "LLM và arbiter chỉ tinh chỉnh ở top, nên P6 giữ chất lượng của B3/B4.",
    "hypotheses": "Quy tắc: chỉ chấp nhận khi cả CI95 nằm đúng phía; với n = 6, p Wilcoxon nhỏ nhất là 0.031. "
                  "Đọc từng dòng: hypothesis, con số đo, CI 95%, kết luận. Hypothesis bị bác bỏ vẫn được báo cáo như "
                  "finding, đúng kế hoạch thực nghiệm.",
    "prior": "Không so tuyệt đối vì khác dữ liệu; so hướng. Cột cuối nói cùng hướng hay không lặp lại được.",
    "faith": "verify_evidence là đóng góp rõ nhất: claim không căn cứ giảm mạnh so với A1. Arbiter chọn lọc gọi LLM ít "
             "hơn LLM-all nhiều lần; nDCG thấp hơn 0.012 nhưng CI95 chưa loại trừ mức kém hơn ε = 0.02 nên H3 bị bác bỏ.",
    "finetune": "Phần huấn luyện chạy trên Colab bằng notebook sinh từ code của repo. Dữ liệu train là cặp công khai "
                "không phải QA để không rò rỉ sang tập đánh giá. Đọc bảng trái (dữ liệu cùng phân bố với train) "
                "rồi biểu đồ phải (tập đánh giá của nhóm). Bảng là trung bình 3 seed: gain test của cả hai mô hình lớn "
                "hơn nhiều lần độ lệch chuẩn, còn gain dev của dense nằm trong nhiễu seed.",
    "agent": "Planner LLM chỉ chọn trong công cụ có guardrail. Với 7B sau khi sửa ngân sách, H7 được chấp nhận nhưng "
             "AG vẫn tốn LLM call hơn luật.",
    "public": "Trên dữ liệu công khai lợi thế không lặp lại ở trung bình: slide sau giải thích. SkillSpan: từ điển "
              "precision cao nhưng recall thấp vì chỉ biết thuật ngữ Automation Test; tagger JobBERT học trên SkillSpan "
              "train đạt span F1 gần mức bài báo (1 seed). Gộp tagger với từ điển không tăng F1: từ điển chỉ thêm ít span "
              "đúng mà precision giảm.",
    "errors": "Phân tích sau khi thấy kết quả, không phải hypothesis. Ba JD công khai là cùng một tin tuyển lập trình "
              "viên Ansible, lọt bộ lọc QA nhờ hai cụm từ lỏng; profile Automation Test áp sai vai trò. Bỏ trùng thì "
              "thứ tự thay đổi, nhưng nhóm không đổi tập đánh giá: bước tiếp theo là lọc JD tốt hơn rồi chạy lại.",
    "nfr": "Fairness: audit đổi tên cho Δrank = 0 nhờ che PII trước encode; luật 4/5 chưa đạt ở mọi JD. Latency: "
           "p50/p95 đo khi tắt cache, đạt mục tiêu vài phút một JD trên laptop.",
    "conclusion": "Tóm tắt điều đứng vững, điều chưa đạt và hướng tiếp theo; mời câu hỏi.",
}


# ------------------------------------------------------------------ patches on existing image slides

def patch_existing(prs, first_new: int, n_new: int) -> None:
    """Slide 26 says 'no results yet' and slide 27 marks the ladder 'not run': overlay updated notes."""
    s26, s27 = prs.slides[25], prs.slides[26]
    box(s26, 904 * PX, 651 * PX, 483 * PX, 55 * PX, fill=GREEN_BG, line=None, radius=0.12, name="Results note")
    text(s26, 916 * PX, 657 * PX, 460 * PX, 43 * PX,
         [[("Đã có kết quả thực nghiệm của nhóm. ", {"bold": True, "color": GREEN}),
           (f"Xem slide {first_new}–{first_new + n_new - 1}: H0–H7 được kiểm chứng trên dữ liệu thật chạy local.", {})]],
         size=8.5, anchor=MSO_ANCHOR.MIDDLE)
    box(s27, 733 * PX, 316 * PX, 149 * PX, 17 * PX, fill=WHITE, line=None, radius=None, name="Ran note")
    text(s27, 733 * PX, 317 * PX, 146 * PX, 15 * PX, f"Đã chạy → slide {first_new + 1}", size=6.5,
         color=GREEN, bold=True, align=PP_ALIGN.RIGHT)


def move_before_last(prs, n_new: int) -> None:
    ids = prs.slides._sldIdLst
    items = list(ids)
    new, closing = items[-n_new:], items[-n_new - 1]
    ids.remove(closing)
    ids.append(closing)
    del new


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default="Slide_Group2_NLP501-2.pptx")
    ap.add_argument("--out", default="Slide_Group2_NLP501-3.pptx")
    ap.add_argument("--synthetic", required=True, help="run dir with the full B0..P6 ladder")
    ap.add_argument("--public", help="run dir on data/public")
    ap.add_argument("--h7", action="append", default=[], help="label=run_dir, in display order")
    ap.add_argument("--spans", help="JSON printed by `cv_agent eval-spans`")
    ap.add_argument("--finetune", action="append", default=[], help="model dir with finetune_metrics.json")
    ap.add_argument("--finetune-seeds", help="JSON with a seed_summary of several fine-tune seeds")
    ap.add_argument("--synthetic-ft", help="run dir: same ladder with the fine-tuned models")
    ap.add_argument("--public-ft", help="run dir on data/public with the fine-tuned models")
    ap.add_argument("--audit", help="JSON lines printed by `cv_agent audit`, one per JD")
    ap.add_argument("--bench", help="bench.json written by `cv_agent bench`")
    ap.add_argument("--errors", help="error_analysis.json written by `cv_agent error-analysis` (public run)")
    ap.add_argument("--tagger", help="model dir with tagger_metrics.json from `cv_agent train-tagger`")
    args = ap.parse_args()

    syn = load_summary(Path(args.synthetic))
    pub = load_summary(Path(args.public)) if args.public else None
    spans = json.loads(Path(args.spans).read_text(encoding="utf-8")) if args.spans else None
    h7 = {lab: load_summary(Path(p)) for lab, p in (item.split("=", 1) for item in args.h7)}
    ft = {}
    for d in args.finetune:
        rep = json.loads((Path(d) / "finetune_metrics.json").read_text(encoding="utf-8"))
        ft[rep["config"]["target"]] = rep
    ctx = {"ft": ft, "ft_seeds": json.loads(Path(args.finetune_seeds).read_text(encoding="utf-8"))
           if args.finetune_seeds else None, "syn_ft": load_summary(Path(args.synthetic_ft)) if args.synthetic_ft else None,
           "pub_ft": load_summary(Path(args.public_ft)) if args.public_ft else None,
           "audit": [json.loads(line) for line in Path(args.audit).read_text(encoding="utf-8").splitlines()
                     if line.strip()] if args.audit else None,
           "bench": json.loads(Path(args.bench).read_text(encoding="utf-8")) if args.bench else None,
           "errors": json.loads(Path(args.errors).read_text(encoding="utf-8")) if args.errors else None,
           "tagger": json.loads((Path(args.tagger) / "tagger_metrics.json").read_text(encoding="utf-8"))
           if args.tagger else None, "spans": spans}

    prs = Presentation(args.deck)
    n_before = len(prs.slides)
    # (key, short label, question that leads to the slide, builder)
    plan = [
        ("setup", "Thiết lập", "", lambda pg, t: slide_setup(prs, syn, pub, spans, pg, t)),
        ("ladder", "Ablation B0 → P6", "Cầu thang cấu hình B0 → P6 cho kết quả ra sao?",
         lambda pg, t: slide_ladder(prs, syn, pg, t)),
        ("hypotheses", "Kiểm chứng H0–H6", "Các hypothesis H0–H6 được chấp nhận hay bác bỏ?",
         lambda pg, t: slide_hypotheses(prs, syn, pg, t)),
        ("prior", "So với prior work", "Đặt cạnh con số của các công trình trước thì sao?",
         lambda pg, t: slide_prior(prs, syn, pub, ctx, pg, t)),
        ("faith", "Faithfulness & cost", "Phần verify và chi phí LLM đóng góp gì?",
         lambda pg, t: slide_faith(prs, syn, pg, t)),
        ("finetune", "Fine-tune trên Colab", "Fine-tune dense encoder kiểu ConFit có lấp được điểm yếu của B1?",
         lambda pg, t: slide_finetune(prs, syn, pub, ctx, pg, t)),
        ("agent", "Agentic H7", "Agent LLM tự lập kế hoạch có tốt hơn luật cố định?",
         lambda pg, t: slide_agent(prs, h7, pg, t)),
        ("public", "Public data & SkillSpan", "Dữ liệu công khai và trích xuất skill cho kết quả gì?",
         lambda pg, t: slide_public(prs, pub, spans, pg, t, ctx)),
        ("errors", "Phân tích lỗi", "Vì sao trên dữ liệu công khai Criteria Profile lại thua?",
         lambda pg, t: slide_errors(prs, pub, ctx, pg, t)),
        ("nfr", "Fairness & latency", "Có thiên lệch theo tên không, và một JD chạy mất bao lâu?",
         lambda pg, t: slide_nfr(prs, syn, ctx, pg, t)),
        ("conclusion", "Kết luận", "Tổng kết: điều gì đứng vững, điều gì cần làm tiếp?",
         lambda pg, t: slide_conclusion(prs, syn, pub, ctx, pg, t)),
    ]
    skip = {"finetune"} if not ft else set()
    skip |= {"nfr"} if not (ctx["audit"] or ctx["bench"]) else set()
    skip |= {"errors"} if not (pub and ctx["errors"] and ctx["errors"]["near_duplicates"]) else set()
    plan = [p for p in plan if p[0] not in skip]
    set_nav(n_before, [(label, question) for _, label, question, _ in plan])
    for i, (key, _, _, build) in enumerate(plan, 1):
        build(i, len(plan))
        prs.slides[-1].notes_slide.notes_text_frame.text = SPEAKER_NOTES[key]
    move_before_last(prs, len(plan))
    patch_existing(prs, n_before, len(plan))
    prs.save(args.out)
    print(f"wrote {args.out}: {n_before} -> {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
