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
TITLE_ONE_LINE = 66  # characters that fit on one line at 19 pt; longer titles get a smaller size
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
    targets = list(plan[1:]) + [("Hỏi đáp", "Xin cảm ơn hội đồng; phần hỏi đáp.")]
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
    size = 19 if len(title) <= TITLE_ONE_LINE else 17 if len(title) <= TITLE_ONE_LINE + 8 else 15.5
    text(slide, 213 * PX, 1.33, 10.0, 0.5, title, size=size, bold=True, name="Title")

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
    s = frame(prs, "Thiết Lập Thực Nghiệm: Dữ Liệu, Mô Hình Và Phương Pháp Kiểm Định",
              [(SECTION, "section"), ("CHẠY CỤC BỘ TRÊN CPU · DỮ LIỆU CÁ NHÂN KHÔNG RỜI MÁY", "green")], page, total,
              *nav(page))
    kappa = list((syn.get("cohen_kappa") or {}).values())
    cards = [
        ("Gold Set tổng hợp", "blue",
         [f"{syn['n_jds']} mô tả công việc (JD) Automation Test × {syn['n_cvs']} CV đã ẩn danh",
          f"2 người gán nhãn mô phỏng, Cohen κ = {min(kappa):.2f}–{max(kappa):.2f}" if kappa else
          "Nhãn mức phù hợp 0–3 theo từng JD",
          "Đánh giá trên toàn bộ danh sách xếp hạng"]),
        ("Dữ liệu công khai", "violet",
         [f"resume-job-description-fit (Hugging Face): {pub['n_jds']} JD kiểm thử, {pub['n_cvs']} CV" if pub else
          "resume-job-description-fit (Hugging Face)",
          "Nhãn Fit / Potential / No Fit quy về 2 / 1 / 0",
          f"SkillSpan, tập test (CC-BY-4.0): {spans['n_gold_spans']} đoạn kỹ năng" if spans else
          "SkillSpan (CC-BY-4.0)"]),
        ("Mô hình (chạy cục bộ)", "green",
         ["Mã hoá dense: multilingual-e5-base", "Cross-encoder: bge-reranker-base", "Suy luận NLI: nli-deberta-v3-base",
          "LLM: Qwen2.5 3B / 7B (Ollama, CPU)"]),
        ("Kiểm định thống kê", "amber",
         ["Khoảng tin cậy 95% bằng bootstrap ghép cặp theo JD", "Kiểm định Wilcoxon và cỡ hiệu ứng r",
          "Giả thuyết được đặt trước khi chạy (slide 26)", "Lưu đệm kết quả LLM để tái lập được"]),
    ]
    cw, gap, x0, y0 = 2.33, 0.2, 213 * PX, 1.95
    for i, (title, kind, lines) in enumerate(cards):
        x = x0 + i * (cw + gap)
        box(s, x, y0, cw, 2.55, fill=WHITE, line=LINE, radius=0.05)
        chip(s, x + 0.15, y0 + 0.17, f"0{i + 1}", kind)
        text(s, x + 0.15, y0 + 0.52, cw - 0.3, 0.35, title, size=11.5, bold=True)
        text(s, x + 0.15, y0 + 0.95, cw - 0.3, 1.5, [[("• " + ln, {})] for ln in lines], size=9.5, spacing=4)
    note(s, x0, 4.7, 4.75, 1.35, "Các cấu hình so sánh:",
         "B0 BM25 · B1 dense · B2 lai ghép (RRF) · B3 thêm Criteria Profile · B4 quy trình cố định (thêm "
         "cross-encoder, bộ phân xử, kiểm chứng) · LLM-all (LLM xếp lại toàn bộ top 20) · P6 hệ thống agentic "
         "đầy đủ · A1 bỏ bước kiểm chứng · A2 prompt chung · P6-noclar bỏ bước hỏi lại HR.", "blue")
    note(s, x0 + 4.95, 4.7, 4.82, 1.35, "Lưu ý khi diễn giải:",
         f"với n = {syn['n_jds']} JD, kiểm định Wilcoxon không thể đạt p < {2 / 2 ** syn['n_jds']:.3f}, vì vậy "
         "kết luận dựa trên khoảng tin cậy 95% thay vì giá trị trung bình. Gold Set và Criteria Profile cùng do "
         "nhóm xây dựng nên mức cải thiện của B3 có thể bị đánh giá cao; dữ liệu công khai được dùng để kiểm "
         "tra lại.", "amber", size=9)


def slide_ladder(prs, syn, page, total):
    s = frame(prs, "Phân Tích Loại Bỏ Thành Phần B0 → P6: Criteria Profile Đóng Góp Lớn Nhất",
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
        ("green", "Criteria Profile (B3)", f"nDCG@10 tăng từ {fmt(b2)} (B2) lên {fmt(b3)}. Trọng số tiêu chí "
                                           f"theo từng công ty và tiêu chí loại trừ là nguồn cải thiện chính. Do "
                                           f"profile và nhãn cùng do nhóm xây dựng, mức tăng này có thể bị đánh "
                                           f"giá cao."),
        ("amber", "Dense chưa tinh chỉnh (B1)", f"nDCG@10 {fmt(b1)}, thấp hơn BM25 ({fmt(b0)}). CV kỹ thuật chứa "
                                                f"nhiều tên công cụ; mô hình e5 chưa tinh chỉnh làm mất tín hiệu "
                                                f"từ khoá này."),
        ("blue", "Hệ thống đầy đủ (P6)", f"nDCG@10 {fmt(p6)}, MRR@10 {fmt(mean(syn, 'P6', 'mrr@10'))}. LLM chỉ "
                                         f"phân xử các cặp ứng viên có điểm gần nhau nên P6 giữ được chất lượng "
                                         f"của B3/B4."),
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
        return verdict_cell(hyp[h].get("accepted"))

    def det(h, key=None):
        d = (hyp.get(h) or {}).get("details") or {}
        return d.get(key) if key else d

    rows = [["", "Giả thuyết (đặt trước khi chạy)", "Kết quả đo (a so với b, Δ, CI 95%, p)", "Kết luận"]]
    h1_parts = []
    for metric in ("recall@100", "mrr@10"):
        pairs = det("H1", metric)
        if isinstance(pairs, list) and pairs:
            h1_parts.append(f"{metric} " + "; ".join(f"{d['a']} vs {d['b']} {fmt(d['mean_a'])} vs {fmt(d['mean_b'])}"
                                                      for d in pairs))
    rows.append(["H1", "Truy hồi lai B2 không thấp hơn max(B0, B1) và cao hơn ít nhất một trong hai",
                 "\n".join(h1_parts) or "—", verdict("H1")])
    d = det("H2")
    rows.append(["H2", "Criteria Profile theo công ty: τ của P6 cao hơn A2 (prompt chung)",
                 "Kendall τ " + (cmp_text(d) if d else "—"), verdict("H2")])
    d = det("H3", "ndcg@10")
    c = det("H3", "llm_calls")
    rows.append(["H3", "Phân xử chọn lọc: P6 không kém LLM-all (cận dưới CI 95% của Δ nDCG@10 ≥ −0.02) và gọi "
                       "LLM ít hơn",
                 ("nDCG@10 " + cmp_text(d) if d else "—") + ("\nlượt gọi LLM đánh giá " + cmp_text(c, 1) if c else ""),
                 verdict("H3")])
    d = det("H4")
    rows.append(["H4", "Kiểm chứng bằng chứng: tỷ lệ nhận định thiếu bằng chứng của P6 thấp hơn A1",
                 "tỷ lệ thiếu bằng chứng " + (cmp_text(d) if d else "—"), verdict("H4")])
    h5 = hyp.get("H5") or {}
    fail = h5.get("failing_jds")
    rows.append(["H5", "Công bằng: impact ratio ≥ 0.8 (quy tắc 4/5) ở mọi JD",
                 f"impact ratio trung bình {fmt(h5.get('impact_ratio_mean'))}"
                 + (f"; JD dưới 0.8: {len(fail)}" if fail is not None else ""), verdict("H5")])
    d = det("H6")
    rows.append(["H6", "Hỏi lại HR làm tăng τ trên JD mơ hồ (P6 so với P6-noclar)",
                 "Kendall τ " + (cmp_text(d) if d else "—"), verdict("H6")])
    d, u = det("H0", "ndcg@10"), det("H0", "unsupported_rate")
    rows.append(["H0", "P6 tốt hơn B4 về nDCG@10 hoặc độ trung thực, chi phí trong ngân sách",
                 ("nDCG@10 " + cmp_text(d) if d else "—") + ("\ntỷ lệ thiếu bằng chứng " + cmp_text(u) if u else ""),
                 verdict("H0")])
    return rows


def verdict_cell(accepted: Optional[bool]) -> Tuple[str, Dict[str, Any]]:
    return ("ỦNG HỘ", {"color": GREEN, "bold": True}) if accepted else ("KHÔNG ỦNG HỘ", {"color": RED, "bold": True})


def slide_hypotheses(prs, syn, page, total):
    s = frame(prs, "Kiểm Định Giả Thuyết H0–H6: Báo Cáo Cả Giả Thuyết Không Được Ủng Hộ",
              [(SECTION, "section"), ("BOOTSTRAP GHÉP CẶP, CI 95% · WILCOXON", "amber")], page, total, *nav(page))
    rows = _h_rows(syn)
    table(s, 213 * PX, 1.8, 1173 * PX, rows, [0.45, 3.4, 4.6, 1.2], size=8.5, row_h=0.47)
    text(s, 213 * PX, 1.8 + 0.47 * len(rows) + 0.12, 1173 * PX, 0.3,
         f"Quy tắc quyết định: giả thuyết được ủng hộ khi toàn bộ khoảng tin cậy 95% nằm đúng phía (với giả thuyết "
         f"\"không kém hơn\": cận dưới ≥ −ε), không dựa vào giá trị trung bình. Với n = {syn['n_jds']} JD, giá trị p "
         f"nhỏ nhất của kiểm định Wilcoxon là {2 / 2 ** syn['n_jds']:.3f}.", size=8.5, color=MUTED)


def slide_prior(prs, syn, pub, ctx, page, total):
    s = frame(prs, "Đối Chiếu Với Nghiên Cứu Liên Quan: Điểm Tương Đồng Và Khác Biệt",
              [(SECTION, "section"), ("SO SÁNH XU HƯỚNG, KHÔNG SO SÁNH GIÁ TRỊ TUYỆT ĐỐI", "amber")], page, total,
              *nav(page))
    b0, b1, b3 = mean(syn, "B0", "ndcg@10"), mean(syn, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    b4, p6, la = mean(syn, "B4", "ndcg@10"), mean(syn, "P6", "ndcg@10"), mean(syn, "LLM-all", "ndcg@10")
    calls_p6, calls_la = mean(syn, "P6", "judge_calls"), mean(syn, "LLM-all", "judge_calls")
    un_p6, un_a1 = mean(syn, "P6", "unsupported_rate"), mean(syn, "A1", "unsupported_rate")
    ir = (syn.get("hypotheses") or {}).get("H5", {}).get("impact_ratio_mean")
    saving = 1 - calls_p6 / calls_la if calls_la and not math.isnan(calls_la) and calls_la > 0 else float("nan")
    g, r = {"color": GREEN, "bold": True}, {"color": RED, "bold": True}
    h3_ok = (syn.get("hypotheses") or {}).get("H3", {}).get("accepted")
    rows = [
        ["Công trình (slide)", "Kết quả công bố", "Kết quả tương ứng của nhóm", "Nhận xét"],
        ["ConFit v2 · Yu et al. 2025 (16)", "AliYun: BM25 40.56 > E5 chưa tinh chỉnh 29.66 nDCG@10; tập "
                                            "Recruiting cho kết quả ngược lại",
         (f"BM25 {fmt(b0)} > e5 chưa tinh chỉnh {fmt(b1)}", {}),
         ("Tương đồng: hiệu quả của BM25 phụ thuộc lĩnh vực, ủng hộ truy hồi lai", g)],
        confit_row(syn, pub, ctx, g, r),
        ["RankGPT · Sun et al. 2023 (17)", "LLM xếp hạng lại theo danh sách: +2.7 nDCG@10 (TREC)",
         f"LLM-all {fmt(la)} so với B4 {fmt(b4)} (Qwen 3B)",
         ("LLM nhỏ xếp lại toàn bộ không cải thiện kết quả", r) if la <= b4 + 0.005 else
         (f"Cùng chiều nhưng nhỏ (+{la - b4:.3f}), chưa có ý nghĩa thống kê với n = {syn['n_jds']}",
          {"color": AMBER, "bold": True})],
        ["FrugalGPT · Chen et al. 2023 (17)", "Chuỗi mô hình giảm tới 98% chi phí, chất lượng tương đương GPT-4",
         f"P6 {fmt(calls_p6, 1)} so với LLM-all {fmt(calls_la, 1)} lượt gọi LLM đánh giá mỗi JD (ít hơn "
         f"{pct(saving, 0)}); nDCG@10 {fmt(p6)} so với {fmt(la)}",
         ("Tương đồng: ít lượt gọi hơn, không kém LLM-all (CI 95%)", g) if h3_ok else
         ("Tương đồng một phần: ít lượt gọi hơn rõ rệt, nhưng CI 95% chưa loại trừ mức kém hơn ε",
          {"color": AMBER, "bold": True})],
        ["Gan et al. 2024; Lo et al. 2025 (18)", "Agent: F1 87.73%, nhanh hơn 11 lần; không đo nDCG, không kiểm "
                                                 "chứng bằng chứng",
         f"Đánh giá bằng chỉ số xếp hạng và độ trung thực: tỷ lệ nhận định thiếu bằng chứng P6 {pct(un_p6)} so "
         f"với A1 {pct(un_a1)}",
         "Bổ sung phương diện đánh giá còn thiếu đã nêu ở slide 18"],
        ["Wilson & Caliskan 2024 (18)", "Truy hồi bằng embedding ưu tiên tên người da trắng 85.1%, tên nữ 11.1%",
         f"Che thông tin định danh trước khi mã hoá{audit_text(ctx.get('audit'))}; impact ratio trung bình {fmt(ir)}",
         ("Đạt quy tắc 4/5", g) if ir and ir >= 0.8 else ("Chưa đạt quy tắc 4/5 ở mọi JD, cần kiểm tra thêm", r)],
    ]
    tagger = ctx.get("tagger")
    if tagger:
        tf1 = tagger["test"]["tagger"]["f1_exact"]
        rows.append(["SkillSpan · Zhang et al. 2022", f"JobBERT, span F1 {SKILLSPAN_PAPER_F1 * 100:.1f} (tập test, "
                                                      "trung bình 5 seed)",
                     f"Mô hình gán nhãn JobBERT: span F1 {fmt(tf1)} (1 seed)" +
                     (f"; từ điển {fmt(ctx['spans']['f1_exact'])}" if ctx.get("spans") else ""),
                     ("Tái lập được kết quả của bài báo (1 seed so với trung bình 5 seed)", g)
                     if abs(tf1 - SKILLSPAN_PAPER_F1) <= 0.03 else
                     (f"Chênh {tf1 - SKILLSPAN_PAPER_F1:+.3f} so với bài báo", {"color": AMBER, "bold": True})])
    if pub:
        pb0, pp6 = mean(pub, "B0", "ndcg@10"), mean(pub, "P6", "ndcg@10")
        rows.append(["resume-job-description-fit (Hugging Face)", "Nhãn Fit / Potential / No Fit, chưa có kết "
                     "quả xếp hạng cơ sở được công bố", f"nDCG@10: BM25 {fmt(pb0)}, P6 {fmt(pp6)} trên {pub['n_jds']} JD",
                     ("Kết quả lặp lại ngoài dữ liệu tổng hợp", g) if pp6 >= pb0 else
                     (f"Không lặp lại: P6 thấp hơn BM25; cao nhất là {best_config(pub)}", r)])
    table(s, 213 * PX, 1.8, 1173 * PX, rows, [2.2, 2.9, 3.0, 2.3], size=8.5, row_h=0.5 if len(rows) <= 8 else 0.44)


def confit_row(syn, pub, ctx, g, r) -> List[Any]:
    b0, b1, b3 = mean(syn, "B0", "ndcg@10"), mean(syn, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    syn_ft = ctx.get("syn_ft")
    if not syn_ft:
        return ["ConFit v2 (16)", "Tinh chỉnh bi-encoder: +41.6 nDCG@10 so với BM25 (82.20)",
                f"Không tinh chỉnh; Criteria Profile: {fmt(b0)} → {fmt(b3)} (+{b3 - b0:.3f})",
                "Cải thiện đến từ tri thức về tiêu chí thay vì dữ liệu gán nhãn lớn"]
    b1_ft = mean(syn_ft, "B1", "ndcg@10")
    pub_ft = ctx.get("pub_ft")
    pub_txt, pub_beats = "", False
    if pub and pub_ft:
        pb0, pb1, pb1_ft = (mean(pub, "B0", "ndcg@10"), mean(pub, "B1", "ndcg@10"), mean(pub_ft, "B1", "ndcg@10"))
        pub_txt, pub_beats = f"; dữ liệu công khai {fmt(pb1)} → {fmt(pb1_ft)} (BM25 {fmt(pb0)})", pb1_ft > pb0
    # ConFit's claim is "fine-tuned dense beats BM25", so the verdict compares B1-ft with B0, not only with B1
    if b1_ft > b0:
        verdict = ("Tương đồng: dense sau tinh chỉnh vượt BM25", g)
    elif pub_beats:
        verdict = ("Tương đồng một phần: vượt BM25 trên dữ liệu công khai, chưa vượt trên Gold Set",
                   {"color": AMBER, "bold": True})
    elif b1_ft > b1 + 0.005:
        verdict = (f"Tương đồng một phần: tinh chỉnh cải thiện dense (+{b1_ft - b1:.3f}) nhưng chưa vượt BM25",
                   {"color": AMBER, "bold": True})
    else:
        verdict = ("Không lặp lại: tinh chỉnh trên JD khác lĩnh vực không cải thiện", r)
    return ["ConFit v2 (16)", "Tinh chỉnh bi-encoder: +41.6 nDCG@10 so với BM25 (82.20)",
            f"e5 tinh chỉnh trên cặp không thuộc kiểm thử: B1 trên Gold Set {fmt(b1)} → {fmt(b1_ft)} (BM25 {fmt(b0)})"
            f"{pub_txt}; Criteria Profile {fmt(b3)}", verdict]


def audit_text(audit: Optional[List[Dict[str, Any]]]) -> str:
    if not audit:
        return ""
    worst = max(a["max_abs_delta_rank"] for a in audit)
    return f"; hoán đổi tên: Δ thứ hạng tối đa {worst} trên {len(audit)} JD"


def best_config(summ, metric="ndcg@10"):
    name = max(summ["means"], key=lambda c: mean(summ, c, metric) or 0)
    return f"{name} {fmt(mean(summ, name, metric))}"


def slide_faith(prs, syn, page, total):
    s = frame(prs, "Độ Trung Thực Và Chi Phí: Vai Trò Của Kiểm Chứng Và Phân Xử Chọn Lọc",
              [(SECTION, "section"), ("H0 · H3 · H4", "green")], page, total, *nav(page))
    cfgs = [c for c in ("B4", "LLM-all", "P6", "A1") if c in syn["means"]]
    bar_chart(s, 213 * PX, 1.9, 4.75, 4.25, cfgs, {"unsupported": [mean(syn, c, "unsupported_rate") for c in cfgs]},
              [RGBColor(0xF4, 0x3F, 0x5E)], "Tỷ lệ nhận định thiếu bằng chứng", number_format="0%",
              highlight=cfgs.index("P6") if "P6" in cfgs else None)
    bar_chart(s, 213 * PX + 4.9, 1.9, 4.85, 4.25, cfgs,
              {"Tổng lượt gọi LLM": [mean(syn, c, "llm_calls") for c in cfgs],
               "Lượt gọi LLM đánh giá": [mean(syn, c, "judge_calls") for c in cfgs]},
              [GREY_BAR, BLUE], "Số lượt gọi LLM trung bình mỗi JD", number_format="0.0", legend=True)


def slide_agent(prs, h7: Dict[str, Dict[str, Any]], page, total):
    s = frame(prs, "Agentic AI (H7): Ổn Định Hơn Ở 7B Nhưng Tốn Kém Hơn Luật",
              [(SECTION, "section"), ("H7 · AG SO VỚI AG-RULES", "violet")], page, total, *nav(page))
    rows = [["Lượt chạy", "Cấu hình", "nDCG@10", "Thiếu bằng chứng", "Lượt gọi LLM/JD", "Số bước",
             "Luật thay quyết định LLM", "Đạt mục tiêu", "H7"]]
    for label, summ in h7.items():
        hyp = (summ.get("hypotheses") or {}).get("H7", {})
        for cfg in ("AG-rules", "AG"):
            if cfg not in summ["means"]:
                continue
            m = summ["means"][cfg]
            goal = m.get("goal_met")
            fb = m.get("planner_fallback_rate") if cfg == "AG" else None
            verdict = verdict_cell(hyp.get("accepted")) if cfg == "AG" else ""
            rows.append([label if cfg == "AG-rules" else "", cfg, fmt(m.get("ndcg@10")), pct(m.get("unsupported_rate")),
                         fmt(m.get("llm_calls"), 1), fmt(m.get("agent_steps"), 1), pct(fb, 0) if fb is not None
                         else "—", f"{round(goal * summ['n_jds'])}/{summ['n_jds']}" if goal is not None else "—",
                         verdict])
    table(s, 213 * PX, 1.85, 1173 * PX, rows, [1.6, 0.9, 0.8, 1.15, 1.0, 0.75, 1.3, 0.95, 1.45], size=9,
          row_h=0.36)
    y = 2.05 + 0.36 * len(rows)
    note(s, 213 * PX, y, 4.8, min(1.5, 6.15 - y), "Kết quả đã kiểm chứng:",
         "bộ lập kế hoạch LLM chỉ chọn trong tập công cụ có ràng buộc an toàn; điểm xác nhận của HR, tiêu chí loại "
         "trừ và ngân sách không bị vi phạm. Với mô hình 7B, luật hiếm khi phải thay quyết định của LLM.", "green")
    note(s, 213 * PX + 4.97, y, 4.8, min(1.5, 6.15 - y), "Hạn chế:",
         "trên dữ liệu tổng hợp, quyết định tối ưu đã được mã hoá sẵn trong luật, nên AG tốn thêm lượt gọi LLM mà "
         "không tăng nDCG. Cần Gold Set thực tế để đánh giá lợi ích của việc tự lập kế hoạch.", "amber")


SKILLSPAN_PAPER_F1 = 0.597  # Zhang et al. 2022, Table 5: JobBERT, COMBINED, TEST, mean of 5 seeds


def _span_rows(spans, tagger) -> List[List[Any]]:
    cols = [("Từ điển kỹ năng", spans)]
    if tagger:
        cols += [(label, tagger["test"][key]) for key, label in (("tagger", "JobBERT gán nhãn"),
                                                                ("hybrid", "Hợp hai phương pháp"))
                 if key in tagger["test"]]
    rows = [["SkillSpan (test)"] + [c for c, _ in cols]]
    for key, label in (("precision_exact", "Precision (khớp đúng)"), ("recall_exact", "Recall (khớp đúng)"),
                       ("f1_exact", "F1 (khớp đúng)"), ("f1_overlap", "F1 (khớp chồng lấn)")):
        best = max((r[key] for _, r in cols if not math.isnan(r[key])), default=float("nan"))
        win = len(cols) > 1
        rows.append([label] + [(fmt(r[key]), {"bold": win and r[key] == best,
                                              "color": GREEN if win and r[key] == best else INK})
                               for _, r in cols])
    return rows


def slide_public(prs, pub, spans, page, total, ctx=None):
    s = frame(prs, "Kiểm Tra Trên Dữ Liệu Công Khai Và Trích Xuất Kỹ Năng",
              [(SECTION, "section"), ("RESUME-JOB-DESCRIPTION-FIT · SKILLSPAN", "violet")], page, total, *nav(page))
    x0 = 213 * PX
    if pub:
        order = [c for c in ("B0", "B1", "B2", "B3", "B4", "P6") if c in pub["means"]]
        bar_chart(s, x0, 1.9, 5.3, 2.95, order, {"nDCG@10": [mean(pub, c, "ndcg@10") for c in order]}, [GREY_BAR],
                  f"Dữ liệu công khai: nDCG@10 ({pub['n_jds']} JD, {pub['n_cvs']} CV)", max_val=1.1,
                  highlight=order.index("P6") if "P6" in order else None)
        note(s, x0, 4.95, 5.3, 1.2, f"Kết quả trung bình {pub['n_jds']} JD không lặp lại:",
             f"{best_config(pub)} cao nhất; P6 ({fmt(mean(pub, 'P6', 'ndcg@10'))}) thấp hơn BM25 "
             f"({fmt(mean(pub, 'B0', 'ndcg@10'))})." + (" Slide tiếp theo phân tích theo từng JD để xác định "
                                                        "nguyên nhân." if (ctx or {}).get("errors") else ""),
             "amber", size=9)
    tagger = (ctx or {}).get("tagger")
    if spans:
        rows = _span_rows(spans, tagger)
        table(s, x0 + 5.5, 1.9, 4.28, rows, [1.45] + [0.94] * (len(rows[0]) - 1), size=9, row_h=0.32)
        if tagger:
            t = tagger["test"]["tagger"]
            body = (f"từ điển chỉ bao phủ thuật ngữ Automation Test nên recall chỉ đạt {fmt(spans['recall_exact'])}. "
                    f"Mô hình gán nhãn JobBERT, huấn luyện trên tập train của SkillSpan ({tagger['data']['train']} "
                    f"câu, chọn epoch theo tập dev), đạt span F1 {fmt(t['f1_exact'])}, tương đương mức "
                    f"{SKILLSPAN_PAPER_F1:.3f} mà bài báo gốc công bố (trung bình 5 seed). Hệ thống dùng từ điển để "
                    "gắn mã khái niệm làm bằng chứng và dùng mô hình gán nhãn để phát hiện kỹ năng ngoài từ điển.")
        else:
            body = (f"bộ trích xuất dựa trên từ điển nên precision cao ({pct(spans['precision_exact'], 0)}) nhưng "
                    "recall thấp: SkillSpan gán nhãn cả kỹ năng mềm và các đoạn dài nằm ngoài phạm vi từ điển "
                    "Automation Test.")
        note(s, x0 + 5.5, 3.65, 4.28, 2.5, "Diễn giải:", body, "violet", size=9)


DUPLICATE_JACCARD = 0.8  # same threshold as cv_agent.eval.error_analysis


def _jd_label(jd: str, dup_of: Dict[str, int]) -> str:
    return f"…{jd[-6:]}" + (f"  (trùng lặp, nhóm {dup_of[jd]})" if jd in dup_of else "")


def slide_errors(prs, pub, ctx, page, total):
    ea = ctx["errors"]
    dups = ea["near_duplicates"]
    dup_of = {jd: i + 1 for i, grp in enumerate(dups) for jd in grp}
    n_dup = sum(len(grp) for grp in dups)
    d, g = ea["dedup"], ea["groups"]
    dup_signals = sorted({sig for grp in dups for jd in grp for sig in ea["per_jd"][jd]["qa_signal_list"]})
    what = "Một Tin Tuyển Dụng Trùng Lặp" if len(dups) == 1 else f"{len(dups)} Tin Tuyển Dụng Trùng Lặp"
    s = frame(prs, f"Phân Tích Lỗi: {n_dup}/{pub['n_jds']} JD Công Khai Là {what}",
              [(SECTION, "section"), ("PHÂN TÍCH HẬU KIỂM · KHÔNG PHẢI GIẢ THUYẾT", "amber")], page, total,
              *nav(page))
    x0 = 213 * PX
    cfgs = [c for c in ("B0", "B1", "B3", "P6") if c in d]
    rows = [["JD công khai", "Số từ khoá kiểm thử"] + cfgs + ["P6 thiếu bằng chứng"]]
    for jd, info in sorted(ea["per_jd"].items(), key=lambda kv: (kv[0] in dup_of, kv[0])):
        vals = info["ndcg@10"]
        best = max(vals.get(c, 0) for c in cfgs)
        rows.append([_jd_label(jd, dup_of), str(info["qa_signals"])] +
                    [(fmt(vals.get(c)), {"bold": vals.get(c) == best, "color": GREEN if vals.get(c) == best else INK})
                     for c in cfgs] + [pct(info["unsupported_rate"].get("P6"), 0)])
    text(s, x0, 1.82, 5.4, 0.3, "nDCG@10 theo từng JD (mô hình chưa tinh chỉnh)", size=10, bold=True)
    table(s, x0, 2.12, 5.4, rows, [1.75, 0.65] + [0.55] * len(cfgs) + [0.8], size=8.5, row_h=0.31)
    y = 2.3 + 0.31 * len(rows)
    quoted = ", ".join(f"'{sig}'" for sig in dup_signals)
    note(s, x0, y, 5.4, 6.12 - y, "Nguyên nhân:",
         f"{n_dup} JD gần như trùng nhau (độ tương đồng Jaccard trên 3-gram ≥ {DUPLICATE_JACCARD}) và chỉ qua được "
         f"bộ lọc nhờ các cụm {quoted}. Nội dung thực tế là tuyển lập trình viên hạ tầng, không phải Automation "
         f"Test, nên Criteria Profile áp dụng tiêu chí không phù hợp: B3 đạt {fmt(g['distinct']['B3'])} trên các JD "
         f"khác nhau nhưng chỉ {fmt(g['duplicate_postings']['B3'])} trên nhóm trùng lặp, và nhóm này được tính "
         f"{n_dup} lần trong giá trị trung bình.", "amber", size=9)
    groups = [("JD khác nhau", g["distinct"]), (f"Nhóm trùng lặp ({n_dup})", g["duplicate_postings"]),
              (f"Sau khử trùng (n = {ea['effective_n_jds']})", d)]
    bar_chart(s, x0 + 5.6, 1.85, 4.18, 2.75, [lab for lab, _ in groups],
              {c: [m[c] for _, m in groups] for c in cfgs}, [GREY_BAR, BLUE, GREEN, VIOLET][:len(cfgs)],
              "nDCG@10 theo nhóm JD", number_format="0.00", max_val=1.1, legend=True)
    best_d, worst_d = max(cfgs, key=lambda c: d[c]), min(cfgs, key=lambda c: d[c])
    note(s, x0 + 5.6, 4.72, 4.18, 1.4, "Lưu ý:",
         f"khi mỗi nhóm trùng lặp chỉ tính một lần, {best_d} cao nhất ({fmt(d[best_d])}) và {worst_d} thấp nhất "
         f"({fmt(d[worst_d])}). Phân tích được thực hiện sau khi có kết quả nên chỉ mang tính gợi ý; tập đánh giá "
         "được giữ nguyên. Bước tiếp theo là lọc JD theo tiêu đề và khử trùng lặp trước khi chạy lại.", "blue",
         size=9)


def slide_conclusion(prs, syn, pub, ctx, page, total):
    s = frame(prs, "Kết Luận Thực Nghiệm: Kết Quả Vững Chắc, Hạn Chế Và Hướng Phát Triển",
              [(SECTION, "section"), ("CÁC YẾU TỐ ẢNH HƯỞNG ĐẾN TÍNH HỢP LỆ", "amber")], page, total, *nav(page))
    p6, b0 = mean(syn, "P6", "ndcg@10"), mean(syn, "B0", "ndcg@10")
    stats = [(fmt(p6), "nDCG@10 của P6 trên Gold Set"), (f"n = {syn['n_jds']}", "JD trong mỗi lượt đánh giá"),
             ("~200", "CV thực tế cần gán nhãn")]
    rejected = [h for h in ("H1", "H2", "H3", "H4", "H5", "H6") if not syn["hypotheses"].get(h, {}).get("accepted")]
    cols = [
        ("green", "Kết quả vững chắc", [
            f"Gold Set: nDCG@10 tăng từ {fmt(b0)} (BM25) lên {fmt(p6)} (P6), chủ yếu nhờ Criteria Profile",
            "Kiểm chứng bằng chứng giảm tỷ lệ nhận định thiếu bằng chứng (H4, H0)",
            "Phân xử chọn lọc cần ít lượt gọi LLM hơn LLM-all (CI 95% < 0)",
            "Agent tuân thủ ràng buộc an toàn, không bỏ qua bước xác nhận của HR",
            *([f"Mô hình gán nhãn JobBERT: span F1 {fmt(ctx['tagger']['test']['tagger']['f1_exact'])} trên "
               f"SkillSpan (bài báo: {SKILLSPAN_PAPER_F1})"]
              if (ctx.get("tagger") or {}).get("test", {}).get("tagger") else []),
            "Hoán đổi tên ứng viên không làm thay đổi thứ hạng (Δ = 0)" if worst_swap(ctx) == 0 else
            "Thông tin định danh được che trước mọi bước mã hoá và gọi LLM"]),
        ("amber", "Hạn chế", [
            (f"Dữ liệu công khai: P6 thấp hơn BM25 ({sum(map(len, ctx['errors']['near_duplicates']))}/"
             f"{pub['n_jds']} JD là tin trùng lặp)"
             if ctx.get("errors") and ctx["errors"]["near_duplicates"] else
             "Dữ liệu công khai: P6 không vượt BM25") if pub and mean(pub, "P6", "ndcg@10") <
            mean(pub, "B0", "ndcg@10") else "Dense chưa tinh chỉnh kém BM25 trên Gold Set",
            (", ".join(rejected) + " không được ủng hộ") if rejected else
            "Mọi giả thuyết H1–H6 được ủng hộ",
            "LLM lập kế hoạch tốn kém hơn luật",
            finetune_bullet(syn, ctx),
            "Profile và nhãn cùng do nhóm xây dựng: B3 có thể được đánh giá cao",
            f"n = {syn['n_jds']} JD: khoảng tin cậy rộng, p ≥ {2 / 2 ** syn['n_jds']:.3f}"]),
        ("blue", "Hướng phát triển", [
            "Xây dựng Gold Set thực tế (~200 CV), hai chuyên gia QA gán nhãn độc lập, đo Cohen κ",
            "Tinh chỉnh trên Gold Set thực tế bằng notebook Colab đã có" if ctx.get("ft") else
            "Tinh chỉnh bi-encoder (theo hướng ConFit) khi đủ dữ liệu gán nhãn",
            "Lọc JD theo tiêu đề, khử trùng lặp; kiểm tra tính công bằng trên dữ liệu thực tế",
            "Dùng GPU hoặc LLM từ 7B trở lên để giảm độ trễ"]),
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
        return "Dense chưa tinh chỉnh kém BM25"
    b1, b1_ft, b3 = mean(syn, "B1", "ndcg@10"), mean(syn_ft, "B1", "ndcg@10"), mean(syn, "B3", "ndcg@10")
    tail = "vẫn thấp hơn B3" if b1_ft < b3 else "ngang hoặc cao hơn B3"
    return f"Tinh chỉnh dense: B1 {fmt(b1)} → {fmt(b1_ft)}, {tail}"


def _ft_table_rows(ft: Dict[str, Dict[str, Any]], seeds: Optional[Dict[str, Any]] = None) -> List[List[Any]]:
    """Before → after per split; with a multi-seed summary, after is the mean ± sd over seeds."""
    rows = [["Mô hình", "Mẫu huấn luyện", "nDCG@10 (dev)", "nDCG@10 (test)"]]
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
        unit = "bộ ba" if target == "dense" else "cặp"
        rows.append([names.get(target, target), f"{rep['n_train_examples']} {unit}", *cells])
    return rows


def _seed_reading(seeds: Dict[str, Any]) -> str:
    """Which gains stand out from the seed-to-seed spread (gain > 2 sd) and which do not."""
    names = {"dense": "dense", "rerank": "cross-encoder"}
    weak = [f"{names.get(t, t)} trên tập {split} (+{fmt(m['gain_mean'])} ± {m['gain_sd']:.3f})"
            for t, splits in seeds["summary"].items() for split, m in splits.items()
            if not m["gain_mean"] > 2 * m["gain_sd"]]
    return ("Mọi mức cải thiện đều lớn hơn 2 lần độ lệch chuẩn." if not weak else
            "Mức cải thiện lớn hơn 2 lần độ lệch chuẩn, trừ " + ", ".join(weak) +
            ": chưa phân biệt được với dao động giữa các seed.")


def slide_finetune(prs, syn, pub, ctx, page, total):
    ft, syn_ft, pub_ft = ctx["ft"], ctx.get("syn_ft"), ctx.get("pub_ft")
    b1 = mean(syn, "B1", "ndcg@10")
    b1_ft = mean(syn_ft, "B1", "ndcg@10") if syn_ft else float("nan")
    title = (f"Tinh Chỉnh Theo ConFit: B1 Tăng Từ {fmt(b1)} Lên {fmt(b1_ft)} Trên Gold Set" if syn_ft
             else "Tinh Chỉnh Theo ConFit: Bộ Mã Hoá Dense Và Cross-encoder")
    s = frame(prs, title,
              [(SECTION, "section"), ("COLAB T4 · notebooks/finetune_colab.ipynb", "violet")], page, total, *nav(page))
    x0 = 213 * PX
    text(s, x0, 1.88, 4.6, 0.3, "Trên dữ liệu công khai ngoài lĩnh vực kiểm thử (trước → sau)", size=10,
         bold=True)
    seeds = ctx.get("ft_seeds")
    table(s, x0, 2.18, 4.6, _ft_table_rows(ft, seeds), [1.3, 0.8, 1.25, 1.25], size=9, row_h=0.34)
    any_rep = next(iter(ft.values()))
    d = any_rep["data"]
    note(s, x0, 3.4, 4.6, 1.25, "Chống rò rỉ dữ liệu:",
         f"loại mọi JD kiểm thử (chính là các JD dùng để đánh giá) và mọi CV gắn với chúng. Train {d['train']['jds']} "
         f"JD, dev {d['dev']['jds']} JD, chia theo JD; test gồm {d['test']['jds']} JD chưa xuất hiện khi huấn luyện. "
         "Dữ liệu tải trực tiếp từ Hugging Face; không CV nào của nhóm rời khỏi máy.", "violet", size=9)
    note(s, x0, 4.8, 4.6, 1.35, "Phương pháp:",
         f"e5 học từ bộ ba (JD, CV phù hợp, CV không phù hợp) với hàm mất mát tương phản; cross-encoder học nhãn "
         f"0 / 0.5 / 1 bằng entropy chéo nhị phân; {any_rep['config']['epochs']} epoch, độ dài tối đa "
         f"{any_rep['config']['max_len']} token, " +
         (f"{len(seeds['seeds'])} seed ({', '.join(map(str, seeds['seeds']))}) trên GPU Colab T4. Bảng báo cáo "
          f"trung bình ± độ lệch chuẩn. {_seed_reading(seeds)} Biểu đồ dùng mô hình seed {any_rep['config']['seed']}."
          if seeds else
          f"1 seed ({any_rep['config']['seed']}), chưa có độ lệch chuẩn."), "blue", size=9)
    cfgs = [c for c in ("B1", "B2", "B3", "P6") if syn_ft and c in syn_ft["means"]]
    if cfgs:
        bar_chart(s, x0 + 4.8, 1.88, 4.98, 2.15, cfgs,
                  {"Chưa tinh chỉnh": [mean(syn, c, "ndcg@10") for c in cfgs],
                   "Đã tinh chỉnh": [mean(syn_ft, c, "ndcg@10") for c in cfgs]},
                  [GREY_BAR, VIOLET], "Gold Set tổng hợp: nDCG@10", max_val=1.1, legend=True)
    pcfgs = [c for c in ("B1", "B2", "B3", "P6") if pub and pub_ft and c in pub_ft["means"]]
    if pcfgs:
        bar_chart(s, x0 + 4.8, 4.05, 4.98, 2.1, pcfgs,
                  {"Chưa tinh chỉnh": [mean(pub, c, "ndcg@10") for c in pcfgs],
                   "Đã tinh chỉnh": [mean(pub_ft, c, "ndcg@10") for c in pcfgs]},
                  [GREY_BAR, VIOLET], "Dữ liệu công khai (JD kiểm thử): nDCG@10", max_val=1.1, legend=True)


def slide_nfr(prs, syn, ctx, page, total):
    audit, bench = ctx.get("audit") or [], ctx.get("bench")
    worst = worst_swap(ctx)
    swap_txt = "Thứ Hạng Không Đổi Khi Đổi Tên" if worst == 0 else "Hoán Đổi Tên Và Quy Tắc 4/5"
    p50 = min((m["summary"]["wall_p50_s"] for m in bench["models"].values()), default=None) if bench else None
    lat_txt = f", {p50 / 60:.0f} Phút Mỗi JD Trên CPU" if p50 else ""
    s = frame(prs, f"Công Bằng Và Độ Trễ: {swap_txt}{lat_txt}",
              [(SECTION, "section"), ("NFR-04 · NFR-06 · H5", "green")], page, total, *nav(page))
    x0 = 213 * PX
    ir = (syn.get("hypotheses") or {}).get("H5", {}).get("impact_ratio_by_jd") or {}
    swap = {a["jd"]: a for a in audit}
    rows = [["JD (Gold Set)", "Impact ratio (≥ 0.8)", "Δ thứ hạng khi đổi tên (TB / tối đa)"]]
    for jd in ir or swap:
        r = ir.get(jd)
        a = swap.get(jd)
        rows.append([jd, (fmt(r, 2), {"color": GREEN if r is not None and r >= 0.8 else RED, "bold": True}),
                     f"{a['mean_abs_delta_rank']:.1f} / {a['max_abs_delta_rank']}" if a else "—"])
    text(s, x0, 1.88, 4.9, 0.3, "Công bằng: quy tắc 4/5 và hoán đổi tên (B3, 200 CV mỗi JD)", size=10, bold=True)
    table(s, x0, 2.18, 4.9, rows, [2.1, 1.35, 1.45], size=9, row_h=0.3)
    y = 2.35 + 0.3 * len(rows)
    failing = [jd for jd, v in ir.items() if v < 0.8]
    swap_msg = ("hoán đổi tên nam/nữ, tên Việt/tên nước ngoài không làm thay đổi thứ hạng vì tên được che trước "
                "mọi bước mã hoá và gọi LLM. " if worst == 0 else
                f"hoán đổi tên làm thứ hạng lệch tối đa {worst} bậc: cần xác định chỗ tên chưa được che. "
                if audit else "")
    ir_msg = (f"Impact ratio < 0.8 ở {len(failing)}/{len(ir)} JD. Vì tên đã được che, chênh lệch đến từ thuộc tính "
              "khác có tương quan với nhóm (ví dụ số năm kinh nghiệm); cần kiểm tra lại trên CV thực tế." if failing
              else "Impact ratio ≥ 0.8 ở mọi JD.")
    note(s, x0, y, 4.9, min(1.45, 6.1 - y), "Diễn giải:", swap_msg + ir_msg, "amber" if failing else "green",
         size=9)
    if bench:
        brow = [["Mô hình (P6, CPU)", "p50 / JD", "p95 / JD", "Lượt gọi LLM", "Dự phòng"]]
        for model, rep in bench["models"].items():
            sm = rep["summary"]
            brow.append([model, f"{sm['wall_p50_s']:.0f}s", f"{sm['wall_p95_s']:.0f}s", fmt(sm["calls_per_jd"], 1),
                         str(sm["fallbacks"])])
        text(s, x0 + 5.1, 1.88, 4.68, 0.3, "Độ trễ của P6: tắt lưu đệm LLM, Ollama trên CPU, không GPU", size=10,
             bold=True)
        table(s, x0 + 5.1, 2.18, 4.68, brow, [1.4, 0.8, 0.8, 0.85, 0.8], size=9, row_h=0.32)
        fast = min(bench["models"].items(), key=lambda kv: kv[1]["summary"]["wall_p50_s"])
        y = 2.35 + 0.32 * len(brow)
        fsum = fast[1]["summary"]
        ok = fsum["wall_p95_s"] <= NFR04_MAX_S and fsum["fallbacks"] == 0
        fb = ("không lần nào phải dùng phương án dự phòng" if fsum["fallbacks"] == 0 else
              f"{fsum['fallbacks']} lần dùng phương án dự phòng")
        note(s, x0 + 5.1, y, 4.68, 1.45, "Đạt yêu cầu NFR-04:" if ok else "Chưa đạt yêu cầu NFR-04:",
             f"{fast[0]} xếp hạng một JD trong khoảng {fsum['wall_p50_s'] / 60:.0f} phút (p50), p95 khoảng "
             f"{fsum['wall_p95_s'] / 60:.0f} phút trên laptop không có GPU, {fb}. Bộ phân xử chỉ xét các cặp có "
             "thể làm thay đổi top 10 và prompt chỉ chứa các dòng bằng chứng liên quan.",
             "green" if ok else "amber", size=9)


SPEAKER_NOTES = {
    "setup": "Giới thiệu dữ liệu, mô hình và phương pháp kiểm định. Toàn bộ thực nghiệm chạy cục bộ, dữ liệu cá nhân "
             "không rời máy. Giả thuyết được đặt trước khi chạy. Với 6 JD, kết luận dựa trên khoảng tin cậy 95%, "
             "không dựa riêng vào giá trị p.",
    "ladder": "Đóng góp lớn nhất đến từ Criteria Profile (B2 → B3). Dense chưa tinh chỉnh kém BM25 trên CV kỹ thuật. "
              "LLM và bộ phân xử chỉ điều chỉnh các vị trí đầu danh sách, nên P6 giữ được chất lượng của B3/B4.",
    "hypotheses": "Một giả thuyết được ủng hộ khi toàn bộ khoảng tin cậy 95% nằm đúng phía; với 6 JD, giá trị p "
                  "Wilcoxon nhỏ nhất là 0.031. Trình bày lần lượt: giả thuyết, kết quả đo, khoảng tin cậy và kết "
                  "luận. Các giả thuyết không được ủng hộ vẫn được báo cáo đầy đủ theo kế hoạch thực nghiệm.",
    "prior": "Do khác dữ liệu, nhóm so sánh xu hướng thay vì giá trị tuyệt đối. Cột cuối cho biết kết quả tương đồng "
             "hay không lặp lại được.",
    "faith": "Kiểm chứng bằng chứng là đóng góp rõ rệt nhất: tỷ lệ nhận định thiếu bằng chứng giảm mạnh so với A1. "
             "Phân xử chọn lọc cần ít lượt gọi LLM hơn nhiều so với LLM-all; nDCG@10 thấp hơn 0.012 nhưng khoảng "
             "tin cậy chưa loại trừ mức kém hơn ε = 0.02, nên H3 không được ủng hộ.",
    "finetune": "Phần huấn luyện chạy trên Colab bằng notebook sinh từ mã nguồn của dự án. Dữ liệu huấn luyện là các "
                "cặp công khai ngoài lĩnh vực kiểm thử để tránh rò rỉ sang tập đánh giá. Bảng bên trái là trung "
                "bình 3 seed: mức cải thiện trên tập test của cả hai mô hình lớn hơn nhiều lần độ lệch chuẩn; mức "
                "cải thiện của dense trên tập dev nằm trong dao động giữa các seed. Biểu đồ bên phải là kết quả "
                "trên tập đánh giá của nhóm.",
    "agent": "Bộ lập kế hoạch LLM chỉ chọn trong tập công cụ có ràng buộc an toàn. Với mô hình 7B sau khi điều chỉnh "
             "ngân sách, H7 được ủng hộ, nhưng AG vẫn cần nhiều lượt gọi LLM hơn phương án dùng luật.",
    "public": "Trên dữ liệu công khai, lợi thế của hệ thống không lặp lại ở giá trị trung bình; slide tiếp theo "
              "phân tích nguyên nhân. Với SkillSpan, từ điển có precision cao nhưng recall thấp vì chỉ bao phủ thuật "
              "ngữ Automation Test; mô hình gán nhãn JobBERT đạt span F1 tương đương bài báo gốc. Kết hợp hai "
              "phương pháp không tăng F1 vì từ điển chỉ bổ sung rất ít đoạn đúng nhưng làm giảm precision.",
    "errors": "Đây là phân tích hậu kiểm, không phải giả thuyết đặt trước. Ba JD công khai là cùng một tin tuyển lập "
              "trình viên Ansible, qua được bộ lọc nhờ hai cụm từ chung chung; Criteria Profile cho Automation Test "
              "vì vậy áp dụng tiêu chí không phù hợp. Sau khi khử trùng lặp, thứ tự các cấu hình thay đổi, nhưng "
              "nhóm giữ nguyên tập đánh giá; bước tiếp theo là cải thiện bộ lọc JD rồi chạy lại.",
    "nfr": "Công bằng: hoán đổi tên không làm thay đổi thứ hạng nhờ che thông tin định danh trước khi mã hoá; quy tắc "
           "4/5 chưa đạt ở mọi JD. Độ trễ: p50 và p95 đo khi tắt lưu đệm, đạt mục tiêu vài phút cho một JD trên "
           "laptop.",
    "conclusion": "Tóm tắt kết quả vững chắc, các hạn chế và hướng phát triển; mời hội đồng đặt câu hỏi.",
}


# ------------------------------------------------------------------ patches on existing image slides

def patch_existing(prs, first_new: int, n_new: int) -> None:
    """Slide 26 says 'no results yet' and slide 27 marks the ladder 'not run': overlay updated notes."""
    s26, s27 = prs.slides[25], prs.slides[26]
    box(s26, 904 * PX, 651 * PX, 483 * PX, 55 * PX, fill=GREEN_BG, line=None, radius=0.12, name="Results note")
    text(s26, 916 * PX, 657 * PX, 460 * PX, 43 * PX,
         [[("Đã có kết quả thực nghiệm của nhóm. ", {"bold": True, "color": GREEN}),
           (f"Xem slide {first_new}–{first_new + n_new - 1}: H0–H7 được kiểm định trên dữ liệu chạy cục bộ.", {})]],
         size=8.5, anchor=MSO_ANCHOR.MIDDLE)
    box(s27, 733 * PX, 316 * PX, 149 * PX, 17 * PX, fill=WHITE, line=None, radius=None, name="Ran note")
    text(s27, 733 * PX, 317 * PX, 146 * PX, 15 * PX, f"Đã thực hiện, xem slide {first_new + 1}", size=6.5,
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
        ("ladder", "Loại bỏ thành phần", "Mỗi thành phần từ B0 đến P6 đóng góp bao nhiêu?",
         lambda pg, t: slide_ladder(prs, syn, pg, t)),
        ("hypotheses", "Kiểm định H0–H6", "Các giả thuyết H0–H6 có được dữ liệu ủng hộ không?",
         lambda pg, t: slide_hypotheses(prs, syn, pg, t)),
        ("prior", "Nghiên cứu liên quan", "Kết quả này so với các nghiên cứu trước ra sao?",
         lambda pg, t: slide_prior(prs, syn, pub, ctx, pg, t)),
        ("faith", "Độ trung thực, chi phí", "Bước kiểm chứng và phân xử đóng góp gì về độ trung thực và chi phí?",
         lambda pg, t: slide_faith(prs, syn, pg, t)),
        ("finetune", "Tinh chỉnh mô hình", "Tinh chỉnh theo hướng ConFit có khắc phục điểm yếu của B1 không?",
         lambda pg, t: slide_finetune(prs, syn, pub, ctx, pg, t)),
        ("agent", "Agentic AI (H7)", "Agent LLM tự lập kế hoạch có tốt hơn luật cố định không?",
         lambda pg, t: slide_agent(prs, h7, pg, t)),
        ("public", "Dữ liệu công khai", "Kết quả trên dữ liệu công khai và trích xuất kỹ năng ra sao?",
         lambda pg, t: slide_public(prs, pub, spans, pg, t, ctx)),
        ("errors", "Phân tích lỗi", "Vì sao Criteria Profile kém hơn trên dữ liệu công khai?",
         lambda pg, t: slide_errors(prs, pub, ctx, pg, t)),
        ("nfr", "Công bằng, độ trễ", "Hệ thống có thiên lệch theo tên không, và xử lý một JD mất bao lâu?",
         lambda pg, t: slide_nfr(prs, syn, ctx, pg, t)),
        ("conclusion", "Kết luận", "Tổng kết: kết quả nào vững chắc, hạn chế nào cần khắc phục?",
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
