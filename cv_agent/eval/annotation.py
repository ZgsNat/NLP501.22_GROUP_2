"""Human relevance labelling for a real Gold Set: pooling, labelling sheets, agreement and merging.

* ``pool``: TREC-style depth-k pooling — the union of the top ``depth`` CVs of several systems plus a random
  sample of the rest (so CVs no system ranks high are also judged). The order is shuffled, so an annotator
  cannot tell which system proposed a CV or how high.
* ``write_sheet`` / ``read_sheet``: one Excel workbook per annotator (guideline, JDs, one row per JD–CV pair with
  a 0–3 drop-down). Each annotator gets a different row order to spread order effects.
* ``weighted_kappa``: quadratic-weighted Cohen κ for the ordinal 0–3 scale.
* ``merge``: agreement → that grade; one step apart → the lower grade; two or more apart → must be adjudicated.
  The result has the ``gold.json`` layout used by the experiments (``judged_only``: unjudged CVs are skipped).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

GRADES = {0: "Không phù hợp", 1: "Phù hợp một phần", 2: "Phù hợp", 3: "Rất phù hợp"}
LABEL_SHEET, JD_SHEET, GUIDE_SHEET = "Nhãn", "JD", "Hướng dẫn"
HEADER = ("STT", "jd_id", "cv_id", "Vị trí (JD)", "CV", "Nhãn (0–3)", "Ghi chú")
GRADE_COL = HEADER.index("Nhãn (0–3)")
MAX_ADJACENT_GAP = 1          # larger disagreements go to adjudication
Item = Tuple[str, str]        # (jd_id, cv_id)

GUIDELINE = [
    "Mục tiêu: đánh giá mức phù hợp của từng CV với JD cho vị trí kiểm thử tự động (Automation Test).",
    "Chỉ dựa trên nội dung JD và CV; không đoán thông tin không có trong CV.",
    "3 · Rất phù hợp: đáp ứng mọi yêu cầu bắt buộc (công cụ/ngôn ngữ chính, số năm kinh nghiệm, loại kiểm thử) và "
    "có bằng chứng tự xây hoặc mở rộng framework; sẵn sàng phỏng vấn ngay.",
    "2 · Phù hợp: đáp ứng phần lớn yêu cầu bắt buộc; thiếu một yêu cầu nhỏ hoặc công cụ tương đương (ví dụ Cypress "
    "thay Playwright); nên phỏng vấn.",
    "1 · Phù hợp một phần: có kinh nghiệm kiểm thử liên quan nhưng thiếu yêu cầu bắt buộc quan trọng (ví dụ chủ yếu "
    "kiểm thử thủ công, sai ngôn ngữ lập trình chính, thiếu nhiều năm kinh nghiệm); chỉ cân nhắc khi thiếu ứng viên.",
    "0 · Không phù hợp: không có kinh nghiệm kiểm thử tự động hoặc khác hẳn vai trò.",
    "Trình độ tiếng Anh và số năm kinh nghiệm chỉ trừ điểm khi JD yêu cầu rõ.",
    "Làm độc lập: không trao đổi với người gán nhãn còn lại, không xem kết quả của hệ thống.",
    "Ghi chú ngắn lý do khi chọn 1 hoặc 2, hoặc khi phân vân giữa hai mức.",
]


def pool(rankings: Mapping[str, Sequence[str]], universe: Sequence[str], depth: int = 20, n_random: int = 10,
         seed: int = 42) -> List[str]:
    top = {cv for ranked in rankings.values() for cv in list(ranked)[:depth]}
    rng = random.Random(seed)
    rest = sorted(set(universe) - top)
    pooled = sorted(top) + rng.sample(rest, min(n_random, len(rest)))
    rng.shuffle(pooled)
    return pooled


def weighted_kappa(a: Mapping[str, int], b: Mapping[str, int], k: int = len(GRADES)) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    n = len(keys)
    obs = [[0.0] * k for _ in range(k)]
    for key in keys:
        obs[a[key]][b[key]] += 1 / n
    pa = [sum(row) for row in obs]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1 - num / den if den else 1.0


def _clean(text: str) -> str:
    """Excel cells reject ASCII control characters (other than tab / newline), which some CVs contain."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    return ILLEGAL_CHARACTERS_RE.sub("", text)


def write_sheet(path: Path, annotator: str, items: Sequence[Item], jds: Mapping[str, Tuple[str, str]],
                cvs: Mapping[str, str], seed: int = 0) -> None:
    """``jds``: jd_id → (title, text); ``cvs``: cv_id → text."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    guide = wb.active
    guide.title = GUIDE_SHEET
    guide.append([f"Người gán nhãn: {annotator}"])
    for line in GUIDELINE:
        guide.append([line])
    guide.column_dimensions["A"].width = 140
    for row in guide.iter_rows():
        row[0].alignment = Alignment(wrap_text=True, vertical="top")

    jd_ws = wb.create_sheet(JD_SHEET)
    jd_ws.append(["jd_id", "Vị trí", "Nội dung JD"])
    for jd_id, (title, text) in sorted(jds.items()):
        jd_ws.append([jd_id, _clean(title), _clean(text)])
    for col, width in zip("ABC", (18, 40, 140)):
        jd_ws.column_dimensions[col].width = width

    ws = wb.create_sheet(LABEL_SHEET)
    ws.append(list(HEADER))
    order = list(items)
    random.Random(f"{seed}-{annotator}").shuffle(order)
    for i, (jd_id, cv_id) in enumerate(order, 1):
        ws.append([i, jd_id, cv_id, _clean(jds[jd_id][0]), _clean(cvs[cv_id]), None, None])
    dv = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True)
    ws.add_data_validation(dv)
    letter = chr(ord("A") + GRADE_COL)
    dv.add(f"{letter}2:{letter}{len(order) + 1}")
    for col, width in zip("ABCDEFG", (6, 16, 16, 34, 110, 11, 40)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    wb.active = wb.index(ws)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_sheet(path: Path) -> Tuple[Dict[Item, int], List[Item]]:
    """Grades found and the items left blank or with a value outside 0–3."""
    from openpyxl import load_workbook
    ws = load_workbook(path, read_only=True, data_only=True)[LABEL_SHEET]
    labels: Dict[Item, int] = {}
    missing: List[Item] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or row[1] is None:
            continue
        item, value = (str(row[1]), str(row[2])), row[GRADE_COL]
        try:
            grade = int(value)
        except (TypeError, ValueError):
            grade = -1
        if grade in GRADES and float(value) == grade:
            labels[item] = grade
        else:
            missing.append(item)
    return labels, missing


def _by_jd(labels: Mapping[Item, int]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (jd, cv), grade in labels.items():
        out[jd][cv] = grade
    return dict(out)


def agreement_report(a: Mapping[Item, int], b: Mapping[Item, int]) -> Dict:
    common = sorted(set(a) & set(b))
    flat_a = {f"{j}|{c}": a[(j, c)] for j, c in common}
    flat_b = {f"{j}|{c}": b[(j, c)] for j, c in common}
    per_jd = {}
    for jd in sorted({j for j, _ in common}):
        ka = {c: g for c, g in _by_jd(a).get(jd, {}).items() if (jd, c) in b}
        kb = {c: b[(jd, c)] for c in ka}
        per_jd[jd] = {"n": len(ka), "weighted_kappa": weighted_kappa(ka, kb),
                      "exact_agreement": sum(ka[c] == kb[c] for c in ka) / len(ka)}
    return {"overall": {"n": len(common), "weighted_kappa": weighted_kappa(flat_a, flat_b),
                        "exact_agreement": sum(flat_a[k] == flat_b[k] for k in flat_a) / len(flat_a)
                        if flat_a else math.nan},
            "per_jd": per_jd}


def merge(a: Mapping[Item, int], b: Mapping[Item, int], splits: Mapping[str, str],
          adjudicated: Optional[Mapping[Item, int]] = None) -> Tuple[Dict, List[Tuple[str, str, int, int]]]:
    """``gold.json`` content and the unresolved far disagreements (gold is empty while any remain)."""
    adjudicated = adjudicated or {}
    final: Dict[Item, int] = {}
    conflicts = []
    for item in sorted(set(a) & set(b)):
        ga, gb = a[item], b[item]
        if item in adjudicated:
            final[item] = adjudicated[item]
        elif abs(ga - gb) <= MAX_ADJACENT_GAP:
            final[item] = min(ga, gb)
        else:
            conflicts.append((*item, ga, gb))
    if conflicts:
        return {}, conflicts
    rel, by_a, by_b = _by_jd(final), _by_jd(a), _by_jd(b)
    gold = {}
    for jd, grades in rel.items():
        gold[jd] = {"relevance": grades, "hr_ranking": sorted(grades, key=lambda c: (-grades[c], c)),
                    "split": splits.get(jd, "test"), "ambiguous": False, "judged_only": True,
                    "annotator_a": {c: by_a[jd][c] for c in grades}, "annotator_b": {c: by_b[jd][c] for c in grades}}
    return gold, []
