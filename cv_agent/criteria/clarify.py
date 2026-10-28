"""Detect vague / conflicting criteria and generate HR questions with options (FR-09, H6).

Each option carries a *patch* applied immutably to the JD's requirements via
``apply_answers``. The orchestrator loops ask -> apply -> re-check, at most
``max_rounds`` times (Loop 1).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

from cv_agent.schema import JD, Requirement


@dataclass(frozen=True)
class Option:
    label: str
    patch: tuple   # tuple of patch dicts (frozen-friendly)


@dataclass(frozen=True)
class Question:
    qid: str
    kind: str              # vague | conflict | missing
    text: str
    options: tuple
    recommended: int = 0   # index used by the simulated HR in experiments


_SENIOR = re.compile(r"\b(?:senior|experienced|seasoned|nhiều\s+kinh\s+nghiệm|lâu\s+năm|dày\s+dặn)\b", re.I)
_JUNIOR = re.compile(r"\b(?:junior|fresher|intern|entry[-\s]level|mới\s+ra\s+trường)\b", re.I)
_VAGUE_EN = re.compile(r"(?:good|fluent|strong|tốt)\s+(?:english|tiếng\s+anh)|english\s+(?:skills?|communication)|"
                       r"tiếng\s+anh\s+(?:tốt|giao\s+tiếp)", re.I)
_VAGUE_TOOLS = re.compile(r"\b(?:automation\s+tools?|testing\s+tools?|công\s+cụ\s+(?:kiểm\s+thử|tự\s+động))\b|\betc\.?", re.I)


def _p(**kw) -> dict:
    return kw


def detect_issues(jd: JD) -> List[Question]:
    qs: List[Question] = []
    exp = [r for r in jd.requirements if r.axis == "experience"]
    musts = jd.must_haves

    if _SENIOR.search(jd.text) and not exp:
        qs.append(Question("seniority_years", "vague", "JD nói 'senior/nhiều kinh nghiệm' nhưng không ghi số năm. Tối thiểu bao nhiêu năm?",
                           tuple(Option(f"≥ {y} năm (bắt buộc)", (_p(op="set_min", axis="experience", value=y, must=True),))
                                 for y in (3, 5, 7)) + (Option("Không bắt buộc", ()),), recommended=0))

    if _JUNIOR.search(jd.text) and any(r.min_level >= 3 for r in exp):
        y = max(r.min_level for r in exp)
        qs.append(Question("junior_vs_years", "conflict", f"JD tuyển 'junior/fresher' nhưng yêu cầu ≥ {y:g} năm. Giữ yêu cầu nào?",
                           (Option("Hạ xuống ≥ 1 năm", (_p(op="set_min", axis="experience", value=1, must=True),)),
                            Option(f"Giữ ≥ {y:g} năm", ()),
                            Option("Bỏ yêu cầu số năm", (_p(op="drop", axis="experience"),))), recommended=0))

    langs = [r for r in jd.requirements if r.axis == "foreign_lang"]
    if _VAGUE_EN.search(jd.text) and langs:
        qs.append(Question("english_level", "vague", "'Tiếng Anh tốt' nghĩa là gì với vị trí này?",
                           (Option("Đọc hiểu tài liệu — điểm cộng", (_p(op="set_must", axis="foreign_lang", must=False),)),
                            Option("Giao tiếp với khách hàng — bắt buộc", (_p(op="set_must", axis="foreign_lang", must=True),)),
                            Option("Không cần", (_p(op="drop", axis="foreign_lang"),))), recommended=0))

    tool_musts = [r for r in musts if r.axis == "tools" and len(r.values) == 1]
    if len(tool_musts) >= 4:
        names = ", ".join(r.values[0] for r in tool_musts)
        qs.append(Question("tools_all_or_any", "conflict",
                           f"JD bắt buộc {len(tool_musts)} công cụ ({names}) — rất ít ứng viên có đủ. Cần TẤT CẢ hay MỘT TRONG CÁC?",
                           (Option("Một trong các công cụ", (_p(op="merge_any", axis="tools"),)),
                            Option("Tất cả", ()),
                            Option("Chuyển thành điểm cộng", (_p(op="set_must", axis="tools", must=False),))), recommended=0))

    if not any(r.axis == "tools" for r in jd.requirements) and _VAGUE_TOOLS.search(jd.text):
        families = {"Web (Selenium/Playwright/Cypress)": ("selenium", "playwright", "cypress"),
                    "Mobile (Appium)": ("appium", "espresso", "xcuitest"),
                    "API (Postman/Rest Assured)": ("postman", "rest assured")}
        qs.append(Question("tool_family", "vague", "JD chỉ ghi 'automation tools'. Nhóm công cụ nào là bắt buộc?",
                           tuple(Option(k, (_p(op="add", axis="tools", values=v, must=True),)) for k, v in families.items())
                           + (Option("Không bắt buộc công cụ cụ thể", ()),), recommended=0))

    if not musts:
        qs.append(Question("no_must_have", "missing", "JD không có tiêu chí bắt buộc nào. Chọn knock-out?",
                           (Option("≥ 2 năm kinh nghiệm", (_p(op="set_min", axis="experience", value=2, must=True),)),
                            Option("Biết viết automation script (depth ≥ 2)", (_p(op="set_min", axis="depth", value=2, must=True),)),
                            Option("Không knock-out", ())), recommended=1))
    return qs


def apply_patch(reqs: List[Requirement], patch: dict) -> List[Requirement]:
    op, axis = patch["op"], patch.get("axis")
    if op == "set_min":
        others = [r for r in reqs if r.axis != axis]
        return others + [Requirement(axis, (), min_level=float(patch["value"]), must=patch.get("must", True),
                                     text=f"[HR] {axis} ≥ {patch['value']}")]
    if op == "drop":
        return [r for r in reqs if r.axis != axis]
    if op == "set_must":
        return [replace(r, must=patch["must"]) if r.axis == axis else r for r in reqs]
    if op == "add":
        return reqs + [Requirement(axis, tuple(patch["values"]), must=patch.get("must", True), text=f"[HR] {axis}")]
    if op == "merge_any":
        musts = [r for r in reqs if r.axis == axis and r.must]
        merged = Requirement(axis, tuple(v for r in musts for v in r.values), must=True, text="[HR] any-of")
        return [r for r in reqs if not (r.axis == axis and r.must)] + [merged]
    raise ValueError(f"Unknown patch op {op}")


def apply_answers(jd: JD, questions: List[Question], answers: Dict[str, int]) -> JD:
    """Return a NEW JD with the chosen options applied. ``answers``: qid -> option index."""
    reqs = list(jd.requirements)
    for q in questions:
        idx = answers.get(q.qid)
        if idx is None:
            continue
        if not 0 <= idx < len(q.options):
            raise ValueError(f"Answer {idx} out of range for question {q.qid}")
        for patch in q.options[idx].patch:
            reqs = apply_patch(reqs, patch)
    return replace(jd, requirements=reqs)


def simulated_hr(questions: List[Question], preset: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Experiment stand-in for HR: gold answers if provided, else the recommended option."""
    preset = preset or {}
    return {q.qid: preset.get(q.qid, q.recommended) for q in questions}
