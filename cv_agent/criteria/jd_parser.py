"""Parse a JD into must-have H(c) and nice-to-have requirements over the 8 axes (FR-07)."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from cv_agent.ingest.extractor import extract_entities
from cv_agent.ingest.pii import mask_contacts
from cv_agent.schema import JD, Requirement

ENTITY_TO_AXIS = {"tool": "tools", "language": "languages", "test_type": "test_types", "cicd": "cicd",
                  "certification": "certification", "foreign_lang": "foreign_lang"}

_MUST_HEADER = re.compile(r"^\s*(?:#+\s*)?(?:requirements?|qualifications?|must[-\s]have|yêu\s*cầu(?:\s*công\s*việc)?|"
                          r"what\s+you\s+(?:need|bring)|job\s+requirements?)\s*:?\s*$", re.I)
_NICE_HEADER = re.compile(r"^\s*(?:#+\s*)?(?:nice[-\s]to[-\s]have|preferred(?:\s+qualifications?)?|bonus|plus|"
                          r"ưu\s*tiên|điểm\s*cộng|is\s+a\s+plus)\s*:?\s*$", re.I)
_OTHER_HEADER = re.compile(r"^\s*(?:#+\s*)?(?:benefits?|responsibilit(?:y|ies)|job\s+description|mô\s*tả\s*công\s*việc|"
                           r"quyền\s*lợi|about\s+us|why\s+join|we\s+offer|what\s+you\s+will\s+do)\s*:?\s*$", re.I)
_MUST_CUE = re.compile(r"\b(?:must|required|requirement|mandatory|at\s+least|minimum|bắt\s*buộc|tối\s*thiểu|"
                       r"cần\s*có|yêu\s*cầu)\b", re.I)
_NICE_CUE = re.compile(r"(?:nice\s+to\s+have|is\s+a\s+plus|\bplus\b|preferred|advantage|bonus|ưu\s*tiên|lợi\s*thế|"
                       r"điểm\s*cộng)", re.I)
_ANY_OF = re.compile(r"\b(?:or|hoặc|such\s+as|e\.g\.|like|như|one\s+of)\b|/", re.I)
# "3-5 years" -> lower bound 3; "at least 2 years" -> 2
_YEARS = re.compile(r"(?:at\s+least|minimum|min\.?|over|more\s+than|tối\s*thiểu|trên|từ)?\s*(\d{1,2})"
                    r"(?:\s*(?:-|–|to|đến)\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?|năm)", re.I)
_FRAMEWORK = re.compile(r"(?:build|design|develop|xây\s*dựng|thiết\s*kế)\w*\s+(?:\w+\s+){0,3}(?:automation\s+)?framework", re.I)
_STRATEGY = re.compile(r"(?:automation|test)\s+(?:strategy|architect)|lead\s+(?:the\s+)?(?:automation|qa)\s+team", re.I)
_SCRIPT = re.compile(r"(?:write|writing|develop)\w*\s+(?:\w+\s+){0,2}(?:automat\w+\s+)?(?:test\s+)?scripts?|"
                     r"automation\s+test(?:ing)?|kiểm\s*thử\s*tự\s*động", re.I)


def _sections(text: str) -> List[Tuple[Optional[bool], str]]:
    """Split into lines tagged with the section's default must-ness (True/False/None)."""
    mode: Optional[bool] = None
    out = []
    for line in text.splitlines():
        if _MUST_HEADER.match(line):
            mode = True
            continue
        if _NICE_HEADER.match(line):
            mode = False
            continue
        if _OTHER_HEADER.match(line):
            mode = None
            continue
        if line.strip():
            out.append((mode, line))
    return out


def _line_is_must(section_mode: Optional[bool], line: str) -> Optional[bool]:
    if _NICE_CUE.search(line):
        return False
    if _MUST_CUE.search(line):
        return True
    return section_mode


def parse_jd(text: str, jd_id: str = "jd", title: str = "", company_type: str = "default") -> JD:
    # recruiter contacts never reach the LLM prompts (the JD text is part of every prompt)
    text = mask_contacts(unicodedata.normalize("NFC", text))
    reqs: List[Requirement] = []
    seen: Dict[Tuple[str, str], int] = {}

    def add(req: Requirement) -> None:
        key = (req.axis, "|".join(req.values))
        if key in seen:  # keep the stronger requirement (must wins)
            idx = seen[key]
            if req.must and not reqs[idx].must:
                reqs[idx] = req
            return
        seen[key] = len(reqs)
        reqs.append(req)

    for section_mode, line in _sections(text):
        must = _line_is_must(section_mode, line)
        if must is None:  # responsibilities / benefits: informative only, treat as nice-to-have signal
            must = False
        line_s = line.strip(" -•*\t")
        by_axis: Dict[str, List[str]] = {}
        for ent in extract_entities(line):
            axis = ENTITY_TO_AXIS.get(ent.type)
            if axis and ent.value not in by_axis.setdefault(axis, []):
                by_axis[axis].append(ent.value)
        any_of = bool(_ANY_OF.search(line))
        for axis, values in by_axis.items():
            if any_of or axis == "foreign_lang":
                add(Requirement(axis, tuple(values), must=must, text=line_s))
            else:
                for v in values:
                    add(Requirement(axis, (v,), must=must, text=line_s))
        m = _YEARS.search(line)
        if m and re.search(r"experience|kinh\s*nghiệm|exp\b", line, re.I):
            add(Requirement("experience", (), min_level=float(m.group(1)), must=must, text=line_s))
        depth = 4 if _STRATEGY.search(line) else 3 if _FRAMEWORK.search(line) else 2 if _SCRIPT.search(line) else 0
        if depth:
            add(Requirement("depth", (), min_level=float(depth), must=must and depth <= 3, text=line_s))
    return JD(jd_id=jd_id, title=title or _guess_title(text), text=text, company_type=company_type, requirements=reqs)


def _guess_title(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip(" #")[:120]
    return "Untitled JD"


def requirements_as_dicts(jd: JD) -> List[Dict]:
    return [{"axis": r.axis, "values": list(r.values), "min_level": r.min_level, "must": r.must} for r in jd.requirements]
