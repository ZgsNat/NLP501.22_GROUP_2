"""Entity extraction with spans (FR-03), automation depth rubric (FR-04),
years of experience and parse-quality flagging (FR-05)."""
from __future__ import annotations

import re
from datetime import date
from typing import List, Optional, Tuple

from cv_agent.ingest.taxonomy import ALIAS_INDEX, SKILLS
from cv_agent.schema import DepthAssessment, Entity, Span

# ---------------------------------------------------------------- entities

def extract_entities(text: str) -> List[Entity]:
    taken: List[Tuple[int, int]] = []
    entities: List[Entity] = []
    for pattern, skill, _alias in ALIAS_INDEX:
        for m in pattern.finditer(text):
            s, e = m.span()
            if any(s < te and e > ts for ts, te in taken):
                continue
            taken.append((s, e))
            uri = SKILLS[skill.name].esco_uri if skill.name in SKILLS else skill.esco_uri
            entities.append(Entity(skill.type, skill.name, Span(s, e, text[s:e]), uri))
    for span, years in _explicit_years(text):
        entities.append(Entity("years", f"{years:g}", span))
    entities.sort(key=lambda en: en.span.start)
    return entities


# ---------------------------------------------------------------- years of experience

_YEARS_RE = re.compile(
    r"(\d{1,2}(?:[.,]\d)?)\s*\+?\s*(?:years?|yrs?|năm)\s*(?:of\s*)?(?:experience|exp|kinh\s*nghiệm|in\s+(?:software\s+)?(?:testing|qa|automation))",
    re.I,
)
# [dd/]mm/yyyy or yyyy, a separator, then the same or "present"
_RANGE_RE = re.compile(
    r"(?:(?:\d{1,2}[/.\-])?(\d{1,2})[/.\-])?((?:19|20)\d{2})\s*(?:[-–—~→]+|to|đến)\s*"
    r"(?:(?:(?:\d{1,2}[/.\-])?(\d{1,2})[/.\-])?((?:19|20)\d{2})|(present|now|current|nay|hiện\s*tại))",
    re.I,
)


def _explicit_years(text: str) -> List[Tuple[Span, float]]:
    out = []
    for m in _YEARS_RE.finditer(text):
        val = float(m.group(1).replace(",", "."))
        if 0 < val <= 40:
            out.append((Span(m.start(), m.end(), m.group(0)), val))
    return out


def estimate_years(text: str, today: Optional[date] = None) -> float:
    """Max of explicit statements; otherwise merged employment ranges."""
    explicit = [v for _, v in _explicit_years(text)]
    if explicit:
        return max(explicit)
    today = today or date.today()
    intervals = []
    for m in _RANGE_RE.finditer(text):
        sm, sy = int(m.group(1) or 1), int(m.group(2))
        if m.group(5):
            ey, em = today.year, today.month
        else:
            ey, em = int(m.group(4)), int(m.group(3) or 12)
        start, end = sy + (sm - 1) / 12, ey + em / 12  # end month is inclusive
        if 1980 <= sy <= today.year and end >= start:
            intervals.append((start, end))
    intervals.sort()
    total, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return round(total, 1)


# ---------------------------------------------------------------- automation depth rubric (1-4)

DEPTH_RUBRIC = {
    4: [r"automation\s+(?:architect|strategy|lead)", r"test\s+architect", r"lead(?:ing)?\s+(?:a\s+)?(?:team|qa|automation)",
        r"(?:define|defined|own(?:ed)?)\s+(?:the\s+)?(?:test|automation)\s+strategy", r"mentor(?:ed|ing)?",
        r"test\s+infrastructure", r"shift[-\s]left", r"quality\s+metrics", r"trưởng\s+nhóm", r"kiến\s+trúc"],
    3: [r"(?:built|build|design(?:ed)?|develop(?:ed)?|set\s*up|xây\s*dựng)\s+(?:an?\s+|the\s+)?(?:\w+\s+){0,3}(?:automation\s+)?framework",
        r"page\s+object(?:\s+model)?", r"\bpom\b", r"data[-\s]driven", r"keyword[-\s]driven", r"hybrid\s+framework",
        r"integrat(?:e|ed|ing)\s+(?:\w+\s+){0,4}(?:ci|pipeline|jenkins|gitlab)", r"parallel\s+execution", r"selenium\s+grid"],
    2: [r"(?:wrote|write|writing|develop(?:ed)?|maintain(?:ed)?|implement(?:ed)?|viết)\s+(?:\w+\s+){0,3}(?:automat\w*\s+)?(?:test\s+)?scripts?",
        r"automat(?:ed|ion)\s+(?:test\s+)?(?:scripts?|cases?|tests?)", r"test\s+automation", r"kiểm\s+thử\s+tự\s+động"],
    1: [r"manual\s+test", r"record\s*(?:and|&)\s*playback", r"test\s+case\s+design", r"kiểm\s+thử\s+thủ\s+công",
        r"execut(?:e|ed|ing)\s+test\s+cases?"],
}
_DEPTH_RE = {lvl: [re.compile(p, re.I) for p in pats] for lvl, pats in DEPTH_RUBRIC.items()}
DEPTH_LABELS = {1: "Manual / record-playback", 2: "Writes automated scripts", 3: "Designs & integrates framework",
                4: "Owns automation strategy / architecture"}


def assess_depth(text: str, has_automation_tool: bool) -> DepthAssessment:
    for level in (4, 3, 2, 1):
        spans = [Span(m.start(), m.end(), m.group(0)) for rx in _DEPTH_RE[level] for m in rx.finditer(text)]
        if spans:
            # level 4 without any framework/script evidence is capped at 3 (needs hands-on automation too)
            if level == 4 and not has_automation_tool:
                level = 3
            return DepthAssessment(level, spans[:3], DEPTH_LABELS[level])
    level = 2 if has_automation_tool else 1
    return DepthAssessment(level, [], DEPTH_LABELS[level] + " (inferred, no explicit evidence)")


# ---------------------------------------------------------------- language & quality

_VI_CHARS = re.compile(r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I)


def detect_language(text: str) -> str:
    letters = sum(ch.isalpha() for ch in text) or 1
    return "vi" if len(_VI_CHARS.findall(text)) / letters > 0.03 else "en"


def parse_quality(text: str, n_entities: int) -> Tuple[float, List[str]]:
    flags: List[str] = []
    stripped = text.strip()
    if len(stripped) < 300:
        flags.append("short_text")
    printable = sum(ch.isprintable() or ch in "\n\t" for ch in stripped) / max(len(stripped), 1)
    alpha = sum(ch.isalpha() for ch in stripped) / max(len(stripped), 1)
    if printable < 0.95 or alpha < 0.45:
        flags.append("garbled_text")
    if n_entities < 3:
        flags.append("few_entities")
    quality = max(0.0, 1.0 - 0.3 * len(flags))
    if quality < 0.7:
        flags.append("low_confidence_parse")
    return round(quality, 2), flags
