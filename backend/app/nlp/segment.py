"""Stage 4: CV structure. A CV is not a bag of words: "Selenium" under *Experience* is evidence, under *Skills* it is a
claim. We recover (a) canonical sections from free-form headings in English and Vietnamese, (b) role/project entries
inside them, and (c) date ranges → months, so years of experience are computed instead of guessed by an LLM."""

import re
from dataclasses import dataclass
from datetime import date
from difflib import SequenceMatcher

from app.nlp.text import fold

SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "professional profile", "about me", "about", "overview",
                "objective", "career objective", "tóm tắt", "giới thiệu", "giới thiệu bản thân", "mục tiêu",
                "mục tiêu nghề nghiệp", "tổng quan"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "employment",
                   "work history", "career history", "kinh nghiệm", "kinh nghiệm làm việc", "quá trình công tác",
                   "quá trình làm việc"],
    "projects": ["projects", "key projects", "selected projects", "project experience", "personal projects", "dự án",
                 "dự án tiêu biểu", "dự án đã tham gia", "các dự án"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "competencies", "tools", "tech stack",
               "skills and tools", "kỹ năng", "kĩ năng", "kỹ năng chuyên môn", "công nghệ"],
    "education": ["education", "education and certifications", "education and training", "academic background",
                  "học vấn", "trình độ học vấn", "đào tạo"],
    "certifications": ["certifications", "certification", "certificates", "licenses", "chứng chỉ", "bằng cấp"],
    "other": ["languages", "interests", "hobbies", "references", "activities", "awards", "ngoại ngữ", "sở thích",
              "hoạt động", "giải thưởng"],
}
ALIAS_TO_KIND = {a: k for k, aliases in SECTION_ALIASES.items() for a in aliases}
FOLDED_ALIASES = {fold(a): k for a, k in ALIAS_TO_KIND.items()}
LOOSE_HEADING = 0.85  # "KINH NGHIM" (OCR) or "KY NANG" (no accents) vs the folded alias; measured in docs/OCR_RESULTS.md
ENTRY_SECTIONS = ("experience", "projects")

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_D = (r"(?:(?:tháng|th\.?|t)\s*\d{1,2}\s*[/.-]\s*\d{4}"     # Tháng 6/2019, T6/2019
      r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{4}"  # Jan 2020, September 2019
      r"|\d{4}\s*[/.-]\s*\d{1,2}(?!\d)"                      # 2019/05, 2019-05
      r"|\d{1,2}\s*[/.-]\s*\d{4}"                            # 03/2021
      r"|(?:19|20)\d{2})")                                    # 2018
_NOW = r"(?:present|now|current|currently|today|ongoing|nay|hiện tại|hiện nay|đến nay|bây giờ)"
RANGE = re.compile(rf"(?P<a>{_D})\s*(?:–|—|-|~|→|\bto\b|\buntil\b|\btill\b|\bđến\b)\s*(?P<b>{_D}|{_NOW})", re.I)
SINCE = re.compile(rf"(?:\bsince\b|\bfrom\b|\btừ\b)\s+(?P<a>{_D})(?!\s*(?:–|—|-|to|đến))", re.I)


@dataclass
class Section:
    kind: str  # summary | experience | projects | skills | education | certifications | other | header
    heading: str
    start: int  # char offsets into the processed CV text (body only, heading excluded)
    end: int


@dataclass
class Entry:
    """One role or project inside an experience/projects section."""

    section: str
    title: str
    start: int
    end: int
    start_month: int | None = None  # months since year 0; None if no dates found
    end_month: int | None = None


def heading_kind(line: str) -> str | None:
    """Canonical section kind if `line` is a section heading, else None. '###' lines are entries, not sections."""
    s = line.strip()
    md = re.match(r"^(#{1,2})\s+(.*)$", s)
    if s.startswith("###"):
        return None
    core = (md[2] if md else s).strip().strip("*_").strip().rstrip(":：").strip().lower().replace("&", "and")
    core = re.sub(r"\s*\(.*\)$", "", re.sub(r"\s+", " ", core))  # "Work Experience (2016–2025)"
    # Bilingual headings: "Summary / Tóm tắt", "Kinh nghiệm | Experience"
    kind = next((ALIAS_TO_KIND[p.strip()] for p in re.split(r"\s*[/|]\s*", core) if p.strip() in ALIAS_TO_KIND), None)
    if not kind and (s.isupper() or s.endswith((":", "："))) and len(core.split()) <= 4:
        kind = loose_heading(core)
    if kind:
        if md or s.isupper() or s.rstrip().endswith((":", "：")) or s.startswith("**") or len(core.split()) <= 6:
            return kind
    return "other" if md else None  # an unknown markdown heading still ends the previous section


def loose_heading(core: str) -> str | None:
    """The section kind of a heading typed without diacritics or misread by OCR, if one alias is close enough."""
    f = fold(core)
    best = max(FOLDED_ALIASES, key=lambda a: SequenceMatcher(None, f, a).ratio())
    return FOLDED_ALIASES[best] if SequenceMatcher(None, f, best).ratio() >= LOOSE_HEADING else None


def sections(text: str) -> list[Section]:
    out: list[Section] = []
    pos, kind, heading, start = 0, "header", "", 0
    for line in text.split("\n"):
        k = heading_kind(line) if line.strip() else None
        if k:
            if text[start:pos].strip():
                out.append(Section(kind, heading, start, pos))
            kind, heading, start = k, line.strip().lstrip("#").strip(), pos + len(line) + 1
        pos += len(line) + 1
    if text[start:].strip():
        out.append(Section(kind, heading, start, len(text)))
    return out


def _month(token: str, end: bool) -> int | None:
    t = token.lower().strip()
    if m := re.match(r"(?:tháng|th\.?|t)\s*(\d{1,2})\s*[/.-]\s*(\d{4})", t):
        mo, y = int(m[1]), int(m[2])
    elif m := re.match(r"([a-z]{3})[a-z]*\.?\s+(\d{4})", t):
        mo, y = MONTHS.get(m[1], 0), int(m[2])
    elif m := re.match(r"(\d{4})\s*[/.-]\s*(\d{1,2})$", t):
        y, mo = int(m[1]), int(m[2])
    elif m := re.match(r"(\d{1,2})\s*[/.-]\s*(\d{4})$", t):
        mo, y = int(m[1]), int(m[2])
    elif m := re.fullmatch(r"(\d{4})", t):
        # Year-only ranges ("2018–2021") are read as Jan→Jan, i.e. the plain year difference people mean.
        return int(m[1]) * 12 if 1970 <= int(m[1]) <= 2100 else None
    else:
        return None
    if not (1 <= mo <= 12 and 1970 <= y <= 2100):
        return None
    return y * 12 + mo - 1 + (1 if end else 0)  # month-precise end is inclusive


def reference_date() -> date:
    """What "Present" means: today, or Settings.reference_date so an evaluation gives the same years on any day."""
    from app.config import get_settings

    return get_settings().reference_date or date.today()


def date_range(line: str, today: date | None = None) -> tuple[int, int] | None:
    """'Jan 2020 – Present' → (start_month, end_month) in months since year 0, or None."""
    today = today or reference_date()
    now = today.year * 12 + today.month
    if m := RANGE.search(line):
        a = _month(m["a"], end=False)
        b = now if re.fullmatch(_NOW, m["b"].strip(), re.I) else _month(m["b"], end=True)
    elif m := SINCE.search(line):
        a, b = _month(m["a"], end=False), now
    else:
        return None
    if a is None or b is None or not 0 <= b - a <= 45 * 12:
        return None
    return a, b


def entries(text: str, secs: list[Section], today: date | None = None) -> list[Entry]:
    """Split experience/projects sections into entries. An entry starts at a '###' line or at a non-bullet line that
    carries a date range; a date-only line attaches to the title line right above it."""
    out: list[Entry] = []
    for sec in (s for s in secs if s.kind in ENTRY_SECTIONS):
        lines, pos, anchors = text[sec.start:sec.end].split("\n"), sec.start, []
        offsets = []
        for line in lines:
            offsets.append(pos)
            pos += len(line) + 1
        for i, line in enumerate(lines):
            s = line.strip()
            if not s or s.startswith("- "):
                continue
            rng = date_range(s, today)
            if s.startswith("###") or s.startswith("**") or rng:
                j = i
                # the previous non-empty line: double-spaced CVs put a blank line between title and dates
                k = next((x for x in range(i - 1, -1, -1) if lines[x].strip()), None)
                # "Senior QA — Bank\nJan 2020 – Present": the dates line belongs to the title above it
                if rng and len(RANGE.sub("", s).strip(" |,()–-")) < 4 and k is not None \
                        and not lines[k].strip().startswith("- ") and (not anchors or anchors[-1][0] != k):
                    j = k
                if anchors and anchors[-1][0] == j:
                    continue
                if anchors and anchors[-1][0] == k and rng and not anchors[-1][1]:
                    anchors[-1] = (anchors[-1][0], rng)  # title line then date line
                    continue
                anchors.append((j, rng))
        if not anchors:
            out.append(Entry(sec.kind, sec.heading, sec.start, sec.end))
            continue
        for n, (i, rng) in enumerate(anchors):
            if rng is None:  # dates may sit on the next non-empty line(s)
                rng = next((r for r in (date_range(x, today) for x in [ln for ln in lines[i + 1:i + 6] if ln.strip()][:2])
                            if r), None)
            end = offsets[anchors[n + 1][0]] if n + 1 < len(anchors) else sec.end
            title = lines[i].strip().lstrip("#*").strip().rstrip("*").strip()
            out.append(Entry(sec.kind, title, offsets[i], end, *(rng or (None, None))))
    return out


def merged_months(spans: list[tuple[int, int]]) -> int:
    """Total months covered by possibly overlapping intervals (a freelance job in parallel is not double-counted)."""
    total, cur = 0, None
    for a, b in sorted(spans):
        if cur and a <= cur[1]:
            cur = (cur[0], max(cur[1], b))
        else:
            total += cur[1] - cur[0] if cur else 0
            cur = (a, b)
    return total + (cur[1] - cur[0] if cur else 0)
