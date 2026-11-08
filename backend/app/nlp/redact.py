"""Stage 3: remove contact details and protected attributes BEFORE anything is indexed or shown to an LLM.

The fairness policy (POL-ALL-fairness) says the ranking must never use name, age, gender, marital status, nationality,
religion, health, photo or address. Asking the LLM to "ignore" them is a hope; deleting them is a guarantee. Labels are
matched in English and Vietnamese because real Vietnamese CVs put "Ngày sinh / Giới tính" in the header."""

import re
from difflib import SequenceMatcher

from app.nlp.injection import sanitize
from app.nlp.segment import heading_kind
from app.nlp.text import fold

# "Label: value" lines whose whole line is removed. Order matters only for the kind reported.
LABELLED = {
    "name": r"full name|name|họ và tên|họ tên|tên",
    "birth_date": r"date of birth|dob|born|birthday|ngày sinh|năm sinh",
    "age": r"age|tuổi",
    "gender": r"gender|sex|giới tính",
    "marital_status": r"marital status|tình trạng hôn nhân|hôn nhân",
    "nationality": r"nationality|quốc tịch|dân tộc",
    "religion": r"religion|tôn giáo",
    "health": r"health|sức khỏe|sức khoẻ",
    "address": r"address|địa chỉ|nơi ở|quê quán",
    "photo": r"photo|ảnh",
}
LABEL_LINE = re.compile(
    r"^[ \t]*(?:[-*#]+\s*)?(?:\*\*)?(?P<label>" + "|".join(f"(?:{v})" for v in LABELLED.values()) + r")(?:\*\*)?\s*[:：]"
    r".*$",
    re.I | re.M,
)
# A "label: value" line whose label is typed without diacritics or misread by OCR ("Dia chi:", "Gii tính:"): matched
# after folding against labels of 5+ letters (shorter ones such as "age" or "tên" are too easy to hit by accident).
LOOSE_LINE = re.compile(r"^[ \t]*(?:[-*#]+\s*)?(?:\*\*)?(?P<label>[^\n:：*\[]{3,25}?)(?:\*\*)?\s*[:：].*$", re.M)
FOLDED_LABELS = {fold(a): k for k, v in LABELLED.items() for a in v.split("|") if len(fold(a)) >= 5}
LOOSE_LABEL = 0.8
INLINE = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "url": re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|facebook)\.com/\S+", re.I),
    # Starts with "+<country>" or a trunk "0" (VN: 09xx…), so year ranges like "2019.05-2021" never match.
    "phone": re.compile(r"(?<![\w/.])(?:\+\d{1,3}[\s.-]?|0)\d{1,4}(?:[\s.-]?\d{2,4}){2,4}(?![\w/])"),
}


# "Label: value" in the middle of a line, as when a PDF lost its line breaks ("… 0928134815 Giới tính: Nữ Ngày sinh: …").
# Only labels that are unambiguous mid-sentence ("name", "age", "address" alone are not: "Project name: Payment app"),
# and only a value of the expected shape, so the words after it survive.
_LABEL_START = r"marital|gender|sex|date|dob|birthday|born|nationality|religion|address|phone|email|mobile|age|" \
               r"giới|ngày|năm|quốc|dân|tôn|tình|địa|điện"
_NAME_TOKEN = rf"(?!(?i:{_LABEL_START})\b)[A-ZĐÀ-Ỹ][\w'’-]*"  # a capitalised word that does not start the next label
_WORDS = r"[^\s:：.,;|]+(?:\s+[^\s:：.,;|]+){0,2}"
INLINE_LABELLED = {
    "name": (r"full name|họ và tên|họ tên", rf"{_NAME_TOKEN}(?:\s+{_NAME_TOKEN}){{0,4}}"),
    "birth_date": (r"date of birth|dob|birthday|ngày sinh|năm sinh", r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}"),
    "gender": (r"gender|giới tính", r"(?i:male|female|nam|nữ|other|khác)\b"),
    "marital_status": (r"marital status|tình trạng hôn nhân", _WORDS),
    "nationality": (r"nationality|quốc tịch|dân tộc", _WORDS),
    "religion": (r"religion|tôn giáo", _WORDS),
}
INLINE_LABEL = {k: re.compile(rf"(?<!\w)(?i:{label})\s*[:：]\s*(?:{value})") for k, (label, value) in INLINE_LABELLED.items()}
# A gender value on a short "label: value" line, even when OCR noise broke the label ("%%nd~r: Female").
GENDER_VALUE = re.compile(r"^[^\n:：]{0,20}[:：]\s*(?:male|female|nam|nữ)\s*$", re.I | re.M)
# Words that make a header line a job title or a heading, not a person's name.
FLAT_STOP = {"email", "e-mail", "phone", "mobile", "điện", "sđt", "address", "địa", "gender", "giới", "ngày", "date"}
# Capitalised words of job titles ("Java Software Engineer", "Full Stack Web Developer"): never part of a person's name.
TITLE_WORDS = {"software", "quality", "architect", "stack", "full", "front", "back", "end", "web", "middle", "manual",
               "principal", "staff", "chief", "head", "devops", "continuous", "integration", "delivery", "data", "cloud",
               "security", "systems", "system", "product", "project", "business", "analyst", "mobile", "game", "embedded",
               "java", "python", "javascript", "typescript", "c#", ".net", "go", "golang", "php", "ruby", "node", "react",
               "frontend", "backend", "fullstack", "performance", "assurance", "control", "technical", "solution"}
NOT_A_NAME = re.compile(
    r"\b(?:qa|qc|tester|test|testing|engineer|automation|developer|senior|junior|lead|manager|sdet|intern|cv|resume|"
    r"curriculum|vitae|profile|summary|experience|skills|kỹ sư|kiểm thử|chuyên viên|lập trình|hồ sơ)\b", re.I)


def header_name(text: str) -> str | None:
    """The candidate's name when it heads the CV unlabelled (the most common layout): one of the first 3 non-empty
    lines with 2-5 capitalised tokens, no digits, no contact punctuation and no job or heading words."""
    lines = [ln.strip().lstrip("#").strip().strip("*").strip() for ln in text.split("\n") if ln.strip()]
    for line in lines[:3]:
        tokens = line.split()
        if (2 <= len(tokens) <= 5 and all(t[0].isupper() for t in tokens) and not re.search(r"[\d:|@/,()]", line)
                and not NOT_A_NAME.search(line) and heading_kind(line) is None):
            return line
    return flattened_name(lines[0]) if lines else None


def flattened_name(line: str) -> str | None:
    """A header that lost its line breaks ("Bùi Thu Thảo Automation Test Engineer Email: …"): the leading capitalised
    words, when 2-5 of them are followed directly by a job or contact-label word. Anything else ("Professional
    SummaryHighly motivated …") is not taken for a name."""
    tokens = line.split()
    if len(tokens) < 3:
        return None
    for i, token in enumerate(tokens[:6]):
        if NOT_A_NAME.search(token) or token.rstrip(":").lower() in FLAT_STOP:
            name = tokens[:i]
            # all-caps first lines are job titles in real CVs ("CUSTOMER SERVICE MANAGER"), not names
            return " ".join(name) if 2 <= i <= 5 and not all(t.isupper() for t in name) else None
        glued = any(a.islower() and b.isupper() for a, b in zip(token, token[1:]))  # "SummaryHighly"
        if (not token[0].isupper() or len(token) < 2 or glued or re.search(r"[\d:|@/,()]", token)
                or token.lower() in TITLE_WORDS):
            return None
    return None


def _kind(label: str) -> str:
    return next(k for k, v in LABELLED.items() if re.fullmatch(v, label.strip(), re.I))


def loose_kind(label: str) -> str | None:
    f = fold(label.strip())
    best = max(FOLDED_LABELS, key=lambda a: SequenceMatcher(None, f, a).ratio())
    return FOLDED_LABELS[best] if SequenceMatcher(None, f, best).ratio() >= LOOSE_LABEL else None


def redact(text: str) -> tuple[str, dict[str, int]]:
    """→ (text with [REDACTED:<kind>] placeholders, {kind: count}). Never stores what it removed.
    Prompt injection ("ignore previous instructions...") is removed here too, as kind "instruction" (nlp/injection.py)."""
    counts: dict[str, int] = {}
    text, k = sanitize(text, "CV")
    if k:
        counts["instruction"] = k

    def labelled(m: re.Match) -> str:
        kind = _kind(m["label"])
        counts[kind] = counts.get(kind, 0) + 1
        return f"[REDACTED:{kind}]"

    text = LABEL_LINE.sub(labelled, text)

    def loose(m: re.Match) -> str:
        if not (kind := loose_kind(m["label"])):
            return m[0]
        counts[kind] = counts.get(kind, 0) + 1
        return f"[REDACTED:{kind}]"

    text = LOOSE_LINE.sub(loose, text)
    for kind, rx in INLINE_LABEL.items():
        text, k = rx.subn(f"[REDACTED:{kind}]", text)
        if k:
            counts[kind] = counts.get(kind, 0) + k
    if name := header_name(text):
        text, k = re.subn(re.escape(name), "[REDACTED:name]", text, flags=re.I)
        counts["name"] = counts.get("name", 0) + k
    text, k = GENDER_VALUE.subn("[REDACTED:gender]", text)
    if k:
        counts["gender"] = counts.get("gender", 0) + k
    for kind, rx in INLINE.items():
        def inline(m: re.Match, kind: str = kind) -> str:
            digits = re.sub(r"\D", "", m[0])
            if kind == "phone" and not 9 <= len(digits) <= 13:
                return m[0]  # a date range or "3,200 cases", not a phone number
            counts[kind] = counts.get(kind, 0) + 1
            return f"[REDACTED:{kind}]"

        text = rx.sub(inline, text)
    return text, counts
