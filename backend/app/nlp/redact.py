"""Stage 3: remove contact details and protected attributes BEFORE anything is indexed or shown to an LLM.

The fairness policy (POL-ALL-fairness) says the ranking must never use name, age, gender, marital status, nationality,
religion, health, photo or address. Asking the LLM to "ignore" them is a hope; deleting them is a guarantee. Labels are
matched in English and Vietnamese because real Vietnamese CVs put "Ngày sinh / Giới tính" in the header."""

import re

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
INLINE = {
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "url": re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github|facebook)\.com/\S+", re.I),
    # Starts with "+<country>" or a trunk "0" (VN: 09xx…), so year ranges like "2019.05-2021" never match.
    "phone": re.compile(r"(?<![\w/.])(?:\+\d{1,3}[\s.-]?|0)\d{1,4}(?:[\s.-]?\d{2,4}){2,4}(?![\w/])"),
}


def _kind(label: str) -> str:
    return next(k for k, v in LABELLED.items() if re.fullmatch(v, label.strip(), re.I))


def redact(text: str) -> tuple[str, dict[str, int]]:
    """→ (text with [REDACTED:<kind>] placeholders, {kind: count}). Never stores what it removed."""
    counts: dict[str, int] = {}

    def labelled(m: re.Match) -> str:
        kind = _kind(m["label"])
        counts[kind] = counts.get(kind, 0) + 1
        return f"[REDACTED:{kind}]"

    text = LABEL_LINE.sub(labelled, text)
    for kind, rx in INLINE.items():
        def inline(m: re.Match, kind: str = kind) -> str:
            digits = re.sub(r"\D", "", m[0])
            if kind == "phone" and not 9 <= len(digits) <= 13:
                return m[0]  # a date range or "3,200 cases", not a phone number
            counts[kind] = counts.get(kind, 0) + 1
            return f"[REDACTED:{kind}]"

        text = rx.sub(inline, text)
    return text, counts
