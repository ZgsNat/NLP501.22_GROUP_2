"""PII masking (FR-02, H5) — runs before ANY encoding or LLM call.

Rule-based so it works offline and is unit-testable; covers name, email,
phone (incl. Vietnamese mobile & landline formats), address, date of birth /
age, gender and photo references. Microsoft Presidio can be plugged in via
``extra_maskers``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from cv_agent.ingest.taxonomy import ALIAS_INDEX, SKILLS

Masker = Callable[[str], Tuple[str, int]]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# VN numbers: (+84) / +84 / 84 / 0, then 8-10 digits with optional separators (mobile 09x..., landline 024 ...)
VN_PHONE_RE = re.compile(r"(?<![\w+])\(?(?:\+?84|0)\)?[\s.\-]?\(?\d{1,3}\)?(?:[\s.\-]?\d){6,8}(?!\d)")
GENERIC_PHONE_RE = re.compile(r"(?<!\w)\(?\+\d{1,3}\)?[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){2,4}\d{2,4}(?!\d)")
URL_PROFILE_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|facebook\.com|fb\.com)/\S+", re.I)

LABELLED = {
    "NAME": r"(?:full\s*name|name|họ\s*(?:và)?\s*tên|tên)",
    "ADDRESS": r"(?:address|địa\s*chỉ|nơi\s*ở|location)",
    "DOB": r"(?:date\s*of\s*birth|dob|birthday|ngày\s*sinh|năm\s*sinh)",
    "AGE": r"(?:age|tuổi)",
    "GENDER": r"(?:gender|sex|giới\s*tính)",
    "MARITAL": r"(?:marital\s*status|tình\s*trạng\s*hôn\s*nhân)",
}
GENDER_WORD_RE = re.compile(r"(?<!\w)(?:male|female|nam|nữ)(?!\w)", re.I)
AGE_INLINE_RE = re.compile(r"\b\d{2}\s*(?:years?\s*old|tuổi)\b", re.I)
FULL_DATE_RE = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-](?:19|20)\d{2}\b")
# a full date next to one of these is part of an employment range and must be kept
RANGE_SEP_RE = re.compile(r"^\s*(?:[-–—~→]|to\b|đến\b|until\b)", re.I)
RANGE_SEP_BEFORE_RE = re.compile(r"(?:[-–—~→]|\bto|\bđến|\bfrom|\btừ)\s*$", re.I)
PHOTO_RE = re.compile(r"\[(?:photo|image|ảnh)[^\]]*\]|<img[^>]*>", re.I)
# Street-level address (house number + street/ward/district). City names alone are
# kept: they are not identifying and job locations often mention them.
ADDRESS_HINT_RE = re.compile(
    r"^.*\b\d+[A-Za-z]?\s*(?:/\d+\s*)?,?\s*(?:[\w\s]{0,30})\b(?:district|quận|phường|ward|street|st\.|đường|ngõ|hẻm)\b.*$",
    re.I | re.M,
)
_NAME_WORD = r"[A-ZÀ-Ỹ][a-zà-ỹ]+"
NAME_LINE_RE = re.compile(rf"^\s*((?:{_NAME_WORD}|[A-ZÀ-Ỹ]{{2,}})(?:\s+(?:{_NAME_WORD}|[A-ZÀ-Ỹ]{{2,}})){{1,4}})\s*$")
_NOT_NAME = {"curriculum", "vitae", "resume", "summary", "profile", "objective", "experience", "education",
             "skills", "automation", "test", "tester", "testing", "engineer", "developer", "qa", "qc", "senior",
             "junior", "contact", "lead", "manager", "software", "quality", "assurance"}
# Name tokens that are also everyday words ("Tiếng Anh", "Việt Nam", "Đức" = Germany):
# only the full name is masked for these, never the lone token.
_COMMON_TOKENS = {"anh", "nam", "đức", "việt", "nhật", "hàn", "trung", "mỹ", "pháp", "an", "bình", "hà", "sơn",
                  "long", "hải", "minh", "quang", "phú", "thành", "tâm", "thiện", "kim", "mai"}
_SKILL_WORDS = {w for s in SKILLS.values() for a in s.aliases for w in a.lower().split()}


@dataclass
class MaskResult:
    text: str
    counts: Dict[str, int] = field(default_factory=dict)
    names: List[str] = field(default_factory=list)


def _labelled_line(label_re: str, tag: str, text: str) -> Tuple[str, int, List[str]]:
    # [ \t]* (not \s*) so an empty value never swallows the following line
    pat = re.compile(rf"(?im)^([ \t]*[-•*]?[ \t]*{label_re}[ \t]*[:：\-][ \t]*)([^\n]*\S[^\n]*)$")
    found = [m.group(2).strip() for m in pat.finditer(text)]
    new, n = pat.subn(lambda m: f"{m.group(1)}[{tag}]", text)
    return new, n, found


def _looks_like_name(candidate: str) -> bool:
    words = {w.lower() for w in candidate.split()}
    if words & _NOT_NAME:
        return False
    # reject lines that mention a skill ("Selenium Developer"), matching whole aliases only
    return not any(pattern.search(candidate) for pattern, _skill, _alias in ALIAS_INDEX)


def _detect_header_name(text: str) -> List[str]:
    """Name is usually one of the first non-empty lines of a CV."""
    for line in [ln for ln in text.splitlines() if ln.strip()][:3]:
        m = NAME_LINE_RE.match(line)
        if m and _looks_like_name(m.group(1)):
            return [m.group(1).strip()]
    return []


def _mask_birth_dates(text: str) -> Tuple[str, int]:
    """Mask full dates unless they belong to an employment range (e.g. 15/03/2020 — 20/05/2022)."""
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        before, after = text[max(0, m.start() - 8):m.start()], text[m.end():m.end() + 8]
        if RANGE_SEP_RE.match(after) or RANGE_SEP_BEFORE_RE.search(before):
            return m.group(0)
        count += 1
        return "[DATE]"

    return FULL_DATE_RE.sub(repl, text), count


def _mask_name(out: str, name: str) -> Tuple[str, int]:
    out, total = re.subn(re.escape(name), "[NAME]", out, flags=re.I)
    for token in name.split():
        low = token.lower()
        if len(token) > 2 and low not in _NOT_NAME and low not in _COMMON_TOKENS and low not in _SKILL_WORDS:
            out, n = re.subn(rf"(?<!\w){re.escape(token)}(?!\w)", "[NAME]", out)
            total += n
    return out, total


def mask_contacts(text: str) -> str:
    """Light masking for JDs: recruiter emails / phones / profile URLs only."""
    for regex, tag in ((EMAIL_RE, "EMAIL"), (URL_PROFILE_RE, "PROFILE_URL"), (VN_PHONE_RE, "PHONE"),
                       (GENERIC_PHONE_RE, "PHONE")):
        text = regex.sub(f"[{tag}]", text)
    return text


def mask_pii(text: str, extra_maskers: Tuple[Masker, ...] = ()) -> MaskResult:
    counts: Dict[str, int] = {}

    def bump(tag: str, n: int) -> None:
        if n:
            counts[tag] = counts.get(tag, 0) + n

    text = unicodedata.normalize("NFC", text)
    names = _detect_header_name(text)
    out = text
    for tag, label in LABELLED.items():
        out, n, found = _labelled_line(label, tag, out)
        bump(tag, n)
        if tag == "NAME":
            names.extend(f for f in found if _looks_like_name(f))

    for regex, tag in ((EMAIL_RE, "EMAIL"), (URL_PROFILE_RE, "PROFILE_URL"), (VN_PHONE_RE, "PHONE"),
                       (GENERIC_PHONE_RE, "PHONE"), (PHOTO_RE, "PHOTO"), (AGE_INLINE_RE, "AGE")):
        out, n = regex.subn(f"[{tag}]", out)
        bump(tag, n)
    out, n = _mask_birth_dates(out)
    bump("DATE", n)

    out, n = ADDRESS_HINT_RE.subn("[ADDRESS]", out)
    bump("ADDRESS", n)

    for name in {nm for nm in names if nm and not nm.startswith("[")}:
        out, n = _mask_name(out, name)
        bump("NAME", n)

    # standalone gender lines such as "Male" / "Nữ" (labelled ones are handled above)
    out, n = re.subn(rf"(?im)^\s*{GENDER_WORD_RE.pattern}\s*$", "[GENDER]", out)
    bump("GENDER", n)

    for masker in extra_maskers:
        out, n = masker(out)
        bump("EXTRA", n)

    return MaskResult(text=out, counts=counts, names=names)


def contains_pii(text: str) -> bool:
    """Cheap guard used before any LLM / API call (NFR-02)."""
    return bool(EMAIL_RE.search(text) or VN_PHONE_RE.search(text) or URL_PROFILE_RE.search(text))
