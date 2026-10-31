"""Stage 6: deterministic information extraction from the structured CV.

- Skill mentions (gazetteer NER) with an *evidence strength*:
    used    in an experience/project entry → demonstrated
    listed  in skills/summary/education    → claimed
    hedged  under a hedge or negation cue  → "familiar with", "currently learning", "(basic)", "chưa có kinh nghiệm"
  Hedge scope follows NegEx: a cue before the term reaches to the end of its sentence (or to "but/however"),
  a cue right after the term applies only to its comma segment ("Java (basic), Selenium" hedges Java only).
- Years from parsed role dates, with overlapping roles merged.
- An automation-depth cue (L1–L4) from action-verb patterns in experience lines only, so a job title like
  "Automation Architect" over L1 bullets does not count."""

import re
from dataclasses import dataclass

from app.nlp.chunking import Chunk
from app.nlp.segment import Entry, merged_months
from app.nlp.taxonomy import alias_regex, automation_evidence, skills

CREDIT = {"used": 1.0, "listed": 0.7, "implied": 0.5, "hedged": 0.2}  # implied: described, not named (N12 verifies)

PRE_HEDGE = re.compile(
    r"\b(?:familiar(?:ity)? with|basic (?:knowledge|understanding) of|some (?:knowledge|exposure)|exposure to|"
    r"exposed to|introduction to|currently learning|learning|self-stud(?:y|ying)|studying|interested in|"
    r"beginner|no hands-on|not hands-on|no (?:production|real|practical|professional) experience|never used|"
    r"theoretical|training (?:on|in)|course (?:on|in)|tutorial|demo(?:ed|nstrated)? .{0,20}to (?:us|me)|"
    r"proof of concept|"
    r"(?:our|the|other) (?:developers|devs|dev team)|a colleague|colleagues|"  # someone else did the work
    r"tìm hiểu|đang học|tự học|biết cơ bản|làm quen|chưa có kinh nghiệm|chưa (?:sử dụng|làm|thực hành)|"
    r"kiến thức cơ bản|khó?a (?:học|đào tạo)|khoá (?:học|đào tạo))"
)
# After the term, within its clause: "(basic)", "- beginner", or a strong cue anywhere ("Cypress: following a
# beginner course"). Plain "basic" further away is not a hedge ("Selenium tests covering basic flows").
POST_HEDGE = re.compile(
    r"^\s*(?:\([^)]*\b(?:basic|beginner|learning|familiar|not used|no production|training|course|theory|cơ bản|"
    r"đang học|tìm hiểu)\b|[-–:]\s*(?:basic|beginner|learning)\b|(?:basic|beginner)\b)"
    r"|\b(?:beginner|course|tutorial|not used|no hands-on|no (?:production|project)|currently learning|"
    r"chưa (?:dùng|sử dụng|tham gia|thực hành)|khóa học|đang học)"
)
SCOPE_BREAK = re.compile(r"\b(?:but|however|whereas|nhưng|tuy nhiên)\b")
AUTOMATION = re.compile(r"automat|\bscripts?\b|framework|tự động")

# Depth cues are matched only at the START of a clause, i.e. on the candidate's own action verb. That is the
# difference between "Executed the existing automated suite" (L1: "automated" is an adjective) and "Automated 80
# cases" (L2), or between "Our developers run Cypress tests" (not the candidate) and "Ran the Cypress suite" (L1).
_OBJ = r"[^.;]{0,90}?"
DEPTH_CUES = {  # checked from L4 down; the first level with a match wins
    4: rf"(?:led|lead|own|defin|drove|driv|design|buil|creat|architect|set up|introduc|establish|roll|xây dựng|thiết kế)"
       rf"\w*\b{_OBJ}(?:(?:test|testing|qa|quality|automation)\s+(?:strategy|architecture|platform|roadmap|guild)|"
       rf"(?:across|for|by|to)\s+(?:\d+|two|three|four|five|six|multiple|several|all|other)\s+(?:\w+\s+)?"
       rf"(?:teams|squads|products|tribes)|quality gates?|release gates?|chiến lược kiểm thử|kiến trúc kiểm thử)",
    3: rf"(?:design|buil|creat|architect|set up|introduc|develop|implement|xây dựng|thiết kế)\w*\b{_OBJ}"
       rf"(?:framework|harness|page object|base (?:layer|classes)|test platform|shared librar|khung)",
    2: rf"(?:wr(?:ote|ite|itten)|author|automat|develop|implement|creat|buil|add|cover|script|migrat|viết|tự động hóa)"
       rf"\w*\b{_OBJ}\b(?:tests?|scripts?|cases|suites?|checks|scenarios|kịch bản)\b",
    1: rf"(?:execut|ran\b|run|maintain|updat|trigger|monitor|chạy)\w*\b{_OBJ}\b(?:tests?|scripts?|suites?|regression|"
       rf"jobs?|locators|test case)",
}
DEPTH_RX = {lv: re.compile(rx) for lv, rx in DEPTH_CUES.items()}
CLAUSE = re.compile(r"[.;]\s+|,\s+(?:and|then)\s+|\s+and then\s+")
SUBJECT = re.compile(r"^(?:i|we|also|then|and|tôi|đã|personally)\s+")
MANUAL = re.compile(r"\bmanual(?:ly)?\b|thủ công")
GAP = re.compile(r"career break|no employment|not employed|unemployed|sabbatical|\bgap\b|nghỉ việc|thời gian nghỉ")
YEARS_CLAIM = re.compile(r"(\d{1,2}(?:[.,]\d)?)\s*\+?\s*(?:years?|yrs|năm)")


@dataclass
class Mention:
    skill: str
    chunk_id: str
    section: str
    strength: str  # used | listed | hedged
    line: str


def is_hedged(line: str, start: int, end: int) -> bool:
    """Is the term at line[start:end] under a hedge/negation cue? `line` must be lowercase."""
    sent_start = max(line.rfind(c, 0, start) for c in ".;!?") + 1
    before = SCOPE_BREAK.split(line[sent_start:start])[-1]
    if PRE_HEDGE.search(before):
        return True
    seg_end = min([i for i in (line.find(c, end) for c in ",;.") if i != -1] or [len(line)])
    if "(" in line[end:seg_end] and ")" not in line[end:seg_end]:  # "(basic, training only)" spans a comma
        seg_end = line.find(")", end) + 1 or len(line)
    return bool(POST_HEDGE.search(line[end:seg_end]))


def skill_mentions(chunks: list[Chunk]) -> dict[str, list[Mention]]:
    out: dict[str, list[Mention]] = {}
    for ch in chunks:
        for raw in ch.text.split("\n"):
            low = raw.lower()
            for key in skills():
                for m in alias_regex(key).finditer(low):
                    if is_hedged(low, m.start(), m.end()):
                        strength = "hedged"
                    else:
                        strength = "used" if ch.section in ("experience", "projects") else "listed"
                    out.setdefault(key, []).append(
                        Mention(key, ch.id, ch.section, strength, raw.strip().removeprefix("- ").strip()))
                    break  # one mention per line per skill is enough
    return out


def best(mentions: list[Mention]) -> Mention | None:
    return max(mentions, key=lambda m: CREDIT[m.strength], default=None)


def years(text: str, ents: list[Entry], chunks: list[Chunk], mentions: dict[str, list[Mention]]
          ) -> tuple[float | None, float | None, list[str]]:
    """→ (total years, years in roles with automation work, source lines). Dated experience entries first; else the
    largest "N years" claim in the text (marked as a claim). None when there is no signal at all."""
    dated = [e for e in ents if e.section == "experience" and e.start_month is not None and not GAP.search(e.title.lower())]
    if dated:
        at = {c.id: c.start for c in chunks}
        used_in = lambda e: any(  # noqa: E731
            m.strength == "used" and automation_evidence(m.skill) and e.start <= at[m.chunk_id] < e.end
            for ms in mentions.values() for m in ms)
        auto = [e for e in dated if used_in(e) or AUTOMATION.search(text[e.start:e.end].lower())]
        total = merged_months([(e.start_month, e.end_month) for e in dated]) / 12
        automation = merged_months([(e.start_month, e.end_month) for e in auto]) / 12
        return round(total, 1), round(automation, 1), [e.title for e in dated]
    claims = [(float(m[1].replace(",", ".")), m[0]) for m in YEARS_CLAIM.finditer(text.lower())]
    claims = [c for c in claims if c[0] <= 40]
    if not claims:
        return None, None, []
    y, src = max(claims)
    return y, y, [f"claimed: “{src}”"]


def _names_tool(clause: str) -> bool:
    return any(alias_regex(k).search(clause) for k in skills() if automation_evidence(k))


def depth_cue(chunks: list[Chunk]) -> tuple[int, str | None]:
    """Highest automation depth shown by the candidate's own action verbs in experience/project lines (not hedged)."""
    return _deepest([ln for c in chunks if c.section in ("experience", "projects") for ln in c.text.split("\n")])


NICE_HEADING = re.compile(r"^\W*(?:nice to have|preferred|bonus|ưu tiên|điểm cộng)\b", re.I)


def jd_depth(text: str) -> tuple[int, str | None]:
    """The depth a JD's own duties and must-haves ask for, read with the same lexicon as CVs ("Design and evolve the
    shared automation framework" → L3). Nice-to-have lines are skipped: a plus is not a requirement."""
    lines, nice = [], False
    for ln in text.split("\n"):
        nice = bool(NICE_HEADING.match(ln)) or (nice and not ln.startswith("#"))
        if not nice and ln.lstrip().startswith("- "):
            lines.append(ln)
    return _deepest(lines)


def _deepest(raw_lines: list[str]) -> tuple[int, str | None]:
    lines = [ln.strip().removeprefix("- ").strip() for ln in raw_lines]
    clauses = [(ln, SUBJECT.sub("", cl.strip().lower())) for ln in lines if ln and not PRE_HEDGE.search(ln.lower())
               for cl in CLAUSE.split(ln)]
    for level in (4, 3, 2, 1):
        for ln, cl in clauses:
            # L1/L2 verbs ("wrote … test cases") are only automation when the clause is about tools or scripts
            if level <= 2 and (MANUAL.search(cl) or not (AUTOMATION.search(cl) or _names_tool(cl))):
                continue
            if DEPTH_RX[level].match(cl):
                return level, ln
    return 0, None
