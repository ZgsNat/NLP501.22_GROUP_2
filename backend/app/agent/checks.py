"""Deterministic checks on what the LLM writes (docs/AI_OUTPUT_CRITERIA.md, Q-ids). The same functions run inside the
graph, where a failed check triggers an action (retry, drop, flag), and in evaluation, where they become the
scorecard. Grounding of quotes (Q2/Q3) lives in scoring.check_evidence; these cover the free-text claims around them.

A check only ever flags or removes LLM text; it never adds facts and never touches the rule scores."""

import re

from app.nlp.pipeline import CVProfile
from app.nlp.taxonomy import aliases, mentions, skills

# Q5: protected attributes and personal circumstances must never appear in a judgement about a candidate.
# Words with common technical meanings ("single sign-on", "race condition", "healthcare") are deliberately left out.
PROTECTED = re.compile(
    r"\b(?:male|female|woman|women|gender|sex|aged?|years?[- ]old|marital|married|unmarried|divorced|pregnan\w*|maternity|"
    r"religio\w*|nationality|ethnic\w*|disabilit\w*|disabled|photo\w*|date of birth|born in|"
    r"giới tính|nam giới|nữ giới|phụ nữ|tuổi|kết hôn|độc thân|ly hôn|mang thai|tôn giáo|dân tộc|quốc tịch|ngày sinh)\b",
    re.I)
# A sentence with one of these cues states an absence or quotes the JD, so naming a tool there is not a claim
# about the candidate ("no Cypress evidence", "the JD asks for Cypress").
NOT_A_CLAIM = re.compile(
    r"\b(?:no|not|without|lacks?|lacking|missing|absent|never|none|gap|unverified|unclear|instead of|rather than|"
    r"jd|job|requires?|required|requirement|asks?|needs?|wants?|expects?|preferred|nice to have|plus|"
    r"chưa|không|thiếu|yêu cầu)\b", re.I)
YEARS = re.compile(r"(\d+(?:[.,]\d+)?)\s*\+?\s*(?:years?|yrs?|năm)\b", re.I)
SENTENCE = re.compile(r"(?<=[.;!?])\s+|\n+")


def _claims(text: str) -> list[str]:
    return [s for s in SENTENCE.split(text or "") if s.strip() and not NOT_A_CLAIM.search(s)]


def protected(text: str) -> list[str]:
    """Q5: protected-attribute words in LLM output."""
    return sorted({m.group(0).lower() for m in PROTECTED.finditer(text or "")})


def unsupported_tools(text: str, p: CVProfile) -> list[str]:
    """Q4a: tools a sentence credits to the candidate that the parsed CV never mentions (entity hallucination)."""
    named = {k for s in _claims(text) for k in skills() if k not in p.skills and mentions(s, k)}
    return sorted(k for k in named if not inflected_in_cv(k, p))


def inflected_in_cv(k: str, p: CVProfile) -> bool:
    """The CV says the skill in another inflection of its alias ("API tests" for "api testing"). The taxonomy and the
    extraction stay as they are (the frozen benchmark depends on them); only Q4 stops calling a true claim invented."""
    forms = [re.sub(r"testing\b", "test(?:s|ing)?", re.escape(a.lower())) for a in aliases(k) if "testing" in a.lower()]
    if not forms:
        return False
    rx = re.compile(rf"(?<![a-z0-9])(?:{'|'.join(forms)})(?![a-z0-9])")
    return any(rx.search(c.text.lower()) for c in p.chunks)


def overstated_years(text: str, p: CVProfile, slack: float = 1.0) -> float | None:
    """Q4b: the largest experience claim above what the dated roles support (+slack), or None. Undated CVs are not
    checked: with no dates there is no fact to contradict."""
    known = max(p.years_total or 0.0, p.years_automation or 0.0)
    if p.years_total is None:
        return None
    said = [float(m.group(1).replace(",", ".")) for s in _claims(text) for m in YEARS.finditer(s)]
    worst = max(said, default=0.0)
    return worst if worst > known + slack else None


def claim_issues(text: str, p: CVProfile) -> list[str]:
    """All Q4/Q5 problems in one piece of LLM text, as short strings for the log, the retry prompt and the UI."""
    out = [f"names {t} but the CV never mentions it" for t in unsupported_tools(text, p)]
    if (y := overstated_years(text, p)) is not None:
        out.append(f"claims {y:g} years but dated roles show {max(p.years_total or 0, p.years_automation or 0):g}")
    out += [f"mentions a protected attribute ({w})" for w in protected(text)]
    return out


def review_issues(review: dict[str, dict], p: CVProfile) -> dict[str, list[str]]:
    """Q4/Q5 per criterion of an llm_review (rationale only: quotes are checked by grounding)."""
    return {k: issues for k, c in review.items() if (issues := claim_issues(c.get("rationale", ""), p))}


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def translation_issues(source: list[str], target: list[str]) -> list[str]:
    """Q18: a translation for HR keeps every item, every number and every tool it names (it may not add a tool)."""
    if len(source) != len(target):
        return [f"{len(target)} items for {len(source)}"]
    out = []
    for s, t in zip(source, target):
        if not t.strip():
            out.append(f"empty translation of {s[:60]!r}")
            continue
        if sorted(NUMBER.findall(s)) != sorted(NUMBER.findall(t)):
            out.append(f"numbers changed in {s[:60]!r}")
        if {k for k in skills() if mentions(s, k)} != {k for k in skills() if mentions(t, k)}:
            out.append(f"tools changed in {s[:60]!r}")
    return out


def strip_protected(text: str) -> str:
    """Q5 action: drop every sentence that mentions a protected attribute."""
    return " ".join(s for s in SENTENCE.split(text or "") if s.strip() and not PROTECTED.search(s))


def report_issues(report: dict, profiles: dict[str, CVProfile]) -> dict[str, list[str]]:
    """Q6: per candidate, claims in the explanation and strengths the CV does not support; plus Q5 in the summary.
    Gaps are not checked for tools: naming a missing tool is what a gap is."""
    out: dict[str, list[str]] = {}
    for c in report.get("candidates", []):
        p = profiles.get(c.get("cv_id"))
        if p is None:
            continue
        text = " ".join([c.get("explanation", ""), *c.get("strengths", [])])
        if issues := claim_issues(text, p) + [f"gap mentions a protected attribute ({w})"
                                               for g in c.get("gaps", []) for w in protected(g)]:
            out[c["cv_id"]] = issues
    if words := protected(report.get("summary", "")):
        out["summary"] = [f"mentions a protected attribute ({w})" for w in words]
    return out


def sanitize_report(report: dict, profiles: dict[str, CVProfile]) -> dict:
    """Q6 action after the retry budget: keep the report, drop only the sentences and strengths that fail."""
    def keep(text: str, p: CVProfile | None) -> bool:
        return not protected(text) and (p is None or not claim_issues(text, p))

    cands = []
    for c in report.get("candidates", []):
        p = profiles.get(c.get("cv_id"))
        cands.append({**c,
                      "explanation": " ".join(s for s in SENTENCE.split(c.get("explanation", "")) if s.strip() and keep(s, p)),
                      "strengths": [s for s in c.get("strengths", []) if keep(s, p)],
                      "gaps": [g for g in c.get("gaps", []) if not protected(g)]})
    return {**report, "summary": strip_protected(report.get("summary", "")), "candidates": cands}
