"""JD parsing without an LLM. The line logic (section headers and cue words decide must vs nice) is ported from the
`coding` branch (Đán, cv_agent/criteria/jd_parser.py); each must line is then read by `canonicalize` for "one of"
groups on our taxonomy. The LLM extraction stays as a cross-check for title, ambiguities and unstructured JDs."""

import re

from app.nlp.extract import jd_depth
from app.nlp.taxonomy import aliases, group, mentions, skills

MUST_HEADER = re.compile(r"^\W*(?:requirements?|qualifications?|must[-\s]have|yêu\s*cầu(?:\s*công\s*việc)?|"
                         r"yeu\s*cau(?:\s*cong\s*viec)?|"
                         r"what\s+you\s+(?:need|bring)|job\s+requirements?)\s*:?\s*$", re.I)
NICE_HEADER = re.compile(r"^\W*(?:nice[-\s]to[-\s]have|preferred(?:\s+qualifications?)?|bonus|plus|ưu\s*tiên|"
                         r"uu\s*tien|điểm\s*cộng|diem\s*cong|is\s+a\s+plus)\s*:?\s*$", re.I)
OTHER_HEADER = re.compile(r"^\W*(?:responsibilit(?:y|ies)|job\s+description|about\s+(?:the\s+role|us)|"
                          r"what\s+you\s+will\s+do|mô\s*tả\s*công\s*việc|mo\s*ta\s*cong\s*viec|trách\s*nhiệm(?:\s*công\s*việc)?|"
                          r"trach\s*nhiem(?:\s*cong\s*viec)?|giới\s*thiệu(?:\s*vị\s*trí)?|gioi\s*thieu(?:\s*vi\s*tri)?)\s*:?\s*$", re.I)
BENEFIT_HEADER = re.compile(r"^\W*(?:benefits?|perks|why\s+join(?:\s+us)?|we\s+offer|quyền\s*lợi|quyen\s*loi|phúc\s*lợi|phuc\s*loi)\s*:?\s*$",
                            re.I)
MUST_CUE = re.compile(r"\b(?:must|required|mandatory|at\s+least|minimum|bắt\s*buộc|bat\s*buoc|tối\s*thiểu|toi\s*thieu|cần\s*có|can\s*co)\b", re.I)
NICE_CUE = re.compile(r"nice\s+to\s+have|is\s+a\s+plus|\ba\s+plus\b|preferred|advantage|bonus|ưu\s*tiên|uu\s*tien|lợi\s*thế|loi\s*the|"
                      r"điểm\s*cộng|diem\s*cong", re.I)
YEARS = re.compile(r"(\d{1,2})(?:\s*(?:-|–|to|đến|den)\s*\d{1,2})?\s*\+?\s*(?:years?|yrs?|năm|nam)\b", re.I)  # \b: "3 names", "namespaces"
TEST_TYPES = {"web": r"\bweb\b|\bui\b|browser|end-to-end|\be2e\b", "mobile": r"mobile|android|\bios\b",
              "api": r"\bapi\b|\brest\b|graphql", "performance": r"performance|\bload\b|stress|hiệu năng"}
FOREIGN = {"english": r"\benglish\b|tiếng anh|tieng anh|toeic|ielts|toefl",
           "japanese": r"japanese|tiếng nhật|tieng nhat|jlpt|nihongo"}
DOMAINS = {"banking": r"\bbank|fintech|payment|core banking|ngân hàng|thanh toán|tài chính",
           "ecommerce": r"e-?commerce|online retail|marketplace|thương mại điện tử",
           "telecom": r"telecom|viễn thông", "healthcare": r"health|medical|y tế", "insurance": r"insurance|bảo hiểm"}

AND = re.compile(r"\s+(?:and|và|va)\s+|;\s*")
OR = re.compile(r"\s(?:or|hoặc|hoac)\s|/")  # "hoac": unaccented Vietnamese
SOFT = re.compile(r"\b(?:or similar|or equivalent|e\.g\.|such as|a plus)\b|tương tự|tương đương|tuong tu|tuong duong")


def canonicalize(phrases: list[str]) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Map free-text requirement phrases onto taxonomy keys → (hard, soft, substitutes). Each and-clause is read on
    its own, in English and Vietnamese, so one phrase can hold several "one of" groups.

    "playwright and typescript"                          → hard [playwright, typescript]
    "selenium or cypress"                                → hard [selenium], substitutes {selenium: [cypress]}
    "github actions or similar" / "jenkins … tương tự"   → soft [github_actions]  (never a knock-out)
    "(java, typescript or python) and ci (gitlab ci or jenkins)" → hard [java, gitlab_ci], two substitute groups
    """
    keys = list(skills())
    hard, soft, subs = [], [], {}
    for clause in (c for phrase in phrases for c in AND.split(phrase.lower())):
        found = [k for k in keys if mentions(clause, k)]
        found.sort(key=lambda k: min(clause.find(a) for a in aliases(k) if a in clause))
        if not found:
            continue
        if SOFT.search(clause):
            soft += found
        elif OR.search(clause):
            hard.append(found[0])
            subs[found[0]] = [*subs.get(found[0], []), *found[1:]]
        else:
            hard += found
    dedup = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return dedup(hard), [s for s in dedup(soft) if s not in hard], {m: dedup(v) for m, v in subs.items()}


def classify(text: str) -> list[tuple[str, str]]:
    """→ [(must | nice | other, line)]. A cue word on the line beats the section it sits in. Benefits are dropped:
    "health insurance" there says nothing about the job."""
    mode, out = "other", []
    for line in text.splitlines():
        if MUST_HEADER.match(line):
            mode = "must"
        elif NICE_HEADER.match(line):
            mode = "nice"
        elif OTHER_HEADER.match(line):
            mode = "other"
        elif BENEFIT_HEADER.match(line):
            mode = "benefits"
        elif line.strip() and mode != "benefits":
            out.append(("nice" if NICE_CUE.search(line) else "must" if MUST_CUE.search(line) else mode, line.strip()))
    return out


def parse(text: str) -> dict:
    """JD text → requirements on taxonomy keys, the same shape the LLM extraction produces (plus foreign_languages
    and domains). Responsibilities count as nice-to-have signal, never as knock-outs."""
    lines = classify(text)
    must, subs, nice, years = [], {}, [], None
    for kind, line in lines:
        if kind == "must":
            hard, soft, alts = canonicalize([line])
            must += [h for h in hard if h not in must]
            for k, v in alts.items():
                subs[k] = list(dict.fromkeys([*subs.get(k, []), *v]))
            nice += soft
            if years is None and (m := YEARS.search(line)):
                years = int(m.group(1))
        else:
            hard, soft, alts = canonicalize([line])
            nice += [*hard, *soft, *(s for v in alts.values() for s in v)]
    alts_all = {s for v in subs.values() for s in v}
    nice = [k for k in dict.fromkeys(nice) if k not in must and k not in alts_all]
    req_text = "\n".join(line for kind, line in lines if kind != "other")
    duty_text = "\n".join(line for kind, line in lines if kind != "nice" and line.startswith("-")).lower()
    return {
        "title": next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")[:120],
        "must_have": must, "substitutes": subs, "nice_to_have": nice, "min_years": years,
        "depth_level": jd_depth(text)[0] or None,
        "languages": [k for k in dict.fromkeys([*must, *alts_all, *nice]) if group(k) == "language"],
        "test_types": [t for t, rx in TEST_TYPES.items() if re.search(rx, duty_text)],
        "foreign_languages": [k for k, rx in FOREIGN.items() if re.search(rx, req_text.lower())],
        "domains": [k for k, rx in DOMAINS.items() if re.search(rx, "\n".join(ln for _, ln in lines).lower())],
    }
