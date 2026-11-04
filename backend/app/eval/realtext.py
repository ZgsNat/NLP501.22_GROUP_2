"""Real text: does the NLP pipeline hold up on CVs written by real people? (external validity, docs/FINAL_REPORT.md §7)

Two sources, neither ever committed to git (the repo rule: no real CVs in data/):

    public     the real, anonymised CVs of datasetmaster/resumes (MIT, Hugging Face), recognised by the absence of
               the Faker e-mail domains its synthetic half uses. The CVs are stored as JSON, so we render them to
               text; the language is real, the layout is ours.
    consented  CVs of real automation testers who agreed in writing (docs/labeling/CONSENT.md). `intake` reads
               .runtime/consented/raw/*, redacts, and keeps only the redacted text in .runtime/consented/redacted/.

Because a rendered CV contains its own structured fields, tool *recall* against them would be circular. We measure
what is not: gazetteer coverage of the tool names real people write, precision of the tools found in free prose
(summary, responsibilities, project descriptions), and whether the pipeline parses real content into sections and
dated roles.

    python -m app.eval.realtext public     → docs/REALTEXT_RESULTS.md
    python -m app.eval.realtext intake     → redacted consented CVs + a leak check to review by hand
"""

import json
import re
import sys
from collections import Counter

from app.config import ROOT, get_settings
from app.nlp import quality
from app.nlp.pipeline import process
from app.nlp.redact import INLINE, header_name
from app.nlp.taxonomy import mentions, skills
from app.nlp.text import normalize, read_file

PUBLIC = get_settings().runtime_dir / "bench" / "public" / "master_resumes.jsonl"
CONSENTED = get_settings().runtime_dir / "consented"
DOC = ROOT / "docs" / "REALTEXT_RESULTS.md"
FAKER = re.compile(r"@example\.(?:com|net|org)$")
UNKNOWN = {"", "unknown", "not provided", "n/a", "none"}


def known(x) -> bool:
    return isinstance(x, str) and x.strip().lower() not in UNKNOWN


def _date(x) -> str:
    if not known(x):
        return ""
    m = re.fullmatch(r"(\d{4})-(\d{2})", x.strip())
    return f"{m[2]}/{m[1]}" if m else x.strip().capitalize()


def tool_strings(r: dict) -> list[str]:
    """Every tool name the CV's own structured fields list (skills, technical environment, project technologies)."""
    out = []

    def walk(x, keep: bool) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, keep or k in ("skills", "technical_environment", "technologies", "tools", "frameworks"))
                if keep and k == "name" and known(v):
                    out.append(v)
        elif isinstance(x, list):
            for v in x:
                walk(v, keep)
        elif keep and known(x):
            out.append(x)

    walk({k: v for k, v in r.items() if k in ("skills", "experience", "projects")}, False)
    return [s for s in out if len(s) < 60]


def render(r: dict) -> tuple[str, str]:
    """→ (CV text, the free-prose part of it: summary, responsibilities, project descriptions)."""
    pi, prose, lines = r.get("personal_info") or {}, [], []
    if known(pi.get("summary")):
        lines += ["SUMMARY", pi["summary"], ""]
        prose.append(pi["summary"])
    if r.get("experience"):
        lines.append("EXPERIENCE")
    for e in r.get("experience") or []:
        d = e.get("dates") or {}
        span = " – ".join(x for x in (_date(d.get("start")), _date(d.get("end"))) if x)
        lines.append(f"{e.get('title') or ''} — {e.get('company') or ''}" + (f" ({span})" if span else ""))
        bullets = [x for x in e.get("responsibilities") or [] if known(x)]
        lines += [f"- {x}" for x in bullets]
        prose += bullets
        env = [t for k in ("technologies", "tools") for t in (e.get("technical_environment") or {}).get(k) or [] if known(t)]
        if env:
            lines.append("Environment: " + ", ".join(env))
    if r.get("projects"):
        lines += ["", "PROJECTS"]
    for p in r.get("projects") or []:
        lines.append(f"{p.get('name') or ''}: {p.get('description') or ''}")
        if known(p.get("description")):
            prose.append(p["description"])
    names = [s for s in tool_strings({"skills": r.get("skills")})]
    if names:
        lines += ["", "SKILLS", ", ".join(dict.fromkeys(names))]
    return "\n".join(lines), "\n".join(prose)


def public() -> str:
    rows = [json.loads(ln) for ln in PUBLIC.read_text().splitlines() if ln.strip()]
    real = [r for r in rows if not FAKER.search((r.get("personal_info") or {}).get("email") or "")]
    keys = list(skills())
    listed = Counter()  # tool strings real people wrote → recognised by the gazetteer or not
    unrecognised = Counter()
    found = supported = 0
    unsupported = Counter()
    flags, secs, dated, crashed = Counter(), 0, 0, 0
    for i, r in enumerate(real):
        text, prose = render(r)
        try:
            p = process(f"PUB-{i:03d}", text)
        except Exception:
            crashed += 1
            continue
        names = tool_strings(r)
        for s in set(names):
            hit = any(mentions(s, k) for k in keys)
            listed[hit] += 1
            if not hit:
                unrecognised[s.lower()] += 1
        own = {k for s in names for k in keys if mentions(s, k)}
        in_prose = {k for k in keys if mentions(prose, k)}
        found += len(in_prose)
        supported += len(in_prose & own)
        unsupported.update(in_prose - own)
        flags.update(quality.flags(p))
        secs += any(s.kind in ("experience", "projects") for s in p.sections)
        dated += p.years_total is not None
    testing = [(s, n) for s, n in unrecognised.most_common()
               if re.search(r"test|qa|selen|cucumber|postman|jmeter|loadrunner|uft|qtp|sikuli|autoit|katalon|appium|"
                            r"quality|jira|alm|hp ?qc|bugzilla|soapui|karate|robot", s)]
    n = len(real) - crashed
    lines = ["# Real-text results (generated by `python -m app.eval.realtext public`; do not edit by hand)", "",
             f"Source: the {len(real)} real, anonymised CVs of datasetmaster/resumes (MIT; the other "
             f"{len(rows) - len(real)} are Faker-generated and excluded). Rendered from JSON to text: the language is "
             "real, the layout is ours. Mostly developer, analyst and engineering roles, so this tests the pipeline on "
             "real language, not test-automation ranking.", "",
             "| Check | result |", "|---|---|",
             f"| CVs parsed without error | {n}/{len(real)} |",
             f"| experience or projects section found | {secs}/{n} ({secs / n:.0%}) |",
             f"| dated roles found (years computable) | {dated}/{n} ({dated / n:.0%}) |",
             f"| tool names written by real people that the gazetteer recognises | {listed[True]}/{sum(listed.values())} "
             f"({listed[True] / max(sum(listed.values()), 1):.0%}); the taxonomy covers test automation only |",
             f"| tools found in free prose that the CV's own fields also list (prose precision) | {supported}/{found} "
             f"({supported / max(found, 1):.0%}) |",
             f"| quality flags | " + (", ".join(f"{k} {v}" for k, v in flags.most_common()) or "none") + " |", "",
             "Tools found in prose but absent from the CV's own fields (review: true mention, or gazetteer false "
             "positive?): " + (", ".join(f"{k} ({v})" for k, v in unsupported.most_common(10)) or "none"), "",
             "Testing-related tool names real people wrote that the gazetteer does not know (candidates for "
             "data/skills_taxonomy.yaml): " + (", ".join(f"{s} ({k})" for s, k in testing[:20]) or "none"), ""]
    DOC.write_text("\n".join(lines))
    return DOC.read_text()


def intake() -> str:
    """Redact consented CVs; keep only the redacted text. Prints what a human must still check before use."""
    raw, out = CONSENTED / "raw", CONSENTED / "redacted"
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for f in sorted(p for p in raw.glob("*") if p.is_file()):
        text = normalize(read_file(f.name, f.read_bytes()))
        p = process(f.stem, text)
        (out / f"{f.stem}.txt").write_text(p.text, encoding="utf-8")
        left = [k for k, rx in INLINE.items() if rx.search(p.text)] + (["name?"] if header_name(p.text) else [])
        report.append(f"{f.stem}: masked {p.redactions or 'nothing'}; still matching {left or 'nothing'}; "
                      f"flags {quality.flags(p) or 'none'}")
    return "\n".join(report) or f"No files in {raw}"


if __name__ == "__main__":
    print(intake() if sys.argv[1:2] == ["intake"] else public())
