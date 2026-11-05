"""E23: the NLP pipeline on real résumé PDFs (Kaggle "Resume Dataset": 2,484 PDFs, 24 job categories, English).

    cd backend && uv run python -m app.eval.kaggle_resume fetch PATH/TO/cv2.zip   → .runtime/bench/kaggle_resume
    cd backend && uv run python -m app.eval.kaggle_resume                         → docs/KAGGLE_RESUME_RESULTS.md

The deck lists this set for "checking the CV parsing step" (slide 14). It has no JD and no relevance label, so it
measures what needs none, on real PDFs instead of synthetic Markdown:
  - reading: does a PDF give text, and how close is it to the dataset's own text of the same résumé (token F1);
  - structure: sections, dated roles and skills found;
  - safety: personal data redacted, instruction-like sentences flagged, quality flags raised;
  - false positives: testing tools "found" in CVs of other trades (chefs, lawyers, ...), and how many of those would pass
    an Automation Test knock-out. There the right answer is close to zero.
The licence is not stated with the data, so it stays in .runtime and only aggregates are reported.
"""

import csv
import io
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from app.agent.scoring import knockout_reason
from app.config import ROOT, get_settings
from app.nlp import quality
from app.nlp.pipeline import CVProfile, process
from app.nlp.taxonomy import group
from app.nlp.text import read_file

DIR = get_settings().runtime_dir / "bench" / "kaggle_resume"
DOC = ROOT / "docs" / "KAGGLE_RESUME_RESULTS.md"
TESTING_CATEGORIES = {"INFORMATION-TECHNOLOGY", "ENGINEERING"}  # where test-automation tools can legitimately appear
# An Automation Test JD's typical knock-out: one web automation tool and one language
KNOCKOUT_PROBES = {"selenium + java": (["selenium", "java"], {}), "playwright / cypress": (["playwright"], {"playwright": ["cypress"]})}
TOKEN = re.compile(r"[a-z0-9]+")


def fetch(archive: Path) -> None:
    """Copy the PDFs and the CSV out of the zip into their own folder (untrusted data: read, never executed)."""
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if name.endswith((".pdf", ".csv")) and ".." not in name:
                out = DIR / Path(name).relative_to(Path(name).parts[0])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(z.read(name))
    print(f"kaggle resume → {DIR} ({len(list(DIR.rglob('*.pdf')))} PDFs)")


def token_f1(a: str, b: str) -> float:
    """Bag-of-tokens F1 between two texts: how much of the reference text the PDF reader recovered."""
    ca, cb = Counter(TOKEN.findall(a.lower())), Counter(TOKEN.findall(b.lower()))
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p, r = common / sum(ca.values()), common / sum(cb.values())
    return 2 * p * r / (p + r)


TEST_TOOL_GROUPS = {"web_e2e", "mobile", "api", "performance", "test_framework", "bdd", "desktop"}


def automation_tools(p: CVProfile) -> list[str]:
    """Test-automation tools (not languages, CI or trackers) the parser credits as used or listed; hedged ones do not count."""
    return sorted(k for k, ms in p.skills.items() if group(k) in TEST_TOOL_GROUPS and any(m.strength != "hedged" for m in ms))


def reference_texts() -> dict[str, str]:
    path = next(DIR.rglob("Resume.csv"), None)
    if not path:
        return {}
    csv.field_size_limit(10**8)
    return {r["ID"]: r["Resume_str"] for r in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8", errors="replace")))}


def measure() -> list[dict]:
    ref, rows = reference_texts(), []
    for pdf in sorted(DIR.rglob("*.pdf")):
        row = {"id": pdf.stem, "category": pdf.parent.name}
        try:
            raw = read_file(pdf.name, pdf.read_bytes())
        except Exception as e:  # an unreadable PDF is a result, not a crash
            rows.append({**row, "error": type(e).__name__})
            continue
        p = process(f"KR-{pdf.stem}", raw)
        row.update(words=len(p.text.split()), f1=token_f1(raw, ref[pdf.stem]) if pdf.stem in ref else None,
                   sections=len(p.sections), has_experience=any(s.kind == "experience" for s in p.sections),
                   dated=p.years_total is not None, skills=len(p.skills), tools=automation_tools(p),
                   redacted=sum(p.redactions.values()), flags=quality.flags(p),
                   passes={name: knockout_reason(p, must, subs) is None for name, (must, subs) in KNOCKOUT_PROBES.items()})
        rows.append(row)
    return rows


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.1f}%" if d else "—"


def report(rows: list[dict]) -> str:
    ok = [r for r in rows if "error" not in r]
    f1s = sorted(r["f1"] for r in ok if r["f1"] is not None)
    median = f1s[len(f1s) // 2] if f1s else float("nan")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        by_cat[r["category"]].append(r)
    other = [r for r in ok if r["category"] not in TESTING_CATEGORIES]
    flag_counts = Counter(f.split(":")[0] for r in ok for f in r["flags"])
    lines = ["# NLP pipeline on real résumé PDFs (generated by `python -m app.eval.kaggle_resume`; do not edit by hand)", "",
             f"Kaggle Resume Dataset: {len(rows)} PDFs in {len(by_cat)} job categories, English, no JD and no relevance "
             "label (licence not stated: data kept in .runtime, aggregates only). The pipeline is the one the app runs "
             "on an upload: read_file (pypdf) → normalize → redact → sections/dates → chunks → skills.", "",
             "## Reading and structure", "", "| Measure | Value |", "|---|---|",
             f"| PDFs read without error | {len(ok)}/{len(rows)} ({pct(len(ok), len(rows))}) |",
             f"| Text vs the dataset's own text of the same résumé (token F1, median) | {median:.3f} |",
             f"| ... CVs with token F1 ≥ 0.9 | {pct(sum(f >= 0.9 for f in f1s), len(f1s))} |",
             f"| An experience section found | {pct(sum(r['has_experience'] for r in ok), len(ok))} |",
             f"| Dated roles parsed (years computed) | {pct(sum(r['dated'] for r in ok), len(ok))} |",
             f"| At least one known skill | {pct(sum(r['skills'] > 0 for r in ok), len(ok))} |",
             f"| CVs with personal data redacted | {pct(sum(r['redacted'] > 0 for r in ok), len(ok))} |",
             f"| Median words after processing | {sorted(r['words'] for r in ok)[len(ok) // 2] if ok else 0} |", "",
             "Quality flags raised: " + (", ".join(f"{k} {v}" for k, v in flag_counts.most_common()) or "none"), "",
             "## False positives: testing tools in other trades", "",
             f"Categories where automation tools can legitimately appear: {', '.join(sorted(TESTING_CATEGORIES))}. In the "
             f"other {len(other)} CVs, a detected test-automation tool is most likely a false positive, and passing the "
             "knock-out of an Automation Test JD is the failure that would matter to HR.", "",
             "| Group | CVs | ≥ 1 testing tool detected | passes `selenium + java` | passes `playwright / cypress` |",
             "|---|---|---|---|---|"]
    for name, group in (("IT + engineering", [r for r in ok if r["category"] in TESTING_CATEGORIES]), ("all other trades", other)):
        lines.append(f"| {name} | {len(group)} | {pct(sum(bool(r['tools']) for r in group), len(group))} | "
                     + " | ".join(pct(sum(r['passes'][k] for r in group), len(group)) for k in KNOCKOUT_PROBES) + " |")
    tool_hits = Counter(t for r in other for t in r["tools"])
    lines += ["", "Tools detected in the other trades (what to check for gazetteer false positives): "
              + (", ".join(f"{t} ×{n}" for t, n in tool_hits.most_common(12)) or "none"), "",
              "## Per category", "", "| Category | CVs | read | dated roles | ≥ 1 testing tool | median token F1 |",
              "|---|---|---|---|---|---|"]
    for cat, group in sorted(by_cat.items()):
        cf = sorted(r["f1"] for r in group if r["f1"] is not None)
        lines.append(f"| {cat} | {len(group)} | {len(group)} | {pct(sum(r['dated'] for r in group), len(group))} | "
                     f"{pct(sum(bool(r['tools']) for r in group), len(group))} | {cf[len(cf) // 2] if cf else float('nan'):.3f} |")
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(DOC)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(Path(sys.argv[2]))
    else:
        print(report(measure()))
