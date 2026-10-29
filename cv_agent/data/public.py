"""Public datasets as a real-data test bed (D2), in the same layout as ``data/synthetic``.

* Resume–Job Fit (``cnamuangtoun/resume-job-description-fit`` on Hugging Face): real resume / job description
  pairs labelled No Fit / Potential Fit / Good Fit. Only QA / test-automation job descriptions are kept.
  The dataset card states no licence: use it for course research only and do not redistribute it
  (``data/public/`` is git-ignored).
* SkillSpan (``jjzha/skillspan``, CC-BY-4.0): job postings with skill / knowledge spans, used to measure
  span-level F1 of the entity extractor (FR-03), see ``cv_agent.eval.span_metrics``.

Each JD is labelled only against ~25 resumes, so ``gold.json`` sets ``judged_only`` and metrics are computed
on the labelled resumes of each JD (condensed-list evaluation), never treating unlabelled CVs as irrelevant.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

log = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/datasets"
RJF_REPO = "cnamuangtoun/resume-job-description-fit"
RJF_FILES = ("train.csv", "test.csv")
SKILLSPAN_REPO = "jjzha/skillspan"
SKILLSPAN_FILES = ("train.json", "dev.json", "test.json")
LABELS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
# Strong QA signals; a JD is kept when it contains at least MIN_QA_SIGNALS *different* ones. A single loose
# mention ("QA", "test cases") also matches business-analyst or electrical-engineer postings.
QA_SIGNAL_RE = re.compile(
    r"\b(?:test automation|automation test\w*|automated test\w*|QA (?:engineer|analyst|automation|tester)|SDET|"
    r"software engineer in test|selenium|cypress|playwright|appium|cucumber|testng|test cases?|test plans?|"
    r"quality assurance (?:engineer|analyst)|software test\w*|regression test\w*)\b", re.IGNORECASE)
MIN_QA_SIGNALS = 2
DEV_SHARE = 0.3          # share of JDs used to tune thresholds (split=dev), chosen by stable hash
DOWNLOAD_TIMEOUT_S = 120
DEFAULT_PROFILE = "Mặc định"


def download(repo: str, filename: str, raw_dir: Path) -> Path:
    """Download one dataset file once (kept in ``raw_dir``); later calls reuse the local copy."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / f"{repo.replace('/', '__')}__{filename}"
    if target.exists() and target.stat().st_size > 0:
        return target
    url = f"{HF_BASE}/{repo}/resolve/main/{filename}"
    log.info("downloading %s", url)
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp, open(tmp, "wb") as fh:  # noqa: S310 (fixed https host)
        expected = int(resp.headers.get("Content-Length") or 0)
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    size = tmp.stat().st_size
    if size == 0 or (expected and size != expected):
        tmp.unlink()
        raise OSError(f"incomplete download of {url}: {size} of {expected or '?'} bytes")
    tmp.replace(target)
    return target


def _stable_id(prefix: str, text: str) -> str:
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def _title(jd_text: str) -> str:
    first = next((line.strip() for line in jd_text.splitlines() if line.strip()), "Job description")
    return first[:80]


def read_pairs(paths: Iterable[Path]) -> List[Tuple[str, str, int]]:
    """(resume_text, job_description_text, label 0..2) rows; rows with an unknown label are skipped."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                label = LABELS.get((row.get("label") or "").strip())
                resume, jd = (row.get("resume_text") or "").strip(), (row.get("job_description_text") or "").strip()
                if label is not None and resume and jd:
                    rows.append((unicodedata.normalize("NFC", resume), unicodedata.normalize("NFC", jd), label))
    return rows


def qa_signals(jd_text: str) -> List[str]:
    return sorted({m.group(0).lower() for m in QA_SIGNAL_RE.finditer(jd_text)})


def build_resume_job_fit(pairs: List[Tuple[str, str, int]], out_dir: Path,
                         min_signals: int = MIN_QA_SIGNALS) -> Dict:
    """Write cvs/, jds.yaml, gold.json and meta.json for the QA / test-automation job descriptions."""
    qa = [(r, j, lab) for r, j, lab in pairs if len(qa_signals(j)) >= min_signals]
    if not qa:
        raise ValueError(f"no job description has {min_signals} or more QA signals")
    cv_dir = out_dir / "cvs"
    cv_dir.mkdir(parents=True, exist_ok=True)
    for stale in cv_dir.glob("rjf_*.txt"):  # a rebuild with another filter must not keep old CVs
        stale.unlink()
    gold: Dict[str, Dict] = {}
    jd_texts: Dict[str, str] = {}
    for resume, jd_text, label in qa:
        cv_id, jd_id = _stable_id("rjf", resume), _stable_id("rjf_jd", jd_text)
        (cv_dir / f"{cv_id}.txt").write_text(resume, encoding="utf-8")
        jd_texts[jd_id] = jd_text
        rel = gold.setdefault(jd_id, {"relevance": {}})["relevance"]
        rel[cv_id] = max(label, rel.get(cv_id, 0))  # duplicate pair: keep the higher label
    ordered = sorted(jd_texts, key=lambda j: hashlib.sha1(j.encode()).hexdigest())
    n_dev = max(1, round(len(ordered) * DEV_SHARE)) if len(ordered) > 1 else 0
    specs = []
    for i, jd_id in enumerate(ordered):
        split = "dev" if i < n_dev else "test"
        rel = gold[jd_id]["relevance"]
        gold[jd_id].update({"hr_ranking": sorted(rel, key=lambda c: (-rel[c], c)), "split": split,
                            "ambiguous": False, "judged_only": True})
        specs.append({"id": jd_id, "title": _title(jd_texts[jd_id]), "company_type": "default",
                      "profile": DEFAULT_PROFILE, "split": split, "qa_signals": qa_signals(jd_texts[jd_id]),
                      "text": jd_texts[jd_id]})
    meta = {"synthetic": False, "source": f"{HF_BASE}/{RJF_REPO}", "license": "not stated on the dataset card",
            "labels": LABELS, "qa_signal_pattern": QA_SIGNAL_RE.pattern, "min_qa_signals": min_signals, "n_pairs": len(qa),
            "n_cvs": len({c for g in gold.values() for c in g["relevance"]}), "n_jds": len(gold)}
    (out_dir / "jds.yaml").write_text(yaml.safe_dump(specs, allow_unicode=True, sort_keys=False, width=120),
                                      encoding="utf-8")
    (out_dir / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def fetch_resume_job_fit(out_dir: Path, raw_dir: Path, min_signals: int = MIN_QA_SIGNALS) -> Dict:
    return build_resume_job_fit(read_pairs(download(RJF_REPO, f, raw_dir) for f in RJF_FILES), out_dir, min_signals)


def fetch_skillspan(raw_dir: Path) -> List[Path]:
    return [download(SKILLSPAN_REPO, f, raw_dir) for f in SKILLSPAN_FILES]
