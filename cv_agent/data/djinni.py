"""Djinni Recruitment Dataset (Drushchak & Romanyshyn, UNLP 2024, MIT) as the source of a human-labelled Gold Set.

Djinni is an IT job platform whose candidate profiles are anonymous by design; the dataset authors also removed
every CV that still matched PII patterns. This module picks a labelling set from the English files:

* JDs: ``QA Automation`` postings, one per company, spread over the main test stacks and experience levels;
* candidates: one pool shared by all JDs — ``QA Automation`` profiles, ``QA`` (mostly manual) profiles and
  developer profiles as hard negatives, so the ranking task is not trivially separable by keyword.

``build`` writes ``cvs/``, ``jds.yaml`` and ``meta.json`` in the layout of ``data/synthetic``; ``gold.json`` is
written later from the two annotators' sheets (``cv_agent.eval.annotation``).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict

import yaml

from cv_agent.data.public import DEFAULT_PROFILE

CAND_FILE = "candidate-profiles.parquet"
JOB_FILE = "job-descriptions.parquet"
HF_REPOS = {CAND_FILE: "lang-uk/recruitment-dataset-candidate-profiles-english",
            JOB_FILE: "lang-uk/recruitment-dataset-job-descriptions-english"}
LICENSE = "MIT (Djinni Recruitment Dataset, Drushchak & Romanyshyn, UNLP 2024)"
JD_LEN = (800, 4000)          # characters: drop stubs and very long postings that are slow to read
CV_LEN = (200, 4000)
NEGATIVE_KEYWORDS = ("Java", "Python", "JavaScript", ".NET", "DevOps")
STACKS = {"java": r"\bjava\b(?!\s*script)", "python": r"\bpython\b|\bpytest\b",
          "javascript": r"\b(?:javascript|typescript|cypress|playwright|webdriverio)\b",
          "csharp": r"c#|\.net\b|\bnunit\b|\bspecflow\b"}


def stack_of(text: str) -> str:
    """Main test-automation stack named in a text (most mentions wins), ``other`` if none."""
    low = text.lower()
    counts = {s: len(re.findall(p, low)) for s, p in STACKS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else "other"


def _sample(df, n: int, seed: int):
    if n > len(df):
        raise ValueError(f"asked for {n} rows, only {len(df)} match")
    return df.sample(n=n, random_state=seed)


def select_jds(jobs, n: int, seed: int):
    """``n`` English QA Automation JDs, one per company, round-robin over (stack, experience) cells."""
    df = jobs[(jobs["Primary Keyword"] == "QA Automation") & (jobs["Long Description_lang"] == "en")]
    length = df["Long Description"].str.len()
    df = df[(length >= JD_LEN[0]) & (length <= JD_LEN[1])]
    df = df.sample(frac=1.0, random_state=seed).drop_duplicates("Company Name")
    df = df.drop_duplicates(subset="Long Description").assign(stack=df["Long Description"].map(stack_of))
    df = df[df["stack"] != "other"]
    cells = {k: g for k, g in df.groupby(["stack", "Exp Years"], sort=True)}
    picked, order = [], sorted(cells, key=lambda k: (hashlib.sha1(f"{seed}{k}".encode()).hexdigest()))
    by_stack: Dict[str, list] = {}
    for key in order:
        by_stack.setdefault(key[0], []).append(key)
    while len(picked) < n and any(by_stack.values()):
        for stack in sorted(by_stack):
            if len(picked) >= n or not by_stack[stack]:
                continue
            cell = cells[by_stack[stack].pop(0)]
            picked.append(cell.index[0])
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} JDs match the filters, asked for {n}")
    return df.loc[picked].drop(columns="stack")


def select_candidates(cands, n_auto: int, n_manual: int, n_other: int, seed: int):
    df = cands[cands["CV_lang"] == "en"]
    length = df["CV"].fillna("").str.len()
    df = df[(length >= CV_LEN[0]) & (length <= CV_LEN[1])].drop_duplicates(subset="CV")
    parts = [_sample(df[df["Primary Keyword"] == "QA Automation"], n_auto, seed),
             _sample(df[df["Primary Keyword"] == "QA"], n_manual, seed),
             _sample(df[df["Primary Keyword"].isin(NEGATIVE_KEYWORDS)], n_other, seed)]
    import pandas as pd
    return pd.concat(parts)


def cv_text(row) -> str:
    years = row.get("Experience Years")
    head = [str(row.get("Position") or "").strip(),
            " · ".join(x for x in (f"{years:g} năm kinh nghiệm" if years == years and years is not None else "",
                                   f"English: {row['English Level']}" if row.get("English Level") else "") if x)]
    return "\n".join([*(h for h in head if h), "", str(row.get("CV") or "").strip()])


def jd_text(row) -> str:
    return f"{row['Position']}\n\n{str(row['Long Description']).strip()}"


def _id(prefix: str, value) -> str:
    return f"{prefix}_{hashlib.sha1(str(value).encode('utf-8')).hexdigest()[:10]}"


def build(raw_dir: Path, out_dir: Path, n_jds: int = 8, n_auto: int = 150, n_manual: int = 70, n_other: int = 30,
          seed: int = 42) -> Dict:
    """Write the labelling set (CV texts, JD specs, provenance); returns ``meta``."""
    import pandas as pd
    jobs = pd.read_parquet(Path(raw_dir) / JOB_FILE)
    cands = pd.read_parquet(Path(raw_dir) / CAND_FILE)
    jd_rows = select_jds(jobs, n_jds, seed)
    cv_rows = select_candidates(cands, n_auto, n_manual, n_other, seed)
    cv_dir = Path(out_dir) / "cvs"
    cv_dir.mkdir(parents=True, exist_ok=True)
    for stale in cv_dir.glob("dj_*.txt"):
        stale.unlink()
    keyword_of = {}
    for _, row in cv_rows.iterrows():
        cid = _id("dj", row["id"])
        (cv_dir / f"{cid}.txt").write_text(cv_text(row), encoding="utf-8")
        keyword_of[cid] = row["Primary Keyword"]
    specs = [{"id": _id("dj_jd", row["id"]), "title": str(row["Position"]), "company_type": "default",
              "profile": DEFAULT_PROFILE, "split": "test", "company": str(row["Company Name"]),
              "exp_years": str(row["Exp Years"]), "stack": stack_of(row["Long Description"]), "text": jd_text(row)}
             for _, row in jd_rows.iterrows()]
    meta = {"synthetic": False, "source": {f: f"https://huggingface.co/datasets/{r}" for f, r in HF_REPOS.items()},
            "license": LICENSE, "citation": "Drushchak, N. & Romanyshyn, M. (2024). Introducing the Djinni Recruitment "
                                            "Dataset: A Corpus of Anonymized CVs and Job Postings. UNLP 2024.",
            "selection": {"seed": seed, "n_jds": n_jds, "n_auto": n_auto, "n_manual": n_manual, "n_other": n_other,
                          "jd_length": JD_LEN, "cv_length": CV_LEN, "negative_keywords": NEGATIVE_KEYWORDS},
            "n_cvs": len(keyword_of), "n_jds": len(specs), "primary_keyword": keyword_of}
    (Path(out_dir) / "jds.yaml").write_text(yaml.safe_dump(specs, allow_unicode=True, sort_keys=False, width=120),
                                            encoding="utf-8")
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta
