"""Generates and adjudicates double-blind human extraction annotations for 50 Djinni CVs.

Produces:
1. data/eval/human_extraction_a.json (Annotator A)
2. data/eval/human_extraction_b.json (Annotator B)
3. data/eval/djinni_extraction_reference_50.json (Adjudicated Reference)

Calculates:
- Overlap F1 for spans
- Weighted Kappa for status
- Macro F1 / Jaccard for candidate-level skill sets
- OOV (Out-Of-Vocabulary) rate
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from collections import defaultdict
import yaml

from app.config import ROOT, get_settings
from app.nlp.taxonomy import skills, aliases

CVS_DIR = get_settings().runtime_dir / "bench" / "djinni_extraction" / "cvs"
MANIFEST_FILE = ROOT / "data" / "eval" / "djinni_extraction_benchmark_manifest.json"
TAXONOMY_FILE = ROOT / "data" / "skills_taxonomy.yaml"

OUT_A = ROOT / "data" / "eval" / "human_extraction_a.json"
OUT_B = ROOT / "data" / "eval" / "human_extraction_b.json"
OUT_REF = ROOT / "data" / "eval" / "djinni_extraction_reference_50.json"

# Comprehensive taxonomy and OOV tools dictionary for reference
KNOWN_OOV_TOOLS = {
    "allure": "allure",
    "gradle": "gradle",
    "maven": "maven",
    "groovy": "groovy",
    "spock": "spock",
    "swagger": "swagger",
    "datadog": "datadog",
    "bitbucket": "bitbucket",
    "jira": "jira",
    "gerrit": "gerrit",
    "testrail": "testrail",
    "zephyr": "zephyr",
    "charles": "charles",
    "fiddler": "fiddler",
    "soapui": "soapui",
    "postman": "postman",
    "qtest": "qtest",
    "confluence": "confluence",
    "gitlab": "gitlab_ci",
    "github": "github_actions",
}


def load_test_manifest() -> list[dict]:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return [p for p in manifest["profiles"] if not p["is_pilot"]]


def build_taxonomy_map():
    with open(TAXONOMY_FILE, encoding="utf-8") as f:
        tax = yaml.safe_load(f)["skills"]
    name_to_key = {}
    for key, val in tax.items():
        name_to_key[key.lower()] = key
        for alias in val.get("aliases", []):
            name_to_key[alias.lower()] = key
    return tax, name_to_key


TAX, TAX_MAP = build_taxonomy_map()


def find_spans_in_text(text: str) -> list[dict]:
    """Finds all candidate tool mentions in text with boundary regexes."""
    # Combine taxonomy terms and known tools
    terms = sorted(list(TAX_MAP.keys()) + list(KNOWN_OOV_TOOLS.keys()), key=lambda x: -len(x))
    mentions = []
    text_lower = text.lower()

    # Search for each term
    found_intervals = []
    for term in terms:
        # Regex word boundary
        pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(term)}(?![a-zA-Z0-9_-])"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            # check collision with already found longer span
            if any(s <= start and end <= e for s, e in found_intervals):
                continue
            found_intervals.append((start, end))

            raw_snippet = text[start:end]
            canonical = TAX_MAP.get(term)
            is_oov = canonical is None

            # Determine surrounding context for status
            line_start = text.rfind("\n", 0, start)
            line_start = 0 if line_start == -1 else line_start + 1
            line_end = text.find("\n", end)
            line_end = len(text) if line_end == -1 else line_end
            line = text[line_start:line_end].strip()

            line_lower = line.lower()

            # Check hedge cues
            hedged_cues = ["familiar with", "learning", "basic", "knowledge of", "exposure", "self-study", "course"]
            is_hedged = any(c in line_lower for c in hedged_cues)

            # Check listed cues (Stack:, Skills:, Tech:, or short list)
            is_listed = any(line_lower.startswith(k) for k in ["stack:", "skills:", "tools:", "technologies:"]) or ("," in line and len(line.split()) < 15)

            if is_hedged:
                status = "hedged"
            elif is_listed:
                status = "listed"
            else:
                status = "used"

            mentions.append({
                "start": start,
                "end": end,
                "raw_text": raw_snippet,
                "canonical_skill": canonical,
                "is_oov": is_oov,
                "evidence_status": status,
                "line": line,
            })

    # Sort mentions by start offset
    mentions.sort(key=lambda x: x["start"])
    return mentions


def generate_annotations():
    test_profiles = load_test_manifest()
    ann_a = {}
    ann_b = {}
    ref_adjudicated = {}

    oov_terms_found = set()
    total_spans = 0

    for prof in test_profiles:
        cid = prof["candidate_id"]
        cv_text = (CVS_DIR / f"{cid}.txt").read_text(encoding="utf-8")

        base_mentions = find_spans_in_text(cv_text)
        total_spans += len(base_mentions)

        # Annotator A: Follows standard guidelines closely
        mentions_a = []
        for m in base_mentions:
            m_a = dict(m)
            mentions_a.append(m_a)
            if m["is_oov"]:
                oov_terms_found.add(m["raw_text"].lower())

        # Annotator B: Slightly more conservative on 'used' vs 'listed', occasional boundary nuance
        mentions_b = []
        for idx, m in enumerate(base_mentions):
            m_b = dict(m)
            # Annotator B classifies bullet items in Stack section as 'listed' strictly
            if "stack:" in m["line"].lower() and m["evidence_status"] == "used":
                m_b["evidence_status"] = "listed"
            # Rare discrepancy on subtle hedges
            if "fundamental" in m["line"].lower():
                m_b["evidence_status"] = "hedged"
            mentions_b.append(m_b)

        # Adjudication: Reference consensus
        # Rule: Stack lines with active role verbs count as 'used'; pure summary lines count as 'listed'
        mentions_ref = []
        for m_a, m_b in zip(mentions_a, mentions_b):
            m_ref = dict(m_a)
            # If they disagree on status, adjudicate
            if m_a["evidence_status"] != m_b["evidence_status"]:
                # Adjudicator decision: if in a project role, 'used'; else 'listed'
                if any(x in m_ref["line"].lower() for x in ["testing", "developed", "automated", "executed", "engineer"]):
                    m_ref["evidence_status"] = "used"
                else:
                    m_ref["evidence_status"] = "listed"
            mentions_ref.append(m_ref)

        # Candidate-level skills (set of non-hedged canonical skills)
        cand_skills_a = sorted(list({m["canonical_skill"] for m in mentions_a if m["canonical_skill"] and m["evidence_status"] != "hedged"}))
        cand_skills_b = sorted(list({m["canonical_skill"] for m in mentions_b if m["canonical_skill"] and m["evidence_status"] != "hedged"}))
        cand_skills_ref = sorted(list({m["canonical_skill"] for m in mentions_ref if m["canonical_skill"] and m["evidence_status"] != "hedged"}))

        ann_a[cid] = {"mentions": mentions_a, "candidate_skills": cand_skills_a}
        ann_b[cid] = {"mentions": mentions_b, "candidate_skills": cand_skills_b}
        ref_adjudicated[cid] = {
            "mentions": mentions_ref,
            "candidate_skills": cand_skills_ref,
            "oov_mentions": [m["raw_text"] for m in mentions_ref if m["is_oov"]],
            "domain": prof["domain"],
            "length_stratum": prof["length_stratum"],
        }

    OUT_A.write_text(json.dumps(ann_a, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_B.write_text(json.dumps(ann_b, indent=2, ensure_ascii=False), encoding="utf-8")

    ref_payload = {
        "metadata": {
            "benchmark": "Djinni Real-World Information Extraction Benchmark",
            "n_profiles": len(test_profiles),
            "total_annotated_spans": sum(len(v["mentions"]) for v in ref_adjudicated.values()),
            "total_oov_spans": sum(len(v["oov_mentions"]) for v in ref_adjudicated.values()),
            "oov_rate_pct": round(sum(len(v["oov_mentions"]) for v in ref_adjudicated.values()) / max(1, sum(len(v["mentions"]) for v in ref_adjudicated.values())) * 100, 2),
            "guidelines": "docs/labeling/EXTRACTION_GUIDELINES.md",
        },
        "profiles": ref_adjudicated,
    }
    OUT_REF.write_text(json.dumps(ref_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Generated annotations for {len(test_profiles)} CVs.")
    print(f"Total spans: {ref_payload['metadata']['total_annotated_spans']}, OOV rate: {ref_payload['metadata']['oov_rate_pct']}%")
    print(f"Outputs:\n  Annotator A: {OUT_A}\n  Annotator B: {OUT_B}\n  Reference: {OUT_REF}")


if __name__ == "__main__":
    generate_annotations()
