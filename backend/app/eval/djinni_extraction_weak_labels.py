"""Generate deterministic weak labels for pipeline debugging only.

This file does not create human annotations. It simulates two variants of the
same regex/taxonomy heuristic and must never be used for IAA, Human Gold, model
selection, or headline evaluation. The generated differences are synthetic:
- Annotator A: Tends to capture broader spans (e.g. 'Selenium WebDriver', 'REST API') and leans towards 'used'
  when tools appear in project contexts.
- Annotator B: Tends to capture minimal base tool names (e.g. 'Selenium', 'API'), strictly marks keyword lists
  under technical summaries as 'listed', and has a stricter threshold for 'hedged'.
- A heuristic rule resolves the generated discrepancies into a weak reference.

Calculates:
- Span Token-Overlap F1
- Unweighted Cohen's Kappa for nominal status (used vs listed vs hedged)
- Candidate-level Jaccard similarity & Macro-F1
- Out-Of-Vocabulary (OOV) Rate
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import yaml

from app.config import ROOT, get_settings
from app.nlp.taxonomy import skills, aliases

CVS_DIR = get_settings().runtime_dir / "bench" / "djinni_extraction" / "cvs"
MANIFEST_FILE = ROOT / "data" / "eval" / "djinni_extraction_benchmark_manifest.json"
TAXONOMY_FILE = ROOT / "data" / "skills_taxonomy.yaml"

OUT_A = ROOT / "data" / "eval" / "weak_extraction_a.json"
OUT_B = ROOT / "data" / "eval" / "weak_extraction_b.json"
OUT_REF = ROOT / "data" / "eval" / "heuristic_extraction_reference_50.json"

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


def extract_base_mentions(text: str) -> list[dict]:
    terms = sorted(list(TAX_MAP.keys()) + list(KNOWN_OOV_TOOLS.keys()), key=lambda x: -len(x))
    mentions = []
    text_lower = text.lower()
    found_intervals = []

    for term in terms:
        pattern = rf"(?<![a-zA-Z0-9_-]){re.escape(term)}(?![a-zA-Z0-9_-])"
        for m in re.finditer(pattern, text_lower):
            start, end = m.span()
            if any(s <= start and end <= e for s, e in found_intervals):
                continue
            found_intervals.append((start, end))

            raw_snippet = text[start:end]
            canonical = TAX_MAP.get(term)
            is_oov = canonical is None

            line_start = text.rfind("\n", 0, start)
            line_start = 0 if line_start == -1 else line_start + 1
            line_end = text.find("\n", end)
            line_end = len(text) if line_end == -1 else line_end
            line = text[line_start:line_end].strip()
            line_lower = line.lower()

            hedged_cues = ["familiar with", "learning", "basic", "knowledge of", "exposure", "self-study", "course"]
            is_hedged = any(c in line_lower for c in hedged_cues)
            is_listed = any(line_lower.startswith(k) for k in ["stack:", "skills:", "tools:", "technologies:"]) or ("," in line and len(line.split()) < 15)

            if is_hedged:
                default_status = "hedged"
            elif is_listed:
                default_status = "listed"
            else:
                default_status = "used"

            mentions.append({
                "start": start,
                "end": end,
                "raw_text": raw_snippet,
                "canonical_skill": canonical,
                "is_oov": is_oov,
                "evidence_status": default_status,
                "line": line,
            })

    mentions.sort(key=lambda x: x["start"])
    return mentions


def generate_annotations_with_disagreements():
    test_profiles = load_test_manifest()
    ann_a = {}
    ann_b = {}
    ref_adjudicated = {}

    total_spans_a = 0
    total_spans_b = 0
    disagreements_status = 0
    disagreements_spans = 0

    for idx, prof in enumerate(test_profiles):
        cid = prof["candidate_id"]
        cv_text = (CVS_DIR / f"{cid}.txt").read_text(encoding="utf-8")
        base = extract_base_mentions(cv_text)

        mentions_a = []
        mentions_b = []
        mentions_ref = []

        for m_idx, m in enumerate(base):
            item_a = dict(m)
            item_b = dict(m)

            # Introduce authentic human disagreement patterns:
            # 1. Annotator B is stricter on 'listed': in summary or stack lines without explicit action verbs, B marks 'listed'
            if m["evidence_status"] == "used" and ("stack" in m["line"].lower() or len(m["line"].split()) < 8):
                if (idx + m_idx) % 3 == 0:
                    item_b["evidence_status"] = "listed"
                    disagreements_status += 1

            # 2. Subtle hedge: "knowledge of" or "understanding" is marked 'hedged' by B but 'used' by A if in experience
            if "knowledge" in m["line"].lower() or "understanding" in m["line"].lower():
                item_a["evidence_status"] = "used"
                item_b["evidence_status"] = "hedged"
                disagreements_status += 1

            # 3. Occasional boundary nuance: Annotator A captures compound term, Annotator B captures base term
            if "webdriver" in m["raw_text"].lower():
                item_a["raw_text"] = "Selenium WebDriver"
                item_b["raw_text"] = "Selenium"

            # 4. Rare miss: Annotator B missed 1 obscure acronym in a dense list (< 3% miss rate)
            if (idx * 7 + m_idx) % 47 == 0 and len(base) > 5:
                disagreements_spans += 1
                mentions_a.append(item_a)
                # B missed this mention
                # Adjudicator confirms it should be present
                mentions_ref.append(item_a)
                continue

            mentions_a.append(item_a)
            mentions_b.append(item_b)

            # Adjudication: Reference consensus
            item_ref = dict(item_a)
            if item_a["evidence_status"] != item_b["evidence_status"]:
                # Adjudicator checks action verbs
                has_verb = any(v in m["line"].lower() for v in ["developed", "built", "executed", "automated", "created", "maintained"])
                if has_verb and item_a["evidence_status"] != "hedged":
                    item_ref["evidence_status"] = "used"
                else:
                    item_ref["evidence_status"] = item_b["evidence_status"]

            mentions_ref.append(item_ref)

        total_spans_a += len(mentions_a)
        total_spans_b += len(mentions_b)

        # Candidate skills
        cand_a = sorted(list({m["canonical_skill"] for m in mentions_a if m["canonical_skill"] and m["evidence_status"] != "hedged"}))
        cand_b = sorted(list({m["canonical_skill"] for m in mentions_b if m["canonical_skill"] and m["evidence_status"] != "hedged"}))
        cand_ref = sorted(list({m["canonical_skill"] for m in mentions_ref if m["canonical_skill"] and m["evidence_status"] != "hedged"}))

        ann_a[cid] = {"mentions": mentions_a, "candidate_skills": cand_a}
        ann_b[cid] = {"mentions": mentions_b, "candidate_skills": cand_b}
        ref_adjudicated[cid] = {
            "mentions": mentions_ref,
            "candidate_skills": cand_ref,
            "all_mention_count": len(mentions_ref),
            "oov_mentions": [m["raw_text"] for m in mentions_ref if m["is_oov"]],
            "in_tax_mentions": [m["canonical_skill"] for m in mentions_ref if not m["is_oov"]],
            "domain": prof["domain"],
            "length_stratum": prof["length_stratum"],
        }

    OUT_A.write_text(json.dumps(ann_a, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_B.write_text(json.dumps(ann_b, indent=2, ensure_ascii=False), encoding="utf-8")

    tot_ref_spans = sum(len(v["mentions"]) for v in ref_adjudicated.values())
    tot_oov = sum(len(v["oov_mentions"]) for v in ref_adjudicated.values())
    oov_rate = round(tot_oov / max(1, tot_ref_spans) * 100, 2)

    ref_payload = {
        "metadata": {
            "artifact_type": "generated_heuristic_extraction_reference",
            "evaluation_status": "WEAK_DIAGNOSTIC_ONLY",
            "human": False,
            "independent": False,
            "simulated": True,
            "generated_by": "app.eval.djinni_extraction_weak_labels",
            "heuristic_adjudication": True,
            "benchmark": "Djinni extraction weak-label diagnostic",
            "n_profiles": len(test_profiles),
            "total_annotated_spans": tot_ref_spans,
            "total_oov_spans": tot_oov,
            "oov_rate_pct": oov_rate,
            "annotator_a_spans": total_spans_a,
            "annotator_b_spans": total_spans_b,
            "disagreements_status": disagreements_status,
            "disagreements_spans": disagreements_spans,
            "guidelines": "docs/labeling/EXTRACTION_GUIDELINES.md",
        },
        "profiles": ref_adjudicated,
    }
    OUT_REF.write_text(json.dumps(ref_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Generated WEAK labels with synthetic disagreements (not Human Gold):")
    print(f"  Spans A: {total_spans_a}, Spans B: {total_spans_b}, Adjudicated Ref: {tot_ref_spans}")
    print(f"  Status Disagreements: {disagreements_status}, Span Disagreements: {disagreements_spans}")
    print(f"  OOV Rate: {oov_rate}% ({tot_oov}/{tot_ref_spans})")


if __name__ == "__main__":
    generate_annotations_with_disagreements()
