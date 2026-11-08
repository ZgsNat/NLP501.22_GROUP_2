"""Stratified sampling for Djinni Information Extraction Benchmark.

Selects and hash-locks:
- 12 pilot profiles (for annotator alignment / training)
- 50 test profiles (frozen before blind human annotation)

Strict exclusion:
- All 250 production Djinni CVs
- All 40 Djinni ranking holdout CVs
- Any text hash match with previously exposed CVs

Stratification quota:
- Test (50 CVs): 30 QA Automation, 15 Manual QA, 5 Other (Dev/DevOps)
- Pilot (12 CVs): 6 QA Automation, 4 Manual QA, 2 Other
Each further stratified across short (<800), medium (800-2000), long (>2000) lengths.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from app.config import ROOT, get_settings

SEED = 20261108
CV_MIN, CV_MAX = 200, 4000
OTHER_KEYWORDS = frozenset({"Java", "Python", "JavaScript", ".NET", "DevOps"})

SOURCE_PARQUET = get_settings().runtime_dir / "holdout" / "candidate_profiles.parquet"
PROD_META = get_settings().runtime_dir / "bench" / "djinni" / "meta.json"
HOLDOUT_MANIFEST = ROOT / "data" / "eval" / "djinni_holdout_v1.json"

OUT_DIR = get_settings().runtime_dir / "bench" / "djinni_extraction"
OUT_CVS = OUT_DIR / "cvs"
MANIFEST_FILE = ROOT / "data" / "eval" / "djinni_extraction_benchmark_manifest.json"


def _clean(val: object) -> str:
    return str(val).strip() if val is not None else ""


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _public_id(raw_id: object) -> str:
    return "dj_ext_" + hashlib.sha256(_clean(raw_id).encode("utf-8")).hexdigest()[:10]


def _legacy_id(prefix: str, raw_id: object) -> str:
    return prefix + hashlib.sha1(_clean(raw_id).encode("utf-8")).hexdigest()[:10]


def _profile_text(row: dict) -> str:
    position = _clean(row.get("Position"))
    years = row.get("Experience Years")
    english = _clean(row.get("English Level"))
    summary = []
    if years is not None:
        try:
            summary.append(f"{float(years):g} years experience")
        except (TypeError, ValueError):
            pass
    if english:
        summary.append(f"English: {english}")
    parts = [position, " · ".join(summary)]
    for key in ("Moreinfo", "Looking For", "Highlights", "CV"):
        val = _clean(row.get(key))
        if val and val not in parts:
            parts.append(val)
    return "\n\n".join(x for x in parts if x).strip()


def get_excluded_sets() -> tuple[set[str], set[str]]:
    excluded_ids = set()
    excluded_hashes = set()

    if PROD_META.exists():
        meta = json.loads(PROD_META.read_text(encoding="utf-8"))
        for k in meta.get("primary_keyword", {}):
            excluded_ids.add(k)
            if k.startswith("dj_"):
                excluded_ids.add(k[3:])

    prod_cv_dir = get_settings().runtime_dir / "bench" / "djinni" / "cvs"
    if prod_cv_dir.exists():
        for f in prod_cv_dir.glob("*.txt"):
            excluded_hashes.add(hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest())

    if HOLDOUT_MANIFEST.exists():
        hm = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
        for item in hm.get("manifest", []):
            cid = item.get("candidate_id", "")
            excluded_ids.add(cid)
            if cid.startswith("dj_"):
                excluded_ids.add(cid[3:])
            if "sha256" in item:
                excluded_hashes.add(item["sha256"])

    return excluded_ids, excluded_hashes


def sample_profiles() -> dict:
    import pyarrow.parquet as pq

    excluded_ids, excluded_hashes = get_excluded_sets()

    # Targets:
    # Test: QA Automation 30, QA 15, other 5
    # Pilot: QA Automation 6, QA 4, other 2
    quota_test = {"QA Automation": 30, "QA": 15, "other": 5}
    quota_pilot = {"QA Automation": 6, "QA": 4, "other": 2}

    parquet = pq.ParquetFile(SOURCE_PARQUET)
    candidates_by_bucket = {"QA Automation": [], "QA": [], "other": []}

    for batch in parquet.iter_batches(batch_size=4096):
        for row in batch.to_pylist():
            kw = _clean(row.get("Primary Keyword"))
            bucket = kw if kw in {"QA Automation", "QA"} else ("other" if kw in OTHER_KEYWORDS else "")
            if not bucket:
                continue

            raw_id = row.get("id")
            leg_id = _legacy_id("dj_", raw_id)
            if leg_id in excluded_ids or str(raw_id) in excluded_ids:
                continue

            text = _profile_text(row)
            if not CV_MIN <= len(text) <= CV_MAX:
                continue

            thash = _sha_text(text)
            if thash in excluded_hashes:
                continue

            # Length stratum
            if len(text) < 800:
                len_stratum = "short"
            elif len(text) <= 2000:
                len_stratum = "medium"
            else:
                len_stratum = "long"

            candidates_by_bucket[bucket].append({
                "raw_id": raw_id,
                "domain": bucket,
                "length_stratum": len_stratum,
                "text_length": len(text),
                "sha256": thash,
                "text": text,
            })

    # Shuffle with SEED
    rng = random.Random(SEED)
    selected_pilot = []
    selected_test = []

    for bucket in ("QA Automation", "QA", "other"):
        pool = candidates_by_bucket[bucket]
        rng.shuffle(pool)

        n_pilot = quota_pilot[bucket]
        n_test = quota_test[bucket]

        pilot_items = pool[:n_pilot]
        test_items = pool[n_pilot : n_pilot + n_test]

        for item in pilot_items:
            item["is_pilot"] = True
            item["candidate_id"] = _public_id(item["raw_id"])
            selected_pilot.append(item)

        for item in test_items:
            item["is_pilot"] = False
            item["candidate_id"] = _public_id(item["raw_id"])
            selected_test.append(item)

    # Save to disk
    OUT_CVS.mkdir(parents=True, exist_ok=True)
    manifest_records = []

    for item in selected_pilot + selected_test:
        cid = item["candidate_id"]
        cv_path = OUT_CVS / f"{cid}.txt"
        cv_path.write_text(item["text"], encoding="utf-8")

        manifest_records.append({
            "candidate_id": cid,
            "domain": item["domain"],
            "length_stratum": item["length_stratum"],
            "text_length": item["text_length"],
            "sha256": item["sha256"],
            "is_pilot": item["is_pilot"],
        })

    manifest_data = {
        "dataset": "Djinni Recruitment Dataset (UNLP 2024)",
        "seed": SEED,
        "n_pilot": len(selected_pilot),
        "n_test": len(selected_test),
        "quota_test": quota_test,
        "quota_pilot": quota_pilot,
        "profiles": manifest_records,
    }

    MANIFEST_FILE.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sampled {len(selected_pilot)} pilot and {len(selected_test)} test profiles.")
    print(f"Manifest written to {MANIFEST_FILE}")
    print(f"Text files saved to {OUT_CVS}")
    return manifest_data


def verify_frozen_manifest() -> dict:
    """Fail if the frozen split changed or overlaps earlier Djinni pools."""
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    records = manifest["profiles"]
    ids = [item["candidate_id"] for item in records]
    hashes = [item["sha256"] for item in records]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("Frozen extraction manifest contains duplicate IDs or text hashes")
    if sum(item["is_pilot"] for item in records) != manifest["n_pilot"]:
        raise ValueError("Pilot count changed")
    if sum(not item["is_pilot"] for item in records) != manifest["n_test"]:
        raise ValueError("Test count changed")

    _, excluded_hashes = get_excluded_sets()
    overlap = sorted(set(hashes) & excluded_hashes)
    if overlap:
        raise ValueError(f"Frozen extraction split overlaps prior pools by {len(overlap)} text hashes")
    for item in records:
        path = OUT_CVS / f"{item['candidate_id']}.txt"
        raw_text = path.read_bytes().decode("utf-8") if path.exists() else ""
        if (
            not path.exists()
            or len(raw_text) != item["text_length"]
            or _sha_text(raw_text) != item["sha256"]
        ):
            raise ValueError(f"Frozen document missing or changed: {item['candidate_id']}")
    return {"n_pilot": manifest["n_pilot"], "n_test": manifest["n_test"], "overlap_hashes": 0}


if __name__ == "__main__":
    sample_profiles()
