"""Freeze and import a genuinely unseen Djinni-QA holdout.

The production Djinni pilot has already been viewed, so it cannot repair the
dev/test-leakage limitation.  This module creates a disjoint, fixed-seed slice
from the original public parquet files and produces two blind local annotation
kits.  It deliberately does *not* run a ranker or manufacture labels.

Real texts live only below ``.runtime/``.  The committed manifest contains
source/content hashes, public source identifiers and the pre-registered
protocol, but no CV/JD text.

Prepare (pyarrow is an ephemeral reader, not a project runtime dependency)::

    uv run --with pyarrow python -m app.eval.djinni_holdout prepare
    python -m app.eval.djinni_holdout verify

After two people export their independent JSON files::

    python -m app.eval.djinni_holdout import-labels A.json B.json

No system may run on this holdout before ``import-labels`` creates Gold.  If
any pair differs by two or more grades, an independent adjudication file is
required via ``--adjudicated``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path
from typing import Iterable

import yaml

from app.config import ROOT, get_settings
from app.eval.metrics import weighted_kappa

SEED = 5_012_026
N_JDS = 6
CV_QUOTA = {"QA Automation": 24, "QA": 10, "other": 6}
OTHER_KEYWORDS = frozenset({"Java", "Python", "JavaScript", ".NET", "DevOps"})
JD_MIN, JD_MAX = 800, 4_000
CV_MIN, CV_MAX = 200, 4_000

SOURCE_DIR = get_settings().runtime_dir / "holdout"
JD_SOURCE = SOURCE_DIR / "job_descriptions.parquet"
CV_SOURCE = SOURCE_DIR / "candidate_profiles.parquet"
OUT = get_settings().runtime_dir / "bench" / "djinni_holdout"
KITS = SOURCE_DIR / "labeling"
MANIFEST = ROOT / "data" / "eval" / "djinni_holdout_v1.json"
GOLD = ROOT / "data" / "eval" / "human_djinni_holdout_gold.json"
CURRENT = get_settings().runtime_dir / "bench" / "djinni"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        h.update(path.name.encode())
        h.update(b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    return h.hexdigest()


def _clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _public_id(prefix: str, raw_id: object) -> str:
    return prefix + hashlib.sha256(_clean(raw_id).encode()).hexdigest()[:12]


def _legacy_id(prefix: str, raw_id: object) -> str:
    """ID recipe used by the measured Djinni pilot (verified 250/250 and 8/8)."""
    return prefix + hashlib.sha1(_clean(raw_id).encode()).hexdigest()[:10]


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
        value = _clean(row.get(key))
        if value and value not in parts:
            parts.append(value)
    return "\n\n".join(x for x in parts if x).strip()


def _jd_text(row: dict) -> str:
    title, body = _clean(row.get("Position")), _clean(row.get("Long Description"))
    return f"{title}\n\n{body}".strip()


def _stack(text: str) -> str:
    low = text.lower()
    tests = (("csharp", ("c#", ".net", "specflow")), ("python", ("python", "pytest")),
             ("javascript", ("javascript", "typescript", "cypress", "playwright")),
             ("java", ("java", "testng", "junit")))
    scores = {name: sum(token in low for token in tokens) for name, tokens in tests}
    return max(scores, key=lambda name: (scores[name], name)) if max(scores.values()) else "other"


def _iter_rows(path: Path, columns: list[str]):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by the documented CLI command
        raise SystemExit("prepare needs pyarrow: use `uv run --with pyarrow python -m "
                         "app.eval.djinni_holdout prepare`") from exc
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()


def _reservoir(rows: Iterable[dict], quota: dict[str, int], seed: int) -> dict[str, list[dict]]:
    rng = {key: random.Random(seed + i * 10_007) for i, key in enumerate(sorted(quota))}
    seen = {key: 0 for key in quota}
    selected = {key: [] for key in quota}
    for row in rows:
        bucket = _clean(row.get("Primary Keyword"))
        bucket = bucket if bucket in {"QA Automation", "QA"} else ("other" if bucket in OTHER_KEYWORDS else "")
        if not bucket:
            continue
        text = _profile_text(row)
        if not CV_MIN <= len(text) <= CV_MAX:
            continue
        seen[bucket] += 1
        item = {**row, "_text": text}
        if len(selected[bucket]) < quota[bucket]:
            selected[bucket].append(item)
        else:
            at = rng[bucket].randrange(seen[bucket])
            if at < quota[bucket]:
                selected[bucket][at] = item
    if any(len(selected[key]) != n for key, n in quota.items()):
        raise ValueError(f"not enough eligible profiles: { {k: len(v) for k, v in selected.items()} }")
    return selected


def _select_jds(rows: Iterable[dict], excluded_ids: set[str], excluded_text_hashes: set[str]) -> list[dict]:
    eligible = []
    for row in rows:
        if _clean(row.get("Primary Keyword")) != "QA Automation":
            continue
        text = _jd_text(row)
        if (_legacy_id("dj_jd_", row.get("id")) in excluded_ids or not JD_MIN <= len(text) <= JD_MAX
                or _sha_bytes(text.encode()) in excluded_text_hashes):
            continue
        eligible.append({**row, "_text": text, "_stack": _stack(text)})
    random.Random(SEED).shuffle(eligible)
    chosen, companies = [], set()
    for target in ("java", "python", "javascript", "csharp"):
        item = next((r for r in eligible if r["_stack"] == target and _clean(r.get("Company Name")) not in companies), None)
        if item:
            chosen.append(item)
            companies.add(_clean(item.get("Company Name")))
    for row in eligible:
        company = _clean(row.get("Company Name"))
        if len(chosen) == N_JDS:
            break
        if row not in chosen and company not in companies:
            chosen.append(row)
            companies.add(company)
    if len(chosen) != N_JDS:
        raise ValueError(f"only {len(chosen)} disjoint eligible JDs")
    return chosen


def _label_html(pairs: list[dict], annotator: str) -> str:
    # JSON is embedded only in .runtime; escape '<' so a public text cannot close the script element.
    payload = json.dumps(pairs, ensure_ascii=False).replace("<", "\\u003c")
    title = f"Djinni unseen holdout — Annotator {annotator}"
    return f"""<!doctype html><html lang="vi"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui;margin:0;background:#f4f6f8;color:#17212b}}header{{position:sticky;top:0;background:#14213d;color:white;padding:12px 20px;z-index:2}}main{{max-width:1200px;margin:auto;padding:18px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}article{{white-space:pre-wrap;background:white;border:1px solid #ccd5df;border-radius:8px;padding:14px;max-height:55vh;overflow:auto}}.grade{{display:flex;gap:10px;margin:16px 0}}button{{padding:10px 16px}}button.sel{{background:#1769aa;color:white}}textarea{{width:100%;height:55px}}.actions{{display:flex;justify-content:space-between;align-items:center}}small{{opacity:.8}}</style>
<header><b>{title}</b> <span id="progress"></span></header><main>
<p><b>Chấm độ phù hợp cho JD này:</b> 0 = không phù hợp; 1 = yếu/chỉ chuyển đổi; 2 = phù hợp; 3 = rất phù hợp. Chỉ dùng nội dung hiển thị; không xem output hệ thống hay nhãn của người kia.</p>
<div class="grid"><article id="jd"></article><article id="cv"></article></div>
<div class="grade" id="grades"></div><textarea id="notes" placeholder="Ghi chú ngắn (khuyến nghị khi chấm 1 hoặc 2)"></textarea>
<div class="actions"><button onclick="move(-1)">← Trước</button><small>Tự lưu trong trình duyệt</small><button onclick="move(1)">Tiếp →</button><button onclick="exportLabels()">Xuất JSON</button></div>
</main><script>const pairs={payload};const who={json.dumps(annotator)};const key='djinni-holdout-v1-'+who;let state=JSON.parse(localStorage.getItem(key)||'{{}}');let i=0;
function save(){{const p=pairs[i];state[p.jd+'|'+p.cv]=state[p.jd+'|'+p.cv]||{{}};state[p.jd+'|'+p.cv].notes=document.querySelector('#notes').value;localStorage.setItem(key,JSON.stringify(state));}}
function render(){{const p=pairs[i],v=state[p.jd+'|'+p.cv]||{{}};document.querySelector('#jd').textContent='JD\n\n'+p.jd_text;document.querySelector('#cv').textContent='CV\n\n'+p.cv_text;document.querySelector('#notes').value=v.notes||'';document.querySelector('#grades').innerHTML=[0,1,2,3].map(g=>`<button class="${{v.grade===g?'sel':''}}" onclick="grade(${{g}})">${{g}}</button>`).join('');const n=Object.values(state).filter(x=>Number.isInteger(x.grade)).length;document.querySelector('#progress').textContent=`${{i+1}}/${{pairs.length}} · đã chấm ${{n}}`;}}
function grade(g){{const p=pairs[i];state[p.jd+'|'+p.cv]={{...(state[p.jd+'|'+p.cv]||{{}}),grade:g,notes:document.querySelector('#notes').value}};localStorage.setItem(key,JSON.stringify(state));render();}}
function move(d){{save();i=Math.max(0,Math.min(pairs.length-1,i+d));render();}}
function exportLabels(){{save();const labels=pairs.map(p=>({{jd:p.jd,cv:p.cv,...(state[p.jd+'|'+p.cv]||{{}})}}));if(labels.some(x=>!Number.isInteger(x.grade)))return alert('Cần chấm đủ 240 cặp trước khi xuất.');const out={{schema:'djinni-holdout-labels-v1',annotator:who,human:true,independent:true,saw_system_output:false,manifest_sha256:{json.dumps(_sha(MANIFEST) if MANIFEST.exists() else 'MANIFEST_PENDING')},labels}};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));a.download='human_djinni_holdout_'+who+'.json';a.click();}}render();</script></html>"""


def _manifest(jds: list[dict], cvs: list[dict]) -> dict:
    return {
        "benchmark": "djinni-unseen-human-holdout",
        "version": "v1",
        "status": "UNLABELLED_LOCKED",
        "created_before_system_run": True,
        "seed": SEED,
        "sampling": {"jds": N_JDS, "shared_cv_pool": sum(CV_QUOTA.values()), "pairs_per_annotator": N_JDS * sum(CV_QUOTA.values()),
                     "cv_quota": CV_QUOTA, "jd_length": [JD_MIN, JD_MAX], "cv_length": [CV_MIN, CV_MAX],
                     "exclusion": "all 8 JD and 250 CV source IDs plus normalized texts from the measured Djinni pilot"},
        "sources": {
            "jobs": {"url": "https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english",
                     "sha256": _sha(JD_SOURCE)},
            "profiles": {"url": "https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english",
                         "sha256": _sha(CV_SOURCE)},
        },
        "ids": {"jds": sorted(row["_id"] for row in jds), "cvs": sorted(row["_id"] for row in cvs)},
        "inputs": {"jds_tree_sha256": _tree_sha([OUT / "jds.yaml"]),
                   "cvs_tree_sha256": _tree_sha((OUT / "cvs").glob("*.txt"))},
        "annotation": {"annotators": 2, "blind": True, "independent": True, "scale": [0, 1, 2, 3],
                       "minimum_weighted_kappa": 0.6, "adjudicate_if_absolute_difference_at_least": 2,
                       "gold": "unrounded mean after required adjudication; conservative P@5 requires both >=2"},
        "gate": "Do not run or tune any system on these inputs until two human files pass import-labels and Gold exists.",
    }


def prepare() -> dict:
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} already exists; holdout v1 is immutable. Run `verify`, never resample v1.")
    for source in (JD_SOURCE, CV_SOURCE):
        if not source.exists():
            raise SystemExit(f"missing {source}; download the official parquet named in this module")
    old_jds = yaml.safe_load((CURRENT / "jds.yaml").read_text())
    old_jd_ids = {j["id"] for j in old_jds}
    old_cv_paths = list((CURRENT / "cvs").glob("*.txt"))
    old_cv_ids = {p.stem for p in old_cv_paths}
    old_jd_hashes = {_sha_bytes(_clean(j.get("text")).encode()) for j in old_jds}
    old_cv_hashes = {_sha_bytes(p.read_bytes()) for p in old_cv_paths}

    jd_cols = ["Position", "Long Description", "Company Name", "Exp Years", "Primary Keyword", "English Level", "id"]
    jds = _select_jds(_iter_rows(JD_SOURCE, jd_cols), old_jd_ids, old_jd_hashes)
    cv_cols = ["Position", "Moreinfo", "Looking For", "Highlights", "Primary Keyword", "English Level",
               "Experience Years", "CV", "id"]
    # Text-hash exclusion is checked in addition to source IDs; public text is never printed.
    cv_rows = (row for row in _iter_rows(CV_SOURCE, cv_cols)
               if _legacy_id("dj_", row.get("id")) not in old_cv_ids
               and _sha_bytes(_profile_text(row).encode()) not in old_cv_hashes)
    by = _reservoir(cv_rows, CV_QUOTA, SEED)
    cvs = [row for key in sorted(by) for row in by[key]]

    (OUT / "cvs").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    jd_payload = []
    for row in jds:
        row["_id"] = _public_id("djh_jd_", row.get("id"))
        jd_payload.append({"id": row["_id"], "title": _clean(row.get("Position")), "company_profile": "default",
                           "split": "unseen_test", "text": row["_text"]})
    (OUT / "jds.yaml").write_text(yaml.safe_dump(jd_payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for row in cvs:
        row["_id"] = _public_id("djh_cv_", row.get("id"))
        (OUT / "cvs" / f"{row['_id']}.txt").write_text(row["_text"], encoding="utf-8")

    manifest = _manifest(jds, cvs)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    pairs = [{"jd": jd["_id"], "cv": cv["_id"], "jd_text": jd["_text"], "cv_text": cv["_text"]}
             for jd in jds for cv in cvs]
    KITS.mkdir(parents=True, exist_ok=True)
    for index, annotator in enumerate(("A", "B")):
        shuffled = list(pairs)
        random.Random(SEED + 100 + index).shuffle(shuffled)
        (KITS / f"annotator_{annotator}.html").write_text(_label_html(shuffled, annotator), encoding="utf-8")
    return manifest


def verify() -> list[str]:
    if not MANIFEST.exists():
        return [f"missing manifest: {MANIFEST}"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = []
    checks = ((JD_SOURCE, manifest["sources"]["jobs"]["sha256"]),
              (CV_SOURCE, manifest["sources"]["profiles"]["sha256"]))
    for path, expected in checks:
        if not path.exists() or _sha(path) != expected:
            bad.append(str(path))
    if not (OUT / "jds.yaml").exists() or _tree_sha([OUT / "jds.yaml"]) != manifest["inputs"]["jds_tree_sha256"]:
        bad.append(str(OUT / "jds.yaml"))
    cv_paths = list((OUT / "cvs").glob("*.txt"))
    if len(cv_paths) != sum(CV_QUOTA.values()) or _tree_sha(cv_paths) != manifest["inputs"]["cvs_tree_sha256"]:
        bad.append(str(OUT / "cvs"))
    for annotator in ("A", "B"):
        if not (KITS / f"annotator_{annotator}.html").exists():
            bad.append(str(KITS / f"annotator_{annotator}.html"))
    return bad


def _read_human(path: Path) -> dict[tuple[str, str], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("schema") != "djinni-holdout-labels-v1" or payload.get("human") is not True
            or payload.get("independent") is not True or payload.get("saw_system_output") is not False):
        raise ValueError(f"{path}: provenance must say human=true, independent=true, saw_system_output=false")
    if payload.get("manifest_sha256") != _sha(MANIFEST):
        raise ValueError(f"{path}: labels were not exported from the frozen holdout manifest")
    out = {}
    for row in payload.get("labels", []):
        grade = row.get("grade")
        if grade not in (0, 1, 2, 3):
            raise ValueError(f"{path}: invalid/missing grade")
        key = (row.get("jd"), row.get("cv"))
        if key in out:
            raise ValueError(f"{path}: duplicate {key}")
        out[key] = row
    return out


def import_labels(a_path: Path, b_path: Path, adjudicated: Path | None = None) -> dict:
    bad = verify()
    if bad:
        raise ValueError("holdout integrity failed: " + ", ".join(bad))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {(j, c) for j in manifest["ids"]["jds"] for c in manifest["ids"]["cvs"]}
    a, b = _read_human(a_path), _read_human(b_path)
    if set(a) != expected or set(b) != expected:
        raise ValueError(f"each annotator must cover exactly {len(expected)} frozen pairs")
    av, bv = [a[k]["grade"] for k in sorted(expected)], [b[k]["grade"] for k in sorted(expected)]
    kappa = weighted_kappa(av, bv)
    if kappa < manifest["annotation"]["minimum_weighted_kappa"]:
        raise ValueError(f"weighted kappa {kappa:.3f} < 0.600: retrain annotators; Gold not created")
    conflicts = {key for key in expected if abs(a[key]["grade"] - b[key]["grade"]) >= 2}
    adjudication = _read_human(adjudicated) if adjudicated else {}
    if conflicts - set(adjudication):
        conflict_path = KITS / "conflicts.json"
        conflict_path.write_text(json.dumps([{"jd": j, "cv": c, "A": a[(j, c)]["grade"], "B": b[(j, c)]["grade"]}
                                             for j, c in sorted(conflicts)], indent=1), encoding="utf-8")
        raise ValueError(f"{len(conflicts)} conflicts differ by >=2; adjudicate them first: {conflict_path}")
    labels = []
    for jd, cv in sorted(expected):
        ga, gb = a[(jd, cv)]["grade"], b[(jd, cv)]["grade"]
        grade = float(adjudication[(jd, cv)]["grade"]) if (jd, cv) in conflicts else (ga + gb) / 2
        labels.append({"jd": jd, "cv": cv, "grade": grade, "annotator_a": ga, "annotator_b": gb,
                       "min": min(ga, gb), "max": max(ga, gb), "both_relevant": ga >= 2 and gb >= 2})
    exact = sum(x == y for x, y in zip(av, bv)) / len(av)
    payload = {"schema": "djinni-holdout-human-gold-v1", "manifest_sha256": _sha(MANIFEST),
               "provenance": {"human_annotators": 2, "independent": True, "blind": True,
                              "saw_system_output": False, "exact_agreement": exact,
                              "quadratic_weighted_kappa": kappa, "adjudicated_conflicts": len(conflicts)},
               "labels": labels}
    GOLD.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload["provenance"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "import-labels"))
    parser.add_argument("a", nargs="?", type=Path)
    parser.add_argument("b", nargs="?", type=Path)
    parser.add_argument("--adjudicated", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), indent=1, ensure_ascii=False))
    elif args.command == "verify":
        bad = verify()
        print("holdout intact" if not bad else "\n".join(bad))
        raise SystemExit(bool(bad))
    else:
        if not args.a or not args.b:
            parser.error("import-labels requires A.json B.json")
        print(json.dumps(import_labels(args.a, args.b, args.adjudicated), indent=1))


if __name__ == "__main__":
    main()
