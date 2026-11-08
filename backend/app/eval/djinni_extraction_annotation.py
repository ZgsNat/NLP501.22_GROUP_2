"""Blind human annotation workflow for the frozen Djinni extraction holdout.

This module deliberately contains no extraction or adjudication heuristics.  It
creates two independently ordered offline annotation kits, validates exported
human files fail-closed, measures pre-adjudication agreement, and creates a
conflict-only kit for a third human.  A Human Gold file can only be built after
all conflicts have a human decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.config import ROOT, get_settings

SCHEMA_VERSION = 2
ALIGNMENT_TOKEN_IOU = 0.5
VALID_STATUS = {"used", "listed", "hedged"}
MANIFEST = ROOT / "data" / "eval" / "djinni_extraction_benchmark_manifest.json"
GUIDELINES = ROOT / "docs" / "labeling" / "EXTRACTION_GUIDELINES.md"
TAXONOMY = ROOT / "data" / "skills_taxonomy.yaml"
CVS_DIR = get_settings().runtime_dir / "bench" / "djinni_extraction" / "cvs"
KIT_DIR = get_settings().runtime_dir / "bench" / "djinni_extraction" / "labeling"
GOLD_PATH = ROOT / "data" / "eval" / "djinni_extraction_human_gold_50.json"
STATUS_PATH = ROOT / "data" / "eval" / "djinni_extraction_status.json"


class AnnotationError(ValueError):
    """Raised when an annotation artifact violates a registered gate."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _sha_text(text: str) -> str:
    return _sha_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hashes() -> dict[str, str]:
    return {
        "manifest_sha256": _sha_file(MANIFEST),
        "guidelines_sha256": _sha_file(GUIDELINES),
        "taxonomy_sha256": _sha_file(TAXONOMY),
    }


def _taxonomy_keys() -> list[str]:
    payload = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    return sorted(payload["skills"])


def _documents() -> dict[str, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    docs: dict[str, dict] = {}
    for item in manifest["profiles"]:
        if item["is_pilot"]:
            continue
        path = CVS_DIR / f"{item['candidate_id']}.txt"
        if not path.exists():
            raise AnnotationError(f"Missing frozen CV: {path}")
        raw_text = path.read_bytes().decode("utf-8")
        text = raw_text.replace("\r\n", "\n")
        if _sha_text(text) != item["sha256"]:
            raise AnnotationError(f"Frozen CV hash mismatch: {item['candidate_id']}")
        if len(raw_text) != item["text_length"]:
            raise AnnotationError(f"Frozen CV length mismatch: {item['candidate_id']}")
        docs[item["candidate_id"]] = {**item, "text": text}
    if len(docs) != manifest["n_test"]:
        raise AnnotationError("Manifest test count does not match frozen documents")
    return docs


def _ordered_ids(ids: list[str], role: str) -> list[str]:
    # Stable, role-specific ordering; neither annotator sees the other's order.
    return sorted(ids, key=lambda cid: hashlib.sha256(f"{role}:{cid}".encode()).hexdigest())


def _kit_payload(role: str) -> dict:
    docs = _documents()
    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "blind_human_extraction_kit",
            "annotator_role": role,
            "human_required": True,
            "independent_required": True,
            "saw_system_output_required": False,
            "alignment_token_iou": ALIGNMENT_TOKEN_IOU,
            **_hashes(),
        },
        "taxonomy": _taxonomy_keys(),
        "profiles": [
            {
                "candidate_id": cid,
                "document_sha256": docs[cid]["sha256"],
                "domain": docs[cid]["domain"],
                "text": docs[cid]["text"],
            }
            for cid in _ordered_ids(list(docs), role)
        ],
    }


ANNOTATION_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind Djinni extraction annotation</title>
<style>
body{font:15px system-ui;margin:0;background:#f4f5f7;color:#18212b}.top{position:sticky;top:0;background:#17212b;color:white;padding:12px 18px;z-index:2}.grid{display:grid;grid-template-columns:minmax(480px,2fr) minmax(330px,1fr);gap:16px;padding:16px}.card{background:white;border:1px solid #d7dce2;border-radius:8px;padding:14px}.cv{white-space:pre-wrap;line-height:1.55;min-height:70vh}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}button,input,select{font:inherit;padding:7px}button{cursor:pointer}.mention{border-top:1px solid #e4e7eb;padding:9px 0}.bad{color:#9b1c1c}.ok{color:#167044}.muted{color:#64707d;font-size:13px}label{display:block;margin:7px 0}textarea{width:100%}</style></head>
<body><div class="top"><b>Blind extraction annotation</b> <span id="role"></span> · <span id="progress"></span></div>
<div class="grid"><section class="card"><div class="row"><button id="prev">Previous</button><button id="next">Next</button><button id="complete">Mark profile complete</button></div><h2 id="cid"></h2><pre id="cv" class="cv"></pre></section>
<aside class="card"><label>Annotator ID <input id="annotator" autocomplete="off" placeholder="real team member ID"></label><p class="muted">Select an exact skill mention in the CV, then add it. Do not consult system outputs or another annotator.</p><button id="add">Add selected span</button><div id="form" hidden><label>Raw span <input id="raw" readonly></label><label>Canonical skill <select id="canonical"></select></label><label>OOV name <input id="oov" placeholder="required when canonical is OOV"></label><label>Status <select id="status"><option>used</option><option>listed</option><option>hedged</option></select></label><label><input type="checkbox" id="confirmed" checked> Candidate genuinely has this skill</label><button id="save">Save mention</button><button id="cancel">Cancel</button></div><h3>Mentions</h3><div id="mentions"></div><hr><button id="export">Export JSON</button><p id="message"></p></aside></div>
<script>const KIT=__KIT__; const KEY='djinni-extraction-'+KIT.metadata.annotator_role+'-'+KIT.metadata.manifest_sha256; let state=JSON.parse(localStorage.getItem(KEY)||'null')||{i:0,annotator_id:'',started_at:new Date().toISOString(),profiles:{}}; let selected=null;
const $=id=>document.getElementById(id), save=()=>localStorage.setItem(KEY,JSON.stringify(state));
function current(){return KIT.profiles[state.i]} function profile(){let p=current(); return state.profiles[p.candidate_id]||(state.profiles[p.candidate_id]={document_sha256:p.document_sha256,completed:false,mentions:[]})}
function render(){let p=current(),a=profile(); $('role').textContent='Role '+KIT.metadata.annotator_role;$('progress').textContent=(state.i+1)+' / '+KIT.profiles.length;$('cid').textContent=p.candidate_id;$('cv').textContent=p.text;$('annotator').value=state.annotator_id;$('complete').textContent=a.completed?'Completed ✓':'Mark profile complete';$('mentions').innerHTML='';a.mentions.forEach((m,i)=>{let d=document.createElement('div');d.className='mention';d.textContent=`[${m.start}, ${m.end}) “${m.raw_text}” → ${m.canonical_skill||'OOV:'+m.oov_name} · ${m.evidence_status} · candidate=${m.candidate_confirmed}`;let b=document.createElement('button');b.textContent='Delete';b.onclick=()=>{a.mentions.splice(i,1);save();render()};d.append(' ',b);$('mentions').appendChild(d)});save()}
$('annotator').oninput=e=>{state.annotator_id=e.target.value.trim();save()};$('prev').onclick=()=>{state.i=Math.max(0,state.i-1);render()};$('next').onclick=()=>{state.i=Math.min(KIT.profiles.length-1,state.i+1);render()};$('complete').onclick=()=>{profile().completed=!profile().completed;render()};
$('add').onclick=()=>{let s=getSelection();if(!s||s.rangeCount!==1||s.isCollapsed){alert('Select an exact span in the CV first.');return}let r=s.getRangeAt(0);if(!$('cv').contains(r.commonAncestorContainer)){alert('Selection must be inside the CV.');return}let before=document.createRange();before.selectNodeContents($('cv'));before.setEnd(r.startContainer,r.startOffset);let start=before.toString().length,end=start+r.toString().length;selected={start,end,raw_text:current().text.slice(start,end)};$('raw').value=selected.raw_text;$('form').hidden=false};
$('cancel').onclick=()=>{$('form').hidden=true;selected=null};$('canonical').innerHTML='<option value="">OOV / not in taxonomy</option>'+KIT.taxonomy.map(x=>`<option>${x}</option>`).join('');
$('save').onclick=()=>{if(!selected)return;let canonical=$('canonical').value||null,oov=$('oov').value.trim();if(!canonical&&!oov){alert('Enter an OOV name.');return}profile().mentions.push({...selected,canonical_skill:canonical,oov_name:canonical?null:oov,evidence_status:$('status').value,candidate_confirmed:$('confirmed').checked});profile().mentions.sort((a,b)=>a.start-b.start||a.end-b.end);selected=null;$('form').hidden=true;$('oov').value='';save();render()};
$('export').onclick=()=>{let missing=KIT.profiles.filter(p=>!state.profiles[p.candidate_id]?.completed);if(!state.annotator_id){alert('Annotator ID is required.');return}if(missing.length){alert(missing.length+' profiles are not marked complete.');return}let out={metadata:{schema_version:2,artifact_type:'human_extraction_annotation',human:true,independent:true,saw_system_output:false,annotator_id:state.annotator_id,annotator_role:KIT.metadata.annotator_role,started_at:state.started_at,completed_at:new Date().toISOString(),manifest_sha256:KIT.metadata.manifest_sha256,guidelines_sha256:KIT.metadata.guidelines_sha256,taxonomy_sha256:KIT.metadata.taxonomy_sha256},profiles:state.profiles};let blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human_extraction_'+KIT.metadata.annotator_role.toLowerCase()+'.json';a.click();URL.revokeObjectURL(a.href)};render();</script></body></html>"""


def prepare_kits() -> list[Path]:
    KIT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for role in ("A", "B"):
        payload = _kit_payload(role)
        rendered = ANNOTATION_HTML.replace("__KIT__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
        path = KIT_DIR / f"djinni_extraction_annotator_{role.lower()}.html"
        path.write_text(rendered, encoding="utf-8")
        outputs.append(path)
    return outputs


def _validate_mention(mention: dict, text: str, taxonomy: set[str], where: str) -> None:
    required = {"start", "end", "raw_text", "canonical_skill", "oov_name", "evidence_status", "candidate_confirmed"}
    if set(mention) != required:
        raise AnnotationError(f"{where}: mention fields must be exactly {sorted(required)}")
    start, end = mention["start"], mention["end"]
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
        raise AnnotationError(f"{where}: invalid [start,end) offsets")
    if text[start:end] != mention["raw_text"]:
        raise AnnotationError(f"{where}: raw_text does not match frozen document offsets")
    canonical = mention["canonical_skill"]
    if canonical is not None and canonical not in taxonomy:
        raise AnnotationError(f"{where}: unknown canonical_skill {canonical!r}")
    if canonical is None and not str(mention["oov_name"] or "").strip():
        raise AnnotationError(f"{where}: OOV mention requires oov_name")
    if canonical is not None and mention["oov_name"] is not None:
        raise AnnotationError(f"{where}: in-taxonomy mention must have oov_name=null")
    if mention["evidence_status"] not in VALID_STATUS:
        raise AnnotationError(f"{where}: invalid evidence_status")
    if not isinstance(mention["candidate_confirmed"], bool):
        raise AnnotationError(f"{where}: candidate_confirmed must be boolean")


def validate_annotation(path: Path, expected_role: str | None = None) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("artifact_type") != "human_extraction_annotation":
        raise AnnotationError(f"{path}: unsupported annotation schema")
    for key, expected in (("human", True), ("independent", True), ("saw_system_output", False)):
        if meta.get(key) is not expected:
            raise AnnotationError(f"{path}: {key} must be {expected!r}")
    if not str(meta.get("annotator_id", "")).strip():
        raise AnnotationError(f"{path}: annotator_id is required")
    if expected_role and meta.get("annotator_role") != expected_role:
        raise AnnotationError(f"{path}: expected annotator role {expected_role}")
    for key, expected in _hashes().items():
        if meta.get(key) != expected:
            raise AnnotationError(f"{path}: {key} does not match the frozen protocol")
    if any(k in meta for k in ("generated_by", "simulated", "model", "heuristic_adjudication")):
        raise AnnotationError(f"{path}: generated/model annotations cannot enter Human Gold")

    docs, taxonomy = _documents(), set(_taxonomy_keys())
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(docs):
        raise AnnotationError(f"{path}: annotation must cover exactly all {len(docs)} frozen test profiles")
    for cid, item in profiles.items():
        if item.get("document_sha256") != docs[cid]["sha256"] or item.get("completed") is not True:
            raise AnnotationError(f"{path}: incomplete or wrong document provenance for {cid}")
        mentions = item.get("mentions")
        if not isinstance(mentions, list):
            raise AnnotationError(f"{path}: mentions must be a list for {cid}")
        seen = set()
        for idx, mention in enumerate(mentions):
            _validate_mention(mention, docs[cid]["text"], taxonomy, f"{path}:{cid}:{idx}")
            key = (mention["start"], mention["end"], mention["canonical_skill"], mention["oov_name"])
            if key in seen:
                raise AnnotationError(f"{path}:{cid}: duplicate mention {key}")
            seen.add(key)
    return data


def validate_pair(path_a: Path, path_b: Path) -> tuple[dict, dict]:
    a, b = validate_annotation(path_a, "A"), validate_annotation(path_b, "B")
    if a["metadata"]["annotator_id"].casefold() == b["metadata"]["annotator_id"].casefold():
        raise AnnotationError("Annotators A and B must be distinct people")
    return a, b


def _token_set(text: str, mention: dict) -> set[int]:
    return {i for i, m in enumerate(re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)) if m.start() < mention["end"] and m.end() > mention["start"]}


def _token_iou(text: str, left: dict, right: dict) -> float:
    a, b = _token_set(text, left), _token_set(text, right)
    return len(a & b) / len(a | b) if a | b else 0.0


def _align(text: str, left: list[dict], right: list[dict]) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates = []
    for ai, a in enumerate(left):
        for bi, b in enumerate(right):
            score = _token_iou(text, a, b)
            if score >= ALIGNMENT_TOKEN_IOU:
                candidates.append((score, ai, bi))
    used_a, used_b, aligned = set(), set(), []
    for score, ai, bi in sorted(candidates, reverse=True):
        if ai not in used_a and bi not in used_b:
            aligned.append((ai, bi, score)); used_a.add(ai); used_b.add(bi)
    return aligned, used_a, used_b


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def _kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        return None
    n = len(left); po = sum(a == b for a, b in zip(left, right)) / n
    ca, cb = Counter(left), Counter(right)
    pe = sum(ca[x] / n * cb[x] / n for x in VALID_STATUS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _skill_key(mention: dict) -> str:
    return mention["canonical_skill"] or "oov:" + re.sub(r"\s+", " ", mention["oov_name"].strip().casefold())


def compute_iaa(path_a: Path, path_b: Path) -> tuple[dict, list[dict]]:
    a, b = validate_pair(path_a, path_b); docs = _documents()
    totals = {"a": 0, "b": 0, "overlap": 0, "exact": 0, "canonical_equal": 0, "oov_equal": 0}
    statuses_a, statuses_b, macro_f1, jaccards, conflicts = [], [], [], [], []
    cand_tp = cand_fp = cand_fn = 0
    for cid in docs:
        ma, mb = a["profiles"][cid]["mentions"], b["profiles"][cid]["mentions"]
        aligned, used_a, used_b = _align(docs[cid]["text"], ma, mb)
        totals["a"] += len(ma); totals["b"] += len(mb); totals["overlap"] += len(aligned)
        for ai, bi, iou in aligned:
            x, y = ma[ai], mb[bi]
            exact = x["start"] == y["start"] and x["end"] == y["end"]
            canon_equal = _skill_key(x) == _skill_key(y)
            totals["exact"] += int(exact); totals["canonical_equal"] += int(canon_equal)
            totals["oov_equal"] += int((x["canonical_skill"] is None) == (y["canonical_skill"] is None))
            statuses_a.append(x["evidence_status"]); statuses_b.append(y["evidence_status"])
            if not (exact and canon_equal and x["evidence_status"] == y["evidence_status"] and x["candidate_confirmed"] == y["candidate_confirmed"]):
                conflicts.append({"candidate_id": cid, "kind": "aligned_disagreement", "a": x, "b": y, "token_iou": iou})
        for ai in set(range(len(ma))) - used_a:
            conflicts.append({"candidate_id": cid, "kind": "a_only", "a": ma[ai], "b": None, "token_iou": 0.0})
        for bi in set(range(len(mb))) - used_b:
            conflicts.append({"candidate_id": cid, "kind": "b_only", "a": None, "b": mb[bi], "token_iou": 0.0})
        sa = {_skill_key(x) for x in ma if x["candidate_confirmed"]}
        sb = {_skill_key(x) for x in mb if x["candidate_confirmed"]}
        inter, union = len(sa & sb), len(sa | sb)
        cand_tp += inter; cand_fp += len(sb - sa); cand_fn += len(sa - sb)
        jaccards.append(inter / union if union else 1.0)
        macro_f1.append(_prf(inter, len(sb - sa), len(sa - sb))["f1"])
    for idx, conflict in enumerate(conflicts, 1):
        conflict["conflict_id"] = f"C{idx:04d}"
    overlap = _prf(totals["overlap"], totals["b"] - totals["overlap"], totals["a"] - totals["overlap"])
    exact = _prf(totals["exact"], totals["b"] - totals["exact"], totals["a"] - totals["exact"])
    candidate = _prf(cand_tp, cand_fp, cand_fn)
    metrics = {
        "status": "HUMAN_IAA_MEASURED_ADJUDICATION_PENDING" if conflicts else "HUMAN_IAA_MEASURED_NO_CONFLICTS",
        "registered_alignment_token_iou": ALIGNMENT_TOKEN_IOU,
        "n_profiles": len(docs), "spans_a": totals["a"], "spans_b": totals["b"],
        "span_exact": exact, "span_overlap": overlap,
        "canonical_agreement_on_aligned": totals["canonical_equal"] / totals["overlap"] if totals["overlap"] else None,
        "status_unweighted_cohen_kappa": _kappa(statuses_a, statuses_b),
        "candidate_micro": candidate,
        "candidate_macro_f1": sum(macro_f1) / len(macro_f1),
        "candidate_mean_jaccard": sum(jaccards) / len(jaccards),
        "oov_type_agreement_on_aligned": totals["oov_equal"] / totals["overlap"] if totals["overlap"] else None,
        "oov_rate_a": sum(x["canonical_skill"] is None for p in a["profiles"].values() for x in p["mentions"]) / max(1, totals["a"]),
        "oov_rate_b": sum(x["canonical_skill"] is None for p in b["profiles"].values() for x in p["mentions"]) / max(1, totals["b"]),
        "n_conflicts": len(conflicts), "measured_at": _utc_now(), **_hashes(),
    }
    return metrics, conflicts


CONFLICT_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><style>body{font:15px system-ui;max-width:1100px;margin:auto;padding:20px}.c{border:1px solid #bbb;padding:14px;margin:14px 0}.cv{white-space:pre-wrap;background:#f5f5f5;padding:10px}textarea{width:100%;min-height:90px}</style></head><body><h1>Human adjudication of extraction conflicts</h1><label>Adjudicator ID <input id="id"></label><p>Review the CV context and both independent decisions. Enter the final mention list as JSON for each conflict. Use [] when neither mention belongs in Gold.</p><div id="list"></div><button id="export">Export adjudication JSON</button><script>const P=__PAYLOAD__,K='dj-ext-adjudication-'+P.metadata.manifest_sha256;let S=JSON.parse(localStorage.getItem(K)||'null')||{started_at:new Date().toISOString(),decisions:{}};const esc=s=>s.replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]));document.getElementById('list').innerHTML=P.conflicts.map(c=>`<div class=c><h3>${c.conflict_id} · ${c.candidate_id} · ${c.kind}</h3><pre class=cv>${esc(c.context)}</pre><b>A</b><pre>${esc(JSON.stringify(c.a,null,2))}</pre><b>B</b><pre>${esc(JSON.stringify(c.b,null,2))}</pre><label>Final mentions JSON<textarea data-id="${c.conflict_id}">${esc(JSON.stringify(S.decisions[c.conflict_id]?.final_mentions??(c.a?[c.a]:c.b?[c.b]:[]),null,2))}</textarea></label></div>`).join('');document.querySelectorAll('textarea').forEach(t=>t.oninput=()=>{try{S.decisions[t.dataset.id]={final_mentions:JSON.parse(t.value)};localStorage.setItem(K,JSON.stringify(S));t.style.borderColor='green'}catch(e){t.style.borderColor='red'}});document.getElementById('export').onclick=()=>{let aid=document.getElementById('id').value.trim();if(!aid)return alert('Adjudicator ID required');document.querySelectorAll('textarea').forEach(t=>t.dispatchEvent(new Event('input')));if(Object.keys(S.decisions).length!==P.conflicts.length)return alert('Every conflict needs a valid decision');let out={metadata:{schema_version:2,artifact_type:'human_extraction_adjudication',human:true,saw_system_output:false,adjudicator_id:aid,started_at:S.started_at,completed_at:new Date().toISOString(),manifest_sha256:P.metadata.manifest_sha256,guidelines_sha256:P.metadata.guidelines_sha256,taxonomy_sha256:P.metadata.taxonomy_sha256,annotation_a_sha256:P.metadata.annotation_a_sha256,annotation_b_sha256:P.metadata.annotation_b_sha256},decisions:S.decisions};let b=new Blob([JSON.stringify(out,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='human_extraction_adjudication.json';a.click()};</script></body></html>"""


def write_iaa_and_conflicts(path_a: Path, path_b: Path, out_dir: Path) -> tuple[Path, Path]:
    metrics, conflicts = compute_iaa(path_a, path_b); docs = _documents(); out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "human_extraction_iaa.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    payload = {"metadata": {**_hashes(), "annotation_a_sha256": _sha_file(path_a), "annotation_b_sha256": _sha_file(path_b)}, "conflicts": []}
    for c in conflicts:
        mentions = [x for x in (c["a"], c["b"]) if x]
        lo = max(0, min(x["start"] for x in mentions) - 180); hi = min(len(docs[c["candidate_id"]]["text"]), max(x["end"] for x in mentions) + 180)
        payload["conflicts"].append({**c, "context": docs[c["candidate_id"]]["text"][lo:hi], "context_start": lo})
    conflicts_path = out_dir / "djinni_extraction_conflicts.html"
    conflicts_path.write_text(CONFLICT_HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    return metrics_path, conflicts_path


def build_gold(path_a: Path, path_b: Path, adjudication_path: Path) -> Path:
    if GOLD_PATH.exists():
        raise AnnotationError(f"Refusing to overwrite frozen Human Gold: {GOLD_PATH}")
    a, b = validate_pair(path_a, path_b); metrics, conflicts = compute_iaa(path_a, path_b)
    adj = json.loads(adjudication_path.read_text(encoding="utf-8")); meta = adj.get("metadata", {})
    if meta.get("artifact_type") != "human_extraction_adjudication" or meta.get("human") is not True or meta.get("saw_system_output") is not False:
        raise AnnotationError("Adjudication must be a blind human artifact")
    if meta.get("adjudicator_id", "").casefold() in {a["metadata"]["annotator_id"].casefold(), b["metadata"]["annotator_id"].casefold()}:
        raise AnnotationError("Adjudicator must be a third person")
    for key, expected in {**_hashes(), "annotation_a_sha256": _sha_file(path_a), "annotation_b_sha256": _sha_file(path_b)}.items():
        if meta.get(key) != expected:
            raise AnnotationError(f"Adjudication provenance mismatch: {key}")
    decisions = adj.get("decisions", {}); expected_ids = {c["conflict_id"] for c in conflicts}
    if set(decisions) != expected_ids:
        raise AnnotationError("Adjudication must cover every and only registered conflict")
    docs, taxonomy = _documents(), set(_taxonomy_keys()); conflict_lookup = {c["conflict_id"]: c for c in conflicts}
    final = {cid: [] for cid in docs}
    for cid in docs:
        ma, mb = a["profiles"][cid]["mentions"], b["profiles"][cid]["mentions"]
        aligned, _, _ = _align(docs[cid]["text"], ma, mb)
        for ai, bi, _ in aligned:
            x, y = ma[ai], mb[bi]
            if x == y:
                final[cid].append(x)
    for conflict_id, decision in decisions.items():
        cid = conflict_lookup[conflict_id]["candidate_id"]
        mentions = decision.get("final_mentions")
        if not isinstance(mentions, list):
            raise AnnotationError(f"{conflict_id}: final_mentions must be a list")
        for idx, mention in enumerate(mentions):
            _validate_mention(mention, docs[cid]["text"], taxonomy, f"{conflict_id}:{idx}")
            final[cid].append(mention)
    profiles = {}
    for cid, mentions in final.items():
        unique = {(m["start"], m["end"], _skill_key(m), m["evidence_status"], m["candidate_confirmed"]): m for m in mentions}
        ordered = sorted(unique.values(), key=lambda m: (m["start"], m["end"], _skill_key(m)))
        profiles[cid] = {
            "document_sha256": docs[cid]["sha256"],
            "domain": docs[cid]["domain"],
            "length_stratum": docs[cid]["length_stratum"],
            "mentions": ordered,
            "candidate_skills": sorted({_skill_key(m) for m in ordered if m["candidate_confirmed"]}),
        }
    gold = {"metadata": {"schema_version": SCHEMA_VERSION, "artifact_type": "human_adjudicated_extraction_gold", "gold_status": "HUMAN_ADJUDICATED_FROZEN", "created_at": _utc_now(), "annotation_a_sha256": _sha_file(path_a), "annotation_b_sha256": _sha_file(path_b), "adjudication_sha256": _sha_file(adjudication_path), "iaa": metrics, **_hashes()}, "profiles": profiles}
    temporary = GOLD_PATH.with_suffix(".json.pending")
    try:
        temporary.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
        validate_gold(temporary)
        temporary.replace(GOLD_PATH)
    finally:
        temporary.unlink(missing_ok=True)
    return GOLD_PATH


def validate_gold(path: Path) -> dict:
    """Validate a frozen Gold artifact without trusting its status string alone."""
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise AnnotationError("Human Gold schema version mismatch")
    if meta.get("artifact_type") != "human_adjudicated_extraction_gold":
        raise AnnotationError("Primary reference is not human-adjudicated extraction Gold")
    if meta.get("gold_status") != "HUMAN_ADJUDICATED_FROZEN":
        raise AnnotationError("Human Gold is not frozen")
    for key, expected in _hashes().items():
        if meta.get(key) != expected:
            raise AnnotationError(f"Human Gold protocol mismatch: {key}")
    for key in ("annotation_a_sha256", "annotation_b_sha256", "adjudication_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get(key, ""))):
            raise AnnotationError(f"Human Gold is missing provenance hash: {key}")
    iaa = meta.get("iaa", {})
    if not str(iaa.get("status", "")).startswith("HUMAN_IAA_MEASURED"):
        raise AnnotationError("Human Gold is missing validated pre-adjudication IAA")

    docs, taxonomy = _documents(), set(_taxonomy_keys())
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != set(docs):
        raise AnnotationError("Human Gold must cover exactly the frozen test profiles")
    for cid, item in profiles.items():
        if item.get("document_sha256") != docs[cid]["sha256"]:
            raise AnnotationError(f"Human Gold document hash mismatch: {cid}")
        mentions = item.get("mentions")
        if not isinstance(mentions, list):
            raise AnnotationError(f"Human Gold mentions must be a list: {cid}")
        for idx, mention in enumerate(mentions):
            _validate_mention(mention, docs[cid]["text"], taxonomy, f"gold:{cid}:{idx}")
        expected_skills = sorted({_skill_key(m) for m in mentions if m["candidate_confirmed"]})
        if item.get("candidate_skills") != expected_skills:
            raise AnnotationError(f"Human Gold candidate skill aggregate mismatch: {cid}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare-kits")
    verify = sub.add_parser("verify"); verify.add_argument("path", type=Path); verify.add_argument("--role", choices=["A", "B"])
    iaa = sub.add_parser("iaa"); iaa.add_argument("a", type=Path); iaa.add_argument("b", type=Path); iaa.add_argument("--out-dir", type=Path, default=KIT_DIR)
    gold = sub.add_parser("build-gold"); gold.add_argument("a", type=Path); gold.add_argument("b", type=Path); gold.add_argument("adjudication", type=Path)
    args = parser.parse_args()
    if args.cmd == "prepare-kits":
        for path in prepare_kits(): print(path)
    elif args.cmd == "verify":
        validate_annotation(args.path, args.role); print(f"VALID: {args.path}")
    elif args.cmd == "iaa":
        for path in write_iaa_and_conflicts(args.a, args.b, args.out_dir): print(path)
    elif args.cmd == "build-gold":
        print(build_gold(args.a, args.b, args.adjudication))


if __name__ == "__main__":
    main()
