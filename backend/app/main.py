"""FastAPI entrypoint:  uv run uvicorn app.main:app --app-dir backend --reload"""

import csv
import io
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from app import observability
from app.agent.graph import build_graph
from app.agent.orchestrator import build_agent_graph
from app.agent.state import public_view
from app.config import get_settings
from app.providers import cache
from app.providers.llm import PROVIDERS, default_spec, is_configured, parse_spec
from app.nlp.pipeline import process, profile
from app.nlp.text import read_file
from app.rag import retrieve
from app.rag.corpus import get_doc, load_dir, load_yaml, upload_dir
from app.rag.store import get_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.runtime_dir.mkdir(exist_ok=True)
    if s.llm_cache:
        cache.enable(s.runtime_dir / "llm_cache.sqlite")
    async with AsyncSqliteSaver.from_conn_string(str(s.runtime_dir / "checkpoints.sqlite")) as saver:
        app.state.graphs = {"pipeline": build_graph(saver), "agent": build_agent_graph(saver)}
        yield
    observability.flush()


app = FastAPI(title="CV Agent", lifespan=lifespan)


class RunRequest(BaseModel):
    jd_text: str
    company_profile: str = "default"
    models: dict[str, str] = {}
    mode: Literal["agent", "pipeline"] = "agent"  # agent = P7 orchestrator (an LLM picks the tools); pipeline = P6
    goal: str = ""  # agent mode: what HR wants, in their words


async def _graph_for(run_id: str):
    """Both graphs share one checkpointer; a P7 run is the one whose state has a goal."""
    state = await app.state.graphs["pipeline"].aget_state(_config(run_id))
    return app.state.graphs["agent" if "goal" in (state.values or {}) else "pipeline"]


def _config(run_id: str) -> dict:
    return {
        "configurable": {"thread_id": run_id},
        "callbacks": [*observability.callbacks(), observability.usage_counter(run_id)],
        "metadata": {"langfuse_session_id": run_id},
        "run_name": "cv-agent",
    }


async def _events(graph, inp, run_id: str) -> AsyncIterator[str]:
    def line(obj: dict) -> str:
        return json.dumps(obj, default=str) + "\n"

    try:
        async for mode, chunk in graph.astream(inp, _config(run_id), stream_mode=["updates", "custom"]):
            if mode == "custom":
                if chunk["type"] == "log":
                    observability.append_log(run_id, chunk)
                yield line(chunk)
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    value = update[0].value
                    yield line({"type": "interrupt", "kind": value["kind"], "payload": value["payload"]})
                elif update:
                    yield line({"type": "update", "node": node, "update": public_view(update)})
        state = await graph.aget_state(_config(run_id))
        if not state.next:
            yield line({"type": "done", "report": state.values.get("report"),
                        "usage": observability.usage_counter(run_id).summary(), "logs": observability.read_logs(run_id)})
    except Exception as e:  # surface to the UI instead of a silently dropped stream
        yield line({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        observability.save_usage(run_id)
        observability.flush()


def _stream(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="application/x-ndjson")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def config():
    s = get_settings()
    crit = load_yaml("criteria_profiles.yaml")
    return {
        "providers": [{"name": p, "configured": is_configured(p)} for p in PROVIDERS],
        "defaults": {"reasoning": default_spec("reasoning"), "fast": default_spec("fast")},
        "company_profiles": [{"key": k, "label": v["label"]} for k, v in crit["profiles"].items()],
        "criteria": [{"key": k, "label": v} for k, v in crit["criteria"].items()],
        "sample_jds": [{"id": d.id, "title": d.meta.get("title", d.id),
                        "company_profile": d.meta.get("company_profile", "default"), "text": d.text}
                       for d in load_dir("jds")],
        "langfuse_url": s.langfuse_base_url if observability.tracing_enabled() else None,
        "tiers": {"strong": s.tier_strong, "possible": s.tier_possible},
    }


@app.post("/api/runs")
async def start_run(req: RunRequest):
    for spec in req.models.values():
        try:
            if not is_configured(parse_spec(spec)[0]):
                raise ValueError(f"provider for {spec!r} is not configured")
        except ValueError as e:
            raise HTTPException(400, str(e))
    run_id = uuid.uuid4().hex[:12]
    inp = {"jd_text": req.jd_text, "company_profile": req.company_profile, "models": req.models}
    if req.mode == "agent":
        inp["goal"] = req.goal

    async def gen():
        yield json.dumps({"type": "run_started", "run_id": run_id, "mode": req.mode,
                          "trace_url": observability.session_url(run_id)}) + "\n"
        async for ev in _events(app.state.graphs[req.mode], inp, run_id):
            yield ev

    return _stream(gen())


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, body: dict):
    try:
        observability.run_dir(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    graph = await _graph_for(run_id)
    state = await graph.aget_state(_config(run_id))
    if not state.next:
        raise HTTPException(409, "Run is not waiting for input")
    return _stream(_events(graph, Command(resume=body), run_id))


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    try:
        observability.run_dir(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state = await (await _graph_for(run_id)).aget_state(_config(run_id))
    if not state.values:
        raise HTTPException(404, "Unknown run")
    pending = next((i.value for t in state.tasks for i in t.interrupts), None)
    return {"run_id": run_id, "mode": "agent" if "goal" in state.values else "pipeline",
            "values": public_view(state.values), "next": list(state.next), "interrupt": pending,
            "usage": observability.usage_counter(run_id).summary(), "logs": observability.read_logs(run_id)}


def _next_cv_id() -> str:
    existing = {d.id for d in load_dir("cvs") if d.id.startswith("UP-")}
    i = 1
    while f"UP-{i:03d}" in existing:
        i += 1
    return f"UP-{i:03d}"


def _save_uploaded_cv(cv_id: str, raw_text: str, filename: str):
    p = process(cv_id, raw_text)
    if len(p.text.split()) < 10:
        raise ValueError("Văn bản CV quá ngắn hoặc không đọc được nội dung (tối thiểu 10 từ). Nếu là ảnh chụp/scan cần cài đặt OCR.")
    headline = next((ln.strip("#* ") for ln in p.text.splitlines() if ln.strip()), "")[:80]
    p = process(cv_id, raw_text, headline)
    upload_dir().mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    content = f"---\nid: {cv_id}\nheadline: {json.dumps(headline)}\nfilename: {json.dumps(filename)}\nuploaded_at: {json.dumps(now)}\n---\n{p.text}\n"
    (upload_dir() / f"{cv_id}.md").write_text(content, encoding="utf-8")
    return {"id": cv_id, "headline": headline, "filename": filename, "uploaded_at": now}, p.text, p


@app.get("/api/cvs")
async def list_cvs():
    uploaded = {p.stem for p in upload_dir().glob("*.md")} if upload_dir().exists() else set()
    result = []
    for d in load_dir("cvs"):
        try:
            prof = profile(d.id)
            skills = sorted(list(prof.skills.keys()))
            years = prof.years_automation
            lang = prof.lang
        except Exception:
            skills = []
            years = 0.0
            lang = "vi"
        result.append({
            "id": d.id,
            "headline": d.meta.get("headline", ""),
            "filename": d.meta.get("filename", f"{d.id}.md"),
            "uploaded_at": d.meta.get("uploaded_at"),
            "lang": lang,
            "skills": skills,
            "years_automation": years,
            "uploaded": d.id in uploaded,
        })
    return result


@app.post("/api/cvs", status_code=201)
async def upload_cv(file: UploadFile):
    """PDF/DOCX/MD/TXT/PNG/JPG (scans and photos by OCR) → NLP pipeline → indexed."""
    try:
        raw = read_file(file.filename or "", await file.read())
        cv_id = _next_cv_id()
        meta, text, p = _save_uploaded_cv(cv_id, raw, file.filename or "")
    except ValueError as e:
        raise HTTPException(400, str(e))
    load_dir.cache_clear()
    profile.cache_clear()
    try:
        store = get_store()
        store.upsert_items(retrieve.WHOLE, [(cv_id, text, {"doc_id": cv_id, "text": text, "headline": meta["headline"]})])
        store.upsert_items(retrieve.CHUNKS, retrieve.chunk_items([p]))
    except Exception:
        # When vector store (Qdrant) is offline, markdown files and NLP profiles remain fully functional
        pass
    view = p.view()
    view["filename"] = meta["filename"]
    view["uploaded_at"] = meta["uploaded_at"]
    return view


@app.post("/api/cvs/batch", status_code=200)
async def batch_upload_cvs(files: list[UploadFile] = File(...)):
    """Upload multiple CV files at once."""
    succeeded = []
    failed = []
    new_profiles = []
    whole_items = []

    for f in files:
        fname = f.filename or "unknown"
        try:
            content_bytes = await f.read()
            raw = read_file(fname, content_bytes)
            cv_id = _next_cv_id()
            meta, text, p = _save_uploaded_cv(cv_id, raw, fname)
            # Clear cache so next iteration sees the newly allocated UP-xxx ID
            load_dir.cache_clear()
            succeeded.append({
                "id": cv_id,
                "filename": fname,
                "headline": meta["headline"],
                "skills": sorted(list(p.skills.keys())),
                "years_automation": p.years_automation,
                "lang": p.lang,
            })
            new_profiles.append(p)
            whole_items.append((cv_id, text, {"doc_id": cv_id, "text": text, "headline": meta["headline"]}))
        except Exception as e:
            failed.append({"filename": fname, "error": str(e)})

    if new_profiles:
        load_dir.cache_clear()
        profile.cache_clear()
        try:
            store = get_store()
            store.upsert_items(retrieve.WHOLE, whole_items)
            store.upsert_items(retrieve.CHUNKS, retrieve.chunk_items(new_profiles))
        except Exception:
            pass

    return {
        "total": len(files),
        "succeeded": len(succeeded),
        "failed": len(failed),
        "items": succeeded,
        "errors": failed,
    }


@app.post("/api/jds/parse", status_code=200)
async def parse_jd_endpoint(file: UploadFile = File(...)):
    """Upload a JD file (.docx, .pdf, .md, .txt) → NLP normalization → extract requirements using app.nlp.jd."""
    from app.nlp.jd import parse as parse_jd
    from app.nlp.text import normalize

    filename = file.filename or "jd.txt"
    try:
        raw = read_file(filename, await file.read())
    except Exception as e:
        raise HTTPException(400, f"Không thể đọc tệp JD: {e}")

    norm_text = normalize(raw)
    if len(norm_text.split()) < 10:
        raise HTTPException(400, "Nội dung tệp JD quá ngắn hoặc không đọc được (tối thiểu 10 từ).")

    parsed = parse_jd(norm_text)
    lines = [ln.strip("#* \t") for ln in norm_text.splitlines() if ln.strip()]
    title = lines[0][:80] if lines else "Automation Test Engineer"
    parsed["title"] = title
    parsed["filename"] = filename
    parsed["raw_text"] = norm_text
    return parsed


@app.delete("/api/cvs/{cv_id}")
async def delete_cv(cv_id: str):
    """Delete an uploaded CV (.runtime/cvs/UP-*.md). Built-in CVs cannot be deleted."""
    if not cv_id.startswith("UP-"):
        raise HTTPException(400, "Only uploaded CVs (UP-*) can be deleted")
    target = upload_dir() / f"{cv_id}.md"
    if not target.exists():
        raise HTTPException(404, f"CV {cv_id} not found in uploads")
    target.unlink()
    load_dir.cache_clear()
    profile.cache_clear()
    try:
        from qdrant_client.http import models as qmodels
        store = get_store()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, cv_id))
        store.client.delete(retrieve.WHOLE, points_selector=[point_id])
        store.client.delete(
            retrieve.CHUNKS,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(must=[qmodels.FieldCondition(key="doc_id", match=qmodels.MatchValue(value=cv_id))])
            ),
        )
    except Exception:
        pass
    return {"status": "deleted", "id": cv_id}


@app.post("/api/cvs/reset")
async def reset_uploaded_cvs():
    """Remove all user-uploaded CVs, returning the pool to pristine benchmark state."""
    deleted_count = 0
    if upload_dir().exists():
        for f in upload_dir().glob("*.md"):
            try:
                f.unlink()
                deleted_count += 1
            except Exception:
                pass
    load_dir.cache_clear()
    profile.cache_clear()
    return {"status": "reset", "deleted_count": deleted_count}


@app.get("/api/cvs/{cv_id}")
async def get_cv(cv_id: str):
    doc = get_doc("cvs", cv_id)
    if not doc:
        raise HTTPException(404, "Unknown CV")
    p = profile(cv_id)
    return {
        "id": doc.id,
        "headline": doc.meta.get("headline", ""),
        "filename": doc.meta.get("filename", f"{doc.id}.md"),
        "uploaded_at": doc.meta.get("uploaded_at"),
        "text": p.text,
        "profile": p.view(),
    }


@app.get("/api/runs/{run_id}/export/csv")
async def export_run_csv(run_id: str):
    """Export screening and ranking results as a formatted CSV with UTF-8 BOM."""
    try:
        graph = await _graph_for(run_id)
        state = await graph.aget_state(_config(run_id))
    except Exception as e:
        raise HTTPException(404, f"Cannot load run: {e}")
    if not state.values:
        raise HTTPException(404, "Unknown run")
    ranking = state.values.get("ranking") or []
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Hạng (Rank)",
        "Mã CV (ID)",
        "Tiêu đề (Headline)",
        "Điểm phù hợp (Score %)",
        "Xếp loại (Tier)",
        "Trạng thái (Status)",
        "Lý do phân loại (Reason)",
        "Ghi chú thẩm định (Arbiter Note)",
        "Cờ cảnh báo (Flags)",
    ])
    for r in ranking:
        total_pct = f"{round(float(r.get('total', 0)) * 100)}%"
        flags_str = ", ".join(r.get("flags") or [])
        writer.writerow([
            r.get("rank", ""),
            r.get("cv_id", ""),
            r.get("headline", ""),
            total_pct,
            r.get("tier", ""),
            r.get("status", ""),
            r.get("reason", ""),
            r.get("arbiter_note", ""),
            flags_str,
        ])
    bom_bytes = b"\xef\xbb\xbf" + output.getvalue().encode("utf-8")
    return Response(
        content=bom_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="ket_qua_tuyen_dung_{run_id}.csv"'},
    )
