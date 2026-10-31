"""FastAPI entrypoint:  uv run uvicorn app.main:app --app-dir backend --reload"""

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from app import observability
from app.agent.graph import build_graph
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
        app.state.graph = build_graph(saver)
        yield
    observability.flush()


app = FastAPI(title="CV Agent", lifespan=lifespan)


class RunRequest(BaseModel):
    jd_text: str
    company_profile: str = "default"
    models: dict[str, str] = {}


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

    async def gen():
        yield json.dumps({"type": "run_started", "run_id": run_id, "trace_url": observability.session_url(run_id)}) + "\n"
        async for ev in _events(app.state.graph, inp, run_id):
            yield ev

    return _stream(gen())


@app.post("/api/runs/{run_id}/resume")
async def resume_run(run_id: str, body: dict):
    try:
        observability.run_dir(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state = await app.state.graph.aget_state(_config(run_id))
    if not state.next:
        raise HTTPException(409, "Run is not waiting for input")
    return _stream(_events(app.state.graph, Command(resume=body), run_id))


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    try:
        observability.run_dir(run_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    state = await app.state.graph.aget_state(_config(run_id))
    if not state.values:
        raise HTTPException(404, "Unknown run")
    pending = next((i.value for t in state.tasks for i in t.interrupts), None)
    return {"run_id": run_id, "values": public_view(state.values), "next": list(state.next), "interrupt": pending,
            "usage": observability.usage_counter(run_id).summary(), "logs": observability.read_logs(run_id)}


@app.get("/api/cvs")
async def list_cvs():
    uploaded = {p.stem for p in upload_dir().glob("*.md")} if upload_dir().exists() else set()
    return [{"id": d.id, "headline": d.meta.get("headline", ""), "lang": profile(d.id).lang, "uploaded": d.id in uploaded}
            for d in load_dir("cvs")]


@app.post("/api/cvs", status_code=201)
async def upload_cv(file: UploadFile):
    """PDF/DOCX/MD/TXT/PNG/JPG (scans and photos by OCR) → NLP pipeline → indexed. Only the redacted text is stored; the raw file never touches disk."""
    try:
        raw = read_file(file.filename or "", await file.read())
    except ValueError as e:
        raise HTTPException(400, str(e))
    n = 1 + sum(1 for d in load_dir("cvs") if d.id.startswith("UP-"))
    cv_id = f"UP-{n:03d}"
    p = process(cv_id, raw)
    if len(p.text.split()) < 30:
        raise HTTPException(400, "No readable text found (scans and photos need OCR: uv sync --group ocr)")
    headline = next((ln.strip("#* ") for ln in p.text.splitlines() if ln.strip()), "")[:80]
    p = process(cv_id, raw, headline)
    upload_dir().mkdir(parents=True, exist_ok=True)
    (upload_dir() / f"{cv_id}.md").write_text(f"---\nid: {cv_id}\nheadline: {json.dumps(headline)}\n---\n{p.text}\n")
    load_dir.cache_clear()
    store = get_store()
    store.upsert_items(retrieve.WHOLE, [(cv_id, p.text, {"doc_id": cv_id, "text": p.text, "headline": headline})])
    store.upsert_items(retrieve.CHUNKS, retrieve.chunk_items([p]))
    return p.view()


@app.get("/api/cvs/{cv_id}")
async def get_cv(cv_id: str):
    doc = get_doc("cvs", cv_id)
    if not doc:
        raise HTTPException(404, "Unknown CV")
    p = profile(cv_id)
    return {"id": doc.id, "headline": doc.meta.get("headline", ""), "text": p.text, "profile": p.view()}
