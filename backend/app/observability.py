"""Langfuse tracing. Every run (and each resume) is one trace; traces of the same run share a session = run_id,
so the whole human-in-the-loop conversation is browsable as one session in the Langfuse UI."""

import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler

from app.config import get_settings


class UsageCounter(BaseCallbackHandler):
    """Counts LLM calls and tokens per graph node — the raw numbers behind any "% tokens saved" claim.
    Provider-agnostic: reads LangChain's standard `usage_metadata` (Ollama, OpenAI, Anthropic, Gemini all fill it)."""

    def __init__(self):
        self._node: dict = {}
        self.by_node: dict[str, dict[str, int]] = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0})

    def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kw):
        self._node[run_id] = (metadata or {}).get("langgraph_node", "?")

    def on_llm_end(self, response, *, run_id, **kw):
        row = self.by_node[self._node.pop(run_id, "?")]
        row["calls"] += 1
        for gen in (g for gens in response.generations for g in gens):
            usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
            row["input_tokens"] += usage.get("input_tokens", 0)
            row["output_tokens"] += usage.get("output_tokens", 0)

    def summary(self) -> dict:
        nodes = {k: dict(v) for k, v in self.by_node.items()}
        total = {f: sum(v[f] for v in nodes.values()) for f in ("calls", "input_tokens", "output_tokens")}
        return {**total, "total_tokens": total["input_tokens"] + total["output_tokens"], "by_node": nodes}


def run_dir(run_id: str, create: bool = False) -> Path:
    """Per-run artefacts (.runtime/runs/<id>/): narration log + usage — so a shared run link keeps its "why"."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):  # run_id comes from the URL: no path traversal
        raise ValueError(f"Bad run id {run_id!r}")
    d = get_settings().runtime_dir / "runs" / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


_usage: dict[str, UsageCounter] = {}


def usage_counter(run_id: str) -> UsageCounter:
    """One counter per run, shared across the start stream and every HR resume; reloaded from disk after a restart."""
    if run_id not in _usage:
        counter = UsageCounter()
        saved = run_dir(run_id) / "usage.json"
        if saved.exists():
            for node, row in json.loads(saved.read_text())["by_node"].items():
                counter.by_node[node].update(row)
        _usage[run_id] = counter
    return _usage[run_id]


def save_usage(run_id: str) -> None:
    (run_dir(run_id, create=True) / "usage.json").write_text(json.dumps(usage_counter(run_id).summary()))


def append_log(run_id: str, event: dict) -> None:
    with open(run_dir(run_id, create=True) / "events.ndjson", "a") as f:
        f.write(json.dumps(event) + "\n")


def read_logs(run_id: str) -> list[dict]:
    f = run_dir(run_id) / "events.ndjson"
    return [json.loads(ln) for ln in f.read_text().splitlines()] if f.exists() else []


@lru_cache
def tracing_enabled() -> bool:
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return False
    from langfuse import Langfuse

    Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, base_url=s.langfuse_base_url)
    return True


def callbacks() -> list:
    if not tracing_enabled():
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def session_url(run_id: str) -> str | None:
    return f"{get_settings().langfuse_base_url}/project/cv-agent/sessions/{run_id}" if tracing_enabled() else None


def flush() -> None:
    if tracing_enabled():
        from langfuse import get_client

        get_client().flush()
