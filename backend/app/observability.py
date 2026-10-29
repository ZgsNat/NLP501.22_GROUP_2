"""Langfuse tracing. Every run (and each resume) is one trace; traces of the same run share a session = run_id,
so the whole human-in-the-loop conversation is browsable as one session in the Langfuse UI."""

from functools import lru_cache

from app.config import get_settings


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
