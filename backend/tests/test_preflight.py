"""The demo preflight reports a down service as a blocker; it never crashes on one."""

from scripts import preflight


def test_unreachable_qdrant_and_ollama_are_blockers_not_crashes(monkeypatch):
    monkeypatch.setattr(preflight, "get", lambda url: None)
    monkeypatch.setattr(preflight, "results", [])
    monkeypatch.setattr(preflight.get_settings(), "qdrant_url", "http://localhost:1")
    preflight.qdrant()
    preflight.ollama(["qwen3:8b"])
    preflight.api()
    marks = [m for m, _, _ in preflight.results]
    assert marks == [preflight.FAIL, preflight.FAIL, preflight.WARN]
