"""Before a demo: is everything the live run needs up and configured? Read-only, no LLM call, a few seconds.

    cd backend && uv run python -m scripts.preflight          # exit 1 if anything blocks the demo

Checks: the model roles and their fallback (and the `claude` CLI for claude_code), Ollama and the models it must
serve, Qdrant and the indexed collections, the corpus (DATA_DIR), HR memory, and the API if it is already running.
"""

import shutil
import sys

import httpx

from app.config import get_settings
from app.providers.llm import default_spec, is_configured, parse_spec
from app.rag.corpus import load_dir
from app.rag.retrieve import CHUNKS, WHOLE

OK, WARN, FAIL = "✅", "⚠️ ", "❌"
results: list[tuple[str, str, str]] = []


def check(mark: str, what: str, detail: str = "") -> None:
    results.append((mark, what, detail))


def get(url: str) -> httpx.Response | None:
    try:
        return httpx.get(url, timeout=3)
    except httpx.HTTPError:
        return None


def models() -> list[str]:
    s = get_settings()
    specs = {"reasoning": default_spec("reasoning"), "fast": default_spec("fast")}
    if s.llm_fallback:
        specs["fallback"] = s.llm_fallback
    for role, spec in specs.items():
        provider, model = parse_spec(spec)
        if not is_configured(provider):
            check(FAIL, f"{role}: {spec}", f"provider '{provider}' has no key in .env")
        elif provider == "claude_code" and not shutil.which("claude"):
            check(FAIL, f"{role}: {spec}", "the `claude` CLI is not on PATH (log in with a subscription first)")
        else:
            check(OK, f"{role}: {spec}")
    return [parse_spec(sp)[1] for sp in specs.values() if parse_spec(sp)[0] == "ollama"]


def ollama(needed: list[str]) -> None:
    s = get_settings()
    if s.injection_model:
        needed = [*needed, s.system_one_model]
    if not needed:
        check(OK, "Ollama", "not needed by this configuration")
        return
    r = get(f"{s.ollama_base_url}/api/tags")
    if r is None or r.status_code != 200:
        check(FAIL, "Ollama", f"not reachable at {s.ollama_base_url}: start the Ollama app (needed for {needed})")
        return
    have = {m["name"].removesuffix(":latest") for m in r.json().get("models", [])}
    missing = [m for m in needed if m.removesuffix(":latest") not in have]
    check(FAIL if missing else OK, "Ollama models", f"missing {missing}: `ollama pull …`" if missing else ", ".join(needed))


def qdrant() -> None:
    s = get_settings()
    if s.qdrant_url == ":memory:":
        check(WARN, "Qdrant", "in-memory index: nothing persists between restarts")
        return
    r = get(f"{s.qdrant_url}/collections")
    if r is None or r.status_code != 200:
        check(FAIL, "Qdrant", f"not reachable at {s.qdrant_url}: `docker compose up -d qdrant` (Docker Desktop on)")
        return
    for name in (WHOLE, CHUNKS, "policies"):
        c = get(f"{s.qdrant_url}/collections/{name}")
        n = (c.json().get("result", {}).get("points_count") or 0) if c is not None and c.status_code == 200 else 0
        check(OK if n else FAIL, f"Qdrant collection '{name}'",
              f"{n} points" if n else "empty or missing: `cd backend && uv run python -m app.rag.ingest`")


def corpus() -> None:
    s = get_settings()
    cvs, jds = list(load_dir("cvs")), list(load_dir("jds"))
    check(OK if cvs and jds else FAIL, f"Corpus {s.data_dir}", f"{len(cvs)} CVs, {len(jds)} JDs")
    if s.hr_memory:
        from app.agent import hr_memory

        data = hr_memory._load(hr_memory.path())
        check(OK, "HR memory on", ", ".join(f"{k}: {v['sessions']} session(s)" for k, v in data.items()) or "nothing learned yet")
    else:
        check(OK, "HR memory off", "each run starts from data/criteria_profiles.yaml (set HR_MEMORY=true to demo learning)")


def api() -> None:
    r = get("http://localhost:8000/api/health")
    if r is None:
        check(WARN, "API :8000", "not running: `uv run uvicorn app.main:app --app-dir backend`")
    else:
        store = r.json().get("store")
        check(OK if store == "qdrant" else WARN, "API :8000", f"store: {store}")


def main() -> int:
    ollama(models())
    qdrant()
    corpus()
    api()
    for mark, what, detail in results:
        print(f"{mark} {what}" + (f" — {detail}" if detail else ""))
    blocked = sum(m == FAIL for m, _, _ in results)
    print(f"\n{'Ready for the demo.' if not blocked else f'{blocked} blocker(s) above.'}")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
