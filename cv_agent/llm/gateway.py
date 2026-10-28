"""Single entry point for every LLM call.

Responsibilities:
* refuse to send unmasked PII (NFR-02)
* token budget accounting + BudgetExceeded for fallback (FR-26, NFR-03)
* JSONL log of every call: tokens, latency, prompt version, result (FR-31, NFR-05)
* on-disk cache keyed by (backend, model, prompt) for reproducibility & speed
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from cv_agent.ingest.pii import contains_pii

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    pass


class PIILeakError(RuntimeError):
    pass


@dataclass(frozen=True)
class LLMResponse:
    data: Dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    raw: str = ""


class LLMBackend(Protocol):
    name: str
    model: str

    def complete_json(self, task: str, system: str, user: str, context: Dict[str, Any]) -> LLMResponse: ...


@dataclass
class Budget:
    max_tokens: int = 60_000
    max_calls: int = 200
    timeout_s: float = 600.0
    used_tokens: int = 0
    calls: int = 0
    started_at: float = field(default_factory=time.time)

    def check(self, estimated: int = 0) -> None:
        if self.used_tokens + estimated > self.max_tokens:
            raise BudgetExceeded(f"token budget {self.max_tokens} exceeded ({self.used_tokens}+{estimated})")
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget {self.max_calls} exceeded")
        if time.time() - self.started_at > self.timeout_s:
            raise BudgetExceeded(f"timeout {self.timeout_s}s exceeded")

    def snapshot(self) -> Dict[str, Any]:
        return {"used_tokens": self.used_tokens, "calls": self.calls, "max_tokens": self.max_tokens,
                "max_calls": self.max_calls, "elapsed_s": round(time.time() - self.started_at, 2)}


class LLMGateway:
    def __init__(self, backend: LLMBackend, log_path: Optional[Path] = None, cache_dir: Optional[Path] = None,
                 budget: Optional[Budget] = None, run_id: str = "") -> None:
        self.backend = backend
        self.log_path = Path(log_path) if log_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.budget = budget or Budget()
        self.run_id = run_id
        self.records: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ public
    def call(self, task: str, prompt_version: str, system: str, user: str,
             context: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None) -> tuple:
        """Return (data, call_id). Raises BudgetExceeded / PIILeakError."""
        if contains_pii(system) or contains_pii(user):
            raise PIILeakError(f"Refusing to send unmasked PII to LLM (task={task})")
        estimated = (len(system) + len(user)) // 4
        self.budget.check(estimated)
        call_id = uuid.uuid4().hex[:12]
        key = self._cache_key(task, prompt_version, system, user)
        started = time.perf_counter()
        cached = self._cache_get(key)
        if cached is not None:
            resp, from_cache = LLMResponse(cached["data"], cached["prompt_tokens"], cached["completion_tokens"]), True
        else:
            resp, from_cache = self.backend.complete_json(task, system, user, context or {}), False
            self._cache_put(key, resp)
        latency = time.perf_counter() - started
        with self._lock:
            self.budget.calls += 1
            self.budget.used_tokens += resp.prompt_tokens + resp.completion_tokens
            self._log({
                "call_id": call_id, "run_id": self.run_id, "ts": time.time(), "task": task,
                "prompt_version": prompt_version, "backend": self.backend.name, "model": self.backend.model,
                "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens,
                "latency_s": round(latency, 4), "cached": from_cache, "result": resp.data,
                "prompt_sha1": key[:16], **({"meta": meta} if meta else {}),
            })
        return resp.data, call_id

    def stats(self) -> Dict[str, Any]:
        tasks: Dict[str, int] = {}
        tokens: Dict[str, int] = {}
        for r in self.records:
            tasks[r["task"]] = tasks.get(r["task"], 0) + 1
            tokens[r["task"]] = tokens.get(r["task"], 0) + r["prompt_tokens"] + r["completion_tokens"]
        return {**self.budget.snapshot(), "by_task": tasks, "tokens_by_task": tokens,
                "latency_s_total": round(sum(r["latency_s"] for r in self.records), 3)}

    # ------------------------------------------------------------------ internals
    def _cache_key(self, task: str, version: str, system: str, user: str) -> str:
        payload = json.dumps([self.backend.name, self.backend.model, task, version, system, user], ensure_ascii=False)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Corrupt cache entry %s ignored", path)
            return None

    def _cache_put(self, key: str, resp: LLMResponse) -> None:
        if self.cache_dir:
            (self.cache_dir / f"{key}.json").write_text(json.dumps(
                {"data": resp.data, "prompt_tokens": resp.prompt_tokens, "completion_tokens": resp.completion_tokens},
                ensure_ascii=False), encoding="utf-8")

    def _log(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
