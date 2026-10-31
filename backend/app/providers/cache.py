"""Persistent LLM response cache (LLM_CACHE=true). Same prompt + same model params → same answer, from disk.

Why: evaluation reruns (ablations A1/A2, B4 vs P6) share most LLM calls; caching makes them cheap and makes a
reported number reproducible from `.runtime/llm_cache.sqlite`. Off by default for live runs."""

import hashlib
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads
from langchain_core.outputs import Generation


class SQLiteLLMCache(BaseCache):
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS llm (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._lock = threading.Lock()

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        return hashlib.sha256(f"{llm_string}\x00{prompt}".encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM llm WHERE k = ?", (self._key(prompt, llm_string),)).fetchone()
        return loads(row[0]) if row else None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (self._key(prompt, llm_string), dumps(return_val)))
            self._db.commit()

    def clear(self, **kwargs) -> None:
        with self._lock:
            self._db.execute("DELETE FROM llm")
            self._db.commit()


def enable(path: Path) -> None:
    set_llm_cache(SQLiteLLMCache(path))
