"""A missing Qdrant is an error with the fix in it, unless running without search was asked for explicitly."""
import pytest

from app.config import Settings
from app.rag.store import KnowledgeStore

DOWN = "http://127.0.0.1:9"  # nothing listens here


def test_unreachable_qdrant_is_a_clear_error_by_default():
    with pytest.raises(RuntimeError, match="docker compose up -d qdrant"):
        KnowledgeStore(Settings(qdrant_url=DOWN, qdrant_fallback_memory=False))


def test_fallback_is_explicit_and_says_it_is_degraded():
    store = KnowledgeStore(Settings(qdrant_url=DOWN, qdrant_fallback_memory=True))
    assert store.degraded and store.hybrid_search("cv_chunks", "selenium", limit=5) == []


def test_a_missing_collection_on_a_real_index_is_an_error():
    store = KnowledgeStore(Settings(qdrant_url=":memory:"))
    with pytest.raises(RuntimeError, match="app.rag.ingest"):
        store.hybrid_search("cv_chunks", "selenium", limit=5)
