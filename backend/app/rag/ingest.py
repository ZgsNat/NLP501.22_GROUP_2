"""Index the corpus into Qdrant:  uv run python -m app.rag.ingest"""

from app.rag.corpus import load_dir
from app.rag.store import KnowledgeStore, get_store

COLLECTIONS = {"cvs": "cvs", "policies": "policies"}


def ingest(store: KnowledgeStore | None = None) -> dict[str, int]:
    store = store or get_store()
    counts = {}
    for collection, folder in COLLECTIONS.items():
        docs = list(load_dir(folder))
        store.index(collection, docs)
        counts[collection] = len(docs)
    return counts


if __name__ == "__main__":
    print(ingest())
