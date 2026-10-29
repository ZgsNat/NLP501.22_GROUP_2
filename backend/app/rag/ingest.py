"""Index the corpus into Qdrant:  uv run python -m app.rag.ingest

cvs        one point per whole CV (v1 behaviour; the B0–B2 baselines)
cv_chunks  structural chunks from the NLP pipeline (what the agent searches)
policies   hiring policies (RAG context for the LLM)"""

from app.nlp.pipeline import profile
from app.rag.corpus import load_dir
from app.rag.retrieve import CHUNKS, WHOLE, chunk_items
from app.rag.store import KnowledgeStore, get_store


def ingest(store: KnowledgeStore | None = None) -> dict[str, int]:
    store = store or get_store()
    cvs = list(load_dir("cvs"))
    profiles = [profile(d.id) for d in cvs]
    store.index(WHOLE, cvs)
    store.index_items(CHUNKS, chunk_items(profiles))
    store.index("policies", list(load_dir("policies")))
    return {WHOLE: len(cvs), CHUNKS: sum(len(p.chunks) for p in profiles), "policies": len(load_dir("policies"))}


if __name__ == "__main__":
    print(ingest())
