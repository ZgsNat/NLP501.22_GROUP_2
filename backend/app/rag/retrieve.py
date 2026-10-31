"""Requirement-wise retrieval over CV chunks.

One query string for a whole JD lets the most frequent term dominate: a CV full of "Selenium" outranks a CV that
covers Selenium *and* Java *and* Jenkins. Instead we issue one query per requirement (each must-have expanded
with its taxonomy aliases, plus the role, the nice-to-haves and the test types), keep each CV's best chunk per
query (max-pooling: a requirement is met if one passage meets it), normalise per query, and take the weighted
mean. The per-query best chunk is also the *evidence passage* we show and rerank.

Indexes: `cvs` holds one point per whole CV (v1, baselines); `cv_chunks` holds structural chunks."""

from dataclasses import dataclass, field

from app.nlp.pipeline import CVProfile
from app.nlp.taxonomy import aliases
from app.rag.store import KnowledgeStore

WHOLE, CHUNKS = "cvs", "cv_chunks"
# Evidence-weighted pooling (OFF by default — a measured negative result, see docs/EVAL.md): down-weighting skills
# and summary passages halved keyword-stuffed CVs in the top 10 but cost 0.06 nDCG@10, because strong candidates
# also name their tools mostly in those sections. Trap filtering is done better by hedge-aware knock-out + rerank.
SECTION_WEIGHT = {"experience": 1.0, "projects": 1.0, "education": 0.8, "certifications": 0.8, "summary": 0.6,
                  "header": 0.6, "skills": 0.5, "other": 0.5}


@dataclass
class Query:
    label: str
    text: str
    weight: float


@dataclass
class CVHit:
    cv_id: str
    score: float
    best: dict[str, tuple[str, float]] = field(default_factory=dict)  # query label → (chunk_id, raw score)


def chunk_items(profiles: list[CVProfile]) -> list[tuple[str, str, dict]]:
    return [(c.id, c.embed_text, {"doc_id": p.cv_id, "chunk_id": c.id, "section": c.section, "text": c.text,
                                  "embed_text": c.embed_text, "headline": p.headline, "lang": p.lang})
            for p in profiles for c in p.chunks]


def queries(req: dict, query_terms: list[str], substitutes: dict[str, list[str]] | None = None) -> list[Query]:
    subs = substitutes or {}
    out = [Query("role", f"{req.get('title', '')}. Automation depth L{req.get('depth_level') or 2}.", 1.0)]
    for m in req.get("must_have", []):
        terms = list(dict.fromkeys([m.replace("_", " "), *aliases(m), *subs.get(m, [])]))
        out.append(Query(f"must:{m}", " ".join(terms), 2.0))
    extra = [t for t in query_terms if t not in req.get("must_have", [])]
    if extra:
        out.append(Query("nice", ", ".join(t.replace("_", " ") for t in extra), 1.0))
    if req.get("test_types"):
        out.append(Query("test_types", f"{', '.join(req['test_types'])} test automation", 1.0))
    return out


def rank_cvs(store: KnowledgeStore, collection: str, qs: list[Query], limit: int = 200,
             mode: str = "hybrid", section_weighted: bool = False) -> list[CVHit]:
    hits: dict[str, CVHit] = {}
    total_w = sum(q.weight for q in qs) or 1.0
    for q in qs:
        best: dict[str, tuple[str, float]] = {}
        for h in store.hybrid_search(collection, q.text, limit=limit, mode=mode):
            sc = h.score * (SECTION_WEIGHT.get(h.payload.get("section"), 1.0) if section_weighted else 1.0)
            if h.id not in best or sc > best[h.id][1]:
                best[h.id] = (h.payload.get("chunk_id", h.id), sc)
        top = max((s for _, s in best.values()), default=0.0) or 1.0
        for cv_id, (chunk_id, s) in best.items():
            hit = hits.setdefault(cv_id, CVHit(cv_id, 0.0))
            hit.best[q.label] = (chunk_id, s)
            hit.score += q.weight * (s / top) / total_w
    return sorted(hits.values(), key=lambda h: h.score, reverse=True)


def rerank_cvs(store: KnowledgeStore, qs: list[Query], hits: list[CVHit], profiles: dict[str, CVProfile]
               ) -> dict[str, float]:
    """Cross-encoder over (requirement query, the CV's evidence passages): weighted mean of per-query max.
    Passages are short, so nothing is truncated at the 512-token limit, unlike whole-CV reranking."""
    pairs, index = [], []
    for h in hits:
        chunks = {c.id: c for c in profiles[h.cv_id].chunks}
        passages = list(dict.fromkeys(cid for cid, _ in h.best.values() if cid in chunks)) or list(chunks)[:3]
        for qi, q in enumerate(qs):
            for cid in passages:
                pairs.append((q.text, chunks[cid].embed_text))
                index.append((h.cv_id, qi))
    scores = store.rerank_pairs(pairs)
    per: dict[tuple[str, int], float] = {}
    for key, s in zip(index, scores):
        per[key] = max(per.get(key, s), s)
    total_w = sum(q.weight for q in qs) or 1.0
    return {h.cv_id: round(sum(q.weight * per.get((h.cv_id, qi), 0.0) for qi, q in enumerate(qs)) / total_w, 4)
            for h in hits}
