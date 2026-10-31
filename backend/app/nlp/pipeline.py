"""The CV processing pipeline, end to end:

    raw file ─► read_file ─► normalize ─► redact ─► detect_language ─► sections ─► entries (+dates)
             ─► chunks (structural | window | whole) ─► skill mentions (+hedges) ─► years ─► depth cue ─► CVProfile

Everything downstream (indexing, knock-out, rule scores, the LLM prompt, evidence verification) reads the
CVProfile, never the raw file. All stages are deterministic, cost zero tokens and are unit-tested."""

from dataclasses import dataclass, field
from functools import lru_cache

from app.nlp import chunking, extract, segment, taxonomy
from app.nlp.redact import redact
from app.nlp.text import detect_language, normalize


@dataclass
class CVProfile:
    cv_id: str
    headline: str
    lang: str
    text: str  # normalised + redacted: the single source of truth for offsets, grounding and the LLM
    redactions: dict[str, int]
    sections: list[segment.Section]
    entries: list[segment.Entry]
    chunks: list[chunking.Chunk]
    skills: dict[str, list[extract.Mention]] = field(default_factory=dict)
    years_total: float | None = None
    years_automation: float | None = None
    years_source: list[str] = field(default_factory=list)
    depth_cue: int = 0
    depth_cue_line: str | None = None

    def best(self, skill: str) -> extract.Mention | None:
        return extract.best(self.skills.get(taxonomy.key(skill), []))

    def for_llm(self) -> str:
        """Sectioned CV for the LLM: canonical section labels make "skills list vs real work" explicit."""
        parts = [f"[{c.section}] {c.text}" for c in self.chunks]
        return "\n\n".join(parts)

    def view(self) -> dict:
        """What the UI's parse preview shows (API: CVProfileView)."""
        return {"id": self.cv_id, "headline": self.headline, "lang": self.lang, "redactions": self.redactions,
                "sections": [{"kind": s.kind, "heading": s.heading} for s in self.sections],
                "chunks": [{"id": c.id, "section": c.section, "header": c.header, "text": c.text} for c in self.chunks],
                "skills": [{"skill": k, "strength": m.strength, "section": m.section, "line": m.line}
                           for k, ms in self.skills.items() for m in [extract.best(ms)]],
                "years_total": self.years_total, "years_automation": self.years_automation,
                "depth_cue": self.depth_cue, "depth_cue_line": self.depth_cue_line}

    def facts_for_llm(self) -> str:
        by = {s: [k for k, ms in self.skills.items() if extract.best(ms).strength == s] for s in extract.CREDIT}
        yrs = (f"{self.years_total:g} years total, {self.years_automation:g} in roles with automation work"
               if self.years_total is not None else "unknown (no dated roles)")
        return (f"Parser facts (deterministic, verified): language={self.lang}; experience: {yrs}; "
                f"tools used in roles: {by['used'] or '—'}; only listed: {by['listed'] or '—'}; "
                f"hedged (familiar/learning, NOT real experience): {by['hedged'] or '—'}")


def process(cv_id: str, raw: str, headline: str = "", strategy: str = "structural") -> CVProfile:
    text, redactions = redact(normalize(raw))
    secs = segment.sections(text)
    ents = segment.entries(text, secs)
    if strategy == "structural":
        chunks = chunking.structural(cv_id, text, secs, ents, headline)
    elif strategy == "window":
        chunks = chunking.window(cv_id, text)
    else:
        chunks = chunking.whole(cv_id, text, headline)
    # Mentions and depth are read from structural chunks whatever the retrieval strategy: they need sections.
    struct = chunks if strategy == "structural" else chunking.structural(cv_id, text, secs, ents, headline)
    mentions = extract.skill_mentions(struct)
    total, auto, src = extract.years(text, ents, struct, mentions)
    level, line = extract.depth_cue(struct)
    return CVProfile(cv_id, headline, detect_language(text), text, redactions, secs, ents, chunks, mentions,
                     total, auto, src, level, line)


@lru_cache(maxsize=1024)
def profile(cv_id: str, strategy: str = "structural") -> CVProfile:
    """Profile of a CV in the corpus (data/cvs or an uploaded CV)."""
    from app.rag.corpus import get_doc

    doc = get_doc("cvs", cv_id)
    if doc is None:
        raise KeyError(cv_id)
    return process(cv_id, doc.text, doc.meta.get("headline", ""), strategy)
