"""Stage 5: chunking. Three strategies, so the choice is measured rather than asserted (see app.eval):

- `whole`      one vector per CV (v1). Long CVs get truncated at the encoder's 512-token limit, and one vector
               averages unrelated topics (a mobile project and a banking role blur together).
- `window`     fixed 120-word windows with 30-word overlap: the textbook RAG default, blind to CV structure.
- `structural` one chunk per role/project entry, one per other section, split at line boundaries if too long.
               Each chunk carries a *contextual header* ("Senior QA | experience: Automation Engineer — Bank
               (2018–2021)") that is embedded with it but not part of its text, so a bullet like "Built 600 API
               tests" is retrieved together with the role and dates it belongs to.

Every chunk keeps (start, end) offsets into the processed CV text, so evidence can be traced to its section."""

import re
from dataclasses import dataclass

from app.nlp.segment import ENTRY_SECTIONS, Entry, Section

STRATEGIES = ("structural", "window", "whole")


@dataclass
class Chunk:
    id: str
    cv_id: str
    section: str
    header: str  # context for embedding/reranking only
    start: int
    end: int
    text: str

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}" if self.header else self.text


def _words(s: str) -> int:
    return len(s.split())


def _make(cv_id: str, kind: str, n: int, header: str, text: str, start: int, end: int) -> Chunk | None:
    body = text[start:end]
    lead, trail = len(body) - len(body.lstrip()), len(body) - len(body.rstrip())
    start, end = start + lead, end - trail
    if start >= end:
        return None
    return Chunk(f"{cv_id}#{kind}-{n}", cv_id, kind, header, start, end, text[start:end])


def _split_lines(start: int, end: int, text: str, max_words: int) -> list[tuple[int, int]]:
    """Greedy split of [start, end) at line boundaries so each part stays under max_words."""
    parts, cur, count, pos = [], start, 0, start
    for line in text[start:end].split("\n"):
        w = _words(line)
        if count and count + w > max_words:
            parts.append((cur, pos))
            cur, count = pos, 0
        count += w
        pos += len(line) + 1
    parts.append((cur, end))
    return parts


def structural(cv_id: str, text: str, secs: list[Section], ents: list[Entry], headline: str = "",
               max_words: int = 150) -> list[Chunk]:
    out: list[Chunk] = []
    n: dict[str, int] = {}
    units = [(e.section, f"{e.section}: {e.title}", e.start, e.end) for e in ents]
    units += [(s.kind, s.kind, s.start, s.end) for s in secs if s.kind not in ENTRY_SECTIONS]
    for kind, label, start, end in sorted(units, key=lambda u: u[2]):
        header = f"{headline} | {label}" if headline else label
        for a, b in _split_lines(start, end, text, max_words):
            n[kind] = n.get(kind, 0) + 1
            if c := _make(cv_id, kind, n[kind], header, text, a, b):
                out.append(c)
    return out


def window(cv_id: str, text: str, size: int = 120, overlap: int = 30) -> list[Chunk]:
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    out, i, n = [], 0, 0
    while i < len(spans):
        j = min(i + size, len(spans))
        n += 1
        out.append(Chunk(f"{cv_id}#window-{n}", cv_id, "window", "", spans[i][0], spans[j - 1][1],
                         text[spans[i][0]:spans[j - 1][1]]))
        if j == len(spans):
            break
        i = j - overlap
    return out


def whole(cv_id: str, text: str, headline: str = "") -> list[Chunk]:
    return [Chunk(f"{cv_id}#whole-1", cv_id, "whole", headline, 0, len(text), text)]
