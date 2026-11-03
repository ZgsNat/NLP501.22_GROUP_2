"""Controlled LLM ranking: the boundary rules hold whatever the model answers (scripted model, no LLM)."""
import asyncio

from app.agent import llm_rank
from app.agent.llm_rank import LLMRanking, RankedCV
from app.nlp.pipeline import process


class Scripted:
    """Answers in a fixed preference order (best first), however the candidates are shown; records what it saw."""
    def __init__(self, prefer, extra=()):
        self.prefer, self.extra, self.seen = prefer, list(extra), []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, msgs, config=None):
        text = msgs[-1].content
        self.seen.append(text)
        shown = [c for c in self.prefer if f"=== {c} ===" in text]
        return LLMRanking(ranking=[RankedCV(cv_id=c, reason="r") for c in [*self.extra, *shown, shown[0]]])


def test_canonical_and_combine():
    assert llm_rank.canonical(["B", "X", "B", "A"], ["A", "B", "C"]) == ["B", "A", "C"]  # invented X dropped, C kept
    assert llm_rank.combine([["A", "B", "C"], ["B", "A", "C"]], ["C", "B", "A"]) == ["B", "A", "C"]  # tie → rule order


def test_rank_is_a_checked_permutation_and_both_orders_are_shown():
    ids = ["CV-1", "CV-2", "CV-3"]
    profiles = {c: process(c, f"## Experience\n- Wrote Selenium tests in Java for {c}.") for c in ids}
    model = Scripted(["CV-3", "CV-1", "CV-2"], extra=["CV-999"])  # invents an id and repeats one
    r = asyncio.run(llm_rank.rank(model, "JD", {"must_have": ["selenium"]}, ids, profiles))
    assert sorted(r["order"]) == sorted(ids) and r["order"][0] == "CV-3" and "CV-999" not in r["order"]
    first = lambda t: min((t.index(f"=== {c} ==="), c) for c in ids)[1]  # noqa: E731
    assert [first(t) for t in model.seen] == ["CV-1", "CV-3"]  # shown forward, then reversed
    assert all("Wrote Selenium tests" in t for t in model.seen)  # every candidate was read, not ranked blind
