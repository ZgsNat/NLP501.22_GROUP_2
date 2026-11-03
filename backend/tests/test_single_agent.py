"""S1 baseline harness (scripted model, keyword store): it measures what it claims to, with no LLM and no Docker."""
import asyncio
import re

from langchain_core.messages import AIMessage

from app.config import get_settings
from app.eval import single_agent as s1
from app.nlp.pipeline import profile
from app.rag.corpus import get_doc, load_dir
from app.rag.store import Hit


class Store:
    def hybrid_search(self, collection, query, limit, where=None, mode="hybrid"):
        words = set(re.findall(r"\w+", query.lower()))
        hits = [Hit(d.id, len(words & set(re.findall(r"\w+", c.text.lower()))), {"doc_id": d.id, "text": c.text})
                for d in load_dir("cvs") for c in profile(d.id).chunks]
        return sorted((h for h in hits if h.score), key=lambda h: -h.score)[:limit]


class Scripted:
    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def bind_tools(self, _):
        return self

    async def ainvoke(self, msgs, config=None):
        self.seen.append(msgs[-1].content)
        return self.steps.pop(0) if self.steps else AIMessage("thinking")


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": name, "type": "tool_call"}])


def test_s1_counts_invented_and_unread_ids_and_sees_planted_text(monkeypatch):
    monkeypatch.setattr(s1, "get_store", lambda: Store())
    model = Scripted([call("search_cvs", query="playwright typescript"), call("read_cv", cv_id="CV-004"),
                      call("submit_ranking", cv_ids=["CV-004", "CV-002", "CV-999"])])
    r = asyncio.run(s1.run_jd(get_doc("jds", "JD-002"), model, planted={"CV-004"}))
    assert r["finished"] and r["invented"] == ["CV-999"] and r["unread"] == ["CV-002"] and r["steps"] == 3
    assert s1.PLANT in model.seen[2]  # the read_cv result carried the planted instruction to the model


def test_s1_stops_at_its_step_budget(monkeypatch):
    monkeypatch.setattr(s1, "get_store", lambda: Store())
    monkeypatch.setattr(get_settings(), "max_single_agent_steps", 3)
    r = asyncio.run(s1.run_jd(get_doc("jds", "JD-002"), Scripted([])))
    assert not r["finished"] and r["steps"] == 3 and r["ranking"] == []


def test_overlap_and_traps():
    assert s1.overlap(["a", "b"] + list("cdefghij"), ["b", "a"] + list("cdefghiz")) == 0.9
    assert s1.traps10(["CV-004", "CV-002"], {"CV-004": 0, "CV-002": 3}) == 1
