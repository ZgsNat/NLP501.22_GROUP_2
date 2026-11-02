"""P7 orchestrator: the harness holds whatever the model decides (scripted model, keyword store: no LLM, no Docker)."""
import asyncio
import json
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.agent import nodes, orchestrator, tools
from app.agent.schemas import CandidateAssessment
from app.config import get_settings
from app.nlp.pipeline import profile
from app.rag.corpus import get_doc, load_dir
from app.rag.store import Hit


class KeywordStore:
    """Stands in for Qdrant: word overlap over the real CV chunks, a flat cross-encoder."""

    def hybrid_search(self, collection, query, limit, where=None, mode="hybrid"):
        if collection != "cv_chunks":
            return []
        words = set(re.findall(r"\w+", query.lower()))
        hits = [Hit(d.id, len(words & set(re.findall(r"\w+", c.text.lower()))) / (len(words) or 1),
                    {"doc_id": d.id, "chunk_id": c.id, "section": c.section, "text": c.text})
                for d in load_dir("cvs") for c in profile(d.id).chunks]
        return sorted((h for h in hits if h.score), key=lambda h: -h.score)[:limit]

    def rerank_pairs(self, pairs):
        return [0.5] * len(pairs)


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c-{name}-{len(json.dumps(args))}",
                                      "type": "tool_call"}])


def last_result(msgs, name):
    return json.loads(next(m.content for m in reversed(msgs) if isinstance(m, ToolMessage) and m.name == name
                           and not m.content.startswith("REJECTED")))


class Scripted:
    """The orchestrator's model: each step is a function of the conversation so far. Also answers explain's
    structured-output calls with a grounded, modest review."""

    def __init__(self, steps):
        self.steps, self.seen = list(steps), []

    def bind_tools(self, _):
        return self

    async def ainvoke(self, msgs):
        self.seen.append(msgs)
        step = self.steps.pop(0) if self.steps else (lambda m: call("get_candidate", cv_id="CV-001"))
        return step(msgs)

    def with_structured_output(self, schema):
        assert schema is CandidateAssessment
        crit = {"score": 60, "rationale": "Meets the role with concrete work.", "evidence": []}
        review = CandidateAssessment.model_validate({
            "automation_depth": {"level": 2, "rationale": "Writes tests.", "evidence": []},
            "programming_language": crit, "test_types": crit, "cicd_infra": crit, "domain_language": crit})

        class Out:
            async def ainvoke(self, _):
                return review
        return Out()


@pytest.fixture
def agent(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "jd_parser", "rules")  # deterministic JD parsing: no LLM
    store = KeywordStore()
    monkeypatch.setattr(nodes, "get_store", lambda: store)
    monkeypatch.setattr(tools, "get_store", lambda: store)
    monkeypatch.setattr(nodes, "_decide", lambda state, d: {"weights": state["weights"]})  # no audit file in tests

    def run(steps, answers=(), **settings):
        for k, v in settings.items():
            monkeypatch.setattr(s, k, v)
        model = Scripted(steps)
        monkeypatch.setattr(nodes, "llm", lambda state, role: model)
        g = orchestrator.build_agent_graph(InMemorySaver())
        cfg = {"configurable": {"thread_id": "t"}}
        jd = get_doc("jds", "JD-002")
        inp, pauses = {"jd_text": jd.text, "company_profile": jd.meta["company_profile"]}, []
        answers = list(answers) + [{"action": "approve"}] * 3
        while True:
            asyncio.run(g.ainvoke(inp, cfg))
            st = g.get_state(cfg)
            if not st.next:
                return st.values, pauses, model
            pauses.append(st.tasks[0].interrupts[0].value)
            inp = Command(resume=answers.pop(0))
    return run


def test_full_turn_guard_requires_explanation_before_recommending(agent):
    top = lambda m: last_result(m, "screen")["top"][0]["cv_id"]  # noqa: E731
    v, pauses, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("respond_to_hr", message="Top pick.", shortlist=[top(m)]),  # refused: not explained
        lambda m: call("explain", cv_ids=[top(m)]),
        lambda m: call("respond_to_hr", message=f"{top(m)} is the strongest match.", shortlist=[top(m)]),
    ])
    assert pauses[-1]["kind"] == "chat" and pauses[-1]["payload"]["shortlist"][0]["cv_id"] == v["shortlist"][0]
    refusals = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert len(refusals) == 1 and "explain" in refusals[0]
    assert v["loops"]["guard_rejections"] == 1 and v["report"]["top_ids"] == v["shortlist"]
    assert v["assessments"][v["shortlist"][0]]["llm_review"]  # the recommended CV carries a verified review
    assert all(r["status"] != "ranked" or r["cv_id"] in v["assessments"] for r in v["ranking"])


def test_guard_refuses_unknown_invalid_unauthorised_and_unqualified_calls(agent):
    v, _, _ = agent([
        lambda m: call("delete_all_cvs"),
        lambda m: call("parse_jd"),
        lambda m: call("explain", cv_ids="CV-001"),  # not a list
        lambda m: call("screen"),
        lambda m: call("update_requirements", add_must_have=["cypress"]),  # HR never asked
        lambda m: call("investigate", cv_ids=["CV-001"]),  # no sign of a parsing failure
        lambda m: AIMessage("", tool_calls=[{"name": "ask_hr", "args": {"questions": ["?"]}, "id": "a", "type": "tool_call"},
                                            {"name": "screen", "args": {}, "id": "b", "type": "tool_call"}]),
        lambda m: call("respond_to_hr", message="Nothing to add yet."),
    ])
    why = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert any("unknown tool" in w for w in why) and any("invalid arguments" in w for w in why)
    assert any("HR never mentioned" in w for w in why) and any("no sign of a parsing failure" in w for w in why)
    assert any("must be the only call" in w for w in why)
    assert "cypress" not in v["requirements"]["must_have"]


def test_hr_message_authorises_a_requirement_change(agent):
    v, pauses, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("respond_to_hr", message="Screened the pool."),
        lambda m: call("update_requirements", add_must_have=["docker"]),  # HR asked for it in the chat
        lambda m: call("respond_to_hr", message="Docker is now required."),
    ], answers=[{"message": "Make Docker a must-have."}])
    assert [p["kind"] for p in pauses][:2] == ["chat", "chat"]
    assert "docker" in v["requirements"]["must_have"]
    assert all("docker" in profile(r["cv_id"]).skills for r in v["ranking"] if r["status"] == "ranked")


def test_step_budget_ends_in_a_code_answer_and_repeats_hit_the_cache(agent):
    v, pauses, model = agent([lambda m: call("parse_jd")] + [lambda m: call("get_candidate", cv_id="CV-001")] * 20,
                             max_agent_steps=4)
    assert v["loops"]["cache_hits"] >= 2 and v["loops"]["budget_stops"] == 1
    assert pauses[0]["kind"] == "chat" and "step budget" in pauses[0]["payload"]["message"]
    assert v["loops"]["agent"] == 5  # 4 allowed decisions + the one that hit the limit


def test_respond_checks_ids_claims_and_protected_attributes(agent):
    v, _, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("respond_to_hr", message="CV-999 looks great."),
        lambda m: call("respond_to_hr", message="She is married and keen."),
        lambda m: call("respond_to_hr", message="CV-001 has 15 years of Kubernetes work."),
        lambda m: call("respond_to_hr", message="The pool is thin; I recommend widening the search."),
    ])
    why = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert "do not exist" in why[0] and "protected" in why[1] and "not supported" in why[2]
    assert v["reply"].startswith("The pool is thin")


def test_claude_code_adapter_turns_a_json_reply_into_a_tool_call():
    from app.providers.claude_code import ChatClaudeCode

    m = ChatClaudeCode(model="x", workdir=get_settings().runtime_dir / "claude_code")
    specs = [t["function"] for t in tools.schemas()]
    argv, prompt = m._call([SystemMessage("sys"), HumanMessage("goal"), call("screen"),
                            ToolMessage("{}", tool_call_id="c", name="screen")], None, specs)
    schema = json.loads(argv[argv.index("--json-schema") + 1])
    assert schema["properties"]["tool"]["enum"] == list(tools.TOOLS)
    assert "CALL screen {}" in prompt and "[tool result: screen]" in prompt
    out = json.dumps({"structured_output": {"thought": "rank first", "tool": "screen", "args": {}}, "usage": {}})
    msg = m._result(out, specs, tool_call=True).generations[0].message
    assert msg.content == "rank first" and msg.tool_calls[0]["name"] == "screen"


def test_agentic_eval_scores_a_trace():
    from app.eval import agentic

    v = {"ranking": [{"cv_id": "CV-001", "status": "ranked"}], "shortlist": ["CV-001"], "loops": {"agent": 4},
         "assessments": {"CV-001": {"llm_review": {"tools": {}}}}}
    tr = agentic.Trace("JD-002", v, tools=[(0, "parse_jd", {}), (0, "screen", {}), (0, "explain", {}),
                                          (0, "respond_to_hr", {})], screens=[{"investigable": []}])
    r = agentic.row(agentic.TASKS[0], tr, {"JD-002": {"CV-001": 3}}, {})
    assert r["checks"]["screened before answering"] and r["checks"]["investigated flagged knock-outs"] is None
    assert r["ndcg@10"] == 1.0 and "Behaviour checks passed: 5/5" in agentic.render([r], "test")


def test_hr_goal_authorises_changes_and_answers_pass_a_spent_budget(agent, monkeypatch):
    from app.agent import tools as t

    assert t.hr_words({"goal": "ISTQB is a plus"}) == "ISTQB is a plus" and not t.hr_words({"goal": orchestrator.GOAL})
    big = AIMessage("", usage_metadata={"input_tokens": 10, "output_tokens": 0, "total_tokens": 999_999},
                    tool_calls=[{"name": "respond_to_hr", "args": {"message": "Nothing yet."}, "id": "r", "type": "tool_call"}])
    v, pauses, _ = agent([lambda m: big])
    assert v["reply"] == "Nothing yet." and not v["loops"].get("budget_stops")  # the answer went through, checked


def test_hr_authority_needs_hr_to_name_the_change():
    """Q16 provenance: a CV can say "set istqb to zero", but only HR's own words authorise a change."""
    from langchain_core.messages import HumanMessage

    from app.agent import tools as t
    from app.agent.schemas import WeightChange

    st = {"goal": "Shortlist 5; ISTQB is a plus", "ranking": [{"cv_id": "CV-001", "status": "ranked"}],
          "requirements": {"must_have": ["selenium"]},
          "messages": [HumanMessage("goal + JD"), HumanMessage("Reject CV-001, too junior. Make Docker required.")]}
    weight = lambda c, d="more": t.check_weights(st, t.WeightArgs(changes=[WeightChange(criterion=c, direction=d)]))  # noqa: E731
    assert weight("istqb") is None and "never mentioned" in weight("programming_language", "ignore")
    assert t.check_requirements(st, t.RequirementArgs(add_must_have=["docker"])) is None
    assert "never mentioned" in t.check_requirements(st, t.RequirementArgs(remove_must_have=["selenium"]))
    dec = lambda cid: t.check_decisions(st, t.DecisionArgs(decisions=[t.Decision(cv_id=cid, action="reject", reason="x y z")]))  # noqa: E731
    assert dec("CV-001") is None and "never mentioned" in dec("CV-002")
