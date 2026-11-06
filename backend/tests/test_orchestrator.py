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

_REAL_DECIDE = nodes._decide  # before the fixture stubs it


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
        from app.agent.llm_rank import LLMRanking, RankedCV

        if schema is LLMRanking:  # prefers the highest CV number, however the shortlist is shown
            class Rank:
                async def ainvoke(self, msgs, config=None):
                    ids = sorted(set(re.findall(r"=== (CV-\d{3}) ===", msgs[-1].content)), reverse=True)
                    return LLMRanking(ranking=[RankedCV(cv_id=c, reason="evidence") for c in ids])
            return Rank()
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

    def run(steps, answers=(), goal=None, **settings):
        for k, v in settings.items():
            monkeypatch.setattr(s, k, v)
        model = Scripted(steps)
        monkeypatch.setattr(nodes, "llm", lambda state, role: model)
        g = orchestrator.build_agent_graph(InMemorySaver())
        cfg = {"configurable": {"thread_id": "t"}}
        jd = get_doc("jds", "JD-002")
        inp, pauses = {"jd_text": jd.text, "company_profile": jd.meta["company_profile"], **({"goal": goal} if goal else {})}, []
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


def test_screen_ignores_indexed_cvs_outside_the_pool(agent, monkeypatch):
    """Regression (demo with a 10-CV DATA_DIR over a 50-CV index): CVs the index holds but the pool does not must never
    reach the ranking, or score() looks them up and fails."""
    orig = KeywordStore.hybrid_search

    def with_ghost(self, collection, query, limit, where=None, mode="hybrid"):
        hits = orig(self, collection, query, limit, where, mode)
        return ([Hit("CV-999", 99.0, {"doc_id": "CV-999", "chunk_id": "CV-999#x", "section": "experience", "text": query})]
                + hits) if collection == "cv_chunks" else hits
    monkeypatch.setattr(KeywordStore, "hybrid_search", with_ghost)
    v, _, _ = agent([lambda m: call("parse_jd"), lambda m: call("screen"),
                     lambda m: call("respond_to_hr", message="Screened.")])
    assert not any("failed" in m.content for m in v["messages"] if isinstance(m, ToolMessage))
    assert "CV-999" not in {r["cv_id"] for r in v["ranking"]} and v["ranking"]


def test_rank_shortlist_reorders_and_the_order_survives_explain(agent):
    v, _, _ = agent([
        lambda m: call("parse_jd"), lambda m: call("screen"), lambda m: call("rank_shortlist"),
        lambda m: call("explain", cv_ids=[last_result(m, "rank_shortlist")["order"][0]]),
        lambda m: call("rank_shortlist"), lambda m: call("rank_shortlist"),  # budget: 2 per run
        lambda m: call("respond_to_hr", message="Ranked."),
    ])
    order = [r["cv_id"] for r in v["ranking"] if r["status"] == "ranked"]
    assert order == sorted(order, reverse=True) and order == v["llm_order"][:len(order)]  # explain kept the LLM order
    refused = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert v["loops"]["llm_rank"] == 2 and any("ranking budget used" in w for w in refused)


HR_GOAL = ("5 CV, trong đó yêu cầu selenium là bắt buộc, playwright là có nhưng có thể dùng các tiêu chí khác để đánh "
           "giá, AI là bắt buộc, nếu biết xử dụng AI trong automation thì là điểm cộng lớn")
AI_TERMS = ["copilot", "chatgpt", "llm", "self-healing", "healenium", "applitools"]


def test_a_complex_goal_is_planned_checked_confirmed_with_hr_and_its_limits_declared(agent):
    """The goal-driven loop on the real pool (no CV mentions AI): understand → feasibility → confirm with HR →
    act → check the goal → declare what could not be met. Each skipped step is refused by the harness."""
    c = lambda kind, target, quote, **kw: {"kind": kind, "target": target, "hr_quote": quote, **kw}  # noqa: E731
    first = [c("must", "selenium", "selenium là bắt buộc"), c("nice", "playwright", "playwright là có"),
             c("must", "ai_in_testing", "AI là bắt buộc"),
             c("bonus", "ai_in_testing", "AI trong automation thì là điểm cộng lớn", strength="strong")]
    revised = [first[0], first[1], first[3]]
    top5 = lambda m: [r["cv_id"] for r in last_result(m, "screen")["top"][:5]]  # noqa: E731
    v, pauses, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),  # refused: HR wrote a goal, plan first
        lambda m: call("formalize_goal", count=5, constraints=[c("must", "selenium", "Cypress là bắt buộc")]),  # not HR's words
        lambda m: call("formalize_goal", count=5, constraints=first),
        lambda m: call("screen"),  # refused: 'ai_in_testing' is not defined yet
        lambda m: AIMessage("", tool_calls=[{"name": "define_concept", "id": "dc", "type": "tool_call", "args": {
            "name": "ai_in_testing", "description": "Uses AI tools in test automation", "terms": AI_TERMS,
            "hr_quote": "AI trong automation"}}]),
        lambda m: call("screen"),  # refused: no CV meets every must-have, HR must decide
        lambda m: call("ask_hr", questions=["No CV in the pool mentions AI. Make AI a big plus instead of a must, "
                                            "and accept fewer than 5?"]),
        lambda m: call("formalize_goal", count=5, constraints=revised, accept_shortfall="ít hơn 5 người cũng được"),
        lambda m: call("screen"),
        lambda m: call("explain", cv_ids=top5(m)),
        lambda m: call("respond_to_hr", message="Here are the five strongest.", shortlist=top5(m)),  # hides the gap
        lambda m: call("respond_to_hr", message="Five Selenium candidates; none shows AI in automation.",
                       shortlist=top5(m), limits=["No CV mentions AI tools, so the AI bonus could not be applied; "
                                                  "widen the pool or relax AI."]),
    ], answers=[{"answers": {"q": "AI chuyển thành điểm cộng lớn thôi, ít hơn 5 người cũng được"}}], goal=HR_GOAL)
    why = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert any("formalize_goal first" in w for w in why)
    assert any("not HR's words" in w for w in why)
    assert any("neither taxonomy skills nor defined concepts" in w for w in why)
    assert any("cannot be met as stated" in w for w in why)
    assert any("does not meet HR's goal" in w for w in why)
    assert pauses[0]["kind"] == "clarification" and pauses[-1]["kind"] == "chat"
    assert "selenium" in v["requirements"]["must_have"] and "playwright" not in v["requirements"]["must_have"]
    assert v["concepts"]["ai_in_testing"]["evidence"] == {}  # code found no AI evidence: none is invented
    assert len(v["shortlist"]) == 5 and "Goal check (by code)" in v["reply"] and v["report"]["goal"]["limits"]
    sel = {r["cv_id"] for r in v["ranking"] if r["status"] == "ranked"}
    assert set(v["shortlist"]) <= sel  # every recommended CV passed the Selenium knock-out


def test_when_the_budget_stops_the_agent_code_declares_the_unmet_goal(agent):
    """Q20 holds for the harness's own answer: a budget stop on a goal says what was not delivered."""
    c = {"kind": "must", "target": "selenium", "hr_quote": "selenium là bắt buộc"}
    v, _, _ = agent([lambda m: call("parse_jd"), lambda m: call("formalize_goal", count=5, constraints=[c])]
                    + [lambda m: call("get_candidate", cv_id=f"CV-0{i:02d}") for i in range(10, 40)],
                    goal=HR_GOAL, max_agent_steps=4)
    assert "Goal check (by code)" in v["reply"] and v["report"]["goal"]["limits"]
    assert any("count: 0 of the 5" in u for u in v["report"]["goal"]["unmet"])


def test_hr_rejection_takes_the_cv_out_of_the_ranking_for_good(agent, monkeypatch, tmp_path):
    """E15 'decide' found this: a rejected #1 stayed #1, because a reject only re-fitted the weights."""
    monkeypatch.setattr(get_settings(), "runtime_dir", tmp_path)  # the audit log goes here
    monkeypatch.setattr(nodes, "_decide", _REAL_DECIDE)  # the fixture stubs it out
    v, _, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("respond_to_hr", message="Screened the pool."),
        lambda m: call("record_decisions", decisions=[{"cv_id": "CV-002", "action": "reject",
                                                       "reason": "needs more test-architecture experience"}]),
        lambda m: call("screen"),  # a later re-screen must not bring it back
        lambda m: call("respond_to_hr", message="CV-002 is out, as you asked."),
    ], answers=[{"message": "Reject CV-002: the client needs more test-architecture experience."}])
    row = next(r for r in v["ranking"] if r["cv_id"] == "CV-002")
    assert row["status"] == "rejected_by_hr" and "test-architecture" in row["reason"]
    assert "CV-002" not in [r["cv_id"] for r in v["ranking"] if r["status"] == "ranked"]
    assert (tmp_path / "hr_audit.jsonl").exists()


def test_a_goal_met_as_written_is_not_held_up_by_questions_and_states_its_assumptions(agent):
    """E26 `feasible`: open questions about JD details HR never mentioned do not stop the run. Asking is refused
    before ranking, screen is allowed, and the answer shows them to HR as assumptions."""
    q = "Is GitHub Actions fine instead of GitLab CI?"
    top = lambda m: [r["cv_id"] for r in last_result(m, "screen")["top"][:2]]  # noqa: E731
    v, pauses, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("formalize_goal", count=2, open_questions=[q], constraints=[
            {"kind": "must", "target": "playwright", "hr_quote": "Playwright is required"}]),
        lambda m: call("ask_hr", questions=[q]),  # refused: nothing HR asked for depends on it
        lambda m: call("screen"),
        lambda m: call("explain", cv_ids=top(m)),
        lambda m: call("respond_to_hr", message="Two Playwright candidates.", shortlist=top(m)),
    ], goal="Top 2 candidates; Playwright is required.")
    why = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert len(why) == 1 and "can be met as written" in why[0]
    assert [p["kind"] for p in pauses] == ["chat"]  # HR was not interrupted before the answer
    assert len(v["shortlist"]) == 2 and f"assumed, not asked (the goal is met as written): {q}" in v["reply"]
    assert v["report"]["goal"]["unmet"] == [] and v["report"]["goal"]["assumed"] == [q]


def test_a_first_answer_that_names_candidates_must_recommend_them(agent):
    """E15: an answer listed five CVs with an empty shortlist, so HR got no recommendation. The first answer of a run
    may not do that; a follow-up that discusses ranked CVs without re-recommending them is fine."""
    top = lambda m: last_result(m, "screen")["top"][0]["cv_id"]  # noqa: E731
    v, _, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("explain", cv_ids=[top(m)]),
        lambda m: call("respond_to_hr", message=f"I recommend {top(m)}."),  # refused: empty shortlist
        lambda m: call("respond_to_hr", message=f"I recommend {top(m)}.", shortlist=[top(m)]),
        lambda m: call("respond_to_hr", message=f"{top(m)} leads on automation depth."),  # follow-up: allowed
    ], answers=[{"message": "Why is it first?"}])
    why = [m.content for m in v["messages"] if isinstance(m, ToolMessage) and m.content.startswith("REJECTED")]
    assert len(why) == 1 and "shortlist is empty" in why[0]
    assert "leads on automation depth" in v["reply"]
    assert len(v["shortlist"]) == 1 and v["report"]["top_ids"] == v["shortlist"]  # the follow-up kept the shortlist


def test_a_follow_up_that_runs_out_of_steps_keeps_the_shortlist_hr_already_has(agent):
    top = lambda m: last_result(m, "screen")["top"][0]["cv_id"]  # noqa: E731
    v, pauses, _ = agent([  # after the answer, the scripted model loops on get_candidate until the turn's budget ends
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("explain", cv_ids=[top(m)]),
        lambda m: call("respond_to_hr", message=f"I recommend {top(m)}.", shortlist=[top(m)]),
    ], answers=[{"message": "Compare it with everyone else in detail."}], max_agent_steps=4)
    first = pauses[0]["payload"]["shortlist"][0]["cv_id"]
    assert v["loops"]["budget_stops"] == 1 and v["shortlist"] == [first]
    assert "the shortlist from my last answer stands" in v["reply"]


def test_a_budget_stop_after_explain_recommends_the_verified_candidates(agent):
    """E26 round 5b: the agent explained the top candidates, then ran out of tokens; the fallback said nothing was
    checked and recommended no one. Code now hands HR the explained ones (respond_to_hr's own precondition)."""
    two = lambda m: [r["cv_id"] for r in last_result(m, "screen")["top"][:2]]  # noqa: E731
    v, pauses, _ = agent([
        lambda m: call("parse_jd"),
        lambda m: call("screen"),
        lambda m: call("explain", cv_ids=two(m)),
    ], max_agent_steps=4)  # then the scripted model loops on get_candidate until the step budget ends
    ranked_ids = [r["cv_id"] for r in v["ranking"] if r["status"] == "ranked"]
    assert v["loops"]["budget_stops"] == 1 and v["shortlist"] == ranked_ids[:2]
    assert "explained and their evidence checked" in v["reply"] and "Not yet explained" in v["reply"]
    assert [c["cv_id"] for c in v["report"]["candidates"]] == v["shortlist"]
    assert pauses[0]["payload"]["shortlist"]  # HR sees the recommendation
