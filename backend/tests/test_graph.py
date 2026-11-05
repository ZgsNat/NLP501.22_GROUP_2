from app.agent.graph import after_ingest, after_review, after_search, after_verify, build_graph


def test_graph_compiles_with_all_nodes():
    g = build_graph().get_graph()
    assert {"ingest", "clarify", "search", "investigate", "expand_query", "rerank", "assess", "verify_evidence",
            "score", "arbitrate", "report", "hr_review"} <= set(g.nodes)


def test_l2_expansion_is_bounded():
    scarce = {"candidates": [{"cv_id": c} for c in "abc"], "knocked_out": {"a": "", "b": ""}}
    assert after_search({**scarce, "loops": {}}) == "expand_query"
    assert after_search({**scarce, "loops": {"expand": 2}}) == "rerank"  # max_expansions reached


def test_other_routers():
    assert after_ingest({"needs_clarification": True}) == "clarify"
    assert after_ingest({}) == "search"
    assert after_verify({"retry_ids": ["CV-1"]}) == "assess"
    assert after_verify({"retry_ids": []}) == "score"
    assert after_review({"hr_action": "adjust"}) == "score"
    assert after_review({"hr_action": "approve"}) == "__end__"


def test_ablation_switches_off_l2():
    scarce = {"candidates": [{"cv_id": c} for c in "abc"], "knocked_out": {}, "loops": {}}
    assert after_search({**scarce, "ablate": ["expand"]}) == "rerank"


def test_requirement_wise_queries_expand_aliases_and_weight_must_haves():
    from app.rag.retrieve import queries

    qs = queries({"title": "SDET", "must_have": ["selenium"], "test_types": ["api"], "depth_level": 3},
                 ["selenium", "jenkins"], {"selenium": ["playwright"]})
    labels = {q.label: q for q in qs}
    assert set(labels) == {"role", "must:selenium", "nice", "test_types"}
    assert "selenium webdriver" in labels["must:selenium"].text and "playwright" in labels["must:selenium"].text
    assert labels["must:selenium"].weight == 2 * labels["nice"].weight


def test_l1_rules_ask_hr_without_the_model():
    import asyncio

    from app.agent.nodes import ask_hr_gate, seniority_conflict
    from app.agent.schemas import JDRequirements

    class NoModel:
        def with_structured_output(self, schema):
            raise AssertionError("a rule must decide before any model is called")

    def req(title, years, must=("selenium",), depth=None, langs=(), types=()):
        return JDRequirements(title=title, must_have=list(must), nice_to_have=[], min_years=years,
                              depth_level=depth, languages=list(langs), test_types=list(types), ambiguities=[])

    assert asyncio.run(ask_hr_gate(NoModel(), "", req("Junior Automation Tester", 4)))[0]
    assert asyncio.run(ask_hr_gate(NoModel(), "", req("Senior QA", None, must=())))[0]
    # A fully-specified JD leaves no soft gap, so the gate proceeds without ever calling the model.
    clear = req("QA Engineer", 3, depth=3, langs=("java",), types=("web",))
    assert asyncio.run(ask_hr_gate(NoModel(), "", clear)) == (False, None)
    assert seniority_conflict("Senior QA Lead", 1) and not seniority_conflict("Senior QA", 5)


def test_vagueness_questions_detect_domain_specific_gaps():
    from app.agent.nodes import vagueness_questions
    from app.agent.schemas import JDRequirements

    def req(must=(), years=None, depth=None, langs=(), types=()):
        return JDRequirements(title="QA Engineer", must_have=list(must), nice_to_have=[], min_years=years,
                              depth_level=depth, languages=list(langs), test_types=list(types), ambiguities=[])

    # Vague years: "several years" but no number
    qs = vagueness_questions("We need several years of experience in testing.", req())
    assert any("years" in q.lower() for q in qs), f"should ask about years, got {qs}"

    # No depth when tools present: can't distinguish L2 from L3
    qs = vagueness_questions("Must have Selenium.", req(must=("selenium",)))
    assert any("framework" in q.lower() or "ownership" in q.lower() for q in qs), f"should ask about depth, got {qs}"

    # No test types
    qs = vagueness_questions("Create automated tests.", req())
    assert any("web" in q.lower() or "mobile" in q.lower() for q in qs), f"should ask about test types, got {qs}"

    # Clear JD: specific years + depth + types → no questions about those
    qs = vagueness_questions("3+ years", req(must=("selenium",), years=3, depth=3, types=("web", "api")))
    assert not any("years" in q.lower() for q in qs), "should not ask about years when specified"
    assert not any("framework" in q.lower() for q in qs), "should not ask about depth when specified"
    assert not any("web" in q.lower() for q in qs), "should not ask about types when specified"


def test_vagueness_questions_enrich_gate_ambiguities(monkeypatch):
    """Soft rules add questions to ambiguities without forcing the gate to ask (the model still decides)."""
    import asyncio

    from app.agent import nodes
    from app.agent.nodes import ask_hr_gate
    from app.agent.schemas import JDRequirements

    gate_calls = []

    async def fake_decide(model, question, options, context):
        gate_calls.append(context)
        return type("D", (), {"choice": "proceed", "confidence": 0.8, "reason": "clear enough"})()

    monkeypatch.setattr(nodes, "decide", fake_decide)

    r = JDRequirements(title="QA Engineer", must_have=["selenium"], nice_to_have=[], min_years=None,
                       depth_level=None, languages=[], test_types=[], ambiguities=[])
    jd = "We need someone with several years of experience for automated tests."

    ask, why = asyncio.run(ask_hr_gate(None, jd, r))

    assert len(gate_calls) == 1, "soft rules should reach the gate, not short-circuit"
    assert len(r.ambiguities) >= 2, f"should have generated questions, got {r.ambiguities}"
    assert not ask, "gate said proceed → should not ask"


def test_type_question_skipped_when_tools_reveal_it():
    """Fix B: a must-have tool's taxonomy group already reveals the test type — don't ask a redundant question."""
    from app.agent.nodes import vagueness_questions
    from app.agent.schemas import JDRequirements

    def req(must):
        return JDRequirements(title="QA Engineer", must_have=list(must), nice_to_have=[], min_years=3, depth_level=3,
                              languages=["java"], test_types=[], ambiguities=[])

    # Appium is a mobile tool; Selenium is web — the type is inferable, so no "which types?" question
    assert not any("types of testing" in q.lower() for q in vagueness_questions("x", req(("appium",))))
    assert not any("types of testing" in q.lower() for q in vagueness_questions("x", req(("selenium",))))
    # A non-type tool (CI server) leaves the type unknown → ask
    assert any("types of testing" in q.lower() for q in vagueness_questions("x", req(("jenkins",))))


def test_foreign_language_hedge_and_laundry_list():
    """Fixes C and D: a hedged English requirement and an over-long must-have list each raise a question."""
    from app.agent.nodes import vagueness_questions
    from app.agent.schemas import JDRequirements

    def req(must=("selenium",), foreign=()):
        return JDRequirements(title="QA Engineer", must_have=list(must), nice_to_have=[], min_years=3, depth_level=3,
                              languages=["java"], test_types=["web"], foreign_languages=list(foreign), ambiguities=[])

    # (C) English hedged between must-have and nice-to-have (JD-003's exact wording)
    jd = "English is a plus, or maybe required depending on the client."
    qs = vagueness_questions(jd, req(foreign=("english",)))
    assert any("english" in q.lower() and ("require" in q.lower() or "prefer" in q.lower()) for q in qs), qs
    # English stated plainly → not a hedge → no question
    assert not any("english" in q.lower() for q in vagueness_questions("English required.", req(foreign=("english",))))
    # "English is a plus" alone is a clear nice-to-have, not a hedge
    assert not any("english" in q.lower() for q in vagueness_questions("English is a plus.", req(foreign=("english",))))

    # (D) 6+ must-haves cannot all be knock-outs
    many = ("selenium", "cypress", "playwright", "appium", "k6", "jenkins")
    assert any("truly required" in q.lower() for q in vagueness_questions("x", req(must=many)))
    assert not any("truly required" in q.lower() for q in vagueness_questions("x", req(must=("selenium", "java"))))


def test_clarification_round_bundles_soft_questions():
    """Fix A: max_clarifications=1, so when the no-tools rule fires the single round must still carry the soft
    questions (years, types), not only the tools question."""
    import asyncio

    from app.agent.nodes import ask_hr_gate
    from app.agent.schemas import JDRequirements

    r = JDRequirements(title="Automation Tester", must_have=[], nice_to_have=[], min_years=None, depth_level=None,
                       languages=[], test_types=[], ambiguities=[])
    jd = "We need several years of experience building automated tests with modern tools."
    ask, _ = asyncio.run(ask_hr_gate(None, jd, r))  # hard rule fires, no model called

    assert ask
    assert any("must-have" in q.lower() for q in r.ambiguities), f"tools question present, got {r.ambiguities}"
    assert any("years" in q.lower() for q in r.ambiguities), f"years bundled in the same round, got {r.ambiguities}"
    assert len(r.ambiguities) <= 4


def test_umbrella_type_is_not_a_concrete_must_have():
    """Fix #1: "API testing" names a test TYPE, not a tool. A JD whose only must-have is the umbrella phrase still
    hasn't named a knock-out tool, so the hard rule asks for concrete tools (no model called); a real tool alongside
    it leaves no gap, so the gate proceeds."""
    import asyncio

    from app.agent.nodes import ask_hr_gate, concrete_must_have
    from app.agent.schemas import JDRequirements

    assert concrete_must_have(["api_testing"]) == []
    assert concrete_must_have(["api_testing", "postman"]) == ["postman"]

    def req(must):
        return JDRequirements(title="QA Engineer", must_have=list(must), nice_to_have=[], min_years=3, depth_level=3,
                              languages=["java"], test_types=["api"], ambiguities=[])

    umbrella_only = req(("api_testing",))
    ask, why = asyncio.run(ask_hr_gate(None, "Looking for API testing experience.", umbrella_only))
    assert ask and "no concrete" in why.lower()
    assert any("tools or frameworks" in q.lower() for q in umbrella_only.ambiguities), umbrella_only.ambiguities
    # A concrete tool named alongside the umbrella → the hard rule does not fire, no gaps → proceed without a model.
    assert asyncio.run(ask_hr_gate(None, "Postman for API testing.", req(("api_testing", "postman")))) == (False, None)


def test_arbiter_only_judges_true_near_ties(monkeypatch):
    import asyncio

    from app.agent import nodes

    asked = []

    async def fake_decide(model, question, options, context):
        asked.append(tuple(options))
        return type("D", (), {"choice": options[0], "confidence": 1.0, "reason": "r"})()

    monkeypatch.setattr(nodes, "decide", fake_decide)
    monkeypatch.setattr(nodes, "emit", lambda *a, **k: None)
    monkeypatch.setattr(nodes, "llm", lambda state, role: None)
    monkeypatch.setattr(nodes, "_versus_next", lambda rows, state: None)
    row = lambda cid, total: {"cv_id": cid, "status": "ranked", "total": total, "arbiter_note": None}  # noqa: E731
    state = {"ranking": [row("A", 0.77), row("B", 0.92), row("C", 0.90)], "requirements": {}, "ablate": [],
             "assessments": {c: {"criteria": {}, "llm_review": {}} for c in "ABC"}}
    asyncio.run(nodes.arbitrate.__wrapped__(state))  # without @traced, which needs the LangGraph runtime
    assert asked == [("B", "C"), ("C", "B")]  # A/B differ by 0.15 (inverted by fusion): not a near-tie
