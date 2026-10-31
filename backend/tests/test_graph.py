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

    def req(title, years, must=("selenium",)):
        return JDRequirements(title=title, must_have=list(must), nice_to_have=[], min_years=years, depth_level=None,
                              languages=[], test_types=[], ambiguities=[])

    assert asyncio.run(ask_hr_gate(NoModel(), "", req("Junior Automation Tester", 4)))[0]
    assert asyncio.run(ask_hr_gate(NoModel(), "", req("Senior QA", None, must=())))[0]
    assert asyncio.run(ask_hr_gate(NoModel(), "", req("QA Engineer", 3))) == (False, None)
    assert seniority_conflict("Senior QA Lead", 1) and not seniority_conflict("Senior QA", 5)


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
