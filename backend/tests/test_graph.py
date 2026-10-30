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
