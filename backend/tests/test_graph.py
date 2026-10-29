from app.agent.graph import after_ingest, after_review, after_search, after_verify, build_graph


def test_graph_compiles_with_all_nodes():
    g = build_graph().get_graph()
    assert {"ingest", "clarify", "search", "expand_query", "rerank", "assess", "verify_evidence",
            "score", "arbitrate", "report", "hr_review"} <= set(g.nodes)


def test_l2_expansion_is_bounded():
    scarce = {"candidates": [{}] * 3, "knocked_out": {"a": "", "b": ""}}
    assert after_search({**scarce, "loops": {}}) == "expand_query"
    assert after_search({**scarce, "loops": {"expand": 2}}) == "rerank"  # max_expansions reached


def test_other_routers():
    assert after_ingest({"needs_clarification": True}) == "clarify"
    assert after_ingest({}) == "search"
    assert after_verify({"retry_ids": ["CV-1"]}) == "assess"
    assert after_verify({"retry_ids": []}) == "score"
    assert after_review({"hr_action": "adjust"}) == "score"
    assert after_review({"hr_action": "approve"}) == "__end__"
