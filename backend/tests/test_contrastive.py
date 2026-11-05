"""E20: the fine-tuned encoder is judged inside our ladder; these are its deterministic parts."""

from app.eval.contrastive import fused, prefixed


def test_e5_gets_its_query_and_passage_prefixes_by_folder_name_only():
    assert prefixed(["jd"], "query", "models/multilingual-e5-base-rjf-ft") == ["query: jd"]
    assert prefixed(["cv"], "passage", "runs/e5-x/bge-m3-ft") == ["cv"]


def test_fusion_ranks_eligible_by_dense_plus_rule_and_knockouts_last():
    pool = ["a", "b", "c", "k"]
    dense = {"a": 0.9, "b": 0.1, "c": 0.5, "k": 1.0}
    rule = {"a": 0.3, "b": 0.9, "c": 0.2, "k": 1.0}  # normalised sums over a, b, c: a 1.14, b 1.0, c 0.5
    assert fused(pool, dense, rule, ko={"k"}) == ["a", "b", "c", "k"]  # k has the best scores but is knocked out
