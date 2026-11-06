from app.eval import benchmark_v1_1 as v11


def test_human_grades_replace_silver_only_where_people_graded():
    gold = {"J": {"a": 1.5, "b": 0.0}, "K": {"c": 3.0}}
    out = v11.with_humans(gold, {("J", "a"): 1, ("X", "z"): 3})
    assert out == {"J": {"a": 1.0, "b": 0.0}, "K": {"c": 3.0}}
    assert gold["J"]["a"] == 1.5  # the v1 gold is left as it was


def test_the_two_annotators_merge_with_the_djinni_rule():
    from app.eval.djinni import adjudicate

    got, conflicts = adjudicate({("J", "a"): 2, ("J", "b"): 1, ("J", "c"): 0}, {("J", "a"): 2, ("J", "b"): 2, ("J", "c"): 3})
    assert got == {("J", "a"): 2, ("J", "b"): 1} and conflicts == [("J", "c")]


def test_order_tau_is_one_when_the_system_order_holds_and_negative_when_it_flips():
    a = {"P6": 0.85, "B4": 0.84, "BM25": 0.70}
    assert v11.order_tau(a, {"P6": 0.86, "B4": 0.83, "BM25": 0.69}) == 1.0
    assert v11.order_tau(a, {"P6": 0.70, "B4": 0.84, "BM25": 0.85}) < 0
