from app.eval.human import SAMPLE_SIZE, annotators, sample


def test_human_sample_contains_every_annotator_disagreement():
    a, b = annotators().values()
    disagreements = {(jd, cv) for jd in a for cv in a[jd] if a[jd][cv] != b[jd][cv]}
    pairs = sample(seed=13)
    selected = {(p["jd"], p["cv"]) for p in pairs}

    assert len(disagreements) == 94
    assert len(pairs) == len(selected) == SAMPLE_SIZE == 120
    assert disagreements <= selected
    assert pairs == sample(seed=13)



def test_sides_compares_a_human_with_both_silver_annotators_without_rounding():
    from app.eval.human import sides

    a = {"J": {"1": 0, "2": 1, "3": 2, "4": 3}}
    b = {"J": {"1": 0, "2": 2, "3": 3, "4": 1}}
    human = {("J", "1"): 0, ("J", "2"): 1, ("J", "3"): 3, ("J", "4"): 2}
    assert sides(human, a, b) == {"controls": 1, "control_same": 1, "disagreements": 3, "as_a": 1, "as_b": 1,
                                  "between": 3, "a_stricter": 2}
