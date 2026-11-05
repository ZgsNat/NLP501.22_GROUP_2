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

