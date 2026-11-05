"""Public benchmark (lecturer feedback 3): Djinni, TREC-pooled pairs graded by two human annotators."""

from app.eval.bench import agreement, djinni_gold, grade
from app.eval.run import score_system


def test_a_sheet_cell_becomes_a_grade_or_nothing():
    assert [grade(x) for x in (2, "3", "1 - một phần", " 0", None, "", "x", 7)] == [2, 3, 1, 0, None, None, None, None]


def test_gold_is_the_mean_of_whoever_graded_the_pair():
    labels = {"A": {("jd", "c1"): 3, ("jd", "c2"): 0}, "B": {("jd", "c1"): 2, ("jd", "c3"): 1}}
    assert djinni_gold(labels) == {"jd": {"c1": 2.5, "c2": 0.0, "c3": 1.0}}


def test_agreement_uses_only_pairs_both_annotators_graded():
    labels = {"A": {("j", "a"): 3, ("j", "b"): 0, ("j", "c"): 2}, "B": {("j", "a"): 3, ("j", "b"): 0, ("j", "d"): 1}}
    ag = agreement(labels)
    assert ag["pairs"] == 2 and ag["exact"] == 1.0 and abs(ag["kappa"] - 1.0) < 1e-9
    assert agreement({"A": labels["A"]}) == {"pairs": 0}


def test_unjudged_cvs_are_skipped_not_scored_as_irrelevant():
    g = {"jd": {"good": 3.0, "bad": 0.0}}
    judged_only = score_system({"jd": ["unjudged1", "unjudged2", "good", "bad"]}, g, frozenset())
    assert judged_only["ndcg@10"] == [1.0]  # condensed list: "good" is first among judged CVs
