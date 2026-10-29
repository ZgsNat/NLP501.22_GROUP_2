import pytest

from app.eval.metrics import (
    bootstrap_ci,
    kendall_tau_b,
    ndcg,
    paired_bootstrap_p,
    precision,
    tau_vs_gold,
    weighted_kappa,
)

GRADES = {"a": 3, "b": 2, "c": 0, "d": 1}


def test_ndcg_is_one_for_the_ideal_order_and_lower_otherwise():
    assert ndcg(["a", "b", "d", "c"], GRADES, 4) == pytest.approx(1.0)
    assert ndcg(["c", "d", "b", "a"], GRADES, 4) < ndcg(["b", "a", "d", "c"], GRADES, 4) < 1.0
    assert ndcg(["x"], {"x": 0}, 10) == 0.0  # no relevant item → defined as 0


def test_precision_counts_grade_two_and_above():
    assert precision(["a", "c", "b", "d"], GRADES, 2) == 0.5


def test_kendall_tau_b_handles_ties():
    assert kendall_tau_b([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert kendall_tau_b([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert 0 < kendall_tau_b([1, 2, 3, 4], [0, 0, 1, 1]) < 1
    assert tau_vs_gold(["a", "b", "d", "c"], GRADES) == pytest.approx(1.0)


def test_weighted_kappa():
    assert weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == pytest.approx(1.0)
    assert weighted_kappa([0, 1, 2, 3], [1, 1, 2, 3]) > weighted_kappa([0, 1, 2, 3], [3, 1, 2, 0])


def test_bootstrap_and_paired_test():
    mean, lo, hi = bootstrap_ci([0.5, 0.6, 0.7, 0.8])
    assert lo <= mean <= hi and mean == pytest.approx(0.65)
    assert paired_bootstrap_p([0.9, 0.8, 0.85], [0.5, 0.4, 0.45]) == 0.0
    assert paired_bootstrap_p([0.5, 0.4, 0.45], [0.9, 0.8, 0.85]) == 1.0
