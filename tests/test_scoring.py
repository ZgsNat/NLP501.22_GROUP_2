from cv_agent.ranking.scoring import rank, score_candidate
from cv_agent.report.explain import build_cards, counterfactuals, decomposition


def test_score_formula_and_bounds(senior_cv, jd, profile):
    c = score_candidate(senior_cv, jd, profile)
    assert c.must_have_ok
    assert 0 <= c.score <= 1
    expected = sum(profile.weights[a] * c.axis_scores[a] for a in profile.weights)
    assert abs(c.score - expected) < 1e-9


def test_knockout_zeroes_score_but_records_reason(manual_cv, jd, profile):
    c = score_candidate(manual_cv, jd, profile)
    assert not c.must_have_ok and c.score == 0
    assert c.knockout_reasons and c.potential_score > 0


def test_ranking_orders_by_score(senior_cv, manual_cv, jd, profile):
    ranked = rank([score_candidate(manual_cv, jd, profile), score_candidate(senior_cv, jd, profile)])
    assert [c.cv_id for c in ranked] == ["senior", "manual"]


def test_l2_decomposition_sums_to_score(senior_cv, manual_cv, jd, profile):
    for cv in (senior_cv, manual_cv):
        c = score_candidate(cv, jd, profile)
        assert abs(sum(r["contribution"] for r in decomposition(c, profile)) - c.score) < 1e-9


def test_counterfactual_adding_missing_must_have_lifts_score(manual_cv, jd, profile):
    c = score_candidate(manual_cv, jd, profile)
    cfs = counterfactuals(manual_cv, jd, profile, c)
    assert cfs and all(cf["delta"] >= 0 for cf in cfs)


def test_cards_have_six_layers(senior_cv, manual_cv, jd, profile):
    scored = rank([score_candidate(cv, jd, profile) for cv in (senior_cv, manual_cv)])
    cards = build_cards(scored, {"senior": senior_cv, "manual": manual_cv}, jd, profile, {}, [], top_n=10)
    assert len(cards) == 2
    assert cards[0].L1["must_have_ok"] and cards[0].L4["vs"] == "manual"
    assert cards[1].L1["note"]  # knock-out explained, not removed
