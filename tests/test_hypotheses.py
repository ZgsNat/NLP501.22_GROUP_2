import csv
import json

import pytest

from cv_agent.eval import experiments as ex
from cv_agent.eval.metrics import wilcoxon_signed_rank

JDS = [f"jd{i}" for i in range(6)]
METRICS = ("recall@100", "mrr@10", "ndcg@10", "kendall_tau", "judge_calls", "judge_tokens", "llm_calls",
           "unsupported_rate", "impact_ratio", "agent_steps", "goal_met", "planner_fallback_rate")


def rows_for(config, values, ambiguous=False, fallback=False, **fixed):
    """One row per JD; `values` gives the per-JD value of every metric in `fixed`'s place."""
    out = []
    for jd, v in zip(JDS, values):
        row = {"config": config, "jd_id": jd, "ambiguous": ambiguous, "fallback": fallback,
               **{m: 0.0 for m in METRICS}, "impact_ratio": 1.0}
        row.update({m: (f(v) if callable(f) else f) for m, f in fixed.items()})
        out.append(row)
    return out


def ndcg_rows(config, ndcgs, **fixed):
    return rows_for(config, ndcgs, **{"ndcg@10": lambda v: v, **fixed})


def test_h3_is_rejected_when_the_ci_lower_bound_crosses_minus_epsilon():
    # mean difference -0.01 (within ε = 0.02) but one JD loses 0.10: the CI lower bound falls below -ε
    p6 = ndcg_rows("P6", [0.80, 0.80, 0.80, 0.80, 0.80, 0.70], judge_calls=8.0)
    llm_all = ndcg_rows("LLM-all", [0.80, 0.80, 0.80, 0.80, 0.80, 0.76], judge_calls=20.0)
    h3 = ex.evaluate_hypotheses(p6 + llm_all)["H3"]
    assert h3["details"]["ndcg@10"]["mean_diff"] > -ex.H3_EPSILON
    assert h3["details"]["ndcg@10"]["ci_low"] < -ex.H3_EPSILON
    assert h3["accepted"] is False


def test_h3_is_accepted_when_the_whole_ci_stays_above_minus_epsilon():
    p6 = ndcg_rows("P6", [0.80, 0.81, 0.79, 0.80, 0.80, 0.80], judge_calls=8.0)
    llm_all = ndcg_rows("LLM-all", [0.80, 0.80, 0.80, 0.80, 0.80, 0.80], judge_calls=20.0)
    assert ex.evaluate_hypotheses(p6 + llm_all)["H3"]["accepted"] is True


def test_h4_needs_the_whole_ci_below_zero():
    noisy = rows_for("P6", [0.1, 0.5, 0.1, 0.5, 0.1, 0.45], unsupported_rate=lambda v: v)
    a1 = rows_for("A1", [0.3, 0.3, 0.3, 0.3, 0.3, 0.29], unsupported_rate=lambda v: v)
    assert ex.evaluate_hypotheses(noisy + a1)["H4"]["accepted"] is False
    clear = rows_for("P6", [0.1] * 6, unsupported_rate=lambda v: v)
    assert ex.evaluate_hypotheses(clear + a1)["H4"]["accepted"] is True


def test_h0_needs_a_significant_gain_not_just_a_positive_mean():
    p6 = ndcg_rows("P6", [0.9, 0.5, 0.9, 0.5, 0.9, 0.52])
    b4 = ndcg_rows("B4", [0.7] * 6)
    assert ex.evaluate_hypotheses(p6 + b4)["H0"]["accepted"] is False


def test_every_hypothesis_states_its_decision_rule():
    hyp = ex.evaluate_hypotheses(ndcg_rows("P6", [0.8] * 6) + ndcg_rows("B4", [0.8] * 6))
    assert all(h.get("rule") for h in hyp.values())


def test_wilcoxon_reports_the_smallest_p_value_n_pairs_can_reach():
    res = wilcoxon_signed_rank([2] * 6, [1] * 6)
    assert res["p_floor"] == pytest.approx(2 / 2 ** 6)


def test_rescore_rewrites_hypotheses_from_results_csv(tmp_path):
    rows = (ndcg_rows("P6", [0.80, 0.80, 0.80, 0.80, 0.80, 0.70], judge_calls=8.0)
            + ndcg_rows("LLM-all", [0.80, 0.80, 0.80, 0.80, 0.80, 0.76], judge_calls=20.0))
    with open(tmp_path / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "summary.json").write_text(json.dumps({"hypotheses": {"H3": {"accepted": True}}, "n_jds": 6}),
                                           encoding="utf-8")
    summary = ex.rescore(tmp_path)
    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["hypotheses"]["H3"]["accepted"] is False and saved["hypotheses"]["H3"]["accepted"] is False
    assert saved["n_jds"] == 6 and saved["hypothesis_rule"] == ex.HYPOTHESIS_RULE


def test_rescore_reads_booleans_from_csv_text():
    row = ex._parse_row({"config": "P6", "jd_id": "a", "ambiguous": "False", "fallback": "True", "ndcg@10": "0.5"})
    assert row["ambiguous"] is False and row["fallback"] is True and row["ndcg@10"] == 0.5


def test_rescore_cli_prints_the_new_verdicts(tmp_path, capsys):
    from cv_agent.cli import main
    test_rescore_rewrites_hypotheses_from_results_csv(tmp_path)
    assert main(["rescore", str(tmp_path)]) == 0
    assert "H3 BÁC BỎ" in capsys.readouterr().out


def test_finetune_cli_rejects_seeds_that_are_not_integers(capsys, monkeypatch):
    from cv_agent.cli import main
    from cv_agent.training import finetune as ft
    monkeypatch.setattr(ft, "run", lambda cfg, raw: pytest.fail("must not train"))
    assert main(["finetune", "--target", "dense", "--seeds", "42,abc"]) == 2
    assert "--seeds" in capsys.readouterr().out


def ag_rows(ag_ndcg, ag_unsupported, fallback_rate=0.0):
    ag = rows_for("AG", ag_ndcg, **{"ndcg@10": lambda v: v, "unsupported_rate": ag_unsupported,
                                    "planner_fallback_rate": fallback_rate})
    rules = rows_for("AG-rules", [0.8] * 6, **{"ndcg@10": lambda v: v, "unsupported_rate": 0.1})
    return ag + rules


def test_h7_accepts_a_planner_that_is_not_worse_on_any_criterion():
    assert ex.evaluate_hypotheses(ag_rows([0.8] * 6, 0.1))["H7"]["accepted"] is True


def test_h7_rejects_a_clearly_higher_unsupported_rate_even_without_significance():
    noisy_worse = [0.1, 0.3, 0.1, 0.3, 0.1, 0.3]  # mean +0.10, CI too wide to be "significant"
    rows = ag_rows([0.8] * 6, 0.1)
    rows = [dict(r, unsupported_rate=v) if r["config"] == "AG" else r
            for r, v in zip(rows, noisy_worse + [None] * 6)]
    assert ex.evaluate_hypotheses(rows)["H7"]["accepted"] is False


def test_h7_rejects_when_rules_replace_too_many_planner_decisions():
    assert ex.evaluate_hypotheses(ag_rows([0.8] * 6, 0.1, fallback_rate=0.5))["H7"]["accepted"] is False


def test_h7_rejects_when_the_fallback_rate_was_not_measured():
    assert ex.evaluate_hypotheses(ag_rows([0.8] * 6, 0.1, fallback_rate=float("nan")))["H7"]["accepted"] is False


def test_too_few_jds_never_count_as_ci_support():
    p6 = ndcg_rows("P6", [0.9, 0.9], unsupported_rate=0.1)[:2]
    a1 = rows_for("A1", [0.5, 0.5], unsupported_rate=0.5)[:2]
    assert ex.evaluate_hypotheses(p6 + a1)["H4"]["accepted"] is False


def test_parse_row_keeps_identifiers_as_text_and_reads_csv_numbers():
    row = ex._parse_row({"config": "B0", "jd_id": "001", "ambiguous": "1", "fallback": "0", "ndcg@10": "nan",
                         "comprehensiveness": ""})
    assert row["jd_id"] == "001" and row["ambiguous"] == 1.0 and not row["fallback"]
    assert row["ndcg@10"] != row["ndcg@10"] and row["comprehensiveness"] == ""


def test_rescore_keeps_a_backup_of_the_first_summary(tmp_path):
    test_rescore_rewrites_hypotheses_from_results_csv(tmp_path)
    ex.rescore(tmp_path)
    backup = json.loads((tmp_path / "summary.pre-rescore.json").read_text(encoding="utf-8"))
    assert backup["hypotheses"]["H3"]["accepted"] is True  # the verdict before the CI rule


def test_rescore_cli_reports_a_run_without_results_and_carries_on(tmp_path, capsys):
    from cv_agent.cli import main
    good = tmp_path / "good"
    good.mkdir()
    test_rescore_rewrites_hypotheses_from_results_csv(good)
    assert main(["rescore", str(tmp_path / "missing"), str(good)]) == 1
    out = capsys.readouterr().out
    assert "missing" in out and "H3 BÁC BỎ" in out


def test_finetune_cli_rejects_duplicate_seeds(monkeypatch):
    from cv_agent.cli import main
    from cv_agent.training import finetune as ft
    monkeypatch.setattr(ft, "run", lambda cfg, raw: pytest.fail("must not train"))
    assert main(["finetune", "--target", "dense", "--seeds", "42,42"]) == 2
