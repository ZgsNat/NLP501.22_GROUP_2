from app.eval.nlp_ablation import VARIANTS, no_dates, no_depth, no_evidence, no_hedge, per_jd_ndcg, table
from app.nlp.pipeline import process

CV = """# QA

## Summary
QA engineer, 9 years in testing.

## Experience
### Automation Engineer — Shop (Jan 2020 – Dec 2022)
- Designed a Playwright framework in TypeScript for the web app.
- Familiar with Java from a course.

## Skills
Cypress, Jenkins
"""


def _strength(p, skill):
    return p.best(skill).strength


def test_no_hedge_turns_a_hedged_mention_into_a_real_one():
    p = process("CV-A", CV)
    assert _strength(p, "java") == "hedged"
    q = no_hedge(p)
    assert _strength(q, "java") == "used"  # it sits in an Experience bullet
    assert _strength(p, "java") == "hedged"  # the original profile is not mutated


def test_no_evidence_counts_listed_as_used_and_keeps_hedges():
    p = process("CV-A", CV)
    assert _strength(p, "cypress") == "listed"
    q = no_evidence(p)
    assert _strength(q, "cypress") == "used"
    assert _strength(q, "java") == "hedged"


def test_no_depth_clears_the_verb_cue():
    p = process("CV-A", CV)
    assert p.depth_cue >= 3  # "Designed a ... framework"
    q = no_depth(p)
    assert (q.depth_cue, q.depth_cue_line) == (0, None)


def test_no_dates_falls_back_to_the_years_claim():
    p = process("CV-A", CV)
    assert p.years_total == 3.0  # parsed from the dated role
    q = no_dates(p)
    assert q.years_total == 9.0 and q.years_source[0].startswith("claimed")


def test_every_variant_keeps_the_text_and_chunks():
    p = process("CV-A", CV)
    for fn in VARIANTS.values():
        q = fn(p)
        assert q.text == p.text and q.chunks == p.chunks


def test_table_reports_delta_against_the_full_pipeline():
    gold = {"J1": {"a": 3, "b": 0}, "J2": {"a": 3, "b": 0}}
    rankings = {"full pipeline": {"J1": ["a", "b"], "J2": ["a", "b"]}, "− x": {"J1": ["b", "a"], "J2": ["b", "a"]}}
    scores = per_jd_ndcg(rankings, gold)
    assert scores["full pipeline"] == {"J1": 1.0, "J2": 1.0}
    rows = table(scores, ["J1", "J2"])
    assert rows[2].startswith("| **full pipeline** | **1.000**")
    assert "| − x |" in rows[3] and "-0.3" in rows[3]


def test_traps_at_10_counts_irrelevant_traps_in_the_top_10():
    from app.eval.nlp_ablation import traps_at_10

    gold = {"J1": {"t1": 0, "t2": 3, "a": 3}, "J2": {"t1": 0, "t2": 0, "a": 3}}
    rankings = {"full": {"J1": ["a", "t2", "t1"], "J2": ["a"] + ["x"] * 9 + ["t1"]}}
    # J1: t1 is a trap and irrelevant (t2 is a trap but relevant to J1); J2: t1 sits at rank 11, outside the top 10
    assert traps_at_10(rankings, gold, frozenset({"t1", "t2"})) == {"full": 0.5}


def test_every_variant_has_a_flat_results_file_name():
    from app.eval.nlp_ablation import result_name

    assert all("/" not in result_name(v) for v in VARIANTS)
