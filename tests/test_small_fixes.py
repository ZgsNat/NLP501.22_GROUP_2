import json
import math

import pytest

from cv_agent.criteria.profiles import list_profiles, profile_filename, save_profile
from cv_agent.eval.experiments import _h5
from cv_agent.ingest import taxonomy
from cv_agent.ranking.scoring import rank
from cv_agent.schema import CriteriaProfile, ScoredCandidate


def test_rank_orders_knocked_out_candidates_by_potential_score():
    # nobody passes the must-haves: score is 0 for all, potential decides
    ranking = rank([ScoredCandidate("a", score=0.0, potential_score=0.2, rerank_score=0.9),
                    ScoredCandidate("b", score=0.0, potential_score=0.7, rerank_score=0.1),
                    ScoredCandidate("c", score=0.5, potential_score=0.5)])
    assert [c.cv_id for c in ranking] == ["c", "b", "a"]


def _p6_rows(ratios):
    return [{"config": "P6", "jd_id": j, "impact_ratio": v} for j, v in ratios.items()]


def test_h5_rejects_when_one_jd_fails_even_if_mean_passes():
    h = _h5(_p6_rows({"a": 1.0, "b": 1.0, "c": 0.7}))
    assert not h["accepted"] and h["failing_jds"] == ["c"] and h["impact_ratio_mean"] >= 0.8


def test_h5_accepts_when_every_jd_passes_and_ignores_nan():
    h = _h5(_p6_rows({"a": 0.85, "b": float("nan")}))
    assert h["accepted"] and h["impact_ratio_min"] == 0.85


def test_h5_without_groups_is_not_accepted():
    h = _h5(_p6_rows({"a": float("nan")}))
    assert not h["accepted"] and math.isnan(h["impact_ratio_min"])


@pytest.mark.parametrize("name,expected", [("Outsourcing – Mobile", "outsourcing_mobile.yaml"),
                                           ("../../etc/passwd", "etc_passwd.yaml")])
def test_profile_filename_is_safe(name, expected):
    assert profile_filename(name) == expected


def test_profile_filename_rejects_empty_name():
    with pytest.raises(ValueError):
        profile_filename(" / ")


def test_save_profile_roundtrip_and_no_overwrite(tmp_path):
    weights = {**{a: 0.0 for a in ("tools", "depth", "languages", "test_types", "cicd", "experience",
                                   "certification", "foreign_lang")}, "tools": 0.6, "depth": 0.4}
    path = save_profile(CriteriaProfile("My Team", weights, "x"), tmp_path)
    assert path.parent == tmp_path and list_profiles(tmp_path)["My Team"].weights["tools"] == 0.6
    with pytest.raises(FileExistsError):
        save_profile(CriteriaProfile("my team", weights), tmp_path)
    save_profile(CriteriaProfile("My Team", weights), tmp_path, overwrite=True)


def test_load_esco_csv_attaches_real_uris(tmp_path, monkeypatch):
    monkeypatch.setattr(taxonomy, "SKILLS", dict(taxonomy.SKILLS))
    csv_path = tmp_path / "skills_en.csv"
    csv_path.write_text('conceptUri,preferredLabel,altLabels\n'
                        'http://data.europa.eu/esco/skill/1,use Selenium,"selenium webdriver\nwebdriver"\n'
                        'http://data.europa.eu/esco/skill/2,Selenium grid,\n'
                        'http://data.europa.eu/esco/skill/3,knit sweaters,\n', encoding="utf-8")
    assert taxonomy.load_esco_csv(csv_path) == 1
    assert taxonomy.SKILLS["selenium"].esco_uri == "http://data.europa.eu/esco/skill/1"  # first match wins
    assert taxonomy.SKILLS["playwright"].esco_uri.startswith("esco-lite:")


def test_load_esco_csv_rejects_wrong_file(tmp_path):
    bad = tmp_path / "x.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        taxonomy.load_esco_csv(bad)


def test_export_cvs_writes_canonical_json_without_raw_text(tmp_path):
    from cv_agent.cli import main
    from tests.conftest import SENIOR_CV
    src = tmp_path / "cvs"
    src.mkdir()
    (src / "cand1.txt").write_text(SENIOR_CV, encoding="utf-8")
    assert main(["export-cvs", "--cvs", str(src), "--out", str(tmp_path / "out")]) == 0
    data = json.loads((tmp_path / "out" / "cand1.json").read_text(encoding="utf-8"))
    assert "raw_text" not in data and data["cv_id"] == "cand1" and data["entities"]
    assert "[EMAIL]" in data["masked_text"] or data["pii_found"]


def test_contrastive_explains_knocked_out_pair_by_axis_scores():
    from cv_agent.report.explain import contrastive
    a = ScoredCandidate("a", score=0.0, potential_score=0.6, must_have_ok=False, axis_scores={"tools": 1.0})
    b = ScoredCandidate("b", score=0.0, potential_score=0.2, must_have_ok=False, axis_scores={"tools": 0.3})
    out = contrastive(a, b)
    assert out["delta_potential"] == 0.4 and out["advantages"][0]["axis"] == "tools"


def test_save_profile_creates_missing_directory(tmp_path):
    weights = {"tools": 1.0}
    path = save_profile(CriteriaProfile("New", weights), tmp_path / "nested" / "dir")
    assert path.exists()


@pytest.mark.parametrize("raw,expected", [
    ('{"winner": "B", "span": "<CANDIDATE B: 3 năm kinh nghiệm', {"winner": "B"}),
    ('{"claims": [{"axis": "tools", "line": 2}, {"axis": "cicd"', {"claims": [{"axis": "tools", "line": 2}]}),
    # "line": 1 may be the start of 12: the half-written value is dropped, never trusted
    ('{"claims": [{"axis": "tools", "line": 2}, {"axis": "cicd", "line": 1',
     {"claims": [{"axis": "tools", "line": 2}, {"axis": "cicd"}]}),
    ('{"span": "a, \\"b, c", "winner": "A', {"span": 'a, "b, c'}),
    ('```json\n{"winner": "A"}\n```', {"winner": "A"}),
])
def test_parse_json_repairs_output_cut_by_token_limit(raw, expected):
    from cv_agent.llm.backends import _parse_json
    assert _parse_json(raw) == expected


def test_parse_json_still_rejects_garbage():
    from cv_agent.llm.backends import LLMError, _parse_json
    with pytest.raises(LLMError):
        _parse_json("no json here")
    with pytest.raises(LLMError):
        _parse_json('{"winner": ')


class _ReverseReranker:
    """Prefers whatever criteria likes least, to show it cannot override the criteria score."""
    name = "reverse"

    def __init__(self, order):
        self.order = order

    def score(self, jd, cvs):
        return [float(self.order.index(c.cv_id)) for c in cvs]


def test_rerank_within_criteria_only_breaks_ties_and_keeps_everyone(jd, profile):
    from types import SimpleNamespace
    from cv_agent.ingest.pipeline import ingest_text
    from cv_agent.orchestration.nodes import _rerank_within_criteria
    from tests.conftest import MANUAL_CV, SENIOR_CV
    cvs = {"s1": ingest_text(SENIOR_CV, "s1"), "s2": ingest_text(SENIOR_CV, "s2"), "m": ingest_text(MANUAL_CV, "m")}
    rt = SimpleNamespace(cfg={"rerank": {"top_n": 3}}, reranker=_ReverseReranker(["s1", "s2", "m"]))
    out = _rerank_within_criteria(rt, {"cvs": cvs, "jd": jd, "profile": profile}, [(c, 0.0) for c in cvs])
    assert [c.cv_id for c in out] == ["s2", "s1", "m"]  # tie s1/s2 broken by reranker, m stays last
    assert out[0].rerank_score == 1.0


def test_rerank_within_criteria_leaves_tail_untouched(jd, profile):
    from types import SimpleNamespace
    from cv_agent.ingest.pipeline import ingest_text
    from cv_agent.orchestration.nodes import _rerank_within_criteria
    from tests.conftest import MANUAL_CV, SENIOR_CV
    cvs = {"s": ingest_text(SENIOR_CV, "s"), "m": ingest_text(MANUAL_CV, "m")}
    rt = SimpleNamespace(cfg={"rerank": {"top_n": 1}}, reranker=_ReverseReranker(["s", "m"]))
    out = _rerank_within_criteria(rt, {"cvs": cvs, "jd": jd, "profile": profile}, [("m", 0.0), ("s", 0.0)])
    assert [c.cv_id for c in out] == ["s", "m"] and out[1].rerank_score == 0.0


def test_rank_pipeline_runs_in_within_criteria_mode(jd, profile):
    from cv_agent.config import load_config
    from cv_agent.ingest.pipeline import ingest_text
    from cv_agent.orchestration.graph import build_runtime, run_pipeline
    from tests.conftest import MANUAL_CV, SENIOR_CV
    cfg = load_config("B4", overrides={"llm": {"backend": "heuristic"}, "rerank": {"mode": "within_criteria"},
                                      "paths": {"cache_dir": None}})
    out = run_pipeline(build_runtime(cfg), jd, [ingest_text(SENIOR_CV, "s"), ingest_text(MANUAL_CV, "m")], profile)
    assert [c.cv_id for c in out["ranking"]] == ["s", "m"]
    assert any("within" in t or "phá hoà" in t for t in out["trace"])


def test_default_rerank_mode_is_within_criteria_and_filter_still_works(jd, profile):
    from cv_agent.config import load_config
    from cv_agent.ingest.pipeline import ingest_text
    from cv_agent.orchestration.graph import build_runtime, run_pipeline
    from tests.conftest import MANUAL_CV, SENIOR_CV
    assert load_config("B4")["rerank"]["mode"] == "within_criteria"
    cfg = load_config("B4", overrides={"llm": {"backend": "heuristic"}, "rerank": {"mode": "filter", "top_n": 1},
                                      "paths": {"cache_dir": None}})
    cvs = [ingest_text(SENIOR_CV, "s"), ingest_text(MANUAL_CV, "m")]
    out = run_pipeline(build_runtime(cfg), jd, cvs, profile)
    assert {c.cv_id for c in out["ranking"]} == {"s", "m"}  # tail kept for Recall@k
    assert any("rerank[lexical] 2 -> 1" in t for t in out["trace"])
