import json
import math

import pytest

from cv_agent.config import load_config
from cv_agent.data.synthetic import generate
from cv_agent.eval.experiments import load_jd_specs, run_all
from cv_agent.eval.fairness import impact_ratio, name_swap_delta, swap_name
from cv_agent.eval.metrics import wilcoxon_signed_rank
from cv_agent.ingest.parsers import UnsupportedFormat, read_text
from cv_agent.ingest.pipeline import ingest_dir, ingest_text
from cv_agent.orchestration.graph import build_graph, build_runtime
from tests.conftest import SENIOR_CV


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    out = tmp_path_factory.mktemp("syn")
    specs = {s["id"]: {**s["gold_spec"], "ambiguous": s.get("ambiguous", False)} for s in load_jd_specs()}
    generate(out, specs, n=40, seed=7)
    return out


def test_synthetic_generator_masks_cleanly(small_data):
    meta = json.loads((small_data / "meta.json").read_text(encoding="utf-8"))
    cvs = ingest_dir(small_data / "cvs")
    assert meta["synthetic"] and len(cvs) == 42
    by_id = {c.cv_id: c for c in cvs}
    for cid, name in meta["names"].items():
        assert name not in by_id[cid].masked_text
    assert any(f.startswith("duplicate_of:") for c in cvs for f in c.flags)


def test_run_all_produces_results_and_hypotheses(small_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # relative cache dir stays in tmp
    run_dir = run_all(["B0", "B2", "B4", "LLM-all", "P6", "A1", "P6-noclar"], small_data, out_root=tmp_path,
                      overrides={"llm": {"backend": "heuristic"}})
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert {"H0", "H1", "H2", "H3", "H4", "H5", "H6"} <= set(summary["hypotheses"])
    assert summary["means"]["P6"]["judge_calls"] <= summary["means"]["LLM-all"]["judge_calls"]
    assert summary["means"]["P6"]["unsupported_rate"] <= summary["means"]["A1"]["unsupported_rate"]
    assert (run_dir / "results.csv").exists()


def test_fairness_helpers():
    ir = impact_ratio(["a", "c"], {"a": "M", "b": "M", "c": "F", "d": "F"})
    assert ir["impact_ratio"] == 1.0
    swapped = swap_name("Nguyễn Văn Hùng\nGender: Male", "Nguyễn Văn Hùng", __import__("random").Random(0))
    assert "Nguyễn Văn Hùng" not in swapped and "Female" in swapped
    rank_fn = lambda texts: sorted(texts, key=lambda k: ingest_text(texts[k], k).masked_text)  # noqa: E731
    res = name_swap_delta(rank_fn, {"x": SENIOR_CV, "y": SENIOR_CV.replace("6 years", "2 years")},
                          {"x": "Nguyen Van Hung", "y": "Nguyen Van Hung"})
    assert res["mean_abs_delta_rank"] == 0  # masked text identical -> rank unchanged


def test_wilcoxon_no_difference():
    assert wilcoxon_signed_rank([1, 2], [1, 2])["p_value"] == 1.0
    assert not math.isnan(wilcoxon_signed_rank([3, 4, 5, 6], [1, 1, 2, 1])["p_value"])


def test_parsers_txt_docx_pdf(tmp_path):
    import docx
    import pymupdf as fitz
    d = docx.Document()
    d.add_paragraph("Automation tester with Selenium")
    d.save(tmp_path / "a.docx")
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 72), "Playwright and Python")
    pdf.save(tmp_path / "b.pdf")
    (tmp_path / "c.txt").write_text("Appium", encoding="utf-8")
    assert "Selenium" in read_text(tmp_path / "a.docx")[0]
    assert "Playwright" in read_text(tmp_path / "b.pdf")[0]
    with pytest.raises(UnsupportedFormat):
        read_text(tmp_path / "x.rtf")


def test_simple_graph_runner_matches_langgraph(jd):
    cfg = load_config("P6", overrides={"llm": {"backend": "heuristic"}, "paths": {"cache_dir": None}})
    cvs = [ingest_text(SENIOR_CV, "a"), ingest_text(SENIOR_CV.replace("Selenium WebDriver", "Cypress"), "b")]
    results = []
    for prefer in (True, False):
        rt = build_runtime(cfg)
        graph, _ = build_graph(rt, prefer_langgraph=prefer)
        out = graph.invoke({"run_id": "t", "jd": jd, "cvs": {c.cv_id: c for c in cvs}, "trace": []})
        results.append([c.cv_id for c in out["ranking"]])
    assert results[0] == results[1]
