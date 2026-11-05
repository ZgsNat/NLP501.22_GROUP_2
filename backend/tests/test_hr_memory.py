"""E19 / lecturer feedback 4: what HR teaches in one session is the next session's starting point, bounded."""

from app.agent import hr_memory
from app.agent.nodes import profile_weights, start_weights
from app.eval.hr_learning import sessions, simulated_hr

BASE = {"a": 0.5, "b": 0.3, "c": 0.2}


def test_drift_from_the_company_profile_is_capped_and_weights_stay_normalised():
    far = {"a": 0.0, "b": 0.0, "c": 1.0}  # Σ|Δw| = 1.6
    w = hr_memory.bounded(far, BASE, 0.4)
    assert abs(sum(w.values()) - 1) < 1e-3 and min(w.values()) >= 0
    assert sum(abs(w[k] - BASE[k]) for k in BASE) <= 0.4 + 1e-3
    assert hr_memory.bounded({"a": 0.45, "b": 0.35, "c": 0.2}, BASE, 0.4) == {"a": 0.45, "b": 0.35, "c": 0.2}


def test_a_session_is_remembered_per_company_and_read_back(tmp_path):
    p = tmp_path / "m.json"
    assert hr_memory.prior("bank", BASE, p) == (BASE, None)  # nothing learned yet
    assert hr_memory.remember("bank", {"a": 0.4, "b": 0.4, "c": 0.2}, BASE, pairs=0, p=p) is None  # nothing taught
    hr_memory.remember("bank", {"a": 0.4, "b": 0.4, "c": 0.2}, BASE, pairs=3, p=p)
    entry = hr_memory.remember("bank", {"a": 0.4, "b": 0.4, "c": 0.2}, BASE, pairs=2, p=p)
    assert (entry["sessions"], entry["pairs"]) == (2, 5)
    w, got = hr_memory.prior("bank", BASE, p)
    assert w == {"a": 0.4, "b": 0.4, "c": 0.2} and got["sessions"] == 2
    assert hr_memory.prior("startup", BASE, p)[1] is None  # another company learns nothing from bank
    hr_memory.forget("bank", p)
    assert hr_memory.prior("bank", BASE, p)[1] is None


def test_memory_with_other_criteria_is_ignored(tmp_path):
    p = tmp_path / "m.json"
    hr_memory.remember("bank", {"a": 1.0}, {"a": 1.0}, pairs=1, p=p)
    assert hr_memory.prior("bank", BASE, p) == (BASE, None)


def test_memory_is_off_by_default_so_benchmarks_never_depend_on_earlier_runs():
    assert start_weights("bank") == profile_weights("bank")


def test_simulated_hr_approves_relevant_rejects_irrelevant_and_skips_the_middle():
    d = simulated_hr(["x", "y", "z", "u"], {"x": 3, "y": 1, "z": 0, "u": 2.5})
    assert [(x["cv_id"], x["action"]) for x in d] == [("x", "approve"), ("z", "reject"), ("u", "approve")]


def test_sessions_are_the_jds_of_one_company_in_id_order():
    class J:
        def __init__(self, i, c):
            self.id, self.meta = i, {"company_profile": c}

    by = sessions([J("JD-3", "bank"), J("JD-1", "bank"), J("JD-2", "startup")], lambda j: j.meta["company_profile"])
    assert [j.id for j in by["bank"]] == ["JD-1", "JD-3"] and [j.id for j in by["startup"]] == ["JD-2"]
