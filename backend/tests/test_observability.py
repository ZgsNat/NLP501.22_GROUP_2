import pytest

from app import observability as obs


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    class S:
        runtime_dir = tmp_path

    monkeypatch.setattr(obs, "get_settings", lambda: S)
    monkeypatch.setattr(obs, "_usage", {})
    return tmp_path


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x" * 65])
def test_run_dir_rejects_path_traversal(runtime, bad):
    with pytest.raises(ValueError):
        obs.run_dir(bad)


def test_reads_have_no_side_effects(runtime):
    assert obs.read_logs("unknown") == []
    assert not (runtime / "runs" / "unknown").exists()


def test_usage_and_logs_survive_a_restart(runtime):
    c = obs.usage_counter("r1")
    c.by_node["assess"].update(calls=2, input_tokens=100, output_tokens=40)
    obs.save_usage("r1")
    obs.append_log("r1", {"type": "log", "node": "search", "level": "loop", "message": "expanded"})

    obs._usage.clear()  # simulate server restart
    assert obs.usage_counter("r1").summary()["total_tokens"] == 140
    assert obs.read_logs("r1")[0]["message"] == "expanded"
