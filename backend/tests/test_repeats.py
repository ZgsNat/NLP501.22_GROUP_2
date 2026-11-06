from app.eval import repeats


def test_wilson_interval_is_bounded_and_honest_at_the_edges():
    lo, hi = repeats.wilson(3, 3)
    assert hi == 1.0 and 0.4 < lo < 0.5  # 3/3 is not proof of 100%
    lo, hi = repeats.wilson(0, 3)
    assert lo == 0.0 and 0.5 < hi < 0.6
    assert repeats.wilson(0, 0) == (0.0, 0.0)


def test_rates_pool_repeats_and_jds_and_skip_not_applicable():
    runs = {1: [{"task": "shortlist", "jd": "A", "checks": {"explained": True, "asked": None}},
                {"task": "shortlist", "jd": "B", "checks": {"explained": False, "asked": None}}],
            2: [{"task": "shortlist", "jd": "A", "checks": {"explained": True, "asked": True}}]}
    assert repeats.rates(runs) == [("shortlist", "explained", 2, 3), ("shortlist", "asked", 1, 1)]


def test_section_flags_checks_that_do_not_always_pass(tmp_path, monkeypatch):
    from app.config import get_settings

    monkeypatch.setattr(get_settings(), "runtime_dir", tmp_path)
    repeats.save("demo", 1, [{"task": "why", "jd": "J", "checks": {"used compare": False}}])
    repeats.save("demo", 2, [{"task": "why", "jd": "J", "checks": {"used compare": True}}])
    text = "\n".join(repeats.section(repeats.load("demo"), "Demo"))
    assert "2 independent repeats" in text and "repeat 1: 0/1, repeat 2: 1/1" in text
    assert "| why | used compare | 1/2 | 50% ⚠ |" in text
