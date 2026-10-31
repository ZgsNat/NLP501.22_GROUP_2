"""N12 / L6: the investigator's controls hold whatever the model does (scripted model, no LLM)."""
import asyncio

from app.agent import investigate as inv
from app.agent.schemas import InvestigatorAction
from app.agent.scoring import knockout_reason
from app.nlp.pipeline import process

NOISY = process("CV-N", "Aut%%ati#n T#@t Engi%#er\nEXPERIENCE\nBank (2019 - 2023)\n"
                        "- Us#d S%lenium W#bDriver and J%va for ui t#sting.\n- Familiar with Appium.\n"
                        "SKILLS\nPython")
TYPO = process("CV-T", "## Experience\n### QA — Shop (2020–2023)\n- Wrote Selenuim tests in Java.")


class Script:
    """A fake model that replays actions (or raises), recording how often it was asked."""

    def __init__(self, actions):
        self.actions, self.calls = list(actions), 0

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, msgs):
        self.calls += 1
        a = self.actions.pop(0) if self.actions else self.actions_last
        self.actions_last = a
        if isinstance(a, Exception):
            raise a
        return InvestigatorAction(thought="t", **a)


def run(model, p, missing, max_steps=6):
    return asyncio.run(inv.investigate_cv(model, p, "QA", missing, {}, "test", max_steps))


def test_triggers_fire_on_parse_failures_not_on_plain_absence():
    assert "selenium" not in NOISY.skills and "selenium" not in TYPO.skills  # the parser misses both
    assert "garbled" in inv.triggers(NOISY, ["selenium"], {})
    assert "misspelt" in inv.triggers(TYPO, ["selenium"], {})
    assert inv.triggers(process("CV-C", "## Experience\n### QA (2020–2023)\n- Wrote Cypress tests."), ["selenium"], {}) is None


def test_search_finds_noisy_lines_and_verify_is_strict():
    assert "S%lenium" in inv.search(NOISY, "selenium").splitlines()[0]
    line = "- Us#d S%lenium W#bDriver and J%va for ui t#sting."
    fact, msg = inv.verify(NOISY, "selenium", line, {"selenium", "appium"})
    assert fact["strength"] == "used" and msg.startswith("accepted")
    assert inv.verify(NOISY, "selenium", "Built a Selenium grid on Kubernetes", {"selenium"})[1].startswith("rejected: the quote is not in the CV")
    assert inv.verify(NOISY, "cypress", line, {"selenium"})[1].startswith("rejected: 'cypress' is not one of")
    assert inv.verify(NOISY, "appium", line, {"appium"})[1].startswith("rejected: the quote does not name")
    assert "hedged" in inv.verify(NOISY, "appium", "- Familiar with Appium.", {"appium"})[1]


def test_hallucinated_proposal_is_rejected_then_a_real_one_ends_the_loop():
    model = Script([{"tool": "search", "arg": "selenium"},
                    {"tool": "propose", "arg": "selenium", "quote": "Led a Selenium framework for 40 engineers"},
                    {"tool": "propose", "arg": "selenium", "quote": "- Us#d S%lenium W#bDriver and J%va for ui t#sting."},
                    {"tool": "search", "arg": "never reached"}])
    out = run(model, NOISY, ["selenium"])
    assert [f["skill"] for f in out["facts"]] == ["selenium"] and model.calls == 3  # stopped once decided
    assert out["trace"][1]["observation"].startswith("rejected")
    p = inv.with_facts(NOISY, out["facts"])
    assert knockout_reason(NOISY, ["selenium"], {}) and knockout_reason(p, ["selenium"], {}) is None


def test_budget_repeats_and_invalid_output_are_bounded():
    looping = Script([{"tool": "search", "arg": "selenium"}] * 10)
    out = run(looping, NOISY, ["selenium"], max_steps=4)
    assert looping.calls == 4 and len(out["trace"]) == 4 and "already did" in out["trace"][1]["observation"]
    broken = run(Script([ValueError("bad json")]), NOISY, ["selenium"])
    assert broken["invalid"] and broken["facts"] == []  # Q1: unusable output → the knock-out stands


def test_match_bar_is_length_aware_and_search_splits_alternatives():
    assert inv.match_bar("jest") == 0.75 and inv.match_bar("playwright") == 0.65
    p = process("CV-P", "EXPERIENCE\nShop (2020 - 2023)\n- Wrote Pl#yw%i~ht tests in J%va.")
    assert inv.verify(p, "playwright", "- Wrote Pl#yw%i~ht tests in J%va.", {"playwright"})[0]  # 0.70: long name, noisy
    assert "Pl#yw%i~ht" in inv.search(p, "cypress|playwright")


def test_near_miss_ignores_short_and_multiword_aliases():
    clean = process("CV-K", "QA engineer with 6 years in testing.\n## Experience\n### QA (2020–2023)\n- Wrote Cypress tests.")
    assert inv.triggers(clean, ["api_testing", "typescript"], {}) is None  # "in testing", "is" are not typos


def test_implied_evidence_only_for_specific_descriptions_in_strong_cvs(monkeypatch):
    from app.agent.scoring import knockout_reason

    monkeypatch.setattr(inv, "get_settings", lambda: type("S", (), {"implied_evidence": True})())  # opt-in feature

    strong = process("CV-P", "## Experience\n### Quality engineer (2016–2024)\n"
                             "- I designed and built the browser test harness from scratch on top of the standard WebDriver protocol.\n"
                             "- The build server runs the service checks on every merge request.")
    assert "described without its name" in inv.triggers(strong, ["selenium", "jenkins"], {})
    web = "- I designed and built the browser test harness from scratch on top of the standard WebDriver protocol."
    ci = "- The build server runs the service checks on every merge request."
    fact, msg = inv.verify(strong, "selenium", web, {"selenium"})
    assert fact["strength"] == "implied" and msg.startswith("accepted as implied")
    assert inv.verify(strong, "jenkins", ci, {"jenkins"})[0]["strength"] == "implied"
    assert knockout_reason(inv.with_facts(strong, [fact]), ["selenium"], {}) is None
    # negatives: another tool's protocol, a different framework, a hedge, a weak CV
    appium = process("CV-A", "## Experience\n### QA (2018–2024)\n- Designed an Appium WebDriver framework for Android.")
    assert inv.verify(appium, "selenium", "- Designed an Appium WebDriver framework for Android.", {"selenium"})[0] is None
    cypress = process("CV-C", "## Experience\n### QA (2018–2024)\n- Designed the Cypress framework for browser tests.")
    assert inv.triggers(cypress, ["selenium"], {}) is None
    assert inv.verify(cypress, "selenium", "- Designed the Cypress framework for browser tests.", {"selenium"})[0] is None
    weak = process("CV-W", "## Experience\n### QA (2018–2024)\n- Ran the tests the build server started.")
    assert inv.triggers(weak, ["jenkins"], {}) is None  # not strong (depth < 3): no investigation
    hedged = process("CV-H", "## Experience\n### QA (2018–2024)\n- Designed a test framework.\n- Familiar with the WebDriver protocol.")
    assert inv.verify(hedged, "selenium", "- Familiar with the WebDriver protocol.", {"selenium"})[0] is None


def test_implied_evidence_is_off_by_default():
    strong = process("CV-P", "## Experience\n### Quality engineer (2016–2024)\n"
                             "- I designed and built the browser test harness on top of the standard WebDriver protocol.")
    assert inv.triggers(strong, ["selenium"], {}) is None
    assert inv.verify(strong, "selenium", "- I designed and built the browser test harness on top of the standard WebDriver protocol.",
                      {"selenium"})[0] is None
