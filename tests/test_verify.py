from cv_agent.schema import Claim
from cv_agent.verify.evidence import check_claim, locate_span, verify_claims
from cv_agent.verify.nli import LexicalNLI

CV_TEXT = "Designed a Page Object Model framework with Selenium WebDriver and Java.\nISTQB CTFL certified."
NLI = LexicalNLI()


def test_locate_span_exact_and_fuzzy():
    assert locate_span("selenium webdriver", CV_TEXT)
    assert locate_span("Page Object Modle framework", CV_TEXT)  # typo -> fuzzy
    assert locate_span("expert in Playwright", CV_TEXT) is None


def test_supported_claim_passes():
    ok, _ = check_claim(Claim("c", "Has Selenium experience", "Selenium WebDriver and Java"), CV_TEXT, NLI)
    assert ok


def test_span_not_in_cv_fails():
    ok, why = check_claim(Claim("c", "Uses Playwright", "expert in Playwright"), CV_TEXT, NLI)
    assert not ok and "not found" in why


def test_span_present_but_not_entailing_fails():
    ok, why = check_claim(Claim("c", "Strong Appium skills", "Selenium WebDriver"), CV_TEXT, NLI)
    assert not ok and "entail" in why


def test_regeneration_loop_fixes_claim():
    bad = Claim("c", "Uses Playwright", "expert in Playwright")
    good = Claim("c", "Uses Selenium", "Selenium WebDriver")
    calls = []

    def regen(cv_id, failed, feedback, attempt):
        calls.append(attempt)
        return [good]

    final, stats = verify_claims({"c": [bad]}, {"c": CV_TEXT}, NLI, regen, max_rounds=3)
    assert [c.status for c in final["c"]] == ["supported"] and calls == [1]
    assert stats["failed_first"] == 1


def test_gives_up_after_max_rounds_and_flags_no_evidence():
    bad = Claim("c", "Uses Playwright", "expert in Playwright")
    attempts = []

    def regen(cv_id, failed, feedback, attempt):
        attempts.append(attempt)
        return [Claim("c", f"Uses Playwright v{attempt}", f"expert in Playwright {attempt}")]

    final, stats = verify_claims({"c": [bad]}, {"c": CV_TEXT}, NLI, regen, max_rounds=3)
    assert attempts == [1, 2, 3]
    assert final["c"][0].status == "no_evidence" and stats["no_evidence"] == 1


def test_remove_mode_drops_unsupported():
    final, _ = verify_claims({"c": [Claim("c", "x Playwright", "nope nope nope")]}, {"c": CV_TEXT}, NLI, None,
                             on_fail="remove")
    assert final["c"] == []
