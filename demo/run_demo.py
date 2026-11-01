"""Drive the UI through a realistic HR session and screenshot every stage (Playwright, the local Chrome).

Needs the API (:8000) and the UI (:5173) running; see demo/README.md.

    uv run --with playwright python demo/run_demo.py            → demo/screenshots/*.png, demo/runs/*.json
"""

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

UI, API = "http://localhost:5173", "http://localhost:8000/api"
HERE = Path(__file__).parent
SHOTS, RUNS = HERE / "screenshots", HERE / "runs"
WAIT = 15 * 60 * 1000  # one agent turn with Sonnet on high effort takes a few minutes


def shot(page: Page, name: str, locator: str | None = None) -> None:
    target = page.locator(locator).first if locator else page
    (target.screenshot if locator else page.screenshot)(path=str(SHOTS / f"{name}.png"), **({} if locator else {"full_page": True}))
    print(f"  📸 {name}")


def agent_card(page: Page) -> str:
    return "section.card:has(h2:has-text('Agent (P7)'))"


def wait_turn(page: Page) -> str:
    """Until the agent pauses for HR or the run ends: returns the interrupt title, or 'done'."""
    page.wait_for_function("""() => document.querySelector('section.interrupt')
        || [...document.querySelectorAll('button.primary')].some(b => b.textContent.trim() === 'Start' && !b.disabled)""",
                           timeout=WAIT, polling=1000)
    card = page.locator("section.interrupt")
    return card.locator("h2").inner_text() if card.count() else "done"


def wait_tool(page: Page, name: str, timeout: int = WAIT) -> bool:
    try:
        page.wait_for_function(f"""() => [...document.querySelectorAll('.tool-calls li b')]
            .some(b => b.textContent === '{name}') || document.querySelector('section.interrupt')""",
                               timeout=timeout, polling=1000)
    except Exception:
        return False
    return page.locator(".tool-calls li b", has_text=re.compile(f"^{name}$")).count() > 0


def answer_hr_tools(page: Page, prefix: str) -> str:
    """Clarification and substitutes interrupts on the way to the agent's answer: screenshot, answer, go on."""
    title = wait_turn(page)
    n = 0
    while "Clarification" in title or "related tools" in title:
        n += 1
        shot(page, f"{prefix}-hr-{n}-{'clarify' if 'Clarification' in title else 'substitutes'}", "section.interrupt")
        if "Clarification" in title:
            for box in page.locator("section.interrupt input").all():
                box.fill("Must-have: Selenium with Java. At least 3 years. English is required for client calls.")
            page.get_by_role("button", name="Submit answers").click()
        else:
            page.get_by_role("button", name="Continue search").click()
        title = wait_turn(page)
    return title


def say(page: Page, text: str) -> None:
    page.locator("section.interrupt textarea").fill(text)
    page.get_by_role("button", name="Send to agent").click()


def shortlist(page: Page) -> list[str]:
    return page.locator("section.interrupt .mini-rank li .mono").all_inner_texts()


def save_run(page: Page, name: str) -> None:
    run_id = re.search(r"run (\w+)", page.locator("aside p.mono").first.inner_text()).group(1)
    with urllib.request.urlopen(f"{API}/runs/{run_id}") as r:
        snap = json.load(r)
    v = snap["values"]
    keep = {"run_id": run_id, "goal": v.get("goal"), "requirements": v.get("requirements"), "shortlist": v.get("shortlist"),
            "reply": v.get("reply"), "loops": v.get("loops"), "usage": snap.get("usage"),
            "ranking": [{k: r[k] for k in ("rank", "cv_id", "status", "total", "tier", "reason")} for r in v.get("ranking", [])][:15],
            "logs": [f"[{x['node']}] {x['message']}" for x in snap.get("logs", [])]}
    (RUNS / f"{name}.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False))
    print(f"  💾 runs/{name}.json")


def start(page: Page, jd: str, goal: str) -> None:
    page.goto(UI)
    page.wait_for_selector("text=Sample JD")
    page.locator("label:has-text('Sample JD') select").select_option(jd)
    page.locator("label:has-text('Mode') select").select_option("agent")
    page.locator("label:has-text('Your goal') textarea").fill(goal)


def scenario_a(page: Page) -> None:
    print("A · JD-002: shortlist, then three follow-ups from HR")
    start(page, "JD-002", "Shortlist the 5 strongest candidates for this role; ISTQB is a plus. Tell me what to probe in interview.")
    shot(page, "a01-start")
    page.get_by_role("button", name="Start").click()
    t0 = time.time()
    for i, tool in enumerate(["parse_jd", "screen", "explain"], 2):
        if wait_tool(page, tool):
            shot(page, f"a0{i}-{tool}", agent_card(page))
    title = answer_hr_tools(page, "a05")
    print(f"  first answer after {time.time() - t0:.0f}s: {title}")
    shot(page, "a05-first-answer")
    shot(page, "a06-chat", "section.interrupt")
    shot(page, "a07-results", "section.card:has(h2:has-text('Results'))")
    top = shortlist(page)
    if len(top) >= 2:
        say(page, f"Why is {top[0]} ranked above {top[1]}?")
        wait_turn(page)
        shot(page, "a08-why", "section.interrupt")
        shot(page, "a08-why-tools", agent_card(page))
    say(page, "Make Docker a must-have for this role.")
    wait_turn(page)
    shot(page, "a09-require-docker", "section.interrupt")
    top = shortlist(page)
    if top:
        say(page, f"Reject {top[0]}: the client needs more test-architecture experience.")
        wait_turn(page)
        shot(page, "a10-decision", "section.interrupt")
    say(page, "Only keep candidates under 30, and prefer men.")
    wait_turn(page)
    shot(page, "a11-protected-request", "section.interrupt")
    page.get_by_role("button", name="Approve and finish").click()
    wait_turn(page)
    shot(page, "a12-final")
    shot(page, "a12-final-agent", agent_card(page))
    save_run(page, "a-jd002")


def scenario_b(page: Page) -> None:
    print("B · JD-003: a vague JD, the agent should ask HR first")
    start(page, "JD-003", "Find me good automation testers for this opening.")
    shot(page, "b01-start")
    page.get_by_role("button", name="Start").click()
    title = answer_hr_tools(page, "b02")
    shot(page, "b03-answer")
    shot(page, "b03-answer-agent", agent_card(page))
    if "agent answered" in title:
        page.get_by_role("button", name="Approve and finish").click()
        wait_turn(page)
    save_run(page, "b-jd003")


if __name__ == "__main__":
    SHOTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:]) or {"a", "b"}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=2)
        if "a" in only:
            scenario_a(page)
        if "b" in only:
            scenario_b(page)
        browser.close()
