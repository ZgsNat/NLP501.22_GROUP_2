"""Run the agent end-to-end in the terminal, auto-answering HR interrupts.

    cd backend && uv run python -m scripts.smoke JD-002            # default models from .env
    cd backend && uv run python -m scripts.smoke JD-003 --reasoning anthropic:claude-sonnet-5-5
    cd backend && uv run python -m scripts.smoke JD-002 --agent "Shortlist 5; ISTQB is a plus"   # P7 orchestrator
"""

import argparse
import asyncio
import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app import observability
from app.agent.graph import build_graph
from app.agent.orchestrator import build_agent_graph
from app.rag.corpus import get_doc


async def main(jd_id: str, models: dict[str, str], goal: str | None = None) -> None:
    jd = get_doc("jds", jd_id)
    graph = build_agent_graph(InMemorySaver()) if goal is not None else build_graph(InMemorySaver())
    run_id = f"smoke-{jd_id}-{int(time.time())}"
    usage = observability.UsageCounter()
    config = {"configurable": {"thread_id": run_id}, "callbacks": [*observability.callbacks(), usage],
              "metadata": {"langfuse_session_id": run_id}, "run_name": "cv-agent"}
    inp = {"jd_text": jd.text, "company_profile": jd.meta["company_profile"], "models": models}
    if goal is not None:
        inp["goal"] = goal
    t0 = time.perf_counter()
    while True:
        async for mode, chunk in graph.astream(inp, config, stream_mode=["updates", "custom"]):
            if mode == "custom" and chunk["type"] == "log":
                print(f"{'↺' if chunk['level'] == 'loop' else '·'} [{chunk['node']}] {chunk['message']}")
            elif mode == "custom" and chunk["type"] == "tool":
                print(f"  🔧 {chunk['name']}({chunk['args']})")
            elif mode == "custom" and chunk["type"] == "node_end":
                print(f"  ✓ {chunk['node']} {chunk['duration_ms']} ms")
        state = await graph.aget_state(config)
        if not state.next:
            break
        pending = state.tasks[0].interrupts[0].value
        if pending["kind"] == "clarification":
            answers = {q: "Please use your best judgement: 3+ years, Selenium or Playwright, English required."
                       for q in pending["payload"]["questions"]}
            print(f"\n👤 HR answers {len(answers)} question(s)\n")
            inp = Command(resume={"answers": answers})
        elif pending["kind"] == "chat":
            print(f"\n🤖 {pending['payload']['message']}\n   shortlist: {[r['cv_id'] for r in pending['payload']['shortlist']]}"
                  "\n\n👤 HR approves\n")
            inp = Command(resume={"action": "approve"})
        elif pending["kind"] == "substitutes":
            accept = {p["must_have"]: p["related"] for p in pending["payload"]["proposals"]}
            print(f"\n👤 HR accepts substitutes {accept}\n")
            inp = Command(resume={"accept": accept})
        else:
            print("\n👤 HR approves\n")
            inp = Command(resume={"action": "approve"})

    v = state.values
    u = usage.summary()
    print(f"\nDone in {time.perf_counter() - t0:.1f}s · loops={v.get('loops')}")
    print(f"LLM: {u['calls']} calls, {u['total_tokens']} tokens · " + ", ".join(f"{k}={n['input_tokens'] + n['output_tokens']}" for k, n in u["by_node"].items()))
    for r in v["ranking"]:
        if r["status"] == "ranked":
            print(f"{r['rank']:>2}. {r['cv_id']} {r['total']:.3f} {r['tier']:<8} {r['arbiter_note'] or ''}")
    for status in ("not_shortlisted", "needs_review", "knocked_out"):
        rows = [r for r in v["ranking"] if r["status"] == status]
        print(f"\n{status} ({len(rows)}):" + "".join(f"\n   {r['cv_id']}: {r['reason']}" for r in rows[:4])
              + ("\n   …" if len(rows) > 4 else ""))
    print("\n" + v["report"]["summary"])
    if url := observability.session_url(run_id):
        print(f"\nTrace: {url}")
    observability.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("jd_id")
    p.add_argument("--reasoning")
    p.add_argument("--fast")
    p.add_argument("--agent", nargs="?", const="", metavar="GOAL", help="run the P7 orchestrator with HR's goal")
    a = p.parse_args()
    asyncio.run(main(a.jd_id, {k: v for k, v in {"reasoning": a.reasoning, "fast": a.fast}.items() if v}, a.agent))
