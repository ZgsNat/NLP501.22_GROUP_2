"""P7: the orchestrator graph. An LLM decides the next step itself (loop L7); code limits and checks every decision.
Read this file top to bottom to see the difference from P6 (graph.py): there, Python routers decide what runs next;
here, the model picks a tool from agent/tools.py, and the routers below only enforce the harness.

    start → agent ⇄ guard → act ────────→ agent      (read/act tools; the result goes back to the model)
                          → ask_hr ─────→ agent      (a question to HR: interrupt, then back to the model)
                          → finalize ───→ hr_turn    (budget hit or repeated errors: code answers HR, not the model)
                     act (respond_to_hr) → hr_turn → agent (HR's next message) | END (HR approves)

The harness (guard): unknown tools and invalid arguments are refused; each tool's preconditions are checked (no
recommending unexplained candidates, no requirement changes HR did not ask for, knock-outs change only through
verified facts, ...); a question to HR or the final answer must be alone in its step; repeated read calls are served
from cache; steps and tokens per HR turn are capped (Settings.max_agent_steps / max_agent_tokens). Every
refusal goes back to the model as the tool's result, so it can adapt, and is counted for the evaluation."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import ValidationError

from app.agent import nodes, prompts, tools
from app.agent.nodes import emit, traced
from app.agent.state import AgentState
from app.agent.tools import TOOLS, Run, ranked
from app.config import get_settings

GOAL = "Shortlist the strongest candidates for this job description and tell me what to probe in interview."


def _text(content) -> str:
    """Message text whether the provider returns a string or content blocks."""
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _key(name: str, args: dict, state: AgentState) -> str:
    return f"{name}:{json.dumps(args, sort_keys=True)}:{state.get('loops', {}).get('version', 0)}"


def budget_reason(state: AgentState) -> str | None:
    s, loops = get_settings(), state.get("loops", {})
    if loops.get("agent", 0) - state.get("turn_start", 0) > s.max_agent_steps:
        return f"step budget reached ({s.max_agent_steps} decisions this turn)"
    if loops.get("agent_tokens", 0) - state.get("turn_tokens", 0) > s.max_agent_tokens:
        return f"token budget reached ({s.max_agent_tokens} orchestrator tokens this turn)"
    if loops.get("agent_errors", 0) >= 2:
        return "the model failed to produce a valid decision twice"
    return None


@traced("start")
async def start(state: AgentState) -> dict:
    goal = (state.get("goal") or "").strip() or GOAL
    emit("start", f"HR goal: “{goal}” → the orchestrator plans its own steps (≤ {get_settings().max_agent_steps} per turn)")
    return {"messages": [HumanMessage(f"HR goal: {goal}\nCompany profile: {state.get('company_profile', 'default')}"
                                      f"\n\nJob description:\n{state['jd_text']}")],
            "goal": goal, "turn_start": 0, "turn_tokens": 0, "seen": {}, "calls": [], "responded": False, "halt": ""}


@traced("agent")
async def agent(state: AgentState) -> dict:
    model = nodes.llm(state, get_settings().agent_role).bind_tools(tools.schemas())
    try:
        msg = await model.ainvoke([SystemMessage(prompts.AGENT), *state["messages"]])
    except Exception as e:  # a provider hiccup or unparseable output costs a step, never the run
        emit("agent", f"Model output unusable ({type(e).__name__}: {str(e)[:120]}) → retrying", "warn")
        return {"loops": {"agent": 1, "agent_errors": 1}}
    text = _text(msg.content).strip()
    if not msg.tool_calls:  # plain text is an answer to HR: it goes through the same checks as respond_to_hr
        n = state.get("loops", {}).get("agent", 0)
        msg = AIMessage(content=msg.content, usage_metadata=msg.usage_metadata, id=msg.id,
                        tool_calls=[{"name": "respond_to_hr", "args": {"message": text, "shortlist": []},
                                     "id": f"auto-{n}", "type": "tool_call"}])
    used = (msg.usage_metadata or {}).get("total_tokens", 0)
    calls = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:80]})" for c in msg.tool_calls)
    emit("agent", (f"{text[:200]} → " if text and msg.tool_calls[0]["name"] != "respond_to_hr" else "") + calls,
         "loop", tokens=used)
    return {"messages": [msg], "loops": {"agent": 1, "agent_tokens": used}}


@traced("guard")
async def guard(state: AgentState) -> dict:
    last = state["messages"][-1] if state.get("messages") else None
    calls = list(getattr(last, "tool_calls", None) or []) if isinstance(last, AIMessage) else []
    refuse = lambda c, why: ToolMessage(f"REJECTED by harness: {why}", tool_call_id=c["id"], name=c["name"],  # noqa: E731
                                        status="error")
    final = calls and all(c["name"] == "respond_to_hr" for c in calls)  # answering costs nothing more: still checked
    if (over := budget_reason(state)) and not final:
        emit("guard", f"{over} → code answers HR with the current ranking", "warn")
        return {"messages": [refuse(c, over) for c in calls], "calls": [], "halt": over, "loops": {"budget_stops": 1}}
    if not calls:  # the model call failed: nothing to check, ask again
        return {"calls": []}
    out, approved, rejected, hits = [], [], 0, 0
    for c in calls:
        t, why, args = TOOLS.get(c["name"]), None, None
        if t is None:
            why = f"unknown tool '{c['name']}'; available: {sorted(TOOLS)}"
        else:
            try:
                args = t.args.model_validate(c["args"] or {})
            except ValidationError as e:
                why = "invalid arguments: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:3])
        if not why and t.kind in ("hr", "final") and len(calls) > 1:
            why = f"{c['name']} must be the only call in its step"
        why = why or t.check(state, args)
        if why:
            rejected += 1
            emit("guard", f"Refused {c['name']}: {why}", "warn")
            out.append(refuse(c, why))
        elif t.kind == "read" and (seen := state.get("seen", {}).get(_key(c["name"], args.model_dump(), state))):
            hits += 1
            emit("guard", f"{c['name']} repeated with nothing changed → answered from cache (0 work)")
            out.append(ToolMessage(seen + "\n(unchanged since your identical earlier call)", tool_call_id=c["id"],
                                   name=c["name"]))
        else:
            approved.append({"id": c["id"], "name": c["name"], "args": args.model_dump()})
    return {"messages": out, "calls": approved, "loops": {"guard_rejections": rejected, "cache_hits": hits}}


def _result(c: dict, obs: dict) -> ToolMessage:
    return ToolMessage(json.dumps(obs, default=str, ensure_ascii=False, separators=(",", ":")), tool_call_id=c["id"],
                       name=c["name"])


@traced("act")
async def act(state: AgentState) -> dict:
    r, msgs, seen, responded = Run(state), [], {}, False
    write = get_stream_writer()
    for c in state["calls"]:
        t = TOOLS[c["name"]]
        try:
            obs, update = await t.run(r.state, t.args.model_validate(c["args"]))
        except GraphBubbleUp:
            raise
        except Exception as e:  # a failing tool is an observation, not a crash: the model can choose another way
            obs, update = {"error": f"{type(e).__name__}: {str(e)[:240]}"}, {}
            emit("act", f"{c['name']} failed: {obs['error']}", "warn")
        r.add(update)
        msgs.append(_result(c, obs))
        if t.kind == "read":
            seen[_key(c["name"], c["args"], r.state)] = msgs[-1].content
        elif t.kind == "act":
            r.add({"loops": {"version": 1}})
        responded |= t.kind == "final"
        write({"type": "tool", "name": c["name"], "args": c["args"], "result": msgs[-1].content[:300]})
    r.add({"seen": {**state.get("seen", {}), **seen}, "calls": [], "responded": responded})
    return {**r.update, "messages": msgs}


@traced("ask_hr")
async def ask_hr(state: AgentState) -> dict:
    # Re-runs from the top on resume: the tool does nothing costly before its interrupt().
    c = state["calls"][0]
    t = TOOLS[c["name"]]
    obs, update = await t.run(state, t.args.model_validate(c["args"]))
    r = Run(state)
    r.add(update)
    r.add({"calls": [], "loops": {"version": 1}})
    return {**r.update, "messages": [_result(c, obs)]}


@traced("finalize")
async def finalize(state: AgentState) -> dict:
    """The harness's fallback answer: the current rule-based ranking, labelled as not verified by the agent."""
    top = ranked(state)[:5]
    msg = (f"I stopped before finishing: {state['halt']}. "
           + (f"The current rule-based ranking is {', '.join(f'{r['cv_id']} ({r['total']:.2f})' for r in top)}; "
              "these candidates are not yet explained or checked by me." if top else "No candidate is ranked yet."))
    emit("finalize", msg, "warn")
    report = {"summary": msg, "candidates": [], "top_ids": sorted(r["cv_id"] for r in top), "checks": {}}
    return {"messages": [AIMessage(msg)], "reply": msg, "shortlist": [], "report": report, "halt": "",
            "responded": True}


@traced("hr_turn")
async def hr_turn(state: AgentState) -> dict:
    s = get_settings()
    left = s.max_hr_turns - state.get("loops", {}).get("hr_turn", 0)
    rows = {r["cv_id"]: r for r in state.get("ranking", [])}
    answer = interrupt({"kind": "chat", "payload": {"message": state.get("reply", ""), "turns_left": left,
                                                    "shortlist": [rows[c] for c in state.get("shortlist", []) if c in rows]}})
    loops = state.get("loops", {})
    base = {"responded": False, "turn_start": loops.get("agent", 0), "turn_tokens": loops.get("agent_tokens", 0),
            "loops": {"hr_turn": 1}}
    text = (answer.get("message") or "").strip()
    if decisions := [d for d in answer.get("decisions", []) if (d.get("reason") or "").strip()]:
        # Deterministic: HR's decisions re-fit the weights (pairwise logistic regression) and re-screen the pool.
        r = Run(state)
        r.add({"weights": nodes._decide(state, decisions)["weights"]})
        view = await tools.screen_into(r)
        r.add({"loops": {"version": 1}})
        text = (f"{text}\n\n" if text else "") + f"HR recorded decisions {json.dumps(decisions)}. The weights were " \
               f"re-fitted to {r.state['weights']} and the pool re-screened: {json.dumps(view, default=str)}"
        return {**r.update, **base, "loops": {**r.update.get("loops", {}), "hr_turn": 1}, "hr_action": "chat",
                "messages": [HumanMessage(text)]}
    if answer.get("action") == "approve" or not text or left <= 0:
        emit("hr_turn", "HR approved the shortlist" if answer.get("action") == "approve" or not text
             else f"Conversation budget used ({s.max_hr_turns} HR turns) → final", "info")
        return {**base, "hr_action": "approve"}
    emit("hr_turn", f"HR: “{text[:160]}”", "loop")
    return {**base, "hr_action": "chat", "messages": [HumanMessage(text)]}


def after_guard(state: AgentState) -> str:
    if state.get("halt"):
        return "finalize"
    calls = state.get("calls") or []
    if not calls:
        return "agent"
    return "ask_hr" if TOOLS[calls[0]["name"]].kind == "hr" else "act"


def after_act(state: AgentState) -> str:
    return "hr_turn" if state.get("responded") else "agent"


def after_hr(state: AgentState) -> str:
    return END if state.get("hr_action") == "approve" else "agent"


def build_agent_graph(checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(AgentState)
    for fn in (start, agent, guard, act, ask_hr, finalize, hr_turn):
        g.add_node(fn.__name__, fn)
    g.add_edge(START, "start")
    g.add_edge("start", "agent")
    g.add_edge("agent", "guard")
    g.add_conditional_edges("guard", after_guard, ["act", "ask_hr", "finalize", "agent"])
    g.add_conditional_edges("act", after_act, ["hr_turn", "agent"])
    g.add_edge("ask_hr", "agent")
    g.add_edge("finalize", "hr_turn")
    g.add_conditional_edges("hr_turn", after_hr, ["agent", END])
    return g.compile(checkpointer=checkpointer)
