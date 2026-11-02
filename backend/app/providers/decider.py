"""System-One-style decisions (inspired by TypeSafe's Jev): a fast model returns a *typed* choice from a closed
set plus a confidence — never free text that needs parsing. Use it for gates and routing ("smart if-statements"),
and keep the slow reasoning model for judgement. Swap the backend here (e.g. a Jev client) without touching nodes.

Backends (Settings.decider): "llm" asks the fast chat model for structured output; "systemone" asks a real decision
model through TypeSafe's System One API, which Jev serves with a key and Ollama >= 0.35 serves locally (nimble)."""

import asyncio
from typing import Literal

import httpx

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model

from app.config import get_settings


class Decision(BaseModel):
    choice: str
    confidence: float
    reason: str


def system_one(state: str, questions: dict[str, dict]) -> dict[str, dict]:
    """POST /v1/systemone: typed questions about one state → {question id: answer}. A noul answer carries P(true) as
    "noul"; a choice answer carries "choice" and/or "probabilities" per option."""
    s = get_settings()
    headers = {"Authorization": f"Bearer {s.system_one_api_key}"} if s.system_one_api_key else {}
    r = httpx.post(f"{s.system_one_base_url.rstrip('/')}/v1/systemone", headers=headers, timeout=120,
                   json={"model": s.system_one_model, "state": state, "questions": questions})
    r.raise_for_status()
    return r.json()["answers"]


async def _decide_system_one(question: str, options: list[str], context: str) -> Decision:
    a = (await asyncio.to_thread(system_one, context, {
        "q": {"type": "choice", "instructions": question, "criteria": {o: o.replace("_", " ") for o in options}}}))["q"]
    probs = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
    choice = a.get("choice") or max(probs, key=probs.get)
    if choice not in options:  # canonicalize at the boundary, as for any model output
        raise ValueError(f"System One answered {choice!r}, not one of {options}")
    # P(choice) when the server gives probabilities: Ollama's "confidence" measures how peaked the distribution is,
    # not the probability that the choice is right (nimble: choice 0.71, confidence 0.13 on the same answer)
    conf = probs[choice] if choice in probs else float(a.get("confidence") or 0.0)
    return Decision(choice=choice, confidence=conf, reason=f"System One ({get_settings().system_one_model}) p={conf:.2f}")


async def decide(llm: BaseChatModel, question: str, options: list[str], context: str) -> Decision:
    """options[0] is the safe default: returned (confidence 0) if the model output is unusable — a gate never crashes a run."""
    if get_settings().decider == "systemone":
        try:
            return await _decide_system_one(question, options, context)
        except Exception as e:
            return Decision(choice=options[0], confidence=0.0, reason=f"fallback to default ({type(e).__name__})")
    schema = create_model(
        "Decision",
        choice=(Literal[tuple(options)], Field(description=f"Exactly one of {options}")),
        confidence=(float, Field(description="Probability the choice is right, between 0 and 1")),
        reason=(str, Field(description="One short sentence")),
    )
    try:
        out = await llm.with_structured_output(schema).ainvoke(
            [
                SystemMessage("You are a precise decision function. Answer only with the requested structure."),
                HumanMessage(f"{context}\n\nDecision: {question}\nOptions: {options}"),
            ]
        )
    except Exception as e:
        return Decision(choice=options[0], confidence=0.0, reason=f"fallback to default ({type(e).__name__})")
    conf = out.confidence / 100 if out.confidence > 1 else out.confidence  # small models answer "85" for 85%
    return Decision(choice=out.choice, confidence=max(0.0, min(conf, 1.0)), reason=out.reason)
