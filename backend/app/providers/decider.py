"""System-One-style decisions (inspired by TypeSafe's Jev): a fast model returns a *typed* choice from a closed
set plus a confidence — never free text that needs parsing. Use it for gates and routing ("smart if-statements"),
and keep the slow reasoning model for judgement. Swap the backend here (e.g. a Jev client) without touching nodes."""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, create_model


class Decision(BaseModel):
    choice: str
    confidence: float
    reason: str


async def decide(llm: BaseChatModel, question: str, options: list[str], context: str) -> Decision:
    """options[0] is the safe default: returned (confidence 0) if the model output is unusable — a gate never crashes a run."""
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
