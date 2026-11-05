"""Controlled LLM ranking (v4): the judgement of an agent on its own, inside the system's controls.

S1 (eval/single_agent.py) showed that LLM judgement carries ranking signal, and also what it does unchecked: ranks CVs
it never read, reads raw text, gives no reasons for what it leaves out. Here the same judgement runs under controls:
  - only over the screened shortlist (rules and knock-out already applied), and the model sees every candidate's
    sanitised, sectioned passages: nothing can be ranked unread;
  - judged twice, in opposite input orders; positions are averaged, so a position-biased answer cannot decide alone;
  - ids are canonicalized at the boundary: only shortlisted ids, each once; a missing one keeps its rule position;
  - the rule scores S and the knock-out do not change; only the order inside the shortlist does, with a reason each.
Adopted only if app.eval.llm_rank says so on dev (docs/LLM_RANK_RESULTS.md)."""

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent import prompts
from app.nlp.pipeline import CVProfile


class RankedCV(BaseModel):
    cv_id: str
    reason: str = Field(description="One sentence: the evidence in this CV that decides its place")


class LLMRanking(BaseModel):
    ranking: list[RankedCV] = Field(description="Every candidate shown, exactly once, best first")


def canonical(answer: list[str], given: list[str]) -> list[str]:
    """Only the given ids, each once, in the model's order; ids it left out keep their input order at the end."""
    seen = list(dict.fromkeys(c for c in answer if c in given))
    return seen + [c for c in given if c not in seen]


def combine(passes: list[list[str]], fallback: list[str]) -> list[str]:
    """Mean position over the passes (Borda); ties keep the rule order."""
    pos = {c: sum(p.index(c) for p in passes) / len(passes) for c in fallback}
    return sorted(fallback, key=lambda c: (pos[c], fallback.index(c)))


def overlap(a: list[str], b: list[str], k: int) -> float:
    return len(set(a[:k]) & set(b[:k])) / max(min(k, len(a)), 1)


async def rank(model: BaseChatModel, jd_text: str, requirements: dict, order: list[str],
               profiles: dict[str, CVProfile], chars: int = 2500, config: dict | None = None,
               lang: str | None = None) -> dict:
    """→ {order, reasons, agreement@5}. `order` is the rule order of the shortlist; the answer is a permutation of it."""
    async def one(ids: list[str]) -> tuple[list[str], dict[str, str]]:
        cvs = "\n\n".join(f"=== {c} ===\n{profiles[c].facts_for_llm()}\n{profiles[c].for_llm()[:chars]}" for c in ids)
        out: LLMRanking = await model.with_structured_output(LLMRanking).ainvoke([
            SystemMessage(prompts.localized(prompts.LLM_RANK, lang)),  # lang: the reasons are shown to HR
            HumanMessage(f"Job description:\n{jd_text}\n\nParsed requirements: {json.dumps(requirements)}\n\n"
                         f"Candidates ({len(ids)}):\n\n{cvs}")], config=config)
        return canonical([r.cv_id.strip() for r in out.ranking], order), {r.cv_id.strip(): r.reason for r in out.ranking}

    forward, reasons = await one(order)
    backward, reasons_b = await one(order[::-1])  # the same candidates, shown in the opposite order
    final = combine([forward, backward], order)
    return {"order": final, "reasons": {c: reasons.get(c) or reasons_b.get(c, "") for c in final},
            "agreement@5": overlap(forward, backward, 5), "passes": [forward, backward]}
