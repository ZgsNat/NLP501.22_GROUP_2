"""Selective Arbiter (FR-18, FR-19) and the LLM-all listwise baseline (FR-20).

Only adjacent near-tie pairs (|ΔScore| < threshold) go to the LLM; each pair is
judged ``passes`` times alternating A/B order to cancel position bias, and the
majority decides whether the pair is swapped.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from cv_agent.llm import prompts
from cv_agent.llm.backends import LLMError
from cv_agent.llm.context import cv_digest, jd_digest
from cv_agent.llm.gateway import BudgetExceeded
from cv_agent.llm.gateway import LLMGateway
from cv_agent.schema import CV, JD, CriteriaProfile, ScoredCandidate

log = logging.getLogger(__name__)
DEFAULT_CV_CHARS = 700


@dataclass
class ArbiterResult:
    ranking: List[ScoredCandidate]
    pairs_checked: List[Tuple[str, str]] = field(default_factory=list)
    swaps: List[Tuple[str, str]] = field(default_factory=list)
    call_ids: List[str] = field(default_factory=list)
    consistent_pairs: int = 0          # pairs where both A/B orders agreed (position-bias diagnostic)
    interrupted: Optional[str] = None  # set when budget / LLM failure stopped the arbiter midway


def criteria_payload(jd: JD, profile: Optional[CriteriaProfile]) -> Dict:
    if profile is None:  # A2: generic prompt
        return {}
    # must-haves are already listed in the JD digest; only the profile weights are added
    return {k: round(v, 2) for k, v in profile.weights.items()}


def find_near_ties(ranking: List[ScoredCandidate], threshold: float) -> List[int]:
    """Indices i such that (i, i+1) is a near-tie pair (FR-18)."""
    return [i for i in range(len(ranking) - 1) if abs(ranking[i].score - ranking[i + 1].score) < threshold]


def _judge_pair(gw: LLMGateway, jd: JD, profile: Optional[CriteriaProfile], a: ScoredCandidate, b: ScoredCandidate,
                cvs: Dict[str, CV], passes: int, cv_chars: int = DEFAULT_CV_CHARS) -> Tuple[int, List[str], int]:
    """Return (+votes for a minus votes for b, call_ids, valid votes)."""
    crit = criteria_payload(jd, profile)
    margin, call_ids, valid = 0, [], 0
    jd_text = jd_digest(jd)
    for p in range(passes):
        first, second = (a, b) if p % 2 == 0 else (b, a)
        user = prompts.pairwise(jd_text, crit, cv_digest(cvs[first.cv_id], cv_chars),
                                cv_digest(cvs[second.cv_id], cv_chars))
        ctx = {"a": {"score": first.score, "spans": [s for v in first.evidence.values() for s in v][:1]},
               "b": {"score": second.score, "spans": [s for v in second.evidence.values() for s in v][:1]}}
        data, cid = gw.call("pairwise", prompts.PROMPT_VERSION, prompts.SYSTEM, user, ctx,
                            meta={"first": first.cv_id, "second": second.cv_id})
        call_ids.append(cid)
        winner = str(data.get("winner", "")).strip().upper()[:1]
        if winner not in {"A", "B"}:
            log.warning("Arbiter got invalid winner %r; counting as abstain", data.get("winner"))
            continue
        winner_is_a = (winner == "A") == (first is a)
        margin += 1 if winner_is_a else -1
        valid += 1
    return margin, call_ids, valid


def selective_arbiter(gw: LLMGateway, jd: JD, profile: Optional[CriteriaProfile], ranking: List[ScoredCandidate],
                      cvs: Dict[str, CV], threshold: float = 0.05, passes: int = 2, max_rank: Optional[int] = None,
                      cv_chars: int = DEFAULT_CV_CHARS) -> ArbiterResult:
    """``max_rank``: only pairs (i, i+1) with i < max_rank are judged — the positions that change the
    shortlist / nDCG@10 (pair (top_n-1, top_n) decides who enters the shortlist)."""
    passes = passes + passes % 2  # even: every pair is seen in both orders, cancelling position bias
    order = list(ranking)
    result = ArbiterResult(ranking=order)
    limit = len(order) - 1 if max_rank is None else min(len(order) - 1, max_rank)
    i = 0
    while i < limit:
        a, b = order[i], order[i + 1]
        # never let the LLM override the must-have filter, and skip knocked-out pairs entirely
        if abs(a.score - b.score) >= threshold or not (a.must_have_ok and b.must_have_ok):
            i += 1
            continue
        try:
            margin, cids, valid = _judge_pair(gw, jd, profile, a, b, cvs, passes, cv_chars)
        except (BudgetExceeded, LLMError) as exc:  # keep the decisions made so far (FR-26)
            result.interrupted = f"{type(exc).__name__}: {exc}"
            log.warning("Arbiter interrupted after %d pairs: %s", len(result.pairs_checked), exc)
            break
        result.pairs_checked.append((a.cv_id, b.cv_id))
        result.call_ids.extend(cids)
        result.consistent_pairs += int(valid == passes and abs(margin) == passes)
        if margin < 0:  # majority prefers b
            note = f"Arbiter: xếp trên {a.cv_id} (Δ={a.score - b.score:+.3f}, thắng {-margin} phiếu chênh / {passes} lượt)"
            order[i], order[i + 1] = replace(b, arbiter_notes=b.arbiter_notes + [note]), a
            result.swaps.append((b.cv_id, a.cv_id))
            i += 2  # a swapped candidate moves at most one place (no drift beyond the near-tie)
        else:
            order[i] = replace(a, arbiter_notes=a.arbiter_notes + [f"Arbiter giữ thứ tự trên {b.cv_id}"])
            i += 1
    result.ranking = order
    return result


def llm_rerank_all(gw: LLMGateway, jd: JD, profile: Optional[CriteriaProfile], ranking: List[ScoredCandidate],
                   cvs: Dict[str, CV], chars_per_cv: int = 400) -> ArbiterResult:
    """LLM-all baseline: one listwise call over the whole top-N (FR-20)."""
    cands = {c.cv_id: cv_digest(cvs[c.cv_id], chars_per_cv) for c in ranking}
    user = prompts.listwise(jd_digest(jd), criteria_payload(jd, profile), cands)
    data, cid = gw.call("listwise", prompts.PROMPT_VERSION, prompts.SYSTEM, user,
                        {"scores": {c.cv_id: c.score for c in ranking}})
    by_id = {c.cv_id: c for c in ranking}
    llm_order = [cid_ for cid_ in data.get("ranking", []) if cid_ in by_id]
    seen = set(llm_order)
    llm_order += [c.cv_id for c in ranking if c.cv_id not in seen]  # LLM omissions keep original order at the end
    return ArbiterResult(ranking=[by_id[x] for x in dict.fromkeys(llm_order)], call_ids=[cid])

