"""verify_evidence (FR-21..23, Loop 4).

1. span check: the quoted span must occur in the masked CV (exact or fuzzy)
2. NLI: the span (+ small context window) must entail the claim
3. failing claims are sent back for regeneration (max ``max_rounds``); still
   failing afterwards => removed or flagged "No evidence".
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

from cv_agent.schema import Claim
from cv_agent.verify.nli import NLIModel

log = logging.getLogger(__name__)
CONTEXT_WINDOW = 80


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def locate_span(span: str, text: str, fuzzy_threshold: float = 0.85) -> Optional[Tuple[int, int]]:
    """Return (start, end) in normalised text if span is present (exact or fuzzy), else None."""
    s, t = _norm(span), _norm(text)
    if not s:
        return None
    idx = t.find(s)
    if idx >= 0:
        return idx, idx + len(s)
    if len(s) < 8:  # too short for meaningful fuzzy matching
        return None
    matcher = difflib.SequenceMatcher(None, t, s, autojunk=False)
    best = (0.0, None)
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        start = max(0, block.a - block.b)
        window = t[start:start + len(s)]
        ratio = difflib.SequenceMatcher(None, window, s).ratio()
        if ratio > best[0]:
            best = (ratio, (start, start + len(s)))
    return best[1] if best[0] >= fuzzy_threshold else None


def check_claim(claim: Claim, cv_text: str, nli: NLIModel, fuzzy_threshold: float = 0.85,
                nli_threshold: float = 0.5) -> Tuple[bool, str]:
    loc = locate_span(claim.span_text, cv_text, fuzzy_threshold)
    if loc is None:
        return False, f"span not found in CV: {claim.span_text!r}"
    t = _norm(cv_text)
    premise = t[max(0, loc[0] - CONTEXT_WINDOW): loc[1] + CONTEXT_WINDOW]
    p = nli.entails(premise, claim.text)
    if p < nli_threshold:
        return False, f"span does not entail claim (p={p:.2f}): {claim.text!r}"
    return True, "ok"


Regenerator = Callable[[str, List[Claim], str, int], List[Claim]]


def verify_claims(claims_by_cv: Dict[str, List[Claim]], cv_texts: Dict[str, str], nli: NLIModel,
                  regenerate: Optional[Regenerator], max_rounds: int = 3, on_fail: str = "flag",
                  fuzzy_threshold: float = 0.85) -> Tuple[Dict[str, List[Claim]], Dict[str, int]]:
    """Return (final claims per cv, stats). ``on_fail``: 'flag' keeps as no_evidence, 'remove' drops."""
    stats = {"checked": 0, "failed_first": 0, "regenerations": 0, "no_evidence": 0, "supported": 0}
    final: Dict[str, List[Claim]] = {}
    for cv_id, claims in claims_by_cv.items():
        text = cv_texts[cv_id]
        accepted: List[Claim] = []
        pending = list(claims)
        for round_no in range(max_rounds + 1):
            failed: List[Tuple[Claim, str]] = []
            for c in pending:
                stats["checked"] += 1
                ok, why = check_claim(c, text, nli, fuzzy_threshold)
                if ok:
                    accepted.append(replace(c, verified=True, status="supported", attempts=round_no))
                else:
                    failed.append((c, why))
            if round_no == 0:
                stats["failed_first"] += len(failed)
            if not failed:
                break
            if round_no == max_rounds or regenerate is None:
                for c, _ in failed:
                    stats["no_evidence"] += 1
                    if on_fail == "flag":
                        accepted.append(replace(c, verified=False, status="no_evidence", attempts=round_no))
                break
            stats["regenerations"] += 1
            feedback = "; ".join(why for _, why in failed)[:800]
            try:
                regenerated = regenerate(cv_id, [c for c, _ in failed], feedback, round_no + 1)
            except Exception as exc:  # budget exceeded / LLM down -> stop looping, treat as unresolved
                log.warning("Regeneration for %s stopped: %s", cv_id, exc)
                for c, _ in failed:
                    stats["no_evidence"] += 1
                    if on_fail == "flag":
                        accepted.append(replace(c, verified=False, status="no_evidence", attempts=round_no))
                break
            known = {(_norm(a.text), _norm(a.span_text)) for a in accepted}
            pending = [c for c in regenerated if (_norm(c.text), _norm(c.span_text)) not in known]
            if not pending:
                # nothing new: the LLM withdrew the failed claims, so they are removed
                stats["removed"] = stats.get("removed", 0) + len(failed)
                break
        stats["supported"] += sum(1 for c in accepted if c.status == "supported")
        final[cv_id] = accepted
    return final, stats
