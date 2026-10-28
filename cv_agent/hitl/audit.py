"""HR decisions (FR-29) and weight calibration from HR preferences (FR-30, Loop 5)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cv_agent.criteria.profiles import normalize_weights
from cv_agent.schema import AXES, ScoredCandidate

DECISIONS = {"approve", "reject", "override"}


@dataclass(frozen=True)
class Decision:
    run_id: str
    jd_id: str
    cv_id: str
    action: str                 # approve | reject | override
    reason: str
    reviewer: str = "hr"
    new_rank: Optional[int] = None
    ts: float = 0.0


def validate_decision(d: Decision) -> None:
    if d.action not in DECISIONS:
        raise ValueError(f"action must be one of {sorted(DECISIONS)}")
    if not d.reason.strip():
        raise ValueError("HR decision requires a reason (FR-29)")
    if d.action == "override" and (d.new_rank is None or d.new_rank < 1):
        raise ValueError("override requires new_rank >= 1")


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, decision: Decision) -> Decision:
        validate_decision(decision)
        stamped = Decision(**{**asdict(decision), "ts": decision.ts or time.time()})
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(stamped), ensure_ascii=False) + "\n")
        return stamped

    def load(self, jd_id: Optional[str] = None) -> List[Decision]:
        if not self.path.exists():
            return []
        rows = [Decision(**json.loads(line)) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [r for r in rows if jd_id is None or r.jd_id == jd_id]


def preferences_from_decisions(decisions: Sequence[Decision], ranking: Sequence[str]) -> List[Tuple[str, str]]:
    """Derive (preferred, other) pairs: approved > rejected, override moves a CV above those it passes."""
    approved = [d.cv_id for d in decisions if d.action == "approve"]
    rejected = [d.cv_id for d in decisions if d.action == "reject"]
    pairs = [(a, r) for a in approved for r in rejected]
    order = list(ranking)
    for d in decisions:
        if d.action == "override" and d.cv_id in order:
            old = order.index(d.cv_id)
            new = min(d.new_rank - 1, len(order) - 1)
            if new < old:
                pairs += [(d.cv_id, other) for other in order[new:old]]
            elif new > old:
                pairs += [(other, d.cv_id) for other in order[old + 1:new + 1]]
    return pairs


def calibrate_weights(pairs: Sequence[Tuple[str, str]], scored: Dict[str, ScoredCandidate],
                      prior: Dict[str, float], l2: float = 1.0, lr: float = 0.5, epochs: int = 500) -> Dict[str, float]:
    """Pairwise logistic regression on ΔS(c,k); returns new weights (Σw = 1), shrunk towards ``prior``."""
    x = np.array([[scored[a].axis_scores.get(k, 0) - scored[b].axis_scores.get(k, 0) for k in AXES]
                  for a, b in pairs if a in scored and b in scored], dtype=float)
    w0 = np.array([prior.get(k, 0.0) for k in AXES], dtype=float)
    if len(x) == 0:
        return normalize_weights(prior)
    w = w0.copy()
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(x @ w)))
        grad = -(x.T @ (1 - p)) / len(x) + l2 * (w - w0)
        w -= lr * grad
    return normalize_weights({k: float(max(v, 0.0)) for k, v in zip(AXES, w)})
