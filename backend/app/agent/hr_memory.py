"""HR learning across sessions (lecturer feedback 4): the weights calibrated from HR's decisions in one run become the
starting weights of the next run for the same company profile.

Within a run, `nodes._decide` already fits the weights that best explain HR's approve / reject / override decisions
(pairwise logistic regression, shrunk towards the current weights). This module only persists that result per
company profile and hands it back at ingest, so each session starts where the last one ended instead of from
`data/criteria_profiles.yaml`. Off by default (`HR_MEMORY`), because a benchmark run must not depend on what earlier
runs learned; `app.eval.hr_learning` measures it with a simulated HR.

Bounds: only explicit HR decisions are learned (never the LLM's), each session moves the weights by at most what the
shrunk regression allows, and `max_memory_drift` caps the total L1 distance from the company profile.
"""

import json
import time
from pathlib import Path

from app.config import get_settings


def path() -> Path:
    return get_settings().runtime_dir / "hr_memory.json"


def _load(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def bounded(weights: dict[str, float], base: dict[str, float], max_drift: float) -> dict[str, float]:
    """Pull `weights` back towards `base` so that Σ|w − base| ≤ max_drift; keep Σw = 1, w ≥ 0."""
    drift = sum(abs(weights.get(k, 0.0) - v) for k, v in base.items())
    if drift <= max_drift or drift == 0:
        return dict(weights)
    t = max_drift / drift
    w = {k: max(v + t * (weights.get(k, 0.0) - v), 0.0) for k, v in base.items()}
    total = sum(w.values()) or 1.0
    return {k: round(v / total, 4) for k, v in w.items()}


def prior(company: str, base: dict[str, float], p: Path | None = None) -> tuple[dict[str, float], dict | None]:
    """The starting weights for a run: what HR taught for this company profile, else `base`. → (weights, entry)."""
    entry = _load(p or path()).get(company)
    if not entry or set(entry.get("weights", {})) != set(base):
        return dict(base), None
    return bounded(entry["weights"], base, get_settings().max_memory_drift), entry


def remember(company: str, weights: dict[str, float], base: dict[str, float], pairs: int,
             p: Path | None = None) -> dict | None:
    """Store the weights HR's decisions produced in this session. A session without preference pairs teaches
    nothing and is not stored. → the stored entry."""
    if pairs <= 0:
        return None
    p = p or path()
    data = _load(p)
    old = data.get(company, {})
    entry = {"weights": bounded(weights, base, get_settings().max_memory_drift),
             "sessions": old.get("sessions", 0) + 1, "pairs": old.get("pairs", 0) + pairs, "updated": time.time()}
    data[company] = entry
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return entry


def forget(company: str | None = None, p: Path | None = None) -> None:
    """Reset what was learned: one company profile, or everything."""
    p = p or path()
    data = {} if company is None else {k: v for k, v in _load(p).items() if k != company}
    if p.exists() or data:
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False))
