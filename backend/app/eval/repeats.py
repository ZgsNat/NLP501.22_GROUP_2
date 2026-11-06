"""Repeated runs of the E15 suite: a pass rate per check instead of one run's pass/fail. (E26 has its own
`--repeat` in eval/goals.py, with the LLM cache off.)

The LLM cache makes a rerun replay the same answers, so repeat k gets a cache file of its own: the repeats are
independent samples of the model. A whole agent run does not replay exactly from its cache (a rerun of E15 repeat k
hit about 60% of its calls, then took another path), so a repeat is a sample, not a reproducible fixture. Results of
repeat k are saved as .runtime/eval/<suite>_rep<k>.json (rows with `task`, `jd`, `checks`), and `section()` turns all
saved repeats into a table: per check, how often it passed across repeats and JDs, with a Wilson 95% interval.

    python -m app.eval.agentic --repeat 6     finished repeats are kept, so a larger N only adds repeats
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from app.config import get_settings


def cache_path(suite: str, k: int) -> Path:
    return get_settings().runtime_dir / f"llm_cache_{suite}_rep{k}.sqlite"


def result_path(suite: str, k: int) -> Path:
    return get_settings().runtime_dir / "eval" / f"{suite}_rep{k}.json"


def save(suite: str, k: int, rows: list[dict]) -> None:
    p = result_path(suite, k)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=1, default=str, ensure_ascii=False), encoding="utf-8")


def load(suite: str) -> dict[int, list[dict]]:
    out = {}
    for p in sorted((get_settings().runtime_dir / "eval").glob(f"{suite}_rep*.json")):
        out[int(p.stem.rsplit("_rep", 1)[1])] = json.loads(p.read_text(encoding="utf-8"))
    return dict(sorted(out.items()))


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a pass rate (sensible at small n and at 0% or 100%)."""
    if n == 0:
        return 0.0, 0.0
    p = passed / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - half) / (1 + z * z / n)), min(1.0, (centre + half) / (1 + z * z / n))


def rates(runs: dict[int, list[dict]]) -> list[tuple[str, str, int, int]]:
    """→ (task, check, passed, applicable) over every repeat and JD, in first-seen order."""
    tally: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for rows in runs.values():
        for r in rows:
            for check, ok in r["checks"].items():
                if ok is not None:
                    t = tally[(r["task"], check)]
                    t[0] += bool(ok)
                    t[1] += 1
    return [(task, check, p, n) for (task, check), (p, n) in tally.items()]


def section(runs: dict[int, list[dict]], title: str) -> list[str]:
    if not runs:
        return []
    totals = []
    for k, rows in runs.items():
        got = [ok for r in rows for ok in r["checks"].values() if ok is not None]
        totals.append(f"repeat {k}: {sum(got)}/{len(got)}")
    all_p = sum(p for *_, p, n in rates(runs))
    all_n = sum(n for *_, n in rates(runs))
    lines = [f"## {title}: pass rate per check over {len(runs)} independent repeats", "",
             "Each repeat has its own LLM cache, so the repeats are separate samples of the model (an agent run "
             "does not replay exactly from its cache). Rate = passed / applicable over repeats × JDs; 95% Wilson "
             "interval.", "",
             f"Per repeat: {', '.join(totals)}. **All: {all_p}/{all_n} ({all_p / all_n:.0%}).**", "",
             "| task | check | passed | rate | 95% CI |", "|---|---|---|---|---|"]
    for task, check, p, n in rates(runs):
        lo, hi = wilson(p, n)
        flag = "" if p == n else " ⚠" if p / n >= 0.5 else " ❌"
        lines.append(f"| {task} | {check} | {p}/{n} | {p / n:.0%}{flag} | {lo:.0%}–{hi:.0%} |")
    return lines + [""]
