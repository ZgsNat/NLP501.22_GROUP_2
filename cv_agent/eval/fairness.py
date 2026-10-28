"""Fairness audit (NFR-06, H5): name-swap Δrank and the 4/5 (impact ratio) rule."""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Sequence

from cv_agent.data.synthetic import GIVEN_F, GIVEN_M, MIDDLE_F, MIDDLE_M, SURNAMES


def impact_ratio(selected: Sequence[str], groups: Dict[str, str]) -> Dict[str, float]:
    """Selection rate per group among candidates with a known group; ratio = min/max."""
    pool: Dict[str, List[str]] = {}
    for cid, g in groups.items():
        pool.setdefault(g, []).append(cid)
    sel = set(selected)
    rates = {g: (sum(c in sel for c in ids) / len(ids)) for g, ids in pool.items() if ids}
    hi = max(rates.values()) if rates else 0.0
    return {"rates": rates, "impact_ratio": (min(rates.values()) / hi) if hi else float("nan")}


def swap_name(raw_text: str, old_name: str, rng: random.Random) -> str:
    """Replace the candidate's name with a random name of the opposite gender pattern."""
    female = any(m in old_name.split() for m in MIDDLE_F)
    mid, giv = (MIDDLE_M, GIVEN_M) if female else (MIDDLE_F, GIVEN_F)
    new = f"{rng.choice(SURNAMES)} {rng.choice(mid)} {rng.choice(giv)}"
    text = raw_text.replace(old_name, new)
    # gender line follows the swapped name
    swap = {"Male": "Female", "Female": "Male", "Nam": "Nữ", "Nữ": "Nam"}
    return re.sub(r"(?m)^((?:Gender|Giới tính):\s*)(\w+)", lambda m: m.group(1) + swap.get(m.group(2), m.group(2)), text)


def name_swap_delta(rank_fn: Callable[[Dict[str, str]], List[str]], raw_texts: Dict[str, str],
                    names: Dict[str, str], seed: int = 42) -> Dict[str, float]:
    """``rank_fn`` maps {cv_id: raw_text} -> ranked ids. Returns mean/max |Δrank| after swapping names."""
    rng = random.Random(seed)
    base = rank_fn(raw_texts)
    swapped = {cid: swap_name(t, names.get(cid, ""), rng) if names.get(cid) else t for cid, t in raw_texts.items()}
    new = rank_fn(swapped)
    pos_a, pos_b = {c: i for i, c in enumerate(base)}, {c: i for i, c in enumerate(new)}
    deltas = [abs(pos_a[c] - pos_b[c]) for c in pos_a if c in pos_b]
    return {"mean_abs_delta_rank": sum(deltas) / len(deltas) if deltas else float("nan"),
            "max_abs_delta_rank": max(deltas) if deltas else float("nan"), "n": len(deltas)}
