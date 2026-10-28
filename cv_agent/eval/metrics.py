"""Ranking metrics, agreement and paired statistics (D5)."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple


def recall_at_k(ranking: Sequence[str], rel: Dict[str, int], k: int, min_grade: int = 1) -> float:
    relevant = {c for c, g in rel.items() if g >= min_grade}
    if not relevant:
        return float("nan")
    return len(relevant & set(ranking[:k])) / len(relevant)


def mrr_at_k(ranking: Sequence[str], rel: Dict[str, int], k: int = 10, min_grade: int = 2) -> float:
    for i, c in enumerate(ranking[:k], start=1):
        if rel.get(c, 0) >= min_grade:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking: Sequence[str], rel: Dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(c, 0) - 1) / math.log2(i + 2) for i, c in enumerate(ranking[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else float("nan")


def kendall_tau(system: Sequence[str], hr: Sequence[str]) -> float:
    """Kendall τ-b between HR order and the system's order over HR-ranked items.

    Items the system did not return are tied at the bottom of the system order.
    """
    pos = {c: i for i, c in enumerate(system)}
    bottom = len(system)
    sys_rank = [pos.get(c, bottom) for c in hr]
    n = len(hr)
    if n < 2:
        return float("nan")
    conc = disc = ties_sys = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = sys_rank[i] - sys_rank[j]
            if d == 0:
                ties_sys += 1
            elif d < 0:
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    denom = math.sqrt(n0 * (n0 - ties_sys))
    return (conc - disc) / denom if denom else 0.0


def cohen_kappa(a: Dict[str, int], b: Dict[str, int]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return float("nan")
    cats = sorted({a[k] for k in keys} | {b[k] for k in keys})
    n = len(keys)
    po = sum(a[k] == b[k] for k in keys) / n
    pe = sum((sum(a[k] == c for k in keys) / n) * (sum(b[k] == c for k in keys) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# ---------------------------------------------------------------- paired statistics over JDs

def _clean(x: Sequence[float], y: Sequence[float]) -> List[Tuple[float, float]]:
    return [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]


def paired_bootstrap(x: Sequence[float], y: Sequence[float], n_boot: int = 10_000, seed: int = 42,
                     alpha: float = 0.05) -> Dict[str, float]:
    """Mean difference x - y with percentile CI, resampling JDs (paired)."""
    pairs = _clean(x, y)
    if not pairs:
        return {"mean_diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    diffs = [a - b for a, b in pairs]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    lo, hi = boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]
    return {"mean_diff": sum(diffs) / len(diffs), "ci_low": lo, "ci_high": hi, "n": len(diffs)}


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank (normal approximation; exact via scipy if installed)."""
    diffs = [a - b for a, b in _clean(x, y) if a - b != 0]
    n = len(diffs)
    if n == 0:
        return {"W": 0.0, "p_value": 1.0, "n": 0, "effect_size_r": 0.0}
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(diffs)
        p = float(res.pvalue)
    except ImportError:
        p = None
    order = sorted(range(n), key=lambda i: abs(diffs[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:  # average ranks for ties
        j = i
        while j + 1 < n and abs(diffs[order[j + 1]]) == abs(diffs[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mean) / sd if sd else 0.0
    if p is None:
        p = math.erfc(abs(z) / math.sqrt(2))
    return {"W": w_plus, "p_value": p, "n": n, "effect_size_r": z / math.sqrt(n)}


def graded_tau(system: Sequence[str], relevance: Dict[str, int]) -> float:
    """Kendall-style agreement with graded labels: only pairs with *different* grades count, so the arbitrary
    order inside one grade (e.g. two "Good Fit" CVs) is not penalised. (concordant - discordant) / both."""
    items = [c for c in system if c in relevance]
    conc = disc = 0
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if relevance[a] > relevance[b]:
                conc += 1
            elif relevance[a] < relevance[b]:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else float("nan")

