"""Ranking metrics and statistics. Pure functions, no dependencies, unit-tested.

Graded relevance g ∈ [0, 3] (mean of annotators, so half-steps occur). nDCG uses gain 2^g − 1 (Järvelin &
Kekäläinen); Kendall's τ-b handles the many ties in graded labels; CIs are percentile bootstraps over JDs, the
unit we sample, so with 12 JDs they are honest about how little 12 JDs can prove."""

import math
import random
from collections.abc import Sequence


def dcg(gains: Sequence[float], k: int) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def ndcg(ranked_ids: Sequence[str], grades: dict[str, float], k: int) -> float:
    ideal = dcg(sorted(grades.values(), reverse=True), k)
    return dcg([grades.get(c, 0.0) for c in ranked_ids], k) / ideal if ideal else 0.0


def precision(ranked_ids: Sequence[str], grades: dict[str, float], k: int, threshold: float = 2.0) -> float:
    return sum(grades.get(c, 0.0) >= threshold for c in ranked_ids[:k]) / k


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """τ-b between two score lists (ties allowed in both)."""
    conc = disc = tx = ty = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom else 0.0


def tau_vs_gold(ranked_ids: Sequence[str], grades: dict[str, float]) -> float:
    """τ-b between the system's order (position) and the gold grades, over every labelled CV."""
    pos = {c: i for i, c in enumerate(ranked_ids)}
    ids = [c for c in grades if c in pos]
    return kendall_tau_b([-pos[c] for c in ids], [grades[c] for c in ids])


def weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = 4) -> float:
    """Quadratic-weighted Cohen's κ between two annotators on a 0..k-1 scale."""
    n = len(a)
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    ra = [sum(r) for r in obs]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = sum((i - j) ** 2 * obs[i][j] for i in range(k) for j in range(k))
    den = sum((i - j) ** 2 * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else 1.0


def bootstrap_ci(values: Sequence[float], n: int = 5000, seed: int = 7) -> tuple[float, float, float]:
    """(mean, 2.5th, 97.5th percentile) of the mean, resampling the per-JD values."""
    rnd = random.Random(seed)
    means = sorted(sum(rnd.choices(values, k=len(values))) / len(values) for _ in range(n))
    return sum(values) / len(values), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired_bootstrap_p(a: Sequence[float], b: Sequence[float], n: int = 5000, seed: int = 7) -> float:
    """One-sided p-value for mean(a) > mean(b) on the same JDs (share of resamples where the gain vanishes)."""
    rnd = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    return sum(sum(rnd.choices(d, k=len(d))) <= 0 for _ in range(n)) / n
