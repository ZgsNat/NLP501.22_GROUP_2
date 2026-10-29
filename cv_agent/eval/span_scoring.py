"""Dependency-free span scoring shared by the taxonomy extractor and the learned skill tagger.

Spans are character offsets ``(start, end)`` in the text built by joining SkillSpan tokens with single spaces.
Matching is ``exact`` (same offsets) or ``overlap`` (any shared character).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

CharSpan = Tuple[int, int]


def bio_spans(tokens: List[str], tags: List[str]) -> Tuple[str, List[CharSpan]]:
    """Join tokens with single spaces and convert B-/I- tags to character spans in that text."""
    text, offsets = "", []
    for tok in tokens:
        if text:
            text += " "
        offsets.append((len(text), len(text) + len(tok)))
        text += tok
    spans: List[CharSpan] = []
    start = end = None
    for (s, e), tag in zip(offsets, tags):
        if tag.startswith("B") or (tag.startswith("I") and start is None):
            if start is not None:
                spans.append((start, end))
            start, end = s, e
        elif tag.startswith("I"):
            end = e
        else:
            if start is not None:
                spans.append((start, end))
            start = None
    if start is not None:
        spans.append((start, end))
    return text, spans


def gold_spans(sent: dict) -> List[CharSpan]:
    """SkillSpan has separate skill and knowledge BIO layers: convert each, then union (never merge token-wise)."""
    _, skill = bio_spans(sent["tokens"], sent["tags_skill"])
    _, knowledge = bio_spans(sent["tokens"], sent["tags_knowledge"])
    return sorted(set(skill) | set(knowledge))


def overlaps(a: CharSpan, b: CharSpan) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def prf(tp_pred: int, n_pred: int, tp_gold: int, n_gold: int) -> Tuple[float, float, float]:
    """Precision over predictions, recall over gold spans (they differ for overlap matching)."""
    p = tp_pred / n_pred if n_pred else float("nan")
    r = tp_gold / n_gold if n_gold else float("nan")
    f = 2 * p * r / (p + r) if n_pred and n_gold and p + r else 0.0
    return p, r, f


def score_spans(pairs: Iterable[Tuple[Sequence[CharSpan], Sequence[CharSpan]]]) -> Dict[str, float]:
    """``pairs``: (predicted spans, gold spans) per sentence."""
    n_pred = n_gold = tp_exact = tp_pred_overlap = tp_gold_overlap = 0
    for pred, gold in pairs:
        gold_set = set(gold)
        n_pred += len(pred)
        n_gold += len(gold)
        tp_exact += sum(1 for p in pred if p in gold_set)
        tp_pred_overlap += sum(1 for p in pred if any(overlaps(p, g) for g in gold))
        tp_gold_overlap += sum(1 for g in gold if any(overlaps(p, g) for p in pred))
    out: Dict[str, float] = {"n_gold_spans": n_gold, "n_pred_spans": n_pred}
    for name, tp_p, tp_g in (("exact", tp_exact, tp_exact), ("overlap", tp_pred_overlap, tp_gold_overlap)):
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = prf(tp_p, n_pred, tp_g, n_gold)
    return out
