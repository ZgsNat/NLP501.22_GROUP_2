"""Span-level evaluation of the entity extractor (FR-03) on SkillSpan (BIO tags over tokens).

SkillSpan marks every skill / knowledge phrase of any domain, while the extractor only knows the
Automation-Test taxonomy, so two recalls are reported:
* ``recall`` over all gold spans (low by construction), and
* ``recall_in_taxonomy`` over gold spans that contain a term the taxonomy knows. With overlap matching this
  is ~1.0 by construction (such a span contains an alias the extractor looks for); the informative number
  is ``recall_in_taxonomy_exact``: are the *boundaries* right (e.g. "Selenium WebDriver" vs "Selenium")?
Matching is ``exact`` (same character offsets) or ``overlap`` (any shared character).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from cv_agent.ingest.extractor import extract_entities
from cv_agent.ingest.taxonomy import ALIAS_INDEX

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


def _overlaps(a: CharSpan, b: CharSpan) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _in_taxonomy(text: str) -> bool:
    return any(pattern.search(text) for pattern, _skill, _alias in ALIAS_INDEX)


def score_sentences(sentences: Iterable[dict]) -> Dict[str, float]:
    tp_exact = tp_overlap = n_pred = n_gold = n_gold_tax = hit_tax = hit_tax_exact = 0
    for sent in sentences:
        # skill and knowledge are separate BIO layers: convert each, then union (never merge tags token-wise)
        text, skill = bio_spans(sent["tokens"], sent["tags_skill"])
        _, knowledge = bio_spans(sent["tokens"], sent["tags_knowledge"])
        gold = sorted(set(skill) | set(knowledge))
        pred = [(e.span.start, e.span.end) for e in extract_entities(text)
                if e.type != "years"]
        n_pred += len(pred)
        n_gold += len(gold)
        tp_exact += sum(1 for p in pred if p in gold)
        tp_overlap += sum(1 for p in pred if any(_overlaps(p, g) for g in gold))
        for g in gold:
            if _in_taxonomy(text[g[0]:g[1]]):
                n_gold_tax += 1
                hit_tax += any(_overlaps(p, g) for p in pred)
                hit_tax_exact += g in pred
    out: Dict[str, float] = {"n_gold_spans": n_gold, "n_pred_spans": n_pred, "n_gold_in_taxonomy": n_gold_tax}
    for name, tp in (("exact", tp_exact), ("overlap", tp_overlap)):
        p = tp / n_pred if n_pred else float("nan")
        r = tp / n_gold if n_gold else float("nan")
        out[f"precision_{name}"], out[f"recall_{name}"] = p, r
        out[f"f1_{name}"] = 2 * p * r / (p + r) if p + r else 0.0
    out["recall_in_taxonomy"] = hit_tax / n_gold_tax if n_gold_tax else float("nan")
    out["recall_in_taxonomy_exact"] = hit_tax_exact / n_gold_tax if n_gold_tax else float("nan")
    return out


def load_skillspan(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
