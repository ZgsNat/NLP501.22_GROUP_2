"""Span-level evaluation of the entity extractor (FR-03) on SkillSpan (BIO tags over tokens).

SkillSpan marks every skill / knowledge phrase of any domain, while the extractor only knows the
Automation-Test taxonomy, so two recalls are reported:
* ``recall`` over all gold spans (low by construction), and
* ``recall_in_taxonomy`` over gold spans that contain a term the taxonomy knows. With overlap matching this
  is ~1.0 by construction (such a span contains an alias the extractor looks for); the informative number
  is ``recall_in_taxonomy_exact``: are the *boundaries* right (e.g. "Selenium WebDriver" vs "Selenium")?
Matching is ``exact`` (same character offsets) or ``overlap`` (any shared character); any span predictor
(the taxonomy extractor, the learned tagger in ``cv_agent.training.skill_tagger``, or both) can be scored.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from cv_agent.eval.span_scoring import CharSpan, bio_spans, gold_spans, overlaps, score_spans
from cv_agent.ingest.extractor import extract_entities
from cv_agent.ingest.taxonomy import ALIAS_INDEX

# (text, tokens, sentence) -> predicted character spans in ``text``
Predictor = Callable[[str, List[str], dict], List[CharSpan]]

__all__ = ["bio_spans", "gold_spans", "taxonomy_predictor", "score_sentences", "load_skillspan"]


def _in_taxonomy(text: str) -> bool:
    return any(pattern.search(text) for pattern, _skill, _alias in ALIAS_INDEX)


def taxonomy_predictor(text: str, tokens: List[str], sent: dict) -> List[CharSpan]:
    return [(e.span.start, e.span.end) for e in extract_entities(text) if e.type != "years"]


def score_sentences(sentences: Iterable[dict], predict: Predictor = taxonomy_predictor) -> Dict[str, float]:
    pairs = []
    n_gold_tax = hit_tax = hit_tax_exact = 0
    for sent in sentences:
        text, _ = bio_spans(sent["tokens"], [])
        gold = gold_spans(sent)
        pred = sorted(set(predict(text, sent["tokens"], sent)))
        pairs.append((pred, gold))
        for g in gold:
            if _in_taxonomy(text[g[0]:g[1]]):
                n_gold_tax += 1
                hit_tax += any(overlaps(p, g) for p in pred)
                hit_tax_exact += g in pred
    out = {**score_spans(pairs), "n_gold_in_taxonomy": n_gold_tax}
    out["recall_in_taxonomy"] = hit_tax / n_gold_tax if n_gold_tax else float("nan")
    out["recall_in_taxonomy_exact"] = hit_tax_exact / n_gold_tax if n_gold_tax else float("nan")
    return out


def load_skillspan(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
