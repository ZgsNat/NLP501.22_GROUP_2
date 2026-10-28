"""Okapi BM25 over normalised CV text (FR-11, config B0)."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

_TOKEN = re.compile(r"[a-zà-ỹđ0-9#+.]+", re.I)
_STOP = {"the", "and", "or", "of", "in", "to", "a", "an", "with", "for", "on", "at", "is", "are", "be", "as", "by",
         "và", "của", "các", "có", "trong", "cho", "với", "là", "được", "một", "những", "về"}


def tokenize(text: str) -> List[str]:
    return [t.strip(".") for t in _TOKEN.findall(text.lower()) if t.strip(".") and t.strip(".") not in _STOP]


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.ids: List[str] = []
        self.tfs: List[Counter] = []
        self.doc_len: List[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, docs: Sequence[Tuple[str, str]]) -> "BM25Index":
        for doc_id, text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.ids.append(doc_id)
            self.tfs.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        return self

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> Dict[str, float]:
        q_terms = tokenize(query)
        out: Dict[str, float] = {}
        for doc_id, tf, dl in zip(self.ids, self.tfs, self.doc_len):
            s = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    s += self.idf(term) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            out[doc_id] = s
        return out

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
