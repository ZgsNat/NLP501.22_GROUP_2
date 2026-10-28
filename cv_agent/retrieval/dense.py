"""Dense retrieval (FR-12, config B1).

Encoders:
* ``sentence_transformers`` — multilingual-e5 / bge-m3 (needs ``requirements-ml.txt``)
* ``ollama``                — embedding model served by Ollama (e.g. ``ollama pull bge-m3``)
* ``hashing``               — dependency-free fallback: hashed char n-grams + taxonomy
                              concepts. Lexical-semantic only; report it as such.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from cv_agent.ingest.extractor import extract_entities
from cv_agent.ingest.taxonomy import SKILLS

log = logging.getLogger(__name__)


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray: ...


def _l2norm(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEncoder:
    name = "hashing"

    def __init__(self, dim: int = 4096, ngram: Tuple[int, int] = (3, 5)) -> None:
        self.dim, self.ngram = dim, ngram

    def _h(self, s: str) -> int:
        return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % self.dim

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = re.sub(r"\s+", " ", text.lower())
            for n in range(self.ngram[0], self.ngram[1] + 1):
                for j in range(max(len(t) - n + 1, 0)):
                    out[i, self._h(t[j:j + n])] += 1.0
            out[i] = np.log1p(out[i])
            # concept features: canonical skill + its related skills (soft synonymy)
            for ent in extract_entities(text):
                out[i, self._h("C:" + ent.value)] += 3.0
                for rel in (SKILLS[ent.value].related if ent.value in SKILLS else ()):
                    out[i, self._h("C:" + rel)] += 1.0
        return _l2norm(out)


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: Optional[str] = None) -> None:
        from sentence_transformers import SentenceTransformer  # optional heavy dependency
        self.model = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"
        self.e5 = "e5" in Path(model_name).name  # folder name only: 'runs/e5-x/bge-m3-ft' is not e5

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if self.e5:
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=16), dtype=np.float32)


class OllamaEncoder:
    def __init__(self, model: str = "bge-m3", host: str = "http://localhost:11434", timeout: float = 120) -> None:
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = f"ollama:{model}"

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        req = urllib.request.Request(f"{self.host}/api/embed", data=json.dumps(
            {"model": self.model, "input": list(texts)}).encode("utf-8"), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - local endpoint
            vecs = json.loads(resp.read())["embeddings"]
        return _l2norm(np.asarray(vecs, dtype=np.float32))


def build_encoder(cfg: Dict) -> Encoder:
    kind = cfg.get("encoder", "hashing")
    try:
        if kind == "sentence_transformers":
            return SentenceTransformerEncoder(cfg.get("model", "intfloat/multilingual-e5-base"))
        if kind == "ollama":
            return OllamaEncoder(cfg.get("model", "bge-m3"), cfg.get("host", "http://localhost:11434"))
    except Exception as exc:  # explicit, logged fallback
        log.warning("Dense encoder %s unavailable (%s); falling back to hashing encoder", kind, exc)
    return HashingEncoder(dim=cfg.get("dim", 4096))


class DenseIndex:
    def __init__(self, encoder: Encoder, max_chars: int = 4000) -> None:
        self.encoder, self.max_chars = encoder, max_chars
        self.ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None

    def fit(self, docs: Sequence[Tuple[str, str]]) -> "DenseIndex":
        self.ids = [d for d, _ in docs]
        self.matrix = self.encoder.encode([t[: self.max_chars] for _, t in docs]) if docs else np.zeros((0, 1))
        return self

    def scores(self, query: str) -> Dict[str, float]:
        if self.matrix is None or not self.ids:
            return {}
        q = self.encoder.encode([query], is_query=True)[0]
        sims = self.matrix @ q
        return {doc_id: float(s) for doc_id, s in zip(self.ids, sims)}

    def search(self, query: str, k: int) -> List[Tuple[str, float]]:
        return sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))[:k]
