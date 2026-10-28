"""Fine-tune the B1 dense encoder and the B4 cross-encoder on public resume–job pairs (ConFit-style, optional).

Data: ``cnamuangtoun/resume-job-description-fit`` (Hugging Face), downloaded directly — no local CV leaves the
machine (NFR-02). Leakage rules, because the same dataset is our real-data test bed (``data/public``):

* every pair whose job description is a QA / test-automation JD is dropped (those are the evaluation JDs);
* every resume that was judged for such a JD — in either CSV file — is dropped from all other pairs too;
* the remaining ``train.csv`` pairs are split by job description into train / dev; ``test.csv`` (also filtered)
  is a held-out check that is never used for early stopping or model selection.

Dense encoder: MultipleNegativesRankingLoss on (JD, Good Fit resume, No Fit resume of the same JD) triplets,
in-batch negatives without duplicates — the contrastive recipe of ConFit, without its data augmentation.
Cross-encoder: binary cross-entropy on (JD, resume) with labels No Fit / Potential / Good = 0 / 0.5 / 1.

The same code runs locally (``python -m cv_agent finetune``) and in ``notebooks/finetune_colab.ipynb``, which is
generated from this file by ``scripts/build_colab_notebook.py``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from cv_agent.data.public import MIN_QA_SIGNALS, RJF_REPO, download, qa_signals, read_pairs
from cv_agent.eval.metrics import ndcg_at_k

log = logging.getLogger(__name__)

Row = Tuple[str, str, int]            # (resume_text, job_description_text, label 0..2)
GOOD, NO_FIT, MAX_LABEL = 2, 0, 2
TARGETS = ("dense", "rerank")
DEFAULT_BASE = {"dense": "intfloat/multilingual-e5-base", "rerank": "BAAI/bge-reranker-base"}
DEV_SHARE = 0.1
EVAL_K = 10
RERANK_INFERENCE_LEN = 512   # same as cv_agent.ranking.reranker.CrossEncoderReranker


@dataclass(frozen=True)
class TrainConfig:
    target: str                       # dense | rerank
    base_model: str
    out_dir: Path
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_len: int = 256
    warmup_ratio: float = 0.1
    seed: int = 42
    limit: Optional[int] = None       # cap training examples / eval JDs for a quick smoke run

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {self.target!r}")
        if self.epochs < 1 or self.batch_size < 2 or self.max_len < 32 or not 0 < self.learning_rate < 1:
            raise ValueError("epochs >= 1, batch_size >= 2, max_len >= 32 and 0 < learning_rate < 1 are required")


# ------------------------------------------------------------------ data (pure, unit-tested)
def is_qa_jd(jd_text: str) -> bool:
    return len(qa_signals(jd_text)) >= MIN_QA_SIGNALS


def eval_resumes(rows: Sequence[Row]) -> Set[str]:
    return {r for r, jd, _ in rows if is_qa_jd(jd)}


def leakage_safe_rows(rows: Sequence[Row], judged: Optional[Set[str]] = None) -> List[Row]:
    """Drop QA JDs (our evaluation JDs) and every resume judged for one of them.

    ``judged`` must cover every file the evaluation set is built from (``data/public`` uses train + test).
    """
    eval_resumes_ = eval_resumes(rows) if judged is None else judged
    return [(r, jd, lab) for r, jd, lab in rows if not is_qa_jd(jd) and r not in eval_resumes_]


def _jd_key(jd_text: str) -> str:
    return hashlib.sha1(jd_text.encode("utf-8")).hexdigest()


def split_by_jd(rows: Sequence[Row], dev_share: float = DEV_SHARE) -> Tuple[List[Row], List[Row]]:
    """Group split: a JD and all its pairs land in exactly one split (stable hash order, input order ignored)."""
    jds = sorted({jd for _, jd, _ in rows}, key=_jd_key)
    n_dev = max(1, round(len(jds) * dev_share)) if len(jds) > 1 else 0
    dev_jds = set(jds[:n_dev])
    return ([row for row in rows if row[1] not in dev_jds], [row for row in rows if row[1] in dev_jds])


def triplets(rows: Sequence[Row]) -> List[Tuple[str, str, str]]:
    """(JD, Good Fit resume, No Fit resume of the same JD); negatives are reused round-robin."""
    good: Dict[str, List[str]] = defaultdict(list)
    bad: Dict[str, List[str]] = defaultdict(list)
    for resume, jd, label in rows:
        if label == GOOD:
            good[jd].append(resume)
        elif label == NO_FIT:
            bad[jd].append(resume)
    out = []
    for jd in sorted(good, key=_jd_key):
        negatives = sorted(bad.get(jd, []))
        for i, pos in enumerate(sorted(good[jd])):
            if negatives:
                out.append((jd, pos, negatives[i % len(negatives)]))
    return out


def cross_pairs(rows: Sequence[Row]) -> List[Tuple[str, str, float]]:
    return [(jd, resume, label / MAX_LABEL) for resume, jd, label in rows]


def with_prefix(texts: Sequence[str], kind: str, model_name: str) -> List[str]:
    """multilingual-e5 expects 'query: ' / 'passage: ' (same rule as SentenceTransformerEncoder)."""
    return [f"{kind}: {t}" for t in texts] if is_e5(model_name) else list(texts)


def is_e5(model_name: str) -> bool:
    """Judge by the model / folder name only, never by parent folders ('runs/e5-x/bge-m3-ft' is not e5)."""
    return "e5" in Path(model_name).name


ScoreFn = Callable[[str, List[str]], Sequence[float]]


def ndcg_by_jd(score_fn: ScoreFn, rows: Sequence[Row], k: int = EVAL_K) -> Dict[str, float]:
    """Mean nDCG@k over JDs with >= 2 judged resumes and >= 1 relevant one (condensed-list evaluation)."""
    by_jd: Dict[str, Dict[str, int]] = defaultdict(dict)
    for resume, jd, label in rows:
        by_jd[jd][resume] = max(label, by_jd[jd].get(resume, 0))
    scores = []
    for jd in sorted(by_jd, key=_jd_key):
        rel = by_jd[jd]
        if len(rel) < 2 or not any(rel.values()):
            continue
        resumes = sorted(rel)
        s = score_fn(jd, resumes)
        ranking = [r for _, r in sorted(zip(s, resumes), key=lambda p: (-p[0], p[1]))]
        scores.append(ndcg_at_k(ranking, rel, k))
    return {f"ndcg@{k}": sum(scores) / len(scores) if scores else float("nan"), "n_jds": len(scores)}


def _limit_jds(rows: Sequence[Row], limit: Optional[int]) -> List[Row]:
    if not limit:
        return list(rows)
    keep = set(sorted({jd for _, jd, _ in rows}, key=_jd_key)[:limit])
    return [row for row in rows if row[1] in keep]


# ------------------------------------------------------------------ models (needs requirements-ml.txt)
def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dense_scorer(model, model_name: str) -> ScoreFn:
    def score(jd: str, resumes: List[str]) -> List[float]:
        q = model.encode(with_prefix([jd], "query", model_name), normalize_embeddings=True)
        d = model.encode(with_prefix(resumes, "passage", model_name), normalize_embeddings=True, batch_size=32)
        return (d @ q[0]).tolist()
    return score


def _rerank_scorer(model) -> ScoreFn:
    def score(jd: str, resumes: List[str]) -> List[float]:
        return [float(x) for x in model.predict([(jd, r) for r in resumes], batch_size=32)]
    return score


def _training_args(cls, cfg: TrainConfig, **extra):
    fp16 = _device() == "cuda"
    return cls(output_dir=str(cfg.out_dir / "checkpoints"), num_train_epochs=cfg.epochs,
               per_device_train_batch_size=cfg.batch_size, learning_rate=cfg.learning_rate,
               warmup_ratio=cfg.warmup_ratio, fp16=fp16, seed=cfg.seed, save_strategy="no", logging_steps=20,
               report_to="none", **extra)


def train_dense(cfg: TrainConfig, train_rows: Sequence[Row], eval_sets: Dict[str, Sequence[Row]]) -> Dict:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments, losses)
    from sentence_transformers.training_args import BatchSamplers

    model = SentenceTransformer(cfg.base_model, device=_device())
    inference_len = model.max_seq_length  # evaluate and save at the length the pipeline uses (512 for e5)
    trips = triplets(train_rows)[: cfg.limit or None]
    before = {name: ndcg_by_jd(_dense_scorer(model, cfg.base_model), rows) for name, rows in eval_sets.items()}
    model.max_seq_length = cfg.max_len  # shorter only while training, for speed
    data = Dataset.from_dict({"anchor": with_prefix([t[0] for t in trips], "query", cfg.base_model),
                              "positive": with_prefix([t[1] for t in trips], "passage", cfg.base_model),
                              "negative": with_prefix([t[2] for t in trips], "passage", cfg.base_model)})
    args = _training_args(SentenceTransformerTrainingArguments, cfg, batch_sampler=BatchSamplers.NO_DUPLICATES)
    SentenceTransformerTrainer(model=model, args=args, train_dataset=data,
                               loss=losses.MultipleNegativesRankingLoss(model)).train()
    model.max_seq_length = inference_len
    model.save(str(cfg.out_dir))
    after = {name: ndcg_by_jd(_dense_scorer(model, cfg.base_model), rows) for name, rows in eval_sets.items()}
    return {"n_train_examples": len(trips), "before": before, "after": after}


def train_rerank(cfg: TrainConfig, train_rows: Sequence[Row], eval_sets: Dict[str, Sequence[Row]]) -> Dict:
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = CrossEncoder(cfg.base_model, num_labels=1, max_length=RERANK_INFERENCE_LEN, device=_device())
    pairs = cross_pairs(train_rows)[: cfg.limit or None]
    before = {name: ndcg_by_jd(_rerank_scorer(model), rows) for name, rows in eval_sets.items()}
    model.max_length = cfg.max_len  # shorter only while training, for speed
    data = Dataset.from_dict({"query": [p[0] for p in pairs], "passage": [p[1] for p in pairs],
                              "label": [p[2] for p in pairs]})
    CrossEncoderTrainer(model=model, args=_training_args(CrossEncoderTrainingArguments, cfg), train_dataset=data,
                        loss=BinaryCrossEntropyLoss(model)).train()
    model.max_length = RERANK_INFERENCE_LEN
    model.save(str(cfg.out_dir))
    after = {name: ndcg_by_jd(_rerank_scorer(model), rows) for name, rows in eval_sets.items()}
    return {"n_train_examples": len(pairs), "before": before, "after": after}


def split_files(train_csv: Sequence[Row], test_csv: Sequence[Row]) -> Dict[str, List[Row]]:
    """Leakage-safe train / dev / held-out test rows; evaluation resumes are collected over BOTH files."""
    judged = eval_resumes(list(train_csv) + list(test_csv))
    train_all, test_all = leakage_safe_rows(train_csv, judged), leakage_safe_rows(test_csv, judged)
    train, dev = split_by_jd(train_all)
    seen = {jd for _, jd, _ in train_all}
    return {"train": train, "dev": dev, "test": [row for row in test_all if row[1] not in seen]}


def load_rows(raw_dir: Path) -> Dict[str, List[Row]]:
    return split_files(read_pairs([download(RJF_REPO, "train.csv", raw_dir)]),
                       read_pairs([download(RJF_REPO, "test.csv", raw_dir)]))


def run(cfg: TrainConfig, raw_dir: Path) -> Dict:
    """Train one target, save the model to ``cfg.out_dir`` and write ``finetune_metrics.json`` next to it."""
    started = time.time()
    rows = load_rows(raw_dir)
    eval_sets = {"dev": _limit_jds(rows["dev"], cfg.limit), "test": _limit_jds(rows["test"], cfg.limit)}
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    trainer = train_dense if cfg.target == "dense" else train_rerank
    result = trainer(cfg, rows["train"], eval_sets)
    report = {"config": {**asdict(cfg), "out_dir": str(cfg.out_dir)}, "device": _device(),
              "data": {name: {"pairs": len(r), "jds": len({jd for _, jd, _ in r})} for name, r in rows.items()},
              **result, "wall_s": round(time.time() - started, 1)}
    (cfg.out_dir / "finetune_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("fine-tuned %s -> %s: %s", cfg.target, cfg.out_dir, result["after"])
    return report
