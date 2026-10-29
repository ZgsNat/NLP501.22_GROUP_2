"""Learned skill / knowledge span tagger (FR-03): BIO token classification fine-tuned on SkillSpan.

The taxonomy extractor (``cv_agent.ingest.extractor``) only knows Automation-Test terms, so its recall on SkillSpan
is low by design. This module trains an open-vocabulary tagger and scores it with exactly the same span metrics:

* data: ``jjzha/skillspan`` (CC-BY-4.0) train / dev / test, downloaded directly; the skill and knowledge layers are
  merged into one tag set (``B-SKILL``, ``I-SKILL``, ``B-KNOWLEDGE``, ``I-KNOWLEDGE``, ``O``), skill first on overlap;
* model: ``jjzha/jobbert-base-cased`` (BERT further pre-trained on job postings, the SkillSpan authors' encoder);
* selection: the epoch with the best exact span F1 on dev; test is scored once, after training;
* report: test P/R/F1 of the tagger, the taxonomy extractor and their union (``tagger_metrics.json``).

Runs locally (``python -m cv_agent train-tagger``) and in ``notebooks/finetune_colab.ipynb``.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from cv_agent.eval.span_scoring import CharSpan, bio_spans, gold_spans, score_spans

log = logging.getLogger(__name__)

LABELS = ("O", "B-SKILL", "I-SKILL", "B-KNOWLEDGE", "I-KNOWLEDGE")
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
KINDS = ("SKILL", "KNOWLEDGE")
IGNORE = -100                     # label id ignored by the loss (special tokens, non-first subwords)
DEFAULT_BASE = "jjzha/jobbert-base-cased"
SKILLSPAN_REPO = "jjzha/skillspan"
SPLITS = ("train", "dev", "test")
PREDICT_MAX_LEN = 512             # long sentences are cut at training time only
WARMUP_RATIO = 0.1

Predictor = Callable[[str, List[str], dict], List[CharSpan]]


@dataclass(frozen=True)
class TaggerConfig:
    base_model: str = DEFAULT_BASE
    out_dir: Path = Path("models") / "jobbert-base-cased-skillspan-tagger"   # same name as train-tagger
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_len: int = 128
    seed: int = 42
    limit: Optional[int] = None       # cap sentences per split for a quick smoke run

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.max_len < 16 or not 0 < self.learning_rate < 1:
            raise ValueError("epochs >= 1, batch_size >= 1, max_len >= 16 and 0 < learning_rate < 1 are required")


# ------------------------------------------------------------------ labels (pure, unit-tested)
def _repair(labels: Sequence[str]) -> List[str]:
    """An I- tag that does not continue a span of the same kind starts a new one."""
    out: List[str] = []
    for label in labels:
        if label.startswith("I-") and (not out or out[-1] == "O" or out[-1][2:] != label[2:]):
            label = "B-" + label[2:]
        out.append(label)
    return out


def merge_layers(tags_skill: Sequence[str], tags_knowledge: Sequence[str]) -> List[str]:
    merged = [f"{s[0]}-SKILL" if s != "O" else f"{k[0]}-KNOWLEDGE" if k != "O" else "O"
              for s, k in zip(tags_skill, tags_knowledge)]
    return _repair(merged)


def align_labels(word_ids: Sequence[Optional[int]], labels: Sequence[str]) -> List[int]:
    out, previous = [], None
    for w in word_ids:
        out.append(IGNORE if w is None or w == previous else LABEL2ID[labels[w]])
        previous = w
    return out


def word_labels(word_ids: Sequence[Optional[int]], pred_ids: Sequence[int], n_words: int) -> List[str]:
    """Label of each word from its first subword; words cut off by truncation stay ``O``."""
    out, seen = ["O"] * n_words, set()
    for w, p in zip(word_ids, pred_ids):
        if w is not None and w not in seen and w < n_words:
            out[w] = LABELS[p]
            seen.add(w)
    return out


def label_spans(tokens: List[str], labels: Sequence[str]) -> Tuple[str, List[CharSpan]]:
    """Character spans per kind, so a SKILL span never swallows an adjacent KNOWLEDGE span."""
    text, spans = bio_spans(tokens, [])
    for kind in KINDS:
        _, kind_spans = bio_spans(tokens, [lab[0] if lab.endswith(kind) else "O" for lab in labels])
        spans.extend(kind_spans)
    return text, sorted(set(spans))


def span_f1_from_ids(pred_ids: Sequence[Sequence[int]], label_ids: Sequence[Sequence[int]]) -> float:
    """Exact span F1 at word level (positions labelled IGNORE are dropped), used to pick the best epoch."""
    pairs = []
    for preds, golds in zip(pred_ids, label_ids):
        kept = [(p, g) for p, g in zip(preds, golds) if g != IGNORE]
        dummy = ["x"] * len(kept)
        pairs.append((label_spans(dummy, [LABELS[p] for p, _ in kept])[1],
                      label_spans(dummy, [LABELS[g] for _, g in kept])[1]))
    return score_spans(pairs)["f1_exact"]


def union_predictor(*predictors: Predictor) -> Predictor:
    def predict(text: str, tokens: List[str], sent: dict) -> List[CharSpan]:
        return sorted({span for p in predictors for span in p(text, tokens, sent)})
    return predict


def evaluate(sentences: Sequence[dict], predict: Predictor) -> Dict[str, float]:
    pairs = []
    for sent in sentences:
        text, _ = bio_spans(sent["tokens"], [])
        pairs.append((sorted(set(predict(text, sent["tokens"], sent))), gold_spans(sent)))
    return score_spans(pairs)


# ------------------------------------------------------------------ data
def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_splits(raw_dir: Path) -> Dict[str, List[dict]]:
    from cv_agent.data.public import download
    return {split: _read_jsonl(download(SKILLSPAN_REPO, f"{split}.json", raw_dir)) for split in SPLITS}


# ------------------------------------------------------------------ training (needs transformers + torch)
def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _warmup(args_cls, ratio: float) -> Dict[str, float]:
    """transformers 5 folds ``warmup_ratio`` into ``warmup_steps`` (a float < 1 is a ratio); 4.x keeps both."""
    import inspect
    params = inspect.signature(args_cls.__init__).parameters
    return {"warmup_ratio": ratio} if "warmup_ratio" in params else {"warmup_steps": ratio}


def _encode(tokenizer, sentences: Sequence[dict], max_len: int) -> List[Dict]:
    rows = []
    for sent in sentences:
        enc = tokenizer(sent["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        labels = align_labels(enc.word_ids(), merge_layers(sent["tags_skill"], sent["tags_knowledge"]))
        rows.append({**dict(enc), "labels": labels})
    return rows


def fit(cfg: TaggerConfig, train: Sequence[dict], dev: Sequence[dict]):
    """Fine-tune, keep the epoch with the best dev span F1, save it to ``cfg.out_dir``."""
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              Trainer, TrainingArguments, set_seed)
    set_seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.base_model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    ckpt = cfg.out_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt), num_train_epochs=cfg.epochs, per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2, learning_rate=cfg.learning_rate,
        **_warmup(TrainingArguments, WARMUP_RATIO), weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="span_f1", greater_is_better=True, seed=cfg.seed,
        fp16=_device() == "cuda", report_to="none", logging_steps=50)

    def compute_metrics(pred) -> Dict[str, float]:
        return {"span_f1": span_f1_from_ids(pred.predictions.argmax(-1).tolist(), pred.label_ids.tolist())}

    trainer = Trainer(model=model, args=args, train_dataset=_encode(tokenizer, train, cfg.max_len),
                      eval_dataset=_encode(tokenizer, dev, cfg.max_len),
                      data_collator=DataCollatorForTokenClassification(tokenizer), compute_metrics=compute_metrics)
    trainer.train()
    trainer.model.save_pretrained(cfg.out_dir)
    tokenizer.save_pretrained(cfg.out_dir)
    shutil.rmtree(ckpt, ignore_errors=True)
    history = [{"epoch": h["epoch"], "dev_span_f1": h["eval_span_f1"]}
               for h in trainer.state.log_history if "eval_span_f1" in h]
    return (trainer.model, tokenizer), {"best_dev_span_f1": trainer.state.best_metric, "dev_by_epoch": history}


def model_predictor(model_and_tokenizer, cfg: TaggerConfig) -> Predictor:
    import torch
    model, tokenizer = model_and_tokenizer
    model.eval()
    device = next(model.parameters()).device

    def predict(text: str, tokens: List[str], sent: dict) -> List[CharSpan]:
        if not tokens:
            return []
        enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=PREDICT_MAX_LEN,
                        return_tensors="pt")
        with torch.no_grad():
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits[0]
        return label_spans(tokens, word_labels(enc.word_ids(), logits.argmax(-1).tolist(), len(tokens)))[1]
    return predict


def load_predictor(model_dir: Path) -> Predictor:
    """Predictor of a tagger saved by :func:`run` (``eval-spans --tagger``)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to(_device())
    return model_predictor((model, AutoTokenizer.from_pretrained(model_dir)), TaggerConfig(out_dir=Path(model_dir)))


def run(cfg: TaggerConfig, raw_dir: Path) -> Dict:
    """Train on SkillSpan train, select on dev, score once on test; write ``tagger_metrics.json``."""
    started = time.time()
    splits = {k: v[:cfg.limit] if cfg.limit else v for k, v in load_splits(raw_dir).items()}
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    model, fit_info = fit(cfg, splits["train"], splits["dev"])
    tagger = model_predictor(model, cfg)
    predictors: Dict[str, Predictor] = {"tagger": tagger}
    try:  # the taxonomy baseline needs the whole package (absent in the Colab notebook)
        from cv_agent.eval.span_metrics import taxonomy_predictor
        predictors.update(taxonomy=taxonomy_predictor, hybrid=union_predictor(tagger, taxonomy_predictor))
    except ImportError:
        log.info("taxonomy extractor not available: scoring the tagger only")
    report = {"config": {**asdict(cfg), "out_dir": str(cfg.out_dir)}, "device": _device(),
              "data": {k: len(v) for k, v in splits.items()}, "fit": fit_info,
              "dev": {"tagger": evaluate(splits["dev"], tagger)},
              "test": {name: evaluate(splits["test"], p) for name, p in predictors.items()},
              "wall_s": round(time.time() - started, 1)}
    (cfg.out_dir / "tagger_metrics.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    log.info("skill tagger -> %s: test %s", cfg.out_dir, report["test"]["tagger"])
    return report
