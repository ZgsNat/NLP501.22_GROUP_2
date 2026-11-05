"""Train the E20 cross-encoder from an exported JSONL; standalone (torch + transformers only), so it runs on Colab.

    local:  cd backend && uv run python -m app.eval.finetune export        → .runtime/colab/ce_train.jsonl
    Colab:  python colab_train_ce.py ce_train.jsonl ce-ft-dev               (GPU if there is one)
    local:  cd backend && uv run python -m app.eval.finetune install ce-ft-dev.zip   → models/ce-ft-dev

Each JSONL line is {"q": requirement queries, "e": evidence passages, "y": label/3}. The hyper-parameters are fixed
(no search) and recorded in <out>/train.json next to the weights. app.eval.finetune.train calls the same function.
"""

import json
import random
import sys
from pathlib import Path

BASE = "cross-encoder/ms-marco-MiniLM-L6-v2"
EPOCHS, LR, BATCH, MAX_LEN, SEED = 2, 2e-5, 8, 512, 0


def train(data: list[tuple[str, str, float]], out: Path, meta: dict | None = None) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn, rnd, data = torch.nn.BCEWithLogitsLoss(), random.Random(SEED), list(data)
    model.train()
    for epoch in range(EPOCHS):
        rnd.shuffle(data)
        total = 0.0
        for i in range(0, len(data), BATCH):
            q, e, y = zip(*data[i:i + BATCH])
            enc = tok(list(q), list(e), truncation="only_second", max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to(device)
            loss = loss_fn(model(**enc).logits.squeeze(-1), torch.tensor(y, dtype=torch.float, device=device))
            loss.backward()
            opt.step()
            opt.zero_grad()
            total += loss.item() * len(y)
        print(f"epoch {epoch + 1}/{EPOCHS}: loss {total / len(data):.4f} over {len(data)} pairs ({device})", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train.json").write_text(json.dumps({"base": BASE, "epochs": EPOCHS, "lr": LR, "batch": BATCH,
                                                "max_len": MAX_LEN, "seed": SEED, "pairs": len(data), "device": device,
                                                **(meta or {})}, indent=1))


def read_jsonl(path: Path) -> tuple[list[tuple[str, str, float]], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    head = rows[0] if rows and "meta" in rows[0] else {"meta": {}}
    return [(r["q"], r["e"], float(r["y"])) for r in rows if "meta" not in r], head["meta"]


if __name__ == "__main__":
    data, meta = read_jsonl(Path(sys.argv[1]))
    train(data, Path(sys.argv[2]), meta)
