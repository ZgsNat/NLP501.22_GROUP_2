"""Generate notebooks/finetune_colab.ipynb from cv_agent/training/finetune.py and the modules it imports.

The notebook embeds the exact source files (%%writefile), so Colab runs the same code as
``python -m cv_agent finetune`` and needs neither this repository nor any local data. Re-run after editing
those modules; tests/test_finetune.py fails while the notebook is stale.

    python scripts/build_colab_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "finetune_colab.ipynb"
# every cv_agent module the training code imports, in package order
EMBEDDED = ("cv_agent/__init__.py", "cv_agent/data/__init__.py", "cv_agent/data/public.py",
            "cv_agent/eval/__init__.py", "cv_agent/eval/metrics.py", "cv_agent/training/__init__.py",
            "cv_agent/training/finetune.py")

INTRO = """# Fine-tune dense encoder & cross-encoder trên Colab (NLP501 · Nhóm 2)

Notebook này chạy phần **huấn luyện** của dự án trên GPU miễn phí của Colab. Không cần tải repo, cũng không cần dữ liệu trên máy:

| Mô hình | Gốc | Cách huấn luyện | Dùng trong |
|---|---|---|---|
| Dense encoder | `intfloat/multilingual-e5-base` | Contrastive (CachedMultipleNegativesRankingLoss), triplet (JD, Good Fit, No Fit), in-batch negative, hướng ConFit | B1, B2 và retrieval của P6 |
| Cross-encoder | `BAAI/bge-reranker-base` | Binary cross-entropy, nhãn No Fit / Potential / Good = 0 / 0.5 / 1 | B4, P6 (rerank) |

**Dữ liệu:** `cnamuangtoun/resume-job-description-fit` được tải thẳng từ Hugging Face. Không có CV nào trên máy nhóm được gửi lên (NFR-02).

**Chống rò rỉ:** bộ dữ liệu này cũng là tập kiểm tra thật của nhóm (`data/public`), nên:
1. Bỏ mọi JD QA / Automation Test (đây chính là các JD dùng để đánh giá).
2. Bỏ mọi CV từng được gán nhãn cho một JD như vậy.
3. Chia `train.csv` thành train / dev theo JD. `test.csv` (đã lọc như trên, bỏ cả JD đã gặp) chỉ dùng để kiểm tra cuối.

**Cách chạy:** chọn *Runtime → Change runtime type → T4 GPU*, rồi chọn *Runtime → Run all*. Ô cài đặt nâng cấp `sentence-transformers` / `transformers`. Nếu Colab báo cần khởi động lại, hoặc ô huấn luyện lỗi import, hãy chọn *Runtime → Restart session* rồi *Run all* lần nữa (lần hai không cài lại). Với T4, mỗi mô hình mất khoảng 10–20 phút cho 1 epoch. Ô cuối nén hai mô hình thành `rjf_finetuned_models.zip` để tải về.

> File này được sinh tự động từ `cv_agent/training/finetune.py` bằng `python scripts/build_colab_notebook.py`. Đừng sửa tay: hãy sửa module rồi sinh lại.
"""

SETUP = """import subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "Không có GPU: vẫn chạy được nhưng chậm.")
!pip -q install "sentence-transformers>=4.0" "datasets>=2.19" "accelerate>=0.26" pyyaml"""

PARAMS = """# Tham số huấn luyện. LIMIT = None: chạy đủ dữ liệu; đặt LIMIT = 64 để chạy thử vài phút.
TARGETS = ["dense", "rerank"]
EPOCHS = 1
BATCH_SIZE = 16
MAX_LEN = 256          # 512 tốt hơn một chút nhưng chậm gấp ~3 lần
LIMIT = None
OUT_DIR = "models"
RAW_DIR = "data/public/raw\""""

TRAIN = """import logging
from pathlib import Path
from cv_agent.training.finetune import DEFAULT_BASE, TrainConfig, run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
reports = {}
for target in TARGETS:
    base = DEFAULT_BASE[target]
    cfg = TrainConfig(target=target, base_model=base, out_dir=Path(OUT_DIR) / f"{base.split('/')[-1]}-rjf-ft",
                      epochs=EPOCHS, batch_size=BATCH_SIZE, max_len=MAX_LEN, limit=LIMIT)
    reports[target] = run(cfg, Path(RAW_DIR))"""

RESULTS = """import pandas as pd

rows = []
for target, rep in reports.items():
    for split in rep["after"]:
        rows.append({"mô hình": target, "tập": split, "số JD": rep["after"][split]["n_jds"],
                     "nDCG@10 trước": round(rep["before"][split]["ndcg@10"], 3),
                     "nDCG@10 sau": round(rep["after"][split]["ndcg@10"], 3),
                     "ví dụ train": rep["n_train_examples"], "thời gian (s)": rep["wall_s"]})
pd.DataFrame(rows)"""

EXPORT = """import shutil
shutil.rmtree(Path(OUT_DIR) / "multilingual-e5-base-rjf-ft" / "checkpoints", ignore_errors=True)
shutil.rmtree(Path(OUT_DIR) / "bge-reranker-base-rjf-ft" / "checkpoints", ignore_errors=True)
archive = shutil.make_archive("rjf_finetuned_models", "zip", OUT_DIR)
print(archive)
try:
    from google.colab import files
    files.download(archive)
except ImportError:
    pass  # not on Colab: the zip is next to the notebook"""

USE_LOCALLY = """## Dùng mô hình đã fine-tune trên máy

Giải nén `rjf_finetuned_models.zip` vào thư mục `models/` của dự án, rồi chạy lại thực nghiệm. Chỉ cần đổi mô hình, mọi cấu hình còn lại giữ nguyên:

```bash
python -m cv_agent experiments --neural --llm ollama --model qwen2.5:3b --configs B0,B1,B2,B3,B4,P6 \\
    --dense-model models/multilingual-e5-base-rjf-ft --rerank-model models/bge-reranker-base-rjf-ft
python -m cv_agent experiments --data data/public --neural --llm ollama --model qwen2.5:3b --configs B0,B1,B2,B3,B4,P6 \\
    --dense-model models/multilingual-e5-base-rjf-ft --rerank-model models/bge-reranker-base-rjf-ft
```

`finetune_metrics.json` trong mỗi thư mục mô hình ghi lại cấu hình, số cặp train / dev / test và nDCG@10 trước / sau.
"""


def _lines(text: str) -> List[str]:
    lines = text.splitlines(keepends=True)
    return lines[:-1] + [lines[-1].rstrip("\n")] if lines else []


def _md(text: str) -> Dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> Dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(text)}


def build_notebook() -> Dict:
    dirs = sorted({str(Path(p).parent.as_posix()) for p in EMBEDDED})
    cells = [_md(INTRO), _code(SETUP),
             _md("## Mã nguồn dự án dùng cho huấn luyện\nCác ô dưới đây ghi nguyên văn các module của repo."),
             _code("import os\nfor d in " + json.dumps(dirs) + ":\n    os.makedirs(d, exist_ok=True)")]
    for rel in EMBEDDED:
        source = (ROOT / rel).read_text(encoding="utf-8")
        cells.append(_code(f"%%writefile {rel}\n{source}"))
    cells += [_md("## Tham số"), _code(PARAMS), _md("## Huấn luyện và đánh giá trước / sau"), _code(TRAIN),
              _code(RESULTS), _md("## Tải mô hình về"), _code(EXPORT), _md(USE_LOCALLY)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}}}


def main() -> None:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK}")


if __name__ == "__main__":
    main()
