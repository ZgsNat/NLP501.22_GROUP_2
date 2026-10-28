"""Config loading: base.yaml + experiment overrides (FR-25, NFR-01)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
EXPERIMENTS = ("B0", "B1", "B2", "B3", "B4", "LLM-all", "P6", "A1", "A2")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(experiment: str = "P6", base_path: Path = CONFIG_DIR / "base.yaml",
                experiments_path: Path = CONFIG_DIR / "experiments.yaml",
                overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    base = yaml.safe_load(Path(base_path).read_text(encoding="utf-8"))
    exps = yaml.safe_load(Path(experiments_path).read_text(encoding="utf-8"))
    if experiment not in exps:
        raise KeyError(f"Unknown experiment {experiment!r}; available: {sorted(exps)}")
    cfg = deep_merge(base, exps[experiment] or {})
    cfg = deep_merge(cfg, overrides or {})
    cfg["experiment"] = experiment
    return cfg


def list_experiments(experiments_path: Path = CONFIG_DIR / "experiments.yaml") -> List[str]:
    return list(yaml.safe_load(Path(experiments_path).read_text(encoding="utf-8")))
