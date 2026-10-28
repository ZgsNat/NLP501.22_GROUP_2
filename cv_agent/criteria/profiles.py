"""Criteria Profiles (FR-08): stored as YAML, weights over the 8 axes, Σw = 1."""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import yaml

from cv_agent.schema import AXES, CriteriaProfile

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "profiles"


def load_profile(path: Path) -> CriteriaProfile:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    weights = {k: float(v) for k, v in data["weights"].items()}
    profile = CriteriaProfile(name=data["name"], weights=weights, description=data.get("description", ""))
    profile.validate()
    return profile


def list_profiles(directory: Path = DEFAULT_DIR) -> Dict[str, CriteriaProfile]:
    return {p.name: p for p in (load_profile(f) for f in sorted(Path(directory).glob("*.yaml")))}


def profile_filename(name: str) -> str:
    """Safe file name for an HR-typed profile name (no path separators or traversal)."""
    slug = re.sub(r"[^\w-]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError("profile name must contain at least one letter or digit")
    return f"{slug}.yaml"


def save_profile(profile: CriteriaProfile, directory: Path = DEFAULT_DIR, overwrite: bool = False) -> Path:
    profile.validate()
    path = Path(directory) / profile_filename(profile.name)
    if not overwrite and (path.exists() or profile.name in list_profiles(directory)):
        raise FileExistsError(f"profile '{profile.name}' already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": profile.name, "description": profile.description,
                                    "weights": profile.weights}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Return a new dict over all AXES summing to 1 (used when HR edits sliders)."""
    full = {axis: max(0.0, float(weights.get(axis, 0.0))) for axis in AXES}
    total = sum(full.values())
    if total <= 0:
        return {axis: 1.0 / len(AXES) for axis in AXES}
    return {axis: w / total for axis, w in full.items()}


def with_weights(profile: CriteriaProfile, weights: Dict[str, float]) -> CriteriaProfile:
    return replace(profile, weights=normalize_weights(weights), approved=False)


def uniform_profile() -> CriteriaProfile:
    """A2 ablation: no criteria profile -> uniform weights."""
    return CriteriaProfile("Uniform", {axis: 1.0 / len(AXES) for axis in AXES}, "No profile (ablation A2)")


def profile_names(directory: Path = DEFAULT_DIR) -> List[str]:
    return list(list_profiles(directory))
