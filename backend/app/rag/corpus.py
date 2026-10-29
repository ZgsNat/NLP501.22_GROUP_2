"""Loads the markdown corpus in data/ (CVs, JDs, policies) — front matter + body."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from app.config import get_settings


@dataclass(frozen=True)
class Doc:
    id: str
    text: str
    meta: dict


def _load(path: Path) -> Doc:
    raw = path.read_text(encoding="utf-8")
    meta: dict = {}
    if raw.startswith("---"):
        _, front, raw = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
    body = raw.split("<!-- edge:")[0].strip()  # authoring notes are not part of the CV
    return Doc(id=str(meta.get("id", path.stem)), text=body, meta=meta)


@lru_cache
def load_dir(name: str) -> tuple[Doc, ...]:
    folder = get_settings().data_dir / name
    return tuple(_load(p) for p in sorted(folder.glob("*.md")))


def get_doc(name: str, doc_id: str) -> Doc | None:
    return next((d for d in load_dir(name) if d.id == doc_id), None)


@lru_cache
def load_yaml(name: str) -> dict:
    return yaml.safe_load((get_settings().data_dir / name).read_text(encoding="utf-8"))
