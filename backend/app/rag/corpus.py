"""Loads the markdown corpus in data/ (CVs, JDs, policies) — front matter + body. Uploaded CVs live in
.runtime/cvs/ (already redacted by the NLP pipeline) and join the `cvs` folder transparently."""

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


def parse_md(raw: str, default_id: str) -> Doc:
    meta: dict = {}
    if raw.startswith("---"):
        _, front, raw = raw.split("---", 2)
        meta = yaml.safe_load(front) or {}
    body = raw.split("<!-- edge:")[0].strip()  # authoring notes are not part of the CV
    return Doc(id=str(meta.get("id", default_id)), text=body, meta=meta)


def _load(path: Path) -> Doc:
    return parse_md(path.read_text(encoding="utf-8"), path.stem)


def upload_dir() -> Path:
    return get_settings().runtime_dir / "cvs"


@lru_cache
def load_dir(name: str) -> tuple[Doc, ...]:
    folders = [get_settings().data_dir / name] + ([upload_dir()] if name == "cvs" else [])
    return tuple(_load(p) for f in folders if f.exists() for p in sorted(f.glob("*.md")))


def get_doc(name: str, doc_id: str) -> Doc | None:
    return next((d for d in load_dir(name) if d.id == doc_id), None)


@lru_cache
def load_yaml(name: str) -> dict:
    return yaml.safe_load((get_settings().data_dir / name).read_text(encoding="utf-8"))
