"""Long-term agent memory (JSON file): what HR already told the agent, reused in later runs.

* Clarification answers, keyed by the JD text hash — an answer is only reused for the *same* JD,
  never guessed across different postings.
* Calibrated Criteria weights per profile (FR-30), saved when HR accepts the calibration.

Only masked, non-personal data is stored (question ids, option indexes, axis weights).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from cv_agent.schema import AXES, JD


def jd_key(jd: JD) -> str:
    return hashlib.sha1(jd.text.strip().encode("utf-8")).hexdigest()[:16]


class AgentMemory:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"answers": {}, "weights": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"agent memory {self.path} is unreadable: {exc}") from exc
        return {"answers": dict(data.get("answers", {})), "weights": dict(data.get("weights", {}))}

    def _save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def recall_answers(self, jd: JD) -> Dict[str, int]:
        return dict(self._load()["answers"].get(jd_key(jd), {}).get("answers", {}))

    def remember_answers(self, jd: JD, answers: Dict[str, int]) -> None:
        if not answers:
            return
        data = self._load()
        known = data["answers"].get(jd_key(jd), {}).get("answers", {})
        data["answers"] = {**data["answers"], jd_key(jd): {"answers": {**known, **{k: int(v) for k, v in answers.items()}},
                                                          "updated": time.time()}}
        self._save(data)

    def recall_weights(self, profile_name: str) -> Optional[Dict[str, float]]:
        entry = self._load()["weights"].get(profile_name)
        return dict(entry["weights"]) if entry else None

    def remember_weights(self, profile_name: str, weights: Dict[str, float], n_pairs: int) -> None:
        unknown = set(weights) - set(AXES)
        total = sum(weights.values())
        if unknown or abs(total - 1.0) > 1e-6 or any(w < 0 for w in weights.values()):
            raise ValueError(f"invalid weights for memory: unknown={sorted(unknown)}, sum={total:.4f}")
        data = self._load()
        data["weights"] = {**data["weights"], profile_name: {"weights": dict(weights), "n_pairs": int(n_pairs),
                                                             "updated": time.time()}}
        self._save(data)
