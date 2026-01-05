from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .checkpoint_store import CheckpointStore

if TYPE_CHECKING:
    from .summary import Summary

if TYPE_CHECKING:
    from pathlib import Path


class FileCheckpoint(CheckpointStore):
    """Persist progress to disk so long runs can resume without duplicating work."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._checkpoint_path = checkpoint_path

    def load(self) -> int:
        if not self._checkpoint_path.exists():
            return 0
        try:
            with self._checkpoint_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return int(data.get("last_index", 0))
        except Exception:
            return 0

    def save(self, index: int, summary: Summary) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_index": index, "summary": summary.to_dict()}
        with self._checkpoint_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
