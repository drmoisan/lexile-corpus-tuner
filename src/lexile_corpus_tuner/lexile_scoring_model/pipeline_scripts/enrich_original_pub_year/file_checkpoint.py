from __future__ import annotations

import json
from typing import TYPE_CHECKING

from .checkpoint_store import CheckpointStore

if TYPE_CHECKING:
    from .summary import Summary

if TYPE_CHECKING:
    from pathlib import Path


class FileCheckpoint(CheckpointStore):
    """
    Filesystem implementation of `CheckpointStore` for resumable enrichment runs.

    Purpose:
        Keep track of progress so reruns can skip already-processed rows.

    Usage:
        Construct with a checkpoint path; use `load` before processing and `save`
        periodically during enrichment.

    Side Effects:
        Performs filesystem I/O and creates parent directories when needed.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        """
        Initialize the checkpoint writer/reader for a specific file path.

        Purpose:
            Capture the target location for storing and retrieving checkpoint data.

        Args:
            checkpoint_path (Path): File where progress markers will be stored.

        Raises:
            None

        Side Effects:
            None.
        """

        self._checkpoint_path = checkpoint_path

    def load(self) -> int:
        """
        Read the last saved index from disk, tolerating missing or invalid files.

        Purpose:
            Resume processing at the correct offset after interruptions.

        Returns:
            int: Next index to process; zero when no valid checkpoint exists.

        Raises:
            OSError: When the checkpoint file cannot be opened.
            json.JSONDecodeError: When stored checkpoint content is malformed.

        Side Effects:
            Performs disk reads; ignores malformed JSON to favor forward progress.
        """

        if not self._checkpoint_path.exists():
            return 0
        try:
            with self._checkpoint_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return int(data.get("last_index", 0))
        except Exception:
            return 0

    def save(self, index: int, summary: Summary) -> None:
        """
        Persist the latest processed index and summary to disk.

        Purpose:
            Commit progress so subsequent runs can continue without duplicating work.

        Args:
            index (int): Row index that should be the starting point next run.
            summary (Summary): Aggregated metrics to include in the checkpoint.

        Raises:
            OSError: When the checkpoint file cannot be written.
            TypeError: When the payload cannot be serialized to JSON.

        Side Effects:
            Creates parent directories as needed and writes JSON to disk.
        """

        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"last_index": index, "summary": summary.to_dict()}
        with self._checkpoint_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
