from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .summary import Summary


class CheckpointStore(Protocol):
    """Checkpoint protocol so resumability can use files or other stores."""

    def load(self) -> int: ...

    def save(self, index: int, summary: Summary) -> None: ...
