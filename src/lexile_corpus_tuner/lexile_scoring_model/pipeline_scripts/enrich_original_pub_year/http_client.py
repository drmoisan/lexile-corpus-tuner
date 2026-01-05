from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from requests import Response


class HttpClient(Protocol):
    def get(self, url: str, *, params: dict[str, str], timeout: float) -> Response: ...
