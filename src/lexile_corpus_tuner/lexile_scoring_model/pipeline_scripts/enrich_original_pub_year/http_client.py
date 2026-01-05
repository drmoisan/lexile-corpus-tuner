from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from requests import Response


class HttpClient(Protocol):
    """
    Minimal HTTP client abstraction so Open Library calls can be mocked or swapped.

    Purpose:
        Allow enrichment to depend on a small subset of HTTP behavior without
        coupling to `requests.Session` directly.

    Side Effects:
        Implementations are expected to perform network I/O.
    """

    def get(self, url: str, *, params: dict[str, str], timeout: float) -> Response:
        """
        Issue an HTTP GET request with query parameters and timeout control.

        Purpose:
            Provide the minimal interface required by `OpenLibraryClient` for HTTP
            requests.

        Args:
            url (str): Target URL for the request.
            params (dict[str, str]): Query string parameters to include.
            timeout (float): Timeout in seconds for the request.

        Returns:
            Response: HTTP response object with status and content.

        Raises:
            Exception: Implementations may surface connection or HTTP errors.

        Side Effects:
            Performs network I/O.
        """

        ...
