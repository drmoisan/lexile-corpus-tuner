from __future__ import annotations

import importlib
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, cast

from ..config import OpenAISettings

logger = logging.getLogger(__name__)

OpenAI: Callable[..., Any] | None = None


@dataclass(slots=True)
class RewriteMetadata:
    """Metadata about the window being rewritten, used for logging."""

    doc_id: str
    window_id: int
    source_lexile: float | None = None
    target_lexile: float | None = None
    reason: str | None = None
    token_count: int | None = None


class OpenAIRewriteClient:
    """Thin wrapper around the OpenAI Responses API with retries and throttling."""

    def __init__(self, settings: OpenAISettings, api_key: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required when rewriting is enabled.")
        self._settings = settings
        self._api_key = api_key
        self._client_factory: Callable[..., Any] = _load_openai_factory()
        self._client: Any | None = None
        self._semaphore: threading.BoundedSemaphore | None = None
        if settings.parallel_requests > 0:
            self._semaphore = threading.BoundedSemaphore(settings.parallel_requests)
        self._max_attempts = 3

    @property
    def settings(self) -> OpenAISettings:
        return self._settings

    def rewrite(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        metadata: RewriteMetadata,
    ) -> str:
        """Send the rewrite request and return the LLM output."""
        attempt = 0
        last_error: Exception | None = None
        while attempt < self._max_attempts:
            attempt += 1
            try:
                with self._acquire_slot():
                    client = self._ensure_client()
                    response: Any = client.responses.create(
                        model=self._settings.model,
                        input=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self._settings.temperature,
                        max_output_tokens=self._settings.max_output_tokens,
                        top_p=self._settings.top_p,
                        timeout=self._settings.request_timeout,
                    )
                text = self._extract_text(response)
                logger.debug(
                    "OpenAI rewrite succeeded for doc=%s window=%s len=%s tokens",
                    metadata.doc_id,
                    metadata.window_id,
                    metadata.token_count,
                )
                return text
            except Exception as exc:  # pragma: no cover - network-related
                last_error = exc
                logger.warning(
                    "OpenAI rewrite failed for doc=%s window=%s (attempt %s/%s): %s",
                    metadata.doc_id,
                    metadata.window_id,
                    attempt,
                    self._max_attempts,
                    exc,
                )
                if attempt >= self._max_attempts:
                    break
                time.sleep(min(2 ** (attempt - 1), 5))
        raise RuntimeError("OpenAI rewrite failed after retries.") from last_error

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(
                api_key=self._api_key,
                base_url=self._settings.base_url,
                organization=self._settings.organization,
            )
        return self._client

    @contextmanager
    def _acquire_slot(self) -> Iterator[None]:
        if self._semaphore is None:
            yield
            return
        self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()

    @staticmethod
    def _extract_text(response: Any) -> str:
        output = getattr(response, "output", None)
        if not output:
            raise RuntimeError("OpenAI response is missing output content.")
        first = OpenAIRewriteClient._materialize_item(output[0])
        content = first.get("content")
        if not content:
            raise RuntimeError("OpenAI response has no content segments.")
        segment = OpenAIRewriteClient._materialize_item(content[0])
        text = segment.get("text")
        if not text:
            raise RuntimeError("OpenAI response segment missing text.")
        return text

    @staticmethod
    def _materialize_item(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return cast(dict[str, Any], item)
        if hasattr(item, "model_dump"):
            dumpable: Any = item
            raw_dump: dict[str, Any] = dumpable.model_dump()
            return raw_dump
        if hasattr(item, "__dict__"):
            dumpable: Any = item
            raw_dict: dict[str, Any] = dict(dumpable.__dict__)
            return raw_dict
        raise RuntimeError("Unexpected OpenAI response format.")


def _load_openai_factory() -> Callable[..., Any]:
    """Dynamically import the OpenAI client factory to avoid hard dependency at import."""
    global OpenAI
    if OpenAI is not None:
        return OpenAI
    try:  # pragma: no cover - import guard
        module = importlib.import_module("openai")
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError(
            "openai package is not installed. Install extras via 'pip install .[llm-openai]'."
        ) from exc
    openai_cls = getattr(module, "OpenAI", None)
    if openai_cls is None:  # pragma: no cover - defensive
        raise RuntimeError(
            "openai.OpenAI client class is unavailable in this environment."
        )
    OpenAI = cast(Callable[..., Any], openai_cls)
    return OpenAI
