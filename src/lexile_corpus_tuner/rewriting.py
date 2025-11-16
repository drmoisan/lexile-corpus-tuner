from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from .llm.openai_client import OpenAIRewriteClient, RewriteMetadata
from .models import ConstraintViolation

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You rewrite English passages so that 10-year-old students (Lexile ≈ {target}) "
    "can understand them.\n"
    "Your responsibilities:\n"
    "- Keep every factual detail, name, and event.\n"
    "- Maintain the same paragraph boundaries and roughly the same length "
    "(±10% tokens).\n"
    "- Simplify vocabulary and sentence structure while keeping the tone age appropriate.\n"
    "- Output plain text only (no Markdown, no quotes, no introductions or commentary).\n"
    "- Never invent new facts or remove safety-critical information."
)

USER_PROMPT_TEMPLATE = (
    "Document ID: {doc_id}\n"
    "Window ID: {window_id}\n"
    "Original Lexile: {source_lexile}\n"
    "Target Lexile: {target}\n"
    "Max Window Lexile: {max_window}\n"
    "Avg Target ± Tolerance: {target} ± {avg_tolerance}\n"
    "Violation Reason: {reason}\n"
    "Paragraphs: {paragraphs}\n"
    "Approximate Tokens: {token_count}\n"
    "\n"
    "Rewrite the following text so it fits the constraints above while preserving "
    "facts and paragraph breaks. Keep the same number of paragraphs separated by "
    "blank lines.\n"
    "-----\n"
    "{text}\n"
    "-----\n"
    "Return only the rewritten passage."
)


@dataclass(slots=True)
class RewriteRequest:
    """Unit of text that must be rewritten."""

    doc_id: str
    window_id: int
    text: str
    target_lexile: float
    violation: ConstraintViolation | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Rewriter(ABC):
    """Abstract interface for rewriting text toward a target Lexile."""

    @abstractmethod
    def rewrite(self, request: RewriteRequest) -> str:
        """Return rewritten text."""
        raise NotImplementedError


class NoOpRewriter(Rewriter):
    """Returns the original text unchanged."""

    def rewrite(self, request: RewriteRequest) -> str:
        return request.text


class CallableRewriter(Rewriter):
    """Adapt an arbitrary callable into the Rewriter interface."""

    def __init__(self, func: Callable[[RewriteRequest], str]) -> None:
        self._func = func

    def rewrite(self, request: RewriteRequest) -> str:
        return self._func(request)


class OpenAIRewriter(Rewriter):
    """Rewriter implementation backed by the OpenAI Responses API."""

    def __init__(
        self,
        client: OpenAIRewriteClient,
        *,
        system_prompt_template: str = SYSTEM_PROMPT,
        user_prompt_template: str = USER_PROMPT_TEMPLATE,
    ) -> None:
        self._client = client
        self._system_prompt_template = system_prompt_template
        self._user_prompt_template = user_prompt_template

    def rewrite(self, request: RewriteRequest) -> str:
        violation = request.violation
        metadata = request.metadata or {}
        paragraphs = [
            paragraph.strip()
            for paragraph in request.text.strip().split("\n\n")
            if paragraph.strip()
        ]
        paragraph_count = len(paragraphs) or 1
        token_count = max(1, len(request.text.split()))
        max_window = metadata.get("max_window_lexile", "unknown")
        avg_tolerance = metadata.get("avg_tolerance", "unknown")
        reason = (
            violation.reason if violation else metadata.get("reason", "unspecified")
        )
        system_prompt = self._system_prompt_template.format(
            target=int(request.target_lexile)
        )
        user_prompt = self._user_prompt_template.format(
            doc_id=request.doc_id,
            window_id=request.window_id,
            source_lexile=(
                f"{violation.lexile:.1f}"
                if violation
                else metadata.get("source_lexile", "n/a")
            ),
            target=int(request.target_lexile),
            max_window=max_window,
            avg_tolerance=avg_tolerance,
            reason=reason,
            paragraphs=paragraph_count,
            token_count=token_count,
            text=request.text.strip(),
        )
        rewrite_metadata = RewriteMetadata(
            doc_id=request.doc_id,
            window_id=request.window_id,
            source_lexile=violation.lexile if violation else None,
            target_lexile=request.target_lexile,
            reason=reason,
            token_count=token_count,
        )
        logger.info(
            "Rewriting doc=%s window=%s lexile=%s→%.1f",
            request.doc_id,
            request.window_id,
            f"{violation.lexile:.1f}" if violation else "unknown",
            request.target_lexile,
        )
        rewritten = self._client.rewrite(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=rewrite_metadata,
        )
        return rewritten.strip()
