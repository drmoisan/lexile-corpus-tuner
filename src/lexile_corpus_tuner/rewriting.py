from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


PROMPT_TEMPLATE = (
    "You are simplifying text for a 10-year-old reader with Lexile ≈ {target}.\n"
    "Rewrite the passage so vocabulary and sentence structure are simpler, but\n"
    "keep all factual information and names the same. Do not shorten the passage\n"
    "significantly; keep the same approximate length.\n"
    "Text: \"\"\"{text}\"\"\""
)


class Rewriter(ABC):
    """Abstract interface for rewriting text toward a target Lexile."""

    @abstractmethod
    def rewrite(self, text: str, target_lexile: float) -> str:
        """Return rewritten text."""
        raise NotImplementedError


class NoOpRewriter(Rewriter):
    """Returns the original text unchanged."""

    def rewrite(self, text: str, target_lexile: float) -> str:
        return text


class LLMRewriter(Rewriter):
    """
    Stub rewriter that relies on an injected API client/callback for the actual LLM call.

    Users should pass a callable via `completion_fn` that accepts `(prompt: str) -> str`
    or `(text: str, target: float) -> str` and handles API authentication.
    """

    def __init__(
        self,
        model_name: str,
        completion_fn: Callable[[str, str, float], str] | None = None,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._completion_fn = completion_fn
        self._client = client

    def rewrite(self, text: str, target_lexile: float) -> str:
        """
        Build a prompt and send it to the injected completion function.
        TODO: Integrate with a real LLM client (OpenAI, Azure, etc.) as desired.
        """
        if self._completion_fn is None:
            raise RuntimeError(
                "LLMRewriter requires a completion_fn callable to perform rewrites."
            )
        prompt = PROMPT_TEMPLATE.format(text=text, target=int(target_lexile))
        return self._completion_fn(prompt, text, target_lexile)
