from __future__ import annotations

import pytest

from lexile_corpus_tuner.config import OpenAISettings
from lexile_corpus_tuner.llm import openai_client as oa_client
from lexile_corpus_tuner.rewriting import OpenAIRewriter, RewriteRequest


def test_openai_rewriter_builds_prompts():
    """OpenAIRewriter injects metadata into prompts passed to the client."""
    captured: dict[str, str] = {}

    class DummyClient:
        def rewrite(self, *, system_prompt: str, user_prompt: str, metadata):
            captured["system"] = system_prompt
            captured["user"] = user_prompt
            captured["doc_id"] = metadata.doc_id
            return "Simpler output"

    rewriter = OpenAIRewriter(DummyClient())
    request = RewriteRequest(
        doc_id="demo-doc",
        window_id=7,
        text="Original paragraph with technical jargon.",
        target_lexile=320,
        metadata={"max_window_lexile": 400, "avg_tolerance": 25},
    )
    output = rewriter.rewrite(request)

    assert output == "Simpler output"
    assert "Document ID: demo-doc" in captured["user"]
    assert "Window ID: 7" in captured["user"]
    assert "max window" in captured["user"].lower()
    assert captured["doc_id"] == "demo-doc"
    assert "You rewrite English passages" in captured["system"]


def test_openai_rewrite_client_requires_api_key(monkeypatch):
    """Client constructor validates that an API key is provided."""
    monkeypatch.setattr(oa_client, "OpenAI", object())
    settings = OpenAISettings(enabled=True)
    with pytest.raises(ValueError):
        oa_client.OpenAIRewriteClient(settings, api_key="")


def test_openai_rewrite_client_retries_then_succeeds(monkeypatch):
    """Client retries failed requests and returns the first successful output."""
    attempts = {"count": 0}

    class DummySegment:
        def __init__(self, text: str) -> None:
            self.text = text

    class DummyOutput:
        def __init__(self, text: str) -> None:
            self.content = [DummySegment(text)]

    class DummyResponse:
        def __init__(self, text: str) -> None:
            self.output = [DummyOutput(text)]

    class DummyResponses:
        def create(self, **_: object):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("transient error")
            return DummyResponse("Rewritten content")

    class DummyOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = DummyResponses()

    monkeypatch.setattr(oa_client, "OpenAI", DummyOpenAI)
    settings = OpenAISettings(enabled=True)
    client = oa_client.OpenAIRewriteClient(settings, api_key="token")
    metadata = oa_client.RewriteMetadata(doc_id="demo", window_id=1)

    result = client.rewrite(
        system_prompt="system",
        user_prompt="Original text",
        metadata=metadata,
    )
    assert result == "Rewritten content"
    assert attempts["count"] == 2
