"""Minimal example showing how to call the OpenAI-backed rewriter directly."""

from __future__ import annotations

import os
from pathlib import Path

from lexile_corpus_tuner.config import load_config
from lexile_corpus_tuner.llm import OpenAIRewriteClient
from lexile_corpus_tuner.rewriting import OpenAIRewriter, RewriteRequest


def main() -> None:
    config = load_config(Path(__file__).with_name("example_config.yaml"))
    config.rewrite_enabled = True
    config.openai.enabled = True
    api_key = (
        config.openai.api_key
        or os.environ.get(config.openai.api_key_env or "OPENAI_API_KEY")
        or ""
    )
    if not api_key:
        raise RuntimeError(
            "Set the OpenAI API key before running this example "
            f"({config.openai.api_key_env})."
        )

    client = OpenAIRewriteClient(config.openai, api_key=api_key)
    rewriter = OpenAIRewriter(client)

    sample_text = (
        "The mechanical manual described gyroscopic stabilization in dense jargon. "
        "Rewrite it so that a ten-year-old reader can understand how the device "
        "keeps a platform steady even when the base wobbles."
    )
    request = RewriteRequest(
        doc_id="demo",
        window_id=1,
        text=sample_text,
        target_lexile=config.target_avg_lexile,
    )
    rewritten = rewriter.rewrite(request)
    print("Original:\n", sample_text)
    print("\nRewritten:\n", rewritten)


if __name__ == "__main__":
    main()
