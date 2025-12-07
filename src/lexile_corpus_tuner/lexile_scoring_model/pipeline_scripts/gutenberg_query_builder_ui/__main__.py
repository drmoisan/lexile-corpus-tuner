"""Entry point for running gutenberg_query_builder_ui as a module.

This allows execution via:
    python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.
        gutenberg_query_builder_ui

Or from the package directory:
    python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.
        gutenberg_query_builder_ui
"""

from __future__ import annotations

from . import main

if __name__ == "__main__":
    main()
