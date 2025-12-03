"""Entry point for running gutenberg_query_builder_ui as a module.

This allows execution via:
    python -m scripts.production.gutenberg_query_builder_ui

Or from the scripts/production directory:
    python -m gutenberg_query_builder_ui
"""

from __future__ import annotations

from . import main

if __name__ == "__main__":
    main()
