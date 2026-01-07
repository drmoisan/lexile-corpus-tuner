from __future__ import annotations

import pytest
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
    DownloadCandidate,
    ManifestEntry,
    generate_stable_slug,
)


def test_generate_stable_slug_converts_to_lowercase_hyphens() -> None:
    """Slugging should normalize case and underscores to hyphens."""
    result = generate_stable_slug("OpenStax_Book")
    assert result == "openstax-book"


def test_generate_stable_slug_is_idempotent() -> None:
    """Repeated slugging of the same identifier must remain stable."""
    first = generate_stable_slug("CK12-Algebra-1")
    second = generate_stable_slug(first)
    assert first == second


def test_catalog_entry_dataclass_validates_required_fields() -> None:
    """Dataclass instantiation without required fields should error."""
    with pytest.raises(TypeError):
        CatalogEntry(  # type: ignore[call-arg]
            source_id=None,
            identifier="missing-fields",
        )


def test_manifest_entry_dataclass_enforces_txt_extension() -> None:
    """ManifestEntry must reject filenames that are not .txt."""
    with pytest.raises(ValueError):
        ManifestEntry(
            source_id="openstax",
            id="openstax-book",
            url="https://example.com/book.html",
            filename="openstax-book.html",
        )


def test_download_candidate_round_trips_fields() -> None:
    """Ensure DownloadCandidate stores provided attributes."""
    candidate = DownloadCandidate(format="text/plain", url="http://x", size=42)
    assert candidate.format == "text/plain"
    assert candidate.url == "http://x"
    assert candidate.size == 42
