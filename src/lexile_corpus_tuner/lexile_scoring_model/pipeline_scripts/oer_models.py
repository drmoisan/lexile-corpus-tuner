"""
Data models and helpers for OER catalog/manifest workflows.

Purpose:
    Provide strongly-typed value objects for catalog entries, manifest entries,
    and download candidates so downstream pipeline stages share a consistent
    contract.

Usage:
    Imported by catalog, enrichment, curation, manifest, and UI modules. These
    dataclasses are designed to be JSON-serializable via asdict or manual
    dictionaries built in each stage.

Flow:
    - Catalog stage builds `CatalogEntry` with basic metadata and a list of
      `DownloadCandidate` items.
    - Curation stage filters CatalogEntry objects.
    - Manifest stage converts curated entries into `ManifestEntry` objects.

Invariants / Constraints:
    - `id` fields are stable slugs derived from immutable IA identifiers.
    - Manifest filenames must end with `.txt` (text assets) or `.pdf` (CK-12
      download artifacts) to align with downstream processing.
    - Download candidates should represent real URLs and declared formats.

Side Effects:
    None. Pure value objects and helpers only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _empty_download_candidates() -> list[DownloadCandidate]:
    """Return a typed empty list for download candidates."""
    return []


def generate_stable_slug(ia_identifier: str) -> str:
    """
    Convert an Internet Archive identifier into a lowercase, hyphenated slug.

    Purpose:
        Ensure manifest `id` values remain stable and idempotent regardless of
        formatting in source metadata.

    Args:
        ia_identifier: The immutable IA identifier to normalize.

    Returns:
        A lowercase slug with non-alphanumeric characters replaced by hyphens.

    Raises:
        ValueError: If ia_identifier is empty after stripping.
    """
    cleaned = ia_identifier.strip().lower()
    if not cleaned:
        raise ValueError("IA identifier cannot be empty")
    # Replace non-alphanumeric runs with single hyphens to ensure a stable slug.
    slug_parts: list[str] = []
    current_run: list[str] = []
    # Walk characters to preserve word boundaries while stripping punctuation.
    for char in cleaned:
        if char.isalnum():
            current_run.append(char)
        else:
            if current_run:
                slug_parts.append("".join(current_run))
                current_run = []
            # Insert a delimiter so punctuation still yields distinct words.
            slug_parts.append("-")
    if current_run:
        slug_parts.append("".join(current_run))
    slug = "".join(slug_parts).strip("-")
    # Collapse repeated hyphens that can emerge from adjacent delimiters.
    while "--" in slug:
        slug = slug.replace("--", "-")
    if not slug:
        raise ValueError("IA identifier slug is empty after normalization")
    return slug


@dataclass(frozen=True)
class DownloadCandidate:
    """
    Describes a single downloadable file for an OER item.

    Attributes:
        format: The declared MIME-like format for the file (e.g., text/plain).
        url: Fully-qualified URL to download the file.
        size: Optional size in bytes if available from metadata.
    """

    format: str
    url: str
    size: int | None = None


@dataclass(frozen=True)
class CatalogEntry:
    """
    Represents a single record from the catalog stage.

    Attributes:
        source_id: Logical source bucket (openstax or ck12) when known.
        identifier: Immutable upstream identifier (e.g., IA identifier).
        title: Human-readable title.
        creator: Primary creator/author if present.
        year: Publication year when available.
        language: Language code(s) associated with the item.
        license_url: Upstream license reference if provided.
        download_candidates: Possible downloads discovered for the item.
        artifact_type: CK-12 artifact type such as "flexbook" when available.
        handle: Canonical CK-12 handle used for Perma/Revision API calls.
        artifact_id: Numeric artifact identifier from the CK-12 Browse API.
    """

    source_id: str | None
    identifier: str
    title: str | None
    creator: str | None
    year: str | None
    language: list[str]
    license_url: str | None
    download_candidates: list[DownloadCandidate] = field(
        default_factory=_empty_download_candidates
    )
    artifact_type: str | None = None
    handle: str | None = None
    artifact_id: int | None = None


@dataclass(frozen=True)
class ManifestEntry:
    """
    Represents a curated manifest row consumable by the downloader.

    Attributes:
        source_id: Required source bucket (openstax or ck12).
        id: Stable slug derived from the immutable identifier.
        url: Direct download URL for the text asset.
        filename: Target filename; must end with .txt (text assets), .json
            (CK-12 revision payloads), or .pdf (legacy PDF downloads).
    """

    source_id: str
    id: str
    url: str
    filename: str

    def __post_init__(self) -> None:  # type: ignore[override]
        # Only allow text, JSON (CK-12 revision payloads), or PDF derivatives to
        # keep the downloader/normalizer contracts strict.
        allowed_suffixes = (".txt", ".pdf", ".json")
        if not self.filename.lower().endswith(allowed_suffixes):
            raise ValueError(
                "ManifestEntry.filename must end with .txt, .pdf, or .json"
            )
