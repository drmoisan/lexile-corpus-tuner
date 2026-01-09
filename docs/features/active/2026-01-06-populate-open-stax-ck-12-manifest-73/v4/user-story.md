# `2026-01-06-populate-open-stax-ck-12-manifest-73` — User Story

- **Status:** Active
- **Outcome:** User story refined to accurately reflect CK-12 reader-based HTML/JSON ingestion while preserving the original goal of automated, text-aligned OER manifest generation.
- **Root Cause:** Empirical validation of CK-12 delivery mechanisms (FlexBook reader APIs) required updating the story to remove PDF dependencies and align with the actual extraction path.
- Issue: #73
- Owner: drmoisan
- Status: In Progress
- Last Updated: 2026-01-06

## Story Statement

- As a corpus pipeline maintainer, I want to generate a curated manifest of OpenStax (IA text derivatives) and CK-12 (reader HTML/JSON) titles automatically, so that the OER downloader can stay aligned with our text-only normalization path without manual errors.
- As a pipeline operator, I want to download text-ready OER assets with stable slugs and filenames, so that `lexile-scoring-model-pipeline corpus download` and `normalize` succeed end-to-end without format conversions or unavailable PDF dependencies.

## Problem / Why

Manual manifest editing is slow, error-prone, and mismatched to the active pipeline schema (`source_id`, `id`, `url`, `filename`). The normalizer only ingests `.txt`/`.jsonl`, so PDF/EPUB links silently fail. CK-12 currently exposes HTML/JSON reader payloads (not PDFs) in both FlexBook 1.0/2.0 flows. We need a repeatable way to discover, filter, and register IA-hosted OpenStax text derivatives and CK-12 reader content, ensuring downloads and normalization work without ad-hoc fixes.

## Personas & Scenarios

- Persona: Data engineer / corpus curator
  - Maintains the Lexile scoring corpus and OER feeds
  - Cares about schema correctness, text-only compatibility, and predictable downloads
  - Constrained by existing pipeline (no PDF/EPUB ingestion) and limited time for manual curation
  - Wants a deterministic, scriptable path from catalog to manifest to download
  - Motivated to expand K-12 coverage (OpenStax, CK-12) with minimal rework
- Scenario: The curator runs the catalog builder to pull OpenStax metadata from IA (with `_djvu.txt` candidates) and CK-12 metadata from the native FlexBook catalog. CK-12 enrichment resolves reader API endpoints and records HTML/JSON payload URLs (no PDFs). After curation, the manifest `oer_sources.json` contains stable slugs, direct text/HTML URLs, and filenames suitable for downstream extraction. They then run `lexile-scoring-model-pipeline corpus download` and `normalize`; downloads land under the correct raw subfolders, an extractor renders CK-12 HTML/JSON to `.txt`, and normalization consumes the files without format errors.


## Acceptance Criteria

- [ ] Catalog builder writes IA search results for OpenStax and CK-12 native catalog rows to `data/meta/catalogs/*.jsonl` with fields: `source_id`, identifiers/slugs, `title`, `creator`, `year/last_modified` (if available), `language`, `license_url`, and `download_candidates` that include `_djvu.txt` (OpenStax) or reader HTML/JSON endpoints (CK-12).
- [ ] Curation step filters catalog rows to entries with at least one text/HTML candidate; rows without retrievable content are skipped with a recorded reason.
- [ ] Manifest generator emits `data/meta/oer_sources.json` entries containing `source_id`, stable slug `id`, direct `url`, and appropriate `filename` extensions (`.txt` for OpenStax text, `.html`/`.json` for CK-12 reader payloads); all URLs return HTTP 200 and the expected content-type (text/* for OpenStax, text/html or application/json for CK-12).
- [ ] `lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"` retrieves the manifest entries into the correct raw subfolders with the expected filenames.
- [ ] A CK-12 extractor renders downloaded reader payloads to `.txt`; `lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"` ingests the resulting `.txt` files without empty outputs or format errors.

## Non-Goals

This effort does not add PDF/EPUB-to-text conversion, does not expand beyond OpenStax/CK-12 sources, and does not change normalization logic beyond consuming text assets registered in the manifest.
