# `2026-01-06-populate-open-stax-ck-12-manifest-73` — User Story

- Issue: #73
- Owner: drmoisan
- Status: In Progress
- Last Updated: 2026-01-06

## Story Statement

- As a corpus pipeline maintainer, I want to generate a curated manifest of OpenStax and CK-12 titles automatically, so that the OER downloader can stay aligned with our text-only normalization path without manual errors.
- As a pipeline operator, I want to download text-ready OER assets with stable slugs and filenames, so that `lexile-scoring-model-pipeline corpus download` and `normalize` succeed end-to-end without format conversions.

## Problem / Why

Manual manifest editing is slow, error-prone, and mismatched to the active pipeline schema (`source_id`, `id`, `url`, `filename`). The normalizer only ingests `.txt`/`.jsonl`, so PDF/EPUB links silently fail. We need a repeatable way to discover, filter, and register IA-hosted OpenStax/CK-12 titles that expose text derivatives, ensuring downloads and normalization work without ad-hoc fixes.

## Personas & Scenarios

- Persona: Data engineer / corpus curator
  - Maintains the Lexile scoring corpus and OER feeds
  - Cares about schema correctness, text-only compatibility, and predictable downloads
  - Constrained by existing pipeline (no PDF/EPUB ingestion) and limited time for manual curation
  - Wants a deterministic, scriptable path from catalog to manifest to download
  - Motivated to expand K-12 coverage (OpenStax, CK-12) with minimal rework
- Scenario: The curator runs the catalog builder to pull OpenStax/CK-12 metadata from Internet Archive, enriches entries with available `_djvu.txt` files, applies filters for subject/grade/language/license, and generates `oer_sources.json` with enforced `.txt` filenames. They then run `lexile-scoring-model-pipeline corpus download` and `normalize`; downloads land under the correct raw subfolders, and normalization consumes the files without format errors.


## Acceptance Criteria

- [ ] Catalog builder writes IA search results for OpenStax and CK-12 to `data/meta/catalogs/*.jsonl` with fields: `source_id`, `identifier/work_id`, `title`, `creator`, `year`, `language`, `license_url`, and `download_candidates` including `_djvu.txt` when present.
- [ ] Curation step filters catalog rows to entries with at least one text/plain candidate; rows without text are skipped with a recorded reason.
- [ ] Manifest generator emits `data/meta/oer_sources.json` entries containing `source_id`, stable slug `id`, direct `url`, and `.txt` `filename` values; all URLs return HTTP 200 and `Content-Type` text.*.
- [ ] `lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"` retrieves the manifest entries into the correct raw subfolders with the expected filenames.
- [ ] `lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"` ingests all downloaded files without empty outputs or format errors.

## Non-Goals

This effort does not add PDF/EPUB-to-text conversion, does not expand beyond OpenStax/CK-12 sources, and does not change normalization logic beyond consuming text assets registered in the manifest.
