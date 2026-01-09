# 2026-01-06-populate-open-stax-ck-12-manifest-73 — Spec

- **Status:** Superseded
- **Outcome:** Plan executed against an invalid data-source assumption; CK-12 ingestion via Internet Archive proved infeasible.
- **Root Cause:** Incorrect assumption that Internet Archive contained CK-12 FlexBook content addressable via metadata search; validation revealed zero CK-12 titles and misclassified OpenStax results.
- Issue: #73
- Owner: drmoisan
- Last Updated: 2026-01-06

## Overview

Implement a Gutenberg-style workflow for OpenStax and CK-12 OER: programmatically build catalogs from Internet Archive, curate entries that expose text derivatives, and emit a manifest (`oer_sources.json`) that the existing corpus downloader/normalizer can consume without manual edits or format conversion.

## Behavior

1) Catalog: Query IA Advanced Search for OpenStax and CK-12 items; persist results to `data/meta/catalogs/*.jsonl` with core metadata and initial download candidates.
2) Enrich: For each item, call IA Metadata API to detect text-friendly derivatives (prefer `_djvu.txt`, allow other text/plain forms); append them to `download_candidates`.
3) Curate: Filter catalog rows to those with valid text/plain candidates, respecting subject/grade/language/license filters; record skip reasons for non-text items.
4) Manifest: Generate `data/meta/oer_sources.json` entries with `source_id` (`openstax`/`ck12`), stable slug `id` (derived from immutable IA `identifier`, not mutable title), direct `url`, and `.txt` `filename`; validate URLs return HTTP 200 + text content.
5) Download: `lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"` reads the manifest and stores files under the correct raw subfolders with provided filenames.
6) Normalize: `lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"` ingests the downloaded text files without format errors and produces non-empty windows.
Alternative: If no text derivative exists for a desired title, the entry is skipped with a logged warning; adding PDF/EPUB conversion is deferred.

## Inputs / Outputs

- Inputs:
	- Sources list: `openstax`, `ck12` (CLI flag `--sources` on catalog/curate and download/normalize steps)
	- IA search query parameters (subject/grade/language/license filters)
	- Paths: `data/meta/catalogs/*.jsonl` (catalog), `data/meta/oer_sources.json` (manifest)
- Outputs:
	- Catalog files with metadata and `download_candidates`
	- Curated manifest entries with `.txt` filenames and reachable text URLs
	- Downloaded text assets under `data/corpus/raw/{openstax|ck12}`
	- Logs for skipped items and HTTP failures; optional summary counts (found/curated/downloaded/normalized)

### Manifest schema (authoritative)

`data/meta/oer_sources.json` must follow:

```json
{
	"sources": [
		{
			"source_id": "openstax | ck12",
			"id": "stable-slug-derived-from-ia-identifier",
			"url": "https://archive.org/download/<identifier>/<file>_djvu.txt",
			"filename": "<slug>.txt"
		}
	]
}
```

- `source_id` controls the raw subfolder used by download/normalize.
- `id` must be deterministic (lowercase hyphen slug from immutable IA `identifier`).
- `url` must be directly downloadable without cookies; prefer `_djvu.txt`, else other `text/plain`.
- `filename` must end with `.txt` to satisfy the current normalizer; normalize to snake/slug plus `.txt`.
- Validation: HTTP 200 status and `Content-Type` starting with `text/`; failures are logged and skipped.

### Internet Archive Option A workflow (catalog → curate → manifest)

- Catalog search endpoint: `https://archive.org/services/search/v1/scrape` with cursor paging to avoid the 10k sorted cap.
- Recommended queries:
	- OpenStax: `(mediatype:texts) AND (openstax OR "OpenStax") AND (textbook OR "open textbook")`
	- CK-12: `(mediatype:texts) AND ("CK-12" OR ck12 OR "CK12") AND (textbook OR "flexbook")`
- Requested fields: `identifier,title,creator,year,language,licenseurl,publicdate`; stable sort `identifier asc`; `count` 100–500.
- Metadata enrichment per item: `https://archive.org/metadata/<identifier>`; pick `_djvu.txt` first, else any `.txt`; build download URL via `https://archive.org/download/<identifier>/<filename>`.
- Source mapping heuristic: if creator/publisher/title contains `openstax` → `openstax`; contains `ck12`/`ck-12` → `ck12`; otherwise leave empty and require manual curation.
- Curation rules: require at least one text/plain candidate; enforce language/license/subject filters; log skip reasons (e.g., missing text derivative, unmapped source).
- Slug rule: `id = generate_stable_slug(identifier)` → lowercase, hyphenated, idempotent; never mutate once emitted.

## API / CLI Surface

- Catalog build (example):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.build_oer_catalog --sources "openstax,ck12" --out-dir data/meta/catalogs`
- Curation + manifest emit (example):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.curate_oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --require-text`
- Visual Curation (UI option):
  - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.curate_oer_ui`
  - Launches a basic Tkinter window to load catalogs, toggle checkboxes for inclusion, filter by metadata, and export `oer_sources.json`.
	- `poetry run lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"`
Return/side effects: catalog/manifest files written, download exits non-zero on unreachable URLs unless `--skip-missing` is set; normalization skips non-text files by design.

## Data & State

- New catalog artifacts in `data/meta/catalogs/` capturing IA metadata + download candidates per source.
- Updated `data/meta/oer_sources.json` manifest with curated text-ready entries.
- Downloaded text assets placed in `data/corpus/raw/openstax` and `data/corpus/raw/ck12` using manifest filenames.
- Logs/metrics indicating counts of candidates found, curated, skipped, downloaded, and normalized.

## Constraints & Risks

- Text-only ingestion: pipeline ignores PDF/EPUB; titles lacking text derivatives are skipped until conversion is added.
- OCR quality of `_djvu.txt` may introduce noise; monitor downstream scoring impact.
- Network availability and IA rate limits/HTTP 403s can block downloads; retries/backoff may be needed.
- Licensing/usage restrictions must be verified per item (especially OpenStax LLM-related notices); ensure licenses are captured in catalog rows.
- Slug stability is required for deduplication; changing `id` should be avoided once emitted.

## Definition of Done

- [ ] Behavior matches acceptance criteria (catalog → curate → manifest → download → normalize, including skip handling)
- [ ] Visual Curation UI implemented with Tkinter (loads catalog, supports filtering/selection, saves manifest)
- [ ] Tests updated/added:
  - Catalog parsing, text-candidate detection, manifest validation.
  - UI ViewModels/Logic (verify filtering and export logic separately from GUI widgets).
  - Download/normalize happy-path integration.
- [ ] Docs updated (README or source-curation-guide, plus this feature folder reflecting final CLI/paths)
- [ ] Telemetry/logging (if applicable) captures counts for found/curated/skipped/downloaded/normalized and surfaces download failures
