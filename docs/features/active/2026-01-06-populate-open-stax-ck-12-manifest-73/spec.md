# 2026-01-06-populate-open-stax-ck-12-manifest-73 — Spec

- Issue: #73
- Owner: drmoisan
- Last Updated: 2026-01-06

## Overview

Implement a Gutenberg-style workflow for OpenStax and CK-12 OER: programmatically build catalogs from Internet Archive, curate entries that expose text derivatives, and emit a manifest (`oer_sources.json`) that the existing corpus downloader/normalizer can consume without manual edits or format conversion.

## Behavior

### OpenStax Workflow (Internet Archive-based)

1) **Catalog**: Query IA Advanced Search for OpenStax items; persist results to `data/meta/catalogs/openstax_catalog.jsonl` with core metadata and initial download candidates.
2) **Enrich**: For each item, call IA Metadata API to detect text-friendly derivatives (prefer `_djvu.txt`, allow other text/plain forms); append them to `download_candidates`.
3) **Curate**: Filter catalog rows to those with valid text/plain candidates, respecting subject/grade/language/license filters; record skip reasons for non-text items.
4) **Manifest**: Generate `data/meta/oer_sources.json` entries with `source_id=openstax`, stable slug `id`, direct `url`, and `.txt` `filename`.
5) **Download**: `lexile-scoring-model-pipeline corpus download --sources "openstax"` reads the manifest and stores files under `data/corpus/raw/openstax`.
6) **Normalize**: `lexile-scoring-model-pipeline corpus normalize --sources "openstax"` ingests the downloaded text files.

### CK-12 Workflow (Native Site Scraping + PDF Extraction)

1) **Catalog**: Scrape the CK-12 FlexBook catalog at `https://www.ck12.org/fbbrowse/list?grade=all%20grades&language=all%20languages&subject=all%20subjects`; extract book titles, URLs, and subject categories; persist to `data/meta/catalogs/ck12_catalog.jsonl`.
2) **Enrich**: For each book URL (e.g., `https://flexbooks.ck12.org/cbook/...`), follow link one level deep to extract:
   - Author/creator information
   - Publication date
   - Grade level
   - Language
   - License information (CK-12 Curriculum Materials License)
   - PDF download link (if available via export/download options)
   Append PDF download URL to `download_candidates`.
3) **Curate**: Filter catalog rows to those with valid PDF download links, respecting subject/grade/language filters; record skip reasons for items without PDF exports.
4) **Manifest**: Generate `data/meta/oer_sources.json` entries with `source_id=ck12`, stable slug `id` (derived from FlexBook URL slug), PDF download URL, and `.pdf` `filename`.
5) **Download**: `lexile-scoring-model-pipeline corpus download --sources "ck12"` downloads PDF files to `data/corpus/raw/ck12`.
6) **PDF Text Extraction**: New pipeline step to extract raw text from PDFs using a library like `pypdf` or `pdfplumber`:
   - Input: `data/corpus/raw/ck12/*.pdf`
   - Output: `data/corpus/raw/ck12/*.txt` (parallel text files)
   - Handle extraction errors gracefully; log PDFs that fail text extraction.
7) **Normalize**: `lexile-scoring-model-pipeline corpus normalize --sources "ck12"` ingests the extracted `.txt` files.

**Note**: If no PDF export option exists for a FlexBook, the entry is skipped with a logged warning. Interactive-only content without downloadable formats is deferred.

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
			"id": "stable-slug-derived-from-source-identifier",
			"url": "https://archive.org/download/<identifier>/<file>_djvu.txt | https://flexbooks.ck12.org/flx/pdf/<book>.pdf",
			"filename": "<slug>.txt | <slug>.pdf"
		}
	]
}
```

- `source_id` controls the raw subfolder used by download/normalize.
- `id` must be deterministic:
	- OpenStax: lowercase hyphen slug from immutable IA `identifier`
	- CK-12: lowercase hyphen slug from FlexBook URL path segment
- `url` must be directly downloadable:
	- OpenStax: Prefer `_djvu.txt` from IA, else other `text/plain` formats
	- CK-12: PDF download URL from FlexBook export/download interface
- `filename` must match source format:
	- OpenStax: `.txt` extension for text files
	- CK-12: `.pdf` extension for PDF files (extracted to `.txt` in separate step)
- Validation: 
	- OpenStax: HTTP 200 status and `Content-Type` starting with `text/`
	- CK-12: HTTP 200 status and `Content-Type` starting with `application/pdf`
	- Failures are logged and skipped

### Internet Archive Workflow (OpenStax Only)

- Catalog search endpoint: `https://archive.org/services/search/v1/scrape` with cursor paging to avoid the 10k sorted cap.
- Recommended query:
	- OpenStax: `(mediatype:texts) AND (openstax OR "OpenStax") AND (textbook OR "open textbook")`
- Requested fields: `identifier,title,creator,year,language,licenseurl,publicdate`; stable sort `identifier asc`; `count` 100–500.
- Metadata enrichment per item: `https://archive.org/metadata/<identifier>`; pick `_djvu.txt` first, else any `.txt`; build download URL via `https://archive.org/download/<identifier>/<filename>`.
- Curation rules: require at least one text/plain candidate; enforce language/license/subject filters; log skip reasons (e.g., missing text derivative).
- Slug rule: `id = generate_stable_slug(identifier)` → lowercase, hyphenated, idempotent; never mutate once emitted.

### CK-12 Native Scraping Workflow (New Approach)

- Catalog page: `https://www.ck12.org/fbbrowse/list?grade=all%20grades&language=all%20languages&subject=all%20subjects`
- Extract book entries with:
	- Title (from link text)
	- FlexBook URL (e.g., `https://flexbooks.ck12.org/cbook/ck-12-interactive-middle-school-math-6-for-ccss/`)
	- Subject category (from page section headers like "Algebra", "Biology", etc.)
- For each book URL, fetch metadata:
	- Author/Creator (typically "CK-12" or contributor name)
	- Last Modified date
	- License (CK-12 Curriculum Materials License)
	- Grade level (if available in title or metadata)
	- Language (default "eng", detect Spanish editions from title)
	- PDF download link discovery:
		- Check for export/download buttons in FlexBook interface
		- Possible URL patterns: `https://flexbooks.ck12.org/flx/pdf/...` or similar
		- If no PDF export available, mark as interactive-only and skip
- Curation rules: require valid PDF download URL; enforce language/grade/subject filters.
- Slug rule: `id = generate_stable_slug(flexbook_url_slug)` → extract last path segment, lowercase, hyphenated.

**PDF Text Extraction Requirements:**
- Library: Use `pypdf` (PyPDF2 successor) or `pdfplumber` for text extraction
- Installation: Add `pypdf>=3.0.0` or `pdfplumber>=0.10.0` to `pyproject.toml`
- Extraction strategy:
	- Prefer `pdfplumber` for better layout preservation and table handling
	- Fallback to `pypdf` for simple extraction if `pdfplumber` fails
	- Extract text page-by-page, preserving paragraph boundaries
	- Handle extraction failures gracefully (log error, skip file, continue pipeline)
- Output: Plain text file parallel to PDF (e.g., `book.pdf` → `book.txt`)
- Quality considerations: PDF extraction may lose formatting, tables, equations; monitor impact on lexile scoring

## API / CLI Surface

### OpenStax (IA-based) Commands

- Catalog build:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_catalog --sources "openstax" --out-dir data/meta/catalogs`
- Enrichment:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_enrichment --catalog-file data/meta/catalogs/openstax_catalog.jsonl --output data/meta/catalogs/openstax_enriched.jsonl`
- Curation + manifest emit:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-text --sources "openstax" --out-dir data/meta/catalogs`
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls`
- Visual Curation (UI option):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_ui`
- Download:
	- `poetry run lexile-scoring-model-pipeline corpus download --sources "openstax"`
- Normalize:
	- `poetry run lexile-scoring-model-pipeline corpus normalize --sources "openstax"`

### CK-12 (Native Scraping) Commands

- Catalog build (new scraper):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
	- Scrapes `https://www.ck12.org/fbbrowse/list`, outputs `ck12_catalog.jsonl`
- Enrichment (fetch per-book metadata):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
	- Follows each FlexBook URL to extract detailed metadata and PDF download links
- Curation + manifest emit:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-pdf --sources "ck12" --out-dir data/meta/catalogs`
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls --allow-pdf`
- Download (PDFs):
	- `poetry run lexile-scoring-model-pipeline corpus download --sources "ck12"`
	- Downloads PDF files to `data/corpus/raw/ck12/`
- PDF Text Extraction (new step):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_pdf_text --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12`
	- Extracts `.txt` files parallel to `.pdf` files
- Normalize:
	- `poetry run lexile-scoring-model-pipeline corpus normalize --sources "ck12"`
	- Processes the extracted `.txt` files

Return/side effects: catalog/manifest files written, download exits non-zero on unreachable URLs unless `--skip-missing` is set; PDF extraction logs failures but continues; normalization skips failed extractions by design.

## Data & State

- New catalog artifacts in `data/meta/catalogs/` capturing IA metadata + download candidates per source.
- Updated `data/meta/oer_sources.json` manifest with curated text-ready entries.
- Downloaded text assets placed in `data/corpus/raw/openstax` and `data/corpus/raw/ck12` using manifest filenames.
- Logs/metrics indicating counts of candidates found, curated, skipped, downloaded, and normalized.

## Constraints & Risks

### Original Risks

- Text-only ingestion: pipeline ignores PDF/EPUB; titles lacking text derivatives are skipped until conversion is added.
- OCR quality of `_djvu.txt` may introduce noise; monitor downstream scoring impact.
- Network availability and IA rate limits/HTTP 403s can block downloads; retries/backoff may be needed.
- Licensing/usage restrictions must be verified per item (especially OpenStax LLM-related notices); ensure licenses are captured in catalog rows.
- Slug stability is required for deduplication; changing `id` should be avoided once emitted.

### Unanticipated Risk: Internet Archive Metadata Quality (CK-12)

**Issue Discovered:** Internet Archive does not have properly tagged CK-12 content with consistent metadata. The IA Advanced Search query for CK-12 returns the same OpenStax results due to:
- IA's `NOT` query operators not working reliably for metadata exclusion
- Inconsistent tagging where OpenStax materials are not properly labeled with "openstax" in all metadata fields
- No actual CK-12 FlexBook content properly cataloged in IA with distinguishing metadata

**Impact:** The original approach (catalog both OpenStax and CK-12 from IA using parallel queries) is not viable for CK-12. All 82 results returned by the CK-12 query are actually OpenStax materials.

**Resolution:** Split the implementation into two distinct workflows:
1. **OpenStax (IA-based)**: Continue using Internet Archive scrape API as originally specified
2. **CK-12 (Native site scraping)**: Scrape CK-12's official catalog at `https://www.ck12.org/fbbrowse/list`, download PDFs directly from FlexBooks platform, and extract text from PDF format

## Implementation Decisions / Deviations

- CK-12 catalog parsing currently captures titles and FlexBook URLs; subject categories, license metadata, and publication dates are not populated in catalog rows.
- CK-12 enrichment extracts author/grade/language via JSON-LD/meta/regex heuristics but does not populate license or publication date; PDF link discovery requires absolute `.pdf` URLs from anchors/buttons, and per-entry enrichment failures are skipped with stderr output (no skip log entry).
- Manifest validation uses HEAD-only checks with source-specific content-type gates (text/* for OpenStax, application/pdf for CK-12); there is no GET/Range fallback when HEAD is blocked.
- PDF extraction uses `pdfplumber` only with a 30s per-file timeout and up to 4 concurrent workers; there is no `pypdf` fallback, and empty extractions raise `ValueError` (logged) while batch processing continues.

## Definition of Done

- [ ] Behavior matches acceptance criteria (catalog → curate → manifest → download → normalize, including skip handling)
- [ ] Visual Curation UI implemented with Tkinter (loads catalog, supports filtering/selection, saves manifest)
- [ ] Tests updated/added:
  - Catalog parsing, text-candidate detection, manifest validation.
  - UI ViewModels/Logic (verify filtering and export logic separately from GUI widgets).
  - Download/normalize happy-path integration.
- [ ] Docs updated (README or source-curation-guide, plus this feature folder reflecting final CLI/paths)
- [ ] Telemetry/logging (if applicable) captures counts for found/curated/skipped/downloaded/normalized and surfaces download failures
