# 2026-01-06-populate-open-stax-ck-12-manifest-73 — Spec

- **Status:** Active
- **Outcome:** CK-12 spec revised to use Browse API for catalog discovery; vanity slugs are NOT resolvable—only canonical handles work; content fetched via JSON API with embedded XHTML.
- **Root Cause:** Research confirmed: (1) HAR evidence shows CK-12 content is served via flx/get/detail/revision/<id> APIs, (2) vanity URL slugs like `ck-12-physics` do not exist in CK-12's database—only canonical handles like `CK-12-Physics-FlexBook-2.0` work.
- Issue: #73
- Owner: drmoisan
- Last Updated: 2026-01-09

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

### CK-12 Workflow (Native Site Scraping + HTML/Text Extraction)

**Observed fact (HAR: artifacts/www.ck12.org.har):** Neither FlexBook 1.0 nor 2.0 sessions expose PDF downloads. The network traffic is limited to JSON/HTML payloads from `https://www.ck12.org/flx/get/detail/revision/<id>?tiny=true`, `flx/get/appdata/reader_library`, and image datastreams. No `.pdf` URLs are present.

**Research finding (2025-01-09):** Vanity URL slugs (e.g., `ck-12-physics` from `https://flexbooks.ck12.org/cbook/ck-12-physics/`) do NOT exist in CK-12's database. Only canonical handles from the Browse API work (e.g., `CK-12-Physics-FlexBook-2.0`). The workflow MUST use the Browse API for catalog discovery rather than parsing vanity URLs.

1) **Catalog**: Use CK-12 Browse API at `https://www.ck12.org/flx/browse/flexbook?limit=200` to enumerate all available FlexBooks; extract `artifactID`, `artifactType`, `handle`, `title`, `perma`; persist to `data/meta/catalogs/ck12_catalog.jsonl`.
   - **Critical**: Do NOT attempt to resolve vanity slugs from the fbbrowse list page—they are marketing URLs that don't map to database artifacts.
2) **Enrich**: For each artifact, call Perma API `https://www.ck12.org/flx/get/perma/<artifactType>/<handle>` to get full metadata including:
	- `revisions[0].children` containing chapter/section revision IDs
	- Author/creator information
	- Last modified / publication metadata if present in the JSON
	- Grade level and language
	- License (CK-12 Curriculum Materials License)
	Append section revision IDs to `download_candidates`.
3) **Curate**: Filter catalog rows to those with retrievable revision content (valid revision IDs that return HTTP 200). Record skip reasons for items that lack resolvable revisions or return errors.
4) **Manifest**: Generate `data/meta/oer_sources.json` entries with `source_id=ck12`, stable slug `id` (derived from canonical `handle`, NOT vanity URL), direct revision API URL (`https://www.ck12.org/flx/get/detail/revision/<id>?tiny=true`), and `.json` `filename`.
5) **Download**: `lexile-scoring-model-pipeline corpus download --sources "ck12"` fetches revision JSON to `data/corpus/raw/ck12` using manifest URLs. Must include browser-like headers (see Required Headers section).
6) **JSON/XHTML/Text Extraction**: New pipeline step to extract raw text from the downloaded revision payloads:
	- Input: `data/corpus/raw/ck12/*.json`
	- Parse JSON and extract `response.lesson.xhtml` (or `response.lesson.xhtml_prime`)
	- Convert XHTML to plain text using BeautifulSoup
	- Output: `data/corpus/raw/ck12/*.txt` (parallel text files)
	- Handle extraction errors gracefully; log files that fail text extraction.
7) **Normalize**: `lexile-scoring-model-pipeline corpus normalize --sources "ck12"` ingests the extracted `.txt` files.

**Note**: Because vanity slugs do not resolve, the catalog step MUST use Browse API discovery. The `id` in the manifest should be derived from the canonical `handle` field, not from parsed URLs.

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
			"url": "https://archive.org/download/<identifier>/<file>_djvu.txt | https://www.ck12.org/flx/get/detail/revision/<id>?tiny=true",
			"filename": "<slug>.txt | <slug>.html"
		}
	]
}
```

- `source_id` controls the raw subfolder used by download/normalize.
- `id` must be deterministic:
	- OpenStax: lowercase hyphen slug from immutable IA `identifier`
	- CK-12: lowercase hyphen slug from canonical `handle` (e.g., `ck-12-physics-flexbook-2-0` from `CK-12-Physics-FlexBook-2.0`). **NOT** from vanity URL slugs.
- `url` must be directly downloadable:
	- OpenStax: Prefer `_djvu.txt` from IA, else other `text/plain` formats
	- CK-12: Revision Detail API URL (`https://www.ck12.org/flx/get/detail/revision/<id>?tiny=true`). Requires browser-like headers for anonymous access.
- `filename` must match source format:
	- OpenStax: `.txt` extension for text files
	- CK-12: `.json` extension for revision API responses (content extraction parses JSON → extracts XHTML → converts to text)
- Validation: 
	- OpenStax: HTTP 200 status and `Content-Type` starting with `text/`
	- CK-12: HTTP 200 status and `Content-Type` of `application/json`; response must contain `response.lesson.xhtml` or `response.lesson.xhtml_prime`
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

**Critical Research Finding (2025-01-09)**: The original plan to scrape `fbbrowse/list` and resolve vanity slugs is NOT viable. Vanity slugs like `ck-12-physics` do not exist in CK-12's database—only canonical handles work.

**Verified Algorithm** (see [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md)):

1. **Browse API** (catalog discovery):
   - Endpoint: `https://www.ck12.org/flx/browse/flexbook?limit=200`
   - Returns: List of all available FlexBooks with `artifactID`, `artifactType`, `handle`, `title`, `perma`
   - Note: `artifactType=cbook` returns 0 results; physics books are under `artifactType=flexbook`

2. **Perma API** (metadata + revision hierarchy):
   - Endpoint: `https://www.ck12.org/flx/get/perma/<artifactType>/<handle>`
   - Example: `/flx/get/perma/cbook/CK-12-Physics-FlexBook-2.0`
   - Returns: `{response: {<artifactType>: {revisions: [{children: [...]}]}}}`
   - Children contain full chapter objects with nested `revisions[0].children` containing section revision IDs

3. **Revision Detail API** (content):
   - Endpoint: `https://www.ck12.org/flx/get/detail/revision/<revisionID>?tiny=true`
   - Returns: JSON with content under `response.lesson` (NOT `response.section`)
   - Fields: `xhtml` (full XHTML document), `xhtml_prime` (alternate format), `title`, `summary`

**Required Headers for Anonymous Access**:
```python
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.ck12.org/",
    "Origin": "https://www.ck12.org",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}
```

**Slug Derivation Rule**:
- `id = generate_stable_slug(handle)` → lowercase, replace dots/spaces with hyphens
- Example: `CK-12-Physics-FlexBook-2.0` → `ck-12-physics-flexbook-2-0`
- **Do NOT** use vanity URL slugs (e.g., `ck-12-physics` from marketing URLs)

**JSON/XHTML/Text Extraction Requirements:**
- Library: `beautifulsoup4` with `lxml` parser for XHTML extraction
- Extraction strategy:
	1. Parse downloaded JSON file
	2. Extract `response.lesson.xhtml` (or `response.lesson.xhtml_prime`)
	3. Parse XHTML with BeautifulSoup
	4. Extract text content, preserving reading order
	5. Strip navigation and boilerplate elements
	6. Handle extraction failures gracefully (log error, skip file, continue pipeline)
- Output: Plain text file parallel to the fetched JSON (e.g., `section-8384007.json` → `section-8384007.txt`)
- Quality considerations: XHTML is well-structured; alt text for images is included in `<img alt="...">` attributes

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

### CK-12 (API-based Discovery) Commands

- Catalog build (Browse API):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
	- Calls `https://www.ck12.org/flx/browse/flexbook?limit=200`, outputs `ck12_catalog.jsonl`
	- **Note**: Uses Browse API, NOT fbbrowse scraping (vanity slugs don't resolve)
- Enrichment (Perma API for metadata + revision IDs):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
	- Calls `/flx/get/perma/<artifactType>/<handle>` for each artifact to get revision hierarchy
- Curation + manifest emit:
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls`
- Download (JSON revision payloads):
	- `poetry run lexile-scoring-model-pipeline corpus download --sources "ck12"`
	- Downloads JSON revision files to `data/corpus/raw/ck12/` using browser-like headers
- JSON/XHTML Text Extraction (new step):
	- `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_ck12_text --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12`
	- Parses JSON, extracts `response.lesson.xhtml`, converts to `.txt` files
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

**Research-driven changes (2025-01-09)**:
- CK-12 catalog source changed from `fbbrowse/list` scraping to Browse API (`/flx/browse/flexbook`) due to vanity slug resolution failure.
- CK-12 manifest URLs use Revision Detail API (`/flx/get/detail/revision/<id>?tiny=true`) instead of hypothetical PDF endpoints.
- CK-12 content extraction uses JSON → XHTML → text path (via `response.lesson.xhtml`), not PDF extraction.
- CK-12 slug derivation uses canonical `handle` from Browse API, not vanity URL path segments.
- CK-12 download requires browser-like headers (User-Agent, Referer, Origin, Sec-Fetch-*) for anonymous access.

**Prior decisions (retained)**:
- CK-12 enrichment extracts author/grade/language via JSON-LD/meta/regex heuristics; per-entry enrichment failures are skipped with stderr output.
- Manifest validation uses HEAD-only checks with source-specific content-type gates (text/* for OpenStax, application/json for CK-12).
- JSON/XHTML extraction uses `beautifulsoup4` with 30s per-file timeout and up to 4 concurrent workers; empty extractions raise `ValueError` (logged) while batch processing continues.

## Definition of Done

- [ ] Behavior matches acceptance criteria (catalog → curate → manifest → download → normalize, including skip handling)
- [ ] Visual Curation UI implemented with Tkinter (loads catalog, supports filtering/selection, saves manifest)
- [ ] Tests updated/added:
  - Catalog parsing, text-candidate detection, manifest validation.
  - UI ViewModels/Logic (verify filtering and export logic separately from GUI widgets).
  - Download/normalize happy-path integration.
- [ ] Docs updated (README or source-curation-guide, plus this feature folder reflecting final CLI/paths)
- [ ] Telemetry/logging (if applicable) captures counts for found/curated/skipped/downloaded/normalized and surfaces download failures
