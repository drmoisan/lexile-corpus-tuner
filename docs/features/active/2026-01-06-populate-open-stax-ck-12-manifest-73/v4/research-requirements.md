# CK-12 Reader Content – Research Requirements

## Confirm endpoints and auth
- Validate the minimal set of reader API calls needed to fetch chapter content (e.g., `https://www.ck12.org/flx/get/detail/revision/<id>?tiny=true`) and any additional calls required for images/inline assets.
- Determine whether requests work anonymously or require session cookies/headers. If auth is required, document the exact headers/cookies and how to refresh them.

## Map FlexBook URL to revision ID
- Derive a deterministic mapping from a FlexBook URL (e.g., `https://flexbooks.ck12.org/cbook/<slug>/`) to the revision ID(s) used by the reader APIs.
- Verify stability across sessions and whether multiple revisions exist per title; pick the canonical/latest one.

## Payload structure and extraction
- Inspect the JSON/HTML payload from `flx/get/detail/revision/<id>?tiny=true` (and related endpoints) to understand where chapter body text resides.
- Identify fields that contain main text vs. navigation/boilerplate; note how inline math, images, and tables are represented.
- Decide on an HTML/JSON-to-text extraction strategy (e.g., readability-like cleaning, JSON field selection) and confirm it preserves reading order.

## Asset handling
- Determine whether inline images/math require additional fetches (e.g., CloudFront `datastreams/f-d:*` URLs) and whether alt text/captions are available for text representation.
- Define how to handle image-only sections (log + placeholder text vs. skip).

## Manifest format and validation
- Finalize whether CK-12 manifest entries should store `.html` or `.json` filenames based on the chosen extraction path.
- Define content-type validation rules for CK-12 entries (expected `text/html` or `application/json`).

## Error handling and coverage
- Measure how often reader payloads fail (4xx/empty) across a sample set; document skip reasons and expected coverage.
- Establish timeouts/retries for the CK-12 fetcher and extractor, and how to surface failures in logs/metrics.

## Tooling updates
- List any new dependencies required for HTML/JSON-to-text extraction (e.g., `beautifulsoup4`, `readability-lxml`) and evaluate their licensing/size impact.
