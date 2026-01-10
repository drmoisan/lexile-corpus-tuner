# CK-12 Reader Content – Research Requirements

Status legend:
- [x] Completed (fully delivered)
- [ ] Not completed (may be partially investigated)

## Confirm endpoints and auth
- [ ] Validate the minimal set of reader API calls needed to fetch chapter content (e.g., `https://www.ck12.org/flx/get/detail/revision/<id>?tiny=true`) and any additional calls required for images/inline assets.
- Evidence so far:
	- Header sensitivity + CloudFront blocking behavior is documented in [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “Technical Requirements” and “Configuration Examples”.
	- `detail/revision` payload structure + inline asset patterns (including CloudFront `datastreams` image URLs) are documented in the same note under “API and Schema Documentation”.
- [ ] Determine whether requests work anonymously or require session cookies/headers. If auth is required, document the exact headers/cookies and how to refresh them.
- Evidence so far:
	- `reader_library` returns `Authentication required` (status 1009) without auth cookies even when header-blocking is bypassed: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “External Research” and “API and Schema Documentation”.

## Map FlexBook URL to revision ID
- [ ] Derive a deterministic mapping from a FlexBook URL (e.g., `https://flexbooks.ck12.org/cbook/<slug>/`) to the revision ID(s) used by the reader APIs.
- Evidence so far:
	- Canonical URL parsing rules (`XH`) and how `artifactCreator` is derived from `realm` are extracted from the frontend bundles: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “Implementation Patterns” / “Complete Examples”.
	- `/flx/artifact/...` requires `(artifactType, artifactHandle, artifactCreator)` and the naive tuple for `ck-12-physics` fails with error 2052: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “External Research” and “API and Schema Documentation”.
- [ ] Verify stability across sessions and whether multiple revisions exist per title; pick the canonical/latest one.
- Evidence so far:
	- The `locationInfo` reducer maintains `isLatest`/`revision` fields and is designed to carry “latest” semantics, but we have not yet verified a stable public-slug mapping end-to-end: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “Complete Examples” and “Follow-up Research Guidance (not implementation)”.

## Payload structure and extraction
- [ ] Inspect the JSON/HTML payload from `flx/get/detail/revision/<id>?tiny=true` (and related endpoints) to understand where chapter body text resides.
- Evidence so far:
	- Recorded HAR shows `flx/get/detail/revision/<id>?tiny=true` responses containing `response.section.xhtml` / `response.section.xhtml_prime` XHTML strings (full documents) and also `response.book` metadata: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “API and Schema Documentation”.
- [ ] Identify fields that contain main text vs. navigation/boilerplate; note how inline math, images, and tables are represented.
- Evidence so far:
	- Section XHTML includes `<img ...>` and `<table ...>` markup in the recorded HAR sample; MathML was not observed in that particular capture: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “API and Schema Documentation”.
- [ ] Decide on an HTML/JSON-to-text extraction strategy (e.g., readability-like cleaning, JSON field selection) and confirm it preserves reading order.
- Evidence so far:
	- We have identified the primary content carrier as XHTML-in-JSON (`response.section.xhtml*`), but no extraction strategy has been validated yet.

## Asset handling
- [ ] Determine whether inline images/math require additional fetches (e.g., CloudFront `datastreams/f-d:*` URLs) and whether alt text/captions are available for text representation.
- Evidence so far:
	- Section XHTML includes `<img>` tags whose `src` points to CloudFront `datastreams/...` URLs, and also includes a `data-flx-url="/flx/show/image/..."` attribute (suggesting an additional internal fetch route may exist): [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “API and Schema Documentation”.
- [ ] Define how to handle image-only sections (log + placeholder text vs. skip).
- Evidence so far:
	- Not yet investigated.

## Manifest format and validation
- [ ] Finalize whether CK-12 manifest entries should store `.html` or `.json` filenames based on the chosen extraction path.
- Evidence so far:
	- Not yet investigated.
- [ ] Define content-type validation rules for CK-12 entries (expected `text/html` or `application/json`).
- Evidence so far:
	- `flx/get/detail/revision/<id>?tiny=true` is `application/json` in the recorded HAR sample: [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “API and Schema Documentation”.

## Error handling and coverage
- [ ] Measure how often reader payloads fail (4xx/empty) across a sample set; document skip reasons and expected coverage.
- Evidence so far:
	- In the recorded HAR sample, 97 `flx/get/detail/revision/...` requests all returned HTTP 200 (note: this reflects that recorded browser context, not a guarantee for anonymous access): [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) under “API and Schema Documentation”.
- [ ] Establish timeouts/retries for the CK-12 fetcher and extractor, and how to surface failures in logs/metrics.
- Evidence so far:
	- Not yet investigated.

## Tooling updates
- [ ] List any new dependencies required for HTML/JSON-to-text extraction (e.g., `beautifulsoup4`, `readability-lxml`) and evaluate their licensing/size impact.
- Evidence so far:
	- Not yet investigated.
