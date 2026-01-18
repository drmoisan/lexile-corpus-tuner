<!-- markdownlint-disable-file -->

# Task Research Notes: CK-12 catalog only returns 10 entries

## Research Executed

### File Analysis

- /workspaces/lexile-corpus-tuner/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py
  - `DEFAULT_CK12_CATALOG_URL` points to `https://www.ck12.org/flx/browse/flexbook?limit=200` and the CLI writes only what `parse_catalog_json` returns. No pagination/offset handling is present.
- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/plan.2026-01-09T22-27.md
  - Plan references CK-12 Browse API endpoint with `limit=200` and required headers, but does not mention pagination/offset.
- /workspaces/lexile-corpus-tuner/artifacts/bundle.js
  - FlexBook browse UI code references a static data source (`https://static.ck12.org/testimonial/fbbrowse-prod.json`) and routes `/fbbrowse` and `/fbbrowse/list`, indicating the UI list is populated from a static JSON feed rather than the `flx/browse` endpoint.

### Code Search Results

- ck12_catalog|flexbook|browse
  - docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/plan.2026-01-09T22-27.md
  - docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v3/code-review.2026-01-08T13-48.md
- fbbrowse-prod.json|fbbrowse/list
  - artifacts/bundle.js (FlexBook browse bundle)

### External Research

- #fetch:https://www.ck12.org/flx/browse/flexbook?limit=10&offset=0
  - Response JSON includes `response.limit: 10`, `response.offset: 0`, and `response.total: 19`, matching the 10-item payload.
- #fetch:https://www.ck12.org/flx/browse/flexbook?limit=10&offset=10
  - Response JSON still reports `response.offset: 0` and returns the same first-page data, indicating the `offset` parameter was ignored in this request.
- #fetch:https://www.ck12.org/flx/browse/flexbook?limit=50&offset=0
  - Response JSON continues to report `response.limit: 10` and `response.offset: 0`, showing that the server caps or ignores the `limit` value for this endpoint in these requests.
- #fetch:https://www.ck12.org/flx/browse/flexbook?limit=10&offset=10
  - Browser-like headers + session cookies captured from https://www.ck12.org/ still returned `response.limit: 10` and `response.offset: 0` with the same first-page entries, so pagination did not activate with basic cookie/header mirroring.
- #fetch:https://static.ck12.org/testimonial/fbbrowse-prod.json
  - Static JSON feed includes `filters`, `books`, and `clubbedBooks` collections. Each `books` entry includes fields like `Title`, `Content_URL`, `Thumb_URL`, `Language`, `Subject`, `Grade`, `Flexbook_2dot0`, and `Community_Contributed`.
- #fetch:https://flexbooks.ck12.org/cbook/ck-12-interactive-middle-school-math-6-for-ccss/
  - FlexBook 2.0 content pages include metadata (title, author, last modified) plus links to teacher editions, standards alignment, and section routes.
- #fetch:https://www.ck12.org/book/CK-12-Middle-School-Math-Grade-6/
  - CK-12 book pages provide a table of contents with `/section/` routes and include a breadcrumb back to `/fbbrowse/list` with subject/language/grade query parameters.
- #fetch:https://dr282zn36sxxg.cloudfront.net/datastreams/f-d%3A7a2401fac0b00f8c17ae80229f9a8c199c134578287a46ccd8a70dfe%2BCOVER_PAGE_THUMB_POSTCARD_TINY%2BCOVER_PAGE_THUMB_POSTCARD_TINY.1
  - Direct thumbnail fetch returned HTTP 403, implying some media URLs may be gated or require request headers.
- #fetch:https://www.ck12.org/fbbrowse/list/?language=all%20languages&Grade=all%20grades&subject=arithmetic
  - Rendered FlexBooks list page enumerates FlexBooks and books for the selected subject and links out to `flexbooks.ck12.org` and `www.ck12.org` resources.
- #fetch:https://flexbooks.ck12.org/tecbook/CK-12-Interactive-Middle-School-Math-6-for-CCSS-Teachers-Guide
  - Teacher edition pages mirror the student pages with authorship and standards alignment links.
- #fetch:https://flexbooks.ck12.org/cbook/ck-12-interactive-middle-school-math-6-for-ccss/section/1.1
  - Section pages surface related content links (lesson, enrichment, and next/previous routes).
- #fetch:https://www.ck12.org/book/ck-12-middle-school-math-grade-6/section/1.0/
  - CK-12 section pages provide per-section table-of-contents links plus breadcrumb navigation back to `/fbbrowse/list`.
- #fetch:https://www.ck12info.org/about//terms-of-use/
  - Terms of Use explicitly prohibit automated or partially automated access (scraping/robots) and usage for building or training AI/ML models without permission.
- #fetch:https://www.ck12info.org/about//technology-2/privacy-policy/
  - Privacy Policy documents platform data practices and links back to CK-12 platform resources.
- #fetch:https://www.ck12info.org/about//attribution-guidelines/
  - Attribution guidelines require visible CK-12 attribution badges and either noindex or canonical tags; links must point back to CK-12 source pages.
- #fetch:https://www.ck12info.org/curriculum-materials-license/
  - Curriculum Materials License restricts use to Educational Purposes, requires attribution, and explicitly bans AI/ML training and automated content creation without written permission.
- #fetch:https://play.google.com/store/apps/details?id=org.ck12.flexi
  - Flexi app listing confirms CK-12’s Flexi AI Tutor branding and links back to CK-12 Foundation.
- #fetch:https://chromewebstore.google.com/detail/flexi-ai-tutor/necpjbiijgficbgfanibojgkcbbooicg
  - Chrome Web Store listing provides CK-12 Foundation developer metadata and links to CK-12 privacy policy.
- #fetch:https://help.ck12.org/hc/en-us/articles/13186687440027-How-to-Bookmark-Flexi-to-Your-Home-Screen
  - Help center article links to the CK-12 Flexi homepage and general support resources.
- #fetch:https://www.ck12.org/browse
  - CK-12 browse page links to interactive FlexBooks and surfaces the same Terms/Privacy/Attribution links in the footer.

### Project Conventions

- Standards referenced: general-code-change.instructions.md, python-code-change.instructions.md, general-unit-test.instructions.md, python-unit-test.instructions.md
- Instructions followed: copilot-instructions.md; plan.2026-01-09T22-27.md requirements

## Key Discoveries

### Project Structure

The CK-12 catalog CLI (`ck12_catalog.py`) fetches a single page from the CK-12 Browse API and parses the returned list. There is no pagination or offset loop, so the output count is entirely determined by the API’s first-page response.

The static `fbbrowse-prod.json` feed currently contains 232 `books` entries. Extracted content handles (derived from `Content_URL` path segments) produce 165 unique handles, with 35 handles appearing multiple times (likely due to multi-subject classification). No `Content_URL` entries were missing a handle.

Field distributions in the feed (top-level counts):
- Language: English 209, Spanish 23
- General: Math 121, Science 96, Social Studies 7, Language Arts 5, Photography 3
- Grade: High School 126, Middle School 100, Elementary School 6
- Flexbook_2dot0: Y 99, N 133
- Community_Contributed: Y 10, N 222

### Implementation Patterns

The catalog builder uses a single GET to `DEFAULT_CK12_CATALOG_URL` and parses `response.flexbook` or related list fields. It filters entries missing `artifactID`, `artifactType`, or `handle`, but does not apply any explicit count limiting beyond what the API returns. The FlexBook browse UI (per `artifacts/bundle.js` and `fbbrowse-prod.json`) appears to load a static JSON payload and route to `/fbbrowse/list`, suggesting an alternate data source for the full catalog. The static feed enumerates book metadata and includes `Content_URL` values that route to either `flexbooks.ck12.org` (FlexBook 2.0) or `www.ck12.org` (book) pages.

### Complete Examples

```json
{
  "response": {
    "flexbook": [
      {"artifactID": 211633, "artifactType": "book", "handle": "CK-12-Chemistry-Intermediate"}
    ],
    "limit": 10,
    "offset": 0,
    "total": 19
  }
}
```

### API and Schema Documentation

The CK-12 Browse API returns a `response` object with pagination fields (`limit`, `offset`, `total`). In observed calls, both `limit` and `offset` are ignored (response shows `limit=10` and `offset=0` regardless of input), which blocks conventional offset-based paging for this endpoint as currently accessed. Browser-style headers and session cookies did not change this behavior. The FlexBook browse UI bundle references a static JSON feed (`https://static.ck12.org/testimonial/fbbrowse-prod.json`) that provides `filters`, `books`, and `clubbedBooks` data for `/fbbrowse/list`, implying the UI list is fed by a static catalog dataset rather than the browse endpoint. The list page at `/fbbrowse/list` renders the same catalog and links out to FlexBook 2.0 and book pages.

### Configuration Examples

```text
https://www.ck12.org/flx/browse/flexbook?limit=200
https://static.ck12.org/testimonial/fbbrowse-prod.json
```

### Technical Requirements

- The CK-12 Browse API advertises pagination fields, but the server ignores `offset` and caps `limit` in the observed responses.
- Retrieving all entries requires either a different API endpoint or stronger request context (auth or internal API) that enables paging; basic browser headers/cookies were insufficient.
- The current implementation does not page, so it can only return the first page in its present form.
- CK-12 Terms of Use explicitly prohibit automated scraping/robot access and AI/ML training without express permission; any catalog ingestion must respect these restrictions or secure written authorization.
- Some media URLs (CloudFront thumbnail assets) returned HTTP 403 when fetched directly, indicating assets may require authenticated or header-qualified requests.
- The Curriculum Materials License further restricts distribution and modification to Educational Purposes, mandates attribution, and explicitly disallows AI/ML training or automated content creation without permission.
- No local `ck12_catalog.jsonl` file was found under `data/meta/catalogs/`, so direct comparison between the static feed and the current catalog output could not be completed in this session.

**Mandatory unachievable objective callout**:
- **A working pagination mechanism could not be verified from the public endpoint; `offset` and `limit` inputs were ignored in all observed responses, so paging through all results is not demonstrably achievable with the current endpoint.**

## Recommended Approach

Pivot to the FlexBook browse UI’s data source: the bundled UI code fetches a static JSON payload from `https://static.ck12.org/testimonial/fbbrowse-prod.json`. Validating and consuming this feed is the most promising alternative to the non-paginated `flx/browse` endpoint.

## Implementation Guidance

- **Objectives**: Validate the FlexBook browse UI’s static JSON feed as the canonical catalog source.
- **Key Tasks**: Retrieve `fbbrowse-prod.json`, confirm it contains the full `books` list and required fields (handle, content URL, titles, etc.), and determine update cadence/versioning; document any gaps versus current catalog needs.
- **Dependencies**: None beyond existing HTTP client unless CK-12 gates the static feed.
- **Success Criteria**: Confirmed feed that enumerates all FlexBooks with stable identifiers, or evidence that the feed is partial and another authenticated API is required.

## Additional Research (Determinism Gaps Closed)

### Static feed schema evidence
- The static feed includes top-level `filters`, `books`, and `clubbedBooks` keys. The `books` list entries observed in the response include fields: `Language`, `Language_Code`, `General`, `Subject`, `Flexbook_2dot0`, `Community_Contributed`, `FB`, `Grade`, `Standard`, `Title`, `Content_URL`, `Thumb_URL`, `Enable`, and `Remarks`.
- `Content_URL` values in the feed include:
  - FlexBook 2.0 patterns under `flexbooks.ck12.org` with `/cbook/<slug>/...` segments.
  - CK-12 book patterns under `www.ck12.org` with `/book/<slug>/...` segments.
  - Community-contributed patterns under `flexbooks.ck12.org` that include `/user:<handle>/cbook/<slug>/...` segments (not just `/cbook/<slug>/`).
- These observed URL patterns confirm parsing must handle both `/cbook/` and `/user:<handle>/cbook/` paths when deriving slugs.

### CatalogEntry invariants (oer_models.py)
- `CatalogEntry` fields `artifact_type`, `handle`, and `artifact_id` are optional (`None` allowed), while `identifier` is required and validated by `generate_stable_slug`, which raises `ValueError` for empty identifiers.
- `source_id` is optional and `language` must be a list of strings, so static-feed parsing should always emit a non-empty identifier and normalize language to a list.
- The module documentation notes invariant expectations (stable identifiers and strict manifest filename suffixes), but does not require `artifact_id` for catalog entries.

### Unit test 1-to-1 mapping confirmation
- `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` is the only test file referencing `ck12_catalog` in the tests tree. No additional `ck12_catalog` references were found in other test modules, so the 1-to-1 mapping constraint is satisfied if updates remain in this file.
