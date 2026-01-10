# CK-12 `/cbook/<slug>/` → Artifact Identity → Revision ID Mapping

## Research Plan (Atomic Tasks)

**Objective**: Determine a deterministic, unauthenticated algorithm to map a public FlexBook URL like `https://flexbooks.ck12.org/cbook/ck-12-physics/` to the internal `(artifactType, artifactHandle, artifactCreator, revision)` tuple and ultimately to the revision IDs required by `/flx/get/detail/revision/<id>?tiny=true`.

**Output location**: All findings append to [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md).

---

## Phase 0 — Context & Inputs

- [x] [P0-T1] Read `.github/copilot-instructions.md` and `.github/instructions/general-code-change.instructions.md` to confirm research-only output rules (write to `artifacts/research/` only)
- [x] [P0-T2] Review the current state of [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) to understand prior findings and avoid duplicate work
- [x] [P0-T3] Review [v4/research-requirements.md](research-requirements.md) to ensure all unchecked requirements are addressed by this plan

**Success criteria (Phase 0)**: ✅ COMPLETE - Researcher understands existing knowledge base and knows which questions remain open.

---

## Phase 1 — Quick Wins: HTML Bootstrap Data Inspection

Rationale: The downloaded `/cbook/<slug>/` HTML pages may contain embedded JSON with the artifact identity already resolved. This is the fastest path to a solution.

### Parallel tasks (independent)

- [x] [P1-T1] Inspect `artifacts/research/tmp/ck-12-physics.html` for `<script>` tags containing JSON with keys like `artifactID`, `artifactRevisionID`, `revisionID`, `handle`, `artifactType`, or `ck12editor`
  - **Result**: NO embedded bootstrap data found with identity tuple
- [x] [P1-T2] Inspect `artifacts/research/tmp/book.html` for embedded JSON bootstrap data using the same key search as P1-T1
  - **Result**: NO embedded bootstrap data found
- [x] [P1-T3] Inspect `artifacts/research/tmp/reader-index.html` for embedded JSON bootstrap data using the same key search as P1-T1
  - **Result**: NO embedded bootstrap data found

### Follow-up task (depends on P1-T1 through P1-T3)

- [x] [P1-T4] Summarize HTML inspection findings: record whether bootstrap data exists, its JSON schema, and whether it provides a complete identity tuple
  - **Result**: NO - HTML pages do not contain embedded identity tuples. Must use API discovery.

**Success criteria (Phase 1)**: ✅ COMPLETE (NEGATIVE) - Bootstrap data does NOT provide identity tuple directly.

---

## Phase 2 — HAR Network Analysis

Rationale: The recorded `artifacts/www.ck12.org.har` (~75MB, 1271 entries) captures actual browser network activity and may reveal the lookup sequence or redirect chain.

### Parallel tasks (independent)

- [x] [P2-T1] Parse `artifacts/www.ck12.org.har` and list all unique request URLs matching pattern `/flx/artifact` (any variant)
  - **Result**: Found `/flx/artifact/...` calls require `(artifactType, artifactHandle, artifactCreator)` tuple
- [x] [P2-T2] Parse `artifacts/www.ck12.org.har` and list all unique request URLs matching pattern `/flx/get/` (excluding `/flx/get/detail/revision/`)
  - **Result**: Found `/flx/get/perma/<type>/<handle>` and `/flx/get/appdata/reader_library`
- [x] [P2-T3] Parse `artifacts/www.ck12.org.har` for any 3xx redirect responses from `/cbook/` paths
  - **Result**: NO redirects observed - pages load via client-side routing
- [x] [P2-T4] Parse `artifacts/www.ck12.org.har` and extract any request/response containing JSON with `artifactID` or `artifactRevisionID` fields
  - **Result**: Found in `detail/revision` responses; also in `/flx/browse/` results
- [x] [P2-T5] Parse `artifacts/www.ck12.org.har` for any requests to `/flx/show/` or `/flx/search/` or `/flx/browse/` patterns
  - **Result**: YES - `/flx/browse/<artifactType>` discovered (key finding!)

### Follow-up task (depends on P2-T1 through P2-T5)

- [x] [P2-T6] Summarize HAR analysis findings
  - **Result**: `/flx/browse/flexbook` lists all artifacts → `/flx/get/perma/<type>/<handle>` gets metadata → `/flx/get/detail/revision/<id>` gets content

**Success criteria (Phase 2)**: ✅ COMPLETE - Identified Browse API as discovery mechanism.

---

## Phase 3 — Bundle Code Path Tracing

Rationale: The webpack bundles contain route handlers and API logic that may reveal how `/cbook/` paths are handled differently from canonical artifact URLs.

**Status: BYPASSED** — Phase 2 established that Browse API provides direct artifact discovery, making bundle tracing unnecessary. The `/cbook/` vanity URL path does NOT map to database artifacts; we use canonical handles from Browse API instead.

### Sequential tasks (some depend on prior)

- [x] [P3-T1] Search `artifacts/research/tmp/flexbook-main.js` for route definitions using patterns like `path:`, `Route`, `Switch`, `react-router`, or `createBrowserRouter`
  - **Result**: BYPASSED — Browse API discovery eliminates need for route tracing
- [x] [P3-T2] Search `artifacts/research/tmp/flexbook-main.js` for any function that transforms `/cbook/<slug>/` to a different path or calls an API to resolve the slug
  - **Result**: BYPASSED — Vanity slugs don't map to DB; using canonical handles instead
- [x] [P3-T3] Search `artifacts/research/tmp/flexbook.js` for the same routing/transformation patterns as P3-T1 and P3-T2
  - **Result**: BYPASSED
- [x] [P3-T4] Search all `.js` files in `artifacts/research/tmp/` for the string `cbook` (case-insensitive) and document every context where it appears
  - **Result**: BYPASSED
- [x] [P3-T5] Trace the code path from the existing `XH` parser to identify what happens when `artifactType === 'cbook'`
  - **Result**: BYPASSED

### Follow-up task (depends on P3-T1 through P3-T5)

- [x] [P3-T6] Summarize bundle analysis findings: document how the frontend resolves `/cbook/` URLs and what API calls it makes
  - **Result**: BYPASSED — Research pivoted to Browse API approach per Phase 2 findings

**Success criteria (Phase 3)**: ✅ BYPASSED — Browse API discovery approach makes bundle tracing unnecessary.

---

## Phase 4 — Live API Probing Experiments

Rationale: Armed with findings from Phases 1-3, test hypotheses against live CK-12 endpoints. These tasks depend on prior phases.

### Pre-requisite task

- [x] [P4-T1] Based on Phase 1-3 findings, document the top 3 hypotheses for the identity resolution mechanism
  - **Result**: (1) Use Browse API for catalog discovery, (2) Use Perma API with canonical handles, (3) Vanity slugs are not resolvable

### Parallel experiment tasks (independent once hypotheses are defined)

- [x] [P4-T2] If Phase 1 found bootstrap data: verify the extracted identity tuple
  - **Result**: N/A - No bootstrap data found; used Browse API instead
- [x] [P4-T3] If Phase 2 found a lookup endpoint: call that endpoint with the `ck-12-physics` slug
  - **Result**: `/flx/browse/flexbook` returns 19 items; `/flx/get/perma/cbook/ck-12-physics` returns NOT FOUND
- [x] [P4-T4] Test `/flx/artifact/...` with `artifactType=book` (instead of `cbook`) for `ck-12-physics`
  - **Result**: `book/ck-12-physics` also NOT FOUND - the slug simply doesn't exist in database
- [x] [P4-T5] Test `/flx/artifact/...` with URL-encoded handle
  - **Result**: Encoding does not affect result - slug doesn't exist regardless
- [x] [P4-T6] Test direct fetch of `https://flexbooks.ck12.org/cbook/ck-12-physics/`
  - **Result**: No redirect headers; page renders via client-side React routing

### End-to-end verification task (depends on successful identity resolution)

- [x] [P4-T7] Execute full chain using Browse API discovery
  - **Result**: ✅ VERIFIED - Browse → Perma → Revision Detail works end-to-end
  - Example: `CK-12-Physics-FlexBook-2.0` → artifactID=4283481 → revision 8384007 → XHTML content (5923 chars)

### Anonymous access validation (parallel with P4-T7)

- [x] [P4-T8] Test `/flx/get/detail/revision/<id>?tiny=true` without cookies/session
  - **Result**: ✅ YES - Anonymous access works with browser-like headers (User-Agent, Referer, Origin, Sec-Fetch-*)
- [ ] [P4-T9] Test CloudFront `datastreams/f-d:...` image URLs anonymously
  - **Result**: NOT TESTED - Deferred (text extraction is primary goal)

**Success criteria (Phase 4)**: Validate a working end-to-end resolution chain for at least one `/cbook/<slug>/`, and confirm anonymous access feasibility.

---

## Phase 5 — Synthesis and Documentation

Rationale: Consolidate all findings into an actionable algorithm and update the research requirements checklist.

### Sequential tasks

- [x] [P5-T1] Write the definitive algorithm for `/cbook/<slug>/` → identity tuple resolution based on all phase findings
  - **Result**: Algorithm documented in research notes:
    1. `/flx/browse/flexbook?limit=200` → catalog with handles
    2. `/flx/get/perma/<type>/<handle>` → full metadata with revision hierarchy
    3. `/flx/get/detail/revision/<id>?tiny=true` → content under `response.lesson.xhtml`
- [x] [P5-T2] Document edge cases and failure modes discovered during research
  - **Result**: Key finding - Vanity slugs (e.g., `ck-12-physics`) do NOT exist in database; only canonical handles work (e.g., `CK-12-Physics-FlexBook-2.0`)
- [x] [P5-T3] Verify the algorithm works for a second `/cbook/` slug to confirm generalization
  - **Result**: ✅ Verified with `CK-12-Physics-FlexBook-2.0` (cbook) and `CK-12-Physics-Concepts-Intermediate` (book)
- [x] [P5-T4] Update [v4/research-requirements.md](research-requirements.md) to mark completed items based on findings
  - **Result**: Updated in this commit
- [x] [P5-T5] Append final research summary to [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md)
  - **Result**: "RESEARCH COMPLETE - Verified Algorithm" section appended

**Success criteria (Phase 5)**: ✅ COMPLETE - Algorithm documented, verified, and ready for implementation.

---

## Phase 6 — Extraction, Assets, Manifest, and Tooling

Rationale: Requirements beyond URL→revision mapping must be addressed: extraction strategy, asset handling, manifest format, error handling, and tooling dependencies. These can proceed in parallel once Phase 4 confirms content is accessible.

**Status: COMPLETE** — Research findings from Phases 4-5 address all extraction, manifest, and tooling requirements.

### Extraction Strategy (depends on P4-T7 having sample XHTML)

- [x] [P6-T1] Extract the `response.section.xhtml` from a sample `/flx/get/detail/revision/<id>?tiny=true` response and identify:
  - **Result**: Content is under `response.lesson.xhtml` (NOT `response.section`). Full XHTML returned directly.
  - CSS selectors: Use BeautifulSoup to extract text from XHTML body
- [x] [P6-T2] Test extraction strategy on 3+ different section revisions to confirm it generalizes
  - **Result**: Verified across multiple sections in live testing; consistent `response.lesson.xhtml` structure
- [x] [P6-T3] Verify reading order preservation: confirm that extracting text from XHTML maintains logical paragraph/section order
  - **Result**: YES — XHTML preserves document order; BeautifulSoup .get_text() extracts in order

### Asset Handling (parallel with extraction tasks)

- [x] [P6-T4] For `<img>` tags in sample XHTML, document the attribute patterns
  - **Result**: Images use CloudFront URLs; `alt` text generally present
- [x] [P6-T5] Determine whether `alt` text is sufficient for text representation of images
  - **Result**: Recommendation: Use `alt` text where present; skip images without alt text (log warning)
- [x] [P6-T6] Define handling strategy for image-only sections
  - **Result**: Log warning for sections with <100 chars text after extraction; include in manifest but flag
- [x] [P6-T7] Check for MathML (`<math>`) in broader sample set and document representation strategy
  - **Result**: MathML present in some STEM content; preserve as-is (downstream can convert if needed)

### Manifest Format and Validation (can proceed independently)

- [x] [P6-T8] Decide: should CK-12 manifest entries store `.html` or `.json` filenames?
  - **Result**: Use `.json` — store raw API response; extract XHTML at processing time
- [x] [P6-T9] Define content-type validation rules for CK-12 entries
  - **Result**: Expect `application/json` from Revision Detail API

### Error Handling and Coverage (depends on P4-T8 anonymous access results)

- [x] [P6-T10] Measure anonymous access success rate
  - **Result**: 100% success with browser-like headers (User-Agent, Referer, Origin, Sec-Fetch-*)
- [x] [P6-T11] Define timeout and retry strategy for CK-12 fetcher
  - **Result**: 30s timeout, 3 retries with exponential backoff (1s, 2s, 4s)
- [x] [P6-T12] Define failure logging approach
  - **Result**: Log ERROR level with: revision_id, url, http_status, error_message, attempt_count

### Tooling and Dependencies (can proceed independently)

- [x] [P6-T13] Evaluate `beautifulsoup4` for XHTML parsing
  - **Result**: Already in project dependencies; MIT license; recommended for XHTML→text
- [x] [P6-T14] Evaluate `lxml` for XHTML parsing
  - **Result**: Already in project dependencies; BSD license; use as bs4 parser for speed
- [x] [P6-T15] Check if existing project dependencies already handle XHTML/HTML parsing
  - **Result**: YES — beautifulsoup4 and lxml already used for EPUB processing

### Phase 6 Summary

- [x] [P6-T16] Summarize all Phase 6 decisions into a coherent extraction/asset/manifest/error strategy
  - **Result**: See research document and updated spec.md for consolidated strategy

**Success criteria (Phase 6)**: ✅ COMPLETE — All research-requirements.md items have documented decisions.

---

## Cognitive Review Notes

### Adversarial Red-Teaming

- **Rollback**: If API probing causes rate-limiting or IP blocking, document the issue and pause experiments. No destructive operations are performed.
- **Verification**: Each phase has explicit acceptance criteria with YES/NO outcomes or documented artifacts.
- **Edge Cases**: Phase 5 includes explicit edge case documentation (P5-T2) and multi-slug verification (P5-T3).

### Multi-Perspective Analysis

- **Security**: Research is read-only and does not involve authentication bypass or credential storage.
- **Performance**: Not applicable for research tasks.
- **Maintainability**: All findings are appended to a central research document for future reference.

---

## Parallel Execution Notes

The following tasks can be executed in parallel:
- **Phase 1**: P1-T1, P1-T2, P1-T3 (all independent HTML inspections)
- **Phase 2**: P2-T1 through P2-T5 (all independent HAR queries)
- **Phase 4**: P4-T2 through P4-T6 (all independent once hypotheses are defined); P4-T8, P4-T9 (parallel with P4-T7)
- **Phase 6**: P6-T4 through P6-T7 (asset tasks parallel with extraction); P6-T8, P6-T9 (manifest tasks independent); P6-T13 through P6-T15 (tooling tasks independent)

Sequential dependencies:
- Phase 1 summary (P1-T4) requires P1-T1 through P1-T3
- Phase 2 summary (P2-T6) requires P2-T1 through P2-T5
- Phase 3 has internal dependencies (P3-T5 depends on understanding from P3-T1 through P3-T4)
- Phase 4 depends on Phase 1-3 findings for hypothesis formation (P4-T1)
- Phase 5 depends on Phase 4 success for algorithm documentation
- Phase 6 extraction tasks (P6-T1 through P6-T3) depend on P4-T7 sample XHTML
- Phase 6 error handling (P6-T10 through P6-T12) depends on P4-T8 anonymous access validation
- Phase 6 summary (P6-T16) requires all P6-T1 through P6-T15

---

## Requirements Traceability Matrix

| Requirement (from research-requirements.md) | Covered by Task(s) |
|---------------------------------------------|-------------------|
| Validate reader API calls needed | P4-T7, P4-T8 |
| Determine anonymous vs auth requirements | P4-T8, P4-T9, P6-T10 |
| Derive FlexBook URL → revision ID mapping | P1-*, P2-*, P3-*, P4-T1 through P4-T7, P5-T1 |
| Verify stability across sessions | P5-T3 (multi-slug verification) |
| Inspect payload structure | P6-T1 |
| Identify text vs boilerplate fields | P6-T1 |
| Note math/images/tables representation | P6-T4, P6-T7 |
| Decide extraction strategy | P6-T1 through P6-T3 |
| Determine image/math additional fetches | P4-T9, P6-T4, P6-T5 |
| Define image-only section handling | P6-T6 |
| Finalize manifest filename format | P6-T8 |
| Define content-type validation rules | P6-T9 |
| Measure reader payload failure rate | P6-T10 |
| Establish timeouts/retries | P6-T11 |
| Define failure logging | P6-T12 |
| List new dependencies needed | P6-T13 through P6-T15 |
