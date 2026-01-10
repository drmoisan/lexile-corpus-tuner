# CK-12 `/cbook/<slug>/` → Artifact Identity → Revision ID Mapping

## Research Plan (Atomic Tasks)

**Objective**: Determine a deterministic, unauthenticated algorithm to map a public FlexBook URL like `https://flexbooks.ck12.org/cbook/ck-12-physics/` to the internal `(artifactType, artifactHandle, artifactCreator, revision)` tuple and ultimately to the revision IDs required by `/flx/get/detail/revision/<id>?tiny=true`.

**Output location**: All findings append to [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md).

---

## Phase 0 — Context & Inputs

- [ ] [P0-T1] Read `.github/copilot-instructions.md` and `.github/instructions/general-code-change.instructions.md` to confirm research-only output rules (write to `artifacts/research/` only)
- [ ] [P0-T2] Review the current state of [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md) to understand prior findings and avoid duplicate work
- [ ] [P0-T3] Review [v4/research-requirements.md](research-requirements.md) to ensure all unchecked requirements are addressed by this plan

**Success criteria (Phase 0)**: Researcher understands existing knowledge base and knows which questions remain open.

---

## Phase 1 — Quick Wins: HTML Bootstrap Data Inspection

Rationale: The downloaded `/cbook/<slug>/` HTML pages may contain embedded JSON with the artifact identity already resolved. This is the fastest path to a solution.

### Parallel tasks (independent)

- [ ] [P1-T1] Inspect `artifacts/research/tmp/ck-12-physics.html` for `<script>` tags containing JSON with keys like `artifactID`, `artifactRevisionID`, `revisionID`, `handle`, `artifactType`, or `ck12editor`
  - Document: exact script tag location, JSON structure, and values found
  - If found: extract the identity tuple and record it
- [ ] [P1-T2] Inspect `artifacts/research/tmp/book.html` for embedded JSON bootstrap data using the same key search as P1-T1
  - Document: whether this file represents a different page type and what data it contains
- [ ] [P1-T3] Inspect `artifacts/research/tmp/reader-index.html` for embedded JSON bootstrap data using the same key search as P1-T1
  - Document: whether this is a catalog/index page with multiple artifact references

### Follow-up task (depends on P1-T1 through P1-T3)

- [ ] [P1-T4] Summarize HTML inspection findings: record whether bootstrap data exists, its JSON schema, and whether it provides a complete identity tuple
  - Acceptance: A clear YES/NO answer with evidence for each HTML file

**Success criteria (Phase 1)**: Determine if embedded bootstrap data provides the identity tuple directly.

---

## Phase 2 — HAR Network Analysis

Rationale: The recorded `artifacts/www.ck12.org.har` (~75MB, 1271 entries) captures actual browser network activity and may reveal the lookup sequence or redirect chain.

### Parallel tasks (independent)

- [ ] [P2-T1] Parse `artifacts/www.ck12.org.har` and list all unique request URLs matching pattern `/flx/artifact` (any variant)
  - Document: exact URLs, query parameters, request/response status
  - Acceptance: Table of all `/flx/artifact/*` calls with parameters
- [ ] [P2-T2] Parse `artifacts/www.ck12.org.har` and list all unique request URLs matching pattern `/flx/get/` (excluding `/flx/get/detail/revision/`)
  - Document: any discovery/lookup endpoints not yet known
  - Acceptance: Table of all `/flx/get/*` calls (non-revision) with parameters
- [ ] [P2-T3] Parse `artifacts/www.ck12.org.har` for any 3xx redirect responses from `/cbook/` paths
  - Document: original URL, redirect target, and whether target contains canonical artifact path
  - Acceptance: YES/NO for redirect presence with redirect chain if found
- [ ] [P2-T4] Parse `artifacts/www.ck12.org.har` and extract any request/response containing JSON with `artifactID` or `artifactRevisionID` fields (excluding already-documented `detail/revision` calls)
  - Document: endpoint URL, response JSON structure, and identity values
  - Acceptance: List of endpoints that return artifact identity data
- [ ] [P2-T5] Parse `artifacts/www.ck12.org.har` for any requests to `/flx/show/` or `/flx/search/` or `/flx/browse/` patterns
  - Document: whether alternative discovery APIs exist
  - Acceptance: YES/NO with endpoint details if found

### Follow-up task (depends on P2-T1 through P2-T5)

- [ ] [P2-T6] Summarize HAR analysis findings: document the request sequence for a `/cbook/` page load, identify the key lookup call(s), and extract the identity resolution algorithm
  - Acceptance: Flow diagram or step list showing URL → identity tuple → revision IDs

**Success criteria (Phase 2)**: Identify the network call(s) that resolve `/cbook/<slug>/` to artifact identity.

---

## Phase 3 — Bundle Code Path Tracing

Rationale: The webpack bundles contain route handlers and API logic that may reveal how `/cbook/` paths are handled differently from canonical artifact URLs.

### Sequential tasks (some depend on prior)

- [ ] [P3-T1] Search `artifacts/research/tmp/flexbook-main.js` for route definitions using patterns like `path:`, `Route`, `Switch`, `react-router`, or `createBrowserRouter`
  - Document: routing configuration structure and `/cbook/` route handler if present
  - Acceptance: Route table or routing logic excerpt
- [ ] [P3-T2] Search `artifacts/research/tmp/flexbook-main.js` for any function that transforms `/cbook/<slug>/` to a different path or calls an API to resolve the slug
  - Document: function name, input/output, and API call if any
  - Acceptance: Code excerpt showing the transformation logic
- [ ] [P3-T3] Search `artifacts/research/tmp/flexbook.js` for the same routing/transformation patterns as P3-T1 and P3-T2
  - Document: any additional routing logic not in flexbook-main.js
  - Acceptance: YES/NO with code excerpts if found
- [ ] [P3-T4] Search all `.js` files in `artifacts/research/tmp/` for the string `cbook` (case-insensitive) and document every context where it appears
  - Document: file, line context, and purpose of each occurrence
  - Acceptance: Table of all `cbook` references with categorization (routing, API, experiment, etc.)
- [ ] [P3-T5] Trace the code path from the existing `XH` parser to identify what happens when `artifactType === 'cbook'`
  - Document: whether `cbook` is treated specially or falls through to default handling
  - Acceptance: Code path analysis with branch conditions

### Follow-up task (depends on P3-T1 through P3-T5)

- [ ] [P3-T6] Summarize bundle analysis findings: document how the frontend resolves `/cbook/` URLs and what API calls it makes
  - Acceptance: Algorithm description or flowchart

**Success criteria (Phase 3)**: Understand the frontend's `/cbook/` handling logic completely.

---

## Phase 4 — Live API Probing Experiments

Rationale: Armed with findings from Phases 1-3, test hypotheses against live CK-12 endpoints. These tasks depend on prior phases.

### Pre-requisite task

- [ ] [P4-T1] Based on Phase 1-3 findings, document the top 3 hypotheses for the identity resolution mechanism
  - Acceptance: Numbered hypothesis list with testable predictions

### Parallel experiment tasks (independent once hypotheses are defined)

- [ ] [P4-T2] If Phase 1 found bootstrap data: verify the extracted identity tuple by calling `/flx/artifact/...` with those exact parameters
  - Document: request URL, response status, and whether artifact metadata is returned
  - Acceptance: HTTP 200 with valid artifact JSON, or document the error
- [ ] [P4-T3] If Phase 2 found a lookup endpoint: call that endpoint with the `ck-12-physics` slug and document the response
  - Document: request URL, headers, response status, and identity data returned
  - Acceptance: Identity tuple extracted from response or error documented
- [ ] [P4-T4] Test `/flx/artifact/...` with `artifactType=book` (instead of `cbook`) for `ck-12-physics`
  - Document: request URL, response status, and whether this alternate type works
  - Acceptance: HTTP 200 with valid artifact JSON, or document the error
- [ ] [P4-T5] Test `/flx/artifact/...` with URL-encoded handle (e.g., `ck-12-physics` vs `ck%2D12%2Dphysics`)
  - Document: whether encoding affects the lookup
  - Acceptance: Comparison table of encoded vs non-encoded results
- [ ] [P4-T6] Test direct fetch of `https://flexbooks.ck12.org/cbook/ck-12-physics/` and inspect response headers for `Location` redirect or `X-Artifact-ID` style headers
  - Document: HTTP status, relevant headers, and any redirect target
  - Acceptance: Header table with values

### End-to-end verification task (depends on successful identity resolution)

- [ ] [P4-T7] Once a working identity tuple is found, execute the full chain: `/flx/artifact/...` → extract `artifactRevisionID` → `/flx/get/detail/revision/<id>?tiny=true`
  - Document: each request/response in sequence
  - Acceptance: Successfully retrieve chapter/section XHTML content

### Anonymous access validation (parallel with P4-T7)

- [ ] [P4-T8] Test `/flx/get/detail/revision/<id>?tiny=true` **without** cookies/session, using only browser-like headers
  - Document: HTTP status, whether content is returned, any auth-required errors
  - Acceptance: YES/NO for anonymous access with specific headers documented
- [ ] [P4-T9] Test CloudFront `datastreams/f-d:...` image URLs anonymously (from `<img src>` in XHTML)
  - Document: HTTP status, content-type, whether images are accessible
  - Acceptance: YES/NO for anonymous image fetch

**Success criteria (Phase 4)**: Validate a working end-to-end resolution chain for at least one `/cbook/<slug>/`, and confirm anonymous access feasibility.

---

## Phase 5 — Synthesis and Documentation

Rationale: Consolidate all findings into an actionable algorithm and update the research requirements checklist.

### Sequential tasks

- [ ] [P5-T1] Write the definitive algorithm for `/cbook/<slug>/` → identity tuple resolution based on all phase findings
  - Document: step-by-step algorithm with exact API calls, headers, and parameter construction
  - Acceptance: Pseudocode or Python-like implementation sketch
- [ ] [P5-T2] Document edge cases and failure modes discovered during research
  - Document: what happens with invalid slugs, removed books, authentication requirements
  - Acceptance: Edge case table with expected behavior
- [ ] [P5-T3] Verify the algorithm works for a second `/cbook/` slug (different from `ck-12-physics`) to confirm generalization
  - Document: second slug tested, results, and any variations needed
  - Acceptance: Two confirmed working examples
- [ ] [P5-T4] Update [v4/research-requirements.md](research-requirements.md) to mark completed items based on findings
  - Document: which requirements are now satisfied
  - Acceptance: Checkboxes updated with evidence links
- [ ] [P5-T5] Append final research summary to [artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md](../../../../../artifacts/research/20260109-ck12-slug-to-revision-mapping-research.md)
  - Document: complete algorithm, examples, and remaining gaps (if any)
  - Acceptance: Research notes are self-contained and actionable for implementation

**Success criteria (Phase 5)**: A documented, verified algorithm ready for implementation.

---

## Phase 6 — Extraction, Assets, Manifest, and Tooling

Rationale: Requirements beyond URL→revision mapping must be addressed: extraction strategy, asset handling, manifest format, error handling, and tooling dependencies. These can proceed in parallel once Phase 4 confirms content is accessible.

### Extraction Strategy (depends on P4-T7 having sample XHTML)

- [ ] [P6-T1] Extract the `response.section.xhtml` from a sample `/flx/get/detail/revision/<id>?tiny=true` response and identify:
  - Main content container (e.g., `<div class="x-ck12-data-lesson">`)
  - Navigation/boilerplate elements to strip
  - Document: CSS selectors or XPath for content extraction
  - Acceptance: Documented selector strategy that isolates body text
- [ ] [P6-T2] Test extraction strategy on 3+ different section revisions to confirm it generalizes
  - Document: any edge cases (empty content, different div structures)
  - Acceptance: Strategy works on all test samples or edge cases documented
- [ ] [P6-T3] Verify reading order preservation: confirm that extracting text from XHTML maintains logical paragraph/section order
  - Document: any re-ordering needed or issues found
  - Acceptance: YES/NO with evidence

### Asset Handling (parallel with extraction tasks)

- [ ] [P6-T4] For `<img>` tags in sample XHTML, document the attribute patterns:
  - `src` (CloudFront datastreams URL)
  - `alt` (alt text availability)
  - `data-flx-url` (internal fetch route)
  - `title`, `longdesc` (caption/description availability)
  - Acceptance: Attribute table with presence frequency across samples
- [ ] [P6-T5] Determine whether `alt` text is sufficient for text representation of images, or if additional metadata fetch is needed
  - Document: sample alt texts, quality assessment
  - Acceptance: Recommendation (use alt / fetch caption / skip)
- [ ] [P6-T6] Define handling strategy for image-only sections (sections with no substantive text, only images)
  - Document: detection heuristic (e.g., text length < N chars) and handling (log + placeholder vs. skip)
  - Acceptance: Documented decision with rationale
- [ ] [P6-T7] Check for MathML (`<math>`) in broader sample set (HAR or live fetch) and document representation strategy
  - Document: whether MathML present, whether to convert to LaTeX/text or preserve as-is
  - Acceptance: MathML handling decision documented

### Manifest Format and Validation (can proceed independently)

- [ ] [P6-T8] Decide: should CK-12 manifest entries store `.html` or `.json` filenames?
  - Consider: downstream pipeline expects what format? consistency with other sources?
  - Document: decision and rationale
  - Acceptance: Clear recommendation with justification
- [ ] [P6-T9] Define content-type validation rules for CK-12 entries
  - Document: expected MIME types (`application/json` for API response, `text/html` if we extract and save XHTML)
  - Acceptance: Validation rule specification

### Error Handling and Coverage (depends on P4-T8 anonymous access results)

- [ ] [P6-T10] Measure anonymous access success rate: attempt `/flx/get/detail/revision/<id>?tiny=true` for 10+ revision IDs from HAR without auth
  - Document: success count, failure count, error types
  - Acceptance: Success rate percentage with failure categorization
- [ ] [P6-T11] Define timeout and retry strategy for CK-12 fetcher
  - Document: recommended timeout (e.g., 30s), retry count (e.g., 2), backoff strategy
  - Acceptance: Strategy specification aligned with existing pipeline patterns
- [ ] [P6-T12] Define failure logging approach: what to log when a revision fetch fails
  - Document: log level, fields to include (URL, status, error message, retry count)
  - Acceptance: Logging specification

### Tooling and Dependencies (can proceed independently)

- [ ] [P6-T13] Evaluate `beautifulsoup4` for XHTML parsing: license, size, existing usage in project
  - Document: license (MIT), bundle size impact, whether already a dependency
  - Acceptance: Recommendation (use / don't use / already present)
- [ ] [P6-T14] Evaluate `lxml` for XHTML parsing (alternative to bs4): license, size, performance
  - Document: comparison with bs4
  - Acceptance: Recommendation
- [ ] [P6-T15] Check if existing project dependencies already handle XHTML/HTML parsing (e.g., for EPUB)
  - Document: existing parsing utilities in `src/lexile_corpus_tuner/`
  - Acceptance: YES/NO with module paths if found

### Phase 6 Summary

- [ ] [P6-T16] Summarize all Phase 6 decisions into a coherent extraction/asset/manifest/error strategy
  - Document: consolidated decision record
  - Acceptance: Single-page strategy document ready for implementation

**Success criteria (Phase 6)**: All research-requirements.md items have documented decisions or findings.

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
