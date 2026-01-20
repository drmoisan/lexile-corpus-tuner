<!-- markdownlint-disable-file -->

# Task Research Notes: CK-12 Missing Handles After Enrichment

## Research Executed

### File Analysis

- src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py
  - Handle assignment for static feed entries uses `handle` or URL-derived slug.
  - URL-derived slug is only extracted for `/cbook/` and `/book/` path patterns.
  - Earlier investigation notes: for `is_static_feed_entry` (default), `entry.handle` originally relied only on the `handle` JSON field, while `identifier` could fall back to the `Content_URL` slug. This mismatch explains why identifiers exist while `handle` is missing in some rows.
- src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py
  - Entries with `handle=None` are skipped with `missing handle`.
  - Earlier investigation notes: enrichment explicitly skips entries with missing handles, which is consistent with the log lines you observed.

### Code Search Results

- "missing handle"
  - `ck12_enrichment.py` logs the skip reason when `entry.handle` is missing.
- "extract_slug_from_content_url"
  - Only `/cbook/` and `/book/` URL patterns are supported.

### External Research

- #fetch:https://static.ck12.org/testimonial/fbbrowse-prod.json
  - Feed contains `Content_URL` values using additional path patterns such as `/tebook/`, `/workbook/`, and `/quizbook/`.

### Project Conventions

- Standards referenced: `docs/source-curation-guide.md` (CK-12 step sequence)
- Instructions followed: research-only mode, artifacts-only edits

## Key Discoveries

### Project Structure

- Catalog produces 165 rows in `ck12_catalog.jsonl`.
- Enrichment produced 133 enriched entries; 32 entries skipped due to missing handles.
- Duplicate identifier check: 0 duplicates in `ck12_catalog.jsonl` (no evidence that dedup order caused missing handles).

### Implementation Patterns

- `ck12_enrichment` requires `entry.handle` to call the Perma API and explicitly skips rows when it is missing:
  - See `ck12_enrichment.py`, loop in `enrich_ck12_catalog`:
    - `handle = entry.handle`
    - `if handle is None: ... continue`
    - The skip reason shown in your logs is emitted here.
- `ck12_catalog` derives `entry.handle` only from fields that exist in the catalog JSON:
  - `entry_handle` assignment in `parse_catalog_json` (near the `entry_handle` block) uses:
    - `handle_raw` when present.
    - `content_url_slug` when present.
    - Otherwise `None`.
- `content_url_slug` only works when the URL parser recognizes **specific path prefixes**:
  - `extract_slug_from_content_url` explicitly lists support for `/cbook/` and `/book/` patterns.
  - It does **not** recognize `/tebook/`, `/workbook/`, or `/quizbook/` paths, which appear in the feed.

### Logic Defect Summary (Merged)

- The catalog stage can produce valid identifiers via `content_url_slug`, while still leaving `entry.handle` as `None`.
- This creates a direct mismatch: identifiers are present, but the enrichment stage requires `entry.handle` to fetch Perma metadata, resulting in skips.

### Complete Examples

```text
Missing handle examples from the feed:
- https://www.ck12.org/tebook/CK-12-Earth-Science-For-Middle-School-Teachers-Edition/
- https://www.ck12.org/workbook/CK-12-Earth-Science-For-Middle-School-Workbook/
- https://www.ck12.org/quizbook/CK-12-Earth-Science-For-Middle-School-Quizzes-and-Tests/
```

### API and Schema Documentation

- The CK-12 browse feed provides `Content_URL` but often omits `handle` fields.

### Configuration Examples

```json
{
  "Title": "CK-12 Earth Science For High School Teacher's Edition",
  "Content_URL": "https://www.ck12.org/tebook/CK-12-Earth-Science-For-High-School-Teachers-Edition/"
}
```

### Technical Requirements

- Entries must have `handle` values to avoid being skipped during enrichment.
- The URL slug extraction should recognize additional CK-12 URL path patterns.

**Mandatory unachievable objective callout**:
- None identified.

## Recommended Approach

Expand URL slug extraction so `entry.handle` is populated for **all** CK-12 `Content_URL` patterns that appear in the browse feed. This removes the 32-entry gap created by handle-less rows.

**Evidence-backed rationale:**
- The feed includes URLs with `/tebook/`, `/workbook/`, and `/quizbook/` prefixes (see examples above).
- `extract_slug_from_content_url` currently only recognizes `/cbook/` and `/book/`.
- Because `entry.handle` depends on `content_url_slug`, these rows remain `None` and get skipped during enrichment.

**What to change (scope):**
- Update `extract_slug_from_content_url` in `ck12_catalog.py` to recognize the additional prefixes when `parsed.netloc == "www.ck12.org"`.
- Keep identifier logic unchanged; only expand slug extraction to more URL patterns.

**Earlier Proposed Change (Retained, De-duplicated):**
The earlier research suggested explicitly adding a `content_url_slug` fallback to `entry_handle`. That logic is now present in the file, so the remaining gap is to support more URL path prefixes in `extract_slug_from_content_url`.

## Implementation Guidance

- **Objectives**: Ensure all catalog entries have handles when `Content_URL` is present.
- **Key Tasks**: Extend URL parsing to accept additional CK-12 path prefixes; rerun catalog → enrichment.
- **Dependencies**: None.
- **Success Criteria**: Enrichment produces ~165 entries with minimal `missing handle` skips.
  - Prior interim success criteria: rerunning `ck12_catalog` and `ck12_enrichment` should produce enriched entries instead of all being skipped.
