# populate-open-stax-ck-12-manifest (Issue #73)

- Date captured: 2026-01-06
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/populate-open-stax-ck-12-manifest/ (Issue #73)

- Issue: #73
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/73
- Last Updated: 2026-01-06
## Problem / Why

The current `oer_sources.json` manifest structure requires manual population, which is unscalable and prone to errors. The existing `source-curation-guide.md` describes a manifest shape that doesn't match the active OER pipeline (which requires `source_id`, `id`, `url`, and explicit `filename`). Furthermore, the OER normalization pipeline currently only accepts `.txt` or `.jsonl` files, ignoring PDF/EPUBs unless a conversion step is added. We need a programmatic way to discover ("catalog"), filter ("curate"), and register ("manifest") K-12 OER titles (OpenStax, CK-12) that are compatible with the existing text-based pipeline.

## Proposed Behavior

Implement a "Gutenberg-analog" workflow to discover and configure OER sources via the Internet Archive (IA):

1.  **Catalog Generation**: Create scripts to query the IA Advanced Search API for "OpenStax" and "CK-12" collections (focusing on K-12 scope, all subjects).
2.  **Enrichment**: For each discovered item, query the IA Metadata API to identify direct text download candidates (specifically `_djvu.txt` or similar plain-text derivatives).
3.  **Curation**: Filter the catalog for availability of text formats.
    *   *Note*: Extraction from PDF/EPUB is deferred until strictly necessary to access key K-12 materials not available as text.
4.  **Manifest Emission**: Generate `data/meta/oer_sources.json` where each entry includes:
    *   `source_id`: "openstax" or "ck12"
    *   `id`: stable slug (e.g., `openstax-algebra-trig`)
    *   `url`: direct download link to the text asset.
    *   `filename`: explicit filename ending in `.txt` (required for the normalizer).

## Acceptance Criteria (early draft)

- [ ] Script created to fetch OpenStax/CK-12 metadata from IA Advanced Search and save to `data/meta/catalogs/*_catalog.jsonl`.
- [ ] Catalog logic enriches entries with valid `download_candidates`, prioritizing `_djvu.txt`.
- [ ] Curation logic produces `oer_sources.json` populated with K-12 titles across all subjects.
- [ ] Generated manifest entries contain valid `source_id`, `id`, `url`, and `filename` fields.
- [ ] `filename` fields in manifest strictly enforce `.txt` extension.
- [ ] `lexile-scoring-model-pipeline corpus download` successfully fetches the curated text files to the correct raw directories.
- [ ] `lexile-scoring-model-pipeline corpus normalize` successfully ingests the downloaded text files.

## Constraints & Risks

-   **Pipeline Compatibility**: Strict dependency on `.txt` format means sources without text derivatives on IA will be omitted in this pass (unless extraction is added).
-   **Text Quality**: IA `_djvu.txt` files are OCR-derived and may have formatting noise compared to raw source HTML, potentially affecting Corpus Tuner metrics.
-   **Missing Coverage**: Some newer or specific K-12 titles might not have clean text derivatives on IA.
-   **Licensing**: Confirmed out of scope for LLM training restrictions (metrics only), but attribution metadata should still be captured.

## Test Conditions to Consider

- [ ] **Unit**: Parser tests for IA API responses; Slug generation logic; Manifest validation (schema compliance).
- [ ] **Integration**: End-to-end run of catalog-fetch -> curate -> manifest-gen -> download-sample.
- [ ] **Data Validation**: Check that generated URLs do not return 403 or HTML login pages.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/populate-open-stax-ck-12-manifest/` folder from the template
