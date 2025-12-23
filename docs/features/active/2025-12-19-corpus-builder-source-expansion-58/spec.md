# corpus-builder-source-expansion — Spec

- Issue: #58
- Owner: drmoisan
- Last Updated: 2025-12-23

## Overview

This feature expands the corpus builder to support multiple data sources (Project Gutenberg, Wikipedia, OER, Modern CC) and implements a weighting mechanism to correct for the historical bias inherent in public domain literature. The goal is to produce a "Lexile-faithful" frequency distribution that approximates the mix of narrative and expository text found in modern K-12 education.

## Behavior

1.  **Ingestion & Normalization:**
    - The pipeline will support distinct handlers for:
        - **Gutenberg:** Existing logic, enhanced with metadata extraction for year/era.
        - **Wikipedia:** Parsers for Simple English and Standard English Wikipedia XML dumps, filtering for article quality and length.
        - **OER/Textbooks:** Ingestion of open educational resources (e.g., from specific OER repositories or datasets).
        - **Modern Fiction:** Ingestion of CC-licensed narrative text.
    - All sources are normalized to a common schema and segmented into comparable "documents" (e.g., 1k-3k word chunks).

2.  **Metadata Enrichment:**
    - Each document is tagged with:
        - `source` (e.g., `gutenberg`, `simple_wiki`)
        - `genre` (`narrative`, `expository`, `instructional`)
        - `publication_year` (or `era_bucket` like `pre-1950`, `modern`)
        - `intended_audience` (inferred from source/metadata)

3.  **Weighted Frequency Analysis:**
    - The `corpus frequencies` command will calculate global word frequencies using a weighted sum.
    - Weights are determined by a configuration mapping `source` and `era` to a scalar value (e.g., `gutenberg:pre-1950` = 0.2).
    - Formula: $Freq(w) = \frac{\sum (count(w, doc) \times weight(doc))}{\sum (length(doc) \times weight(doc))}$

## Inputs / Outputs

- **Inputs:**
    - `config.yaml`: Defines active sources, paths to raw data, and the weighting matrix.
    - **Raw Data:**
        - Gutenberg mirror/cache.
        - Wikipedia XML dumps (bz2).
        - OER text collections.

- **Outputs:**
    - **Normalized Corpus:** Parquet or JSONL files partitioned by source, containing text and metadata.
    - **Frequency Artifacts:** `data/freq/weighted_word_frequencies.tsv` (and metadata JSON).
    - **Corpus Stats:** Report on token counts per source/era before and after weighting.

## API / CLI Surface

- **Download:**
  ```bash
  lexile-scoring-model-pipeline corpus download --sources "gutenberg,simple_wiki,oer"
  ```
- **Normalize:**
  ```bash
  lexile-scoring-model-pipeline corpus normalize
  ```
- **Frequencies:**
  ```bash
  lexile-scoring-model-pipeline corpus frequencies --config config.yaml --weighted
  ```

## Data & State

- **Schema Changes:** The normalized corpus schema must include `source`, `genre`, `year`, `era`, and `weight`.
- **Storage:** Normalized shards will be stored in `data/corpus/normalized/`.
- **State:** No persistent database; relies on file-based artifacts (Parquet/JSONL).

## Constraints & Risks

- **Disk Space:** Wikipedia dumps and uncompressed corpora can be large (>50GB).
- **Memory:** Frequency calculation on a multi-billion word corpus requires efficient streaming or chunked processing (e.g., using Polars or Dask).
- **Data Quality:** Metadata for Gutenberg (publication year) is often missing or messy; fallback strategies are needed.

## Definition of Done

- [ ] Ingestion handlers implemented for Gutenberg, Simple Wiki, and one OER source.
- [ ] Normalization pipeline produces consistent schema with metadata.
- [ ] Frequency calculator supports weighted aggregation.
- [ ] Unit tests for weighting logic and parsers.
- [ ] Documentation updated to explain how to configure sources and weights.

