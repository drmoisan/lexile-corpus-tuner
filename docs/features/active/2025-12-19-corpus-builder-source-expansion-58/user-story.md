# `2025-12-19-corpus-builder-source-expansion` — User Story

- Issue: #58
- Owner: drmoisan
- Status: In Progress
- Last Updated: 2025-12-23

## Story Statement

- As a **Lexile Model Developer**, I want **to build a corpus from diverse sources with age-based weighting**, so that **the resulting difficulty scores accurately reflect modern reading standards and are not skewed by archaic vocabulary**.

## Problem / Why

The current corpus relies heavily on Project Gutenberg, which introduces a strong bias towards 19th-century vocabulary and sentence structures. This makes the Lexile approximation inaccurate for modern K-12 texts, as "simple" modern words may appear rare in an archaic corpus, and complex archaic structures may be over-represented.

## Personas & Scenarios

- **Persona: Dr. Metrics (Educational Data Scientist)**
  - **Who:** A data scientist responsible for calibrating the text difficulty scoring model.
  - **Goals:** Create a statistically valid model that generalizes well to modern student reading materials (textbooks, YA fiction, news).
  - **Frustrations:** The model currently rates 19th-century children's books as "easy" but modern news articles as "hard" due to vocabulary mismatch.
  - **Constraints:** Limited to public domain or open-license text sources.

- **Scenario: Calibrating for Modern Readers**
  - **Trigger:** Dr. Metrics notices that the model is over-penalizing modern vocabulary.
  - **Action:** She configures the corpus builder to include Simple Wikipedia and OER textbooks alongside Gutenberg. She sets a configuration to down-weight pre-1950 Gutenberg texts (0.3 weight) and up-weight modern sources (1.0 weight).
  - **Steps:**
    1. Run `lexile-scoring-model-pipeline corpus download` to fetch new sources.
    2. Run `lexile-scoring-model-pipeline corpus normalize` to clean and segment.
    3. Run `lexile-scoring-model-pipeline corpus frequencies --weighted` to generate a balanced frequency table.
  - **Outcome:** The generated frequency table shows that words like "internet" and "spacecraft" have appropriate frequencies, while archaic terms like "thou" are reduced in rank. The resulting model scores modern texts more accurately.

## Acceptance Criteria

- [ ] **Multi-Source Ingestion:** Pipeline successfully downloads and normalizes text from Project Gutenberg, Wikipedia (Simple/Standard), and OER sources.
- [ ] **Metadata Tagging:** Every document in the normalized corpus is tagged with `source`, `genre`, and `publication_year` (or estimated era).
- [ ] **Weighted Frequencies:** The frequency calculation step accepts a configuration to apply weights based on source and era (e.g., `gutenberg` < `wikipedia`).
- [ ] **Bias Correction:** Generated frequency tables show a demonstrable reduction in the relative rank of archaic vocabulary compared to a Gutenberg-only baseline.
- [ ] **Configurability:** Weights and source selections are defined in a configuration file, not hardcoded.

## Non-Goals

- **Copyrighted Material:** We will not ingest copyrighted modern fiction or textbooks that are not under an open license (CC, OER, Public Domain).
- **Perfect Dating:** We do not expect exact publication years for all Gutenberg texts; approximate eras or "unknown" handling is acceptable.

## Usage Notes & Risks

- **CLI examples**
  - `poetry run lexile-scoring-model-pipeline corpus download --sources "gutenberg,simple_wiki,oer"`
  - `poetry run lexile-scoring-model-pipeline corpus normalize --sources "gutenberg,simple_wiki,oer"`
  - `poetry run lexile-scoring-model-pipeline corpus frequencies --weighted --config examples/example_config.yaml`
- **Configuration**
  - Weight matrix keys are `weights.<source>.<era_bucket>`; example defaults live in `examples/example_config.yaml`.
  - Required normalized fields: `source_id`, `text_id`, `tokens`, `genre`, `era_bucket`, `intended_audience`; documents missing these are skipped before aggregation.
- **Risks / caveats**
  - Wikipedia dumps and normalized shards can be large; ensure sufficient disk for multi-source runs.
  - Some Gutenberg texts lack reliable publication years; these are bucketed as `unknown` and receive default weights.

