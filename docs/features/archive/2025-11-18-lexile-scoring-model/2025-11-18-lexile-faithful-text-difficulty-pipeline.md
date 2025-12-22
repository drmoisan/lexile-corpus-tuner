# Lexile-Faithful Text Difficulty Pipeline (Issue #10)

- Date captured: 2025-11-18
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/Lexile-Faithful Text Difficulty Pipeline/ (Issue #10)
- Issue: #10
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/10

## Problem / Why

The existing Lexile estimators in `lexile_corpus_tuner` use either heuristic approaches (DummyLexileEstimator) or external ML models (lexile-determination-v2). Neither approach replicates the **official MetaMetrics Lexile framework** methodology, which relies on:

1. **Mean Sentence Length (MSL)** - Syntactic complexity measure
2. **Mean Log Word Frequency (MLF)** - Semantic difficulty based on a large reference corpus (~600M-1.4B words) scaled to per-5-million-word rates
3. **Text slicing** - Breaking texts into ≥125-word slices extended to sentence boundaries
4. **Regression calibration** - Fitting features to official Lexile measures from known texts

This creates a gap: users who want **Lexile-faithful text difficulty estimates** aligned with the actual MetaMetrics approach cannot achieve this with current tooling. The project needs a purpose-built pipeline that replicates the Lexile methodology as closely as practical using open-source components.

## Proposed Behavior

A new three-layer pipeline architecture that implements the Lexile framework methodology:

**Layer 1: Corpus Builder**

- Build a large proxy MetaMetrics corpus from open sources (Gutenberg, Simple English Wikipedia, OpenStax/CK-12 OER)
- Apply consistent normalization and tokenization across all sources
- Compute global word frequencies and log frequencies per 5M words
- Save frequency tables (TSV) with metadata for runtime use

**Layer 2: Lexile-Style Analyzer**

- Slice texts using ≥125-word rule extended to sentence boundaries (matching MetaMetrics)
- Compute MSL (mean sentence length) and MLF (mean log word frequency) per slice and document
- Apply special-case adjustments (picture books, emergent nonfiction)
- Generate DocumentFeatures with slice-level and aggregate statistics

**Layer 3: Regression Calibration**

- Build calibration catalog (`lexile_catalog.csv`) with texts that have official Lexile measures
- Automatically fetch texts from Gutenberg, HTTP URLs, or use local files
- Analyze calibration texts to extract features
- Fit regression model (features → official Lexile measures) using sklearn
- Export trained model as JSON specification for runtime use (no sklearn dependency at runtime)

**CLI Interface**: New `lexile-scoring-model-pipeline` command with subcommands (keep `text-difficulty-pipeline` as a legacy alias):

- `corpus download` - Fetch Gutenberg subset, Simple Wiki dump, OER sources
- `corpus normalize` - Normalize and shard corpus texts
- `corpus frequencies` - Compute word frequencies from shards
- `analyze slices` - Slice a text and compute features
- `analyze features` - Extract MSL/MLF features from text
- `analyze estimate` - Estimate Lexile using calibrated model
- `calibration fetch-texts` - Fetch calibration catalog texts
- `calibration build-dataset` - Analyze calibration texts to build feature table
- `calibration fit-model` - Train regression model against official Lexile measures

## Acceptance Criteria (early draft)

### Corpus Layer

- [ ] Corpus download module fetches Gutenberg subset, Simple Wiki dump, and OER sources
- [ ] Normalization module creates shards of ~100k tokens with consistent tokenization
- [ ] Frequency computation generates word_frequencies.tsv with counts, per-5M rates, and log frequencies
- [ ] Frequency loader reads TSV and provides lookup for MLF computation

### Analyzer Layer

- [ ] Sentence splitter handles `.?!;` boundaries with deterministic behavior
- [ ] Slicer creates ≥125-word slices extended to sentence boundaries
- [ ] Feature extractor computes per-slice MSL and MLF
- [ ] Document-level aggregation produces overall MSL/MLF statistics
- [ ] Special-case adjustments support picture book and emergent nonfiction flags

### Calibration Layer

- [ ] Calibration catalog structure defined with text_id, title, lexile_official, acquisition_type, etc.
- [ ] `fetch-texts` command auto-downloads from Gutenberg and HTTP URLs
- [ ] `build-dataset` command analyzes texts and exports features to Parquet
- [ ] `fit-model` command trains regression and saves JSON model spec
- [ ] Runtime estimator loads JSON spec and computes Lexile estimates without sklearn

### Integration

- [ ] All modules use shared textutils.py for consistent normalization/tokenization
- [ ] CLI provides full workflow from corpus download → frequency tables → calibration → estimation
- [ ] Documentation explains how to add new corpus sources and calibration texts
- [ ] All code passes Black, Ruff, Pyright, and Pytest checks

## Constraints & Risks

**Data Volume & Performance**

- Large corpus (hundreds of MB to few GB) requires efficient streaming and sharding
- Frequency table loading must be fast enough for runtime use
- Consider memory constraints when processing full Simple Wiki dump

**Corpus Quality**

- Proxy corpus will differ from MetaMetrics' actual corpus (which includes proprietary textbooks)
- Open sources skew toward public domain (older) and certain genres
- Mitigation: Use source weights to balance contemporary vs. archaic language

**Calibration Data Acquisition**

- Official Lexile measures for known texts may be limited (proprietary data)
- Manual curation required for many calibration points
- Legal constraints on using copyrighted texts even with known Lexile scores
- Mitigation: Focus on public domain and OER texts with published Lexile measures

**Model Fidelity**

- Regression model will approximate but not perfectly replicate MetaMetrics approach
- They may use additional features or proprietary adjustments not documented
- Risk: Estimates may diverge from official Lexile measures, especially at extremes

**Maintenance & Dependencies**

- Sklearn required for training but not runtime (JSON spec approach)
- Corpus sources may change URLs or availability over time
- Frequency tables must be regenerated if tokenization logic changes

**Scope Creep**

- Feature is large (three layers with many modules)
- Risk of expanding to include additional text difficulty frameworks
- Mitigation: Stay focused on Lexile methodology; other frameworks are separate features

## Test Conditions to Consider

### Corpus Layer

- [ ] Download idempotency (skip existing files)
- [ ] Normalization with various Unicode edge cases (accents, ligatures, non-Latin scripts)
- [ ] Shard size boundaries (last shard handling, token count accuracy)
- [ ] Frequency computation with weighted sources
- [ ] Frequency loader with missing words (return default log freq)

### Analyzer Layer

- [ ] Sentence splitting with edge cases (abbreviations, ellipses, quoted speech)
- [ ] Slice construction with texts <125 words, exactly 125 words, and long texts
- [ ] MSL computation with single-word sentences and very long sentences
- [ ] MLF computation with all-known-words vs. mix of rare/unknown words
- [ ] Special adjustments (verify picture book and emergent nonfiction modifiers)

### Calibration Layer

- [ ] Catalog parsing with various acquisition types (gutenberg, http, local, manual)
- [ ] fetch-texts with successful downloads, missing files, HTTP errors
- [ ] build-dataset round-trip (CSV → features → Parquet → reload)
- [ ] fit-model with small dataset (avoid overfitting), cross-validation scores
- [ ] JSON model serialization/deserialization
- [ ] Runtime estimation with various feature inputs

### Integration

- [ ] End-to-end workflow: download → normalize → frequencies → calibration → estimation
- [ ] CLI error handling for missing corpus, missing frequency table, missing model
- [ ] Tokenization consistency across corpus building and runtime analysis
- [ ] Deterministic results (same input → same features → same estimate)

## Next Step

- [ ] Review with stakeholders to confirm scope and priorities
- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/lexile-faithful-text-difficulty-pipeline/` folder from the template
- [ ] Break into milestones: Corpus Layer → Analyzer Layer → Calibration Layer
- [ ] Identify minimal calibration catalog to bootstrap training (20-50 texts with known Lexile measures)
