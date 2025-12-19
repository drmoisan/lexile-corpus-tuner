# lexile-scoring-model-pipeline — Initiative Overview

- Issue: #10
- Owner: drmoisan
- Last Updated: 2025-12-06

## Goal & Outcomes

Deliver a Lexile-faithful text difficulty pipeline that mirrors the MetaMetrics methodology using open-source components: corpus-derived word frequencies, Lexile-style analyzer (MSL/MLF on ≥125-word slices), and calibrated regression against official Lexile measures, with deterministic CLIs and portable artifacts (frequency table, model JSON).

## Decomposition (Child Features)

- Corpus Builder Layer (Issue #11) - `../lexile-corpus-builder-layer/`
- Analyzer Layer (Issue #12) - `../lexile-analyzer-layer/`
- Calibration Layer (Issue #13) - `../lexile-calibration-layer/`
- Pipeline Structure Alignment (Issue #15) - `../lexile-refactor-pipeline-structure/`

Dependencies: Analyzer requires corpus frequencies; Calibration requires analyzer features; execute in order #11 → #12 → #13.

## Cross-Cutting Constraints & Assumptions

- Shared normalization/tokenization (`textutils.normalize_text` / `iter_tokens`) for both corpus and analyzer paths; frequency table version must align with analyzer runs.
- Deterministic, seed-stable outputs; reruns on identical inputs produce identical features and estimates.
- Data paths and artifacts: `data/corpus/*`, `data/freq/word_frequencies.tsv`, `data/calibration/*`, `data/model/lexile_regression_model.json` (all git-ignored).
- Runtime inference is sklearn-free; training uses ElasticNet/Ridge then exports JSON spec (coefficients, intercept, metrics, feature order).
- Legal/compliance: calibration texts and corpus sources must be permissible (public domain/OER/owned); no proprietary Lexile content stored.
- Quality gates: Black/Ruff/Pyright/Pytest must pass for all layers.

## Milestones & Status

- M1 Corpus frequencies available (download → normalize → frequencies) — ✔ Implemented
- M2 Analyzer features available (slices, MSL/MLF, adjustments) — ✔ Implemented
- M3 Calibrated model exported (dataset build, fit, JSON spec) — ✔ Implemented
- CLI alignment: `lexile-corpus-tuner` exposes `corpus`, `analyze`, and `calibration` subcommands; end-to-end pipeline reachable via `lexile-scoring-model-pipeline` workflow (alias: `text-difficulty-pipeline`).

## Initiative-Level Validation

- End-to-end CLI: `corpus download` → `corpus normalize` → `corpus frequencies` → `calibration build-dataset` → `calibration fit` → `analyze text` returns stable Lexile estimate.
- Integration checks: analyzer uses the same frequency table version; calibration dataset features match analyzer feature definitions; model JSON loads in runtime estimator.
- Determinism: repeated runs with same inputs, seeds, and artifacts yield identical frequencies, features, and estimates.
- Error handling: clear failures for missing frequency table, missing model spec, missing catalog/texts; corpus and calibration commands are idempotent where applicable.
- Regression guardrails: metrics (RMSE/MAE/r) recorded in model spec; drift detected by re-fitting and comparing metrics against prior specs.
