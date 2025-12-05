# lexile-calibration-layer — Spec

- Issue: #13
- Owner: drmoisan
- Last Updated: 2025-12-05

## Overview

Calibration layer that fetches Lexile-labeled texts, builds a feature+target dataset from analyzer outputs, trains a regularized regression model, and saves a portable JSON spec for runtime inference.

## Behavior

1) `calibration fetch-texts` reads catalog CSV and ensures `data/calibration/texts/<text_id>.txt` exist (downloads http/gutenberg/local copies, reports pending manual items).
2) `calibration build-dataset` runs analyzer on catalog texts to compute features (MSL/MLF) and produces `calibration_dataset.parquet` (or CSV), filtering low-token/low-slice/NaN rows.
3) `calibration fit` trains ElasticNet/Ridge regression on the dataset, computes metrics (RMSE/MAE/r), and writes `lexile_regression_model.json` with coefficients, intercept, feature list, metrics, and version.
4) All commands are deterministic given same catalog, texts, and seed; emit clear errors when inputs are missing.

## Inputs / Outputs

- Inputs
	- Files: `data/calibration/catalog/lexile_catalog.csv`, `data/calibration/texts/*.txt`, analyzer frequency table
	- CLI flags: `--catalog`, `--texts-root`, `--output` (dataset), `--out` (model), optional HTTP URLs per catalog
- Outputs
	- Texts placed under `data/calibration/texts/`
	- Dataset file `calibration_dataset.parquet` (or CSV) under `data/calibration/`
	- Model spec `data/model/lexile_regression_model.json`
	- Console logs for pending/failed fetches, filtered rows, metrics

## API / CLI Surface

- `lexile-corpus-tuner calibration fetch-texts --catalog <csv> --texts-root <dir>`
- `lexile-corpus-tuner calibration build-dataset --catalog <csv> --texts-root <dir> --output <path>`
- `lexile-corpus-tuner calibration fit <dataset> --out <model.json>`

Example:

```bash
lexile-corpus-tuner calibration fetch-texts --catalog data/calibration/catalog/lexile_catalog.csv --texts-root data/calibration/texts
lexile-corpus-tuner calibration build-dataset --catalog data/calibration/catalog/lexile_catalog.csv --texts-root data/calibration/texts --output data/calibration/calibration_dataset.parquet
lexile-corpus-tuner calibration fit data/calibration/calibration_dataset.parquet --out data/model/lexile_regression_model.json
```

## Data & State

- Calibration texts stored under `data/calibration/texts`
- Dataset artifacts under `data/calibration/`
- Model spec under `data/model/`
- No persistent app state beyond files; commands overwrite outputs deterministically

## Constraints & Risks

- Catalog integrity is critical; missing or malformed rows prevent fetch/build
- Must align with analyzer feature definitions and frequency table version to avoid drift
- ElasticNet config should remain stable; changes require retrain and spec bump
- Legal/ licensing of texts: assume catalog only lists permissible sources

## Definition of Done

- [ ] Behavior matches acceptance criteria (fetch/build/fit deterministic, JSON spec emitted)
- [ ] Tests updated/added (featureset, train, model_store, CLI)
- [ ] Docs updated (README, docs/features/active/... links)
- [ ] Logging covers pending/missing fetches, filtering, and training metrics

