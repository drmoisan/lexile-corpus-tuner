# lexile-calibration-layer - Plan

- Issue: #13
- Owner: drmoisan
- Last Updated: 2025-12-05

## Required References (read, do not restate)

- Coding workflow and standards: [`docs/code-change.instructions.md`](../../code-change.instructions.md)
- Unit test policy: [`docs/unit-test-policy.md`](../../unit-test-policy.md)

**All work must comply with these policies; do not duplicate their content here.**

## Phases

### Phase 1: Calibration text acquisition [0%]
- [x] Implement `calibration fetch-texts` (catalog-driven, http/local/manual), create texts dir, skip existing
- [x] Log pending manual items and download results
- [x] Tests/docs for this phase

### Phase 2: Dataset build [0%]
- [x] Implement `calibration build-dataset` to run analyzer features on catalog texts and write dataset
- [x] Filter invalid rows (low tokens/slices, NaNs), enforce deterministic outputs
- [x] Tests/docs for this phase

### Phase 3: Model fit & export [0%]
- [x] Implement `calibration fit` training (ElasticNet/Ridge), compute metrics, deterministic seed
- [x] Save JSON model spec with coefficients, intercept, features, metrics
- [x] Tests/docs for this phase

## Test Plan

- Unit
	- [x] Feature engineering (`make_regression_features`) math validated
	- [x] Metrics (`compute_metrics`) correctness and types
	- [x] Model store save/load round-trip for JSON spec
	- [x] Training handles minimal/filtered datasets deterministically

- Integration
	- [x] `calibration fetch-texts` handles required options, skips existing, downloads http
	- [x] `calibration build-dataset` produces dataset from catalog+texts (analyzer mocked) with filtering
	- [x] `calibration fit` trains and emits metrics/spec

- CLI/UX examples
	- [x] Help shows `fetch-texts`, `build-dataset`, `fit`; commands run under CliRunner
	- [x] JSON model output contains expected keys (version, features, coefficients, metrics)

- Performance/edge cases
	- [x] Missing/invalid catalog errors clearly
	- [x] Rows with low tokens/slices or NaNs are filtered
	- [x] Model training uses seeded randomness for reproducibility

## Open Questions / Notes

- Keep catalog schema stable; changes require dataset builder update
- Coordinate analyzer feature changes with retraining and spec bump

