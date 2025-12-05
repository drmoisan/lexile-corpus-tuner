# `lexile-calibration-layer` — User Story

- Issue: #13
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2025-12-05

## Story Statement

- As a data scientist, I want to fit a Lexile-style regression model against official Lexile measures using analyzer features so that runtime estimates align with trusted ground truth.
- As an assessment PM, I want deterministic CLI commands to fetch calibration texts, build the dataset, and fit/save the model so that we can refresh scores on demand without manual glue.

## Problem / Why

Existing heuristics and teacher-model approaches do not align with official Lexile measures. We need a reproducible calibration pipeline that ingests trusted Lexile-labeled texts, computes analyzer features, trains a regression model, and emits a portable JSON spec for production scoring.

## Personas & Scenarios

- Persona: Data scientist
  - Cares about statistically sound, reproducible calibration to official Lexile targets
  - Constraints: offline-friendly, deterministic, no hidden cloud dependencies
  - Goals: build dataset, fit regression, persist JSON model spec for runtime
  - Frustrations: opaque models, missing provenance for features/targets
  - Context: maintaining periodic refresh of model as corpus/analyzer updates

- Scenario: Refresh calibration model
  - Trigger: New batch of Lexile-labeled texts added to catalog
  - Steps: run `calibration fetch-texts` to ensure files exist; run `calibration build-dataset` to generate features+targets; run `calibration fit` to emit updated JSON model spec
  - Obstacles: missing texts, malformed catalog, or analyzer/frequency drift could break reproducibility
  - Outcome: Updated `lexile_regression_model.json` with metrics ready for deployment


## Acceptance Criteria

- [ ] CLI provides `fetch-texts`, `build-dataset`, and `fit` commands with clear required options
- [ ] Calibration dataset builder uses analyzer features and enforces filtering (tokens, slices, NaNs)
- [ ] Model fit produces JSON spec with coefficients, intercept, feature list, metrics
- [ ] Commands are deterministic given same inputs and seeded training

## Non-Goals

- Changing analyzer feature definitions (owned by analyzer layer)
- Adding new model families beyond the specified ElasticNet/Ridge baseline
- Shipping proprietary Lexile texts; assumes catalog references legally usable texts

