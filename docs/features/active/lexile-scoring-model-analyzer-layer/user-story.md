# `lexile-analyzer-layer` — User Story

- Issue: #12
- Owner: drmoisan
- Status: Draft | In Progress | Complete
- Last Updated: 2025-12-05

## Story Statement

- As a curriculum data engineer, I want a Lexile-style analyzer that slices text and computes MSL/MLF using the same normalization and frequency table as the corpus layer so that downstream calibration and scoring are reproducible.
- As an assessment PM, I want deterministic CLI commands to analyze a text file and return slice- and document-level features so that I can audit difficulty spikes before assigning passages to students.

## Problem / Why

Existing heuristics and ML estimators are not Lexile-faithful and produce unstable difficulty measures. We need a deterministic analyzer that mirrors MetaMetrics methodology (≥125-word slices extended to sentence boundaries, MSL and MLF using corpus frequencies) to support calibration and production scoring.

## Personas & Scenarios

- Persona: Curriculum data engineer
  - Cares about reproducible difficulty scores aligned to Lexile methodology
  - Constraints: must run offline, deterministic, no external APIs
  - Goals: compute MSL/MLF per slice and document for audit and calibration
  - Frustrations: drifting scores when normalization or tokenization changes
  - Context: preparing passages for grade-banded collections

- Scenario: Auditing a new passage
  - Trigger: new passage added to a grade 4 reading set
  - Steps: run `lexile-corpus-tuner analyze text passage.txt --json-output out.json`; review slice-level MSL/MLF for spikes; optionally flag as picture book/emergent nonfiction for adjustment
  - Obstacles: missing frequency table or inconsistent tokenization would invalidate results
  - Outcome: receives JSON with slices, document features, and adjusted Lexile estimate


## Acceptance Criteria

- [ ] Analyzer uses shared `normalize_text`/`iter_tokens` and the corpus frequency table for MLF
- [ ] Slices enforce ≥125 words and extend to sentence boundaries
- [ ] CLI returns slice and document features plus adjusted Lexile when flags are set
- [ ] Behavior is deterministic given the same inputs and frequency table

## Non-Goals

- Implementing corpus download/normalization/frequency generation (covered by #11)
- Training regression models (covered by calibration layer #13)
- UI/visualization of slice metrics beyond CLI/JSON output

