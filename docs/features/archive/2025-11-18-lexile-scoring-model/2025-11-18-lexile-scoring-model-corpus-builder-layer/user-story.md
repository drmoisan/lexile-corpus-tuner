# `lexile-corpus-builder-layer` — User Story

- Issue: #11
- Owner: drmoisan
- Status: Draft
- Last Updated: 2025-12-05

## Story Statement

- As a curriculum data engineer, I want to build a proxy Lexile corpus and frequency tables, so that downstream analyzers can compute mean log word frequency (MLF) consistently.
- As an NLP engineer, I want idempotent corpus download/normalize/frequency CLI commands, so that I can regenerate artifacts without manual cleanup.

## Problem / Why

We need a reproducible, open-source corpus pipeline that mirrors the MetaMetrics Lexile reference corpus. Without stable word-frequency tables and shared normalization/tokenization, the analyzer cannot produce Lexile-faithful MLF features, and calibration accuracy suffers.

## Personas & Scenarios

- Persona: Curriculum data engineer
  - Cares about deterministic artifacts, repeatable runs, and minimal manual steps.
  - Constraints: Limited time, must avoid committing large corpora to git.
  - Goals: Generate frequency tables and metadata usable by the analyzer and calibration layers.
- Scenario: Refreshing the corpus and frequencies
  - Trigger: New source list or updated normalization rules.
  - Steps: Run `corpus download` (Gutenberg + Simple Wiki + OER), run `corpus normalize` to shard ~100k-token JSONL files, run `corpus frequencies` to emit TSV + meta.
  - Obstacles: Large downloads, Unicode/text noise, keeping outputs out of VCS, ensuring idempotence.
  - Outcome: Updated `data/corpus/normalized/shards`, `word_frequencies.tsv`, and metadata ready for analyzer/calibration.

## Acceptance Criteria

- [ ] CLI provides `corpus download`, `corpus normalize`, and `corpus frequencies` commands.
- [ ] Downloads are idempotent and create required directory structure under `data/corpus/raw`.
- [ ] Normalization emits shard files (~100k tokens) plus summary metadata; uses shared `textutils`.
- [ ] Frequency computation outputs `word_frequencies.tsv` with counts, per-5M rates, and log frequencies, plus `word_frequencies.meta.json`.
- [ ] No large corpus artifacts are committed to git; outputs live under `data/` and are .gitignored.

## Non-Goals

- Training or calibration of Lexile models (handled in calibration layer).
- Analyzer feature extraction or slicing logic (handled in analyzer layer).

