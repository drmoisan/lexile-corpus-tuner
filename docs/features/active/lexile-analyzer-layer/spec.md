# lexile-analyzer-layer — Spec

- Issue: #12
- Owner: drmoisan
- Last Updated: 2025-12-05

## Overview

Lexile-style analyzer that builds ≥125-word slices (extended to sentence boundaries), computes mean sentence length (MSL) and mean log word frequency (MLF) using the shared normalization/tokenization pipeline and corpus frequency table, and outputs slice- and document-level features plus optional adjustments.

## Behavior

1) Load frequency table (`word_frequencies.tsv`) using shared loader; fail clearly if missing.
2) Read input text, normalize with `textutils.normalize_text`, tokenize with `iter_tokens`.
3) Split into sentences and construct slices of at least 125 words, extending to sentence boundaries.
4) Compute slice features (MSL, MLF) and aggregate document features; apply optional adjustments (picture book, emergent nonfiction) to Lexile estimate.
5) Expose results via CLI: print human-readable summary and optionally write JSON with slices, features, and adjusted estimate.

## Inputs / Outputs

- Inputs
	- CLI: `input_file`, `--json-output`, `--picture-book`, `--emergent-nonfiction`
	- Files: `data/freq/word_frequencies.tsv` (required), `data/freq/word_frequencies.meta.json` (metadata)
- Outputs
	- Console summary of slice and document metrics
	- Optional JSON file containing slices, document features, raw and adjusted Lexile
	- Logs for missing frequency table, adjustments applied

## API / CLI Surface

- Command: `lexile-corpus-tuner analyze text <input_file> [--json-output <path>] [--picture-book] [--emergent-nonfiction]`
- Response: stdout summary; JSON when requested with fields {slices, document_features, raw_lexile, adjusted_lexile, adjustments_applied}

Example:

```bash
lexile-corpus-tuner analyze text examples/example_corpus/chapter1.txt --json-output artifacts/ch1_features.json
```

## Data & State

- Read-only dependency on frequency table under `data/freq`.
- No persistent state besides optional JSON output written to user-specified path.
- Uses shared text normalization/tokenization to keep MLF consistent with corpus layer.

## Constraints & Risks

- Deterministic outputs required; changes to normalization/tokenization or frequencies will change scores and must be coordinated.
- Depends on frequency artifacts existing; CLI should emit clear error if missing.
- Sentence segmentation is heuristic; very long sentences could dominate slices but must still respect ≥125-word rule.
- Keep memory footprint reasonable; process text streaming where possible.

## Definition of Done

- [ ] Behavior matches acceptance criteria (125-word slices, shared normalization, MSL/MLF computation, adjustments)
- [ ] Tests updated/added (slices, features, adjustments, CLI)
- [ ] Docs updated (README, docs/features/active/... links)
- [ ] Logging for missing frequency table and adjustments applied

