# lexile-analyzer-layer - Plan

- Issue: #12
- Owner: drmoisan
- Last Updated: 2025-12-05

## Required References (read, do not restate)

- Coding workflow and standards: [`docs/code-change.instructions.md`](../../code-change.instructions.md)
- Unit test policy: [`docs/unit-test-policy.md`](../../unit-test-policy.md)

**All work must comply with these policies; do not duplicate their content here.**

## Phases

### Phase 1: Slicing and normalization [0%]
- [x] Implement sentence segmentation and ≥125-word slice construction (extend to sentence boundary)
- [x] Ensure shared `normalize_text`/`iter_tokens` are used for analyzer path
- [x] Tests/docs for this phase

### Phase 2: Features and adjustments [0%]
- [x] Compute slice/document features (MSL, MLF) using frequency table
- [x] Apply optional picture-book / emergent-nonfiction adjustments; keep deterministic
- [x] Tests/docs for this phase

### Phase 3: CLI and outputs [0%]
- [x] Wire `analyze text` CLI: flags, JSON output, error handling for missing frequencies
- [x] Add logging of adjustments and missing artifacts; ensure human-readable summary
- [x] Tests/docs for this phase

## Test Plan

- Unit
	- [x] Sentence splitter yields deterministic sentences
	- [x] Slice builder enforces ≥125 tokens and closes at sentence boundaries
	- [x] Feature computation uses shared tokenizer and frequency loader; MSL/MLF math validated on fixtures
	- [x] Adjustment flags modify Lexile as specified

- Integration
	- [x] Analyzer consumes `word_frequencies.tsv` and errors clearly when missing
	- [x] End-to-end: text → slices → features → adjusted Lexile

- CLI/UX examples
	- [x] `lexile-corpus-tuner analyze text <file>` prints summary and writes JSON when requested
	- [x] Flags `--picture-book` / `--emergent-nonfiction` reflected in output

- Performance/edge cases
	- [x] Handles very long sentences without hanging or memory blowup
	- [x] Deterministic outputs across runs given same inputs/frequencies
	- [x] Behavior when frequency table missing or empty is guarded with clear error

## Open Questions / Notes

- Any need for additional slice heuristics (e.g., paragraph-aware boundaries)?
- Should we expose per-sentence diagnostics in JSON for audit?
- Coordinate changes to normalization/tokenization with corpus layer to avoid drift

