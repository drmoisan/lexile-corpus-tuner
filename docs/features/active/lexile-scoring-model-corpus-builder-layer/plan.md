# lexile-corpus-builder-layer - Plan

- Issue: #11
- Owner: drmoisan
- Last Updated: 2025-12-05

## Required References (read, do not restate)

- Coding workflow and standards: [`docs/code-change.instructions.md`](../../code-change.instructions.md)
- Unit test policy: [`docs/unit-test-policy.md`](../../unit-test-policy.md)

**All work must comply with these policies; do not duplicate their content here.**

## Phases

### Phase 1: Download pipeline [0%]
- [x] Wire `corpus download` CLI to downloader using manifests (`corpus_sources.json`, optional `oer_sources.json`)
- [x] Enforce idempotence: skip existing files, log counts and destinations
- [x] Tests/docs for this phase

### Phase 2: Normalization/sharding [0%]
- [x] Implement `corpus normalize` using shared tokenizer/normalizer
- [x] Emit shard JSONL (~100k tokens) and `normalized_summary.json`; deterministic ordering
- [x] Tests/docs for this phase

### Phase 3: Frequency generation [0%]
- [x] Implement `corpus frequencies` streaming shards to counts/rates/ranks
- [x] Write `word_frequencies.tsv` and `word_frequencies.meta.json`; support source weighting flag
- [x] Tests/docs for this phase

### Phase 4: UX and docs [0%]
- [ ] Document commands and data layout in README/feature docs
- [ ] Ensure `data/` is git-ignored; add guardrails if needed
- [ ] Tests/docs for this phase

### Phase 5: Hardening [0%]
- [ ] Validate manifests and shard-size params with clear errors
- [ ] Add schema checks for outputs; add perf guardrails (streaming reads)
- [ ] Tests/docs for this phase

## Test Plan

- Unit
    - [x] Normalization uses shared `textutils` pipeline; shard boundaries align with token counts
    - [x] Frequency math on fixture corpus (counts, per-5M rate, log frequency, rank)

- Integration
    - [x] `corpus download` creates expected directories; rerun is idempotent
    - [x] `corpus normalize` produces shard count/summary matching input tokens
    - [x] `corpus frequencies` reads shards and yields deterministic outputs

- CLI/UX examples
    - [ ] End-to-end run on small fixture corpus exercising all commands
    - [ ] Flags: `--gutenberg-limit`, `--shard-size-tokens`, source-weighting flag

- Performance/edge cases
    - [ ] Streaming shard read (no full corpus in memory)
    - [x] Missing manifests → clear error; bad shard size → validation error
    - [x] Partial downloads present → safe skip without corruption

## Open Questions / Notes

- Consider parallel normalization if perf is a bottleneck
- Keep frequency schema compatible with analyzer expectations

