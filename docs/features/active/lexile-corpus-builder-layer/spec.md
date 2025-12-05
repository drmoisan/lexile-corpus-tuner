# lexile-corpus-builder-layer — Spec

- Issue: #11
- Owner: drmoisan
- Last Updated: 2025-12-05

## Overview

Build and maintain a proxy MetaMetrics-style corpus and frequency tables to support Lexile-faithful MLF computation. Provide deterministic, idempotent CLI commands for download, normalization/sharding, and frequency generation.

## Behavior

1) `corpus download` fetches sources (Gutenberg subset, Simple English Wikipedia dump, OER manifest) into `data/corpus/raw`, creating directories if missing; skips existing files.
2) `corpus normalize` applies shared `textutils.normalize_text` and `iter_tokens`, produces ~100k-token shard JSONL files under `data/corpus/normalized/shards`, and writes `normalized_summary.json`.
3) `corpus frequencies` loads shards, counts tokens with optional source weighting, writes `word_frequencies.tsv` (counts, per-5M rate, log frequency, rank) and `word_frequencies.meta.json` under `data/freq`.
4) All outputs stay in `data/` (git-ignored) and are reproducible given the same inputs.

## Inputs / Outputs

- Inputs
	- CLI flags: `--gutenberg-limit` (download), `--shard-size-tokens` (normalize)
	- Files: `data/meta/corpus_sources.json`, `data/meta/oer_sources.json` (optional), raw downloads
- Outputs
	- Directories: `data/corpus/raw`, `data/corpus/normalized/shards`, `data/freq`
	- Files: `normalized_summary.json`, shard JSONL files, `word_frequencies.tsv`, `word_frequencies.meta.json`
	- Logs: progress for downloads, counts for shards/frequencies

## API / CLI Surface

- `lexile-corpus-tuner corpus download --gutenberg-limit <int>`
- `lexile-corpus-tuner corpus normalize --shard-size-tokens <int>`
- `lexile-corpus-tuner corpus frequencies`

Example:

```bash
lexile-corpus-tuner corpus download --gutenberg-limit 200
lexile-corpus-tuner corpus normalize --shard-size-tokens 100000
lexile-corpus-tuner corpus frequencies
```

## Data & State

- Raw sources stored under `data/corpus/raw/*` (gutenberg, simple_wiki, oer)
- Normalized shards under `data/corpus/normalized/shards/*.jsonl`
- Frequency artifacts under `data/freq/word_frequencies.tsv` and `.meta.json`
- Metadata files track corpus version, source weighting, and shard summary

## Constraints & Risks

- Large downloads; keep idempotent and resumable where possible
- Do not commit corpus/frequency artifacts to git
- Unicode/encoding noise; rely on shared normalization/tokenization
- Source availability may change (URLs/IDs); manifest-driven to minimize breakage
- Memory: frequency computation should stream shards, not load entire corpus

## Definition of Done

- [ ] CLI commands work end-to-end (download → normalize → frequencies)
- [ ] Frequency outputs reproducible and match expected schema
- [ ] Tests cover download idempotence, normalization/token boundaries, frequency counting
- [ ] Docs: README/feature docs updated with commands and data layout
- [ ] Outputs are kept out of git; .gitignore respected

