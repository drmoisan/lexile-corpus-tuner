# Lexile Corpus Tuner

Lexile Corpus Tuner now ships two complementary toolsets:

1. **Lexile Tuner CLI (`lexile-tuner`)** – a Lexile-inspired analysis + rewriting pipeline. It slices overlapping windows, scores them with pluggable estimators, and optionally rewrites high-Lexile spans via an LLM-powered rewriter.
2. **Lexile-Faithful Text Difficulty Pipeline (`text-difficulty-pipeline`) _(in process)_** – a corpus/analyzer/calibration workflow that mirrors the official Lexile analyzer: it builds a proxy corpus (Gutenberg + Simple Wiki + OpenStax/CK-12), computes word frequencies per 5M tokens, extracts Lexile-style features, and fits a ridge regression against curated texts with known Lexiles.

Both layers share the same normalization/tokenization utilities so lexicon frequencies and analyzer features stay aligned.

For a detailed, step-by-step build specification, see [`docs/text-difficulty-pipeline.md`](docs/text-difficulty-pipeline.md).

---
## Features

### Completed
- Character-offset tokenization + deterministic sliding windowing.
- Pluggable estimator interface (dummy heuristic + TensorFlow `lexile_v2`).
- Constraint detection for max-window and document-level Lexile targets.
- OpenAI-backed rewriter abstraction (or no-op).
- Typer CLI (`lexile-tuner`) with `analyze`, `rewrite`, `print-config`.
- Corpus tooling (`text-difficulty-pipeline corpus`): Gutenberg ID bootstrap (`scripts/build_gutenberg_id_list.py`), Simple Wiki dump downloader/extractor (`scripts/extract_simple_wiki_dump.py`), OpenStax/CK-12 manifest fetcher (`data/meta/oer_sources.json`), shared normalization/sharding (`data/corpus/normalized/shards/`), and frequency computation with per-source weights (`data/meta/corpus_sources.json`).
- Analyzer/calibration stack: sentence segmentation, Lexile-style slice builder, MSL/MLF feature computation, regression inference, special-case adjustments, and calibration CLIs (`fetch-texts`, `build-dataset`, `fit`).

### In Process
- Promoting the `docs/text-difficulty-pipeline.md` content into first-class docs/README sections.
- Adding more modern CC-BY/CC0 sources (NOAA/NASA/CDC/etc.) via manifest downloads.
- Richer calibration diagnostics (per-band MAE/RMSE, residual plots, reporting).
- Example calibration catalogs + staging instructions.

---
## Installation
```bash
poetry install --with dev
# Optional extras
# poetry install --with dev --extras "lexile-v2"
# poetry install --with dev --extras "llm-openai"
```
Formatting uses `black` + `isort`.

---
## Usage

### Lexile Tuner CLI
```bash
poetry run lexile-tuner analyze --input-path examples/example_corpus --config examples/example_config.yaml
poetry run lexile-tuner rewrite --input-path examples/example_corpus --output-path artifacts/tuned --config examples/example_config.yaml
poetry run lexile-tuner print-config
poetry run lexile-tuner analyze --input-path examples/example_corpus/pg2701-images-3.epub
```

### Text Difficulty Pipeline
```bash
# Optional: regenerate Gutenberg IDs (strict English)
poetry run python scripts/build_gutenberg_id_list.py

# Optional: copy examples/meta/oer_sources.example.json → data/meta/oer_sources.json and fill in OpenStax/CK-12 excerpts

# 1) Download raw sources (Gutenberg + Simple Wiki + OER manifest)
poetry run text-difficulty-pipeline corpus download

# 2) Convert Simple Wiki dump to JSONL articles
poetry run python scripts/extract_simple_wiki_dump.py \
  --dump data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2 \
  --output data/corpus/raw/simple_wiki/simplewiki_articles.jsonl

# 3) Normalize & shard
poetry run text-difficulty-pipeline corpus normalize --shard-size-tokens 100000

# 4) Compute frequency table (per-source weights honored)
poetry run text-difficulty-pipeline corpus frequencies

# 5) Calibration workflow (requires data/calibration/catalog/lexile_catalog.csv)
poetry run text-difficulty-pipeline calibration fetch-texts \
  --catalog data/calibration/catalog/lexile_catalog.csv \
  --texts-root data/calibration/texts

poetry run text-difficulty-pipeline calibration build-dataset \
  --catalog data/calibration/catalog/lexile_catalog.csv \
  --texts-root data/calibration/texts \
  --output data/calibration/calibration_dataset.parquet

poetry run text-difficulty-pipeline calibration fit \
  data/calibration/calibration_dataset.parquet \
  --out data/model/lexile_regression_model.json

# 6) Analyze new text with the Lexile-faithful analyzer
poetry run text-difficulty-pipeline analyze text path/to/doc.txt --json-output report.json
```

---
## Architecture Snapshot

### Lexile Tuner
1. `tokenization.py` – regex tokenizer with offsets.
2. `windowing.py` – overlapping windows.
3. `scoring.py` – applies `LexileEstimator` implementations.
4. `constraints.py` – per-window + document checks.
5. `rewriting.py` – OpenAI-backed rewriter abstraction.
6. `pipeline.py` – orchestrates the tokenize → window → score → rewrite loop.
7. `cli.py` – Typer entry points.

### Text-Difficulty Pipeline
1. **Corpus Builder** – Gutenberg/Simple Wiki/OER downloaders, Normalization (`corpus/normalize.py`), Frequency computation (`corpus/frequencies.py`).
2. **Analyzer Layer** – slice construction (`analyzer/slices.py`), feature computation (`analyzer/features.py`), regression inference (`analyzer/model.py`), CLI (`analyzer/cli.py`).
3. **Calibration Layer** – feature engineering (`calibration/featureset.py`), ElasticNet training (`calibration/train.py`), JSON model store (`calibration/model_store.py`), CLI (`calibration/cli.py`).

Shared utilities live in `textutils.py`.

---
## Configuration
`lexile_corpus_tuner.config` provides helpers for loading `LexileTunerConfig` from YAML (see `examples/example_config.yaml`).

---
## Using the `lexile_v2` Estimator
The TensorFlow-based estimator is still available for backward compatibility. Install via `poetry install --with dev --extras "lexile-v2"`, set `estimator_name: lexile_v2`, and point to the artifacts in `examples/lexile_v2_artifacts/`. Download required NLTK corpora once:
```bash
poetry run python -m nltk.downloader punkt punkt_tab wordnet averaged_perceptron_tagger omw-1.4
```
> **Deprecation note:** Once the regression-calibrated analyzer reaches feature parity, the TensorFlow `lexile_v2` implementation will be retired.

---
## Examples
- `examples/example_corpus/chapter1.txt`
- `examples/example_corpus/pg2701-images-3.epub`
- `examples/run_lexile_v2_adapter.py`
- `examples/run_openai_rewrite.py`
- `examples/meta/oer_sources.example.json`

---
## Rewriting with OpenAI
Install the `llm-openai` extra, enable `rewrite_enabled: true`, and optionally use `scripts/load-openai-key.ps1` (LastPass helper). CLI overrides (`--openai-model`, `--openai-temperature`, etc.) control behavior per run.

---
## Testing
```bash
poetry run pytest
poetry run pyright
poetry run black --check .
poetry run isort --check-only .
```

---
## Code Statistics
`cloc` binaries live in `tools/`. Run `./scripts/run-cloc.sh` or `pwsh ./scripts/run-cloc.ps1`.

---
## Next Steps / Roadmap
1. Docs consolidation (promote plan → docs/README).
2. Corpus expansion (more CC-BY/CC0 informational sources + weighting).
3. Advanced weighting/stratified sharding to keep Gutenberg under target share.
4. Calibration diagnostics (per-band metrics, residual plots, automated reporting).
5. Packaging/release to PyPI once docs/tests stabilize.
6. Evaluator benchmarks (regression tests against known Lexile values).
7. Deprecate `lexile_v2` once the calibrated analyzer fully replaces it.

Contributions and issue reports are welcome!
