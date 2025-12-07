# Lexile Corpus Tuner

Lexile Corpus Tuner is a toolkit for measuring, constraining, and rewriting a corpus based on the text difficulty level. It was designed to expand the availability of high / low reading material for struggling readers. It pairs a Lexile-style rewriting pipeline with a Lexile-approximation scoring model to produce texts that are calibrated to a reader's skill level. Since Lexile is a proprietary and closed system, a scoring model was developed to approximate lexile scores. This includes a corpus/analyzer/calibration stack that curates a custom mega-corpus, analyzes word frequency and sentence length, and then calibrates an overal score against known lexile measurements. 

## What You Can Do

- Identify portions of a text that exceed a configurable maximium difficulty level.
- Determine key words and sentences that trigger the violation
- Rewrite portions of this text to avoid exceeding configured maximums
- Build a custom Lexile-approximation model:

  - Semi-automated curation of sources used for the mega-corpus
  - Download and consolidate the mega-corpus
  - Analyze the mega-corpus and individual texts to extract meta-metrics
  - Calibrate these metrics by fitting a model based on word frequency and sentence length to known lexile levels

## Install

```bash
poetry install --with dev
# Optional extras
poetry install --with dev --extras "lexile-v2"   # TensorFlow Lexile v2 adapter
poetry install --with dev --extras "llm-openai"  # OpenAI-backed rewriting
```

## CLI Quickstarts

### Lexile Tuner (`lexile-tuner`)

```bash
# Analyze and report Lexile-style violations (no rewrites)
poetry run lexile-tuner analyze --input-path examples/example_corpus --config examples/example_config.yaml

# Rewrite violating windows and emit tuned copies + summary.json
poetry run lexile-tuner rewrite \
  --input-path examples/example_corpus \
  --output-path artifacts/tuned \
  --config examples/example_config.yaml

# Inspect defaults
poetry run lexile-tuner print-config
poetry run lexile-tuner analyze --input-path examples/example_corpus/pg2701-images-3.epub
```

### Lexile Scoring Model Pipeline (`lexile-scoring-model-pipeline`, alias: `text-difficulty-pipeline`)

```bash
# Optional helpers
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.build_gutenberg_id_list
# Copy examples/meta/oer_sources.example.json to data/meta/oer_sources.json if using OER excerpts

# 1) Download raw sources (Gutenberg + Simple Wiki + OER manifest)
poetry run lexile-scoring-model-pipeline corpus download

# 2) Convert Simple Wiki dump to JSONL articles
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_simple_wiki_dump \
  --dump data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml.bz2 \
  --output data/corpus/raw/simple_wiki/simplewiki_articles.jsonl

# 3) Normalize & shard (shared tokenizer/normalizer)
poetry run lexile-scoring-model-pipeline corpus normalize --shard-size-tokens 100000

# 4) Compute frequency table (per-source weights honored)
poetry run lexile-scoring-model-pipeline corpus frequencies

# 5) Calibration workflow (needs data/calibration/catalog/lexile_catalog.csv and downloaded texts)
poetry run lexile-scoring-model-pipeline calibration fetch-texts \
  --catalog data/calibration/catalog/lexile_catalog.csv \
  --texts-root data/calibration/texts

poetry run lexile-scoring-model-pipeline calibration build-dataset \
  --catalog data/calibration/catalog/lexile_catalog.csv \
  --texts-root data/calibration/texts \
  --output data/calibration/calibration_dataset.parquet

poetry run lexile-scoring-model-pipeline calibration fit \
  data/calibration/calibration_dataset.parquet \
  --out data/model/lexile_regression_model.json

# 6) Analyze new text with the calibrated analyzer
poetry run lexile-scoring-model-pipeline analyze text path/to/doc.txt --json-output report.json
```

### Gutenberg Corpus Explorer

```bash
# CLI (Boolean query engine over metadata parquet)
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.explore_gutenberg

# GUI (Tkinter query builder with save/load/export)
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui
```

Requires `data/meta/gutenberg_metadata.parquet` to be present.

## Flow Maps

**Build the calibrated Lexile-style model**
```
lexile-scoring-model-pipeline corpus download
  -> corpus normalize --shard-size-tokens 100000
  -> corpus frequencies
  -> calibration fetch-texts --catalog data/calibration/catalog/lexile_catalog.csv --texts-root data/calibration/texts
  -> calibration build-dataset --catalog data/calibration/catalog/lexile_catalog.csv --texts-root data/calibration/texts --output data/calibration/calibration_dataset.parquet
  -> calibration fit data/calibration/calibration_dataset.parquet --out data/model/lexile_regression_model.json
```

**Run the rewriting pipeline**
```
lexile-tuner analyze --input-path <txt|epub|dir> [--config ...]           # inspect violations only
lexile-tuner rewrite --input-path <...> --output-path <out> [--config ...] \
  [--estimator-name lexile_v2|dummy] [--openai-* flags when using llm-openai extra]
  -> Baseline stats (no rewrite)
  -> Optional OpenAI-backed rewrites of violating windows
  -> Tuned documents + summary.json under output-path
```

## Data & Artifacts

- Corpus artifacts: `data/corpus/raw`, `data/corpus/normalized/shards`
- Frequency tables: `data/freq/word_frequencies.tsv` (+ `.meta.json`)
- Calibration assets: `data/calibration/catalog/*.csv`, `data/calibration/texts`, `data/calibration/calibration_dataset.parquet`
- Model output: `data/model/lexile_regression_model.json`
  All of the above are git-ignored; commands are deterministic given the same inputs.

## Package Layout (src/lexile_corpus_tuner)

- `cli.py`: entry points `lexile-tuner` and `lexile-scoring-model-pipeline` (alias: `text-difficulty-pipeline`)
- `corpus_tuning_pipeline/`: tokenization, windowing, constraints, scoring, rewriting, EPUB ingestion, text difficulty pipeline runner
- `lexile_scoring_model/`: `corpus` (download/normalize/frequencies), `analyzer` (slices/features/CLI), `calibration` (dataset build + fit/CLI), `pipeline_scripts` (Gutenberg helpers, query builder, Simple Wiki extractor)
- `estimators/`: dummy estimator and optional TensorFlow Lexile v2 adapter
- `llm/`: OpenAI rewrite client used by `lexile-tuner` when `llm-openai` extra is installed
- `config.py`, `models.py`, `frequency_loader.py`: shared config, domain models, and frequency loading helpers

## Development & CI

- Tooling: Black (88), Ruff, Pyright (strict), Pytest. See `docs/developer-tooling.md`.
- CI: matrix for Python 3.10–3.13 plus security/build/docs checks (`docs/ci-documentation.md`).
- Policies: `.github/instructions/general-code-change.instructions.md`, `.github/instructions/python-code-change.instructions.md`, `.github/instructions/general-unit-test.instructions.md`, `.github/instructions/python-unit-test.instructions.md`.
- Common commands:
  - `poetry run black .`
  - `poetry run ruff check`
  - `poetry run pyright`
  - `poetry run pytest`

## Examples

- `examples/example_corpus/chapter1.txt`
- `examples/example_corpus/pg2701-images-3.epub`
- `examples/run_lexile_v2_adapter.py`
- `examples/run_openai_rewrite.py`
- `examples/meta/oer_sources.example.json`

## Docs & Roadmap

- Lexile-faithful initiative: `docs/features/active/lexile-faithful-text-difficulty-pipeline/initiative.md`
- Refactor structure spec: `docs/features/active/lexile-refactor-pipeline-structure/spec.md`
- Active feature specs/plans: `docs/features/active/`
- Backlog: `docs/features/backlog.md`
- Testing guidance: `docs/unit-test-policy.md`

Contributions and issue reports are welcome!
