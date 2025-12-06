# Lexile Corpus Tuner

Lexile Corpus Tuner helps keep text at a ~350 Lexile level (10-year-old readers) by analyzing overlapping windows and rewriting the hard parts. The project ships two coordinated toolsets that share tokenization/normalization utilities so corpus statistics and analyzer features stay aligned.

1. **Lexile Tuner CLI (`lexile-tuner`)** - analyze + optionally rewrite text with pluggable estimators and an OpenAI-backed rewriter.
2. **Lexile-Faithful Text Difficulty Pipeline (`text-difficulty-pipeline`)** - build a proxy corpus (Gutenberg, Simple Wiki, OpenStax/CK-12), compute 5M-token frequency tables, extract Lexile-style features, and fit a regression-calibrated analyzer.
3. **Gutenberg Query Builder (CLI + GUI)** - explore and export Gutenberg metadata with structured queries.

For the step-by-step build specification, see [`docs/text-difficulty-pipeline.md`](docs/text-difficulty-pipeline.md).

---

## Quick Install

```bash
poetry install --with dev
# Optional extras
# poetry install --with dev --extras "lexile-v2"
# poetry install --with dev --extras "llm-openai"
```

Formatting uses `black` (88 chars) + `isort` (per-project config).

---

## CLI Quickstarts

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
poetry run python -m lexile_corpus_tuner.pipeline_scripts.build_gutenberg_id_list

# Optional: copy examples/meta/oer_sources.example.json to data/meta/oer_sources.json and fill in OpenStax/CK-12 excerpts

# 1) Download raw sources (Gutenberg + Simple Wiki + OER manifest)
poetry run text-difficulty-pipeline corpus download

# 2) Convert Simple Wiki dump to JSONL articles
poetry run python -m lexile_corpus_tuner.pipeline_scripts.extract_simple_wiki_dump \
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

### Gutenberg Corpus Explorer

```bash
# CLI
poetry run python -m lexile_corpus_tuner.pipeline_scripts.explore_gutenberg

# GUI
poetry run python -m lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui
```

Features: structured queries (field/value, ranges, boolean), sortable results table (first 100 rows), save/load/export to CSV or Parquet, keyboard shortcuts, and drag-and-drop nested groups.

---

## Architecture and Domain Model

**Core pipeline:** `Document -> Tokenization -> Windowing -> Scoring -> Constraint Checking -> (Rewriting?) -> Output`.

**Module map (lexile_tuner):** `models.py` (Document/Token/Window/WindowScore/ConstraintViolation), `tokenization.py`, `windowing.py`, `estimators/` (dummy + optional `lexile_determination_v2_adapter.py`), `scoring.py`, `constraints.py`, `rewriting.py`, `pipeline.py`, `cli.py`, `config.py`.

**Domain entities:**

- `Document` with ID + raw text
- `Token` with character offsets
- `Window` (overlapping token spans)
- `WindowScore` (window + Lexile score)
- `DocumentLexileStats` (aggregate statistics)
- `ConstraintViolation` (per-window/document issues)

**Design principles:** pluggable estimators/rewriters, isolation of external deps, deterministic/testable core, YAML-configurable CLI overrides.

---

## Configuration

`LexileTunerConfig` defaults: `window_size=500`, `stride=250`, `max_window_lexile=450.0`, `target_avg_lexile=350.0`, `avg_tolerance=20.0`, `max_passes=3`, `estimator_name="dummy"`, `rewrite_enabled=False`, `rewrite_model` optional. Load via `config_from_dict` or `config_from_yaml`, or pass overrides on the CLI.

---

## Extension Points

- **Estimators:** implement `LexileEstimator.predict_scalar(text: str) -> float` and register via `create_estimator(name, **kwargs)`.
- **Rewriters:** implement `Rewriter.rewrite(request: RewriteRequest) -> str`. Built-ins: `NoOpRewriter`, `CallableRewriter`, `OpenAIRewriter`.

---

## External Integrations and Secrets

- **Lexile V2 (optional):** TensorFlow adapter for backwards compatibility (`estimator_name: lexile_v2`, artifacts under `examples/lexile_v2_artifacts/`).
- **OpenAI rewriting (optional):** install `llm-openai`, set `rewrite_enabled: true`, and pass `--openai-*` flags as needed.
- **Secret handling:** never commit keys. Load OpenAI keys from LastPass: `pwsh ./src/lexile_corpus_tuner/pipeline_scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"`. Config supports direct values or env-var indirection.

---

## Developer Workflow and Quality

- Policies: follow [`docs/code-change.instructions.md`](docs/code-change.instructions.md) for coding standards and workflow, [`docs/developer-tooling.md`](docs/developer-tooling.md) for tooling, and [`docs/unit-test-policy.md`](docs/unit-test-policy.md) for tests.
- Common tasks: `poetry run black .`, `poetry run ruff check`, `poetry run pyright`, `poetry run pytest` (or VS Code tasks: Run All Checks).
- CI mirrors these checks across Python 3.10-3.13; see [`docs/ci-documentation.md`](docs/ci-documentation.md).

---

## Examples

- `examples/example_corpus/chapter1.txt`
- `examples/example_corpus/pg2701-images-3.epub`
- `examples/run_lexile_v2_adapter.py`
- `examples/run_openai_rewrite.py`
- `examples/meta/oer_sources.example.json`

---

## Testing

```bash
poetry run pytest
poetry run pyright
poetry run black --check .
poetry run isort --check-only .
```

---

## Documentation Map

- Implementation plan: [`docs/text-difficulty-pipeline.md`](docs/text-difficulty-pipeline.md)
- Coding standards: [`docs/code-change.instructions.md`](docs/code-change.instructions.md)
- Developer tooling: [`docs/developer-tooling.md`](docs/developer-tooling.md)
- Unit tests: [`docs/unit-test-policy.md`](docs/unit-test-policy.md)
- CI setup: [`docs/ci-documentation.md`](docs/ci-documentation.md)
- Feature backlog/ideas: [`docs/features/backlog.md`](docs/features/backlog.md), [`docs/features/ideas/ideas.md`](docs/features/ideas/ideas.md)

---

## Next Steps / Roadmap

1. Promote pipeline docs into README/docs.
2. Expand corpus with CC-BY/CC0 informational sources and weighting.
3. Advanced weighting/stratified sharding to keep Gutenberg under target share.
4. Rich calibration diagnostics (per-band metrics, residual plots, reporting).
5. Package/release to PyPI when docs/tests stabilize.
6. Evaluator benchmarks against known Lexile values.
7. Deprecate `lexile_v2` after calibrated analyzer fully replaces it.

Contributions and issue reports are welcome!

