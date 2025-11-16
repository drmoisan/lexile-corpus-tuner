# Lexile Corpus Tuner

Lexile Corpus Tuner is a reference implementation of a Lexile-inspired analysis and rewriting pipeline. It ingests raw English text, slices overlapping 500-word windows, scores each span with a pluggable estimator, and optionally rewrites the most difficult passages with an LLM-powered rewriter so that the final corpus fits a ~350 Lexile target appropriate for 10-year-old readers.

## Features

- Tokenization with character offsets suitable for round-trip rewriting.
- Sliding-window segmentation and deterministic scoring pipeline.
- Pluggable estimator interface with a reproducible dummy heuristic out of the box.
- Constraint detection for max window Lexile and document-level averages.
- Rewriter abstraction that can wrap an external LLM (OpenAI Responses API) or operate as a no-op.
- Typer-based CLI with analyze, rewrite, and print-config commands.

## Installation

```bash
poetry install --with dev
# Include TensorFlow adapter extras if you plan to use the lexile_v2 estimator
# poetry install --with dev --extras "lexile-v2"
# Include OpenAI extras if you plan to run the LLM-backed rewriter
# poetry install --with dev --extras "llm-openai"
```

Formatting uses `black` + `isort`; run `black .` and `isort .` before committing if desired.

## Usage

```bash
# Analyze a folder of .txt files
poetry run lexile-tuner analyze --input-path examples/example_corpus --config examples/example_config.yaml

# Rewrite difficult passages and write tuned files to a new directory
poetry run lexile-tuner rewrite --input-path examples/example_corpus --output-path tuned --config examples/example_config.yaml

# Print the default configuration
poetry run lexile-tuner print-config
```

## Architecture

1. **Tokenization** (`tokenization.py`) – regex word tokenizer that emits tokens with offsets.
2. **Windowing** (`windowing.py`) – overlapping windows derived from tokens according to the configuration.
3. **Scoring** (`scoring.py`) – applies a `LexileEstimator` to every window and computes document statistics.
4. **Constraints** (`constraints.py`) – detects hard per-window violations and document-level average deviations.
5. **Rewriting** (`rewriting.py`) – pluggable abstractions for LLM-based rewriting with a prompt template for guidance.
6. **Pipeline** (`pipeline.py`) – orchestrates the iterative tokenize → window → score → rewrite loop.
7. **CLI** (`cli.py`) – Typer commands that wrap the pipeline for analyzing or rewriting corpora.

Custom estimators can be plugged in via `lexile_corpus_tuner.estimators.create_estimator`.

## Configuration

`lexile_corpus_tuner.config` exposes helpers to load `LexileTunerConfig` from dictionaries or YAML files. See `examples/example_config.yaml` for an end-to-end sample that tweaks window size, stride, and estimator options.

## Using the `lexile_v2` estimator

To run the TensorFlow-based estimator you need artifacts exported from the upstream `lexile-determination-v2` training project. Install the extra dependencies by including the Poetry extra (`poetry install --with dev --extras "lexile-v2"`), set `estimator_name: lexile_v2`, and supply the artifact paths in your config. Optional CLI overrides are available: `--estimator-name lexile_v2 --lexile-v2-model-path /path/model.h5 --lexile-v2-vectorizer-path ...`.

When enabling the estimator, set the following additional fields (paths should point to your trained artifacts):

```yaml
estimator_name: lexile_v2
lexile_v2_model_path: /abs/path/to/model.h5
lexile_v2_vectorizer_path: /abs/path/to/vectorizer.pkl
lexile_v2_label_encoder_path: /abs/path/to/labels.pkl
lexile_v2_band_to_midpoint:
  200-299: 250
  300-399: 350
```

## Examples

`examples/example_corpus/chapter1.txt` contains a short excerpt that you can analyze immediately. Run the CLI commands shown above after installing the project in editable mode.

## Rewriting with OpenAI

Install the optional extra to pull in the official SDK:

```bash
pip install .[llm-openai]
# or with Poetry
poetry install --with dev --extras "llm-openai"
```

Set an API key in your environment (never commit keys to the repo):

```bash
export OPENAI_API_KEY="sk-..."
```

Update your configuration to enable rewriting via the Responses API:

```yaml
# examples/example_config.yaml
rewrite_enabled: true
rewrite_model: gpt-4.1-mini
openai:
  enabled: true
  api_key_env: OPENAI_API_KEY
  temperature: 0.3
  max_output_tokens: 450
  top_p: 0.95
```

Then run the CLI with overrides as needed:

```bash
poetry run lexile-tuner rewrite \
  --input-path examples/example_corpus \
  --output-path artifacts/tuned \
  --config examples/example_config.yaml \
  --rewrite-enabled \
  --openai-enabled \
  --openai-model gpt-4.1-mini
```

You can override advanced options such as `--openai-temperature`, `--openai-request-timeout`, or `--openai-base-url` (for Azure/private gateways). The rewriter keeps paragraph boundaries, avoids adding/removing facts, and targets the configured Lexile range. See `examples/run_openai_rewrite.py` for a quick-start script that rewrites an inline string.

⚠️ **Privacy & Cost**: Sending data to OpenAI incurs usage costs and may transmit student content. Use a policy-compliant deployment (e.g., Azure OpenAI) when required, and never commit secrets to Git.

## Testing

```bash
# Run all tests
poetry run pytest

# Or use pytest directly if installed in your environment
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=lexile_corpus_tuner
```

**Note:** Use `pytest` or `python -m pytest`, **not** `python -m test`. The latter runs Python's internal standard library test suite (480+ tests), which is unrelated to this project.

The provided test suite covers tokenization, windowing, estimators, scoring + constraint logic, and pipeline orchestration.
