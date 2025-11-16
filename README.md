# Lexile Corpus Tuner

Lexile Corpus Tuner is a reference implementation of a Lexile-inspired analysis and rewriting pipeline. It ingests raw English text, slices overlapping 500-word windows, scores each span with a pluggable estimator, and optionally rewrites the most difficult passages with an LLM-powered rewriter so that the final corpus fits a ~350 Lexile target appropriate for 10-year-old readers.

## Features
- Tokenization with character offsets suitable for round-trip rewriting.
- Sliding-window segmentation and deterministic scoring pipeline.
- Pluggable estimator interface with a reproducible dummy heuristic out of the box.
- Constraint detection for max window Lexile and document-level averages.
- Rewriter abstraction that can wrap an external LLM or operate as a no-op.
- Typer-based CLI with analyze, rewrite, and print-config commands.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
# Install TensorFlow adapter extras if you plan to use the lexile_v2 estimator
# pip install -e .[lexile-v2]
```

Formatting uses `black` + `isort`; run `black .` and `isort .` before committing if desired.

## Usage

```bash
# Analyze a folder of .txt files
lexile-tuner analyze --input-path examples/example_corpus --config examples/example_config.yaml

# Rewrite difficult passages and write tuned files to a new directory
lexile-tuner rewrite --input-path examples/example_corpus --output-path tuned --config examples/example_config.yaml

# Print the default configuration
lexile-tuner print-config
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

To run the TensorFlow-based estimator you need artifacts exported from the upstream `lexile-determination-v2` training project. Install the extra dependencies via `pip install -e .[lexile-v2]`, set `estimator_name: lexile_v2`, and supply the artifact paths in your config. Optional CLI overrides are available: `--estimator-name lexile_v2 --lexile-v2-model-path /path/model.h5 --lexile-v2-vectorizer-path ...`.

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

## Testing

```bash
pytest
```

The provided test suite covers tokenization, windowing, estimators, scoring + constraint logic, and pipeline orchestration.
