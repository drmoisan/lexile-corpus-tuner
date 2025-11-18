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
# Analyze a folder of .txt or .epub files
poetry run lexile-tuner analyze --input-path examples/example_corpus --config examples/example_config.yaml

# Rewrite difficult passages and write tuned files to a new directory
poetry run lexile-tuner rewrite --input-path examples/example_corpus --output-path tuned --config examples/example_config.yaml

# Print the default configuration
poetry run lexile-tuner print-config

# Analyze a single EPUB source
poetry run lexile-tuner analyze --input-path examples/example_corpus/pg2701-images-3.epub
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
lexile_v2_model_path: examples/lexile_v2_artifacts/model.h5
lexile_v2_vectorizer_path: examples/lexile_v2_artifacts/tokenizer.pickle
lexile_v2_label_encoder_path: examples/lexile_v2_artifacts/labels.pickle
lexile_v2_stopwords_path: examples/lexile_v2_artifacts/stopwords.pickle
lexile_v2_band_to_midpoint:
  200-299: 250
  300-399: 350
```

The repository bundles pre-trained artifacts from [Elizabeth Fawcett's lexile-determination-v2 project](https://github.com/eannefawcett/lexile-determination-v2) under `examples/lexile_v2_artifacts/`. They remain licensed under MIT per the upstream project (see `examples/lexile_v2_artifacts/LICENSE` for full text). Please credit the original author if you redistribute or build on these files.

> **NLTK resources:** The preprocessing pipeline relies on NLTK's tokenizers and lemmatizer. After installing the `lexile-v2` extra, download the required corpora once per machine:
>
> ```bash
> poetry run python -m nltk.downloader punkt punkt_tab wordnet averaged_perceptron_tagger omw-1.4
> ```

## Examples

`examples/example_corpus/chapter1.txt` contains a short excerpt that you can analyze immediately. `examples/example_corpus/pg2701-images-3.epub` is a full Project Gutenberg title that exercises the EPUB ingestion path (rewritten output is emitted as `.txt`). Run the CLI commands shown above after installing the project in editable mode.

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

### Loading `OPENAI_API_KEY` from LastPass

If you manage secrets in LastPass, keep using environment variables while letting LastPass supply the value at runtime:

1. Store the key as a secure note or password entry named **Lexile OpenAI Key** (or pick any name).
2. Install the [LastPass CLI](https://support.lastpass.com/help/install-lastpass-cli-lp040011) and run `lpass login you@example.com` once per machine/session.
3. Load the key into your shell when you need to run the rewriter:

```powershell
pwsh ./scripts/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"
# Optional flags:
#   -UsePasswordField  # use the stored password instead of the note body
#   -EnvVar "OPENAI_API_KEY"  # change the destination env var
```

The script simply runs `lpass show` and assigns the secret to `OPENAI_API_KEY` for the current session, so no secrets touch tracked files or shell history. Add the command to your PowerShell profile if you want it available automatically.

#### Windows setup via WSL

LastPass no longer ships a native Windows CLI build. If you are on Windows, the easiest way to keep using the script above is to run the official Linux CLI through WSL and expose it to PowerShell:

1. Install WSL with an Ubuntu distro if you haven’t already (`wsl --install -d Ubuntu`) and run the initial setup.
2. Update the package index and install the CLI inside WSL:
   ```powershell
   wsl -e sudo apt-get update
   wsl -e sudo apt-get install lastpass-cli
   wsl -e bash -lc "mkdir -p ~/.config/lpass"
   ```
3. Log into LastPass from WSL once per Windows session so the CLI can read your vault: `wsl -e lpass login you@example.com`.
4. Create a small Windows wrapper so PowerShell can invoke the WSL binary (this lives anywhere on disk; `%USERPROFILE%\bin` keeps things tidy):
   ```powershell
   $wrapperDir = "$env:USERPROFILE\bin"
   if (-not (Test-Path $wrapperDir)) { New-Item -ItemType Directory -Path $wrapperDir | Out-Null }
   "@echo off`nwsl.exe lpass %*" | Set-Content "$wrapperDir\lpass.cmd" -Encoding ASCII
   ```
5. Add `$wrapperDir` to your **user** `PATH` (System Properties → Environment Variables → Path → New → paste the folder path) and open a new PowerShell window. `Get-Command lpass` should now resolve to the wrapper and `lpass --version` should print the WSL version.
6. Use the repo helper script normally. It reads secure notes by default; pass `-UsePasswordField` if your entry stores the key in the password field instead.

If `lpass` reports “Session token missing or expired,” just re-run `wsl -e lpass login ...` before calling `load-openai-key.ps1`.

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
