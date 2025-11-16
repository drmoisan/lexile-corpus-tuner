# Canonical Instructions

## 1. Project Overview

1.1 **Project Name (working)**

* `lexile_corpus_tuner`

1.2 **Primary Goal**

* Build a Python package + CLI that:

  * Ingests arbitrary English text documents.
  * Splits them into overlapping ~500-word windows.
  * Estimates a **Lexile-like complexity score** per window using a pluggable estimator.
  * Computes document-level stats (average, max).
  * Identifies windows that violate constraints:

    * Target average Lexile ≈ **350**.
    * No 500-word window > **450**.
  * Optionally calls an external LLM to **rewrite** violating windows to bring them into range.
  * Iteratively re-evaluates until constraints are satisfied (or max passes reached).

1.3 **Key Use Case Context**

* Intended for modifying content for a ~10-year-old student reading at ~350 Lexile:

  * Content should be **age-appropriate in topic** but simplified in language.
  * We care about **local spikes** in difficulty: no 500-word passage should be too hard.
* The system must be:

  * Deterministic and reproducible where possible.
  * Modular, testable, and usable as both a library and CLI tool.

---

## 2. Tech Stack & Standards

2.1 **Language & Version**

* Python 3.10+ (safe default, type hints, dataclasses, etc.).

2.2 **Core Dependencies (minimal)**

* `click` or `typer` for CLI (pick one and use consistently).
* `pydantic` or `dataclasses` (dataclasses for core, Pydantic for config if useful).
* `pytest` for tests.
* **Optional / pluggable**: any ML or NLP libs needed for the Lexile estimator (e.g., `scikit-learn`, `tensorflow`, etc.), but these should be isolated in a dedicated module.

2.3 **Code Quality**

* Use **type hints** everywhere.
* Use `black` + `isort` style (document in README).
* Use docstrings with clear descriptions of arguments, returns, and behavior.

2.4 **Packaging & Layout**

* Use `src/` layout.
* Include:

  * `pyproject.toml` (preferred) or `setup.cfg` + `setup.py`.
  * `README.md` with explanation and usage examples.
  * `LICENSE` (MIT is fine).

---

## 3. Repository Structure

3.1 **Top-level layout**

* `README.md`
* `LICENSE`
* `pyproject.toml`
* `src/lexile_corpus_tuner/`

  * `__init__.py`
  * `config.py`
  * `models.py`
  * `tokenization.py`
  * `windowing.py`
  * `estimators/`

    * `__init__.py`
    * `base.py`
    * `dummy_estimator.py`
    * `lexile_determination_v2_adapter.py` (stub / optional)
  * `scoring.py`
  * `constraints.py`
  * `rewriting.py`
  * `pipeline.py`
  * `cli.py`
* `tests/`

  * `test_tokenization.py`
  * `test_windowing.py`
  * `test_estimators.py`
  * `test_scoring_and_constraints.py`
  * `test_pipeline.py`
* `examples/`

  * `example_corpus/` (a couple of small text files)
  * `example_config.yaml`

---

## 4. Domain Models & Data Structures (`models.py`)

4.1 **Document**

```python
@dataclass
class Document:
    doc_id: str
    text: str  # full original text
```

4.2 **Token**

* Represented implicitly as a tuple `(token: str, start_char: int, end_char: int)`.
* Optionally define a small dataclass for clarity:

```python
@dataclass
class Token:
    text: str
    start_char: int
    end_char: int
```

4.3 **Window**

```python
@dataclass
class Window:
    doc_id: str
    window_id: int
    start_token_idx: int
    end_token_idx: int  # exclusive
    text: str           # raw substring from Document.text
```

4.4 **WindowScore**

```python
@dataclass
class WindowScore:
    window: Window
    lexile: float  # numeric approximation
```

4.5 **DocumentLexileStats**

```python
@dataclass
class DocumentLexileStats:
    doc_id: str
    avg_lexile: float
    max_lexile: float
    window_scores: list[WindowScore]
```

4.6 **ConstraintViolation**

```python
@dataclass
class ConstraintViolation:
    doc_id: str
    window_id: int  # -1 for document-level violation
    lexile: float
    reason: str
    start_token_idx: int
    end_token_idx: int
```

---

## 5. Configuration (`config.py`)

5.1 **Config Structure**

* Use a `Config` class (Pydantic or dataclass) to hold tunable parameters:

```python
@dataclass
class LexileTunerConfig:
    window_size: int = 500
    stride: int = 250
    max_window_lexile: float = 450.0
    target_avg_lexile: float = 350.0
    avg_tolerance: float = 20.0
    max_passes: int = 3
    smoothing_kernel_size: int = 3
    estimator_name: str = "dummy"  # or "lexile_v2"
    rewrite_enabled: bool = False
    rewrite_model: str | None = None
```

5.2 **Config Loading**

* Add helper functions for loading from:

  * A `dict`.
  * A YAML file (`config_from_yaml(path: str) -> LexileTunerConfig`).

---

## 6. Tokenization (`tokenization.py`)

6.1 **Requirements**

* Stable word tokenization with character offsets.
* Use a simple regex-based tokenizer.

6.2 **Functions**

* `tokenize_words(text: str) -> list[Token]`:

  * Uses regex like `r"\w+('\w+)?"` with `re.UNICODE`.
  * Returns a list of `Token` objects with `start_char`/`end_char`.

* Include unit tests verifying:

  * Token boundaries.
  * Behavior with punctuation, apostrophes, and multiple spaces.

---

## 7. Windowing (`windowing.py`)

7.1 **Requirements**

* Create overlapping windows of `window_size` tokens with `stride` tokens between starts.
* Each `Window` must have:

  * `text` slice from original document.
  * Accurate `start_token_idx` and `end_token_idx`.

7.2 **Functions**

* `create_windows(doc: Document, tokens: list[Token], window_size: int, stride: int) -> list[Window]`:

  * Loop over token indices with step = `stride`.
  * For each window:

    * `end_idx = min(start_idx + window_size, len(tokens))`.
    * Derive character slice via tokens `start_char` and `end_char`.
  * Stop when `start_idx` has reached the final token.

* Tests:

  * Check number of windows given a known token count.
  * Verify that adjacent windows overlap by expected amount.
  * Verify that concatenating windows would cover the original text without off-by-one errors (at least at edges).

---

## 8. Estimators Package (`estimators/`)

### 8.1 Base Class (`base.py`)

Define a pluggable interface for Lexile estimators:

```python
class LexileEstimator(ABC):
    @abstractmethod
    def predict_scalar(self, text: str) -> float:
        """
        Return a numeric Lexile-like difficulty score for the input text.
        """
        ...
```

Optionally expose a `from_config` factory.

### 8.2 Dummy Estimator (`dummy_estimator.py`)

Implement a basic heuristic estimator as a fallback / test double, using something like average sentence length and word length:

* `DummyLexileEstimator(LexileEstimator)`:

  * `predict_scalar(text: str) -> float`:

    * Split text into sentences (simple `.`/`!`/`?` heuristic).
    * Compute:

      * average words per sentence,
      * average characters per word.
    * Map to a pseudo-Lexile via a simple linear formula (document in docstring).
    * This gives reproducible outputs without external ML models.

### 8.3 Adapter for external model (`lexile_determination_v2_adapter.py`)

* Define `LexileDeterminationV2Estimator(LexileEstimator)`:

  * Designed to wrap an external Keras model that predicts Lexile bands.
  * Include:

    * `__init__(model_path: str, band_to_midpoint: dict[str, float])`.
    * `predict_band(text: str) -> str` (internal helper).
    * `predict_scalar(text: str) -> float` (map band to numeric using `band_to_midpoint`).
  * For now, implement as a **stub**:

    * Include method signatures and docstrings.
    * Optional TODO comments explaining where to load the model and how to call it.

### 8.4 Factory Function (`estimators/__init__.py`)

* Implement a function:

```python
def create_estimator(name: str, **kwargs) -> LexileEstimator:
    """
    Factory to create a LexileEstimator by name.
    Supported names: "dummy", "lexile_v2", etc.
    """
```

* Use `name` to select `DummyLexileEstimator` or `LexileDeterminationV2Estimator`.

---

## 9. Scoring (`scoring.py`)

9.1 **Functions**

* `score_windows(windows: list[Window], estimator: LexileEstimator) -> list[WindowScore]`:

  * Loop over windows and run `estimator.predict_scalar(w.text)`.
  * Return `WindowScore` list.

* `smooth_window_lexiles(window_scores: list[WindowScore], kernel_size: int) -> list[float]`:

  * Implement a simple moving average smoothing over Lexile values.
  * Use window order as given in `window_scores`.

* `compute_document_stats(all_window_scores: list[WindowScore]) -> list[DocumentLexileStats]`:

  * Group scores by `doc_id`.
  * For each doc:

    * Compute average and max Lexile.
  * Return a list of `DocumentLexileStats`.

---

## 10. Constraints (`constraints.py`)

10.1 **Functions**

* `find_violations(doc_stats: DocumentLexileStats, config: LexileTunerConfig) -> list[ConstraintViolation]`:

  * For each `WindowScore`:

    * If `lexile > config.max_window_lexile`, emit a `ConstraintViolation` with window indices.
  * For document average:

    * If `doc_stats.avg_lexile` is outside `[target_avg - avg_tolerance, target_avg + avg_tolerance]`, emit a violation with `window_id = -1`.

* `has_hard_window_violations(violations: list[ConstraintViolation]) -> bool`:

  * Returns True if any violation has `window_id >= 0`.

10.2 **Tests**

* Test with synthetic stats to ensure:

  * Correct detection of high window Lexiles.
  * Correct detection of global average issues.

---

## 11. Rewriting (`rewriting.py`)

11.1 **Purpose**

* Provide an abstraction for LLM-based rewriting of violating windows.
* Keep external API calls isolated from core logic.

11.2 **Interfaces**

* Define `@dataclass RewriteRequest` with:

  * `doc_id`, `window_id`, `text`, `target_lexile`.
  * Optional `violation: ConstraintViolation | None`.
  * Optional metadata dict for constraint thresholds.

* `class Rewriter(ABC)` must expose `rewrite(request: RewriteRequest) -> str`.
* `class NoOpRewriter(Rewriter)` returns `request.text`.
* `class CallableRewriter(Rewriter)` adapts arbitrary functions for testing.
* `class OpenAIRewriter(Rewriter)` composes the prompt templates with an injected `OpenAIRewriteClient`.

11.3 **Prompt Template & Guardrails**

* Split the prompt into:

  * `SYSTEM_PROMPT` – instructions, tone, Lexile target, formatting rules (plain text only, keep facts, maintain paragraph count, ±10 % tokens, age-appropriate).
  * `USER_PROMPT_TEMPLATE` – includes doc/window ids, source Lexile, reason, constraint thresholds, and the span text.

* Pass metadata (max window Lexile, avg tolerance) so the LLM understands the constraint.
* Explicitly ask for:

  * No Markdown, quotes, or additional commentary.
  * Preservation of factual information and names.
  * Roughly same length and paragraph boundaries.

11.4 **Secrets & API Keys**

* Never commit API keys, tokens, or `.env` files.
* Read API keys from config (`openai.api_key`) or environment variables (`openai.api_key_env`) only.
* Codex must surface configuration knobs (CLI + YAML) so users provide credentials securely.

---

## 12. Pipeline Orchestration (`pipeline.py`)

12.1 **Function: `process_document`**

Signature:

```python
def process_document(
    doc: Document,
    config: LexileTunerConfig,
    estimator: LexileEstimator,
    rewriter: Rewriter,
) -> tuple[Document, DocumentLexileStats, list[ConstraintViolation]]:
    ...
```

Behavior:

1. Start with `current_doc = doc`.
2. For up to `config.max_passes`:

   1. Tokenize `current_doc.text`.
   2. Create windows via `create_windows`.
   3. Score windows.
   4. Compute `DocumentLexileStats`.
   5. Call `find_violations`.
   6. Identify “hard” window violations (where `window_id >= 0`).
   7. If no hard violations:

      * Return `(current_doc, stats, violations)`.
   8. Else:

      * Sort the hard violations by severity (highest Lexile first).
      * Take the worst violation.
      * Extract the window span text (helper function).
      * Build a `RewriteRequest` with doc/window metadata, target Lexile, the violation, and constraint context.
      * Call `rewriter.rewrite(rewrite_request)` to obtain replacement text.
      * Replace the span in `current_doc` with rewritten text.
      * Loop again (re-tokenizing, re-windowing, re-scoring).
3. If max passes reached:

   * Return last `(current_doc, stats, violations)` even if violations remain.

Helper functions inside `pipeline.py`:

* `get_window_span_text(doc: Document, window: Window, tokens: list[Token]) -> str`
* `replace_window_span(doc: Document, window: Window, tokens: list[Token], new_text: str) -> Document`

  * Rebuilds `Document.text` with the new span.

12.2 **Function: `process_corpus`**

```python
def process_corpus(
    documents: list[Document],
    config: LexileTunerConfig,
    estimator: LexileEstimator,
    rewriter: Rewriter,
) -> dict[str, tuple[Document, DocumentLexileStats, list[ConstraintViolation]]]:
    ...
```

* Loops over documents and calls `process_document` for each.
* Returns a mapping `doc_id -> (final_doc, stats, violations)`.

---

## 13. CLI (`cli.py`)

13.1 **CLI Tool Name**

* Console entry point: `lexile-tuner`.

13.2 **Commands**

1. `analyze`

   * Inputs:

     * `--input-path` (file or directory).
     * `--config` (YAML file, optional).
     * Optional overrides: `--rewrite-enabled/--no-rewrite-enabled`, `--rewrite-model`, and all `--openai-*` flags (even though `analyze` still forces rewriting off for summaries).
   * Behavior:

     * Load text documents.
     * Run `process_corpus` with `rewrite_enabled = False`.
     * Output:

       * JSON or YAML summary with:

         * per-document average Lexile, max Lexile.
         * list of violating windows (ranges, Lexile values).

2. `rewrite`

   * Inputs:

     * `--input-path`
     * `--output-path` (directory to write tuned documents).
     * `--config`
     * Optional overrides: `--rewrite-enabled/--no-rewrite-enabled`, `--rewrite-model`, `--openai-model`, `--openai-api-key`, `--openai-api-key-env`, `--openai-base-url`, `--openai-organization`, `--openai-temperature`, `--openai-max-output-tokens`, `--openai-top-p`, `--openai-request-timeout`, `--openai-parallel-requests`.
   * Behavior:

     * Load documents.
     * Create `Rewriter` based on config (`OpenAIRewriter` when enabled, otherwise `NoOpRewriter`).
     * Run `process_corpus`.
     * Save final rewritten documents to `output-path` (same filenames).
     * Save a summary JSON file with Lexile stats pre/post, and violation logs.

3. `print-config`

   * Prints default config to stdout as YAML.

13.3 **Error Handling**

* Validate that:

  * `input-path` exists.
  * `output-path` is creatable.
  * LLM settings are provided if `rewrite_enabled` is true.

13.4 **Entry Point**

* Add an entry-point in `pyproject.toml`:

```toml
[project.scripts]
lexile-tuner = "lexile_corpus_tuner.cli:main"
```

---

## 14. Tests (`tests/`)

14.1 **Unit Tests**

* `test_tokenization.py`:

  * Confirm token boundaries and offsets.
* `test_windowing.py`:

  * Verify window counts and overlaps.
* `test_estimators.py`:

  * Test `DummyLexileEstimator` on synthetic text.
* `test_scoring_and_constraints.py`:

  * Use a few `WindowScore`s and verify computed stats and violations.
* `test_pipeline.py`:

  * Use `DummyLexileEstimator` and `NoOpRewriter` on a tiny document to ensure the pipeline runs end-to-end without rewriting.
  * Add a test where a fake rewriter is injected to ensure violating windows are rewritten when enabled.
* `test_openai_rewriter.py`:

  * Mock the OpenAI client to assert prompt construction, retry handling, and missing-key errors without real API calls.

14.2 **Integration Test (optional)**

* In `tests/test_cli.py`:

  * Run the CLI `analyze` command against a small example corpus.
  * Assert that the CLI returns exit code 0 and outputs a summary file.

---

## 15. Documentation & Examples

15.1 **README.md**

* Explain:

  * Motivation (Lexile-based tuning for child readers).
  * High-level architecture (tokenization → windowing → scoring → constraints → rewriting).
  * How to install (`pip install -e .` style).
  * How to run CLI (`lexile-tuner analyze ...`, `lexile-tuner rewrite ...`).
  * How to plug in a custom estimator via `estimators.create_estimator`.

15.2 **Examples**

* `examples/example_corpus/` with:

  * `chapter1.txt` (a couple of paragraphs).
* `examples/example_config.yaml`:

  * Slightly modified parameters to show how to change window size, stride, etc.

---

## 16. Autonomy Notes for Codex / Copilot

16.1 **What Codex Should Generate Automatically**

* All modules and files listed above, including:

  * Dataclasses and type hints.
  * Factory functions and interfaces.
  * Stubs for external integrations (Lexile model, LLM client).
* Reasonable defaults in the dummy estimator so the project is functional out-of-the-box without external ML models or API keys.

16.2 **Where Codex Should Use TODOs / Placeholders**

* In `lexile_determination_v2_adapter.py`:

  * Add TODO comments for model loading, with docstrings explaining expected behavior.
* In `LLMRewriter`:

  * Abstract away any concrete API calls and mark places where the user must plug in their own client / key handling.

16.3 **Design Requirements for Autonomy**

* Every module should be importable on its own without external configuration.
* Running `pytest` at the repo root should pass with only the dummy estimator and no external APIs.
* Running `lexile-tuner analyze --input-path examples/example_corpus` should work immediately after `pip install -e .`.


