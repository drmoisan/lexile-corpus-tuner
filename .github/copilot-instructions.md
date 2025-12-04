# Canonical Instructions

> **Note:** This document defines the project's domain model, architecture, and functional requirements. For coding standards, testing policies, and development workflow, see [`docs/code-change.instructions.md`](../docs/code-change.instructions.md).

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

* `typer` for CLI.
* `dataclasses` for core domain models.
* `pytest` for tests.
* **Optional / pluggable**: any ML or NLP libs needed for the Lexile estimator (e.g., `scikit-learn`, `tensorflow`, etc.), but these should be isolated in a dedicated module.

2.3 **Code Quality**

* ✅ **All coding standards, testing policies, and development workflows are documented in:**
  * **Coding Standards**: [`docs/code-change.instructions.md`](../docs/code-change.instructions.md)
  * **Developer Tooling**: [`docs/developer-tooling.md`](../docs/developer-tooling.md)

2.4 **Packaging & Layout**

* ✅ Uses `src/` layout with Poetry packaging
* ✅ Includes:
  * `pyproject.toml` with Poetry configuration
  * `README.md` with comprehensive documentation
  * `LICENSE` (MIT)

---

## 3. Repository Structure

✅ **Implementation Status**: Core structure complete

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

✅ **Implementation Status**: All models implemented and tested

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

✅ **Implementation Status**: Configuration system complete with YAML support and OpenAI settings

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

✅ **Implementation Status**: Tokenization complete with full test coverage

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

✅ **Implementation Status**: Windowing complete with full test coverage

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

✅ **Implementation Status**: Core estimator interface complete with DummyLexileEstimator

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

✅ **Complete**: Basic heuristic estimator implemented and tested

* `DummyLexileEstimator(LexileEstimator)`:

  * `predict_scalar(text: str) -> float`:

    * Split text into sentences (simple `.`/`!`/`?` heuristic).
    * Compute:

      * average words per sentence,
      * average characters per word.
    * Map to a pseudo-Lexile via a simple linear formula (document in docstring).
    * This gives reproducible outputs without external ML models.

      ### 8.4 Factory Function (`estimators/__init__.py`)
