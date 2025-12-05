# Initial Development Plan

> **Status**: ✅ Completed
>
> This document contains the detailed specifications and implementation requirements that were completed during the initial development of the Lexile Corpus Tuner project. These specifications are now implemented, tested, and operational.
>
> For current project architecture and domain model, see [`.github/copilot-instructions.md`](../.github/copilot-instructions.md).

## Overview

This plan describes the core implementation requirements for the `lexile_corpus_tuner` package, which provides:

* Text tokenization with character offsets
* Overlapping windowing for text difficulty analysis
* Pluggable Lexile-like estimators
* Constraint-based document scoring
* Automated rewriting pipeline
* Full CLI interface

---

## 1. Repository Structure

✅ **Complete**: Core structure implemented

### Required Layout

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

  * `example_corpus/` (sample text files)
  * `example_config.yaml`

---

## 2. Domain Models (`models.py`)

✅ **Complete**: All models implemented and tested

### Required Dataclasses

**Document**

```python
@dataclass
class Document:
    doc_id: str
    text: str  # full original text
```

**Token**

```python
@dataclass
class Token:
    text: str
    start_char: int
    end_char: int
```

**Window**

```python
@dataclass
class Window:
    doc_id: str
    window_id: int
    start_token_idx: int
    end_token_idx: int  # exclusive
    text: str           # raw substring from Document.text
```

**WindowScore**

```python
@dataclass
class WindowScore:
    window: Window
    lexile: float  # numeric approximation
```

**DocumentLexileStats**

```python
@dataclass
class DocumentLexileStats:
    doc_id: str
    avg_lexile: float
    max_lexile: float
    window_scores: list[WindowScore]
```

**ConstraintViolation**

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

## 3. Configuration (`config.py`)

✅ **Complete**: Configuration system with YAML support

### Config Structure

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
    estimator_name: str = "dummy"
    rewrite_enabled: bool = False
    rewrite_model: str | None = None
```

### Required Functions

* `config_from_dict(d: dict) -> LexileTunerConfig`
* `config_from_yaml(path: str) -> LexileTunerConfig`

---

## 4. Tokenization (`tokenization.py`)

✅ **Complete**: Tokenization with full test coverage

### Requirements

* Stable word tokenization with character offsets
* Simple regex-based tokenizer: `r"\w+('\w+)?"`
* Unicode support via `re.UNICODE`

### Required Function

```python
def tokenize_words(text: str) -> list[Token]:
    """
    Tokenize text into words with character offsets.
    
    Uses regex pattern r"\w+('\w+)?" with re.UNICODE.
    Returns list of Token objects with start_char/end_char.
    """
```

### Test Requirements

* Verify token boundaries
* Test punctuation handling
* Test apostrophes and contractions
* Test multiple spaces and whitespace

---

## 5. Windowing (`windowing.py`)

✅ **Complete**: Windowing with full test coverage

### Requirements

* Create overlapping windows of `window_size` tokens
* Step between windows = `stride` tokens
* Each window includes accurate text slice and indices

### Required Function

```python
def create_windows(
    doc: Document,
    tokens: list[Token],
    window_size: int,
    stride: int
) -> list[Window]:
    """
    Create overlapping windows from tokenized document.
    
    Parameters:
    - doc: Source document
    - tokens: Tokenized words with character offsets
    - window_size: Number of tokens per window
    - stride: Number of tokens to advance between windows
    
    Returns list of Window objects with text slices.
    """
```

### Algorithm

1. Loop over token indices with step = `stride`
2. For each window:
   * `end_idx = min(start_idx + window_size, len(tokens))`
   * Derive character slice via tokens' `start_char` and `end_char`
3. Stop when `start_idx` reaches final token

### Test Requirements

* Verify window count for known token count
* Verify adjacent windows overlap by expected amount
* Verify text slices are correct without off-by-one errors

---

## 6. Estimators Package (`estimators/`)

✅ **Complete**: Core estimator interface with DummyLexileEstimator

### Base Class (`estimators/base.py`)

```python
class LexileEstimator(ABC):
    @abstractmethod
    def predict_scalar(self, text: str) -> float:
        """
        Return a numeric Lexile-like difficulty score for the input text.
        """
        ...
```

### Dummy Estimator (`estimators/dummy_estimator.py`)

✅ **Complete**: Basic heuristic estimator

Implementation requirements:

* Split text into sentences (`.`, `!`, `?` heuristic)
* Compute average words per sentence
* Compute average characters per word
* Map to pseudo-Lexile via linear formula
* Must be deterministic and reproducible

```python
class DummyLexileEstimator(LexileEstimator):
    def predict_scalar(self, text: str) -> float:
        """
        Heuristic-based Lexile estimation using:
        - Average sentence length (words)
        - Average word length (characters)
        
        Returns reproducible pseudo-Lexile score.
        """
```

### Factory Function (`estimators/__init__.py`)

✅ **Complete**: Estimator factory

```python
def create_estimator(name: str, **kwargs) -> LexileEstimator:
    """
    Factory to create a LexileEstimator by name.
    Supported names: "dummy", "lexile_v2", etc.
    """
```

---

## 7. Scoring (`scoring.py`)

✅ **Complete**: Scoring with full test coverage

### Required Functions

**Score Windows**

```python
def score_windows(
    windows: list[Window],
    estimator: LexileEstimator
) -> list[WindowScore]:
    """
    Score each window using the estimator.
    
    Returns list of WindowScore objects.
    """
```

**Smooth Window Lexiles**

```python
def smooth_window_lexiles(
    window_scores: list[WindowScore],
    kernel_size: int
) -> list[float]:
    """
    Apply moving average smoothing over Lexile values.
    
    Uses window order as given in window_scores.
    """
```

**Compute Document Stats**

```python
def compute_document_stats(
    all_window_scores: list[WindowScore]
) -> list[DocumentLexileStats]:
    """
    Compute per-document statistics.
    
    Groups scores by doc_id.
    For each doc: compute average and max Lexile.
    Returns list of DocumentLexileStats.
    """
```

---

## 8. Constraints (`constraints.py`)

✅ **Complete**: Constraint checking with full test coverage

### Required Functions

**Find Violations**

```python
def find_violations(
    doc_stats: DocumentLexileStats,
    config: LexileTunerConfig
) -> list[ConstraintViolation]:
    """
    Identify constraint violations in document.
    
    Checks:
    1. Window-level: lexile > config.max_window_lexile
    2. Document-level: avg_lexile outside [target - tolerance, target + tolerance]
    
    Returns list of ConstraintViolation objects.
    """
```

**Check Hard Window Violations**

```python
def has_hard_window_violations(
    violations: list[ConstraintViolation]
) -> bool:
    """
    Returns True if any violation has window_id >= 0.
    """
```

### Test Requirements

* Test detection of high window Lexiles
* Test detection of global average issues
* Test with synthetic stats

---

## 9. Rewriting (`rewriting.py`)

✅ **Complete**: Rewriting interfaces

### Purpose

* Abstraction for LLM-based rewriting of violating windows
* Isolate external API calls from core logic

### Required Interfaces

**RewriteRequest**

```python
@dataclass
class RewriteRequest:
    doc_id: str
    window_id: int
    text: str
    target_lexile: float
    violation: ConstraintViolation | None
    metadata: dict  # constraint thresholds, etc.
```

**Rewriter Base Class**

```python
class Rewriter(ABC):
    @abstractmethod
    def rewrite(self, request: RewriteRequest) -> str:
        """
        Rewrite text to meet target Lexile.
        """
        ...
```

**NoOpRewriter**

```python
class NoOpRewriter(Rewriter):
    def rewrite(self, request: RewriteRequest) -> str:
        """Returns request.text unchanged."""
        return request.text
```

**CallableRewriter**

```python
class CallableRewriter(Rewriter):
    """Adapts arbitrary functions for testing."""
    def __init__(self, func: Callable[[RewriteRequest], str]):
        self.func = func
    
    def rewrite(self, request: RewriteRequest) -> str:
        return self.func(request)
```

### Prompt Guidelines

* System prompt: instructions, tone, Lexile target, formatting rules
* User prompt template: doc/window IDs, source Lexile, reason, constraint thresholds, text
* Guardrails:
  * Plain text only (no Markdown)
  * Preserve factual information and names
  * Maintain paragraph count
  * ±10% token count
  * Age-appropriate content

### Secret Handling

* Never commit API keys
* Read from config or environment variables
* Support `api_key` and `api_key_env` in config
* See [`docs/developer-tooling.md`](developer-tooling.md#secret-management) for details

---

## 10. Pipeline Orchestration (`pipeline.py`)

✅ **Complete**: Core pipeline with document and corpus processing

### Required Functions

**Process Document**

```python
def process_document(
    doc: Document,
    config: LexileTunerConfig,
    estimator: LexileEstimator,
    rewriter: Rewriter,
) -> tuple[Document, DocumentLexileStats, list[ConstraintViolation]]:
    """
    Process single document through tuning pipeline.
    
    Algorithm:
    1. Start with current_doc = doc
    2. For up to config.max_passes:
       a. Tokenize current_doc.text
       b. Create windows
       c. Score windows
       d. Compute DocumentLexileStats
       e. Find violations
       f. Identify hard window violations (window_id >= 0)
       g. If no hard violations: return (current_doc, stats, violations)
       h. Else:
          - Sort violations by severity (highest Lexile first)
          - Take worst violation
          - Extract window span text
          - Build RewriteRequest
          - Call rewriter.rewrite()
          - Replace span in current_doc
          - Loop again
    3. If max passes reached: return last state even if violations remain
    """
```

**Process Corpus**

```python
def process_corpus(
    documents: list[Document],
    config: LexileTunerConfig,
    estimator: LexileEstimator,
    rewriter: Rewriter,
) -> dict[str, tuple[Document, DocumentLexileStats, list[ConstraintViolation]]]:
    """
    Process multiple documents.
    
    Loops over documents and calls process_document for each.
    Returns mapping: doc_id -> (final_doc, stats, violations)
    """
```

**Helper Functions**

```python
def get_window_span_text(
    doc: Document,
    window: Window,
    tokens: list[Token]
) -> str:
    """Extract text for a specific window."""

def replace_window_span(
    doc: Document,
    window: Window,
    tokens: list[Token],
    new_text: str
) -> Document:
    """Rebuild Document.text with new span."""
```

---

## 11. CLI (`cli.py`)

✅ **Complete**: CLI with analyze, rewrite, print-config commands

### CLI Tool Name

* Console entry point: `lexile-tuner`

### Required Commands

**1. analyze**

Purpose: Analyze documents without rewriting

Inputs:
* `--input-path` (file or directory)
* `--config` (YAML file, optional)
* Optional overrides for all config parameters

Behavior:
* Load text documents
* Run process_corpus with `rewrite_enabled = False`
* Output JSON/YAML summary with:
  * Per-document average and max Lexile
  * List of violating windows (ranges, Lexile values)

**2. rewrite**

Purpose: Analyze and rewrite documents to meet constraints

Inputs:
* `--input-path`
* `--output-path` (directory for tuned documents)
* `--config`
* Optional overrides: `--rewrite-enabled`, `--rewrite-model`, `--openai-*` flags

Behavior:
* Load documents
* Create Rewriter based on config
* Run process_corpus
* Save rewritten documents to output-path
* Save summary JSON with pre/post stats and violation logs

**3. print-config**

Purpose: Print default configuration

Behavior: Output default config to stdout as YAML

### Error Handling

Required validations:
* `input-path` exists
* `output-path` is creatable
* LLM settings provided if `rewrite_enabled = true`

### Entry Point

```toml
[project.scripts]
lexile-tuner = "lexile_corpus_tuner.cli:main"
```

---

## 12. Tests (`tests/`)

✅ **Complete**: Comprehensive test suite with 400+ passing tests

### Required Unit Tests

**test_tokenization.py**
* Token boundaries and offsets
* Punctuation handling
* Apostrophes and contractions
* Multiple spaces

**test_windowing.py**
* Window counts for known token counts
* Adjacent window overlaps
* Text coverage without off-by-one errors

**test_estimators.py**
* DummyLexileEstimator on synthetic text
* Deterministic behavior
* Factory function

**test_scoring_and_constraints.py**
* Window scoring
* Smoothing algorithm
* Document stats computation
* Violation detection (window-level and document-level)

**test_pipeline.py**
* End-to-end with DummyLexileEstimator and NoOpRewriter
* Rewriting behavior with fake rewriter
* Multi-pass iteration
* Max passes limit

### Integration Tests

**test_cli.py**
* Run CLI analyze command on example corpus
* Verify exit code 0
* Verify summary file output

---

## 13. Documentation & Examples

✅ **Complete**: Comprehensive documentation in README.md

### Required README Sections

* **Motivation**: Lexile-based tuning for child readers
* **Architecture**: tokenization → windowing → scoring → constraints → rewriting
* **Installation**: `pip install -e .` or `poetry install`
* **CLI Usage**: `lexile-tuner analyze ...`, `lexile-tuner rewrite ...`
* **Extensibility**: How to plug in custom estimator via `estimators.create_estimator`

### Required Examples

**examples/example_corpus/**
* `chapter1.txt` - Sample text for testing

**examples/example_config.yaml**
* Modified parameters showing:
  * Different window_size and stride
  * Different constraint thresholds
  * Estimator selection

---

## 14. Packaging & Build System

✅ **Complete**: Poetry-based packaging

### Required Configuration

**pyproject.toml**
* Package metadata (name, version, description, authors)
* Python version requirement (3.10+)
* Core dependencies: typer, pytest
* Optional dependency groups:
  * `lexile-v2`: ML dependencies for Keras model
  * `llm-openai`: OpenAI client dependencies
* Entry point: `lexile-tuner = lexile_corpus_tuner.cli:main`
* Black, Ruff, Pyright, Pytest configuration

### Project Layout

* Use `src/` layout for proper packaging
* Include `LICENSE` (MIT)
* Include comprehensive `README.md`

---

## 15. Code Quality Requirements

✅ **Complete**: All code quality tooling configured and enforced

See [`docs/code-change.instructions.md`](code-change.instructions.md) and [`docs/developer-tooling.md`](developer-tooling.md) for complete specifications:

* **Formatting**: Black (default settings)
* **Linting**: Ruff with project configuration
* **Type Checking**: Pyright in strict mode
* **Testing**: Pytest with coverage reporting
* **Documentation**: Comprehensive docstrings for public APIs

---

## 16. Autonomy Requirements

✅ **Complete**: All requirements met

### Operational Requirements

* Every module importable without external configuration
* `pytest` passes with only DummyLexileEstimator (no external ML models)
* `lexile-tuner analyze --input-path examples/example_corpus` works immediately after install
* No required external dependencies for basic functionality

### Extensibility Points

* Pluggable estimators via `LexileEstimator` ABC
* Pluggable rewriters via `Rewriter` ABC
* Factory functions for easy registration
* Configuration-driven behavior

---

## Implementation Notes

### Completed Features

All sections marked ✅ in this document have been:

1. **Implemented**: Code written and integrated
2. **Tested**: Unit tests passing with good coverage
3. **Documented**: Docstrings and README coverage
4. **Validated**: Working end-to-end in CLI and library modes

### External Integrations

Some features require external components and are documented in separate plans:

* **Lexile V2 Model**: See [`docs/features/archive/lexile_v2_plan.md`](features/archive/lexile_v2_plan.md)
* **OpenAI LLM Integration**: See [`docs/features/archive/llm_plan.md`](features/archive/llm_plan.md)

### Usage

This plan served as the development specification. Now that implementation is complete, it serves as:

* Historical record of design decisions
* Reference for understanding core architecture
* Baseline for future enhancements

For ongoing development, see:
* [`.github/copilot-instructions.md`](../.github/copilot-instructions.md) - Current architecture
* [`docs/code-change.instructions.md`](code-change.instructions.md) - Coding standards
* [`docs/developer-tooling.md`](developer-tooling.md) - Development workflow
