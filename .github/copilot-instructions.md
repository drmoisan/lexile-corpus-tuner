# Canonical Instructions

> **Note:** This document defines the current project architecture and domain model. For detailed implementation specifications that have been completed, see [`docs/initial-development-plan.md`](../docs/initial-development-plan.md). For coding standards, testing policies, and development workflow, see [`docs/code-change.instructions.md`](../docs/code-change.instructions.md) and [`docs/developer-tooling.md`](../docs/developer-tooling.md).

## 1. Project Overview

**Project Name**: `lexile_corpus_tuner`

**Purpose**: A Python package and CLI tool for analyzing and rewriting text to meet Lexile readability constraints.

### Core Functionality

The system:

* Ingests English text documents
* Splits them into overlapping ~500-word windows
* Estimates Lexile-like complexity scores per window using pluggable estimators
* Computes document-level statistics (average, max)
* Identifies constraint violations:
  * Target average Lexile ≈ 350
  * No 500-word window > 450
* Optionally rewrites violating windows using LLM
* Iteratively re-evaluates until constraints are satisfied or max passes reached

### Target Use Case

Content modification for ~10-year-old students reading at ~350 Lexile:

* Age-appropriate topics with simplified language
* Focuses on preventing local spikes in difficulty
* Deterministic and reproducible processing
* Modular architecture for testing and extension

---

## 2. Architecture Overview

### Core Pipeline

```
Document Input → Tokenization → Windowing → Scoring → Constraint Checking
                                                           ↓
                                                      Violations?
                                                           ↓
                                         Yes → Rewriting → Re-evaluate
                                          ↓
                                         No → Output Results
```

### Module Organization

* **models.py**: Domain dataclasses (Document, Token, Window, etc.)
* **tokenization.py**: Word tokenization with character offsets
* **windowing.py**: Overlapping window creation
* **estimators/**: Pluggable Lexile estimators
  * base.py: Abstract interface
  * dummy_estimator.py: Heuristic-based fallback
  * lexile_determination_v2_adapter.py: Optional ML model wrapper
* **scoring.py**: Window scoring and document statistics
* **constraints.py**: Violation detection
* **rewriting.py**: LLM-based text rewriting interfaces
* **pipeline.py**: End-to-end orchestration
* **cli.py**: Command-line interface (analyze, rewrite, print-config)
* **config.py**: Configuration management with YAML support

### Key Design Principles

1. **Pluggability**: Estimators and rewriters use abstract interfaces
2. **Isolation**: External dependencies (ML models, LLM APIs) are isolated
3. **Testability**: Core logic testable without external services
4. **Configurability**: YAML-based configuration with CLI overrides

---

## 3. Domain Model

See [`docs/initial-development-plan.md`](../docs/initial-development-plan.md#2-domain-models-modelspy) for complete specifications.

**Core Entities**:

* `Document`: Input text with ID
* `Token`: Word with character offsets
* `Window`: Overlapping text span with token indices
* `WindowScore`: Window + Lexile score
* `DocumentLexileStats`: Aggregate statistics per document
* `ConstraintViolation`: Detected constraint failures

---

## 4. Configuration

**Configuration Class**: `LexileTunerConfig`

**Key Parameters**:

* `window_size`: Tokens per window (default: 500)
* `stride`: Token step between windows (default: 250)
* `max_window_lexile`: Maximum allowed for any window (default: 450.0)
* `target_avg_lexile`: Target document average (default: 350.0)
* `avg_tolerance`: Acceptable deviation from target (default: 20.0)
* `max_passes`: Maximum rewriting iterations (default: 3)
* `estimator_name`: Which estimator to use ("dummy", "lexile_v2", etc.)
* `rewrite_enabled`: Whether to enable LLM rewriting
* `rewrite_model`: LLM model identifier

**Loading**:

* From dict: `config_from_dict(d: dict)`
* From YAML: `config_from_yaml(path: str)`
* CLI overrides via flags

---

## 5. Extension Points

### Custom Estimators

Implement `LexileEstimator` ABC from `estimators/base.py`:

```python
class LexileEstimator(ABC):
    @abstractmethod
    def predict_scalar(self, text: str) -> float:
        """Return numeric Lexile-like difficulty score."""
        ...
```

Register via `create_estimator(name: str, **kwargs)` factory.

### Custom Rewriters

Implement `Rewriter` ABC from `rewriting.py`:

```python
class Rewriter(ABC):
    @abstractmethod
    def rewrite(self, request: RewriteRequest) -> str:
        """Rewrite text to meet target Lexile."""
        ...
```

Built-in implementations:

* `NoOpRewriter`: Pass-through (no rewriting)
* `CallableRewriter`: Adapts arbitrary functions
* `OpenAIRewriter`: LLM-based rewriting

---

## 6. CLI Interface

**Entry Point**: `lexile-tuner`

**Commands**:

1. **analyze**: Analyze documents without rewriting

   * `--input-path`: File or directory to analyze
   * `--config`: YAML configuration file
   * Output: JSON/YAML summary with stats and violations
2. **rewrite**: Analyze and rewrite documents

   * `--input-path`: Input file or directory
   * `--output-path`: Output directory for rewritten documents
   * `--config`: YAML configuration file
   * `--rewrite-enabled`: Enable LLM rewriting
   * `--openai-*`: OpenAI API configuration flags
   * Output: Rewritten documents + summary report
3. **print-config**: Print default configuration as YAML

---

## 7. Development Guidelines

### Code Quality

See [`docs/code-change.instructions.md`](../docs/code-change.instructions.md) for comprehensive standards:

* **Formatting**: Black (88 char line length)
* **Linting**: Ruff with project configuration
* **Type Checking**: Pyright in strict mode
* **Testing**: Pytest with coverage reporting
* **Documentation**: Docstrings for all public APIs

### Development Workflow

See [`docs/developer-tooling.md`](../docs/developer-tooling.md) for tooling documentation:

* **Installation**: Poetry or pip with optional extras
* **Secret Management**: LastPass integration for API keys
* **VS Code Integration**: Tasks for all quality checks
* **Continuous Integration**: GitHub Actions with automated checks

### Testing Requirements

* All new code must have unit tests
* Core logic testable without external dependencies
* Integration tests for CLI commands
* Mock external services (OpenAI, ML models)

---

## 8. External Integrations

### Lexile V2 Model (Optional)

Keras-based model for Lexile band prediction. See [`docs/features/archive/lexile_v2_plan.md`](../docs/features/archive/lexile_v2_plan.md) for details.

**Status**: Adapter stub implemented, full integration documented separately.

### OpenAI LLM (Optional)

LLM-based text rewriting for constraint satisfaction. See [`docs/features/archive/llm_plan.md`](../docs/features/archive/llm_plan.md) for details.

**Status**: Interface complete, OpenAI client integration documented separately.

---

## 9. Secret Management

**Critical Rules**:

* NEVER commit API keys, tokens, or `.env` files
* Use environment variables or secure secret storage (LastPass)
* Configuration supports both direct keys and env var references

**OpenAI API Key Loading**:

```powershell
pwsh ./scripts/production/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"
```

See [`docs/developer-tooling.md#secret-management`](../docs/developer-tooling.md#secret-management) for complete guide.

---


## 11. Key Documentation

* **Initial Development Plan**: [`docs/initial-development-plan.md`](../docs/initial-development-plan.md) - Completed implementation specifications
* **Coding Standards**: [`docs/code-change.instructions.md`](../docs/code-change.instructions.md) - Python coding policy
* **Developer Tooling**: [`docs/developer-tooling.md`](../docs/developer-tooling.md) - Setup and workflow guide
* **Unit Test Policy**: [`docs/unit-test-policy.md`](../docs/unit-test-policy.md) - Testing standards
* **CI Documentation**: [`docs/ci-documentation.md`](../docs/ci-documentation.md) - Continuous integration setup
* **README**: Project overview, installation, and usage examples
