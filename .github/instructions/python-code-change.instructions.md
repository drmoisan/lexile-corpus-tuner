---
applyTo: "**/*.py"
name: python-code-change-policy
description: "Python-specific code change rules layered on top of the general code change policy"
---

# Python Code Change Policy

This policy **extends** `general-code-change.instructions.md` and applies to all **Python source, test, and script files** (`*.py`) in this repo.   

You must:

- Apply **all** rules in the general code change policy.
- Apply **all** Python-specific rules in this file.
- Apply the unit test policies (`general-unit-test.instructions.md` and `python-unit-test.instructions.md`) for any work involving tests.

If you encounter any conflicting instructions between these documents, **halt and notify the user.**

---

## 1. Tooling & Baseline for Python

These are the required tools for Python code in this repo:

1. **Formatting — Black**

   - All Python code must be formatted with **Black** (default settings).
   - Do not hand-format; if a diff disagrees with Black, Black wins.
   - Approved command: `poetry run black .`

2. **Linting — Ruff**

   - Python code must pass **Ruff** using the project’s configuration.
   - **Suppression Authorization** (see `python-suppressions.instructions.md`):
     - All `# noqa` suppressions must either:
       1. **Match a pre-authorized pattern** in `python-suppressions.instructions.md`, OR
       2. **Have explicit user approval** for that specific suppression
     - If you encounter a Ruff error that seems to require a suppression:
       1. First, attempt to resolve it without a suppression (refactor, restructure, use approved patterns)
       2. If that fails, try at least five more distinct approaches
       3. Continue iterating until you solve the problem or demonstrate why each approach fails
       4. Only after multiple documented failed attempts may you request user approval, providing:
          - The specific Ruff rule and error message
          - Each approach you tried and why it failed
          - Why a suppression is the only remaining option
   - Use **targeted, single-line** suppressions with required comment format from `python-suppressions.instructions.md`.
   - Approved command: `poetry run ruff check`

3. **Typing — Pyright**

   - Python code must be **fully type-annotated** and pass **Pyright**.
   - Avoid `Any` unless absolutely unavoidable. If `Any` is used, include a short comment explaining why.
   - **Suppression Authorization** (see `python-suppressions.instructions.md`):
     - All `# type: ignore` suppressions must either:
       1. **Match a pre-authorized pattern** in `python-suppressions.instructions.md`, OR
       2. **Have explicit user approval** for that specific suppression
     - If you encounter a Pyright error that seems to require a suppression:
       1. First, attempt to resolve it without a suppression (add proper types, use typed wrappers, refactor)
       2. If that fails, try at least five more distinct approaches
       3. Continue iterating until you solve the problem or demonstrate why each approach fails
       4. Only after multiple documented failed attempts may you request user approval, providing:
          - The specific Pyright error and diagnostic code
          - Each approach you tried and why it failed
          - Why a suppression is the only remaining option
   - **All custom Python code** (src, tests, scripts) must be type-checked.
   - Only exclude third-party packages without proper stubs (e.g., `tkinter`, `pandas`).
   - When using untyped third-party libraries:
     - Wrap usage in custom functions or classes with proper type hints.
     - Use **line-specific** `# type: ignore[...]` comments instead of excluding whole files or directories.
   - Approved command: `poetry run pyright`

> **Testing tools and behavior are defined in the unit test policies.** Do not define test behavior here; instead, obey `general-unit-test.instructions.md` and `python-unit-test.instructions.md`.

---

## 2. Python Design & Typing Principles

These refine the general design principles for Python code.

1. **Strong typing by default**

   - All public functions, methods, and class constructors must have full type hints for parameters and return values.
   - Internal helpers should also be annotated unless they are extremely trivial and short-lived.

2. **dataclasses and value objects**

   - Prefer `@dataclass` for value objects and simple data carriers.
   - Use `frozen=True` where appropriate to enforce immutability.
   - Keep dataclasses focused on representing data + invariants, not on performing orchestration.

3. **Protocols and abstract base classes**

   - Use `typing.Protocol` or `abc.ABC` when multiple implementations are expected (e.g., different corpus sources, stores, or pipelines).
   - Code should depend on these interfaces rather than concrete implementations where flexibility is important.

4. **Utility code**

   - Avoid static-method-only “utility” classes.
   - In Python, prefer modules with top-level functions for helpers and simple transforms.
   - If you need multiple interchangeable implementations, use protocols/ABCs + classes, not utility classes.

---

## 3. Classes, Functions, and APIs (Python-Specific Guidance)

This section refines the general “classes vs functions” rules for Python. :contentReference[oaicite:4]{index=4}  

### 3.1 Classes for domain concepts and workflows

Use classes for:

- **Domain concepts** with data + behavior  
  - e.g. `QifTransaction`, `LexileCorpus`, `ContactMatcher`, `CorpusPipeline`.
- **State + invariants** that must stay consistent  
  - e.g. a `LexileModel` that must keep weights, vocabulary, and metadata in sync.
- **Multiple implementations behind a shared contract**  
  - e.g. `ITextSource` / `TextSourceProtocol` with `EpubTextSource`, `GutenbergTextSource`, etc.
- **Multi-step workflows** that share context  
  - e.g. a pipeline with `.download()`, `.normalize()`, `.index()`, `.export()`.

When using classes in Python:

- Prefer `@dataclass` for value objects.
- Keep methods small and focused; one conceptual responsibility per method.
- Avoid “God objects” that accumulate too many unrelated concerns.

### 3.2 Functions for small, pure helpers

Use standalone functions when:

- The operation is **pure, stateless, and simple**, for example:
  - `normalize_whitespace(text: str) -> str`
  - `slugify(title: str) -> str`
- It is a **small helper** that does not naturally belong on a specific domain class.
- It is a **simple transformation** from inputs to outputs.

Rules for Python helper functions:

- Fully annotate parameters and return types.
- Name functions by what they do (`parse_qif_file`, `compute_lexile_score`).
- Keep functions short, readable, and low in branching; factor complex logic into smaller helpers.

---

## 4. Error Handling, Logging, and Contracts (Python)

These refine the general error-handling rules with Python-specific details.   

1. **Exceptions**

   - Fail **fast and explicitly** by raising clear, specific exceptions when invariants are violated.
   - Avoid broad `except:` clauses.
   - Avoid `except Exception:` unless you:
     - Immediately re-raise with added context, or
     - Are at a well-defined boundary (e.g., CLI entry point) and log the full context.

2. **Logging**

   - Use the project’s logging pattern, typically the standard `logging` module.
   - Do not add ad-hoc `print` statements for permanent behavior.
   - Log at appropriate levels (`debug`, `info`, `warning`, `error`) and include enough context to debug issues.

3. **Contracts / invariants**

   - Enforce invariants at construction time (`__init__` or `__post_init__` for dataclasses).
   - Use `assert` only for internal sanity checks, not for user-facing validation or recoverable errors.

---

## 5. Module & File Structure (Python)

The general policy covers cohesion and file size; this section adds Python-specific structure rules.   

1. **Cohesive modules**

   - A module should have a clear purpose (e.g. “QIF parsing”, “Lexile model”, “corpus download”).
   - Avoid “grab-bag” modules like `utils.py` that mix many unrelated concerns.

2. **Public vs internal**

   - Keep the public surface area **small and intentional**.
   - Use `_`-prefixed module members or `_internal` modules for code that should not be used outside the module/package.
   - Do not expose internal helpers via `__all__` unless strictly necessary.

3. **Imports**

   - Prefer **absolute imports** within the project (e.g. `from project.module import Thing`) instead of deep relative imports.
   - Avoid circular dependencies; if they appear, refactor shared logic into a lower-level module.

---

## 6. Naming, Docs, and Comments (Python)

This section specializes the general naming/documentation rules for Python using PEP 8.   

1. **PEP 8 naming**

   - Use `snake_case` for functions, methods, and variables.
   - Use `PascalCase` for classes and exceptions.
   - Use `CONSTANT_CASE` for module-level constants.
   - Avoid cryptic abbreviations unless they are standard (`id`, `url`, `db`).

2. **Docstrings**

   - Public classes and methods should have a short docstring describing:
     - What it does.
     - Important arguments.
     - What it returns or any side effects.
   - Follow the prevailing docstring style in this repo (e.g., Google-style, NumPy-style, or simple one-paragraph docstrings).

3. **Comments**

   - Comment **why**, not what; the code should make the “what” clear.
   - For non-obvious patterns, workarounds, or `# type: ignore[...]` and `# noqa` uses, add a short comment explaining the reasoning.

---

## 7. Dependencies and Third-Party Libraries (Python)

The general policy defines overall dependency rules; this section notes Python-specific expectations.   

- Prefer libraries with good type stubs (built-in or via `types-...` packages).
- Do not add new runtime dependencies casually; only add them when:
  - There is no reasonable standard-library or existing-dependency alternative, and
  - The library is well-maintained and widely used.
- When wrapping third-party libraries, hide them behind small, typed adapter functions or classes so the rest of the codebase depends on your interfaces, not the raw third-party APIs.






