---
applyTo: "**/*.py"
name: python-unit-test-policy
description: "Python-specific unit test rules, layered on top of the general unit test policy"
---
# Python Unit Test Policy

This policy **extends** `general-unit-test.instructions.md` and applies to all Python unit tests in this repo.

You must follow **both**:

- The general unit test policy, and
- The Python-specific rules below.

If there is any conflict between these documents, halt and notify the user.

---

## 1. Framework and Scope

- **Testing framework**
  - All Python unit tests must use **Pytest** as the test runner and framework.
- **Coverage expectation**
  - All new Python logic must be covered by Pytest tests that follow the general unit test policy.

---

## 2. Test Style and Structure (Python)

- **Focused unit tests**

  - Write focused tests that exercise a single function, method, or behavior.
  - Prefer testing behavior directly over testing implementation details.
- **Mocking**

  - Use mocking sparingly. Prefer real code paths and pure functions where possible.
  - Only introduce mocks/stubs when needed to satisfy isolation and “avoid external dependencies” requirements (e.g., external services, heavy resources).
- **Organization**

  - Organize tests into modules and classes in a way that mirrors the code under test where practical (e.g., `tests/test_module_name.py` for `module_name.py`).
  - Use Pytest fixtures for common setup where it improves clarity and reduces duplication, while keeping fixture scope as narrow as possible.

---

## 3. Naming and Readability (Python)

- **Naming conventions**

  - Use descriptive `test_...` function names that clearly express the scenario and expected outcome.
  - Group related tests logically within the same file or test class.
- **Docstrings and comments**

  - Where the intent is not obvious from the name alone, include a short docstring or comment summarizing:
    - The scenario being tested.
    - The expected outcome.

---

## 4. Respecting the Toolchain Loop

- When running the “After Making Changes” toolchain loop from the general code change policy on Python work, your **testing step** must be performed with **Pytest**.
- Approved command: `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
- Do **not** substitute other test runners or frameworks for Python code unless explicitly instructed to do so.

This file defines **how** Python tests are written and structured; the general code change policy defines **when** the toolchain (including tests) must be run and how strictly that loop must be followed.






