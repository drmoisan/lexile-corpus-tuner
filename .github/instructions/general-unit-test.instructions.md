applyTo: "**"
name: general-unit-test-policy
description: "Baseline unit test policy that applies to all languages in this repo"
---

# General Unit Test Policy

This policy applies to **all unit tests** in this repository, regardless of language or framework.

Every new or modified unit test must adhere to these guidelines.

---

## 1. Core Principles

- **Independence**  
  Tests must be able to run in any order without impacting each other.

- **Isolation**  
  Each unit test should target a single function, method, or unit of behavior so failures clearly identify the faulty unit.

- **Fast Execution**  
  Tests must be fast enough to support frequent runs and rapid feedback loops.

- **Determinism**  
  Given the same inputs and environment, tests must produce the same results. Avoid flakiness.

- **Readability and Maintainability**  
  Test names, structure, and assertions should be clear and easy to understand.

---

## 2. Coverage and Scenarios

- **Comprehensive Coverage (within reason)**  
  - These coverage expectations apply across all languages in the repo.
  - Aim to exercise critical paths and important edge conditions.
  - Configure coverage tooling to exclude test files (e.g., `tests/`), so metrics reflect the application code, not the tests themselves.
  - Repository-wide line coverage must remain `>= 80%`.
  - Any new modules, classes, or methods added must target `>= 90%` coverage.
  - Code changes or refactors must not reduce coverage for the lines that were changed.
  - Coverage is a supporting metric, not the sole quality gate; untested critical behavior is not acceptable even if the overall percentage looks good.

- **Scenario Completeness**  
  For each unit or behavior, tests should cover:
  - Positive flows with valid inputs.
  - Negative flows for invalid or missing inputs.
  - Edge cases and boundary conditions.
  - Error-handling behavior.
  - Concurrency behavior when relevant.
  - State transitions for stateful components.

---

## 3. Test Structure and Diagnostics

- **Clear Failure Messages**  
  Assertions should produce clear, actionable failure messages that make it easy to see what went wrong.

- **Arrange–Act–Assert pattern**  
  Organize tests into:
  - *Arrange* — set up inputs, environment, and dependencies.
  - *Act* — execute the behavior under test.
  - *Assert* — verify outcomes via assertions.

- **Document Intent**  
  Each test must clearly communicate its purpose:
  - Use descriptive test names, and/or
  - Include a short docstring or comment summarizing the scenario and expected outcome.

---

## 4. External Dependencies and Environment

- **Avoid External Dependencies**  
  Unit tests must not depend on external services such as databases, networks, remote APIs, or external processes.

- **Use Mocks / Stubs as Needed**  
  When code interacts with external systems or heavy resources, use mocks, stubs, or fakes to isolate the unit under test.

- **Environment Stability**  
  Tests must not rely on mutable global state or external configuration that can change between runs. Creation and use of temporary files on the local filesystem is expressly prohibited unless explicitly authorized as an exception. 
  - Currently approved exceptions: none.
  - If an exception is ever approved, list it explicitly here. A possible future example would be a static, read-only sample file committed to the repo and reused without runtime creation; this is not approved today.

---

## 5. Policy Audit

Before submitting any change that includes unit tests:

- Review each new or modified test against this policy.
- Confirm that:
  - It is independent, isolated, fast, and deterministic.
  - It is readable and clearly documents its intent.
  - It covers relevant positive, negative, edge, and error scenarios.
  - It does not rely on external dependencies without proper mocking/stubbing.

If any test cannot comply with these rules for a good reason, **call out the exception explicitly** in the change description.







