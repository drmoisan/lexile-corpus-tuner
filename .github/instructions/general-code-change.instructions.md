---
applyTo: "**"
name: "general-code-change-policy"
description: "Baseline rules that apply to any code change in this repo"
---

# Agent Code Change Policy

**CRITICAL**: When implementing any **any** code, tests, tasks, or scripts, you **must** adhere to these repo policies **without exception**. This includes but is not limited to adding, removing, or changing any code, tasks, scripts, modules, packages, tests or their components.

Read each policy document **thoroughly** before starting work. Implement them **exactly as written**. Do not interpret, modify, or skip any requirements. If you encounter **any** conflicting instructions, halt and notify the user.

Language-specific standards (e.g. for Python) are defined in additional instructions files and **layer on top of** this general policy.

**Reading order / authority:** Apply this general policy first, then any language-specific code-change instructions, then any unit-test addenda. Operational guidance (e.g., developer tooling, CI docs) sits underneath these policies.

## Before Making Changes

- [ ] Clarify the objective. Begin reasoning from clearly stated assumptions or axioms.
- [ ] Read existing change plans (e.g., `change-plan.md`).
- [ ] Document the plan to make changes. If it is part of an existing change plan, make any relevant updates to the plan before executing.

---

## Bugfix Workflow (all languages, defects only)

Use this workflow only when addressing a bug or defect. Feature work, refactors, and
new capabilities should follow the general planning steps and design principles
rather than this bugfix sequence.

1. **Create a failing regression test first**
   - Add the smallest deterministic test that reproduces the bug using the project’s standard test layout (prefer the module’s existing test file; use `tests/bugs/<YYYY>/<issue>-<desc>.py` only when no clear home exists).
   - Ensure the test fails before the fix and will pass after; avoid external services or temporary files.

2. **Implement the minimal, targeted fix**
   - Change only what is needed to make the failing test pass; keep boundaries intact and avoid opportunistic refactors.
   - If you uncover deeper design problems, open a new issue instead of widening scope; add logging only when it materially aids diagnosis.

3. **Verify locally before review**
   - Re-run the original repro and the new regression test.
   - Run the full toolchain in order (format → lint → type-check → test) using the repo-standard commands or tasks; rerun from the start if any step changes files or fails.

---

## 1. Design Principles

High-level design priorities (applies to all languages):

1. **Simplicity first**

   - Prefer the simplest design that works and is easy to read.
   - Avoid cleverness and deep indirection. The next maintainer should be able to understand a module in one reading.

2. **Reusability**

   - Factor out logic that is clearly reusable into small methods or pure functions.
   - Avoid copy-paste; share behavior via composition, helper methods, or shared base classes/interfaces.

3. **Extensibility**

   - Design public APIs so they can be extended without breaking callers:
     - Prefer keyword-style parameters with defaults (or equivalent in the language).
     - Prefer composition over inheritance when possible.
     - Use interfaces/abstract types/protocols to support multiple implementations behind an interface.

4. **Separation of concerns**

   - Keep **pure logic** (transforms, calculations, parsing) separate from:
     - I/O (disk, network, DB)
     - UI / CLI
     - Framework-specific glue
   - Orchestration code (e.g., “main” pipeline classes) may depend on many things; pure core logic should depend on very little.

---

## 2. Classes, Functions, and APIs

**Overall rule:**  
Use **strongly-typed, well-structured classes** to model domain concepts and workflows. Use **functions** (or equivalent) for small, stateless helpers and glue code.

### 2.1 Prefer classes for domain concepts and workflows

Create a class when at least one is true:

- There is a **clear domain concept** with data + behavior  
  - e.g. “transaction”, “corpus”, “contact matcher”, “pipeline”.
- You have **state + invariants** that should travel together  
  - e.g. a model that must keep weights, vocabulary, and metadata in sync.
- You expect **multiple implementations** behind a common interface  
  - e.g. different text sources, storage backends, or pipelines.
- You are modeling a **multi-step workflow** that shares context  
  - e.g. `download()`, `normalize()`, `index()`, `export()` steps on a pipeline object.

When you use classes:

- Keep methods **small and focused**; a method should do one conceptual thing.
- Avoid “god objects” that know about too many unrelated concerns.

### 2.2 Use functions for small, pure helpers

Create a standalone function when:

- The operation is **pure, stateless, and simple**:
  - e.g. “normalize whitespace in this string”
  - e.g. “compute a score from inputs”
- It’s a **small helper** that doesn’t naturally belong on a specific domain class.
- It is a **simple transformation** from inputs to outputs.

Rules for functions:

- Functions should be short, readable, and clearly named by what they do.
- Avoid long, deeply branching functions—factor logic into smaller helpers.

### 2.3 Interfaces and contracts

- Use interfaces / abstract types / protocols when multiple implementations are likely (e.g. different storage backends or text sources).
- Public methods and functions must have clear, documented contracts (inputs, outputs, invariants).

---

## 3. Error Handling, Logging, and Contracts

1. **Error handling**

   - Fail **fast and explicitly**: raise or return clear, specific errors when invariants are violated.
   - Don’t silently ignore errors or broad-catch (e.g. a “catch all”) unless you immediately re-raise or propagate with added context.

2. **Logging**

   - Use the project’s logging pattern instead of ad-hoc `print`/console output.
   - Log at appropriate levels (`debug`, `info`, `warning`, `error`) and include enough context to debug issues.

3. **Contracts / invariants**

   - Enforce invariants at construction/initialization time.
   - Use assertions only for **internal sanity checks**, not user-facing error handling.

---

## 4. Module & File Structure

1. Keep modules **cohesive**:

   - A module/file should have a clear purpose (e.g. “QIF parsing,” “Lexile model,” “corpus download”).
   - Avoid dumping unrelated classes/functions into the same file.
   - Do not exceed 500 lines for any one file.
   - This 500-line limit applies to production code, test code, and reusable scripts.
   - Exceptions: temporary throwaway scripts created and deleted during an agent session; raw text fixtures used for language-processing test data; Markdown documentation files.

2. Public vs internal

   - Make the public surface area **small and intentional**.
   - Use “internal” helpers and naming conventions (e.g. underscore-prefix or equivalent) for things that should not be used outside the module.

3. Imports / dependencies

   - Prefer clear, explicit imports within the project.
   - Avoid circular dependencies; if they appear, refactor shared logic into a lower-level module.

---

## 5. Naming, Docs, and Comments

1. Naming

   - Names should be descriptive, not cryptic.
   - Abbreviations are okay only when they are standard and widely understood (e.g. `id`, `url`, `db`).

2. Docs / docstrings

   - Public classes and methods should have a short description covering:
     - What it does.
     - Important arguments/parameters.
     - What it returns or side effects.

3. Comments

   - Comment **why**, not what. The code should generally explain *what*.
   - If you use workarounds or non-obvious patterns, add a short comment explaining the reasoning.

---

## 6. Performance, I/O, and Dependencies

1. **Performance**

   - Prefer clarity first; optimize only where there is a demonstrated need.
   - Avoid obviously quadratic (O(N²)) or worse algorithms on large inputs unless justified.

2. **I/O boundaries**

   - Isolate I/O (disk, network, APIs) into specific classes or modules.
   - Core domain logic should be testable **without** touching the network or filesystem. 
   - **Use of temporary files within tests is strictly prohibited**.

3. **Dependencies**

   - Use only the libraries already approved in the project unless specifically told to add more.
   - If adding a dependency is unavoidable, choose a well-maintained, widely used package, and document why it’s required.

---

## 7. How to Interact with Existing Code

1. **Follow existing patterns**

   - Where the repo already has a clear style (e.g. how pipelines or models are structured), **match that style**.
   - If you need to improve an existing pattern, keep it **compatible** with current usages.

2. **API changes**

   - Avoid breaking public APIs. If a breaking change is necessary, call it out clearly in comments or the PR description.

3. **Tests as specification**

   - Treat existing unit tests as **part of the spec**.
   - When adding new behavior, add tests that make the behavior explicit (using the language’s standard test framework).

---

## 8. After Making Changes

### 1. Run the full toolchain (no shortcuts)

You **must** run the full toolchain in this exact order and repeat it until everything passes:

1. **Formatting**
2. **Linting**
3. **Type checking**
4. **Testing**

Treat these four steps as one **toolchain pass**.

1. Run the formatter on the relevant files (e.g. Black).

2. Run the linter (e.g. Ruff).
   - If the linter **fails** or **auto-fixes** anything:
     - Fix all reported issues (including applying any auto-fixes).
     - Then **restart the toolchain pass from step 1 (Formatting)**.

3. Run the type checker (e.g. Pyright).
   - If type checking **fails**:
     - Fix all reported issues.
     - Then **restart the toolchain pass from step 1 (Formatting)**.

4. Run the tests (e.g. Pytest).
   - If any test **fails**:
     - Fix all reported issues.
     - Then **restart the toolchain pass from step 1 (Formatting)**.

You **may not stop** this loop while any of the following are true:

- Formatting would change the code.
- Linting reports errors.
- Type checking reports errors.
- Tests fail.

Only when **all four steps complete without errors in a single pass** are you allowed to consider the change complete.

When you report back, explicitly state:
- Which formatting, linting, type-checking, and test commands you ran, and  
- That all four steps passed without errors in the final pass.

---

### 2. Summarize key changes and rationale

- Summarize the key changes made and how they relate to the original objective.
- Explain any important design choices and other options you considered but did not implement.

---

### 3. Update supporting documents

- Update any supporting documents (e.g., README, design docs, runbooks).
- Update any workplan, change plan, or instructions document to show progress and reflect the new behavior.

---

### 4. Provide clear next steps

- Provide clear development next steps (what should happen next, and by whom).
- If development is complete, provide detailed instructions on usage and any operational caveats (limits, known issues, rollout steps).







