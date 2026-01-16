---
plan_id: copilot-cli-permission-failure-83-remediation
issue_id: 83
owner: drmoisan
created_at: 2026-01-14T22-35:00Z
updated_at: 2026-01-15T22-40:00Z
status: Completed
status_color: green
status_badge: "![status: Completed](https://img.shields.io/badge/status-Completed-green)"
spec_path: docs/features/active/2026-01-11-copilot-cli-permission-failure-83/v2/spec.md
plan_path: docs/features/active/2026-01-11-copilot-cli-permission-failure-83/v2/plan.2026-01-14T22-35.md
primary_code_paths:
  - scripts/dev_tools/atomic_executor/prompt_builder.py
  - tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py
optional_code_paths:
  - scripts/dev_tools/atomic_executor/cli.py
  - docs/features/active/2026-01-11-copilot-cli-permission-failure-83/v2/spec.md
  - docs/features/active/2026-01-11-copilot-cli-permission-failure-83/v2/issue.md
---

# copilot-cli-permission-failure-83-remediation (Implementation Plan)

![status: Planned](https://img.shields.io/badge/status-Planned-blue)

## Introduction

This plan remediates Issue #83 by updating the prompt emitted by
`scripts.dev_tools.atomic_executor.prompt_builder.PromptBuilder.build()` so that
Copilot sessions do not hit the Copilot CLI “Permission denied and could not request
permission from user” dead-end caused by Poetry script entrypoints.

### Root Cause (deterministic statement)

Copilot CLI path permission enforcement (including symlink resolution for existing
files) blocks the Poetry script entrypoint interpreter chain in this devcontainer.
The prompt currently instructs `poetry run black --check` / `poetry run ruff check` /
`poetry run pyright` / `poetry run pytest` and includes interactive-only `/model`
guidance, which increases the likelihood of triggering the headless permission dead-end.

### Non-negotiable repo constraints (inlined; no external dependencies to understand)

- Unit tests must not create temporary files or temporary directories.
- Python toolchain loop must be executed in this exact order and repeated until a
  single clean pass completes: **Black → Ruff → Pyright → Pytest (+ coverage)**.
- New/changed behavior must be driven by failing unit tests first (TDD). Refactors that
  do not change behavior may occur before new failing tests.

## Requirements Traceability (machine-parseable)

| REQ_ID | Source | Requirement (deterministic) | Verification (deterministic) |
| --- | --- | --- | --- |
| REQ-001 | spec.md: Acceptance Criteria | Prompt QC instructions MUST use `python -m poetry run black .`, `python -m poetry run ruff check`, `python -m poetry run pyright`, and `python -m poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`, and MUST NOT include any `poetry run <tool>` forms (including `poetry run black .`). | Unit test asserts `"python -m poetry run black ."` is present and `"poetry run black"` is absent. |
| REQ-002 | spec.md: Acceptance Criteria | Prompt MUST NOT include `/model` anywhere (interactive-only instruction). | Unit test asserts `"/model"` is absent from the built prompt when `preferred_model` is set. |
| REQ-003 | spec.md: Acceptance Criteria | Prompt MUST NOT include the phrase `"interactive session"` anywhere. | Unit test asserts `"interactive session"` is absent from the built prompt when `preferred_model` is set. |
| REQ-004 | spec.md: Root Cause Analysis | `<feature>` placeholder substitution MUST produce correct “Authoritative Documents” paths when `feature_dir` is under `workspace / "docs" / "features" / "active"` by using the POSIX relative path under that root (example: `2026-01-06-populate-open-stax-ck-12-manifest-73/v4`). | Unit test asserts `<feature>` substitution uses `feature_dir.relative_to(workspace / "docs" / "features" / "active").as_posix()` semantics. |
| REQ-005 | repo policy | Tests for this remediation MUST NOT use `tmp_path` or write files at runtime. | `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py` contains no `tmp_path` fixture usage and no `.write_text()` calls. |
| REQ-006 | spec.md: Acceptance Criteria | End-to-end `execute-all` repro MUST NOT contain the Copilot permission-denied substring `Permission denied and could not request permission from user` anywhere under `.agent_logs/`. | Run `python -m poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10` and then run `grep -R "Permission denied and could not request permission from user" .agent_logs` and assert zero matches. |
| REQ-007 | spec.md: Proposed Fix (conditional) | If prompt remediation is insufficient and the repro still hits the permissions dead-end, executor MUST add `--allow-all-paths` to Copilot argv as a last-resort headless workaround. | After implementing `--allow-all-paths`, rerun the same repro and assert substring is absent. |

## Implementation Phases (atomic tasks)

Notes:
- Each checkbox line is a machine-stable atomic task ID required by the executor: `[P#-T#]`.
- Each task is additionally labeled with a `TASK-###` identifier and references one or more `REQ-###` requirements.
- “Line references” are the current line numbers at plan-writing time. If line numbers drift,
  the symbol names and exact string anchors are authoritative.


**Phase 0 — Context & Baselines (mandatory, read-only)**

- [x] [P0-T1] TASK-001 Read repo Copilot instructions (policy precondition) (REQ-005)
  - File: `.github/copilot-instructions.md`
  - Acceptance: confirm the “no secrets” and “follow repo policies” constraints are understood.

- [x] [P0-T2] TASK-002 Read general code-change policy (policy precondition) (REQ-005)
  - File: `.github/instructions/general-code-change.instructions.md`
  - Acceptance: confirm the required toolchain loop order (format → lint → type-check → test) and restart rules are understood.

- [x] [P0-T3] TASK-003 Read general unit-test policy (policy precondition) (REQ-005)
  - File: `.github/instructions/general-unit-test.instructions.md`
  - Acceptance: confirm determinism/isolation requirements and “no temporary files in tests” constraints are understood.

- [x] [P0-T4] TASK-004 Read Python code-change policy (policy precondition) (REQ-005)
  - File: `.github/instructions/python-code-change.instructions.md`
  - Acceptance: confirm approved commands for Black/Ruff/Pyright and suppression rules are understood.

- [x] [P0-T5] TASK-005 Read Python unit-test policy (policy precondition) (REQ-005)
  - File: `.github/instructions/python-unit-test.instructions.md`
  - Acceptance: confirm approved pytest + coverage command and test organization rules are understood.

- [x] [P0-T6] TASK-006 Capture current line anchors for the prompt literals that must change (REQ-001, REQ-002, REQ-003, REQ-004)
  - Target file: `scripts/dev_tools/atomic_executor/prompt_builder.py`
  - Required anchors (current line numbers):
    - Model section `/model` instruction at line 125 and line 131 (string contains `/model` and `interactive session`).
    - `<feature>` substitution uses `feature_dir.name` at line 142–144.
    - QC toolchain lines include `poetry run black --check` at line 183 and `poetry run ruff/pyright/pytest` at lines 184–186.
  - Acceptance: record these line numbers and anchors in the PR description or commit notes.

- [x] [P0-T7] TASK-007 Capture baseline unit test failures for prompt builder tests (REQ-005)
  - Command (preferred for this environment): `python -m poetry run pytest tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py -q`
  - Acceptance: capture the full output and confirm the file currently uses `tmp_path` (see `test_prompt_builder.py` line 15+).

- [x] [P0-T8] TASK-008 Capture baseline QC statuses (Black, Ruff, Pyright, Pytest) (REQ-006)
  - Commands (use these exact commands in this order):
    - `python -m poetry run black .`
    - `python -m poetry run ruff check`
    - `python -m poetry run pyright`
    - `python -m poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
  - Acceptance: save raw outputs (pass/fail + key error summaries) into the PR notes or artifacts.

---

**Phase 1 — Refactor seams for in-memory tests (no behavior change intended)**

- [x] [P1-T1] TASK-010 Add `PromptBuilderFileSystem` protocol in `scripts/dev_tools/atomic_executor/prompt_builder.py` (REQ-005)
  - Target symbol: `class PromptBuilder` (starts at line 19).
  - Add new symbol (top-level): `class PromptBuilderFileSystem(Protocol):`.
  - Required protocol methods (exact signatures):
    - `def is_file(self, path: Path) -> bool: raise NotImplementedError`
    - `def is_dir(self, path: Path) -> bool: raise NotImplementedError`
    - `def read_text(self, path: Path) -> str: raise NotImplementedError`
    - `def glob(self, directory: Path, pattern: str) -> list[Path]: raise NotImplementedError`
  - Acceptance: `python -m poetry run pyright` recognizes the protocol and all methods are fully typed.

- [x] [P1-T2] TASK-011 Add `RealPromptBuilderFileSystem` implementation in `scripts/dev_tools/atomic_executor/prompt_builder.py` (REQ-005)
  - Required behavior: delegate to `path.is_file()`, `path.is_dir()`, `path.read_text(encoding="utf-8")`, and `sorted(directory.glob(pattern))`.
  - Acceptance: no code outside the new class changes behavior.

- [x] [P1-T3] TASK-012 Add injectable `fs` parameter to `PromptBuilder.__init__` with default `RealPromptBuilderFileSystem()` (REQ-005)
  - Target symbol: `PromptBuilder.__init__`.
  - Exact parameter addition: `fs: PromptBuilderFileSystem | None = None`.
  - Exact defaulting rule: `self._fs = fs or RealPromptBuilderFileSystem()`.
  - Acceptance: existing call sites compile unchanged and unit tests not yet migrated still pass.

- [x] [P1-T4] TASK-013 Add injectable `plan_resolver` parameter to `PromptBuilder.__init__` with default `resolve_feature_plan` (REQ-005)
  - Target symbol: `PromptBuilder.__init__`.
  - Exact parameter addition: `plan_resolver: Callable[[Path], ResolvedPlan] | None = None`.
  - Exact defaulting rule: `self._plan_resolver = plan_resolver or resolve_feature_plan`.
  - Acceptance: `PromptBuilder.build()` can be refactored to call `self._plan_resolver(feature_dir)` without changing behavior.

- [x] [P1-T5] TASK-014 Replace all direct filesystem reads/checks in `PromptBuilder` with `self._fs` calls (REQ-005)
  - Target symbols:
    - `PromptBuilder.__init__` template validation MUST use `self._fs.is_file(template_path)`.
    - `PromptBuilder.build` MUST use `self._fs.read_text(path)` and `self._fs.is_file(path)`.
    - Instruction discovery MUST replace `instructions_dir.is_dir()` and `instructions_dir.glob("*.instructions.md")` with `self._fs.is_dir(instructions_dir)` and `self._fs.glob(instructions_dir, "*.instructions.md")`.
  - Acceptance: `python -m poetry run pytest tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py -q` still passes before migrating tests.

- [x] [P1-T6] TASK-015 Add `InMemoryPromptBuilderFileSystem` test helper in `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py` (REQ-005)
  - Required API: implement `PromptBuilderFileSystem` and store:
    - `files: dict[str, str]` keyed by `path.as_posix()`.
    - `dirs: set[str]` keyed by `path.as_posix()`.
  - Required semantics:
    - `is_file(path)` returns True iff `path.as_posix()` in `files`.
    - `is_dir(path)` returns True iff `path.as_posix()` in `dirs`.
    - `read_text(path)` returns `files[path.as_posix()]` or raises `FileNotFoundError`.
    - `glob(directory, pattern)` supports only `"*.instructions.md"` by returning Paths under the directory whose filenames match and are present in `files`.
  - Acceptance: helper is pure in-memory and uses no filesystem calls.

---

**Phase 2 — Remove filesystem temp usage from PromptBuilder tests (MUST complete before new regression tests)**

- [x] [P2-T1] TASK-020 Update `TestPromptBuilderInit.test_init_with_valid_template` to in-memory filesystem + injected `fs` (REQ-005)
  - File: `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py` (tmp_path usage starts at line 15+).
  - Replace `tmp_path` fixture usage with:
    - `workspace = Path("/workspace")` (literal test path only; no filesystem).
    - `template_path = Path("/workspace/template.md")`.
    - `fs = InMemoryPromptBuilderFileSystem(files={template_path.as_posix(): "Template content"}, dirs=set())`.
    - Construct `PromptBuilder(workspace=workspace, template_path=template_path, fs=fs, plan_resolver=lambda feature_dir: ResolvedPlan(path=feature_dir / "plan.md", display_label="plan.md", update_filename="plan.md"))`.
  - Acceptance: test asserts `builder.workspace == workspace` and `builder.template_path == template_path`.

- [x] [P2-T2] TASK-021 Update `TestPromptBuilderInit.test_init_raises_for_nonexistent_template` to in-memory (REQ-005)
  - Required setup: `fs` has no entry for the template path.
  - Acceptance: `FileNotFoundError` is raised with the same message substring.

- [x] [P2-T3] TASK-022 Update `TestPromptBuilderInit.test_init_raises_for_directory_template` to in-memory (REQ-005)
  - Required setup: `fs.dirs` contains the template path and `fs.files` does not.
  - Acceptance: `FileNotFoundError` is raised.

- [x] [P2-T4] TASK-023 Update `TestPromptBuilderBuild.test_build_combines_template_and_context` to in-memory + injected `plan_resolver` (REQ-005)
  - Required fake plan resolver: `lambda feature_dir: ResolvedPlan(path=Path("/workspace/docs/features/active/my-feature/plan.md"), display_label="plan.md", update_filename="plan.md")`.
  - Required in-memory contents: template, plan.md, spec.md, user-story.md.
  - Acceptance: existing prompt assertions still pass (no behavior change intended in this phase).

- [x] [P2-T5] TASK-024 Update `TestPromptBuilderBuild.test_build_handles_missing_user_story` to in-memory (REQ-005)
  - Required setup: omit `user-story.md` from `fs.files`.
  - Acceptance: prompt includes the user-story section markers with empty content.

- [x] [P2-T6] TASK-025 Update `TestPromptBuilderBuild.test_build_raises_for_missing_plan` to in-memory (REQ-005)
  - Required setup: fake plan resolver returns a plan path that is absent from `fs.files`.
  - Acceptance: `FileNotFoundError` matches "Missing required plan file".

- [x] [P2-T7] TASK-026 Update `TestPromptBuilderBuild.test_build_raises_for_missing_spec` to in-memory (REQ-005)
  - Required setup: omit spec.md path from `fs.files`.
  - Acceptance: `FileNotFoundError` matches "Missing required spec.md".

- [x] [P2-T8] TASK-027 Update `TestPromptBuilderBuild.test_build_injects_task_details` to in-memory (REQ-005)
  - Acceptance: prompt contains `- [P2-T5] Important task` and the CURRENT TASK header.

- [x] [P2-T9] TASK-028 Update `TestPromptBuilderBuild.test_build_includes_toolchain_instructions` to in-memory (REQ-005)
  - Acceptance: keep current assertions unchanged in this phase (still expects `poetry run black --check`, `poetry run ruff check`, `poetry run pyright`, and `poetry run pytest` strings).

- [x] [P2-T10] TASK-029 Update `TestPromptBuilderBuild.test_build_includes_plan_update_instructions` to in-memory (REQ-005)
  - Acceptance: existing assertions still pass.

- [x] [P2-T11] TASK-030 Update `TestPromptBuilderEdgeCases.test_build_handles_empty_template` to in-memory (REQ-005)
  - Acceptance: existing assertions still pass.

- [x] [P2-T12] TASK-031 Update `TestPromptBuilderEdgeCases.test_build_handles_empty_plan` to in-memory (REQ-005)
  - Acceptance: existing assertions still pass.

- [x] [P2-T13] TASK-032 Update `TestPromptBuilderEdgeCases.test_build_uses_posix_paths_in_output` to in-memory (REQ-005)
  - Acceptance: existing assertions still pass.

- [x] [P2-T14] TASK-033 Update `TestPromptBuilderEdgeCases.test_read_text_helper_uses_utf8` to in-memory (REQ-005)
  - Acceptance: existing Unicode assertions still pass.

- [x] [P2-T15] TASK-034 Verify `test_prompt_builder.py` contains no `tmp_path` usage and no `.write_text()` calls (REQ-005)
  - Verification rule (deterministic): `grep -n "tmp_path" tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py` returns zero matches AND `grep -n "write_text" tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py` returns zero matches.
  - Acceptance: both grep commands return no matches.

---

**Phase 3 — Add failing regression tests (TDD; MUST fail before Phase 4 changes)**

Target under test: `scripts.dev_tools.atomic_executor.prompt_builder.PromptBuilder.build()` (function starts at line 66).

- [x] [P3-T1] TASK-040 Update toolchain assertion test to require `python -m poetry run` forms and forbid `poetry run` forms (REQ-001)
  - File: `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py`
  - Update existing test: `TestPromptBuilderBuild.test_build_includes_toolchain_instructions`.
  - Replace assertions:
    - Replace `assert "poetry run black --check" in prompt` with `assert "python -m poetry run black ." in prompt`.
    - Add `assert "poetry run black" not in prompt`.
  - Acceptance: test fails against current `prompt_builder.py` because it still emits `poetry run black --check` / `poetry run ruff check` / `poetry run pyright` / `poetry run pytest` lines (see `prompt_builder.py` lines 183–186).

- [x] [P3-T2] TASK-041 Add assertion that built prompt does not contain `/model` (REQ-002)
  - File: `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py`
  - Add assertion to `test_build_combines_template_and_context` OR add a new test named
    `test_build_does_not_include_model_command_when_preferred_model_is_set`.
  - Required construction: create `PromptBuilder(workspace=workspace, template_path=template_path, preferred_model="gpt-5.1-codex-max", fs=fs, plan_resolver=fake_plan_resolver)`.
  - Assertion: `assert "/model" not in prompt`.
  - Acceptance: test fails against current `prompt_builder.py` because the model section contains `/model` (line 125+).

- [x] [P3-T3] TASK-042 Add assertion that built prompt does not contain `"interactive session"` (REQ-003)
  - File: `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py`
  - Assertion: `assert "interactive session" not in prompt`.
  - Acceptance: test fails against current `prompt_builder.py` because the model section contains that phrase (line 131+).

- [x] [P3-T4] TASK-043 Add failing test for `<feature>` substitution when feature_dir is under `workspace / "docs" / "features" / "active"` (REQ-004)
  - File: `tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py`
  - Required in-memory paths:
    - `workspace = Path("/workspace")`
    - `feature_dir = Path("/workspace/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4")`
  - Required template input string: include `<feature>` literal in the template body.
  - Expected replacement string: `"2026-01-06-populate-open-stax-ck-12-manifest-73/v4"`.
  - Acceptance: test fails against current `prompt_builder.py` because `<feature>` is replaced with only `feature_dir.name` (line 142).

- [x] [P3-T5] TASK-044 Run prompt builder tests and confirm Phase 3 tests fail (REQ-001..REQ-004)
  - Command: `python -m poetry run pytest tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py -q`
  - Acceptance: failures explicitly reference the mismatched strings.

---

**Phase 4 — Minimal prompt fixes to satisfy regressions (code changes)**

- [x] [P4-T1] TASK-050 Replace QC toolchain instruction lines in `PromptBuilder.build()` (REQ-001)
  - File: `scripts/dev_tools/atomic_executor/prompt_builder.py`
  - Target block: appended prompt envelope where QC commands are listed (currently lines 180–186).
  - Required exact replacements:
    - Replace the entire bullet list of per-file commands with these four exact full-repo commands:
      - `python -m poetry run black .`
      - `python -m poetry run ruff check`
      - `python -m poetry run pyright`
      - `python -m poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
  - Add deterministic fallback sentence immediately above the list:
    - `If any python -m poetry command fails, retry the same command with python3 -m poetry.`
  - Acceptance: `TASK-040` passes and the substring `poetry run black` does not appear anywhere in the prompt.

- [x] [P4-T2] TASK-051 Remove interactive-only `/model` guidance from `PromptBuilder.build()` (REQ-002, REQ-003)
  - File: `scripts/dev_tools/atomic_executor/prompt_builder.py`
  - Target block: `model_section` construction (currently line 120+).
  - Required new behavior:
    - When `preferred_model` is set, include a non-interactive statement:
      - `Preferred model (already selected by executor): {self.preferred_model}`
    - MUST NOT include `/model` and MUST NOT include the phrase `interactive session`.
  - Acceptance: `TASK-041` and `TASK-042` pass.

- [x] [P4-T3] TASK-052 Implement deterministic `<feature>` placeholder substitution for feature folders under docs/features/active (REQ-004)
  - File: `scripts/dev_tools/atomic_executor/prompt_builder.py`
  - Target lines: feature placeholder replacement uses `feature_dir.name` (currently line 142–144).
  - Required algorithm (exact):
    - Define `active_root = self.workspace / "docs" / "features" / "active"`.
    - If `feature_dir.is_relative_to(active_root)` (or equivalent try/except `relative_to`):
      - `feature_name = feature_dir.relative_to(active_root).as_posix()`.
    - Else:
      - `feature_name = feature_dir.name`.
    - Replace `<feature>` with `feature_name`.
  - Acceptance: `TASK-043` passes.

---

**Phase 5 — Verify regressions + full toolchain loop (mandatory)**

- [x] [P5-T1] TASK-060 Re-run prompt builder tests and confirm all pass (REQ-001..REQ-005)
  - Command: `python -m poetry run pytest tests/scripts/dev_tools/atomic_executor/test_prompt_builder.py -q`
  - Acceptance: all tests pass.

- [x] [P5-T2] TASK-061 Run the full repo toolchain loop until one clean pass completes (REQ-006)
  - Toolchain pass commands (must run in this order):
    - `python -m poetry run black .`
    - `python -m poetry run ruff check`
    - `python -m poetry run pyright`
    - `python -m poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
  - Deterministic restart rule:
    - If Black changes files, restart the pass from Black.
    - If Ruff fails or applies fixes, restart the pass from Black.
    - If Pyright fails, restart the pass from Black.
    - If Pytest fails, restart the pass from Black.
  - Acceptance: record the final pass outputs showing all four steps succeeded without further changes.

---

**Phase 6 — End-to-end evidence (must satisfy REQ-006)**

- [x] [P6-T1] TASK-070 Generate a real prompt via atomic executor `--print-prompt` and assert the fixed strings (REQ-001..REQ-004)
  - Command:
    - `python -m poetry run python -m scripts.dev_tools.atomic_executor.cli execute --print-prompt /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max`
  - Acceptance criteria (exact string checks):
    - Output contains `python -m poetry run black .`.
    - Output contains the fallback sentence about `python3 -m poetry`.
    - Output contains neither `/model` nor `interactive session`.

- [x] [P6-T2] TASK-071 Re-run a representative `execute-all` repro and assert no permission-denied substring appears (REQ-006)
  - Repro command (use this exact pattern):
    - `python -m poetry run python -m scripts.dev_tools.atomic_executor.cli execute-all /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-06-populate-open-stax-ck-12-manifest-73/v4/ --workspace /workspaces/lexile-corpus-tuner --preferred-model gpt-5.1-codex-max --max-fix-attempts 10`
  - Verification rule (deterministic): grep `.agent_logs/` for `Permission denied and could not request permission from user` and assert zero matches.

**Phase 7 — Conditional fallback: broaden Copilot path permissions only if REQ-006 fails (REQ-007)**

- [x] [P7-T1] TASK-080 Run the Phase 6 repro once and branch deterministically on the result (REQ-007)
  - If `.agent_logs/` contains `Permission denied and could not request permission from user`, proceed to `TASK-081`.
  - If `.agent_logs/` contains zero matches, mark `TASK-081` and `TASK-082` as skipped with note: `SKIPPED: REQ-006 satisfied without allow-all-paths.`
  - **RESULT**: Latest log `atomic_executor_2026-01-15_221048.log` contains 0 matches. REQ-006 satisfied.

- [x] [P7-T2] TASK-081 Add `--allow-all-paths` to Copilot argv in `scripts/dev_tools/atomic_executor/cli.py` (REQ-007)
  - SKIPPED: REQ-006 satisfied without allow-all-paths.

- [x] [P7-T3] TASK-082 Re-run Phase 6 repro and assert substring is absent (REQ-007)
  - SKIPPED: REQ-006 satisfied without allow-all-paths.

---

**Phase 8 — Documentation updates (required for closure)**

- [x] [P8-T1] TASK-090 Update `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/v2/spec.md` status and acceptance criteria evidence (REQ-001..REQ-007)
  - Required edits:
    - Change status line from `Implemented (partial) — remediation required` to `Completed`.
    - Add a short post-fix evidence excerpt showing:
      - prompt includes `python -m poetry run black .`
      - prompt excludes `/model`
      - execute-all log contains no `Permission denied and could not request permission from user`
  - Acceptance: spec reflects final behavior and contains evidence excerpts.

- [x] [P8-T2] TASK-091 Update `docs/features/active/2026-01-11-copilot-cli-permission-failure-83/v2/issue.md` with post-fix evidence excerpt (REQ-006)
  - Acceptance: issue includes a minimal excerpt from the latest `.agent_logs/atomic_executor_*.log` (use `tail -n 200 "$(ls -1 .agent_logs/atomic_executor_*.log | sort | tail -n 1)"`) showing the repro run completed without the permission-denied substring.
