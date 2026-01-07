# 2026-01-07-unapproved-noqa (Plan)

- Issue: #78
- Owner: 2026-01-07-unapproved-noqa
- Date: 2026-01-07
- Status: Complete

**Phase 0 — Context & Inputs**
- [x] [P0-T1] Read .github/copilot-instructions.md and general-code-change.instructions.md to establish baseline rules
- [x] [P0-T2] Read .github/instructions/python-code-change.instructions.md to confirm Python standards
- [x] [P0-T3] Read .github/instructions/python-unit-test.instructions.md to confirm testing standards
- [x] [P0-T4] Conduct comprehensive grep search for all `# noqa` instances across Python codebase
- [x] [P0-T5] Analyze each noqa pattern to categorize: pre-authorized, needs authorization, or needs fix

**Phase 1 — Policy Documentation (Pre-Authorized Patterns)**
- [x] [P1-T1] Add ARG002 pre-authorized pattern (test mock unused args) to python-suppressions.instructions.md with required comment format
- [x] [P1-T2] Add B008 pre-authorized pattern (Typer Option() defaults) to python-suppressions.instructions.md with required comment format
- [x] [P1-T3] Add TCH002/TCH003 pre-authorized pattern (dual runtime+type imports) to python-suppressions.instructions.md with required comment format
- [x] [P1-T4] Add S310 pre-authorized pattern (trusted HTTPS urllib) to python-suppressions.instructions.md with required comment format
- [x] [P1-T5] Add S314 pre-authorized pattern (trusted XML parsing) to python-suppressions.instructions.md with required comment format
- [x] [P1-T6] Add BLE001 pre-authorized pattern (CLI top-level except) to python-suppressions.instructions.md with required comment format
- [x] [P1-T7] Add S301 pre-authorized pattern (trusted pickle load) to python-suppressions.instructions.md with required comment format
- [x] [P1-T8] Add S108/S105 pre-authorized pattern (test fixture literals) to python-suppressions.instructions.md with required comment format

**Phase 2 — Policy Documentation (Non-Authorized Patterns with Workarounds)**
- [x] [P2-T1] Add S110 non-authorized pattern with explicit platform detection workaround to python-suppressions.instructions.md
- [x] [P2-T2] Add TID252 non-authorized pattern with absolute import workaround to python-suppressions.instructions.md
- [x] [P2-T3] Add S607 non-authorized pattern with shutil.which() workaround to python-suppressions.instructions.md
- [x] [P2-T4] Add D401 non-authorized pattern with imperative mood workaround to python-suppressions.instructions.md
- [x] [P2-T5] Add F401 non-authorized pattern with remove/use workaround to python-suppressions.instructions.md
- [x] [P2-T6] Add UP017 non-authorized pattern with timezone-aware workaround to python-suppressions.instructions.md

**Phase 3 — Code Fixes (TID252: Relative Imports)**
- [x] [P3-T1] Convert relative import to absolute in gutenberg_query_builder_ui/widgets.py (QueryConstraintModel, QueryGroupModel) with E501 for line length
- [x] [P3-T2] Convert relative import to absolute in gutenberg_query_builder_ui/app.py (explore_gutenberg imports) with E501 for line length
- [x] [P3-T3] Convert relative import to absolute in gutenberg_query_builder_ui/app.py (gutenberg_query_core imports) with E501 for line length

**Phase 4 — Code Fixes (S607: Subprocess Validation)**
- [x] [P4-T1] Add shutil import to collect_commit_context.py
- [x] [P4-T2] Add shutil.which() validation at start of run_git() function with FileNotFoundError for missing git
- [x] [P4-T3] Replace ["git", *args] with [git_exe, *args] in subprocess.run call
- [x] [P4-T4] Update S607 suppression to S603 with proper comment explaining runtime validation
- [x] [P4-T5] Enhance run_git() docstring with Args, Returns, and Raises sections

**Phase 5 — Code Fixes (D401: Docstring Mood)**
- [x] [P5-T1] Change MockClipboard.copy() docstring in test_resolve_execute_plan_prompt.py from descriptive to imperative mood
- [x] [P5-T2] Change FakeIssueFetcher.__call__() docstring in test_new_active_feature_folder.py from descriptive to imperative mood
- [x] [P5-T3] Change FakeCodeLauncher.__call__() docstring in test_new_active_feature_folder.py from descriptive to imperative mood

**Phase 6 — Code Fixes (F401 & UP017)**
- [x] [P6-T1] Verify render imports in pr_context/collector.py are properly re-exported via __all__ and remove unnecessary F401 suppression
- [x] [P6-T2] Verify append_generation_timestamp() in summary_helpers.py already uses timezone.utc and remove unnecessary UP017 suppression
- [x] [P6-T3] Add proper docstring to append_generation_timestamp() explaining UTC timestamp generation

**Phase 7 — Test Updates (shutil.which() Mocking)**
- [x] [P7-T1] Add shutil import to test_collect_commit_context.py
- [x] [P7-T2] Replace lambda with typed mock_which() function in test_successful_command_returns_stdout and update assertion to expect /usr/bin/git
- [x] [P7-T3] Replace lambda with typed mock_which() function in test_failed_command_raises_by_default and update assertion to expect /usr/bin/git
- [x] [P7-T4] Replace lambda with typed mock_which() function in test_failed_command_returns_empty_when_allowed and update assertion to expect /usr/bin/git
- [x] [P7-T5] Replace lambda with typed mock_which() function in test_subprocess_run_called_with_correct_args and update assertion to expect /usr/bin/git
- [x] [P7-T6] Replace lambda with typed mock_which() function in test_handles_none_stdout and update assertion to expect /usr/bin/git

**Phase 8 — Test Updates (Platform & Clipboard)**
- [x] [P8-T1] Add sys.platform mock to test_copy_tries_multiple_fallback_commands in atomic_executor/test_cli.py to test Linux clipboard behavior
- [x] [P8-T2] Update test_copy_tries_multiple_fallback_commands docstring to clarify Linux testing scenario

**Phase 9 — Test Updates (Prompt Template Rename)**
- [x] [P9-T1] Update test_parse_execute_subcommand_with_path assertion in atomic_executor/test_cli.py from execute-atomic-plan.prompt.md to execute-plan-template.md
- [x] [P9-T2] Replace all 7 instances of execute-atomic-plan.prompt.md with execute-plan-template.md in test fixture creation code

**Phase 10 — Documentation Generation**
- [x] [P10-T1] Run sync-agents-from-instructions.ps1 to regenerate AGENTS.md with all new policy patterns
- [x] [P10-T2] Verify AGENTS.md includes all 11 pre-authorized patterns in correct format
- [x] [P10-T3] Verify AGENTS.md includes all 6 non-authorized patterns with workarounds

**Phase 11 — Audit Documentation**
- [x] [P11-T1] Create artifacts/noqa-audit-report.md documenting comprehensive pattern analysis
- [x] [P11-T2] Document all 100+ noqa instances categorized by status (pre-authorized, needs authorization, needs fix)
- [x] [P11-T3] Create artifacts/noqa-fixes-required.md with detailed fix plan, priority, and estimated effort

**Phase 12 — Verification (Type Checking)**
- [x] [P12-T1] Run Black formatter on all modified files to ensure consistent formatting
- [x] [P12-T2] Run Ruff linter to verify no new violations and all fixes applied correctly
- [x] [P12-T3] Run Pyright type checker to verify no type errors (especially lambda → typed function conversions)
- [x] [P12-T4] If Pyright fails, fix type errors and restart from P12-T1

**Phase 13 — Verification (Testing)**
- [x] [P13-T1] Run full Pytest suite to verify all 1156 tests pass
- [x] [P13-T2] Verify coverage remains ≥90% (target: maintain or improve from 87% baseline)
- [x] [P13-T3] If any tests fail, fix failures and restart from P12-T1

**Phase 14 — Verification (Full Toolchain)**
- [x] [P14-T1] Run complete fix_all toolchain: JSON → Shell → Python (Black → Ruff → Pyright → Pytest) → PowerShell
- [x] [P14-T2] Verify all branches pass: JSON ✓, Shell ✓, Python ✓, PowerShell ✓
- [x] [P14-T3] Document final metrics: test count (1156), coverage (92%), pass rate (100%)

**Phase 15 — Spec Update**
- [x] [P15-T1] Update spec.md status from Draft to Complete
- [x] [P15-T2] Fill out Scope & Non-Goals with completed work and deferred items
- [x] [P15-T3] Fill out Assumptions, Constraints, Dependencies with technical details
- [x] [P15-T4] Fill out Data/API/Config Impact with behavioral changes
- [x] [P15-T5] Fill out Test Strategy with actual test updates and coverage improvements
- [x] [P15-T6] Fill out Acceptance Criteria with all completed deliverables
- [x] [P15-T7] Fill out Risks & Mitigations with identified risks and solutions
- [x] [P15-T8] Fill out Rollout & Follow-up with deployment steps and optional follow-up tasks

**Phase 16 — Handoff & Follow-up**
- [ ] [P16-T1] Create PR with all changes including clear commit message following template
- [ ] [P16-T2] Add PR description summarizing: 8 new pre-authorized patterns, 5 non-authorized workarounds, fixes applied, test updates, metrics
- [ ] [P16-T3] Link PR to issue #78 and spec.md for traceability
- [ ] [P16-T4] Close issue #78 after PR merge with link to merged PR
- [ ] [P16-T5] Document optional follow-up tasks: ANN401 evaluation (3 instances), comment format audit (~60 instances), CI automation
