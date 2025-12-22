# 2025-12-18-devcontainer-latency (Plan)

- Issue: #55
- Owner: 2025-12-18-devcontainer-latency
- Date: 2025-12-22
- Status: In progress (yellow badge)

**Phase 0 — Context & Inputs**
- [ ] [P0-T1] Read .github/copilot-instructions.md and .github/instructions/general-code-change.instructions.md to confirm baseline repo policies before any edits
- [ ] [P0-T2] Read .github/instructions/general-unit-test.instructions.md, .github/instructions/python-code-change.instructions.md, and .github/instructions/python-unit-test.instructions.md to capture language-specific and testing rules
- [ ] [P0-T3] Review feature source docs [2025-12-18-devcontainer-latency.md](../../active/2025-12-18-devcontainer-latency/2025-12-18-devcontainer-latency.md) and [spec.md](../../active/2025-12-18-devcontainer-latency/spec.md) for scope, RCA, and acceptance criteria
- [ ] [P0-T4] Record working branch and commit hash (chore/improve-container-performance-#55, `git rev-parse HEAD`) into this plan for traceability
- [ ] [P0-T5] Note required environment: VS Code devcontainer, Python 3.12.x via Poetry, commands run from /workspaces/lexile-corpus-tuner with Docker socket mount available

**Phase 1 — Scope & Baseline Capture**
- [ ] [P1-T1] Confirm acceptance criteria and non-goals from [spec.md](../../active/2025-12-18-devcontainer-latency/spec.md) are closed (extension policy, mount guidance, pytest timing target ~1–2s)
- [ ] [P1-T2] Document current devcontainer extension list (including spmeesseman.vscode-taskexplorer) and settings from [.devcontainer/devcontainer.json](../../../.devcontainer/devcontainer.json) to establish pre-change reference
- [ ] [P1-T3] Capture baseline timing for `poetry run pytest --collect-only` inside the current devcontainer (with existing extension set) and record the measured seconds + command output snippet in [2025-12-18-devcontainer-latency.md](../../active/2025-12-18-devcontainer-latency/2025-12-18-devcontainer-latency.md)

**Phase 2 — Design Decisions (Mounts & Extensions)**
- [ ] [P2-T1] Decide mount strategy (Option A: keep workspace on WSL2 filesystem; Option B: switch devcontainer to a named volume for code with a separate host bind for artifacts) and document the chosen option plus rationale in [spec.md](../../active/2025-12-18-devcontainer-latency/spec.md)
- [ ] [P2-T2] Define the allowed VS Code extension set for the devcontainer (remove spmeesseman.vscode-taskexplorer; keep required Python/PowerShell/Git/coverage tools) and record the final list in [spec.md](../../active/2025-12-18-devcontainer-latency/spec.md)
- [ ] [P2-T3] Specify devcontainer settings changes needed for the chosen mount strategy (workspaceMount/workspaceFolder values; additional `mounts` entries for artifacts) and add the exact target paths in [spec.md](../../active/2025-12-18-devcontainer-latency/spec.md)

**Phase 3 — Devcontainer Configuration Changes**
- [ ] [P3-T1] Update [.devcontainer/devcontainer.json](../../../.devcontainer/devcontainer.json) to apply the decided mount configuration (workspaceMount/workspaceFolder and artifact bind mounts) with explicit source/target paths
- [ ] [P3-T2] Update [.devcontainer/devcontainer.json](../../../.devcontainer/devcontainer.json) to enforce the finalized extension allow list (remove spmeesseman.vscode-taskexplorer and add/confirm required IDs) under `customizations.vscode.extensions`
- [ ] [P3-T3] Adjust any related scripts (e.g., [.devcontainer/post-create.sh](../../../.devcontainer/post-create.sh)) if needed to support the new mount layout (clone/sync into volume, ensure permissions) and document the change rationale inline

**Phase 4 — Documentation & Guidance**
- [ ] [P4-T1] Update [.devcontainer/README.md](../../../.devcontainer/README.md) and [.devcontainer/QUICKSTART.md](../../../.devcontainer/QUICKSTART.md) with step-by-step setup for the chosen mount strategy, including how to keep code on a fast path and how to mount large artifacts separately
- [ ] [P4-T2] Add troubleshooting notes to [.devcontainer/TROUBLESHOOTING.md](../../../.devcontainer/TROUBLESHOOTING.md) describing symptoms of slow pytest collection and the remediation steps (disable aggressive workspace scanners, ensure fast workspace storage)
- [ ] [P4-T3] Update [2025-12-18-devcontainer-latency.md](../../active/2025-12-18-devcontainer-latency/2025-12-18-devcontainer-latency.md) with the finalized mitigation steps and links to relevant config/doc updates

**Phase 5 — Validation & Benchmarks**
- [ ] [P5-T1] Rebuild the devcontainer from a clean state to ensure new mounts/extensions apply, noting the rebuild command and any prompts
- [ ] [P5-T2] Verify the installed extension set inside the rebuilt container excludes spmeesseman.vscode-taskexplorer and matches the allow list (capture `code --list-extensions` output snippet in [2025-12-18-devcontainer-latency.md](../../active/2025-12-18-devcontainer-latency/2025-12-18-devcontainer-latency.md))
- [ ] [P5-T3] Run `poetry run pytest --collect-only` in the rebuilt container, record wall-clock time (target ~1–2s) and command output snippet in [2025-12-18-devcontainer-latency.md](../../active/2025-12-18-devcontainer-latency/2025-12-18-devcontainer-latency.md); compare vs baseline
- [ ] [P5-T4] Run a representative short pytest subset (e.g., `poetry run pytest tests/src`) to confirm end-to-end improvement and capture timing in [2025-12-18-devcontainer-latency.md](../../active/2025-12-18-devcontainer-latency/2025-12-18-devcontainer-latency.md)

**Phase 6 — Toolchain Loop**
- [ ] [P6-T1] Run formatter: `poetry run black .`
- [ ] [P6-T2] Run linter: `poetry run ruff check`
- [ ] [P6-T3] Run type checker: `poetry run pyright`
- [ ] [P6-T4] Run tests: `poetry run pytest` (full suite) and ensure zero failures; rerun the loop from [P6-T1] if any step changes files or fails

**Phase 7 — PR & Handoff**
- [ ] [P7-T1] Update [spec.md](../../active/2025-12-18-devcontainer-latency/spec.md) and the GitHub issue #55 with final decisions, benchmark results, and residual risks
- [ ] [P7-T2] Prepare PR notes summarizing changes (devcontainer mounts, extension policy, docs updates), risks, and validation evidence; request review and link all artifacts for traceability

