# Feature Playbook

A structured, traceable workflow for taking a feature from idea to archive in this repo.

Key paths:
- Backlog: `docs/features/backlog.md`
- Potential ideas: `docs/features/potential/` (promoted items move to `docs/features/potential/promoted/`)
- Active features: `docs/features/active/<feature>/` (templates live in `docs/features/templates/feature/`)
- Archive: `docs/features/archive/<YYYY-MM-DD>-<feature>/`

Key scripts (PowerShell unless noted):
- `new-potential-entry.ps1` – create a dated potential file and open backlog.
- `scripts/dev_tools/potential_to_issue.py` – promote a potential to a GitHub issue, mark as promoted, move to promoted folder (Python).
- `scripts/dev_tools/new_active_feature_folder.py` – create/seed `docs/features/active/<feature>-<issue>/` from templates and matching potential/promoted doc; auto-fill issue/owner/last updated and move the promoted potential into the active folder as `issue.md` (Python).
- `link-feature-docs.ps1` – add/update “Feature Docs” links in a GitHub issue (user-story/spec/plan).

VS Code tasks wrap these scripts: see `.vscode/tasks.json` (e.g., “Feature: New Potential Entry”, “GitHub: Feature Issue from Potential”, “Feature: Create Active Folder”, “GitHub: Link Feature Docs”).

---

## 0. Idea capture
- Add a short entry to `docs/features/backlog.md`.
- If you want a paper trail, create a potential file: `docs/features/potential/<YYYY-MM-DD>-<shortname>.md` (use `scripts/new-potential-entry.ps1`).
- Keep it light; no formal spec yet.

## 1. Promote to GitHub issue
- Use `poetry run python -m scripts.dev_tools.potential_to_issue --potential-path <file>` (or VS Code task “GitHub: Feature Issue from Potential”).
- This creates the issue via `gh`, records the issue number in the potential file, and moves it to `docs/features/potential/promoted/`.
- Issue body should include: Problem/Why, Proposed Behavior, Acceptance Criteria, Constraints/Risks, Test Conditions, and a link to the potential/promoted doc.

-## 2. Create the active feature folder
- Run `poetry run python -m scripts.dev_tools.new_active_feature_folder --feature-name <name> --type feature --issue-number <issue|auto>` (or VS Code task “Feature: Create Active Folder”).
- Creates `docs/features/active/<name>-<issue>/` with:
  - `user-story.md`
  - `spec.md`
  - `plan.md`
- Seeds Problem/Behavior/AC/Test Conditions from the promoted doc, fills headers (Issue/Owner/Last Updated), and moves the promoted potential file into the active folder as `issue.md`.

## 3. Branch
- Name: `feature/<feature-name>-#<issue>`
  - Example: `feature/speakerless-auto-detection-#42`
- All commits/PRs should reference the issue number.

## 4. Docs and tests first
- Update/add tests before or alongside code.
- Update `user-story.md`, `spec.md`, `plan.md` in the active folder.
- Test locations: `tests/transform/`, `tests/document/`, `tests/integration/`, `tests/features/<feature-name>/` (as needed).

## 5. Incremental development
- Each commit: Black/Ruff/Pyright clean; tests updated/passing.
- Prefer small, scoped changes; use feature flags if needed.
- If a non-trivial bug is found, open an issue (don’t hide it in the feature).

## 6. Implementation guidelines
- Reuse existing ETL entry points; avoid duplicating pipeline logic.
- Keep changes modular; log/telemetry only where helpful.
- Document new CLI flags in `spec.md` and `--help`; add tests for flag behavior.

## 7. Local verification
- Run `Run All Checks` task or `poetry run python -m scripts.dev_tools.fix_all` (PowerShell wrapper available at `scripts/dev-tools/fix-all.ps1`) for JSON/shell QC → Black/Ruff/Pyright/Pytest → PoshQC.
- Run relevant integration/end-to-end checks and sample transcripts.
- Confirm performance impact if applicable.

## 8. Pull request
- Title: `feat: add <feature-name> (fixes #<issue>)`
- Body: summary, links to `docs/features/active/<feature-name>/`, test summary, behavior/flags, breaking changes/migrations, samples or screenshots if relevant.
- Keep review discussion in the PR.

## 9. Review cycle
- Validate behavior vs spec and acceptance criteria.
- Ensure tests cover the change; push back on missing tests.
- Open a new issue for deeper design gaps; don’t expand PR scope.

## 10. Merge and archive
- After merge to `development`, move `docs/features/active/<feature>/` to `docs/features/archive/<YYYY-MM-DD>-<feature>/`.
- Keep user-story/spec/plan as the historical record.
- Update backlog/feature status as needed.

## Checklists

**Before PR**
- [ ] Issue created and linked
- [ ] Active folder created/seeded (user-story/spec/plan)
- [ ] Tests added/updated
- [ ] Docs updated (spec/plan/user-story; CLI help if applicable)
- [ ] Branch created with issue number
- [ ] Black/Ruff/Pyright clean
- [ ] Tests passing (unit/integration as relevant)

**In PR**
- [ ] Summary and links to active folder
- [ ] Test summary
- [ ] Behavior/flags documented
- [ ] Migration/breaking changes noted

**After merge**
- [ ] Archive active folder to `docs/features/archive/<date>-<feature>/`
- [ ] Issue closed (auto-close preferred)
- [ ] Backlog/status updated if needed

## Anti-patterns
- Starting code without an issue or active folder.
- Skipping tests or type checks for new behavior.
- Mixing unrelated bugfixes with feature work.
- Letting agents bypass tests or Pyright.
- Keeping design/decision notes outside the issue/PR/feature docs.
- Failing to archive after merge.

## Agentic Tooling

| Phase                                      | Phase Name (Corrected)                                  | Responsible Agent                       | Notes / Constraints                                                                                                                                                              |
| ------------------------------------------ | ------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1**                                      | **User Story Creation (Problem Framing)**               | *(Gap / Human-Owned)*                   | No agent currently authors the initial user story from raw intent. This is the starting artifact.                                                                                |
| **2**                                      | **Story Elaboration / Discovery**                       | **Task Researcher**                     | Research-only, evidence-based discovery. Surfaces constraints, assumptions, feasibility, and alternatives. No commitments, no code, no spec writing.                             |
| **3**                                      | **Requirements & Constraint Discovery**                 | **Task Researcher**                     | Functional + non-functional constraints emerge here but are not yet finalized or committed. Still exploratory.                                                                   |
| **4**                                      | **Domain & Impact Analysis**                            | **Task Researcher**                     | Identifies affected subsystems, data, APIs, compatibility risks. Output is research evidence, not decisions.                                                                     |
| **5**                                      | **Solution Exploration (Option Space)**                 | **Task Researcher**                     | Multiple approaches evaluated; tradeoffs documented; one approach recommended; alternatives deleted.                                                                             |
| **6**                                      | **Feature Specification Authoring (Design Commitment)** | **PRD / Feature Docs Completion Agent** | Completes `user-story.md` and `spec.md`. This is where intent becomes a **commitment**: behavior, APIs, data, constraints, risks.                                                |
| **7**                                      | **Acceptance Criteria & DoD Finalization**              | **PRD / Feature Docs Completion Agent** | Acceptance criteria become stable, testable, and exhaustive. Feature is now “commit-ready.”                                                                                      |
| **8**                                      | **Atomic Development Plan Authoring (Plan-of-Record)**  | **Atomic Planning Agent**               | **Critical distinction**: this is the *true* atomic plan. It cannot exist until the spec is complete. Tasks are binary, ordered, verifiable.                                     |
| **9**                                      | **Deterministic Plan Execution**                        | **Atomic Execution Agent**              | Executes the atomic plan verbatim. No replanning, no scope change, strict policy and acceptance-criteria verification.                                                           |
| **10 (Optional but Strongly Recommended)** | **Post-Execution Feature & Policy Audit**               | **Feature Review Agent**                | Audits the completed feature against repo policy *and* against the committed documentation (user story, spec, acceptance criteria). Generates remediation inputs if gaps exist.  |
