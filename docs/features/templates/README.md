# Feature Folder Templates

Use these templates to keep planning consistent.

- Copy the `feature` folder to `docs/features/active/<feature-name>/`.
- Keep the folder name kebab-case and include the GitHub issue number if known (for example `speakerless-auto-detection-42`).
- Fill in `user-story.md`, `spec.md`, and `plan.md` before coding.
- When the feature ships, move the folder to `docs/features/archive/<YYYY-MM-DD>-<feature-name>/` to keep a clean working set.
- For refactors (no user-facing change), use `refactor/spec.md` and `refactor/plan.md` to capture intent, invariants, and execution steps.
- For epics/initiatives (tracking multiple child features/workstreams), use `epic/initiative.md` to record goals, decomposition, milestones, and validation across children.
