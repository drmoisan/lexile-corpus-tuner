# new-folder-squashes-spec-template (Issue #102)

- Date captured: 2026-01-21
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/new-folder-squashes-spec-template/ (Issue #102)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #102
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/102
- Last Updated: 2026-01-21
## Summary

Creating an active bug folder via `Dev: 3 Create Active Folder` overwrites the bug `spec.md` template’s intent by pre-populating the `## Proposed Fix` (and `## Test Strategy`) sections using the early “Proposed Fix / Validation Ideas” content from the potential bug file that becomes `issue.md`.

This prematurely locks the design direction and loses the intended separation where early hypotheses belong in `issue.md`, and the final `spec.md` is filled only after research is complete.

## Environment

- OS/version: Linux (devcontainer: Debian GNU/Linux 12 (bookworm))
- Python version: Poetry-managed; repo CI targets Python 3.10–3.13 (exact runtime version TBD)
- Command/flags used:
	- VS Code task: `Dev: 3 Create Active Folder`
	- Under the hood:
		- `poetry run python -m scripts.dev_tools.new_active_feature_folder --feature-name <bug-name> --type bug --issue-number <issue|auto>`
- Data source or fixture:
	- Potential bug markdown file under `docs/features/potential/` containing early “Proposed Fix / Validation Ideas” content
	- Bug `spec.md` template under `docs/features/templates/bug/spec.md`

## Steps to Reproduce

1. Create a potential bug file under `docs/features/potential/` that includes an early hypothesis in `## Proposed Fix / Validation Ideas`.
2. Run the VS Code task `Dev: 3 Create Active Folder` with:
	- `--type bug`
	- `--feature-name <bug-name>`
	- `--issue-number <issue|auto>`
3. Open the generated active folder under `docs/features/active/<slug>/`.
4. Inspect `docs/features/active/<slug>/spec.md`.

## Expected Behavior

The created `docs/features/active/<slug>/spec.md` should preserve the robust bug `spec.md` template structure and leave `## Proposed Fix` (and other design-commitment sections) unpopulated.

Early proposed fixes/hypotheses should remain only in the moved `issue.md` (the promoted potential file), and research should drive what ultimately gets written into `spec.md`.

## Actual Behavior

`docs/features/active/<slug>/spec.md` has its `## Proposed Fix` (and `## Test Strategy`) sections pre-populated from the potential bug file’s `## Proposed Fix / Validation Ideas` section.

This effectively “squashes” (overwrites the intent of) the `spec.md` template by transferring early thinking directly into the spec, narrowing the later research space.

## Logs / Screenshots

- [ ] Attached minimal logs or screenshot
- Snippet:

	The active-folder creation script currently injects the potential bug’s “Proposed Fix / Validation Ideas” into the bug spec:

	- `scripts/dev_tools/new_active_feature_folder.py` (bug doc seeding flow)
		- When `bug_validation` content exists, it is applied to both `Proposed Fix` and `Test Strategy` in `spec.md`.

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

The seeding logic for bug folders copies sections out of the promoted potential bug file and applies them directly into the bug `spec.md`.

Specifically, the bug flow treats the potential bug’s `## Proposed Fix / Validation Ideas` as `bug_validation` and then writes that body into `spec.md` under both:
- `## Proposed Fix`
- `## Test Strategy`

Relevant implementation entry point:
- `scripts/dev_tools/new_active_feature_folder.py` (called by the VS Code task `Dev: 3 Create Active Folder`)

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
	- Update/add tests in `tests/scripts/dev_tools/test_new_active_feature_folder.py` to assert that, for bug folders, `spec.md` does **not** receive the “Proposed Fix / Validation Ideas” content from the potential file.
	- Keep the existing behavior that moves the potential file into the active folder as `issue.md` (so early hypotheses are preserved there).
- [ ] Integration scenario to retest
	- Create a potential bug file with non-trivial “Proposed Fix / Validation Ideas”, run `Dev: 3 Create Active Folder` with `--type bug`, and verify the resulting `spec.md` retains only template structure (no pre-populated Proposed Fix/Test Strategy content).
- [ ] Manual verification notes
	- Confirm the resulting `issue.md` still contains the early proposed fix content verbatim.
	- Confirm `spec.md` still contains the required headings but with empty/placeholder bodies in `Proposed Fix`/`Test Strategy`.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch