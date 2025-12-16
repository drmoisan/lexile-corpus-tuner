# tooling-bug-fix-process (Issue #22)

- Date captured: 2025-12-08
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/tooling-bug-fix-process/ (Issue #22)

- Issue: #22
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/22
- Last Updated: 2025-12-09
## Problem / Why

The feature development workflow has been fully automated with:
- `Dev: 1 New Potential Entry` task to create potential feature files
- `Dev: 2 Promote Potential to GitHub Issue` task to promote features to issues
- `Dev: 3 Create Active Folder` task to create active feature documentation folders
- Templates for user-story.md, spec.md, and plan.md in `docs/features/templates/feature/`

However, the bug fix workflow only has partial automation:
- `Dev: 1A New Potential Bug` exists for creating potential bug files
- Bug promotion uses the same `Dev: 2` task but lacks proper bug-specific handling
- No equivalent "Create Active Bug Folder" task exists
- No templates exist in `docs/features/templates/bug/` for active bug documentation (only `potential_bug.md`)

This gap means bug fixes lack the same level of structured documentation and tracking as features, leading to inconsistent processes and lost context during bug resolution.

## Proposed Behavior

### 1. Create Bug-Specific Documentation Templates

Add to `docs/features/templates/bug/`:
- **diagnosis.md** - Root cause analysis, reproduction steps, environment details
- **fix-plan.md** - Implementation plan, phases, test strategy
- **validation.md** - Verification steps, regression tests, acceptance criteria

### 2. Enhance Potential Bug Promotion

Update `scripts/dev-tools/potential-to-issue.ps1` to:
- Properly handle `bug` promotion type (currently fails)
- Use GitHub's `bug-report.md` template for bugs vs `feature-request.md` for features
- Extract bug-specific fields (environment, reproduction steps, impact/severity)
- Support both interactive and non-interactive modes for bug promotion

### 3. Create "Dev: 3A Create Active Bug Folder" Task

New script: `scripts/dev-tools/new-active-bug-folder.ps1`
- Accepts `-BugName`, `-IssueNumber` parameters
- Copies templates from `docs/features/templates/bug/` to `docs/features/active/<bug-name>/`
- Populates issue number, date, author placeholders
- Opens all three files in VS Code for immediate editing
- Creates folder structure: `docs/features/active/<bug-name>/{diagnosis.md, fix-plan.md, validation.md}`

### 4. Update VS Code Tasks

Add new task to `.vscode/tasks.json`:
```json
{
  "label": "Dev: 3A Create Active Bug Folder",
  "type": "shell",
  "command": "pwsh",
  "args": [
    "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", "${workspaceFolder}/scripts/dev-tools/new-active-bug-folder.ps1",
    "-BugName", "${input:ActiveBugName}",
    "-IssueNumber", "${input:ActiveBugIssueNumber}"
  ]
}
```

Add corresponding inputs for ActiveBugName and ActiveBugIssueNumber.

### 5. Documentation Updates

Create `docs/engineering/Bugfix Playbook.md` (parallel to Feature Playbook) with:
- Step-by-step workflow: potential → issue → active folder → diagnosis → fix → validation → close
- VS Code task usage instructions
- Links to templates and examples
- Integration with existing link-feature-docs and link-parent-child scripts

## Acceptance Criteria (early draft)

- [ ] Three bug documentation templates exist in `docs/features/templates/bug/`: diagnosis.md, fix-plan.md, validation.md
- [ ] `potential-to-issue.ps1` successfully promotes bugs using `bug-report.md` template without errors
- [ ] `new-active-bug-folder.ps1` script creates active bug folders with all three templates populated
- [ ] VS Code task "Dev: 3A Create Active Bug Folder" executes successfully with proper input prompts
- [ ] Bugfix Playbook documentation provides clear step-by-step workflow guidance
- [ ] All scripts follow PowerShell best practices and pass PSScriptAnalyzer (per QC: PowerShell: analyze task)
- [ ] End-to-end workflow tested: create potential → promote to issue → create active folder → document fix

## Constraints & Risks

### Constraints
- Must maintain consistency with existing feature tooling patterns (naming, structure, workflow)
- Scripts must work on Windows with PowerShell 7+ (repo standard)
- Must integrate with existing VS Code tasks without breaking current functionality
- Templates must align with general-code-change.instructions.md and related policies
- GitHub CLI (`gh`) must be available and authenticated for issue creation

### Risks
- **Scope creep**: Keep initial implementation simple; defer advanced features (auto-linking, metrics) to future iterations
- **Template quality**: Bug documentation templates need careful design to be useful without being overly prescriptive
- **Adoption**: Developers may not use the tooling if it adds friction; must be as streamlined as feature tooling
- **Maintenance burden**: Adding more templates and scripts increases maintenance surface area

### Mitigation
- Start with minimal viable templates that can be refined based on actual usage
- Leverage existing scripts (link-feature-docs, link-parent-child) without modification
- Include clear examples in Bugfix Playbook to demonstrate value
- Use Pester tests for PowerShell scripts to ensure reliability

## Test Conditions to Consider

### Unit Coverage (PowerShell scripts)
- [ ] Pester tests for `new-active-bug-folder.ps1`:
  - Valid bug name creates folder structure correctly
  - Invalid bug name (non-kebab-case) fails with clear error
  - Placeholders replaced correctly (issue number, date, author)
  - Files created with UTF-8 encoding
- [ ] Enhanced tests for `potential-to-issue.ps1`:
  - Bug promotion type uses correct GitHub template
  - Feature promotion type continues to work (regression)
  - Missing `gh` CLI fails gracefully with clear error
  - Invalid potential file path handled properly

### Integration Scenarios
- [ ] End-to-end workflow:
  1. Run "Dev: 1A New Potential Bug" → creates potential file
  2. Edit potential file with bug details
  3. Run "Dev: 2 Promote Potential to GitHub Issue" with bug type → creates GitHub issue
  4. Run "Dev: 3A Create Active Bug Folder" with issue number → creates active folder
  5. Edit diagnosis.md, fix-plan.md, validation.md
  6. Run "Dev: 5 Link Feature Docs to GitHub" → links docs to issue (existing script)
- [ ] Parallel workflows don't interfere:
  - Creating bug folder doesn't affect feature folder creation
  - Bug promotion doesn't break feature promotion

### CLI/Task Examples
- [ ] All VS Code tasks appear in task list with proper labels
- [ ] Input prompts display correctly with helpful descriptions
- [ ] Task output shows clear success/failure messages
- [ ] Error handling displays actionable messages (not raw stack traces)

### Edge Cases
- [ ] Bug name conflicts with existing feature name
- [ ] Issue number already linked to different active folder
- [ ] Template files missing or corrupted
- [ ] VS Code not installed (script should still succeed, just skip auto-open)
- [ ] Git not configured (fallback to $env:USERNAME for author)

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/tooling-bug-fix-process/` folder from the template (adapt feature templates or create bug-specific initiative template)


