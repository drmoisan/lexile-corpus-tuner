# Policy Audit Template

## Purpose

This template provides a structured format for documenting compliance with repository policies during agent-driven development. It ensures that all code changes, whether features, bugfixes, or refactors, meet the standards defined in:

- `general-code-change.instructions.md`
- `python-code-change.instructions.md` OR `powershell-code-change.instructions.md`
- `general-unit-test.instructions.md`
- `python-unit-test.instructions.md` OR `powershell-unit-test.instructions.md`

## When to Use

Create a policy audit document:

- ✅ **Before submitting any PR** - Required by repository policy
- ✅ **After completing feature development** - Part of Feature Playbook
- ✅ **After implementing bugfixes** - Part of Bugfix Playbook
- ✅ **When adding or updating tests** - Especially for test coverage expansion
- ✅ **During refactoring work** - To confirm no regressions
- ✅ **When requested by reviewers** - As part of PR review process

## How to Use

### Step 1: Copy the Template

```powershell
# From repo root
Copy-Item docs/features/templates/policy_audit/PolicyAudit.template.md `
  -Destination [your-working-location]/PolicyAudit.md
```

### Step 2: Fill in Header Section

Replace placeholders in the header:

- `[Component Name]` - Name of what you're auditing (e.g., "fix-all.ps1 Unit Tests", "Lexile Analyzer Module")
- `[YYYY-MM-DD]` - Today's date
- `[path/to/test_file]` - Test file location (if applicable)
- `[path/to/code_file]` - Source code location
- `[N]` - Test counts (if applicable)

### Step 3: Delete Inapplicable Sections

**For Python work:**
- Keep Section 3A (Python Code Change Policy)
- Delete Section 3B (PowerShell Code Change Policy)
- Keep Section 4A (Python Unit Test Policy) if testing
- Delete Section 4B (PowerShell Unit Test Policy)

**For PowerShell work:**
- Delete Section 3A (Python Code Change Policy)
- Keep Section 3B (PowerShell Code Change Policy)
- Delete Section 4A (Python Unit Test Policy)
- Keep Section 4B (PowerShell Unit Test Policy) if testing

**For non-test work:**
- Delete Section 1 (General Unit Test Policy)
- Delete Section 4A and 4B (Language-Specific Unit Test Policies)
- Simplify Section 6 (Test Execution Metrics) to focus on code metrics

### Step 4: Complete Each Section

Work through each table systematically:

1. **Run the actual toolchain commands** and record results
2. **Inspect the code** to verify design principles
3. **Review test output** to gather metrics
4. **Document evidence** in the "Evidence" column
5. **Mark status** as ✅ PASS, ❌ FAIL, or N/A

### Step 5: Use Status Markers Consistently

- **✅ PASS** - Requirement fully met
- **❌ FAIL** - Requirement not met (document in Section 8: Gaps and Exceptions)
- **⚠️ PARTIAL** - Requirement partially met (document plan to complete)
- **N/A** - Requirement not applicable to this work

### Step 6: Complete Supporting Sections

- **Section 5** - Detail test coverage for each function/class
- **Section 6** - Report objective metrics from test runs
- **Section 7** - Show actual toolchain command output
- **Section 8** - Document any gaps, exceptions, or removed tests
- **Section 9** - List all commits and files changed
- **Section 10** - Provide verdict and recommendation

### Step 7: Delete Template Instructions

Before finalizing, delete:
- The template usage instruction block at the top
- Any `[placeholder]` text that wasn't filled in
- Any guidance comments in square brackets

### Optional: Running a Policy Audit with AI Assistants (Codex & Copilot)

You can use Codex and GitHub Copilot to help fill out this template, but they should **follow this document and the canonical policy instructions**; they do not replace your judgment.

#### Using Codex (web) for a full audit

When working in Codex with this repo attached, you can kick off a full audit with a prompt like:

> **Policy Audit – [short scope name]**  
>  
> Perform a **Policy Audit** for the following scope:  
> - Scope: `[short description – e.g., “new PoshQC entrypoint tests for fix-all.ps1 on branch feature/PoshQc-#21”]`  
> - Branch: `[branch name]`  
>  
> Use the **local audit instructions** in:  
> - `docs/features/templates/policy_audit/AGENTS.md`  
> - `docs/features/templates/policy_audit/README.md`  
> - `docs/features/templates/policy_audit/PolicyAudit.template.md`  
>  
> Follow the README’s process and the AGENTS instructions to:  
> 1. Create a new `PolicyAudit.md` for this scope (you may propose a pathname).  
> 2. Evaluate compliance against all relevant policies:  
>    - `.github/instructions/general-code-change.instructions.md`  
>    - `.github/instructions/general-unit-test.instructions.md`  
>    - `.github/instructions/python-code-change.instructions.md` / `python-unit-test.instructions.md`  
>    - `.github/instructions/powershell-code-change.instructions.md` / `powershell-unit-test.instructions.md`  
> 3. Fill out the template with concrete findings, status for each policy item, gaps/exceptions, and a final recommendation.  
>  
> Show the completed `PolicyAudit.md` as your final answer.

Edit the **Scope** and **Branch** lines, then paste this into Codex. Always review the generated audit for accuracy before treating it as authoritative.

#### Combined workflow: generate tests and audit in a single Codex task

Sometimes you will ask Codex to both **write unit tests** for a file and then **audit** the original code plus those new tests. In that case:

1. First describe the test work as usual, for example:  
   > Write Pester tests for `scripts/dev-tools/fix-all.ps1` that comply with the PowerShell unit test policy…

2. Then append a Policy Audit request, for example:  
   > In addition to generating the unit tests described above, perform a **formal Policy Audit** for `fix-all.ps1` and the new tests you create.  
   >  
   > Use:  
   > - `docs/features/templates/policy_audit/AGENTS.md`  
   > - `docs/features/templates/policy_audit/README.md`  
   > - `docs/features/templates/policy_audit/PolicyAudit.template.md`  
   >  
   > Create a new audit document named `YYYY-MM-DD-fix-all.PolicyAudit.md` (for today’s date) under an appropriate feature folder (for example `docs/features/active/PoshQc/`).  
   > Evaluate compliance against:  
   > - `.github/instructions/general-code-change.instructions.md`  
   > - `.github/instructions/general-unit-test.instructions.md`  
   > - `.github/instructions/powershell-code-change.instructions.md`  
   > - `.github/instructions/powershell-unit-test.instructions.md`  
   > and fill out the template with status (PASS / PARTIAL / FAIL / N/A), evidence, gaps/exceptions, and a final recommendation.

This makes it explicit that you can treat “write tests” + “run audit” as one integrated Codex task, while still reusing the same policy-audit process and naming convention described above.

#### Using GitHub Copilot Chat inside VS Code

When you have a working copy of `PolicyAudit.template.md` open in VS Code (for example, `docs/features/active/<feature>/PolicyAudit.md`), you can ask Copilot Chat to help fill it in:

> I am performing a **Policy Audit** for:  
> - `[scope – e.g., “PowerShell PoshQC changes on feature/PoshQc-#21”]`  
>  
> Use the **policy audit process** defined in:  
> - `docs/features/templates/policy_audit/README.md`  
> - `docs/features/templates/policy_audit/AGENTS.md`  
>  
> And evaluate compliance against the canonical policies in:  
> - `.github/instructions/general-code-change.instructions.md`  
> - `.github/instructions/general-unit-test.instructions.md`  
> - `.github/instructions/powershell-code-change.instructions.md`  
> - `.github/instructions/powershell-unit-test.instructions.md`  
>  
> Fill in this `PolicyAudit.md` (the selected text) according to the template:  
> - Mark each requirement as PASS / PARTIAL / FAIL / N/A with evidence.  
> - Capture any gaps, exceptions, and the final recommendation.

Best practice is to:

- Select the body of the template before invoking Copilot Chat so it knows what to fill in.
- Inspect every section of the generated audit against actual code, tests, and tool output.
- Treat the AI output as a **draft**; final responsibility for correctness remains with the reviewer.

## Examples

See the original completed audit that this template was derived from:
- **Location:** PR #26 audit document
- **Component:** fix-all.ps1 Unit Tests
- **Type:** PowerShell unit testing
- **Result:** Fully compliant

## Common Patterns

### For Unit Test Audits

Focus on:
- Section 1 (General Unit Test Policy) - Core testing principles
- Section 4 (Language-Specific Unit Test Policy) - Framework requirements
- Section 5 (Test Coverage Detail) - Detailed coverage analysis
- Section 6 (Test Execution Metrics) - Timing and pass rates

### For Feature Development

Focus on:
- Section 2 (General Code Change Policy) - Design and structure
- Section 3 (Language-Specific Code Change Policy) - Language standards
- Section 7 (Code Quality Checks) - Toolchain results
- Section 9 (Summary of Changes) - What was built

### For Bugfixes

Use the Bugfix Playbook alongside this template:
1. Create failing regression test (Section 5)
2. Implement minimal fix (Section 2, 3)
3. Run toolchain (Section 7)
4. Verify fix (Section 6)
5. Document (Section 9)

## Integration with Playbooks

This template is referenced by:

- **Bugfix Playbook** (`docs/engineering/Bugfix Playbook.md`)
  - Use during Step 3: "Verify locally before review"
  - Required before PR submission

- **Feature Playbook** (`docs/engineering/Feature Playbook.md`)
  - Use during "Validation" phase
  - Required before marking feature complete

## Tips for Success

1. **Fill as you go** - Don't wait until the end to complete the audit
2. **Run commands for real** - Copy actual output, don't guess
3. **Be honest about gaps** - Better to document issues than hide them
4. **Link to evidence** - Reference commit hashes, line numbers, command output
5. **Update during review** - If issues are found, update the audit and re-run checks

## Questions?

If you're unsure about:
- **What counts as compliance** - Review the specific policy documents
- **How to interpret a requirement** - Ask for clarification in PR comments
- **Whether to audit non-code work** - Audit anything that changes repo behavior
- **How detailed to be** - More detail is better; provide concrete evidence

## Maintenance

When policy documents are updated:
1. Update this template to match new requirements
2. Add new sections for new policy areas
3. Update examples to reflect current best practices
4. Notify active developers of template changes
