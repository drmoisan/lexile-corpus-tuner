# Policy Audit Agent Instructions (Local Scope)

> NOTE: This file is scoped **only** to the policy audit templates directory.
> It does **not** change the global repo policy or day-to-day coding behavior.
> Apply these instructions only when the user explicitly requests a **Policy Audit**.

## When these instructions apply

Only follow this file when:

- The user explicitly asks for a "Policy Audit", "policy compliance audit", or similar, **or**
- The user is editing or creating a `PolicyAudit*.md` file in connection with a specific change, PR, or feature.

If the user is doing normal coding, test authoring, or refactoring without asking for an audit, **ignore this AGENTS.md** and follow the canonical instructions in:

- `.github/copilot-instructions.md`
- `.github/instructions/*.instructions.md`

## Inputs

When performing a policy audit:

1. Use the local process guide  
   - Read [README.md](./README.md) in this directory for the detailed audit process and usage instructions.

2. Use the policy audit template  
   - Use [policy-audit.yyyy-MM-ddTHH-mm.md](./policy-audit.yyyy-MM-ddTHH-mm.md) as the structural starting point for the audit document.
   - Create a working copy (for example: `docs/features/active/<feature>/policy-audit.2026-01-08T14-30.md`) rather than editing the template in place.

3. Use the canonical policy documents for evaluation (read-only)  
   - General code change policy: `.github/instructions/general-code-change.instructions.md`
   - Language-specific code change policy:
     - Python: `.github/instructions/python-code-change.instructions.md`
     - PowerShell: `.github/instructions/powershell-code-change.instructions.md`
   - General unit test policy: `.github/instructions/general-unit-test.instructions.md`
   - Language-specific unit test policy:
     - Python: `.github/instructions/python-unit-test.instructions.md`
     - PowerShell: `.github/instructions/powershell-unit-test.instructions.md`

## Behavior: How to run a Policy Audit

When asked to perform a Policy Audit for a component, branch, or PR:

1. **Identify scope**
   - Determine what is being audited (e.g., "fix-all.ps1 unit tests", "Lexile scoring pipeline refactor", "new Python module X").
   - Identify the relevant code files and test files.
   - **Document baseline coverage BEFORE making changes** (if development has not yet started, or shelve changes temporarily to establish baseline).

2. **Create an audit document**
   - Generate a timestamp in ISO-8601 format `yyyy-MM-ddTHH-mm` (e.g., "2026-01-08T14-30" for Jan 8, 2026 at 2:30 PM).
   - Copy [policy-audit.yyyy-MM-ddTHH-mm.md](./policy-audit.yyyy-MM-ddTHH-mm.md) to the requested location with timestamped filename: `policy-audit.<timestamp>.md` (e.g., feature folder or PR docs).
   - Replace placeholders (`[Component Name]`, dates, paths, counts) as described in [README.md](./README.md).
   - **For multi-language changes:**
     - Fill in the Coverage Metrics by Language table with one row per language.
     - Complete all applicable language sections (3A, 3B, 3C, 3D for code; 4A, 4B for tests).
     - Delete sections for languages NOT involved in this change.
   - **Fill in baseline coverage metrics** from pre-development measurement (per language that has coverage).

3. **Evaluate policy compliance**
   - For each section of the template:
     - Run the actual toolchain commands required by the general and language-specific policies.
     - Inspect the code and tests for compliance with both:
       - General policies, and
       - Language-specific addenda.
     - Record **status** (`✅ PASS`, `⚠️ PARTIAL`, `❌ FAIL`, `N/A`) and **evidence** in the appropriate table rows.
   - **For multi-language changes:**
     - Run each language's toolchain separately and document results in its section.
     - **Python:** Black → Ruff → Pyright → Pytest (with coverage)
     - **PowerShell:** Invoke-PoshQCFormat → Invoke-PoshQCAnalyze → Invoke-PoshQCTest (with coverage)
     - **Bash:** shfmt → shellcheck → bats (coverage N/A)
     - **JSON:** format_json → validate_json (coverage N/A)
   - **For coverage metrics (Python and PowerShell only):**
     - Compare post-change coverage to baseline to verify no regression (per language).
     - Isolate and measure coverage of new/modified code only (must be ≥90% per language).
     - Use concrete examples showing calculations: "Baseline: 85.2% → Post-change: 87.1% (+1.9%) ✅"
   - **For scenario testing:**
     - Use deterministic, input→output→assertion format for all examples.
     - **Positive flows:** Show concrete valid inputs and expected outputs.
     - **Negative flows:** Show concrete invalid inputs, expected exceptions, and error messages.
     - **Edge cases:** Show concrete boundary conditions (empty, max length, Unicode, whitespace).
     - **Error handling:** Show concrete error conditions and expected exception types.

4. **Verify temporary artifacts cleanup**
   - **Identify all scripts created during development** (check commit history, staged files, dev-tools folders).
   - **Categorize each script:**
     - Temporary/one-time: Delete before finalizing audit.
     - Ongoing tooling: Must have full test coverage and pass all repo policies.
   - **Document disposition** in "Temporary artifacts cleanup" section of audit.

5. **Document gaps and exceptions**
   - If any requirement is not fully met:
     - Document the gap, rationale, and proposed follow-up in the "Gaps and Exceptions" section.
     - List any removed/skipped tests and their justifications.
   - If exceptions to policy are explicitly approved, record them in the "Approved Exceptions" section.

6. **Summarize and recommend**
   - Complete the summary sections at the end of the template:
     - Policy-by-policy summary
     - Metrics summary (with deterministic baseline/post-change/new code coverage)
     - Recommendation (`Ready for merge`, `Needs revision`, or `Blocked`)
   - Ensure the final audit document is consistent with the underlying evidence and toolchain output.

## Constraints

- Do **not** modify the canonical `.instructions.md` policy documents as part of an audit.
- Do **not** treat this AGENTS file as a reason to change how normal code or tests are written; it is for **evaluation and documentation** only.
- If you detect any conflict between this file and the canonical policy documents, halt and ask the user for clarification instead of guessing.
