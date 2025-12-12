# Policy Audit Agent Instructions (Local Scope)

> NOTE: This file is scoped **only** to the policy audit templates directory.
> It does **not** change the global repo policy or day-to-day coding behavior.
> Apply these instructions only when the user explicitly requests a **Policy Audit**.

## When these instructions apply

Only follow this file when:

- The user explicitly asks for a “Policy Audit”, “policy compliance audit”, or similar, **or**
- The user is editing or creating a `PolicyAudit*.md` file in connection with a specific change, PR, or feature.

If the user is doing normal coding, test authoring, or refactoring without asking for an audit, **ignore this AGENTS.md** and follow the canonical instructions in:

- `.github/copilot-instructions.md`
- `.github/instructions/*.instructions.md`

## Inputs

When performing a policy audit:

1. Use the local process guide  
   - Read [README.md](./README.md) in this directory for the detailed audit process and usage instructions.

2. Use the policy audit template  
   - Use [PolicyAudit.template.md](./PolicyAudit.template.md) as the structural starting point for the audit document.
   - Create a working copy (for example: `docs/features/active/<feature>/PolicyAudit.md`) rather than editing the template in place.

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
   - Determine what is being audited (e.g., “fix-all.ps1 unit tests”, “Lexile scoring pipeline refactor”, “new Python module X”).
   - Identify the relevant code files and test files.

2. **Create an audit document**
   - Copy [PolicyAudit.template.md](./PolicyAudit.template.md) to the requested location (e.g., feature folder or PR docs).
   - Replace placeholders (`[Component Name]`, dates, paths, counts) as described in [README.md](./README.md).
   - Delete non-applicable sections (Python vs PowerShell, test vs non-test work) as described in the README.

3. **Evaluate policy compliance**
   - For each section of the template:
     - Run the actual toolchain commands required by the general and language-specific policies.
     - Inspect the code and tests for compliance with both:
       - General policies, and
       - Language-specific addenda.
     - Record **status** (`✅ PASS`, `⚠️ PARTIAL`, `❌ FAIL`, `N/A`) and **evidence** in the appropriate table rows.

4. **Document gaps and exceptions**
   - If any requirement is not fully met:
     - Document the gap, rationale, and proposed follow-up in the “Gaps and Exceptions” section.
     - List any removed/skipped tests and their justifications.
   - If exceptions to policy are explicitly approved, record them in the “Approved Exceptions” section.

5. **Summarize and recommend**
   - Complete the summary sections at the end of the template:
     - Policy-by-policy summary
     - Metrics summary
     - Recommendation (`Ready for merge`, `Needs revision`, or `Blocked`)
   - Ensure the final audit document is consistent with the underlying evidence and toolchain output.

## Constraints

- Do **not** modify the canonical `.instructions.md` policy documents as part of an audit.
- Do **not** treat this AGENTS file as a reason to change how normal code or tests are written; it is for **evaluation and documentation** only.
- If you detect any conflict between this file and the canonical policy documents, halt and ask the user for clarification instead of guessing.