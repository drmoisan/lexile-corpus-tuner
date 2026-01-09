---
agent: 'staged_code_review_agent'
description: 'Prompt for reviewing currently staged changes before commit. Generates PolicyAudit.md + CodeReview.md in the active feature folder; if remediation is needed, produces remediation inputs and delegates remediation-plan.md creation to atomic_planner.'
---

# Staged Code Review Prompt

## Goal

Act as a pre-commit staged-change reviewer and policy auditor. Review ONLY what is currently staged (`git diff --staged`) and generate audit-grade documentation suitable for a PR.

Do not ask clarifying questions. Make best-effort inferences, document assumptions, and produce the required artifacts.

## Output Format

Write outputs to the **active feature folder** determined by the agent. If no active feature folder can be inferred, use the agent’s fallback (e.g., `docs/features/active/_staged-review/`).

### Required deliverables

All filenames must include a timestamp in ISO-8601 format `yyyy-MM-ddTHH-mm` (e.g., `policy-audit.2026-01-08T14-30.md`).

1. `policy-audit.<timestamp>.md`
   - A completed Policy Audit document based on the repo's policy-audit template and the policy_audit template guidance.
   - Must include PASS/PARTIAL/FAIL (or equivalent) statuses with evidence (commands run, outputs, file references).

2. `code-review.<timestamp>.md`
   - A staged-diff-based review emphasizing strongly typed Python best practices and repo compliance.
   - Must include a clear go/no-go recommendation for committing.

### Conditional deliverables (only if remediation is required)

3. `remediation-inputs.<timestamp>.md`
   - Concrete, enumerated fix list with acceptance criteria and verification commands.

4. An `atomic_planner` prompt (copy/paste-ready) that instructs `atomic_planner` to WRITE:
   - `remediation-plan.<timestamp>.md` in the same active feature folder as the audit documents.

## Context Template

- **Intent / Change summary (optional):** [One sentence on what this commit is meant to accomplish]
- **Staged scope note (optional):** [Any known constraints or areas to pay extra attention to]
- **Verification hints (optional):** [Repo-specific tasks/commands to run if multiple exist]
- **Risk tolerance (optional):** [e.g., “fail if any PARTIAL”, or “allow PARTIAL with remediation plan”]
