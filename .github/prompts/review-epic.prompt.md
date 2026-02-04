---
agent: 'epic_review_agent'
description: 'Prompt for reviewing an epic documentation root folder with a delivery-first audit for a single-developer, multi-feature initiative. Derives all feature subfolders and current versions/plans from the epic root input, validates acceptance criteria against current code, reconciles plan checklists, generates epic-audit + feature-delivery-inventory + policy-audit (plus pre-execution orchestration-review when applicable); if remediation is needed, produces remediation inputs and automatically creates remediation-plan.md via atomic_planner.'
---

# Epic Review Prompt

## Goal

Act as an epic reviewer with a **delivery-first** posture. Review the epic **from its documentation root folder**, derive all constituent feature documentation (including version selection), validate acceptance criteria against the current codebase and tests, reconcile plan checklists, and produce audit-grade outputs suitable for merge readiness and execution planning.

An epic is defined here as:
> “A multi-feature initiative with interdependencies, used to organize scope and delivery for a single developer (or small team), without requiring formal program increments or business-case artifacts.”

Do not ask clarifying questions. Make best-effort inferences, document assumptions, and provide evidence-based findings. Missing formal business-case documentation should be **noted** but is **non-blocking** for merge readiness.

## Inputs (required)

- **Epic root folder (required):** `${input:EpicRootFolder}`
  - Example: `docs/features/active/2026-02-02-some-epic-47`

## What you must derive from the epic root

From the epic root folder contents:
- Read `initiative.md`, `issue.md`, `orchestration.md` (if present)
- Enumerate each feature subfolder
- For each feature:
  - If version folders exist (`v1`, `v2`, ...), select the highest `vN` as current
  - Otherwise treat the feature root as current
  - Select latest `plan.<timestamp>.md` using max ISO timestamp (`yyyy-MM-ddTHH-mm`)
  - Read `issue.md`, `spec.md`, `user-story.md`, and the selected plan (best-effort)
   - Extract acceptance criteria from `spec.md` and `user-story.md`
   - Verify each acceptance criterion against code/tests and capture evidence
   - Reconcile unchecked plan items against delivered code/tests (auto-check if delivered)

## Evidence provenance and freshness rules (blocking metrics)

Apply these requirements to any **numeric/metric claim** (coverage, pass rates, counts, etc.) used in audits or blocking decisions:

1. **Evidence provenance requirement (blocking claim):**
   - Every numeric claim MUST cite **source file + timestamp + command** (if applicable).
   - Example: “Coverage $= 45\%$ (source: `implementation-summary.md`, 2026-02-02, command not recorded).”

2. **Evidence classification (mandatory tag):**
   - **Verified**: toolchain output, `coverage.xml`, or a CI run URL.
   - **Reported**: doc-only claim without toolchain output.
   - **Stale**: doc-only claim **older than the review date** or not backed by toolchain output.

3. **Freshness rule:**
   - If the source is not a toolchain output OR is older than the review date, mark the claim **Stale**.
   - **Stale claims cannot be used as blocking evidence** without re-validation.

4. **Blocking claims must be Verified:**
   - Any blocking item based on metrics requires **Verified** status.
   - Otherwise, phrase it as **“needs verification”** rather than **“fails.”**

## Output format

Write all outputs to the epic root folder `${input:EpicRootFolder}`.

All filenames must include a timestamp in ISO-8601 format `yyyy-MM-ddTHH-mm` (e.g., `2026-02-02T15-30`).

### Required deliverables

1. `epic-audit.<timestamp>.md`
   - Single-developer work-planning clarity and decision-quality audit
   - Objective/scope mapping across features
   - Overall epic readiness: PASS / NEEDS REVISION / BLOCKED

2. `feature-delivery-inventory.<timestamp>.md`
   - Inventory of features + versions + selected “current” docs
   - Summary chart including requirements delivered (Met/Total)
   - Per-feature acceptance criteria list with status (Met / Partially Met / Not Met / Unknown)
   - Concrete evidence from code/tests for each criterion
   - Plan reconciliation notes (auto-checked items with evidence)

3. `policy-audit.<timestamp>.md`
   - Repo-wide policy audit using `docs/features/templates/policy_audit/policy-audit.yyyy-MM-ddTHH-mm.md`

### Conditional deliverables (only if remediation is required)

4. `remediation-inputs.<timestamp>.md`
   - Concrete, enumerated fix list with “done” criteria and exact doc locations

5. A **completed** `remediation-plan.<timestamp>.md` created by automatically invoking `atomic_planner`

### Conditional deliverables (pre-execution only)

6. `orchestration-review.<timestamp>.md`
   - Dependency, sequencing, and integration review of `orchestration.md`
   - Cross-check orchestration claims against feature plans
   - Only generate when execution has not begun (no delivered plan items and no code/test evidence yet)