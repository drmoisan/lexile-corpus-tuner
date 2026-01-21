# automated-planner (Issue #97)

- Date captured: 2026-01-20
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/automated-planner/ (Issue #97)

- Issue: #97
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/97
- Last Updated: 2026-01-21
## Problem / Why

Today, taking an idea from “potential feature” to an actionable, executable implementation plan is manual and inconsistent. The repo already has strong building blocks (notably the `atomic_executor` package and prompt-driven agents), but there is no scripted, repeatable workflow that:

- Promotes a filled-out potential doc into a GitHub issue in a consistent way
- Creates the correct `docs/features/active/<feature>/` folder scaffold
- Runs a standardized research + PRD/spec completion loop
- Produces an atomic plan and validates that plan for execution readiness (without executing it)

This feature (“automated planner”) aims to standardize that end-to-end flow so that planning is faster, higher quality, and less dependent on human memory of which scripts/prompts to run.

## Proposed Behavior

Provide a scripted, prompt-resolving pipeline that orchestrates GitHub Copilot CLI agents and existing repo tooling to convert a “potential” entry into a validated atomic plan.

Key design requirements:

- Support four promotion types: `feature`, `epic`, `bug`, `refactor`.
- Implement the `bug` type first, but design the pipeline so other types are handled via small “type modifiers” (prompt/templating variations), not completely branched codepaths.
- Model the orchestration style after `atomic_executor` (plan-driven, preflight validation mindset, bounded iterations), but focused on *planning* rather than execution.

End-to-end flow (initial target = `bug`):

1. Script promotes potential doc to GitHub issue.
2. Script creates a new `docs/features/active/<feature_name>/` folder from the template, recording type + issue.
3. Task Researcher agent runs `.github/prompts/research-issue.prompt.md` to produce a research artifact.
4. `prd_feature` agent runs `.github/prompts/fillout-prd-feature.prompt.md` to complete `user-story.md` + `spec.md`.
5. `atomic_planner` agent reads the research artifact and checks whether `spec.md` is sufficient to build an atomic plan; if not, it enumerates the missing research.
6. Task Researcher performs the additional research.
7. `prd_feature` incorporates the additional research into `spec.md`.
8. Script resolves the atomic plan prompt.
9. `atomic_planner` generates the atomic plan.
10. `atomic_executor` validates plan readiness (preflight checks) without executing.
11. `atomic_planner` applies changes based on validator feedback.
12. `atomic_executor` re-validates readiness (again: validate only, do not execute).

## Acceptance Criteria (early draft)

- [ ] A user can run the planning pipeline for a `bug` promotion and produce a plan file that passes execution-readiness validation.
- [ ] The workflow follows the prescribed steps (promotion → active folder creation → research → PRD/spec fill → plan generation → readiness validation), and each step produces an explicit artifact or structured output indicating success/failure.
- [ ] The pipeline treats `promotion-type` as a modifier (e.g., prompt selection, template variation, or validation rules) rather than separate hard-forked code paths; adding `feature`, `epic`, and `refactor` requires only minimal, localized type-specific logic.
- [ ] If `atomic_planner` determines the research/spec is insufficient, it outputs a concrete checklist of missing details (inputs, outputs, invariants, constraints, test requirements) and the pipeline can resume after additional research is produced.
- [ ] Failure modes are explicit and actionable:
	- [ ] Missing/invalid `--potential-path` or `--feature-name` fails fast with a clear error.
	- [ ] Missing/invalid `--promotion-type` fails fast and lists valid values.
	- [ ] Missing/invalid issue number (when required) fails fast.
- [ ] Idempotency/guardrails exist for reruns:
	- [ ] Re-running the pipeline does not silently overwrite an existing active feature folder or plan; it either reuses it safely or fails with a clear message about what already exists.
	- [ ] The “validate readiness” steps never execute tasks; they only report readiness and feedback.

## Constraints & Risks

- **Copilot CLI variability / throttling:** prompt runs can be rate-limited or produce different results; the orchestration should be resilient (clear retries/backoff policy at the script layer) and produce deterministic artifacts where possible.
- **Type-driven variability:** keeping the workflow “same steps, small modifiers” requires careful prompt and template design so type-specific needs don’t creep into separate pipelines.
- **Safety / non-execution guarantee:** the validation steps must never execute the generated plan; accidental execution would be a high-impact workflow bug.
- **Repo policy compliance:** generated specs/plans must align with existing repo policies (toolchain loop, testing rules, etc.) to avoid producing unexecutable plans.
- **Dependency boundaries:** prefer existing repo tooling; avoid adding new external dependencies unless necessary.

## Test Conditions to Consider

- [ ] Unit: promotion type parsing/validation (`bug|feature|epic|refactor`), including error messages for invalid values.
- [ ] Unit: step orchestration state machine (or equivalent) transitions and resumability.
- [ ] Unit: artifact path resolution (potential path, active feature path, research output path, plan output path).
- [ ] Integration (mocked): end-to-end `bug` run where Copilot agents are stubbed/mocked and scripts operate on in-memory or fixture data (no temp files in tests).
- [ ] Integration (workflow): “insufficient spec” branch where `atomic_planner` requests extra research and the pipeline resumes after step 7.
- [ ] CLI examples:
	- [ ] `poetry run python -m scripts.dev_tools.potential_to_issue --potential-path "docs/features/potential/<feature_name>" --promotion-type bug`
	- [ ] `poetry run python -m scripts.dev_tools.new_active_feature_folder --feature-name <feature_name> --type bug --issue-number <issue>`

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/automated-planner/` folder from the template
