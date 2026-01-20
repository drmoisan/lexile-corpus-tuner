# <feature-name> — Spec

- **Issue:** <issue>
- **Parent (optional):** <parent-id>
- **Owner:** <name>
- **Last Updated:** <yyyy-MM-ddTHH-mm>
- **Status:** <status>
- **Version:** <version_number>

## Overview

Brief summary of the behavior and scope.
- Target users/personas and primary use cases:
- Success metrics or expected impact:

## Behavior

Describe how the feature should behave end-to-end.
- Main user flow (happy path):
- Alternate/edge flows:
- Error handling and recovery behavior:

## Inputs / Outputs

- Inputs (CLI flags, files, env vars)
- Outputs (artifacts, logs, telemetry)
- Config keys and defaults:
- Versioning or backward-compatibility constraints:

## API / CLI Surface

List commands, flags, request/response shapes, and examples.
- Example invocations with expected outputs (concise):
- Contracts and validation rules:

## Data & State

Data flow, storage, or state changes introduced by this feature.
- Data transformations and invariants:
- Caching or persistence details:
- Migration or backfill requirements (if any):

## Constraints & Risks

Performance, compatibility, security, rollout constraints.
- Limits (latency/throughput/memory) and acceptable trade-offs:
- Security/privacy considerations:
- Operational/rollout risks and mitigations:

## Implementation Strategy

- Implementation scope (what changes, not sequencing):
- New classes/functions/commands to add or update:
- Dependency changes (new/removed packages) and rationale:
- Logging/telemetry additions and locations:
- Rollout plan (feature flags, staged deploys, fallback path):

## Definition of Done

- [ ] Acceptance criteria documented and mapped to tests or demos
- [ ] Behavior matches acceptance criteria in all documented environments
- [ ] Tests updated/added (unit/integration as applicable)
- [ ] Edge cases and error handling covered by tests
- [ ] Docs updated (README, docs/features/active/... links)
- [ ] Telemetry/logging added or updated (if applicable)
- [ ] Toolchain pass completed (format → lint → type-check → test)
