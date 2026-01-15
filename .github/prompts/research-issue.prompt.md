---
agent: 'Task Researcher Instructions'
description: 'Research-only kickoff: investigate best implementation approach for a feature using provided issue/spec docs.'
---

# Research Prompt: Feature implementation approach (research-only)

## Goal

Determine the **best implementation approach** for the feature described in the documentation provided after this prompt (typically an `issue.md`, and sometimes also a `user-story.md`).

Your output must be actionable for an engineer/agent to implement, and must be grounded in verified evidence from:

- the current codebase (existing patterns, constraints, and seams), and
- authoritative external references (official docs, well-maintained repos, etc.).

## Scope and operating rules

Follow the operating rules in your agent definition (research-only, write to `artifacts/research/`, evidence-based). Do not restate those rules in the research notes; apply them.

## Inputs / Context (authoritative)

After this kickoff prompt, you will be given one or more feature documents. Treat them as the primary specification.

Common examples (not exhaustive):

- `docs/features/active/<feature>/issue.md` (usually)
- `docs/features/active/<feature>/user-story.md` (sometimes)

Also identify (from the feature docs) the **implementation target files** in the repo (e.g., the script/module to change), and treat those as primary technical context.

## Research Outputs (required)

Create (or update) a single research file under `artifacts/research/` using the Task Researcher template exactly.

Filename convention:

- `artifacts/research/YYYYMMDD-<short-feature-name>-implementation-research.md`

In the research file, keep any discussion of non-selected approaches *brief and non-exhaustive*. After selecting a recommendation, remove detailed notes for other approaches and keep only a short “Rejected alternatives” summary (what was rejected and why).

Because the research template must be followed exactly, place the “Rejected alternatives” summary inside the required template under `## Recommended Approach` (as a short subsection/paragraph), not as a separate top-level header.

Also place the core context (issue link, feature folder, spec doc paths, target files) inside the required template under `## Research Executed` (e.g., `### File Analysis`) and/or `### Project Conventions`. Do not add a separate custom header outside the template.

## What to Investigate

### 1) Current state analysis (internal)

- Identify the implementation targets from the feature docs.
- Read the relevant modules end-to-end.
- Document:
  - current behavior and user-visible outputs
  - key abstractions and extension points
  - constraints imposed by repo policies/toolchain
  - how success/failure is currently computed and surfaced

### 2) Candidate implementation approaches (external + internal feasibility)

Research and compare at least two viable approaches (but select ONE final recommendation). Examples of “approaches” include:

- different UX/output strategies (interactive vs non-interactive fallback)
- dependency-free vs library-backed implementations
- architectural patterns (small new component vs minimal inline changes)

For any approach that introduces a new dependency:

- confirm whether it is already present in the repo’s dependency set, and
- justify why adding it is necessary and low-risk.

For each approach, gather **authoritative docs** and/or examples to support the recommendation.

### 3) Behavior semantics and edge cases

Extract the intended behavior from the feature docs and define:

- success/failure conditions
- ordering rules (if any)
- cancellation/abort behavior (if applicable)
- CI vs local terminal behavior expectations

Research comparable tools/projects that implement similar semantics.

### 4) Requirements mapping → design

Map the acceptance criteria into a concrete design and internal API boundaries.

Your mapping should include:

- a proposed state model (states, transitions)
- where/when updates occur
- how the final summary/reporting is produced
- what changes are required (files/functions), and why

### 5) Testing implications (design-level)

Without writing tests, propose a test strategy consistent with repo policy:

- Unit tests for a pure “status state machine” component.
- Unit tests for fail-fast coordination behavior.
- Integration test concept (if feasible) that avoids external dependencies and avoids temp files.

## Selection Criteria (how you choose the recommendation)

Recommend ONE approach based on:

- Alignment with the feature docs’ acceptance criteria.
- Compatibility with the repo’s supported environments (local dev + CI).
- Maintainability and simplicity (prefer small, testable units).
- Minimal disruption to existing behavior unless explicitly required.
- Dependency impact (avoid new deps unless strongly justified).

## Deliverable Requirements (in the research doc)

Your final research file must include:

- A single recommended approach with rationale, plus a brief (non-exhaustive) “Rejected alternatives” summary.
- A proposed high-level design:
  - components/classes/functions to add
  - data structures (shared status map, locks)
  - threading and cancellation model
  - pseudocode for the rendering loop and the branch update calls
- Specific implementation hooks in the relevant target file(s) (identify functions/locations).
- Risks and mitigations (Windows/CI output, subprocess behavior).
- Verification plan aligned to the repo’s toolchain and acceptance criteria.
