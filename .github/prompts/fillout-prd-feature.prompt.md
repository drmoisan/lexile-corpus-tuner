---
agent: 'prd_feature'
description: 'Standard loading prompt for completing partially filled user-story.md and spec.md using provided context paths.'
---

# PRD Feature Loading Prompt

Use this prompt when invoking the prd_feature agent via `/fillout-prd-feature <path1> <path2> <path3>`, where each `<path>` is a workspace-relative file containing context (e.g., partially filled `spec.md`, optional `user-story.md`, and optional supporting docs). The agent must read every provided path before writing and preserve all prefilled content.

## Objective

- Ingest the supplied context files and finish the existing `spec.md` template (and `user-story.md` if supplied) for the feature.
- Do not re-embed or rewrite the templates themselves—only supply the missing content.
- Maintain all metadata and headings exactly as provided.

## Steps

1. Load each supplied path in order; note missing or unreadable files explicitly.
2. Extract available details (issue number, owner, status, story statements, constraints, inputs/outputs, CLI flags, personas, risks, acceptance criteria, non-goals).
3. If research exists under `artifacts/research`, the caller must include the specific research file paths in the `/fillout-prd-feature` command; the agent must read each provided research file before writing.
4. Identify gaps; if the research document is insufficient to populate required sections, delegate additional research to the Task Researcher Agent and pause drafting until that research is complete.
5. Before writing, run a technical completeness check: confirm Proposed Fix, Assumptions, Data/Config Impact, Test Strategy, and Acceptance Criteria can be filled with domain-specific, testable details (no placeholders).
6. Apply the prd_feature agent’s section guidance when filling content:
   - `user-story.md` (if supplied): story statements, problem/why, personas and scenarios, acceptance criteria (checkbox, testable), non-goals.
   - `spec.md`: overview, behavior (main + notable alternatives), inputs/outputs, API/CLI surface with examples, data/state, constraints/risks, DoD checklist with evidence.
7. Use AI-to-AI optimized language: concise, unambiguous, testable, and implementation-focused; avoid narrative or marketing tone.
8. Preserve checkbox syntax and any existing text; do not delete prefilled content unless instructed.
9. Enforce template alignment: mirror each template section’s sub-bullets; do not replace with unrelated content.
10. Evidence rule: each technical claim must be grounded in supplied context; if not, request research via Task Researcher Agent.
11. Acceptance criteria rule: replace generic checklists with bug-specific, measurable criteria tied to repro and outcomes.
12. Do-not-invent rule: do not introduce new files/paths/modules unless explicitly present in context.

## Output

- Edit and fill out the target `spec.md` file (and `user-story.md` if supplied).
- If information is insufficient, pause and wait for Task Researcher Agent results before drafting.
