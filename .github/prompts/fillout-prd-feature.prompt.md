---
agent: 'prd_feature'
description: 'Standard loading prompt for completing partially filled user-story.md and spec.md using provided context paths.'
---

# PRD Feature Loading Prompt

Use this prompt when invoking the prd_feature agent via `/prd-feature-loading <path1> <path2> <path3>`, where each `<path>` is a workspace-relative file containing context (e.g., partially filled `user-story.md`, `spec.md`, and optional supporting docs). The agent must read every provided path before writing and preserve all prefilled content.

## Objective

- Ingest the supplied context files and finish the existing `user-story.md` and `spec.md` templates for the feature.
- Do not re-embed or rewrite the templates themselves—only supply the missing content.
- Maintain all metadata and headings exactly as provided.

## Steps

1. Load each supplied path in order; note missing or unreadable files explicitly.
2. Extract available details (issue number, owner, status, story statements, constraints, inputs/outputs, CLI flags, personas, risks, acceptance criteria, non-goals).
3. Identify gaps; if critical info is absent or ambiguous, ask 3-5 concise clarifying questions before drafting.
4. Apply the prd_feature agent’s section guidance when filling content:
   - `user-story.md`: story statements, problem/why, personas and scenarios, acceptance criteria (checkbox, testable), non-goals.
   - `spec.md`: overview, behavior (main + notable alternatives), inputs/outputs, API/CLI surface with examples, data/state, constraints/risks, DoD checklist with evidence.
5. Keep language crisp, actionable, and testable; avoid marketing tone.
6. Preserve checkbox syntax and any existing text; do not delete prefilled content unless instructed.

## Output

- Edit and fill out the target `user-story.md` and `spec.md` files that are supplied.
- If information is insufficient, pause after clarifying questions and wait for answers before drafting.
