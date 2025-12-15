# `awesome-copilot-adapted` — User Story

- Issue: #42
- Owner: drmoisan
- Status: Done
- Last Updated: 2025-12-15

## Story Statement

- As a repo maintainer, I want upstream agents adapted with repo guardrails so that contributors use compliant, policy-aligned prompts by default.
- As a contributor, I want a short, repeatable checklist to add an upstream agent so that I can ship new roles quickly without violating repo policies.

## Problem / Why

This repo benefits from a set of well-crafted, role-specific Copilot agents (e.g., “unit test engineer”, “bug fixer”, “PowerShell specialist”), but maintaining high-quality prompts and keeping them aligned with repo policies is ongoing work.

The `awesome-copilot` project provides a curated set of open-source agent definitions (MIT licensed) that can serve as strong starting points. The gap is a repeatable, policy-compliant way to:

- Import/adapt upstream agent definitions.
- Apply repo-specific constraints (tooling, test policy, no temp files in tests, no secrets, etc.).
- Track provenance and ensure MIT attribution is preserved.


## Personas & Scenarios

- Persona: Repo Maintainer (drmoisan)
  - cares about policy compliance, licensing correctness, and keeping prompts consistent with tooling (Black/Ruff/Pyright, PoshQC/Pester)
  - constrained by time and must avoid regressions or license risk
  - goal: curate a small set of vetted, attributed agents that always defer to repo policy

- Persona: Contributor
  - cares about speed and clarity
  - constrained by learning repo policies; wants a copy/paste-safe process
  - goal: bring in a new upstream agent with minimal steps and confidence it meets policy

- Scenario: Adapting a new upstream agent
  - Trigger: contributor identifies an `awesome-copilot` agent to adopt
  - Steps: follow the adaptation checklist → add agent under `.github/agents/` → add repo-policy precedence + provenance note → update THIRD_PARTY_NOTICES → run Pester checks → promote potential doc via task
  - Obstacles: conflicting upstream instructions (e.g., auto-creating `.env`, skipping toolchain loop)
  - Outcome: agent merged with attribution and policy guardrails, promotion workflow still passes


## Acceptance Criteria

- [x] At least one upstream agent is imported and adapted under `.github/agents/`.
- [x] Each adapted agent explicitly states that repo policy documents are authoritative (and lists the relevant ones).
- [x] Conflicting instructions in adapted agents are resolved in favor of repo policy (e.g., no auto-creating `.env`, no secrets in repo).
- [x] MIT attribution is present and discoverable in the repo (e.g., `THIRD_PARTY_NOTICES.md`/`NOTICE.md`, plus per-file notes).
- [x] A short set of “How to add another upstream agent” steps exists (doc or checklist) so future updates are consistent.


## Non-Goals

- Building a generalized agent framework or runtime beyond prompt files and docs
- Adding new runtime dependencies or services for agent handling
- Auto-syncing upstream repository content; updates remain an explicit, manual decision

