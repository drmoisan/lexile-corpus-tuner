# awesome-copilot-adapted — Spec

- Issue: #42
- Owner: drmoisan
- Last Updated: 2025-12-15

## Overview

This repo benefits from a set of well-crafted, role-specific Copilot agents (e.g., “unit test engineer”, “bug fixer”, “PowerShell specialist”), but maintaining high-quality prompts and keeping them aligned with repo policies is ongoing work.

The `awesome-copilot` project provides a curated set of open-source agent definitions (MIT licensed) that can serve as strong starting points. The gap is a repeatable, policy-compliant way to:

- Import/adapt upstream agent definitions.
- Apply repo-specific constraints (tooling, test policy, no temp files in tests, no secrets, etc.).
- Track provenance and ensure MIT attribution is preserved.


## Behavior

Establish a lightweight, repeatable workflow for leveraging MIT-licensed agent definitions from `awesome-copilot` and adapting them for this repo.

At a high level:

1) **Adopt** selected upstream agent files into `.github/agents/`.
2) **Normalize** them so they always defer to repo policy documents (Copilot instructions, code-change policy, unit-test policy, and language-specific policies).
3) **Guardrail** against common policy conflicts (e.g., auto-creating `.env`, writing secrets, using temp files in tests, skipping the toolchain loop).
4) **Attribute** upstream sources clearly:
	- Maintain a repo-level third-party notice (or equivalent) listing the upstream project and the specific files adapted.
	- Include a short “adapted from” note in each imported agent file.
5) **Track** which agents are upstream-derived vs. repo-native and what modifications were applied.


## Inputs / Outputs

- Inputs
	- Upstream agent source path/URL and license (MIT) from `awesome-copilot`
	- Target agent file path under `.github/agents/`
	- Repo instruction files (Copilot, general code-change, general-unit-test, language-specific policies)
	- Optional: THIRD_PARTY_NOTICES/NOTICE target file location

- Outputs
	- Adapted agent files with repo-policy precedence and provenance note
	- Updated attribution entry (THIRD_PARTY_NOTICES/NOTICE)
	- Adaptation checklist doc in `docs/features/active/awesome-copilot-adapted/`
	- Passing Pester checks that assert required headers in adapted agents

## API / CLI Surface

- Dev task: `potential-to-issue.ps1 -PotentialPath <path> -PromotionType feature` (already exists; must continue to work)
- Dev task: `new-active-feature-folder.ps1 -FeatureName awesome-copilot-adapted -Type feature -IssueNumber <n>` (already used for folder creation)
- No new runtime CLI; guidance/checklists for adapting agents is documentation-only

## Data & State

- Agent files live in `.github/agents/`; adapted files gain provenance header and repo-policy precedence section
- Attribution lives in THIRD_PARTY_NOTICES/NOTICE (exact file to be chosen) referencing upstream repo/paths
- Active feature docs stored under `docs/features/active/awesome-copilot-adapted/` (spec, plan, user-story)

## Constraints & Risks

- **License compliance (MIT)**: ensure the upstream copyright + license notice is preserved for substantial portions.
- **Instruction conflicts**: upstream agents may include habits that violate repo policy (e.g., temp files in tests, skipping the toolchain loop, auto-creating `.env`). Must be overridden.
- **Scope creep**: avoid turning this into a full “agent framework”; keep it to a lightweight adoption/attribution/guardrail workflow.
- **Maintenance burden**: upstream agents evolve; decide whether to pin to a specific commit/date and treat updates as explicit work.


## Definition of Done

- [ ] Behavior matches acceptance criteria
- [ ] Tests updated/added (Pester checks for agent headers and workflow invariants)
- [ ] Docs updated (README pointer to attribution + adaptation checklist; active docs kept current)
- [ ] Telemetry/logging (if applicable)

## Seeded Test Conditions (from potential)
- [ ] Agents that touch Python enforce: Black → Ruff → Pyright → Pytest.
- [ ] Agents that touch PowerShell enforce: PoshQC format → analyze → Pester.
- [ ] Unit tests introduced by agents do not use temporary files.
- [ ] No instructions in adapted agents encourage committing secrets or auto-generating secret files.
- [ ] “Bugfix workflow” behavior is preserved: failing regression test first, then minimal fix.

