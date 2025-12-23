# awesome-copilot-adapted (Issue #42)

- Date captured: 2025-12-15
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/awesome-copilot-adapted/ (Issue #42)

- Issue: #42
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/42
## Problem / Why

This repo benefits from a set of well-crafted, role-specific Copilot agents (e.g., “unit test engineer”, “bug fixer”, “PowerShell specialist”), but maintaining high-quality prompts and keeping them aligned with repo policies is ongoing work.

The `awesome-copilot` project provides a curated set of open-source agent definitions (MIT licensed) that can serve as strong starting points. The gap is a repeatable, policy-compliant way to:

- Import/adapt upstream agent definitions.
- Apply repo-specific constraints (tooling, test policy, no temp files in tests, no secrets, etc.).
- Track provenance and ensure MIT attribution is preserved.

## Proposed Behavior

Establish a lightweight, repeatable workflow for leveraging MIT-licensed agent definitions from `awesome-copilot` and adapting them for this repo.

At a high level:

1) **Adopt** selected upstream agent files into `.github/agents/`.
2) **Normalize** them so they always defer to repo policy documents (Copilot instructions, code-change policy, unit-test policy, and language-specific policies).
3) **Guardrail** against common policy conflicts (e.g., auto-creating `.env`, writing secrets, using temp files in tests, skipping the toolchain loop).
4) **Attribute** upstream sources clearly:
	- Maintain a repo-level third-party notice (or equivalent) listing the upstream project and the specific files adapted.
	- Include a short “adapted from” note in each imported agent file.
5) **Track** which agents are upstream-derived vs. repo-native and what modifications were applied.

## Acceptance Criteria (early draft)

- [ ] At least one upstream agent is imported and adapted under `.github/agents/`.
- [ ] Each adapted agent explicitly states that repo policy documents are authoritative (and lists the relevant ones).
- [ ] Conflicting instructions in adapted agents are resolved in favor of repo policy (e.g., no auto-creating `.env`, no secrets in repo).
- [ ] MIT attribution is present and discoverable in the repo (e.g., `THIRD_PARTY_NOTICES.md`/`NOTICE.md`, plus per-file notes).
- [ ] A short set of “How to add another upstream agent” steps exists (doc or checklist) so future updates are consistent.

## Constraints & Risks

- **License compliance (MIT)**: ensure the upstream copyright + license notice is preserved for substantial portions.
- **Instruction conflicts**: upstream agents may include habits that violate repo policy (e.g., temp files in tests, skipping the toolchain loop, auto-creating `.env`). Must be overridden.
- **Scope creep**: avoid turning this into a full “agent framework”; keep it to a lightweight adoption/attribution/guardrail workflow.
- **Maintenance burden**: upstream agents evolve; decide whether to pin to a specific commit/date and treat updates as explicit work.

## Test Conditions to Consider

- [ ] Agents that touch Python enforce: Black → Ruff → Pyright → Pytest.
- [ ] Agents that touch PowerShell enforce: PoshQC format → analyze → Pester.
- [ ] Unit tests introduced by agents do not use temporary files.
- [ ] No instructions in adapted agents encourage committing secrets or auto-generating secret files.
- [ ] “Bugfix workflow” behavior is preserved: failing regression test first, then minimal fix.

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/awesome-copilot-adapted/` folder from the template
- [ ] Add a lightweight attribution doc (`THIRD_PARTY_NOTICES.md`/`NOTICE.md`) describing upstream agent sources and modifications


