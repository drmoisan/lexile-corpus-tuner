---
applyTo: "**"
---
# copilot-instructions.md

## Purpose of This Document

Canonical rules for future enhancements to the Lexile Corpus Tuner. Architecture, domain model, configuration defaults, and extension points now live in `README.md`; follow these instructions to keep new work aligned with repo policy.

---

## 1. Policies - Always Follow These

**CRITICAL**: When implementing **any** code, tests, or tasks, you **must** strictly adhere to these policies **without exception**. These are not guidelines—they are requirements.

Read each policy document **thoroughly** before starting work. Implement them **exactly as written**. Do not interpret, modify, or skip any requirements.

- **Reading order / authority:** General instructions first, then language-specific instructions, then unit-test addenda. developer-tooling.md and CI docs are operational guidance layered underneath.
- **Coding standards & workflow:** [general-code-change.instructions.md](./instructions/general-code-change.instructions.md)
- **Language-specific coding:** [python-code-change.instructions.md](./instructions/python-code-change.instructions.md)
- **Unit test policy (general):** [general-unit-test.instructions.md](./instructions/general-unit-test.instructions.md)
- **Unit test policy (Python):** [python-unit-test.instructions.md](./instructions/python-unit-test.instructions.md)
- **CI expectations:** [ci-documentation.md](../docs/ci-documentation.md)
- **Developer tooling:** [developer-tooling.md](../docs/developer-tooling.md)

---

## 2. Future Enhancement Work

- Current backlog and priorities: [/docs/features/backlog.md](../docs/features/backlog.md)
- Active initiatives: `/docs/features/active/` 
- Idea parking lot: [/docs/features/ideas/ideas.md](../docs/features/ideas/ideas.md)

Use these sources to align scope, status, and acceptance criteria before starting changes.

---

## 3. Operational Reminders

- Architecture/behavior reference: see [README.md](../README.md).
- Secrets: never commit keys; load OpenAI keys via `pwsh ./scripts/production/load-openai-key.ps1 -ItemName "Lexile OpenAI Key"`.







