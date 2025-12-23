# Docs v3 Governance and Telemetry

- Owners: docs-v3 upgrade working group (PM, Architect, QA). Primary contact: docs-governance@localhost.
- Artifact storage: pr_context outputs remain under `artifacts/pr_context.txt` and `artifacts/pr_context.appendix.txt` per run; keep last 10 runs in git-ignored `artifacts/` for audits.
- Audit cadence:
  - Monthly review of prompts/agents (`.github/prompts`, `.github/agents`)
  - Quarterly check of feature-doc extraction rules and PR intent scaffold
  - Triggered audits after any policy change in `.github/instructions/`
- Escalation: file an issue tagged `governance-docs` within 2 business days for conflicts, stale guidance, or autoclose misbehavior.

## Telemetry

- Metrics collected (manual/CI-friendly):
  - Template usage: count of pr_context generations per week (CI log scrape)
  - PR summary completeness: presence of PR Intent + Feature doc excerpts + GitHub Auto-close section
  - Reviewer satisfaction: optional 1–3 rating captured in PR review comment with tag `docs-v3-feedback`
- Reporting path: summarize metrics monthly in `docs/features/active/2025-12-18-docs-v3-upgrade/telemetry.md` (new entry per month), and email governance owners.
- Privacy: no personal data; only counts and boolean completeness flags.

## Storage & Retention

- Retain last 10 pr_context artifacts locally; do not commit artifacts.
- Prompt/agent changes must note rationale in feature plan and update governance log.

## Roles

- PM: owns cadence, agendas, and enforcement.
- Architect: reviews feature-doc extraction coverage and autoclose validation behavior.
- QA: validates tests stay green and that pr_context summary includes required sections.
