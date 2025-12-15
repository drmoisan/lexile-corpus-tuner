# Add Another Upstream Agent (awesome-copilot)

1) Identify upstream agent
- Note agent file path and commit/URL in awesome-copilot.
- Confirm license is MIT.

2) Copy agent into this repo
- Place under `.github/agents/` with an `-adjusted` suffix.
- Preserve upstream content before adjustments.

3) Apply repo policy guardrails
- Ensure the "Repo Policy Compliance (Highest Priority)" section lists required instruction files.
- Remove or override any upstream guidance that conflicts with repo policy (no temp files in tests, no auto `.env`, follow toolchain loop).

4) Add provenance header
- Insert the standard header: "Adapted from github/awesome-copilot (MIT), source file: <upstream path> (retrieved <date>). See THIRD_PARTY_NOTICES.md for license and provenance."
- Update `THIRD_PARTY_NOTICES.md` with the new file and source URL/commit.

5) Validate
- Run Pester tests: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Import-Module ./scripts/powershell/PoshQC; Invoke-PoshQCTest -Root ."`
- Verify the new agent passes the attribution and policy-precedence checks.
