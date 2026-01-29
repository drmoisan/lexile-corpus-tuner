# Plan: Restore full toolchain green status

## Objective
- Run all repo toolchains (JSON, shell, Python, PowerShell) and fix any failures until all steps pass.

## Steps
1. ✅ Run JSON formatting/validation checks and fix any issues.
2. ✅ Run shell-qc format/check/test and resolve failures.
3. ✅ Run Python toolchain: Black → Ruff → Pyright → Pytest (with coverage) and resolve failures.
4. ✅ Run PowerShell toolchain: PoshQC format → analyze → Pester and resolve failures.
5. ✅ Re-run each toolchain in order until a full green pass is achieved.

## Notes
- Keep changes minimal and targeted to failing checks.
- Avoid introducing new dependencies unless absolutely necessary.
