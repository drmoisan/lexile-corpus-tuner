# Remediation Plan — populate-open-stax-ck-12-manifest-73 (v4)

## Overview

This plan remediates the audit blockers for feature #73: failing Bash tests, Python coverage shortfalls for OER/CK-12 pipeline modules, 500-line file limit violations, and missing end-to-end acceptance verification. It follows repo policy requirements for Phase 0 context and baseline capture, then applies targeted fixes with scenario-specific tests, finishing with a full QA loop.

### Phase 0 — Context & Inputs
- [x] [P0-T1] Read `.github/copilot-instructions.md` to establish repo-wide policies
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T2] Read `.github/instructions/general-code-change.instructions.md` to confirm general code-change requirements
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T3] Read `.github/instructions/general-unit-test.instructions.md` to confirm unit test requirements
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T4] Read `.github/instructions/python-code-change.instructions.md` to confirm Python requirements
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T5] Read `.github/instructions/python-unit-test.instructions.md` to confirm Pytest requirements
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T6] Read `.github/instructions/powershell-code-change.instructions.md` to confirm PowerShell requirements
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T7] Read `.github/instructions/powershell-unit-test.instructions.md` to confirm Pester requirements
  - Acceptance: The file has been reviewed and any required constraints are noted for this remediation run.
- [x] [P0-T8] Capture Python baseline toolchain outputs in `artifacts/qa/remediation_73_baseline_python.txt` by running in order: `poetry run black .` → `poetry run ruff check` → `poetry run pyright` → `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
  - Acceptance: `artifacts/qa/remediation_73_baseline_python.txt` exists and contains each command string verbatim.
- [x] [P0-T9] Capture Bash baseline toolchain outputs in `artifacts/qa/remediation_73_baseline_shell.txt` by running in order: `poetry run shell-qc format` → `poetry run shell-qc check` → `poetry run shell-qc test`
  - Acceptance: `artifacts/qa/remediation_73_baseline_shell.txt` exists and contains each command string verbatim.
- [x] [P0-T10] Capture PowerShell baseline toolchain outputs in `artifacts/qa/remediation_73_baseline_powershell.txt` by running in order: `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/format-powershell.ps1` → `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1` → `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
  - Acceptance: `artifacts/qa/remediation_73_baseline_powershell.txt` exists and contains each command string verbatim.
- [x] [P0-T11] Capture JSON baseline toolchain outputs in `artifacts/qa/remediation_73_baseline_json.txt` by running in order: `poetry run python -m scripts.dev_tools.format_json` → `poetry run python -m scripts.dev_tools.validate_json`
  - Acceptance: `artifacts/qa/remediation_73_baseline_json.txt` exists and contains each command string verbatim.

### Phase 1 — Fix Bash test failures (agent_mvp)
- [x] [P1-T1] Identify the failing expectations in `tests/shell/test_agent_mvp.bats` for the dirty-tree status case and document the observed vs expected exit code in a short note at the top of the test file
  - Acceptance: The test file contains an explicit comment describing the expected exit code and the observed failure in the current run.
- [x] [P1-T2] Update `scripts/bash/agent_mvp.sh` to return the expected exit code when the working tree is dirty (per `test_agent_mvp.bats`)
  - Acceptance: `poetry run shell-qc test` passes the “dirty working tree” test.
- [x] [P1-T3] Update `scripts/bash/agent_mvp.sh` to return the expected exit code when the branch is protected (per `test_agent_mvp.bats`)
  - Acceptance: `poetry run shell-qc test` passes the “protected branch” test.
- [x] [P1-T4] Update `scripts/bash/agent_mvp.sh` to return the expected exit code when QC fails until max iterations (per `test_agent_mvp.bats`)
  - Acceptance: `poetry run shell-qc test` passes the “max iterations” test.

### Phase 2 — Reduce file sizes to ≤500 lines (policy compliance)
- [x] [P2-T1] Split `scripts/dev_tools/atomic_executor/cli.py` into cohesive modules by extracting argument parsing into a new module under `scripts/dev_tools/atomic_executor/` and update imports
  - Acceptance: `scripts/dev_tools/atomic_executor/cli.py` is ≤500 lines and unit tests still import the new module successfully.
- [ ] [P2-T2] Split `scripts/dev_tools/atomic_executor/cli.py` further by extracting execution orchestration helpers into a new module under `scripts/dev_tools/atomic_executor/` and update imports
  - Acceptance: `scripts.dev_tools/atomic_executor/cli.py` remains ≤500 lines and existing tests pass for the extracted helpers.
- [ ] [P2-T3] Split `scripts/dev_tools/fix_all.py` by extracting status-board rendering helpers into a new module under `scripts/dev_tools/` and update imports
  - Acceptance: `scripts/dev_tools/fix_all.py` is ≤500 lines and tests in `tests/scripts/dev_tools/test_fix_all.py` continue to pass.
- [ ] [P2-T4] Split `scripts/dev_tools/fix_all.py` further by extracting branch runner logic into a new module under `scripts/dev_tools/` and update imports
  - Acceptance: `scripts/dev_tools/fix_all.py` remains ≤500 lines and tests in `tests/scripts/dev_tools/test_fix_all.py` continue to pass.
- [ ] [P2-T5] Split `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py` into cohesive modules (request/parse helpers separated from CLI)
  - Acceptance: `ck12_enrichment.py` is ≤500 lines and `tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py` still passes.
- [ ] [P2-T6] Split `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py` into cohesive modules (API fetch/parsing separated from CLI)
  - Acceptance: `ck12_catalog.py` is ≤500 lines and `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` still passes.

### Phase 3 — Coverage improvements for OER/CK-12 modules (scenario-specific)
- [ ] [P3-T1] Add a test scenario for `oer_catalog.build_catalog_rows` handling empty IA search results in `tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py`
  - Acceptance: The focused scenario `test_build_catalog_rows_empty_results` passes in the target test module.
- [ ] [P3-T2] Add a test scenario for `oer_catalog.select_text_candidates` preferring `_djvu.txt` over other text candidates in `tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py`
  - Acceptance: The focused scenario `test_select_text_candidates_prefers_djvu_txt` passes in the target test module.
- [ ] [P3-T3] Add a test scenario for `oer_enrichment.enrich_row` when IA metadata lacks text files in `tests/lexile_scoring_model/pipeline_scripts/test_oer_enrichment.py`
  - Acceptance: The focused scenario `test_enrich_row_no_text_files` passes in the target test module.
- [ ] [P3-T4] Add a test scenario for `oer_curation.curate_rows` recording skip reasons for missing candidates in `tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`
  - Acceptance: The focused scenario `test_curate_rows_records_skip_reason` passes in the target test module.
- [ ] [P3-T5] Add a test scenario for `oer_manifest.validate_manifest_entry` rejecting non-text OpenStax content types in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`
  - Acceptance: The focused scenario `test_validate_manifest_entry_rejects_non_text_openstax` passes in the target test module.
- [ ] [P3-T6] Add a test scenario for `ck12_catalog.parse_browse_response` handling missing `handle` with clear skip reason in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
  - Acceptance: The focused scenario `test_parse_browse_response_missing_handle_skips` passes in the target test module.
- [ ] [P3-T7] Add a test scenario for `ck12_enrichment.resolve_revision_ids` when perma response lacks revisions in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py`
  - Acceptance: The focused scenario `test_resolve_revision_ids_no_revisions` passes in the target test module.

### Phase 4 — Verify v4 acceptance criteria (end-to-end commands)
- [ ] [P4-T1] Run OpenStax catalog build and enrichment commands from remediation inputs and capture logs in `artifacts/qa/remediation_73_openstax_catalog.log`
  - Acceptance: Log file exists and contains the command outputs for catalog and enrichment.
- [ ] [P4-T2] Run CK-12 catalog build and enrichment commands from remediation inputs and capture logs in `artifacts/qa/remediation_73_ck12_catalog.log`
  - Acceptance: Log file exists and contains the command outputs for catalog and enrichment.
- [ ] [P4-T3] Run OER curation commands for OpenStax and CK-12 from remediation inputs and capture logs in `artifacts/qa/remediation_73_curation.log`
  - Acceptance: Log file exists and contains the command outputs for both curation runs.
- [ ] [P4-T4] Run manifest generation with URL validation from remediation inputs and capture logs in `artifacts/qa/remediation_73_manifest.log`
  - Acceptance: Log file exists and `oer_sources.json` is written successfully with validation output.
- [ ] [P4-T5] Run `lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"` and capture logs in `artifacts/qa/remediation_73_download.log`
  - Acceptance: Log file exists and download command exits 0.
- [ ] [P4-T6] Run CK-12 text extraction command and capture logs in `artifacts/qa/remediation_73_extract.log`
  - Acceptance: Log file exists and extraction command exits 0.
- [ ] [P4-T7] Run `lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"` and capture logs in `artifacts/qa/remediation_73_normalize.log`
  - Acceptance: Log file exists and normalize command exits 0.

### Phase 5 — Final QA toolchain loop (mandatory)
- [ ] [P5-T1] Run formatter `poetry run black .`; if it changes files, restart Phase 5 from P5-T1
  - Acceptance: Black completes with no changes in the final pass.
- [ ] [P5-T2] Run linter `poetry run ruff check`; if it fails or fixes, restart Phase 5 from P5-T1
  - Acceptance: Ruff completes with exit code 0 in the final pass.
- [ ] [P5-T3] Run type checker `poetry run pyright`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: Pyright completes with exit code 0 in the final pass.
- [ ] [P5-T4] Run tests `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`; if tests fail, restart Phase 5 from P5-T1
  - Acceptance: Pytest completes with exit code 0 in the final pass.
- [ ] [P5-T5] Run shell lint `poetry run shell-qc check`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: shell-qc check completes with exit code 0 in the final pass.
- [ ] [P5-T6] Run shell tests `poetry run shell-qc test`; if they fail, restart Phase 5 from P5-T1
  - Acceptance: shell-qc test completes with exit code 0 in the final pass.
- [ ] [P5-T7] Run PowerShell analyzer `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: PSScriptAnalyzer completes with no findings in the final pass.
- [ ] [P5-T8] Run Pester `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: Pester completes with all tests passing in the final pass.
- [ ] [P5-T9] Run JSON formatting `poetry run python -m scripts.dev_tools.format_json`; if it changes files, restart Phase 5 from P5-T1
  - Acceptance: JSON formatting completes with no changes in the final pass.
- [ ] [P5-T10] Run JSON validation `poetry run python -m scripts.dev_tools.validate_json`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: JSON validation completes with exit code 0 in the final pass.
