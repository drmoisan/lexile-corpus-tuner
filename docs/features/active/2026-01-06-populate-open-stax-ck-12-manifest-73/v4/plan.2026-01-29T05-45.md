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
- [x] [P2-T2] Split `scripts/dev_tools/atomic_executor/cli.py` further by extracting execution orchestration helpers into a new module under `scripts/dev_tools/atomic_executor/` and update imports
  - Acceptance: `scripts.dev_tools/atomic_executor/cli.py` remains ≤500 lines and existing tests pass for the extracted helpers.
- [ ] [P2-T3] Create `scripts/dev_tools/fix_all_status_board.py` containing the status-board helpers currently defined in `scripts/dev_tools/fix_all.py`
  - Preconditions: `scripts/dev_tools/fix_all.py` still defines `format_status_transition_line`, `render_status_board`, `format_ansi_redraw`, `should_use_interactive_board`, and `is_vt_enabled_for_stream`.
  - Acceptance: `python -c "from scripts.dev_tools.fix_all_status_board import render_status_board"` exits 0.
- [ ] [P2-T4] Refactor `scripts/dev_tools/fix_all.py` to import + re-export the status-board helpers from `scripts/dev_tools/fix_all_status_board.py`
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_fix_all.py -k "format_status_transition_line|render_status_board|should_use_interactive_board|format_ansi_redraw|is_vt_enabled_for_stream"` exits 0.
- [ ] [P2-T5] Create `scripts/dev_tools/fix_all_runner.py` containing the branch-orchestration logic currently implemented in `scripts/dev_tools/fix_all.py`
  - Preconditions: The orchestration includes `run_fix_all` plus any helper functions/classes it requires for branch execution.
  - Acceptance: `python -c "from scripts.dev_tools.fix_all_runner import run_fix_all"` exits 0.
- [ ] [P2-T6] Refactor `scripts/dev_tools/fix_all.py` to import + re-export `run_fix_all` (and required public types) from `scripts/dev_tools/fix_all_runner.py`
  - Acceptance: `poetry run pytest tests/scripts/dev_tools/test_fix_all.py` exits 0.
- [ ] [P2-T7] Reduce `scripts/dev_tools/fix_all.py` to ≤500 lines by leaving it as a façade module (re-exports + CLI entry points only)
  - Acceptance: `python -c "import pathlib; print(sum(1 for _ in pathlib.Path('scripts/dev_tools/fix_all.py').open(encoding='utf-8')))"` prints a value `<= 500`.
- [ ] [P2-T8] Create `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog_cli.py` and move the Typer CLI (`app`, `build_ck12_catalog`, `__main__` dispatch) out of `ck12_catalog.py`
  - Acceptance: `python -c "from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog_cli import app"` exits 0.
- [ ] [P2-T9] Update `ck12_catalog.py` to delegate `python -m ...ck12_catalog` execution to `ck12_catalog_cli.app()` without changing the public parsing API
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py` exits 0.
- [ ] [P2-T10] Verify `ck12_catalog.py` is ≤500 lines after extracting the CLI
  - Acceptance: `python -c "import pathlib; print(sum(1 for _ in pathlib.Path('src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py').open(encoding='utf-8')))"` prints a value `<= 500`.
- [ ] [P2-T11] Create `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment_core.py` and move revision-extraction helpers into it (`extract_revision_download_candidates`, `collect_revision_candidates_with_skip_reason`)
  - Acceptance: `python -c "from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment_core import extract_revision_download_candidates"` exits 0.
- [ ] [P2-T12] Refactor `ck12_enrichment.py` to import + re-export revision-extraction helpers from `ck12_enrichment_core.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py -k "revision"` exits 0.
- [ ] [P2-T13] Move HTTP Perma retrieval helpers into `ck12_enrichment_core.py` (`PERMA_REQUEST_HEADERS`, `REQUEST_TIMEOUT_SECONDS`, `fetch_perma_metadata`) and keep `ck12_enrichment.py` importing them
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py -k "fetch_perma_metadata"` exits 0.
- [ ] [P2-T14] Move HTML parsing helpers into `ck12_enrichment_core.py` (`REQUEST_HEADERS`, `fetch_flexbook_html`, `parse_flexbook_metadata`, `extract_pdf_url`) and keep `ck12_enrichment.py` importing them
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py -k "fetch_flexbook_html|parse_flexbook_metadata|extract_pdf_url"` exits 0.
- [ ] [P2-T15] Refactor `enrich_ck12_catalog` CLI implementation to depend on `ck12_enrichment_core.py` helpers without changing CLI flags/behavior
  - Acceptance: `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --help` exits 0.
- [ ] [P2-T16] Update `tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py` ordering test so it asserts the CLI dispatch block remains at EOF after refactor
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py -k "cli_dispatch"` exits 0.
- [ ] [P2-T17] Verify `ck12_enrichment.py` is ≤500 lines after extracting core helpers
  - Acceptance: `python -c "import pathlib; print(sum(1 for _ in pathlib.Path('src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py').open(encoding='utf-8')))"` prints a value `<= 500`.
- [ ] [P2-T18] Verify `ck12_enrichment_core.py` is ≤500 lines
  - Acceptance: `python -c "import pathlib; print(sum(1 for _ in pathlib.Path('src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment_core.py').open(encoding='utf-8')))"` prints a value `<= 500`.

### Phase 3 — Coverage improvements for OER/CK-12 modules (scenario-specific)
- [x] [P3-T1] Add `test_build_catalog_rows_empty_results` covering `oer_catalog.build_catalog_rows` returning an empty list when IA search results are empty in `tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py -k test_build_catalog_rows_empty_results` exits 0.
- [x] [P3-T2] Add `test_select_text_candidates_prefers_djvu_txt` covering `oer_catalog.select_text_candidates` preferring `_djvu.txt` over other candidates in `tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_catalog.py -k test_select_text_candidates_prefers_djvu_txt` exits 0.
- [x] [P3-T3] Add `test_enrich_row_no_text_files` covering `oer_enrichment.enrich_row` behavior when IA metadata contains no text file candidates in `tests/lexile_scoring_model/pipeline_scripts/test_oer_enrichment.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_enrichment.py -k test_enrich_row_no_text_files` exits 0.
- [x] [P3-T4] Add `test_curate_rows_records_skip_reason` covering `oer_curation.curate_rows` recording a skip reason when required candidates are missing in `tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_curation.py -k test_curate_rows_records_skip_reason` exits 0.
- [x] [P3-T5] Add `test_validate_manifest_entry_rejects_non_text_openstax` covering `oer_manifest.validate_manifest_entry` rejecting non-text OpenStax content types in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py -k test_validate_manifest_entry_rejects_non_text_openstax` exits 0.
- [x] [P3-T6] Add `test_parse_catalog_json_missing_handle_skips` covering CK-12 Browse entries missing `handle` being skipped deterministically in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_catalog.py -k test_parse_catalog_json_missing_handle_skips` exits 0.
- [x] [P3-T7] Add `test_collect_revision_candidates_reports_skip_when_no_revisions` covering CK-12 perma payloads with no revisions producing a structured skip reason in `tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py`
  - Acceptance: `poetry run pytest tests/lexile_scoring_model/pipeline_scripts/test_ck12_enrichment.py -k test_collect_revision_candidates_reports_skip_when_no_revisions` exits 0.

### Phase 4 — Verify v4 acceptance criteria (end-to-end commands)
- [x] [P4-T1] Run OpenStax catalog build command and capture logs in `artifacts/qa/remediation_73_openstax_catalog.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_catalog --sources "openstax" --out-dir data/meta/catalogs 2>&1 | tee artifacts/qa/remediation_73_openstax_catalog.log'` exits 0.
- [x] [P4-T2] Run OpenStax catalog enrichment command and append logs to `artifacts/qa/remediation_73_openstax_catalog.log`
  - Acceptance: `bash -lc 'set -euo pipefail; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_enrichment --catalog-file data/meta/catalogs/openstax_catalog.jsonl --output data/meta/catalogs/openstax_enriched.jsonl 2>&1 | tee -a artifacts/qa/remediation_73_openstax_catalog.log'` exits 0.
- [ ] [P4-T3] Run CK-12 catalog build command and capture logs in `artifacts/qa/remediation_73_ck12_catalog.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs 2>&1 | tee artifacts/qa/remediation_73_ck12_catalog.log'` exits 0.
- [ ] [P4-T4] Run CK-12 catalog enrichment command and append logs to `artifacts/qa/remediation_73_ck12_catalog.log`
  - Acceptance: `bash -lc 'set -euo pipefail; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl 2>&1 | tee -a artifacts/qa/remediation_73_ck12_catalog.log'` exits 0.
- [x] [P4-T5] Run OpenStax OER curation command and capture logs in `artifacts/qa/remediation_73_curation_openstax.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-text --sources "openstax" --out-dir data/meta/catalogs 2>&1 | tee artifacts/qa/remediation_73_curation_openstax.log'` exits 0.
- [x] [P4-T6] Run CK-12 OER curation command and capture logs in `artifacts/qa/remediation_73_curation_ck12.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs 2>&1 | tee artifacts/qa/remediation_73_curation_ck12.log'` exits 0.
- [x] [P4-T7] Run manifest generation + URL validation and capture logs in `artifacts/qa/remediation_73_manifest.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls 2>&1 | tee artifacts/qa/remediation_73_manifest.log'` exits 0.
- [x] [P4-T8] Run corpus download for OpenStax + CK-12 and capture logs in `artifacts/qa/remediation_73_download.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run lexile-scoring-model-pipeline corpus download --sources "openstax,ck12" 2>&1 | tee artifacts/qa/remediation_73_download.log'` exits 0.
- [x] [P4-T9] Run CK-12 text extraction and capture logs in `artifacts/qa/remediation_73_extract.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_ck12_text --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12 2>&1 | tee artifacts/qa/remediation_73_extract.log'` exits 0.
- [x] [P4-T10] Run corpus normalization for OpenStax + CK-12 and capture logs in `artifacts/qa/remediation_73_normalize.log`
  - Acceptance: `bash -lc 'set -euo pipefail; mkdir -p artifacts/qa; poetry run lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12" 2>&1 | tee artifacts/qa/remediation_73_normalize.log'` exits 0.

### Phase 5 — Final QA toolchain loop (mandatory)
- [x] [P5-T1] Run formatter `poetry run black .`; if it changes files, restart Phase 5 from P5-T1
  - Acceptance: Black completes with no changes in the final pass.
- [x] [P5-T2] Run linter `poetry run ruff check`; if it fails or fixes, restart Phase 5 from P5-T1
  - Acceptance: Ruff completes with exit code 0 in the final pass.
- [x] [P5-T3] Run type checker `poetry run pyright`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: Pyright completes with exit code 0 in the final pass.
- [x] [P5-T4] Run tests `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`; if tests fail, restart Phase 5 from P5-T1
  - Acceptance: Pytest completes with exit code 0 in the final pass.
- [x] [P5-T5] Run shell lint `poetry run shell-qc check`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: shell-qc check completes with exit code 0 in the final pass.
- [x] [P5-T6] Run shell tests `poetry run shell-qc test`; if they fail, restart Phase 5 from P5-T1
  - Acceptance: shell-qc test completes with exit code 0 in the final pass.
- [x] [P5-T7] Run PowerShell analyzer `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: PSScriptAnalyzer completes with no findings in the final pass.
- [x] [P5-T8] Run Pester `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: Pester completes with all tests passing in the final pass.
- [x] [P5-T9] Run JSON formatting `poetry run python -m scripts.dev_tools.format_json`; if it changes files, restart Phase 5 from P5-T1
  - Acceptance: JSON formatting completes with no changes in the final pass.
- [x] [P5-T10] Run JSON validation `poetry run python -m scripts.dev_tools.validate_json`; if it fails, restart Phase 5 from P5-T1
  - Acceptance: JSON validation completes with exit code 0 in the final pass.
