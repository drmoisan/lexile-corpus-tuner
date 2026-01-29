# Remediation Inputs — populate-open-stax-ck-12-manifest-73 (v4)

## Required Fixes (ordered)

1) **Fix Bash test failures**
   - **Files:** `tests/shell/test_agent_mvp.bats`, `scripts/bash/agent_mvp.sh`
   - **Current issue:** `shell-qc test` fails 3 assertions (status codes 3/4/5).
   - **Acceptance criteria:** `poetry run shell-qc test` exits 0 with all tests passing.
   - **Verification:**
     - `poetry run shell-qc check`
     - `poetry run shell-qc test`

2) **Raise new-module coverage to ≥90%**
   - **Files (current coverage below 90%):**
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py` (54%)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py` (65%)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_catalog.py` (72%)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_enrichment.py` (67%)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_curation.py` (68%)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py` (71%)
   - **Acceptance criteria:** Each module reaches ≥90% coverage for new/modified lines.
   - **Verification:**
     - `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`

3) **Enforce 500-line file limit**
   - **Files exceeding 500 lines:**
     - `scripts/dev_tools/atomic_executor/cli.py` (2389)
     - `scripts/dev_tools/fix_all.py` (1132)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_enrichment.py` (841)
     - `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/ck12_catalog.py` (521)
   - **Acceptance criteria:** Split into cohesive modules ≤500 lines without changing public behavior; update imports/tests accordingly.
   - **Verification:**
     - `wc -l <files>`
     - Full toolchain pass (see QA step).

4) **Verify v4 acceptance criteria with end-to-end commands**
   - **Acceptance criteria:**
     - Manifest entries validated with HTTP 200 + expected content-type.
     - Download and normalization succeed for `openstax,ck12` sources.
   - **Verification:**
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_catalog --sources "openstax" --out-dir data/meta/catalogs`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_enrichment --catalog-file data/meta/catalogs/openstax_catalog.jsonl --output data/meta/catalogs/openstax_enriched.jsonl`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog --out-dir data/meta/catalogs`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment --catalog-file data/meta/catalogs/ck12_catalog.jsonl --output data/meta/catalogs/ck12_enriched.jsonl`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-text --sources "openstax" --out-dir data/meta/catalogs`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation --catalog-dir data/meta/catalogs --require-json --sources "ck12" --out-dir data/meta/catalogs`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json --validate-urls`
     - `poetry run lexile-scoring-model-pipeline corpus download --sources "openstax,ck12"`
     - `poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_ck12_text --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12`
     - `poetry run lexile-scoring-model-pipeline corpus normalize --sources "openstax,ck12"`

## Do Not Do

- Do not change acceptance criteria scope or rewrite requirements.
- Do not add new runtime dependencies without explicit approval.
- Do not weaken policy thresholds (coverage, file-size limit, test requirements).

## Unmet Acceptance Criteria (v4)

- HTTP 200/content-type verification for manifest URLs is not demonstrated in this audit.
- End-to-end `download` + `normalize` for `openstax,ck12` not demonstrated.

## Required QA (after fixes)

Run in order and ensure a clean pass in a single loop:
1. `poetry run black .`
2. `poetry run ruff check`
3. `poetry run pyright`
4. `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`
5. `poetry run shell-qc check`
6. `poetry run shell-qc test`
7. `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-psscriptanalyzer.ps1`
8. `pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/dev-tools/run-pester.ps1`
9. `poetry run python -m scripts.dev_tools.format_json`
10. `poetry run python -m scripts.dev_tools.validate_json`
