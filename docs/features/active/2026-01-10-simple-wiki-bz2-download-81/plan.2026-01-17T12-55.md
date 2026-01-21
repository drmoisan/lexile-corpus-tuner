---
id: 2026-01-10-simple-wiki-bz2-download-81
status: Planned
status_color: blue
owner: drmoisan
last_updated: 2026-01-17
---

# 2026-01-10-simple-wiki-bz2-download-81 (Plan)

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

- **Issue:** 81
- **Owner:** Dan Moisan
- **Date:** 2026-01-17T12-55
- **Status:** Planned
- **Outcome:** Automated Simple Wiki download produces both `.bz2` and `.xml` with idempotent extraction and updated docs.
- **Root Cause:** Download path lacks post-processing to extract `.xml` after `.bz2` download.

## Requirements Traceability

| REQ-ID | Requirement | Source | Verification |
| --- | --- | --- | --- |
| REQ-01 | `corpus download --sources "simple_wiki"` produces both `.bz2` and `.xml` in `data/corpus/raw/simple_wiki/`. | spec.md | Pytest: new regression tests + manual CLI check in QA notes |
| REQ-02 | Extraction is idempotent: if `.xml` exists and size > 0, skip extraction. | spec.md | Pytest: idempotency test |
| REQ-03 | If `.bz2` exists and `.xml` missing/empty, extract without re-download. | spec.md | Pytest: recovery test |
| REQ-04 | Extraction uses temp file + atomic rename; temp is cleaned on error. | spec.md | Pytest: failure cleanup test |
| REQ-05 | Logging includes download path, extraction path, and skip reason. | spec.md | Pytest: log capture or code review in QA |
| REQ-06 | Implementation uses Python stdlib only (bz2 + shutil), streaming to avoid large memory use. | spec.md | Code review + lint/type checks |
| REQ-07 | README + source-curation guide document automated extraction and command sequence. | spec.md | Doc diff review |
| REQ-08 | Tests avoid temp files and external dependencies. | unit-test policy | Pytest runs without filesystem temp writes |
| REQ-09 | Full toolchain passes (Black → Ruff → Pyright → Pytest). | repo policy | QA phase commands exit 0 |

## Task Index

| Task ID | Summary |
| --- | --- |
| P0-T1 | Read copilot instructions |
| P0-T2 | Read general code-change policy |
| P0-T3 | Read general unit-test policy |
| P0-T4 | Read python code-change policy |
| P0-T5 | Read python unit-test policy |
| P0-T6 | Read self-explanatory commenting policy |
| P0-T7 | Read spec.md |
| P0-T8 | Read implementation research |
| P0-T9 | Read issue.md |
| P0-T10 | Create QA baseline output directory |
| P0-T11 | Capture Ruff baseline |
| P0-T12 | Capture Pyright baseline |
| P0-T13 | Capture Pytest baseline |
| P1-T1 | Add extraction helper success test |
| P1-T2 | Add extraction helper cleanup test |
| P1-T3 | Add idempotent skip test |
| P1-T4 | Add recovery extraction test |
| P1-T5 | Add download+extract test |
| P1-T6 | Run regression tests to confirm failure |
| P2-T1 | Add _extract_simple_wiki_bz2 helper |
| P2-T2 | Update download_simple_wiki_dump flow |
| P2-T3 | Add logging and decision comments |
| P3-T1 | Update README.md commands and notes |
| P3-T2 | Update source-curation guide steps |
| P4-T1 | Run Black (format) |
| P4-T2 | Run Ruff (lint) |
| P4-T3 | Run Pyright (type check) |
| P4-T4 | Run Pytest with coverage |

## Implementation Plan (Atomic Tasks)

### Phase 0 — Context & Inputs
- [ ] [P0-T1] Read `/workspaces/lexile-corpus-tuner/.github/copilot-instructions.md` to confirm repo-wide rules.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T2] Read `/workspaces/lexile-corpus-tuner/.github/instructions/general-code-change.instructions.md` to confirm workflow and toolchain loop requirements.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T3] Read `/workspaces/lexile-corpus-tuner/.github/instructions/general-unit-test.instructions.md` to confirm test isolation and “no temp files” policy.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T4] Read `/workspaces/lexile-corpus-tuner/.github/instructions/python-code-change.instructions.md` for typing, docstring, and lint requirements.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T5] Read `/workspaces/lexile-corpus-tuner/.github/instructions/python-unit-test.instructions.md` for pytest rules and coverage command.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T6] Read `/workspaces/lexile-corpus-tuner/.github/instructions/self-explanatory-code-commenting.instructions.md` for docstring/comment requirements.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T7] Read `/workspaces/lexile-corpus-tuner/docs/features/active/2026-01-10-simple-wiki-bz2-download-81/spec.md` for technical requirements.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T8] Read `/workspaces/lexile-corpus-tuner/docs/features/active/2026-01-10-simple-wiki-bz2-download-81/20260117-simple-wiki-bz2-download-81-implementation-research.md` for implementation guidance.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T9] Read `/workspaces/lexile-corpus-tuner/docs/features/active/2026-01-10-simple-wiki-bz2-download-81/issue.md` for repro context.
	- Acceptance: Notes recorded in plan execution log; no code changes.
- [ ] [P0-T10] Create directory `artifacts/qa` for baseline outputs.
	- Acceptance: `artifacts/qa` exists.
- [ ] [P0-T11] Capture Ruff baseline with `poetry run ruff check` into `artifacts/qa/baseline-ruff.txt`.
	- Acceptance: File exists and includes command output (exit code recorded).
- [ ] [P0-T12] Capture Pyright baseline with `poetry run pyright` into `artifacts/qa/baseline-pyright.txt`.
	- Acceptance: File exists and includes command output (exit code recorded).
- [ ] [P0-T13] Capture Pytest baseline with `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing` into `artifacts/qa/baseline-pytest.txt`.
	- Acceptance: File exists and includes command output (exit code recorded).

### Phase 1 — Regression Tests (TDD, must fail first)
- [ ] [P1-T1] Add pytest `test_extract_simple_wiki_bz2_writes_xml_and_replaces_temp` in `tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download.py` for `_extract_simple_wiki_bz2` success path using monkeypatched `Path.open`, `Path.replace`, and in-memory streams (no temp files). Covers REQ-01, REQ-04, REQ-08.
	- Acceptance: Test added and references `_extract_simple_wiki_bz2` explicitly.
- [ ] [P1-T2] Add pytest `test_extract_simple_wiki_bz2_cleans_temp_on_error` in the same file, forcing an exception during copy and asserting temp cleanup + `.bz2` preservation via monkeypatched `Path.unlink` and `Path.exists`. Covers REQ-04, REQ-08.
	- Acceptance: Test added and simulates failure cleanup.
- [ ] [P1-T3] Add pytest `test_download_simple_wiki_dump_skips_extraction_when_xml_exists` in the same file, stubbing `Path.exists`/`Path.stat` to indicate non-empty `.xml` and asserting `_extract_simple_wiki_bz2` is not called. Covers REQ-02, REQ-08.
	- Acceptance: Test added and asserts skip behavior.
- [ ] [P1-T4] Add pytest `test_download_simple_wiki_dump_extracts_when_xml_missing` in the same file, stubbing `.bz2` exists and `.xml` missing/empty, asserting `_extract_simple_wiki_bz2` is called without `_download_file`. Covers REQ-03, REQ-08.
	- Acceptance: Test added and asserts recovery behavior.
- [ ] [P1-T5] Add pytest `test_download_simple_wiki_dump_downloads_then_extracts` in the same file, stubbing `_download_file` and `_extract_simple_wiki_bz2` to verify call order when `.bz2` missing. Covers REQ-01, REQ-06, REQ-08.
	- Acceptance: Test added and asserts download+extract flow.
- [ ] [P1-T6] Run `poetry run pytest -k "extract_simple_wiki_bz2 or download_simple_wiki_dump"` and confirm failure (non-zero exit) before implementation; save output to `artifacts/qa/regression-fail.txt`.
	- Acceptance: File exists with failing output; exit code recorded as non-zero.

### Phase 2 — Minimal Fix Implementation
- [ ] [P2-T1] Implement `_extract_simple_wiki_bz2(bz2_path: Path) -> Path` in `src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py` using `bz2.open`, `shutil.copyfileobj`, temp `.xml.tmp`, and atomic `replace()`. Include full docstring (Purpose/Args/Returns/Raises/Side Effects) and intent comments for decisions. Covers REQ-01, REQ-04, REQ-06.
	- Acceptance: Function exists with docstring and uses streaming copy + atomic replace.
- [ ] [P2-T2] Update `download_simple_wiki_dump` in the same file to implement the idempotent flow: (a) skip when `.xml` exists with size > 0, (b) extract when `.bz2` exists but `.xml` missing/empty, (c) download then extract when `.bz2` missing. Add decision-logic comments. Covers REQ-01, REQ-02, REQ-03.
	- Acceptance: Branch logic matches spec; all tests in Phase 1 can pass.
- [ ] [P2-T3] Add INFO-level logs for download start, extraction start, and skip reason with `.bz2` and `.xml` paths. Covers REQ-05.
	- Acceptance: Logging statements present and include both paths.

### Phase 3 — Documentation Updates
- [ ] [P3-T1] Update `README.md` Simple Wiki workflow section to state that download now produces `.xml` automatically, and show the extractor command using the `.xml` path (also mention `.bz2` still supported). Covers REQ-07.
	- Acceptance: README reflects automated extraction and command sequence.
- [ ] [P3-T2] Update `docs/source-curation-guide.md` Simple Wiki section to note automatic extraction and list exact command sequence including where the `.xml` file appears. Covers REQ-07.
	- Acceptance: Curation guide reflects automated extraction and paths.

### Phase 4 — QA Toolchain Loop (must be a clean single pass)
- [ ] [P4-T1] Run `poetry run black .` and confirm no changes are required; if files change, repeat from P4-T1 after updates. Covers REQ-09.
	- Acceptance: Black exits 0 with no file modifications in final pass.
- [ ] [P4-T2] Run `poetry run ruff check`; if it fails or fixes files, resolve issues and restart at P4-T1. Covers REQ-09.
	- Acceptance: Ruff exits 0 in final pass.
- [ ] [P4-T3] Run `poetry run pyright`; if it fails, fix issues and restart at P4-T1. Covers REQ-09.
	- Acceptance: Pyright exits 0 in final pass.
- [ ] [P4-T4] Run `poetry run pytest --cov=src/lexile_corpus_tuner --cov=scripts/dev_tools --cov-report=term-missing`; if it fails, fix issues and restart at P4-T1. Covers REQ-09.
	- Acceptance: Pytest exits 0 with coverage output in final pass.
