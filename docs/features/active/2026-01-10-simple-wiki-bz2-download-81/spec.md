# 2026-01-10-simple-wiki-bz2-download (Spec)

- Issue: #81
- Owner: Dan Moisan
- Date: 2026-01-10
- Status: Draft

## Context
`lexile-scoring-model-pipeline corpus download --sources "simple_wiki"` currently produces only a `.xml.bz2` archive. The intended workflow and contributor docs assume a plain `.xml` exists after download, so the pipeline appears broken without an undocumented manual decompression step. The extractor can read `.bz2`, but the download step should still materialize `.xml` to align with the documented flow and acceptance criteria.

Environment:
- OS/version: Windows 10.0.26200.7462 (worktree host)
- Python version: 3.13.7 (poetry virtualenv)
- Command/flags used: `poetry run lexile-scoring-model-pipeline corpus download --sources "simple_wiki"`
- Data source or fixture: Simple English Wikipedia latest dump (default URL)

Impact / Severity:
- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low


## Repro & Evidence
Steps to Reproduce:
1. From a clean worktree, run `poetry run lexile-scoring-model-pipeline corpus download --sources "simple_wiki"`.
2. Inspect `data/corpus/raw/simple_wiki/`.
3. Attempt to feed the downloaded file into the extractor step (`extract_simple_wiki_dump`) which requires an uncompressed `.xml` file.

Expected:
- The download task should either decompress the dump automatically or emit both the `.bz2` and a plain `.xml` so the extractor can read it without any manual intervention.

Actual:
- The download task leaves only `simplewiki-latest-pages-articles.xml.bz2` in `data/corpus/raw/simple_wiki/`.
- No `.xml` file is created, so the extractor fails immediately (or requires a manual decompression step that is undocumented for the corpus download workflow).

Logs / Screenshots:
- [x] Attached minimal logs or screenshot
- Snippet:
  ```
  Directory: data/corpus/raw/simple_wiki
  simplewiki-latest-pages-articles.xml.bz2 (341,244,630 bytes) — no .xml output present
  ```


## Scope & Non-Goals
- In scope:
  - Ensure the Simple Wiki download task leaves behind a ready-to-parse XML file so the extractor CLI can run without manual intervention.
  - Keep the downloaded `.bz2` archive for caching purposes while adding an automatic extraction pass (or extractor support for `.bz2`).
  - Update contributor documentation to describe the new, fully automated workflow.
- Out of scope / non-goals:
  - Changing download behavior for Gutenberg, CK-12, or other corpus sources.
  - Reworking the extractor UI/CLI beyond enabling it to consume the prepared artifact.
  - Introducing external compression utilities (goal is to stay within Python stdlib).

## Root Cause Analysis
- `SimpleWikiDownloader` stops once `simplewiki-latest-pages-articles.xml.bz2` is persisted; there is no post-processing hook to extract the XML.
- The documented workflow implies a plain `.xml` is available for downstream steps, so the missing file looks like a failure even though the extractor can read `.bz2`.
- A manual `bunzip2` step is required but undocumented, leading to inconsistent behavior across worktrees.


## Proposed Fix
Implement automatic extraction in `download_simple_wiki_dump` so the download step produces both the `.bz2` archive and a plain `.xml` file. The approach must remain stdlib-only and stream data to avoid high memory usage.

### Detailed Technical Design

#### File naming and paths
- Input archive path: `data/corpus/raw/simple_wiki/<filename>.xml.bz2`.
- Output XML path: same directory, derived by stripping the final `.bz2` suffix.
- Fallback filename: if the URL does not include a filename, use `simplewiki_dump.xml.bz2` and `simplewiki_dump.xml`.

#### Extraction helper (new private function)
- Location: `lexile_scoring_model/corpus/download.py`.
- Signature: `_extract_simple_wiki_bz2(bz2_path: Path) -> Path`.
- Responsibilities:
  - Compute `xml_path` from `bz2_path`.
  - Build a temp path: `xml_path.with_suffix(".xml.tmp")`.
  - Stream-decompress using `bz2.open(bz2_path, "rb")` and `tmp_path.open("wb")`.
  - Use `shutil.copyfileobj` with an explicit chunk size to ensure bounded memory usage.
  - Atomically replace: `tmp_path.replace(xml_path)`.
  - On any exception, delete `tmp_path` if it exists, and re-raise.

#### Download + extraction flow (updated behavior)
- Call `ensure_dirs()`.
- Resolve `dump_url` using `LEXILE_SIMPLE_WIKI_DUMP_URL` or default.
- Compute `bz2_path` (destination) and `xml_path`.
- If `bz2_path` exists:
  - If `xml_path` exists and has size > 0, log “Skipping extraction” and return.
  - Otherwise, run `_extract_simple_wiki_bz2(bz2_path)` and return.
- If `bz2_path` does not exist:
  - Download using `_download_file(dump_url, bz2_path)`.
  - Run `_extract_simple_wiki_bz2(bz2_path)`.

#### Idempotency and failure handling
- Idempotent success state: `.bz2` exists and `.xml` exists with size > 0.
- Partial output handling: delete temp file on errors; do not delete `.bz2`.
- Retry behavior: rerunning the command should skip download and extraction when outputs are valid.

#### Logging
- INFO logs should include:
  - Download start and destination path.
  - Extraction start with source `.bz2` and destination `.xml` paths.
  - Extraction skip reason (existing `.xml` with non-zero size).

#### Optional opt-out (only if needed)
- Environment variable: `LEXILE_SIMPLE_WIKI_SKIP_EXTRACT=1`.
- Behavior: skip extraction entirely, leaving only `.bz2`.
- Must be documented and default must remain auto-extract.


## Assumptions, Constraints, Dependencies
- Assumptions:
  - Contributors have enough disk space for both the compressed and decompressed dumps (~1.5 GB total).
  - Python 3.10+ environments include `bz2` support (true for the supported interpreter matrix).
- Constraints:
  - Avoid introducing platform-specific tooling; the fix must work identically on Windows, macOS, Linux, and within the dev container.
  - Keep memory usage low by streaming rather than loading the full archive.
- External dependencies:
  - Wikimedia dump URLs retain the `simplewiki-latest-pages-articles.xml.bz2` naming convention (monitor for changes).

## Data / API / Config Impact
- User-facing or API changes:
  - CLI usage is unchanged; the improvement is transparent.
- Data or migration considerations:
  - The `data/corpus/raw/simple_wiki/` folder will contain an additional large XML file; ensure `.gitignore` keeps ignoring the folder.
  - Re-running the download should not override the XML if nothing changed, preserving determinism.
- Config additions (if opt-out implemented):
  - `LEXILE_SIMPLE_WIKI_SKIP_EXTRACT=1` skips XML extraction and leaves only the `.bz2` artifact.
- Logging/telemetry updates:
  - Add INFO-level messages for "Downloading", "Extracting", and "Skipping extraction" to aid troubleshooting.
  - Include both source `.bz2` path and destination `.xml` path in logs.

## Test Strategy
- Unit coverage:
  - Add tests for `_extract_simple_wiki_bz2` using monkeypatched `Path.open`, `Path.exists`, `Path.stat`, and `Path.replace` to avoid filesystem temp files.
  - Mock `_download_file` to emit a small, valid `.bz2` byte stream and verify both outputs are generated.
  - Verify idempotency: when `.bz2` and `.xml` exist with non-zero size, extraction is skipped.
  - Verify recovery: when `.bz2` exists and `.xml` is missing or size 0, extraction runs without re-downloading.
  - Verify failure cleanup: temp file is removed on extraction errors and `.bz2` remains.
- Integration:
  - Run `lexile-scoring-model-pipeline corpus download --sources "simple_wiki"` followed by the extractor CLI inside CI (or as part of a nightly smoke test) to guard against regressions.
- Manual validation:
  - Verify the produced XML can be opened and begins with `<mediawiki>`.
  - Confirm disk usage is acceptable and partial outputs are cleaned up if extraction is interrupted.


## Acceptance Criteria
- After running the download command, `data/corpus/raw/simple_wiki/` contains both the `.bz2` archive and an extracted XML file.
- The extractor CLI succeeds without any manual decompression instructions.
- Automated tests verifying the new behavior run in CI.
- Documentation clearly states that the workflow is automated and no longer mentions manual extraction.
- Documentation shows exact commands to run and in what sequence.

## Risks & Mitigations
- Risks:
  - Extraction failures could leave corrupt XML files that break future runs.
  - Contributors with low disk space might not be able to store both artifacts.
  - Additional processing time may slightly increase pipeline runtime.
- Mitigations:
  - Write the XML to a temp file and rename atomically; delete temp files on exceptions.
  - Document an opt-out mechanism to skip extraction if necessary (at the cost of requiring manual decompression) while keeping automated mode as default.
  - Stream the decompression to avoid extra CPU/RAM overhead and log progress so users know what's happening.

## Rollout & Follow-up
- Release/rollout steps:
  - Submit the fix via PR, ensure CI passes, and merge.
  - Announce in project notes/changelog that Simple Wiki downloads now self-extract.
- Post-fix monitoring or clean-up tasks:
  - Periodically verify that Wikimedia hasn't changed the dump naming scheme.
  - Consider adding routine checks that delete stale XML files when a newer dump is downloaded.
- Links: issue #81, future implementation PR(s), updated documentation sections.
