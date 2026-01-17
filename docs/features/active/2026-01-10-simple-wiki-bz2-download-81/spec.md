# 2026-01-10-simple-wiki-bz2-download (Spec)

- Issue: #81
- Owner: Dan Moisan
- Date: 2026-01-10
- Status: Draft

## Context
`lexile-scoring-model-pipeline corpus download --sources "simple_wiki"` saves the dump as a compressed `.xml.bz2` blob and never extracts it to the plain `.xml` file the downstream tooling expects, so the Simple Wiki extractor cannot proceed.

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
- The extractor script assumes the XML already exists, so it fails when pointed at the compressed file produced in this worktree.
- The main worktree apparently runs a manual `bunzip2` step, but that knowledge never made it into the repo docs, so down-level worktrees silently diverge.


## Proposed Fix
- Extend `SimpleWikiDownloader` with an extraction phase:
  - After the HTTP download completes, stream the `.bz2` archive through `bz2.BZ2File` and write `simplewiki-latest-pages-articles.xml` next to it (using a temporary filename + atomic rename).
  - Skip decompression when both files already exist and the XML size is >0 to keep the command idempotent.
- If we discover streaming extraction inside the downloader is too invasive, update `extract_simple_wiki_dump` to accept `.bz2` input transparently (still no manual steps).
- Emit concise log messages indicating whether extraction was executed or skipped.
- Update `README.md` and `docs/source-curation-guide.md` so contributors know the workflow is fully automated.


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
- Logging/telemetry updates:
  - Add INFO-level messages for "Downloading", "Extracting", and "Skipping extraction" to aid troubleshooting.

## Test Strategy
- Unit coverage:
  - Mock the HTTP downloader to emit a tiny `.bz2` payload and assert that both files are written and that extraction is skipped on subsequent runs.
  - If extractor support for `.bz2` is implemented, add a focused unit test ensuring it can read compressed input paths.
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

## Risks & Mitigations
- Risks:
  - Extraction failures could leave corrupt XML files that break future runs.
  - Contributors with low disk space might not be able to store both artifacts.
  - Additional processing time may slightly increase pipeline runtime.
- Mitigations:
  - Write the XML to a temp file and rename atomically; delete temp files on exceptions.
  - Document an opt-out flag to skip extraction if necessary (at the cost of requiring manual decompression) while keeping automated mode as default.
  - Stream the decompression to avoid extra CPU/RAM overhead and log progress so users know what's happening.

## Rollout & Follow-up
- Release/rollout steps:
  - Submit the fix via PR, ensure CI passes, and merge.
  - Announce in project notes/changelog that Simple Wiki downloads now self-extract.
- Post-fix monitoring or clean-up tasks:
  - Periodically verify that Wikimedia hasn't changed the dump naming scheme.
  - Consider adding routine checks that delete stale XML files when a newer dump is downloaded.
- Links: issue #81, future implementation PR(s), updated documentation sections.
