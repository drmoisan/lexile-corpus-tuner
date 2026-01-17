<!-- markdownlint-disable-file -->

# Task Research Notes: simple-wiki-bz2-download-81

## Research Executed

### File Analysis

- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-10-simple-wiki-bz2-download-81/issue.md
  - Issue details, repro steps, and expected behavior for auto-extraction or dual output.
- /workspaces/lexile-corpus-tuner/docs/features/active/2026-01-10-simple-wiki-bz2-download-81/spec.md
  - Spec mandates post-download extraction, atomic rename, idempotency, and doc updates.
- /workspaces/lexile-corpus-tuner/src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py
  - `download_simple_wiki_dump` downloads `.bz2` only; no extraction step.
- /workspaces/lexile-corpus-tuner/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/extract_simple_wiki_dump.py
  - `open_dump` already accepts `.bz2` via `bz2.open`, so extractor can consume compressed input.
- /workspaces/lexile-corpus-tuner/src/lexile_corpus_tuner/lexile_scoring_model/corpus/cli.py
  - `corpus download` invokes `download_simple_wiki_dump` without post-processing.
- /workspaces/lexile-corpus-tuner/tests/src/lexile_corpus_tuner/lexile_scoring_model/corpus/test_corpus_download.py
  - Existing tests cover `download_simple_wiki_dump` behavior (no extraction).
- /workspaces/lexile-corpus-tuner/tests/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/test_extract_simple_wiki_dump.py
  - Tests confirm `.bz2` handling in `open_dump`.
- /workspaces/lexile-corpus-tuner/docs/source-curation-guide.md
  - Documentation states only compressed dump is saved; no extraction mentioned.

### Code Search Results

- SimpleWikiDownloader|simple_wiki|simplewiki|extract_simple_wiki_dump
  - Matches in `corpus/download.py`, `pipeline_scripts/extract_simple_wiki_dump.py`, `corpus/normalize.py`, and relevant tests.
- download_simple_wiki_dump|corpus download
  - Matches in `corpus/cli.py` and `tests/src/.../test_corpus_download.py`.
- simplewiki|simple_wiki.*xml|simple_wiki.*bz2
  - Matches in issue/spec docs and `docs/source-curation-guide.md`.

### External Research

- #githubRepo:"(not used)"
  - Not used; repo browsing handled via fetches + local code inspection.
- #fetch:https://github.com/drmoisan/lexile-corpus-tuner/issues/81
  - Confirms issue details, expected behavior, and impact.
- #fetch:https://dumps.wikimedia.org/simplewiki/latest/
  - Directory listing confirms `simplewiki-latest-pages-articles.xml.bz2` exists with current timestamp/size.
- #fetch:https://dumps.wikimedia.org/
  - Notes rate limiting and XML dump availability cadence; suggests using mirrors when needed.
- #fetch:https://dumps.wikimedia.org/backup-index.html
  - Confirms XML dumps are provided in bz2/gz/7z and recommends bzip2 for large wiki files on Windows.
- #fetch:https://dumps.wikimedia.org/mirrors.html
  - Lists mirrors for XML dumps (useful if default host is rate limited).
- #fetch:https://docs.python.org/3/library/bz2.html
  - `bz2.open` and `BZ2File` provide streaming decompression; suitable for large files.
- #fetch:https://docs.python.org/3/library/io.html#io.TextIOWrapper
  - Clarifies text/binary stream wrapping; relevant for stream-based write handling.
- #fetch:https://docs.python.org/3/library/io.html#io.BufferedIOBase
  - Buffered binary streams support incremental read/write operations.
- #fetch:https://github.com/drmoisan/lexile-corpus-tuner
  - README shows extractor usage with `.bz2` input.
- #fetch:https://docs.github.com/
  - General reference (non-functional for this fix).

### Project Conventions

- Standards referenced: general-code-change, python-code-change, general-unit-test, python-unit-test, self-explanatory-code-commenting.
- Instructions followed: research-issue.prompt.md, repo policies in `.github/instructions/*`.

## Key Discoveries

### Project Structure

- Simple Wiki download is implemented in `lexile_scoring_model/corpus/download.py` and invoked via `corpus/cli.py`.
- Extraction logic lives in `lexile_scoring_model/pipeline_scripts/extract_simple_wiki_dump.py` and already supports `.bz2` input via `open_dump`.
- Docs (`docs/source-curation-guide.md`) still describe only the compressed dump output, no auto-extraction.
- README (per GitHub fetch) shows extractor CLI using `.bz2` input, contradicting the issue’s assumption that `.xml` is required.

### Implementation Patterns

- Downloads are streamed using `_download_file` (requests streaming + temp file + atomic replace).
- `open_dump` uses `bz2.open` for `.bz2` paths and `Path.open` otherwise, enabling transparent compressed reads.
- Tests for download functions rely on monkeypatching and mocks; extraction isn’t covered in `download_simple_wiki_dump` tests yet.

### Complete Examples

```python
def download_simple_wiki_dump(dump_url: str | None = None) -> Path:
    """
    Download a Simple English Wikipedia XML dump into RAW_ROOT/'simple_wiki'.
    """
    ensure_dirs()
    if dump_url is None:
        dump_url = os.environ.get(
            "LEXILE_SIMPLE_WIKI_DUMP_URL", DEFAULT_SIMPLE_WIKI_URL
        )
    filename = dump_url.split("/")[-1] or "simplewiki_dump.xml.bz2"
    dest = SIMPLE_WIKI_DIR / filename
    if dest.exists():
        LOGGER.info("Simple Wiki dump already exists at %s; skipping.", dest)
        return dest
    LOGGER.info("Downloading Simple Wiki dump from %s", dump_url)
    _download_file(dump_url, dest)
    return dest
```

Source: `/workspaces/lexile-corpus-tuner/src/lexile_corpus_tuner/lexile_scoring_model/corpus/download.py`

### API and Schema Documentation

- `download_simple_wiki_dump(dump_url: str | None) -> Path`: downloads dump to `data/corpus/raw/simple_wiki` but returns only `.bz2` path.
- `open_dump(path: Path) -> IO[bytes]`: handles `.bz2` with `bz2.open`, otherwise open raw file.
- `bz2.open` / `BZ2File` (Python stdlib) supports streaming decompression for large files.

### Configuration Examples

```bash
export LEXILE_SIMPLE_WIKI_DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
```

Source: `docs/source-curation-guide.md`

### Technical Requirements

- Spec requires post-download extraction with temp file + atomic rename, keeping `.bz2` as cache.
- Must remain dependency-free (Python stdlib), low memory by streaming.
- Must update README and source-curation guide to describe automated extraction.
- Wikimedia dumps list confirms `.xml.bz2` naming and availability; their docs note bzip2 recommended on Windows for large files.

**Mandatory unachievable objective callout**:
- None found.

## Recommended Approach

Implement post-download extraction in `download_simple_wiki_dump` to materialize a `.xml` file alongside the `.bz2` archive, using streaming `bz2.open` + buffered writes to a temporary `.xml.tmp` and atomic rename. Keep `.bz2` for caching, skip extraction when `.xml` already exists and is non-empty, and log whether extraction ran or was skipped. This matches the spec’s acceptance criteria, remains stdlib-only, and avoids refactoring the extractor (which already accepts `.bz2`). Update docs to clarify the automated extraction output and default workflow.

Rejected alternatives (brief): updating only `extract_simple_wiki_dump` to accept `.bz2` was rejected because it already supports `.bz2` and does not satisfy the acceptance criteria requiring a plain `.xml` output; leaving the workflow unchanged and updating docs alone fails the issue’s expected behavior.

## Implementation Guidance

- **Objectives**: ensure `corpus download --sources simple_wiki` produces both `.bz2` and `.xml`; keep behavior idempotent; avoid new dependencies; update docs to reflect automation.
- **Key Tasks**:
  - Add extraction helper in `corpus/download.py` (e.g., `_extract_bz2_to_xml(bz2_path: Path) -> Path`).
  - Invoke extraction after `_download_file` in `download_simple_wiki_dump` and when `.bz2` already exists but `.xml` is missing.
  - Use `bz2.open(bz2_path, "rb")` and `Path.open("wb")` with `shutil.copyfileobj` to stream data.
  - Write to a temp path (e.g., `.xml.tmp`) and `Path.replace()` to ensure atomicity; delete temp on exceptions.
  - Add unit tests in `test_corpus_download.py` to mock download and verify xml output is created and skip logic works.
  - Update `README.md` and `docs/source-curation-guide.md` to state `.xml` is generated automatically.
- **Dependencies**: None (stdlib `bz2`, `shutil`).
- **Success Criteria**: download step produces `.bz2` + `.xml`, extractor CLI runs without manual decompression, tests updated to cover extraction, docs updated.