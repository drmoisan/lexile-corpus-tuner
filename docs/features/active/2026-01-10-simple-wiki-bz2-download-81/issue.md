# simple-wiki-bz2-download (Issue #81)

- Date captured: 2026-01-10
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/simple-wiki-bz2-download/ (Issue #81)

> Automation note: Keep the section headings below unchanged; the promotion tooling maps each of them into the GitHub bug issue template.

- Issue: #81
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/81
- Last Updated: 2026-01-10
## Summary

`lexile-scoring-model-pipeline corpus download --sources "simple_wiki"` saves the dump as a compressed `.xml.bz2` blob and never extracts it to the plain `.xml` file the downstream tooling expects, so the Simple Wiki extractor cannot proceed.

## Environment

- OS/version: Windows 10.0.26200.7462 (worktree host)
- Python version: 3.13.7 (poetry virtualenv)
- Command/flags used: `poetry run lexile-scoring-model-pipeline corpus download --sources "simple_wiki"`
- Data source or fixture: Simple English Wikipedia latest dump (default URL)

## Steps to Reproduce

1. From a clean worktree, run `poetry run lexile-scoring-model-pipeline corpus download --sources "simple_wiki"`.
2. Inspect `data/corpus/raw/simple_wiki/`.
3. Attempt to feed the downloaded file into the extractor step (`extract_simple_wiki_dump`) which requires an uncompressed `.xml` file.

## Expected Behavior

- The download task should either decompress the dump automatically or emit both the `.bz2` and a plain `.xml` so the extractor can read it without any manual intervention.

## Actual Behavior

- The download task leaves only `simplewiki-latest-pages-articles.xml.bz2` in `data/corpus/raw/simple_wiki/`.
- No `.xml` file is created, so the extractor fails immediately (or requires a manual decompression step that is undocumented for the corpus download workflow).

## Logs / Screenshots

- [x] Attached minimal logs or screenshot
- Snippet:
  ```
  Directory: data/corpus/raw/simple_wiki
  simplewiki-latest-pages-articles.xml.bz2 (341,244,630 bytes) — no .xml output present
  ```

## Impact / Severity

- [ ] Blocker
- [ ] High
- [x] Medium
- [ ] Low

## Suspected Cause / Notes

- `SimpleWikiDownloader` in `lexile_corpus_tuner.lexile_scoring_model.corpus.download` streams the `.bz2` payload to disk but never passes it through `bz2` for extraction, diverging from the documented workflow in the main worktree where the extractor reads an uncompressed `.xml`.

## Proposed Fix / Validation Ideas

- [ ] Unit coverage areas
- [x] Integration scenario to retest
- [x] Manual verification notes

Ideas:
- Add an automatic decompression step after download (using Python's `bz2` module or invoking `lbzip2`) and leave the `.bz2` as a cached artifact.
- Alternatively, update `extract_simple_wiki_dump` to transparently open `.bz2` inputs so the download step can remain unchanged.
- Validation: rerun the download + extractor sequence and verify `simplewiki-latest-pages-articles.xml` exists and the extractor no longer errors.

## Next Step

- [ ] Promote to GitHub issue (bug-report template)
- [ ] Move to active fix folder / branch
