<!-- markdownlint-disable-file -->

# Task Research Notes: dropped-links CK-12 URL validation failure

## Research Executed

### File Analysis

- /workspaces/lexile-corpus-tuner/src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py
  - Verified CK-12 candidate selection and `validate_url()` implementation. Confirmed the User-Agent header requirement is now encoded in the request and validation is performed with HEAD.
- /workspaces/lexile-corpus-tuner/tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py
  - Located coverage for `validate_url()` and CK-12 manifest behavior; tests rely on monkeypatched `urlopen` and do not validate headers.
- /workspaces/lexile-corpus-tuner/data/meta/catalogs/ck12_curated.jsonl
  - Confirmed CK-12 download candidates are `application/json` with `/flx/get/detail/revision/{id}?tiny=true` URLs.

### Code Search Results

- validate_url
  - Definition: `oer_manifest.py` line 143.
  - Usages: `oer_manifest.py` line 248; tests in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` lines 119, 135, 150, 166.

### External Research

- #githubRepo:"" 
  - No external repository search performed (not required for this fix).
- #fetch:none (no external URLs provided for documentation lookup)
  - No fetches performed.

### Project Conventions

- Standards referenced: `general-code-change.instructions.md`, `python-code-change.instructions.md`, `general-unit-test.instructions.md`, `python-unit-test.instructions.md`, `self-explanatory-code-commenting.instructions.md`.
- Instructions followed: repo policy requires fully typed Python, Black/Ruff/Pyright, and Pytest coverage for new logic; no file-level suppressions.

## Key Discoveries

### Project Structure

The CK-12 manifest generation logic lives in `src/lexile_corpus_tuner/lexile_scoring_model/pipeline_scripts/oer_manifest.py`. Validation is performed in `validate_url()` via `urllib.request.urlopen` HEAD requests. Tests for the manifest and validation are in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py`.

### Implementation Patterns

- CK-12 candidates are selected by `application/json` format or revision-detail URL pattern.
- `validate_url()` performs HEAD requests, checks for HTTP 200, and validates content-type prefix. Failures return `(False, None, None)` when exceptions occur.
- The function now includes a browser-like User-Agent header with a comment explaining the CK-12 CloudFront requirement.

### Complete Examples

```python
def validate_url(
    url: str, allowed_content_types: list[str] | None = None
) -> tuple[bool, int | None, str | None]:
    """
    Perform a HEAD request to verify reachability and content type.
    """
    # Normalize allowed content-type prefixes for case-insensitive comparison.
    content_type_prefixes = [
        value.lower() for value in (allowed_content_types or ["text"])
    ]
    # User-Agent required: CK-12 CloudFront blocks requests without it.
    req = urllib.request.Request(  # noqa: S310 - trusted HTTPS endpoint
        url,
        method="HEAD",
        headers={"User-Agent": "Mozilla/5.0 (compatible; LexileCorpusTuner/1.0)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            status = resp.getcode()
            content_type = resp.headers.get("Content-Type")
    except Exception:
        return False, None, None
    if status != 200:
        return False, status, content_type
    if content_type and not any(
        content_type.lower().startswith(prefix) for prefix in content_type_prefixes
    ):
        return False, status, content_type
    return True, status, content_type
```

```python
def test_ck12_validation_allows_application_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CK-12 validation should allow `application/json` responses."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        del req, timeout
        return _MockResponse(200, "application/json")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    manifest_entries = oer_manifest.generate_manifest(
        [_ck12_entry()], validate_urls=True
    )
    assert manifest_entries
    assert manifest_entries[0].filename.endswith(".json")
```

### API and Schema Documentation

- CK-12 candidate URLs point to revision-detail API endpoints: `https://www.ck12.org/flx/get/detail/revision/{id}?tiny=true`.
- Validation accepts `application/json` for CK-12 and `text/*` for other sources.

### Configuration Examples

```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest \
  --catalog-dir data/meta/catalogs \
  --out data/meta/oer_sources.json \
  --validate-urls
```

### Technical Requirements

- CK-12 CloudFront blocks requests without a browser-like User-Agent.
- HEAD requests without User-Agent return HTTP 403; adding User-Agent returns HTTP 200 and `application/json`.

**Mandatory unachievable objective callout**:
- **No unachievable objectives identified.**

## Recommended Approach

Implement or confirm the User-Agent header on `urllib.request.Request` in `validate_url()` for CK-12 validation. This aligns with existing patterns (urllib, HEAD requests, Content-Type checks) and avoids new dependencies. Rejected alternatives (brief): switching to GET or `requests` would increase bandwidth or add dependency overhead without solving the core header requirement.

## Implementation Guidance

- **Objectives**: Ensure CK-12 URL validation passes by including a browser-like User-Agent header; preserve existing validation logic and content-type filtering.
- **Key Tasks**:
  - Keep or add the User-Agent header in `validate_url()`.
  - Re-run the CK-12 manifest generation with `--validate-urls` and confirm non-zero entries.
  - Run the Python toolchain (Black → Ruff → Pyright → Pytest) per repo policy after any changes.
- **Dependencies**: None; use existing `urllib.request`.
- **Success Criteria**:
  - CK-12 URL validation returns HTTP 200 for revision endpoints.
  - `oer_manifest --validate-urls` writes CK-12 entries (non-zero).
  - Tests in `tests/lexile_scoring_model/pipeline_scripts/test_oer_manifest.py` pass unchanged.