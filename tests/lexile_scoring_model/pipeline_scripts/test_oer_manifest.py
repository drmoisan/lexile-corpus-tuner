from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts import oer_manifest
from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_models import (
    CatalogEntry,
    DownloadCandidate,
)


class _MockResponse:
    def __init__(self, status: int, content_type: str) -> None:
        self.headers = {"Content-Type": content_type}
        self._status = status

    def getcode(self) -> int:
        return self._status

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None


def _entry() -> CatalogEntry:
    return CatalogEntry(
        source_id="openstax",
        identifier="OpenStax_Book",
        title="Physics",
        creator="OpenStax",
        year="2020",
        language=["eng"],
        license_url="http://license",
        download_candidates=[
            DownloadCandidate(format="text/plain", url="http://example.com/file.txt")
        ],
    )


def _ck12_entry() -> CatalogEntry:
    return CatalogEntry(
        source_id="ck12",
        identifier="CK-12-Physics-FlexBook-2.0",
        title="CK-12 Physics",
        creator="CK-12",
        year="2024",
        language=["eng"],
        license_url="http://license",
        download_candidates=[
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/12345?tiny=true",
            )
        ],
    )


def _ck12_entry_multiple_revisions() -> CatalogEntry:
    return CatalogEntry(
        source_id="ck12",
        identifier="CK-12-Physics-FlexBook-2.0",
        title="CK-12 Physics",
        creator="CK-12",
        year="2024",
        language=["eng"],
        license_url="http://license",
        download_candidates=[
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/111?tiny=true",
            ),
            DownloadCandidate(
                format="application/json",
                url="https://www.ck12.org/flx/get/detail/revision/222?tiny=true",
            ),
        ],
    )


def test_build_manifest_entry_uses_stable_slug_from_identifier() -> None:
    """Manifest entry ID should derive from stable slug."""
    entry = _entry()
    manifest = oer_manifest.build_manifest_entry(entry, entry.download_candidates[0])
    assert manifest.id == "openstax-book"


def test_build_manifest_entry_sets_filename_to_txt_extension() -> None:
    """Filename should always end with .txt."""
    entry = _entry()
    manifest = oer_manifest.build_manifest_entry(entry, entry.download_candidates[0])
    assert manifest.filename.endswith(".txt")


def test_build_manifest_entry_sets_pdf_extension_for_pdf_content_type() -> None:
    """Filename should use .pdf when the candidate is a PDF derivative."""
    entry = _entry()
    candidate = DownloadCandidate(
        format="application/pdf", url="http://example.com/file.pdf"
    )
    manifest = oer_manifest.build_manifest_entry(entry, candidate)
    assert manifest.filename.endswith(".pdf")


def test_validate_url_returns_true_for_http_200_text_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation should pass for 200 responses with text content types."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        del req, timeout
        return _MockResponse(200, "text/plain")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url("http://example.com")
    assert ok is True
    assert status == 200
    assert content_type == "text/plain"


def test_validate_url_sets_user_agent_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_url should set a browser-like User-Agent header."""
    captured: dict[str, object] = {}

    def _fake_request(
        url: str,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> object:
        """Capture Request parameters for header validation."""
        captured["url"] = url
        captured["method"] = method
        captured["headers"] = headers or {}
        return object()

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        """Return a successful response without touching the network."""
        del req, timeout
        return _MockResponse(200, "text/plain")

    monkeypatch.setattr(oer_manifest.urllib.request, "Request", _fake_request)
    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url(
        "https://www.ck12.org/flx/get/detail/revision/123?tiny=true"
    )
    assert ok is True
    assert status == 200
    assert content_type == "text/plain"
    assert captured["headers"] == {
        "User-Agent": "Mozilla/5.0 (compatible; LexileCorpusTuner/1.0)"
    }


def test_validate_url_still_uses_head_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validate_url should continue using HEAD requests."""
    captured: dict[str, object] = {}

    def _fake_request(
        url: str,
        method: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> object:
        """Capture Request method without performing I/O."""
        captured["url"] = url
        captured["method"] = method
        captured["headers"] = headers or {}
        return object()

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        """Return a valid response to allow validation to proceed."""
        del req, timeout
        return _MockResponse(200, "text/plain")

    monkeypatch.setattr(oer_manifest.urllib.request, "Request", _fake_request)
    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url(
        "https://www.ck12.org/flx/get/detail/revision/123?tiny=true"
    )
    assert ok is True
    assert status == 200
    assert content_type == "text/plain"
    assert captured["method"] == "HEAD"


def test_validate_url_returns_false_for_http_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """404 responses must fail validation."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        del req, timeout
        return _MockResponse(404, "text/plain")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, _ = oer_manifest.validate_url("http://example.com")
    assert ok is False
    assert status == 404


def test_validate_url_rejects_non_200_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-200 responses should return False with the status code."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        """Return a non-200 response for status handling coverage."""
        del req, timeout
        return _MockResponse(403, "text/plain")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url("http://example.com")
    assert ok is False
    assert status == 403
    assert content_type == "text/plain"


def test_validate_url_returns_false_for_application_pdf_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-text content types must fail validation."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        del req, timeout
        return _MockResponse(200, "application/pdf")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url("http://example.com")
    assert ok is False
    assert status == 200
    assert content_type == "application/pdf"


def test_validate_url_rejects_content_type_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Content types outside allowed prefixes must fail validation."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        """Return application/json to trigger content-type mismatch."""
        del req, timeout
        return _MockResponse(200, "application/json")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url("http://example.com")
    assert ok is False
    assert status == 200
    assert content_type == "application/json"


def test_validate_url_accepts_pdf_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validation should allow PDFs when explicitly permitted."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        del req, timeout
        return _MockResponse(200, "application/pdf")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    ok, status, content_type = oer_manifest.validate_url(
        "http://example.com", allowed_content_types=["application/pdf"]
    )
    assert ok is True
    assert status == 200
    assert content_type == "application/pdf"


def test_write_manifest_json_creates_valid_json_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write_manifest_json should emit schema with sources array."""
    captured: dict[str, str] = {}

    def _fake_write_text(self: object, data: str, encoding: str = "utf-8") -> int:  # type: ignore[override]
        del encoding
        captured["payload"] = data
        return len(data)

    monkeypatch.setattr(oer_manifest.Path, "write_text", _fake_write_text)
    manifests = oer_manifest.generate_manifest([_entry()], validate_urls=False)
    oer_manifest.write_manifest_json(manifests, oer_manifest.Path("/fake.json"))
    payload = json.loads(captured["payload"])
    assert "sources" in payload
    assert payload["sources"][0]["filename"].endswith(".txt")


def test_openstax_manifest_preserves_txt_extension() -> None:
    """OpenStax entries must continue to emit `.txt` filenames."""
    manifest_entries = oer_manifest.generate_manifest([_entry()], validate_urls=False)
    assert manifest_entries
    assert manifest_entries[0].filename.endswith(".txt")


def test_ck12_manifest_uses_json_and_revision_endpoint() -> None:
    """CK-12 entries must use revision-detail URLs and `.json` filenames."""
    manifest_entries = oer_manifest.generate_manifest(
        [_ck12_entry()], validate_urls=False
    )
    assert manifest_entries
    manifest = manifest_entries[0]
    assert manifest.filename.endswith(".json")
    assert "/flx/get/detail/revision/" in manifest.url


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


def test_ck12_validation_rejects_text_plain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CK-12 validation should reject `text/plain` responses."""

    def _fake_urlopen(req: object, timeout: int = 0) -> _MockResponse:  # type: ignore[override]
        del req, timeout
        return _MockResponse(200, "text/plain")

    monkeypatch.setattr(oer_manifest.urllib.request, "urlopen", _fake_urlopen)
    manifest_entries = oer_manifest.generate_manifest(
        [_ck12_entry()], validate_urls=True
    )
    assert manifest_entries == []


def test_ck12_manifest_emits_one_entry_per_revision_candidate() -> None:
    """CK-12 entries should generate a manifest entry per revision-detail candidate."""
    manifest_entries = oer_manifest.generate_manifest(
        [_ck12_entry_multiple_revisions()], validate_urls=False
    )
    assert len(manifest_entries) == 2
    assert manifest_entries[0].filename != manifest_entries[1].filename
    assert manifest_entries[0].id != manifest_entries[1].id
