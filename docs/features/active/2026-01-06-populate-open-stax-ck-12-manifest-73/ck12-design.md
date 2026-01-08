# CK-12 PDF validation and fallback strategy

## Purpose
- Define how CK-12 PDF links are accepted into the manifest and how extraction recovers when PDFs are malformed or blocked.
- Keep behavior deterministic and aligned with spec: require HTTP 200 and `application/pdf`, log-and-skip when validation fails.

## PDF URL validation (pre-download)
- **Discovery constraint:** only accept export/download links surfaced in FlexBook UI (expected pattern `https://flexbooks.ck12.org/flx/pdf/<slug>.pdf` or equivalent). If no export link is present, mark the entry as interactive-only and skip.
- **HEAD first:** issue an HTTP `HEAD` with a short timeout; require status 200, `Content-Type` starting with `application/pdf`, and (if present) `Content-Length > 0`.
- **Fallback when HEAD is blocked:** retry with streamed `GET` and `Range: bytes=0-0` (or equivalent minimal read) to confirm content-type without pulling the full file.
- **Skip rule:** if both HEAD and the minimal GET fail or return a non-PDF content-type, record a skip reason on the row and do not emit a manifest entry.

## Download-time safeguards
- Preserve the same validation gates before writing to disk; abort the download if the response is not `application/pdf` or is zero-length after headers.
- Keep filenames deterministic: `<stable-slug>.pdf` as emitted by `oer_manifest`.

## Extraction fallback (post-download)
- **Primary extractor:** `pdfplumber` page-by-page; fail fast on exceptions and log the PDF filename and page index.
- **Secondary extractor:** if `pdfplumber` raises or yields empty text for the entire document, retry once with `pypdf` to salvage plain text.
- **Failure handling:** when both extractors fail, log the reason and continue; do not write an empty `.txt`. Maintain a summary of failed PDFs for manual review.

## Logging expectations
- Catalog/enrichment: record “no PDF export found” vs “PDF failed HEAD/GET validation.”
- Download: record HTTP status/content-type mismatches and zero-length responses.
- Extraction: record extractor used and exception messages; keep counts for success vs failure.

## pdfplumber system prerequisites (P1-T8)
- Runtime: pure-Python stack (pdfminer.six + Pillow); no poppler/ghostscript required on Windows/macOS/Linux.
- Python: pdfplumber 0.10.x supports CPython 3.8+; ensure the runtime matches.
- Linux native libs for Pillow wheels/source builds: install `libjpeg-dev`, `zlib1g-dev`, `libopenjp2-7-dev`, and `libfreetype6-dev` (Debian/Ubuntu names) before `poetry install` to avoid missing image codecs.
- Windows/macOS: prebuilt wheels bundle the imaging codecs; no additional system packages typically needed.
- Fonts: pdfplumber uses embedded fonts via pdfminer.six; no system fonts required beyond defaults.
