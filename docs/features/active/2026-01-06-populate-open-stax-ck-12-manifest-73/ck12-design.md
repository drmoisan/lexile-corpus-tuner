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

## Manual integration run (P8-T6, 2026-01-08)
- Offline sample executed with locally hosted PDF (`http://localhost:8765/sample-flexbook.pdf`); HEAD validation to localhost failed under sandbox restrictions, so manifest was emitted without `--validate-urls`.
- Artifacts:
  - Catalog (raw): `artifacts/manual_ck12_sample/catalog_raw/ck12_catalog.jsonl`
  - Enriched: `artifacts/manual_ck12_sample/catalog_enriched/ck12_catalog.jsonl` (1 entry with PDF candidate)
  - Curated: `artifacts/manual_ck12_sample/curated/ck12_curated.jsonl` (1 kept, 0 skipped)
  - Manifest: `artifacts/manual_ck12_sample/manifest/oer_sources.sample.json` (1 entry, filename `sample-flexbook.pdf`)
  - Extraction output: `artifacts/manual_ck12_sample/corpus/ck12/sample-flexbook.txt` successfully extracted from `sample-flexbook.pdf`

## PDF extraction benchmark (P8-T7)
- Ran `extract_text_from_pdf` (pdfplumber 0.11.9) against the available CK-12 sample `artifacts/manual_ck12_sample/http_root/sample-flexbook.pdf` (609 bytes); wall time ≈0.003s, extracted 34 characters. The file is a placeholder and not representative of full FlexBooks.
- Tried to source a real CK-12 PDF via live catalog/flexbook pages: CK-12 catalog returned HTTP 403 without browser headers, and even with browser-like headers the HTML contained no `/cbook/` links or `.pdf` references (client-rendered). Need an API/JS discovery path to a true PDF export before rerunning the benchmark on a full-length FlexBook.

## Manual text quality review (P8-T8)
- Reviewed placeholder PDF `artifacts/manual_ck12_sample/http_root/sample-flexbook.pdf` against extracted text `artifacts/manual_ck12_sample/corpus/ck12/sample-flexbook.txt`: pdfplumber output matches the single-line content exactly.
- The sample lacks paragraphs, tables, or layout features, so formatting and table retention cannot yet be assessed; quality risk remains unknown for real FlexBooks.
- Next step once a genuine CK-12 export is reachable: repeat the review on a multi-page PDF with tables/figures to validate extraction fidelity before wider rollout.
