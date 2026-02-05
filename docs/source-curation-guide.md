# Source Curation & Download Guide

This guide provides step-by-step instructions for curating and downloading the raw data sources required for the Lexile Corpus Tuner. It assumes no prior knowledge of the tools or data locations.

## Prerequisites

1.  **Terminal Access:** Open a terminal in the root of the workspace.
2.  **Poetry:** Ensure `poetry` is installed (it is pre-installed in the dev container).
3.  **Disk Space:** ~20GB+ recommended if downloading full Wikipedia dumps and large Gutenberg subsets.
4.  **Environments:** All commands run the same on host or in the dev container when invoked from the repo root with `poetry run ...`. GUI steps require a display (use the desktop-lite dev container/VNC flow or your local desktop).

---

## 1. Project Gutenberg

The pipeline downloads individual text files based on a list of IDs. You must first generate this list by querying the Gutenberg metadata.

### Step 1.1: Fetch Metadata & Generate ID List

We use a helper script to fetch book metadata from [Gutendex](https://gutendex.com) and generate a list of English book IDs.

**Command:**
```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.build_gutenberg_id_list
```

**What this does:**
1.  Connects to the Gutendex API.
2.  Downloads metadata for all available English books (incrementally, saving progress).
3.  Saves the full metadata to `data/meta/gutenberg/gutenberg_books.parquet`.
4.  Saves a list of all English book IDs to `data/meta/gutenberg/gutenberg_ids.txt`.

*Note: This process can take several minutes to complete the first time.*

### Step 1.2: Enrich Gutenberg Metadata with Original Publication Year

After generating `gutenberg_books.parquet`, run the enrichment pipeline to add `original_pub_year`, `pub_year_confidence` (high/low/none), and `original_pub_source` while preserving the existing `issued_date` field.

**Command:**
```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.enrich_original_pub_year \
  --input data/meta/gutenberg/gutenberg_books.parquet \
  --output data/meta/gutenberg/gutenberg_books_enhanced.parquet \
  --checkpoint data/meta/gutenberg/.original_pub_year.ckpt \
  --cache-dir data/cache/original_pub_year \
  --rate-limit 5 \
  --fuzzy-threshold 0.9
```
```powershell
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.enrich_original_pub_year `
  --input data/meta/gutenberg/gutenberg_books.parquet `
  --output data/meta/gutenberg/gutenberg_books_enhanced.parquet `
  --checkpoint data/meta/gutenberg/.original_pub_year.ckpt `
  --cache-dir data/cache/original_pub_year `
  --rate-limit 5 `
  --fuzzy-threshold 0.9
```

**What this does:**
1. Normalizes title/author strings and queries Open Library for candidates (respects rate limit and retries).
2. Selects the best match (exact → fuzzy) and records source + confidence; leaves nulls when no acceptable match.
3. Supports resume via checkpoint file and optional cache to avoid repeat lookups.
4. Writes enriched parquet to the output path without altering other columns.

**Flags to adjust (common):**
- `--rate-limit` (requests/sec) for throughput vs. quota.
- `--fuzzy-threshold` or `--disable-fuzzy` to tighten/loosen fuzzy matching.
- `--checkpoint-every` to control how often progress is saved.

**Tips:**
- Keep the input parquet read-only; write to a new output file so you can diff/rollback.
- If rerunning, delete or move the checkpoint when you need a full refresh.
- Cache directory can be reused across runs to reduce API calls.

### Step 1.3: (Optional) Curate the List

By default, `gutenberg_ids.txt` contains *all* English books. If you want to download only a specific subset (e.g., only "Fiction"), you can filter this list.

**Option A: Manual Editing**
Open `data/meta/gutenberg/gutenberg_ids.txt` and remove IDs you don't want.

**Option B: Interactive Explorer**
Use the CLI explorer to query the metadata file you just created:
```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.explore_gutenberg
```
*Type `help` inside the explorer for query syntax.*

**Option C: GUI Query Builder**
For a visual interface to build complex queries (e.g., "Subject contains Fiction AND Year > 1950"):
```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui
```
*This tool allows you to construct queries visually, preview results, and export the matching IDs directly to `data/meta/gutenberg/gutenberg_ids.txt` via **File → Export Results…** (default type: Gutenberg ID List).*

**In Dev Container:** After rebuilding the container with the desktop-lite feature, access the GUI via:
1. Open browser to `http://localhost:6080`
2. Password: `vscode`
3. Run the GUI command in the container terminal

**On Host Machine:** If you have the project installed locally, run the command directly.

### Step 1.4: Download Texts

Once `data/meta/gutenberg/gutenberg_ids.txt` is ready, run the pipeline to download the actual text files.

**Command:**
```bash
poetry run lexile-scoring-model-pipeline corpus download --sources "gutenberg"
```

*   **Output:** Text files are saved to `data/corpus/raw/gutenberg/`.
*   **Testing:** To download just a few books for testing, add `--gutenberg-limit 10`:
    ```bash
    poetry run lexile-scoring-model-pipeline corpus download --sources "gutenberg" --gutenberg-limit 10
    ```

---

## 2. Wikipedia (Simple & Standard)

The pipeline downloads XML dumps directly from Wikimedia.

### Step 2.1: Configuration (Optional)

The pipeline defaults to the latest "Simple English" dump.
*   **Default URL:** `https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2`

If you want a specific version or the Standard English dump, set the `LEXILE_SIMPLE_WIKI_DUMP_URL` environment variable.

**Example (Linux/Mac):**
```bash
export LEXILE_SIMPLE_WIKI_DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
```

### Step 2.2: Download Dump

Run the pipeline to fetch the dump.

**Command:**
```bash
poetry run lexile-scoring-model-pipeline corpus download --sources "simple_wiki"
```

*   **Output:** The compressed `.bz2` dump and extracted `.xml` file are saved to
    `data/corpus/raw/simple_wiki/`.

### Step 2.3: Extract Articles

Use the extracted XML file to build the JSONL article list (the extractor also
accepts the `.bz2` file if you prefer streaming the archive).

**Command:**
```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_simple_wiki_dump \
  --dump data/corpus/raw/simple_wiki/simplewiki-latest-pages-articles.xml \
  --output data/corpus/raw/simple_wiki/simplewiki_articles.jsonl
```

---

## 3. Open Educational Resources (OER)

OpenStax remains Internet Archive–based (text derivatives), while CK-12 uses CK-12 APIs and JSON/XHTML extraction (no PDFs).

### 3.1 OpenStax (text via Internet Archive)

#### 3.1.1. **Catalog:**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_catalog \
    build-oer-catalog --sources "openstax" --out-dir data/meta/catalogs
  ```
#### 3.1.2. **Enrich (find text derivatives):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_enrichment \
    enrich-oer-catalog --catalog-file data/meta/catalogs/openstax_catalog.jsonl \
    --output data/meta/catalogs/openstax_enriched.jsonl
  ```
#### 3.1.3. **Curate to text-only rows:**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation \
    curate-oer-catalog --catalog-dir data/meta/catalogs --require-text --sources "openstax" \
    --out-dir data/meta/catalogs
  ```
#### 3.1.4. **Generate manifest (text only):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest \
    generate-oer-manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json \
    --validate-urls
  ```

### 3.2 CK-12 (Browse API + JSON/XHTML extraction)

Note: PowerShell uses the backtick (`` ` ``) for line continuation; bash uses `\`.

#### 3.2.1. **Catalog (Browse API discovery):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog \
    --out-dir data/meta/catalogs
  ```
  ```powershell
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog `
    --out-dir data/meta/catalogs
  ```
  *Uses the CK-12 Browse API (`/flx/browse/flexbook?limit=200`) and derives stable IDs from canonical `handle` values.*

  **What this does:**
  1. Downloads the CK-12 browse catalog JSON (a CK-12-provided listing of FlexBooks).
  2. Parses each row into a normalized catalog entry with stable identifiers.
  3. Derives CK-12 artifact routing metadata (artifact type + handle/slug) so the next step can call the correct Perma endpoints.
  4. Emits the result as JSONL (one JSON object per line) for deterministic downstream processing.

  **Output artifact:**
  - `data/meta/catalogs/ck12_catalog.jsonl`

  **How to know it’s correct (quick checks):**
  - The file exists and is non-empty.
  - Each line is valid JSON.
  - Rows include `identifier` and CK-12 routing metadata such as `artifact_type` and (when present) `handle`.
  - Spot-check a handful of entries: `identifier` should look like a stable slug (lowercase, hyphenated).
  
  **Note on artifact types:** The catalog parser automatically detects artifact types from the `Content_URL` path prefix. Supported artifact types are:
  - `cbook` (from `flexbooks.ck12.org/cbook/`)
  - `book` (from `www.ck12.org/book/`)
  - `tebook` (from `www.ck12.org/tebook/`)
  - `workbook` (from `www.ck12.org/workbook/`)
  - `quizbook` (from `www.ck12.org/quizbook/`)
  
  These artifact types are critical for the Perma API to correctly retrieve revision JSON data during enrichment.

#### 3.2.2. **Enrich (Perma API + revision IDs):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment \
    --catalog-file data/meta/catalogs/ck12_catalog.jsonl \
    --output data/meta/catalogs/ck12_enriched.jsonl
  ```
  ```powershell
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment `
    --catalog-file data/meta/catalogs/ck12_catalog.jsonl `
    --output data/meta/catalogs/ck12_enriched.jsonl
  ```

  **What this does:**
  1. Reads `ck12_catalog.jsonl`.
  2. For each catalog entry that has a canonical `handle`, calls the CK-12 Perma API (using the correct artifact type).
  3. Traverses revisions/children to collect revision IDs.
  4. Attaches *revision-detail JSON* download candidates (`download_candidates`) to each enriched entry.

  **Output artifact:**
  - `data/meta/catalogs/ck12_enriched.jsonl`

  **How to know it’s correct (quick checks):**
  - The file exists and contains JSONL rows with `download_candidates`.
  - Many CK-12 rows should have a candidate URL that includes `/flx/get/detail/revision/` and/or a candidate format like `application/json`.
  - The command prints skip reasons to stderr (e.g., missing handle). A small number is expected; a near-100% skip rate suggests an upstream feed/schema change.

#### 3.2.3. **Curate (require revision JSON candidates):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation \
    --catalog-dir data/meta/catalogs --require-json --sources "ck12" \
    --out-dir data/meta/catalogs
  ```
  ```powershell
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation `
    --catalog-dir data/meta/catalogs --require-json --sources "ck12" `
    --out-dir data/meta/catalogs
  ```

  **What this does:**
  1. Scans `data/meta/catalogs` for catalogs and prefers enriched files when present.
  2. Filters to entries that have at least one CK-12 revision JSON download candidate (`--require-json`).
  3. Writes curated entries plus an explicit skip log for excluded rows.

  **Output artifacts:**
  - `data/meta/catalogs/ck12_curated.jsonl` (included entries)
  - `data/meta/catalogs/ck12_skips.jsonl` (skipped entries + reason)

  **How to know it’s correct (quick checks):**
  - `ck12_curated.jsonl` exists and is non-empty.
  - Each curated row includes at least one revision JSON candidate in `download_candidates`.
  - If `ck12_skips.jsonl` is large, review the reasons to understand whether the issue is missing handles, missing revisions, or upstream payload changes.

#### 3.2.4. **Generate manifest (revision JSON entries):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest \
    --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json \
    --validate-urls
  ```
  ```powershell
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest `
    --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json `
    --validate-urls
  ```
  Notes: CK-12 entries use `.json` filenames and revision-detail URLs (`/flx/get/detail/revision/<id>?tiny=true`). After curating multiple sources, rerun the manifest generator once so `data/meta/oer_sources.json` includes every curated source in `data/meta/catalogs`.

  **What this does:**
  1. Reads every `*_curated.jsonl` file under `data/meta/catalogs`.
    2. Selects download candidates per entry:
      - CK-12: emits **one manifest row per revision-detail candidate** (so a single
       FlexBook identifier can expand to many lesson/section revision downloads).
      - Other OER sources: chooses a single `text/*` candidate.
  3. Writes a consolidated manifest consumed by `lexile-scoring-model-pipeline corpus download`.
  4. When `--validate-urls` is enabled, performs an HTTP HEAD check and requires `application/json` for CK-12 entries.

  **Output artifact:**
  - `data/meta/oer_sources.json`

  **How to know it’s correct (quick checks):**
  - The file exists and contains a top-level `sources` array.
  - CK-12 entries in `sources` have:
    - `source_id: "ck12"`
    - `filename` ending in `.json`
    - `id` / `filename` typically including a `--rev-<revision_id>` suffix (to prevent collisions)
    - URLs that typically contain `/flx/get/detail/revision/`.
  - With `--validate-urls`, invalid URLs are reported and excluded; if you end up with zero CK-12 entries, revisit steps 3.2.1–3.2.3.

#### 3.2.5. **Download revision JSON (from the manifest):**  
  ```bash
  poetry run lexile-scoring-model-pipeline corpus download --sources "oer"
  ```
  *Downloader injects the required browser-like headers for CK-12 automatically.*

  **Important:** The `corpus download` command currently treats OER as the source key `oer`.
  Passing `--sources "ck12"` will **not** trigger OER downloads.

  **What this does:**
  1. Loads `data/meta/oer_sources.json`.
  2. Iterates every manifest row and downloads each URL into a source-specific raw folder:
     - `data/corpus/raw/<source_id>/<filename>`
  3. Adds browser-like headers for CK-12 endpoints to support anonymous download.

  **Output artifacts:**
  - Downloaded CK-12 revision JSON files under `data/corpus/raw/ck12/` (filenames should end with `.json`).
  - If your manifest includes other OER sources (e.g., OpenStax), those downloads land under
    `data/corpus/raw/openstax/`.

  **How to know it’s correct (quick checks):**
  - `data/corpus/raw/ck12/` contains `.json` files with non-trivial size.
  - Spot-check a downloaded file: it should parse as JSON and typically contain a top-level `response` object.
  - If you see HTML/error payloads saved as `.json`, re-run step 3.2.4 with `--validate-urls` and confirm catalog artifact types/handles are being derived correctly.
  - It is normal for `data/corpus/raw/ck12/` to contain **many more** `.json` files than the number of curated CK-12 FlexBook identifiers, because each book identifier can expand to many revision downloads.
  - If the number of files under `data/corpus/raw/ck12/` is far smaller than the number of CK-12 entries in
    `data/meta/oer_sources.json`, re-run this step and watch the logs for skip/error messages.

#### 3.2.6. **Extract CK-12 JSON → XHTML → text:**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_ck12_text \
    --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12
  ```
  ```powershell
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_ck12_text `
    --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12
  ```

  **What this does:**
  1. Reads each downloaded CK-12 revision `.json` file.
  2. Selects the best XHTML fragment from the payload (`xhtml` preferred; `xhtml_prime` fallback).
  3. Converts XHTML to plain text (markup stripped; whitespace normalized).
  4. Writes a parallel `.txt` file next to each `.json` file using the same filename stem.

  **Output artifacts:**
  - Extracted CK-12 text files under `data/corpus/raw/ck12/`.
    - Example: `my-ck12-item--rev-123.json` → `my-ck12-item--rev-123.txt`

  **How to know it’s correct (quick checks):**
  - For most `.json` inputs, there is a corresponding `.txt` output.
  - `.txt` files are non-empty and readable.
  - It is normal to produce multiple `.txt` files per CK-12 FlexBook identifier (one per revision-detail JSON input).
  - Some short outputs are expected for image-heavy sections; the extractor logs a warning when extracted text is under 100 characters.
  - It is normal for a minority of `.json` files to have **no** `.txt` output.
    The extractor will log an error when:
    - the payload has no `response.lesson` / `response.section` / `response.chapter`, or
    - the XHTML fields are present but render to no text (e.g., markup/comments only).

#### 3.2.7. **Normalize extracted text:**  
  ```bash
  poetry run lexile-scoring-model-pipeline corpus normalize --sources "ck12"
  ```

  **What this does:**
  1. Reads CK-12 `.txt` files from `data/corpus/raw/ck12/`.
  2. Normalizes text and tokenizes it.
  3. Chunks each document into 1k–3k token windows.
  4. Writes normalized shard JSONL files used by frequency analysis and calibration.

  **Output artifacts:**
  - Normalized shard files: `data/corpus/normalized/shards/shard-<NNNNNN>-ck12.jsonl`
  - Summary index: `data/corpus/normalized/normalized_summary.json`

  **How to know it’s correct (quick checks):**
  - `data/corpus/normalized/normalized_summary.json` exists and lists shard entries where `source_id` is `ck12`.
  - Each CK-12 shard file is JSONL; each line includes `source_id`, `text_id`, and a `tokens` list.
  - Most records should have token counts between ~1,000 and ~3,000 (the normalizer enforces this for corpus windows).

### Optional: Visual Curation UI

```bash
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_ui
```

This Tkinter UI loads catalog files, lets you toggle inclusion, and exports `data/meta/oer_sources.json` without manual edits.

---

## 4. Running All Downloads

To download all configured sources in one go:

```bash
poetry run lexile-scoring-model-pipeline corpus download --sources "gutenberg,simple_wiki,oer"
```

## Next Steps

After downloading, proceed to normalization:

```bash
poetry run lexile-scoring-model-pipeline corpus normalize --sources "gutenberg,simple_wiki,oer"
```
