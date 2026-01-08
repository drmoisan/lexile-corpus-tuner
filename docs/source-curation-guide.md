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

### Step 1: Fetch Metadata & Generate ID List

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

### Step 1.5: Enrich Gutenberg Metadata with Original Publication Year

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

### Step 2: (Optional) Curate the List

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

### Step 3: Download Texts

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

### Step 1: Configuration (Optional)

The pipeline defaults to the latest "Simple English" dump.
*   **Default URL:** `https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2`

If you want a specific version or the Standard English dump, set the `LEXILE_SIMPLE_WIKI_DUMP_URL` environment variable.

**Example (Linux/Mac):**
```bash
export LEXILE_SIMPLE_WIKI_DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
```

### Step 2: Download Dump

Run the pipeline to fetch the dump.

**Command:**
```bash
poetry run lexile-scoring-model-pipeline corpus download --sources "simple_wiki"
```

*   **Output:** The compressed dump is saved to `data/corpus/raw/simple_wiki/`.

---

## 3. Open Educational Resources (OER)

OpenStax remains Internet Archive–based (text derivatives), while CK-12 is scraped natively and works through PDF download + extraction.

### OpenStax (text via Internet Archive)

1. **Catalog:**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_catalog \
    build-oer-catalog --sources "openstax" --out-dir data/meta/catalogs
  ```
2. **Enrich (find text derivatives):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_enrichment \
    enrich-oer-catalog --catalog-file data/meta/catalogs/openstax_catalog.jsonl \
    --output data/meta/catalogs/openstax_enriched.jsonl
  ```
3. **Curate to text-only rows:**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation \
    curate-oer-catalog --catalog-dir data/meta/catalogs --require-text --sources "openstax" \
    --out-dir data/meta/catalogs
  ```
4. **Generate manifest (text only):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest \
    generate-oer-manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json \
    --validate-urls
  ```

### CK-12 (static FlexBook catalog + PDF extraction)

1. **Catalog (static JSON endpoint):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_catalog \
    build-ck12-catalog --out-dir data/meta/catalogs
  ```
  *Uses the default CK-12 FlexBook catalog JSON (`fbbrowse-prod.json`) discovered via Issue #73; override with `--catalog-url` if needed.*
2. **Enrich (per-book metadata + PDF link discovery):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.ck12_enrichment \
    enrich-ck12-catalog --catalog-file data/meta/catalogs/ck12_catalog.jsonl \
    --output data/meta/catalogs/ck12_enriched.jsonl
  ```
3. **Curate (require PDF):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_curation \
    curate-oer-catalog --catalog-dir data/meta/catalogs --require-pdf --sources "ck12" \
    --out-dir data/meta/catalogs
  ```
4. **Generate manifest (PDF-aware, optional HEAD checks):**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.oer_manifest \
    generate-oer-manifest --catalog-dir data/meta/catalogs --out data/meta/oer_sources.json \
    --validate-urls
  ```
  Notes: The manifest builder auto-selects PDFs for CK-12 and `text/*` for other sources (no `--allow-pdf` flag). After curating multiple sources, rerun `generate-oer-manifest` once so `data/meta/oer_sources.json` includes every curated source in `data/meta/catalogs`.
5. **Download PDFs (from the manifest):**  
  ```bash
  poetry run lexile-scoring-model-pipeline corpus download --sources "ck12"
  ```
6. **Extract PDF text:**  
  ```bash
  poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.extract_pdf_text \
    extract-pdf-text --source ck12 --input-dir data/corpus/raw/ck12 --output-dir data/corpus/raw/ck12
  ```
7. **Normalize extracted text:**  
  ```bash
  poetry run lexile-scoring-model-pipeline corpus normalize --sources "ck12"
  ```

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
