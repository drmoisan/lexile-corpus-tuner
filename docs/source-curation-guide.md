# Source Curation & Download Guide

This guide provides step-by-step instructions for curating and downloading the raw data sources required for the Lexile Corpus Tuner. It assumes no prior knowledge of the tools or data locations.

## Prerequisites

1.  **Terminal Access:** Open a terminal in the root of the workspace.
2.  **Poetry:** Ensure `poetry` is installed (it is pre-installed in the dev container).
3.  **Disk Space:** ~20GB+ recommended if downloading full Wikipedia dumps and large Gutenberg subsets.

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
  --batch-size 50 \
  --fuzzy-threshold 0.9
```
```powershell
poetry run python -m lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.enrich_original_pub_year `
  --input data/meta/gutenberg/gutenberg_books.parquet `
  --output data/meta/gutenberg/gutenberg_books_enhanced.parquet `
  --checkpoint data/meta/gutenberg/.original_pub_year.ckpt `
  --cache-dir data/cache/original_pub_year `
  --rate-limit 5 `
  --batch-size 50 `
  --fuzzy-threshold 0.9
```

**What this does:**
1. Normalizes title/author strings and queries Open Library for candidates (respects rate limit and retries).
2. Selects the best match (exact → fuzzy) and records source + confidence; leaves nulls when no acceptable match.
3. Supports resume via checkpoint file and optional cache to avoid repeat lookups.
4. Writes enriched parquet to the output path without altering other columns.

**Flags to adjust (common):**
- `--rate-limit` (requests/sec) and `--batch-size` for throughput vs. quota.
- `--fuzzy-threshold` or `--disable-fuzzy` to tighten/loosen fuzzy matching.
- `--enable-wikidata` / `--enable-loc` to try fallbacks after Open Library.
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

OER content (OpenStax, CK-12) is downloaded based on a JSON manifest you must create.

### Step 1: Create the Manifest

Create a new file at `data/meta/oer_sources.json`. This file tells the downloader which textbooks to fetch.

**Command to create file (or use your editor):**
```bash
mkdir -p data/meta
touch data/meta/oer_sources.json
```

### Step 2: Populate the Manifest

Open `data/meta/oer_sources.json` and add entries. You can copy the example below.

**Content for `data/meta/oer_sources.json`:**
```json
[
  {
    "source": "openstax",
    "title": "Anatomy and Physiology",
    "url": "https://assets.openstax.org/oscms-prodcms/media/documents/AnatomyandPhysiology-OP.pdf",
    "grade_band": "9-12"
  },
  {
    "source": "ck12",
    "title": "Biology",
    "url": "https://www.ck12.org/flx/show/epub/...",
    "grade_band": "6-8"
  }
]
```
*Note: Ensure URLs are direct download links to supported formats (PDF, EPUB, HTML).*

### Step 3: Download Resources

Run the pipeline to process the manifest.

**Command:**
```bash
poetry run lexile-scoring-model-pipeline corpus download --sources "oer"
```

*   **Output:** Files are saved to `data/corpus/raw/openstax/` or `data/corpus/raw/ck12/`.

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
