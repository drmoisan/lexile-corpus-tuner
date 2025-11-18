# Lexile-Faithful Text Difficulty Pipeline

This document specifies a **Lexile-faithful text difficulty pipeline** for your `lexile-corpus-tuner` project, aligned as closely as practical with the MetaMetrics / Lexile approach, and integrated with a regression calibration layer that uses **official Lexile measures for known texts** as ground truth (rather than `lexile-determination-v2`).

It is designed so Codex (or any codegen agent) can implement it module-by-module.

High-level layers:

1. **Corpus Layer** – Build a large proxy Lexile/MetaMetrics corpus and compute word frequencies (per 5M words) and log frequencies.
2. **Analyzer Layer** – Implement a Lexile-style analyzer:

   * Shared normalization and tokenization.
   * Sentence segmentation.
   * 125+ word slices extended to sentence boundaries.
   * Features: mean sentence length (MSL) and mean log word frequency (MLF).
   * Special-case adjustments (picture books, emergent nonfiction).
3. **Calibration Layer** – Fit a regression model so:

   * `estimate_lexile_from_features` maps analyzer features → **official Lexile scores** from a curated catalog of texts with known measures.
   * The fitted model is stored as a JSON spec and used at runtime with no sklearn dependency.

---

## I. Objectives and Overview

### 1.1 What we are replicating from the Lexile framework

From the MetaMetrics documentation:

- Text difficulty is modeled using two primary text features:
  - **Syntactic complexity**: Mean sentence length (MSL) in words.
  - **Semantic difficulty**: Mean log word frequency (MLF), where word frequencies are computed from a large reference corpus (~600M–1.4B words) and scaled to a **per-5-million-word** rate.

- The Lexile Analyzer:
  - Breaks texts into slices containing at least **125 words**; if the 125th word falls in the middle of a sentence, it continues to the **end of the sentence**.
  - Computes MSL and MLF for slices / whole text.
  - Combines these via a **specification equation** (a regression) to produce a Lexile measure.
  - Applies special-case adjustments for:
    - **Picture books** (direct picture support).
    - **Short nonfiction emergent reader texts** (< 500 words).

### 1.2 High-level architecture

We mirror this with three layers:

1. **Corpus Builder (Proxy MetaMetrics Corpus)**  
   - Build a large, mixed corpus from open sources (e.g., Gutenberg, Simple English Wikipedia, OER textbooks).
   - Normalize and tokenize text using a **shared text utils module**.
   - Shard into normalized JSONL files.
   - Compute global word frequencies and **log frequency per 5M words**; save as TSV + metadata.

2. **Lexile-Style Analyzer**  
   - Use shared normalization and tokenization for any input text.
   - Split into sentences.
   - Build slices of ≥125 tokens extended to sentence boundaries.
   - For each slice:
     - Compute MSL and MLF (using the frequency table).
   - Aggregate to document-level features.
   - Apply a regression model (from calibration) to estimate a Lexile-like score.
   - Apply optional adjustments for picture books / emergent nonfiction.

3. **Regression Calibration Layer (against `lexile-determination-v2`)**  
   - Collect a calibration set of texts.
   - For each text: run both:
     - Our Lexile-style analyzer → features.
     - `lexile-determination-v2` → teacher Lexile.
   - Fit a regression model (e.g., ElasticNet/Ridge) using engineered features (MSL, MLF, length, variability, interactions).
   - Store learned coefficients as a JSON model spec.
   - Runtime: `estimate_lexile_from_features` loads JSON spec and computes the final estimate.

---

## II. Corpus Builder (Proxy MetaMetrics Corpus)

The goal is not an exact replica of the MetaMetrics corpus but a **large, reasonably school-like English corpus** that supports stable word frequency estimation.

### 2.1 Data layout

At repo root:

```text
data/
  corpus/
    raw/
      gutenberg/
      simple_wiki/
    normalized/
      shards/
        shard-000001-gutenberg_child.jsonl
        shard-000002-gutenberg_other.jsonl
        shard-000003-simple_wiki.jsonl
        ...
      normalized_summary.json
  freq/
    word_frequencies.tsv
    word_frequencies.meta.json
  meta/
    corpus_sources.json
````

Constraints:

* **Do not** commit `data/corpus` or `data/freq` to git by default; add them to `.gitignore`.
* If needed, commit tiny examples elsewhere (e.g., `examples/`) but keep real corpus out of VCS.

### 2.2 Corpus sources (recommended, but extensible)

**Tier A – “School-ish” and educational (preferred)**

* **Open-access textbooks / OERs** (e.g., OpenStax, CK-12), especially for science and social studies.
* **Children’s / YA public-domain books** from Project Gutenberg:

  * Filter by:

    * Language = English.
    * Prose (exclude poetry/drama).
    * Tags: “juvenile”, “children’s”, etc.
* **Simple English Wikipedia**:

  * Entire Simple English Wikipedia dump; edited informational text with simplified vocab.

**Tier B – General edited English (to fill out tails)**

* Remaining Project Gutenberg prose.
* Open newswire/magazine corpora if available.

You can manage source weighting later via metadata (see `corpus_sources.json`).

### 2.3 Corpus metadata file

Create:

```json
{
  "version": "2025-11-17",
  "sources": [
    {
      "id": "gutenberg_child",
      "description": "Project Gutenberg juvenile / children’s prose",
      "path_glob": "data/corpus/normalized/shards/*-gutenberg_child.jsonl",
      "weight": 1.0
    },
    {
      "id": "simple_wiki",
      "description": "Simple English Wikipedia articles",
      "path_glob": "data/corpus/normalized/shards/*-simple_wiki.jsonl",
      "weight": 1.0
    }
  ],
  "notes": "Proxy Lexile-like corpus built from open sources. Tokenization + normalization are fixed by lexile_corpus_tuner.corpus."
}
```

File: `data/meta/corpus_sources.json`.

Codex SHOULD keep this machine-readable; you can add more sources as needed.

---

## III. Corpus Download Module

Create:

```text
src/lexile_corpus_tuner/corpus/
  __init__.py
  download.py
  normalize.py
  frequencies.py
  cli.py
```

### 3.1 `download.py`

```python
# src/lexile_corpus_tuner/corpus/download.py

from pathlib import Path
from typing import Iterable

RAW_ROOT = Path("data/corpus/raw")

def ensure_dirs() -> None:
    (RAW_ROOT / "gutenberg").mkdir(parents=True, exist_ok=True)
    (RAW_ROOT / "simple_wiki").mkdir(parents=True, exist_ok=True)

def download_gutenberg_subset(limit: int | None = None) -> None:
    """
    Download a curated subset of Project Gutenberg texts into RAW_ROOT/'gutenberg'.

    - Limit by language == 'en'
    - Prefer prose over poetry/drama
    - Prefer later publication dates
    - (Optional) allow a hard limit on number of books for testing.
    - Use a local list of ebook IDs in, e.g., data/meta/gutenberg_ids.txt
      rather than trying to scrape everything dynamically.
    """
    ...

def download_simple_wiki_dump(dump_url: str | None = None) -> Path:
    """
    Download a Simple English Wikipedia XML dump (or accept a local path)
    into RAW_ROOT/'simple_wiki' and return the file path.

    - dump_url can be provided explicitly or via an environment variable
      LEXILE_SIMPLE_WIKI_DUMP_URL.
    - If the file already exists, skip re-downloading.
    """
    ...
```

Implementation details:

* Use HTTP download via `requests` or `urllib` (Codex to implement).
* Use idempotent behavior: if target file exists, skip download.
* Log meaningful progress steps.

---

## IV. Normalization & Tokenization

### 4.1 Shared text utilities

Create a shared module:

```text
src/lexile_corpus_tuner/textutils.py
```

Functions:

```python
# src/lexile_corpus_tuner/textutils.py
import re
import unicodedata
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*", re.UNICODE)

def normalize_text(s: str) -> str:
    """
    Normalize text for both corpus frequency computation and analyzer.
    Steps:
    - Unicode normalize (NFKC or NFC).
    - Lowercase.
    - Replace any sequence of whitespace with a single space.
    """
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def iter_tokens(s: str) -> Iterable[str]:
    """
    Yield word tokens from normalized text using TOKEN_RE.

    Tokens include letters/digits, plus internal apostrophes/hyphens.
    Example: "don't" and "mother-in-law" are single tokens.
    """
    norm = normalize_text(s)
    for m in TOKEN_RE.finditer(norm):
        yield m.group(0)
```

**Important:** The **same** `normalize_text` and `iter_tokens` are used:

* When building corpus frequencies.
* When analyzing any text for Lexile features.

This is critical to keep MLF consistent.

---

## V. Normalization & Sharding (Corpus)

### 5.1 Normalized shards

Create:

```python
# src/lexile_corpus_tuner/corpus/normalize.py

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, List, Dict, Any
import json

from ..textutils import normalize_text, iter_tokens

NORMALIZED_ROOT = Path("data/corpus/normalized")
SHARDS_ROOT = NORMALIZED_ROOT / "shards"

@dataclass
class NormalizedShardMeta:
    shard_id: str
    source_id: str
    num_tokens: int
    num_texts: int

def normalize_all_sources(shard_size_tokens: int = 100_000) -> List[NormalizedShardMeta]:
    """
    Iterate through all raw sources (Gutenberg + Simple Wiki),
    normalize and tokenize them, and emit fixed-size shards as JSONL.

    Returns metadata for all shards and writes:
      - data/corpus/normalized/shards/<shard-id>-<source-id>.jsonl
      - data/corpus/normalized/normalized_summary.json
    """
    ...
```

Implementation outline:

1. Ensure `NORMALIZED_ROOT` and `SHARDS_ROOT` exist.

2. Implement helper(s):

   ```python
   def iter_raw_texts() -> Iterator[tuple[str, str, str]]:
       """
       Yields (source_id, text_id, raw_text).

       - Scan RAW_ROOT/'gutenberg' for *.txt -> ('gutenberg_child' or 'gutenberg_other', 'gutenberg-<id>', text)
       - Scan RAW_ROOT/'simple_wiki' for extracted articles -> ('simple_wiki', 'simple_wiki-<id>', text)
       """
       ...
   ```

3. For each `(source_id, text_id, raw_text)`:

   * Normalize: `norm = normalize_text(raw_text)`.

   * Tokenize: `tokens = list(iter_tokens(norm))`.

   * Append to current shard buffers:

     * `current_tokens` (list of tokens).
     * `current_records` (list of JSON-serializable objects).

   * For each text, you can represent its contribution within shard as:

     ```python
     record = {
         "source_id": source_id,
         "text_id": text_id,
         "tokens": tokens,
     }
     ```

4. When `len(current_tokens) >= shard_size_tokens`:

   * Write all `current_records` to `SHARDS_ROOT / f"shard-{shard_index:06d}-{source_id}.jsonl"`.
   * Each `record` is a JSON line.
   * Record `NormalizedShardMeta` for that shard.
   * Reset `current_records` and `current_tokens`, increment `shard_index`.

5. At the end, if any tokens remain, write a final shard.

6. Write `normalized_summary.json` with:

   ```json
   {
     "version": "2025-11-17",
     "num_shards": N,
     "shards": [
       {
         "shard_id": "shard-000001-gutenberg_child",
         "source_id": "gutenberg_child",
         "num_tokens": ...,
         "num_texts": ...
       }
       ...
     ]
   }
   ```

---

## VI. Frequency Computation (Corpus)

### 6.1 Frequencies module

Create:

```python
# src/lexile_corpus_tuner/corpus/frequencies.py

from pathlib import Path
from collections import Counter
import json
import math
import csv

SHARDS_ROOT = Path("data/corpus/normalized/shards")
FREQ_ROOT = Path("data/freq")
FREQ_TSV = FREQ_ROOT / "word_frequencies.tsv"
FREQ_META = FREQ_ROOT / "word_frequencies.meta.json"

def compute_global_frequencies() -> None:
    """
    Walk all normalized shard files, count tokens globally,
    and write:

      - data/freq/word_frequencies.tsv
      - data/freq/word_frequencies.meta.json

    TSV columns:
      token, count, freq_per_5m, log_freq_per_5m, rank
    """
    ...
```

Implementation details:

1. Initialize `Counter[str]`.

2. For each `*.jsonl` in `SHARDS_ROOT`:

   * For each line:

     * `record = json.loads(line)`.
     * `tokens = record["tokens"]`.
     * `counter.update(tokens)`.

3. After counting:

   * `total_tokens = sum(counter.values())`.
   * Sort `(token, count)` by `count` descending to assign `rank`.
   * For each token:

     * `freq_per_5m = count * 5_000_000.0 / total_tokens`.
     * `log_freq_per_5m = math.log(freq_per_5m + epsilon)`, choose `epsilon` small (e.g., `1e-12`).

4. Write `word_frequencies.tsv` with header:

   ```text
   token    count    freq_per_5m    log_freq_per_5m    rank
   ```

5. Write `word_frequencies.meta.json` with:

   ```json
   {
     "version": "2025-11-17",
     "total_tokens": 123456789,
     "num_types": 987654,
     "source_meta_path": "data/meta/corpus_sources.json",
     "notes": "Computed from normalized shards; frequencies are per 5M tokens for Lexile-style MLF."
   }
   ```

---

## VII. Frequency Loader (Analyzer-side)

Create:

```python
# src/lexile_corpus_tuner/frequency_loader.py

from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import csv
import math

@dataclass
class WordFrequency:
    count: int
    freq_per_5m: float
    log_freq_per_5m: float
    rank: int

def load_frequency_table(path: Path | None = None) -> Dict[str, WordFrequency]:
    """
    Load word_frequencies.tsv into a dict keyed by token.
    """
    if path is None:
        path = Path("data/freq/word_frequencies.tsv")
    table: Dict[str, WordFrequency] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            token = row["token"]
            table[token] = WordFrequency(
                count=int(row["count"]),
                freq_per_5m=float(row["freq_per_5m"]),
                log_freq_per_5m=float(row["log_freq_per_5m"]),
                rank=int(row["rank"]),
            )
    return table
```

You can extend this later with caching or more sophisticated data structures.

---

## VIII. Lexile-Style Analyzer

Create:

```text
src/lexile_corpus_tuner/analyzer/
  __init__.py
  slices.py
  features.py
  model.py
  adjustments.py
  cli.py
```

### 8.1 Slices: Sentence Segmentation and Slice Construction

#### 8.1.1 Sentence segmentation

In `slices.py`:

```python
# src/lexile_corpus_tuner/analyzer/slices.py

from dataclasses import dataclass
from typing import List
import re

from ..textutils import normalize_text, iter_tokens

SENTENCE_END_RE = re.compile(r'([.?!;])')

@dataclass
class Slice:
    slice_id: int
    text: str
    tokens: list[str]
    sentence_lengths: list[int]  # tokens per sentence in this slice

def split_into_sentences(text: str) -> List[str]:
    """
    Basic rule-based sentence segmentation.

    - Normalize whitespace but preserve original punctuation.
    - Split on '.', '?', '!', or ';' while keeping punctuation with the sentence.
    - Keep the segmentation deterministic and simple.
    """
    # Codex: implement a reasonable heuristic:
    # - iterate over characters, accumulate until you hit [.!?;]
    # - treat that as end of sentence, including the punctuation.
    ...
```

Heuristics can be improved later, but must remain deterministic.

#### 8.1.2 Slice construction (≥125 words, extended to sentence boundary)

```python
def build_slices(text: str, min_words: int = 125) -> list[Slice]:
    """
    Build Lexile-style slices of >= min_words, extended to sentence boundary.

    Algorithm:
    - Split text into sentences with split_into_sentences.
    - For each sentence:
      - Tokenize using iter_tokens(sentence).
    - Append sentences (and their tokens) to current slice until the slice
      has at least min_words tokens.
    - Once >= min_words is reached, keep adding tokens until the end
      of the current sentence; then close the slice and start a new one.
    - If the entire text has fewer than min_words tokens, return a single
      slice containing all tokens.
    """
    ...
```

Implementation outline:

1. `sentences = split_into_sentences(text)`.

2. Initialize accumulators:

   ```python
   slices: list[Slice] = []
   slice_id = 0
   current_tokens: list[str] = []
   current_sentence_lengths: list[int] = []
   current_text_parts: list[str] = []
   ```

3. For each sentence in `sentences`:

   * `sent_tokens = list(iter_tokens(sentence))`.
   * If `not sent_tokens`, skip (empty).
   * Append `sentence` to `current_text_parts`.
   * Extend `current_tokens` with `sent_tokens`.
   * Append `len(sent_tokens)` to `current_sentence_lengths`.
   * If `len(current_tokens) >= min_words`:

     * Create `slice_text = " ".join(current_text_parts)` (or keep original).
     * Create `Slice(slice_id, slice_text, list(current_tokens), list(current_sentence_lengths))`.
     * Append to `slices`.
     * Increment `slice_id`.
     * Reset `current_tokens`, `current_sentence_lengths`, `current_text_parts` to empty.

4. After iteration:

   * If `current_tokens` not empty, emit final slice with remaining tokens.
   * Edge case: if no sentences => treat whole text as one "sentence".

Return `slices`.

---

### 8.2 Features: MSL & MLF

#### 8.2.1 Data classes

In `features.py`:

```python
# src/lexile_corpus_tuner/analyzer/features.py

from dataclasses import dataclass
from typing import List, Dict
import statistics

from .slices import Slice
from ..frequency_loader import load_frequency_table, WordFrequency

@dataclass
class SliceFeatures:
    slice_id: int
    num_tokens: int
    num_sentences: int
    mean_sentence_length: float
    mean_log_word_freq: float

@dataclass
class DocumentFeatures:
    num_slices: int
    total_tokens: int
    overall_mean_sentence_length: float
    overall_mean_log_word_freq: float
    slice_features: List[SliceFeatures]
```

#### 8.2.2 Computing slice and document-level features

```python
def compute_document_features(slices: list[Slice]) -> DocumentFeatures:
    """
    Compute Lexile-style features for a list of slices.

    - Slice-level:
      - num_tokens, num_sentences, mean_sentence_length, mean_log_word_freq.
    - Document-level:
      - num_slices, total_tokens,
      - overall_mean_sentence_length,
      - overall_mean_log_word_freq.
    """
    freq_table = load_frequency_table()

    # Precompute an "unseen" log frequency floor:
    # e.g., global minimum log_freq_per_5m minus 1.0
    if freq_table:
        min_log_freq = min(w.log_freq_per_5m for w in freq_table.values())
    else:
        min_log_freq = -20.0
    unseen_log_freq = min_log_freq - 1.0

    slice_feats: list[SliceFeatures] = []
    total_tokens = 0
    all_sentence_lengths: list[int] = []
    all_token_log_freqs: list[float] = []

    for sl in slices:
        num_tokens = len(sl.tokens)
        num_sentences = len(sl.sentence_lengths)
        total_tokens += num_tokens

        # MSL per slice
        if num_sentences > 0:
            msl = statistics.mean(sl.sentence_lengths)
        else:
            msl = float(num_tokens)  # all tokens in one "sentence"

        # MLF per slice
        token_log_freqs: list[float] = []
        for t in sl.tokens:
            wf = freq_table.get(t)
            if wf is None:
                token_log_freqs.append(unseen_log_freq)
            else:
                token_log_freqs.append(wf.log_freq_per_5m)

        if token_log_freqs:
            mlf = statistics.mean(token_log_freqs)
        else:
            mlf = unseen_log_freq

        slice_feats.append(
            SliceFeatures(
                slice_id=sl.slice_id,
                num_tokens=num_tokens,
                num_sentences=num_sentences,
                mean_sentence_length=msl,
                mean_log_word_freq=mlf,
            )
        )

        all_sentence_lengths.extend(sl.sentence_lengths or [num_tokens])
        all_token_log_freqs.extend(token_log_freqs)

    num_slices = len(slice_feats)
    if all_sentence_lengths:
        overall_msl = statistics.mean(all_sentence_lengths)
    else:
        overall_msl = 0.0

    if all_token_log_freqs:
        overall_mlf = statistics.mean(all_token_log_freqs)
    else:
        overall_mlf = unseen_log_freq

    return DocumentFeatures(
        num_slices=num_slices,
        total_tokens=total_tokens,
        overall_mean_sentence_length=overall_msl,
        overall_mean_log_word_freq=overall_mlf,
        slice_features=slice_feats,
    )
```

---

## IX. Special-Case Adjustments

Create:

```python
# src/lexile_corpus_tuner/analyzer/adjustments.py

def adjust_for_special_cases(
    raw_lexile: float,
    *,
    is_picture_book: bool = False,
    is_emergent_nonfiction: bool = False,
) -> float:
    """
    Apply Lexile-style special-case adjustments.

    - Picture books with direct picture support: -120L
    - Nonfiction emergent reader texts with fewer than 500 words: -120L

    These flags should initially be provided explicitly (via CLI or metadata).
    """
    adjustment = 0.0

    if is_picture_book:
        adjustment -= 120.0

    if is_emergent_nonfiction:
        adjustment -= 120.0

    return raw_lexile + adjustment
```

For now, pass flags via CLI options (`--picture-book`, `--emergent-nonfiction`) or via metadata in higher-level tooling.

---

## X. Regression Calibration Against Official Lexile Measures

The analyzer yields **DocumentFeatures** (MSL, MLF, etc.). To convert these into a Lexile-like number, we fit a regression model using **official Lexile measures from known texts** as targets, *not* `lexile-determination-v2`.

Instead of treating `lexile-determination-v2` as a teacher, we:

1. Build a **calibration catalog** of texts that have:
   - A locally available full text (or large representative excerpt).
   - A published Lexile measure from MetaMetrics or another trusted Lexile provider.
2. Automatically fetch or locate the corresponding text files.
3. Run our analyzer on those texts to compute features.
4. Fit the regression to map features → official Lexile values.

### 10.1 Calibration data model

Calibration data lives under:

```text
data/calibration/
  texts/
    <text-id>.txt              # raw text used for analysis
  catalog/
    lexile_catalog.csv         # master list of calibration items
  calibration_dataset.parquet  # or .csv (features + targets)
````

#### 10.1.1 Calibration catalog (`lexile_catalog.csv`)

`lexile_catalog.csv` is the **single source of truth** for calibration items. Each row represents one text with a known Lexile measure.

Columns (suggested):

* `text_id` (string) – stable, file-system–safe identifier (e.g., `hp1_sample_01`, `gutenberg_1342`, `openstax_physics_ch1`).
* `title` (string) – human-readable title.
* `author` (string) – optional, for reference.
* `lexile_official` (float) – the published Lexile measure for this text (e.g., `880`, `350`, `BR` converted to numeric scheme as needed).
* `grade_band` (string) – optional, e.g. `K-2`, `3-5`, `6-8`, `9-12`.
* `source_url` (string) – where the Lexile measure came from (Lexile “Find a Book,” district list, etc.).
* `lexile_source` (string) – short code for the source of the Lexile number, e.g.:

  * `meta_metrics_official`
  * `district_list_csv`
  * `manual_entry`
* `acquisition_type` (string) – how to acquire the text content:

  * `local` – file is already present as `data/calibration/texts/<text-id>.txt`.
  * `gutenberg` – fetch from Project Gutenberg (public domain).
  * `http` – fetch plain text from a given URL (must be legally permitted).
  * `manual` – human will place the file manually; tool only checks for presence.
* `acquisition_key` (string) – key needed based on `acquisition_type`:

  * For `gutenberg`: ebook ID (`1342`, `2701`, etc.).
  * For `http`: direct URL to a plain-text resource.
  * For `local`/`manual`: optional path or notes.
* `license` (string) – short indication of legal status (e.g., `public_domain`, `oer`, `all_rights_reserved`).
* `notes` (string) – free-form notes (edition, excerpt length, etc.).

This catalog can be assembled from multiple human- or machine-assisted sources:

* Exported CSVs of school reading lists that include Lexile measures.
* Manual curation of known public-domain works that also have Lexile scores.
* OER providers that tag content with Lexile levels.

#### 10.1.2 Text files (`data/calibration/texts`)

For each row in `lexile_catalog.csv`, we expect a corresponding **text file**:

```text
data/calibration/texts/<text-id>.txt
```

* For `acquisition_type=local` or `manual`, a human is responsible for placing the text file there.
* For `acquisition_type=gutenberg` or `http`, we will provide an automated fetch command (see §10.2).

Text must be:

* A full book, full chapter, or sufficiently large excerpt (preferably ≥ 500 words, ideally much longer).
* Plain UTF-8 text (no HTML markup; if downloaded as HTML, a preprocessing step should strip tags).

#### 10.1.3 Calibration dataset (`calibration_dataset.parquet`)

The **calibration dataset** is a feature table derived from `lexile_catalog.csv` and the corresponding text files.

Each row in `calibration_dataset` should contain:

* `text_id`: copied from catalog.
* `title`, `author`, `grade_band`, `lexile_source`: optional, for analysis and diagnostics.
* `lexile_official`: numeric target (official Lexile measure).
* `num_tokens`: from `DocumentFeatures`.
* `num_slices`: from `DocumentFeatures`.
* `overall_msl`: `DocumentFeatures.overall_mean_sentence_length`.
* `overall_mlf`: `DocumentFeatures.overall_mean_log_word_freq`.
* All engineered features from `make_regression_features` (see §10.3).

This dataset is the direct input to the regression training step.

---

### 10.2 Calibration dataset builder and automated aggregation (CLI)

We introduce **two** CLI commands for calibration data management:

1. `calibration fetch-texts` – ensures that all text files exist for rows in `lexile_catalog.csv`.
2. `calibration build-dataset` – analyzes those texts to build `calibration_dataset.parquet`.

#### 10.2.1 `calibration fetch-texts`

CLI spec:

```bash
lexile-corpus-tuner calibration fetch-texts \
  --catalog data/calibration/catalog/lexile_catalog.csv \
  --texts-root data/calibration/texts
```

Implementation outline:

1. Read `lexile_catalog.csv`.
2. For each row:

   * Compute expected path: `texts_root / f"{text_id}.txt"`.
   * If the file already exists, skip.
   * Otherwise, act based on `acquisition_type`:

     * `local`:

       * Log a warning that the file is missing and must be created manually.
     * `manual`:

       * Same: warn; do not attempt automated fetch.
     * `gutenberg`:

       * Use `acquisition_key` as Gutenberg ID.
       * Download the plain-text version from Project Gutenberg.
       * Strip any headers/footers if desired.
       * Save to `<text-id>.txt`.
     * `http`:

       * Fetch content from `acquisition_key` (URL).
       * If HTML, strip tags; if already text, use as-is.
       * Save to `<text-id>.txt`.
3. At the end, emit a summary:

   * Number of texts successfully fetched.
   * Number of texts still missing (due to `local`/`manual` or fetch failure).

This provides **automated aggregation** of calibration points for public-domain/OER resources and any plain-text URLs you legally control or have permission to use.

#### 10.2.2 `calibration build-dataset`

CLI spec (similar to before, but now driven by the catalog):

```bash
lexile-corpus-tuner calibration build-dataset \
  --catalog data/calibration/catalog/lexile_catalog.csv \
  --texts-root data/calibration/texts \
  --output data/calibration/calibration_dataset.parquet
```

Implementation outline:

1. Read `lexile_catalog.csv` into memory.

2. For each row:

   * Derive `text_path = texts_root / f"{text_id}.txt"`.
   * If `text_path` doesn’t exist, log a warning and **skip** that row (or fail, depending on a `--strict` flag).
   * Read text (UTF-8).
   * Build slices: `slices = build_slices(text)`.
   * Compute features: `doc_features = compute_document_features(slices)`.
   * Use `make_regression_features(doc_features)` to compute the engineered feature dict.
   * Construct a row with:

     * Metadata from catalog (`text_id`, `title`, `author`, `grade_band`, `lexile_official`, `lexile_source`, etc.).
     * Base features: `num_tokens`, `num_slices`, `overall_msl`, `overall_mlf`.
     * Engineered features from the feature dict.

3. Append each row into a list and write as:

   * Parquet (`.parquet`) using pandas, or
   * CSV (`.csv`) if Parquet not desired.

This command is the **automated aggregation step** that:

* Combines human-curated Lexile values,
* Automatically fetched or locally provided texts,
* And analyzer-derived features,

into a single, ready-to-train dataset.

---

### 10.3 Feature engineering (regression features)

*(Structure unchanged, but `lexile_teacher` is now conceptually `lexile_official` in the dataset; the function itself remains the same.)*

Create:

```text
src/lexile_corpus_tuner/calibration/
  __init__.py
  featureset.py
  train.py
  cli.py
  model_store.py
```

In `featureset.py`:

```python
# src/lexile_corpus_tuner/calibration/featureset.py

from ..analyzer.features import DocumentFeatures
import math
import statistics

def make_regression_features(doc: DocumentFeatures) -> dict[str, float]:
    """
    Map DocumentFeatures (+ per-slice details) into a flat feature dict
    suitable for regression training and inference.
    """

    # base
    f = {
        "overall_msl": doc.overall_mean_sentence_length,
        "overall_mlf": doc.overall_mean_log_word_freq,
        "num_tokens": float(doc.total_tokens),
        "num_slices": float(doc.num_slices),
    }

    # derived
    f["log_num_tokens"] = math.log(max(1.0, doc.total_tokens))

    # slice variability
    msl_values = [s.mean_sentence_length for s in doc.slice_features]
    mlf_values = [s.mean_log_word_freq for s in doc.slice_features]

    f["msl_std"] = statistics.pstdev(msl_values) if len(msl_values) > 1 else 0.0
    f["mlf_std"] = statistics.pstdev(mlf_values) if len(mlf_values) > 1 else 0.0

    # polynomial / interaction terms
    f["overall_msl_sq"] = f["overall_msl"] ** 2
    f["overall_mlf_sq"] = f["overall_mlf"] ** 2
    f["msl_times_mlf"] = f["overall_msl"] * f["overall_mlf"]

    return f
```
The calibration dataset builder should call this function so **train and inference share the exact same feature definition**.


## XI. Regression Training Pipeline

### 11.1 Base model choice

Use a linear model with mild regularization:

* **ElasticNet** with `l1_ratio=0.0` (i.e., Ridge / L2) to start.
* Reason:

  * Lexile spec equation is linear in its core features.
  * You add a few non-linear transformations and variability measures.
  * Regularization prevents overfitting.

### 11.2 Training module

In `train.py`:

```python
# src/lexile_corpus_tuner/calibration/train.py

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.stats import pearsonr

FEATURE_COLS = [
    "overall_msl",
    "overall_mlf",
    "log_num_tokens",
    "msl_std",
    "mlf_std",
    "overall_msl_sq",
    "overall_mlf_sq",
    "msl_times_mlf",
]

TARGET_COL = "lexile_teacher"

def compute_metrics(y_true, y_pred) -> dict:
    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    r, _ = pearsonr(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r": float(r)}

def train_regression_model(df: pd.DataFrame) -> tuple[ElasticNet, dict]:
    """
    Train an ElasticNet regression model on calibration data.

    - Splits into train/validation.
    - Returns the fitted model and validation metrics.
    """

    # Filter low-quality rows:
    df = df.dropna(subset=[TARGET_COL, "overall_msl", "overall_mlf"])
    df = df[df["num_tokens"] >= 100]
    df = df[df["num_slices"] >= 1]

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[TARGET_COL].values

    model = ElasticNet(
        alpha=0.001,     # light regularization
        l1_ratio=0.0,    # pure ridge to start; tweak later if needed
        fit_intercept=True,
        max_iter=10000,
    )
    model.fit(X_train, y_train)

    yhat_val = model.predict(X_val)
    metrics = compute_metrics(y_val, yhat_val)

    return model, metrics
```

Target metrics (as a rough goal):

* RMSE ≤ ~100L.
* MAE as low as possible.
* Pearson r ≥ ~0.95.

You can add K-fold CV later to refine `alpha` and `l1_ratio`.

---

## XII. Model Storage & Runtime Inference

### 12.1 JSON model spec and model_store

You don’t want scikit-learn in the runtime analyzer; instead, export a JSON spec containing:

* Feature list (names and order).
* Coefficients.
* Intercept.
* Training metrics.

Example JSON schema (stored at `data/model/lexile_regression_model.json`):

```json
{
  "version": "2025-11-18",
  "features": [
    "overall_msl",
    "overall_mlf",
    "log_num_tokens",
    "msl_std",
    "mlf_std",
    "overall_msl_sq",
    "overall_mlf_sq",
    "msl_times_mlf"
  ],
  "coefficients": [123.4, -210.2, 15.0, 5.2, -3.3, 0.8, -1.1, 2.5],
  "intercept": 450.0,
  "metrics": {
    "rmse_val": 72.3,
    "mae_val": 55.1,
    "r_val": 0.97,
    "num_texts": 500
  },
  "notes": "Trained against lexile-determination-v2 teacher scores using proxy corpus v2025-11-17."
}
```

Implement:

```python
# src/lexile_corpus_tuner/calibration/model_store.py

import json
from pathlib import Path
from sklearn.linear_model import ElasticNet

def save_model(model: ElasticNet, metrics: dict, feature_names: list[str], path: Path) -> None:
    spec = {
        "version": "2025-11-18",
        "features": feature_names,
        "coefficients": list(map(float, model.coef_)),
        "intercept": float(model.intercept_),
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

def load_model_spec(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
```

### 12.2 Runtime `estimate_lexile_from_features`

In `analyzer/model.py`:

```python
# src/lexile_corpus_tuner/analyzer/model.py

from pathlib import Path
from functools import lru_cache
from ..calibration.featureset import make_regression_features
from ..calibration.model_store import load_model_spec
from .features import DocumentFeatures

MODEL_PATH = Path("data/model/lexile_regression_model.json")

@lru_cache(maxsize=1)
def _load_model():
    spec = load_model_spec(MODEL_PATH)
    return spec

def estimate_lexile_from_features(features: DocumentFeatures) -> float:
    """
    Estimate a Lexile-like measure from Lexile-style features,
    using a stored linear regression model spec.
    """
    spec = _load_model()
    feat_dict = make_regression_features(features)

    x = []
    for name in spec["features"]:
        x.append(float(feat_dict.get(name, 0.0)))

    intercept = float(spec["intercept"])
    coefs = list(map(float, spec["coefficients"]))

    y_hat = intercept
    for c, xi in zip(coefs, x):
        y_hat += c * xi

    return float(y_hat)
```

At runtime, your pipeline will:

1. Build slices.
2. Compute DocumentFeatures.
3. Call `estimate_lexile_from_features`.
4. Optionally apply `adjust_for_special_cases`.

---

## XIII. Calibration CLI

Create `calibration/cli.py`:

```python
# src/lexile_corpus_tuner/calibration/cli.py

import click
import pandas as pd
from pathlib import Path
from .train import train_regression_model, FEATURE_COLS
from .model_store import save_model

@click.group(name="calibration")
def calibration_group() -> None:
    """Commands for Lexile regression calibration."""

@calibration_group.command("fit")
@click.argument("dataset", type=click.Path(exists=True))
@click.option("--out", type=click.Path(), default="data/model/lexile_regression_model.json")
def fit(dataset: str, out: str) -> None:
    """
    Fit regression model on a calibration dataset and save JSON spec.
    """
    df = pd.read_parquet(dataset)  # or read_csv
    model, metrics = train_regression_model(df)
    save_model(model, metrics, FEATURE_COLS, Path(out))
    click.echo(f"Saved model to {out}")
    click.echo(
        f"Validation RMSE: {metrics['rmse']:.1f}L, "
        f"MAE: {metrics['mae']:.1f}L, r: {metrics['r']:.3f}"
    )
```

Wire it into your main CLI:

```python
# src/lexile_corpus_tuner/cli.py

import click
from .corpus.cli import corpus_group
from .analyzer.cli import analyze_group
from .calibration.cli import calibration_group

@click.group()
def main() -> None:
    """Main entry point for lexile-corpus-tuner."""
    ...

main.add_command(corpus_group)
main.add_command(analyze_group)
main.add_command(calibration_group)
```

---

## XIV. Analyzer CLI

In `analyzer/cli.py`:

```python
# src/lexile_corpus_tuner/analyzer/cli.py

import click
from pathlib import Path
import json

from .slices import build_slices
from .features import compute_document_features
from .model import estimate_lexile_from_features
from .adjustments import adjust_for_special_cases

@click.group(name="analyze")
def analyze_group() -> None:
    """Lexile-style text analysis commands."""

@analyze_group.command("text")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--json-output", type=click.Path(), default=None)
@click.option("--picture-book", is_flag=True, default=False, help="Apply picture book adjustment (-120L).")
@click.option("--emergent-nonfiction", is_flag=True, default=False, help="Apply emergent nonfiction adjustment (-120L).")
def analyze_text(input_file: str, json_output: str | None, picture_book: bool, emergent_nonfiction: bool) -> None:
    """
    Analyze a text file and print Lexile-style features + estimated score.
    """
    path = Path(input_file)
    text = path.read_text(encoding="utf-8")

    slices = build_slices(text)
    doc_features = compute_document_features(slices)
    raw_lexile = estimate_lexile_from_features(doc_features)
    adjusted_lexile = adjust_for_special_cases(
        raw_lexile,
        is_picture_book=picture_book,
        is_emergent_nonfiction=emergent_nonfiction,
    )

    click.echo(f"File: {input_file}")
    click.echo(f"Num slices: {doc_features.num_slices}")
    click.echo(f"Total tokens: {doc_features.total_tokens}")
    click.echo(f"Overall MSL: {doc_features.overall_mean_sentence_length:.2f}")
    click.echo(f"Overall MLF: {doc_features.overall_mean_log_word_freq:.4f}")
    click.echo(f"Estimated Lexile (raw): {raw_lexile:.1f}L")
    click.echo(f"Estimated Lexile (adjusted): {adjusted_lexile:.1f}L")

    if json_output is not None:
        out_data = {
            "file": str(path),
            "raw_lexile": raw_lexile,
            "adjusted_lexile": adjusted_lexile,
            "document_features": {
                "num_slices": doc_features.num_slices,
                "total_tokens": doc_features.total_tokens,
                "overall_msl": doc_features.overall_mean_sentence_length,
                "overall_mlf": doc_features.overall_mean_log_word_freq,
            },
            "slice_features": [
                {
                    "slice_id": sf.slice_id,
                    "num_tokens": sf.num_tokens,
                    "num_sentences": sf.num_sentences,
                    "mean_sentence_length": sf.mean_sentence_length,
                    "mean_log_word_freq": sf.mean_log_word_freq,
                }
                for sf in doc_features.slice_features
            ],
        }
        Path(json_output).write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        click.echo(f"Wrote detailed JSON to {json_output}")
```

---

## XV. Corpus CLI

In `corpus/cli.py`:

```python
# src/lexile_corpus_tuner/corpus/cli.py

import click
from . import download, normalize, frequencies

@click.group(name="corpus")
def corpus_group() -> None:
    """Commands for managing the proxy Lexile corpus."""

@corpus_group.command("download")
@click.option("--gutenberg-limit", type=int, default=None, help="Max Gutenberg books to download")
def corpus_download(gutenberg_limit: int | None) -> None:
    """Download raw corpus sources."""
    download.ensure_dirs()
    download.download_gutenberg_subset(limit=gutenberg_limit)
    download.download_simple_wiki_dump()

@corpus_group.command("normalize")
@click.option("--shard-size-tokens", type=int, default=100_000)
def corpus_normalize(shard_size_tokens: int) -> None:
    """Normalize and tokenize raw corpora into shards."""
    normalize.normalize_all_sources(shard_size_tokens=shard_size_tokens)

@corpus_group.command("frequencies")
def corpus_frequencies() -> None:
    """Compute global word frequencies from normalized shards."""
    frequencies.compute_global_frequencies()
```

---

## XVI. Guardrails & Diagnostics

### 16.1 Data quality for calibration

Before or during training:

* Drop texts with:

  * `num_tokens < 100` (too short).
  * `num_slices == 0` (parsing failure).
  * missing `lexile_teacher`.
* Log how many rows are dropped and why.

### 16.2 Diagnostics after training

After fitting:

* Compute and log:

  * RMSE, MAE, Pearson r.
  * MAE and RMSE per Lexile band (e.g. `[BR–500L]`, `[500–800L]`, `[800–1000L]`, `[1000–1400L]`).
* Inspect coefficients:

  * Expect coefficient for `overall_msl` to be **positive** (longer sentences → harder).
  * Expect coefficient for `overall_mlf` to be **negative** (higher average log frequency → easier).
* Plot (for your own inspection):

  * Residuals vs `lexile_teacher` to check for systematic errors at extremes.
* If residuals balloon at very low/high Lexiles:

  * Consider more non-linear terms (already included, e.g., `overall_mlf_sq`).
  * Or in extreme cases, piecewise modeling (low vs high band) – but keep it as simple as possible.

---

## XVII. End-to-End Workflows

### 17.1 Build or update corpus

```bash
# 1) Download sources
lexile-corpus-tuner corpus download --gutenberg-limit 200   # example limit

# 2) Normalize and shard
lexile-corpus-tuner corpus normalize --shard-size-tokens 100000

# 3) Compute frequency table
lexile-corpus-tuner corpus frequencies
```

This produces:

* `data/freq/word_frequencies.tsv`
* `data/freq/word_frequencies.meta.json`

### 17.2 Build calibration dataset

1. Prepare calibration texts:

   * Place candidate texts in `data/calibration/texts/` as `<text-id>.txt`.
   * Ensure they cover a broad Lexile range and genres.

2. Build dataset (your `build-dataset` command should:

   * For each text:

     * Run `build_slices` + `compute_document_features`.
     * `make_regression_features`.
     * Call `lexile-determination-v2` to get `lexile_teacher`.
   * Save to `calibration_dataset.parquet`):

   ```bash
   lexile-corpus-tuner calibration build-dataset \
     --texts-root data/calibration/texts \
     --output data/calibration/calibration_dataset.parquet
   ```

### 17.3 Fit regression model

```bash
lexile-corpus-tuner calibration fit \
  data/calibration/calibration_dataset.parquet \
  --out data/model/lexile_regression_model.json
```

This:

* Trains an ElasticNet model.
* Evaluates on a validation holdout.
* Saves the JSON model spec with coefficients and metrics.

### 17.4 Analyze new texts

```bash
lexile-corpus-tuner analyze text path/to/new_text.txt
```

Runtime steps:

1. `build_slices(new_text)` → slices.
2. `compute_document_features(slices)` → DocumentFeatures.
3. `estimate_lexile_from_features(DocumentFeatures)` → raw Lexile-like estimate.
4. Optionally `adjust_for_special_cases` (e.g., `--picture-book`, `--emergent-nonfiction`).
5. Print / export results.