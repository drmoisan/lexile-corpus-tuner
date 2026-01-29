# corpus-builder-source-expansion (Issue #58)

- Date captured: 2025-12-19
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/corpus-builder-source-expansion/ (Issue #58)

- Issue: #58
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/58
- Last Updated: 2025-12-19
## Problem / Why

This will ensure that the corpus used for the lexile scoring is not improperly biased by the age of the gutenberg tests in the public domain. 

## Proposed Behavior

### 1. Goal & Overall Strategy

**Goal:** Build a ~multi-billion-word, multi-source English reading corpus that *roughly approximates* the MetaMetrics Lexile corpus in style and distribution, while staying within what you can realistically scrape or download (Gutenberg + open/CC sources).

**High-level strategy:**

1. Use **four source buckets**:

   * **S1 – Classic narrative (Project Gutenberg)**
   * **S2 – Modern expository/reference (Wikipedia / Simple Wikipedia)**
   * **S3 – Modern instructional / textbook-like (open K–12/OER)**
   * **S4 – Modern children’s / YA narrative (open fiction / CC / misc.)**
2. Tag every document with **source, genre, publication era, and difficulty band**.
3. Compute global word frequencies and feature stats using **source- and era-weights** so Gutenberg doesn’t dominate.
4. Use this corpus both for:

   * **Frequency tables** for your Lexile-style feature extraction.
   * **Calibration/teacher-model training**, where you have Lexile labels.

---

### 2. Source 1 — Project Gutenberg (Children’s + School Literature)

**Purpose:** Provide a large, high-quality base of narrative prose, especially at elementary–middle difficulty, while controlling for historical bias.

**Selection:**

* Use **Project Gutenberg** English texts with:

  * Shelves / subjects like: *Children’s literature*, *Juvenile fiction*, *Young adult*, *School stories*, *Readers*, *Fairy tales*.
  * Language = English.
* Exclude:

  * Poetry, drama/plays, collections of verse.
  * Extremely long 19th-century tomes that would massively skew token counts (either drop or downsample).
  * Obvious outliers (legal treatises, pure reference works, etc.).

**Cleaning & segmentation:**

* Strip front/back matter:

  * The PG license block.
  * Prefaces, editorial notes, catalogs.
* Normalize:

  * Unicode normalization.
  * Normalize whitespace; remove page numbers, headers, running footers.
* Segment into **“documents”**:

  * Either full books *or* fixed-size chunks (e.g. 1–3k word segments) with a stable chunking strategy (chapter boundaries if possible, otherwise word windows).

**Metadata on each doc/chunk:**

* `source = "gutenberg"`
* `gutenberg_id`
* `genre = "narrative"`
* `intended_audience` (child/YA/adult, inferred from shelf/subject)
* `publication_year` (approximate; use earliest publication date you can find)
* Estimated **difficulty band** (once your estimator exists).

---

### 3. Source 2 — Modern Expository / Reference (Wikipedia & Simple English)

**Purpose:** Stand in for modern expository texts (textbooks, reference books, informational articles), especially grades 3–12.

**Selection:**

* **Simple English Wikipedia**:

  * All main-namespace articles, or a filtered subset focused on kid-relevant domains (animals, geography, basic science, history).
* **English Wikipedia (standard)**:

  * A curated subset: shorter, general-audience articles that aren’t hyper-technical.
  * Filters such as:

    * Article length between some min and max (e.g. 300–3,000 words).
    * Exclude stubs, lists, disambiguation pages.
    * Prefer high-quality/“Good” or “Featured” articles in child-relevant categories.

**Cleaning & segmentation:**

* Strip markup, templates, infoboxes, references, and navigation junk.
* Keep only the **main expository prose** (no talk pages, no comments).
* Segment:

  * Article-level or section-level units (e.g. sections as documents).

**Metadata:**

* `source = "simple_wikipedia"` or `"wikipedia"`
* `genre = "expository"`
* `topic`, `categories`
* `last_revision_year` (modernity proxy)
* Difficulty band estimate (once available).

---

### 4. Source 3 — Modern Instructional / Textbook-Like (OER K–12)

**Purpose:** Approximate the Lexile corpus’ heavy use of textbooks and instructional materials without violating copyright.

**Typical choices (depending what you actually decided to use):**

* **Open-licensed K–12 materials**, e.g.:

  * CK-12, OpenStax (upper grades), state-released CC-licensed curriculum units.
  * US government education resources (NASA, NOAA, NIH “for kids”, etc.).
* Focus on:

  * Science, social studies, math word-problem narratives, health, etc.
  * Grades 2–12.

**Cleaning & segmentation:**

* Strip navigation, problem numbering if it breaks language flow, sidebars.
* Keep expository paragraphs and example passages.
* Segment into passages of ~200–1000 words as “documents”.

**Metadata:**

* `source = "oer_textbook"` (or similar per provider)
* `genre = "instructional" or "expository"`
* `grade_band` if provided (K-2, 3-5, 6-8, 9-12)
* `publication_year` or `edition_year`.

---

### 5. Source 4 — Modern Children’s / YA Narrative (Open/CC Fiction)

**Purpose:** Compensate for Gutenberg’s dated narrative style with more contemporary kids’ and YA fiction.

**What we likely had in mind (within legal constraints):**

* Modern **creative-commons fiction**, story collections, or **web fiction** that is:

  * Clearly licensed for re-use.
  * Aimed at children / YA readers, or at least general-audience narrative.
* Possible buckets:

  * CC-licensed stories from educational sites.
  * Modern CC anthologies / story archives.
  * Some fan-fiction / web fiction communities if licensing allows bulk use.

**Cleaning & segmentation:**

* Similar to Gutenberg:

  * Strip metadata, comments, navigation.
  * Segment into doc-chunks (1–3k words, or story-level).

**Metadata:**

* `source = "modern_fiction_cc"`
* `genre = "narrative"`
* `intended_audience` (child/YA/general)
* `publication_year` (based on site metadata or first posted date).

---

### 6. Handling Gutenberg’s Age Bias — Filters & Weighting

This is the part you explicitly called out: **“special filters and weighting in place to compensate for the fact that Gutenberg texts are older.”**

#### 6.1. Document-level weighting

When computing global word frequencies and training features, don’t treat all tokens equally. Use **per-source weights** (and optionally **per-decade weights**) so that older Gutenberg material doesn’t swamp more modern text.

For example, in your frequency aggregation:

```text
weight(source="gutenberg")          ≈ 0.3
weight(source="simple_wikipedia")   ≈ 0.2
weight(source="wikipedia")          ≈ 0.2
weight(source="oer_textbook")       ≈ 0.2
weight(source="modern_fiction_cc")  ≈ 0.1
```

Or more structurally:

* Target distribution by **era** (for narrative):

  * Pre-1950: ~20–25%
  * 1950–1989: ~25–30%
  * 1990–present: ~45–50%

Then:

* For each document, assign `doc_weight` = f(source, era, genre).
* When you build frequency tables, compute:

[
\text{freq}(w) = \frac{\sum_{d} \text{doc_weight}(d) \cdot \text{count}*d(w)}{\sum*{d} \text{doc_weight}(d) \cdot \text{length}(d)}
]

So Gutenberg can be large in raw tokens but limited in *effective* influence.

#### 6.2. Lexical & orthographic normalization

To reduce the “oldness” of Gutenberg texts:

* Normalize common spelling variants:

  * `colour` → `color`, `honour` → `honor`, etc. (if you choose to Americanize).
* Optionally maintain a mapping table for a small set of archaic forms → modern forms.
* Filter extremely archaic, rare words **out** of your “target vocabulary” when building difficulty features, or cap their influence.

#### 6.3. Genre and register controls

* Explicitly **exclude Gutenberg poetry/plays**.
* Prefer narrative prose that is at least somewhat aligned with modern sentence structures (books that *feel* readable to a modern child).
* For frequency tables used in your *children’s* difficulty estimator, filter to:

  * narrative for narrative comparison,
  * expository for expository comparison (separate frequency tables or genre-conditioned features if you want to be fancy).

---

### 7. Difficulty Banding and Labeling

Because you’re approximating the MetaMetrics corpus, we had an extra layer:

1. **If Lexile labels exist** (e.g., some OER or passages you can match to Lexile Find-a-Book / Text Analyzer outputs):

   * Store `lexile_true` on those documents.
   * Use them later for regression calibration against your feature set.

2. **Where Lexiles don’t exist** (Gutenberg, Wikipedia, etc.):

   * Initially, use a **teacher model** (the Text Analyzer or your own estimator) to assign `lexile_est`.
   * Optionally bucket into **bands** that mimic grade ranges:

     * Pre-K–1, 2–3, 4–5, 6–8, 9–12, post-secondary.
   * Ensure your overall corpus doesn’t massively over-represent e.g. high-Lexile content relative to K–8.

3. Store for each document:

   * `lexile_label_type = "true" | "estimated"`
   * `lexile_value` or `lexile_band`.

---

### 8. Practical Pipeline Sketch (Reconstruct)

What we probably sketched as the ETL pipeline:

1. **Source ingestion layer**

   * Module per source (Gutenberg, SimpleWiki, Wiki, OER, modern CC fiction).
   * Each produces raw text + metadata.

2. **Cleaning & normalization**

   * Strip markup, boilerplate, navigation.
   * Normalize spelling & Unicode.
   * Remove junk sections (TOCs, indexes, references lists).
   * Optionally language-ID filter for English.

3. **Segmentation**

   * Convert each source document into 1+ standardized “doc units” with:

     * `text`
     * `source`
     * `genre`
     * `approx_word_count`

4. **Metadata enrichment**

   * Add `publication_year`, `era_bucket`, `intended_audience`, `topic`.
   * Run your *current* difficulty estimator/teacher model to attach `lexile_est`.

5. **Weighting & sampling**

   * Compute `doc_weight` based on `source`, `era_bucket`, `genre`.
   * Build:

     * A **full corpus index** (every doc unit with metadata + weight).
     * Sub-corpora: narrative only, expository only, grade-band subsets, etc.

6. **Feature extraction**

   * Build **global frequency tables** using weighted counts (as above).
   * Compute per-document features: mean log word frequency, sentence length stats, type-token metrics, etc.

7. **Calibration corpus**

   * For the subset with true Lexiles:

     * Use their features + Lexiles to calibrate your regression model (your Lexile-faithful pipeline).
   * For others:

     * Use estimated Lexiles only for ranking/sorting, not for training, or treat them as noisy labels with lower weight.



## Acceptance Criteria (early draft)

- [ ] Criterion 1
- [ ] Criterion 2

## Constraints & Risks

List notable constraints (performance, compatibility, scope) or risks.

## Test Conditions to Consider

- [ ] Unit coverage areas
- [ ] Integration scenarios
- [ ] CLI/API examples

## Next Step

- [ ] Promote to GitHub issue (feature request template)
- [ ] Create `docs/features/active/corpus-builder-source-expansion/` folder from the template
