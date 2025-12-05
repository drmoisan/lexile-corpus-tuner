# `query-builder-ui` — User Story

- Issue: #9
- Owner: drmoisan
- Status: Complete
- Last Updated: 2025-12-04

## Story Statement

- As a **corpus curator**, I want a **visual query builder for Gutenberg metadata**, so that I can **easily construct complex queries without memorizing CLI syntax**.
- As a **machine learning researcher**, I want to **save and reuse query definitions**, so that I can **reproduce corpus selection criteria across experiments**.
- As a **text analysis practitioner**, I want to **preview query results before export**, so that I can **verify the corpus matches my requirements**.

## Problem / Why

The existing Gutenberg metadata explorer CLI requires users to manually construct complex query strings with proper syntax for boolean operators, ranges, and nested logic. This is error-prone and has a steep learning curve. Users need a visual, interactive way to build queries without memorizing syntax, especially for complex nested queries with multiple fields and operators.

Curating a high-quality corpus for training text difficulty models requires:
- Filtering by multiple metadata fields (subjects, authors, languages, download counts)
- Combining criteria with complex boolean logic (AND/OR with nesting)
- Iterating on queries to refine corpus selection
- Sharing and reproducing query definitions across team members

Manually writing query strings like `(subject:Fiction OR subject:Science) AND download_count>1000 AND language:en` is tedious and error-prone.

## Personas & Scenarios

### Persona: Research Data Curator (Sarah)
- **Who**: Graduate student building a training corpus for a Lexile measurement pipeline
- **Cares about**: High-quality, representative texts that match specific criteria; reproducible selection process
- **Constraints**: Limited time; needs to document corpus selection decisions; must share queries with advisor
- **Goals**: Select 500+ English fiction books with >1000 downloads from specific subjects
- **Frustrations**: CLI syntax is hard to remember; mistakes in query strings waste time; hard to experiment with different filters
- **Context**: Working on thesis; needs to train multiple model variants; must document methodology

### Scenario 1: Building a Fiction Corpus for Young Readers
**Who**: Sarah (research data curator)  
**Trigger**: Advisor asks for a corpus of popular fiction books appropriate for young adult readers

**Steps**:
1. Sarah launches the Query Builder UI from VS Code
2. She clicks "subject" in the field palette, which adds a constraint to the query builder
3. She opens the multi-select dialog for subjects, searches for "fiction", and selects "Children's Literature" and "Young Adult Fiction"
4. She adds another constraint by clicking "download_count" and selecting the ">" operator with value "1000" (to ensure popularity)
5. She adds a language constraint for "en" (English only)
6. She groups the subject constraints with OR logic and the rest with AND
7. She clicks "Run Query" (F5) and sees 347 matching books in the preview panel
8. She reviews the results (titles, authors, download counts) to verify they match expectations
9. She saves the query as "young-adult-fiction-popular.json" (Ctrl+S) for documentation
10. She exports the results to "young-adult-corpus.parquet" (Ctrl+E) for pipeline ingestion

**Outcome**: Sarah has a reproducible query definition and a validated corpus file in 5 minutes, instead of 30+ minutes of trial-and-error with CLI syntax.

### Scenario 2: Refining a Corpus with Complex Filters
**Who**: Sarah (research data curator)  
**Trigger**: Initial corpus is too large; needs to narrow by excluding certain bookshelves and adding date constraints

**Steps**:
1. Sarah opens her previously saved query "young-adult-fiction-popular.json" (Ctrl+O)
2. The GUI recreates all constraints and groups from the saved file
3. She adds a new group with OR logic for excluding specific bookshelves
4. She clicks "bookshelf" in the palette, opens the multi-select dialog, and selects "Harvard Classics" and "Best Books Ever Listings" to exclude
5. She changes the operator to "!=" (not equals) to exclude these bookshelves
6. She nests this new exclusion group under the main AND group
7. The query string display updates in real-time showing the parentheses and logic
8. She runs the query and sees the count drop from 347 to 298 books
9. She reviews a few sample titles to confirm the filters work as expected
10. She saves the refined query as "young-adult-fiction-curated-v2.json" (Ctrl+Shift+S)

**Outcome**: Sarah iteratively refines her corpus using visual feedback and saves versioned query definitions for her methodology documentation.

## Acceptance Criteria

- [x] GUI launches and displays main window with field palette, query builder, and results panels
- [x] Users can add constraints by clicking fields in palette
- [x] Users can select appropriate operators based on field type (text: contains/=, numeric: >/</>=/<=/range, boolean: =/!=)
- [x] Users can group constraints with AND/OR logic
- [x] Users can nest groups to arbitrary depth
- [x] Query string display updates in real-time as query is built
- [x] Users can execute queries and see results in preview panel (first 100 rows)
- [x] Users can save queries to JSON files with versioning metadata
- [x] Users can load previously saved queries and recreate UI state
- [x] Users can export results to CSV or Parquet formats
- [x] Multi-select dialogs for subjects/bookshelves with search/filter, Select All/Clear All
- [x] Keyboard shortcuts work (F5: Run Query, Ctrl+N/O/S/Shift+S/E/Q)
- [x] Tooltips appear on hover for all interactive elements
- [x] Status bar provides feedback during operations
- [x] Confirm dialogs prevent accidental data loss
- [x] All code passes Black, Ruff, Pyright, and Pytest checks (24 tests passing)

## Non-Goals

- **Advanced text search**: No full-text search within book contents (metadata only)
- **Direct book download**: No integration with Gutenberg download APIs (use separate corpus download tool)
- **Query optimization**: No automatic query rewriting or performance tuning
- **Collaborative editing**: No multi-user or real-time collaboration features
- **Query versioning**: No git-like diff/merge for query files (use external version control)
- **Custom fields**: No support for user-defined metadata fields (only Gutenberg schema)
- **Drag-and-drop reordering**: Constraints cannot be reordered via drag-and-drop (delete and re-add instead)
- **Query history**: No undo/redo or automatic query history tracking
- **Scheduled queries**: No background execution or cron-like scheduling

