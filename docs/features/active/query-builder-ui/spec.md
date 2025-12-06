# query-builder-ui — Spec

- Issue: #9
- Owner: drmoisan
- Last Updated: 2025-12-04

## Overview

The Gutenberg Query Builder UI is a visual, desktop GUI application that enables users to construct complex metadata queries without memorizing CLI syntax. Built with Python's Tkinter (standard library), it provides an intuitive drag-and-drop interface for building queries, executing them against the Gutenberg metadata corpus, and exporting results.

The UI follows Model-View-Controller (MVC) architecture with clear separation between data models (`QueryConstraintModel`, `QueryGroupModel`, `SavedQuery`), widgets (`QueryConstraintWidget`, `QueryGroupWidget`), and the main application controller (`QueryBuilderApp`).

## Behavior

### Core Features

**Visual Query Construction**
- Click fields in the palette (left panel) to add constraints to the query builder
- Each constraint has: field dropdown, operator dropdown, value input, and delete button
- Operators are context-aware based on field type:
  - Text fields (title, authors, subjects, etc.): `contains`, `=`
  - Numeric fields (id, download_count): `=`, `>`, `<`, `>=`, `<=`, `range` (e.g., "100..500")
  - Boolean fields (copyright): `=`, `!=`
- Multi-select dialogs for subjects/bookshelves with search, Select All/Clear All
- Constraints generate OR clauses when multiple values selected (semicolon-separated)

**Logical Grouping**
- Group constraints with AND/OR logic using radio buttons
- Nest groups to arbitrary depth using "+ Group" button
- Visual hierarchy via indentation (20px per nesting level)
- Each group shows its logic (AND/OR) and can contain constraints or nested groups
- Delete groups and constraints via delete buttons

**Real-Time Query Preview**
- Query string display (bottom panel) updates automatically as query is built
- Uses monospace font (Consolas) for readability
- Formats nested groups with proper parentheses
- Includes line breaks for long queries with multiple groups
- Copy button to clipboard (Ctrl+C)

**Query Execution**
- "Run Query" button (also F5 keyboard shortcut) executes current query
- Integrates with `BooleanQueryEngine` from `explore_gutenberg.py`
- Results displayed in Treeview widget (right panel) showing first 100 rows
- Columns: id, title, authors, download_count (with proper column widths)
- Results metadata: total count, query string, truncation warning if >100 results
- Status bar feedback: "Executing query...", success with count, or error messages

**Save/Load Queries**
- File > Save Query (Ctrl+S): Save to current file
- File > Save Query As (Ctrl+Shift+S): Save with file dialog
- File > Open Query (Ctrl+O): Load from JSON file with validation
- JSON format includes:
  - `version`: "1.0" (for future compatibility)
  - `created`: ISO timestamp when first saved
  - `modified`: ISO timestamp of last save
  - `query`: Nested `QueryGroupModel` with all constraints and groups
- Recursive serialization/deserialization handles arbitrary nesting depth
- Error handling with dialogs for invalid/corrupted files

**Example JSON Query File**:
```json
{
  "version": "1.0",
  "created": "2025-12-02T10:30:00Z",
  "modified": "2025-12-02T11:45:00Z",
  "query": {
    "logic": "AND",
    "constraints": [
      {
        "field": "subject",
        "operator": "contains",
        "value": "Fiction"
      },
      {
        "logic": "OR",
        "constraints": [
          {
            "field": "download_count",
            "operator": ">",
            "value": "1000"
          },
          {
            "field": "download_count",
            "operator": "range",
            "value": "500..1000"
          }
        ]
      }
    ]
  }
}
```

**Export Results**
- File > Export Results (Ctrl+E): Export query results to file
- File format selection via dialog: CSV (`.csv`) or Parquet (`.parquet`)
- Exports full result set (not just preview)
- Success feedback in status bar with row count
- Warning if no results to export

**User Experience Polish**
- Tooltips on all interactive elements (buttons, fields, operators)
- Keyboard shortcuts for all common actions
- Confirm dialogs for destructive actions (New Query, Close with unsaved changes)
- Numeric input validation (0-999999 range for id/download_count)
- Window size: 1400x900 with resizable panels
- Status bar for operation feedback

## Inputs / Outputs

### Inputs

**Required Data File**
- `data/meta/gutenberg_metadata.parquet`: Gutenberg metadata corpus
- Columns: id, title, authors, subjects, bookshelves, languages, download_count, media_type, copyright
- Used for query execution and multi-select dialog options

**Query Files (Optional)**
- JSON files with `.json` extension
- User-selected via File > Open Query dialog
- Example: `young-adult-fiction-popular.json`

**No CLI Flags or Environment Variables**
- Launched via: `python -m lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui`
- No command-line arguments or configuration files

### Outputs

**Query Files**
- JSON files with query definitions
- User-selected location via Save/Save As dialogs
- Contains version, timestamps, and nested query structure

**Result Export Files**
- CSV files (`.csv`): Comma-separated values with header row
- Parquet files (`.parquet`): Columnar binary format for efficient storage
- User-selected location via Export Results dialog
- Contains full result set (id, title, authors, subjects, bookshelves, languages, download_count, media_type, copyright)

**No Logs or Telemetry**
- All feedback via GUI status bar and message dialogs
- No external logging or analytics

## API / CLI Surface

### Module Execution

```bash
# Launch GUI
poetry run python -m lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui

# Or with explicit Python path
python -m lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui
```

### Programmatic API

```python
# Import and launch
from lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui import main
main()

# Use data models programmatically
from lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui import (
    QueryConstraintModel,
    QueryGroupModel,
    SavedQuery,
)

# Build a query programmatically
constraint = QueryConstraintModel(
    field="subject",
    operator="contains",
    value="Fiction"
)
query_group = QueryGroupModel(
    logic="AND",
    constraints=[constraint]
)
query_string = query_group.to_query_string()  # "subject:Fiction"

# Serialize to JSON
saved = SavedQuery(
    created="2025-12-04T10:00:00Z",
    modified="2025-12-04T10:00:00Z",
    query=query_group
)
json_str = saved.to_json()

# Deserialize from JSON
loaded = SavedQuery.from_json(json_str)
```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| F5 | Run Query |
| Ctrl+N | New Query |
| Ctrl+O | Open Query |
| Ctrl+S | Save Query |
| Ctrl+Shift+S | Save Query As |
| Ctrl+E | Export Results |
| Ctrl+Q | Quit Application |

## Data & State

### Data Models

**QueryConstraintModel** (dataclass)
- `field: str` — Field name (e.g., "subject", "download_count")
- `operator: str` — Operator (e.g., "contains", ">", "range")
- `value: str | list[str]` — Value(s) for comparison
- `to_query_string() -> str` — Generates query syntax

**QueryGroupModel** (dataclass)
- `logic: str` — "AND" or "OR"
- `constraints: list[QueryConstraintModel | QueryGroupModel]` — Nested structure
- `to_query_string() -> str` — Generates nested query with parentheses

**SavedQuery** (dataclass)
- `version: str` — Format version ("1.0")
- `created: str` — ISO timestamp
- `modified: str` — ISO timestamp
- `query: QueryGroupModel` — Root query group
- `to_json() -> str` — Serialize to JSON
- `from_json(json_str: str) -> SavedQuery` — Deserialize from JSON

### Application State

**QueryBuilderApp** maintains:
- `root_group: QueryGroupWidget` — Root query group (always present)
- `current_file: Path | None` — Currently open file (for Save without dialog)
- `last_results: pd.DataFrame | None` — Most recent query results
- `engine: BooleanQueryEngine` — Query execution engine

**State Transitions**:
1. Launch → Empty query (single root group with no constraints)
2. Add constraints → Update UI, regenerate query string
3. Run query → Execute, display results, store in `last_results`
4. Save query → Serialize to JSON, update `current_file`
5. Open query → Deserialize JSON, rebuild UI widgets, clear `last_results`
6. Export results → Write `last_results` to file
7. New query → Confirm unsaved changes, reset to empty query

**No Persistent State**:
- No settings file or preferences
- No query history or recent files list
- State exists only in GUI widgets during session

### Data Flow

```
Gutenberg Metadata Parquet
    ↓
BooleanQueryEngine (from explore_gutenberg.py)
    ↓
QueryBuilderApp._run_query()
    ↓
Results DataFrame → QueryBuilderApp.last_results
    ↓
Treeview Display (first 100 rows)
    ↓
Export to CSV/Parquet
```

## Constraints & Risks

### Technical Constraints

- **Standard library only**: Must use Tkinter (no additional GUI dependencies like Qt, wxPython)
- **Performance**: Large result sets (>10k rows) may slow UI; preview limited to first 100 rows
- **Data dependency**: Requires `data/meta/gutenberg_metadata.parquet` file to be present at launch
- **Type safety**: Third-party API calls (tkinter, pandas) require line-specific type ignores due to incomplete stubs

### Design Constraints

- **Complexity**: Nested query groups add UI complexity; must remain intuitive
- **No drag-and-drop reordering**: Tkinter's DND is complex; use delete and re-add instead
- **Single window**: No multi-window support (single main window)
- **Synchronous execution**: Query execution blocks UI thread (acceptable for metadata queries)

### Risks

- **Testing**: GUI testing is challenging; mitigated by focusing on unit tests for models/logic with mocked UI
- **Maintenance**: Tkinter has dated look/feel but provides cross-platform stability and zero dependencies
- **Schema changes**: If Gutenberg metadata schema changes, field types and operators may need updates
- **Large queries**: Deeply nested queries (>10 levels) may be hard to visualize; no pagination for groups

## Definition of Done

- [x] Behavior matches acceptance criteria (all 15 items)
- [x] Tests updated/added (24 tests passing)
  - Data model serialization/deserialization
  - Query string generation for all operators
  - Nested group query string generation with proper parentheses
  - JSON save/load round-trip with complex nested structures
  - Integration with BooleanQueryEngine
  - Multi-select OR clause generation
  - Error handling for invalid queries, missing files, export failures
  - Numeric input validation
  - Edge cases: empty queries, single constraint, deeply nested groups
- [x] Docs updated (README with GUI instructions, keyboard shortcuts, usage examples)
- [x] Code passes all quality checks:
  - [x] Black formatting
  - [x] Ruff linting (no errors)
  - [x] Pyright type checking (0 errors, 0 warnings)
  - [x] Pytest (24 tests passing)
- [x] Module structure refactored for maintainability:
  - [x] 6 files: `__init__.py`, `__main__.py`, `constants.py`, `tk_helpers.py`, `widgets.py`, `app.py`
  - [x] Type suppressions isolated to `tk_helpers.py`
  - [x] Executable via `python -m lexile_corpus_tuner.pipeline_scripts.gutenberg_query_builder_ui`


