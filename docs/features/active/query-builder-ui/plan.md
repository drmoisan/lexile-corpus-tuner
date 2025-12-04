# query-builder-ui - Plan

- Issue: #9
- Owner: drmoisan
- Last Updated: 2025-12-04

## Required References (read, do not restate)

- Coding workflow and standards: [`docs/code-change.instructions.md`](../../../code-change.instructions.md)
- Unit test policy: [`docs/unit-test-policy.md`](../../../unit-test-policy.md)

**All work must comply with these policies; do not duplicate their content here.**

## Implementation Strategy

Build a visual query builder GUI using Python's Tkinter (standard library) with MVC architecture. The implementation proceeds bottom-up: data models → widgets → application controller → integration → polish.

Key technical decisions:
- **Tkinter over Qt/wxPython**: Zero dependencies, cross-platform, sufficient for requirements
- **MVC pattern**: Separate models (testable without GUI) from views (widgets) from controller (app)
- **Click-to-add over drag-and-drop**: Simpler implementation, avoids complex Tkinter DND API
- **JSON for persistence**: Human-readable, version-able, compatible with git
- **Module package structure**: Refactor into `constants.py`, `tk_helpers.py`, `widgets.py`, `app.py` for maintainability

## Phases

### Phase 1: Foundation & Data Models [100%] ✓
- [x] Design architecture (MVC with Tkinter, JSON persistence, BooleanQueryEngine integration)
- [x] Define `QueryConstraintModel` dataclass (field, operator, value, to_query_string())
- [x] Define `QueryGroupModel` dataclass (logic, constraints list, to_query_string() with recursion)
- [x] Define `SavedQuery` dataclass (version, timestamps, query, to_json/from_json)
- [x] Implement recursive query string generation with proper parentheses for nested groups
- [x] Write unit tests for data models (query string generation, all operators, nesting)
- [x] Write unit tests for JSON serialization/deserialization (round-trip, complex structures)
- [x] Pass Black/Ruff/Pyright checks for models

### Phase 2: Core Application & Layout [100%] ✓
- [x] Create `QueryBuilderApp` class with Tk root window (1400x900)
- [x] Implement menu bar (File: New/Open/Save/Save As/Export/Exit, Query: Run, Help: About)
- [x] Set up main layout with PanedWindow (left: palette, center: builder, right: results)
- [x] Add status bar at bottom for operation feedback
- [x] Implement `_new_query()` with confirm dialog for unsaved changes
- [x] Implement `_on_closing()` with save prompt
- [x] Bind keyboard shortcuts (Ctrl+N/O/S/Shift+S/E/Q, F5)
- [x] Pass Black/Ruff/Pyright checks for app skeleton

### Phase 3: Query Constraint Widgets [100%] ✓
- [x] Create `QueryConstraintWidget` class with field/operator/value/delete layout
- [x] Implement field dropdown (Combobox) with all Gutenberg fields
- [x] Implement operator dropdown (context-aware based on field type)
  - [x] Text fields: contains, =
  - [x] Numeric fields: =, >, <, >=, <=, range
  - [x] Boolean fields: =, !=
- [x] Implement value input (multi-type support)
  - [x] Text entry for text/numeric fields
  - [x] Spinbox for id/download_count (0-999999 validation)
  - [x] Range entry for range operator ("100..500" format)
  - [x] Browse button for subjects/bookshelves (opens multi-select dialog)
- [x] Implement multi-select dialog (Listbox with search, Select All/Clear All)
- [x] Load canonical subjects/bookshelves from parquet via `get_canonical_sets()`
- [x] Generate OR clauses from multi-select (semicolon-separated values)
- [x] Add delete button with callback to parent
- [x] Add `to_model()` method to convert widget state to `QueryConstraintModel`
- [x] Write unit tests for constraint widget logic (operator selection, value conversion)
- [x] Pass Black/Ruff/Pyright checks

### Phase 4: Query Group Widgets [100%] ✓
- [x] Create `QueryGroupWidget` class with hierarchical layout
- [x] Implement AND/OR radio buttons with callback to parent
- [x] Implement constraint container (Frame with vertical layout)
- [x] Implement grouping controls (+ Constraint, + Group, Delete, Ungroup buttons)
- [x] Add visual hierarchy via indentation (nesting_level * 20px)
- [x] Support nested `QueryGroupWidget` instances recursively
- [x] Implement `add_constraint()` to create and manage `QueryConstraintWidget` instances
- [x] Implement `add_group()` to create and manage nested `QueryGroupWidget` instances
- [x] Implement delete functionality for child widgets
- [x] Add `to_model()` method to recursively convert to `QueryGroupModel`
- [x] Integrate root `QueryGroupWidget` into `QueryBuilderApp`
- [x] Wire up callbacks to `_update_query_from_root()` for real-time updates
- [x] Write unit tests for group widget logic (nesting, deletion, model conversion)
- [x] Pass Black/Ruff/Pyright checks

### Phase 5: Field Palette & Constraint Creation [100%] ✓
- [x] Create field palette in left panel (LabelFrame "Available Fields")
- [x] Organize fields by type (Text Fields, Numeric Fields, Boolean Fields)
- [x] Add buttons for each field (id, title, authors, subjects, bookshelves, languages, download_count, media_type, copyright)
- [x] Implement click handler to add constraint to root group
- [x] Select appropriate default operator based on field type
  - [x] Text → "contains"
  - [x] Numeric → ">"
  - [x] Boolean → "="
- [x] Provide status bar feedback when field added ("Added constraint for [field]")
- [x] Add visual instructions ("Click to add constraint")
- [x] Pass Black/Ruff/Pyright checks

### Phase 6: Query String Display [100%] ✓
- [x] Create query string display in bottom panel (Text widget, read-only)
- [x] Use monospace font (Consolas 10) for readability
- [x] Add Copy button to copy query string to clipboard
- [x] Implement `_update_query_display()` to regenerate string from root group
- [x] Wire up callbacks from all constraint/group changes
- [x] Add formatting for readability (line breaks on AND/OR for long queries)
- [x] Make text selectable for manual copy
- [x] Pass Black/Ruff/Pyright checks

### Phase 7: Query Execution & Results Display [100%] ✓
- [x] Import `BooleanQueryEngine` from `scripts.production.explore_gutenberg`
- [x] Initialize engine with parquet path in `QueryBuilderApp.__init__()`
- [x] Implement `_run_query()` method
  - [x] Generate query string from root group
  - [x] Call `engine.execute_query(query_string)`
  - [x] Handle None result (invalid query) with error dialog
  - [x] Store result in `self.last_results`
  - [x] Update status bar with result count
- [x] Implement `_display_results()` method
  - [x] Clear existing Treeview items
  - [x] Configure columns (id, title, authors, download_count)
  - [x] Set column widths (80, 300, 200, 100)
  - [x] Insert first 100 rows into Treeview
  - [x] Handle NaN values safely (try/except with isinstance check)
  - [x] Show results metadata (count, query, truncation warning)
- [x] Add "Run Query" button to query builder panel header
- [x] Add "Run Query" menu item with F5 keyboard shortcut
- [x] Add status bar feedback ("Executing query...", success/failure)
- [x] Handle errors with messagebox dialogs
- [x] Use proper type narrowing for DataFrame (avoid assert, satisfy Pyright)
- [x] Write integration tests for query execution (mock BooleanQueryEngine)
- [x] Pass Black/Ruff/Pyright checks

### Phase 8: Save/Load Queries [100%] ✓
- [x] Implement `_save_query()` method
  - [x] If `current_file` exists, save to that path
  - [x] Otherwise, call `_save_query_as()`
- [x] Implement `_save_query_as()` method
  - [x] Open file dialog (asksaveasfilename) with .json default
  - [x] Call `_save_query_to_file(path)`
  - [x] Update `current_file` on success
- [x] Implement `_save_query_to_file(path)` method
  - [x] Generate `SavedQuery` from root group via `to_model()`
  - [x] Set timestamps (created on first save, modified on subsequent)
  - [x] Serialize to JSON via `saved_query.to_json()`
  - [x] Write to file with error handling
  - [x] Show success in status bar
- [x] Implement `_open_query()` method
  - [x] Open file dialog (askopenfilename) with .json filter
  - [x] Read file and deserialize via `SavedQuery.from_json()`
  - [x] Validate query format (try/except with messagebox)
  - [x] Clear existing root group
  - [x] Rebuild UI widgets from loaded `QueryGroupModel`
  - [x] Update `current_file`
  - [x] Clear `last_results`
  - [x] Show success in status bar
- [x] Add File > Save Query (Ctrl+S) menu item
- [x] Add File > Save Query As (Ctrl+Shift+S) menu item
- [x] Add File > Open Query (Ctrl+O) menu item
- [x] Write unit tests for serialization edge cases (empty query, deeply nested)
- [x] Pass Black/Ruff/Pyright checks

### Phase 9: Export Results [100%] ✓
- [x] Implement `_export_results()` method
  - [x] Check if `last_results` is None or empty → show warning dialog
  - [x] Open file dialog (asksaveasfilename) with filetypes: CSV, Parquet
  - [x] Detect format from path suffix (.csv or .parquet)
  - [x] Export CSV via `last_results.to_csv(path, index=False)`
  - [x] Export Parquet via `last_results.to_parquet(path, index=False)`
  - [x] Show success in status bar with row count
  - [x] Handle export errors with messagebox
- [x] Add File > Export Results (Ctrl+E) menu item
- [x] Write integration tests for export (mock pandas methods)
- [x] Pass Black/Ruff/Pyright checks

### Phase 10: Polish & User Experience [100%] ✓
- [x] Create `ToolTip` helper class for hover tooltips
- [x] Add tooltips to all interactive elements
  - [x] Field palette buttons ("Click to add [field] constraint")
  - [x] Query builder buttons ("Add constraint", "Add nested group", "Delete group")
  - [x] Constraint widgets ("Select operator", "Enter value", "Delete constraint")
  - [x] Results panel ("Run query to see results")
  - [x] Query display ("Copy to clipboard")
- [x] Add numeric input validation (`_validate_numeric` for 0-999999 range)
- [x] Add confirm dialogs for destructive actions
  - [x] New Query: "You have unsaved changes. Continue?"
  - [x] Close window: "Save changes before closing?"
- [x] Improve About dialog with keyboard shortcuts list
- [x] Ensure status bar updates for all operations
- [x] Test window resizing and panel behavior
- [x] Pass Black/Ruff/Pyright checks

### Phase 11: Testing [100%] ✓
- [x] Write unit tests for `QueryConstraintModel` (to_query_string for all operators)
- [x] Write unit tests for `QueryGroupModel` (nested groups, recursion)
- [x] Write unit tests for `SavedQuery` (JSON serialization, round-trip)
- [x] Write integration tests for complex queries (deeply nested, all operator types)
- [x] Mock Tkinter widgets (Tk, StringVar, Combobox, Text, Treeview)
- [x] Mock pandas DataFrame and parquet operations
- [x] Test error cases (invalid query syntax, missing files, export failures)
- [x] Test edge cases (empty query, single constraint, 10-level nesting)
- [x] Achieve 24 passing tests
- [x] Pass Pytest with no failures

### Phase 12: Module Refactoring [100%] ✓
- [x] Create `scripts/production/gutenberg_query_builder_ui/` package directory
- [x] Create `constants.py` with field types, operators, config (40 lines)
- [x] Create `tk_helpers.py` with third-party wrappers and type ignores (180 lines)
  - [x] Tkinter wrappers: `tk_listbox_*`, `tk_canvas_*`, `tk_treeview_*`, `tk_panedwindow_*`
  - [x] Pandas wrappers: `pandas_read_parquet`, `pandas_to_csv`, `pandas_to_parquet`, `pandas_is_na`
  - [x] Isolate all `# type: ignore` comments to this module
- [x] Create `widgets.py` with widget classes (660 lines)
  - [x] `ToolTip`, `QueryConstraintWidget`, `QueryGroupWidget`
- [x] Create `app.py` with main application class (580 lines)
  - [x] `QueryBuilderApp` with all controller logic
- [x] Create `__init__.py` with public exports (60 lines)
  - [x] Re-export all public classes and functions
  - [x] Provide `main()` entry point
- [x] Create `__main__.py` for module execution (15 lines)
- [x] Update imports to absolute paths (`scripts.production.explore_gutenberg`)
- [x] Move runtime-unused imports to `TYPE_CHECKING` blocks (tkinter, pandas)
- [x] Verify all 466 tests still pass (no behavioral changes)
- [x] Pass Black/Ruff/Pyright checks (0 errors, 0 warnings)

### Phase 13: Documentation [100%] ✓
- [x] Update README.md with GUI section
  - [x] Installation and launch instructions
  - [x] Screenshot or ASCII art mockup
  - [x] Usage workflow (7 steps: launch, add constraints, group, preview, run, save, export)
  - [x] Keyboard shortcuts table
  - [x] Query file format (JSON example)
  - [x] Field types and operators list
- [x] Document module structure and execution method
- [x] Add inline code comments for complex logic (recursion, serialization)
- [x] Write docstrings for all public methods
- [x] Verify documentation completeness

## Test Plan

### Unit Tests (test_query_builder_ui.py)

**Data Models** (8 tests)
- `QueryConstraintModel.to_query_string()` for all operators
  - Text: `contains` ("field:value"), `=` ("field=value")
  - Numeric: `=`, `>`, `<`, `>=`, `<=` ("field>100")
  - Range: "field:100..500"
- `QueryGroupModel.to_query_string()` for single and nested groups
  - AND logic: "(field1:value1 AND field2:value2)"
  - OR logic: "(field1:value1 OR field2:value2)"
  - Nested: "((field1:value1 AND field2:value2) OR field3:value3)"
- `SavedQuery` JSON serialization/deserialization
  - Round-trip test (serialize → deserialize → compare)
  - Complex nested structure (3-level nesting, 10 constraints)
  - Version field preserved

**Serialization** (4 tests)
- Empty query serialization
- Single constraint serialization
- Deeply nested groups (10 levels)
- Mixed constraint types (text, numeric, boolean, range)

**Integration** (12 tests)
- Complex query string generation
  - "(subject:Fiction OR subject:Science) AND download_count>1000"
  - "(title:Python AND authors:Guido) OR (subject:Programming AND download_count:100..500)"
- Multi-select OR clause generation ("subject:Fiction;Science;History")
- Query execution with mocked `BooleanQueryEngine`
- Results display with mocked DataFrame
- Export to CSV/Parquet with mocked pandas methods
- Error handling (invalid query, missing file, export failure)

**Edge Cases** (4 tests)
- Empty query → "" (no constraints)
- Single constraint → "field:value" (no parentheses)
- 10-level nested groups → proper parentheses nesting
- NaN values in results → safe display (no exceptions)

**Total: 24 tests, all passing**

### Manual Testing (GUI UX)

**Smoke Test**
1. Launch GUI → window appears with all panels
2. Click "subject" in palette → constraint appears in query builder
3. Select "Fiction" from multi-select → constraint value updates
4. Click "Run Query" → results appear in right panel
5. Save query → file dialog appears, saves successfully
6. Open saved query → UI recreates all constraints
7. Export results → file dialog appears, exports successfully

**Complex Query Test**
1. Build nested query: `(subject:Fiction OR subject:Science) AND download_count>1000 AND language:en`
2. Verify query string display matches expected syntax
3. Run query and verify result count is reasonable
4. Add another nested group inside first group
5. Verify parentheses nesting in query string
6. Save and reload query → verify all nesting preserved

**Error Handling Test**
1. Open invalid JSON file → error dialog appears, app doesn't crash
2. Run query with no constraints → error dialog or empty results
3. Export results before running query → warning dialog
4. Close window with unsaved changes → confirm dialog

### Performance Testing

**Large Result Sets**
- Query returning >10k results → preview shows first 100, no lag
- Export 10k+ results → file saves successfully, no UI freeze

**Deep Nesting**
- 10-level nested groups → query string generates correctly
- 20+ constraints in one query → UI remains responsive

## Open Questions / Notes

### Design Decisions

**Why click-to-add instead of drag-and-drop?**
- Tkinter's DND API is complex and poorly documented
- Click-to-add is simpler, more reliable, and sufficient for UX
- Users can still build queries quickly by clicking fields

**Why limit results preview to 100 rows?**
- Tkinter Treeview slows down with >1000 items
- Users only need preview to verify query correctness
- Full export is available for complete dataset

**Why JSON for query persistence?**
- Human-readable for debugging and manual editing
- Version-able in git for reproducibility
- Compatible with future schema migrations via version field
- Easy to parse and generate (built-in `json` module)

**Why refactor into module package?**
- Single 1567-line file was hard to navigate and maintain
- Module structure allows clean separation of concerns
- Type safety improved by isolating third-party API calls
- Future enhancements can add new modules without modifying existing code

### Known Limitations

**No drag-and-drop reordering**
- Constraint order doesn't affect query logic (AND/OR is commutative)
- Users can delete and re-add if order matters visually

**No query history**
- Users must manually save queries to files
- External version control (git) handles query history

**No advanced text search**
- Only supports field:value syntax, not full-text search
- Gutenberg metadata doesn't include full book text

**No query optimization**
- Query string generated as-is from UI structure
- No automatic rewriting or optimization for performance

### Future Enhancements (Not in Scope)

- Import queries from CLI command history
- Export query as CLI command for reproducibility
- Visual query tree diagram (separate from linear display)
- Query templates library (pre-built common queries)
- Batch query execution (run multiple saved queries)
- Result visualization (charts, histograms)
- Advanced filters (regex, date ranges, custom fields)

