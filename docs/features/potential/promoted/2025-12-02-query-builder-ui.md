# Gutenberg Query Builder UI (Issue #9)

- Date captured: 2025-12-02
- Author: Dan Moisan
- Status: Promoted -> docs/features/active/Gutenberg_Query_Builder_UI/ (Issue #9)

- Issue: #9
- Issue URL: https://github.com/drmoisan/lexile-corpus-tuner/issues/9
- Last Updated: 2025-12-04
## Problem / Why

The existing Gutenberg metadata explorer CLI requires users to manually construct complex query strings with proper syntax for boolean operators, ranges, and nested logic. This is error-prone and has a steep learning curve. Users need a visual, interactive way to build queries without memorizing syntax, especially for complex nested queries with multiple fields and operators.

## Proposed Behavior

A lightweight, visual query builder GUI using Python's Tkinter that allows users to:

- Visually construct queries by selecting fields, operators, and values
- Group constraints with AND/OR logic with unlimited nesting
- See real-time query string preview as they build
- Execute queries and preview results (first 100 rows)
- Save/load queries to JSON files for reuse
- Export results to CSV or Parquet formats
- Use multi-select dialogs for subjects/bookshelves with search functionality

## Acceptance Criteria (early draft)

- [ ] GUI launches and displays main window with field palette, query builder, and results panels
- [ ] Users can add constraints by clicking fields in palette
- [ ] Users can select appropriate operators based on field type (text: contains/=, numeric: >/</>=/<=/range, boolean: =/!=)
- [ ] Users can group constraints with AND/OR logic
- [ ] Users can nest groups to arbitrary depth
- [ ] Query string display updates in real-time as query is built
- [ ] Users can execute queries and see results in preview panel
- [ ] Users can save queries to JSON files
- [ ] Users can load previously saved queries
- [ ] Users can export results to CSV or Parquet
- [ ] Multi-select dialogs for subjects/bookshelves with search/filter
- [ ] All code passes Black, Ruff, Pyright, and Pytest checks

## Constraints & Risks

- **Standard library only**: Must use Tkinter (no additional GUI dependencies)
- **Performance**: Large result sets (>10k rows) may slow UI; preview limited to first 100 rows
- **Complexity**: Nested query groups add UI complexity; must remain intuitive
- **Data dependency**: Requires Gutenberg metadata parquet file to be present
- **Testing**: GUI testing is challenging; focus on model/logic unit tests with mocked UI
- **Maintenance**: Tkinter has dated look/feel but provides cross-platform stability

## Test Conditions to Consider

- [ ] Data model serialization/deserialization (QueryConstraintModel, QueryGroupModel, SavedQuery)
- [ ] Query string generation for all operators (contains, =, >, <, >=, <=, range)
- [ ] Nested group query string generation with proper parentheses
- [ ] JSON save/load round-trip with complex nested structures
- [ ] Integration with BooleanQueryEngine from explore_gutenberg.py
- [ ] Multi-select OR clause generation (semicolon-separated values)
- [ ] Error handling for invalid queries, missing files, export failures
- [ ] Numeric input validation (0-999999 range)
- [ ] Edge cases: empty queries, single constraint, deeply nested groups

## Next Step

- [X] Implementation complete (all 13 phases, 24 tests passing)
- [ ] Promote to GitHub issue for tracking
- [ ] Create `docs/features/active/query-builder-ui/` folder with formal spec/plan/user-story

