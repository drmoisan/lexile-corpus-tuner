from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# =============================================================================
# Pandas Operations Section - Isolated Type Suppressions
# =============================================================================
# All pandas DataFrame/Series operations with incomplete type stubs are
# isolated in this section. Application logic remains fully typed.


def _pandas_get_column(df: pd.DataFrame, column: str) -> pd.Series:  # type: ignore[type-arg,misc]
    """Get column from DataFrame as Series.

    Isolated pandas operation: pandas-stubs incomplete for column access.

    Args:
        df: DataFrame
        column: Column name

    Returns:
        Series
    """
    return df[column]  # type: ignore[no-any-return]


def _pandas_string_series(series: pd.Series) -> Any:  # type: ignore[misc]
    """Convert Series to string type.

    Isolated pandas operation: pandas-stubs incomplete for Series.astype().

    Args:
        series: pandas Series to convert

    Returns:
        Series with string type
    """
    return series.astype(str)  # type: ignore[no-any-return]


def _pandas_split_explode(series: pd.Series, delimiter: str) -> Any:  # type: ignore[misc]
    """Split string series by delimiter and explode to rows.

    Isolated pandas operation: pandas-stubs incomplete for str accessor.

    Args:
        series: pandas Series with string data
        delimiter: Delimiter to split on

    Returns:
        Exploded series
    """
    return series.str.split(delimiter).explode()  # type: ignore[no-any-return]


def _pandas_series_is_empty(series: pd.Series) -> bool:
    """Check if series is empty.

    Isolated pandas operation: pandas-stubs incomplete for empty property.

    Args:
        series: pandas Series to check

    Returns:
        True if empty, False otherwise
    """
    return series.empty  # type: ignore[no-any-return]


def _pandas_strip_unique(series: pd.Series) -> Any:  # type: ignore[misc]
    """Strip whitespace and get unique values.

    Isolated pandas operation: pandas-stubs incomplete for str accessor.

    Args:
        series: pandas Series with string data

    Returns:
        Array of unique values
    """
    items = series.dropna().astype(str).str.strip()  # type: ignore[union-attr]
    return items.unique()  # type: ignore[no-any-return]


def _pandas_contains(series: pd.Series, term: str, case: bool = False) -> Any:  # type: ignore[misc]
    """Check if series contains search term.

    Isolated pandas operation: pandas-stubs incomplete for str.contains().

    Args:
        series: pandas Series to search
        term: Search term
        case: Case-sensitive search flag

    Returns:
        Boolean mask series
    """
    return series.astype(str).str.contains(term, case=case, regex=False, na=False)  # type: ignore[no-any-return,union-attr]


def _pandas_comparison_mask(series: pd.Series, operator: str, value: float) -> Any:  # type: ignore[misc]
    """Apply numeric comparison operator to series.

    Isolated pandas operation: pandas-stubs incomplete for comparison operators.

    Args:
        series: pandas Series with numeric data
        operator: Comparison operator (>, <, >=, <=)
        value: Value to compare against

    Returns:
        Boolean mask series
    """
    if operator == ">":
        return series > value  # type: ignore[no-any-return]
    elif operator == "<":
        return series < value  # type: ignore[no-any-return]
    elif operator == ">=":
        return series >= value  # type: ignore[no-any-return]
    elif operator == "<=":
        return series <= value  # type: ignore[no-any-return]
    return pd.Series([True] * len(series), index=series.index)  # type: ignore[no-any-return]


def _pandas_range_mask(series: pd.Series, min_val: float, max_val: float) -> Any:  # type: ignore[misc]
    """Check if series values are within range.

    Isolated pandas operation: pandas-stubs incomplete for comparison operators.

    Args:
        series: pandas Series with numeric data
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Boolean mask series
    """
    return (series >= min_val) & (series <= max_val)  # type: ignore[no-any-return]


def _pandas_exact_match(series: pd.Series, term: str) -> Any:  # type: ignore[misc]
    """Check for exact case-insensitive match.

    Isolated pandas operation: pandas-stubs incomplete for str accessor.

    Args:
        series: pandas Series to search
        term: Term to match

    Returns:
        Boolean mask series
    """
    return series.astype(str).str.lower() == term.lower()  # type: ignore[no-any-return,union-attr]


def _pandas_filter_by_mask(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:  # type: ignore[type-arg]
    """Filter DataFrame by boolean mask.

    Isolated pandas operation: pandas-stubs incomplete for indexing.

    Args:
        df: DataFrame to filter
        mask: Boolean mask series

    Returns:
        Filtered DataFrame
    """
    return df[mask]  # type: ignore[no-any-return]


def _pandas_read_parquet(path: Path) -> pd.DataFrame:  # type: ignore[type-arg]
    """Read parquet file.

    Isolated pandas operation: pandas-stubs incomplete for read_parquet.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame
    """
    return pd.read_parquet(path)  # type: ignore[no-any-return]


def _pandas_to_csv(df: pd.DataFrame, path: Path) -> None:  # type: ignore[type-arg]
    """Write DataFrame to CSV.

    Isolated pandas operation: pandas-stubs incomplete for to_csv.

    Args:
        df: DataFrame to write
        path: Output path
    """
    df.to_csv(path, index=False)  # type: ignore[call-overload]


def _pandas_to_parquet(df: pd.DataFrame, path: Path) -> None:  # type: ignore[type-arg]
    """Write DataFrame to Parquet.

    Isolated pandas operation: pandas-stubs incomplete for to_parquet.

    Args:
        df: DataFrame to write
        path: Output path
    """
    df.to_parquet(path, index=False)  # type: ignore[call-overload]


def _pandas_to_string(df: pd.DataFrame, columns: list[str], max_rows: int = 10) -> str:  # type: ignore[type-arg]
    """Convert DataFrame to string representation.

    Isolated pandas operation: pandas-stubs incomplete for to_string.

    Args:
        df: DataFrame to convert
        columns: Columns to include
        max_rows: Maximum rows to display

    Returns:
        String representation
    """
    return df[columns].head(max_rows).to_string(index=False)  # type: ignore[no-any-return,call-overload]


# =============================================================================
# End Pandas Operations Section
# =============================================================================


def get_canonical_sets(df: pd.DataFrame) -> tuple[set[str], set[str]]:  # type: ignore[type-arg]
    """Extract canonical sets of subjects and bookshelves from DataFrame.

    Args:
        df: DataFrame containing 'subjects' and 'bookshelves' columns
            with semicolon-delimited strings.

    Returns:
        Tuple of (unique_subjects, unique_bookshelves) sets.
    """

    def extract_unique(column: str) -> set[str]:
        if column not in df.columns:
            return set()

        # Convert to string, split by semicolon, explode to rows
        series = _pandas_string_series(_pandas_get_column(df, column))
        exploded = _pandas_split_explode(series, ";")

        if _pandas_series_is_empty(exploded):
            return set()

        # Strip whitespace and get unique values
        unique_items = set(_pandas_strip_unique(exploded))

        # Clean up artifacts
        unique_items.discard("")
        unique_items.discard("nan")
        unique_items.discard("None")

        return unique_items

    subjects = extract_unique("subjects")
    bookshelves = extract_unique("bookshelves")

    return subjects, bookshelves


def save_canonical_sets(
    subjects: set[str], bookshelves: set[str], output_dir: Path
) -> None:
    """Save canonical sets to text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    subj_path = output_dir / "subjects.txt"
    with subj_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(sorted(subjects)))
    print(f"Saved {len(subjects)} subjects to {subj_path}")

    shelf_path = output_dir / "bookshelves.txt"
    with shelf_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(sorted(bookshelves)))
    print(f"Saved {len(bookshelves)} bookshelves to {shelf_path}")


class BooleanQueryEngine:
    """Helper to evaluate boolean queries against a DataFrame column.

    Supports:
    - Boolean operators: AND, OR, NOT, parentheses
    - Field-specific queries: field:value (e.g., "subject:Fiction")
    - Numeric comparisons: field>value, field<value, field>=value, field<=value
    - Range queries: field:min..max (e.g., "download_count:1000..5000")
    - Exact matches: field="value" (case-insensitive)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_fields = {"download_count", "id"}

    def _tokenize(self, query: str) -> list[str]:
        # Pattern matches:
        # 1. ( or )
        # 2. Field operators with quoted values: field:"quoted value"
        # 3. Field operators with unquoted values: field:value, field>value, etc.
        # 4. Standalone quoted strings (double or single)
        # 5. Non-whitespace, non-paren sequences (terms)
        pattern = (
            r"\(|\)"
            r'|[a-zA-Z_][a-zA-Z0-9_]*:"[^"]*"'  # field:"quoted" (double quotes)
            r"|[a-zA-Z_][a-zA-Z0-9_]*:'[^']*'"  # field:'quoted' (single quotes)
            r"|[a-zA-Z_][a-zA-Z0-9_]*(?:[><=:][^\s()]+)"  # field:unquoted or field>123
            r'|"[^"]*"'  # standalone double-quoted strings
            r"|'[^']*'"  # standalone single-quoted strings
            r"|[^\s()]+"  # unquoted terms
        )
        return re.findall(pattern, query)

    def _parse_field_query(self, token: str) -> tuple[str | None, str, str | None]:
        """Parse field:value, field>value, field:min..max patterns.

        Returns: (field_name, operator, value)
        operator can be: ':', '>', '<', '>=', '<=', '..'
        """
        # Check for comparison operators
        for op in (">=", "<=", ">", "<"):
            if op in token:
                parts = token.split(op, 1)
                if len(parts) == 2:
                    return parts[0].strip(), op, parts[1].strip()

        # Check for field:value or field:range
        if ":" in token:
            field, value = token.split(":", 1)
            field = field.strip()
            value = value.strip()

            # Check for range syntax
            if ".." in value:
                return field, "..", value

            return field, ":", value

        return None, "contains", token

    def _apply_field_filter(self, field: str | None, operator: str, value: str) -> Any:
        """Apply a field-specific filter and return a boolean mask."""
        # If no field specified, search all text columns
        if field is None:
            # Remove quotes from value
            search_term = value
            if (search_term.startswith('"') and search_term.endswith('"')) or (
                search_term.startswith("'") and search_term.endswith("'")
            ):
                search_term = search_term[1:-1]

            # Search across subjects and bookshelves
            mask = pd.Series([False] * len(self.df), index=self.df.index)
            for col in ["subjects", "bookshelves", "title", "authors"]:
                if col in self.df.columns:
                    mask |= _pandas_contains(
                        _pandas_get_column(self.df, col), search_term
                    )
            return mask

        # Field-specific query
        if field not in self.df.columns:
            print(f"Warning: Unknown field '{field}', ignoring.", file=sys.stderr)
            return pd.Series([True] * len(self.df), index=self.df.index)

        col = _pandas_get_column(self.df, field)

        # Handle range queries (field:min..max)
        if operator == "..":
            if field not in self.numeric_fields:
                print(
                    f"Warning: Range query only works on numeric fields, "
                    f"ignoring '{field}'.",
                    file=sys.stderr,
                )
                return pd.Series([True] * len(self.df), index=self.df.index)

            try:
                min_val, max_val = value.split("..", 1)
                min_num = float(min_val.strip())
                max_num = float(max_val.strip())
                return _pandas_range_mask(col, min_num, max_num)
            except ValueError:
                print(
                    f"Warning: Invalid range '{value}', expected 'min..max'.",
                    file=sys.stderr,
                )
                return pd.Series([True] * len(self.df), index=self.df.index)

        # Handle numeric comparisons
        if operator in (">", "<", ">=", "<="):
            if field not in self.numeric_fields:
                print(
                    f"Warning: Numeric comparison only works on numeric fields, "
                    f"ignoring '{field}'.",
                    file=sys.stderr,
                )
                return pd.Series([True] * len(self.df), index=self.df.index)

            try:
                num_val = float(value)
                return _pandas_comparison_mask(col, operator, num_val)
            except ValueError:
                print(
                    f"Warning: Invalid numeric value '{value}'.",
                    file=sys.stderr,
                )
                return pd.Series([True] * len(self.df), index=self.df.index)

        # Handle field:value (contains or exact match)
        # Remove quotes for exact match
        search_term = value
        exact_match = False
        if (search_term.startswith('"') and search_term.endswith('"')) or (
            search_term.startswith("'") and search_term.endswith("'")
        ):
            search_term = search_term[1:-1]
            exact_match = True

        if exact_match:
            # For semicolon-delimited fields (subjects/bookshelves),
            # check if search term matches any delimited value
            if field in ("subjects", "bookshelves"):
                # Split on semicolon, strip whitespace, and check for exact match
                def has_exact_item(cell_value: Any) -> bool:
                    # pandas isna check has incomplete stubs
                    if pd.isna(cell_value):  # type: ignore[reportUnknownMemberType]
                        return False
                    items = [item.strip() for item in str(cell_value).split(";")]
                    return any(item.lower() == search_term.lower() for item in items)

                return col.apply(has_exact_item)  # type: ignore[no-any-return]
            else:
                # Case-insensitive exact match for single-value fields
                return _pandas_exact_match(col, search_term)
        else:
            # Substring match
            return _pandas_contains(col, search_term)

    def _to_rpn(self, tokens: list[str]) -> list[str]:
        # Shunting-yard algorithm
        output_queue: list[str] = []
        operator_stack: list[str] = []
        precedence = {"NOT": 3, "AND": 2, "OR": 1, "(": 0}

        for item in tokens:
            upper_item = item.upper()
            if upper_item in ("AND", "OR", "NOT"):
                while (
                    operator_stack
                    and operator_stack[-1] != "("
                    and precedence.get(operator_stack[-1], 0) >= precedence[upper_item]
                ):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(upper_item)
            elif item == "(":
                operator_stack.append("(")
            elif item == ")":
                while operator_stack and operator_stack[-1] != "(":
                    output_queue.append(operator_stack.pop())
                if operator_stack and operator_stack[-1] == "(":
                    operator_stack.pop()
            else:
                # Term
                output_queue.append(item)

        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue

    def evaluate(self, query: str, column: str | None = None) -> pd.DataFrame:
        """Evaluate a query against the DataFrame.

        Args:
            query: Boolean query string
            column: Legacy parameter for backward compatibility (ignored)

        Returns:
            Filtered DataFrame
        """
        if not query.strip():
            return self.df

        try:
            tokens = self._tokenize(query)
            rpn = self._to_rpn(tokens)
        except Exception as e:
            print(f"Error parsing query: {e}", file=sys.stderr)
            return pd.DataFrame()

        stack: list[Any] = []

        for item in rpn:
            upper_item = item.upper()
            if upper_item == "AND":
                if len(stack) < 2:
                    print("Error: Missing operand for AND", file=sys.stderr)
                    return pd.DataFrame()
                b = stack.pop()
                a = stack.pop()
                stack.append(a & b)
            elif upper_item == "OR":
                if len(stack) < 2:
                    print("Error: Missing operand for OR", file=sys.stderr)
                    return pd.DataFrame()
                b = stack.pop()
                a = stack.pop()
                stack.append(a | b)
            elif upper_item == "NOT":
                if len(stack) < 1:
                    print("Error: Missing operand for NOT", file=sys.stderr)
                    return pd.DataFrame()
                a = stack.pop()
                stack.append(~a)
            else:
                # Parse field query
                field, operator, value = self._parse_field_query(item)
                if value is None:
                    # Skip invalid field queries
                    continue
                mask = self._apply_field_filter(field, operator, value)
                stack.append(mask)

        if not stack:
            return pd.DataFrame()

        if len(stack) > 1:
            print(
                "Error: Invalid query. Use quotes for phrases "
                "(e.g., 'Science Fiction') and explicit operators (AND, OR).",
                file=sys.stderr,
            )
            return pd.DataFrame()

        final_mask = stack[0]
        return _pandas_filter_by_mask(self.df, final_mask)


class QueryHistory:
    """Manages query history with save/load capabilities."""

    def __init__(self, history_file: Path | None = None):
        self.history_file = history_file or Path(".gutenberg_query_history.json")
        self.queries: list[str] = []
        self.load()

    def load(self) -> None:
        """Load query history from file."""
        if self.history_file.exists():
            try:
                with self.history_file.open("r", encoding="utf-8") as f:
                    self.queries = json.load(f)
            except Exception:
                self.queries = []

    def save(self) -> None:
        """Save query history to file."""
        try:
            with self.history_file.open("w", encoding="utf-8") as f:
                json.dump(self.queries[-100:], f, indent=2)  # Keep last 100
        except Exception as e:
            print(f"Warning: Could not save query history: {e}", file=sys.stderr)

    def add(self, query: str) -> None:
        """Add query to history."""
        if query and (not self.queries or query != self.queries[-1]):
            self.queries.append(query)
            self.save()

    def get_recent(self, n: int = 10) -> list[str]:
        """Get n most recent queries."""
        return self.queries[-n:]


def interactive_explorer(parquet_path: Path) -> None:
    """Run interactive explorer loop for Gutenberg metadata."""
    if not parquet_path.exists():
        print(f"File not found: {parquet_path}", file=sys.stderr)
        print(
            "Please run 'scripts/production/build_gutenberg_id_list.py' first.",
            file=sys.stderr,
        )
        return

    print(f"Loading {parquet_path}...")
    try:
        df = _pandas_read_parquet(parquet_path)
    except Exception as e:
        print(f"Error reading parquet: {e}", file=sys.stderr)
        return

    print(f"Loaded {len(df)} records.")

    print("Extracting metadata...")
    subjects, bookshelves = get_canonical_sets(df)
    print(f"Unique Subjects: {len(subjects)}")
    print(f"Unique Bookshelves: {len(bookshelves)}")

    engine = BooleanQueryEngine(df)
    last_results: pd.DataFrame | None = None
    history = QueryHistory()

    print("\nCommands:")
    print("  q <query>          Universal search (e.g., 'q Fiction AND Space')")
    print("  q <field>:<term>   Field search (e.g., 'q subject:Fiction')")
    print("  q <field>><value>  Numeric filter (e.g., 'q download_count>1000')")
    print("  q <field>:min..max Range filter (e.g., 'q download_count:1000..5000')")
    print("  s <query>          Search subjects (legacy, use 'q subject:...')")
    print("  b <query>          Search bookshelves (legacy, use 'q shelf:...')")
    print("  ls s               List all subjects")
    print("  ls b               List all bookshelves")
    print("  ls f               List all available fields")
    print("  history            Show recent queries")
    print("  export_sets [dir]  Save unique subjects/bookshelves to files")
    print("  export_results <file> Save last search results to CSV/Parquet")
    print("  help               Show query syntax examples")
    print("  quit / exit        Quit")

    while True:
        try:
            cmd_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd_input or cmd_input.lower() in ("q", "quit", "exit"):
            break

        parts = cmd_input.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "q" and arg:
            # Universal query command
            history.add(arg)
            last_results = engine.evaluate(arg)
            print(f"Found {len(last_results)} matches:")
            if not last_results.empty:
                cols = [
                    c
                    for c in ["id", "title", "subjects", "bookshelves"]
                    if c in df.columns
                ]
                print(_pandas_to_string(last_results, cols, 10))
                if len(last_results) > 10:
                    print(f"... {len(last_results) - 10} more")

        elif cmd == "ls":
            if arg == "s":
                print("\n".join(sorted(subjects)))
            elif arg == "b":
                print("\n".join(sorted(bookshelves)))
            elif arg == "f":
                print("Available fields:")
                for col in df.columns:  # type: ignore[misc]
                    col_name: str = str(col)  # type: ignore[arg-type]
                    print(f"  {col_name}")
            else:
                print("Usage: ls s | ls b | ls f")

        elif cmd == "history":
            recent = history.get_recent(20)
            if recent:
                print("Recent queries:")
                for i, q in enumerate(recent, 1):
                    print(f"  {i}. {q}")
            else:
                print("No query history yet.")

        elif cmd == "help":
            print(
                """
Query Syntax Examples:
  Fiction                          Search all text fields for 'Fiction'
  subject:Fiction                  Search subjects field only
  shelf:Children                   Search bookshelves field
  author:Dickens                   Search authors field
  subject:"Science Fiction"        Exact phrase in subjects
  download_count>1000              Numeric comparison
  download_count:1000..5000        Range query
  id<100                           ID less than 100
  
Boolean operators:
  Fiction AND Space                Both terms must match
  Fiction OR Fantasy               Either term matches
  NOT Horror                       Exclude Horror
  (Fiction OR Fantasy) AND Space   Complex queries with parentheses
  
Combined examples:
  subject:Fiction AND download_count>5000
  (subject:Science OR subject:Space) AND NOT Horror
  author:Dickens AND download_count:100..1000
                """
            )

        elif cmd == "s":
            if not arg:
                print("Usage: s <query>")
                continue
            history.add(f"subject:{arg}")
            last_results = engine.evaluate(f"subject:{arg}")
            print(f"Found {len(last_results)} matches:")
            if not last_results.empty:
                cols = ["id", "title", "subjects"]
                print(_pandas_to_string(last_results, cols, 10))
                if len(last_results) > 10:
                    print(f"... {len(last_results) - 10} more")

        elif cmd == "b":
            if not arg:
                print("Usage: b <query>")
                continue
            history.add(f"bookshelves:{arg}")
            last_results = engine.evaluate(f"bookshelves:{arg}")
            print(f"Found {len(last_results)} matches:")
            if not last_results.empty:
                cols = ["id", "title", "bookshelves"]
                print(_pandas_to_string(last_results, cols, 10))
                if len(last_results) > 10:
                    print(f"... {len(last_results) - 10} more")

        elif cmd == "export_sets":
            out_dir = Path(arg) if arg else Path("data/meta/sets")
            save_canonical_sets(subjects, bookshelves, out_dir)

        elif cmd == "export_results":
            if not arg:
                print("Usage: export_results <filename.csv|parquet>")
                continue
            if last_results is None or last_results.empty:
                print("No results to export.")
                continue

            out_path = Path(arg)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if out_path.suffix == ".parquet":
                    _pandas_to_parquet(last_results, out_path)
                else:
                    _pandas_to_csv(last_results, out_path)
                print(f"Saved {len(last_results)} records to {out_path}")
            except Exception as e:
                print(f"Error saving file: {e}", file=sys.stderr)

        else:
            print(
                "Unknown command. Type 'help' for syntax or use: "
                "q, s, b, ls, history, export_sets, export_results, quit"
            )


if __name__ == "__main__":
    # Default path based on project structure
    default_path = Path("data/meta/gutenberg_books.parquet")
    interactive_explorer(default_path)
