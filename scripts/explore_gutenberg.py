from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def get_canonical_sets(df: pd.DataFrame) -> tuple[set[str], set[str]]:
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
        # We use astype(str) to handle potential non-string types safely
        series = df[column].astype(str)
        exploded = series.str.split(";").explode()

        if exploded.empty:
            return set()

        # Strip whitespace and get unique values
        # Filter out 'nan' which results from astype(str) on None/NaN
        items = exploded.dropna().astype(str).str.strip()
        unique_items = set(items.unique())

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
        # 2. Field operators: field:value, field>value, field<value, etc.
        # 3. Quoted strings (double or single)
        # 4. Non-whitespace, non-paren sequences (terms)
        pattern = (
            r"\(|\)|[a-zA-Z_][a-zA-Z0-9_]*(?:[><=:][^\s()]+|:[^\s()]+)"
            r'|"[^"]+"|\'[^\']+\'|[^\s()]+'
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
                    mask |= (
                        self.df[col]
                        .astype(str)
                        .str.contains(search_term, case=False, regex=False, na=False)
                    )
            return mask

        # Field-specific query
        if field not in self.df.columns:
            print(f"Warning: Unknown field '{field}', ignoring.", file=sys.stderr)
            return pd.Series([True] * len(self.df), index=self.df.index)

        col = self.df[field]

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
                return (col >= min_num) & (col <= max_num)
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
                if operator == ">":
                    return col > num_val
                elif operator == "<":
                    return col < num_val
                elif operator == ">=":
                    return col >= num_val
                elif operator == "<=":
                    return col <= num_val
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
            # Case-insensitive exact match
            return col.astype(str).str.lower() == search_term.lower()
        else:
            # Substring match
            return col.astype(str).str.contains(
                search_term, case=False, regex=False, na=False
            )

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
        return self.df[final_mask]  # type: ignore[return-value]


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
        print("Please run 'scripts/build_gutenberg_id_list.py' first.", file=sys.stderr)
        return

    print(f"Loading {parquet_path}...")
    try:
        df: pd.DataFrame = pd.read_parquet(parquet_path)  # type: ignore[assignment]
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
                print(last_results[cols].head(10).to_string(index=False))  # type: ignore[call-overload]
                if len(last_results) > 10:
                    print(f"... {len(last_results) - 10} more")

        elif cmd == "ls":
            if arg == "s":
                print("\n".join(sorted(subjects)))
            elif arg == "b":
                print("\n".join(sorted(bookshelves)))
            elif arg == "f":
                print("Available fields:")
                for col in df.columns:
                    print(f"  {col}")
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
                print(last_results[cols].head(10).to_string(index=False))  # type: ignore[call-overload]
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
                print(last_results[cols].head(10).to_string(index=False))  # type: ignore[call-overload]
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
                    last_results.to_parquet(out_path, index=False)  # type: ignore[call-overload]
                else:
                    last_results.to_csv(out_path, index=False)
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
