from __future__ import annotations

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
    """Helper to evaluate boolean queries against a DataFrame column."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _tokenize(self, query: str) -> list[str]:
        # Pattern matches:
        # 1. ( or )
        # 2. Quoted strings (double or single)
        # 3. Non-whitespace, non-paren sequences (terms)
        pattern = r'\(|\)|"[^"]+"|\'[^\']+\'|[^\s()]+'
        return re.findall(pattern, query)

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

    def evaluate(self, query: str, column: str) -> pd.DataFrame:
        if not query.strip():
            return self.df

        if column not in self.df.columns:
            print(f"Column '{column}' not found.", file=sys.stderr)
            return pd.DataFrame()

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
                # Term
                term = item
                # Remove quotes if present
                if (term.startswith('"') and term.endswith('"')) or (
                    term.startswith("'") and term.endswith("'")
                ):
                    term = term[1:-1]

                # Create mask
                # Handle NaN by filling with False
                mask = (
                    self.df[column]
                    .astype(str)
                    .str.contains(term, case=False, regex=False, na=False)
                )
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

    print("\nCommands:")
    print("  s <query>          Search subjects (e.g., 'Science AND Fiction')")
    print("  b <query>          Search bookshelves (e.g., 'Children OR School')")
    print("  ls s               List all subjects")
    print("  ls b               List all bookshelves")
    print("  export_sets [dir]  Save unique subjects/bookshelves to files")
    print("  export_results <file> Save last search results to CSV/Parquet")
    print("  q                  Quit")

    while True:
        try:
            cmd_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd_input or cmd_input.lower() == "q":
            break

        parts = cmd_input.split(" ", 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "ls":
            if arg == "s":
                print("\n".join(sorted(subjects)))
            elif arg == "b":
                print("\n".join(sorted(bookshelves)))
            else:
                print("Usage: ls s | ls b")

        elif cmd == "s":
            if not arg:
                print("Usage: s <query>")
                continue
            last_results = engine.evaluate(arg, "subjects")
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
            last_results = engine.evaluate(arg, "bookshelves")
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
                "Unknown command. Available: s, b, ls, export_sets, export_results, q"
            )


if __name__ == "__main__":
    # Default path based on project structure
    default_path = Path("data/meta/gutenberg_books.parquet")
    interactive_explorer(default_path)
