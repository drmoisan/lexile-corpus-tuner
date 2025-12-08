"""Main application class for Gutenberg Query Builder.

Contains QueryBuilderApp which orchestrates the UI, data loading, and query execution.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any, cast

from ..explore_gutenberg import (  # noqa: TID252
    BooleanQueryEngine,
    get_canonical_sets,
)
from ..gutenberg_query_core import QueryGroupModel, SavedQuery  # noqa: TID252

if TYPE_CHECKING:
    import pandas as pd

from .constants import FIELD_TYPES, PARQUET_PATH, WINDOW_SIZE, WINDOW_TITLE
from .tk_helpers import (
    pandas_is_na,
    pandas_read_parquet,
    pandas_to_csv,
    pandas_to_parquet,
    tk_canvas_yview,
    tk_panedwindow_add,
    tk_treeview_xview,
    tk_treeview_yview,
)
from .widgets import QueryGroupWidget, ToolTip


class QueryBuilderApp:
    """Main application window for visual query building."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the query builder application.

        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_SIZE)

        # Application state
        self.df: pd.DataFrame | None = None
        self.subjects: set[str] = set()
        self.bookshelves: set[str] = set()
        self.current_query: QueryGroupModel = QueryGroupModel(logic="AND")
        self.last_results: pd.DataFrame | None = None
        self.current_file: Path | None = None
        self.root_group_widget: QueryGroupWidget | None = None

        # Initialize UI
        self._create_menu()
        self._create_layout()
        self._create_status_bar()

        # Load data
        self._load_data()

    def _create_menu(self) -> None:
        """Create menu bar with File, Edit, Help menus."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="New Query", command=self._new_query, accelerator="Ctrl+N"
        )
        file_menu.add_command(
            label="Open Query...", command=self._open_query, accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Save Query", command=self._save_query, accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Save Query As...",
            command=self._save_query_as,
            accelerator="Ctrl+Shift+S",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Export Results...",
            command=self._export_results,
            accelerator="Ctrl+E",
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit", command=self._on_closing, accelerator="Ctrl+Q"
        )

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(
            label="Copy Query String", command=self._copy_query_string
        )

        # Query menu
        query_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Query", menu=query_menu)
        query_menu.add_command(
            label="Run Query", command=self._run_query, accelerator="F5"
        )

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)

        # Bind keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self._new_query())
        self.root.bind("<Control-o>", lambda e: self._open_query())
        self.root.bind("<Control-s>", lambda e: self._save_query())
        self.root.bind("<Control-Shift-S>", lambda e: self._save_query_as())
        self.root.bind("<Control-e>", lambda e: self._export_results())
        self.root.bind("<Control-q>", lambda e: self._on_closing())
        self.root.bind("<F5>", lambda e: self._run_query())

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_layout(self) -> None:
        """Create main window layout with panes."""
        # Main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel: Field palette
        left_frame = ttk.Frame(main_paned, width=200)
        tk_panedwindow_add(main_paned, left_frame, weight=0)
        self._create_field_palette(left_frame)

        # Center panel: Query builder
        center_frame = ttk.Frame(main_paned)
        tk_panedwindow_add(main_paned, center_frame, weight=2)
        self._create_query_builder(center_frame)

        # Right panel: Results
        right_frame = ttk.Frame(main_paned, width=400)
        tk_panedwindow_add(main_paned, right_frame, weight=1)
        self._create_results_panel(right_frame)

    def _create_field_palette(self, parent: ttk.Frame) -> None:
        """Create field palette with click-to-add functionality.

        Args:
            parent: Parent frame widget
        """
        ttk.Label(parent, text="Available Fields", font=("Arial", 12, "bold")).pack(
            pady=5
        )

        ttk.Label(
            parent,
            text="Click to add constraint",
            font=("Arial", 9, "italic"),
            foreground="gray",
        ).pack(pady=(0, 10))

        # Field buttons with type indicators
        fields_frame = ttk.Frame(parent)
        fields_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        # Group fields by type for better organization
        text_fields = [(f, t) for f, t in FIELD_TYPES.items() if t == "text"]
        numeric_fields = [(f, t) for f, t in FIELD_TYPES.items() if t == "numeric"]
        boolean_fields = [(f, t) for f, t in FIELD_TYPES.items() if t == "boolean"]

        # Add fields in organized groups
        for field_name, field_type in text_fields + numeric_fields + boolean_fields:
            btn = ttk.Button(
                fields_frame,
                text=f"{field_name} ({field_type})",
                command=lambda f=field_name: self._add_constraint_from_field(f),
            )
            btn.pack(fill=tk.X, pady=2)
            # Add tooltip
            ToolTip(btn, f"Click to add '{field_name}' constraint to query")

    def _create_query_builder(self, parent: ttk.Frame) -> None:
        """Create query builder panel with root QueryGroupWidget.

        Args:
            parent: Parent frame widget
        """
        # Header
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(header_frame, text="Query Builder", font=("Arial", 12, "bold")).pack(
            side=tk.LEFT
        )

        run_btn = ttk.Button(header_frame, text="Run Query", command=self._run_query)
        run_btn.pack(side=tk.RIGHT, padx=2)
        ToolTip(run_btn, "Execute query and display results (F5)")

        # Query group container (scrollable)
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient=tk.VERTICAL,
            command=lambda *args: tk_canvas_yview(canvas, *args),
        )
        self.query_container = ttk.Frame(canvas)

        self.query_container.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.query_container, anchor=tk.NW)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Query string display
        query_string_frame = ttk.LabelFrame(parent, text="Generated Query", padding=5)
        query_string_frame.pack(fill=tk.X, padx=5, pady=5)

        # Query text with copy button
        text_button_frame = ttk.Frame(query_string_frame)
        text_button_frame.pack(fill=tk.X)

        self.query_text = tk.Text(
            text_button_frame,
            height=4,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 10),
        )
        self.query_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Copy button
        copy_btn = ttk.Button(
            text_button_frame,
            text="Copy",
            command=self._copy_query_to_clipboard,
            width=8,
        )
        copy_btn.pack(side=tk.RIGHT, padx=(5, 0))
        ToolTip(copy_btn, "Copy query string to clipboard")

    def _create_results_panel(self, parent: ttk.Frame) -> None:
        """Create results display panel.

        Args:
            parent: Parent frame widget
        """
        ttk.Label(parent, text="Query Results", font=("Arial", 12, "bold")).pack(pady=5)

        # Results metadata
        self.results_label = ttk.Label(parent, text="No results yet")
        self.results_label.pack(pady=5)

        # Results preview (Treeview)
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Placeholder for results tree
        self.results_tree = ttk.Treeview(tree_frame, show="headings")
        scrollbar_y = ttk.Scrollbar(
            tree_frame,
            orient=tk.VERTICAL,
            command=lambda *args: tk_treeview_yview(self.results_tree, *args),
        )
        scrollbar_x = ttk.Scrollbar(
            tree_frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: tk_treeview_xview(self.results_tree, *args),
        )

        self.results_tree.configure(
            yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set
        )

        self.results_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        scrollbar_x.grid(row=1, column=0, sticky="ew")

        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

    def _create_status_bar(self) -> None:
        """Create status bar at bottom of window."""
        self.status_bar = ttk.Label(
            self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_data(self) -> None:
        """Load Gutenberg metadata from parquet file."""
        try:
            if not PARQUET_PATH.exists():
                messagebox.showerror(
                    "Data Not Found",
                    f"Gutenberg data not found at {PARQUET_PATH}.\n"
                    "Please run python -m lexile_corpus_tuner.lexile_scoring_model."
                    "pipeline_scripts.build_gutenberg_id_list first.",
                )
                self.status_bar.config(text="Error: Data file not found")
                return

            self.df = pandas_read_parquet(PARQUET_PATH)
            self._load_canonical_sets()
            self.status_bar.config(
                text=f"Loaded {len(self.df)} books from {PARQUET_PATH}"
            )

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load data:\n{e}")
            self.status_bar.config(text=f"Error: {e}")

    def _load_canonical_sets(self) -> None:
        """Extract unique subjects and bookshelves from DataFrame."""
        if self.df is None:
            return

        self.subjects, self.bookshelves = get_canonical_sets(self.df)

        # Initialize root query group widget after data is loaded
        self._initialize_root_group()

    def _initialize_root_group(self) -> None:
        """Initialize the root QueryGroupWidget."""
        if self.root_group_widget:
            self.root_group_widget.destroy()

        self.root_group_widget = QueryGroupWidget(
            parent=self.query_container,
            model=self.current_query,
            subjects=self.subjects,
            bookshelves=self.bookshelves,
            on_change=self._update_query_from_root,
            nesting_level=0,
        )
        self.root_group_widget.pack(fill=tk.BOTH, expand=True)
        self._update_query_from_root()

    def _update_query_from_root(self) -> None:
        """Update current_query from root QueryGroupWidget."""
        if self.root_group_widget:
            self.current_query = self.root_group_widget.get_model()
            self._update_query_display()

    def _add_constraint_from_field(self, field_name: str) -> None:
        """Add constraint from field palette drag/click.

        Args:
            field_name: Name of the field to add
        """
        # Note: With QueryGroupWidget, users add constraints via the group's controls
        # This is kept for field palette button functionality
        # We could add to root group programmatically, but it's better UX
        # to let users use the "+ Constraint" button in the group
        messagebox.showinfo(
            "Add Constraint",
            f'Click "+ Constraint" in the query builder to add a '
            f"{field_name} constraint.",
        )

    def _new_query(self) -> None:
        """Create a new query (reset current)."""
        if messagebox.askyesno("New Query", "Discard current query and create new?"):
            self.current_query = QueryGroupModel(logic="AND")
            self.current_file = None
            self._initialize_root_group()
            self.status_bar.config(text="New query created")

    def _open_query(self) -> None:
        """Open a saved query from file."""
        filepath = filedialog.askopenfilename(
            title="Open Query",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if filepath:
            try:
                with open(filepath, encoding="utf-8") as f:
                    json_str = f.read()
                saved_query = SavedQuery.from_json(json_str)
                self.current_query = saved_query.to_query_group()
                self.current_file = Path(filepath)
                self._initialize_root_group()
                self.status_bar.config(text=f"Opened query from {filepath}")
            except Exception as e:
                messagebox.showerror("Open Error", f"Failed to open query:\n{e}")

    def _save_query(self) -> None:
        """Save current query to file."""
        if self.current_file:
            self._save_query_to_file(self.current_file)
        else:
            self._save_query_as()

    def _save_query_as(self) -> None:
        """Save current query to a new file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Query As",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if filepath:
            self._save_query_to_file(Path(filepath))

    def _save_query_to_file(self, filepath: Path) -> None:
        """Save query to specified file."""
        try:
            saved_query = SavedQuery.from_query_group(self.current_query)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(saved_query.to_json())
            self.current_file = filepath
            self.status_bar.config(text=f"Saved query to {filepath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save query:\n{e}")

    def _export_results(self) -> None:
        """Export query results to file."""
        if self.last_results is None or self.last_results.empty:
            messagebox.showwarning(
                "No Results", "Run a query first before exporting results"
            )
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".csv",
            filetypes=[
                ("CSV Files", "*.csv"),
                ("Parquet Files", "*.parquet"),
                ("All Files", "*.*"),
            ],
        )
        if filepath:
            try:
                path = Path(filepath)
                if path.suffix == ".parquet":
                    pandas_to_parquet(self.last_results, path)
                else:
                    pandas_to_csv(self.last_results, path)
                self.status_bar.config(
                    text=f"Exported {len(self.last_results)} results to {filepath}"
                )
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{e}")

    def _copy_query_string(self) -> None:
        """Copy generated query string to clipboard."""
        query_str = self.current_query.to_query_string()
        self.root.clipboard_clear()
        self.root.clipboard_append(query_str)
        self.status_bar.config(text="Query string copied to clipboard")

    def _copy_query_to_clipboard(self) -> None:
        """Copy generated query to clipboard (called by Copy button)."""
        query_str = self.current_query.to_query_string()
        if query_str:
            self.root.clipboard_clear()
            self.root.clipboard_append(query_str)
            self.status_bar.config(text="Query copied to clipboard")
        else:
            self.status_bar.config(text="No query to copy")

    def _run_query(self) -> None:
        """Execute current query and display results."""
        query_str = self.current_query.to_query_string()

        if not query_str:
            messagebox.showinfo("Empty Query", "Please build a query first.")
            return

        self.status_bar.config(text="Executing query...")
        self.root.update_idletasks()

        try:
            # Create query engine and execute
            # Type narrowing for Pyright
            df = self.df
            if df is None:
                # This should never happen due to earlier check
                raise ValueError("DataFrame is None")
            engine = BooleanQueryEngine(df)
            results = engine.evaluate(query_str)

            # Store results
            self.last_results = results

            # Display results
            self._display_results(results, query_str)

            # Update status
            self.status_bar.config(text=f"Query executed: {len(results)} results found")

        except Exception as e:
            messagebox.showerror(
                "Query Execution Error", f"Failed to execute query:\n{e}"
            )
            self.status_bar.config(text="Query execution failed")

    def _display_results(self, results: pd.DataFrame, query_str: str) -> None:
        """Display query results in the results panel.

        Args:
            results: DataFrame with query results
            query_str: Query string used (for display)
        """
        # Update results label
        count = len(results)
        query_display = f"{query_str[:100]}..." if len(query_str) > 100 else query_str
        self.results_label.config(
            text=f"Results: {count} books matching query\nQuery: {query_display}"
        )

        # Clear existing tree
        self.results_tree.delete(*self.results_tree.get_children())

        if count == 0:
            return

        # Limit display to first 100 rows
        display_df = results.head(100)

        # Configure columns (show subset of interesting fields)
        display_columns = ["id", "title", "authors", "download_count"]
        available_columns = [
            col for col in display_columns if col in display_df.columns
        ]

        self.results_tree["columns"] = available_columns
        self.results_tree["show"] = "headings"

        # Configure column headings and widths
        for col in available_columns:
            self.results_tree.heading(col, text=col.title())
            if col == "id":
                self.results_tree.column(col, width=80, anchor=tk.CENTER)
            elif col == "download_count":
                self.results_tree.column(col, width=120, anchor=tk.CENTER)
            elif col == "title":
                self.results_tree.column(col, width=400, anchor=tk.W)
            else:
                self.results_tree.column(col, width=200, anchor=tk.W)

        # Insert rows
        for _, row_any in display_df.iterrows():
            row = cast("Any", row_any)
            values: list[str] = []
            for col in available_columns:
                try:
                    val = row[col]
                    # Convert to string, using empty string for missing values
                    values.append(
                        ""
                        if (
                            val is None
                            or (isinstance(val, float) and pandas_is_na(val))
                        )
                        else str(val)
                    )
                except Exception:
                    values.append("")
            self.results_tree.insert("", tk.END, values=values)

        # Show warning if results truncated
        if count > 100:
            query_display = f"{query_str[:80]}..." if len(query_str) > 80 else query_str
            truncated_text = (
                f"Results: {count} books found (showing first 100)\n"
                f"Query: {query_display}"
            )
            self.results_label.config(text=truncated_text)

    def _update_query_display(self) -> None:
        """Update the query string display with real-time formatting."""
        query_str = self.current_query.to_query_string()
        self.query_text.config(state=tk.NORMAL)
        self.query_text.delete(1.0, tk.END)

        if query_str:
            # Format query string with line breaks for better readability
            formatted_query = self._format_query_string(query_str)
            self.query_text.insert(1.0, formatted_query)
        else:
            self.query_text.insert(1.0, "(empty query)")

        self.query_text.config(state=tk.DISABLED)

    def _format_query_string(self, query: str) -> str:
        """Format query string for better readability.

        Args:
            query: Raw query string

        Returns:
            Formatted query string with line breaks and indentation
        """
        # Add line breaks after AND/OR for complex queries
        # but keep it simple for now
        formatted = query

        # Add line breaks before AND/OR if query is long
        if len(query) > 60:
            formatted = formatted.replace(" AND ", "\nAND ")
            formatted = formatted.replace(" OR ", "\nOR ")

        return formatted

    def _show_about(self) -> None:
        """Show about dialog with keyboard shortcuts."""
        messagebox.showinfo(
            "About",
            f"{WINDOW_TITLE}\n\n"
            "Visual query builder for Gutenberg metadata.\n\n"
            "Features:\n"
            "• Click-to-add field selection\n"
            "• Field-specific operators & value inputs\n"
            "• Multi-select for subjects/bookshelves\n"
            "• Nested query groups with AND/OR logic\n"
            "• Query persistence and results export\n\n"
            "Keyboard Shortcuts:\n"
            "• F5 - Run Query\n"
            "• Ctrl+N - New Query\n"
            "• Ctrl+O - Open Query\n"
            "• Ctrl+S - Save Query\n"
            "• Ctrl+Shift+S - Save Query As\n"
            "• Ctrl+E - Export Results\n"
            "• Ctrl+Q - Quit\n\n"
            "Part of lexile-corpus-tuner project.",
        )

    def _on_closing(self) -> None:
        """Handle window close event."""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()

    def run(self) -> None:
        """Start the application main loop."""
        self.root.mainloop()
