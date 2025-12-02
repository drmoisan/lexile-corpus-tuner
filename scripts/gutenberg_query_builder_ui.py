"""Gutenberg Query Builder - Visual Query Construction UI.

This module provides a lightweight Tkinter-based GUI for building and executing
queries against the Gutenberg metadata database. It supports:
- Visual query construction with drag-and-drop
- Field-specific operators and values
- Grouping with AND/OR logic
- Multi-select for subjects/bookshelves
- Query persistence (save/load)
- Results export (CSV/Parquet)
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Any

import pandas as pd

# Import query engine from explore_gutenberg
sys.path.insert(0, str(Path(__file__).parent))
from explore_gutenberg import BooleanQueryEngine  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable


# =============================================================================
# Third-Party Operations Section - Isolated Type Suppressions
# =============================================================================
# All tkinter widget operations and pandas DataFrame operations with
# incomplete type stubs are isolated in this section.
# Application logic remains fully typed.


def _tk_listbox_curselection(listbox: tk.Listbox) -> tuple[int, ...]:  # type: ignore[misc]
    """Get selected indices from Listbox.

    Isolated tkinter operation: tkinter stubs incomplete for curselection().

    Args:
        listbox: Listbox widget

    Returns:
        Tuple of selected indices
    """
    return listbox.curselection()  # type: ignore[no-any-return]


def _tk_listbox_get(listbox: tk.Listbox, index: int) -> str:
    """Get item at index from Listbox.

    Isolated tkinter operation: tkinter stubs incomplete for get().

    Args:
        listbox: Listbox widget
        index: Item index

    Returns:
        Item text
    """
    return listbox.get(index)  # type: ignore[no-any-return]


def _tk_listbox_yview(listbox: tk.Listbox, *args: Any) -> None:
    """Scroll Listbox vertically.

    Isolated tkinter operation: tkinter stubs incomplete for yview().

    Args:
        listbox: Listbox widget
        args: Scroll arguments
    """
    listbox.yview(*args)  # type: ignore[no-untyped-call]


def _tk_canvas_yview(canvas: tk.Canvas, *args: Any) -> None:
    """Scroll Canvas vertically.

    Isolated tkinter operation: tkinter stubs incomplete for yview().

    Args:
        canvas: Canvas widget
        args: Scroll arguments
    """
    canvas.yview(*args)  # type: ignore[no-untyped-call]


def _tk_treeview_xview(treeview: ttk.Treeview, *args: Any) -> None:
    """Scroll Treeview horizontally.

    Isolated tkinter operation: tkinter stubs incomplete for xview().

    Args:
        treeview: Treeview widget
        args: Scroll arguments
    """
    treeview.xview(*args)  # type: ignore[no-untyped-call]


def _tk_treeview_yview(treeview: ttk.Treeview, *args: Any) -> None:
    """Scroll Treeview vertically.

    Isolated tkinter operation: tkinter stubs incomplete for yview().

    Args:
        treeview: Treeview widget
        args: Scroll arguments
    """
    treeview.yview(*args)  # type: ignore[no-untyped-call]


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


def _pandas_is_na(value: Any) -> bool:
    """Check if value is NA/NaN.

    Isolated pandas operation: pandas-stubs incomplete for isna().

    Args:
        value: Value to check

    Returns:
        True if NA/NaN, False otherwise
    """
    return pd.isna(value)  # type: ignore[no-any-return]


def _pandas_get_column(df: pd.DataFrame, column: str) -> Any:  # type: ignore[type-arg,misc]
    """Get column from DataFrame.

    Isolated pandas operation: pandas-stubs incomplete for column access.

    Args:
        df: DataFrame
        column: Column name

    Returns:
        Series or value
    """
    return df[column]  # type: ignore[no-any-return]


def _tk_panedwindow_add(paned: ttk.PanedWindow, child: tk.Widget, **kw: Any) -> None:
    """Add child widget to PanedWindow.

    Isolated tkinter operation: tkinter stubs incomplete for add().

    Args:
        paned: PanedWindow widget
        child: Child widget to add
        kw: Additional keyword arguments
    """
    paned.add(child, **kw)  # type: ignore[no-untyped-call]


# =============================================================================
# End Third-Party Operations Section
# =============================================================================


class ToolTip:
    """Simple tooltip widget that appears on hover.

    Creates a small label that appears near the widget when the mouse hovers over it.
    """

    def __init__(self, widget: tk.Widget, text: str):
        """Initialize tooltip.

        Args:
            widget: Widget to attach tooltip to
            text: Tooltip text to display
        """
        self.widget = widget
        self.text = text
        self.tipwindow: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show_tip)
        self.widget.bind("<Leave>", self._hide_tip)

    def _show_tip(self, event: Any = None) -> None:
        """Show the tooltip."""
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", 9),
            padx=5,
            pady=3,
        )
        label.pack()

    def _hide_tip(self, event: Any = None) -> None:
        """Hide the tooltip."""
        if self.tipwindow:
            self.tipwindow.destroy()
        self.tipwindow = None


# Constants
WINDOW_TITLE = "Gutenberg Query Builder"
WINDOW_SIZE = "1400x900"
PARQUET_PATH = Path("data/gutenberg_books.parquet")

# Field metadata
FIELD_TYPES = {
    "id": "numeric",
    "title": "text",
    "authors": "text",
    "subjects": "text",
    "bookshelves": "text",
    "languages": "text",
    "download_count": "numeric",
    "media_type": "text",
    "copyright": "boolean",
}

TEXT_OPERATORS = ["contains", "=", "!="]
NUMERIC_OPERATORS = [">", "<", ">=", "<=", "=", "!=", "range"]
BOOLEAN_OPERATORS = ["=", "!="]


@dataclass
class QueryConstraintModel:
    """Model for a single query constraint (field:operator:value)."""

    field: str
    operator: str
    value: str | list[str]

    def to_query_string(self) -> str:
        """Convert to query syntax compatible with BooleanQueryEngine.

        Returns:
            Query string like 'field:value', 'field>100', 'field:100..500'
        """
        if isinstance(self.value, list):
            # Multi-select converts to OR clause
            terms = [f'{self.field}:"{v}"' for v in self.value]
            return f"({' OR '.join(terms)})"

        if self.operator == "contains":
            return f"{self.field}:{self.value}"
        elif self.operator == "=":
            return f'{self.field}="{self.value}"'
        elif self.operator == "!=":
            return f'NOT {self.field}="{self.value}"'
        elif self.operator == "range":
            return f"{self.field}:{self.value}"
        else:
            # >, <, >=, <=
            return f"{self.field}{self.operator}{self.value}"


@dataclass
class QueryGroupModel:
    """Model for a group of constraints with AND/OR logic."""

    logic: str  # 'AND' or 'OR'
    constraints: list[QueryConstraintModel | QueryGroupModel] = field(  # type: ignore[misc]
        default_factory=list  # type: ignore[misc]
    )

    def to_query_string(self) -> str:
        """Convert to nested query syntax with parentheses.

        Returns:
            Query string like '(a AND b) OR (c AND d)'
        """
        if not self.constraints:
            return ""

        parts: list[str] = [c.to_query_string() for c in self.constraints]
        parts = [p for p in parts if p]  # Filter empty strings

        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]

        joined = f" {self.logic} ".join(parts)
        return f"({joined})"


@dataclass
class SavedQuery:
    """Persistent query format for save/load."""

    version: str
    created: str
    modified: str
    query: dict[str, Any]  # Serialized QueryGroupModel

    @classmethod
    def from_query_group(cls, group: QueryGroupModel) -> SavedQuery:
        """Create SavedQuery from QueryGroupModel."""
        now = datetime.now().isoformat()
        return cls(
            version="1.0",
            created=now,
            modified=now,
            query=cls._serialize_group(group),
        )

    @staticmethod
    def _serialize_group(
        group: QueryGroupModel | QueryConstraintModel,
    ) -> dict[str, Any]:
        """Recursively serialize query structure."""
        if isinstance(group, QueryConstraintModel):
            return {
                "type": "constraint",
                "field": group.field,
                "operator": group.operator,
                "value": group.value,
            }
        else:
            constraints_serialized: list[dict[str, Any]] = [
                SavedQuery._serialize_group(c) for c in group.constraints
            ]
            return {
                "type": "group",
                "logic": group.logic,
                "constraints": constraints_serialized,
            }

    @classmethod
    def from_json(cls, json_str: str) -> SavedQuery:
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=2)

    def to_query_group(self) -> QueryGroupModel:
        """Deserialize query structure to QueryGroupModel.

        Raises:
            ValueError: If the root query is not a QueryGroupModel.
        """
        result = self._deserialize_group(self.query)
        if not isinstance(result, QueryGroupModel):
            # Root should always be a group, but wrap single constraint if needed
            result = QueryGroupModel(logic="AND", constraints=[result])
        return result

    @staticmethod
    def _deserialize_group(
        data: dict[str, Any],
    ) -> QueryGroupModel | QueryConstraintModel:
        """Recursively deserialize query structure."""
        if data["type"] == "constraint":
            return QueryConstraintModel(
                field=data["field"],
                operator=data["operator"],
                value=data["value"],
            )
        # Must be a group
        return QueryGroupModel(
            logic=data["logic"],
            constraints=[SavedQuery._deserialize_group(c) for c in data["constraints"]],
        )


class QueryConstraintWidget(ttk.Frame):
    """Widget for a single query constraint (field:operator:value)."""

    def __init__(
        self,
        parent: tk.Widget,
        model: QueryConstraintModel | None = None,
        subjects: set[str] | None = None,
        bookshelves: set[str] | None = None,
        on_delete: Callable[[QueryConstraintWidget], None] | None = None,
        on_change: Callable[[], None] | None = None,
    ) -> None:
        """Initialize constraint widget.

        Args:
            parent: Parent widget
            model: Initial constraint model (optional)
            subjects: Set of available subjects for dropdown
            bookshelves: Set of available bookshelves for dropdown
            on_delete: Callback when delete button clicked
            on_change: Callback when constraint changes
        """
        super().__init__(parent, relief=tk.RAISED, borderwidth=1, padding=5)

        self.subjects = subjects or set()
        self.bookshelves = bookshelves or set()
        self.on_delete_callback = on_delete
        self.on_change_callback = on_change

        # Constraint data
        self.field_var = tk.StringVar(value=model.field if model else "title")
        self.operator_var = tk.StringVar(value=model.operator if model else "contains")
        self.value_var = tk.StringVar(value=str(model.value) if model else "")

        # Build UI
        self._create_widgets()

        # Bind change events
        self.field_var.trace_add("write", self._on_field_change)
        self.operator_var.trace_add("write", self._on_constraint_change)
        self.value_var.trace_add("write", self._on_constraint_change)

    def _create_widgets(self) -> None:
        """Create constraint UI widgets."""
        # Field dropdown
        ttk.Label(self, text="Field:").grid(row=0, column=0, sticky=tk.W, padx=2)
        self.field_combo = ttk.Combobox(
            self,
            textvariable=self.field_var,
            values=list(FIELD_TYPES.keys()),
            state="readonly",
            width=15,
        )
        self.field_combo.grid(row=0, column=1, padx=2)

        # Operator dropdown
        ttk.Label(self, text="Operator:").grid(row=0, column=2, sticky=tk.W, padx=2)
        self.operator_combo = ttk.Combobox(
            self,
            textvariable=self.operator_var,
            state="readonly",
            width=10,
        )
        self.operator_combo.grid(row=0, column=3, padx=2)

        # Value input (will be replaced based on field type)
        ttk.Label(self, text="Value:").grid(row=0, column=4, sticky=tk.W, padx=2)
        self.value_widget_frame = ttk.Frame(self)
        self.value_widget_frame.grid(row=0, column=5, padx=2)

        # Delete button
        self.delete_btn = ttk.Button(
            self,
            text="✖",
            width=3,
            command=self._on_delete_clicked,
        )
        self.delete_btn.grid(row=0, column=6, padx=2)

        # Initialize operator list and value widget
        self._update_operators()
        self._update_value_widget()

    def _update_operators(self) -> None:
        """Update operator dropdown based on selected field type."""
        field = self.field_var.get()
        field_type = FIELD_TYPES.get(field, "text")

        if field_type == "numeric":
            operators = NUMERIC_OPERATORS
        elif field_type == "boolean":
            operators = BOOLEAN_OPERATORS
        else:
            operators = TEXT_OPERATORS

        self.operator_combo["values"] = operators

        # Reset operator if current one is invalid
        if self.operator_var.get() not in operators:
            self.operator_var.set(operators[0])

    def _update_value_widget(self) -> None:
        """Update value input widget based on field and operator."""
        # Clear existing widget
        for widget in self.value_widget_frame.winfo_children():
            widget.destroy()

        field = self.field_var.get()
        field_type = FIELD_TYPES.get(field, "text")
        operator = self.operator_var.get()

        # Multi-select for subjects/bookshelves
        if field == "subjects" and operator == "contains":
            self._create_multiselect_widget(sorted(self.subjects))
        elif field == "bookshelves" and operator == "contains":
            self._create_multiselect_widget(sorted(self.bookshelves))
        # Boolean field
        elif field_type == "boolean":
            self._create_boolean_widget()
        # Numeric field
        elif field_type == "numeric":
            if operator == "range":
                self._create_range_widget()
            else:
                self._create_numeric_widget()
        # Text field (default)
        else:
            self._create_text_widget()

    def _create_text_widget(self) -> None:
        """Create text entry widget."""
        entry = ttk.Entry(
            self.value_widget_frame,
            textvariable=self.value_var,
            width=25,
        )
        entry.pack()

    def _create_numeric_widget(self) -> None:
        """Create numeric spinbox widget with validation."""
        spinbox = ttk.Spinbox(
            self.value_widget_frame,
            textvariable=self.value_var,
            from_=0,
            to=999999,
            width=15,
            validate="key",
            validatecommand=(
                self.value_widget_frame.register(self._validate_numeric),
                "%P",
            ),
        )
        spinbox.pack()
        ToolTip(spinbox, "Enter a numeric value (0-999999)")

    def _validate_numeric(self, value: str) -> bool:
        """Validate numeric input.

        Args:
            value: Input value to validate

        Returns:
            True if valid, False otherwise
        """
        if value == "":
            return True
        try:
            num = int(value)
            return 0 <= num <= 999999
        except ValueError:
            return False

    def _create_range_widget(self) -> None:
        """Create range input widget (min..max) with validation."""
        frame = ttk.Frame(self.value_widget_frame)
        frame.pack()

        # Parse existing value if present
        value = self.value_var.get()
        min_val, max_val = "0", "1000"
        if ".." in value:
            parts = value.split("..")
            if len(parts) == 2:
                min_val, max_val = parts[0], parts[1]

        min_var = tk.StringVar(value=min_val)
        max_var = tk.StringVar(value=max_val)

        min_spin = ttk.Spinbox(
            frame,
            textvariable=min_var,
            from_=0,
            to=999999,
            width=8,
            validate="key",
            validatecommand=(frame.register(self._validate_numeric), "%P"),
        )
        min_spin.pack(side=tk.LEFT, padx=2)
        ToolTip(min_spin, "Minimum value")

        ttk.Label(frame, text="..").pack(side=tk.LEFT)

        max_spin = ttk.Spinbox(
            frame,
            textvariable=max_var,
            from_=0,
            to=999999,
            width=8,
            validate="key",
            validatecommand=(frame.register(self._validate_numeric), "%P"),
        )
        max_spin.pack(side=tk.LEFT, padx=2)
        ToolTip(max_spin, "Maximum value")

        # Update value_var when either spinbox changes
        def update_range(*args: Any) -> None:
            self.value_var.set(f"{min_var.get()}..{max_var.get()}")

        min_var.trace_add("write", update_range)
        max_var.trace_add("write", update_range)

    def _create_boolean_widget(self) -> None:
        """Create boolean radio button widget."""
        frame = ttk.Frame(self.value_widget_frame)
        frame.pack()

        # Use a separate variable for the radio buttons
        bool_var = tk.StringVar(value=self.value_var.get() or "true")

        ttk.Radiobutton(frame, text="True", variable=bool_var, value="true").pack(
            side=tk.LEFT, padx=2
        )
        ttk.Radiobutton(frame, text="False", variable=bool_var, value="false").pack(
            side=tk.LEFT, padx=2
        )

        # Update value_var when radio selection changes
        def update_bool(*args: Any) -> None:
            self.value_var.set(bool_var.get())

        bool_var.trace_add("write", update_bool)

    def _create_multiselect_widget(self, items: list[str]) -> None:
        """Create multi-select listbox widget.

        Args:
            items: List of items to display
        """
        frame = ttk.Frame(self.value_widget_frame)
        frame.pack()

        # Button to open multi-select dialog
        selected_count = (
            len(self.value_var.get().split(";")) if self.value_var.get() else 0
        )
        btn_text = f"Select... ({selected_count} selected)"

        btn = ttk.Button(
            frame,
            text=btn_text,
            command=lambda: self._show_multiselect_dialog(items),
        )
        btn.pack()

        # Store button reference for updating
        self.multiselect_btn = btn

    def _show_multiselect_dialog(self, items: list[str]) -> None:
        """Show dialog for multi-select.

        Args:
            items: List of items to display
        """
        dialog = tk.Toplevel(self)
        dialog.title("Select Values")
        dialog.geometry("400x500")

        # Search box
        search_var = tk.StringVar()
        search_frame = ttk.Frame(dialog)
        search_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT)
        ttk.Entry(search_frame, textvariable=search_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )

        # Listbox with scrollbar
        list_frame = ttk.Frame(dialog)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        listbox = tk.Listbox(
            list_frame,
            selectmode=tk.MULTIPLE,
            yscrollcommand=scrollbar.set,
        )
        scrollbar.config(command=lambda *args: _tk_listbox_yview(listbox, *args))

        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Populate listbox
        def update_listbox(*args: Any) -> None:
            listbox.delete(0, tk.END)
            search_term = search_var.get().lower()
            for item in items:
                if search_term in item.lower():
                    listbox.insert(tk.END, item)

        update_listbox()
        search_var.trace_add("write", update_listbox)

        # Pre-select currently selected items
        current_values = self.value_var.get().split(";") if self.value_var.get() else []
        for i, item in enumerate(items):
            if item in current_values:
                listbox.selection_set(i)

        # Buttons
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        def on_ok() -> None:
            selected_indices = _tk_listbox_curselection(listbox)
            selected_items = [_tk_listbox_get(listbox, i) for i in selected_indices]
            self.value_var.set(";".join(selected_items))

            # Update button text
            if hasattr(self, "multiselect_btn"):
                self.multiselect_btn.config(
                    text=f"Select... ({len(selected_items)} selected)"
                )

            dialog.destroy()

        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=2
        )

        # Select All / Clear All buttons
        def select_all() -> None:
            listbox.selection_set(0, tk.END)

        def clear_all() -> None:
            listbox.selection_clear(0, tk.END)

        ttk.Button(btn_frame, text="Select All", command=select_all).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Clear All", command=clear_all).pack(
            side=tk.LEFT, padx=2
        )

    def _on_field_change(self, *args: Any) -> None:
        """Handle field dropdown change."""
        self._update_operators()
        self._update_value_widget()
        self._on_constraint_change()

    def _on_constraint_change(self, *args: Any) -> None:
        """Handle any constraint change."""
        if self.on_change_callback:
            self.on_change_callback()

    def _on_delete_clicked(self) -> None:
        """Handle delete button click."""
        if self.on_delete_callback:
            self.on_delete_callback(self)

    def get_model(self) -> QueryConstraintModel:
        """Get current constraint as a model.

        Returns:
            QueryConstraintModel representing current state
        """
        value = self.value_var.get()

        # Convert multi-select semicolon-separated to list
        if ";" in value and value:
            value_list = [v.strip() for v in value.split(";") if v.strip()]
            return QueryConstraintModel(
                field=self.field_var.get(),
                operator=self.operator_var.get(),
                value=value_list,
            )

        return QueryConstraintModel(
            field=self.field_var.get(),
            operator=self.operator_var.get(),
            value=value,
        )


class QueryGroupWidget(ttk.Frame):
    """Widget for a query group with AND/OR logic and nested constraints/groups.

    A QueryGroup contains:
    - Logic selector (AND/OR radio buttons)
    - List of child widgets (QueryConstraintWidget or nested QueryGroupWidget)
    - Controls for adding constraints, grouping, and ungrouping
    - Visual hierarchy through indentation
    """

    def __init__(
        self,
        parent: tk.Widget,
        model: QueryGroupModel,
        subjects: set[str],
        bookshelves: set[str],
        on_change: Callable[[], None],
        nesting_level: int = 0,
    ) -> None:
        """Initialize query group widget.

        Args:
            parent: Parent widget
            model: QueryGroupModel to display
            subjects: Available subject values for multi-select
            bookshelves: Available bookshelf values for multi-select
            on_change: Callback when group or children change
            nesting_level: Nesting depth (0 = root, increases with nesting)
        """
        super().__init__(parent)
        self.model = model
        self.subjects = subjects
        self.bookshelves = bookshelves
        self.on_change = on_change
        self.nesting_level = nesting_level
        self.child_widgets: list[QueryConstraintWidget | QueryGroupWidget] = []

        self._create_widgets()
        self._load_children_from_model()

    def _create_widgets(self) -> None:
        """Create the group UI: logic selector, children container, controls."""
        # Main container with padding based on nesting level
        indent = self.nesting_level * 20
        self.configure(padding=(indent, 5, 5, 5), relief="solid", borderwidth=1)

        # Header frame: logic selector + controls
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # Logic selector (AND/OR)
        self.logic_var = tk.StringVar(value=self.model.logic)
        self.logic_var.trace_add(
            "write", lambda *args: self._on_logic_change()  # type: ignore[misc]
        )

        ttk.Label(header_frame, text="Logic:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(
            header_frame, text="AND", variable=self.logic_var, value="AND"
        ).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(
            header_frame, text="OR", variable=self.logic_var, value="OR"
        ).pack(side=tk.LEFT, padx=2)

        # Separator
        ttk.Separator(header_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        # Control buttons
        add_constraint_btn = ttk.Button(
            header_frame, text="+ Constraint", command=self._add_constraint, width=12
        )
        add_constraint_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(add_constraint_btn, "Add a new constraint to this group")

        add_group_btn = ttk.Button(
            header_frame, text="+ Group", command=self._add_group, width=10
        )
        add_group_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(add_group_btn, "Add a nested group with its own logic")

        if self.nesting_level > 0:
            # Show ungroup button only for nested groups
            ungroup_btn = ttk.Button(
                header_frame, text="Ungroup", command=self._ungroup, width=10
            )
            ungroup_btn.pack(side=tk.LEFT, padx=2)
            ToolTip(ungroup_btn, "Remove this group and move children to parent")

        # Children container (scrollable)
        self.children_frame = ttk.Frame(self)
        self.children_frame.grid(row=1, column=0, sticky="nsew")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

    def _load_children_from_model(self) -> None:
        """Load child widgets from the current model."""
        for item in self.model.constraints:
            if isinstance(item, QueryConstraintModel):
                self._add_constraint_widget(item)
            else:
                # Must be QueryGroupModel (union type)
                self._add_group_widget(item)

    def _add_constraint(self) -> None:
        """Add a new constraint widget."""
        constraint_model = QueryConstraintModel(
            field="title", operator="contains", value=""
        )
        self._add_constraint_widget(constraint_model)
        self.on_change()

    def _add_constraint_widget(
        self, model: QueryConstraintModel
    ) -> QueryConstraintWidget:
        """Add a constraint widget with the given model.

        Args:
            model: QueryConstraintModel to display

        Returns:
            The created QueryConstraintWidget
        """
        widget = QueryConstraintWidget(
            parent=self.children_frame,
            model=model,
            subjects=self.subjects,
            bookshelves=self.bookshelves,
            on_delete=lambda w=None: self._delete_child(widget),  # type: ignore[misc]
            on_change=self.on_change,
        )
        widget.pack(fill=tk.X, pady=2)
        self.child_widgets.append(widget)
        return widget

    def _add_group(self) -> None:
        """Add a new nested group widget."""
        group_model = QueryGroupModel(logic="AND")
        self._add_group_widget(group_model)
        self.on_change()

    def _add_group_widget(self, model: QueryGroupModel) -> QueryGroupWidget:
        """Add a group widget with the given model.

        Args:
            model: QueryGroupModel to display

        Returns:
            The created QueryGroupWidget
        """
        widget = QueryGroupWidget(
            parent=self.children_frame,
            model=model,
            subjects=self.subjects,
            bookshelves=self.bookshelves,
            on_change=self.on_change,
            nesting_level=self.nesting_level + 1,
        )
        widget.pack(fill=tk.X, pady=5)
        self.child_widgets.append(widget)
        return widget

    def _delete_child(self, widget: QueryConstraintWidget | QueryGroupWidget) -> None:
        """Delete a child widget.

        Args:
            widget: Child widget to delete
        """
        if widget in self.child_widgets:
            self.child_widgets.remove(widget)
            widget.destroy()
            self.on_change()

    def _ungroup(self) -> None:
        """Ungroup this group (move children to parent, remove this group).

        This is handled by the parent, so we just signal via callback.
        The parent will need to implement the ungroup logic.
        """
        # Placeholder - actual implementation will be in parent handling
        # For now, just delete all children
        for widget in list(self.child_widgets):
            widget.destroy()
        self.child_widgets.clear()
        self.on_change()

    def _on_logic_change(self) -> None:
        """Handle logic selector change."""
        self.model.logic = self.logic_var.get()
        self.on_change()

    def get_model(self) -> QueryGroupModel:
        """Get the current QueryGroupModel from widget state.

        Returns:
            QueryGroupModel with current logic and children
        """
        constraints: list[QueryConstraintModel | QueryGroupModel] = []
        for widget in self.child_widgets:
            if isinstance(widget, QueryConstraintWidget):
                constraints.append(widget.get_model())
            else:
                # Must be QueryGroupWidget (union type)
                constraints.append(widget.get_model())

        return QueryGroupModel(logic=self.logic_var.get(), constraints=constraints)

    def clear_children(self) -> None:
        """Remove all child widgets."""
        for widget in list(self.child_widgets):
            widget.destroy()
        self.child_widgets.clear()

    def add_constraint_with_field(self, field_name: str) -> None:
        """Add a new constraint with pre-selected field.

        Args:
            field_name: Field name to pre-select
        """
        # Determine default operator based on field type
        field_type = FIELD_TYPES.get(field_name, "text")
        if field_type == "numeric":
            operator = ">"
        elif field_type == "boolean":
            operator = "=="
        else:
            operator = "contains"

        model = QueryConstraintModel(field=field_name, operator=operator, value="")
        self._add_constraint_widget(model)
        self.on_change()


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
        _tk_panedwindow_add(main_paned, left_frame, weight=0)
        self._create_field_palette(left_frame)

        # Center panel: Query builder
        center_frame = ttk.Frame(main_paned)
        _tk_panedwindow_add(main_paned, center_frame, weight=2)
        self._create_query_builder(center_frame)

        # Right panel: Results
        right_frame = ttk.Frame(main_paned, width=400)
        _tk_panedwindow_add(main_paned, right_frame, weight=1)
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
            command=lambda *args: _tk_canvas_yview(canvas, *args),
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
            command=lambda *args: _tk_treeview_yview(self.results_tree, *args),
        )
        scrollbar_x = ttk.Scrollbar(
            tree_frame,
            orient=tk.HORIZONTAL,
            command=lambda *args: _tk_treeview_xview(self.results_tree, *args),
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
                    "Please run scripts/build_gutenberg_id_list.py first.",
                )
                self.status_bar.config(text="Error: Data file not found")
                return

            self.df = _pandas_read_parquet(PARQUET_PATH)
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

        # Import function from explore_gutenberg.py
        sys.path.insert(0, str(Path(__file__).parent))
        from explore_gutenberg import get_canonical_sets

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
                    _pandas_to_parquet(self.last_results, path)
                else:
                    _pandas_to_csv(self.last_results, path)
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
        self.results_label.config(
            text=(
                f"Results: {count} books matching query\n"
                f"Query: {query_str[:100]}..."
                if len(query_str) > 100
                else query_str
            )
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
        for _, row in display_df.iterrows():
            values: list[str] = []
            for col in available_columns:
                # Access Series element directly without wrapper
                val = row[col]  # type: ignore[index]
                # Convert to string, using empty string for missing values
                try:
                    values.append(
                        ""
                        if (
                            val is None
                            or (isinstance(val, float) and _pandas_is_na(val))
                        )
                        else str(val)  # type: ignore[arg-type]
                    )
                except Exception:
                    values.append("")
            self.results_tree.insert("", tk.END, values=values)

        # Show warning if results truncated
        if count > 100:
            self.results_label.config(
                text=(
                    f"Results: {count} books found (showing first 100)\n"
                    f"Query: {query_str[:80]}..."
                    if len(query_str) > 80
                    else query_str
                )
            )

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


def main() -> None:
    """Entry point for the query builder UI."""
    root = tk.Tk()
    app = QueryBuilderApp(root)
    app.run()


if __name__ == "__main__":
    main()
