"""UI widgets for Gutenberg Query Builder.

Contains ToolTip, QueryConstraintWidget, and QueryGroupWidget classes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Any

from ..gutenberg_query_core import QueryConstraintModel, QueryGroupModel  # noqa: TID252
from .constants import (
    BOOLEAN_OPERATORS,
    FIELD_TYPES,
    NUMERIC_OPERATORS,
    TEXT_OPERATORS,
)
from .tk_helpers import tk_listbox_curselection, tk_listbox_get, tk_listbox_yview

if TYPE_CHECKING:
    from collections.abc import Callable


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
        scrollbar.config(command=lambda *args: tk_listbox_yview(listbox, *args))

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
            selected_indices = tk_listbox_curselection(listbox)
            selected_items = [tk_listbox_get(listbox, i) for i in selected_indices]
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
        parent_group: QueryGroupWidget | None = None,
    ) -> None:
        """Initialize query group widget.

        Args:
            parent: Parent widget
            model: QueryGroupModel to display
            subjects: Available subject values for multi-select
            bookshelves: Available bookshelf values for multi-select
            on_change: Callback when group or children change
            nesting_level: Nesting depth (0 = root, increases with nesting)
            parent_group: Owning QueryGroupWidget (None for root)
        """
        super().__init__(parent)
        self.model = model
        self.subjects = subjects
        self.bookshelves = bookshelves
        self.on_change = on_change
        self.nesting_level = nesting_level
        self.parent_group = parent_group
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
            parent_group=self,
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

    def _replace_child_with_models(
        self,
        child: QueryConstraintWidget | QueryGroupWidget,
        replacement_models: list[QueryConstraintModel | QueryGroupModel],
    ) -> None:
        """Replace a child widget with a sequence of models in the same position."""
        if child not in self.child_widgets:
            return

        new_models: list[QueryConstraintModel | QueryGroupModel] = []
        for existing in self.child_widgets:
            if existing is child:
                new_models.extend(replacement_models)
            elif isinstance(existing, QueryConstraintWidget):
                new_models.append(existing.get_model())
            else:
                new_models.append(existing.get_model())

        self.clear_children()
        for model in new_models:
            if isinstance(model, QueryConstraintModel):
                self._add_constraint_widget(model)
            else:
                self._add_group_widget(model)
        self.on_change()

    def _ungroup(self) -> None:
        """Ungroup this group (move children to parent, remove this group).

        Moves this group's children up one level and removes the group widget.
        """
        if self.parent_group is None:
            return  # Root cannot be ungrouped

        replacement_models: list[QueryConstraintModel | QueryGroupModel] = []
        for widget in self.child_widgets:
            if isinstance(widget, QueryConstraintWidget):
                replacement_models.append(widget.get_model())
            else:
                replacement_models.append(widget.get_model())

        self.parent_group._replace_child_with_models(self, replacement_models)

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
