from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


def _find_child(parent: Any, cls: type[Any]) -> Any:
    for child in parent.children:
        if isinstance(child, cls):
            return child
    raise AssertionError(f"{cls} not found among children")


def _collect_descendants(parent: Any, cls: type[Any]) -> list[Any]:
    matches: list[Any] = []
    for child in getattr(parent, "children", []):
        if isinstance(child, cls):
            matches.append(child)
        matches.extend(_collect_descendants(child, cls))
    return matches


def test_tooltip_show_and_hide(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    container = ui_modules.ttk.Frame(None)

    tooltip = widgets.ToolTip(container, "hint")
    assert any(args[0] == "<Enter>" for args, _ in container.bindings)
    assert any(args[0] == "<Leave>" for args, _ in container.bindings)

    tooltip._show_tip()
    assert isinstance(tooltip.tipwindow, ui_modules.tk.Toplevel)

    tooltip._hide_tip()
    assert tooltip.tipwindow is None


def test_constraint_widget_numeric_and_boolean(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    constants = ui_modules.constants
    changes = MagicMock()
    widget = widgets.QueryConstraintWidget(
        ui_modules.ttk.Frame(None),
        subjects={"Alpha"},
        bookshelves={"Beta"},
        on_change=changes,
    )

    widget.field_var.set("id")
    widget.operator_var.set("range")
    widget._update_value_widget()
    assert widget.operator_combo["values"] == constants.NUMERIC_OPERATORS

    # Range widget updates value_var via traces
    range_frame = _find_child(widget.value_widget_frame, ui_modules.ttk.Frame)
    min_spin = range_frame.children[0]
    max_spin = range_frame.children[2]
    min_spin.textvariable.set("5")
    max_spin.textvariable.set("15")
    assert widget.value_var.get() == "5..15"

    widget.field_var.set("copyright")
    bool_frame = _find_child(widget.value_widget_frame, ui_modules.ttk.Frame)
    radio_buttons = [
        child
        for child in bool_frame.children
        if isinstance(child, ui_modules.ttk.Radiobutton)
    ]
    assert len(radio_buttons) == 2
    radio_buttons[1].variable.set("false")
    assert widget.value_var.get() == "false"
    assert changes.call_count >= 1


def test_constraint_widget_multiselect_dialog(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    widget = widgets.QueryConstraintWidget(
        ui_modules.ttk.Frame(None),
        subjects={"Alpha", "Beta", "Gamma"},
    )

    widget.field_var.set("subjects")
    widget.operator_var.set("contains")
    widget._update_value_widget()
    widget._show_multiselect_dialog(["Alpha", "Beta", "Gamma"])

    dialog = _find_child(widget, ui_modules.tk.Toplevel)
    search_entry = _collect_descendants(dialog, ui_modules.ttk.Entry)[0]
    search_entry.textvariable.set("Al")

    buttons = [
        b
        for b in _collect_descendants(dialog, ui_modules.ttk.Button)
        if b.kwargs.get("text")
    ]
    select_all_btn = next(b for b in buttons if b.kwargs.get("text") == "Select All")
    clear_all_btn = next(b for b in buttons if b.kwargs.get("text") == "Clear All")
    ok_btn = next(b for b in buttons if b.kwargs.get("text") == "OK")

    select_all_btn.invoke()
    clear_all_btn.invoke()
    select_all_btn.invoke()

    ok_btn.invoke()
    assert widget.value_var.get() != ""
    assert dialog.destroyed is True


def test_constraint_widget_get_model_and_delete(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    deleted = MagicMock()
    widget = widgets.QueryConstraintWidget(
        ui_modules.ttk.Frame(None),
        on_delete=deleted,
    )

    widget.field_var.set("title")
    widget.operator_var.set("contains")
    widget.value_var.set("python")
    model = widget.get_model()
    assert model.field == "title"
    assert model.operator == "contains"
    assert model.value == "python"

    widget._on_delete_clicked()
    deleted.assert_called_once_with(widget)


def test_group_widget_add_delete_and_models(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    changes = MagicMock()
    group = widgets.QueryGroupWidget(
        ui_modules.ttk.Frame(None),
        model=widgets.QueryGroupModel(logic="AND"),
        subjects={"Alpha"},
        bookshelves={"Beta"},
        on_change=changes,
    )

    group._add_constraint()
    group._add_group()
    assert len(group.child_widgets) == 2
    assert changes.call_count >= 2

    group.logic_var.set("OR")
    model = group.get_model()
    assert model.logic == "OR"
    assert len(model.constraints) == 2

    first_child = group.child_widgets[0]
    group._delete_child(first_child)
    assert first_child not in group.child_widgets

    group.add_constraint_with_field("copyright")
    new_model = group.get_model()
    assert any(getattr(c, "field", None) == "copyright" for c in new_model.constraints)


def test_group_widget_ungroup(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    parent = widgets.QueryGroupWidget(
        ui_modules.ttk.Frame(None),
        model=widgets.QueryGroupModel(logic="AND"),
        subjects=set(),
        bookshelves=set(),
        on_change=MagicMock(),
    )
    parent._add_group()
    child_group = parent.child_widgets[0]

    child_group._add_constraint()
    assert len(child_group.child_widgets) == 1
    child_group._ungroup()
    assert all(
        not isinstance(widget, type(child_group)) for widget in parent.child_widgets
    )
