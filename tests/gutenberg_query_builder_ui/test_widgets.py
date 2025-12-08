from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    import pytest


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


def test_validate_numeric(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    widget = widgets.QueryConstraintWidget(ui_modules.ttk.Frame(None))

    assert widget._validate_numeric("") is True
    assert widget._validate_numeric("0") is True
    assert widget._validate_numeric("999999") is True
    assert widget._validate_numeric("-1") is False
    assert widget._validate_numeric("1000000") is False
    assert widget._validate_numeric("not-a-number") is False


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


def test_update_value_widget_bookshelves_contains(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    widget = widgets.QueryConstraintWidget(
        ui_modules.ttk.Frame(None),
        subjects={"SubjectOnly"},
        bookshelves={"ShelfB", "ShelfA"},
    )

    widget.field_var.set("bookshelves")
    widget.operator_var.set("contains")
    widget.value_var.set("ShelfB")
    widget._update_value_widget()

    button = _collect_descendants(widget.value_widget_frame, ui_modules.ttk.Button)[0]
    assert button.kwargs.get("text") == "Select... (1 selected)"

    button.invoke()
    dialog = _find_child(widget, ui_modules.tk.Toplevel)
    listbox = _collect_descendants(dialog, ui_modules.tk.Listbox)[0]
    assert listbox.items == ["ShelfA", "ShelfB"]
    assert listbox.selected == [1]


def test_create_range_widget_uses_existing_value(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    widget = widgets.QueryConstraintWidget(ui_modules.ttk.Frame(None))

    widget.field_var.set("download_count")
    widget.operator_var.set("range")
    widget.value_var.set("10..20")
    widget._update_value_widget()

    range_frame = _find_child(widget.value_widget_frame, ui_modules.ttk.Frame)
    min_spin = range_frame.children[0]
    max_spin = range_frame.children[2]
    assert min_spin.textvariable.get() == "10"
    assert max_spin.textvariable.get() == "20"

    min_spin.textvariable.set("15")
    max_spin.textvariable.set("25")
    assert widget.value_var.get() == "15..25"


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


def test_show_multiselect_dialog_preselects_existing(
    ui_modules: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    widgets = ui_modules.widgets
    selection_calls: list[tuple[int, Any]] = []
    original_selection_set = ui_modules.tk.Listbox.selection_set

    def tracking_selection_set(self: Any, start: int, end: Any = None) -> None:
        selection_calls.append((start, end))
        original_selection_set(self, start, end)

    monkeypatch.setattr(ui_modules.tk.Listbox, "selection_set", tracking_selection_set)

    widget = widgets.QueryConstraintWidget(ui_modules.ttk.Frame(None))
    widget.value_var.set("Beta;Gamma")
    widget._show_multiselect_dialog(["Alpha", "Beta", "Gamma"])

    dialog = _find_child(widget, ui_modules.tk.Toplevel)
    listbox = _collect_descendants(dialog, ui_modules.tk.Listbox)[0]
    assert selection_calls == [(1, None), (2, None)]
    assert listbox.selected == [2]


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


def test_constraint_widget_get_model_splits_semicolon_values(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    widget = widgets.QueryConstraintWidget(ui_modules.ttk.Frame(None))

    widget.field_var.set("subjects")
    widget.operator_var.set("contains")
    widget.value_var.set("alpha; beta ; ;gamma")
    model = widget.get_model()

    assert model.field == "subjects"
    assert model.operator == "contains"
    assert model.value == ["alpha", "beta", "gamma"]


def test_replace_child_with_models_noop_when_missing(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    changes = MagicMock()
    group = widgets.QueryGroupWidget(
        ui_modules.ttk.Frame(None),
        model=widgets.QueryGroupModel(logic="AND"),
        subjects=set(),
        bookshelves=set(),
        on_change=changes,
    )
    external_child = widgets.QueryConstraintWidget(ui_modules.ttk.Frame(None))
    original_children = list(group.child_widgets)

    group._replace_child_with_models(
        external_child,
        [widgets.QueryConstraintModel(field="title", operator="contains", value="x")],
    )

    assert group.child_widgets == original_children
    changes.assert_not_called()


def test_replace_child_with_models_replaces_constraint_and_keeps_peers(
    ui_modules: Any,
) -> None:
    widgets = ui_modules.widgets
    changes = MagicMock()
    group = widgets.QueryGroupWidget(
        ui_modules.ttk.Frame(None),
        model=widgets.QueryGroupModel(logic="AND"),
        subjects=set(),
        bookshelves=set(),
        on_change=changes,
    )
    first = group._add_constraint_widget(
        widgets.QueryConstraintModel(field="title", operator="contains", value="first")
    )
    group._add_constraint_widget(
        widgets.QueryConstraintModel(field="authors", operator="contains", value="keep")
    )
    changes.reset_mock()

    group._replace_child_with_models(
        first,
        [widgets.QueryConstraintModel(field="id", operator=">", value="5")],
    )

    models = [child.get_model() for child in group.child_widgets]
    assert [(m.field, m.operator, m.value) for m in models] == [
        ("id", ">", "5"),
        ("authors", "contains", "keep"),
    ]
    changes.assert_called_once()


def test_replace_child_with_models_preserves_group_siblings(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    changes = MagicMock()
    group = widgets.QueryGroupWidget(
        ui_modules.ttk.Frame(None),
        model=widgets.QueryGroupModel(logic="AND"),
        subjects=set(),
        bookshelves=set(),
        on_change=changes,
    )
    target = group._add_constraint_widget(
        widgets.QueryConstraintModel(field="title", operator="contains", value="old")
    )
    sibling_group_model = widgets.QueryGroupModel(
        logic="OR",
        constraints=[
            widgets.QueryConstraintModel(
                field="authors", operator="contains", value="nested"
            )
        ],
    )
    group._add_group_widget(sibling_group_model)
    changes.reset_mock()

    group._replace_child_with_models(
        target,
        [widgets.QueryConstraintModel(field="languages", operator="=", value="en")],
    )

    parent_model = group.get_model()
    assert len(parent_model.constraints) == 2
    assert isinstance(parent_model.constraints[0], widgets.QueryConstraintModel)
    assert parent_model.constraints[0].field == "languages"
    assert isinstance(parent_model.constraints[1], widgets.QueryGroupModel)
    assert parent_model.constraints[1].constraints[0].value == "nested"
    changes.assert_called_once()


def test_add_constraint_with_field_sets_operator_by_type(ui_modules: Any) -> None:
    widgets = ui_modules.widgets
    changes = MagicMock()
    group = widgets.QueryGroupWidget(
        ui_modules.ttk.Frame(None),
        model=widgets.QueryGroupModel(logic="AND"),
        subjects=set(),
        bookshelves=set(),
        on_change=changes,
    )
    changes.reset_mock()

    group.add_constraint_with_field("download_count")
    numeric_model = group.child_widgets[-1].get_model()
    assert numeric_model.operator == ">"

    group.add_constraint_with_field("copyright")
    boolean_widget = group.child_widgets[-1]
    boolean_model = boolean_widget.get_model()
    assert (
        boolean_widget.operator_combo["values"]
        == ui_modules.constants.BOOLEAN_OPERATORS
    )
    assert boolean_model.operator == "="
    assert changes.call_count == 2


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
