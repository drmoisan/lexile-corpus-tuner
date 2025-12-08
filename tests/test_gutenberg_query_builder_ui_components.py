"""Unit tests for gutenberg_query_builder_ui components.

Tests constants, tk_helpers, widgets, and app modules.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch


# Define a dummy Frame class to avoid MagicMock inheritance issues
class MockFrame:
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        self.master = master
        self.tk = MagicMock()
        self._w = "mock_w"

    def pack(self, **kwargs: Any) -> None:
        pass

    def grid(self, **kwargs: Any) -> None:
        pass

    def destroy(self) -> None:
        pass

    def configure(self, **kwargs: Any) -> None:
        pass

    def bind(self, *args: Any) -> None:
        pass

    def grid_rowconfigure(self, *args: Any, **kwargs: Any) -> None:
        pass

    def grid_columnconfigure(self, *args: Any, **kwargs: Any) -> None:
        pass

    def winfo_children(self) -> list[Any]:
        return []

    def register(self, func: Callable[..., Any]) -> str:
        return "func_name"

    def add(self, child: Any, **kwargs: Any) -> None:
        pass  # For PanedWindow


# Mock tkinter before importing UI modules
mock_tk = MagicMock()
mock_ttk = MagicMock()
mock_ttk.Frame = MockFrame  # Use our dummy class
sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.ttk"] = mock_ttk

# Ensure submodules are consistent
mock_filedialog = MagicMock()
sys.modules["tkinter.filedialog"] = mock_filedialog
mock_tk.filedialog = mock_filedialog

mock_messagebox = MagicMock()
sys.modules["tkinter.messagebox"] = mock_messagebox
mock_tk.messagebox = mock_messagebox

# Also ensure tkinter.ttk resolves to our mock_ttk
mock_tk.ttk = mock_ttk

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui import (  # noqa: E402, E501
    app,
    constants,
    tk_helpers,
    widgets,
)


class TestConstants:
    """Test constants module."""

    def test_window_config(self) -> None:
        """Test window configuration constants."""
        assert constants.WINDOW_TITLE == "Gutenberg Query Builder"
        assert constants.WINDOW_SIZE == "1400x900"

    def test_parquet_path(self) -> None:
        """Test parquet path resolution."""
        assert isinstance(constants.PARQUET_PATH, Path)
        assert constants.PARQUET_PATH.name == "gutenberg_books.parquet"

    def test_field_types(self) -> None:
        """Test field types dictionary."""
        assert "id" in constants.FIELD_TYPES
        assert constants.FIELD_TYPES["id"] == "numeric"
        assert constants.FIELD_TYPES["title"] == "text"
        assert constants.FIELD_TYPES["copyright"] == "boolean"

    def test_operators(self) -> None:
        """Test operator lists."""
        assert "contains" in constants.TEXT_OPERATORS
        assert ">" in constants.NUMERIC_OPERATORS
        assert "=" in constants.BOOLEAN_OPERATORS


class TestTkHelpers:
    """Test tk_helpers module."""

    def test_tk_listbox_curselection(self) -> None:
        """Test tk_listbox_curselection helper."""
        mock_listbox = MagicMock()
        mock_listbox.curselection.return_value = (1, 2, 3)

        result = tk_helpers.tk_listbox_curselection(mock_listbox)

        assert result == (1, 2, 3)
        mock_listbox.curselection.assert_called_once()

    def test_tk_listbox_get(self) -> None:
        """Test tk_listbox_get helper."""
        mock_listbox = MagicMock()
        mock_listbox.get.return_value = "item"

        result = tk_helpers.tk_listbox_get(mock_listbox, 0)

        assert result == "item"
        mock_listbox.get.assert_called_once_with(0)

    def test_tk_listbox_yview(self) -> None:
        """Test tk_listbox_yview helper."""
        mock_listbox = MagicMock()

        tk_helpers.tk_listbox_yview(mock_listbox, "scroll", 1, "units")

        mock_listbox.yview.assert_called_once_with("scroll", 1, "units")

    def test_tk_canvas_yview(self) -> None:
        """Test tk_canvas_yview helper."""
        mock_canvas = MagicMock()

        tk_helpers.tk_canvas_yview(mock_canvas, "moveto", 0.5)

        mock_canvas.yview.assert_called_once_with("moveto", 0.5)

    def test_tk_treeview_xview(self) -> None:
        """Test tk_treeview_xview helper."""
        mock_tree = MagicMock()

        tk_helpers.tk_treeview_xview(mock_tree, "scroll", 1, "pages")

        mock_tree.xview.assert_called_once_with("scroll", 1, "pages")

    def test_tk_treeview_yview(self) -> None:
        """Test tk_treeview_yview helper."""
        mock_tree = MagicMock()

        tk_helpers.tk_treeview_yview(mock_tree, "moveto", 0.0)

        mock_tree.yview.assert_called_once_with("moveto", 0.0)

    def test_tk_panedwindow_add(self) -> None:
        """Test tk_panedwindow_add helper."""
        mock_paned = MagicMock()
        mock_child = MagicMock()

        tk_helpers.tk_panedwindow_add(mock_paned, mock_child, weight=1)

        mock_paned.add.assert_called_once_with(mock_child, weight=1)


class TestWidgets:
    """Test widgets module."""

    @patch("tkinter.Toplevel")
    @patch("tkinter.Label")
    def test_tooltip(self, mock_label: MagicMock, mock_toplevel: MagicMock) -> None:
        """Test ToolTip widget."""
        mock_widget = MagicMock()
        tooltip = widgets.ToolTip(mock_widget, "Test tooltip")

        # Verify bindings
        # fmt: off
        mock_widget.bind.assert_any_call(
            "<Enter>", tooltip._show_tip  # pyright: ignore[reportPrivateUsage]
        )
        mock_widget.bind.assert_any_call(
            "<Leave>", tooltip._hide_tip  # pyright: ignore[reportPrivateUsage]
        )
        # fmt: on

        # Test show tip
        mock_widget.winfo_rootx.return_value = 100
        mock_widget.winfo_rooty.return_value = 100
        mock_widget.winfo_height.return_value = 20

        tooltip._show_tip()  # pyright: ignore[reportPrivateUsage]
        mock_toplevel.assert_called_once()
        mock_label.assert_called_once()

        # Test hide tip
        tooltip._hide_tip()  # pyright: ignore[reportPrivateUsage]
        mock_toplevel.return_value.destroy.assert_called_once()

    def test_query_constraint_widget_init(self) -> None:
        """Test QueryConstraintWidget initialization."""
        mock_parent = MagicMock()

        # Mock StringVar to behave like a real one for testing logic
        with patch("tkinter.StringVar") as mock_string_var:
            # Return a new mock for each instantiation to avoid StopIteration
            mock_string_var.side_effect = MagicMock

            widget = widgets.QueryConstraintWidget(mock_parent)

            # Verify UI components created
            # Access the mock from the module since we mocked it at module level
            # widgets.py imports ttk from tkinter, so it uses mock_tk.ttk
            mock_combobox = mock_tk.ttk.Combobox
            mock_button = mock_tk.ttk.Button

            assert mock_combobox.call_count >= 2  # Field and Operator combos
            assert mock_button.call_count >= 1  # Delete button

            # Verify default values
            assert widget.field_var is not None
            assert widget.operator_var is not None

    def test_query_constraint_widget_update_operators(self) -> None:
        """Test operator updates based on field type."""
        mock_parent = MagicMock()
        with patch("tkinter.StringVar") as mock_string_var:
            # Setup mock vars
            field_var = MagicMock()
            operator_var = MagicMock()
            value_var = MagicMock()

            # We need to return these specific mocks for the first 3 calls
            # and then return generic mocks for any subsequent calls
            counter = [0]

            def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
                if counter[0] == 0:
                    counter[0] += 1
                    return field_var
                elif counter[0] == 1:
                    counter[0] += 1
                    return operator_var
                elif counter[0] == 2:
                    counter[0] += 1
                    return value_var
                return MagicMock()

            mock_string_var.side_effect = side_effect

            widget = widgets.QueryConstraintWidget(mock_parent)

            # Test numeric field
            field_var.get.return_value = "id"  # numeric
            widget._update_operators()  # pyright: ignore[reportPrivateUsage]
            # Verify __setitem__ was called with correct values
            widget.operator_combo.__setitem__.assert_called_with(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                "values", constants.NUMERIC_OPERATORS
            )

            # Test boolean field
            field_var.get.return_value = "copyright"  # boolean
            widget._update_operators()  # pyright: ignore[reportPrivateUsage]
            widget.operator_combo.__setitem__.assert_called_with(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                "values", constants.BOOLEAN_OPERATORS
            )

    def test_query_constraint_widget_get_model(self) -> None:
        """Test model generation from widget state."""
        mock_parent = MagicMock()
        with patch("tkinter.StringVar") as mock_string_var:
            field_var = MagicMock()
            operator_var = MagicMock()
            value_var = MagicMock()

            counter = [0]

            def side_effect(*args: Any, **kwargs: Any) -> MagicMock:
                if counter[0] == 0:
                    counter[0] += 1
                    return field_var
                elif counter[0] == 1:
                    counter[0] += 1
                    return operator_var
                elif counter[0] == 2:
                    counter[0] += 1
                    return value_var
                return MagicMock()

            mock_string_var.side_effect = side_effect

            widget = widgets.QueryConstraintWidget(mock_parent)

            field_var.get.return_value = "title"
            operator_var.get.return_value = "contains"
            value_var.get.return_value = "test"

            model = widget.get_model()
            assert model.field == "title"
            assert model.operator == "contains"
            assert model.value == "test"

    def test_query_group_widget_init(self) -> None:
        """Test QueryGroupWidget initialization."""
        mock_parent = MagicMock()
        mock_model = MagicMock()
        mock_model.logic = "AND"
        mock_model.constraints = []

        with patch("tkinter.StringVar") as mock_string_var:
            mock_string_var.side_effect = MagicMock

            widget = widgets.QueryGroupWidget(
                mock_parent, mock_model, set(), set(), MagicMock()
            )

            assert widget.model == mock_model
            assert len(widget.child_widgets) == 0

    def test_query_group_widget_add_constraint(self) -> None:
        """Test adding constraint to group."""
        mock_parent = MagicMock()
        mock_model = MagicMock()
        mock_model.constraints = []

        with (
            patch("tkinter.StringVar") as mock_string_var,
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.widgets.QueryConstraintWidget"
            ) as mock_constraint_widget,
        ):

            mock_string_var.side_effect = MagicMock

            widget = widgets.QueryGroupWidget(
                mock_parent, mock_model, set(), set(), MagicMock()
            )

            widget._add_constraint()  # pyright: ignore[reportPrivateUsage]

            assert len(widget.child_widgets) == 1
            mock_constraint_widget.assert_called()

    def test_query_group_widget_get_model(self) -> None:
        """Test model generation from group widget."""
        mock_parent = MagicMock()
        mock_model = MagicMock()
        mock_model.logic = "AND"
        mock_model.constraints = []

        with patch("tkinter.StringVar") as mock_string_var:
            mock_string_var.side_effect = MagicMock

            widget = widgets.QueryGroupWidget(
                mock_parent, mock_model, set(), set(), MagicMock()
            )

            # Mock child widgets
            mock_child1 = MagicMock()
            cast(MagicMock, mock_child1.get_model).return_value = "constraint1"
            mock_child2 = MagicMock()
            cast(MagicMock, mock_child2.get_model).return_value = "constraint2"

            widget.child_widgets = [mock_child1, mock_child2]
            cast(MagicMock, widget.logic_var).get.return_value = "OR"

            result_model = widget.get_model()

            assert result_model.logic == "OR"
            assert result_model.constraints == ["constraint1", "constraint2"]


class TestApp:
    """Test app module."""

    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.get_canonical_sets"
    )
    def test_app_init(
        self, mock_get_sets: MagicMock, mock_read_parquet: MagicMock
    ) -> None:
        """Test QueryBuilderApp initialization."""
        mock_root = MagicMock()
        mock_df = MagicMock()
        mock_read_parquet.return_value = mock_df
        mock_get_sets.return_value = ({"subject1"}, {"shelf1"})

        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.PanedWindow", MockFrame),
            patch("tkinter.ttk.Frame", MockFrame),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.LabelFrame", MockFrame),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Treeview"),
        ):

            app_instance = app.QueryBuilderApp(mock_root)

            assert app_instance.root == mock_root
            assert app_instance.subjects == {"subject1"}
            assert app_instance.bookshelves == {"shelf1"}
            mock_read_parquet.assert_called_once()

    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.get_canonical_sets"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.BooleanQueryEngine"
    )
    def test_run_query(
        self,
        mock_engine_cls: MagicMock,
        mock_get_sets: MagicMock,
        mock_read_parquet: MagicMock,
    ) -> None:
        """Test running a query."""
        mock_root = MagicMock()
        mock_df = MagicMock()
        mock_read_parquet.return_value = mock_df
        mock_get_sets.return_value = (set(), set())

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_engine.evaluate.return_value = mock_df

        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.PanedWindow", MockFrame),
            patch("tkinter.ttk.Frame", MockFrame),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.LabelFrame", MockFrame),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Treeview"),
        ):

            app_instance = app.QueryBuilderApp(mock_root)

            # Mock query model
            app_instance.current_query = MagicMock()
            app_instance.current_query.to_query_string.return_value = (
                "title:test"  # pyright: ignore[reportAttributeAccessIssue]
            )

            app_instance._run_query()  # pyright: ignore[reportPrivateUsage]

            mock_engine.evaluate.assert_called_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            assert app_instance.last_results == mock_df

    @patch("tkinter.StringVar")
    @patch("tkinter.messagebox.askyesno")
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.get_canonical_sets"
    )
    def test_new_query(
        self,
        mock_get_sets: MagicMock,
        mock_read_parquet: MagicMock,
        mock_askyesno: MagicMock,
        mock_string_var: MagicMock,
    ) -> None:
        """Test creating a new query."""
        mock_root = MagicMock()
        mock_read_parquet.return_value = MagicMock()
        mock_get_sets.return_value = (set(), set())

        # Configure StringVar to return "AND"
        mock_string_var.return_value.get.return_value = "AND"

        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.PanedWindow", MockFrame),
            patch("tkinter.ttk.Frame", MockFrame),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.LabelFrame", MockFrame),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Treeview"),
        ):

            app_instance = app.QueryBuilderApp(mock_root)

            # Simulate user confirming new query
            mock_askyesno.return_value = True

            # Set some state
            app_instance.current_file = Path("test.json")

            app_instance._new_query()  # pyright: ignore[reportPrivateUsage]

            assert app_instance.current_file is None
            assert app_instance.current_query.logic == "AND"
            assert len(app_instance.current_query.constraints) == 0

    @patch("tkinter.filedialog.asksaveasfilename")
    @patch("builtins.open", new_callable=MagicMock)
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.SavedQuery"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.get_canonical_sets"
    )
    def test_save_query(
        self,
        mock_get_sets: MagicMock,
        mock_read_parquet: MagicMock,
        mock_saved_query: MagicMock,
        mock_open: MagicMock,
        mock_asksaveas: MagicMock,
    ) -> None:
        """Test saving a query."""
        mock_root = MagicMock()
        mock_read_parquet.return_value = MagicMock()
        mock_get_sets.return_value = (set(), set())

        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.PanedWindow", MockFrame),
            patch("tkinter.ttk.Frame", MockFrame),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.LabelFrame", MockFrame),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Treeview"),
        ):

            app_instance = app.QueryBuilderApp(mock_root)

            # Mock SavedQuery.from_query_group
            mock_saved_query_instance = MagicMock()
            mock_saved_query.from_query_group.return_value = (
                mock_saved_query_instance  # pyright: ignore[reportAttributeAccessIssue]
            )
            mock_saved_query_instance.to_json.return_value = (
                '{"test": "json"}'  # pyright: ignore[reportAttributeAccessIssue]
            )

            # Test Save As (no current file)
            mock_asksaveas.return_value = "test_query.json"
            app_instance._save_query()  # pyright: ignore[reportPrivateUsage]

            mock_asksaveas.assert_called_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            mock_open.assert_called_with(
                Path("test_query.json"), "w", encoding="utf-8"
            )  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            mock_open.return_value.__enter__.return_value.write.assert_called_with(  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                '{"test": "json"}'
            )
            assert app_instance.current_file == Path("test_query.json")

            # Test Save (existing file)
            mock_asksaveas.reset_mock()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            mock_open.reset_mock()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

            app_instance._save_query()  # pyright: ignore[reportPrivateUsage]

            mock_asksaveas.assert_not_called()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            mock_open.assert_called_with(
                Path("test_query.json"), "w", encoding="utf-8"
            )  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.QueryBuilderApp._initialize_root_group"
    )
    @patch("tkinter.filedialog.askopenfilename")
    @patch("builtins.open", new_callable=MagicMock)
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.SavedQuery"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet"
    )
    @patch(
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.get_canonical_sets"
    )
    def test_open_query(
        self,
        mock_get_sets: MagicMock,
        mock_read_parquet: MagicMock,
        mock_saved_query: MagicMock,
        mock_open: MagicMock,
        mock_askopen: MagicMock,
        mock_init_root: MagicMock,
    ) -> None:
        """Test opening a query."""
        mock_root = MagicMock()
        mock_read_parquet.return_value = MagicMock()
        mock_get_sets.return_value = (set(), set())

        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.PanedWindow", MockFrame),
            patch("tkinter.ttk.Frame", MockFrame),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.LabelFrame", MockFrame),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Treeview"),
        ):

            app_instance = app.QueryBuilderApp(mock_root)

            # Reset mock call from init
            mock_init_root.reset_mock()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

            # Mock file selection
            mock_askopen.return_value = "test_query.json"

            # Mock file read
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            mock_file.read.return_value = (
                '{"test": "json"}'  # pyright: ignore[reportAttributeAccessIssue]
            )

            # Mock SavedQuery.from_json
            mock_saved_query_instance = MagicMock()
            mock_saved_query.from_json.return_value = (
                mock_saved_query_instance  # pyright: ignore[reportAttributeAccessIssue]
            )
            mock_query_group = MagicMock()
            mock_saved_query_instance.to_query_group.return_value = (
                mock_query_group  # pyright: ignore[reportAttributeAccessIssue]
            )

            app_instance._open_query()  # pyright: ignore[reportPrivateUsage]

            mock_open.assert_called_with(
                "test_query.json", encoding="utf-8"
            )  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            mock_saved_query.from_json.assert_called_with(
                '{"test": "json"}'
            )  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
            assert app_instance.current_query == mock_query_group
            assert app_instance.current_file == Path("test_query.json")
            mock_init_root.assert_called_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

    @patch("tkinter.messagebox.askokcancel")
    def test_on_closing(self, mock_askokcancel: MagicMock) -> None:
        """Test window closing handler."""
        mock_root = MagicMock()

        with (
            patch("tkinter.Menu"),
            patch("tkinter.ttk.PanedWindow", MockFrame),
            patch("tkinter.ttk.Frame", MockFrame),
            patch("tkinter.ttk.Label"),
            patch("tkinter.ttk.Button"),
            patch("tkinter.Canvas"),
            patch("tkinter.ttk.Scrollbar"),
            patch("tkinter.ttk.LabelFrame", MockFrame),
            patch("tkinter.Text"),
            patch("tkinter.ttk.Treeview"),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet"
            ),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.get_canonical_sets",
                return_value=(set(), set()),
            ),
        ):

            app_instance = app.QueryBuilderApp(mock_root)

            # Test cancel
            mock_askokcancel.return_value = False
            app_instance._on_closing()  # pyright: ignore[reportPrivateUsage]
            mock_root.destroy.assert_not_called()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

            # Test confirm
            mock_askokcancel.return_value = True
            app_instance._on_closing()  # pyright: ignore[reportPrivateUsage]
            mock_root.destroy.assert_called_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


def test_main() -> None:
    """Test main entry point."""
    with (
        patch("tkinter.Tk") as mock_tk_cls,
        patch(
            "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.QueryBuilderApp"
        ) as mock_app_cls,
    ):

        from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui import (  # noqa: E501
            main,
        )

        main()

        mock_tk_cls.assert_called_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        mock_app_cls.assert_called_once_with(
            mock_tk_cls.return_value
        )  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        mock_app_cls.return_value.run.assert_called_once()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
