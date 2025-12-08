"""Additional unit tests for Gutenberg Query Builder App to improve coverage."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


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

from lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui import (
    app,
)


class TestQueryBuilderAppCoverage:
    """Additional tests for QueryBuilderApp to reach 90% coverage."""

    @pytest.fixture
    def mock_root(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def app_instance(self, mock_root: MagicMock) -> app.QueryBuilderApp:
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
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.PARQUET_PATH",
                MagicMock(exists=lambda: True),
            ),
        ):
            return app.QueryBuilderApp(mock_root)

    def test_load_data_file_not_found(self, mock_root: MagicMock) -> None:
        """Test _load_data when parquet file does not exist."""
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
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.PARQUET_PATH"
            ) as mock_path,
            patch("tkinter.messagebox.showerror") as mock_showerror,
        ):
            mock_path.exists.return_value = False

            app_instance = app.QueryBuilderApp(mock_root)

            mock_showerror.assert_called_once()
            assert "Data Not Found" in mock_showerror.call_args[0][0]
            # Verify status bar updated
            app_instance.status_bar.config.assert_called_with(
                text="Error: Data file not found"
            )

    def test_load_data_exception(self, mock_root: MagicMock) -> None:
        """Test _load_data when an exception occurs."""
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
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.PARQUET_PATH"
            ) as mock_path,
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_read_parquet",
                side_effect=Exception("Read error"),
            ),
            patch("tkinter.messagebox.showerror") as mock_showerror,
        ):
            mock_path.exists.return_value = True

            app_instance = app.QueryBuilderApp(mock_root)

            mock_showerror.assert_called_once()
            assert "Load Error" in mock_showerror.call_args[0][0]
            app_instance.status_bar.config.assert_called_with(text="Error: Read error")

    def test_load_canonical_sets_none_df(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _load_canonical_sets returns early if df is None."""
        app_instance.df = None
        # Should not raise error and not call get_canonical_sets (mocked in fixture)
        app_instance._load_canonical_sets()

    def test_add_constraint_from_field(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _add_constraint_from_field shows info message."""
        with patch("tkinter.messagebox.showinfo") as mock_showinfo:
            app_instance._add_constraint_from_field("title")
            mock_showinfo.assert_called_once()
            assert "Add Constraint" in mock_showinfo.call_args[0][0]

    def test_open_query_exception(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _open_query handles exceptions."""
        with (
            patch("tkinter.filedialog.askopenfilename", return_value="test.json"),
            patch("builtins.open", side_effect=Exception("File error")),
            patch("tkinter.messagebox.showerror") as mock_showerror,
        ):
            app_instance._open_query()
            mock_showerror.assert_called_once()
            assert "Open Error" in mock_showerror.call_args[0][0]

    def test_save_query_to_file_exception(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _save_query_to_file handles exceptions."""
        with (
            patch("builtins.open", side_effect=Exception("Write error")),
            patch("tkinter.messagebox.showerror") as mock_showerror,
        ):
            app_instance._save_query_to_file(Path("test.json"))
            mock_showerror.assert_called_once()
            assert "Save Error" in mock_showerror.call_args[0][0]

    def test_export_results_no_results(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _export_results shows warning when no results."""
        app_instance.last_results = None
        with patch("tkinter.messagebox.showwarning") as mock_showwarning:
            app_instance._export_results()
            mock_showwarning.assert_called_once()
            assert "No Results" in mock_showwarning.call_args[0][0]

    def test_export_results_csv(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _export_results to CSV."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__len__.return_value = 10
        app_instance.last_results = mock_df

        with (
            patch("tkinter.filedialog.asksaveasfilename", return_value="results.csv"),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_to_csv"
            ) as mock_to_csv,
        ):
            app_instance._export_results()
            mock_to_csv.assert_called_once_with(mock_df, Path("results.csv"))
            app_instance.status_bar.config.assert_called_with(
                text="Exported 10 results to results.csv"
            )

    def test_export_results_parquet(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _export_results to Parquet."""
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.__len__.return_value = 10
        app_instance.last_results = mock_df

        with (
            patch(
                "tkinter.filedialog.asksaveasfilename", return_value="results.parquet"
            ),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_to_parquet"
            ) as mock_to_parquet,
        ):
            app_instance._export_results()
            mock_to_parquet.assert_called_once_with(mock_df, Path("results.parquet"))

    def test_export_results_exception(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _export_results handles exceptions."""
        mock_df = MagicMock()
        mock_df.empty = False
        app_instance.last_results = mock_df

        with (
            patch("tkinter.filedialog.asksaveasfilename", return_value="results.csv"),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.pandas_to_csv",
                side_effect=Exception("Export error"),
            ),
            patch("tkinter.messagebox.showerror") as mock_showerror,
        ):
            app_instance._export_results()
            mock_showerror.assert_called_once()
            assert "Export Error" in mock_showerror.call_args[0][0]

    def test_copy_query_string(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _copy_query_string."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = "query"

        app_instance._copy_query_string()

        app_instance.root.clipboard_clear.assert_called_once()
        app_instance.root.clipboard_append.assert_called_with("query")
        app_instance.status_bar.config.assert_called_with(
            text="Query string copied to clipboard"
        )

    def test_copy_query_to_clipboard_success(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _copy_query_to_clipboard with valid query."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = "query"

        app_instance._copy_query_to_clipboard()

        app_instance.root.clipboard_clear.assert_called_once()
        app_instance.root.clipboard_append.assert_called_with("query")
        app_instance.status_bar.config.assert_called_with(
            text="Query copied to clipboard"
        )

    def test_copy_query_to_clipboard_empty(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _copy_query_to_clipboard with empty query."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = ""

        app_instance._copy_query_to_clipboard()

        app_instance.root.clipboard_clear.assert_not_called()
        app_instance.status_bar.config.assert_called_with(text="No query to copy")

    def test_run_query_empty(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _run_query with empty query string."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = ""

        with patch("tkinter.messagebox.showinfo") as mock_showinfo:
            app_instance._run_query()
            mock_showinfo.assert_called_once()
            assert "Empty Query" in mock_showinfo.call_args[0][0]

    def test_run_query_exception(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _run_query handles exceptions."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = "query"

        with (
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.BooleanQueryEngine",
                side_effect=Exception("Engine error"),
            ),
            patch("tkinter.messagebox.showerror") as mock_showerror,
        ):
            app_instance._run_query()
            mock_showerror.assert_called_once()
            assert "Query Execution Error" in mock_showerror.call_args[0][0]
            app_instance.status_bar.config.assert_called_with(
                text="Query execution failed"
            )

    def test_display_results_empty(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _display_results with empty results."""
        mock_results = MagicMock()
        mock_results.__len__.return_value = 0

        app_instance._display_results(mock_results, "query")

        app_instance.results_tree.delete.assert_called()
        app_instance.results_tree.insert.assert_not_called()

    def test_display_results_truncated(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _display_results with more than 100 results."""
        mock_results = MagicMock()
        mock_results.__len__.return_value = 150
        mock_results.head.return_value = mock_results  # Mock head return
        mock_results.columns = ["id", "title"]

        # Mock iterrows
        mock_row = MagicMock()
        mock_row.__getitem__.return_value = "val"
        mock_results.iterrows.return_value = [(0, mock_row)]

        app_instance._display_results(mock_results, "query")

        # Check if label indicates truncation
        args, kwargs = app_instance.results_label.config.call_args
        assert "showing first 100" in kwargs["text"]

    def test_format_query_string(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _format_query_string."""
        # Short query
        assert app_instance._format_query_string("short query") == "short query"

        # Long query
        long_query = "A" * 30 + " AND " + "B" * 30
        formatted = app_instance._format_query_string(long_query)
        assert "\nAND " in formatted

    def test_show_about(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _show_about."""
        with patch("tkinter.messagebox.showinfo") as mock_showinfo:
            app_instance._show_about()
            mock_showinfo.assert_called_once()
            assert "About" in mock_showinfo.call_args[0][0]

    def test_run(self, app_instance: app.QueryBuilderApp) -> None:
        """Test run method."""
        app_instance.run()
        app_instance.root.mainloop.assert_called_once()

    def test_initialize_root_group_destroy(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _initialize_root_group destroys existing widget."""
        mock_widget = MagicMock()
        app_instance.root_group_widget = mock_widget

        with patch(
            "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.QueryGroupWidget"
        ):
            app_instance._initialize_root_group()

        mock_widget.destroy.assert_called_once()

    def test_new_query_yes(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _new_query when user confirms."""
        with (
            patch("tkinter.messagebox.askyesno", return_value=True),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.QueryGroupWidget"
            ),
        ):
            app_instance.current_file = Path("old.json")
            app_instance._new_query()

            assert app_instance.current_file is None
            app_instance.status_bar.config.assert_called_with(text="New query created")

    def test_new_query_no(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _new_query when user cancels."""
        with patch("tkinter.messagebox.askyesno", return_value=False):
            app_instance.current_file = Path("old.json")
            app_instance._new_query()

            assert app_instance.current_file == Path("old.json")

    def test_save_query_existing_file(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _save_query with existing file."""
        app_instance.current_file = Path("test.json")
        app_instance.current_query = MagicMock()

        with (
            patch("builtins.open", MagicMock()),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.SavedQuery"
            ),
        ):
            app_instance._save_query()

            app_instance.status_bar.config.assert_called_with(
                text="Saved query to test.json"
            )

    def test_save_query_new_file(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _save_query without existing file (calls save as)."""
        app_instance.current_file = None
        app_instance.current_query = MagicMock()

        with (
            patch("tkinter.filedialog.asksaveasfilename", return_value="new.json"),
            patch("builtins.open", MagicMock()),
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.SavedQuery"
            ),
        ):
            app_instance._save_query()

            assert app_instance.current_file == Path("new.json")
            app_instance.status_bar.config.assert_called_with(
                text="Saved query to new.json"
            )

    def test_run_query_success(self, app_instance: app.QueryBuilderApp) -> None:
        """Test _run_query success path."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = "query"
        app_instance.df = MagicMock()

        mock_results = MagicMock()
        mock_results.__len__.return_value = 5

        with (
            patch(
                "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app.BooleanQueryEngine"
            ) as mock_engine_cls,
        ):
            mock_engine = mock_engine_cls.return_value
            mock_engine.evaluate.return_value = mock_results

            # Mock _display_results to avoid complex setup
            with patch.object(app_instance, "_display_results") as mock_display:
                app_instance._run_query()

                mock_display.assert_called_once_with(mock_results, "query")
                app_instance.status_bar.config.assert_called_with(
                    text="Query executed: 5 results found"
                )

    def test_display_results_row_exception(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _display_results handles exception during row processing."""
        mock_results = MagicMock()
        mock_results.__len__.return_value = 1
        mock_results.head.return_value = mock_results
        mock_results.columns = ["id"]

        # Mock iterrows to yield a row that raises exception on access
        mock_row = MagicMock()
        # Raising exception when accessing 'id'
        type(mock_row).__getitem__ = MagicMock(side_effect=Exception("Row error"))
        mock_results.iterrows.return_value = [(0, mock_row)]

        app_instance._display_results(mock_results, "query")

        # Should insert empty string for the failed value
        # Verify insert was called with empty string in values
        # args[0] is "", args[1] is tk.END, kwargs['values'] is list
        call_args = app_instance.results_tree.insert.call_args
        assert call_args.kwargs["values"] == [""]

    def test_update_query_display_empty(
        self, app_instance: app.QueryBuilderApp
    ) -> None:
        """Test _update_query_display with empty query."""
        app_instance.current_query = MagicMock()
        app_instance.current_query.to_query_string.return_value = ""

        app_instance._update_query_display()

        app_instance.query_text.insert.assert_called_with(1.0, "(empty query)")
