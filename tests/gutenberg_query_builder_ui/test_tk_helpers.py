from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from tests.gutenberg_query_builder_ui.conftest import StubListbox


def test_listbox_helpers(ui_modules: Any) -> None:
    tk_helpers = ui_modules.tk_helpers
    listbox = StubListbox()
    listbox.items = ["a", "b", "c"]
    listbox.selection_set(0, 1)

    assert tk_helpers.tk_listbox_curselection(listbox) == (0, 1)
    assert tk_helpers.tk_listbox_get(listbox, 2) == "c"

    tk_helpers.tk_listbox_yview(listbox, "moveto", 0.5)
    assert listbox.kwargs["yview_args"] == ("moveto", 0.5)


def test_canvas_and_treeview_helpers(ui_modules: Any) -> None:
    tk_helpers = ui_modules.tk_helpers
    canvas = ui_modules.tk.Canvas(None)
    tree = ui_modules.ttk.Treeview(None)
    paned = ui_modules.ttk.PanedWindow(None)
    child = ui_modules.ttk.Frame(None)

    tk_helpers.tk_canvas_yview(canvas, "moveto", 0.1)
    assert canvas.kwargs["yview_args"] == ("moveto", 0.1)

    tk_helpers.tk_treeview_xview(tree, "moveto", 0.2)
    tk_helpers.tk_treeview_yview(tree, "moveto", 0.3)
    assert tree.kwargs["xview_args"] == ("moveto", 0.2)
    assert tree.kwargs["yview_args"] == ("moveto", 0.3)

    tk_helpers.tk_panedwindow_add(paned, child, weight=1)
    assert paned.added == [(child, {"weight": 1})]


def test_pandas_helpers(ui_modules: Any) -> None:
    tk_helpers = ui_modules.tk_helpers
    pandas_mod = ui_modules.pandas

    sentinel_df = object()
    read_parquet_mock = cast(MagicMock, pandas_mod.read_parquet)
    read_parquet_mock.return_value = sentinel_df
    result = tk_helpers.pandas_read_parquet(Path("dummy.parquet"))
    assert result is sentinel_df
    read_parquet_mock.assert_called_once_with(Path("dummy.parquet"))

    df_mock = MagicMock()
    tk_helpers.pandas_to_csv(df_mock, Path("out.csv"))
    df_mock.to_csv.assert_called_once_with(Path("out.csv"), index=False)

    df_mock_parquet = MagicMock()
    tk_helpers.pandas_to_parquet(df_mock_parquet, Path("out.parquet"))
    df_mock_parquet.to_parquet.assert_called_once_with(Path("out.parquet"), index=False)

    isna_mock = cast(MagicMock, pandas_mod.isna)
    assert tk_helpers.pandas_is_na(None) is True
    isna_mock.assert_called()

    df_column = MagicMock()
    df_column.__getitem__.return_value = "value"
    assert tk_helpers.pandas_get_column(df_column, "col") == df_column["col"]
