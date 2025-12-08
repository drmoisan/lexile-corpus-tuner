from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class DummyDataFrame:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.columns = list(rows[0].keys()) if rows else []

    def head(self, _: int) -> DummyDataFrame:
        return self

    def iterrows(self) -> Iterator[tuple[int, dict[str, Any]]]:
        yield from enumerate(self._rows)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def empty(self) -> bool:
        return not self._rows


@pytest.fixture()
def app_factory(ui_modules: Any, monkeypatch: pytest.MonkeyPatch) -> Callable[..., Any]:
    def _make(parquet_exists: bool = True, df: Any | None = None) -> Any:
        monkeypatch.setattr(
            ui_modules.app,
            "pandas_read_parquet",
            MagicMock(return_value=df or DummyDataFrame([{"id": 1}])),
        )
        monkeypatch.setattr(
            ui_modules.app,
            "get_canonical_sets",
            MagicMock(return_value=(set(), set())),
        )
        monkeypatch.setattr(
            ui_modules.app, "PARQUET_PATH", MagicMock(exists=lambda: parquet_exists)
        )
        return ui_modules.app.QueryBuilderApp(ui_modules.tk.Tk())

    return _make


def test_load_data_file_missing(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    ui_modules.messagebox.showerror.reset_mock()
    app_instance = app_factory(parquet_exists=False)
    ui_modules.messagebox.showerror.assert_called_once()
    assert "Error: Data file not found" in app_instance.status_bar.kwargs.get(
        "text", ""
    )


def test_load_data_exception(ui_modules: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    ui_modules.messagebox.showerror.reset_mock()
    monkeypatch.setattr(
        ui_modules.app,
        "pandas_read_parquet",
        MagicMock(side_effect=ValueError("bad file")),
    )
    monkeypatch.setattr(
        ui_modules.app,
        "get_canonical_sets",
        MagicMock(return_value=(set(), set())),
    )
    monkeypatch.setattr(ui_modules.app, "PARQUET_PATH", MagicMock(exists=lambda: True))
    app_instance = ui_modules.app.QueryBuilderApp(ui_modules.tk.Tk())
    ui_modules.messagebox.showerror.assert_called()
    assert "Error:" in app_instance.status_bar.kwargs.get("text", "")


def test_add_constraint_from_field(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    ui_modules.messagebox.showinfo.reset_mock()
    app_instance = app_factory()
    app_instance._add_constraint_from_field("title")
    ui_modules.messagebox.showinfo.assert_called_once()


def test_export_results_no_results(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    ui_modules.messagebox.showwarning.reset_mock()
    app_instance = app_factory()
    app_instance.last_results = None
    app_instance._export_results()
    ui_modules.messagebox.showwarning.assert_called_once()


def test_export_results_csv_and_parquet(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    df = DummyDataFrame([{"id": 1, "title": "a"}])
    app_instance.last_results = df

    ui_modules.filedialog.asksaveasfilename.return_value = "out.csv"
    mock_to_csv = MagicMock()
    mock_to_parquet = MagicMock()
    monkeypatch.setattr(ui_modules.app, "pandas_to_csv", mock_to_csv)
    monkeypatch.setattr(ui_modules.app, "pandas_to_parquet", mock_to_parquet)

    app_instance._export_results()
    mock_to_csv.assert_called_once_with(df, Path("out.csv"))
    mock_to_parquet.assert_not_called()

    ui_modules.filedialog.asksaveasfilename.return_value = "out.parquet"
    app_instance._export_results()
    mock_to_parquet.assert_called_once_with(df, Path("out.parquet"))


def test_export_results_exception(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    app_instance.last_results = DummyDataFrame([{"id": 1}])

    ui_modules.filedialog.asksaveasfilename.return_value = "out.csv"
    monkeypatch.setattr(
        ui_modules.app,
        "pandas_to_csv",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    ui_modules.messagebox.showerror.reset_mock()
    app_instance._export_results()
    ui_modules.messagebox.showerror.assert_called_once()


def test_run_query_empty(ui_modules: Any, app_factory: Callable[..., Any]) -> None:
    app_instance = app_factory()
    ui_modules.messagebox.showinfo.reset_mock()
    app_instance.current_query = MagicMock(to_query_string=MagicMock(return_value=""))
    app_instance._run_query()
    ui_modules.messagebox.showinfo.assert_called_once()


def test_run_query_df_missing(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    app_instance.df = None
    app_instance.current_query = MagicMock(
        to_query_string=MagicMock(return_value="query")
    )
    monkeypatch.setattr(ui_modules.app, "BooleanQueryEngine", MagicMock())
    ui_modules.messagebox.showerror.reset_mock()
    app_instance._run_query()
    ui_modules.messagebox.showerror.assert_called_once()
    assert app_instance.status_bar.kwargs.get("text") == "Query execution failed"


def test_run_query_success_displays_results(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {"id": 1, "title": "A", "authors": "Author", "download_count": 5},
        {"id": 2, "title": None, "authors": "B", "download_count": 10},
    ] * 60  # >100 rows triggers truncation message
    df = DummyDataFrame(rows)
    app_instance = app_factory(df=df)
    app_instance.current_query = MagicMock(
        to_query_string=MagicMock(return_value="field:value")
    )

    engine_mock = MagicMock()
    engine_mock.evaluate.return_value = df
    monkeypatch.setattr(
        ui_modules.app, "BooleanQueryEngine", MagicMock(return_value=engine_mock)
    )
    ui_modules.messagebox.showerror.reset_mock()

    app_instance._run_query()
    assert ui_modules.messagebox.showerror.call_count == 0
    assert app_instance.last_results is df
    assert app_instance.status_bar.kwargs.get("text", "").startswith("Query executed")
    # Results tree should have rows inserted and columns configured
    assert app_instance.results_tree.rows
    assert "download_count" in app_instance.results_tree.columns_config
    assert app_instance.results_label.kwargs.get("text", "").startswith("Results: ")


def test_copy_query_string_and_button(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    app_instance.current_query = MagicMock(
        to_query_string=MagicMock(return_value="abc")
    )

    app_instance._copy_query_string()
    assert app_instance.root.clipboard == ["abc"]
    assert "copied" in app_instance.status_bar.kwargs.get("text", "")

    app_instance.current_query.to_query_string.return_value = ""
    app_instance._copy_query_to_clipboard()
    assert app_instance.status_bar.kwargs.get("text") == "No query to copy"


def test_save_and_open_query(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    ui_modules.filedialog.asksaveasfilename.return_value = "save.json"
    saved_query_mock = MagicMock()
    saved_query_cls = MagicMock()
    saved_query_cls.from_query_group.return_value = saved_query_mock
    saved_query_mock.to_json.return_value = '{"a": 1}'
    monkeypatch.setattr(ui_modules.app, "SavedQuery", saved_query_cls)
    monkeypatch.setattr("builtins.open", MagicMock())

    app_instance._save_query()
    assert app_instance.current_file == Path("save.json")

    # Open flow
    ui_modules.filedialog.askopenfilename.return_value = "save.json"
    saved_query_cls.from_json.return_value = saved_query_mock
    saved_query_mock.to_query_group.return_value = MagicMock()
    app_instance._open_query()
    assert app_instance.current_file == Path("save.json")


def test_new_query_confirmation(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    app_instance.current_file = Path("existing.json")
    ui_modules.messagebox.askyesno.return_value = False
    app_instance._new_query()
    assert app_instance.current_file == Path("existing.json")

    ui_modules.messagebox.askyesno.return_value = True
    app_instance._new_query()
    assert app_instance.current_file is None


def test_show_about(ui_modules: Any, app_factory: Callable[..., Any]) -> None:
    ui_modules.messagebox.showinfo.reset_mock()
    app_instance = app_factory()
    app_instance._show_about()
    ui_modules.messagebox.showinfo.assert_called_once()


def test_load_canonical_sets_no_df(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    app_instance.df = None
    app_instance.subjects = {"keep"}
    app_instance.bookshelves = {"stay"}

    app_instance._load_canonical_sets()

    assert app_instance.subjects == {"keep"}
    assert app_instance.bookshelves == {"stay"}


def test_initialize_root_group_replaces_existing(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    previous_widget = MagicMock()
    app_instance.root_group_widget = previous_widget

    app_instance._initialize_root_group()

    previous_widget.destroy.assert_called_once()
    assert app_instance.root_group_widget is not None
    assert app_instance.current_query.logic in {"AND", "OR"}


def test_update_query_from_root_no_widget(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    app_instance.root_group_widget = None
    current = app_instance.current_query

    app_instance._update_query_from_root()

    assert app_instance.current_query is current


def test_display_results_empty(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    empty_df = DummyDataFrame([])

    app_instance._display_results(empty_df, "query")

    assert app_instance.results_label.kwargs.get("text", "").startswith("Results: 0")
    assert app_instance.results_tree.rows == []


def test_update_query_display_formats_long_query(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    long_query = "title:alpha AND authors:beta OR subjects:gamma " * 3
    app_instance.current_query = MagicMock(
        to_query_string=MagicMock(return_value=long_query)
    )

    app_instance._update_query_display()

    assert "\nAND" in app_instance.query_text.contents


def test_on_closing_accepts(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    monkeypatch.setattr(
        ui_modules.app.messagebox,
        "askokcancel",
        MagicMock(return_value=True),
        raising=False,
    )

    app_instance._on_closing()

    assert app_instance.root.destroyed is True


def test_save_query_to_file_error(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    failing_saved_query = MagicMock(side_effect=RuntimeError("fail"))
    monkeypatch.setattr(
        ui_modules.app, "SavedQuery", MagicMock(from_query_group=failing_saved_query)
    )
    ui_modules.messagebox.showerror.reset_mock()

    app_instance._save_query_to_file(Path("out.json"))

    ui_modules.messagebox.showerror.assert_called_once()


def test_save_query_uses_existing_file(
    ui_modules: Any, app_factory: Callable[..., Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    app_instance = app_factory()
    existing_path = Path("existing.json")
    app_instance.current_file = existing_path
    save_mock = MagicMock()
    monkeypatch.setattr(app_instance, "_save_query_to_file", save_mock)

    app_instance._save_query()

    save_mock.assert_called_once_with(existing_path)


def test_export_results_empty_df(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    app_instance.last_results = DummyDataFrame([])
    ui_modules.messagebox.showwarning.reset_mock()

    app_instance._export_results()

    ui_modules.messagebox.showwarning.assert_called_once()


def test_copy_query_to_clipboard_nonempty(
    ui_modules: Any, app_factory: Callable[..., Any]
) -> None:
    app_instance = app_factory()
    app_instance.current_query = MagicMock(
        to_query_string=MagicMock(return_value="copy-me")
    )

    app_instance._copy_query_to_clipboard()

    assert app_instance.root.clipboard == ["copy-me"]
    assert app_instance.status_bar.kwargs.get("text") == "Query copied to clipboard"
