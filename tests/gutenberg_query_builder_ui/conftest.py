from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


class StubWidget:
    def __init__(self, master: Any | None = None, **kwargs: Any) -> None:
        self.master = master
        self.children: list[StubWidget] = []
        if master is not None and hasattr(master, "children"):
            master.children.append(self)
        self.kwargs = kwargs
        self.destroyed = False
        self.bindings: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.grid_kwargs: dict[str, Any] | None = None
        self.pack_kwargs: dict[str, Any] | None = None

    def pack(self, **kwargs: Any) -> None:
        self.pack_kwargs = kwargs

    def grid(self, **kwargs: Any) -> None:
        self.grid_kwargs = kwargs

    def destroy(self) -> None:
        self.destroyed = True
        if self.master is not None and hasattr(self.master, "children"):
            try:
                self.master.children.remove(self)
            except ValueError:
                pass

    def configure(self, **kwargs: Any) -> None:
        self.kwargs.update(kwargs)

    def config(self, **kwargs: Any) -> None:
        self.configure(**kwargs)

    def bind(self, *args: Any, **kwargs: Any) -> None:
        self.bindings.append((args, kwargs))

    def grid_rowconfigure(self, *args: Any, **kwargs: Any) -> None:
        return None

    def grid_columnconfigure(self, *args: Any, **kwargs: Any) -> None:
        return None

    def winfo_children(self) -> list[StubWidget]:
        return list(self.children)

    def winfo_rootx(self) -> int:
        return 10

    def winfo_rooty(self) -> int:
        return 20

    def winfo_height(self) -> int:
        return 5

    def register(self, func: Callable[..., Any]) -> Callable[..., Any]:
        return func


class StubMenu(StubWidget):
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.entries: list[tuple[str, dict[str, Any]]] = []

    def add_cascade(self, **kwargs: Any) -> None:
        self.entries.append(("cascade", kwargs))

    def add_command(self, **kwargs: Any) -> None:
        self.entries.append(("command", kwargs))

    def add_separator(self, **kwargs: Any) -> None:
        self.entries.append(("separator", kwargs))


class StubToplevel(StubWidget):
    def wm_overrideredirect(self, flag: bool) -> None:
        self.kwargs["overrideredirect"] = flag

    def wm_geometry(self, value: str) -> None:
        self.kwargs["geometry"] = value

    def title(self, value: str) -> None:
        self.kwargs["title"] = value

    def geometry(self, value: str) -> None:
        self.kwargs["geometry"] = value


class StubStringVar:
    def __init__(self, value: str | None = None) -> None:
        self._value = value or ""
        self._callbacks: list[Callable[..., Any]] = []

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value
        for callback in list(self._callbacks):
            try:
                callback()
            except TypeError:
                try:
                    callback(None, None, None)
                except TypeError:
                    callback(None)

    def trace_add(self, mode: str, callback: Callable[..., Any]) -> str:  # noqa: ARG002
        self._callbacks.append(callback)
        return f"trace_{len(self._callbacks)}"


class StubFrame(StubWidget):
    pass


class StubLabel(StubWidget):
    pass


class StubButton(StubWidget):
    def __init__(
        self,
        master: Any = None,
        command: Callable[[], Any] | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.command = command

    def invoke(self) -> None:
        if self.command:
            self.command()


class StubEntry(StubWidget):
    def __init__(
        self,
        master: Any = None,
        textvariable: StubStringVar | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.textvariable = textvariable


class StubSpinbox(StubWidget):
    def __init__(
        self,
        master: Any = None,
        textvariable: StubStringVar | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.textvariable = textvariable


class StubRadiobutton(StubWidget):
    def __init__(
        self,
        master: Any = None,
        variable: StubStringVar | None = None,
        value: str | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.variable = variable
        self.value = value


class StubSeparator(StubWidget):
    pass


class StubScrollbar(StubWidget):
    def __init__(
        self,
        master: Any = None,
        command: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.command = command

    def config(self, **kwargs: Any) -> None:
        if "command" in kwargs:
            self.command = kwargs["command"]

    def set(self, *args: Any) -> None:
        self.kwargs["set_args"] = args


class StubListbox(StubWidget):
    def __init__(
        self,
        master: Any = None,
        selectmode: str | None = None,
        yscrollcommand: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.selectmode = selectmode
        self.yscrollcommand = yscrollcommand
        self.items: list[str] = []
        self.selected: list[int] = []

    def curselection(self) -> tuple[int, ...]:
        return tuple(self.selected)

    def get(self, index: int) -> str:
        return self.items[index]

    def insert(self, index: Any, item: str) -> None:
        if index == "end":
            self.items.append(item)
        else:
            self.items.insert(int(index), item)

    def delete(self, start: int, end: Any = None) -> None:
        if end is None or end == "end":
            del self.items[start:]
        else:
            del self.items[start:end]
        self.selected = []

    def selection_set(self, start: int, end: Any = None) -> None:
        if end is None or end == "end":
            end_index = len(self.items) - 1
        else:
            end_index = int(end)
        self.selected = list(range(int(start), end_index + 1))

    def selection_clear(self, start: int, end: Any = None) -> None:
        self.selected = []

    def yview(self, *args: Any) -> None:
        self.kwargs["yview_args"] = args


class StubCombobox(StubWidget):
    def __init__(
        self,
        master: Any = None,
        textvariable: StubStringVar | None = None,
        values: list[str] | None = None,
        **kwargs: Any,
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.textvariable = textvariable
        self.values = values or []
        self.state = kwargs.get("state")

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "values":
            self.values = list(value)
        else:
            self.kwargs[key] = value

    def __getitem__(self, key: str) -> Any:
        if key == "values":
            return self.values
        raise KeyError(key)


class StubPanedWindow(StubWidget):
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.added: list[tuple[Any, dict[str, Any]]] = []

    def add(self, child: Any, **kwargs: Any) -> None:
        self.children.append(child)
        self.added.append((child, kwargs))


class StubLabelFrame(StubFrame):
    pass


class StubTreeview(StubWidget):
    def __init__(
        self, master: Any = None, columns: list[str] | None = None, **kwargs: Any
    ) -> None:  # noqa: B008
        super().__init__(master, **kwargs)
        self.columns = columns or []
        self.rows: list[tuple[Any, ...]] = []
        self.headings: dict[str, dict[str, Any]] = {}
        self.columns_config: dict[str, dict[str, Any]] = {}

    def __setitem__(self, key: str, value: Any) -> None:
        self.kwargs[key] = value

    def get_children(self) -> list[int]:
        return list(range(len(self.rows)))

    def delete(self, *items: Any) -> None:
        if not items:
            self.rows.clear()
        else:
            for index in sorted(int(i) for i in items if isinstance(i, int))[::-1]:
                if 0 <= index < len(self.rows):
                    self.rows.pop(index)

    def heading(self, column: str, **kwargs: Any) -> None:
        self.headings[column] = kwargs

    def column(self, column: str, **kwargs: Any) -> None:
        self.columns_config[column] = kwargs

    def insert(
        self,
        parent: Any,
        index: Any,
        iid: Any | None = None,
        values: tuple[Any, ...] | list[Any] | None = None,
    ) -> None:  # noqa: B008
        self.rows.append(tuple(values or ()))

    def xview(self, *args: Any) -> None:
        self.kwargs["xview_args"] = args

    def yview(self, *args: Any) -> None:
        self.kwargs["yview_args"] = args


class StubCanvas(StubWidget):
    def yview(self, *args: Any) -> None:
        self.kwargs["yview_args"] = args

    def create_window(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs["create_window_args"] = (args, kwargs)


class StubText(StubWidget):
    def __init__(self, master: Any = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.contents = ""
        self.state = kwargs.get("state")

    def delete(self, start: Any, end: Any | None = None) -> None:
        self.contents = ""

    def insert(self, index: Any, text: str) -> None:
        self.contents = text

    def get(self, start: Any, end: Any) -> str:
        return self.contents

    def config(self, **kwargs: Any) -> None:
        if "state" in kwargs:
            self.state = kwargs["state"]
        self.kwargs.update(kwargs)


class StubTk(StubWidget):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(None, **kwargs)
        self._menu = None
        self.protocols: dict[str, Any] = {}
        self.clipboard: list[str] = []
        self.titles: list[str] = []
        self.geometry_values: list[str] = []

    def title(self, value: str) -> None:
        self.titles.append(value)

    def geometry(self, value: str) -> None:
        self.geometry_values.append(value)

    def config(self, **kwargs: Any) -> None:
        self._menu = kwargs.get("menu", self._menu)

    def bind(self, sequence: str, func: Any) -> None:
        super().bind(sequence, func)

    def protocol(self, name: str, func: Any) -> None:
        self.protocols[name] = func

    def clipboard_clear(self) -> None:
        self.clipboard.clear()

    def clipboard_append(self, value: str) -> None:
        self.clipboard.append(value)

    def update_idletasks(self) -> None:
        return None

    def mainloop(self) -> None:
        return None


def _build_tk_module() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    tk_mod = cast(Any, ModuleType("tkinter"))
    ttk_mod = cast(Any, ModuleType("tkinter.ttk"))
    filedialog_mod = cast(Any, ModuleType("tkinter.filedialog"))
    messagebox_mod = cast(Any, ModuleType("tkinter.messagebox"))

    # tk module
    tk_mod.Widget = StubWidget
    tk_mod.Toplevel = StubToplevel
    tk_mod.Label = StubLabel
    tk_mod.Menu = StubMenu
    tk_mod.Canvas = StubCanvas
    tk_mod.Text = StubText
    tk_mod.Listbox = StubListbox
    tk_mod.StringVar = StubStringVar
    tk_mod.Tk = StubTk
    tk_mod.BOTH = "both"
    tk_mod.LEFT = "left"
    tk_mod.RIGHT = "right"
    tk_mod.TOP = "top"
    tk_mod.BOTTOM = "bottom"
    tk_mod.X = "x"
    tk_mod.Y = "y"
    tk_mod.END = "end"
    tk_mod.W = "w"
    tk_mod.E = "e"
    tk_mod.N = "n"
    tk_mod.S = "s"
    tk_mod.HORIZONTAL = "horizontal"
    tk_mod.VERTICAL = "vertical"
    tk_mod.SOLID = "solid"
    tk_mod.RAISED = "raised"
    tk_mod.SUNKEN = "sunken"
    tk_mod.MULTIPLE = "multiple"
    tk_mod.CENTER = "center"
    tk_mod.NW = "nw"
    tk_mod.NORMAL = "normal"
    tk_mod.DISABLED = "disabled"
    tk_mod.WORD = "word"

    # ttk module
    ttk_mod.Frame = StubFrame
    ttk_mod.Label = StubLabel
    ttk_mod.Button = StubButton
    ttk_mod.Entry = StubEntry
    ttk_mod.Combobox = StubCombobox
    ttk_mod.Spinbox = StubSpinbox
    ttk_mod.Radiobutton = StubRadiobutton
    ttk_mod.Scrollbar = StubScrollbar
    ttk_mod.Separator = StubSeparator
    ttk_mod.PanedWindow = StubPanedWindow
    ttk_mod.LabelFrame = StubLabelFrame
    ttk_mod.Treeview = StubTreeview

    # filedialog and messagebox are MagicMock-backed to observe calls
    filedialog_mod.askopenfilename = MagicMock()
    filedialog_mod.asksaveasfilename = MagicMock()
    messagebox_mod.showwarning = MagicMock()
    messagebox_mod.showinfo = MagicMock()
    messagebox_mod.showerror = MagicMock()
    messagebox_mod.askyesno = MagicMock()

    return tk_mod, ttk_mod, filedialog_mod, messagebox_mod


def _build_pandas_module() -> ModuleType:
    pandas_mod = cast(Any, ModuleType("pandas"))

    class StubDataFrame(MagicMock):
        pass

    pandas_mod.DataFrame = StubDataFrame
    pandas_mod.read_parquet = MagicMock(name="read_parquet")

    def _is_na(value: Any) -> bool:
        return value is None

    pandas_mod.isna = MagicMock(name="isna", side_effect=_is_na)
    return pandas_mod


@pytest.fixture()
def ui_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide patched tkinter/pandas modules and re-import UI modules under test."""
    tk_mod, ttk_mod, filedialog_mod, messagebox_mod = _build_tk_module()
    pandas_mod = _build_pandas_module()

    monkeypatch.setitem(sys.modules, "tkinter", tk_mod)
    monkeypatch.setitem(sys.modules, "tkinter.ttk", ttk_mod)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog_mod)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", messagebox_mod)
    monkeypatch.setitem(sys.modules, "pandas", pandas_mod)

    module_names = [
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.constants",
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.tk_helpers",
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.widgets",
        "lexile_corpus_tuner.lexile_scoring_model.pipeline_scripts.gutenberg_query_builder_ui.app",
    ]
    for name in module_names:
        sys.modules.pop(name, None)

    constants = importlib.import_module(module_names[0])
    tk_helpers = importlib.import_module(module_names[1])
    widgets = importlib.import_module(module_names[2])
    app = importlib.import_module(module_names[3])

    return SimpleNamespace(
        tk=tk_mod,
        ttk=ttk_mod,
        filedialog=filedialog_mod,
        messagebox=messagebox_mod,
        pandas=pandas_mod,
        constants=constants,
        tk_helpers=tk_helpers,
        widgets=widgets,
        app=app,
    )


@pytest.fixture(autouse=True)
def reset_ui_mocks(ui_modules: SimpleNamespace) -> None:
    """Reset shared MagicMocks between tests to preserve independence."""
    for mock in [
        ui_modules.filedialog.askopenfilename,
        ui_modules.filedialog.asksaveasfilename,
        ui_modules.messagebox.showwarning,
        ui_modules.messagebox.showinfo,
        ui_modules.messagebox.showerror,
        ui_modules.messagebox.askyesno,
    ]:
        mock.reset_mock()
