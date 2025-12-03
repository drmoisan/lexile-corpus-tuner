"""Tkinter and pandas helper functions with isolated type suppressions.

All tkinter widget operations and pandas DataFrame operations with
incomplete type stubs are isolated in this module. Application logic
remains fully typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import tkinter as tk
    from pathlib import Path
    from tkinter import ttk

import pandas as pd


def tk_listbox_curselection(listbox: tk.Listbox) -> tuple[int, ...]:  # type: ignore[misc]
    """Get selected indices from Listbox.

    Isolated tkinter operation: tkinter stubs incomplete for curselection().

    Args:
        listbox: Listbox widget

    Returns:
        Tuple of selected indices
    """
    return listbox.curselection()  # type: ignore[no-any-return]


def tk_listbox_get(listbox: tk.Listbox, index: int) -> str:
    """Get item at index from Listbox.

    Isolated tkinter operation: tkinter stubs incomplete for get().

    Args:
        listbox: Listbox widget
        index: Item index

    Returns:
        Item text
    """
    return listbox.get(index)  # type: ignore[no-any-return]


def tk_listbox_yview(listbox: tk.Listbox, *args: Any) -> None:
    """Scroll Listbox vertically.

    Isolated tkinter operation: tkinter stubs incomplete for yview().

    Args:
        listbox: Listbox widget
        args: Scroll arguments
    """
    listbox.yview(*args)  # type: ignore[no-untyped-call]


def tk_canvas_yview(canvas: tk.Canvas, *args: Any) -> None:
    """Scroll Canvas vertically.

    Isolated tkinter operation: tkinter stubs incomplete for yview().

    Args:
        canvas: Canvas widget
        args: Scroll arguments
    """
    canvas.yview(*args)  # type: ignore[no-untyped-call]


def tk_treeview_xview(treeview: ttk.Treeview, *args: Any) -> None:
    """Scroll Treeview horizontally.

    Isolated tkinter operation: tkinter stubs incomplete for xview().

    Args:
        treeview: Treeview widget
        args: Scroll arguments
    """
    treeview.xview(*args)  # type: ignore[no-untyped-call]


def tk_treeview_yview(treeview: ttk.Treeview, *args: Any) -> None:
    """Scroll Treeview vertically.

    Isolated tkinter operation: tkinter stubs incomplete for yview().

    Args:
        treeview: Treeview widget
        args: Scroll arguments
    """
    treeview.yview(*args)  # type: ignore[no-untyped-call]


def tk_panedwindow_add(paned: ttk.PanedWindow, child: tk.Widget, **kw: Any) -> None:
    """Add child widget to PanedWindow.

    Isolated tkinter operation: tkinter stubs incomplete for add().

    Args:
        paned: PanedWindow widget
        child: Child widget to add
        kw: Additional keyword arguments
    """
    paned.add(child, **kw)  # type: ignore[no-untyped-call]


def pandas_read_parquet(path: Path) -> pd.DataFrame:  # type: ignore[type-arg]
    """Read parquet file.

    Isolated pandas operation: pandas-stubs incomplete for read_parquet.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame
    """
    return pd.read_parquet(path)  # type: ignore[no-any-return]


def pandas_to_csv(df: pd.DataFrame, path: Path) -> None:  # type: ignore[type-arg]
    """Write DataFrame to CSV.

    Isolated pandas operation: pandas-stubs incomplete for to_csv.

    Args:
        df: DataFrame to write
        path: Output path
    """
    df.to_csv(path, index=False)  # type: ignore[call-overload]


def pandas_to_parquet(df: pd.DataFrame, path: Path) -> None:  # type: ignore[type-arg]
    """Write DataFrame to Parquet.

    Isolated pandas operation: pandas-stubs incomplete for to_parquet.

    Args:
        df: DataFrame to write
        path: Output path
    """
    df.to_parquet(path, index=False)  # type: ignore[call-overload]


def pandas_is_na(value: Any) -> bool:
    """Check if value is NA/NaN.

    Isolated pandas operation: pandas-stubs incomplete for isna().

    Args:
        value: Value to check

    Returns:
        True if NA/NaN, False otherwise
    """
    return pd.isna(value)  # type: ignore[no-any-return]


def pandas_get_column(df: pd.DataFrame, column: str) -> Any:  # type: ignore[type-arg,misc]
    """Get column from DataFrame.

    Isolated pandas operation: pandas-stubs incomplete for column access.

    Args:
        df: DataFrame
        column: Column name

    Returns:
        Series or value
    """
    return df[column]  # type: ignore[no-any-return]
