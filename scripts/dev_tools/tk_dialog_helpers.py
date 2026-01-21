"""Tkinter file-dialog helpers with HiDPI scaling.

Purpose:
    Provide a small, reusable seam for dev tools that want a standard OS
    file-open dialog while avoiding Tkinter typing friction under strict type
    checking.

Notes:
    - Tkinter is imported dynamically, because it may not be installed in
      minimal/headless environments.
    - Scaling is best-effort. It should never crash callers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def resolve_workspace_root(script_path: Path | None = None) -> Path:
    """Resolve the repository root.

    Purpose:
        Dev tools in this repo live under `scripts/dev_tools/`. The repo root is
        therefore two directories above this file.

    Args:
        script_path (Path | None): Optional path override for tests.

    Returns:
        Path: Repo root directory.
    """

    anchor = script_path or Path(__file__)
    return anchor.resolve().parents[2]


def resolve_initial_dir(
    *,
    workspace_root: Path,
    relative_start_dir: Path,
    exists: Callable[[Path], bool],
) -> Path | None:
    """Resolve an initial directory for a file dialog.

    Purpose:
        Provide a friendly starting folder for GUI selection, while remaining
        robust when the folder doesn't exist yet.

    Args:
        workspace_root (Path): Repo root.
        relative_start_dir (Path): Relative path to use as the starting folder.
        exists (Callable[[Path], bool]): Existence check (injected for tests).

    Returns:
        Path | None: Absolute directory path if it exists, otherwise None.
    """

    candidate = workspace_root / relative_start_dir
    if exists(candidate):
        return candidate
    return None


def pick_file_with_tkinter(*, title: str, initial_dir: Path | None) -> Path | None:
    """Pick a file using Tkinter's standard file-open dialog.

    Purpose:
        Use the platform's standard file picker (via Tk) when available. Return
        None when Tk is not installed or the user cancels.

    Args:
        title (str): Dialog title.
        initial_dir (Path | None): Optional initial directory hint.

    Returns:
        Path | None: Selected file path, or None if cancelled/unavailable.
    """

    try:
        import importlib

        # Import Tkinter dynamically to avoid strict-mode typing friction while
        # keeping callers fully typed.
        tk_mod = cast(Any, importlib.import_module("tkinter"))
        filedialog_mod = cast(Any, importlib.import_module("tkinter.filedialog"))
    except ImportError:
        return None

    # Ensure Windows won't bitmap-scale Tk at low DPI, which can produce tiny UI.
    _enable_windows_dpi_awareness()

    root = tk_mod.Tk()

    # Configure Tk scaling before showing dialogs so widgets/dialogs inherit it.
    # This is especially important on Linux where Tk frequently ignores desktop
    # scaling and reports a constant 96 DPI.
    _configure_tk_scaling(root, env=os.environ)

    root.withdraw()
    try:
        dialog_kwargs: dict[str, object] = {"title": title}
        if initial_dir is not None:
            dialog_kwargs["initialdir"] = str(initial_dir)
        selected = filedialog_mod.askopenfilename(**dialog_kwargs)
    finally:
        # Always clean up the hidden root window.
        root.destroy()

    if not selected:
        return None
    return Path(str(selected))


def _enable_windows_dpi_awareness() -> None:
    """Enable DPI awareness on Windows.

    Purpose:
        Tkinter on Windows can be bitmap-scaled when the process is not marked
        DPI-aware, which often shows up as tiny or blurry dialogs on HiDPI
        displays. This function attempts the best available DPI-awareness API
        and becomes a no-op on non-Windows platforms.

    Side Effects:
        Updates process DPI awareness for the current process (Windows only).
    """

    if sys.platform != "win32":
        return

    # Prefer modern per-monitor DPI awareness when available; fall back to the
    # older system DPI-aware API.
    shcore_configured = False
    try:
        import ctypes

        shcore = ctypes.windll.shcore
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        shcore.SetProcessDpiAwareness(2)
    except Exception:
        shcore_configured = False
    else:
        shcore_configured = True

    if shcore_configured:
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
    except Exception:
        return


def _parse_env_float(env: Mapping[str, str], key: str) -> float | None:
    """Parse a float from an environment variable if present and valid."""

    raw = env.get(key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _compute_physical_dpi(
    *, screen_px: int | None, screen_mm: int | None
) -> float | None:
    """Estimate physical DPI using screen pixels and millimeters."""

    if not screen_px or not screen_mm or screen_px <= 0 or screen_mm <= 0:
        return None

    inches = screen_mm / 25.4
    if inches <= 0:
        return None

    dpi = screen_px / inches

    # Guard against bogus EDID values. These bounds are intentionally wide but
    # exclude the most common failure modes (0 mm, 1 mm, 10,000 mm, etc.).
    if dpi < 60 or dpi > 400:
        return None
    return dpi


def compute_tk_scaling(
    *,
    env: Mapping[str, str],
    logical_dpi: float | None,
    screen_px: int | None,
    screen_mm: int | None,
) -> float:
    """Compute a best-effort Tk scaling factor for HiDPI environments."""

    physical_dpi = _compute_physical_dpi(screen_px=screen_px, screen_mm=screen_mm)

    # Use Tk-reported DPI when present; otherwise fall back to a common default.
    base_dpi = logical_dpi if logical_dpi and logical_dpi > 0 else 96.0

    # If physical DPI is meaningfully larger, it's a strong signal that Tk isn't
    # honoring desktop scaling.
    if physical_dpi is not None and physical_dpi > base_dpi * 1.25:
        base_dpi = physical_dpi

    scaling = base_dpi / 72.0

    # Collect desktop environment scaling hints (best-effort; many setups won't
    # provide them).
    gdk_scale = _parse_env_float(env, "GDK_SCALE")
    qt_scale = _parse_env_float(env, "QT_SCALE_FACTOR")
    override = _parse_env_float(env, "LEXILE_TK_SCALE")

    # Apply desktop hints only when Tk appears unscaled (roughly <= 125%).
    # This avoids double-scaling when Tk already reports a high DPI.
    if scaling <= (120.0 / 72.0):
        hinted = max([v for v in [gdk_scale, qt_scale] if v and v > 0] or [1.0])
        if hinted > 1.0:
            scaling *= hinted

    # Allow an explicit multiplier to force the desired size (useful on Linux
    # where accurate scaling detection can be unreliable).
    if override is not None and override > 0:
        scaling *= override

    # Clamp to a safe range to prevent unusable UIs if detection goes sideways.
    if scaling < 0.5:
        return 0.5
    if scaling > 10.0:
        return 10.0
    return scaling


def _configure_tk_scaling(root: Any, *, env: Mapping[str, str]) -> None:
    """Configure Tk scaling on a root window."""

    try:
        # Gather screen info while a root exists.
        logical_dpi = float(root.winfo_fpixels("1i"))
        screen_px = int(root.winfo_screenwidth())
        screen_mm = int(root.winfo_screenmmwidth())
        scaling = compute_tk_scaling(
            env=env,
            logical_dpi=logical_dpi,
            screen_px=screen_px,
            screen_mm=screen_mm,
        )

        # Apply scaling before showing any UI.
        root.tk.call("tk", "scaling", scaling)
    except Exception:
        return
