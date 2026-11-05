"""Terminal capability detection."""

from __future__ import annotations

import os
import shutil


def detect_image_protocol() -> str | None:
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    term = os.environ.get("TERM", "").lower()

    if term_program == "kitty" or "kitty" in term:
        return "kitty"

    if term_program in ("iterm.app", "iterm2"):
        return "iterm2"

    if term_program == "wezterm":
        return "iterm2"

    term_features = os.environ.get("TERM_FEATURES", "").lower()
    colorterm = os.environ.get("COLORTERM", "").lower()
    if "sixel" in term or "sixel" in term_features or "sixel" in colorterm:
        return "sixel"

    return "block"


def get_terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def get_orientation(cols: int, rows: int) -> str:
    """The aspect-ratio verdict: ``"horizontal"`` when *rows* is zero or
    the ``cols / rows`` ratio exceeds 2.3, ``"vertical"`` otherwise.

    Implemented by the Rust extension
    (``_ytm_player_native.get_orientation``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return "horizontal" if rows == 0 or cols / rows > 2.3 else "vertical"


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same gate; this binding replaces the
# pure-Python definition above when the extension is importable.  The
# rebinding must happen AFTER the definition above — assigning the native
# function any earlier would be clobbered by the ``def`` statement.

try:
    from ._ytm_player_native import (  # type: ignore[import-not-found]
        get_orientation as _native_get_orientation,
    )
except ImportError:
    pass
else:
    get_orientation = _native_get_orientation  # type: ignore[assignment]
