"""SelectionInfoBar — fixed 1-row bar showing the currently-focused item's full name."""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

logger = logging.getLogger(__name__)


def _visible_width(width: int) -> int:
    """The truncation width — the ``width`` engine.

    The pure core of ``SelectionInfoBar.watch_text``: the character
    budget the bar's text is truncated to — ``80`` while the widget
    has no width yet (not laid out), otherwise the widget width minus
    the bar's 1-column padding on each side, floored at ``10``
    (``max(width - 2, 10)``, CPython's own two-argument ``max`` — a
    zero width reads the default; any other width, negative ones
    included, the floor). The truncation itself is the already-ported
    formatting engine; the Static update and the untruncated fallback
    stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.selection_info_visible_width``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return max(width - 2, 10) if width else 80


class SelectionChanged(Message):
    """Posted by widgets when their currently-selected/focused item changes.

    Bubbles up the DOM; SelectionInfoBar handles it and updates its display.
    Empty *text* signals no current selection (bar shows nothing).
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class SelectionInfoBar(Widget):
    """A 1-row bar that shows the full name of the currently-focused item.

    Displays text from SelectionChanged messages. Centered, single line.
    Hidden entirely when `[ui] show_selection_info = false` (display = False
    so the row is reclaimed by the layout).
    """

    DEFAULT_CSS = """
    SelectionInfoBar {
        height: 1;
        width: 100%;
        color: $text-muted;
        content-align: center middle;
        padding: 0 1;
    }
    """

    text: reactive[str] = reactive("")

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self._inner: Static | None = None

    def compose(self) -> ComposeResult:
        self._inner = Static("", id="selection-info-text")
        yield self._inner

    def watch_text(self, old: str, new: str) -> None:
        """Reactive watcher — re-render the inner Static when text changes."""
        if self._inner is None:
            return
        try:
            from ytm_player.utils.formatting import truncate

            # The width choice is resolved by the ``_visible_width``
            # engine (native when the Rust extension is built, the
            # pure-Python transcription otherwise); the truncate call
            # and the update stay here.
            visible_width = _visible_width(self.size.width)
            self._inner.update(truncate(new, visible_width))
        except Exception:
            logger.exception("SelectionInfoBar.watch_text failed")
            try:
                self._inner.update(new)
            except Exception:
                pass

    def on_resize(self, _event) -> None:
        """Re-truncate on resize so the visible text fits the new width."""
        try:
            self.watch_text(self.text, self.text)
        except Exception:
            pass

    async def on_selection_changed(self, message: SelectionChanged) -> None:
        """Receive a SelectionChanged message bubbled from a descendant."""
        message.stop()
        self.text = message.text


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same width engine; this binding
# overrides the pure-Python transcription above when the extension is
# importable. The rebinding must happen AFTER the definition above —
# assigning the native function any earlier would be clobbered by the
# ``def`` statement (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        selection_info_visible_width as _native_selection_info_visible_width,
    )
except ImportError:
    pass
else:

    def _visible_width(width: int) -> int:  # noqa: F811
        """The truncation width (native Rust engine)."""
        return _native_selection_info_visible_width(width)
