"""Shared track-filter wiring for track-list pages.

``TrackFilterHost`` is a plain mixin (no ``__init__``, no ``super()``
calls) providing the ``/``-to-filter behaviour that every track-list page
shares: pressing ``/`` reveals a docked filter input, typing filters the
table live, Enter hides the input while keeping the filter, and Escape
clears the filter and refocuses the table.

Each host page declares ``_filter_table_id`` (the ``#id`` selector of its
``TrackTable``) and embeds :data:`TRACK_FILTER_CSS` in its ``DEFAULT_CSS``.
Textual only collects ``DEFAULT_CSS`` from ``DOMNode`` bases along a
widget's inheritance chain, so a plain ``object`` mixin's ``DEFAULT_CSS``
is ignored — hence the CSS lives in a module constant the pages concatenate
rather than on the mixin.

Under ``TYPE_CHECKING`` the mixin inherits from ``Widget`` so Pyright can
resolve ``query_one`` etc.; at runtime it inherits from ``object`` so it
stays a genuine mixin. Pages must list it BEFORE the Textual base class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.events import Key
from textual.widgets import Input

from ytm_player.ui.widgets.track_table import TrackTable

if TYPE_CHECKING:
    from textual.widget import Widget

    _MixinBase = Widget
else:
    _MixinBase = object


TRACK_FILTER_CSS = """
.track-filter {
    dock: bottom;
    display: none;
}
.track-filter.visible {
    display: block;
}
"""


def _owns_input(input_id: str | None) -> bool:
    """The input-ownership gate — the ``owns`` engine.

    The pure core of ``TrackFilterHost.on_input_changed`` and
    ``on_input_submitted``: only the shared ``#track-filter`` Input's
    events are the filter's — a widget with no id, or any other id,
    is not ours (``Input.id`` is ``str | None``; ``None ==
    "track-filter"`` is ``False``, CPython's own ``==`` on the None
    literal).

    Implemented by the Rust extension
    (``_ytm_player_native.track_filter_owns_input``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return input_id == "track-filter"


def _escape_clears(key: str, filter_visible: bool) -> bool:
    """The escape-clear verdict — the ``escape`` engine.

    The pure core of ``TrackFilterHost.on_key``: Escape clears the
    filter (hides the input, refocuses the table) exactly when the key
    is ``"escape"`` and the filter input is still visible — an Escape
    after Enter hid the input (the keep-filter path) leaves the page
    alone. The event stopping, the clear, the class removal and the
    refocus stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_filter_escape_clears``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return key == "escape" and filter_visible


class TrackFilterHost(_MixinBase):
    """Mixin wiring the ``/`` track filter for track-list pages."""

    # ``#id`` selector of the host page's TrackTable. Set by each page.
    _filter_table_id: str

    def on_track_table_filter_requested(self, event: TrackTable.FilterRequested) -> None:
        event.stop()
        try:
            f = self.query_one("#track-filter", Input)
            f.value = ""
            f.add_class("visible")
            f.focus()
        except Exception:
            pass

    def on_track_table_filter_closed(self, event: TrackTable.FilterClosed) -> None:
        event.stop()
        try:
            f = self.query_one("#track-filter", Input)
            f.remove_class("visible")
            self.query_one(self._filter_table_id, TrackTable).focus()
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        # The ownership gate is resolved by the ``_owns_input`` engine
        # (native when the Rust extension is built, the pure-Python
        # transcription otherwise); the table call stays here.
        if _owns_input(event.input.id):
            try:
                self.query_one(self._filter_table_id, TrackTable).apply_filter(event.value)
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # The ownership gate is resolved by the ``_owns_input`` engine
        # (native when the Rust extension is built, the pure-Python
        # transcription otherwise); the hide and the refocus stay here.
        if _owns_input(event.input.id):
            try:
                f = self.query_one("#track-filter", Input)
                f.remove_class("visible")
                self.query_one(self._filter_table_id, TrackTable).focus()
            except Exception:
                pass

    def on_key(self, event: object) -> None:
        """Clear the filter, hide the input and refocus the table on Escape."""
        if not isinstance(event, Key):
            return
        # The escape verdict is resolved by the ``_escape_clears``
        # engine (native when the Rust extension is built, the
        # pure-Python transcription otherwise); the event stopping, the
        # clear, the class removal and the refocus stay here.
        try:
            f = self.query_one("#track-filter", Input)
            if _escape_clears(event.key, f.has_class("visible")):
                event.stop()
                event.prevent_default()
                table = self.query_one(self._filter_table_id, TrackTable)
                table.clear_filter()
                f.remove_class("visible")
                table.focus()
        except Exception:
            pass


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same track-filter engines; these
# bindings override the pure-Python transcriptions above when the
# extension is importable. The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_filter_escape_clears as _native_track_filter_escape_clears,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_filter_owns_input as _native_track_filter_owns_input,
    )
except ImportError:
    pass
else:

    def _owns_input(input_id: str | None) -> bool:  # noqa: F811
        """The input-ownership gate (native Rust engine)."""
        return _native_track_filter_owns_input(input_id)

    def _escape_clears(key: str, filter_visible: bool) -> bool:  # noqa: F811
        """The escape-clear verdict (native Rust engine)."""
        return _native_track_filter_escape_clears(key, filter_visible)
