"""Top header bar with sidebar toggle buttons."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


def _click_route(target_id: str | None) -> str:
    """The click-routing verdict — the ``route`` engine.

    The pure core of ``HeaderBar.on_click``: which of the four toggle
    routes a click on the bar takes — ``"playlist"`` for the playlist
    toggle, ``"lyrics"`` for the lyrics toggle, ``"back"`` /
    ``"forward"`` for the navigation buttons — or ``"none"`` for any
    other id, and for a widget with no id at all. The message
    construction and the event stopping stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.header_bar_click_route``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if target_id == "toggle-playlist":
        return "playlist"
    if target_id == "toggle-lyrics":
        return "lyrics"
    if target_id == "back-btn":
        return "back"
    if target_id == "forward-btn":
        return "forward"
    return "none"


def _lyrics_class(is_dimmed: bool, is_on: bool) -> str | None:
    """The lyrics-toggle class choice — the ``class`` engine.

    The pure core of ``HeaderBar._apply_lyrics_classes``:
    ``"dimmed"`` when the lyrics pane is dimmed (open without a synced
    payload), ``"active"`` when it is open and lit, ``None`` when it
    is closed — the caller removes both classes first and then adds
    the answer, exactly the reference's remove-then-add.

    Implemented by the Rust extension
    (``_ytm_player_native.header_bar_lyrics_class``) when it is built;
    this pure-Python transcription is the fallback otherwise. The DOM
    reads and writes stay with the caller.
    """
    if is_dimmed:
        return "dimmed"
    if is_on:
        return "active"
    return None


class HeaderBar(Widget):
    """Top bar with toggle buttons for playlist and lyrics sidebars."""

    class TogglePlaylistSidebar(Message):
        """Emitted when the playlist sidebar toggle is clicked."""

    class ToggleLyricsSidebar(Message):
        """Emitted when the lyrics sidebar toggle is clicked."""

    class BackRequested(Message):
        """Emitted when the back button is clicked."""

    class ForwardRequested(Message):
        """Emitted when the forward button is clicked."""

    DEFAULT_CSS = """
    HeaderBar {
        dock: top;
        height: 1;
        background: $playback-bar-bg;
    }
    HeaderBar #header-inner {
        height: 1;
        width: 1fr;
    }
    HeaderBar #header-spacer {
        width: 1fr;
        height: 1;
    }
    HeaderBar .hb-toggle {
        height: 1;
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }
    HeaderBar .hb-toggle:hover {
        background: $accent 30%;
    }
    HeaderBar .hb-toggle.active {
        color: $primary;
        text-style: bold;
    }
    HeaderBar .hb-toggle.dimmed {
        text-style: dim;
    }
    HeaderBar #back-btn {
        display: none;
    }
    HeaderBar #back-btn.visible {
        display: block;
    }
    HeaderBar #forward-btn {
        display: none;
    }
    HeaderBar #forward-btn.visible {
        display: block;
    }
    """

    is_playlist_on: reactive[bool] = reactive(False)
    is_lyrics_on: reactive[bool] = reactive(False)
    is_lyrics_dimmed: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        with Horizontal(id="header-inner"):
            yield Static("\u2630 Playlists", id="toggle-playlist", classes="hb-toggle")
            yield Static("\u2190 Back", id="back-btn", classes="hb-toggle")
            yield Static("Forward \u2192", id="forward-btn", classes="hb-toggle")
            yield Static(id="header-spacer")
            yield Static("\u266b Lyrics", id="toggle-lyrics", classes="hb-toggle")

    def on_click(self, event: Click) -> None:
        """Route clicks on toggle buttons to the correct message."""
        target = event.widget
        if target is None:
            return
        # The routing itself is resolved by the ``_click_route`` engine
        # (native when the Rust extension is built, the pure-Python
        # transcription otherwise); the message construction and the
        # event stopping stay here.
        route = _click_route(target.id)
        if route == "playlist":
            event.stop()
            self.post_message(self.TogglePlaylistSidebar())
        elif route == "lyrics":
            event.stop()
            self.post_message(self.ToggleLyricsSidebar())
        elif route == "back":
            event.stop()
            self.post_message(self.BackRequested())
        elif route == "forward":
            event.stop()
            self.post_message(self.ForwardRequested())

    # ── State updates ────────────────────────────────────────────────

    def set_playlist_state(self, is_open: bool) -> None:
        self.is_playlist_on = is_open
        try:
            btn = self.query_one("#toggle-playlist", Static)
            if is_open:
                btn.add_class("active")
            else:
                btn.remove_class("active")
        except Exception:
            pass

    def set_lyrics_state(self, is_open: bool) -> None:
        self.is_lyrics_on = is_open
        self._apply_lyrics_classes()

    def set_lyrics_dimmed(self, dimmed: bool) -> None:
        self.is_lyrics_dimmed = dimmed
        self._apply_lyrics_classes()

    def _apply_lyrics_classes(self) -> None:
        # The class choice is resolved by the ``_lyrics_class`` engine
        # (native when the Rust extension is built, the pure-Python
        # transcription otherwise); the DOM reads and writes stay here.
        try:
            btn = self.query_one("#toggle-lyrics", Static)
            btn.remove_class("active", "dimmed")
            cls = _lyrics_class(self.is_lyrics_dimmed, self.is_lyrics_on)
            if cls is not None:
                btn.add_class(cls)
        except Exception:
            pass

    def set_back_visible(self, visible: bool) -> None:
        """Show/hide the back button. Called when the nav stack changes."""
        try:
            btn = self.query_one("#back-btn", Static)
            if visible:
                btn.add_class("visible")
            else:
                btn.remove_class("visible")
        except Exception:
            pass

    def set_forward_visible(self, visible: bool) -> None:
        """Show/hide the forward button. Called when the forward stack changes."""
        try:
            btn = self.query_one("#forward-btn", Static)
            if visible:
                btn.add_class("visible")
            else:
                btn.remove_class("visible")
        except Exception:
            pass


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same header-bar engines; these
# bindings override the pure-Python transcriptions above when the
# extension is importable. The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        header_bar_click_route as _native_header_bar_click_route,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        header_bar_lyrics_class as _native_header_bar_lyrics_class,
    )
except ImportError:
    pass
else:

    def _click_route(target_id: str | None) -> str:  # noqa: F811
        """The click-routing verdict (native Rust engine)."""
        return _native_header_bar_click_route(target_id)

    def _lyrics_class(is_dimmed: bool, is_on: bool) -> str | None:  # noqa: F811
        """The lyrics-toggle class choice (native Rust engine)."""
        return _native_header_bar_lyrics_class(is_dimmed, is_on)
