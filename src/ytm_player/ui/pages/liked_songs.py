"""Liked Songs page showing the user's liked music.

The module's pure decision cores are dual-implemented: the Rust
extension (``_ytm_player_native``) implements them when it is built, and
the pure-Python transcriptions below run otherwise — the empty-state
copy (``_empty_state``), the footer text (``_footer_text``), the
restore-cursor bounds gate (``_cursor_restore_row``), the nav-state
gate (``_nav_state``) and the Start-Radio seed draw (``_radio_seeds``,
the five-track sample the recently-played page's own Start Radio
shares). The bottom-of-module seam rebinds each to the native function
when the extension is importable. The workers, the table loads, the
tail fetch and the queue hand-off stay Python-side in the page.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, Static

from ytm_player.config.keymap import Action
from ytm_player.ui.track_filter import TRACK_FILTER_CSS, TrackFilterHost
from ytm_player.ui.widgets.track_table import TrackTable
from ytm_player.utils.formatting import normalize_tracks

if TYPE_CHECKING:
    from ytm_player.app._base import YTMHostBase

logger = logging.getLogger(__name__)

_LIKED_LOAD_FAILED_MSG = (
    "Couldn't load liked songs. Check the log at ~/.config/ytm-player/logs/ytm.log for details."
)


def _empty_state(load_failed: Any) -> str:
    """The empty-state copy — the message the loading label shows when
    the liked-music playlist rendered no rows.

    The ``bool`` of *load_failed* — the flag ``_load_liked_songs`` sets
    when the fetch itself failed (the service answers ``None`` on
    failure, never raises, so an empty playlist and a failed fetch
    render different messages) — picks the failure copy that points at
    the log file, over the genuine empty-state copy. The table/label
    visibility flips and the mount stay in the page.

    Implemented by the Rust extension
    (``_ytm_player_native.liked_empty_state``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if load_failed:
        return _LIKED_LOAD_FAILED_MSG
    return "No liked songs found."


def _footer_text(total: Any, loading_more: Any) -> str:
    """The footer text — the pure core of ``_update_footer``.

    The count interpolates through ``str`` — the conversion the
    f-string applies to ``table.track_count`` — and the truthiness of
    *loading_more* appends the ``" (loading more…)"`` suffix the first
    batch shows while the background top-up runs. The widget queries
    and the footer update stay in the page.

    Implemented by the Rust extension
    (``_ytm_player_native.liked_footer_text``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    text = f"{total} liked songs"
    if loading_more:
        text += " (loading more…)"
    return text


def _cursor_restore_row(row: Any, row_count: Any) -> Any:
    """The restore-cursor bounds gate — the row the liked table's
    cursor lands on after a load.

    The saved nav row answers itself when it is not ``None`` and
    ``0 <= row < row_count`` (the chained comparison evaluated left to
    right); anything else answers ``None``, the cursor staying where
    the load left it. The saved-row consume and the ``move_cursor``
    call stay in the page.

    Implemented by the Rust extension
    (``_ytm_player_native.liked_cursor_restore_row``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if row is not None and 0 <= row < row_count:
        return row
    return None


def _nav_state(cursor_row: Any) -> dict[str, Any]:
    """The nav-state gate — the state the page preserves when
    navigating away.

    The row is remembered only when it is not ``None`` and
    ``row > 0`` — a cursor still resting on the first row (or not laid
    out yet) needs no restore — and rides home as the very object it
    is. The ``query_one`` and its broad catch stay in the page.

    Implemented by the Rust extension
    (``_ytm_player_native.liked_nav_state``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    state: dict[str, Any] = {}
    if cursor_row is not None and cursor_row > 0:
        state["cursor_row"] = cursor_row
    return state


def _radio_seeds(tracks: Any) -> list:
    """The Start-Radio seed draw — the five-track sample a liked radio
    is seeded from.

    The draw size ``min(5, len(tracks))`` — capped at five and at the
    pool's own length — then ``random.sample`` routed through the
    ``random`` module, so ``random.seed(...)`` before the call and a
    monkeypatched ``random.sample`` steer this exactly like the native
    engine. The recently-played page's own Start Radio draws its seeds
    through the identical expression. The empty-tracks guard and the
    radio fetch stay in the pages.

    Implemented by the Rust extension
    (``_ytm_player_native.radio_seed_sample``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    import random

    return random.sample(tracks, min(5, len(tracks)))


class LikedSongsPage(TrackFilterHost, Widget):
    """Displays the user's Liked Music playlist."""

    _filter_table_id = "#liked-table"

    DEFAULT_CSS = (
        """
    LikedSongsPage {
        layout: vertical;
        width: 1fr;
        height: 1fr;
    }
    .liked-header {
        height: auto;
        max-height: 3;
        padding: 1 2;
        background: $surface;
    }
    .liked-header-row {
        height: auto;
        width: 1fr;
    }
    .liked-header-row Label {
        width: auto;
    }
    .liked-header-title {
        text-style: bold;
        color: $primary;
    }
    #start-radio-btn {
        width: auto;
        min-width: 14;
        height: 1;
        margin: 0 0 0 1;
        padding: 0 1;
        color: $primary;
    }
    #start-radio-btn:hover {
        background: $primary 30%;
    }
    .liked-footer {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        dock: bottom;
    }
    .liked-loading {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    """
        + TRACK_FILTER_CSS
    )

    track_count: reactive[int] = reactive(0)
    _load_failed: bool

    def __init__(self, *, cursor_row: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._restore_cursor_row = cursor_row
        # Set when ``_load_liked_songs`` catches a fetch failure so
        # ``_display_tracks`` renders the failure message instead of the
        # genuine empty-state copy.
        self._load_failed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="liked-header", classes="liked-header"):
            with Horizontal(classes="liked-header-row"):
                yield Label("Liked Songs", classes="liked-header-title")
                yield Static("[▶ Start Radio]", id="start-radio-btn", markup=True)
        yield Label("Loading liked songs...", id="liked-loading", classes="liked-loading")
        yield TrackTable(show_album=False, id="liked-table")
        yield Static("", id="liked-footer", classes="liked-footer")
        yield Input(placeholder="/ Filter tracks...", id="track-filter", classes="track-filter")

    def on_mount(self) -> None:
        self.query_one("#liked-table", TrackTable).display = False
        self.run_worker(self._load_liked_songs(), group="liked-load")

    # First batch size for progressive loading.
    _FIRST_BATCH = 300

    async def _load_liked_songs(self) -> None:
        ytmusic = self.app.ytmusic  # type: ignore[attr-defined]
        if not ytmusic:
            self.query_one("#liked-loading", Label).update("YouTube Music not connected.")
            return

        try:
            raw_tracks = await ytmusic.get_liked_songs(limit=self._FIRST_BATCH)
            # The service returns None on failure (never raises) so an empty
            # playlist and a failed fetch render different messages.
            self._load_failed = raw_tracks is None
            tracks = normalize_tracks(raw_tracks or [])
        except Exception:
            logger.exception("Failed to load liked songs")
            tracks = []
            self._load_failed = True

        self._display_tracks(tracks)

        # Kick off background fetch if the first batch was full (likely more tracks).
        if len(tracks) >= self._FIRST_BATCH:
            self._update_footer(loading_more=True)
            self.run_worker(self._fetch_remaining_liked(), group="liked-remaining")

    def _display_tracks(self, tracks: list[dict]) -> None:
        table = self.query_one("#liked-table", TrackTable)
        loading = self.query_one("#liked-loading", Label)

        if not tracks:
            table.display = False
            loading.update(_empty_state(self._load_failed))
            loading.display = True
            return

        loading.display = False
        table.display = True
        table.load_tracks(tracks)

        self.track_count = len(tracks)
        self._update_footer()

        # Restore cursor position from navigation state.
        row = self._restore_cursor_row
        self._restore_cursor_row = None
        target = _cursor_restore_row(row, table.row_count)
        if target is not None:
            table.move_cursor(row=target)

        # Land keyboard focus on the table so Tab / j / k have a starting point.
        table.focus()

    def _update_footer(self, loading_more: bool = False) -> None:
        try:
            table = self.query_one("#liked-table", TrackTable)
            footer = self.query_one("#liked-footer", Static)
            footer.update(_footer_text(table.track_count, loading_more))
        except Exception:
            pass

    async def _fetch_remaining_liked(self) -> None:
        """Background fetch for liked songs beyond the first batch."""
        from ytm_player.services.ytmusic import YTMusicService

        ytmusic = self.app.ytmusic  # type: ignore[attr-defined]
        if not ytmusic:
            return
        try:
            remaining_raw = await ytmusic.get_liked_songs(
                limit=None, timeout=YTMusicService._LARGE_PLAYLIST_TIMEOUT
            )
        except Exception:
            logger.debug("Background fetch for remaining liked songs failed", exc_info=True)
            self._update_footer()
            return

        if remaining_raw is None:
            # Background top-up failed; keep the first batch we already show.
            logger.debug("Background fetch for remaining liked songs failed")
            self._update_footer()
            return

        table = self.query_one("#liked-table", TrackTable)
        already_have = table.track_count

        remaining_raw = remaining_raw[already_have:]
        if not remaining_raw:
            self._update_footer()
            return

        remaining = normalize_tracks(remaining_raw)
        table.append_tracks(remaining)
        self.track_count = table.track_count
        self._update_footer()

    def get_nav_state(self) -> dict[str, Any]:
        """Return state to preserve when navigating away."""
        try:
            table = self.query_one("#liked-table", TrackTable)
            return _nav_state(table.cursor_row)
        except Exception:
            return {}

    async def on_track_table_track_selected(self, event: TrackTable.TrackSelected) -> None:
        event.stop()
        table = self.query_one("#liked-table", TrackTable)
        host = cast("YTMHostBase", self.app)
        await host._replace_queue_and_play(
            table.tracks,
            entity_id=self._CONTEXT_ID,
            start_index=event.index,
            autoplay=False,
        )
        await host.play_track(event.track)

    _CONTEXT_ID = "__LIKED_SONGS__"

    async def handle_action(self, action: Action, count: int = 1) -> None:
        table = self.query_one("#liked-table", TrackTable)

        match action:
            case _:
                await table.handle_action(action, count)

    def on_click(self, event: Click) -> None:
        if event.widget is not None and event.widget.id == "start-radio-btn":
            event.stop()
            self.run_worker(
                self._start_radio(), name="start_radio", group="start-radio", exclusive=True
            )

    async def _start_radio(self) -> None:
        table = self.query_one("#liked-table", TrackTable)
        tracks = table.tracks
        if not tracks:
            return
        seeds = _radio_seeds(tracks)
        host = cast("YTMHostBase", self.app)
        await host._fetch_and_play_radio(seeds, label="Radio: Liked Songs")


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same liked-songs decision cores;
# these bindings replace the pure-Python transcriptions above when the
# extension is importable.  The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        liked_cursor_restore_row as _native_liked_cursor_restore_row,
    )
    from ytm_player.utils._ytm_player_native import (
        liked_empty_state as _native_liked_empty_state,
    )
    from ytm_player.utils._ytm_player_native import (
        liked_footer_text as _native_liked_footer_text,
    )
    from ytm_player.utils._ytm_player_native import (
        liked_nav_state as _native_liked_nav_state,
    )
    from ytm_player.utils._ytm_player_native import (
        radio_seed_sample as _native_radio_seed_sample,
    )
except ImportError:
    pass
else:
    _empty_state = _native_liked_empty_state  # type: ignore[assignment]
    _footer_text = _native_liked_footer_text  # type: ignore[assignment]
    _cursor_restore_row = _native_liked_cursor_restore_row  # type: ignore[assignment]
    _nav_state = _native_liked_nav_state  # type: ignore[assignment]
    _radio_seeds = _native_radio_seed_sample  # type: ignore[assignment]
