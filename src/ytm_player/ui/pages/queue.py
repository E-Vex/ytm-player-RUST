"""Queue management page showing the playback queue.

The page's view and mutation engines — the now-playing header view model
(shared by the lightweight track-change update and the full refresh),
the radio-seed source label, the footer text, and the decisions behind
the remove/reorder mutations (the queue-position mapping, the
post-remove cursor re-landing, the clamped reorder destination) — are
implemented by the Rust extension (``_ytm_player_native``) when it is
built and by the pure-Python transcriptions in this module otherwise;
the bottom-of-module seam rebinds the engines to the native functions
when the extension is importable. The methods below only do I/O and
rendering.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, Static

from ytm_player.config.keymap import Action
from ytm_player.config.settings import get_settings
from ytm_player.services.player import PlayerEvent
from ytm_player.ui.track_filter import TRACK_FILTER_CSS, TrackFilterHost
from ytm_player.ui.widgets.track_table import TrackTable

if TYPE_CHECKING:
    from ytm_player.app._base import YTMHostBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Queue-page view & mutation engines
# ---------------------------------------------------------------------------


def _now_playing_view(track: dict[str, Any] | None) -> dict[str, Any] | None:
    """The now-playing header's view model — the pure core shared by
    ``_update_current_track`` and ``_refresh_queue``.

    A falsy track (the ``None`` a queue with nothing selected reports)
    hides the header and answers ``None``; otherwise the view carries
    ``title_label`` — ``f"Now Playing: {title}"`` over the title read as
    ``track.get("title", "Unknown")`` (absent key ``"Unknown"``, a
    present value whatever it is) — ``artist`` —
    ``track.get("artist", "Unknown")`` raw, the object the Label mounts
    — and ``video_id`` — ``track.get("video_id", "")`` raw, the value
    ``set_playing`` receives to light the playing row.

    Implemented by the Rust extension
    (``_ytm_player_native.queue_now_playing``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if not track:
        return None
    title = track.get("title", "Unknown")
    artist = track.get("artist", "Unknown")
    return {
        "title_label": f"Now Playing: {title}",
        "artist": artist,
        "video_id": track.get("video_id", ""),
    }


def _queue_source_view(seeds: list[dict[str, Any]] | None, show: bool) -> tuple[str, str] | None:
    """The seed-source label's view model — the pure core of
    ``_update_queue_source``.

    The label stays hidden (``None``) when there are no radio seeds or
    the ``show_queue_source`` setting is off; otherwise each seed's
    title — ``seed.get("title", "Unknown")`` — feeds both the one-line
    summary (``", ".join(titles)`` for up to three seeds,
    ``"{first}, {second} + N more"`` from the fourth on) and the
    tooltip's bulleted ``"Generated from:"`` block, returned as
    ``(text, tooltip)``.

    Implemented by the Rust extension
    (``_ytm_player_native.queue_source_view``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if not (seeds and show):
        return None
    titles = [s.get("title", "Unknown") for s in seeds]
    if len(titles) <= 3:
        summary = ", ".join(titles)
    else:
        summary = f"{titles[0]}, {titles[1]} + {len(titles) - 2} more"
    return (
        f"Generated from: {summary}",
        "Generated from:\n" + "\n".join(f"  • {t}" for t in titles),
    )


def _queue_footer_text(count: int) -> str:
    """The footer bar's text — the pure core of ``_update_footer``.

    ``f"Tracks: {count}"`` over the queue's length, so the emptied
    queue's ``"Tracks: 0"`` reads exactly as the reference's f-string
    renders it.

    Implemented by the Rust extension
    (``_ytm_player_native.queue_footer_text``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return f"Tracks: {count}"


def _remove_target(selected_index: int | None, queue_length: int) -> int | None:
    """The queue-position decision — the pure half of
    ``_remove_selected``, the queue page's remove mutation.

    The highlighted row's original-order index names a removable queue
    position only when it is present and ``0 <= index < queue_length``
    (a stale table — one the queue shrank underneath without a refresh
    — refuses rather than removing the wrong entry); ``None`` means "no
    removal".

    Implemented by the Rust extension
    (``_ytm_player_native.queue_remove_target``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if selected_index is None or not 0 <= selected_index < queue_length:
        return None
    return selected_index


def _post_remove_cursor_row(visible_row: int, row_count: int) -> int | None:
    """The cursor re-landing — the pure half of the remove mutation's
    optimistic UI update.

    After the removal and the refresh, the row that took the removed
    entry's place in the displayed order gets the cursor —
    ``min(visible_row, row_count - 1)``, the visible row the cursor sat
    on before the mutation, clamped into the refreshed table's bounds —
    and ``None`` (cursor untouched) when the refreshed table has no
    rows left.

    Implemented by the Rust extension
    (``_ytm_player_native.queue_post_remove_cursor``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if row_count <= 0:
        return None
    return min(visible_row, row_count - 1)


def _move_target(
    from_index: int | None, direction: int, count: int, queue_length: int
) -> int | None:
    """The reorder destination — the pure half of ``_move_track``, the
    queue page's move mutation.

    No highlighted row (``None``) or a destination equal to the source —
    including the clamped-at-the-bounds no-ops a too-large count
    produces — answers ``None`` (nothing moves); otherwise the
    destination is ``max(0, min(from + direction * count, queue_length
    - 1))``: the count-multiplied queue positions up or down, clamped
    to the queue bounds.

    Implemented by the Rust extension
    (``_ytm_player_native.queue_move_target``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if from_index is None:
        return None
    to_index = max(0, min(from_index + direction * count, queue_length - 1))
    if to_index == from_index:
        return None
    return to_index


class QueuePage(TrackFilterHost, Widget):
    """Displays and manages the playback queue.

    Shows the currently playing track at the top, the upcoming queue in a
    table, and repeat/shuffle state in a footer bar. Supports reordering,
    removal, and jumping to tracks.
    """

    _filter_table_id = "#queue-table"

    DEFAULT_CSS = (
        """
    QueuePage {
        layout: vertical;
        width: 1fr;
        height: 1fr;
    }
    .queue-now-playing {
        height: auto;
        max-height: 3;
        padding: 1 2;
        background: $surface;
    }
    .queue-now-playing-title {
        text-style: bold;
        color: $primary;
    }
    .queue-now-playing-artist {
        color: $text-muted;
    }
    .queue-footer {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        dock: bottom;
    }
    .queue-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    .queue-source {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        display: none;
    }
    """
        + TRACK_FILTER_CSS
    )

    queue_length: reactive[int] = reactive(0)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._track_change_callback: Any = None

    def compose(self) -> ComposeResult:
        yield Vertical(id="queue-header", classes="queue-now-playing")
        yield Static("", id="queue-source", classes="queue-source")
        yield Label("Queue is empty.", id="queue-empty", classes="queue-empty")
        yield TrackTable(show_album=False, id="queue-table", queue_entry_keys=True)
        yield Static("", id="queue-footer", classes="queue-footer")
        yield Input(placeholder="/ Filter tracks...", id="track-filter", classes="track-filter")

    def on_mount(self) -> None:
        self._register_player_events()
        self._refresh_queue()
        # Land keyboard focus on the table (populated synchronously above) so
        # Tab / j / k have a starting point. Only here, not on every refresh —
        # reorder/select must not yank focus back.
        try:
            table = self.query_one("#queue-table", TrackTable)
            if table.display:
                table.focus()
        except Exception:
            logger.debug("Failed to focus queue table on mount", exc_info=True)

    def on_unmount(self) -> None:
        self._unregister_player_events()

    # ── Player event integration ──────────────────────────────────────

    def _register_player_events(self) -> None:
        player = self.app.player  # type: ignore[attr-defined]
        if player is None:
            # Startup stopped before the player existed; the page still shows.
            return
        self._track_change_callback = self._on_track_change
        player.on(PlayerEvent.TRACK_CHANGE, self._track_change_callback)

    def _unregister_player_events(self) -> None:
        try:
            player = self.app.player  # type: ignore[attr-defined]
            if self._track_change_callback:
                player.off(PlayerEvent.TRACK_CHANGE, self._track_change_callback)
        except Exception:
            logger.debug("Failed to unregister player events in queue page", exc_info=True)

    def _on_track_change(self, _track_info: dict) -> None:
        """Update the queue display when the track changes.

        Player events are dispatched onto the asyncio loop via
        ``call_soon_threadsafe`` (see services/player.py), so this
        callback already runs on the main thread — call directly.
        """
        try:
            self._update_current_track()
        except Exception:
            logger.debug("Failed to update queue page on track change", exc_info=True)

    def _update_current_track(self) -> None:
        """Lightweight update: refresh header and play indicator without rebuilding the table."""
        view = _now_playing_view(self.app.queue.current_track)  # type: ignore[attr-defined]
        self._mount_now_playing_header(view)

        # Update the playing indicator via TrackTable.
        try:
            table = self.query_one("#queue-table", TrackTable)
            table.set_playing(view["video_id"] if view is not None else None)
        except Exception:
            logger.debug("Failed to update playing indicator", exc_info=True)

    # ── Queue rendering ───────────────────────────────────────────────

    def _mount_now_playing_header(self, view: dict[str, Any] | None) -> None:
        """Render the "Now Playing" header from the (native-backed) view model."""
        header = self.query_one("#queue-header", Vertical)
        header.remove_children()
        if view is not None:
            header.mount(Label(view["title_label"], classes="queue-now-playing-title"))
            header.mount(Label(view["artist"], classes="queue-now-playing-artist"))
            header.display = True
        else:
            header.display = False

    def _refresh_queue(self) -> None:
        """Re-render the queue display from the QueueManager state.

        The table keeps its sort, filter, cursor and marks across the
        re-render: each row is keyed by its queue entry id, so a mark stays
        on the very occurrence it was put on even when the same track is
        queued twice, and goes away with that entry. An empty queue clears
        the table — marks included — before hiding it.
        """
        queue = self.app.queue  # type: ignore[attr-defined]
        entries = queue.entries
        tracks = [track for _, track in entries]
        view = _now_playing_view(queue.current_track)

        self._mount_now_playing_header(view)

        table = self.query_one("#queue-table", TrackTable)

        if not tracks:
            table.load_tracks([])
            table.display = False
            self.query_one("#queue-empty").display = True
        else:
            table.display = True
            self.query_one("#queue-empty").display = False
            table.refresh_tracks(tracks, keys=[entry_id for entry_id, _ in entries])

            # Set the playing indicator on the current track.
            table.set_playing(view["video_id"] if view is not None else None)

        self.queue_length = len(tracks)
        self._update_queue_source()
        self._update_footer()

    def _update_queue_source(self) -> None:
        """Update the seed source label from QueueManager state."""
        source = self.query_one("#queue-source", Static)
        view = _queue_source_view(
            self.app.queue.radio_seeds,  # type: ignore[attr-defined]
            get_settings().ui.show_queue_source,
        )
        if view is not None:
            text, tooltip = view
            source.update(text)
            source.tooltip = tooltip
            source.display = True
        else:
            source.update("")
            source.tooltip = None
            source.display = False

    def _update_footer(self) -> None:
        """Update the footer bar with track count info."""
        footer_text = _queue_footer_text(self.app.queue.length)  # type: ignore[attr-defined]

        try:
            footer = self.query_one("#queue-footer", Static)
            footer.update(footer_text)
        except Exception:
            logger.debug("Failed to update queue footer", exc_info=True)

    # ── Track selection ───────────────────────────────────────────────

    async def on_track_table_track_selected(self, event: TrackTable.TrackSelected) -> None:
        """Jump to and play the selected track."""
        event.stop()
        queue = self.app.queue  # type: ignore[attr-defined]
        track = queue.jump_to(event.index)
        if track:
            host = cast("YTMHostBase", self.app)
            await host.play_track(track)
            self._refresh_queue()

    # ── Action handling ───────────────────────────────────────────────

    async def handle_action(self, action: Action, count: int = 1) -> None:
        """Process vim-style navigation and queue management actions."""
        table = self.query_one("#queue-table", TrackTable)
        queue = self.app.queue  # type: ignore[attr-defined]

        match action:
            case Action.DELETE_ITEM:
                self._remove_selected(table, queue)

            case Action.REORDER_UP:
                self._move_track(table, queue, direction=-1, count=count)

            case Action.REORDER_DOWN:
                self._move_track(table, queue, direction=1, count=count)

            case Action.CYCLE_REPEAT:
                queue.cycle_repeat()
                self._update_footer()

            # NOTE: Action.TOGGLE_SHUFFLE is handled at the app level in
            # app/_keys.py and never delegated here, so no QueuePage case
            # for it.

            case _:
                await table.handle_action(action, count)

    def _remove_selected(self, table: TrackTable, queue: Any) -> None:
        """Remove the currently highlighted track from the queue.

        The queue-position decision — cursor_row is a visible-row index
        that no longer matches the queue position after a sort or filter,
        so the mapped original index must still name a live queue entry
        (the (native-backed) engine's refusal) — runs first; the
        post-refresh cursor re-landing (the same engine family) puts the
        cursor on the row that took the removed entry's place in the
        displayed order.
        """
        idx = _remove_target(table.selected_original_index, queue.length)
        if idx is None:
            return
        visible_row = table.cursor_row
        queue.remove(idx)
        self._refresh_queue()
        # The removed entry can't keep the cursor, so the row that took
        # its place in the displayed order gets it (the view — sort and
        # filter — survives the refresh).
        cursor_row = _post_remove_cursor_row(visible_row, table.row_count)
        if cursor_row is not None:
            table.move_cursor(row=cursor_row)

    def _move_track(self, table: TrackTable, queue: Any, direction: int, count: int = 1) -> None:
        """Move the highlighted track ``count`` queue positions up or down.

        The destination — clamped to the queue bounds, so a ``count``
        larger than the remaining distance simply moves the track to the
        top/bottom (e.g. ``15 J`` near the end lands it at the bottom) —
        comes from the (native-backed) engine as a queue position. The
        move is in queue positions: an active sort is cleared first, so
        the table shows queue order and the row is seen to move. A
        filter stays; hidden rows still count as positions, so a short
        move past one may leave the visible order unchanged. Marks stay,
        and the cursor stays on the moved entry (the refresh finds it
        again by its queue entry id).
        """
        from_idx = table.selected_original_index
        to_idx = _move_target(from_idx, direction, count, queue.length)
        if to_idx is None:
            return
        table.clear_sort()
        queue.move(from_idx, to_idx)
        self._refresh_queue()


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same view and mutation engines; these
# bindings replace the pure-Python transcriptions above when the extension
# is importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead native
# bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        queue_footer_text as _native_queue_footer_text,
    )
    from ytm_player.utils._ytm_player_native import (
        queue_move_target as _native_queue_move_target,
    )
    from ytm_player.utils._ytm_player_native import (
        queue_now_playing as _native_queue_now_playing,
    )
    from ytm_player.utils._ytm_player_native import (
        queue_post_remove_cursor as _native_queue_post_remove_cursor,
    )
    from ytm_player.utils._ytm_player_native import (
        queue_remove_target as _native_queue_remove_target,
    )
    from ytm_player.utils._ytm_player_native import (
        queue_source_view as _native_queue_source_view,
    )
except ImportError:
    pass
else:
    _now_playing_view = _native_queue_now_playing  # type: ignore[assignment]
    _queue_source_view = _native_queue_source_view  # type: ignore[assignment]
    _queue_footer_text = _native_queue_footer_text  # type: ignore[assignment]
    _remove_target = _native_queue_remove_target  # type: ignore[assignment]
    _post_remove_cursor_row = _native_queue_post_remove_cursor  # type: ignore[assignment]
    _move_target = _native_queue_move_target  # type: ignore[assignment]
