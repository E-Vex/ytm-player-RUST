"""Library page — content-only track view (sidebar moved to PlaylistSidebar).

The module's pure decision cores are dual-implemented: the Rust
extension (``_ytm_player_native``) implements them when it is built, and
the pure-Python transcriptions below run otherwise — the load view model
(``_playlist_view``), the Shuffle-lock button pair
(``_shuffle_lock_button``), the lock-toggle plan
(``_shuffle_lock_toggle_plan``), the playing-track id resolution
(``_playing_video_id``) and the cursor-restore row
(``_cursor_restore_row``). The bottom-of-module seam rebinds each to the
native function when the extension is importable. The workers, the
header mounts, the table loads and the popup pushes stay Python-side in
the page.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, Static

from ytm_player.config.keymap import Action
from ytm_player.ui.track_filter import TRACK_FILTER_CSS, TrackFilterHost
from ytm_player.ui.widgets.track_table import TrackTable
from ytm_player.utils.formatting import build_playlist_subtitle, normalize_tracks

if TYPE_CHECKING:
    from ytm_player.app._base import YTMHostBase

logger = logging.getLogger(__name__)


def _playlist_view(data: dict) -> dict[str, Any]:
    """The library page's load view model — the pure core of
    ``load_playlist``'s post-fetch walk.

    Everything the header and the table need from a ``get_playlist``
    answer, read in the reference's own order: the ``title`` (default
    ``"Unknown Playlist"``), the owner (``author.get("name", "Unknown")``
    for a dict ``author``, ``str(author)`` otherwise), the ``privacy``
    (the ``or ""`` gate then ``strip()``), the raw ``year``, the
    normalized track list with its count, and the total count
    (``data.get("trackCount") or track_count``, each read short-circuited
    by every earlier falsy one). The subtitle composes through
    ``build_playlist_subtitle`` and grows the ``" (loading N total…)"``
    suffix when the total exceeds the normalized count; the unavailable
    count is ``len(raw) - len(normalized)``; and the
    ``fetch_remaining``/``raw_count`` pair gates the background fetch of
    the playlist's tail (``total_count > len(raw)``).

    Implemented by the Rust extension
    (``_ytm_player_native.library_playlist_view``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    title = data.get("title", "Unknown Playlist")
    author = data.get("author", {})
    owner = author.get("name", "Unknown") if isinstance(author, dict) else str(author)
    privacy = (data.get("privacy") or "").strip()
    year = data.get("year")
    raw_tracks = data.get("tracks", [])
    tracks = normalize_tracks(raw_tracks)
    track_count = len(tracks)
    total_count = data.get("trackCount") or track_count

    subtitle = build_playlist_subtitle(owner, privacy, year, track_count)
    if total_count > track_count:
        subtitle += f" (loading {total_count} total\u2026)"

    return {
        "title": title,
        "owner": owner,
        "privacy": privacy,
        "year": year,
        "tracks": tracks,
        "track_count": track_count,
        "total_count": total_count,
        "subtitle": subtitle,
        "unavailable": len(raw_tracks) - track_count,
        "fetch_remaining": total_count > len(raw_tracks),
        "raw_count": len(raw_tracks),
    }


def _shuffle_lock_button(locked: Any) -> dict[str, str]:
    """The Shuffle-lock header button's label/classes pair.

    The ``bool`` of *locked* picks the pair — ``"Shuffle lock: ON"`` with
    the ``"locked"`` class when on, ``"Shuffle lock: off"`` with no
    class when off — the very pair all three button assemblies share
    (the load header, the edit-refresh header and the toggle's in-place
    update).

    Implemented by the Rust extension
    (``_ytm_player_native.shuffle_lock_button``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    on = bool(locked)
    return {
        "label": "Shuffle lock: ON" if on else "Shuffle lock: off",
        "classes": "locked" if on else "",
    }


def _shuffle_lock_toggle_plan(
    prefs_value: Any, current_context_id: Any, playlist_id: Any
) -> dict[str, Any]:
    """The decision core behind ``_toggle_shuffle_lock``.

    The new lock state is the flip of the stored pref's ``bool``, and
    the force-shuffle gate is the reference's own
    ``new_locked and current_context_id == playlist_id`` — a lock just
    turned on for the playlist the queue is currently playing forces
    the queue into shuffle (the caller's own toggle), while an unlock
    never touches the mode. The pref write, the button update, the
    shuffle toggle and the refreshes stay in the page.

    Implemented by the Rust extension
    (``_ytm_player_native.shuffle_lock_toggle_plan``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    new_locked = not bool(prefs_value)
    force_shuffle = bool(new_locked and current_context_id == playlist_id)
    return {"new_locked": new_locked, "force_shuffle": force_shuffle}


def _playing_video_id(player_track: Any, queue_track: Any) -> Any:
    """The video id of the track that is currently playing, or a falsy
    value naming none.

    The reference's own fallback chain: the player's current track read
    first — a dict contributes its ``video_id`` (a truthy non-dict rides
    home as itself) — and only when that answers nothing does the
    queue's current track get its ``video_id`` read.

    Implemented by the Rust extension
    (``_ytm_player_native.playing_video_id``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    playing_id = player_track
    if playing_id and isinstance(playing_id, dict):
        playing_id = playing_id.get("video_id")
    if not playing_id and queue_track:
        playing_id = queue_track.get("video_id")
    return playing_id


def _cursor_restore_row(row: Any, row_count: Any, playing_id: Any, tracks: Any) -> Any:
    """The row the library table's cursor lands on after a load.

    The saved nav row wins when it is present and within the table's
    bounds (``0 <= row < row_count``); otherwise the first track whose
    ``video_id`` equals the currently-playing id names the row; neither
    resolving answers ``None`` (the cursor stays put).

    Implemented by the Rust extension
    (``_ytm_player_native.cursor_restore_row``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if row_count == 0:
        return None
    if row is not None and 0 <= row < row_count:
        return row
    if playing_id:
        for i, track in enumerate(tracks):
            if track.get("video_id") == playing_id:
                return i
    return None


class LibraryPage(TrackFilterHost, Widget):
    """Library content area — displays tracks for the selected playlist.

    The playlist sidebar has been extracted to PlaylistSidebar (persistent).
    This page receives a ``playlist_id`` kwarg from navigation and loads
    the corresponding tracks inline.
    """

    _filter_table_id = "#library-tracks"

    DEFAULT_CSS = (
        """
    LibraryPage {
        height: 1fr;
        width: 1fr;
    }

    #content-header {
        height: auto;
        max-height: 6;
        padding: 1 2;
    }

    .content-title-row {
        height: auto;
        width: 1fr;
    }
    .content-title-row Label {
        width: auto;
    }
    .content-title {
        text-style: bold;
    }

    .content-description {
        color: $text-muted;
        text-style: italic;
    }

    .content-subtitle {
        color: $text-muted;
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

    .content-title-spacer {
        width: 1fr;
        height: 1;
    }

    #shuffle-lock-btn {
        width: auto;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #shuffle-lock-btn.locked {
        color: $primary;
        text-style: bold;
    }
    #shuffle-lock-btn:hover {
        background: $accent 30%;
    }

    #empty-state {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }

    #loading-state {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }

    #library-tracks {
        height: 1fr;
    }
    """
        + TRACK_FILTER_CSS
    )

    is_loading: reactive[bool] = reactive(True)

    # Group of a playlist load and the tail fetch it starts: a new load
    # supersedes both, and nothing else on this page cancels them.
    LOAD_GROUP: ClassVar[str] = "library-load"

    def __init__(
        self,
        *,
        playlist_id: str | None = None,
        cursor_row: int | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._active_playlist_id: str | None = playlist_id
        self._restore_cursor_row: int | None = cursor_row
        self._cached_owner: str = "Unknown"
        self._cached_year: int | str | None = None
        self._cached_privacy: str = ""

    def compose(self) -> ComposeResult:
        yield Vertical(id="content-header")
        yield Static("Select a playlist from the sidebar", id="empty-state")
        yield Static("Loading...", id="loading-state")
        yield TrackTable(show_album=True, id="library-tracks")
        yield Input(placeholder="/ Filter tracks...", id="track-filter", classes="track-filter")

    def on_mount(self) -> None:
        self.query_one("#loading-state").display = False
        self.query_one("#library-tracks").display = False
        self.query_one("#content-header").display = False

        # Auto-load if a playlist_id was provided.
        if self._active_playlist_id:
            self.reload(self._active_playlist_id)
        elif not self._active_playlist_id:
            # Try the currently-playing playlist from the app.
            target_id = getattr(self.app, "_active_library_playlist_id", None)
            if target_id:
                self._active_playlist_id = target_id
                self.reload(target_id)

    def get_nav_state(self) -> dict[str, Any]:
        """Return state to preserve when navigating away."""
        state: dict[str, Any] = {}
        if self._active_playlist_id:
            state["playlist_id"] = self._active_playlist_id
        try:
            table = self.query_one("#library-tracks", TrackTable)
            if table.display and table.cursor_row is not None:
                state["cursor_row"] = table.cursor_row
        except Exception:
            pass
        return state

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    # First batch size for progressive playlist loading.
    _FIRST_BATCH = 300

    def reload(self, playlist_id: str) -> None:
        """Load *playlist_id* as this page's worker.

        Supersedes the load in flight and its tail fetch, so the tracks a
        previous load was still fetching are never appended to this one.
        """
        self.run_worker(
            self.load_playlist(playlist_id),
            name="load-playlist",
            group=self.LOAD_GROUP,
            exclusive=True,
        )

    async def load_playlist(self, playlist_id: str) -> None:
        """Fetch and display a playlist's tracks.

        The load view model — the title/owner/privacy/year reads, the
        normalized track list, the subtitle with its loading suffix, the
        unavailable count and the tail-fetch gate — resolves through the
        ``_playlist_view`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise); the header
        mounts, the table load and the workers stay here.
        """
        self._active_playlist_id = playlist_id

        # Show loading state.
        self.query_one("#empty-state").display = False
        self.query_one("#library-tracks").display = False
        self.query_one("#content-header").display = False
        loading = self.query_one("#loading-state")
        loading.display = True

        try:
            ytmusic = cast("YTMHostBase", self.app).ytmusic
            assert ytmusic is not None
            data = await ytmusic.get_playlist(
                playlist_id, limit=self._FIRST_BATCH, order="recently_added"
            )

            # If user selected a different playlist while we were loading, discard.
            if self._active_playlist_id != playlist_id:
                return

            view = _playlist_view(data)

            # Cache metadata for use in refresh_header / _fetch_remaining without extra API calls.
            self._cached_owner = view["owner"]
            self._cached_year = view["year"]
            self._cached_privacy = view["privacy"]

            # Store data for radio button — ensure playlistId is set since
            # get_playlist() returns 'id' but _start_playlist_radio expects 'playlistId'.
            self._playlist_data = data
            self._playlist_data.setdefault("playlistId", playlist_id)
            # Update header.
            header = self.query_one("#content-header", Vertical)
            await header.remove_children()
            header.display = True
            title_row = Horizontal(classes="content-title-row")
            await header.mount(title_row)
            await title_row.mount(Label(view["title"], classes="content-title"))
            await title_row.mount(Static("[▶ Start Radio]", id="start-radio-btn", markup=True))
            await title_row.mount(Static(classes="content-title-spacer"))
            host = cast("YTMHostBase", self.app)
            lock = _shuffle_lock_button(host.shuffle_prefs.get(playlist_id))
            await title_row.mount(
                Static(lock["label"], id="shuffle-lock-btn", classes=lock["classes"])
            )
            description = (data.get("description") or "").strip()
            if description:
                await header.mount(Label(description, classes="content-description"))
            self._subtitle_label = Label(view["subtitle"], classes="content-subtitle")
            await header.mount(self._subtitle_label)
            unavailable = view["unavailable"]
            if unavailable:
                await header.mount(
                    Label(f"{unavailable} unavailable tracks hidden", classes="content-subtitle")
                )

            # Load tracks into the table.
            loading.display = False
            table = self.query_one("#library-tracks", TrackTable)
            table.display = True
            table.load_tracks(view["tracks"])

            # Restore cursor position or scroll to the currently-playing track.
            self._restore_track_cursor(table)

            # Land keyboard focus on the table so Tab / j / k have a starting
            # point once the page's data has rendered.
            table.focus()

            # Kick off background fetch for remaining tracks.
            if view["fetch_remaining"]:
                self.run_worker(
                    self._fetch_remaining(playlist_id, view["raw_count"]),
                    name="fetch-remaining",
                    group=self.LOAD_GROUP,
                )

        except Exception:
            logger.exception("Failed to load playlist %s", playlist_id)
            loading.display = False
            self.query_one("#empty-state").display = True
            empty = self.query_one("#empty-state", Static)
            empty.update("Failed to load playlist")

    async def refresh_header(self, title: str, description: str, privacy: str) -> None:
        """Update the playlist header in-place after an edit, without reloading tracks.

        The Shuffle-lock button's label/classes pair resolves through
        the ``_shuffle_lock_button`` engine (native when the Rust
        extension is built, the pure-Python transcription otherwise);
        the subtitle keeps composing through the shared
        ``build_playlist_subtitle``.
        """
        try:
            header = self.query_one("#content-header", Vertical)
            if not header.display:
                return
            table = self.query_one("#library-tracks", TrackTable)
            track_count = len(table.tracks)
            await header.remove_children()
            title_row = Horizontal(classes="content-title-row")
            await header.mount(title_row)
            await title_row.mount(Label(title, classes="content-title"))
            await title_row.mount(Static("[▶ Start Radio]", id="start-radio-btn", markup=True))
            await title_row.mount(Static(classes="content-title-spacer"))
            host = cast("YTMHostBase", self.app)
            playlist_id = self._active_playlist_id or ""
            lock = _shuffle_lock_button(
                host.shuffle_prefs.get(playlist_id) if playlist_id else False
            )
            await title_row.mount(
                Static(lock["label"], id="shuffle-lock-btn", classes=lock["classes"])
            )
            if description:
                await header.mount(Label(description, classes="content-description"))
            self._cached_privacy = privacy
            subtitle = build_playlist_subtitle(
                self._cached_owner, privacy, self._cached_year, track_count
            )
            self._subtitle_label = Label(subtitle, classes="content-subtitle")
            await header.mount(self._subtitle_label)
        except Exception:
            logger.exception("Failed to refresh library header after edit")

    def update_track_count(self, delta: int = 0) -> None:
        """Update the subtitle label to reflect the current table track count.

        *delta* is added to the count — useful when the table hasn't been
        updated yet (e.g. after adding tracks via the API).
        """
        try:
            table = self.query_one("#library-tracks", TrackTable)
            track_count = len(table.tracks) + delta
            subtitle = build_playlist_subtitle(
                self._cached_owner, self._cached_privacy, self._cached_year, track_count
            )
            if hasattr(self, "_subtitle_label") and self._subtitle_label:
                self._subtitle_label.update(subtitle)
        except Exception:
            logger.exception("Failed to update library track count label")

    async def _fetch_remaining(self, playlist_id: str, already_have: int) -> None:
        """Background fetch for tracks beyond the first batch."""
        ytmusic = cast("YTMHostBase", self.app).ytmusic
        assert ytmusic is not None
        remaining = await ytmusic.get_playlist_remaining(
            playlist_id, already_have, order="recently_added"
        )
        # Discard if user switched playlists while we were fetching.
        if self._active_playlist_id != playlist_id:
            return
        if not remaining:
            return
        tracks = normalize_tracks(remaining)
        try:
            table = self.query_one("#library-tracks", TrackTable)
            table.append_tracks(tracks)
            self.update_track_count()
        except Exception:
            logger.debug("Failed to append remaining tracks in library", exc_info=True)

    def _restore_track_cursor(self, table: TrackTable) -> None:
        """Move cursor to the saved row, or to the currently-playing track.

        The landing row resolves through the ``_cursor_restore_row``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise), with the currently-playing
        id resolved through ``_playing_video_id``; the cursor move stays
        here.
        """
        if table.row_count == 0:
            return

        row = self._restore_cursor_row
        self._restore_cursor_row = None

        player = cast("YTMHostBase", self.app).player
        player_track = getattr(player, "_current_track", None) if player is not None else None
        queue = getattr(self.app, "queue", None)
        queue_track = queue.current_track if queue else None
        playing_id = _playing_video_id(player_track, queue_track)

        target = _cursor_restore_row(row, table.row_count, playing_id, table._tracks)
        if target is not None:
            table.move_cursor(row=target)

    # ------------------------------------------------------------------
    # Header button clicks
    # ------------------------------------------------------------------

    def on_click(self, event: Click) -> None:
        """Handle clicks on header action buttons."""
        widget = event.widget
        if widget is None:
            return
        if widget.id == "start-radio-btn":
            event.stop()
            data = getattr(self, "_playlist_data", None)
            if data:
                self.run_worker(
                    cast("YTMHostBase", self.app)._start_playlist_radio(data),
                    name="start_radio",
                    group="start-radio",
                    exclusive=True,
                )
        elif widget.id == "shuffle-lock-btn":
            event.stop()
            self._toggle_shuffle_lock()

    def _toggle_shuffle_lock(self) -> None:
        """Flip the per-playlist Shuffle lock and update the button label.

        The toggle plan — the new lock state and the force-shuffle gate
        — resolves through the ``_shuffle_lock_toggle_plan`` engine
        (native when the Rust extension is built, the pure-Python
        transcription otherwise), the button's new label/classes pair
        through ``_shuffle_lock_button``; the pref write, the button
        update, the shuffle toggle and the refreshes stay here.
        """
        playlist_id = self._active_playlist_id
        if not playlist_id:
            return
        host = cast("YTMHostBase", self.app)
        plan = _shuffle_lock_toggle_plan(
            host.shuffle_prefs.get(playlist_id), host.queue.current_context_id, playlist_id
        )
        new_locked = plan["new_locked"]
        host.shuffle_prefs.set(playlist_id, new_locked)
        # Update button visuals.
        try:
            btn = self.query_one("#shuffle-lock-btn", Static)
            lock = _shuffle_lock_button(new_locked)
            btn.update(lock["label"])
            if lock["classes"]:
                btn.add_class("locked")
            else:
                btn.remove_class("locked")
        except Exception:
            logger.debug("Failed to update shuffle-lock-btn label", exc_info=True)
        # If we just locked AND the queue is currently this playlist, force
        # shuffle on; if we unlocked, leave shuffle as-is (lock is one-way
        # enforcement on entry, not a toggle of current state).
        if plan["force_shuffle"]:
            if not host.queue.shuffle_enabled:
                host.queue.toggle_shuffle()
                host._refresh_queue_page()
                try:
                    bar = host.query_one("#playback-bar")
                    bar.update_shuffle(host.queue.shuffle_enabled)  # type: ignore[attr-defined]
                except Exception:
                    pass
        # Refresh the playback-bar shuffle button enabled/disabled state.
        try:
            bar = host.query_one("#playback-bar")
            bar.refresh_shuffle_lock_state()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Failed to refresh playback-bar lock state", exc_info=True)

    # ------------------------------------------------------------------
    # Track selection → play
    # ------------------------------------------------------------------

    async def on_track_table_track_selected(self, event: TrackTable.TrackSelected) -> None:
        """Queue all tracks and start playback from the selected one."""
        event.stop()
        table = self.query_one("#library-tracks", TrackTable)
        host = cast("YTMHostBase", self.app)
        if self._active_playlist_id:
            host._active_library_playlist_id = self._active_playlist_id
        await host._replace_queue_and_play(
            table.tracks,
            entity_id=self._active_playlist_id,
            start_index=event.index,
            autoplay=False,
        )
        await host.play_track(event.track)

    # ------------------------------------------------------------------
    # Vim-style action handler
    # ------------------------------------------------------------------

    async def handle_action(self, action: Action, count: int = 1) -> None:
        """Process vim-style navigation actions dispatched from the app."""
        try:
            table = self.query_one("#library-tracks", TrackTable)
            if table.display:
                await table.handle_action(action, count)
        except Exception:
            logger.debug("Failed to dispatch action to library track table", exc_info=True)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same library-page decision cores;
# these bindings replace the pure-Python transcriptions above when the
# extension is importable.  The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        cursor_restore_row as _native_cursor_restore_row,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        library_playlist_view as _native_library_playlist_view,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        playing_video_id as _native_playing_video_id,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        shuffle_lock_button as _native_shuffle_lock_button,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        shuffle_lock_toggle_plan as _native_shuffle_lock_toggle_plan,
    )
except ImportError:
    pass
else:
    _playlist_view = _native_library_playlist_view  # type: ignore[assignment]
    _shuffle_lock_button = _native_shuffle_lock_button  # type: ignore[assignment]
    _shuffle_lock_toggle_plan = (  # type: ignore[assignment]
        _native_shuffle_lock_toggle_plan
    )
    _playing_video_id = _native_playing_video_id  # type: ignore[assignment]
    _cursor_restore_row = _native_cursor_restore_row  # type: ignore[assignment]
