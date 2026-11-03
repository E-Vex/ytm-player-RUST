"""Playlist picker popup for adding tracks to a playlist."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Label, ListItem, ListView, Static
from textual.worker import Worker, WorkerState

from ytm_player.config.paths import RECENT_PLAYLISTS_FILE
from ytm_player.ui.popups.base import BasePopup
from ytm_player.ui.popups.confirm_popup import ConfirmPopup
from ytm_player.ui.popups.create_playlist_popup import CreatePlaylistPopup

if TYPE_CHECKING:
    from ytm_player.app._base import YTMHostBase

logger = logging.getLogger(__name__)
MAX_RECENT = 20


def _load_recent_ids() -> list[str]:
    """Load recently-used playlist IDs from disk."""
    try:
        if RECENT_PLAYLISTS_FILE.exists():
            data = json.loads(RECENT_PLAYLISTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data[:MAX_RECENT]
    except Exception:
        logger.debug("Could not load recent playlists", exc_info=True)
    return []


def _save_recent_ids(ids: list[str]) -> None:
    """Persist recently-used playlist IDs to disk."""
    try:
        RECENT_PLAYLISTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECENT_PLAYLISTS_FILE.write_text(json.dumps(ids[:MAX_RECENT]), encoding="utf-8")
    except Exception:
        logger.debug("Could not save recent playlists", exc_info=True)


def _record_recent_ids(recent: list[str], playlist_id: str) -> list[str]:
    """The move-to-front update — the ``_record_recent`` engine.

    The pure half of the recently-used bookkeeping: the first equal
    occurrence of *playlist_id* is removed (``list.remove`` drops exactly
    one) and the id is inserted at the front. The mutated list is
    returned; the caller persists it (with the ``MAX_RECENT`` cap applied
    at save time).

    Implemented by the Rust extension
    (``_ytm_player_native.playlist_record_recent``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if playlist_id in recent:
        recent.remove(playlist_id)
    recent.insert(0, playlist_id)
    return recent


def _record_recent(playlist_id: str) -> None:
    """Move *playlist_id* to the front of the recent list and persist."""
    recent = _load_recent_ids()
    _save_recent_ids(_record_recent_ids(recent, playlist_id))


def _sorted_by_recent(
    playlists: list[dict[str, Any]], recent_ids: list[str]
) -> list[dict[str, Any]]:
    """The recency sort — the ``_sort_by_recent`` engine.

    The pure half of the picker's list ordering: no recent ids means no
    sort at all (the playlist order is whatever the fetch produced);
    otherwise the key is ``(0, recent_order[pid])`` for a playlist whose
    id is in the recent list and ``(1, 0)`` for one that is not, and the
    sort is stable. ``recent_order`` is the dict comprehension, so a
    duplicated recent id counts at its last position.

    Implemented by the Rust extension
    (``_ytm_player_native.playlist_picker_sort``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if not recent_ids:
        return playlists

    recent_set = set(recent_ids)
    recent_order = {pid: i for i, pid in enumerate(recent_ids)}

    def sort_key(pl: dict) -> tuple[int, int]:
        pid = pl.get("playlistId", pl.get("id", ""))
        if pid in recent_set:
            return (0, recent_order[pid])
        return (1, 0)

    return sorted(playlists, key=sort_key)


def _visible_picker_rows(
    playlists: list[dict[str, Any]], filter_text: str
) -> list[tuple[Any, Any, Any]]:
    """The filtered rows — the ``_populate_list`` row engine.

    The pure half of the picker's list rebuild: the query is
    ``filter_text.strip().lower()`` (stripped first, then lowercased); an
    empty query shows every playlist, a non-empty one only those whose
    title lowercased contains it; and each surviving playlist contributes
    the row ``(playlist_id, title, count)`` — the id through the
    ``playlistId``-then-``id``-then-``""`` read, the title with its
    "Untitled" default, and the count as ``""`` when absent or falsy,
    ``str(count)`` when truthy.

    Implemented by the Rust extension
    (``_ytm_player_native.playlist_picker_rows``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    query = filter_text.strip().lower()

    rows: list[tuple[Any, Any, Any]] = []
    for pl in playlists:
        title = pl.get("title", "Untitled")
        if query and query not in title.lower():
            continue
        playlist_id = pl.get("playlistId", pl.get("id", ""))
        track_count = pl.get("count", "")
        if track_count:
            track_count = str(track_count)
        rows.append((playlist_id, title, track_count))
    return rows


def _count_label(count: int) -> str:
    """The playlist-count status label — the ``_populate_list`` label engine.

    The pure half of the picker's status line: ``f"{count} playlist{'s' if
    count != 1 else ''}"``.

    Implemented by the Rust extension
    (``_ytm_player_native.playlist_picker_count_label``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return f"{count} playlist{'s' if count != 1 else ''}"


class _PlaylistItem(ListItem):
    """A single playlist entry in the picker list."""

    def __init__(self, playlist_id: str, title: str, count: str = "") -> None:
        super().__init__()
        self.playlist_id = playlist_id
        self._title = title
        self._count = count

    def compose(self) -> ComposeResult:
        text = self._title
        if self._count:
            text = f"{self._title}  ({self._count})"
        yield Label(text, markup=False)


class _CreateNewItem(ListItem):
    """Sentinel entry for creating a new playlist."""

    def __init__(self) -> None:
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Label("[+] Create New Playlist")


class PlaylistPicker(BasePopup[str | None]):
    """Select a playlist to add track(s) to.

    Returns the playlist ID on success, or ``None`` if cancelled.
    """

    DEFAULT_CSS = """
    PlaylistPicker > Vertical {
        width: 50;
        max-height: 80%;
    }

    PlaylistPicker #picker-title {
        text-align: center;
        text-style: bold;
        width: 100%;
        color: $text;
    }

    PlaylistPicker #picker-status {
        text-align: center;
        width: 100%;
        color: $text-muted;
        margin-bottom: 1;
    }

    PlaylistPicker #filter-input {
        margin-bottom: 1;
    }

    PlaylistPicker ListView {
        height: auto;
        max-height: 18;
        background: $surface;
    }

    PlaylistPicker ListItem {
        padding: 0 1;
        height: 1;
    }

    PlaylistPicker ListItem:hover {
        background: $accent 30%;
    }

    PlaylistPicker ListView:focus > ListItem.--highlight {
        background: $primary 40%;
    }

    """

    def __init__(self, video_ids: list[str], tracks: list[dict[str, Any]] | None = None) -> None:
        super().__init__()
        self.video_ids = video_ids
        self.tracks = tracks or []
        self._playlists: list[dict[str, Any]] = []
        # One flow owns the picker at a time, from the gesture that starts
        # it (Enter on a playlist, or on Create New) until it ends in a
        # failure the user can retry or in the picker closing. Gestures that
        # arrive while a flow owns it are ignored — including ones already
        # queued behind the first. After an outcome that could not be
        # confirmed (a timeout) the flow keeps ownership for good: the
        # request may still be applied, so this picker takes no other.
        self._submitting = False

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Add to Playlist", id="picker-title")
            yield Static("Loading playlists...", id="picker-status")
            yield Input(placeholder="Filter playlists...", id="filter-input")
            yield ListView(id="playlist-list")

    def on_mount(self) -> None:
        self.query_one("#filter-input", Input).display = False
        self._load_playlists()

    # ── Data loading ────────────────────────────────────────────────

    def _load_playlists(self) -> None:
        """Kick off an async worker to fetch user playlists."""
        self.run_worker(self._fetch_playlists(), name="fetch_playlists")

    async def _fetch_playlists(self) -> list[dict[str, Any]]:
        try:
            ytmusic = cast("YTMHostBase", self.app).ytmusic
            assert ytmusic is not None
            playlists = await ytmusic.get_library_playlists(limit=None)
            return playlists
        except Exception:
            logger.exception("Failed to fetch library playlists")
            return []

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "fetch_playlists":
            return

        if event.state == WorkerState.SUCCESS:
            self._playlists = event.worker.result or []
            self._sort_by_recent()
            self._populate_list()
        elif event.state == WorkerState.ERROR:
            status = self.query_one("#picker-status", Static)
            status.update("Failed to load playlists")

    def _sort_by_recent(self) -> None:
        """Sort playlists so recently-used ones appear first."""
        recent_ids = _load_recent_ids()
        self._playlists = _sorted_by_recent(self._playlists, recent_ids)

    def _populate_list(self, filter_text: str = "", *, focus: bool = True) -> None:
        """Rebuild the ListView with current playlists, optionally filtered.

        *focus* moves keyboard focus to the list (the initial load); a
        rebuild while the user types in the filter leaves focus alone.
        While a flow owns the picker the status line is its (progress, or
        the guidance after an unconfirmed outcome) and a rebuild leaves
        that alone too.
        """
        status = self.query_one("#picker-status", Static)
        list_view = self.query_one("#playlist-list", ListView)
        filter_input = self.query_one("#filter-input", Input)

        list_view.clear()

        # Always show "Create New" at the top.
        list_view.append(_CreateNewItem())

        if not self._playlists:
            status.update("No playlists found")
            filter_input.display = False
        else:
            if not self._submitting:
                status.update(_count_label(len(self._playlists)))
            filter_input.display = True

        for playlist_id, title, track_count in _visible_picker_rows(self._playlists, filter_text):
            list_view.append(_PlaylistItem(playlist_id, title, track_count))

        if focus:
            list_view.focus()

    # ── Filtering ───────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._populate_list(event.value, focus=False)

    # ── Selection ───────────────────────────────────────────────────

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        event.stop()
        if self._submitting:
            return
        item = event.item

        if isinstance(item, _CreateNewItem):
            # Owned from here: a second Create selection queued behind this
            # one must not open a second dialog.
            self._submitting = True
            self.app.push_screen(CreatePlaylistPopup(), self._on_create_result)
            return

        if isinstance(item, _PlaylistItem):
            self._add_to_playlist(item.playlist_id, item._title)

    def _release(self) -> None:
        """The flow ended in a failure the user can retry: the picker is free again."""
        self._submitting = False

    def _report_unconfirmed(self, status: Static, prefix: str, check: str) -> None:
        """A timeout: keep ownership, tell the user what to do.

        No confirmation came in time. Whether the request reached the
        server is unknown, and the thread behind it is not cancelled, so
        the change may still be applied; this picker accepts no further
        submission. *check* names what the user should reload and look at.
        """
        from ytm_player.services.ytmusic import mutation_failure_suffix

        self.notify(
            f"{prefix} — {mutation_failure_suffix('timeout')}", severity="warning", timeout=8
        )
        status.update(f"Unconfirmed — close this, reload {check} and check before retrying")

    def _duplicate_question(self, name: str) -> str:
        """The confirmation text for a ``"duplicate"`` answer from the server.

        With more than one track the server rejects the whole request when
        any of them is already there — the selection itself may hold two
        copies of a song — so the copy names the selection, not one track.
        """
        count = len(self.video_ids)
        if count == 1:
            return f"This track is already in '{name}'.\nAdd anyway?"
        return f"Duplicate tracks detected. Add all {count} selected tracks to '{name}' anyway?"

    def _on_create_result(self, result: tuple[str, str, str] | None) -> None:
        """Handle the result from CreatePlaylistPopup."""
        if result is None:
            self._release()
            return
        name, description, privacy = result
        self.run_worker(
            self._create_and_add(name, description, privacy),
            name="create_playlist",
        )

    async def _create_and_add(
        self, name: str, description: str = "", privacy: str = "PRIVATE"
    ) -> None:
        """Create a new playlist, then add the tracks to it."""
        status = self.query_one("#picker-status", Static)
        status.update(f"Creating '{name}'...")

        try:
            ytmusic = cast("YTMHostBase", self.app).ytmusic
            assert ytmusic is not None
            from ytm_player.services.ytmusic import mutation_failure_suffix

            create_status, playlist_id = await ytmusic.create_playlist(
                name, description=description, privacy=privacy
            )
            if create_status == "timeout":
                self._report_unconfirmed(
                    status, f"Couldn't confirm creating '{name}'", "your playlist list"
                )
                return
            if create_status != "success" or not playlist_id:
                self.notify(
                    f"Failed to create playlist — {mutation_failure_suffix(create_status)}",
                    severity="error",
                )
                status.update("Creation failed")
                self._release()
                return

            status.update(f"Adding tracks to '{name}'...")
            await self._add_to_created(playlist_id, name)

        except Exception:
            logger.exception("Failed to create playlist and add tracks")
            self.notify("Failed to create playlist", severity="error")
            status.update("Error")
            self._release()

    async def _add_to_created(self, playlist_id: str, name: str, duplicates: bool = False) -> None:
        """Add the tracks to the playlist just created, then finish.

        A ``"duplicate"`` answer asks the same question as for an existing
        playlist; accepting retries against this same *playlist_id* (never
        a second playlist), declining reports the playlist as created with
        nothing added and closes the picker with ``None``, which keeps the
        caller's marks. A timeout keeps the picker owned (the add may have
        landed). Any other failure leaves the picker open with the marks
        as well.
        """
        status = self.query_one("#picker-status", Static)
        try:
            ytmusic = cast("YTMHostBase", self.app).ytmusic
            assert ytmusic is not None
            from ytm_player.services.ytmusic import mutation_failure_suffix

            result = await ytmusic.add_playlist_items(
                playlist_id, self.video_ids, duplicates=duplicates
            )
            if result == "duplicate":
                status.update("Already in playlist")

                def _on_duplicate_confirm(confirmed: bool | None) -> None:
                    # Still owned either way: the retry and the empty-playlist
                    # report both belong to this flow.
                    if confirmed:
                        self.run_worker(
                            self._add_to_created(playlist_id, name, duplicates=True),
                            name="create_playlist",
                        )
                    else:
                        self.run_worker(
                            self._finish_created_empty(playlist_id, name), name="create_playlist"
                        )

                self.app.push_screen(
                    ConfirmPopup(
                        self._duplicate_question(name),
                        confirm_label="Add anyway",
                        cancel_label="Cancel",
                    ),
                    _on_duplicate_confirm,
                )
                return
            if result == "timeout":
                self._report_unconfirmed(
                    status, f"Created '{name}' but couldn't confirm the add", f"'{name}'"
                )
                return
            if result != "success":
                self.notify(
                    f"Created '{name}' but couldn't add tracks — {mutation_failure_suffix(result)}",
                    severity="warning",
                )
                status.update("Add failed")
                self._release()
                return

            await self._finish_created(playlist_id, name)

        except Exception:
            logger.exception("Failed to add tracks to the created playlist %r", playlist_id)
            self.notify(f"Created '{name}' but couldn't add tracks", severity="error")
            status.update("Error")
            self._release()

    async def _refresh_sidebar(self) -> None:
        """Full sidebar refresh: a new playlist isn't in the cached items yet."""
        try:
            from ytm_player.ui.sidebars.playlist_sidebar import PlaylistSidebar

            ps = self.app.query_one("#playlist-sidebar", PlaylistSidebar)
            await ps.refresh_playlists()
        except Exception:
            logger.exception("Sidebar refresh failed after create")

    async def _finish_created_empty(self, playlist_id: str, name: str) -> None:
        """Declined the duplicate prompt after creating: the playlist exists, empty.

        Say so and show it in the sidebar. Nothing was added, so it is not
        recorded as a recent target and the picker closes with ``None`` (the
        caller keeps its marks).
        """
        await self._refresh_sidebar()
        self.notify(f"Created '{name}' with nothing added", severity="information")
        self.dismiss(None)

    async def _finish_created(self, playlist_id: str, name: str) -> None:
        """The completion path after tracks landed in a newly created playlist."""
        try:
            _record_recent(playlist_id)
            await self._refresh_sidebar()

            # Update library page header if the target playlist is currently open.
            try:
                from ytm_player.ui.pages.library import LibraryPage
                from ytm_player.utils.formatting import strip_vl_prefix

                host = cast("YTMHostBase", self.app)
                current_pid = host._current_page_kwargs.get("playlist_id", "")
                if host._current_page == "library" and strip_vl_prefix(
                    current_pid
                ) == strip_vl_prefix(playlist_id):
                    library = self.app.query_one(LibraryPage)
                    library.update_track_count(+len(self.video_ids))
            except Exception:
                logger.exception("Library track count update failed")

            track_word = "track" if len(self.video_ids) == 1 else "tracks"
            self.notify(
                f"Added {len(self.video_ids)} {track_word} to '{name}'",
                severity="information",
            )
            self.dismiss(playlist_id)

        except Exception:
            logger.exception("Failed to finish adding tracks to the created playlist")
            self.notify(f"Created '{name}' but couldn't finish adding tracks", severity="error")
            self.query_one("#picker-status", Static).update("Error")

    def _tracks_for_append(self, set_video_ids: dict[str, str]) -> list[dict[str, Any]]:
        """Build the track dicts to append to the open playlist's table.

        ``self.tracks`` are usually already-normalized dicts (the playing or
        focused track). Re-running ``normalize_tracks()`` on those would drop
        ``thumbnail_url`` — it reads the raw ``thumbnails`` key that normalized
        dicts no longer carry — so only raw dicts are normalized here.

        *set_video_ids* maps videoId -> server-assigned setVideoId from the add
        response; appended rows are stamped with it so "Remove from Playlist"
        works immediately. A ``setVideoId`` on the source dict names the row
        the track was taken from (a row of this or another playlist), never
        the row just appended, so it is dropped. Rows without a setVideoId
        from the response are flagged so the remove action can explain a
        reload is needed.
        """
        from ytm_player.utils.formatting import normalize_tracks

        out: list[dict[str, Any]] = []
        for src in self.tracks:
            if "video_id" in src:
                track = dict(src)
            else:
                normed = normalize_tracks([src])
                if not normed:
                    continue
                track = normed[0]
            track.pop("setVideoId", None)
            svid = set_video_ids.get(track.get("video_id", ""))
            if svid:
                track["setVideoId"] = svid
            else:
                track["_needs_reload_for_removal"] = True
            out.append(track)
        return out

    def _add_to_playlist(self, playlist_id: str, title: str, duplicates: bool = False) -> None:
        """Start the add flow for an existing playlist, unless one owns the picker."""
        if self._submitting:
            return
        self._submitting = True
        self.run_worker(
            self._do_add(playlist_id, title, duplicates=duplicates),
            name="add_to_playlist",
        )

    async def _do_add(self, playlist_id: str, title: str, duplicates: bool = False) -> None:
        status = self.query_one("#picker-status", Static)
        status.update(f"Adding to '{title}'...")

        try:
            from ytm_player.services.ytmusic import mutation_failure_suffix

            ytmusic = cast("YTMHostBase", self.app).ytmusic
            assert ytmusic is not None
            result = await ytmusic.add_playlist_items(
                playlist_id, self.video_ids, duplicates=duplicates
            )
            if result == "duplicate":
                status.update("Already in playlist")

                def _on_duplicate_confirm(confirmed: bool | None) -> None:
                    if confirmed:
                        # The retry belongs to the flow that owns the picker.
                        self.run_worker(
                            self._do_add(playlist_id, title, duplicates=True),
                            name="add_to_playlist",
                        )
                    else:
                        self.dismiss(None)

                self.app.push_screen(
                    ConfirmPopup(
                        self._duplicate_question(title),
                        confirm_label="Add anyway",
                        cancel_label="Cancel",
                    ),
                    _on_duplicate_confirm,
                )
                return
            if result == "timeout":
                self._report_unconfirmed(
                    status, f"Couldn't confirm adding to '{title}'", "the playlist"
                )
                return
            if result != "success":
                self.notify(
                    f"Couldn't add to '{title}' — {mutation_failure_suffix(result)}",
                    severity="error",
                )
                status.update("Error")
                self._release()
                return

            _record_recent(playlist_id)

            # Optimistic sidebar count update — before next library reload.
            try:
                from ytm_player.ui.sidebars.playlist_sidebar import LibraryPanel

                panel = self.app.query_one("#ps-playlists", LibraryPanel)
                panel.update_item_count(playlist_id, +len(self.video_ids))
            except Exception:
                logger.exception("Sidebar count update failed")

            # Append tracks to the table and update header if playlist is open.
            try:
                from ytm_player.ui.pages.library import LibraryPage
                from ytm_player.ui.widgets.track_table import TrackTable
                from ytm_player.utils.formatting import strip_vl_prefix

                host = cast("YTMHostBase", self.app)
                current_pid = host._current_page_kwargs.get("playlist_id", "")
                if host._current_page == "library" and strip_vl_prefix(
                    current_pid
                ) == strip_vl_prefix(playlist_id):
                    library = self.app.query_one(LibraryPage)
                    if len(set(self.video_ids)) != len(self.video_ids):
                        # Two copies of one video were added. The response
                        # maps videoId -> setVideoId, one per video, so the
                        # copies can't be told apart; reload rather than
                        # stamp both rows with the same setVideoId.
                        library.reload(current_pid)
                    elif self.tracks:
                        table = library.query_one("#library-tracks", TrackTable)
                        table.append_tracks(
                            self._tracks_for_append(ytmusic.last_added_set_video_ids)
                        )
                        library.update_track_count()
                    else:
                        library.update_track_count(+len(self.video_ids))
            except Exception:
                logger.exception("Library track/count update failed")

            track_word = "track" if len(self.video_ids) == 1 else "tracks"
            self.notify(
                f"Added {len(self.video_ids)} {track_word} to '{title}'",
                severity="information",
            )
            self.dismiss(playlist_id)

        except Exception:
            logger.exception("Failed to add tracks to playlist %r", playlist_id)
            self.notify(f"Failed to add to '{title}'", severity="error")
            status.update("Error")
            self._release()


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same view-model engines; these
# bindings override the pure-Python transcriptions above when the
# extension is importable.  The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        playlist_picker_count_label as _native_playlist_picker_count_label,
    )
    from ytm_player.utils._ytm_player_native import (
        playlist_picker_rows as _native_playlist_picker_rows,
    )
    from ytm_player.utils._ytm_player_native import (
        playlist_picker_sort as _native_playlist_picker_sort,
    )
    from ytm_player.utils._ytm_player_native import (
        playlist_record_recent as _native_playlist_record_recent,
    )
except ImportError:
    pass
else:

    def _record_recent_ids(recent: list[str], playlist_id: str) -> list[str]:  # noqa: F811
        """The move-to-front update (native Rust engine)."""
        return _native_playlist_record_recent(recent, playlist_id)

    def _sorted_by_recent(  # noqa: F811
        playlists: list[dict[str, Any]], recent_ids: list[str]
    ) -> list[dict[str, Any]]:
        """The recency sort (native Rust engine)."""
        return _native_playlist_picker_sort(playlists, recent_ids)

    def _visible_picker_rows(  # noqa: F811
        playlists: list[dict[str, Any]], filter_text: str
    ) -> list[tuple[Any, Any, Any]]:
        """The filtered rows (native Rust engine)."""
        return _native_playlist_picker_rows(playlists, filter_text)

    def _count_label(count: int) -> str:  # noqa: F811
        """The playlist-count status label (native Rust engine)."""
        return _native_playlist_picker_count_label(count)
