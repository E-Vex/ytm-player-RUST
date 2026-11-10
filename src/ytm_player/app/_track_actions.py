"""Track action handling mixin for YTMPlayerApp.

The module's pure decision cores are dual-implemented: the Rust
extension (``_ytm_player_native``) implements them when it is built,
and the pure-Python transcriptions below run otherwise — the ID
extractions (``_first_artist_id``, ``_album_browse_id``), the
occurrence gate (``_queue_entry_id``), the popup-answer classification
(``_track_action_route``), the entity-action dispatch (``_entity_action_route``),
the like-toggle rating (``_like_toggle_rating``), the Remove-from-Queue
plan (``_remove_from_queue_plan``), the in-queue verdict
(``_in_queue_verdict``) and the artist-songs merge plan
(``_artist_songs_merge_plan``). The bottom-of-module seam rebinds each
to the native function when the extension is importable. The popups,
the notifications, the awaited workers, the queue mutations and the
navigation calls stay Python-side in the mixin.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable
from typing import Any

from ytm_player.app._base import YTMHostBase
from ytm_player.app._ownership import adopt_queue, owns_request, playback_request
from ytm_player.ui.playback_bar import PlaybackBar
from ytm_player.ui.popups.actions import ActionsPopup
from ytm_player.ui.popups.playlist_picker import PlaylistPicker
from ytm_player.ui.widgets.track_table import TrackTable
from ytm_player.utils.formatting import copy_to_clipboard, get_video_id, normalize_tracks

logger = logging.getLogger(__name__)


def _first_artist_id(source: dict) -> str:
    """Browse ID of the first artist listed on a track/entity dict, or ''.

    The gate is the reference's own — ``isinstance(artists, list) and
    artists and isinstance(artists[0], dict)`` — and the answer the raw
    ``artists[0].get("id") or artists[0].get("browseId", "")`` result
    (the browseId default ``""`` included).

    Implemented by the Rust extension
    (``_ytm_player_native.first_artist_id``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    artists = source.get("artists")
    if isinstance(artists, list) and artists and isinstance(artists[0], dict):
        return artists[0].get("id") or artists[0].get("browseId", "")
    return ""


def _album_browse_id(source: dict) -> str:
    """Album browse ID from a track/entity dict, or ''.

    The nested album id must outrank the top-level ``browseId``: track dicts
    can carry a non-album ``browseId``, while album entities have no nested
    ``album`` dict and legitimately resolve through ``browseId``. The reads
    happen in the reference's own order — ``album`` first, the nested ``id``
    only when the album is a dict — and the truthiness chain
    ``album_id or nested or browseId or ""`` short-circuits each read.

    Implemented by the Rust extension
    (``_ytm_player_native.album_browse_id``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    album = source.get("album")
    nested = album.get("id") if isinstance(album, dict) else None
    return source.get("album_id") or nested or source.get("browseId") or ""


def _queue_entry_id(from_queue: Any, key: Any) -> Any:
    """The queue entry id a row's occurrence *key* names, or None.

    Only the Queue page's table keys its rows by queue entry id
    (``from_queue``); any other table's key, or no key, names no queue
    occurrence. The gate is ``from_queue and isinstance(key, int)`` — a
    ``bool`` key counts, CPython's own subclass rule — and the key
    object passes through unchanged.

    Implemented by the Rust extension
    (``_ytm_player_native.queue_entry_id``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return key if from_queue and isinstance(key, int) else None


def _track_action_route(action_id: Any) -> dict[str, Any]:
    """The classification behind the track ActionsPopup's result callback.

    Which handler ``_handle_action_result`` routes an *action_id* to:
    ``None`` answers nothing (``"none"``), the first equal match among
    the twelve known action ids answers its own arm name, and anything
    no arm claims is ``"unhandled"`` — the reference's if/elif chain
    has no else, so an unknown id simply does nothing. Every comparison
    is CPython's own ``==``, and the arm order is the reference's own
    (first equal match wins).

    Implemented by the Rust extension
    (``_ytm_player_native.track_action_route``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if action_id is None:
        return {"handler": "none"}
    arms = (
        "add_to_playlist",
        "play",
        "download",
        "play_next",
        "add_to_queue",
        "remove_from_queue",
        "start_radio",
        "go_to_artist",
        "go_to_album",
        "toggle_like",
        "copy_link",
        "remove_from_playlist",
    )
    for arm in arms:
        if action_id == arm:
            return {"handler": arm}
    return {"handler": "unhandled"}


def _entity_action_route(action_id: Any, item: Any, item_type: Any) -> dict[str, Any]:
    """The dispatch behind ``_dispatch_entity_action``.

    The pure half of the entity dispatch: the entity id and name are
    extracted first (``item.get("browseId") or item.get("album_id") or
    item.get("playlistId") or ""`` and ``item.get("title") or
    item.get("name") or item.get("artist") or "Unknown"``, each read
    short-circuited by every earlier falsy one), then the arms, in the
    reference's own order: ``play_all``/``shuffle_play`` answer
    ``"_play_album"``/``"_play_playlist"`` (with the ``shuffle`` flag)
    for the album and playlist types, ``add_to_queue`` and ``play_next``
    answer their album/playlist enqueue methods, ``start_radio`` answers
    ``"_start_artist_radio"`` for artists and ``"_start_playlist_radio"``
    for playlists (the playlist arm with no id gate), ``play_top_songs``
    answers ``"_play_artist_top_songs"`` for artists,
    ``go_to_artist``/``view_similar`` answer ``"_navigate_artist"`` with
    the first-artist id when the item lists artists and the entity id
    for artist items otherwise, ``go_to_album`` answers
    ``"_navigate_album"`` with the album browse id, and ``add_to_library``
    answers ``"_add_album_to_library"`` for albums. Every typed arm
    without an entity id answers ``"notify"`` (handled — the caller's
    own "No ID available" toast); an action no arm claims — or a type
    the arm doesn't serve — answers ``"unhandled"`` with ``handled``
    false.

    Implemented by the Rust extension
    (``_ytm_player_native.entity_action_route``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    entity_id = item.get("browseId") or item.get("album_id") or item.get("playlistId") or ""
    entity_name = item.get("title") or item.get("name") or item.get("artist") or "Unknown"

    if action_id in ("play_all", "shuffle_play"):
        shuffle = action_id == "shuffle_play"
        if item_type == "album":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_play_album",
                "shuffle": shuffle,
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        if item_type == "playlist":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_play_playlist",
                "shuffle": shuffle,
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        return {"handled": False, "handler": "unhandled"}

    if action_id == "add_to_queue":
        if item_type == "album":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_add_album_to_queue",
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        if item_type == "playlist":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_add_playlist_to_queue",
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        return {"handled": False, "handler": "unhandled"}

    if action_id == "play_next":
        if item_type == "album":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_add_album_next",
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        if item_type == "playlist":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_add_playlist_next",
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        return {"handled": False, "handler": "unhandled"}

    if action_id == "start_radio":
        if item_type == "artist":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {"handled": True, "handler": "_start_artist_radio", "entity_id": entity_id}
        if item_type == "playlist":
            # No id gate — the caller passes the item itself.
            return {"handled": True, "handler": "_start_playlist_radio"}
        return {"handled": False, "handler": "unhandled"}

    if action_id == "play_top_songs":
        if item_type == "artist":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {"handled": True, "handler": "_play_artist_top_songs", "entity_id": entity_id}
        return {"handled": False, "handler": "unhandled"}

    if action_id in ("go_to_artist", "view_similar"):
        artists = item.get("artists")
        if isinstance(artists, list) and artists:
            artist_id = _first_artist_id(item)
        else:
            artist_id = entity_id if item_type == "artist" else ""
        if not artist_id:
            return {"handled": True, "handler": "notify"}
        return {"handled": True, "handler": "_navigate_artist", "artist_id": artist_id}

    if action_id == "go_to_album":
        album_id = _album_browse_id(item)
        if not album_id:
            return {"handled": True, "handler": "notify"}
        return {"handled": True, "handler": "_navigate_album", "album_id": album_id}

    if action_id == "add_to_library":
        if item_type == "album":
            if not entity_id:
                return {"handled": True, "handler": "notify"}
            return {
                "handled": True,
                "handler": "_add_album_to_library",
                "entity_id": entity_id,
                "entity_name": entity_name,
            }
        return {"handled": False, "handler": "unhandled"}

    return {"handled": False, "handler": "unhandled"}


def _like_toggle_rating(track: Any) -> dict[str, Any]:
    """The like-toggle rating behind the track ActionsPopup's Like arm.

    ``is_liked`` is the raw ``track.get("likeStatus") == "LIKE" or
    track.get("liked", False)`` result — the ``or`` keeping the very
    object it selected — and the rating/label pair
    (``"INDIFFERENT"``/``"Unliked"`` when the or-result is truthy,
    ``"LIKE"``/``"Liked"`` otherwise) resolves through that same
    truthiness.

    Implemented by the Rust extension
    (``_ytm_player_native.like_toggle_rating``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    is_liked = track.get("likeStatus") == "LIKE" or track.get("liked", False)
    rating = "INDIFFERENT" if is_liked else "LIKE"
    label = "Unliked" if is_liked else "Liked"
    return {"is_liked": is_liked, "rating": rating, "label": label}


def _remove_from_queue_plan(
    from_queue: Any,
    queue_entry_id: Any,
    queue_tracks: Any,
    video_id: Any,
) -> dict[str, Any]:
    """The removal plan behind ``_remove_from_queue``.

    The occurrence arm ``from_queue or queue_entry_id is not None``
    answers ``"entry"`` when an id is present (the caller's own
    ``remove_entry`` attempt deciding success — only that occurrence
    may go) and ``"stale"`` when ``from_queue`` carries no id (a Queue
    page row whose id could not be read is stale, never nameless).
    Without any occurrence to name, a falsy video id answers ``"none"``,
    and otherwise the first queue row whose ``t.get("video_id") ==
    video_id`` answers ``"remove_at"`` carrying that row's index — a
    no-match walk answering ``"none"``.

    Implemented by the Rust extension
    (``_ytm_player_native.remove_from_queue_plan``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if from_queue or queue_entry_id is not None:
        return {"arm": "entry" if queue_entry_id is not None else "stale"}
    if not video_id:
        return {"arm": "none"}
    for i, t in enumerate(queue_tracks):
        if t.get("video_id") == video_id:
            return {"arm": "remove_at", "index": i}
    return {"arm": "none"}


def _in_queue_verdict(video_id: Any, queue_tracks: Any) -> bool:
    """Whether the track the actions popup opens on is already queued.

    The popup swaps "Add to Queue" for "Remove from Queue" when it is.
    A falsy video id answers False without touching the queue; otherwise
    the ``any()`` walk — each row's ``t.get("video_id") == video_id`` —
    short-circuits on the first equal row.

    Implemented by the Rust extension
    (``_ytm_player_native.in_queue_verdict``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return bool(video_id) and any(t.get("video_id") == video_id for t in queue_tracks)


def _artist_songs_merge_plan(
    all_tracks: Any,
    initial_tracks: Any,
    queue_tracks: Any,
) -> dict[str, Any]:
    """The enrich-and-append plan behind ``_fetch_remaining_artist_songs``.

    The full list's index ``{t.get("video_id", ""): t for t in all_tracks
    if t.get("video_id")}`` (last-wins per video id) and the initial
    batch's id set are built first, then the enrichment pass answers one
    ``(queue_track, full_track)`` pair per queue track whose id is in the
    index — the very objects, so the caller's in-place updates hit the
    queue's own dicts — and the append pass answers every all-track
    whose id is not in the initial set (duplicates within the fetch
    included, the membership-only check the reference runs).

    Implemented by the Rust extension
    (``_ytm_player_native.artist_songs_merge_plan``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    full_by_id = {t.get("video_id", ""): t for t in all_tracks if t.get("video_id")}
    existing_ids = {t.get("video_id", "") for t in initial_tracks}
    updates = [
        (qt, full_by_id[qt.get("video_id", "")])
        for qt in queue_tracks
        if qt.get("video_id", "") in full_by_id
    ]
    appends = [t for t in all_tracks if t.get("video_id", "") not in existing_ids]
    return {"updates": updates, "appends": appends}


class TrackActionsMixin(YTMHostBase):
    """Track table integration and actions popup wiring."""

    async def on_track_table_track_selected(self, message: TrackTable.TrackSelected) -> None:
        """Handle track selection from any TrackTable widget."""
        track = message.track
        index = message.index

        # Set the queue position and play.
        self.queue.jump_to_real(index)
        await self.play_track(track)

    def _focused_track_table(self) -> TrackTable | None:
        """The TrackTable that has focus, or contains the focused widget."""
        widget = self.focused
        while widget is not None:
            if isinstance(widget, TrackTable):
                return widget
            widget = widget.parent
        return None

    def _get_focused_track(self) -> dict | None:
        """Try to get a track dict from the currently focused widget."""
        table = self._focused_track_table()
        return table.selected_track if table is not None else None

    @staticmethod
    def _queue_entry_id(from_queue: bool, key: Hashable | None) -> int | None:
        """The queue entry id a row's occurrence *key* names, or None.

        Only the Queue page's table keys its rows by queue entry id
        (``from_queue``); any other table's key, or no key, names no queue
        occurrence. Delegates to the ``_queue_entry_id`` engine (native
        when the Rust extension is built, the pure-Python transcription
        otherwise) — the module-level function of the same name, looked
        up at call time so the bottom-of-module seam steers it.
        """
        return _queue_entry_id(from_queue, key)

    def _resolve_action_track(self) -> dict | None:
        """Focused-table track, falling back to the currently playing track."""
        track = self._get_focused_track()
        if track:
            return track
        if self.player and self.player.current_track:
            return self.player.current_track
        return None

    def _enqueue_track(self, track: dict) -> None:
        """Append *track* to the queue, refresh the queue page, and notify."""
        self.queue.add(track)
        self._refresh_queue_page()
        self.notify(f"Added to queue: {track.get('title') or 'track'}", timeout=2)

    def _play_track_next(self, track: dict) -> None:
        """Insert *track* right after the current track, refresh, and notify."""
        self.queue.add_next(track)
        self._refresh_queue_page()
        self.notify(f"Playing next: {track.get('title') or 'track'}", timeout=2)

    def _add_focused_to_queue(self) -> None:
        """Enqueue the focused (or playing) track — the app-level Add-to-Queue key."""
        track = self._resolve_action_track()
        if not track:
            self.notify("No track selected.", severity="warning", timeout=2)
            return
        self._enqueue_track(track)

    def _play_focused_next(self) -> None:
        """Play the focused (or playing) track next — the app-level Play-Next key."""
        track = self._resolve_action_track()
        if not track:
            self.notify("No track selected.", severity="warning", timeout=2)
            return
        self._play_track_next(track)

    async def _open_add_to_playlist(self) -> None:
        """Open PlaylistPicker for the marked tracks, else the playing track.

        Marks are read from the current page's tables whether or not one
        has focus — the table shows them (check marks and its status line).
        A page with marks in more than one table gets a warning instead of
        a guess. Success clears the marks that were submitted and only
        those: the table's selection generation is captured now and
        compared when the picker returns, so a selection that changed
        underneath (a marked row gone in a background refresh, a reloaded
        list) stays put. Cancel and failure keep the marks.
        """
        page = self._get_current_page()
        marked = [t for t in page.query(TrackTable) if t.marked_count] if page else []
        if len(marked) > 1:
            self.notify(
                "Tracks are marked in more than one list — clear one first (Esc).",
                severity="warning",
                timeout=3,
            )
            return
        if marked:
            self._add_marked_to_playlist(marked[0])
            return

        track = None

        # Prefer the currently playing track.
        if self.player and self.player.current_track:
            track = self.player.current_track

        if not track:
            self.notify("No track is playing.", severity="warning", timeout=2)
            return

        video_id = get_video_id(track)
        if not video_id:
            self.notify("Track has no video ID.", severity="warning", timeout=2)
            return

        self.push_screen(PlaylistPicker(video_ids=[video_id], tracks=[track]))

    def _add_marked_to_playlist(self, table: TrackTable) -> None:
        """Push the picker for *table*'s marks, in displayed order.

        Hidden marked rows are included (the table orders them by the same
        sort). Every marked occurrence is submitted, so two marked copies of
        a song go through the picker's duplicate confirmation as two copies.
        A range in progress ends first — its marks stay — so cancelling and
        moving on doesn't extend it. The marks are cleared only on success
        and only if the selection is still the one that was submitted.
        """
        table.end_range_mode()
        video_ids: list[str] = []
        submitted: list[dict] = []
        for track in table.marked_tracks():
            video_id = get_video_id(track)
            if video_id:
                video_ids.append(video_id)
                submitted.append(track)
        if not video_ids:
            self.notify("The marked tracks have no video IDs.", severity="warning", timeout=2)
            return
        generation = table.selection_generation

        def _on_picker_closed(playlist_id: str | None) -> None:
            if playlist_id and table.is_attached and table.selection_generation == generation:
                table.clear_marks()

        self.push_screen(PlaylistPicker(video_ids=video_ids, tracks=submitted), _on_picker_closed)

    async def _open_track_actions(self) -> None:
        """Open ActionsPopup for the focused track."""
        table = self._focused_track_table()
        track = table.selected_track if table is not None else None
        from_queue = table is not None and table.queue_entry_keys
        queue_entry_id = None
        if table is not None and track is not None:
            original = track.get("_original_index")
            key = table.occurrence_key(original) if isinstance(original, int) else None
            queue_entry_id = self._queue_entry_id(from_queue, key)
        if not track:
            # Fall back to currently playing track. It did not come from a
            # Queue page row (the table may be filtered to nothing), so it
            # names no occurrence and carries no Queue origin.
            if self.player and self.player.current_track:
                track = self.player.current_track
                queue_entry_id = None
                from_queue = False
            else:
                self.notify("No track selected.", severity="warning", timeout=2)
                return

        self._open_actions_for_track(track, queue_entry_id=queue_entry_id, from_queue=from_queue)

    def _open_actions_for_track(
        self, track: dict, *, queue_entry_id: int | None = None, from_queue: bool = False
    ) -> None:
        """Push ActionsPopup for a specific track dict.

        *queue_entry_id* names the queue occurrence the popup was opened on
        (a Queue page row); *from_queue* says the track came from that page
        even if no id could be read, so removal never guesses by video id.
        The popup-answer routing — which handler each picked action goes
        to — resolves through the ``_track_action_route`` engine (native
        when the Rust extension is built, the pure-Python transcription
        otherwise), the in-queue verdict through ``_in_queue_verdict``, the
        like-toggle rating through ``_like_toggle_rating``, and the
        artist/album navigation targets through the ``_first_artist_id``
        and ``_album_browse_id`` engines; this method performs each arm's
        calls.
        """

        def _handle_action_result(action_id: str | None) -> None:
            """Callback when the user picks an action from the popup."""
            handler = _track_action_route(action_id)["handler"]

            if handler == "none" or handler == "unhandled":
                # The reference's if/elif chain has no else — an unknown
                # id simply does nothing.
                return

            if handler == "add_to_playlist":
                video_id = get_video_id(track)
                if video_id:
                    self.push_screen(PlaylistPicker(video_ids=[video_id], tracks=[track]))
                return

            if handler == "play":
                self.run_worker(
                    self._play_popup_track(track, queue_entry_id, from_queue=from_queue)
                )
            elif handler == "download":
                self.run_worker(self._download_track(track))
            elif handler == "play_next":
                self._play_track_next(track)
            elif handler == "add_to_queue":
                self._enqueue_track(track)
            elif handler == "remove_from_queue":
                self._remove_from_queue(track, queue_entry_id, from_queue=from_queue)
            elif handler == "start_radio":
                self.run_worker(self._fetch_and_play_radio(track))
            elif handler == "go_to_artist":
                artist_id = _first_artist_id(track)
                if artist_id:
                    self.run_worker(
                        self.navigate_to("context", context_type="artist", context_id=artist_id)
                    )
            elif handler == "go_to_album":
                album_id = _album_browse_id(track)
                if album_id:
                    self.run_worker(
                        self.navigate_to("context", context_type="album", context_id=album_id)
                    )
            elif handler == "toggle_like":
                video_id = get_video_id(track)
                ytmusic = self.ytmusic
                if video_id and ytmusic is not None:
                    verdict = _like_toggle_rating(track)
                    rating = verdict["rating"]
                    label = verdict["label"]

                    async def _rate(vid: str, r: str, lbl: str) -> None:
                        from ytm_player.services.ytmusic import mutation_failure_suffix

                        result = await ytmusic.rate_song(vid, r)
                        if result == "success":
                            track["likeStatus"] = r
                            self.notify(lbl, timeout=2)
                        else:
                            self.notify(
                                f"Couldn't {lbl.lower()} — {mutation_failure_suffix(result)}",
                                severity="error",
                                timeout=3,
                            )

                    self.run_worker(_rate(video_id, rating, label))
            elif handler == "copy_link":
                video_id = get_video_id(track)
                if video_id:
                    link = f"https://music.youtube.com/watch?v={video_id}"
                    if copy_to_clipboard(link):
                        self.notify("Link copied", timeout=2)
                    else:
                        self.notify(link, timeout=5)
            elif handler == "remove_from_playlist":
                self.run_worker(self._remove_track_from_playlist(track))

        # Detect whether this track is currently in the queue so the popup
        # can swap "Add to Queue" for "Remove from Queue" — the verdict
        # resolved by the ``_in_queue_verdict`` engine (native when the
        # Rust extension is built, the pure-Python transcription
        # otherwise).
        track_vid = get_video_id(track)
        in_queue = _in_queue_verdict(track_vid, self.queue.tracks)
        # Detect whether we're viewing a playlist so "Remove from Playlist"
        # can be offered.
        in_playlist = self._current_page == "library" and bool(
            self._current_page_kwargs.get("playlist_id")
        )
        self.push_screen(
            ActionsPopup(track, item_type="track", in_queue=in_queue, in_playlist=in_playlist),
            _handle_action_result,
        )

    def _remove_from_queue(
        self, track: dict, queue_entry_id: int | None, *, from_queue: bool = False
    ) -> None:
        """Remove *track* from the queue on the popup's Remove from Queue.

        With an entry id only that occurrence goes. If it is gone by the time
        the popup closes (removed, or the queue rebuilt, meanwhile) nothing is
        removed — never another copy of the same song — and the page is
        re-rendered so the stale row disappears. A Queue page row whose id
        could not be read (*from_queue* without an id) is treated the same
        way. Only without any occurrence to name (popup opened from the
        playback bar or another page) does the first copy by video id go.

        The removal plan — which arm the removal takes — resolves through
        the ``_remove_from_queue_plan`` engine (native when the Rust
        extension is built, the pure-Python transcription otherwise); the
        removals, the queue-page refreshes and the toasts stay here.
        """
        plan = _remove_from_queue_plan(
            from_queue, queue_entry_id, self.queue.tracks, get_video_id(track)
        )
        arm = plan["arm"]
        if arm == "entry":
            # The engine's entry arm answers only with an id present; the
            # id re-check narrows for the type checker exactly like the
            # reference's own compound gate — nothing runs between the
            # consult and here that could clear it.
            if queue_entry_id is not None and self.queue.remove_entry(queue_entry_id):
                self._refresh_queue_page()
                self.notify("Removed from queue", timeout=2)
            else:
                self._refresh_queue_page()
                self.notify("That queue entry is no longer present", severity="warning", timeout=3)
            return
        if arm == "stale":
            self._refresh_queue_page()
            self.notify("That queue entry is no longer present", severity="warning", timeout=3)
            return
        if arm == "remove_at":
            self.queue.remove(plan["index"])
            self._refresh_queue_page()
            self.notify("Removed from queue", timeout=2)

    def _sync_shuffle_bar(self) -> None:
        try:
            bar = self.query_one("#playback-bar")
            bar.update_shuffle(self.queue.shuffle_enabled)  # type: ignore[attr-defined]
            bar.refresh_shuffle_lock_state()  # type: ignore[attr-defined]
        except Exception:
            pass

    @playback_request
    async def _play_popup_track(
        self, track: dict, entry_id: int | None, *, from_queue: bool
    ) -> None:
        if from_queue:
            queued = self.queue.jump_to_entry(entry_id) if entry_id is not None else None
            if queued is not None:
                self._refresh_queue_page()
                await self.play_track(queued)
                return
            self._refresh_queue_page()
            self.notify("That queue entry is no longer present", severity="warning", timeout=3)
            return
        await self._replace_queue_and_play([track])

    @playback_request
    async def _replace_queue_and_play(
        self,
        tracks: list[dict],
        *,
        entity_id: str | None = None,
        start_index: int | None = 0,
        shuffle: bool | None = None,
        autoplay: bool = True,
    ) -> None:
        if not tracks:
            self.notify("No tracks found", severity="warning", timeout=3)
            return
        if shuffle is True:
            import random

            tracks = list(tracks)
            random.shuffle(tracks)
        self.queue.clear()
        adopt_queue(self)
        self.queue.add_multiple(tracks)
        self.queue.set_context(entity_id)
        saved_pref = self.shuffle_prefs.get(entity_id)
        if saved_pref is not None:
            # Per-collection shuffle memory wins in both directions — a
            # saved OFF must clear a currently-ON shuffle, not just force ON.
            if self.queue.shuffle_enabled != saved_pref:
                self.queue.toggle_shuffle()
        elif shuffle is False or (shuffle is None and start_index is not None):
            # A chosen row retains the explicit-play policy. A whole-playlist
            # start (no source index) instead follows the active shuffle mode.
            if self.queue.shuffle_enabled:
                self.queue.toggle_shuffle()
        if start_index is None:
            # Start at the front of the effective playback order, not source
            # row zero (which may sit anywhere inside the shuffled queue).
            self.queue.jump_to(0)
        else:
            self.queue.jump_to_real(start_index)
        self._refresh_queue_page()
        self._sync_shuffle_bar()
        if autoplay and self.queue.current_track is not None:
            await self.play_track(self.queue.current_track)

    def _append_to_queue(self, tracks: list[dict], label: str) -> None:
        self.queue.add_multiple(tracks)
        self._refresh_queue_page()
        self.notify(f"Added {label} ({len(tracks)} tracks) to queue", timeout=3)

    def _play_next_multiple(self, tracks: list[dict], label: str) -> None:
        self.queue.add_next_multiple(tracks)
        self._refresh_queue_page()
        self.notify(f"Playing {label} next ({len(tracks)} tracks)", timeout=3)

    def _refresh_queue_page(self) -> None:
        """Refresh the queue page if it's currently displayed."""
        try:
            from ytm_player.ui.pages.queue import QueuePage

            queue_page = self.query_one(QueuePage)
            queue_page._refresh_queue()
        except Exception:
            pass

    async def _remove_track_from_playlist(self, track: dict) -> None:
        """Remove *track* from the currently open playlist."""
        playlist_id = self._current_page_kwargs.get("playlist_id", "")
        if not playlist_id or not self.ytmusic:
            self.notify("No playlist context", severity="error", timeout=2)
            return

        video_id = track.get("video_id", "")
        set_video_id = track.get("setVideoId", "")
        if not video_id or not set_video_id:
            if track.get("_needs_reload_for_removal"):
                self.notify(
                    "Reload the playlist before removing a just-added track",
                    severity="warning",
                    timeout=3,
                )
            else:
                self.notify("Track missing required IDs", severity="error", timeout=2)
            return

        from ytm_player.services.ytmusic import mutation_failure_suffix
        from ytm_player.ui.pages.library import LibraryPage
        from ytm_player.ui.sidebars.playlist_sidebar import LibraryPanel, PlaylistSidebar
        from ytm_player.utils.formatting import strip_vl_prefix

        result = await self.ytmusic.remove_playlist_items(
            strip_vl_prefix(playlist_id),
            [{"videoId": video_id, "setVideoId": set_video_id}],
        )
        if result != "success":
            suffix = mutation_failure_suffix(result)
            self.notify(f"Failed to remove track — {suffix}", severity="error", timeout=3)
            return

        self.notify("Track removed", timeout=2)

        # Remove from the visible TrackTable and update counts.
        try:
            library = self.query_one(LibraryPage)
            table = library.query_one("#library-tracks", TrackTable)
            removed = table.remove_track(video_id, set_video_id=set_video_id)
            if removed:
                library.update_track_count()
            # Update sidebar count.
            ps = self.query_one("#playlist-sidebar", PlaylistSidebar)
            panel = ps.query_one("#ps-playlists", LibraryPanel)
            panel.update_item_count(playlist_id, -1)
        except Exception:
            logger.exception("Failed to update UI after track removal")

    def on_track_table_track_right_clicked(self, message: TrackTable.TrackRightClicked) -> None:
        """Handle right-click on any TrackTable -- open actions popup.

        Works from the message's snapshot only: by now the table may have
        re-rendered (a queue change) or been removed (navigation), so the
        clicked row must not be looked up again.
        """
        queue_entry_id = self._queue_entry_id(message.from_queue, message.occurrence_key)
        self._open_actions_for_track(
            message.track, queue_entry_id=queue_entry_id, from_queue=message.from_queue
        )

    def on_playback_bar_track_right_clicked(self, message: PlaybackBar.TrackRightClicked) -> None:
        """Handle right-click on the playback bar -- open actions popup."""
        self._open_actions_for_track(message.track)

    # ── Column-aware right-click handlers ─────────────────────────────

    def on_track_table_artist_right_clicked(self, message: TrackTable.ArtistRightClicked) -> None:
        """Handle right-click on Artist column -- open artist actions popup."""
        self._open_actions_for_artist(message.track)

    def on_track_table_album_right_clicked(self, message: TrackTable.AlbumRightClicked) -> None:
        """Handle right-click on Album column -- open album actions popup."""
        self._open_actions_for_album(message.track)

    def _open_actions_for_artist(self, track: dict) -> None:
        """Push ActionsPopup for an artist of a track.

        For multi-artist tracks, shows a picker first.
        """
        artists = track.get("artists", [])
        if not artists or not isinstance(artists, list):
            self.notify("No artist info available.", severity="warning", timeout=2)
            return
        valid = [a for a in artists if a.get("id") or a.get("browseId")]
        if not valid:
            self.notify("No artist info available.", severity="warning", timeout=2)
            return
        if len(valid) == 1:
            self._show_artist_actions(valid[0])
            return
        self._show_artist_picker(track, valid)

    def _show_artist_picker(self, track: dict, artists: list[dict]) -> None:
        """Show a picker popup for multi-artist tracks."""
        picker_actions = [(str(i), a.get("name", "Unknown Artist")) for i, a in enumerate(artists)]
        picker_item = {"title": "Select Artist"}

        def _on_pick(action_id: str | None) -> None:
            if action_id is None:
                return
            self._show_artist_actions(artists[int(action_id)], back_to=(track, artists))

        self.push_screen(ActionsPopup(picker_item, actions=picker_actions), _on_pick)

    def _show_artist_actions(
        self, artist: dict, back_to: tuple[dict, list[dict]] | None = None
    ) -> None:
        """Push the artist actions popup for a single artist dict."""
        browse_id = artist.get("id") or artist.get("browseId", "")
        if not browse_id:
            return
        item: dict[str, Any] = {
            "browseId": browse_id,
            "artist": artist.get("name", "Unknown Artist"),
        }

        def _handle(action_id: str | None) -> None:
            if action_id is None:
                return
            if action_id == "_back" and back_to:
                self._show_artist_picker(back_to[0], back_to[1])
            elif action_id == "toggle_subscribe":
                self.run_worker(self._toggle_artist_subscribe_simple(browse_id))
            elif action_id == "copy_link":
                link = f"https://music.youtube.com/browse/{browse_id}"
                if copy_to_clipboard(link):
                    self.notify("Link copied", timeout=2)
                else:
                    self.notify(link, timeout=5)
            else:
                self.run_worker(self._dispatch_entity_action(action_id, item, "artist"))

        actions: list[tuple[str, str]] | None = None
        if back_to:
            from ytm_player.ui.popups.actions import ARTIST_ACTIONS

            actions = list(ARTIST_ACTIONS) + [("_back", "← Back")]

        self.push_screen(ActionsPopup(item, item_type="artist", actions=actions), _handle)

    def _open_actions_for_album(self, track: dict) -> None:
        """Push ActionsPopup for the album of a track."""
        album = track.get("album", {})
        album_id = (
            track.get("album_id") or (album.get("id") if isinstance(album, dict) else None) or ""
        )
        if not album_id:
            self.notify("No album info available.", severity="warning", timeout=2)
            return
        album_name = album if isinstance(album, str) else album.get("name", "Unknown Album")
        item: dict[str, Any] = {
            "browseId": album_id,
            "title": album_name,
        }
        if track.get("artists"):
            item["artists"] = track["artists"]

        def _handle(action_id: str | None) -> None:
            if action_id is None:
                return
            if action_id == "copy_link":
                link = f"https://music.youtube.com/browse/{album_id}"
                if copy_to_clipboard(link):
                    self.notify("Link copied", timeout=2)
                else:
                    self.notify(link, timeout=5)
            else:
                self.run_worker(self._dispatch_entity_action(action_id, item, "album"))

        self.push_screen(ActionsPopup(item, item_type="album"), _handle)

    # ── Artist action methods (shared with SearchPage dispatch) ───────

    @playback_request
    async def _start_artist_radio(self, browse_id: str) -> None:
        """Fetch artist data and start a radio from their radioId or top songs."""
        if not self.ytmusic:
            return
        self.notify("Loading radio...", timeout=3)
        data = await self.ytmusic.get_artist(browse_id)
        if not owns_request(self):
            return
        if not data:
            self.notify("Couldn't load artist data.", severity="warning", timeout=3)
            return
        artist_name = data.get("name", "Unknown Artist")
        radio_id = data.get("radioId")
        if radio_id:
            tracks = await self.ytmusic.get_watch_playlist(playlist_id=radio_id, radio=True)
            if not owns_request(self):
                return
            normalized = normalize_tracks(tracks)
            if normalized:
                self.queue.clear()
                adopt_queue(self)
                self.queue.set_context(None)
                self.queue.set_radio_tracks(normalized)
                self._refresh_queue_page()
                first = self.queue.next_track()
                if first:
                    await self.play_track(first)
                if not owns_request(self):
                    return
                self.notify(f"Playing: Radio from {artist_name}", timeout=4)
            else:
                self.notify("No radio suggestions available.", severity="warning", timeout=3)
        else:
            songs = data.get("songs", {})
            results = songs.get("results", []) if isinstance(songs, dict) else []
            seeds = [t for t in results if t.get("videoId")]
            if seeds:
                await self._fetch_and_play_radio(seeds, label=f"Radio from {artist_name}")
            else:
                self.notify("No songs to seed radio.", severity="warning", timeout=3)

    @playback_request
    async def _play_artist_top_songs(self, browse_id: str) -> None:
        """Fetch artist top songs, queue them, and start playback."""
        if not self.ytmusic:
            return
        self.notify("Loading top songs...", timeout=3)
        data = await self.ytmusic.get_artist(browse_id)
        if not owns_request(self):
            return
        if not data:
            self.notify("Couldn't load artist data.", severity="warning", timeout=3)
            return
        songs_section = data.get("songs", {})
        top_tracks = normalize_tracks(
            songs_section.get("results", []) if isinstance(songs_section, dict) else []
        )
        if not top_tracks:
            self.notify("No songs found for this artist.", severity="warning", timeout=3)
            return
        artist_name = data.get("name", "Unknown Artist")
        await self._replace_queue_and_play(top_tracks, shuffle=None)
        if not owns_request(self):
            return
        self.notify(f"Playing top songs from {artist_name}", timeout=4)
        songs_browse_id = songs_section.get("browseId") if isinstance(songs_section, dict) else None
        if songs_browse_id:
            self.run_worker(
                self._fetch_remaining_artist_songs(songs_browse_id, top_tracks),
                name="fetch-artist-songs",
                group="fetch-artist-songs",
                exclusive=True,
            )

    @playback_request
    async def _fetch_remaining_artist_songs(
        self, browse_id: str, initial_tracks: list[dict[str, Any]]
    ) -> None:
        """Background-fetch the full artist song list, enrich initial tracks, and append rest.

        The enrich-and-append plan — which queued initial tracks get
        enriched from which full-list entries, and which fetched tracks
        are appended — resolves through the ``_artist_songs_merge_plan``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); the in-place updates, the
        queue adds and the queue-page refresh stay here.
        """
        try:
            if not self.ytmusic:
                return
            pl = await self.ytmusic.get_playlist(browse_id)
            if not owns_request(self):
                return
            all_tracks = normalize_tracks(pl.get("tracks", []) if isinstance(pl, dict) else [])
            plan = _artist_songs_merge_plan(all_tracks, initial_tracks, self.queue.tracks)
            for queue_track, full in plan["updates"]:
                queue_track.update(full)
            for t in plan["appends"]:
                self.queue.add(t)
            self._refresh_queue_page()
        except Exception:
            logger.debug("Background artist songs fetch failed", exc_info=True)

    async def _toggle_artist_subscribe_simple(self, browse_id: str) -> None:
        """Subscribe/unsubscribe without cached state (used from track table context)."""
        from ytm_player.services.ytmusic import mutation_failure_suffix

        if not self.ytmusic:
            return
        data = await self.ytmusic.get_artist(browse_id)
        if not data or not data.get("channelId"):
            self.notify("Couldn't load artist data.", severity="warning", timeout=3)
            return
        channel_id = data["channelId"]
        is_subscribed = bool(data.get("subscribed"))
        if is_subscribed:
            result = await self.ytmusic.unsubscribe_artist(channel_id)
        else:
            result = await self.ytmusic.subscribe_artist(channel_id)
        if result == "success":
            label = "Unsubscribed" if is_subscribed else "Subscribed"
            self.notify(label, timeout=2)
        else:
            verb = "unsubscribe" if is_subscribed else "subscribe"
            self.notify(
                f"Couldn't {verb} — {mutation_failure_suffix(result)}",
                severity="error",
                timeout=3,
            )

    async def _add_album_to_library(self, album_id: str, album_name: str) -> None:
        """Add an album to the user's library."""
        from ytm_player.services.ytmusic import mutation_failure_suffix

        if not self.ytmusic:
            return
        album_data = await self.ytmusic.get_album(album_id)
        playlist_id = album_data.get("audioPlaylistId", "")
        if not playlist_id:
            self.notify("Couldn't add to library.", severity="warning", timeout=3)
            return
        result = await self.ytmusic.add_to_library(playlist_id)
        if result == "success":
            self.notify(f"Added {album_name} to library", timeout=2)
        else:
            self.notify(
                f"Couldn't add — {mutation_failure_suffix(result)}", severity="error", timeout=3
            )

    @playback_request
    async def _play_album(self, album_id: str, album_name: str, *, shuffle: bool = False) -> None:
        """Fetch album tracks, replace queue, and start playback."""
        if not self.ytmusic:
            return
        self.notify("Loading album...", timeout=3)
        data = await self.ytmusic.get_album(album_id)
        if not owns_request(self):
            return
        tracks = normalize_tracks(data.get("tracks", []) if isinstance(data, dict) else [])
        await self._replace_queue_and_play(tracks, entity_id=album_id, shuffle=shuffle)
        if not owns_request(self):
            return
        action = "Shuffling" if shuffle else "Playing"
        self.notify(f"{action}: {album_name}", timeout=4)

    async def _add_album_to_queue(self, album_id: str, album_name: str) -> None:
        """Fetch album tracks and add them to the queue."""
        if not self.ytmusic:
            return
        data = await self.ytmusic.get_album(album_id)
        tracks = normalize_tracks(data.get("tracks", []) if isinstance(data, dict) else [])
        if not tracks:
            self.notify("No tracks found.", severity="warning", timeout=3)
            return
        self._append_to_queue(tracks, album_name)

    async def _add_album_next(self, album_id: str, album_name: str) -> None:
        """Fetch album tracks and insert them right after the current track."""
        if not self.ytmusic:
            return
        data = await self.ytmusic.get_album(album_id)
        tracks = normalize_tracks(data.get("tracks", []) if isinstance(data, dict) else [])
        if not tracks:
            self.notify("No tracks found.", severity="warning", timeout=3)
            return
        self._play_next_multiple(tracks, album_name)

    @playback_request
    async def _play_entity_and_navigate(
        self, action_id: str, item: dict, item_type: str, *, sidebar: bool = False
    ) -> None:
        """Own the whole popup action, including its post-play navigation."""
        previous = self.queue.current_track
        await self._dispatch_entity_action(action_id, item, item_type)
        if (
            owns_request(self)
            and self.queue.current_track is not None
            and self.queue.current_track is not previous
        ):
            if sidebar:
                self._active_library_playlist_id = item.get("playlistId") or item.get("browseId")
            await self.navigate_to("queue")

    @playback_request
    async def _dispatch_entity_action(self, action_id: str, item: dict, item_type: str) -> bool:
        """Dispatch an entity action to the correct method. Returns True if handled.

        The dispatch classification — which method each (action_id,
        item_type) pair routes to, the entity id/name extraction from the
        item, the missing-id early-outs, and the go_to_artist/go_to_album
        target resolution — runs through the ``_entity_action_route``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs each
        arm's calls.
        """
        route = _entity_action_route(action_id, item, item_type)
        if not route["handled"]:
            return False
        handler = route["handler"]
        if handler == "notify":
            self.notify("No ID available", severity="error", timeout=2)
            return True
        if handler == "_play_album":
            await self._play_album(
                route["entity_id"], route["entity_name"], shuffle=route["shuffle"]
            )
        elif handler == "_play_playlist":
            await self._play_playlist(
                route["entity_id"], route["entity_name"], shuffle=route["shuffle"]
            )
        elif handler == "_add_album_to_queue":
            await self._add_album_to_queue(route["entity_id"], route["entity_name"])
        elif handler == "_add_playlist_to_queue":
            await self._add_playlist_to_queue(route["entity_id"], route["entity_name"])
        elif handler == "_add_album_next":
            await self._add_album_next(route["entity_id"], route["entity_name"])
        elif handler == "_add_playlist_next":
            await self._add_playlist_next(route["entity_id"], route["entity_name"])
        elif handler == "_start_artist_radio":
            await self._start_artist_radio(route["entity_id"])
        elif handler == "_start_playlist_radio":
            await self._start_playlist_radio(item)
        elif handler == "_play_artist_top_songs":
            await self._play_artist_top_songs(route["entity_id"])
        elif handler == "_navigate_artist":
            await self.navigate_to("context", context_type="artist", context_id=route["artist_id"])
        elif handler == "_navigate_album":
            await self.navigate_to("context", context_type="album", context_id=route["album_id"])
        elif handler == "_add_album_to_library":
            await self._add_album_to_library(route["entity_id"], route["entity_name"])
        return True

    _PLAYLIST_FIRST_BATCH = 300

    @playback_request
    async def _play_playlist(
        self,
        playlist_id: str,
        name: str,
        *,
        shuffle: bool | None = None,
        order: str | None = None,
    ) -> None:
        if not self.ytmusic:
            return
        try:
            data = await self.ytmusic.get_playlist(
                playlist_id, limit=self._PLAYLIST_FIRST_BATCH, order=order
            )
            if not owns_request(self):
                return
            raw_tracks = data.get("tracks", [])
            tracks = normalize_tracks(raw_tracks)
            if not tracks:
                self.notify("Playlist is empty", severity="warning")
                return
            await self._replace_queue_and_play(
                tracks, entity_id=playlist_id, start_index=None, shuffle=shuffle
            )
            if not owns_request(self):
                return
            suffix = " (shuffled)" if self.queue.shuffle_enabled else ""
            self.notify(f"Playing: {name}{suffix}", timeout=4)
            total_count = data.get("trackCount") or len(raw_tracks)
            if total_count > len(raw_tracks):
                self.run_worker(
                    self._fetch_remaining_for_queue(playlist_id, len(raw_tracks), order=order),
                    name="fetch-remaining-playlist",
                )
        except Exception:
            logger.exception("Failed to load playlist %s", playlist_id)
            if owns_request(self):
                self.notify("Failed to load playlist", severity="error")

    async def _add_playlist_to_queue(self, playlist_id: str, name: str) -> None:
        if not self.ytmusic:
            return
        try:
            data = await self.ytmusic.get_playlist(playlist_id)
            tracks = normalize_tracks(data.get("tracks", []))
            if tracks:
                self._append_to_queue(tracks, name)
            else:
                self.notify("Playlist is empty", severity="warning", timeout=2)
        except Exception:
            logger.debug("Failed to add playlist to queue", exc_info=True)
            self.notify("Failed to add to queue", severity="error", timeout=2)

    async def _add_playlist_next(self, playlist_id: str, name: str) -> None:
        if not self.ytmusic:
            return
        try:
            data = await self.ytmusic.get_playlist(playlist_id)
            tracks = normalize_tracks(data.get("tracks", []))
            if tracks:
                self._play_next_multiple(tracks, name)
            else:
                self.notify("Playlist is empty", severity="warning", timeout=2)
        except Exception:
            logger.debug("Failed to add playlist to play next", exc_info=True)
            self.notify("Failed to play next", severity="error", timeout=2)

    @playback_request
    async def _start_playlist_radio(self, item: dict) -> None:
        """Start radio seeded from a playlist."""
        playlist_id = item.get("playlistId") or item.get("browseId")
        if not playlist_id or not self.ytmusic:
            return

        playlist_name = item.get("title", "playlist")
        self.notify(f"Starting radio from {playlist_name}...", timeout=3)
        try:
            radio_tracks = normalize_tracks(await self.ytmusic.get_playlist_radio(playlist_id))
        except Exception:
            logger.exception("Failed to start playlist radio for %r", playlist_id)
            if owns_request(self):
                self.notify("Failed to start radio", severity="error")
            return

        if not owns_request(self):
            return
        if radio_tracks:
            self.queue.clear()
            adopt_queue(self)
            self.queue.set_radio_tracks(radio_tracks)
            self.queue.radio_seeds = [{"title": f"{playlist_name} Playlist"}]
            self.queue.set_context(playlist_id)
            if self.shuffle_prefs.get(playlist_id) and not self.queue.shuffle_enabled:
                self.queue.toggle_shuffle()
            self._refresh_queue_page()
            self._sync_shuffle_bar()
            first = self.queue.next_track()
            if first:
                await self.play_track(first)
                if owns_request(self):
                    self.notify(f"Playing: Radio from {playlist_name}", timeout=4)
        else:
            self.notify("No radio tracks found", severity="warning", timeout=3)

    @playback_request
    async def _fetch_remaining_for_queue(
        self, playlist_id: str, already_have: int, *, order: str | None = None
    ) -> None:
        """Background fetch remaining tracks and append to the queue."""
        if not self.ytmusic:
            return
        try:
            remaining = await self.ytmusic.get_playlist_remaining(
                playlist_id, already_have, order=order
            )
            if not owns_request(self):
                return
            if remaining:
                tracks = normalize_tracks(remaining)
                self.queue.add_multiple(tracks)
                self._refresh_queue_page()
        except Exception:
            logger.debug("Failed to fetch remaining playlist tracks for queue", exc_info=True)

    def on_selection_changed(self, message: Any) -> None:
        """Relay SelectionChanged messages to the SelectionInfoBar.

        SelectionChanged bubbles up the DOM from descendants (sidebar items,
        TrackTable rows). The bar is a sibling of those widgets — it's NOT
        an ancestor — so the bubble never reaches it directly. The App is
        the common ancestor: catch the message here and push the text to
        the bar by id.
        """
        try:
            from ytm_player.ui.selection_info_bar import SelectionInfoBar

            bar = self.query_one("#selection-info-bar", SelectionInfoBar)
            bar.text = getattr(message, "text", "")
        except Exception:
            logger.debug("Failed to relay SelectionChanged to bar", exc_info=True)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same track-action decision cores; these
# bindings replace the pure-Python transcriptions above when the extension
# is importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead native
# bindings).  The mixin's ``_queue_entry_id`` staticmethod delegates to the
# module-level ``_queue_entry_id`` engine through this same name, so the
# seam steers it too.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        album_browse_id as _native_album_browse_id,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        artist_songs_merge_plan as _native_artist_songs_merge_plan,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        entity_action_route as _native_entity_action_route,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        first_artist_id as _native_first_artist_id,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        in_queue_verdict as _native_in_queue_verdict,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        like_toggle_rating as _native_like_toggle_rating,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        queue_entry_id as _native_queue_entry_id,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        remove_from_queue_plan as _native_remove_from_queue_plan,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_action_route as _native_track_action_route,
    )
except ImportError:
    pass
else:
    _first_artist_id = _native_first_artist_id  # type: ignore[assignment]
    _album_browse_id = _native_album_browse_id  # type: ignore[assignment]
    _queue_entry_id = _native_queue_entry_id  # type: ignore[assignment]
    _track_action_route = _native_track_action_route  # type: ignore[assignment]
    _entity_action_route = _native_entity_action_route  # type: ignore[assignment]
    _like_toggle_rating = _native_like_toggle_rating  # type: ignore[assignment]
    _remove_from_queue_plan = (  # type: ignore[assignment]
        _native_remove_from_queue_plan
    )
    _in_queue_verdict = _native_in_queue_verdict  # type: ignore[assignment]
    _artist_songs_merge_plan = (  # type: ignore[assignment]
        _native_artist_songs_merge_plan
    )
