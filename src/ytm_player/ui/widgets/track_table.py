"""Reusable track listing table widget.

The table's pure rendering logic — the view engine (``_filter_sort_positions``
/ ``_row_values`` / ``_compose_view_rows``: the filter + sort pass and the
per-row cell assembly) and its rendering integration (``_stamp_tracks``:
the load/append stamping pass, ``_mark_status_text``: the marked-rows
status line, ``_selection_label``: the info-bar label,
``_reindex_after_removal``: the post-removal renumbering,
``_sort_state``: the header-click sort toggle, ``_column_keys``: the
composed cells' column keys) — and its mark/range selection engine
(``_visible_run``: the visible-run core, ``_extend_marks``: the
shift-click extend decision, ``_range_marks``: the keyboard-range
marks, ``_toggle_marks``: the toggle's symmetric difference,
``_shift_selection``: the post-removal selection shift,
``_marked_tracks``: the marked-tracks walk, ``_hidden_marked_count``:
the hidden-marks count, ``_carry_selection_keys``: the
occurrence-key carry across a background reload) are implemented by
the Rust extension (``_ytm_player_native``) when it is built and by
the pure-Python transcriptions in this module otherwise; the
bottom-of-module seam rebinds the engines to the native functions
when the extension is importable. The widget below only holds state
and feeds Textual.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import AbstractSet, Any

from textual.events import Click, MouseDown, MouseMove, MouseUp
from textual.geometry import Size
from textual.message import Message
from textual.timer import Timer
from textual.widgets import DataTable
from textual.widgets.data_table import Column, RowKey

from ytm_player.config import Action
from ytm_player.config.settings import get_settings
from ytm_player.ui.selection_info_bar import SelectionChanged
from ytm_player.utils.formatting import extract_artist, extract_duration, format_duration

logger = logging.getLogger(__name__)

_MARK_GLYPH = "✓"

# Sort key per sortable column, shared by the visible sort and by
# ``marked_tracks`` (which orders the whole backing list the same way).
# The pure-Python reference for the view engine below; the native
# override at the bottom of this module replaces it when built.
_SORT_KEYS: dict[str, Callable[[dict], Any]] = {
    "index": lambda t: t.get("_original_index", 0),
    "title": lambda t: (t.get("title") or "").lower(),
    "artist": lambda t: extract_artist(t).lower(),
    "album": lambda t: (t.get("album") or "").lower(),
    "duration": lambda t: extract_duration(t),
}


def _matches_filter(track: dict, query: str) -> bool:
    """Check if a track matches the filter (title + artist + album).

    *query* arrives already lowercased (``apply_filter``'s preprocessing);
    the three fields are lowercased here for the comparison.
    """
    title = (track.get("title") or "").lower()
    artist = extract_artist(track).lower()
    album = (track.get("album") or "").lower()
    return query in title or query in artist or query in album


def _filter_sort_positions(
    tracks: list[dict],
    query: str,
    sort_column: str | None,
    sort_reverse: bool,
) -> list[int]:
    """Compute the visible-row order of *tracks* — the filter, then the sort.

    The view engine behind ``TrackTable._rebuild_view`` and
    ``marked_tracks``: *query* (already stripped and lowercased by the
    caller) keeps the tracks whose lowercased title, artist or album
    contain it; *sort_column* (one of the ``_SORT_KEYS`` names) then
    orders them, stably, with ties kept in input order even when
    *sort_reverse* is set. Returns the positions (indices into *tracks*)
    of the visible rows, in display order.
    """
    if query:
        kept = [i for i, t in enumerate(tracks) if _matches_filter(t, query)]
    else:
        kept = list(range(len(tracks)))
    key_fn = _SORT_KEYS.get(sort_column or "")
    if key_fn is not None:
        kept.sort(key=lambda i: key_fn(tracks[i]), reverse=sort_reverse)
    return kept


def _row_values(
    track: dict,
    index: int,
    show_index: bool,
    show_album: bool,
    marked: AbstractSet[int],
) -> tuple[str, list[str]]:
    """One visible row's ``(row_key, cells)`` pair.

    The pure core of ``TrackTable._add_track_row``: the mark glyph
    (``✓`` when the row's ``_original_index`` — the visible *index* as
    its default — is a member of *marked*), the ``#`` cell
    ``str(orig + 1)`` (only when *show_index*), the BiDi-isolated
    title/artist/album cells (the album only when *show_album*, the
    album itself read through the ``or ""`` truthiness gate), the
    duration text (``format_duration`` when the duration is truthy,
    ``"--:--"`` otherwise) and the ``f"{video_id}_{index}"`` row key
    (the ``video_id`` read with its ``"row_{index}"`` default). The
    ``add_row`` call stays with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_row_values``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    from ytm_player.utils.bidi import render_track_cell

    title = track.get("title", "Unknown")
    artist = extract_artist(track)
    album = track.get("album") or ""
    duration = extract_duration(track)
    orig = track.get("_original_index", index)
    cells: list[str] = [_MARK_GLYPH if orig in marked else " "]
    if show_index:
        cells.append(str(orig + 1))
    cells.append(render_track_cell(title))
    cells.append(render_track_cell(artist))
    if show_album:
        cells.append(render_track_cell(album))
    cells.append(format_duration(duration) if duration else "--:--")
    video_id = track.get("video_id", f"row_{index}")
    return f"{video_id}_{index}", cells


def _compose_view_rows(
    tracks: list[dict],
    query: str,
    sort_column: str | None,
    sort_reverse: bool,
    show_index: bool,
    show_album: bool,
    marked: AbstractSet[int],
) -> tuple[list[int], list[tuple[str, list[str]]]]:
    """The visible rows' ``(positions, (row_key, cells) pairs)``.

    The fused view engine behind every table rebuild —
    ``TrackTable.load_tracks``' row walk and ``_rebuild_view`` +
    ``_reload_sorted``'s chain: the filter + sort order (the
    ``_filter_sort_positions`` engine) fused with each visible row's
    cell assembly (the ``_row_values`` engine, the VISIBLE row number
    as its index). The ``add_row`` calls stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_view_rows``) when it is built;
    this pure-Python composition is the fallback otherwise.
    """
    positions = _filter_sort_positions(tracks, query, sort_column, sort_reverse)
    rows = [
        _row_values(tracks[pos], visible, show_index, show_album, marked)
        for visible, pos in enumerate(positions)
    ]
    return positions, rows


def _stamp_tracks(tracks: list[dict], start: int = 0) -> list[dict]:
    """Shallow copies of *tracks*, each stamped with its load position.

    ``_original_index`` is ``start`` + position — the fresh load stamps
    from 0, the append path from its ``start_idx``. The copies are new
    dicts holding the same value objects; the inputs are untouched.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_stamp_tracks``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    stamped: list[dict] = []
    for i, track in enumerate(tracks, start=start):
        t = dict(track)
        t["_original_index"] = i
        stamped.append(t)
    return stamped


def _mark_status_text(count: int, hidden: int) -> str:
    """The marked-rows status line: "<count> selected" plus the hidden
    count and the bulk-action hint.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_mark_status``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    text = f"{count} selected"
    if hidden:
        text += f" ({hidden} hidden)"
    return text + " · A: add · Esc: clear"


def _selection_label(track: dict) -> str:
    """The selection-info label for the highlighted row: the title and
    artist joined by an em dash (the artist through the shared
    ``extract_artist``, whose ``"Unknown"`` fallback joins like any
    other name).

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_selection_label``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    title = track.get("title", "") or ""
    artist = extract_artist(track) or ""
    return f"{title} — {artist}" if artist else title


def _reindex_after_removal(
    all_tracks: list[dict], tracks: list[dict]
) -> tuple[list[int], list[str]]:
    """Renumber after a removal: re-stamp every backing-list row with its
    new position, and answer the visible list's rebuilt ``_filtered_map``
    plus each visible row's ``#`` cell text (``str(original + 1)``).

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_reindex_after_removal``) when it
    is built; this pure-Python transcription is the fallback otherwise.
    """
    for new_idx, t in enumerate(all_tracks):
        t["_original_index"] = new_idx
    filtered_map = [t["_original_index"] for t in tracks]
    index_cells = [str(t["_original_index"] + 1) for t in tracks]
    return filtered_map, index_cells


def _sort_state(
    column: str, sort_column: str | None, sort_reverse: bool
) -> tuple[str, bool] | None:
    """The header-click sort toggle: ``None`` for a column outside the
    sortable set; the same column flips the direction; a new column
    starts ascending.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_sort_state``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if column not in _SORT_KEYS:
        return None
    if sort_column == column:
        return column, not sort_reverse
    return column, False


def _column_keys(show_index: bool, show_album: bool) -> list[str]:
    """The column keys of the composed cells: ``mark``, then ``index``
    when shown, ``title``/``artist``, ``album`` when shown, then
    ``duration``.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_column_keys``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    keys = ["mark"]
    if show_index:
        keys.append("index")
    keys += ["title", "artist"]
    if show_album:
        keys.append("album")
    keys.append("duration")
    return keys


def _visible_run(filtered_map: Sequence[int], start: int, end: int) -> set[int]:
    """Original indices of the visible rows from *start* to *end*, inclusive.

    The run core behind the shift-click extend (``_extend_to``) and the
    keyboard range marks (``_apply_range``): the two occurrences resolve
    through ``list.index`` and order through ``sorted``, so the run
    reads the same whichever end the gesture started from. A *start*
    the filter hides counts as no start — the run is *end* alone, so
    an invisible row is never marked through a run.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_visible_run``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    try:
        lo, hi = sorted((filtered_map.index(start), filtered_map.index(end)))
    except ValueError:
        return {end}
    return set(filtered_map[lo : hi + 1])


def _extend_marks(
    anchor: int | None,
    filtered_map: Sequence[int],
    marked: AbstractSet[int],
    original: int,
) -> tuple[int | None, set[int]]:
    """The shift-click extend decision: ``(new_anchor, new_marks)``.

    With no anchor, or an anchor the filter hides (the ``in
    filtered_map`` list containment), the row is marked and becomes the
    anchor — a run never reaches a row that isn't shown — while an
    anchored view marks the visible run from the anchor to the row
    (existing marks outside the run stay). The repaint (``_set_marks``)
    stays with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_extend_marks``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if anchor is None or anchor not in filtered_map:
        return original, set(marked) | {original}
    return anchor, set(marked) | _visible_run(filtered_map, anchor, original)


def _range_marks(
    anchor: int | None,
    cursor: int | None,
    filtered_map: Sequence[int],
    range_base: AbstractSet[int],
) -> set[int] | None:
    """The keyboard-range marks: the baseline plus the anchor-to-cursor run.

    Range mode's decision core behind ``_apply_range`` — ``None`` when
    there is no anchor (the caller leaves the marks alone), otherwise
    the range baseline unioned with the run from the anchor to the
    cursor, an empty run when the cursor is off the table. The
    generation bump and the repaint stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_range_marks``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if anchor is None:
        return None
    run: set[int] = set() if cursor is None else _visible_run(filtered_map, anchor, cursor)
    return set(range_base) | run


def _toggle_marks(marked: AbstractSet[int], original: int) -> set[int]:
    """The toggle's marked set: the marks with *original*'s membership flipped.

    ``_toggle_mark``'s symmetric difference — the anchor assignment and
    the repaint stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_toggle_marks``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return set(marked) ^ {original}


def _shift_selection(
    removed: int,
    marked: AbstractSet[int],
    range_base: AbstractSet[int],
    anchor: int | None,
    range_mode: bool,
) -> tuple[set[int], frozenset[int], int | None, bool, bool]:
    """The post-removal selection shift: ``(marks, base, anchor, range_mode, bump)``.

    When the occurrence *removed* leaves the list, every mark above it
    slides down one and the removed mark itself goes (the *bump* flag
    answers whether the removed occurrence was marked, so the caller
    advances the selection generation exactly when the marked set
    actually shrank). The range baseline shifts the same way; the
    anchor dies with its occurrence — an un-anchored range mode drops
    out and empties the baseline. The repaint stays with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_shift_selection``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """

    def shift(original: int) -> int:
        return original - 1 if original > removed else original

    bump = removed in marked
    new_marked = {shift(i) for i in marked if i != removed}
    new_base = frozenset(shift(i) for i in range_base if i != removed)
    new_anchor = None if anchor is None else (None if anchor == removed else shift(anchor))
    new_range_mode = range_mode
    if new_anchor is None and range_mode:
        new_range_mode = False
        new_base = frozenset()
    return new_marked, new_base, new_anchor, new_range_mode, bump


def _marked_tracks(
    all_tracks: list[dict],
    marked: AbstractSet[int],
    sort_column: str | None,
    sort_reverse: bool,
) -> list[dict]:
    """The marked tracks in the displayed order, hidden ones included.

    ``marked_tracks``'s walk: under a sort the whole backing list is
    ordered by that sort (the ``_filter_sort_positions`` engine with
    the empty query), so a marked track the filter hides lands where it
    would show without the filter; otherwise the load order applies.
    The answer holds the backing list's own track objects, no copies.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_marked_tracks``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if not marked:
        return []
    positions = _filter_sort_positions(all_tracks, "", sort_column, sort_reverse)
    return [all_tracks[i] for i in positions if all_tracks[i]["_original_index"] in marked]


def _hidden_marked_count(marked: AbstractSet[int], filtered_map: Sequence[int]) -> int:
    """The marked rows the active filter currently hides.

    ``hidden_marked_count``'s count: the marked originals absent from
    the visible map.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_hidden_marked_count``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    visible = set(filtered_map)
    return sum(1 for i in marked if i not in visible)


def _carry_selection_keys(
    old_keys: Sequence[Hashable],
    new_keys: Sequence[Hashable],
    marked: Iterable[int],
    range_base: Iterable[int],
    anchor: int | None,
    cursor: int | None,
) -> tuple[set[int], frozenset[int], int | None, int | None, bool]:
    """Resolve the selection's occurrence keys across a reload.

    ``refresh_tracks``'s carry core: which occurrences of the new list
    the marks, the range baseline, the anchor and the cursor carried
    over to. The old list's keys count through a ``Counter``, and
    ``key_at`` names an occurrence by its key only when that key is
    unique in the old list — a duplicated key resolves nothing, so a
    mark on it is dropped rather than moved to another row. The new
    list's keys count the same way, and ``position`` maps each key
    unique in the new list to its index; ``resolve`` then carries each
    candidate key to its new position, skipping keys missing from or
    duplicated in the new list.

    Answers ``(marked, range_base, anchor, cursor, dropped)`` — the
    marked set and range baseline resolved onto the new list (the
    baseline as a frozenset), the anchor and cursor as single
    positions or ``None``, and *dropped*, true when any marked key
    failed to resolve — the caller's signal to advance the selection
    generation. The reload itself stays with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.track_table_carry_selection_keys``) when it
    is built; this pure-Python transcription is the fallback otherwise.
    """
    old_counts = Counter(old_keys)

    def key_at(index: int) -> Hashable | None:
        key = old_keys[index]
        return key if old_counts[key] == 1 else None

    marked_keys = [key_at(i) for i in marked]
    base_keys = [key_at(i) for i in range_base]
    anchor_key = key_at(anchor) if anchor is not None else None
    cursor_key = key_at(cursor) if cursor is not None else None

    new_counts = Counter(new_keys)
    position = {k: i for i, k in enumerate(new_keys) if new_counts[k] == 1}

    def resolve(candidates: Iterable[Hashable | None]) -> set[int]:
        return {position[k] for k in candidates if k is not None and k in position}

    resolved = resolve(marked_keys)
    dropped = len(resolved) < len(marked_keys)
    new_base = frozenset(resolve(base_keys))
    new_anchor = next(iter(resolve([anchor_key])), None)
    new_cursor = next(iter(resolve([cursor_key])), None)
    return resolved, new_base, new_anchor, new_cursor, dropped


class TrackTable(DataTable):
    """A DataTable subclass for displaying lists of tracks.

    Columns: mark, #, Title, Artist, Album, Duration.

    Tracks are stored as dicts matching the queue/search result format:
        {
            "video_id": str,
            "title": str,
            "artist": str,
            "album": str | None,
            "duration": int | None,       # seconds
            "duration_seconds": int | None,
            ...
        }

    Marks: rows can be marked for a bulk action (``v``/``V``/Escape, Ctrl-
    click, Shift-click, or a click on the mark column). A mark belongs to
    one occurrence — the row's position in the loaded list — not to a video
    ID, so two rows of the same track are marked independently. Marks live
    until the table is reloaded (``load_tracks``), cleared, or the page is
    left; ``refresh_tracks`` carries them across a background reload. While
    any row is marked the table draws its own status line underneath.
    """

    DEFAULT_CSS = """
    TrackTable {
        height: 1fr;
        width: 1fr;
    }
    TrackTable > .datatable--cursor {
        background: $selected-item;
    }
    TrackTable.-marked {
        border-bottom: hkey $accent;
        border-subtitle-align: left;
        border-subtitle-color: $text;
    }
    """

    class TrackSelected(Message):
        """Emitted when a track row is activated (Enter key)."""

        def __init__(self, track: dict, index: int) -> None:
            super().__init__()
            self.track = track
            self.index = index

    class TrackRightClicked(Message):
        """Emitted when a track row is right-clicked (title or non-specific column).

        *index* is the visible row and *track* the table's own copy of it
        (unlike ``TrackSelected``, whose index is the load-order one). The
        row's occurrence is snapshotted here, when the message is created:
        ``occurrence_key`` is the key the row was loaded with (see
        ``occurrence_key()``) and ``from_queue`` says whether that key is a
        queue entry id (``queue_entry_keys`` on the table). A handler runs
        later, possibly after the table re-rendered or was removed, so it
        must use these and not look the row up again. ``control`` is the table.
        """

        def __init__(self, table: TrackTable, track: dict, index: int) -> None:
            super().__init__()
            self.table = table
            self.track = track
            self.index = index
            original = track.get("_original_index")
            self.occurrence_key: Hashable | None = (
                table.occurrence_key(original) if isinstance(original, int) else None
            )
            self.from_queue: bool = table.queue_entry_keys

        @property
        def control(self) -> TrackTable:
            return self.table

    class ArtistRightClicked(Message):
        """Emitted when the Artist column of a track row is right-clicked."""

        def __init__(self, track: dict, index: int) -> None:
            super().__init__()
            self.track = track
            self.index = index

    class AlbumRightClicked(Message):
        """Emitted when the Album column of a track row is right-clicked."""

        def __init__(self, track: dict, index: int) -> None:
            super().__init__()
            self.track = track
            self.index = index

    class TrackHighlighted(Message):
        """Emitted when the cursor moves to a different row."""

        def __init__(self, track: dict | None, index: int) -> None:
            super().__init__()
            self.track = track
            self.index = index

    class FilterRequested(Message):
        """Emitted when the user presses / to start filtering."""

    class FilterClosed(Message):
        """Emitted when the filter is dismissed."""

    def __init__(
        self,
        *,
        show_index: bool = True,
        show_album: bool = True,
        zebra_stripes: bool = True,
        queue_entry_keys: bool = False,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            cursor_type="row",
            zebra_stripes=zebra_stripes,
            cursor_foreground_priority="renderable",
            name=name,
            id=id,
            classes=classes,
        )
        self._show_index = show_index
        self._show_album = show_album
        # True on the Queue page only: its rows are loaded with QueueManager
        # entry ids as keys, so a row names one queue occurrence.
        self.queue_entry_keys = queue_entry_keys
        self._all_tracks: list[dict] = []
        self._tracks: list[dict] = []
        self._filtered_map: list[int] = []
        self._row_keys: list[RowKey] = []
        self._playing_video_id: str | None = None
        self._playing_index: int | None = None
        self._right_clicked: bool = False
        # Armed on right-click; only escalated to _suppress_select_on_refocus
        # once the table actually blurs (a modal took focus). Right-clicks
        # whose handler early-returns without opening a popup never blur, so
        # they no longer leave the suppress flag stale.
        self._refill_pending: bool = False
        self._popup_pending: bool = False
        self._suppress_select_on_refocus: bool = False
        self._sort_column: str | None = None
        self._sort_reverse: bool = False
        self._filter_text: str = ""
        self._filter_active: bool = False
        self._filter_timer: Timer | None = None
        # Column resize drag state.
        self._resize_col: Column | None = None
        self._resize_start_x: int = 0
        self._resize_start_width: int = 0
        self._title_manual_width: bool = False
        # Selection state. Every index here is an ``_original_index`` into
        # ``_all_tracks`` (one occurrence), never a visible row number.
        self._marked: set[int] = set()
        self._anchor: int | None = None
        self._range_mode: bool = False
        self._range_base: frozenset[int] = frozenset()
        # Bumped whenever the marked set changes or the list is replaced;
        # a bulk action captures it when it starts and only clears the
        # marks on completion if it still matches.
        self._selection_generation: int = 0
        # One key per loaded track (load order) naming its occurrence for
        # ``refresh_tracks``; None when the list was loaded without keys.
        self._occurrence_keys: list[Hashable] | None = None
        # Set up columns at construction time, not on_mount. Otherwise a
        # caller that mounts the table and immediately calls load_tracks()
        # synchronously (e.g. context._build_artist's nested-mount chain)
        # hits add_row() before on_mount runs, and add_row raises because
        # there are 0 columns.
        self._setup_columns()

    @property
    def tracks(self) -> list[dict]:
        """Return ALL tracks regardless of filter (for queue integration)."""
        return list(self._all_tracks)

    @property
    def visible_tracks(self) -> list[dict]:
        """Return only visible (possibly filtered) tracks."""
        return list(self._tracks)

    @property
    def track_count(self) -> int:
        return len(self._all_tracks)

    @property
    def selected_track(self) -> dict | None:
        """Return the track dict for the currently highlighted row."""
        if self.cursor_row is not None and 0 <= self.cursor_row < len(self._tracks):
            return self._tracks[self.cursor_row]
        return None

    @property
    def selected_original_index(self) -> int | None:
        """Return the highlighted row's index in the originally loaded order.

        ``cursor_row`` is a visible-row index that diverges from the load
        order once a sort or filter is active. Callers that mutate the
        backing collection (e.g. queue remove/move) must use this mapped
        index — the same mapping ``TrackSelected`` messages carry.
        """
        if self.cursor_row is not None and 0 <= self.cursor_row < len(self._tracks):
            return self._filtered_map[self.cursor_row] if self._filtered_map else self.cursor_row
        return None

    @property
    def marked_count(self) -> int:
        """Number of marked rows, hidden (filtered-out) ones included."""
        return len(self._marked)

    @property
    def hidden_marked_count(self) -> int:
        """Marked rows the active filter currently hides."""
        # The count — the ``_hidden_marked_count`` engine (native when
        # the Rust extension is built, the pure-Python transcription
        # otherwise).
        return _hidden_marked_count(self._marked, self._filtered_map)

    @property
    def selection_generation(self) -> int:
        """Changes whenever the marked set changes or the list is replaced."""
        return self._selection_generation

    def occurrence_key(self, original_index: int) -> Hashable | None:
        """The key naming occurrence *original_index*, or None.

        Keys come from the ``keys`` the list was loaded with (queue entry ids
        on the Queue page); None when the list was loaded without keys or the
        index is out of range.
        """
        keys = self._occurrence_keys
        if keys is None or not 0 <= original_index < len(keys):
            return None
        return keys[original_index]

    def marked_tracks(self) -> list[dict]:
        """Marked tracks in the displayed order, hidden ones included.

        Under a sort the whole backing list is ordered by that sort, so a
        marked track the filter hides lands where it would show without
        the filter; otherwise the load order applies — the
        ``_marked_tracks`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise).
        """
        return _marked_tracks(self._all_tracks, self._marked, self._sort_column, self._sort_reverse)

    # -- Setup ------------------------------------------------------------

    def on_mount(self) -> None:
        # Pick up the currently-playing video_id from the app's player so
        # the playing-row highlight survives navigating away and back.
        try:
            player = getattr(self.app, "player", None)
            current = player.current_track if player else None
            video_id = current.get("video_id", "") if current else ""
            if video_id:
                self._playing_video_id = video_id
                # _highlight_playing runs after columns/rows are populated;
                # load_tracks will trigger it. If the table is already empty,
                # this is a no-op until tracks land.
                self._highlight_playing()
        except Exception:
            logger.debug("Failed to pick up current playing track on mount", exc_info=True)

    def _setup_columns(self) -> None:
        """Add the standard track table columns."""
        ui = get_settings().ui

        def w(v: int) -> int | None:
            return v if v > 0 else None

        # Mark column first: one cell, blank or a check mark.
        self.add_column(" ", width=1, key="mark")
        if self._show_index:
            self.add_column("#", width=w(ui.col_index), key="index")
        self.add_column("Title", width=w(ui.col_title), key="title")
        self.add_column("Artist", width=w(ui.col_artist), key="artist")
        if self._show_album:
            self.add_column("Album", width=w(ui.col_album), key="album")
        self.add_column("Duration", width=w(ui.col_duration), key="duration")

    # -- Data loading -----------------------------------------------------

    def load_tracks(self, tracks: list[dict], *, keys: Sequence[Hashable] | None = None) -> None:
        """Replace the table contents with a new list of tracks.

        This is the deliberate replacement: the sort, filter and every mark
        go. *keys*, when given, name each track's occurrence (one per
        track, in order) so a later ``refresh_tracks`` can carry marks
        across a background reload of the same list.
        """
        if keys is not None and len(keys) != len(tracks):
            raise ValueError("keys must have one entry per track")
        self._occurrence_keys = list(keys) if keys is not None else None
        self._reset_selection()
        self.clear()
        # Stamp each track with its original playlist position — the
        # ``_stamp_tracks`` engine (native when the Rust extension is
        # built, the pure-Python transcription otherwise).
        self._all_tracks = _stamp_tracks(tracks)
        self._tracks = list(self._all_tracks)
        self._filtered_map = list(range(len(self._all_tracks)))
        self._row_keys = []
        self._playing_index = None
        self._sort_column = None
        self._sort_reverse = False
        self._filter_text = ""
        self._filter_active = False

        # The fresh load's rows — identity order, no marks yet — in one
        # engine pass (the filter+sort walk runs trivially: an empty
        # query and no sort keep load order).
        _, rows = _compose_view_rows(
            self._all_tracks, "", None, False, self._show_index, self._show_album, self._marked
        )
        self._append_rows(rows)

        # Reflow the title column now that rows (with row labels) exist
        # — the row-label column eats ~3 cells, so the initial column
        # widths from settings would push the rightmost column off-screen.
        self._fill_title_column()
        self._invalidate_table()

        self._highlight_playing()
        self._update_mark_status()

    def refresh_tracks(self, tracks: list[dict], keys: Sequence[Hashable]) -> None:
        """Re-render the same list after a background change, keeping the view.

        For reloads the user didn't ask for (a play landing in Recently
        Played, a queue entry added from elsewhere): the active sort and
        filter, the cursor, the marks, the range anchor and the range
        baseline all carry over to the occurrences that survive. *keys*
        names each row's occurrence, one per track and unique, in the same
        scheme the previous load used — queue entry ids, or video IDs where
        the list holds each track once. A key that is missing, or that
        appears more than once, resolves nothing: the mark on it is dropped
        rather than moved to another row.

        Dropping a marked occurrence, or ending up with a different marked
        set, advances the selection generation so a bulk action started
        before the refresh won't clear marks it didn't submit. A refresh
        that keeps every marked occurrence leaves the generation alone.
        Use ``load_tracks`` for a deliberate replacement.
        """
        if len(keys) != len(tracks):
            raise ValueError("keys must have one entry per track")
        old_keys = self._occurrence_keys
        if old_keys is None or len(old_keys) != len(self._all_tracks):
            self.load_tracks(tracks, keys=keys)
            return

        cursor_index = self.selected_original_index
        range_mode = self._range_mode
        sort_column, sort_reverse = self._sort_column, self._sort_reverse
        filter_text, filter_active = self._filter_text, self._filter_active
        generation = self._selection_generation

        # The occurrence-key carry — the ``_carry_selection_keys``
        # engine (native when the Rust extension is built, the
        # pure-Python transcription otherwise): the marks, the range
        # baseline, the anchor and the cursor resolved onto the new
        # list, read before the reload replaces the selection state.
        resolved, base, new_anchor, new_cursor, dropped = _carry_selection_keys(
            old_keys, keys, self._marked, self._range_base, self._anchor, cursor_index
        )

        self.load_tracks(tracks, keys=keys)

        # The view first, so the rows exist in their final order before
        # marks are painted and the cursor is placed.
        self._filter_text, self._filter_active = filter_text, filter_active
        self._sort_column, self._sort_reverse = sort_column, sort_reverse
        if filter_text or sort_column is not None:
            self._rebuild_view()

        self._range_base = base
        self._anchor = new_anchor
        self._range_mode = range_mode and new_anchor is not None
        self._set_marks(resolved, bump=False)

        if new_cursor is not None:
            try:
                self.move_cursor(row=self._filtered_map.index(new_cursor))
            except ValueError:
                pass
        if self._range_mode:
            self._apply_range(bump=False)

        same_selection = not dropped and self._marked == resolved
        self._selection_generation = generation if same_selection else generation + 1
        self._update_mark_status()

    def append_tracks(self, tracks: list[dict], *, keys: Sequence[Hashable] | None = None) -> None:
        """Append additional tracks without clearing existing ones.

        The new rows join the view as it is: filtered like the rest and, under
        an active sort, placed where the sort puts them (the whole view is
        rebuilt then, with the cursor kept on its occurrence). Marks are
        untouched either way. A keyed table stays keyed only when *keys*
        arrives with one entry per track; otherwise the keys are dropped and
        the next ``refresh_tracks`` reloads instead of carrying marks over.
        """
        if self._occurrence_keys is not None:
            if keys is not None and len(keys) == len(tracks):
                self._occurrence_keys.extend(keys)
            else:
                self._occurrence_keys = None
        start_idx = len(self._all_tracks)
        if self._sort_column is not None:
            self._all_tracks.extend(_stamp_tracks(tracks, start=start_idx))
            current = self.selected_original_index
            self._rebuild_view()
            self._place_cursor_on(current)
            self._fill_title_column()
            self._invalidate_table()
            return
        # The appended rows join the view as it is: filtered like the
        # rest, each stamped copy rendered through the row engine.
        for t in _stamp_tracks(tracks, start=start_idx):
            self._all_tracks.append(t)
            # If filter is active, only add matching tracks to visible table.
            if self._filter_active and self._filter_text:
                if not _matches_filter(t, self._filter_text):
                    continue
            self._tracks.append(t)
            self._filtered_map.append(t["_original_index"])
            row_key = self._add_track_row(len(self._tracks) - 1, t)
            self._row_keys.append(row_key)
        # Same reflow as load_tracks — the row-label column may have been
        # 0-width if append_tracks fires before any rows existed.
        self._fill_title_column()
        self._invalidate_table()

    def remove_track(self, video_id: str, set_video_id: str = "") -> bool:
        """Remove a track from the table.

        When *set_video_id* is provided it is used as the primary key so that
        duplicate tracks (same ``video_id``, different ``setVideoId``) are
        removed correctly.  Falls back to matching by ``video_id`` alone when
        *set_video_id* is empty.

        Returns ``True`` if the track was found and removed.
        """

        def _matches(t: dict) -> bool:
            if set_video_id:
                return t.get("setVideoId") == set_video_id
            return t.get("video_id") == video_id

        # Find in visible tracks first (handles filtered view).
        for visible_idx, track in enumerate(self._tracks):
            if _matches(track):
                # Remove from DataTable.
                row_key = self._row_keys[visible_idx]
                try:
                    self.remove_row(row_key)
                except Exception:
                    logger.exception("Failed to remove row %r from table", row_key)

                # Remove from visible list.
                self._tracks.pop(visible_idx)
                self._row_keys.pop(visible_idx)

                # Remove from all tracks.
                for all_idx, t in enumerate(self._all_tracks):
                    if _matches(t):
                        self._all_tracks.pop(all_idx)
                        if self._occurrence_keys is not None and all_idx < len(
                            self._occurrence_keys
                        ):
                            self._occurrence_keys.pop(all_idx)
                        self._shift_selection_after_removal(all_idx)
                        break

                # Re-number all remaining tracks so the # column stays
                # contiguous after the removal, and rebuild the visible
                # map + the # cells — the ``_reindex_after_removal``
                # engine (native when the Rust extension is built, the
                # pure-Python transcription otherwise).
                self._filtered_map, index_cells = _reindex_after_removal(
                    self._all_tracks, self._tracks
                )

                # Refresh the # cell for every visible row that shifted.
                for rk, cell_value in zip(self._row_keys, index_cells):
                    try:
                        self.update_cell(rk, "index", cell_value)
                    except Exception:
                        pass

                self._fill_title_column()
                self._invalidate_table()
                self._update_mark_status()
                return True
        return False

    def _append_rows(self, rows: Sequence[tuple[str, list[str]]]) -> None:
        """Add pre-composed ``(row_key, cells)`` rows, tracking their keys.

        The row-label reservation (``label=" "`` on every row so
        Textual reserves the column once any row has a label —
        ``_highlight_playing`` mutates the slot to show ▶ on the
        playing row) lives here, next to the ``add_row`` calls.
        """
        for row_key, cells in rows:
            self._row_keys.append(self.add_row(*cells, key=row_key, label=" "))

    def _add_track_row(self, index: int, track: dict) -> RowKey:
        """Add a single track as a row in the table.

        The cell assembly — the mark glyph, the position number, the
        BiDi-isolated text cells and the duration — is resolved by the
        ``_row_values`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise), fused with the
        FSI...PDI isolation so RTL cells cannot bleed into adjacent
        columns.
        """
        row_key, cells = _row_values(track, index, self._show_index, self._show_album, self._marked)
        return self.add_row(*cells, key=row_key, label=" ")

    # -- Playing state ----------------------------------------------------

    def set_playing(self, video_id: str | None) -> None:
        """Mark a track as currently playing (updates visual indicator)."""
        self._playing_video_id = video_id
        self._highlight_playing()

    def _jump_to_current(self) -> None:
        """Move cursor to the currently playing track if visible."""
        if not self._tracks:
            return
        # Prefer the app's current track over our cached _playing_video_id
        # because a freshly-mounted page won't have received set_playing yet.
        video_id = self._playing_video_id
        try:
            queue = self.app.queue  # type: ignore[attr-defined]
            current = queue.current_track if queue else None
            if current and current.get("video_id"):
                video_id = current["video_id"]
        except Exception:
            pass
        if not video_id:
            return
        for i, track in enumerate(self._tracks):
            if track.get("video_id") == video_id:
                self.move_cursor(row=i)
                return

    def _highlight_playing(self) -> None:
        """Style the now-playing row across its full width.

        Only touches the previously-playing and newly-playing rows
        instead of iterating every row. Every cell of the playing row
        gets bold + accent-color styling so the row stays visually
        distinct even when the cursor is elsewhere \u2014 without competing
        with the cursor-row CSS background.
        """
        if not self._show_index:
            return

        from rich.text import Text

        # Find the new playing index by matching video_id.
        new_index: int | None = None
        if self._playing_video_id is not None:
            for i, track in enumerate(self._tracks):
                if track.get("video_id") == self._playing_video_id:
                    new_index = i
                    break

        old_index = self._playing_index

        # Nothing changed -- skip the update.
        if old_index == new_index:
            return

        def _plain_cells(track: dict, row_index: int) -> dict[str, Any]:
            """Restore-cells: original (un-styled) values for a row.

            The assembly runs through the ``_row_values`` engine —
            one fused call (native when the Rust extension is built,
            the pure-Python transcription otherwise) instead of the
            per-field walk, its positional cells mapped back onto the
            column keys in the table's own column order (the
            ``_column_keys`` engine).
            """
            _, cells = _row_values(
                track, row_index, self._show_index, self._show_album, self._marked
            )
            return dict(zip(_column_keys(self._show_index, self._show_album), cells))

        def _styled_cells(track: dict, row_index: int, style: str) -> dict[str, Any]:
            """Active-cells: same data values wrapped in Rich Text with *style*.

            Unlike before, we do NOT overload the # column with a play
            glyph — the playing indicator now lives in Textual's row-label
            column (left of #). The # column keeps its original number.
            The values themselves come from the ``_row_values`` engine
            via ``_plain_cells``, so the styled row shows exactly the
            cells a rebuild would compose for it.
            """
            plain = _plain_cells(track, row_index)
            return {key: Text(value, style=style) for key, value in plain.items()}

        def _set_row_label(row_key: RowKey, label_text: Text) -> None:
            """Mutate Textual's per-row label slot. Internal access
            wrapped so any breaking change degrades to a debug log."""
            try:
                row = self.rows[row_key]
                row.label = label_text
                self._update_count += 1
                self.refresh()
            except Exception:
                logger.debug("Failed to set row label", exc_info=True)

        # Resolve the theme's text color for the ▶ glyph. Normalize to
        # #rrggbb so Rich always parses (Textual may emit rgb(...) form).
        from textual.color import Color

        from ytm_player.ui.theme import get_theme

        # Restore the old row: plain data cells + blank label.
        if old_index is not None and old_index < len(self._row_keys):
            row_key = self._row_keys[old_index]
            try:
                cells = _plain_cells(self._tracks[old_index], old_index)
                for col_key, value in cells.items():
                    self.update_cell(row_key, col_key, value)
            except Exception:
                logger.debug("Failed to restore row %d cells", old_index, exc_info=True)
            _set_row_label(row_key, Text(" "))

        # Mark the new row: bold (no color) on data cells + ▶ label.
        # Why no color: the user's theme can have $primary close to
        # $selected-item (the cursor bg), so any colored foreground on the
        # playing row's data cells produces unreadable monochrome blocks
        # when the cursor lands on it. The ▶ glyph in the row-label column
        # is the unambiguous primary signal — it lives in its own render
        # path so it's always visible. Bold on the data cells is the
        # secondary cue for at-a-glance recognition without color clash.
        if new_index is not None and new_index < len(self._row_keys):
            row_key = self._row_keys[new_index]
            try:
                cells = _styled_cells(self._tracks[new_index], new_index, "bold")
                for col_key, value in cells.items():
                    self.update_cell(row_key, col_key, value)
            except Exception:
                logger.debug("Failed to style row %d cells", new_index, exc_info=True)
            try:
                text_hex = Color.parse(get_theme().text or "#ffffff").hex
            except Exception:
                text_hex = "#ffffff"
            _set_row_label(row_key, Text("▶", style=f"bold {text_hex}"))

        self._playing_index = new_index

    # -- Marks ------------------------------------------------------------

    def clear_marks(self) -> None:
        """Drop every mark, the anchor and range mode."""
        self._range_mode = False
        self._range_base = frozenset()
        self._anchor = None
        self._set_marks(set())

    def end_range_mode(self) -> None:
        """Leave range mode keeping the marks as they are; later movement won't extend them."""
        self._range_mode = False
        self._range_base = frozenset()

    def _reset_selection(self) -> None:
        """Forget the selection without repainting (the rows are being rebuilt)."""
        self._marked = set()
        self._anchor = None
        self._range_mode = False
        self._range_base = frozenset()
        self._selection_generation += 1

    def _mark_glyph(self, original: int) -> str:
        return _MARK_GLYPH if original in self._marked else " "

    def _set_marks(self, marks: set[int], *, bump: bool = True) -> None:
        """Make *marks* the marked set, repainting only the rows that changed."""
        changed = self._marked ^ marks
        if not changed:
            return
        self._marked = set(marks)
        if bump:
            self._selection_generation += 1
        self._paint_marks(changed)
        self._update_mark_status()

    def _paint_marks(self, originals: Iterable[int]) -> None:
        """Rewrite the mark cell of each visible row among *originals*."""
        row_of = {orig: row for row, orig in enumerate(self._filtered_map)}
        for original in originals:
            row = row_of.get(original)
            if row is None or row >= len(self._row_keys):
                continue
            try:
                self.update_cell(self._row_keys[row], "mark", self._mark_glyph(original))
            except Exception:
                logger.debug("Failed to paint mark on row %d", row, exc_info=True)

    def _toggle_mark(self, original: int) -> None:
        self._anchor = original
        # The symmetric difference — the ``_toggle_marks`` engine
        # (native when the Rust extension is built, the pure-Python
        # transcription otherwise).
        self._set_marks(_toggle_marks(self._marked, original))

    def _extend_to(self, original: int) -> None:
        """Shift-click: mark the displayed run from the anchor to *original*.

        Existing marks outside the run stay. With no anchor, or an anchor
        the filter hides, the row is marked and becomes the anchor — a run
        never reaches a row that isn't shown. The ``(anchor, marks)``
        decision — the ``_extend_marks`` engine (native when the Rust
        extension is built, the pure-Python transcription otherwise).
        """
        self._anchor, marks = _extend_marks(
            self._anchor, self._filtered_map, self._marked, original
        )
        self._set_marks(marks)

    def _apply_range(self, *, bump: bool = True) -> None:
        """Range mode: marks are the baseline plus the anchor-to-cursor run.

        The decision — the ``_range_marks`` engine (native when the Rust
        extension is built, the pure-Python transcription otherwise);
        ``None`` for a range without an anchor leaves the marks alone.
        """
        if self._anchor is None:
            return
        marks = _range_marks(
            self._anchor, self.selected_original_index, self._filtered_map, self._range_base
        )
        if marks is not None:
            self._set_marks(marks, bump=bump)

    def _shift_selection_after_removal(self, removed: int) -> None:
        """Keep marks on their rows when the occurrence *removed* leaves the list.

        The shift — the ``_shift_selection`` engine (native when the
        Rust extension is built, the pure-Python transcription
        otherwise): the marks and the range baseline slide down past
        the removed occurrence, the anchor dies with its occurrence,
        and the *bump* flag advances the selection generation exactly
        when the removed occurrence was marked.
        """
        marked, base, anchor, range_mode, bump = _shift_selection(
            removed, self._marked, self._range_base, self._anchor, self._range_mode
        )
        if bump:
            self._selection_generation += 1
        self._marked = marked
        self._range_base = base
        self._anchor = anchor
        self._range_mode = range_mode

    def _update_mark_status(self) -> None:
        """Show the count on the table's own bottom line while anything is marked.

        The line is the table's border subtitle on a border that only
        exists while the ``-marked`` class is set, so it costs a row only
        then and doesn't depend on the optional selection-info bar.
        """
        count = len(self._marked)
        if count == 0:
            self.border_subtitle = ""
            self.remove_class("-marked")
            return
        # The status line — the ``_mark_status_text`` engine (native
        # when the Rust extension is built, the pure-Python
        # transcription otherwise).
        self.border_subtitle = _mark_status_text(count, self.hidden_marked_count)
        self.add_class("-marked")

    # -- Column resize (drag header border) ------------------------------

    def _invalidate_table(self) -> None:
        """Clear render caches and update virtual size after column changes."""
        if hasattr(self, "_clear_caches"):
            self._clear_caches()
        # Recalculate virtual_size so the scrollbar reflects new widths.
        try:
            data_width = sum(col.get_render_width(self) for col in self.columns.values())
            label_w = (
                self._row_label_column_width if hasattr(self, "_row_label_column_width") else 0
            )
            total_width = data_width + label_w
            header_h = self.header_height if self.show_header else 0
            self.virtual_size = Size(total_width, self._total_row_height + header_h)
        except Exception:
            pass
        self.refresh()
        # Row-label width, auto column widths and scrollbar visibility are only
        # correct once the layout has settled, so re-fill once after the refresh.
        if not self._refill_pending:
            self._refill_pending = True
            self.call_after_refresh(self._refill_after_layout)

    def _refill_after_layout(self) -> None:
        """Re-run the title fill against the settled layout (see above).

        ``_refill_pending`` stays set across the ``_invalidate_table`` call so
        the pass doesn't re-arm itself into a refresh loop.
        """
        self._fill_title_column()
        self._invalidate_table()
        self._refill_pending = False

    def _fill_title_column(self) -> None:
        """Expand the Title column to fill any remaining table width."""
        if self._resize_col is not None or self._title_manual_width:
            return
        # Not self.size.width: that still counts the vertical scrollbar's cells.
        width = self.scrollable_content_region.width
        if width == 0:
            return
        title_col = next((c for c in self.ordered_columns if c.key == "title"), None)
        if title_col is None:
            return

        available = width - self._row_label_column_width
        used = sum(col.get_render_width(self) for col in self.ordered_columns if col.key != "title")
        remaining = available - used - 2 * self.cell_padding
        if remaining > 10:
            title_col.width = remaining
            title_col.auto_width = False

    def _column_at_edge(self, x: int) -> Column | None:
        """Return the Column whose right edge is near *x*, or None."""
        edge = self._row_label_column_width
        for col in self.ordered_columns:
            edge += col.get_render_width(self)
            if abs(x - edge) <= 1:
                return col
        return None

    def _column_at_x(self, x: int) -> Column | None:
        """Return the Column whose body contains *x*, or None."""
        edge = self._row_label_column_width
        for col in self.ordered_columns:
            w = col.get_render_width(self)
            if edge <= x < edge + w:
                return col
            edge += w
        return None

    _SORTABLE_KEYS = {"index", "title", "artist", "album", "duration"}

    def on_mouse_down(self, event: MouseDown) -> None:
        """Start column resize on header edge drag."""
        if event.button != 1:
            return
        if event.y != 0 or not self.show_header:
            return
        scroll_x = event.x + int(self.scroll_x)
        col = self._column_at_edge(scroll_x)
        if col is not None:
            event.stop()
            event.prevent_default()
            self._resize_col = col
            self._resize_start_x = event.screen_x
            self._resize_start_width = col.get_render_width(self)
            self.capture_mouse()

    def on_mouse_move(self, event: MouseMove) -> None:
        """Resize column while dragging."""
        if self._resize_col is None:
            return
        event.stop()
        self.suppress_click()
        delta = event.screen_x - self._resize_start_x
        padding = 2 * self.cell_padding
        new_width = max(3, self._resize_start_width + delta - padding)
        self._resize_col.width = new_width
        self._resize_col.auto_width = False
        self._fill_title_column()
        self._invalidate_table()

    def on_mouse_up(self, event: MouseUp) -> None:
        """End column resize."""
        if self._resize_col is not None:
            col_key = getattr(self._resize_col.key, "value", self._resize_col.key)
            if col_key == "title":
                self._title_manual_width = True
            self._resize_col = None
            self.release_mouse()
            self.suppress_click()
            event.stop()
            self._fill_title_column()
            self._invalidate_table()

    def on_resize(self, event: object) -> None:
        """Re-fill title column when widget is resized."""
        self._title_manual_width = False
        self._fill_title_column()
        self._invalidate_table()

    def on_blur(self) -> None:
        """Clean up drag state if widget loses focus."""
        if self._resize_col is not None:
            self._resize_col = None
            self.release_mouse()
        # A right-click that actually opened a modal blurs the table; arm the
        # one-shot suppression so the spurious RowSelected on refocus is eaten.
        if self._popup_pending:
            self._popup_pending = False
            self._suppress_select_on_refocus = True

    # -- Event handlers ---------------------------------------------------

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Forward track selection as a TrackSelected message."""
        if self._right_clicked:
            self._right_clicked = False
            return
        # Suppress the spurious RowSelected that fires when a modal popup
        # (e.g. ActionsPopup) dismisses and focus returns to this table.
        # The flag is set on right-click and consumed here once.
        if self._suppress_select_on_refocus:
            self._suppress_select_on_refocus = False
            return
        # A genuine selection got through — drop any pending right-click intent
        # so it can't linger and arm suppression on a later, unrelated blur.
        self._popup_pending = False
        row_idx = event.cursor_row
        if 0 <= row_idx < len(self._tracks):
            original_idx = self._filtered_map[row_idx] if self._filtered_map else row_idx
            self.post_message(self.TrackSelected(self._tracks[row_idx], original_idx))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Forward row highlight as a TrackHighlighted message and update SelectionInfoBar.

        Only posts SelectionChanged when this table is actually focused —
        otherwise a freshly-mounted table fires RowHighlighted at row 0
        on init and stomps the sidebar's selection in the info bar.
        """
        row_idx = event.cursor_row
        track = self._tracks[row_idx] if 0 <= row_idx < len(self._tracks) else None
        self.post_message(self.TrackHighlighted(track, row_idx))

        # Range mode follows the cursor: the marks are the baseline plus
        # whatever now lies between the anchor and the highlighted row.
        if self._range_mode:
            self._apply_range()

        if not self.has_focus:
            return

        try:
            if track is None:
                self.post_message(SelectionChanged(""))
                return
            # The info-bar label — the ``_selection_label`` engine
            # (native when the Rust extension is built, the
            # pure-Python transcription otherwise).
            self.post_message(SelectionChanged(_selection_label(track)))
        except Exception:
            self.post_message(SelectionChanged(""))

    def on_focus(self) -> None:
        """When focus moves to this table, push the current row into the bar."""
        try:
            row_idx = self.cursor_row
            if row_idx is None or not (0 <= row_idx < len(self._tracks)):
                self.post_message(SelectionChanged(""))
                return
            track = self._tracks[row_idx]
            self.post_message(SelectionChanged(_selection_label(track)))
        except Exception:
            self.post_message(SelectionChanged(""))

    def on_click(self, event: Click) -> None:
        """Selection gestures on left-click; column-specific messages on right-click.

        Ctrl-click toggles the row, Shift-click marks the run from the
        anchor to the row (displayed order, on top of existing marks) and a
        click on the mark column toggles the row. Each moves the highlight
        to that row and never starts playback: ``prevent_default`` keeps
        DataTable from moving the cursor itself and from posting the
        RowSelected that a click on the highlighted row would raise. Plain
        clicks set the range anchor without changing their playback behaviour;
        in keyboard range mode they extend the existing range instead.
        Whether Ctrl-click or Shift-click
        arrives with its modifier is up to the terminal; ``v``/``V`` are the
        keyboard route.
        """
        if event.button == 1:
            meta = event.style.meta
            row_idx = meta.get("row") if meta else None
            if row_idx is None or not (0 <= row_idx < len(self._tracks)):
                return
            # DataTable tags the filler right of the last column as column 0
            # "out of bounds"; a plain click there is a row click, not a
            # click on the mark column.
            on_mark_column = meta.get("column") == 0 and not meta.get("out_of_bounds")
            if not (event.ctrl or event.shift or on_mark_column):
                if not self._range_mode:
                    self._anchor = self._filtered_map[row_idx]
                return
            event.stop()
            event.prevent_default()
            self._range_mode = False
            self._range_base = frozenset()
            original = self._filtered_map[row_idx]
            if event.shift and not event.ctrl:
                self._extend_to(original)
            else:
                self._toggle_mark(original)
            self.move_cursor(row=row_idx)
            return
        if event.button == 3:
            event.stop()
            event.prevent_default()
            self._right_clicked = True
            self._popup_pending = True
            meta = event.style.meta
            row_idx = meta.get("row") if meta else None
            if row_idx is not None and 0 <= row_idx < len(self._tracks):
                track = self._tracks[row_idx]
                col = self._column_at_x(int(event.x + self.scroll_x))
                col_key = col.key.value if col and col.key else ""
                if col_key == "artist":
                    self.post_message(self.ArtistRightClicked(track, row_idx))
                elif col_key == "album":
                    self.post_message(self.AlbumRightClicked(track, row_idx))
                else:
                    self.post_message(self.TrackRightClicked(self, track, row_idx))

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Sort by the clicked column header."""
        event.stop()
        key = event.column_key.value
        if key in self._SORTABLE_KEYS:
            self.sort_by(key)

    # -- Filtering --------------------------------------------------------

    def apply_filter(self, query: str) -> None:
        """Filter visible rows by query. Empty string restores all tracks."""
        self._filter_text = query.strip().lower()
        self._filter_active = bool(self._filter_text)
        if self._filter_timer is not None:
            try:
                self._filter_timer.stop()
            except Exception:
                pass
        if not self._filter_text:
            self._rebuild_view()
            return
        self._filter_timer = self.set_timer(0.15, self._execute_filter)

    def _execute_filter(self) -> None:
        """Rebuild the table with only matching tracks (debounced)."""
        self._filter_timer = None
        self._rebuild_view()

    def _rebuild_view(self) -> None:
        """Recompute the visible rows from the backing list: the filter, then the sort.

        One path for filtering, sorting and ``refresh_tracks``, so the
        rows come out the same whichever order the user applied them in
        — a filter typed on a sorted list keeps the sort. The order
        itself — and, through the ``_compose_view_rows`` engine, every
        visible row's cell assembly — comes from one fused pass
        (the native Rust engine when the extension is built, the
        pure-Python composition otherwise).
        """
        positions, rows = _compose_view_rows(
            self._all_tracks,
            self._filter_text,
            self._sort_column,
            self._sort_reverse,
            self._show_index,
            self._show_album,
            self._marked,
        )
        self._tracks = [self._all_tracks[i] for i in positions]
        self._filtered_map = [t["_original_index"] for t in self._tracks]
        self._reload_sorted(rows)

    def show_filter(self) -> None:
        """Signal that filter mode should begin."""
        self._filter_active = True
        self.post_message(self.FilterRequested())

    def clear_filter(self) -> None:
        """Remove the filter, restoring all tracks."""
        self._filter_text = ""
        self._filter_active = False
        self._rebuild_view()
        self.post_message(self.FilterClosed())

    # -- Sorting ----------------------------------------------------------

    def sort_by(self, column: str) -> None:
        """Sort tracks by column. Toggles direction if the same column is sorted again."""
        if not self._tracks:
            return

        # The toggle decision — the ``_sort_state`` engine (native when
        # the Rust extension is built, the pure-Python transcription
        # otherwise); ``None`` leaves the sort untouched.
        state = _sort_state(column, self._sort_column, self._sort_reverse)
        if state is None:
            return
        self._sort_column, self._sort_reverse = state

        # Restore by occurrence, not video ID: with the same track listed
        # twice, the cursor stays on the copy it was on.
        current = self.selected_original_index
        self._rebuild_view()
        self._place_cursor_on(current)

    def clear_sort(self) -> None:
        """Drop the active sort, back to load order; a no-op when unsorted.

        The filter, the marks and the cursor's occurrence all stay.
        """
        if self._sort_column is None:
            return
        current = self.selected_original_index
        self._sort_column = None
        self._sort_reverse = False
        self._rebuild_view()
        self._place_cursor_on(current)

    def _place_cursor_on(self, original: int | None) -> None:
        """Move the cursor to the visible row of occurrence *original*, if shown."""
        if original is None:
            return
        try:
            self.move_cursor(row=self._filtered_map.index(original))
        except ValueError:
            pass

    def _reload_sorted(self, rows: Sequence[tuple[str, list[str]]]) -> None:
        """Rebuild table rows from the pre-composed ``(row_key, cells)`` pairs."""
        saved_scroll_x = self.scroll_x
        self.clear()
        self._row_keys = []
        self._playing_index = None
        self._append_rows(rows)
        self._highlight_playing()
        self.scroll_x = saved_scroll_x
        self._update_mark_status()

    # -- Vim-style navigation ---------------------------------------------

    async def handle_action(self, action: Action, count: int = 1) -> None:
        """Process navigation actions dispatched from the app."""
        match action:
            case Action.MOVE_DOWN:
                for _ in range(count):
                    self.action_cursor_down()
            case Action.MOVE_UP:
                for _ in range(count):
                    self.action_cursor_up()
            case Action.PAGE_DOWN:
                self.action_scroll_down()
            case Action.PAGE_UP:
                self.action_scroll_up()
            case Action.GO_TOP:
                if self.row_count > 0:
                    self.move_cursor(row=0)
            case Action.GO_BOTTOM:
                if self.row_count > 0:
                    self.move_cursor(row=self.row_count - 1)
            case Action.SELECT:
                if self.cursor_row is not None and 0 <= self.cursor_row < len(self._tracks):
                    original_idx = (
                        self._filtered_map[self.cursor_row]
                        if self._filtered_map
                        else self.cursor_row
                    )
                    self.post_message(
                        self.TrackSelected(self._tracks[self.cursor_row], original_idx)
                    )
            case Action.FILTER:
                self.show_filter()
            case Action.MARK_TOGGLE:
                # A range in progress ends first (its marks stay), then the
                # highlighted row toggles.
                self.end_range_mode()
                original = self.selected_original_index
                if original is not None:
                    self._toggle_mark(original)
            case Action.MARK_RANGE:
                if self._range_mode:
                    self.end_range_mode()
                    return
                original = self.selected_original_index
                if original is not None:
                    self._range_base = frozenset(self._marked)
                    self._anchor = original
                    self._range_mode = True
                    self._apply_range()
            case Action.MARK_CLEAR:
                self.clear_marks()
            case Action.JUMP_TO_CURRENT:
                self._jump_to_current()
            case Action.SORT_TITLE:
                self.sort_by("title")
            case Action.SORT_ARTIST:
                self.sort_by("artist")
            case Action.SORT_ALBUM:
                self.sort_by("album")
            case Action.SORT_DURATION:
                self.sort_by("duration")
            case Action.REVERSE_SORT:
                if self._sort_column and self._tracks:
                    self._sort_reverse = not self._sort_reverse
                    current = self.selected_original_index
                    self._rebuild_view()
                    self._place_cursor_on(current)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same view engine, the same row
# assembly and the same rendering-integration engines; these bindings
# replace the pure-Python references above when the extension is
# importable.  The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        filter_sort_tracks as _native_filter_sort_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_carry_selection_keys as _native_carry_selection_keys,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_column_keys as _native_column_keys,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_extend_marks as _native_extend_marks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_hidden_marked_count as _native_hidden_marked_count,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_mark_status as _native_mark_status,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_marked_tracks as _native_marked_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_range_marks as _native_range_marks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_reindex_after_removal as _native_reindex_after_removal,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_row_values as _native_row_values,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_selection_label as _native_selection_label,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_shift_selection as _native_shift_selection,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_sort_state as _native_sort_state,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_stamp_tracks as _native_stamp_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_toggle_marks as _native_toggle_marks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_view_rows as _native_view_rows,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_table_visible_run as _native_visible_run,
    )
except ImportError:
    pass
else:

    def _filter_sort_positions(  # noqa: F811
        tracks: list[dict],
        query: str,
        sort_column: str | None,
        sort_reverse: bool,
    ) -> list[int]:
        """Compute the visible-row order (native Rust engine).

        Same contract as the pure-Python reference above: returns the
        positions of the visible rows in display order. Non-dict rows and
        non-string title/album values degrade gracefully where the
        reference would raise — ``load_tracks`` never produces them.
        """
        return _native_filter_sort_tracks(tracks, query, sort_column, sort_reverse)

    def _row_values(  # noqa: F811
        track: dict,
        index: int,
        show_index: bool,
        show_album: bool,
        marked: AbstractSet[int],
    ) -> tuple[str, list[str]]:
        """One visible row's (row_key, cells) pair (native Rust engine).

        Same contract as the pure-Python reference above; the whole
        per-row assembly — the reads, the glyph containment, the
        position render, the BiDi cell renders and the row key — runs
        inside one native call, with only the cached ``bidi_mode``
        verdict read here.
        """
        from ytm_player.utils.bidi import _get_reorder_enabled

        return _native_row_values(
            track, index, show_index, show_album, marked, _get_reorder_enabled()
        )

    def _compose_view_rows(  # noqa: F811
        tracks: list[dict],
        query: str,
        sort_column: str | None,
        sort_reverse: bool,
        show_index: bool,
        show_album: bool,
        marked: AbstractSet[int],
    ) -> tuple[list[int], list[tuple[str, list[str]]]]:
        """The visible rows' (positions, rows) (native Rust engine).

        Same contract as the pure-Python composition above: the whole
        rebuild — the filter, the sort and every visible row's cell
        assembly — runs inside one native call, with only the cached
        ``bidi_mode`` verdict read here (once per rebuild instead of
        once per cell).
        """
        from ytm_player.utils.bidi import _get_reorder_enabled

        return _native_view_rows(
            tracks,
            query,
            sort_column,
            sort_reverse,
            show_index,
            show_album,
            marked,
            _get_reorder_enabled(),
        )

    def _stamp_tracks(tracks: list[dict], start: int = 0) -> list[dict]:  # noqa: F811
        """The load/append stamping pass (native Rust engine)."""
        return _native_stamp_tracks(tracks, start)

    def _mark_status_text(count: int, hidden: int) -> str:  # noqa: F811
        """The marked-rows status line (native Rust engine)."""
        return _native_mark_status(count, hidden)

    def _selection_label(track: dict) -> str:  # noqa: F811
        """The selection-info label (native Rust engine).

        Same contract as the pure-Python reference above; a non-dict
        track degrades through the defaults where the reference's
        ``.get`` would raise — the table's own paths only ever hold
        dicts.
        """
        return _native_selection_label(track)

    def _reindex_after_removal(  # noqa: F811
        all_tracks: list[dict], tracks: list[dict]
    ) -> tuple[list[int], list[str]]:
        """The post-removal renumbering (native Rust engine)."""
        return _native_reindex_after_removal(all_tracks, tracks)

    def _sort_state(  # noqa: F811
        column: str, sort_column: str | None, sort_reverse: bool
    ) -> tuple[str, bool] | None:
        """The header-click sort toggle (native Rust engine)."""
        return _native_sort_state(column, sort_column, sort_reverse)

    def _visible_run(  # noqa: F811
        filtered_map: Sequence[int], start: int, end: int
    ) -> set[int]:
        """The visible-run core (native Rust engine).

        Same contract as the pure-Python reference above: the original
        indices of the visible rows from *start* to *end*, inclusive —
        a *start* the filter hides counting as no start, the run *end*
        alone.
        """
        return _native_visible_run(filtered_map, start, end)

    def _extend_marks(  # noqa: F811
        anchor: int | None,
        filtered_map: Sequence[int],
        marked: AbstractSet[int],
        original: int,
    ) -> tuple[int | None, set[int]]:
        """The shift-click extend decision (native Rust engine).

        Same contract as the pure-Python reference above: the new
        ``(anchor, marked)`` pair, with no anchor (or an anchor the
        filter hides) marking the row and taking the anchor.
        """
        return _native_extend_marks(anchor, filtered_map, marked, original)

    def _range_marks(  # noqa: F811
        anchor: int | None,
        cursor: int | None,
        filtered_map: Sequence[int],
        range_base: AbstractSet[int],
    ) -> set[int] | None:
        """The keyboard-range marks (native Rust engine).

        Same contract as the pure-Python reference above: ``None``
        without an anchor, otherwise the range baseline unioned with
        the anchor-to-cursor run.
        """
        return _native_range_marks(anchor, cursor, filtered_map, range_base)

    def _toggle_marks(  # noqa: F811
        marked: AbstractSet[int], original: int
    ) -> set[int]:
        """The toggle's marked set (native Rust engine)."""
        return _native_toggle_marks(marked, original)

    def _shift_selection(  # noqa: F811
        removed: int,
        marked: AbstractSet[int],
        range_base: AbstractSet[int],
        anchor: int | None,
        range_mode: bool,
    ) -> tuple[set[int], frozenset[int], int | None, bool, bool]:
        """The post-removal selection shift (native Rust engine).

        Same contract as the pure-Python reference above: the
        ``(marks, base, anchor, range_mode, bump)`` quintuple.
        """
        return _native_shift_selection(removed, marked, range_base, anchor, range_mode)

    def _marked_tracks(  # noqa: F811
        all_tracks: list[dict],
        marked: AbstractSet[int],
        sort_column: str | None,
        sort_reverse: bool,
    ) -> list[dict]:
        """The marked tracks in displayed order (native Rust engine).

        Same contract as the pure-Python reference above; a non-dict
        track raises where the reference's ``track["_original_index"]``
        would raise — ``load_tracks`` never produces them.
        """
        return _native_marked_tracks(all_tracks, marked, sort_column, sort_reverse)

    def _hidden_marked_count(  # noqa: F811
        marked: AbstractSet[int], filtered_map: Sequence[int]
    ) -> int:
        """The hidden-marks count (native Rust engine)."""
        return _native_hidden_marked_count(marked, filtered_map)

    def _carry_selection_keys(  # noqa: F811
        old_keys: Sequence[Hashable],
        new_keys: Sequence[Hashable],
        marked: Iterable[int],
        range_base: Iterable[int],
        anchor: int | None,
        cursor: int | None,
    ) -> tuple[set[int], frozenset[int], int | None, int | None, bool]:
        """Resolve the selection's occurrence keys across a reload
        (native Rust engine).

        Same contract as the pure-Python reference above: the
        ``(marked, range_base, anchor, cursor, dropped)`` quintuple,
        with duplicated or missing keys resolving nothing. The
        ``old_keys``/``new_keys`` lists and the mark sets pass through
        as the objects they are, so CPython's own hash/eq semantics
        decide key identity.
        """
        return _native_carry_selection_keys(old_keys, new_keys, marked, range_base, anchor, cursor)

    _column_keys = _native_column_keys  # type: ignore[assignment]
