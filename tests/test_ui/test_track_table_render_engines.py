"""Tests for the track-table rendering-integration engines.

``_stamp_tracks`` (the load/append stamping pass), ``_mark_status_text``
(the marked-rows status line), ``_selection_label`` (the selection-info
label), ``_reindex_after_removal`` (the post-removal renumbering),
``_sort_state`` (the header-click sort toggle) and ``_column_keys`` (the
composed cells' column keys) are the rendering glue behind
``TrackTable``'s load, append, remove, mark, highlight and sort paths.
The pure-Python references live in ``ui/widgets/track_table.py``; when
the Rust extension is built the same engines run natively
(``_ytm_player_native``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.widgets.track_table import (
    _column_keys,
    _mark_status_text,
    _reindex_after_removal,
    _selection_label,
    _sort_state,
    _stamp_tracks,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Stamping ──────────────────────────────────────────────────────────


class TestStampTracks:
    def test_copies_are_stamped_from_zero(self) -> None:
        tracks = [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}]
        stamped = _stamp_tracks(tracks)
        assert [t["_original_index"] for t in stamped] == [0, 1, 2]

    def test_start_offsets_the_stamps(self) -> None:
        stamped = _stamp_tracks([{"video_id": "a"}, {"video_id": "b"}], start=5)
        assert [t["_original_index"] for t in stamped] == [5, 6]

    def test_copies_are_new_dicts_with_the_same_values(self) -> None:
        track = {"video_id": "a", "title": "A", "nested": ["shared"]}
        stamped = _stamp_tracks([track])
        assert stamped[0] is not track
        assert stamped[0]["nested"] is track["nested"]
        assert stamped[0]["video_id"] == "a"

    def test_a_preexisting_stamp_is_overwritten(self) -> None:
        stamped = _stamp_tracks([{"video_id": "a", "_original_index": 99}])
        assert stamped[0]["_original_index"] == 0

    def test_the_input_list_and_dicts_are_untouched(self) -> None:
        track = {"video_id": "a"}
        _stamp_tracks([track], start=7)
        assert track == {"video_id": "a"}

    def test_empty_input(self) -> None:
        assert _stamp_tracks([]) == []


# ── Marked-rows status line ───────────────────────────────────────────


class TestMarkStatusText:
    def test_one_selected(self) -> None:
        assert _mark_status_text(1, 0) == "1 selected · A: add · Esc: clear"

    def test_many_selected_no_hidden(self) -> None:
        assert _mark_status_text(5, 0) == "5 selected · A: add · Esc: clear"

    def test_hidden_marked_rows_are_called_out(self) -> None:
        assert _mark_status_text(3, 2) == "3 selected (2 hidden) · A: add · Esc: clear"


# ── Selection-info label ──────────────────────────────────────────────


class TestSelectionLabel:
    def test_title_and_artist(self) -> None:
        assert _selection_label({"title": "Song", "artist": "Artist"}) == "Song — Artist"

    def test_artists_list_joins_into_the_label(self) -> None:
        track = {"title": "Song", "artists": [{"name": "A", "id": "1"}, {"name": "B", "id": "2"}]}
        assert _selection_label(track) == "Song — A, B"

    def test_missing_artist_falls_back_to_unknown(self) -> None:
        # extract_artist's "Unknown" fallback joins like any other name.
        assert _selection_label({"title": "Song"}) == "Song — Unknown"

    def test_missing_title(self) -> None:
        assert _selection_label({"artist": "Artist"}) == " — Artist"

    def test_empty_track(self) -> None:
        assert _selection_label({}) == " — Unknown"


# ── Post-removal renumbering ──────────────────────────────────────────


class TestReindexAfterRemoval:
    def _tracks(self, *originals: int) -> list[dict]:
        return [{"video_id": f"v{o}", "_original_index": o} for o in originals]

    def test_renumbers_the_backing_list_in_place(self) -> None:
        all_tracks = self._tracks(0, 2, 3)  # row 1 was removed
        _filtered_map, _cells = _reindex_after_removal(all_tracks, all_tracks)
        assert [t["_original_index"] for t in all_tracks] == [0, 1, 2]

    def test_the_map_and_cells_follow_the_visible_order(self) -> None:
        all_tracks = self._tracks(0, 2, 3)
        visible = [all_tracks[2], all_tracks[0], all_tracks[1]]  # sorted view
        filtered_map, index_cells = _reindex_after_removal(all_tracks, visible)
        # After the renumber, the dicts the visible list holds are the
        # same objects — their new stamps answer the map, in visible
        # order, and the # cells are the stamps + 1.
        assert filtered_map == [2, 0, 1]
        assert index_cells == ["3", "1", "2"]

    def test_empty_lists(self) -> None:
        assert _reindex_after_removal([], []) == ([], [])


# ── Sort toggle ───────────────────────────────────────────────────────


class TestSortState:
    def test_unsortable_columns_answer_none(self) -> None:
        assert _sort_state("nope", None, False) is None
        assert _sort_state("", "title", True) is None

    @pytest.mark.parametrize("column", ["index", "title", "artist", "album", "duration"])
    def test_every_sortable_column_from_an_unsorted_table(self, column: str) -> None:
        assert _sort_state(column, None, False) == (column, False)
        # A stale reverse flag never leaks into a fresh sort.
        assert _sort_state(column, "other", True) == (column, False)

    def test_the_same_column_flips_the_direction(self) -> None:
        assert _sort_state("title", "title", False) == ("title", True)
        assert _sort_state("title", "title", True) == ("title", False)


# ── Column keys ───────────────────────────────────────────────────────


class TestColumnKeys:
    @pytest.mark.parametrize(
        ("show_index", "show_album", "expected"),
        [
            (True, True, ["mark", "index", "title", "artist", "album", "duration"]),
            (True, False, ["mark", "index", "title", "artist", "duration"]),
            (False, True, ["mark", "title", "artist", "album", "duration"]),
            (False, False, ["mark", "title", "artist", "duration"]),
        ],
    )
    def test_the_column_order(
        self, show_index: bool, show_album: bool, expected: list[str]
    ) -> None:
        assert _column_keys(show_index, show_album) == expected


# ── Native / fallback parity ──────────────────────────────────────────
#
# The child interpreter runs the pure-Python reference engines with the
# extension import blocked and prints the outcomes as JSON; the parity
# tests compare them against the loaded (native) engines case for case.
# The two mutating engines (stamping, reindexing) run on deep-copied
# inputs on each side, so their outcomes compare without sharing state.

_STAMP_CASES = [
    [{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}],
    [{"video_id": "a", "_original_index": 99}],
    [{"video_id": "a", "title": "A", "nested": ["shared"]}],
    [],
]

_STAMP_STARTS = [0, 5, 0, 0]

_REINDEX_CASES = [
    # (all_tracks, tracks) — visible rows named by their video_id.
    (
        [{"video_id": "v0", "_original_index": 0}, {"video_id": "v2", "_original_index": 2}],
        ["v2", "v0"],
    ),
    (
        [
            {"video_id": "v0", "_original_index": 0},
            {"video_id": "v3", "_original_index": 3},
            {"video_id": "v5", "_original_index": 5},
        ],
        ["v5", "v0", "v3"],
    ),
    ([], []),
]

_MARK_CASES = [(1, 0), (5, 0), (3, 2), (12, 1)]

_LABEL_CASES = [
    {"title": "Song", "artist": "Artist"},
    {"title": "Song", "artists": [{"name": "A", "id": "1"}, {"name": "B", "id": "2"}]},
    {"title": "Song"},
    {"artist": "Artist"},
    {},
    {"title": "X", "artists": [{"name": "358K views", "id": None}]},
]

_SORT_CASES = [
    ("nope", None, False),
    ("", "title", True),
    ("title", None, False),
    ("title", "other", True),
    ("title", "title", False),
    ("title", "title", True),
    ("duration", "duration", False),
    ("index", "index", True),
    ("artist", None, True),
    ("album", "album", False),
]

_KEY_CASES = [(True, True), (True, False), (False, True), (False, False)]


def _run_reindex(case: tuple[list[dict], list[str]]) -> tuple[list[int], list[str]]:
    all_tracks = [dict(t) for t in case[0]]
    by_id = {t["video_id"]: t for t in all_tracks}
    visible = [by_id[vid] for vid in case[1]]
    filtered_map, index_cells = _reindex_after_removal(all_tracks, visible)
    return filtered_map, index_cells


_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets.track_table import (
    _column_keys,
    _mark_status_text,
    _reindex_after_removal,
    _selection_label,
    _sort_state,
    _stamp_tracks,
)

payload = json.load(sys.stdin)

stamp = []
for case, start in zip(payload["stamp_cases"], payload["stamp_starts"]):
    stamp.append([dict(t) for t in _stamp_tracks([dict(c) for c in case], start)])

reindex = []
for all_case, visible_ids in payload["reindex_cases"]:
    all_tracks = [dict(t) for t in all_case]
    by_id = {t["video_id"]: t for t in all_tracks}
    visible = [by_id[vid] for vid in visible_ids]
    filtered_map, index_cells = _reindex_after_removal(all_tracks, visible)
    reindex.append([filtered_map, index_cells])

out = {
    "stamp": stamp,
    "reindex": reindex,
    "mark": [_mark_status_text(c, h) for c, h in payload["mark_cases"]],
    "label": [_selection_label(t) for t in payload["label_cases"]],
    "sort": [
        None if (s := _sort_state(c, col, rev)) is None else list(s)
        for c, col, rev in payload["sort_cases"]
    ],
    "keys": [_column_keys(i, a) for i, a in payload["key_cases"]],
}
json.dump(out, sys.stdout, sort_keys=True)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "stamp_cases": _STAMP_CASES,
            "stamp_starts": _STAMP_STARTS,
            "reindex_cases": _REINDEX_CASES,
            "mark_cases": _MARK_CASES,
            "label_cases": _LABEL_CASES,
            "sort_cases": _SORT_CASES,
            "key_cases": _KEY_CASES,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _native_outcomes() -> dict:
    stamp = []
    for case, start in zip(_STAMP_CASES, _STAMP_STARTS):
        stamp.append([dict(t) for t in _stamp_tracks([dict(c) for c in case], start)])

    # The reindex answer is a (map, cells) pair; JSON spells both as
    # lists, so the native tuples are compared through the same shape.
    reindex = [list(_run_reindex(case)) for case in _REINDEX_CASES]

    return {
        "stamp": stamp,
        "reindex": reindex,
        "mark": [_mark_status_text(c, h) for c, h in _MARK_CASES],
        "label": [_selection_label(t) for t in _LABEL_CASES],
        "sort": [
            None if (state := _sort_state(c, col, rev)) is None else list(state)
            for c, col, rev in _SORT_CASES
        ],
        "keys": [_column_keys(i, a) for i, a in _KEY_CASES],
    }


class TestNativeParity:
    """When the Rust extension is built, the native rendering-integration
    engines must produce exactly the same outcomes as the pure-Python
    references — same stamped copies, same renumbered maps and cells,
    same status lines, labels, sort toggles and column keys — for every
    case, so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        """Every rendering-integration outcome — the stamped copies, the
        post-removal renumbering, the status line, the selection label,
        the sort toggle and the column keys — must agree with the
        pure-Python reference case for case."""
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key in native:
            assert native[key] == fallback[key], (
                f"native and fallback track-table render outcomes differ for {key}:\n"
                f"native:   {native[key]!r}\n"
                f"fallback: {fallback[key]!r}"
            )
