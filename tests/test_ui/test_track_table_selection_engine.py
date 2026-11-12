"""Tests for the track-table mark/range selection engines.

The selection core of ``TrackTable`` — ``_visible_run`` (the
visible-run computation), ``_extend_marks`` (the shift-click extend
decision), ``_range_marks`` (the keyboard-range marks),
``_toggle_marks`` (the toggle's symmetric difference),
``_shift_selection`` (the post-removal selection shift),
``_marked_tracks`` (the marked-tracks walk in displayed order),
``_hidden_marked_count`` (the hidden-marks count) and
``_carry_selection_keys`` (the occurrence-key carry across a reload) —
is the engine behind the widget's mark gestures (``v``/``V``/Escape,
Ctrl-click, Shift-click, mark-column clicks), its removal paths and
its background refresh. The pure-Python references live in
``ui/widgets/track_table.py``; when the Rust extension is built the
same engines run natively (``_ytm_player_native``).

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
    _carry_selection_keys,
    _extend_marks,
    _hidden_marked_count,
    _marked_tracks,
    _range_marks,
    _shift_selection,
    _toggle_marks,
    _visible_run,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _track(
    title: str,
    video_id: str,
    index: int,
    duration: int = 100,
    artist: str = "Artist",
) -> dict:
    """Build a stamped track dict like the ones ``load_tracks`` builds."""
    return {
        "video_id": video_id,
        "title": title,
        "artist": artist,
        "artists": [{"name": artist, "id": "1"}],
        "album": "Album",
        "duration": duration,
        "_original_index": index,
    }


def _stamp(tracks: list[dict]) -> list[dict]:
    """Stamp load-order ``_original_index`` values like ``load_tracks``."""
    for i, t in enumerate(tracks):
        t["_original_index"] = i
    return tracks


# ── The visible run ────────────────────────────────────────────────────


class TestVisibleRun:
    def test_identity_map_run_is_inclusive(self) -> None:
        assert _visible_run([0, 1, 2, 3, 4], 1, 3) == {1, 2, 3}

    def test_run_reads_the_same_backwards(self) -> None:
        # The run between the rows carrying 4 (row 0) and 2 (row 2).
        assert _visible_run([4, 3, 2, 1, 0], 4, 2) == {2, 3, 4}

    def test_scrambled_map_runs_over_visible_rows_not_originals(self) -> None:
        # Visible rows 1..3 carry originals 3, 0, 4.
        assert _visible_run([2, 3, 0, 4, 1], 3, 4) == {3, 0, 4}

    def test_single_row_run(self) -> None:
        assert _visible_run([0, 1, 2], 2, 2) == {2}

    def test_hidden_start_counts_as_no_start(self) -> None:
        assert _visible_run([0, 1, 2], 9, 1) == {1}

    def test_hidden_end_answers_the_end_alone(self) -> None:
        assert _visible_run([0, 1, 2], 1, 9) == {9}

    def test_empty_map_answers_the_end_alone(self) -> None:
        assert _visible_run([], 0, 5) == {5}


# ── The shift-click extend decision ────────────────────────────────────


class TestExtendMarks:
    def test_no_anchor_marks_the_row_and_takes_the_anchor(self) -> None:
        anchor, marks = _extend_marks(None, [0, 1, 2], {5}, 1)
        assert anchor == 1
        assert marks == {1, 5}

    def test_hidden_anchor_marks_the_row_and_takes_the_anchor(self) -> None:
        anchor, marks = _extend_marks(7, [0, 1, 2], {5}, 2)
        assert anchor == 2
        assert marks == {2, 5}

    def test_anchored_view_marks_the_run(self) -> None:
        anchor, marks = _extend_marks(0, [0, 1, 2], {9}, 2)
        assert anchor == 0
        assert marks == {0, 1, 2, 9}

    def test_anchored_view_keeps_marks_outside_the_run(self) -> None:
        anchor, marks = _extend_marks(0, [0, 1, 2, 3, 4], {9}, 1)
        assert anchor == 0
        assert marks == {0, 1, 9}

    def test_the_callers_marked_set_is_never_mutated(self) -> None:
        marked = {4}
        _anchor, _marks = _extend_marks(0, [0, 1, 2], marked, 2)
        assert marked == {4}

    def test_scrambled_run_extends_over_the_displayed_order(self) -> None:
        # Rows 1..3 carry originals 3, 0, 4 — the run covers them all.
        anchor, marks = _extend_marks(3, [2, 3, 0, 4, 1], {8}, 4)
        assert anchor == 3
        assert marks == {0, 3, 4, 8}


# ── The keyboard-range marks ────────────────────────────────────────────


class TestRangeMarks:
    def test_no_anchor_answers_none(self) -> None:
        assert _range_marks(None, 2, [0, 1, 2], frozenset({5})) is None

    def test_the_baseline_alone_when_the_cursor_is_off_the_table(self) -> None:
        assert _range_marks(0, None, [0, 1, 2], frozenset({5})) == {5}

    def test_baseline_plus_the_anchor_to_cursor_run(self) -> None:
        assert _range_marks(0, 2, [0, 1, 2], frozenset({5})) == {0, 1, 2, 5}

    def test_the_run_reads_the_same_when_the_cursor_is_above_the_anchor(
        self,
    ) -> None:
        assert _range_marks(3, 0, [0, 1, 2, 3], frozenset()) == {0, 1, 2, 3}

    def test_empty_baseline_and_empty_run(self) -> None:
        assert _range_marks(1, 1, [0, 1, 2], frozenset()) == {1}

    def test_the_callers_baseline_is_never_mutated(self) -> None:
        base = frozenset({5})
        _marks = _range_marks(0, 2, [0, 1, 2], base)
        assert base == frozenset({5})


# ── The toggle's symmetric difference ───────────────────────────────────


class TestToggleMarks:
    def test_toggle_adds_the_row(self) -> None:
        assert _toggle_marks({1, 2}, 3) == {1, 2, 3}

    def test_toggle_drops_the_row(self) -> None:
        assert _toggle_marks({1, 2}, 1) == {2}

    def test_toggle_on_an_empty_set(self) -> None:
        assert _toggle_marks(set(), 7) == {7}

    def test_the_callers_set_is_never_mutated(self) -> None:
        marked = {1}
        assert _toggle_marks(marked, 2) == {1, 2}
        assert marked == {1}


# ── The post-removal selection shift ────────────────────────────────────


class TestShiftSelection:
    def test_marks_above_slide_down_and_marks_below_stay(self) -> None:
        marked, base, anchor, mode, bump = _shift_selection(
            1, {0, 1, 2, 3}, frozenset({2, 3}), 2, False
        )
        assert marked == {0, 1, 2}
        assert base == frozenset({1, 2})
        assert anchor == 1
        assert mode is False
        # The removed occurrence was itself marked, so the generation bumps.
        assert bump is True

    def test_removing_a_marked_row_answers_bump(self) -> None:
        marked, _base, _anchor, _mode, bump = _shift_selection(
            1, {0, 1, 2}, frozenset(), None, False
        )
        assert marked == {0, 1}
        assert bump is True

    def test_removing_an_unmarked_row_answers_no_bump(self) -> None:
        _marked, _base, _anchor, _mode, bump = _shift_selection(1, {0, 2}, frozenset(), None, False)
        assert bump is False

    def test_the_anchor_dies_with_its_occurrence(self) -> None:
        marked, base, anchor, mode, bump = _shift_selection(2, {0, 4}, frozenset({4}), 2, True)
        assert marked == {0, 3}
        assert anchor is None
        assert mode is False
        assert base == frozenset()
        # The removed occurrence was not marked, so no generation bump.
        assert bump is False

    def test_range_mode_survives_when_the_anchor_survives(self) -> None:
        marked, base, anchor, mode, bump = _shift_selection(0, {1, 2}, frozenset({2}), 2, True)
        assert marked == {0, 1}
        assert base == frozenset({1})
        assert anchor == 1
        assert mode is True
        assert bump is False

    def test_range_mode_without_an_anchor_from_the_start_drops_out(self) -> None:
        _marked, base, _anchor, mode, _bump = _shift_selection(1, {3}, frozenset({3}), None, True)
        assert mode is False
        assert base == frozenset()

    def test_removal_at_zero_shifts_everything_down(self) -> None:
        marked, _base, _anchor, _mode, _bump = _shift_selection(
            0, {0, 1, 2}, frozenset(), None, False
        )
        assert marked == {0, 1}

    def test_the_callers_sets_are_never_mutated(self) -> None:
        marked, base = {0, 1}, frozenset({1})
        _shift_selection(0, marked, base, None, False)
        assert marked == {0, 1}
        assert base == frozenset({1})


# ── The marked-tracks walk ──────────────────────────────────────────────


class TestMarkedTracks:
    def test_no_marks_answer_no_tracks(self) -> None:
        tracks = _stamp([_track("A", "a", 0)])
        assert _marked_tracks(tracks, set(), None, False) == []

    def test_load_order_walk(self) -> None:
        tracks = _stamp(
            [_track("Alpha", "a", 0), _track("Bravo", "b", 1), _track("Charlie", "c", 2)]
        )
        out = _marked_tracks(tracks, {0, 2}, None, False)
        assert [t["video_id"] for t in out] == ["a", "c"]

    def test_sorted_walk_orders_the_whole_backing_list(self) -> None:
        tracks = _stamp(
            [_track("Charlie", "c", 0), _track("Alpha", "a", 1), _track("Bravo", "b", 2)]
        )
        out = _marked_tracks(tracks, {0, 1, 2}, "title", False)
        assert [t["video_id"] for t in out] == ["a", "b", "c"]

    def test_reverse_sort_keeps_ties_in_input_order(self) -> None:
        tracks = _stamp(
            [
                _track("Alpha", "a1", 0),
                _track("Alpha", "a2", 1),
                _track("Bravo", "b", 2),
            ]
        )
        out = _marked_tracks(tracks, {0, 1, 2}, "title", True)
        assert [t["video_id"] for t in out] == ["b", "a1", "a2"]

    def test_hidden_marked_tracks_land_where_they_would_show(self) -> None:
        # A sort orders the whole backing list, so the marked track the
        # filter hides lands where it would show without the filter.
        tracks = _stamp(
            [_track("Alpha", "a", 0), _track("Bravo", "b", 1), _track("Charlie", "c", 2)]
        )
        out = _marked_tracks(tracks, {0, 2}, "title", False)
        assert [t["video_id"] for t in out] == ["a", "c"]

    def test_the_answer_holds_the_backing_lists_own_objects(self) -> None:
        tracks = _stamp([_track("Alpha", "a", 0), _track("Bravo", "b", 1)])
        out = _marked_tracks(tracks, {0}, None, False)
        assert out == [tracks[0]]
        assert out[0] is tracks[0]


# ── The hidden-marks count ──────────────────────────────────────────────


class TestHiddenMarkedCount:
    def test_nothing_hidden_when_everything_is_visible(self) -> None:
        assert _hidden_marked_count({1, 2}, [0, 1, 2]) == 0

    def test_marks_the_filter_hides_count(self) -> None:
        assert _hidden_marked_count({1, 3, 5}, [0, 1, 2]) == 2

    def test_empty_visible_map_hides_everything(self) -> None:
        assert _hidden_marked_count({1, 2}, []) == 2

    def test_empty_marked_set_answers_zero(self) -> None:
        assert _hidden_marked_count(set(), [0, 1]) == 0

    def test_duplicates_in_the_visible_map_do_not_double_count(self) -> None:
        assert _hidden_marked_count({1}, [1, 1, 1]) == 0


# ── The occurrence-key carry ────────────────────────────────────────────


class TestCarrySelectionKeys:
    def test_identity_reload_carries_everything(self) -> None:
        marked, base, anchor, cursor, dropped = _carry_selection_keys(
            ["a", "b", "c"], ["a", "b", "c"], {0, 2}, frozenset({2}), 2, 0
        )
        assert marked == {0, 2}
        assert base == frozenset({2})
        assert anchor == 2
        assert cursor == 0
        assert dropped is False

    def test_reordered_keys_carry_by_key_not_position(self) -> None:
        marked, _base, anchor, cursor, dropped = _carry_selection_keys(
            ["a", "b", "c"], ["c", "a", "b"], {1}, frozenset(), 1, 0
        )
        assert marked == {2}
        assert anchor == 2
        assert cursor == 1
        assert dropped is False

    def test_a_key_missing_from_the_new_list_drops_its_mark(self) -> None:
        marked, _base, _anchor, _cursor, dropped = _carry_selection_keys(
            ["a", "b", "c"], ["a", "c"], {0, 1}, frozenset(), None, None
        )
        assert marked == {0}
        assert dropped is True

    def test_a_duplicated_new_key_resolves_nothing(self) -> None:
        marked, _base, _anchor, _cursor, dropped = _carry_selection_keys(
            ["a", "b"], ["a", "a", "b"], {1}, frozenset(), None, None
        )
        assert marked == {2}
        assert dropped is False

    def test_a_duplicated_old_key_resolves_nothing(self) -> None:
        marked, _base, _anchor, _cursor, dropped = _carry_selection_keys(
            ["a", "a", "b"], ["a", "b"], {0, 2}, frozenset(), None, None
        )
        # The mark on the duplicated "a" resolves to nothing (dropped),
        # while the unique "b" carries.
        assert marked == {1}
        assert dropped is True

    def test_the_anchor_and_cursor_die_when_their_keys_do_not_resolve(
        self,
    ) -> None:
        _marked, _base, anchor, cursor, _dropped = _carry_selection_keys(
            ["a", "b", "c"], ["a", "c"], {0}, frozenset(), 1, 1
        )
        assert anchor is None
        assert cursor is None

    def test_none_anchor_and_none_cursor_pass_through(self) -> None:
        marked, base, anchor, cursor, dropped = _carry_selection_keys(
            ["a"], ["a"], set(), frozenset(), None, None
        )
        assert marked == set()
        assert base == frozenset()
        assert anchor is None
        assert cursor is None
        assert dropped is False

    def test_every_mark_carried_answers_not_dropped(self) -> None:
        marked, _base, _anchor, _cursor, dropped = _carry_selection_keys(
            ["a", "b"], ["b", "a"], {0, 1}, frozenset(), None, None
        )
        assert marked == {0, 1}
        assert dropped is False

    def test_a_markless_reload_answers_no_dropped_marks(self) -> None:
        marked, _base, _anchor, _cursor, dropped = _carry_selection_keys(
            ["a", "b"], ["a"], set(), frozenset(), None, None
        )
        assert marked == set()
        assert dropped is False


# ── Native / fallback parity ───────────────────────────────────────────
#
# The child interpreter runs the pure-Python reference engines with the
# extension import blocked and prints the outcomes as JSON; the parity
# tests compare them against the loaded (native) engines case for case.
# Sets compare as sorted lists (their order is an interpreter detail);
# the marked-tracks walk compares as its ordered video-id list, since
# its order is part of the contract.

_RUN_CASES = [
    # (filtered_map, start, end)
    ([0, 1, 2, 3, 4], 1, 3),
    ([4, 3, 2, 1, 0], 4, 2),
    ([2, 3, 0, 4, 1], 3, 4),
    ([0, 1, 2], 2, 2),
    ([0, 1, 2], 9, 1),
    ([0, 1, 2], 1, 9),
    ([], 0, 5),
    ([5, 0, 4, 1, 3, 2], 4, 1),
]

_EXTEND_CASES = [
    # (anchor, filtered_map, marked, original)
    (None, [0, 1, 2], [5], 1),
    (7, [0, 1, 2], [5], 2),
    (0, [0, 1, 2], [9], 2),
    (0, [0, 1, 2, 3, 4], [9], 1),
    (3, [2, 3, 0, 4, 1], [8], 4),
    (1, [0, 1], [], 0),
    (0, [0, 1, 2], [0, 1, 2], 2),
    (2, [0, 2, 4], [1], 4),
]

_RANGE_CASES = [
    # (anchor, cursor, filtered_map, range_base)
    (None, 2, [0, 1, 2], [5]),
    (0, None, [0, 1, 2], [5]),
    (0, 2, [0, 1, 2], [5]),
    (3, 0, [0, 1, 2, 3], []),
    (1, 1, [0, 1, 2], []),
    (0, 4, [4, 3, 2, 1, 0], [2, 9]),
    (2, None, [0, 1, 2, 3], [0, 1]),
    (1, 3, [1, 0, 2], [8]),
]

_TOGGLE_CASES = [
    # (marked, original)
    ([1, 2], 3),
    ([1, 2], 1),
    ([], 7),
    ([5], 5),
    ([0, 1, 2], 2),
]

_SHIFT_CASES = [
    # (removed, marked, range_base, anchor, range_mode)
    (1, [0, 1, 2, 3], [2, 3], 2, False),
    (1, [0, 1, 2], [], None, False),
    (1, [0, 2], [], None, False),
    (2, [0, 4], [4], 2, True),
    (0, [1, 2], [2], 2, True),
    (1, [3], [3], None, True),
    (0, [0, 1, 2], [], None, False),
    (3, [0, 1, 2, 3, 4, 5], [1, 4, 5], 5, True),
    (0, [], [], None, False),
    (2, [2], [2], 2, True),
]

_MARKED_TRACKS_CASES = [
    # (titles in load order, marked originals, sort_column, sort_reverse)
    (["Alpha", "Bravo", "Charlie"], [], None, False),
    (["Alpha", "Bravo", "Charlie"], [0, 2], None, False),
    (["Charlie", "Alpha", "Bravo"], [0, 1, 2], "title", False),
    (["Alpha", "Alpha", "Bravo"], [0, 1, 2], "title", True),
    (["Charlie", "Alpha", "Bravo"], [0, 2], "title", False),
    (["Alpha", "Bravo", "Charlie"], [1], "duration", True),
    (["Bravo", "Charlie", "Alpha"], [0, 1], "artist", False),
]

_HIDDEN_CASES = [
    # (marked, filtered_map)
    ([1, 2], [0, 1, 2]),
    ([1, 3, 5], [0, 1, 2]),
    ([1, 2], []),
    ([], [0, 1]),
    ([1], [1, 1, 1]),
    ([0, 2, 4], [2]),
]

_CARRY_CASES = [
    # (old_keys, new_keys, marked, range_base, anchor, cursor)
    (["a", "b", "c"], ["a", "b", "c"], [0, 2], [2], 2, 0),
    (["a", "b", "c"], ["c", "a", "b"], [1], [], 1, 0),
    (["a", "b", "c"], ["a", "c"], [0, 1], [], None, None),
    (["a", "b"], ["a", "a", "b"], [1], [], None, None),
    (["a", "a", "b"], ["a", "b"], [0, 2], [], None, None),
    (["a", "b", "c"], ["a", "c"], [0], [], 1, 1),
    (["a"], ["a"], [], [], None, None),
    (["a", "b"], ["b", "a"], [0, 1], [], None, None),
    (["a", "b"], ["a"], [], [], None, None),
    (["q1", "q2", "q3", "q4"], ["q2", "q4", "q1"], [1, 3], [0, 3], 3, 2),
    (["k"], ["k", "k"], [0], [0], 0, 0),
    ([], [], [], [], None, None),
]


def _run_marked_tracks_case(case: tuple[list[str], list[int], str | None, bool]) -> list[str]:
    titles, marked, sort_column, sort_reverse = case
    tracks = _stamp([_track(title, f"v{i}", i, duration=100 + i) for i, title in enumerate(titles)])
    out = _marked_tracks(tracks, set(marked), sort_column, sort_reverse)
    return [t["video_id"] for t in out]


_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets.track_table import (
    _carry_selection_keys,
    _extend_marks,
    _hidden_marked_count,
    _marked_tracks,
    _range_marks,
    _shift_selection,
    _toggle_marks,
    _visible_run,
)

payload = json.load(sys.stdin)


def stamp(titles):
    tracks = [
        {"video_id": f"v{i}", "title": t, "artist": "Artist",
         "artists": [{"name": "Artist", "id": "1"}], "album": "Album",
         "duration": 100 + i, "_original_index": i}
        for i, t in enumerate(titles)
    ]
    return tracks


out = {
    "run": [sorted(_visible_run(m, s, e)) for m, s, e in payload["run_cases"]],
    "extend": [
        [a, sorted(marks)] for a, marks in (
            _extend_marks(a, m, set(mk), o)
            for a, m, mk, o in payload["extend_cases"]
        )
    ],
    "range": [
        None if r is None else sorted(r)
        for r in (
            _range_marks(a, c, m, frozenset(b))
            for a, c, m, b in payload["range_cases"]
        )
    ],
    "toggle": [sorted(_toggle_marks(set(m), o)) for m, o in payload["toggle_cases"]],
    "shift": [
        [sorted(marked), sorted(base), anchor, mode, bump]
        for marked, base, anchor, mode, bump in (
            _shift_selection(r, set(m), frozenset(b), a, mode)
            for r, m, b, a, mode in payload["shift_cases"]
        )
    ],
    "marked_tracks": [
        [t["video_id"] for t in _marked_tracks(stamp(titles), set(m), col, rev)]
        for titles, m, col, rev in payload["marked_tracks_cases"]
    ],
    "hidden": [
        _hidden_marked_count(set(m), mlist) for m, mlist in payload["hidden_cases"]
    ],
    "carry": [
        [sorted(marked), sorted(base), anchor, cursor, dropped]
        for marked, base, anchor, cursor, dropped in (
            _carry_selection_keys(ok, nk, set(m), frozenset(b), a, c)
            for ok, nk, m, b, a, c in payload["carry_cases"]
        )
    ],
}
json.dump(out, sys.stdout, sort_keys=True)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "run_cases": _RUN_CASES,
            "extend_cases": _EXTEND_CASES,
            "range_cases": _RANGE_CASES,
            "toggle_cases": _TOGGLE_CASES,
            "shift_cases": _SHIFT_CASES,
            "marked_tracks_cases": _MARKED_TRACKS_CASES,
            "hidden_cases": _HIDDEN_CASES,
            "carry_cases": _CARRY_CASES,
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
    """Run the loaded (native) engines over the same cases, spelled in
    the same JSON-friendly shapes the child prints."""
    run = [sorted(_visible_run(m, s, e)) for m, s, e in _RUN_CASES]
    extend = [
        [a, sorted(marks)]
        for a, marks in (_extend_marks(a, m, set(mk), o) for a, m, mk, o in _EXTEND_CASES)
    ]
    range_out = [
        None if r is None else sorted(r)
        for r in (_range_marks(a, c, m, frozenset(b)) for a, c, m, b in _RANGE_CASES)
    ]
    toggle = [sorted(_toggle_marks(set(m), o)) for m, o in _TOGGLE_CASES]
    shift = [
        [sorted(marked), sorted(base), anchor, mode, bump]
        for marked, base, anchor, mode, bump in (
            _shift_selection(r, set(m), frozenset(b), a, mode) for r, m, b, a, mode in _SHIFT_CASES
        )
    ]
    marked_tracks = [_run_marked_tracks_case(case) for case in _MARKED_TRACKS_CASES]
    hidden = [_hidden_marked_count(set(m), mlist) for m, mlist in _HIDDEN_CASES]
    carry = [
        [sorted(marked), sorted(base), anchor, cursor, dropped]
        for marked, base, anchor, cursor, dropped in (
            _carry_selection_keys(ok, nk, set(m), frozenset(b), a, c)
            for ok, nk, m, b, a, c in _CARRY_CASES
        )
    ]
    return {
        "run": run,
        "extend": extend,
        "range": range_out,
        "toggle": toggle,
        "shift": shift,
        "marked_tracks": marked_tracks,
        "hidden": hidden,
        "carry": carry,
    }


class TestNativeParity:
    """When the Rust extension is built, the native selection engines
    must produce exactly the same outcomes as the pure-Python
    references — same runs, same extend/range/toggle decisions, same
    post-removal shifts, same marked-track walks, same hidden counts
    and same occurrence-key carries — for every case, so the two
    implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        """Every selection outcome — the visible runs, the extend, range
        and toggle decisions, the post-removal shifts, the
        marked-tracks walks, the hidden-marks counts and the
        occurrence-key carries — must agree with the pure-Python
        reference case for case."""
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key in native:
            assert native[key] == fallback[key], (
                f"native and fallback track-table selection outcomes differ for {key}:\n"
                f"native:   {native[key]!r}\n"
                f"fallback: {fallback[key]!r}"
            )
