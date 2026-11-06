"""Tests for the track-table view engine (filter + sort).

``_filter_sort_positions`` is the engine behind ``TrackTable._rebuild_view``
and ``marked_tracks``: the filter pass, then the column sort, producing the
visible-row order. The pure-Python reference lives in
``ui/widgets/track_table.py``; when the Rust extension is built the same
engine runs natively (``_ytm_player_native.filter_sort_tracks``).

The unit tests below assert the engine's contract on whichever backend is
loaded. When the extension is built, the parity tests additionally compare
the native engine against the pure-Python reference (run in a child
interpreter with the extension import blocked) case for case, so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.widgets.track_table import (
    _compose_view_rows,
    _filter_sort_positions,
    _matches_filter,
    _row_values,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _track(
    title: str,
    artist: str = "",
    album: str = "",
    duration: int | str | None = None,
    index: int = 0,
    artists: list | None = None,
) -> dict:
    """Build a minimal track dict like the ones ``load_tracks`` stamps."""
    t: dict = {"_original_index": index, "video_id": f"v{index}", "title": title}
    if artists is not None:
        t["artists"] = artists
    elif artist:
        t["artist"] = artist
    if album:
        t["album"] = album
    if duration is not None:
        t["duration"] = duration
    return t


def _stamp(tracks: list[dict]) -> list[dict]:
    """Stamp load-order ``_original_index`` values like ``load_tracks``."""
    for i, t in enumerate(tracks):
        t["_original_index"] = i
    return tracks


# ── Filter ────────────────────────────────────────────────────────────


class TestMatchesFilter:
    def test_matches_title_case_insensitively(self):
        assert _matches_filter({"title": "Bohemian Rhapsody"}, "rhap")

    def test_matches_artist(self):
        assert _matches_filter({"title": "X", "artist": "Queen"}, "quee")

    def test_matches_album(self):
        assert _matches_filter({"title": "X", "album": "A Night at the Opera"}, "opera")

    def test_matches_artists_list(self):
        track = {"title": "X", "artists": [{"name": "Cara", "id": "1"}, {"name": "Dan", "id": "2"}]}
        assert _matches_filter(track, "dan")

    def test_no_match_returns_false(self):
        assert not _matches_filter({"title": "Alpha", "artist": "Beta", "album": "Gamma"}, "delta")

    def test_missing_fields_read_as_empty(self):
        assert not _matches_filter({}, "anything")
        assert _matches_filter({}, "")


class TestFilterPositions:
    def test_empty_query_keeps_every_position_in_load_order(self):
        tracks = _stamp([_track("C"), _track("A"), _track("B")])
        assert _filter_sort_positions(tracks, "", None, False) == [0, 1, 2]

    def test_query_keeps_only_matching_positions(self):
        tracks = _stamp([_track("Zebra", artist="Ann"), _track("apple"), _track("Banana")])
        assert _filter_sort_positions(tracks, "an", None, False) == [0, 2]

    def test_query_matching_nothing_returns_empty(self):
        tracks = _stamp([_track("Alpha")])
        assert _filter_sort_positions(tracks, "zzz", None, False) == []

    def test_empty_track_list(self):
        assert _filter_sort_positions([], "", None, False) == []
        assert _filter_sort_positions([], "q", "title", True) == []


# ── Sort ───────────────────────────────────────────────────────────────


class TestSortPositions:
    def test_title_sort_is_case_insensitive(self):
        # "Zebra" < "apple" < "banana" once lowercased.
        tracks = _stamp([_track("Zebra"), _track("apple"), _track("Banana")])
        assert _filter_sort_positions(tracks, "", "title", False) == [1, 2, 0]

    def test_title_sort_reverse(self):
        tracks = _stamp([_track("Zebra"), _track("apple"), _track("Banana")])
        assert _filter_sort_positions(tracks, "", "title", True) == [0, 2, 1]

    def test_artist_sort_uses_the_artists_list(self):
        tracks = _stamp(
            [
                _track("X", artists=[{"name": "Cara"}, {"name": "Dan"}]),
                _track("X", artist="Ann"),
                _track("X", artist="Bob"),
            ]
        )
        assert _filter_sort_positions(tracks, "", "artist", False) == [1, 2, 0]

    def test_album_sort(self):
        tracks = _stamp(
            [_track("X", album="Opus"), _track("X", album="alpha"), _track("X", album="Beta")]
        )
        assert _filter_sort_positions(tracks, "", "album", False) == [1, 2, 0]

    def test_duration_sort_parses_string_durations(self):
        # 50 < 100 < 205 ("3:25").
        tracks = _stamp(
            [_track("X", duration=100), _track("X", duration=50), _track("X", duration="3:25")]
        )
        assert _filter_sort_positions(tracks, "", "duration", False) == [1, 0, 2]

    def test_duration_sort_treats_missing_as_zero(self):
        tracks = _stamp([_track("X", duration=30), _track("X"), _track("X", duration=10)])
        assert _filter_sort_positions(tracks, "", "duration", False) == [1, 2, 0]

    def test_index_sort_orders_by_original_index_value(self):
        tracks = [
            {"_original_index": 2, "title": "A"},
            {"_original_index": 0, "title": "B"},
            {"_original_index": 1, "title": "C"},
        ]
        assert _filter_sort_positions(tracks, "", "index", False) == [1, 2, 0]

    def test_unknown_column_leaves_load_order(self):
        tracks = _stamp([_track("C"), _track("A"), _track("B")])
        assert _filter_sort_positions(tracks, "", "bogus", False) == [0, 1, 2]

    def test_none_column_leaves_load_order(self):
        tracks = _stamp([_track("C"), _track("A"), _track("B")])
        assert _filter_sort_positions(tracks, "", None, False) == [0, 1, 2]

    def test_ties_keep_input_order_ascending(self):
        tracks = _stamp(
            [_track("same", artist="A"), _track("same", artist="B"), _track("same", artist="C")]
        )
        assert _filter_sort_positions(tracks, "", "title", False) == [0, 1, 2]

    def test_ties_keep_input_order_with_reverse(self):
        # Python's `list.sort(reverse=True)` is stable: equal keys retain
        # the original order rather than being flipped.
        tracks = _stamp(
            [_track("same", artist="A"), _track("same", artist="B"), _track("same", artist="C")]
        )
        assert _filter_sort_positions(tracks, "", "title", True) == [0, 1, 2]

    def test_ties_keep_input_order_with_reverse_on_duration(self):
        tracks = _stamp(
            [_track("X", duration=10), _track("Y", duration=10), _track("Z", duration=10)]
        )
        assert _filter_sort_positions(tracks, "", "duration", True) == [0, 1, 2]


# ── Filter + sort combined ─────────────────────────────────────────────


class TestFilterAndSortCombined:
    def test_filter_applies_before_the_sort(self):
        # Visible rows: "banana" and "anchor" ("zebra" filtered out) —
        # descending puts banana before anchor, ascending the reverse.
        tracks = _stamp([_track("Banana"), _track("Anchor"), _track("zebra")])
        assert _filter_sort_positions(tracks, "an", "title", True) == [0, 1]
        assert _filter_sort_positions(tracks, "an", "title", False) == [1, 0]

    def test_sort_survives_a_query_that_filters_everything(self):
        tracks = _stamp([_track("Alpha"), _track("Beta")])
        assert _filter_sort_positions(tracks, "zzz", "title", False) == []


# ── Native / fallback parity ───────────────────────────────────────────
#
# When the Rust extension is built, the engine above IS the native one;
# each case below is also run in a child interpreter with the extension
# import blocked (pure-Python reference), and the two orders must agree
# exactly.

# (tracks, query, sort_column, sort_reverse)
_PARITY_CASES = [
    # No filter, no sort — identity.
    ([{"title": "C"}, {"title": "A"}, {"title": "B"}], "", None, False),
    # Each sort column, ascending and reversed.
    ([{"title": "Zebra"}, {"title": "apple"}, {"title": "Banana"}], "", "title", False),
    ([{"title": "Zebra"}, {"title": "apple"}, {"title": "Banana"}], "", "title", True),
    (
        [
            {"artist": "Cara", "title": "X"},
            {"artists": [{"name": "Ann"}, {"name": "Dan"}], "title": "X"},
            {"title": "X"},
        ],
        "",
        "artist",
        False,
    ),
    (
        [
            {"artist": "Cara", "title": "X"},
            {"artists": [{"name": "Ann"}, {"name": "Dan"}], "title": "X"},
            {"title": "X"},
        ],
        "",
        "artist",
        True,
    ),
    ([{"album": "Opus"}, {"album": "alpha"}, {"album": ""}], "", "album", False),
    ([{"album": "Opus"}, {"album": "alpha"}, {"album": ""}], "", "album", True),
    ([{"duration": 100}, {"duration": 50}, {}], "", "duration", False),
    ([{"duration": 100}, {"duration": 50}, {}], "", "duration", True),
    ([{"_original_index": 2}, {"_original_index": 0}, {"_original_index": 1}], "", "index", False),
    ([{"_original_index": 2}, {"_original_index": 0}, {"_original_index": 1}], "", "index", True),
    # Duration shapes: int, "M:SS", "H:MM:SS", duration_seconds, length.
    (
        [
            {"duration": "3:25"},
            {"duration_seconds": 61},
            {"length": "1:02:03"},
            {"duration": 10},
        ],
        "",
        "duration",
        False,
    ),
    # Ties keep input order, ascending and reversed.
    ([{"title": "same"}, {"title": "same"}, {"title": "same"}], "", "title", False),
    ([{"title": "same"}, {"title": "same"}, {"title": "same"}], "", "title", True),
    ([{"duration": 7}, {"duration": 7}, {}], "", "duration", True),
    # Filter-only, over title / artist / album.
    (
        [{"title": "Zebra", "artist": "Ann"}, {"title": "apple"}, {"title": "Banana"}],
        "an",
        None,
        False,
    ),
    (
        [{"title": "Zebra", "artist": "Ann"}, {"title": "apple"}, {"title": "Banana"}],
        "a",
        None,
        False,
    ),
    ([{"album": "A Night at the Opera"}, {"album": "Jazz"}], "opera", None, False),
    ([{"title": "Alpha"}], "zzz", None, False),
    # View-count pseudo-artist: filtered from the artist key/match like a
    # real artist never is.
    (
        [
            {
                "artists": [
                    {"name": "A", "id": "1"},
                    {"name": "358K views", "id": None},
                ],
                "views": "358K views",
                "title": "V",
            },
            {"artist": "B", "title": "W"},
        ],
        "",
        "artist",
        False,
    ),
    (
        [
            {
                "artists": [
                    {"name": "A", "id": "1"},
                    {"name": "358K views", "id": None},
                ],
                "views": "358K views",
                "title": "V",
            },
            {"artist": "B", "title": "W"},
        ],
        "views",
        None,
        False,
    ),
    # Empty-string artist falls through to the artists list; "Unknown" sorts
    # after real names but before "zzz".
    ([{"artist": "", "artists": [{"name": "Zed"}]}, {"artist": "Mel"}], "", "artist", False),
    ([{}, {"artist": "Mel"}], "", "artist", False),
    # Unicode: lowercased ordering and matching agree on both backends.
    ([{"title": "GRÜßE"}, {"title": "grün"}, {"title": "Zebras"}], "", "title", False),
    ([{"title": "GRÜßE"}, {"title": "grün"}, {"title": "Zebras"}], "grü", None, False),
    ([{"title": "夜に駆ける"}, {"title": "アイスクリーム"}], "駆", None, False),
    # Filter + sort together, both directions.
    ([{"title": "Zebra"}, {"title": "apple"}, {"title": "Banana"}], "a", "title", True),
    ([{"title": "Zebra"}, {"title": "apple"}, {"title": "Banana"}], "a", "title", False),
    # Unknown column name and None both mean load order.
    ([{"title": "C"}, {"title": "A"}], "", "bogus", False),
    ([{"title": "C"}, {"title": "A"}], "", None, True),
    # Empty list.
    ([], "", "title", False),
    ([], "q", "title", True),
]

# The child interpreter runs this program with the case list on stdin and
# prints the reference engine's orders as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets.track_table import _filter_sort_positions

cases = json.load(sys.stdin)
orders = [
    _filter_sort_positions(tracks, query, column, reverse)
    for tracks, query, column, reverse in cases
]
json.dump(orders, sys.stdout)
"""


def _fallback_orders(cases: list) -> list:
    """Run the pure-Python reference engine over *cases* in a child
    interpreter with the native extension import blocked."""
    payload = json.dumps(cases)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native view engine must
    produce exactly the same visible-row orders as the pure-Python
    reference — same filtering, same ordering, same tie behaviour — for
    every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native_orders = [
            _filter_sort_positions(tracks, query, column, reverse)
            for tracks, query, column, reverse in _PARITY_CASES
        ]
        fallback_orders = _fallback_orders(_PARITY_CASES)
        for case, native, fallback in zip(_PARITY_CASES, native_orders, fallback_orders):
            query, column, reverse = case[1], case[2], case[3]
            assert native == fallback, (
                f"native and fallback view orders differ for "
                f"query={query!r}, column={column!r}, reverse={reverse!r}"
            )


# ── Row assembly (_row_values) ────────────────────────────────────────


class TestRowValues:
    def test_full_assembly(self):
        track = {
            "_original_index": 3,
            "video_id": "vid9",
            "title": "Song",
            "artist": "Artist",
            "album": "Album",
            "duration": 125,
        }
        key, cells = _row_values(track, 0, True, True, set())
        assert key == "vid9_0"
        assert cells == [" ", "4", "Song", "Artist", "Album", "2:05"]

    def test_marked_row_carries_the_glyph(self):
        track = {"_original_index": 1, "title": "T"}
        _, cells = _row_values(track, 0, False, False, {1})
        assert cells[0] == "✓"
        _, cells = _row_values(track, 0, False, False, {2})
        assert cells[0] == " "

    def test_missing_fields_default(self):
        key, cells = _row_values({}, 5, True, True, set())
        assert key == "row_5_5"
        assert cells == [" ", "6", "Unknown", "Unknown", "", "--:--"]

    def test_missing_original_index_defaults_to_the_visible_slot(self):
        # The row-key suffix and the # cell both name the VISIBLE slot
        # when the stamp is missing.
        key, cells = _row_values({"title": "T", "video_id": "v"}, 7, True, False, set())
        assert key == "v_7"
        assert cells[1] == "8"

    def test_show_flags_shape_the_cells(self):
        track = {"_original_index": 0, "title": "T", "artist": "A", "album": "B", "duration": 5}
        _, cells = _row_values(track, 0, False, False, set())
        assert cells == [" ", "T", "A", "0:05"]
        assert len(cells) == 4

    def test_duration_shapes(self):
        assert _row_values({"duration": 61}, 0, False, False, set())[1][-1] == "1:01"
        assert _row_values({"duration": "3:25"}, 0, False, False, set())[1][-1] == "3:25"
        assert _row_values({"duration_seconds": 61}, 0, False, False, set())[1][-1] == "1:01"
        assert _row_values({}, 0, False, False, set())[1][-1] == "--:--"

    def test_artists_list_and_view_count_exclusion(self):
        track = {
            "artists": [{"name": "Cara"}, {"name": "358K views", "id": None}],
            "views": "358K views",
        }
        _, cells = _row_values(track, 0, False, False, set())
        assert cells[2] == "Cara"

    def test_rtl_title_is_isolated(self):
        _, cells = _row_values({"title": "שלום עולם"}, 0, False, False, set())
        assert cells[1].startswith("⁨")
        assert cells[1].endswith("⁩")

    def test_the_track_and_marked_set_are_not_mutated(self):
        track = {"_original_index": 0, "title": "T"}
        marked = {0}
        _row_values(track, 0, True, True, marked)
        assert track == {"_original_index": 0, "title": "T"}
        assert marked == {0}


# ── Row assembly (_compose_view_rows) ─────────────────────────────────


class TestComposeViewRows:
    def test_identity_load_shape(self):
        tracks = _stamp([_track("A"), _track("B")])
        tracks[1]["video_id"] = "v1"
        positions, rows = _compose_view_rows(tracks, "", None, False, True, True, set())
        assert positions == [0, 1]
        assert [key for key, _ in rows] == ["v0_0", "v1_1"]
        assert rows[0][1] == [" ", "1", "A", "Unknown", "", "--:--"]

    def test_filter_and_sort_fused_with_the_assembly(self):
        tracks = _stamp([_track("Zebra", artist="Ann"), _track("apple"), _track("Banana")])
        tracks[1]["video_id"] = "v1"
        tracks[2]["video_id"] = "v2"
        positions, rows = _compose_view_rows(tracks, "an", "title", True, True, True, set())
        assert positions == [0, 2]
        assert [key for key, _ in rows] == ["v0_0", "v2_1"]
        assert rows[0][1][2] == "Zebra"
        assert rows[1][1][2] == "Banana"

    def test_the_row_key_names_the_visible_slot_under_a_sort(self):
        tracks = _stamp([_track("B", duration=10), _track("A", duration=20)])
        tracks[1]["video_id"] = "v1"
        positions, rows = _compose_view_rows(tracks, "", "title", False, True, False, set())
        # A (position 1) shows first: its key suffix is the VISIBLE 0,
        # its # cell still the original position 2.
        assert positions == [1, 0]
        assert rows[0][0] == "v1_0"
        assert rows[0][1][1] == "2"

    def test_marks_painted_per_position(self):
        tracks = _stamp([_track("A"), _track("B")])
        _, rows = _compose_view_rows(tracks, "", "title", True, False, False, {1})
        assert [cells[0] for _, cells in rows] == ["✓", " "]

    def test_empty_tracks(self):
        assert _compose_view_rows([], "", "title", False, True, True, set()) == ([], [])


# ── Row-assembly native vs fallback parity ────────────────────────────
#
# The row engines travel over the same JSON bridge, with the cached
# bidi_mode verdict pinned to the same value in both interpreters (the
# child sets bidi._should_reorder before importing the engine; the
# parent monkeypatches the same global), so the render paths compare
# exactly — both the reorder-off pass and the reorder-on pass.

_ROW_PARITY_CASES = {
    "row": [
        # (track, index, show_index, show_album, marked)
        (
            {
                "_original_index": 3,
                "video_id": "vid9",
                "title": "Song",
                "artist": "Artist",
                "album": "Album",
                "duration": 125,
            },
            0,
            True,
            True,
            [],
        ),
        ({}, 5, True, True, []),
        ({"title": "T", "video_id": "v"}, 7, True, False, []),
        ({"_original_index": 1, "title": "T"}, 0, False, False, [1]),
        ({"_original_index": 1, "title": "T"}, 0, False, False, [2]),
        ({"duration": 61}, 0, False, False, []),
        ({"duration": "3:25"}, 0, False, False, []),
        ({"duration_seconds": 61}, 0, False, False, []),
        ({"length": "1:02:03"}, 0, False, False, []),
        ({"artists": [{"name": "Cara"}, {"name": "358K views", "id": None}]}, 0, False, False, []),
        # RTL cells: the isolate wrap (and, under the reorder flag, the
        # word-level reversal) both apply.
        ({"title": "שלום עולם", "artist": "سلام", "album": "مرحبا"}, 0, True, True, []),
        ({"title": "mix English עברית end"}, 0, True, True, []),
        # A non-str video_id renders through str() into the key.
        ({"video_id": 5, "title": "T"}, 0, True, True, []),
    ],
    "compose": [
        # (tracks, query, sort_column, reverse, show_index, show_album, marked)
        ([{"title": "A"}, {"title": "B"}], "", None, False, True, True, []),
        (
            [{"title": "Zebra", "artist": "Ann"}, {"title": "apple"}, {"title": "Banana"}],
            "an",
            "title",
            True,
            True,
            True,
            [],
        ),
        ([{"title": "B"}, {"title": "A"}], "", "title", False, True, False, []),
        ([{"title": "B"}, {"title": "A"}], "", "title", False, True, False, [1]),
        ([{"duration": 100}, {"duration": 50}, {}], "", "duration", True, True, True, [0, 2]),
        ([{"title": "שלום"}, {"title": "hello"}], "", "title", False, True, True, []),
        ([], "", "title", False, True, True, []),
    ],
}

_ROW_CHILD_PROG = """
import json, sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets import track_table
from ytm_player.utils import bidi

payload = json.load(sys.stdin)
bidi._should_reorder = payload["flag"]


def row_outcome(case):
    track, index, show_index, show_album, marked = case
    try:
        key, cells = track_table._row_values(track, index, show_index, show_album, set(marked))
        return {"ok": [key, cells]}
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def compose_outcome(case):
    tracks, query, column, reverse, show_index, show_album, marked = case
    try:
        positions, rows = track_table._compose_view_rows(
            tracks, query, column, reverse, show_index, show_album, set(marked)
        )
        return {"ok": [positions, [[key, cells] for key, cells in rows]]}
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


out = {
    "flag": payload["flag"],
    "row": [row_outcome(case) for case in payload["row"]],
    "compose": [compose_outcome(case) for case in payload["compose"]],
}
json.dump(out, sys.stdout)
"""


class TestRowAssemblyNativeParity:
    """When the Rust extension is built, the row-assembly engines must
    answer every case exactly like the pure-Python references run in a
    child interpreter with the extension import blocked — same row keys,
    same cell lists (the BiDi renders included, under both the
    reorder-off and reorder-on verdicts), same positions — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("flag", [False, True])
    def test_row_engines_match_fallback(self, monkeypatch, flag):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", flag)
        payload = {"flag": flag, **_ROW_PARITY_CASES}
        result = subprocess.run(
            [sys.executable, "-c", _ROW_CHILD_PROG],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        native_row = []
        for track, index, show_index, show_album, marked in _ROW_PARITY_CASES["row"]:
            try:
                key, cells = _row_values(track, index, show_index, show_album, set(marked))
                native_row.append({"ok": [key, cells]})
            except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
                native_row.append({"error": f"{type(exc).__name__}: {exc}"})

        native_compose = []
        for case in _ROW_PARITY_CASES["compose"]:
            tracks, query, column, reverse, show_index, show_album, marked = case
            try:
                positions, rows = _compose_view_rows(
                    tracks, query, column, reverse, show_index, show_album, set(marked)
                )
                native_compose.append({"ok": [positions, [[key, cells] for key, cells in rows]]})
            except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
                native_compose.append({"error": f"{type(exc).__name__}: {exc}"})

        for case, n, f in zip(_ROW_PARITY_CASES["row"], native_row, fallback["row"]):
            assert n == f, f"native and fallback _row_values differ for {case!r}:\n{n!r}\n{f!r}"
        for case, n, f in zip(_ROW_PARITY_CASES["compose"], native_compose, fallback["compose"]):
            assert n == f, (
                f"native and fallback _compose_view_rows differ for {case!r}:\n{n!r}\n{f!r}"
            )
