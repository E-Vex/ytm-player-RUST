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

from ytm_player.ui.widgets.track_table import _filter_sort_positions, _matches_filter

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
