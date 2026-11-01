"""Tests for the Browse home-feed shelf classification engine.

``_is_event_shelf`` / ``_chart_sort_key`` / ``_split_events_and_charts``
are the engine behind the Charts tab's pill rows (the event-vs-chart
partition and the canonical priority sort), and ``_is_playlist_row`` /
``_playlist_rows`` are the engine behind the Playlists tab (which home-feed
rows are playlists, extracted in feed order with one row per playlist id).
The pure-Python reference lives in ``ui/pages/browse.py``; when the Rust
extension is built the same engines run natively
(``_ytm_player_native``).

The unit tests below assert the engines' contract on whichever backend is
loaded. When the extension is built, the parity tests additionally compare
the native engines against the pure-Python references (run in a child
interpreter with the extension import blocked) case for case, so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from ytm_player.ui.pages.browse import (
    _chart_sort_key,
    _is_event_shelf,
    _is_playlist_row,
    _playlist_rows,
    _split_events_and_charts,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _shelf(title: Any, playlist_id: str = "PL", **extra: Any) -> dict[str, Any]:
    return {"title": title, "playlistId": playlist_id, **extra}


# ── Event-shelf classification ────────────────────────────────────────


class TestIsEventShelf:
    @pytest.mark.parametrize(
        "title",
        [
            "Coachella 2026: Daily Top 100 Songs",
            "Coachella 2026: Trending 20",
            "A: B",
            "brand: ",
        ],
    )
    def test_brand_prefix_titles_are_events(self, title: str) -> None:
        assert _is_event_shelf(_shelf(title)) is True

    @pytest.mark.parametrize(
        "title",
        [
            "Trending 20 - United Kingdom",
            "Top 100 Songs",
            "Daily Top Music Videos",
            "Title:Sub",  # colon without the space is not the brand form
            "",
        ],
    )
    def test_country_chart_titles_are_not_events(self, title: str) -> None:
        assert _is_event_shelf(_shelf(title)) is False

    @pytest.mark.parametrize(
        "shelf",
        [
            {},  # missing title → the "" default
            {"title": None},  # str(None) == "None" — no colon-space
            {"title": 42},  # str(42) == "42"
            {"title": ["ab"]},  # str(["ab"]) == "['ab']" — no ": "
            {"playlistId": "PL"},  # no title key at all
        ],
        ids=["missing", "none", "int", "list", "no-key"],
    )
    def test_non_string_titles_classify_as_charts(self, shelf: dict) -> None:
        # The fallback str()-ifies whatever the title field holds; every
        # one of these shapes reads as a chart.
        assert _is_event_shelf(shelf) is False


# ── Chart-pill sort key ───────────────────────────────────────────────


class TestChartSortKey:
    @pytest.mark.parametrize(
        ("title", "rank"),
        [
            ("Top 100 Songs - United Kingdom", 0),
            ("Top 100 Songs", 0),
            ("Weekly Top Songs on Shorts - GB", 1),
            ("Trending 20 - United States", 2),
            ("Trending 20", 2),
        ],
    )
    def test_priority_prefixes_rank_first(self, title: str, rank: int) -> None:
        assert _chart_sort_key(_shelf(title)) == (rank, title)

    @pytest.mark.parametrize(
        "title",
        [
            "Top Music Videos - United Kingdom",
            "Daily Top 100 Songs",  # not a startswith of "Top 100 Songs"
            "The Top 100 Songs",  # prefix must anchor at the start
            "Weekly Top Songs (Shorts)",
            "New & hot",
            "",
        ],
    )
    def test_unranked_titles_share_rank_99(self, title: str) -> None:
        assert _chart_sort_key(_shelf(title)) == (99, title)

    def test_missing_title_ranks_99_with_empty_string(self) -> None:
        assert _chart_sort_key({"playlistId": "PL"}) == (99, "")

    def test_none_title_str_ifies_to_none(self) -> None:
        # str(None) == "None" — transcribed verbatim from the fallback.
        assert _chart_sort_key({"title": None}) == (99, "None")


# ── Event/chart partition ─────────────────────────────────────────────


class TestSplitEventsAndCharts:
    def test_events_keep_feed_order_and_charts_sort_by_priority(self) -> None:
        event_a = _shelf("Coachella 2026: Daily Top 100 Songs", playlist_id="PLevent")
        event_b = _shelf("Grammys 2026: Trending 20", playlist_id="PLevent2")
        videos = _shelf("Top Music Videos - United Kingdom", playlist_id="PLvideos")
        trending = _shelf("Trending 20 - United Kingdom", playlist_id="PLtrend")
        top = _shelf("Top 100 Songs - United Kingdom", playlist_id="PLtop")

        events, charts = _split_events_and_charts([top, event_a, videos, event_b, trending])

        # Events keep their original order (editorial intent)...
        assert events == [event_a, event_b]
        # ...charts sort by canonical priority, then unranked alphabetically.
        assert charts == [top, trending, videos]

    def test_equal_keys_keep_input_order(self) -> None:
        # Two distinct shelves with the SAME (rank, title) key — Python's
        # sorted() is stable, so ties keep input order, and the native
        # engine's stable sort matches that.
        first = _shelf("Top Music Videos - United Kingdom", playlist_id="PLa")
        second = _shelf("Top Music Videos - United Kingdom", playlistId="PLb")
        third = _shelf("Top Music Videos - United Kingdom", playlist_id="PLc")

        _events, charts = _split_events_and_charts([second, third, first])

        assert charts == [second, third, first]

    def test_returns_the_input_objects_by_reference(self) -> None:
        event = _shelf("Coachella 2026: Daily Top 100 Songs")
        chart = _shelf("Trending 20 - United Kingdom")

        events, charts = _split_events_and_charts([chart, event])

        assert events[0] is event
        assert charts[0] is chart

    def test_input_list_is_not_mutated(self) -> None:
        shelves = [
            _shelf("Trending 20 - United Kingdom"),
            _shelf("Coachella 2026: Trending 20"),
            _shelf("Top 100 Songs - United Kingdom"),
        ]
        before = list(shelves)

        _split_events_and_charts(shelves)

        assert shelves == before

    def test_empty_input(self) -> None:
        assert _split_events_and_charts([]) == ([], [])

    def test_all_events(self) -> None:
        shelves = [_shelf("A: x"), _shelf("B: y")]
        events, charts = _split_events_and_charts(shelves)
        assert events == shelves
        assert charts == []

    def test_all_charts(self) -> None:
        shelves = [_shelf("Trending 20"), _shelf("Top 100 Songs")]
        events, charts = _split_events_and_charts(shelves)
        assert events == []
        assert charts == [shelves[1], shelves[0]]

    def test_none_title_classifies_as_chart(self) -> None:
        # str(None) == "None" — no colon-space, so a None-titled shelf is
        # a chart whose sort title is "None".
        shelf = {"title": None, "playlistId": "PL"}
        events, charts = _split_events_and_charts([shelf])
        assert events == []
        assert charts == [shelf]


# ── Playlist-row classification ───────────────────────────────────────


class TestIsPlaylistRow:
    @pytest.mark.parametrize(
        "item",
        [
            {"title": "Mix", "playlistId": "RD123"},
            {"playlistId": "LM"},  # Liked Music
            {"playlistId": "LRSRrecap"},  # recap
            {"playlistId": "PLx", "videoId": None},
            {"playlistId": "PLx", "videoId": ""},  # falsy videoId = absent
            {"playlistId": "PLx", "videoId": 0},
            {"playlistId": "PLx", "videoId": []},
            {"playlistId": "PLx", "videoId": {}},
            {"playlistId": "PLx", "videoId": False},
        ],
        ids=[
            "mix",
            "liked",
            "recap",
            "vid-none",
            "vid-empty",
            "vid-zero",
            "vid-list",
            "vid-dict",
            "vid-false",
        ],
    )
    def test_playlist_rows_accept_falsy_videoids(self, item: dict) -> None:
        assert _is_playlist_row(item) is True

    @pytest.mark.parametrize(
        "item",
        [
            {"title": "Song", "videoId": "v1"},  # songs always carry videoId
            {"title": "Song", "videoId": "v1", "playlistId": "PL1"},  # some songs too
            {"title": "Song", "videoId": True},  # truthy videoId
            {"title": "Song", "videoId": 1},
            {"title": "Song", "videoId": [0]},
            {"title": "Album", "audioPlaylistId": "OLAK5uy_x"},  # albums never carry playlistId
            {"playlistId": None},
            {"playlistId": ""},
            {"playlistId": 5},
            {"playlistId": ["PL"]},
            {"title": "no id"},
            None,
            "junk",
            42,
            ["playlistId"],
        ],
        ids=[
            "song",
            "song-with-playlist",
            "vid-true",
            "vid-one",
            "vid-truthy-list",
            "album",
            "pid-none",
            "pid-empty",
            "pid-int",
            "pid-list",
            "no-id",
            "none",
            "str",
            "int",
            "list",
        ],
    )
    def test_non_playlist_rows_are_rejected(self, item: Any) -> None:
        assert _is_playlist_row(item) is False


# ── Playlist-row extraction ───────────────────────────────────────────


class TestPlaylistRows:
    def _feed(self) -> list[Any]:
        mix = {"title": "Discover Mix", "playlistId": "RDdis"}
        user = {"title": "Road trip", "playlistId": "PLroad"}
        song = {"title": "Song A", "videoId": "vA", "playlistId": "PLsong"}
        album = {"title": "Album", "browseId": "MPREb1", "audioPlaylistId": "OLAK5"}
        return [
            {"title": "Quick picks", "contents": [song, mix]},
            "not a shelf",
            {"title": "Listen again", "contents": [user, mix]},  # duplicate id
            {"title": "Albums", "contents": [album]},
            {"title": "Artists", "contents": [{"title": "A", "browseId": "UC1"}]},
            {"title": "Mixed", "contents": [None, "junk", {"playlistId": None}, {"playlistId": 5}]},
            {"title": "No contents"},
            {"title": "Contents not a list", "contents": {"playlistId": "PLx"}},
            {"title": "Contents None", "contents": None},
        ]

    def test_keeps_feed_order_one_row_per_id_first_occurrence_wins(self) -> None:
        rows = _playlist_rows(self._feed())
        # The song (videoId + playlistId) is not a playlist row; the mix
        # and the user playlist are, in feed order, once each.
        assert [r["playlistId"] for r in rows] == ["RDdis", "PLroad"]

    def test_rows_are_the_input_objects(self) -> None:
        feed = self._feed()
        rows = _playlist_rows(feed)
        # The mix row of the first shelf (index 0, second item) is the one
        # kept — the duplicate in the later shelf is dropped.
        assert rows[0] is feed[0]["contents"][1]
        assert rows[1] is feed[2]["contents"][0]

    def test_input_feed_is_not_mutated(self) -> None:
        feed = self._feed()
        before = json.loads(json.dumps(feed))
        _playlist_rows(feed)
        assert json.loads(json.dumps(feed)) == before

    def test_empty_and_malformed_feeds(self) -> None:
        assert _playlist_rows([]) == []
        assert _playlist_rows([None, 5, "x", {}, {"contents": []}]) == []

    def test_dedup_across_many_shelves(self) -> None:
        a = {"playlistId": "PL1"}
        b = {"playlistId": "PL2"}
        c = {"playlistId": "PL1"}  # duplicate of a in a later shelf
        shelves = [
            {"contents": [a]},
            {"contents": [b, c]},
            {"contents": [{"playlistId": "PL1"}]},  # equal id, fresh dict
        ]
        rows = _playlist_rows(shelves)
        assert rows == [a, b]
        assert rows[0] is a  # the FIRST occurrence's object wins


# ── Native / fallback parity ──────────────────────────────────────────
#
# The child interpreter runs the pure-Python reference engines with the
# extension import blocked and prints the outcomes as JSON; the parity
# tests compare them against the loaded (native) engine case for case.

_EVENT_CASES = [
    {"title": "Coachella 2026: Daily Top 100 Songs"},
    {"title": "Trending 20 - United Kingdom"},
    {"title": "A: B"},
    {"title": "Title:Sub"},
    {},
    {"title": None},
    {"title": 42},
    {"title": ""},
    {"playlistId": "PL"},
]

_SORT_KEY_CASES = [
    {"title": "Top 100 Songs - United Kingdom"},
    {"title": "Weekly Top Songs on Shorts - GB"},
    {"title": "Trending 20 - United States"},
    {"title": "Top Music Videos - United Kingdom"},
    {"title": "Daily Top 100 Songs"},
    {"title": "The Top 100 Songs"},
    {"title": "New & hot"},
    {"title": ""},
    {},
    {"title": None},
    {"title": 42},
]

_SPLIT_CASES = [
    [
        {"title": "Top 100 Songs - GB", "playlistId": "PLtop"},
        {"title": "Coachella 2026: Daily Top 100 Songs", "playlistId": "PLevent"},
        {"title": "Top Music Videos - GB", "playlistId": "PLvideos"},
        {"title": "Trending 20 - GB", "playlistId": "PLtrend"},
    ],
    # Stability: equal (rank, title) keys keep input order.
    [
        {"title": "Top Music Videos - GB", "playlistId": "PLa"},
        {"title": "Top Music Videos - GB", "playlistId": "PLb"},
        {"title": "Top Music Videos - GB", "playlistId": "PLc"},
    ],
    # Two events keep feed order; charts sort alphabetically within rank 99.
    [
        {"title": "B: second", "playlistId": "PLeB"},
        {"title": "A: first", "playlistId": "PLeA"},
        {"title": "Zed chart", "playlistId": "PLz"},
        {"title": "Ana chart", "playlistId": "PLa2"},
    ],
    [],
    [{"title": "Only event: x"}],
    [{"title": "Only chart"}],
    # str() warts: None/numeric/missing titles classify as charts.
    [
        {"title": None, "playlistId": "PLnone"},
        {"title": 42, "playlistId": "PL42"},
        {"playlistId": "PLmissing"},
        {"title": "Trending 20", "playlistId": "PLtrend"},
    ],
    # Unicode titles exercise the code-point-order tie-break.
    [
        {"title": "Ünïcode chart", "playlistId": "PLu"},
        {"title": "Ana chart", "playlistId": "PLa"},
        {"title": "Änother chart", "playlistId": "PLa2"},
    ],
]

_ROW_CASES = [
    [
        {
            "title": "Quick picks",
            "contents": [
                {"title": "Song A", "videoId": "vA", "playlistId": "PLsong"},
                {"title": "Mix", "playlistId": "RDmix"},
            ],
        },
        "not a shelf",
        {
            "title": "Listen again",
            "contents": [
                {"title": "Road trip", "playlistId": "PLroad"},
                {"title": "Mix", "playlistId": "RDmix"},  # duplicate
            ],
        },
        {"title": "Albums", "contents": [{"title": "A", "audioPlaylistId": "OLAK5"}]},
        {
            "title": "Mixed",
            "contents": [None, "junk", {"playlistId": None}, {"playlistId": 5}, {"playlistId": ""}],
        },
        {"title": "No contents"},
        {"title": "Contents not a list", "contents": {"playlistId": "PLx"}},
    ],
    [],
    [None, 5, "x", {}, {"contents": []}],
    # videoId truthiness variants against a valid playlistId.
    [
        {
            "contents": [
                {"playlistId": "PLa", "videoId": None},
                {"playlistId": "PLb", "videoId": ""},
                {"playlistId": "PLc", "videoId": 0},
                {"playlistId": "PLd", "videoId": []},
                {"playlistId": "PLe", "videoId": {}},
                {"playlistId": "PLf", "videoId": False},
                {"playlistId": "PLg", "videoId": True},
                {"playlistId": "PLh", "videoId": 1},
                {"playlistId": "PLi", "videoId": [0]},
            ]
        },
    ],
    # Same id repeated across and within shelves; first occurrence wins.
    [
        {"contents": [{"playlistId": "PL1"}, {"playlistId": "PL1"}]},
        {"contents": [{"playlistId": "PL2"}, {"playlistId": "PL1"}]},
        {"contents": [{"playlistId": "PL3", "videoId": "v"}]},
    ],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.browse import (
    _chart_sort_key,
    _is_event_shelf,
    _is_playlist_row,
    _playlist_rows,
    _split_events_and_charts,
)

payload = json.load(sys.stdin)

# Split outcomes keep the shelf objects themselves; compare them as the
# dicts they are (the payload cases are JSON round-trips, so every field
# survives) and keep the event/chart blocks separate so the partition
# point is pinned, not just the concatenation.
out = {
    "event": [_is_event_shelf(s) for s in payload["event_cases"]],
    "sort_key": [list(_chart_sort_key(s)) for s in payload["sort_key_cases"]],
    "split": [
        [[dict(e) for e in events], [dict(c) for c in charts]]
        for events, charts in (
            _split_events_and_charts(s) for s in payload["split_cases"]
        )
    ],
    "is_row": [_is_playlist_row(i) for i in payload["row_gate_cases"]],
    "rows": [[dict(r) for r in _playlist_rows(f)] for f in payload["row_cases"]],
}
json.dump(out, sys.stdout, sort_keys=True)
"""

_ROW_GATE_CASES = [
    {"title": "Mix", "playlistId": "RD123"},
    {"title": "Song", "videoId": "v1"},
    {"title": "Song", "videoId": "v1", "playlistId": "PL1"},
    {"playlistId": None},
    {"playlistId": ""},
    {"playlistId": 5},
    {"title": "no id"},
    None,
    "junk",
    {"playlistId": "PLx", "videoId": None},
    {"playlistId": "PLx", "videoId": ""},
    {"playlistId": "PLx", "videoId": 0},
    {"playlistId": "PLx", "videoId": []},
    {"playlistId": "PLx", "videoId": {}},
    {"playlistId": "PLx", "videoId": False},
    {"playlistId": "PLx", "videoId": True},
    {"playlistId": "PLx", "videoId": 1},
    {"playlistId": "PLx", "videoId": [0]},
]


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "event_cases": _EVENT_CASES,
            "sort_key_cases": _SORT_KEY_CASES,
            "split_cases": _SPLIT_CASES,
            "row_gate_cases": _ROW_GATE_CASES,
            "row_cases": _ROW_CASES,
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
    split = []
    for case in _SPLIT_CASES:
        events, charts = _split_events_and_charts(case)
        split.append([[dict(e) for e in events], [dict(c) for c in charts]])
    return {
        "event": [_is_event_shelf(s) for s in _EVENT_CASES],
        "sort_key": [list(_chart_sort_key(s)) for s in _SORT_KEY_CASES],
        "split": split,
        "is_row": [_is_playlist_row(i) for i in _ROW_GATE_CASES],
        "rows": [[dict(r) for r in _playlist_rows(f)] for f in _ROW_CASES],
    }


class TestNativeParity:
    """When the Rust extension is built, the native classification engines
    must produce exactly the same outcomes as the pure-Python references —
    same event/chart partition, same priority order, same playlist-row
    gate and extraction — for every case, so the two implementations cannot
    drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        """Every engine outcome — the event/chart gate, the priority sort
        key, the partition (with the event and chart blocks kept separate
        so the split point is pinned, not just the concatenation), the
        playlist-row gate and the deduplicated extraction — must agree
        with the pure-Python reference case for case."""
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key in native:
            assert native[key] == fallback[key], (
                f"native and fallback browse-engine outcomes differ for {key}:\n"
                f"native:   {native[key]!r}\n"
                f"fallback: {fallback[key]!r}"
            )
