"""Tests for the Browse page's rendering engines.

``_browse_item_route`` (the routing decision behind
``navigate_browse_item``), ``_foryou_shelf_rows`` (the For You shelf
view model), ``_charts_raw_shelves`` / ``_charts_pill_rows`` /
``_charts_country_label`` / ``_charts_event_row_visible`` (the Charts
tab's shelf extraction, pill view model, country line and visibility
gate) and ``_new_release_label`` / ``_playlist_row_label`` /
``_subscription_label`` (the Releases / Playlists / Subscriptions row
labels) are the rendering path behind the page's sections. The
pure-Python references live in ``ui/pages/browse.py``; when the Rust
extension is built the same engines run natively
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
from typing import Any

import pytest

from ytm_player.ui.pages.browse import (
    _browse_item_route,
    _charts_country_label,
    _charts_event_row_visible,
    _charts_pill_rows,
    _charts_raw_shelves,
    _foryou_shelf_rows,
    _new_release_label,
    _playlist_row_label,
    _subscription_label,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Item routing ──────────────────────────────────────────────────────


class TestBrowseItemRoute:
    @pytest.mark.parametrize("tag", ["song", "video", "flat_song", "SONG", "Video"])
    def test_track_tags_route_to_play(self, tag: str) -> None:
        assert _browse_item_route({"resultType": tag, "videoId": "v1"}) == ("play", "")

    def test_a_bare_video_id_routes_to_play(self) -> None:
        # No resultType at all — the videoId alone decides (the route a
        # For You shelf row takes).
        assert _browse_item_route({"title": "Song", "videoId": "v1"}) == ("play", "")

    def test_the_type_key_stands_in_for_result_type(self) -> None:
        assert _browse_item_route({"type": "artist", "browseId": "UC1"}) == ("artist", "UC1")

    @pytest.mark.parametrize("tag", ["album", "single", "ALBUM"])
    def test_album_tags_route_by_browse_id(self, tag: str) -> None:
        assert _browse_item_route({"resultType": tag, "browseId": "MPREb1"}) == ("album", "MPREb1")

    def test_album_tag_without_browse_id_routes_nowhere(self) -> None:
        assert _browse_item_route({"resultType": "album"}) == ("", "")

    def test_artist_routes_by_browse_id(self) -> None:
        assert _browse_item_route({"resultType": "artist", "browseId": "UC1"}) == ("artist", "UC1")

    def test_artist_without_browse_id_routes_nowhere(self) -> None:
        assert _browse_item_route({"resultType": "artist"}) == ("", "")

    def test_playlist_tag_prefers_playlist_id(self) -> None:
        item = {"resultType": "playlist", "playlistId": "PL1", "browseId": "MPREb2"}
        assert _browse_item_route(item) == ("playlist", "PL1")

    def test_playlist_tag_falls_back_to_browse_id(self) -> None:
        assert _browse_item_route({"resultType": "playlist", "browseId": "MPREb2"}) == (
            "playlist",
            "MPREb2",
        )

    def test_playlist_tag_without_any_id_routes_nowhere(self) -> None:
        assert _browse_item_route({"resultType": "playlist"}) == ("", "")

    def test_untagged_playlist_id_routes_to_playlist(self) -> None:
        # Shelves like "Mixed for you" carry a playlistId but no tag.
        assert _browse_item_route({"title": "Discover Mix", "playlistId": "RD1"}) == (
            "playlist",
            "RD1",
        )

    def test_audio_playlist_id_stands_in_for_playlist_id(self) -> None:
        # Albums from the feed carry audioPlaylistId, never playlistId —
        # and usually a browseId too, which wins the album fallback.
        assert _browse_item_route({"audioPlaylistId": "OLAK5"}) == ("playlist", "OLAK5")

    def test_remaining_browse_id_routes_to_album(self) -> None:
        assert _browse_item_route({"title": "Album", "browseId": "MPREb1"}) == ("album", "MPREb1")

    def test_item_with_no_ids_routes_nowhere(self) -> None:
        assert _browse_item_route({"title": "Nothing"}) == ("", "")

    def test_a_video_id_beats_the_playlist_tag(self) -> None:
        # A playlist-tagged row that also carries a videoId plays — the
        # track branch is checked first, exactly like the reference.
        item = {"resultType": "playlist", "playlistId": "PL1", "videoId": "v1"}
        assert _browse_item_route(item) == ("play", "")

    def test_falsy_ids_do_not_route(self) -> None:
        assert _browse_item_route({"resultType": "album", "browseId": None}) == ("", "")
        assert _browse_item_route({"playlistId": ""}) == ("", "")


# ── For You shelf view model ──────────────────────────────────────────


class TestForyouShelfRows:
    def _song(self, title: str = "Song", **extra: Any) -> dict[str, Any]:
        return {"title": title, "videoId": "v", "artists": [], **extra}

    def test_full_rows_with_artist_and_description(self) -> None:
        item = self._song("Song", artist="Artist", description="Fresh picks")
        title, rows = _foryou_shelf_rows({"title": "Quick picks", "contents": [item]})
        assert title == "Quick picks"
        assert len(rows) == 1
        display, kept = rows[0]
        assert display == "Song  Artist - Fresh picks"
        assert kept is item

    def test_unknown_artist_is_dropped_from_the_subtitle(self) -> None:
        # extract_artist answers "Unknown" for a bare item — the label
        # shows the bare title, not "Song  Unknown".
        title, rows = _foryou_shelf_rows({"title": "Shelf", "contents": [self._song()]})
        assert rows[0][0] == "Song"

    def test_description_alone_builds_the_subtitle(self) -> None:
        item = self._song("Mix", description="Mixed for you")
        _title, rows = _foryou_shelf_rows({"title": "Shelf", "contents": [item]})
        assert rows[0][0] == "Mix  Mixed for you"

    def test_missing_title_defaults_to_unknown(self) -> None:
        _title, rows = _foryou_shelf_rows({"title": "Shelf", "contents": [{"artists": []}]})
        assert rows[0][0] == "Unknown"

    def test_shelf_title_defaults_to_recommendations(self) -> None:
        title, rows = _foryou_shelf_rows({"contents": [self._song()]})
        assert title == "Recommendations"
        assert len(rows) == 1

    @pytest.mark.parametrize("contents", [[], None, ""], ids=["empty", "none", "blank"])
    def test_falsy_contents_answer_no_rows(self, contents: Any) -> None:
        title, rows = _foryou_shelf_rows({"title": "Shelf", "contents": contents})
        assert title == "Shelf"
        assert rows == []

    def test_missing_contents_answer_no_rows(self) -> None:
        _title, rows = _foryou_shelf_rows({"title": "Shelf"})
        assert rows == []

    def test_rows_cap_at_eight_items(self) -> None:
        contents = [self._song(f"S{i}") for i in range(12)]
        _title, rows = _foryou_shelf_rows({"title": "Shelf", "contents": contents})
        assert len(rows) == 8
        assert [r[1] for r in rows] == contents[:8]

    def test_long_lines_truncate_to_eighty(self) -> None:
        item = self._song("T" * 100, artist="A" * 60)
        _title, rows = _foryou_shelf_rows({"title": "Shelf", "contents": [item]})
        display = rows[0][0]
        assert len(display) == 80
        assert display.endswith("…")


# ── Charts tab ────────────────────────────────────────────────────────


class TestChartsRawShelves:
    def test_daily_weekly_and_videos_in_order(self) -> None:
        daily = {"title": "Daily", "playlistId": "PLd"}
        weekly = {"title": "Weekly", "playlistId": "PLw"}
        videos = {"title": "Videos", "playlistId": "PLv"}
        payload = {"daily": [daily], "weekly": [weekly], "videos": [videos]}
        assert _charts_raw_shelves(payload) == [daily, weekly, videos]

    def test_non_dict_shelves_are_dropped(self) -> None:
        shelf = {"title": "Daily"}
        payload = {"daily": [shelf, "junk", None, 5]}
        assert _charts_raw_shelves(payload) == [shelf]

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"daily": None},
            {"daily": "not a list"},
            {"daily": {}, "weekly": 3},
            None,  # the service's failure marker
            ["not", "a", "dict"],
        ],
        ids=["empty", "none", "string", "junk-values", "none-payload", "list-payload"],
    )
    def test_payloads_without_dict_shelves_answer_none(self, payload: Any) -> None:
        assert _charts_raw_shelves(payload) == []

    def test_extra_keys_are_ignored(self) -> None:
        assert _charts_raw_shelves({"artists": [{"title": "A"}]}) == []


class TestChartsPillRows:
    def test_labels_are_cleaned_and_classes_mark_event_and_active(self) -> None:
        event = {"title": "Coachella 2026: Daily Top 100 Songs", "playlistId": "PLe"}
        top = {"title": "Top 100 Songs - United Kingdom", "playlistId": "PLt"}
        videos = {"title": "Top Music Videos - United Kingdom", "playlistId": "PLv"}
        rows = _charts_pill_rows([event, top, videos], 1, 1)
        assert rows == [
            ("Coachella 2026: Daily Top 100", "shelf-pill event"),
            ("Top 100 Songs", "shelf-pill active"),
            ("Top Music Videos", "shelf-pill"),
        ]

    def test_the_active_pill_can_be_an_event(self) -> None:
        event = {"title": "Coachella 2026: Daily Top 100 Songs", "playlistId": "PLe"}
        top = {"title": "Top 100 Songs - United Kingdom", "playlistId": "PLt"}
        rows = _charts_pill_rows([event, top], 1, 0)
        assert rows[0] == ("Coachella 2026: Daily Top 100", "shelf-pill event active")
        assert rows[1] == ("Top 100 Songs", "shelf-pill")

    def test_missing_or_falsy_titles_fall_back_to_shelf_n(self) -> None:
        rows = _charts_pill_rows([{}, {"title": None}, {"title": ""}], 0, 0)
        assert [label for label, _classes in rows] == ["Shelf 1", "Shelf 2", "Shelf 3"]

    def test_non_dict_shelves_fall_back_to_shelf_n(self) -> None:
        rows = _charts_pill_rows(["junk", None], 0, 0)
        assert [label for label, _classes in rows] == ["Shelf 1", "Shelf 2"]

    def test_no_active_pill_when_active_daily_is_out_of_range(self) -> None:
        rows = _charts_pill_rows([{"title": "Top 100 Songs"}], 0, -1)
        assert rows[0][1] == "shelf-pill"

    def test_empty_dailies_answer_no_rows(self) -> None:
        assert _charts_pill_rows([], 0, 0) == []


class TestChartsCountryLabel:
    def test_the_exact_line(self) -> None:
        assert _charts_country_label("ZZ") == "Country: ZZ    (press 'c' to change)"
        assert _charts_country_label("GB") == "Country: GB    (press 'c' to change)"


class TestChartsEventRowVisible:
    @pytest.mark.parametrize(
        ("event_count", "term_width", "expected"),
        [
            (1, 80, True),
            (1, 120, True),
            (1, 79, False),
            (0, 200, False),
            (2, 80, True),
        ],
    )
    def test_the_gate(self, event_count: int, term_width: int, expected: bool) -> None:
        assert _charts_event_row_visible(event_count, term_width, 80) is expected


# ── Releases / Playlists / Subscriptions row labels ───────────────────


class TestNewReleaseLabel:
    def test_full_assembly(self) -> None:
        album = {
            "title": "Album",
            "artists": [{"name": "Artist", "id": "UC1"}],
            "type": "Album",
            "year": 2024,
        }
        assert _new_release_label(album) == "Album by Artist (Album, 2024)"

    def test_unknown_artist_drops_the_by_clause(self) -> None:
        assert _new_release_label({"title": "Album", "type": "EP"}) == "Album (EP)"

    def test_meta_clause_needs_at_least_one_part(self) -> None:
        assert _new_release_label({"title": "Album"}) == "Album"

    def test_year_alone_builds_the_meta_clause(self) -> None:
        assert _new_release_label({"title": "Album", "year": 2023}) == "Album (2023)"

    def test_missing_title_defaults(self) -> None:
        assert _new_release_label({"artists": [{"name": "A", "id": "i"}]}) == "Unknown Album by A"

    def test_long_labels_truncate_to_eighty(self) -> None:
        album = {"title": "T" * 100, "artists": [{"name": "A" * 60, "id": "i"}]}
        label = _new_release_label(album)
        assert len(label) == 80
        assert label.endswith("…")


class TestPlaylistRowLabel:
    def test_description_wins_over_count(self) -> None:
        row = {"title": "Mix", "description": "Mixed for you", "count": "50"}
        assert _playlist_row_label(row) == "Mix (Mixed for you)"

    def test_count_builds_the_songs_clause(self) -> None:
        assert _playlist_row_label({"title": "Road trip", "count": "12"}) == "Road trip (12 songs)"

    def test_missing_title_falls_back_to_untitled(self) -> None:
        assert _playlist_row_label({}) == "Untitled playlist"
        assert _playlist_row_label({"title": ""}) == "Untitled playlist"

    def test_bare_title_without_description_or_count(self) -> None:
        assert _playlist_row_label({"title": "Liked Music", "description": ""}) == "Liked Music"

    def test_long_labels_truncate_to_eighty(self) -> None:
        label = _playlist_row_label({"title": "P" * 100, "count": "5"})
        assert len(label) == 80
        assert label.endswith("…")


class TestSubscriptionLabel:
    def test_with_subscribers(self) -> None:
        assert _subscription_label({"artist": "Sia", "subscribers": "3.88M"}) == (
            "Sia  (3.88M subscribers)"
        )

    def test_without_subscribers(self) -> None:
        assert _subscription_label({"artist": "Nobody"}) == "Nobody"

    def test_missing_name_defaults(self) -> None:
        assert _subscription_label({}) == "Unknown artist"

    def test_long_labels_truncate_to_eighty(self) -> None:
        label = _subscription_label({"artist": "A" * 100, "subscribers": "1M"})
        assert len(label) == 80
        assert label.endswith("…")


# ── Native / fallback parity ──────────────────────────────────────────
#
# The child interpreter runs the pure-Python reference engines with the
# extension import blocked and prints the outcomes as JSON; the parity
# tests compare them against the loaded (native) engines case for case.

_ROUTE_CASES = [
    {"resultType": "song", "videoId": "v1"},
    {"resultType": "video"},
    {"resultType": "flat_song"},
    {"resultType": "SONG", "videoId": "v1"},
    {"type": "artist", "browseId": "UC1"},
    {"resultType": "album", "browseId": "MPREb1"},
    {"resultType": "single", "browseId": "MPREb1"},
    {"resultType": "album"},
    {"resultType": "artist"},
    {"resultType": "playlist", "playlistId": "PL1", "browseId": "MPREb2"},
    {"resultType": "playlist", "browseId": "MPREb2"},
    {"resultType": "playlist"},
    {"title": "Discover Mix", "playlistId": "RD1"},
    {"audioPlaylistId": "OLAK5"},
    {"title": "Album", "browseId": "MPREb1"},
    {"title": "Nothing"},
    {"resultType": "playlist", "playlistId": "PL1", "videoId": "v1"},
    {"resultType": "album", "browseId": None},
    {"playlistId": ""},
    {"resultType": "episode"},
    {"video_id": "v2"},  # the snake_case spelling get_video_id reads
    {},
]

_FORYOU_CASES = [
    {
        "title": "Quick picks",
        "contents": [
            {"title": "Song A", "videoId": "vA", "artists": [{"name": "Artist", "id": "UC1"}]},
            {"title": "Mix", "playlistId": "RD1"},
        ],
    },
    {"title": "Shelf", "contents": [{"title": "S", "description": "Fresh picks", "artists": []}]},
    {"contents": [{"artists": []}]},  # missing shelf and item titles
    {"title": "No contents"},
    {"title": "None contents", "contents": None},
    {"title": "Empty", "contents": []},
    {
        "title": "Deep",
        "contents": [
            {"title": f"S{i}", "artist": f"A{i}", "description": f"D{i}"} for i in range(11)
        ],
    },
    {
        "title": "Long",
        "contents": [{"title": "T" * 100, "artist": "A" * 60}],
    },
]

_RAW_CASES = [
    {
        "daily": [{"title": "Daily", "playlistId": "PLd"}],
        "weekly": [{"title": "Weekly", "playlistId": "PLw"}],
        "videos": [{"title": "Videos", "playlistId": "PLv"}],
    },
    {"daily": [{"title": "D"}, "junk", None, 5]},
    {"daily": None},
    {"daily": "not a list"},
    {"artists": [{"title": "A"}]},
    {},
    None,
    ["not", "a", "dict"],
]

_PILL_CASES = [
    (
        [
            {"title": "Coachella 2026: Daily Top 100 Songs", "playlistId": "PLe"},
            {"title": "Top 100 Songs - United Kingdom", "playlistId": "PLt"},
            {"title": "Top Music Videos - United Kingdom", "playlistId": "PLv"},
        ],
        1,
        1,
    ),
    ([{}, {"title": None}, {"title": ""}], 0, 0),
    (["junk", None], 0, 0),
    ([{"title": "Trending 20 - United States"}], 0, 0),
    ([], 0, 0),
    ([{"title": "A: x"}, {"title": "B: y"}], 2, 0),
]

_RELEASE_CASES = [
    {"title": "Album", "artists": [{"name": "Artist", "id": "UC1"}], "type": "Album", "year": 2024},
    {"title": "Album", "type": "EP"},
    {"title": "Album"},
    {"title": "Album", "year": 2023},
    {"artists": [{"name": "A", "id": "i"}]},
    {"title": "T" * 100, "artists": [{"name": "A" * 60, "id": "i"}]},
    {"title": "Album", "artists": [], "type": "", "year": ""},
]

_PLAYLIST_CASES = [
    {"title": "Mix", "description": "Mixed for you", "count": "50"},
    {"title": "Road trip", "count": "12"},
    {},
    {"title": ""},
    {"title": "Liked Music", "description": ""},
    {"title": "P" * 100, "count": "5"},
]

_SUBSCRIPTION_CASES = [
    {"artist": "Sia", "subscribers": "3.88M"},
    {"artist": "Nobody"},
    {},
    {"artist": "A" * 100, "subscribers": "1M"},
    {"artist": "X", "subscribers": ""},
]

_VISIBLE_CASES = [(1, 80, 80), (1, 79, 80), (0, 200, 80), (2, 120, 100), (1, 100, 120)]

_COUNTRY_CASES = ["ZZ", "GB", ""]


_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.browse import (
    _browse_item_route,
    _charts_country_label,
    _charts_event_row_visible,
    _charts_pill_rows,
    _charts_raw_shelves,
    _foryou_shelf_rows,
    _new_release_label,
    _playlist_row_label,
    _subscription_label,
)

payload = json.load(sys.stdin)

# The view models hold the input objects; compare them through their
# JSON-able spellings (titles and displays stringified, items as dicts)
# so object identity never leaks into the comparison.
foryou = []
for shelf in payload["foryou_cases"]:
    title, rows = _foryou_shelf_rows(shelf)
    foryou.append([str(title), [[str(display), dict(item)] for display, item in rows]])

out = {
    "route": [list(_browse_item_route(i)) for i in payload["route_cases"]],
    "foryou": foryou,
    "raw": [[dict(s) for s in _charts_raw_shelves(c)] for c in payload["raw_cases"]],
    "pill": [list(map(list, _charts_pill_rows(d, e, a))) for d, e, a in payload["pill_cases"]],
    "country": [_charts_country_label(c) for c in payload["country_cases"]],
    "visible": [
        _charts_event_row_visible(e, w, t) for e, w, t in payload["visible_cases"]
    ],
    "release": [_new_release_label(a) for a in payload["release_cases"]],
    "playlist": [_playlist_row_label(p) for p in payload["playlist_cases"]],
    "subscription": [_subscription_label(a) for a in payload["subscription_cases"]],
}
json.dump(out, sys.stdout, sort_keys=True)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "route_cases": _ROUTE_CASES,
            "foryou_cases": _FORYOU_CASES,
            "raw_cases": _RAW_CASES,
            "pill_cases": _PILL_CASES,
            "country_cases": _COUNTRY_CASES,
            "visible_cases": _VISIBLE_CASES,
            "release_cases": _RELEASE_CASES,
            "playlist_cases": _PLAYLIST_CASES,
            "subscription_cases": _SUBSCRIPTION_CASES,
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
    foryou = []
    for shelf in _FORYOU_CASES:
        title, rows = _foryou_shelf_rows(shelf)
        foryou.append([str(title), [[str(display), dict(item)] for display, item in rows]])
    return {
        "route": [list(_browse_item_route(i)) for i in _ROUTE_CASES],
        "foryou": foryou,
        "raw": [[dict(s) for s in _charts_raw_shelves(c)] for c in _RAW_CASES],
        "pill": [list(map(list, _charts_pill_rows(d, e, a))) for d, e, a in _PILL_CASES],
        "country": [_charts_country_label(c) for c in _COUNTRY_CASES],
        "visible": [_charts_event_row_visible(e, w, t) for e, w, t in _VISIBLE_CASES],
        "release": [_new_release_label(a) for a in _RELEASE_CASES],
        "playlist": [_playlist_row_label(p) for p in _PLAYLIST_CASES],
        "subscription": [_subscription_label(a) for a in _SUBSCRIPTION_CASES],
    }


class TestNativeParity:
    """When the Rust extension is built, the native rendering engines
    must produce exactly the same outcomes as the pure-Python references
    — same routes, same shelf view models, same pill rows, same section
    labels — for every case, so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        """Every rendering outcome — the item routes, the For You shelf
        rows, the raw chart shelves, the pill rows, the country line, the
        event-row gate and the three section labels — must agree with the
        pure-Python reference case for case."""
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key in native:
            assert native[key] == fallback[key], (
                f"native and fallback browse-render outcomes differ for {key}:\n"
                f"native:   {native[key]!r}\n"
                f"fallback: {fallback[key]!r}"
            )
