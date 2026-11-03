"""Tests for the search-page label glue engines.

The label glue behind ``SearchPage`` and ``SearchResultPanel``: the
bottom-of-panel count label (``_panel_count_label`` — "No results" /
"N result(s)" from ``load_items``), the mode indicator label
(``_mode_label`` — the ``▶ Music`` / ``▶ All`` display beside the input,
shared by ``compose``, ``on_mount`` and ``_toggle_search_mode``), the
suggestion-overlay pick decision (``_suggestion_pick`` — the index
bounds check behind the suggestion-list's selection handling), the
recent-searches label pass (``_recent_search_labels`` — the history rows
becoming suggestion labels), and the ``_search_all`` categorization
(``categorize_search_results`` — bucketing one unfiltered search into
the four result panels by ``resultType``). The pure-Python references
live in ``ui/pages/search.py``; when the Rust extension is built the
same engines run natively (``_ytm_player_native.search_panel_count_label``
& co).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart — over the reachable
shapes. The deliberate native-only degradations (a query-less history
row skipped where the reference raises ``KeyError``; a ``.get``-less
search item dropped where the reference raises ``AttributeError``) are
pinned by the native-only ``TestTotalityEngine`` instead, the same
stance the label engine's suite takes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from ytm_player.ui.pages.search import (
    SearchPage,
    SearchResultPanel,
    SuggestionList,
    _mode_label,
    _panel_count_label,
    _recent_search_labels,
    _suggestion_pick,
    categorize_search_results,
)
from ytm_player.ui.widgets.track_table import TrackTable

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Panel count label ─────────────────────────────────────────────────


class TestPanelCountLabel:
    @pytest.mark.parametrize(
        ("count", "label"),
        [
            (0, "No results"),
            (1, "1 result"),
            (2, "2 results"),
            (5, "5 results"),
            (100, "100 results"),
        ],
    )
    def test_the_label_follows_the_plural_rule(self, count: int, label: str) -> None:
        assert _panel_count_label(count) == label


# ── Mode indicator label ─────────────────────────────────────────────


class TestModeLabel:
    def test_music_mode(self) -> None:
        assert _mode_label("music", "#ff4e45") == "[#ff4e45]▶[/#ff4e45] Music"

    def test_all_mode(self) -> None:
        assert _mode_label("all", "#ff4e45") == "[#ff4e45]▶[/#ff4e45] All"

    def test_anything_else_is_the_all_results_label(self) -> None:
        # The reactive only ever holds "music"/"all", but the composer
        # is a plain == branch: junk reads as the all-results mode.
        assert _mode_label("", "x") == "[x]▶[/x] All"
        assert _mode_label("MUSIC", "x") == "[x]▶[/x] All"

    def test_the_colour_interpolates_twice(self) -> None:
        label = _mode_label("music", "#0f0")
        assert label.count("#0f0") == 2
        assert label.startswith("[#0f0]")
        assert label.endswith("[/#0f0] Music")


# ── Suggestion-overlay pick decision ─────────────────────────────────


class TestSuggestionPick:
    def test_first_middle_and_last(self) -> None:
        suggestions = ["a", "b", "c"]
        assert _suggestion_pick(suggestions, 0) == "a"
        assert _suggestion_pick(suggestions, 1) == "b"
        assert _suggestion_pick(suggestions, 2) == "c"

    def test_no_highlighted_row_is_ignored(self) -> None:
        assert _suggestion_pick(["a"], None) is None

    def test_a_stale_selection_is_ignored(self) -> None:
        # The list was repopulated under the cursor: both an index past
        # the end and a negative one read as no pick.
        assert _suggestion_pick(["a"], 1) is None
        assert _suggestion_pick(["a"], -1) is None
        assert _suggestion_pick([], 0) is None

    def test_the_raw_stored_row_is_returned(self) -> None:
        # Not a copy: the picked object is the suggestion object itself,
        # whatever it is.
        marker = object()
        picked = _suggestion_pick([marker], 0)
        assert picked is marker

    def test_str_subclass_suggestions_pass_through(self) -> None:
        class S(str):
            pass

        s = S("query")
        assert _suggestion_pick([s], 0) is s


# ── Recent-searches label pass ───────────────────────────────────────


class TestRecentSearchLabels:
    def test_rows_become_labels_in_order(self) -> None:
        entries = [{"query": "first"}, {"query": "second"}, {"query": "third"}]
        assert _recent_search_labels(entries) == ["first", "second", "third"]

    def test_empty_history_is_an_empty_list(self) -> None:
        assert _recent_search_labels([]) == []

    def test_the_query_values_pass_through_raw(self) -> None:
        # Labels are the stored values themselves, str or not, empty
        # string included.
        entries = [{"query": 42}, {"query": ""}, {"query": None}]
        assert _recent_search_labels(entries) == [42, "", None]

    def test_the_rows_are_not_copied(self) -> None:
        entries = [{"query": "q"}]
        labels = _recent_search_labels(entries)
        assert labels[0] is entries[0]["query"]


# ── _search_all categorization ───────────────────────────────────────


def _raw_results() -> list[dict[str, Any]]:
    """One item per branch plus the drops, in arrival order."""
    return [
        {"resultType": "song", "videoId": "v1"},
        {"resultType": "video", "videoId": "v2"},
        {"resultType": "album", "browseId": "b1"},
        {"resultType": "artist", "browseId": "b2"},
        {"resultType": "playlist", "browseId": "b3"},
        {"resultType": "podcast", "title": "dropped"},
        {"resultType": "", "title": "dropped"},
        {"title": "no type at all"},
        {"resultType": 42, "title": "dropped"},
        {"resultType": None, "title": "dropped"},
    ]


class TestCategorizeSearchResults:
    def test_every_branch_and_every_drop(self) -> None:
        results = categorize_search_results(_raw_results())

        assert [item["videoId"] for item in results["songs"]] == ["v1", "v2"]
        assert [item["browseId"] for item in results["albums"]] == ["b1"]
        assert [item["browseId"] for item in results["artists"]] == ["b2"]
        assert [item["browseId"] for item in results["playlists"]] == ["b3"]

    def test_song_and_video_share_the_songs_bucket(self) -> None:
        results = categorize_search_results(_raw_results())
        assert len(results["songs"]) == 2

    def test_the_four_buckets_are_always_present(self) -> None:
        results = categorize_search_results([])
        assert results == {"songs": [], "albums": [], "artists": [], "playlists": []}
        assert list(results) == ["songs", "albums", "artists", "playlists"]

    def test_arrival_order_is_preserved_within_a_bucket(self) -> None:
        raw = [{"resultType": "artist", "n": n} for n in range(5)]
        results = categorize_search_results(raw)
        assert [item["n"] for item in results["artists"]] == [0, 1, 2, 3, 4]

    def test_buckets_hold_the_item_objects_themselves(self) -> None:
        raw = [{"resultType": "album"}]
        results = categorize_search_results(raw)
        assert results["albums"][0] is raw[0]

    def test_the_input_list_is_not_mutated(self) -> None:
        raw = _raw_results()
        snapshot = json.dumps(raw, sort_keys=True)
        categorize_search_results(raw)
        assert json.dumps(raw, sort_keys=True) == snapshot

    def test_dict_subclasses_steer_their_own_get(self) -> None:
        class Overriding(dict):
            def get(self, key, default=None):
                if key == "resultType":
                    return "album"
                return super().get(key, default)

        item = Overriding(resultType="song", title="T")
        results = categorize_search_results([item])
        assert results["albums"] == [item]
        assert results["songs"] == []

    def test_str_subclass_result_types_classify_by_equality(self) -> None:
        class S(str):
            pass

        item = {"resultType": S("song")}
        results = categorize_search_results([item])
        assert results["songs"] == [item]


# ── Total engine (native only) ──────────────────────────────────────


class TestTotalityEngine:
    """The native engines stay total where the references would raise.

    A ``.get``-less search item raises ``AttributeError`` in the
    reference; a history row without a ``query`` key raises ``KeyError``
    in the reference. The native engines degrade instead — the
    convention the ported engines before it follow — so these tests
    only run with the extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_dict_search_items_are_dropped(self) -> None:
        raw: list[Any] = [42, None, "str", ["list"], {"resultType": "song"}]
        results = categorize_search_results(raw)
        assert results["songs"] == [{"resultType": "song"}]
        assert results["albums"] == []
        assert results["artists"] == []
        assert results["playlists"] == []

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_query_less_history_row_is_skipped(self) -> None:
        entries = [{"nope": 1}, {"query": "kept"}, {}]
        assert _recent_search_labels(entries) == ["kept"]

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_dict_subclasses_steer_their_own_getitem(self) -> None:
        class Overriding(dict):
            def __getitem__(self, key):
                return "synthetic"

        assert _recent_search_labels([Overriding(nope=1)]) == ["synthetic"]


# ── Native vs fallback parity ────────────────────────────────────────

# One entry per engine case; the child interpreter below JSON
# round-trips them, so every value is a JSON type. The payload sticks
# to the reachable shapes — the deliberate native-only degradations
# (the .get-less item, the query-less row) are pinned by
# TestTotalityEngine instead, exactly like the queue engines' suite.

_PANEL_COUNTS: list[int] = [0, 1, 2, 5, 100]

_MODE_CASES: list[list[str]] = [
    ["music", "#ff4e45"],
    ["all", "#ff4e45"],
    ["", "x"],
    ["MUSIC", "x"],
    ["junk", "#123"],
]

_PICK_CASES: list[list[Any]] = [
    [["a", "b", "c"], 0],
    [["a", "b", "c"], 1],
    [["a", "b", "c"], 2],
    [["a"], None],
    [["a"], 1],
    [["a"], -1],
    [[], 0],
    [[], None],
]

_RECENT_CASES: list[list[dict[str, Any]]] = [
    [],
    [{"query": "x"}],
    [{"query": "a"}, {"query": "b"}, {"query": "c"}],
    [{"query": ""}],
    [{"query": 42}],
    [{"query": None}],
]

_CATEGORIZE_CASES: list[list[dict[str, Any]]] = [
    [],
    [{"resultType": "song", "videoId": "v1"}],
    [
        {"resultType": "song", "videoId": "v1"},
        {"resultType": "video", "videoId": "v2"},
        {"resultType": "album", "browseId": "b1"},
        {"resultType": "artist", "browseId": "b2"},
        {"resultType": "playlist", "browseId": "b3"},
    ],
    [{"resultType": "podcast"}, {"resultType": ""}, {}, {"resultType": 42}],
    [{"resultType": "artist", "n": n} for n in range(5)],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.search import (
    _mode_label,
    _panel_count_label,
    _recent_search_labels,
    _suggestion_pick,
    categorize_search_results,
)

payload = json.load(sys.stdin)

out = {
    "panel": [_panel_count_label(count) for count in payload["panel_counts"]],
    "mode": [
        _mode_label(mode, color) for mode, color in payload["mode_cases"]
    ],
    "pick": [
        _suggestion_pick(suggestions, index)
        for suggestions, index in payload["pick_cases"]
    ],
    "recent": [
        _recent_search_labels(entries) for entries in payload["recent_cases"]
    ],
    "categorize": [
        categorize_search_results(raw) for raw in payload["categorize_cases"]
    ],
}
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_results(payload: dict) -> dict:
    """Run whichever engines the module loaded (native when built) over
    the payload."""
    return {
        "panel": [_panel_count_label(count) for count in payload["panel_counts"]],
        "mode": [_mode_label(mode, color) for mode, color in payload["mode_cases"]],
        "pick": [
            _suggestion_pick(suggestions, index) for suggestions, index in payload["pick_cases"]
        ],
        "recent": [_recent_search_labels(entries) for entries in payload["recent_cases"]],
        "categorize": [categorize_search_results(raw) for raw in payload["categorize_cases"]],
    }


def _fallback_results(payload: dict) -> dict:
    """Run the pure-Python reference engines over the payload in a child
    interpreter with the native extension import blocked."""
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native search-page engines
    must produce exactly the same outcomes as the pure-Python references
    for every case — the plural rule, the colour markup, the pick
    bounds, the raw label values, and every categorization branch — so
    the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self) -> None:
        payload = {
            "panel_counts": _PANEL_COUNTS,
            "mode_cases": _MODE_CASES,
            "pick_cases": _PICK_CASES,
            "recent_cases": _RECENT_CASES,
            "categorize_cases": _CATEGORIZE_CASES,
        }
        native = _native_results(payload)
        fallback = _fallback_results(payload)
        assert native == fallback, (
            "native and fallback search-page outcomes differ:\n"
            f"native:   {native!r}\n"
            f"fallback: {fallback!r}"
        )


# ── The page glue, through the real Textual loop ─────────────────────


class _SearchHost(App[None]):
    """Minimal host app exposing the attributes SearchPage reads."""

    def __init__(self) -> None:
        super().__init__()
        self.ytmusic = MagicMock()
        self.ytmusic.search = AsyncMock(return_value=[])
        self.ytmusic.get_search_suggestions = AsyncMock(return_value=[])
        self.history = MagicMock()
        self.history.log_search = AsyncMock()
        self.history.get_search_history = AsyncMock(return_value=[])

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield SearchPage(id="search")


def _panel_count_text(app: App, panel_id: str) -> str:
    return str(app.query_one(f"#{panel_id} .panel-count", Static).render())


def _mode_text(app: App) -> str:
    # The Static renders the markup-tagged label; the rendered Text is
    # the plain "▶ Music" / "▶ All" with the tags resolved to styles.
    return str(app.query_one("#search-mode", Static).render())


def _stub_selection(index: int | None) -> MagicMock:
    event = MagicMock(name="ListView.Selected")
    event.list_view = MagicMock(name="list_view")
    event.list_view.id = "suggestion-list"
    event.list_view.index = index
    return event


class TestSearchPageGlue:
    """The engines' output reaching the panels, the mode indicator and
    the suggestion overlay through the real Textual loop."""

    async def test_the_panel_count_labels_follow_load_items(self) -> None:
        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            panel = app.query_one("#albums-panel", SearchResultPanel)

            panel.load_items([{"resultType": "album", "title": "One"}])
            await pilot.pause()
            assert _panel_count_text(app, "albums-panel") == "1 result"

            panel.load_items(
                [{"resultType": "album", "title": "One"}, {"resultType": "album", "title": "Two"}]
            )
            await pilot.pause()
            assert _panel_count_text(app, "albums-panel") == "2 results"

            panel.load_items([])
            await pilot.pause()
            assert _panel_count_text(app, "albums-panel") == "No results"

    async def test_a_music_search_fills_the_panel_count_labels(self) -> None:
        def _results(query: str, filter: str | None = None, limit: int = 10) -> list[dict]:
            return {
                "songs": [{"videoId": "v1", "title": "Song One"}],
                "albums": [
                    {"resultType": "album", "title": "A1"},
                    {"resultType": "album", "title": "A2"},
                ],
                "artists": [{"resultType": "artist", "title": "R1"}],
                "community_playlists": [],
            }.get(filter, [])

        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            page = app.query_one(SearchPage)
            app.ytmusic.search = AsyncMock(side_effect=_results)

            await page._execute_search("query")
            await pilot.pause()

            assert _panel_count_text(app, "albums-panel") == "2 results"
            assert _panel_count_text(app, "artists-panel") == "1 result"
            assert _panel_count_text(app, "playlists-panel") == "No results"
            assert app.query_one("#songs-table", TrackTable).row_count == 1

    async def test_the_mode_indicator_toggles_between_music_and_all(self) -> None:
        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            page = app.query_one(SearchPage)

            assert page.search_mode == "music"  # the settings default
            assert _mode_text(app) == "▶ Music"

            page._toggle_search_mode()
            await pilot.pause()
            assert page.search_mode == "all"
            assert _mode_text(app) == "▶ All"

            page._toggle_search_mode()
            await pilot.pause()
            assert _mode_text(app) == "▶ Music"

    async def test_picking_a_suggestion_populates_the_input_and_searches(self) -> None:
        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            page = app.query_one(SearchPage)
            overlay = app.query_one("#suggestion-overlay", SuggestionList)
            overlay.show_suggestions(["picked query", "other"])
            await pilot.pause()
            page.search_mode = "all"  # one unfiltered call — the pick's flow

            page.on_list_view_selected(_stub_selection(0))
            await pilot.pause()

            assert app.query_one("#search-input").value == "picked query"
            assert page._last_query == "picked query"
            assert app.ytmusic.search.await_count == 1
            assert app.ytmusic.search.await_args.args[0] == "picked query"
            assert app.ytmusic.search.await_args.kwargs == {"filter": None, "limit": 40}
            assert not overlay.has_class("visible")

    async def test_a_stale_suggestion_selection_is_ignored(self) -> None:
        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            page = app.query_one(SearchPage)
            overlay = app.query_one("#suggestion-overlay", SuggestionList)
            overlay.show_suggestions(["only"])
            await pilot.pause()

            page.on_list_view_selected(_stub_selection(3))
            await pilot.pause()

            assert app.query_one("#search-input").value == ""
            assert page._last_query == ""
            assert app.ytmusic.search.await_count == 0

    async def test_recent_searches_reach_the_overlay_through_the_engine(self) -> None:
        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            page = app.query_one(SearchPage)
            app.history.get_search_history = AsyncMock(
                return_value=[{"query": "earlier"}, {"query": "later"}]
            )

            await page._load_recent_searches()
            await pilot.pause()

            overlay = app.query_one("#suggestion-overlay", SuggestionList)
            assert overlay.has_class("visible")
            assert overlay.suggestions == ["earlier", "later"]

    async def test_all_mode_buckets_the_raw_search_into_the_panels(self) -> None:
        raw = [
            {"resultType": "song", "videoId": "v1", "title": "Song One"},
            {"resultType": "video", "videoId": "v2", "title": "Video Two"},
            {"resultType": "album", "browseId": "b1", "title": "Album"},
            {"resultType": "artist", "browseId": "b2", "title": "Artist"},
            {"resultType": "playlist", "browseId": "b3", "title": "Playlist"},
            {"resultType": "podcast", "title": "Dropped"},
        ]

        async with _SearchHost().run_test() as pilot:
            app = pilot.app
            page = app.query_one(SearchPage)
            page.search_mode = "all"
            app.ytmusic.search = AsyncMock(return_value=raw)

            await page._execute_search("query")
            await pilot.pause()

            # One unfiltered call, categorized into the four panels.
            assert app.ytmusic.search.await_args.kwargs == {"filter": None, "limit": 40}
            assert app.query_one("#songs-table", TrackTable).row_count == 2
            assert len(app.query_one("#albums-panel", SearchResultPanel)._items) == 1
            assert len(app.query_one("#artists-panel", SearchResultPanel)._items) == 1
            assert len(app.query_one("#playlists-panel", SearchResultPanel)._items) == 1
            assert _panel_count_text(app, "albums-panel") == "1 result"
            assert _panel_count_text(app, "playlists-panel") == "1 result"

    async def test_the_mode_indicator_colour_follows_the_theme(self) -> None:
        # The indicator interpolates the theme primary through the
        # engine; the mounted Static carries it as the ▶ glyph's span.
        async with _SearchHost().run_test() as pilot:
            from ytm_player.ui.theme import get_theme

            static = pilot.app.query_one("#search-mode", Static)
            content = static.render()
            assert str(content) in ("▶ Music", "▶ All")
            (span,) = content.spans
            assert str(span.style) == get_theme().primary
