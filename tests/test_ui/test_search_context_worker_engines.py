"""Tests for the search/context worker engines — the deeper coverage
inside the two pages' workers and click handlers.

The engines behind the remaining pure decisions of
``ui/pages/search.py`` and ``ui/pages/context.py``:

- the search result-item navigation decisions — the left-click
  context-page pick (``_item_navigation`` — the panel/type branch and
  the browse-id fallback chain behind
  ``on_search_result_panel_item_selected``), the right-click item type
  (``_panel_item_type`` — the panel→ActionsPopup mapping), the
  subscribe-id resolution (``_subscribe_browse_id``) and the copy-link
  URL (``_browse_link``);
- the music-mode worker merge (``music_search_results`` — the
  classification loop behind ``_search_music``'s gathered concurrent
  fetches: which lists land in which bucket, which fetches failed);
- the context worker-end failure copies (``_load_failure_message``,
  ``_empty_load_message`` — the log-pointing copy and the empty-load
  branch's message pick);
- the context header-button resolutions (``library_playlist_id`` — the
  Add-to-Library id chain with its VL strip; ``radio_item`` — the
  Start-Radio item assembly) and the discography swap
  (``apply_expanded_sections``).

The pure-Python references live in the pages themselves; when the Rust
extension is built the same engines run natively
(``_ytm_player_native.search_item_navigation`` & co). The unit tests
below assert the engines' contract on whichever backend is loaded.
When the extension is built, the parity test additionally compares the
native engines against the pure-Python references (run in a child
interpreter with the extension import blocked) case for case, so the
two implementations cannot drift apart — over the reachable shapes.
The deliberate native-only degradations (a ``.get``-less item, an
unhashable panel id, a non-str library id, a non-mapping radio base, an
unassignable section) are pinned by the native-only
``TestTotalityEngine`` instead, the same stance the other engine suites
take. The last classes run the wiring — the click handlers, the
right-click ActionsPopup arms, the music-mode merge end to end, the
Add-to-Library VL strip, the Start-Radio item and the worker-state
message branches — through the real Textual loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.worker import Worker, WorkerState

from ytm_player.ui.pages.context import (
    ContextPage,
    _empty_load_message,
    _load_failure_message,
    apply_expanded_sections,
    library_playlist_id,
    radio_item,
)
from ytm_player.ui.pages.search import (
    SearchPage,
    SearchResultPanel,
    _browse_link,
    _item_navigation,
    _panel_item_type,
    _subscribe_browse_id,
    music_search_results,
)
from ytm_player.ui.widgets.track_table import TrackTable

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Left-click navigation decision ───────────────────────────────────


class TestItemNavigation:
    def test_each_panel_branch(self) -> None:
        assert _item_navigation({"resultType": "album", "browseId": "b1"}, "albums-panel") == {
            "context_type": "album",
            "context_id": "b1",
        }
        assert _item_navigation({"resultType": "artist", "browseId": "b2"}, "artists-panel") == {
            "context_type": "artist",
            "context_id": "b2",
        }
        assert _item_navigation(
            {"resultType": "playlist", "browseId": "b3"}, "playlists-panel"
        ) == {"context_type": "playlist", "context_id": "b3"}

    def test_the_result_type_steers_when_the_panel_does_not_match(self) -> None:
        assert (
            _item_navigation({"resultType": "album", "browseId": "b1"}, "junk-panel")[
                "context_type"
            ]
            == "album"
        )
        assert (
            _item_navigation({"resultType": "artist", "artist_id": "UC1"}, "")["context_type"]
            == "artist"
        )
        assert (
            _item_navigation({"resultType": "playlist", "playlistId": "PL1"}, None)["context_type"]
            == "playlist"
        )

    def test_the_browse_id_fallback_keys(self) -> None:
        assert (
            _item_navigation({"resultType": "album", "album_id": "a1"}, "albums-panel")[
                "context_id"
            ]
            == "a1"
        )
        assert (
            _item_navigation({"resultType": "artist", "artist_id": "r1"}, "artists-panel")[
                "context_id"
            ]
            == "r1"
        )
        assert (
            _item_navigation({"resultType": "playlist", "playlistId": "p1"}, "playlists-panel")[
                "context_id"
            ]
            == "p1"
        )
        # browseId wins over the branch's fallback key.
        assert (
            _item_navigation(
                {"resultType": "album", "browseId": "b", "album_id": "a"}, "albums-panel"
            )["context_id"]
            == "b"
        )

    def test_a_falsy_browse_id_falls_through_to_the_fallback(self) -> None:
        item = {"resultType": "artist", "browseId": "", "artist_id": "UC9"}
        assert _item_navigation(item, "artists-panel")["context_id"] == "UC9"

    def test_a_panel_matched_branch_with_no_id_answers_none(self) -> None:
        # The reference's nested `if browse_id:` gate: no fallthrough to
        # a later branch, the click is simply ignored.
        assert _item_navigation({"resultType": "album"}, "albums-panel") is None
        assert (
            _item_navigation(
                {"resultType": "artist", "browseId": None, "artist_id": ""}, "artists-panel"
            )
            is None
        )
        # The panel match outranks the type tag: a playlist-typed item on
        # the albums panel reads as the album branch, which has no id.
        assert (
            _item_navigation({"resultType": "playlist", "playlistId": "p1"}, "albums-panel") is None
        )

    def test_no_branch_answers_none(self) -> None:
        assert _item_navigation({"resultType": "song", "videoId": "v"}, "junk-panel") is None
        assert _item_navigation({"resultType": "podcast"}, "junk") is None
        assert _item_navigation({}, "") is None

    def test_non_str_result_types_never_match(self) -> None:
        assert _item_navigation({"resultType": 42, "browseId": "b"}, "junk") is None
        assert _item_navigation({"resultType": None, "browseId": "b"}, "junk") is None

    def test_the_context_id_is_the_raw_stored_value(self) -> None:
        marker = object()
        nav = _item_navigation({"resultType": "album", "browseId": marker}, "albums-panel")
        assert nav["context_id"] is marker

    def test_dict_subclasses_steer_their_own_get(self) -> None:
        class Overriding(dict):
            def get(self, key, default=None):
                if key == "resultType":
                    return "artist"
                return super().get(key, default)

        item = Overriding(resultType="song", browseId="b1")
        nav = _item_navigation(item, "junk-panel")
        assert nav["context_type"] == "artist"
        assert nav["context_id"] == "b1"

    def test_str_subclass_result_types_classify_by_equality(self) -> None:
        class S(str):
            pass

        item = {"resultType": S("playlist"), "browseId": "b"}
        assert _item_navigation(item, "junk")["context_type"] == "playlist"


# ── Right-click item-type mapping ────────────────────────────────────


class TestPanelItemType:
    @pytest.mark.parametrize(
        ("panel_id", "item_type"),
        [
            ("albums-panel", "album"),
            ("artists-panel", "artist"),
            ("playlists-panel", "playlist"),
            ("songs-panel", "track"),
            ("", "track"),
            (None, "track"),
            ("junk", "track"),
        ],
    )
    def test_the_mapping_and_its_default(self, panel_id: Any, item_type: str) -> None:
        assert _panel_item_type(panel_id) == item_type

    def test_a_str_subclass_panel_id_classifies(self) -> None:
        class S(str):
            pass

        assert _panel_item_type(S("albums-panel")) == "album"


# ── Subscribe-id resolution ──────────────────────────────────────────


class TestSubscribeBrowseId:
    def test_browse_id_wins(self) -> None:
        assert _subscribe_browse_id({"browseId": "b", "artist_id": "a"}) == "b"

    def test_artist_id_is_the_fallback(self) -> None:
        assert _subscribe_browse_id({"artist_id": "UC1"}) == "UC1"

    def test_a_falsy_browse_id_falls_through(self) -> None:
        assert _subscribe_browse_id({"browseId": "", "artist_id": "UC1"}) == "UC1"
        assert _subscribe_browse_id({"browseId": None, "artist_id": "UC1"}) == "UC1"

    def test_no_id_reads_as_the_empty_string(self) -> None:
        assert _subscribe_browse_id({}) == ""
        assert _subscribe_browse_id({"browseId": None, "artist_id": 0}) == ""

    def test_the_raw_value_is_returned(self) -> None:
        marker = object()
        assert _subscribe_browse_id({"browseId": marker}) is marker


# ── Copy-link URL ─────────────────────────────────────────────────────


class TestBrowseLink:
    def test_the_precedence_chain(self) -> None:
        item = {"browseId": "b", "album_id": "a", "playlistId": "p", "artist_id": "r"}
        assert _browse_link(item) == "https://music.youtube.com/browse/b"
        del item["browseId"]
        assert _browse_link(item) == "https://music.youtube.com/browse/a"
        del item["album_id"]
        assert _browse_link(item) == "https://music.youtube.com/browse/p"
        del item["playlistId"]
        assert _browse_link(item) == "https://music.youtube.com/browse/r"

    def test_falsy_ids_are_skipped(self) -> None:
        assert (
            _browse_link({"browseId": "", "album_id": "a"}) == "https://music.youtube.com/browse/a"
        )
        assert (
            _browse_link({"browseId": None, "playlistId": "p"})
            == "https://music.youtube.com/browse/p"
        )

    def test_no_id_answers_none(self) -> None:
        assert _browse_link({}) is None
        assert (
            _browse_link({"browseId": None, "album_id": "", "playlistId": None, "artist_id": 0})
            is None
        )

    def test_a_non_str_id_interpolates_through_str(self) -> None:
        # The reference's f-string str() conversion, so an int id
        # stringifies exactly like the reference's interpolation.
        assert _browse_link({"browseId": 42}) == "https://music.youtube.com/browse/42"


# ── Music-mode worker merge ────────────────────────────────────────────


class TestMusicSearchResults:
    def test_the_four_buckets_are_always_present(self) -> None:
        merged = music_search_results([], [])
        assert merged["results"] == {"songs": [], "albums": [], "artists": [], "playlists": []}
        assert list(merged["results"]) == ["songs", "albums", "artists", "playlists"]
        assert merged["failed"] == []

    def test_lists_land_under_their_category(self) -> None:
        songs = [{"videoId": "v1"}]
        artists: list[dict[str, Any]] = []
        merged = music_search_results(["songs", "albums", "artists"], [songs, None, artists])
        assert merged["results"]["songs"] == [{"videoId": "v1"}]
        assert merged["results"]["artists"] == []
        assert merged["failed"] == []

    def test_the_landed_lists_are_the_objects_themselves(self) -> None:
        songs = [{"videoId": "v1"}]
        merged = music_search_results(["songs"], [songs])
        assert merged["results"]["songs"] is songs

    def test_exceptions_land_in_the_failed_pairs(self) -> None:
        boom = ValueError("albums endpoint down")
        merged = music_search_results(["songs", "albums"], [[{"v": 1}], boom])
        assert merged["results"]["songs"] == [{"v": 1}]
        assert merged["results"]["albums"] == []
        assert len(merged["failed"]) == 1
        category, failure = merged["failed"][0]
        assert category == "albums"
        assert failure is boom

    def test_a_non_list_non_exception_is_silently_dropped(self) -> None:
        merged = music_search_results(["songs", "albums"], ["str", 42, {"d": 1}])
        assert merged["results"] == {"songs": [], "albums": [], "artists": [], "playlists": []}
        assert merged["failed"] == []

    def test_a_base_exception_that_is_not_an_exception_is_dropped(self) -> None:
        # isinstance(ki, Exception) is False: neither logged (the
        # caller's logging loop walks `failed`) nor landed, exactly the
        # reference's elif ladder.
        merged = music_search_results(["songs", "albums"], [[{"a": 1}], KeyboardInterrupt()])
        assert merged["results"]["songs"] == [{"a": 1}]
        assert merged["failed"] == []

    def test_the_pairs_stop_at_the_shorter_list(self) -> None:
        # gathered longer than the categories: the extras are unused.
        merged = music_search_results(["songs", "albums"], [[1], [2], [3]])
        assert merged["results"]["songs"] == [1]
        assert merged["results"]["albums"] == [2]
        # categories longer than gathered: the unpaired buckets stay
        # empty.
        merged = music_search_results(["songs", "albums", "artists"], [[1]])
        assert merged["results"]["artists"] == []

    def test_a_category_outside_the_four_is_appended(self) -> None:
        merged = music_search_results(["custom"], [[{"c": 1}]])
        assert merged["results"] == {
            "songs": [],
            "albums": [],
            "artists": [],
            "playlists": [],
            "custom": [{"c": 1}],
        }
        assert list(merged["results"]) == ["songs", "albums", "artists", "playlists", "custom"]


# ── Worker-end failure copies ─────────────────────────────────────────


class TestLoadFailureMessage:
    @pytest.mark.parametrize("context_type", ["album", "artist", "playlist"])
    def test_the_copy_points_at_the_log(self, context_type: str) -> None:
        assert _load_failure_message(context_type) == (
            f"Couldn't load this {context_type}. Check the log at "
            "~/.config/ytm-player/logs/ytm.log for details."
        )

    def test_an_unknown_type_interpolates_the_same_way(self) -> None:
        assert _load_failure_message("unknown-type").startswith("Couldn't load this unknown-type.")


class TestEmptyLoadMessage:
    def test_the_failed_flag_picks_the_log_pointing_copy(self) -> None:
        assert _empty_load_message(True, "artist") == _load_failure_message("artist")
        assert "ytm.log" in _empty_load_message(True, "artist")

    def test_the_genuine_empty_payload_keeps_the_empty_copy(self) -> None:
        assert _empty_load_message(False, "artist") == "No data found."
        assert _empty_load_message(False, "") == "No data found."


# ── Add-to-Library id resolution ──────────────────────────────────────


class TestLibraryPlaylistId:
    def test_the_payload_playlist_id_wins(self) -> None:
        assert library_playlist_id({"playlistId": "PL1", "id": "x"}, "ctx") == "PL1"

    def test_the_fallback_chain(self) -> None:
        assert library_playlist_id({"playlistId": "", "audioPlaylistId": "AP1"}, "ctx") == "AP1"
        assert library_playlist_id({"audioPlaylistId": None, "id": "i1"}, "ctx") == "i1"
        assert library_playlist_id({}, "ctx") == "ctx"
        assert library_playlist_id({}, "") == ""
        assert library_playlist_id({"playlistId": None, "audioPlaylistId": 0, "id": ""}, "") == ""

    def test_the_vl_prefix_is_stripped(self) -> None:
        assert library_playlist_id({"playlistId": "VLPL9"}, "ctx") == "PL9"
        assert library_playlist_id({}, "VLctx") == "ctx"
        assert library_playlist_id({"playlistId": "VL"}, "ctx") == ""

    def test_a_non_vl_prefix_is_left_alone(self) -> None:
        assert library_playlist_id({"playlistId": "XLVL9"}, "ctx") == "XLVL9"
        assert library_playlist_id({"playlistId": "V"}, "ctx") == "V"
        assert library_playlist_id({"playlistId": "VLa"}, "ctx") == "a"

    def test_the_answer_is_the_raw_value(self) -> None:
        # A str subclass: the value rides home as itself on both
        # backends (a plain non-str object is the total-engine case
        # below — the reference's .startswith would raise on it).
        class S(str):
            pass

        marker = S("PL1")
        assert library_playlist_id({"playlistId": marker}, "ctx") is marker


# ── Start-Radio item assembly ─────────────────────────────────────────


class TestRadioItem:
    def test_the_payload_is_spread_and_the_playlist_id_resolved(self) -> None:
        data = {"title": "T", "trackCount": 5, "owned": True}
        assert radio_item(data, "PL1") == {
            "title": "T",
            "trackCount": 5,
            "owned": True,
            "playlistId": "PL1",
        }

    def test_the_payloads_own_playlist_id_wins(self) -> None:
        assert radio_item({"playlistId": "PL9", "title": "T"}, "ctx")["playlistId"] == "PL9"

    def test_a_falsy_playlist_id_yields_the_context_id(self) -> None:
        assert radio_item({"playlistId": None, "title": "T"}, "PL2")["playlistId"] == "PL2"
        assert radio_item({"playlistId": ""}, "PL3")["playlistId"] == "PL3"
        assert radio_item({}, "PL4")["playlistId"] == "PL4"

    def test_a_fresh_dict_not_the_payload_itself(self) -> None:
        data = {"title": "T", "playlistId": "PL9"}
        item = radio_item(data, "ctx")
        assert item is not data
        assert data == {"title": "T", "playlistId": "PL9"}
        assert item == data

    def test_the_playlist_id_value_is_the_raw_object(self) -> None:
        marker = object()
        assert radio_item({"playlistId": marker}, "ctx")["playlistId"] is marker


# ── Discography swap ──────────────────────────────────────────────────


class TestApplyExpandedSections:
    def test_a_truthy_full_list_swaps_the_results(self) -> None:
        section = {"results": [1, 2], "browseId": "MPAD"}
        full = [3, 4, 5]
        apply_expanded_sections([section], [full])
        assert section["results"] is full

    def test_a_falsy_full_list_keeps_the_preview(self) -> None:
        section = {"results": [1, 2]}
        apply_expanded_sections([section], [[]])
        apply_expanded_sections([section], [None])
        assert section["results"] == [1, 2]

    def test_the_pairs_stop_at_the_shorter_list(self) -> None:
        # more sections than fetched lists: the unpaired sections keep
        # their preview.
        first: dict[str, Any] = {"results": [1]}
        second: dict[str, Any] = {"results": [2]}
        apply_expanded_sections([first, second], [[9]])
        assert first["results"] == [9]
        assert second["results"] == [2]
        # more fetched lists than sections: the extras are unused.
        only: dict[str, Any] = {"results": [1]}
        apply_expanded_sections([only], [[9], [10]])
        assert only["results"] == [9]

    def test_the_swap_lands_in_the_original_section_objects(self) -> None:
        data: dict[str, Any] = {"albums": {"results": [1]}}
        section = data["albums"]
        apply_expanded_sections([section], [[7]])
        assert data["albums"]["results"] == [7]

    def test_empty_inputs_swap_nothing(self) -> None:
        apply_expanded_sections([], [])
        apply_expanded_sections([{"results": [1]}], [])
        # nothing raised, nothing changed above.


# ── Total engine (native only) ─────────────────────────────────────────


class TestTotalityEngine:
    """The native engines stay total where the references would raise.

    A ``.get``-less item raises ``AttributeError`` in the references; an
    unhashable panel id raises ``TypeError`` in the item-type map
    lookup; a truthy non-str library id raises ``AttributeError`` on
    ``startswith``; a non-mapping radio base raises ``TypeError`` in
    the spread; an unassignable section raises ``TypeError`` on the
    subscript store. The native engines degrade instead — the
    convention the ported engines before them follow — so these tests
    only run with the extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_get_less_item_reads_as_no_navigation(self) -> None:
        assert _item_navigation(42, "albums-panel") is None
        assert _item_navigation("str", "junk") is None

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_an_unhashable_panel_id_reads_as_the_track_default(self) -> None:
        assert _panel_item_type(["not", "hashable"]) == "track"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_truthy_non_str_library_id_passes_through_unstripped(self) -> None:
        assert library_playlist_id({"id": 42}, "ctx") == 42
        assert library_playlist_id({"playlistId": 7}, "ctx") == 7

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_non_mapping_radio_base_starts_from_the_playlist_id(self) -> None:
        assert radio_item(42, "ctx") == {"playlistId": "ctx"}
        assert radio_item("nope", "PL1") == {"playlistId": "PL1"}

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_an_unassignable_section_is_skipped(self) -> None:
        # A tuple section: the reference's subscript store would raise;
        # the engine skips the store and moves on.
        apply_expanded_sections([(1, 2)], [[3]])
        # A falsy full list: the store is never attempted anyway.


# ── Native vs fallback parity ─────────────────────────────────────────

# One entry per engine case; the child interpreter below JSON
# round-trips them, so every value is a JSON type. The payload sticks
# to the reachable shapes — the deliberate native-only degradations
# are pinned by TestTotalityEngine instead, exactly like the search
# engines' suite. Exception inputs to the merge (not JSON
# round-trippable) are pinned by the contract tests above, which run
# on whichever backend is loaded.

_NAV_CASES: list[list[Any]] = [
    [{"resultType": "album", "browseId": "b1"}, "albums-panel"],
    [{"resultType": "artist", "browseId": "b2"}, "artists-panel"],
    [{"resultType": "playlist", "browseId": "b3"}, "playlists-panel"],
    [{"resultType": "album", "album_id": "a1"}, "junk-panel"],
    [{"resultType": "artist", "artist_id": "r1"}, ""],
    [{"resultType": "playlist", "playlistId": "p1"}, None],
    [{"resultType": "playlist", "playlistId": "p1"}, "albums-panel"],
    [{"resultType": "song", "videoId": "v"}, "junk-panel"],
    [{"resultType": "album", "browseId": ""}, "junk"],
    [{"resultType": 42, "browseId": "x"}, "junk"],
    [{"resultType": None}, "junk"],
    [{}, "albums-panel"],
]

_ITEM_TYPE_CASES: list[Any] = [
    "albums-panel",
    "artists-panel",
    "playlists-panel",
    "songs-panel",
    "junk",
    "",
    None,
]

_SUBSCRIBE_CASES: list[dict[str, Any]] = [
    {"browseId": "b", "artist_id": "a"},
    {"artist_id": "UC1"},
    {"browseId": "", "artist_id": "UC1"},
    {"browseId": None, "artist_id": 0},
    {"browseId": 42},
    {},
]

_LINK_CASES: list[dict[str, Any]] = [
    {"browseId": "b", "album_id": "a", "playlistId": "p", "artist_id": "r"},
    {"album_id": "a"},
    {"playlistId": "p"},
    {"artist_id": "r"},
    {"browseId": "", "album_id": "a"},
    {"browseId": 42},
    {"browseId": None, "album_id": ""},
    {},
]

_MERGE_CASES: list[list[list[Any]]] = [
    [[], []],
    [["songs", "albums", "artists", "playlists"], [[{"v": 1}], [{"a": 1}], [], None]],
    [["songs", "albums"], [[1, 2], "str", 42, {"d": 1}]],
    [["songs", "albums", "artists"], [[{"x": 1}]]],
    [["custom"], [[{"c": 1}]]],
]

_FAILURE_CASES: list[str] = ["album", "artist", "playlist", "unknown-type"]

_EMPTY_CASES: list[list[Any]] = [
    [True, "album"],
    [False, "album"],
    [True, "playlist"],
    [False, "junk"],
]

_LIBRARY_CASES: list[list[Any]] = [
    [{"playlistId": "PL1", "id": "x"}, "ctx"],
    [{"playlistId": "", "audioPlaylistId": "AP1"}, "ctx"],
    [{"audioPlaylistId": None, "id": "i1"}, "ctx"],
    [{}, "ctx"],
    [{}, ""],
    [{"playlistId": "VLPL9"}, "ctx"],
    [{}, "VLctx"],
    [{"playlistId": "VL"}, "ctx"],
    [{"playlistId": "XLVL9"}, "ctx"],
    [{"playlistId": "V"}, "ctx"],
    [{"playlistId": None, "audioPlaylistId": 0, "id": ""}, ""],
]

_RADIO_CASES: list[list[Any]] = [
    [{"title": "T"}, "PL1"],
    [{"playlistId": "PL9"}, "ctx"],
    [{"playlistId": None, "title": "T"}, "PL2"],
    [{}, ""],
    [{"title": "T", "trackCount": 5, "owned": True}, "PL3"],
]

_APPLY_CASES: list[list[list[Any]]] = [
    [[{"results": [1, 2], "browseId": "MPAD"}], [[3, 4, 5]]],
    [[{"results": [1]}], [[]]],
    [[{"results": [1]}], [None]],
    [[{"results": [1]}, {"results": [2]}], [[9]]],
    [[{"results": [1]}], [[9], [10]]],
    [
        [{"results": [1], "params": "p"}, {"results": [2], "params": "q"}],
        [[7], None],
    ],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.search import (
    _browse_link,
    _item_navigation,
    _panel_item_type,
    _subscribe_browse_id,
    music_search_results,
)
from ytm_player.ui.pages.context import (
    _empty_load_message,
    _load_failure_message,
    apply_expanded_sections,
    library_playlist_id,
    radio_item,
)

payload = json.load(sys.stdin)

# The swap mutates its sections: capture the post-state per case.
applied = []
for sections, full_lists in payload["apply"]:
    apply_expanded_sections(sections, full_lists)
    applied.append(sections)

json.dump(
    {
        "nav": [_item_navigation(item, panel_id) for item, panel_id in payload["nav"]],
        "item_type": [_panel_item_type(panel_id) for panel_id in payload["item_type"]],
        "subscribe": [_subscribe_browse_id(item) for item in payload["subscribe"]],
        "link": [_browse_link(item) for item in payload["link"]],
        "merge": [
            music_search_results(categories, gathered)
            for categories, gathered in payload["merge"]
        ],
        "failure": [_load_failure_message(t) for t in payload["failure"]],
        "empty": [
            _empty_load_message(failed, t) for failed, t in payload["empty"]
        ],
        "library": [
            library_playlist_id(data, context_id)
            for data, context_id in payload["library"]
        ],
        "radio": [
            radio_item(data, context_id) for data, context_id in payload["radio"]
        ],
        "apply": applied,
    },
    sys.stdout,
    ensure_ascii=False,
)
"""


def _native_outcomes() -> dict[str, Any]:
    """Every engine over every case, run by whichever backend the pages
    loaded (the native one, when the parity test runs)."""

    applied = []
    for sections, full_lists in _APPLY_CASES:
        apply_expanded_sections(sections, full_lists)
        applied.append(sections)

    return {
        "nav": [_item_navigation(item, panel_id) for item, panel_id in _NAV_CASES],
        "item_type": [_panel_item_type(panel_id) for panel_id in _ITEM_TYPE_CASES],
        "subscribe": [_subscribe_browse_id(item) for item in _SUBSCRIBE_CASES],
        "link": [_browse_link(item) for item in _LINK_CASES],
        "merge": [
            music_search_results(categories, gathered) for categories, gathered in _MERGE_CASES
        ],
        "failure": [_load_failure_message(t) for t in _FAILURE_CASES],
        "empty": [_empty_load_message(failed, t) for failed, t in _EMPTY_CASES],
        "library": [library_playlist_id(data, context_id) for data, context_id in _LIBRARY_CASES],
        "radio": [radio_item(data, context_id) for data, context_id in _RADIO_CASES],
        "apply": applied,
    }


def _fallback_outcomes() -> dict[str, Any]:
    """The pure-Python reference outcomes, run in a child interpreter
    with the native extension import blocked."""
    payload = {
        "nav": _NAV_CASES,
        "item_type": _ITEM_TYPE_CASES,
        "subscribe": _SUBSCRIBE_CASES,
        "link": _LINK_CASES,
        "merge": _MERGE_CASES,
        "failure": _FAILURE_CASES,
        "empty": _EMPTY_CASES,
        "library": _LIBRARY_CASES,
        "radio": _RADIO_CASES,
        "apply": _APPLY_CASES,
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native search/context
    worker engines must produce exactly the same outcomes as the
    pure-Python references for every case — the navigation branches and
    their id chains, the item-type mapping, the subscribe and link
    resolutions, the merge classifications, the failure copies, the
    library id chains with the VL strip, the radio item assemblies and
    the discography swaps — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_every_engine_matches_the_fallback(self) -> None:
        fallback = _fallback_outcomes()
        native = json.loads(json.dumps(_native_outcomes()))
        for family, cases in [
            ("nav", _NAV_CASES),
            ("item_type", _ITEM_TYPE_CASES),
            ("subscribe", _SUBSCRIBE_CASES),
            ("link", _LINK_CASES),
            ("merge", _MERGE_CASES),
            ("failure", _FAILURE_CASES),
            ("empty", _EMPTY_CASES),
            ("library", _LIBRARY_CASES),
            ("radio", _RADIO_CASES),
            ("apply", _APPLY_CASES),
        ]:
            assert native[family] == fallback[family], (
                f"native and fallback {family} engine differ:\n"
                f"native:   {native[family]!r}\n"
                f"fallback: {fallback[family]!r}"
            )


# ── The page glue, through the real Textual loop ───────────────────────


class _SearchHost(App[None]):
    """A host exposing every attribute the search page's click handlers
    read, recording the navigations, popups, notifications and worker
    starts they cause."""

    def __init__(self) -> None:
        super().__init__()
        self.ytmusic = MagicMock()
        self.ytmusic.search = AsyncMock(return_value=[])
        self.ytmusic.get_search_suggestions = AsyncMock(return_value=[])
        self.history = MagicMock()
        self.history.log_search = AsyncMock()
        self.history.get_search_history = AsyncMock(return_value=[])
        self.navigations: list[tuple[str, dict[str, Any]]] = []
        self.pushed: list[tuple[Any, Any]] = []
        self.notifications: list[tuple[str, dict[str, Any]]] = []
        self.worker_names: list[Any] = []
        self.subscribe_ids: list[Any] = []
        self.entity_actions: list[tuple[str, dict[str, Any], str]] = []
        self.dispatched: list[tuple[str, dict[str, Any], str]] = []

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield SearchPage(id="search")

    async def navigate_to(self, target: str, **kwargs: Any) -> None:
        self.navigations.append((target, kwargs))

    def push_screen(self, screen: Any, callback: Any = None) -> None:
        self.pushed.append((screen, callback))

    def notify(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.notifications.append((message, kwargs))

    def run_worker(self, work: Any, **kwargs: Any) -> None:
        self.worker_names.append(kwargs.get("name"))

    def _toggle_artist_subscribe_simple(self, browse_id: Any) -> None:
        self.subscribe_ids.append(browse_id)

    def _play_entity_and_navigate(
        self, action_id: str, item: dict[str, Any], item_type: str
    ) -> None:
        self.entity_actions.append((action_id, item, item_type))

    def _dispatch_entity_action(self, action_id: str, item: dict[str, Any], item_type: str) -> None:
        self.dispatched.append((action_id, item, item_type))


def _panel_count_text(app: App, panel_id: str) -> str:
    return str(app.query_one(f"#{panel_id} .panel-count", Static).render())


class TestSearchClickGlue:
    async def test_a_left_clicked_album_row_navigates_to_the_context_page(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            page = host.query_one(SearchPage)
            event = SearchResultPanel.ItemSelected(
                {"resultType": "album", "browseId": "MPREb_1"}, "albums-panel"
            )

            await page.on_search_result_panel_item_selected(event)

            assert host.navigations == [
                ("context", {"context_type": "album", "context_id": "MPREb_1"})
            ]

    async def test_a_left_clicked_row_without_an_id_is_ignored(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            page = host.query_one(SearchPage)
            event = SearchResultPanel.ItemSelected({"resultType": "artist"}, "artists-panel")

            await page.on_search_result_panel_item_selected(event)

            assert host.navigations == []

    async def test_a_left_clicked_row_through_the_type_tag_not_the_panel(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            page = host.query_one(SearchPage)
            event = SearchResultPanel.ItemSelected(
                {"resultType": "playlist", "playlistId": "PL7"}, "junk-panel"
            )

            await page.on_search_result_panel_item_selected(event)

            assert host.navigations == [
                ("context", {"context_type": "playlist", "context_id": "PL7"})
            ]


class TestSearchRightClickGlue:
    async def _right_click(self, pilot, item: dict[str, Any], panel_id: str) -> tuple[Any, Any]:
        host = pilot.app
        page = host.query_one(SearchPage)
        before = len(host.pushed)
        event = SearchResultPanel.ItemRightClicked(item, panel_id)
        page.on_search_result_panel_item_right_clicked(event)
        await pilot.pause()
        assert len(host.pushed) == before + 1
        return host.pushed[-1]

    async def test_the_popup_opens_with_the_engines_item_type(self) -> None:
        async with _SearchHost().run_test() as pilot:
            screen, _callback = await self._right_click(
                pilot, {"resultType": "album", "browseId": "b1"}, "albums-panel"
            )
            assert screen.item_type == "album"
            assert screen.item == {"resultType": "album", "browseId": "b1"}

            screen, _callback = await self._right_click(pilot, {"title": "Song"}, "songs-panel")
            assert screen.item_type == "track"

    async def test_the_copy_link_arm_copies_the_engines_url(self, monkeypatch) -> None:
        monkeypatch.setattr("ytm_player.ui.pages.search.copy_to_clipboard", lambda link: True)
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            _screen, callback = await self._right_click(
                pilot, {"resultType": "album", "browseId": "MPREb_x"}, "albums-panel"
            )

            callback("copy_link")

            assert host.notifications == [("Link copied", {"timeout": 2})]

    async def test_a_failed_copy_falls_back_to_notifying_the_link(self, monkeypatch) -> None:
        monkeypatch.setattr("ytm_player.ui.pages.search.copy_to_clipboard", lambda link: False)
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            item = {"resultType": "artist", "browseId": "UC_9"}
            _screen, callback = await self._right_click(pilot, item, "artists-panel")

            callback("copy_link")

            assert host.notifications == [("https://music.youtube.com/browse/UC_9", {"timeout": 5})]

    async def test_the_copy_link_arm_without_an_id_is_silent(self, monkeypatch) -> None:
        monkeypatch.setattr("ytm_player.ui.pages.search.copy_to_clipboard", lambda link: True)
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            _screen, callback = await self._right_click(pilot, {"title": "No id"}, "albums-panel")

            callback("copy_link")

            assert host.notifications == []

    async def test_the_toggle_subscribe_arm_uses_the_engines_id(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            item = {"resultType": "artist", "browseId": "UC_1"}
            _screen, callback = await self._right_click(pilot, item, "artists-panel")

            callback("toggle_subscribe")

            assert host.subscribe_ids == ["UC_1"]
            assert host.notifications == []

    async def test_the_toggle_subscribe_arm_without_an_id_warns(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            _screen, callback = await self._right_click(pilot, {"title": "No id"}, "artists-panel")

            callback("toggle_subscribe")

            assert host.subscribe_ids == []
            assert host.notifications == [
                ("No ID available for this item.", {"severity": "warning", "timeout": 2})
            ]

    async def test_a_dismissed_popup_does_nothing(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            _screen, callback = await self._right_click(pilot, {"title": "T"}, "albums-panel")

            callback(None)

            assert host.notifications == []
            assert host.subscribe_ids == []
            assert host.entity_actions == []
            assert host.dispatched == []

    async def test_the_play_all_arm_dispatches_the_entity(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            item = {"resultType": "playlist", "playlistId": "PL2"}
            _screen, callback = await self._right_click(pilot, item, "playlists-panel")

            callback("play_all")

            assert host.entity_actions == [("play_all", item, "playlist")]

    async def test_an_unknown_action_falls_through_to_the_dispatch(self) -> None:
        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            item = {"resultType": "album", "browseId": "b"}
            _screen, callback = await self._right_click(pilot, item, "albums-panel")

            callback("add_to_playlist")

            assert host.dispatched == [("add_to_playlist", item, "album")]


class TestMusicModeMergeGlue:
    async def test_the_surviving_lists_land_and_the_failures_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def _results(query: str, filter: str | None = None, limit: int = 10) -> Any:
            if filter == "albums":
                raise ValueError("albums endpoint down")
            if filter == "community_playlists":
                return None  # neither a list nor an exception — dropped
            if filter == "songs":
                return [{"videoId": "v1", "title": "Song One"}]
            return [{"resultType": "artist", "browseId": "b2"}]

        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            page = host.query_one(SearchPage)
            host.ytmusic.search = AsyncMock(side_effect=_results)

            with caplog.at_level(logging.ERROR, logger="ytm_player.ui.pages.search"):
                await page._execute_search("query")
                await pilot.pause()

            assert host.query_one("#songs-table", TrackTable).row_count == 1
            assert _panel_count_text(host, "albums-panel") == "No results"
            assert _panel_count_text(host, "artists-panel") == "1 result"
            assert _panel_count_text(host, "playlists-panel") == "No results"
            assert page.is_loading is False

            failures = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.ERROR]
            assert any("Search for albums failed: albums endpoint down" in msg for msg in failures)

    async def test_a_clean_music_search_lands_every_bucket(self) -> None:
        def _results(query: str, filter: str | None = None, limit: int = 10) -> Any:
            return {
                "songs": [{"videoId": "v1", "title": "Song One"}],
                "albums": [{"resultType": "album", "title": "A1"}],
                "artists": [{"resultType": "artist", "title": "R1"}],
                "community_playlists": [{"resultType": "playlist", "title": "P1"}],
            }.get(filter, [])

        async with _SearchHost().run_test() as pilot:
            host = pilot.app
            page = host.query_one(SearchPage)
            host.ytmusic.search = AsyncMock(side_effect=_results)

            await page._execute_search("query")
            await pilot.pause()

            assert host.query_one("#songs-table", TrackTable).row_count == 1
            assert _panel_count_text(host, "albums-panel") == "1 result"
            assert _panel_count_text(host, "artists-panel") == "1 result"
            assert _panel_count_text(host, "playlists-panel") == "1 result"


# ── The context page glue ──────────────────────────────────────────────


class _QuietYTMusic:
    """A playlist fetch that settles instantly and records library adds."""

    def __init__(self) -> None:
        self.added: list[Any] = []

    async def get_playlist(self, playlist_id, limit=None, order=None):
        return {
            "id": playlist_id,
            "title": "Quiet Playlist",
            "tracks": [],
            "author": {"name": "someone"},
            "owned": True,
        }

    async def add_to_library(self, playlist_id):
        self.added.append(playlist_id)
        return "success"


class _ContextHost(App[None]):
    def __init__(self, context_type: str, context_id: str) -> None:
        super().__init__()
        self.ytmusic = _QuietYTMusic()
        self.context_args = (context_type, context_id)
        self.radio_items: list[dict[str, Any]] = []

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield ContextPage(*self.context_args, id="context")

    async def _start_playlist_radio(self, item: dict[str, Any]) -> None:
        self.radio_items.append(item)


async def _settled_page(pilot) -> ContextPage:
    await asyncio.wait_for(pilot.app.workers.wait_for_complete(), 5)
    await pilot.pause()
    return pilot.app.query_one(ContextPage)


class TestContextHeaderButtonGlue:
    async def test_add_to_library_strips_the_vl_prefix_through_the_engine(self) -> None:
        async with _ContextHost("playlist", "PL1").run_test() as pilot:
            host = pilot.app
            page = await _settled_page(pilot)
            page._data = {"title": "Playlist", "playlistId": "VLPL9"}

            await page._add_to_library()

            assert host.ytmusic.added == ["PL9"]

    async def test_add_to_library_falls_back_to_the_context_id(self) -> None:
        async with _ContextHost("playlist", "PL1").run_test() as pilot:
            host = pilot.app
            page = await _settled_page(pilot)
            page._data = {"title": "Playlist"}

            await page._add_to_library()

            assert host.ytmusic.added == ["PL1"]

    async def test_start_radio_assembles_the_item_through_the_engine(self) -> None:
        async with _ContextHost("playlist", "PL1").run_test() as pilot:
            host = pilot.app
            page = await _settled_page(pilot)
            page._data = {"title": "My List", "playlistId": None}

            await page._start_radio()

            assert host.radio_items == [{"title": "My List", "playlistId": "PL1"}]

    async def test_start_radio_keeps_the_payloads_own_playlist_id(self) -> None:
        async with _ContextHost("playlist", "PL1").run_test() as pilot:
            host = pilot.app
            page = await _settled_page(pilot)
            page._data = {"title": "My List", "playlistId": "PL9"}

            await page._start_radio()

            assert host.radio_items == [{"title": "My List", "playlistId": "PL9"}]


def _fetch_event(state: WorkerState, result: Any = None) -> MagicMock:
    worker = MagicMock(spec=Worker)
    worker.name = "fetch_context"
    worker.result = result
    event = MagicMock()
    event.worker = worker
    event.state = state
    return event


def _bare_context_page() -> ContextPage:
    page = ContextPage("playlist", "PL1")
    object.__setattr__(page, "query_one", lambda *_a, **_kw: MagicMock())
    return page


class TestContextWorkerStateGlue:
    def test_an_empty_success_with_a_failed_load_shows_the_log_pointing_copy(self) -> None:
        page = _bare_context_page()
        page._load_failed = True

        page.on_worker_state_changed(_fetch_event(WorkerState.SUCCESS, None))

        assert page.error_message == _load_failure_message("playlist")
        assert "ytm.log" in page.error_message

    def test_an_empty_success_with_a_genuine_empty_payload_shows_the_empty_copy(self) -> None:
        page = _bare_context_page()
        page._load_failed = False

        page.on_worker_state_changed(_fetch_event(WorkerState.SUCCESS, None))

        assert page.error_message == "No data found."
