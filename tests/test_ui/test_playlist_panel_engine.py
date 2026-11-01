"""Tests for the playlist panel view engine.

The engine behind the sidebar's ``LibraryPanel`` — the row label
(``format_panel_item``), the footer count label (``panel_count_label``),
the ``/``-filter pass (``filter_panel_items``), the playlist-id matching
behind the optimistic sidebar mutations (``matches_removal`` /
``matches_count_update`` / ``matches_auto_select`` /
``is_liked_music_playlist``), the count arithmetic
(``bump_playlist_count``) and the info-bar title read (``selection_title``).
The pure-Python reference lives in ``ui/sidebars/playlist_sidebar.py``;
when the Rust extension is built the same engines run natively
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
from unittest.mock import MagicMock

import pytest

from ytm_player.ui.sidebars.playlist_sidebar import (
    LibraryPanel,
    bump_playlist_count,
    filter_panel_items,
    format_panel_item,
    is_liked_music_playlist,
    matches_auto_select,
    matches_count_update,
    matches_removal,
    panel_count_label,
    selection_title,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Row label ─────────────────────────────────────────────────────────


class TestFormatPanelItem:
    def test_title_and_count(self) -> None:
        assert format_panel_item({"title": "Roadtrip", "count": 12}) == "Roadtrip (12 tracks)"

    def test_count_none_omits_the_suffix(self) -> None:
        assert format_panel_item({"title": "Roadtrip", "count": None}) == "Roadtrip"

    def test_count_missing_omits_the_suffix(self) -> None:
        assert format_panel_item({"title": "Roadtrip", "playlistId": "PL1"}) == "Roadtrip"

    def test_count_zero_is_appended(self) -> None:
        # `is not None`, not truthiness — a 0 count still shows.
        assert format_panel_item({"title": "Empty", "count": 0}) == "Empty (0 tracks)"

    def test_name_fallback(self) -> None:
        assert format_panel_item({"name": "Chill Vibes"}) == "Chill Vibes"

    def test_unknown_when_both_keys_missing(self) -> None:
        assert format_panel_item({}) == "Unknown"

    def test_present_none_title_flows_raw(self) -> None:
        # .get("title", …) returns the present None; only a MISSING key
        # falls back to name/Unknown.
        assert format_panel_item({"title": None}) is None

    def test_present_none_title_stringifies_with_count(self) -> None:
        assert format_panel_item({"title": None, "count": 3}) == "None (3 tracks)"

    def test_raw_title_passthrough(self) -> None:
        # Without a count the title is returned raw, whatever object it is.
        assert format_panel_item({"title": 42}) == 42

    def test_int_count_renders_via_str(self) -> None:
        assert format_panel_item({"title": "X", "count": 7}) == "X (7 tracks)"
        assert format_panel_item({"title": "X", "count": "7"}) == "X (7 tracks)"


# ── Footer count label ────────────────────────────────────────────────


class TestPanelCountLabel:
    def test_all_shown_singular(self) -> None:
        assert panel_count_label(1, 1) == "1 item"

    def test_all_shown_plural(self) -> None:
        assert panel_count_label(0, 0) == "0 items"
        assert panel_count_label(3, 3) == "3 items"

    def test_filtered_shows_shown_over_total(self) -> None:
        assert panel_count_label(2, 5) == "2/5"
        assert panel_count_label(0, 5) == "0/5"


# ── Filter pass ───────────────────────────────────────────────────────


class TestFilterPanelItems:
    def test_matches_case_insensitively(self) -> None:
        items = [{"title": "Rock Classics"}, {"title": "chill"}]
        assert filter_panel_items(items, "ROCK") == [{"title": "Rock Classics"}]
        assert filter_panel_items(items, "Chill") == [{"title": "chill"}]

    def test_query_is_stripped(self) -> None:
        items = [{"title": "Rock Classics"}]
        assert filter_panel_items(items, "  Rock\t\n") == [{"title": "Rock Classics"}]

    def test_empty_query_resets_to_every_item(self) -> None:
        items = [{"title": "A"}, {"title": "B"}]
        assert filter_panel_items(items, "") == items
        assert filter_panel_items(items, "   ") == items

    def test_matches_against_the_count_suffix(self) -> None:
        # The filter matches the whole label, count included.
        items = [{"title": "Rock Classics", "count": 12}, {"title": "Jazz"}]
        assert filter_panel_items(items, "12") == [{"title": "Rock Classics", "count": 12}]
        assert filter_panel_items(items, "tracks") == [{"title": "Rock Classics", "count": 12}]

    def test_matches_against_the_unknown_label(self) -> None:
        items = [{"title": "A"}, {}]
        assert filter_panel_items(items, "unknown") == [{}]

    def test_no_match_returns_empty(self) -> None:
        assert filter_panel_items([{"title": "A"}], "zzz") == []

    def test_order_is_preserved(self) -> None:
        items = [{"title": "b x"}, {"title": "a x"}, {"title": "c x"}]
        assert [i["title"] for i in filter_panel_items(items, "x")] == ["b x", "a x", "c x"]

    def test_greek_capital_sigma_lowercases_contextually(self) -> None:
        # Python's str.lower() follows Unicode §3.13: a word-final capital
        # sigma lowercases to ς, a mid-word one to σ — and the engine's
        # lowercasing is Python's on both backends, so a query typed with
        # either sigma agrees on both.
        final = [{"title": "ΑΣ"}]
        assert filter_panel_items(final, "ας") == [{"title": "ΑΣ"}]
        assert filter_panel_items(final, "ασ") == []
        mid = [{"title": "ΑΣΑ"}]
        assert filter_panel_items(mid, "ασα") == [{"title": "ΑΣΑ"}]
        assert filter_panel_items(mid, "αςα") == []


class TestFilterIdentityAndImmutability:
    def test_kept_items_are_the_same_objects(self) -> None:
        a, b = {"title": "x"}, {"title": "y"}
        out = filter_panel_items([a, b], "")
        assert out == [a, b]
        assert all(kept is original for kept, original in zip(out, [a, b]))

        out = filter_panel_items([a, b], "x")
        assert out == [a]
        assert out[0] is a

    def test_input_list_is_not_mutated(self) -> None:
        items = [{"title": "x"}, {"title": "y"}]
        filter_panel_items(items, "x")
        assert items == [{"title": "x"}, {"title": "y"}]

    def test_empty_query_returns_a_fresh_list(self) -> None:
        items = [{"title": "x"}]
        out = filter_panel_items(items, " ")
        assert out is not items
        assert out == items


# ── Removal matching ──────────────────────────────────────────────────


class TestMatchesRemoval:
    def test_exact_match(self) -> None:
        assert matches_removal({"playlistId": "PL1"}, "PL1") is True

    def test_vl_prefixed_item_matches_the_raw_id(self) -> None:
        assert matches_removal({"browseId": "VLPL1"}, "PL1") is True

    def test_vl_prefixed_id_does_not_match_the_raw_item(self) -> None:
        # One-directional tolerance: the TARGET is never VL-expanded
        # beyond the single prefix form.
        assert matches_removal({"playlistId": "PL1"}, "VLPL1") is False

    def test_non_match(self) -> None:
        assert matches_removal({"playlistId": "PL1"}, "PL2") is False
        assert matches_removal({}, "PL1") is False

    def test_browseid_fallback(self) -> None:
        assert matches_removal({"browseId": "PL1"}, "PL1") is True
        assert matches_removal({"playlistId": "", "browseId": "PL1"}, "PL1") is True

    def test_falsy_read_matches_an_empty_target(self) -> None:
        # Both ids falsy/missing read as "" — so an empty playlist_id
        # matches an id-less row. A wart, transcribed faithfully.
        assert matches_removal({"title": "x"}, "") is True

    def test_present_none_browseid_passes_through(self) -> None:
        # The `or` chain returns the present None browseId read, and
        # None == "" is False — unlike the count-update predicate below.
        assert matches_removal({"playlistId": None, "browseId": None}, "") is False


# ── Count-update matching ─────────────────────────────────────────────


class TestMatchesCountUpdate:
    def test_exact_match(self) -> None:
        assert matches_count_update({"playlistId": "PL1"}, "PL1") is True

    def test_vl_prefixed_item_matches_the_raw_id(self) -> None:
        assert matches_count_update({"browseId": "VLPL1"}, "PL1") is True
        assert matches_count_update({"playlistId": "VLPL1"}, "PL1") is True

    def test_vl_prefixed_id_matches_the_raw_item(self) -> None:
        # Tolerant in BOTH directions, unlike the removal predicate.
        assert matches_count_update({"browseId": "PL1"}, "VLPL1") is True

    def test_vl_stripped_target_matches_the_raw_item(self) -> None:
        assert matches_count_update({"playlistId": "PL1"}, "VLPL1") is True

    def test_non_match(self) -> None:
        assert matches_count_update({"playlistId": "PL1"}, "PL2") is False
        assert matches_count_update({}, "PL1") is False

    def test_falsy_read_matches_an_empty_target(self) -> None:
        # The trailing `or ""` maps a falsy read to "", so an empty
        # playlist_id matches an id-less row here too.
        assert matches_count_update({"title": "x"}, "") is True

    def test_falsy_read_maps_through_even_when_none(self) -> None:
        # …and unlike the removal predicate, a present-None browseId also
        # lands on "" through that trailing operand.
        assert matches_count_update({"playlistId": None, "browseId": None}, "") is True


# ── Count arithmetic ──────────────────────────────────────────────────


class TestBumpPlaylistCount:
    def test_none_stays_none(self) -> None:
        assert bump_playlist_count(None, 5) is None

    def test_int_math(self) -> None:
        assert bump_playlist_count(5, 3) == 8
        assert bump_playlist_count(10, -2) == 8

    def test_clamps_at_zero(self) -> None:
        assert bump_playlist_count(5, -8) == 0
        assert bump_playlist_count(0, -1) == 0

    def test_bools_count_as_their_int(self) -> None:
        assert bump_playlist_count(True, 1) == 2
        assert bump_playlist_count(False, 5) == 5

    def test_floats_truncate_toward_zero(self) -> None:
        assert bump_playlist_count(5.7, 0) == 5
        assert bump_playlist_count(-5.7, 10) == 5

    def test_decimal_strings_parse(self) -> None:
        assert bump_playlist_count("5", 3) == 8
        assert bump_playlist_count(" 5 ", 3) == 8
        assert bump_playlist_count("+5", 3) == 8
        assert bump_playlist_count("-5", 10) == 5

    def test_underscore_separators_parse(self) -> None:
        assert bump_playlist_count("5_0", 0) == 50


# ── Selection title ───────────────────────────────────────────────────


class TestSelectionTitle:
    def test_title_then_name_then_empty(self) -> None:
        assert selection_title({"title": "T"}) == "T"
        assert selection_title({"name": "N"}) == "N"
        assert selection_title({}) == ""

    def test_present_empty_title_wins_over_name(self) -> None:
        assert selection_title({"title": "", "name": "N"}) == ""

    def test_raw_value_passthrough(self) -> None:
        assert selection_title({"title": 42}) == 42


# ── Liked Music exclusion ─────────────────────────────────────────────


class TestIsLikedMusicPlaylist:
    def test_lm_under_either_key(self) -> None:
        assert is_liked_music_playlist({"playlistId": "LM"}) is True
        assert is_liked_music_playlist({"browseId": "LM"}) is True

    def test_playlistid_wins_when_truthy(self) -> None:
        assert is_liked_music_playlist({"playlistId": "LM", "browseId": "other"}) is True
        assert is_liked_music_playlist({"playlistId": "", "browseId": "LM"}) is True

    def test_other_ids_are_not_liked_music(self) -> None:
        assert is_liked_music_playlist({"playlistId": "PL1"}) is False
        assert is_liked_music_playlist({}) is False
        assert is_liked_music_playlist({"title": "Liked Music"}) is False


# ── Auto-select matching ──────────────────────────────────────────────


class TestMatchesAutoSelect:
    def test_exact_match_under_either_key(self) -> None:
        assert matches_auto_select({"playlistId": "PL1"}, "PL1") is True
        assert matches_auto_select({"browseId": "PL1"}, "PL1") is True

    def test_no_vl_tolerance_in_either_direction(self) -> None:
        assert matches_auto_select({"playlistId": "PL1"}, "VLPL1") is False
        assert matches_auto_select({"browseId": "VLPL1"}, "PL1") is False

    def test_missing_browseid_reads_as_none(self) -> None:
        # No "" default: an id-less row never matches an empty target,
        # unlike the removal/count-update predicates.
        assert matches_auto_select({"title": "x"}, "") is False

    def test_non_match(self) -> None:
        assert matches_auto_select({"playlistId": "PL1"}, "PL2") is False
        assert matches_auto_select({}, "PL1") is False


# ── Dict protocol ─────────────────────────────────────────────────────


class TestDictProtocol:
    def test_field_reads_go_through_get(self) -> None:
        # A dict subclass overriding `.get` must steer the engine exactly
        # like it steers the reference: both go through the method.
        class Overriding(dict):
            def get(self, key: str, default: Any = None) -> Any:
                if key == "title":
                    return "Overridden"
                return super().get(key, default)

        item = Overriding(count=3)
        assert format_panel_item(item) == "Overridden (3 tracks)"

    def test_pid_reads_go_through_get(self) -> None:
        class Overriding(dict):
            def get(self, key: str, default: Any = None) -> Any:
                if key == "playlistId":
                    return "PL9"
                return super().get(key, default)

        assert matches_removal(Overriding(), "PL9") is True


# ── Panel wiring (the widgets are thin callers) ───────────────────────


class TestPanelWiring:
    """The LibraryPanel methods delegate to the engine and only do I/O."""

    @staticmethod
    def _bare_panel(items: list[dict[str, Any]]) -> tuple[LibraryPanel, list[list[dict]]]:
        panel = LibraryPanel.__new__(LibraryPanel)
        panel._items = list(items)
        panel._filtered_items = list(items)
        rebuilds: list[list[dict]] = []

        def fake_rebuild(shown: list[dict[str, Any]]) -> None:
            rebuilds.append(list(shown))

        panel._rebuild_list = fake_rebuild  # type: ignore[method-assign]
        return panel, rebuilds

    def test_remove_item_prunes_both_lists(self) -> None:
        panel, rebuilds = self._bare_panel(
            [
                {"playlistId": "PL1", "title": "A"},
                {"playlistId": "PL2", "title": "B"},
                {"browseId": "VLPL1", "title": "C"},
            ]
        )
        panel.remove_item("PL1")
        assert panel._items == [{"playlistId": "PL2", "title": "B"}]
        assert panel._filtered_items == [{"playlistId": "PL2", "title": "B"}]
        assert rebuilds == [[{"playlistId": "PL2", "title": "B"}]]

    def test_on_input_changed_filters_through_the_engine(self) -> None:
        panel, rebuilds = self._bare_panel(
            [
                {"playlistId": "PL1", "title": "Alpha"},
                {"playlistId": "PL2", "title": "Beta"},
            ]
        )
        event = MagicMock()
        event.input.id = "ps-playlists-filter"
        event.value = "  alp "
        panel.on_input_changed(event)
        assert panel._filtered_items == [{"playlistId": "PL1", "title": "Alpha"}]
        assert rebuilds[-1] == [{"playlistId": "PL1", "title": "Alpha"}]

    def test_on_input_changed_ignores_foreign_inputs(self) -> None:
        panel, rebuilds = self._bare_panel([{"playlistId": "PL1", "title": "A"}])
        event = MagicMock()
        event.input.id = "search-box"
        event.value = "zzz"
        panel.on_input_changed(event)
        assert panel._filtered_items == [{"playlistId": "PL1", "title": "A"}]
        assert rebuilds == []

    def test_update_item_count_leaves_an_unknown_count_alone(self) -> None:
        panel, rebuilds = self._bare_panel([{"playlistId": "PL1", "count": None}])
        panel.update_item_count("PL1", 3)
        assert panel._items[0]["count"] is None
        # No rebuild either — nothing to show.
        assert rebuilds == []


# ── Total engine (native only) ────────────────────────────────────────


class TestTotalityEngine:
    """The native engine stays total where the reference would raise.

    A non-dict item's `.get` raises AttributeError in the reference and a
    raw non-string title's `.lower()` raises it too; a count `int()` would
    reject raises ValueError. The native engine degrades instead — the
    convention the ported engines before it follow — so these tests only
    run with the extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_dict_items_never_match(self) -> None:
        items: list[Any] = [{"title": "x"}, 42, "str", None]
        assert filter_panel_items(items, "x") == [{"title": "x"}]
        assert filter_panel_items(items, "unknown") == []

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_dict_items_are_copied_by_the_empty_query(self) -> None:
        # The empty-query reset never reads the items, so even a non-dict
        # row flows into the copy — like the reference's list(items).
        assert filter_panel_items([42, {"title": "x"}], "") == [42, {"title": "x"}]

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_string_raw_title_never_matches(self) -> None:
        # {"title": 42} has no count, so the label is the raw 42 — the
        # reference would raise on .lower(); the engine skips the row.
        assert filter_panel_items([{"title": 42}], "42") == []
        # With a count the label IS the stringified "42 (1 tracks)" and
        # matches, on both backends.
        assert filter_panel_items([{"title": 42, "count": 1}], "42") == [{"title": 42, "count": 1}]

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_int_proof_counts_no_op(self) -> None:
        assert bump_playlist_count("abc", 1) is None
        assert bump_playlist_count("0x10", 0) is None
        assert bump_playlist_count([1], 1) is None

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_underscore_rules_match_int(self) -> None:
        # int("1__0"), int("1_") and int("_1") all raise — the engine
        # no-ops instead.
        assert bump_playlist_count("1__0", 0) is None
        assert bump_playlist_count("1_", 0) is None
        assert bump_playlist_count("_1", 0) is None

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_ascii_digits_no_op(self) -> None:
        # Python's int() accepts non-ASCII decimal digits (int("٥") == 5);
        # the engine's parser is ASCII-only, so such a count no-ops. A
        # documented divergence — the loader only ever stores int-or-None.
        assert bump_playlist_count("٥", 0) is None

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_huge_ints_clamp_by_sign(self) -> None:
        assert bump_playlist_count(10**30, 0) == 9223372036854775807
        assert bump_playlist_count(-(10**30), 0) == 0


# ── Native vs fallback parity ─────────────────────────────────────────

# One entry per engine case; the child interpreter below JSON round-trips
# them, so every value is a JSON type. Shapes where the pure-Python
# reference would raise (non-dict items under a query, raw non-string
# titles in the filter, int()-proof counts) stay out of the parity cases
# and are pinned by TestTotalityEngine above.

_LABEL_CASES: list[dict[str, Any]] = [
    {"title": "Roadtrip", "count": 12},
    {"title": "Roadtrip"},
    {"name": "Chill Vibes"},
    {},
    {"title": None},
    {"title": None, "count": 3},
    {"title": "", "count": 0},
    {"count": 5},
    {"title": 123},
    {"title": 123, "count": 2},
    {"title": "t", "count": "7"},
    {"title": "t", "count": True},
    {"title": "t", "count": False},
    {"title": "Mix", "count": None},
    {"title": "ΑΣ", "count": 1},
    {"title": "Roadtrip", "playlistId": "PL1"},
]

_COUNT_LABEL_CASES: list[tuple[int, int]] = [
    (0, 0),
    (1, 1),
    (1, 0),
    (3, 3),
    (5, 3),
    (0, 5),
    (1, 2),
    (7, 7),
    (2, 1),
    (-1, 3),
    (0, -1),
]

_FILTER_ITEMS: list[dict[str, Any]] = [
    {"playlistId": "P1", "title": "Rock Classics", "count": 12},
    {"playlistId": "P2", "title": "chill"},
    {"playlistId": "P3", "name": "Jazz"},
    {"playlistId": "P4", "title": "ΑΣ"},
    {"playlistId": "P5"},
]

_FILTER_CASES: list[dict[str, Any]] = [
    {"items": _FILTER_ITEMS, "query": ""},
    {"items": _FILTER_ITEMS, "query": "   "},
    {"items": _FILTER_ITEMS, "query": "rock"},
    {"items": _FILTER_ITEMS, "query": "ROCK"},
    {"items": _FILTER_ITEMS, "query": "  Rock "},
    {"items": _FILTER_ITEMS, "query": "\tROCK\n"},
    # NBSP is whitespace to CPython's strip() and to the engine's set.
    {"items": _FILTER_ITEMS, "query": "\u00a0Rock\u00a0"},
    {"items": _FILTER_ITEMS, "query": "classics"},
    {"items": _FILTER_ITEMS, "query": "12"},
    {"items": _FILTER_ITEMS, "query": "tracks"},
    {"items": _FILTER_ITEMS, "query": "jazz"},
    {"items": _FILTER_ITEMS, "query": "unknown"},
    {"items": _FILTER_ITEMS, "query": "chill"},
    {"items": _FILTER_ITEMS, "query": "rock classics"},
    # Sigma shapes against the ΑΣ title, whose §3.13 lowercase is "ας":
    # "ασ" is \u03b1\u03c3 (plain), "ας" is \u03b1\u03c2 (final).
    {"items": _FILTER_ITEMS, "query": "\u03b1\u03c3"},
    {"items": _FILTER_ITEMS, "query": "\u03b1\u03c2"},
    {"items": _FILTER_ITEMS, "query": "\u03c3"},
    {"items": _FILTER_ITEMS, "query": "\u03c2"},
    {"items": _FILTER_ITEMS, "query": "zzz"},
    {"items": [{"title": "İ"}], "query": "i̇"},
    {"items": [{"title": 42, "count": 1}], "query": "42"},
    {"items": [42, {"title": "x"}], "query": ""},
]

_REMOVAL_CASES: list[tuple[dict[str, Any], str]] = [
    ({"playlistId": "PL1"}, "PL1"),
    ({"playlistId": "PL1"}, "VLPL1"),
    ({"browseId": "VLPL1"}, "PL1"),
    ({"browseId": "PL1"}, "PL2"),
    ({}, "PL1"),
    ({"playlistId": None, "browseId": "PL1"}, "PL1"),
    ({"playlistId": None, "browseId": None}, "PL1"),
    ({"playlistId": None, "browseId": None}, ""),
    ({"title": "x"}, ""),
    ({"playlistId": "LM"}, "LM"),
]

_COUNT_UPDATE_CASES: list[tuple[dict[str, Any], str]] = [
    ({"playlistId": "PL1"}, "PL1"),
    ({"browseId": "VLPL1"}, "PL1"),
    ({"browseId": "PL1"}, "VLPL1"),
    ({"playlistId": "VLPL1"}, "PL1"),
    ({"playlistId": "PLX"}, "VLPLX"),
    ({"playlistId": "PL1"}, "PL2"),
    ({"playlistId": "PL1"}, "PL1VL"),
    ({}, "PL1"),
    ({}, ""),
    ({"playlistId": None, "browseId": None}, ""),
    ({"browseId": None}, ""),
    ({"title": "x"}, ""),
]

_BUMP_CASES: list[tuple[Any, int]] = [
    (None, 5),
    (5, 3),
    (5, -8),
    (0, 0),
    (10, -2),
    (True, 1),
    (False, 5),
    (5.7, 0),
    (-5.7, 10),
    (5.0, 2),
    ("5", 3),
    (" 5 ", 3),
    ("+5", 3),
    ("-5", 10),
    ("5_0", 0),
    ("0", -3),
]

_TITLE_CASES: list[dict[str, Any]] = [
    {"title": "T"},
    {"name": "N"},
    {},
    {"title": None},
    {"title": "", "name": "N"},
    {"title": 42},
]

_LIKED_CASES: list[dict[str, Any]] = [
    {"playlistId": "LM"},
    {"browseId": "LM"},
    {"playlistId": "LM", "browseId": "other"},
    {"playlistId": "", "browseId": "LM"},
    {"playlistId": "PL1"},
    {},
    {"playlistId": None, "browseId": None},
    {"title": "Liked Music"},
]

_AUTO_SELECT_CASES: list[tuple[dict[str, Any], str]] = [
    ({"playlistId": "PL1"}, "PL1"),
    ({"browseId": "PL1"}, "PL1"),
    ({"playlistId": "PL1"}, "VLPL1"),
    ({"browseId": "VLPL1"}, "PL1"),
    ({"title": "x"}, ""),
    ({}, "PL1"),
    ({"playlistId": None, "browseId": None}, "PL1"),
]

_PARITY_CASES: dict[str, list[Any]] = {
    "label": _LABEL_CASES,
    "count_label": _COUNT_LABEL_CASES,
    "filter": _FILTER_CASES,
    "removal": _REMOVAL_CASES,
    "count_update": _COUNT_UPDATE_CASES,
    "bump": _BUMP_CASES,
    "title": _TITLE_CASES,
    "liked": _LIKED_CASES,
    "auto_select": _AUTO_SELECT_CASES,
}

_CHILD_PROG = r"""
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.sidebars.playlist_sidebar import (
    bump_playlist_count,
    filter_panel_items,
    format_panel_item,
    is_liked_music_playlist,
    matches_auto_select,
    matches_count_update,
    matches_removal,
    panel_count_label,
    selection_title,
)

data = json.load(sys.stdin)
out = {
    "label": [format_panel_item(c) for c in data["label_cases"]],
    "count_label": [panel_count_label(s, t) for s, t in data["count_label_cases"]],
    "filter": [
        filter_panel_items(c["items"], c["query"]) for c in data["filter_cases"]
    ],
    "removal": [matches_removal(i, p) for i, p in data["removal_cases"]],
    "count_update": [matches_count_update(i, p) for i, p in data["count_update_cases"]],
    "bump": [bump_playlist_count(c, d) for c, d in data["bump_cases"]],
    "title": [selection_title(c) for c in data["title_cases"]],
    "liked": [is_liked_music_playlist(c) for c in data["liked_cases"]],
    "auto_select": [matches_auto_select(i, p) for i, p in data["auto_select_cases"]],
}
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_outcomes() -> dict:
    """Run the engines in this process — native when the extension is
    built, the fallback otherwise (in which case the parity test is
    skipped)."""
    return {
        "label": [format_panel_item(c) for c in _LABEL_CASES],
        "count_label": [panel_count_label(s, t) for s, t in _COUNT_LABEL_CASES],
        "filter": [filter_panel_items(c["items"], c["query"]) for c in _FILTER_CASES],
        "removal": [matches_removal(i, p) for i, p in _REMOVAL_CASES],
        "count_update": [matches_count_update(i, p) for i, p in _COUNT_UPDATE_CASES],
        "bump": [bump_playlist_count(c, d) for c, d in _BUMP_CASES],
        "title": [selection_title(c) for c in _TITLE_CASES],
        "liked": [is_liked_music_playlist(c) for c in _LIKED_CASES],
        "auto_select": [matches_auto_select(i, p) for i, p in _AUTO_SELECT_CASES],
    }


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "label_cases": _LABEL_CASES,
            "count_label_cases": _COUNT_LABEL_CASES,
            "filter_cases": _FILTER_CASES,
            "removal_cases": _REMOVAL_CASES,
            "count_update_cases": _COUNT_UPDATE_CASES,
            "bump_cases": _BUMP_CASES,
            "title_cases": _TITLE_CASES,
            "liked_cases": _LIKED_CASES,
            "auto_select_cases": _AUTO_SELECT_CASES,
        }
    )
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native panel view engines
    must produce exactly the same outcomes as the pure-Python references
    — the same labels, footer counts and filtered views, the same
    playlist-id matches under every id shape, the same count arithmetic,
    and the same title reads — for every case, so the two implementations
    cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key, cases in _PARITY_CASES.items():
            for case, got, want in zip(cases, native[key], fallback[key]):
                assert got == want, (
                    f"native and fallback {key} outcomes differ for case {case!r}: "
                    f"native={got!r} fallback={want!r}"
                )
