"""Tests for the sidebar pane-routing engine.

The engine behind ``SidebarMixin``'s Ctrl+w pane traversal and per-page
sidebar bookkeeping: ``sidebar_visible_for_page`` /
``toggle_sidebar_state`` / ``ensure_sidebar_visible`` (the per-page
playlist-sidebar visibility), ``visible_panes`` (the left-to-right
visible pane list), ``pane_of_focused`` (resolving the focused widget's
pane from its ancestor chain), ``pane_focus_target`` (the focus gate:
Playlists always focusable — it auto-shows — lyrics only while open,
anything else means content) and ``next_pane_left`` / ``next_pane_right``
/ ``next_pane_cycle`` (the movement decisions). The pure-Python reference
lives in ``app/_sidebar.py``; when the Rust extension is built the same
engines run natively (``_ytm_player_native``).

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
from types import SimpleNamespace

import pytest

from ytm_player.app._sidebar import (
    PANE_CONTENT,
    PANE_LYRICS,
    PANE_PLAYLISTS,
    ensure_sidebar_visible,
    next_pane_cycle,
    next_pane_left,
    next_pane_right,
    pane_focus_target,
    pane_of_focused,
    sidebar_visible_for_page,
    toggle_sidebar_state,
    visible_panes,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Per-page playlist-sidebar visibility ─────────────────────────────


class TestSidebarVisibleForPage:
    def test_stored_value_is_returned(self) -> None:
        assert sidebar_visible_for_page({"library": True}, "library", False) is True
        assert sidebar_visible_for_page({"library": False}, "library", True) is False

    def test_missing_entry_reads_the_default(self) -> None:
        assert sidebar_visible_for_page({"search": True}, "library", True) is True
        assert sidebar_visible_for_page({"search": True}, "library", False) is False

    def test_empty_page_reads_library(self) -> None:
        # No page mounted yet — the read falls back to Library.
        assert sidebar_visible_for_page({"library": False}, "", True) is False
        assert sidebar_visible_for_page({"library": True}, "", False) is True

    def test_other_pages_keep_their_own_entries(self) -> None:
        assert sidebar_visible_for_page({"library": False, "search": True}, "search", False) is True

    def test_raw_stored_value_flows_through(self) -> None:
        # A hand-edited session.json may store non-bools; the read returns
        # them raw, and callers decide by truthiness.
        assert sidebar_visible_for_page({"library": 1}, "library", False) == 1
        assert sidebar_visible_for_page({"library": ""}, "library", True) == ""

    def test_present_none_is_returned_not_the_default(self) -> None:
        # dict.get: a stored None is a value, not a missing entry.
        assert sidebar_visible_for_page({"library": None}, "library", True) is None


class TestToggleSidebarState:
    def test_missing_entry_toggles_the_default(self) -> None:
        outcome = toggle_sidebar_state({}, "library", True)
        assert outcome["visible"] is False
        assert outcome["sidebar_per_page"] == {"library": False}

        outcome = toggle_sidebar_state({}, "library", False)
        assert outcome["visible"] is True
        assert outcome["sidebar_per_page"] == {"library": True}

    def test_stored_value_is_flipped(self) -> None:
        assert toggle_sidebar_state({"library": True}, "library", True)["visible"] is False
        assert toggle_sidebar_state({"library": False}, "library", True)["visible"] is True

    def test_falsy_stored_value_toggles_true(self) -> None:
        # `not 0` is True — Python truthiness decides, not bool type.
        outcome = toggle_sidebar_state({"library": 0}, "library", True)
        assert outcome["visible"] is True
        assert outcome["sidebar_per_page"]["library"] is True

    def test_empty_page_toggles_library(self) -> None:
        outcome = toggle_sidebar_state({"library": True}, "", True)
        assert outcome["visible"] is False
        assert outcome["sidebar_per_page"] == {"library": False}

    def test_other_pages_are_preserved(self) -> None:
        per_page = {"library": True, "search": False, "queue": True}
        outcome = toggle_sidebar_state(per_page, "search", True)
        assert outcome["sidebar_per_page"] == {"library": True, "search": True, "queue": True}

    def test_input_map_is_not_mutated(self) -> None:
        per_page = {"library": True}
        toggle_sidebar_state(per_page, "library", True)
        assert per_page == {"library": True}

    def test_outcome_map_is_a_fresh_dict(self) -> None:
        per_page = {"library": True}
        outcome = toggle_sidebar_state(per_page, "library", True)
        assert outcome["sidebar_per_page"] is not per_page


class TestEnsureSidebarVisible:
    def test_hidden_page_becomes_visible(self) -> None:
        outcome = ensure_sidebar_visible({"library": False}, "library", True)
        assert outcome["changed"] is True
        assert outcome["sidebar_per_page"] == {"library": True}

    def test_missing_entry_with_falsy_default_becomes_visible(self) -> None:
        outcome = ensure_sidebar_visible({}, "library", False)
        assert outcome["changed"] is True
        assert outcome["sidebar_per_page"] == {"library": True}

    def test_already_visible_is_unchanged(self) -> None:
        outcome = ensure_sidebar_visible({"library": True}, "library", False)
        assert outcome["changed"] is False
        assert outcome["sidebar_per_page"] == {"library": True}

    def test_missing_entry_with_truthy_default_is_unchanged(self) -> None:
        outcome = ensure_sidebar_visible({}, "library", True)
        assert outcome["changed"] is False
        assert outcome["sidebar_per_page"] == {}

    def test_falsy_stored_value_is_upgraded(self) -> None:
        outcome = ensure_sidebar_visible({"library": 0}, "library", True)
        assert outcome["changed"] is True
        assert outcome["sidebar_per_page"]["library"] is True

    def test_empty_page_ensures_library(self) -> None:
        outcome = ensure_sidebar_visible({"library": False}, "", True)
        assert outcome["changed"] is True
        assert outcome["sidebar_per_page"] == {"library": True}

    def test_input_map_is_not_mutated(self) -> None:
        per_page = {"library": False, "search": True}
        ensure_sidebar_visible(per_page, "library", True)
        assert per_page == {"library": False, "search": True}


# ── Visible pane list ────────────────────────────────────────────────


class TestVisiblePanes:
    def test_all_three_visible(self) -> None:
        assert visible_panes(True, True) == [PANE_PLAYLISTS, PANE_CONTENT, PANE_LYRICS]

    def test_sidebar_and_content(self) -> None:
        assert visible_panes(True, False) == [PANE_PLAYLISTS, PANE_CONTENT]

    def test_content_and_lyrics(self) -> None:
        assert visible_panes(False, True) == [PANE_CONTENT, PANE_LYRICS]

    def test_content_only(self) -> None:
        assert visible_panes(False, False) == [PANE_CONTENT]

    def test_content_is_always_present_and_order_is_left_to_right(self) -> None:
        # Whatever the flags say, content is never dropped and the order
        # stays [playlists, content, lyrics].
        order = {PANE_PLAYLISTS: 0, PANE_CONTENT: 1, PANE_LYRICS: 2}
        for playlist_visible in (True, False):
            for lyrics_open in (True, False):
                panes = visible_panes(playlist_visible, lyrics_open)
                assert PANE_CONTENT in panes
                ranks = [order[pane] for pane in panes]
                assert ranks == sorted(ranks)

    def test_truthy_non_bool_flags_count_as_visible(self) -> None:
        # The flags are read with Python truthiness, like the fallback's
        # `if flag:` — a raw stored `1` includes its pane.
        assert visible_panes(1, 0) == [PANE_PLAYLISTS, PANE_CONTENT]
        assert visible_panes("", "yes") == [PANE_CONTENT, PANE_LYRICS]


# ── Pane resolution from the focused widget ──────────────────────────


def _chain(*ids: str | None) -> SimpleNamespace:
    """A widget-like ancestor chain: the first id is the focused widget,
    the last its outermost ancestor; each node exposes ``id``/``parent``
    like a Textual widget stand-in."""
    root: SimpleNamespace | None = None
    for node_id in reversed(ids):
        root = SimpleNamespace(id=node_id, parent=root)
    assert root is not None
    return root


class TestPaneOfFocused:
    def test_focus_under_playlist_sidebar(self) -> None:
        assert pane_of_focused(_chain("ps-playlists-list", "playlist-sidebar")) == PANE_PLAYLISTS

    def test_focus_under_lyrics_sidebar(self) -> None:
        assert pane_of_focused(_chain("ls-scroll", "lyrics-sidebar")) == PANE_LYRICS

    def test_focus_under_main_content(self) -> None:
        assert pane_of_focused(_chain("library-tracks", "main-content")) == PANE_CONTENT

    def test_focused_widget_is_the_pane_container_itself(self) -> None:
        assert pane_of_focused(_chain("playlist-sidebar")) == PANE_PLAYLISTS
        assert pane_of_focused(_chain("lyrics-sidebar")) == PANE_LYRICS
        assert pane_of_focused(_chain("main-content")) == PANE_CONTENT

    def test_deep_chain_finds_the_container_at_the_root(self) -> None:
        widget = _chain("row", "table", "scroll", "main-content")
        assert pane_of_focused(widget) == PANE_CONTENT

    def test_unknown_chain_defaults_to_content(self) -> None:
        # NOT the _active_pane hint — that fallback only applies when
        # nothing is focused, which the mixin handles before calling.
        assert pane_of_focused(_chain("stray", "also-stray")) == PANE_CONTENT

    def test_idless_nodes_are_skipped(self) -> None:
        assert pane_of_focused(_chain(None, None, "lyrics-sidebar")) == PANE_LYRICS

    def test_node_without_id_attribute_reads_as_idless(self) -> None:
        node = SimpleNamespace(parent=SimpleNamespace(id="main-content", parent=None))
        assert pane_of_focused(node) == PANE_CONTENT

    def test_focused_none_means_content(self) -> None:
        # The mixin routes None through the _active_pane hint instead, but
        # the engine itself stays total: an empty walk is content.
        assert pane_of_focused(None) == PANE_CONTENT

    def test_id_equality_is_python_eq(self) -> None:
        # The container match goes through `==`, so an id object that
        # compares equal to a pane container resolves to that pane on
        # both backends.

        class _WeirdId:
            def __eq__(self, other: object) -> bool:
                return other == "playlist-sidebar"

            def __hash__(self) -> int:
                return 0

        assert pane_of_focused(SimpleNamespace(id=_WeirdId(), parent=None)) == PANE_PLAYLISTS


# ── Focus gate ───────────────────────────────────────────────────────


class TestPaneFocusTarget:
    def test_playlists_is_always_focusable(self) -> None:
        assert pane_focus_target(PANE_PLAYLISTS, False) == PANE_PLAYLISTS
        assert pane_focus_target(PANE_PLAYLISTS, True) == PANE_PLAYLISTS

    def test_lyrics_is_focusable_only_while_open(self) -> None:
        assert pane_focus_target(PANE_LYRICS, True) == PANE_LYRICS
        assert pane_focus_target(PANE_LYRICS, False) is None

    def test_content_is_focusable(self) -> None:
        assert pane_focus_target(PANE_CONTENT, False) == PANE_CONTENT
        assert pane_focus_target(PANE_CONTENT, True) == PANE_CONTENT

    def test_unknown_pane_normalises_to_content(self) -> None:
        assert pane_focus_target("nonsense", True) == PANE_CONTENT
        assert pane_focus_target("", False) == PANE_CONTENT

    def test_truthy_non_bool_open_flag_opens_lyrics(self) -> None:
        assert pane_focus_target(PANE_LYRICS, 1) == PANE_LYRICS
        assert pane_focus_target(PANE_LYRICS, 0) is None


# ── Movement decisions ───────────────────────────────────────────────


class TestNextPaneLeft:
    def test_from_lyrics_steps_to_content(self) -> None:
        assert next_pane_left(PANE_LYRICS) == PANE_CONTENT

    def test_from_content_focuses_playlists(self) -> None:
        assert next_pane_left(PANE_CONTENT) == PANE_PLAYLISTS

    def test_from_playlists_stays_on_playlists(self) -> None:
        # Left never fails — refocusing the sidebar is harmless and keeps
        # the state machine total.
        assert next_pane_left(PANE_PLAYLISTS) == PANE_PLAYLISTS

    def test_unknown_pane_goes_to_playlists(self) -> None:
        assert next_pane_left("nonsense") == PANE_PLAYLISTS


class TestNextPaneRight:
    def test_from_playlists_steps_to_content(self) -> None:
        assert next_pane_right(PANE_PLAYLISTS, True) == PANE_CONTENT
        assert next_pane_right(PANE_PLAYLISTS, False) == PANE_CONTENT

    def test_from_content_enters_lyrics_when_open(self) -> None:
        assert next_pane_right(PANE_CONTENT, True) == PANE_LYRICS

    def test_from_content_stays_when_lyrics_closed(self) -> None:
        assert next_pane_right(PANE_CONTENT, False) is None

    def test_from_lyrics_stays(self) -> None:
        assert next_pane_right(PANE_LYRICS, True) is None
        assert next_pane_right(PANE_LYRICS, False) is None

    def test_unknown_pane_stays(self) -> None:
        assert next_pane_right("nonsense", True) is None

    def test_truthy_non_bool_open_flag_enters_lyrics(self) -> None:
        assert next_pane_right(PANE_CONTENT, "open") == PANE_LYRICS
        assert next_pane_right(PANE_CONTENT, "") is None


class TestNextPaneCycle:
    def test_cycles_rightward_through_all_panes(self) -> None:
        assert next_pane_cycle(PANE_PLAYLISTS, True, True) == PANE_CONTENT
        assert next_pane_cycle(PANE_CONTENT, True, True) == PANE_LYRICS
        assert next_pane_cycle(PANE_LYRICS, True, True) == PANE_PLAYLISTS

    def test_wraps_without_lyrics(self) -> None:
        assert next_pane_cycle(PANE_PLAYLISTS, True, False) == PANE_CONTENT
        assert next_pane_cycle(PANE_CONTENT, True, False) == PANE_PLAYLISTS

    def test_wraps_without_playlists(self) -> None:
        assert next_pane_cycle(PANE_CONTENT, False, True) == PANE_LYRICS
        assert next_pane_cycle(PANE_LYRICS, False, True) == PANE_CONTENT

    def test_content_only_wraps_to_itself(self) -> None:
        assert next_pane_cycle(PANE_CONTENT, False, False) == PANE_CONTENT

    def test_hidden_current_pane_is_treated_as_content(self) -> None:
        # Current is lyrics but lyrics is closed: start from content.
        assert next_pane_cycle(PANE_LYRICS, True, False) == PANE_PLAYLISTS

    def test_unknown_pane_is_treated_as_content(self) -> None:
        assert next_pane_cycle("nonsense", True, True) == PANE_LYRICS
        assert next_pane_cycle("nonsense", False, False) == PANE_CONTENT

    def test_truthy_non_bool_visibility_flags(self) -> None:
        assert next_pane_cycle(PANE_CONTENT, 1, 0) == PANE_PLAYLISTS
        assert next_pane_cycle(PANE_CONTENT, "", "open") == PANE_LYRICS


# ── Native/fallback parity ───────────────────────────────────────────

# Case tables for the parity runner: every argument is JSON so the same
# cases can run in this process (native, when built) and in a child
# interpreter with the extension blocked (pure-Python fallback).

_STATE_CASES = [
    # (per_page, page, default)
    ({"library": True}, "library", True),
    ({"library": False}, "library", True),
    ({"library": True}, "library", False),
    ({"library": False}, "library", False),
    ({}, "library", True),
    ({}, "library", False),
    ({"search": True}, "library", True),
    ({"library": False}, "", True),
    ({"library": True}, "", False),
    ({"library": False, "search": True}, "search", False),
    ({"library": 1}, "library", False),
    ({"library": 0}, "library", True),
    ({"library": ""}, "library", True),
    ({"library": "x"}, "library", False),
    ({"library": None}, "library", True),
    ({"library": None}, "library", False),
    ({"queue": True}, "queue", False),
    ({"context": 0}, "context", True),
]

_PANES_CASES = [
    # (playlist_sidebar_visible, lyrics_open) — bools and the raw
    # truthy/falsy shapes a stored value can take.
    (True, True),
    (True, False),
    (False, True),
    (False, False),
    (1, 0),
    ("yes", ""),
    (0, "open"),
    ([], 2),
]

_CHAIN_SPECS = [
    # Ancestor id chains, focused widget first (null = id-less node).
    ["ps-playlists-list", "playlist-sidebar"],
    ["ls-scroll", "lyrics-sidebar"],
    ["library-tracks", "main-content"],
    ["playlist-sidebar"],
    ["lyrics-sidebar"],
    ["main-content"],
    ["row", "table", "scroll", "main-content"],
    ["stray", "also-stray"],
    [None, None, "lyrics-sidebar"],
    ["ps-playlists-list", None, "playlist-sidebar", "screen"],
    ["main-content", "playlist-sidebar"],
    [None],
    ["ps-playlists-list", "playlist-sidebar", "main-content"],
]

_TARGET_CASES = [
    # (pane, lyrics_open)
    ("playlists", False),
    ("playlists", True),
    ("lyrics", True),
    ("lyrics", False),
    ("lyrics", 1),
    ("lyrics", 0),
    ("content", False),
    ("content", True),
    ("nonsense", True),
    ("", False),
]

_LEFT_CASES = ["lyrics", "content", "playlists", "nonsense", ""]

_RIGHT_CASES = [
    # (current_pane, lyrics_open)
    ("playlists", True),
    ("playlists", False),
    ("content", True),
    ("content", False),
    ("content", "open"),
    ("content", ""),
    ("lyrics", True),
    ("lyrics", False),
    ("nonsense", True),
    ("", True),
]

_CYCLE_CASES = [
    # (current_pane, playlist_sidebar_visible, lyrics_open)
    ("playlists", True, True),
    ("content", True, True),
    ("lyrics", True, True),
    ("playlists", True, False),
    ("content", True, False),
    ("content", False, True),
    ("lyrics", False, True),
    ("content", False, False),
    ("lyrics", True, False),
    ("playlists", False, True),
    ("nonsense", True, True),
    ("nonsense", False, False),
    ("content", 1, 0),
    ("content", "", "open"),
]

_CHILD_PROG = """
import json
import sys
from types import SimpleNamespace

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._sidebar import (
    ensure_sidebar_visible,
    next_pane_cycle,
    next_pane_left,
    next_pane_right,
    pane_focus_target,
    pane_of_focused,
    sidebar_visible_for_page,
    toggle_sidebar_state,
    visible_panes,
)


def chain(ids):
    root = None
    for node_id in reversed(ids):
        root = SimpleNamespace(id=node_id, parent=root)
    return root


payload = json.load(sys.stdin)
out = {
    "visible": [
        sidebar_visible_for_page(per_page, page, default)
        for per_page, page, default in payload["state_cases"]
    ],
    "toggle": [
        toggle_sidebar_state(per_page, page, default)
        for per_page, page, default in payload["state_cases"]
    ],
    "ensure": [
        ensure_sidebar_visible(per_page, page, default)
        for per_page, page, default in payload["state_cases"]
    ],
    "panes": [visible_panes(pv, lo) for pv, lo in payload["panes_cases"]],
    "chain": [pane_of_focused(chain(ids)) for ids in payload["chain_specs"]],
    "chain_none": pane_of_focused(None),
    "target": [pane_focus_target(pane, lo) for pane, lo in payload["target_cases"]],
    "left": [next_pane_left(cur) for cur in payload["left_cases"]],
    "right": [next_pane_right(cur, lo) for cur, lo in payload["right_cases"]],
    "cycle": [
        next_pane_cycle(cur, pv, lo) for cur, pv, lo in payload["cycle_cases"]
    ],
}
json.dump(out, sys.stdout, sort_keys=True)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "state_cases": _STATE_CASES,
            "panes_cases": _PANES_CASES,
            "chain_specs": _CHAIN_SPECS,
            "target_cases": _TARGET_CASES,
            "left_cases": _LEFT_CASES,
            "right_cases": _RIGHT_CASES,
            "cycle_cases": _CYCLE_CASES,
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


def _native_outcomes() -> dict:
    """Run the engines in this process — native when the extension is
    built, the fallback otherwise (in which case the parity test is
    skipped)."""
    return {
        "visible": [
            sidebar_visible_for_page(per_page, page, default)
            for per_page, page, default in _STATE_CASES
        ],
        "toggle": [
            toggle_sidebar_state(per_page, page, default)
            for per_page, page, default in _STATE_CASES
        ],
        "ensure": [
            ensure_sidebar_visible(per_page, page, default)
            for per_page, page, default in _STATE_CASES
        ],
        "panes": [visible_panes(pv, lo) for pv, lo in _PANES_CASES],
        "chain": [pane_of_focused(_chain(*ids)) for ids in _CHAIN_SPECS],
        "chain_none": pane_of_focused(None),
        "target": [pane_focus_target(pane, lo) for pane, lo in _TARGET_CASES],
        "left": [next_pane_left(cur) for cur in _LEFT_CASES],
        "right": [next_pane_right(cur, lo) for cur, lo in _RIGHT_CASES],
        "cycle": [next_pane_cycle(cur, pv, lo) for cur, pv, lo in _CYCLE_CASES],
    }


class TestNativeParity:
    """When the Rust extension is built, the native pane-routing engines
    must produce exactly the same outcomes as the pure-Python references
    — the same raw visibility values, the same fresh maps, the same pane
    lists, resolutions, gates and movements — for every case, so the two
    implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key, cases in _PARITY_CASE_KEYS.items():
            if cases is None:
                # The single no-focus case ("chain_none").
                assert json.dumps(native[key], sort_keys=True) == json.dumps(
                    fallback[key], sort_keys=True
                ), f"native and fallback {key} outcomes differ"
                continue
            for case, got, want in zip(cases, native[key], fallback[key]):
                assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), (
                    f"native and fallback {key} outcomes differ for case {case!r}: "
                    f"native={got!r} fallback={want!r}"
                )


# The case list backing each parity key, for failure messages.
_PARITY_CASE_KEYS = {
    "visible": _STATE_CASES,
    "toggle": _STATE_CASES,
    "ensure": _STATE_CASES,
    "panes": _PANES_CASES,
    "chain": _CHAIN_SPECS,
    "chain_none": None,
    "target": _TARGET_CASES,
    "left": _LEFT_CASES,
    "right": _RIGHT_CASES,
    "cycle": _CYCLE_CASES,
}
