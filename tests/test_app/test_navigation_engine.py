"""Tests for the navigation stack engine (back/forward stacks + purge).

``resolve_navigation`` is the engine behind ``NavigationMixin.navigate_to``:
the back/forward stack machine (leaving-page snapshots, the 20-entry caps,
the same-page-back shortcut, forward-history invalidation, cached-state
restore). ``purge_dead_playlist_refs`` is the pure half of
``_purge_playlist_nav_state``. The pure-Python reference lives in
``app/_navigation.py``; when the Rust extension is built the same engines
run natively (``_ytm_player_native``).

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

import pytest

from ytm_player.app._navigation import (
    _MAX_NAV_STACK,
    PAGE_NAMES,
    purge_dead_playlist_refs,
    resolve_navigation,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _resolve(
    current_page: str = "",
    current_kwargs: dict | None = None,
    nav_stack: list | None = None,
    forward_stack: list | None = None,
    page_state_cache: dict | None = None,
    current_page_state: dict | None = None,
    page_name: str = "library",
    kwargs: dict | None = None,
    page_names: tuple[str, ...] = PAGE_NAMES,
    max_stack: int = _MAX_NAV_STACK,
) -> dict:
    """Call the engine with the given state, filling in empty defaults."""
    return resolve_navigation(
        current_page,
        current_kwargs or {},
        nav_stack or [],
        forward_stack or [],
        page_state_cache or {},
        current_page_state,
        page_name,
        kwargs or {},
        page_names=page_names,
        max_stack=max_stack,
    )


# ── First navigation & plain switches ─────────────────────────────────


class TestSwitches:
    def test_first_navigation_switches_without_pushing(self):
        outcome = _resolve(current_page="", page_name="library")
        assert outcome["action"] == "switch"
        assert outcome["page"] == "library"
        assert outcome["kwargs"] == {}
        assert outcome["nav_stack"] == []
        assert outcome["forward_stack"] == []
        assert outcome["reason"] is None

    def test_new_page_pushes_leaving_snapshot(self):
        outcome = _resolve(
            current_page="library",
            current_kwargs={"playlist_id": "PL1"},
            page_name="search",
        )
        assert outcome["action"] == "switch"
        assert outcome["page"] == "search"
        assert outcome["nav_stack"] == [("library", {"playlist_id": "PL1"})]
        assert outcome["forward_stack"] == []

    def test_snapshot_merges_cached_state_over_ctor_kwargs(self):
        outcome = _resolve(
            current_page="library",
            current_kwargs={"playlist_id": "PL1", "cursor_row": 0},
            page_state_cache={"library": {"cursor_row": 7}},
            page_name="search",
        )
        # Cached nav state applies over the constructor kwargs.
        assert outcome["nav_stack"] == [("library", {"playlist_id": "PL1", "cursor_row": 7})]

    def test_new_navigation_invalidates_forward_history(self):
        outcome = _resolve(
            current_page="library",
            forward_stack=[("search", {"q": "x"}), ("queue", {})],
            page_name="queue",
        )
        assert outcome["nav_stack"] == [("library", {})]
        assert outcome["forward_stack"] == []

    def test_same_page_with_kwargs_remounts_without_pushing(self):
        outcome = _resolve(
            current_page="library",
            page_name="library",
            kwargs={"playlist_id": "PL2"},
        )
        assert outcome["action"] == "switch"
        assert outcome["kwargs"] == {"playlist_id": "PL2"}
        assert outcome["nav_stack"] == []

    def test_unknown_page_is_noop(self):
        outcome = _resolve(current_page="library", page_name="nonsense")
        assert outcome["action"] == "noop"
        assert outcome["reason"] == "unknown_page"
        assert outcome["page"] == "nonsense"


# ── Back / forward ────────────────────────────────────────────────────


class TestBackForward:
    def test_back_pops_and_pushes_current_onto_forward(self):
        outcome = _resolve(
            current_page="search",
            current_kwargs={"q": "queen"},
            nav_stack=[("library", {"playlist_id": "PL1"})],
            page_name="back",
        )
        assert outcome["action"] == "switch"
        assert outcome["page"] == "library"
        # The popped entry's kwargs flow through unchanged.
        assert outcome["kwargs"] == {"playlist_id": "PL1"}
        assert outcome["nav_stack"] == []
        assert outcome["forward_stack"] == [("search", {"q": "queen"})]

    def test_back_with_empty_stack_goes_to_library(self):
        outcome = _resolve(current_page="search", page_name="back")
        assert outcome["action"] == "switch"
        assert outcome["page"] == "library"

    def test_back_with_empty_stack_and_no_current_page(self):
        outcome = _resolve(current_page="", page_name="back")
        assert outcome["action"] == "switch"
        assert outcome["page"] == "library"
        assert outcome["nav_stack"] == []
        assert outcome["forward_stack"] == []

    def test_forward_pops_and_pushes_current_onto_nav(self):
        outcome = _resolve(
            current_page="library",
            current_kwargs={"playlist_id": "PL1"},
            forward_stack=[("search", {"q": "x"})],
            page_name="forward",
        )
        assert outcome["action"] == "switch"
        assert outcome["page"] == "search"
        assert outcome["kwargs"] == {"q": "x"}
        assert outcome["nav_stack"] == [("library", {"playlist_id": "PL1"})]
        assert outcome["forward_stack"] == []

    def test_forward_with_empty_stack_is_noop(self):
        outcome = _resolve(
            current_page="search",
            page_state_cache={"search": {"cursor_row": 3}},
            current_page_state={"cursor_row": 9},
            page_name="forward",
        )
        assert outcome["action"] == "noop"
        assert outcome["reason"] == "no_forward_history"
        # The cache refresh still happened before the no-op was decided.
        assert outcome["page_state_cache"] == {"search": {"cursor_row": 9}}

    def test_back_pop_pushes_forward_snapshot_with_cached_state(self):
        outcome = _resolve(
            current_page="library",
            current_kwargs={"playlist_id": "PL1"},
            page_state_cache={"library": {"cursor_row": 4}},
            nav_stack=[("queue", {})],
            page_name="back",
        )
        assert outcome["page"] == "queue"
        assert outcome["forward_stack"] == [("library", {"playlist_id": "PL1", "cursor_row": 4})]

    def test_back_then_forward_round_trips(self):
        state = {
            "current_page": "library",
            "current_kwargs": {"playlist_id": "PL1"},
            "nav_stack": [("queue", {"cursor_row": 2})],
            "forward_stack": [],
        }
        first = _resolve(**state, page_name="back")
        assert first["page"] == "queue"
        assert first["kwargs"] == {"cursor_row": 2}
        second = _resolve(
            current_page=first["page"],
            current_kwargs=first["kwargs"],
            nav_stack=first["nav_stack"],
            forward_stack=first["forward_stack"],
            page_state_cache=first["page_state_cache"],
            page_name="forward",
        )
        assert second["action"] == "switch"
        assert second["page"] == "library"
        assert second["kwargs"] == {"playlist_id": "PL1"}
        assert second["nav_stack"] == [("queue", {"cursor_row": 2})]
        assert second["forward_stack"] == []


# ── Same-page back ────────────────────────────────────────────────────


class TestSamePageBack:
    def test_clicking_same_page_pops_back(self):
        outcome = _resolve(
            current_page="library",
            nav_stack=[("search", {"q": "x"})],
            page_name="library",
        )
        assert outcome["action"] == "switch"
        assert outcome["page"] == "search"
        assert outcome["kwargs"] == {"q": "x"}
        assert outcome["nav_stack"] == []

    def test_clicking_same_page_without_history_is_noop(self):
        outcome = _resolve(current_page="library", page_name="library")
        assert outcome["action"] == "noop"
        assert outcome["reason"] == "same_page_no_history"

    def test_back_to_same_page_then_same_page_back(self):
        # back pops ("library", {}) while ON library — the popped page
        # equals the current one with empty kwargs, so the same-page-back
        # rule pops again and finds nothing: a no-op.
        outcome = _resolve(
            current_page="library",
            nav_stack=[("library", {})],
            page_name="back",
        )
        assert outcome["action"] == "noop"
        assert outcome["reason"] == "same_page_no_history"
        # The pop still happened — the caller applies the returned state.
        assert outcome["nav_stack"] == []


# ── Stack caps ────────────────────────────────────────────────────────


class TestCaps:
    def test_nav_stack_capped_at_max(self):
        nav = [(f"p{i}", {}) for i in range(5)]
        outcome = _resolve(
            current_page="library",
            nav_stack=nav,
            page_name="search",
            max_stack=3,
        )
        assert outcome["nav_stack"] == [
            ("p3", {}),
            ("p4", {}),
            ("library", {}),
        ]

    def test_forward_stack_capped_on_back(self):
        forward = [("search", {}), ("queue", {})]
        outcome = _resolve(
            current_page="library",
            nav_stack=[("help", {})],
            forward_stack=forward,
            page_name="back",
            max_stack=3,
        )
        assert outcome["forward_stack"] == [
            ("search", {}),
            ("queue", {}),
            ("library", {}),
        ]
        # One more back overflows the cap: the oldest forward entry drops.
        outcome = _resolve(
            current_page="help",
            nav_stack=[("browse", {})],
            forward_stack=outcome["forward_stack"],
            page_name="back",
            max_stack=3,
        )
        assert outcome["forward_stack"] == [
            ("queue", {}),
            ("library", {}),
            ("help", {}),
        ]

    def test_cap_of_zero_keeps_everything(self):
        # Python's x[-0:] == x[0:] — a zero cap keeps the whole stack.
        nav = [("a", {}), ("b", {})]
        outcome = _resolve(
            current_page="library",
            nav_stack=nav,
            page_name="search",
            max_stack=0,
        )
        assert outcome["nav_stack"] == [("a", {}), ("b", {}), ("library", {})]


# ── Cache refresh & restore ───────────────────────────────────────────


class TestCacheRefreshAndRestore:
    def test_fresh_state_replaces_cache_entry(self):
        fresh = {"cursor_row": 5}
        outcome = _resolve(
            current_page="library",
            page_state_cache={"library": {"cursor_row": 1}},
            current_page_state=fresh,
            page_name="search",
        )
        assert outcome["page_state_cache"]["library"] is fresh

    def test_empty_fresh_state_drops_cache_entry(self):
        outcome = _resolve(
            current_page="library",
            page_state_cache={"library": {"cursor_row": 1}, "search": {}},
            current_page_state={},
            page_name="search",
        )
        assert "library" not in outcome["page_state_cache"]

    def test_none_fresh_state_keeps_cache_entry(self):
        cached = {"cursor_row": 1}
        outcome = _resolve(
            current_page="library",
            page_state_cache={"library": cached},
            current_page_state=None,
            page_name="search",
        )
        assert outcome["page_state_cache"]["library"] is cached

    def test_no_current_page_leaves_cache_alone(self):
        cached = {"cursor_row": 1}
        outcome = _resolve(
            current_page="",
            page_state_cache={"library": cached},
            current_page_state={"cursor_row": 9},
            page_name="library",
        )
        assert outcome["page_state_cache"]["library"] is cached

    def test_empty_kwargs_restore_cached_state(self):
        outcome = _resolve(
            current_page="library",
            page_state_cache={"search": {"cursor_row": 5}},
            page_name="search",
        )
        assert outcome["kwargs"] == {"cursor_row": 5}

    def test_restored_kwargs_are_a_copy_not_the_cache_entry(self):
        cached = {"cursor_row": 5}
        outcome = _resolve(
            current_page="library",
            page_state_cache={"search": cached},
            page_name="search",
        )
        assert outcome["kwargs"] == {"cursor_row": 5}
        assert outcome["kwargs"] is not cached
        assert outcome["page_state_cache"]["search"] is cached

    def test_explicit_kwargs_beat_the_cache(self):
        outcome = _resolve(
            current_page="library",
            page_state_cache={"search": {"cursor_row": 5}},
            page_name="search",
            kwargs={"q": "x"},
        )
        assert outcome["kwargs"] == {"q": "x"}

    def test_popped_kwargs_flow_through_by_reference(self):
        popped = {"q": "x"}
        outcome = _resolve(
            current_page="search",
            nav_stack=[("library", popped)],
            page_name="back",
        )
        assert outcome["kwargs"] is popped


# ── Input immutability ────────────────────────────────────────────────


class TestInputsNotMutated:
    def test_resolve_navigation_never_mutates_its_inputs(self):
        nav = [("library", {"playlist_id": "PL1"})]
        forward = [("queue", {})]
        cache = {"library": {"cursor_row": 1}}
        current_kwargs = {"playlist_id": "PL1"}
        kwargs = {}
        _resolve(
            current_page="library",
            current_kwargs=current_kwargs,
            nav_stack=nav,
            forward_stack=forward,
            page_state_cache=cache,
            current_page_state={"cursor_row": 9},
            page_name="back",
            kwargs=kwargs,
        )
        assert nav == [("library", {"playlist_id": "PL1"})]
        assert forward == [("queue", {})]
        assert cache == {"library": {"cursor_row": 1}}
        assert current_kwargs == {"playlist_id": "PL1"}
        assert kwargs == {}

    def test_purge_never_mutates_its_inputs(self):
        nav = [("library", {"playlist_id": "PL1"}), ("search", {"q": "x"})]
        forward = [("context", {"context_type": "playlist", "context_id": "PL2"})]
        cache = {"library": {"playlist_id": "PL1", "cursor_row": 3}}
        purge_dead_playlist_refs({"PL1", "PL2"}, "PL1", nav, forward, cache)
        assert nav == [("library", {"playlist_id": "PL1"}), ("search", {"q": "x"})]
        assert forward == [("context", {"context_type": "playlist", "context_id": "PL2"})]
        assert cache == {"library": {"playlist_id": "PL1", "cursor_row": 3}}


# ── Purge ─────────────────────────────────────────────────────────────


class TestPurge:
    def test_drops_dead_library_cache_entry(self):
        outcome = purge_dead_playlist_refs(
            {"VLPL9"},
            None,
            [],
            [],
            {"library": {"playlist_id": "VLPL9", "cursor_row": 3}},
        )
        assert outcome["page_state_cache"] == {}

    def test_keeps_unrelated_library_cache_entry(self):
        cache = {"library": {"playlist_id": "OTHER", "cursor_row": 3}}
        outcome = purge_dead_playlist_refs({"VLPL9"}, None, [], [], cache)
        assert outcome["page_state_cache"] is not cache
        assert outcome["page_state_cache"] == cache

    def test_drops_dead_playlist_entries_from_both_stacks(self):
        outcome = purge_dead_playlist_refs(
            {"VLPL9", "PL9"},
            None,
            [("library", {"playlist_id": "VLPL9"}), ("search", {"cursor_row": 2})],
            [("library", {"playlist_id": "PL9"})],
            {},
        )
        assert outcome["nav_stack"] == [("search", {"cursor_row": 2})]
        assert outcome["forward_stack"] == []

    def test_playlist_context_entries_drop_via_context_id(self):
        outcome = purge_dead_playlist_refs(
            {"PL9"},
            None,
            [
                ("context", {"context_type": "playlist", "context_id": "PL9"}),
                ("context", {"context_type": "album", "context_id": "PL9"}),
            ],
            [],
            {},
        )
        # An album context page with the same id is NOT a playlist.
        assert outcome["nav_stack"] == [("context", {"context_type": "album", "context_id": "PL9"})]

    def test_missing_keys_match_nothing(self):
        outcome = purge_dead_playlist_refs({"PL9"}, None, [("search", {"q": "x"})], [], {})
        assert outcome["nav_stack"] == [("search", {"q": "x"})]

    def test_active_id_cleared_when_dead(self):
        outcome = purge_dead_playlist_refs({"VLPL9"}, "VLPL9", [], [], {})
        assert outcome["active_library_playlist_id"] is None

    def test_active_id_kept_when_alive(self):
        outcome = purge_dead_playlist_refs({"VLPL9"}, "OTHER", [], [], {})
        assert outcome["active_library_playlist_id"] == "OTHER"

    def test_active_id_none_stays_none(self):
        outcome = purge_dead_playlist_refs({"VLPL9"}, None, [], [], {})
        assert outcome["active_library_playlist_id"] is None


# ── Native / fallback parity ──────────────────────────────────────────
#
# The child interpreter runs the pure-Python reference engines with the
# extension import blocked and prints the outcomes as JSON; the parity
# tests compare them against the loaded (native) engine case for case.

_NAV_CASES = [
    # First navigation from nothing.
    ("", {}, [], [], {}, None, "library", {}),
    # Plain switch pushes a snapshot of the leaving page.
    ("library", {"playlist_id": "PL1"}, [], [], {}, None, "search", {}),
    # Snapshot merges cached state over constructor kwargs.
    (
        "library",
        {"playlist_id": "PL1", "cursor_row": 0},
        [],
        [],
        {"library": {"cursor_row": 7}},
        {"cursor_row": 9},
        "search",
        {},
    ),
    # Empty fresh state drops the cached entry.
    ("library", {}, [], [], {"library": {"cursor_row": 7}, "search": {"q": 1}}, {}, "search", {}),
    # None fresh state keeps the cached entry.
    ("library", {}, [], [], {"library": {"cursor_row": 7}}, None, "search", {}),
    # back: pops, pushes current onto forward.
    ("search", {"q": "queen"}, [("library", {"playlist_id": "PL1"})], [], {}, None, "back", {}),
    # back with an empty stack goes to library (kwargs pass through).
    ("search", {"q": "x"}, [], [], {}, None, "back", {}),
    # back with no current page: no forward push.
    ("", {}, [("library", {})], [], {}, None, "back", {}),
    # forward: pops, pushes current onto nav.
    ("library", {"playlist_id": "PL1"}, [], [("search", {"q": "x"})], {}, None, "forward", {}),
    # forward with an empty stack is a no-op (cache already refreshed).
    ("search", {}, [], [], {"search": {"cursor_row": 3}}, {"cursor_row": 9}, "forward", {}),
    # unknown page is a no-op.
    ("library", {}, [], [], {}, None, "nonsense", {}),
    # a dead name popped from the stack is rejected as unknown.
    ("search", {}, [("bogus", {"x": 1})], [], {}, None, "back", {}),
    # same-page click pops back.
    ("library", {}, [("search", {"q": "x"})], [], {}, None, "library", {}),
    # same-page click without history is a no-op.
    ("library", {}, [], [], {}, None, "library", {}),
    # back popping the page you're on triggers same-page-back again.
    ("library", {}, [("library", {})], [], {}, None, "back", {}),
    # same page with explicit kwargs remounts.
    ("library", {}, [], [], {}, None, "library", {"playlist_id": "PL2"}),
    # a new navigation invalidates the forward history.
    ("library", {}, [], [("search", {"q": "x"}), ("queue", {})], {}, None, "queue", {}),
    # empty kwargs restore the target's cached state.
    ("library", {}, [], [], {"search": {"cursor_row": 5}}, None, "search", {}),
    # explicit kwargs beat the cache.
    ("library", {}, [], [], {"search": {"cursor_row": 5}}, None, "search", {"q": "x"}),
    # nav stack capped at max (overflow pushes).
    (
        "library",
        {},
        [("p0", {}), ("p1", {}), ("p2", {}), ("p3", {}), ("p4", {})],
        [],
        {},
        None,
        "search",
        {},
    ),
    # forward stack capped on back.
    ("library", {}, [("help", {})], [("search", {}), ("queue", {})], {}, None, "back", {}),
    # max_stack=0 keeps everything (x[-0:] == x[0:]).
    ("library", {}, [("a", {}), ("b", {})], [], {}, None, "search", {}),
    # a noop still returns the refreshed/popped state.
    (
        "search",
        {},
        [],
        [("queue", {})],
        {"search": {"cursor_row": 1}},
        {"cursor_row": 8},
        "forward",
        {"q": 1},
    ),
]

_NAV_CASE_ARGS = [case + (list(PAGE_NAMES), _MAX_NAV_STACK) for case in _NAV_CASES]
# The cap cases need non-default max_stack values.
_NAV_CASE_ARGS[19] = _NAV_CASES[19] + (list(PAGE_NAMES), 3)
_NAV_CASE_ARGS[20] = _NAV_CASES[20] + (list(PAGE_NAMES), 3)
_NAV_CASE_ARGS[21] = _NAV_CASES[21] + (list(PAGE_NAMES), 0)

_PURGE_CASES = [
    (["PL1", "PL2"], None, [], [], {}),
    (["PL1"], "PL1", [], [], {"library": {"playlist_id": "PL1", "cursor_row": 3}}),
    (["PL1"], "OTHER", [], [], {"library": {"playlist_id": "OTHER", "cursor_row": 3}}),
    (
        ["PL1", "PL2"],
        None,
        [("library", {"playlist_id": "PL1"}), ("search", {"cursor_row": 2})],
        [("library", {"playlist_id": "PL2"})],
        {"library": {"playlist_id": "PL1", "cursor_row": 3}, "search": {"q": 1}},
    ),
    (
        ["PL9"],
        None,
        [
            ("context", {"context_type": "playlist", "context_id": "PL9"}),
            ("context", {"context_type": "album", "context_id": "PL9"}),
            ("context", {"context_type": "playlist", "context_id": "OTHER"}),
        ],
        [("context", {"context_type": "playlist", "context_id": "PL9"})],
        {},
    ),
    (["PL1"], None, [("search", {"q": "x"}), ("queue", {})], [("help", {})], {}),
    # A dead playlist_id with no context fields, and vice versa.
    (["PL9"], None, [("library", {"playlist_id": "PL9"}), ("x", {"context_id": "PL9"})], [], {}),
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._navigation import purge_dead_playlist_refs, resolve_navigation

payload = json.load(sys.stdin)
nav_outcomes = [
    resolve_navigation(*case) for case in payload["nav_cases"]
]
purge_outcomes = [
    purge_dead_playlist_refs(set(dead), active, nav, forward, cache)
    for dead, active, nav, forward, cache in payload["purge_cases"]
]
json.dump(
    {"nav": nav_outcomes, "purge": purge_outcomes},
    sys.stdout,
    sort_keys=True,
)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps({"nav_cases": _NAV_CASE_ARGS, "purge_cases": _PURGE_CASES})
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native navigation engines
    must produce exactly the same outcomes as the pure-Python references
    — same action, same page and kwargs, same stack and cache contents —
    for every case, so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_resolve_navigation_matches_fallback(self):
        native = [resolve_navigation(*case) for case in _NAV_CASE_ARGS]
        fallback = _fallback_outcomes()["nav"]
        assert len(native) == len(fallback)
        for case, got, want in zip(_NAV_CASE_ARGS, native, fallback):
            assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), (
                f"native and fallback navigation outcomes differ for case {case!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_purge_dead_playlist_refs_matches_fallback(self):
        native = [
            purge_dead_playlist_refs(set(dead), active, nav, forward, cache)
            for dead, active, nav, forward, cache in _PURGE_CASES
        ]
        fallback = _fallback_outcomes()["purge"]
        assert len(native) == len(fallback)
        for case, got, want in zip(_PURGE_CASES, native, fallback):
            assert json.dumps(got, sort_keys=True) == json.dumps(want, sort_keys=True), (
                f"native and fallback purge outcomes differ for case {case!r}"
            )
