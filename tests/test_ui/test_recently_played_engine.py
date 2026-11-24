"""Tests for the Recently Played history-merge engine.

The engine behind the Recently Played page's caches: occurrence stamping
and identity (``_stamp`` / ``_occurrence_key``), the accepted-play
supersession (``_with_accepted_play``) and its local-tab and pending-play
counterparts, and the All-tab local+account composition (``_compose_all``)
— plus the page's own display tails: the empty-state copy pick
(``_recent_empty_message`` — the active tab naming only its own failure,
the All tab naming every missing source), the footer line
(``_recent_footer_text`` — the count, and the partial-result notices
said out loud and first) and the nav-state remember-gates
(``_recent_nav_state`` — the non-default tab, the above-first row).
The pure-Python reference lives in ``ui/pages/recently_played.py``; when
the Rust extension is built the same engine runs natively
(``_ytm_player_native.with_accepted_play`` & co).

The unit tests below assert the engine's contract on whichever backend is
loaded. When the extension is built, the parity tests additionally compare
the native engine against the pure-Python reference (run in a child
interpreter with the extension import blocked) case for case, so the two
implementations cannot drift apart. Occurrence ids the engine allocates
freshly are compared through their observable properties — distinctness
and first-appearance order — because the native and fallback counters are
independent by design (only one backend is ever active in a process, and a
fresh id's only contract is that it differs from every id the cache
already holds).
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from ytm_player.ui.pages.recently_played import (
    _DEFAULT_TAB,
    _MAX_TRACKS,
    _TAB_ALL,
    _TAB_LOCAL,
    _TAB_YTM,
    _compose_all,
    _enrich,
    _occurrence_key,
    _recent_empty_message,
    _recent_footer_text,
    _recent_nav_state,
    _stamp,
    _with_accepted_play,
    merge_pending_account_plays,
    park_pending_play,
    with_accepted_local_play,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

# Occurrence ids in the deterministic cases live far above anything either
# backend's fresh-id counter can reach, so the parity comparison can tell
# them apart from engine-allocated ids (see _normalized).
_EXPLICIT_ID = 10**9


def _cached(video_id: str, title: str = "", occurrence: int | None = None) -> dict:
    """A cache row with a deterministic (explicit) occurrence id."""
    row: dict = {"video_id": video_id, "title": title}
    if occurrence is not None:
        row["_occurrence"] = occurrence
    return row


# ── Stamping and occurrence identity ─────────────────────────────────


class TestStamp:
    def test_result_is_a_shallow_copy_and_the_input_is_untouched(self):
        track = {"video_id": "a", "title": "A", "artists": [{"name": "X"}]}
        stamped = _stamp(track, _EXPLICIT_ID + 1)
        assert stamped is not track
        assert stamped["_occurrence"] == _EXPLICIT_ID + 1
        assert stamped["artists"] is track["artists"]  # shallow: values shared
        stamped["title"] = "changed"
        assert track["title"] == "A"

    def test_explicit_occurrence_is_kept_verbatim(self):
        assert _stamp({"video_id": "a"}, _EXPLICIT_ID + 1)["_occurrence"] == _EXPLICIT_ID + 1

    def test_fresh_occurrences_are_distinct_and_increasing(self):
        ids = [_stamp({"video_id": "a"})["_occurrence"] for _ in range(3)]
        assert len(set(ids)) == 3
        assert ids == sorted(ids)

    def test_an_existing_stamp_is_overwritten(self):
        stamped = _stamp({"video_id": "a", "_occurrence": 5}, _EXPLICIT_ID + 2)
        assert stamped["_occurrence"] == _EXPLICIT_ID + 2


class TestOccurrenceKey:
    def test_a_stamped_row_keys_by_its_stamp(self):
        assert _occurrence_key({"_occurrence": _EXPLICIT_ID + 1, "video_id": "a"}) == (
            _EXPLICIT_ID + 1
        )

    def test_an_unstamped_row_keys_by_its_video_id(self):
        assert _occurrence_key({"videoId": "abc"}) == "abc"
        assert _occurrence_key({"video_id": "xyz"}) == "xyz"

    def test_an_explicit_none_stamp_keys_as_none(self):
        # ``.get(key, default)`` substitutes only when the key is ABSENT —
        # a row stamped with None keys as None, not as its video id.
        assert _occurrence_key({"_occurrence": None, "video_id": "a"}) is None


# ── Accepted plays ───────────────────────────────────────────────────


class TestWithAcceptedPlay:
    def test_untouched_rows_keep_identity_and_the_cache_is_untouched(self):
        other, latest, older = (
            _cached("b", "B", _EXPLICIT_ID + 3),
            _cached("a", "A", _EXPLICIT_ID + 1),
            _cached("a", "A", _EXPLICIT_ID + 2),
        )
        cache = [other, latest, older]

        updated = _with_accepted_play(cache, {"video_id": "a", "title": "A again"})

        assert updated[0]["title"] == "A again"
        assert updated[1] is other
        assert updated[2] is older
        assert cache == [other, latest, older]  # the input list is left alone

    def test_a_row_without_a_stamp_gets_a_fresh_id_on_move(self):
        cache = [{"video_id": "a", "title": "unstamped"}]

        updated = _with_accepted_play(cache, {"video_id": "a", "title": "A"})

        assert isinstance(updated[0]["_occurrence"], int)

    def test_a_row_stamped_none_gets_a_fresh_id_on_move(self):
        cache = [{"video_id": "a", "_occurrence": None}]

        updated = _with_accepted_play(cache, {"video_id": "a"})

        assert updated[0]["_occurrence"] is not None

    def test_rows_without_an_id_match_a_track_without_an_id(self):
        # ``get_video_id`` is "" on both sides; "" == "" supersedes.
        cache = [{"title": "no id"}]

        updated = _with_accepted_play(cache, {"title": "still no id"})

        assert len(updated) == 1
        assert updated[0]["title"] == "still no id"


class TestWithAcceptedLocalPlay:
    def test_prepends_a_copy_and_makes_same_track_rows_way(self):
        a, b = _cached("a", "A"), _cached("b", "B")

        updated = with_accepted_local_play([a, b, _cached("a", "A old")], {"video_id": "a"}, 10)

        assert [row["video_id"] for row in updated] == ["a", "b"]
        assert updated[0] is not None
        assert b in updated

    def test_caps_at_cap(self):
        cache = [_cached(f"v{i}") for i in range(5)]

        updated = with_accepted_local_play(cache, {"video_id": "new"}, 3)

        assert [row["video_id"] for row in updated] == ["new", "v0", "v1"]

    def test_cap_zero_empties_the_cache(self):
        assert with_accepted_local_play([_cached("a")], {"video_id": "n"}, 0) == []

    def test_kept_rows_are_the_same_objects_and_the_head_is_a_copy(self):
        a, b = _cached("a"), _cached("b")
        played = {"video_id": "c", "title": "C"}

        updated = with_accepted_local_play([a, b], played, 10)

        assert updated[0] is not played  # a copy, so later mutation can't leak
        assert updated[0] == played
        assert updated[1] is a
        assert updated[2] is b


class TestParkPendingPlay:
    def test_newest_goes_on_top_and_same_track_entries_are_dropped(self):
        old_same = (2, {"video_id": "a", "title": "old"})
        other = (1, {"video_id": "b"})

        parked = park_pending_play([old_same, other], 3, {"video_id": "a", "title": "new"}, 10)

        assert parked[0] == (3, {"video_id": "a", "title": "new"})
        assert [entry[1]["video_id"] for entry in parked[1:]] == ["b"]

    def test_kept_entries_are_the_same_tuple_objects(self):
        e1, e2 = (1, {"video_id": "b"}), (2, {"video_id": "c"})

        parked = park_pending_play([e1, e2], 3, {"video_id": "a"}, 10)

        assert parked[1] is e1
        assert parked[2] is e2

    def test_the_parked_track_is_a_copy(self):
        played = {"video_id": "a", "title": "A"}

        parked = park_pending_play([], 7, played, 10)

        assert parked[0][1] is not played
        assert parked[0][1] == played

    def test_cap_trims_from_the_front(self):
        pending = [(i, {"video_id": f"v{i}"}) for i in range(5)]

        parked = park_pending_play(pending, 9, {"video_id": "new"}, 3)

        assert [entry[0] for entry in parked] == [9, 0, 1]


class TestMergePendingAccountPlays:
    def test_nothing_pending_returns_the_feed_itself(self):
        feed = [_cached("a")]

        assert merge_pending_account_plays(feed, 0, []) is feed

    def test_a_play_after_the_fetch_supersedes_the_feed_latest_row(self):
        latest, older = (
            _cached("a", "old", _EXPLICIT_ID + 1),
            _cached("a", "older", _EXPLICIT_ID + 2),
        )
        feed = [latest, older]

        merged = merge_pending_account_plays(feed, 5, [(6, {"video_id": "a", "title": "A"})])

        assert [row["title"] for row in merged] == ["A", "older"]
        assert merged[0]["_occurrence"] == latest["_occurrence"]
        assert merged[1] is older

    def test_a_play_before_the_fetch_already_in_the_feed_stays_put(self):
        feed = [_cached("a", "server order")]

        merged = merge_pending_account_plays(feed, 5, [(4, {"video_id": "a", "title": "mine"})])

        assert [row["title"] for row in merged] == ["server order"]

    def test_a_play_before_the_feed_not_listed_merges_behind_newer_plays(self):
        feed = [_cached("z")]

        merged = merge_pending_account_plays(
            feed, 5, [(6, {"video_id": "new"}), (4, {"video_id": "pre"})]
        )

        assert [row["video_id"] for row in merged] == ["new", "pre", "z"]

    def test_the_newest_play_ends_on_top(self):
        feed = [_cached("z")]

        merged = merge_pending_account_plays(
            feed, 0, [(2, {"video_id": "y"}), (1, {"video_id": "x", "title": "older"})]
        )

        assert [row["video_id"] for row in merged] == ["y", "x", "z"]


# ── Enrichment ───────────────────────────────────────────────────────


class TestEnrich:
    def test_fills_absent_none_and_empty_fields_only(self):
        track = {"video_id": "a", "title": "Local", "artist": "", "album": None}
        extra = {
            "title": "Server title",
            "artist": "Server artist",
            "album": "Server album",
            "album_id": "AL1",
        }

        merged = _enrich(track, extra)

        assert merged == {
            "video_id": "a",
            "title": "Local",
            "artist": "Server artist",
            "album": "Server album",
            "album_id": "AL1",
        }

    def test_zero_and_false_count_as_present(self):
        track = {"plays": 0, "liked": False}

        merged = _enrich(track, {"plays": 9, "liked": True})

        assert merged == {"plays": 0, "liked": False}

    def test_an_empty_extra_value_enriches_nothing(self):
        track = {"video_id": "a"}

        merged = _enrich(track, {"thumbnail_url": "", "artist": None})

        assert merged == {"video_id": "a"}

    def test_non_string_values_fill_absent_keys(self):
        track = {"video_id": "a"}

        merged = _enrich(track, {"artists": [{"name": "X"}], "is_video": False, "duration": 0})

        assert merged == {
            "video_id": "a",
            "artists": [{"name": "X"}],
            "is_video": False,
            "duration": 0,
        }

    def test_the_local_row_is_untouched_and_the_result_is_a_new_dict(self):
        track = {"video_id": "a", "artist": ""}
        extra = {"artist": "Art"}

        merged = _enrich(track, extra)

        assert merged is not track
        assert track["artist"] == ""


# ── All-tab composition: identity and cap interactions ───────────────


class TestComposeAllIdentity:
    def test_a_local_row_without_an_account_counterpart_is_the_same_object(self):
        local = [{"video_id": "a", "title": "A"}]

        rows = _compose_all(local, [])

        assert rows[0] is local[0]

    def test_an_enriched_row_is_a_new_dict_and_the_account_row_is_untouched(self):
        local = [{"video_id": "a", "artist": ""}]
        account = [{"video_id": "a", "artist": "Art", "album_id": "AL1"}]

        rows = _compose_all(local, account)

        assert rows[0] is not local[0]
        assert rows[0]["artist"] == "Art"
        assert account[0] == {"video_id": "a", "artist": "Art", "album_id": "AL1"}

    def test_account_only_rows_are_the_first_occurrence_objects(self):
        first, second = {"video_id": "a", "title": "first"}, {"video_id": "a", "title": "second"}
        only = {"video_id": "b"}

        rows = _compose_all([], [first, second, only])

        assert rows == [first, only]
        assert rows[0] is first
        assert rows[1] is only

    def test_the_local_cap_applies_before_dedup(self):
        # A duplicate inside the cap shrinks the block; the row beyond the
        # cap is never pulled in to fill the gap.
        local = [
            {"video_id": "a"},
            {"video_id": "a"},
            {"video_id": "b"},
            {"video_id": "c"},
        ]

        rows = _compose_all(local, [], cap=2)

        assert [row["video_id"] for row in rows] == ["a"]

    def test_each_source_is_capped_separately(self):
        local = [{"video_id": f"l{i}"} for i in range(4)]
        account = [{"video_id": f"a{i}"} for i in range(5)]

        rows = _compose_all(local, account, cap=3)

        assert [row["video_id"] for row in rows] == ["l0", "l1", "l2", "a0", "a1", "a2"]

    def test_cap_zero_composes_nothing(self):
        assert _compose_all([{"video_id": "a"}], [{"video_id": "b"}], cap=0) == []


# ── The empty-state copy pick ─────────────────────────────────────


class TestRecentEmptyMessage:
    """``_recent_empty_message``'s copy picks, on whichever backend."""

    def test_the_ytm_tab_distinguishes_its_three_empty_states(self):
        assert _recent_empty_message(_TAB_YTM, False, True, False) == (
            "Sign in to YT Music to see your account play history."
        )
        assert _recent_empty_message(_TAB_YTM, False, False, True) == (
            "Couldn't load YT Music history. Check the log for details."
        )
        assert _recent_empty_message(_TAB_YTM, False, False, False) == (
            "No YT Music play history found."
        )
        # The other tabs' failures never leak in.
        assert _recent_empty_message(_TAB_YTM, True, False, False) == (
            "No YT Music play history found."
        )

    def test_the_local_tab_picks_between_failure_and_empty_copies(self):
        assert _recent_empty_message(_TAB_LOCAL, True, False, False) == (
            "Couldn't load history. Check the log at ~/.config/ytm-player/logs/ytm.log for details."
        )
        assert _recent_empty_message(_TAB_LOCAL, False, True, True) == (
            "No play history yet. Start listening!"
        )

    def test_the_all_tab_names_every_missing_source(self):
        assert _recent_empty_message(_TAB_ALL, False, False, False) == (
            "No play history yet. Start listening!"
        )
        # A failed source appends the log pointer on top of its problem.
        assert _recent_empty_message(_TAB_ALL, True, False, False) == (
            "Couldn't load local history. "
            "Check the log at ~/.config/ytm-player/logs/ytm.log for details."
        )
        # An auth-missing source has not failed — no log pointer.
        assert _recent_empty_message(_TAB_ALL, False, True, False) == (
            "sign in to YT Music to include your account history."
        )

    def test_the_all_tab_joins_two_problems_with_the_log_pointer(self):
        assert _recent_empty_message(_TAB_ALL, True, True, False) == (
            "Couldn't load local history; sign in to YT Music to include "
            "your account history. "
            "Check the log at ~/.config/ytm-player/logs/ytm.log for details."
        )
        assert _recent_empty_message(_TAB_ALL, True, False, True) == (
            "Couldn't load local history; couldn't load YT Music history. "
            "Check the log at ~/.config/ytm-player/logs/ytm.log for details."
        )

    def test_the_all_tab_ladder_skips_the_ytm_failure_when_auth_is_missing(self):
        # The elif — an auth-missing user is told to sign in, not that
        # the history failed to load. The ytm-failure flag still opens
        # the log-pointer gate, though: that fetch genuinely failed.
        assert _recent_empty_message(_TAB_ALL, False, True, True) == (
            "sign in to YT Music to include your account history. "
            "Check the log at ~/.config/ytm-player/logs/ytm.log for details."
        )

    def test_a_signed_in_user_with_an_empty_account_history_has_no_problem(self):
        # Only a source that actually failed appends the log pointer.
        assert _recent_empty_message(_TAB_ALL, False, False, True) == (
            "couldn't load YT Music history. "
            "Check the log at ~/.config/ytm-player/logs/ytm.log for details."
        )

    def test_the_flag_gates_are_truthiness(self):
        assert _recent_empty_message(_TAB_LOCAL, 1, "x", 0) == _recent_empty_message(
            _TAB_LOCAL, True, True, False
        )
        assert _recent_empty_message(_TAB_LOCAL, 0, None, []) == _recent_empty_message(
            _TAB_LOCAL, False, False, False
        )


# ── The footer line ──────────────────────────────────────────────


class TestRecentFooterText:
    """``_recent_footer_text``'s lines, on whichever backend."""

    def test_the_local_and_ytm_tabs_state_their_own_counts(self):
        assert _recent_footer_text(_TAB_LOCAL, 12, True, True, True) == (
            "12 recently played tracks (local)"
        )
        assert _recent_footer_text(_TAB_YTM, 3, True, True, True) == (
            "3 recently played tracks (YT Music)"
        )

    def test_the_count_interpolates_through_str(self):
        assert _recent_footer_text(_TAB_LOCAL, "7", False, False, False) == (
            "7 recently played tracks (local)"
        )
        assert _recent_footer_text(_TAB_LOCAL, 2.5, False, False, False) == (
            "2.5 recently played tracks (local)"
        )
        assert _recent_footer_text(_TAB_LOCAL, 0, False, False, False) == (
            "0 recently played tracks (local)"
        )

    def test_a_clean_all_tab_names_the_two_group_composition(self):
        assert _recent_footer_text(_TAB_ALL, 5, False, False, False) == (
            "5 tracks — local history first, then additional account history"
        )

    def test_partial_results_are_said_out_loud_and_first(self):
        assert _recent_footer_text(_TAB_ALL, 4, True, False, False) == (
            "Local history couldn't be loaded; showing YT Music history only — 4 tracks"
        )
        assert _recent_footer_text(_TAB_ALL, 4, False, True, False) == (
            "Sign in to YT Music to include your account history; "
            "showing local history only — 4 tracks"
        )
        assert _recent_footer_text(_TAB_ALL, 4, False, False, True) == (
            "YT Music history couldn't be loaded; showing local history only — 4 tracks"
        )

    def test_the_flag_gates_are_truthiness(self):
        assert _recent_footer_text(_TAB_ALL, 1, 1, 0, 0) == _recent_footer_text(
            _TAB_ALL, 1, True, False, False
        )
        assert _recent_footer_text(_TAB_ALL, 1, None, "x", []) == _recent_footer_text(
            _TAB_ALL, 1, False, True, False
        )


# ── The nav-state remember-gates ─────────────────────────────────


class TestRecentNavState:
    """``_recent_nav_state``'s remember gates, on whichever backend."""

    def test_the_default_tab_is_not_remembered(self):
        assert _recent_nav_state(_DEFAULT_TAB, 4) == {"cursor_row": 4}

    def test_a_non_default_tab_rides_home_as_itself(self):
        assert _recent_nav_state(_TAB_YTM, None) == {"active_tab": _TAB_YTM}
        assert _recent_nav_state(_TAB_LOCAL, 0) == {"active_tab": _TAB_LOCAL}

    def test_a_cursor_on_the_first_row_is_not_remembered(self):
        assert _recent_nav_state(_DEFAULT_TAB, 0) == {}
        assert _recent_nav_state(_TAB_YTM, 0) == {"active_tab": _TAB_YTM}

    def test_none_and_negative_rows_are_not_remembered(self):
        assert _recent_nav_state(_DEFAULT_TAB, None) == {}
        assert _recent_nav_state(_DEFAULT_TAB, -1) == {}

    def test_a_positive_row_is_remembered_as_itself(self):
        assert _recent_nav_state(_DEFAULT_TAB, 2) == {"cursor_row": 2}
        assert _recent_nav_state(_TAB_LOCAL, 4) == {"active_tab": _TAB_LOCAL, "cursor_row": 4}

    def test_a_float_row_compares_like_python(self):
        assert _recent_nav_state(_DEFAULT_TAB, 2.5) == {"cursor_row": 2.5}
        assert _recent_nav_state(_DEFAULT_TAB, 0.5) == {"cursor_row": 0.5}
        assert _recent_nav_state(_DEFAULT_TAB, -0.5) == {}

    def test_a_bool_row_compares_like_python(self):
        assert _recent_nav_state(_DEFAULT_TAB, True) == {"cursor_row": True}
        assert _recent_nav_state(_DEFAULT_TAB, False) == {}


# ── Native / fallback parity ─────────────────────────────────────────
#
# When the Rust extension is built, the engine above IS the native one;
# each case below is also run in a child interpreter with the extension
# import blocked (pure-Python reference), and the two results must agree
# exactly.

# (track, occurrence) — occurrence None means "allocate a fresh id".
_STAMP_CASES = [
    ({"video_id": "a", "title": "A"}, _EXPLICIT_ID + 1),
    ({"_occurrence": 5, "video_id": "a"}, _EXPLICIT_ID + 2),
    ({"video_id": "b", "artists": [{"name": "X", "id": "1"}]}, None),
    ({"video_id": "c"}, "occ-9"),
    ({}, None),
]

_KEY_CASES = [
    {"_occurrence": _EXPLICIT_ID + 1, "video_id": "a"},
    {"videoId": "abc"},
    {"video_id": "xyz"},
    {"_occurrence": None, "video_id": "n"},
    {},
]

# (cache, track)
_ACCEPTED_CASES = [
    (
        [
            _cached("b", "B", _EXPLICIT_ID + 3),
            _cached("a", "latest", _EXPLICIT_ID + 1),
            _cached("a", "older", _EXPLICIT_ID + 2),
        ],
        {"video_id": "a", "title": "A again"},
    ),
    ([_cached("b", "B", _EXPLICIT_ID + 4)], {"video_id": "aaa", "title": "A"}),
    # The played track carries the stamp of the row it was played from;
    # the cache's own identity decides.
    (
        [_cached("a", "latest", _EXPLICIT_ID + 1), _cached("a", "older", _EXPLICIT_ID + 2)],
        {"video_id": "a", "title": "played from the older row", "_occurrence": _EXPLICIT_ID + 2},
    ),
    # A cache row without a stamp gets a fresh id on the move.
    ([{"video_id": "a", "title": "unstamped"}], {"video_id": "a", "title": "A"}),
    # "" == "": rows without an id supersede a track without one.
    ([{"title": "no id"}], {"title": "still no id"}),
    ([], {"video_id": "fresh", "title": "F"}),
    ([], {"title": "no id at all"}),
]

# (cache, track, cap)
_LOCAL_PLAY_CASES = [
    ([_cached("a"), _cached("b"), _cached("a")], {"video_id": "a", "title": "A"}, 10),
    ([_cached(f"v{i}") for i in range(5)], {"video_id": "new"}, 3),
    ([_cached("a")], {"video_id": "n"}, 0),
    ([], {"video_id": "n"}, 10),
    ([_cached("a")], {"video_id": "n"}, 10),
]

# (pending, seq, track, cap)
_PARK_CASES = [
    ([(2, {"video_id": "a", "title": "old"}), (1, {"video_id": "b"})], 3, {"video_id": "a"}, 10),
    ([(i, {"video_id": f"v{i}"}) for i in range(5)], 9, {"video_id": "new"}, 3),
    ([], 7, {"video_id": "a", "title": "A"}, 10),
    ([(1, {"video_id": "b"})], 2, {"video_id": "a"}, 0),
]

# (feed, fetch_seq, pending)
_MERGE_CASES = [
    ([_cached("a", "A")], 0, []),
    (
        [_cached("a", "latest", _EXPLICIT_ID + 1), _cached("a", "older", _EXPLICIT_ID + 2)],
        5,
        [(6, {"video_id": "a", "title": "A"})],
    ),
    ([_cached("a", "server order")], 5, [(4, {"video_id": "a", "title": "mine"})]),
    ([_cached("z")], 5, [(6, {"video_id": "new"}), (4, {"video_id": "pre"})]),
    ([_cached("z")], 0, [(2, {"video_id": "y"}), (1, {"video_id": "x"})]),
    # Supersede one row of a repeated track only.
    (
        [
            _cached("b", "B", _EXPLICIT_ID + 1),
            _cached("a", "latest", _EXPLICIT_ID + 2),
            _cached("a", "older", _EXPLICIT_ID + 3),
        ],
        0,
        [(1, {"video_id": "a", "title": "A"})],
    ),
]

# (track, extra)
_ENRICH_CASES = [
    (
        {"video_id": "a", "title": "Local", "artist": "", "album": None},
        {"title": "Server title", "artist": "Server artist", "album": "Server album"},
    ),
    ({"plays": 0, "liked": False}, {"plays": 9, "liked": True}),
    ({"video_id": "a"}, {"thumbnail_url": "", "artist": None}),
    ({"video_id": "a"}, {"artists": [{"name": "X"}], "is_video": False, "duration": 0}),
    ({}, {}),
]

# (local, account, cap)
_COMPOSE_CASES = [
    (
        [{"video_id": "loc0"}, {"video_id": "loc1"}],
        [{"video_id": "acc0"}, {"video_id": "loc1", "title": "Server"}, {"video_id": "acc1"}],
        _MAX_TRACKS,
    ),
    (
        [{"video_id": "loc0", "artist": "", "played_at": "t"}],
        [
            {
                "video_id": "loc0",
                "title": "Server",
                "artist": "Art",
                "album_id": "AL1",
                "thumbnail_url": "https://x",
            }
        ],
        _MAX_TRACKS,
    ),
    # A duplicate inside the cap shrinks the local block.
    (
        [{"video_id": "a"}, {"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}],
        [],
        2,
    ),
    # Account duplicates keep the first occurrence.
    (
        [],
        [
            {"video_id": "a", "title": "first"},
            {"video_id": "a", "title": "second"},
            {"video_id": "b"},
        ],
        3,
    ),
    # Each source capped separately, dedup before the account-only cap.
    (
        [{"video_id": f"l{i}"} for i in range(4)],
        [{"video_id": f"l{i}"} for i in range(4)] + [{"video_id": f"a{i}"} for i in range(5)],
        3,
    ),
    ([{"video_id": "a"}], [{"video_id": "b"}], 0),
    ([], [], _MAX_TRACKS),
    # The real cap boundary: 121 local rows (one duplicate inside the cap)
    # against a 190-id account feed.
    (
        [{"video_id": f"loc{i:04d}", "artist": ""} for i in range(120)]
        + [{"video_id": "loc0000", "title": "dupe"}],
        [{"video_id": f"loc{i:04d}", "artist": "Server"} for i in range(0, 120, 2)]
        + [{"video_id": f"acc{i:04d}"} for i in range(130)],
        _MAX_TRACKS,
    ),
]

# Empty-message cases: [active_tab, load_failed, ytm_auth_required,
# ytm_load_failed].
_EMPTY_CASES = [
    [0, False, False, False],
    [0, True, False, False],
    [0, False, True, False],
    [0, False, False, True],
    [0, True, True, False],
    [0, True, False, True],
    [0, False, True, True],
    [0, True, True, True],
    [1, True, True, True],
    [1, False, True, True],
    [2, False, True, False],
    [2, False, False, True],
    [2, True, False, False],
    [2, False, False, False],
    [1, 0, None, []],
    [2, 1, "x", 0],
    [0, 0, "", 1],
]

# Footer cases: [active_tab, count, load_failed, ytm_auth_required,
# ytm_load_failed].
_FOOTER_CASES = [
    [0, 12, False, False, False],
    [0, 4, True, False, False],
    [0, 4, False, True, False],
    [0, 4, False, False, True],
    [0, 0, True, False, False],
    [1, 12, True, True, True],
    [2, 3, True, True, True],
    [1, "7", False, False, False],
    [2, 2.5, False, False, False],
    [0, None, False, False, False],
    [0, 1, 1, 0, 0],
    [0, 1, None, "x", []],
]

# Nav-state cases: [active_tab, cursor_row].
_NAV_STATE_CASES = [
    [0, None],
    [0, 0],
    [0, 1],
    [0, -1],
    [0, 2.5],
    [0, 0.5],
    [0, -0.5],
    [0, True],
    [0, False],
    [1, None],
    [1, 0],
    [1, 4],
    [2, None],
    [2, 4],
    ["x", 3],
]

_CASES = {
    "stamp": _STAMP_CASES,
    "key": _KEY_CASES,
    "accepted": _ACCEPTED_CASES,
    "local_play": _LOCAL_PLAY_CASES,
    "park": _PARK_CASES,
    "merge": _MERGE_CASES,
    "enrich": _ENRICH_CASES,
    "compose": _COMPOSE_CASES,
    "empty": _EMPTY_CASES,
    "footer": _FOOTER_CASES,
    "nav_state": _NAV_STATE_CASES,
}

# The child interpreter runs this program with the case list on stdin and
# prints the reference engine's results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.recently_played import (
    _compose_all,
    _enrich,
    _occurrence_key,
    _recent_empty_message,
    _recent_footer_text,
    _recent_nav_state,
    _stamp,
    _with_accepted_play,
    merge_pending_account_plays,
    park_pending_play,
    with_accepted_local_play,
)

cases = json.load(sys.stdin)

out = {
    "stamp": [_stamp(t, occ) for t, occ in cases["stamp"]],
    "key": [_occurrence_key(t) for t in cases["key"]],
    "accepted": [_with_accepted_play(c, t) for c, t in cases["accepted"]],
    "local_play": [
        with_accepted_local_play(c, t, cap) for c, t, cap in cases["local_play"]
    ],
    "park": [park_pending_play(p, seq, t, cap) for p, seq, t, cap in cases["park"]],
    "merge": [merge_pending_account_plays(f, fs, p) for f, fs, p in cases["merge"]],
    "enrich": [_enrich(t, e) for t, e in cases["enrich"]],
    "compose": [_compose_all(loc, acc, cap) for loc, acc, cap in cases["compose"]],
    "empty": [
        _recent_empty_message(tab, lf, ar, fl)
        for tab, lf, ar, fl in cases["empty"]
    ],
    "footer": [
        _recent_footer_text(tab, count, lf, ar, fl)
        for tab, count, lf, ar, fl in cases["footer"]
    ],
    "nav_state": [
        _recent_nav_state(tab, row) for tab, row in cases["nav_state"]
    ],
}
json.dump(out, sys.stdout)
"""


def _native_results(cases: dict) -> dict:
    """Run whichever engine the module loaded (native when built) over *cases*."""
    return {
        "stamp": [_stamp(t, occ) for t, occ in cases["stamp"]],
        "key": [_occurrence_key(t) for t in cases["key"]],
        "accepted": [_with_accepted_play(c, t) for c, t in cases["accepted"]],
        "local_play": [with_accepted_local_play(c, t, cap) for c, t, cap in cases["local_play"]],
        "park": [park_pending_play(p, seq, t, cap) for p, seq, t, cap in cases["park"]],
        "merge": [merge_pending_account_plays(f, fs, p) for f, fs, p in cases["merge"]],
        "enrich": [_enrich(t, e) for t, e in cases["enrich"]],
        "compose": [_compose_all(loc, acc, cap) for loc, acc, cap in cases["compose"]],
        "empty": [_recent_empty_message(tab, lf, ar, fl) for tab, lf, ar, fl in cases["empty"]],
        "footer": [
            _recent_footer_text(tab, count, lf, ar, fl)
            for tab, count, lf, ar, fl in cases["footer"]
        ],
        "nav_state": [_recent_nav_state(tab, row) for tab, row in cases["nav_state"]],
    }


def _fallback_results(cases: dict) -> dict:
    """Run the pure-Python reference engine over *cases* in a child
    interpreter with the native extension import blocked."""
    payload = json.dumps(cases)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(out.stdout)


def _normalized(value: object) -> object:
    """A JSON-shaped copy of *value* with engine-fresh occurrence ids
    replaced by per-run placeholders.

    Fresh ids are the ``_occurrence`` ints below ``_EXPLICIT_ID`` — no
    counter reaches that in a test run, so every smaller id under that key
    was allocated by the engine. The native and fallback counters are
    independent (only one backend is ever active in a process), so their
    exact values must not be compared — but their distinctness and
    first-appearance order must, which the per-value placeholders pin.
    """
    fresh: dict[int, str] = {}

    def norm(v: object) -> object:
        if isinstance(v, dict):
            out = {}
            for key, item in v.items():
                if (
                    key == "_occurrence"
                    and isinstance(item, int)
                    and not isinstance(item, bool)
                    and item < _EXPLICIT_ID
                ):
                    item = fresh.setdefault(item, f"<fresh{len(fresh)}>")
                else:
                    item = norm(item)
                out[key] = item
            return out
        if isinstance(v, (list, tuple)):
            return [norm(item) for item in v]
        return v

    return norm(value)


class TestNativeParity:
    """When the Rust extension is built, the native history-merge engine
    must produce exactly the same caches as the pure-Python reference —
    same supersession, same occurrence identity, same enrichment, same
    caps — for every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _native_results(_CASES)
        fallback = _fallback_results(_CASES)
        for group, cases in _CASES.items():
            for i in range(len(cases)):
                assert _normalized(native[group][i]) == _normalized(fallback[group][i]), (
                    f"native and fallback {group} results differ for case {i}"
                )


# ── Wiring: the page consults the engines with its own state ────────


class TestEmptyMessageWiring:
    """``_empty_message`` consults the empty-state engine with the
    page's own tab and failure flags."""

    def test_the_engine_is_consulted_with_the_flags(self, monkeypatch):
        from ytm_player.ui.pages import recently_played as rp
        from ytm_player.ui.pages.recently_played import RecentlyPlayedPage

        calls: list[tuple[int, bool, bool, bool]] = []

        def recording(active_tab, load_failed, ytm_auth_required, ytm_load_failed):
            calls.append((active_tab, load_failed, ytm_auth_required, ytm_load_failed))
            return "the copy"

        monkeypatch.setattr(rp, "_recent_empty_message", recording)
        page = MagicMock()
        page._active_tab = _TAB_YTM
        page._load_failed = True
        page._ytm_auth_required = False
        page._ytm_load_failed = True

        assert RecentlyPlayedPage._empty_message(page) == "the copy"
        assert calls == [(_TAB_YTM, True, False, True)]


class TestFooterWiring:
    """``_footer_text`` consults the footer engine with the page's own
    tab, flags and the row count."""

    def test_the_engine_is_consulted_with_the_tab_flags_and_count(self, monkeypatch):
        from ytm_player.ui.pages import recently_played as rp
        from ytm_player.ui.pages.recently_played import RecentlyPlayedPage

        calls: list[tuple[int, int, bool, bool, bool]] = []

        def recording(active_tab, count, load_failed, ytm_auth_required, ytm_load_failed):
            calls.append((active_tab, count, load_failed, ytm_auth_required, ytm_load_failed))
            return "the footer"

        monkeypatch.setattr(rp, "_recent_footer_text", recording)
        page = MagicMock()
        page._active_tab = _TAB_ALL
        page._load_failed = False
        page._ytm_auth_required = True
        page._ytm_load_failed = False

        assert RecentlyPlayedPage._footer_text(page, 9) == "the footer"
        assert calls == [(_TAB_ALL, 9, False, True, False)]


class TestNavStateWiring:
    """``get_nav_state`` consults the nav engine with the table's own
    cursor row."""

    def test_the_engine_is_consulted_with_the_cursor_row(self, monkeypatch):
        from ytm_player.ui.pages import recently_played as rp
        from ytm_player.ui.pages.recently_played import RecentlyPlayedPage

        calls: list[tuple[int, int]] = []

        def recording(active_tab, cursor_row):
            calls.append((active_tab, cursor_row))
            return {"active_tab": active_tab, "cursor_row": cursor_row}

        monkeypatch.setattr(rp, "_recent_nav_state", recording)
        page = MagicMock()
        page._active_tab = _TAB_YTM
        page.query_one.return_value.cursor_row = 4

        assert RecentlyPlayedPage.get_nav_state(page) == {"active_tab": _TAB_YTM, "cursor_row": 4}
        assert calls == [(_TAB_YTM, 4)]

    def test_a_raising_capture_answers_no_cursor_row(self, monkeypatch):
        from ytm_player.ui.pages import recently_played as rp
        from ytm_player.ui.pages.recently_played import RecentlyPlayedPage

        calls: list[tuple[int, Any]] = []

        def recording(active_tab, cursor_row):
            calls.append((active_tab, cursor_row))
            return {"active_tab": active_tab}

        monkeypatch.setattr(rp, "_recent_nav_state", recording)
        page = MagicMock()
        page._active_tab = _TAB_LOCAL
        page.query_one.side_effect = RuntimeError("table not mounted")

        assert RecentlyPlayedPage.get_nav_state(page) == {"active_tab": _TAB_LOCAL}
        # A failed capture and a cursorless table both answer None.
        assert calls == [(_TAB_LOCAL, None)]
