"""Tests for the playlist-picker view-model engines.

The engines behind ``PlaylistPicker``: the recent-list move-to-front
(``_record_recent_ids``, the ``_record_recent`` core), the recency sort
(``_sorted_by_recent``, the ``_sort_by_recent`` core), the filtered
visible-row walk (``_visible_picker_rows``, the ``_populate_list`` row
core) and the playlist-count status label (``_count_label``). The
pure-Python references live in ``ui/popups/playlist_picker.py``; when
the Rust extension is built the same engines run natively
(``_ytm_player_native.playlist_record_recent`` & co).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.app import App
from textual.widgets import Label, Static

from ytm_player.ui.popups.playlist_picker import (
    PlaylistPicker,
    _count_label,
    _record_recent_ids,
    _sorted_by_recent,
    _visible_picker_rows,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── _record_recent_ids ──────────────────────────────────────────────


class TestRecordRecentIds:
    def test_absent_id_is_prepended(self):
        assert _record_recent_ids(["a", "b"], "c") == ["c", "a", "b"]

    def test_present_at_front_stays_once(self):
        assert _record_recent_ids(["a", "b"], "a") == ["a", "b"]

    def test_present_in_middle_moves_to_front(self):
        assert _record_recent_ids(["a", "b", "c"], "b") == ["b", "a", "c"]

    def test_present_at_end_moves_to_front(self):
        assert _record_recent_ids(["a", "b"], "b") == ["b", "a"]

    def test_duplicate_entries_drop_only_the_first(self):
        """``list.remove`` drops exactly the first equal occurrence, so the
        second copy survives the move."""
        assert _record_recent_ids(["a", "b", "a"], "a") == ["a", "b", "a"]

    def test_empty_recent_becomes_the_single_entry(self):
        assert _record_recent_ids([], "x") == ["x"]

    def test_junk_entries_pass_through_untouched(self):
        """A corrupted recent-playlists file can hold non-string entries;
        they never match the playlist id and keep their place."""
        assert _record_recent_ids([1, None, "x"], "x") == ["x", 1, None]

    def test_str_vs_int_junk_never_matches(self):
        assert _record_recent_ids([5], "5") == ["5", 5]


# ── _sorted_by_recent ───────────────────────────────────────────────


def _plists():
    return [
        {"playlistId": "x", "title": "X"},
        {"playlistId": "y", "title": "Y"},
        {"playlistId": "z", "title": "Z"},
    ]


class TestSortedByRecent:
    def test_no_recents_leaves_order_untouched(self):
        playlists = _plists()
        assert _sorted_by_recent(playlists, []) == playlists

    def test_recency_order_wins(self):
        out = _sorted_by_recent(_plists(), ["z", "x"])
        assert [p["playlistId"] for p in out] == ["z", "x", "y"]

    def test_unrecents_keep_fetch_order_after_recents(self):
        out = _sorted_by_recent(_plists(), ["q"])
        assert [p["playlistId"] for p in out] == ["x", "y", "z"]

    def test_equal_keys_are_stable(self):
        playlists = [
            {"playlistId": "a", "title": "first"},
            {"playlistId": "a", "title": "second"},
            {"playlistId": "b", "title": "third"},
        ]
        out = _sorted_by_recent(playlists, ["b"])
        assert [p["title"] for p in out] == ["third", "first", "second"]

    def test_duplicate_recent_ids_count_at_their_last_position(self):
        """The dict comprehension keeps the LAST index of a duplicated
        recent id, so with recents ``["a", "b", "a"]`` the ``a`` playlists
        rank at position 2 — after ``b`` at position 1 — even though ``a``
        came first in the list."""
        playlists = [
            {"playlistId": "a", "title": "a1"},
            {"playlistId": "b", "title": "b1"},
            {"playlistId": "a", "title": "a2"},
        ]
        out = _sorted_by_recent(playlists, ["a", "b", "a"])
        assert [p["title"] for p in out] == ["b1", "a1", "a2"]

    def test_id_field_falls_back_to_id_key(self):
        playlists = [{"id": "i1", "title": "via id"}, {"playlistId": "p", "title": "via pid"}]
        out = _sorted_by_recent(playlists, ["i1"])
        assert [p["title"] for p in out] == ["via id", "via pid"]

    def test_playlist_id_is_preferred_over_id(self):
        playlists = [{"playlistId": "p", "id": "i", "title": "both"}]
        out = _sorted_by_recent(playlists, ["p"])
        assert out == playlists
        out = _sorted_by_recent(playlists, ["i"])
        assert [p["title"] for p in out] == ["both"]  # unmatched — keeps its place

    def test_missing_pid_is_unrecenced(self):
        playlists = [{"title": "no id"}, {"playlistId": "p", "title": "has id"}]
        out = _sorted_by_recent(playlists, ["p"])
        assert [p["title"] for p in out] == ["has id", "no id"]

    def test_junk_recent_entries_never_match(self):
        playlists = _plists()
        out = _sorted_by_recent(playlists, [1, None])
        assert [p["playlistId"] for p in out] == ["x", "y", "z"]

    def test_input_list_is_not_mutated(self):
        playlists = _plists()
        snapshot = [dict(p) for p in playlists]
        _sorted_by_recent(playlists, ["z", "x"])
        assert playlists == snapshot

    def test_empty_playlists(self):
        assert _sorted_by_recent([], ["a"]) == []


# ── _visible_picker_rows ────────────────────────────────────────────


class TestVisiblePickerRows:
    def test_no_filter_shows_every_playlist(self):
        playlists = [
            {"playlistId": "p1", "title": "My List", "count": 12},
            {"playlistId": "p2", "title": "Other", "count": 0},
        ]
        assert _visible_picker_rows(playlists, "") == [
            ("p1", "My List", "12"),
            ("p2", "Other", 0),
        ]

    def test_filter_matches_title_case_insensitively(self):
        playlists = [{"playlistId": "p1", "title": "My List"}]
        assert _visible_picker_rows(playlists, "MY LIST") == [("p1", "My List", "")]

    def test_filter_is_containment_not_prefix(self):
        playlists = [{"playlistId": "p1", "title": "Best of Rock"}]
        assert _visible_picker_rows(playlists, "of r") == [("p1", "Best of Rock", "")]

    def test_non_matching_title_is_filtered_out(self):
        playlists = [
            {"playlistId": "p1", "title": "Rock"},
            {"playlistId": "p2", "title": "Jazz"},
        ]
        assert _visible_picker_rows(playlists, "jazz") == [("p2", "Jazz", "")]

    def test_query_is_stripped(self):
        playlists = [{"playlistId": "p1", "title": "Rock"}]
        assert _visible_picker_rows(playlists, "  rock  ") == [("p1", "Rock", "")]

    def test_whitespace_only_query_filters_nothing(self):
        playlists = [{"playlistId": "p1", "title": "Rock"}]
        assert _visible_picker_rows(playlists, "   ") == [("p1", "Rock", "")]

    def test_missing_title_defaults_to_untitled(self):
        playlists = [{"playlistId": "p1"}]
        assert _visible_picker_rows(playlists, "") == [("p1", "Untitled", "")]

    def test_untitled_default_is_filterable(self):
        playlists = [{"playlistId": "p1"}]
        assert _visible_picker_rows(playlists, "untitled") == [("p1", "Untitled", "")]

    def test_missing_count_defaults_to_empty_string(self):
        playlists = [{"playlistId": "p1", "title": "T"}]
        assert _visible_picker_rows(playlists, "") == [("p1", "T", "")]

    def test_falsy_count_passes_through_as_itself(self):
        """A falsy count keeps its value — 0 stays the int, None stays
        None — and only truthy counts are stringified."""
        playlists = [
            {"playlistId": "p1", "title": "T1", "count": 0},
            {"playlistId": "p2", "title": "T2", "count": None},
            {"playlistId": "p3", "title": "T3", "count": ""},
        ]
        rows = _visible_picker_rows(playlists, "")
        assert [r[2] for r in rows] == [0, None, ""]

    def test_truthy_count_is_stringified(self):
        playlists = [{"playlistId": "p1", "title": "T", "count": 42}]
        assert _visible_picker_rows(playlists, "") == [("p1", "T", "42")]

    def test_playlist_id_falls_back_to_id_key(self):
        playlists = [{"id": "i1", "title": "T"}]
        assert _visible_picker_rows(playlists, "") == [("i1", "T", "")]

    def test_missing_playlist_id_reads_as_empty_string(self):
        playlists = [{"title": "T"}]
        assert _visible_picker_rows(playlists, "") == [("", "T", "")]

    def test_non_str_title_survives_an_empty_query(self):
        """``query and query not in title.lower()`` short-circuits before
        ``.lower()`` runs, so a non-str title passes an empty query
        through on both backends."""
        playlists = [{"playlistId": "p1", "title": 123}]
        assert _visible_picker_rows(playlists, "") == [("p1", 123, "")]

    def test_empty_playlists(self):
        assert _visible_picker_rows([], "") == []
        assert _visible_picker_rows([], "rock") == []


# ── _count_label ────────────────────────────────────────────────────


class TestCountLabel:
    @pytest.mark.parametrize(
        "count, expected",
        [
            (0, "0 playlists"),
            (1, "1 playlist"),
            (2, "2 playlists"),
            (20, "20 playlists"),
            (101, "101 playlists"),
        ],
    )
    def test_pluralisation(self, count, expected):
        assert _count_label(count) == expected


# ── Native / fallback parity ─────────────────────────────────────────

_RECORD_CASES = [
    ([], "x"),
    (["a", "b"], "c"),
    (["a", "b"], "a"),
    (["a", "b", "c"], "b"),
    (["a", "b"], "b"),
    (["a", "b", "a"], "a"),
    ([1, None, "x"], "x"),
    ([5], "5"),
    (["only"], "only"),
]

_SORT_CASES = [
    ([], []),
    ([], ["a"]),
    ([{"playlistId": "x", "title": "X"}, {"playlistId": "y"}, {"playlistId": "z"}], []),
    ([{"playlistId": "x"}, {"playlistId": "y"}, {"playlistId": "z"}], ["z", "x"]),
    ([{"playlistId": "x"}, {"playlistId": "y"}], ["q"]),
    (
        [
            {"playlistId": "a", "title": "first"},
            {"playlistId": "a", "title": "second"},
            {"playlistId": "b", "title": "third"},
        ],
        ["b"],
    ),
    (
        [{"playlistId": "a"}, {"playlistId": "b"}, {"playlistId": "a"}],
        ["a", "b", "a"],
    ),
    ([{"id": "i1"}, {"playlistId": "p"}], ["i1"]),
    ([{"playlistId": "p", "id": "i"}], ["p"]),
    ([{"playlistId": "p", "id": "i"}], ["i"]),
    ([{"title": "no id"}, {"playlistId": "p"}], ["p"]),
    ([{"playlistId": "x"}, {"playlistId": "y"}], [1, None]),
]

_ROWS_CASES = [
    ([], ""),
    ([], "rock"),
    ([{"playlistId": "p1", "title": "My List", "count": 12}], ""),
    ([{"playlistId": "p1", "title": "My List", "count": 12}], "MY LIST"),
    ([{"playlistId": "p1", "title": "Best of Rock"}], "of r"),
    ([{"playlistId": "p1", "title": "Rock"}, {"playlistId": "p2", "title": "Jazz"}], "jazz"),
    ([{"playlistId": "p1", "title": "Rock"}], "  rock  "),
    ([{"playlistId": "p1", "title": "Rock"}], "   "),
    ([{"playlistId": "p1"}], ""),
    ([{"playlistId": "p1"}], "untitled"),
    ([{"playlistId": "p1", "title": "T", "count": 0}], ""),
    ([{"playlistId": "p2", "title": "T2", "count": None}], ""),
    ([{"playlistId": "p3", "title": "T3", "count": ""}], ""),
    ([{"playlistId": "p1", "title": "T", "count": 42}], ""),
    ([{"id": "i1", "title": "T"}], ""),
    ([{"title": "T"}], ""),
    ([{"playlistId": "p1", "title": 123}], ""),
    ([{"playlistId": "p1", "title": "Ünïcödé"}, {"playlistId": "p2", "title": "日本"}], "ünï"),
    ([{"playlistId": "p1", "title": "日本"}], "日本"),
]

_COUNT_CASES = [0, 1, 2, 20, 101]

# The child interpreter runs this program with the case lists on stdin
# and prints the reference engines' results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.playlist_picker import (
    _count_label,
    _record_recent_ids,
    _sorted_by_recent,
    _visible_picker_rows,
)

cases = json.load(sys.stdin)

out = {
    "record": [
        _record_recent_ids(recent, pid) for recent, pid in cases["record"]
    ],
    "sort": [
        _sorted_by_recent(playlists, recents)
        for playlists, recents in cases["sort"]
    ],
    "rows": [
        _visible_picker_rows(playlists, text) for playlists, text in cases["rows"]
    ],
    "labels": [_count_label(count) for count in cases["labels"]],
}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native picker engines must
    produce exactly the same recent lists, sort orders, visible rows and
    status labels as the pure-Python references for every case, so the
    two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        cases = {
            "record": _RECORD_CASES,
            "sort": _SORT_CASES,
            "rows": _ROWS_CASES,
            "labels": _COUNT_CASES,
        }
        native = json.loads(
            json.dumps(
                {
                    "record": [_record_recent_ids(recent, pid) for recent, pid in _RECORD_CASES],
                    "sort": [
                        _sorted_by_recent(playlists, recents) for playlists, recents in _SORT_CASES
                    ],
                    "rows": [
                        _visible_picker_rows(playlists, text) for playlists, text in _ROWS_CASES
                    ],
                    "labels": [_count_label(count) for count in _COUNT_CASES],
                }
            )
        )
        fallback = json.loads(
            subprocess.run(
                [sys.executable, "-c", _CHILD_PROG],
                input=json.dumps(cases),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        for group, case_list in cases.items():
            for i, case in enumerate(case_list):
                assert native[group][i] == fallback[group][i], (
                    f"native and fallback {group} results differ for case {i}: {case}"
                )


# ── Mounted widget integration ───────────────────────────────────────


class _PickerHost(App):
    """Minimal host app pushing a picker through the real Textual loop."""

    def __init__(self) -> None:
        super().__init__()
        self.ytmusic = MagicMock()
        self.ytmusic.get_library_playlists = AsyncMock(
            return_value=[
                {"playlistId": "ALPHA", "title": "Alpha", "count": 0},
                {"playlistId": "BETA", "title": "Beta", "count": 3},
            ]
        )

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables


class TestPickerWidget:
    """The PlaylistPicker glue: the engines' output reaching the ListView
    and the status line — the recency sort applied on the worker's
    SUCCESS, and the pluralised count label shown once the list lands."""

    async def test_recency_sort_and_count_label_reach_the_list(self):
        picker = PlaylistPicker(video_ids=["v1"])
        async with _PickerHost().run_test() as pilot:
            with (
                patch(
                    "ytm_player.ui.popups.playlist_picker._load_recent_ids",
                    return_value=["BETA"],
                ),
                patch("ytm_player.ui.popups.playlist_picker._record_recent"),
            ):
                pilot.app.push_screen(picker)
                await pilot.pause()
                await pilot.pause()

            status = picker.query_one("#picker-status", Static)
            assert str(status.render()) == "2 playlists"

            items = picker.query("#playlist-list ListItem")
            assert len(items) == 3  # Create New + the two playlists
            # BETA is the recent one, so it leads the list.
            labels = [str(i.query_one(Label).render()) for i in items[1:]]
            assert labels == ["Beta  (3)", "Alpha"]
