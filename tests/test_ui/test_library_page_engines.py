"""Tests for the library page's engines.

The pure decision cores behind ``ui/pages/library.py`` — the largest
UI module that had no native engines of its own before this port.
Five engines: the load view model (``_playlist_view`` — the
title/owner/privacy/year reads, the normalized track list, the
subtitle with its loading suffix, the unavailable count and the
tail-fetch gate), the Shuffle-lock header button's label/classes pair
(``_shuffle_lock_button``), the lock-toggle plan
(``_shuffle_lock_toggle_plan`` — the new lock state and the
force-shuffle gate), the playing-track id resolution
(``_playing_video_id`` — the player-then-queue fallback chain) and the
cursor-restore row (``_cursor_restore_row`` — the saved nav row in
bounds, else the first currently-playing row). The engines run with
two interchangeable backends: the pure-Python transcriptions in the
module and, when the Rust extension is built, the native
implementations bound at the module's tail.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case — error shapes included — so the two
implementations cannot drift apart. The wiring tests pin the page's
routing: the load consults the view engine with the fetched payload
and mounts its answers, the toggle consults the plan engine with the
host's own values and applies its effects, and the cursor restore
consults the row engine with the table's own state. The worker-suite
in ``test_library_workers.py`` and the lock flows in
``test_library_shuffle_lock.py`` keep exercising the page end to end
on both backends.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

from ytm_player.services.queue import QueueManager
from ytm_player.ui.pages import library as lib
from ytm_player.ui.pages.library import (
    LibraryPage,
    _cursor_restore_row,
    _playing_video_id,
    _playlist_view,
    _shuffle_lock_button,
    _shuffle_lock_toggle_plan,
)
from ytm_player.ui.widgets.track_table import TrackTable

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _raw_track(video_id: str = "v", **extra: object) -> dict:
    track = {"videoId": video_id, "title": "T"}
    track.update(extra)
    return track


# ── The load view model ──────────────────────────────────────────────


class TestPlaylistView:
    """``_playlist_view``'s reads and gates, on whichever backend."""

    def test_the_full_walk(self):
        view = _playlist_view(
            {
                "title": "Mixtape",
                "author": {"name": "me"},
                "privacy": "  PRIVATE  ",
                "year": 2020,
                "tracks": [_raw_track("v1"), _raw_track("v2")],
                "trackCount": 5,
            }
        )
        assert view["title"] == "Mixtape"
        assert view["owner"] == "me"
        assert view["privacy"] == "PRIVATE"
        assert view["year"] == 2020
        assert [t["video_id"] for t in view["tracks"]] == ["v1", "v2"]
        assert view["track_count"] == 2
        assert view["total_count"] == 5
        assert view["subtitle"] == "me · Private · 2020 · 2 tracks (loading 5 total…)"
        assert view["unavailable"] == 0
        assert view["fetch_remaining"] is True
        assert view["raw_count"] == 2

    def test_a_missing_title_answers_the_default(self):
        assert _playlist_view({})["title"] == "Unknown Playlist"

    def test_a_missing_author_answers_the_default(self):
        assert _playlist_view({})["owner"] == "Unknown"

    def test_a_non_dict_author_stringifies(self):
        assert _playlist_view({"author": "someone"})["owner"] == "someone"

    def test_a_missing_privacy_answers_empty(self):
        assert _playlist_view({})["privacy"] == ""

    def test_a_falsy_privacy_answers_empty(self):
        assert _playlist_view({"privacy": None})["privacy"] == ""
        assert _playlist_view({"privacy": "  "})["privacy"] == ""

    def test_the_privacy_is_stripped(self):
        assert _playlist_view({"privacy": "PUBLIC "})["privacy"] == "PUBLIC"

    def test_the_year_rides_home_raw(self):
        assert _playlist_view({"year": "2021"})["year"] == "2021"
        assert _playlist_view({})["year"] is None

    def test_the_subtitle_omits_empty_privacy_and_year(self):
        view = _playlist_view({"author": {"name": "me"}, "tracks": [_raw_track()]})
        assert view["subtitle"] == "me · 1 track"

    def test_a_falsy_track_count_falls_back_to_the_track_count(self):
        view = _playlist_view({"tracks": [_raw_track()], "trackCount": 0})
        assert view["total_count"] == 1
        assert view["fetch_remaining"] is False
        assert "loading" not in view["subtitle"]

    def test_the_suffix_only_fires_above_the_track_count(self):
        view = _playlist_view({"tracks": [_raw_track()], "trackCount": 1})
        assert "loading" not in view["subtitle"]
        assert view["fetch_remaining"] is False

    def test_the_fetch_gate_reads_the_raw_count(self):
        # trackCount above the normalized count but not above the raw
        # count (unavailable entries): no tail to fetch.
        view = _playlist_view({"tracks": [_raw_track("v1"), {"title": "no id"}], "trackCount": 2})
        assert view["track_count"] == 1
        assert view["unavailable"] == 1
        assert view["fetch_remaining"] is False
        assert view["raw_count"] == 2

    def test_unavailable_tracks_are_counted(self):
        view = _playlist_view({"tracks": [_raw_track("v1"), {"title": "no id"}, {"title": "gone"}]})
        assert view["unavailable"] == 2

    def test_a_truthy_non_str_privacy_raises_the_strip_error(self):
        with pytest.raises(AttributeError):
            _playlist_view({"privacy": 5})


# ── The Shuffle-lock button pair ─────────────────────────────────────


class TestShuffleLockButton:
    """``_shuffle_lock_button``'s pair, on whichever backend."""

    def test_locked_answers_the_on_pair(self):
        assert _shuffle_lock_button(True) == {"label": "Shuffle lock: ON", "classes": "locked"}

    def test_unlocked_answers_the_off_pair(self):
        assert _shuffle_lock_button(False) == {"label": "Shuffle lock: off", "classes": ""}

    def test_the_gate_is_truthiness(self):
        assert _shuffle_lock_button(1)["label"] == "Shuffle lock: ON"
        assert _shuffle_lock_button("x")["classes"] == "locked"
        assert _shuffle_lock_button(0)["label"] == "Shuffle lock: off"
        assert _shuffle_lock_button(None)["classes"] == ""
        assert _shuffle_lock_button([])["label"] == "Shuffle lock: off"


# ── The lock-toggle plan ─────────────────────────────────────────────


class TestShuffleLockTogglePlan:
    """``_shuffle_lock_toggle_plan``'s flip and gate, on whichever backend."""

    def test_an_unlocked_pref_flips_to_locked(self):
        plan = _shuffle_lock_toggle_plan(False, "PL2", "PL1")
        assert plan == {"new_locked": True, "force_shuffle": False}

    def test_a_locked_pref_flips_to_unlocked(self):
        plan = _shuffle_lock_toggle_plan(True, "PL1", "PL1")
        assert plan == {"new_locked": False, "force_shuffle": False}

    def test_locking_the_playing_playlist_forces_shuffle(self):
        plan = _shuffle_lock_toggle_plan(False, "PL1", "PL1")
        assert plan == {"new_locked": True, "force_shuffle": True}

    def test_the_flip_reads_truthiness(self):
        assert _shuffle_lock_toggle_plan(0, "PL1", "PL1")["new_locked"] is True
        assert _shuffle_lock_toggle_plan("on", "PL1", "PL1")["new_locked"] is False

    def test_the_force_gate_needs_an_exact_context_match(self):
        assert _shuffle_lock_toggle_plan(False, "PL1 ", "PL1")["force_shuffle"] is False
        assert _shuffle_lock_toggle_plan(False, None, "PL1")["force_shuffle"] is False


# ── The playing-track id resolution ──────────────────────────────────


class TestPlayingVideoId:
    """``_playing_video_id``'s fallback chain, on whichever backend."""

    def test_a_dict_player_track_contributes_its_video_id(self):
        assert _playing_video_id({"video_id": "v1"}, None) == "v1"

    def test_a_player_track_without_an_id_falls_back_to_the_queue(self):
        assert _playing_video_id({"title": "T"}, {"video_id": "q"}) == "q"

    def test_a_truthy_non_dict_player_track_rides_home_as_itself(self):
        assert _playing_video_id("v1", {"video_id": "q"}) == "v1"

    def test_a_falsy_player_track_falls_back_to_the_queue(self):
        assert _playing_video_id(None, {"video_id": "q"}) == "q"

    def test_a_falsy_player_dict_falls_back_to_the_queue(self):
        assert _playing_video_id({}, {"video_id": "q"}) == "q"

    def test_a_queue_track_without_an_id_answers_falsy(self):
        assert _playing_video_id(None, {"title": "T"}) is None

    def test_nothing_playing_answers_none(self):
        assert _playing_video_id(None, None) is None

    def test_a_falsy_queue_track_is_not_consulted(self):
        assert _playing_video_id(None, None) is None

    def test_a_truthy_non_dict_queue_track_raises_the_get_error(self):
        with pytest.raises(AttributeError):
            _playing_video_id(None, "plain")


# ── The cursor-restore row ───────────────────────────────────────────


class TestCursorRestoreRow:
    """``_cursor_restore_row``'s row math, on whichever backend."""

    def test_a_zero_row_count_answers_none(self):
        assert _cursor_restore_row(0, 0, "v", []) is None

    def test_a_saved_row_in_bounds_wins(self):
        assert _cursor_restore_row(1, 3, "v2", [{"video_id": "v2"}]) == 1

    def test_the_bounds_are_half_open(self):
        assert _cursor_restore_row(0, 3, None, []) == 0
        assert _cursor_restore_row(2, 3, None, []) == 2
        assert _cursor_restore_row(3, 3, None, []) is None
        assert _cursor_restore_row(-1, 3, None, []) is None

    def test_none_is_not_a_row(self):
        assert _cursor_restore_row(None, 3, None, []) is None

    def test_the_playing_id_walk_picks_the_first_match(self):
        tracks = [{"video_id": "v1"}, {"video_id": "v2"}, {"video_id": "v2"}]
        assert _cursor_restore_row(None, 3, "v2", tracks) == 1

    def test_a_playing_id_no_track_matches_answers_none(self):
        assert _cursor_restore_row(None, 3, "zz", [{"video_id": "v1"}]) is None

    def test_a_falsy_playing_id_skips_the_walk(self):
        assert _cursor_restore_row(None, 3, "", [{"video_id": ""}]) is None

    def test_a_saved_row_out_of_bounds_falls_back_to_the_walk(self):
        assert _cursor_restore_row(9, 2, "v2", [{"video_id": "v1"}, {"video_id": "v2"}]) == 1

    def test_a_float_saved_row_compares_like_python(self):
        assert _cursor_restore_row(1.5, 3, None, []) == 1.5
        assert _cursor_restore_row(3.0, 3, None, []) is None

    def test_a_raising_row_read_propagates(self):
        class Boom:
            def get(self, *_args: object) -> object:
                raise RuntimeError("row read failed")

        with pytest.raises(RuntimeError, match="row read failed"):
            _cursor_restore_row(None, 3, "v", [Boom()])


# ── Native-only total shapes ─────────────────────────────────────────


class TestTotalityEngine:
    """Where the pure-Python reference would raise, the native engine
    stays total (matching the ported view engines before it): a
    non-list ``tracks`` value reads as empty with an unknown count, and
    an uncomparable ``trackCount`` skips the gated decisions."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_list_tracks_read_empty(self):
        view = _playlist_view(
            {"title": "P", "author": {"name": "me"}, "trackCount": 5, "tracks": "nope"}
        )
        assert view["tracks"] == []
        assert view["unavailable"] is None
        assert view["fetch_remaining"] is False
        assert view["raw_count"] == 0

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_uncomparable_track_count_skips_the_gates(self):
        # A string trackCount would make the reference's `>` raise
        # TypeError; the engine skips the suffix and the tail fetch.
        view = _playlist_view(
            {
                "title": "P",
                "author": {"name": "me"},
                "trackCount": "500",
                "tracks": [_raw_track("v1")],
            }
        )
        assert view["subtitle"] == "me · 1 track"
        assert view["fetch_remaining"] is False


# ── Wiring ───────────────────────────────────────────────────────────


class _LoadHost(App[None]):
    """A minimal host whose ytmusic serves one canned playlist."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self._ytmusic = SimpleNamespace(
            get_playlist=self._get_playlist, get_playlist_remaining=self._get_remaining
        )
        self.payload = payload
        self.shuffle_prefs = SimpleNamespace(get=lambda pid: False, set=lambda *a: None)
        self.queue = SimpleNamespace(current_track=None)
        self.player = SimpleNamespace(current_track=None)
        self.radio_started = False

    async def _get_playlist(self, playlist_id, limit=None, order=None):
        return self.payload

    async def _get_remaining(self, playlist_id, already_have, order=None):
        return []

    @property
    def ytmusic(self):
        return self._ytmusic

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield LibraryPage(playlist_id="PL1", id="library")

    async def _start_playlist_radio(self, data) -> None:
        self.radio_started = True


def _label_texts(page: LibraryPage) -> list[str]:
    from textual.widgets import Label, Static

    texts = [str(w.render()) for w in page.query(Label)]
    texts += [str(w.render()) for w in page.query(Static)]
    return texts


_PAYLOAD = {
    "id": "PL1",
    "title": "Mixtape",
    "author": {"name": "me"},
    "privacy": "PRIVATE",
    "year": 2020,
    "description": "A mix",
    "trackCount": 2,
    "tracks": [_raw_track("v1"), _raw_track("v2")],
}


class TestLoadWiring:
    """``load_playlist`` consults the view engine and mounts its answers."""

    async def test_the_view_engine_is_consulted_with_the_fetched_payload(self, monkeypatch):
        calls: list[dict] = []
        real = lib._playlist_view

        def recording(data):
            calls.append(data)
            return real(data)

        monkeypatch.setattr(lib, "_playlist_view", recording)
        app = _LoadHost(_PAYLOAD)

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()

        assert calls == [_PAYLOAD]

    async def test_the_headers_mount_the_engines_answers(self):
        app = _LoadHost(_PAYLOAD)

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LibraryPage)

            texts = _label_texts(page)
            assert any("Mixtape" in text for text in texts)
            assert any("A mix" in text for text in texts)
            assert any("me · Private · 2020 · 2 tracks" in text for text in texts)
            assert not any("unavailable" in text for text in texts)

            table = page.query_one("#library-tracks", TrackTable)
            assert table.row_count == 2

    async def test_the_loading_suffix_and_unavailable_count_ride_the_engine(self):
        payload = dict(_PAYLOAD)
        payload["trackCount"] = 5
        payload["tracks"] = [_raw_track("v1"), {"title": "no id"}]
        app = _LoadHost(payload)

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LibraryPage)

            texts = _label_texts(page)
            assert any("loading 5 total…" in text for text in texts)
            assert any("1 unavailable tracks hidden" in text for text in texts)
            # trackCount > len(raw): the tail fetch was kicked off.
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)

    async def test_the_lock_button_mounts_the_engines_pair(self):
        app = _LoadHost(_PAYLOAD)
        app.shuffle_prefs = SimpleNamespace(get=lambda pid: True, set=lambda *a: None)

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LibraryPage)

            from textual.widgets import Static

            btn = page.query_one("#shuffle-lock-btn", Static)
            assert str(btn.render()) == "Shuffle lock: ON"
            assert "locked" in btn.classes


class TestToggleWiring:
    """``_toggle_shuffle_lock`` consults the plan engine and applies its
    effects."""

    def _page(self, queue_context: str) -> MagicMock:
        page = MagicMock()
        page._active_playlist_id = "PL1"
        host = page.app
        host.queue = QueueManager()
        host.queue.add_multiple(
            [
                {"video_id": f"v{i}", "title": f"T{i}", "artist": "A", "duration": 60}
                for i in range(3)
            ]
        )
        host.queue.jump_to(0)
        host.queue.set_context(queue_context)
        host.shuffle_prefs.get = MagicMock(return_value=False)
        return page

    def test_the_plan_engine_is_consulted_with_the_hosts_own_values(self, monkeypatch):
        calls: list[tuple] = []

        def recording_engine(prefs_value, current_context_id, playlist_id):
            calls.append((prefs_value, current_context_id, playlist_id))
            return {"new_locked": True, "force_shuffle": False}

        monkeypatch.setattr(lib, "_shuffle_lock_toggle_plan", recording_engine)
        page = self._page("PL1")

        LibraryPage._toggle_shuffle_lock(page)

        host = page.app
        assert calls == [(False, "PL1", "PL1")]
        host.shuffle_prefs.set.assert_called_once_with("PL1", True)

    def test_the_button_is_updated_with_the_engines_pair(self, monkeypatch):
        monkeypatch.setattr(
            lib,
            "_shuffle_lock_toggle_plan",
            lambda *a: {"new_locked": True, "force_shuffle": False},
        )
        page = self._page("PL2")
        LibraryPage._toggle_shuffle_lock(page)

        btn = page.query_one.return_value
        btn.update.assert_called_once_with("Shuffle lock: ON")
        btn.add_class.assert_called_once_with("locked")

    def test_the_button_pair_unlocks(self, monkeypatch):
        monkeypatch.setattr(
            lib,
            "_shuffle_lock_toggle_plan",
            lambda *a: {"new_locked": False, "force_shuffle": False},
        )
        page = self._page("PL1")
        page.app.shuffle_prefs.get = MagicMock(return_value=True)
        LibraryPage._toggle_shuffle_lock(page)

        btn = page.query_one.return_value
        btn.update.assert_called_once_with("Shuffle lock: off")
        btn.remove_class.assert_called_once_with("locked")

    def test_the_force_shuffle_gate_drives_the_shuffle_toggle(self, monkeypatch):
        monkeypatch.setattr(
            lib, "_shuffle_lock_toggle_plan", lambda *a: {"new_locked": True, "force_shuffle": True}
        )
        page = self._page("PL2")

        LibraryPage._toggle_shuffle_lock(page)

        host = page.app
        assert host.queue.shuffle_enabled  # the plan's gate forced it on
        host._refresh_queue_page.assert_called_once()

    def test_no_forced_shuffle_leaves_the_queue_alone(self, monkeypatch):
        monkeypatch.setattr(
            lib,
            "_shuffle_lock_toggle_plan",
            lambda *a: {"new_locked": True, "force_shuffle": False},
        )
        page = self._page("PL2")

        LibraryPage._toggle_shuffle_lock(page)

        host = page.app
        assert not host.queue.shuffle_enabled
        host._refresh_queue_page.assert_not_called()


class TestCursorRestoreWiring:
    """``_restore_track_cursor`` consults the engines with the table's
    own state and moves the cursor to the engine's row."""

    def _page(self, player_track: object, queue_track: object) -> MagicMock:
        page = MagicMock()
        page.app.player = SimpleNamespace(_current_track=player_track)
        page.app.queue = SimpleNamespace(current_track=queue_track)
        return page

    def test_the_engines_receive_the_tables_own_state(self, monkeypatch):
        playing_calls: list[tuple] = []
        row_calls: list[tuple] = []

        def recording_playing(player_track, queue_track):
            playing_calls.append((player_track, queue_track))
            return "v2"

        def recording_row(row, row_count, playing_id, tracks):
            row_calls.append((row, row_count, playing_id, tracks))
            return 1

        monkeypatch.setattr(lib, "_playing_video_id", recording_playing)
        monkeypatch.setattr(lib, "_cursor_restore_row", recording_row)
        page = self._page(None, {"video_id": "v2"})
        page._restore_cursor_row = None
        table = MagicMock()
        table.row_count = 2
        table._tracks = [{"video_id": "v1"}, {"video_id": "v2"}]

        LibraryPage._restore_track_cursor(page, table)

        assert playing_calls == [(None, {"video_id": "v2"})]
        assert row_calls == [(None, 2, "v2", table._tracks)]
        table.move_cursor.assert_called_once_with(row=1)

    def test_the_saved_row_reaches_the_engine_once(self, monkeypatch):
        row_calls: list[tuple] = []

        def recording_row(row, row_count, playing_id, tracks):
            row_calls.append(row)
            return row

        monkeypatch.setattr(lib, "_playing_video_id", lambda *a: None)
        monkeypatch.setattr(lib, "_cursor_restore_row", recording_row)
        page = self._page(None, None)
        page._restore_cursor_row = 4
        table = MagicMock()
        table.row_count = 9
        table._tracks = []

        LibraryPage._restore_track_cursor(page, table)

        assert row_calls == [4]
        assert page._restore_cursor_row is None  # consumed exactly once
        table.move_cursor.assert_called_once_with(row=4)

    def test_an_empty_table_consults_nothing_and_keeps_the_saved_row(self, monkeypatch):
        def _boom(*_args: object) -> object:
            raise AssertionError("no engine may run for an empty table")

        monkeypatch.setattr(lib, "_playing_video_id", _boom)
        monkeypatch.setattr(lib, "_cursor_restore_row", _boom)
        page = self._page(None, None)
        saved = object()
        page._restore_cursor_row = saved
        table = MagicMock()
        table.row_count = 0

        LibraryPage._restore_track_cursor(page, table)

        table.move_cursor.assert_not_called()
        assert page._restore_cursor_row is saved  # the reference keeps it

    def test_no_engine_row_leaves_the_cursor_alone(self, monkeypatch):
        monkeypatch.setattr(lib, "_playing_video_id", lambda *a: None)
        monkeypatch.setattr(lib, "_cursor_restore_row", lambda *a: None)
        page = self._page(None, None)
        page._restore_cursor_row = None
        table = MagicMock()
        table.row_count = 3
        table._tracks = [{"video_id": "v1"}]

        LibraryPage._restore_track_cursor(page, table)

        table.move_cursor.assert_not_called()


# ── Native vs fallback parity ────────────────────────────────────────

# View cases: the playlist payloads. Shapes where the pure-Python
# reference raises and the native engine stays total (non-list tracks,
# uncomparable trackCount) stay out of the parity cases and are pinned
# by TestTotalityEngine above; a non-str owner name raises TypeError
# through the shared subtitle builder on both sides but with different
# messages, so it stays out too.

_VIEW_PARITY_CASES: list[dict[str, Any]] = [
    {},
    {"title": "Mixtape", "author": {"name": "me"}, "tracks": []},
    {
        "title": "Big",
        "author": {"name": "me"},
        "privacy": "PRIVATE",
        "year": 2020,
        "tracks": [_raw_track("v1"), _raw_track("v2")],
        "trackCount": 5,
    },
    {"title": "P", "author": {"name": "me"}, "privacy": " public ", "tracks": [_raw_track()]},
    {"title": "P", "author": {"name": "me"}, "year": "2021", "tracks": [_raw_track()]},
    {"title": "P", "author": "someone", "tracks": []},
    {"title": "P", "author": {}, "tracks": []},
    {"title": "P", "author": {"name": ""}, "tracks": []},
    {"title": "P", "author": None, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": 0, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": None, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": 1, "tracks": [_raw_track()]},
    {"title": "P", "author": {"name": "me"}, "trackCount": 8.5, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": True, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": 10**30, "tracks": []},
    {
        "title": "P",
        "author": {"name": "me"},
        "trackCount": 2,
        "tracks": [_raw_track(), {"title": "no id"}],
    },
    {"title": "P", "author": {"name": "me"}, "privacy": 5, "tracks": []},
    {"title": 42, "author": {"name": "me"}, "tracks": []},
]

# Button cases: the locked values.
_BUTTON_PARITY_CASES = [True, False, 0, 1, "", "x", None, []]

# Toggle-plan cases: [prefs_value, current_context_id, playlist_id].
_TOGGLE_PARITY_CASES = [
    [False, "PL1", "PL1"],
    [False, "PL2", "PL1"],
    [True, "PL1", "PL1"],
    [0, "X", "X"],
    ["", None, None],
    ["on", "PL1", "PL2"],
]

# Playing-id cases: [player_track, queue_track].
_PLAYING_PARITY_CASES = [
    [None, None],
    [{"video_id": "v1"}, None],
    [{"video_id": None}, {"video_id": "q"}],
    ["v1", {"video_id": "q"}],
    [{"video_id": 5}, None],
    [None, {"video_id": "q"}],
    [{}, {"video_id": "q"}],
    [None, {"title": "T"}],
    [None, "plain"],
]

# Cursor-row cases: [row, row_count, playing_id, tracks].
_CURSOR_PARITY_CASES = [
    [0, 0, "v", []],
    [1, 3, "v2", [{"video_id": "v2"}]],
    [0, 3, None, []],
    [2, 3, None, []],
    [3, 3, None, []],
    [-1, 3, None, []],
    [None, 3, None, []],
    [None, 3, "v2", [{"video_id": "v1"}, {"video_id": "v2"}, {"video_id": "v2"}]],
    [None, 3, "zz", [{"video_id": "v1"}]],
    [None, 3, "", [{"video_id": ""}]],
    [9, 2, "v2", [{"video_id": "v1"}, {"video_id": "v2"}]],
    [1.5, 3, None, []],
    [3.0, 3, None, []],
    [True, 3, None, []],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.library import (
    _cursor_restore_row,
    _playing_video_id,
    _playlist_view,
    _shuffle_lock_button,
    _shuffle_lock_toggle_plan,
)


def outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


payload = json.load(sys.stdin)
json.dump(
    {
        "view": [outcome(_playlist_view, case) for case in payload["view"]],
        "button": [outcome(_shuffle_lock_button, case) for case in payload["button"]],
        "toggle": [
            outcome(_shuffle_lock_toggle_plan, *case) for case in payload["toggle"]
        ],
        "playing": [
            outcome(_playing_video_id, *case) for case in payload["playing"]
        ],
        "cursor": [
            outcome(_cursor_restore_row, *case) for case in payload["cursor"]
        ],
    },
    sys.stdout,
    ensure_ascii=False,
)
"""


def _native_outcomes():
    """The parent-side outcomes — every engine over every case, run by
    whichever backend the module loaded (the native one, when the parity
    tests run)."""

    def outcome(fn, *args):
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"ok": result}

    return {
        "view": [outcome(_playlist_view, case) for case in _VIEW_PARITY_CASES],
        "button": [outcome(_shuffle_lock_button, case) for case in _BUTTON_PARITY_CASES],
        "toggle": [outcome(_shuffle_lock_toggle_plan, *case) for case in _TOGGLE_PARITY_CASES],
        "playing": [outcome(_playing_video_id, *case) for case in _PLAYING_PARITY_CASES],
        "cursor": [outcome(_cursor_restore_row, *case) for case in _CURSOR_PARITY_CASES],
    }


def _run_child():
    """The pure-Python reference outcomes, run in a child interpreter
    with the extension import blocked."""
    payload = {
        "view": _VIEW_PARITY_CASES,
        "button": _BUTTON_PARITY_CASES,
        "toggle": _TOGGLE_PARITY_CASES,
        "playing": _PLAYING_PARITY_CASES,
        "cursor": _CURSOR_PARITY_CASES,
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same views, same pairs,
    same plans, same rows, same errors with the same messages — so the
    two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_every_engine_matches_the_fallback(self):
        fallback = _run_child()
        native = json.loads(json.dumps(_native_outcomes()))
        for family, cases in [
            ("view", _VIEW_PARITY_CASES),
            ("button", _BUTTON_PARITY_CASES),
            ("toggle", _TOGGLE_PARITY_CASES),
            ("playing", _PLAYING_PARITY_CASES),
            ("cursor", _CURSOR_PARITY_CASES),
        ]:
            for case, n, f in zip(cases, native[family], fallback[family]):
                assert n == f, (
                    f"native and fallback {family} engine differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
