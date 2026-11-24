"""Tests for the liked-songs page's engines.

The pure decision cores behind ``ui/pages/liked_songs.py`` — the last
page that had no native engines of its own before this port. Five
engines: the empty-state copy (``_empty_state`` — the load-failed flag
picking the failure message that points at the log over the genuine
empty-state copy), the footer text (``_footer_text`` — the count
through ``str``, the loading-more suffix under the truthiness gate),
the restore-cursor bounds gate (``_cursor_restore_row`` — the saved
nav row in the half-open bounds, else ``None``), the nav-state gate
(``_nav_state`` — the row remembered only when not ``None`` and above
zero) and the Start-Radio seed draw (``_radio_seeds`` — the five-track
sample the recently-played page's own Start Radio shares, routed
through the ``random`` module so a seed steers both backends). The
engines run with two interchangeable backends: the pure-Python
transcriptions in the module and, when the Rust extension is built,
the native implementations bound at the module's tail.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case — error shapes included — so the two
implementations cannot drift apart. The wiring tests pin the page's
routing: the display consults the empty-state and cursor engines with
the page's own state, the footer consults the footer engine with the
table's own count, the nav state consults the nav engine, and Start
Radio consults the seed engine with the table's own tracks. The
load-worker tests keep exercising the page end to end on both
backends — first batch, background top-up, failure and empty states.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult

from ytm_player.ui.pages import liked_songs as ls
from ytm_player.ui.pages.liked_songs import (
    LikedSongsPage,
    _cursor_restore_row,
    _empty_state,
    _footer_text,
    _nav_state,
    _radio_seeds,
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


# ── The empty-state copy ─────────────────────────────────────────────


class TestEmptyState:
    """``_empty_state``'s copy pair, on whichever backend."""

    def test_a_failed_load_answers_the_failure_copy(self):
        assert _empty_state(True) == (
            "Couldn't load liked songs. Check the log at "
            "~/.config/ytm-player/logs/ytm.log for details."
        )

    def test_a_clean_load_answers_the_empty_copy(self):
        assert _empty_state(False) == "No liked songs found."

    def test_the_gate_is_truthiness(self):
        assert _empty_state(1) == _empty_state(True)
        assert _empty_state("x") == _empty_state(True)
        assert _empty_state(0) == _empty_state(False)
        assert _empty_state(None) == _empty_state(False)
        assert _empty_state([]) == _empty_state(False)


# ── The footer text ──────────────────────────────────────────────────


class TestFooterText:
    """``_footer_text``'s assembly, on whichever backend."""

    def test_the_count_composes_the_line(self):
        assert _footer_text(12, False) == "12 liked songs"

    def test_loading_more_appends_the_suffix(self):
        assert _footer_text(12, True) == "12 liked songs (loading more…)"

    def test_zero_is_a_count(self):
        assert _footer_text(0, False) == "0 liked songs"

    def test_the_count_interpolates_through_str(self):
        assert _footer_text("7", False) == "7 liked songs"
        assert _footer_text(2.5, False) == "2.5 liked songs"

    def test_the_suffix_gate_is_truthiness(self):
        assert _footer_text(1, 1) == "1 liked songs (loading more…)"
        assert _footer_text(1, "x") == "1 liked songs (loading more…)"
        assert _footer_text(1, 0) == "1 liked songs"
        assert _footer_text(1, None) == "1 liked songs"
        assert _footer_text(1, []) == "1 liked songs"


# ── The restore-cursor bounds gate ───────────────────────────────────


class TestCursorRestoreRow:
    """``_cursor_restore_row``'s bounds math, on whichever backend."""

    def test_a_saved_row_in_bounds_wins(self):
        assert _cursor_restore_row(1, 3) == 1

    def test_the_first_row_is_in_bounds(self):
        assert _cursor_restore_row(0, 3) == 0

    def test_the_bounds_are_half_open(self):
        assert _cursor_restore_row(2, 3) == 2
        assert _cursor_restore_row(3, 3) is None
        assert _cursor_restore_row(-1, 3) is None

    def test_none_is_not_a_row(self):
        assert _cursor_restore_row(None, 3) is None

    def test_a_zero_row_count_admits_nothing(self):
        assert _cursor_restore_row(0, 0) is None

    def test_a_float_saved_row_compares_like_python(self):
        assert _cursor_restore_row(1.5, 3) == 1.5
        assert _cursor_restore_row(3.0, 3) is None

    def test_a_bool_saved_row_compares_like_python(self):
        assert _cursor_restore_row(True, 3) is True

    def test_a_float_row_count_compares_like_python(self):
        assert _cursor_restore_row(1, 2.5) == 1
        assert _cursor_restore_row(2, 2.5) == 2
        assert _cursor_restore_row(3, 2.5) is None


# ── The nav-state gate ───────────────────────────────────────────────


class TestNavState:
    """``_nav_state``'s remember gate, on whichever backend."""

    def test_nothing_playing_on_the_first_row_is_not_remembered(self):
        assert _nav_state(0) == {}

    def test_none_is_not_remembered(self):
        assert _nav_state(None) == {}

    def test_a_negative_row_is_not_remembered(self):
        assert _nav_state(-1) == {}

    def test_a_positive_row_is_remembered_as_itself(self):
        assert _nav_state(2) == {"cursor_row": 2}

    def test_a_float_row_is_remembered_as_itself(self):
        assert _nav_state(2.5) == {"cursor_row": 2.5}

    def test_a_bool_row_compares_like_python(self):
        assert _nav_state(True) == {"cursor_row": True}


# ── The Start-Radio seed draw ────────────────────────────────────────


class TestRadioSeeds:
    """``_radio_seeds``'s draw, on whichever backend."""

    def test_a_seeded_draw_matches_the_modules_own_sample(self):
        tracks = list(range(12))
        random.seed(42)
        expected = random.sample(tracks, min(5, len(tracks)))
        random.seed(42)
        assert _radio_seeds(tracks) == expected

    def test_the_draw_caps_at_five(self):
        tracks = list(range(12))
        random.seed(7)
        seeds = _radio_seeds(tracks)
        assert len(seeds) == 5
        assert len(set(seeds)) == 5
        assert all(seed in tracks for seed in seeds)

    def test_a_small_pool_draws_itself_whole(self):
        tracks = [10, 20, 30]
        random.seed(7)
        assert sorted(_radio_seeds(tracks)) == [10, 20, 30]

    def test_an_empty_pool_draws_nothing(self):
        assert _radio_seeds([]) == []

    def test_the_draw_routes_through_the_random_module(self, monkeypatch):
        calls: list[tuple[Any, int]] = []

        def fake_sample(population, k):
            calls.append((population, k))
            return ["sentinel"]

        monkeypatch.setattr(random, "sample", fake_sample)
        assert _radio_seeds(list(range(9))) == ["sentinel"]
        assert calls == [(list(range(9)), 5)]

    def test_a_lengthless_pool_raises_the_len_error(self):
        with pytest.raises(TypeError):
            _radio_seeds(5)


# ── Wiring ───────────────────────────────────────────────────────────


class TestDisplayWiring:
    """``_display_tracks`` consults the empty-state and cursor engines
    with the page's own state and applies their answers."""

    def _page(self) -> MagicMock:
        page = MagicMock()
        page._load_failed = False
        page._restore_cursor_row = None
        self._widgets = {
            "#liked-table": MagicMock(),
            "#liked-loading": MagicMock(),
            "#liked-footer": MagicMock(),
        }
        page.query_one.side_effect = lambda selector, cls=None: self._widgets[selector]
        return page

    def test_the_empty_state_engine_is_consulted_with_the_flag(self, monkeypatch):
        calls: list[bool] = []

        def recording(load_failed):
            calls.append(load_failed)
            return "the copy"

        monkeypatch.setattr(ls, "_empty_state", recording)
        page = self._page()
        page._load_failed = True

        LikedSongsPage._display_tracks(page, [])

        assert calls == [True]
        loading = self._widgets["#liked-loading"]
        loading.update.assert_called_once_with("the copy")
        assert loading.display is True
        assert self._widgets["#liked-table"].display is False

    def test_the_cursor_engine_receives_the_saved_row_and_row_count(self, monkeypatch):
        calls: list[tuple[Any, Any]] = []

        def recording(row, row_count):
            calls.append((row, row_count))
            return 4

        monkeypatch.setattr(ls, "_cursor_restore_row", recording)
        page = self._page()
        page._restore_cursor_row = 4
        table = self._widgets["#liked-table"]
        table.row_count = 9

        LikedSongsPage._display_tracks(page, [_raw_track("v1")])

        assert calls == [(4, 9)]
        table.move_cursor.assert_called_once_with(row=4)
        assert page._restore_cursor_row is None  # consumed exactly once
        assert self._widgets["#liked-loading"].display is False

    def test_no_engine_row_leaves_the_cursor_alone(self, monkeypatch):
        monkeypatch.setattr(ls, "_cursor_restore_row", lambda *a: None)
        page = self._page()
        page._restore_cursor_row = 99
        table = self._widgets["#liked-table"]

        LikedSongsPage._display_tracks(page, [_raw_track("v1")])

        table.move_cursor.assert_not_called()
        assert page._restore_cursor_row is None

    def test_the_footer_engine_is_consulted_with_the_tables_own_count(self, monkeypatch):
        calls: list[tuple[Any, Any]] = []

        def recording(total, loading_more):
            calls.append((total, loading_more))
            return "the footer"

        monkeypatch.setattr(ls, "_footer_text", recording)
        page = self._page()
        table = self._widgets["#liked-table"]
        table.track_count = 5

        LikedSongsPage._update_footer(page, loading_more=True)

        assert calls == [(5, True)]
        footer = self._widgets["#liked-footer"]
        footer.update.assert_called_once_with("the footer")


class TestNavStateWiring:
    """``get_nav_state`` consults the nav engine with the table's own
    cursor row."""

    def test_the_nav_engine_is_consulted_with_the_cursor_row(self, monkeypatch):
        calls: list[Any] = []

        def recording(cursor_row):
            calls.append(cursor_row)
            return {"cursor_row": cursor_row}

        monkeypatch.setattr(ls, "_nav_state", recording)
        page = MagicMock()
        table = page.query_one.return_value
        table.cursor_row = 3

        assert LikedSongsPage.get_nav_state(page) == {"cursor_row": 3}
        assert calls == [3]

    def test_a_raising_query_answers_the_empty_state(self, monkeypatch):
        def _boom(*_args: object) -> object:
            raise RuntimeError("table not mounted")

        monkeypatch.setattr(ls, "_nav_state", _boom)
        page = MagicMock()
        page.query_one.side_effect = RuntimeError("table not mounted")

        assert LikedSongsPage.get_nav_state(page) == {}


class TestRadioWiring:
    """``_start_radio`` consults the seed engine with the table's own
    tracks and hands the draw to the host."""

    def _page(self, tracks: list) -> MagicMock:
        page = MagicMock()
        table = page.query_one.return_value
        table.tracks = tracks
        page.app._fetch_and_play_radio = AsyncMock()
        return page

    async def test_the_seed_engine_is_consulted_with_the_tables_tracks(self, monkeypatch):
        calls: list[list] = []
        tracks = [_raw_track(f"v{i}") for i in range(6)]

        def recording(pool):
            calls.append(pool)
            return tracks[:5]

        monkeypatch.setattr(ls, "_radio_seeds", recording)
        page = self._page(tracks)

        await LikedSongsPage._start_radio(page)

        assert calls == [tracks]
        page.app._fetch_and_play_radio.assert_awaited_once_with(
            tracks[:5], label="Radio: Liked Songs"
        )

    async def test_no_tracks_consult_nothing_and_start_no_radio(self, monkeypatch):
        def _boom(*_args: object) -> object:
            raise AssertionError("no engine may run without tracks")

        monkeypatch.setattr(ls, "_radio_seeds", _boom)
        page = self._page([])

        await LikedSongsPage._start_radio(page)

        page.app._fetch_and_play_radio.assert_not_awaited()


class _LikedHost(App[None]):
    """A minimal host whose ytmusic serves one canned liked playlist."""

    def __init__(self, first_batch: list | None, remaining: list | None = None) -> None:
        super().__init__()
        self._first_batch = first_batch
        self._remaining = remaining
        self.fetch_calls: list[dict] = []
        self._ytmusic = SimpleNamespace(get_liked_songs=self._get_liked_songs)

    async def _get_liked_songs(self, limit=None, timeout=None):
        self.fetch_calls.append({"limit": limit, "timeout": timeout})
        if limit is None:
            return self._remaining
        return self._first_batch

    @property
    def ytmusic(self):
        return self._ytmusic

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield LikedSongsPage(id="liked")


def _label_texts(page: LikedSongsPage) -> list[str]:
    from textual.widgets import Label, Static

    texts = [str(w.render()) for w in page.query(Label)]
    texts += [str(w.render()) for w in page.query(Static)]
    return texts


class TestLoadWiring:
    """``_load_liked_songs`` consults the engines end to end — the
    first batch, the background top-up and both empty states."""

    async def test_the_first_batch_mounts_and_footers_through_the_engine(self):
        app = _LikedHost([_raw_track("v1"), _raw_track("v2")])

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LikedSongsPage)

            table = page.query_one("#liked-table", TrackTable)
            assert table.display is True
            assert table.row_count == 2
            assert page.track_count == 2

            texts = _label_texts(page)
            assert any("2 liked songs" in text for text in texts)
            assert not any("loading more" in text for text in texts)
            # Fewer tracks than the first batch: no tail fetch.
            assert all(call["limit"] is not None for call in app.fetch_calls)

    async def test_a_full_first_batch_kicks_off_the_top_up(self):
        app = _LikedHost(
            [_raw_track(f"v{i}") for i in range(300)],
            remaining=[_raw_track(f"v{i}") for i in range(305)],
        )

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LikedSongsPage)

            table = page.query_one("#liked-table", TrackTable)
            assert table.row_count == 305
            assert page.track_count == 305

            texts = _label_texts(page)
            assert any("305 liked songs" in text for text in texts)
            assert not any("loading more" in text for text in texts)
            # The top-up fetched the whole playlist again, unlimited.
            assert app.fetch_calls[-1]["limit"] is None

    async def test_a_failed_fetch_answers_the_failure_copy(self):
        app = _LikedHost(None)

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LikedSongsPage)

            table = page.query_one("#liked-table", TrackTable)
            assert table.display is False
            texts = _label_texts(page)
            assert any("Couldn't load liked songs" in text for text in texts)

    async def test_a_genuinely_empty_playlist_answers_the_empty_copy(self):
        app = _LikedHost([])

        async with app.run_test() as pilot:
            await asyncio.wait_for(app.workers.wait_for_complete(), 5)
            await pilot.pause()
            page = app.query_one(LikedSongsPage)

            table = page.query_one("#liked-table", TrackTable)
            assert table.display is False
            texts = _label_texts(page)
            assert any("No liked songs found." in text for text in texts)
            assert not any("Couldn't load" in text for text in texts)


# ── Native vs fallback parity ────────────────────────────────────────

# Empty-state cases: the load-failed values.
_EMPTY_PARITY_CASES = [True, False, 0, 1, "", "x", None, [], [0]]

# Footer cases: [total, loading_more].
_FOOTER_PARITY_CASES = [
    [12, False],
    [12, True],
    [0, True],
    [1, 1],
    [1, 0],
    [1, None],
    [1, ""],
    [2.5, False],
    ["7", "x"],
    [None, None],
]

# Cursor cases: [row, row_count].
_CURSOR_PARITY_CASES = [
    [None, 3],
    [0, 0],
    [0, 3],
    [1, 3],
    [2, 3],
    [3, 3],
    [-1, 3],
    [1.5, 3],
    [3.0, 3],
    [True, 3],
    [1, 2.5],
]

# Nav cases: the cursor rows.
_NAV_PARITY_CASES = [None, 0, 1, -1, 2, 2.5, True, False]

# Radio cases: {"seed": the RNG seed before the draw, "tracks": the pool}.
# A lengthless pool raises the reference's own len TypeError on both
# sides, under a seed that stays untouched either way.
_RADIO_PARITY_CASES = [
    {"seed": 1, "tracks": []},
    {"seed": 2, "tracks": [10, 20]},
    {"seed": 3, "tracks": list(range(12))},
    {"seed": 4, "tracks": [{"videoId": f"v{i}", "title": "T"} for i in range(9)]},
    {"seed": 5, "tracks": 5},
]

_CHILD_PROG = """
import json
import random
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.liked_songs import (
    _cursor_restore_row,
    _empty_state,
    _footer_text,
    _nav_state,
    _radio_seeds,
)


def outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


payload = json.load(sys.stdin)

radio = []
for case in payload["radio"]:
    random.seed(case["seed"])
    radio.append(outcome(_radio_seeds, case["tracks"]))

json.dump(
    {
        "empty": [outcome(_empty_state, case) for case in payload["empty"]],
        "footer": [outcome(_footer_text, *case) for case in payload["footer"]],
        "cursor": [outcome(_cursor_restore_row, *case) for case in payload["cursor"]],
        "nav": [outcome(_nav_state, case) for case in payload["nav"]],
        "radio": radio,
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

    radio = []
    for case in _RADIO_PARITY_CASES:
        random.seed(case["seed"])
        radio.append(outcome(_radio_seeds, case["tracks"]))

    return {
        "empty": [outcome(_empty_state, case) for case in _EMPTY_PARITY_CASES],
        "footer": [outcome(_footer_text, *case) for case in _FOOTER_PARITY_CASES],
        "cursor": [outcome(_cursor_restore_row, *case) for case in _CURSOR_PARITY_CASES],
        "nav": [outcome(_nav_state, case) for case in _NAV_PARITY_CASES],
        "radio": radio,
    }


def _run_child():
    """The pure-Python reference outcomes, run in a child interpreter
    with the extension import blocked."""
    payload = {
        "empty": _EMPTY_PARITY_CASES,
        "footer": _FOOTER_PARITY_CASES,
        "cursor": _CURSOR_PARITY_CASES,
        "nav": _NAV_PARITY_CASES,
        "radio": _RADIO_PARITY_CASES,
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same copies, same lines,
    same rows, same states, same draws under the same seeds, same
    errors with the same messages — so the two implementations cannot
    drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_every_engine_matches_the_fallback(self):
        fallback = _run_child()
        native = json.loads(json.dumps(_native_outcomes()))
        for family, cases in [
            ("empty", _EMPTY_PARITY_CASES),
            ("footer", _FOOTER_PARITY_CASES),
            ("cursor", _CURSOR_PARITY_CASES),
            ("nav", _NAV_PARITY_CASES),
            ("radio", _RADIO_PARITY_CASES),
        ]:
            for case, n, f in zip(cases, native[family], fallback[family]):
                assert n == f, (
                    f"native and fallback {family} engine differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
