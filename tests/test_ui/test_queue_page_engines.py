"""Tests for the queue-page view and mutation engines.

The engines behind ``QueuePage``: the now-playing header view model
(``_now_playing_view``, the shared core of ``_update_current_track`` and
``_refresh_queue``), the radio-seed source label (``_queue_source_view``,
the ``_update_queue_source`` core), the footer text (``_queue_footer_text``),
and the mutation decisions behind the page's ``d`` / ``J`` / ``K``
(``_remove_target``'s stale-table queue-position refusal,
``_post_remove_cursor_row``'s cursor re-landing, and ``_move_target``'s
clamped count-multiplied reorder destination). The pure-Python references
live in ``ui/pages/queue.py``; when the Rust extension is built the same
engines run natively (``_ytm_player_native.queue_now_playing`` & co).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart.

The parity payload sticks to seeds with str titles (or the title key
missing): the up-to-three summary's ``", ".join`` is the one consumer
that requires str titles, and there the native engine deliberately
renders through ``str()`` (staying total where the reference's join
would raise on junk seeds) — the unreachable-junk divergence documented
on ``queue_source_view`` itself, out of parity scope exactly like the
picker's non-str-title skip. The now-playing view model has no such
seam — its f-string always rendered through ``str()`` — so its payload
includes the junk-shape tracks.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from ytm_player.config.keymap import Action
from ytm_player.services.queue import QueueManager
from ytm_player.ui.pages import queue as queue_module
from ytm_player.ui.pages.queue import (
    QueuePage,
    _move_target,
    _now_playing_view,
    _post_remove_cursor_row,
    _queue_footer_text,
    _queue_source_view,
    _remove_target,
)
from ytm_player.ui.widgets.track_table import TrackTable

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── _now_playing_view ───────────────────────────────────────────────


class TestNowPlayingView:
    def test_no_track_hides_the_header(self):
        assert _now_playing_view(None) is None

    def test_an_empty_track_is_falsy(self):
        """``if current_track:`` is a truthiness gate, not an is-None test."""
        assert _now_playing_view({}) is None

    def test_full_track(self):
        view = _now_playing_view({"title": "Song", "artist": "Artist", "video_id": "vid"})
        assert view == {
            "title_label": "Now Playing: Song",
            "artist": "Artist",
            "video_id": "vid",
        }

    def test_missing_title_and_artist_default_to_unknown(self):
        view = _now_playing_view({"video_id": "v"})
        assert view["title_label"] == "Now Playing: Unknown"
        assert view["artist"] == "Unknown"
        assert view["video_id"] == "v"

    def test_missing_video_id_defaults_to_empty_string(self):
        """``set_playing("")`` — not None — is what a track without a
        video_id produces, so the indicator clears without the "no
        current track" distinction."""
        view = _now_playing_view({"title": "T", "artist": "A"})
        assert view["video_id"] == ""

    def test_present_but_none_values_pass_through_raw(self):
        """``.get(key, default)`` defaults only a MISSING key — a present
        ``None`` flows on raw, and the f-string renders it as ``None``."""
        view = _now_playing_view({"title": None, "artist": None, "video_id": None})
        assert view == {"title_label": "Now Playing: None", "artist": None, "video_id": None}

    def test_non_str_title_renders_through_str(self):
        view = _now_playing_view({"title": 42, "artist": 7, "video_id": True})
        assert view["title_label"] == "Now Playing: 42"
        assert view["artist"] == 7
        assert view["video_id"] is True


# ── _queue_source_view ──────────────────────────────────────────────


class TestQueueSourceView:
    def test_no_seeds_hides_the_label(self):
        assert _queue_source_view(None, True) is None

    def test_empty_seeds_hides_the_label(self):
        assert _queue_source_view([], True) is None

    def test_setting_off_hides_the_label(self):
        assert _queue_source_view([{"title": "S"}], False) is None

    def test_one_seed(self):
        assert _queue_source_view([{"title": "Radio"}], True) == (
            "Generated from: Radio",
            "Generated from:\n  • Radio",
        )

    def test_three_seed_titles_join_comma_separated(self):
        seeds = [{"title": f"S{i}"} for i in range(1, 4)]
        assert _queue_source_view(seeds, True) == (
            "Generated from: S1, S2, S3",
            "Generated from:\n  • S1\n  • S2\n  • S3",
        )

    def test_four_seed_titles_collapse_the_tail(self):
        seeds = [{"title": f"S{i}"} for i in range(1, 5)]
        assert _queue_source_view(seeds, True) == (
            "Generated from: S1, S2 + 2 more",
            "Generated from:\n  • S1\n  • S2\n  • S3\n  • S4",
        )

    def test_five_seed_titles_count_the_whole_tail(self):
        seeds = [{"title": f"S{i}"} for i in range(1, 6)]
        assert _queue_source_view(seeds, True)[0] == "Generated from: S1, S2 + 3 more"

    def test_missing_seed_title_defaults_to_unknown(self):
        assert _queue_source_view([{}], True) == (
            "Generated from: Unknown",
            "Generated from:\n  • Unknown",
        )

    def test_non_str_seed_title_renders_through_str_from_the_fourth_seed_on(self):
        """The four-plus summary and the tooltip always interpolated
        through ``str()`` — a non-str title renders the same way on both
        backends. (Among up to three seeds the reference's ``", ".join``
        would instead raise on a non-str title, where the native engine
        deliberately stays total — the documented divergence on
        ``queue_source_view``, out of contract scope because the two
        backends disagree on it by design.)"""
        seeds = [{"title": 7}, {"title": "S2"}, {"title": "S3"}, {"title": "S4"}]
        assert _queue_source_view(seeds, True) == (
            "Generated from: 7, S2 + 2 more",
            "Generated from:\n  • 7\n  • S2\n  • S3\n  • S4",
        )


# ── _queue_footer_text ──────────────────────────────────────────────


class TestQueueFooterText:
    @pytest.mark.parametrize(
        "count, text",
        [(0, "Tracks: 0"), (1, "Tracks: 1"), (42, "Tracks: 42")],
    )
    def test_footer_text(self, count, text):
        assert _queue_footer_text(count) == text


# ── _remove_target ──────────────────────────────────────────────────


class TestRemoveTarget:
    def test_no_highlighted_row_refuses(self):
        """A filtered-to-nothing table names no queue position."""
        assert _remove_target(None, 5) is None

    def test_first_middle_and_last_positions_are_removable(self):
        for idx in (0, 2, 4):
            assert _remove_target(idx, 5) == idx

    def test_a_stale_table_refuses_rather_than_removing_the_wrong_entry(self):
        """The queue shrank underneath an unrefreshed table: the mapped
        index is at (or past) the queue length and must be refused."""
        assert _remove_target(5, 5) is None
        assert _remove_target(6, 5) is None

    def test_negative_index_refuses(self):
        assert _remove_target(-1, 5) is None

    def test_empty_queue_refuses_everything(self):
        assert _remove_target(0, 0) is None


# ── _post_remove_cursor_row ─────────────────────────────────────────


class TestPostRemoveCursorRow:
    def test_no_rows_left_leaves_the_cursor_untouched(self):
        assert _post_remove_cursor_row(0, 0) is None

    def test_negative_row_count_is_no_rows(self):
        assert _post_remove_cursor_row(2, -1) is None

    def test_the_visible_row_stays_when_still_in_bounds(self):
        assert _post_remove_cursor_row(2, 5) == 2
        assert _post_remove_cursor_row(0, 1) == 0

    def test_the_row_taking_the_removed_place_gets_the_cursor(self):
        """The cursor clamps into the refreshed table's bounds."""
        assert _post_remove_cursor_row(7, 5) == 4
        assert _post_remove_cursor_row(3, 3) == 2


# ── _move_target ────────────────────────────────────────────────────


class TestMoveTarget:
    def test_no_highlighted_row_moves_nothing(self):
        assert _move_target(None, 1, 1, 5) is None

    def test_move_down_one_queue_position(self):
        assert _move_target(0, 1, 1, 5) == 1

    def test_move_up_one_queue_position(self):
        assert _move_target(2, -1, 1, 5) == 1

    def test_the_count_multiplies_the_distance(self):
        assert _move_target(0, 1, 3, 5) == 3
        assert _move_target(3, -1, 2, 5) == 1

    def test_the_destination_clamps_at_the_bottom(self):
        """``15J`` near the end simply lands at the bottom."""
        assert _move_target(3, 1, 10, 5) == 4

    def test_the_destination_clamps_at_the_top(self):
        assert _move_target(2, -1, 10, 5) == 0

    def test_a_clamped_noop_moves_nothing(self):
        """Already at the top/bottom — even with a count — nothing moves."""
        assert _move_target(0, -1, 1, 5) is None
        assert _move_target(0, -1, 10, 5) is None
        assert _move_target(4, 1, 1, 5) is None

    def test_empty_queue_moves_nowhere(self):
        """``max(0, min(anything, -1))`` is 0 — equal to the source, so
        the move is refused exactly as the reference's early return."""
        assert _move_target(0, 1, 1, 0) is None

    def test_single_item_queue_moves_nowhere(self):
        assert _move_target(0, 1, 1, 1) is None


# ── Native / fallback parity ─────────────────────────────────────────

_TRACK_CASES: list[Any] = [
    None,
    {},
    {"title": "Song", "artist": "Artist", "video_id": "vid"},
    {"video_id": "v"},
    {"title": "T", "artist": "A"},
    {"title": None, "artist": None, "video_id": None},
    {"title": 42, "artist": 7, "video_id": True},
    {"title": "Ünïcødé", "artist": "ø", "video_id": "id-1"},
]

_SOURCE_CASES: list[tuple[Any, bool]] = [
    (None, True),
    ([], True),
    ([{"title": "S"}], True),
    ([{"title": "S"}], False),
    ([{"title": "S1"}, {"title": "S2"}, {"title": "S3"}], True),
    ([{"title": "S1"}, {"title": "S2"}, {"title": "S3"}, {"title": "S4"}], True),
    ([{"title": "S1"}, {"title": "S2"}, {"title": "S3"}, {"title": "S4"}, {"title": "S5"}], False),
    ([{}], True),
    ([{"title": "A"}, {}], True),
]

_FOOTER_COUNTS = [0, 1, 42, 10**12]

_REMOVE_CASES: list[tuple[Any, int]] = [
    (None, 5),
    (0, 5),
    (2, 5),
    (4, 5),
    (5, 5),
    (6, 5),
    (-1, 5),
    (0, 0),
    (0, 1),
]

_CURSOR_CASES: list[tuple[int, int]] = [
    (0, 0),
    (2, -1),
    (2, 5),
    (0, 1),
    (7, 5),
    (3, 3),
]

_MOVE_CASES: list[tuple[Any, int, int, int]] = [
    (None, 1, 1, 5),
    (0, 1, 1, 5),
    (2, -1, 1, 5),
    (0, 1, 3, 5),
    (3, -1, 2, 5),
    (3, 1, 10, 5),
    (2, -1, 10, 5),
    (0, -1, 1, 5),
    (0, -1, 10, 5),
    (4, 1, 1, 5),
    (0, 1, 1, 0),
    (0, 1, 1, 1),
    (4, -1, 2, 7),
    (6, 1, 4, 9),
]

# The child interpreter runs this program with the case lists on stdin
# and prints the reference engines' results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.queue import (
    _move_target,
    _now_playing_view,
    _post_remove_cursor_row,
    _queue_footer_text,
    _queue_source_view,
    _remove_target,
)

payload = json.load(sys.stdin)

out = {
    "now_playing": [_now_playing_view(track) for track in payload["tracks"]],
    "source": [
        _queue_source_view(seeds, show) for seeds, show in payload["source_cases"]
    ],
    "footer": [_queue_footer_text(count) for count in payload["footer_counts"]],
    "remove": [
        _remove_target(selected, length) for selected, length in payload["remove_cases"]
    ],
    "cursor": [
        _post_remove_cursor_row(visible, row_count)
        for visible, row_count in payload["cursor_cases"]
    ],
    "move": [
        _move_target(from_index, direction, count, length)
        for from_index, direction, count, length in payload["move_cases"]
    ],
}
json.dump(out, sys.stdout)
"""


def _native_results(payload: dict) -> dict:
    """Run whichever engines the module loaded (native when built) over
    the payload."""
    return {
        "now_playing": [_now_playing_view(track) for track in payload["tracks"]],
        "source": [_queue_source_view(seeds, show) for seeds, show in payload["source_cases"]],
        "footer": [_queue_footer_text(count) for count in payload["footer_counts"]],
        "remove": [
            _remove_target(selected, length) for selected, length in payload["remove_cases"]
        ],
        "cursor": [
            _post_remove_cursor_row(visible, row_count)
            for visible, row_count in payload["cursor_cases"]
        ],
        "move": [
            _move_target(from_index, direction, count, length)
            for from_index, direction, count, length in payload["move_cases"]
        ],
    }


def _fallback_results(payload: dict) -> dict:
    """Run the pure-Python reference engines over the payload in a child
    interpreter with the native extension import blocked."""
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native queue-page engines
    must produce exactly the same view models and mutation decisions as
    the pure-Python references for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        payload = {
            "tracks": _TRACK_CASES,
            "source_cases": _SOURCE_CASES,
            "footer_counts": _FOOTER_COUNTS,
            "remove_cases": _REMOVE_CASES,
            "cursor_cases": _CURSOR_CASES,
            "move_cases": _MOVE_CASES,
        }
        native = json.loads(json.dumps(_native_results(payload)))
        fallback = _fallback_results(payload)
        for key, cases in (
            ("now_playing", _TRACK_CASES),
            ("source", _SOURCE_CASES),
            ("footer", _FOOTER_COUNTS),
            ("remove", _REMOVE_CASES),
            ("cursor", _CURSOR_CASES),
            ("move", _MOVE_CASES),
        ):
            for i, case in enumerate(cases):
                assert native[key][i] == fallback[key][i], (
                    f"native and fallback {key} differ for case {i}: {case}"
                )


# ── Mounted widget integration ───────────────────────────────────────


class _FakePlayer:
    """Just enough Player surface for QueuePage/TrackTable mounting."""

    current_track: dict | None = None

    def on(self, *_args: Any) -> None: ...

    def off(self, *_args: Any) -> None: ...


class _QueueHost(App):
    """Minimal host app exposing the attributes QueuePage reads."""

    def __init__(self, queue: QueueManager) -> None:
        super().__init__()
        self.queue = queue
        self.player = _FakePlayer()

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield QueuePage()


def _track(video_id: str, title: str) -> dict:
    return {
        "video_id": video_id,
        "title": title,
        "artist": "A",
        "artists": [{"name": "A", "id": "1"}],
        "album": "",
        "album_id": None,
        "duration": 120,
        "thumbnail_url": None,
        "is_video": False,
    }


def _header_labels(page: QueuePage) -> list[str]:
    return [str(label.render()) for label in page.query("#queue-header Label")]


class TestQueuePageWidget:
    """The QueuePage glue: the engines' output reaching the header, the
    footer, the source label and the empty state through the real
    Textual loop.
    """

    async def test_header_shows_the_current_track(self):
        queue = QueueManager()
        queue.add_multiple([_track("t1", "first"), _track("t2", "second")])
        queue.jump_to(1)
        async with _QueueHost(queue).run_test() as pilot:
            page = pilot.app.query_one(QueuePage)

            assert _header_labels(page) == ["Now Playing: second", "A"]
            assert page.query_one("#queue-header").display is True

    async def test_empty_queue_hides_the_header_and_shows_the_empty_state(self):
        async with _QueueHost(QueueManager()).run_test() as pilot:
            page = pilot.app.query_one(QueuePage)

            assert _header_labels(page) == []
            assert page.query_one("#queue-header").display is False
            assert page.query_one("#queue-empty").display is True
            assert str(page.query_one("#queue-footer", Static).render()) == "Tracks: 0"

    async def test_footer_counts_the_queue(self):
        queue = QueueManager()
        queue.add_multiple([_track(f"t{i}", f"song {i}") for i in range(4)])
        async with _QueueHost(queue).run_test() as pilot:
            page = pilot.app.query_one(QueuePage)

            assert str(page.query_one("#queue-footer", Static).render()) == "Tracks: 4"

    async def test_lightweight_update_renders_the_new_track_without_rebuilding(self):
        queue = QueueManager()
        queue.add_multiple([_track("t1", "first"), _track("t2", "second")])
        queue.jump_to(0)
        async with _QueueHost(queue).run_test() as pilot:
            app = pilot.app
            page = app.query_one(QueuePage)
            table = app.query_one("#queue-table", TrackTable)
            table.move_cursor(row=1)
            await pilot.pause()
            rows_before = table.row_count

            queue.jump_to(1)
            page._update_current_track()
            await pilot.pause()

            assert _header_labels(page) == ["Now Playing: second", "A"]
            # The table was not rebuilt — the cursor the user moved stays.
            assert table.row_count == rows_before
            assert table.cursor_row == 1

    async def test_source_label_summarises_the_radio_seeds(self, monkeypatch):
        queue = QueueManager()
        queue.add_multiple([_track("t1", "song")])
        queue.radio_seeds = [{"title": "Seed One"}, {"title": "Seed Two"}]
        settings = SimpleNamespace(ui=SimpleNamespace(show_queue_source=True))
        monkeypatch.setattr(queue_module, "get_settings", lambda: settings)
        async with _QueueHost(queue).run_test() as pilot:
            source = pilot.app.query_one("#queue-source", Static)

            assert source.display is True
            assert str(source.render()) == "Generated from: Seed One, Seed Two"
            assert source.tooltip == "Generated from:\n  • Seed One\n  • Seed Two"

    async def test_source_label_hides_when_the_setting_is_off(self, monkeypatch):
        queue = QueueManager()
        queue.add_multiple([_track("t1", "song")])
        queue.radio_seeds = [{"title": "Seed One"}]
        settings = SimpleNamespace(ui=SimpleNamespace(show_queue_source=False))
        monkeypatch.setattr(queue_module, "get_settings", lambda: settings)
        async with _QueueHost(queue).run_test() as pilot:
            source = pilot.app.query_one("#queue-source", Static)

            assert source.display is False
            assert str(source.render()) == ""
            assert source.tooltip is None

    async def test_source_label_hides_without_seeds(self, monkeypatch):
        queue = QueueManager()
        queue.add_multiple([_track("t1", "song")])
        settings = SimpleNamespace(ui=SimpleNamespace(show_queue_source=True))
        monkeypatch.setattr(queue_module, "get_settings", lambda: settings)
        async with _QueueHost(queue).run_test() as pilot:
            source = pilot.app.query_one("#queue-source", Static)

            assert source.display is False

    async def test_delete_re_renders_the_footer_through_the_engine(self):
        queue = QueueManager()
        queue.add_multiple([_track("t1", "one"), _track("t2", "two")])
        async with _QueueHost(queue).run_test() as pilot:
            app = pilot.app
            page = app.query_one(QueuePage)
            app.query_one("#queue-table", TrackTable).move_cursor(row=0)
            await pilot.pause()

            await page.handle_action(Action.DELETE_ITEM)

            assert str(app.query_one("#queue-footer", Static).render()) == "Tracks: 1"
