"""Tests for the lyrics-scroll engines (the sidebar's follow mode).

Four engines behind ``LyricsSidebar``'s synced view: the current-line
lookup (``_line_for_position`` — the bisect behind
``_on_position_change``), the highlight changes (``_highlight_changes``
— which rows change class, behind ``_apply_line_highlight``), the
follow-mode verdict (``_scroll_follows`` — still on, or resumed once
the manual-scroll pause has lapsed) and the centering target
(``_scroll_target`` — the y that centers the current line). The
pure-Python references live in ``ui/sidebars/lyrics_sidebar.py``; when
the Rust extension is built the same engines run natively
(``_ytm_player_native.lyrics_*``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart. The wiring tests mount
the sidebar itself and check the follow mode drives the real reactive,
class list and scroll widget.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from ytm_player.ui.sidebars.lyrics_sidebar import (
    _highlight_changes,
    _line_for_position,
    _scroll_follows,
    _scroll_target,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Current-line lookup ──────────────────────────────────────────────


class TestLineForPosition:
    def test_before_the_first_timestamp_answers_minus_one(self):
        assert _line_for_position([1.0, 2.0, 3.0], 0.5) == -1

    def test_exactly_on_a_timestamp_lands_on_that_line(self):
        assert _line_for_position([1.0, 2.0, 3.0], 2.0) == 1

    def test_between_timestamps_lands_on_the_previous_line(self):
        assert _line_for_position([1.0, 2.0, 3.0], 2.5) == 1

    def test_past_the_last_timestamp_lands_on_the_last_line(self):
        assert _line_for_position([1.0, 2.0, 3.0], 99.0) == 2

    def test_empty_list_answers_minus_one(self):
        assert _line_for_position([], 5.0) == -1

    def test_duplicate_timestamps_pick_the_last_duplicate(self):
        # bisect_right counts every equal timestamp as reached.
        assert _line_for_position([1.0, 2.0, 2.0, 3.0], 2.0) == 2

    def test_zero_position_on_zero_timestamp_lands_on_line_zero(self):
        assert _line_for_position([0.0, 5.0], 0.0) == 0

    def test_int_timestamps_compare_by_value(self):
        assert _line_for_position([0, 5, 10], 6) == 1

    def test_nan_position_falls_through_to_the_last_line(self):
        # Every NaN comparison is false, so the search keeps walking
        # right — the same verdict the reference's own bisect reaches.
        assert _line_for_position([1.0, 2.0, 3.0], float("nan")) == 2

    def test_infinite_position_lands_on_the_last_line(self):
        assert _line_for_position([1.0, 2.0, 3.0], float("inf")) == 2
        assert _line_for_position([1.0, 2.0, 3.0], float("-inf")) == -1

    def test_str_timestamp_raises_the_comparisons_own_type_error(self):
        with pytest.raises(TypeError):
            _line_for_position(["a", "b"], 1.0)


# ── Highlight changes ────────────────────────────────────────────────


class TestHighlightChanges:
    def test_unchanged_index_answers_nothing(self):
        assert _highlight_changes(3, 3, 10) == []

    def test_first_highlight_marks_only_the_new_row_current(self):
        assert _highlight_changes(-1, 0, 5) == [(0, "lyrics-current")]

    def test_forward_move_marks_the_span_played(self):
        assert _highlight_changes(0, 3, 5) == [
            (0, "lyrics-played"),
            (1, "lyrics-played"),
            (2, "lyrics-played"),
            (3, "lyrics-current"),
        ]

    def test_backward_move_marks_the_span_upcoming(self):
        assert _highlight_changes(3, 1, 5) == [
            (1, "lyrics-current"),
            (2, "lyrics-upcoming"),
            (3, "lyrics-upcoming"),
        ]

    def test_reset_to_top_restyles_only_the_remembered_row(self):
        # new_index -1 with an in-range old: the negative new drops out
        # of the set, the range walk needs both endpoints at or after
        # the top — only the old row restyles, as upcoming.
        assert _highlight_changes(2, -1, 5) == [(2, "lyrics-upcoming")]

    def test_stale_old_beyond_the_count_drops_out(self):
        # A rebuilt view resets to an in-range line while the
        # remembered old index points past the new (shorter) list.
        assert _highlight_changes(7, 2, 5) == [
            (2, "lyrics-current"),
            (3, "lyrics-upcoming"),
            (4, "lyrics-upcoming"),
        ]

    def test_new_beyond_the_count_marks_the_span_played(self):
        assert _highlight_changes(1, 6, 5) == [
            (1, "lyrics-played"),
            (2, "lyrics-played"),
            (3, "lyrics-played"),
            (4, "lyrics-played"),
        ]

    def test_both_out_of_range_answers_nothing(self):
        assert _highlight_changes(7, 8, 5) == []
        assert _highlight_changes(-1, 9, 5) == []
        assert _highlight_changes(4, -1, 3) == []

    def test_zero_count_answers_nothing(self):
        assert _highlight_changes(0, 1, 0) == []

    def test_pairs_are_ascending_and_unique(self):
        pairs = _highlight_changes(4, 1, 10)
        indices = [index for index, _cls in pairs]
        assert indices == sorted(set(indices))

    def test_every_verdict_is_one_of_the_three_classes(self):
        for old, new, count in [(-1, 0, 3), (2, 0, 3), (0, 2, 3), (1, 5, 3)]:
            for _index, cls in _highlight_changes(old, new, count):
                assert cls in ("lyrics-played", "lyrics-current", "lyrics-upcoming")


# ── Follow-mode verdict ──────────────────────────────────────────────


class TestScrollFollows:
    def test_follow_already_on_stays_on(self):
        assert _scroll_follows(True, 0.0, 5.0, 3.0) is True

    def test_recent_manual_scroll_stays_paused(self):
        assert _scroll_follows(False, 4.0, 5.0, 3.0) is False

    def test_lapsed_pause_resumes(self):
        assert _scroll_follows(False, 1.0, 5.0, 3.0) is True

    def test_exactly_at_the_delay_stays_paused(self):
        # The gate is strictly greater-than, like the reference.
        assert _scroll_follows(False, 2.0, 5.0, 3.0) is False

    def test_small_clocks_resume_the_same(self):
        # time.monotonic can start near zero; the verdict is the same
        # arithmetic on any clock scale.
        assert _scroll_follows(False, 0.0, 0.1, 0.05) is True
        assert _scroll_follows(False, 0.0, 0.04, 0.05) is False


# ── Centering target ─────────────────────────────────────────────────


class TestScrollTarget:
    def test_centers_the_line_in_the_viewport(self):
        # y 17, height 2, viewport 12: 17 - 6 + 1 = 12.
        assert _scroll_target(17, 2, 12) == 12

    def test_never_scrolls_above_the_top(self):
        assert _scroll_target(0, 1, 10) == 0
        assert _scroll_target(3, 2, 20) == 0

    def test_floor_divides_the_halves(self):
        # Odd viewport and widget heights floor-divide, like the
        # reference's //.
        assert _scroll_target(10, 3, 5) == 9
        assert _scroll_target(9, 5, 3) == 9 - 1 + 2

    def test_zero_geometry_answers_zero(self):
        assert _scroll_target(0, 0, 0) == 0


# ── Sidebar wiring ───────────────────────────────────────────────────


def _mounted_sidebar():
    """The test-body helper: build the host app class and return it."""
    from textual.app import App, ComposeResult

    from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

    class _Host(App):
        """Minimal host providing the theme variables the CSS needs."""

        def get_css_variables(self) -> dict[str, str]:
            variables = super().get_css_variables()
            variables["lyrics-played"] = "#999999"
            variables["lyrics-current"] = "#ff4e45"
            variables["lyrics-upcoming"] = "#cccccc"
            return variables

        def compose(self) -> ComposeResult:
            yield LyricsSidebar()

    return _Host


class TestSidebarWiring:
    async def test_position_change_drives_the_reactive_and_classes(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(0.0, "a"), (5.0, "b"), (10.0, "c")]
            sidebar._synced_timestamps = [0.0, 5.0, 10.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            sidebar._on_position_change(6.0)

            assert sidebar.current_line_index == 1
            widgets = sidebar._lyric_widgets
            # The first highlight (-1 -> 1) only restyles the new row:
            # the -1 endpoint drops out of the set and the range walk
            # needs both endpoints at or after the top, so row 0 keeps
            # its initial upcoming class — the reference's own walk.
            assert widgets[0].has_class("lyrics-upcoming")
            assert widgets[1].has_class("lyrics-current")
            assert widgets[2].has_class("lyrics-upcoming")

            sidebar._on_position_change(11.0)

            assert sidebar.current_line_index == 2
            # The 1 -> 2 transition restyles the span between them.
            assert widgets[1].has_class("lyrics-played")
            assert widgets[2].has_class("lyrics-current")
            assert widgets[0].has_class("lyrics-upcoming")

    async def test_position_before_the_first_line_clears_the_highlight(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            from textual.containers import VerticalScroll

            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(2.0, "a"), (5.0, "b")]
            sidebar._synced_timestamps = [2.0, 5.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            sidebar._on_position_change(3.0)
            assert sidebar.current_line_index == 0

            # Back before the first timestamp: index -1, the one
            # remembered row restyles as upcoming.
            sidebar._on_position_change(1.0)
            assert sidebar.current_line_index == -1
            assert sidebar._lyric_widgets[0].has_class("lyrics-upcoming")
            assert not sidebar._lyric_widgets[0].has_class("lyrics-current")

            # The scroll container is mounted and untouched by the
            # out-of-range new index.
            assert sidebar.query_one("#ls-scroll", VerticalScroll) is not None

    async def test_auto_scroll_lands_on_the_centered_target(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            from textual.containers import VerticalScroll

            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(0.0, "a"), (5.0, "b"), (10.0, "c")]
            sidebar._synced_timestamps = [0.0, 5.0, 10.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            scroll = sidebar.query_one("#ls-scroll", VerticalScroll)
            scroll.scroll_to = MagicMock()
            sidebar._on_position_change(6.0)

            scroll.scroll_to.assert_called_once()
            kwargs = scroll.scroll_to.call_args.kwargs
            assert kwargs.get("animate") is True
            widget = sidebar._lyric_widgets[1]
            expected = _scroll_target(
                widget.virtual_region.y,
                widget.virtual_region.height,
                scroll.scrollable_content_region.height,
            )
            assert kwargs.get("y") == expected

    async def test_repeated_position_does_not_restyle_or_rescroll(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            from textual.containers import VerticalScroll

            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(0.0, "a"), (5.0, "b"), (10.0, "c")]
            sidebar._synced_timestamps = [0.0, 5.0, 10.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            scroll = sidebar.query_one("#ls-scroll", VerticalScroll)
            scroll.scroll_to = MagicMock()

            sidebar._on_position_change(6.0)
            sidebar._on_position_change(6.5)

            # Same line: the reactive gate holds, no second scroll.
            scroll.scroll_to.assert_called_once()
            assert sidebar.current_line_index == 1

    async def test_manual_pause_within_the_delay_holds_the_view(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            import time as _time

            from textual.containers import VerticalScroll

            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(0.0, "a"), (5.0, "b"), (10.0, "c")]
            sidebar._synced_timestamps = [0.0, 5.0, 10.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            scroll = sidebar.query_one("#ls-scroll", VerticalScroll)
            scroll.scroll_to = MagicMock()

            # A manual scroll just happened: the follow mode pauses...
            sidebar._auto_scroll = False
            sidebar._last_manual_scroll = _time.monotonic()
            sidebar._on_position_change(6.0)

            assert sidebar._auto_scroll is False
            scroll.scroll_to.assert_not_called()
            # ...but the highlight itself still moves.
            assert sidebar.current_line_index == 1
            assert sidebar._lyric_widgets[1].has_class("lyrics-current")

    async def test_lapsed_pause_resumes_the_follow_mode(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            import time as _time

            from textual.containers import VerticalScroll

            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(0.0, "a"), (5.0, "b"), (10.0, "c")]
            sidebar._synced_timestamps = [0.0, 5.0, 10.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            scroll = sidebar.query_one("#ls-scroll", VerticalScroll)
            scroll.scroll_to = MagicMock()

            # The manual scroll happened long ago: the next highlight
            # resumes the follow mode and scrolls.
            sidebar._auto_scroll = False
            sidebar._last_manual_scroll = _time.monotonic() - 60.0
            sidebar._on_position_change(6.0)

            assert sidebar._auto_scroll is True
            scroll.scroll_to.assert_called_once()

    async def test_rebuild_resets_the_highlight(self):
        host_cls = _mounted_sidebar()
        app = host_cls()
        async with app.run_test(size=(60, 20)):
            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(0.0, "a"), (5.0, "b"), (10.0, "c")]
            sidebar._synced_timestamps = [0.0, 5.0, 10.0]
            sidebar._is_synced = True
            sidebar._build_synced_view()

            sidebar._on_position_change(11.0)
            assert sidebar.current_line_index == 2

            # A rebuild (new track's lyrics) resets to a fresh view:
            # every mounted row upcoming, index back to -1.
            sidebar._build_synced_view()
            assert sidebar.current_line_index == -1
            assert all(w.has_class("lyrics-upcoming") for w in sidebar._lyric_widgets)


# ── Native / fallback parity ────────────────────────────────────────
#
# When the Rust extension is built, the engines above ARE the native
# ones; each case below is also run in a child interpreter with the
# extension import blocked (pure-Python reference), and the two answers
# must agree exactly — indices, class pairs, verdicts and targets,
# error shapes included.

# (timestamps, position)
_PARITY_LOOKUPS = [
    ([], 5.0),
    ([], 0.0),
    ([1.0, 2.0, 3.0], 0.0),
    ([1.0, 2.0, 3.0], 1.0),
    ([1.0, 2.0, 3.0], 1.5),
    ([1.0, 2.0, 3.0], 2.999),
    ([1.0, 2.0, 3.0], 3.0),
    ([1.0, 2.0, 3.0], 100.0),
    ([0.0], 0.0),
    ([0.0, 0.0, 1.0], 0.0),
    ([0, 5, 10], 6),
    ([0.5, 1.5, 2.5, 3.5], 2.5),
    ([10.0, 20.0, 30.0], float("inf")),
    ([10.0, 20.0, 30.0], float("-inf")),
    ([1.0, 2.0], float("nan")),
    (["a", "b"], 1.0),
]

# (old_index, new_index, count)
_PARITY_HIGHLIGHTS = [
    (-1, 0, 5),
    (0, 3, 5),
    (3, 0, 5),
    (3, 3, 5),
    (5, 2, 5),
    (7, 8, 5),
    (-1, 7, 5),
    (0, -1, 5),
    (4, -1, 3),
    (1, 6, 5),
    (0, 1, 0),
    (-2, 1, 3),
    (0, 1, 1),
    (2, 0, 3),
    (12, 13, 40),
]

# (auto_scroll, last_manual_scroll, now, resume_delay)
_PARITY_FOLLOWS = [
    (True, 0.0, 5.0, 3.0),
    (True, 100.0, 5.0, 3.0),
    (False, 4.0, 5.0, 3.0),
    (False, 1.0, 5.0, 3.0),
    (False, 2.0, 5.0, 3.0),
    (False, 0.0, 0.0, 3.0),
    (False, 0.0, 3.5, 3.0),
    (False, -10.0, -5.0, 3.0),
    (False, float("nan"), 5.0, 3.0),
    (False, 0.0, float("nan"), 3.0),
]

# (widget_y, widget_h, viewport_h)
_PARITY_TARGETS = [
    (0, 1, 10),
    (17, 2, 12),
    (5, 3, 4),
    (0, 2, 3),
    (1, -3, 5),
    (10, 3, 5),
    (100, 0, 0),
    (0, 0, 0),
    (7, 7, 7),
    (2, 5, 9),
    (-4, 3, 7),
]

_CHILD_PROG = """
import json, sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.sidebars import lyrics_sidebar

payload = json.load(sys.stdin)

lookups = []
for timestamps, position in payload["lookups"]:
    try:
        lookups.append({"ok": lyrics_sidebar._line_for_position(timestamps, position)})
    except BaseException as exc:
        lookups.append({"error": f"{type(exc).__name__}: {exc}"})

highlights = []
for old, new, count in payload["highlights"]:
    try:
        highlights.append(
            {"ok": [list(pair) for pair in lyrics_sidebar._highlight_changes(old, new, count)]}
        )
    except BaseException as exc:
        highlights.append({"error": f"{type(exc).__name__}: {exc}"})

follows = [lyrics_sidebar._scroll_follows(*case) for case in payload["follows"]]
targets = [lyrics_sidebar._scroll_target(*case) for case in payload["targets"]]

json.dump(
    {
        "lookups": lookups,
        "highlights": highlights,
        "follows": follows,
        "targets": targets,
    },
    sys.stdout,
)
"""


class TestNativeParity:
    """When the Rust extension is built, the native scroll engines must
    decide exactly like the pure-Python references — same lookups, same
    class pairs, same verdicts, same targets, same error shapes — for
    every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native_lookups = []
        for timestamps, position in _PARITY_LOOKUPS:
            try:
                native_lookups.append({"ok": _line_for_position(timestamps, position)})
            except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
                native_lookups.append({"error": f"{type(exc).__name__}: {exc}"})

        native_highlights = []
        for old, new, count in _PARITY_HIGHLIGHTS:
            try:
                native_highlights.append(
                    {"ok": [list(pair) for pair in _highlight_changes(old, new, count)]}
                )
            except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
                native_highlights.append({"error": f"{type(exc).__name__}: {exc}"})

        native_follows = [_scroll_follows(*case) for case in _PARITY_FOLLOWS]
        native_targets = [_scroll_target(*case) for case in _PARITY_TARGETS]

        payload = json.dumps(
            {
                "lookups": _PARITY_LOOKUPS,
                "highlights": _PARITY_HIGHLIGHTS,
                "follows": _PARITY_FOLLOWS,
                "targets": _PARITY_TARGETS,
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        fallback = json.loads(result.stdout)

        for case, native, ref in zip(_PARITY_LOOKUPS, native_lookups, fallback["lookups"]):
            assert native == ref, (
                f"native and fallback line lookups differ for {case!r}:\n{native!r}\n{ref!r}"
            )
        for case, native, ref in zip(_PARITY_HIGHLIGHTS, native_highlights, fallback["highlights"]):
            assert native == ref, (
                f"native and fallback highlight changes differ for {case!r}:\n{native!r}\n{ref!r}"
            )
        for case, native, ref in zip(_PARITY_FOLLOWS, native_follows, fallback["follows"]):
            assert native == ref, (
                f"native and fallback follow verdicts differ for {case!r}: {native!r} != {ref!r}"
            )
        for case, native, ref in zip(_PARITY_TARGETS, native_targets, fallback["targets"]):
            assert native == ref, (
                f"native and fallback scroll targets differ for {case!r}: {native!r} != {ref!r}"
            )
