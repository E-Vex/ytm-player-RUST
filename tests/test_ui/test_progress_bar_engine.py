"""Tests for the playback progress-bar render-geometry engines.

The engines behind ``PlaybackProgress``: the progress fraction
(``_progress_fraction``), the time-label/bar-width metrics
(``_bar_metrics``), the bar cell computation with the scroll-seek
marker (``_bar_cells``) and the click-to-seek column mapping
(``_x_to_seconds``). The pure-Python references live in
``ui/widgets/progress_bar.py``; when the Rust extension is built the
same engines run natively (``_ytm_player_native.progress_fraction`` &
co).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.widgets.progress_bar import (
    _bar_cells,
    _bar_metrics,
    _progress_fraction,
    _x_to_seconds,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── _progress_fraction ─────────────────────────────────────────────


class TestProgressFraction:
    def test_zero_duration_is_zero(self):
        assert _progress_fraction(10.0, 0.0) == 0.0

    def test_negative_duration_is_zero(self):
        assert _progress_fraction(10.0, -5.0) == 0.0

    def test_midpoint(self):
        assert _progress_fraction(30.0, 60.0) == 0.5

    def test_position_beyond_duration_clamps_to_one(self):
        assert _progress_fraction(150.0, 100.0) == 1.0

    def test_negative_position_clamps_to_zero(self):
        assert _progress_fraction(-25.0, 100.0) == 0.0

    def test_exact_end_is_one(self):
        assert _progress_fraction(100.0, 100.0) == 1.0


# ── _bar_metrics ───────────────────────────────────────────────────


class TestBarMetrics:
    def test_minutes_label_pair(self):
        prefix, suffix, bar_width = _bar_metrics(30.0, 296.0, None, 40)
        assert prefix == " 0:30 "
        assert suffix == " 4:56 "
        # 6 + 6 label columns reserved from the widget width.
        assert bar_width == 28

    def test_hours_label_is_longer(self):
        prefix, suffix, bar_width = _bar_metrics(1.5, 7325.0, None, 40)
        assert prefix == " 0:01 "
        assert suffix == " 2:02:05 "
        assert bar_width == 40 - 6 - 9

    def test_preview_position_stands_in_for_elapsed(self):
        prefix, _, _ = _bar_metrics(30.0, 296.0, 200.0, 40)
        assert prefix == " 3:20 "

    def test_width_narrower_than_labels_floors_at_zero(self):
        _, _, bar_width = _bar_metrics(30.0, 296.0, None, 12)
        assert bar_width == 0

    def test_width_exactly_labels_is_zero_bar(self):
        _, _, bar_width = _bar_metrics(30.0, 296.0, None, 13)
        assert bar_width == 1

    def test_zero_width(self):
        _, _, bar_width = _bar_metrics(30.0, 296.0, None, 0)
        assert bar_width == 0


# ── _bar_cells ─────────────────────────────────────────────────────


class TestBarCells:
    def test_plain_half_bar(self):
        filled, empty, marker_col = _bar_cells(30, 0.5, None, 60.0)
        assert (filled, empty, marker_col) == (15, 15, -1)

    def test_marker_at_three_quarters(self):
        _, _, marker_col = _bar_cells(30, 0.5, 45.0, 60.0)
        assert marker_col == 22

    def test_marker_clamped_to_last_column(self):
        # Preview at (past) the end still lands on the final column.
        _, _, marker_col = _bar_cells(30, 0.5, 90.0, 60.0)
        assert marker_col == 29

    def test_marker_at_start(self):
        _, _, marker_col = _bar_cells(30, 0.5, 0.0, 60.0)
        assert marker_col == 0

    def test_marker_without_bar_width_is_absent(self):
        _, _, marker_col = _bar_cells(0, 0.5, 45.0, 60.0)
        assert marker_col == -1

    def test_marker_with_zero_duration_lands_at_zero(self):
        # duration <= 0 → preview_frac 0.0 → the marker pins to column 0.
        _, _, marker_col = _bar_cells(30, 0.5, 45.0, 0.0)
        assert marker_col == 0

    def test_filled_truncates_toward_zero(self):
        # int(10 * 0.29) == 2, not 3.
        filled, empty, _ = _bar_cells(10, 0.29, None, 100.0)
        assert (filled, empty) == (2, 8)

    def test_full_bar(self):
        filled, empty, _ = _bar_cells(10, 1.0, None, 5.0)
        assert (filled, empty) == (10, 0)

    def test_empty_bar(self):
        filled, empty, _ = _bar_cells(10, 0.0, None, 5.0)
        assert (filled, empty) == (0, 10)


# ── _x_to_seconds ──────────────────────────────────────────────────


class TestXToSeconds:
    def test_no_duration_is_no_seek(self):
        assert _x_to_seconds(10, 30.0, 0.0, None, 40) is None

    def test_no_bar_width_is_no_seek(self):
        assert _x_to_seconds(5, 30.0, 60.0, None, 12) is None

    def test_click_on_prefix_clamps_to_start(self):
        assert _x_to_seconds(0, 30.0, 60.0, None, 40) == 0.0

    def test_first_bar_column(self):
        # Prefix " 0:30 " eats 6 columns; bar is 28 wide.
        assert _x_to_seconds(6, 30.0, 60.0, None, 40) == 0.0

    def test_click_beyond_bar_clamps_to_last_column(self):
        # 45 is past the widget: bar_x clamps to 27 → 27/28 of a minute.
        assert _x_to_seconds(45, 30.0, 60.0, None, 40) == pytest.approx(27 / 28 * 60.0)

    def test_mid_bar_click(self):
        # bar_x = 20 - 6 = 14 → half the bar.
        assert _x_to_seconds(20, 30.0, 60.0, None, 40) == pytest.approx(0.5 * 60.0)

    def test_preview_widens_the_elapsed_prefix(self):
        # Preview 200s renders " 3:20 " — still 6 columns, so the same
        # click maps through the same bar geometry.
        assert _x_to_seconds(10, 30.0, 296.0, 200.0, 40) == pytest.approx(4 / 28 * 296.0)

    def test_negative_x_clamps_to_start(self):
        assert _x_to_seconds(-5, 30.0, 60.0, None, 40) == 0.0


# ── Native / fallback parity ───────────────────────────────────────

_PROGRESS_CASES = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, -5.0),
    (0.0, 100.0),
    (50.0, 100.0),
    (100.0, 100.0),
    (150.0, 100.0),
    (-25.0, 100.0),
    (1.0, 3.0),
    (30.0, 296.0),
    (123.456, 456.789),
    (3600.0, 7200.0),
]

_METRICS_CASES = [
    (30.0, 296.0, None, 40),
    (30.0, 296.0, 61.0, 40),
    (0.0, 0.0, None, 20),
    (1.5, 7325.0, None, 40),
    (30.0, 296.0, None, 12),
    (30.0, 296.0, None, 13),
    (30.0, 296.0, None, 14),
    (-3.7, 296.0, None, 40),
    (30.0, 296.0, 999.0, 40),
    (0.0, 0.0, 5.0, 30),
    (17.0, 60.0, None, 0),
]

_CELLS_CASES = [
    (30, 0.5, None, 60.0),
    (30, 0.5, 45.0, 60.0),
    (30, 0.5, 60.0, 60.0),
    (30, 0.5, 90.0, 60.0),
    (30, 0.5, 0.0, 60.0),
    (30, 0.5, 45.0, 0.0),
    (0, 0.5, 45.0, 60.0),
    (30, 0.29, None, 100.0),
    (30, 0.0, None, 100.0),
    (30, 1.0, None, 100.0),
    (7, 1 / 3, 10.0, 30.0),
    (10, 0.999, None, 5.0),
    (30, 0.5, 30.0, 60.0),
    (25, 0.479999, 12.5, 25.25),
]

_X_CASES = [
    (0, 30.0, 60.0, None, 40),
    (5, 30.0, 60.0, None, 40),
    (6, 30.0, 60.0, None, 40),
    (20, 30.0, 60.0, None, 40),
    (39, 30.0, 60.0, None, 40),
    (45, 30.0, 60.0, None, 40),
    (0, 30.0, 0.0, None, 40),
    (5, 30.0, 60.0, None, 12),
    (10, 30.0, 296.0, 200.0, 40),
    (3, 0.0, 3600.0, None, 40),
    (-4, 12.5, 90.5, None, 33),
]

_CASES = {
    "fraction": _PROGRESS_CASES,
    "metrics": _METRICS_CASES,
    "cells": _CELLS_CASES,
    "x": _X_CASES,
}

# The child interpreter runs this program with the case list on stdin and
# prints the reference engines' results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets.progress_bar import (
    _bar_cells,
    _bar_metrics,
    _progress_fraction,
    _x_to_seconds,
)

cases = json.load(sys.stdin)

out = {
    "fraction": [_progress_fraction(p, d) for p, d in cases["fraction"]],
    "metrics": [_bar_metrics(p, d, pv, w) for p, d, pv, w in cases["metrics"]],
    "cells": [_bar_cells(bw, pr, pv, d) for bw, pr, pv, d in cases["cells"]],
    "x": [_x_to_seconds(x, p, d, pv, w) for x, p, d, pv, w in cases["x"]],
}
json.dump(out, sys.stdout)
"""


def _native_results(cases: dict) -> dict:
    """Run whichever engines the module loaded (native when built) over *cases*."""
    return {
        "fraction": [_progress_fraction(p, d) for p, d in cases["fraction"]],
        "metrics": [_bar_metrics(p, d, pv, w) for p, d, pv, w in cases["metrics"]],
        "cells": [_bar_cells(bw, pr, pv, d) for bw, pr, pv, d in cases["cells"]],
        "x": [_x_to_seconds(x, p, d, pv, w) for x, p, d, pv, w in cases["x"]],
    }


def _fallback_results(cases: dict) -> dict:
    """Run the pure-Python reference engines over *cases* in a child
    interpreter with the native extension import blocked."""
    payload = json.dumps(cases)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native render-geometry
    engines must produce exactly the same fractions, metrics, cell
    counts and seek targets as the pure-Python references for every
    case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = json.loads(json.dumps(_native_results(_CASES)))
        fallback = _fallback_results(_CASES)
        for group, cases in _CASES.items():
            for i in range(len(cases)):
                assert native[group][i] == fallback[group][i], (
                    f"native and fallback {group} results differ for case {i}: {cases[i]}"
                )
