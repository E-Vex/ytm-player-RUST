"""Tests for the selection info bar's truncation-width engine.

``_visible_width`` (ui/selection_info_bar.py) runs with two
interchangeable backends: the pure-Python transcription in the module
and, when the Rust extension is built, the native implementation bound
at the module's tail. The engine: the character budget the bar's text
is truncated to — ``80`` while the widget has no width yet (not laid
out), otherwise the widget width minus the bar's 1-column padding on
each side, floored at ``10``.

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from textual.geometry import Size

from ytm_player.ui.selection_info_bar import SelectionInfoBar, _visible_width
from ytm_player.utils.formatting import truncate

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestVisibleWidth:
    """The width budget (whichever backend)."""

    def test_zero_width_reads_the_default(self):
        """A widget not yet laid out has no width — the 80-column default."""
        assert _visible_width(0) == 80

    def test_the_padding_is_reserved(self):
        assert _visible_width(100) == 98
        assert _visible_width(80) == 78
        assert _visible_width(13) == 11

    def test_the_floor_is_ten_columns(self):
        """Widths whose padding reservation would fall below 10 stay at 10."""
        assert _visible_width(12) == 10
        assert _visible_width(2) == 10
        assert _visible_width(1) == 10

    def test_a_negative_width_keeps_the_floor(self):
        """A negative width is truthy — the max keeps the 10 floor."""
        assert _visible_width(-5) == 10
        assert _visible_width(-100) == 10

    def test_wide_widths_pass_through(self):
        assert _visible_width(10_000) == 9_998


class _SizedBar(SelectionInfoBar):
    """A SelectionInfoBar pinned to a fixed size for watcher tests."""

    @property
    def size(self) -> Size:  # type: ignore[override]
        return Size(100, 1)


class _RecordingStatic:
    def __init__(self) -> None:
        self.updates: list[str] = []

    def update(self, content: str) -> None:
        self.updates.append(content)


class TestWatcherWiring:
    """The reactive watcher routes its budget into the truncation."""

    def test_text_changes_truncate_to_the_width_budget(self):
        bar = _SizedBar()
        inner = _RecordingStatic()
        bar._inner = inner  # type: ignore[assignment]
        long_text = "x" * 200
        bar.text = long_text
        # The reactive's init pass renders the empty default first; the
        # assignment's watcher renders the text under the same budget.
        assert inner.updates[-1] == truncate(long_text, 98)
        assert all(u == truncate(t, 98) for u, t in zip(inner.updates, ["", long_text]))

    def test_short_text_passes_through_the_same_budget(self):
        bar = _SizedBar()
        inner = _RecordingStatic()
        bar._inner = inner  # type: ignore[assignment]
        bar.text = "Hello"
        assert inner.updates[-1] == truncate("Hello", 98)
        assert all(u == truncate(t, 98) for u, t in zip(inner.updates, ["", "Hello"]))


# ── Native vs fallback parity ─────────────────────────────────────────

_WIDTH_CASES = [0, 1, 2, 3, 10, 11, 12, 13, 80, 100, 1000, -1, -5, -100, 10_000]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.selection_info_bar import _visible_width

payload = json.load(sys.stdin)
json.dump([_visible_width(case) for case in payload], sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native widths must equal
    the pure-Python reference's for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        payload = json.dumps(_WIDTH_CASES)
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_visible_width(case) for case in _WIDTH_CASES]
        assert native == fallback
