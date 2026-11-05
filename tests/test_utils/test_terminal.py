"""Tests for the terminal orientation gate.

``get_orientation`` (utils/terminal.py) runs with two interchangeable
backends: the pure-Python transcription in the module and, when the
Rust extension is built, the native implementation bound at the
module's tail. The engine is one expression — ``"horizontal"`` when
*rows* is zero or ``cols / rows`` exceeds 2.3, ``"vertical"``
otherwise — with every operator the interpreter's own on the native
side, so arbitrary-precision ints, exotic numeric shapes and the
TypeError of a non-numeric pair all match the reference exactly.

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case,
error shapes included, so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.utils.terminal import get_orientation

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestGetOrientation:
    """The aspect-ratio verdict (whichever backend)."""

    def test_zero_rows_is_horizontal(self):
        assert get_orientation(80, 0) == "horizontal"
        assert get_orientation(0, 0) == "horizontal"

    def test_wide_terminals_are_horizontal(self):
        assert get_orientation(80, 24) == "horizontal"
        assert get_orientation(231, 100) == "horizontal"
        assert get_orientation(1000, 24) == "horizontal"

    def test_narrow_terminals_are_vertical(self):
        assert get_orientation(229, 100) == "vertical"
        assert get_orientation(40, 80) == "vertical"
        assert get_orientation(20, 100) == "vertical"

    def test_the_boundary_ratio_is_exactly_2_3(self):
        # 230 / 100 == 2.3 exactly (both the quotient and the literal
        # land on the same double) — not strictly greater, so vertical.
        assert 230 / 100 == 2.3
        assert get_orientation(230, 100) == "vertical"

    def test_negative_columns_stay_vertical(self):
        assert get_orientation(-80, 24) == "vertical"

    def test_arbitrary_precision_ints_divide_like_the_reference(self):
        # A ratio far past f64's exact-integer range still divides
        # correctly-rounded on both backends.
        assert get_orientation(10**30, 3) == "horizontal"
        assert get_orientation(-(10**30), 3) == "vertical"

    def test_floats_and_bools_flow_through_the_operators(self):
        assert get_orientation(230.0, 100) == "vertical"
        assert get_orientation(True, True) == "vertical"  # 1 / 1 = 1.0

    def test_non_numeric_pair_raises_the_division_error(self):
        with pytest.raises(TypeError, match="unsupported operand"):
            get_orientation(80, "24")


# Each parity case: [cols, rows] as JSON-round-trippable values.
_PARITY_CASES = [
    [80, 24],
    [230, 100],
    [231, 100],
    [229, 100],
    [80, 0],
    [0, 0],
    [0, 24],
    [-80, 24],
    [80, -24],
    [-80, -24],
    [10**30, 3],
    [-(10**30), 3],
    [3, 10**30],
    [230.0, 100],
    [229.9999999999, 100],
    [80.5, 35],
    [True, True],
    [0, False],
    [80, "24"],
    ["80", 24],
    [80, None],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.terminal import get_orientation


def outcome(cols, rows):
    try:
        return {"ok": get_orientation(cols, rows)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
out = [outcome(*case) for case in payload]
json.dump(out, sys.stdout)
"""


def _native_outcome(cols, rows):
    try:
        return {"ok": get_orientation(cols, rows)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same verdicts, same errors
    with the same messages — for every case, so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        payload = json.dumps(_PARITY_CASES)
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        assert len(fallback) == len(_PARITY_CASES)
        for case, expected in zip(_PARITY_CASES, fallback):
            native = _native_outcome(*case)
            assert native == expected, (
                f"native and fallback get_orientation differ for {case!r}:\n"
                f"native:   {native!r}\n"
                f"fallback: {expected!r}"
            )
