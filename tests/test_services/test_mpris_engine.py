"""Tests for the MPRIS volume engines (the clamp and the D-Bus-echo gate).

``_clamp_volume`` and ``_volume_is_echo`` (services/mpris.py) run with
two interchangeable backends: the pure-Python transcriptions in the
module and, when the Rust extension is built, the native
implementations bound at the module's tail. The engines: the D-Bus
``Volume`` property's ``[0, 1]`` clamp behind ``_PlayerInterface``'s
setter, its ``set_volume`` helper and ``MPRISService.update_volume``'s
push, and the ``abs(volume - current) < 1e-6`` echo skip behind
``update_volume`` (re-emitting a D-Bus-initiated set would ping-pong
with the client).

The unit tests below assert each engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest

from ytm_player.services.mpris import _clamp_volume, _volume_is_echo

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestClampVolume:
    """The ``[0, 1]`` domain write (whichever backend)."""

    def test_the_domain_bounds_pass_through(self):
        assert _clamp_volume(0.0) == 0.0
        assert _clamp_volume(1.0) == 1.0

    def test_an_in_domain_volume_passes_through_unchanged(self):
        for volume in (0.007, 0.25, 0.5, 0.731, 0.999):
            assert _clamp_volume(volume) == volume

    def test_above_the_domain_clamps_to_one(self):
        assert _clamp_volume(1.5) == 1.0
        assert _clamp_volume(100.0) == 1.0
        assert _clamp_volume(float("inf")) == 1.0

    def test_below_the_domain_clamps_to_zero(self):
        assert _clamp_volume(-0.5) == 0.0
        assert _clamp_volume(-100.0) == 0.0
        assert _clamp_volume(float("-inf")) == 0.0

    def test_nan_reads_one(self):
        """``min(1.0, nan)`` answers 1.0 — CPython keeps the running
        first argument when the comparison is false, so a NaN volume
        carries through the clamp as 1.0 on both backends."""
        assert _clamp_volume(float("nan")) == 1.0


class TestVolumeIsEcho:
    """The echo skip's verdict (whichever backend)."""

    def test_equal_values_are_the_echo(self):
        assert _volume_is_echo(0.5, 0.5) is True
        assert _volume_is_echo(0.0, 0.0) is True
        assert _volume_is_echo(1.0, 1.0) is True

    def test_a_small_distance_is_the_echo(self):
        assert _volume_is_echo(0.5, 0.5 + 0.9e-6) is True
        assert _volume_is_echo(math.nextafter(1e-6, 0.0), 0.0) is True

    def test_exactly_the_threshold_is_not_the_echo(self):
        """The comparison is CPython's own strict ``<`` — a distance of
        exactly 1e-6 does not skip the push."""
        assert _volume_is_echo(1e-6, 0.0) is False

    def test_a_real_change_is_not_the_echo(self):
        assert _volume_is_echo(0.4, 0.5) is False
        assert _volume_is_echo(0.0, 1.0) is False
        assert _volume_is_echo(0.5 + 2e-6, 0.5) is False

    def test_a_nan_distance_is_not_the_echo(self):
        """``abs(nan - x) < 1e-6`` compares false — a NaN never skips."""
        assert _volume_is_echo(float("nan"), 0.5) is False


# ── Native vs fallback parity ─────────────────────────────────────────

_CLAMP_CASES = [
    0.0,
    1.0,
    0.007,
    0.25,
    0.5,
    0.731,
    0.999,
    1.5,
    100.0,
    -0.5,
    -100.0,
    float("inf"),
    float("-inf"),
    float("nan"),
]

_ECHO_CASES = [
    [0.5, 0.5],
    [0.0, 0.0],
    [1.0, 1.0],
    [0.5, 0.5 + 0.9e-6],
    [1e-6, 0.0],
    [math.nextafter(1e-6, 0.0), 0.0],
    [0.4, 0.5],
    [0.0, 1.0],
    [0.5 + 2e-6, 0.5],
    [float("nan"), 0.5],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.mpris import _clamp_volume, _volume_is_echo

payload = json.load(sys.stdin)
out = {
    "clamps": [_clamp_volume(case) for case in payload["clamp_cases"]],
    "echoes": [_volume_is_echo(volume, current) for volume, current in payload["echo_cases"]],
}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native verdicts must equal
    the pure-Python references' for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self):
        payload = json.dumps({"clamp_cases": _CLAMP_CASES, "echo_cases": _ECHO_CASES})
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_clamps = [_clamp_volume(case) for case in _CLAMP_CASES]
        native_echoes = [_volume_is_echo(volume, current) for volume, current in _ECHO_CASES]
        assert native_clamps == fallback["clamps"]
        assert native_echoes == fallback["echoes"]
