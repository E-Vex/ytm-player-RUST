"""Tests for the MPRIS callback unit-conversion engines.

``_us_to_seconds`` and ``_volume_percent`` (app/_mpris.py) run with two
interchangeable backends: the pure-Python transcriptions in the module
and, when the Rust extension is built, the native implementations bound
at the module's tail. The engines: the ``microseconds / 1_000_000``
true division behind both MPRIS time callbacks — ``_mpris_seek``'s
D-Bus ``Seek`` offset (which a backward-seeking client delivers
negative, unclamped) and ``_mpris_set_position``'s ``SetPosition``
position — and the ``round(volume * 100)`` fraction→percent conversion
behind ``_mpris_set_volume`` (CPython's own single-argument round:
ties go to the even neighbour, not half-away-from-zero).

The unit tests below assert each engine's contract on whichever backend
is loaded, and the mixin tests pin the callback methods' routing of
their arguments through the engines. When the extension is built, the
parity test additionally compares the native engines against the
pure-Python references (run in a child interpreter with the extension
import blocked) case for case, so the two implementations cannot drift
apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.app._mpris import MPRISMixin, _us_to_seconds, _volume_percent

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestUsToSeconds:
    """The microseconds→seconds true division (whichever backend)."""

    def test_whole_seconds(self):
        assert _us_to_seconds(0) == 0.0
        assert _us_to_seconds(1_000_000) == 1.0
        assert _us_to_seconds(60_000_000) == 60.0

    def test_fractional_seconds(self):
        assert _us_to_seconds(250_000) == 0.25
        assert _us_to_seconds(1_500_000) == 1.5
        assert _us_to_seconds(123) == 0.000123

    def test_a_backward_seek_stays_negative(self):
        """A D-Bus client's backward seek arrives as a negative offset
        and must reach the player as negative seconds — the reference
        divides without clamping."""
        assert _us_to_seconds(-2_000_000) == -2.0
        assert _us_to_seconds(-1) == -0.000001


class TestVolumePercent:
    """The fraction→percent round (whichever backend)."""

    def test_the_domain_ends(self):
        assert _volume_percent(0.0) == 0
        assert _volume_percent(1.0) == 100

    def test_whole_percents(self):
        assert _volume_percent(0.5) == 50
        assert _volume_percent(0.12) == 12
        assert _volume_percent(0.99) == 99

    def test_ties_go_to_the_even_neighbour(self):
        """CPython's single-argument round is ties-to-even, not
        half-away-from-zero: 0.125 and 0.375 scale to exactly 12.5 and
        37.5, which answer 12 and 38 — the 12.5 case is the
        half-away-from-zero discriminator (it would read 13)."""
        assert _volume_percent(0.125) == 12
        assert _volume_percent(0.375) == 38
        assert _volume_percent(-0.125) == -12

    def test_nan_answers_cpythons_valueerror(self):
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            _volume_percent(float("nan"))

    def test_infinity_answers_cpythons_overflowerror(self):
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            _volume_percent(float("inf"))
        with pytest.raises(OverflowError, match="cannot convert float infinity to integer"):
            _volume_percent(float("-inf"))


class _FakePlayer:
    """A player stand-in recording the transport calls."""

    def __init__(self) -> None:
        self.seeks: list[float] = []
        self.absolute_seeks: list[float] = []
        self.volumes: list[int] = []

    async def seek(self, offset: float) -> None:
        self.seeks.append(offset)

    async def seek_absolute(self, position: float) -> None:
        self.absolute_seeks.append(position)

    async def set_volume(self, level: int) -> None:
        self.volumes.append(level)


class _Host(MPRISMixin):
    """A minimal host exposing just the mixin's player attribute."""

    def __init__(self, player: _FakePlayer) -> None:
        self.player = player


class TestMixinRouting:
    """The callback methods route their conversions through the
    engines, with the player guard intact."""

    async def test_seek_divides_the_offset(self):
        player = _FakePlayer()
        host = _Host(player)
        await host._mpris_seek(2_500_000)
        await host._mpris_seek(-1_000_000)
        assert player.seeks == [2.5, -1.0]

    async def test_set_position_divides_the_position(self):
        player = _FakePlayer()
        host = _Host(player)
        await host._mpris_set_position(90_000_000)
        assert player.absolute_seeks == [90.0]

    async def test_set_volume_rounds_the_fraction_to_a_percent(self):
        player = _FakePlayer()
        host = _Host(player)
        await host._mpris_set_volume(0.5)
        await host._mpris_set_volume(0.125)
        assert player.volumes == [50, 12]

    async def test_no_player_is_a_no_op(self):
        host = _Host(None)
        await host._mpris_seek(1_000_000)
        await host._mpris_set_position(1_000_000)
        await host._mpris_set_volume(0.5)


# ── Native vs fallback parity ─────────────────────────────────────────

_US_CASES = [
    0,
    1,
    123,
    250_000,
    1_000_000,
    1_500_000,
    2_500_000,
    60_000_000,
    -1,
    -2_000_000,
    1_000_000_000_000,
]

_PERCENT_CASES = [
    0.0,
    1.0,
    0.5,
    0.12,
    0.99,
    0.125,
    0.375,
    -0.125,
    0.475,
    0.625,
    0.875,
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._mpris import _us_to_seconds, _volume_percent

payload = json.load(sys.stdin)
seconds = [_us_to_seconds(case) for case in payload["us_cases"]]
percents = []
errors = []
for case in payload["percent_cases"]:
    try:
        percents.append([_volume_percent(case), None])
    except (ValueError, OverflowError) as exc:
        percents.append([None, f"{type(exc).__name__}: {exc}"])
out = {"seconds": seconds, "percents": percents}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native verdicts must equal
    the pure-Python references' for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self):
        payload = json.dumps({"us_cases": _US_CASES, "percent_cases": _PERCENT_CASES})
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        native_seconds = [_us_to_seconds(case) for case in _US_CASES]
        assert native_seconds == fallback["seconds"]

        native_percents: list[list[object]] = []
        for case in _PERCENT_CASES:
            try:
                native_percents.append([_volume_percent(case), None])
            except (ValueError, OverflowError) as exc:
                native_percents.append([None, f"{type(exc).__name__}: {exc}"])
        assert native_percents == fallback["percents"]
