"""Tests for the Last.fm scrobble-threshold engine.

The pure core behind ``services/lastfm.py``'s scrobbling decision:
``_scrobble_due``, the gate ``check_scrobble``'s periodic poll
consults — the Last.fm spec's threshold walk (a track with no
duration info is due at the 4-minute cap; otherwise 50% of the
duration, capped at 4 minutes) and the ``position >= threshold``
verdict, the boundary included. The engine runs with two
interchangeable backends: the pure-Python transcription in the
module and, when the Rust extension is built, the native
implementation bound at the module's tail. Both constants are read
through the module's own ``_SCROBBLE_PERCENT`` /
``_SCROBBLE_MAX_SECONDS`` names at call time, so patched values steer
both backends identically — which the tests below use for the
patched-cap arms and the parity class pins as the contract.

The unit tests below assert the engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engine against the pure-Python
reference (run in a child interpreter with the extension import
blocked) case for case — error shapes and the min-tie threshold
spellings included, via position stand-ins whose ``__ge__`` answers
depend on the threshold's very type — so the two implementations
cannot drift apart. The wiring tests pin ``check_scrobble``'s
routing: the early-out state gates never consult the engine, the
verdict steers the scrobble submission, and the real engine drives
the whole now-playing → check → scrobble flow end to end.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.services import lastfm as lastfm_module
from ytm_player.services.lastfm import LastFMService, _scrobble_due

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class _FloatOnlyGe:
    """A position stand-in whose ``>=`` answers True only against a
    float — pins that the threshold it was compared against is
    ``min``'s first argument (the float product), never the int cap."""

    def __ge__(self, other):
        return isinstance(other, float)


class _IntOnlyGe:
    """A position stand-in whose ``>=`` answers True only against an
    int — the mirror pin: the tie must answer the product, so a
    patched int product beats a float cap."""

    def __ge__(self, other):
        return isinstance(other, int)


# ── The threshold arms ────────────────────────────────────────────────────


class TestNoDurationArm:
    """``duration <= 0`` answers the bare 4-minute cap."""

    def test_boundary_is_due(self):
        assert _scrobble_due(240, 0) is True

    def test_just_under_the_cap_is_not_due(self):
        assert _scrobble_due(239.99, 0) is False

    def test_negative_duration_takes_the_same_arm(self):
        assert _scrobble_due(239.99, -30) is False
        assert _scrobble_due(240, -30) is True

    def test_false_duration_is_the_no_duration_arm(self):
        # False <= 0 is True — the bool spelling stays CPython's.
        assert _scrobble_due(239.99, False) is False
        assert _scrobble_due(240, False) is True


class TestHalfArm:
    """A duration under the cap halves: threshold = duration * 50%."""

    def test_boundary_is_due(self):
        assert _scrobble_due(60, 120) is True

    def test_just_under_half_is_not_due(self):
        assert _scrobble_due(59.99, 120) is False

    def test_a_float_duration_spells_the_same_threshold(self):
        # 120.0 * 0.5 == 60.0 — the 3 == 3.0 spelling of the product.
        assert _scrobble_due(60, 120.0) is True
        assert _scrobble_due(59.99, 120.0) is False

    def test_true_duration_takes_the_half_arm(self):
        # True <= 0 is False, True * 0.5 == 0.5.
        assert _scrobble_due(0.5, True) is True
        assert _scrobble_due(0.49, True) is False


class TestCapArm:
    """A duration over 8 minutes caps the threshold at 4 minutes."""

    def test_boundary_is_due(self):
        assert _scrobble_due(240, 600) is True

    def test_just_under_the_cap_is_not_due(self):
        assert _scrobble_due(239.99, 600) is False

    def test_the_exact_eight_minute_tie_answers_the_product(self):
        # min(240.0, 240) — CPython's min walk keeps the running
        # minimum (the product) when the cap does not compare strictly
        # less, so the tie's threshold is the float 240.0. A position
        # stand-in that answers >= only against floats pins which
        # object the verdict ran against.
        assert _scrobble_due(_FloatOnlyGe(), 480) is True
        # And the mirror: the int-only stand-in must NOT be due
        # against the float product.
        assert _scrobble_due(_IntOnlyGe(), 480) is False


class TestPatchedConstants:
    """Both constants resolve through the module's own names at call
    time, so patched values steer whichever backend is loaded."""

    def test_patched_cap(self, monkeypatch):
        monkeypatch.setattr(lastfm_module, "_SCROBBLE_MAX_SECONDS", 10)
        assert _scrobble_due(9.9, 100) is False
        assert _scrobble_due(10, 100) is True

    def test_patched_percent(self, monkeypatch):
        monkeypatch.setattr(lastfm_module, "_SCROBBLE_PERCENT", 1.0)
        # 6000 * 1.0 == 6000.0, well past the 240 cap.
        assert _scrobble_due(239.99, 6000) is False
        assert _scrobble_due(240, 6000) is True

    def test_patched_tie_answers_the_int_product(self, monkeypatch):
        # percent=1, cap=2.0, duration=2 → min(2, 2.0) — the tie
        # answers the product (the int 2), pinned by the int-only
        # stand-in; a float-cap winner would answer False.
        monkeypatch.setattr(lastfm_module, "_SCROBBLE_PERCENT", 1)
        monkeypatch.setattr(lastfm_module, "_SCROBBLE_MAX_SECONDS", 2.0)
        assert _scrobble_due(_IntOnlyGe(), 2) is True


class TestErrorShapes:
    """The comparisons raise CPython's own errors with its own
    messages — the engine never pre-validates."""

    def test_non_numeric_duration_raises_the_le_error(self):
        with pytest.raises(TypeError, match="'<=' not supported"):
            _scrobble_due(60, "120")

    def test_none_duration_raises_the_le_error(self):
        with pytest.raises(TypeError, match="'<=' not supported"):
            _scrobble_due(60, None)

    def test_non_numeric_position_raises_the_ge_error(self):
        with pytest.raises(TypeError, match="'>=' not supported"):
            _scrobble_due("60", 120)


# ── The check_scrobble wiring ─────────────────────────────────────────────


class _FakeNetwork:
    """A pylast stand-in recording the to_thread calls."""

    def __init__(self):
        self.now_playing_calls: list[dict] = []
        self.scrobble_calls: list[dict] = []

    def update_now_playing(self, **kwargs):
        self.now_playing_calls.append(kwargs)

    def scrobble(self, **kwargs):
        self.scrobble_calls.append(kwargs)


class TestCheckScrobbleWiring:
    """``check_scrobble`` routes through the engine name and keeps its
    own state gates, the duration read and the submission."""

    @pytest.fixture
    def connected_service(self):
        svc = LastFMService(api_key="key", api_secret="secret")
        svc._network = _FakeNetwork()
        svc._connected = True
        return svc

    async def test_the_real_engine_drives_the_whole_flow(self, connected_service):
        svc = connected_service
        await svc.now_playing("Title", "Artist", "Album", 120)

        await svc.check_scrobble(59.99)
        assert svc._scrobbled is False
        assert svc._network.scrobble_calls == []

        await svc.check_scrobble(60)
        assert svc._scrobbled is True
        assert svc._network.scrobble_calls == [
            {
                "artist": "Artist",
                "title": "Title",
                "timestamp": int(svc._track_start),
                "album": "Album",
                "duration": 120,
            }
        ]

    async def test_the_engine_is_consulted_with_position_and_duration(
        self, connected_service, monkeypatch
    ):
        calls: list[tuple[float, object]] = []

        def recording_engine(position, duration):
            calls.append((position, duration))
            return False

        monkeypatch.setattr(lastfm_module, "_scrobble_due", recording_engine)

        svc = connected_service
        await svc.now_playing("Title", "Artist", "Album", 200)
        await svc.check_scrobble(120.5)

        assert calls == [(120.5, 200)]
        assert svc._scrobbled is False

    async def test_a_true_verdict_submits_the_scrobble(self, monkeypatch):
        submitted = []

        async def recording_scrobble():
            submitted.append(True)

        monkeypatch.setattr(lastfm_module, "_scrobble_due", lambda position, duration: True)

        svc = LastFMService(api_key="key", api_secret="secret")
        svc._network = _FakeNetwork()
        svc._connected = True
        svc._scrobble = recording_scrobble
        svc._current_track = {"title": "T", "artist": "A", "album": "", "duration": 10}

        await svc.check_scrobble(1)
        # The recorder stands in for _scrobble — the verdict reached it,
        # and only the real submission would flip _scrobbled.
        assert submitted == [True]

    async def test_the_state_gates_short_circuit_before_the_engine(
        self, connected_service, monkeypatch
    ):
        calls: list[tuple[object, object]] = []

        def recording_engine(position, duration):
            calls.append((position, duration))
            return True

        monkeypatch.setattr(lastfm_module, "_scrobble_due", recording_engine)

        svc = connected_service
        await svc.now_playing("Title", "Artist", "Album", 120)

        # Already scrobbled — the engine is never consulted.
        svc._scrobbled = True
        await svc.check_scrobble(10_000)
        assert calls == []

        # Disconnected — the engine is never consulted.
        svc._scrobbled = False
        svc._connected = False
        await svc.check_scrobble(10_000)
        assert calls == []

        # No current track — the engine is never consulted.
        svc._connected = True
        svc._current_track = None
        await svc.check_scrobble(10_000)
        assert calls == []


# ── Native parity ─────────────────────────────────────────────────────────

# Each case: [position, duration, percent, max_seconds] — the constants
# installed on the module before the call (None → the module defaults),
# the position either a JSON scalar or a stand-in descriptor. The
# outcome is the due verdict, or the error's own shape.
_PARITY_CASES = [
    [239.99, 0, None, None],
    [240, 0, None, None],
    [59.99, 120, None, None],
    [60, 120, None, None],
    [239.99, 600, None, None],
    [240, 600, None, None],
    [240, 480, None, None],
    [239.999999, 480, None, None],
    [0.5, True, None, None],
    [0.49, True, None, None],
    [240, False, None, None],
    [239.99, False, None, None],
    [60, 120.0, None, None],
    [9.9, 100, 1.0, 10],
    [10, 100, 1.0, 10],
    [239.99, 6000, 1.0, None],
    [240, 6000, 1.0, None],
    [{"float_only_ge": True}, 480, None, None],
    [{"int_only_ge": True}, 480, None, None],
    [{"int_only_ge": True}, 2, 1, 2.0],
    [60, "120", None, None],
    [60, None, None, None],
    ["60", 120, None, None],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import lastfm

_DEFAULT_PERCENT = lastfm._SCROBBLE_PERCENT
_DEFAULT_MAX = lastfm._SCROBBLE_MAX_SECONDS


class FloatOnlyGe:
    def __ge__(self, other):
        return isinstance(other, float)


class IntOnlyGe:
    def __ge__(self, other):
        return isinstance(other, int)


def make_position(desc):
    if isinstance(desc, dict):
        if desc.get("float_only_ge"):
            return FloatOnlyGe()
        return IntOnlyGe()
    return desc


def outcome(case):
    position, duration, percent, cap = case
    if percent is not None:
        lastfm._SCROBBLE_PERCENT = percent
        lastfm._SCROBBLE_MAX_SECONDS = cap
    try:
        result = lastfm._scrobble_due(make_position(position), duration)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        lastfm._SCROBBLE_PERCENT = _DEFAULT_PERCENT
        lastfm._SCROBBLE_MAX_SECONDS = _DEFAULT_MAX
    return {"ok": result}


payload = json.load(sys.stdin)
out = [outcome(case) for case in payload]
json.dump(out, sys.stdout)
"""


def _make_position(desc):
    """The position stand-in rebuilt from the JSON case — the same
    shape the child spells inside its own program."""
    if isinstance(desc, dict):
        if desc.get("float_only_ge"):
            return _FloatOnlyGe()
        return _IntOnlyGe()
    return desc


def _native_outcomes():
    """The parent-side outcomes — the constants installed per case and
    restored once the engine has run, so the module never leaks."""

    def outcome(case):
        position, duration, percent, cap = case
        saved = (lastfm_module._SCROBBLE_PERCENT, lastfm_module._SCROBBLE_MAX_SECONDS)
        if percent is not None:
            lastfm_module._SCROBBLE_PERCENT = percent
            lastfm_module._SCROBBLE_MAX_SECONDS = cap
        try:
            result = lastfm_module._scrobble_due(_make_position(position), duration)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            lastfm_module._SCROBBLE_PERCENT, lastfm_module._SCROBBLE_MAX_SECONDS = saved
        return {"ok": result}

    return [outcome(case) for case in _PARITY_CASES]


class TestNativeParity:
    """When the Rust extension is built, the native engine must answer
    exactly like the pure-Python reference — same verdicts, same
    errors with the same messages, the same tie threshold object — so
    the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_scrobble_gate_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes()
        for case, n, f in zip(_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback _scrobble_due differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
