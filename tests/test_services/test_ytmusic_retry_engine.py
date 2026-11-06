"""Tests for the YTMusic retry/renewal bookkeeping engines.

The pure cores behind the service's retry machinery in
``services/ytmusic.py``: the auth-expiry sniff ``_is_auth_expired``
(the gate every expired-session retry consults before renewing — the
``YTMusicServerError`` arm delegating to the mutation-failure
classifier, the ``KeyError`` arm's sign-in marker sniff over the
lowered message), the renewal-cooldown gate ``_renewal_cooldown_gate``
and stamp ``_renewal_cooldown_stamp`` (the remaining-cooldown
arithmetic and the failed-renewal stamp behind
``_renewed_client_for``), and the consecutive-failure step
``_api_failure_step`` (the counter bump and the reinit threshold
verdict behind ``_run``'s except arm).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked, both reading
the same frozen module clock) case for case, error shapes included,
so the two implementations cannot drift.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import requests.exceptions
from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError

from tests.conftest import make_ytmusic_service
from ytm_player.services import ytmusic as ytmusic_module
from ytm_player.services.ytmusic import (
    _api_failure_step,
    _is_auth_expired,
    _renewal_cooldown_gate,
    _renewal_cooldown_stamp,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_FROZEN_NOW = 1_000.0

EXPIRED = YTMusicServerError("Server returned HTTP 401: Unauthorized.\n")
FORBIDDEN = YTMusicServerError("Server returned HTTP 403: Forbidden.\n")
SERVER_ERROR = YTMusicServerError("Server returned HTTP 500: Internal Server Error.\n")
NO_STATUS = YTMusicServerError("something unexpected")


# ── _is_auth_expired ───────────────────────────────────────────────


class TestAuthExpired:
    def test_401_and_403_are_auth_expired(self):
        assert _is_auth_expired(EXPIRED) is True
        assert _is_auth_expired(FORBIDDEN) is True

    def test_other_server_errors_are_not(self):
        assert _is_auth_expired(SERVER_ERROR) is False
        assert _is_auth_expired(NO_STATUS) is False

    def test_sign_in_endpoint_keyerror(self):
        assert _is_auth_expired(KeyError("signInEndpoint")) is True

    def test_sign_in_to_marker(self):
        assert _is_auth_expired(KeyError("Please sign in to continue")) is True

    def test_logged_in_fragment_marker(self):
        assert _is_auth_expired(KeyError("['logged_in', 'value': '0']")) is True

    def test_unrelated_keyerror_is_not(self):
        assert _is_auth_expired(KeyError("missingRenderer")) is False
        assert _is_auth_expired(KeyError("signinendpoin")) is False  # prefix only

    def test_markers_match_case_insensitively(self):
        assert _is_auth_expired(KeyError("SIGNINENDPOINT")) is True
        assert _is_auth_expired(KeyError("Please SIGN IN TO continue")) is True

    def test_keyerror_subclass_is_instance(self):
        class DriftError(KeyError):
            pass

        assert _is_auth_expired(DriftError("signInEndpoint")) is True

    def test_other_exceptions_answer_false(self):
        assert _is_auth_expired(ValueError("sign in to")) is False
        assert _is_auth_expired(requests.exceptions.ConnectionError("sign in")) is False
        assert _is_auth_expired(YTMusicUserError("not signed in")) is False
        assert _is_auth_expired(RuntimeError("Server returned HTTP 401")) is False

    def test_multiple_inheritance_takes_the_server_arm(self):
        # isinstance(exc, YTMusicServerError) runs first, so a class
        # inheriting both delegates to the classifier — which checks
        # KeyError first and answers "server_error", not auth_expired.
        class BothError(KeyError, YTMusicServerError):
            pass

        assert _is_auth_expired(BothError("Server returned HTTP 401: nope")) is False

    def test_server_arm_delegates_through_the_module_classifier(self, monkeypatch):
        """The engine resolves ``_classify_mutation_failure`` through the
        module's own name, so whichever classifier the module is running
        (or a test stub) steers the verdict."""

        def fake_classifier(exc):
            return "auth_expired"

        monkeypatch.setattr(ytmusic_module, "_classify_mutation_failure", fake_classifier)
        assert _is_auth_expired(SERVER_ERROR) is True

        monkeypatch.setattr(
            ytmusic_module, "_classify_mutation_failure", lambda exc: "server_error"
        )
        assert _is_auth_expired(EXPIRED) is False


# ── _renewal_cooldown_gate ─────────────────────────────────────────


class TestRenewalCooldownGate:
    @pytest.fixture
    def frozen_clock(self, monkeypatch):
        monkeypatch.setattr(ytmusic_module, "time", SimpleNamespace(monotonic=lambda: _FROZEN_NOW))

    def test_future_retry_after_is_blocked(self, frozen_clock):
        blocked, remaining = _renewal_cooldown_gate(1_060.0)
        assert blocked is True
        assert remaining == 60.0

    def test_past_retry_after_is_not_blocked(self, frozen_clock):
        blocked, remaining = _renewal_cooldown_gate(940.0)
        assert blocked is False
        assert remaining == -60.0

    def test_the_exact_edge_is_not_blocked(self, frozen_clock):
        # `> 0`, never `>= 0`: a retry_after exactly at the clock has
        # elapsed and the renewal may run.
        blocked, remaining = _renewal_cooldown_gate(1_000.0)
        assert blocked is False
        assert remaining == 0.0

    def test_int_retry_after_lands_the_same_arithmetic(self, frozen_clock):
        blocked, remaining = _renewal_cooldown_gate(1500)
        assert blocked is True
        assert remaining == 500.0

    def test_the_clock_is_the_modules_own(self, frozen_clock):
        # The frozen namespace the module's `time` name was swapped for
        # steers the engine — the proof the clock is read through the
        # module attribute, not captured at import time.
        assert _renewal_cooldown_gate(1_030.0)[1] == 30.0

    def test_non_numeric_retry_after_raises_type_error(self, frozen_clock):
        with pytest.raises(TypeError, match="unsupported operand type"):
            _renewal_cooldown_gate("soon")


# ── _renewal_cooldown_stamp ────────────────────────────────────────


class TestRenewalCooldownStamp:
    @pytest.fixture
    def frozen_clock(self, monkeypatch):
        monkeypatch.setattr(ytmusic_module, "time", SimpleNamespace(monotonic=lambda: _FROZEN_NOW))

    def test_stamp_is_now_plus_the_cooldown(self, frozen_clock):
        assert _renewal_cooldown_stamp() == 1_060.0

    def test_patched_cooldown_constant_steer_both_backends(self, frozen_clock, monkeypatch):
        monkeypatch.setattr(ytmusic_module, "_RENEWAL_COOLDOWN_SECONDS", 90.5)
        assert _renewal_cooldown_stamp() == 1_090.5


# ── _api_failure_step ──────────────────────────────────────────────


class TestApiFailureStep:
    def test_below_threshold(self):
        assert _api_failure_step(0) == (1, False)
        assert _api_failure_step(1) == (2, False)

    def test_at_threshold_is_due(self):
        # The reference's `>=`: the third consecutive failure re-creates
        # the client (the module's cap is 3).
        assert _api_failure_step(2) == (3, True)

    def test_above_threshold_is_due(self):
        assert _api_failure_step(5) == (6, True)

    def test_patched_threshold_steers(self, monkeypatch):
        monkeypatch.setattr(ytmusic_module, "_MAX_API_FAILURES_BEFORE_REINIT", 10)
        assert _api_failure_step(5) == (6, False)
        assert _api_failure_step(9) == (10, True)

    def test_the_bump_is_cpythons_own_addition(self):
        assert _api_failure_step(2.5) == (3.5, True)
        assert _api_failure_step(True) == (2, False)
        assert _api_failure_step(-1) == (0, False)
        assert _api_failure_step(10**20) == (10**20 + 1, True)

    def test_non_numeric_count_raises_type_error(self):
        with pytest.raises(TypeError, match="can only concatenate str"):
            _api_failure_step("2")


# ── through the service ────────────────────────────────────────────


class TestServiceWiring:
    """The engines inside the service's own retry flow, on whichever
    backend is loaded — complements the renewal suite's fake-clock
    coverage with the two bookkeeping writes the engines own."""

    async def test_failed_renewal_stamps_the_cooldown(self, monkeypatch):
        from unittest.mock import MagicMock

        class Client:
            pass

        expired = Client()
        manager = MagicMock()
        manager.try_auto_refresh.return_value = False
        manager.create_bound_client.return_value = (Client(), "UCchan")
        svc = make_ytmusic_service(_ytm=expired, _auth_manager=manager)
        svc._client_identities[expired] = "UCchan"

        now = 5_000.0
        monkeypatch.setattr(ytmusic_module, "time", SimpleNamespace(monotonic=lambda: now))

        with pytest.raises(RuntimeError, match="expired"):
            await svc._renewed_client_for(expired, RuntimeError("expired"))
        # The failed-renewal arm stamped the cooldown through the engine:
        # the frozen clock plus the module's own constant.
        assert svc._renewal_retry_after == now + ytmusic_module._RENEWAL_COOLDOWN_SECONDS

    async def test_consecutive_failures_reinit_at_the_threshold(self):
        from unittest.mock import MagicMock

        class FailingClient:
            def get_song(self, video_id):
                raise requests.exceptions.ConnectionError("down")

        failing = FailingClient()
        manager = MagicMock()
        replacement = FailingClient()
        manager.create_bound_client.return_value = (replacement, "UCchan")
        svc = make_ytmusic_service(_ytm=failing, _auth_manager=manager)

        for _ in range(ytmusic_module._MAX_API_FAILURES_BEFORE_REINIT):
            with pytest.raises(requests.exceptions.ConnectionError):
                await svc._run(failing.get_song, "v1")

        # The third failure hit the cap: the client was replaced (the
        # counter reset with it) and the streak started over.
        assert svc.client is replacement
        assert svc._consecutive_api_failures == 0


# ── native vs fallback parity ──────────────────────────────────────

# Each case travels over a JSON bridge to a child interpreter with the
# extension import blocked; both sides freeze the module clock to the
# same value first, so the clock-dependent engines compare exactly.
#
# - auth_expired: {"t": "server"/"key"/"user"/"conn"/"value"/"badkey"/
#   "badserver", ...} names the exception to build.
# - cooldown_gate: the retry_after value ("bad" spellings raise).
# - cooldown_stamp: no arguments.
# - api_failure_step: the consecutive-failure count ("bad" spellings
#   raise).
_PARITY_CASES = {
    "auth_expired": [
        {"t": "server", "msg": "Server returned HTTP 401: Unauthorized.\n"},
        {"t": "server", "msg": "Server returned HTTP 403: Forbidden.\n"},
        {"t": "server", "msg": "Server returned HTTP 500: boom.\n"},
        {"t": "server", "msg": "no status at all"},
        {"t": "server", "msg": "Server returned HTTP 40: short.\n"},
        {"t": "server", "msg": "Server returned HTTP 4011: long.\n"},
        {"t": "key", "key": "signInEndpoint"},
        {"t": "key", "key": "SIGNINENDPOINT"},
        {"t": "key", "key": "Please sign in to continue"},
        {"t": "key", "key": "Please SIGN IN TO continue"},
        {"t": "key", "key": "['logged_in', 'value': '0']"},
        {"t": "key", "key": "missingRenderer"},
        {"t": "key", "key": "signinendpoin"},
        {"t": "key", "key": ""},
        # The İ lowering expands to i + combining dot, so the marker
        # never matches — on both backends, because both lower through
        # the interpreter's own str.lower.
        {"t": "key", "key": "SIGNİNENDPOINT"},
        {"t": "key", "key": "ſigninendpoint"},
        {"t": "user", "msg": "not signed in"},
        {"t": "conn", "msg": "sign in"},
        {"t": "value", "msg": "Server returned HTTP 401: nope"},
        {"t": "badkey"},
        {"t": "badserver"},
    ],
    "cooldown_gate": [1060.0, 1000.0, 940.0, 1500, 1000.5, 999.5, "soon", None],
    "cooldown_stamp": [[]],
    "api_failure_step": [0, 1, 2, 3, 5, 2.5, True, -1, 10**20, "2", None],
}


def _make_exc(spec):
    kind = spec["t"]
    if kind == "server":
        return YTMusicServerError(spec["msg"])
    if kind == "key":
        return KeyError(spec["key"])
    if kind == "user":
        return YTMusicUserError(spec["msg"])
    if kind == "conn":
        return requests.exceptions.ConnectionError(spec["msg"])
    if kind == "value":
        return ValueError(spec["msg"])
    if kind == "badkey":

        class BadStrKeyError(KeyError):
            def __str__(self):
                raise RuntimeError("no message for you")

        return BadStrKeyError("signInEndpoint")
    if kind == "badserver":

        class BadStrServerError(YTMusicServerError):
            def __str__(self):
                raise RuntimeError("no message for you")

        return BadStrServerError("Server returned HTTP 401: nope")
    raise AssertionError(f"unknown exception kind {kind!r}")


def _outcome(name, args):
    """One case's outcome on whichever backend this interpreter loaded,
    with the module clock frozen (the caller installs the freeze)."""
    try:
        if name == "auth_expired":
            result = _is_auth_expired(_make_exc(args))
        elif name == "cooldown_gate":
            result = _renewal_cooldown_gate(args)
        elif name == "cooldown_stamp":
            result = _renewal_cooldown_stamp()
        elif name == "api_failure_step":
            result = _api_failure_step(args)
        else:
            raise AssertionError(name)
    except BaseException as exc:  # noqa: BLE001 - repr'd for comparison
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, tuple):
        return {"ok": list(result)}
    return {"ok": result}


_CHILD_PROG = """
import json, sys
from types import SimpleNamespace

sys.modules["ytm_player.utils._ytm_player_native"] = None

import requests.exceptions
from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError

from ytm_player.services import ytmusic

ytmusic.time = SimpleNamespace(monotonic=lambda: 1000.0)


def make_exc(spec):
    kind = spec["t"]
    if kind == "server":
        return YTMusicServerError(spec["msg"])
    if kind == "key":
        return KeyError(spec["key"])
    if kind == "user":
        return YTMusicUserError(spec["msg"])
    if kind == "conn":
        return requests.exceptions.ConnectionError(spec["msg"])
    if kind == "value":
        return ValueError(spec["msg"])
    if kind == "badkey":
        class BadStrKeyError(KeyError):
            def __str__(self):
                raise RuntimeError("no message for you")
        return BadStrKeyError("signInEndpoint")
    if kind == "badserver":
        class BadStrServerError(YTMusicServerError):
            def __str__(self):
                raise RuntimeError("no message for you")
        return BadStrServerError("Server returned HTTP 401: nope")
    raise AssertionError(f"unknown exception kind {kind!r}")


def outcome(name, args):
    try:
        if name == "auth_expired":
            result = ytmusic._is_auth_expired(make_exc(args))
        elif name == "cooldown_gate":
            result = ytmusic._renewal_cooldown_gate(args)
        elif name == "cooldown_stamp":
            result = ytmusic._renewal_cooldown_stamp()
        elif name == "api_failure_step":
            result = ytmusic._api_failure_step(args)
        else:
            raise AssertionError(name)
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, tuple):
        return {"ok": list(result)}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {name: [outcome(name, args) for args in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the retry/renewal bookkeeping
    engines must answer every case exactly like the pure-Python
    references run in a child interpreter with the extension import
    blocked — same verdicts, same pairs, same errors with the same
    messages, both sides reading the same frozen module clock — so the
    two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_bookkeeping_matches_fallback(self, monkeypatch):
        monkeypatch.setattr(ytmusic_module, "time", SimpleNamespace(monotonic=lambda: _FROZEN_NOW))
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = {
            name: [_outcome(name, args) for args in cases] for name, cases in _PARITY_CASES.items()
        }
        assert set(native) == set(fallback) == set(_PARITY_CASES)
        for name, cases in _PARITY_CASES.items():
            for case, n, f in zip(cases, native[name], fallback[name]):
                assert n == f, (
                    f"native and fallback {name} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
