"""Tests for the playback-coordination engines.

The two pure decision cores behind ``app/_playback.py``'s failure and
history-report paths: ``_play_failure_policy``, the verdict
``_handle_play_failure`` consults once the consecutive-failure counter
has been bumped (repeat-one's hard stop, the below-threshold queue
advance — the queue's next track pulled through the caller's own bound
method exactly when the reference's arm consults it — the end-of-queue
toast, or the escalation whose resolver reset is gated on the failure
kind and the resolver's truthiness), and ``_history_report_step``, the
poll decision both report timer chains (``_report_local_play`` and
``_report_ytm_play``) consult on every tick (the listen-threshold gate
with its boundary kept on the poll side, the live-track reset of the
idle count, and the two-poll stability window's stop). The engines run
with two interchangeable backends: the pure-Python transcriptions in
the module and, when the Rust extension is built, the native
implementations bound at the module's tail. The failure cap resolves
through the module's own ``_MAX_CONSECUTIVE_FAILURES`` name at call
time, so patched values steer both backends identically — which the
tests below use for the patched-cap arms and the parity class pins as
the contract.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case — error shapes and the never-pulled
``next_track`` arms included, via callables that raise when pulled —
so the two implementations cannot drift apart. The wiring tests pin
the mixin's routing: ``_handle_play_failure`` consults the policy
engine with the host's own values (the bumped count, the very resolver
object, the queue's own bound method), both report chains consult the
step engine with the very listened value the insert worker receives,
and the verdicts steer the toasts, the deferred advance, the resolver
reset, the counter reset and the timer re-arms.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from ytm_player.app import _playback as playback_module
from ytm_player.app._playback import (
    _MAX_CONSECUTIVE_FAILURES,
    _YTM_HISTORY_POLL_SECONDS,
    PlaybackMixin,
    _history_report_step,
    _LocalHistoryClaim,
    _play_failure_policy,
)
from ytm_player.config.settings import DEFAULT_HISTORY_MIN_LISTEN_SECONDS

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_TRACK = {"video_id": "vid00000001", "title": "Next"}
_OTHER = {"video_id": "vid00000002", "title": "Other"}


def _pulls_the_queue():
    """A ``next_track`` stand-in that records its pulls and answers a track."""

    def next_track():
        next_track.pulls += 1  # type: ignore[attr-defined]
        return _TRACK

    next_track.pulls = 0  # type: ignore[attr-defined]
    return next_track


def _must_not_pull():
    """A ``next_track`` stand-in that raises if any backend pulls it."""

    def next_track() -> dict:
        raise RuntimeError("must not pull the queue in this arm")

    return next_track


# ── The playback-failure policy's arms ────────────────────────────────────


class TestFailurePolicyArms:
    """``_handle_play_failure``'s four verdicts, on whichever backend."""

    def test_repeat_one_stops_without_touching_the_queue(self):
        # A failing track in repeat-one must not skip itself forever —
        # the queue's own advance is never pulled.
        verdict = _play_failure_policy("one", 1, "stream", object(), _must_not_pull())
        assert verdict == {"action": "stop"}

    def test_below_the_cap_advances_with_the_queues_track(self):
        pulls = _pulls_the_queue()
        assert _play_failure_policy("off", 1, "stream", None, pulls) == {
            "action": "advance",
            "track": _TRACK,
        }
        assert pulls.pulls == 1  # type: ignore[attr-defined]

    def test_just_under_the_cap_still_advances(self):
        verdict = _play_failure_policy(
            "off", _MAX_CONSECUTIVE_FAILURES - 1, None, None, _pulls_the_queue()
        )
        assert verdict == {"action": "advance", "track": _TRACK}

    def test_at_the_cap_escalates(self):
        verdict = _play_failure_policy(
            "off", _MAX_CONSECUTIVE_FAILURES, "stream", "r", lambda: None
        )
        assert verdict == {"action": "escalate", "reset_resolver": True}

    def test_an_empty_queue_answers_end_of_queue(self):
        assert _play_failure_policy("off", 1, None, None, lambda: None) == {
            "action": "end_of_queue"
        }

    def test_a_falsy_next_track_is_end_of_queue(self):
        # The reference's ``if next_track:`` is truthiness, never identity.
        for falsy in ("", 0, [], {}):
            verdict = _play_failure_policy("off", 2, None, None, lambda f=falsy: f)
            assert verdict == {"action": "end_of_queue"}

    def test_escalate_reset_resolver_needs_a_kind(self):
        # The no-video-id path passes no kind — no resolver reset.
        verdict = _play_failure_policy("off", 5, None, "resolver", lambda: None)
        assert verdict == {"action": "escalate", "reset_resolver": False}

    def test_escalate_reset_resolver_needs_a_truthy_resolver(self):
        verdict = _play_failure_policy("off", 5, "stream", None, lambda: None)
        assert verdict == {"action": "escalate", "reset_resolver": False}

    def test_the_cap_resolves_through_the_modules_own_name(self, monkeypatch):
        # Patched at call time, steering whichever backend is loaded.
        monkeypatch.setattr(playback_module, "_MAX_CONSECUTIVE_FAILURES", 1)
        assert _play_failure_policy("off", 1, None, None, lambda: None) == {
            "action": "escalate",
            "reset_resolver": False,
        }
        assert _play_failure_policy("off", 1, "stream", "r", lambda: None) == {
            "action": "escalate",
            "reset_resolver": True,
        }
        monkeypatch.setattr(playback_module, "_MAX_CONSECUTIVE_FAILURES", 10)
        assert _play_failure_policy("off", 9, None, None, lambda: None) == {
            "action": "end_of_queue"
        }


class TestFailurePolicyErrorShapes:
    """The comparisons raise CPython's own errors — the engine never
    pre-validates."""

    def test_non_numeric_count_raises_the_lt_error(self):
        with pytest.raises(TypeError, match="'<' not supported"):
            _play_failure_policy("off", "many", None, None, lambda: None)

    def test_a_raising_next_track_propagates(self):
        def boom():
            raise RuntimeError("queue locked")

        with pytest.raises(RuntimeError, match="queue locked"):
            _play_failure_policy("off", 1, None, None, boom)


# ── The history-report poll step's arms ───────────────────────────────────


class TestReportStepArms:
    """The threshold gate, the idle reset and the two-poll window."""

    def test_a_listen_at_the_threshold_keeps_polling(self):
        # The reference's boundary: ``listened <= min`` polls — only a
        # strictly greater listen is reportable.
        assert _history_report_step(30, 30, _TRACK, 4) == {"action": "poll", "idle_polls": 0}

    def test_a_live_track_resets_the_idle_count(self):
        assert _history_report_step(0, 30, _TRACK, 4) == {"action": "poll", "idle_polls": 0}

    def test_a_listen_over_the_threshold_reports(self):
        assert _history_report_step(31, 30, _TRACK, 0) == {"action": "report"}

    def test_the_first_idle_poll_carries_the_bump(self):
        assert _history_report_step(5, 30, None, 0) == {"action": "poll", "idle_polls": 1}

    def test_the_second_idle_poll_stops(self):
        assert _history_report_step(5, 30, None, 1) == {"action": "stop"}

    def test_a_zero_minimum_still_gates_on_the_boundary(self):
        # 0 <= 0 polls — a zero threshold still demands a positive listen.
        assert _history_report_step(0, 0, None, 0) == {"action": "poll", "idle_polls": 1}

    def test_the_bool_spellings_stay_cpythons(self):
        # False <= 0 is True — the falsy spelling keeps polling.
        assert _history_report_step(False, 0, _TRACK, 0) == {"action": "poll", "idle_polls": 0}
        # True <= 0 is False — the truthy spelling reports.
        assert _history_report_step(True, 0, None, 1) == {"action": "report"}


class TestReportStepErrorShapes:
    """The comparisons raise CPython's own errors with its own
    messages — the engine never pre-validates."""

    def test_non_numeric_listened_raises_the_le_error(self):
        with pytest.raises(TypeError, match="'<=' not supported"):
            _history_report_step("30", 30, None, 0)

    def test_non_numeric_minimum_raises_the_le_error(self):
        with pytest.raises(TypeError, match="'<=' not supported"):
            _history_report_step(30, "30", None, 0)

    def test_non_numeric_idle_polls_raises_the_add_error(self):
        with pytest.raises(TypeError, match="can only concatenate str"):
            _history_report_step(5, 30, None, "x")


# ── The _handle_play_failure wiring ───────────────────────────────────────


def _failure_host(
    *,
    repeat_mode: str = "off",
    next_track: dict | None = None,
    resolver: object = "default",
    failures: int = 0,
) -> MagicMock:
    """A host carrying the attrs _handle_play_failure reads."""
    host = MagicMock()
    host.queue.repeat_mode = repeat_mode
    host.queue.next_track = MagicMock(return_value=next_track)
    host.queue.generation = 7
    host.stream_resolver = MagicMock() if resolver == "default" else resolver
    host._consecutive_failures = failures
    host._play_generation = 3
    host._last_play_video_id = "vid00000009"
    host._last_play_time = 12.5
    host.notify = MagicMock()
    host.call_later = MagicMock()
    host.run_worker = MagicMock()
    return host


def _run_failure(host, *, failure_kind="stream", clear_debounce=True, exhausted="exhausted"):
    PlaybackMixin._handle_play_failure.__get__(host)(
        exhausted_message=exhausted, failure_kind=failure_kind, clear_debounce=clear_debounce
    )


class TestHandlePlayFailureWiring:
    """``_handle_play_failure`` routes through the engine name and keeps
    its own state writes — the debounce clear, the counter bump, the
    toasts, the deferred advance and the resets."""

    def test_the_engine_is_consulted_with_the_hosts_own_values(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(**kwargs):
            calls.append(kwargs)
            return {"action": "end_of_queue"}

        monkeypatch.setattr(playback_module, "_play_failure_policy", recording_engine)

        host = _failure_host(failures=2)
        _run_failure(host, failure_kind="stream")

        assert calls == [
            {
                "repeat_mode": "off",
                "consecutive_failures": 3,  # bumped before the consult
                "failure_kind": "stream",
                "resolver": host.stream_resolver,  # the very object
                "next_track": host.queue.next_track,  # the queue's own bound method
            }
        ]
        host.notify.assert_called_once_with(
            "Playback failed — end of queue. Press Play to retry.", severity="error"
        )

    def test_the_debounce_is_cleared_before_the_consult(self):
        host = _failure_host()
        _run_failure(host, failure_kind=None)

        assert (host._last_play_video_id, host._last_play_time) == ("", 0.0)

    def test_clear_debounce_false_keeps_the_debounce_state(self):
        host = _failure_host()
        _run_failure(host, clear_debounce=False)

        assert (host._last_play_video_id, host._last_play_time) == ("vid00000009", 12.5)

    def test_the_stop_verdict_toasts_and_leaves_the_queue_alone(self):
        host = _failure_host(repeat_mode="one", next_track=_TRACK)
        _run_failure(host)

        host.notify.assert_called_once_with(
            "Playback stopped after failure. Press Play to retry.", severity="error"
        )
        host.queue.next_track.assert_not_called()
        host.call_later.assert_not_called()
        assert host._consecutive_failures == 1  # bumped, never reset

    async def test_the_advance_verdict_defers_through_call_later(self):
        host = _failure_host(next_track=_OTHER)
        host.play_track = AsyncMock()
        _run_failure(host, failure_kind=None)

        host.notify.assert_not_called()
        host.call_later.assert_called_once()
        # The deferred lambda schedules the worker…
        host.call_later.call_args.args[0]()
        host.run_worker.assert_called_once()
        # …and the worker coroutine plays the engine's very track.
        await host.run_worker.call_args.args[0]
        host.play_track.assert_awaited_once_with(_OTHER)

    def test_the_escalate_verdict_resets_the_resolver_and_the_counter(self):
        host = _failure_host(failures=_MAX_CONSECUTIVE_FAILURES - 1)
        _run_failure(host, exhausted="Multiple tracks failed")

        host.stream_resolver.clear_cache.assert_called_once()
        host.notify.assert_called_once_with("Multiple tracks failed", severity="error", timeout=6)
        assert host._consecutive_failures == 0

    def test_an_escalate_without_a_kind_skips_the_resolver_reset(self):
        host = _failure_host(failures=_MAX_CONSECUTIVE_FAILURES - 1)
        _run_failure(host, failure_kind=None, exhausted="exhausted")

        host.stream_resolver.clear_cache.assert_not_called()
        host.notify.assert_called_once_with("exhausted", severity="error", timeout=6)
        assert host._consecutive_failures == 0

    def test_a_patched_cap_steers_the_real_flow(self, monkeypatch):
        monkeypatch.setattr(playback_module, "_MAX_CONSECUTIVE_FAILURES", 1)
        host = _failure_host(failures=0, next_track=_TRACK)
        _run_failure(host, exhausted="escalated")

        host.stream_resolver.clear_cache.assert_called_once()
        assert host._consecutive_failures == 0


# ── The report-chain wiring ───────────────────────────────────────────────


def _report_host(
    *,
    position: float = 1.0,
    current_track: dict | None = _TRACK,
    min_listen: int = DEFAULT_HISTORY_MIN_LISTEN_SECONDS,
) -> MagicMock:
    """A host carrying the attrs both report chains read."""
    host = MagicMock()
    host.settings.playback.history_min_listen_seconds = min_listen
    host.settings.playback.sync_history_to_ytmusic = True
    host.ytmusic = MagicMock()
    host.history = MagicMock()
    host._play_generation = 1
    host._ytm_reported_generation = -1
    host._local_history_claim = None
    host._track_start_position = 0.0
    host.player.position = position
    host.player.current_track = current_track
    host.set_timer = MagicMock()
    host.run_worker = MagicMock()
    host._history_min_listen_seconds = PlaybackMixin._history_min_listen_seconds.__get__(host)
    # Bind the real poll callbacks so re-armed timer lambdas (which call
    # self._report_*) exercise the real chain instead of a MagicMock no-op.
    host._report_local_play = PlaybackMixin._report_local_play.__get__(host)
    host._report_ytm_play = PlaybackMixin._report_ytm_play.__get__(host)
    return host


def _claim(video_id: str = "vid1", generation: int = 1) -> _LocalHistoryClaim:
    return _LocalHistoryClaim(
        video_id=video_id, track={"video_id": video_id}, generation=generation
    )


class TestLocalReportWiring:
    """``_report_local_play`` routes through the engine name and keeps
    the claim/insert plumbing."""

    def test_the_engine_is_consulted_with_the_read_values(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(**kwargs):
            calls.append(kwargs)
            return {"action": "stop"}

        monkeypatch.setattr(playback_module, "_history_report_step", recording_engine)

        host = _report_host(position=12.5, current_track=_TRACK)
        host._history_min_listen_seconds = lambda: 10
        claim = _claim()
        host._local_history_claim = claim
        PlaybackMixin._report_local_play.__get__(host)(claim, idle_polls=3)

        assert calls == [
            {
                "listened": 12,
                "min_listen_seconds": 10,
                "current_track": _TRACK,
                "idle_polls": 3,
            }
        ]

    def test_the_report_verdict_hands_the_insert_the_same_listened_value(self):
        host = _report_host(position=42.5, current_track=_TRACK)
        host._history_min_listen_seconds = lambda: 10
        seen: list[tuple[_LocalHistoryClaim, int]] = []

        def fake_insert(claim, listened):
            seen.append((claim, listened))
            return "insert-coroutine"

        host._insert_local_history_play = fake_insert
        claim = _claim()
        host._local_history_claim = claim

        PlaybackMixin._report_local_play.__get__(host)(claim)

        host.run_worker.assert_called_once_with("insert-coroutine", group="local-history-report")
        assert seen == [(claim, 42)]
        assert claim.insert_started is True

    def test_the_poll_verdict_re_arms_with_the_stepped_idle_count(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_history_report_step",
            lambda **kwargs: {"action": "poll", "idle_polls": 9},
        )
        host = _report_host()
        claim = _claim()
        host._local_history_claim = claim

        PlaybackMixin._report_local_play.__get__(host)(claim, idle_polls=4)

        host.set_timer.assert_called_once()
        delay, rearm = host.set_timer.call_args.args
        assert delay == _YTM_HISTORY_POLL_SECONDS
        rearm()
        # The re-armed chain ran again — a second consult is visible as
        # a second poll re-arm (the recording engine answers poll 9).
        assert host.set_timer.call_count == 2

    def test_the_stop_verdict_re_arms_nothing(self, monkeypatch):
        monkeypatch.setattr(
            playback_module, "_history_report_step", lambda **kwargs: {"action": "stop"}
        )
        host = _report_host()
        claim = _claim()
        host._local_history_claim = claim

        PlaybackMixin._report_local_play.__get__(host)(claim)

        host.set_timer.assert_not_called()
        host.run_worker.assert_not_called()


class TestYtmReportWiring:
    """``_report_ytm_play`` routes through the same engine name."""

    def test_the_engine_is_consulted_with_the_read_values(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(**kwargs):
            calls.append(kwargs)
            return {"action": "stop"}

        monkeypatch.setattr(playback_module, "_history_report_step", recording_engine)

        host = _report_host(position=5.5, current_track=None)
        host._history_min_listen_seconds = lambda: 30
        PlaybackMixin._report_ytm_play.__get__(host)({"video_id": "vid1"}, "vid1", 1, idle_polls=1)

        assert calls == [
            {
                "listened": 5,
                "min_listen_seconds": 30,
                "current_track": None,
                "idle_polls": 1,
            }
        ]

    def test_the_report_verdict_schedules_the_push_worker(self):
        host = _report_host(position=61.0, current_track=_TRACK)
        host._history_min_listen_seconds = lambda: 60
        seen: list[tuple[dict, str, int]] = []

        def fake_push(track, video_id, generation):
            seen.append((track, video_id, generation))
            return "push-coroutine"

        host._push_ytm_history_report = fake_push
        track = {"video_id": "vid1"}

        PlaybackMixin._report_ytm_play.__get__(host)(track, "vid1", 1)

        host.run_worker.assert_called_once_with("push-coroutine", group="ytm-history-report")
        assert seen == [(track, "vid1", 1)]

    def test_the_poll_verdict_re_arms_with_the_stepped_idle_count(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_history_report_step",
            lambda **kwargs: {"action": "poll", "idle_polls": 7},
        )
        host = _report_host()
        track = {"video_id": "vid1"}

        PlaybackMixin._report_ytm_play.__get__(host)(track, "vid1", 1, idle_polls=2)

        host.set_timer.assert_called_once()
        delay, rearm = host.set_timer.call_args.args
        assert delay == _YTM_HISTORY_POLL_SECONDS
        rearm()
        assert host.set_timer.call_count == 2

    def test_the_stop_verdict_re_arms_nothing(self, monkeypatch):
        monkeypatch.setattr(
            playback_module, "_history_report_step", lambda **kwargs: {"action": "stop"}
        )
        host = _report_host()

        PlaybackMixin._report_ytm_play.__get__(host)({"video_id": "vid1"}, "vid1", 1)

        host.set_timer.assert_not_called()
        host.run_worker.assert_not_called()


# ── Native parity ─────────────────────────────────────────────────────────

# Failure-policy cases: [repeat_mode, consecutive_failures, failure_kind,
# resolver truthy, next_track result ("track" | null | "empty_str" |
# "raises"), patched cap (None → the module default)]. The "raises"
# spellings pin the arms that must never pull the queue — a pull would
# surface as the callable's own error instead of a verdict.
_FAILURE_PARITY_CASES = [
    ["one", 1, "stream", True, "track", None],
    ["one", 5, None, True, "raises", None],
    ["off", 1, None, True, "track", None],
    ["off", 4, "stream", True, "track", None],
    ["off", 4, None, False, None, None],
    ["off", 5, "stream", True, None, None],
    ["off", 5, "stream", False, None, None],
    ["off", 5, None, True, None, None],
    ["off", 6, "stream", True, None, None],
    ["off", 2, "stream", True, "empty_str", None],
    ["off", 3, "stream", True, "track", 1],
    ["off", 1, "stream", True, "track", 10],
    ["off", 9, None, False, None, 10],
    ["off", "many", None, True, None, None],
    [None, 1, None, True, None, None],
    [123, 1, "stream", True, None, None],
]

# Report-step cases: [listened, min_listen_seconds, current track present,
# idle_polls] — scalars only, the outcome row or the error's own shape.
_STEP_PARITY_CASES = [
    [30, 30, True, 4],
    [29, 30, True, 3],
    [31, 30, True, 0],
    [0, 30, True, 4],
    [5, 30, False, 0],
    [5, 30, False, 1],
    [0, 0, False, 0],
    [0, 0, True, 2],
    [1, 0, False, 1],
    [False, 0, True, 0],
    [True, 0, False, 1],
    [120.5, 240, True, 0],
    ["30", 30, False, 0],
    [30, "30", False, 0],
    [5, 30, False, "x"],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app import _playback

_DEFAULT_MAX = _playback._MAX_CONSECUTIVE_FAILURES


def make_next_track(desc):
    if desc == "raises":
        def next_track():
            raise RuntimeError("must not pull")
        return next_track
    if desc == "track":
        return lambda: {"video_id": "t"}
    if desc == "empty_str":
        return lambda: ""
    return lambda: None


def failure_outcome(case):
    repeat_mode, failures, kind, resolver_truthy, nt_desc, cap = case
    saved = _playback._MAX_CONSECUTIVE_FAILURES
    if cap is not None:
        _playback._MAX_CONSECUTIVE_FAILURES = cap
    resolver = object() if resolver_truthy else None
    try:
        result = _playback._play_failure_policy(
            repeat_mode, failures, kind, resolver, make_next_track(nt_desc)
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        _playback._MAX_CONSECUTIVE_FAILURES = saved
    return {"ok": result}


def step_outcome(case):
    listened, minimum, has_track, idle = case
    try:
        result = _playback._history_report_step(
            listened, minimum, {"video_id": "t"} if has_track else None, idle
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


payload = json.load(sys.stdin)
json.dump(
    {
        "failure": [failure_outcome(c) for c in payload["failure"]],
        "step": [step_outcome(c) for c in payload["step"]],
    },
    sys.stdout,
)
"""


def _child_next_track(desc):
    """The parent-side twin of the child program's callable builder."""
    if desc == "raises":

        def next_track() -> dict:
            raise RuntimeError("must not pull")

        return next_track
    if desc == "track":
        return lambda: {"video_id": "t"}
    if desc == "empty_str":
        return lambda: ""
    return lambda: None


def _native_failure_outcomes():
    """The parent-side failure outcomes — the cap installed per case and
    restored once the engine has run, so the module never leaks."""

    def outcome(case):
        repeat_mode, failures, kind, resolver_truthy, nt_desc, cap = case
        saved = playback_module._MAX_CONSECUTIVE_FAILURES
        if cap is not None:
            playback_module._MAX_CONSECUTIVE_FAILURES = cap
        resolver = object() if resolver_truthy else None
        try:
            result = playback_module._play_failure_policy(
                repeat_mode, failures, kind, resolver, _child_next_track(nt_desc)
            )
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            playback_module._MAX_CONSECUTIVE_FAILURES = saved
        return {"ok": result}

    return [outcome(case) for case in _FAILURE_PARITY_CASES]


def _native_step_outcomes():
    def outcome(case):
        listened, minimum, has_track, idle = case
        try:
            result = playback_module._history_report_step(
                listened, minimum, {"video_id": "t"} if has_track else None, idle
            )
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"ok": result}

    return [outcome(case) for case in _STEP_PARITY_CASES]


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same verdicts, same rows,
    same errors with the same messages, the never-pulled queue arms
    included — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_failure_policy_matches_the_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps({"failure": _FAILURE_PARITY_CASES, "step": _STEP_PARITY_CASES}),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_failure = _native_failure_outcomes()
        for case, n, f in zip(_FAILURE_PARITY_CASES, native_failure, fallback["failure"]):
            assert n == f, (
                f"native and fallback _play_failure_policy differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_report_step_matches_the_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps({"failure": _FAILURE_PARITY_CASES, "step": _STEP_PARITY_CASES}),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_step = _native_step_outcomes()
        for case, n, f in zip(_STEP_PARITY_CASES, native_step, fallback["step"]):
            assert n == f, (
                f"native and fallback _history_report_step differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
