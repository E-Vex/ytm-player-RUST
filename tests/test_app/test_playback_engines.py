"""Tests for the playback-coordination engines.

The pure decision cores behind ``app/_playback.py``'s failure and
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
idle count, and the two-poll stability window's stop).

The event and history cores: ``_attempt_gate``, the stale-attempt
consume gate both player-event handlers (``_on_player_error`` and
``_on_track_end``) consult before acting on a payload's attempt token;
``_poll_failure_reset``, the four-arm gate ``_poll_position`` consults
before clearing the failure counter and the recovery marker (the
player's own ``is_playing`` / ``current_attempt`` read exactly when the
chain reaches them, so a short-circuited chain never touches the
player); ``_notification_message``, ``_on_track_change``'s toast
assembly (the settings template through ``str.format`` with the
``KeyError`` / ``ValueError`` fallback spelling); ``_prefetch_next_id``,
the queue-head gate behind ``_prefetch_next_track``; ``_refill_plan``,
the refill gates and seed draw behind ``_refill_queue``;
``_local_listen_plan``, the finalize branch tree behind
``_log_local_listen`` (claim ownership, the consumed dedup, the measured
listen, and the update-row / stash-pending / direct-log branch);
``_insert_finalize_plan``, the pending handoff and still-current
generation check behind ``_insert_local_history_play``; and the timer
arm pair ``_min_listen_seconds`` / ``_timer_arm_delay`` behind
``_history_min_listen_seconds`` and ``_history_timer_delay``.

The engines run with two interchangeable backends: the pure-Python
transcriptions in the module and, when the Rust extension is built, the
native implementations bound at the module's tail. The failure cap and
the poll interval resolve through the module's own names at call
time, so patched values steer both backends identically — which the
tests below use for the patched-cap arms and the parity class pins as
the contract.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case — error shapes and the never-pulled arms
included, via callables that raise when pulled — so the two
implementations cannot drift apart. The wiring tests pin the mixin's
routing: ``_handle_play_failure`` consults the policy engine with the
host's own values (the bumped count, the very resolver object, the
queue's own bound method), both report chains consult the step engine
with the very listened value the insert worker receives, both event
handlers consult the attempt gate with their own handled counters, the
poll consults the reset engine with the very player object, and the
verdicts steer the toasts, the deferred advance, the resolver reset,
the counter reset, the claim writes, the history calls and the timer
re-arms.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from ytm_player.app import _playback as playback_module
from ytm_player.app._playback import (
    _MAX_CONSECUTIVE_FAILURES,
    _YTM_HISTORY_POLL_SECONDS,
    PlaybackMixin,
    _attempt_gate,
    _history_report_step,
    _insert_finalize_plan,
    _local_listen_plan,
    _LocalHistoryClaim,
    _min_listen_seconds,
    _notification_message,
    _play_failure_policy,
    _poll_failure_reset,
    _prefetch_next_id,
    _refill_plan,
    _timer_arm_delay,
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


# ── The stale-attempt gate's arms ──────────────────────────────────────────


class TestAttemptGateArms:
    """``_attempt_gate``'s verdict, on whichever backend."""

    def test_a_current_attempt_passes(self):
        assert _attempt_gate(3, 3, 2, 0, 0) is True

    def test_a_bool_attempt_is_an_int_exactly_like_cpython(self):
        assert _attempt_gate(True, 1, 0, 0, 0) is True

    def test_a_float_attempt_is_not_an_int(self):
        assert _attempt_gate(3.0, 3, 2, 0, 0) is False

    def test_a_handled_attempt_is_dropped(self):
        assert _attempt_gate(3, 3, 3, 0, 0) is False

    def test_an_attempt_from_an_older_play_is_dropped(self):
        assert _attempt_gate(2, 3, 1, 0, 0) is False

    def test_a_non_int_attempt_is_dropped(self):
        assert _attempt_gate("3", 3, 2, 0, 0) is False
        assert _attempt_gate(None, 3, 2, 0, 0) is False

    def test_a_stop_supersession_is_dropped(self):
        assert _attempt_gate(3, 3, 2, 1, 2) is False

    def test_agreeing_stop_generations_pass(self):
        assert _attempt_gate(3, 3, 2, 1, 1) is True

    def test_a_non_numeric_handled_counter_raises_the_le_error(self):
        with pytest.raises(TypeError, match="<="):
            _attempt_gate(3, 3, "x", 0, 0)


# ── The poll-step failure reset's arms ─────────────────────────────────────


class _BoomIsPlaying:
    """A player stand-in whose ``is_playing`` must never be read."""

    @property
    def is_playing(self):
        raise RuntimeError("must not read is_playing on this arm")


class _BoomCurrentAttempt:
    """A player stand-in whose ``current_attempt`` must never be read."""

    is_playing = True

    @property
    def current_attempt(self):
        raise RuntimeError("must not read current_attempt on this arm")


class TestPollFailureResetArms:
    """``_poll_failure_reset``'s verdict, on whichever backend."""

    def test_advancing_audio_with_a_failure_state_resets(self):
        player = SimpleNamespace(is_playing=True, current_attempt=4)
        assert _poll_failure_reset(5.0, 2, None, player, 4) is True

    def test_a_recovery_marker_alone_is_a_failure_state(self):
        player = SimpleNamespace(is_playing=True, current_attempt=4)
        assert _poll_failure_reset(5.0, 0, 7, player, 4) is True

    def test_an_untouched_state_never_touches_the_player(self):
        # The chain short-circuits before is_playing is read.
        assert _poll_failure_reset(5.0, 0, None, _BoomIsPlaying(), 4) is False

    def test_a_zero_position_never_touches_the_player(self):
        assert _poll_failure_reset(0.0, 2, None, _BoomIsPlaying(), 4) is False

    def test_a_paused_player_does_not_reset(self):
        player = SimpleNamespace(is_playing=False, current_attempt=4)
        assert _poll_failure_reset(5.0, 2, None, player, 4) is False

    def test_a_stale_attempt_does_not_reset(self):
        player = SimpleNamespace(is_playing=True, current_attempt=3)
        assert _poll_failure_reset(5.0, 2, None, player, 4) is False

    def test_an_int_position_passes_like_cpython(self):
        player = SimpleNamespace(is_playing=True, current_attempt=4)
        assert _poll_failure_reset(5, 2, None, player, 4) is True

    def test_a_non_numeric_position_raises_the_gt_error(self):
        with pytest.raises(TypeError, match=">"):
            _poll_failure_reset("x", 2, None, SimpleNamespace(is_playing=True), 4)


# ── The track-change notification's arms ───────────────────────────────────


class TestNotificationMessageArms:
    """``_notification_message``'s copy, on whichever backend."""

    def test_the_template_interpolates_the_payload(self):
        msg = _notification_message(
            {"title": "T", "artist": "A", "album": "AL"}, "{title} — {artist} ({album})"
        )
        assert msg == "T — A (AL)"

    def test_a_missing_album_travels_as_the_empty_string(self):
        assert _notification_message({"title": "T", "artist": "A"}, "{title}|{album}|") == "T||"

    def test_missing_title_and_artist_take_the_unknown_defaults(self):
        assert _notification_message({}, "{title} by {artist}") == "Unknown by Unknown"

    def test_an_unknown_placeholder_falls_back(self):
        assert _notification_message({"title": "T", "artist": "A"}, "{title} {nope}") == "T — A"

    def test_a_malformed_template_falls_back(self):
        assert _notification_message({"title": "T", "artist": "A"}, "{title") == "T — A"

    def test_the_fallback_stringifies_non_str_titles(self):
        assert _notification_message({"title": 7, "artist": None}, "{nope}") == "7 — None"

    def test_conversions_and_specs_are_cpythons_own(self):
        assert _notification_message({"title": "T", "artist": "A"}, "{title!r:>6}") == "   'T'"

    def test_a_non_str_template_raises_the_attribute_error(self):
        with pytest.raises(AttributeError, match="'int' object has no attribute 'format'"):
            _notification_message({"title": "T", "artist": "A"}, 42)


# ── The prefetch gate's arms ───────────────────────────────────────────────


class TestPrefetchNextIdArms:
    """``_prefetch_next_id``'s answer, on whichever backend."""

    def test_a_falsy_queue_head_answers_none(self):
        assert _prefetch_next_id(None) is None
        assert _prefetch_next_id({}) is None

    def test_a_head_without_a_video_id_answers_none(self):
        assert _prefetch_next_id({"other": 1}) is None
        assert _prefetch_next_id({"video_id": ""}) is None

    def test_a_truthy_id_answers_the_very_object(self):
        assert _prefetch_next_id({"video_id": "abc"}) == "abc"
        assert _prefetch_next_id({"video_id": 7}) == 7

    def test_a_head_without_get_raises_the_attribute_error(self):
        with pytest.raises(AttributeError):
            _prefetch_next_id(5)


# ── The queue-refill plan's arms ───────────────────────────────────────────


def _queue_stand_in(tracks, real_index):
    return SimpleNamespace(tracks=tracks, real_index=real_index)


class TestRefillPlanArms:
    """``_refill_plan``'s answer, on whichever backend."""

    _TRACKS = [{"i": i} for i in range(8)]

    def test_repeat_mode_other_than_off_answers_none(self):
        assert _refill_plan("one", True, 2, False, _queue_stand_in(self._TRACKS, 1), None) is None

    def test_disabled_autoplay_answers_none(self):
        assert _refill_plan("off", False, 2, False, _queue_stand_in(self._TRACKS, 1), None) is None

    def test_more_than_three_remaining_answers_none(self):
        assert _refill_plan("off", True, 4, False, _queue_stand_in(self._TRACKS, 1), None) is None

    def test_three_remaining_is_the_boundary_that_proceeds(self):
        seeds = _refill_plan(
            "off", True, 3, False, _queue_stand_in([{"i": 0}, {"i": 1}, {"i": 2}], 1), None
        )
        assert seeds == [{"i": 0}, {"i": 1}]

    def test_a_running_extend_worker_answers_none(self):
        assert _refill_plan("off", True, 2, True, _queue_stand_in(self._TRACKS, 1), None) is None

    def test_the_seeds_are_the_last_five_of_the_played_prefix(self):
        seeds = _refill_plan("off", True, 2, False, _queue_stand_in(self._TRACKS, 6), None)
        assert seeds == [{"i": i} for i in range(2, 7)]

    def test_a_short_prefix_seeds_itself_whole(self):
        seeds = _refill_plan("off", True, 2, False, _queue_stand_in([{"i": 0}], 0), None)
        assert seeds == [{"i": 0}]

    def test_empty_seeds_fall_back_to_the_players_current_track(self):
        player = SimpleNamespace(current_track={"video_id": "v"})
        seeds = _refill_plan("off", True, 2, False, _queue_stand_in([], -1), player)
        assert seeds == [{"video_id": "v"}]

    def test_a_negative_index_walks_the_fallback_too(self):
        player = SimpleNamespace(current_track={"video_id": "v"})
        seeds = _refill_plan("off", True, 2, False, _queue_stand_in(self._TRACKS, -1), player)
        assert seeds == [{"video_id": "v"}]

    def test_still_empty_seeds_answer_none(self):
        assert _refill_plan("off", True, 2, False, _queue_stand_in([], -1), None) is None
        assert (
            _refill_plan(
                "off", True, 2, False, _queue_stand_in([], -1), SimpleNamespace(current_track=None)
            )
            is None
        )

    def test_a_non_numeric_remaining_raises_the_gt_error(self):
        with pytest.raises(TypeError, match=">"):
            _refill_plan("off", True, "x", False, _queue_stand_in([], 0), None)


# ── The local-history finalize plan's arms ─────────────────────────────────


def _claim_stand_in(
    video_id="v",
    finalized=False,
    row_id=None,
    insert_started=False,
    pending_seconds=None,
    generation=3,
    track=None,
):
    return SimpleNamespace(
        video_id=video_id,
        finalized=finalized,
        row_id=row_id,
        insert_started=insert_started,
        pending_seconds=pending_seconds,
        generation=generation,
        track=track if track is not None else {"video_id": video_id},
    )


class TestLocalListenPlanArms:
    """``_local_listen_plan``'s branch tree, on whichever backend."""

    _PLAYER = SimpleNamespace(position=42.0)

    def test_an_unowned_claim_direct_logs_without_marking(self):
        plan = _local_listen_plan(None, "v", self._PLAYER, 2.0)
        assert plan == {"action": "direct_log", "listened": 40, "mark_finalized": False}

    def test_a_claim_for_another_video_is_not_owned(self):
        plan = _local_listen_plan(_claim_stand_in(video_id="other"), "v", self._PLAYER, 2.0)
        assert plan == {"action": "direct_log", "listened": 40, "mark_finalized": False}

    def test_an_owned_claim_without_a_row_direct_logs_marked(self):
        plan = _local_listen_plan(_claim_stand_in(), "v", self._PLAYER, 2.0)
        assert plan == {"action": "direct_log", "listened": 40, "mark_finalized": True}

    def test_an_already_finalized_claim_is_consumed_unmeasured(self):
        plan = _local_listen_plan(_claim_stand_in(finalized=True), "v", self._PLAYER, 2.0)
        assert plan == {"action": "consumed", "mark_finalized": False}
        assert "listened" not in plan

    def test_an_owned_claim_with_a_row_updates_that_row(self):
        plan = _local_listen_plan(_claim_stand_in(row_id=9), "v", self._PLAYER, 2.0)
        assert plan == {
            "action": "update_row",
            "listened": 40,
            "mark_finalized": True,
            "row_id": 9,
        }

    def test_an_owned_claim_with_a_started_insert_stashes(self):
        plan = _local_listen_plan(_claim_stand_in(insert_started=True), "v", self._PLAYER, 2.0)
        assert plan == {"action": "stash_pending", "listened": 40, "mark_finalized": True}

    def test_a_zero_listen_writes_nothing_but_still_marks(self):
        plan = _local_listen_plan(_claim_stand_in(), "v", SimpleNamespace(position=2.0), 2.0)
        assert plan == {"action": "no_write", "listened": 0, "mark_finalized": True}
        plan = _local_listen_plan(None, "v", SimpleNamespace(position=2.0), 2.0)
        assert plan == {"action": "no_write", "listened": 0, "mark_finalized": False}

    def test_a_fractional_negative_listen_truncates_to_zero(self):
        plan = _local_listen_plan(None, "v", SimpleNamespace(position=1.5), 2.0)
        assert plan == {"action": "no_write", "listened": 0, "mark_finalized": False}

    def test_a_non_numeric_position_raises_the_sub_error(self):
        with pytest.raises(TypeError, match="-"):
            _local_listen_plan(None, "v", SimpleNamespace(position="x"), 2.0)

    def test_an_infinite_position_raises_the_int_overflow(self):
        with pytest.raises(OverflowError):
            _local_listen_plan(None, "v", SimpleNamespace(position=float("inf")), 2.0)


# ── The insert-finalize plan's arms ────────────────────────────────────────


class TestInsertFinalizePlanArms:
    """``_insert_finalize_plan``'s verdict, on whichever backend."""

    def _claim(self, pending_seconds, generation=3):
        return SimpleNamespace(pending_seconds=pending_seconds, generation=generation)

    def test_a_pending_beating_the_row_is_applied(self):
        assert _insert_finalize_plan(self._claim(50), 40, 3) == {
            "had_pending": True,
            "pending": 50,
            "apply_pending": True,
            "optimistic_add": False,
        }

    def test_a_shorter_pending_keeps_the_rows_own_value(self):
        assert _insert_finalize_plan(self._claim(30), 40, 3)["apply_pending"] is False

    def test_an_equal_pending_is_not_applied(self):
        assert _insert_finalize_plan(self._claim(40), 40, 3)["apply_pending"] is False

    def test_without_a_pending_the_generation_check_answers(self):
        assert _insert_finalize_plan(self._claim(None, 3), 40, 3) == {
            "had_pending": False,
            "pending": None,
            "apply_pending": False,
            "optimistic_add": True,
        }
        assert _insert_finalize_plan(self._claim(None, 4), 40, 3)["optimistic_add"] is False

    def test_a_non_numeric_pending_raises_the_gt_error(self):
        with pytest.raises(TypeError, match=">"):
            _insert_finalize_plan(self._claim("x"), 40, 3)


# ── The timer arm pair's arms ──────────────────────────────────────────────


class TestTimerArmPairArms:
    """``_min_listen_seconds`` / ``_timer_arm_delay``, on whichever backend."""

    def test_the_clamp_coerces_and_floors(self):
        assert _min_listen_seconds(30) == 30
        assert _min_listen_seconds(2.7) == 2
        assert _min_listen_seconds(-5) == 0
        assert _min_listen_seconds("30") == 30
        assert _min_listen_seconds(True) == 1

    def test_the_clamp_raises_the_int_errors(self):
        with pytest.raises(ValueError):
            _min_listen_seconds("x")
        with pytest.raises(TypeError):
            _min_listen_seconds(None)

    def test_the_arm_delay_floors_at_the_poll_interval(self):
        assert _timer_arm_delay(0) == _YTM_HISTORY_POLL_SECONDS
        assert _timer_arm_delay(30) == 30.0

    def test_a_patched_poll_interval_steers_both_backends(self, monkeypatch):
        monkeypatch.setattr(playback_module, "_YTM_HISTORY_POLL_SECONDS", 2.5)
        assert _timer_arm_delay(0) == 2.5
        assert _timer_arm_delay(30) == 30.0

    def test_the_arm_delay_coerces_through_float(self):
        assert _timer_arm_delay("30") == 30.0

    def test_the_arm_delay_raises_the_float_errors(self):
        with pytest.raises(ValueError):
            _timer_arm_delay("x")
        with pytest.raises(TypeError):
            _timer_arm_delay(None)


# ── The player-event handler wiring ────────────────────────────────────────


def _event_host(**overrides):
    """A host carrying the attrs both event handlers read."""
    host = MagicMock()
    host._play_generation = 3
    host._handled_error_attempt = 1
    host._handled_end_attempt = 2
    host._active_stop_generation = 0
    host._queue_stop_generation = 0
    host._recovery_generation = None
    host._play_request = 0
    host.queue.generation = 7
    host.stream_resolver = None
    host.player = None
    for key, value in overrides.items():
        setattr(host, key, value)
    return host


class TestOnPlayerErrorWiring:
    """``_on_player_error`` routes through the gate engine name."""

    async def test_the_gate_is_consulted_with_the_hosts_own_values(self, monkeypatch):
        calls: list[tuple] = []

        def recording_gate(*args):
            calls.append(args)
            return False

        monkeypatch.setattr(playback_module, "_attempt_gate", recording_gate)

        host = _event_host()
        await PlaybackMixin._on_player_error.__get__(host)({"attempt": 3, "track": {}})

        assert calls == [(3, 3, 1, 0, 0)]
        assert host._handled_error_attempt == 1  # not consumed

    async def test_a_current_attempt_is_consumed_as_the_new_counter(self):
        host = _event_host()
        await PlaybackMixin._on_player_error.__get__(host)(
            {"attempt": 3, "track": {"video_id": "vid1"}}
        )

        assert host._handled_error_attempt == 3

    async def test_a_stale_attempt_leaves_the_counter_alone(self):
        host = _event_host()
        await PlaybackMixin._on_player_error.__get__(host)(
            {"attempt": 2, "track": {"video_id": "vid1"}}
        )

        assert host._handled_error_attempt == 1


class TestOnTrackEndWiring:
    """``_on_track_end`` routes through the same gate engine name."""

    async def test_the_gate_is_consulted_with_the_defaulted_counter(self, monkeypatch):
        calls: list[tuple] = []
        # A SimpleNamespace host: unlike a MagicMock it does not invent
        # ``_handled_end_attempt``, so the consult carries the
        # reference's own -1 default for a first-ever end event.
        host = SimpleNamespace(
            _play_generation=3,
            _active_stop_generation=0,
            _queue_stop_generation=0,
            _play_request=0,
            queue=SimpleNamespace(generation=7),
        )

        def recording_gate(*args):
            calls.append(args)
            return False

        monkeypatch.setattr(playback_module, "_attempt_gate", recording_gate)

        await PlaybackMixin._on_track_end.__get__(host)({"attempt": 3})

        assert calls == [(3, 3, -1, 0, 0)]
        assert not hasattr(host, "_handled_end_attempt")  # not consumed

    async def test_a_current_eof_is_consumed_and_advances(self):
        host = _event_host()
        host._play_next = AsyncMock()

        await PlaybackMixin._on_track_end.__get__(host)({"attempt": 3, "track": None})

        assert host._handled_end_attempt == 3
        host._play_next.assert_awaited_once_with(ended_track=None)

    async def test_a_duplicate_eof_is_not_consumed_twice(self):
        host = _event_host()
        host._play_next = AsyncMock()

        await PlaybackMixin._on_track_end.__get__(host)({"attempt": 3, "track": None})
        await PlaybackMixin._on_track_end.__get__(host)({"attempt": 3, "track": None})

        host._play_next.assert_awaited_once()


# ── The poll / notification / prefetch / refill wiring ─────────────────────


class TestPollPositionWiring:
    """``_poll_position`` routes through the reset engine name."""

    def test_the_engine_is_consulted_with_the_very_player(self, monkeypatch):
        calls: list[tuple] = []
        player = SimpleNamespace(position=5.0, duration=100.0, is_playing=True, current_attempt=4)

        def recording_reset(*args):
            calls.append(args)
            return True

        monkeypatch.setattr(playback_module, "_poll_failure_reset", recording_reset)

        host = MagicMock()
        host.player = player
        host.query_one = MagicMock(return_value=MagicMock())
        host.mpris = None
        host.mac_media = None
        host.lastfm = None
        host._consecutive_failures = 2
        host._recovery_generation = None
        host._play_generation = 4

        PlaybackMixin._poll_position.__get__(host)()

        assert calls == [(5.0, 2, None, player, 4)]
        assert host._consecutive_failures == 0
        assert host._recovery_generation is None

    def test_a_false_verdict_keeps_the_failure_state(self, monkeypatch):
        monkeypatch.setattr(playback_module, "_poll_failure_reset", lambda *args: False)

        host = MagicMock()
        host.player = SimpleNamespace(position=5.0, duration=100.0)
        host.query_one = MagicMock(return_value=MagicMock())
        host.mpris = None
        host.mac_media = None
        host.lastfm = None
        host._consecutive_failures = 2
        host._recovery_generation = 9
        host._play_generation = 4

        PlaybackMixin._poll_position.__get__(host)()

        assert host._consecutive_failures == 2
        assert host._recovery_generation == 9


class TestOnTrackChangeWiring:
    """``_on_track_change`` routes through the notification engine name."""

    def _host(self, *, enabled=True, fmt="{title} — {artist}"):
        host = MagicMock()
        host._refill_queue = MagicMock()
        host.query_one = MagicMock(return_value=MagicMock())
        host._get_current_page = MagicMock(return_value=None)
        host._prefetch_next_track = MagicMock()
        host.settings.notifications.enabled = enabled
        host.settings.notifications.format = fmt
        host.settings.notifications.timeout_seconds = 4
        return host

    def test_the_engine_is_consulted_with_the_track_and_the_template(self, monkeypatch):
        calls: list[tuple] = []
        track = {"video_id": "vid1", "title": "T", "artist": "A"}

        def recording_message(*args):
            calls.append(args)
            return "Now: T"

        monkeypatch.setattr(playback_module, "_notification_message", recording_message)

        host = self._host()
        PlaybackMixin._on_track_change.__get__(host)(track)

        assert calls == [(track, "{title} — {artist}")]
        host.notify.assert_called_once_with("Now: T", timeout=4)

    def test_a_disabled_setting_never_consults_the_engine(self, monkeypatch):
        engine = MagicMock()
        monkeypatch.setattr(playback_module, "_notification_message", engine)

        host = self._host(enabled=False)
        PlaybackMixin._on_track_change.__get__(host)({"video_id": "vid1"})

        engine.assert_not_called()
        host.notify.assert_not_called()

    def test_a_raising_engine_is_swallowed_by_the_broad_catch(self, monkeypatch):
        def boom(*args):
            raise RuntimeError("boom")

        monkeypatch.setattr(playback_module, "_notification_message", boom)

        host = self._host()
        PlaybackMixin._on_track_change.__get__(host)({"video_id": "vid1"})

        host.notify.assert_not_called()


class TestPrefetchWiring:
    """``_prefetch_next_track`` routes through the prefetch engine name."""

    def test_the_engines_id_travels_to_the_resolver(self, monkeypatch):
        monkeypatch.setattr(playback_module, "_prefetch_next_id", lambda peeked: "abc123")

        host = MagicMock()
        host.stream_resolver = MagicMock()
        host.queue.peek_next = MagicMock(return_value={"video_id": "abc123"})

        PlaybackMixin._prefetch_next_track.__get__(host)()

        host.run_worker.assert_called_once()
        assert host.run_worker.call_args.kwargs == {"group": "prefetch", "exclusive": True}
        host.stream_resolver.prefetch.assert_called_once_with("abc123")

    def test_a_none_answer_schedules_no_worker(self, monkeypatch):
        monkeypatch.setattr(playback_module, "_prefetch_next_id", lambda peeked: None)

        host = MagicMock()
        host.stream_resolver = MagicMock()
        host.queue.peek_next = MagicMock(return_value=None)

        PlaybackMixin._prefetch_next_track.__get__(host)()

        host.run_worker.assert_not_called()


class TestRefillQueueWiring:
    """``_refill_queue`` routes through the refill engine name."""

    def _host(self, *, workers=()):
        host = MagicMock()
        host.workers = list(workers)
        host.queue = SimpleNamespace(
            repeat_mode="off",
            remaining_tracks=2,
            tracks=[{"i": 0}, {"i": 1}],
            real_index=1,
        )
        host.settings.playback.autoplay = True
        host.player = SimpleNamespace(current_track=None)
        return host

    def test_the_engine_is_consulted_with_the_very_queue_and_player(self, monkeypatch):
        calls: list[tuple] = []

        def recording_plan(*args):
            calls.append(args)
            return [{"i": 0}, {"i": 1}]

        monkeypatch.setattr(playback_module, "_refill_plan", recording_plan)

        host = self._host()
        PlaybackMixin._refill_queue.__get__(host)()

        assert calls == [("off", True, 2, False, host.queue, host.player)]
        host.run_worker.assert_called_once()
        assert host.run_worker.call_args.kwargs == {"group": "queue_extend", "exclusive": True}

    def test_the_worker_walk_feeds_the_running_flag(self, monkeypatch):
        seen: list[bool] = []

        def recording_plan(*args):
            seen.append(args[3])
            return None

        monkeypatch.setattr(playback_module, "_refill_plan", recording_plan)

        host = self._host(
            workers=[
                SimpleNamespace(group="other", is_running=True),
                SimpleNamespace(group="queue_extend", is_running=True),
            ]
        )
        PlaybackMixin._refill_queue.__get__(host)()

        assert seen == [True]
        host.run_worker.assert_not_called()

    def test_a_none_answer_schedules_no_worker(self, monkeypatch):
        monkeypatch.setattr(playback_module, "_refill_plan", lambda *args: None)

        host = self._host()
        PlaybackMixin._refill_queue.__get__(host)()

        host.run_worker.assert_not_called()


# ── The local-history wiring ───────────────────────────────────────────────


def _listen_host(**overrides):
    """A host carrying the attrs ``_log_local_listen`` reads."""
    host = MagicMock()
    host.history = MagicMock()
    host.history.update_play_listened_seconds = AsyncMock()
    host.history.log_play = AsyncMock()
    host.player = SimpleNamespace(position=42.0)
    host._track_start_position = 2.0
    host.settings.playback.history_min_listen_seconds = 30
    for key, value in overrides.items():
        setattr(host, key, value)
    return host


class TestLogLocalListenWiring:
    """``_log_local_listen`` routes through the finalize engine name."""

    async def test_the_engine_is_consulted_with_the_very_claim_and_player(self, monkeypatch):
        calls: list[tuple] = []
        claim = _claim_stand_in()
        host = _listen_host(_local_history_claim=claim)

        def recording_plan(*args):
            calls.append(args)
            return {"action": "no_write", "listened": 0, "mark_finalized": True}

        monkeypatch.setattr(playback_module, "_local_listen_plan", recording_plan)

        await PlaybackMixin._log_local_listen.__get__(host)({"video_id": "v"})

        assert calls == [(claim, "v", host.player, 2.0)]
        assert claim.finalized is True  # consumed even on a nothing-to-write finish
        host.history.log_play.assert_not_awaited()

    async def test_a_consumed_verdict_writes_nothing(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_local_listen_plan",
            lambda *args: {"action": "consumed", "mark_finalized": False},
        )
        claim = _claim_stand_in()
        host = _listen_host(_local_history_claim=claim)

        await PlaybackMixin._log_local_listen.__get__(host)({"video_id": "v"})

        assert claim.finalized is False
        host.history.log_play.assert_not_awaited()
        host.history.update_play_listened_seconds.assert_not_awaited()

    async def test_an_update_row_verdict_updates_the_claims_row(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_local_listen_plan",
            lambda *args: {
                "action": "update_row",
                "listened": 40,
                "mark_finalized": True,
                "row_id": 9,
            },
        )
        claim = _claim_stand_in(row_id=9)
        host = _listen_host(_local_history_claim=claim)

        await PlaybackMixin._log_local_listen.__get__(host)({"video_id": "v"})

        assert claim.finalized is True
        host.history.update_play_listened_seconds.assert_awaited_once_with(9, 40)
        host.history.log_play.assert_not_awaited()

    async def test_a_stash_pending_verdict_parks_the_duration(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_local_listen_plan",
            lambda *args: {"action": "stash_pending", "listened": 40, "mark_finalized": True},
        )
        claim = _claim_stand_in(insert_started=True)
        host = _listen_host(_local_history_claim=claim)

        await PlaybackMixin._log_local_listen.__get__(host)({"video_id": "v"})

        assert claim.pending_seconds == 40
        host.history.log_play.assert_not_awaited()

    async def test_a_direct_log_verdict_logs_with_the_clamped_minimum(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_local_listen_plan",
            lambda *args: {"action": "direct_log", "listened": 40, "mark_finalized": True},
        )
        claim = _claim_stand_in()
        host = _listen_host(_local_history_claim=claim)
        host._history_min_listen_seconds = MagicMock(return_value=30)
        track = {"video_id": "v"}

        await PlaybackMixin._log_local_listen.__get__(host)(track)

        assert claim.finalized is True
        host.history.log_play.assert_awaited_once_with(
            track=track, listened_seconds=40, source="tui", min_listen_seconds=30
        )

    async def test_an_unowned_direct_log_leaves_the_claim_alone(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_local_listen_plan",
            lambda *args: {"action": "direct_log", "listened": 40, "mark_finalized": False},
        )
        claim = _claim_stand_in()
        host = _listen_host(_local_history_claim=claim)

        await PlaybackMixin._log_local_listen.__get__(host)({"video_id": "v"})

        assert claim.finalized is False
        host.history.log_play.assert_awaited_once()


class TestInsertFinalizeWiring:
    """``_insert_local_history_play`` routes through the plan engine name."""

    def _host(self, claim):
        host = MagicMock()
        host.history = MagicMock()
        host.history.log_play = AsyncMock(return_value=99)
        host.history.update_play_listened_seconds = AsyncMock()
        host._play_generation = 3
        host._optimistic_local_history_add = MagicMock()
        host._local_history_claim = claim
        return host

    async def test_the_engine_is_consulted_with_the_claim_and_generation(self, monkeypatch):
        calls: list[tuple] = []
        claim = _claim_stand_in(generation=3)

        def recording_plan(*args):
            calls.append(args)
            return {
                "had_pending": False,
                "pending": None,
                "apply_pending": False,
                "optimistic_add": True,
            }

        monkeypatch.setattr(playback_module, "_insert_finalize_plan", recording_plan)

        host = self._host(claim)
        await PlaybackMixin._insert_local_history_play.__get__(host)(claim, 40)

        assert calls == [(claim, 40, 3)]
        assert claim.row_id == 99  # the insert's own row id lands on the claim
        host._optimistic_local_history_add.assert_called_once_with(claim.track)

    async def test_a_pending_beating_the_row_is_applied_to_the_row(self, monkeypatch):
        monkeypatch.setattr(
            playback_module,
            "_insert_finalize_plan",
            lambda *args: {
                "had_pending": True,
                "pending": 50,
                "apply_pending": True,
                "optimistic_add": False,
            },
        )
        claim = _claim_stand_in(pending_seconds=50)

        host = self._host(claim)
        await PlaybackMixin._insert_local_history_play.__get__(host)(claim, 40)

        assert claim.pending_seconds is None  # the stash is consumed
        host.history.update_play_listened_seconds.assert_awaited_once_with(99, 50)
        host._optimistic_local_history_add.assert_not_called()

    async def test_a_failed_insert_never_consults_the_plan(self, monkeypatch):
        engine = MagicMock()
        monkeypatch.setattr(playback_module, "_insert_finalize_plan", engine)

        claim = _claim_stand_in(generation=3)
        host = self._host(claim)
        host.history.log_play = AsyncMock(return_value=None)

        await PlaybackMixin._insert_local_history_play.__get__(host)(claim, 40)

        engine.assert_not_called()
        assert claim.row_id is None
        host._optimistic_local_history_add.assert_not_called()


class TestHistoryTimerPairWiring:
    """``_history_min_listen_seconds`` / ``_history_timer_delay`` route
    through their engine names, the delay chained through the clamp."""

    def test_the_clamp_is_consulted_with_the_settings_value(self, monkeypatch):
        calls: list[tuple] = []

        def recording_clamp(*args):
            calls.append(args)
            return 30

        monkeypatch.setattr(playback_module, "_min_listen_seconds", recording_clamp)

        host = MagicMock()
        host.settings.playback.history_min_listen_seconds = 30

        assert PlaybackMixin._history_min_listen_seconds.__get__(host)() == 30
        assert calls == [(30,)]

    def test_the_delay_is_consulted_with_the_clamps_answer(self, monkeypatch):
        calls: list[tuple] = []

        def recording_delay(*args):
            calls.append(args)
            return 12.5

        monkeypatch.setattr(playback_module, "_timer_arm_delay", recording_delay)

        host = MagicMock()
        host._history_min_listen_seconds = MagicMock(return_value=12)

        assert PlaybackMixin._history_timer_delay.__get__(host)() == 12.5
        assert calls == [(12,)]


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

# Attempt-gate cases: [attempt, play_generation, handled_attempt,
# active_stop_generation, queue_stop_generation] — scalars only.
_GATE_PARITY_CASES = [
    [3, 3, 2, 0, 0],
    [3, 3, 3, 0, 0],
    [3, 4, 2, 0, 0],
    ["3", 3, 2, 0, 0],
    [3.0, 3, 2, 0, 0],
    [True, 1, 0, 0, 0],
    [None, 3, 2, 0, 0],
    [3, 3, 2, 1, 1],
    [3, 3, 2, 1, 2],
    [3, 3, "x", 0, 0],
    [3, "3", 2, 0, 0],
]

# Poll-reset cases: [position, consecutive_failures, recovery_generation,
# player desc (null | [is_playing, current_attempt] | "boom_is_playing" |
# "boom_current_attempt"), play_generation]. The boom spellings pin the
# arms that must never read the player — a read would surface as the
# stand-in's own error instead of a verdict.
_POLL_PARITY_CASES = [
    [5.0, 2, None, [True, 4], 4],
    [5.0, 0, 7, [True, 4], 4],
    [5.0, 0, None, [True, 4], 4],
    [0.0, 2, None, [True, 4], 4],
    [5.0, 2, None, [False, 4], 4],
    [5.0, 2, None, [True, 3], 4],
    [5, 2, None, [True, 4], 4],
    [5.0, 2, None, "boom_is_playing", 4],
    [0.0, 2, None, "boom_is_playing", 4],
    [5.0, 0, None, "boom_current_attempt", 4],
    ["x", 2, None, [True, 4], 4],
]

# Notification cases: [track, fmt]. Malformed templates, unknown
# placeholders and non-str templates pin the fallback and the error
# shapes; the conversion/spec cases pin CPython's own format.
_NOTIF_PARITY_CASES = [
    [{"title": "T", "artist": "A", "album": "AL"}, "{title} — {artist} ({album})"],
    [{"title": "T", "artist": "A"}, "{title}|{album}|"],
    [{}, "{title} by {artist}"],
    [{"title": "T", "artist": "A"}, "{title} {nope}"],
    [{"title": "T", "artist": "A"}, "{title"],
    [{"title": 7, "artist": None}, "{nope}"],
    [{"title": "T", "artist": "A"}, "{title!r:>6}"],
    [{"title": "T", "artist": "A"}, "{album}"],
    [{"title": "T", "artist": "A"}, 42],
    [{"title": "T", "artist": "A"}, None],
]

# Prefetch cases: [peeked] — the head stand-in or a non-dict scalar.
_PREFETCH_PARITY_CASES = [
    [None],
    [{}],
    [{"video_id": ""}],
    [{"video_id": "abc"}],
    [{"video_id": 7}],
    [{"other": 1}],
    [5],
]

# Refill cases: [repeat_mode, autoplay, remaining_tracks, extend_running,
# queue desc ([tracks, real_index]), player desc (null | [current_track])].
_REFILL_PARITY_CASES = [
    ["one", True, 2, False, [[{"i": 0}, {"i": 1}, {"i": 2}], 1], None],
    ["off", False, 2, False, [[{"i": 0}], 0], None],
    ["off", True, 4, False, [[{"i": 0}], 0], None],
    ["off", True, 2, True, [[{"i": 0}], 0], None],
    ["off", True, 3, False, [[{"i": 0}, {"i": 1}, {"i": 2}], 1], None],
    ["off", True, 2, False, [[{"i": i} for i in range(8)], 6], None],
    ["off", True, 2, False, [[{"i": 0}], 0], None],
    ["off", True, 2, False, [[], -1], [{"video_id": "v"}]],
    ["off", True, 2, False, [[], -1], [None]],
    ["off", True, 2, False, [[], -1], None],
    ["off", True, 2, False, [[{"i": i} for i in range(3)], -1], [{"video_id": "v"}]],
    ["off", True, "x", False, [[], 0], None],
]

# Local-listen cases: [claim desc (null | [video_id, finalized, row_id,
# insert_started]), video_id, position (number | "inf"), track_start].
_LOCAL_PARITY_CASES = [
    [None, "v", 42.0, 2.0],
    [["v", False, None, False], "v", 42.0, 2.0],
    [["other", False, None, False], "v", 42.0, 2.0],
    [["v", True, None, False], "v", 42.0, 2.0],
    [["v", False, 9, False], "v", 42.0, 2.0],
    [["v", False, None, True], "v", 42.0, 2.0],
    [["v", False, None, False], "v", 2.0, 2.0],
    [["v", False, None, False], "v", 1.5, 2.0],
    [None, "v", 2.0, 2.0],
    [None, "v", 42.5, 2.0],
    [None, "v", "x", 2.0],
    [None, "v", "inf", 2.0],
]

# Insert-finalize cases: [claim desc ([pending_seconds, generation]),
# listened, play_generation].
_INSERT_PARITY_CASES = [
    [[50, 7], 40, 3],
    [[30, 7], 40, 3],
    [[40, 7], 40, 3],
    [[None, 3], 40, 3],
    [[None, 4], 40, 3],
    [["x", 3], 40, 3],
]

# Min-listen cases: [raw configured value].
_MIN_PARITY_CASES = [
    [30],
    [0],
    [-5],
    [2.7],
    [-2.7],
    ["30"],
    ["2_0"],
    [True],
    [False],
    [None],
    ["x"],
]

# Timer-arm cases: [min_listen_seconds, patched poll interval (None → the
# module default)].
_TIMER_PARITY_CASES = [
    [0, None],
    [30, None],
    [10.5, None],
    ["30", None],
    [0, 2.5],
    [30, 2.5],
    ["x", None],
    [None, None],
]

_CHILD_PROG = """
import json
import sys
from types import SimpleNamespace

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


class BoomIsPlaying:
    @property
    def is_playing(self):
        raise RuntimeError("must not read is_playing on this arm")


class BoomCurrentAttempt:
    is_playing = True

    @property
    def current_attempt(self):
        raise RuntimeError("must not read current_attempt on this arm")


def make_poll_player(desc):
    if desc is None:
        return None
    if desc == "boom_is_playing":
        return BoomIsPlaying()
    if desc == "boom_current_attempt":
        return BoomCurrentAttempt()
    return SimpleNamespace(is_playing=desc[0], current_attempt=desc[1])


def make_position(desc):
    if desc == "inf":
        return float("inf")
    return desc


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


def gate_outcome(case):
    try:
        result = _playback._attempt_gate(*case)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def poll_outcome(case):
    position, failures, recovery, player_desc, generation = case
    try:
        result = _playback._poll_failure_reset(
            position, failures, recovery, make_poll_player(player_desc), generation
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def notif_outcome(case):
    track, fmt = case
    try:
        result = _playback._notification_message(track, fmt)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def prefetch_outcome(case):
    (peeked,) = case
    try:
        result = _playback._prefetch_next_id(peeked)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def refill_outcome(case):
    repeat_mode, autoplay, remaining, extend, queue_desc, player_desc = case
    queue = SimpleNamespace(tracks=queue_desc[0], real_index=queue_desc[1])
    player = None if player_desc is None else SimpleNamespace(current_track=player_desc[0])
    try:
        result = _playback._refill_plan(repeat_mode, autoplay, remaining, extend, queue, player)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def local_outcome(case):
    claim_desc, video_id, position_desc, track_start = case
    claim = (
        None
        if claim_desc is None
        else SimpleNamespace(
            video_id=claim_desc[0],
            finalized=claim_desc[1],
            row_id=claim_desc[2],
            insert_started=claim_desc[3],
        )
    )
    player = SimpleNamespace(position=make_position(position_desc))
    try:
        result = _playback._local_listen_plan(claim, video_id, player, track_start)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def insert_outcome(case):
    claim_desc, listened, generation = case
    claim = SimpleNamespace(pending_seconds=claim_desc[0], generation=claim_desc[1])
    try:
        result = _playback._insert_finalize_plan(claim, listened, generation)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def min_outcome(case):
    (value,) = case
    try:
        result = _playback._min_listen_seconds(value)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def timer_outcome(case):
    min_listen, poll_patch = case
    saved = _playback._YTM_HISTORY_POLL_SECONDS
    if poll_patch is not None:
        _playback._YTM_HISTORY_POLL_SECONDS = poll_patch
    try:
        result = _playback._timer_arm_delay(min_listen)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        _playback._YTM_HISTORY_POLL_SECONDS = saved
    return {"ok": result}


FAMILIES = {
    "failure": failure_outcome,
    "step": step_outcome,
    "gate": gate_outcome,
    "poll": poll_outcome,
    "notif": notif_outcome,
    "prefetch": prefetch_outcome,
    "refill": refill_outcome,
    "local": local_outcome,
    "insert": insert_outcome,
    "min": min_outcome,
    "timer": timer_outcome,
}

payload = json.load(sys.stdin)
json.dump(
    {family: [FAMILIES[family](case) for case in payload[family]] for family in payload},
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


def _parent_poll_player(desc):
    """The parent-side twin of the child program's poll-player builder."""
    if desc is None:
        return None
    if desc == "boom_is_playing":
        return _BoomIsPlaying()
    if desc == "boom_current_attempt":
        return _BoomCurrentAttempt()
    return SimpleNamespace(is_playing=desc[0], current_attempt=desc[1])


def _parent_position(desc):
    """The parent-side twin of the child program's position builder."""
    if desc == "inf":
        return float("inf")
    return desc


def _native_new_family_outcomes(family):
    """The parent-side outcomes for the nine new engine families."""

    def gate(case):
        try:
            return {"ok": playback_module._attempt_gate(*case)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def poll(case):
        position, failures, recovery, player_desc, generation = case
        try:
            return {
                "ok": playback_module._poll_failure_reset(
                    position, failures, recovery, _parent_poll_player(player_desc), generation
                )
            }
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def notif(case):
        track, fmt = case
        try:
            return {"ok": playback_module._notification_message(track, fmt)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def prefetch(case):
        (peeked,) = case
        try:
            return {"ok": playback_module._prefetch_next_id(peeked)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def refill(case):
        repeat_mode, autoplay, remaining, extend, queue_desc, player_desc = case
        queue = SimpleNamespace(tracks=queue_desc[0], real_index=queue_desc[1])
        player = None if player_desc is None else SimpleNamespace(current_track=player_desc[0])
        try:
            return {
                "ok": playback_module._refill_plan(
                    repeat_mode, autoplay, remaining, extend, queue, player
                )
            }
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def local(case):
        claim_desc, video_id, position_desc, track_start = case
        claim = (
            None
            if claim_desc is None
            else SimpleNamespace(
                video_id=claim_desc[0],
                finalized=claim_desc[1],
                row_id=claim_desc[2],
                insert_started=claim_desc[3],
            )
        )
        player = SimpleNamespace(position=_parent_position(position_desc))
        try:
            return {"ok": playback_module._local_listen_plan(claim, video_id, player, track_start)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def insert(case):
        claim_desc, listened, generation = case
        claim = SimpleNamespace(pending_seconds=claim_desc[0], generation=claim_desc[1])
        try:
            return {"ok": playback_module._insert_finalize_plan(claim, listened, generation)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def minimum(case):
        (value,) = case
        try:
            return {"ok": playback_module._min_listen_seconds(value)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}

    def timer(case):
        min_listen, poll_patch = case
        saved = playback_module._YTM_HISTORY_POLL_SECONDS
        if poll_patch is not None:
            playback_module._YTM_HISTORY_POLL_SECONDS = poll_patch
        try:
            return {"ok": playback_module._timer_arm_delay(min_listen)}
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            playback_module._YTM_HISTORY_POLL_SECONDS = saved

    builders = {
        "gate": gate,
        "poll": poll,
        "notif": notif,
        "prefetch": prefetch,
        "refill": refill,
        "local": local,
        "insert": insert,
        "min": minimum,
        "timer": timer,
    }
    cases = {
        "gate": _GATE_PARITY_CASES,
        "poll": _POLL_PARITY_CASES,
        "notif": _NOTIF_PARITY_CASES,
        "prefetch": _PREFETCH_PARITY_CASES,
        "refill": _REFILL_PARITY_CASES,
        "local": _LOCAL_PARITY_CASES,
        "insert": _INSERT_PARITY_CASES,
        "min": _MIN_PARITY_CASES,
        "timer": _TIMER_PARITY_CASES,
    }
    return [builders[family](case) for case in cases[family]]


_FAMILY_ENGINE_NAMES = {
    "gate": "_attempt_gate",
    "poll": "_poll_failure_reset",
    "notif": "_notification_message",
    "prefetch": "_prefetch_next_id",
    "refill": "_refill_plan",
    "local": "_local_listen_plan",
    "insert": "_insert_finalize_plan",
    "min": "_min_listen_seconds",
    "timer": "_timer_arm_delay",
}


def _assert_family_parity(family):
    """Run the child for one family and compare every case."""
    cases = {
        "gate": _GATE_PARITY_CASES,
        "poll": _POLL_PARITY_CASES,
        "notif": _NOTIF_PARITY_CASES,
        "prefetch": _PREFETCH_PARITY_CASES,
        "refill": _REFILL_PARITY_CASES,
        "local": _LOCAL_PARITY_CASES,
        "insert": _INSERT_PARITY_CASES,
        "min": _MIN_PARITY_CASES,
        "timer": _TIMER_PARITY_CASES,
    }[family]
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps({family: cases}),
        capture_output=True,
        text=True,
        check=True,
    )
    fallback = json.loads(result.stdout)[family]
    native = _native_new_family_outcomes(family)
    engine_name = _FAMILY_ENGINE_NAMES[family]
    for case, n, f in zip(cases, native, fallback):
        assert n == f, (
            f"native and fallback {engine_name} differ for {case!r}:\n"
            f"native:   {n!r}\n"
            f"fallback: {f!r}"
        )


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

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_attempt_gate_matches_the_fallback(self):
        _assert_family_parity("gate")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_poll_failure_reset_matches_the_fallback(self):
        _assert_family_parity("poll")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_notification_message_matches_the_fallback(self):
        _assert_family_parity("notif")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_prefetch_gate_matches_the_fallback(self):
        _assert_family_parity("prefetch")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_refill_plan_matches_the_fallback(self):
        _assert_family_parity("refill")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_local_listen_plan_matches_the_fallback(self):
        _assert_family_parity("local")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_insert_finalize_plan_matches_the_fallback(self):
        _assert_family_parity("insert")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_min_listen_clamp_matches_the_fallback(self):
        _assert_family_parity("min")

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_timer_arm_delay_matches_the_fallback(self):
        _assert_family_parity("timer")
