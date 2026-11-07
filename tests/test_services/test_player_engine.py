"""Tests for the mpv-wrapper glue engines.

The pure cores of ``services/player.py`` run with two
interchangeable backends: the pure-Python transcriptions in the module
and, when the Rust extension is built, the native implementations (the
tail rebinding for most of them, the import-time binding plus call-time
check for ``_find_brew_libmpv``). ``_find_brew_libmpv`` is the
Homebrew-prefix libmpv discovery walk behind the import-time fallback
find; ``_end_file_verdict`` is the event-matching gate behind
``_end_file_data``'s handling of mpv's end-file callbacks;
``_clamp_seek`` is the ``[0, duration]`` target clamp behind
``_clamp_seek_target``; ``_position_due`` is the 0.4 s throttle behind
``_on_time_pos_change``'s POSITION_CHANGE dispatches;
``_command_current`` is the supersession predicate behind
``_dispatch``'s still-current check and ``_finish_load``'s acceptance
check; ``_dispatch_arm`` is the six-way branch behind ``_dispatch``'s
per-callback walk; and ``_mpv_log_level`` is the level map behind
``_on_mpv_log``'s routing of mpv's log events into Python logging. The
clock-reading engine resolves ``time.monotonic()`` through the module's
own ``time`` name at call time — a frozen or patched clock steers both
backends identically, which the tests below use for determinism and
the parity class pins as the contract; the log-level engine reads the
logging constants through the player module's own ``logging`` name at
call time — a patched or customised attribute steers both backends
identically too.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked, the same frozen clock installed and the same discovery
fixtures patched) case for case — error shapes included — so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ytm_player.services.player import (
    _clamp_seek,
    _clamp_volume,
    _command_current,
    _dispatch_arm,
    _end_file_verdict,
    _find_brew_libmpv,
    _mpv_log_level,
    _position_due,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_FROZEN_NOW = 10_000.0


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze player.py's clock at 10000 — the engine's own module-clock
    read makes both backends see the frozen value."""
    from ytm_player.services import player

    monkeypatch.setattr(player, "time", SimpleNamespace(monotonic=lambda: _FROZEN_NOW))
    return _FROZEN_NOW


class TestFindBrewLibmpv:
    """The Homebrew libmpv discovery walk."""

    @pytest.fixture
    def brew_dirs(self, monkeypatch):
        """Patch _BREW_LIB_DIRS through the module's own name — the
        engine reads it there, so both backends steer identically."""
        from ytm_player.services import player

        holder = SimpleNamespace(dirs=())

        def install(dirs):
            holder.dirs = tuple(dirs)
            monkeypatch.setattr(player, "_BREW_LIB_DIRS", holder.dirs)

        install(holder.dirs)
        return install

    def test_returns_none_when_no_prefix_matches(self, brew_dirs, tmp_path):
        brew_dirs((tmp_path / "missing",))
        assert _find_brew_libmpv() is None

    def test_finds_linux_so(self, monkeypatch, brew_dirs, tmp_path):
        # The .so pattern pair is the non-darwin arm; pin the platform
        # so the test holds on macOS runners too (the dylib test pins
        # "darwin" the same way for the other arm).
        monkeypatch.setattr(sys, "platform", "linux")
        lib = tmp_path / "libmpv.so.2"
        lib.touch()
        brew_dirs((tmp_path,))
        assert _find_brew_libmpv() == str(lib)

    def test_finds_macos_dylib(self, monkeypatch, brew_dirs, tmp_path):
        monkeypatch.setattr(sys, "platform", "darwin")
        lib = tmp_path / "libmpv.2.dylib"
        lib.touch()
        brew_dirs((tmp_path,))
        assert _find_brew_libmpv() == str(lib)

    def test_prefers_the_unversioned_name(self, monkeypatch, brew_dirs, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        # The pattern order decides — libmpv.so before libmpv.so.*.
        (tmp_path / "libmpv.so.2").touch()
        plain = tmp_path / "libmpv.so"
        plain.touch()
        brew_dirs((tmp_path,))
        assert _find_brew_libmpv() == str(plain)

    def test_ignores_unrelated_files(self, brew_dirs, tmp_path):
        (tmp_path / "libfoo.so").touch()
        (tmp_path / "libmpv.txt").touch()
        brew_dirs((tmp_path,))
        assert _find_brew_libmpv() is None

    def test_skips_non_dir_entries(self, monkeypatch, brew_dirs, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        a_file = tmp_path / "afile"
        a_file.touch()
        real = tmp_path / "real"
        real.mkdir()
        (real / "libmpv.so").touch()
        brew_dirs((a_file, real))
        assert _find_brew_libmpv() == str(real / "libmpv.so")

    def test_the_first_prefix_with_a_hit_wins(self, monkeypatch, brew_dirs, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "libmpv.so.1").touch()
        (second / "libmpv.so").touch()
        brew_dirs((second, first))
        assert _find_brew_libmpv() == str(second / "libmpv.so")

    def test_an_oserror_from_a_prefix_continues_to_the_next(self, monkeypatch, brew_dirs, tmp_path):
        monkeypatch.setattr(sys, "platform", "linux")

        def _raising_glob(_pattern):
            raise OSError("probe failed")

        raising = SimpleNamespace(is_dir=lambda: True, glob=_raising_glob)
        real = tmp_path / "real"
        real.mkdir()
        (real / "libmpv.so").touch()
        brew_dirs((raising, real))
        assert _find_brew_libmpv() == str(real / "libmpv.so")

    def test_a_non_oserror_from_a_prefix_propagates(self, brew_dirs):
        def _exploding_glob(_pattern):
            raise ValueError("boom")

        exploding = SimpleNamespace(is_dir=lambda: True, glob=_exploding_glob)
        brew_dirs((exploding,))
        with pytest.raises(ValueError, match="boom"):
            _find_brew_libmpv()


class TestEndFileVerdict:
    """The event-matching gate behind _end_file_data."""

    def test_an_end_from_a_foreign_instance_is_dropped(self):
        assert _end_file_verdict(object(), object(), False, 10, 10, {"t": 1}, 0) == "drop"

    def test_the_live_instance_is_not_dropped(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 10, {"t": 1}, 0) == "track_end"

    def test_loading_parks_even_a_matching_end(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, True, 10, 10, {"t": 1}, 0) == "park"

    def test_a_non_int_entry_id_is_ignored(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, "10", 10, {"t": 1}, 0) == "ignore"
        assert _end_file_verdict(mpv, mpv, False, None, 10, {"t": 1}, 0) == "ignore"

    def test_a_bool_entry_id_is_an_int_to_isinstance(self):
        # isinstance(True, int) is True and True == 1 — the reference's
        # own gate admits it.
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, True, 1, {"t": 1}, 0) == "track_end"

    def test_a_stale_entry_id_is_ignored(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 11, {"t": 1}, 0) == "ignore"
        assert _end_file_verdict(mpv, mpv, False, 10, None, {"t": 1}, 0) == "ignore"

    def test_no_current_track_is_ignored(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 10, None, 0) == "ignore"

    def test_a_non_terminal_reason_is_ignored(self):
        mpv = object()
        for reason in (2, 3, "0", "eof", None):
            assert _end_file_verdict(mpv, mpv, False, 10, 10, {"t": 1}, reason) == "ignore"

    def test_reason_zero_is_the_natural_end(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 10, {"t": 1}, 0) == "track_end"

    def test_reason_four_is_the_error_end(self):
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 10, {"t": 1}, 4) == "error"

    def test_a_float_four_is_the_error_end(self):
        # 4.0 == 4 — the membership and the fork both go through ==.
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 10, {"t": 1}, 4.0) == "error"

    def test_a_false_reason_is_the_natural_end(self):
        # False == 0 — terminal; then False == 4 is False.
        mpv = object()
        assert _end_file_verdict(mpv, mpv, False, 10, 10, {"t": 1}, False) == "track_end"


class TestClampSeek:
    """The seek-target clamp."""

    def test_an_unknown_duration_leaves_the_target(self):
        assert _clamp_seek(50.0, 0.0) == 50.0
        assert _clamp_seek(50.0, -3.0) == 50.0

    def test_a_positive_duration_caps_the_target(self):
        assert _clamp_seek(200.0, 120.0) == 120.0

    def test_within_the_duration_is_unchanged(self):
        assert _clamp_seek(30.0, 120.0) == 30.0

    def test_the_upper_boundary_keeps_the_target(self):
        assert _clamp_seek(120.0, 120.0) == 120.0

    def test_negative_targets_clamp_to_zero(self):
        assert _clamp_seek(-5.0, 120.0) == 0.0
        assert _clamp_seek(-5.0, 0.0) == 0.0

    def test_a_nan_duration_does_not_cap(self):
        assert _clamp_seek(200.0, math.nan) == 200.0

    def test_a_nan_target_clamps_to_zero(self):
        # min(nan, 100) keeps nan (100 < nan is False); max(0.0, nan)
        # keeps the 0.0 literal (nan > 0.0 is False).
        assert _clamp_seek(math.nan, 100.0) == 0.0

    def test_int_operands_pass_through_as_ints(self):
        assert _clamp_seek(50, 0) == 50
        assert isinstance(_clamp_seek(50, 0), int)

    def test_a_non_numeric_duration_raises_the_comparison_error(self):
        with pytest.raises(TypeError, match="'>' not supported"):
            _clamp_seek(1.0, "long")

    def test_a_non_numeric_target_raises_the_min_error(self):
        with pytest.raises(TypeError, match="'<' not supported"):
            _clamp_seek("far", 100.0)


class TestClampVolume:
    """The volume clamp."""

    def test_a_negative_level_clamps_to_zero(self):
        assert _clamp_volume(-10) == 0

    def test_a_level_above_one_hundred_clamps_to_one_hundred(self):
        assert _clamp_volume(150) == 100

    def test_the_boundaries_pass_through(self):
        assert _clamp_volume(0) == 0
        assert _clamp_volume(100) == 100

    def test_an_in_range_level_is_unchanged(self):
        assert _clamp_volume(37) == 37

    def test_ints_stay_ints(self):
        out = _clamp_volume(99)
        assert out == 99 and isinstance(out, int)

    def test_a_bool_level_answers_the_bool(self):
        # min(100, True) keeps the bool object itself; max(0, True)
        # keeps it too — the operand identity CPython's walks preserve.
        assert _clamp_volume(True) is True
        # False < 100 keeps the bool for the min, but False > 0 is
        # False so the max answers its own 0 literal.
        assert _clamp_volume(False) == 0

    def test_a_non_numeric_level_raises_the_comparison_error(self):
        with pytest.raises(TypeError, match="'<' not supported"):
            _clamp_volume("loud")


class TestPositionDue:
    """The position-dispatch throttle."""

    def test_within_the_window_is_throttled(self, frozen_clock):
        assert _position_due(_FROZEN_NOW - 0.25) is None
        assert _position_due(_FROZEN_NOW) is None
        # A last-dispatch ahead of the clock (negative elapsed) is
        # throttled too.
        assert _position_due(_FROZEN_NOW + 100.0) is None

    def test_past_the_window_is_due(self, frozen_clock):
        assert _position_due(_FROZEN_NOW - 0.5) == _FROZEN_NOW
        assert _position_due(_FROZEN_NOW - 10_000.0) == _FROZEN_NOW
        assert _position_due(0.0) == _FROZEN_NOW

    def test_the_returned_value_is_the_clock_read_itself(self, frozen_clock):
        # The caller stamps the returned value as the new
        # last-dispatch — it must be the very `now` the gate read.
        assert _position_due(0.0) == _FROZEN_NOW

    def test_a_non_numeric_last_dispatch_raises_the_subtraction_error(self, frozen_clock):
        with pytest.raises(TypeError, match="unsupported operand"):
            _position_due("recently")


class TestCommandCurrent:
    """The command-supersession gate."""

    def test_a_none_command_is_unconditionally_current(self):
        assert _command_current(None, 99, None) is True

    def test_a_matching_generation_with_a_track_is_current(self):
        assert _command_current(7, 7, {"video_id": "A"}) is True

    def test_a_matching_generation_without_a_track_is_stale(self):
        assert _command_current(7, 7, None) is False

    def test_a_superseded_generation_is_stale(self):
        assert _command_current(7, 8, {"video_id": "A"}) is False

    def test_equality_not_identity_decides_the_generation(self):
        # command == generation through CPython's ==, so equal-valued
        # non-identical objects match.
        assert _command_current(True, 1, {"t": 1}) is True


class TestDispatchArm:
    """The event-dispatch arm — the six-way branch behind ``_dispatch``'s
    per-callback walk."""

    def test_the_full_arm_matrix(self):
        # (loop_live, is_coroutine, has_context) → arm, for every
        # combination the walk can see.
        assert _dispatch_arm(True, True, True) == "async_ctx"
        assert _dispatch_arm(True, True, False) == "async"
        assert _dispatch_arm(True, False, True) == "sync_ctx"
        assert _dispatch_arm(True, False, False) == "sync"
        assert _dispatch_arm(False, True, True) == "drop"
        assert _dispatch_arm(False, True, False) == "drop"
        assert _dispatch_arm(False, False, True) == "direct"
        assert _dispatch_arm(False, False, False) == "direct"

    def test_a_dead_loop_never_schedules(self):
        # Whatever the coroutine/context shape, a dead loop only drops
        # async callbacks and runs sync ones inline — the context is
        # irrelevant without a loop to schedule under.
        for is_coroutine in (True, False):
            for has_context in (True, False):
                arm = _dispatch_arm(False, is_coroutine, has_context)
                assert arm in ("drop", "direct")

    def test_a_live_loop_schedules_everything(self):
        for is_coroutine in (True, False):
            for has_context in (True, False):
                arm = _dispatch_arm(True, is_coroutine, has_context)
                assert arm in ("async_ctx", "async", "sync_ctx", "sync")


class TestMpvLogLevel:
    """The mpv log-event level routing — the map behind ``_on_mpv_log``."""

    def test_the_four_mapped_levels(self):
        import logging

        assert _mpv_log_level("fatal") == logging.CRITICAL
        assert _mpv_log_level("error") == logging.ERROR
        assert _mpv_log_level("warn") == logging.WARNING
        assert _mpv_log_level("info") == logging.INFO

    def test_every_other_string_answers_debug(self):
        import logging

        # The python-mpv strings filtered out at the C side (v / debug /
        # trace), the empty string and a case-mismatched spelling — the
        # reference's dict walk is exact-match only.
        for level in ("v", "debug", "trace", "", "FATAL", "Warn", "unknown"):
            assert _mpv_log_level(level) == logging.DEBUG

    def test_the_constants_are_read_at_call_time(self, monkeypatch):
        # A patched constant on the logging module steers both backends
        # identically — the engine reads the constant at call time,
        # never bakes it.
        import logging

        monkeypatch.setattr(logging, "WARNING", 31)
        assert _mpv_log_level("warn") == 31
        # The undo restores the real constant; the engine follows.
        monkeypatch.undo()
        assert _mpv_log_level("warn") == logging.WARNING

    def test_the_module_logging_name_is_read_at_call_time(self, monkeypatch):
        # The engine resolves the constants through the player module's
        # own ``logging`` name — the same patch-steering the module
        # clock's ``time`` read established — so swapping the module
        # attribute for a stub namespace steers both backends
        # identically.
        from types import SimpleNamespace

        from ytm_player.services import player as player_module

        fake = SimpleNamespace(CRITICAL=51, ERROR=41, WARNING=31, INFO=21, DEBUG=11)
        monkeypatch.setattr(player_module, "logging", fake)
        assert _mpv_log_level("fatal") == 51
        assert _mpv_log_level("error") == 41
        assert _mpv_log_level("warn") == 31
        assert _mpv_log_level("info") == 21
        assert _mpv_log_level("v") == 11


class TestPlayerGlueThroughTheMethods:
    """The engines behind the Player's own methods, on whichever backend
    is loaded — the end-file, seek and dispatch paths the existing
    player suites already exercise, plus the throttle the observer owns."""

    @pytest.fixture
    def player(self, monkeypatch):
        from ytm_player.services.player import Player

        Player._instance = None
        mock_mpv = MagicMock()
        mock_mpv.command.return_value = {"playlist_entry_id": 10}
        monkeypatch.setattr("ytm_player.services.player.mpv.MPV", MagicMock(return_value=mock_mpv))
        instance = Player()
        instance._mpv = mock_mpv
        yield instance
        Player._instance = None

    def test_the_observer_throttles_position_dispatches(self, player, monkeypatch):
        from ytm_player.services import player as player_module
        from ytm_player.services.player import PlayerEvent

        monkeypatch.setattr(player_module, "time", SimpleNamespace(monotonic=lambda: _FROZEN_NOW))
        positions = []
        player.on(PlayerEvent.POSITION_CHANGE, positions.append)

        player._on_time_pos_change("time-pos", 1.0)
        assert positions == [1.0]  # the first dispatch always goes out
        player._on_time_pos_change("time-pos", 1.2)
        assert positions == [1.0]  # within 0.4 s — throttled
        player._last_position_dispatch = _FROZEN_NOW - 0.5
        player._on_time_pos_change("time-pos", 1.4)
        assert positions == [1.0, 1.4]  # past the window — due again

    def test_the_observer_ignores_a_none_position(self, player):
        player._on_time_pos_change("time-pos", None)
        assert player._last_position_dispatch == 0.0  # untouched

    def test_the_seek_clamp_runs_through_the_engine(self, player):
        player._mpv.duration = 120.0
        player._mpv.time_pos = 110.0
        assert player._clamp_seek_target(200.0) == 120.0
        assert player._clamp_seek_target(-5.0) == 0.0


# The cases every engine is compared over: each entry is the argument
# list one call receives (the identity-bearing end-file cases carry
# instance tags the harness resolves to shared objects, the discovery
# cases carry a platform and a prefix spec the harness resolves against
# the fixtures the test created).
_PARITY_CASES = {
    "player_clamp_seek": [
        [50.0, 0.0],
        [50.0, -3.0],
        [200.0, 120.0],
        [30.0, 120.0],
        [120.0, 120.0],
        [-5.0, 120.0],
        [-5.0, 0.0],
        [200.0, float("nan")],
        [float("nan"), 100.0],
        [50, 0],
        [50, 30],
        [1.0, "long"],
        ["far", 100.0],
        [0.0, 0.0],
        [-0.0, 100.0],
    ],
    "player_clamp_volume": [
        [50],
        [0],
        [100],
        [-10],
        [150],
        [99],
        [1],
        [True],
        [False],
        ["loud"],
        [50.0],
        [-0.5],
        [100.5],
        [float("nan")],
    ],
    "player_position_due": [
        [9975.0],
        [10000.0],
        [9999.75],
        [9999.5],
        [0.0],
        [-1000.0],
        [10100.0],
        ["recently"],
    ],
    "player_command_current": [
        [None, 99, None],
        [7, 7, {"video_id": "A"}],
        [7, 7, None],
        [7, 8, {"video_id": "A"}],
        [True, 1, {"t": 1}],
        [0, 0, None],
        [None, None, None],
    ],
    "player_dispatch_arm": [
        # [loop_live, is_coroutine, has_context] — the full matrix.
        [True, True, True],
        [True, True, False],
        [True, False, True],
        [True, False, False],
        [False, True, True],
        [False, True, False],
        [False, False, True],
        [False, False, False],
    ],
    "player_mpv_log_level": [
        ["fatal"],
        ["error"],
        ["warn"],
        ["info"],
        ["v"],
        ["debug"],
        ["trace"],
        [""],
        ["FATAL"],
        ["Warn"],
    ],
    "player_end_file_verdict": [
        # [instance, current_instance, loading, entry_id,
        #  current_entry_id, current_track, reason]
        ["other", "live", False, 10, 10, {"t": 1}, 0],
        ["live", "live", False, 10, 10, {"t": 1}, 0],
        ["live", "live", True, 10, 10, {"t": 1}, 0],
        ["live", "live", False, "10", 10, {"t": 1}, 0],
        ["live", "live", False, None, 10, {"t": 1}, 0],
        ["live", "live", False, True, 1, {"t": 1}, 0],
        ["live", "live", False, 10, 11, {"t": 1}, 0],
        ["live", "live", False, 10, None, {"t": 1}, 0],
        ["live", "live", False, 10, 10, None, 0],
        ["live", "live", False, 10, 10, {"t": 1}, 2],
        ["live", "live", False, 10, 10, {"t": 1}, "0"],
        ["live", "live", False, 10, 10, {"t": 1}, None],
        ["live", "live", False, 10, 10, {"t": 1}, 4],
        ["live", "live", False, 10, 10, {"t": 1}, 0],
        ["live", "live", False, 10, 10, {"t": 1}, 4.0],
        ["live", "live", False, 10, 10, {"t": 1}, False],
        ["other", "live", True, "x", None, None, "y"],
    ],
}

_CHILD_PROG = """
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import player

player.time = SimpleNamespace(monotonic=lambda: 10000.0)


def build_dirs(spec):
    out = []
    for entry in spec:
        if entry["kind"] == "raising":

            def raising(_pattern):
                raise OSError("probe failed")

            out.append(SimpleNamespace(is_dir=lambda: True, glob=raising))
        else:
            out.append(Path(entry["path"]))
    return out


def call_args(name, args):
    if name == "player_find_libmpv":
        sys.platform = args[0]
        player._BREW_LIB_DIRS = tuple(build_dirs(args[1]))
        return []
    if name == "player_end_file_verdict":
        objs = {}

        def obj(tag):
            return objs.setdefault(tag, object())

        return [obj(args[0]), obj(args[1]), args[2], args[3], args[4], args[5], args[6]]
    return args


PY_NAMES = {
    "player_find_libmpv": "_find_brew_libmpv",
    "player_end_file_verdict": "_end_file_verdict",
    "player_clamp_seek": "_clamp_seek",
    "player_clamp_volume": "_clamp_volume",
    "player_position_due": "_position_due",
    "player_command_current": "_command_current",
    "player_dispatch_arm": "_dispatch_arm",
    "player_mpv_log_level": "_mpv_log_level",
}


def outcome(name, args):
    fn = getattr(player, PY_NAMES[name])
    try:
        result = fn(*call_args(name, args))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {name: [outcome(name, args) for args in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""

# The Python module-level name each native engine replaces (the tail
# rebinding assigns the native function to the fallback's own name;
# _find_brew_libmpv keeps its name and delegates at call time).
_PY_NAMES = {
    "player_find_libmpv": "_find_brew_libmpv",
    "player_end_file_verdict": "_end_file_verdict",
    "player_clamp_seek": "_clamp_seek",
    "player_clamp_volume": "_clamp_volume",
    "player_position_due": "_position_due",
    "player_command_current": "_command_current",
    "player_dispatch_arm": "_dispatch_arm",
    "player_mpv_log_level": "_mpv_log_level",
}


def _build_dirs(spec):
    """The prefix list rebuilt from the JSON case — a raising fake for
    the raising entries, a Path for the real ones."""
    from types import SimpleNamespace

    out = []
    for entry in spec:
        if entry["kind"] == "raising":

            def raising(_pattern):
                raise OSError("probe failed")

            out.append(SimpleNamespace(is_dir=lambda: True, glob=raising))
        else:
            out.append(Path(entry["path"]))
    return out


def _native_outcomes(cases):
    """The parent-side outcomes, computed with the native bindings live
    and the same patched platform, prefixes and frozen clock the child
    installs — restored once every engine has run."""

    def outcome(name, args):
        from ytm_player.services import player

        saved = (sys.platform, player._BREW_LIB_DIRS)
        try:
            call_args = args
            if name == "player_find_libmpv":
                sys.platform = args[0]
                player._BREW_LIB_DIRS = tuple(_build_dirs(args[1]))
                call_args = []
            elif name == "player_end_file_verdict":
                objs = {}

                def obj(tag):
                    return objs.setdefault(tag, object())

                call_args = [
                    obj(args[0]),
                    obj(args[1]),
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    args[6],
                ]
            fn = getattr(player, _PY_NAMES[name])
            try:
                result = fn(*call_args)
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                return {"error": f"{type(exc).__name__}: {exc}"}
            return {"ok": result}
        finally:
            sys.platform, player._BREW_LIB_DIRS = saved

    return {name: [outcome(name, args) for args in case_list] for name, case_list in cases.items()}


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same verdicts, same
    clamps, same throttle decisions, same discovery results, same errors
    with the same messages — with both sides reading the same frozen
    module clock and the same patched platform and prefixes, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_player_glue_matches_fallback(self, monkeypatch, tmp_path):
        from ytm_player.services import player

        monkeypatch.setattr(player, "time", SimpleNamespace(monotonic=lambda: _FROZEN_NOW))

        # The discovery fixtures: real directories with real lib names,
        # a missing path and a raising fake, walked under both
        # platforms' pattern pairs.
        so_dir = tmp_path / "so"
        so_dir.mkdir()
        (so_dir / "libmpv.so.2").touch()
        (so_dir / "libmpv.so").touch()
        dylib_dir = tmp_path / "dylib"
        dylib_dir.mkdir()
        (dylib_dir / "libmpv.2.dylib").touch()
        (dylib_dir / "libmpv.dylib").touch()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        discovery_cases = [
            ["linux", [{"kind": "dir", "path": str(so_dir)}]],
            ["darwin", [{"kind": "dir", "path": str(dylib_dir)}]],
            ["linux", [{"kind": "dir", "path": str(dylib_dir)}]],
            ["darwin", [{"kind": "dir", "path": str(so_dir)}]],
            [
                "linux",
                [
                    {"kind": "raising"},
                    {"kind": "missing", "path": str(tmp_path / "nope")},
                    {"kind": "dir", "path": str(so_dir)},
                ],
            ],
            ["linux", [{"kind": "dir", "path": str(empty_dir)}]],
            [
                "linux",
                [{"kind": "missing", "path": str(tmp_path / "gone")}],
            ],
            ["windows", [{"kind": "dir", "path": str(so_dir)}]],
        ]

        cases = dict(_PARITY_CASES)
        cases["player_find_libmpv"] = discovery_cases

        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes(cases)
        assert set(native) == set(fallback) == set(cases)
        for name, case_list in cases.items():
            for case, n, f in zip(case_list, native[name], fallback[name]):
                assert n == f, (
                    f"native and fallback {name} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
