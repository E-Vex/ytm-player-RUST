"""Tests for the mpv-wrapper glue engines.

The five pure cores of ``services/player.py`` run with two
interchangeable backends: the pure-Python transcriptions in the module
and, when the Rust extension is built, the native implementations (the
tail rebinding for four of them, the import-time binding plus call-time
check for ``_find_brew_libmpv``). ``_find_brew_libmpv`` is the
Homebrew-prefix libmpv discovery walk behind the import-time fallback
find; ``_end_file_verdict`` is the event-matching gate behind
``_end_file_data``'s handling of mpv's end-file callbacks;
``_clamp_seek`` is the ``[0, duration]`` target clamp behind
``_clamp_seek_target``; ``_position_due`` is the 0.4 s throttle behind
``_on_time_pos_change``'s POSITION_CHANGE dispatches; and
``_command_current`` is the supersession predicate behind
``_dispatch``'s still-current check and ``_finish_load``'s acceptance
check. The clock-reading engine resolves ``time.monotonic()`` through
the module's own ``time`` name at call time — a frozen or patched clock
steers both backends identically, which the tests below use for
determinism and the parity class pins as the contract.

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
    _command_current,
    _end_file_verdict,
    _find_brew_libmpv,
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

    def test_finds_linux_so(self, brew_dirs, tmp_path):
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

    def test_prefers_the_unversioned_name(self, brew_dirs, tmp_path):
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

    def test_skips_non_dir_entries(self, brew_dirs, tmp_path):
        a_file = tmp_path / "afile"
        a_file.touch()
        real = tmp_path / "real"
        real.mkdir()
        (real / "libmpv.so").touch()
        brew_dirs((a_file, real))
        assert _find_brew_libmpv() == str(real / "libmpv.so")

    def test_the_first_prefix_with_a_hit_wins(self, brew_dirs, tmp_path):
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "libmpv.so.1").touch()
        (second / "libmpv.so").touch()
        brew_dirs((second, first))
        assert _find_brew_libmpv() == str(second / "libmpv.so")

    def test_an_oserror_from_a_prefix_continues_to_the_next(self, brew_dirs, tmp_path):
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
    "player_position_due": "_position_due",
    "player_command_current": "_command_current",
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
    "player_position_due": "_position_due",
    "player_command_current": "_command_current",
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
