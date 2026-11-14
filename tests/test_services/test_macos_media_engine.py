"""Tests for the macOS Now-Playing engines (the state choice and the
µs→seconds conversion).

``_state_choice`` and ``_now_playing_seconds`` (services/macos_media.py)
run with two interchangeable backends: the pure-Python transcriptions
in the module and, when the Rust extension is built, the native
implementations bound at the module's tail. The engines: the playback
state choice behind ``MacOSMediaService._playback_state`` — the exact
``playing`` / ``paused`` / ``stopped`` match, with a no-status publish
resolving through the service's own playing flag — and the
``max(0.0, microseconds / 1_000_000)`` conversion behind both
Now-Playing time writes (``playbackDuration`` and
``elapsedPlaybackTime``).

The unit tests below assert each engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart. The state-choice answer
is a name, not the framework constant — the mapping through the
module's constant table (whose entries are None without the MediaPlayer
bindings) is pinned here too, alongside the existing behavioural suite
in test_macos_media.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pytest

import ytm_player.services.macos_media as macos_media
from ytm_player.services.macos_media import (
    MacOSMediaService,
    _now_playing_seconds,
    _state_choice,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestStateChoice:
    """The playback-state choice (whichever backend)."""

    def test_the_three_statuses_match_exactly(self):
        assert _state_choice("playing", True) == "playing"
        assert _state_choice("paused", True) == "paused"
        assert _state_choice("stopped", True) == "stopped"

    def test_the_status_match_wins_over_the_playing_flag(self):
        """A ``stopped`` publish answers stopped even while the service
        still believes it is playing."""
        assert _state_choice("stopped", True) == "stopped"
        assert _state_choice("paused", True) == "paused"

    def test_a_no_status_publish_resolves_through_the_playing_flag(self):
        assert _state_choice(None, True) == "playing"
        assert _state_choice(None, False) == "paused"

    def test_the_match_is_exact_not_case_insensitive(self):
        """A ``Playing`` that skipped the caller's lower() falls to the
        playing-flag branch, exactly as the reference's == does."""
        assert _state_choice("Playing", True) == "playing"
        assert _state_choice("Playing", False) == "paused"
        assert _state_choice("PLAYING", False) == "paused"

    def test_any_other_string_resolves_through_the_playing_flag(self):
        assert _state_choice("", True) == "playing"
        assert _state_choice("buffering", False) == "paused"
        assert _state_choice("playing ", True) == "playing"


class TestNowPlayingSeconds:
    """The µs→seconds conversion (whichever backend)."""

    def test_whole_seconds(self):
        assert _now_playing_seconds(0) == 0.0
        assert _now_playing_seconds(1_000_000) == 1.0
        assert _now_playing_seconds(180_000_000) == 180.0

    def test_fractional_seconds(self):
        assert _now_playing_seconds(500_000) == 0.5
        assert _now_playing_seconds(2_500_000) == 2.5
        assert _now_playing_seconds(32_000_000) == 32.0

    def test_a_negative_count_floors_at_zero(self):
        """``max(0.0, ...)`` — a negative duration or position reads
        0.0, exactly as the reference's max does."""
        assert _now_playing_seconds(-1) == 0.0
        assert _now_playing_seconds(-5_000_000) == 0.0


class TestPlaybackStateMapping:
    """The framework-constant resolution around the state choice."""

    def test_the_choice_maps_through_the_constant_table(self):
        sentinels = {"playing": "P", "paused": "A", "stopped": "S"}
        svc = MacOSMediaService()
        with patch.object(macos_media, "_PLAYBACK_STATES", sentinels):
            svc._is_playing = True
            assert svc._playback_state(None) == "P"
            svc._is_playing = False
            assert svc._playback_state(None) == "A"
            assert svc._playback_state("stopped") == "S"

    def test_without_the_framework_the_constants_are_none(self):
        """The table's entries are None without the MediaPlayer
        bindings — the setPlaybackState_ write then skips exactly as
        the reference's None returns did."""
        svc = MacOSMediaService()
        svc._is_playing = True
        assert macos_media._PLAYBACK_STATES["playing"] is None
        assert svc._playback_state("playing") is None
        assert svc._playback_state(None) is None


# ── Native vs fallback parity ─────────────────────────────────────────

_STATE_CASES = [
    [None, True],
    [None, False],
    ["playing", True],
    ["playing", False],
    ["paused", True],
    ["paused", False],
    ["stopped", True],
    ["stopped", False],
    ["Playing", True],
    ["Playing", False],
    ["PLAYING", False],
    ["", True],
    ["buffering", False],
    ["playing ", True],
]

_SECONDS_CASES = [
    0,
    1,
    500_000,
    1_000_000,
    1_500_000,
    2_500_000,
    32_000_000,
    180_000_000,
    -1,
    -5_000_000,
    1_000_000_000_000,
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.macos_media import _now_playing_seconds, _state_choice

payload = json.load(sys.stdin)
out = {
    "states": [_state_choice(status, is_playing) for status, is_playing in payload["state_cases"]],
    "seconds": [_now_playing_seconds(case) for case in payload["seconds_cases"]],
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
        payload = json.dumps({"state_cases": _STATE_CASES, "seconds_cases": _SECONDS_CASES})
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_states = [_state_choice(status, is_playing) for status, is_playing in _STATE_CASES]
        native_seconds = [_now_playing_seconds(case) for case in _SECONDS_CASES]
        assert native_states == fallback["states"]
        assert native_seconds == fallback["seconds"]
