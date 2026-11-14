"""Tests for the macOS event tap's media-key classification engine.

``_key_action`` (services/macos_eventtap.py) runs with two
interchangeable backends: the pure-Python transcription in the module
and, when the Rust extension is built, the native implementation bound
at the module's tail. The engine: the NSEvent ``data1`` word's bit
unpack behind ``_event_action`` — the key code in the high 16 bits, the
flags in the low 16 with the key state in bits 8-15 — the
media-key-down state gate (``0xA``) and the key-code → action map (16
play/pause, 17 next, 18 previous, 19 fast-forward → next, 20 rewind →
previous).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart. ``_event_action``'s own
gates — the ``ns_event`` None gate, the ``subtype()`` comparison
against the raw framework value and the ``int()`` coercion with its
broad ``except`` — stay Python-side and are pinned here alongside the
existing behavioural suite in test_macos_eventtap.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from ytm_player.services.macos_eventtap import _event_action, _key_action

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_MEDIA_EVENT_SUBTYPE = 8


def _media_data1(key_code: int, state: int, low_flags: int = 0) -> int:
    """Assemble an NSEvent data1 word: key code in the high 16 bits,
    key state in bits 8-15, remaining flags in the low 8 bits."""
    return (key_code << 16) | (state << 8) | low_flags


class _FakeNSEvent:
    def __init__(self, subtype: int, data1: int) -> None:
        self._subtype = subtype
        self._data1 = data1

    def subtype(self) -> int:
        return self._subtype

    def data1(self) -> int:
        return self._data1


class TestKeyAction:
    """The data1 classification (whichever backend)."""

    def test_the_five_media_keys_map_to_their_actions(self):
        assert _key_action(_media_data1(16, 0xA)) == "play_pause"
        assert _key_action(_media_data1(17, 0xA)) == "next"
        assert _key_action(_media_data1(18, 0xA)) == "previous"
        # Fast-forward and rewind join their siblings' actions.
        assert _key_action(_media_data1(19, 0xA)) == "next"
        assert _key_action(_media_data1(20, 0xA)) == "previous"

    def test_key_up_events_do_not_fire(self):
        """Only the media-key-down state (0xA) is intercepted."""
        assert _key_action(_media_data1(16, 0xB)) is None
        assert _key_action(_media_data1(17, 0x1)) is None
        assert _key_action(_media_data1(18, 0x0)) is None

    def test_the_low_flag_bits_are_ignored(self):
        """The key state lives in bits 8-15; the low eight flag bits
        (the shifted-key mask etc.) never gate the classification."""
        assert _key_action(_media_data1(16, 0xA, low_flags=0xFF)) == "play_pause"
        assert _key_action(_media_data1(20, 0xA, low_flags=0x01)) == "previous"

    def test_unknown_key_codes_map_nowhere(self):
        assert _key_action(_media_data1(21, 0xA)) is None
        assert _key_action(_media_data1(0, 0xA)) is None
        assert _key_action(_media_data1(0xFFFF, 0xA)) is None

    def test_a_zero_word_has_no_key_down_state(self):
        assert _key_action(0) is None

    def test_a_negative_word_masks_to_a_positive_key_code(self):
        """A negative data1 masks to the same positive key code the
        arbitrary-precision reference computes (−1 & 0xFFFF0000 →
        0xFFFF0000 → key code 0xFFFF, unmapped)."""
        assert _key_action(-1) is None
        assert _key_action(-(2**62)) is None


class TestEventActionGates:
    """The Python-side gates around the engine."""

    def test_the_none_gate_stays_python_side(self):
        assert _event_action(None) is None

    def test_a_non_media_subtype_is_ignored_before_the_engine(self):
        """The subtype comparison runs against the raw framework value
        before the classification — a non-media event never reaches
        the key-code map."""
        assert _event_action(_FakeNSEvent(0, _media_data1(16, 0xA))) is None
        assert _event_action(_FakeNSEvent(22, _media_data1(17, 0xA))) is None

    def test_a_media_event_routes_through_the_engine(self):
        assert _event_action(_FakeNSEvent(8, _media_data1(16, 0xA))) == "play_pause"

    def test_a_broken_event_object_is_swallowed(self):
        class _Broken:
            def subtype(self) -> int:
                raise RuntimeError("framework gone")

        assert _event_action(_Broken()) is None

    def test_the_data1_int_coercion_stays_python_side(self):
        """``int()`` truncates whatever the framework returns before the
        engine sees it."""

        class _FloatyEvent:
            def subtype(self) -> int:
                return _MEDIA_EVENT_SUBTYPE

            def data1(self) -> Any:
                return float(_media_data1(16, 0xA)) + 0.9

        assert _event_action(_FloatyEvent()) == "play_pause"


# ── Native vs fallback parity ─────────────────────────────────────────

_DATA1_CASES = [
    0,
    _media_data1(16, 0xA),
    _media_data1(17, 0xA),
    _media_data1(18, 0xA),
    _media_data1(19, 0xA),
    _media_data1(20, 0xA),
    _media_data1(21, 0xA),
    _media_data1(0, 0xA),
    _media_data1(0xFFFF, 0xA),
    _media_data1(16, 0xB),
    _media_data1(17, 0x1),
    _media_data1(16, 0xA, low_flags=0xFF),
    _media_data1(20, 0xA, low_flags=0x01),
    _media_data1(16, 0xA) | 1,
    (16 << 16),
    -1,
    -(2**62),
    (0xFFFF << 16) | (0xA << 8),
    1,
    0xFFFF0000 | (0xA << 8),
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.macos_eventtap import _key_action

payload = json.load(sys.stdin)
out = {"actions": [_key_action(case) for case in payload["data1_cases"]]}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native verdicts must equal
    the pure-Python reference's for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        payload = json.dumps({"data1_cases": _DATA1_CASES})
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_actions = [_key_action(case) for case in _DATA1_CASES]
        assert native_actions == fallback["actions"]
