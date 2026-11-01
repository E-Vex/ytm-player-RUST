"""Tests for KeyHandlingMixin._normalize_key and count buffer behavior.

These cover:
- Modifier translation (ctrl+x → C-x, shift+tab → S-tab, alt+v → M-v)
- Special-key remap (pageup → page_up, return → enter)
- Passthrough for unmodified printable keys
- Cap on count buffer (1000)

``_normalize_key`` delegates to the label engine
``_normalize_key_components`` — implemented by the Rust extension
(``_ytm_player_native.normalize_key``) when it is built, with the
pure-Python transcription in ``app/_keys.py`` as the fallback. The
engine-level tests below assert the engine's contract on whichever
backend is loaded, and the parity test compares the native engine
against the reference (run in a child interpreter with the extension
import blocked) case for case.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from textual.events import Key

from ytm_player.app._keys import (
    _MAX_KEY_COUNT,
    KeyHandlingMixin,
    _normalize_key_components,
)
from ytm_player.config import Action
from ytm_player.services.queue import QueueManager

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _make_event(key: str) -> MagicMock:
    """Build a minimal mock that quacks like textual.events.Key."""
    e = MagicMock()
    e.key = key
    return e


class TestNormalizeKey:
    def test_ctrl_modifier(self):
        assert KeyHandlingMixin._normalize_key(_make_event("ctrl+r")) == "C-r"

    def test_shift_modifier(self):
        assert KeyHandlingMixin._normalize_key(_make_event("shift+tab")) == "S-tab"

    def test_alt_modifier(self):
        assert KeyHandlingMixin._normalize_key(_make_event("alt+v")) == "M-v"

    def test_pageup_remap(self):
        assert KeyHandlingMixin._normalize_key(_make_event("pageup")) == "page_up"

    def test_return_aliases_enter(self):
        assert KeyHandlingMixin._normalize_key(_make_event("return")) == "enter"

    def test_question_mark_remap(self):
        assert KeyHandlingMixin._normalize_key(_make_event("question_mark")) == "?"

    def test_symbolic_punctuation_uses_character(self):
        """Regression for #143: ">" "<" "^" reach the app as symbolic names."""
        for key, char in [
            ("greater_than_sign", ">"),
            ("less_than_sign", "<"),
            ("circumflex_accent", "^"),
        ]:
            assert KeyHandlingMixin._normalize_key(Key(key, char)) == char

    def test_space_keeps_its_name(self):
        assert KeyHandlingMixin._normalize_key(Key("space", " ")) == "space"

    def test_modifier_wins_over_character(self):
        assert KeyHandlingMixin._normalize_key(Key("ctrl+r", None)) == "C-r"

    def test_unmodified_passthrough(self):
        assert KeyHandlingMixin._normalize_key(_make_event("j")) == "j"

    def test_arrow_keys_passthrough(self):
        assert KeyHandlingMixin._normalize_key(_make_event("up")) == "up"


class TestNormalizeKeyEngine:
    """The decomposed engine's contract, on whichever backend is loaded."""

    def test_modifier_prefixes(self) -> None:
        assert _normalize_key_components("ctrl+r", None, False) == "C-r"
        assert _normalize_key_components("shift+tab", None, False) == "S-tab"
        assert _normalize_key_components("alt+v", None, False) == "M-v"

    def test_modifier_prefix_wins_over_character(self) -> None:
        # The modifier branch fires before the character is ever consulted.
        assert _normalize_key_components("ctrl+r", "r", True) == "C-r"

    def test_multibyte_modifier_rest_survives(self) -> None:
        # The slice after the prefix is code-point aligned on both
        # backends (byte-slicing the Rust &str would panic instead).
        assert _normalize_key_components("ctrl+é", None, False) == "C-é"
        assert _normalize_key_components("shift+é", None, False) == "S-é"

    def test_printable_character_is_itself(self) -> None:
        assert _normalize_key_components("j", "j", True) == "j"
        assert _normalize_key_components("greater_than_sign", ">", True) == ">"
        assert _normalize_key_components("é", "é", True) == "é"

    def test_whitespace_keeps_its_name(self) -> None:
        # "space" (and "tab") stay names: the character branch excludes
        # whitespace so the name mapping runs instead.
        assert _normalize_key_components("space", " ", True) == "space"
        assert _normalize_key_components("tab", "\t", True) == "tab"

    def test_cpython_isspace_separators_are_whitespace(self) -> None:
        # \x1c-\x1f are whitespace to CPython's str.isspace (Rust's
        # char::is_whitespace disagrees), so the character branch must
        # reject them and fall through to the name mapping.
        for sep in "\x1c\x1d\x1e\x1f":
            assert _normalize_key_components("tab", sep, True) == "tab", repr(sep)

    def test_non_printable_character_falls_through(self) -> None:
        assert _normalize_key_components("a", "a", False) == "a"

    def test_multi_code_point_character_falls_through(self) -> None:
        # len() counts code points: a decomposed character is not one.
        assert _normalize_key_components("e\u0301", "e\u0301", True) == "e\u0301"

    def test_none_and_empty_characters_fall_through(self) -> None:
        assert _normalize_key_components("f1", None, False) == "f1"
        assert _normalize_key_components("f1", "", False) == "f1"

    def test_special_name_remap(self) -> None:
        cases = {
            "pageup": "page_up",
            "pagedown": "page_down",
            "page_up": "page_up",
            "page_down": "page_down",
            "return": "enter",
            "question_mark": "?",
            "plus": "+",
            "minus": "-",
            "equals": "=",
            "slash": "/",
        }
        for key, expected in cases.items():
            assert _normalize_key_components(key, None, False) == expected, key

    def test_unknown_names_pass_through(self) -> None:
        assert _normalize_key_components("f5", None, False) == "f5"
        assert _normalize_key_components("ctrl_x", None, False) == "ctrl_x"


class TestKeyCountCap:
    def test_max_count_constant_is_1000(self):
        """Sanity: regression guard if someone changes the cap silently."""
        assert _MAX_KEY_COUNT == 1000


def _shuffle_host(*, locked: bool = False) -> MagicMock:
    """A host with a real queue in a playlist context; the rest is mocked."""
    host = MagicMock()
    host.queue = QueueManager()
    host.queue.add_multiple(
        [{"video_id": f"v{i}", "title": f"T{i}", "artist": "A", "duration": 60} for i in range(3)]
    )
    host.queue.jump_to(0)
    host.queue.set_context("PL1")
    host.shuffle_prefs.get.return_value = locked
    return host


class TestToggleShuffleAction:
    async def test_toggle_refreshes_the_queue_page(self):
        host = _shuffle_host()

        await KeyHandlingMixin._handle_action(host, Action.TOGGLE_SHUFFLE)

        assert host.queue.shuffle_enabled
        host._refresh_queue_page.assert_called_once()

    async def test_locked_context_toggles_nothing(self):
        host = _shuffle_host(locked=True)

        await KeyHandlingMixin._handle_action(host, Action.TOGGLE_SHUFFLE)

        assert not host.queue.shuffle_enabled
        host._refresh_queue_page.assert_not_called()
        assert host.notify.call_args.kwargs.get("severity") == "warning"


class TestDiscoveryMixAction:
    async def test_runs_in_a_worker_group_of_its_own(self):
        host = MagicMock()

        await KeyHandlingMixin._handle_action(host, Action.DISCOVERY_MIX)

        host.run_worker.assert_called_once_with(
            host._start_discovery_mix.return_value, group="discovery-mix", exclusive=True
        )


# ── Native vs fallback parity ─────────────────────────────────────────

# (key, character, is_printable) triples covering every branch of the
# engine: the three modifier prefixes (including a multibyte rest), the
# character gate (printable, non-printable, whitespace — including the
# \x1c-\x1f separators CPython alone treats as whitespace — and
# multi-code-point characters), every special-name remap, and the
# passthrough fall-through.
_PARITY_CASES: list[tuple[str, str | None, bool]] = [
    ("ctrl+r", None, False),
    ("ctrl+r", "r", True),
    ("ctrl+é", None, False),
    ("ctrl+", None, False),
    ("shift+tab", None, False),
    ("shift+é", "é", True),
    ("alt+v", None, False),
    ("alt+", None, False),
    ("ctrl_x", None, False),
    ("j", "j", True),
    ("space", " ", True),
    ("tab", "\t", False),
    ("tab", "\t", True),
    ("tab", "\x1c", True),
    ("tab", "\x1d", True),
    ("tab", "\x1e", True),
    ("tab", "\x1f", True),
    ("tab", "\x85", True),
    ("tab", "\xa0", True),
    ("tab", "\u2028", True),
    ("tab", "\u2029", True),
    ("enter", "\r", False),
    ("greater_than_sign", ">", True),
    ("less_than_sign", "<", True),
    ("circumflex_accent", "^", True),
    ("question_mark", "?", True),
    ("plus", "+", True),
    ("minus", None, False),
    ("equals", "=", True),
    ("slash", "/", True),
    ("e\u0301", "e\u0301", True),
    ("é", "é", True),
    ("a", "a", False),
    ("f1", None, False),
    ("f1", "", False),
    ("", None, False),
    ("up", None, False),
    ("down", None, False),
    ("left", None, False),
    ("right", None, False),
    ("home", None, False),
    ("end", None, False),
    ("pageup", None, False),
    ("pagedown", None, False),
    ("page_up", None, False),
    ("page_down", None, False),
    ("backspace", None, False),
    ("delete", None, False),
    ("tab", None, False),
    ("enter", None, False),
    ("return", None, False),
    ("escape", None, False),
    ("unknown_key_name", None, False),
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._keys import _normalize_key_components

cases = json.load(sys.stdin)
json.dump([_normalize_key_components(k, c, p) for k, c, p in cases], sys.stdout)
"""


def _fallback_outcomes() -> list[str]:
    """Run the pure-Python reference engine in a child interpreter with
    the native extension import blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(_PARITY_CASES),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native key normaliser must
    produce exactly the same key names as the pure-Python reference for
    every case — every modifier prefix, the character gate including the
    CPython-only whitespace separators, every special-name remap, and the
    passthrough fall-through — so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_normalize_key_matches_fallback(self) -> None:
        native = [_normalize_key_components(*case) for case in _PARITY_CASES]
        fallback = _fallback_outcomes()
        assert native == fallback, (
            "native and fallback key-normalisation outcomes differ:\n"
            f"native:   {native!r}\n"
            f"fallback: {fallback!r}"
        )
