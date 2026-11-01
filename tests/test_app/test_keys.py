"""Tests for KeyHandlingMixin: key normalisation, the count/sequence
buffer state machine and the ``on_key`` dispatch glue.

These cover:
- Modifier translation (ctrl+x → C-x, shift+tab → S-tab, alt+v → M-v)
- Special-key remap (pageup → page_up, return → enter)
- Passthrough for unmodified printable keys
- Cap on count buffer (1000)
- The count/sequence buffer state machine (``_feed_key``): digit
  accumulation, count resolution, sequence pending/clear, the buffers'
  fresh-copy contract and the digit-classified-but-``int()``-proof wart
- The ``on_key`` glue: modal/text-entry early returns, event suppression
  per outcome and the awaited dispatch

``_normalize_key`` delegates to the label engine
``_normalize_key_components`` and ``on_key`` to the state-machine engine
``_feed_key`` — both implemented by the Rust extension
(``_ytm_player_native.normalize_key`` / ``feed_key``) when it is built,
with the pure-Python transcriptions in ``app/_keys.py`` as the fallback.
The engine-level tests below assert each engine's contract on whichever
backend is loaded, and the parity tests compare the native engines
against the references (run in a child interpreter with the extension
import blocked) case for case.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from textual.events import Key

from ytm_player.app._keys import (
    _MAX_KEY_COUNT,
    KeyHandlingMixin,
    _feed_key,
    _normalize_key_components,
)
from ytm_player.config import Action, MatchResult
from ytm_player.config.keymap import KeyMap
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


def _default_keymap() -> KeyMap:
    """A fresh KeyMap with the default binding table."""
    keymap = KeyMap()
    keymap._load_defaults()
    return keymap


class TestFeedKeyEngine:
    """The count/sequence buffer state machine's contract, on whichever
    backend is loaded."""

    def test_digit_accumulates_count_prefix(self) -> None:
        out = _feed_key("5", [], "", _default_keymap())
        assert out["outcome"] == "count_digit"
        assert out["prevent_default"] is True
        assert out["stop"] is False
        assert out["count_buffer"] == "5"
        assert out["key_buffer"] == []
        assert out["count"] is None
        assert out["action"] is None

    def test_digit_appends_to_the_existing_count(self) -> None:
        out = _feed_key("3", [], "2", _default_keymap())
        assert out["outcome"] == "count_digit"
        assert out["count_buffer"] == "23"

    def test_digit_with_a_pending_sequence_is_not_a_count(self) -> None:
        # A digit only accumulates while the sequence buffer is empty;
        # mid-sequence it is just another (unbound) key.
        out = _feed_key("5", ["g"], "", _default_keymap())
        assert out["outcome"] == "no_match"

    def test_exact_returns_action_and_default_count(self) -> None:
        out = _feed_key("j", [], "", _default_keymap())
        assert out["outcome"] == "exact"
        assert out["prevent_default"] is True
        assert out["stop"] is True
        assert out["count"] == 1
        assert out["action"] is Action.MOVE_DOWN
        assert out["key_buffer"] == []
        assert out["count_buffer"] == ""

    def test_exact_parses_the_count_buffer(self) -> None:
        out = _feed_key("j", [], "42", _default_keymap())
        assert out["outcome"] == "exact"
        assert out["count"] == 42

    def test_exact_zero_count_stays_zero(self) -> None:
        # int("0") is 0 — a no-op dispatch, not the default 1.
        out = _feed_key("j", [], "0", _default_keymap())
        assert out["count"] == 0

    def test_count_is_capped_at_max(self) -> None:
        assert _feed_key("j", [], "9999", _default_keymap())["count"] == 1000
        assert _feed_key("j", [], "9" * 50, _default_keymap())["count"] == 1000

    def test_non_ascii_decimal_digits_count(self) -> None:
        # int() reads Nd digits of any script: Arabic-Indic ٣ is 3.
        assert _feed_key("j", [], "٣", _default_keymap())["count"] == 3
        assert _feed_key("j", [], "٣٤", _default_keymap())["count"] == 34

    def test_sequence_exact_clears_both_buffers(self) -> None:
        out = _feed_key("g", ["g"], "5", _default_keymap())
        assert out["outcome"] == "exact"
        assert out["action"] is Action.GO_TOP
        assert out["count"] == 5
        assert out["key_buffer"] == []
        assert out["count_buffer"] == ""

    def test_count_survives_a_pending_prefix(self) -> None:
        # 5 g g goes to top with count 5 — the count buffer outlives the
        # sequence, and the exact after the prefix still sees it.
        pending = _feed_key("g", [], "5", _default_keymap())
        assert pending["outcome"] == "pending"
        assert pending["key_buffer"] == ["g"]
        assert pending["count_buffer"] == "5"
        exact = _feed_key("l", ["C-w"], "3", _default_keymap())
        assert exact["outcome"] == "exact"
        assert exact["action"] is Action.FOCUS_PANE_RIGHT
        assert exact["count"] == 3

    def test_pending_suppresses_and_keeps_the_grown_buffer(self) -> None:
        out = _feed_key("g", [], "", _default_keymap())
        assert out["outcome"] == "pending"
        assert out["prevent_default"] is True
        assert out["stop"] is True
        assert out["key_buffer"] == ["g"]
        assert out["count_buffer"] == ""

    def test_no_match_clears_buffers_without_suppression(self) -> None:
        out = _feed_key("x", ["g"], "7", _default_keymap())
        assert out["outcome"] == "no_match"
        assert out["prevent_default"] is False
        assert out["stop"] is False
        assert out["key_buffer"] == []
        assert out["count_buffer"] == ""
        assert out["count"] is None
        assert out["action"] is None

    def test_digit_classified_non_decimal_key_raises_on_resolution(self) -> None:
        # '²' passes the isdigit gate (Numeric_Type=Digit) but int()
        # rejects it — the reference's latent wart, transcribed on both
        # backends down to CPython's message.
        out = _feed_key("²", [], "", _default_keymap())
        assert out["outcome"] == "count_digit"
        assert out["count_buffer"] == "²"
        with pytest.raises(ValueError, match=r"invalid literal for int\(\) with base 10"):
            _feed_key("j", [], "²", _default_keymap())

    def test_empty_key_is_not_a_digit(self) -> None:
        assert _feed_key("", [], "", _default_keymap())["outcome"] == "no_match"

    def test_inputs_are_never_mutated_and_outputs_are_fresh(self) -> None:
        key_buffer = ["g"]
        out = _feed_key("x", key_buffer, "5", _default_keymap())
        assert key_buffer == ["g"]
        assert out["key_buffer"] is not key_buffer
        exact = _feed_key("g", key_buffer, "", _default_keymap())
        assert key_buffer == ["g"]
        assert exact["key_buffer"] == []
        assert exact["key_buffer"] is not key_buffer
        fresh_input: list[str] = []
        pending = _feed_key("g", fresh_input, "", _default_keymap())
        assert fresh_input == []
        assert pending["key_buffer"] == ["g"]
        assert pending["key_buffer"] is not fresh_input

    def test_outcome_map_shape(self) -> None:
        out = _feed_key("j", [], "", _default_keymap())
        assert set(out) == {
            "outcome",
            "prevent_default",
            "stop",
            "key_buffer",
            "count_buffer",
            "count",
            "action",
        }

    def test_stub_match_returning_a_list_is_accepted(self) -> None:
        # Python's two-name unpacking accepts any 2-iterable; the engine
        # must not narrow that to tuples.
        class StubKeyMap:
            def match(self, seq: tuple[str, ...]) -> list:
                return [MatchResult.PENDING, None]

        assert _feed_key("j", [], "", StubKeyMap())["outcome"] == "pending"

    def test_stub_exact_with_none_action_passes_through(self) -> None:
        # _handle_action treats None as a no-op; the engine just passes it
        # through instead of second-guessing the keymap.
        class StubKeyMap:
            def match(self, seq: tuple[str, ...]) -> tuple[MatchResult, None]:
                return (MatchResult.EXACT, None)

        out = _feed_key("j", [], "", StubKeyMap())
        assert out["outcome"] == "exact"
        assert out["action"] is None
        assert out["count"] == 1

    def test_stub_result_without_a_string_value_reads_as_no_match(self) -> None:
        # Anything the reference's `== MatchResult.EXACT` would reject —
        # a mock without a string .value — reads as the else branch.
        class StubKeyMap:
            def match(self, seq: tuple[str, ...]) -> tuple[object, None]:
                return (object(), None)

        assert _feed_key("j", [], "", StubKeyMap())["outcome"] == "no_match"

    def test_stub_match_arity_errors_match_cpython_unpacking(self) -> None:
        class Short:
            def match(self, seq: tuple[str, ...]) -> tuple[MatchResult]:
                return (MatchResult.EXACT,)

        class Long:
            def match(self, seq: tuple[str, ...]) -> tuple[MatchResult, None, None]:
                return (MatchResult.EXACT, None, None)

        with pytest.raises(ValueError, match="not enough values to unpack"):
            _feed_key("j", [], "", Short())
        with pytest.raises(ValueError, match="too many values to unpack"):
            _feed_key("j", [], "", Long())


class _KeyEvent:
    """A duck-typed ``textual.events.Key`` with spyable suppression."""

    def __init__(self, key: str) -> None:
        self.key = key
        self.prevented = False
        self.stopped = False

    def prevent_default(self) -> None:
        self.prevented = True

    def stop(self) -> None:
        self.stopped = True


def _on_key_host(
    *,
    is_modal: bool = False,
    focused: object = None,
    key_buffer: list[str] | None = None,
    count_buffer: str = "",
) -> SimpleNamespace:
    """A duck-typed host for ``KeyHandlingMixin.on_key``: a real default
    keymap, plain buffer attributes and spies for normalisation and
    dispatch."""
    return SimpleNamespace(
        screen=SimpleNamespace(is_modal=is_modal),
        focused=focused,
        keymap=_default_keymap(),
        _key_buffer=key_buffer if key_buffer is not None else [],
        _count_buffer=count_buffer,
        _normalize_key=Mock(side_effect=lambda event: event.key),
        _handle_action=AsyncMock(),
    )


class TestOnKeyDispatch:
    """The ``on_key`` glue around the engine: early returns, suppression
    and the awaited dispatch."""

    async def test_modal_screen_returns_early(self) -> None:
        host = _on_key_host(is_modal=True)
        event = _KeyEvent("j")

        await KeyHandlingMixin.on_key(host, event)  # type: ignore[arg-type]

        host._normalize_key.assert_not_called()
        assert not event.prevented and not event.stopped
        host._handle_action.assert_not_awaited()

    async def test_focused_text_entry_returns_early(self) -> None:
        from textual.widgets import Input

        host = _on_key_host(focused=Input())
        event = _KeyEvent("j")

        await KeyHandlingMixin.on_key(host, event)  # type: ignore[arg-type]

        host._normalize_key.assert_not_called()
        assert not event.prevented and not event.stopped
        host._handle_action.assert_not_awaited()

    async def test_digit_prevents_default_without_stopping_or_dispatch(self) -> None:
        host = _on_key_host()
        event = _KeyEvent("5")

        await KeyHandlingMixin.on_key(host, event)  # type: ignore[arg-type]

        assert event.prevented and not event.stopped
        assert host._count_buffer == "5"
        assert host._key_buffer == []
        host._handle_action.assert_not_awaited()

    async def test_exact_dispatches_with_the_default_count(self) -> None:
        host = _on_key_host()
        event = _KeyEvent("j")

        await KeyHandlingMixin.on_key(host, event)  # type: ignore[arg-type]

        assert event.prevented and event.stopped
        host._handle_action.assert_awaited_once_with(Action.MOVE_DOWN, 1)
        assert host._key_buffer == [] and host._count_buffer == ""

    async def test_count_prefix_flows_into_the_dispatch(self) -> None:
        host = _on_key_host()

        await KeyHandlingMixin.on_key(host, _KeyEvent("5"))  # type: ignore[arg-type]
        await KeyHandlingMixin.on_key(host, _KeyEvent("j"))  # type: ignore[arg-type]

        host._handle_action.assert_awaited_once_with(Action.MOVE_DOWN, 5)

    async def test_pending_suppresses_without_dispatching(self) -> None:
        host = _on_key_host()
        event = _KeyEvent("g")

        await KeyHandlingMixin.on_key(host, event)  # type: ignore[arg-type]

        assert event.prevented and event.stopped
        host._handle_action.assert_not_awaited()
        assert host._key_buffer == ["g"]

    async def test_no_match_resets_buffers_without_suppression(self) -> None:
        host = _on_key_host(key_buffer=["g"], count_buffer="7")
        event = _KeyEvent("x")

        await KeyHandlingMixin.on_key(host, event)  # type: ignore[arg-type]

        assert not event.prevented and not event.stopped
        host._handle_action.assert_not_awaited()
        assert host._key_buffer == [] and host._count_buffer == ""

    async def test_int_proof_count_raises_like_the_reference(self) -> None:
        host = _on_key_host(count_buffer="²")

        with pytest.raises(ValueError, match=r"invalid literal for int\(\) with base 10"):
            await KeyHandlingMixin.on_key(host, _KeyEvent("j"))  # type: ignore[arg-type]


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


# ── Count/sequence state-machine native vs fallback parity ────────────

# (key, key_buffer, count_buffer) triples covering every branch of the
# state machine: the digit gate (ASCII, non-ASCII decimal and the
# isdigit-but-not-int()-able Numeric_Type=Digit warts, a multi-character
# all-digit key, digits arriving mid-sequence), every MatchResult branch
# (exact singles, pending prefixes, exact completions across a pending
# prefix with the count surviving it, dead ends), the count resolution
# (default, parsed, zero, the 1000 cap including a 40-digit bigint, mixed
# scripts, the ValueError wart), and the buffer resets.
_FEED_PARITY_CASES: list[tuple[str, list[str], str]] = [
    # Digit accumulation.
    ("5", [], ""),
    ("3", [], "2"),
    ("0", [], ""),
    ("٩", [], ""),
    ("٣", [], "1"),
    ("²", [], ""),
    ("٣٥", [], ""),
    # Digits mid-sequence are just unbound keys, not counts.
    ("5", ["g"], ""),
    ("5", ["C-w"], ""),
    # Exact single-key matches.
    ("j", [], ""),
    ("k", [], ""),
    ("J", [], ""),
    ("space", [], ""),
    ("?", [], ""),
    ("C-r", [], ""),
    ("S-tab", [], ""),
    ("+", [], ""),
    # Exact with count prefixes.
    ("j", [], "5"),
    ("j", [], "0"),
    ("j", [], "9999"),
    ("j", [], "9" * 40),
    ("j", [], "٣"),
    ("j", [], "٣٤"),
    ("j", [], "²"),
    # Pending prefixes.
    ("g", [], ""),
    ("g", [], "5"),
    ("s", [], ""),
    ("C-w", [], ""),
    ("C-w", [], "3"),
    ("d", [], ""),
    # Exact completions of a pending sequence.
    ("g", ["g"], ""),
    ("g", ["g"], "5"),
    ("l", ["C-w"], "3"),
    ("w", ["C-w"], ""),
    ("h", ["C-w"], ""),
    ("space", ["g"], ""),
    ("t", ["s"], ""),
    ("d", ["d"], ""),
    # No match.
    ("x", [], ""),
    ("x", ["g"], ""),
    ("g", ["x"], ""),
    ("", [], ""),
    ("x", ["g"], "9"),
]

_FEED_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._keys import _feed_key
from ytm_player.config.keymap import KeyMap

keymap = KeyMap()
keymap._load_defaults()


def normalise(outcome):
    action = outcome["action"]
    return {
        "outcome": outcome["outcome"],
        "prevent_default": outcome["prevent_default"],
        "stop": outcome["stop"],
        "key_buffer": list(outcome["key_buffer"]),
        "count_buffer": outcome["count_buffer"],
        "count": outcome["count"],
        "action": None if action is None else action.name,
    }


results = []
for key, key_buffer, count_buffer in json.load(sys.stdin):
    try:
        results.append(normalise(_feed_key(key, key_buffer, count_buffer, keymap)))
    except Exception as exc:
        results.append({"error": f"{type(exc).__name__}: {exc}"})
json.dump(results, sys.stdout)
"""


def _normalise_feed_outcome(outcome: dict) -> dict:
    action = outcome["action"]
    return {
        "outcome": outcome["outcome"],
        "prevent_default": outcome["prevent_default"],
        "stop": outcome["stop"],
        "key_buffer": list(outcome["key_buffer"]),
        "count_buffer": outcome["count_buffer"],
        "count": outcome["count"],
        "action": None if action is None else action.name,
    }


def _feed_fallback_outcomes() -> list[dict]:
    """Run the pure-Python reference state machine in a child interpreter
    with the native extension import blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _FEED_CHILD_PROG],
        input=json.dumps(_FEED_PARITY_CASES),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestFeedKeyNativeParity:
    """When the Rust extension is built, the native state machine must
    produce exactly the same outcome maps — outcome kind, both suppression
    flags, both buffers, the resolved count and the dispatched action —
    as the pure-Python reference for every case, and raise the same
    ValueError where the reference's ``int()`` does, so the two
    implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_feed_key_matches_fallback(self) -> None:
        keymap = _default_keymap()
        native = []
        for key, key_buffer, count_buffer in _FEED_PARITY_CASES:
            try:
                native.append(
                    _normalise_feed_outcome(_feed_key(key, list(key_buffer), count_buffer, keymap))
                )
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                native.append({"error": f"{type(exc).__name__}: {exc}"})
        fallback = _feed_fallback_outcomes()
        assert native == fallback, (
            "native and fallback key-feed outcomes differ:\n"
            f"cases:   {[c for c, n, f in zip(_FEED_PARITY_CASES, native, fallback) if n != f]!r}\n"
            f"native:   {native!r}\n"
            f"fallback: {fallback!r}"
        )
