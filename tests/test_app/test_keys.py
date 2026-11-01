"""Tests for KeyHandlingMixin: key normalisation, the count/sequence
buffer state machine, the action dispatch table and the ``on_key``
dispatch glue.

These cover:
- Modifier translation (ctrl+x → C-x, shift+tab → S-tab, alt+v → M-v)
- Special-key remap (pageup → page_up, return → enter)
- Passthrough for unmodified printable keys
- Cap on count buffer (1000)
- The count/sequence buffer state machine (``_feed_key``): digit
  accumulation, count resolution, sequence pending/clear, the buffers'
  fresh-copy contract and the digit-classified-but-``int()``-proof wart
- The dispatch table (``_dispatch_route``): every action's branch, the
  None / unhandled fall-throughs, the str-enum plain-string wart, the
  first-equal-wins walk with stub tables, and CPython's row-unpack
  errors
- The pane routing (``_navigation_action_route``), the CURRENT_CONTEXT
  album-target extraction (``_current_context_target``) and the
  Shuffle-lock gate (``_shuffle_lock_blocked``)
- The ``on_key`` glue: modal/text-entry early returns, event suppression
  per outcome and the awaited dispatch
- The ``_handle_action`` glue: the engine's branch applied to the host

``_normalize_key`` delegates to the label engine
``_normalize_key_components``, ``on_key`` to the state-machine engine
``_feed_key``, and ``_handle_action`` to the dispatch engine
``_dispatch_route`` (with the pane-routing, album-target and lock-gate
engines behind its branches) — all implemented by the Rust extension
(``_ytm_player_native.normalize_key`` / ``feed_key`` /
``dispatch_route`` & co.) when it is built, with the pure-Python
transcriptions in ``app/_keys.py`` as the fallback. The engine-level
tests below assert each engine's contract on whichever backend is
loaded, and the parity tests compare the native engines against the
references (run in a child interpreter with the extension import
blocked) case for case.
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
    _DISPATCH_TABLE,
    _LYRICS_PANE_ACTIONS,
    _MAX_KEY_COUNT,
    _SIDEBAR_PANE_ACTIONS,
    KeyHandlingMixin,
    _current_context_target,
    _dispatch_route,
    _feed_key,
    _navigation_action_route,
    _normalize_key_components,
    _shuffle_lock_blocked,
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


# ── Action dispatch engine ──────────────────────────────────────────

# Every action's branch (and page, for the navigations) pinned
# action-for-action — the reference match statement's routing, spelled
# out so a table edit that re-routes anything fails loudly.
_EXPECTED_ROUTES: dict[str, tuple[str, str | None]] = {
    "PLAY_PAUSE": ("toggle_play_pause", None),
    "NEXT_TRACK": ("play_next", None),
    "PREVIOUS_TRACK": ("play_previous", None),
    "PLAY_RANDOM": ("play_random", None),
    "CYCLE_REPEAT": ("cycle_repeat", None),
    "TOGGLE_SHUFFLE": ("toggle_shuffle", None),
    "VOLUME_UP": ("volume_up", None),
    "VOLUME_DOWN": ("volume_down", None),
    "MUTE": ("mute", None),
    "SEEK_FORWARD": ("seek_forward", None),
    "SEEK_BACKWARD": ("seek_backward", None),
    "SEEK_START": ("seek_start", None),
    "LIBRARY": ("navigate", "library"),
    "SEARCH": ("navigate", "search"),
    "QUEUE": ("navigate", "queue"),
    "LYRICS": ("toggle_lyrics_sidebar", None),
    "TOGGLE_SIDEBAR": ("toggle_playlist_sidebar", None),
    "TOGGLE_TRANSLITERATION": ("toggle_transliteration", None),
    "TOGGLE_ALBUM_ART": ("toggle_album_art", None),
    "BROWSE": ("navigate", "browse"),
    "HELP": ("navigate", "help"),
    "LIKED_SONGS": ("navigate", "liked_songs"),
    "RECENTLY_PLAYED": ("navigate", "recently_played"),
    "CURRENT_CONTEXT": ("current_context", None),
    "GO_BACK": ("navigate", "back"),
    "GO_FORWARD": ("navigate", "forward"),
    "QUIT": ("quit", None),
    "ADD_TO_PLAYLIST": ("open_add_to_playlist", None),
    "DISCOVERY_MIX": ("run_discovery_mix", None),
    "TRACK_ACTIONS": ("open_track_actions", None),
    "PLAY_NEXT": ("play_focused_next", None),
    "ADD_TO_QUEUE": ("add_focused_to_queue", None),
    "LIKE_TOGGLE": ("toggle_like_current", None),
    "FOCUS_PANE_LEFT": ("focus_pane_left", None),
    "FOCUS_PANE_RIGHT": ("focus_pane_right", None),
    "FOCUS_PANE_CYCLE": ("cycle_pane", None),
    "FOCUS_NEXT": ("focus_next", None),
    "FOCUS_PREV": ("focus_prev", None),
    "MOVE_DOWN": ("route_navigation", None),
    "MOVE_UP": ("route_navigation", None),
    "PAGE_DOWN": ("route_navigation", None),
    "PAGE_UP": ("route_navigation", None),
    "GO_TOP": ("route_navigation", None),
    "GO_BOTTOM": ("route_navigation", None),
    "SELECT": ("route_navigation", None),
    "DELETE_ITEM": ("route_navigation", None),
    "FILTER": ("route_navigation", None),
    "SORT_TITLE": ("route_navigation", None),
    "SORT_ARTIST": ("route_navigation", None),
    "SORT_ALBUM": ("route_navigation", None),
    "SORT_DURATION": ("route_navigation", None),
    "SORT_DATE": ("route_navigation", None),
    "REVERSE_SORT": ("route_navigation", None),
    "JUMP_TO_CURRENT": ("route_navigation", None),
    "TOGGLE_SEARCH_MODE": ("route_navigation", None),
    "PICK_COUNTRY": ("route_navigation", None),
    "REORDER_DOWN": ("route_navigation", None),
    "REORDER_UP": ("route_navigation", None),
    "MARK_TOGGLE": ("route_navigation", None),
    "MARK_RANGE": ("route_navigation", None),
    "MARK_CLEAR": ("route_navigation", None),
}


class TestDispatchRouteEngine:
    """The dispatch-table engine's contract, on whichever backend is loaded."""

    def test_every_action_routes_to_its_pinned_branch(self) -> None:
        for member in Action:
            branch, page = _EXPECTED_ROUTES[member.name]
            assert _dispatch_route(member, _DISPATCH_TABLE) == {
                "branch": branch,
                "page": page,
            }

    def test_the_table_covers_every_member_exactly_once(self) -> None:
        members = [member for member, _, _ in _DISPATCH_TABLE]
        assert len(members) == len(set(members)) == len(list(Action))
        assert set(members) == set(Action)

    def test_none_dispatches_nothing(self) -> None:
        assert _dispatch_route(None, _DISPATCH_TABLE) == {"branch": "none", "page": None}

    def test_plain_string_values_dispatch_like_their_members(self) -> None:
        # Action is a str enum, and the reference's value patterns compare
        # with == — so "play_pause" dispatches exactly like the member,
        # on both backends.
        for member in Action:
            assert _dispatch_route(member.value, _DISPATCH_TABLE) == _dispatch_route(
                member, _DISPATCH_TABLE
            )

    def test_unclaimed_subjects_are_unhandled(self) -> None:
        for junk in ("zzz", "", " play_pause", "PLAY_PAUSE", 5, ("play_pause",), []):
            assert _dispatch_route(junk, _DISPATCH_TABLE) == {
                "branch": "unhandled",
                "page": None,
            }

    def test_outcome_map_shape(self) -> None:
        assert set(_dispatch_route(Action.QUIT, _DISPATCH_TABLE)) == {"branch", "page"}

    def test_outcomes_are_fresh_and_inputs_never_mutated(self) -> None:
        table = [(Action.QUIT, "quit", None)]
        first = _dispatch_route(Action.QUIT, table)
        second = _dispatch_route(Action.QUIT, table)
        assert first == second
        assert first is not second
        assert first["page"] is None
        assert table == [(Action.QUIT, "quit", None)]

    def test_first_equal_row_wins(self) -> None:
        table = [
            (Action.QUIT, "first", None),
            (Action.QUIT, "second", None),
        ]
        assert _dispatch_route(Action.QUIT, table)["branch"] == "first"

    def test_row_comparison_is_python_eq_with_the_subject_on_the_left(self) -> None:
        # The reference's `case Action.X` compiles to `subject == member`
        # (subject on the left); an always-equal member therefore matches
        # any subject, exactly like it would in the match statement.
        class AlwaysEq:
            def __eq__(self, other: object) -> bool:
                return True

        assert _dispatch_route("anything", [(AlwaysEq(), "hit", None)])["branch"] == "hit"

    def test_stub_table_tokens_pass_through_unchanged(self) -> None:
        out = _dispatch_route("x", [("x", ("token",), 5)])
        assert out == {"branch": ("token",), "page": 5}

    def test_generator_tables_and_rows_are_accepted(self) -> None:
        table = (
            row for row in ((name, branch, page) for name, branch, page in [("x", "hit", None)])
        )
        assert _dispatch_route("x", table)["branch"] == "hit"

    def test_row_unpack_arity_errors_match_cpython(self) -> None:
        for row, message in [
            ((), "not enough values to unpack \\(expected 3, got 0\\)"),
            (("x",), "not enough values to unpack \\(expected 3, got 1\\)"),
            (("x", "y"), "not enough values to unpack \\(expected 3, got 2\\)"),
            (("x", "y", "z", "w"), "too many values to unpack \\(expected 3\\)"),
        ]:
            with pytest.raises(ValueError, match=message):
                _dispatch_route("x", [row])

    def test_non_iterable_row_message_matches_cpython(self) -> None:
        with pytest.raises(TypeError, match="cannot unpack non-iterable int object"):
            _dispatch_route("x", [5])

    def test_iter_raising_row_keeps_its_own_error(self) -> None:
        # CPython only rewrites the message for a type with no iteration
        # support at all; an __iter__ that itself raises propagates.
        class BadIter:
            def __iter__(self) -> object:
                raise TypeError("boom from iter")

        with pytest.raises(TypeError, match="boom from iter"):
            _dispatch_route("x", [BadIter()])

    def test_non_iterable_table_message_matches_cpython(self) -> None:
        with pytest.raises(TypeError, match="'int' object is not iterable"):
            _dispatch_route("x", 5)


class TestNavigationRouteEngine:
    """The pane-routing engine's contract, on whichever backend is loaded."""

    def test_playlists_pane_receives_its_actions(self) -> None:
        for action in _SIDEBAR_PANE_ACTIONS:
            assert (
                _navigation_action_route(
                    "playlists", action, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
                )
                == "sidebar"
            )

    def test_lyrics_pane_receives_its_actions(self) -> None:
        for action in _LYRICS_PANE_ACTIONS:
            assert (
                _navigation_action_route(
                    "lyrics", action, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
                )
                == "lyrics"
            )

    def test_both_set_members_still_route_to_the_sidebar_from_playlists(self) -> None:
        # MOVE_DOWN is in both sets; from the playlists pane the sidebar
        # gate wins — the reference's elif is unreachable there because
        # the pane can't be both playlists and lyrics.
        assert (
            _navigation_action_route(
                "playlists", Action.MOVE_DOWN, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
            )
            == "sidebar"
        )

    def test_non_sidebar_actions_reach_the_page_from_playlists(self) -> None:
        for action in (Action.SORT_TITLE, Action.MARK_TOGGLE, Action.REORDER_DOWN):
            assert (
                _navigation_action_route(
                    "playlists", action, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
                )
                == "page"
            )

    def test_select_is_not_a_lyrics_action(self) -> None:
        assert (
            _navigation_action_route(
                "lyrics", Action.SELECT, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
            )
            == "page"
        )

    def test_content_and_unknown_panes_route_to_the_page(self) -> None:
        for pane in ("content", "", "nonsense"):
            assert (
                _navigation_action_route(
                    pane, Action.MOVE_DOWN, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
                )
                == "page"
            )

    def test_plain_string_actions_route_like_members(self) -> None:
        assert (
            _navigation_action_route(
                "playlists", "select", _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
            )
            == "sidebar"
        )
        assert (
            _navigation_action_route(
                "lyrics", "move_down", _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
            )
            == "lyrics"
        )

    def test_stub_sets_decide_membership(self) -> None:
        # Membership goes through Python's in, so a stub's sets decide
        # exactly like the module's frozensets.
        assert _navigation_action_route("playlists", "weird", {"weird"}, set()) == "sidebar"
        assert _navigation_action_route("lyrics", "weird", set(), {"weird"}) == "lyrics"
        assert _navigation_action_route("playlists", "weird", set(), {"weird"}) == "page"


class TestCurrentContextTargetEngine:
    """The album-target engine's contract, on whichever backend is loaded."""

    def test_no_track_notifies(self) -> None:
        for track in (None, {}):
            assert _current_context_target(track) == {
                "outcome": "notify",
                "reason": "no_track",
            }

    def test_album_id_navigates(self) -> None:
        assert _current_context_target({"album_id": "ALB"}) == {
            "outcome": "navigate",
            "album_id": "ALB",
        }

    def test_falsy_album_id_falls_back_to_the_album_dict(self) -> None:
        for falsy in ("", None, 0):
            out = _current_context_target({"album_id": falsy, "album": {"id": "ALB2"}})
            assert out == {"outcome": "navigate", "album_id": "ALB2"}

    def test_non_dict_album_never_provides_an_id(self) -> None:
        assert _current_context_target({"album": "Greatest Hits"}) == {
            "outcome": "notify",
            "reason": "no_album",
        }

    def test_missing_everything_notifies_no_album(self) -> None:
        assert _current_context_target({"title": "T"}) == {
            "outcome": "notify",
            "reason": "no_album",
        }

    def test_album_dict_without_id_notifies_no_album(self) -> None:
        assert _current_context_target({"album": {}}) == {
            "outcome": "notify",
            "reason": "no_album",
        }

    def test_truthy_non_string_ids_pass_through(self) -> None:
        assert _current_context_target({"album_id": 5}) == {
            "outcome": "navigate",
            "album_id": 5,
        }

    def test_dict_subclass_album_counts_as_a_dict(self) -> None:
        class Album(dict):
            pass

        assert _current_context_target({"album": Album(id="SUB")}) == {
            "outcome": "navigate",
            "album_id": "SUB",
        }

    def test_duck_typed_track_is_read_via_get(self) -> None:
        class Track:
            def get(self, key: str) -> object:
                return {"album_id": "DUCK"}[key]

        assert _current_context_target(Track()) == {
            "outcome": "navigate",
            "album_id": "DUCK",
        }

    def test_inputs_never_mutated(self) -> None:
        track = {"album": {"id": "ALB"}}
        _current_context_target(track)
        assert track == {"album": {"id": "ALB"}}

    def test_track_without_get_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            _current_context_target(object())


class TestShuffleLockEngine:
    """The Shuffle-lock gate's contract, on whichever backend is loaded."""

    def test_no_context_is_never_locked(self) -> None:
        for context_id in (None, ""):
            assert _shuffle_lock_blocked(context_id, {}) is False

    def test_short_circuit_never_consults_prefs(self) -> None:
        class Exploding:
            def get(self, key: object) -> bool:
                raise AssertionError("prefs.get must not be called")

        assert _shuffle_lock_blocked(None, Exploding()) is False
        assert _shuffle_lock_blocked("", Exploding()) is False

    def test_stored_preference_decides(self) -> None:
        prefs = {"PL1": True, "PL2": False, "PL3": None, "PL4": 1, "PL5": ""}
        assert _shuffle_lock_blocked("PL1", prefs) is True
        assert _shuffle_lock_blocked("PL4", prefs) is True
        assert _shuffle_lock_blocked("PL2", prefs) is False
        assert _shuffle_lock_blocked("PL3", prefs) is False
        assert _shuffle_lock_blocked("PL5", prefs) is False
        assert _shuffle_lock_blocked("missing", prefs) is False

    def test_prefs_without_get_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            _shuffle_lock_blocked("PL1", object())


# ── _handle_action glue ─────────────────────────────────────────────


class TestHandleActionDispatch:
    """The mixin applies the engine's branch: each branch performs its
    handler calls, and nothing else is touched along the way."""

    def _host(self) -> MagicMock:
        host = MagicMock()
        host.navigate_to = AsyncMock()
        host.play_track = AsyncMock()
        host._toggle_play_pause = AsyncMock()
        host._play_next = AsyncMock()
        host._play_previous = AsyncMock()
        host._open_add_to_playlist = AsyncMock()
        host._open_track_actions = AsyncMock()
        host._toggle_like_current = AsyncMock()
        host.player = AsyncMock()
        return host

    async def test_none_action_calls_nothing(self, caplog) -> None:
        host = self._host()
        with caplog.at_level("DEBUG", logger="ytm_player.app._keys"):
            await KeyHandlingMixin._handle_action(host, None)
        host._toggle_play_pause.assert_not_called()
        host.navigate_to.assert_not_awaited()
        host.notify.assert_not_called()
        host.exit.assert_not_called()
        # The reference returned before the dispatch table was ever
        # consulted — no "Unhandled action" debug line for None.
        assert "Unhandled action" not in caplog.text

    async def test_unhandled_action_only_logs(self, caplog) -> None:
        host = self._host()
        with caplog.at_level("DEBUG", logger="ytm_player.app._keys"):
            await KeyHandlingMixin._handle_action(host, "bogus")  # type: ignore[arg-type]
        host._toggle_play_pause.assert_not_called()
        host.navigate_to.assert_not_awaited()
        host.notify.assert_not_called()
        assert "Unhandled action: bogus" in caplog.text

    async def test_toggle_play_pause_branch(self) -> None:
        host = self._host()
        await KeyHandlingMixin._handle_action(host, Action.PLAY_PAUSE)
        host._toggle_play_pause.assert_awaited_once()

    async def test_volume_branches_scale_with_the_count(self) -> None:
        host = self._host()
        await KeyHandlingMixin._handle_action(host, Action.VOLUME_UP, 3)
        host.player.change_volume.assert_awaited_once_with(15)
        await KeyHandlingMixin._handle_action(host, Action.VOLUME_DOWN, 2)
        host.player.change_volume.assert_awaited_with(-10)

    async def test_volume_without_a_player_is_a_noop(self) -> None:
        host = self._host()
        host.player = None
        await KeyHandlingMixin._handle_action(host, Action.VOLUME_UP, 3)
        host.notify.assert_not_called()

    async def test_seek_branches_use_the_configured_step(self) -> None:
        host = self._host()
        host.settings.playback.seek_step = 7
        await KeyHandlingMixin._handle_action(host, Action.SEEK_FORWARD, 2)
        host.player.seek.assert_awaited_once_with(14)
        await KeyHandlingMixin._handle_action(host, Action.SEEK_BACKWARD, 2)
        host.player.seek.assert_awaited_with(-14)

    async def test_navigate_branches_carry_their_page(self) -> None:
        host = self._host()
        for member, page in [
            (Action.LIBRARY, "library"),
            (Action.SEARCH, "search"),
            (Action.QUEUE, "queue"),
            (Action.BROWSE, "browse"),
            (Action.HELP, "help"),
            (Action.LIKED_SONGS, "liked_songs"),
            (Action.RECENTLY_PLAYED, "recently_played"),
            (Action.GO_BACK, "back"),
            (Action.GO_FORWARD, "forward"),
        ]:
            await KeyHandlingMixin._handle_action(host, member)
            host.navigate_to.assert_awaited_with(page)

    async def test_quit_branch_exits(self) -> None:
        host = self._host()
        await KeyHandlingMixin._handle_action(host, Action.QUIT)
        host.exit.assert_called_once()

    async def test_play_random_branch_gates_on_the_draw(self) -> None:
        host = self._host()
        host.queue.play_random.return_value = None
        await KeyHandlingMixin._handle_action(host, Action.PLAY_RANDOM)
        host.play_track.assert_not_awaited()

        track = {"video_id": "v1"}
        host.queue.play_random.return_value = track
        await KeyHandlingMixin._handle_action(host, Action.PLAY_RANDOM)
        host.play_track.assert_awaited_once_with(track)

    async def test_current_context_branch_navigates_to_the_album(self) -> None:
        host = self._host()
        host.queue.current_track = {"album_id": "ALB"}
        await KeyHandlingMixin._handle_action(host, Action.CURRENT_CONTEXT)
        host.navigate_to.assert_awaited_once_with("context", context_type="album", context_id="ALB")

    async def test_current_context_branch_falls_back_to_the_album_dict(self) -> None:
        host = self._host()
        host.queue.current_track = {"album": {"id": "ALB2"}}
        await KeyHandlingMixin._handle_action(host, Action.CURRENT_CONTEXT)
        host.navigate_to.assert_awaited_once_with(
            "context", context_type="album", context_id="ALB2"
        )

    async def test_current_context_branch_notifies_without_an_album(self) -> None:
        host = self._host()
        host.queue.current_track = {"title": "T"}
        await KeyHandlingMixin._handle_action(host, Action.CURRENT_CONTEXT)
        host.navigate_to.assert_not_awaited()
        assert host.notify.call_args.args == ("No album info for current track",)

    async def test_current_context_branch_notifies_without_a_track(self) -> None:
        host = self._host()
        host.queue.current_track = None
        await KeyHandlingMixin._handle_action(host, Action.CURRENT_CONTEXT)
        host.navigate_to.assert_not_awaited()
        assert host.notify.call_args.args == ("No track playing",)

    async def test_route_navigation_branch_delegates(self) -> None:
        host = self._host()
        host._route_navigation_action = AsyncMock()
        await KeyHandlingMixin._handle_action(host, Action.MOVE_DOWN, 4)
        host._route_navigation_action.assert_awaited_once_with(Action.MOVE_DOWN, 4)


class TestRouteNavigationActionGlue:
    """The pane-routing glue: widget lookups, the missing-widget
    fall-back-to-page, and the no-page no-op."""

    def _host(self, pane: str, page: object) -> KeyHandlingMixin:
        host = KeyHandlingMixin()
        host._current_pane = MagicMock(return_value=pane)  # type: ignore[method-assign]
        host._get_current_page = MagicMock(return_value=page)  # type: ignore[method-assign]
        return host

    async def test_missing_sidebar_widget_falls_back_to_the_page(self) -> None:
        host = self._host("playlists", MagicMock())
        host._get_current_page.return_value.handle_action = AsyncMock()
        host.query_one = MagicMock(side_effect=Exception("no widget in tests"))
        await host._route_navigation_action(Action.SELECT, 1)
        host._get_current_page.return_value.handle_action.assert_awaited_once_with(Action.SELECT, 1)

    async def test_missing_lyrics_widget_falls_back_to_the_page(self) -> None:
        host = self._host("lyrics", MagicMock())
        host._get_current_page.return_value.handle_action = AsyncMock()
        host.query_one = MagicMock(side_effect=Exception("no widget in tests"))
        await host._route_navigation_action(Action.MOVE_DOWN, 1)
        host._get_current_page.return_value.handle_action.assert_awaited_once_with(
            Action.MOVE_DOWN, 1
        )

    async def test_no_page_means_no_dispatch(self) -> None:
        host = self._host("content", None)
        host.query_one = MagicMock(side_effect=Exception("no widget in tests"))
        await host._route_navigation_action(Action.MOVE_DOWN, 1)  # must not raise

    async def test_page_without_handle_action_is_skipped(self) -> None:
        host = self._host("content", object())  # truthy, but no handle_action
        host.query_one = MagicMock(side_effect=Exception("no widget in tests"))
        await host._route_navigation_action(Action.MOVE_DOWN, 1)  # must not raise


# ── Dispatch-engine native vs fallback parity ───────────────────────

# Subject/table specs covering every branch of the dispatch engine: the
# whole real table (every member, None, the plain-string str-enum wart,
# junk subjects, custom-equality objects — an always-equal subject
# matches the FIRST row, a never-equal one nothing), stub tables
# (first-match-wins, tokens passed through, generator tables and rows),
# and the CPython for-loop unpack errors (arity ValueErrors, the
# non-iterable row/table TypeErrors, an __iter__ that itself raises).
_REAL_TABLE = {"t": "real"}
_DISPATCH_PARITY_CASES: list[tuple[dict, dict]] = [
    # Every action through the real table.
    *[
        ({"t": "action", "name": member.name}, _REAL_TABLE)
        for member in sorted(Action, key=lambda m: m.name)
    ],
    # None, the plain-string wart, and unclaimed subjects.
    ({"t": "none"}, _REAL_TABLE),
    *[
        ({"t": "str", "v": member.value}, _REAL_TABLE)
        for member in sorted(Action, key=lambda m: m.name)
    ],
    ({"t": "str", "v": "zzz"}, _REAL_TABLE),
    ({"t": "str", "v": ""}, _REAL_TABLE),
    ({"t": "str", "v": " play_pause"}, _REAL_TABLE),
    ({"t": "str", "v": "PLAY_PAUSE"}, _REAL_TABLE),
    ({"t": "json", "v": 5}, _REAL_TABLE),
    ({"t": "json", "v": ["play_pause"]}, _REAL_TABLE),
    ({"t": "json", "v": True}, _REAL_TABLE),
    # Custom equality: always-equal matches the first row, never-equal
    # nothing — on both backends, because the comparison is the
    # reference's own `subject == member`.
    ({"t": "alwayseq"}, _REAL_TABLE),
    ({"t": "nevereq"}, _REAL_TABLE),
    # Stub tables: first-match-wins, pass-through tokens, generators.
    (
        {"t": "str", "v": "x"},
        {
            "t": "rows",
            "rows": [
                [{"t": "str", "v": "x"}, "first", None],
                [{"t": "str", "v": "x"}, "second", None],
            ],
        },
    ),
    ({"t": "str", "v": "anything"}, {"t": "rows", "rows": [[{"t": "alwayseq"}, "hit", None]]}),
    ({"t": "str", "v": "anything"}, {"t": "rows", "rows": [[{"t": "nevereq"}, "hit", None]]}),
    (
        {"t": "str", "v": "x"},
        {"t": "rows", "rows": [[{"t": "str", "v": "x"}, "token", "page"]]},
    ),
    (
        {"t": "str", "v": "x"},
        {"t": "rows_gen", "rows": [[{"t": "str", "v": "x"}, "hit", None]]},
    ),
    ({"t": "str", "v": "x"}, {"t": "empty"}),
    ({"t": "none"}, {"t": "empty"}),
    # CPython's for-loop unpack errors, verbatim on both backends.
    ({"t": "str", "v": "x"}, {"t": "raw_rows", "rows": [[]]}),
    ({"t": "str", "v": "x"}, {"t": "raw_rows", "rows": [["x"]]}),
    ({"t": "str", "v": "x"}, {"t": "raw_rows", "rows": [["x", "y"]]}),
    ({"t": "str", "v": "x"}, {"t": "raw_rows", "rows": [["x", "y", "z", "w"]]}),
    ({"t": "str", "v": "x"}, {"t": "raw_rows", "rows": [5]}),
    ({"t": "str", "v": "x"}, {"t": "raw_rows", "rows": [None]}),
    ({"t": "str", "v": "x"}, {"t": "baditer_row"}),
    ({"t": "str", "v": "x"}, {"t": "non_iterable"}),
]

_REAL_SIDEBAR = {"t": "real_sidebar"}
_REAL_LYRICS = {"t": "real_lyrics"}
_EMPTY_SET = {"t": "empty"}
_NAV_PARITY_CASES: list[tuple[str, dict, dict, dict]] = [
    # Every sidebar action from the playlists pane, every lyrics action
    # from the lyrics pane.
    *[
        ("playlists", {"t": "action", "name": action.name}, _REAL_SIDEBAR, _REAL_LYRICS)
        for action in sorted(_SIDEBAR_PANE_ACTIONS, key=lambda m: m.name)
    ],
    *[
        ("lyrics", {"t": "action", "name": action.name}, _REAL_SIDEBAR, _REAL_LYRICS)
        for action in sorted(_LYRICS_PANE_ACTIONS, key=lambda m: m.name)
    ],
    # The both-sets member from each pane, the falls-through actions,
    # unknown panes, the plain-string wart and stub sets.
    ("playlists", {"t": "action", "name": "MOVE_DOWN"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("playlists", {"t": "action", "name": "SORT_TITLE"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("playlists", {"t": "action", "name": "MARK_CLEAR"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("lyrics", {"t": "action", "name": "SELECT"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("lyrics", {"t": "action", "name": "FILTER"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("content", {"t": "action", "name": "MOVE_DOWN"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("", {"t": "action", "name": "MOVE_DOWN"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("nonsense", {"t": "action", "name": "MOVE_DOWN"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("playlists", {"t": "str", "v": "select"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("lyrics", {"t": "str", "v": "move_down"}, _REAL_SIDEBAR, _REAL_LYRICS),
    ("playlists", {"t": "str", "v": "weird"}, {"t": "set", "items": ["weird"]}, _EMPTY_SET),
    ("lyrics", {"t": "str", "v": "weird"}, _EMPTY_SET, {"t": "set", "items": ["weird"]}),
    ("playlists", {"t": "str", "v": "weird"}, _EMPTY_SET, {"t": "set", "items": ["weird"]}),
    (
        "nonsense",
        {"t": "str", "v": "weird"},
        {"t": "set", "items": ["weird"]},
        {"t": "set", "items": ["weird"]},
    ),
]

_CONTEXT_PARITY_CASES: list[dict] = [
    {"t": "none"},
    {"t": "empty_dict"},
    {"t": "dict", "d": {"album_id": "ALB"}},
    {"t": "dict", "d": {"album": {"id": "ALB"}}},
    {"t": "dict", "d": {"album_id": "", "album": {"id": "ALB2"}}},
    {"t": "dict", "d": {"album_id": None, "album": {"id": "ALB2"}}},
    {"t": "dict", "d": {"album_id": 0, "album": {"id": "ALB2"}}},
    {"t": "dict", "d": {"album": "Greatest Hits"}},
    {"t": "dict", "d": {"album": False}},
    {"t": "dict", "d": {"title": "T"}},
    {"t": "dict", "d": {"album": {}}},
    {"t": "dict", "d": {"album_id": 5}},
    {"t": "album_subclass", "d": {"id": "SUB"}},
    {"t": "duck", "d": {"album_id": "DUCK"}},
    {"t": "no_get"},
]

_SHUFFLE_PARITY_CASES: list[tuple[dict, dict]] = [
    ({"t": "none"}, {"t": "dict", "d": {}}),
    ({"t": "str", "v": ""}, {"t": "dict", "d": {}}),
    ({"t": "json", "v": 0}, {"t": "dict", "d": {"0": True}}),
    ({"t": "str", "v": "PL1"}, {"t": "dict", "d": {"PL1": True}}),
    ({"t": "str", "v": "PL1"}, {"t": "dict", "d": {"PL1": False}}),
    ({"t": "str", "v": "PL1"}, {"t": "dict", "d": {"PL1": None}}),
    ({"t": "str", "v": "PL1"}, {"t": "dict", "d": {"PL1": 1}}),
    ({"t": "str", "v": "PL1"}, {"t": "dict", "d": {"PL1": ""}}),
    ({"t": "str", "v": "PL1"}, {"t": "dict", "d": {}}),
    ({"t": "str", "v": "0"}, {"t": "dict", "d": {"0": True}}),
    ({"t": "str", "v": "PL1"}, {"t": "no_get"}),
]

# The builder/run vocabulary shared with the child interpreter: the
# child blocks the native import and runs the pure-Python reference
# engines over the same specs, so any divergence between the backends
# (outcomes, passed-through tokens, error shapes and messages) fails.
_DISPATCH_ENGINES_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._keys import (
    _DISPATCH_TABLE,
    _LYRICS_PANE_ACTIONS,
    _SIDEBAR_PANE_ACTIONS,
    _current_context_target,
    _dispatch_route,
    _navigation_action_route,
    _shuffle_lock_blocked,
)
from ytm_player.config import Action


class AlwaysEq:
    def __eq__(self, other):
        return True


class NeverEq:
    def __eq__(self, other):
        return False


class BadIter:
    def __iter__(self):
        raise TypeError("boom from iter")


class AlbumDict(dict):
    pass


class DuckTrack:
    def __init__(self, data):
        self._data = data

    def get(self, key):
        return self._data[key]


def build_subject(spec):
    kind = spec["t"]
    if kind == "none":
        return None
    if kind == "action":
        return Action[spec["name"]]
    if kind in ("str", "json"):
        return spec["v"]
    if kind == "alwayseq":
        return AlwaysEq()
    if kind == "nevereq":
        return NeverEq()
    raise AssertionError(kind)


def build_table(spec):
    kind = spec["t"]
    if kind == "real":
        return _DISPATCH_TABLE
    if kind == "rows":
        return tuple((build_subject(m), b, p) for m, b, p in spec["rows"])
    if kind == "rows_gen":
        rows = tuple((build_subject(m), b, p) for m, b, p in spec["rows"])
        return (row for row in rows)
    if kind == "raw_rows":
        return tuple(spec["rows"])
    if kind == "baditer_row":
        return [BadIter()]
    if kind == "non_iterable":
        return 5
    if kind == "empty":
        return ()
    raise AssertionError(kind)


def build_set(spec):
    kind = spec["t"]
    if kind == "real_sidebar":
        return _SIDEBAR_PANE_ACTIONS
    if kind == "real_lyrics":
        return _LYRICS_PANE_ACTIONS
    if kind == "set":
        return set(spec["items"])
    if kind == "empty":
        return set()
    raise AssertionError(kind)


def build_track(spec):
    kind = spec["t"]
    if kind == "none":
        return None
    if kind == "empty_dict":
        return {}
    if kind == "dict":
        return spec["d"]
    if kind == "album_subclass":
        return {"album": AlbumDict(spec["d"])}
    if kind == "duck":
        return DuckTrack(spec["d"])
    if kind == "no_get":
        return object()
    raise AssertionError(kind)


def build_prefs(spec):
    if spec["t"] == "dict":
        return spec["d"]
    if spec["t"] == "no_get":
        return object()
    raise AssertionError(spec["t"])


def outcome(fn, *args):
    try:
        return {"ok": fn(*args)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
results = {}
for case in payload["dispatch"]:
    subject, table = case
    results.setdefault("dispatch", []).append(
        outcome(_dispatch_route, build_subject(subject), build_table(table))
    )
for pane, action, sidebar, lyrics in payload["nav"]:
    results.setdefault("nav", []).append(
        outcome(
            _navigation_action_route,
            pane,
            build_subject(action),
            build_set(sidebar),
            build_set(lyrics),
        )
    )
for track in payload["context"]:
    results.setdefault("context", []).append(
        outcome(_current_context_target, build_track(track))
    )
for context_id, prefs in payload["shuffle"]:
    results.setdefault("shuffle", []).append(
        outcome(_shuffle_lock_blocked, build_subject(context_id), build_prefs(prefs))
    )
json.dump(results, sys.stdout)
"""


class _AlwaysEq:
    def __eq__(self, other: object) -> bool:
        return True


class _NeverEq:
    def __eq__(self, other: object) -> bool:
        return False


class _BadIter:
    def __iter__(self) -> object:
        raise TypeError("boom from iter")


class _AlbumDict(dict):
    pass


class _DuckTrack:
    def __init__(self, data: dict) -> None:
        self._data = data

    def get(self, key: str) -> object:
        return self._data[key]


def _build_subject(spec: dict) -> object:
    kind = spec["t"]
    if kind == "none":
        return None
    if kind == "action":
        return Action[spec["name"]]
    if kind in ("str", "json"):
        return spec["v"]
    if kind == "alwayseq":
        return _AlwaysEq()
    if kind == "nevereq":
        return _NeverEq()
    raise AssertionError(kind)


def _build_table(spec: dict) -> object:
    kind = spec["t"]
    if kind == "real":
        return _DISPATCH_TABLE
    if kind == "rows":
        return tuple((_build_subject(m), b, p) for m, b, p in spec["rows"])
    if kind == "rows_gen":
        rows = tuple((_build_subject(m), b, p) for m, b, p in spec["rows"])
        return (row for row in rows)
    if kind == "raw_rows":
        return tuple(spec["rows"])
    if kind == "baditer_row":
        return [_BadIter()]
    if kind == "non_iterable":
        return 5
    if kind == "empty":
        return ()
    raise AssertionError(kind)


def _build_set(spec: dict) -> object:
    kind = spec["t"]
    if kind == "real_sidebar":
        return _SIDEBAR_PANE_ACTIONS
    if kind == "real_lyrics":
        return _LYRICS_PANE_ACTIONS
    if kind == "set":
        return set(spec["items"])
    if kind == "empty":
        return set()
    raise AssertionError(kind)


def _build_track(spec: dict) -> object:
    kind = spec["t"]
    if kind == "none":
        return None
    if kind == "empty_dict":
        return {}
    if kind == "dict":
        return spec["d"]
    if kind == "album_subclass":
        return {"album": _AlbumDict(spec["d"])}
    if kind == "duck":
        return _DuckTrack(spec["d"])
    if kind == "no_get":
        return object()
    raise AssertionError(kind)


def _build_prefs(spec: dict) -> object:
    if spec["t"] == "dict":
        return spec["d"]
    if spec["t"] == "no_get":
        return object()
    raise AssertionError(spec["t"])


def _outcome(fn: object, *args: object) -> dict:
    try:
        return {"ok": fn(*args)}  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def _run_child_engines() -> dict:
    payload = {
        "dispatch": _DISPATCH_PARITY_CASES,
        "nav": _NAV_PARITY_CASES,
        "context": _CONTEXT_PARITY_CASES,
        "shuffle": _SHUFFLE_PARITY_CASES,
    }
    result = subprocess.run(
        [sys.executable, "-c", _DISPATCH_ENGINES_CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _native_engines() -> dict:
    results: dict[str, list[dict]] = {}
    for subject, table in _DISPATCH_PARITY_CASES:
        out = _outcome(_dispatch_route, _build_subject(subject), _build_table(table))
        if "ok" in out:
            out = {"ok": {"branch": out["ok"]["branch"], "page": out["ok"]["page"]}}
        results.setdefault("dispatch", []).append(out)
    for pane, action, sidebar, lyrics in _NAV_PARITY_CASES:
        results.setdefault("nav", []).append(
            _outcome(
                _navigation_action_route,
                pane,
                _build_subject(action),
                _build_set(sidebar),
                _build_set(lyrics),
            )
        )
    for track in _CONTEXT_PARITY_CASES:
        results.setdefault("context", []).append(
            _outcome(_current_context_target, _build_track(track))
        )
    for context_id, prefs in _SHUFFLE_PARITY_CASES:
        results.setdefault("shuffle", []).append(
            _outcome(_shuffle_lock_blocked, _build_subject(context_id), _build_prefs(prefs))
        )
    return results


class TestDispatchEnginesNativeParity:
    """When the Rust extension is built, the native dispatch engines must
    produce exactly the same outcomes — routed branch and page, pane
    target, album-target map, lock verdict, and the same errors with the
    same messages — as the pure-Python references for every case, so the
    two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_dispatch_engines_match_fallback(self) -> None:
        native = _native_engines()
        fallback = _run_child_engines()
        for suite, cases in (
            ("dispatch", _DISPATCH_PARITY_CASES),
            ("nav", _NAV_PARITY_CASES),
            ("context", _CONTEXT_PARITY_CASES),
            ("shuffle", _SHUFFLE_PARITY_CASES),
        ):
            native_results = native[suite]
            fallback_results = fallback[suite]
            assert len(native_results) == len(fallback_results) == len(cases)
            for case, n, f in zip(cases, native_results, fallback_results):
                assert n == f, (
                    f"native and fallback {suite} outcomes differ:\n"
                    f"case:    {case!r}\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
