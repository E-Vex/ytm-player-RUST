"""Tests for the input popup's submitted-value engine.

``_submitted_value`` (ui/popups/input_popup.py) runs with two
interchangeable backends: the pure-Python transcription in the module
and, when the Rust extension is built, the native implementation bound
at the module's tail. The engine: the submitted value stripped, an
empty or all-whitespace submission answering ``None`` (the popup
dismisses with the answer either way).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.popups.input_popup import InputPopup, _submitted_value

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestSubmittedValue:
    """The normalization (whichever backend)."""

    def test_plain_value_passes_through(self):
        assert _submitted_value("hello") == "hello"

    def test_surrounding_whitespace_is_stripped(self):
        assert _submitted_value("  hello\t\n") == "hello"

    def test_empty_submits_none(self):
        assert _submitted_value("") is None

    def test_whitespace_only_submits_none(self):
        assert _submitted_value("   \t\r\n") is None

    def test_internal_whitespace_survives(self):
        assert _submitted_value("  a b\tc  ") == "a b\tc"

    def test_a_zero_string_is_a_value_not_none(self):
        assert _submitted_value("0") == "0"

    def test_cpystons_own_strip_set_is_respected(self):
        # ``\\x1c``–``\\x1f``, ``\\x85`` and ``\\xa0`` are in CPython's
        # ``str.isspace()`` set (so ``str.strip()`` trims them) but not
        # in Rust's ``str::trim`` White_Space property — the engine must
        # follow the CPython set, exactly like the fallback.
        assert _submitted_value("\x1c\x1d\x1e\x1f x \x85\xa0") == "x"

    def test_unicode_whitespace_forms(self):
        assert _submitted_value("\u2028\u2029x\u3000") == "x"


class TestPopupWiring:
    """The popup's submit handler routes the normalized value into the
    dismissal — the stripped value when non-empty, None otherwise.
    """

    @staticmethod
    def _submit(value: str, dismissed: list) -> None:
        popup = InputPopup("Enter")
        popup.dismiss = dismissed.append  # type: ignore[method-assign]

        class _Event:
            def __init__(self, submitted: str) -> None:
                self.value = submitted

        popup.on_input_submitted(_Event(value))  # type: ignore[arg-type]

    def test_a_padded_answer_dismisses_stripped(self):
        dismissed: list = []
        self._submit("  answer  ", dismissed)
        assert dismissed == ["answer"]

    def test_a_blank_answer_dismisses_none(self):
        dismissed: list = []
        self._submit("   ", dismissed)
        assert dismissed == [None]


# ── Native vs fallback parity ─────────────────────────────────────────

_CASES = [
    "hello",
    "  padded  ",
    "",
    "   ",
    "\t\n\r x \x0b\x0c",
    "\x1c\x1d x \x1e\x1f",
    "\x85\xa0x\u2028\u2029",
    "\u3000x\u3000",
    "0",
    " a b ",
    "\u00a0",
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.input_popup import _submitted_value

payload = json.load(sys.stdin)
json.dump([_submitted_value(case) for case in payload], sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native normalization must
    equal the pure-Python reference's for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        payload = json.dumps(_CASES)
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_submitted_value(case) for case in _CASES]
        assert native == fallback
