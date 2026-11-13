"""Tests for the confirm popup's button-verdict engine.

``_button_result`` (ui/popups/confirm_popup.py) runs with two
interchangeable backends: the pure-Python transcription in the module
and, when the Rust extension is built, the native implementation bound
at the module's tail. The engine: the ``True``/``False`` dismissal
verdict for a button press — ``True`` only for the confirm button's id,
``False`` for the cancel button's, any other id, or a widget with no
id at all.

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

from ytm_player.ui.popups.confirm_popup import ConfirmPopup, _button_result

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestButtonResult:
    """The verdict (whichever backend)."""

    def test_confirm_button_confirms(self):
        assert _button_result("confirm-yes") is True

    def test_cancel_button_declines(self):
        assert _button_result("confirm-no") is False

    def test_other_ids_decline(self):
        assert _button_result("btn-create") is False
        assert _button_result("confirm") is False
        assert _button_result("") is False

    def test_the_match_is_exact_not_case_insensitive(self):
        assert _button_result("CONFIRM-YES") is False
        assert _button_result("confirm-yes ") is False
        assert _button_result(" confirm-yes") is False

    def test_a_widget_without_an_id_declines(self):
        """``Button.id`` is ``str | None``; a None id never confirms."""
        assert _button_result(None) is False


class TestPopupWiring:
    """The popup's button handler routes its verdict into the dismissal."""

    @staticmethod
    def _press(button_id: str | None, dismissed: list) -> None:
        popup = ConfirmPopup("Sure?")
        popup.dismiss = dismissed.append  # type: ignore[method-assign]

        class _Button:
            id = button_id

        class _Event:
            button = _Button()  # type: ignore[attr-defined]

        popup.on_button_pressed(_Event())  # type: ignore[arg-type]

    def test_yes_button_dismisses_true(self):
        dismissed: list = []
        self._press("confirm-yes", dismissed)
        assert dismissed == [True]

    def test_no_button_dismisses_false(self):
        dismissed: list = []
        self._press("confirm-no", dismissed)
        assert dismissed == [False]

    def test_missing_id_dismisses_false(self):
        dismissed: list = []
        self._press(None, dismissed)
        assert dismissed == [False]


# ── Native vs fallback parity ─────────────────────────────────────────

_CASES = [
    None,
    "confirm-yes",
    "confirm-no",
    "btn-create",
    "",
    "confirm-yes ",
    "CONFIRM-YES",
    "yes",
    "confirm-YES",
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.confirm_popup import _button_result

payload = json.load(sys.stdin)
json.dump([_button_result(case) for case in payload], sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native verdict must equal
    the pure-Python reference's for every case, so the two
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
        native = [_button_result(case) for case in _CASES]
        assert native == fallback
