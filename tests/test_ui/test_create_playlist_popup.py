"""Tests for CreatePlaylistPopup init, submit logic and popup engines."""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from ytm_player.ui.popups.create_playlist_popup import (
    PRIVACY_OPTIONS,
    CreatePlaylistPopup,
    _chrome_labels,
    _submit_values,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestInit:
    def test_defaults(self):
        popup = CreatePlaylistPopup()
        assert popup._initial_name == ""
        assert popup._initial_description == ""
        assert popup._initial_privacy == "PRIVATE"
        assert popup._edit_mode is False

    def test_edit_mode(self):
        popup = CreatePlaylistPopup(edit_mode=True)
        assert popup._edit_mode is True

    def test_prefills(self):
        popup = CreatePlaylistPopup(
            initial_name="My List",
            initial_description="A description",
            initial_privacy="PUBLIC",
        )
        assert popup._initial_name == "My List"
        assert popup._initial_description == "A description"
        assert popup._initial_privacy == "PUBLIC"

    def test_privacy_options_constant(self):
        assert PRIVACY_OPTIONS == [
            ("Private", "PRIVATE"),
            ("Public", "PUBLIC"),
            ("Unlisted", "UNLISTED"),
        ]


class TestSubmit:
    def test_submit_dismisses_with_tuple_when_name_present(self):
        popup = CreatePlaylistPopup()
        dismissed = []

        def fake_dismiss(value):
            dismissed.append(value)

        popup.dismiss = fake_dismiss

        popup.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#input-name": MagicMock(value="  My Playlist  "),
                "#input-description": MagicMock(value="  Desc  "),
                "#select-privacy": MagicMock(value="PUBLIC"),
            }[selector]
        )

        popup._submit()
        assert dismissed == [("My Playlist", "Desc", "PUBLIC")]

    def test_submit_warns_and_does_not_dismiss_when_name_empty(self):
        popup = CreatePlaylistPopup()
        dismissed = []

        def fake_dismiss(value):
            dismissed.append(value)

        popup.dismiss = fake_dismiss
        popup.notify = MagicMock()

        name_input = MagicMock(value="   ")
        popup.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#input-name": name_input,
                "#input-description": MagicMock(value=""),
                "#select-privacy": MagicMock(value="PRIVATE"),
            }[selector]
        )

        popup._submit()
        assert dismissed == []
        popup.notify.assert_called_once()
        assert popup.notify.call_args.kwargs.get("severity") == "warning"
        name_input.focus.assert_called_once()

    def test_submit_strips_whitespace(self):
        popup = CreatePlaylistPopup()
        dismissed = []

        def fake_dismiss(value):
            dismissed.append(value)

        popup.dismiss = fake_dismiss

        popup.query_one = MagicMock(
            side_effect=lambda selector, _type=None: {
                "#input-name": MagicMock(value="\t Playlist \n"),
                "#input-description": MagicMock(value="\n Description \t"),
                "#select-privacy": MagicMock(value="UNLISTED"),
            }[selector]
        )

        popup._submit()
        assert dismissed == [("Playlist", "Description", "UNLISTED")]


class TestButtonPressed:
    def test_cancel_button_dismisses_none(self):
        popup = CreatePlaylistPopup()
        dismissed = []

        def fake_dismiss(value):
            dismissed.append(value)

        popup.dismiss = fake_dismiss

        class FakeButton:
            id = "btn-cancel"

        class FakeEvent:
            button = FakeButton()

        popup.on_button_pressed(FakeEvent())
        assert dismissed == [None]

    def test_create_button_calls_submit(self):
        popup = CreatePlaylistPopup()
        submitted = []

        def fake_submit():
            submitted.append(True)

        popup._submit = fake_submit

        class FakeButton:
            id = "btn-create"

        class FakeEvent:
            button = FakeButton()

        popup.on_button_pressed(FakeEvent())
        assert submitted == [True]


# ── Chrome-labels engine ─────────────────────────────────────────────


class TestChromeLabels:
    """The title / submit-button pair (whichever backend)."""

    def test_create_mode(self):
        assert _chrome_labels(False) == ("New Playlist", "Create")

    def test_edit_mode(self):
        assert _chrome_labels(True) == ("Edit Playlist", "Edit")


# ── Submit-values engine ─────────────────────────────────────────────


class _BadStr:
    """An object whose str() raises — pins the gate-before-str() order."""

    def __str__(self):
        raise RuntimeError("no str for you")


class TestSubmitValues:
    """The submission normalization (whichever backend)."""

    def test_strips_name_and_description(self):
        assert _submit_values("  My Playlist  ", "  Desc  ", "PUBLIC") == (
            "My Playlist",
            "Desc",
            "PUBLIC",
        )

    def test_whitespace_only_name_answers_none(self):
        assert _submit_values("   ", "Desc", "PUBLIC") is None

    def test_empty_name_answers_none(self):
        assert _submit_values("", "Desc", "PUBLIC") is None

    def test_an_empty_description_is_allowed(self):
        assert _submit_values("Name", "   ", "UNLISTED") == ("Name", "", "UNLISTED")

    def test_cpython_whitespace_set_is_respected(self):
        # \x1c-\x1f, \x85 and \xa0 are in CPython's str.isspace() set but
        # not in Rust's str::trim White_Space property.
        assert _submit_values("\x1c Name \x85\xa0", "\x1f Desc \x1d", "PUBLIC") == (
            "Name",
            "Desc",
            "PUBLIC",
        )

    def test_the_privacy_runs_through_str(self):
        assert _submit_values("N", "D", None) == ("N", "D", "None")
        assert _submit_values("N", "D", 5) == ("N", "D", "5")
        assert _submit_values("N", "D", True) == ("N", "D", "True")
        assert _submit_values("N", "D", []) == ("N", "D", "[]")

    def test_every_privacy_option_round_trips(self):
        for _, value in PRIVACY_OPTIONS:
            assert _submit_values("N", "D", value) == ("N", "D", value)

    def test_a_failing_str_propagates_its_own_error(self):
        with pytest.raises(RuntimeError, match="no str for you"):
            _submit_values("N", "D", _BadStr())

    def test_the_name_gate_runs_before_the_privacy_str(self):
        """A bad privacy is never consulted when the name is empty."""
        assert _submit_values("   ", "D", _BadStr()) is None


# ── Native vs fallback parity ─────────────────────────────────────────

# (name, description, privacy) triples; the privacy rides as a JSON value
# so the str() conversion parity covers the None/int/bool/list shapes.
_SUBMIT_CASES = [
    ["  My Playlist  ", "  Desc  ", "PUBLIC"],
    ["   ", "Desc", "PUBLIC"],
    ["", "Desc", "PRIVATE"],
    ["Name", "   ", "UNLISTED"],
    ["\t Playlist \n", "\n Description \t", "PUBLIC"],
    ["\x1c Name \x85\xa0", "\x1f Desc \x1d", "UNLISTED"],
    ["N", "D", None],
    ["N", "D", 5],
    ["N", "D", True],
    ["N", "D", []],
    ["0", "", "PUBLIC"],
]

_CHROME_CASES = [True, False]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.create_playlist_popup import (
    _chrome_labels,
    _submit_values,
)


def submit_outcome(case):
    name, description, privacy = case
    try:
        value = _submit_values(name, description, privacy)
        return {"ok": list(value) if value is not None else None}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
out = {
    "chrome": [_chrome_labels(edit_mode) for edit_mode in payload["chrome"]],
    "submit": [submit_outcome(case) for case in payload["submit"]],
}
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_child_payload():
    return {
        "chrome": [list(_chrome_labels(edit_mode)) for edit_mode in _CHROME_CASES],
        "submit": [_submit_outcome(case) for case in _SUBMIT_CASES],
    }


def _submit_outcome(case):
    name, description, privacy = case
    try:
        value = _submit_values(name, description, privacy)
        return {"ok": list(value) if value is not None else None}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestNativeParity:
    """When the Rust extension is built, the native chrome and submit
    engines must answer exactly like the pure-Python references for
    every case — the strips, the empty-name gate, the str() conversions
    — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self):
        payload = json.dumps({"chrome": _CHROME_CASES, "submit": _SUBMIT_CASES}, ensure_ascii=False)
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_child_payload()
        assert native == fallback
