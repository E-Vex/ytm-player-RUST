"""Tests for the track filter's ownership and escape-verdict engines.

``_owns_input`` and ``_escape_clears`` (ui/track_filter.py) run with
two interchangeable backends: the pure-Python transcriptions in the
module and, when the Rust extension is built, the native
implementations bound at the module's tail. The engines: the
input-ownership gate (only the shared ``#track-filter`` Input's events
are the filter's) behind ``on_input_changed`` / ``on_input_submitted``,
and the escape-clear verdict (Escape clears the filter exactly when
the filter input is still visible) behind ``on_key``.

The unit tests below assert each engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest
from textual.events import Key
from textual.widget import Widget

from ytm_player.ui.track_filter import TrackFilterHost, _escape_clears, _owns_input

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestOwnsInput:
    """The ownership gate (whichever backend)."""

    def test_the_shared_filter_input_is_ours(self):
        assert _owns_input("track-filter") is True

    def test_other_ids_are_not_ours(self):
        assert _owns_input("trackfilter") is False
        assert _owns_input("") is False
        assert _owns_input("search-input") is False

    def test_a_widget_without_an_id_is_not_ours(self):
        """``Input.id`` is ``str | None``; a None id is never ours."""
        assert _owns_input(None) is False

    def test_the_match_is_exact_not_case_insensitive(self):
        assert _owns_input("TRACK-FILTER") is False
        assert _owns_input("track-filter ") is False
        assert _owns_input(" track-filter") is False


class TestEscapeClears:
    """The escape verdict (whichever backend)."""

    def test_escape_over_a_visible_filter_clears(self):
        assert _escape_clears("escape", True) is True

    def test_escape_over_a_hidden_filter_does_not(self):
        """Enter hides the input while keeping the filter — a later
        Escape must not clear what the user chose to keep visible in
        the table.
        """
        assert _escape_clears("escape", False) is False

    def test_any_other_key_does_not(self):
        assert _escape_clears("enter", True) is False
        assert _escape_clears("esc", True) is False
        assert _escape_clears("", True) is False

    def test_the_key_match_is_exact(self):
        assert _escape_clears("escape ", True) is False
        assert _escape_clears("ESCAPE", True) is False


class _Host(TrackFilterHost, Widget):
    """A bare host page listing the mixin before the Textual base."""

    _filter_table_id = "#table"


def _make_host() -> tuple[_Host, MagicMock, MagicMock]:
    """A host whose ``query_one`` returns fake widgets."""
    host = _Host.__new__(_Host)
    input_widget = MagicMock(name="track-filter-input")
    table = MagicMock(name="track-table")

    def query_one(selector, *args, **kwargs):
        if selector == "#track-filter":
            return input_widget
        if selector == "#table":
            return table
        raise AssertionError(f"unexpected selector {selector!r}")

    object.__setattr__(host, "query_one", query_one)
    return host, input_widget, table


class TestWiring:
    """The host's handlers route their verdicts into the DOM."""

    def test_escape_clears_filter_and_refocuses_table(self):
        host, input_widget, table = _make_host()
        input_widget.has_class.return_value = True
        event = MagicMock(spec=Key)
        event.key = "escape"

        host.on_key(event)

        event.stop.assert_called_once()
        event.prevent_default.assert_called_once()
        table.clear_filter.assert_called_once()
        input_widget.remove_class.assert_called_once_with("visible")
        table.focus.assert_called_once()

    def test_escape_ignored_when_filter_hidden(self):
        host, input_widget, table = _make_host()
        input_widget.has_class.return_value = False
        event = MagicMock(spec=Key)
        event.key = "escape"

        host.on_key(event)

        event.stop.assert_not_called()
        table.clear_filter.assert_not_called()

    def test_non_escape_keys_leave_the_page_alone(self):
        host, input_widget, table = _make_host()
        input_widget.has_class.return_value = True
        event = MagicMock(spec=Key)
        event.key = "j"

        host.on_key(event)

        event.stop.assert_not_called()
        table.clear_filter.assert_not_called()
        input_widget.remove_class.assert_not_called()

    def test_non_key_events_are_ignored(self):
        host, input_widget, table = _make_host()
        host.on_key(MagicMock())

        table.clear_filter.assert_not_called()

    def test_input_changed_applies_the_filter(self):
        host, _input, table = _make_host()
        event = MagicMock()
        event.input.id = "track-filter"
        event.value = "query"

        host.on_input_changed(event)

        table.apply_filter.assert_called_once_with("query")

    def test_other_inputs_changes_are_ignored(self):
        host, _input, table = _make_host()
        event = MagicMock()
        event.input.id = "some-other-input"
        event.value = "query"

        host.on_input_changed(event)

        table.apply_filter.assert_not_called()

    def test_input_submitted_hides_input_keeps_filter(self):
        host, input_widget, table = _make_host()
        event = MagicMock()
        event.input.id = "track-filter"

        host.on_input_submitted(event)

        input_widget.remove_class.assert_called_once_with("visible")
        table.focus.assert_called_once()
        table.clear_filter.assert_not_called()

    def test_other_inputs_submissions_are_ignored(self):
        host, input_widget, table = _make_host()
        event = MagicMock()
        event.input.id = "some-other-input"

        host.on_input_submitted(event)

        input_widget.remove_class.assert_not_called()
        table.focus.assert_not_called()


# ── Native vs fallback parity ─────────────────────────────────────────

_INPUT_CASES = [
    None,
    "track-filter",
    "trackfilter",
    "",
    "track-filter ",
    "TRACK-FILTER",
    " search-input",
]

_ESCAPE_CASES = [
    ["escape", True],
    ["escape", False],
    ["enter", True],
    ["esc", True],
    ["", True],
    ["escape ", True],
    ["ESCAPE", False],
    ["j", True],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.track_filter import _escape_clears, _owns_input

payload = json.load(sys.stdin)
out = {
    "owns": [_owns_input(case) for case in payload["input_cases"]],
    "escapes": [_escape_clears(key, visible) for key, visible in payload["escape_cases"]],
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
        payload = json.dumps({"input_cases": _INPUT_CASES, "escape_cases": _ESCAPE_CASES})
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_owns = [_owns_input(case) for case in _INPUT_CASES]
        native_escapes = [_escape_clears(key, visible) for key, visible in _ESCAPE_CASES]
        assert native_owns == fallback["owns"]
        assert native_escapes == fallback["escapes"]
