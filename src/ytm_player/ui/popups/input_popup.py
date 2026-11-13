"""Simple text-input popup that returns the entered string or None."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static

from ytm_player.ui.popups.base import BasePopup


def _submitted_value(value: str) -> str | None:
    """Normalize a submitted input value — the ``submit`` engine.

    The pure core of ``InputPopup.on_input_submitted``: the value
    stripped, an empty or all-whitespace submission answering ``None``.

    Implemented by the Rust extension
    (``_ytm_player_native.input_popup_submitted_value``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    value = value.strip()
    if value:
        return value
    return None


class InputPopup(BasePopup[str | None]):
    """Modal prompt with a single text input.

    Returns the entered text on submit, or ``None`` if dismissed.
    """

    DEFAULT_CSS = """
    InputPopup > Vertical {
        width: 50;
        height: auto;
    }

    InputPopup #input-title {
        text-align: center;
        text-style: bold;
        width: 100%;
        margin-bottom: 1;
        color: $text;
    }

    InputPopup Input {
        width: 100%;
    }
    """

    def __init__(self, title: str, placeholder: str = "") -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._title, id="input-title")
            yield Input(placeholder=self._placeholder, id="input-field")

    def on_mount(self) -> None:
        self.query_one("#input-field", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(_submitted_value(event.value))


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same normalization; this binding
# overrides the pure-Python transcription above when the extension is
# importable.  The rebinding must happen AFTER the definition above —
# assigning the native function any earlier would be clobbered by the
# ``def`` statement (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        input_popup_submitted_value as _native_input_popup_submitted_value,
    )
except ImportError:
    pass
else:

    def _submitted_value(value: str) -> str | None:  # noqa: F811
        """The submitted-value normalization (native Rust engine)."""
        return _native_input_popup_submitted_value(value)
