"""Yes/No confirmation popup."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ytm_player.ui.popups.base import BasePopup


def _button_result(button_id: str | None) -> bool:
    """The dismissal verdict for a button press — the ``button`` engine.

    The pure core of ``ConfirmPopup.on_button_pressed``: ``True`` only
    for the confirm button's id, ``False`` for the cancel button's, any
    other id, or a widget with no id at all.

    Implemented by the Rust extension
    (``_ytm_player_native.confirm_popup_button_result``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return button_id == "confirm-yes"


class ConfirmPopup(BasePopup[bool]):
    """Modal confirmation dialog.

    Returns ``True`` if the user confirms, ``False`` if cancelled.
    """

    _CANCEL_RESULT: ClassVar[bool] = False

    BINDINGS = [
        Binding("n", "cancel", "No", show=False),
        Binding("y", "confirm", "Yes", show=False),
    ]

    DEFAULT_CSS = """
    ConfirmPopup {
        height: 100%;
    }

    ConfirmPopup > Vertical {
        width: 50;
        height: auto;
    }

    ConfirmPopup #confirm-message {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        color: $text;
    }

    ConfirmPopup Horizontal {
        align: center middle;
        height: auto;
        margin-top: 1;
    }

    ConfirmPopup Button {
        width: 1fr;
        margin: 0 1;
    }

    """

    def __init__(
        self,
        message: str,
        confirm_label: str = "Yes",
        cancel_label: str = "No",
    ) -> None:
        super().__init__()
        self._message = message
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self._message, id="confirm-message")
            with Horizontal():
                yield Button(self._cancel_label, variant="default", id="confirm-no")
                yield Button(self._confirm_label, variant="primary", id="confirm-yes")

    def on_mount(self) -> None:
        self.query_one("#confirm-no", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(_button_result(event.button.id))

    def action_confirm(self) -> None:
        self.dismiss(True)


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same verdict; this binding overrides
# the pure-Python transcription above when the extension is importable.
# The rebinding must happen AFTER the definition above — assigning the
# native function any earlier would be clobbered by the ``def``
# statement (the lesson that produced formatting.py's dead native
# bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        confirm_popup_button_result as _native_confirm_popup_button_result,
    )
except ImportError:
    pass
else:

    def _button_result(button_id: str | None) -> bool:  # noqa: F811
        """The button-press verdict (native Rust engine)."""
        return _native_confirm_popup_button_result(button_id)
