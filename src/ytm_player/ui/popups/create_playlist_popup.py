"""Popup for creating or editing a playlist with name and privacy selection."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Select, Static

from ytm_player.ui.popups.base import BasePopup

PRIVACY_OPTIONS: list[tuple[str, str]] = [
    ("Private", "PRIVATE"),
    ("Public", "PUBLIC"),
    ("Unlisted", "UNLISTED"),
]


def _chrome_labels(edit_mode: bool) -> tuple[str, str]:
    """The popup's chrome labels — the ``chrome`` engine.

    The pure core of ``CreatePlaylistPopup.compose``'s title and
    submit-button pair: "Edit Playlist" / "Edit" in edit mode,
    "New Playlist" / "Create" otherwise.

    Implemented by the Rust extension
    (``_ytm_player_native.create_playlist_chrome_labels``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if edit_mode:
        return ("Edit Playlist", "Edit")
    return ("New Playlist", "Create")


def _submit_values(name: str, description: str, privacy: Any) -> tuple[str, str, str] | None:
    """Normalize the popup's submission — the ``submit`` engine.

    The pure core of ``CreatePlaylistPopup._submit``: the name and the
    description stripped, the privacy through ``str()`` — ``None``
    when the stripped name is empty (the popup warns and stays open).

    Implemented by the Rust extension
    (``_ytm_player_native.create_playlist_submit_values``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    name = name.strip()
    if not name:
        return None
    return (name, description.strip(), str(privacy))


class CreatePlaylistPopup(BasePopup[tuple[str, str, str] | None]):
    """Modal prompt for creating or editing a playlist.

    Pass ``initial_name``, ``initial_description``, and ``initial_privacy`` to
    pre-fill the fields for edit mode.  Set ``edit_mode=True`` to swap the
    title and submit-button labels to "Edit Playlist" / "Edit".

    Returns ``(name, description, privacy)`` on submit, or ``None`` if dismissed.
    """

    DEFAULT_CSS = """
    CreatePlaylistPopup {
        height: 100%;
    }

    CreatePlaylistPopup > Vertical {
        width: 50;
        height: auto;
    }

    CreatePlaylistPopup #popup-title {
        text-align: center;
        text-style: bold;
        width: 100%;
        margin-bottom: 1;
        color: $text;
    }

    CreatePlaylistPopup Input {
        width: 100%;
        margin-bottom: 1;
    }

    CreatePlaylistPopup #input-description {
        height: 3;
    }

    CreatePlaylistPopup Select {
        width: 100%;
        margin-bottom: 1;
    }

    CreatePlaylistPopup #button-row {
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    CreatePlaylistPopup Button {
        width: 1fr;
        margin: 0 1;
    }
    """

    def __init__(
        self,
        *,
        initial_name: str = "",
        initial_description: str = "",
        initial_privacy: str = "PRIVATE",
        edit_mode: bool = False,
    ) -> None:
        super().__init__()
        self._initial_name = initial_name
        self._initial_description = initial_description
        self._initial_privacy = initial_privacy
        self._edit_mode = edit_mode

    def compose(self) -> ComposeResult:
        title, submit_label = _chrome_labels(self._edit_mode)
        with Vertical():
            yield Static(title, id="popup-title")
            yield Input(
                value=self._initial_name,
                placeholder="Playlist name...",
                id="input-name",
            )
            yield Input(
                value=self._initial_description,
                placeholder="Description (optional)...",
                id="input-description",
            )
            yield Select(
                PRIVACY_OPTIONS,
                value=self._initial_privacy,
                id="select-privacy",
                allow_blank=False,
            )
            with Horizontal(id="button-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button(submit_label, variant="primary", id="btn-create")

    def on_mount(self) -> None:
        self.query_one("#input-name", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-create":
            self._submit()
        else:
            self.dismiss(None)

    def _submit(self) -> None:
        name_input = self.query_one("#input-name", Input)
        values = _submit_values(
            name_input.value,
            self.query_one("#input-description", Input).value,
            self.query_one("#select-privacy", Select).value,
        )
        if values is None:
            self.notify("Playlist name cannot be empty", severity="warning", timeout=3)
            name_input.focus()
            return
        self.dismiss(values)


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same chrome pair and submission
# normalization; these bindings override the pure-Python transcriptions
# above when the extension is importable.  The rebinding must happen
# AFTER the definitions above — assigning the native functions any
# earlier would be clobbered by the ``def`` statements (the lesson that
# produced formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        create_playlist_chrome_labels as _native_create_playlist_chrome_labels,
    )
    from ytm_player.utils._ytm_player_native import (
        create_playlist_submit_values as _native_create_playlist_submit_values,
    )
except ImportError:
    pass
else:

    def _chrome_labels(edit_mode: bool) -> tuple[str, str]:  # noqa: F811
        """The popup's chrome labels (native Rust engine)."""
        return _native_create_playlist_chrome_labels(edit_mode)

    def _submit_values(  # noqa: F811
        name: str, description: str, privacy: Any
    ) -> tuple[str, str, str] | None:
        """The submission normalization (native Rust engine)."""
        return _native_create_playlist_submit_values(name, description, privacy)
