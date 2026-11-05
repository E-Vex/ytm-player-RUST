"""Modal popup for picking a YouTube Music chart region.

Pattern follows playlist_picker.py: ModalScreen[str | None] subclass
with a filter Input above a ListView, BINDINGS for escape, theme
variables only ($surface, $primary, $text, $text-muted).

Returns the selected ISO 3166-1 alpha-2 code on success, or None if
the user pressed Esc.
"""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Label, ListItem, ListView, Static

from ytm_player.services.regions import CHART_REGIONS
from ytm_player.ui.popups.base import BasePopup

logger = logging.getLogger(__name__)


def filter_regions(regions: tuple[tuple[str, str], ...], query: str) -> tuple[tuple[str, str], ...]:
    """Return regions matching *query* on either ISO code or display name.

    Empty query returns all. Case-insensitive substring match on either field.

    Implemented by the Rust extension
    (``_ytm_player_native.filter_regions``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    q = query.strip().lower()
    if not q:
        return regions
    return tuple((code, name) for code, name in regions if q in code.lower() or q in name.lower())


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same filter pass; this binding replaces
# the pure-Python transcription above when the extension is importable.  The
# rebinding must happen AFTER the definition above — assigning the native
# function any earlier would be clobbered by the ``def`` statement.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        filter_regions as _native_filter_regions,
    )
except ImportError:
    pass
else:
    filter_regions = _native_filter_regions  # type: ignore[assignment]


class _RegionItem(ListItem):
    """Single region entry in the picker list."""

    def __init__(self, code: str, name: str) -> None:
        super().__init__()
        self.code = code
        self._name = name

    def compose(self) -> ComposeResult:
        yield Label(f"{self.code} — {self._name}")


class CountryPickerModal(BasePopup[str | None]):
    """Pick a YouTube Music chart region.

    Returns the ISO code on select, or None on cancel.
    """

    DEFAULT_CSS = """
    CountryPickerModal > Vertical {
        width: 50;
        max-height: 80%;
    }

    CountryPickerModal #picker-title {
        text-align: center;
        text-style: bold;
        width: 100%;
        color: $text;
    }

    CountryPickerModal #picker-status {
        text-align: center;
        width: 100%;
        color: $text-muted;
        margin-bottom: 1;
    }

    CountryPickerModal #filter-input {
        margin-bottom: 1;
    }

    CountryPickerModal ListView {
        height: auto;
        max-height: 18;
        background: $surface;
    }

    CountryPickerModal ListItem {
        padding: 0 1;
        height: 1;
    }
    """

    def __init__(self, current_code: str = "ZZ") -> None:
        super().__init__()
        self._current_code = current_code
        self._all_regions: tuple[tuple[str, str], ...] = CHART_REGIONS

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Pick a region", id="picker-title")
            yield Static(f"Current: {self._current_code}", id="picker-status")
            yield Input(placeholder="Filter by code or name…", id="filter-input")
            yield ListView(id="region-list")

    async def on_mount(self) -> None:
        await self._populate(self._all_regions)
        list_view = self.query_one("#region-list", ListView)
        for i, item in enumerate(list_view.children):
            if isinstance(item, _RegionItem) and item.code == self._current_code:
                list_view.index = i
                break
        list_view.focus()

    async def _populate(self, regions: tuple[tuple[str, str], ...]) -> None:
        list_view = self.query_one("#region-list", ListView)
        await list_view.clear()
        for code, name in regions:
            await list_view.append(_RegionItem(code, name))

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "filter-input":
            return
        filtered = filter_regions(self._all_regions, query=event.value)
        await self._populate(filtered)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if isinstance(event.item, _RegionItem):
            self.dismiss(event.item.code)
