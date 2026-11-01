"""Persistent playlist sidebar — visible across all views.

The panel view engine — the label/count view model behind every row and
footer of the sidebar's ``LibraryPanel`` — is implemented by the Rust
extension (``_ytm_player_native``) when it is built and by the pure-Python
transcriptions in this module otherwise; the bottom-of-module seam rebinds
the engine to the native functions when the extension is importable. The
engine covers the whole pure core: the row label (``format_panel_item``),
the footer count label (``panel_count_label``), the ``/``-filter pass
(``filter_panel_items``), the playlist-id matching behind the optimistic
sidebar mutations (``matches_removal`` / ``matches_count_update`` /
``matches_auto_select`` / ``is_liked_music_playlist``), the count
arithmetic (``bump_playlist_count``) and the info-bar title read
(``selection_title``). The widgets below only do I/O and rendering.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, ListItem, ListView, Rule, Static

from ytm_player.config.settings import get_settings
from ytm_player.ui.selection_info_bar import SelectionChanged
from ytm_player.utils.formatting import copy_to_clipboard, truncate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Panel view engine
# ---------------------------------------------------------------------------


def format_panel_item(item: dict[str, Any]) -> Any:
    """The row label for one panel *item*.

    The title is ``title`` falling back to ``name`` (``"Unknown"`` when
    both keys are missing — a present-but-``None`` title reads as ``None``),
    and a non-``None`` ``count`` appends ``" (<count> tracks)"``. Without a
    count the title is returned **raw** — whatever object it is — exactly
    like the reference's bare ``return title``.

    Implemented by the Rust extension
    (``_ytm_player_native.format_panel_item``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    title = item.get("title", item.get("name", "Unknown"))
    count = item.get("count")
    if count is not None:
        return f"{title} ({count} tracks)"
    return title


def panel_count_label(shown: int, total: int) -> str:
    """The footer count label: ``"N item(s)"`` when every item is shown,
    ``"shown/total"`` once the filter hides some.

    Implemented by the Rust extension
    (``_ytm_player_native.panel_count_label``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if shown == total:
        return f"{total} item{'s' if total != 1 else ''}"
    return f"{shown}/{total}"


def filter_panel_items(items: list[dict[str, Any]], raw_query: str) -> list[dict[str, Any]]:
    """The ``/``-filter pass: *raw_query* stripped and lowercased, then
    kept as a fresh list of the *items* (same objects) whose lowercased
    row label contains it. An empty query resets the view to a fresh copy
    of every item.

    Implemented by the Rust extension
    (``_ytm_player_native.filter_panel_items``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    query = raw_query.strip().lower()
    if not query:
        return list(items)
    return [item for item in items if query in format_panel_item(item).lower()]


def matches_removal(item: dict[str, Any], playlist_id: str) -> bool:
    """``LibraryPanel.remove_item``'s predicate: the item's playlist id
    (``playlistId`` or ``browseId``, ``""`` when both read falsy/missing)
    equals *playlist_id* or its ``VL``-prefixed form.

    Implemented by the Rust extension
    (``_ytm_player_native.matches_removal``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    pid = item.get("playlistId") or item.get("browseId", "")
    return pid == playlist_id or pid == f"VL{playlist_id}"


def matches_count_update(item: dict[str, Any], playlist_id: str) -> bool:
    """``LibraryPanel.update_item_count``'s predicate: the item's playlist
    id (``playlistId`` or ``browseId`` or ``""``) equals *playlist_id*,
    its ``VL``-prefixed form, its ``VL``-stripped form, or prefixes the
    target — tolerant in **both** ``VL`` directions, unlike the removal
    predicate.

    Implemented by the Rust extension
    (``_ytm_player_native.matches_count_update``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    pid = item.get("playlistId") or item.get("browseId", "") or ""
    return (
        pid == playlist_id
        or pid == f"VL{playlist_id}"
        or f"VL{pid}" == playlist_id
        or pid == playlist_id.removeprefix("VL")
    )


def bump_playlist_count(current: Any, delta: int) -> Any:
    """The optimistic count arithmetic: ``None`` (unknown) stays ``None``
    so the caller leaves the count for the next library reload; otherwise
    ``max(0, int(current) + delta)`` — never negative.

    Implemented by the Rust extension
    (``_ytm_player_native.bump_playlist_count``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if current is None:
        return None
    return max(0, int(current) + delta)


def selection_title(item: dict[str, Any]) -> Any:
    """The info-bar read behind ``SelectionChanged``: ``title`` falling
    back to ``name``, ``""`` when both keys are missing — *not* the label
    composer's ``"Unknown"`` — and returned raw.

    Implemented by the Rust extension
    (``_ytm_player_native.selection_title``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return item.get("title", item.get("name", ""))


def is_liked_music_playlist(item: dict[str, Any]) -> bool:
    """Whether *item* is the account's auto-generated Liked Music
    playlist (id ``LM``) — the row ``ensure_loaded`` filters out because
    it is already a pinned nav item.

    Implemented by the Rust extension
    (``_ytm_player_native.is_liked_music_playlist``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (item.get("playlistId") or item.get("browseId", "")) == "LM"


def matches_auto_select(item: dict[str, Any], playlist_id: str) -> bool:
    """``PlaylistSidebar.auto_select_playlist``'s predicate: the item's
    playlist id (``playlistId`` or ``browseId``, no ``""`` default — a
    missing ``browseId`` reads as ``None``) equals *playlist_id* exactly,
    with no ``VL`` tolerance.

    Implemented by the Rust extension
    (``_ytm_player_native.matches_auto_select``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    pid = item.get("playlistId") or item.get("browseId")
    return pid == playlist_id


# ---------------------------------------------------------------------------
# LibraryPanel (moved from library.py)
# ---------------------------------------------------------------------------


class LibraryPanel(Widget):
    """A panel showing a list of library items with filtering and selection.

    When ``instant_select`` is True, single-click and Enter both fire
    ``ItemSelected`` immediately (no double-click gate).
    """

    DEFAULT_CSS = """
    LibraryPanel {
        height: 1fr;
        border: solid $secondary;
        padding: 0 1;
    }

    LibraryPanel .panel-title {
        text-style: bold;
        color: $text;
        height: 1;
        padding: 0 0 0 0;
    }

    LibraryPanel .panel-count {
        color: $text-muted;
        dock: bottom;
        height: 1;
    }

    LibraryPanel ListView {
        height: 1fr;
        width: 1fr;
    }

    LibraryPanel ListView ListItem:hover {
        background: $accent 30%;
    }

    LibraryPanel.truncate-items ListView ListItem {
        height: 1;
    }

    LibraryPanel.truncate-items ListView ListItem Static {
        height: 1;
    }

    LibraryPanel .panel-loading {
        height: 1fr;
        width: 1fr;
        content-align: center middle;
        color: $text-muted;
    }

    LibraryPanel .panel-filter {
        dock: bottom;
        height: 1;
        display: none;
    }

    LibraryPanel .panel-filter.visible {
        display: block;
    }
    """

    is_loading: reactive[bool] = reactive(False)
    filter_visible: reactive[bool] = reactive(False)

    class ItemSelected(Message):
        """Emitted when an item in this panel is activated."""

        def __init__(self, item_data: dict[str, Any], panel_id: str) -> None:
            super().__init__()
            self.item_data = item_data
            self.panel_id = panel_id

    class ItemDoubleClicked(Message):
        """Emitted when an item is double-clicked in instant_select mode."""

        def __init__(self, item_data: dict[str, Any], panel_id: str) -> None:
            super().__init__()
            self.item_data = item_data
            self.panel_id = panel_id

    class ItemRightClicked(Message):
        """Emitted when an item (or empty space) is right-clicked."""

        def __init__(self, item_data: dict[str, Any] | None, panel_id: str) -> None:
            super().__init__()
            self.item_data = item_data
            self.panel_id = panel_id

    def __init__(
        self,
        title: str,
        *,
        instant_select: bool = False,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._title = title
        self._instant_select = instant_select
        self._items: list[dict[str, Any]] = []
        self._filtered_items: list[dict[str, Any]] = []
        self._last_click_time: float = 0.0
        self._last_click_index: int | None = None
        self._click_activated: bool = False
        self._suppress_next_selected: bool = False

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="panel-title")
        yield Static("Loading...", classes="panel-loading")
        yield ListView(id=f"{self.id}-list")
        yield Static("", classes="panel-count")
        yield Input(
            placeholder="Filter...",
            id=f"{self.id}-filter",
            classes="panel-filter",
        )

    def on_mount(self) -> None:
        self._set_loading_visible(True)
        # Apply [ui] sidebar_overflow setting via CSS class.
        try:
            from ytm_player.config.settings import get_settings

            if get_settings().ui.sidebar_overflow == "truncate":
                self.add_class("truncate-items")
        except Exception:
            logger.debug("Failed to apply sidebar_overflow class", exc_info=True)

    def _set_loading_visible(self, visible: bool) -> None:
        try:
            loading = self.query_one(".panel-loading", Static)
            list_view = self.query_one(ListView)
            loading.display = visible
            list_view.display = not visible
        except Exception:
            logger.debug("Failed to toggle loading visibility in library panel", exc_info=True)

    def load_items(self, items: list[dict[str, Any]]) -> None:
        """Replace panel contents with *items* and hide loading indicator."""
        self._items = list(items)
        # The unfiltered view is the filter pass's empty-query reset — one
        # definition of "every item, fresh list" on both backends.
        self._filtered_items = filter_panel_items(self._items, "")
        self._rebuild_list(self._filtered_items)
        self._set_loading_visible(False)
        self.is_loading = False

    def prepend_item(self, item: dict[str, Any]) -> None:
        """Optimistically insert *item* at the top of the panel."""
        self._items.insert(0, item)
        self._filtered_items.insert(0, item)
        full_text = format_panel_item(item)
        lbl = Static(self._render_text(full_text))
        list_view = self.query_one(ListView)
        list_view.insert(0, [ListItem(lbl)])
        count_label = self.query_one(".panel-count", Static)
        count_label.update(panel_count_label(len(self._filtered_items), len(self._items)))

    def remove_item(self, playlist_id: str) -> None:
        """Optimistically remove the item with *playlist_id* from the panel."""
        self._items = [i for i in self._items if not matches_removal(i, playlist_id)]
        self._filtered_items = [
            i for i in self._filtered_items if not matches_removal(i, playlist_id)
        ]
        self._rebuild_list(self._filtered_items)

    def update_item_count(self, playlist_id: str, delta: int) -> None:
        """Optimistically update the cached track count for a playlist.

        Matches against both 'playlistId' and 'browseId' keys. Tolerates a
        VL-prefix mismatch in either direction. If the cached count is None
        (unknown), leaves it None — never fabricates.
        """
        target_item = next((i for i in self._items if matches_count_update(i, playlist_id)), None)
        if target_item is None:
            return

        new_count = bump_playlist_count(target_item.get("count"), delta)
        if new_count is None:
            return  # unknown count — wait for next library reload
        target_item["count"] = new_count

        # Rebuild the list so the visible label reflects the new count.
        # Wrapped in try/except because tests instantiate via __new__ without
        # mounting the ListView; production code always has it mounted.
        try:
            self._rebuild_list(self._filtered_items)
        except Exception:
            logger.exception("update_item_count: _rebuild_list failed")

    def _rebuild_list(self, items: list[dict[str, Any]]) -> None:
        list_view = self.query_one(ListView)
        list_view.clear()
        for item in items:
            full_text = format_panel_item(item)
            lbl = Static(self._render_text(full_text))
            list_view.append(ListItem(lbl))
        count_label = self.query_one(".panel-count", Static)
        count_label.update(panel_count_label(len(items), len(self._items)))

    def _render_text(self, full_text: str) -> str:
        """Apply the configured overflow rule to *full_text*.

        - "truncate" (default): hard-cut to 60 chars with `…` suffix on overflow.
          The `truncate-items` CSS class additionally clips the row to height 1
          so even truncated-but-still-too-wide text won't wrap visually.
        - "wrap": pass the text through unchanged; Textual's Static widget will
          wrap to multiple lines naturally.
        """
        try:
            from ytm_player.config.settings import get_settings

            mode = get_settings().ui.sidebar_overflow
        except Exception:
            mode = "truncate"
        if mode == "wrap":
            return full_text
        return truncate(full_text, 60)

    # -- Filtering --

    def show_filter(self) -> None:
        try:
            filter_input = self.query_one(f"#{self.id}-filter", Input)
            filter_input.add_class("visible")
            filter_input.value = ""
            filter_input.focus()
            self.filter_visible = True
        except Exception:
            logger.debug("Failed to show filter input in library panel", exc_info=True)

    def hide_filter(self) -> None:
        try:
            filter_input = self.query_one(f"#{self.id}-filter", Input)
            filter_input.remove_class("visible")
            self.filter_visible = False
            self._filtered_items = filter_panel_items(self._items, "")
            self._rebuild_list(self._filtered_items)
        except Exception:
            logger.debug("Failed to hide filter input in library panel", exc_info=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        if not event.input.id or not event.input.id.endswith("-filter"):
            return
        self._filtered_items = filter_panel_items(self._items, event.value)
        self._rebuild_list(self._filtered_items)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id and event.input.id.endswith("-filter"):
            filter_input = self.query_one(f"#{self.id}-filter", Input)
            filter_input.remove_class("visible")
            self.filter_visible = False
            list_view = self.query_one(ListView)
            list_view.focus()

    # -- Highlight --

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        """Post SelectionChanged so SelectionInfoBar can display the full title.

        Only posts when this panel's ListView actually has focus — otherwise
        a freshly-mounted ListView fires Highlighted at index 0 on init and
        stomps a different focused widget's selection in the info bar.
        """
        try:
            list_view = event.list_view
            if not list_view.has_focus:
                return
            sel_idx = list_view.index
            if sel_idx is not None and 0 <= sel_idx < len(self._filtered_items):
                self.post_message(SelectionChanged(selection_title(self._filtered_items[sel_idx])))
            else:
                self.post_message(SelectionChanged(""))
        except Exception:
            self.post_message(SelectionChanged(""))

    def on_focus(self) -> None:
        """When the panel gains focus, push the current highlight into the info bar."""
        try:
            list_view = self.query_one(ListView)
            sel_idx = list_view.index
            if sel_idx is not None and 0 <= sel_idx < len(self._filtered_items):
                self.post_message(SelectionChanged(selection_title(self._filtered_items[sel_idx])))
            else:
                self.post_message(SelectionChanged(""))
        except Exception:
            self.post_message(SelectionChanged(""))

    # -- Selection --

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._instant_select:
            if self._suppress_next_selected:
                self._suppress_next_selected = False
                return
            idx = event.list_view.index
            if idx is not None and 0 <= idx < len(self._filtered_items):
                self.post_message(self.ItemSelected(self._filtered_items[idx], self.id or ""))
            return
        if self._click_activated:
            self._click_activated = False
            return
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(self._filtered_items):
            self.post_message(self.ItemSelected(self._filtered_items[idx], self.id or ""))

    def _find_clicked_item_index(self, event: Click) -> int | None:
        node = event.widget
        while node is not None and not isinstance(node, ListItem):
            if node is self:
                break
            node = node.parent
        if not isinstance(node, ListItem):
            return None
        list_view = self.query_one(ListView)
        try:
            return list(list_view.children).index(node)
        except ValueError:
            return None

    def on_click(self, event: Click) -> None:
        if event.button == 3:
            event.stop()
            idx = self._find_clicked_item_index(event)
            if idx is not None and 0 <= idx < len(self._filtered_items):
                self.post_message(self.ItemRightClicked(self._filtered_items[idx], self.id or ""))
            else:
                self.post_message(self.ItemRightClicked(None, self.id or ""))
            return
        if event.button != 1:
            return
        if self._instant_select:
            idx = self._find_clicked_item_index(event)
            if idx is not None and 0 <= idx < len(self._filtered_items):
                now = time.monotonic()
                if self._last_click_index == idx and (now - self._last_click_time) < 0.4:
                    self._last_click_time = 0.0
                    self._last_click_index = None
                    event.stop()
                    self._suppress_next_selected = True
                    self.post_message(
                        self.ItemDoubleClicked(self._filtered_items[idx], self.id or "")
                    )
                    return
                self._suppress_next_selected = False
                self._last_click_time = now
                self._last_click_index = idx
            return
        idx = self._find_clicked_item_index(event)
        if idx is None:
            return
        now = time.monotonic()
        if self._last_click_index == idx and (now - self._last_click_time) < 0.4:
            self._last_click_time = 0.0
            self._last_click_index = None
            self._click_activated = True
            if 0 <= idx < len(self._filtered_items):
                self.post_message(self.ItemSelected(self._filtered_items[idx], self.id or ""))
            return
        self._last_click_time = now
        self._last_click_index = idx
        self._click_activated = True


# ---------------------------------------------------------------------------
# PlaylistSidebar — wraps pinned-nav + LibraryPanel
# ---------------------------------------------------------------------------


class PlaylistSidebar(Widget):
    """Persistent playlist sidebar visible across all views."""

    DEFAULT_CSS = """
    PlaylistSidebar {
        width: 30;
        height: 1fr;
        border-right: solid $border;
    }

    PlaylistSidebar.hidden {
        display: none;
    }

    #ps-pinned-nav {
        height: auto;
        padding: 0 1;
    }

    .ps-pinned-item {
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }

    .ps-pinned-item:hover {
        background: $accent 30%;
        color: $text;
    }

    .ps-pinned-item.active {
        color: $primary;
        text-style: bold;
    }

    #ps-separator,
    #ps-separator-bottom {
        margin: 0 1;
        color: $border;
    }

    #ps-playlists {
        width: 1fr;
    }
    """

    class PlaylistSelected(Message):
        """A playlist was selected in the sidebar."""

        def __init__(self, item_data: dict[str, Any]) -> None:
            super().__init__()
            self.item_data = item_data

    class PlaylistDoubleClicked(Message):
        """A playlist was double-clicked in the sidebar."""

        def __init__(self, item_data: dict[str, Any]) -> None:
            super().__init__()
            self.item_data = item_data

    class PlaylistRightClicked(Message):
        """A playlist was right-clicked in the sidebar."""

        def __init__(self, item_data: dict[str, Any] | None) -> None:
            super().__init__()
            self.item_data = item_data

    class NavItemClicked(Message):
        """A pinned nav item (Liked Songs / Recently Played) was clicked."""

        def __init__(self, nav_id: str) -> None:
            super().__init__()
            self.nav_id = nav_id

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._loaded: bool = False

    def compose(self) -> ComposeResult:
        with Vertical(id="ps-pinned-nav"):
            yield Static("\u2665 Liked Songs", id="ps-nav-liked", classes="ps-pinned-item")
            yield Static("\u23f1 Recently Played", id="ps-nav-recent", classes="ps-pinned-item")
            yield Static("\u266b Discovery Mix", id="ps-nav-discovery", classes="ps-pinned-item")
        yield Rule(id="ps-separator")
        yield LibraryPanel("Playlists", id="ps-playlists", instant_select=True)
        yield Rule(id="ps-separator-bottom")

    def on_mount(self) -> None:
        settings = get_settings()
        self.styles.width = settings.ui.sidebar_width

    async def ensure_loaded(self) -> None:
        """Load playlists if not already loaded. Gated by auth readiness."""
        if self._loaded:
            return
        ytmusic = getattr(self.app, "ytmusic", None)
        if ytmusic is None:
            return
        self._loaded = True
        try:
            playlists = await ytmusic.get_library_playlists(limit=None)
            panel = self.query_one("#ps-playlists", LibraryPanel)
            if isinstance(playlists, list):
                # Filter out "Liked Music" — it's already a pinned nav item.
                playlists = [p for p in playlists if not is_liked_music_playlist(p)]
                panel.load_items(playlists)
            else:
                logger.error("Failed to load playlists: %s", playlists)
                panel.load_items([])
        except Exception:
            logger.exception("Failed to load library playlists in sidebar")

    async def refresh_playlists(self) -> None:
        """Force-reload playlists."""
        self._loaded = False
        await self.ensure_loaded()

    def auto_select_playlist(self, playlist_id: str) -> None:
        """Highlight a specific playlist in the panel."""
        panel = self.query_one("#ps-playlists", LibraryPanel)
        for item in panel._items:
            if matches_auto_select(item, playlist_id):
                self.post_message(self.PlaylistSelected(item))
                break

    # -- Bubble LibraryPanel messages as PlaylistSidebar messages --

    def on_library_panel_item_selected(self, event: LibraryPanel.ItemSelected) -> None:
        event.stop()
        self.post_message(self.PlaylistSelected(event.item_data))

    def on_library_panel_item_double_clicked(self, event: LibraryPanel.ItemDoubleClicked) -> None:
        event.stop()
        self.post_message(self.PlaylistDoubleClicked(event.item_data))

    def on_library_panel_item_right_clicked(self, event: LibraryPanel.ItemRightClicked) -> None:
        event.stop()
        self.post_message(self.PlaylistRightClicked(event.item_data))

    # -- Pinned nav clicks --

    def on_click(self, event: Click) -> None:
        target = event.widget
        if target is None:
            return
        if target.id == "ps-nav-liked":
            event.stop()
            self.post_message(self.NavItemClicked("liked_songs"))
        elif target.id == "ps-nav-recent":
            event.stop()
            self.post_message(self.NavItemClicked("recently_played"))
        elif target.id == "ps-nav-discovery":
            event.stop()
            self.post_message(self.NavItemClicked("discovery_mix"))

    # -- Public helpers for sidebar actions --

    def handle_sidebar_action(self, action: str, count: int = 1) -> None:
        """Dispatch vim-style actions to the sidebar ListView."""
        from ytm_player.config.keymap import Action

        try:
            list_view = self.query_one("#ps-playlists-list", ListView)
        except Exception:
            return

        match Action(action) if isinstance(action, str) else action:
            case Action.MOVE_DOWN:
                for _ in range(count):
                    list_view.action_cursor_down()
            case Action.MOVE_UP:
                for _ in range(count):
                    list_view.action_cursor_up()
            case Action.PAGE_DOWN:
                list_view.action_scroll_down()
            case Action.PAGE_UP:
                list_view.action_scroll_up()
            case Action.GO_TOP:
                if len(list_view.children) > 0:
                    list_view.index = 0
            case Action.GO_BOTTOM:
                if len(list_view.children) > 0:
                    list_view.index = len(list_view.children) - 1
            case Action.SELECT:
                list_view.action_select_cursor()
            case Action.FILTER:
                panel = self.query_one("#ps-playlists", LibraryPanel)
                panel.show_filter()

    def get_highlighted_item(self) -> dict[str, Any] | None:
        """Return the currently highlighted playlist item."""
        try:
            panel = self.query_one("#ps-playlists", LibraryPanel)
            list_view = panel.query_one(ListView)
            idx = list_view.index
            if idx is not None and 0 <= idx < len(panel._filtered_items):
                return panel._filtered_items[idx]
        except Exception:
            logger.debug("Failed to get highlighted item from sidebar", exc_info=True)
        return None

    def copy_item_link(self, item: dict[str, Any]) -> None:
        """Copy a YouTube Music playlist link to clipboard."""
        pid = item.get("playlistId") or item.get("browseId", "")
        if not pid:
            self.app.notify("No link available", severity="warning", timeout=2)
            return
        link = f"https://music.youtube.com/playlist?list={pid}"
        if copy_to_clipboard(link):
            self.app.notify("Link copied", timeout=2)
        else:
            self.app.notify(link, timeout=5)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same panel view engine; these bindings
# replace the pure-Python transcriptions above when the extension is
# importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead native
# bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        bump_playlist_count as _native_bump_playlist_count,
    )
    from ytm_player.utils._ytm_player_native import (
        filter_panel_items as _native_filter_panel_items,
    )
    from ytm_player.utils._ytm_player_native import (
        format_panel_item as _native_format_panel_item,
    )
    from ytm_player.utils._ytm_player_native import (
        is_liked_music_playlist as _native_is_liked_music_playlist,
    )
    from ytm_player.utils._ytm_player_native import (
        matches_auto_select as _native_matches_auto_select,
    )
    from ytm_player.utils._ytm_player_native import (
        matches_count_update as _native_matches_count_update,
    )
    from ytm_player.utils._ytm_player_native import (
        matches_removal as _native_matches_removal,
    )
    from ytm_player.utils._ytm_player_native import (
        panel_count_label as _native_panel_count_label,
    )
    from ytm_player.utils._ytm_player_native import (
        selection_title as _native_selection_title,
    )
except ImportError:
    pass
else:
    format_panel_item = _native_format_panel_item  # type: ignore[assignment]
    panel_count_label = _native_panel_count_label  # type: ignore[assignment]
    filter_panel_items = _native_filter_panel_items  # type: ignore[assignment]
    matches_removal = _native_matches_removal  # type: ignore[assignment]
    matches_count_update = _native_matches_count_update  # type: ignore[assignment]
    bump_playlist_count = _native_bump_playlist_count  # type: ignore[assignment]
    selection_title = _native_selection_title  # type: ignore[assignment]
    is_liked_music_playlist = _native_is_liked_music_playlist  # type: ignore[assignment]
    matches_auto_select = _native_matches_auto_select  # type: ignore[assignment]
