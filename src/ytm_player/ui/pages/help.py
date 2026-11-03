"""Searchable keybinding reference page."""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, Input, Label

from ytm_player.config.keymap import Action, get_keymap

logger = logging.getLogger(__name__)


# ── Action metadata ──────────────────────────────────────────────────

ACTION_DESCRIPTIONS: dict[Action, str] = {
    # Playback
    Action.PLAY_PAUSE: "Toggle play/pause",
    Action.NEXT_TRACK: "Skip to next track",
    Action.PREVIOUS_TRACK: "Go to previous track",
    Action.PLAY_RANDOM: "Play a random track from queue",
    Action.CYCLE_REPEAT: "Cycle repeat mode (Off/All/One)",
    Action.TOGGLE_SHUFFLE: "Toggle shuffle on/off",
    Action.VOLUME_UP: "Increase volume",
    Action.VOLUME_DOWN: "Decrease volume",
    Action.MUTE: "Toggle mute",
    Action.SEEK_FORWARD: "Seek forward",
    Action.SEEK_BACKWARD: "Seek backward",
    Action.SEEK_START: "Seek to beginning of track",
    # Navigation
    Action.MOVE_DOWN: "Move cursor down",
    Action.MOVE_UP: "Move cursor up",
    Action.PAGE_DOWN: "Scroll page down",
    Action.PAGE_UP: "Scroll page up",
    Action.GO_TOP: "Jump to top of list",
    Action.GO_BOTTOM: "Jump to bottom of list",
    Action.SELECT: "Select / confirm",
    Action.FOCUS_NEXT: "Focus next section",
    Action.FOCUS_PREV: "Focus previous section",
    Action.FOCUS_PANE_LEFT: "Focus the Playlists sidebar (vim window: left)",
    Action.FOCUS_PANE_RIGHT: "Focus the main content, then lyrics (vim window: right)",
    Action.FOCUS_PANE_CYCLE: "Cycle focus through the visible panes",
    Action.GO_BACK: "Go back / close page",
    Action.GO_FORWARD: "Go forward (after going back)",
    # Pages
    Action.LIBRARY: "Go to library",
    Action.SEARCH: "Go to search",
    Action.BROWSE: "Go to browse",
    Action.LIKED_SONGS: "Go to liked songs",
    Action.RECENTLY_PLAYED: "Go to recently played",
    Action.LYRICS: "Show lyrics",
    Action.CURRENT_CONTEXT: "Go to current context",
    Action.JUMP_TO_CURRENT: "Jump to currently playing track",
    Action.QUEUE: "Show playback queue",
    Action.HELP: "Show this help page",
    # Actions
    Action.DELETE_ITEM: "Delete / remove item",
    Action.TRACK_ACTIONS: "Show track actions menu",
    Action.LIKE_TOGGLE: "Toggle like on the currently playing track",
    Action.PLAY_NEXT: "Play the focused track next (insert after current)",
    Action.ADD_TO_QUEUE: "Add track to queue",
    Action.ADD_TO_PLAYLIST: "Add the marked tracks (or the playing track) to a playlist",
    Action.MARK_TOGGLE: "Mark / unmark the highlighted track (ends a range first)",
    Action.MARK_RANGE: "Start a range of marks that follows the cursor (again to end)",
    Action.MARK_CLEAR: "Clear all marks",
    Action.DISCOVERY_MIX: "Start discovery roulette — random mix from a rotating source",
    Action.FILTER: "Filter / search within list",
    Action.PICK_COUNTRY: "Pick chart region (Browse → Charts only)",
    Action.REORDER_DOWN: "Move selected track down in the queue (Queue only)",
    Action.REORDER_UP: "Move selected track up in the queue (Queue only)",
    # Sorting
    Action.SORT_TITLE: "Sort by title",
    Action.SORT_ARTIST: "Sort by artist",
    Action.SORT_ALBUM: "Sort by album",
    Action.SORT_DURATION: "Sort by duration",
    Action.SORT_DATE: "Sort by date",
    Action.REVERSE_SORT: "Reverse sort order",
    # Search
    Action.TOGGLE_SEARCH_MODE: "Toggle search mode",
}

ACTION_CATEGORIES: dict[str, list[Action]] = {
    "Playback": [
        Action.PLAY_PAUSE,
        Action.NEXT_TRACK,
        Action.PREVIOUS_TRACK,
        Action.PLAY_RANDOM,
        Action.CYCLE_REPEAT,
        Action.TOGGLE_SHUFFLE,
        Action.VOLUME_UP,
        Action.VOLUME_DOWN,
        Action.MUTE,
        Action.SEEK_FORWARD,
        Action.SEEK_BACKWARD,
        Action.SEEK_START,
    ],
    "Navigation": [
        Action.MOVE_DOWN,
        Action.MOVE_UP,
        Action.PAGE_DOWN,
        Action.PAGE_UP,
        Action.GO_TOP,
        Action.GO_BOTTOM,
        Action.SELECT,
        Action.FOCUS_NEXT,
        Action.FOCUS_PREV,
        Action.FOCUS_PANE_LEFT,
        Action.FOCUS_PANE_RIGHT,
        Action.FOCUS_PANE_CYCLE,
        Action.GO_BACK,
        Action.GO_FORWARD,
    ],
    "Pages": [
        Action.LIBRARY,
        Action.SEARCH,
        Action.BROWSE,
        Action.LIKED_SONGS,
        Action.RECENTLY_PLAYED,
        Action.LYRICS,
        Action.CURRENT_CONTEXT,
        Action.JUMP_TO_CURRENT,
        Action.QUEUE,
        Action.HELP,
    ],
    "Actions": [
        Action.DELETE_ITEM,
        Action.TRACK_ACTIONS,
        Action.LIKE_TOGGLE,
        Action.PLAY_NEXT,
        Action.ADD_TO_QUEUE,
        Action.ADD_TO_PLAYLIST,
        Action.MARK_TOGGLE,
        Action.MARK_RANGE,
        Action.MARK_CLEAR,
        Action.DISCOVERY_MIX,
        Action.FILTER,
        Action.PICK_COUNTRY,
        Action.REORDER_DOWN,
        Action.REORDER_UP,
    ],
    "Sorting": [
        Action.SORT_TITLE,
        Action.SORT_ARTIST,
        Action.SORT_ALBUM,
        Action.SORT_DURATION,
        Action.SORT_DATE,
        Action.REVERSE_SORT,
    ],
    "Search": [
        Action.TOGGLE_SEARCH_MODE,
    ],
}


def _title_case_action_value(action_value: str) -> str:
    """``action.value.replace("_", " ").title()`` — the name pipeline.

    The body of the reference's ``_format_action_name``, kept as its own
    helper so the fallback engine below composes from the same pipeline
    the reference used.
    """
    return action_value.replace("_", " ").title()


def _format_action_name(action: Action) -> str:
    """Convert an Action enum value to a human-readable name."""
    return _title_case_action_value(action.value)


def _rows_from_spec(
    spec: list[tuple[str, str, str, list[str]]],
) -> list[tuple[str, str, str, str]]:
    """The all-rows list — the ``_build_rows`` engine.

    The pure half of the help page's row assembly: each spec entry
    ``(category, action_value, description, key_strs)`` becomes the row
    ``(category, name, keys_str, description)`` — the name through the
    ``replace("_", " ").title()`` pipeline, the keys as the ``" / "``
    join of the formatted key strings, or ``"(unbound)"`` when the
    action has no binding. The keymap walk that produces the spec (the
    ``get_keys_for_action``/``format_key`` calls over
    ``ACTION_CATEGORIES`` and ``ACTION_DESCRIPTIONS``) stays Python-side
    next to the objects it reads.

    Implemented by the Rust extension
    (``_ytm_player_native.help_page_rows``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    rows: list[tuple[str, str, str, str]] = []
    for category, action_value, description, key_strs in spec:
        name = _title_case_action_value(action_value)
        keys_str = " / ".join(key_strs) if key_strs else "(unbound)"
        rows.append((category, name, keys_str, description))
    return rows


def _visible_rows(
    all_rows: list[tuple[str, str, str, str]],
    filter_text: str,
) -> list[tuple[bool, str, str, str, str]]:
    """The filtered walk — the ``_populate_table`` engine.

    The pure half of the help table's rebuild: the needle is
    ``filter_text.lower().strip()`` (lowercased first, then stripped), an
    empty needle filters nothing, and a non-empty one matches a row only
    when it is a substring of the lowercased
    ``"{category} {name} {keys} {desc}"`` haystack. A category header is
    emitted whenever the category changes — starting from ``""``, so the
    first category always earns one. Headers come back as
    ``(True, category, "", "", "")`` and data rows as
    ``(False, category, name, keys, description)``; the Textual markup
    and the DataTable calls stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.help_page_visible_rows``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    needle = filter_text.lower().strip()
    current_category = ""
    out: list[tuple[bool, str, str, str, str]] = []

    for category, name, keys, desc in all_rows:
        # Apply filter if active.
        if needle:
            haystack = f"{category} {name} {keys} {desc}".lower()
            if needle not in haystack:
                continue

        # Insert a category header row when the category changes.
        if category != current_category:
            current_category = category
            out.append((True, category, "", "", ""))

        out.append((False, category, name, keys, desc))

    return out


class HelpPage(Widget):
    """Searchable keybinding reference.

    Generates a grouped table from the active KeyMap, showing each action's
    name, keybinding(s), and description. Supports filtering with `/`.
    """

    DEFAULT_CSS = """
    HelpPage {
        layout: vertical;
        width: 1fr;
        height: 1fr;
    }
    .help-header {
        height: auto;
        max-height: 2;
        padding: 1 2;
        text-style: bold;
    }
    .help-filter {
        height: auto;
        max-height: 3;
        padding: 0 2;
        display: none;
    }
    .help-table {
        height: 1fr;
        width: 1fr;
    }
    .help-table > .datatable--cursor {
        background: $selected-item;
    }
    """

    filter_visible: reactive[bool] = reactive(False)
    filter_text: reactive[str] = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._all_rows: list[tuple[str, str, str, str]] = []  # (category, name, keys, desc)

    def compose(self) -> ComposeResult:
        yield Label("[b]Keybindings[/b]", markup=True, classes="help-header")
        yield Input(
            placeholder="Filter keybindings...",
            id="help-filter-input",
            classes="help-filter",
        )
        yield DataTable(
            cursor_type="row",
            zebra_stripes=True,
            id="help-table",
            classes="help-table",
        )

    def on_mount(self) -> None:
        table = self.query_one("#help-table", DataTable)
        table.add_column("Action", width=None, key="action")
        table.add_column("Key", width=16, key="key")
        table.add_column("Description", width=None, key="description")

        self._build_rows()
        self._populate_table()

        # Land keyboard focus on the table so Tab / j / k have a starting point.
        table.focus()

    def _build_rows(self) -> None:
        """Generate the full row list from the keymap, grouped by category."""
        keymap = get_keymap()
        spec: list[tuple[str, str, str, list[str]]] = []

        for category, actions in ACTION_CATEGORIES.items():
            for action in actions:
                description = ACTION_DESCRIPTIONS.get(action, "")
                key_seqs = keymap.get_keys_for_action(action)
                key_strs = [keymap.format_key(seq) for seq in key_seqs]
                spec.append((category, action.value, description, key_strs))

        self._all_rows = _rows_from_spec(spec)

    def _populate_table(self, filter_text: str = "") -> None:
        """Fill the DataTable, optionally filtering rows."""
        table = self.query_one("#help-table", DataTable)
        table.clear()

        for is_header, category, name, keys, desc in _visible_rows(self._all_rows, filter_text):
            if is_header:
                table.add_row(
                    f"[b]{category}[/b]",
                    "",
                    "",
                    key=f"header_{category}",
                )
            else:
                table.add_row(name, keys, desc)

    # ── Filter management ─────────────────────────────────────────────

    def watch_filter_visible(self, visible: bool) -> None:
        try:
            filter_input = self.query_one("#help-filter-input", Input)
            filter_input.display = visible
            if visible:
                filter_input.focus()
            else:
                filter_input.value = ""
                self.filter_text = ""
        except Exception:
            logger.debug("Failed to toggle help filter visibility", exc_info=True)

    def watch_filter_text(self, text: str) -> None:
        self._populate_table(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "help-filter-input":
            self.filter_text = event.value

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "help-filter-input":
            # Close filter and keep the result.
            self.filter_visible = False
            self.query_one("#help-table", DataTable).focus()

    # ── Action handling ───────────────────────────────────────────────

    async def handle_action(self, action: Action, count: int = 1) -> None:
        """Process vim-style navigation actions."""
        table = self.query_one("#help-table", DataTable)

        # Guard on ACTUAL input focus, not the ``filter_visible`` flag. The app
        # only routes keys here once focus has LEFT the filter input (it defers
        # to a focused Input), so keying off the flag alone would dead-key the
        # page after Tab moved focus out of the filter. When the input genuinely
        # holds focus this branch is never reached — kept defensive.
        try:
            filter_input = self.query_one("#help-filter-input", Input)
        except Exception:
            filter_input = None
        if filter_input is not None and self.app.focused is filter_input:
            return

        match action:
            case Action.FILTER:
                self.filter_visible = True
                # Always (re)focus the input — the reactive watch won't refire
                # when filter_visible is already True (e.g. after Tab left it).
                if filter_input is not None:
                    filter_input.focus()

            case Action.MOVE_DOWN:
                for _ in range(count):
                    table.action_cursor_down()

            case Action.MOVE_UP:
                for _ in range(count):
                    table.action_cursor_up()

            case Action.PAGE_DOWN:
                table.action_scroll_down()

            case Action.PAGE_UP:
                table.action_scroll_up()

            case Action.GO_TOP:
                if table.row_count > 0:
                    table.move_cursor(row=0)

            case Action.GO_BOTTOM:
                if table.row_count > 0:
                    table.move_cursor(row=table.row_count - 1)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same row engines; these bindings
# override the pure-Python transcriptions above when the extension is
# importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        help_page_rows as _native_help_page_rows,
    )
    from ytm_player.utils._ytm_player_native import (
        help_page_visible_rows as _native_help_page_visible_rows,
    )
except ImportError:
    pass
else:

    def _rows_from_spec(  # noqa: F811
        spec: list[tuple[str, str, str, list[str]]],
    ) -> list[tuple[str, str, str, str]]:
        """The all-rows list (native Rust engine)."""
        return _native_help_page_rows(spec)

    def _visible_rows(  # noqa: F811
        all_rows: list[tuple[str, str, str, str]],
        filter_text: str,
    ) -> list[tuple[bool, str, str, str, str]]:
        """The filtered walk (native Rust engine)."""
        return _native_help_page_visible_rows(all_rows, filter_text)
