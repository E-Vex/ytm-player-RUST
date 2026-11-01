"""Key handling and action dispatch mixin for YTMPlayerApp.

The count/sequence buffer state machine is split out of ``on_key``: the
pure engine — ``_feed_key`` (the count-prefix accumulation and the
sequence-buffer resolution behind every keystroke) — is implemented by
the Rust extension (``_ytm_player_native.feed_key``) when it is built and
by the pure-Python transcription below otherwise. The mixin inspects the
event's focus context, applies the engine's outcome to the buffers,
suppresses the event as told and awaits the resolved action; the
``_handle_action`` dispatch table and the pane routing below it remain
pure mixin concerns (widget I/O).
"""

from __future__ import annotations

import logging
from typing import Any

from textual.events import Key

from ytm_player.app._base import YTMHostBase
from ytm_player.config import Action, KeyMap, MatchResult
from ytm_player.ui.playback_bar import PlaybackBar
from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar
from ytm_player.ui.sidebars.playlist_sidebar import PlaylistSidebar

logger = logging.getLogger(__name__)

_MAX_KEY_COUNT = 1000

# Actions the Playlists sidebar can handle when it holds keyboard focus
# (see PlaylistSidebar.handle_sidebar_action).
_SIDEBAR_PANE_ACTIONS = frozenset(
    {
        Action.MOVE_DOWN,
        Action.MOVE_UP,
        Action.PAGE_DOWN,
        Action.PAGE_UP,
        Action.GO_TOP,
        Action.GO_BOTTOM,
        Action.SELECT,
        Action.FILTER,
    }
)

# Actions the lyrics pane can handle when it holds keyboard focus
# (see LyricsSidebar.handle_action — scroll only, no select/filter).
_LYRICS_PANE_ACTIONS = frozenset(
    {
        Action.MOVE_DOWN,
        Action.MOVE_UP,
        Action.PAGE_DOWN,
        Action.PAGE_UP,
        Action.GO_TOP,
        Action.GO_BOTTOM,
    }
)


def _normalize_key_components(key: str, character: str | None, is_printable: bool) -> str:
    """Convert a Textual key event's primitive components into the string
    format used by KeyMap.

    Textual key names like 'ctrl+r' become 'C-r', 'shift+tab' becomes
    'S-tab', 'alt+v' becomes 'M-v'; a single-code-point printable
    non-whitespace character is itself (Textual names punctuation
    symbolically — '>' arrives as 'greater_than_sign' — while the keymap
    binds the literal character; whitespace keeps its name); the special
    names below are remapped and everything else passes through.

    Implemented by the Rust extension
    (``_ytm_player_native.normalize_key``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    # Textual uses names like "ctrl+x", "shift+tab", "alt+v".
    if key.startswith("ctrl+"):
        return f"C-{key[5:]}"
    if key.startswith("shift+"):
        return f"S-{key[6:]}"
    if key.startswith("alt+"):
        return f"M-{key[4:]}"

    # Textual names punctuation symbolically (">" arrives as
    # "greater_than_sign", "^" as "circumflex_accent") while the keymap
    # binds the literal character, so prefer the character itself.
    # Whitespace is excluded so "space" keeps its name.
    if character is not None and len(character) == 1 and is_printable and not character.isspace():
        return character

    # Map Textual's special key names to our keymap names.
    key_map = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "home": "home",
        "end": "end",
        "pageup": "page_up",
        "pagedown": "page_down",
        "page_up": "page_up",
        "page_down": "page_down",
        "backspace": "backspace",
        "delete": "delete",
        "tab": "tab",
        "enter": "enter",
        "return": "enter",
        "escape": "escape",
        "plus": "+",
        "minus": "-",
        "equals": "=",
        "question_mark": "?",
        "slash": "/",
    }

    return key_map.get(key, key)


def _feed_key(
    key: str,
    key_buffer: list[str],
    count_buffer: str,
    keymap: KeyMap,
) -> dict[str, Any]:
    """The count/sequence buffer state machine behind ``on_key``.

    The pure half of ``KeyHandlingMixin.on_key``: a digit key with an
    empty *key_buffer* accumulates the *count_buffer* prefix (the event
    is default-prevented but not stopped); anything else is appended to
    the sequence buffer and classified by ``keymap.match`` — an exact
    match resolves the count (``int(count_buffer)`` — which raises for
    digit-classified non-decimal keys like ``'²'``, the reference's
    latent wart — defaulting to 1 when no count was typed, capped at
    ``_MAX_KEY_COUNT``) and clears both buffers, a pending prefix keeps
    the grown sequence waiting for more keys (the count survives the
    prefix: ``5`` ``g`` ``g`` goes to top with count 5), and no match
    clears both buffers while letting the event flow on.

    Implemented by the Rust extension (``_ytm_player_native.feed_key``)
    when it is built; this pure-Python transcription is the fallback
    otherwise. The outcome map is fresh in every field and neither input
    is mutated: ``{"outcome": "count_digit" | "exact" | "pending" |
    "no_match", "prevent_default": bool, "stop": bool, "key_buffer":
    list, "count_buffer": str, "count": int | None, "action": Action |
    None}``.
    """
    # Digit handling: accumulate count prefix if no keys buffered yet.
    if key.isdigit() and not key_buffer:
        return {
            "outcome": "count_digit",
            "prevent_default": True,
            "stop": False,
            "key_buffer": list(key_buffer),
            "count_buffer": count_buffer + key,
            "count": None,
            "action": None,
        }

    sequence = (*key_buffer, key)
    result, action = keymap.match(sequence)

    if result == MatchResult.EXACT:
        count = int(count_buffer) if count_buffer else 1
        count = min(count, _MAX_KEY_COUNT)  # Safety cap.
        return {
            "outcome": "exact",
            "prevent_default": True,
            "stop": True,
            "key_buffer": [],
            "count_buffer": "",
            "count": count,
            "action": action,
        }

    if result == MatchResult.PENDING:
        # Waiting for more keys in the sequence.
        return {
            "outcome": "pending",
            "prevent_default": True,
            "stop": True,
            "key_buffer": list(sequence),
            "count_buffer": count_buffer,
            "count": None,
            "action": None,
        }

    # No match -- reset buffers.
    return {
        "outcome": "no_match",
        "prevent_default": False,
        "stop": False,
        "key_buffer": [],
        "count_buffer": "",
        "count": None,
        "action": None,
    }


class KeyHandlingMixin(YTMHostBase):
    """Keyboard input processing and action dispatch."""

    async def on_key(self, event: Key) -> None:
        """Process keyboard input through the KeyMap system.

        Supports vim-style count prefixes (e.g. "5j" to move down 5 rows)
        and multi-key sequences (e.g. "g g" to go to top).

        The count/sequence buffer state machine (digit accumulation,
        sequence classification, count resolution and buffer resets) is
        delegated to the ``_feed_key`` engine above — native when the
        Rust extension is built, the pure-Python transcription otherwise;
        this handler keeps the event/focus inspection, applies the
        engine's outcome and awaits the dispatched action.
        """
        # Don't intercept keys when a modal screen is active -- let the
        # modal's own widgets (Input, ListView, etc.) handle them.
        if self.screen.is_modal:
            return

        # Don't intercept keys when an Input or TextArea is focused -- let
        # the widget handle normal text entry.
        from textual.widgets import Input, TextArea

        focused = self.focused
        if isinstance(focused, (Input, TextArea)):
            return

        key = self._normalize_key(event)

        outcome = _feed_key(key, self._key_buffer, self._count_buffer, self.keymap)
        self._key_buffer = outcome["key_buffer"]
        self._count_buffer = outcome["count_buffer"]

        if outcome["prevent_default"]:
            event.prevent_default()
        if outcome["stop"]:
            event.stop()

        if outcome["outcome"] == "exact":
            await self._handle_action(outcome["action"], outcome["count"])

    @staticmethod
    def _normalize_key(event: Key) -> str:
        """Convert a Textual Key event into the string format used by KeyMap.

        Textual key names like 'ctrl+r' become 'C-r', 'shift+tab' becomes
        'S-tab', etc. Delegates to the engine (native when built) with the
        event's primitive components; a duck-typed stand-in's non-string
        character reads as ``None``, degenerating exactly like the
        reference's own ``len(character) == 1`` gate.
        """
        character = event.character
        return _normalize_key_components(
            event.key,
            character if isinstance(character, str) else None,
            bool(event.is_printable),
        )

    async def _handle_action(self, action: Action | None, count: int = 1) -> None:
        """Dispatch a resolved action to the appropriate handler."""
        if action is None:
            return

        match action:
            # -- Playback controls --
            case Action.PLAY_PAUSE:
                await self._toggle_play_pause()

            case Action.NEXT_TRACK:
                await self._play_next()

            case Action.PREVIOUS_TRACK:
                await self._play_previous()

            case Action.PLAY_RANDOM:
                track = self.queue.play_random()
                if track:
                    await self.play_track(track)

            case Action.VOLUME_UP:
                if self.player:
                    await self.player.change_volume(5 * count)

            case Action.VOLUME_DOWN:
                if self.player:
                    await self.player.change_volume(-5 * count)

            case Action.MUTE:
                if self.player:
                    await self.player.mute()

            case Action.SEEK_FORWARD:
                if self.player:
                    await self.player.seek(self.settings.playback.seek_step * count)

            case Action.SEEK_BACKWARD:
                if self.player:
                    await self.player.seek(-self.settings.playback.seek_step * count)

            case Action.SEEK_START:
                if self.player:
                    await self.player.seek_start()

            case Action.CYCLE_REPEAT:
                mode = self.queue.cycle_repeat()
                bar = self.query_one("#playback-bar", PlaybackBar)
                bar.update_repeat(mode)
                self.notify(f"Repeat: {mode.value}", timeout=2)

            case Action.TOGGLE_SHUFFLE:
                # If the current playlist has Shuffle lock on, the keyboard
                # shortcut is also a no-op — direct the user to the lock toggle.
                ctx = self.queue.current_context_id
                if ctx and self.shuffle_prefs.get(ctx):
                    self.notify(
                        "Shuffle is locked for this playlist — "
                        "toggle Shuffle lock in the playlist header.",
                        severity="warning",
                        timeout=4,
                    )
                    return
                self.queue.toggle_shuffle()
                self._refresh_queue_page()
                bar = self.query_one("#playback-bar", PlaybackBar)
                bar.update_shuffle(self.queue.shuffle_enabled)
                state = "on" if self.queue.shuffle_enabled else "off"
                self.notify(f"Shuffle: {state}", timeout=2)

            # -- Page navigation --
            case Action.LIBRARY:
                await self.navigate_to("library")
            case Action.SEARCH:
                await self.navigate_to("search")
            case Action.QUEUE:
                await self.navigate_to("queue")
            case Action.LYRICS:
                self._toggle_lyrics_sidebar()
            case Action.TOGGLE_SIDEBAR:
                self._toggle_playlist_sidebar()
            case Action.TOGGLE_TRANSLITERATION:
                try:
                    self.query_one("#lyrics-sidebar", LyricsSidebar).toggle_transliteration()
                except Exception:
                    pass
            case Action.TOGGLE_ALBUM_ART:
                self._toggle_album_art()
            case Action.BROWSE:
                await self.navigate_to("browse")
            case Action.HELP:
                await self.navigate_to("help")
            case Action.LIKED_SONGS:
                await self.navigate_to("liked_songs")
            case Action.RECENTLY_PLAYED:
                await self.navigate_to("recently_played")
            case Action.CURRENT_CONTEXT:
                track = self.queue.current_track
                if track:
                    album_id = track.get("album_id")
                    album = track.get("album")
                    if not album_id and isinstance(album, dict):
                        album_id = album.get("id")
                    if album_id:
                        await self.navigate_to("context", context_type="album", context_id=album_id)
                    else:
                        self.notify(
                            "No album info for current track", severity="warning", timeout=2
                        )
                else:
                    self.notify("No track playing", severity="warning", timeout=2)

            case Action.GO_BACK:
                await self.navigate_to("back")

            case Action.GO_FORWARD:
                await self.navigate_to("forward")

            case Action.QUIT:
                self.exit()

            # -- Add to playlist (quick shortcut for current track) --
            case Action.ADD_TO_PLAYLIST:
                await self._open_add_to_playlist()

            # -- Discovery roulette: random mix from one of seven sources --
            case Action.DISCOVERY_MIX:
                self.run_worker(self._start_discovery_mix(), group="discovery-mix", exclusive=True)

            # -- Track actions (opens popup, handles result) --
            case Action.TRACK_ACTIONS:
                await self._open_track_actions()

            # -- Enqueue the focused track (works on every page) --
            case Action.PLAY_NEXT:
                self._play_focused_next()

            case Action.ADD_TO_QUEUE:
                self._add_focused_to_queue()

            case Action.LIKE_TOGGLE:
                await self._toggle_like_current()

            # -- Pane focus traversal (vim window split: Ctrl+w h/l/w) --
            case Action.FOCUS_PANE_LEFT:
                self._focus_pane_left()

            case Action.FOCUS_PANE_RIGHT:
                self._focus_pane_right()

            case Action.FOCUS_PANE_CYCLE:
                self._cycle_pane()

            # -- Section focus traversal (Tab / Shift+Tab) --
            # Handled once, here, via Textual's native focus chain, which
            # walks every displayed focusable widget in DOM order — content
            # widgets AND any visible sidebar, skipping hidden ones. Pages no
            # longer implement FOCUS_NEXT/FOCUS_PREV themselves.
            case Action.FOCUS_NEXT:
                self.action_focus_next()

            case Action.FOCUS_PREV:
                self.action_focus_previous()

            # -- Navigation actions routed to the active pane --
            case (
                Action.MOVE_DOWN
                | Action.MOVE_UP
                | Action.PAGE_DOWN
                | Action.PAGE_UP
                | Action.GO_TOP
                | Action.GO_BOTTOM
                | Action.SELECT
                | Action.DELETE_ITEM
                | Action.FILTER
                | Action.SORT_TITLE
                | Action.SORT_ARTIST
                | Action.SORT_ALBUM
                | Action.SORT_DURATION
                | Action.SORT_DATE
                | Action.REVERSE_SORT
                | Action.JUMP_TO_CURRENT
                | Action.TOGGLE_SEARCH_MODE
                | Action.PICK_COUNTRY
                | Action.REORDER_DOWN
                | Action.REORDER_UP
                | Action.MARK_TOGGLE
                | Action.MARK_RANGE
                | Action.MARK_CLEAR
            ):
                await self._route_navigation_action(action, count)

            case _:
                logger.debug("Unhandled action: %s", action)

    async def _route_navigation_action(self, action: Action, count: int) -> None:
        """Route a movement/select/filter action to the active pane.

        When a sidebar pane holds keyboard focus, the relevant subset of
        actions drives that sidebar's widget; everything else (and the
        default ``content`` pane) falls through to the current page's
        ``handle_action``. This is what makes ``j``/``k``/``Enter``/``/``
        actually operate the Playlists or lyrics pane after ``Ctrl+w``
        focuses it — a bare ``.focus()`` is not enough because ``on_key``
        intercepts every keystroke before Textual's focus chain runs.

        The active pane is derived from the *actually-focused* widget
        (``_current_pane()``) rather than a cached field, so routing stays
        correct no matter how focus moved — ``Tab``/``Shift+Tab``, ``Ctrl+w``,
        or a mouse click.
        """
        pane = self._current_pane()
        if pane == "playlists" and action in _SIDEBAR_PANE_ACTIONS:
            try:
                sidebar = self.query_one("#playlist-sidebar", PlaylistSidebar)
            except Exception:
                sidebar = None
            if sidebar is not None:
                sidebar.handle_sidebar_action(action, count)
                return
        elif pane == "lyrics" and action in _LYRICS_PANE_ACTIONS:
            try:
                lyrics = self.query_one("#lyrics-sidebar", LyricsSidebar)
            except Exception:
                lyrics = None
            if lyrics is not None:
                await lyrics.handle_action(action, count)
                return

        page = self._get_current_page()
        if page and hasattr(page, "handle_action"):
            await page.handle_action(action, count)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace
# the pure-Python transcriptions above when the extension is importable.
# The rebinding must happen AFTER the definitions above — assigning the
# native functions any earlier would be clobbered by the ``def``
# statements (the lesson that produced formatting.py's dead native
# bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        feed_key as _native_feed_key,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        normalize_key as _native_normalize_key,
    )
except ImportError:
    pass
else:
    _normalize_key_components = (  # type: ignore[assignment]
        _native_normalize_key
    )
    _feed_key = _native_feed_key  # type: ignore[assignment]
