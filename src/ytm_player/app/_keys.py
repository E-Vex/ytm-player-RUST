"""Key handling and action dispatch mixin for YTMPlayerApp.

The count/sequence buffer state machine is split out of ``on_key``: the
pure engine — ``_feed_key`` (the count-prefix accumulation and the
sequence-buffer resolution behind every keystroke) — is implemented by
the Rust extension (``_ytm_player_native.feed_key``) when it is built and
by the pure-Python transcription below otherwise. The mixin inspects the
event's focus context, applies the engine's outcome to the buffers,
suppresses the event as told and awaits the resolved action.

``_handle_action``'s dispatch table is split out the same way: the pure
engines — ``_dispatch_route`` (which branch of the dispatch handles an
action), ``_navigation_action_route`` (which pane a routed navigation
action targets), ``_current_context_target`` (the CURRENT_CONTEXT
album-target extraction) and ``_shuffle_lock_blocked`` (the TOGGLE_SHUFFLE
lock gate) — follow the same native-when-built / pure-Python-fallback
split. The mixin applies the branch and does the widget I/O; the awaited
handler calls remain pure mixin concerns.
"""

from __future__ import annotations

import logging
from typing import Any, cast

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


# ── Action dispatch engine ──────────────────────────────────────────

# The dispatch table behind ``_handle_action``: for each action, the
# branch that handles it (and, for the page navigations, the target
# page), in the reference's case order. ``match``'s value patterns
# compare with ``==`` (subject on the left) and are evaluated lazily
# case by case, so the walk below — first row whose member compares
# equal wins — is the table's exact semantics. Because ``Action`` is a
# str enum, a plain string like ``"play_pause"`` dispatches exactly like
# the member (``Action.PLAY_PAUSE == "play_pause"``), on both backends.
#
# Branch tokens name the handler the mixin applies; the routed
# navigation actions (the reference's one ``case`` with 23 alternatives)
# share the "route_navigation" branch, whose pane decision is
# ``_navigation_action_route`` below.
_DISPATCH_TABLE: tuple[tuple[Action, str, str | None], ...] = (
    # -- Playback controls --
    (Action.PLAY_PAUSE, "toggle_play_pause", None),
    (Action.NEXT_TRACK, "play_next", None),
    (Action.PREVIOUS_TRACK, "play_previous", None),
    (Action.PLAY_RANDOM, "play_random", None),
    (Action.VOLUME_UP, "volume_up", None),
    (Action.VOLUME_DOWN, "volume_down", None),
    (Action.MUTE, "mute", None),
    (Action.SEEK_FORWARD, "seek_forward", None),
    (Action.SEEK_BACKWARD, "seek_backward", None),
    (Action.SEEK_START, "seek_start", None),
    (Action.CYCLE_REPEAT, "cycle_repeat", None),
    (Action.TOGGLE_SHUFFLE, "toggle_shuffle", None),
    # -- Page navigation --
    (Action.LIBRARY, "navigate", "library"),
    (Action.SEARCH, "navigate", "search"),
    (Action.QUEUE, "navigate", "queue"),
    (Action.LYRICS, "toggle_lyrics_sidebar", None),
    (Action.TOGGLE_SIDEBAR, "toggle_playlist_sidebar", None),
    (Action.TOGGLE_TRANSLITERATION, "toggle_transliteration", None),
    (Action.TOGGLE_ALBUM_ART, "toggle_album_art", None),
    (Action.BROWSE, "navigate", "browse"),
    (Action.HELP, "navigate", "help"),
    (Action.LIKED_SONGS, "navigate", "liked_songs"),
    (Action.RECENTLY_PLAYED, "navigate", "recently_played"),
    (Action.CURRENT_CONTEXT, "current_context", None),
    (Action.GO_BACK, "navigate", "back"),
    (Action.GO_FORWARD, "navigate", "forward"),
    (Action.QUIT, "quit", None),
    # -- Add to playlist (quick shortcut for current track) --
    (Action.ADD_TO_PLAYLIST, "open_add_to_playlist", None),
    # -- Discovery roulette: random mix from one of seven sources --
    (Action.DISCOVERY_MIX, "run_discovery_mix", None),
    # -- Track actions (opens popup, handles result) --
    (Action.TRACK_ACTIONS, "open_track_actions", None),
    # -- Enqueue the focused track (works on every page) --
    (Action.PLAY_NEXT, "play_focused_next", None),
    (Action.ADD_TO_QUEUE, "add_focused_to_queue", None),
    (Action.LIKE_TOGGLE, "toggle_like_current", None),
    # -- Pane focus traversal (vim window split: Ctrl+w h/l/w) --
    (Action.FOCUS_PANE_LEFT, "focus_pane_left", None),
    (Action.FOCUS_PANE_RIGHT, "focus_pane_right", None),
    (Action.FOCUS_PANE_CYCLE, "cycle_pane", None),
    # -- Section focus traversal (Tab / Shift+Tab) --
    (Action.FOCUS_NEXT, "focus_next", None),
    (Action.FOCUS_PREV, "focus_prev", None),
    # -- Navigation actions routed to the active pane --
    (Action.MOVE_DOWN, "route_navigation", None),
    (Action.MOVE_UP, "route_navigation", None),
    (Action.PAGE_DOWN, "route_navigation", None),
    (Action.PAGE_UP, "route_navigation", None),
    (Action.GO_TOP, "route_navigation", None),
    (Action.GO_BOTTOM, "route_navigation", None),
    (Action.SELECT, "route_navigation", None),
    (Action.DELETE_ITEM, "route_navigation", None),
    (Action.FILTER, "route_navigation", None),
    (Action.SORT_TITLE, "route_navigation", None),
    (Action.SORT_ARTIST, "route_navigation", None),
    (Action.SORT_ALBUM, "route_navigation", None),
    (Action.SORT_DURATION, "route_navigation", None),
    (Action.SORT_DATE, "route_navigation", None),
    (Action.REVERSE_SORT, "route_navigation", None),
    (Action.JUMP_TO_CURRENT, "route_navigation", None),
    (Action.TOGGLE_SEARCH_MODE, "route_navigation", None),
    (Action.PICK_COUNTRY, "route_navigation", None),
    (Action.REORDER_DOWN, "route_navigation", None),
    (Action.REORDER_UP, "route_navigation", None),
    (Action.MARK_TOGGLE, "route_navigation", None),
    (Action.MARK_RANGE, "route_navigation", None),
    (Action.MARK_CLEAR, "route_navigation", None),
)


def _dispatch_route(
    action: Action | None,
    table: tuple[tuple[Action, str, str | None], ...],
) -> dict[str, Any]:
    """The dispatch table behind ``KeyHandlingMixin._handle_action``.

    The pure half of the dispatch: which branch handles *action* (and,
    for the page navigations, the target page). ``None`` dispatches
    nothing; each row of *table* is compared in order with ``==``
    (subject on the left, exactly like the reference ``match``'s value
    patterns — lazily, case by case, first equal match wins) and
    anything no row claims is unhandled. The str-enum wart carries
    over: a plain ``"play_pause"`` string dispatches like the member.

    Implemented by the Rust extension
    (``_ytm_player_native.dispatch_route``) when it is built; this
    pure-Python transcription is the fallback otherwise. Returns a
    fresh ``{"branch": str, "page": str | None}`` map — ``"none"`` for
    ``None``, ``"unhandled"`` when no row matches — and never mutates
    *action* or *table*.
    """
    if action is None:
        return {"branch": "none", "page": None}
    for member, branch, page in table:
        if action == member:
            return {"branch": branch, "page": page}
    return {"branch": "unhandled", "page": None}


def _navigation_action_route(
    pane: str,
    action: Action,
    sidebar_actions: frozenset[Action],
    lyrics_actions: frozenset[Action],
) -> str:
    """The pane decision behind ``KeyHandlingMixin._route_navigation_action``.

    The pure half of the pane routing: a routed navigation action
    targets the Playlists sidebar's widget while that pane holds focus
    and the action is in *sidebar_actions* (movement, paging, top /
    bottom, select and filter — see ``PlaylistSidebar
    .handle_sidebar_action``), the lyrics pane's widget while IT holds
    focus and the action is in *lyrics_actions* (scroll only — see
    ``LyricsSidebar.handle_action``), and the current page otherwise.
    Membership goes through Python's ``in`` (hash + ``==``), so a plain
    string action value routes like its member on both backends.

    The reference's ``if`` / ``elif`` collapses to two independent
    gates here: the ``elif`` is only skipped when the first condition
    holds, which already means the pane is ``"playlists"`` and the
    second gate (``pane == "lyrics"``) could never fire — so the
    outcomes are identical. Whether the target widget actually exists
    (and the fall-back-to-page when it doesn't) stays in the mixin.

    Implemented by the Rust extension
    (``_ytm_player_native.route_navigation_action``) when it is built;
    this pure-Python transcription is the fallback otherwise. Returns
    ``"sidebar" | "lyrics" | "page"``.
    """
    if pane == "playlists" and action in sidebar_actions:
        return "sidebar"
    if pane == "lyrics" and action in lyrics_actions:
        return "lyrics"
    return "page"


def _current_context_target(track: dict | None) -> dict[str, Any]:
    """The album-target extraction behind the CURRENT_CONTEXT dispatch.

    The pure half of the ``Action.CURRENT_CONTEXT`` branch: a falsy
    *track* (nothing playing — an empty dict counts) can only notify;
    otherwise the album id is ``track["album_id"]``, falling back to
    ``track["album"]["id"]`` when the former is falsy and the album is
    a dict (the normalized track format stores both shapes). A truthy
    id navigates to the album's context page; anything else notifies.

    Implemented by the Rust extension
    (``_ytm_player_native.current_context_target``) when it is built;
    this pure-Python transcription is the fallback otherwise. Returns a
    fresh ``{"outcome": "navigate", "album_id": <the id unchanged>}``
    or ``{"outcome": "notify", "reason": "no_track" | "no_album"}``;
    *track* is read duck-typed (``.get``) and never mutated.
    """
    if not track:
        return {"outcome": "notify", "reason": "no_track"}
    album_id = track.get("album_id")
    if not album_id:
        album = track.get("album")
        if isinstance(album, dict):
            album_id = album.get("id")
    if album_id:
        return {"outcome": "navigate", "album_id": album_id}
    return {"outcome": "notify", "reason": "no_album"}


def _shuffle_lock_blocked(context_id: str | None, prefs: Any) -> bool:
    """The Shuffle-lock gate behind the TOGGLE_SHUFFLE dispatch.

    The pure half of the ``Action.TOGGLE_SHUFFLE`` branch's no-op guard:
    with no current context there is nothing locked, and otherwise the
    playlist's stored preference decides. Both reads use Python
    truthiness, and ``prefs.get`` is only consulted for a truthy
    *context_id* — the reference's ``ctx and self.shuffle_prefs
    .get(ctx)`` short-circuits exactly the same way.

    Implemented by the Rust extension
    (``_ytm_player_native.shuffle_lock_blocked``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if not context_id:
        return False
    return bool(prefs.get(context_id))


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
        """Dispatch a resolved action to the appropriate handler.

        The dispatch table — which branch handles which action, in the
        reference's case order — is resolved by the ``_dispatch_route``
        engine above (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs each
        branch's calls. The branch decisions that read app state go
        through their engines too (the CURRENT_CONTEXT album-target
        extraction and the TOGGLE_SHUFFLE lock gate); everything else is
        direct widget/service I/O.
        """
        route = _dispatch_route(action, _DISPATCH_TABLE)

        match route["branch"]:
            case "none":
                return

            # -- Playback controls --
            case "toggle_play_pause":
                await self._toggle_play_pause()

            case "play_next":
                await self._play_next()

            case "play_previous":
                await self._play_previous()

            case "play_random":
                track = self.queue.play_random()
                if track:
                    await self.play_track(track)

            case "volume_up":
                if self.player:
                    await self.player.change_volume(5 * count)

            case "volume_down":
                if self.player:
                    await self.player.change_volume(-5 * count)

            case "mute":
                if self.player:
                    await self.player.mute()

            case "seek_forward":
                if self.player:
                    await self.player.seek(self.settings.playback.seek_step * count)

            case "seek_backward":
                if self.player:
                    await self.player.seek(-self.settings.playback.seek_step * count)

            case "seek_start":
                if self.player:
                    await self.player.seek_start()

            case "cycle_repeat":
                mode = self.queue.cycle_repeat()
                bar = self.query_one("#playback-bar", PlaybackBar)
                bar.update_repeat(mode)
                self.notify(f"Repeat: {mode.value}", timeout=2)

            case "toggle_shuffle":
                # If the current playlist has Shuffle lock on, the keyboard
                # shortcut is also a no-op — direct the user to the lock toggle.
                if _shuffle_lock_blocked(self.queue.current_context_id, self.shuffle_prefs):
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
            case "navigate":
                await self.navigate_to(route["page"])

            case "toggle_lyrics_sidebar":
                self._toggle_lyrics_sidebar()

            case "toggle_playlist_sidebar":
                self._toggle_playlist_sidebar()

            case "toggle_transliteration":
                try:
                    self.query_one("#lyrics-sidebar", LyricsSidebar).toggle_transliteration()
                except Exception:
                    pass

            case "toggle_album_art":
                self._toggle_album_art()

            case "current_context":
                target = _current_context_target(self.queue.current_track)
                if target["outcome"] == "navigate":
                    await self.navigate_to(
                        "context", context_type="album", context_id=target["album_id"]
                    )
                elif target["reason"] == "no_track":
                    self.notify("No track playing", severity="warning", timeout=2)
                else:
                    self.notify("No album info for current track", severity="warning", timeout=2)

            case "quit":
                self.exit()

            # -- Add to playlist (quick shortcut for current track) --
            case "open_add_to_playlist":
                await self._open_add_to_playlist()

            # -- Discovery roulette: random mix from one of seven sources --
            case "run_discovery_mix":
                self.run_worker(self._start_discovery_mix(), group="discovery-mix", exclusive=True)

            # -- Track actions (opens popup, handles result) --
            case "open_track_actions":
                await self._open_track_actions()

            # -- Enqueue the focused track (works on every page) --
            case "play_focused_next":
                self._play_focused_next()

            case "add_focused_to_queue":
                self._add_focused_to_queue()

            case "toggle_like_current":
                await self._toggle_like_current()

            # -- Pane focus traversal (vim window split: Ctrl+w h/l/w) --
            case "focus_pane_left":
                self._focus_pane_left()

            case "focus_pane_right":
                self._focus_pane_right()

            case "cycle_pane":
                self._cycle_pane()

            # -- Section focus traversal (Tab / Shift+Tab) --
            # Handled once, here, via Textual's native focus chain, which
            # walks every displayed focusable widget in DOM order — content
            # widgets AND any visible sidebar, skipping hidden ones. Pages no
            # longer implement FOCUS_NEXT/FOCUS_PREV themselves.
            case "focus_next":
                self.action_focus_next()

            case "focus_prev":
                self.action_focus_previous()

            # -- Navigation actions routed to the active pane --
            case "route_navigation":
                # The None branch of the dispatch table consumes a missing
                # action before any row is consulted, so a row-routed
                # action is never None (a plain string member value flows
                # on with its own equality, exactly like the reference's
                # `case Action.MOVE_DOWN | ...` pattern match).
                await self._route_navigation_action(cast("Action", action), count)

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
        or a mouse click. The pane/action decision itself is delegated to
        the ``_navigation_action_route`` engine above (native when the Rust
        extension is built, the pure-Python transcription otherwise); this
        method looks up the target widget — falling back to the page when
        the sidebar isn't mounted — and performs the dispatch.
        """
        target = _navigation_action_route(
            self._current_pane(), action, _SIDEBAR_PANE_ACTIONS, _LYRICS_PANE_ACTIONS
        )

        if target == "sidebar":
            try:
                sidebar = self.query_one("#playlist-sidebar", PlaylistSidebar)
            except Exception:
                sidebar = None
            if sidebar is not None:
                sidebar.handle_sidebar_action(action, count)
                return
        elif target == "lyrics":
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
        current_context_target as _native_current_context_target,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        dispatch_route as _native_dispatch_route,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        feed_key as _native_feed_key,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        normalize_key as _native_normalize_key,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        route_navigation_action as _native_route_navigation_action,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        shuffle_lock_blocked as _native_shuffle_lock_blocked,
    )
except ImportError:
    pass
else:
    _normalize_key_components = (  # type: ignore[assignment]
        _native_normalize_key
    )
    _feed_key = _native_feed_key  # type: ignore[assignment]
    _dispatch_route = _native_dispatch_route  # type: ignore[assignment]
    _navigation_action_route = (  # type: ignore[assignment]
        _native_route_navigation_action
    )
    _current_context_target = (  # type: ignore[assignment]
        _native_current_context_target
    )
    _shuffle_lock_blocked = _native_shuffle_lock_blocked  # type: ignore[assignment]
