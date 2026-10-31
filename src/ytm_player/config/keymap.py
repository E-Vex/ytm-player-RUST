"""Keybinding system with multi-key sequences and modifier support."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    # Python 3.10 backport via PyPI
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

if sys.version_info >= (3, 11):
    from typing import Self
else:
    # Python 3.10 backport via PyPI
    from typing_extensions import Self  # pyright: ignore[reportMissingImports]

from ytm_player.config.paths import KEYMAP_FILE

# Try to load the native Rust matcher via PyO3.  If the compiled extension
# is unavailable (e.g. during development without a Rust toolchain), fall
# back to the pure-Python matching defined below.
try:
    from ..utils._ytm_player_native import (  # type: ignore[import-not-found]
        KeyMatcher as _NativeKeyMatcher,
    )
except ImportError:
    _NativeKeyMatcher = None

logger = logging.getLogger(__name__)

# Sentinel for "native matcher missing or failed to build — stop trying":
# distinguishes "not built yet" (None) from "unavailable" so match() never
# retries a failing construction on every keystroke.
_NATIVE_FAILED = object()


class Action(str, Enum):
    # Playback
    PLAY_PAUSE = "play_pause"
    NEXT_TRACK = "next_track"
    PREVIOUS_TRACK = "previous_track"
    PLAY_RANDOM = "play_random"
    CYCLE_REPEAT = "cycle_repeat"
    TOGGLE_SHUFFLE = "toggle_shuffle"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MUTE = "mute"
    SEEK_FORWARD = "seek_forward"
    SEEK_BACKWARD = "seek_backward"
    SEEK_START = "seek_start"

    # Navigation
    MOVE_DOWN = "move_down"
    MOVE_UP = "move_up"
    PAGE_DOWN = "page_down"
    PAGE_UP = "page_up"
    GO_TOP = "go_top"
    GO_BOTTOM = "go_bottom"
    SELECT = "select"
    FOCUS_NEXT = "focus_next"
    FOCUS_PREV = "focus_prev"
    FOCUS_PANE_LEFT = "focus_pane_left"
    FOCUS_PANE_RIGHT = "focus_pane_right"
    FOCUS_PANE_CYCLE = "focus_pane_cycle"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    QUIT = "quit"

    # Pages
    LIBRARY = "library"
    SEARCH = "search"
    BROWSE = "browse"
    LIKED_SONGS = "liked_songs"
    RECENTLY_PLAYED = "recently_played"
    LYRICS = "lyrics"
    CURRENT_CONTEXT = "current_context"
    JUMP_TO_CURRENT = "jump_to_current"
    QUEUE = "queue"
    HELP = "help"

    # Sidebars
    TOGGLE_SIDEBAR = "toggle_sidebar"
    TOGGLE_TRANSLITERATION = "toggle_transliteration"
    TOGGLE_ALBUM_ART = "toggle_album_art"

    # Actions
    DELETE_ITEM = "delete_item"
    TRACK_ACTIONS = "track_actions"
    LIKE_TOGGLE = "like_toggle"
    PLAY_NEXT = "play_next"
    ADD_TO_QUEUE = "add_to_queue"
    ADD_TO_PLAYLIST = "add_to_playlist"
    MARK_TOGGLE = "mark_toggle"
    MARK_RANGE = "mark_range"
    MARK_CLEAR = "mark_clear"
    DISCOVERY_MIX = "discovery_mix"
    FILTER = "filter"
    PICK_COUNTRY = "pick_country"
    REORDER_DOWN = "reorder_down"
    REORDER_UP = "reorder_up"

    # Sorting
    SORT_TITLE = "sort_title"
    SORT_ARTIST = "sort_artist"
    SORT_ALBUM = "sort_album"
    SORT_DURATION = "sort_duration"
    SORT_DATE = "sort_date"
    REVERSE_SORT = "reverse_sort"

    # Search
    TOGGLE_SEARCH_MODE = "toggle_search_mode"


class MatchResult(Enum):
    NO_MATCH = "no_match"
    PENDING = "pending"
    EXACT = "exact"


DEFAULT_BINDINGS: dict[str, list[str]] = {
    # Playback
    "play_pause": ["space"],
    "next_track": ["n"],
    "previous_track": ["p"],
    "play_random": ["."],
    "cycle_repeat": ["C-r"],
    "toggle_shuffle": ["C-s"],
    "volume_up": ["+"],
    "volume_down": ["-"],
    "mute": ["_"],
    "seek_forward": [">"],
    "seek_backward": ["<"],
    "seek_start": ["^"],
    # Navigation
    "move_down": ["j", "down", "C-n"],
    "move_up": ["k", "up", "C-p"],
    "page_down": ["C-f", "page_down"],
    "page_up": ["C-b", "page_up"],
    "go_top": ["g g", "home"],
    "go_bottom": ["G", "end"],
    "select": ["enter"],
    "focus_next": ["tab"],
    "focus_prev": ["S-tab"],
    # Vim window-split idiom: Ctrl+w then a direction moves between panes.
    "focus_pane_left": ["C-w h"],
    "focus_pane_right": ["C-w l"],
    "focus_pane_cycle": ["C-w w"],
    "go_back": ["backspace"],
    "go_forward": ["S-backspace"],
    "quit": ["q", "C-q"],
    # Pages
    "library": ["g l"],
    "search": ["g s"],
    "browse": ["g b"],
    "liked_songs": ["g y"],
    "recently_played": ["g r"],
    "lyrics": ["g L"],
    "current_context": ["g space"],
    "jump_to_current": ["g c"],
    "queue": ["z"],
    "help": ["?", "C-h"],
    # Sidebars
    "toggle_sidebar": ["C-e"],
    "toggle_transliteration": ["T"],
    "toggle_album_art": ["C-a"],
    # Actions
    "delete_item": ["delete", "d d"],
    "track_actions": ["a"],
    "like_toggle": ["l"],
    "play_next": ["X", "C-x"],
    "add_to_queue": ["Z", "C-z"],
    "add_to_playlist": ["A"],
    # Marks (multi-select): v toggles the highlighted track, V starts a
    # range that follows the cursor, Escape clears. A then adds every
    # marked track to a playlist.
    "mark_toggle": ["v"],
    "mark_range": ["V"],
    "mark_clear": ["escape"],
    "discovery_mix": ["D"],
    "filter": ["/"],
    "pick_country": ["c"],
    # Queue track reorder. Capital J/K (like the existing G/Z/A bindings):
    # lowercase j/k move the cursor, Shift+them move the selected track.
    # Count-aware — "15 J" moves the selected track down 15 positions.
    "reorder_down": ["J"],
    "reorder_up": ["K"],
    # Sorting
    "sort_title": ["s t"],
    "sort_artist": ["s a"],
    "sort_album": ["s A"],
    "sort_duration": ["s d"],
    "sort_date": ["s D"],
    "reverse_sort": ["s r"],
    # Search — no default binding: terminals commonly intercept Alt+letter
    # (menu accelerators), making any Alt default unreliable. Users can bind
    # their own key via keymap.toml.
    "toggle_search_mode": [],
}


def parse_key_sequence(raw: str) -> tuple[str, ...]:
    return tuple(raw.strip().split())


def _is_valid_binding(keys: object) -> bool:
    """A binding value must be a string or a list of strings."""
    if isinstance(keys, str):
        return True
    if isinstance(keys, list):
        return all(isinstance(k, str) for k in keys)
    return False


@dataclass
class KeyMap:
    """Ordered key-sequence → action bindings with prefix matching.

    When the native Rust extension is built, ``match`` resolves sequences
    through a precomputed hash table (exact map + prefix set) instead of
    scanning ``bindings`` on every keystroke.  The matcher is rebuilt
    lazily after every load; code that mutates ``bindings`` directly must
    reset ``_native_matcher`` to ``None`` so the native table cannot go
    stale.
    """

    bindings: dict[tuple[str, ...], Action] = field(default_factory=dict)
    # Tri-state native-matcher cache: None = not built yet, _NATIVE_FAILED
    # = extension missing / construction failed, KeyMatcher = current.
    _native_matcher: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: Path = KEYMAP_FILE) -> Self:
        keymap = cls()

        if not path.exists():
            keymap._load_defaults()
            return keymap

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            logger.warning(
                "Keymap file %s is unreadable/corrupted (%s) — backing up and using defaults.",
                path,
                exc,
            )
            backup = path.with_suffix(".toml.bak")
            try:
                # replace(), not rename(): on Windows rename raises
                # FileExistsError when a stale .bak already exists, which
                # would leave the corrupt file in place forever.
                path.replace(backup)
                logger.warning("Backed up broken keymap to %s", backup)
            except OSError:
                pass
            keymap._load_defaults()
            return keymap

        keymap._load_from_dict(data)
        return keymap

    def _load_defaults(self) -> None:
        for action_name, key_strs in DEFAULT_BINDINGS.items():
            action = Action(action_name)
            for key_str in key_strs:
                seq = parse_key_sequence(key_str)
                self.bindings[seq] = action
        self._native_matcher = None  # table changed; rebuild lazily

    def _load_from_dict(self, data: dict) -> None:
        self._load_defaults()

        for section in data.values():
            if not isinstance(section, dict):
                continue
            for action_name, keys in section.items():
                try:
                    action = Action(action_name)
                except ValueError:
                    logger.warning("Unknown keymap action %r — ignoring.", action_name)
                    continue

                if not _is_valid_binding(keys):
                    logger.warning(
                        "Keymap binding for %r must be a string or list of strings, "
                        "got %s — ignoring.",
                        action_name,
                        type(keys).__name__,
                    )
                    continue

                self._remove_action(action)

                if isinstance(keys, str):
                    keys = [keys]
                for key_str in keys:
                    seq = parse_key_sequence(key_str)
                    self.bindings[seq] = action
        self._native_matcher = None  # table changed; rebuild lazily

    def _remove_action(self, action: Action) -> None:
        to_remove = [k for k, v in self.bindings.items() if v == action]
        for key in to_remove:
            del self.bindings[key]
        self._native_matcher = None  # table changed; rebuild lazily

    def _native_match(
        self, key_sequence: tuple[str, ...]
    ) -> tuple[MatchResult, Action | None] | None:
        """Resolve *key_sequence* through the native Rust matcher.

        Returns ``None`` when the compiled extension is unavailable or
        failed to build, so ``match`` falls through to the pure-Python
        path — never a wrong answer, just the slower one.
        """
        matcher = self._native_matcher
        if matcher is None:
            if _NativeKeyMatcher is None:
                self._native_matcher = _NATIVE_FAILED
                return None
            try:
                matcher = _NativeKeyMatcher(self.bindings)
            except Exception:
                logger.debug(
                    "Native keymap matcher build failed; using Python fallback.",
                    exc_info=True,
                )
                self._native_matcher = _NATIVE_FAILED
                return None
            self._native_matcher = matcher
        if matcher is _NATIVE_FAILED:
            return None
        result, action_name = matcher.match_keys(key_sequence)
        if result == "exact":
            return MatchResult.EXACT, Action(action_name)
        if result == "pending":
            return MatchResult.PENDING, None
        return MatchResult.NO_MATCH, None

    def match(self, key_sequence: tuple[str, ...]) -> tuple[MatchResult, Action | None]:
        native = self._native_match(key_sequence)
        if native is not None:
            return native

        if key_sequence in self.bindings:
            return MatchResult.EXACT, self.bindings[key_sequence]

        for bound_seq in self.bindings:
            if (
                len(bound_seq) > len(key_sequence)
                and bound_seq[: len(key_sequence)] == key_sequence
            ):
                return MatchResult.PENDING, None

        return MatchResult.NO_MATCH, None

    def get_keys_for_action(self, action: Action) -> list[tuple[str, ...]]:
        return [seq for seq, act in self.bindings.items() if act == action]

    def format_key(self, seq: tuple[str, ...]) -> str:
        return " ".join(seq)


_keymap: KeyMap | None = None


def get_keymap() -> KeyMap:
    global _keymap
    if _keymap is None:
        _keymap = KeyMap.load()
    return _keymap
