"""Per-collection shuffle preferences, persisted to disk.

Keys are the collection's stable ID (playlistId, browseId, channelId).
Value is a bool: True = shuffle on, False = shuffle off.
LRU-capped at 1000 entries to prevent unbounded growth.

The LRU bookkeeping — the ordered ``{context_id: shuffle}`` map itself
(the get promotion, the set with its move_to_end and evict-while-over
walk, the hand-edited-file ingestion and the snapshot + serialisation
behind ``_save``) — lives in a ``ShufflePrefsCore``: the Rust
implementation from the native extension
(``_ytm_player_native.ShufflePrefsCore``) when it is built, and the
pure-Python ``_ShufflePrefsCore`` transcription below otherwise. The
``ShufflePreferences`` facade keeps the thread lock, the
falsy-context guards, the JSON parse and the atomic file write.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 1000

# Try to load the native Rust engine via PyO3. If the compiled extension
# is unavailable (e.g. during development without a Rust toolchain), fall
# back to the pure-Python core defined below.
try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ShufflePrefsCore as _NativeShufflePrefsCore,
    )
except ImportError:
    _NativeShufflePrefsCore = None


class ShufflePrefsCoreLike(Protocol):
    """The LRU-bookkeeping contract both core implementations satisfy.

    ``ShufflePreferences`` calls these under its lock; the falsy-context
    guards, the JSON parse and the file I/O stay in the facade.
    """

    def get(self, context_id: Any) -> bool | None: ...
    def set(self, context_id: Any, shuffle: bool) -> None: ...
    def clear(self) -> None: ...
    def load(self, data: Any) -> None: ...
    def dump_json(self) -> str: ...
    def length(self) -> int: ...


def _new_core() -> ShufflePrefsCoreLike:
    """Build the best available shuffle-preferences core."""
    if _NativeShufflePrefsCore is not None:
        return _NativeShufflePrefsCore(_MAX_ENTRIES)
    return _ShufflePrefsCore(_MAX_ENTRIES)


class _ShufflePrefsCore:
    """Pure-Python shuffle-preferences LRU engine (fallback for the
    native Rust core).

    A transcription of the reference's ``OrderedDict`` bookkeeping minus
    the lock (the facade holds it), the falsy-context guards and the file
    I/O: the entries are ordered least-recent first, ``get`` promotes a
    hit to most-recent, ``set`` updates-or-appends and then promotes
    (an existing key's value updated in place, its stored key object
    kept) and evicts oldest-first while over the cap, ``load`` ingests a
    parsed prefs file (the dict gate, the str-key filter before the
    value is ever coerced, and the ``bool(v)`` truthiness coercion — the
    whole mapping built before the state is swapped, so a mid-walk
    failure leaves the old state intact), and ``dump_json`` renders the
    LRU order exactly as ``json.dumps(dict(...), indent=2)`` renders it.
    """

    def __init__(self, max_entries: int) -> None:
        self._prefs: OrderedDict[Any, bool] = OrderedDict()
        self._max_entries = max_entries

    def get(self, context_id: Any) -> bool | None:
        value = self._prefs.get(context_id)
        if value is not None:
            self._prefs.move_to_end(context_id)
        return value

    def set(self, context_id: Any, shuffle: bool) -> None:
        self._prefs[context_id] = shuffle
        self._prefs.move_to_end(context_id)
        while len(self._prefs) > self._max_entries:
            self._prefs.popitem(last=False)

    def clear(self) -> None:
        self._prefs.clear()

    def load(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        self._prefs = OrderedDict((k, bool(v)) for k, v in data.items() if isinstance(k, str))

    def dump_json(self) -> str:
        return json.dumps(dict(self._prefs), indent=2)

    def length(self) -> int:
        return len(self._prefs)


class ShufflePreferences:
    """Thread-safe persistent dict of {context_id: shuffle_enabled}.

    Thin locking facade over a ``ShufflePrefsCore`` (the native Rust
    engine when the extension is built, the pure-Python transcription
    otherwise): the lock, the falsy-context guards, the JSON parse of
    the file and its atomic write live here; the ordered map, its
    ingestion and its serialisation live in the core.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._core: ShufflePrefsCoreLike = _new_core()
        self._load()

    def get(self, context_id: str | None) -> bool | None:
        """Return saved shuffle state, or None if no preference / no context."""
        if not context_id:
            return None
        with self._lock:
            return self._core.get(context_id)

    def set(self, context_id: str | None, shuffle: bool) -> None:
        """Persist shuffle state for *context_id*. No-op if context_id is None."""
        if not context_id:
            return
        with self._lock:
            self._core.set(context_id, shuffle)
        self._save()

    def clear(self) -> None:
        with self._lock:
            self._core.clear()
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._core.load(data)
        except Exception:
            logger.exception("Failed to load shuffle prefs from %s", self._path)

    def _save(self) -> None:
        # Snapshot + serialise under the lock so a concurrent set()/clear()
        # can't mutate the state mid-serialisation; the file I/O stays
        # outside the lock.
        with self._lock:
            text = self._core.dump_json()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(self._path)
        except Exception:
            logger.exception("Failed to save shuffle prefs to %s", self._path)
