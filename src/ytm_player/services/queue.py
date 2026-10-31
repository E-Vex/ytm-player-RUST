"""Playback queue management.

The ordering engine — the track list, one entry id per occurrence, the
shuffle order, the repeat/shuffle state and the removed-current
bookkeeping — lives in a ``QueueCore``: the Rust implementation from the
native extension (``_ytm_player_native.QueueCore``) when it is built, and
the pure-Python ``_QueueCore`` transcription below otherwise.
``QueueManager`` is the locking facade the rest of the app talks to: it
owns the ``threading.Lock``, maps ``RepeatMode`` to the core's integer
codes, and keeps the non-ordering state (collection context id, radio
seeds) that does not belong to the engine.

Both cores draw randomness through the global ``random`` module (the
native core holds the module object and calls back into it), so
``random.seed(...)`` and monkeypatched ``random.shuffle`` /
``random.randint`` / ``random.randrange`` behave identically on either
backend — the native/fallback parity tests rely on that.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import TYPE_CHECKING, Protocol

from ytm_player.utils.compat import StrEnum, auto

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Try to load the native Rust engine via PyO3. If the compiled extension
# is unavailable (e.g. during development without a Rust toolchain), fall
# back to the pure-Python core defined below.
try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        QueueCore as _NativeQueueCore,
    )
except ImportError:
    _NativeQueueCore = None


class RepeatMode(StrEnum):
    """Repeat mode for the playback queue."""

    OFF = auto()
    ALL = auto()
    ONE = auto()


# Repeat-mode codes shared with both core implementations (the RepeatMode
# member order). The facade translates between the enum and the codes.
_REPEAT_CODES: dict[RepeatMode, int] = {
    RepeatMode.OFF: 0,
    RepeatMode.ALL: 1,
    RepeatMode.ONE: 2,
}
_REPEAT_MODES: tuple[RepeatMode, ...] = (RepeatMode.OFF, RepeatMode.ALL, RepeatMode.ONE)


class QueueCoreLike(Protocol):
    """The ordering-engine contract both core implementations satisfy.

    ``QueueManager`` calls these under its lock. Methods return plain
    ints/bools/tuples and the *same track objects* they were given — never
    copies — so occurrence identity survives every operation.
    """

    def add(self, track: dict, position: int | None = None) -> None: ...
    def add_next(self, track: dict) -> None: ...
    def add_next_multiple(self, tracks: list[dict]) -> None: ...
    def add_multiple(self, tracks: list[dict]) -> None: ...
    def remove(self, index: int) -> None: ...
    def remove_entry(self, entry_id: int) -> bool: ...
    def clear(self) -> None: ...
    def move(self, from_idx: int, to_idx: int) -> None: ...
    def current(self) -> dict | None: ...
    def next_track(self) -> dict | None: ...
    def previous_track(self) -> dict | None: ...
    def peek_next(self) -> dict | None: ...
    def play_random(self) -> dict | None: ...
    def filter_not_in_queue(self, tracks: list[dict]) -> list[dict]: ...
    def jump_to(self, index: int) -> dict | None: ...
    def jump_to_real(self, real_index: int) -> dict | None: ...
    def jump_to_entry(self, entry_id: int) -> dict | None: ...
    def toggle_shuffle(self) -> None: ...
    def set_repeat(self, code: int) -> None: ...
    def cycle_repeat(self) -> int: ...
    def generation(self) -> int: ...
    def current_index(self) -> int: ...
    def length(self) -> int: ...
    def is_empty(self) -> bool: ...
    def repeat_code(self) -> int: ...
    def shuffle_enabled(self) -> bool: ...
    def real_index(self) -> int: ...
    def remaining_tracks(self) -> int: ...
    def tracks(self) -> tuple[dict, ...]: ...
    def entries(self) -> tuple[tuple[int, dict], ...]: ...
    def force_shuffle_state(self, order: list[int], position: int) -> None: ...


def _new_core() -> QueueCoreLike:
    """Build the best available queue core."""
    if _NativeQueueCore is not None:
        return _NativeQueueCore()
    return _QueueCore()


class _QueueCore:
    """Pure-Python queue-ordering engine (fallback for the native Rust core).

    A transcription of the pre-port ``QueueManager`` internals minus the
    lock (the facade holds it) and the enum (the facade maps codes): the
    same entry-id/shuffle bookkeeping, the same removed-current semantics,
    and the same draws from the global ``random`` module in the same order,
    so seeded sequences match the native implementation exactly.
    """

    def __init__(self) -> None:
        self._tracks: list[dict] = []
        # One internal id per queue occurrence, parallel to ``_tracks``. An
        # id is minted when a track is inserted and dies with that entry, so
        # it identifies "this occurrence" even when the same track — or the
        # very same dict object — sits in the queue twice. Never stored in
        # the track dicts, so it can't leak into session files or API calls.
        self._entry_ids: list[int] = []
        self._next_entry_id: int = 0
        self._generation: int = 0
        # A removed audible occurrence can finish outside the queue. Keep
        # its successor by occurrence, never pretend that successor is playing.
        self._removed_current: bool = False
        self._removed_next_id: int | None = None
        self._current_index: int = -1
        self._repeat: int = 0  # RepeatMode codes: 0 off, 1 all, 2 one
        self._shuffle: bool = False

        # When shuffle is enabled, _shuffle_order maps playback positions
        # to indices in _tracks. _original_order preserves the insertion order.
        self._shuffle_order: list[int] = []
        self._shuffle_position: int = -1

    # -- Helpers ----------------------------------------------------------

    def _removed_next_position(self) -> int:
        """Visible position of the removed occurrence's successor, or end."""
        if self._removed_next_id in self._entry_ids:
            real = self._entry_ids.index(self._removed_next_id)
            return self._shuffle_order.index(real) if self._shuffle else real
        return len(self._tracks)

    def _real_index(self) -> int:
        """Current index into _tracks (resolving shuffle indirection)."""
        if self._shuffle and 0 <= self._shuffle_position < len(self._shuffle_order):
            return self._shuffle_order[self._shuffle_position]
        return self._current_index

    def _new_entry_ids(self, count: int) -> list[int]:
        """Mint *count* fresh entry ids."""
        start = self._next_entry_id + 1
        self._next_entry_id += count
        return list(range(start, start + count))

    def _rebuild_shuffle(self, keep_current: bool = True) -> None:
        """Rebuild the shuffle order, optionally keeping the current track first."""
        indices = list(range(len(self._tracks)))
        current_real = self._real_index()

        if keep_current and not self._removed_current and 0 <= current_real < len(self._tracks):
            indices.remove(current_real)
            random.shuffle(indices)
            self._shuffle_order = [current_real, *indices]
            self._shuffle_position = 0
        else:
            random.shuffle(indices)
            self._shuffle_order = indices
            self._shuffle_position = -1

    # -- Queue manipulation -------------------------------------------------

    def add(self, track: dict, position: int | None = None) -> None:
        """Add a track to the queue. None = append to end."""
        self._add_unlocked(track, position)

    def _add_unlocked(self, track: dict, position: int | None = None) -> None:
        """Add a track (the facade already holds the lock)."""
        (entry_id,) = self._new_entry_ids(1)
        if self._removed_current and self._removed_next_id is None:
            self._removed_next_id = entry_id
        if position is None or position >= len(self._tracks):
            self._tracks.append(track)
            self._entry_ids.append(entry_id)
            new_idx = len(self._tracks) - 1
        else:
            position = max(0, position)
            self._tracks.insert(position, track)
            self._entry_ids.insert(position, entry_id)
            new_idx = position
            # Adjust current index if we inserted before it.
            if not self._shuffle and position <= self._current_index:
                self._current_index += 1

        if self._shuffle:
            # Insert the new track at a random future position in shuffle order.
            insert_at = self._shuffle_position + 1 if self._shuffle_position >= 0 else 0
            # Shift existing shuffle indices that are >= new_idx.
            self._shuffle_order = [(i + 1 if i >= new_idx else i) for i in self._shuffle_order]
            future_pos = (
                random.randint(insert_at + 1, len(self._shuffle_order))
                if insert_at < len(self._shuffle_order)
                else len(self._shuffle_order)
            )
            self._shuffle_order.insert(future_pos, new_idx)

    def add_next(self, track: dict) -> None:
        """Insert a track immediately after the currently playing track."""
        if self._removed_current:
            self._add_next_removed([track])
            return
        if self._shuffle:
            # Insert into _tracks and put it next in shuffle order.
            self._tracks.append(track)
            self._entry_ids.extend(self._new_entry_ids(1))
            new_idx = len(self._tracks) - 1
            insert_pos = self._shuffle_position + 1
            self._shuffle_order.insert(insert_pos, new_idx)
        else:
            insert_pos = self._current_index + 1 if self._current_index >= 0 else 0
            self._add_unlocked(track, position=insert_pos)

    def add_next_multiple(self, tracks: list[dict]) -> None:
        """Insert *tracks* immediately after the current track, preserving order.

        The N tracks become the next N to play, in the given order. Shuffle-aware:
        under shuffle they're spliced into the shuffle order right after the
        current position (so they play next, in order) while appended to
        ``_tracks``; otherwise they're spliced into ``_tracks`` after the current
        index. Duplicates are inserted as-is, mirroring :meth:`add_next`.
        """
        if not tracks:
            return
        if self._removed_current:
            self._add_next_removed(tracks)
            return
        entry_ids = self._new_entry_ids(len(tracks))
        if self._shuffle:
            start_idx = len(self._tracks)
            self._tracks.extend(tracks)
            self._entry_ids.extend(entry_ids)
            new_indices = range(start_idx, start_idx + len(tracks))
            insert_pos = self._shuffle_position + 1
            for offset, new_idx in enumerate(new_indices):
                self._shuffle_order.insert(insert_pos + offset, new_idx)
        else:
            insert_pos = self._current_index + 1 if self._current_index >= 0 else 0
            self._tracks[insert_pos:insert_pos] = tracks
            self._entry_ids[insert_pos:insert_pos] = entry_ids

    def _add_next_removed(self, tracks: list[dict]) -> None:
        """Insert at the removed audible row's boundary."""
        position = self._removed_next_position()
        ids = self._new_entry_ids(len(tracks))
        if self._shuffle:
            start = len(self._tracks)
            self._tracks.extend(tracks)
            self._entry_ids.extend(ids)
            self._shuffle_order[position:position] = range(start, start + len(tracks))
        else:
            self._tracks[position:position] = tracks
            self._entry_ids[position:position] = ids
        self._removed_next_id = ids[0]

    def _add_multiple_unlocked(self, tracks: list[dict]) -> None:
        """Append *tracks* to the queue and update shuffle order.

        On a fresh (empty) queue with shuffle enabled this rebuilds the whole
        shuffle order rather than inserting piecemeal into an empty list.
        """
        was_empty = len(self._tracks) == 0
        start_idx = len(self._tracks)
        self._tracks.extend(tracks)
        self._entry_ids.extend(self._new_entry_ids(len(tracks)))
        if tracks and self._removed_current and self._removed_next_id is None:
            self._removed_next_id = self._entry_ids[start_idx]
        new_indices = list(range(start_idx, start_idx + len(tracks)))

        if self._shuffle:
            if was_empty:
                # Fresh queue after clear() — do a full shuffle build
                # instead of piecemeal insertion into an empty list.
                self._rebuild_shuffle(keep_current=False)
            else:
                # Insert new indices at random future positions.
                insert_after = self._shuffle_position + 1 if self._shuffle_position >= 0 else 0
                random.shuffle(new_indices)
                for new_idx in new_indices:
                    pos = random.randint(insert_after, len(self._shuffle_order))
                    self._shuffle_order.insert(pos, new_idx)

    def add_multiple(self, tracks: list[dict]) -> None:
        """Append multiple tracks to the end of the queue."""
        if not tracks:
            return
        self._add_multiple_unlocked(tracks)

    def filter_not_in_queue(self, tracks: list[dict]) -> list[dict]:
        """Return the tracks whose video_id is not already queued."""
        existing_ids = {t.get("video_id") for t in self._tracks}
        return [t for t in tracks if t.get("video_id") not in existing_ids]

    def remove(self, index: int) -> None:
        """Remove the track at the given index (in visible/playback order)."""
        self._remove_unlocked(index)

    def remove_entry(self, entry_id: int) -> bool:
        """Remove the one occurrence *entry_id* names (see :meth:`entries`).

        Returns False and changes nothing when no entry carries that id any
        more — it was removed, or the queue was rebuilt, after the id was
        read. The caller decides what that means; this never falls back to
        another occurrence of the same track.
        """
        try:
            real_idx = self._entry_ids.index(entry_id)
            index = self._shuffle_order.index(real_idx) if self._shuffle else real_idx
        except ValueError:
            return False
        self._remove_unlocked(index)
        return True

    def _remove_unlocked(self, index: int) -> None:
        """Remove the entry at visible/playback *index*."""
        if not 0 <= index < len(self._tracks):
            return

        order = self._shuffle_order if self._shuffle else list(range(len(self._tracks)))
        real_idx = order[index]
        current_pos = self._shuffle_position if self._shuffle else self._current_index
        if (not self._removed_current and index == current_pos) or (
            self._removed_current and self._entry_ids[real_idx] == self._removed_next_id
        ):
            self._removed_current = True
            self._removed_next_id = (
                self._entry_ids[order[index + 1]] if index + 1 < len(order) else None
            )

        if self._shuffle:
            if index >= len(self._shuffle_order):
                return
            real_idx = self._shuffle_order[index]
            del self._shuffle_order[index]
            # Shift indices that pointed beyond the removed track.
            self._shuffle_order = [(i - 1 if i > real_idx else i) for i in self._shuffle_order]
            del self._tracks[real_idx]
            del self._entry_ids[real_idx]
            if index < self._shuffle_position:
                self._shuffle_position -= 1
            elif index == self._shuffle_position:
                # Current track removed; clamp position (mirrors the
                # non-shuffle branch: current() lands on the next track).
                if self._shuffle_position >= len(self._shuffle_order):
                    self._shuffle_position = len(self._shuffle_order) - 1
        else:
            del self._tracks[index]
            del self._entry_ids[index]
            if index < self._current_index:
                self._current_index -= 1
            elif index == self._current_index:
                # Current track removed; clamp index.
                if self._current_index >= len(self._tracks):
                    self._current_index = len(self._tracks) - 1

    def clear(self) -> None:
        """Remove all tracks from the queue.

        Resets shuffle state so that the next ``add_multiple`` + ``jump_to``
        sequence starts from a clean slate.  The user-visible shuffle
        *preference* (on/off) is preserved — the internal ordering is rebuilt
        automatically when new tracks are added.
        """
        self._generation += 1
        self._removed_current = False
        self._removed_next_id = None
        self._tracks.clear()
        self._entry_ids.clear()
        self._current_index = -1
        self._shuffle_order.clear()
        self._shuffle_position = -1

    def move(self, from_idx: int, to_idx: int) -> None:
        """Move a track from one position to another in the visible order."""
        if from_idx == to_idx:
            return
        if not (0 <= from_idx < len(self._tracks)):
            return
        to_idx = max(0, min(to_idx, len(self._tracks) - 1))

        if self._shuffle:
            # Move within shuffle order.
            if from_idx >= len(self._shuffle_order) or to_idx >= len(self._shuffle_order):
                return
            item = self._shuffle_order.pop(from_idx)
            self._shuffle_order.insert(to_idx, item)
            # Update shuffle_position if it was affected.
            if from_idx == self._shuffle_position:
                self._shuffle_position = to_idx
            elif from_idx < self._shuffle_position <= to_idx:
                self._shuffle_position -= 1
            elif to_idx <= self._shuffle_position < from_idx:
                self._shuffle_position += 1
        else:
            track = self._tracks.pop(from_idx)
            self._tracks.insert(to_idx, track)
            self._entry_ids.insert(to_idx, self._entry_ids.pop(from_idx))
            # Update current_index if it was affected.
            if from_idx == self._current_index:
                self._current_index = to_idx
            elif from_idx < self._current_index <= to_idx:
                self._current_index -= 1
            elif to_idx <= self._current_index < from_idx:
                self._current_index += 1

    # -- Playback navigation ----------------------------------------------

    def current(self) -> dict | None:
        """Return the currently selected track, or None."""
        if self._removed_current:
            return None
        real = self._real_index()
        if 0 <= real < len(self._tracks):
            return self._tracks[real]
        return None

    def next_track(self) -> dict | None:
        """Advance to the next track and return it.

        Respects repeat and shuffle modes.
        Returns None if there is no next track to play.
        """
        if len(self._tracks) == 0:
            return None

        if self._removed_current:
            pos = self._removed_next_position()
            if pos == len(self._tracks):
                if self._repeat != 1:  # RepeatMode.ALL
                    return None
                pos = 0
            self._removed_current = False
            self._removed_next_id = None
            if self._shuffle:
                self._shuffle_position = pos
            else:
                self._current_index = pos
            return self._tracks[self._real_index()]

        if self._repeat == 2:  # RepeatMode.ONE
            real = self._real_index()
            if 0 <= real < len(self._tracks):
                return self._tracks[real]
            return None

        if self._shuffle:
            next_pos = self._shuffle_position + 1
            if next_pos >= len(self._shuffle_order):
                if self._repeat == 1:  # RepeatMode.ALL
                    self._rebuild_shuffle(keep_current=False)
                    self._shuffle_position = 0
                else:
                    return None
            else:
                self._shuffle_position = next_pos
        else:
            next_idx = self._current_index + 1
            if next_idx >= len(self._tracks):
                if self._repeat == 1:  # RepeatMode.ALL
                    self._current_index = 0
                else:
                    return None
            else:
                self._current_index = next_idx

        real = self._real_index()
        if 0 <= real < len(self._tracks):
            return self._tracks[real]
        return None

    def previous_track(self) -> dict | None:
        """Go back to the previous track and return it.

        Respects repeat and shuffle modes.
        Returns None if there is no previous track.
        """
        if len(self._tracks) == 0:
            return None

        if self._removed_current:
            pos = self._removed_next_position() - 1
            if pos < 0:
                if self._repeat != 1:  # RepeatMode.ALL
                    return None
                pos = len(self._tracks) - 1
            self._removed_current = False
            self._removed_next_id = None
            if self._shuffle:
                self._shuffle_position = pos
            else:
                self._current_index = pos
            return self._tracks[self._real_index()]

        if self._repeat == 2:  # RepeatMode.ONE
            real = self._real_index()
            if 0 <= real < len(self._tracks):
                return self._tracks[real]
            return None

        if self._shuffle:
            prev_pos = self._shuffle_position - 1
            if prev_pos < 0:
                if self._repeat == 1:  # RepeatMode.ALL
                    self._shuffle_position = len(self._shuffle_order) - 1
                else:
                    return None
            else:
                self._shuffle_position = prev_pos
        else:
            prev_idx = self._current_index - 1
            if prev_idx < 0:
                if self._repeat == 1:  # RepeatMode.ALL
                    self._current_index = len(self._tracks) - 1
                else:
                    return None
            else:
                self._current_index = prev_idx

        real = self._real_index()
        if 0 <= real < len(self._tracks):
            return self._tracks[real]
        return None

    # -- Repeat / Shuffle ---------------------------------------------------

    def set_repeat(self, code: int) -> None:
        self._repeat = code

    def cycle_repeat(self) -> int:
        """Cycle through repeat modes: OFF -> ALL -> ONE -> OFF."""
        cycle = {0: 1, 1: 2, 2: 0}
        self._repeat = cycle[self._repeat]
        return self._repeat

    def toggle_shuffle(self) -> None:
        """Toggle shuffle mode on or off."""
        # Resolve the playing occurrence under the mode being left. Once
        # the flag flips, _real_index() reads the other mode's pointer,
        # and _current_index is stale after next/previous under shuffle.
        real = self._real_index()
        self._shuffle = not self._shuffle
        if self._shuffle:
            self._rebuild_shuffle(keep_current=True)
        else:
            # Exiting shuffle: the real index becomes the current position.
            self._current_index = real if 0 <= real < len(self._tracks) else -1
            self._shuffle_order.clear()
            self._shuffle_position = -1

    # -- Random / Radio -----------------------------------------------------

    def play_random(self) -> dict | None:
        """Pick and jump to a random track from the queue."""
        if len(self._tracks) == 0:
            return None
        self._removed_current = False
        self._removed_next_id = None
        idx = random.randrange(len(self._tracks))
        if self._shuffle:
            # Find or set position in shuffle order.
            try:
                self._shuffle_position = self._shuffle_order.index(idx)
            except ValueError:
                self._shuffle_position = 0
        else:
            self._current_index = idx
        real = self._real_index()
        if 0 <= real < len(self._tracks):
            return self._tracks[real]
        return None

    def peek_next(self) -> dict | None:
        """Return the next track WITHOUT advancing the position.

        Useful for prefetching stream URLs ahead of time.
        """
        if len(self._tracks) == 0:
            return None

        if self._removed_current:
            pos = self._removed_next_position()
            if pos == len(self._tracks):
                if self._repeat != 1:  # RepeatMode.ALL
                    return None
                pos = 0
            real = self._shuffle_order[pos] if self._shuffle else pos
            return self._tracks[real]

        if self._repeat == 2:  # RepeatMode.ONE
            real = self._real_index()
            if 0 <= real < len(self._tracks):
                return self._tracks[real]
            return None

        if self._shuffle:
            next_pos = self._shuffle_position + 1
            if next_pos >= len(self._shuffle_order):
                # End of the shuffle order. Even under RepeatMode.ALL we
                # can't predict the next reshuffle, so there's nothing to
                # prefetch.
                return None
            real_idx = self._shuffle_order[next_pos]
            if 0 <= real_idx < len(self._tracks):
                return self._tracks[real_idx]
            return None
        else:
            next_idx = self._current_index + 1
            if next_idx >= len(self._tracks):
                if self._repeat == 1:  # RepeatMode.ALL
                    return self._tracks[0] if self._tracks else None
                return None
            return self._tracks[next_idx]

    # -- Jumps ---------------------------------------------------------------

    def jump_to_entry(self, entry_id: int) -> dict | None:
        """Resolve and select an occurrence atomically, including under shuffle."""
        try:
            real = self._entry_ids.index(entry_id)
        except ValueError:
            return None
        self._removed_current = False
        self._removed_next_id = None
        self._current_index = real
        if self._shuffle:
            self._shuffle_position = self._shuffle_order.index(real)
        return self._tracks[real]

    def jump_to(self, index: int) -> dict | None:
        """Jump to a specific index in the visible order and return the track."""
        if not 0 <= index < len(self._tracks):
            return None
        self._removed_current = False
        self._removed_next_id = None
        if self._shuffle:
            if not 0 <= index < len(self._shuffle_order):
                return None
            self._shuffle_position = index
            # Keep _current_index in sync as a safety fallback.
            self._current_index = self._shuffle_order[index]
        else:
            if not 0 <= index < len(self._tracks):
                return None
            self._current_index = index
        real = self._real_index()
        if 0 <= real < len(self._tracks):
            return self._tracks[real]
        return None

    def jump_to_real(self, real_index: int) -> dict | None:
        """Jump to a track by its index in the internal track list.

        Unlike :meth:`jump_to` (which interprets *index* as a position in
        the current playback order — i.e. shuffle order when shuffled),
        this always resolves *real_index* as a position in ``_tracks``.
        """
        if not 0 <= real_index < len(self._tracks):
            return None
        self._removed_current = False
        self._removed_next_id = None
        if self._shuffle:
            try:
                pos = self._shuffle_order.index(real_index)
                self._shuffle_position = pos
            except ValueError:
                # Track not in shuffle order — insert it right after the
                # current position so playback can continue from here.
                insert_pos = self._shuffle_position + 1
                self._shuffle_order.insert(insert_pos, real_index)
                self._shuffle_position = insert_pos
        else:
            self._current_index = real_index
        real = self._real_index()
        if 0 <= real < len(self._tracks):
            return self._tracks[real]
        return None

    # -- State accessors -------------------------------------------------------

    def generation(self) -> int:
        """Lifetime of this queue; ordinary navigation/insertion keeps it."""
        return self._generation

    def current_index(self) -> int:
        """Index of the currently playing track in the visible order."""
        if self._removed_current:
            return -1
        if self._shuffle:
            return self._shuffle_position
        return self._current_index

    def length(self) -> int:
        return len(self._tracks)

    def is_empty(self) -> bool:
        return len(self._tracks) == 0

    def repeat_code(self) -> int:
        return self._repeat

    def shuffle_enabled(self) -> bool:
        return self._shuffle

    def real_index(self) -> int:
        """Current index into _tracks regardless of shuffle mode."""
        if self._removed_current:
            return -1
        return self._real_index()

    def remaining_tracks(self) -> int:
        """Number of tracks remaining after the current position (shuffle-aware)."""
        if self._removed_current:
            return len(self._tracks) - self._removed_next_position()
        if self._shuffle:
            return len(self._shuffle_order) - self._shuffle_position - 1
        return len(self._tracks) - self._current_index - 1

    def tracks(self) -> tuple[dict, ...]:
        """Tracks in the current playback order."""
        if self._shuffle:
            return tuple(self._tracks[i] for i in self._shuffle_order)
        return tuple(self._tracks)

    def entries(self) -> tuple[tuple[int, dict], ...]:
        """``(entry_id, track)`` pairs in the current playback order.

        The entry id names one queue occurrence: it stays with the entry
        through moves and shuffle and disappears when that entry is
        removed. Two occurrences of the same track never share an id. Read
        both halves from this one call — reading ``tracks`` and the ids
        separately could interleave with a mutation on another thread.
        """
        if self._shuffle:
            return tuple((self._entry_ids[i], self._tracks[i]) for i in self._shuffle_order)
        return tuple(zip(self._entry_ids, self._tracks))

    # -- Test seam -----------------------------------------------------------

    def force_shuffle_state(self, order: list[int], position: int) -> None:
        """Overwrite the shuffle order/position wholesale. No validation —
        exists so tests can pin a deterministic playback order on both the
        native and the pure-Python core."""
        self._shuffle = True
        self._shuffle_order = list(order)
        self._shuffle_position = position


class QueueManager:
    """Manages an ordered playback queue with shuffle and repeat.

    Thin locking facade over a ``QueueCore`` (the native Rust engine when
    the extension is built, the pure-Python transcription otherwise): the
    lock, the ``RepeatMode`` mapping and the non-ordering state (collection
    context id, radio seeds) live here; the ordering state machine lives
    in the core.

    Tracks are stored as dicts matching the standardized track format:
        {
            "video_id": str,
            "title": str,
            "artist": str,
            "artists": list[dict],
            "album": str | None,
            "album_id": str | None,
            "duration": int | None,
            "thumbnail_url": str | None,
            "is_video": bool,
        }
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._core: QueueCoreLike = _new_core()
        # Stable ID of the collection (playlist/album/artist) the queue
        # was populated from, or None for ephemeral queues (search,
        # discovery roulette, single-track radio).  Used by the
        # per-collection shuffle memory feature — see ShufflePreferences.
        self._current_context_id: str | None = None

        self.radio_seeds: list[dict] | None = None

    # -- Properties -------------------------------------------------------

    @property
    def generation(self) -> int:
        """Lifetime of this queue; ordinary navigation/insertion keeps it."""
        with self._lock:
            return self._core.generation()

    @property
    def current_index(self) -> int:
        """Index of the currently playing track in the visible order."""
        with self._lock:
            return self._core.current_index()

    @property
    def current_track(self) -> dict | None:
        return self.current()

    @property
    def tracks(self) -> tuple[dict, ...]:
        """Tracks in the current playback order."""
        with self._lock:
            return self._core.tracks()

    @property
    def entries(self) -> tuple[tuple[int, dict], ...]:
        """``(entry_id, track)`` pairs in the current playback order.

        The entry id names one queue occurrence: it stays with the entry
        through moves and shuffle and disappears when that entry is
        removed. Two occurrences of the same track never share an id. Read
        both halves from this one call — reading ``tracks`` and the ids
        separately could interleave with a mutation on another thread.
        """
        with self._lock:
            return self._core.entries()

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._core.is_empty()

    @property
    def length(self) -> int:
        with self._lock:
            return self._core.length()

    @property
    def repeat_mode(self) -> RepeatMode:
        return _REPEAT_MODES[self._core.repeat_code()]

    @property
    def shuffle_enabled(self) -> bool:
        return self._core.shuffle_enabled()

    @property
    def real_index(self) -> int:
        """Current index into the track list regardless of shuffle mode."""
        with self._lock:
            return self._core.real_index()

    @property
    def current_context_id(self) -> str | None:
        """ID of the collection this queue was populated from, or None."""
        with self._lock:
            return self._current_context_id

    def set_context(self, context_id: str | None) -> None:
        """Set (or clear with None) the collection ID backing this queue."""
        with self._lock:
            self._current_context_id = context_id

    @property
    def remaining_tracks(self) -> int:
        """Number of tracks remaining after the current position (shuffle-aware)."""
        with self._lock:
            return self._core.remaining_tracks()

    # -- Queue manipulation ------------------------------------------------

    def add(self, track: dict, position: int | None = None) -> None:
        """Add a track to the queue. None = append to end."""
        with self._lock:
            self._core.add(track, position)

    def add_next(self, track: dict) -> None:
        """Insert a track immediately after the currently playing track."""
        with self._lock:
            self._core.add_next(track)

    def add_next_multiple(self, tracks: list[dict]) -> None:
        """Insert *tracks* immediately after the current track, preserving order."""
        with self._lock:
            self._core.add_next_multiple(tracks)

    def add_multiple(self, tracks: list[dict]) -> None:
        """Append multiple tracks to the end of the queue."""
        with self._lock:
            self._core.add_multiple(tracks)

    def remove(self, index: int) -> None:
        """Remove the track at the given index (in visible/playback order)."""
        with self._lock:
            self._core.remove(index)

    def remove_entry(self, entry_id: int) -> bool:
        """Remove the one occurrence *entry_id* names (see :attr:`entries`).

        Returns False and changes nothing when no entry carries that id any
        more — it was removed, or the queue was rebuilt, after the id was
        read. The caller decides what that means; this never falls back to
        another occurrence of the same track.
        """
        with self._lock:
            return self._core.remove_entry(entry_id)

    def clear(self) -> None:
        """Remove all tracks from the queue.

        Resets shuffle state so that the next ``add_multiple`` + ``jump_to``
        sequence starts from a clean slate.  The user-visible shuffle
        *preference* (on/off) is preserved — the internal ordering is rebuilt
        automatically when new tracks are added.
        """
        with self._lock:
            self._core.clear()
            self.radio_seeds = None

    def move(self, from_idx: int, to_idx: int) -> None:
        """Move a track from one position to another in the visible order."""
        with self._lock:
            self._core.move(from_idx, to_idx)

    # -- Playback navigation -------------------------------------------------

    def current(self) -> dict | None:
        """Return the currently selected track, or None."""
        with self._lock:
            return self._core.current()

    def next_track(self) -> dict | None:
        """Advance to the next track and return it.

        Respects repeat and shuffle modes.
        Returns None if there is no next track to play.
        """
        with self._lock:
            return self._core.next_track()

    def previous_track(self) -> dict | None:
        """Go back to the previous track and return it.

        Respects repeat and shuffle modes.
        Returns None if there is no previous track.
        """
        with self._lock:
            return self._core.previous_track()

    # -- Repeat / Shuffle -----------------------------------------------------

    def set_repeat(self, mode: RepeatMode) -> None:
        with self._lock:
            self._core.set_repeat(_REPEAT_CODES[mode])

    def cycle_repeat(self) -> RepeatMode:
        """Cycle through repeat modes: OFF -> ALL -> ONE -> OFF."""
        with self._lock:
            return _REPEAT_MODES[self._core.cycle_repeat()]

    def toggle_shuffle(self) -> None:
        """Toggle shuffle mode on or off."""
        with self._lock:
            self._core.toggle_shuffle()

    # -- Random / Radio ---------------------------------------------------------

    def play_random(self) -> dict | None:
        """Pick and jump to a random track from the queue."""
        with self._lock:
            return self._core.play_random()

    def set_radio_tracks(self, tracks: list[dict]) -> None:
        """Append radio/autoplay suggestion tracks to the queue.

        Skips tracks already in the queue (by video_id).
        """
        with self._lock:
            new_tracks = self._core.filter_not_in_queue(tracks)
            if new_tracks:
                logger.debug("Adding %d radio tracks to queue", len(new_tracks))
                self._core.add_multiple(new_tracks)

    def peek_next(self) -> dict | None:
        """Return the next track WITHOUT advancing the position.

        Useful for prefetching stream URLs ahead of time.
        """
        with self._lock:
            return self._core.peek_next()

    # -- Jumps -----------------------------------------------------------------

    def jump_to_entry(self, entry_id: int) -> dict | None:
        """Resolve and select an occurrence atomically, including under shuffle."""
        with self._lock:
            return self._core.jump_to_entry(entry_id)

    def jump_to(self, index: int) -> dict | None:
        """Jump to a specific index in the visible order and return the track."""
        with self._lock:
            return self._core.jump_to(index)

    def jump_to_real(self, real_index: int) -> dict | None:
        """Jump to a track by its index in the internal track list.

        Unlike :meth:`jump_to` (which interprets *index* as a position in
        the current playback order — i.e. shuffle order when shuffled),
        this always resolves *real_index* as a position in the track list.
        """
        with self._lock:
            return self._core.jump_to_real(real_index)

    # -- Test seam ---------------------------------------------------------------

    def _force_shuffle_state(self, order: list[int], position: int) -> None:
        """Pin the shuffle order/position wholesale (tests only, no validation).

        Replaces what the pre-port tests did by assigning ``_shuffle_order``
        directly; works identically on the native and fallback cores.
        """
        with self._lock:
            self._core.force_shuffle_state(order, position)
