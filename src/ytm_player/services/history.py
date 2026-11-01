"""Play and search history tracking using async SQLite.

The three pure cores of the play-history surface are dual-implemented:
``_play_log_row`` (the skip gate and insert-row assembly behind
``log_play``), ``_listened_delta`` (the stored-vs-new
listened-seconds decision behind ``update_play_listened_seconds``)
and ``_stats_payload`` (the aggregate payload assembly behind
``get_stats``) are implemented by the Rust extension
(``_ytm_player_native.play_log_row`` /
``_ytm_player_native.listened_delta`` /
``_ytm_player_native.stats_payload``) when it is built and by the
pure-Python transcriptions below otherwise; the manager's methods only
perform the reads and writes around them.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

from ytm_player.config.paths import HISTORY_DB, SECURE_FILE_MODE, secure_chmod
from ytm_player.config.settings import DEFAULT_HISTORY_MIN_LISTEN_SECONDS

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    filter_mode TEXT DEFAULT 'music',
    result_count INTEGER,
    search_count INTEGER DEFAULT 1,
    first_searched TEXT DEFAULT (datetime('now')),
    last_searched TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_search_query
    ON search_history(query, filter_mode);

CREATE TABLE IF NOT EXISTS play_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    title TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    duration_seconds INTEGER,
    listened_seconds INTEGER,
    source TEXT,
    played_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS play_stats (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT,
    play_count INTEGER DEFAULT 0,
    total_listened_seconds INTEGER DEFAULT 0,
    last_played TEXT
);
"""


def _play_log_row(
    track: Any,
    listened_seconds: Any,
    source: Any,
    min_listen_seconds: Any,
) -> dict[str, Any] | None:
    """The row a counted play writes — or ``None`` when the listen doesn't count.

    The pure half of ``HistoryManager.log_play``: the skip gate
    (``listened_seconds <= min_listen_seconds`` — skips never reach the
    database, checked before anything else is touched) and the
    insert-row assembly — ``video_id`` through ``[]`` (a missing key
    raises, the reference's one indexing read), ``title`` / ``artist`` /
    ``album`` through duck-typed ``.get`` with the ``""`` defaults, and
    ``duration`` through ``extract_duration`` (the
    ``duration_seconds`` → ``duration`` → ``length`` chain). Every
    value passes through as the object it is — ``listened_seconds``
    and ``source`` included — into a fresh map; the inputs are never
    mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.play_log_row``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if listened_seconds <= min_listen_seconds:
        return None

    from ytm_player.utils.formatting import extract_duration

    return {
        "video_id": track["video_id"],
        "title": track.get("title", ""),
        "artist": track.get("artist", ""),
        "album": track.get("album", ""),
        "duration": extract_duration(track),
        "listened_seconds": listened_seconds,
        "source": source,
    }


def _listened_delta(old_value: Any, listened_seconds: Any) -> Any:
    """The delta a finalised listen writes — or ``None`` when there is nothing to write.

    The pure half of ``HistoryManager.update_play_listened_seconds``:
    the stored value is coerced with ``int(old_value or 0)`` (every
    falsy cell — ``None``, ``0``, ``""``, ``False`` — reads as zero)
    and a listen that does not strictly exceed the stored seconds
    writes nothing. Otherwise the delta is
    ``listened_seconds - old_seconds``, exact at any magnitude. Where
    the coercion would raise (``int("4.5")``'s ``ValueError``, the
    ``TypeError`` for an unconvertible value) the error propagates,
    exactly like the reference's inline arithmetic.

    Implemented by the Rust extension
    (``_ytm_player_native.listened_delta``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    old_seconds = int(old_value or 0)
    if listened_seconds <= old_seconds:
        return None
    return listened_seconds - old_seconds


def _stats_payload(
    total_plays: Any,
    total_listen_time: Any,
    unique_tracks: Any,
    top_tracks: Any,
    top_artists: Any,
) -> dict[str, Any]:
    """The aggregate listening-stats payload ``get_stats`` returns.

    The pure half of ``HistoryManager.get_stats``: the five-key dict
    assembled from the three scalar aggregates (total plays, total
    listened seconds, unique tracks) and the two top lists, in this
    insertion order — the key order the JSON consumers' output
    preserves. Every value passes through as the object it is — the
    scalar counts straight from the SQLite rows, the top lists the
    very list objects the caller fetched — into a fresh map; the
    inputs are never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.stats_payload``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return {
        "total_plays": total_plays,
        "total_listen_time": total_listen_time,
        "unique_tracks": unique_tracks,
        "top_tracks": top_tracks,
        "top_artists": top_artists,
    }


class HistoryManager:
    """Manages play and search history in a local SQLite database."""

    def __init__(self, db_path: Path = HISTORY_DB, max_history: int = 10000) -> None:
        self._db_path = db_path
        self._max_history = max_history
        self._db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Open the database and create tables if they don't exist.

        Raises ``RuntimeError`` when the file can't be opened or prepared.
        On failure or cancellation the connection is closed before raising;
        a close that fails itself is logged, not hidden. This code never
        deletes, renames or replaces the file, but SQLite may already have
        touched its journal/WAL state while opening it.
        """
        db: aiosqlite.Connection | None = None
        phase = "open"
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            db = await aiosqlite.connect(self._db_path)
            secure_chmod(self._db_path, SECURE_FILE_MODE)
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.executescript(_SCHEMA)
            await db.commit()

            # Prune old entries beyond retention limit.
            phase = "write"
            await db.execute(
                "DELETE FROM play_history WHERE rowid NOT IN "
                "(SELECT rowid FROM play_history ORDER BY played_at DESC LIMIT ?)",
                (self._max_history,),
            )
            await db.commit()
        except asyncio.CancelledError:
            await self._discard(db)
            raise
        except (OSError, sqlite3.Error) as exc:
            await self._discard(db)
            if phase == "open":
                logger.warning("Failed to open history database %s: %s", self._db_path, exc)
                raise RuntimeError(f"Failed to open history database: {exc}") from exc
            logger.warning("Failed to prune history database: %s", exc)
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

        self._db = db
        logger.info("History database initialised at %s", self._db_path)

    async def _discard(self, db: aiosqlite.Connection | None) -> None:
        """Best-effort close of a connection whose initialisation failed.

        This code never touches the file itself; a failed close is logged.
        """
        self._db = None
        if db is None:
            return
        try:
            await db.close()
        except Exception:
            logger.exception("Failed to close the history database connection after a failed init")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Search history
    # ------------------------------------------------------------------

    async def log_search(
        self,
        query: str,
        filter_mode: str,
        result_count: int,
    ) -> None:
        """Record a search query, incrementing the counter on duplicates."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            await self._db.execute(
                """
                INSERT INTO search_history (query, filter_mode, result_count)
                VALUES (?, ?, ?)
                ON CONFLICT(query, filter_mode) DO UPDATE SET
                    search_count = search_count + 1,
                    result_count = excluded.result_count,
                    last_searched = datetime('now')
                """,
                (query, filter_mode, result_count),
            )
            await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Failed to log search to history database: %s", exc)
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

    async def get_search_history(self, limit: int = 50) -> list[dict]:
        """Return recent searches ordered by last_searched descending."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            """
            SELECT query, filter_mode, result_count, search_count,
                   first_searched, last_searched
            FROM search_history
            ORDER BY last_searched DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_search_suggestions(self, prefix: str, limit: int = 10) -> list[str]:
        """Return matching query strings for autocomplete."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            """
            SELECT query
            FROM search_history
            WHERE query LIKE ? || '%'
            ORDER BY search_count DESC, last_searched DESC
            LIMIT ?
            """,
            (prefix, limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["query"] for row in rows]

    async def clear_search_history(self) -> None:
        """Delete all search history records."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            await self._db.execute("DELETE FROM search_history")
            await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Failed to clear search history: %s", exc)
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

    # ------------------------------------------------------------------
    # Play history
    # ------------------------------------------------------------------

    async def log_play(
        self,
        track: dict,
        listened_seconds: int,
        source: str,
        *,
        min_listen_seconds: int = DEFAULT_HISTORY_MIN_LISTEN_SECONDS,
    ) -> int | None:
        """Record a track play event.

        Skips are ignored: the play is only logged when *listened_seconds*
        exceeds the minimum threshold. The skip gate and the insert-row
        assembly are resolved by the ``_play_log_row`` engine (native
        when the Rust extension is built, the pure-Python transcription
        otherwise); this method performs the writes. The engine runs
        ahead of the initialised-DB check, so with a well-formed track
        an uninitialised manager raises exactly the reference's
        ``RuntimeError`` — only a junk track on an uninitialised manager
        surfaces its own ``KeyError`` first, a doubly-unreachable
        combination.
        """
        row = _play_log_row(track, listened_seconds, source, min_listen_seconds)
        if row is None:
            return None

        if self._db is None:
            raise RuntimeError("Database not initialized")

        try:
            cursor = await self._db.execute(
                """
                INSERT INTO play_history
                    (video_id, title, artist, album, duration_seconds,
                     listened_seconds, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["video_id"],
                    row["title"],
                    row["artist"],
                    row["album"],
                    row["duration"],
                    row["listened_seconds"],
                    row["source"],
                ),
            )

            await self._db.execute(
                """
                INSERT INTO play_stats (video_id, title, artist, play_count,
                                        total_listened_seconds, last_played)
                VALUES (?, ?, ?, 1, ?, datetime('now'))
                ON CONFLICT(video_id) DO UPDATE SET
                    title = excluded.title,
                    artist = excluded.artist,
                    play_count = play_count + 1,
                    total_listened_seconds = total_listened_seconds + excluded.total_listened_seconds,
                    last_played = datetime('now')
                """,
                (row["video_id"], row["title"], row["artist"], row["listened_seconds"]),
            )

            await self._db.commit()
            return cursor.lastrowid
        except (OSError, sqlite3.Error) as exc:
            logger.exception("Failed to log play (video_id=%r)", row["video_id"])
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

    async def update_play_listened_seconds(self, play_id: int, listened_seconds: int) -> None:
        """Update an already-counted play with its final listen duration.

        The stored-vs-new listened-seconds decision — the ``int(...) or 0``
        coercion, the no-write gate and the delta — is resolved by the
        ``_listened_delta`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise); the row read
        and the writes stay here, exactly as lazy as the reference.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")

        try:
            async with self._db.execute(
                "SELECT video_id, listened_seconds FROM play_history WHERE id = ?",
                (play_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return

            delta = _listened_delta(row["listened_seconds"], listened_seconds)
            if delta is None:
                return

            await self._db.execute(
                "UPDATE play_history SET listened_seconds = ? WHERE id = ?",
                (listened_seconds, play_id),
            )
            await self._db.execute(
                """
                UPDATE play_stats
                SET total_listened_seconds = total_listened_seconds + ?,
                    last_played = datetime('now')
                WHERE video_id = ?
                """,
                (delta, row["video_id"]),
            )
            await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.exception("Failed to update play history row %r", play_id)
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

    async def get_play_history(self, limit: int = 100) -> list[dict]:
        """Return play history ordered by most recent first."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            """
            SELECT video_id, title, artist, album, duration_seconds,
                   listened_seconds, source, played_at
            FROM play_history
            ORDER BY played_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_recently_played(self, limit: int = 50) -> list[dict]:
        """Return recently-played tracks, deduplicated by video_id.

        The bare columns alongside ``MAX(played_at)`` are deliberate: with
        exactly one min/max aggregate, SQLite documents that bare columns
        take their values from the max row — i.e. each video's MOST RECENT
        play. Not portable SQL, but this app is SQLite-only.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            """
            SELECT video_id, title, artist, album, duration_seconds,
                   MAX(played_at) AS played_at
            FROM play_history
            GROUP BY video_id
            ORDER BY played_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_played_video_ids(self) -> set[str]:
        """Return the set of video_ids ever played locally.

        Used by the Recently Played page to filter TUI plays out of the
        YT Music tab — the account feed mixes genuine online plays with
        plays this app synced there (sync_history_to_ytmusic).
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute("SELECT DISTINCT video_id FROM play_history") as cursor:
            rows = await cursor.fetchall()
            return {row["video_id"] for row in rows}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> dict:
        """Return aggregate listening statistics.

        The payload assembly — the five-key aggregate dict — is
        resolved by the ``_stats_payload`` engine (native when the
        Rust extension is built, the pure-Python transcription
        otherwise); this method performs the queries.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")

        async with self._db.execute("SELECT COUNT(*) AS total_plays FROM play_history") as cur:
            row = await cur.fetchone()
            assert row is not None  # COUNT(*) always returns one row
            total_plays = row["total_plays"]

        async with self._db.execute(
            "SELECT COALESCE(SUM(listened_seconds), 0) AS s FROM play_history"
        ) as cur:
            row = await cur.fetchone()
            assert row is not None  # SUM(...) always returns one row
            total_listen_time = row["s"]

        async with self._db.execute("SELECT COUNT(*) AS c FROM play_stats") as cur:
            row = await cur.fetchone()
            assert row is not None  # COUNT(*) always returns one row
            unique_tracks = row["c"]

        top_tracks = await self.get_top_tracks(limit=10)
        top_artists = await self.get_top_artists(limit=10)

        return _stats_payload(
            total_plays, total_listen_time, unique_tracks, top_tracks, top_artists
        )

    async def get_top_tracks(self, limit: int = 10) -> list[dict]:
        """Return the most-played tracks."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            """
            SELECT video_id, title, artist, play_count,
                   total_listened_seconds, last_played
            FROM play_stats
            ORDER BY play_count DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_top_artists(self, limit: int = 10) -> list[dict]:
        """Return the most-played artists ranked by total play count."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            """
            SELECT artist,
                   SUM(play_count) AS play_count,
                   SUM(total_listened_seconds) AS total_listened_seconds
            FROM play_stats
            WHERE artist IS NOT NULL AND artist != ''
            GROUP BY artist
            ORDER BY play_count DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (
        listened_delta as _native_listened_delta,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_log_row as _native_play_log_row,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stats_payload as _native_stats_payload,
    )
except ImportError:
    pass
else:
    _play_log_row = _native_play_log_row  # type: ignore[assignment]
    _listened_delta = _native_listened_delta  # type: ignore[assignment]
    _stats_payload = _native_stats_payload  # type: ignore[assignment]
