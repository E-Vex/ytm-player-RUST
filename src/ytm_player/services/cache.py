"""Audio file cache with LRU eviction.

The pure core of the eviction surface is dual-implemented: the LRU
eviction walk — which entries fall out of an over-budget cache, LRU
first, the budget re-checked before each entry — behind
``CacheManager.evict`` is implemented by the Rust extension
(``_ytm_player_native.eviction_plan``) when it is built and by the
pure-Python ``_eviction_plan`` transcription below otherwise; the
manager's method only performs the reads and the writes around it.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

from ytm_player.config.paths import CACHE_DB, SECURE_FILE_MODE, secure_chmod
from ytm_player.config.settings import get_settings
from ytm_player.utils.formatting import VALID_VIDEO_ID

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_index (
    video_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    format TEXT,
    cached_at TEXT DEFAULT (datetime('now')),
    last_accessed TEXT DEFAULT (datetime('now'))
);
"""


class CacheError(OSError):
    """Raised when a cache write operation fails (e.g. disk full)."""


_COMMIT_EVERY = 10  # Commit after this many cache hits instead of every one.


def _eviction_plan(
    total_size: Any,
    max_bytes: Any,
    rows: Any,
) -> list[tuple[Any, Any, Any]]:
    """The entries an over-budget cache evicts — LRU first, until it fits.

    The pure half of ``CacheManager.evict``: the walk over the
    LRU-ordered index rows (oldest access first — the order the
    reference's ``ORDER BY last_accessed ASC`` fetched). Before each
    entry the running total is compared against the budget —
    ``total_size <= max_bytes`` — so an entry is evicted only while
    the total is still over budget, exactly the reference's
    check-then-evict shape: the entry that brings the total exactly to
    the budget is the LAST evicted, and a total already within budget
    answers an empty plan (the manager's own pre-walk gate makes that
    combination unreachable in production — the walk only runs when
    the SUM exceeded the limit). Each evicted row is read through the
    subscript protocol — ``file_path`` first (the unlink's read),
    then ``video_id`` (the append's), then ``file_size`` (the
    subtraction's) — the reference's own read order, and the three
    values pass through into the returned ``(video_id, file_path,
    file_size)`` triple as the objects they are; the running total is
    decremented through the reference's augmented assignment
    (``total_size -= file_size``). The plan is computed before any
    file is touched — the reference unlinked mid-walk, so a read that
    raised would leave its earlier files already gone; with SQLite
    rows (the only production input) no read can fail, and the index
    — the source of truth — is untouched either way. The iteration
    begins before the first budget check, exactly like the reference's
    ``for`` (its first row was fetched before the ``if`` ran), so a
    non-iterable *rows* raises even when the total already fits. The
    result is a fresh list of fresh triples in eviction order; the
    input rows are never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.eviction_plan``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    evicted: list[tuple[Any, Any, Any]] = []
    for row in rows:
        if total_size <= max_bytes:
            break
        file_path = row["file_path"]
        video_id = row["video_id"]
        file_size = row["file_size"]
        evicted.append((video_id, file_path, file_size))
        total_size -= file_size
    return evicted


class CacheManager:
    """Manages a local audio-file cache with LRU eviction."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        db_path: Path = CACHE_DB,
        max_size_mb: int | None = None,
    ) -> None:
        settings = get_settings()
        self._cache_dir = cache_dir or settings.cache_dir
        self._db_path = db_path
        self._max_size_mb = max_size_mb if max_size_mb is not None else settings.cache.max_size_mb
        self._db: aiosqlite.Connection | None = None
        self._hit_count: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Create the cache directory, open the database, and ensure the schema."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        secure_chmod(self._db_path, SECURE_FILE_MODE)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info(
            "Cache initialised: dir=%s  max=%d MB",
            self._cache_dir,
            self._max_size_mb,
        )

    async def close(self) -> None:
        """Flush pending writes and close the database connection."""
        if self._db is not None:
            if self._hit_count > 0:
                await self._db.commit()
                self._hit_count = 0
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def get(self, video_id: str) -> Path | None:
        """Return the cached file path if it exists, else None.

        Updates ``last_accessed`` on hit so the LRU eviction works correctly.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            "SELECT file_path FROM cache_index WHERE video_id = ?",
            (video_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        path = Path(row["file_path"])
        if not path.exists():
            # Stale index entry -- the file was removed externally.
            await self.remove(video_id)
            return None

        await self._db.execute(
            "UPDATE cache_index SET last_accessed = datetime('now') WHERE video_id = ?",
            (video_id,),
        )
        self._hit_count += 1
        if self._hit_count >= _COMMIT_EVERY:
            await self._db.commit()
            self._hit_count = 0
        return path

    async def put(self, video_id: str, data: bytes, format: str) -> Path:
        """Write raw audio *data* into the cache and return its path."""
        if not VALID_VIDEO_ID.match(video_id):
            raise ValueError(f"Invalid video_id: {video_id!r}")
        dest = self._cache_dir / f"{video_id}.{format}"
        try:
            await asyncio.to_thread(dest.write_bytes, data)
            await self._index(video_id, dest, len(data), format)
            await self.evict()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cache write failed for %s: %s", video_id, exc)
            raise CacheError(f"Failed to cache {video_id}: {exc}") from exc
        return dest

    async def put_file(self, video_id: str, source_path: Path, format: str) -> Path:
        """Copy an external file, or index a download already in the cache."""
        if not VALID_VIDEO_ID.match(video_id):
            raise ValueError(f"Invalid video_id: {video_id!r}")
        dest = self._cache_dir / f"{video_id}.{format}"
        try:
            if not dest.exists() or not source_path.samefile(dest):
                await asyncio.to_thread(shutil.copy2, source_path, dest)
            file_size = dest.stat().st_size
            await self._index(video_id, dest, file_size, format)
            await self.evict()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cache write failed for %s: %s", video_id, exc)
            raise CacheError(f"Failed to cache {video_id}: {exc}") from exc
        return dest

    async def has(self, video_id: str) -> bool:
        """Return True if *video_id* is cached."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            "SELECT 1 FROM cache_index WHERE video_id = ?",
            (video_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def remove(self, video_id: str) -> None:
        """Remove a single cached file and its index entry."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            async with self._db.execute(
                "SELECT file_path FROM cache_index WHERE video_id = ?",
                (video_id,),
            ) as cursor:
                row = await cursor.fetchone()

            if row is not None:
                path = Path(row["file_path"])
                path.unlink(missing_ok=True)
                await self._db.execute(
                    "DELETE FROM cache_index WHERE video_id = ?",
                    (video_id,),
                )
                await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cache remove failed for %s: %s", video_id, exc)
            raise CacheError(f"Failed to remove {video_id} from cache: {exc}") from exc

    async def clear(self) -> None:
        """Wipe the entire cache (files and index)."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            async with self._db.execute("SELECT file_path FROM cache_index") as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                Path(row["file_path"]).unlink(missing_ok=True)
            await self._db.execute("DELETE FROM cache_index")
            await self._db.commit()
            logger.info("Cache cleared")
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cache clear failed: %s", exc)
            raise CacheError(f"Failed to clear cache: {exc}") from exc

    # ------------------------------------------------------------------
    # Status & eviction
    # ------------------------------------------------------------------

    async def get_status(self) -> dict:
        """Return a summary of cache usage."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(
            "SELECT COUNT(*) AS file_count, COALESCE(SUM(file_size), 0) AS total_size "
            "FROM cache_index"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None  # COUNT/SUM aggregate query always returns one row
        return {
            "total_size": row["total_size"],
            "file_count": row["file_count"],
            "max_size": self._max_size_mb * 1024 * 1024,
            "cache_dir": str(self._cache_dir),
        }

    async def evict(self) -> None:
        """Remove least-recently-accessed files until total size is within limits.

        The LRU eviction walk — which entries fall out, LRU-first,
        until the running total fits the budget — is resolved by the
        ``_eviction_plan`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise); this method
        performs the reads and the writes around it.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        max_bytes = self._max_size_mb * 1024 * 1024

        async with self._db.execute(
            "SELECT COALESCE(SUM(file_size), 0) AS total FROM cache_index"
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None  # SUM(...) always returns one row
            total = row["total"]

        if total <= max_bytes:
            return

        try:
            # Fetch entries ordered by oldest access first (LRU).
            async with self._db.execute(
                "SELECT video_id, file_path, file_size FROM cache_index ORDER BY last_accessed ASC"
            ) as cursor:
                rows = await cursor.fetchall()

            evicted = _eviction_plan(total, max_bytes, rows)
            for video_id, file_path, file_size in evicted:
                Path(file_path).unlink(missing_ok=True)
                logger.debug("Evicted %s (%d bytes)", video_id, file_size)

            if evicted:
                evict_ids = [video_id for video_id, _, _ in evicted]
                placeholders = ",".join("?" for _ in evict_ids)
                await self._db.execute(
                    f"DELETE FROM cache_index WHERE video_id IN ({placeholders})",
                    evict_ids,
                )
                await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cache eviction failed: %s", exc)
            raise CacheError(f"Failed to evict cache entries: {exc}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _index(
        self,
        video_id: str,
        path: Path,
        file_size: int,
        format: str,
    ) -> None:
        """Insert or replace an entry in the cache index."""
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO cache_index
                    (video_id, file_path, file_size, format)
                VALUES (?, ?, ?, ?)
                """,
                (video_id, str(path), file_size, format),
            )
            await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Cache index write failed for %s: %s", video_id, exc)
            raise CacheError(f"Failed to index {video_id} in cache: {exc}") from exc


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        eviction_plan as _native_eviction_plan,
    )
except ImportError:
    pass
else:
    _eviction_plan = _native_eviction_plan  # type: ignore[assignment]
