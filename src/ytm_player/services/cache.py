"""Audio file cache with LRU eviction.

The pure core of the eviction surface is dual-implemented: the LRU
eviction walk — which entries fall out of an over-budget cache, LRU
first, the budget re-checked before each entry — behind
``CacheManager.evict`` is implemented by the Rust extension
(``_ytm_player_native.eviction_plan``) when it is built and by the
pure-Python ``_eviction_plan`` transcription below otherwise; the
manager's method only performs the reads and the writes around it.
The maintenance surface around it is dual-implemented too: the schema
DDL body ``_cache_schema`` behind ``init``, the hit-counter
batched-commit gate ``_hit_commit_plan`` behind ``get``'s
last_accessed UPDATE (the module's own ``_COMMIT_EVERY`` read at
call time, so a patched constant steers both backends identically),
the usage summary ``_status_summary`` behind ``get_status``, the
index write's statement body ``_index_insert_statement`` behind
``_index``, the plan-triples → IN-clause DELETE
``_index_delete_statement`` behind ``evict``'s index cleanup and the
wipe's bare DELETE ``_index_delete_all_statement`` behind
``clear`` — all implemented by the Rust extension
(``_ytm_player_native.cache_schema`` /
``_ytm_player_native.cache_hit_commit_plan`` /
``_ytm_player_native.cache_status_summary`` /
``_ytm_player_native.cache_index_insert_statement`` /
``_ytm_player_native.cache_index_delete_statement`` /
``_ytm_player_native.cache_index_delete_all_statement``) when it is
built and by the pure-Python transcriptions below otherwise; the
connection handling, the row reads, the file unlinking and the
commits stay Python-side in the manager.
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


def _cache_schema() -> str:
    """The schema DDL script ``init`` executes.

    The pure half of ``CacheManager.init``'s schema step: the
    single-table index the module's ``_SCHEMA`` constant spells,
    handed to ``executescript`` verbatim.

    Implemented by the Rust extension
    (``_ytm_player_native.cache_schema``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return _SCHEMA


def _hit_commit_plan(hit_count: Any) -> tuple[Any, bool]:
    """The hit-counter batched-commit gate behind ``CacheManager.get``.

    The pure core of the hit path's tail: the count incremented
    through CPython's own ``+`` and compared against the module's own
    ``_COMMIT_EVERY`` — read at call time, so a patched constant
    steers both backends identically — a count that reaches the
    constant answering ``(0, True)``: the batched commit is due and
    the stored count resets, exactly the reference's
    commit-then-reset; a count still short answering ``(count + 1,
    False)``. The ``last_accessed`` UPDATE around the engine and the
    commit itself stay Python-side in the manager.

    Implemented by the Rust extension
    (``_ytm_player_native.cache_hit_commit_plan``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    hit_count = hit_count + 1
    if hit_count >= _COMMIT_EVERY:
        return (0, True)
    return (hit_count, False)


def _status_summary(
    total_size: Any,
    file_count: Any,
    max_size_mb: Any,
    cache_dir: Any,
) -> dict[str, Any]:
    """The usage summary behind ``CacheManager.get_status``.

    The pure core of the status read: the four-key payload in the
    reference's insertion order — the two aggregates passed through
    as the objects the row reads answered, the MB → bytes budget
    computed through CPython's own ``max_size_mb * 1024 * 1024``, and
    the cache directory rendered through CPython's own ``str()`` —
    into a fresh dict. The row reads and the aggregate assert stay
    Python-side in the manager.

    Implemented by the Rust extension
    (``_ytm_player_native.cache_status_summary``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return {
        "total_size": total_size,
        "file_count": file_count,
        "max_size": max_size_mb * 1024 * 1024,
        "cache_dir": str(cache_dir),
    }


def _index_insert_statement(
    video_id: Any,
    path: Any,
    file_size: Any,
    format: Any,
) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``CacheManager._index``.

    The pure half of the index write: the ``INSERT OR REPLACE``
    statement text — exactly as this transcription spells it,
    leading newline and indentation included — and the four-value
    parameter tuple, ``video_id`` / ``file_size`` / ``format`` passed
    through as the objects they are and the path rendered through
    CPython's own ``str()``. The execute and the commit stay
    Python-side in the manager.

    Implemented by the Rust extension
    (``_ytm_player_native.cache_index_insert_statement``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        INSERT OR REPLACE INTO cache_index
            (video_id, file_path, file_size, format)
        VALUES (?, ?, ?, ?)
        """,
        (video_id, str(path), file_size, format),
    )


def _index_delete_statement(evicted: Any) -> tuple[str, list[Any]]:
    """The statement and bindings behind ``CacheManager.evict``'s cleanup.

    The pure half of the post-unlink index write: the video ids
    extracted from the plan's triples — each entry unpacked through
    the reference's own ``(video_id, _, _)`` contract, so any
    3-iterable works and a wrong-shaped entry raises the unpack's
    own ``ValueError`` — and the placeholder-joined ``IN`` clause,
    packed as the ``(sql, ids)`` pair the reference's execute call
    takes. The ids pass through as the objects they are; the unlink
    loop and the commit stay Python-side in the manager.

    Implemented by the Rust extension
    (``_ytm_player_native.cache_index_delete_statement``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    evict_ids = [video_id for video_id, _, _ in evicted]
    placeholders = ",".join("?" for _ in evict_ids)
    return (f"DELETE FROM cache_index WHERE video_id IN ({placeholders})", evict_ids)


def _index_delete_all_statement() -> str:
    """The statement behind ``CacheManager.clear``'s index wipe.

    The pure half of the wipe's index write: the text handed to the
    reference's execute call, verbatim.

    Implemented by the Rust extension
    (``_ytm_player_native.cache_index_delete_all_statement``) when it
    is built; this pure-Python transcription is the fallback
    otherwise.
    """
    return "DELETE FROM cache_index"


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
        await self._db.executescript(_cache_schema())
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
        # The hit-counter gate — which count to store and whether the
        # batched commit is due — is resolved by the ``_hit_commit_plan``
        # engine (native when the Rust extension is built, the
        # pure-Python transcription otherwise); the module's own
        # ``_COMMIT_EVERY`` constant decides on both backends.
        self._hit_count, commit_due = _hit_commit_plan(self._hit_count)
        if commit_due:
            await self._db.commit()
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
            await self._db.execute(_index_delete_all_statement())
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
        # The summary assembly — the four-key payload, the MB → bytes
        # budget and the cache-dir ``str()`` — is resolved by the
        # ``_status_summary`` engine (native when the Rust extension is
        # built, the pure-Python transcription otherwise); the row
        # reads stay Python-side next to the assert.
        return _status_summary(
            row["total_size"], row["file_count"], self._max_size_mb, self._cache_dir
        )

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
                # The index DELETE's statement body — the video_id
                # extraction from the plan's triples and the
                # placeholder-joined IN clause — is resolved by the
                # ``_index_delete_statement`` engine (native when the
                # Rust extension is built, the pure-Python
                # transcription otherwise); the unlink loop and the
                # commit stay Python-side.
                sql, evict_ids = _index_delete_statement(evicted)
                await self._db.execute(sql, evict_ids)
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
            # The statement body — the INSERT text and the bound
            # ``(video_id, str(path), file_size, format)`` tuple — is
            # resolved by the ``_index_insert_statement`` engine
            # (native when the Rust extension is built, the
            # pure-Python transcription otherwise); the commit stays
            # here.
            sql, params = _index_insert_statement(video_id, path, file_size, format)
            await self._db.execute(sql, params)
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
        cache_hit_commit_plan as _native_hit_commit_plan,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        cache_index_delete_all_statement as _native_index_delete_all_statement,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        cache_index_delete_statement as _native_index_delete_statement,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        cache_index_insert_statement as _native_index_insert_statement,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        cache_schema as _native_cache_schema,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        cache_status_summary as _native_status_summary,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        eviction_plan as _native_eviction_plan,
    )
except ImportError:
    pass
else:
    _eviction_plan = _native_eviction_plan  # type: ignore[assignment]
    _cache_schema = _native_cache_schema  # type: ignore[assignment]
    _hit_commit_plan = _native_hit_commit_plan  # type: ignore[assignment]
    _status_summary = _native_status_summary  # type: ignore[assignment]
    _index_insert_statement = _native_index_insert_statement  # type: ignore[assignment]
    _index_delete_statement = _native_index_delete_statement  # type: ignore[assignment]
    _index_delete_all_statement = _native_index_delete_all_statement  # type: ignore[assignment]
