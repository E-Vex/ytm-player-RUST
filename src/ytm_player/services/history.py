"""Play and search history tracking using async SQLite.

The pure cores of the history surface are dual-implemented. The four
play-history engines: ``_play_log_row`` (the skip gate and insert-row
assembly behind ``log_play``), ``_listened_delta`` (the stored-vs-new
listened-seconds decision behind ``update_play_listened_seconds``),
``_stats_payload`` (the aggregate payload assembly behind
``get_stats``) and ``_row_dicts`` (the row materialisation behind
every history read — the getters' ``fetchall()`` lists and the CLI's
``_query_history`` alike). The row-assembly peripherals: the
``_suggestion_strings`` list build behind ``get_search_suggestions``,
the ``_played_video_id_set`` set build behind ``get_played_video_ids``
and the ``_play_insert_params`` / ``_play_stats_params`` parameter
tuples ``log_play`` binds into its two INSERTs. All are implemented
by the Rust extension
(``_ytm_player_native.play_log_row`` /
``_ytm_player_native.listened_delta`` /
``_ytm_player_native.stats_payload`` /
``_ytm_player_native.row_dicts`` /
``_ytm_player_native.suggestion_strings`` /
``_ytm_player_native.played_video_id_set`` /
``_ytm_player_native.play_insert_params`` /
``_ytm_player_native.play_stats_params``) when it is built and by the
pure-Python transcriptions below otherwise; the manager's methods only
perform the reads and writes around them.

The SQL bodies are dual-implemented too: every statement the manager
executes — the ``_history_schema`` DDL script and the
``_history_retention_delete`` prune behind ``init``, the
``_search_upsert`` / ``_search_history_select`` /
``_search_suggestion_select`` / ``_search_history_delete`` statements
behind the search-history surface, the ``_play_history_insert_sql``
and ``_play_stats_upsert_sql`` statement texts behind ``log_play``'s
two writes, the ``_play_listened_select`` / ``_play_listened_update``
/ ``_play_stats_listened_update`` statements behind
``update_play_listened_seconds``, the ``_play_history_select`` /
``_recently_played_select`` / ``_played_video_ids_select`` reads, the
``_stats_total_plays_select`` / ``_stats_total_listen_time_select`` /
``_stats_unique_tracks_select`` aggregates behind ``get_stats`` and
the ``_top_tracks_select`` / ``_top_artists_select`` rankings — is
implemented by the Rust extension when it is built and by the
pure-Python transcriptions below otherwise. The engines return the
statement text (and the parameter tuple, wherever the reference binds
one); the connection handling, the ``executescript``/``execute``/
``commit`` calls and the fetches stay Python-side in the manager.
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


def _row_dicts(rows: Any) -> list[dict]:
    """The row materialisation every history read returns — rows as dicts.

    The pure half of the five getters (``get_search_history``,
    ``get_play_history``, ``get_recently_played``, ``get_top_tracks``
    and ``get_top_artists``) and of the CLI's ``_query_history``:
    ``[dict(row) for row in rows]``, each fetched row converted
    through the ``dict`` constructor. An ``sqlite3.Row`` (the
    production shape, both backends) reads through its ``keys()``
    mapping protocol into a plain dict; a dict row is copied; anything
    else the constructor accepts works too, and anything it rejects
    raises its own ``TypeError``. The result is a fresh list of fresh
    dicts; the input rows are never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.row_dicts``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return [dict(row) for row in rows]


def _suggestion_strings(rows: Any) -> list[Any]:
    """The autocomplete string list behind ``get_search_suggestions``.

    The pure half of the suggestion read: the
    ``[row["query"] for row in rows]`` comprehension — each fetched row
    read through the subscript protocol (a missing key's ``KeyError``
    propagating), the read values collected as the objects they are
    into a fresh list. The input rows are never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.suggestion_strings``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return [row["query"] for row in rows]


def _played_video_id_set(rows: Any) -> set[Any]:
    """The played-video-id set behind ``get_played_video_ids``.

    The pure half of the Recently-Played filter read: the
    ``{row["video_id"] for row in rows}`` set comprehension — each
    fetched row read through the subscript protocol (a missing key's
    ``KeyError`` propagating), each read value inserted into a fresh
    set through CPython's own set-add: the value hashed first (an
    unhashable value raises the hash protocol's own ``TypeError``),
    then the identity-or-``==`` probe deciding membership, the FIRST
    inserted of equal values the one the set keeps (``True`` and ``1``
    alias one entry — whichever appeared first survives). The input
    rows are never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.played_video_id_set``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return {row["video_id"] for row in rows}


def _play_insert_params(row: Any) -> tuple[Any, ...]:
    """The parameter tuple the play-history INSERT binds.

    The pure half of ``HistoryManager.log_play``'s first write: the
    seven values the ``INSERT INTO play_history`` statement binds,
    read from the engine row through the subscript protocol in the
    reference's left-to-right order — ``video_id``, ``title``,
    ``artist``, ``album``, ``duration``, ``listened_seconds``,
    ``source`` (a missing key's ``KeyError`` propagating from exactly
    the read that missed) — and passed through as the objects they
    are into a fresh tuple. The input row is never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.play_insert_params``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return (
        row["video_id"],
        row["title"],
        row["artist"],
        row["album"],
        row["duration"],
        row["listened_seconds"],
        row["source"],
    )


def _play_stats_params(row: Any) -> tuple[Any, ...]:
    """The parameter tuple the play-stats upsert binds.

    The pure half of ``HistoryManager.log_play``'s second write: the
    four values the ``INSERT INTO play_stats ... ON CONFLICT``
    statement binds — ``video_id``, ``title``, ``artist``,
    ``listened_seconds`` — read through the subscript protocol in the
    reference's left-to-right order and passed through as the objects
    they are into a fresh tuple. The input row is never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.play_stats_params``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return (
        row["video_id"],
        row["title"],
        row["artist"],
        row["listened_seconds"],
    )


def _history_schema() -> str:
    """The schema DDL script ``init`` executes.

        The pure half of ``HistoryManager.init``'s first write: the three
    tables (search history with its unique query index, play history,
    play stats) the module's ``_SCHEMA`` constant spells, handed to
    ``executescript`` verbatim.

        Implemented by the Rust extension
        (``_ytm_player_native.history_schema``) when it is built; this
        pure-Python transcription is the fallback otherwise.
    """
    return _SCHEMA


def _history_retention_delete(max_history: Any) -> tuple[str, tuple[Any, ...]]:
    """The retention prune ``init`` runs after the schema.

    The pure half of ``HistoryManager.init``'s post-schema write: the
    statement that drops play-history rows beyond the retention limit
    (keeping the ``max_history`` most recent by ``played_at``), and
    the bound limit, packed as the ``(sql, params)`` pair the
    reference's ``execute`` call takes.

    Implemented by the Rust extension
    (``_ytm_player_native.history_retention_delete``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (
        "DELETE FROM play_history WHERE rowid NOT IN "
        "(SELECT rowid FROM play_history ORDER BY played_at DESC LIMIT ?)",
        (max_history,),
    )


def _search_upsert(
    query: Any,
    filter_mode: Any,
    result_count: Any,
) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``log_search``'s write.

    The pure half of ``HistoryManager.log_search``: the counter
    incrementing upsert — ``INSERT ... ON CONFLICT(query,
    filter_mode) DO UPDATE SET search_count = search_count + 1, ...``
    — with ``query``, ``filter_mode`` and ``result_count`` passed
    through as the objects they are into the parameter tuple, packed
    as the ``(sql, params)`` pair the reference's ``execute`` call
    takes.

    Implemented by the Rust extension
    (``_ytm_player_native.search_upsert``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return (
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


def _search_history_select(limit: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``get_search_history``'s read.

    The pure half of ``HistoryManager.get_search_history``: the
    recent-searches select — every column, ordered by
    ``last_searched`` descending, bounded by ``limit`` — with the
    limit passed through as the object it is.

    Implemented by the Rust extension
    (``_ytm_player_native.search_history_select``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        SELECT query, filter_mode, result_count, search_count,
               first_searched, last_searched
        FROM search_history
        ORDER BY last_searched DESC
        LIMIT ?
        """,
        (limit,),
    )


def _search_suggestion_select(prefix: Any, limit: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``get_search_suggestions``' read.

    The pure half of ``HistoryManager.get_search_suggestions``: the
    autocomplete select — queries matching the ``prefix`` anchor,
    ranked by search count then recency, bounded by ``limit`` — with
    both values passed through as the objects they are.

    Implemented by the Rust extension
    (``_ytm_player_native.search_suggestion_select``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        SELECT query
        FROM search_history
        WHERE query LIKE ? || '%'
        ORDER BY search_count DESC, last_searched DESC
        LIMIT ?
        """,
        (prefix, limit),
    )


def _search_history_delete() -> str:
    """The statement behind ``clear_search_history``'s write.

    The pure half of ``HistoryManager.clear_search_history``: the
    bare delete-all statement, no parameter tuple — the shape the
    reference's single-argument ``execute`` call takes.

    Implemented by the Rust extension
    (``_ytm_player_native.search_history_delete``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return "DELETE FROM search_history"


def _play_history_insert_sql() -> str:
    """The statement text behind ``log_play``'s first write.

    The pure half of ``HistoryManager.log_play``'s play-history
    INSERT: the seven-placeholder insert whose bound values the
    ``_play_insert_params`` engine assembles, so this engine returns
    the statement text alone — the reference's
    ``execute(sql, params)`` two-argument shape stays the caller's
    to build.

    Implemented by the Rust extension
    (``_ytm_player_native.play_history_insert_sql``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return """
        INSERT INTO play_history
            (video_id, title, artist, album, duration_seconds,
             listened_seconds, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """


def _play_stats_upsert_sql() -> str:
    """The statement text behind ``log_play``'s second write.

    The pure half of ``HistoryManager.log_play``'s play-stats
    upsert: the four-placeholder counter increment whose bound values
    the ``_play_stats_params`` engine assembles, so this engine
    returns the statement text alone.

    Implemented by the Rust extension
    (``_ytm_player_native.play_stats_upsert_sql``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return """
        INSERT INTO play_stats (video_id, title, artist, play_count,
                                total_listened_seconds, last_played)
        VALUES (?, ?, ?, 1, ?, datetime('now'))
        ON CONFLICT(video_id) DO UPDATE SET
            title = excluded.title,
            artist = excluded.artist,
            play_count = play_count + 1,
            total_listened_seconds = total_listened_seconds + excluded.total_listened_seconds,
            last_played = datetime('now')
        """


def _play_listened_select(play_id: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind the listened-seconds read.

    The pure half of ``HistoryManager.update_play_listened_seconds``'s
    row read: the stored ``video_id`` and ``listened_seconds`` lookup
    by row id, with ``play_id`` passed through as the object it is.

    Implemented by the Rust extension
    (``_ytm_player_native.play_listened_select``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (
        "SELECT video_id, listened_seconds FROM play_history WHERE id = ?",
        (play_id,),
    )


def _play_listened_update(listened_seconds: Any, play_id: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind the finalised-listen update.

    The pure half of ``HistoryManager.update_play_listened_seconds``'s
    first write: the row's listened-seconds overwrite, with
    ``listened_seconds`` and ``play_id`` passed through as the objects
    they are, in the statement's placeholder order.

    Implemented by the Rust extension
    (``_ytm_player_native.play_listened_update``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (
        "UPDATE play_history SET listened_seconds = ? WHERE id = ?",
        (listened_seconds, play_id),
    )


def _play_stats_listened_update(delta: Any, video_id: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind the stats-side listen accrual.

    The pure half of ``HistoryManager.update_play_listened_seconds``'s
    second write: the play-stats total accrual and last-played stamp,
    with ``delta`` and ``video_id`` passed through as the objects they
    are, in the statement's placeholder order.

    Implemented by the Rust extension
    (``_ytm_player_native.play_stats_listened_update``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        UPDATE play_stats
        SET total_listened_seconds = total_listened_seconds + ?,
            last_played = datetime('now')
        WHERE video_id = ?
        """,
        (delta, video_id),
    )


def _play_history_select(limit: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``get_play_history``'s read.

    The pure half of ``HistoryManager.get_play_history``: the
    full-history select — every column, ordered by ``played_at``
    descending, bounded by ``limit`` — with the limit passed through
    as the object it is.

    Implemented by the Rust extension
    (``_ytm_player_native.play_history_select``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        SELECT video_id, title, artist, album, duration_seconds,
               listened_seconds, source, played_at
        FROM play_history
        ORDER BY played_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def _recently_played_select(limit: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``get_recently_played``'s read.

    The pure half of ``HistoryManager.get_recently_played``: the
    deduplicated select — the bare columns alongside ``MAX(played_at)``
    (with exactly one min/max aggregate, SQLite documents that the
    bare columns take their values from the max row — each video's
    MOST RECENT play; not portable SQL, but this app is SQLite-only),
    grouped by ``video_id``, bounded by ``limit``.

    Implemented by the Rust extension
    (``_ytm_player_native.recently_played_select``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        SELECT video_id, title, artist, album, duration_seconds,
               MAX(played_at) AS played_at
        FROM play_history
        GROUP BY video_id
        ORDER BY played_at DESC
        LIMIT ?
        """,
        (limit,),
    )


def _played_video_ids_select() -> str:
    """The statement behind ``get_played_video_ids``' read.

    The pure half of ``HistoryManager.get_played_video_ids``: the
    bare DISTINCT video-id walk — the Recently-Played filter's
    feed — no parameter tuple, the shape the reference's
    single-argument ``execute`` call takes.

    Implemented by the Rust extension
    (``_ytm_player_native.played_video_ids_select``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return "SELECT DISTINCT video_id FROM play_history"


def _stats_total_plays_select() -> str:
    """The statement behind ``get_stats``' total-plays aggregate.

    The pure half of ``HistoryManager.get_stats``'s first read: the
    bare play-count aggregate — ``COUNT(*)`` over every logged play —
    no parameter tuple.

    Implemented by the Rust extension
    (``_ytm_player_native.stats_total_plays_select``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return "SELECT COUNT(*) AS total_plays FROM play_history"


def _stats_total_listen_time_select() -> str:
    """The statement behind ``get_stats``' total-listen-time aggregate.

    The pure half of ``HistoryManager.get_stats``'s second read: the
    bare listened-seconds sum with its ``COALESCE`` zero floor, no
    parameter tuple.

    Implemented by the Rust extension
    (``_ytm_player_native.stats_total_listen_time_select``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return "SELECT COALESCE(SUM(listened_seconds), 0) AS s FROM play_history"


def _stats_unique_tracks_select() -> str:
    """The statement behind ``get_stats``' unique-tracks aggregate.

    The pure half of ``HistoryManager.get_stats``'s third read: the
    bare distinct-track count over the stats table, no parameter
    tuple.

    Implemented by the Rust extension
    (``_ytm_player_native.stats_unique_tracks_select``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return "SELECT COUNT(*) AS c FROM play_stats"


def _top_tracks_select(limit: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``get_top_tracks``' read.

    The pure half of ``HistoryManager.get_top_tracks``: the
    most-played ranking — the stats columns ordered by ``play_count``
    descending, bounded by ``limit`` — with the limit passed through
    as the object it is.

    Implemented by the Rust extension
    (``_ytm_player_native.top_tracks_select``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return (
        """
        SELECT video_id, title, artist, play_count,
               total_listened_seconds, last_played
        FROM play_stats
        ORDER BY play_count DESC
        LIMIT ?
        """,
        (limit,),
    )


def _top_artists_select(limit: Any) -> tuple[str, tuple[Any, ...]]:
    """The statement and bindings behind ``get_top_artists``' read.

    The pure half of ``HistoryManager.get_top_artists``: the
    artist-grouped ranking — summed play counts and listen time with
    the NULL/blank artist filter, ordered by summed ``play_count``
    descending, bounded by ``limit`` — with the limit passed through
    as the object it is.

    Implemented by the Rust extension
    (``_ytm_player_native.top_artists_select``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return (
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
    )


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
            await db.executescript(_history_schema())
            await db.commit()

            # Prune old entries beyond retention limit.
            phase = "write"
            await db.execute(*_history_retention_delete(self._max_history))
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
        """Record a search query, incrementing the counter on duplicates.

        The statement and its bindings are resolved by the
        ``_search_upsert`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise); this method
        performs the write.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            await self._db.execute(*_search_upsert(query, filter_mode, result_count))
            await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Failed to log search to history database: %s", exc)
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

    async def get_search_history(self, limit: int = 50) -> list[dict]:
        """Return recent searches ordered by last_searched descending.

        The row materialisation is resolved by the ``_row_dicts``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(*_search_history_select(limit)) as cursor:
            rows = await cursor.fetchall()
            return _row_dicts(rows)

    async def get_search_suggestions(self, prefix: str, limit: int = 10) -> list[str]:
        """Return matching query strings for autocomplete.

        The string-list build is resolved by the ``_suggestion_strings``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(*_search_suggestion_select(prefix, limit)) as cursor:
            rows = await cursor.fetchall()
            return _suggestion_strings(rows)

    async def clear_search_history(self) -> None:
        """Delete all search history records.

        The statement is resolved by the ``_search_history_delete``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the write.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        try:
            await self._db.execute(_search_history_delete())
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
        otherwise), and the two INSERT parameter tuples by the
        ``_play_insert_params`` and ``_play_stats_params`` engines;
        this method performs the writes. The engines run ahead of the
        initialised-DB check, so with a well-formed track an
        uninitialised manager raises exactly the reference's
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
                _play_history_insert_sql(),
                _play_insert_params(row),
            )

            await self._db.execute(_play_stats_upsert_sql(), _play_stats_params(row))

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
            async with self._db.execute(*_play_listened_select(play_id)) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return

            delta = _listened_delta(row["listened_seconds"], listened_seconds)
            if delta is None:
                return

            await self._db.execute(*_play_listened_update(listened_seconds, play_id))
            await self._db.execute(*_play_stats_listened_update(delta, row["video_id"]))
            await self._db.commit()
        except (OSError, sqlite3.Error) as exc:
            logger.exception("Failed to update play history row %r", play_id)
            raise RuntimeError(f"Failed to write to history database: {exc}") from exc

    async def get_play_history(self, limit: int = 100) -> list[dict]:
        """Return play history ordered by most recent first.

        The row materialisation is resolved by the ``_row_dicts``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(*_play_history_select(limit)) as cursor:
            rows = await cursor.fetchall()
            return _row_dicts(rows)

    async def get_recently_played(self, limit: int = 50) -> list[dict]:
        """Return recently-played tracks, deduplicated by video_id.

        The bare columns alongside ``MAX(played_at)`` are deliberate: with
        exactly one min/max aggregate, SQLite documents that bare columns
        take their values from the max row — i.e. each video's MOST RECENT
        play. Not portable SQL, but this app is SQLite-only.

        The row materialisation is resolved by the ``_row_dicts``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(*_recently_played_select(limit)) as cursor:
            rows = await cursor.fetchall()
            return _row_dicts(rows)

    async def get_played_video_ids(self) -> set[str]:
        """Return the set of video_ids ever played locally.

        Used by the Recently Played page to filter TUI plays out of the
        YT Music tab — the account feed mixes genuine online plays with
        plays this app synced there (sync_history_to_ytmusic).

        The video-id set build is resolved by the
        ``_played_video_id_set`` engine (native when the Rust
        extension is built, the pure-Python transcription otherwise);
        this method performs the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(_played_video_ids_select()) as cursor:
            rows = await cursor.fetchall()
            return _played_video_id_set(rows)

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

        async with self._db.execute(_stats_total_plays_select()) as cur:
            row = await cur.fetchone()
            assert row is not None  # COUNT(*) always returns one row
            total_plays = row["total_plays"]

        async with self._db.execute(_stats_total_listen_time_select()) as cur:
            row = await cur.fetchone()
            assert row is not None  # SUM(...) always returns one row
            total_listen_time = row["s"]

        async with self._db.execute(_stats_unique_tracks_select()) as cur:
            row = await cur.fetchone()
            assert row is not None  # COUNT(*) always returns one row
            unique_tracks = row["c"]

        top_tracks = await self.get_top_tracks(limit=10)
        top_artists = await self.get_top_artists(limit=10)

        return _stats_payload(
            total_plays, total_listen_time, unique_tracks, top_tracks, top_artists
        )

    async def get_top_tracks(self, limit: int = 10) -> list[dict]:
        """Return the most-played tracks.

        The row materialisation is resolved by the ``_row_dicts``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(*_top_tracks_select(limit)) as cursor:
            rows = await cursor.fetchall()
            return _row_dicts(rows)

    async def get_top_artists(self, limit: int = 10) -> list[dict]:
        """Return the most-played artists ranked by total play count.

        The row materialisation is resolved by the ``_row_dicts``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs
        the query.
        """
        if self._db is None:
            raise RuntimeError("Database not initialized")
        async with self._db.execute(*_top_artists_select(limit)) as cursor:
            rows = await cursor.fetchall()
            return _row_dicts(rows)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        history_retention_delete as _native_history_retention_delete,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        history_schema as _native_history_schema,
    )
    from ytm_player.utils._ytm_player_native import (
        listened_delta as _native_listened_delta,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_history_insert_sql as _native_play_history_insert_sql,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_history_select as _native_play_history_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_insert_params as _native_play_insert_params,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_listened_select as _native_play_listened_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_listened_update as _native_play_listened_update,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_log_row as _native_play_log_row,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_stats_listened_update as _native_play_stats_listened_update,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_stats_params as _native_play_stats_params,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        play_stats_upsert_sql as _native_play_stats_upsert_sql,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        played_video_id_set as _native_played_video_id_set,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        played_video_ids_select as _native_played_video_ids_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        recently_played_select as _native_recently_played_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        row_dicts as _native_row_dicts,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        search_history_delete as _native_search_history_delete,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        search_history_select as _native_search_history_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        search_suggestion_select as _native_search_suggestion_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        search_upsert as _native_search_upsert,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stats_payload as _native_stats_payload,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stats_total_listen_time_select as _native_stats_total_listen_time_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stats_total_plays_select as _native_stats_total_plays_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stats_unique_tracks_select as _native_stats_unique_tracks_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        suggestion_strings as _native_suggestion_strings,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        top_artists_select as _native_top_artists_select,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        top_tracks_select as _native_top_tracks_select,
    )
except ImportError:
    pass
else:
    _play_log_row = _native_play_log_row  # type: ignore[assignment]
    _listened_delta = _native_listened_delta  # type: ignore[assignment]
    _stats_payload = _native_stats_payload  # type: ignore[assignment]
    _row_dicts = _native_row_dicts  # type: ignore[assignment]
    _suggestion_strings = _native_suggestion_strings  # type: ignore[assignment]
    _played_video_id_set = _native_played_video_id_set  # type: ignore[assignment]
    _play_insert_params = _native_play_insert_params  # type: ignore[assignment]
    _play_stats_params = _native_play_stats_params  # type: ignore[assignment]
    _history_schema = _native_history_schema  # type: ignore[assignment]
    _history_retention_delete = _native_history_retention_delete  # type: ignore[assignment]
    _search_upsert = _native_search_upsert  # type: ignore[assignment]
    _search_history_select = _native_search_history_select  # type: ignore[assignment]
    _search_suggestion_select = _native_search_suggestion_select  # type: ignore[assignment]
    _search_history_delete = _native_search_history_delete  # type: ignore[assignment]
    _play_history_insert_sql = _native_play_history_insert_sql  # type: ignore[assignment]
    _play_stats_upsert_sql = _native_play_stats_upsert_sql  # type: ignore[assignment]
    _play_listened_select = _native_play_listened_select  # type: ignore[assignment]
    _play_listened_update = _native_play_listened_update  # type: ignore[assignment]
    _play_stats_listened_update = _native_play_stats_listened_update  # type: ignore[assignment]
    _play_history_select = _native_play_history_select  # type: ignore[assignment]
    _recently_played_select = _native_recently_played_select  # type: ignore[assignment]
    _played_video_ids_select = _native_played_video_ids_select  # type: ignore[assignment]
    _stats_total_plays_select = _native_stats_total_plays_select  # type: ignore[assignment]
    _stats_total_listen_time_select = _native_stats_total_listen_time_select  # type: ignore[assignment]
    _stats_unique_tracks_select = _native_stats_unique_tracks_select  # type: ignore[assignment]
    _top_tracks_select = _native_top_tracks_select  # type: ignore[assignment]
    _top_artists_select = _native_top_artists_select  # type: ignore[assignment]
