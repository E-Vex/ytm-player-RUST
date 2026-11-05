"""Tests for the history SQL-body engines.

Every statement ``HistoryManager`` executes is dual-implemented: the
schema DDL script and the retention prune behind ``init``
(``_history_schema`` / ``_history_retention_delete``), the search
surface's upsert / selects / delete (``_search_upsert`` /
``_search_history_select`` / ``_search_suggestion_select`` /
``_search_history_delete``), the two ``log_play`` writes' statement
texts (``_play_history_insert_sql`` / ``_play_stats_upsert_sql`` —
their bound values are the row-assembly engines tested in
``test_history_engine.py``), the listened-seconds read and its two
updates (``_play_listened_select`` / ``_play_listened_update`` /
``_play_stats_listened_update``), the play-history reads
(``_play_history_select`` / ``_recently_played_select`` /
``_played_video_ids_select``), the ``get_stats`` aggregates
(``_stats_total_plays_select`` / ``_stats_total_listen_time_select`` /
``_stats_unique_tracks_select``) and the two top-list rankings
(``_top_tracks_select`` / ``_top_artists_select``). Each engine
returns the statement text — and the parameter tuple, wherever the
reference binds one — with the bound values passed through as the
objects they are; the connection handling and the fetches stay
Python-side in the manager.

The unit tests below assert each engine's contract on whichever
backend is loaded: the exact statement texts (the multi-line bodies
carry their leading newline and eight-space indentation), the
parameter tuples' contents and object pass-through, and — against a
real schema'd SQLite connection — that every statement prepares and
binds, so a malformed or mis-placeholder'd text can never slip
through either backend. When the extension is built, the parity test
additionally compares the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case, so the two implementations cannot drift
apart.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from ytm_player.services.history import (
    _history_retention_delete,
    _history_schema,
    _play_history_insert_sql,
    _play_history_select,
    _play_listened_select,
    _play_listened_update,
    _play_stats_listened_update,
    _play_stats_upsert_sql,
    _played_video_ids_select,
    _recently_played_select,
    _search_history_delete,
    _search_history_select,
    _search_suggestion_select,
    _search_upsert,
    _stats_total_listen_time_select,
    _stats_total_plays_select,
    _stats_unique_tracks_select,
    _top_artists_select,
    _top_tracks_select,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestHistorySchema:
    """The DDL script behind ``init``'s ``executescript``."""

    def test_script_matches_the_module_constant(self):
        from ytm_player.services.history import _SCHEMA

        assert _history_schema() == _SCHEMA

    def test_script_spells_the_three_tables_and_the_index(self):
        script = _history_schema()
        assert script.startswith("\nCREATE TABLE IF NOT EXISTS search_history (")
        assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_search_query" in script
        assert "CREATE TABLE IF NOT EXISTS play_history (" in script
        assert "CREATE TABLE IF NOT EXISTS play_stats (" in script
        assert script.endswith(");\n")

    def test_script_carries_the_reference_defaults(self):
        script = _history_schema()
        assert "filter_mode TEXT DEFAULT 'music'" in script
        assert "search_count INTEGER DEFAULT 1" in script
        assert "played_at TEXT DEFAULT (datetime('now'))" in script
        assert "play_count INTEGER DEFAULT 0" in script


class TestHistoryRetentionDelete:
    """The retention prune behind ``init``'s post-schema write."""

    def test_statement_and_binding(self):
        sql, params = _history_retention_delete(10000)
        assert sql == (
            "DELETE FROM play_history WHERE rowid NOT IN "
            "(SELECT rowid FROM play_history ORDER BY played_at DESC LIMIT ?)"
        )
        assert params == (10000,)

    def test_binding_passes_the_object_through(self):
        sentinel = object()
        _, params = _history_retention_delete(sentinel)
        assert params == (sentinel,)
        assert params[0] is sentinel


class TestSearchStatements:
    """The four statements behind the search-history surface."""

    def test_upsert_statement_and_bindings(self):
        sql, params = _search_upsert("bohemian rhapsody", "music", 12)
        assert sql == (
            """
        INSERT INTO search_history (query, filter_mode, result_count)
        VALUES (?, ?, ?)
        ON CONFLICT(query, filter_mode) DO UPDATE SET
            search_count = search_count + 1,
            result_count = excluded.result_count,
            last_searched = datetime('now')
        """
        )
        assert params == ("bohemian rhapsody", "music", 12)

    def test_upsert_bindings_pass_the_objects_through(self):
        query, mode, count = object(), object(), object()
        _, params = _search_upsert(query, mode, count)
        assert params == (query, mode, count)
        assert params[0] is query and params[1] is mode and params[2] is count

    def test_history_select_statement_and_binding(self):
        sql, params = _search_history_select(50)
        assert sql == (
            """
        SELECT query, filter_mode, result_count, search_count,
               first_searched, last_searched
        FROM search_history
        ORDER BY last_searched DESC
        LIMIT ?
        """
        )
        assert params == (50,)

    def test_suggestion_select_statement_and_bindings(self):
        sql, params = _search_suggestion_select("qu", 10)
        assert sql == (
            """
        SELECT query
        FROM search_history
        WHERE query LIKE ? || '%'
        ORDER BY search_count DESC, last_searched DESC
        LIMIT ?
        """
        )
        assert params == ("qu", 10)

    def test_delete_statement_takes_no_parameters(self):
        assert _search_history_delete() == "DELETE FROM search_history"
        result = _search_history_delete()
        assert isinstance(result, str)


class TestPlayStatements:
    """The five statements behind the play-history writes."""

    def test_play_history_insert_statement(self):
        assert _play_history_insert_sql() == (
            """
        INSERT INTO play_history
            (video_id, title, artist, album, duration_seconds,
             listened_seconds, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        )

    def test_play_stats_upsert_statement(self):
        assert _play_stats_upsert_sql() == (
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
        """
        )

    def test_listened_select_statement_and_binding(self):
        sql, params = _play_listened_select(7)
        assert sql == "SELECT video_id, listened_seconds FROM play_history WHERE id = ?"
        assert params == (7,)
        sentinel = object()
        assert _play_listened_select(sentinel)[1][0] is sentinel

    def test_listened_update_statement_and_bindings(self):
        sql, params = _play_listened_update(120, 7)
        assert sql == "UPDATE play_history SET listened_seconds = ? WHERE id = ?"
        assert params == (120, 7)

    def test_stats_listened_update_statement_and_bindings(self):
        sql, params = _play_stats_listened_update(30, "dQw4w9WgXcQ")
        assert sql == (
            """
        UPDATE play_stats
        SET total_listened_seconds = total_listened_seconds + ?,
            last_played = datetime('now')
        WHERE video_id = ?
        """
        )
        assert params == (30, "dQw4w9WgXcQ")


class TestReadStatements:
    """The eight statements behind the history reads."""

    def test_play_history_select_statement_and_binding(self):
        sql, params = _play_history_select(100)
        assert sql == (
            """
        SELECT video_id, title, artist, album, duration_seconds,
               listened_seconds, source, played_at
        FROM play_history
        ORDER BY played_at DESC
        LIMIT ?
        """
        )
        assert params == (100,)

    def test_recently_played_select_statement_and_binding(self):
        sql, params = _recently_played_select(50)
        assert sql == (
            """
        SELECT video_id, title, artist, album, duration_seconds,
               MAX(played_at) AS played_at
        FROM play_history
        GROUP BY video_id
        ORDER BY played_at DESC
        LIMIT ?
        """
        )
        assert params == (50,)

    def test_played_video_ids_select_statement(self):
        assert _played_video_ids_select() == "SELECT DISTINCT video_id FROM play_history"

    def test_stats_aggregate_statements(self):
        assert _stats_total_plays_select() == "SELECT COUNT(*) AS total_plays FROM play_history"
        assert (
            _stats_total_listen_time_select()
            == "SELECT COALESCE(SUM(listened_seconds), 0) AS s FROM play_history"
        )
        assert _stats_unique_tracks_select() == "SELECT COUNT(*) AS c FROM play_stats"

    def test_top_tracks_select_statement_and_binding(self):
        sql, params = _top_tracks_select(10)
        assert sql == (
            """
        SELECT video_id, title, artist, play_count,
               total_listened_seconds, last_played
        FROM play_stats
        ORDER BY play_count DESC
        LIMIT ?
        """
        )
        assert params == (10,)

    def test_top_artists_select_statement_and_binding(self):
        sql, params = _top_artists_select(10)
        assert sql == (
            """
        SELECT artist,
               SUM(play_count) AS play_count,
               SUM(total_listened_seconds) AS total_listened_seconds
        FROM play_stats
        WHERE artist IS NOT NULL AND artist != ''
        GROUP BY artist
        ORDER BY play_count DESC
        LIMIT ?
        """
        )
        assert params == (10,)


class TestStatementsPrepareAgainstSqlite:
    """Every engine's statement must prepare and bind against a real
    schema'd SQLite connection — the SQL texts are behavior, not just
    data, and a malformed or mis-placeholder'd text fails here on
    whichever backend is loaded.
    """

    def test_every_statement_prepares_and_binds(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        try:
            conn.executescript(_history_schema())

            conn.execute(*_search_upsert("queen", "music", 5))
            conn.execute(*_search_upsert("queen", "music", 6))  # the conflict arm
            conn.execute(*_search_history_select(10)).fetchall()
            conn.execute(*_search_suggestion_select("qu", 10)).fetchall()
            conn.execute(*_play_listened_select(1)).fetchall()
            conn.execute(
                _play_history_insert_sql(),
                ("dQw4w9WgXcQ", "Never Gonna Give You Up", "Rick Astley", "", 213, 30, "tui"),
            )
            conn.execute(
                _play_stats_upsert_sql(),
                ("dQw4w9WgXcQ", "Never Gonna Give You Up", "Rick Astley", 30),
            )
            conn.execute(
                _play_stats_upsert_sql(),
                ("dQw4w9WgXcQ", "Never Gonna Give You Up", "Rick Astley", 30),
            )
            conn.execute(*_play_listened_update(120, 1))
            conn.execute(*_play_stats_listened_update(90, "dQw4w9WgXcQ"))
            conn.execute(*_play_history_select(10)).fetchall()
            conn.execute(*_recently_played_select(10)).fetchall()
            conn.execute(_played_video_ids_select()).fetchall()
            conn.execute(_stats_total_plays_select()).fetchall()
            conn.execute(_stats_total_listen_time_select()).fetchall()
            conn.execute(_stats_unique_tracks_select()).fetchall()
            conn.execute(*_top_tracks_select(10)).fetchall()
            conn.execute(*_top_artists_select(10)).fetchall()
            conn.execute(*_history_retention_delete(10000))
            conn.commit()

            # The engine-driven round trip behaves like the reference:
            # the upsert counted both searches, the play landed once.
            search_row = conn.execute(*_search_history_select(1)).fetchone()
            assert search_row is not None
            assert search_row["search_count"] == 2
            plays = conn.execute(_stats_total_plays_select()).fetchone()
            assert plays["total_plays"] == 1

            # The clear-all statement empties the search surface.
            conn.execute(_search_history_delete())
            conn.commit()
            assert conn.execute(*_search_history_select(10)).fetchall() == []
        finally:
            conn.close()


# Each parity case: the engine's arguments as a JSON-round-trippable
# list, keyed by the engine's native name (the Python module-level
# fallback spells the same name with a leading underscore).
_PARITY_CASES = {
    "history_schema": [[]],
    "history_retention_delete": [[10000], [0], [1]],
    "search_upsert": [
        ["bohemian rhapsody", "music", 12],
        ["q", "songs", 0],
        ["", "music", None],
    ],
    "search_history_select": [[50], [1], [0]],
    "search_suggestion_select": [["qu", 10], ["", 10]],
    "search_history_delete": [[]],
    "play_history_insert_sql": [[]],
    "play_stats_upsert_sql": [[]],
    "play_listened_select": [[1], [999999]],
    "play_listened_update": [[120, 5], [0, 0]],
    "play_stats_listened_update": [[30, "dQw4w9WgXcQ"], [0, ""]],
    "play_history_select": [[100]],
    "recently_played_select": [[50]],
    "played_video_ids_select": [[]],
    "stats_total_plays_select": [[]],
    "stats_total_listen_time_select": [[]],
    "stats_unique_tracks_select": [[]],
    "top_tracks_select": [[10]],
    "top_artists_select": [[10]],
}

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import history


def outcome(name, args):
    fn = getattr(history, "_" + name)
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, tuple):
        sql, params = result
        return {"ok": [sql, list(params)]}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {name: [outcome(name, args) for args in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""


def _native_outcomes():
    def outcome(name, args):
        fn = getattr(sys.modules["ytm_player.services.history"], "_" + name)
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, tuple):
            sql, params = result
            return {"ok": [sql, list(params)]}
        return {"ok": result}

    return {name: [outcome(name, args) for args in cases] for name, cases in _PARITY_CASES.items()}


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same statement texts,
    same parameter tuples — for every case, so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_statement_engines_match_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes()
        assert set(native) == set(fallback) == set(_PARITY_CASES)
        for name, cases in _PARITY_CASES.items():
            for case, n, f in zip(cases, native[name], fallback[name]):
                assert n == f, (
                    f"native and fallback {name} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
