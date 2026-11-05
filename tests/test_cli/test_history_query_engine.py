"""Tests for the CLI history-query statement engines.

The two statement bodies the standalone ``ytm history`` and
``ytm history search`` commands run are dual-implemented: the
play-history select (``_cli_history_select``) and the search-history
select (``_cli_search_history_select``), each returning the statement
text with the caller's limit passed through as the object it is into
the parameter tuple — the ``(sql, params)`` pair ``_query_history``'s
``execute`` call takes. The connect/row-factory/execute/fetch glue and
the error exits stay Python-side in ``_query_history``.

The unit tests below assert each engine's contract on whichever
backend is loaded — the exact statement texts, the parameter tuples'
contents and object pass-through, and, against a real schema'd SQLite
connection, that both statements prepare and bind — and, when the
extension is built, the parity test compares the native engines
against the pure-Python references (run in a child interpreter with
the extension import blocked) case for case, so the two
implementations cannot drift apart. The existing CLI suites keep
exercising the commands end-to-end on both backends.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from ytm_player.cli import (
    _cli_history_select,
    _cli_search_history_select,
    _query_history,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestHistorySelect:
    """The play-history select behind ``ytm history``."""

    def test_statement_and_binding(self):
        sql, params = _cli_history_select(50)
        assert sql == "SELECT * FROM play_history ORDER BY played_at DESC LIMIT ?"
        assert params == (50,)

    def test_binding_passes_the_object_through(self):
        sentinel = object()
        _, params = _cli_history_select(sentinel)
        assert params == (sentinel,)
        assert params[0] is sentinel


class TestSearchHistorySelect:
    """The search-history select behind ``ytm history search``."""

    def test_statement_and_binding(self):
        sql, params = _cli_search_history_select(50)
        assert sql == "SELECT * FROM search_history ORDER BY last_searched DESC LIMIT ?"
        assert params == (50,)

    def test_binding_passes_the_object_through(self):
        sentinel = object()
        _, params = _cli_search_history_select(sentinel)
        assert params == (sentinel,)
        assert params[0] is sentinel


class TestPrepareAndBindRoundTrip:
    """Both statements prepare and bind against a real schema."""

    def test_both_statements_execute_and_bind(self):
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(
                """
                CREATE TABLE play_history (
                    id INTEGER PRIMARY KEY, video_id TEXT, title TEXT, artist TEXT,
                    album TEXT, duration_seconds INTEGER, listened_seconds INTEGER,
                    source TEXT, played_at TEXT
                );
                CREATE TABLE search_history (
                    id INTEGER PRIMARY KEY, query TEXT, filter_mode TEXT,
                    result_count INTEGER, search_count INTEGER,
                    first_searched TEXT, last_searched TEXT
                );
                """
            )
            conn.execute(*_cli_history_select(1)).fetchall()
            conn.execute(*_cli_search_history_select(1)).fetchall()
        finally:
            conn.close()


class TestQueryHistoryGlue:
    """``_query_history`` executes the engines' statement pair."""

    def test_missing_database_answers_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", tmp_path / "absent.db")
        assert _query_history(_cli_history_select(5), "history") == []

    def test_rows_come_back_as_dicts(self, tmp_path, monkeypatch):
        db = tmp_path / "history.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE play_history (id INTEGER PRIMARY KEY, video_id TEXT, played_at TEXT)"
        )
        conn.execute("INSERT INTO play_history (video_id, played_at) VALUES ('v1', '2026-01-01')")
        conn.commit()
        conn.close()
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        rows = _query_history(_cli_history_select(10), "history")
        assert rows == [{"id": 1, "video_id": "v1", "played_at": "2026-01-01"}]

    def test_sqlite_failure_exits_with_the_label(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "history.db"
        db.write_text("not a database", encoding="utf-8")
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        with pytest.raises(SystemExit) as excinfo:
            _query_history(_cli_search_history_select(10), "search history")
        assert excinfo.value.code == 1
        assert "Failed to read search history database" in capsys.readouterr().err


# Each parity case: the engine's arguments as JSON-round-trippable
# lists, keyed by the engine's native name (the Python module-level
# fallback spells the same name with a leading underscore).
_PARITY_CASES = {
    "cli_history_select": [[50], [1], [0], [-3], [10**30], [None], ["50"], [1.5]],
    "cli_search_history_select": [[50], [1], [0], [-3], [None], ["50"], [1.5]],
}

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player import cli


def outcome(name, args):
    fn = getattr(cli, "_" + name)
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    sql, params = result
    return {"ok": [sql, list(params)]}


payload = json.load(sys.stdin)
out = {name: [outcome(name, args) for args in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""


def _native_outcomes():
    def outcome(name, args):
        fn = getattr(sys.modules["ytm_player.cli"], "_" + name)
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        sql, params = result
        return {"ok": [sql, list(params)]}

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
