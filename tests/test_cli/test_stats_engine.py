"""Tests for the CLI stats aggregation engine.

``_cli_stats_payload`` is the engine behind the standalone ``ytm stats``
command: the four-key listening-stats payload assembled from the three
scalar aggregates and the top-tracks list, with the top-tracks rows
rebuilt from their fetched tuples' first four fields. The pure-Python
transcription lives in ``cli.py``; when the Rust extension is built the
same engine runs natively
(``_ytm_player_native.cli_stats_payload``).

The unit tests below assert the engine's contract on whichever backend
is loaded, plus the command glue (the SQLite read, the no-database
zero payload, and the corrupt-database error path). When the extension
is built, the parity test additionally compares the native engine
against the pure-Python reference (run in a child interpreter with the
extension import blocked) case for case — including the error shapes,
where CPython's own messages must surface identically — so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest
from click.testing import CliRunner

from ytm_player.cli import _cli_stats_payload, main

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_PLAY_HISTORY_DDL = """
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
)
"""


class TestCliStatsPayload:
    """The four-key aggregate payload behind ``ytm stats``."""

    def test_payload_carries_the_four_aggregates(self):
        payload = _cli_stats_payload(12, 3450, 7, [("v1", "T1", "A1", 5), ("v2", "T2", "A2", 3)])
        assert payload == {
            "total_plays": 12,
            "total_seconds": 3450,
            "unique_tracks": 7,
            "top_tracks": [
                {"video_id": "v1", "title": "T1", "artist": "A1", "play_count": 5},
                {"video_id": "v2", "title": "T2", "artist": "A2", "play_count": 3},
            ],
        }

    def test_key_order_is_the_json_output_order(self):
        # The key order is the contract the JSON output preserves — it
        # must never silently reorder.
        payload = _cli_stats_payload(0, 0, 0, [])
        assert list(payload) == ["total_plays", "total_seconds", "unique_tracks", "top_tracks"]

    def test_row_key_order_is_the_json_output_order(self):
        payload = _cli_stats_payload(0, 0, 0, [("v", "T", "A", 2)])
        assert list(payload["top_tracks"][0]) == ["video_id", "title", "artist", "play_count"]

    def test_empty_top_tracks(self):
        payload = _cli_stats_payload(0, 0, 0, [])
        assert payload["top_tracks"] == []
        assert type(payload["top_tracks"]) is list

    def test_scalars_pass_through_as_the_objects_they_are(self):
        payload = _cli_stats_payload(10**30, 2.5, True, [])
        assert payload["total_plays"] == 10**30
        assert type(payload["total_plays"]) is int  # bigint, never truncated
        assert type(payload["total_seconds"]) is float
        assert payload["unique_tracks"] is True  # the bool object, never 1

    def test_none_scalar_passes_through(self):
        # COALESCE keeps the SQL aggregates non-NULL today, but the
        # engine passes whatever it is given through untouched.
        payload = _cli_stats_payload(None, None, None, [])
        assert payload == {
            "total_plays": None,
            "total_seconds": None,
            "unique_tracks": None,
            "top_tracks": [],
        }

    def test_row_values_pass_through_as_the_objects_they_are(self):
        rows = [(10**30, "T", None, True)]
        payload = _cli_stats_payload(1, 1, 1, rows)
        item = payload["top_tracks"][0]
        assert item["video_id"] == 10**30
        assert type(item["video_id"]) is int
        assert item["title"] == "T"
        assert item["artist"] is None
        assert item["play_count"] is True  # the bool object, never 1

    def test_longer_rows_extras_are_never_read(self):
        # The comprehension reads exactly r[0]..r[3] — a fifth field is
        # invisible to the payload.
        payload = _cli_stats_payload(1, 1, 1, [("v", "T", "A", 2, "ignored")])
        assert payload["top_tracks"] == [
            {"video_id": "v", "title": "T", "artist": "A", "play_count": 2}
        ]

    def test_list_rows_index_like_tuples(self):
        payload = _cli_stats_payload(1, 1, 1, [["v", "T", "A", 2]])
        assert payload["top_tracks"] == [
            {"video_id": "v", "title": "T", "artist": "A", "play_count": 2}
        ]

    def test_fresh_map_every_call(self):
        args = (12, 3450, 7, [("v", "T", "A", 5)])
        first = _cli_stats_payload(*args)
        second = _cli_stats_payload(*args)
        assert first is not second
        assert first == second
        assert first["top_tracks"] is not second["top_tracks"]
        assert first["top_tracks"][0] is not second["top_tracks"][0]

    def test_inputs_never_mutated(self):
        rows = [("v", "T", "A", 5)]
        _cli_stats_payload(12, 3450, 7, rows)
        assert rows == [("v", "T", "A", 5)]

    def test_short_tuple_row_raises_indexerror(self):
        with pytest.raises(IndexError, match="tuple index out of range"):
            _cli_stats_payload(1, 1, 1, [(1, 2, 3)])

    def test_short_list_row_raises_indexerror(self):
        with pytest.raises(IndexError, match="list index out of range"):
            _cli_stats_payload(1, 1, 1, [[1, 2, 3]])

    def test_dict_row_raises_keyerror(self):
        # r[0] on a dict is the int key 0 — KeyError, the reference
        # comprehension's own error.
        with pytest.raises(KeyError):
            _cli_stats_payload(1, 1, 1, [{"video_id": "v"}])

    def test_non_subscriptable_row_raises_typeerror(self):
        with pytest.raises(TypeError):
            _cli_stats_payload(1, 1, 1, [object()])

    def test_duck_typed_getitem_row(self):
        class GetItemRow:
            def __getitem__(self, index):
                return index * 10

        payload = _cli_stats_payload(1, 1, 1, [GetItemRow()])
        assert payload["top_tracks"] == [
            {"video_id": 0, "title": 10, "artist": 20, "play_count": 30}
        ]

    def test_generator_input_is_consumed(self):
        rows = iter([("v1", "T1", "A1", 1), ("v2", "T2", "A2", 2)])
        payload = _cli_stats_payload(2, 2, 2, rows)
        assert payload["top_tracks"] == [
            {"video_id": "v1", "title": "T1", "artist": "A1", "play_count": 1},
            {"video_id": "v2", "title": "T2", "artist": "A2", "play_count": 2},
        ]

    def test_non_iterable_top_tracks_raises_typeerror(self):
        with pytest.raises(TypeError, match=r"'int' object is not iterable"):
            _cli_stats_payload(1, 1, 1, 5)

    def test_subscript_order_is_left_to_right(self):
        # The dict display evaluates r[0], r[1], r[2], r[3] in order —
        # a row that fails partway must have raised on the FIRST
        # out-of-range index, never a later one.
        class Recorder:
            def __init__(self):
                self.reads = []

            def __getitem__(self, index):
                self.reads.append(index)
                return index

        row = Recorder()
        _cli_stats_payload(1, 1, 1, [row])
        assert row.reads == [0, 1, 2, 3]


# ── Command glue ────────────────────────────────────────────────────────


def _make_db(path, plays: list[tuple[str, str, str, int]]) -> None:
    """Insert *plays* with strictly-distinct played_at stamps so every
    ORDER BY in the command glue is deterministic (no SQLite tie order)."""
    conn = sqlite3.connect(str(path))
    conn.execute(_PLAY_HISTORY_DDL)
    conn.executemany(
        "INSERT INTO play_history (video_id, title, artist, listened_seconds, played_at) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (video_id, title, artist, listened, f"2026-01-{i + 1:02d} 12:00:00")
            for i, (video_id, title, artist, listened) in enumerate(plays)
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def stats_db(tmp_path, monkeypatch):
    """A populated history database the stats command reads."""
    db = tmp_path / "history.db"
    _make_db(
        db,
        [
            ("v1", "T1", "A1", 120),
            ("v1", "T1", "A1", 130),
            ("v1", "T1", "A1", 125),
            ("v2", "Tü", "A2", 60),
            ("v2", "Tü", "A2", 65),
            ("v3", "T3", None, 0),
        ],
    )
    monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
    return db


class TestStatsCommand:
    def test_outputs_the_aggregate_payload(self, stats_db):
        result = CliRunner().invoke(main, ["stats"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == {
            "total_plays": 6,
            "total_seconds": 500,
            "unique_tracks": 3,
            "top_tracks": [
                {"video_id": "v1", "title": "T1", "artist": "A1", "play_count": 3},
                {"video_id": "v2", "title": "Tü", "artist": "A2", "play_count": 2},
                {"video_id": "v3", "title": "T3", "artist": None, "play_count": 1},
            ],
        }

    def test_key_order_is_the_json_output_order(self, stats_db):
        result = CliRunner().invoke(main, ["stats"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert list(payload) == ["total_plays", "total_seconds", "unique_tracks", "top_tracks"]
        assert list(payload["top_tracks"][0]) == ["video_id", "title", "artist", "play_count"]

    def test_compact_json_flag(self, stats_db):
        result = CliRunner().invoke(main, ["stats", "--json"])
        assert result.exit_code == 0, result.output
        assert "\n" not in result.output.strip()

    def test_top_tracks_caps_at_ten(self, tmp_path, monkeypatch):
        db = tmp_path / "history.db"
        _make_db(db, [(f"v{i}", f"T{i}", "A", 10) for i in range(13)])
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        result = CliRunner().invoke(main, ["stats"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["total_plays"] == 13
        assert payload["unique_tracks"] == 13
        assert len(payload["top_tracks"]) == 10  # the query's LIMIT 10

    def test_missing_database_outputs_zero_payload(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["stats"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == {
            "total_plays": 0,
            "total_seconds": 0,
            "unique_tracks": 0,
        }

    def test_corrupt_database_exits_with_error(self, tmp_path, monkeypatch):
        db = tmp_path / "history.db"
        db.write_text("not a database at all")
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        result = CliRunner().invoke(main, ["stats"])
        assert result.exit_code == 1
        assert "Failed to read history database" in result.stderr


class TestQueryHistoryGlue:
    """The row materialisation behind `ytm history` / `ytm history search`."""

    def test_history_outputs_rows_as_dicts(self, stats_db):
        result = CliRunner().invoke(main, ["history", "--limit", "2"])
        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert len(rows) == 2
        # played_at DESC — the two most recent plays, v3 then v2's last.
        assert [row["video_id"] for row in rows] == ["v3", "v2"]
        assert list(rows[0]) == [
            "id",
            "video_id",
            "title",
            "artist",
            "album",
            "duration_seconds",
            "listened_seconds",
            "source",
            "played_at",
        ]

    def test_history_search_outputs_rows_as_dicts(self, tmp_path, monkeypatch):
        db = tmp_path / "history.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                filter_mode TEXT DEFAULT 'music',
                result_count INTEGER,
                search_count INTEGER DEFAULT 1,
                first_searched TEXT DEFAULT (datetime('now')),
                last_searched TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            "INSERT INTO search_history (query, filter_mode, result_count) VALUES (?, ?, ?)",
            ("never gonna", "music", 10),
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        result = CliRunner().invoke(main, ["history", "search"])
        assert result.exit_code == 0, result.output
        rows = json.loads(result.output)
        assert rows == [
            {
                "id": 1,
                "query": "never gonna",
                "filter_mode": "music",
                "result_count": 10,
                "search_count": 1,
                "first_searched": rows[0]["first_searched"],  # the DB's own stamp
                "last_searched": rows[0]["last_searched"],
            }
        ]

    def test_missing_database_outputs_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["history"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == []


# ── Native vs fallback parity ──────────────────────────────────────────

# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable.
_STATS_PARITY_CASES = [
    (12, 3450, 7, [["v1", "T1", "A1", 5], ["v2", "T2", "A2", 3]]),
    (0, 0, 0, []),  # the empty database
    (10**30, 2.5, True, []),  # scalars pass through exactly
    (None, None, None, []),  # a NULL aggregate would pass through raw
    ("12", "3450", "7", []),  # strings
    # Rows as JSON lists index exactly like the fetched tuples.
    (4, 310, 3, [["v1", "T1", "A1", 2], ["v2", "Tü", "A2", 1], ["v3", "T3", None, 1]]),
    (1, 1, 1, [["v", "T", "A", 2, "ignored"]]),  # a fifth field is never read
    (1, 1, 1, [[10**30, "T", None, True]]),  # row values pass through exactly
    (1, 1, 1, [[1, 2, 3]]),  # short row -> IndexError
    (1, 1, 1, [[1, 2, 3, 4, 5, 6]]),  # long row: fields 5-6 never read
    (1, 1, 1, [{"video_id": "v"}]),  # dict row -> KeyError(0)
    (1, 1, 1, ["ab"]),  # str row: reads 'a','b' then IndexError
    (1, 1, 1, 5),  # non-iterable -> TypeError
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _cli_stats_payload


def outcome(total_plays, total_seconds, unique_tracks, top_tracks):
    try:
        return {"ok": _cli_stats_payload(total_plays, total_seconds, unique_tracks, top_tracks)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
json.dump([outcome(*case) for case in payload], sys.stdout)
"""


def _outcome(case: tuple) -> dict:
    try:
        return {"ok": _cli_stats_payload(*case)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same payload, same row
    values (object pass-through included), same errors with the same
    messages — for every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_cli_stats_payload_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_STATS_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_outcome(case) for case in _STATS_PARITY_CASES]
        for case, n, f in zip(_STATS_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback stats outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # Key order is part of the contract (the JSON output preserves
        # it) — compare the serialized payloads too, which pins the
        # order, not just the mapping.
        for n, f in zip(native, fallback):
            if "ok" in n and "ok" in f:
                assert list(n["ok"]) == list(f["ok"]), (
                    f"native and fallback stats key orders differ:\n"
                    f"native:   {list(n['ok'])!r}\n"
                    f"fallback: {list(f['ok'])!r}"
                )
                for n_row, f_row in zip(n["ok"]["top_tracks"], f["ok"]["top_tracks"]):
                    assert list(n_row) == list(f_row)
