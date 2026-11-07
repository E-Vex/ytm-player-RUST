"""Tests for the audio-cache LRU eviction engine.

``_eviction_plan`` is the engine behind ``CacheManager.evict``: the LRU
walk over the index rows (oldest access first — the order the
reference's ``ORDER BY last_accessed ASC`` fetched) that decides which
entries fall out of an over-budget cache. Before each entry the running
total is compared against the budget — ``total_size <= max_bytes`` — so
an entry is evicted only while the total is still over budget: the
entry that brings the total exactly to the budget is the LAST evicted.
Each evicted row is read through the subscript protocol — ``file_path``
first, then ``video_id``, then ``file_size`` (the reference's read
order) — and the running total is decremented through the augmented
assignment ``total_size -= file_size`` (an ``__isub__``-defining total
mutates exactly like it mutated under the reference's ``-=``; a
``None`` size raises the in-place protocol's own ``'-='``-spelled
``TypeError``). The result is the ``(video_id, file_path, file_size)``
triples to unlink, in eviction order; the file unlinking, the index
DELETE and the commit stay in the manager. The pure-Python
transcription lives in ``services/cache.py``; when the Rust extension
is built the same engine runs natively
(``_ytm_player_native.eviction_plan``).

The maintenance engines around it — the schema DDL body
``_cache_schema`` behind ``init``, the hit-counter batched-commit gate
``_hit_commit_plan`` behind ``get``'s ``last_accessed`` UPDATE (the
module's own ``_COMMIT_EVERY`` read at call time, so a patched
constant steers both backends identically), the usage summary
``_status_summary`` behind ``get_status`` (the four-key payload, the
MB → bytes budget through CPython's own multiplication and the
cache-dir ``str()``), the index write's statement body
``_index_insert_statement`` behind ``_index``, the plan-triples →
IN-clause DELETE ``_index_delete_statement`` behind ``evict``'s index
cleanup (its triples unpacked through the reference's own
``(video_id, _, _)`` contract) and the wipe's bare DELETE
``_index_delete_all_statement`` behind ``clear`` — carry the same
dual implementation, native
(``_ytm_player_native.cache_schema`` / ``cache_hit_commit_plan`` /
``cache_status_summary`` / ``cache_index_insert_statement`` /
``cache_index_delete_statement`` / ``cache_index_delete_all_statement``)
when the extension is built.

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case —
including the error shapes, where CPython's own messages must surface
identically — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import ytm_player.services.cache as cache_module
from ytm_player.services.cache import (
    _SCHEMA,
    _cache_schema,
    _eviction_plan,
    _hit_commit_plan,
    _index_delete_all_statement,
    _index_delete_statement,
    _index_insert_statement,
    _status_summary,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class RecordingRow:
    """A row-shaped object recording the order its keys were read."""

    def __init__(self, data: dict, log: list) -> None:
        self._data = data
        self._log = log

    def __getitem__(self, key: str) -> object:
        self._log.append(key)
        return self._data[key]


class RaisingKeyRow:
    """A row whose ``__getitem__`` raises for one key."""

    def __init__(self, data: dict, boom_key: str) -> None:
        self._data = data
        self._boom_key = boom_key

    def __getitem__(self, key: str) -> object:
        if key == self._boom_key:
            raise KeyError(key)
        return self._data[key]


class _Rows:
    """The row shapes, built identically in this process and the child."""

    @staticmethod
    def sql(data: list) -> list[sqlite3.Row]:
        """The production shape: sqlite3.Row lists from an in-memory table."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE cache_index (video_id TEXT, file_path TEXT, file_size INTEGER)")
        conn.executemany(
            "INSERT INTO cache_index VALUES (?, ?, ?)",
            [tuple(row) for row in data],
        )
        return conn.execute("SELECT video_id, file_path, file_size FROM cache_index").fetchall()

    @staticmethod
    def build(spec: dict) -> object:
        kind = spec["t"]
        if kind == "sql":
            return _Rows.sql(spec["data"])
        if kind == "list":
            return spec["items"]
        if kind == "empty":
            return []
        if kind == "gen":
            return iter(spec["items"])
        if kind == "int":
            return 5
        if kind == "raising_getitem":

            class RaisingGetItem:
                def __getitem__(self, key):
                    raise RuntimeError("boom")

            return [RaisingGetItem()]

        raise AssertionError(kind)


class TestEvictionGate:
    """The budget check before each entry — evict only while over budget."""

    def test_under_budget_answers_empty_plan(self):
        rows = _Rows.sql([("v1", "/p1", 5)])
        assert _eviction_plan(10, 20, rows) == []

    def test_at_budget_answers_empty_plan(self):
        rows = _Rows.sql([("v1", "/p1", 5)])
        assert _eviction_plan(20, 20, rows) == []

    def test_under_budget_never_reads_a_row(self):
        # The check runs before each entry — the fetched row is never
        # subscripted when the total already fits.
        rows = [RaisingKeyRow({"file_path": "/p1", "video_id": "v1", "file_size": 5}, "file_path")]
        assert _eviction_plan(5, 10, rows) == []

    def test_iteration_begins_before_the_first_check(self):
        # The reference's `for` fetched the first row before its `if`
        # ran — a non-iterable raises even when the total already fits.
        with pytest.raises(TypeError, match="object is not iterable"):
            _eviction_plan(0, 100, 5)

    def test_raising_iterator_raises_even_under_budget(self):
        class BadIter:
            def __iter__(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _eviction_plan(0, 100, BadIter())


class TestEvictionWalk:
    """The LRU walk — the caller's row order is the eviction order."""

    def test_one_eviction_when_one_entry_suffices(self):
        # 12 - 5 = 7 <= 10: the walk stops, v2 stays.
        rows = _Rows.sql([("v1", "/p1", 5), ("v2", "/p2", 5)])
        assert _eviction_plan(12, 10, rows) == [("v1", "/p1", 5)]

    def test_walk_evicts_until_the_budget_is_met(self):
        # 16 → 11 (still over) → 6: two evictions, LRU order preserved.
        rows = _Rows.sql([("v1", "/p1", 5), ("v2", "/p2", 5), ("v3", "/p3", 5)])
        assert _eviction_plan(16, 10, rows) == [
            ("v1", "/p1", 5),
            ("v2", "/p2", 5),
        ]

    def test_exact_fit_is_the_last_eviction(self):
        # 15 - 5 = 10 == max: the entry that landed exactly on the
        # budget is evicted; the NEXT check breaks.
        rows = _Rows.sql([("v1", "/p1", 5), ("v2", "/p2", 5)])
        assert _eviction_plan(15, 10, rows) == [("v1", "/p1", 5)]

    def test_exact_fit_after_two_evictions(self):
        # 20 - 3 = 17 (over) - 7 = 10 == max: two evictions, exact fit.
        rows = _Rows.sql([("v1", "/p1", 3), ("v2", "/p2", 7)])
        assert _eviction_plan(20, 10, rows) == [("v1", "/p1", 3), ("v2", "/p2", 7)]

    def test_zero_budget_evicts_everything(self):
        rows = _Rows.sql([("v1", "/p1", 5), ("v2", "/p2", 7)])
        assert _eviction_plan(12, 0, rows) == [("v1", "/p1", 5), ("v2", "/p2", 7)]

    def test_negative_budget_evicts_until_nothing_remains(self):
        rows = _Rows.sql([("v1", "/p1", 5)])
        assert _eviction_plan(5, -1, rows) == [("v1", "/p1", 5)]

    def test_zero_size_entry_never_relieves_the_budget(self):
        # total -= 0 leaves the total over budget — the walk evicts on.
        rows = _Rows.sql([("v0", "/p0", 0), ("v1", "/p1", 5)])
        assert _eviction_plan(6, 5, rows) == [("v0", "/p0", 0), ("v1", "/p1", 5)]

    def test_negative_size_entry_raises_the_running_total(self):
        # total -= -3 climbs to 9 — still over budget after v1, but the
        # rows run out.
        rows = _Rows.sql([("vneg", "/pneg", -3), ("v1", "/p1", 5)])
        assert _eviction_plan(6, 5, rows) == [("vneg", "/pneg", -3), ("v1", "/p1", 5)]

    def test_empty_rows_over_budget_answers_empty_plan(self):
        assert _eviction_plan(100, 10, []) == []

    def test_bigint_arithmetic_stays_exact(self):
        huge = 10**30
        # SQLite's INTEGER column tops out at 64 bits — dict rows carry
        # the huge size the index itself never could.
        rows = [
            {"file_path": "/p1", "video_id": "v1", "file_size": huge},
            {"file_path": "/p2", "video_id": "v2", "file_size": 5},
        ]
        # huge - huge = 0 <= 5: one eviction, exact at any magnitude.
        assert _eviction_plan(huge, 5, rows) == [("v1", "/p1", huge)]

    def test_float_arithmetic(self):
        rows = [{"file_path": "/p1", "video_id": "v1", "file_size": 2.5}]
        # 5.0 - 2.5 = 2.5 <= 2.5: one eviction.
        assert _eviction_plan(5.0, 2.5, rows) == [("v1", "/p1", 2.5)]

    def test_values_pass_through_as_the_objects_they_are(self):
        sentinel_id = object()
        sentinel_path = object()
        rows = [
            {
                "file_path": sentinel_path,
                "video_id": sentinel_id,
                "file_size": 5,
            }
        ]
        plan = _eviction_plan(10, 5, rows)
        assert len(plan) == 1
        assert plan[0][0] is sentinel_id  # the very object, never copied
        assert plan[0][1] is sentinel_path

    def test_generator_input_is_consumed(self):
        rows = iter([{"file_path": "/p1", "video_id": "v1", "file_size": 5}])
        assert _eviction_plan(10, 5, rows) == [("v1", "/p1", 5)]

    def test_fresh_list_every_call(self):
        rows = _Rows.sql([("v1", "/p1", 5)])
        first = _eviction_plan(10, 5, rows)
        second = _eviction_plan(10, 5, rows)
        assert first == second
        assert first is not second

    def test_input_rows_never_mutated(self):
        rows = [{"file_path": "/p1", "video_id": "v1", "file_size": 5}]
        snapshot = [dict(r) for r in rows]
        _eviction_plan(10, 5, rows)
        assert rows == snapshot


class TestEvictionReads:
    """The subscript reads — order, laziness and the error shapes."""

    def test_read_order_within_a_row(self):
        log: list[str] = []
        rows = [RecordingRow({"file_path": "/p1", "video_id": "v1", "file_size": 5}, log)]
        assert _eviction_plan(10, 5, rows) == [("v1", "/p1", 5)]
        # The reference's read order: the unlink's file_path, the
        # append's video_id, the subtraction's file_size.
        assert log == ["file_path", "video_id", "file_size"]

    def test_rows_past_the_stop_are_never_read(self):
        log: list[str] = []
        rows = [
            RecordingRow({"file_path": "/p1", "video_id": "v1", "file_size": 5}, log),
            RaisingKeyRow({"file_path": "/p2", "video_id": "v2", "file_size": 5}, "file_path"),
        ]
        # 10 - 5 = 5 <= 5: the walk stops at the second row's check —
        # its subscript never runs.
        assert _eviction_plan(10, 5, rows) == [("v1", "/p1", 5)]
        assert log == ["file_path", "video_id", "file_size"]

    def test_missing_file_size_raises_keyerror(self):
        rows = [{"file_path": "/p1", "video_id": "v1"}]
        with pytest.raises(KeyError, match="file_size"):
            _eviction_plan(10, 5, rows)

    def test_first_read_is_file_path(self):
        rows = [{"video_id": "v1", "file_size": 5}]
        with pytest.raises(KeyError, match="file_path"):
            _eviction_plan(10, 5, rows)

    def test_non_subscriptable_row_raises_typeerror(self):
        with pytest.raises(TypeError, match="not subscriptable"):
            _eviction_plan(10, 5, [5])

    def test_raising_getitem_keeps_its_own_error(self):
        class Boom:
            def __getitem__(self, key):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _eviction_plan(10, 5, [Boom()])


class TestEvictionArithmetic:
    """The ``<=`` gate and the ``total -= file_size`` semantics."""

    def test_incomparable_total_raises_typeerror(self):
        # The check runs before the row is ever read.
        with pytest.raises(TypeError, match="not supported between instances"):
            _eviction_plan("x", 5, [object()])

    def test_incomparable_max_bytes_raises_typeerror(self):
        with pytest.raises(TypeError, match="not supported between instances"):
            _eviction_plan(10, "5", [object()])

    def test_none_size_raises_the_in_place_spelling(self):
        # CPython's augmented assignment formats the failure with the
        # augmented symbol — '-=', not the binary subtraction's '-'.
        rows = [{"file_path": "/p1", "video_id": "v1", "file_size": None}]
        with pytest.raises(
            TypeError, match=r"unsupported operand type\(s\) for -=: 'int' and 'NoneType'"
        ):
            _eviction_plan(10, 5, rows)

    def test_isub_total_mutates_in_place(self):
        class IsubTotal:
            def __init__(self, value: int) -> None:
                self.value = value

            def __le__(self, other: object) -> bool:
                return isinstance(other, int) and self.value <= other

            def __isub__(self, other: object) -> "IsubTotal":
                assert isinstance(other, int)
                self.value -= other
                return self

        total = IsubTotal(10)
        rows = [{"file_path": "/p1", "video_id": "v1", "file_size": 5}]
        assert _eviction_plan(total, 5, rows) == [("v1", "/p1", 5)]
        # The -= mutated the caller's object, exactly like the
        # reference's augmented assignment.
        assert total.value == 5

    def test_instance_attribute_isub_never_runs(self):
        # The augmented assignment reads the type's slot — an
        # instance-attribute __isub__ is invisible to the operator.
        class BinaryTotal:
            def __init__(self, value: int) -> None:
                self.value = value

            def __le__(self, other: object) -> bool:
                return isinstance(other, int) and self.value <= other

            def __sub__(self, other: object) -> "BinaryTotal":
                assert isinstance(other, int)
                self.value -= other
                return self

        total = BinaryTotal(10)

        def _boom(other: object) -> None:
            raise AssertionError("instance __isub__ must never run")

        total.__isub__ = _boom  # type: ignore[method-assign]
        rows = [{"file_path": "/p1", "video_id": "v1", "file_size": 5}]
        assert _eviction_plan(total, 5, rows) == [("v1", "/p1", 5)]
        assert total.value == 5

    def test_rsub_reflection_drives_the_walk(self):
        # int.__sub__(duck) is NotImplemented — the duck's __rsub__ runs.
        class Size:
            def __init__(self, value: int) -> None:
                self.value = value

            def __rsub__(self, other: object) -> int:
                assert isinstance(other, int)
                return other - self.value

        size = Size(5)
        rows = [
            {"file_path": "/p1", "video_id": "v1", "file_size": size},
            {"file_path": "/p2", "video_id": "v2", "file_size": 5},
        ]
        # 10 - 5 = 5 <= 5 at the next check — the reflected result is
        # what stopped the walk.
        plan = _eviction_plan(10, 5, rows)
        assert len(plan) == 1
        assert plan[0][0] == "v1"
        assert plan[0][2] is size  # the duck object passes through


# ── The maintenance engines' contracts ─────────────────────────────


class TestCacheSchema:
    """The DDL body ``init`` hands to executescript."""

    def test_answers_the_modules_schema_text(self):
        assert _cache_schema() == _SCHEMA

    def test_spells_the_single_table_index(self):
        text = _cache_schema()
        assert text.startswith("\nCREATE TABLE IF NOT EXISTS cache_index (")
        assert text.rstrip().endswith(");")
        assert "video_id TEXT PRIMARY KEY" in text
        assert "last_accessed TEXT DEFAULT (datetime('now'))" in text

    def test_the_text_is_stable(self):
        assert _cache_schema() == _cache_schema()


class TestHitCommitPlan:
    """The batched-commit gate — the reference's commit-then-reset."""

    def test_counts_below_the_threshold_accumulate(self):
        assert [_hit_commit_plan(n) for n in range(0, 9)] == [
            (1, False),
            (2, False),
            (3, False),
            (4, False),
            (5, False),
            (6, False),
            (7, False),
            (8, False),
            (9, False),
        ]

    def test_the_count_that_reaches_the_constant_commits_and_resets(self):
        assert _hit_commit_plan(9) == (0, True)
        assert _hit_commit_plan(10) == (0, True)
        assert _hit_commit_plan(100) == (0, True)

    def test_floats_and_bools_steer_through_the_operators(self):
        assert _hit_commit_plan(8.5) == (9.5, False)
        assert _hit_commit_plan(9.5) == (0, True)
        assert _hit_commit_plan(True) == (2, False)
        assert _hit_commit_plan(False) == (1, False)

    def test_a_patched_constant_steers_whichever_backend_is_loaded(self):
        original = cache_module._COMMIT_EVERY
        try:
            cache_module._COMMIT_EVERY = 2
            assert _hit_commit_plan(1) == (0, True)
            assert _hit_commit_plan(0) == (1, False)
        finally:
            cache_module._COMMIT_EVERY = original

    def test_a_non_numeric_count_raises_the_additions_own_error(self):
        with pytest.raises(TypeError):
            _hit_commit_plan("x")


class TestStatusSummary:
    """The four-key usage summary behind ``get_status``."""

    def test_the_payload_in_insertion_order(self):
        # The dir renders through the platform's own str(Path) — a
        # POSIX-literal "/cache" would only hold on POSIX runners.
        summary = _status_summary(500, 3, 2, Path("/cache"))
        assert summary == {
            "total_size": 500,
            "file_count": 3,
            "max_size": 2 * 1024 * 1024,
            "cache_dir": str(Path("/cache")),
        }
        assert list(summary) == ["total_size", "file_count", "max_size", "cache_dir"]

    def test_the_aggregates_pass_through_as_the_objects_they_are(self):
        total = object()
        count = object()
        summary = _status_summary(total, count, 1, "/c")
        assert summary["total_size"] is total
        assert summary["file_count"] is count

    def test_a_str_cache_dir_passes_through_as_the_same_object(self):
        directory = "/already/a/string"
        assert _status_summary(0, 0, 1, directory)["cache_dir"] is directory

    def test_float_and_bool_budgets_steer_through_the_multiplication(self):
        assert _status_summary(0, 0, 1.5, "/c")["max_size"] == 1.5 * 1024 * 1024
        assert _status_summary(0, 0, True, "/c")["max_size"] == 1048576

    def test_a_fresh_dict_every_call(self):
        assert _status_summary(1, 2, 3, "x") is not _status_summary(1, 2, 3, "x")

    def test_an_unmultiplicable_budget_raises_the_operators_own_error(self):
        with pytest.raises(TypeError):
            _status_summary(0, 0, None, "/c")

    def test_a_raising_str_keeps_its_own_error(self):
        class BoomStr:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _status_summary(0, 0, 1, BoomStr())


class TestIndexInsertStatement:
    """The (sql, params) pair ``_index`` binds."""

    def test_the_statement_text_exactly_as_spelled(self):
        sql, _ = _index_insert_statement("v", "/p", 5, "opus")
        assert sql == (
            "\n"
            "        INSERT OR REPLACE INTO cache_index\n"
            "            (video_id, file_path, file_size, format)\n"
            "        VALUES (?, ?, ?, ?)\n"
            "        "
        )

    def test_the_bound_values(self):
        path = Path("/cache/v.opus")
        video_id, file_size, fmt = object(), object(), object()
        sql, params = _index_insert_statement(video_id, path, file_size, fmt)
        assert len(params) == 4
        assert params[0] is video_id
        assert params[1] == str(Path("/cache/v.opus"))
        assert params[2] is file_size
        assert params[3] is fmt

    def test_a_str_path_passes_through_as_the_same_object(self):
        path = "/already/str"
        _, params = _index_insert_statement("v", path, 5, "opus")
        assert params[1] is path

    def test_a_fresh_tuple_every_call(self):
        first = _index_insert_statement("v", "/p", 5, "opus")[1]
        second = _index_insert_statement("v", "/p", 5, "opus")[1]
        assert first is not second

    def test_a_raising_str_keeps_its_own_error(self):
        class BoomStr:
            def __str__(self) -> str:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _index_insert_statement("v", BoomStr(), 5, "opus")

    def test_prepare_and_bind_round_trip(self):
        # The exact (sql, params) pair binds against the real schema'd
        # index — the same table init's executescript builds. The bound
        # file path is the platform's own str(Path) spelling.
        conn = sqlite3.connect(":memory:")
        conn.executescript(_cache_schema())
        sql, params = _index_insert_statement("vid", Path("/c/vid.opus"), 12, "opus")
        conn.execute(sql, params)
        row = conn.execute(
            "SELECT video_id, file_path, file_size, format FROM cache_index"
        ).fetchone()
        assert tuple(row) == ("vid", str(Path("/c/vid.opus")), 12, "opus")
        conn.close()


class TestIndexDeleteStatement:
    """The plan-triples → IN-clause DELETE ``evict`` binds."""

    def test_the_ids_in_plan_order_and_the_placeholders(self):
        sql, ids = _index_delete_statement([("v1", "/p1", 5), ("v2", "/p2", 7)])
        assert sql == "DELETE FROM cache_index WHERE video_id IN (?,?)"
        assert ids == ["v1", "v2"]

    def test_the_ids_pass_through_as_the_objects_they_are(self):
        first, second = object(), object()
        _, ids = _index_delete_statement([(first, "/p1", 5), (second, "/p2", 7)])
        assert ids[0] is first
        assert ids[1] is second

    def test_the_placeholder_count_tracks_the_id_count(self):
        assert _index_delete_statement([("v", "/p", 5)])[0].endswith("IN (?)")
        sql, ids = _index_delete_statement([("v1", "/p1", 5), ("v2", "/p2", 5), ("v3", "/p3", 5)])
        assert sql.endswith("IN (?,?,?)")
        assert ids == ["v1", "v2", "v3"]

    def test_an_empty_plan_spells_the_unreachable_empty_clause(self):
        # The manager's `if evicted:` gate never sends an empty plan,
        # but the engine still mirrors the reference's join.
        assert _index_delete_statement([]) == ("DELETE FROM cache_index WHERE video_id IN ()", [])

    def test_any_three_iterable_works(self):
        sql, ids = _index_delete_statement([["v1", "/p1", 5], ("v2", "/p2", 7)])
        assert ids == ["v1", "v2"]

    def test_a_three_char_string_unpacks_its_characters(self):
        _, ids = _index_delete_statement(["abc"])
        assert ids == ["a"]

    def test_a_generator_plan_is_consumed(self):
        sql, ids = _index_delete_statement(iter([("v1", "/p1", 5)]))
        assert ids == ["v1"]

    def test_a_short_entry_raises_the_unpacks_own_valueerror(self):
        with pytest.raises(ValueError, match=r"not enough values to unpack \(expected 3, got 2\)"):
            _index_delete_statement([("v1", "/p1")])

    def test_a_long_entry_raises_the_unpacks_own_valueerror(self):
        with pytest.raises(ValueError, match="too many values to unpack"):
            _index_delete_statement([("v1", "/p1", 5, "extra")])

    def test_a_non_iterable_entry_raises_the_unpacks_own_typeerror(self):
        with pytest.raises(TypeError, match="cannot unpack non-iterable int object"):
            _index_delete_statement([5])

    def test_a_raising_iter_keeps_its_own_error(self):
        class BadIter:
            def __iter__(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _index_delete_statement([BadIter()])

    def test_prepare_and_bind_round_trip(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(_cache_schema())
        for vid in ("v1", "v2", "v3"):
            conn.execute(
                "INSERT INTO cache_index (video_id, file_path, file_size) VALUES (?, ?, ?)",
                (vid, f"/p/{vid}", 5),
            )
        sql, ids = _index_delete_statement([("v1", "/p/v1", 5), ("v3", "/p/v3", 5)])
        conn.execute(sql, ids)
        remaining = [row[0] for row in conn.execute("SELECT video_id FROM cache_index")]
        assert remaining == ["v2"]
        conn.close()


class TestIndexDeleteAllStatement:
    """The bare DELETE ``clear`` binds."""

    def test_the_text_verbatim(self):
        assert _index_delete_all_statement() == "DELETE FROM cache_index"

    def test_prepare_round_trip(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(_cache_schema())
        conn.execute(
            "INSERT INTO cache_index (video_id, file_path, file_size) VALUES ('v', '/p', 5)"
        )
        conn.execute(_index_delete_all_statement())
        assert conn.execute("SELECT COUNT(*) FROM cache_index").fetchone()[0] == 0
        conn.close()


# ── Native parity ──────────────────────────────────────────────────────
# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable: the row shapes spelled out
# as specs both sides build identically (the "sql" kind reconstructs
# the production shape — an sqlite3.Row list — from an in-memory
# table; integers stay within JSON's exact range, and the duck-typed
# shape zoo lives in the unit tests above).

_EVICTION_PARITY_CASES = [
    # The budget gate: under, at, and over.
    (10, 20, {"t": "sql", "data": [["v1", "/p1", 5]]}),
    (20, 20, {"t": "sql", "data": [["v1", "/p1", 5]]}),
    (12, 10, {"t": "sql", "data": [["v1", "/p1", 5], ["v2", "/p2", 5]]}),
    (
        16,
        10,
        {"t": "sql", "data": [["v1", "/p1", 5], ["v2", "/p2", 5], ["v3", "/p3", 5]]},
    ),
    (15, 10, {"t": "sql", "data": [["v1", "/p1", 5], ["v2", "/p2", 5]]}),
    # The budget corners: zero and negative.
    (12, 0, {"t": "sql", "data": [["v1", "/p1", 5], ["v2", "/p2", 7]]}),
    (5, -1, {"t": "sql", "data": [["v1", "/p1", 5]]}),
    # The size corners: zero (never relieves), negative (raises the
    # total) and NULL (the in-place subtraction's own TypeError).
    (6, 5, {"t": "sql", "data": [["v0", "/p0", 0], ["v1", "/p1", 5]]}),
    (6, 5, {"t": "sql", "data": [["vneg", "/pneg", -3], ["v1", "/p1", 5]]}),
    (10, 5, {"t": "sql", "data": [["v1", "/p1", None]]}),
    # No rows to walk.
    (100, 10, {"t": "empty"}),
    # Big integers within JSON's exact range, and floats.
    (4503599627370496, 5, {"t": "sql", "data": [["v1", "/p1", 4503599627370496]]}),
    (5.0, 2.5, {"t": "sql", "data": [["v1", "/p1", 2.5]]}),
    # The comparison TypeError, from either side.
    ("20", 20, {"t": "sql", "data": [["v1", "/p1", 5]]}),
    (10, "5", {"t": "sql", "data": [["v1", "/p1", 5]]}),
    # Dict rows with non-string ids passing through.
    (10, 5, {"t": "list", "items": [{"file_path": "/p", "video_id": 123, "file_size": 5}]}),
    # A missing key and a non-subscriptable row.
    (10, 5, {"t": "list", "items": [{"file_path": "/p", "video_id": "v1"}]}),
    (10, 5, {"t": "list", "items": [5]}),
    # The iteration protocol: a non-iterable rows object raises even
    # under budget; a generator is consumed.
    (10, 5, {"t": "int"}),
    (0, 100, {"t": "int"}),
    (
        10,
        5,
        {"t": "gen", "items": [{"file_path": "/p", "video_id": "v", "file_size": 5}]},
    ),
    # A row read that raises keeps its own error.
    (10, 5, {"t": "raising_getitem"}),
]


def _outcome(total: object, max_bytes: object, spec: dict) -> dict:
    rows = _Rows.build(spec)
    try:
        plan = _eviction_plan(total, max_bytes, rows)  # type: ignore[operator]
        # Normalise the triples the way the child's JSON wire format
        # already does — tuples become lists.
        return {"ok": [list(triple) for triple in plan]}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


# The child interpreter runs this program with the case matrices on
# the stdin and prints the reference engines' outcomes as JSON.
_CHILD_PROG = """
import json
import pathlib
import sqlite3
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.cache import (
    _cache_schema,
    _eviction_plan,
    _hit_commit_plan,
    _index_delete_all_statement,
    _index_delete_statement,
    _index_insert_statement,
    _status_summary,
)


def build_rows(spec):
    kind = spec["t"]
    if kind == "sql":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            "CREATE TABLE cache_index (video_id TEXT, file_path TEXT, file_size INTEGER)"
        )
        conn.executemany(
            "INSERT INTO cache_index VALUES (?, ?, ?)",
            [tuple(row) for row in spec["data"]],
        )
        return conn.execute(
            "SELECT video_id, file_path, file_size FROM cache_index"
        ).fetchall()
    if kind == "list":
        return spec["items"]
    if kind == "empty":
        return []
    if kind == "gen":
        return iter(spec["items"])
    if kind == "int":
        return 5
    if kind == "raising_getitem":
        class RaisingGetItem:
            def __getitem__(self, key):
                raise RuntimeError("boom")

        return [RaisingGetItem()]
    raise AssertionError(kind)


def build_arg(spec):
    if isinstance(spec, dict) and spec.get("t") == "path":
        return pathlib.Path(spec["v"])
    if isinstance(spec, dict) and spec.get("t") == "gen":
        return iter(spec["items"])
    if isinstance(spec, dict) and spec.get("t") == "boomstr":
        class BoomStr:
            def __str__(self):
                raise RuntimeError("boom")

        return BoomStr()
    if isinstance(spec, dict) and spec.get("t") == "baditer":
        class BadIter:
            def __iter__(self):
                raise RuntimeError("boom")

        return [BadIter()]
    return spec


def outcome(total, max_bytes, spec):
    rows = build_rows(spec)
    try:
        return {"ok": _eviction_plan(total, max_bytes, rows)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


MAINT = {
    "cache_schema": _cache_schema,
    "hit_commit_plan": _hit_commit_plan,
    "status_summary": _status_summary,
    "index_insert_statement": _index_insert_statement,
    "index_delete_statement": _index_delete_statement,
    "index_delete_all_statement": _index_delete_all_statement,
}


def maint_outcome(case):
    args = [build_arg(spec) for spec in case["args"]]
    try:
        result = MAINT[case["engine"]](*args)
        if isinstance(result, tuple):
            return {"ok": [list(part) if isinstance(part, (list, tuple)) else part
                            for part in result]}
        return {"ok": result}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
out = {
    "plans": [
        outcome(total, max_bytes, spec) for total, max_bytes, spec in payload["plans"]
    ]
}
if "maint" in payload:
    out["maint"] = [maint_outcome(case) for case in payload["maint"]]
json.dump(out, sys.stdout)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines over the parity cases in
    a child interpreter with the native extension import blocked."""
    payload = json.dumps({"plans": _EVICTION_PARITY_CASES, "maint": _MAINT_PARITY_CASES})
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _native_outcomes() -> list:
    return [_outcome(total, max_bytes, spec) for total, max_bytes, spec in _EVICTION_PARITY_CASES]


# ── The maintenance engines' parity matrix ──────────────────────────

_MAINT_PARITY_CASES = [
    # ── _cache_schema ──
    {"engine": "cache_schema", "args": []},
    # ── _hit_commit_plan ──
    {"engine": "hit_commit_plan", "args": [0]},
    {"engine": "hit_commit_plan", "args": [1]},
    {"engine": "hit_commit_plan", "args": [8]},
    {"engine": "hit_commit_plan", "args": [9]},
    {"engine": "hit_commit_plan", "args": [10]},
    {"engine": "hit_commit_plan", "args": [100]},
    {"engine": "hit_commit_plan", "args": [-1]},
    {"engine": "hit_commit_plan", "args": [8.5]},
    {"engine": "hit_commit_plan", "args": [9.5]},
    {"engine": "hit_commit_plan", "args": [True]},
    {"engine": "hit_commit_plan", "args": [False]},
    {"engine": "hit_commit_plan", "args": ["x"]},
    # ── _status_summary ──
    {"engine": "status_summary", "args": [500, 3, 2, "/cache"]},
    {"engine": "status_summary", "args": [0, 0, 0, "/c"]},
    {"engine": "status_summary", "args": [1.5, 2, 1.5, "/c"]},
    {"engine": "status_summary", "args": [True, False, True, "/c"]},
    {"engine": "status_summary", "args": [500, 3, 2, {"t": "path", "v": "/p"}]},
    {"engine": "status_summary", "args": [0, 0, None, "/c"]},
    {"engine": "status_summary", "args": [0, 0, 1, {"t": "boomstr"}]},
    # ── _index_insert_statement ──
    {"engine": "index_insert_statement", "args": ["v", "/p", 5, "opus"]},
    {"engine": "index_insert_statement", "args": ["v", {"t": "path", "v": "/c/x"}, None, ""]},
    {"engine": "index_insert_statement", "args": [0, 1, 2, 3]},
    {"engine": "index_insert_statement", "args": ["v", {"t": "boomstr"}, 5, "opus"]},
    # ── _index_delete_statement ──
    {"engine": "index_delete_statement", "args": [[]]},
    {"engine": "index_delete_statement", "args": [[["v1", "/p1", 5]]]},
    {"engine": "index_delete_statement", "args": [[["v1", "/p1", 5], ["v2", "/p2", 7]]]},
    {
        "engine": "index_delete_statement",
        "args": [[[1, "/p1", 5], ["v2", 2, 7], [None, None, None]]],
    },
    {"engine": "index_delete_statement", "args": [[["ab", "/p1"]]]},
    {"engine": "index_delete_statement", "args": [[["abcd", "/p1", 5, "x"]]]},
    {"engine": "index_delete_statement", "args": [[5]]},
    {"engine": "index_delete_statement", "args": ["abc"]},
    {"engine": "index_delete_statement", "args": [{"t": "gen", "items": [["v1", "/p1", 5]]}]},
    {"engine": "index_delete_statement", "args": [{"t": "baditer"}]},
    # ── _index_delete_all_statement ──
    {"engine": "index_delete_all_statement", "args": []},
]


def _maint_outcome(case: dict) -> dict:
    args: list = []
    for spec in case["args"]:
        if isinstance(spec, dict) and spec.get("t") == "path":
            args.append(Path(spec["v"]))
        elif isinstance(spec, dict) and spec.get("t") == "boomstr":

            class BoomStr:
                def __str__(self) -> str:
                    raise RuntimeError("boom")

            args.append(BoomStr())
        elif isinstance(spec, dict) and spec.get("t") == "baditer":

            class BadIter:
                def __iter__(self):
                    raise RuntimeError("boom")

            args.append([BadIter()])
        elif isinstance(spec, dict) and spec.get("t") == "gen":
            args.append(iter(spec["items"]))
        else:
            args.append(spec)
    engine = {
        "cache_schema": _cache_schema,
        "hit_commit_plan": _hit_commit_plan,
        "status_summary": _status_summary,
        "index_insert_statement": _index_insert_statement,
        "index_delete_statement": _index_delete_statement,
        "index_delete_all_statement": _index_delete_all_statement,
    }[case["engine"]]
    try:
        result = engine(*args)
        if isinstance(result, tuple):
            # Normalise the (sql, ids / params) pairs the way the
            # child's JSON wire format already does — tuples become
            # lists.
            return {
                "ok": [list(part) if isinstance(part, (list, tuple)) else part for part in result]
            }
        return {"ok": result}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def _native_maint_outcomes() -> list:
    return [_maint_outcome(case) for case in _MAINT_PARITY_CASES]


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same gate, same walk, same
    triples, same errors with the same messages — for every case, so
    the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_eviction_plan_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()["plans"]
        for case, n, f in zip(_EVICTION_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback eviction outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_maintenance_engines_match_fallback(self):
        native = _native_maint_outcomes()
        fallback = _fallback_outcomes()["maint"]
        for case, n, f in zip(_MAINT_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback maintenance outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
