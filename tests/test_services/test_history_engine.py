"""Tests for the play-history row and stats engines.

``_play_log_row`` is the engine behind ``HistoryManager.log_play``: the
skip gate (``listened_seconds <= min_listen_seconds`` — a listen that
does not strictly exceed the threshold never reaches the database) and
the insert-row assembly (``video_id`` through ``[]``, the ``""``-default
``.get`` reads, and the ``duration_seconds`` → ``duration`` → ``length``
duration chain). ``_listened_delta`` is the engine behind
``HistoryManager.update_play_listened_seconds``: the ``int(old or 0)``
coercion of the stored cell, the no-write gate and the delta.
``_stats_payload`` is the engine behind ``HistoryManager.get_stats``:
the five-key aggregate dict assembled from the three scalar aggregates
and the two top lists, in the JSON consumers' key order.
``_row_dicts`` is the engine behind every history read — the five
getters' and the CLI's ``_query_history``'s ``[dict(row) for row in
rows]`` materialisation, each fetched row converted through the
``dict`` constructor. The pure-Python transcriptions live in
``services/history.py``; when the Rust extension is built the same
engines run natively (``_ytm_player_native.play_log_row`` /
``listened_delta`` / ``stats_payload`` / ``row_dicts``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case —
including the error shapes, where CPython's own messages must surface
identically — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from ytm_player.services.history import (
    _listened_delta,
    _play_insert_params,
    _play_log_row,
    _play_stats_params,
    _played_video_id_set,
    _row_dicts,
    _stats_payload,
    _suggestion_strings,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class DuckGet:
    """A track-shaped object steering the ``[]`` and ``.get`` reads
    through Python method dispatch."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str) -> object:
        return self._data[key]

    def get(self, key: str, default: object = None) -> object:
        return self._data[key] if key in self._data else default


class GetItemOnly:
    """Subscriptable but ``.get``-less — reaches the AttributeError read."""

    def __getitem__(self, key: str) -> str:
        return "VID"


class ReadLogRow:
    """A row-shaped object recording the order its keys were read."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self.reads: list[str] = []

    def __getitem__(self, key: str) -> object:
        self.reads.append(key)
        return self._data[key]


def _sql_rows(columns: list[str], data: list[tuple]) -> list[sqlite3.Row]:
    """The production row shape: sqlite3.Row lists from an in-memory table."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(f"CREATE TABLE t ({', '.join(columns)})")
    conn.executemany(
        f"INSERT INTO t VALUES ({', '.join('?' * len(columns))})",
        data,
    )
    return conn.execute("SELECT * FROM t").fetchall()


class TestPlayLogRowGate:
    """The skip gate: only a listen strictly exceeding the threshold counts."""

    @pytest.mark.parametrize(
        ("listened", "min_listen", "counts"),
        [
            (30, 5, True),
            (6, 5, True),
            (5, 5, False),  # exactly at the threshold is still a skip
            (3, 5, False),
            (0, 0, False),
            (1, 0, True),
            (0, 5, False),
            (-1, -2, True),  # a negative listen past a negative threshold logs
            (5.5, 5, True),  # floats compare numerically
            (5.0, 5, False),
            (6, 5.5, True),
            (True, 0, True),  # bools compare as ints: 1 > 0
            (True, 1, False),
            (False, 0, False),
        ],
    )
    def test_gate(self, listened, min_listen, counts):
        track = {"video_id": "v1"}
        row = _play_log_row(track, listened, "tui", min_listen)
        assert (row is not None) is counts

    def test_skip_returns_none_before_any_read(self):
        # A junk track below the threshold never raises: the gate is the
        # first thing evaluated, exactly like the reference.
        assert _play_log_row(object(), 3, "tui", 5) is None


class TestPlayLogRow:
    def test_full_track_row(self):
        track = {
            "video_id": "dQw4w9WgXcQ",
            "title": "Never Gonna Give You Up",
            "artist": "Rick Astley",
            "album": "Whenever You Need Somebody",
            "duration_seconds": 213,
        }
        row = _play_log_row(track, 30, "search", 5)
        assert row == {
            "video_id": "dQw4w9WgXcQ",
            "title": "Never Gonna Give You Up",
            "artist": "Rick Astley",
            "album": "Whenever You Need Somebody",
            "duration": 213,
            "listened_seconds": 30,
            "source": "search",
        }

    def test_missing_optional_fields_default_to_empty_strings(self):
        row = _play_log_row({"video_id": "v1"}, 30, "queue", 5)
        assert row is not None
        assert row["title"] == ""
        assert row["artist"] == ""
        assert row["album"] == ""

    def test_none_field_values_pass_through_unchanged(self):
        # Only a MISSING key takes the "" default — a present None lands
        # in the row raw, exactly like the reference's .get semantics.
        row = _play_log_row({"video_id": "v1", "title": None}, 30, "queue", 5)
        assert row is not None
        assert row["title"] is None

    def test_values_pass_through_as_the_objects_they_are(self):
        listened = 30
        source = "tui"
        row = _play_log_row({"video_id": "v1"}, listened, source, 5)
        assert row is not None
        assert row["listened_seconds"] is listened
        assert row["source"] is source

    def test_bool_listened_passes_through(self):
        row = _play_log_row({"video_id": "v1"}, True, "tui", 0)
        assert row is not None
        assert row["listened_seconds"] is True

    def test_missing_video_id_raises_keyerror(self):
        with pytest.raises(KeyError, match="video_id"):
            _play_log_row({"title": "no id"}, 30, "tui", 5)

    def test_non_subscriptable_track_raises_typeerror(self):
        with pytest.raises(TypeError):
            _play_log_row(object(), 30, "tui", 5)

    def test_list_track_raises_typeerror(self):
        with pytest.raises(TypeError, match="list indices"):
            _play_log_row(["v1", "title"], 30, "tui", 5)

    def test_track_without_get_raises_attributeerror(self):
        # video_id resolves through []; the next read wants .get.
        with pytest.raises(AttributeError, match="has no attribute 'get'"):
            _play_log_row(GetItemOnly(), 30, "tui", 5)

    def test_duck_typed_get_track(self):
        row = _play_log_row(DuckGet({"video_id": "duck1", "title": "Duck Song"}), 30, "tui", 5)
        assert row is not None
        assert row["video_id"] == "duck1"
        assert row["title"] == "Duck Song"
        assert row["artist"] == ""
        assert row["album"] == ""
        assert row["duration"] == 0

    def test_incomparable_threshold_raises_typeerror(self):
        # 30 <= "5" — CPython's own comparison error, on either backend.
        with pytest.raises(TypeError, match="not supported between instances"):
            _play_log_row({"video_id": "v1"}, 30, "tui", "5")

    def test_fresh_map_every_call(self):
        track = {"video_id": "v1"}
        first = _play_log_row(track, 30, "tui", 5)
        second = _play_log_row(track, 30, "tui", 5)
        assert first is not None and second is not None
        assert first is not second
        assert first == second

    def test_input_track_never_mutated(self):
        track = {
            "video_id": "v1",
            "title": "T",
            "artist": "A",
            "album": "B",
            "duration_seconds": 200,
        }
        snapshot = dict(track)
        _play_log_row(track, 30, "tui", 5)
        assert track == snapshot


class TestPlayLogRowDuration:
    """The ``duration_seconds`` → ``duration`` → ``length`` chain, warts included."""

    @pytest.mark.parametrize(
        ("track", "expected"),
        [
            ({"duration_seconds": 213}, 213),
            ({"duration_seconds": "212"}, 212),  # int("212")
            ({"duration_seconds": " 42 "}, 42),  # surrounding whitespace
            ({"duration_seconds": "4_2"}, 42),  # underscore separator
            ({"duration_seconds": "٤٢"}, 42),  # non-ASCII decimal digits
            ({"duration_seconds": 4.9}, 4),  # float truncates
            ({"duration_seconds": 212.9}, 212),
            ({"duration_seconds": True}, 1),  # int(True)
            ({"duration_seconds": False}, 0),  # present-but-False coerces
            ({"duration_seconds": ""}, 0),  # int("") swallowed, nothing else
            ({"duration_seconds": "3:30"}, 0),  # colon strings never parse here
            ({"duration_seconds": None, "duration": 210}, 210),  # is-not-None gate
            ({"duration_seconds": {}, "duration": 210}, 210),  # int({}) swallowed
            ({"duration_seconds": [], "duration": 210}, 210),
            ({"duration": 210}, 210),
            ({"duration": True}, True),  # isinstance int: the object itself
            ({"duration": 0}, 0),
            ({"duration": 10**30}, 10**30),  # bigints pass through exactly
            ({"duration": "3:30"}, 210),
            ({"duration": "1:02:03"}, 3723),
            ({"duration": "0:00"}, 0),
            ({"duration": "bad"}, 0),  # no colon, not an int
            ({"duration": "1:2:3:4"}, 0),  # four parts match neither branch
            ({"duration": "a:b"}, 0),  # int("a") ValueError swallowed
            ({"duration": "bad", "length": "3:30"}, 210),  # fall through to length
            ({"duration": "3:30", "length": "9:59"}, 210),  # duration wins
            ({"length": "3:30"}, 210),
            ({"length": 210}, 210),
            ({}, 0),
            ({"duration": [1, 2]}, 0),  # neither int nor str
        ],
    )
    def test_duration(self, track, expected):
        track = {"video_id": "v1", **track}
        row = _play_log_row(track, 30, "tui", 5)
        assert row is not None
        assert row["duration"] == expected

    def test_bool_duration_is_the_bool_object(self):
        # The reference returns the stored object itself — True stays
        # True (it would bind to SQLite as 1 either way), never 1.
        row = _play_log_row({"video_id": "v1", "duration": True}, 30, "tui", 5)
        assert row is not None
        assert row["duration"] is True

    def test_huge_duration_passes_through_exactly(self):
        huge = 10**30
        row = _play_log_row({"video_id": "v1", "duration": huge}, 30, "tui", 5)
        assert row is not None
        assert row["duration"] is huge

    def test_string_duration_parts_parse_through_int(self):
        # " 3 : 30 " splits to [" 3 ", " 30 "]; int() strips each part.
        row = _play_log_row({"video_id": "v1", "duration": " 3 : 30 "}, 30, "tui", 5)
        assert row is not None
        assert row["duration"] == 210


class TestListenedDelta:
    @pytest.mark.parametrize(
        ("old_value", "listened", "expected"),
        [
            (None, 30, 30),  # falsy cells read as zero
            (0, 30, 30),
            (0.0, 30, 30),
            ("", 30, 30),
            (False, 30, 30),
            ([], 30, 30),
            ({}, 30, 30),
            (10, 30, 20),
            (10, 10, None),  # exactly equal writes nothing
            (10, 5, None),
            (30, 5, None),
            ("42", 100, 58),  # numeric strings coerce
            (" 42 ", 100, 58),
            ("٤٢", 100, 58),
            ("42", 42, None),
            (4.9, 5, 1),  # int(4.9) == 4
            (4.9, 4, None),
            (0.5, 1, 1),  # int(0.5) == 0
            (True, 2, 1),  # int(True) == 1
            (True, 1, None),
            (-10, 5, 15),
            (10**30, 10**30 + 5, 5),  # bigint arithmetic stays exact
            (10**30, 5, None),
        ],
    )
    def test_delta(self, old_value, listened, expected):
        assert _listened_delta(old_value, listened) == expected

    def test_huge_delta_is_exact(self):
        old = 10**30
        delta = _listened_delta(old, old + 5)
        assert delta == 5
        assert type(delta) is int

    @pytest.mark.parametrize(
        ("old_value", "listened"),
        [
            ("4.5", 30),  # int("4.5") ValueError
            ("abc", 30),
            (["x"], 30),  # int(["x"]) TypeError
        ],
    )
    def test_coercion_errors_propagate(self, old_value, listened):
        with pytest.raises((ValueError, TypeError)):
            _listened_delta(old_value, listened)

    def test_incomparable_pair_raises_typeerror(self):
        # 30 <= "5" after the coercion — CPython's comparison error.
        with pytest.raises(TypeError, match="not supported between instances"):
            _listened_delta(30, "5")


class TestStatsPayload:
    """The five-key aggregate payload behind ``get_stats``."""

    def test_payload_carries_the_five_aggregates(self):
        top_tracks = [
            {
                "video_id": "dQw4w9WgXcQ",
                "title": "Never Gonna Give You Up",
                "artist": "Rick Astley",
                "play_count": 3,
                "total_listened_seconds": 639,
                "last_played": "2026-10-31 12:00:00",
            }
        ]
        top_artists = [{"artist": "Rick Astley", "play_count": 3}]
        payload = _stats_payload(12, 3450, 7, top_tracks, top_artists)
        assert payload == {
            "total_plays": 12,
            "total_listen_time": 3450,
            "unique_tracks": 7,
            "top_tracks": top_tracks,
            "top_artists": top_artists,
        }

    def test_key_order_is_the_json_output_order(self):
        # The key order is the contract the JSON consumers' output
        # preserves — it must never silently reorder.
        payload = _stats_payload(0, 0, 0, [], [])
        assert list(payload) == [
            "total_plays",
            "total_listen_time",
            "unique_tracks",
            "top_tracks",
            "top_artists",
        ]

    def test_values_pass_through_as_the_objects_they_are(self):
        top_tracks = [{"video_id": "v1"}]
        top_artists = []
        payload = _stats_payload(10**30, 2.5, True, top_tracks, top_artists)
        assert payload["total_plays"] == 10**30
        assert type(payload["total_plays"]) is int  # bigint, never truncated
        assert payload["total_listen_time"] == 2.5
        assert type(payload["total_listen_time"]) is float
        assert payload["unique_tracks"] is True  # the bool object, never 1
        assert payload["top_tracks"] is top_tracks  # the very list object
        assert payload["top_artists"] is top_artists

    def test_none_aggregate_passes_through(self):
        # COALESCE keeps the SQL aggregates non-NULL today, but the
        # engine passes whatever it is given through untouched.
        payload = _stats_payload(None, None, None, None, None)
        assert payload == {
            "total_plays": None,
            "total_listen_time": None,
            "unique_tracks": None,
            "top_tracks": None,
            "top_artists": None,
        }

    def test_fresh_map_every_call(self):
        args = (12, 3450, 7, [], [])
        first = _stats_payload(*args)
        second = _stats_payload(*args)
        assert first is not second
        assert first == second

    def test_inputs_never_mutated(self):
        top_tracks = [{"video_id": "v1", "play_count": 1}]
        top_artists = [{"artist": "A"}]
        _stats_payload(12, 3450, 7, top_tracks, top_artists)
        assert top_tracks == [{"video_id": "v1", "play_count": 1}]
        assert top_artists == [{"artist": "A"}]


class TestRowDicts:
    """The row materialisation behind every history read."""

    def _fetch_rows(self, columns: list[str], data: list[tuple]) -> list[sqlite3.Row]:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE t ({', '.join(columns)})")
        conn.executemany(
            f"INSERT INTO t VALUES ({', '.join('?' * len(columns))})",
            data,
        )
        return conn.execute("SELECT * FROM t").fetchall()

    def test_sqlite_rows_become_plain_dicts(self):
        rows = self._fetch_rows(
            ["video_id", "title", "played_at"],
            [("v1", "T1", "2026-01-01"), ("v2", "T2", None)],
        )
        out = _row_dicts(rows)
        assert out == [
            {"video_id": "v1", "title": "T1", "played_at": "2026-01-01"},
            {"video_id": "v2", "title": "T2", "played_at": None},
        ]
        for item in out:
            assert type(item) is dict  # plain dicts, never sqlite3.Row

    def test_column_order_is_the_json_output_order(self):
        # The dicts' key order is the SELECT's column order — the key
        # order the JSON consumers' output preserves. It must never
        # silently reorder.
        rows = self._fetch_rows(["c", "a", "b"], [(1, 2, 3)])
        out = _row_dicts(rows)
        assert list(out[0]) == ["c", "a", "b"]

    def test_result_is_a_list(self):
        assert type(_row_dicts([])) is list

    def test_empty_rows(self):
        assert _row_dicts([]) == []

    def test_dict_rows_are_copied_not_passed_through(self):
        rows = [{"a": 1}]
        out = _row_dicts(rows)
        assert out[0] == rows[0]
        assert out[0] is not rows[0]  # copied, never the same object
        assert type(out[0]) is dict  # a dict subclass row flattens to a dict

    def test_dict_subclass_rows_are_copied_from_internal_storage(self):
        # CPython's dict() fast path copies a dict subclass that only
        # overrides keys() wholesale from its internal storage — the
        # override does not steer the conversion until __iter__ is
        # overridden too, which drops the row onto the mapping-protocol
        # path. Both shapes behave identically on both backends.
        class KeysOverride(dict):
            def keys(self):
                return ["renamed"]

        class KeysAndIterOverride(dict):
            def keys(self):
                return ["renamed"]

            def __iter__(self):
                return iter(["renamed"])

        rows = [KeysOverride({"renamed": 7, "ignored": 8})]
        assert _row_dicts(rows) == [{"renamed": 7, "ignored": 8}]
        rows = [KeysAndIterOverride({"renamed": 7, "ignored": 8})]
        assert _row_dicts(rows) == [{"renamed": 7}]

    def test_values_pass_through_as_the_objects_they_are(self):
        rows = [{"bigint": 10**30, "flag": True, "f": 2.5, "none": None, "s": "ü"}]
        out = _row_dicts(rows)
        item = out[0]
        assert item["bigint"] == 10**30
        assert type(item["bigint"]) is int  # bigint, never truncated
        assert item["flag"] is True  # the bool object, never 1
        assert type(item["f"]) is float
        assert item["none"] is None
        assert item["s"] == "ü"

    def test_duck_typed_keys_row(self):
        class RowDuck:
            def keys(self):
                return ["x", "y"]

            def __getitem__(self, key):
                return key.upper()

        assert _row_dicts([RowDuck()]) == [{"x": "X", "y": "Y"}]

    def test_pair_iterable_rows(self):
        assert _row_dicts([[("a", 1), ("b", 2)]]) == [{"a": 1, "b": 2}]

    def test_duplicate_pair_keys_last_wins(self):
        assert _row_dicts([[("a", 1), ("a", 2)]]) == [{"a": 2}]

    def test_generator_input_is_consumed(self):
        rows = iter([{"a": 1}, {"a": 2}])
        assert _row_dicts(rows) == [{"a": 1}, {"a": 2}]

    def test_fresh_list_and_fresh_dicts_every_call(self):
        rows = [{"a": 1}]
        first = _row_dicts(rows)
        second = _row_dicts(rows)
        assert first is not second
        assert first == second
        assert first[0] is not second[0]

    def test_non_convertible_row_raises_typeerror(self):
        # dict(5) — no keys method, not iterable — CPython's own message.
        with pytest.raises(TypeError, match=r"'int' object is not iterable"):
            _row_dicts([5])

    def test_non_iterable_rows_raises_typeerror(self):
        # iter(5) — the comprehension's own iteration error.
        with pytest.raises(TypeError, match=r"'int' object is not iterable"):
            _row_dicts(5)

    def test_iter_error_keeps_its_own_error(self):
        class BadIter:
            def __iter__(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _row_dicts(BadIter())

    def test_keys_attribute_not_callable_raises_typeerror(self):
        class KeysNotCallable:
            keys = 5

        with pytest.raises(TypeError, match=r"'int' object is not callable"):
            _row_dicts([KeysNotCallable()])

    def test_unhashable_duck_key_raises_typeerror(self):
        class UnhashableKeys:
            def keys(self):
                return [["nested"]]

            def __getitem__(self, key):
                return 1

        with pytest.raises(TypeError, match="unhashable type"):
            _row_dicts([UnhashableKeys()])

    def test_input_rows_never_mutated(self):
        rows = [{"a": 1}, [("b", 2)]]
        _row_dicts(rows)
        assert rows == [{"a": 1}, [("b", 2)]]


class TestSuggestionStrings:
    """The autocomplete string list behind ``get_search_suggestions``."""

    def test_sqlite_rows_become_the_query_list(self):
        rows = _sql_rows(["query"], [("never gonna",), ("never mind",), ("other",)])
        assert _suggestion_strings(rows) == ["never gonna", "never mind", "other"]

    def test_values_pass_through_as_the_objects_they_are(self):
        sentinel = object()
        assert _suggestion_strings([{"query": sentinel}])[0] is sentinel

    def test_result_is_a_list(self):
        assert type(_suggestion_strings([])) is list

    def test_empty_rows(self):
        assert _suggestion_strings([]) == []

    def test_dict_rows(self):
        rows = [{"query": "a"}, {"query": "b"}]
        assert _suggestion_strings(rows) == ["a", "b"]

    def test_duck_rows(self):
        assert _suggestion_strings([DuckGet({"query": "duck"})]) == ["duck"]

    def test_generator_input_is_consumed(self):
        rows = iter([{"query": "a"}, {"query": "b"}])
        assert _suggestion_strings(rows) == ["a", "b"]

    def test_missing_query_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="query"):
            _suggestion_strings([{"other": "x"}])

    def test_non_subscriptable_row_raises_typeerror(self):
        with pytest.raises(TypeError, match="not subscriptable"):
            _suggestion_strings([5])

    def test_non_iterable_rows_raises_typeerror(self):
        with pytest.raises(TypeError, match="object is not iterable"):
            _suggestion_strings(5)

    def test_raising_getitem_keeps_its_own_error(self):
        class Boom:
            def __getitem__(self, key):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _suggestion_strings([Boom()])

    def test_raising_iter_keeps_its_own_error(self):
        class BadIter:
            def __iter__(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _suggestion_strings(BadIter())

    def test_fresh_list_every_call(self):
        rows = [{"query": "a"}]
        first = _suggestion_strings(rows)
        second = _suggestion_strings(rows)
        assert first is not second
        assert first == second

    def test_input_rows_never_mutated(self):
        rows = [{"query": "a"}, {"query": "b"}]
        _suggestion_strings(rows)
        assert rows == [{"query": "a"}, {"query": "b"}]


class TestPlayedVideoIdSet:
    """The played-video-id set behind ``get_played_video_ids``."""

    def test_sqlite_rows_become_the_id_set(self):
        rows = _sql_rows(["video_id"], [("v1",), ("v2",)])
        assert _played_video_id_set(rows) == {"v1", "v2"}

    def test_duplicates_collapse(self):
        rows = _sql_rows(["video_id"], [("v1",), ("v1",), ("v2",)])
        assert _played_video_id_set(rows) == {"v1", "v2"}

    def test_true_and_one_alias_one_entry_first_inserted_survives(self):
        first = _played_video_id_set([{"video_id": True}, {"video_id": 1}])
        assert first == {True}
        assert next(iter(first)) is True  # the first inserted is the one kept
        second = _played_video_id_set([{"video_id": 1}, {"video_id": True}])
        assert second == {1}
        one = 1
        assert next(iter(second)) is one  # the int object that was inserted first

    def test_same_nan_object_collapses_distinct_nans_do_not(self):
        nan1 = float("nan")
        nan2 = float("nan")
        same = _played_video_id_set([{"video_id": nan1}, {"video_id": nan1}])
        assert len(same) == 1  # the identity probe collapses it with itself
        assert next(iter(same)) is nan1
        distinct = _played_video_id_set([{"video_id": nan1}, {"video_id": nan2}])
        assert len(distinct) == 2  # NaN != NaN: two entries

    def test_custom_eq_ids_collapse_first_inserted_kept(self):
        class Id:
            def __init__(self, label: str) -> None:
                self.label = label

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Id) and other.label == self.label

            def __hash__(self) -> int:
                return hash(self.label)

        a = Id("v1")
        b = Id("v1")
        result = _played_video_id_set([{"video_id": a}, {"video_id": b}])
        assert len(result) == 1
        assert next(iter(result)) is a

    def test_unhashable_video_id_raises_typeerror(self):
        with pytest.raises(TypeError, match="unhashable type"):
            _played_video_id_set([{"video_id": ["nested"]}])

    def test_raising_hash_keeps_its_own_error(self):
        class RaisingHash:
            def __hash__(self) -> int:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _played_video_id_set([{"video_id": RaisingHash()}])

    def test_missing_video_id_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="video_id"):
            _played_video_id_set([{"other": "x"}])

    def test_non_subscriptable_row_raises_typeerror(self):
        with pytest.raises(TypeError, match="not subscriptable"):
            _played_video_id_set([5])

    def test_non_iterable_rows_raises_typeerror(self):
        with pytest.raises(TypeError, match="object is not iterable"):
            _played_video_id_set(5)

    def test_result_is_a_set(self):
        assert type(_played_video_id_set([])) is set

    def test_empty_rows(self):
        assert _played_video_id_set([]) == set()

    def test_fresh_set_every_call(self):
        rows = [{"video_id": "v1"}]
        first = _played_video_id_set(rows)
        second = _played_video_id_set(rows)
        assert first is not second
        assert first == second

    def test_input_rows_never_mutated(self):
        rows = [{"video_id": "v1"}, {"video_id": "v2"}]
        _played_video_id_set(rows)
        assert rows == [{"video_id": "v1"}, {"video_id": "v2"}]


_FULL_PLAY_ROW = {
    "video_id": "dQw4w9WgXcQ",
    "title": "Never Gonna Give You Up",
    "artist": "Rick Astley",
    "album": "Whenever You Need Somebody",
    "duration": 213,
    "listened_seconds": 30,
    "source": "tui",
}


class TestPlayInsertParams:
    """The seven-value parameter tuple behind ``log_play``'s INSERT."""

    def test_full_row_read_in_the_reference_order(self):
        row = ReadLogRow(dict(_FULL_PLAY_ROW))
        assert _play_insert_params(row) == (
            "dQw4w9WgXcQ",
            "Never Gonna Give You Up",
            "Rick Astley",
            "Whenever You Need Somebody",
            213,
            30,
            "tui",
        )
        assert row.reads == [
            "video_id",
            "title",
            "artist",
            "album",
            "duration",
            "listened_seconds",
            "source",
        ]

    def test_result_is_a_tuple(self):
        assert type(_play_insert_params(dict(_FULL_PLAY_ROW))) is tuple

    def test_values_pass_through_as_the_objects_they_are(self):
        sentinel = object()
        row = dict(_FULL_PLAY_ROW)
        row["video_id"] = sentinel
        row["duration"] = True  # the bool object, never 1
        row["listened_seconds"] = 10**30  # bigint, never truncated
        result = _play_insert_params(row)
        assert result[0] is sentinel
        assert result[4] is True
        assert type(result[4]) is bool
        assert result[5] == 10**30
        assert type(result[5]) is int

    def test_missing_key_raises_from_the_read_that_missed(self):
        row = {k: v for k, v in _FULL_PLAY_ROW.items() if k != "duration"}
        with pytest.raises(KeyError, match="duration"):
            _play_insert_params(row)
        row = {k: v for k, v in _FULL_PLAY_ROW.items() if k != "video_id"}
        with pytest.raises(KeyError, match="video_id"):
            _play_insert_params(row)

    def test_duck_row(self):
        assert _play_insert_params(DuckGet(dict(_FULL_PLAY_ROW))) == (
            "dQw4w9WgXcQ",
            "Never Gonna Give You Up",
            "Rick Astley",
            "Whenever You Need Somebody",
            213,
            30,
            "tui",
        )

    def test_non_subscriptable_row_raises_typeerror(self):
        with pytest.raises(TypeError, match="not subscriptable"):
            _play_insert_params(object())

    def test_raising_getitem_keeps_its_own_error(self):
        class Boom:
            def __getitem__(self, key):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _play_insert_params(Boom())

    def test_fresh_tuple_every_call(self):
        row = dict(_FULL_PLAY_ROW)
        first = _play_insert_params(row)
        second = _play_insert_params(row)
        assert first is not second
        assert first == second

    def test_input_row_never_mutated(self):
        row = dict(_FULL_PLAY_ROW)
        snapshot = dict(row)
        _play_insert_params(row)
        assert row == snapshot


class TestPlayStatsParams:
    """The four-value parameter tuple behind ``log_play``'s upsert."""

    def test_four_values_read_in_the_reference_order(self):
        row = ReadLogRow(dict(_FULL_PLAY_ROW))
        assert _play_stats_params(row) == (
            "dQw4w9WgXcQ",
            "Never Gonna Give You Up",
            "Rick Astley",
            30,
        )
        # Exactly four reads — album, duration and source are never
        # touched by the upsert.
        assert row.reads == ["video_id", "title", "artist", "listened_seconds"]

    def test_result_is_a_tuple(self):
        assert type(_play_stats_params(dict(_FULL_PLAY_ROW))) is tuple

    def test_missing_listened_seconds_raises_from_the_fourth_read(self):
        row = {k: v for k, v in _FULL_PLAY_ROW.items() if k != "listened_seconds"}
        with pytest.raises(KeyError, match="listened_seconds"):
            _play_stats_params(row)

    def test_values_pass_through_as_the_objects_they_are(self):
        row = dict(_FULL_PLAY_ROW)
        row["listened_seconds"] = True
        result = _play_stats_params(row)
        assert result[3] is True  # the bool object, never 1

    def test_duck_row(self):
        assert _play_stats_params(DuckGet(dict(_FULL_PLAY_ROW))) == (
            "dQw4w9WgXcQ",
            "Never Gonna Give You Up",
            "Rick Astley",
            30,
        )

    def test_non_subscriptable_row_raises_typeerror(self):
        with pytest.raises(TypeError, match="not subscriptable"):
            _play_stats_params(object())

    def test_fresh_tuple_every_call(self):
        row = dict(_FULL_PLAY_ROW)
        first = _play_stats_params(row)
        second = _play_stats_params(row)
        assert first is not second
        assert first == second

    def test_input_row_never_mutated(self):
        row = dict(_FULL_PLAY_ROW)
        snapshot = dict(row)
        _play_stats_params(row)
        assert row == snapshot


# ── Native vs fallback parity ──────────────────────────────────────────
# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable: plain dict tracks, JSON
# scalars, and the special track shapes spelled out as specs both sides
# build identically.

_PLAY_PARITY_CASES = [
    # Gate: below / at / above the threshold, plus the float, bool and
    # negative spellings.
    ({"t": "dict", "d": {"video_id": "v1"}}, 3, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 5, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 6, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 0, "tui", 0),
    ({"t": "dict", "d": {"video_id": "v1"}}, 1, "tui", 0),
    ({"t": "dict", "d": {"video_id": "v1"}}, -1, "tui", -2),
    ({"t": "dict", "d": {"video_id": "v1"}}, 5.5, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 5.0, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 6, "tui", 5.5),
    ({"t": "dict", "d": {"video_id": "v1"}}, True, "tui", 0),
    ({"t": "dict", "d": {"video_id": "v1"}}, True, "tui", 1),
    ({"t": "dict", "d": {"video_id": "v1"}}, False, "tui", 0),
    # The gate runs before anything is read — junk tracks below the
    # threshold never raise.
    ({"t": "no_get"}, 3, "tui", 5),
    ({"t": "list", "v": [1, 2]}, 3, "tui", 5),
    # Row assembly: full and partial tracks.
    (
        {
            "t": "dict",
            "d": {
                "video_id": "dQw4w9WgXcQ",
                "title": "Never Gonna Give You Up",
                "artist": "Rick Astley",
                "album": "Whenever You Need Somebody",
                "duration_seconds": 213,
            },
        },
        180,
        "search",
        5,
    ),
    ({"t": "dict", "d": {"video_id": "v1", "title": None}}, 30, "queue", 5),
    ({"t": "dict", "d": {"video_id": "夜に駆ける"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 30, "", 5),
    # The duration chain, key by key.
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": 213}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": "212"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": " 42 "}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": "4_2"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": "٤٢"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": 4.9}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": 212.9}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": True}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": False}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": ""}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration_seconds": "3:30"}}, 30, "tui", 5),
    (
        {"t": "dict", "d": {"video_id": "v1", "duration_seconds": None, "duration": 210}},
        30,
        "tui",
        5,
    ),
    (
        {"t": "dict", "d": {"video_id": "v1", "duration_seconds": {}, "duration": 210}},
        30,
        "tui",
        5,
    ),
    (
        {"t": "dict", "d": {"video_id": "v1", "duration_seconds": [], "duration": 210}},
        30,
        "tui",
        5,
    ),
    ({"t": "dict", "d": {"video_id": "v1", "duration": 210}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": True}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": 0}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": 10**30}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": "3:30"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": "1:02:03"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": "0:00"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": "bad"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": "1:2:3:4"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": "a:b"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": " 3 : 30 "}}, 30, "tui", 5),
    (
        {"t": "dict", "d": {"video_id": "v1", "duration": "bad", "length": "3:30"}},
        30,
        "tui",
        5,
    ),
    (
        {"t": "dict", "d": {"video_id": "v1", "duration": "3:30", "length": "9:59"}},
        30,
        "tui",
        5,
    ),
    ({"t": "dict", "d": {"video_id": "v1", "length": "3:30"}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "length": 210}}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1", "duration": [1, 2]}}, 30, "tui", 5),
    ({"t": "dict", "d": {}}, 30, "tui", 5),
    # Duck-typed and error shapes: the [] read, the .get dispatch and
    # CPython's own errors on both backends.
    ({"t": "duck", "d": {"video_id": "duck1", "title": "Duck Song"}}, 30, "tui", 5),
    ({"t": "duck", "d": {"video_id": "duck1", "duration": "3:30"}}, 30, "tui", 5),
    ({"t": "duck", "d": {}}, 30, "tui", 5),
    ({"t": "getitem_only"}, 30, "tui", 5),
    ({"t": "override"}, 30, "tui", 5),
    ({"t": "no_get"}, 30, "tui", 5),
    ({"t": "list", "v": [1, 2]}, 30, "tui", 5),
    ({"t": "dict", "d": {"video_id": "v1"}}, 30, "tui", "5"),  # TypeError
]

_DELTA_PARITY_CASES = [
    (None, 30),
    (0, 30),
    (0.0, 30),
    ("", 30),
    (False, 30),
    ([], 30),
    ({}, 30),
    (10, 30),
    (10, 10),
    (10, 5),
    (30, 5),
    ("42", 100),
    (" 42 ", 100),
    ("٤٢", 100),
    ("42", 42),
    (4.9, 5),
    (4.9, 4),
    (0.5, 1),
    (True, 2),
    (True, 1),
    (-10, 5),
    (10**30, 10**30 + 5),
    (10**30, 5),
    ("4.5", 30),  # ValueError
    ("abc", 30),  # ValueError
    (["x"], 30),  # TypeError
    (30, "5"),  # comparison TypeError
]

# Stats payload cases: every value must be JSON-round-trippable (the
# matrices cross the process boundary as JSON), so the shape zoo lives
# in the unit tests above and this matrix pins the realistic + edge
# value combinations.
_STATS_PARITY_CASES = [
    (12, 3450, 7, [], []),
    (0, 0, 0, [], []),  # empty database
    (
        12,
        3450,
        7,
        [
            {
                "video_id": "dQw4w9WgXcQ",
                "title": "Never Gonna Give You Up",
                "artist": "Rick Astley",
                "play_count": 3,
                "total_listened_seconds": 639,
                "last_played": "2026-10-31 12:00:00",
            }
        ],
        [
            {
                "artist": "Rick Astley",
                "play_count": 3,
                "total_listened_seconds": 639,
            }
        ],
    ),
    (10**30, 10**30 + 5, 10**18, [], []),  # bigints pass through exactly
    (True, False, True, [], []),  # bools pass through as the objects they are
    (None, None, None, None, None),  # a NULL aggregate would pass through raw
    (1.5, 2.0, 3, [], []),  # floats
    ("12", "3450", "7", [], []),  # strings
    (12, 3450, 7, ["nested", {"a": 1}], [None]),  # the lists pass through verbatim
]

# Row-materialisation cases: the same spec grammar both processes build
# identically (the "sql" kind reconstructs the production shape — an
# sqlite3.Row list — from an in-memory table).
_ROWS_PARITY_CASES = [
    # The production shape: sqlite3.Row lists.
    {
        "t": "sql",
        "columns": ["video_id", "title", "played_at"],
        "data": [["v1", "T1", "2026-01-01"], ["v2", "T2", None]],
    },
    {"t": "sql", "columns": ["a", "b"], "data": [[1.5, 1], [0, 0]]},
    {"t": "sql", "columns": ["s"], "data": [["ü"], [""], [None]]},
    {"t": "sql", "columns": ["a"], "data": []},  # an empty fetch
    # Dict rows (copied) with the JSON-transportable value zoo.
    {"t": "list", "items": []},
    {"t": "list", "items": [{"a": 1}, {"a": 2}]},
    {"t": "list", "items": [{"bigint": 10**30, "f": 2.5, "none": None, "flag": True}]},
    # A mixed list: a dict row, a pair-list row, then a non-convertible one.
    {"t": "list", "items": [{"a": 1}, [["b", 2]], 5]},
    # A string row: iterable of characters, each not a pair.
    {"t": "list", "items": ["ab"]},
    # Duck-typed rows through the mapping protocol.
    {"t": "duck", "items": [{"x": 1, "y": 2}]},
    {"t": "duck", "items": [{"bigint": 10**30, "f": 2.5}]},
    # Rows that are lists of pairs, duplicate keys last-wins.
    {"t": "pairs", "items": [[["a", 1], ["b", 2]], [["a", 1], ["a", 2]]]},
    # A generator input.
    {"t": "gen", "items": [{"a": 1}, {"b": 2}]},
    {"t": "gen", "items": []},
    # Error shapes: non-iterable rows input, non-convertible rows,
    # unhashable duck keys, a raising keys(), a non-callable keys attr.
    {"t": "int"},
    {"t": "list", "items": [1, 2]},
    {"t": "unhashable"},
    {"t": "raising_keys"},
    {"t": "keys_not_callable"},
]

# Suggestion-string cases: the same row grammar, read through the
# ``query`` key.
_SUGGESTIONS_PARITY_CASES = [
    # The production shape: sqlite3.Row lists.
    {"t": "sql", "columns": ["query"], "data": [["never gonna"], ["never mind"]]},
    {"t": "sql", "columns": ["query"], "data": [["ü"], [""], [None]]},
    {"t": "sql", "columns": ["query"], "data": []},
    # Dict rows with the JSON-transportable value zoo.
    {"t": "list", "items": []},
    {"t": "list", "items": [{"query": "a"}, {"query": "b"}, {"query": "a"}]},
    {"t": "list", "items": [{"query": 1}, {"query": None}, {"query": True}]},
    # Duck-typed rows through the subscript protocol.
    {"t": "duck", "items": [{"query": "duck"}]},
    # A generator input.
    {"t": "gen", "items": [{"query": "a"}, {"query": "b"}]},
    # Error shapes: a missing key, a non-subscriptable row, a raising
    # __getitem__, a non-iterable rows input.
    {"t": "list", "items": [{"query": "a"}, {}]},
    {"t": "list", "items": [5]},
    {"t": "raising_getitem"},
    {"t": "int"},
]

# Played-video-id cases: the same row grammar, read through the
# ``video_id`` key and inserted into a set (the outcome records the
# sorted reprs of the survivors — the order-independent witness of
# which values the set kept).
_VIDEOIDS_PARITY_CASES = [
    # The production shape: sqlite3.Row lists.
    {"t": "sql", "columns": ["video_id"], "data": [["v1"], ["v2"]]},
    {"t": "sql", "columns": ["video_id"], "data": [["v1"], ["v1"]]},  # collapse
    {"t": "sql", "columns": ["video_id"], "data": []},
    {"t": "sql", "columns": ["video_id"], "data": [["v1"], [None], [""]]},
    # Dict rows: True/1 alias one entry — the first inserted survives.
    {"t": "list", "items": [{"video_id": True}, {"video_id": 1}]},
    {"t": "list", "items": [{"video_id": 1}, {"video_id": True}]},
    {"t": "list", "items": [{"video_id": "v1"}, {"video_id": "v1"}]},
    # An unhashable video_id — the set-add's own TypeError.
    {"t": "list", "items": [{"video_id": ["nested"]}]},
    # A generator input.
    {"t": "gen", "items": [{"video_id": "v1"}]},
    # Error shapes: a missing key, a non-subscriptable row, a raising
    # __getitem__, a non-iterable rows input.
    {"t": "list", "items": [{"video_id": "v1"}, {}]},
    {"t": "list", "items": [5]},
    {"t": "raising_getitem"},
    {"t": "int"},
]

# Play-parameter cases: the track/row grammar (the same specs
# _build_track builds), read through the INSERT's left-to-right key
# order.
_INSERT_PARAMS_PARITY_CASES = [
    {"t": "dict", "d": dict(_FULL_PLAY_ROW)},
    {
        "t": "dict",
        "d": {
            "video_id": "夜に駆ける",
            "title": None,
            "artist": None,
            "album": None,
            "duration": 0,
            "listened_seconds": True,
            "source": "",
        },
    },
    {"t": "dict", "d": {"title": "no id"}},  # KeyError video_id (first read)
    # KeyError source — the last of the seven reads.
    {
        "t": "dict",
        "d": {
            "video_id": "v1",
            "title": "T",
            "artist": "A",
            "album": "B",
            "duration": 1,
            "listened_seconds": 2,
        },
    },
    {"t": "duck", "d": dict(_FULL_PLAY_ROW)},
    {"t": "getitem_only"},  # every read answers "VID"
    {"t": "no_get"},  # not subscriptable
    {"t": "list", "v": ["v1", "t"]},  # list indices TypeError
    {"t": "override"},  # a dict subclass row — only video_id present
]

# Stats-parameter cases: the same grammar, read through the upsert's
# four-key order.
_STATS_PARAMS_PARITY_CASES = [
    {"t": "dict", "d": dict(_FULL_PLAY_ROW)},
    # Only four of the seven keys are ever read by the upsert.
    {
        "t": "dict",
        "d": {
            "video_id": "v1",
            "title": "T",
            "artist": "A",
            "album": "never read",
            "duration": "never read",
            "listened_seconds": 9,
            "source": "never read",
        },
    },
    # KeyError listened_seconds — the fourth read.
    {"t": "dict", "d": {"video_id": "v1", "title": "T", "artist": "A"}},
    {"t": "duck", "d": dict(_FULL_PLAY_ROW)},
    {"t": "getitem_only"},
    {"t": "no_get"},
]


class RowDuck:
    """A row-shaped object steering ``dict()`` through the mapping
    protocol — its ``keys()`` decides what the materialised dict reads."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def keys(self) -> list:
        return list(self._data)

    def __getitem__(self, key: str) -> object:
        return self._data[key]


def _build_rows(spec: dict) -> object:
    """Build the parity-matrix row shapes, identically in both processes."""
    kind = spec["t"]
    if kind == "sql":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE t ({', '.join(spec['columns'])})")
        conn.executemany(
            f"INSERT INTO t VALUES ({', '.join('?' * len(spec['columns']))})",
            [tuple(row) for row in spec["data"]],
        )
        return conn.execute("SELECT * FROM t").fetchall()
    if kind == "list":
        return spec["items"]
    if kind == "duck":
        return [RowDuck(d) for d in spec["items"]]
    if kind == "pairs":
        return [[tuple(pair) for pair in row] for row in spec["items"]]
    if kind == "gen":
        return iter(spec["items"])
    if kind == "int":
        return 5
    if kind == "unhashable":

        class UnhashableKeys:
            def keys(self):
                return [["nested"]]

            def __getitem__(self, key):
                return 1

        return [UnhashableKeys()]
    if kind == "raising_keys":

        class RaisingKeys:
            def keys(self):
                raise RuntimeError("boom")

            def __getitem__(self, key):
                return 1

        return [RaisingKeys()]
    if kind == "keys_not_callable":

        class KeysNotCallable:
            keys = 5

        return [KeysNotCallable()]
    if kind == "raising_getitem":

        class RaisingGetItem:
            def __getitem__(self, key):
                raise RuntimeError("boom")

        return [RaisingGetItem()]
    raise AssertionError(kind)


def _build_track(spec: dict) -> object:
    """Build the parity-matrix track shapes, identically in both processes."""
    kind = spec["t"]
    if kind == "dict":
        return spec["d"]
    if kind == "duck":
        return DuckGet(spec["d"])
    if kind == "no_get":
        return object()
    if kind == "list":
        return spec["v"]
    if kind == "getitem_only":
        return GetItemOnly()
    if kind == "override":

        class GetOverride(dict):
            """A dict whose .get is overridden — steering both backends."""

            def get(self, key: str, default: object = None) -> str:
                return "OV"

        return GetOverride({"video_id": "ov1"})
    raise AssertionError(kind)


def _outcome(fn: object, *args: object) -> dict:
    try:
        return {"ok": fn(*args)}  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def _set_outcome(fn: object, *args: object) -> dict:
    try:
        result = fn(*args)  # type: ignore[operator]
        # A set is not JSON-transportable — the sorted reprs pin which
        # values survived (True vs 1, "" vs None), the
        # order-independent witness of the set's contents.
        return {"ok": sorted(repr(x) for x in result)}  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tuple_outcome(fn: object, *args: object) -> dict:
    try:
        result = fn(*args)  # type: ignore[operator]
        # Tuples flatten to lists on the JSON wire — normalise the
        # native side the same way.
        return {"ok": list(result)}  # type: ignore[operator]
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


# The child interpreter runs this program with the case matrices on
# stdin and prints the reference engines' outcomes as JSON.
_CHILD_PROG = """
import json
import sqlite3
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.history import (
    _listened_delta,
    _played_video_id_set,
    _play_insert_params,
    _play_log_row,
    _play_stats_params,
    _row_dicts,
    _stats_payload,
    _suggestion_strings,
)


class DuckGet:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, key, default=None):
        return self._data[key] if key in self._data else default


class GetItemOnly:
    def __getitem__(self, key):
        return "VID"


class GetOverride(dict):
    def get(self, key, default=None):
        return "OV"


class RowDuck:
    def __init__(self, data):
        self._data = data

    def keys(self):
        return list(self._data)

    def __getitem__(self, key):
        return self._data[key]


def build_track(spec):
    kind = spec["t"]
    if kind == "dict":
        return spec["d"]
    if kind == "duck":
        return DuckGet(spec["d"])
    if kind == "no_get":
        return object()
    if kind == "list":
        return spec["v"]
    if kind == "getitem_only":
        return GetItemOnly()
    if kind == "override":
        return GetOverride({"video_id": "ov1"})
    raise AssertionError(kind)


def build_rows(spec):
    kind = spec["t"]
    if kind == "sql":
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(f"CREATE TABLE t ({', '.join(spec['columns'])})")
        conn.executemany(
            f"INSERT INTO t VALUES ({', '.join('?' * len(spec['columns']))})",
            [tuple(row) for row in spec["data"]],
        )
        return conn.execute("SELECT * FROM t").fetchall()
    if kind == "list":
        return spec["items"]
    if kind == "duck":
        return [RowDuck(d) for d in spec["items"]]
    if kind == "pairs":
        return [[tuple(pair) for pair in row] for row in spec["items"]]
    if kind == "gen":
        return iter(spec["items"])
    if kind == "int":
        return 5
    if kind == "unhashable":
        class UnhashableKeys:
            def keys(self):
                return [["nested"]]

            def __getitem__(self, key):
                return 1

        return [UnhashableKeys()]
    if kind == "raising_keys":
        class RaisingKeys:
            def keys(self):
                raise RuntimeError("boom")

            def __getitem__(self, key):
                return 1

        return [RaisingKeys()]
    if kind == "keys_not_callable":
        class KeysNotCallable:
            keys = 5

        return [KeysNotCallable()]
    if kind == "raising_getitem":
        class RaisingGetItem:
            def __getitem__(self, key):
                raise RuntimeError("boom")

        return [RaisingGetItem()]
    raise AssertionError(kind)


def outcome(fn, *args):
    try:
        return {"ok": fn(*args)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def set_outcome(fn, *args):
    try:
        result = fn(*args)
        return {"ok": sorted(repr(x) for x in result)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def tuple_outcome(fn, *args):
    try:
        result = fn(*args)
        return {"ok": list(result)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
rows = [
    outcome(_play_log_row, build_track(spec), listened, source, min_listen)
    for spec, listened, source, min_listen in payload["rows"]
]
deltas = [outcome(_listened_delta, old, listened) for old, listened in payload["deltas"]]
stats = [outcome(_stats_payload, *case) for case in payload["stats"]]
row_dicts = [outcome(_row_dicts, build_rows(spec)) for spec in payload["row_dicts"]]
suggestions = [
    outcome(_suggestion_strings, build_rows(spec)) for spec in payload["suggestions"]
]
video_ids = [
    set_outcome(_played_video_id_set, build_rows(spec)) for spec in payload["video_ids"]
]
insert_params = [
    tuple_outcome(_play_insert_params, build_track(spec))
    for spec in payload["insert_params"]
]
stats_params = [
    tuple_outcome(_play_stats_params, build_track(spec)) for spec in payload["stats_params"]
]
json.dump(
    {
        "rows": rows,
        "deltas": deltas,
        "stats": stats,
        "row_dicts": row_dicts,
        "suggestions": suggestions,
        "video_ids": video_ids,
        "insert_params": insert_params,
        "stats_params": stats_params,
    },
    sys.stdout,
)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines over the parity cases in a
    child interpreter with the native extension import blocked."""
    payload = json.dumps(
        {
            "rows": _PLAY_PARITY_CASES,
            "deltas": _DELTA_PARITY_CASES,
            "stats": _STATS_PARITY_CASES,
            "row_dicts": _ROWS_PARITY_CASES,
            "suggestions": _SUGGESTIONS_PARITY_CASES,
            "video_ids": _VIDEOIDS_PARITY_CASES,
            "insert_params": _INSERT_PARAMS_PARITY_CASES,
            "stats_params": _STATS_PARAMS_PARITY_CASES,
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _native_outcomes() -> dict:
    rows = [
        _outcome(_play_log_row, _build_track(spec), listened, source, min_listen)
        for spec, listened, source, min_listen in _PLAY_PARITY_CASES
    ]
    deltas = [_outcome(_listened_delta, old, listened) for old, listened in _DELTA_PARITY_CASES]
    stats = [_outcome(_stats_payload, *case) for case in _STATS_PARITY_CASES]
    row_dicts = [_outcome(_row_dicts, _build_rows(spec)) for spec in _ROWS_PARITY_CASES]
    suggestions = [
        _outcome(_suggestion_strings, _build_rows(spec)) for spec in _SUGGESTIONS_PARITY_CASES
    ]
    video_ids = [
        _set_outcome(_played_video_id_set, _build_rows(spec)) for spec in _VIDEOIDS_PARITY_CASES
    ]
    insert_params = [
        _tuple_outcome(_play_insert_params, _build_track(spec))
        for spec in _INSERT_PARAMS_PARITY_CASES
    ]
    stats_params = [
        _tuple_outcome(_play_stats_params, _build_track(spec))
        for spec in _STATS_PARAMS_PARITY_CASES
    ]
    return {
        "rows": rows,
        "deltas": deltas,
        "stats": stats,
        "row_dicts": row_dicts,
        "suggestions": suggestions,
        "video_ids": video_ids,
        "insert_params": insert_params,
        "stats_params": stats_params,
    }


class TestNativeParity:
    """When the Rust extension is built, the native engines must decide
    exactly like the pure-Python references — same gate, same row values
    (object pass-through included), same errors with the same messages —
    for every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_play_log_row_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(_PLAY_PARITY_CASES, native["rows"], fallback["rows"]):
            assert n == f, (
                f"native and fallback row outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_listened_delta_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(_DELTA_PARITY_CASES, native["deltas"], fallback["deltas"]):
            assert n == f, (
                f"native and fallback delta outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_stats_payload_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(_STATS_PARITY_CASES, native["stats"], fallback["stats"]):
            assert n == f, (
                f"native and fallback stats outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # Key order is part of the contract (the JSON consumers'
        # output preserves it) — compare the serialized payloads too,
        # which pins the order, not just the mapping.
        for n, f in zip(native["stats"], fallback["stats"]):
            assert list(n["ok"]) == list(f["ok"]), (
                f"native and fallback stats key orders differ:\n"
                f"native:   {list(n['ok'])!r}\n"
                f"fallback: {list(f['ok'])!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_row_dicts_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(_ROWS_PARITY_CASES, native["row_dicts"], fallback["row_dicts"]):
            assert n == f, (
                f"native and fallback row-dict outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # Key order is part of the contract (the JSON consumers'
        # output preserves the SELECT's column order) — compare the
        # materialised dicts' key lists too, which pins the order.
        for n, f in zip(native["row_dicts"], fallback["row_dicts"]):
            if "ok" in n and "ok" in f:
                native_keys = [list(d) for d in n["ok"]]
                fallback_keys = [list(d) for d in f["ok"]]
                assert native_keys == fallback_keys, (
                    f"native and fallback row-dict key orders differ:\n"
                    f"native:   {native_keys!r}\n"
                    f"fallback: {fallback_keys!r}"
                )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_suggestion_strings_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(
            _SUGGESTIONS_PARITY_CASES, native["suggestions"], fallback["suggestions"]
        ):
            assert n == f, (
                f"native and fallback suggestion outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_played_video_id_set_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(_VIDEOIDS_PARITY_CASES, native["video_ids"], fallback["video_ids"]):
            assert n == f, (
                f"native and fallback video-id outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_play_insert_params_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(
            _INSERT_PARAMS_PARITY_CASES, native["insert_params"], fallback["insert_params"]
        ):
            assert n == f, (
                f"native and fallback insert-param outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_play_stats_params_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        for case, n, f in zip(
            _STATS_PARAMS_PARITY_CASES, native["stats_params"], fallback["stats_params"]
        ):
            assert n == f, (
                f"native and fallback stats-param outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
