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
and the two top lists, in the JSON consumers' key order. The
pure-Python transcriptions live in ``services/history.py``; when the
Rust extension is built the same engines run natively
(``_ytm_player_native.play_log_row`` / ``listened_delta`` /
``stats_payload``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case —
including the error shapes, where CPython's own messages must surface
identically — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.services.history import _listened_delta, _play_log_row, _stats_payload

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


# The child interpreter runs this program with the case matrices on
# stdin and prints the reference engines' outcomes as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.history import _listened_delta, _play_log_row, _stats_payload


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


def outcome(fn, *args):
    try:
        return {"ok": fn(*args)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
rows = [
    outcome(_play_log_row, build_track(spec), listened, source, min_listen)
    for spec, listened, source, min_listen in payload["rows"]
]
deltas = [outcome(_listened_delta, old, listened) for old, listened in payload["deltas"]]
stats = [outcome(_stats_payload, *case) for case in payload["stats"]]
json.dump({"rows": rows, "deltas": deltas, "stats": stats}, sys.stdout)
"""


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines over the parity cases in a
    child interpreter with the native extension import blocked."""
    payload = json.dumps(
        {"rows": _PLAY_PARITY_CASES, "deltas": _DELTA_PARITY_CASES, "stats": _STATS_PARITY_CASES}
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
    return {"rows": rows, "deltas": deltas, "stats": stats}


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
