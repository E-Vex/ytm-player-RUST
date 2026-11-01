"""Tests for IPCMixin._handle_ipc_command dispatch + seek parsing.

The CLI shells (`ytm play`, `ytm seek 1:30`, etc.) all hit
_handle_ipc_command via the Unix socket. These tests cover the
no-player guards, unknown-command fallthrough, and the three seek
formats (relative, mm:ss, absolute seconds).

``_ipc_dispatch_route`` is the engine behind the dispatch: the table
walk deciding which branch handles each command string, and the rating
the like/dislike/unlike branches rate with. ``_ipc_seek_offset`` is
the engine behind the seek branch: the classification of the offset
value into relative/clock/absolute seconds with its missing/invalid
verdicts. The pure-Python transcriptions live in ``app/_ipc.py``; when
the Rust extension is built the same engines run natively
(``_ytm_player_native.ipc_dispatch_route`` /
``_ytm_player_native.ipc_seek_offset``). The unit tests below
assert each engine's contract on whichever backend is loaded, plus the
mixin glue (every branch's handler calls, including the rating the
rate branches pass and the seek calls each parsed kind names). When
the extension is built, the parity tests additionally compare the
native engines against the pure-Python references (run in a child
interpreter with the extension import blocked) case for case —
including the error shapes, where CPython's own messages must surface
identically — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ytm_player.app._ipc import (
    _IPC_DISPATCH_TABLE,
    IPCMixin,
    _ipc_dispatch_route,
    _ipc_seek_offset,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _fresh_ipc_host():
    h = IPCMixin()
    h.player = MagicMock()
    h.player.resume = AsyncMock()
    h.player.pause = AsyncMock()
    h.player.seek = AsyncMock()
    h.player.seek_absolute = AsyncMock()
    h.queue = MagicMock()
    h.queue.clear = MagicMock()
    h._refresh_queue_page = MagicMock()
    h.ytmusic = MagicMock()
    h._play_next = AsyncMock()
    h._play_previous = AsyncMock()
    return h


class TestUnknownCommand:
    async def test_unknown_command_returns_error(self):
        h = _fresh_ipc_host()
        result = await h._handle_ipc_command("does_not_exist", {})
        assert result["ok"] is False
        assert "unknown command" in result["error"]


class TestNoPlayerGuard:
    async def test_play_with_no_player_returns_error(self):
        h = _fresh_ipc_host()
        h.player = None
        result = await h._handle_ipc_command("play", {})
        assert result == {"ok": False, "error": "player not ready"}

    async def test_pause_with_no_player_returns_error(self):
        h = _fresh_ipc_host()
        h.player = None
        result = await h._handle_ipc_command("pause", {})
        assert result == {"ok": False, "error": "player not ready"}

    async def test_seek_with_no_player_returns_error(self):
        h = _fresh_ipc_host()
        h.player = None
        result = await h._handle_ipc_command("seek", {"offset": "+10"})
        assert result == {"ok": False, "error": "player not ready"}


class TestPlayPauseDispatch:
    async def test_play_calls_resume(self):
        h = _fresh_ipc_host()
        result = await h._handle_ipc_command("play", {})
        assert result == {"ok": True}
        h.player.resume.assert_awaited_once()

    async def test_pause_calls_pause(self):
        h = _fresh_ipc_host()
        result = await h._handle_ipc_command("pause", {})
        assert result == {"ok": True}
        h.player.pause.assert_awaited_once()

    async def test_queue_clear_clears_queue(self):
        h = _fresh_ipc_host()
        result = await h._handle_ipc_command("queue_clear", {})
        assert result == {"ok": True}
        h.queue.clear.assert_called_once()
        h._refresh_queue_page.assert_called_once()


class TestSeekParsing:
    async def test_relative_positive(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "+15"})
        assert result == {"ok": True}
        h.player.seek.assert_awaited_once_with(15.0)

    async def test_relative_negative(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "-10"})
        assert result == {"ok": True}
        h.player.seek.assert_awaited_once_with(-10.0)

    async def test_mm_ss_format(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "1:30"})
        assert result == {"ok": True}
        h.player.seek_absolute.assert_awaited_once_with(90.0)

    async def test_hh_mm_ss_format(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "1:00:00"})
        assert result == {"ok": True}
        h.player.seek_absolute.assert_awaited_once_with(3600.0)

    async def test_absolute_seconds(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "42"})
        assert result == {"ok": True}
        h.player.seek_absolute.assert_awaited_once_with(42.0)

    async def test_missing_offset(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({})
        assert result == {"ok": False, "error": "missing offset"}

    async def test_invalid_offset(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "+notanumber"})
        assert result["ok"] is False
        assert "invalid offset" in result["error"]

    async def test_invalid_time_format(self):
        h = _fresh_ipc_host()
        result = await h._ipc_seek({"offset": "1:2:3:4"})
        assert result["ok"] is False
        assert "invalid time format" in result["error"]


class TestSeekOffset:
    """The offset parser behind ``_ipc_seek``."""

    @pytest.mark.parametrize(
        ("offset", "kind", "seconds"),
        [
            ("+15", "relative", 15.0),
            ("-10", "relative", -10.0),
            ("+0", "relative", 0.0),
            ("-0", "relative", -0.0),
            ("+1e1", "relative", 10.0),
            ("-1.5e-2", "relative", -0.015),
            ("1:30", "clock", 90.0),
            ("0:0", "clock", 0.0),
            ("01:30", "clock", 90.0),
            ("90:00", "clock", 5400.0),
            ("1:00:00", "clock", 3600.0),
            ("1:30:60", "clock", 5460.0),
            ("42", "absolute", 42.0),
            ("42.5", "absolute", 42.5),
            ("5.", "absolute", 5.0),
            (".5", "absolute", 0.5),
            (" 5 ", "absolute", 5.0),
            ("1_0", "absolute", 10.0),
        ],
    )
    def test_every_format_parses(self, offset, kind, seconds):
        assert _ipc_seek_offset(offset) == {"kind": kind, "seconds": seconds}

    def test_seconds_is_the_float_object(self):
        # The parsed value is the float float()/int()-arithmetic produced,
        # never a re-typed copy.
        route = _ipc_seek_offset("+10")
        assert type(route["seconds"]) is float
        assert route["seconds"] == 10.0

    def test_special_float_spellings(self):
        assert _ipc_seek_offset("inf") == {"kind": "absolute", "seconds": float("inf")}
        assert _ipc_seek_offset("-inf") == {"kind": "relative", "seconds": float("-inf")}
        nan_route = _ipc_seek_offset("nan")
        assert nan_route["kind"] == "absolute"
        assert math.isnan(nan_route["seconds"])  # nan != nan — compare by predicate

    def test_unicode_digit_and_underscore_forms(self):
        # int()/float() are the builtins: Unicode decimal digits and
        # underscore separators parse exactly like they would at the
        # call site.
        assert _ipc_seek_offset("١:٣٠") == {"kind": "clock", "seconds": 90.0}
        assert _ipc_seek_offset("1_0:30") == {"kind": "clock", "seconds": 630.0}
        assert _ipc_seek_offset(" 1:30 ") == {"kind": "clock", "seconds": 90.0}

    @pytest.mark.parametrize("offset", [None, "", 0, 0.0, False, [], {}, (), set()])
    def test_falsy_values_are_missing(self, offset):
        assert _ipc_seek_offset(offset) == {"kind": "missing"}

    def test_truthiness_gate_precedes_startswith(self):
        class FalsyWithRaisingStartswith:
            def __bool__(self):
                return False

            def startswith(self, *args):
                raise AssertionError("unreachable")

        assert _ipc_seek_offset(FalsyWithRaisingStartswith()) == {"kind": "missing"}

    @pytest.mark.parametrize(
        ("offset", "kind"),
        [
            ("+notanumber", "invalid_offset"),
            ("notanumber", "invalid_offset"),
            ("-1:30", "invalid_offset"),  # signed values never reach the clock branch
            ("1:2:3:4", "invalid_time"),
            ("1:", "invalid_time"),
            (":30", "invalid_time"),
            (":", "invalid_time"),
            ("::", "invalid_time"),
            ("a:b", "invalid_time"),
            ("1:30:", "invalid_time"),
        ],
    )
    def test_invalid_verdicts(self, offset, kind):
        assert _ipc_seek_offset(offset) == {"kind": kind, "offset": offset}

    def test_offset_object_passes_through(self):
        # The invalid rows carry the raw offset for the response message.
        marker = "+nope"
        route = _ipc_seek_offset(marker)
        assert route["offset"] is marker

    def test_non_str_raises_attribute_error(self):
        # A truthy non-str dies on its own startswith lookup, exactly
        # where the reference's first call would.
        with pytest.raises(AttributeError, match="'int' object has no attribute 'startswith'"):
            _ipc_seek_offset(15)

    def test_bytes_raises_type_error(self):
        with pytest.raises(TypeError, match="startswith first arg must be bytes"):
            _ipc_seek_offset(b"+10")

    def test_startswith_short_circuit(self):
        # A value claiming "+" never probes "-" — the reference or's
        # short-circuit.
        class PlusOnly:
            def startswith(self, prefix):
                if prefix == "+":
                    return True
                raise RuntimeError("minus probe must not run")

            def __float__(self):
                return 1.5

        assert _ipc_seek_offset(PlusOnly()) == {"kind": "relative", "seconds": 1.5}

    def test_str_subclass_drives_split(self):
        class S(str):
            def split(self, sep=None, maxsplit=-1):
                return ["1", "30"]

        # The inherited containment still reads the content ("1:30" has
        # the colon); the overridden split drives the parse.
        assert _ipc_seek_offset(S("1:30")) == {"kind": "clock", "seconds": 90.0}

    def test_split_error_propagates(self):
        class S(str):
            def split(self, sep=None, maxsplit=-1):
                raise RuntimeError("split boom")

        with pytest.raises(RuntimeError, match="split boom"):
            _ipc_seek_offset(S("1:30"))

    def test_len_error_inside_the_try(self):
        # len(parts) sits inside the reference's try — a ValueError from
        # it maps to the invalid-time verdict.
        class BadLenParts:
            def __len__(self):
                raise ValueError("len boom")

        class S(str):
            def split(self, sep=None, maxsplit=-1):
                return BadLenParts()

        assert _ipc_seek_offset(S("1:30")) == {
            "kind": "invalid_time",
            "offset": S("1:30"),
        }

    def test_getitem_type_error_propagates(self):
        class BadGetItemParts:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                raise TypeError("getitem boom")

        class S(str):
            def split(self, sep=None, maxsplit=-1):
                return BadGetItemParts()

        with pytest.raises(TypeError, match="getitem boom"):
            _ipc_seek_offset(S("1:30"))

    def test_huge_clock_overflows_in_float(self):
        # float(total) is outside the reference's try — a total too
        # large for a float keeps its OverflowError (a 400-digit minute
        # count times 3600 overflows; a mere 10^29 would not).
        with pytest.raises(OverflowError, match="int too large to convert to float"):
            _ipc_seek_offset("1" + "0" * 400 + ":0")

    def test_fresh_map_every_call(self):
        first = _ipc_seek_offset("+10")
        second = _ipc_seek_offset("+10")
        assert first is not second
        assert first == second


class TestQueueAdd:
    async def test_queue_add_refreshes_queue_page(self):
        h = _fresh_ipc_host()
        h.ytmusic.get_watch_playlist = AsyncMock(
            return_value=[{"videoId": "v1", "title": "T", "artists": []}]
        )
        result = await h._handle_ipc_command("queue_add", {"video_id": "v1"})
        assert result == {"ok": True}
        h.queue.add.assert_called_once()
        assert h.queue.add.call_args.args[0]["video_id"] == "v1"
        h._refresh_queue_page.assert_called_once()

    async def test_unresolved_track_does_not_refresh(self):
        h = _fresh_ipc_host()
        h.ytmusic.get_watch_playlist = AsyncMock(return_value=[])
        result = await h._handle_ipc_command("queue_add", {"video_id": "v1"})
        assert result["ok"] is False
        h.queue.add.assert_not_called()
        h._refresh_queue_page.assert_not_called()


class TestDispatchRoute:
    """The engine behind ``_handle_ipc_command``'s dispatch table."""

    @pytest.mark.parametrize(
        ("command", "branch", "rating"),
        [
            ("play", "play", None),
            ("pause", "pause", None),
            ("next", "next", None),
            ("prev", "prev", None),
            ("seek", "seek", None),
            ("now", "now", None),
            ("status", "status", None),
            ("queue", "queue", None),
            ("queue_add", "queue_add", None),
            ("queue_clear", "queue_clear", None),
            ("like", "rate", "LIKE"),
            ("dislike", "rate", "DISLIKE"),
            ("unlike", "rate", "INDIFFERENT"),
        ],
    )
    def test_every_command_routes_to_its_branch(self, command, branch, rating):
        route = _ipc_dispatch_route(command, _IPC_DISPATCH_TABLE)
        assert route == {"branch": branch, "rating": rating}

    def test_unknown_command(self):
        assert _ipc_dispatch_route("does_not_exist", _IPC_DISPATCH_TABLE) == {
            "branch": "unknown",
            "rating": None,
        }

    @pytest.mark.parametrize(
        "command",
        ["", "PLAY", "Play", "play ", " play", "plays", "lik", "queueadd", "b", "play\x00"],
    )
    def test_near_misses_are_unknown(self, command):
        assert _ipc_dispatch_route(command, _IPC_DISPATCH_TABLE)["branch"] == "unknown"

    @pytest.mark.parametrize(
        "command",
        [b"play", 1, None, True, 1.0, ("play",), ["play"], {"play"}],
    )
    def test_non_str_commands_are_unknown(self, command):
        # No row's literal compares equal — the reference match's own
        # fallthrough, on both backends.
        assert _ipc_dispatch_route(command, _IPC_DISPATCH_TABLE)["branch"] == "unknown"

    def test_the_table_covers_the_reference_case_order(self):
        # The table's rows are the reference match's cases, in its
        # order — a reordering would change which branch wins a
        # pathological always-equal command.
        assert [row[0] for row in _IPC_DISPATCH_TABLE] == [
            "play",
            "pause",
            "next",
            "prev",
            "seek",
            "now",
            "status",
            "queue",
            "queue_add",
            "queue_clear",
            "like",
            "dislike",
            "unlike",
        ]

    def test_stub_command_matching_everything_takes_the_first_row(self):
        class AlwaysEqual:
            def __eq__(self, other):
                return True

            def __hash__(self):
                return hash("stub")

        route = _ipc_dispatch_route(AlwaysEqual(), _IPC_DISPATCH_TABLE)
        assert route == {"branch": "play", "rating": None}

    def test_stub_command_matching_one_literal(self):
        class EqualsLike:
            def __eq__(self, other):
                return other == "like"

            def __hash__(self):
                return hash("stub")

        route = _ipc_dispatch_route(EqualsLike(), _IPC_DISPATCH_TABLE)
        assert route == {"branch": "rate", "rating": "LIKE"}

    def test_fresh_map_every_call(self):
        first = _ipc_dispatch_route("play", _IPC_DISPATCH_TABLE)
        second = _ipc_dispatch_route("play", _IPC_DISPATCH_TABLE)
        assert first is not second
        assert first == second

    def test_inputs_never_mutated(self):
        command = "like"
        table = list(_IPC_DISPATCH_TABLE)
        _ipc_dispatch_route(command, table)
        assert command == "like"
        assert table == list(_IPC_DISPATCH_TABLE)

    def test_stub_table_tokens_pass_through(self):
        # A stub table's branch/rating objects are returned as the
        # objects they are — the engine never re-types them.
        sentinel = object()
        table = [("anything", sentinel, sentinel)]
        route = _ipc_dispatch_route("anything", table)
        assert route["branch"] is sentinel
        assert route["rating"] is sentinel

    def test_generator_table(self):
        table = iter([("play", "play", None)])
        assert _ipc_dispatch_route("play", table) == {"branch": "play", "rating": None}

    def test_short_row_raises_valueerror(self):
        with pytest.raises(ValueError, match="not enough values to unpack"):
            _ipc_dispatch_route("play", [("play", "play")])

    def test_long_row_raises_valueerror(self):
        with pytest.raises(ValueError, match="too many values to unpack"):
            _ipc_dispatch_route("play", [("play", "play", None, "extra")])

    def test_non_iterable_table_raises_typeerror(self):
        # The for-loop's own GET_ITER error — not the unpack rewrite
        # (which only applies to a non-iterable ROW inside the table).
        with pytest.raises(TypeError, match=r"'int' object is not iterable"):
            _ipc_dispatch_route("play", 5)

    def test_non_iterable_row_raises_typeerror(self):
        with pytest.raises(TypeError, match="cannot unpack non-iterable int object"):
            _ipc_dispatch_route("play", [5])

    def test_iter_error_keeps_its_own_error(self):
        class BadIter:
            def __iter__(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _ipc_dispatch_route("play", BadIter())

    def test_first_matching_row_wins(self):
        # Two rows both matching the command — the first one's branch.
        table = [("play", "first", "R1"), ("play", "second", "R2")]
        assert _ipc_dispatch_route("play", table) == {"branch": "first", "rating": "R1"}


class TestDispatchGlue:
    """Every branch's handler calls, through the ported dispatch."""

    async def test_next_calls_play_next(self):
        h = _fresh_ipc_host()
        result = await h._handle_ipc_command("next", {})
        assert result == {"ok": True}
        h._play_next.assert_awaited_once()

    async def test_prev_calls_play_previous(self):
        h = _fresh_ipc_host()
        result = await h._handle_ipc_command("prev", {})
        assert result == {"ok": True}
        h._play_previous.assert_awaited_once()

    async def test_next_with_no_player_returns_error(self):
        h = _fresh_ipc_host()
        h.player = None
        result = await h._handle_ipc_command("next", {})
        assert result == {"ok": False, "error": "player not ready"}

    async def test_prev_with_no_player_returns_error(self):
        h = _fresh_ipc_host()
        h.player = None
        result = await h._handle_ipc_command("prev", {})
        assert result == {"ok": False, "error": "player not ready"}

    @pytest.mark.parametrize(
        ("command", "rating"),
        [("like", "LIKE"), ("dislike", "DISLIKE"), ("unlike", "INDIFFERENT")],
    )
    async def test_rate_commands_pass_their_rating(self, command, rating):
        h = _fresh_ipc_host()
        h._ipc_rate = AsyncMock(return_value={"ok": True})
        result = await h._handle_ipc_command(command, {})
        assert result == {"ok": True}
        h._ipc_rate.assert_awaited_once_with(rating)

    async def test_now_delegates(self):
        h = _fresh_ipc_host()
        h._ipc_now_playing = MagicMock(return_value={"ok": True, "data": None})
        result = await h._handle_ipc_command("now", {})
        assert result == {"ok": True, "data": None}
        h._ipc_now_playing.assert_called_once()

    async def test_status_delegates(self):
        h = _fresh_ipc_host()
        h._ipc_status = MagicMock(return_value={"ok": True, "data": {}})
        result = await h._handle_ipc_command("status", {})
        assert result == {"ok": True, "data": {}}
        h._ipc_status.assert_called_once()

    async def test_queue_delegates(self):
        h = _fresh_ipc_host()
        h._ipc_queue_list = MagicMock(return_value={"ok": True, "data": {}})
        result = await h._handle_ipc_command("queue", {})
        assert result == {"ok": True, "data": {}}
        h._ipc_queue_list.assert_called_once()

    async def test_seek_delegates_with_args(self):
        h = _fresh_ipc_host()
        h._ipc_seek = AsyncMock(return_value={"ok": True})
        result = await h._handle_ipc_command("seek", {"offset": "+10"})
        assert result == {"ok": True}
        h._ipc_seek.assert_awaited_once_with({"offset": "+10"})

    async def test_queue_add_delegates_with_args(self):
        h = _fresh_ipc_host()
        h._ipc_queue_add = AsyncMock(return_value={"ok": True})
        result = await h._handle_ipc_command("queue_add", {"video_id": "v1"})
        assert result == {"ok": True}
        h._ipc_queue_add.assert_awaited_once_with({"video_id": "v1"})

    async def test_engine_error_is_caught_by_the_command_guard(self):
        # A pathological command whose __eq__ raises surfaces through
        # the mixin's except as the error response, exactly like the
        # reference match's first comparison would have.
        class RaisingEq:
            def __eq__(self, other):
                raise RuntimeError("eq boom")

            def __hash__(self):
                return hash("stub")

        h = _fresh_ipc_host()
        result = await h._handle_ipc_command(RaisingEq(), {})
        assert result["ok"] is False
        assert "eq boom" in result["error"]

    async def test_handler_error_is_caught_by_the_command_guard(self):
        h = _fresh_ipc_host()
        h._play_next = AsyncMock(side_effect=RuntimeError("handler boom"))
        result = await h._handle_ipc_command("next", {})
        assert result["ok"] is False
        assert "handler boom" in result["error"]


# ── Native vs fallback parity ──────────────────────────────────────────

# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable: plain command strings and
# JSON scalars, plus the stub shapes spelled out as specs both sides
# build identically.
_ROUTE_PARITY_CASES = [
    {"t": "str", "v": "play"},
    {"t": "str", "v": "pause"},
    {"t": "str", "v": "next"},
    {"t": "str", "v": "prev"},
    {"t": "str", "v": "seek"},
    {"t": "str", "v": "now"},
    {"t": "str", "v": "status"},
    {"t": "str", "v": "queue"},
    {"t": "str", "v": "queue_add"},
    {"t": "str", "v": "queue_clear"},
    {"t": "str", "v": "like"},
    {"t": "str", "v": "dislike"},
    {"t": "str", "v": "unlike"},
    {"t": "str", "v": "does_not_exist"},
    {"t": "str", "v": ""},
    {"t": "str", "v": "PLAY"},
    {"t": "str", "v": "play "},
    {"t": "bytes", "v": "cGxBWQ=="},  # b"play" — never equal to the literal
    {"t": "json", "v": 1},
    {"t": "json", "v": None},
    {"t": "json", "v": True},
    {"t": "json", "v": 1.0},
    {"t": "json", "v": ["play"]},
    {"t": "always_eq"},
    {"t": "like_eq"},
    {"t": "raising_eq"},
]

_TABLE_PARITY_CASES = [
    {"t": "real"},  # the module's own production table
    {"t": "stub", "rows": [["play", "first", "R1"], ["play", "second", "R2"]]},
    {"t": "stub", "rows": [["anything", "rate", "LIKE"]]},
    {"t": "gen", "rows": [["seek", "seek", None], ["like", "rate", "LIKE"]]},
    {"t": "short_row", "rows": [["play", "play"]]},
    {"t": "long_row", "rows": [["play", "play", None, "extra"]]},
    {"t": "int"},  # a non-iterable table
    {"t": "int_rows", "rows": [5]},
    {"t": "bad_iter"},
]


def _build_command(spec: dict) -> Any:
    kind = spec["t"]
    if kind == "str":
        return spec["v"]
    if kind == "bytes":
        return spec["v"].encode("latin-1")
    if kind == "json":
        return spec["v"]
    if kind == "always_eq":
        return _AlwaysEqual()
    if kind == "like_eq":
        return _EqualsLike()
    if kind == "raising_eq":
        return _RaisingEq()
    raise AssertionError(kind)


class _AlwaysEqual:
    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return hash("stub")


class _EqualsLike:
    def __eq__(self, other: object) -> bool:
        return other == "like"

    def __hash__(self) -> int:
        return hash("stub")


class _RaisingEq:
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("eq boom")

    def __hash__(self) -> int:
        return hash("stub")


class _BadIter:
    def __iter__(self):
        raise RuntimeError("iter boom")


def _build_table(spec: dict) -> Any:
    kind = spec["t"]
    if kind == "real":
        return _IPC_DISPATCH_TABLE
    if kind == "stub":
        return [tuple(row) for row in spec["rows"]]
    if kind == "gen":
        return iter([tuple(row) for row in spec["rows"]])
    if kind == "short_row":
        return [tuple(row) for row in spec["rows"]]
    if kind == "long_row":
        return [tuple(row) for row in spec["rows"]]
    if kind == "int":
        return 5
    if kind == "int_rows":
        return spec["rows"]
    if kind == "bad_iter":
        return _BadIter()
    raise AssertionError(kind)


def _outcome(command: Any, table: Any) -> dict:
    try:
        return {"ok": _ipc_dispatch_route(command, table)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._ipc import _IPC_DISPATCH_TABLE, _ipc_dispatch_route


class AlwaysEqual:
    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash("stub")


class EqualsLike:
    def __eq__(self, other):
        return other == "like"

    def __hash__(self):
        return hash("stub")


class RaisingEq:
    def __eq__(self, other):
        raise RuntimeError("eq boom")

    def __hash__(self):
        return hash("stub")


class BadIter:
    def __iter__(self):
        raise RuntimeError("iter boom")


def build_command(spec):
    kind = spec["t"]
    if kind == "str":
        return spec["v"]
    if kind == "bytes":
        return spec["v"].encode("latin-1")
    if kind == "json":
        return spec["v"]
    if kind == "always_eq":
        return AlwaysEqual()
    if kind == "like_eq":
        return EqualsLike()
    if kind == "raising_eq":
        return RaisingEq()
    raise AssertionError(kind)


def build_table(spec):
    kind = spec["t"]
    if kind == "real":
        return _IPC_DISPATCH_TABLE
    if kind in ("stub", "short_row", "long_row"):
        return [tuple(row) for row in spec["rows"]]
    if kind == "gen":
        return iter([tuple(row) for row in spec["rows"]])
    if kind == "int":
        return 5
    if kind == "int_rows":
        return spec["rows"]
    if kind == "bad_iter":
        return BadIter()
    raise AssertionError(kind)


def outcome(command, table):
    try:
        return {"ok": _ipc_dispatch_route(command, table)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
routes = [
    outcome(build_command(cspec), build_table(tspec))
    for cspec in payload["commands"]
    for tspec in payload["tables"]
]
json.dump(routes, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same branch, same rating
    (object pass-through included), same errors with the same messages
    — for every command × table combination, so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_ipc_dispatch_route_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps({"commands": _ROUTE_PARITY_CASES, "tables": _TABLE_PARITY_CASES}),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [
            _outcome(_build_command(cspec), _build_table(tspec))
            for cspec in _ROUTE_PARITY_CASES
            for tspec in _TABLE_PARITY_CASES
        ]
        cases = [(cspec, tspec) for cspec in _ROUTE_PARITY_CASES for tspec in _TABLE_PARITY_CASES]
        for (cspec, tspec), n, f in zip(cases, native, fallback):
            assert n == f, (
                f"native and fallback route outcomes differ for "
                f"command {cspec!r} x table {tspec!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_ipc_seek_offset_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _SEEK_CHILD_PROG],
            input=json.dumps(_SEEK_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_seek_outcome(spec) for spec in _SEEK_PARITY_CASES]
        for spec, n, f in zip(_SEEK_PARITY_CASES, native, fallback):
            # Both sides re-serialized before comparison so nan/inf
            # outcomes compare by their JSON spellings (nan != nan).
            assert json.dumps(n) == json.dumps(f), (
                f"native and fallback seek outcomes differ for {spec!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )


# ── Seek-offset native vs fallback parity ──────────────────────────────

# The matrix below is shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable: offset strings and JSON
# scalars, plus the stub shapes spelled out as specs both sides build
# identically.
_SEEK_PARITY_CASES = [
    # The three accepted formats, in their spellings.
    {"t": "str", "v": "+15"},
    {"t": "str", "v": "-10"},
    {"t": "str", "v": "+0"},
    {"t": "str", "v": "-0"},
    {"t": "str", "v": "+1e1"},
    {"t": "str", "v": "-1.5e-2"},
    {"t": "str", "v": "1:30"},
    {"t": "str", "v": "0:0"},
    {"t": "str", "v": "01:30"},
    {"t": "str", "v": "90:00"},
    {"t": "str", "v": "1:00:00"},
    {"t": "str", "v": "1:30:60"},
    {"t": "str", "v": "42"},
    {"t": "str", "v": "42.5"},
    {"t": "str", "v": ".5"},
    {"t": "str", "v": " 5 "},
    {"t": "str", "v": "1_0"},
    {"t": "str", "v": "1_0:30"},
    {"t": "str", "v": " 1:30 "},
    {"t": "str", "v": "١:٣٠"},
    {"t": "str", "v": "inf"},
    {"t": "str", "v": "-inf"},
    {"t": "str", "v": "nan"},
    # The verdicts.
    {"t": "str", "v": "+notanumber"},
    {"t": "str", "v": "notanumber"},
    {"t": "str", "v": "-1:30"},  # signed values never reach the clock branch
    {"t": "str", "v": "1:2:3:4"},
    {"t": "str", "v": "1:"},
    {"t": "str", "v": ":30"},
    {"t": "str", "v": ":"},
    {"t": "str", "v": "::"},
    {"t": "str", "v": "a:b"},
    {"t": "str", "v": "9999999999999999999999999999:0"},  # float(total) overflows
    # Falsy values are missing, before any attribute is touched.
    {"t": "str", "v": ""},
    {"t": "json", "v": None},
    {"t": "json", "v": 0},
    {"t": "json", "v": 0.0},
    {"t": "json", "v": False},
    {"t": "json", "v": []},
    {"t": "json", "v": {}},
    # Truthy non-str values die on their own startswith lookup.
    {"t": "json", "v": 15},
    {"t": "json", "v": 1.5},
    {"t": "json", "v": True},
    {"t": "bytes", "v": "KzEw"},  # b"+10" — the str-prefixed startswith TypeError
    # Stubs driving the protocol: the short-circuit, the contains gate,
    # the split-driven clock, and the raising edges.
    {"t": "plus_only"},
    {"t": "no_sign"},
    {"t": "clock_stub"},
    {"t": "raising_startswith"},
    {"t": "raising_contains"},
]


class _PlusOnly:
    def startswith(self, prefix):
        if prefix == "+":
            return True
        raise RuntimeError("minus probe must not run")

    def __float__(self):
        return 1.5


class _NoSign:
    def startswith(self, prefix):
        return False

    def __contains__(self, item):
        return False

    def __float__(self):
        return 42.5


class _ClockStub:
    def startswith(self, prefix):
        return False

    def __contains__(self, item):
        return True

    def split(self, sep=None, maxsplit=-1):
        return ["1", "30"]


class _RaisingStartswith:
    def startswith(self, prefix):
        raise RuntimeError("sw boom")


class _RaisingContains:
    def startswith(self, prefix):
        return False

    def __contains__(self, item):
        raise RuntimeError("contains boom")


def _build_seek_offset(spec: dict) -> Any:
    kind = spec["t"]
    if kind == "str":
        return spec["v"]
    if kind == "json":
        return spec["v"]
    if kind == "bytes":
        return spec["v"].encode("latin-1")
    if kind == "plus_only":
        return _PlusOnly()
    if kind == "no_sign":
        return _NoSign()
    if kind == "clock_stub":
        return _ClockStub()
    if kind == "raising_startswith":
        return _RaisingStartswith()
    if kind == "raising_contains":
        return _RaisingContains()
    raise AssertionError(kind)


def _seek_outcome(spec: dict) -> dict:
    try:
        route = _ipc_seek_offset(_build_seek_offset(spec))
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}
    if "offset" in route and not isinstance(route["offset"], str):
        # A stub's offset object is not JSON-transportable — its type
        # name is (the object pass-through itself is pinned by the unit
        # tests above).
        route = {**route, "offset": f"<{type(route['offset']).__name__}>"}
    return {"ok": route}


_SEEK_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._ipc import _ipc_seek_offset


class PlusOnly:
    def startswith(self, prefix):
        if prefix == '+':
            return True
        raise RuntimeError('minus probe must not run')

    def __float__(self):
        return 1.5


class NoSign:
    def startswith(self, prefix):
        return False

    def __contains__(self, item):
        return False

    def __float__(self):
        return 42.5


class ClockStub:
    def startswith(self, prefix):
        return False

    def __contains__(self, item):
        return True

    def split(self, sep=None, maxsplit=-1):
        return ['1', '30']


class RaisingStartswith:
    def startswith(self, prefix):
        raise RuntimeError('sw boom')


class RaisingContains:
    def startswith(self, prefix):
        return False

    def __contains__(self, item):
        raise RuntimeError('contains boom')


def build(spec):
    kind = spec['t']
    if kind == 'str':
        return spec['v']
    if kind == 'json':
        return spec['v']
    if kind == 'bytes':
        return spec['v'].encode('latin-1')
    if kind == 'plus_only':
        return PlusOnly()
    if kind == 'no_sign':
        return NoSign()
    if kind == 'clock_stub':
        return ClockStub()
    if kind == 'raising_startswith':
        return RaisingStartswith()
    if kind == 'raising_contains':
        return RaisingContains()
    raise AssertionError(kind)


def outcome(spec):
    try:
        route = _ipc_seek_offset(build(spec))
    except Exception as exc:
        return {'error': '%s: %s' % (type(exc).__name__, exc)}
    if 'offset' in route and not isinstance(route['offset'], str):
        route = dict(route, offset='<%s>' % type(route['offset']).__name__)
    return {'ok': route}


payload = json.load(sys.stdin)
json.dump([outcome(spec) for spec in payload], sys.stdout)
"""
