"""Tests for IPCMixin._handle_ipc_command dispatch + seek parsing.

The CLI shells (`ytm play`, `ytm seek 1:30`, etc.) all hit
_handle_ipc_command via the Unix socket. These tests cover the
no-player guards, unknown-command fallthrough, and the three seek
formats (relative, mm:ss, absolute seconds).

``_ipc_dispatch_route`` is the engine behind the dispatch: the table
walk deciding which branch handles each command string, and the rating
the like/dislike/unlike branches rate with. The pure-Python
transcription lives in ``app/_ipc.py``; when the Rust extension is
built the same engine runs natively
(``_ytm_player_native.ipc_dispatch_route``). The unit tests below
assert the engine's contract on whichever backend is loaded, plus the
mixin glue (every branch's handler calls, including the rating the
rate branches pass). When the extension is built, the parity test
additionally compares the native engine against the pure-Python
reference (run in a child interpreter with the extension import
blocked) case for case — including the error shapes, where CPython's
own messages must surface identically — so the two implementations
cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ytm_player.app._ipc import (
    _IPC_DISPATCH_TABLE,
    IPCMixin,
    _ipc_dispatch_route,
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
