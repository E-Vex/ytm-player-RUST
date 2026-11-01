"""IPC command handling mixin for YTMPlayerApp.

Two pure engines are split out of the mixin. The dispatch table —
which branch of ``_handle_ipc_command``'s match handles each IPC
command string, and the rating the like/dislike/unlike branches rate
with — is ``_ipc_dispatch_route``; the seek-offset parser — the
classification of ``args["offset"]`` into relative/clock/absolute
seconds with its missing/invalid verdicts — is ``_ipc_seek_offset``.
Both are implemented by the Rust extension
(``_ytm_player_native.ipc_dispatch_route`` /
``_ytm_player_native.ipc_seek_offset``) when it is built and by the
pure-Python transcriptions below otherwise. The mixin resolves the
routes and performs the branches' calls; the awaited handler calls
remain pure mixin concerns.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from ytm_player.app._base import YTMHostBase

logger = logging.getLogger(__name__)

# The dispatch table behind ``_handle_ipc_command`` — one row per
# ``case`` of the reference match, in its order, each carrying the
# command literal, the branch token the mixin applies, and the rating
# payload (the like/dislike/unlike branches' argument; ``None``
# everywhere else). The reference match's literal value patterns compile
# to ``command == literal`` comparisons (subject on the left) evaluated
# lazily case by case, so the walk below — first row whose member
# compares equal wins — is the table's exact semantics.
_IPC_DISPATCH_TABLE: tuple[tuple[str, str, str | None], ...] = (
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
)


def _ipc_dispatch_route(
    command: Any,
    table: tuple[tuple[str, str, str | None], ...],
) -> dict[str, Any]:
    """The dispatch table behind ``IPCMixin._handle_ipc_command``.

    The pure half of the dispatch: which branch handles *command* (and,
    for the rate branches, the rating to rate with). Each row of *table*
    is compared in order with ``==`` (subject on the left, exactly like
    the reference ``match``'s lazily evaluated literal value patterns,
    which compile to ``command == literal`` comparisons — first equal
    row wins) and anything no row claims is unknown. The branch tokens
    name the handler the mixin applies; the like/dislike/unlike rows
    share the "rate" branch, whose rating argument the row carries.

    Implemented by the Rust extension
    (``_ytm_player_native.ipc_dispatch_route``) when it is built; this
    pure-Python transcription is the fallback otherwise. Returns a
    fresh ``{"branch": str, "rating": str | None}`` map — ``"unknown"``
    when no row matches — and never mutates *command* or *table*.
    """
    for member, branch, rating in table:
        if command == member:
            return {"branch": branch, "rating": rating}
    return {"branch": "unknown", "rating": None}


def _ipc_seek_offset(offset: Any) -> dict[str, Any]:
    """The seek-offset parser behind ``IPCMixin._ipc_seek``.

    The pure half of the seek command: classifying the offset value
    (whatever ``args.get("offset", "")`` returned) and computing the
    seconds it names. A falsy value (the missing key's ``""`` default,
    ``None``, ``0``, every empty container) is missing — gated before
    any attribute is touched. A value starting with ``+`` or ``-`` (two
    ``startswith`` calls, short-circuited) is relative and parsed with
    ``float()``; a value containing ``:`` is a clock — split on ``:``
    and parsed as ``mm:ss`` (2 parts) or ``hh:mm:ss`` (3 parts) with
    ``int()`` parts combined by the reference's ``*``/``+`` arithmetic
    — and everything else is absolute seconds parsed with ``float()``.
    A ``ValueError`` from the numeric parses maps to the matching
    invalid verdict; every other error keeps its own shape and
    propagates (a non-str value's ``AttributeError`` from
    ``startswith``, a bytes value's ``TypeError``, the ``OverflowError``
    from ``float(total)`` on a clock too large for one).

    Implemented by the Rust extension
    (``_ytm_player_native.ipc_seek_offset``) when it is built; this
    pure-Python transcription is the fallback otherwise. Returns a
    fresh ``{"kind": str, …}`` map — ``"missing"``,
    ``"relative"``/``"clock"``/``"absolute"`` with the parsed
    ``seconds`` float, or ``"invalid_offset"``/``"invalid_time"``
    carrying the raw ``offset`` — and never mutates *offset*.
    """
    if not offset:
        return {"kind": "missing"}
    if offset.startswith("+") or offset.startswith("-"):
        try:
            return {"kind": "relative", "seconds": float(offset)}
        except ValueError:
            return {"kind": "invalid_offset", "offset": offset}
    if ":" in offset:
        parts = offset.split(":")
        try:
            if len(parts) == 2:
                total = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                total = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                return {"kind": "invalid_time", "offset": offset}
        except ValueError:
            return {"kind": "invalid_time", "offset": offset}
        return {"kind": "clock", "seconds": float(total)}
    try:
        return {"kind": "absolute", "seconds": float(offset)}
    except ValueError:
        return {"kind": "invalid_offset", "offset": offset}


class IPCMixin(YTMHostBase):
    """Handles IPC commands from the CLI."""

    async def _handle_ipc_command(self, command: str, args: dict) -> dict:
        """Dispatch an IPC command from the CLI and return a response dict.

        The dispatch table — which branch handles which command string,
        in the reference's case order — is resolved by the
        ``_ipc_dispatch_route`` engine above (native when the Rust
        extension is built, the pure-Python transcription otherwise);
        this method performs each branch's calls, with the rating the
        rate branches rate with carried by the matched row.
        """
        try:
            route = _ipc_dispatch_route(command, _IPC_DISPATCH_TABLE)

            match route["branch"]:
                case "play":
                    if not self.player:
                        return {"ok": False, "error": "player not ready"}
                    await self.player.resume()
                    return {"ok": True}

                case "pause":
                    if not self.player:
                        return {"ok": False, "error": "player not ready"}
                    await self.player.pause()
                    return {"ok": True}

                case "next":
                    if not self.player:
                        return {"ok": False, "error": "player not ready"}
                    await self._play_next()
                    return {"ok": True}

                case "prev":
                    if not self.player:
                        return {"ok": False, "error": "player not ready"}
                    await self._play_previous()
                    return {"ok": True}

                case "seek":
                    return await self._ipc_seek(args)

                case "now":
                    return self._ipc_now_playing()

                case "status":
                    return self._ipc_status()

                case "queue":
                    return self._ipc_queue_list()

                case "queue_add":
                    return await self._ipc_queue_add(args)

                case "queue_clear":
                    self.queue.clear()
                    self._refresh_queue_page()
                    return {"ok": True}

                case "rate":
                    return await self._ipc_rate(cast("str", route["rating"]))

                case _:
                    return {"ok": False, "error": f"unknown command: {command}"}
        except Exception as exc:
            logger.exception("IPC command '%s' failed", command)
            return {"ok": False, "error": str(exc)}

    async def _ipc_rate(self, rating: str) -> dict:
        """Rate the currently playing track (LIKE / DISLIKE / INDIFFERENT)."""
        if (
            not self.player
            or not self.player.current_track
            or not self.player.current_track.get("video_id")
        ):
            return {"ok": False, "error": "no track is playing"}
        if not self.ytmusic:
            return {"ok": False, "error": "ytmusic not initialized"}
        result = await self.ytmusic.rate_song(self.player.current_track["video_id"], rating)
        if result == "success":
            return {"ok": True}
        return {"ok": False, "error": result}

    async def _ipc_seek(self, args: dict) -> dict:
        """Handle seek IPC command. Accepts relative (+10, -10) or absolute (1:30).

        The offset parse — the classification of ``args["offset"]``
        into relative/clock/absolute seconds with its missing/invalid
        verdicts — is resolved by the ``_ipc_seek_offset`` engine above
        (native when the Rust extension is built, the pure-Python
        transcription otherwise); this method performs the seek calls
        the parsed kind names and builds the response messages.
        """
        if not self.player:
            return {"ok": False, "error": "player not ready"}

        route = _ipc_seek_offset(args.get("offset", ""))
        match route["kind"]:
            case "missing":
                return {"ok": False, "error": "missing offset"}
            case "relative":
                await self.player.seek(route["seconds"])
            case "clock" | "absolute":
                await self.player.seek_absolute(route["seconds"])
            case "invalid_offset":
                return {"ok": False, "error": f"invalid offset: {route['offset']}"}
            case _:
                return {"ok": False, "error": f"invalid time format: {route['offset']}"}
        return {"ok": True}

    def _ipc_now_playing(self) -> dict:
        """Return current track info and position."""
        if not self.player or not self.player.current_track:
            return {"ok": True, "data": None}

        track = self.player.current_track
        return {
            "ok": True,
            "data": {
                "track": track,
                "position": self.player.position,
                "duration": self.player.duration,
                "is_playing": self.player.is_playing,
                "is_paused": self.player.is_paused,
            },
        }

    def _ipc_status(self) -> dict:
        """Return full player state."""
        playing = False
        paused = False
        volume = 0
        position = 0.0
        duration = 0.0
        track = None

        if self.player:
            playing = self.player.is_playing
            paused = self.player.is_paused
            volume = self.player.volume
            position = self.player.position
            duration = self.player.duration
            track = self.player.current_track

        return {
            "ok": True,
            "data": {
                "track": track,
                "is_playing": playing,
                "is_paused": paused,
                "volume": volume,
                "position": position,
                "duration": duration,
                "repeat": self.queue.repeat_mode.value,
                "shuffle": self.queue.shuffle_enabled,
                "queue_length": self.queue.length,
            },
        }

    def _ipc_queue_list(self) -> dict:
        """Return the current queue as a list of tracks."""
        return {
            "ok": True,
            "data": {
                "tracks": list(self.queue.tracks),
                "current_index": self.queue.current_index,
                "length": self.queue.length,
                "repeat": self.queue.repeat_mode.value,
                "shuffle": self.queue.shuffle_enabled,
            },
        }

    async def _ipc_queue_add(self, args: dict) -> dict:
        """Resolve a video_id via ytmusic and add to queue."""
        video_id = args.get("video_id", "")
        if not video_id:
            return {"ok": False, "error": "missing video_id"}

        if not self.ytmusic:
            return {"ok": False, "error": "ytmusic not initialized"}

        # Use get_watch_playlist -- it returns tracks in the flat format
        # that normalize_tracks() expects (unlike get_song() which returns
        # a nested videoDetails structure).
        try:
            watch_tracks = await self.ytmusic.get_watch_playlist(video_id)
        except Exception as exc:
            return {"ok": False, "error": f"failed to resolve track: {exc}"}

        if not watch_tracks:
            return {"ok": False, "error": f"track not found: {video_id}"}

        from ytm_player.utils.formatting import normalize_tracks

        normalized = normalize_tracks(watch_tracks[:1])
        if normalized:
            self.queue.add(normalized[0])
            self._refresh_queue_page()
            return {"ok": True}
        return {"ok": False, "error": "failed to normalize track"}


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python transcriptions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements (the
# lesson that produced formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ipc_dispatch_route as _native_ipc_dispatch_route,
    )
    from ytm_player.utils._ytm_player_native import (
        ipc_seek_offset as _native_ipc_seek_offset,
    )
except ImportError:
    pass
else:
    _ipc_dispatch_route = _native_ipc_dispatch_route  # type: ignore[assignment]
    _ipc_seek_offset = _native_ipc_seek_offset  # type: ignore[assignment]
