"""Discord Rich Presence integration for ytm-player.

Shows the currently playing track in the user's Discord status.
Requires the optional `pypresence` package.

The presence-kwargs assembly ``_presence_kwargs`` (the seven-key dict
behind ``update``'s ``rpc.update(**kwargs)`` — the fallback-text
arms, the 128 cuts, the ``int`` timestamp casts and the end gate) is
implemented by the Rust extension
(``_ytm_player_native.discord_presence_kwargs``) when it is built
and by the pure-Python transcription below otherwise; the clock
read and the ``_start_time`` state write, the pypresence imports,
the reconnect loop and the RPC calls stay Python-side in the
service.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Default Discord application ID — the bundled "YouTube Music" Rich Presence
# app. Users can point at their own app via [discord] client_id in config.
DEFAULT_DISCORD_CLIENT_ID = "1517429270115258490"


def _presence_kwargs(
    title: Any,
    artist: Any,
    album: Any,
    duration: Any,
    thumbnail_url: Any,
    start_time: Any,
    activity_type: Any,
) -> dict:
    """The presence-kwargs assembly behind ``update``.

    The dict handed to ``rpc.update(**kwargs)``: the fallback-text
    arms (a falsy title/artist/album answers ``Unknown Track`` /
    ``Unknown Artist`` / ``YouTube Music``), the 128-char cuts on the
    three text fields (``large_image`` passes through unsliced), the
    ``int(start_time)`` timestamp, and — only when ``duration > 0`` —
    the ``int(start_time + duration)`` end edge. Every truthiness
    gate, slice, comparison, addition and ``int()`` cast is
    CPython's own, so the error shapes and the bool spellings stay
    the interpreter's; the dict carries the reference's own key
    order.

    Implemented by the Rust extension
    (``_ytm_player_native.discord_presence_kwargs``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    kwargs: dict = {
        "activity_type": activity_type,
        "details": (title or "Unknown Track")[:128],
        "state": (artist or "Unknown Artist")[:128],
        "large_image": thumbnail_url or "ytm_icon",
        "large_text": (album or "YouTube Music")[:128],
        "start": int(start_time),
    }

    if duration > 0:
        kwargs["end"] = int(start_time + duration)

    return kwargs


class DiscordRPC:
    """Manages Discord Rich Presence connection and updates.

    Gracefully handles Discord not running or pypresence not installed.
    Reconnects automatically if the connection drops.
    """

    def __init__(self, client_id: str = "") -> None:
        # Blank/whitespace falls back to the bundled app, so an empty config
        # value doesn't break RPC; `enabled = false` is the real off switch.
        self._client_id = client_id.strip() or DEFAULT_DISCORD_CLIENT_ID
        self._rpc: object | None = None
        self._connected = False
        self._start_time: float = 0
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Attempt to connect to Discord. Returns True on success."""
        async with self._lock:
            return await self._connect_unlocked()

    async def _connect_unlocked(self) -> bool:
        if self._connected and self._rpc:
            return True

        try:
            # pypresence's stubs don't export AioPresence but it exists at
            # runtime. The except ImportError below handles missing-package.
            from pypresence import AioPresence  # type: ignore[attr-defined]
        except ImportError:
            logger.info("pypresence not installed — Discord RPC disabled")
            return False

        try:
            self._rpc = AioPresence(self._client_id)
            await self._rpc.connect()  # type: ignore[union-attr]
            self._connected = True
            logger.info("Connected to Discord Rich Presence")
            return True
        except Exception:
            logger.debug("Could not connect to Discord (not running?)", exc_info=True)
            self._rpc = None
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Discord."""
        async with self._lock:
            await self._reset_connection_unlocked()

    async def _reset_connection_unlocked(self) -> None:
        rpc = self._rpc
        self._rpc = None
        self._connected = False
        if not rpc:
            return
        try:
            result = rpc.close()  # type: ignore[attr-defined]
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.debug("Failed to close Discord RPC connection", exc_info=True)

    async def _call_with_reconnect(
        self, operation_name: str, operation: Callable[[object], Awaitable[Any]]
    ) -> None:
        async with self._lock:
            if not await self._connect_unlocked():
                return

            for attempt in range(2):
                rpc = self._rpc
                if not rpc:
                    return
                try:
                    await operation(rpc)
                    self._connected = True
                    return
                except Exception:
                    logger.debug("Failed to %s", operation_name, exc_info=True)
                    await self._reset_connection_unlocked()
                    if attempt == 0 and await self._connect_unlocked():
                        continue
                    return

    async def update(
        self,
        title: str,
        artist: str,
        album: str = "",
        duration: int = 0,
        position: float = 0,
        thumbnail_url: str = "",
    ) -> None:
        """Update the Discord presence with current track info.

        The kwargs handed to the RPC are resolved by the
        ``_presence_kwargs`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise); this method
        reads the clock, keeps the start-time state and owns the
        pypresence import.
        """
        self._start_time = time.time() - position

        try:
            from pypresence import ActivityType  # type: ignore[attr-defined]

            activity_type = ActivityType.LISTENING
        except (ImportError, AttributeError):
            activity_type = None

        kwargs = _presence_kwargs(
            title, artist, album, duration, thumbnail_url, self._start_time, activity_type
        )

        async def _update(rpc: object) -> Any:
            return await rpc.update(**kwargs)  # type: ignore[attr-defined]

        await self._call_with_reconnect("update Discord presence", _update)

    async def clear(self) -> None:
        """Clear the Discord presence (paused/stopped)."""

        async def _clear(rpc: object) -> Any:
            return await rpc.clear()  # type: ignore[attr-defined]

        await self._call_with_reconnect("clear Discord presence", _clear)

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same presence-kwargs assembly; this
# binding replaces the pure-Python transcription above when the extension
# is importable.  The rebinding must happen AFTER the definition above —
# assigning the native function any earlier would be clobbered by the
# ``def`` statement.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        discord_presence_kwargs as _native_discord_presence_kwargs,
    )
except ImportError:
    pass
else:
    _presence_kwargs = _native_discord_presence_kwargs  # type: ignore[assignment]
