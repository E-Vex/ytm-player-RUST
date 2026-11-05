"""Windows media key listener using pynput.

Uses pynput to capture global media key events (play/pause, next, previous)
and dispatch them to the app's playback callbacks. Linux uses MPRIS (D-Bus),
and macOS uses native event-tap + Now Playing services.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ytm_player.services._dispatch import (
    PlayerCallback,
    capture_dispatch_context,
    dispatch_coro_threadsafe,
)

logger = logging.getLogger(__name__)

try:
    from pynput.keyboard import Key, Listener  # type: ignore[import-not-found]

    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False
    Key = None  # type: ignore[assignment,misc]
    Listener = None  # type: ignore[assignment,misc]

# Map pynput media Key enum members to callback action names (same as MPRIS).
# Uses the Key enum directly — not str() — because pynput delivers Key enum
# instances for special keys and they're hashable singletons.
_KEY_MAP: dict[Any, str] = {}
if _PYNPUT_AVAILABLE and Key is not None:
    _KEY_MAP = {
        Key.media_play_pause: "play_pause",
        Key.media_next: "next",
        Key.media_previous: "previous",
    }


class MediaKeysService:
    """Listens for global media key presses on Windows."""

    def __init__(self) -> None:
        self._listener: Any | None = None
        self._callbacks: dict[str, PlayerCallback] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    async def start(
        self,
        callbacks: dict[str, PlayerCallback],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Begin listening for media key events.

        *callbacks* maps action names (play_pause, next, previous) to async
        functions — the same dict that MPRISService uses.
        """
        if not _PYNPUT_AVAILABLE:
            logger.debug("pynput is not installed — media keys disabled")
            return

        self._callbacks = callbacks
        self._loop = loop
        # start() runs on the app's loop thread: snapshot its context so
        # dispatched key callbacks keep Textual's active_app ContextVar.
        capture_dispatch_context()

        # Suppress pynput's own "This process is not trusted!" warning —
        # we emit a friendlier message ourselves.
        logging.getLogger("pynput").setLevel(logging.ERROR)

        listener_cls = Listener
        if listener_cls is None:
            logger.debug("pynput listener unavailable — media keys disabled")
            return

        listener = listener_cls(on_press=self._on_press)
        listener.daemon = True
        listener.start()
        self._listener = listener
        self._running = True
        logger.info("Media key listener started")

    def stop(self) -> None:
        """Stop the media key listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        self._running = False
        logger.info("Media key listener stopped")

    def _on_press(self, key: Any) -> None:
        """Handle a key press event from pynput (runs on listener thread)."""
        if not self._running or not self._loop:
            return

        action_name = _KEY_MAP.get(key)
        if action_name is None:
            return

        callback = self._callbacks.get(action_name)
        if callback is None:
            return

        # Dispatch to the asyncio event loop from pynput's thread.
        dispatch_coro_threadsafe(self._loop, callback)
