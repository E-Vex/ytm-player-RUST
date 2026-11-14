"""
Purpose: Capture macOS hardware media keys for terminal ytm-player sessions.
Interface: MacOSEventTapService.start/stop registers and tears down a global event tap.
Invariants: Only media-key key-down events are intercepted; all other events pass through.
Decisions: Swallow handled media keys so Apple Music does not auto-launch while ytm-player is active.
"""
# pyright: reportAttributeAccessIssue=false, reportOptionalMemberAccess=false

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from ytm_player.services._dispatch import (
    PlayerCallback,
    capture_dispatch_context,
    dispatch_coro_threadsafe,
)

logger = logging.getLogger(__name__)

try:
    import AppKit  # type: ignore[reportMissingImports]
    import Quartz  # type: ignore[reportMissingImports]

    _EVENT_TAP_AVAILABLE = True
except ImportError:
    AppKit = None
    Quartz = None
    _EVENT_TAP_AVAILABLE = False

_MEDIA_EVENT_SUBTYPE = 8
_MEDIA_KEY_DOWN_STATE = 0xA

_PLAY_PAUSE_KEY = 16
_NEXT_KEY = 17
_PREVIOUS_KEY = 18
_FAST_FORWARD_KEY = 19
_REWIND_KEY = 20

_KEY_TO_ACTION = {
    _PLAY_PAUSE_KEY: "play_pause",
    _NEXT_KEY: "next",
    _PREVIOUS_KEY: "previous",
    _FAST_FORWARD_KEY: "next",
    _REWIND_KEY: "previous",
}


def _key_action(data1: int) -> str | None:
    """The media-key classification — the ``action`` engine.

    The pure core of ``_event_action``: the NSEvent ``data1`` word's
    bit unpack — the key code in the high 16 bits, the flags in the
    low 16 with the key state in bits 8-15 — the media-key-down state
    gate, and the key-code → action map (16 play/pause, 17 next, 18
    previous, 19 fast-forward → next, 20 rewind → previous).

    Implemented by the Rust extension
    (``_ytm_player_native.macos_event_tap_action``) when it is built;
    this pure-Python transcription is the fallback otherwise. The
    ``ns_event`` None gate, the ``subtype()`` comparison against the
    raw framework value (not int()-coerced — the reference compares
    whatever ``subtype()`` returns), the ``int()`` coercion of
    ``data1()`` and the broad ``except`` around the attribute
    extraction stay with the caller.
    """
    key_code = (data1 & 0xFFFF0000) >> 16
    key_flags = data1 & 0x0000FFFF
    key_state = (key_flags & 0xFF00) >> 8
    if key_state != _MEDIA_KEY_DOWN_STATE:
        return None

    return _KEY_TO_ACTION.get(key_code)


def _event_action(ns_event: Any) -> str | None:
    """Return callback action for a media key event, else None."""
    if ns_event is None:
        return None
    try:
        if ns_event.subtype() != _MEDIA_EVENT_SUBTYPE:
            return None
        data1 = int(ns_event.data1())
    except Exception:
        return None

    # The classification itself is resolved by the ``_key_action``
    # engine (native when the Rust extension is built, the
    # pure-Python transcription otherwise); the None gate, the
    # subtype comparison, the int() coercion and the except stay here.
    return _key_action(data1)


class MacOSEventTapService:
    """Global media-key listener using Quartz event taps."""

    def __init__(self) -> None:
        self._callbacks: dict[str, PlayerCallback] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._tap: Any = None
        self._run_loop: Any = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    async def start(
        self,
        callbacks: dict[str, PlayerCallback],
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Start background event tap thread."""
        if not _EVENT_TAP_AVAILABLE:
            logger.info("Quartz/AppKit not available — macOS media key tap disabled")
            return False

        if self._running:
            return True

        self._callbacks = callbacks
        self._loop = loop
        # start() runs on the app's loop thread: snapshot its context so
        # dispatched key callbacks keep Textual's active_app ContextVar.
        capture_dispatch_context()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run_tap_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=1.0)
        return self._running

    def stop(self) -> None:
        """Stop event tap and run loop."""
        if not _EVENT_TAP_AVAILABLE:
            return

        self._running = False
        if self._tap is not None:
            try:
                Quartz.CGEventTapEnable(self._tap, False)
            except Exception:
                logger.debug("Failed to disable macOS event tap", exc_info=True)
        if self._run_loop is not None:
            try:
                Quartz.CFRunLoopStop(self._run_loop)
            except Exception:
                logger.debug("Failed to stop macOS event tap loop", exc_info=True)
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._tap = None
        self._run_loop = None

    def _run_tap_loop(self) -> None:
        mask = Quartz.CGEventMaskBit(Quartz.NSSystemDefined)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionDefault,
            mask,
            self._tap_callback,
            None,
        )

        if self._tap is None:
            logger.info(
                "Could not create macOS media key event tap. "
                "Grant Accessibility permission to your terminal app."
            )
            self._ready.set()
            return

        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        self._run_loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(self._run_loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        self._running = True
        self._ready.set()
        logger.info("macOS media key event tap enabled")
        Quartz.CFRunLoopRun()

        try:
            Quartz.CFMachPortInvalidate(self._tap)
        except Exception:
            logger.debug("Failed to invalidate macOS event tap", exc_info=True)

    def _tap_callback(self, _proxy: Any, event_type: int, event: Any, _refcon: Any) -> Any:
        if not self._running or event_type != Quartz.NSSystemDefined:
            return event

        ns_event = AppKit.NSEvent.eventWithCGEvent_(event)
        action = _event_action(ns_event)
        if action is None:
            return event

        callback = self._callbacks.get(action)
        if callback is None or self._loop is None or self._loop.is_closed():
            return event

        if not dispatch_coro_threadsafe(self._loop, callback):
            return event

        # Swallow event so Apple Music does not steal media-key presses.
        return None


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same media-key classification engine;
# this binding overrides the pure-Python transcription above when the
# extension is importable. The rebinding must happen AFTER the definition
# above — assigning the native function any earlier would be clobbered by
# the ``def`` statement (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        macos_event_tap_action as _native_macos_event_tap_action,
    )
except ImportError:
    pass
else:

    def _key_action(data1: int) -> str | None:  # noqa: F811
        """The media-key classification (native Rust engine)."""
        return _native_macos_event_tap_action(data1)
