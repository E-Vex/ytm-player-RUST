"""Last.fm scrobbling integration for ytm-player.

Scrobbles tracks after 50% playback or 4 minutes (whichever comes first),
per the Last.fm scrobbling spec. Sends "Now Playing" on track start.
Requires the optional `pylast` package.

The scrobble-threshold gate ``_scrobble_due`` (the decision behind
``check_scrobble``'s periodic poll — the 50%-or-4-minutes threshold
walk and the position comparison that says whether the current track
has played far enough to be scrobbled) is implemented by the Rust
extension (``_ytm_player_native.lastfm_scrobble_due``) when it is
built and by the pure-Python transcription below otherwise; the
connection/current-track/scrobbled state, the ``duration`` read out
of the current track and the pylast network calls stay Python-side
in the service. The threshold constants resolve through this
module's own ``_SCROBBLE_PERCENT`` / ``_SCROBBLE_MAX_SECONDS`` names
at call time, so patched values steer both backends identically.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Scrobble threshold: 50% of track or 4 minutes, whichever is less.
_SCROBBLE_PERCENT = 0.5
_SCROBBLE_MAX_SECONDS = 240


def _scrobble_due(position: Any, duration: Any) -> bool:
    """The scrobble-threshold gate behind ``check_scrobble``.

    The Last.fm spec's threshold walk: a track with no duration info
    (or a non-positive one) is due once the position reaches the
    4-minute cap; otherwise the threshold is 50% of the duration,
    capped at 4 minutes — ``min(duration * _SCROBBLE_PERCENT,
    _SCROBBLE_MAX_SECONDS)``, the tie answering the product — and the
    track is due once ``position >= threshold`` (the boundary
    included). Every comparison, the product and the ``min`` walk are
    CPython's own, so the bool spellings (``False <= 0`` is true) and
    the non-numeric error shapes stay the interpreter's; the
    constants are read through this module's own names at call time,
    so patched values steer both backends identically.

    Implemented by the Rust extension
    (``_ytm_player_native.lastfm_scrobble_due``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if duration <= 0:
        # No duration info — scrobble after 4 minutes.
        threshold = _SCROBBLE_MAX_SECONDS
    else:
        threshold = min(duration * _SCROBBLE_PERCENT, _SCROBBLE_MAX_SECONDS)
    return position >= threshold


class LastFMService:
    """Manages Last.fm authentication and scrobbling.

    API credentials are stored in the app config. The session key is
    cached after first auth so the user doesn't need to re-authenticate.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        session_key: str = "",
        username: str = "",
        password_hash: str = "",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._session_key = session_key
        self._username = username
        self._password_hash = password_hash
        self._network: object | None = None
        self._connected = False
        self._current_track: dict | None = None
        self._track_start: float = 0
        self._scrobbled = False

    async def connect(self) -> bool:
        """Authenticate with Last.fm. Returns True on success."""
        if not self._api_key or not self._api_secret:
            logger.info("Last.fm API credentials not configured")
            return False

        try:
            import pylast  # type: ignore[reportMissingImports]
        except ImportError:
            logger.info("pylast not installed — Last.fm scrobbling disabled")
            return False

        try:
            self._network = pylast.LastFMNetwork(
                api_key=self._api_key,
                api_secret=self._api_secret,
                session_key=self._session_key,
                username=self._username,
                password_hash=self._password_hash,
            )
            self._connected = True
            logger.info("Connected to Last.fm")
            return True
        except Exception:
            logger.debug("Failed to connect to Last.fm", exc_info=True)
            self._connected = False
            return False

    async def now_playing(
        self, title: str, artist: str, album: str = "", duration: int = 0
    ) -> None:
        """Send a "Now Playing" notification to Last.fm."""
        if not self._connected or not self._network:
            return

        self._current_track = {
            "title": title,
            "artist": artist,
            "album": album,
            "duration": duration,
        }
        self._track_start = time.time()
        self._scrobbled = False

        try:
            await asyncio.to_thread(
                self._network.update_now_playing,  # type: ignore[union-attr]
                artist=artist,
                title=title,
                album=album or None,
                duration=duration or None,
            )
        except Exception:
            logger.debug("Failed to update Last.fm Now Playing", exc_info=True)

    async def check_scrobble(self, position: float) -> None:
        """Check if the current track should be scrobbled based on position.

        Call this periodically (e.g. every few seconds) with the current
        playback position. The threshold decision is resolved by the
        ``_scrobble_due`` engine (native when the Rust extension is
        built, the pure-Python transcription otherwise).
        """
        if not self._connected or not self._current_track or self._scrobbled:
            return

        duration = self._current_track.get("duration", 0)
        if _scrobble_due(position, duration):
            await self._scrobble()

    async def _scrobble(self) -> None:
        """Submit the current track as a scrobble."""
        if not self._network or not self._current_track:
            return

        self._scrobbled = True
        track = self._current_track
        timestamp = int(self._track_start)

        try:
            await asyncio.to_thread(
                self._network.scrobble,  # type: ignore[union-attr]
                artist=track["artist"],
                title=track["title"],
                timestamp=timestamp,
                album=track.get("album") or None,
                duration=track.get("duration") or None,
            )
            logger.debug("Scrobbled: %s - %s", track["artist"], track["title"])
        except Exception:
            logger.debug("Failed to scrobble to Last.fm", exc_info=True)

    @property
    def is_connected(self) -> bool:
        return self._connected


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same scrobble-threshold gate; this
# binding replaces the pure-Python transcription above when the extension
# is importable.  The rebinding must happen AFTER the definition above —
# assigning the native function any earlier would be clobbered by the
# ``def`` statement.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        lastfm_scrobble_due as _native_lastfm_scrobble_due,
    )
except ImportError:
    pass
else:
    _scrobble_due = _native_lastfm_scrobble_due  # type: ignore[assignment]
