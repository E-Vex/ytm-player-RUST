"""MPRIS / media-key callback mixin for YTMPlayerApp."""

from __future__ import annotations

from typing import Any

from ytm_player.app._base import YTMHostBase


def _us_to_seconds(microseconds: int) -> float:
    """The µs→seconds conversion — the ``seconds`` engine.

    The pure core behind both MPRIS time callbacks —
    ``microseconds / 1_000_000``, CPython's own int/int true division —
    from ``_mpris_seek``'s D-Bus ``Seek`` offset and
    ``_mpris_set_position``'s ``SetPosition`` position. Neither arm
    clamps: a D-Bus client's backward seek arrives as a negative
    offset and must reach the player as negative seconds, exactly as
    the reference passes it.

    Implemented by the Rust extension
    (``_ytm_player_native.mpris_us_to_seconds``) when it is built;
    this pure-Python transcription is the fallback otherwise. The
    player guards and the seek calls stay with the mixin.
    """
    return microseconds / 1_000_000


def _volume_percent(volume: float) -> int:
    """The volume-fraction→percent conversion — the ``percent`` engine.

    The pure core of ``_mpris_set_volume`` — ``round(volume * 100)``,
    CPython's own single-argument ``round``: the nearest integer with
    ties going to the even neighbour (``round(12.5) == 12``,
    ``round(37.5) == 38``), not half-away-from-zero. The ``Volume``
    setter's clamped ``[0, 1]`` domain lands every real write in
    ``[0, 100]``.

    Implemented by the Rust extension
    (``_ytm_player_native.mpris_volume_percent``) when it is built;
    this pure-Python transcription is the fallback otherwise. The
    player guard and the ``set_volume`` call stay with the mixin.
    """
    return round(volume * 100)


class MPRISMixin(YTMHostBase):
    """Builds the callback dict expected by MPRISService / MacOS / Windows media keys."""

    def _build_mpris_callbacks(self) -> dict[str, Any]:
        """Build the callback dict expected by MPRISService.start()."""
        return {
            "play": self._mpris_play,
            "pause": self._mpris_pause,
            "play_pause": self._mpris_play_pause,
            "stop": self._mpris_stop,
            "next": self._mpris_next,
            "previous": self._mpris_previous,
            "seek": self._mpris_seek,
            "set_position": self._mpris_set_position,
            "set_volume": self._mpris_set_volume,
            "quit": self._mpris_quit,
        }

    async def _mpris_play(self) -> None:
        if self.player and self.player.is_paused:
            await self.player.resume()

    async def _mpris_pause(self) -> None:
        if self.player:
            await self.player.pause()

    async def _mpris_play_pause(self) -> None:
        await self._toggle_play_pause()

    async def _mpris_stop(self) -> None:
        from ytm_player.app._ownership import invalidate_requests

        invalidate_requests(self)
        self._last_play_video_id = ""
        self._last_play_time = 0.0
        if self.player:
            # Stop supersedes the current play attempt: a stream error
            # arriving for it afterwards must not restart playback.
            self._play_generation += 1
            await self.player.stop()

    async def _mpris_next(self) -> None:
        await self._play_next()

    async def _mpris_previous(self) -> None:
        await self._play_previous()

    async def _mpris_seek(self, offset_us: int) -> None:
        if self.player:
            # The unit conversion is resolved by the ``_us_to_seconds``
            # engine (native when the Rust extension is built, the
            # pure-Python transcription otherwise); the player guard
            # and the seek call stay here.
            await self.player.seek(_us_to_seconds(offset_us))

    async def _mpris_set_position(self, position_us: int) -> None:
        if self.player:
            await self.player.seek_absolute(_us_to_seconds(position_us))

    async def _mpris_set_volume(self, volume: float) -> None:
        if self.player:
            # The percent conversion is resolved by the
            # ``_volume_percent`` engine (native when the Rust
            # extension is built, the pure-Python transcription
            # otherwise); the player guard and the set_volume call
            # stay here.
            await self.player.set_volume(_volume_percent(volume))

    async def _mpris_quit(self) -> None:
        self.exit()


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same unit-conversion engines; these
# bindings override the pure-Python transcriptions above when the
# extension is importable. The rebinding must happen AFTER the
# definitions above — assigning the native functions any earlier would
# be clobbered by the ``def`` statements (the lesson that produced
# formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        mpris_us_to_seconds as _native_mpris_us_to_seconds,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        mpris_volume_percent as _native_mpris_volume_percent,
    )
except ImportError:
    pass
else:

    def _us_to_seconds(microseconds: int) -> float:  # noqa: F811
        """The µs→seconds conversion (native Rust engine)."""
        return _native_mpris_us_to_seconds(microseconds)

    def _volume_percent(volume: float) -> int:  # noqa: F811
        """The volume-fraction→percent conversion (native Rust engine)."""
        return _native_mpris_volume_percent(volume)
