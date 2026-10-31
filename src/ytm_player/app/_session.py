"""Session state persistence mixin for YTMPlayerApp.

The restore path is split in two: ``coerce_session_state`` is the pure
validation engine — every isinstance gate, clamp and fallback applied to
the raw parsed session.json object before any app state is touched —
implemented by the Rust extension
(``_ytm_player_native.coerce_session_state``) when it is built and by
the pure-Python transcription below otherwise. ``SessionMixin`` only
*applies* the coerced values. ``find_track_index`` is the resume-point
scan over the restored queue, with the same dual implementation.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

from ytm_player.app._base import YTMHostBase
from ytm_player.services.queue import RepeatMode
from ytm_player.ui.playback_bar import PlaybackBar
from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

logger = logging.getLogger(__name__)

_SESSION_SCHEMA_VERSION = 1


def coerce_session_state(state: dict, default_volume: object) -> dict:
    """Validate and coerce a parsed session.json object in one pass.

    The pure-Python fallback for the native Rust engine. A literal
    transcription of the validation/coercion logic that used to live
    inline in ``_restore_session_state`` — the same gates, the same
    clamps, the same fallbacks — so the two implementations stay
    interchangeable:

    - ``volume`` — a non-bool int/float clamps into 0..=100; anything
      else yields the raw *default_volume* (unclamped, like the
      original). An absent key clamps the default itself.
    - ``repeat`` — "off" / "all" / "one"; anything else degrades to
      "off". (Unhashable values like lists used to escape as TypeError
      from the ``RepeatMode`` lookup; the engine treats every non-member
      alike on both backends.)
    - ``queue_tracks`` — ``normalize_tracks()`` of the stored list when
      it is a truthy list, else ``[]``.
    - ``queue_index`` — the stored index when it is an int (bools count:
      they are ints in Python, so ``true`` validates as index 1) inside
      the normalized range, else ``None``.
    - ``queue_context_id`` — non-empty str or ``None``.
    - ``sidebar_per_page`` — the stored dict itself (no copy) when
      truthy, else ``None``.
    - ``first_run_hint_shown`` / ``mpris_hint_shown`` — truthiness.
    - ``has_transliteration`` + ``transliteration_enabled`` — key
      presence and the RAW stored value (assigned unconverted, like the
      original).
    - ``resume`` — the stored resume dict (the original object, not a
      copy: the caller copies it for ``_loaded_resume`` and reads
      position / playlist_id from it directly) when it is a dict with a
      truthy ``video_id``, else ``None``.
    """
    from ytm_player.utils.formatting import normalize_tracks

    # -- resume staging ------------------------------------------------------
    resume = state.get("resume")
    if not (isinstance(resume, dict) and resume.get("video_id")):
        resume = None

    # -- volume ---------------------------------------------------------------
    volume = state.get("volume", default_volume)
    if isinstance(volume, bool) or not isinstance(volume, (int, float)):
        volume = default_volume
    else:
        volume = int(max(0, min(100, volume)))

    # -- repeat mode ------------------------------------------------------------
    repeat = state.get("repeat", "off")
    if not (isinstance(repeat, str) and repeat in ("off", "all", "one")):
        repeat = "off"

    # -- queue -------------------------------------------------------------------
    saved_tracks = state.get("queue_tracks", [])
    queue_tracks: list[dict] = []
    queue_index: int | None = None
    if saved_tracks and isinstance(saved_tracks, list):
        queue_tracks = normalize_tracks(saved_tracks)
        saved_index = state.get("queue_index", 0)
        if isinstance(saved_index, int) and 0 <= saved_index < len(queue_tracks):
            queue_index = saved_index

    # -- shuffle -------------------------------------------------------------------
    shuffle = bool(state.get("shuffle", False))

    # -- collection context ------------------------------------------------------
    saved_context = state.get("queue_context_id")
    queue_context_id = saved_context if isinstance(saved_context, str) and saved_context else None

    # -- sidebar state ------------------------------------------------------------
    saved_sidebar = state.get("sidebar_per_page")
    sidebar_per_page = saved_sidebar if saved_sidebar and isinstance(saved_sidebar, dict) else None

    # -- hint flags ------------------------------------------------------------------
    first_run_hint_shown = bool(state.get("first_run_hint_shown", False))
    mpris_hint_shown = bool(state.get("mpris_hint_shown", False))

    # -- transliteration toggle (raw value, assigned unconverted) ------------------
    has_transliteration = "transliteration_enabled" in state
    transliteration_enabled = state.get("transliteration_enabled")

    return {
        "volume": volume,
        "repeat": repeat,
        "shuffle": shuffle,
        "queue_tracks": queue_tracks,
        "queue_index": queue_index,
        "queue_context_id": queue_context_id,
        "sidebar_per_page": sidebar_per_page,
        "first_run_hint_shown": first_run_hint_shown,
        "mpris_hint_shown": mpris_hint_shown,
        "has_transliteration": has_transliteration,
        "transliteration_enabled": transliteration_enabled,
        "resume": resume,
    }


def find_track_index(tracks: Iterable[dict], video_id: object) -> int | None:
    """Return the index of the first track whose ``video_id`` equals *video_id*.

    The resume-point lookup from ``_restore_session_state``, run over
    the restored (possibly shuffled) queue. ``video_id`` may be any
    truthy JSON value — comparison is plain ``==``.
    """
    for i, t in enumerate(tracks):
        if t.get("video_id") == video_id:
            return i
    return None


class SessionMixin(YTMHostBase):
    """Persist and restore session state (volume, shuffle, repeat, queue, etc.)."""

    def _load_session_state(self) -> dict:
        """Read session.json.

        Returns ``{}`` when the file is missing, unreadable, not valid JSON,
        not a JSON object, or written by another schema version.
        """
        from ytm_player.config.paths import SESSION_STATE_FILE

        state: object = {}
        try:
            if SESSION_STATE_FILE.exists():
                state = json.loads(SESSION_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Could not read session state", exc_info=True)

        if not isinstance(state, dict):
            logger.warning("Discarding session state — not a JSON object")
            return {}

        # Schema version check: discard state from incompatible older/future formats.
        file_version = state.get("schema_version")
        if file_version != _SESSION_SCHEMA_VERSION:
            if file_version is not None:
                logger.warning(
                    "Discarding session state — schema_version %r != %d (expected). "
                    "Settings will reset to defaults.",
                    file_version,
                    _SESSION_SCHEMA_VERSION,
                )
            return {}
        return state

    async def _restore_session_state(self) -> None:
        """Restore volume, shuffle, repeat, queue and resume point from the last session.

        ``_session_restored`` is set only when this runs to completion;
        ``_save_session_state`` writes nothing before that, so a start that
        fails part-way can't overwrite the previous session.

        All validation and coercion of the raw state happens up front in
        :func:`coerce_session_state` — the native Rust engine when the
        extension is built, the pure-Python transcription otherwise. This
        method only applies the coerced values.
        """
        state = self._load_session_state()
        c = coerce_session_state(state, self.settings.playback.default_volume)

        # The resume point read from disk stays valid until a track's load
        # is accepted this session (play_track clears it). Until then
        # _save_session_state writes it back unchanged — whether or not
        # resume-on-launch stages it for a seek below.
        resume = c["resume"]
        self._loaded_resume = dict(resume) if resume is not None else None

        volume = c["volume"]
        if self.player:
            await self.player.set_volume(volume)

        mode = RepeatMode(c["repeat"])
        self.queue.set_repeat(mode)

        # Restore queue from last session (before enabling shuffle so the
        # shuffle order is built from a populated queue).
        restored = c["queue_tracks"]
        if restored:
            self.queue.add_multiple(restored)
            if c["queue_index"] is not None:
                self.queue.jump_to(c["queue_index"])

        if c["shuffle"]:
            self.queue.toggle_shuffle()

        # Restore the queue's collection identity so a post-restart
        # shuffle toggle persists to the right key (TP-7).
        if c["queue_context_id"]:
            self.queue.set_context(c["queue_context_id"])

        # Update the playback bar to reflect restored state.
        try:
            bar = self.query_one("#playback-bar", PlaybackBar)
            bar.update_volume(volume)
            bar.update_repeat(mode)
            bar.update_shuffle(self.queue.shuffle_enabled)
        except Exception:
            logger.debug(
                "Failed to update playback bar after restoring session state", exc_info=True
            )

        # Restore sidebar state.
        if c["sidebar_per_page"] is not None:
            self._sidebar_per_page = c["sidebar_per_page"]
        # Always start with lyrics sidebar closed regardless of previous session.
        self._lyrics_sidebar_open = False

        # Restore first-run hint flag — defaults False so legacy session.json
        # files (or fresh installs) trigger the toast on first launch.
        self._first_run_hint_shown = c["first_run_hint_shown"]
        self._mpris_hint_shown = c["mpris_hint_shown"]

        # Restore transliteration toggle state (session overrides config).
        if c["has_transliteration"]:
            try:
                self.query_one("#lyrics-sidebar", LyricsSidebar)._transliteration_enabled = c[
                    "transliteration_enabled"
                ]
            except Exception:
                pass

        # Restore last-playing track + position if user has resume enabled.
        # The track is shown paused; the first play_track call for this
        # video_id seeks to the saved position via _pending_resume_position
        # (handled in app/_playback.py).
        if not self.settings.playback.resume_on_launch:
            self._session_restored = True
            return

        if resume is not None:
            video_id = resume.get("video_id", "")
            if video_id:
                self._active_library_playlist_id = resume.get("playlist_id")
                # Find the track in the restored queue and jump to it.
                idx = find_track_index(self.queue.tracks, video_id)

                if idx is not None:
                    self.queue.jump_to(idx)
                    track = self.queue.current_track
                    if track:
                        # Stash the resume target so the first play_track
                        # call for this video_id seeks to the saved position.
                        self._pending_resume_video_id = video_id
                        try:
                            self._pending_resume_position = float(resume.get("position", 0))
                        except (TypeError, ValueError):
                            self._pending_resume_position = 0.0
                        # Show the track + saved position in the UI without
                        # starting playback.
                        try:
                            bar = self.query_one("#playback-bar", PlaybackBar)
                            bar.update_track(track)
                            bar.update_playback_state(is_playing=False, is_paused=False)
                            bar.update_position(
                                self._pending_resume_position,
                                track.get("duration") or 0,
                            )
                        except Exception:
                            logger.debug(
                                "Playback bar not ready during resume restore",
                                exc_info=True,
                            )

        self._session_restored = True

    def _save_session_state(self) -> None:
        """Persist volume, shuffle, repeat, queue and resume point to disk."""
        from ytm_player.config.paths import SESSION_STATE_FILE

        if not self._session_restored:
            # Nothing in memory is authoritative yet (startup failed before
            # the restore, or the restore itself raised): the file on disk
            # is the user's real state — leave it alone.
            logger.warning("Session was never restored this run — not overwriting session.json")
            return

        volume = 80
        if self.player:
            try:
                volume = self.player.volume
            except Exception:
                logger.debug("Failed to read player volume for session save", exc_info=True)

        queue_tracks = list(self.queue.tracks)
        queue_index = self.queue.current_index

        # Resume point. The live track wins, but only past 1.0 s so a
        # premature exit can't record "position 0". Otherwise the value
        # read from disk at restore is written back: play_track clears it
        # the moment a track's load is accepted, so a session that played
        # nothing keeps its resume point and one that played and then
        # stopped saves none. Whether to RESTORE on next launch is gated by
        # settings.playback.resume_on_launch in _restore_session_state.
        resume = None
        if self.player and self.player.current_track and self.player.position > 1.0:
            video_id = self.player.current_track.get("video_id", "")
            if video_id:
                resume = {
                    "video_id": video_id,
                    "position": self.player.position,
                    "playlist_id": self._active_library_playlist_id,
                }
        if resume is None and self._loaded_resume is not None:
            resume = self._loaded_resume

        state = {
            "schema_version": _SESSION_SCHEMA_VERSION,
            "volume": volume,
            "repeat": self.queue.repeat_mode.value,
            "shuffle": self.queue.shuffle_enabled,
            "queue_tracks": queue_tracks,
            "queue_index": queue_index,
            "queue_context_id": self.queue.current_context_id,
            "resume": resume,
            "sidebar_per_page": self._sidebar_per_page,
            "transliteration_enabled": self._get_transliteration_state(),
            "first_run_hint_shown": self._first_run_hint_shown,
            "mpris_hint_shown": self._mpris_hint_shown,
        }
        try:
            import os

            from ytm_player.config.paths import SECURE_FILE_MODE, secure_chmod

            SESSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = SESSION_STATE_FILE.with_suffix(SESSION_STATE_FILE.suffix + ".tmp")
            try:
                tmp_path.write_text(json.dumps(state), encoding="utf-8")
                secure_chmod(tmp_path, SECURE_FILE_MODE)
                os.replace(tmp_path, SESSION_STATE_FILE)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
        except (OSError, TypeError):
            logger.exception("Could not save session state")
            try:
                self.notify(
                    "Could not save session state — your queue and "
                    "position may not restore on next launch.",
                    severity="warning",
                    timeout=8,
                )
            except Exception:
                # If notify itself fails (e.g. app shutting down), log and move on.
                logger.exception("Failed to surface save-failure notify")

    def _get_transliteration_state(self) -> bool:
        """Read transliteration toggle from the lyrics sidebar."""
        try:
            return self.query_one("#lyrics-sidebar", LyricsSidebar)._transliteration_enabled
        except Exception:
            return False


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engine; these bindings replace the
# pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        coerce_session_state as _native_coerce_session_state,
    )
    from ytm_player.utils._ytm_player_native import (
        find_track_index as _native_find_track_index,
    )
except ImportError:
    pass
else:
    coerce_session_state = _native_coerce_session_state  # type: ignore[assignment]
    find_track_index = _native_find_track_index  # type: ignore[assignment]
