"""Background audio download service using yt-dlp.

The service's pure cores are dual-implemented: the base options dict
``_base_opts`` (the nine-key yt-dlp configuration behind
``DownloadService._build_opts``, before the configured extras) and
the candidate-name build ``_candidate_names`` (the five probe
filenames behind ``_download_sync``'s publication walk and
``get_path``'s lookup). Both are implemented by the Rust extension
(``_ytm_player_native.download_base_opts`` /
``_ytm_player_native.download_candidates``) when it is built and by
the pure-Python transcriptions below otherwise; the staging, the
yt-dlp invocation, the atomic publication and the thread ownership
stay Python-side in the service.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ytm_player.config.paths import SECURE_FILE_MODE, secure_chmod
from ytm_player.config.settings import get_settings
from ytm_player.services.yt_dlp_options import apply_configured_yt_dlp_options
from ytm_player.utils.formatting import VALID_VIDEO_ID

logger = logging.getLogger(__name__)

# Quality presets for downloading (prefer opus for smaller size).
_DOWNLOAD_FORMAT = "bestaudio[ext=webm]/bestaudio/best"

# The extension probe order a completed download is looked up under —
# opus first (the postprocessor's target), then the container/audio
# extensions yt-dlp and FFmpeg may leave behind.
_PROBE_EXTS = ("opus", "webm", "m4a", "mp3", "ogg")


def _base_opts(output_path: str) -> dict:
    """The yt-dlp options dict a download runs with.

    The pure half of ``DownloadService._build_opts``: the nine-key
    dict — the audio format selector, the staged ``outtmpl``, the
    quiet/no-warning/noplaylist flags, the write gates and the
    one-entry FFmpegExtractAudio postprocessor — before
    ``apply_configured_yt_dlp_options`` layers the configured extras
    on top. The output path passes through as the object the caller
    assembled.

    Implemented by the Rust extension
    (``_ytm_player_native.download_base_opts``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    opts = {
        "format": _DOWNLOAD_FORMAT,
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extract_flat": False,
        "writethumbnail": False,
        "writeinfojson": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "opus",
                "preferredquality": "128",
            }
        ],
    }
    return opts


def _candidate_names(video_id: str) -> tuple[str, ...]:
    """The five probe filenames a completed download is looked up under.

    The pure half of ``DownloadService._download_sync``'s publication
    walk and ``get_path``'s lookup: the ``<video_id>.<ext>`` names over
    the probe order — opus first, then webm, m4a, mp3 and ogg — built
    as a fresh tuple of fresh strings.

    Implemented by the Rust extension
    (``_ytm_player_native.download_candidates``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return tuple(f"{video_id}.{ext}" for ext in _PROBE_EXTS)


@dataclass
class DownloadResult:
    """Result of a single download attempt."""

    video_id: str
    success: bool
    file_path: Path | None = None
    error: str | None = None


class DownloadService:
    """Downloads audio files for offline playback using yt-dlp.

    Downloads are saved to the audio cache directory. Each download runs
    in a background thread to avoid blocking the event loop.
    """

    def __init__(self, download_dir: Path | None = None) -> None:
        settings = get_settings()
        self._download_dir = download_dir or settings.cache_dir
        self._active: dict[str, asyncio.Task[DownloadResult]] = {}

    def _ensure_dir(self) -> None:
        self._download_dir.mkdir(parents=True, exist_ok=True)

    def _build_opts(self, output_path: str) -> dict:
        """Build yt-dlp options for a download.

        The base options dict is resolved by the ``_base_opts`` engine
        (native when the Rust extension is built, the pure-Python
        transcription otherwise); this method layers the configured
        extras on top.
        """
        yt_dlp_settings = get_settings().yt_dlp
        return apply_configured_yt_dlp_options(_base_opts(output_path), yt_dlp_settings)

    def _download_sync(self, video_id: str) -> DownloadResult:
        """Synchronous download (runs in a thread)."""
        import yt_dlp

        if not VALID_VIDEO_ID.match(video_id):
            return DownloadResult(video_id=video_id, success=False, error="Invalid video ID")

        url = f"https://music.youtube.com/watch?v={video_id}"

        try:
            self._ensure_dir()
            # FFmpeg can leave its final extension behind on failure. Keep all
            # job output private until yt-dlp and its context finish successfully.
            # Staging on the same filesystem allows atomic publication below.
            with tempfile.TemporaryDirectory(
                prefix=f".{video_id}-", dir=self._download_dir
            ) as stage:
                output_template = str(Path(stage) / f"{video_id}.%(ext)s")
                opts = self._build_opts(output_template)
                with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[arg-type]
                    status = ydl.download([url])
                if status != 0:
                    return DownloadResult(
                        video_id=video_id, success=False, error="yt-dlp reported a download failure"
                    )

                # Find the downloaded file (extension may vary).
                for name in _candidate_names(video_id):
                    path = Path(stage) / name
                    if path.is_file():
                        secure_chmod(path, SECURE_FILE_MODE)
                        dest = self._download_dir / path.name
                        if os.path.lexists(dest):
                            return DownloadResult(
                                video_id=video_id,
                                success=False,
                                error="Download destination already exists",
                            )
                        # The active-ID owner excludes other writers in this service.
                        # This check/replace is not an external-writer no-clobber lock.
                        os.replace(path, dest)
                        return DownloadResult(video_id=video_id, success=True, file_path=dest)

            return DownloadResult(
                video_id=video_id,
                success=False,
                error="Download completed but file not found",
            )

        except Exception as exc:
            logger.warning("Download failed for %s: %s", video_id, exc)
            return DownloadResult(video_id=video_id, success=False, error=str(exc))

    async def download(self, video_id: str) -> DownloadResult:
        """Download or reuse a completed track, retaining ownership until its thread exits.

        Cancelling the caller does not stop yt-dlp. Shield the owned task so
        retries cannot start another writer while that thread is still running.
        """
        if not VALID_VIDEO_ID.match(video_id):
            return DownloadResult(video_id=video_id, success=False, error="Invalid video ID")
        if video_id in self._active:
            return DownloadResult(video_id=video_id, success=False, error="Already downloading")

        existing = self.get_path(video_id)
        if existing is not None:
            return DownloadResult(video_id=video_id, success=True, file_path=existing)

        task = asyncio.create_task(asyncio.to_thread(self._download_sync, video_id))
        self._active[video_id] = task
        task.add_done_callback(lambda completed: self._download_done(video_id, completed))
        return await asyncio.shield(task)

    def _download_done(self, video_id: str, task: asyncio.Task[DownloadResult]) -> None:
        """Release the writer and observe errors even when its caller was cancelled."""
        if self._active.get(video_id) is task:
            del self._active[video_id]
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Download worker failed for %s",
                    video_id,
                    exc_info=(type(error), error, error.__traceback__),
                )

    async def download_multiple(
        self,
        tracks: list[dict],
    ) -> list[DownloadResult]:
        """Download multiple tracks sequentially."""
        results: list[DownloadResult] = []

        for i, track in enumerate(tracks):
            video_id = track.get("video_id", "")
            if not video_id:
                results.append(DownloadResult(video_id="", success=False, error="No video ID"))
                continue

            # Skip if already downloaded.
            if self.is_downloaded(video_id):
                results.append(
                    DownloadResult(
                        video_id=video_id, success=True, file_path=self.get_path(video_id)
                    )
                )
                continue

            result = await self.download(video_id)
            results.append(result)

        return results

    def is_downloaded(self, video_id: str) -> bool:
        """Check if a track has been downloaded."""
        return self.get_path(video_id) is not None

    def is_downloading(self, video_id: str) -> bool:
        """True while this service owns a running writer for *video_id*."""
        return video_id in self._active

    def get_path(self, video_id: str) -> Path | None:
        """Return a completed download, never a file its worker may still be writing.

        The probe filenames are resolved by the ``_candidate_names``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); this method performs the
        lookups.
        """
        if video_id in self._active:
            return None
        for name in _candidate_names(video_id):
            path = self._download_dir / name
            try:
                if path.exists():
                    return path
            except OSError as exc:
                # An unreadable download directory must not escape into the UI.
                logger.warning("Cannot check for downloaded file %s: %s", path, exc)
                return None
        return None

    @property
    def active_count(self) -> int:
        return len(self._active)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same download-service engines; these
# bindings replace the pure-Python transcriptions above when the extension
# is importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        download_base_opts as _native_download_base_opts,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        download_candidates as _native_download_candidates,
    )
except ImportError:
    pass
else:
    _base_opts = _native_download_base_opts  # type: ignore[assignment]
    _candidate_names = _native_download_candidates  # type: ignore[assignment]
