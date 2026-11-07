"""Stream URL resolution using yt-dlp.

The resolver's pure cores are dual-implemented: the expiry gate
``_is_stale`` (the buffered staleness verdict behind every cache read
path), the available_at wait gate ``_wait_seconds`` (the pre-roll-ad
wait behind ``StreamInfo.wait_seconds``), the quality-input gate
``_validate_quality`` (the validation the resolver's ``quality``
setter raises through), the base options dict ``_base_ydl_opts`` (the
ten-key yt-dlp configuration behind ``_build_ydl_opts``, before the
configured extras), the extract-info field assembly
``_stream_info_fields`` (the ``info`` dict → the eight ``StreamInfo``
field values behind ``_try_resolve``), the DownloadError hint
``_download_error_hint`` (the stale-cookiejar diagnostic the error
handler appends), the expired-entry walk ``_cache_expired_ids`` (the
prune phase behind both ``_put_cache``'s and ``prune_expired``'s
maintenance), the over-cap eviction walk ``_cache_cap_evictions``
(the oldest-by-``expires_at`` excess ``_put_cache`` drops once the
prune has run), the cookiejar-signature reuse gate
``_ydl_cookiejar_current`` (the verdict behind ``_get_ydl``'s reuse
arm — the cached YoutubeDL instance's baked cookiejar signature
against the signature the resolver should be using right now, so a
rewritten jar rebuilds the instance instead of serving stale cookies)
and the mid-build publication verdict ``_ydl_publish_verdict`` (the
decision behind ``_get_ydl``'s post-build re-entry — adopt another
thread's instance, retry after a mid-build reset, or install the built
one). The resolver's remaining pure cores — the retry-delay
iteration ``_retry_delays`` (the ``(attempt, delay)`` schedule walk
behind ``_resolve_sync``'s retry loop) and the pending-future
bookkeeping of ``_resolve``'s request dedup: the pending-join verdict
``_pending_join_verdict`` (the join gate behind ``_current_pending``
— the in-flight resolve a same-generation caller joins instead of
extracting again), the pending-drop verdict ``_pending_drop_verdict``
(the identity gate behind the finally cleanup — the pending slot is
dropped only when it still holds this owner's very future) and the
pending publication gate ``_pending_publication_gate`` (the
unresolved gate behind the shared future's three publications —
``set_result``, ``set_exception`` and the owner-cancelled
``cancel``) — answer the attempts the resolve makes, the waits
between them and the dedup decisions around its shared futures. All
are implemented by the Rust extension
(``_ytm_player_native.is_stale`` /
``_ytm_player_native.wait_seconds`` /
``_ytm_player_native.validate_quality`` /
``_ytm_player_native.stream_base_opts`` /
``_ytm_player_native.stream_info_fields`` /
``_ytm_player_native.download_error_hint`` /
``_ytm_player_native.stream_cache_expired_ids`` /
``_ytm_player_native.stream_cache_cap_evictions`` /
``_ytm_player_native.stream_ydl_cookiejar_current`` /
``_ytm_player_native.stream_ydl_publish_verdict`` /
``_ytm_player_native.stream_retry_delays`` /
``_ytm_player_native.stream_pending_join_verdict`` /
``_ytm_player_native.stream_pending_drop_verdict`` /
``_ytm_player_native.stream_pending_publication_gate``) when it is
built and by the pure-Python transcriptions below otherwise; the
YoutubeDL construction, the cookiejar vouching and its signature
probe, the network extraction, the sleeps, the cache state, the
pending-future state and its mutations stay Python-side in the
resolver. The clock-reading engines resolve ``time.time()`` through
this module's own ``time`` name at call time, so a patched or frozen
clock steers both backends identically.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from ytm_player.config.settings import get_settings
from ytm_player.services.yt_dlp_options import (
    apply_configured_yt_dlp_options,
    normalize_cookiefile,
)
from ytm_player.utils.formatting import VALID_VIDEO_ID

logger = logging.getLogger(__name__)

# Cache resolved URLs for 5 hours (YouTube URLs typically expire after ~6 hours).
_CACHE_TTL_SECONDS = 5 * 60 * 60

# Maximum number of entries to keep in cache.  When exceeded, the oldest
# entries are evicted regardless of TTL.
_CACHE_MAX_SIZE = 128

# Treat a cached URL as stale this many seconds before its actual expiry so
# playback is never handed a URL that dies mid-stream.  Applied consistently
# by every read path (_get_cached, is_expired, resolve).
_EXPIRY_BUFFER_SECONDS = 300

# The retry schedule _resolve_sync walks: the initial attempt (no wait)
# followed by two retries after short back-offs.  Read through this
# module's own name by the _retry_delays engine, so a patched schedule
# steers both backends identically.
_RETRY_DELAYS: list[float] = [0, 1.0, 2.0]


def _is_stale(expires_at: float) -> bool:
    """True when a cached URL has passed, or is within the buffer of, expiry.

    Implemented by the Rust extension
    (``_ytm_player_native.is_stale``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return time.time() >= expires_at - _EXPIRY_BUFFER_SECONDS


def _wait_seconds(available_at: float) -> float:
    """Seconds until a stream URL behind a pre-roll ad can be fetched.

    The pure half of ``StreamInfo.wait_seconds``: the ``max(0.0, ... )``
    clamp over the distance from the module's own clock to
    *available_at* — never negative, zero when the stream is
    fetchable now.

    Implemented by the Rust extension
    (``_ytm_player_native.wait_seconds``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return max(0.0, available_at - time.time())


def _detect_stream_cookies() -> str | None:
    """Return the AuthManager-written stream cookiejar path if present, else None.

    AuthManager (services/auth.py) is the only component that ever decrypts
    browser cookies — during `ytm setup` or a validate()-triggered
    try_auto_refresh(). This function never triggers extraction itself; if
    the file doesn't exist (e.g. extraction failed, or extraction hasn't
    happened yet), resolution proceeds without cookies rather than
    decrypting independently.
    """
    from ytm_player.config.paths import STREAM_COOKIES_FILE

    return str(STREAM_COOKIES_FILE) if STREAM_COOKIES_FILE.exists() else None


def _detect_session_record() -> tuple[str, str] | None:
    """``(auth.json path, account.json path)`` the stream cookiejar is checked
    against — AuthManager's record vouches for the jar by hash (see
    auth.read_vouched_stream_jar). No existence check: a missing file is
    part of what the check decides on. Tests point this at a scratch pair.
    """
    from ytm_player.config.paths import ACCOUNT_FILE, AUTH_FILE

    return str(AUTH_FILE), str(ACCOUNT_FILE)


def _session_cookiejar_signature() -> tuple[Any, ...] | None:
    """Signature of the stream cookiejar the resolver should be using right
    now — ``(jar path, jar, auth.json, account.json)`` with each file as an
    ``(st_ino, st_mtime_ns, st_size)`` triple — or None.

    None unless ``[yt_dlp] use_session_cookies`` is on (streaming is anonymous
    by default) and no explicit ``[yt_dlp] cookies_file`` is configured (that
    file is the user's own and goes through yt-dlp's ``cookiefile``), or when
    the jar file doesn't exist. Compared on every _get_ydl() call so a change
    to ANY of the three files — a jar rewritten by ``ytm setup`` or a
    mid-session auto-refresh, but also an auth.json or account.json replaced
    on its own — rebuilds the instance and reruns the vouching check instead
    of the cached YoutubeDL instance keeping cookies it may no longer use.
    """
    settings = get_settings().yt_dlp
    if not settings.use_session_cookies or normalize_cookiefile(settings.cookies_file):
        return None
    path = _detect_stream_cookies()
    if path is None:
        return None
    from ytm_player.services.auth import file_signature

    session = _detect_session_record()
    auth_path, record_path = session if session is not None else (None, None)
    return (path, file_signature(path), file_signature(auth_path), file_signature(record_path))


def _ydl_cookiejar_current(cached_sig: Any, current_sig: Any) -> bool:
    """The cookiejar-signature reuse gate behind ``StreamResolver._get_ydl``.

    True when the signature the cached YoutubeDL instance was built
    against still equals the signature the resolver should be using
    right now — ``None`` matching ``None`` (an instance built anonymous
    while streaming is still anonymous), element-wise over the
    signature tuples otherwise — so a change to ANY of the three files
    the signature stats (a stream cookiejar rewritten by ``ytm setup``
    or a mid-session auto-refresh, but also an auth.json or
    account.json replaced on its own) answers False and the caller
    discards the instance and rebuilds one that loads the new cookies.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_ydl_cookiejar_current``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    The signature probe itself (the settings read and the three file
    stats — ``_session_cookiejar_signature``, evaluated by the caller
    only when an instance exists) and the instance state stay with the
    caller.
    """
    return cached_sig == current_sig


def _ydl_publish_verdict(instance: Any, snapshot_generation: Any, current_generation: Any) -> str:
    """The mid-build publication verdict behind ``StreamResolver._get_ydl``.

    The second locked section's decision, consulted after the unlocked
    ``_build_ydl_opts()`` pass: ``"adopt"`` when another thread's build
    published an instance meanwhile (any instance is strictly fresher
    than this caller's in-flight opts — never construct a duplicate),
    ``"retry"`` when a ``_reset_ydl()`` landed while the opts were
    building (the just-built opts already have pre-reset settings baked
    in, so the build is discarded and the loop re-reads current
    settings), and ``"install"`` otherwise — the built instance is
    published with the signature it vouched for.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_ydl_publish_verdict``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    The locks, the ``_active_resolves`` accounting, the YoutubeDL
    construction and the cached instance fields stay with the caller.
    """
    if instance is not None:
        return "adopt"
    if current_generation != snapshot_generation:
        return "retry"
    return "install"


class _YtDlpLogger:
    """Routes yt-dlp's own diagnostic messages into our logger.

    With quiet=True/no_warnings=True and no logger, yt-dlp discards its own
    warnings and errors entirely (report_warning/to_stderr only check
    params['logger'] before checking no_warnings/quiet) — so failures like
    "Signature solving failed" or "Skipping client ... since it does not
    support cookies" never reached ytm.log. Mirrors the log_handler pattern
    already used for mpv in player.py.
    """

    def debug(self, message: str) -> None:
        # yt-dlp routes both normal progress lines ("Downloading webpage")
        # and its own "[debug] ..."-prefixed messages through here.
        logger.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        # A default logger doesn't implement to_screen(), so without this
        # method, yt-dlp's own info-level messages during extract_info()
        # would be silently dropped rather than surfacing in ytm.log.
        logger.info("yt-dlp: %s", message)

    def warning(self, message: str, **_kwargs: object) -> None:
        logger.warning("yt-dlp: %s", message)

    def error(self, message: str) -> None:
        logger.error("yt-dlp: %s", message)


# bestaudio first (real audio-only DASH formats when the resolve is
# authenticated), then a combined audio+video format as the fallback for
# clients that expose no audio-only stream. mpv runs with video=False, so
# the video track of a combined format is never rendered.
_COMBINED_FALLBACK = "best[vcodec!=none][acodec!=none]"

QUALITY_FORMATS = {
    "high": f"bestaudio/{_COMBINED_FALLBACK}",
    "medium": f"bestaudio[abr<=128]/bestaudio/{_COMBINED_FALLBACK}",
    "low": f"bestaudio[abr<=64]/bestaudio/{_COMBINED_FALLBACK}",
}


def _validate_quality(value: str) -> None:
    """The quality-input gate the resolver's ``quality`` setter raises through.

    Raises ``ValueError`` for a value outside ``QUALITY_FORMATS``, the
    reference's f-string message naming the value and the valid tiers.

    Implemented by the Rust extension
    (``_ytm_player_native.validate_quality``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if value not in QUALITY_FORMATS:
        raise ValueError(f"Unknown quality '{value}'. Choose from: {list(QUALITY_FORMATS)}")


def _base_ydl_opts(quality: str, logger: Any) -> dict:
    """The base yt-dlp options dict every resolve runs with.

    The pure half of ``StreamResolver._build_ydl_opts``: the ten-key
    dict — the quality's format selector (the ``QUALITY_FORMATS``
    lookup with the ``"high"`` default), the quiet/no-warning flags,
    the logger, the extraction gates and the Android-client
    ``extractor_args`` — before ``apply_configured_yt_dlp_options``
    layers the configured extras on top. The logger object passes
    through as the object the caller built.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_base_opts``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    opts = {
        "format": QUALITY_FORMATS.get(quality, QUALITY_FORMATS["high"]),
        "quiet": True,
        "no_warnings": True,
        "logger": logger,
        "extract_flat": False,
        "noplaylist": True,
        # Skip video-related processing.
        "skip_download": True,
        # Avoid writing any files to disk.
        "writeinfojson": False,
        "writethumbnail": False,
        # Android client provides a non-PoT fallback path (legacy format 18)
        # that unblocks madeForKids content which the default web client refuses.
        "extractor_args": {"youtube": {"player_client": ["default", "android"]}},
    }
    return opts


def _stream_info_fields(info: Any, video_id: str) -> tuple[Any, ...] | None:
    """The eight ``StreamInfo`` field values behind a successful extraction.

    The pure half of ``StreamResolver._try_resolve``: the stream-URL
    read (the direct ``url`` key, the ``requested_formats`` fallback
    walk when it is falsy, ``None`` when no URL survived), the
    ``float(... or 0)`` available_at coercion, the acodec/audio-ext
    format naming, the ``abr``/``tbr`` and ``duration`` integer
    coercions, the bare ``thumbnail`` read and the module-clock
    expiry stamp — the values in the dataclass's declared field order,
    passed through as the objects they are; the ``StreamInfo``
    construction stays with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_info_fields``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    stream_url: Any = info.get("url", "")
    available_at = float(info.get("available_at") or 0)
    if not stream_url:
        # Some formats nest the URL under requested_formats.
        formats = info.get("requested_formats") or []
        for fmt in formats:
            if fmt.get("vcodec") == "none" or fmt.get("acodec") != "none":
                stream_url = fmt.get("url", "")
                available_at = float(fmt.get("available_at") or 0)
                break

    if not stream_url:
        return None

    # Determine audio format and bitrate from the info dict.
    acodec = info.get("acodec", "unknown")
    audio_ext = info.get("audio_ext") or info.get("ext", "unknown")
    abr = int(info.get("abr") or info.get("tbr") or 0)
    duration = int(info.get("duration") or 0)
    thumbnail = info.get("thumbnail")

    # Pick a readable format name.
    fmt_name = acodec if acodec != "none" else audio_ext

    expires_at = time.time() + _CACHE_TTL_SECONDS

    return (stream_url, video_id, fmt_name, abr, duration, expires_at, thumbnail, available_at)


def _download_error_hint(message: str, session_cookiejar_active: bool) -> str:
    """The stale-cookiejar diagnostic a signed-in DownloadError appends.

    The pure half of ``StreamResolver._try_resolve``'s error handler:
    the hint — with its leading blank and the ``ytm setup`` advice —
    only when the session cookiejar is in use and the error text
    matches the auth-related pattern (``sign in``, ``login_required``
    or ``confirm you``, case-insensitively); the empty string
    otherwise. The cookiejar-signature probe and the ``str(exc)``
    conversion stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.download_error_hint``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if session_cookiejar_active and re.search(r"sign in|login_required|confirm you", message, re.I):
        return " (stream cookiejar may be stale/revoked — try `ytm setup` to refresh it)"
    return ""


def _cache_expired_ids(cache: dict[str, StreamInfo]) -> list[str]:
    """Video IDs of the cache entries whose expiry has passed.

    The pure half of ``StreamResolver._put_cache``'s maintenance walk
    and of ``prune_expired``: the ids of every entry whose
    ``expires_at`` the module's own clock has reached — the ``del``
    targets of the reference's prune loop — in the cache's own
    iteration order. The clock is resolved through this module's own
    ``time`` name at call time, so a patched or frozen clock steers
    both backends identically; the cache state itself stays
    Python-side in the resolver (the caller holds the lock and
    performs the deletions).

    Implemented by the Rust extension
    (``_ytm_player_native.stream_cache_expired_ids``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    now = time.time()
    return [vid for vid, info in cache.items() if now >= info.expires_at]


def _cache_cap_evictions(cache: dict[str, StreamInfo]) -> list[str]:
    """Video IDs to evict for the cache being over its size cap.

    The pure half of ``StreamResolver._put_cache``'s maintenance
    walk's second phase, run once the expired entries are pruned:
    the ids of the oldest entries by ``expires_at`` that bring the
    cache back within ``_CACHE_MAX_SIZE`` — the cap read through this
    module's own name, so a patched cap steers both backends
    identically — ties kept in the cache's own iteration order by
    the sort's stability. An empty list when the cache is within its
    cap; the cache state itself stays Python-side in the resolver.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_cache_cap_evictions``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if len(cache) <= _CACHE_MAX_SIZE:
        return []
    sorted_ids = sorted(cache, key=lambda vid: cache[vid].expires_at)
    excess = len(cache) - _CACHE_MAX_SIZE
    return sorted_ids[:excess]


def _retry_delays() -> list[tuple[int, float]]:
    """The ``(attempt, delay)`` iteration the resolve loop walks.

    The pure half of ``StreamResolver._resolve_sync``'s retry loop:
    the retry schedule — the module's own ``_RETRY_DELAYS`` — paired
    with each attempt's index exactly like the reference's
    ``enumerate`` walk, so the loop can break out early on the first
    success without walking the rest of the schedule. The schedule is
    read through this module's own ``_RETRY_DELAYS`` name (the walk
    itself the iterator protocol, so any iterable schedule steers
    both backends identically); the sleep gate, the ``time.sleep``
    call and the extraction attempts stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_retry_delays``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return list(enumerate(_RETRY_DELAYS))


def _pending_join_verdict(entry: Any, generation: Any) -> Any:
    """The pending-join verdict behind ``StreamResolver._current_pending``.

    The dedup gate both ``_resolve``'s join arm and ``prefetch``'s
    already-resolving check consult: the in-flight resolve's future
    when the pending slot holds one that started under the caller's
    own generation — so the caller awaits it instead of extracting
    again — and ``None`` otherwise (an empty slot, or one from before
    a ``clear_cache()`` bumped the generation, whose result comes
    from the old resolver setup). The generation comparison is
    CPython's own ``!=``; the slot read (the ``_pending`` dict
    lookup) and the awaits stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_pending_join_verdict``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if entry is None or entry[0] != generation:
        return None
    return entry[1]


def _pending_drop_verdict(entry: Any, future: Any) -> bool:
    """The slot-drop verdict behind ``_resolve``'s finally cleanup.

    True only when the pending slot still holds this owner's very
    future — the identity comparison, never equality — so the cleanup
    of a resolve from before a ``clear_cache()`` (whose slot a newer
    same-video request has taken over) leaves the newer request's
    entry in place for its own waiters. The ``del`` and the slot
    state stay with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_pending_drop_verdict``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return entry is not None and entry[1] is future


def _pending_publication_gate(future: Any) -> bool:
    """The unresolved gate behind the shared pending future's publications.

    ``not future.done()`` — True while the future still wants a
    resolution, so the owner's ``set_result`` (the try arm),
    ``set_exception`` (the except arm) or ``cancel`` (the finally
    arm, for an owner cancelled before the future resolved) may land
    on it; False once the future is resolved, cancelled or
    exception-set, so a publication never raises
    ``InvalidStateError`` over a resolution that already landed. The
    publications themselves and the future state stay with the
    caller.

    Implemented by the Rust extension
    (``_ytm_player_native.stream_pending_publication_gate``) when it
    is built; this pure-Python transcription is the fallback
    otherwise.
    """
    return not future.done()


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """Resolved stream information for a YouTube Music track."""

    url: str
    video_id: str
    format: str  # e.g., "opus", "m4a"
    bitrate: int  # kbps
    duration: int  # seconds
    expires_at: float  # unix timestamp
    thumbnail_url: str | None = None
    # yt-dlp's available_at: googlevideo answers 403 until this unix time
    # (streams served behind a pre-roll ad). 0 = fetchable now.
    available_at: float = 0.0

    def wait_seconds(self) -> float:
        """Seconds until the stream URL can be fetched.

        The wait computation is resolved by the ``_wait_seconds``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise).
        """
        return _wait_seconds(self.available_at)


class StreamResolver:
    """Resolves YouTube Music video IDs to direct audio stream URLs.

    Uses the yt-dlp Python API to extract stream information without
    downloading. Caches results in memory with automatic expiry.
    """

    def __init__(self, quality: str = "high") -> None:
        self._quality = quality
        self._cache: dict[str, StreamInfo] = {}
        self._cache_lock = threading.Lock()
        # In-flight async resolves, keyed by video_id, tagged with the
        # _cache_generation they started under so a request made after
        # clear_cache() never joins (or gets cleaned up by) a stale one.
        self._pending: dict[str, tuple[int, asyncio.Future[StreamInfo | None]]] = {}
        # yt_dlp.YoutubeDL — typed Any because yt-dlp ships no stubs.
        self._ydl: Any | None = None
        self._ydl_lock = threading.Lock()
        # Count of extract_info() calls currently running against self._ydl,
        # on background threads — guards _reset_ydl() against closing an
        # instance a resolve is still actively using. See _reset_ydl.
        self._active_resolves = 0
        # Bumped by _reset_ydl(). _get_ydl() snapshots this before its
        # unlocked opts build and re-checks it after — if it changed, a
        # reset landed while the build was in flight and the just-built
        # opts already have stale settings baked in, so the build is
        # retried instead of caching a stale instance.
        self._ydl_generation = 0
        # Signature of the stream cookiejar file self._ydl was built against
        # (see _session_cookiejar_signature); None when none was loaded.
        self._ydl_cookiejar_sig: tuple[Any, ...] | None = None
        # YoutubeDL is not thread-safe; a play resolve and a prefetch can run
        # on separate threads against the shared instance, so extract_info()
        # calls are serialized. Only the network call is held under it.
        self._extract_lock = threading.Lock()
        # Bumped by clear_cache(). A resolve that started before the clear
        # must not put its (pre-clear) result back into the emptied cache.
        self._cache_generation = 0

    @property
    def quality(self) -> str:
        return self._quality

    @quality.setter
    def quality(self, value: str) -> None:
        _validate_quality(value)
        if value != self._quality:
            self._quality = value
            self._reset_ydl()

    def _build_ydl_opts(self) -> dict:
        """Build yt-dlp options for audio extraction.

        The base options dict — the ten keys every resolve runs with —
        is resolved by the ``_base_ydl_opts`` engine (native when the
        Rust extension is built, the pure-Python transcription
        otherwise); this method layers the configured extras on top.
        """
        settings = get_settings().yt_dlp
        opts = _base_ydl_opts(self._quality, _YtDlpLogger())
        return apply_configured_yt_dlp_options(opts, settings)

    def _build_ydl(self, opts: dict) -> tuple[Any, tuple[Any, ...] | None]:
        """Construct a YoutubeDL instance from *opts* and, when
        ``[yt_dlp] use_session_cookies`` is on, load the stream cookiejar into it.

        The jar is loaded into the instance's cookiejar rather than passed as
        ``cookiefile``: YoutubeDL.close() writes ``cookiefile`` back to disk
        from memory, so a discarded instance closing later (quality change,
        failure reset) would overwrite a jar that ``ytm setup`` or a session
        refresh had just replaced. Loaded this way the file is read-only to
        yt-dlp. A user-configured ``[yt_dlp] cookies_file`` still goes
        through ``cookiefile`` — that file is theirs, and write-back is
        yt-dlp's normal behaviour for it.

        Only a jar the sign-in record vouches for is loaded (see
        auth.read_vouched_stream_jar); the bytes loaded are the bytes that
        were checked, and the signature cached with the instance is the one
        of that exact snapshot — never a separate stat taken before or after.

        Returns the instance and the signature it was built against.
        """
        import yt_dlp  # Lazy import

        # yt-dlp's _Params TypedDict is internal; the dict we build is a
        # plain options dict that yt-dlp accepts at runtime.
        ydl = yt_dlp.YoutubeDL(opts)  # type: ignore[arg-type]
        sig = _session_cookiejar_signature()
        if sig is not None:
            from ytm_player.services.auth import read_vouched_stream_jar

            vouched = read_vouched_stream_jar(sig[0], _detect_session_record())
            sig = vouched.signature
            if vouched.payload is not None:
                try:
                    # load()'s stub types filename as str | None, but its
                    # open() accepts a file object directly at runtime
                    # (non-path-like branch yields it as is) — verified
                    # against yt-dlp source.
                    ydl.cookiejar.load(
                        io.StringIO(vouched.payload.decode("utf-8")),  # type: ignore[arg-type]
                        ignore_discard=True,
                        ignore_expires=True,
                    )
                except Exception:
                    logger.warning("Could not load stream cookiejar %s", sig[0], exc_info=True)
        return ydl, sig

    def _get_ydl(self) -> Any:
        """Return a reusable YoutubeDL instance, creating it lazily, with
        _active_resolves already incremented on the caller's behalf.

        Incrementing _active_resolves here, still inside self._ydl_lock, at
        every return point closes a TOCTOU gap: with a separate, later
        increment, _reset_ydl() could observe _active_resolves == 0 for an
        instance a caller already held and was about to call extract_info()
        on, and close it out from under that in-flight call. Callers must
        decrement in a finally block (see _try_resolve).

        _build_ydl_opts() runs OUTSIDE self._ydl_lock, so a _reset_ydl()
        (clear_cache(), the quality setter) can land mid-build. If self._ydl
        is still None at that point there is nothing to reset and the reset
        would be silently lost — the in-flight build would then cache an
        instance with pre-reset settings. The generation check below
        catches that: a reset landing mid-build invalidates that build's
        result and the loop re-reads current settings.

        The cookiejar-signature reuse gate is resolved by the
        ``_ydl_cookiejar_current`` engine and the post-build publication
        decision by the ``_ydl_publish_verdict`` engine (native when the
        Rust extension is built, the pure-Python transcriptions
        otherwise); the signature probe, the locks, the opts build, the
        YoutubeDL construction and the _active_resolves accounting stay
        here.
        """
        while True:
            with self._ydl_lock:
                if self._ydl is not None:
                    if _ydl_cookiejar_current(
                        self._ydl_cookiejar_sig, _session_cookiejar_signature()
                    ):
                        self._active_resolves += 1
                        return self._ydl
                    # The jar file changed since this instance loaded it
                    # (ytm setup / auto-refresh wrote a new one): drop the
                    # instance so the rebuild below loads the new cookies.
                    self._discard_ydl_locked()
                generation = self._ydl_generation

            opts = self._build_ydl_opts()

            with self._ydl_lock:
                verdict = _ydl_publish_verdict(self._ydl, generation, self._ydl_generation)
                if verdict == "adopt":
                    self._active_resolves += 1
                    return self._ydl
                if verdict == "retry":
                    continue  # reset landed mid-build; opts are stale, retry
                self._ydl, self._ydl_cookiejar_sig = self._build_ydl(opts)
                self._active_resolves += 1
                return self._ydl

    def _reset_ydl(self) -> None:
        """Discard the cached YoutubeDL instance, closing it only if safe.

        clear_cache() (and therefore this) can be triggered from the UI
        thread while a background resolve (a prefetch or an in-flight play)
        is still mid extract_info() on the SAME instance. YoutubeDL.close()
        tears down the shared request director's connection pool; closing
        it out from under a live request left that request hanging until an
        underlying socket timeout gave up (a ~30s stall). Detaching the
        reference is enough: the next _get_ydl() builds a fresh instance,
        the in-flight call keeps its own local reference to the old one and
        finishes normally, and that instance's pool is cleaned up by garbage
        collection once nothing references it.

        Always bumps _ydl_generation, even when self._ydl is already None —
        that's exactly the case that used to lose the reset silently. See
        _get_ydl for the other half.
        """
        with self._ydl_lock:
            self._ydl_generation += 1
            self._discard_ydl_locked()

    def _discard_ydl_locked(self) -> None:
        """Detach self._ydl (caller holds _ydl_lock), closing it only when no
        resolve is still using it — see _reset_ydl."""
        ydl, self._ydl = self._ydl, None
        self._ydl_cookiejar_sig = None
        if ydl is None or self._active_resolves > 0:
            return
        try:
            ydl.close()
        except Exception:
            pass

    def _resolve_sync(self, video_id: str) -> StreamInfo | None:
        """Synchronous stream resolution (runs in a thread) with retry.

        The retry-delay iteration is resolved by the ``_retry_delays``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); the sleeps, the
        extraction attempts and the early return stay here.
        """

        if not VALID_VIDEO_ID.match(video_id):
            logger.warning("Invalid video_id rejected: %r", video_id)
            return None
        url = f"https://music.youtube.com/watch?v={video_id}"

        for attempt, delay in _retry_delays():
            if delay > 0:
                time.sleep(delay)
            info = self._try_resolve(url, video_id, attempt)
            if info is not None:
                return info
        return None

    def _try_resolve(self, url: str, video_id: str, attempt: int) -> StreamInfo | None:
        """Single resolution attempt.

        The extract-info field assembly is resolved by the
        ``_stream_info_fields`` engine (native when the Rust extension
        is built, the pure-Python transcription otherwise) and the
        DownloadError hint by the ``_download_error_hint`` engine; the
        extraction call, the active-resolve accounting and the logging
        stay here.
        """
        import yt_dlp  # Lazy import: needed for exception types

        try:
            ydl = self._get_ydl()  # increments _active_resolves; see _get_ydl's docstring
            try:
                with self._extract_lock:
                    info = ydl.extract_info(url, download=False)
            finally:
                with self._ydl_lock:
                    self._active_resolves -= 1

            if info is None:
                logger.error("yt-dlp returned no info for video_id=%s", video_id)
                return None

            # Formats served behind a pre-roll ad carry yt-dlp's available_at;
            # googlevideo answers 403 until it passes. It is recorded on the
            # StreamInfo and resolve() waits for it, not this worker thread.
            fields = _stream_info_fields(info, video_id)
            if fields is None:
                logger.error("No stream URL found for video_id=%s", video_id)
                return None

            return StreamInfo(*fields)

        except yt_dlp.utils.DownloadError as exc:  # type: ignore[attr-defined]
            hint = _download_error_hint(str(exc), _session_cookiejar_signature() is not None)
            logger.warning(
                "yt-dlp download error for video_id=%s (attempt %d): %s%s",
                video_id,
                attempt + 1,
                exc,
                hint,
            )
            return None
        except Exception:
            logger.warning(
                "Unexpected error resolving stream for video_id=%s (attempt %d)",
                video_id,
                attempt + 1,
                exc_info=True,
            )
            return None

    def _get_cached(self, video_id: str) -> StreamInfo | None:
        """Return cached StreamInfo if it exists and isn't (near-)expired."""
        with self._cache_lock:
            cached = self._cache.get(video_id)
            if cached is None:
                return None
            if _is_stale(cached.expires_at):
                del self._cache[video_id]
                return None
            return cached

    def _put_cache(self, info: StreamInfo, generation: int | None = None) -> None:
        """Store a StreamInfo in the cache, evicting stale/excess entries.

        *generation* is the _cache_generation the caller read before it
        started resolving; if clear_cache() ran in between, the result is
        from the old resolver setup and is dropped instead of repopulating
        the cache that was just cleared.
        """
        with self._cache_lock:
            if generation is not None and generation != self._cache_generation:
                logger.debug(
                    "Dropping resolve result for %s: cache cleared mid-resolve", info.video_id
                )
                return
            self._cache[info.video_id] = info

            # Prune expired entries on every write to prevent unbounded
            # growth, then, if still over the cap, evict the oldest entries
            # by expires_at — both maintenance phases resolved by the
            # _cache_expired_ids / _cache_cap_evictions engines (native when
            # the Rust extension is built, the pure-Python transcriptions
            # otherwise); the cache state itself stays Python-side in the
            # resolver.
            for vid in _cache_expired_ids(self._cache):
                del self._cache[vid]
            for vid in _cache_cap_evictions(self._cache):
                del self._cache[vid]

    def resolve_sync(self, video_id: str) -> StreamInfo | None:
        """Resolve a video ID to a StreamInfo, using cache when possible.

        This is the synchronous version. Prefer `resolve()` in async code.
        Like `resolve()`, it returns once the stream URL can be fetched.
        """
        cached = self._get_cached(video_id)
        if cached is not None:
            logger.debug("Cache hit for video_id=%s", video_id)
            time.sleep(cached.wait_seconds())
            return cached

        logger.debug("Cache miss for video_id=%s, resolving via yt-dlp", video_id)
        generation = self._cache_generation
        info = self._resolve_sync(video_id)
        if info is not None:
            self._put_cache(info, generation)
            time.sleep(info.wait_seconds())
        return info

    def is_expired(self, video_id: str) -> bool:
        """Check if a cached stream URL has expired or will expire soon."""
        with self._cache_lock:
            cached = self._cache.get(video_id)
            if cached is None:
                return True
            return _is_stale(cached.expires_at)

    async def resolve(self, video_id: str) -> StreamInfo | None:
        """Resolve a video ID to a StreamInfo whose URL can be fetched now.

        Streams YouTube serves behind a pre-roll ad answer 403 until their
        available_at. yt-dlp's own downloader sleeps through that, but mpv
        opens the URL at once, so wait here: on the event loop, where the
        wait never holds a worker thread and shutdown can cancel it. A
        prefetched entry is usually past it already.
        """
        info = await self._resolve(video_id)
        if info is not None and (wait := info.wait_seconds()) > 0:
            logger.debug("Waiting %.1fs for video_id=%s to become available", wait, video_id)
            await asyncio.sleep(wait)
        return info

    async def _resolve(self, video_id: str) -> StreamInfo | None:
        """Resolve a video ID to a StreamInfo asynchronously, without waiting
        for its available_at.

        Runs the synchronous yt-dlp extraction in a thread to avoid
        blocking the event loop. Deduplicates concurrent requests for
        the same video_id.
        """
        # _get_cached already applies the expiry buffer, so a hit here is fresh
        # enough to hand to playback; near-expired entries were evicted and fall
        # through to a fresh resolve below.
        cached = self._get_cached(video_id)
        if cached is not None:
            logger.debug("Cache hit for video_id=%s", video_id)
            return cached

        # Deduplicate concurrent requests for the same video.  Shield the
        # shared future: cancelling a waiter task would otherwise propagate
        # into the bare future (Task.cancel cancels what it awaits) and break
        # the owner's set_result with InvalidStateError.  A pending resolve
        # from before a clear_cache() is not joined: its result comes from
        # the old resolver setup, so this request starts its own and takes
        # over the slot (the old owner's cleanup below checks identity).
        # The join decision is resolved by the ``_pending_join_verdict``
        # engine, the publications by the ``_pending_publication_gate``
        # engine and the cleanup's drop by the ``_pending_drop_verdict``
        # engine (native when the Rust extension is built, the
        # pure-Python transcriptions otherwise); the slot state, the
        # awaits and the mutations stay here.
        generation = self._cache_generation
        pending = self._current_pending(video_id, generation)
        if pending is not None:
            return await asyncio.shield(pending)

        logger.debug("Cache miss for video_id=%s, resolving via yt-dlp", video_id)
        future: asyncio.Future[StreamInfo | None] = asyncio.get_running_loop().create_future()
        self._pending[video_id] = (generation, future)
        try:
            info = await asyncio.to_thread(self._resolve_sync, video_id)
            if info is not None:
                self._put_cache(info, generation)
            if _pending_publication_gate(future):
                future.set_result(info)
            return info
        except Exception as exc:
            if _pending_publication_gate(future):
                future.set_exception(exc)
            raise
        finally:
            # The drop verdict, the del, then the publication gate — the
            # reference's statement order, so a raising done() never skips
            # the del.
            entry = self._pending.get(video_id)
            if _pending_drop_verdict(entry, future):
                del self._pending[video_id]
            if _pending_publication_gate(future):
                # Owner was cancelled before the future resolved (a BaseException
                # such as CancelledError bypasses the except above).  Cancel the
                # shared future so concurrent waiters get CancelledError instead
                # of hanging forever on an orphaned, never-resolved future.
                future.cancel()

    def _current_pending(
        self, video_id: str, generation: int
    ) -> asyncio.Future[StreamInfo | None] | None:
        """The in-flight resolve for *video_id* if it started under *generation*.

        The join verdict is resolved by the ``_pending_join_verdict``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); the slot read stays
        with the caller.
        """
        return _pending_join_verdict(self._pending.get(video_id), generation)

    async def prefetch(self, video_id: str) -> None:
        """Resolve a video ID in the background without blocking the caller.

        Used to pre-cache the next track's stream URL so playback starts
        instantly when the user hits next or the current track ends.
        """
        if self._get_cached(video_id) is not None:
            return  # Already cached, nothing to do.
        if self._current_pending(video_id, self._cache_generation) is not None:
            return  # Already being resolved.
        try:
            await self._resolve(video_id)  # caches without waiting; playback waits in resolve()
        except Exception:
            logger.debug("Prefetch failed for video_id=%s", video_id, exc_info=True)

    @staticmethod
    def warm_import() -> None:
        """Import yt_dlp eagerly to avoid the 200-400ms cold-start penalty."""
        try:
            import yt_dlp  # noqa: F401
        except ImportError:
            logger.warning("yt-dlp is not installed")

    def invalidate(self, video_id: str) -> None:
        """Remove a specific video ID from the cache."""
        with self._cache_lock:
            self._cache.pop(video_id, None)

    def clear_cache(self) -> None:
        """Remove all entries from the cache and reset the yt-dlp instance."""
        with self._cache_lock:
            self._cache.clear()
            self._cache_generation += 1
        self._reset_ydl()

    def prune_expired(self) -> int:
        """Remove expired entries from the cache. Returns number removed.

        The expired-entry walk is resolved by the ``_cache_expired_ids``
        engine (native when the Rust extension is built, the
        pure-Python transcription otherwise); the deletions and the
        count stay Python-side in the resolver.
        """
        with self._cache_lock:
            expired = _cache_expired_ids(self._cache)
            for vid in expired:
                del self._cache[vid]
            return len(expired)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same resolver-glue engines; these
# bindings replace the pure-Python transcriptions above when the extension
# is importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        download_error_hint as _native_download_error_hint,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        is_stale as _native_is_stale,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_base_opts as _native_stream_base_opts,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_cache_cap_evictions as _native_stream_cache_cap_evictions,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_cache_expired_ids as _native_stream_cache_expired_ids,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_info_fields as _native_stream_info_fields,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_pending_drop_verdict as _native_stream_pending_drop_verdict,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_pending_join_verdict as _native_stream_pending_join_verdict,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_pending_publication_gate as _native_stream_pending_publication_gate,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_retry_delays as _native_stream_retry_delays,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_ydl_cookiejar_current as _native_stream_ydl_cookiejar_current,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        stream_ydl_publish_verdict as _native_stream_ydl_publish_verdict,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        validate_quality as _native_validate_quality,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        wait_seconds as _native_wait_seconds,
    )
except ImportError:
    pass
else:
    _is_stale = _native_is_stale  # type: ignore[assignment]
    _wait_seconds = _native_wait_seconds  # type: ignore[assignment]
    _ydl_cookiejar_current = _native_stream_ydl_cookiejar_current  # type: ignore[assignment]
    _ydl_publish_verdict = _native_stream_ydl_publish_verdict  # type: ignore[assignment]
    _validate_quality = _native_validate_quality  # type: ignore[assignment]
    _base_ydl_opts = _native_stream_base_opts  # type: ignore[assignment]
    _stream_info_fields = _native_stream_info_fields  # type: ignore[assignment]
    _download_error_hint = _native_download_error_hint  # type: ignore[assignment]
    _cache_expired_ids = _native_stream_cache_expired_ids  # type: ignore[assignment]
    _cache_cap_evictions = _native_stream_cache_cap_evictions  # type: ignore[assignment]
    _retry_delays = _native_stream_retry_delays  # type: ignore[assignment]
    _pending_join_verdict = _native_stream_pending_join_verdict  # type: ignore[assignment]
    _pending_drop_verdict = _native_stream_pending_drop_verdict  # type: ignore[assignment]
    _pending_publication_gate = _native_stream_pending_publication_gate  # type: ignore[assignment]
