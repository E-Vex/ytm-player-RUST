"""Async wrapper around ytmusicapi providing all YouTube Music functionality.

The service's pure computation is dual-implemented: the
mutation-failure classifier ``_classify_mutation_failure``, the
retry/renewal bookkeeping — the auth-expiry sniff ``_is_auth_expired``
behind every expired-session retry, the renewal-cooldown gate
``_renewal_cooldown_gate`` and stamp ``_renewal_cooldown_stamp``
behind ``_renewed_client_for``, and the consecutive-failure step
``_api_failure_step`` behind ``_run``'s except arm — the patch-window
in-flight bookkeeping — the admission step ``_patch_window_enter``
behind ``_call``'s gate arm and the release step
``_patch_window_exit`` behind its finally arm — the browse wrappers'
cores — the watch-playlist kwargs assembly
``_watch_playlist_kwargs``, the result-tracks extraction
``_result_tracks``, the playlist-tracks gate ``_playlist_tracks``,
the radio seed pool ``_radio_seed_pool``, the discovery source
rotation ``_discovery_source_order``, the chart-shelf assembly
``_chart_shelves`` and its round-robin shelf pick
``_chart_shelf_index``, the playable filters
``_discovery_playable_tracks`` / ``_chart_playable_tracks``, the
explore new-releases extraction ``_explore_new_releases`` and the
playlist sort-order lookup ``_playlist_order_params`` — and the
RNG-driven selection cores — the radio mix's multi-seed shuffle and
limit trim ``_radio_mix`` behind ``get_radio``'s result assembly, the
three-track seed sample ``_sample_seeds`` behind every discovery mix
source and the random artist pick ``_pick_artist`` behind the artist
mix, the RNG itself flowing through the ``random`` module so seeded
sequences steer both backends identically — are implemented by the
Rust extension
(``_ytm_player_native.classify_mutation_failure`` /
``_ytm_player_native.ytm_auth_expired`` /
``_ytm_player_native.ytm_renewal_cooldown_gate`` /
``_ytm_player_native.ytm_renewal_cooldown_stamp`` /
``_ytm_player_native.ytm_api_failure_step`` /
``_ytm_player_native.ytm_patch_window_enter`` /
``_ytm_player_native.ytm_patch_window_exit`` /
``_ytm_player_native.ytm_watch_playlist_kwargs`` /
``_ytm_player_native.ytm_result_tracks`` /
``_ytm_player_native.ytm_playlist_tracks`` /
``_ytm_player_native.ytm_radio_seed_pool`` /
``_ytm_player_native.ytm_radio_mix`` /
``_ytm_player_native.ytm_discovery_source_order`` /
``_ytm_player_native.ytm_chart_shelves`` /
``_ytm_player_native.ytm_chart_shelf_index`` /
``_ytm_player_native.ytm_discovery_playable_tracks`` /
``_ytm_player_native.ytm_chart_playable_tracks`` /
``_ytm_player_native.ytm_sample_seeds`` /
``_ytm_player_native.ytm_pick_artist`` /
``_ytm_player_native.ytm_explore_new_releases`` /
``_ytm_player_native.ytm_playlist_order_params``) when it is built
and by the pure-Python transcriptions below otherwise; the client
construction and renewal, the ``asyncio.to_thread`` dispatch, the
patch-window gate and the track normalisation stay Python-side in
the service.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import requests.exceptions
from ytmusicapi import YTMusic
from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError

from ytm_player.config.paths import AUTH_FILE
from ytm_player.config.settings import get_settings
from ytm_player.services.auth import AuthManager

logger = logging.getLogger(__name__)

# After this many consecutive API failures, re-create the YTMusic client
# in case the session has gone stale (expired cookies, broken connection).
_MAX_API_FAILURES_BEFORE_REINIT = 3
# After a failed automatic session renewal, expired calls in this window fail
# without repeating the browser cookie extraction and account probes.
_RENEWAL_COOLDOWN_SECONDS = 60.0

# Exception types that count as "expected API/network failure" — these increment
# the failure counter and may trigger a client reinit. Programming-error
# exceptions (TypeError, AttributeError, etc.) propagate without bumping the
# counter so bugs surface in development instead of silently triggering reinit.
#
# Note: KeyError is here because ytmusicapi's parsers raise KeyError on
# renderer-key mismatches whenever YouTube tweaks its response shape — that's
# an upstream API drift, not our bug. Without this entry the KeyError
# propagates out of _call as an "unexpected exception" and may surface as a
# hard crash in worker contexts.
_EXPECTED_API_EXCEPTIONS = (
    requests.exceptions.RequestException,  # covers ConnectionError, Timeout, HTTPError
    asyncio.TimeoutError,  # from wait_for
    KeyError,  # ytmusicapi parser drift on YouTube response shape changes
)

# Same as ``_EXPECTED_API_EXCEPTIONS`` plus the typed errors raised by
# ytmusicapi itself. Used by mutation methods (rate_song, add_playlist_items,
# remove_playlist_items) which classify the failure cause for per-cause
# UI toasts (see Task 4.11).
_EXPECTED_MUTATION_EXCEPTIONS = (
    *_EXPECTED_API_EXCEPTIONS,
    YTMusicServerError,
    YTMusicUserError,
)

# Result type for mutation methods (rate_song, add_playlist_items,
# remove_playlist_items). Replaces the bool contract introduced in Task 4.3
# so callers can show a per-cause toast instead of a generic failure message.
#
# - "success":       server accepted the mutation
# - "auth_required": user has no auth set up at all (run `ytm setup`)
# - "auth_expired":  HTTP 401/403 from the server (cookies/session stale)
# - "network":       requests.RequestException other than a read timeout
# - "timeout":       no confirmation in time (asyncio.TimeoutError from wait_for,
#                    or a requests ReadTimeout): the change may have gone through
# - "rejected":      the server answered STATUS_FAILED (remove_playlist_items only)
# - "server_error":  any other YTMusicServerError (4xx/5xx other than auth), or
#                    a malformed answer
# - "duplicate":     track already exists in the playlist (add_playlist_items only)
MutationResult = Literal[
    "success",
    "auth_required",
    "auth_expired",
    "network",
    "timeout",
    "rejected",
    "server_error",
    "duplicate",
]

# ytmusicapi formats _send_request errors as:
#     "Server returned HTTP <code>: <reason>.\n<body-error>"
# This regex captures the status code from the start of the message. If
# upstream changes the format, we fall through to "server_error", which is
# a sensible default.
_HTTP_STATUS_RE = re.compile(r"^Server returned HTTP (\d{3})\b")


def _classify_mutation_failure(exc: BaseException) -> MutationResult:
    """Map an exception raised by a mutation call to a MutationResult.

    Never returns "success".

    The auth-vs-server-error distinction is made by parsing the HTTP status
    out of YTMusicServerError's message string — ytmusicapi does not expose
    a typed AuthenticationError subclass, so this is the only way short of
    monkey-patching ``_send_request`` to surface the status code. The format
    is stable in ytmusicapi 1.x but explicitly fall through to
    "server_error" if the regex doesn't match.

    Implemented by the Rust extension
    (``_ytm_player_native.classify_mutation_failure``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if isinstance(exc, KeyError):
        # ytmusicapi parser drift: YouTube changed a response shape. That's a
        # server-side problem, not connectivity — classifying it "network"
        # would wrongly tell the user to "check your connection".
        return "server_error"
    if isinstance(exc, (asyncio.TimeoutError, requests.exceptions.ReadTimeout)):
        # No confirmation arrived in time. The thread behind ``_run`` is not
        # cancelled and the server may still apply the change.
        return "timeout"
    if isinstance(exc, requests.exceptions.RequestException):
        return "network"
    if isinstance(exc, YTMusicUserError):
        # _check_auth() raises this when auth_type is UNAUTHORIZED — i.e.
        # the user never set up auth at all. Other YTMusicUserError raises
        # are programming errors (invalid args) and should NOT reach here
        # because the mutation methods don't catch them; but if one does
        # slip through, "auth_required" is a reasonable fallback.
        return "auth_required"
    if isinstance(exc, YTMusicServerError):
        match = _HTTP_STATUS_RE.match(str(exc))
        if match:
            status = int(match.group(1))
            if status in (401, 403):
                return "auth_expired"
        return "server_error"
    # Shouldn't be reached — caller filters to known types — but be safe.
    return "server_error"


# Try to load the native Rust classifier via PyO3. If the compiled
# extension is unavailable (e.g. during development without a Rust
# toolchain), the pure-Python definition above stays.
try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        classify_mutation_failure as _native_classify_mutation_failure,
    )
except ImportError:
    pass
else:

    def _classify_mutation_failure(exc: BaseException) -> MutationResult:  # noqa: F811
        """Map an exception raised by a mutation call to a MutationResult
        (native Rust engine)."""
        return _native_classify_mutation_failure(exc)


# Suffix templates for the failure kinds. Cascade sites combine these with
# their own action-specific prefix (e.g. "Couldn't like — <suffix>").
_MUTATION_TOAST_SUFFIX: dict[MutationResult, str] = {
    "success": "",
    "auth_required": "sign in first (run `ytm setup`)",
    "auth_expired": "session expired, run `ytm setup` to sign in again",
    "network": "check your connection",
    "timeout": (
        "No confirmation in time; the change may have gone through. "
        "Reload to check before retrying."
    ),
    "rejected": "the playlist has changed or you can't edit it; reload and check",
    "server_error": "YouTube Music had a problem, try again",
    "duplicate": "track already in playlist",
}


def _is_auth_expired(exc: BaseException) -> bool:
    """The auth-expiry sniff behind every expired-session retry.

    The gate ``_call``'s except arm and ``get_playlist``'s ordered
    branch consult before renewing: a ``YTMusicServerError`` delegates
    to the mutation-failure classifier and answers whether it
    classified ``"auth_expired"``; a ``KeyError`` (ytmusicapi parser
    drift on a sign-in endpoint where data was expected) renders
    through ``str()``, lowercases and answers whether the message
    carries any of the three sign-in markers — ``signinendpoint``,
    ``sign in to`` or the ``'logged_in', 'value': '0'`` fragment;
    everything else answers ``False``.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_auth_expired``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if isinstance(exc, YTMusicServerError):
        return _classify_mutation_failure(exc) == "auth_expired"
    if isinstance(exc, KeyError):
        message = str(exc).lower()
        return (
            "signinendpoint" in message
            or "sign in to" in message
            or ("'logged_in', 'value': '0'" in message)
        )
    return False


def _renewal_cooldown_gate(retry_after: float) -> tuple[bool, float]:
    """The renewal-cooldown gate behind ``_renewed_client_for``.

    The bookkeeping that keeps a failed automatic renewal from being
    retried (with its browser cookie extraction and account probes)
    until the cooldown has elapsed: the remaining cooldown —
    ``retry_after`` minus the module clock's own ``time.monotonic()``
    read — and the ``> 0`` verdict naming whether the retry is still
    blocked, the pair the caller logs its ``%.0fs`` line from.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_renewal_cooldown_gate``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    remaining = retry_after - time.monotonic()
    return (remaining > 0, remaining)


def _renewal_cooldown_stamp() -> float:
    """The cooldown stamp a failed renewal sets.

    The write side of ``_renewed_client_for``'s cooldown bookkeeping:
    the module clock's own ``time.monotonic()`` plus the module's own
    ``_RENEWAL_COOLDOWN_SECONDS`` (a patched constant steers both
    backends identically) — the value the failed-renewal arm assigns
    so the gate above blocks the retries until it has passed.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_renewal_cooldown_stamp``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return time.monotonic() + _RENEWAL_COOLDOWN_SECONDS


def _api_failure_step(consecutive_failures: int) -> tuple[int, bool]:
    """The consecutive-failure step behind ``_run``'s except arm.

    The retry bookkeeping that counts the expected API failures and
    re-creates the YTMusic client once a streak of them suggests the
    session has gone stale: the bumped count and the verdict naming
    whether the streak has reached the module's own
    ``_MAX_API_FAILURES_BEFORE_REINIT`` (a patched cap steers both
    backends identically). The caller assigns the bump, and the reinit
    branch's own reset to 0 — after the client clear — stays with it,
    at the reference's own points.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_api_failure_step``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    bumped = consecutive_failures + 1
    return (bumped, bumped >= _MAX_API_FAILURES_BEFORE_REINIT)


def _patch_window_enter(inflight: int) -> int:
    """The in-flight admission step behind ``_call``'s gate arm.

    The input half of the patch-window bookkeeping: every normal
    client call counts itself in — ``inflight + 1`` — so a later
    ``get_playlist(order=...)`` patch window can wait for the calls
    already past the gate to drain before it touches
    ``client._send_request``. The caller assigns the bumped count to
    its ``_inflight`` and clears ``_no_inflight`` right after, at the
    reference's own points.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_patch_window_enter``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return inflight + 1


def _patch_window_exit(inflight: int) -> tuple[int, bool]:
    """The in-flight release step behind ``_call``'s finally arm.

    The exit half of the patch-window bookkeeping: a call leaving the
    gate counts itself out — ``inflight - 1`` — and the drained
    verdict names whether the counter reached zero, the moment the
    caller sets ``_no_inflight`` so a patch window parked on it can
    open.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_patch_window_exit``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    left = inflight - 1
    return (left, left == 0)


def mutation_failure_suffix(kind: MutationResult) -> str:
    """Return the user-facing suffix text for a non-success MutationResult.

    Empty string for ``"success"``. Cascade sites compose this with their
    own action prefix and a sensible separator.
    """
    return _MUTATION_TOAST_SUFFIX.get(kind, "")


def _watch_playlist_kwargs(
    video_id: Any,
    playlist_id: Any,
    limit: Any,
    radio: Any,
) -> dict[str, Any]:
    """The kwargs dict ``get_watch_playlist`` forwards to the client.

    The pure core of the wrapper's call assembly: the conditional
    members — ``limit`` always, ``videoId`` / ``playlistId`` only when
    the caller passed them (an ``is None`` identity gate, so a falsy
    but non-None id still travels) and ``radio`` only when truthy (the
    literal ``True`` travels, never the flag object) — in the
    reference's insertion order, every value passed through as the
    object it is. The both-None early return stays in the wrapper, as
    does the ``**`` unpacking at the call.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_watch_playlist_kwargs``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    kwargs: dict[str, Any] = {"limit": limit}
    if video_id is not None:
        kwargs["videoId"] = video_id
    if playlist_id is not None:
        kwargs["playlistId"] = playlist_id
    if radio:
        kwargs["radio"] = True
    return kwargs


def _result_tracks(result: Any) -> list[Any]:
    """The ``tracks`` member of a watch / liked-songs payload.

    The pure core of the tracks extraction ``get_liked_songs``,
    ``get_watch_playlist``, ``get_radio``'s per-seed fetch,
    ``get_playlist_radio`` and the discovery trending/liked mixes
    share: the member read through the mapping's own ``get`` with a
    fresh empty list as the default, over the subclass-aware
    ``isinstance(result, dict)`` gate — a non-dict answers a fresh
    empty list (never the input itself), a dict passes its ``tracks``
    object through as it is.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_result_tracks``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return result.get("tracks", []) if isinstance(result, dict) else []


def _playlist_tracks(playlist: Any) -> list[Any]:
    """The ``tracks`` member of a playlist payload, list-gated.

    The pure core of ``get_chart_shelf_tracks``'s extraction — the
    stricter sibling of ``_result_tracks``: over the
    ``isinstance(playlist, dict)`` gate the member is read through the
    mapping's own ``get`` with a fresh empty list default, then the
    reference's ``or []`` (a falsy member — ``None`` where the payload
    carries the key's null shape — answers a fresh empty list), then
    the ``isinstance(raw, list)`` gate: only a list passes through,
    as the object it is; every other member shape — a dict, a string,
    a generator — answers a fresh empty list.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_playlist_tracks``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if isinstance(playlist, dict):
        raw = playlist.get("tracks", []) or []
        return raw if isinstance(raw, list) else []
    return []


def _radio_seed_pool(results: Any) -> list[Any]:
    """The deduplicated pool of radio tracks across all seeds' fetches.

    The pure core of ``get_radio``'s mix assembly: the per-seed result
    lists (``asyncio.gather``'s output, walked through the iteration
    protocol) each walked through the iteration protocol in turn, and
    each track read its id through the mapping's own ``get`` —
    ``videoId`` first, ``video_id`` with the fresh ``""`` default only
    when the first read answers falsy (the reference's ``or`` — a
    missing or falsy first key falls to the second, and only then is
    the second call made) — and the track kept when that id is truthy
    and not yet seen: first occurrence wins, the set's own
    hash-then-identity-or-``==`` membership and add deciding, an
    unhashable id raising the containment protocol's own
    ``TypeError``. Kept tracks pass through as the objects they are,
    in first-sight order; the shuffle and the limit trim run in the
    mix engine below, and the normalisation stays in the wrapper.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_radio_seed_pool``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    pool: list[Any] = []
    seen_ids: set[Any] = set()
    for tracks in results:
        for track in tracks:
            vid = track.get("videoId") or track.get("video_id", "")
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                pool.append(track)
    return pool


def _radio_mix(pool: Any, seed_count: Any, limit: Any) -> list[Any]:
    """The shuffled, trimmed mix ``get_radio`` returns.

    The pure core of ``get_radio``'s result assembly: the multi-seed
    shuffle gate — ``seed_count > 1``, the comparison through
    CPython's own ``>`` (single-seed callers expect the seed track to
    lead, so their pool keeps its first-sight order) — under which
    the pool is shuffled in place through the ``random`` module's own
    ``shuffle``, and the ``pool[:limit]`` trim — the slice through
    CPython's own protocol — that caps the mix at ``limit`` tracks.
    The RNG state is the ``random`` module's own, so a
    ``random.seed(...)`` before the call steers both backends
    identically; the normalisation stays in the wrapper.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_radio_mix``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    import random

    if seed_count > 1:
        random.shuffle(pool)
    return pool[:limit]


def _discovery_source_order(last_source: Any) -> list[Any]:
    """The six discovery sources' rotation order.

    The pure core of ``get_discovery_mix``'s source cycling: the
    ``(last % 6) + 1`` start (0 → 1 first, 5 → 6 first, 6 wraps to 1)
    and the six-entry rotation that follows it — every operator
    through CPython's own, so a non-int last source steers both
    backends identically.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_discovery_source_order``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    total = 6
    start = (last_source % total) + 1
    return [(start + i - 1) % total + 1 for i in range(total)]


def _chart_shelves(charts: Any) -> list[Any]:
    """The chart shelves of a ``get_charts`` payload, key order kept.

    The pure core of the discovery charts mix's shelf assembly: the
    ``daily``, ``weekly`` and ``videos`` members read through the
    mapping's own ``get`` — in that order, a missing member skipped —
    each gated by the subclass-aware ``isinstance(arr, list)``, and
    each survivor's items filtered by the ``isinstance(s, dict)`` gate
    into the flat shelf list. The dict shelves pass through as the
    objects they are; the payload is never mutated.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_chart_shelves``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    shelves: list[Any] = []
    for key in ("daily", "weekly", "videos"):
        arr = charts.get(key)
        if isinstance(arr, list):
            shelves.extend(s for s in arr if isinstance(s, dict))
    return shelves


def _chart_shelf_index(last_index: Any, shelf_count: Any) -> Any:
    """The next chart shelf's position in the rotation.

    The pure core of the discovery charts mix's shelf pick: the
    ``(last + 1) % count`` round-robin step — every operator through
    CPython's own, so a zero count raises the modulo's own
    ``ZeroDivisionError`` exactly like the reference would (the mix
    guards with ``if not shelves`` before it ever reaches the pick).

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_chart_shelf_index``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return (last_index + 1) % shelf_count


def _discovery_playable_tracks(items: Any) -> list[Any]:
    """The playable track dicts of a discovery payload.

    The pure core of the playable filter the trending, home, liked,
    artist and history mixes run over their payloads: only dict items
    (the subclass-aware ``isinstance`` gate) whose ``videoId`` member —
    read through the mapping's own ``get`` — is truthy pass through,
    as the objects they are, in the payload's order.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_discovery_playable_tracks``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return [t for t in items if isinstance(t, dict) and t.get("videoId")]


def _chart_playable_tracks(tracks: Any) -> list[Any]:
    """The playable track dicts of a chart shelf's track list.

    The charts mix's own playable filter — the two-key spelling the
    chart shelves' mixed payload shapes need: only dict items (the
    subclass-aware ``isinstance`` gate) pass whose id read —
    ``videoId`` first through the mapping's own ``get``, ``video_id``
    with the fresh ``""`` default only when the first read answers
    falsy (the reference's ``or``) — is truthy. Kept tracks pass
    through as the objects they are, in the list's order.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_chart_playable_tracks``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return [t for t in tracks if isinstance(t, dict) and (t.get("videoId") or t.get("video_id"))]


def _sample_seeds(pool: Any) -> list[Any]:
    """The three-track seed sample a discovery mix draws.

    The pure core of the sampling step every discovery mix source
    shares (the charts, trending, home, liked-songs, artist and
    history mixes): ``random.sample(pool, min(3, len(pool)))`` — the
    length through CPython's own sequence protocol, the ``min``
    capping the draw at three and at the pool's own length (so a
    short pool answers every track it has), and the sample itself
    through the ``random`` module's own ``sample``. The RNG state is
    the ``random`` module's own, so a ``random.seed(...)`` before the
    call steers both backends identically; the tracks pass through as
    the objects they are.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_sample_seeds``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    import random

    return random.sample(pool, min(3, len(pool)))


def _pick_artist(artists: Any) -> Any:
    """The random artist pick behind the artist mix's seed selection.

    The pure core of the artist discovery mix's first draw:
    ``random.choice(artists)`` — through the ``random`` module's own
    ``choice``, so a ``random.seed(...)`` before the call steers both
    backends identically — the artist object passing through as it
    is; the empty-artists guard stays in the wrapper (the reference's
    ``random.choice`` would raise its own ``IndexError`` on an empty
    sequence, which the guard keeps unreachable).

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_pick_artist``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    import random

    return random.choice(artists)


def _explore_new_releases(result: Any) -> list[Any]:
    """The new-releases member of an explore payload.

    The pure core of ``get_new_releases``: over the subclass-aware
    ``isinstance(result, dict)`` gate the ``new_releases`` member is
    read through the mapping's own ``get`` with a fresh empty list
    default and passed through — as the list object it is — only when
    the subclass-aware ``isinstance(releases, list)`` gate holds;
    every other shape answers a fresh empty list.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_explore_new_releases``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if isinstance(result, dict):
        releases = result.get("new_releases", [])
        if isinstance(releases, list):
            return releases
    return []


def _playlist_order_params(order: Any) -> Any:
    """The sort-params token for a ``get_playlist`` order name.

    The pure core of ``get_playlist``'s order resolution: the
    ``order or ""`` truthiness gate (``None`` and every other falsy
    spelling read as the default order) and the ``_ORDER_PARAMS``
    table lookup — read from the service class at call time, so a
    patched table steers both backends identically — answering the
    a_to_z / z_to_a / recently_added browse ``params`` tokens, or
    ``None`` for the playlist's server-side default.

    Implemented by the Rust extension
    (``_ytm_player_native.ytm_playlist_order_params``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return YTMusicService._ORDER_PARAMS.get(order or "")


class YTMusicService:
    """Async wrapper around ytmusicapi.YTMusic.

    All public methods are async and delegate to ytmusicapi's synchronous API
    through ``asyncio.to_thread`` so they never block the event loop.
    """

    def __init__(
        self,
        auth_path: Path = AUTH_FILE,
        auth_manager: AuthManager | None = None,
        user: str | None = None,
        on_auth_refresh: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._auth_path = auth_path
        self._auth_manager = auth_manager
        self._user = user or None  # normalise "" → None
        self._on_auth_refresh = on_auth_refresh
        self._ytm: YTMusic | None = None
        self._consecutive_api_failures: int = 0
        # Guards lazy init of self._ytm against concurrent first-access from
        # asyncio.to_thread workers.
        self._client_init_lock = threading.Lock()
        self._auth_refresh_lock = asyncio.Lock()
        # time.monotonic() before which no renewal is attempted (set by a
        # failed attempt, cleared by a successful one). Read and written
        # under _auth_refresh_lock; deliberately not tied to the client
        # object, which the consecutive-failure reinit in _run replaces.
        self._renewal_retry_after: float = 0.0
        # Serializes get_playlist(order=...) monkey-patches so concurrent
        # calls don't stack patches on client._send_request.
        self._order_lock = asyncio.Lock()
        # Keeps ALL other client calls out of a patch window: while
        # client._send_request is patched, any concurrent "browse" request
        # would get the sort params injected into its body. Normal calls
        # wait on this before running; it is cleared during the window.
        self._no_patch = asyncio.Event()
        self._no_patch.set()
        # In-flight normal calls; a patch window waits for them to drain
        # so a call that grabbed the un-patched send can't race the swap.
        self._inflight = 0
        self._no_inflight = asyncio.Event()
        self._no_inflight.set()
        self._last_discovery_source: int = 0
        self._last_chart_shelf: int = 0
        # videoId -> server-assigned setVideoId from the most recent successful
        # add_playlist_items() call. Lets callers stamp freshly-added rows with
        # the setVideoId they need for later removal, without a reload.
        self.last_added_set_video_ids: dict[str, str] = {}
        # Channel ID each client this service built was bound to (see
        # AuthManager.create_bound_client). An expired call is only replayed
        # on a replacement client carrying the SAME identity; None (no
        # account record) means no automatic renewal for that client.
        self._client_identities: weakref.WeakKeyDictionary[Any, str | None] = (
            weakref.WeakKeyDictionary()
        )

    @property
    def client(self) -> YTMusic:
        """Lazily initialise and return the underlying YTMusic client.

        Thread-safe under concurrent first-access via asyncio.to_thread.
        """
        if self._ytm is None:
            with self._client_init_lock:
                # Double-check: another thread may have initialised it
                # between our None check and acquiring the lock.
                if self._ytm is None:
                    if self._auth_manager is not None:
                        client, channel_id = self._auth_manager.create_bound_client(user=self._user)
                        self._client_identities[client] = channel_id
                        self._ytm = client
                    else:
                        self._ytm = YTMusic(str(self._auth_path), user=self._user)
        return self._ytm

    async def _call(self, func: Any, *args: Any, timeout: int | None = None, **kwargs: Any) -> Any:
        """Run a sync ytmusicapi method in a thread with timeout.

        Waits out any active get_playlist(order=...) patch window first —
        running while client._send_request is patched would inject sort
        params into an unrelated browse request.
        """
        await self._no_patch.wait()
        # The in-flight admission and release bookkeeping is resolved
        # by the _patch_window_enter/_patch_window_exit engines (native
        # when the Rust extension is built, the pure-Python
        # transcriptions otherwise); the counter assignments and the
        # _no_inflight clear/set stay here, at the reference's own
        # points.
        self._inflight = _patch_window_enter(self._inflight)
        self._no_inflight.clear()
        failed_client = getattr(func, "__self__", None)
        try:
            try:
                return await self._run(func, *args, timeout=timeout, **kwargs)
            except (YTMusicServerError, KeyError) as exc:
                if self._auth_manager is None or failed_client is None or not _is_auth_expired(exc):
                    raise
                method_name = getattr(func, "__name__", "")
                client = await self._renewed_client_for(failed_client, exc, method_name)
                retry_func = getattr(client, method_name, None)
                if not callable(retry_func):
                    raise RuntimeError(f"Cannot retry YTMusic method {method_name!r}")
                return await self._run(retry_func, *args, timeout=timeout, **kwargs)
        finally:
            # Known limitation: on timeout/cancellation the to_thread worker
            # may still be running when we decrement, so a later patch window
            # could inject sort params into that zombie's browse body — but
            # its response is already discarded (the awaiter is gone), so no
            # consumed data can be corrupted. Tracking worker lifetime instead
            # would stall ordered fetches behind hung sockets; not worth it.
            self._inflight, drained = _patch_window_exit(self._inflight)
            if drained:
                self._no_inflight.set()

    async def _renewed_client_for(
        self, failed_client: Any, error: BaseException, method_name: str = ""
    ) -> Any:
        """Renew the expired session of *failed_client*; return the client to retry on.

        The retry is bound to the account of the client the call failed on.
        A replacement client for any other account — `ytm setup` switching
        accounts while the call was pending, or between the renewal check
        and the rebuild below — must not have the call replayed on it.
        Raises *error* when renewal is refused, fails, or a recent attempt
        failed and the cooldown has not elapsed.
        """
        auth_manager = self._auth_manager
        if auth_manager is None:
            raise error
        identity = self._client_identities.get(failed_client)
        if identity is None:
            logger.warning(
                "YouTube Music session expired and this client has no recorded account "
                "identity; not renewing automatically. Run `ytm setup`."
            )
            raise error
        async with self._auth_refresh_lock:
            if self._ytm is failed_client or self._ytm is None:
                # The cooldown gate and the failed-renewal stamp are
                # resolved by the _renewal_cooldown_gate / stamp
                # engines (native when the Rust extension is built, the
                # pure-Python transcriptions otherwise); the lock, the
                # renewal itself and the raise stay here.
                blocked, remaining = _renewal_cooldown_gate(self._renewal_retry_after)
                if blocked:
                    logger.warning(
                        "YouTube Music session expired; the last automatic renewal failed, "
                        "not trying again for another %.0fs",
                        remaining,
                    )
                    raise error
                logger.info("YouTube Music session expired; refreshing browser credentials")
                if self._on_auth_refresh is not None:
                    try:
                        await self._on_auth_refresh()
                    except Exception:
                        logger.exception("Authentication refresh notification failed")
                if not await asyncio.to_thread(auth_manager.try_auto_refresh, identity):
                    self._renewal_retry_after = _renewal_cooldown_stamp()
                    logger.warning("Automatic YouTube Music session refresh failed")
                    raise error
                self._renewal_retry_after = 0.0
                with self._client_init_lock:
                    self._ytm = None
                logger.info("YouTube Music session refreshed")

            client = self.client
            if self._client_identities.get(client) != identity:
                logger.warning(
                    "The renewed YouTube Music session belongs to a different account; "
                    "not retrying the pending %s",
                    method_name or "call",
                )
                raise error
            return client

    async def _run(self, func: Any, *args: Any, timeout: int | None = None, **kwargs: Any) -> Any:
        """``_call`` without the patch-window gate (the ordered call itself uses this)."""
        effective_timeout = timeout if timeout is not None else get_settings().playback.api_timeout
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=effective_timeout,
            )
            self._consecutive_api_failures = 0
            return result
        except _EXPECTED_API_EXCEPTIONS:
            logger.exception(
                "ytmusicapi call failed (func=%s, consecutive_failures=%d)",
                getattr(func, "__name__", str(func)),
                self._consecutive_api_failures + 1,
            )
            # The bump and the threshold verdict are resolved by the
            # _api_failure_step engine (native when the Rust extension
            # is built, the pure-Python transcription otherwise); the
            # assignment, the reinit branch and the post-reinit reset
            # stay here, at the reference's own points.
            bumped, reinit = _api_failure_step(self._consecutive_api_failures)
            self._consecutive_api_failures = bumped
            if reinit:
                logger.warning(
                    "Re-initializing YTMusic client after %d consecutive API failures",
                    bumped,
                )
                # Hold the lock while clearing _ytm so a concurrent .client
                # access doesn't race with the reinit signal.
                with self._client_init_lock:
                    self._ytm = None
                self._consecutive_api_failures = 0
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        filter: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search YouTube Music.

        Args:
            query: Search query string.
            filter: One of ``"songs"``, ``"videos"``, ``"albums"``,
                ``"artists"``, ``"playlists"``, or ``None`` for all.
            limit: Maximum number of results.

        Returns:
            List of result dicts.
        """
        try:
            return await self._call(self.client.search, query, filter=filter, limit=limit)
        except asyncio.TimeoutError:
            logger.warning("Search timed out for query=%r", query)
            return []
        except Exception:
            logger.exception("Search failed for query=%r filter=%r", query, filter)
            return []

    async def get_search_suggestions(self, query: str) -> list[str]:
        """Return autocomplete suggestions for *query*."""
        try:
            return await self._call(self.client.get_search_suggestions, query)
        except Exception:
            logger.exception("get_search_suggestions failed for query=%r", query)
            return []

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    async def get_library_playlists(self, limit: int | None = 25) -> list[dict[str, Any]]:
        """Return the user's library playlists. ``limit=None`` fetches all of them."""
        try:
            return await self._call(self.client.get_library_playlists, limit=limit)
        except Exception:
            logger.exception("get_library_playlists failed")
            return []

    async def get_library_albums(self, limit: int = 25) -> list[dict[str, Any]]:
        """Return the user's saved albums."""
        try:
            return await self._call(self.client.get_library_albums, limit=limit)
        except Exception:
            logger.exception("get_library_albums failed")
            return []

    async def get_library_artists(self, limit: int | None = 25) -> list[dict[str, Any]] | None:
        """Return the user's subscribed/followed artists.

        Returns ``None`` on failure (auth expired, network, server error);
        an empty list means the account has no subscriptions.
        """
        try:
            return await self._call(self.client.get_library_subscriptions, limit=limit)
        except Exception:
            logger.exception("get_library_artists failed")
            return None

    async def get_liked_songs(
        self, limit: int | None = None, timeout: int | None = None
    ) -> list[dict[str, Any]] | None:
        """Return tracks from the user's Liked Music playlist.

        Returns ``None`` on failure (auth expired, network, server error) so
        callers can distinguish a fetch failure from a genuinely empty
        playlist — mirrors ``get_history``.
        """
        try:
            playlist = await self._call(self.client.get_liked_songs, timeout=timeout, limit=limit)
            return _result_tracks(playlist)
        except Exception:
            logger.exception("get_liked_songs failed")
            return None

    # ------------------------------------------------------------------
    # Browsing
    # ------------------------------------------------------------------

    async def get_home(self, limit: int = 3) -> list[dict[str, Any]] | None:
        """Return personalised home page recommendations.

        Returns ``None`` on failure (auth expired, network, server error) so
        callers can distinguish a fetch failure from an account with no
        recommendations — mirrors ``get_history``.
        """
        try:
            return await self._call(self.client.get_home, limit=limit)
        except Exception:
            logger.exception("get_home failed")
            return None

    async def get_charts(self, country: str = "ZZ") -> dict[str, Any] | None:
        """Return chart data for *country* (ISO 3166-1 alpha-2, e.g. ``"GB"``).

        ``"ZZ"`` is YouTube's catch-all "Global" region and is the default
        here, matching the ``settings.ui.region`` default; production code
        passes the configured region explicitly. Locale-style codes
        (``ES-ES``, ``en-GB``) are normalised to bare two-letter codes —
        YouTube's endpoint silently falls back to Global on locale-shaped
        input.

        Returns ``None`` on failure so callers can distinguish a fetch
        failure from a region with no chart data — mirrors ``get_history``.
        """
        from ytm_player.services.regions import normalise_region

        country = normalise_region(country)
        try:
            return await self._call(self.client.get_charts, country=country)
        except Exception:
            logger.exception("get_charts failed for country=%r", country)
            return None

    async def get_new_releases(self) -> list[dict[str, Any]] | None:
        """Return new album releases.

        ytmusicapi has no dedicated ``get_new_releases`` endpoint; the
        explore page bundles the data under ``new_releases``. Returns
        ``None`` on failure (auth expired, network, server error); an empty
        list means the explore page listed no releases.
        """
        try:
            result = await self._call(self.client.get_explore)
            return _explore_new_releases(result)
        except Exception:
            logger.exception("get_new_releases failed")
            return None

    # ------------------------------------------------------------------
    # Content details
    # ------------------------------------------------------------------

    async def get_album(self, album_id: str) -> dict[str, Any]:
        """Return full album details including track listing."""
        try:
            return await self._call(self.client.get_album, album_id)
        except Exception:
            logger.exception("get_album failed for %r", album_id)
            return {}

    async def get_artist(self, artist_id: str) -> dict[str, Any]:
        """Return artist page data (top songs, albums, related, etc.)."""
        try:
            return await self._call(self.client.get_artist, artist_id)
        except Exception:
            logger.exception("get_artist failed for %r", artist_id)
            return {}

    async def get_artist_albums(self, channel_id: str, params: str) -> list[dict[str, Any]]:
        """Return an artist's complete albums or singles list.

        ``get_artist`` only carries the first page of each section; the
        section's ``browseId`` + ``params`` address its "see all" page.
        """
        try:
            result = await self._call(self.client.get_artist_albums, channel_id, params, limit=None)
            return result if isinstance(result, list) else []
        except Exception:
            logger.exception("get_artist_albums failed for %r", channel_id)
            return []

    _ORDER_PARAMS = {
        "a_to_z": "ggMGKgQIARAA",
        "z_to_a": "ggMGKgQIARAB",
        "recently_added": "ggMGKgQIABAB",
    }

    # Timeout (seconds) for background fetches of large playlists.
    _LARGE_PLAYLIST_TIMEOUT = 120

    async def get_playlist(
        self,
        playlist_id: str,
        limit: int | None = None,
        order: str | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Return playlist metadata and tracks.

        Args:
            playlist_id: Playlist ID.
            limit: Max tracks to return.  ``None`` retrieves all.
            order: Sort order — ``"a_to_z"``, ``"z_to_a"``, or
                ``"recently_added"``.  ``None`` uses the playlist's
                server-side default.
            timeout: Override the default API timeout (seconds).
        """
        try:
            params = _playlist_order_params(order)
            if params:
                client = self.client
                try:
                    return await self._get_playlist_ordered(
                        client, playlist_id, params, limit=limit, timeout=timeout
                    )
                except (YTMusicServerError, KeyError) as exc:
                    if self._auth_manager is None or not _is_auth_expired(exc):
                        raise
                    # The window above is closed again by now: renewal runs
                    # with client._send_request restored and the gate open,
                    # then the sorted fetch is retried once in a fresh window
                    # on the verified replacement client.
                    renewed = await self._renewed_client_for(client, exc, "get_playlist")
                    return await self._get_playlist_ordered(
                        renewed, playlist_id, params, limit=limit, timeout=timeout
                    )
            return await self._call(
                self.client.get_playlist, playlist_id, timeout=timeout, limit=limit
            )
        except Exception:
            logger.exception("get_playlist failed for %r", playlist_id)
            return {}

    async def _get_playlist_ordered(
        self,
        client: Any,
        playlist_id: str,
        params: str,
        *,
        limit: int | None,
        timeout: int | None,
    ) -> dict[str, Any]:
        """One ``get_playlist`` on *client* with sort *params* injected into its browse request.

        Temporarily patches ``client._send_request`` inside a patch window:
        ``_order_lock`` serializes windows so two ordered calls don't stack
        patches, and the gate keeps every other client call out — a
        concurrent "browse" request would otherwise get the sort params
        too. The patch is restored and the gate reopened on any exit,
        including cancellation.
        """
        async with self._order_lock:
            # Open the window: stop new normal calls, then wait for in-flight
            # ones to drain before touching the shared client._send_request.
            self._no_patch.clear()
            try:
                await self._no_inflight.wait()
                original_send = client._send_request

                def _patched_send(endpoint: str, body: dict, *a: Any, **kw: Any) -> Any:
                    if endpoint == "browse" and isinstance(body, dict):
                        body["params"] = params
                    return original_send(endpoint, body, *a, **kw)

                client._send_request = _patched_send
                try:
                    # _run, not _call: the gate is closed for the window.
                    return await self._run(
                        client.get_playlist, playlist_id, timeout=timeout, limit=limit
                    )
                finally:
                    client._send_request = original_send
            finally:
                self._no_patch.set()

    async def get_playlist_remaining(
        self, playlist_id: str, already_have: int, order: str | None = None
    ) -> list[dict[str, Any]]:
        """Fetch all tracks for a playlist and return only those beyond *already_have*.

        Uses an extended timeout since large playlists (1500+) need 30-60s.
        """
        data = await self.get_playlist(
            playlist_id, limit=None, order=order, timeout=self._LARGE_PLAYLIST_TIMEOUT
        )
        all_tracks = data.get("tracks", [])
        return all_tracks[already_have:]

    async def get_song(self, video_id: str) -> dict[str, Any]:
        """Return detailed info for a single song/video."""
        try:
            return await self._call(self.client.get_song, video_id)
        except Exception:
            logger.exception("get_song failed for %r", video_id)
            return {}

    async def get_lyrics(self, video_id: str) -> dict[str, Any] | None:
        """Return lyrics for a song, or None if unavailable.

        Requires the *browseId* for the lyrics tab. We first fetch the watch
        playlist to obtain it, then request the actual lyrics.
        Tries timestamped (mobile) first, falls back to plain text.
        """
        try:
            watch = await self._call(self.client.get_watch_playlist, video_id)
            lyrics_browse_id = watch.get("lyrics")
            if not lyrics_browse_id:
                return None
            # Try timed lyrics first (uses mobile client internally)
            try:
                result = await self._call(self.client.get_lyrics, lyrics_browse_id, timestamps=True)
                if result is not None:
                    return result
            except Exception:
                logger.debug("Timed lyrics request failed for %r, trying plain", video_id)
            # Fall back to plain lyrics
            return await self._call(self.client.get_lyrics, lyrics_browse_id)
        except Exception:
            logger.exception("get_lyrics failed for %r", video_id)
            return None

    # ------------------------------------------------------------------
    # Playback related
    # ------------------------------------------------------------------

    async def get_watch_playlist(
        self,
        video_id: str | None = None,
        playlist_id: str | None = None,
        limit: int = 25,
        *,
        radio: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the "Up Next" queue for a song or the tracks of a playlist.

        Two call shapes are supported:
        - ``video_id`` + optional ``playlist_id``: original "what plays next
          after this video" use, optionally seeded with playlist context.
        - ``playlist_id`` only: fetch a playlist's tracks via the watch
          endpoint. Used as a fallback for OLAK5-prefixed auto-generated
          playlists (e.g. YouTube Music's Trending charts) where
          ``get_playlist`` hits an upstream parser bug in ytmusicapi
          (``parse_audio_playlist`` dereferences ``tracks[0]["album"]``
          which is None, raising TypeError).

        When *radio* is True, the endpoint returns radio-shuffled results
        instead of a fixed-order queue (matches ``get_radio``'s usage).

        Returns:
            List of track dicts.
        """
        if video_id is None and playlist_id is None:
            return []
        try:
            kwargs = _watch_playlist_kwargs(video_id, playlist_id, limit, radio)
            result = await self._call(self.client.get_watch_playlist, **kwargs)
            return _result_tracks(result)
        except Exception:
            logger.exception(
                "get_watch_playlist failed for video=%r playlist=%r",
                video_id,
                playlist_id,
            )
            return []

    async def get_chart_shelf_tracks(
        self, playlist_id: str, limit: int = 25
    ) -> list[dict[str, Any]]:
        """Resolve a chart shelf playlist into tracks.

        OLAK5-prefixed playlists use ``get_watch_playlist`` because
        ytmusicapi's ``parse_audio_playlist`` dereferences
        ``tracks[0]['album']`` which is None for these auto-generated
        playlists, raising TypeError.
        """
        if playlist_id.startswith("OLAK5"):
            return await self.get_watch_playlist(playlist_id=playlist_id, limit=limit)
        playlist = await self.get_playlist(playlist_id, limit=limit)
        return _playlist_tracks(playlist)

    async def get_radio(self, video_ids: str | list[str], limit: int = 25) -> list[dict]:
        """Fetch radio tracks from one or more seeds and return a deduplicated mix.

        Fetches all seeds in parallel. Individual seed failures are
        swallowed so a single bad ID doesn't abort the batch.
        Returns the pool normalized, shuffled, and trimmed to *limit*.
        *limit* is also forwarded to each seed's ``get_watch_playlist``
        call; ``None`` is not supported there (ytmusicapi subtracts from it).
        """
        from ytm_player.utils.formatting import normalize_tracks

        if isinstance(video_ids, str):
            video_ids = [video_ids]
        if not video_ids:
            return []

        async def _fetch_one(video_id: str) -> list[dict]:
            try:
                result = await self._call(
                    self.client.get_watch_playlist, videoId=video_id, radio=True, limit=limit
                )
                return _result_tracks(result)
            except Exception:
                logger.debug("get_radio: seed %r failed, skipping", video_id)
                return []

        results = await asyncio.gather(*[_fetch_one(vid) for vid in video_ids])

        pool = _radio_seed_pool(results)

        # Single-seed callers (e.g., "Start Radio on this track") expect the seed
        # track to lead. Only shuffle when there are multiple seeds (refill,
        # discovery) — the mix engine's own gate, the RNG flowing through the
        # random module so seeded sequences steer both backends identically.
        mix = _radio_mix(pool, len(video_ids), limit)
        return normalize_tracks(mix)

    async def get_discovery_mix(self) -> tuple[list[dict], str]:
        """Select seed tracks from one of six sources for discovery playback.

        Round-robins through sources so each press of D cycles to the
        next source.  On failure, skips to the next in sequence.
        Returns (seed_tracks, label).  Returns ([], "") if all fail.
        """
        sources = _discovery_source_order(self._last_discovery_source)

        async def _charts_mix() -> tuple[list[dict], str]:
            from ytm_player.config.settings import get_settings

            region = get_settings().ui.region
            charts = await self._call(self.client.get_charts, country=region)
            shelves = _chart_shelves(charts)
            if not shelves:
                return [], ""
            idx = _chart_shelf_index(self._last_chart_shelf, len(shelves))
            self._last_chart_shelf = idx
            shelf = shelves[idx]
            playlist_id = shelf.get("playlistId", "")
            from ytm_player.utils.formatting import clean_shelf_title

            shelf_title = clean_shelf_title(shelf.get("title", "Charts"))
            if not playlist_id:
                return [], ""
            tracks = await self.get_chart_shelf_tracks(playlist_id)
            playable = _chart_playable_tracks(tracks)
            if not playable:
                return [], ""
            sampled = _sample_seeds(playable)
            return sampled, f"{region} {shelf_title}"

        async def _trending_mix() -> tuple[list[dict], str]:
            result = await self._call(self.client.get_explore)
            playlist_id = result.get("trending", {}).get("playlist")
            if not playlist_id:
                return [], ""
            wl = await self._call(
                self.client.get_watch_playlist, playlistId=playlist_id, shuffle=True
            )
            items = _result_tracks(wl)
            playable = _discovery_playable_tracks(items)
            if not playable:
                return [], ""
            sampled = _sample_seeds(playable)
            return sampled, "Trending"

        async def _home_mix() -> tuple[list[dict], str]:
            shelves = await self._call(self.client.get_home, limit=3)
            all_playable: list[dict] = []
            for shelf in shelves:
                items = shelf.get("contents", []) if isinstance(shelf, dict) else []
                all_playable.extend(_discovery_playable_tracks(items))
            if not all_playable:
                return [], ""
            sampled = _sample_seeds(all_playable)
            return sampled, "For You"

        async def _liked_songs_mix() -> tuple[list[dict], str]:
            liked = await self._call(self.client.get_liked_songs, limit=50)
            items = _result_tracks(liked)
            playable = _discovery_playable_tracks(items)
            if not playable:
                return [], ""
            sampled = _sample_seeds(playable)
            return sampled, "Your Liked Songs"

        async def _artist_mix() -> tuple[list[dict], str]:
            artists = await self._call(self.client.get_library_artists, limit=25)
            if not artists:
                return [], ""
            artist = _pick_artist(artists)
            channel_id = artist.get("browseId", "")
            if not channel_id:
                return [], ""
            artist_data = await self._call(self.client.get_artist, channel_id)
            songs = artist_data.get("songs", {})
            results = songs.get("results", []) if isinstance(songs, dict) else []
            playable = _discovery_playable_tracks(results)
            if not playable:
                return [], ""
            sampled = _sample_seeds(playable)
            artist_name = artist.get("artist", "Artist")
            return sampled, f"Artist: {artist_name}"

        async def _history_mix() -> tuple[list[dict], str]:
            history = await self._call(self.client.get_history)
            if not history:
                return [], ""
            playable = _discovery_playable_tracks(history)
            if not playable:
                return [], ""
            recent = playable[:20]
            sampled = _sample_seeds(recent)
            return sampled, "Recently Played"

        _source_fns: dict[int, Any] = {
            1: _charts_mix,
            2: _trending_mix,
            3: _home_mix,
            4: _liked_songs_mix,
            5: _artist_mix,
            6: _history_mix,
        }

        for source in sources:
            try:
                seeds, label = await _source_fns[source]()
                if seeds:
                    self._last_discovery_source = source
                    return seeds, label
            except Exception:
                logger.exception("get_discovery_mix: source %d failed", source)

        return [], ""

    async def get_playlist_radio(self, playlist_id: str) -> list[dict]:
        """Start a radio from a playlist via RDAMPL prefix."""
        stripped = playlist_id.removeprefix("VL")
        try:
            result = await self._call(
                self.client.get_watch_playlist,
                playlistId="RDAMPL" + stripped,
                radio=True,
            )
        except Exception:
            logger.exception("Playlist radio failed for %s", playlist_id)
            return []
        tracks = _result_tracks(result)
        if not tracks:
            logger.warning("Playlist radio returned no tracks for %s", playlist_id)
        return tracks

    # ------------------------------------------------------------------
    # Library actions
    # ------------------------------------------------------------------

    async def rate_song(self, video_id: str, rating: str) -> MutationResult:
        """Rate a song.

        Args:
            video_id: The video ID to rate.
            rating: ``"LIKE"``, ``"DISLIKE"``, or ``"INDIFFERENT"`` (remove rating).

        Returns:
            ``"success"`` if the server accepted the rating, otherwise one
            of ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            await self._call(self.client.rate_song, video_id, rating)
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("rate_song failed for %r rating=%r (kind=%s)", video_id, rating, kind)
            return kind

    async def add_playlist_items(
        self, playlist_id: str, video_ids: list[str], duplicates: bool = False
    ) -> MutationResult:
        """Add songs to an existing playlist.

        Args:
            duplicates: When ``True`` the tracks are added even if they already
                exist in the playlist (mirrors the ytmusicapi ``duplicates``
                flag).  When ``False`` (default) the server rejects the request
                and this method returns ``"duplicate"``.

        Returns:
            ``"success"`` if the server accepted the add, ``"duplicate"`` if
            one or more tracks already exist and *duplicates* is ``False``,
            or one of ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"`` on failure.
        """
        self.last_added_set_video_ids = {}
        try:
            result = await self._call(
                self.client.add_playlist_items,
                playlist_id,
                video_ids,
                duplicates=duplicates,
            )
            # ytmusicapi returns a dict with "status": "STATUS_SUCCEEDED" on
            # success.  When a duplicate is detected (and duplicates=False) the
            # server returns HTTP 200 with "status": "STATUS_FAILED" — no
            # exception is raised (genuine API errors raise YTMusicServerError
            # and are handled below).  Any other non-success status is
            # unexpected and classified as a server error so the duplicate
            # confirm-prompt only fires for real duplicates.
            if isinstance(result, dict):
                status = result.get("status", "")
                if "SUCCEEDED" in status:
                    # Capture the videoId -> setVideoId map so callers can make
                    # the just-added rows immediately removable.
                    self.last_added_set_video_ids = {
                        r["videoId"]: r["setVideoId"]
                        for r in result.get("playlistEditResults", [])
                        if isinstance(r, dict) and r.get("videoId") and r.get("setVideoId")
                    }
                    return "success"
                if "FAILED" in status:
                    logger.debug(
                        "add_playlist_items duplicate detected for playlist=%r", playlist_id
                    )
                    return "duplicate"
                logger.warning(
                    "add_playlist_items unexpected status=%r for playlist=%r",
                    status,
                    playlist_id,
                )
                return "server_error"
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception(
                "add_playlist_items failed for playlist=%r (kind=%s)", playlist_id, kind
            )
            return kind

    async def create_playlist(
        self,
        title: str,
        description: str = "",
        privacy: str = "PRIVATE",
    ) -> tuple[MutationResult, str]:
        """Create a new playlist.

        Returns:
            ``("success", playlist_id)`` when the server created the playlist,
            otherwise ``(kind, "")`` where *kind* is one of ``"auth_required"``,
            ``"auth_expired"``, ``"network"``, ``"server_error"``. Unexpected
            exceptions propagate.
        """
        try:
            result = await self._call(
                self.client.create_playlist, title, description, privacy_status=privacy
            )
            if isinstance(result, str) and result:
                return "success", result
            logger.warning("create_playlist returned no id for title=%r: %r", title, result)
            return "server_error", ""
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("create_playlist failed for title=%r (kind=%s)", title, kind)
            return kind, ""

    async def edit_playlist(
        self,
        playlist_id: str,
        title: str | None = None,
        description: str | None = None,
        privacy_status: str | None = None,
    ) -> MutationResult:
        """Edit an existing playlist's metadata.

        Returns:
            ``"success"`` if the server accepted the edit, otherwise one of
            ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        kwargs: dict[str, Any] = {}
        if title is not None:
            kwargs["title"] = title
        if description is not None:
            kwargs["description"] = description
        if privacy_status is not None:
            kwargs["privacyStatus"] = privacy_status
        try:
            result = await self._call(self.client.edit_playlist, playlist_id, **kwargs)
            succeeded = result == "STATUS_SUCCEEDED" if isinstance(result, str) else bool(result)
            if not succeeded:
                logger.warning("edit_playlist returned non-success for %r: %r", playlist_id, result)
                return "server_error"
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("edit_playlist failed for %r (kind=%s)", playlist_id, kind)
            return kind

    async def delete_playlist(self, playlist_id: str) -> MutationResult:
        """Delete a playlist by its ID.

        Returns:
            ``"success"`` if the server accepted the delete, otherwise one of
            ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            result = await self._call(self.client.delete_playlist, playlist_id)
            succeeded = result == "STATUS_SUCCEEDED" if isinstance(result, str) else bool(result)
            if not succeeded:
                logger.warning(
                    "delete_playlist returned non-success for %r: %r", playlist_id, result
                )
                return "server_error"
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("delete_playlist failed for %r (kind=%s)", playlist_id, kind)
            return kind

    async def add_to_library(self, playlist_id: str) -> MutationResult:
        """Add an album or playlist to the user's library via rate_playlist(LIKE).

        Args:
            playlist_id: The album's or playlist's playlistId.

        Returns:
            ``"success"`` if the server accepted the add, otherwise one of
            ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            await self._call(self.client.rate_playlist, playlist_id, "LIKE")
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("add_to_library failed for %r (kind=%s)", playlist_id, kind)
            return kind

    async def remove_album_from_library(self, playlist_id: str) -> MutationResult:
        """Remove an album from the user's library via rate_playlist(INDIFFERENT).

        Args:
            playlist_id: The album's playlistId (browseId often starts with
                ``MPREb_``; the corresponding playlistId starts with ``OLAK5``).

        Returns:
            ``"success"`` if the server accepted the remove, otherwise one of
            ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            await self._call(self.client.rate_playlist, playlist_id, "INDIFFERENT")
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("remove_album_from_library failed for %r (kind=%s)", playlist_id, kind)
            return kind

    async def subscribe_artist(self, channel_id: str) -> MutationResult:
        """Subscribe to an artist (add to library).

        Returns:
            ``"success"`` if the server accepted the subscribe, otherwise
            one of ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            await self._call(self.client.subscribe_artists, [channel_id])
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("subscribe_artist failed for %r (kind=%s)", channel_id, kind)
            return kind

    async def unsubscribe_artist(self, channel_id: str) -> MutationResult:
        """Unsubscribe from an artist (remove from library).

        Returns:
            ``"success"`` if the server accepted the unsubscribe, otherwise
            one of ``"auth_required"``, ``"auth_expired"``, ``"network"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            await self._call(self.client.unsubscribe_artists, [channel_id])
            return "success"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception("unsubscribe_artist failed for %r (kind=%s)", channel_id, kind)
            return kind

    async def remove_playlist_items(
        self, playlist_id: str, videos: list[dict[str, Any]]
    ) -> MutationResult:
        """Remove items from a playlist.

        Args:
            playlist_id: The playlist to modify.
            videos: List of video dicts as returned by ``get_playlist()`` — each
                must contain ``videoId`` and ``setVideoId``.

        Returns:
            ``"success"`` if the server accepted the remove, ``"rejected"``
            if it answered ``STATUS_FAILED`` (the row is gone, or the account
            can't edit the playlist), otherwise one of ``"auth_required"``,
            ``"auth_expired"``, ``"network"``, ``"timeout"``,
            ``"server_error"``. Unexpected exceptions propagate.
        """
        try:
            result = await self._call(self.client.remove_playlist_items, playlist_id, videos)
            # ytmusicapi returns the response's status string, or the whole
            # response when it carries none. An explicit "STATUS_FAILED" is
            # the server refusing the removal (the row is no longer in the
            # playlist, or the account can't edit it) — HTTP 200, no
            # exception. Anything else is a malformed answer.
            if result == "STATUS_SUCCEEDED":
                return "success"
            if result == "STATUS_FAILED":
                logger.warning("remove_playlist_items rejected for %r", playlist_id)
                return "rejected"
            logger.warning(
                "remove_playlist_items returned no usable status for %r: %r", playlist_id, result
            )
            return "server_error"
        except _EXPECTED_MUTATION_EXCEPTIONS as exc:
            kind = _classify_mutation_failure(exc)
            logger.exception(
                "remove_playlist_items failed for playlist=%r (kind=%s)", playlist_id, kind
            )
            return kind

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    async def get_history(self) -> list[dict[str, Any]] | None:
        """Return the user's recently played tracks.

        Returns ``None`` on failure (auth expired, network, server error) so
        callers can distinguish a genuine empty history from an error without
        relying on shared service state. An empty list means the account has
        no history.
        """
        try:
            return await self._call(self.client.get_history)
        except Exception:
            logger.exception("get_history failed")
            return None

    async def add_history_item(self, video_id: str) -> bool:
        """Register a play in the account's YouTube Music history.

        ytmusicapi requires the ``playbackTracking`` payload from
        ``get_song()`` and that both calls use the same client instance;
        both go through ``self.client`` here, so that holds. Best-effort:
        any failure is logged and swallowed (returns ``False``) so callers
        can fire-and-forget without affecting playback.

        Returns ``True`` when the server accepts the report (HTTP 204).
        """
        if not video_id:
            return False
        # ``add_history_item`` needs the ``playbackTracking`` payload, which is
        # only available from ``get_song``. There is no combined endpoint, so
        # one extra round trip per reported play is unavoidable.
        try:
            song = await self._call(self.client.get_song, video_id)
        except Exception:
            logger.exception("get_song failed for %r", video_id)
            return False
        # Unavailable / non-playable tracks come back without playbackTracking;
        # skip explicitly instead of letting a KeyError surface as a generic
        # "add_history_item failed".
        if not isinstance(song, dict) or "playbackTracking" not in song:
            logger.debug("add_history_item: no playbackTracking for %r; skipping", video_id)
            return False
        try:
            resp = await self._call(self.client.add_history_item, song)
        except Exception:
            logger.exception("add_history_item failed for %r", video_id)
            return False
        return getattr(resp, "status_code", None) == 204


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_api_failure_step as _native_api_failure_step,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_auth_expired as _native_auth_expired,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_chart_playable_tracks as _native_chart_playable_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_chart_shelf_index as _native_chart_shelf_index,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_chart_shelves as _native_chart_shelves,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_discovery_playable_tracks as _native_discovery_playable_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_discovery_source_order as _native_discovery_source_order,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_explore_new_releases as _native_explore_new_releases,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_patch_window_enter as _native_patch_window_enter,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_patch_window_exit as _native_patch_window_exit,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_pick_artist as _native_pick_artist,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_playlist_order_params as _native_playlist_order_params,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_playlist_tracks as _native_playlist_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_radio_mix as _native_radio_mix,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_radio_seed_pool as _native_radio_seed_pool,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_renewal_cooldown_gate as _native_renewal_cooldown_gate,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_renewal_cooldown_stamp as _native_renewal_cooldown_stamp,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_result_tracks as _native_result_tracks,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_sample_seeds as _native_sample_seeds,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ytm_watch_playlist_kwargs as _native_watch_playlist_kwargs,
    )
except ImportError:
    pass
else:
    _is_auth_expired = _native_auth_expired  # type: ignore[assignment]
    _renewal_cooldown_gate = _native_renewal_cooldown_gate  # type: ignore[assignment]
    _renewal_cooldown_stamp = _native_renewal_cooldown_stamp  # type: ignore[assignment]
    _api_failure_step = _native_api_failure_step  # type: ignore[assignment]
    _patch_window_enter = _native_patch_window_enter  # type: ignore[assignment]
    _patch_window_exit = _native_patch_window_exit  # type: ignore[assignment]
    _watch_playlist_kwargs = _native_watch_playlist_kwargs  # type: ignore[assignment]
    _result_tracks = _native_result_tracks  # type: ignore[assignment]
    _playlist_tracks = _native_playlist_tracks  # type: ignore[assignment]
    _radio_mix = _native_radio_mix  # type: ignore[assignment]
    _radio_seed_pool = _native_radio_seed_pool  # type: ignore[assignment]
    _discovery_source_order = _native_discovery_source_order  # type: ignore[assignment]
    _chart_shelves = _native_chart_shelves  # type: ignore[assignment]
    _chart_shelf_index = _native_chart_shelf_index  # type: ignore[assignment]
    _discovery_playable_tracks = _native_discovery_playable_tracks  # type: ignore[assignment]
    _chart_playable_tracks = _native_chart_playable_tracks  # type: ignore[assignment]
    _sample_seeds = _native_sample_seeds  # type: ignore[assignment]
    _pick_artist = _native_pick_artist  # type: ignore[assignment]
    _explore_new_releases = _native_explore_new_releases  # type: ignore[assignment]
    _playlist_order_params = _native_playlist_order_params  # type: ignore[assignment]
