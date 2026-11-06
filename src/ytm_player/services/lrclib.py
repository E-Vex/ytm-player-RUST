"""LRCLIB.net fallback for synced lyrics when YouTube Music doesn't provide them."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE_URL = "https://lrclib.net/api/get"
_TIMEOUT = 5


def _query_params(clean_title: Any, artist: Any, duration_seconds: Any) -> dict[str, Any]:
    """The query params behind the LRCLIB lookup request.

    The pure core of ``get_synced_lyrics``' request build: the fresh
    dict with ``track_name`` (the sanitized title) and ``artist_name``
    in the reference's insertion order, then — only when
    ``duration_seconds`` is not ``None``, an identity test so a
    falsy-but-present ``0`` still travels — the ``duration`` key set
    to ``str(int(duration_seconds))``. The title sanitizer is the
    formatting module's (already-ported) engine and stays at the
    caller; the requests fetch stays Python-side.

    Implemented by the Rust extension
    (``_ytm_player_native.lrclib_query_params``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    params: dict[str, Any] = {
        "track_name": clean_title,
        "artist_name": artist,
    }
    if duration_seconds is not None:
        params["duration"] = str(int(duration_seconds))
    return params


def _synced_lyrics(data: Any) -> Any:
    """The synced-lyrics extraction behind the LRCLIB response.

    The pure core of ``get_synced_lyrics``' parse step: the
    ``syncedLyrics`` member read through the mapping's own ``get``,
    then the ``or None`` truthiness gate (an empty string answers
    ``None``). The ``resp.json()`` decode stays Python-side (requests'
    own decoder and its exception shapes).

    Implemented by the Rust extension
    (``_ytm_player_native.lrclib_synced_lyrics``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return data.get("syncedLyrics") or None


async def get_synced_lyrics(
    title: str, artist: str, duration_seconds: float | None = None
) -> str | None:
    """Fetch synced LRC lyrics from LRCLIB.net.

    Returns the LRC-format string if available, or None.
    """
    from ytm_player.utils.formatting import sanitize_title_for_lyric_lookup

    clean_title = sanitize_title_for_lyric_lookup(title, artist)
    params = _query_params(clean_title, artist, duration_seconds)

    def _fetch() -> str | None:
        try:
            resp = requests.get(
                _BASE_URL,
                params=params,
                headers={"User-Agent": "ytm-player/1.0"},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return _synced_lyrics(data)
        except Exception:
            logger.debug("LRCLIB request failed for %r by %r", title, artist, exc_info=True)
            return None

    return await asyncio.to_thread(_fetch)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace the
# pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        lrclib_query_params as _native_query_params,
    )
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        lrclib_synced_lyrics as _native_synced_lyrics,
    )
except ImportError:
    pass
else:
    _query_params = _native_query_params  # type: ignore[assignment]
    _synced_lyrics = _native_synced_lyrics  # type: ignore[assignment]
