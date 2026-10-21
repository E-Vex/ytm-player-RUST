"""Optional lyrics transliteration.

Uses the native Rust implementation (PyO3) when available, falling back
to the ``anyascii`` Python package, and finally to a no-op when neither
is installed.  The native path embeds the same ``any_ascii`` table, so
results are identical while dropping the Python dependency.
"""

from __future__ import annotations

from functools import lru_cache

try:
    from ._ytm_player_native import (  # type: ignore[import-not-found]
        transliterate as _native_transliterate,
    )
except ImportError:
    _native_transliterate = None


def has_non_ascii(text: str) -> bool:
    """Return True if *text* contains any non-ASCII characters."""
    return bool(text) and not text.isascii()


@lru_cache(maxsize=512)
def transliterate_line(text: str) -> str:
    """Transliterate *text* to ASCII via the native module or anyascii.

    Returns the original string unchanged when no transliteration backend
    is available or when the text is already pure ASCII.  The ASCII fast
    path stays in Python so repeated calls return the same string object.
    """
    if not text or not has_non_ascii(text):
        return text
    if _native_transliterate is not None:
        return _native_transliterate(text)
    try:
        from anyascii import anyascii

        return anyascii(text)
    except ImportError:
        return text
