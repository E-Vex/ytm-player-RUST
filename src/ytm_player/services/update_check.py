"""Check PyPI for a newer ytm-player release.

The check runs in a background worker on app startup. Results are
cached for 24 hours in CONFIG_DIR/update_check.json so we don't hit
PyPI more than once a day. Network failures are silent — being offline
is not an error worth surfacing to the user.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)

_PYPI_URL = "https://pypi.org/pypi/ytm-player/json"
_CHECK_INTERVAL_SECONDS = 24 * 60 * 60  # 24h
_REQUEST_TIMEOUT = 5.0


def _is_newer(latest: str, current: str) -> bool:
    """True when *latest* is strictly newer than *current* under PEP 440.

    Only ``InvalidVersion`` is caught: a version whose numeric components
    exceed CPython's int-conversion digit limit raises the plain
    ``ValueError`` the constructor's own ``int()`` raises — not an
    ``InvalidVersion`` — and propagates, exactly like the reference this
    engine is transcribed from.

    The comparison is whatever the installed ``packaging`` says it is:
    through 26.0 its grammar admitted the simple Unicode case folds
    (``1.0+ſ`` parses), 26.1+ scopes them out — so the verdict for such
    spellings follows the installed release. The native engine probes
    the same behaviour once and mirrors it (see
    ``_packaging_admits_unicode_folds``).

    Implemented by the Rust extension
    (``_ytm_player_native.pep440_is_newer``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return False


def _fetch_latest_from_pypi() -> str | None:
    """Hit PyPI for the latest ytm-player version. None on any failure."""
    try:
        resp = requests.get(
            _PYPI_URL,
            headers={"User-Agent": "ytm-player update-check"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("info", {}).get("version")
    except (requests.RequestException, ValueError):
        return None
    except Exception:  # pragma: no cover — belt-and-braces
        logger.debug("Unexpected PyPI fetch failure", exc_info=True)
        return None


def _read_cache(cache_file: Path) -> dict:
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_cache(cache_file: Path, latest: str) -> None:
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"checked_at": time.time(), "latest": latest}
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        logger.debug("Failed to write update-check cache", exc_info=True)


def check_for_update(current_version: str, cache_file: Path) -> str | None:
    """Return the latest PyPI version string IF strictly newer than current.

    Returns None if:
    - The cache says we checked within the last 24h.
    - PyPI is unreachable.
    - The latest version is not newer than *current_version*.
    """
    cache = _read_cache(cache_file)
    try:
        last_checked = float(cache.get("checked_at", 0) or 0)
    except (TypeError, ValueError):
        last_checked = 0.0
    elapsed = time.time() - last_checked
    # Negative elapsed (clock went backwards) → treat as stale and re-fetch.
    if 0 <= elapsed < _CHECK_INTERVAL_SECONDS:
        latest = cache.get("latest")
        if isinstance(latest, str) and latest and _is_newer(latest, current_version):
            return latest
        return None

    latest = _fetch_latest_from_pypi()
    if latest is None:
        return None
    _write_cache(cache_file, latest)
    if _is_newer(latest, current_version):
        return latest
    return None


# ── Native (Rust/PyO3) overrides, if available ─────────────────────────────
# The Rust extension implements the same engines; these bindings override
# the pure-Python transcriptions above when the extension is importable.
# The rebinding must happen AFTER the definitions above — assigning the
# native functions any earlier would be clobbered by the ``def``
# statements (the lesson that produced formatting.py's dead native
# bindings).

# Cache for the installed-packaging fold probe (see
# _packaging_admits_unicode_folds below): None = not probed yet.
_PACKAGING_ADMITS_UNICODE_FOLDS: bool | None = None

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        pep440_is_newer as _native_pep440_is_newer,
    )
except ImportError:
    pass
else:

    def _packaging_admits_unicode_folds() -> bool:
        """True when the installed ``packaging`` PEP 440 grammar admits the
        simple Unicode case folds — the long s ``ſ`` (U+017F), the Kelvin
        sign ``K`` (U+212A), the dotted and dotless ``İ``/``ı``
        (U+0130/U+0131) — wherever a letter can appear.

        ``packaging`` compiled its version grammar with plain
        ``re.IGNORECASE`` through 26.0, so those folds parsed as valid
        versions there (``Version("1.0+ſ")`` works); 26.1 scoped the
        grammar's groups ``(?a:...)`` and the local alphabet ASCII, so
        the same spellings are ``InvalidVersion`` now. The pure-Python
        fallback follows the installed ``packaging`` by construction — it
        calls ``Version`` directly — so the native engine must follow it
        too: this probe (one parse attempt, cached) is the verdict both
        sides agree on, whichever ``packaging`` the uncapped ``>=21.0``
        dependency range admits.
        """
        global _PACKAGING_ADMITS_UNICODE_FOLDS
        if _PACKAGING_ADMITS_UNICODE_FOLDS is None:
            try:
                Version("1.0+\u017f")
            except ValueError:
                _PACKAGING_ADMITS_UNICODE_FOLDS = False
            else:
                _PACKAGING_ADMITS_UNICODE_FOLDS = True
        return _PACKAGING_ADMITS_UNICODE_FOLDS

    def _is_newer(latest: str, current: str) -> bool:  # noqa: F811
        """Strictly-newer verdict (native), on the installed packaging's grammar."""
        return _native_pep440_is_newer(latest, current, _packaging_admits_unicode_folds())
