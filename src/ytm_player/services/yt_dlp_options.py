"""Helpers for adapting app config to yt-dlp Python API options."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ytm_player.config.settings import YtDlpSettings

# Try to load the native Rust implementation via PyO3.  If the compiled
# extension is unavailable (e.g. during development without a Rust toolchain),
# fall back to the pure-Python implementations defined below.  The extension
# module lives in the utils package next to the other native bindings.
try:
    from ..utils._ytm_player_native import (  # type: ignore[import-not-found]
        normalize_js_runtimes as _native_normalize_js_runtimes,
    )
    from ..utils._ytm_player_native import (
        normalize_remote_components as _native_normalize_remote_components,
    )
except ImportError:
    _native_normalize_js_runtimes = None
    _native_normalize_remote_components = None

logger = logging.getLogger(__name__)


def _split_csv_or_space(value: str) -> list[str]:
    """Split a string by commas/whitespace and drop empties."""
    normalized = value.replace(",", " ")
    return [part for part in normalized.split() if part]


def _normalize_path(value: str | os.PathLike[str] | None) -> str | None:
    """Return an expanded filesystem path, or None when unset/blank."""
    if value is None:
        return None
    if isinstance(value, os.PathLike):
        return str(Path(value).expanduser())
    stripped = value.strip()
    if not stripped:
        return None
    return str(Path(stripped).expanduser())


def normalize_cookiefile(value: str | os.PathLike[str] | None) -> str | None:
    """Return expanded cookie file path for yt-dlp, or None when unset."""
    return _normalize_path(value)


def normalize_remote_components(value: str | list[str] | None) -> list[str] | None:
    """Return yt-dlp compatible remote_components list."""
    if _native_normalize_remote_components is not None:
        return _native_normalize_remote_components(value)
    if value is None:
        return None
    if isinstance(value, str):
        parts = _split_csv_or_space(value)
        return parts or None
    parts = [str(part).strip() for part in value if str(part).strip()]
    return parts or None


def _parse_runtime_token(runtime_spec: str) -> tuple[str, dict] | None:
    """Parse a runtime token in ``runtime[:path]`` form."""
    token = runtime_spec.strip()
    if not token:
        return None
    runtime, sep, path = token.partition(":")
    runtime_name = runtime.lower().strip()
    if not runtime_name:
        return None
    if sep and path.strip():
        return runtime_name, {"path": path.strip()}
    return runtime_name, {}


def normalize_js_runtimes(
    value: str | list[str] | dict[str, dict] | None,
) -> dict[str, dict] | None:
    """Return yt-dlp compatible js_runtimes dict.

    yt-dlp Python API expects: {"runtime": {<config>}}
    """
    if _native_normalize_js_runtimes is not None:
        return _native_normalize_js_runtimes(value)
    if value is None:
        return None

    if isinstance(value, dict):
        result: dict[str, dict] = {}
        for runtime, config in value.items():
            name = str(runtime).strip().lower()
            if not name:
                continue
            result[name] = config if isinstance(config, dict) else {}
        return result or None

    runtime_specs = _split_csv_or_space(value) if isinstance(value, str) else value
    result: dict[str, dict] = {}
    for spec in runtime_specs:
        parsed = _parse_runtime_token(str(spec))
        if parsed is None:
            continue
        runtime_name, config = parsed
        result[runtime_name] = config
    return result or None


def normalize_cafile(value: str | os.PathLike[str] | None) -> str | None:
    """Return expanded CA bundle path, or None when unset."""
    return _normalize_path(value)


def apply_configured_yt_dlp_options(opts: dict, yt_dlp_settings: YtDlpSettings) -> dict:
    """Mutate and return yt-dlp options with app-configured extras."""
    cookies_file = normalize_cookiefile(yt_dlp_settings.cookies_file)
    if cookies_file:
        opts["cookiefile"] = cookies_file

    # When a custom CA bundle is configured (e.g. for Zscaler), tell yt-dlp to
    # skip its bundled certifi CA store and use the system/SSL_CERT_FILE certs
    # instead.  SSL_CERT_FILE is set at app startup in cli.py.
    ca_bundle = normalize_cafile(yt_dlp_settings.ca_bundle)
    if ca_bundle:
        if not Path(ca_bundle).is_file():
            logger.warning("Configured [yt_dlp] ca_bundle does not exist: %s", ca_bundle)
        compat = list(opts.get("compat_opts", []))
        if "no-certifi" not in compat:
            compat.append("no-certifi")
        opts["compat_opts"] = set(compat)

    remote_components = normalize_remote_components(yt_dlp_settings.remote_components)
    if remote_components:
        logger.warning(
            "yt-dlp remote_components is enabled; this allows remote JavaScript component downloads: %s",
            ", ".join(remote_components),
        )
        opts["remote_components"] = remote_components

    js_runtimes = normalize_js_runtimes(yt_dlp_settings.js_runtimes)
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes

    return opts
