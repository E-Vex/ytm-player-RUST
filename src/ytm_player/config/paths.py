"""Centralized path definitions for ytm-player.

Single source of truth for all filesystem paths used across the application.
Respects $XDG_CONFIG_HOME and $XDG_CACHE_HOME when set.
On Windows, uses %APPDATA% for config and %LOCALAPPDATA% for cache.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SECURE_FILE_MODE = 0o600
SECURE_DIR_MODE = 0o700


def _compute_paths(platform: str, home: str, env: dict[str, str]) -> dict[str, str]:
    """Compute the app's path table (pure-Python fallback).

    The exact transcription of the platform/env arithmetic behind the
    module's constants; runs natively via
    ``_ytm_player_native.paths_core`` when the Rust extension is built
    (a pure function of ``(platform, home, env)``, so the win32 branch
    is exercisable from any host). An absent entry means the constant is
    ``None`` on that platform — ``SOCKET_PATH`` on Windows,
    ``IPC_PORT_FILE`` elsewhere.
    """
    table: dict[str, str] = {}

    if platform == "win32":
        # get-with-default: the fallback fires only when the variable is
        # *unset* — an empty APPDATA stays empty (and Path("") then
        # lands on the relative table, exactly like the reference).
        app_data = env.get("APPDATA")
        if app_data is None:
            app_data = str(Path(home) / "AppData" / "Roaming")
        local_data = env.get("LOCALAPPDATA")
        if local_data is None:
            local_data = str(Path(home) / "AppData" / "Local")

        config_dir = str(Path(app_data) / "ytm-player")
        table["CONFIG_DIR"] = config_dir
        table["CACHE_DIR"] = str(Path(local_data) / "ytm-player" / "audio")
        table["CACHE_DB"] = str(Path(local_data) / "ytm-player" / "cache.db")
        # Windows uses TCP; the port file lives in the config dir.
        table["IPC_PORT_FILE"] = str(Path(config_dir) / "ipc_port")
    else:
        # Truthiness gates: None *or* the empty string reads as unset.
        xdg_config = env.get("XDG_CONFIG_HOME")
        if xdg_config:
            config_dir = str(Path(xdg_config) / "ytm-player")
        else:
            config_dir = str(Path(home) / ".config" / "ytm-player")
        table["CONFIG_DIR"] = config_dir

        xdg_cache = env.get("XDG_CACHE_HOME")
        if xdg_cache:
            table["CACHE_DIR"] = str(Path(xdg_cache) / "ytm-player" / "audio")
            table["CACHE_DB"] = str(Path(xdg_cache) / "ytm-player" / "cache.db")
        else:
            table["CACHE_DIR"] = str(Path(home) / ".cache" / "ytm-player" / "audio")
            table["CACHE_DB"] = str(Path(home) / ".cache" / "ytm-player" / "cache.db")

        # Unix sockets have a ~108 byte path limit. Prefer
        # XDG_RUNTIME_DIR (short, per-user, tmpfs) when available.
        xdg_runtime = env.get("XDG_RUNTIME_DIR")
        if xdg_runtime:
            table["SOCKET_PATH"] = str(Path(xdg_runtime) / "ytm-player.sock")
        else:
            table["SOCKET_PATH"] = str(Path(config_dir) / "ytm.sock")

    for key, leaf in (
        ("CONFIG_FILE", "config.toml"),
        ("AUTH_FILE", "auth.json"),
        ("ACCOUNT_FILE", "account.json"),
        ("STREAM_COOKIES_FILE", "stream_cookies.txt"),
        ("OAUTH_FILE", "oauth.json"),
        ("OAUTH_CREDS_FILE", "oauth_creds.json"),
        ("SPOTIFY_CREDS_FILE", "spotify.json"),
        ("PID_FILE", "ytm.pid"),
        ("KEYMAP_FILE", "keymap.toml"),
        ("THEME_FILE", "theme.toml"),
        ("RECENT_PLAYLISTS_FILE", "recent_playlists.json"),
        ("SESSION_STATE_FILE", "session.json"),
        ("SHUFFLE_PREFS_FILE", "shuffle_prefs.json"),
        ("UPDATE_CHECK_CACHE", "update_check.json"),
        ("HISTORY_DB", "history.db"),
    ):
        table[key] = str(Path(config_dir) / leaf)

    log_dir = str(Path(config_dir) / "logs")
    table["LOG_DIR"] = log_dir
    table["LOG_FILE"] = str(Path(log_dir) / "ytm.log")
    crash_dir = str(Path(config_dir) / "crashes")
    table["CRASH_DIR"] = crash_dir
    table["CRASH_LOG"] = str(Path(crash_dir) / "ytm-crash.log")
    return table


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same platform-aware path table; the
# module's constants are computed through it when the extension is
# importable, with the pure-Python transcription above as the fallback.
# (Unlike the other tail bindings this one feeds the import-time table
# below, so it must be resolved before — not after — the constants are
# assigned.) The engine is pinned to the fallback by the parity tests.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        paths_core as _native_paths_core,
    )
except ImportError:
    _native_paths_core = None

_TABLE: dict[str, str] = (
    dict(_native_paths_core(sys.platform, str(Path.home()), dict(os.environ)))
    if _native_paths_core is not None
    else _compute_paths(sys.platform, str(Path.home()), dict(os.environ))
)

# --- Platform-aware base directories ---

CONFIG_DIR = Path(_TABLE["CONFIG_DIR"])
CACHE_DIR = Path(_TABLE["CACHE_DIR"])
CACHE_DB = Path(_TABLE["CACHE_DB"])

CONFIG_FILE = Path(_TABLE["CONFIG_FILE"])
AUTH_FILE = Path(_TABLE["AUTH_FILE"])
ACCOUNT_FILE = Path(_TABLE["ACCOUNT_FILE"])  # identity of the auth.json session
STREAM_COOKIES_FILE = Path(_TABLE["STREAM_COOKIES_FILE"])
OAUTH_FILE = Path(_TABLE["OAUTH_FILE"])
OAUTH_CREDS_FILE = Path(_TABLE["OAUTH_CREDS_FILE"])
SPOTIFY_CREDS_FILE = Path(_TABLE["SPOTIFY_CREDS_FILE"])
PID_FILE = Path(_TABLE["PID_FILE"])

# IPC: Unix sockets on Linux/macOS, TCP localhost on Windows (see ipc.py).
# Unix sockets have a ~108 byte path limit. Use XDG_RUNTIME_DIR (short,
# per-user, tmpfs) when available, fall back to CONFIG_DIR.
if sys.platform == "win32":
    # Windows uses TCP; port is stored in this file by the server.
    IPC_PORT_FILE = Path(_TABLE["IPC_PORT_FILE"])
    SOCKET_PATH = None  # Not used on Windows.
else:
    SOCKET_PATH = Path(_TABLE["SOCKET_PATH"])
    IPC_PORT_FILE = None  # Not used on Unix.

KEYMAP_FILE = Path(_TABLE["KEYMAP_FILE"])
THEME_FILE = Path(_TABLE["THEME_FILE"])
RECENT_PLAYLISTS_FILE = Path(_TABLE["RECENT_PLAYLISTS_FILE"])
SESSION_STATE_FILE = Path(_TABLE["SESSION_STATE_FILE"])
SHUFFLE_PREFS_FILE = Path(_TABLE["SHUFFLE_PREFS_FILE"])
UPDATE_CHECK_CACHE = Path(_TABLE["UPDATE_CHECK_CACHE"])
HISTORY_DB = Path(_TABLE["HISTORY_DB"])

# Log files (rotated by RotatingFileHandler).
LOG_DIR = Path(_TABLE["LOG_DIR"])
LOG_FILE = Path(_TABLE["LOG_FILE"])

# Crash tracebacks (kept indefinitely; pruned to last 10 by setup_logging).
CRASH_DIR = Path(_TABLE["CRASH_DIR"])

# CRASH_LOG kept for backwards-compatibility with anything that imports it,
# but new crash files are written to CRASH_DIR/ytm-crash-<timestamp>.log
# via install_excepthooks().  Do not append to this file in new code.
CRASH_LOG = Path(_TABLE["CRASH_LOG"])

_dirs_ensured = False


def secure_chmod(path: Path | str, mode: int) -> None:
    """Set file permissions — no-op on Windows where Unix modes are meaningless."""
    if sys.platform != "win32":
        os.chmod(path, mode)


def ensure_dirs() -> None:
    """Create config and cache directories with secure permissions.

    Called lazily on first invocation (not at import time) so that merely
    importing the module does not create directories on disk — important
    for test isolation.
    """
    global _dirs_ensured
    if _dirs_ensured:
        return
    for _dir in (CONFIG_DIR, CACHE_DIR, LOG_DIR, CRASH_DIR):
        _dir.mkdir(parents=True, exist_ok=True)
        secure_chmod(_dir, SECURE_DIR_MODE)
    _dirs_ensured = True
