"""Tests for the config path-table engine.

``_compute_paths`` (config/paths.py) runs with two interchangeable
backends: the pure-Python transcription in the module and, when the
Rust extension is built, the native ``paths_core`` the module's table
is computed through. The engine: the platform-aware base directories —
win32's ``APPDATA``/``LOCALAPPDATA`` (defaults only when the variable
is *unset*; an empty ``APPDATA`` stays empty, and the empty base then
lands on the relative table) and the XDG ``XDG_CONFIG_HOME`` /
``XDG_CACHE_HOME`` / ``XDG_RUNTIME_DIR`` gates (an empty string reads
as unset, matching the reference's truthiness checks) — plus the
derived files and directories under ``CONFIG_DIR``. ``SOCKET_PATH``
and ``IPC_PORT_FILE`` trade places by platform: an absent table entry
is the ``None`` constant on that platform.

The engine is a pure function of ``(platform, home, env)``, so the
unit tests below assert its contract on synthetic triples — the win32
branch exercisable from any host — on whichever backend is loaded.
Comparisons go through the *simulated platform's* flavour lens
(``PurePosixPath`` / ``PureWindowsPath``, never the host ``Path`` —
the host lens re-spelled ``/`` as ``\\`` on Windows and broke the
posix contract on the Windows CI legs), which is also the level the
module's constants render at once re-wrapped in the native host's
``Path()``.
When the extension is built, the parity tests additionally compare the
native engine against the pure-Python reference case for case at the
same lens. The module-level tests pin the import-time constants via
reload and ``ensure_dirs``' directory creation; ``secure_chmod``'s
no-op on Windows is pinned alongside.
"""

from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterator

import pytest

from ytm_player.config import paths
from ytm_player.config.paths import _compute_paths

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_IS_WIN32 = sys.platform == "win32"

_WIN_APPDATA = "C:\\Users\\u\\AppData\\Roaming"
_WIN_LOCAL = "C:\\Users\\u\\AppData\\Local"

# Synthetic (platform, home, env) triples: the default XDG tree, the
# XDG bases set, the XDG bases set to empty strings (falsy — must read
# as unset), a runtime dir alone, degenerate trailing-slash bases, and
# the win32 set / defaults / empty-APPDATA spellings.
_CASES: list[tuple[str, str, dict[str, str]]] = [
    ("linux", "/home/u", {}),
    ("darwin", "/Users/u", {}),
    (
        "linux",
        "/home/u",
        {
            "XDG_CONFIG_HOME": "/xdg/c",
            "XDG_CACHE_HOME": "/xdg/cache",
            "XDG_RUNTIME_DIR": "/run/user/1000",
        },
    ),
    (
        "linux",
        "/home/u",
        {"XDG_CONFIG_HOME": "", "XDG_CACHE_HOME": "", "XDG_RUNTIME_DIR": ""},
    ),
    ("linux", "/home/u", {"XDG_RUNTIME_DIR": ""}),
    (
        "linux",
        "/home/u",
        {"XDG_CONFIG_HOME": "/xdg/c/", "XDG_CACHE_HOME": "/xdg/cache/"},
    ),
    ("win32", "C:\\Users\\u", {"APPDATA": _WIN_APPDATA, "LOCALAPPDATA": _WIN_LOCAL}),
    ("win32", "C:\\Users\\u", {}),
    ("win32", "C:\\Users\\u", {"APPDATA": "", "LOCALAPPDATA": _WIN_LOCAL}),
]


def _engine(platform: str, home: str, env: dict[str, str]) -> dict[str, str]:
    """The effective backend — the same choice the module's import makes."""
    if _HAVE_NATIVE:
        return dict(_ytm_player_native.paths_core(platform, home, env))
    return _compute_paths(platform, home, env)


def _norm(table: dict[str, str], platform: str) -> dict[str, str]:
    """Render every value through the *simulated platform's* flavour —
    ``PurePosixPath`` for the XDG branches, ``PureWindowsPath`` for the
    win32 branch — never the host ``Path``: the engine is a pure function
    of ``(platform, home, env)`` and must read identically on any host
    (the host lens re-spelled ``/`` as ``\\`` on Windows, which is what
    broke the posix contract on the Windows CI legs). The flavour lens
    also lexically normalises the native engine's joined spelling
    (collapsing ``//``, converting ``/`` to ``\\`` under the win32
    flavour), reconciling it with the fallback's."""
    pure = PureWindowsPath if platform == "win32" else PurePosixPath
    return {key: str(pure(value)) for key, value in table.items()}


@contextmanager
def _reloaded_paths(env: dict[str, str | None]) -> Iterator[Any]:
    """Reload ``config.paths`` with *env* applied (``None`` deletes the
    variable), then reload it again under the restored environment — the
    module's constants are import-time values, so the second reload is
    what protects the rest of the session from the patched table."""
    saved = {key: os.environ.get(key) for key in env}
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(paths)
        yield paths
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        importlib.reload(paths)


class TestEngineContract:
    """The path table's platform/env contract, on whichever backend is
    loaded (synthetic triples — the win32 branch runs from any host)."""

    def test_posix_default_tree(self):
        table = _engine("linux", "/home/u", {})
        assert _norm(table, "linux")["CONFIG_DIR"] == "/home/u/.config/ytm-player"
        assert _norm(table, "linux")["CACHE_DIR"] == "/home/u/.cache/ytm-player/audio"
        assert _norm(table, "linux")["CACHE_DB"] == "/home/u/.cache/ytm-player/cache.db"
        assert _norm(table, "linux")["SOCKET_PATH"] == "/home/u/.config/ytm-player/ytm.sock"
        assert "IPC_PORT_FILE" not in table

    def test_posix_xdg_bases(self):
        table = _engine(
            "linux",
            "/home/u",
            {
                "XDG_CONFIG_HOME": "/xdg/c",
                "XDG_CACHE_HOME": "/xdg/cache",
                "XDG_RUNTIME_DIR": "/run/user/1000",
            },
        )
        assert _norm(table, "linux")["CONFIG_DIR"] == "/xdg/c/ytm-player"
        assert _norm(table, "linux")["CACHE_DIR"] == "/xdg/cache/ytm-player/audio"
        assert _norm(table, "linux")["CACHE_DB"] == "/xdg/cache/ytm-player/cache.db"
        assert _norm(table, "linux")["SOCKET_PATH"] == "/run/user/1000/ytm-player.sock"

    def test_posix_empty_xdg_reads_as_unset(self):
        empty = _engine("linux", "/home/u", {"XDG_CONFIG_HOME": "", "XDG_CACHE_HOME": ""})
        default = _engine("linux", "/home/u", {})
        assert _norm(empty, "linux") == _norm(default, "linux")

    def test_posix_empty_runtime_falls_back_to_config_dir(self):
        table = _engine("linux", "/home/u", {"XDG_RUNTIME_DIR": ""})
        assert _norm(table, "linux")["SOCKET_PATH"] == "/home/u/.config/ytm-player/ytm.sock"

    def test_posix_trailing_slash_bases_normalise(self):
        slashed = _engine(
            "linux",
            "/home/u",
            {"XDG_CONFIG_HOME": "/xdg/c/", "XDG_CACHE_HOME": "/xdg/cache/"},
        )
        clean = _engine(
            "linux",
            "/home/u",
            {"XDG_CONFIG_HOME": "/xdg/c", "XDG_CACHE_HOME": "/xdg/cache"},
        )
        assert _norm(slashed, "linux") == _norm(clean, "linux")

    def test_win32_bases(self):
        table = _engine(
            "win32", "C:\\Users\\u", {"APPDATA": _WIN_APPDATA, "LOCALAPPDATA": _WIN_LOCAL}
        )
        assert _norm(table, "win32")["CONFIG_DIR"] == str(
            PureWindowsPath(_WIN_APPDATA) / "ytm-player"
        )
        assert _norm(table, "win32")["CACHE_DIR"] == str(
            PureWindowsPath(_WIN_LOCAL) / "ytm-player" / "audio"
        )
        assert _norm(table, "win32")["CACHE_DB"] == str(
            PureWindowsPath(_WIN_LOCAL) / "ytm-player" / "cache.db"
        )
        assert _norm(table, "win32")["IPC_PORT_FILE"] == str(
            PureWindowsPath(_WIN_APPDATA) / "ytm-player" / "ipc_port"
        )
        assert "SOCKET_PATH" not in table

    def test_win32_home_tree_defaults(self):
        table = _engine("win32", "C:\\Users\\u", {})
        assert _norm(table, "win32")["CONFIG_DIR"] == str(
            PureWindowsPath("C:\\Users\\u") / "AppData" / "Roaming" / "ytm-player"
        )
        assert _norm(table, "win32")["CACHE_DIR"] == str(
            PureWindowsPath("C:\\Users\\u") / "AppData" / "Local" / "ytm-player" / "audio"
        )

    def test_win32_empty_appdata_stays_empty(self):
        # get-with-default, not a truthiness gate: an empty APPDATA is
        # kept, and Path("") / "ytm-player" is the *relative* table.
        table = _engine("win32", "C:\\Users\\u", {"APPDATA": "", "LOCALAPPDATA": _WIN_LOCAL})
        assert _norm(table, "win32")["CONFIG_DIR"] == "ytm-player"
        assert _norm(table, "win32")["CONFIG_FILE"] == str(
            PureWindowsPath("ytm-player") / "config.toml"
        )
        assert _norm(table, "win32")["CACHE_DIR"] == str(
            PureWindowsPath(_WIN_LOCAL) / "ytm-player" / "audio"
        )
        assert _norm(table, "win32")["IPC_PORT_FILE"] == str(
            PureWindowsPath("ytm-player") / "ipc_port"
        )

    def test_derived_files_under_config_dir(self):
        table = _engine("linux", "/home/u", {})
        norm = _norm(table, "linux")
        config_dir = norm["CONFIG_DIR"]
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
            assert norm[key] == str(PurePosixPath(config_dir) / leaf)
        assert norm["LOG_DIR"] == str(PurePosixPath(config_dir) / "logs")
        assert norm["LOG_FILE"] == str(PurePosixPath(config_dir) / "logs" / "ytm.log")
        assert norm["CRASH_DIR"] == str(PurePosixPath(config_dir) / "crashes")
        assert norm["CRASH_LOG"] == str(PurePosixPath(config_dir) / "crashes" / "ytm-crash.log")

    def test_table_has_the_full_constant_surface(self):
        table = _engine("linux", "/home/u", {})
        expected = {
            "CONFIG_DIR",
            "CACHE_DIR",
            "CACHE_DB",
            "CONFIG_FILE",
            "AUTH_FILE",
            "ACCOUNT_FILE",
            "STREAM_COOKIES_FILE",
            "OAUTH_FILE",
            "OAUTH_CREDS_FILE",
            "SPOTIFY_CREDS_FILE",
            "PID_FILE",
            "KEYMAP_FILE",
            "THEME_FILE",
            "RECENT_PLAYLISTS_FILE",
            "SESSION_STATE_FILE",
            "SHUFFLE_PREFS_FILE",
            "UPDATE_CHECK_CACHE",
            "HISTORY_DB",
            "LOG_DIR",
            "LOG_FILE",
            "CRASH_DIR",
            "CRASH_LOG",
            "SOCKET_PATH",
        }
        assert set(table) == expected
        win_table = _engine("win32", "C:\\Users\\u", {})
        assert set(win_table) == (expected - {"SOCKET_PATH"}) | {"IPC_PORT_FILE"}


class TestNativeParity:
    """When the Rust extension is built, the native table must equal the
    pure-Python reference's for every case, so the two implementations
    cannot drift (compared at the simulated platform's flavour lens —
    the flavour the module re-wraps every entry in ``Path()`` with on a
    native host)."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize(("platform", "home", "env"), _CASES)
    def test_engine_matches_fallback(self, platform: str, home: str, env: dict[str, str]):
        native = dict(_ytm_player_native.paths_core(platform, home, env))
        fallback = _compute_paths(platform, home, env)
        assert set(native) == set(fallback)
        assert _norm(native, platform) == _norm(fallback, platform)


class TestModuleConstants:
    """The import-time constants and the module's small I/O surface."""

    @pytest.mark.skipif(_IS_WIN32, reason="XDG bases are only read off Windows")
    def test_reload_honours_xdg_bases(self, tmp_path: Path):
        cfg, cache, runtime = (
            tmp_path / "cfg",
            tmp_path / "cache",
            tmp_path / "run",
        )
        with _reloaded_paths(
            {
                "XDG_CONFIG_HOME": str(cfg),
                "XDG_CACHE_HOME": str(cache),
                "XDG_RUNTIME_DIR": str(runtime),
            }
        ) as p:
            assert p.CONFIG_DIR == cfg / "ytm-player"
            assert p.CACHE_DIR == cache / "ytm-player" / "audio"
            assert p.CACHE_DB == cache / "ytm-player" / "cache.db"
            assert p.SOCKET_PATH == runtime / "ytm-player.sock"
            assert p.IPC_PORT_FILE is None

    @pytest.mark.skipif(_IS_WIN32, reason="XDG bases are only read off Windows")
    def test_reload_empty_xdg_reads_as_unset(self, tmp_path: Path):
        with _reloaded_paths(
            {
                "HOME": str(tmp_path),
                "XDG_CONFIG_HOME": "",
                "XDG_CACHE_HOME": "",
                "XDG_RUNTIME_DIR": "",
            }
        ) as p:
            assert p.CONFIG_DIR == tmp_path / ".config" / "ytm-player"
            assert p.CACHE_DIR == tmp_path / ".cache" / "ytm-player" / "audio"
            assert p.SOCKET_PATH == p.CONFIG_DIR / "ytm.sock"

    @pytest.mark.skipif(_IS_WIN32, reason="XDG bases are only read off Windows")
    def test_reload_runtime_fallback_to_config_dir(self):
        with _reloaded_paths({"XDG_RUNTIME_DIR": None}) as p:
            assert p.SOCKET_PATH == p.CONFIG_DIR / "ytm.sock"

    def test_reload_home_tree_defaults(self, tmp_path: Path):
        # Platform-neutral: clear every base override and move the home
        # directory; on Windows the AppData trees hang off it, elsewhere
        # the .config/.cache trees do.
        with _reloaded_paths(
            {
                "HOME": str(tmp_path),
                "USERPROFILE": str(tmp_path),
                "APPDATA": None,
                "LOCALAPPDATA": None,
                "XDG_CONFIG_HOME": None,
                "XDG_CACHE_HOME": None,
            }
        ) as p:
            assert p.CONFIG_DIR.is_relative_to(tmp_path)
            assert p.CACHE_DIR.is_relative_to(tmp_path)
            assert p.CONFIG_FILE.parent == p.CONFIG_DIR
            assert p.CRASH_LOG.parent == p.CRASH_DIR

    def test_platform_none_constants(self):
        # On win32 SOCKET_PATH is None (TCP transport); elsewhere
        # IPC_PORT_FILE is None (Unix sockets) — the branch the module
        # itself takes on this host.
        if _IS_WIN32:
            assert paths.SOCKET_PATH is None
            assert isinstance(paths.IPC_PORT_FILE, Path)
        else:
            assert isinstance(paths.SOCKET_PATH, Path)
            assert paths.IPC_PORT_FILE is None

    def test_ensure_dirs_creates_secure_directories(self, tmp_path: Path):
        with _reloaded_paths(
            {
                "HOME": str(tmp_path),
                "USERPROFILE": str(tmp_path),
                "APPDATA": None,
                "LOCALAPPDATA": None,
                "XDG_CONFIG_HOME": None,
                "XDG_CACHE_HOME": None,
            }
        ) as p:
            assert p._dirs_ensured is False
            p.ensure_dirs()
            for directory in (p.CONFIG_DIR, p.CACHE_DIR, p.LOG_DIR, p.CRASH_DIR):
                assert directory.is_dir()
            if not _IS_WIN32:
                # secure_chmod is a no-op on Windows (no Unix modes).
                assert (p.CONFIG_DIR.stat().st_mode & 0o777) == 0o700
            assert p._dirs_ensured is True

    @pytest.mark.skipif(_IS_WIN32, reason="secure_chmod is a no-op on Windows")
    def test_secure_chmod_sets_mode(self, tmp_path: Path):
        target = tmp_path / "file.json"
        target.write_text("{}", encoding="utf-8")
        paths.secure_chmod(target, 0o600)
        assert (target.stat().st_mode & 0o777) == 0o600
