//! Native core for `src/ytm_player/config/paths.py` — the platform-aware
//! path table behind every filesystem constant the module exports.

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// The path table (config/paths.py)
// ---------------------------------------------------------------------------

/// Join path components the way the module's ``Path(base) / part``
/// arithmetic lands in a string: a plain ``/``-separated append, except
/// that an empty base is ``.`` to pathlib (``Path("") / "ytm-player"``
/// is the relative ``ytm-player``), so the separator is dropped there.
/// The Python caller re-wraps every value in ``Path()``, whose lexical
/// normalisation (collapsing ``//`` and ``./``) reconciles the joined
/// spelling with the pathlib one for every non-degenerate base.
fn pjoin(base: &str, parts: &[&str]) -> String {
    if base.is_empty() {
        return parts.join("/");
    }
    let mut out =
        String::with_capacity(base.len() + parts.iter().map(|p| p.len() + 1).sum::<usize>());
    out.push_str(base);
    for part in parts {
        out.push('/');
        out.push_str(part);
    }
    out
}

/// Read a string variable from the caller's environment snapshot. A
/// missing key and a ``None`` value both read as unset; an empty string
/// is returned as-is (the caller applies the reference's own
/// set-vs-empty distinction).
fn env_get(env: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match env.get_item(key)? {
        Some(v) if !v.is_none() => Ok(Some(v.extract()?)),
        _ => Ok(None),
    }
}

/// The platform-aware path table from `config/paths.py` — the pure core
/// of every path constant the module exports. Runs natively when the
/// extension is built; the module's `_compute_paths` transcription is
/// the fallback.
///
/// The engine: the win32 `APPDATA` / `LOCALAPPDATA` bases (defaults only
/// when the variable is *unset* — an empty `APPDATA` stays empty, and
/// pathlib then lands on the relative table, which `pjoin`'s empty-base
/// rule reproduces), the XDG `XDG_CONFIG_HOME` / `XDG_CACHE_HOME` /
/// `XDG_RUNTIME_DIR` bases (an empty string reads as unset, matching the
/// reference's truthiness gates), and the twenty-one derived files and
/// directories under `CONFIG_DIR`. `SOCKET_PATH` is absent from the
/// table on win32 and `IPC_PORT_FILE` elsewhere — an absent key is the
/// module's ``None`` constant on that platform.
///
/// The engine is a pure function of `(platform, home, env)` so the
/// win32 branch is exercisable from any host; the module passes
/// `sys.platform`, `str(Path.home())` and `dict(os.environ)` at import.
/// `Path.home()`'s own resolution, the `ensure_dirs` / `secure_chmod`
/// I/O and the `_dirs_ensured` latch stay Python-side in the module.
#[pyfunction]
#[pyo3(signature = (platform, home, env))]
pub(crate) fn paths_core(
    _py: Python,
    platform: &str,
    home: &str,
    env: &Bound<'_, PyDict>,
) -> PyResult<Py<PyDict>> {
    let table = PyDict::new(_py);

    let config_dir;
    if platform == "win32" {
        // os.environ.get("APPDATA", home/AppData/Roaming): the default
        // only fires when the variable is unset — an empty APPDATA
        // stays empty (the reference's get-with-default, not a
        // truthiness gate).
        let app_data = match env_get(env, "APPDATA")? {
            Some(v) => v,
            None => pjoin(home, &["AppData", "Roaming"]),
        };
        let local_data = match env_get(env, "LOCALAPPDATA")? {
            Some(v) => v,
            None => pjoin(home, &["AppData", "Local"]),
        };
        config_dir = pjoin(&app_data, &["ytm-player"]);
        table.set_item("CACHE_DIR", pjoin(&local_data, &["ytm-player", "audio"]))?;
        table.set_item("CACHE_DB", pjoin(&local_data, &["ytm-player", "cache.db"]))?;
        table.set_item("IPC_PORT_FILE", pjoin(&config_dir, &["ipc_port"]))?;
        // SOCKET_PATH is None on Windows — absent from the table.
    } else {
        // if _xdg_config: — None *or* the empty string is falsy, so
        // both read as unset here (unlike the win32 get-with-default).
        let xdg_config = env_get(env, "XDG_CONFIG_HOME")?;
        config_dir = match xdg_config {
            Some(ref v) if !v.is_empty() => pjoin(v, &["ytm-player"]),
            _ => pjoin(home, &[".config", "ytm-player"]),
        };
        let xdg_cache = env_get(env, "XDG_CACHE_HOME")?;
        let cache_base = match xdg_cache {
            Some(ref v) if !v.is_empty() => pjoin(v, &["ytm-player"]),
            _ => pjoin(home, &[".cache", "ytm-player"]),
        };
        table.set_item("CACHE_DIR", pjoin(&cache_base, &["audio"]))?;
        table.set_item("CACHE_DB", pjoin(&cache_base, &["cache.db"]))?;
        let xdg_runtime = env_get(env, "XDG_RUNTIME_DIR")?;
        let socket_path = match xdg_runtime {
            Some(ref v) if !v.is_empty() => pjoin(v, &["ytm-player.sock"]),
            _ => pjoin(&config_dir, &["ytm.sock"]),
        };
        table.set_item("SOCKET_PATH", socket_path)?;
        // IPC_PORT_FILE is None on Unix — absent from the table.
    }

    for (key, leaf) in [
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
    ] {
        table.set_item(key, pjoin(&config_dir, &[leaf]))?;
    }
    let log_dir = pjoin(&config_dir, &["logs"]);
    table.set_item("LOG_DIR", &log_dir)?;
    table.set_item("LOG_FILE", pjoin(&log_dir, &["ytm.log"]))?;
    let crash_dir = pjoin(&config_dir, &["crashes"]);
    table.set_item("CRASH_DIR", &crash_dir)?;
    // CRASH_LOG is the legacy crash-file path, kept for
    // backwards-compatibility with old imports; new crash files are
    // written under CRASH_DIR with timestamped names.
    table.set_item("CRASH_LOG", pjoin(&crash_dir, &["ytm-crash.log"]))?;
    table.set_item("CONFIG_DIR", &config_dir)?;

    Ok(table.unbind())
}
