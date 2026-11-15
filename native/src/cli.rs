//! Native cores for `src/ytm_player/cli.py` — stats/history/cache-status
//! payload assembly and the config editor plan for `ytm` subcommands.

use crate::utils::formatting::format_size;
use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

/// The aggregate listening-stats payload behind the CLI's standalone
/// `ytm stats` command — the pure core of the stats read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cli_stats_payload`): the
/// four-key dict assembled from the three scalar aggregates and the
/// top-tracks list, in the reference's insertion order (the key order
/// the JSON output preserves — `total_plays`, `total_seconds`,
/// `unique_tracks`, `top_tracks`). The three scalars pass through as
/// the objects they are — the counts straight from the SQLite
/// `fetchone()` cells. The top-tracks list is rebuilt by walking
/// *top_tracks* through Python's iteration protocol and reading each
/// row's first four fields through `r[0]` … `r[3]` subscript semantics
/// (`PyObject_GetItem`, exactly what the comprehension's
/// `BINARY_SUBSCR` compiles to): a tuple row yields `IndexError` when
/// shorter than four, a dict row `KeyError` on the int key, a
/// duck-typed `__getitem__` row steers both backends identically, and
/// a longer row's extras are simply never read — each row becoming a
/// fresh `{"video_id", "title", "artist", "play_count"}` map in that
/// order. The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (total_plays, total_seconds, unique_tracks, top_tracks))]
pub(crate) fn cli_stats_payload<'py>(
    py: Python<'py>,
    total_plays: &Bound<'py, PyAny>,
    total_seconds: &Bound<'py, PyAny>,
    unique_tracks: &Bound<'py, PyAny>,
    top_tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // Only str keys are inserted — always hashable, so the store cannot
    // fail; each scalar is stored as the object it is, never coerced.
    out.set_item("total_plays", total_plays)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("total_seconds", total_seconds)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("unique_tracks", unique_tracks)
        .expect("dict insertion with a str key cannot fail");
    // [{"video_id": r[0], "title": r[1], "artist": r[2],
    //   "play_count": r[3]} for r in top_tracks]
    let items = PyList::empty(py);
    for row in top_tracks.try_iter()? {
        let row = row?;
        let item = PyDict::new(py);
        for (key, index) in [
            ("video_id", 0usize),
            ("title", 1usize),
            ("artist", 2usize),
            ("play_count", 3usize),
        ] {
            // r[index] — PyObject_GetItem: IndexError on a short
            // sequence, KeyError(0) on a dict row, whatever a
            // duck-typed __getitem__ decides.
            item.set_item(key, row.get_item(index)?)?;
        }
        items.append(item)?;
    }
    out.set_item("top_tracks", items)
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The play-history select behind the CLI's standalone `ytm history`
/// command — the pure half of the history read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cli_history_select`): the
/// statement text exactly as the fallback spells it — every column,
/// newest play first, bounded by the caller's limit — and `limit`
/// passed through as the object it is into the parameter tuple, packed
/// as the `(sql, params)` pair the reference's `_query_history` hands
/// `execute`.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn cli_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "SELECT * FROM play_history ORDER BY played_at DESC LIMIT ?".to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The search-history select behind the CLI's standalone
/// `ytm history search` command — the pure half of the search-history
/// read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cli_search_history_select`):
/// the statement text exactly as the fallback spells it — every
/// column, most recently searched first, bounded by the caller's
/// limit — and `limit` passed through as the object it is into the
/// parameter tuple, packed as the `(sql, params)` pair the reference's
/// `_query_history` hands `execute`.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn cli_search_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "SELECT * FROM search_history ORDER BY last_searched DESC LIMIT ?".to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The status payload behind the CLI's standalone `ytm cache status`
/// command — the pure core of the status read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cache_status_payload`): the
/// six-key dict assembled from the cache directory and the walk's
/// three aggregates, in the reference's insertion order (the key order
/// the JSON output preserves — `cache_dir`, `file_count`,
/// `total_bytes`, `total_human`, `db_size_bytes`, `db_size_human`).
/// `str(cache_dir)` renders the directory exactly like the reference's
/// dict literal (a `Path` as its path text); `file_count`,
/// `total_bytes` and `db_size` pass through as the objects they are —
/// the count and the sums straight from the directory walk — and the
/// two human spellings come from the same `format_size` engine the
/// Python side's `from …formatting import format_size` resolves to
/// when the extension is built, applied through the same conversion.
/// The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (cache_dir, file_count, total_bytes, db_size))]
pub(crate) fn cache_status_payload<'py>(
    py: Python<'py>,
    cache_dir: &Bound<'py, PyAny>,
    file_count: &Bound<'py, PyAny>,
    total_bytes: &Bound<'py, PyAny>,
    db_size: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // str(cache_dir) — rendered like the reference's dict literal.
    out.set_item("cache_dir", py_str(cache_dir)?)
        .expect("dict insertion with a str key cannot fail");
    // Each aggregate is stored as the object it is, never coerced.
    out.set_item("file_count", file_count)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("total_bytes", total_bytes)
        .expect("dict insertion with a str key cannot fail");
    // format_size(total_bytes) — the same engine the Python side binds
    // to, through the same conversion its own call would perform.
    out.set_item("total_human", format_size(total_bytes.extract::<i64>()?))
        .expect("dict insertion with a str key cannot fail");
    out.set_item("db_size_bytes", db_size)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("db_size_human", format_size(db_size.extract::<i64>()?))
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The human-readable status text behind the CLI's standalone
/// `ytm cache status` command — the four aligned lines the command
/// echoes when `--json` is absent, joined by the newlines the four
/// `click.echo` calls appended (the caller's own echo supplies the
/// trailing one).
///
/// Mirrors the pure-Python reference (`_cache_status_text`): the
/// `Cache directory:` line renders `str(cache_dir)` exactly like the
/// reference's f-string (a `Path` as its path text), `Cached files:`
/// interpolates the count object through `str()` as it is, and the
/// two size lines come from the same `format_size` engine the Python
/// side binds to, through the same conversion its own calls would
/// perform. The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (cache_dir, file_count, total_bytes, db_size))]
pub(crate) fn cache_status_text(
    cache_dir: &Bound<'_, PyAny>,
    file_count: &Bound<'_, PyAny>,
    total_bytes: &Bound<'_, PyAny>,
    db_size: &Bound<'_, PyAny>,
) -> PyResult<String> {
    Ok(format!(
        "Cache directory: {}\nCached files:    {}\nTotal size:      {}\nDatabase size:   {}",
        py_str(cache_dir)?,
        py_str(file_count)?,
        format_size(total_bytes.extract::<i64>()?),
        format_size(db_size.extract::<i64>()?),
    ))
}

/// The directory walk behind the CLI's standalone `ytm cache clear`
/// command — the pure core of the delete in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cache_clear_walk`): when the
/// *dir_exists* thunk (the reference's `CACHE_DIR.exists()`) reports the
/// directory, every entry the *list_entries* thunk yields (the
/// reference's `CACHE_DIR.iterdir()`) that the *is_file* thunk accepts
/// is handed to the *unlink* thunk and counted — subdirectories and
/// everything that is not a plain file are left untouched — and the
/// walk returns the count the reference's `removed` accumulates. Each
/// thunk is invoked exactly where the reference performs its own
/// operation — the existence probe once up front, the listing only
/// when the directory exists, the file test once per entry in
/// iteration order, the unlink only for the entries that pass it — so
/// the filesystem side effects stay Python's (a `Path`, a `Path`
/// subclass or a stub steering both backends identically) while the
/// walk's control flow runs native. An error from any thunk propagates
/// exactly where the reference's own operation would raise it, leaving
/// the count behind it unreturned; the count is a fresh `int`; the
/// inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (dir_exists, list_entries, is_file, unlink))]
pub(crate) fn cache_clear_walk(
    dir_exists: &Bound<'_, PyAny>,
    list_entries: &Bound<'_, PyAny>,
    is_file: &Bound<'_, PyAny>,
    unlink: &Bound<'_, PyAny>,
) -> PyResult<u64> {
    let mut removed: u64 = 0;
    if dir_exists.call0()?.is_truthy()? {
        for entry in list_entries.call0()?.try_iter()? {
            let entry = entry?;
            if is_file.call1((&entry,))?.is_truthy()? {
                unlink.call1((&entry,))?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

/// The directory walk behind the CLI's standalone `ytm cache status`
/// command — the pure core of the status read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cache_status_walk`): when the
/// *dir_exists* thunk (the reference's `CACHE_DIR.exists()`) reports the
/// directory, every entry the *list_entries* thunk yields (the
/// reference's `CACHE_DIR.iterdir()`) that the *is_file* thunk accepts
/// contributes its *stat_size* thunk result (the reference's
/// `f.stat().st_size`) to the running totals — subdirectories and
/// everything that is not a plain file are left out — and the walk
/// returns the pair the reference's two accumulators build, the count
/// first like the reference's return. Each thunk is invoked exactly
/// where the reference performs its own operation — the existence probe
/// once up front, the listing only when the directory exists, the file
/// test once per entry in iteration order, the size read only for the
/// entries that pass it — so the filesystem side effects stay Python's
/// (a `Path`, a `Path` subclass or a stub steering both backends
/// identically) while the walk's control flow runs native. The size
/// read precedes the count increment exactly like the reference's
/// statement order, so an error from it leaves that entry uncounted
/// behind it; both accumulations run through CPython's own in-place add
/// (`operator.iadd` — `PyNumber_InPlaceAdd`), so the int + int case
/// stays an exact int whatever the sizes, and a stub's `st_size` that
/// returns something exotic follows the same numeric protocol the
/// reference's `+=` would (a float sum stays a float, a bigint stays
/// unbounded). An error from any thunk propagates exactly where the
/// reference's own operation would raise it, leaving the pair
/// unreturned; the pair is a fresh tuple; the inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (dir_exists, list_entries, is_file, stat_size))]
pub(crate) fn cache_status_walk<'py>(
    py: Python<'py>,
    dir_exists: &Bound<'py, PyAny>,
    list_entries: &Bound<'py, PyAny>,
    is_file: &Bound<'py, PyAny>,
    stat_size: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    // total_bytes = 0 / file_count = 0 — the reference's starting ints,
    // kept as the Python objects the additions build (an unbounded sum
    // renders exactly like the reference's, never a fixed-width
    // coercion).
    let mut total_bytes: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    let mut file_count: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    // operator.iadd — CPython's own += (PyNumber_InPlaceAdd), so both
    // accumulations follow the reference's numeric protocol exactly.
    let iadd = py.import("operator")?.getattr("iadd")?;
    // file_count += 1 — the reference's literal increment, pre-built.
    let one: Bound<'py, PyAny> = 1i64.into_pyobject(py)?.into_any();
    if dir_exists.call0()?.is_truthy()? {
        for entry in list_entries.call0()?.try_iter()? {
            let entry = entry?;
            if is_file.call1((&entry,))?.is_truthy()? {
                // total_bytes += f.stat().st_size — the size read first,
                // its result accumulated through the in-place add.
                let size = stat_size.call1((&entry,))?;
                total_bytes = iadd.call1((&total_bytes, &size))?;
                // file_count += 1 — then the count increment, exactly the
                // reference's statement order.
                file_count = iadd.call1((&file_count, &one))?;
            }
        }
    }
    // return file_count, total_bytes — the reference's pair order.
    Ok((file_count, total_bytes))
}

/// The editor resolution behind the CLI's standalone `ytm config`
/// command — the pure core of the config open in `cli.py`.
///
/// Mirrors the pure-Python reference (`_config_editor_plan`): a truthy
/// *editor_env* (the `EDITOR` environment value — an empty string is
/// falsy and falls through, exactly like the reference's `if editor:`)
/// opens the config *file* with that editor; otherwise a truthy
/// *is_win32* (the reference's `sys.platform == "win32"`, computed by
/// the caller) shell-opens the file (`os.startfile`, whose call stays
/// in the caller); otherwise the *resolve_xdg* thunk — the reference's
/// `shutil.which("xdg-open")` probe, passed as a zero-argument
/// callable and invoked exactly here, exactly once, only on this
/// branch, so the PATH scan never runs when an `EDITOR` is set — is
/// asked for `xdg-open`, whose result opens the config *directory*.
/// The `which` verdict is an `is None` check, not a truthiness one —
/// an empty-string path is still the editor — and a `None` result is
/// the manual-open error, its message naming the directory through
/// `str()` exactly like the reference's f-string. The editor value
/// passes through as the object it is; the targets render through
/// `str()` exactly like the reference's assignments; the plan is a
/// fresh map; the inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (editor_env, is_win32, resolve_xdg, config_file, config_dir))]
pub(crate) fn config_editor_plan<'py>(
    py: Python<'py>,
    editor_env: &Bound<'py, PyAny>,
    is_win32: &Bound<'py, PyAny>,
    resolve_xdg: &Bound<'py, PyAny>,
    config_file: &Bound<'py, PyAny>,
    config_dir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if editor_env.is_truthy()? {
        out.set_item("kind", "editor")?;
        out.set_item("editor", editor_env)?;
        out.set_item("target", py_str(config_file)?)?;
    } else if is_win32.is_truthy()? {
        out.set_item("kind", "startfile")?;
        out.set_item("target", py_str(config_file)?)?;
    } else {
        // shutil.which("xdg-open") — the probe, invoked exactly here
        // and exactly once, only on the no-EDITOR no-Windows branch.
        let editor = resolve_xdg.call0()?;
        let target = py_str(config_dir)?;
        // `if editor is None` — not a truthiness gate: an empty-string
        // path is still the editor.
        if editor.is_none() {
            out.set_item("kind", "error")?;
            out.set_item(
                "message",
                format!(
                    "No $EDITOR set and xdg-open not found. Open manually: {}",
                    py_str(config_dir)?
                ),
            )?;
        } else {
            out.set_item("kind", "editor")?;
            out.set_item("editor", editor)?;
            out.set_item("target", target)?;
        }
    }
    Ok(out)
}
