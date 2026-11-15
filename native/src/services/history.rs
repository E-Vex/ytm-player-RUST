//! Native cores for `src/ytm_player/services/history.py` — play/search
//! history SQL builders, stat payloads and log-row assembly.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PySet, PyString, PyTuple};

// ---------------------------------------------------------------------------
// Play-history row engine (mirrors ytm_player/services/history.py)
// ---------------------------------------------------------------------------

/// `int(value)` — CPython's own builtin, dispatched directly.
///
/// The history row engine coerces caller-wide values (SQLite cells,
/// configured thresholds, track duration fields), where `int()`'s
/// acceptance set is larger than any transcription wants to spell out:
/// float truncation, decimal strings with signs, underscores, surrounding
/// whitespace and non-ASCII digits, `__int__`/`__index__` dispatch,
/// bigints beyond i64 — and the exact `ValueError`/`TypeError` messages
/// where the reference would raise. The coercion runs once per logged
/// play or finalised listen, as far from a hot path as the extension
/// gets, so exactness costs nothing measurable.
pub(crate) fn builtin_int<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let int_type = py.import("builtins")?.getattr("int")?;
    int_type.call1((value,))
}

/// The duration chain behind `log_play`'s row assembly — a duck-typed
/// transcription of `extract_duration` from `utils/formatting.py`, every
/// read through the track's own `get` method so a dict subclass
/// overriding it steers both backends identically.
///
/// `duration_seconds` is coerced through `int()` whenever present (the
/// reference's `is not None` gate — `False` and `""` both coerce), with
/// the `(ValueError, TypeError)` swallow: a numeric string parses, a
/// `"M:SS"` spelling falls through, a float truncates. `duration` and
/// `length` accept any int — bools included, the object itself returned
/// (`True` stays `True`, exactly like the fallback) — and
/// `":"`-bearing strings, split and recombined with Python arithmetic
/// (`int(a) * 60 + int(b)` for `M:SS`, the `H:MM:SS` three-part form,
/// a `ValueError` from either parse swallowed, everything else passing
/// through). No match yields 0. Where a read raises — a track without
/// `get`, or a `get` that raises — the error surfaces exactly where the
/// fallback's would.
fn play_extract_duration<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // dur = track.get("duration_seconds") — duck-typed, no default.
    let dur = track.call_method1("get", ("duration_seconds",))?;
    if !dur.is_none() {
        // return int(dur) — except (ValueError, TypeError): pass
        match builtin_int(py, &dur) {
            Ok(coerced) => return Ok(coerced),
            Err(err)
                if err.is_instance_of::<PyValueError>(py)
                    || err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {}
            Err(err) => return Err(err),
        }
    }
    for key in ["duration", "length"] {
        // dur = track.get(key)
        let dur = track.call_method1("get", (key,))?;
        // if isinstance(dur, int): return dur — the object itself, bools
        // included (the reference returns True, never 1).
        if dur.is_instance_of::<PyInt>() {
            return Ok(dur);
        }
        // if isinstance(dur, str) and ":" in dur
        if dur.is_instance_of::<PyString>() && dur.contains(":")? {
            // parts = dur.split(":") — the str protocol, subclass
            // overrides included.
            let parts = dur.call_method1("split", (":",))?;
            if let Some(seconds) = duration_parts_seconds(py, &parts)? {
                return Ok(seconds);
            }
        }
    }
    Ok(0i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any())
}

/// `int(parts[0]) * 60 + int(parts[1])` (and the `H:MM:SS` three-part
/// form) over a split duration string — Python's own arithmetic through
/// the number protocol, exact at any magnitude. A `ValueError` from
/// either `int()` parse maps to None: the reference's
/// `except ValueError: pass` falls through to the next duration key.
/// Everything else — a split override returning something unindexable,
/// a `TypeError` from a non-string part — propagates, exactly like the
/// fallback.
fn duration_parts_seconds<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // len(parts) == 2 / == 3 — PyObject_Size, raising like len() would.
    // Any other length (a "1:2:3:4" spelling, say) matches neither of the
    // reference's two if-statements and falls through to the next
    // duration key without parsing anything. The parses and arithmetic
    // live in the helper so a `?` there returns from *it*, landing in
    // the ValueError match below — not straight out of this function.
    let total = match parts.len()? {
        2 => combine_duration_parts(py, parts, 2),
        3 => combine_duration_parts(py, parts, 3),
        _ => return Ok(None),
    };
    match total {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.is_instance_of::<PyValueError>(py) => Ok(None),
        Err(err) => Err(err),
    }
}

/// The pieces of one split duration string, parsed and recombined —
/// `int(parts[0]) * 60 + int(parts[1])` for `M:SS`, the `H:MM:SS`
/// three-part form for a length of 3. Python's own arithmetic through
/// the number protocol, exact at any magnitude.
fn combine_duration_parts<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyAny>,
    len: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let first = builtin_int(py, &parts.get_item(0)?)?;
    let second = builtin_int(py, &parts.get_item(1)?)?;
    if len == 2 {
        return first.mul(60i64)?.add(&second);
    }
    let third = builtin_int(py, &parts.get_item(2)?)?;
    first.mul(3600i64)?.add(&second.mul(60i64)?)?.add(&third)
}

/// The row assembly behind `HistoryManager.log_play` — the pure core of
/// `services/history.py`'s play logging.
///
/// Mirrors the pure-Python reference (`_play_log_row`): the skip gate
/// comes first — `listened_seconds <= min_listen_seconds` through
/// Python's own `<=` (bools compare as ints, floats numerically, and the
/// same `TypeError` where Python's comparison would reject the pair) —
/// returning None for every listen the reference would drop before
/// touching anything else. Otherwise the insert row is assembled from
/// the track: `video_id` through `[]` (the reference's one indexing
/// read, a missing key raising KeyError), `title` / `artist` / `album`
/// through duck-typed `.get` with the `""` defaults, and `duration`
/// through the duration chain above. Every value passes through as the
/// object it is — `listened_seconds` and `source` included — into a
/// fresh map, and the inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (track, listened_seconds, source, min_listen_seconds))]
pub(crate) fn play_log_row<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
    listened_seconds: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
    min_listen_seconds: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // if listened_seconds <= min_listen_seconds: return None
    if listened_seconds.le(min_listen_seconds)? {
        return Ok(None);
    }
    // video_id = track["video_id"] — the [] read, KeyError wart included.
    let video_id = track.get_item("video_id")?;
    // title/artist/album = track.get(key, "") — duck-typed .get with the
    // "" defaults; an object without .get raises the reference's
    // AttributeError.
    let empty: Bound<'py, PyAny> = PyString::new(py, "").into_any();
    let title = track.call_method1("get", ("title", empty.clone()))?;
    let artist = track.call_method1("get", ("artist", empty.clone()))?;
    let album = track.call_method1("get", ("album", empty))?;
    // duration = extract_duration(track)
    let duration = play_extract_duration(py, track)?;
    let out = PyDict::new(py);
    out.set_item("video_id", video_id)?;
    out.set_item("title", title)?;
    out.set_item("artist", artist)?;
    out.set_item("album", album)?;
    out.set_item("duration", duration)?;
    out.set_item("listened_seconds", listened_seconds)?;
    out.set_item("source", source)?;
    Ok(Some(out))
}

/// The listened-seconds update decision behind
/// `HistoryManager.update_play_listened_seconds`.
///
/// Mirrors the pure-Python reference (`_listened_delta`): the stored
/// value is coerced with `int(old_value or 0)` — Python truthiness
/// picking the 0 for every falsy cell (None, 0, 0.0, "", False, empty
/// containers), the builtin `int()` for everything else, with its exact
/// `ValueError`/`TypeError` where the reference's coercion would raise
/// (both propagate identically out of the caller). A listen that does
/// not strictly exceed the stored seconds writes nothing (None).
/// Otherwise the delta — `listened_seconds - old_seconds` through
/// Python's own subtraction — comes back as the object the reference's
/// arithmetic would produce, exact at any magnitude.
#[pyfunction]
#[pyo3(signature = (old_value, listened_seconds))]
pub(crate) fn listened_delta<'py>(
    py: Python<'py>,
    old_value: &Bound<'py, PyAny>,
    listened_seconds: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // old_seconds = int(old_value or 0)
    let old_seconds = if old_value.is_truthy()? {
        builtin_int(py, old_value)?
    } else {
        0i64.into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any()
    };
    // if listened_seconds <= old_seconds: return
    if listened_seconds.le(&old_seconds)? {
        return Ok(None);
    }
    // delta = listened_seconds - old_seconds
    Ok(Some(listened_seconds.sub(&old_seconds)?))
}

/// The aggregate listening-stats payload behind
/// `HistoryManager.get_stats` — the pure core of the stats read.
///
/// Mirrors the pure-Python reference (`_stats_payload`): the five-key
/// dict assembled from the three scalar aggregates and the two top
/// lists. Every value passes through as the object it is — the scalar
/// counts straight from the SQLite rows, the top lists the very list
/// objects the caller fetched — into a fresh map, in the reference's
/// insertion order (the key order the JSON consumers' output
/// preserves). The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (total_plays, total_listen_time, unique_tracks, top_tracks, top_artists))]
pub(crate) fn stats_payload<'py>(
    py: Python<'py>,
    total_plays: &Bound<'py, PyAny>,
    total_listen_time: &Bound<'py, PyAny>,
    unique_tracks: &Bound<'py, PyAny>,
    top_tracks: &Bound<'py, PyAny>,
    top_artists: &Bound<'py, PyAny>,
) -> Bound<'py, PyDict> {
    let out = PyDict::new(py);
    // Only str keys are inserted — always hashable, so the store cannot
    // fail; each value is stored as the object it is, never coerced.
    out.set_item("total_plays", total_plays)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("total_listen_time", total_listen_time)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("unique_tracks", unique_tracks)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("top_tracks", top_tracks)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("top_artists", top_artists)
        .expect("dict insertion with a str key cannot fail");
    out
}

/// The row materialisation behind every history read — the
/// `[dict(row) for row in rows]` list comprehension the history
/// getters (and the CLI's `_query_history`) each ran inline.
///
/// Mirrors the pure-Python reference (`_row_dicts`): *rows* is walked
/// through Python's iteration protocol (a non-iterable raises
/// CPython's own `TypeError`; an `__iter__` that itself raises keeps
/// its own error) and each yielded row is converted through CPython's
/// own `dict` constructor — fetched from `builtins` and called with
/// the row as its single argument, so every acceptance shape and every
/// rejection message comes free: an exact dict is copied, anything
/// with a `keys` method is read through the mapping protocol
/// (`sqlite3.Row` — the production shape — and duck-typed rows
/// alike, a `keys`/`__getitem__` override on a dict subclass
/// steering the conversion), any other iterable is consumed as
/// key/value pairs, and everything else raises the constructor's own
/// `TypeError`. The result is a fresh list of fresh dicts — the input
/// rows are never mutated, and a dict row is copied, never passed
/// through by identity.
#[pyfunction]
#[pyo3(signature = (rows))]
pub(crate) fn row_dicts<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    // dict(row) — CPython's own constructor, fetched from builtins at
    // call time exactly like the comprehension's global lookup, so the
    // conversion semantics (mapping protocol, pair iteration, dict
    // copy, and every TypeError message) are the interpreter's own.
    let dict_ctor = PyModule::import(py, "builtins")?.getattr("dict")?;
    let out = PyList::empty(py);
    for row in rows.try_iter()? {
        let converted = dict_ctor.call1((row?,))?;
        // dict(...) always returns an exact dict, never a subclass.
        out.append(converted)?;
    }
    Ok(out)
}

// ── History row-assembly peripherals ───────────────────────────────────

/// The autocomplete string list behind
/// `HistoryManager.get_search_suggestions` — the pure core of the
/// suggestion read.
///
/// Mirrors the pure-Python reference (`_suggestion_strings`): the
/// `[row["query"] for row in rows]` comprehension — *rows* walked
/// through Python's iteration protocol (a non-iterable raises
/// CPython's own `TypeError`; an `__iter__` that itself raises keeps
/// its own error), each yielded row read through CPython's own
/// subscript (`row["query"]`, its `KeyError` wart included), and the
/// read values collected — as the objects they are — into a fresh
/// list. The input rows are never mutated.
#[pyfunction]
#[pyo3(signature = (rows))]
pub(crate) fn suggestion_strings<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for row in rows.try_iter()? {
        out.append(row?.get_item("query")?)?;
    }
    Ok(out)
}

/// The played-video-id set behind `HistoryManager.get_played_video_ids`
/// — the pure core of the Recently-Played filter read.
///
/// Mirrors the pure-Python reference (`_played_video_id_set`): the
/// `{row["video_id"] for row in rows}` set comprehension — *rows*
/// walked through Python's iteration protocol (a non-iterable raises
/// CPython's own `TypeError`; an `__iter__` that itself raises keeps
/// its own error), each yielded row read through CPython's own
/// subscript (`row["video_id"]`, its `KeyError` wart included), and
/// each read value inserted — as the object it is — into a fresh set
/// through CPython's own set-add: the value hashed first (an
/// unhashable value raises the hash protocol's own `TypeError`), then
/// the identity-or-`==` probe deciding membership, the FIRST inserted
/// of equal values the one the set keeps (`True` and `1` alias one
/// entry — whichever appeared first survives; the very same NaN
/// object collapses with itself, two distinct NaNs are two entries).
/// The input rows are never mutated.
#[pyfunction]
#[pyo3(signature = (rows))]
pub(crate) fn played_video_id_set<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PySet>> {
    let out = PySet::empty(py)?;
    for row in rows.try_iter()? {
        out.add(row?.get_item("video_id")?)?;
    }
    Ok(out)
}

/// The parameter tuple the play-history INSERT binds — the row
/// assembly behind `HistoryManager.log_play`'s first write.
///
/// Mirrors the pure-Python reference (`_play_insert_params`): the
/// seven values the `INSERT INTO play_history` statement binds, read
/// from the engine row through CPython's own subscript in the
/// reference's left-to-right order — `video_id`, `title`, `artist`,
/// `album`, `duration`, `listened_seconds`, `source` (a missing key's
/// `KeyError` propagating from exactly the read that missed) — and
/// passed through as the objects they are into a fresh tuple. The
/// input row is never mutated.
#[pyfunction]
#[pyo3(signature = (row))]
pub(crate) fn play_insert_params<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            row.get_item("video_id")?,
            row.get_item("title")?,
            row.get_item("artist")?,
            row.get_item("album")?,
            row.get_item("duration")?,
            row.get_item("listened_seconds")?,
            row.get_item("source")?,
        ],
    )
}

/// The parameter tuple the play-stats upsert binds — the row assembly
/// behind `HistoryManager.log_play`'s second write.
///
/// Mirrors the pure-Python reference (`_play_stats_params`): the four
/// values the `INSERT INTO play_stats ... ON CONFLICT` statement binds
/// — `video_id`, `title`, `artist`, `listened_seconds` — read through
/// CPython's own subscript in the reference's left-to-right order and
/// passed through as the objects they are into a fresh tuple. The
/// input row is never mutated.
#[pyfunction]
#[pyo3(signature = (row))]
pub(crate) fn play_stats_params<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            row.get_item("video_id")?,
            row.get_item("title")?,
            row.get_item("artist")?,
            row.get_item("listened_seconds")?,
        ],
    )
}

// ── History SQL bodies (services/history.py) ──────────────────────────────

/// The schema DDL script behind `HistoryManager.init`'s
/// `executescript` — the pure half of the database's first write.
///
/// Mirrors the pure-Python reference (`_history_schema`, returning the
/// module's `_SCHEMA` constant): the search-history table with its
/// unique query index, the play-history table and the play-stats
/// table, leading newline and zero-indent lines included — the exact
/// text the fallback returns, so a schema executes identically on
/// either backend.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn history_schema() -> String {
    "
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    filter_mode TEXT DEFAULT 'music',
    result_count INTEGER,
    search_count INTEGER DEFAULT 1,
    first_searched TEXT DEFAULT (datetime('now')),
    last_searched TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_search_query
    ON search_history(query, filter_mode);

CREATE TABLE IF NOT EXISTS play_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    title TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    duration_seconds INTEGER,
    listened_seconds INTEGER,
    source TEXT,
    played_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS play_stats (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT,
    play_count INTEGER DEFAULT 0,
    total_listened_seconds INTEGER DEFAULT 0,
    last_played TEXT
);
"
    .to_string()
}

/// The retention prune behind `HistoryManager.init`'s post-schema
/// write — the statement and the bound retention limit.
///
/// Mirrors the pure-Python reference (`_history_retention_delete`):
/// the statement text exactly as the fallback spells it (the
/// reference's two adjacent literals joined into the one line they
/// concatenate to), and the `max_history` value passed through as the
/// object it is into the parameter tuple — `(sql, (max_history,))`,
/// the shape `aiosqlite`'s `execute(*stmt)` unpacks.
#[pyfunction]
#[pyo3(signature = (max_history))]
pub(crate) fn history_retention_delete<'py>(
    py: Python<'py>,
    max_history: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "DELETE FROM play_history WHERE rowid NOT IN (SELECT rowid FROM play_history ORDER BY played_at DESC LIMIT ?)"
        .to_string();
    let params = PyTuple::new(py, [max_history])?;
    Ok((sql, params))
}

/// The search-history upsert behind `HistoryManager.log_search` — the
/// statement and the three bound values.
///
/// Mirrors the pure-Python reference (`_search_upsert`): the statement
/// text exactly as the fallback spells it (leading newline and
/// eight-space indentation included — the reference's method-body
/// spelling re-indented to the engine's module-level home, a spelling
/// SQLite parses identically), and `query` / `filter_mode` /
/// `result_count` passed through as the objects they are into the
/// parameter tuple.
#[pyfunction]
#[pyo3(signature = (query, filter_mode, result_count))]
pub(crate) fn search_upsert<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    filter_mode: &Bound<'py, PyAny>,
    result_count: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        INSERT INTO search_history (query, filter_mode, result_count)
        VALUES (?, ?, ?)
        ON CONFLICT(query, filter_mode) DO UPDATE SET
            search_count = search_count + 1,
            result_count = excluded.result_count,
            last_searched = datetime('now')
        "
    .to_string();
    let params = PyTuple::new(py, [query, filter_mode, result_count])?;
    Ok((sql, params))
}

/// The recent-searches select behind `HistoryManager.get_search_history`.
///
/// Mirrors the pure-Python reference (`_search_history_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is into the parameter tuple.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn search_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT query, filter_mode, result_count, search_count,
               first_searched, last_searched
        FROM search_history
        ORDER BY last_searched DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The autocomplete select behind
/// `HistoryManager.get_search_suggestions`.
///
/// Mirrors the pure-Python reference (`_search_suggestion_select`):
/// the statement text exactly as the fallback spells it, and `prefix`
/// / `limit` passed through as the objects they are into the
/// parameter tuple.
#[pyfunction]
#[pyo3(signature = (prefix, limit))]
pub(crate) fn search_suggestion_select<'py>(
    py: Python<'py>,
    prefix: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT query
        FROM search_history
        WHERE query LIKE ? || '%'
        ORDER BY search_count DESC, last_searched DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [prefix, limit])?;
    Ok((sql, params))
}

/// The clear-all statement behind
/// `HistoryManager.clear_search_history`.
///
/// Mirrors the pure-Python reference (`_search_history_delete`): the
/// bare statement text, no parameter tuple — the shape the reference
/// hands `execute` as its single argument.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn search_history_delete() -> String {
    "DELETE FROM search_history".to_string()
}

/// The play-history INSERT statement text behind
/// `HistoryManager.log_play`'s first write.
///
/// Mirrors the pure-Python reference (`_play_history_insert_sql`):
/// the statement text exactly as the fallback spells it. The seven
/// bound values are the `play_insert_params` engine's tuple (ported
/// earlier), so this engine returns the statement alone — the
/// reference's `execute(sql, params)` two-argument shape stays the
/// caller's to build.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn play_history_insert_sql() -> String {
    "
        INSERT INTO play_history
            (video_id, title, artist, album, duration_seconds,
             listened_seconds, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "
    .to_string()
}

/// The play-stats upsert statement text behind
/// `HistoryManager.log_play`'s second write.
///
/// Mirrors the pure-Python reference (`_play_stats_upsert_sql`): the
/// statement text exactly as the fallback spells it. The four bound
/// values are the `play_stats_params` engine's tuple (ported
/// earlier), so this engine returns the statement alone.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn play_stats_upsert_sql() -> String {
    "
        INSERT INTO play_stats (video_id, title, artist, play_count,
                                total_listened_seconds, last_played)
        VALUES (?, ?, ?, 1, ?, datetime('now'))
        ON CONFLICT(video_id) DO UPDATE SET
            title = excluded.title,
            artist = excluded.artist,
            play_count = play_count + 1,
            total_listened_seconds = total_listened_seconds + excluded.total_listened_seconds,
            last_played = datetime('now')
        "
    .to_string()
}

/// The stored-listening read behind
/// `HistoryManager.update_play_listened_seconds` — the statement and
/// the bound row id.
///
/// Mirrors the pure-Python reference (`_play_listened_select`): the
/// statement text exactly as the fallback spells it, and `play_id`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (play_id))]
pub(crate) fn play_listened_select<'py>(
    py: Python<'py>,
    play_id: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "SELECT video_id, listened_seconds FROM play_history WHERE id = ?".to_string();
    let params = PyTuple::new(py, [play_id])?;
    Ok((sql, params))
}

/// The finalised-listen update behind
/// `HistoryManager.update_play_listened_seconds` — the statement and
/// the two bound values.
///
/// Mirrors the pure-Python reference (`_play_listened_update`): the
/// statement text exactly as the fallback spells it, and
/// `listened_seconds` / `play_id` passed through as the objects they
/// are, in the statement's placeholder order.
#[pyfunction]
#[pyo3(signature = (listened_seconds, play_id))]
pub(crate) fn play_listened_update<'py>(
    py: Python<'py>,
    listened_seconds: &Bound<'py, PyAny>,
    play_id: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "UPDATE play_history SET listened_seconds = ? WHERE id = ?".to_string();
    let params = PyTuple::new(py, [listened_seconds, play_id])?;
    Ok((sql, params))
}

/// The stats-side listen accrual behind
/// `HistoryManager.update_play_listened_seconds` — the statement and
/// the two bound values.
///
/// Mirrors the pure-Python reference
/// (`_play_stats_listened_update`): the statement text exactly as the
/// fallback spells it, and `delta` / `video_id` passed through as the
/// objects they are, in the statement's placeholder order.
#[pyfunction]
#[pyo3(signature = (delta, video_id))]
pub(crate) fn play_stats_listened_update<'py>(
    py: Python<'py>,
    delta: &Bound<'py, PyAny>,
    video_id: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        UPDATE play_stats
        SET total_listened_seconds = total_listened_seconds + ?,
            last_played = datetime('now')
        WHERE video_id = ?
        "
    .to_string();
    let params = PyTuple::new(py, [delta, video_id])?;
    Ok((sql, params))
}

/// The play-history select behind `HistoryManager.get_play_history`.
///
/// Mirrors the pure-Python reference (`_play_history_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn play_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT video_id, title, artist, album, duration_seconds,
               listened_seconds, source, played_at
        FROM play_history
        ORDER BY played_at DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The deduplicated recently-played select behind
/// `HistoryManager.get_recently_played` — the one min/max-aggregate
/// statement whose bare columns read from the max row.
///
/// Mirrors the pure-Python reference (`_recently_played_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn recently_played_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT video_id, title, artist, album, duration_seconds,
               MAX(played_at) AS played_at
        FROM play_history
        GROUP BY video_id
        ORDER BY played_at DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The played-video-id read behind
/// `HistoryManager.get_played_video_ids` — the Recently-Played
/// filter's DISTINCT walk.
///
/// Mirrors the pure-Python reference (`_played_video_ids_select`):
/// the bare statement text, no parameter tuple — the shape the
/// reference hands `execute` as its single argument.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn played_video_ids_select() -> String {
    "SELECT DISTINCT video_id FROM play_history".to_string()
}

/// The total-plays aggregate behind `HistoryManager.get_stats`.
///
/// Mirrors the pure-Python reference (`_stats_total_plays_select`):
/// the bare statement text, no parameter tuple.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn stats_total_plays_select() -> String {
    "SELECT COUNT(*) AS total_plays FROM play_history".to_string()
}

/// The total-listen-time aggregate behind
/// `HistoryManager.get_stats`.
///
/// Mirrors the pure-Python reference
/// (`_stats_total_listen_time_select`): the bare statement text, no
/// parameter tuple.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn stats_total_listen_time_select() -> String {
    "SELECT COALESCE(SUM(listened_seconds), 0) AS s FROM play_history".to_string()
}

/// The unique-tracks aggregate behind `HistoryManager.get_stats`.
///
/// Mirrors the pure-Python reference (`_stats_unique_tracks_select`):
/// the bare statement text, no parameter tuple.
#[pyfunction]
#[pyo3(signature = ())]
pub(crate) fn stats_unique_tracks_select() -> String {
    "SELECT COUNT(*) AS c FROM play_stats".to_string()
}

/// The most-played-tracks select behind
/// `HistoryManager.get_top_tracks`.
///
/// Mirrors the pure-Python reference (`_top_tracks_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn top_tracks_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT video_id, title, artist, play_count,
               total_listened_seconds, last_played
        FROM play_stats
        ORDER BY play_count DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The most-played-artists select behind
/// `HistoryManager.get_top_artists` — the artist-grouped ranking with
/// its NULL/blank filter.
///
/// Mirrors the pure-Python reference (`_top_artists_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
pub(crate) fn top_artists_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT artist,
               SUM(play_count) AS play_count,
               SUM(total_listened_seconds) AS total_listened_seconds
        FROM play_stats
        WHERE artist IS NOT NULL AND artist != ''
        GROUP BY artist
        ORDER BY play_count DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}
