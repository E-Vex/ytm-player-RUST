//! Native cores for `src/ytm_player/services/cache.py` — eviction plans,
//! schema and index statements, hit-commit plans.

use crate::app::keys::unpack_dispatch_row;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

// ── Audio-cache LRU eviction engine ────────────────────────────────────

/// The `total -= file_size` of the eviction walk — CPython's in-place
/// subtraction, replicated protocol step for protocol step. The
/// augmented assignment reads the `__isub__` slot from the operand's
/// TYPE (an instance-attribute `__isub__` never runs — the operator
/// protocol reads the type) and uses its result unless it answers
/// `NotImplemented`; otherwise the binary subtraction runs — the very
/// `PyNumber_Subtract` the reference's `x - y` would run, subclass
/// reflection and all — and its result is what the assignment stores
/// back. Ints carry no `__isub__` (the production walk's int total,
/// straight from SQLite's SUM, always takes the binary path), and
/// where every slot misses, the binary failure's canonical
/// "unsupported operand type(s) for -" message is respelled with the
/// in-place protocol's own `-=` — the one shape the binary API cannot
/// spell.
fn py_inplace_sub<'py>(
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = lhs.py();
    let not_implemented = py.NotImplemented();
    if let Ok(in_place) = lhs.get_type().getattr("__isub__") {
        let result = in_place.call1((lhs, rhs))?;
        if !result.is(&not_implemented) {
            return Ok(result);
        }
    }
    match lhs.sub(rhs) {
        Ok(result) => Ok(result),
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(py) {
                let message = err.value(py).str()?.to_string_lossy().into_owned();
                if let Some(rest) = message.strip_prefix("unsupported operand type(s) for -: ") {
                    return Err(PyTypeError::new_err(format!(
                        "unsupported operand type(s) for -=: {rest}"
                    )));
                }
            }
            Err(err)
        }
    }
}

/// The LRU eviction walk behind `CacheManager.evict` from
/// `services/cache.py` — the pure core of the audio cache's overflow
/// handling.
///
/// Mirrors the pure-Python reference (`_eviction_plan`): *rows* is
/// walked in the caller's LRU order (oldest access first — the order
/// the reference's `ORDER BY last_accessed ASC` fetched), and before
/// each row the running total is compared against the budget —
/// `total_size <= max_bytes` through CPython's own `<=`, its
/// `TypeError` for an incomparable pair propagating — so an entry is
/// evicted only while the total is still over budget, exactly the
/// reference's check-then-evict shape: the entry that brings the total
/// exactly to the budget is the LAST evicted (the next check breaks).
/// The iteration begins before the first budget check — the reference's
/// `for` fetched the first row before its `if` ran — so a
/// non-iterable *rows* raises the iteration protocol's own `TypeError`
/// even when the total already fits. Each evicted row is read through
/// CPython's own subscript in the reference's read order —
/// `file_path` (the unlink's read), then `video_id` (the append's),
/// then `file_size` (the subtraction's) — and the three values pass
/// through into the returned `(video_id, file_path, file_size)` triple
/// as the objects they are; the running total is decremented through
/// the reference's augmented assignment (`total -= file_size`, see
/// `py_inplace_sub`). The result is a fresh list of fresh triples in
/// eviction order; the input rows are never mutated, and no file is
/// touched — the plan is computed before the first unlink (the
/// reference unlinked mid-walk; with SQLite rows, the only production
/// input, no read can fail, and the index — the source of truth — is
/// untouched either way), with the unlinking, the index DELETE and the
/// commit staying in the manager.
#[pyfunction]
#[pyo3(signature = (total_size, max_bytes, rows))]
pub(crate) fn eviction_plan<'py>(
    py: Python<'py>,
    total_size: &Bound<'py, PyAny>,
    max_bytes: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut total = total_size.clone();
    let out = PyList::empty(py);
    for row in rows.try_iter()? {
        let row = row?;
        // if total <= max_bytes: break — the check before each entry.
        if total.le(max_bytes)? {
            break;
        }
        // The reference's read order: the unlink's file_path, the
        // append's video_id, the subtraction's file_size. The triple
        // and the subtraction share the read object — a clone here is
        // the same Python object, never a copy.
        let file_path = row.get_item("file_path")?;
        let video_id = row.get_item("video_id")?;
        let file_size = row.get_item("file_size")?;
        let triple = PyTuple::new(py, [video_id, file_path, file_size.clone()])?;
        out.append(triple)?;
        // total -= file_size — the augmented assignment's own semantics.
        total = py_inplace_sub(&total, &file_size)?;
    }
    Ok(out)
}

// ── Audio-cache maintenance engines (services/cache.py) ───────────────

/// The schema DDL script behind `CacheManager.init` — the pure half
/// of the cache index's creation step (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_cache_schema`): the module's
/// `_SCHEMA` text — the single `cache_index` table with its
/// `video_id` primary key, its `file_path` / `file_size` / `format`
/// columns and its `cached_at` / `last_accessed` datetime defaults —
/// handed to `executescript` verbatim, the leading newline and
/// indentation the fallback spells included. The connection handling
/// stays Python-side in the manager.
#[pyfunction]
pub(crate) fn cache_schema() -> String {
    "\nCREATE TABLE IF NOT EXISTS cache_index (\n    video_id TEXT PRIMARY KEY,\n    file_path TEXT NOT NULL,\n    file_size INTEGER NOT NULL,\n    format TEXT,\n    cached_at TEXT DEFAULT (datetime('now')),\n    last_accessed TEXT DEFAULT (datetime('now'))\n);\n".to_string()
}

/// The hit-counter batched-commit gate behind `CacheManager.get` —
/// the pure core of the hit path's tail (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_hit_commit_plan`): the count
/// incremented through CPython's own `+` and the sum compared against
/// the module's own `_COMMIT_EVERY` — read at call time from
/// `ytm_player.services.cache`, so a patched constant steers both
/// backends identically (the same pattern as
/// `stream_cache_max_size`) — a sum that reaches the constant
/// answering `(0, True)`: the batched commit is due and the stored
/// count resets, exactly the reference's commit-then-reset; a sum
/// still short answering `(sum, False)`. The `last_accessed` UPDATE
/// around the engine and the commit itself stay Python-side in the
/// manager.
#[pyfunction]
#[pyo3(signature = (hit_count))]
pub(crate) fn cache_hit_commit_plan<'py>(
    py: Python<'py>,
    hit_count: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let commit_every =
        PyModule::import(py, "ytm_player.services.cache")?.getattr("_COMMIT_EVERY")?;
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let count = hit_count.add(&one)?;
    if count.ge(&commit_every)? {
        let zero = 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any();
        return Ok((zero, true));
    }
    Ok((count, false))
}

/// The usage summary behind `CacheManager.get_status` — the pure core
/// of the status read (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_status_summary`): the
/// four-key payload in the reference's insertion order — the two
/// aggregates passed through as the objects the row reads answered,
/// the MB → bytes budget computed through CPython's own
/// `max_size_mb * 1024 * 1024` (both multiplications, their errors
/// included), and the cache directory rendered through CPython's own
/// `str()` (a str input passes through as the same object) — into a
/// fresh dict. The row reads and the aggregate assert stay
/// Python-side in the manager.
#[pyfunction]
#[pyo3(signature = (total_size, file_count, max_size_mb, cache_dir))]
pub(crate) fn cache_status_summary<'py>(
    py: Python<'py>,
    total_size: &Bound<'py, PyAny>,
    file_count: &Bound<'py, PyAny>,
    max_size_mb: &Bound<'py, PyAny>,
    cache_dir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let k1024 = 1024i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let out = PyDict::new(py);
    out.set_item("total_size", total_size)?;
    out.set_item("file_count", file_count)?;
    out.set_item("max_size", max_size_mb.mul(&k1024)?.mul(&k1024)?)?;
    out.set_item("cache_dir", cache_dir.str()?)?;
    Ok(out)
}

/// The statement and bindings behind `CacheManager._index` — the
/// pure half of the index write (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_index_insert_statement`):
/// the `INSERT OR REPLACE` statement text — exactly as the fallback
/// spells it, leading newline and indentation included — and the
/// four-value parameter tuple: `video_id` / `file_size` / `format`
/// passed through as the objects they are, the path rendered through
/// CPython's own `str()` (a raising `__str__` keeps its own error,
/// exactly like the reference's `str()` call). The execute and the
/// commit stay Python-side in the manager.
#[pyfunction]
#[pyo3(signature = (video_id, path, file_size, format))]
pub(crate) fn cache_index_insert_statement<'py>(
    py: Python<'py>,
    video_id: &Bound<'py, PyAny>,
    path: &Bound<'py, PyAny>,
    file_size: &Bound<'py, PyAny>,
    format: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let params = PyTuple::new(
        py,
        [
            video_id.clone(),
            path.str()?.into_any(),
            file_size.clone(),
            format.clone(),
        ],
    )?;
    Ok((
        "\n        INSERT OR REPLACE INTO cache_index\n            (video_id, file_path, file_size, format)\n        VALUES (?, ?, ?, ?)\n        ".to_string(),
        params,
    ))
}

/// The statement and bindings behind `CacheManager.evict`'s index
/// cleanup — the pure half of the post-unlink write
/// (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_index_delete_statement`):
/// the video ids extracted from the plan's `(video_id, file_path,
/// file_size)` triples — each entry unpacked through the
/// reference's own `(video_id, _, _)` contract (any 3-iterable
/// works, a wrong-shaped entry raising CPython's own not-enough /
/// too-many `ValueError` and a non-iterable the unpack's own
/// `TypeError` rewrite; see `unpack_dispatch_row`) — the ids passing
/// through as the objects they are, in plan order — and the
/// placeholder-joined `IN` clause packed as the `(sql, ids)` pair
/// the reference's execute call takes: n ids spell n question
/// marks comma-joined, so an empty plan spells an `IN ()` the
/// manager's `if evicted:` gate never sends. The unlink loop and
/// the commit stay Python-side in the manager.
#[pyfunction]
#[pyo3(signature = (evicted))]
pub(crate) fn cache_index_delete_statement<'py>(
    py: Python<'py>,
    evicted: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let mut ids: Vec<Bound<'py, PyAny>> = Vec::new();
    for triple in evicted.try_iter()? {
        let (video_id, _file_path, _file_size) = unpack_dispatch_row(&triple?)?;
        ids.push(video_id);
    }
    let placeholders = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("DELETE FROM cache_index WHERE video_id IN ({placeholders})");
    let out = PyList::new(py, ids).expect("list of ids cannot fail");
    Ok((sql, out))
}

/// The statement behind `CacheManager.clear`'s index wipe — the pure
/// half of the wipe's index write (`services/cache.py`).
///
/// Mirrors the pure-Python reference
/// (`_index_delete_all_statement`): the text handed to the
/// reference's execute call, verbatim. The file unlink walk and the
/// commit stay Python-side in the manager.
#[pyfunction]
pub(crate) fn cache_index_delete_all_statement() -> &'static str {
    "DELETE FROM cache_index"
}
