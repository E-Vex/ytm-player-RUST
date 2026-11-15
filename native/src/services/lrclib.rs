//! Native cores for `src/ytm_player/services/lrclib.py` — LRCLIB query
//! params and the synced-lyrics response adapter.

use crate::services::history::builtin_int;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The query-params assembly behind the LRCLIB lookup — the pure core
/// of `get_synced_lyrics`' request build in `services/lrclib.py`.
///
/// Mirrors the pure-Python transcription (`_query_params`): the fresh
/// dict with `track_name` (the sanitized title, passed through as the
/// object it is) and `artist_name` (the artist, likewise) in the
/// reference's insertion order, then — only when *duration_seconds* is
/// not `None`, an identity test so a falsy-but-present `0` still
/// travels — the `duration` key set to
/// `str(int(duration_seconds))`, the `int()` through the interpreter's
/// own (so a float truncates, a bool coerces, a numeric string parses
/// and a bad one raises `int()`'s own `ValueError`) and the `str()`
/// through the object's own render. The title sanitizer itself is the
/// already-ported formatting engine and stays at the caller; the
/// `requests` fetch stays Python-side.
#[pyfunction]
pub(crate) fn lrclib_query_params<'py>(
    py: Python<'py>,
    clean_title: &Bound<'py, PyAny>,
    artist: &Bound<'py, PyAny>,
    duration_seconds: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let params = PyDict::new(py);
    params.set_item("track_name", clean_title)?;
    params.set_item("artist_name", artist)?;
    // if duration_seconds is not None — the identity gate, never
    // truthiness (a 0 duration still travels).
    if !duration_seconds.is_none() {
        // str(int(duration_seconds)) — int() through the interpreter's
        // own, str() through the result's own render.
        let coerced = builtin_int(py, duration_seconds)?;
        let rendered = coerced.str()?;
        params.set_item("duration", rendered)?;
    }
    Ok(params)
}

/// The synced-lyrics extraction behind the LRCLIB response — the pure
/// core of `get_synced_lyrics`' parse step in `services/lrclib.py`.
///
/// Mirrors the pure-Python transcription (`_synced_lyrics`):
/// `data.get("syncedLyrics")` — the member read through the mapping's
/// own `get`, so a dict subclass override steers both backends
/// identically and a payload without `get` raises the call's own
/// `AttributeError` — then the `or None` truthiness gate: a falsy
/// member (the empty string, `None`, anything falsy a stub might
/// answer) becomes `None`, a truthy one passes through as the object
/// it is. The `resp.json()` decode stays Python-side (requests' own
/// decoder and its exception shapes).
#[pyfunction]
pub(crate) fn lrclib_synced_lyrics<'py>(data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // data.get("syncedLyrics") — through the mapping's own get.
    let value = data.call_method1("get", ("syncedLyrics",))?;
    // `or None` — Python truthiness decides.
    if value.is_truthy()? {
        Ok(value)
    } else {
        Ok(data.py().None().into_bound(data.py()))
    }
}
