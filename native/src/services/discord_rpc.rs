//! Native cores for `src/ytm_player/services/discord_rpc.py` — presence
//! kwargs assembly (incl. the 128-char `or` slice).

use crate::services::history::builtin_int;
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Discord presence assembly (mirrors ytm_player/services/discord_rpc.py)
// ---------------------------------------------------------------------------

/// `(value or fallback)[:128]` — the presence text arm: the
/// truthiness gate then the 128 cut, both through CPython's own
/// protocols, so a falsy value (the empty string, `None`, `0`) answers
/// the fallback, a truthy value is sliced through its own
/// `__getitem__` over the very slice object the reference's `[:128]`
/// compiles to (a str answers its 128-head, a non-subscriptable
/// object raises its own `TypeError`), and a truthiness error
/// propagates.
fn discord_or_slice_128<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    fallback: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let chosen = if value.is_truthy()? {
        value.clone()
    } else {
        fallback.into_pyobject(py)?.into_any()
    };
    // chosen[:128] — the getitem protocol over the slice object.
    let slice_type = py.import("builtins")?.getattr("slice")?;
    let window = slice_type.call1((py.None(), 128i64, py.None()))?;
    chosen.get_item(&window)
}

/// The presence-kwargs assembly — `services/discord_rpc.py`'s `update`
/// kwargs build, the seven-key dict handed to pypresence's
/// `update(**kwargs)` on every presence refresh.
///
/// Mirrors the reference's dict literal exactly, in its own key order
/// and with each value's own error surfacing first: `activity_type`
/// the very object handed in (the `ActivityType.LISTENING` member
/// where pypresence provides the enum, `None` where it does not — the
/// import gate stays Python-side in the caller), `details`/`state`/
/// `large_text` the `(x or "…")[:128]` arms — the truthiness gate
/// through CPython's own `bool` protocol, the 128 cut through the
/// value's own `__getitem__` — `large_image` the bare `or` arm
/// (`thumbnail_url or "ytm_icon"`, unsliced: a truthy thumbnail
/// passes through whole), and `start` the interpreter's own
/// `int(start_time)`. Once the six-key dict is complete, the end gate
/// `duration > 0` through CPython's own `>` appends `end` as
/// `int(start_time + duration)` — the addition through CPython's own
/// `+`, the cut through the interpreter's own `int()` — so a positive
/// duration spells the countdown's far edge, a zero or negative one
/// leaves the key absent, and a non-numeric duration or start time
/// raises the comparison's, addition's or `int()`'s own error. The
/// clock read and the `_start_time` state write, the pypresence
/// import, the reconnect loop and the RPC call stay Python-side in
/// the service.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the presence's seven inputs — one parameter per field the reference's kwargs read
#[pyo3(signature = (title, artist, album, duration, thumbnail_url, start_time, activity_type))]
pub(crate) fn discord_presence_kwargs<'py>(
    py: Python<'py>,
    title: &Bound<'py, PyAny>,
    artist: &Bound<'py, PyAny>,
    album: &Bound<'py, PyAny>,
    duration: &Bound<'py, PyAny>,
    thumbnail_url: &Bound<'py, PyAny>,
    start_time: &Bound<'py, PyAny>,
    activity_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    // "activity_type" — the very object handed in.
    kwargs
        .set_item("activity_type", activity_type)
        .expect("dict insertion with a str key cannot fail");
    // "details" — (title or "Unknown Track")[:128].
    let details = discord_or_slice_128(py, title, "Unknown Track")?;
    kwargs
        .set_item("details", details)
        .expect("dict insertion with a str key cannot fail");
    // "state" — (artist or "Unknown Artist")[:128].
    let state = discord_or_slice_128(py, artist, "Unknown Artist")?;
    kwargs
        .set_item("state", state)
        .expect("dict insertion with a str key cannot fail");
    // "large_image" — thumbnail_url or "ytm_icon", unsliced.
    let large_image = if thumbnail_url.is_truthy()? {
        thumbnail_url.clone()
    } else {
        "ytm_icon".into_pyobject(py)?.into_any()
    };
    kwargs
        .set_item("large_image", large_image)
        .expect("dict insertion with a str key cannot fail");
    // "large_text" — (album or "YouTube Music")[:128].
    let large_text = discord_or_slice_128(py, album, "YouTube Music")?;
    kwargs
        .set_item("large_text", large_text)
        .expect("dict insertion with a str key cannot fail");
    // "start" — int(start_time), the interpreter's own int().
    let start = builtin_int(py, start_time)?;
    kwargs
        .set_item("start", start)
        .expect("dict insertion with a str key cannot fail");
    // if duration > 0 — CPython's own >, the end gate, consulted only
    // once the six-key dict is complete.
    let zero: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    if duration.gt(&zero)? {
        // int(start_time + duration) — the addition through CPython's
        // own +, then the interpreter's own int().
        let end = builtin_int(py, &start_time.add(duration)?)?;
        kwargs
            .set_item("end", end)
            .expect("dict insertion with a str key cannot fail");
    }
    Ok(kwargs)
}
