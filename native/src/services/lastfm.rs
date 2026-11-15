//! Native cores for `src/ytm_player/services/lastfm.py` — scrobble
//! due computations (percent + max-seconds policy).

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Last.fm scrobble-threshold gate (mirrors ytm_player/services/lastfm.py)
// ---------------------------------------------------------------------------

/// The lastfm module's `_SCROBBLE_PERCENT`, read through the module
/// attribute — the very constant the fallback's own product runs
/// against, a patched value included.
fn lastfm_scrobble_percent(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.lastfm")?.getattr("_SCROBBLE_PERCENT")
}

/// The lastfm module's `_SCROBBLE_MAX_SECONDS` cap, read through the
/// module attribute — the very constant the fallback's own threshold
/// arms run against, a patched value included.
fn lastfm_scrobble_max_seconds(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.lastfm")?.getattr("_SCROBBLE_MAX_SECONDS")
}

/// The scrobble-threshold gate — `services/lastfm.py`'s
/// `check_scrobble` threshold walk, the decision behind every
/// periodic scrobble poll: whether the playback position has crossed
/// the Last.fm spec's threshold (50% of the track or 4 minutes,
/// whichever is less).
///
/// Mirrors the reference's two arms exactly: the no-duration gate
/// `duration <= 0` through CPython's own `<=` (so a non-numeric
/// duration raises the comparison's own `TypeError`, and the bool
/// spellings — `False <= 0` is True, `True <= 0` is False — stay
/// CPython's) answers the bare cap `_SCROBBLE_MAX_SECONDS` read
/// through the module's own name exactly like the reference's global
/// lookup, a patched cap included; otherwise the threshold is
/// `min(duration * _SCROBBLE_PERCENT, _SCROBBLE_MAX_SECONDS)` — the
/// percent read through the module's own name too, the product and
/// the cap compared through CPython's own `*` and min walk (the cap
/// takes over only when it compares strictly less, so the tie at an
/// eight-minute track answers the product — the reference's first
/// argument — and a non-numeric cap raises the comparison's own
/// `TypeError`). The verdict itself is `position >= threshold`
/// through CPython's own `>=`, the boundary included: a position
/// exactly at the threshold is due. The connection/current-track/
/// scrobbled state, the `duration` read out of the current track and
/// the scrobble submission stay Python-side in the service.
#[pyfunction]
#[pyo3(signature = (position, duration))]
pub(crate) fn lastfm_scrobble_due(
    py: Python<'_>,
    position: &Bound<'_, PyAny>,
    duration: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    // duration <= 0 — CPython's own <=, the reference's own
    // no-duration gate.
    let zero: Bound<'_, PyAny> = 0i64.into_pyobject(py)?.into_any();
    let threshold = if duration.le(&zero)? {
        // threshold = _SCROBBLE_MAX_SECONDS — the cap read through
        // the module's own name, the reference's global lookup.
        lastfm_scrobble_max_seconds(py)?
    } else {
        // duration * _SCROBBLE_PERCENT — CPython's own *, the
        // percent read through the module's own name.
        let percent = lastfm_scrobble_percent(py)?;
        let product = duration.mul(&percent)?;
        // min(product, _SCROBBLE_MAX_SECONDS) — CPython's min walk:
        // the cap takes over only when it compares strictly less
        // than the product, so the tie answers the product (the
        // reference's first argument).
        let cap = lastfm_scrobble_max_seconds(py)?;
        if cap.lt(&product)? {
            cap
        } else {
            product
        }
    };
    // position >= threshold — CPython's own >=, the due verdict.
    position.ge(&threshold)
}
