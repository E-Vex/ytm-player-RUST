//! Native cores for `src/ytm_player/app/_playback.py` — the playback
//! failure policy and the history-report poll step.

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Playback coordination engines (mirror ytm_player/app/_playback.py)
// ---------------------------------------------------------------------------

/// The playback module's own `_MAX_CONSECUTIVE_FAILURES` cap, read
/// through the module attribute — the very constant the fallback's own
/// threshold arm runs against, a patched value included.
fn playback_max_consecutive_failures(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.app._playback")?.getattr("_MAX_CONSECUTIVE_FAILURES")
}

/// The playback-failure policy — `app/_playback.py`'s
/// `_handle_play_failure` decision core, the verdict every failed play
/// runs through once the consecutive-failure counter has been bumped.
///
/// Mirrors the reference's walk exactly, every comparison through
/// CPython's own protocol: the repeat gate `repeat_mode == "one"`
/// first — a failing track in repeat-one must not skip itself forever,
/// so the verdict is `stop` and nothing else is consulted. Then the
/// threshold `consecutive_failures < _MAX_CONSECUTIVE_FAILURES` — the
/// cap read through the playback module's own name at call time (a
/// patched cap steers both backends identically), the comparison
/// CPython's own `<` so a non-numeric count raises the comparison's
/// own `TypeError`. Under the cap, the queue's own bound `next_track`
/// method is pulled — called exactly here, exactly once, the
/// reference's own advance-and-read — and a truthy answer is the
/// `advance` verdict carrying the very track object the queue handed
/// back (the reference's `if next_track:` truthiness, never identity),
/// a falsy one the `end_of_queue` verdict. At or over the cap the
/// verdict is `escalate`, with `reset_resolver` true only when
/// `failure_kind` is not `None` (the identity check, short-circuited
/// before the resolver is touched) and *resolver* is truthy through
/// CPython's own `bool` protocol — the systemic failures get a fresh
/// yt-dlp instance, the no-video-id path does not. The counter bump
/// and reset, the queue state, the resolver reset itself, the toasts
/// and the deferred advance stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (repeat_mode, consecutive_failures, failure_kind, resolver, next_track))]
pub(crate) fn playback_failure_policy<'py>(
    py: Python<'py>,
    repeat_mode: &Bound<'py, PyAny>,
    consecutive_failures: &Bound<'py, PyAny>,
    failure_kind: &Bound<'py, PyAny>,
    resolver: &Bound<'py, PyAny>,
    next_track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // repeat_mode == "one" — CPython's own ==, the reference's repeat gate.
    if repeat_mode.eq("one")? {
        let out = PyDict::new(py);
        out.set_item("action", "stop")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // consecutive_failures < _MAX_CONSECUTIVE_FAILURES — CPython's own
    // <, the cap read through the playback module's own name.
    let cap = playback_max_consecutive_failures(py)?;
    if consecutive_failures.lt(&cap)? {
        // next_track() — the queue's own bound method, pulled exactly
        // here and exactly once, the reference's advance-and-read.
        let track = next_track.call0()?;
        if track.is_truthy()? {
            let out = PyDict::new(py);
            out.set_item("action", "advance")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("track", track)
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
        let out = PyDict::new(py);
        out.set_item("action", "end_of_queue")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // failure_kind is not None and resolver — the identity check
    // short-circuited before the resolver's truthiness, the reference's
    // own gate order.
    let reset_resolver = !failure_kind.is_none() && resolver.is_truthy()?;
    let out = PyDict::new(py);
    out.set_item("action", "escalate")
        .expect("dict insertion with a str key cannot fail");
    out.set_item("reset_resolver", reset_resolver)
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The history-report poll step — the shared decision core behind
/// `app/_playback.py`'s two report timer chains,
/// `_report_local_play` and `_report_ytm_play`, consulted on every
/// timer tick: whether the play is reportable yet, whether the chain
/// keeps polling, and what idle count the next poll carries.
///
/// Mirrors the reference's walk exactly, every comparison through
/// CPython's own protocol: the threshold gate `listened <=
/// min_listen_seconds` — the boundary kept on the poll side, so a
/// listen exactly at the configured minimum is not yet reportable, and
/// a non-numeric listen raises the comparison's own `TypeError`.
/// Above the threshold the verdict is `report`. At or under it, a
/// live player (`current_track is not None` — the identity check, not
/// truthiness, so only the genuine None clear counts as idle) answers
/// `poll` with the idle count reset to zero; a None current track
/// walks the two-poll stability window — `idle_polls + 1 >= 2`, the
/// bump through CPython's own `+` and the window edge through
/// CPython's own `>=` — answering `stop` once two consecutive polls
/// found the player idle (the window absorbing the transient
/// current_track clear a duplicate mpv end-file causes on natural
/// advance), and `poll` carrying the bumped idle count otherwise (the
/// bumped object itself, the reference's own `idle_polls + 1` result).
/// The position reads, the min-listen resolution, the timer re-arms
/// and the report workers stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (listened, min_listen_seconds, current_track, idle_polls))]
pub(crate) fn playback_history_report_step<'py>(
    py: Python<'py>,
    listened: &Bound<'py, PyAny>,
    min_listen_seconds: &Bound<'py, PyAny>,
    current_track: &Bound<'py, PyAny>,
    idle_polls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // listened <= min_listen_seconds — CPython's own <=, the boundary
    // kept on the poll side.
    if listened.le(min_listen_seconds)? {
        // current_track is None — identity, not truthiness.
        if current_track.is_none() {
            // idle_polls + 1 >= 2 — the bump through CPython's own +,
            // the window edge through CPython's own >=.
            let one: Bound<'py, PyAny> = 1i64.into_pyobject(py)?.into_any();
            let bumped = idle_polls.add(&one)?;
            let two: Bound<'py, PyAny> = 2i64.into_pyobject(py)?.into_any();
            if bumped.ge(&two)? {
                let out = PyDict::new(py);
                out.set_item("action", "stop")
                    .expect("dict insertion with a str key cannot fail");
                return Ok(out);
            }
            let out = PyDict::new(py);
            out.set_item("action", "poll")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("idle_polls", bumped)
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
        // A live track resets the idle count — the reference's else arm.
        let out = PyDict::new(py);
        out.set_item("action", "poll")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("idle_polls", 0i64)
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    let out = PyDict::new(py);
    out.set_item("action", "report")
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}
