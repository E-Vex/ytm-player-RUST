//! Native cores for `src/ytm_player/app/_session.py` — session state
//! coercion (volume clamps), the track-index lookup and the
//! resume-staging plan.

use crate::ui::pages::browse::field_get;
use crate::utils::formatting::normalize_tracks;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString};

// ---------------------------------------------------------------------------
// Session-state validation engine (mirrors ytm_player/app/_session.py)
// ---------------------------------------------------------------------------

/// `int(max(0, min(100, v)))` for a value already known to be a non-bool
/// int. Ints beyond i64 still clamp by sign, exactly like CPython's
/// comparison-based `min`/`max` (min(100, huge) == 100, max(0, -huge) == 0).
fn clamp_volume_int(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(i) = v.extract::<i64>() {
        return Ok(i.clamp(0, 100));
    }
    let negative = v
        .str()?
        .to_str()
        .map(|s| s.starts_with('-'))
        .unwrap_or(false);
    Ok(if negative { 0 } else { 100 })
}

/// The same clamp for a float, transcribing Python's comparison order so
/// NaN maps to 100 exactly like `max(0, min(100, nan))` does
/// (`min(a, b)` keeps `a` unless `b < a`; NaN comparisons are always
/// false, so `min(100, nan)` is 100 and `min(100, -inf)` is -inf).
fn clamp_volume_float(f: f64) -> i64 {
    let m = if f < 100.0 { f } else { 100.0 };
    let m = if m > 0.0 { m } else { 0.0 };
    m.trunc() as i64
}

/// Apply the Python volume gate: a non-bool int/float clamps into
/// 0..=100; anything else fails (the caller substitutes the raw default).
/// Exact type checks suffice because `json.loads` never produces
/// subclasses.
fn try_clamp_volume(v: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    // bool is checked first: it passes PyLong_Check but fails the gate.
    if v.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if v.downcast::<PyFloat>().is_ok() {
        let f: f64 = v.extract()?;
        return Ok(Some(clamp_volume_float(f)));
    }
    if v.is_instance_of::<PyInt>() {
        return clamp_volume_int(v).map(Some);
    }
    Ok(None)
}

/// `isinstance(x, int)` as an i64 — bools count (they are ints in
/// Python, so a stored `true` queue_index validates as index 1, matching
/// the fallback's `isinstance(saved_index, int)` check).
fn python_int_value(x: &Bound<'_, PyAny>) -> Option<i64> {
    if let Ok(b) = x.downcast::<PyBool>() {
        return Some(i64::from(b.is_true()));
    }
    x.extract::<i64>().ok()
}

/// Validate and coerce a parsed session.json object in one pass.
///
/// Transcribes the pure validation half of
/// `SessionMixin._restore_session_state` from `app/_session.py`: every
/// isinstance gate, clamp and fallback applied to the raw parsed state
/// before any app object is touched. The mixin reads the returned dict
/// and only *applies* the values — all decisions are made here,
/// identically on either backend.
///
/// Returned keys:
///
/// - `volume` — clamped int, or the raw *default_volume* when the stored
///   value's type is invalid (the fallback returns the raw default
///   unclamped, so an absent key with a non-numeric default round-trips).
/// - `repeat` — "off" / "all" / "one"; anything else degrades to "off".
/// - `shuffle` — truthiness of the stored `shuffle`.
/// - `queue_tracks` — `normalize_tracks()` of the stored list when it is
///   a truthy list, else `[]`.
/// - `queue_index` — the stored index when it is an int (bool included)
///   inside the normalized range, else None.
/// - `queue_context_id` — non-empty str or None.
/// - `sidebar_per_page` — the stored dict itself (no copy) when truthy,
///   else None.
/// - `first_run_hint_shown` / `mpris_hint_shown` — truthiness.
/// - `has_transliteration` + `transliteration_enabled` — key presence
///   and the RAW stored value (the fallback assigns it unconverted).
/// - `resume` — the stored resume dict (the original object, not a copy:
///   the mixin copies for `_loaded_resume` and reads position /
///   playlist_id from it) when it is a dict with a truthy `video_id`,
///   else None.
#[pyfunction]
pub(crate) fn coerce_session_state<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    default_volume: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // -- volume ------------------------------------------------------------
    // `state.get("volume", default)` then the isinstance gate: a non-bool
    // int/float clamps into 0..=100, anything else (including an absent
    // key with a non-numeric default) yields the raw default.
    let volume_clamped: Option<i64> = match state.get_item("volume")? {
        Some(ref v) => try_clamp_volume(v)?,
        None => try_clamp_volume(default_volume)?,
    };

    // -- repeat mode ---------------------------------------------------------
    let repeat: String = match state.get_item("repeat")? {
        Some(ref v) => match v.downcast::<PyString>() {
            Ok(s) => {
                let t = s.to_str()?;
                if t == "off" || t == "all" || t == "one" {
                    t.to_string()
                } else {
                    "off".to_string()
                }
            }
            Err(_) => "off".to_string(),
        },
        None => "off".to_string(),
    };

    // -- queue -----------------------------------------------------------------
    let (queue_tracks, queue_index): (Bound<'py, PyList>, Option<i64>) =
        match state.get_item("queue_tracks")? {
            Some(ref v) if v.is_truthy()? && v.is_instance_of::<PyList>() => {
                let normalized = normalize_tracks(py, v.downcast::<PyList>().unwrap())?;
                let len = normalized.len() as i64;
                // `state.get("queue_index", 0)` gated by
                // `isinstance(saved_index, int) and 0 <= saved_index < len`.
                let idx = match state.get_item("queue_index")? {
                    Some(ref x) => python_int_value(x).filter(|i| *i >= 0 && *i < len),
                    None => {
                        if len > 0 {
                            Some(0)
                        } else {
                            None
                        }
                    }
                };
                (normalized, idx)
            }
            _ => (PyList::empty(py), None),
        };

    // -- shuffle -----------------------------------------------------------------
    let shuffle: bool = match state.get_item("shuffle")? {
        Some(v) => v.is_truthy()?,
        None => false,
    };

    // -- collection context ----------------------------------------------------
    let queue_context_id: Option<String> = match state.get_item("queue_context_id")? {
        Some(ref v) => match v.downcast::<PyString>() {
            Ok(s) => {
                let t = s.to_str()?;
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                }
            }
            Err(_) => None,
        },
        None => None,
    };

    // -- sidebar state ----------------------------------------------------------
    let sidebar_per_page: Option<Bound<'py, PyAny>> = match state.get_item("sidebar_per_page")? {
        Some(ref v) if v.is_truthy()? && v.is_instance_of::<PyDict>() => Some(v.clone()),
        _ => None,
    };

    // -- hint flags ------------------------------------------------------------
    let first_run_hint_shown: bool = match state.get_item("first_run_hint_shown")? {
        Some(v) => v.is_truthy()?,
        None => false,
    };
    let mpris_hint_shown: bool = match state.get_item("mpris_hint_shown")? {
        Some(v) => v.is_truthy()?,
        None => false,
    };

    // -- transliteration toggle (raw value, assigned unconverted) ---------------
    let has_transliteration = state.contains("transliteration_enabled")?;
    let transliteration_enabled: Option<Bound<'py, PyAny>> =
        state.get_item("transliteration_enabled")?;

    // -- resume staging -----------------------------------------------------------
    // The original dict (not a copy): the mixin copies it for
    // `_loaded_resume` and reads position / playlist_id from it directly.
    let resume: Option<Bound<'py, PyAny>> = match state.get_item("resume")? {
        Some(ref r) if r.is_instance_of::<PyDict>() => {
            let has_video_id = match r.downcast::<PyDict>().unwrap().get_item("video_id")? {
                Some(v) => v.is_truthy()?,
                None => false,
            };
            if has_video_id {
                Some(r.clone())
            } else {
                None
            }
        }
        _ => None,
    };

    // -- assemble ------------------------------------------------------------
    let out = PyDict::new(py);
    match volume_clamped {
        Some(i) => {
            out.set_item("volume", i)?;
        }
        None => {
            out.set_item("volume", default_volume.clone())?;
        }
    }
    out.set_item("repeat", repeat.as_str())?;
    out.set_item("shuffle", shuffle)?;
    out.set_item("queue_tracks", queue_tracks)?;
    match queue_index {
        Some(i) => {
            out.set_item("queue_index", i)?;
        }
        None => {
            out.set_item("queue_index", pyo3::types::PyNone::get(py))?;
        }
    }
    match &queue_context_id {
        Some(s) => {
            out.set_item("queue_context_id", s.as_str())?;
        }
        None => {
            out.set_item("queue_context_id", pyo3::types::PyNone::get(py))?;
        }
    }
    match &sidebar_per_page {
        Some(d) => {
            out.set_item("sidebar_per_page", d.clone())?;
        }
        None => {
            out.set_item("sidebar_per_page", pyo3::types::PyNone::get(py))?;
        }
    }
    out.set_item("first_run_hint_shown", first_run_hint_shown)?;
    out.set_item("mpris_hint_shown", mpris_hint_shown)?;
    out.set_item("has_transliteration", has_transliteration)?;
    match transliteration_enabled {
        Some(v) => {
            out.set_item("transliteration_enabled", v)?;
        }
        None => {
            out.set_item("transliteration_enabled", pyo3::types::PyNone::get(py))?;
        }
    }
    match resume {
        Some(r) => {
            out.set_item("resume", r)?;
        }
        None => {
            out.set_item("resume", pyo3::types::PyNone::get(py))?;
        }
    }
    Ok(out)
}

/// Linear scan for the first track whose `video_id` equals *video_id* —
/// the resume-point lookup from `SessionMixin._restore_session_state`,
/// run over the restored (possibly shuffled) queue. Comparison uses
/// Python `==` semantics (the stored resume `video_id` can be any truthy
/// JSON value), and non-dict entries are skipped where the Python
/// fallback would raise — queue entries are always normalized dicts, and
/// the engine staying total keeps one malformed entry from taking the
/// restore down with it.
#[pyfunction]
pub(crate) fn find_track_index(
    tracks: &Bound<'_, PyAny>,
    video_id: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
    for (i, item) in tracks.try_iter()?.enumerate() {
        let item = item?;
        if let Ok(dict) = item.downcast::<PyDict>() {
            if let Some(vid) = dict.get_item("video_id")? {
                if vid.eq(video_id)? {
                    return Ok(Some(i));
                }
            }
        }
    }
    Ok(None)
}

/// The resume-staging plan — the pure core of the resume arm
/// `SessionMixin._restore_session_state` runs after the queue is
/// restored: which track comes back, from which playlist, at which
/// position.
///
/// A *resume* that is `None`, or whose `video_id` reads falsy (the
/// `.get` protocol, `""` substituting only an absent key), stages
/// nothing — the answer is `None`. Otherwise the plan carries the
/// `video_id` and the `playlist_id` exactly as read (raw objects: the
/// mixin assigns the playlist id straight onto
/// `_active_library_playlist_id`, and the `find_track_index` scan
/// compares the very video_id object), plus the `position` through
/// CPython's own `float()` — `float("1.5")` parses, `float(True)` is
/// 1.0, and anything the call rejects as a `TypeError` or `ValueError`
/// (None, lists, non-numeric strings — the shapes the reference's own
/// `except (TypeError, ValueError)` arm swallows) reads as 0.0;
/// anything else (a huge int's `OverflowError`, say) escapes to the
/// same caller the reference's would.
///
/// The plan reads the playlist id and position up front, where the
/// reference reads them after the queue jump — the only observable
/// difference is which side effects, if any, preceded a raising read on
/// a dict subclass steering its own `get`; through the real flow the
/// dict is `json.loads`'s own plain product and every read lands. The
/// `resume_on_launch` settings gate stays in the mixin (its early
/// return also marks the session restored).
#[pyfunction]
pub(crate) fn resume_staging_plan<'py>(
    py: Python<'py>,
    resume: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if resume.is_none() {
        return Ok(None);
    }
    // resume.get("video_id", "") — the .get protocol, then the
    // truthiness gate.
    let empty = PyString::new(py, "").into_any();
    let video_id = field_get(py, resume, "video_id", Some(empty))?;
    if !video_id.is_truthy()? {
        return Ok(None);
    }
    // resume.get("playlist_id") — no default, a missing key reads None.
    let playlist_id = field_get(py, resume, "playlist_id", None)?;
    // float(resume.get("position", 0)) — the read and the conversion
    // both inside the reference's except (TypeError, ValueError) arm;
    // the float is CPython's own, fetched from builtins at call time
    // (the json.rs str() precedent), so string positions parse and
    // exotic __float__ implementations run exactly as they would in
    // the reference.
    let zero = 0i64.into_pyobject(py)?.into_any();
    let raw_position = field_get(py, resume, "position", Some(zero))?;
    let position: f64 = match py
        .import("builtins")?
        .getattr("float")?
        .call1((raw_position,))
    {
        Ok(value) => value.extract::<f64>()?,
        Err(err)
            if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                || err.is_instance_of::<pyo3::exceptions::PyValueError>(py) =>
        {
            0.0
        }
        Err(err) => return Err(err),
    };
    let out = PyDict::new(py);
    out.set_item("video_id", video_id)?;
    out.set_item("playlist_id", playlist_id)?;
    out.set_item("position", position)?;
    Ok(Some(out))
}
