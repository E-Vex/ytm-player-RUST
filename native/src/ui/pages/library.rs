//! Native cores for `src/ytm_player/ui/pages/library.py` — the library
//! view, shuffle-lock button/toggle and cursor restore.

use crate::ui::pages::browse::field_get;
use crate::ui::pages::context::normalized_tracks_and_len;
use crate::ui::pages::context::numeric_gt;
use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyList, PyString};

// ---------------------------------------------------------------------------
// Library-page view & cursor engines (mirror ytm_player/ui/pages/library.py)
// ---------------------------------------------------------------------------

/// The library page's load view model — the pure core of
/// `LibraryPage.load_playlist`'s post-fetch walk, from
/// `ui/pages/library.py`.
///
/// Returned keys:
///
/// - `title` — `data.get("title", "Unknown Playlist")`, raw.
/// - `owner` — `author.get("name", "Unknown")` for a dict `author`,
///   `str(author)` otherwise.
/// - `privacy` — `(data.get("privacy") or "").strip()`.
/// - `year` — `data.get("year")`, raw.
/// - `tracks` — `normalize_tracks(data.get("tracks", []))`.
/// - `track_count` — the normalized count.
/// - `total_count` — `data.get("trackCount") or track_count`, the raw
///   stored value kept (each read short-circuited by every earlier
///   falsy one).
/// - `subtitle` — `build_playlist_subtitle(owner, privacy, year,
///   track_count)` — read through the formatting module's own name at
///   call time, so whichever backend is bound steers the assembly
///   exactly like the fallback's own call — plus the
///   `" (loading <total> total…)"` suffix when the total is
///   numerically above the normalized count (a stored count Python's
///   `>` would reject reads as uncomparable and skips the suffix, the
///   engine staying total).
/// - `unavailable` — `len(raw) - len(normalized)`; `None` when the raw
///   value is not a list.
/// - `fetch_remaining` — `total_count > len(raw)`, the gate behind the
///   background fetch of the playlist's tail.
/// - `raw_count` — `len(raw)`, the `already_have` the tail fetch
///   resumes from; 0 whenever the raw value is not a list.
#[pyfunction]
pub(crate) fn library_playlist_view<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let unknown_playlist = PyString::new(py, "Unknown Playlist").into_any();
    let title = field_get(py, data, "title", Some(unknown_playlist))?;

    // owner = author.get("name", "Unknown") if isinstance(author, dict)
    //         else str(author)
    let empty_author = PyDict::new(py).into_any();
    let author = field_get(py, data, "author", Some(empty_author))?;
    let owner = if author.is_instance_of::<PyDict>() {
        let unknown_name = PyString::new(py, "Unknown").into_any();
        field_get(py, &author, "name", Some(unknown_name))?
    } else {
        author.str()?.into_any()
    };

    // privacy = (data.get("privacy") or "").strip()
    let empty = PyString::new(py, "").into_any();
    let raw_privacy = field_get(py, data, "privacy", None)?;
    let privacy = if raw_privacy.is_truthy()? {
        raw_privacy
    } else {
        empty
    }
    .call_method0("strip")?;

    // year = data.get("year")
    let year = field_get(py, data, "year", None)?;

    // raw_tracks = data.get("tracks", []) — normalize, then count.
    let empty_tracks = PyList::empty(py).into_any();
    let raw_tracks = field_get(py, data, "tracks", Some(empty_tracks))?;
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    let track_count = tracks.len() as i64;

    // total_count = data.get("trackCount") or track_count — the raw
    // stored value kept (the suffix interpolates it, exactly like the
    // reference's f-string).
    let raw_total = field_get(py, data, "trackCount", None)?;
    let total_count = if raw_total.is_truthy()? {
        raw_total
    } else {
        track_count.into_pyobject(py)?.into_any()
    };

    // subtitle = build_playlist_subtitle(owner, privacy, year, track_count)
    // — through the formatting module's own name, so the shared engine
    // steers the assembly whichever backend is bound.
    let subtitle_fn = py
        .import("ytm_player.utils.formatting")?
        .getattr("build_playlist_subtitle")?;
    let mut subtitle =
        py_str(&subtitle_fn.call1((owner.clone(), privacy.clone(), year.clone(), track_count))?)?;
    // if total_count > track_count: subtitle += " (loading N total…)"
    if numeric_gt(&total_count, track_count) == Some(true) {
        subtitle.push_str(&format!(
            " (loading {} total\u{2026})",
            py_str(&total_count)?
        ));
    }

    // fetch_remaining = total_count > len(raw_tracks)
    let fetch_remaining = match raw_len {
        Some(len) => numeric_gt(&total_count, len) == Some(true),
        None => false,
    };

    let out = PyDict::new(py);
    out.set_item("title", &title)?;
    out.set_item("owner", &owner)?;
    out.set_item("privacy", &privacy)?;
    out.set_item("year", &year)?;
    out.set_item("tracks", &tracks)?;
    out.set_item("track_count", track_count)?;
    out.set_item("total_count", &total_count)?;
    out.set_item("subtitle", PyString::new(py, &subtitle))?;
    match raw_len {
        Some(len) => out.set_item("unavailable", len - track_count)?,
        None => out.set_item("unavailable", pyo3::types::PyNone::get(py))?,
    }
    out.set_item("fetch_remaining", fetch_remaining)?;
    out.set_item("raw_count", raw_len.unwrap_or(0))?;
    Ok(out)
}

/// The Shuffle-lock header button's label/classes pair — the shared
/// assembly behind `ui/pages/library.py`'s load header, its
/// edit-refresh header and the toggle's in-place button update.
///
/// Mirrors the reference exactly: the `bool` of *locked* picks the
/// pair — `"Shuffle lock: ON"` with the `"locked"` class when on,
/// `"Shuffle lock: off"` with no class when off. The button mounts and
/// updates stay Python-side in the page.
#[pyfunction]
pub(crate) fn shuffle_lock_button<'py>(
    py: Python<'py>,
    locked: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let on = locked.is_truthy()?;
    let out = PyDict::new(py);
    out.set_item(
        "label",
        if on {
            "Shuffle lock: ON"
        } else {
            "Shuffle lock: off"
        },
    )
    .expect("dict insertion with a str key cannot fail");
    out.set_item("classes", if on { "locked" } else { "" })
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The lock-toggle plan — `ui/pages/library.py`'s
/// `_toggle_shuffle_lock` decision core: the new lock state and the
/// force-shuffle gate.
///
/// Mirrors the reference exactly: `new_locked` is the flip of the
/// stored pref's `bool`, and `force_shuffle` is the reference's own
/// `new_locked and current_context_id == playlist_id` — the `==`
/// through CPython's own protocol, consulted only when the lock just
/// turned on (an unlock never forces the mode). The pref write, the
/// button update, the shuffle toggle and the refreshes stay
/// Python-side in the page.
#[pyfunction]
#[pyo3(signature = (prefs_value, current_context_id, playlist_id))]
pub(crate) fn shuffle_lock_toggle_plan<'py>(
    py: Python<'py>,
    prefs_value: &Bound<'py, PyAny>,
    current_context_id: &Bound<'py, PyAny>,
    playlist_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let new_locked = !prefs_value.is_truthy()?;
    let force_shuffle = new_locked && current_context_id.eq(playlist_id)?;
    let out = PyDict::new(py);
    out.set_item("new_locked", new_locked)?;
    out.set_item("force_shuffle", force_shuffle)?;
    Ok(out)
}

/// The playing-track id resolution — `ui/pages/library.py`'s
/// `_restore_track_cursor` fallback chain: which video id names the
/// track that is currently playing.
///
/// Mirrors the reference exactly, every read through CPython's own
/// protocol: the player's current track read first — a dict contributes
/// its `video_id` (the gate `playing_id and isinstance(playing_id,
/// dict)`, a truthy non-dict riding home as itself) — and only when
/// that answers nothing falsy does the queue's current track get its
/// `video_id` read (the gate `not playing_id and queue_track`). The
/// player/queue attribute reads stay Python-side in the page.
#[pyfunction]
#[pyo3(signature = (player_track, queue_track))]
pub(crate) fn playing_video_id<'py>(
    player_track: &Bound<'py, PyAny>,
    queue_track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut playing_id = player_track.clone();
    if playing_id.is_truthy()? && playing_id.is_instance_of::<PyDict>() {
        playing_id = playing_id.call_method1("get", ("video_id",))?;
    }
    if !playing_id.is_truthy()? && queue_track.is_truthy()? {
        playing_id = queue_track.call_method1("get", ("video_id",))?;
    }
    Ok(playing_id)
}

/// The cursor-restore row — `ui/pages/library.py`'s
/// `_restore_track_cursor` decision core: the row the library table's
/// cursor lands on after a load.
///
/// Mirrors the reference exactly, every comparison through CPython's
/// own protocol: a zero `row_count` answers `None` (the reference's
/// own early return, the cursor untouched); the saved nav row wins
/// when it is not `None` and `0 <= row < row_count` (the chained
/// comparison evaluated left to right); otherwise the first track
/// whose `t.get("video_id") == playing_id` answers its index; nothing
/// resolving answers `None`. The `move_cursor` call stays Python-side
/// in the page.
#[pyfunction]
#[pyo3(signature = (row, row_count, playing_id, tracks))]
pub(crate) fn cursor_restore_row<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
    row_count: &Bound<'py, PyAny>,
    playing_id: &Bound<'py, PyAny>,
    tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // if row_count == 0: return — the reference's own early return.
    if row_count.eq(0i64)? {
        return Ok(py.None().into_bound(py));
    }
    // if row is not None and 0 <= row < row_count: return row
    if !row.is_none() {
        let ge = row.rich_compare(0i64, CompareOp::Ge)?;
        if ge.is_truthy()? {
            let lt = row.rich_compare(row_count, CompareOp::Lt)?;
            if lt.is_truthy()? {
                return Ok(row.clone());
            }
        }
    }
    // if playing_id: the first track whose video_id matches.
    if playing_id.is_truthy()? {
        for (i, track) in tracks.try_iter()?.enumerate() {
            let track = track?;
            let vid = track.call_method1("get", ("video_id",))?;
            if vid.eq(playing_id)? {
                return Ok(i.into_pyobject(py)?.into_any());
            }
        }
    }
    Ok(py.None().into_bound(py))
}
