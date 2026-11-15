//! Native cores for `src/ytm_player/ui/pages/queue.py` — queue page view
//! state and remove/move targets.

use crate::ui::pages::browse::field_get;
use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

// ---------------------------------------------------------------------------
// Queue-page view & mutation engines (mirrors ui/pages/queue.py)
// ---------------------------------------------------------------------------

/// The now-playing header's view model — the pure core shared by
/// `QueuePage._update_current_track` and `QueuePage._refresh_queue`.
///
/// Mirrors the pure-Python reference: a falsy track (the `None` a queue
/// with nothing selected reports, an empty dict never occurring in
/// practice) hides the header and answers `None`; otherwise the dict
/// `{"title_label": ..., "artist": ..., "video_id": ...}` where
/// `title_label` is `f"Now Playing: {title}"` — the title read as
/// `track.get("title", "Unknown")` (absent key `"Unknown"`, a present
/// value whatever it is, `str()`'d by the f-string) — `artist` is
/// `track.get("artist", "Unknown")` raw (the object the Label mounts),
/// and `video_id` is `track.get("video_id", "")` raw, the value
/// `set_playing` receives to light the playing row. Field reads resolve
/// exactly like the reference's `.get` calls (see `field_get`); a
/// `.get`-less track reads as field-less where the reference would
/// raise — the engine stays total. The header mounts and the
/// `set_playing` call stay on the Python side.
#[pyfunction(signature = (track=None))]
pub(crate) fn queue_now_playing<'py>(
    py: Python<'py>,
    track: Option<Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // `if current_track:` — Python truthiness, not an is-None test.
    let track = match track {
        Some(track) if track.is_truthy()? => track,
        _ => return Ok(None),
    };
    let unknown_title = PyString::new(py, "Unknown").into_any();
    let title = field_get(py, &track, "title", Some(unknown_title))?;
    let unknown_artist = PyString::new(py, "Unknown").into_any();
    let artist = field_get(py, &track, "artist", Some(unknown_artist))?;
    let empty_id = PyString::new(py, "").into_any();
    let video_id = field_get(py, &track, "video_id", Some(empty_id))?;

    let out = PyDict::new(py);
    out.set_item("title_label", format!("Now Playing: {}", py_str(&title)?))?;
    out.set_item("artist", artist)?;
    out.set_item("video_id", video_id)?;
    Ok(Some(out))
}

/// The seed-source label's view model — the pure core of
/// `QueuePage._update_queue_source`.
///
/// Mirrors the pure-Python reference: the label stays hidden (`None`)
/// when there are no radio seeds or the `show_queue_source` setting is
/// off; otherwise each seed's title — `seed.get("title", "Unknown")`,
/// absent key `"Unknown"`, a present value whatever it is — feeds both
/// the one-line summary (`", ".join(titles)` for up to three seeds,
/// `"{titles[0]}, {titles[1]} + {n} more"` from the fourth on) and the
/// tooltip's bulleted `"Generated from:"` block. Titles render through
/// `str()`: the tooltip's and the four-plus-summary's f-strings always
/// did, and the up-to-three join — which would raise on a non-str title
/// — reads the same rendering, so the engine stays total where the
/// reference would crash on junk seeds. Returned as `(text, tooltip)`;
/// the Static update and the tooltip assignment stay on the Python side.
#[pyfunction]
#[pyo3(signature = (seeds, show))]
pub(crate) fn queue_source_view(
    py: Python<'_>,
    seeds: Option<Bound<'_, PyList>>,
    show: bool,
) -> PyResult<Option<(String, String)>> {
    // `if seeds and show:` — an empty list hides the label like None does.
    let seeds = match seeds {
        Some(seeds) if show && !seeds.is_empty() => seeds,
        _ => return Ok(None),
    };
    let unknown = PyString::new(py, "Unknown").into_any();
    let mut titles: Vec<String> = Vec::with_capacity(seeds.len());
    for seed in seeds.iter() {
        let title = field_get(py, &seed, "title", Some(unknown.clone()))?;
        titles.push(py_str(&title)?);
    }
    let summary = if titles.len() <= 3 {
        titles.join(", ")
    } else {
        format!("{}, {} + {} more", titles[0], titles[1], titles.len() - 2)
    };
    let text = format!("Generated from: {summary}");
    let tooltip = format!(
        "Generated from:\n{}",
        titles
            .iter()
            .map(|title| format!("  \u{2022} {title}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(Some((text, tooltip)))
}

/// The footer bar's text — the pure core of `QueuePage._update_footer`:
/// `f"Tracks: {count}"` over the queue's length, so `"Tracks: 0"` for
/// the emptied queue reads exactly as the reference's f-string renders
/// it.
#[pyfunction]
pub(crate) fn queue_footer_text(count: i64) -> String {
    format!("Tracks: {count}")
}

/// The queue-position decision behind `QueuePage._remove_selected` —
/// the pure half of the queue page's remove mutation.
///
/// Mirrors the pure-Python reference: the highlighted row's
/// original-order index names a removable queue position only when it
/// is present and `0 <= index < queue_length` (a stale table — one the
/// queue shrank underneath without a refresh — refuses rather than
/// removing the wrong entry); `None` means "no removal". The
/// `queue.remove` call, the page refresh and the cursor re-landing stay
/// on the Python side.
#[pyfunction]
#[pyo3(signature = (selected_index, queue_length))]
pub(crate) fn queue_remove_target(selected_index: Option<i64>, queue_length: i64) -> Option<i64> {
    match selected_index {
        Some(index) if (0..queue_length).contains(&index) => Some(index),
        _ => None,
    }
}

/// The cursor re-landing behind `QueuePage._remove_selected`'s refresh —
/// the pure half of the remove mutation's optimistic UI update.
///
/// Mirrors the pure-Python reference: after the removal and the refresh
/// the row that took the removed entry's place in the displayed order
/// gets the cursor — `min(visible_row, row_count - 1)`, the visible row
/// the cursor sat on before the mutation, clamped into the refreshed
/// table's bounds — and `None` (cursor untouched) when the refreshed
/// table has no rows left. The `move_cursor` call stays on the Python
/// side.
#[pyfunction]
#[pyo3(signature = (visible_row, row_count))]
pub(crate) fn queue_post_remove_cursor(visible_row: i64, row_count: i64) -> Option<i64> {
    if row_count <= 0 {
        return None;
    }
    Some(visible_row.min(row_count - 1))
}

/// The reorder destination behind `QueuePage._move_track` — the pure
/// half of the queue page's move mutation.
///
/// Mirrors the pure-Python reference: no highlighted row (`None`) or a
/// destination equal to the source — including the clamped-at-the-bounds
/// no-ops a too-large count produces — answers `None` (nothing moves);
/// otherwise the destination is `max(0, min(from + direction * count,
/// queue_length - 1))`: the count-multiplied queue positions up or
/// down, clamped to the queue bounds so `15J` near the end simply lands
/// at the bottom. The arithmetic saturates where Python's
/// arbitrary-precision ints cannot overflow — the clamp makes the two
/// agree for every input. The `clear_sort` call, the `queue.move` and
/// the page refresh stay on the Python side.
#[pyfunction]
#[pyo3(signature = (from_index, direction, count, queue_length))]
pub(crate) fn queue_move_target(
    from_index: Option<i64>,
    direction: i64,
    count: i64,
    queue_length: i64,
) -> Option<i64> {
    let from = from_index?;
    // max(0, min(from + direction * count, queue_length - 1)).
    let destination = from
        .saturating_add(direction.saturating_mul(count))
        .min(queue_length - 1)
        .max(0);
    if destination == from {
        return None;
    }
    Some(destination)
}
