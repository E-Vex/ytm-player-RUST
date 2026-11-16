//! Native cores for `src/ytm_player/ui/pages/liked_songs.py` — the
//! liked-songs page's empty-state copy, footer text, cursor gates and
//! the Start-Radio seed draw (the five-track sample
//! `ui/pages/recently_played.py`'s own Start Radio shares).

use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::PyDict;

// ---------------------------------------------------------------------------
// Liked-songs page engines (mirror ytm_player/ui/pages/liked_songs.py)
// ---------------------------------------------------------------------------

/// The empty-state copy behind `LikedSongsPage._display_tracks`'s
/// no-tracks branch — the message the page's loading label shows when
/// the liked-music playlist rendered no rows, from
/// `ui/pages/liked_songs.py`.
///
/// Mirrors the reference exactly: the `bool` of *load_failed* — the
/// flag `_load_liked_songs` sets when the fetch itself failed (the
/// service answers `None` on failure, never raises, so an empty
/// playlist and a failed fetch render different messages) — picks the
/// failure copy that points at the log file, over the genuine
/// empty-state "No liked songs found.". The table/label visibility
/// flips and the mount stay Python-side in the page.
#[pyfunction]
#[pyo3(signature = (load_failed))]
pub(crate) fn liked_empty_state(load_failed: &Bound<'_, PyAny>) -> PyResult<String> {
    let failed = load_failed.is_truthy()?;
    Ok(if failed {
        "Couldn't load liked songs. Check the log at ~/.config/ytm-player/logs/ytm.log for details."
    } else {
        "No liked songs found."
    }
    .to_string())
}

/// The footer text behind `LikedSongsPage._update_footer` — the pure
/// core of the page's bottom-bar line, from `ui/pages/liked_songs.py`.
///
/// Mirrors the reference exactly: the count interpolates through
/// CPython's own `str` — the conversion the reference's f-string
/// applies to `table.track_count` — and the truthiness of
/// *loading_more* appends the `" (loading more…)"` suffix the first
/// batch shows while the background top-up runs. The widget queries
/// and the footer update stay Python-side in the page.
#[pyfunction]
#[pyo3(signature = (total, loading_more))]
pub(crate) fn liked_footer_text<'py>(
    total: &Bound<'py, PyAny>,
    loading_more: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // f"{total} liked songs" — str(total) through CPython's own
    // conversion, exactly the f-string's.
    let mut text = format!("{} liked songs", py_str(total)?);
    if loading_more.is_truthy()? {
        text.push_str(" (loading more\u{2026})");
    }
    Ok(text)
}

/// The restore-cursor bounds gate behind
/// `LikedSongsPage._display_tracks`' cursor restore — the row the
/// liked table's cursor lands on after a load, from
/// `ui/pages/liked_songs.py`.
///
/// Mirrors the reference exactly, every comparison through CPython's
/// own protocol: the saved nav row answers itself when it is not
/// `None` and `0 <= row < row_count` (the chained comparison
/// evaluated left to right — the saved row a `float` or a `bool`
/// compares like Python itself does); anything else answers `None`,
/// the cursor staying where the load left it. The saved-row consume
/// and the `move_cursor` call stay Python-side in the page.
#[pyfunction]
#[pyo3(signature = (row, row_count))]
pub(crate) fn liked_cursor_restore_row<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
    row_count: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
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
    Ok(py.None().into_bound(py))
}

/// The nav-state gate behind `LikedSongsPage.get_nav_state` — the
/// state the page preserves when navigating away, from
/// `ui/pages/liked_songs.py`.
///
/// Mirrors the reference exactly: the row is remembered only when it
/// is not `None` and `row > 0` — a cursor still resting on the first
/// row (or not laid out yet) needs no restore — the comparison through
/// CPython's own protocol, and the remembered row riding home as the
/// very object it is. The `query_one` and its broad catch stay
/// Python-side in the page.
#[pyfunction]
#[pyo3(signature = (cursor_row))]
pub(crate) fn liked_nav_state<'py>(
    py: Python<'py>,
    cursor_row: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // if cursor_row is not None and cursor_row > 0:
    //     state["cursor_row"] = cursor_row
    if !cursor_row.is_none() {
        let gt = cursor_row.rich_compare(0i64, CompareOp::Gt)?;
        if gt.is_truthy()? {
            out.set_item("cursor_row", cursor_row)?;
        }
    }
    Ok(out)
}

/// The Start-Radio seed draw — the pure core of the sampling step both
/// `LikedSongsPage._start_radio` (`ui/pages/liked_songs.py`) and
/// `RecentlyPlayedPage._start_radio` (`ui/pages/recently_played.py`)
/// share: the five-track sample a liked/recent radio is seeded from.
///
/// Mirrors the reference exactly: the draw size `min(5, len(tracks))`
/// — the length through CPython's own sequence protocol, so a
/// lengthless pool raises the reference's own `len` `TypeError`,
/// capped at five and at the pool's own length — then
/// `random.sample(tracks, k)` routed through the Python `random`
/// module, so `random.seed(...)` before the call and a monkeypatched
/// `random.sample` steer the native engine identically to the
/// fallback. The answer is the sample the module's own `sample`
/// produced, the tracks passing through as the objects they are; the
/// empty-tracks guard and the radio fetch stay Python-side in the
/// pages.
#[pyfunction]
#[pyo3(signature = (tracks))]
pub(crate) fn radio_seed_sample<'py>(
    py: Python<'py>,
    tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let random = PyModule::import(py, "random")?;
    let len = tracks.len()?;
    let k = len.min(5);
    random.call_method1("sample", (tracks.clone(), k))
}
