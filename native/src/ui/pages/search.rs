//! Native cores for `src/ytm_player/ui/pages/search.py` — result
//! formatting, panel labels and categorisation.

use crate::ui::pages::browse::field_get;
use crate::ui::pages::browse::first_truthy_get;
use crate::ui::sidebars::eq_any_str;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::truncate_str;
use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

/// Build the human-readable label for one search-result item.
///
/// Mirrors `SearchResultPanel._format_item` from `ui/pages/search.py`:
/// the type tag — `resultType`, falling back to `category` (the nested
/// default is evaluated before the outer lookup, exactly like the
/// reference's eager argument evaluation) — picks the branch; the title
/// is the first truthy of `title`/`name`/`artist` (`"Unknown"` when all
/// are falsy or missing); album rows append `" - <artist>"` (through the
/// shared, already-ported `extract_artist`) and `" (<year>)"` for a
/// truthy `year`; artist rows append `"  [<subscribers>]"`; playlist
/// rows append `" by <author>"` and `" (<count> tracks)"` (`itemCount`
/// falling back to `count`, again with the eager default); everything
/// else is the bare title. Every branch truncates to 60 characters with
/// the shared `truncate` semantics.
///
/// Field reads resolve exactly like the reference's `.get` calls and
/// truthiness is Python's, so dict subclasses and exotic-but-truthy
/// values behave like the reference (see `field_get`). Where the reference
/// would raise — a `.get`-less item, or a non-string bare title hitting
/// `len()` — the engine stays total: the item reads as field-less and the
/// bare title is stringified before truncation, exactly like the other
/// branches' f-string interpolation already does.
#[pyfunction]
pub(crate) fn format_search_result(py: Python, item: &Bound<'_, PyAny>) -> PyResult<String> {
    // result_type = item.get("resultType", item.get("category", ""))
    let empty = PyString::new(py, "").into_any();
    let category = field_get(py, item, "category", Some(empty))?;
    let result_type = field_get(py, item, "resultType", Some(category))?;

    // title = item.get("title") or item.get("name") or item.get("artist")
    //         or "Unknown"
    let title = match first_truthy_get(py, item, &["title", "name", "artist"])? {
        Some(value) => value,
        None => PyString::new(py, "Unknown").into_any(),
    };

    if eq_any_str(&result_type, &["album", "albums"])? {
        // extract_artist(item) — the shared native engine; it reads
        // C-level dict fields, so a non-dict mapping that classified
        // itself as an album through its `get` degrades to Unknown.
        let artist = match item.downcast::<PyDict>() {
            Ok(dict) => extract_artist(py, dict),
            Err(_) => "Unknown".to_string(),
        };
        let year = field_get(py, item, "year", Some(PyString::new(py, "").into_any()))?;
        let mut label = py_str(&title)?;
        if !artist.is_empty() {
            label.push_str(" - ");
            label.push_str(&artist);
        }
        if year.is_truthy()? {
            label.push_str(" (");
            label.push_str(&py_str(&year)?);
            label.push(')');
        }
        return Ok(truncate_str(&label, 60));
    }

    if eq_any_str(&result_type, &["artist", "artists"])? {
        let subs = field_get(
            py,
            item,
            "subscribers",
            Some(PyString::new(py, "").into_any()),
        )?;
        let mut label = py_str(&title)?;
        if subs.is_truthy()? {
            label.push_str("  [");
            label.push_str(&py_str(&subs)?);
            label.push(']');
        }
        return Ok(truncate_str(&label, 60));
    }

    if eq_any_str(&result_type, &["playlist", "playlists"])? {
        let author = field_get(py, item, "author", Some(PyString::new(py, "").into_any()))?;
        // count = item.get("itemCount", item.get("count", "")) — the
        // inner default is evaluated first, like the reference.
        let count_default = field_get(py, item, "count", Some(PyString::new(py, "").into_any()))?;
        let count = field_get(py, item, "itemCount", Some(count_default))?;
        let mut label = py_str(&title)?;
        if author.is_truthy()? {
            label.push_str(" by ");
            label.push_str(&py_str(&author)?);
        }
        if count.is_truthy()? {
            label.push_str(" (");
            label.push_str(&py_str(&count)?);
            label.push_str(" tracks)");
        }
        return Ok(truncate_str(&label, 60));
    }

    // Fallback: the bare title, truncated.
    Ok(truncate_str(&py_str(&title)?, 60))
}

/// The bottom-of-panel count label behind `SearchResultPanel.load_items`
/// from `ui/pages/search.py`: `"No results"` for an empty panel,
/// `"<count> result"` for exactly one and `"<count> results"` from two
/// on (the `!= 1` plural rule).
#[pyfunction]
pub(crate) fn search_panel_count_label(count: i64) -> String {
    if count == 0 {
        return "No results".to_string();
    }
    let plural = if count != 1 { "s" } else { "" };
    format!("{count} result{plural}")
}

/// The search-mode indicator label — the `\u{25b6} Music` / `\u{25b6} All`
/// display beside the search input, wrapped in the theme colour markup.
///
/// Mirrors the inline f-string shared by `SearchPage.compose`,
/// `SearchPage.on_mount` and `SearchPage._toggle_search_mode` from
/// `ui/pages/search.py`: `"music"` renders the music-only label,
/// anything else the all-results one.
#[pyfunction]
pub(crate) fn search_mode_label(mode: &str, color: &str) -> String {
    if mode == "music" {
        format!("[{color}]\u{25b6}[/{color}] Music")
    } else {
        format!("[{color}]\u{25b6}[/{color}] All")
    }
}

/// The suggestion-overlay pick decision behind the suggestion-list's
/// `ListView.Selected` handling from `ui/pages/search.py`: a `None`
/// index (no row highlighted) or an index outside the suggestion list
/// — a stale selection arriving after the overlay was repopulated —
/// answers `None`; a valid index answers that suggestion, the raw
/// stored row object.
#[pyfunction]
#[pyo3(signature = (suggestions, index))]
pub(crate) fn search_suggestion_pick<'py>(
    suggestions: &Bound<'py, PyList>,
    index: Option<i64>,
) -> Option<Bound<'py, PyAny>> {
    // idx is not None and 0 <= idx < len(suggestions)
    let index = index?;
    if index < 0 || index >= suggestions.len() as i64 {
        return None;
    }
    Some(
        suggestions
            .get_item(index as usize)
            .expect("index bounds-checked above"),
    )
}

/// The recent-searches label pass behind `SearchPage._load_recent_searches`
/// from `ui/pages/search.py`: each history row becomes its suggestion
/// label — the row's `query` value, raw — and an empty history reads as
/// an empty list so the caller skips the overlay update.
///
/// `entry["query"]` resolves through the full subscription protocol,
/// so a dict subclass's `__getitem__` steers exactly like the
/// reference. Where the reference would raise `KeyError` — a row
/// without a `query` key (unreachable through the history service's
/// SQL-materialised rows, which always carry the column) — the engine
/// stays total and skips the row instead.
#[pyfunction]
pub(crate) fn recent_search_labels<'py>(
    py: Python<'py>,
    entries: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let labels = PyList::empty(py);
    for entry in entries.iter() {
        match entry.get_item("query") {
            Ok(query) => labels.append(query)?,
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyKeyError>(py) => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(labels)
}

/// The `_search_all` categorization — the bucketing of one unfiltered
/// search into the results area's four panels, from
/// `SearchPage._search_all` in `ui/pages/search.py`.
///
/// Each item's `resultType` (defaulting to `""`, the nested default
/// evaluated before the outer lookup exactly like the reference)
/// picks its bucket through Python `==` semantics (see `eq_any_str`):
/// `song` and `video` go to `songs`, `album` to `albums`, `artist` to
/// `artists` and `playlist` to `playlists`; everything else (the
/// empty default included) is dropped. The returned dict always
/// carries the four buckets — the empty ones as empty lists — in the
/// reference's insertion order, and each bucket holds the item
/// objects themselves in arrival order. The input list is never
/// mutated.
///
/// Field reads resolve exactly like the reference's `.get` calls, so
/// dict subclasses steer the bucketing like they steer the reference.
/// Where the reference would raise — a `.get`-less item
/// (`AttributeError`) — the engine stays total: the item reads as
/// type-less and is dropped, the same stance the label engine takes.
#[pyfunction]
pub(crate) fn categorize_search_results<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    let songs = PyList::empty(py);
    let albums = PyList::empty(py);
    let artists = PyList::empty(py);
    let playlists = PyList::empty(py);
    let empty = PyString::new(py, "").into_any();

    for item in raw.iter() {
        // result_type = item.get("resultType", "")
        let result_type = field_get(py, &item, "resultType", Some(empty.clone()))?;
        if eq_any_str(&result_type, &["song", "video"])? {
            songs.append(&item)?;
        } else if eq_any_str(&result_type, &["album"])? {
            albums.append(&item)?;
        } else if eq_any_str(&result_type, &["artist"])? {
            artists.append(&item)?;
        } else if eq_any_str(&result_type, &["playlist"])? {
            playlists.append(&item)?;
        }
    }

    let results = PyDict::new(py);
    results.set_item("songs", songs)?;
    results.set_item("albums", albums)?;
    results.set_item("artists", artists)?;
    results.set_item("playlists", playlists)?;
    Ok(results)
}
