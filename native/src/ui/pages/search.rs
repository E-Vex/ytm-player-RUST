//! Native cores for `src/ytm_player/ui/pages/search.py` — result
//! formatting, panel labels and categorisation, the result-item
//! navigation decisions behind the panel click handlers, the
//! music-mode worker merge, and the nav-state remember-gates behind
//! `get_nav_state`.

use crate::ui::pages::browse::field_get;
use crate::ui::pages::browse::first_truthy_get;
use crate::ui::sidebars::eq_any_str;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::truncate_str;
use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

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

// ---------------------------------------------------------------------------
// Result-item navigation decisions and the music-mode merge
// (mirrors ui/pages/search.py)
// ---------------------------------------------------------------------------

/// The left-click navigation decision behind
/// `SearchPage.on_search_result_panel_item_selected` from
/// `ui/pages/search.py`: which context page a clicked Albums/Artists/
/// Playlists row opens.
///
/// Mirrors the reference:
///
/// ```python
/// result_type = item.get("resultType", "")
/// if panel_id == "albums-panel" or result_type == "album":
///     browse_id = item.get("browseId") or item.get("album_id")
///     if browse_id:
///         navigate("context", context_type="album", context_id=browse_id)
/// elif ...
/// ```
///
/// The panel id match — Python `==` on the widget's own id string — or
/// the item's type tag (the `resultType` read, defaulting to `""`, the
/// nested default evaluated before the outer lookup exactly like the
/// reference) picks the branch; the browse id is the first truthy of
/// the branch's key pair (`browseId`, then `album_id` / `artist_id` /
/// `playlistId`), and a branch that resolves no truthy id answers
/// `None` so the handler ignores the click — the reference's nested
/// `if browse_id:` gate, which also means a panel-matched branch with
/// no id never falls through to a later branch.
///
/// The answer is `None` or `{"context_type": ..., "context_id": ...}`
/// with the context id the raw stored object. Field reads resolve
/// exactly like the reference's `.get` calls (see `field_get`), so
/// dict subclasses steer the branch and the id chain like they steer
/// the reference. Where the reference would raise — a `.get`-less
/// item (`AttributeError`) — the engine stays total and answers
/// `None`, the same stance the label engine takes.
#[pyfunction]
pub(crate) fn search_item_navigation<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
    panel_id: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let empty = PyString::new(py, "").into_any();
    let result_type = field_get(py, item, "resultType", Some(empty))?;

    // The if/elif ladder in the reference's order: a panel match or the
    // item's own type tag (Python `==` on both) picks the branch.
    let branch: (&str, &[&str]) =
        if panel_id.eq("albums-panel")? || eq_any_str(&result_type, &["album"])? {
            ("album", &["browseId", "album_id"])
        } else if panel_id.eq("artists-panel")? || eq_any_str(&result_type, &["artist"])? {
            ("artist", &["browseId", "artist_id"])
        } else if panel_id.eq("playlists-panel")? || eq_any_str(&result_type, &["playlist"])? {
            ("playlist", &["browseId", "playlistId"])
        } else {
            return Ok(None);
        };

    // browse_id = item.get("browseId") or item.get(<branch fallback>) —
    // the first truthy; no truthy id answers None (no navigation).
    let browse_id = match first_truthy_get(py, item, branch.1)? {
        Some(id) => id,
        None => return Ok(None),
    };

    let out = PyDict::new(py);
    out.set_item("context_type", branch.0)?;
    out.set_item("context_id", browse_id)?;
    Ok(Some(out))
}

/// The panel→item-type mapping behind
/// `SearchPage.on_search_result_panel_item_right_clicked` from
/// `ui/pages/search.py`: the right-clicked row's panel id picks the
/// item type the ActionsPopup is opened with — `albums-panel` →
/// `album`, `artists-panel` → `artist`, `playlists-panel` → `playlist`,
/// anything else (the songs table's rows and junk ids both) → `track`.
///
/// Mirrors the reference's `type_map.get(panel_id, "track")` through
/// Python `==` against the three keys, so a str subclass panel id
/// classifies like it does against the reference's dict lookup. Where
/// the reference's lookup would raise — an unhashable panel_id — the
/// engine stays total and answers the `track` default (unreachable
/// through the panels, whose ids are plain strs).
#[pyfunction]
pub(crate) fn search_panel_item_type(panel_id: &Bound<'_, PyAny>) -> PyResult<String> {
    if panel_id.eq("albums-panel")? {
        Ok("album".to_string())
    } else if panel_id.eq("artists-panel")? {
        Ok("artist".to_string())
    } else if panel_id.eq("playlists-panel")? {
        Ok("playlist".to_string())
    } else {
        Ok("track".to_string())
    }
}

/// The subscribe-id resolution behind the right-click ActionsPopup's
/// `toggle_subscribe` arm in
/// `SearchPage.on_search_result_panel_item_right_clicked` from
/// `ui/pages/search.py`: `item.get("browseId") or item.get("artist_id")
/// or ""` — the first truthy id, raw, the empty string when neither
/// reads truthy (the handler's no-id warning branch).
#[pyfunction]
pub(crate) fn search_subscribe_browse_id<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match first_truthy_get(py, item, &["browseId", "artist_id"])? {
        Some(id) => Ok(id),
        None => Ok(PyString::new(py, "").into_any()),
    }
}

/// The copy-link URL behind the right-click ActionsPopup's `copy_link`
/// arm in `SearchPage.on_search_result_panel_item_right_clicked` from
/// `ui/pages/search.py`: the browse-id fallback chain — `browseId`,
/// then `album_id`, `playlistId`, `artist_id`, the first truthy —
/// interpolated into `https://music.youtube.com/browse/<id>`, the
/// reference's f-string `str()` conversion (so a truthy non-str id
/// stringifies exactly like the reference's interpolation), or `None`
/// when no id resolves (the handler's silent no-id branch).
#[pyfunction]
pub(crate) fn search_browse_link<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    let keys = ["browseId", "album_id", "playlistId", "artist_id"];
    let browse_id = match first_truthy_get(py, item, &keys)? {
        Some(id) => id,
        None => return Ok(None),
    };
    Ok(Some(format!(
        "https://music.youtube.com/browse/{}",
        py_str(&browse_id)?
    )))
}

/// The `_search_music` merge — the classification loop behind
/// `SearchPage._search_music`'s gathered concurrent fetches from
/// `ui/pages/search.py`.
///
/// Mirrors the reference:
///
/// ```python
/// gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
/// for category, result in zip(tasks.keys(), gathered):
///     if isinstance(result, Exception):
///         logger.error("Search for %s failed: %s", category, result)
///     elif isinstance(result, list):
///         results[category] = result
/// ```
///
/// Returned keys:
///
/// - `results` — the four-bucket dict (always present, the empty ones
///   as empty lists, in the reference's insertion order) with every
///   gathered list assigned under its category — the original list
///   objects themselves, in arrival order.
/// - `failed` — the `(category, exception)` pairs whose logging stays
///   in the coroutine (I/O side of the seam). A gathered entry that is
///   neither an `Exception` nor a list — unreachable through
///   `asyncio.gather`'s return-exceptions contract — is silently
///   dropped, exactly like the reference's elif ladder.
///
/// The pairs stop at the shorter list, like the reference's `zip`.
#[pyfunction]
pub(crate) fn music_search_results<'py>(
    py: Python<'py>,
    categories: &Bound<'py, PyList>,
    gathered: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    let results = PyDict::new(py);
    results.set_item("songs", PyList::empty(py))?;
    results.set_item("albums", PyList::empty(py))?;
    results.set_item("artists", PyList::empty(py))?;
    results.set_item("playlists", PyList::empty(py))?;

    let failed = PyList::empty(py);
    // zip(tasks.keys(), gathered) — pairs up to the shorter list.
    let pairs = categories.len().min(gathered.len());
    for index in 0..pairs {
        let category = categories.get_item(index)?;
        let result = gathered.get_item(index)?;
        if result.is_instance_of::<pyo3::exceptions::PyException>() {
            failed.append(PyTuple::new(py, [category, result])?)?;
        } else if result.is_instance_of::<PyList>() {
            results.set_item(category, result)?;
        }
    }

    let out = PyDict::new(py);
    out.set_item("results", results)?;
    out.set_item("failed", failed)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// The nav-state remember-gates (mirrors ui/pages/search.py)
// ---------------------------------------------------------------------------

/// `any(mapping.values())` — the results key's remember-gate: the
/// results ride home only when one of the buckets holds something.
/// An exact dict iterates its own values; anything else resolves
/// `values()` through the attribute protocol exactly like the
/// reference (a mapping whose own `values` answers anything iterable
/// steers the gate like it steers the reference). Where the reference
/// would raise `AttributeError` — a `.values`-less results object,
/// unreachable through the page's own always-a-plain-dict results —
/// the engine stays total and the gate reads closed.
fn any_value_truthy(py: Python<'_>, results: &Bound<'_, PyAny>) -> PyResult<bool> {
    if results.is_exact_instance_of::<PyDict>() {
        let dict = results.downcast::<PyDict>().unwrap();
        for (_key, value) in dict.iter() {
            if value.is_truthy()? {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    match results.call_method0("values") {
        Ok(values) => {
            for item in values.try_iter()? {
                if item?.is_truthy()? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => Ok(false),
        Err(err) => Err(err),
    }
}

/// The nav-state remember-gates behind `SearchPage.get_nav_state` from
/// `ui/pages/search.py` — which of the four preserved keys ride home
/// when the page is navigated away from.
///
/// Mirrors the reference:
///
/// ```python
/// state = {}
/// if self._last_query:
///     state["last_query"] = self._last_query
/// if self.search_mode != "music":
///     state["search_mode"] = self.search_mode
/// if any(self._search_results.values()):
///     state["search_results"] = self._search_results
/// if table.cursor_row is not None:
///     state["cursor_row"] = table.cursor_row
/// ```
///
/// Every gate keeps its reference shape: the query remembered under
/// truthiness (an empty query needs no restore); the mode remembered
/// only when it differs from `"music"` — the comparison through
/// CPython's own `!=` protocol, so a str subclass steers like it
/// steers the reference — the default music mode being the fresh-entry
/// default; the results remembered when any bucket holds something
/// (the reference's `any(values())`, see `any_value_truthy`) and riding
/// home as the very dict object they are; and the cursor row remembered
/// whenever it is not `None` — a cursor resting on the first row IS
/// worth restoring (unlike the liked-songs gate, which skips row 0)
/// because the page restore skips a row past the table's end, not a
/// row at the top. The `query_one` and its broad catch stay
/// Python-side in the page (a failed query answers `None`, the same
/// shape a laid-out-but-cursorless table answers, so both leave the
/// key out).
#[pyfunction]
#[pyo3(signature = (last_query, search_mode, results, cursor_row))]
pub(crate) fn search_nav_state<'py>(
    py: Python<'py>,
    last_query: &Bound<'py, PyAny>,
    search_mode: &Bound<'py, PyAny>,
    results: &Bound<'py, PyAny>,
    cursor_row: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // if last_query: state["last_query"] = last_query
    if last_query.is_truthy()? {
        out.set_item("last_query", last_query)?;
    }
    // if search_mode != "music": state["search_mode"] = search_mode
    if search_mode
        .rich_compare("music", CompareOp::Ne)?
        .is_truthy()?
    {
        out.set_item("search_mode", search_mode)?;
    }
    // if any(results.values()): state["search_results"] = results
    if any_value_truthy(py, results)? {
        out.set_item("search_results", results)?;
    }
    // if cursor_row is not None: state["cursor_row"] = cursor_row
    if !cursor_row.is_none() {
        out.set_item("cursor_row", cursor_row)?;
    }
    Ok(out)
}
