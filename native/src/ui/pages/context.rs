//! Native cores for `src/ytm_player/ui/pages/context.py` — artist/album/
//! playlist headers, discography and progressive artist-songs batches.

use crate::ui::pages::browse::field_get;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::normalize_tracks;
use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString};

// ---------------------------------------------------------------------------
// Context-page view-model assembly (mirrors ui/pages/context.py)
// ---------------------------------------------------------------------------

/// `"<n> track(s)"` — the count segment of every context subtitle, with
/// the reference's `!= 1` plural rule (0 and 2 both pluralise).
fn track_count_label(n: i64) -> String {
    if n != 1 {
        format!("{n} tracks")
    } else {
        format!("{n} track")
    }
}

/// The normalized track list for a raw `tracks` value, plus that raw
/// value's `len()`. A raw value that is not a list — where the fallback's
/// `normalize_tracks` would raise and the ported native one rejects at
/// the boundary — reads as empty with an unknown count, so the engine
/// stays total and the caller's unavailable/remaining decisions degrade
/// to "nothing hidden" / "nothing more to fetch".
pub(crate) fn normalized_tracks_and_len<'py>(
    py: Python<'py>,
    raw_tracks: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Option<i64>)> {
    match raw_tracks.downcast::<PyList>() {
        Ok(list) => {
            let len = list.len() as i64;
            Ok((normalize_tracks(py, list)?, Some(len)))
        }
        Err(_) => Ok((PyList::empty(py), None)),
    }
}

/// Python's `value > n` for the numeric shapes a stored `trackCount` can
/// take: bools compare as 0/1, ints as themselves (a Python int beyond
/// i64 clamps by sign, which only ever decides against realistic
/// counts), floats as their f64. Any other type — where the reference's
/// `>` would raise `TypeError` — reads as `None`, so the caller skips
/// the gated decision instead of raising.
pub(crate) fn numeric_gt(value: &Bound<'_, PyAny>, n: i64) -> Option<bool> {
    if let Ok(b) = value.downcast::<PyBool>() {
        return Some(i64::from(b.is_true()) > n);
    }
    if let Ok(i) = value.extract::<i64>() {
        return Some(i > n);
    }
    if value.is_instance_of::<PyInt>() {
        // A Python int beyond the i64 range: only its sign can decide.
        let negative = value.str().ok()?.to_str().ok()?.starts_with('-');
        return Some(!negative);
    }
    if let Ok(f) = value.extract::<f64>() {
        return Some(f > n as f64);
    }
    None
}

/// `section.get("results", [])` for one of the artist page's sections —
/// `None` when *data* has no such section, the section is not a dict, or
/// its `results` is not a list (each a raise in the fallback's
/// concatenation; the engine stays total and the section contributes no
/// rows).
fn section_results_opt<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let section = field_get(py, data, key, None)?;
    if !section.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    let empty = PyList::empty(py).into_any();
    let results = field_get(py, &section, "results", Some(empty))?;
    Ok(match results.downcast::<PyList>() {
        Ok(list) => Some(list.clone()),
        Err(_) => None,
    })
}

/// The sections of a `get_artist` payload that carry a "see all" page —
/// a dict section with truthy `browseId` and `params` — in the
/// albums-then-singles order the reference's comprehension produces. The
/// returned sections are the original objects, so the caller's
/// `section["results"] = full` swap lands in the fetched payload.
///
/// Mirrors the selection half of
/// `ContextPage._expand_artist_discography` from `ui/pages/context.py`;
/// the concurrent `get_artist_albums` fetch stays in Python (it is I/O).
#[pyfunction]
pub(crate) fn expandable_sections<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut sections: Vec<Bound<'py, PyAny>> = Vec::new();
    for key in ["albums", "singles"] {
        let section = field_get(py, data, key, None)?;
        if !section.is_instance_of::<PyDict>() {
            continue;
        }
        let browse_id = field_get(py, &section, "browseId", None)?;
        if !browse_id.is_truthy()? {
            continue;
        }
        let params = field_get(py, &section, "params", None)?;
        if !params.is_truthy()? {
            continue;
        }
        sections.push(section);
    }
    PyList::new(py, sections)
}

/// The album page's view model — the pure core of
/// `ContextPage._build_album` from `ui/pages/context.py`.
///
/// Returned keys:
///
/// - `title` — `data.get("title", "Unknown Album")`, raw.
/// - `subtitle` — `"<artist> · <year> · <n> track(s)"`: the artist
///   through the shared `extract_artist`, the year segment only when
///   truthy and interpolated via `str()`, the count through the shared
///   plural rule.
/// - `tracks` — `normalize_tracks(data.get("tracks", []))`.
/// - `unavailable` — `len(raw) - len(normalized)`, the count of
///   unplayable entries the header reports as hidden; `None` when the
///   raw value is not a list (the header's label is truthiness-gated,
///   so `None` hides it where the fallback would have raised).
#[pyfunction]
pub(crate) fn album_header<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let unknown_album = PyString::new(py, "Unknown Album").into_any();
    let title = field_get(py, data, "title", Some(unknown_album))?;

    // extract_artist(data) — the shared native engine. A non-dict data
    // (whose field reads above already degraded to their defaults)
    // reads as Unknown, like the search engine's album branch.
    let artist = match data.downcast::<PyDict>() {
        Ok(dict) => extract_artist(py, dict),
        Err(_) => "Unknown".to_string(),
    };

    let year = field_get(py, data, "year", Some(PyString::new(py, "").into_any()))?;
    let empty_tracks = PyList::empty(py).into_any();
    let raw_tracks = field_get(py, data, "tracks", Some(empty_tracks))?;
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    let track_count = tracks.len() as i64;

    // subtitle_parts = [artist_name]; if year: append str(year); append
    // f"{n} track(s)" — joined on " · ".
    let mut parts: Vec<String> = vec![artist];
    if year.is_truthy()? {
        parts.push(py_str(&year)?);
    }
    parts.push(track_count_label(track_count));

    let out = PyDict::new(py);
    out.set_item("title", title)?;
    out.set_item("subtitle", parts.join(" \u{b7} "))?;
    out.set_item("tracks", tracks)?;
    match raw_len {
        Some(len) => out.set_item("unavailable", len - track_count)?,
        None => out.set_item("unavailable", pyo3::types::PyNone::get(py))?,
    }
    Ok(out)
}

/// The playlist page's view model — the pure core of
/// `ContextPage._build_playlist` from `ui/pages/context.py`.
///
/// Returned keys:
///
/// - `title` — `data.get("title", "Unknown Playlist")`, raw.
/// - `subtitle` — `"<owner> · <n> track(s)"` plus `" (loading <total>
///   total…)"` when the stored `trackCount` is truthy and numerically
///   above the normalized count (the owner is `author.get("name",
///   "Unknown")` for a dict `author`, `str(author)` otherwise).
/// - `tracks` — `normalize_tracks(data.get("tracks", []))`.
/// - `unavailable` — `len(raw) - len(normalized)`; `None` when the raw
///   value is not a list.
/// - `owned` — truthiness of `data.get("owned", False)` (the header's
///   Add-to-Library gate).
/// - `fetch_remaining` — `trackCount > len(raw)`, the gate behind the
///   background fetch of the playlist's tail.
/// - `raw_count` — `len(raw)`, the `already_have` the tail fetch
///   resumes from; 0 whenever the gate above is closed.
#[pyfunction]
pub(crate) fn playlist_header<'py>(
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
        py_str(&field_get(py, &author, "name", Some(unknown_name))?)?
    } else {
        py_str(&author)?
    };

    let empty_tracks = PyList::empty(py).into_any();
    let raw_tracks = field_get(py, data, "tracks", Some(empty_tracks))?;
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    let track_count = tracks.len() as i64;

    // total_count = data.get("trackCount") or track_count — the suffix
    // and the remaining-fetch gate only ever fire for a truthy stored
    // count that is numerically above the comparison count; a stored
    // count Python's `>` would reject reads as uncomparable and skips
    // both (the engine stays total).
    let raw_total = field_get(py, data, "trackCount", None)?;
    let total = if raw_total.is_truthy()? {
        Some(raw_total)
    } else {
        None
    };

    let mut subtitle = format!("{owner} \u{b7} {}", track_count_label(track_count));
    let mut fetch_remaining = false;
    if let Some(ref total_value) = total {
        // if total_count > track_count: subtitle += f" (loading {total} total…)"
        if numeric_gt(total_value, track_count) == Some(true) {
            subtitle.push_str(&format!(
                " (loading {} total\u{2026})",
                py_str(total_value)?
            ));
        }
        // if total_count > len(raw_tracks): fetch the tail
        if let Some(raw_count) = raw_len {
            fetch_remaining = numeric_gt(total_value, raw_count) == Some(true);
        }
    }

    // owned = data.get("owned", False) — the gate is truthiness and the
    // default False is falsy like the None a missing key reads as, so
    // one truthiness read covers both.
    let owned = field_get(py, data, "owned", None)?.is_truthy()?;

    let out = PyDict::new(py);
    out.set_item("title", title)?;
    out.set_item("subtitle", subtitle)?;
    out.set_item("tracks", tracks)?;
    match raw_len {
        Some(len) => out.set_item("unavailable", len - track_count)?,
        None => out.set_item("unavailable", pyo3::types::PyNone::get(py))?,
    }
    out.set_item("owned", owned)?;
    out.set_item("fetch_remaining", fetch_remaining)?;
    out.set_item("raw_count", raw_len.unwrap_or(0))?;
    Ok(out)
}

/// The artist page's view model — the pure core of
/// `ContextPage._build_artist` from `ui/pages/context.py`, the
/// discography assembly at the heart of the context page.
///
/// Returned keys:
///
/// - `name` — `data.get("name", "Unknown Artist")`, raw.
/// - `subscriber_text` — `f"{subs} subscribers"` for a truthy
///  `subscribers` value (interpolated via `str()`), `""` otherwise.
/// - `top_songs` — `normalize_tracks(songs.get("results", []))` for a
///   dict `songs` section, `[]` otherwise.
/// - `songs_browse_id` — `songs.get("browseId")` for a dict `songs`
///   section (the playlist holding the artist's full song list, fetched
///   in the background), `None` otherwise.
/// - `albums` — the albums section's `results` concatenated after the
///   singles section's — the discography rows, original objects in
///   feed order.
/// - `related_text` — `"Similar Artists: " + " · ".join(truthy names)`
///   over the first 8 related artists (a dict row's `name` with its
///   truthiness gating the join, `str(row)` for anything else);
///   `""` when there are no truthy `related` results.
#[pyfunction]
pub(crate) fn artist_discography<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let unknown_artist = PyString::new(py, "Unknown Artist").into_any();
    let name = field_get(py, data, "name", Some(unknown_artist))?;

    // subscriber_text = f"{subs} subscribers" if subs else ""
    let subs = field_get(
        py,
        data,
        "subscribers",
        Some(PyString::new(py, "").into_any()),
    )?;
    let subscriber_text = if subs.is_truthy()? {
        format!("{} subscribers", py_str(&subs)?)
    } else {
        String::new()
    };

    // songs_section = data.get("songs", {}); top_songs and browseId are
    // read through the isinstance gate.
    let empty_section = PyDict::new(py).into_any();
    let songs = field_get(py, data, "songs", Some(empty_section.clone()))?;
    let (top_songs, songs_browse_id) = if songs.is_instance_of::<PyDict>() {
        let empty_results = PyList::empty(py).into_any();
        let results = field_get(py, &songs, "results", Some(empty_results))?;
        let top = normalized_tracks_and_len(py, &results)?.0;
        let browse_id = field_get(py, &songs, "browseId", None)?;
        (top, browse_id)
    } else {
        (PyList::empty(py), py.None().into_bound(py))
    };

    // all_albums = albums_list + singles_list
    let mut album_rows: Vec<Bound<'_, PyAny>> = Vec::new();
    for key in ["albums", "singles"] {
        if let Some(list) = section_results_opt(py, data, key)? {
            album_rows.extend(list.iter());
        }
    }

    // related_text = "Similar Artists: " + " · ".join(names of the first
    // 8 related artists, keeping only truthy ones) — gated on truthy
    // results; a truthy non-list (where the fallback's [:8] would
    // raise) contributes no text.
    let related = field_get(py, data, "related", Some(empty_section))?;
    let mut related_text = String::new();
    if related.is_instance_of::<PyDict>() {
        let empty_results = PyList::empty(py).into_any();
        let results = field_get(py, &related, "results", Some(empty_results))?;
        if results.is_truthy()? {
            if let Ok(list) = results.downcast::<PyList>() {
                let mut names: Vec<String> = Vec::new();
                for entry in list.iter().take(8) {
                    // a.get("name", "") if isinstance(a, dict) else str(a)
                    // — the join's `if n` filter runs on the value the
                    // comprehension produced: the raw name for dict rows
                    // (stringified only where the fallback's join would
                    // raise on a non-str), the already-stringified row
                    // for everything else.
                    let name_str = if entry.is_instance_of::<PyDict>() {
                        let name_value =
                            field_get(py, &entry, "name", Some(PyString::new(py, "").into_any()))?;
                        if name_value.is_truthy()? {
                            Some(py_str(&name_value)?)
                        } else {
                            None
                        }
                    } else {
                        let s = py_str(&entry)?;
                        if !s.is_empty() {
                            Some(s)
                        } else {
                            None
                        }
                    };
                    if let Some(s) = name_str {
                        names.push(s);
                    }
                }
                related_text = format!("Similar Artists: {}", names.join(" \u{b7} "));
            }
        }
    }

    let out = PyDict::new(py);
    out.set_item("name", name)?;
    out.set_item("subscriber_text", subscriber_text)?;
    out.set_item("top_songs", top_songs)?;
    out.set_item("songs_browse_id", songs_browse_id)?;
    out.set_item("albums", PyList::new(py, album_rows)?)?;
    out.set_item("related_text", related_text)?;
    Ok(out)
}

/// The closed first-batch plan — nothing to load, no tail to chain —
/// for the worker's early-return gates (an empty raw batch, a batch
/// normalization dropped everything from) and for the shapes the
/// reference could not read at all (a non-dict `data`, a truthy
/// non-list `tracks`).
fn artist_songs_closed_plan(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("tracks", PyList::empty(py))?;
    out.set_item("load", false)?;
    out.set_item("fetch_remaining", false)?;
    out.set_item("raw_count", 0)?;
    Ok(out)
}

/// The first-batch plan behind the context-page worker merge — the pure
/// decision core of the first half of
/// `ContextPage._fetch_full_artist_songs` from `ui/pages/context.py`,
/// the background fetch that swaps the artist's full song list (the
/// playlist behind `songs.browseId`, fetched because ytmusicapi
/// truncates artist top songs) into the top-songs table.
///
/// Mirrors the reference:
///
/// ```python
/// raw_tracks = data.get("tracks", []) if isinstance(data, dict) else []
/// if not raw_tracks: return
/// full_tracks = normalize_tracks(raw_tracks)
/// if not full_tracks: return
/// ```
///
/// Returned keys:
///
/// - `tracks` — `normalize_tracks(raw_tracks)`, the batch to swap into
///   the top-songs table; `[]` whenever the swap gate below is closed
///   (the caller leaves the existing truncated list in place).
/// - `load` — the swap gate: the batch only loads when the raw list is
///   non-empty AND normalization kept something (an all-unplayable
///   batch swaps nothing in).
/// - `fetch_remaining` — the remaining-fetch decision, the merge's
///   core: the chained `get_playlist_remaining` tail fetch runs only
///   when the batch loaded AND the stored `trackCount` does not
///   already cover it — a missing `trackCount` (or an explicit `None`)
///   falls back to attempting the call, with empty/error there reading
///   as "nothing more to fetch"; a count at or below the raw batch
///   length closes the gate.
/// - `raw_count` — `len(raw_tracks)`, the `already_have` the tail
///   fetch resumes from; 0 whenever the tail gate is closed.
///
/// Field reads resolve exactly like the reference's `.get` calls, so
/// dict subclasses steer the plan like they steer the reference. Where
/// the reference would raise — a truthy non-list `tracks` (its
/// `normalize_tracks` iteration) or a stored `trackCount` Python's
/// `<=` would reject (`TypeError`, both unreachable through the
/// service's dict-shaped, int-counted responses) — the engine stays
/// total and reads the batch as closed, the same stance the header
/// engines take.
#[pyfunction]
pub(crate) fn artist_songs_first_batch<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // raw_tracks = data.get("tracks", []) if isinstance(data, dict) else []
    let raw_tracks = if data.is_instance_of::<PyDict>() {
        let empty = PyList::empty(py).into_any();
        field_get(py, data, "tracks", Some(empty))?
    } else {
        PyList::empty(py).into_any()
    };
    // if not raw_tracks: return — no swap, no tail fetch.
    if !raw_tracks.is_truthy()? {
        return artist_songs_closed_plan(py);
    }
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    // if not full_tracks: return — same gate one step later. A truthy
    // non-list raw (where the reference's normalize_tracks would raise
    // in the iteration) reads as no tracks here.
    let raw_count = match raw_len {
        Some(len) if !tracks.is_empty() => len,
        _ => return artist_songs_closed_plan(py),
    };

    // The remaining-fetch decision:
    //   total_count = data.get("trackCount")
    //   if total_count is not None and total_count <= len(raw_tracks):
    //       return
    // — a `None` (missing or explicit) falls back to attempting the
    // call; any comparable count above the batch length chains the
    // tail; anything else closes the gate. `data` is a dict or dict
    // subclass on every path that reaches here (a non-dict read as an
    // empty batch above), so the `trackCount` read steers exactly like
    // the reference's.
    let total = field_get(py, data, "trackCount", None)?;
    let fetch_remaining = if total.is_none() {
        true
    } else {
        // numeric_gt(total, raw_count) == Some(true) is exactly
        // `total > raw_count` — the one comparable outcome whose `<=`
        // reads False and the tail chains. Some(false) is
        // `total <= raw_count` (gate closed), and an uncomparable
        // value — where the reference's `<=` would raise TypeError —
        // closes the gate too, the header engines' stance.
        numeric_gt(&total, raw_count) == Some(true)
    };

    let out = PyDict::new(py);
    out.set_item("tracks", tracks)?;
    out.set_item("load", true)?;
    out.set_item("fetch_remaining", fetch_remaining)?;
    out.set_item("raw_count", raw_count)?;
    Ok(out)
}

/// The tail plan behind the context-page worker merge — the pure
/// decision core of the second half of
/// `ContextPage._fetch_full_artist_songs` from `ui/pages/context.py`:
/// what the chained `get_playlist_remaining` fetch appends to the
/// top-songs table.
///
/// Mirrors the reference:
///
/// ```python
/// if not remaining: return
/// remaining_tracks = normalize_tracks(remaining)
/// if not remaining_tracks: return
/// ```
///
/// Returned keys:
///
/// - `tracks` — `normalize_tracks(remaining)`, the tail rows to
///   append; `[]` whenever the append gate below is closed.
/// - `append` — the append gate: the tail only lands when the fetched
///   remaining list is non-empty AND normalization kept something (an
///   all-unplayable tail is dropped, the batch stays as it loaded).
///
/// A truthy non-list `remaining` (where the reference's
/// `normalize_tracks` iteration would raise — unreachable through
/// `get_playlist_remaining`, which slices a list) reads as no rows —
/// the engine stays total, the same stance the first-batch engine
/// takes.
#[pyfunction]
pub(crate) fn artist_songs_tail<'py>(
    py: Python<'py>,
    remaining: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // if not remaining: return — nothing to append.
    if !remaining.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("tracks", PyList::empty(py))?;
        out.set_item("append", false)?;
        return Ok(out);
    }
    let (tracks, _) = normalized_tracks_and_len(py, remaining)?;
    let append = !tracks.is_empty();
    let out = PyDict::new(py);
    out.set_item("tracks", tracks)?;
    out.set_item("append", append)?;
    Ok(out)
}
