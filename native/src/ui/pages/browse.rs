//! Native cores for `src/ytm_player/ui/pages/browse.py` — shelf/chart/
//! playlist row assembly and the event/charts split.

use crate::utils::formatting::clean_shelf_title;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::get_video_id;
use crate::utils::formatting::truncate_str;
use crate::utils::pycompat::py_str;
use crate::utils::pycompat::python_lower;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Browse home-feed shelf classification (mirrors ui/pages/browse.py)
// ---------------------------------------------------------------------------

/// Chart-pill priority prefixes — the `_CHART_PRIORITY` table in
/// `ui/pages/browse.py`: the canonical "what's hot in country X" shelves
/// sort first, in this order; every other title shares rank 99 and sorts
/// alphabetically. Lower rank = earlier.
const CHART_PRIORITY: [&str; 3] = ["Top 100 Songs", "Weekly Top Songs on Shorts", "Trending 20"];

const CHART_UNRANKED: i64 = 99;

/// `str(shelf.get("title", ""))` — the shelf title as Python's `str()` sees
/// it: a missing title is the `""` default, `None` becomes `"None"`, a
/// number its decimal form, any other object its `__str__`. The `.get` goes
/// through the method (so a dict subclass overriding `get` behaves exactly
/// like the fallback), and an object without a `.get` — where the fallback
/// would raise `AttributeError` — reads as `""`, keeping the engine total;
/// the charts caller pre-filters its raw shelf list to dicts, so that path
/// is defensive only.
fn shelf_title(py: Python, shelf: &Bound<'_, PyAny>) -> PyResult<String> {
    let title = match shelf.call_method1("get", ("title", "")) {
        Ok(title) => title,
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
            return Ok(String::new());
        }
        Err(err) => return Err(err),
    };
    Ok(title.str()?.to_string_lossy().into_owned())
}

/// The `_chart_sort_key` pair for a title: the rank of the first priority
/// prefix the title starts with, else 99; the title itself breaks ties.
/// String comparison is byte order, which for UTF-8 equals Python's
/// code-point order, so the tie-break matches the fallback's exactly.
fn chart_key_for(title: &str) -> (i64, &str) {
    for (rank, prefix) in CHART_PRIORITY.iter().enumerate() {
        if title.starts_with(prefix) {
            return (rank as i64, title);
        }
    }
    (CHART_UNRANKED, title)
}

/// A shelf is a global event playlist iff its title carries a brand prefix
/// — YouTube formats events as `"<Brand>: <Generic Title>"` (e.g.
/// `"Coachella 2026: Daily Top 100 Songs"`). Country chart shelves never
/// use the colon-space form.
///
/// Mirrors `_is_event_shelf` from `ui/pages/browse.py`; a non-dict shelf
/// (see `shelf_title`) reads as title-less and classifies as a chart.
#[pyfunction]
pub(crate) fn is_event_shelf(py: Python, shelf: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(shelf_title(py, shelf)?.contains(": "))
}

/// The sort key behind the country-chart pill order: a tuple
/// `(rank, title)` where *rank* is the index of the first `_CHART_PRIORITY`
/// prefix the title starts with (99 for none of them) and *title* is the
/// `str()`-ified shelf title that breaks ties.
///
/// Mirrors `_chart_sort_key` from `ui/pages/browse.py`.
#[pyfunction]
pub(crate) fn chart_sort_key(py: Python, shelf: &Bound<'_, PyAny>) -> PyResult<(i64, String)> {
    let title = shelf_title(py, shelf)?;
    let (rank, title) = chart_key_for(&title);
    Ok((rank, title.to_string()))
}

/// Partition shelves into `(events, charts)`, with the charts pre-sorted by
/// canonical priority. Events keep their original order — there are rarely
/// more than one, and YouTube's ordering reflects editorial intent.
///
/// Mirrors `_split_events_and_charts` from `ui/pages/browse.py`: the charts
/// sort is Python's `sorted(..., key=_chart_sort_key)` — stable, so equal
/// `(rank, title)` keys keep input order — which Rust's stable `sort_by`
/// over the same comparator reproduces. The returned lists hold the input
/// shelf objects themselves (no copies), and the input list is never
/// mutated.
#[pyfunction]
pub(crate) fn split_events_and_charts<'py>(
    py: Python<'py>,
    shelves: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    let mut events: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut charts: Vec<(i64, String, Bound<'py, PyAny>)> = Vec::new();
    for shelf in shelves.iter() {
        let title = shelf_title(py, &shelf)?;
        if title.contains(": ") {
            events.push(shelf);
        } else {
            let (rank, _) = chart_key_for(&title);
            charts.push((rank, title, shelf));
        }
    }
    charts.sort_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));
    let chart_row: Vec<Bound<'py, PyAny>> = charts.into_iter().map(|(_, _, s)| s).collect();
    Ok((PyList::new(py, events)?, PyList::new(py, chart_row)?))
}

/// True for a home-feed row that is a playlist: a mix, a curated or a user
/// playlist, a recap, Liked Music.
///
/// Mirrors `_is_playlist_row` from `ui/pages/browse.py` exactly: the row
/// must be a dict (a subclass counts, like the fallback's `isinstance`),
/// its `playlistId` must be a non-empty `str` (`bool()` truthiness, so a
/// str subclass's `__bool__`/`__len__` governs emptiness), and its
/// `videoId` must be falsy — Python truthiness, so `None`, `""`, `0`, `[]`
/// and `{}` all count as absent. Songs always carry `videoId` (some with a
/// `playlistId` too); albums carry `audioPlaylistId` and never
/// `playlistId`; artists and podcasts carry neither.
#[pyfunction]
pub(crate) fn is_playlist_row(item: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !item.is_instance_of::<PyDict>() {
        return Ok(false);
    }
    let dict = item.downcast::<PyDict>().unwrap();
    match dict.get_item("playlistId")? {
        Some(v) if v.is_instance_of::<PyString>() && v.is_truthy()? => {}
        _ => return Ok(false),
    }
    match dict.get_item("videoId")? {
        Some(v) => Ok(!v.is_truthy()?),
        None => Ok(true),
    }
}

/// The playlist rows across *shelves*, in feed order, one per playlist id.
///
/// Mirrors `_playlist_rows` from `ui/pages/browse.py`: walk the shelves in
/// order (non-dict shelves and shelves whose `contents` is not a list are
/// skipped — subclass dicts and lists count, matching the fallback's
/// `isinstance` gates), keep every `is_playlist_row` item whose
/// `playlistId` has not been seen yet, and return the row objects
/// themselves (the first occurrence wins, no copies). Inputs are never
/// mutated.
#[pyfunction]
pub(crate) fn playlist_rows<'py>(
    py: Python<'py>,
    shelves: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut rows: Vec<Bound<'py, PyAny>> = Vec::new();
    for shelf in shelves.iter() {
        if !shelf.is_instance_of::<PyDict>() {
            continue;
        }
        let dict = shelf.downcast::<PyDict>().unwrap();
        let contents = match dict.get_item("contents")? {
            Some(v) if v.is_instance_of::<PyList>() => v,
            _ => continue,
        };
        let contents = contents.downcast::<PyList>().unwrap();
        for item in contents.iter() {
            if !is_playlist_row(&item)? {
                continue;
            }
            // `item["playlistId"]` — a non-empty str by the gate above.
            let playlist_id = item.downcast::<PyDict>().unwrap().get_item("playlistId")?;
            let pid = match playlist_id {
                Some(v) => v
                    .downcast::<PyString>()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                None => continue, // unreachable through the gate; stays total
            };
            if !seen.insert(pid) {
                continue;
            }
            rows.push(item);
        }
    }
    PyList::new(py, rows)
}

// ---------------------------------------------------------------------------
// Browse rendering path (mirrors ui/pages/browse.py)
// ---------------------------------------------------------------------------

/// The browse-item routing decision core — the classification half of
/// `navigate_browse_item` from `ui/pages/browse.py`.
///
/// Mirrors the reference's reads in order: the type tag (`resultType`
/// falling back to `type` through the truthiness chain, then
/// `.lower()`), the video id (the shared `get_video_id` engine),
/// `browseId`, and the playlist id (`playlistId` falling back to
/// `audioPlaylistId`). The answer names the branch the reference takes:
/// `("play", "")` for a track (the caller plays the item itself), one
/// of `("album", id)` / `("artist", id)` / `("playlist", id)` for a
/// context navigation, or `("", "")` when no branch applies. Each
/// branch's id gate folds in — an album/artist route without a browse
/// id, or a playlist route without any id, classifies as no route,
/// exactly like the reference's `if` guards — and the chosen id is
/// stringified, as the navigation call's interpolation would.
///
/// Where the reference would raise — a `.get`-less item, or a
/// non-string type tag hitting `.lower()` — the engine stays total: the
/// item reads as field-less and the tag is stringified before lowering
/// (see `field_get`); `get_home()` rows are always dicts with string
/// tags.
#[pyfunction]
pub(crate) fn browse_item_route<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
) -> PyResult<(String, String)> {
    // result_type = (item.get("resultType") or item.get("type") or "").lower()
    let result_type = match first_truthy_get(py, item, &["resultType", "type"])? {
        Some(value) => python_lower(&py_str(&value)?),
        None => String::new(),
    };
    // video_id = get_video_id(item) — the shared engine.
    let video_id = match item.downcast::<PyDict>() {
        Ok(dict) => get_video_id(py, dict),
        Err(_) => String::new(),
    };
    // browse_id = item.get("browseId")
    let browse_id = field_get(py, item, "browseId", None)?;
    // playlist_id = item.get("playlistId") or item.get("audioPlaylistId")
    let playlist_id = first_truthy_get(py, item, &["playlistId", "audioPlaylistId"])?;

    if matches!(result_type.as_str(), "song" | "video" | "flat_song") || !video_id.is_empty() {
        return Ok(("play".to_string(), String::new()));
    }
    if matches!(result_type.as_str(), "album" | "single") {
        if browse_id.is_truthy()? {
            return Ok(("album".to_string(), py_str(&browse_id)?));
        }
        return Ok((String::new(), String::new()));
    }
    if result_type == "artist" {
        if browse_id.is_truthy()? {
            return Ok(("artist".to_string(), py_str(&browse_id)?));
        }
        return Ok((String::new(), String::new()));
    }
    if result_type == "playlist" {
        // playlist_id or browse_id — the truthiness chain, re-evaluated.
        let chosen = match &playlist_id {
            Some(v) if v.is_truthy()? => v.clone(),
            _ => browse_id.clone(),
        };
        if chosen.is_truthy()? {
            return Ok(("playlist".to_string(), py_str(&chosen)?));
        }
        return Ok((String::new(), String::new()));
    }
    if let Some(v) = &playlist_id {
        if v.is_truthy()? {
            return Ok(("playlist".to_string(), py_str(v)?));
        }
    }
    if browse_id.is_truthy()? {
        return Ok(("album".to_string(), py_str(&browse_id)?));
    }
    Ok((String::new(), String::new()))
}

/// The For You shelf view model — `ForYouSection._populate_shelves`'s
/// per-shelf render pass from `ui/pages/browse.py`.
///
/// Mirrors the reference's reads: the shelf title (the
/// `"Recommendations"` default, returned as the object it is), the
/// contents (the `[]` default; a falsy value answers no rows), and for
/// each of the first eight items the `(display, item)` pair — the item
/// title (the `"Unknown"` default, the object itself when no subtitle
/// joins it) over the `extract_artist` artist (dropped when `"Unknown"`)
/// and the stringified `description`, joined with `" - "`, the combined
/// line truncated to 80 characters through the shared `truncate`
/// semantics. The row's item is the input object itself (no copy), so
/// the caller's `_shelf_items` list keeps identity with the feed.
///
/// Where the reference would raise — a `.get`-less shelf or item, or
/// non-list contents failing the slice — the engine stays total: the
/// shelf reads as title-less with no rows, a non-dict item renders
/// through its defaults, and non-list contents answer no rows;
/// `get_home()` shelves are always dicts with list contents.
#[pyfunction]
pub(crate) fn foryou_shelf_rows<'py>(
    py: Python<'py>,
    shelf: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyList>)> {
    // title = shelf.get("title", "Recommendations")
    let title = field_get(
        py,
        shelf,
        "title",
        Some(PyString::new(py, "Recommendations").into_any()),
    )?;
    // contents = shelf.get("contents", [])
    let contents = field_get(py, shelf, "contents", Some(PyList::empty(py).into_any()))?;
    if !contents.is_truthy()? {
        return Ok((title, PyList::empty(py)));
    }
    let mut rows: Vec<Bound<'py, PyTuple>> = Vec::new();
    if let Ok(list) = contents.downcast::<PyList>() {
        for item in list.iter().take(8) {
            // item_title = item.get("title", "Unknown")
            let item_title = field_get(
                py,
                &item,
                "title",
                Some(PyString::new(py, "Unknown").into_any()),
            )?;
            // artist_str = extract_artist(item)
            let artist_str = match item.downcast::<PyDict>() {
                Ok(dict) => extract_artist(py, dict),
                Err(_) => "Unknown".to_string(),
            };
            let mut subtitle_parts: Vec<String> = Vec::new();
            if !artist_str.is_empty() && artist_str != "Unknown" {
                subtitle_parts.push(artist_str);
            }
            // description = item.get("description", "")
            let description = field_get(
                py,
                &item,
                "description",
                Some(PyString::new(py, "").into_any()),
            )?;
            if description.is_truthy()? {
                subtitle_parts.push(py_str(&description)?);
            }
            let subtitle = subtitle_parts.join(" - ");
            let display: Bound<'py, PyAny> = if subtitle.is_empty() {
                item_title
            } else {
                let combined = format!("{}  {}", py_str(&item_title)?, subtitle);
                PyString::new(py, &truncate_str(&combined, 80)).into_any()
            };
            rows.push(PyTuple::new(py, [display, item])?);
        }
    }
    Ok((title, PyList::new(py, rows)?))
}

/// The raw chart shelves from a `get_charts()` payload —
/// `ChartsSection.load_data`'s extraction walk from
/// `ui/pages/browse.py`: the `daily`, `weekly` and `videos` arrays in
/// that order, keeping only the dict shelves. A payload that is not a
/// dict (the service's `None` failure marker included) answers no
/// shelves; the payload is never mutated.
#[pyfunction]
pub(crate) fn charts_raw_shelves<'py>(
    py: Python<'py>,
    chart_data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut raw: Vec<Bound<'py, PyAny>> = Vec::new();
    if chart_data.is_instance_of::<PyDict>() {
        for key in ["daily", "weekly", "videos"] {
            // arr = chart_data.get(key)
            let arr = field_get(py, chart_data, key, None)?;
            if let Ok(list) = arr.downcast::<PyList>() {
                for shelf in list.iter() {
                    if shelf.is_instance_of::<PyDict>() {
                        raw.push(shelf);
                    }
                }
            }
        }
    }
    PyList::new(py, raw)
}

/// The chart pill row view model — `ChartsSection._render_pills`' label
/// and class pass from `ui/pages/browse.py`.
///
/// Mirrors the reference's per-shelf reads: the raw title (`shelf.get`
/// only for a dict shelf, the `Shelf {n}` fallback for everything
/// missing or falsy), normalised through the shared `clean_shelf_title`
/// engine with *region_names* travelling in from the caller (the
/// Python-side `CHART_REGIONS` spelling of the same table the module's
/// own seam passes); the CSS classes — `shelf-pill event` for the
/// leading event shelves (`index < *event_count*`), `shelf-pill` for
/// the country charts, ` active` appended on the shelf the table shows.
/// The pill ids and the mounting stay with the caller.
#[pyfunction]
#[pyo3(signature = (dailies, event_count, active_daily, region_names))]
pub(crate) fn charts_pill_rows(
    py: Python,
    dailies: &Bound<'_, PyList>,
    event_count: i64,
    active_daily: i64,
    region_names: &Bound<'_, PyList>,
) -> PyResult<Vec<(String, String)>> {
    let mut rows: Vec<(String, String)> = Vec::with_capacity(dailies.len());
    for (i, shelf) in dailies.iter().enumerate() {
        let idx = i as i64;
        // raw_title = (shelf.get("title") if isinstance(shelf, dict) else None)
        //             or f"Shelf {i + 1}"
        let raw_title = if shelf.is_instance_of::<PyDict>() {
            field_get(py, &shelf, "title", None)?
        } else {
            py.None().into_bound(py)
        };
        let raw_title = if raw_title.is_truthy()? {
            raw_title
        } else {
            PyString::new(py, &format!("Shelf {}", idx + 1)).into_any()
        };
        // title = clean_shelf_title(str(raw_title))
        let title = clean_shelf_title(py, &py_str(&raw_title)?, region_names);
        let mut classes = if idx < event_count {
            "shelf-pill event"
        } else {
            "shelf-pill"
        }
        .to_string();
        if idx == active_daily {
            classes.push_str(" active");
        }
        rows.push((title, classes));
    }
    Ok(rows)
}

/// The chart country line — the label `ChartsSection` mounts above the
/// pills and refreshes on every load, from `ui/pages/browse.py`.
#[pyfunction]
pub(crate) fn charts_country_label(country: &str) -> String {
    format!("Country: {country}    (press 'c' to change)")
}

/// The event pill row's visibility gate —
/// `ChartsSection._update_event_row_visibility` from
/// `ui/pages/browse.py`: at least one event shelf and a terminal at
/// least *narrow_threshold* columns wide. The threshold travels in from
/// the caller, where the `_NARROW_THRESHOLD` constant lives.
#[pyfunction]
pub(crate) fn charts_event_row_visible(
    event_count: i64,
    term_width: i64,
    narrow_threshold: i64,
) -> bool {
    event_count > 0 && term_width >= narrow_threshold
}

/// The Releases row label for one album —
/// `NewReleasesSection._populate_releases`' per-album assembly from
/// `ui/pages/browse.py`.
///
/// Mirrors the reference's reads: the title (the `"Unknown Album"`
/// default), the artist through the `extract_artist` engine (an
/// `"Unknown"` answer drops the `by` clause), the album `type` and
/// `year` (both the `""` default, truthiness-gated) joined into the
/// parenthesised meta clause, and the whole line truncated to 80
/// characters. Where the reference's `" ".join` would raise on a
/// non-string title or type, the engine stringifies it first (the
/// f-string branches already do); `get_new_releases()` rows always
/// carry strings.
#[pyfunction]
pub(crate) fn new_release_label<'py>(
    py: Python<'py>,
    album: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // title = album.get("title", "Unknown Album")
    let title = field_get(
        py,
        album,
        "title",
        Some(PyString::new(py, "Unknown Album").into_any()),
    )?;
    // artist_str = extract_artist(album); "Unknown" reads as absent.
    let artist_str = match album.downcast::<PyDict>() {
        Ok(dict) => extract_artist(py, dict),
        Err(_) => "Unknown".to_string(),
    };
    let artist_str = if artist_str == "Unknown" {
        String::new()
    } else {
        artist_str
    };
    // album_type = album.get("type", "")  /  year = album.get("year", "")
    let album_type = field_get(py, album, "type", Some(PyString::new(py, "").into_any()))?;
    let year = field_get(py, album, "year", Some(PyString::new(py, "").into_any()))?;

    let mut parts: Vec<String> = vec![py_str(&title)?];
    if !artist_str.is_empty() {
        parts.push(format!("by {artist_str}"));
    }
    let mut meta_parts: Vec<String> = Vec::new();
    if album_type.is_truthy()? {
        meta_parts.push(py_str(&album_type)?);
    }
    if year.is_truthy()? {
        meta_parts.push(py_str(&year)?);
    }
    if !meta_parts.is_empty() {
        parts.push(format!("({})", meta_parts.join(", ")));
    }
    Ok(truncate_str(&parts.join(" "), 80))
}

/// The Playlists row label for one home-feed playlist row —
/// `PlaylistsSection._populate_playlists`' per-playlist assembly from
/// `ui/pages/browse.py`.
///
/// Mirrors the reference's reads: the title (the first truthy
/// `playlist.get("title")`, else `"Untitled playlist"`, stringified),
/// then the parenthesised `description` when truthy, else the
/// parenthesised `<count> songs` for a truthy `count`, and the whole
/// line truncated to 80 characters. Field reads resolve exactly like
/// the reference's `.get` calls and truthiness is Python's (see
/// `field_get`).
#[pyfunction]
pub(crate) fn playlist_row_label<'py>(
    py: Python<'py>,
    playlist: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // title = playlist.get("title") or "Untitled playlist"
    let title = {
        let raw = field_get(py, playlist, "title", None)?;
        if raw.is_truthy()? {
            raw
        } else {
            PyString::new(py, "Untitled playlist").into_any()
        }
    };
    // description = playlist.get("description", "")  /  count = playlist.get("count", "")
    let description = field_get(
        py,
        playlist,
        "description",
        Some(PyString::new(py, "").into_any()),
    )?;
    let count = field_get(
        py,
        playlist,
        "count",
        Some(PyString::new(py, "").into_any()),
    )?;

    let mut parts: Vec<String> = vec![py_str(&title)?];
    if description.is_truthy()? {
        parts.push(format!("({})", py_str(&description)?));
    } else if count.is_truthy()? {
        parts.push(format!("({} songs)", py_str(&count)?));
    }
    Ok(truncate_str(&parts.join(" "), 80))
}

/// The Subscriptions row label for one subscribed artist —
/// `SubscriptionsSection._populate_subscriptions`' per-artist assembly
/// from `ui/pages/browse.py`.
///
/// Mirrors the reference's reads: the name (the `"Unknown artist"`
/// default) and the subscriber count (the `""` default); a truthy
/// count appends the two-space `(<count> subscribers)` clause, and the
/// whole line truncates to 80 characters. Where the reference would
/// hand a non-string bare name through to `truncate` (which would
/// raise on `len()`), the engine stringifies it first — the subscriber
/// line's f-string already does, and `get_library_artists()` always
/// answers strings.
#[pyfunction]
pub(crate) fn subscription_label<'py>(
    py: Python<'py>,
    artist: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // name = artist.get("artist", "Unknown artist")
    let name = field_get(
        py,
        artist,
        "artist",
        Some(PyString::new(py, "Unknown artist").into_any()),
    )?;
    // subscribers = artist.get("subscribers", "")
    let subscribers = field_get(
        py,
        artist,
        "subscribers",
        Some(PyString::new(py, "").into_any()),
    )?;
    let display = if subscribers.is_truthy()? {
        format!(
            "{}  ({} subscribers)",
            py_str(&name)?,
            py_str(&subscribers)?
        )
    } else {
        py_str(&name)?
    };
    Ok(truncate_str(&display, 80))
}

// ---------------------------------------------------------------------------
// Search-result label formatting (mirrors ui/pages/search.py)
// ---------------------------------------------------------------------------

/// `item.get(key, default)`, exactly like the reference reads it.
///
/// Two indistinguishable fast/slow paths: an exact `dict` goes through the
/// C-level lookup (where `dict.get` and `PyDict_GetItemWithError` are the
/// same operation, minus the method dispatch the boundary would otherwise
/// pay per field), while anything else — a dict subclass overriding
/// `get`, or a mapping-like object — dispatches the real `get` method so
/// overrides steer the engine exactly like they steer the reference. A
/// `.get`-less item (where the reference would raise `AttributeError`)
/// reads as the default — the engine stays total. *default* `None` means
/// the reference's one-argument `item.get(key)`.
pub(crate) fn field_get<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
    key: &str,
    default: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if item.is_exact_instance_of::<PyDict>() {
        let dict = item.downcast::<PyDict>().unwrap();
        return Ok(match dict.get_item(key)? {
            Some(value) => value,
            None => default.unwrap_or_else(|| py.None().into_bound(py)),
        });
    }
    let default = default.unwrap_or_else(|| py.None().into_bound(py));
    match item.call_method1("get", (key, default.clone())) {
        Ok(value) => Ok(value),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => Ok(default),
        Err(err) => Err(err),
    }
}

/// The first truthy `item.get(key)` across *keys* (Python truthiness),
/// or `None` when every value is missing or falsy — the reference's
/// `item.get("title") or item.get("name") or …` chain.
pub(crate) fn first_truthy_get<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
    keys: &[&str],
) -> PyResult<Option<Bound<'py, PyAny>>> {
    for key in keys {
        let value = field_get(py, item, key, None)?;
        if value.is_truthy()? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}
