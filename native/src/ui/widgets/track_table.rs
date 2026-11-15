//! Native cores for `src/ytm_player/ui/widgets/track_table.py` —
//! filter/sort keys, view rows, marks and the range/shift selection
//! engine.

use crate::ui::pages::browse::field_get;
use crate::ui::pages::recently_played::shallow_copy_dict;
use crate::utils::bidi::gated_reorder;
use crate::utils::bidi::isolate_bidi_impl;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::extract_duration;
use crate::utils::formatting::format_duration;
use crate::utils::pycompat::py_str;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFrozenSet, PyList, PySet, PyString};

// ---------------------------------------------------------------------------
// Track-table view engine (mirrors ui/widgets/track_table.py)
// ---------------------------------------------------------------------------

/// The sortable columns of the track table — the `_SORT_KEYS` table in
/// `ui/widgets/track_table.py`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TrackSortColumn {
    Index,
    Title,
    Artist,
    Album,
    Duration,
}

impl TrackSortColumn {
    /// Resolve a sort-column name. Unknown names sort nothing, mirroring
    /// the Python table lookup `_SORT_KEYS.get(column or "")`.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "index" => Some(Self::Index),
            "title" => Some(Self::Title),
            "artist" => Some(Self::Artist),
            "album" => Some(Self::Album),
            "duration" => Some(Self::Duration),
            _ => None,
        }
    }
}

/// Sort-key payload for one row. Every row of one view shares the same
/// variant (the column fixes it), so ordering never crosses variants;
/// each payload compares exactly like the Python key expression it
/// mirrors — strings by `<` (UTF-8 byte order equals code-point order,
/// so it matches Python's `str` comparison) and ints numerically.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum TrackSortKey {
    Str(String),
    Int(i64),
}

/// Lowercased `title`/`album` with the Python fallback's
/// `(t.get(key) or "")` semantics: missing, `None` and empty all read as
/// `""`. Non-string values degrade to `""` where the fallback would raise
/// — `load_tracks` only ever stores strings there.
fn lowercased_str_field(track: &Bound<'_, PyDict>, key: &str) -> String {
    track
        .get_item(key)
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .unwrap_or_default()
        .to_lowercase()
}

/// The `_matches_filter` test: a substring of the already-lowercased
/// *query* against the lowercased title, artist and album (the caller's
/// `apply_filter` strips and lowercases the query before it gets here).
fn track_matches_filter(py: Python, track: &Bound<'_, PyDict>, query: &str) -> bool {
    let title = lowercased_str_field(track, "title");
    let artist = extract_artist(py, track).to_lowercase();
    let album = lowercased_str_field(track, "album");
    title.contains(query) || artist.contains(query) || album.contains(query)
}

/// Sort key for one row under *column*; `None` reads the row as an empty
/// track (missing/`None` fields, `"Unknown"` artist, 0 index/duration).
fn track_sort_key(
    py: Python,
    column: TrackSortColumn,
    track: Option<&Bound<'_, PyDict>>,
) -> TrackSortKey {
    let dict = match track {
        Some(d) => d,
        None => {
            return match column {
                TrackSortColumn::Index | TrackSortColumn::Duration => TrackSortKey::Int(0),
                _ => TrackSortKey::Str(String::new()),
            };
        }
    };
    match column {
        TrackSortColumn::Index => TrackSortKey::Int(
            dict.get_item("_original_index")
                .ok()
                .flatten()
                .and_then(|v| v.extract::<i64>().ok())
                .unwrap_or(0),
        ),
        TrackSortColumn::Title => TrackSortKey::Str(lowercased_str_field(dict, "title")),
        TrackSortColumn::Artist => TrackSortKey::Str(extract_artist(py, dict).to_lowercase()),
        TrackSortColumn::Album => TrackSortKey::Str(lowercased_str_field(dict, "album")),
        TrackSortColumn::Duration => TrackSortKey::Int(extract_duration(py, dict)),
    }
}

/// Compute the track-table view order: filter, then sort.
///
/// Mirrors `TrackTable._rebuild_view` + `_matches_filter` + `_SORT_KEYS`
/// from `ui/widgets/track_table.py`:
///
/// - *query* arrives already stripped and lowercased (the caller's
///   `apply_filter` preprocessing); a row matches when the query is a
///   substring of its lowercased title, artist, or album. An empty query
///   keeps every row.
/// - *sort_column* is one of `index` / `title` / `artist` / `album` /
///   `duration`; `None` or an unknown name leaves load order.
/// - The sort is stable, and with *sort_reverse* ties still keep their
///   input order — exactly Python's `list.sort(key=..., reverse=True)`.
///
/// Returns the positions (indices into *tracks*) of the visible rows, in
/// display order, so the caller can slice both its track list and the
/// `_original_index` map from one pass.
///
/// Non-dict rows and non-string title/album values degrade gracefully
/// (never match a filter, read as empty fields) where the Python fallback
/// would raise — `load_tracks` guarantees dicts with string fields, and
/// the engine staying total keeps a malformed row from taking the whole
/// view down with it.
#[pyfunction(signature = (tracks, query, sort_column, sort_reverse))]
pub(crate) fn filter_sort_tracks(
    py: Python,
    tracks: &Bound<'_, PyList>,
    query: &str,
    sort_column: Option<&str>,
    sort_reverse: bool,
) -> Vec<usize> {
    let column = sort_column.and_then(TrackSortColumn::from_name);
    let mut positions: Vec<usize> = Vec::with_capacity(tracks.len());

    for (i, item) in tracks.iter().enumerate() {
        if query.is_empty() {
            positions.push(i);
        } else if let Ok(dict) = item.downcast::<PyDict>() {
            if track_matches_filter(py, dict, query) {
                positions.push(i);
            }
        }
    }

    if let Some(column) = column {
        let mut keyed: Vec<(usize, TrackSortKey)> = Vec::with_capacity(positions.len());
        for &i in &positions {
            let dict = tracks
                .get_item(i)
                .ok()
                .and_then(|item| item.downcast_into::<PyDict>().ok());
            keyed.push((i, track_sort_key(py, column, dict.as_ref())));
        }
        // Stable sort; a reversed comparator keeps ties in input order,
        // matching Python's `list.sort(key=..., reverse=True)`.
        keyed.sort_by(|a, b| {
            let ord = a.1.cmp(&b.1);
            if sort_reverse {
                ord.reverse()
            } else {
                ord
            }
        });
        positions = keyed.into_iter().map(|(i, _)| i).collect();
    }

    positions
}

/// One rendered cell — the `render_track_cell` chain (the gated line
/// reorder, then the FSI...PDI isolate wrap) with the wrapper's falsy
/// passthrough folded in, so the fused row engines render each text
/// cell exactly like the per-cell engine the call sites used to reach
/// for. Only the cached `bidi_mode` verdict stays Python-side,
/// travelling in as a plain bool.
fn track_cell(text: &str, reorder_enabled: bool) -> String {
    if text.is_empty() {
        text.to_string()
    } else {
        isolate_bidi_impl(&gated_reorder(text, reorder_enabled), true)
    }
}

/// One composed visible row — the `(row_key, cells)` pair the fused
/// view engine answers per visible position, converted to a Python
/// tuple-of-(str, list-of-str) on the way out.
type TrackRow = (String, Vec<String>);

/// The single-row assembly — one visible row's `(row_key, cells)` pair,
/// the pure core of `TrackTable._add_track_row` from
/// `ui/widgets/track_table.py`.
///
/// Mirrors the reference's reads in the reference's own order, every
/// read through the dict's own `.get` method: the `title` (the
/// `"Unknown"` default), the artist through the `extract_artist`
/// engine (the artists-list walk with its view-count pseudo-artist
/// exclusion and the `"Unknown"` fallback), the `album` through the
/// `or ""` truthiness gate, and the duration through the
/// `extract_duration` engine. The mark glyph is the `✓` when the
/// `_original_index` read (the visible *index* as its default) is a
/// member of *marked* — CPython's own containment, so the set's
/// hash/eq semantics decide — the `#` cell `str(orig + 1)` through
/// CPython's own `+` and `str()` (only when *show_index*), the three
/// text cells through the per-cell render chain with its falsy
/// passthrough (`title` and `artist` always, `album` only when
/// *show_album*), the duration cell `format_duration(duration)` when
/// the duration is truthy else `"--:--"`, and the row key
/// `f"{video_id}_{index}"` — the `video_id` read (the `"row_{index}"`
/// default evaluated before the read runs, exactly like the
/// reference's argument evaluation) rendered through `str()`.
///
/// The reference reads `_original_index` twice (once for the glyph,
/// once for the `#` cell); the engine reads it once — the same value
/// both times on the plain dicts `load_tracks` builds. Non-dict
/// tracks raise the downcast's own `TypeError` where the reference's
/// first `.get` would raise `AttributeError` — `load_tracks` wraps
/// every track in a fresh `dict`, so the shape is unreachable through
/// the table's own paths.
fn track_row_values_impl(
    py: Python,
    track: &Bound<'_, PyDict>,
    index: usize,
    show_index: bool,
    show_album: bool,
    marked: &Bound<'_, PyAny>,
    reorder_enabled: bool,
) -> PyResult<(String, Vec<String>)> {
    // title = track.get("title", "Unknown")
    let title_obj = track.call_method1("get", ("title", "Unknown"))?;
    let title: String = title_obj.extract()?;
    // artist = extract_artist(track)
    let artist = extract_artist(py, track);
    // album = track.get("album") or ""
    let album_raw = track.call_method1("get", ("album",))?;
    let album: String = if album_raw.is_truthy()? {
        album_raw.extract()?
    } else {
        String::new()
    };
    // duration = extract_duration(track)
    let duration = extract_duration(py, track);
    // orig = track.get("_original_index", index)
    let index_obj: usize = index;
    let orig = track.call_method1("get", ("_original_index", index_obj))?;

    let mut cells: Vec<String> = Vec::with_capacity(6);
    // _mark_glyph(orig): the containment over the caller's marked set.
    cells.push(if marked.contains(&orig)? {
        "✓".to_string()
    } else {
        " ".to_string()
    });
    if show_index {
        // str(orig + 1) — CPython's own + then str().
        let one = 1i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any();
        let bumped = orig.add(&one)?;
        cells.push(py_str(&bumped)?);
    }
    cells.push(track_cell(&title, reorder_enabled));
    cells.push(track_cell(&artist, reorder_enabled));
    if show_album {
        cells.push(track_cell(&album, reorder_enabled));
    }
    // format_duration(duration) if duration else "--:--"
    if duration != 0 {
        cells.push(format_duration(duration));
    } else {
        cells.push("--:--".to_string());
    }

    // video_id = track.get("video_id", f"row_{index}")
    let row_default = format!("row_{}", index);
    let video_id = track.call_method1("get", ("video_id", row_default.as_str()))?;
    // f"{video_id}_{index}"
    let key = format!("{}_{}", py_str(&video_id)?, index);
    Ok((key, cells))
}

/// The single-row assembly engine — `ui/widgets/track_table.py`'s
/// `_row_values`, the per-row core behind the append path
/// (`append_tracks`' unsorted branch) and the fallback composition of
/// the fused engine below. See `track_row_values_impl` for the
/// mirrored contract; the answer is the `(row_key, cells)` pair the
/// caller hands to Textual's `add_row`, with the `add_row` call itself
/// staying Python-side.
#[pyfunction]
#[pyo3(signature = (track, index, show_index, show_album, marked, reorder_enabled))]
pub(crate) fn track_table_row_values<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
    index: usize,
    show_index: bool,
    show_album: bool,
    marked: &Bound<'py, PyAny>,
    reorder_enabled: bool,
) -> PyResult<(String, Vec<String>)> {
    let dict = track
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("expected a dict track"))?;
    track_row_values_impl(
        py,
        dict,
        index,
        show_index,
        show_album,
        marked,
        reorder_enabled,
    )
}

/// The fused view-rows engine — `ui/widgets/track_table.py`'s
/// `_compose_view_rows`, the one native call behind every table
/// rebuild: `load_tracks`' row walk and `_rebuild_view` +
/// `_reload_sorted`'s filter/sort/row-assembly chain, folded together.
///
/// The visible-row order is the `filter_sort_tracks` engine's own walk
/// (the filter, then the stable column sort — see its contract above),
/// and each visible row's `(row_key, cells)` pair the
/// `track_row_values_impl` assembly over the track at that position,
/// with the VISIBLE row number as the assembly's *index* (the row-key
/// suffix, the `_original_index` default and the `row_{index}`
/// fallback all name the visible slot, exactly like the reference's
/// `enumerate` walk). The answer is the `(positions, rows)` pair — the
/// positions the caller slices its track list and `_filtered_map`
/// from, the rows it hands to `add_row` — with the marks set passed
/// through as the object it is (the glyph containment runs through
/// CPython's own protocol) and the cached `bidi_mode` verdict
/// travelling in as a bool, read once per rebuild instead of once per
/// cell. The `add_row` calls, the cursor/scroll preservation and the
/// playing-row highlight stay Python-side in the widget.
#[allow(clippy::too_many_arguments)] // the view params plus the marks set and the cached bidi verdict
#[pyfunction]
#[pyo3(signature = (tracks, query, sort_column, sort_reverse, show_index, show_album, marked, reorder_enabled))]
pub(crate) fn track_table_view_rows(
    py: Python,
    tracks: &Bound<'_, PyList>,
    query: &str,
    sort_column: Option<&str>,
    sort_reverse: bool,
    show_index: bool,
    show_album: bool,
    marked: &Bound<'_, PyAny>,
    reorder_enabled: bool,
) -> PyResult<(Vec<usize>, Vec<TrackRow>)> {
    let positions = filter_sort_tracks(py, tracks, query, sort_column, sort_reverse);
    let mut rows: Vec<TrackRow> = Vec::with_capacity(positions.len());
    for (visible, &pos) in positions.iter().enumerate() {
        let item = tracks.get_item(pos)?;
        let dict = item
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("expected a dict track"))?;
        rows.push(track_row_values_impl(
            py,
            dict,
            visible,
            show_index,
            show_album,
            marked,
            reorder_enabled,
        )?);
    }
    Ok((positions, rows))
}

/// The load/append stamping pass — `TrackTable.load_tracks`' and
/// `append_tracks`' per-track copies from `ui/widgets/track_table.py`:
/// a shallow `dict(track)` copy per track carrying `_original_index` =
/// *start* + position (the fresh load stamps from 0, the append path
/// from its `start_idx`), the copies in list order. The input list and
/// its dicts are never touched; the copies are new plain dicts holding
/// the same value objects (see `shallow_copy_dict`), so a dict subclass
/// in normalises to a plain dict exactly like the reference's
/// `dict(track)`.
///
/// A non-dict track raises `TypeError` where the reference's
/// `dict(track)` would raise for the shapes the table's own load paths
/// produce — they receive normalized track dicts, never pair-iterables
/// or scalars.
#[pyfunction]
#[pyo3(signature = (tracks, start=0))]
pub(crate) fn track_table_stamp_tracks<'py>(
    py: Python<'py>,
    tracks: &Bound<'py, PyList>,
    start: i64,
) -> PyResult<Bound<'py, PyList>> {
    let stamped = PyList::empty(py);
    for (i, track) in tracks.iter().enumerate() {
        let dict = track
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("expected a dict track"))?;
        let copy = shallow_copy_dict(py, dict)?;
        copy.set_item("_original_index", start + (i as i64))?;
        stamped.append(copy)?;
    }
    Ok(stamped)
}

/// The marked-rows status line — `TrackTable._update_mark_status`'s text
/// from `ui/widgets/track_table.py`: `"<count> selected"`, plus
/// `" (<hidden> hidden)"` when the active filter hides any marked row,
/// plus the bulk-action hint. The widget only ever asks with a count of
/// one or more — it clears the line and drops the `-marked` class
/// itself before any zero-count text could be shown.
#[pyfunction]
pub(crate) fn track_table_mark_status(count: i64, hidden: i64) -> String {
    let mut text = format!("{count} selected");
    if hidden > 0 {
        text.push_str(&format!(" ({hidden} hidden)"));
    }
    text.push_str(" · A: add · Esc: clear");
    text
}

/// The selection-info label for the highlighted row — the shared body
/// of `TrackTable.on_data_table_row_highlighted` and `on_focus` from
/// `ui/widgets/track_table.py`: the title (the `""` default through the
/// `or ""` truthiness gate) joined with the `extract_artist` engine's
/// artist by an em dash. `extract_artist` never answers empty (its
/// `"Unknown"` fallback), so the join is the live path; the bare-title
/// arm stays mirrored for safety. Where the reference would hand a
/// non-string title object through to the info bar, the label
/// stringifies it first (the bar renders the same text).
#[pyfunction]
pub(crate) fn track_table_selection_label<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // title = track.get("title", "") or ""
    let title = field_get(py, track, "title", Some(PyString::new(py, "").into_any()))?;
    let title = if title.is_truthy()? {
        title
    } else {
        PyString::new(py, "").into_any()
    };
    // artist = extract_artist(track) or ""
    let artist = match track.downcast::<PyDict>() {
        Ok(dict) => extract_artist(py, dict),
        Err(_) => "Unknown".to_string(),
    };
    if artist.is_empty() {
        return py_str(&title);
    }
    Ok(format!("{} — {}", py_str(&title)?, artist))
}

/// The post-removal renumbering — the tail of
/// `TrackTable.remove_track` from `ui/widgets/track_table.py`: every
/// remaining row of the backing list is re-stamped `_original_index` =
/// its new position (mutated in place, in list order), and the answer
/// carries the visible list's rebuilt `_filtered_map` plus each visible
/// row's `#` cell text (`str(original + 1)`) for the cell refresh pass,
/// so the widget only feeds Textual's `update_cell`.
///
/// A non-dict row raises `TypeError` where the reference's
/// `t["_original_index"] = ...` would raise — `load_tracks` wraps every
/// track in a fresh dict, so the shape is unreachable through the
/// table's own paths.
#[pyfunction]
pub(crate) fn track_table_reindex_after_removal<'py>(
    py: Python<'py>,
    all_tracks: &Bound<'py, PyList>,
    tracks: &Bound<'py, PyList>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>)> {
    for (i, track) in all_tracks.iter().enumerate() {
        let dict = track
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("expected a dict track"))?;
        dict.set_item("_original_index", i as i64)?;
    }
    let mut filtered_map: Vec<i64> = Vec::with_capacity(tracks.len());
    let mut index_cells: Vec<String> = Vec::with_capacity(tracks.len());
    for track in tracks.iter() {
        let dict = track
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("expected a dict track"))?;
        let orig: i64 = dict
            .get_item("_original_index")?
            .ok_or_else(|| PyKeyError::new_err("_original_index"))?
            .extract()?;
        filtered_map.push(orig);
        index_cells.push((orig + 1).to_string());
    }
    Ok((
        PyList::new(py, filtered_map)?,
        PyList::new(py, index_cells)?,
    ))
}

/// The header-click sort toggle — `TrackTable.sort_by`'s state decision
/// from `ui/widgets/track_table.py`: a column outside the sortable set
/// answers `None` (no sort happens); the already-sorted column flips
/// the direction; a new column starts ascending. The empty-table gate
/// that precedes the decision stays widget-side, where it reads the
/// live table.
#[pyfunction]
#[pyo3(signature = (column, sort_column, sort_reverse))]
pub(crate) fn track_table_sort_state(
    column: &str,
    sort_column: Option<String>,
    sort_reverse: bool,
) -> Option<(String, bool)> {
    const SORTABLE: [&str; 5] = ["index", "title", "artist", "album", "duration"];
    if !SORTABLE.contains(&column) {
        return None;
    }
    if sort_column.as_deref() == Some(column) {
        return Some((column.to_string(), !sort_reverse));
    }
    Some((column.to_string(), false))
}

/// The column keys of the composed cells — `_plain_cells`' key list from
/// `ui/widgets/track_table.py`, mapping the `_row_values` engine's
/// positional cells onto the table's own column order: `mark`, then
/// `index` when shown, `title`/`artist`, `album` when shown, then
/// `duration`.
#[pyfunction]
pub(crate) fn track_table_column_keys(show_index: bool, show_album: bool) -> Vec<String> {
    let mut keys = vec!["mark".to_string()];
    if show_index {
        keys.push("index".to_string());
    }
    keys.push("title".to_string());
    keys.push("artist".to_string());
    if show_album {
        keys.push("album".to_string());
    }
    keys.push("duration".to_string());
    keys
}

// ---------------------------------------------------------------------------
// Track-table mark/range selection engine
// (mirrors ui/widgets/track_table.py's selection core)
// ---------------------------------------------------------------------------

/// The post-removal shift's answer — the new
/// `(marked, range_base, anchor, range_mode, bump)` quintuple.
type TrackSelectionShift<'py> = (
    Bound<'py, PySet>,
    Bound<'py, PyFrozenSet>,
    Option<i64>,
    bool,
    bool,
);

/// The reload carry's answer — the resolved
/// `(marked, range_base, anchor, cursor, dropped)` quintuple.
type TrackSelectionCarry<'py> = (
    Bound<'py, PySet>,
    Bound<'py, PyFrozenSet>,
    Option<i64>,
    Option<i64>,
    bool,
);

/// `filtered_map.index(value)` — the first visible position whose
/// original is *value* (int equality; the map's own invariant is plain
/// ints), or `None` where CPython's `list.index` would raise
/// `ValueError`. A non-int map entry compares unequal, exactly like the
/// reference's `==` scan over the list.
fn filtered_map_index_of(filtered_map: &Bound<'_, PyList>, value: i64) -> Option<usize> {
    for (row, item) in filtered_map.iter().enumerate() {
        if item
            .extract::<i64>()
            .map(|orig| orig == value)
            .unwrap_or(false)
        {
            return Some(row);
        }
    }
    None
}

/// The visible-run core — `TrackTable._run` from
/// `ui/widgets/track_table.py`: the original indices of the visible rows
/// from *start* to *end*, inclusive. The two occurrences resolve through
/// `list.index` (the first row carrying each original) and order through
/// `sorted`, so the run reads the same whichever end the drag started
/// from; a *start* the filter hides counts as no start — the run is
/// *end* alone — exactly like the reference's swallowed `ValueError`.
/// The slice's own items join the set through CPython's `set(...)`
/// construction, so an (unreachable) duplicate original dedups the way
/// the reference's set comprehension would.
fn visible_run_impl<'py>(
    py: Python<'py>,
    filtered_map: &Bound<'py, PyList>,
    start: i64,
    end: i64,
) -> PyResult<Bound<'py, PySet>> {
    let run = PySet::empty(py)?;
    let pos_start = filtered_map_index_of(filtered_map, start);
    let pos_end = filtered_map_index_of(filtered_map, end);
    if let (Some(a), Some(b)) = (pos_start, pos_end) {
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        for item in filtered_map.iter().skip(lo).take(hi - lo + 1) {
            run.add(item)?;
        }
    } else {
        // {end} — the ValueError arm: an invisible row is never marked
        // through a run.
        run.add(
            end.into_pyobject(py)
                .expect("int conversion is infallible")
                .into_any(),
        )?;
    }
    Ok(run)
}

/// The visible-run engine — `ui/widgets/track_table.py`'s `_run`, the
/// run core behind the shift-click extend (`_extend_to`) and the
/// keyboard range marks (`_apply_range`). See `visible_run_impl` for the
/// mirrored contract; the answer is the set of original indices the
/// caller unions into its marks.
#[pyfunction]
#[pyo3(signature = (filtered_map, start, end))]
pub(crate) fn track_table_visible_run<'py>(
    py: Python<'py>,
    filtered_map: &Bound<'py, PyList>,
    start: i64,
    end: i64,
) -> PyResult<Bound<'py, PySet>> {
    visible_run_impl(py, filtered_map, start, end)
}

/// A fresh set seeded from an iterable — `set(obj)`, so the caller's own
/// set is never mutated by the union the engines build on top of it.
fn set_from_iterable<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PySet>> {
    let out = PySet::empty(py)?;
    for item in obj.try_iter()? {
        out.add(item?)?;
    }
    Ok(out)
}

/// The shift-click extend decision — `TrackTable._extend_to` from
/// `ui/widgets/track_table.py`: with no anchor, or an anchor the filter
/// hides (the `in filtered_map` list containment), the row is marked and
/// becomes the anchor — a run never reaches a row that isn't shown —
/// while an anchored view marks the visible run from the anchor to the
/// row, existing marks outside the run staying. Answers the new
/// `(anchor, marked)` pair; the repaint (`_set_marks`) stays
/// widget-side.
#[pyfunction]
#[pyo3(signature = (anchor, filtered_map, marked, original))]
pub(crate) fn track_table_extend_marks<'py>(
    py: Python<'py>,
    anchor: Option<i64>,
    filtered_map: &Bound<'py, PyList>,
    marked: &Bound<'py, PyAny>,
    original: i64,
) -> PyResult<(Option<i64>, Bound<'py, PySet>)> {
    // set(marked) | ... — the union seeded from a copy, so the caller's
    // set is untouched.
    let marks = set_from_iterable(py, marked)?;
    match anchor {
        Some(a) if filtered_map_index_of(filtered_map, a).is_some() => {
            let run = visible_run_impl(py, filtered_map, a, original)?;
            for item in run.iter() {
                marks.add(item)?;
            }
            Ok((Some(a), marks))
        }
        _ => {
            marks.add(
                original
                    .into_pyobject(py)
                    .expect("int conversion is infallible")
                    .into_any(),
            )?;
            Ok((Some(original), marks))
        }
    }
}

/// The keyboard-range marks — `TrackTable._apply_range`'s decision from
/// `ui/widgets/track_table.py`: range mode's marks are the baseline
/// (`range_base`) plus the anchor-to-cursor run — an empty run when the
/// cursor is off the table (the whole displayed run between the anchor
/// row and the highlighted row). No anchor answers `None` and the caller
/// leaves the marks alone. The generation bump and the repaint stay
/// widget-side.
#[pyfunction]
#[pyo3(signature = (anchor, cursor, filtered_map, range_base))]
pub(crate) fn track_table_range_marks<'py>(
    py: Python<'py>,
    anchor: Option<i64>,
    cursor: Option<i64>,
    filtered_map: &Bound<'py, PyList>,
    range_base: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PySet>>> {
    // if self._anchor is None: return
    let Some(anchor) = anchor else {
        return Ok(None);
    };
    // set(self._range_base) | run — a fresh set seeded from the baseline.
    let marks = set_from_iterable(py, range_base)?;
    // run = set() if cursor is None else self._run(anchor, cursor)
    if let Some(cursor) = cursor {
        let run = visible_run_impl(py, filtered_map, anchor, cursor)?;
        for item in run.iter() {
            marks.add(item)?;
        }
    }
    Ok(Some(marks))
}

/// The toggle's marked set — `TrackTable._toggle_mark`'s symmetric
/// difference from `ui/widgets/track_table.py`: the caller's marks with
/// *original*'s membership flipped, built as a fresh set. The anchor
/// assignment and the repaint stay widget-side.
#[pyfunction]
#[pyo3(signature = (marked, original))]
pub(crate) fn track_table_toggle_marks<'py>(
    py: Python<'py>,
    marked: &Bound<'py, PyAny>,
    original: i64,
) -> PyResult<Bound<'py, PySet>> {
    // marked ^ {original} — the copy with the membership flipped.
    let marks = set_from_iterable(py, marked)?;
    let original = original
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    if marks.contains(&original)? {
        marks.discard(&original)?;
    } else {
        marks.add(original)?;
    }
    Ok(marks)
}

/// The post-removal selection shift —
/// `TrackTable._shift_selection_after_removal` from
/// `ui/widgets/track_table.py`: when the occurrence *removed* leaves
/// the list, every mark above it slides down one and the removed mark
/// itself goes (the bump flag answers whether the removed occurrence
/// was marked, so the caller advances the selection generation exactly
/// when the marked set actually shrank). The range baseline shifts the
/// same way; the anchor dies with its occurrence (an un-anchored range
/// mode drops out and empties the baseline). The answers are the new
/// `(marked, range_base, anchor, range_mode, bump)` quintuple, all
/// built fresh; the repaint stays widget-side.
#[pyfunction]
#[pyo3(signature = (removed, marked, range_base, anchor, range_mode))]
pub(crate) fn track_table_shift_selection<'py>(
    py: Python<'py>,
    removed: i64,
    marked: &Bound<'py, PyAny>,
    range_base: &Bound<'py, PyAny>,
    anchor: Option<i64>,
    range_mode: bool,
) -> PyResult<TrackSelectionShift<'py>> {
    // if removed in self._marked — containment over the caller's set.
    let removed_obj = removed
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let bump = marked.contains(&removed_obj)?;

    // {shift(i) for i in marked if i != removed} — the shift being
    // `original - 1 if original > removed else original`.
    let new_marked = PySet::empty(py)?;
    for item in marked.try_iter()? {
        let item = item?;
        let orig: i64 = item
            .extract()
            .map_err(|_| PyTypeError::new_err("expected int marked indices"))?;
        if orig == removed {
            continue;
        }
        new_marked.add(
            (if orig > removed { orig - 1 } else { orig })
                .into_pyobject(py)
                .expect("int conversion is infallible"),
        )?;
    }

    // frozenset(shift(i) for i in range_base if i != removed)
    let mut shifted_base: Vec<i64> = Vec::new();
    for item in range_base.try_iter()? {
        let item = item?;
        let orig: i64 = item
            .extract()
            .map_err(|_| PyTypeError::new_err("expected int range-base indices"))?;
        if orig == removed {
            continue;
        }
        shifted_base.push(if orig > removed { orig - 1 } else { orig });
    }

    // self._anchor = None if removed else shift(self._anchor)
    let new_anchor = match anchor {
        None => None,
        Some(a) if a == removed => None,
        Some(a) => Some(if a > removed { a - 1 } else { a }),
    };

    // An un-anchored range mode drops out and empties the baseline.
    let new_range_mode;
    let new_base = if new_anchor.is_none() && range_mode {
        new_range_mode = false;
        PyFrozenSet::empty(py)?
    } else {
        new_range_mode = range_mode;
        PyFrozenSet::new(py, shifted_base)?
    };

    Ok((new_marked, new_base, new_anchor, new_range_mode, bump))
}

/// The marked-tracks walk — `TrackTable.marked_tracks` from
/// `ui/widgets/track_table.py`: the marked tracks in the displayed
/// order, hidden (filtered-out) ones included — under a sort the whole
/// backing list is ordered by that sort (`filter_sort_tracks` with the
/// empty query), so a marked track the filter hides lands where it
/// would show without the filter; otherwise the load order applies. The
/// answer holds the backing list's own track objects (no copies), the
/// mark test reading each row's `_original_index` through the dict's
/// own lookup (`KeyError` where missing) and the containment running
/// through CPython's protocol over *marked*.
#[pyfunction]
#[pyo3(signature = (all_tracks, marked, sort_column, sort_reverse))]
pub(crate) fn track_table_marked_tracks<'py>(
    py: Python<'py>,
    all_tracks: &Bound<'py, PyList>,
    marked: &Bound<'py, PyAny>,
    sort_column: Option<&str>,
    sort_reverse: bool,
) -> PyResult<Bound<'py, PyList>> {
    // if not self._marked: return []
    if !marked.is_truthy()? {
        return Ok(PyList::empty(py));
    }
    let positions = filter_sort_tracks(py, all_tracks, "", sort_column, sort_reverse);
    let out = PyList::empty(py);
    for i in positions {
        let track = all_tracks.get_item(i)?;
        // track["_original_index"] in marked
        let dict = track
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("expected a dict track"))?;
        let orig = dict
            .get_item("_original_index")?
            .ok_or_else(|| PyKeyError::new_err("_original_index"))?;
        if marked.contains(&orig)? {
            out.append(track)?;
        }
    }
    Ok(out)
}

/// The hidden-marks count — `TrackTable.hidden_marked_count` from
/// `ui/widgets/track_table.py`: the marked rows the active filter
/// currently hides, i.e. the marked originals absent from the visible
/// map. Both the set construction (`set(filtered_map)`) and the
/// membership run through CPython's own protocol.
#[pyfunction]
#[pyo3(signature = (marked, filtered_map))]
pub(crate) fn track_table_hidden_marked_count(
    py: Python<'_>,
    marked: &Bound<'_, PyAny>,
    filtered_map: &Bound<'_, PyList>,
) -> PyResult<i64> {
    // visible = set(self._filtered_map)
    let visible = PySet::new(py, filtered_map.iter())?;
    // sum(1 for i in self._marked if i not in visible)
    let mut hidden: i64 = 0;
    for item in marked.try_iter()? {
        if !visible.contains(&item?)? {
            hidden += 1;
        }
    }
    Ok(hidden)
}

/// `old_keys[index]` with CPython's own negative-wrap semantics — an
/// in-range negative counts from the end, anything else answers the
/// `IndexError` the reference's subscript would raise.
fn py_list_get_item_i64<'py>(list: &Bound<'py, PyList>, index: i64) -> PyResult<Bound<'py, PyAny>> {
    let len = list.len() as i64;
    let effective = if index < 0 { index + len } else { index };
    if effective < 0 || effective >= len {
        return Err(PyIndexError::new_err("list index out of range"));
    }
    list.get_item(effective as usize)
}

/// The occurrence-key carry — the pure resolution core of
/// `TrackTable.refresh_tracks` from `ui/widgets/track_table.py`: which
/// occurrences of a reloaded list the selection's marks, range baseline,
/// anchor and cursor carried over to.
///
/// The old list's keys count through a `Counter` (a dict keyed by the
/// key objects themselves, so hash/eq semantics decide identity), and
/// `key_at` names an occurrence by its key only when that key is
/// unique in the old list — a duplicated key resolves nothing, so a
/// mark on it is dropped rather than moved to another row. The new
/// list's keys count the same way, and `position` maps each key unique
/// in the new list to its index. `resolve` then carries each candidate
/// key to its new position, skipping `None` (the duplicated-old-key
/// case) and keys missing from or duplicated in the new list.
///
/// Answers the `(marked, range_base, anchor, cursor, dropped)`
/// quintuple: the marked set and range baseline resolved onto the new
/// list (the baseline as a frozenset), the anchor and cursor as single
/// positions or `None`, and *dropped* — true when any marked key failed
/// to resolve (the reference's `len(resolved) < len(marked_keys)`), the
/// caller's signal to advance the selection generation. The reload
/// itself, the view rebuild, the cursor placement and the generation
/// arithmetic stay widget-side.
#[pyfunction]
#[pyo3(signature = (old_keys, new_keys, marked, range_base, anchor, cursor))]
pub(crate) fn track_table_carry_selection_keys<'py>(
    py: Python<'py>,
    old_keys: &Bound<'py, PyList>,
    new_keys: &Bound<'py, PyList>,
    marked: &Bound<'py, PyAny>,
    range_base: &Bound<'py, PyAny>,
    anchor: Option<i64>,
    cursor: Option<i64>,
) -> PyResult<TrackSelectionCarry<'py>> {
    // Counter(old_keys)
    let old_counts = PyDict::new(py);
    for key in old_keys.iter() {
        let count: i64 = match old_counts.get_item(&key)? {
            Some(c) => c.extract()?,
            None => 0,
        };
        old_counts.set_item(&key, count + 1)?;
    }

    // key_at(index) — the occurrence's key when unique in the old list.
    let key_at = |index: i64| -> PyResult<Option<Bound<'py, PyAny>>> {
        let key = py_list_get_item_i64(old_keys, index)?;
        let count: i64 = old_counts
            .get_item(&key)?
            .map(|c| c.extract())
            .transpose()?
            .unwrap_or(0);
        Ok(if count == 1 { Some(key) } else { None })
    };

    // marked_keys / base_keys / anchor_key / cursor_key
    let mut marked_keys: Vec<Option<Bound<'py, PyAny>>> = Vec::new();
    for item in marked.try_iter()? {
        marked_keys.push(key_at(
            item?
                .extract()
                .map_err(|_| PyTypeError::new_err("expected int marked indices"))?,
        )?);
    }
    let mut base_keys: Vec<Option<Bound<'py, PyAny>>> = Vec::new();
    for item in range_base.try_iter()? {
        base_keys.push(key_at(item?.extract().map_err(|_| {
            PyTypeError::new_err("expected int range-base indices")
        })?)?);
    }
    let anchor_key = match anchor {
        Some(a) => key_at(a)?,
        None => None,
    };
    let cursor_key = match cursor {
        Some(c) => key_at(c)?,
        None => None,
    };

    // Counter(keys) + position = {k: i for i, k in enumerate(keys) if
    // new_counts[k] == 1}
    let new_counts = PyDict::new(py);
    for key in new_keys.iter() {
        let count: i64 = match new_counts.get_item(&key)? {
            Some(c) => c.extract()?,
            None => 0,
        };
        new_counts.set_item(&key, count + 1)?;
    }
    let position = PyDict::new(py);
    for (i, key) in new_keys.iter().enumerate() {
        let count: i64 = new_counts
            .get_item(&key)?
            .map(|c| c.extract())
            .transpose()?
            .unwrap_or(0);
        if count == 1 {
            position.set_item(&key, i as i64)?;
        }
    }

    // resolve(candidates) — the new positions of the keys that resolve.
    let resolve = |candidates: &[Option<Bound<'py, PyAny>>]| -> PyResult<Bound<'py, PySet>> {
        let out = PySet::empty(py)?;
        for candidate in candidates.iter().flatten() {
            if let Some(pos) = position.get_item(candidate)? {
                out.add(pos)?;
            }
        }
        Ok(out)
    };

    let resolved = resolve(&marked_keys)?;
    // dropped = len(resolved) < len(marked_keys)
    let dropped = resolved.len() < marked_keys.len();
    let new_base = PyFrozenSet::new(py, resolve(&base_keys)?.iter())?;
    // next(iter(resolve([anchor_key])), None) — a zero-or-one-element
    // resolution, so the single answer or None.
    let new_anchor = match &anchor_key {
        Some(key) => match position.get_item(key)? {
            Some(pos) => Some(pos.extract()?),
            None => None,
        },
        None => None,
    };
    let new_cursor = match &cursor_key {
        Some(key) => match position.get_item(key)? {
            Some(pos) => Some(pos.extract()?),
            None => None,
        },
        None => None,
    };

    Ok((resolved, new_base, new_anchor, new_cursor, dropped))
}
