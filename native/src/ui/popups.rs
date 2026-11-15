//! Native cores for the popups — `ui/popups/actions.py`,
//! `confirm_popup.py`, `input_popup.py`, `create_playlist_popup.py`,
//! `playlist_picker.py` and `country_picker.py`: action lists, chrome
//! labels, picker sorting/rows and region filtering.

use crate::services::ytmusic::py_unpack_pair;
use crate::ui::pages::browse::field_get;
use crate::ui::pages::browse::first_truthy_get;
use crate::ui::pages::help::py_capitalize;
use crate::ui::sidebars::python_strip;
use crate::utils::pycompat::py_isspace;
use crate::utils::pycompat::python_lower;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Actions-popup menu builder
// ---------------------------------------------------------------------------

/// The track catalog: `(action_id, label)` in dispatch order. Mirrors
/// `TRACK_ACTIONS` from `ui/popups/actions.py` — static data on both
/// backends, pinned together by the parity tests.
const ACTIONS_TRACK: &[(&str, &str)] = &[
    ("play", "Play"),
    ("play_next", "Play Next"),
    ("add_to_queue", "Add to Queue"),
    ("download", "Download for Offline"),
    ("start_radio", "Start Radio"),
    ("go_to_artist", "Go to Artist"),
    ("go_to_album", "Go to Album"),
    ("add_to_playlist", "Add to Playlist"),
    ("remove_from_playlist", "Remove from Playlist"),
    ("toggle_like", "Like"),
    ("copy_link", "Copy Link"),
];

/// The album catalog. Mirrors `ALBUM_ACTIONS` from `ui/popups/actions.py`.
const ACTIONS_ALBUM: &[(&str, &str)] = &[
    ("play_all", "Play All"),
    ("shuffle_play", "Shuffle Play"),
    ("add_to_library", "Add to Library"),
    ("play_next", "Play Next"),
    ("add_to_queue", "Add to Queue"),
    ("go_to_artist", "Go to Artist"),
    ("copy_link", "Copy Link"),
];

/// The artist catalog. Mirrors `ARTIST_ACTIONS` from `ui/popups/actions.py`.
const ACTIONS_ARTIST: &[(&str, &str)] = &[
    ("go_to_artist", "Go to Artist"),
    ("play_top_songs", "Play Top Songs"),
    ("start_radio", "Start Radio"),
    ("toggle_subscribe", "Subscribe"),
    ("view_similar", "View Similar Artists"),
    ("copy_link", "Copy Link"),
];

/// The playlist catalog. Mirrors `PLAYLIST_ACTIONS` from
/// `ui/popups/actions.py`.
const ACTIONS_PLAYLIST: &[(&str, &str)] = &[
    ("play_all", "Play All"),
    ("shuffle_play", "Shuffle Play"),
    ("play_next", "Play Next"),
    ("add_to_queue", "Add to Queue"),
    ("start_radio", "Start Radio"),
    ("copy_link", "Copy Link"),
    ("edit", "Edit Playlist"),
    ("delete", "Delete Playlist"),
];

/// The context-menu assembly behind `ActionsPopup` — the pure core of
/// `_build_actions` from `ui/popups/actions.py`. Walks the *item_type*'s
/// catalog in dispatch order, arm for arm like the reference:
///
/// - `add_to_queue` becomes `("remove_from_queue", "Remove from Queue")`
///   when *in_queue* is set (the swap runs before the gates, so the
///   replaced id is what the later checks see — a no-op here, but the
///   reference's order).
/// - `remove_from_playlist` is dropped unless *in_playlist* is set.
/// - `toggle_like` reads `item.get("likeStatus") == "LIKE"` (Python `==`,
///   exotic objects steering) `or item.get("liked", False)` for its
///   `Unlike`/`Like` label.
/// - `toggle_subscribe` reads truthy `item.get("subscribed", False)` for
///   its `Unsubscribe`/`Subscribe` label.
/// - `go_to_artist` needs a truthy `artists` or a truthy `artist`.
/// - `go_to_album` needs a truthy `album_id`, or an `album` that is a
///   dict (subclasses included) with a truthy `id` — the `.get` runs
///   only when the isinstance gate passes, like the reference's `and`.
/// - `delete` is dropped for playlists when *source* is `"search"`.
///
/// Field reads resolve exactly like the reference's `.get` calls (see
/// `field_get`); a `.get`-less item reads as field-less where the
/// reference would raise — the engine stays total (the `filter_sort_tracks`
/// precedent). A fresh list of fresh tuples comes back on every call;
/// the item is never mutated.
#[pyfunction]
#[pyo3(signature = (item, item_type, *, in_queue = false, in_playlist = false, source = "default"))]
pub(crate) fn build_actions<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
    item_type: &str,
    in_queue: bool,
    in_playlist: bool,
    source: &str,
) -> PyResult<Vec<(String, String)>> {
    // base = list(_ACTIONS_BY_TYPE.get(item_type, TRACK_ACTIONS))
    let base = match item_type {
        "album" => ACTIONS_ALBUM,
        "artist" => ACTIONS_ARTIST,
        "playlist" => ACTIONS_PLAYLIST,
        // "track" and every unknown type alike — the dict lookup's default.
        _ => ACTIONS_TRACK,
    };
    let false_default = false.into_pyobject(py)?.to_owned().into_any();
    let mut out: Vec<(String, String)> = Vec::with_capacity(base.len());
    for (base_id, base_label) in base {
        let mut action_id = *base_id;
        let mut label = *base_label;
        // if action_id == "add_to_queue" and in_queue: swap both halves.
        if action_id == "add_to_queue" && in_queue {
            action_id = "remove_from_queue";
            label = "Remove from Queue";
        }
        // if action_id == "remove_from_playlist" and not in_playlist: continue
        if action_id == "remove_from_playlist" && !in_playlist {
            continue;
        }
        // if action_id == "toggle_like": Unlike / Like.
        if action_id == "toggle_like" {
            let like_status = field_get(py, item, "likeStatus", None)?;
            let liked = field_get(py, item, "liked", Some(false_default.clone()))?;
            let is_liked = like_status.eq("LIKE")? || liked.is_truthy()?;
            label = if is_liked { "Unlike" } else { "Like" };
        }
        // if action_id == "toggle_subscribe": Unsubscribe / Subscribe.
        if action_id == "toggle_subscribe" {
            let subscribed = field_get(py, item, "subscribed", Some(false_default.clone()))?;
            label = if subscribed.is_truthy()? {
                "Unsubscribe"
            } else {
                "Subscribe"
            };
        }
        // if action_id == "go_to_artist": needs an artist to go to.
        if action_id == "go_to_artist" {
            // artists = item.get("artists") or []
            let artists = field_get(py, item, "artists", None)?;
            let artist = field_get(py, item, "artist", None)?;
            if !artists.is_truthy()? && !artist.is_truthy()? {
                continue;
            }
        }
        // if action_id == "go_to_album": needs an album to go to.
        if action_id == "go_to_album" {
            let album_id = field_get(py, item, "album_id", None)?;
            let album = field_get(py, item, "album", None)?;
            // isinstance(album, dict) and album.get("id") — the second
            // read only runs once the first gate passes.
            let album_has_id = album.is_instance_of::<PyDict>()
                && field_get(py, &album, "id", None)?.is_truthy()?;
            if !album_id.is_truthy()? && !album_has_id {
                continue;
            }
        }
        // if action_id == "delete" and source == "search" and
        // item_type == "playlist": continue
        if action_id == "delete" && source == "search" && item_type == "playlist" {
            continue;
        }
        out.push((action_id.to_string(), label.to_string()));
    }
    Ok(out)
}

/// The title derivation behind `ActionsPopup`'s header — the pure core
/// of `_title_text` from `ui/popups/actions.py`: the first truthy of
/// `item.get("title")` / `name` / `artist` (each read exactly like the
/// reference's `.get` calls — see `field_get`; a `.get`-less item reads
/// as field-less where the reference would raise, the `build_actions`
/// totality precedent), the capitalized *item_type* when none of the
/// three is truthy (`py_capitalize`, a faithful `str.capitalize()`),
/// and the 34-character truncation — a name longer than 34 characters
/// (CPython's own `len()`) becomes its first 31 characters plus
/// `"..."`.
///
/// The truncation runs through CPython's own protocol so every shape
/// the reference survives, the engine survives identically: an exact
/// `str` takes the code-point fast path (returning the very same
/// object when it fits, a fresh string when it truncates), while a
/// `str` subclass, a non-UTF-8 `str` (lone surrogates) or any other
/// object flows through `len()`, `[:31]` (the `slice(None, 31, None)`
/// the reference's subscription compiles to) and `+ "..."` — the
/// number protocol, the subscript protocol and the binary-add protocol
/// exactly, so a sized exotic answers whatever its own protocol
/// answers (a 40-item list: `len` 40, a 31-item slice, then the list
/// + str `TypeError` — the reference's own), and an unsized one
/// raises `len()`'s own `TypeError` on both sides.
#[pyfunction]
pub(crate) fn actions_popup_title<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
    item_type: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // name = item.get("title") or item.get("name") or item.get("artist")
    //        or item_type.capitalize()
    let name = match first_truthy_get(py, item, &["title", "name", "artist"])? {
        Some(name) => name,
        None => PyString::new(py, &py_capitalize(item_type)).into_any(),
    };
    // if len(name) > 34: name = name[:31] + "..."
    if name.is_exact_instance_of::<PyString>() {
        let s = name.downcast::<PyString>().unwrap();
        if let Ok(contents) = s.to_str() {
            if contents.chars().count() > 34 {
                let mut truncated: String = contents.chars().take(31).collect();
                truncated.push_str("...");
                return Ok(PyString::new(py, &truncated).into_any());
            }
            // A name that fits is handed back unchanged — the very
            // same object the reference returns.
            return Ok(name.clone());
        }
        // A str PyO3 cannot borrow as UTF-8 (lone surrogates) falls
        // through to the protocol path, which handles any str content.
    }
    let length = name.len()?;
    if length > 34 {
        // slice(None, 31, None) — the descriptor `name[:31]` compiles
        // to; built through the builtins `slice` type so an overridden
        // `__getitem__` sees exactly the object the reference passes.
        let slice = py
            .import("builtins")?
            .getattr("slice")?
            .call1((py.None(), 31i64))?;
        let head = name.get_item(&slice)?;
        return head.add("...");
    }
    Ok(name.clone())
}

// ---------------------------------------------------------------------------
// The confirm / input / create-playlist popup cores
// ---------------------------------------------------------------------------

/// The dismissal verdict behind `ConfirmPopup`'s button press — the
/// pure core of `on_button_pressed` from `ui/popups/confirm_popup.py`:
/// `True` only for the confirm button's id, `False` for the cancel
/// button's, any other id, or a widget with no id at all (`Button.id`
/// is `str | None`; `None == "confirm-yes"` is `False`, CPython's own
/// `==` on the None literal). The popup's `y`/`n` key bindings and the
/// `BasePopup` cancel shell stay Python-side.
#[pyfunction]
#[pyo3(signature = (button_id))]
pub(crate) fn confirm_popup_button_result(button_id: Option<&str>) -> bool {
    button_id == Some("confirm-yes")
}

/// The submission normalization behind `InputPopup`'s submit — the pure
/// core of `on_input_submitted` from `ui/popups/input_popup.py`: the
/// value stripped (CPython's own `str.strip()` set — see
/// `py_str_strip`, wider than Rust's `str::trim`), `Some` when what
/// remains is non-empty, `None` when the submission was empty or all
/// whitespace. The popup dismisses with the answer; the Input widget,
/// the focus and the event plumbing stay Python-side.
#[pyfunction]
pub(crate) fn input_popup_submitted_value(value: &str) -> Option<String> {
    let stripped = py_str_strip(value);
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

/// The chrome labels behind `CreatePlaylistPopup`'s compose — the pure
/// core of the title / submit-button pair from
/// `ui/popups/create_playlist_popup.py`: `("Edit Playlist", "Edit")`
/// in edit mode, `("New Playlist", "Create")` otherwise. The Input /
/// Select / Button mounts stay Python-side in the popup.
#[pyfunction]
pub(crate) fn create_playlist_chrome_labels(edit_mode: bool) -> (String, String) {
    if edit_mode {
        ("Edit Playlist".to_string(), "Edit".to_string())
    } else {
        ("New Playlist".to_string(), "Create".to_string())
    }
}

/// The submission normalization behind `CreatePlaylistPopup`'s submit —
/// the pure core of `_submit` from
/// `ui/popups/create_playlist_popup.py`: the name and the description
/// stripped (CPython's own `str.strip()` set), the privacy through
/// `str()` (the protocol `__str__` call — an exact `str` hands back
/// the very same object), `None` when the stripped name is empty (the
/// popup warns and stays open; the notification and the re-focus stay
/// Python-side), otherwise the `(name, description, privacy)`
/// dismissal tuple. The name gate runs first, exactly like the
/// reference — a bad name answers `None` before the privacy's `str()`
/// is ever consulted.
#[pyfunction]
pub(crate) fn create_playlist_submit_values<'py>(
    name: &str,
    description: &str,
    privacy: &Bound<'py, PyAny>,
) -> PyResult<Option<(String, String, Bound<'py, PyString>)>> {
    let name = py_str_strip(name);
    if name.is_empty() {
        return Ok(None);
    }
    let description = py_str_strip(description);
    let privacy = privacy.str()?;
    Ok(Some((name.to_string(), description.to_string(), privacy)))
}

// ---------------------------------------------------------------------------
// Playlist-picker view model (mirrors ui/popups/playlist_picker.py)
// ---------------------------------------------------------------------------

/// `pl.get("playlistId", pl.get("id", ""))` — the picker's playlist-id
/// read: the inner `.get("id", "")` evaluates first (argument order), so
/// a missing `playlistId` falls back to `id`, and both missing give `""`.
/// Uses the shared `field_get` (a `.get`-less item reads as the default —
/// the engine stays total where the reference would raise).
fn playlist_pid<'py>(py: Python<'py>, pl: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let fallback = field_get(py, pl, "id", Some(PyString::new(py, "").into_any()))?;
    field_get(py, pl, "playlistId", Some(fallback))
}

/// The recent-list move-to-front behind `_record_recent` — the pure core
/// of the picker's recently-used bookkeeping.
///
/// Mirrors the pure-Python reference: `if playlist_id in recent:
/// recent.remove(playlist_id)` — the FIRST equal occurrence is dropped —
/// then `recent.insert(0, playlist_id)`, so the result is the playlist id
/// followed by the recent list minus its first equal entry. A fresh list
/// comes back; the input is never mutated. The membership test compares
/// str values (a str subclass entry with content equal to *playlist_id*
/// matches, exactly as the reference's `==` does); a non-str entry — junk
/// from a corrupted recent-playlists file — never matches and passes
/// through untouched, where the reference's `==` against JSON-shaped
/// junk (numbers, nulls, lists) also never matches. The load/save around
/// this core (the JSON round-trip and the MAX_RECENT cap applied at
/// save time) stays on the Python side.
#[pyfunction]
#[pyo3(signature = (recent, playlist_id))]
pub(crate) fn playlist_record_recent<'py>(
    py: Python<'py>,
    recent: &Bound<'py, PyList>,
    playlist_id: &str,
) -> PyResult<Bound<'py, PyList>> {
    let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(recent.len() + 1);
    items.push(PyString::new(py, playlist_id).into_any());
    let mut removed = false;
    for item in recent.iter() {
        if !removed {
            if let Ok(entry) = item.downcast::<PyString>() {
                if entry.to_string_lossy() == playlist_id {
                    // list.remove drops exactly the first occurrence.
                    removed = true;
                    continue;
                }
            }
        }
        items.push(item);
    }
    PyList::new(py, items)
}

/// The recency sort behind `PlaylistPicker._sort_by_recent` — the pure
/// core of the picker's list ordering.
///
/// Mirrors the pure-Python reference: no recent ids means no sort at all
/// (the playlist order is whatever the fetch produced); otherwise the
/// key is `(0, recent_order[pid])` for a playlist whose id is in the
/// recent list and `(1, 0)` for one that is not, and the sort is stable
/// (equal keys keep fetch order — `sort_by` is Rust's stable sort, the
/// reference's `list.sort`). `recent_order` is the reference's dict
/// comprehension, so a duplicated recent id counts at its LAST position.
/// The id is read through `playlist_pid` (`playlistId`, then `id`, then
/// `""`); a str id (subclasses included) matches by content, any other
/// id — and any non-str recent entry, junk a corrupted file could hold —
/// never matches, where the reference's set membership would also never
/// match same-shaped JSON junk (and would raise on an unhashable entry:
/// the engine stays total instead). A fresh list comes back; the input
/// is never mutated.
#[pyfunction]
#[pyo3(signature = (playlists, recent_ids))]
pub(crate) fn playlist_picker_sort<'py>(
    py: Python<'py>,
    playlists: &Bound<'py, PyList>,
    recent_ids: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    // `if not recent_ids: return` — an empty recent list leaves the
    // playlist order untouched.
    if recent_ids.is_empty() {
        let items: Vec<Bound<'py, PyAny>> = playlists.iter().collect();
        return PyList::new(py, items);
    }

    // {pid: i for i, pid in enumerate(recent_ids)} — the LAST position of
    // a duplicated id wins, exactly like the dict comprehension.
    let mut recent_order: HashMap<String, usize> = HashMap::with_capacity(recent_ids.len());
    for (i, item) in recent_ids.iter().enumerate() {
        if let Ok(entry) = item.downcast::<PyString>() {
            recent_order.insert(entry.to_string_lossy().into_owned(), i);
        }
    }

    // Decorate with (0, order) / (1, 0), stable-sort, undecorate.
    let mut decorated: Vec<((u8, usize), Bound<'py, PyAny>)> = Vec::with_capacity(playlists.len());
    for pl in playlists.iter() {
        let pid = playlist_pid(py, &pl)?;
        let key = match pid.downcast::<PyString>() {
            Ok(entry) => {
                let pid_str = entry.to_string_lossy();
                match recent_order.get(pid_str.as_ref()) {
                    // (0, recent_order[pid]) — the recency rank.
                    Some(&order) => (0u8, order),
                    // (1, 0) — not recently used.
                    None => (1u8, 0usize),
                }
            }
            Err(_) => (1u8, 0usize),
        };
        decorated.push((key, pl));
    }
    // sort_by_key is stable: equal keys keep fetch order, like list.sort.
    decorated.sort_by_key(|entry| entry.0);

    let items: Vec<Bound<'py, PyAny>> = decorated.into_iter().map(|(_, pl)| pl).collect();
    PyList::new(py, items)
}

/// The visible-row walk behind `PlaylistPicker._populate_list` — the pure
/// core of the picker's list rebuild.
///
/// Mirrors the pure-Python reference: the query is
/// `filter_text.strip().lower()` (stripped FIRST, then lowercased — the
/// opposite order of the help page's needle); an empty query shows every
/// playlist, a non-empty one only those whose title lowercased contains
/// it; and each surviving playlist contributes the row
/// `(playlist_id, title, count)` — the id through `playlist_pid`
/// (`playlistId`, then `id`, then `""`, the object it is), the title with
/// its `"Untitled"` default (the object it is), and the count as `""`
/// when absent or falsy, `str(count)` when truthy (CPython's own `str()`
/// — `__str__` and all — through the binding, so exotic counts render
/// exactly as the reference renders them). A non-str title survives an
/// empty query (the reference's `query and ...` short-circuits before
/// `.lower()` runs) and is skipped by a non-empty one (where the
/// reference would raise `AttributeError` — the engine stays total).
/// The `"Create New"` sentinel row, the status line and the ListView
/// calls stay on the Python side.
#[pyfunction]
#[pyo3(signature = (playlists, filter_text))]
pub(crate) fn playlist_picker_rows<'py>(
    py: Python<'py>,
    playlists: &Bound<'py, PyList>,
    filter_text: &str,
) -> PyResult<Bound<'py, PyList>> {
    // filter_text.strip().lower() — strip first, then lower.
    let query = python_lower(python_strip(filter_text));
    let mut rows: Vec<Bound<'py, PyAny>> = Vec::with_capacity(playlists.len());
    for pl in playlists.iter() {
        let title = field_get(
            py,
            &pl,
            "title",
            Some(PyString::new(py, "Untitled").into_any()),
        )?;
        // `if query and query not in title.lower(): continue` — the
        // short-circuit keeps a non-str title through an empty query.
        let matches = if query.is_empty() {
            true
        } else {
            match title.downcast::<PyString>() {
                Ok(t) => python_lower(&t.to_string_lossy()).contains(query.as_str()),
                Err(_) => false,
            }
        };
        if !matches {
            continue;
        }
        let playlist_id = playlist_pid(py, &pl)?;
        let raw_count = field_get(py, &pl, "count", Some(PyString::new(py, "").into_any()))?;
        // `if track_count: track_count = str(track_count)` — the object
        // itself when falsy, CPython's str() of it when truthy.
        let count = if raw_count.is_truthy()? {
            raw_count.str()?.into_any()
        } else {
            raw_count
        };
        let row = (playlist_id, title, count).into_pyobject(py)?;
        rows.push(row.into_any());
    }
    PyList::new(py, rows)
}

/// The playlist-count status label behind `PlaylistPicker._populate_list`
/// — the pure core of the picker's status line: `f"{count}
/// playlist{'s' if count != 1 else ''}"`, so `"1 playlist"`, `"2
/// playlists"`, and the empty-list `"0 playlists"` (the label the
/// no-playlists branch never shows — the count branch owns it) all
/// pluralise exactly as the reference's f-string does.
#[pyfunction]
pub(crate) fn playlist_picker_count_label(count: usize) -> String {
    let plural = if count == 1 { "" } else { "s" };
    format!("{count} playlist{plural}")
}

// ---------------------------------------------------------------------------
// Album-art render engines
// ---------------------------------------------------------------------------

/// Python's `str.strip()`: trims the characters CPython's `str.isspace()`
/// accepts from both ends. Rust's `str::trim()` (the Unicode `White_Space`
/// property) would disagree on `\x1c`–`\x1f`, so the acceptance set is
/// `py_isspace`'s, not the property's.
pub(crate) fn py_str_strip(s: &str) -> &str {
    s.trim_matches(|c: char| py_isspace(c))
}

// ── Region-picker filter (ui/popups/country_picker.py) ───────────────────

/// The chart-region picker's filter pass —
/// `ui/popups/country_picker.py`'s `filter_regions`, behind the
/// picker's `on_input_changed`.
///
/// Mirrors the reference: the query stripped and lowercased (Python's
/// per-code-point mapping — see `python_lower`; CPython's own strip
/// set — see `py_strip`), an empty query handing the *regions* object
/// back unchanged (the very same tuple the caller passed, not a
/// copy), and otherwise every row unpacked through the `for code,
/// name in …` contract (see `py_unpack_pair` — a malformed row raises
/// the reference's own ValueError) and kept when the query is a
/// substring of either lowercased field. Lowercasing matches the
/// reference's `.lower()` on plain strs; where the reference would
/// raise — a non-string code or name hitting `.lower()` — the engine
/// stays total and never matches the row (the `filter_panel_items`
/// precedent). The kept rows answer as a fresh tuple of fresh
/// `(code, name)` pairs holding the same string objects, exactly like
/// the reference's comprehension.
#[pyfunction]
#[pyo3(signature = (regions, query))]
pub(crate) fn filter_regions<'py>(
    regions: &Bound<'py, PyAny>,
    query: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let q = python_lower(python_strip(query));
    if q.is_empty() {
        return Ok(regions.clone());
    }
    let py = regions.py();
    let mut kept: Vec<Bound<'py, PyTuple>> = Vec::new();
    for row in regions.try_iter()? {
        let row = row?;
        let (code, name) = py_unpack_pair(&row)?;
        // q in code.lower() or q in name.lower() — a non-string field
        // never matches (the reference's AttributeError there).
        let code_matches = code
            .extract::<&str>()
            .map(|c| python_lower(c).contains(&q))
            .unwrap_or(false);
        let name_matches = name
            .extract::<&str>()
            .map(|n| python_lower(n).contains(&q))
            .unwrap_or(false);
        if code_matches || name_matches {
            kept.push(PyTuple::new(py, [code, name])?);
        }
    }
    Ok(PyTuple::new(py, kept)?.into_any())
}
