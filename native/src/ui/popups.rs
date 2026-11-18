//! Native cores for the popups — `ui/popups/actions.py`,
//! `confirm_popup.py`, `input_popup.py`, `create_playlist_popup.py`,
//! `playlist_picker.py`, `country_picker.py` and `spotify_import.py`:
//! action lists, chrome labels, picker sorting/rows, region filtering
//! and the import flow's decision cores (the candidate row, the result
//! arm, the multi-part verdict, the URL-slot verdict, the summary
//! stats, the create-batch outcome, the input-routing table and the
//! tab-activation phase gate).

use crate::services::ytmusic::py_unpack_pair;
use crate::ui::pages::browse::field_get;
use crate::ui::pages::browse::first_truthy_get;
use crate::ui::pages::help::py_capitalize;
use crate::ui::sidebars::python_strip;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::get_video_id;
use crate::utils::pycompat::py_isspace;
use crate::utils::pycompat::py_str;
use crate::utils::pycompat::python_lower;
use pyo3::exceptions::PyTypeError;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

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

// ---------------------------------------------------------------------------
// Spotify-import popup flow engines (mirrors ui/popups/spotify_import.py)
// ---------------------------------------------------------------------------

/// The URL prefix the popup's `_URL_PATTERN` accepts — the literal the
/// reference's `re.match` anchors at position 0 (every dot escaped, so
/// the pattern spells a literal `http(s)://open.spotify.com/` plus the
/// `playlist`/`album` alternation and the trailing slash).
static SPOTIFY_URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"https?://open\.spotify\.com/(playlist|album)/").expect("valid spotify url regex")
});

/// The disambiguation-candidate row — the pure core of `_CandidateItem`'s
/// label line from `ui/popups/spotify_import.py`.
///
/// Mirrors the reference arm for arm: the title through the mapping's
/// own `get` with the `"?"` default (the fresh object interpolated
/// through the str protocol, like the reference's f-string); the artist
/// through the shared `extract_artist` engine (the candidate is a dict
/// in the ported contract — a non-dict reports a TypeError where the
/// fallback would read it duck-typed, the `spotify_summarize_results`
/// precedent); the duration through the candidate's `duration` field,
/// a falsy read falling to `duration_seconds` and its `m:ss`
/// rendering — floor division and modulo throughout (Python's int
/// semantics, negative operands included), the seconds width
/// zero-padded, the rendering only for a truthy seconds read — and the
/// ` ({dur})` part exactly when the (possibly rebound) duration reads
/// truthy. The row composes through the reference's f-string shape.
#[pyfunction]
#[pyo3(signature = (idx, candidate))]
pub(crate) fn spotify_import_candidate_row<'py>(
    py: Python<'py>,
    idx: i64,
    candidate: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // title = candidate.get("title", "?")
    let question = PyString::new(py, "?").into_any();
    let title = field_get(py, candidate, "title", Some(question))?;
    let title_s = py_str(&title)?;
    // artist_str = extract_artist(candidate)
    let dict = candidate
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("candidate must be a dict"))?;
    let artist = extract_artist(py, dict);
    // dur = candidate.get("duration", "") — the truthiness gate below
    // is the reference's own `if not dur`, on the raw object.
    let empty = PyString::new(py, "").into_any();
    let dur = field_get(py, candidate, "duration", Some(empty))?;
    let (dur_display, dur_truthy) = if dur.is_truthy()? {
        // The f-string's rendering of the original read.
        (py_str(&dur)?, true)
    } else {
        // dur_sec = candidate.get("duration_seconds", 0)
        let zero: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
        let dur_sec = field_get(py, candidate, "duration_seconds", Some(zero))?;
        if dur_sec.is_truthy()? {
            // f"{dur_sec // 60}:{dur_sec % 60:02d}"
            let seconds: i64 = dur_sec.extract()?;
            (
                format!("{}:{:02}", seconds.div_euclid(60), seconds.rem_euclid(60)),
                true,
            )
        } else {
            (String::new(), false)
        }
    };
    // dur_part = f" ({dur})" if dur else "" — the gate on the rebound
    // duration: truthy when the original read was, or when the seconds
    // fallback rendered.
    let dur_part = if dur_truthy {
        format!(" ({dur_display})")
    } else {
        String::new()
    };
    Ok(format!("  {idx}. {title_s} — {artist}{dur_part}"))
}

/// The per-track result-row arm — the display decision both import
/// workers render per matched track (`_result_symbol` from
/// `ui/popups/spotify_import.py`): an `EXACT` verdict the check arm
/// styled `success`, a `MULTIPLE` one the question arm styled
/// `warning`, anything else the cross arm styled `error`.
///
/// The comparisons run through CPython's own `==` against the
/// service's `MatchType` members, imported at call time — the workers'
/// own lazy-import pattern, so exotic `__eq__` implementations steer
/// the arm exactly like they steer the reference's `if` chain. The
/// theme colour read (`getattr(get_theme(), field)`) stays at the call
/// site, on both backends.
#[pyfunction]
pub(crate) fn spotify_import_result_symbol(
    py: Python,
    match_type: &Bound<'_, PyAny>,
) -> PyResult<(&'static str, &'static str)> {
    let si = py.import("ytm_player.services.spotify_import")?;
    let members = si.getattr("MatchType")?;
    if match_type.eq(&members.getattr("EXACT")?)? {
        return Ok(("✓", "success"));
    }
    if match_type.eq(&members.getattr("MULTIPLE")?)? {
        return Ok(("?", "warning"));
    }
    Ok(("✗", "error"))
}

/// The multi-mode setup verdict — the validation cascade behind
/// `_generate_url_inputs` in `ui/popups/spotify_import.py`.
///
/// The reference's gates in the reference's order: a falsy (stripped)
/// *name* answers `("name", 0)`; a *count_str* CPython's own `int()`
/// rejects as a `ValueError` answers `("count", 0)` — only the
/// `ValueError` arm degrades, a `TypeError` (or anything else the call
/// raises) escaping to the caller exactly like the reference's
/// `except ValueError`-only arm; a parsed count outside `2..=20`
/// answers `("range", 0)` — the comparisons against fresh int objects
/// through the interpreter's own rich comparisons, so a huge parsed
/// count (a bignum in the reference) reads out-of-range exactly like
/// it would there; otherwise the verdict is `("ok", count)` with the
/// very int object the `int()` call produced. Only the `("ok", …)`
/// verdict's count means anything — the failed gates answer the int
/// `0` — and the widget reads (the strips), the notifications, the
/// focus moves and the input mounting stay at the call site.
#[pyfunction]
#[pyo3(signature = (name, count_str))]
pub(crate) fn spotify_import_multi_part_verdict<'py>(
    py: Python<'py>,
    name: &str,
    count_str: &str,
) -> PyResult<(String, Bound<'py, PyAny>)> {
    if name.is_empty() {
        let zero: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
        return Ok(("name".to_string(), zero));
    }
    // int(count_str) — fetched from builtins at call time (the json.rs
    // str() precedent), so underscores, whitespace and Unicode digits
    // parse exactly like they do in the reference.
    let count = match py.import("builtins")?.getattr("int")?.call1((count_str,)) {
        Ok(value) => value,
        Err(err) if err.is_instance_of::<PyValueError>(py) => {
            let zero: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
            return Ok(("count".to_string(), zero));
        }
        Err(err) => return Err(err),
    };
    // count < 2 or count > 20
    let two: Bound<'py, PyAny> = 2i64.into_pyobject(py)?.into_any();
    let twenty: Bound<'py, PyAny> = 20i64.into_pyobject(py)?.into_any();
    if count.lt(&two)? || count.gt(&twenty)? {
        let zero: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
        return Ok(("range".to_string(), zero));
    }
    Ok(("ok".to_string(), count))
}

/// The multi-mode URL-slot verdict — the validation pass behind
/// `_start_multi_import` in `ui/popups/spotify_import.py`.
///
/// The first slot (0-based) whose value is bad, with the badness kind:
/// a `None` entry — the caller's widget read failed, a missing URL
/// input — answers `("missing", slot)`; a value the module's
/// `_URL_PATTERN` does not match answers `("invalid", slot)`. A list
/// whose every slot reads a valid playlist/album URL answers `None`.
/// The pattern check is the reference's own `re.match` — a match
/// anchored at position 0, spelt by the literal the module compiles.
/// The widget reads and the notify/focus moves stay at the call site.
#[pyfunction]
#[pyo3(signature = (values))]
pub(crate) fn spotify_import_url_slot_verdict(
    values: Vec<Option<String>>,
) -> Option<(String, usize)> {
    for (slot, value) in values.iter().enumerate() {
        match value {
            None => return Some(("missing".to_string(), slot)),
            Some(url) => {
                if !SPOTIFY_URL_RE.is_match_at(url, 0) {
                    return Some(("invalid".to_string(), slot));
                }
            }
        }
    }
    None
}

/// The summary assembly — the counting pass behind `_show_summary` in
/// `ui/popups/spotify_import.py`.
///
/// Mirrors the reference's own pass order: the three bucket counts —
/// `exact`, then `fuzzy`, then `not_found` — each a full walk comparing
/// `r.match_type` through CPython's own `==` against the service's
/// `MatchType` members (imported at call time, the workers' own
/// lazy-import pattern); the `total`; then the video-id collection —
/// `get_video_id(r.selected)` for every result whose `selected` is not
/// `None` (the identity gate, not truthiness — a falsy-but-present
/// selection still reads its id), the empty ids dropped. The id read is
/// the shared `get_video_id` engine; a selected candidate that is not
/// a dict is outside the ported contract and reports a TypeError
/// rather than the fallback's duck-typed read (the
/// `spotify_summarize_results` precedent). The answer carries
/// `matched` — the reference's `exact + fuzzy` — so the wrapper
/// interpolates the status line from it; the widget arms stay at the
/// call site.
#[pyfunction]
#[pyo3(signature = (results))]
pub(crate) fn spotify_import_summary_stats<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let si = py.import("ytm_player.services.spotify_import")?;
    let members = si.getattr("MatchType")?;
    let exact_member = members.getattr("EXACT")?;
    let multiple_member = members.getattr("MULTIPLE")?;
    let none_member = members.getattr("NONE")?;

    let mut n_exact = 0i64;
    for r in results.try_iter()? {
        let r = r?;
        if r.getattr("match_type")?.eq(&exact_member)? {
            n_exact += 1;
        }
    }
    let mut n_fuzzy = 0i64;
    for r in results.try_iter()? {
        let r = r?;
        if r.getattr("match_type")?.eq(&multiple_member)? {
            n_fuzzy += 1;
        }
    }
    let mut n_not_found = 0i64;
    for r in results.try_iter()? {
        let r = r?;
        if r.getattr("match_type")?.eq(&none_member)? {
            n_not_found += 1;
        }
    }
    let total = results.len()? as i64;

    // video_ids = [get_video_id(r.selected) for r in results
    //              if r.selected is not None]
    // video_ids = [vid for vid in video_ids if vid]
    let video_ids = PyList::empty(py);
    for r in results.try_iter()? {
        let r = r?;
        let selected = r.getattr("selected")?;
        if !selected.is_none() {
            let dict = selected
                .downcast::<PyDict>()
                .map_err(|_| PyTypeError::new_err("selected candidate must be a dict"))?;
            let vid = get_video_id(py, dict);
            if !vid.is_empty() {
                video_ids.append(vid)?;
            }
        }
    }

    let out = PyDict::new(py);
    out.set_item("exact", n_exact)?;
    out.set_item("fuzzy", n_fuzzy)?;
    out.set_item("not_found", n_not_found)?;
    out.set_item("total", total)?;
    out.set_item("matched", n_exact + n_fuzzy)?;
    out.set_item("video_ids", video_ids)?;
    Ok(out)
}

/// The create-batch toast verdict — the final arm behind `_do_create`
/// in `ui/popups/spotify_import.py`: which toast the create flow
/// raises once every add-batch settled.
///
/// The per-batch answers, collected in batch order with their sizes,
/// walk the reference's own accounting: a batch answer comparing equal
/// (CPython's own `==`) to `"success"` adds its size to the accepted
/// total; anything else is a failed batch and its answer rides into
/// the failure kinds. No failed batches answers `"success"`; every
/// batch failed and all failed the same way — the reference's
/// `set(failure_kinds)` of size one, spelt here as every kind
/// comparing equal to the first — answers `"all_failed_single"` with
/// that kind riding along as the very first failing object (the
/// reference's `failure_kinds[0]`); every batch failed for more than
/// one reason answers `"all_failed_mixed"`; anything between answers
/// `"partial"`. The two lists are the caller's own paired collection
/// (equal lengths by construction); the message wording, the
/// `mutation_failure_suffix` call and the notify calls stay at the
/// call site.
#[pyfunction]
#[pyo3(signature = (batch_results, batch_sizes))]
pub(crate) fn spotify_import_batch_outcome<'py>(
    py: Python<'py>,
    batch_results: &Bound<'py, PyList>,
    batch_sizes: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    let success = PyString::new(py, "success");
    let mut failed_batches = 0i64;
    let mut added_total = 0i64;
    let mut failure_kinds: Vec<Bound<'py, PyAny>> = Vec::new();
    for (idx, result) in batch_results.iter().enumerate() {
        let size: i64 = batch_sizes.get_item(idx)?.extract()?;
        if result.eq(&success)? {
            added_total += size;
        } else {
            failed_batches += 1;
            failure_kinds.push(result.clone());
        }
    }
    let batches = batch_results.len() as i64;

    let (verdict, kind) = if failed_batches == 0 {
        ("success", None)
    } else if failed_batches == batches {
        // set(failure_kinds) of size one — spelt as every kind
        // comparing equal to the first (the set's own membership rule
        // for strings); the kind rides home as the very first failing
        // object, the reference's failure_kinds[0].
        let first = &failure_kinds[0];
        let mut all_same = true;
        for k in &failure_kinds {
            if !k.eq(first)? {
                all_same = false;
                break;
            }
        }
        if all_same {
            ("all_failed_single", Some(failure_kinds[0].clone()))
        } else {
            ("all_failed_mixed", None)
        }
    } else {
        ("partial", None)
    };

    let out = PyDict::new(py);
    out.set_item("verdict", verdict)?;
    match kind {
        Some(k) => {
            out.set_item("kind", k)?;
        }
        None => {
            out.set_item("kind", py.None())?;
        }
    }
    out.set_item("added_total", added_total)?;
    out.set_item("failed_batches", failed_batches)?;
    out.set_item("batches", batches)?;
    Ok(out)
}

/// A one-key outcome map for the popup's routing engines —
/// `{"action": <name>}`.
fn spotify_route_outcome<'py>(py: Python<'py>, action: &str) -> Bound<'py, PyDict> {
    let out = PyDict::new(py);
    out.set_item("action", action)
        .expect("dict insertion with a str key cannot fail");
    out
}

/// The input-submission routing table — the pure core of
/// `on_input_submitted` from `ui/popups/spotify_import.py`: which
/// action an (input-id × phase) pair routes to, with each claimed
/// arm's own data decision inside.
///
/// Mirrors the reference's arm order exactly (the ids and phases are
/// the popup's own literals, every comparison a plain string
/// equality):
///
/// - `si-url-input` in the `url` phase answers the single-URL submit:
///   the value stripped (CPython's own `str.strip()` set — see
///   `py_str_strip`) and validated through the one-slot URL pass —
///   the very `spotify_import_url_slot_verdict` engine the multi-mode
///   start run validates through — a bad URL answering
///   `"invalid_url"` (the caller's warning toast), a good one
///   `"start_single"` with the stripped url riding along (the caller
///   stages it as the pending url and starts the import worker).
/// - `si-cred-secret-input` in the `creds` phase answers
///   `"save_creds"`; `si-cred-id-input` in the same phase answers
///   `"focus_secret"`.
/// - `si-name-input` in the `summary` phase answers the name submit:
///   the value stripped, an empty name answering `"name_empty"` (the
///   caller's warning toast), anything else `"create_playlist"` with
///   the stripped name riding along.
/// - `si-multi-count-input` in the `multi_setup` phase answers
///   `"generate_urls"`; `si-multi-name-input` in the same phase
///   answers `"focus_count"`.
/// - an id starting with `si-multi-url-` in the `multi_urls` phase
///   answers the advance walk: the index suffix — every
///   `si-multi-url-` occurrence replaced away, the reference's own
///   `str.replace` — parsed through CPython's own `int()` (fetched
///   from builtins at call time, so whitespace, underscores and
///   Unicode digits parse exactly like they do in the reference); a
///   `ValueError` answers nothing (the reference's
///   `except ValueError: pass`), anything else the call raises
///   escapes to the caller like the reference's except-ValueError-only
///   arm, and a parse answers `"advance_url"` with the next input's
///   id — `si-multi-url-{idx + 1}`, the increment and the rendering
///   through the interpreter's own operators, so a huge index (a
///   bignum in the reference) renders exactly like it would there.
///
/// Anything else — an unknown id, a known id in the wrong phase, a
/// widget with no id at all (`Input.id` is `str | None`) — answers
/// `None`: the reference's chain has no else, so an unmatched
/// submission simply does nothing. The notifications, the focus
/// moves, the worker starts and the start-button fallback (the focus
/// query failing) stay at the call site.
#[pyfunction]
#[pyo3(signature = (input_id, phase, value))]
pub(crate) fn spotify_import_input_route<'py>(
    py: Python<'py>,
    input_id: Option<&str>,
    phase: &str,
    value: &str,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let is = |expected: &str| input_id == Some(expected);
    // ── Single mode ──
    if is("si-url-input") && phase == "url" {
        // url = value.strip(); if _url_slot_verdict([url]) is not None
        let url = py_str_strip(value);
        let verdict = spotify_import_url_slot_verdict(vec![Some(url.to_string())]);
        if verdict.is_some() {
            return Ok(Some(spotify_route_outcome(py, "invalid_url")));
        }
        let out = spotify_route_outcome(py, "start_single");
        out.set_item("url", url)?;
        return Ok(Some(out));
    }
    if is("si-cred-secret-input") && phase == "creds" {
        return Ok(Some(spotify_route_outcome(py, "save_creds")));
    }
    if is("si-cred-id-input") && phase == "creds" {
        return Ok(Some(spotify_route_outcome(py, "focus_secret")));
    }
    if is("si-name-input") && phase == "summary" {
        // name = value.strip(); if not name
        let name = py_str_strip(value);
        if name.is_empty() {
            return Ok(Some(spotify_route_outcome(py, "name_empty")));
        }
        let out = spotify_route_outcome(py, "create_playlist");
        out.set_item("name", name)?;
        return Ok(Some(out));
    }
    // ── Multi mode ──
    if is("si-multi-count-input") && phase == "multi_setup" {
        return Ok(Some(spotify_route_outcome(py, "generate_urls")));
    }
    if is("si-multi-name-input") && phase == "multi_setup" {
        return Ok(Some(spotify_route_outcome(py, "focus_count")));
    }
    if let Some(id) = input_id {
        if id.starts_with("si-multi-url-") && phase == "multi_urls" {
            // idx_str = input_id.replace("si-multi-url-", "")
            let idx_str = id.replace("si-multi-url-", "");
            // int(idx_str) — fetched from builtins at call time (the
            // multi-part verdict's own pattern), so underscores,
            // whitespace and Unicode digits parse exactly like they do
            // in the reference.
            let parsed = match py.import("builtins")?.getattr("int")?.call1((idx_str,)) {
                Ok(value) => value,
                Err(err) if err.is_instance_of::<PyValueError>(py) => return Ok(None),
                Err(err) => return Err(err),
            };
            // next_idx = idx + 1; next_id = f"si-multi-url-{next_idx}"
            let one: Bound<'py, PyAny> = 1i64.into_pyobject(py)?.into_any();
            let next_idx = parsed.add(&one)?;
            let next_id = format!("si-multi-url-{}", next_idx.str()?);
            let out = spotify_route_outcome(py, "advance_url");
            out.set_item("next_id", next_id)?;
            return Ok(Some(out));
        }
    }
    Ok(None)
}

/// The tab-activation phase gate — the decision core of
/// `_activate_tab` from `ui/popups/spotify_import.py`: which phase and
/// which widget arms activating a tab produces.
///
/// `tab_id == "single"` answers the single arm — the phase resets to
/// `"url"`, the single pane shows (the multi pane hides — the two
/// displays are always complementary), the status line reads the
/// paste prompt, the focus moves to the URL input and no multi state
/// resets. Anything else — the multi tab, but also the separator or
/// any other label the reference's `else` would catch — answers the
/// multi arm: the phase resets to `"multi_setup"`, the multi pane
/// shows (the single pane hides), the status line reads the
/// split-playlist instructions, the focus moves to the combined-name
/// input and the caller resets the multi setup (the URL container
/// emptied, the start button hidden, the name and count inputs
/// cleared). The mode assignment (`self._mode = tab_id`), the
/// tab-label class toggles, the display writes, the status update,
/// the resets and the focus moves stay at the call site, applied from
/// the answer in the reference's own order.
#[pyfunction]
#[pyo3(signature = (tab_id))]
pub(crate) fn spotify_import_tab_activation<'py>(
    py: Python<'py>,
    tab_id: &str,
) -> Bound<'py, PyDict> {
    let out = PyDict::new(py);
    if tab_id == "single" {
        out.set_item("phase", "url")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("single_visible", true)
            .expect("dict insertion with a bool key cannot fail");
        out.set_item("status", "Paste a Spotify playlist URL and press Enter")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("focus", "#si-url-input")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("reset_multi", false)
            .expect("dict insertion with a bool key cannot fail");
    } else {
        out.set_item("phase", "multi_setup")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("single_visible", false)
            .expect("dict insertion with a bool key cannot fail");
        out.set_item(
            "status",
            "Split your 100+ track playlist into chunks of ≤100 on Spotify,\n\
             then enter the combined name and number of parts below",
        )
        .expect("dict insertion with a str key cannot fail");
        out.set_item("focus", "#si-multi-name-input")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("reset_multi", true)
            .expect("dict insertion with a bool key cannot fail");
    }
    out
}
