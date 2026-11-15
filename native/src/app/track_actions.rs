//! Native cores for `src/ytm_player/app/_track_actions.py` — track and
//! entity action routing, like toggling, queue removal and the
//! artist/album popup assembly plans.

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{PyDict, PyInt, PyList, PySet, PyString};

// ---------------------------------------------------------------------------
// Track-action engines (mirror ytm_player/app/_track_actions.py)
// ---------------------------------------------------------------------------

/// `item.get(k1) or item.get(k2) or … or default` — the reference's
/// truthiness chain: each read happens only when every earlier one came
/// back falsy, and the answer is the first truthy object itself (the
/// final default when every read is falsy).
fn or_get_chain<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    keys: &[&str],
    default: &str,
) -> PyResult<Bound<'py, PyAny>> {
    for key in keys {
        let value = source.call_method1("get", (*key,))?;
        if value.is_truthy()? {
            return Ok(value);
        }
    }
    Ok(default.into_pyobject(py)?.into_any())
}

/// The first-artist browse-ID extraction — `app/_track_actions.py`'s
/// `_first_artist_id`, shared by the track popup's `go_to_artist` arm
/// and the entity dispatch's `go_to_artist`/`view_similar` arms.
///
/// Mirrors the reference's walk exactly, every read through CPython's
/// own protocol: `source.get("artists")`, then the `isinstance(artists,
/// list) and artists and isinstance(artists[0], dict)` gate (the
/// truthiness consulted only once the isinstance holds), then
/// `artists[0].get("id") or artists[0].get("browseId", "")` — the id
/// read short-circuited before the browseId read, and the answer the
/// raw object the `or` selected (the browseId default `""` included).
/// Anything else answers `""`.
fn first_artist_id_impl<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let artists = source.call_method1("get", ("artists",))?;
    if artists.is_instance_of::<PyList>() && artists.is_truthy()? {
        let first = artists.get_item(0)?;
        if first.is_instance_of::<PyDict>() {
            let id = first.call_method1("get", ("id",))?;
            if id.is_truthy()? {
                return Ok(id);
            }
            return first.call_method1("get", ("browseId", ""));
        }
    }
    Ok("".into_pyobject(py)?.into_any())
}

/// The first-artist browse-ID extraction behind the track popup's and
/// the entity dispatch's artist-navigation arms.
///
/// Implemented as the Rust engine for `app/_track_actions.py`'s
/// `_first_artist_id`; the pure-Python transcription in that module is
/// the fallback for installs without the extension.
#[pyfunction]
pub(crate) fn first_artist_id<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    first_artist_id_impl(py, source)
}

/// The album browse-ID extraction — `app/_track_actions.py`'s
/// `_album_browse_id` — with the reference's own read order and
/// priority: `source.get("album")` first, the nested
/// `album.get("id")` only when the album is a dict (`isinstance`, so a
/// dict subclass passes), then the truthiness chain
/// `source.get("album_id") or nested or source.get("browseId") or ""`
/// — each read short-circuited by every earlier falsy one, the answer
/// the raw object the chain selected. The nested id must outrank the
/// top-level `browseId`: track dicts can carry a non-album `browseId`,
/// while album entities have no nested `album` dict and legitimately
/// resolve through `browseId`.
fn album_browse_id_impl<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let album = source.call_method1("get", ("album",))?;
    let nested = if album.is_instance_of::<PyDict>() {
        album.call_method1("get", ("id",))?
    } else {
        py.None().into_bound(py)
    };
    let album_id = source.call_method1("get", ("album_id",))?;
    if album_id.is_truthy()? {
        return Ok(album_id);
    }
    if nested.is_truthy()? {
        return Ok(nested);
    }
    let browse = source.call_method1("get", ("browseId",))?;
    if browse.is_truthy()? {
        return Ok(browse);
    }
    Ok("".into_pyobject(py)?.into_any())
}

/// The album browse-ID extraction behind the track popup's and the
/// entity dispatch's album-navigation arms.
///
/// Implemented as the Rust engine for `app/_track_actions.py`'s
/// `_album_browse_id`; the pure-Python transcription in that module is
/// the fallback for installs without the extension.
#[pyfunction]
pub(crate) fn album_browse_id<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    album_browse_id_impl(py, source)
}

/// The occurrence gate — `app/_track_actions.py`'s `_queue_entry_id`:
/// the queue entry id a row's occurrence key names, or None. Only the
/// Queue page's table keys its rows by queue entry id (`from_queue`);
/// any other table's key, or no key, names no queue occurrence.
///
/// Mirrors the reference exactly: the `from_queue and isinstance(key,
/// int)` gate — the truthiness of *from_queue* through CPython's own
/// `bool`, the isinstance against `int` (a `bool` key counts, CPython's
/// own subclass rule) — and the key object itself passed through
/// unchanged when the gate holds.
#[pyfunction]
pub(crate) fn queue_entry_id<'py>(
    from_queue: &Bound<'py, PyAny>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    if from_queue.is_truthy()? && key.is_instance_of::<PyInt>() {
        return Ok(Some(key.clone().unbind()));
    }
    Ok(None)
}

/// A one-key outcome map for the action-classification engines —
/// `{"handler": <name>}`.
fn handler_outcome<'py>(py: Python<'py>, handler: &str) -> Bound<'py, PyDict> {
    let out = PyDict::new(py);
    out.set_item("handler", handler)
        .expect("dict insertion with a str key cannot fail");
    out
}

/// The track-actions classification — `app/_track_actions.py`'s
/// `_handle_action_result` dispatch, the routing of an ActionsPopup
/// answer for a track: which handler each *action_id* goes to.
///
/// Mirrors the reference's arm order exactly — `None` answers nothing,
/// then the first equal match among the twelve known action ids wins
/// (every comparison CPython's own `==`, so a plain string answers like
/// anything that compares equal to it), and anything no arm claims is
/// unhandled (the reference's chain has no else — an unknown id simply
/// does nothing). Returns `{"handler": str}` — `"none"` for `None`,
/// `"unhandled"` for an unclaimed id, the arm's own name otherwise.
#[pyfunction]
pub(crate) fn track_action_route<'py>(
    py: Python<'py>,
    action_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if action_id.is_none() {
        return Ok(handler_outcome(py, "none"));
    }
    // The reference's own arm order — first equal match wins.
    const ARMS: &[&str] = &[
        "add_to_playlist",
        "play",
        "download",
        "play_next",
        "add_to_queue",
        "remove_from_queue",
        "start_radio",
        "go_to_artist",
        "go_to_album",
        "toggle_like",
        "copy_link",
        "remove_from_playlist",
    ];
    for arm in ARMS {
        if action_id.eq(*arm)? {
            return Ok(handler_outcome(py, arm));
        }
    }
    Ok(handler_outcome(py, "unhandled"))
}

/// A two-key outcome map for the entity-dispatch engine —
/// `{"handled": bool, "handler": str}`.
fn entity_outcome<'py>(py: Python<'py>, handled: bool, handler: &str) -> Bound<'py, PyDict> {
    let out = PyDict::new(py);
    out.set_item("handled", handled)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("handler", handler)
        .expect("dict insertion with a str key cannot fail");
    out
}

/// The entity-action dispatch — `app/_track_actions.py`'s
/// `_dispatch_entity_action` decision core: which method an
/// (action_id, item_type) pair routes to, and the ids that steer it.
///
/// Mirrors the reference's walk exactly, every comparison through
/// CPython's own protocol. The entity id and name are extracted first,
/// exactly as the reference reads them up front —
/// `item.get("browseId") or item.get("album_id") or
/// item.get("playlistId") or ""` and `item.get("title") or
/// item.get("name") or item.get("artist") or "Unknown"`, each read
/// short-circuited by every earlier falsy one — and ride home in the
/// arms that use them. Then the arms, in the reference's own order:
/// `play_all`/`shuffle_play` (the shuffle flag `action_id ==
/// "shuffle_play"`) answering `_play_album`/`_play_playlist` for the
/// album and playlist types, `add_to_queue` and `play_next` answering
/// their album/playlist enqueue methods, `start_radio` answering
/// `_start_artist_radio` for artists and `_start_playlist_radio` for
/// playlists (the playlist arm with no id gate, exactly the
/// reference's), `play_top_songs` answering `_play_artist_top_songs`
/// for artists, `go_to_artist`/`view_similar` answering
/// `_navigate_artist` with the first-artist id when the item lists
/// artists and the entity id for artist items otherwise, `go_to_album`
/// answering `_navigate_album` with the album browse id, and
/// `add_to_library` answering `_add_album_to_library` for albums.
/// Every typed arm without an entity id answers `notify` (the
/// reference's own "No ID available" early-out, handled); an action no
/// arm claims — or a type the arm doesn't serve — answers `unhandled`
/// with `handled` false. The notifies, the navigations and the awaited
/// methods stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (action_id, item, item_type))]
pub(crate) fn entity_action_route<'py>(
    py: Python<'py>,
    action_id: &Bound<'py, PyAny>,
    item: &Bound<'py, PyAny>,
    item_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // The eager extraction, the reference's own two chains.
    let entity_id = or_get_chain(py, item, &["browseId", "album_id", "playlistId"], "")?;
    let entity_name = or_get_chain(py, item, &["title", "name", "artist"], "Unknown")?;

    // action_id in ("play_all", "shuffle_play")
    if action_id.eq("play_all")? || action_id.eq("shuffle_play")? {
        // shuffle = action_id == "shuffle_play"
        let shuffle = action_id.eq("shuffle_play")?;
        if item_type.eq("album")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_play_album");
            out.set_item("shuffle", shuffle)?;
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        if item_type.eq("playlist")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_play_playlist");
            out.set_item("shuffle", shuffle)?;
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        return Ok(entity_outcome(py, false, "unhandled"));
    }

    // action_id == "add_to_queue"
    if action_id.eq("add_to_queue")? {
        if item_type.eq("album")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_add_album_to_queue");
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        if item_type.eq("playlist")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_add_playlist_to_queue");
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        return Ok(entity_outcome(py, false, "unhandled"));
    }

    // action_id == "play_next"
    if action_id.eq("play_next")? {
        if item_type.eq("album")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_add_album_next");
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        if item_type.eq("playlist")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_add_playlist_next");
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        return Ok(entity_outcome(py, false, "unhandled"));
    }

    // action_id == "start_radio"
    if action_id.eq("start_radio")? {
        if item_type.eq("artist")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_start_artist_radio");
            out.set_item("entity_id", &entity_id)?;
            return Ok(out);
        }
        if item_type.eq("playlist")? {
            // The playlist arm has no id gate — the reference passes the
            // item itself to _start_playlist_radio.
            return Ok(entity_outcome(py, true, "_start_playlist_radio"));
        }
        return Ok(entity_outcome(py, false, "unhandled"));
    }

    // action_id == "play_top_songs"
    if action_id.eq("play_top_songs")? {
        if item_type.eq("artist")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_play_artist_top_songs");
            out.set_item("entity_id", &entity_id)?;
            return Ok(out);
        }
        return Ok(entity_outcome(py, false, "unhandled"));
    }

    // action_id in ("go_to_artist", "view_similar")
    if action_id.eq("go_to_artist")? || action_id.eq("view_similar")? {
        let artists = item.call_method1("get", ("artists",))?;
        let artist_id = if artists.is_instance_of::<PyList>() && artists.is_truthy()? {
            // The item lists artists — the first-artist walk (its own
            // second "artists" read, exactly the reference's).
            first_artist_id_impl(py, item)?
        } else if item_type.eq("artist")? {
            entity_id.clone()
        } else {
            "".into_pyobject(py)?.into_any()
        };
        if !artist_id.is_truthy()? {
            return Ok(entity_outcome(py, true, "notify"));
        }
        let out = entity_outcome(py, true, "_navigate_artist");
        out.set_item("artist_id", &artist_id)?;
        return Ok(out);
    }

    // action_id == "go_to_album"
    if action_id.eq("go_to_album")? {
        let album_id = album_browse_id_impl(py, item)?;
        if !album_id.is_truthy()? {
            return Ok(entity_outcome(py, true, "notify"));
        }
        let out = entity_outcome(py, true, "_navigate_album");
        out.set_item("album_id", &album_id)?;
        return Ok(out);
    }

    // action_id == "add_to_library"
    if action_id.eq("add_to_library")? {
        if item_type.eq("album")? {
            if !entity_id.is_truthy()? {
                return Ok(entity_outcome(py, true, "notify"));
            }
            let out = entity_outcome(py, true, "_add_album_to_library");
            out.set_item("entity_id", &entity_id)?;
            out.set_item("entity_name", &entity_name)?;
            return Ok(out);
        }
        return Ok(entity_outcome(py, false, "unhandled"));
    }

    Ok(entity_outcome(py, false, "unhandled"))
}

/// The like-toggle decision — `app/_track_actions.py`'s
/// `toggle_like` arm rating resolution: which rating the popup's Like
/// toggle submits, and which label its toast carries.
///
/// Mirrors the reference exactly: `is_liked` is the raw
/// `track.get("likeStatus") == "LIKE" or track.get("liked", False)`
/// result — the `==` through CPython's own protocol, the `or` keeping
/// the very object it selected (a truthy comparison result, or the
/// liked read itself, falsy included) — and the rating/label pair
/// (`"INDIFFERENT"`/`"Unliked"` when the or-result is truthy,
/// `"LIKE"`/`"Liked"` otherwise) resolved through that same
/// truthiness. The rate_song submission, the track's likeStatus write
/// and the toasts stay Python-side in the mixin.
#[pyfunction]
pub(crate) fn like_toggle_rating<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let like_status = track.call_method1("get", ("likeStatus",))?;
    let eq_result = like_status.rich_compare("LIKE", CompareOp::Eq)?;
    let is_liked = if eq_result.is_truthy()? {
        eq_result
    } else {
        track.call_method1("get", ("liked", false))?
    };
    let rating = if is_liked.is_truthy()? {
        "INDIFFERENT"
    } else {
        "LIKE"
    };
    let label = if is_liked.is_truthy()? {
        "Unliked"
    } else {
        "Liked"
    };
    let out = PyDict::new(py);
    out.set_item("is_liked", &is_liked)?;
    out.set_item("rating", rating)?;
    out.set_item("label", label)?;
    Ok(out)
}

/// The removal plan — `app/_track_actions.py`'s `_remove_from_queue`
/// decision core: how the popup's Remove from Queue takes the track
/// out of the queue.
///
/// Mirrors the reference's walk exactly, every comparison through
/// CPython's own protocol: the occurrence arm `from_queue or
/// queue_entry_id is not None` — an id present answers `entry` (the
/// caller's own `remove_entry` attempt deciding success; the entry
/// may be gone by the time the popup closed, and only that occurrence
/// may go), `from_queue` without an id answers `stale` (a Queue page
/// row whose id could not be read is stale, never nameless — the
/// warning without touching the queue). Without any occurrence to
/// name, a falsy video id answers `none`, and otherwise the first
/// queue row whose `t.get("video_id") == video_id` answers
/// `remove_at` carrying that row's index (the row walk through the
/// caller's own tracks list, the comparison CPython's own `==`), a
/// no-match walk answering `none`. The removals, the queue-page
/// refreshes and the toasts stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (from_queue, queue_entry_id, queue_tracks, video_id))]
pub(crate) fn remove_from_queue_plan<'py>(
    py: Python<'py>,
    from_queue: &Bound<'py, PyAny>,
    queue_entry_id: &Bound<'py, PyAny>,
    queue_tracks: &Bound<'py, PyAny>,
    video_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // if from_queue or queue_entry_id is not None — the occurrence arm.
    if from_queue.is_truthy()? || !queue_entry_id.is_none() {
        // An id names the occurrence (the caller's remove_entry attempt
        // decides success); from_queue without an id is stale.
        let arm = if queue_entry_id.is_none() {
            "stale"
        } else {
            "entry"
        };
        let out = PyDict::new(py);
        out.set_item("arm", arm)
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // if not video_id — nothing to look for, nothing removed.
    if !video_id.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("arm", "none")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // The first copy by video id — the row walk with CPython's own ==.
    for (i, row) in queue_tracks.try_iter()?.enumerate() {
        let row = row?;
        let row_vid = row.call_method1("get", ("video_id",))?;
        if row_vid.eq(video_id)? {
            let out = PyDict::new(py);
            out.set_item("arm", "remove_at")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("index", i)?;
            return Ok(out);
        }
    }
    let out = PyDict::new(py);
    out.set_item("arm", "none")
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The in-queue verdict — `app/_track_actions.py`'s popup-flags read:
/// whether the track the actions popup was opened on is currently in
/// the playback queue (the popup swaps "Add to Queue" for "Remove
/// from Queue" when it is).
///
/// Mirrors the reference exactly: a falsy video id answers False
/// without touching the queue, and otherwise the `any()` walk — each
/// row's `t.get("video_id") == track_vid` through CPython's own `==`,
/// short-circuiting on the first equal row.
#[pyfunction]
#[pyo3(signature = (video_id, queue_tracks))]
pub(crate) fn in_queue_verdict<'py>(
    video_id: &Bound<'py, PyAny>,
    queue_tracks: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    if !video_id.is_truthy()? {
        return Ok(false);
    }
    for row in queue_tracks.try_iter()? {
        let row = row?;
        let row_vid = row.call_method1("get", ("video_id",))?;
        if row_vid.eq(video_id)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The artist-songs merge plan — `app/_track_actions.py`'s
/// `_fetch_remaining_artist_songs` decision core: how the background
/// fetch's full song list enriches the already-queued initial tracks
/// and which of its tracks get appended.
///
/// Mirrors the reference's walk exactly, every read and every hash
/// through CPython's own protocol. The full-list index first —
/// `{t.get("video_id", ""): t for t in all_tracks if
/// t.get("video_id")}`, the filter read short-circuited before the
/// key read, the dict built through CPython's own hash so a
/// duplicate video id keeps the reference's last-wins entry — then
/// the initial batch's id set — `{t.get("video_id", "") for t in
/// initial_tracks}`, no filter — then the enrichment pass over the
/// caller's own queue list, each queue track whose `qt.get("video_id",
/// "")` is in the index answering one `(queue_track, full_track)`
/// pair carrying the very objects (the caller's in-place updates hit
/// the queue's own dicts), and finally the append pass, each
/// all-track whose id is not in the initial set answering itself
/// (duplicates within the fetch included, exactly the reference's
/// membership-only check). The fetch, the ownership gate, the
/// in-place updates, the queue adds and the queue-page refresh stay
/// Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (all_tracks, initial_tracks, queue_tracks))]
pub(crate) fn artist_songs_merge_plan<'py>(
    py: Python<'py>,
    all_tracks: &Bound<'py, PyAny>,
    initial_tracks: &Bound<'py, PyAny>,
    queue_tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // full_by_id — the full list's index, last-wins per video id.
    let full_by_id = PyDict::new(py);
    for t in all_tracks.try_iter()? {
        let t = t?;
        let filter_vid = t.call_method1("get", ("video_id",))?;
        if filter_vid.is_truthy()? {
            let key = t.call_method1("get", ("video_id", ""))?;
            full_by_id.set_item(&key, &t)?;
        }
    }
    // existing_ids — the initial batch's ids, "" included.
    let existing_ids = PySet::empty(py)?;
    for t in initial_tracks.try_iter()? {
        let t = t?;
        let key = t.call_method1("get", ("video_id", ""))?;
        existing_ids.add(key)?;
    }
    // The enrichment pass — (queue track, full track) pairs, the very
    // objects, so the caller's in-place updates hit the queue's own
    // dicts.
    let updates = PyList::empty(py);
    for qt in queue_tracks.try_iter()? {
        let qt = qt?;
        let vid = qt.call_method1("get", ("video_id", ""))?;
        if full_by_id.contains(&vid)? {
            let full = match full_by_id.get_item(&vid)? {
                Some(full) => full,
                None => {
                    return Err(PyKeyError::new_err(vid.str()?.to_string()));
                }
            };
            updates.append((qt, full))?;
        }
    }
    // The append pass — every all-track whose id is not one of the
    // initial batch's, duplicates within the fetch included.
    let appends = PyList::empty(py);
    for t in all_tracks.try_iter()? {
        let t = t?;
        let vid = t.call_method1("get", ("video_id", ""))?;
        if !existing_ids.contains(&vid)? {
            appends.append(t)?;
        }
    }
    let out = PyDict::new(py);
    out.set_item("updates", updates)?;
    out.set_item("appends", appends)?;
    Ok(out)
}

/// The artist-popup gate — `app/_track_actions.py`'s
/// `_open_actions_for_artist` decision core: which popup a track's
/// artist column opens.
///
/// Mirrors the reference's walk exactly, every read through CPython's
/// own protocol: `track.get("artists", [])` first, then the
/// `not artists or not isinstance(artists, list)` gate (the truthiness
/// consulted before the isinstance), then the valid list — every
/// artist whose `a.get("id") or a.get("browseId")` chain resolves
/// truthy (the id read short-circuited before the browseId read) —
/// and finally the count gate: no artists at all, or none valid,
/// answers `"none"` (the caller's own "No artist info available"
/// warning); exactly one valid artist answers `"single"` carrying that
/// very artist; anything else answers `"pick"` carrying the valid
/// list. The notifications and the popup pushes stay Python-side in
/// the mixin.
#[pyfunction]
pub(crate) fn artist_popup_plan<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let empty_list = PyList::empty(py).into_any();
    let artists = track.call_method1("get", ("artists", empty_list))?;
    if artists.is_truthy()? && artists.is_instance_of::<PyList>() {
        let valid = PyList::empty(py);
        for a in artists.try_iter()? {
            let a = a?;
            // a.get("id") or a.get("browseId") — the id read
            // short-circuited before the browseId read.
            let id = a.call_method1("get", ("id",))?;
            let is_valid = if id.is_truthy()? {
                true
            } else {
                let browse = a.call_method1("get", ("browseId",))?;
                browse.is_truthy()?
            };
            if is_valid {
                valid.append(a)?;
            }
        }
        let out = PyDict::new(py);
        if valid.len() == 1 {
            let first = valid.get_item(0)?;
            out.set_item("arm", "single")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("valid", &valid)?;
            out.set_item("artist", &first)?;
            return Ok(out);
        }
        if valid.len() >= 2 {
            out.set_item("arm", "pick")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("valid", &valid)?;
            return Ok(out);
        }
        out.set_item("arm", "none")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    let out = PyDict::new(py);
    out.set_item("arm", "none")
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The artist picker's rows — `app/_track_actions.py`'s
/// `_show_artist_picker` assembly: one `(str(i), name)` pair per
/// artist, in the list's own order.
///
/// Mirrors the reference exactly: the index through Python's own
/// `str()`, the name the raw `a.get("name", "Unknown Artist")` result
/// (the default included, the very object the caller's popup renders).
#[pyfunction]
pub(crate) fn artist_picker_rows<'py>(
    py: Python<'py>,
    artists: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let rows = PyList::empty(py);
    for (i, a) in artists.try_iter()?.enumerate() {
        let a = a?;
        let name = a.call_method1("get", ("name", "Unknown Artist"))?;
        let row = (PyString::new(py, &i.to_string()), name);
        rows.append(row)?;
    }
    Ok(rows)
}

/// The artist popup's item — `app/_track_actions.py`'s
/// `_show_artist_actions` assembly: the browse id gate and the item
/// the entity dispatch reads.
///
/// Mirrors the reference exactly, every read through CPython's own
/// protocol: the browse id — `artist.get("id") or
/// artist.get("browseId", "")`, the id read short-circuited before the
/// browseId read with its `""` default — gates the popup (a falsy id
/// answers `"none"`, the reference's own silent return), and a truthy
/// one answers `"item"` carrying the id and the
/// `{"browseId": id, "artist": name}` assembly, the name the raw
/// `artist.get("name", "Unknown Artist")` result. The answer handling
/// and the popup push stay Python-side in the mixin.
#[pyfunction]
pub(crate) fn artist_popup_item<'py>(
    py: Python<'py>,
    artist: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // browse_id = artist.get("id") or artist.get("browseId", "")
    let id = artist.call_method1("get", ("id",))?;
    let browse_id = if id.is_truthy()? {
        id
    } else {
        artist.call_method1("get", ("browseId", ""))?
    };
    if !browse_id.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("arm", "none")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    let name = artist.call_method1("get", ("name", "Unknown Artist"))?;
    let item = PyDict::new(py);
    item.set_item("browseId", &browse_id)?;
    item.set_item("artist", &name)?;
    let out = PyDict::new(py);
    out.set_item("arm", "item")
        .expect("dict insertion with a str key cannot fail");
    out.set_item("browse_id", &browse_id)?;
    out.set_item("item", &item)?;
    Ok(out)
}

/// The album popup's item — `app/_track_actions.py`'s
/// `_open_actions_for_album` assembly: the album id gate, the
/// album-name resolution and the artists attachment.
///
/// Mirrors the reference exactly, every read through CPython's own
/// protocol: the album id — `track.get("album_id") or
/// (album.get("id") if isinstance(album, dict) else None) or ""`, the
/// nested read gated on the album being a dict, each read
/// short-circuited by every earlier falsy one — gates the popup (a
/// falsy id answers `"none"`, the caller's own "No album info
/// available" warning). A truthy one answers `"item"` carrying the id
/// and the assembly — `{"browseId": id, "title": album_name}` with the
/// album name the raw album string for a string album and
/// `album.get("name", "Unknown Album")` otherwise, plus the track's
/// `artists` list attached verbatim (its own `__getitem__` read, only
/// once the truthiness gate holds) when the track lists one. The
/// warning, the answer handling and the popup push stay Python-side in
/// the mixin.
#[pyfunction]
pub(crate) fn album_popup_item<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // album = track.get("album", {})
    let empty_album = PyDict::new(py).into_any();
    let album = track.call_method1("get", ("album", empty_album))?;
    // album_id = track.get("album_id") or (album.get("id") if
    // isinstance(album, dict) else None) or ""
    let album_id_read = track.call_method1("get", ("album_id",))?;
    let album_id = if album_id_read.is_truthy()? {
        album_id_read
    } else {
        let nested = if album.is_instance_of::<PyDict>() {
            album.call_method1("get", ("id",))?
        } else {
            py.None().into_bound(py)
        };
        if nested.is_truthy()? {
            nested
        } else {
            PyString::new(py, "").into_any()
        }
    };
    if !album_id.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("arm", "none")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // album_name = album if isinstance(album, str) else
    //              album.get("name", "Unknown Album")
    let album_name = if album.is_instance_of::<PyString>() {
        album
    } else {
        album.call_method1("get", ("name", "Unknown Album"))?
    };
    let item = PyDict::new(py);
    item.set_item("browseId", &album_id)?;
    item.set_item("title", &album_name)?;
    // if track.get("artists"): item["artists"] = track["artists"]
    let artists = track.call_method1("get", ("artists",))?;
    if artists.is_truthy()? {
        item.set_item("artists", track.get_item("artists")?)?;
    }
    let out = PyDict::new(py);
    out.set_item("arm", "item")
        .expect("dict insertion with a str key cannot fail");
    out.set_item("album_id", &album_id)?;
    out.set_item("item", &item)?;
    Ok(out)
}
