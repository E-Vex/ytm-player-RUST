//! Native cores for `src/ytm_player/ui/pages/recently_played.py` —
//! occurrence stamping, accepted-play merges and row composition.

use crate::utils::formatting::get_video_id;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};

// ---------------------------------------------------------------------------
// Recently Played history-merge engine (mirrors ui/pages/recently_played.py)
// ---------------------------------------------------------------------------

/// Fresh occurrence ids for account-history rows — the native twin of the
/// fallback's module-level `itertools.count(1)`. Only one backend is ever
/// active in a process (the Python seam binds either the native functions
/// or the pure-Python ones, never both), and a fresh id's only contract is
/// that it differs from every id the cache already holds, so the two
/// counters never need to agree.
static OCCURRENCE_COUNTER: AtomicI64 = AtomicI64::new(1);

/// `get_video_id(item)` for a list element: the track's id, or `""` when
/// the element is not a dict (where the Python fallback would raise —
/// every cache row is a normalized track dict, and staying total keeps one
/// malformed row from taking the view down with it).
fn item_video_id(py: Python, item: &Bound<'_, PyAny>) -> String {
    match item.downcast::<PyDict>() {
        Ok(dict) => get_video_id(py, dict),
        Err(_) => String::new(),
    }
}

/// A shallow `dict(track)` copy — new dict, same value objects.
pub(crate) fn shallow_copy_dict<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let copy = PyDict::new(py);
    for (key, value) in track.iter() {
        copy.set_item(key, value)?;
    }
    Ok(copy)
}

/// The `_enrich` notion of a lacking field value: `None` or `""` (Python's
/// `value in (None, "")` — every other value, `0` and `False` included,
/// counts as present). Non-string, non-None values are never lacking.
fn is_lacking(value: &Bound<'_, PyAny>) -> bool {
    if value.is_none() {
        return true;
    }
    match value.downcast::<PyString>() {
        Ok(s) => s.to_str().map(|t| t.is_empty()).unwrap_or(false),
        Err(_) => false,
    }
}

/// `len(list[:n])` — Python prefix-slice semantics, negative bounds
/// included (`[:-2]` keeps all but the last two).
fn prefix_slice_len(len: usize, n: isize) -> usize {
    if n >= 0 {
        (n as usize).min(len)
    } else {
        len.saturating_sub(n.unsigned_abs())
    }
}

/// The core of `_stamp`: a shallow copy of *track* carrying *occurrence*.
/// An absent or `None` occurrence allocates a fresh id from the counter —
/// `_stamp(track, cached.get("_occurrence"))` passes `None` for a cache
/// row that carries no stamp, so both shapes mean "fresh" — and any other
/// value is stored verbatim (cache stamps are ints, but the engine never
/// needs to assume that).
fn stamp_track_impl<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyDict>,
    occurrence: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let id: Bound<'py, PyAny> = match occurrence {
        Some(v) if !v.is_none() => v,
        _ => {
            let n = OCCURRENCE_COUNTER.fetch_add(1, Ordering::Relaxed);
            n.into_pyobject(py)?.into_any()
        }
    };
    let stamped = shallow_copy_dict(py, track)?;
    stamped.set_item("_occurrence", id)?;
    Ok(stamped)
}

/// A copy of *track* carrying an occurrence id — a fresh one unless given.
///
/// Mirrors `_stamp` from `ui/pages/recently_played.py`: the account cache
/// stamps every row that enters it, one id per row, so two rows of the
/// same track (the feed lists a track once per day it was played) keep
/// distinct identities across background refreshes.
#[pyfunction(signature = (track, occurrence=None))]
pub(crate) fn stamp_track<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyDict>,
    occurrence: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    stamp_track_impl(py, track, occurrence)
}

/// The key a cached account row is matched by across background
/// refreshes.
///
/// Mirrors `_occurrence_key`: the stored `_occurrence` verbatim — an
/// explicit `None` included, since `.get(key, default)` only substitutes
/// when the key is *absent* — and the video id only for a row that never
/// got stamped.
#[pyfunction]
pub(crate) fn occurrence_key<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    match track.get_item("_occurrence")? {
        Some(v) => Ok(v),
        None => Ok(PyString::new(py, &get_video_id(py, track)).into_any()),
    }
}

/// The core of `_with_accepted_play`, over a borrowed row vector, so
/// `merge_pending_account_plays` can fold plays without rebuilding a
/// Python list per step. Non-dict cache rows are skipped rather than
/// raising (see `item_video_id`).
fn with_accepted_play_impl<'py>(
    py: Python<'py>,
    cache: Vec<Bound<'py, PyAny>>,
    track: &Bound<'py, PyDict>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let video_id = get_video_id(py, track);
    for (index, item) in cache.iter().enumerate() {
        if let Ok(dict) = item.downcast::<PyDict>() {
            if get_video_id(py, dict) == video_id {
                // The cache's first listing of the track is its most
                // recent one: that row moves to the top keeping its
                // identity (a mark on it moves along) and takes the
                // played track's details. Older listings stay put.
                let occurrence = dict.get_item("_occurrence")?;
                let moved = stamp_track_impl(py, track, occurrence)?;
                let mut out: Vec<Bound<'py, PyAny>> = Vec::with_capacity(cache.len());
                out.push(moved.into_any());
                for (i, item) in cache.iter().enumerate() {
                    if i != index {
                        out.push(item.clone());
                    }
                }
                return Ok(out);
            }
        }
    }
    // A track the cache does not list gets a new occurrence. Identity is
    // the cache's to decide: a stamp the incoming track carries (a row it
    // was played from) is never kept.
    let stamped = stamp_track_impl(py, track, None)?;
    let mut out: Vec<Bound<'py, PyAny>> = Vec::with_capacity(cache.len() + 1);
    out.push(stamped.into_any());
    out.extend(cache.iter().cloned());
    Ok(out)
}

/// *cache* with an accepted play of *track* on top.
///
/// Mirrors `_with_accepted_play` from `ui/pages/recently_played.py`: the
/// play supersedes the track's most recent listing — that row moves to the
/// top keeping its occurrence identity, with the played track's details —
/// while older listings of the track stay where they are (a play never
/// deletes history). A track the cache does not list gets a new
/// occurrence.
#[pyfunction(signature = (cache, track))]
pub(crate) fn with_accepted_play<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyList>,
    track: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyList>> {
    let rows: Vec<Bound<'py, PyAny>> = cache.iter().collect();
    let merged = with_accepted_play_impl(py, rows, track)?;
    PyList::new(py, merged)
}

/// *cache* with an accepted play of *track* on top, capped at *cap* rows —
/// the Local tab's optimistic update.
///
/// Mirrors the local branch of `RecentlyPlayedPage.optimistic_add`: the
/// local history query groups by track, so the cache lists a track once —
/// rows of the same track make way for the played track's fresh copy at
/// the top, other rows keep their place and identity, and the cap trims
/// from the end. No occurrence stamp: the local view keys rows by video
/// id.
#[pyfunction(signature = (cache, track, cap))]
pub(crate) fn with_accepted_local_play<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyList>,
    track: &Bound<'py, PyDict>,
    cap: isize,
) -> PyResult<Bound<'py, PyList>> {
    let video_id = get_video_id(py, track);
    let mut out: Vec<Bound<'py, PyAny>> = Vec::with_capacity(cache.len() + 1);
    out.push(shallow_copy_dict(py, track)?.into_any());
    for item in cache.iter() {
        if item_video_id(py, &item) != video_id {
            out.push(item);
        }
    }
    let keep = prefix_slice_len(out.len(), cap);
    out.truncate(keep);
    PyList::new(py, out)
}

/// *pending* with an accepted play of *track* parked ahead of the fetch.
///
/// Mirrors the parking branch of
/// `PlaybackMixin._add_to_ytm_history_cache`: the fresh entry goes on top
/// (newest first) carrying *seq* and a copy of *track*; older entries for
/// the same track are dropped — one pending row per track, the newest —
/// and the list is capped at *cap* entries from the front. Kept entries
/// are the same tuple objects; only the new entry is built.
#[pyfunction(signature = (pending, seq, track, cap))]
pub(crate) fn park_pending_play<'py>(
    py: Python<'py>,
    pending: &Bound<'py, PyList>,
    seq: i64,
    track: &Bound<'py, PyDict>,
    cap: isize,
) -> PyResult<Bound<'py, PyList>> {
    let video_id = get_video_id(py, track);
    let mut out: Vec<Bound<'py, PyAny>> = Vec::with_capacity(pending.len() + 1);
    let entry = PyTuple::new(
        py,
        [
            seq.into_pyobject(py)?.into_any(),
            shallow_copy_dict(py, track)?.into_any(),
        ],
    )?;
    out.push(entry.into_any());
    for item in pending.iter() {
        let keep = match item.downcast::<PyTuple>() {
            Ok(t) => match t.get_item(1) {
                Ok(second) => item_video_id(py, &second) != video_id,
                Err(_) => true,
            },
            Err(_) => true,
        };
        if keep {
            out.push(item);
        }
    }
    let keep = prefix_slice_len(out.len(), cap);
    out.truncate(keep);
    PyList::new(py, out)
}

/// Fold *pending* accepted plays into *feed*.
///
/// Mirrors `RecentlyPlayedPage._merge_pending_account_plays` (its pure
/// core; the page method consumes the app's pending list and delegates
/// here): a play accepted after the fetch started (sequence number above
/// *fetch_seq*) post-dates the feed, so it goes ahead of it, superseding
/// the feed's most recent listing of the track if there is one. A play
/// accepted before the fetch started is already reflected in the feed —
/// its server position stands — unless the feed does not list it yet, in
/// which case it goes ahead of the feed, behind the newer plays. *feed*
/// itself is returned untouched when nothing is pending.
///
/// Malformed pending entries (not a `(seq, track)` tuple of an int and a
/// dict) are skipped where the Python fallback would raise — the app only
/// ever parks well-formed entries.
#[pyfunction(signature = (feed, fetch_seq, pending))]
pub(crate) fn merge_pending_account_plays<'py>(
    py: Python<'py>,
    feed: &Bound<'py, PyList>,
    fetch_seq: i64,
    pending: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    if pending.is_empty() {
        return Ok(feed.clone());
    }
    // The feed's video ids, "" included (the fallback's set comprehension
    // adds it too; the membership test below can query it).
    let mut feed_ids: HashSet<String> = HashSet::new();
    for item in feed.iter() {
        feed_ids.insert(item_video_id(py, &item));
    }
    let mut merged: Vec<Bound<'py, PyAny>> = feed.iter().collect();
    // Oldest first (the pending list is newest-first), so the newest play
    // ends up on top.
    for item in pending.iter().rev() {
        let entry = match item.downcast::<PyTuple>() {
            Ok(t) => t,
            Err(_) => continue,
        };
        let (seq, track) = match (entry.get_item(0), entry.get_item(1)) {
            (Ok(seq), Ok(track)) => (seq, track),
            _ => continue,
        };
        let seq: i64 = match seq.extract() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let track = match track.downcast::<PyDict>() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if seq <= fetch_seq && feed_ids.contains(&get_video_id(py, track)) {
            continue;
        }
        merged = with_accepted_play_impl(py, merged, track)?;
    }
    PyList::new(py, merged)
}

/// A copy of *track* with the fields it lacks filled in from *extra*.
///
/// Mirrors `_enrich`: a field counts as lacking when it is absent, `None`
/// or `""` — in *either* dict, so the local row's own values (`played_at`
/// included) are never replaced and an empty account value never replaces
/// anything either. `0` and `False` count as present on both sides, exactly
/// like Python's `value in (None, "")`.
#[pyfunction(signature = (track, extra))]
pub(crate) fn enrich_track<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyDict>,
    extra: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    enrich_track_impl(py, track, extra)
}

/// The `_enrich` core, shared with `compose_all_rows`.
fn enrich_track_impl<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyDict>,
    extra: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let merged = shallow_copy_dict(py, track)?;
    for (key, value) in extra.iter() {
        // An empty account value enriches nothing.
        if is_lacking(&value) {
            continue;
        }
        // The local row's own values are never replaced: only a key the
        // local row lacks (absent, None or "") is filled in.
        if let Some(existing) = track.get_item(&key)? {
            if !is_lacking(&existing) {
                continue;
            }
        }
        merged.set_item(key, value)?;
    }
    Ok(merged)
}

/// Local rows first, in their own order, then account-only rows in server
/// order — the All tab's composition.
///
/// Mirrors `_compose_all` from `ui/pages/recently_played.py`: one row per
/// video id. A track in both keeps the local row — its position and
/// `played_at` — with the fields the local row lacks (artist and album
/// ids, thumbnail) filled in from the account row; a local row without an
/// account counterpart is passed through as the same object, as is the
/// first account row of every account-only id. Deduplication happens
/// before the account-only cap, so that cap never counts rows the local
/// block already shows; each source is capped at *cap* rows separately.
#[pyfunction(signature = (local, account, cap))]
pub(crate) fn compose_all_rows<'py>(
    py: Python<'py>,
    local: &Bound<'py, PyList>,
    account: &Bound<'py, PyList>,
    cap: isize,
) -> PyResult<Bound<'py, PyList>> {
    // by_id: the FIRST account row per video id, in first-seen order —
    // Python's dict insertion order. Empty ids never enter the map.
    let mut by_id: Vec<(String, Bound<'py, PyDict>)> = Vec::new();
    let mut by_id_index: HashMap<String, usize> = HashMap::new();
    for item in account.iter() {
        if let Ok(dict) = item.downcast::<PyDict>() {
            let vid = get_video_id(py, dict);
            if !vid.is_empty() && !by_id_index.contains_key(&vid) {
                by_id_index.insert(vid.clone(), by_id.len());
                by_id.push((vid, dict.clone()));
            }
        }
    }

    // The local block: capped first (the slice), then deduplicated — a
    // duplicate inside the cap shrinks the block rather than pulling in a
    // row from beyond the cap.
    let local_cap = prefix_slice_len(local.len(), cap);
    let mut rows: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for item in local.iter().take(local_cap) {
        let Ok(dict) = item.downcast::<PyDict>() else {
            continue;
        };
        let vid = get_video_id(py, dict);
        if vid.is_empty() || !seen.insert(vid.clone()) {
            continue;
        }
        let row: Bound<'py, PyAny> = match by_id_index.get(&vid) {
            Some(&i) => enrich_track_impl(py, dict, &by_id[i].1)?.into_any(),
            None => item, // no account counterpart: the local row itself
        };
        rows.push(row);
    }

    // The account-only block: first-seen order, ids the local block did
    // not show, capped separately at *cap* rows.
    let account_cap = prefix_slice_len(by_id.len(), cap);
    let mut account_only = 0usize;
    for (vid, dict) in &by_id {
        if account_only >= account_cap {
            break;
        }
        if !seen.contains(vid) {
            rows.push(dict.clone().into_any());
            account_only += 1;
        }
    }
    PyList::new(py, rows)
}
