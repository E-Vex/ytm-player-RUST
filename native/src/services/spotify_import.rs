//! Native cores for `src/ytm_player/services/spotify_import.py` (and the
//! import popup in `ui/popups/spotify_import.py`) — match classification,
//! scraped-track rows and result bucketing.

use crate::services::history::builtin_int;
use crate::utils::formatting::extract_artist;
use crate::utils::formatting::extract_duration;
use crate::utils::formatting::format_duration;
use crate::utils::formatting::get_video_id;
use crate::utils::pycompat::py_str;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Spotify-import matching cores
// ---------------------------------------------------------------------------

/// The `(playlist|album)/([a-zA-Z0-9]+)` ID pattern from
/// `services/spotify_import.py`'s `_extract_playlist_id` — `re.search`
/// semantics: the leftmost-first match anywhere in the URL.
static SPOTIFY_ID_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(playlist|album)/([a-zA-Z0-9]+)").expect("valid spotify id regex")
});

/// Pull the playlist/album ID from a Spotify URL. Mirrors
/// `spotify_import._extract_playlist_id`: the base-62 run after the
/// first `playlist/` or `album/` the URL contains, or the empty string
/// when it has neither.
#[pyfunction]
pub(crate) fn extract_playlist_id(url: &str) -> String {
    SPOTIFY_ID_RE
        .captures(url)
        .and_then(|caps| caps.get(2))
        .map(|m| m.as_str().to_string())
        .unwrap_or_default()
}

/// Normalise a spotipy track item to the internal Spotify-track format.
/// Mirrors `spotify_import._parse_spotipy_item`: the nested `track`
/// mapping (the item itself when the key is absent — a present value
/// passes through as-is, `None` included, into the falsy gate), the
/// falsy-track / falsy-name gate that yields the empty dict, and the
/// four-key output — the artist through the same `extract_artist`
/// engine the Python side's module global resolves to, the album name
/// read only when the `album` value is a dict (a present-but-`None`
/// `name` inside it passes through as `None`, exactly like `.get`), and
/// the `duration_ms` value passed through as the object it is (the `0`
/// default only on an absent key). Key order mirrors the reference's
/// dict literal.
///
/// The item and the nested track are the plain dicts spotipy's JSON
/// payloads produce; a truthy non-dict `track` value — unreachable
/// through `extract_spotify_tracks_spotipy`, the only caller — is
/// outside the ported contract and reports a TypeError rather than the
/// reference's AttributeError.
#[pyfunction]
pub(crate) fn parse_spotipy_item<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    // item.get("track", item)
    let track_any: Bound<'py, PyAny> = match item.get_item("track")? {
        Some(value) => value,
        None => item.as_any().clone(),
    };
    // `not track or ...` — falsiness first, so an empty/None track
    // never reaches the name read.
    if !track_any.is_truthy()? {
        return Ok(PyDict::new(py));
    }
    let track = track_any
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("spotipy track must be a dict"))?;

    // `not track.get("name")` — absent and falsy (None, "", 0, ...)
    // both yield the empty dict.
    let name = match track.get_item("name")? {
        Some(value) if value.is_truthy()? => value,
        _ => return Ok(PyDict::new(py)),
    };

    let artist = extract_artist(py, track);

    // track.get("album", {})
    let album: Bound<'py, PyAny> = match track.get_item("album")? {
        Some(value) => value,
        None => PyDict::new(py).into_any(),
    };
    // album.get("name", "") if isinstance(album, dict) else ""
    let album_name: Bound<'py, PyAny> = if album.is_instance_of::<PyDict>() {
        match album.downcast::<PyDict>().unwrap().get_item("name")? {
            Some(value) => value,
            None => PyString::new(py, "").into_any(),
        }
    } else {
        PyString::new(py, "").into_any()
    };

    // track.get("duration_ms", 0)
    let duration_ms: Bound<'py, PyAny> = match track.get_item("duration_ms")? {
        Some(value) => value,
        None => 0i64.into_pyobject(py)?.into_any(),
    };

    let out = PyDict::new(py);
    out.set_item("name", name)?;
    out.set_item("artist", artist)?;
    out.set_item("album", album_name)?;
    out.set_item("duration_ms", duration_ms)?;
    Ok(out)
}

/// The weighted fuzzy-match score behind the import matcher's candidate
/// ranking — the pure core of `services/spotify_import.py`'s
/// `_fuzzy_score`.
///
/// Mirrors the pure-Python reference: both sides lowercased through the
/// strings' own `.lower` (the Spotify side's `name`/`artist` read
/// duck-typed through the mapping's own `get` with the `""` defaults,
/// the YTM side's title additionally through the `or ""` truthiness
/// gate, the YTM artist through the same `extract_artist` engine the
/// Python side's module global resolves to when the extension is
/// built), both ratios through the module's own `fuzz` attribute read
/// at call time — so the installed thefuzz, and any test stand-in
/// patched over it, steer the native engine exactly like the fallback —
/// and the weighted sum `int(title_score * TITLE_MATCH_WEIGHT +
/// artist_score * ARTIST_MATCH_WEIGHT)` computed through CPython's own
/// multiply, add and `int()` with the weights read from the module at
/// call time, so patched weights steer both backends identically and
/// the truncation, float rounding and error shapes stay CPython's.
#[pyfunction]
#[pyo3(signature = (spotify_track, ytm_track))]
pub(crate) fn spotify_fuzzy_score<'py>(
    py: Python<'py>,
    spotify_track: &Bound<'py, PyAny>,
    ytm_track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let si = py.import("ytm_player.services.spotify_import")?;
    let fuzz = si.getattr("fuzz")?;
    let title_weight = si.getattr("TITLE_MATCH_WEIGHT")?;
    let artist_weight = si.getattr("ARTIST_MATCH_WEIGHT")?;

    let empty: Bound<'py, PyAny> = PyString::new(py, "").into_any();
    // spotify_track.get("name", "").lower()
    let sp_title = spotify_track
        .call_method1("get", ("name", empty.clone()))?
        .call_method0("lower")?;
    // spotify_track.get("artist", "").lower()
    let sp_artist = spotify_track
        .call_method1("get", ("artist", empty.clone()))?
        .call_method0("lower")?;
    // (ytm_track.get("title", "") or "").lower()
    let raw_title = ytm_track.call_method1("get", ("title", empty.clone()))?;
    let ytm_title = if raw_title.is_truthy()? {
        raw_title
    } else {
        empty
    }
    .call_method0("lower")?;
    // extract_artist(ytm_track).lower() — the shared native engine; a
    // truthy non-dict ytm_track is outside the ported contract (every
    // search result is a dict) and reports a TypeError rather than the
    // fallback's duck-typed AttributeError.
    let ytm_dict = ytm_track
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("ytm_track must be a dict"))?;
    let ytm_artist = PyString::new(py, &extract_artist(py, ytm_dict)).call_method0("lower")?;

    // fuzz.ratio(sp_title, ytm_title) / fuzz.ratio(sp_artist, ytm_artist)
    let ratio = fuzz.getattr("ratio")?;
    let title_score = ratio.call1((sp_title, ytm_title))?;
    let artist_score = ratio.call1((sp_artist, ytm_artist))?;
    // int(title_score * TITLE_MATCH_WEIGHT + artist_score * ARTIST_MATCH_WEIGHT)
    let weighted = title_score
        .mul(&title_weight)?
        .add(&artist_score.mul(&artist_weight)?)?;
    builtin_int(py, &weighted)
}

/// The score-sort-threshold classification behind `_search_and_score`'s
/// post-search walk — the pure core of the import matcher's per-track
/// decision, from `services/spotify_import.py`.
///
/// Mirrors the pure-Python reference (`_match_classification`): the
/// empty-results gate first — `not search_results` through Python's own
/// truthiness — answering a `MatchType.NONE` result with the dataclass's
/// own default `candidates`/`selected`. Otherwise every candidate is
/// scored through the module's own `_fuzzy_score` (read at call time,
/// so the native override and any patched scorer steer the engine
/// exactly like the fallback), the `(score, candidate)` pairs are
/// sorted by CPython's own `list.sort(key=..., reverse=True)` — routed
/// through the list's own sort method with `operator.itemgetter(0)` as
/// the key, so the stable reverse ordering, the `<`-only comparisons
/// over the score objects and the sort's own error shapes stay
/// CPython's — and the leading pair gates the verdict:
/// `best_score >= AUTO_MATCH_THRESHOLD` (the threshold read from the
/// module at call time) answers a `MatchType.EXACT` result whose
/// `selected` is the very best-candidate object and whose `candidates`
/// is the best-first list, every other score a `MatchType.MULTIPLE`
/// result over the same list. The `MatchResult`/`MatchType` classes are
/// read from the module at call time too, so the results are the very
/// dataclass instances the rest of the import flow consumes; the
/// `ytmusic.search` call, the query build and the exception arm that
/// answers empty results for a failed search stay Python-side in the
/// wrapper.
#[pyfunction]
#[pyo3(signature = (spotify_track, search_results))]
pub(crate) fn spotify_match_classification<'py>(
    py: Python<'py>,
    spotify_track: &Bound<'py, PyAny>,
    search_results: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let si = py.import("ytm_player.services.spotify_import")?;
    let match_result_cls = si.getattr("MatchResult")?;
    let match_type_cls = si.getattr("MatchType")?;

    // if not search_results: return MatchResult(..., MatchType.NONE)
    if !search_results.is_truthy()? {
        let none_type = match_type_cls.getattr("NONE")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("spotify_track", spotify_track)?;
        kwargs.set_item("match_type", none_type)?;
        return match_result_cls.call((), Some(&kwargs));
    }

    // scored = [(_fuzzy_score(sp_track, c), c) for c in search_results]
    let fuzzy_score = si.getattr("_fuzzy_score")?;
    let scored = PyList::empty(py);
    for candidate in search_results.try_iter()? {
        let candidate = candidate?;
        let score = fuzzy_score.call1((spotify_track, &candidate))?;
        scored.append((score, &candidate))?;
    }

    // scored.sort(key=lambda x: x[0], reverse=True) — CPython's own
    // sort, with `operator.itemgetter(0)` spelling the reference's key
    // lambda (both read `x[0]` through the subscript protocol).
    let itemgetter = py
        .import("operator")?
        .getattr("itemgetter")?
        .call1((0i64,))?;
    let sort_kwargs = PyDict::new(py);
    sort_kwargs.set_item("key", itemgetter)?;
    sort_kwargs.set_item("reverse", true)?;
    scored.call_method("sort", (), Some(&sort_kwargs))?;

    // best_score, best_candidate = scored[0] — the reference's own
    // IndexError for a truthy-but-empty search_results (a generator).
    let best = scored.get_item(0)?;
    let best_score = best.get_item(0)?;
    let best_candidate = best.get_item(1)?;

    // candidates = [c for _, c in scored]
    let candidates = PyList::empty(py);
    for entry in scored.iter() {
        candidates.append(entry.get_item(1)?)?;
    }

    // if best_score >= AUTO_MATCH_THRESHOLD: EXACT else MULTIPLE
    let threshold = si.getattr("AUTO_MATCH_THRESHOLD")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("spotify_track", spotify_track)?;
    if best_score.ge(&threshold)? {
        kwargs.set_item("match_type", match_type_cls.getattr("EXACT")?)?;
        kwargs.set_item("candidates", candidates)?;
        kwargs.set_item("selected", best_candidate)?;
    } else {
        kwargs.set_item("match_type", match_type_cls.getattr("MULTIPLE")?)?;
        kwargs.set_item("candidates", candidates)?;
    }
    match_result_cls.call((), Some(&kwargs))
}

/// One page's parsed track rows — the pure core of the spotipy
/// pagination walk behind `extract_spotify_tracks_spotipy`, from
/// `services/spotify_import.py`.
///
/// Mirrors the pure-Python reference (`_page_items`): the page's
/// `items` member read through the mapping's own `get` with a fresh
/// empty list default (a missing key answers no items, exactly like
/// the reference), each item normalised through the module's own
/// `_parse_spotipy_item` (read at call time, so the native override
/// steers the engine exactly like the fallback), and each falsy parse
/// — the empty dict the normaliser answers for an episode entry or a
/// region-locked track — dropped by the `if parsed:` truthiness gate.
/// The `while results` pagination loop, the `sp.next` fetch and the
/// `next` member gate stay Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (results))]
pub(crate) fn spotify_page_items<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let si = py.import("ytm_player.services.spotify_import")?;
    let parse_item = si.getattr("_parse_spotipy_item")?;
    // results.get("items", []) — a fresh empty list as the default.
    let empty_items: Bound<'py, PyAny> = PyList::empty(py).into_any();
    let items = results.call_method1("get", ("items", empty_items))?;
    let out = PyList::empty(py);
    for item in items.try_iter()? {
        let item = item?;
        let parsed = parse_item.call1((item,))?;
        if parsed.is_truthy()? {
            out.append(parsed)?;
        }
    }
    Ok(out)
}

/// The scraper fallback's track rows — the pure core of the
/// `spotify_scraper` walk behind `extract_spotify_tracks`, from
/// `services/spotify_import.py`.
///
/// Mirrors the pure-Python reference (`_scraper_track_rows`): the
/// playlist's `tracks` walked through the attribute and iteration
/// protocols, each item's `track` read through its own attribute, the
/// artist as the `", "`-joined `name`s of `track.artists` — the join
/// through the str's own `.join` method over the very name objects,
/// so a non-string name raises the join's own TypeError — with the `or
/// "Unknown"` truthiness gate behind it (no artists, or a join that
/// answers an empty string, reads as `Unknown`), the album name read
/// only through a truthy `track.album` (a falsy album answers `""`),
/// and `name`/`duration_ms` passing through as the objects they are,
/// into a fresh four-key dict per row in the reference's insertion
/// order. The playlist name and total-count reads, and the
/// short-pagination warning, stay Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (playlist))]
pub(crate) fn spotify_scraper_track_rows<'py>(
    py: Python<'py>,
    playlist: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    // for item in playlist.tracks:
    let tracks = playlist.getattr("tracks")?;
    for item in tracks.try_iter()? {
        let item = item?;
        // track = item.track
        let track = item.getattr("track")?;
        // artist_name = ", ".join(a.name for a in track.artists) or "Unknown"
        let artists = track.getattr("artists")?;
        let names = PyList::empty(py);
        for artist in artists.try_iter()? {
            names.append(artist?.getattr("name")?)?;
        }
        let joined: Bound<'py, PyAny> = PyString::new(py, ", ").call_method1("join", (names,))?;
        let artist_name: Bound<'py, PyAny> = if joined.is_truthy()? {
            joined
        } else {
            PyString::new(py, "Unknown").into_any()
        };
        // album_name = track.album.name if track.album else ""
        let album: Bound<'py, PyAny> = match track.getattr("album")? {
            album if album.is_truthy()? => album.getattr("name")?,
            _ => PyString::new(py, "").into_any(),
        };
        let row = PyDict::new(py);
        row.set_item("name", track.getattr("name")?)?;
        row.set_item("artist", artist_name)?;
        row.set_item("album", album)?;
        row.set_item("duration_ms", track.getattr("duration_ms")?)?;
        out.append(row)?;
    }
    Ok(out)
}

/// One candidate's display row — the pure core of the import prompt's
/// candidate line, from `services/spotify_import.py`'s
/// `_display_candidate`.
///
/// Mirrors the pure-Python reference: the title read duck-typed through
/// the mapping's own `get` with the `"?"` default, the artist through
/// the same `extract_artist` engine the Python side's module global
/// resolves to when the extension is built, the duration through the
/// same `extract_duration` engine with its `if dur_sec else "?"`
/// truthiness gate rendering through the shared `format_duration`, the
/// `resultType` read with the `""` default gating the ` [type]` suffix
/// (only a truthy type that is not `"song"` — the comparison through
/// CPython's own `!=` — earns the suffix), and the whole line composed
/// through the same f-string shape — every interpolated object through
/// `str()`, the two leading spaces, the `. ` index separator and the
/// em-dash separator around the artist kept verbatim.
#[pyfunction]
#[pyo3(signature = (idx, candidate))]
pub(crate) fn spotify_display_candidate(
    py: Python,
    idx: &Bound<'_, PyAny>,
    candidate: &Bound<'_, PyAny>,
) -> PyResult<String> {
    // title = candidate.get("title", "?")
    let title = candidate.call_method1("get", ("title", "?"))?;
    // artist_str = extract_artist(candidate) — a non-dict candidate is
    // outside the ported contract (every search result is a dict) and
    // reports a TypeError rather than the fallback's AttributeError.
    let dict = candidate
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("candidate must be a dict"))?;
    let artist_str = extract_artist(py, dict);
    // dur_sec = extract_duration(candidate); duration = format_duration(...) if dur_sec else "?"
    let dur_sec = extract_duration(py, dict);
    let duration = if dur_sec != 0 {
        format_duration(dur_sec)
    } else {
        "?".to_string()
    };
    // result_type = candidate.get("resultType", "")
    let result_type = candidate.call_method1("get", ("resultType", ""))?;
    // suffix = f" [{result_type}]" if result_type and result_type != "song" else ""
    let suffix = if result_type.is_truthy()? && result_type.ne("song")? {
        format!(" [{}]", py_str(&result_type)?)
    } else {
        String::new()
    };
    // f"  {idx}. {title} — {artist_str} ({duration}){suffix}"
    Ok(format!(
        "  {}. {} \u{2014} {} ({}){}",
        py_str(idx)?,
        py_str(&title)?,
        artist_str,
        duration,
        suffix
    ))
}

/// The confirmed/video-ids/skipped accounting behind the import flow's
/// step-6 collection — the pure core of `services/spotify_import.py`'s
/// `_summarize_results`.
///
/// Mirrors the pure-Python reference: the confirmed walk keeps every
/// result whose `selected` attribute `is not None` (the identity test,
/// never truthiness — a falsy-but-set candidate still confirms), the
/// results pass through as the very objects they are; `skipped` is
/// `len(results) - len(confirmed)` — the lengths through the objects'
/// own `__len__` — so every unconfirmed result counts exactly once
/// (the off-by-99 the reference was written to fix); and the playable
/// id list re-reads each confirmed result's `selected`, keeps the
/// truthy ones, reads each id through the same native `get_video_id`
/// the module's global resolves to when the extension is built, and
/// drops the empty answers — so a confirmed-but-unplayable result is
/// not counted as skipped. The answer is the reference's
/// `(confirmed, video_ids, skipped)` triple.
#[pyfunction]
#[pyo3(signature = (results))]
pub(crate) fn spotify_summarize_results<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyList>, i64)> {
    // confirmed = [r for r in results if r.selected is not None]
    let confirmed = PyList::empty(py);
    for r in results.try_iter()? {
        let r = r?;
        if !r.getattr("selected")?.is_none() {
            confirmed.append(r)?;
        }
    }
    // skipped = len(results) - len(confirmed)
    let skipped = results.len()? as i64 - confirmed.len() as i64;
    // video_ids = [get_video_id(r.selected) for r in confirmed if r.selected]
    // video_ids = [vid for vid in video_ids if vid]
    let video_ids = PyList::empty(py);
    for r in confirmed.iter() {
        let selected = r.getattr("selected")?;
        if !selected.is_truthy()? {
            continue;
        }
        // get_video_id(selected) — the shared native engine; a truthy
        // non-dict selection is outside the ported contract and reports
        // a TypeError rather than the fallback's duck-typed behavior.
        let dict = selected
            .downcast::<PyDict>()
            .map_err(|_| PyTypeError::new_err("selected candidate must be a dict"))?;
        let vid = get_video_id(py, dict);
        if !vid.is_empty() {
            video_ids.append(vid)?;
        }
    }
    Ok((confirmed, video_ids, skipped))
}

/// The match-type bucket walk behind the import flow's result
/// categorisation — the pure core of the exact/multiple/none splits in
/// `services/spotify_import.py`'s `run_import`.
///
/// Mirrors the pure-Python reference (`_bucket_results`): every result
/// whose `match_type` attribute compares equal — through CPython's own
/// `==`, so enum identity, an overridden `__eq__` and any equal-but-
/// distinct spelling all classify exactly like the reference — to the
/// given `MatchType` member passes through as the very object it is,
/// in the results' own order, into a fresh list. The interactive
/// resolution steps that flip match types between buckets stay
/// Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (results, match_type))]
pub(crate) fn spotify_bucket_results<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
    match_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for r in results.try_iter()? {
        let r = r?;
        if r.getattr("match_type")?.eq(match_type)? {
            out.append(r)?;
        }
    }
    Ok(out)
}
