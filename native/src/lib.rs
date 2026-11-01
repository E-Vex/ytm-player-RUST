use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PyString, PyTuple};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};
use unicode_bidi::bidi_class;

/// Regex for validating YouTube video IDs.
static VALID_VIDEO_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9_-]{1,64}$").expect("valid video id regex"));

/// Matches view-count pseudo-artist runs like "358K views", "3.8M views",
/// "1,234 views". Used only when the track carries no `views` field.
static VIEW_COUNT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[\d.,]+\s*[KMB]?\s*views$").expect("valid view count regex")
});

/// Patterns stripped from titles during lyric-lookup sanitization.
/// `(?xi)`: case-insensitive + verbose, matching the Python fallback's
/// `re.IGNORECASE | re.VERBOSE`.
static LYRIC_NOISE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?xi)
        \s*
        [\(\[]\s*
        (?:
            official\s*(?:music\s*)?(?:video|audio|lyric|lyrics)
          | lyrics?\s*video
          | official
          | audio
          | video
          | hd
          | 4k
          | (?:feat\.?|ft\.?|featuring)\b(?:[^()\]]|\([^)]*\))*
          | remaster(?:ed)?(?:\s+\d{4})?
          | (?:[^)\]]+\s+)?remix(?:\s+[^)\]]*)?
          | deluxe(?:\s+edition)?
          | live(?:\s+[^)\]]*)?
          | acoustic(?:\s+[^)\]]*)?
        )
        \s*[\)\]]
        \s*
    "#,
    )
    .expect("valid lyric noise regex")
});

static COLLAPSE_WS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s{2,}").expect("valid collapse ws regex"));

/// Pattern for "Daily Top 100 Songs" shelf-title rewrite.
static SHELF_DAILY_TOP_100_SONGS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bDaily Top 100 Songs\b").expect("valid shelf title regex"));

/// Pattern for "Daily Top Music Videos" shelf-title rewrite.
static SHELF_DAILY_TOP_MUSIC_VIDEOS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bDaily Top Music Videos\b").expect("valid shelf title regex"));

/// Pattern for "Daily Top Songs on Shorts" shelf-title rewrite.
static SHELF_DAILY_TOP_SHORTS_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bDaily Top Songs on Shorts\b").expect("valid shelf title regex")
});

/// Stream URLs in diagnostics may contain the public IP and signed playback
/// parameters; the whole URL is removed rather than guessing which query
/// fields are sensitive. Mirrors `doctor._URL_PATTERN`.
static DIAG_URL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\b(?:https?|tcp)://[^\s<>"']+"#).expect("valid diagnostic url regex")
});

/// Header-value pairs redacted from diagnostics before the report is shown.
/// Each keeps its prefix (group 1) and replaces the value with `[REDACTED]`.
/// Mirrors `doctor._REDACT_PATTERNS` (same order, same grouping).
static DIAG_REDACT_RES: LazyLock<[Regex; 6]> = LazyLock::new(|| {
    [
        Regex::new(r"(?i)(authorization\s*[:=]\s*)(.+)").expect("valid redact regex"),
        Regex::new(r"(?i)(cookie\s*[:=]\s*)(.+)").expect("valid redact regex"),
        Regex::new(r"(?i)(bearer\s+)(\S+)").expect("valid redact regex"),
        Regex::new(r"(?i)(token\s*[:=]\s*)(\S+)").expect("valid redact regex"),
        Regex::new(r"(?i)(x-goog-pageid\s*[:=]\s*)(\S+)").expect("valid redact regex"),
        Regex::new(r"(SAPISID\s*=\s*)(\S+)").expect("valid redact regex"),
    ]
});

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn _ytm_player_native(_py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(format_duration, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(truncate, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(format_count, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(format_size, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(get_video_id, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(is_valid_video_id, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(extract_artist, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(extract_duration, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(normalize_tracks, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(strip_vl_prefix, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(build_playlist_subtitle, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        sanitize_title_for_lyric_lookup,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(format_ago, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(clean_shelf_title, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(normalise_region, module)?)?;

    // BiDi text utilities (ytm_player.utils.bidi equivalents).
    module.add_function(pyo3::wrap_pyfunction!(has_rtl, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(isolate_bidi, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(reorder_line, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(wrap_line, module)?)?;

    // Transliteration (ytm_player.utils.transliteration equivalent).
    module.add_function(pyo3::wrap_pyfunction!(transliterate, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(has_non_ascii, module)?)?;

    // Diagnostics (ytm_player.utils.doctor / logging equivalents).
    module.add_function(pyo3::wrap_pyfunction!(redact, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(argv_is_ytm, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(filter_log_lines, module)?)?;

    // yt-dlp option normalisation (ytm_player.services.yt_dlp_options
    // equivalents). The path expanders (normalize_cookiefile /
    // normalize_cafile) stay in Python: pathlib's expanduser handles
    // `~otheruser` via the pwd database, which has no std equivalent.
    module.add_function(pyo3::wrap_pyfunction!(normalize_remote_components, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(normalize_js_runtimes, module)?)?;

    // IPC single-instance guard (ytm_player.ipc equivalent).
    module.add_function(pyo3::wrap_pyfunction!(looks_like_ytm, module)?)?;

    // Keymap sequence matching (ytm_player.config.keymap equivalent).
    module.add_class::<KeyMatcher>()?;
    // Key-name normalisation (app/_keys.py equivalent): the
    // Textual-key-to-keymap-name translation that feeds sequence
    // matching — modifier prefixes, printable-character preference,
    // and the special-name remap.
    module.add_function(pyo3::wrap_pyfunction!(normalize_key, module)?)?;

    // Playback queue engine (ytm_player.services.queue equivalent).
    module.add_class::<QueueCore>()?;

    // Track-table view engine (ui/widgets/track_table.py equivalent):
    // the filter + sort pass that computes the visible row order.
    module.add_function(pyo3::wrap_pyfunction!(filter_sort_tracks, module)?)?;

    // Session-state validation engine (app/_session.py equivalent):
    // the one-pass validation + coercion of the parsed session.json
    // object, plus the resume-point scan over the restored queue.
    module.add_function(pyo3::wrap_pyfunction!(coerce_session_state, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(find_track_index, module)?)?;

    // Recently Played history-merge engine (ui/pages/recently_played.py
    // equivalent): the account-history cache view model — occurrence
    // stamping/identity, accepted-play supersession, pending-play parking
    // and merging, and the All-tab local+account composition.
    module.add_function(pyo3::wrap_pyfunction!(stamp_track, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(occurrence_key, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(with_accepted_play, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(with_accepted_local_play, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(park_pending_play, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(merge_pending_account_plays, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(enrich_track, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(compose_all_rows, module)?)?;

    // Navigation stack engine (app/_navigation.py equivalent): the
    // back/forward stack machine behind page navigation — snapshotting
    // the page being left, the stack caps, the same-page-back shortcut,
    // forward-history invalidation and cached-state restore — plus the
    // purge that drops every stack/cache reference to deleted playlists.
    module.add_function(pyo3::wrap_pyfunction!(resolve_navigation, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(purge_dead_playlist_refs, module)?)?;

    // Browse home-feed shelf classification (ui/pages/browse.py
    // equivalent): the event-vs-chart partition with priority pill
    // ordering behind the Charts tab, and the playlist-row extraction
    // with per-id dedup behind the Playlists tab.
    module.add_function(pyo3::wrap_pyfunction!(is_event_shelf, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(chart_sort_key, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(split_events_and_charts, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(is_playlist_row, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(playlist_rows, module)?)?;

    // Search-result label formatting (ui/pages/search.py equivalent):
    // the item-to-label composer behind the Albums/Artists/Playlists
    // panels of the search results area.
    module.add_function(pyo3::wrap_pyfunction!(format_search_result, module)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get an optional string value from a dict.
fn get_optional_string<'py>(
    _py: Python<'py>,
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(v) => {
            if v.is_none() {
                return Ok(None);
            }
            let s: String = v.extract()?;
            if s.is_empty() {
                Ok(None)
            } else {
                Ok(Some(s))
            }
        }
        None => Ok(None),
    }
}

/// Get an optional PyAny value from a dict (alias for clarity).
fn get_optional_any<'py>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    dict.get_item(key)
}

// ---------------------------------------------------------------------------
// Pure formatting functions
// ---------------------------------------------------------------------------

/// Format seconds as H:MM:SS or M:SS. Negative values clamp to 0.
#[pyfunction]
fn format_duration(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes}:{secs:02}")
    }
}

/// Truncate *text* to *max_len* characters, appending `…` on overflow.
#[pyfunction]
fn truncate(text: &str, max_len: isize) -> String {
    truncate_str(text, max_len)
}

/// The `truncate` body, shared with `format_search_result` so the label
/// composer truncates through exactly one definition.
fn truncate_str(text: &str, max_len: isize) -> String {
    if max_len < 1 {
        return String::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_len as usize {
        return text.to_string();
    }
    if max_len <= 1 {
        return chars[..max_len as usize].iter().collect();
    }
    let mut out: String = chars[..(max_len as usize - 1)].iter().collect();
    out.push('…');
    out
}

/// Format a count with K/M/B suffixes.
#[pyfunction]
fn format_count(n: i64) -> String {
    let abs = n.abs();
    if abs >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if abs >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if abs >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Format a byte size with B/KB/MB/GB/TB/PB suffixes.
#[pyfunction]
fn format_size(bytes_val: i64) -> String {
    let mut size = bytes_val as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if size.abs() < 1024.0 {
            if unit == "B" {
                return format!("{} {}", size as i64, unit);
            }
            return format!("{:.1} {}", size, unit);
        }
        size /= 1024.0;
    }
    format!("{:.1} PB", size)
}

/// Extract video ID from a track dict (checks both `videoId` and `video_id`).
#[pyfunction]
fn get_video_id(_py: Python, track: &Bound<'_, PyDict>) -> String {
    for key in ["videoId", "video_id"] {
        if let Ok(Some(v)) = track.get_item(key) {
            if !v.is_none() {
                if let Ok(s) = v.extract::<String>() {
                    if !s.is_empty() {
                        return s;
                    }
                }
            }
        }
    }
    String::new()
}

/// True if *video_id* matches the YouTube video ID pattern.
#[pyfunction]
fn is_valid_video_id(vid: &str) -> bool {
    VALID_VIDEO_ID_RE.is_match(vid)
}

/// Strip the `VL` prefix from a playlist ID if present.
#[pyfunction]
fn strip_vl_prefix(playlist_id: &str) -> String {
    if let Some(rest) = playlist_id.strip_prefix("VL") {
        rest.to_string()
    } else {
        playlist_id.to_string()
    }
}

/// Normalise a YouTube chart shelf title for compact display.
/// Mirrors `ytm_player.utils.formatting.clean_shelf_title`.
/// *region_names* is a Python list of region display names (e.g. "United Kingdom").
#[pyfunction]
fn clean_shelf_title(_py: Python, raw: &str, region_names: &Bound<'_, PyList>) -> String {
    let mut s = raw.trim().to_string();

    // Collect region display names from the Python list.
    let names: Vec<String> = region_names
        .iter()
        .filter_map(|item| item.extract::<String>().ok())
        .collect();

    // Strip " - Country" suffix (split on the LAST " - ").
    if let Some(idx) = s.rfind(" - ") {
        let tail = &s[idx + 3..];
        if names.iter().any(|name| tail == *name) {
            s = s[..idx].trim().to_string();
        }
    }

    // Strip " Country" suffix.
    for name in &names {
        let suffix = format!(" {name}");
        if s.ends_with(&suffix) {
            s = s[..s.len() - suffix.len()].trim().to_string();
            break;
        }
    }

    // Regex rewrites (applied in order, all occurrences).
    s = SHELF_DAILY_TOP_100_SONGS_RE
        .replace_all(&s, "Daily Top 100")
        .to_string();
    s = SHELF_DAILY_TOP_MUSIC_VIDEOS_RE
        .replace_all(&s, "Daily Top Videos")
        .to_string();
    s = SHELF_DAILY_TOP_SHORTS_RE
        .replace_all(&s, "Daily Top Songs (Shorts)")
        .to_string();

    s
}

/// Normalise a YouTube chart region value to an ISO 3166-1 alpha-2 code.
/// Mirrors `ytm_player.services.regions.normalise_region`.
///
/// Accepts a bare code ("ES", "es") or a language-territory locale
/// ("en-GB", "es_ES", " EN-gb ") and returns the territory uppercased.
/// Non-locale input is returned uppercased as-is; empty input stays empty.
#[pyfunction]
fn normalise_region(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }

    let replaced = value.trim().replace('_', "-");

    // Partition on the first '-', mirroring str.partition("-").
    let (head, territory) = match replaced.split_once('-') {
        Some((h, t)) => (h, Some(t)),
        None => (replaced.as_str(), None),
    };

    if let Some(t) = territory {
        if head.len() >= 2
            && head.len() <= 3
            && head.is_ascii()
            && head.chars().all(char::is_alphabetic)
            && t.len() == 2
            && t.is_ascii()
            && t.chars().all(char::is_alphabetic)
        {
            return t.to_uppercase();
        }
    }

    head.trim().to_uppercase()
}

/// Build the metadata subtitle line for a playlist header.
/// Format: `Owner · Privacy · Year · N tracks`
/// *year* is a PyAny that may be None, int, or str.
#[pyfunction]
fn build_playlist_subtitle(
    owner: &str,
    privacy: &str,
    year: &Bound<'_, PyAny>,
    track_count: i64,
) -> PyResult<String> {
    let mut parts: Vec<String> = vec![owner.to_string()];
    if !privacy.is_empty() {
        parts.push(capitalize_first(privacy));
    }
    if !year.is_none() {
        let y_str = year.to_string();
        if !y_str.is_empty() && y_str != "None" {
            parts.push(y_str);
        }
    }
    let track_label = if track_count == 1 { "track" } else { "tracks" };
    parts.push(format!("{track_count} {track_label}"));
    Ok(parts.join(" \u{b7} "))
}

fn capitalize_first(s: &str) -> String {
    // Mirror Python's str.capitalize(): first char uppercased, the rest
    // lowercased ("PUBLIC" -> "Public", not "PUBLIC").
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => {
            let mut out: String = f.to_uppercase().collect();
            out.extend(c.flat_map(char::to_lowercase));
            out
        }
    }
}

// ---------------------------------------------------------------------------
// Artist / duration extraction
// ---------------------------------------------------------------------------

fn is_view_count_run(entry: &Bound<'_, PyAny>, views: Option<&Bound<'_, PyAny>>) -> bool {
    // Must be a dict with no `id`.
    if !entry.is_instance_of::<PyDict>() {
        return false;
    }
    let dict = entry.downcast::<PyDict>().unwrap();
    match dict.get_item("id") {
        Ok(Some(v)) if !v.is_none() => return false,
        _ => {}
    }
    let name = dict
        .get_item("name")
        .ok()
        .flatten()
        .and_then(|v| v.extract::<String>().ok())
        .filter(|s| !s.is_empty())
        .unwrap_or_default();
    if name.is_empty() {
        return false;
    }
    if let Some(views) = views {
        if let Ok(vs) = views.downcast::<PyString>() {
            if let Ok(vs_str) = vs.to_str() {
                return name == vs_str;
            }
        }
    }
    VIEW_COUNT_RE.is_match(name.as_str())
}

fn extract_artist_names<'py>(
    py: Python<'py>,
    artists: &Bound<'py, PyList>,
    views: Option<&Bound<'py, PyAny>>,
) -> Vec<String> {
    let mut names = Vec::new();
    for item in artists.iter() {
        if is_view_count_run(&item, views) {
            continue;
        }
        if let Ok(dict) = item.downcast::<PyDict>() {
            if let Ok(Some(name)) = get_optional_string(py, dict, "name") {
                names.push(name);
            }
        } else if let Ok(s) = item.str().and_then(|s| s.to_str().map(|s| s.to_string())) {
            // Mirrors the Python reference's ``str(a)`` for non-dict runs.
            if !s.is_empty() {
                names.push(s);
            }
        }
    }
    names
}

fn parse_duration_string(dur: &str) -> Option<i64> {
    let parts: Vec<&str> = dur.split(':').collect();
    match parts.len() {
        2 => {
            let m = parts[0].trim().parse::<i64>().ok()?;
            let s = parts[1].trim().parse::<i64>().ok()?;
            Some(m * 60 + s)
        }
        3 => {
            let h = parts[0].trim().parse::<i64>().ok()?;
            let m = parts[1].trim().parse::<i64>().ok()?;
            let s = parts[2].trim().parse::<i64>().ok()?;
            Some(h * 3600 + m * 60 + s)
        }
        _ => None,
    }
}

/// Extract duration in seconds from various track dict formats.
/// Mirrors the Python reference exactly: ``duration_seconds`` goes through
/// ``int()`` (int, numeric string, truncating float — never colon parsing);
/// ``duration``/``length`` accept plain ints and "M:SS"/"H:MM:SS" strings.
#[pyfunction]
fn extract_duration(_py: Python, track: &Bound<'_, PyDict>) -> i64 {
    // duration_seconds
    if let Ok(Some(dur)) = track.get_item("duration_seconds") {
        if let Ok(i) = dur.extract::<i64>() {
            return i;
        }
        // int("12") — numeric strings, no colon parsing.
        if let Some(s) = dur
            .downcast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
        {
            if let Ok(i) = s.trim().parse::<i64>() {
                return i;
            }
        }
        // int(12.7) — Python truncates floats toward zero.
        if let Ok(f) = dur.extract::<f64>() {
            return f as i64;
        }
    }
    // duration
    if let Ok(Some(dur)) = track.get_item("duration") {
        if let Ok(i) = dur.extract::<i64>() {
            return i;
        }
        if let Some(s) = dur
            .downcast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
        {
            if s.contains(':') {
                if let Some(secs) = parse_duration_string(s) {
                    return secs;
                }
            }
        }
    }
    // length
    if let Ok(Some(dur)) = track.get_item("length") {
        if let Ok(i) = dur.extract::<i64>() {
            return i;
        }
        if let Some(s) = dur
            .downcast::<PyString>()
            .ok()
            .and_then(|s| s.to_str().ok())
        {
            if let Some(secs) = parse_duration_string(s) {
                return secs;
            }
        }
    }
    0
}

/// Extract display-friendly artist string from a track dict.
#[pyfunction]
fn extract_artist(py: Python, track: &Bound<'_, PyDict>) -> String {
    if let Ok(Some(artist)) = track.get_item("artist") {
        if !artist.is_none() {
            if let Ok(s) = artist.extract::<String>() {
                if !s.is_empty() {
                    return s;
                }
            }
        }
    }
    if let Ok(Some(artists)) = track.get_item("artists") {
        let views = track.get_item("views").ok().flatten();
        let views_ref = views.as_ref();
        if let Ok(list) = artists.downcast::<PyList>() {
            let names = extract_artist_names(py, list, views_ref);
            if !names.is_empty() {
                return names.join(", ");
            }
        }
    }
    "Unknown".to_string()
}

// ---------------------------------------------------------------------------
// Track normalization
// ---------------------------------------------------------------------------

#[pyfunction]
fn normalize_tracks<'py>(
    py: Python<'py>,
    raw_tracks: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let normalized = PyList::empty(py);
    for item in raw_tracks.iter() {
        let track = match item.downcast::<PyDict>() {
            Ok(t) => t,
            Err(_) => continue,
        };

        let video_id = match get_optional_string(py, track, "videoId")? {
            Some(v) if !v.is_empty() => v,
            _ => match get_optional_string(py, track, "video_id")? {
                Some(v) if !v.is_empty() => v,
                _ => continue,
            },
        };

        let title = match get_optional_string(py, track, "title")? {
            Some(t) => t,
            _ => "Unknown".to_string(),
        };
        let artist = extract_artist(py, track);

        let raw_artists = get_optional_any(track, "artists")?;
        let views = get_optional_any(track, "views")?;
        // Mirror the Python reference: the ``artists`` list keeps the
        // ORIGINAL entries (dicts stay dicts), dropping only view-count
        // pseudo-artist runs — downstream consumers navigate by name/id.
        let mut artists_entries: Vec<Bound<'py, PyAny>> = Vec::new();
        if let Some(ref raw) = raw_artists {
            if let Ok(list) = raw.downcast::<PyList>() {
                let views_ref = views.as_ref();
                for item in list.iter() {
                    if !is_view_count_run(&item, views_ref) {
                        artists_entries.push(item);
                    }
                }
            }
        }

        let album_info = get_optional_any(track, "album")?;
        let (album, album_id) = match album_info {
            Some(ref a) if a.is_instance_of::<PyDict>() => {
                let dict = a.downcast::<PyDict>().unwrap();
                let album = get_optional_string(py, dict, "name")?.unwrap_or_default();
                let album_id = get_optional_string(py, dict, "id")?;
                (album, album_id)
            }
            Some(ref a) => {
                let album = a
                    .downcast::<PyString>()
                    .ok()
                    .and_then(|s| s.to_str().ok())
                    .map(|s| s.to_string())
                    .unwrap_or_default();
                (album, get_optional_string(py, track, "album_id")?)
            }
            None => (String::new(), get_optional_string(py, track, "album_id")?),
        };

        // Mirror the Python reference's
        // `next((t[k] for k in (...) if t.get(k) is not None), None)`:
        // a present-but-None value counts as absent, so the next key is
        // tried (a `duration: null` entry must leave duration None, not 0).
        let mut raw_dur: Option<Bound<'py, PyAny>> = None;
        for key in ["duration_seconds", "duration", "length"] {
            if let Some(v) = get_optional_any(track, key)? {
                if !v.is_none() {
                    raw_dur = Some(v);
                    break;
                }
            }
        }
        let duration = if raw_dur.is_some() {
            Some(extract_duration(py, track))
        } else {
            None
        };

        let thumbnail = {
            let thumbs = get_optional_any(track, "thumbnails")?
                .or_else(|| get_optional_any(track, "thumbnail").unwrap_or(None));
            let mut url: Option<String> = None;
            if let Some(ref t) = thumbs {
                if let Ok(list) = t.downcast::<PyList>() {
                    let len = list.len();
                    if len > 0 {
                        if let Ok(last) = list.get_item(len - 1) {
                            if let Ok(dict) = last.downcast::<PyDict>() {
                                url = get_optional_string(py, dict, "url")?;
                            }
                        }
                    }
                }
            }
            url
        };

        let is_video = track
            .get_item("isVideo")
            .ok()
            .flatten()
            .or_else(|| track.get_item("is_video").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        let like_status = get_optional_string(py, track, "likeStatus")?;
        let set_video_id = get_optional_string(py, track, "setVideoId")?;

        let entry = PyDict::new(py);
        // Key order mirrors the Python reference's dict literal so
        // serialized/iterated shapes are identical.
        entry.set_item("video_id", video_id.as_str())?;
        entry.set_item("title", title.as_str())?;
        entry.set_item("artist", artist.as_str())?;
        let artists_pylist = PyList::new(py, artists_entries)?;
        entry.set_item("artists", artists_pylist)?;
        entry.set_item("album", album.as_str())?;
        match &album_id {
            Some(aid) => entry.set_item("album_id", aid.as_str())?,
            None => entry.set_item("album_id", pyo3::types::PyNone::get(py))?,
        }
        match duration {
            Some(d) => entry.set_item("duration", d)?,
            None => entry.set_item("duration", pyo3::types::PyNone::get(py))?,
        }
        match &thumbnail {
            Some(thumb) => entry.set_item("thumbnail_url", thumb.as_str())?,
            None => entry.set_item("thumbnail_url", pyo3::types::PyNone::get(py))?,
        }
        entry.set_item("is_video", is_video)?;
        match &like_status {
            Some(ls) => entry.set_item("likeStatus", ls.as_str())?,
            None => entry.set_item("likeStatus", pyo3::types::PyNone::get(py))?,
        }
        match &set_video_id {
            Some(svid) => entry.set_item("setVideoId", svid.as_str())?,
            None => entry.set_item("setVideoId", pyo3::types::PyNone::get(py))?,
        }

        normalized.append(entry)?;
    }
    Ok(normalized)
}

/// Strip common noise from a track title for better LRCLIB matching.
#[pyfunction(signature = (title, artist=None))]
fn sanitize_title_for_lyric_lookup(title: &str, artist: Option<&str>) -> String {
    if title.is_empty() {
        return title.to_string();
    }
    let mut cleaned = LYRIC_NOISE_RE.replace_all(title, " ").to_string();
    cleaned = COLLAPSE_WS_RE.replace_all(&cleaned, " ").to_string();
    if let Some(artist) = artist {
        if !artist.is_empty() {
            let cleaned_lower = cleaned.to_lowercase();
            let prefix_lower = format!("{} - ", artist.to_lowercase());
            if cleaned_lower.starts_with(&prefix_lower) {
                // Python slices by ``len(prefix_lower)`` — code points, not
                // bytes — so a multi-byte lowercase mapping can't split a
                // char boundary or panic here.
                let skip = prefix_lower.chars().count();
                cleaned = cleaned.chars().skip(skip).collect();
            }
        }
    }
    cleaned = cleaned.trim().to_string();
    if cleaned.is_empty() {
        return title.to_string();
    }
    cleaned
}

/// Format a Unix timestamp (seconds since epoch) as a human-readable
/// "N units ago" string. Mirrors the Python ``format_ago`` behaviour.
#[pyfunction]
fn format_ago(timestamp_seconds: f64) -> String {
    let now = SystemTime::now();
    let then = match UNIX_EPOCH.checked_add(std::time::Duration::from_secs_f64(timestamp_seconds)) {
        Some(t) => t,
        None => return "just now".to_string(),
    };

    // A timestamp in the future (or exactly now) -> "just now".
    let Ok(delta) = now.duration_since(then) else {
        return "just now".to_string();
    };
    let total_seconds = delta.as_secs() as i64;

    if total_seconds < 60 {
        return if total_seconds == 1 {
            "1 second ago".to_string()
        } else {
            format!("{total_seconds} seconds ago")
        };
    }

    let minutes = total_seconds / 60;
    if minutes < 60 {
        return if minutes == 1 {
            "1 minute ago".to_string()
        } else {
            format!("{minutes} minutes ago")
        };
    }

    let hours = minutes / 60;
    if hours < 24 {
        return if hours == 1 {
            "1 hour ago".to_string()
        } else {
            format!("{hours} hours ago")
        };
    }

    let days = hours / 24;
    if days < 30 {
        return if days == 1 {
            "1 day ago".to_string()
        } else {
            format!("{days} days ago")
        };
    }

    let months = days / 30;
    if months < 12 {
        return if months == 1 {
            "1 month ago".to_string()
        } else {
            format!("{months} months ago")
        };
    }

    let years = days / 365;
    if years == 1 {
        "1 year ago".to_string()
    } else {
        format!("{years} years ago")
    }
}

// ---------------------------------------------------------------------------
// BiDi text utilities (mirrors ytm_player/utils/bidi.py)
// ---------------------------------------------------------------------------

/// Unicode directional isolate marks (UAX #9, since Unicode 6.3).
const FSI: char = '\u{2068}'; // FIRST STRONG ISOLATE
const PDI: char = '\u{2069}'; // POP DIRECTIONAL ISOLATE

/// Matches any character from RTL scripts (Arabic, Hebrew, Thaana, Syriac,
/// N'Ko).  Mirrors the `_RTL_RE` range set in `bidi.py` exactly.
fn is_rtl_char(c: char) -> bool {
    matches!(c,
        '\u{0590}'..='\u{05FF}'     // Hebrew
      | '\u{0600}'..='\u{06FF}'     // Arabic
      | '\u{0700}'..='\u{074F}'     // Syriac
      | '\u{0750}'..='\u{077F}'     // Arabic Supplement
      | '\u{0780}'..='\u{07BF}'     // Thaana
      | '\u{07C0}'..='\u{07FF}'     // N'Ko
      | '\u{08A0}'..='\u{08FF}'     // Arabic Extended-A
      | '\u{FB1D}'..='\u{FB4F}'     // Hebrew Presentation Forms
      | '\u{FB50}'..='\u{FDFF}'     // Arabic Presentation Forms-A
      | '\u{FE70}'..='\u{FEFF}'     // Arabic Presentation Forms-B
    )
}

/// Return True if text contains any RTL script characters.
#[pyfunction]
fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl_char)
}

/// Wrap *text* in FSI...PDI so its directional context cannot leak.
///
/// Matches `bidi.isolate_bidi`: skips empty text, skips pure-LTR text when
/// *only_if_rtl* is set, and never double-wraps.
#[pyfunction(signature = (text, *, only_if_rtl = true))]
fn isolate_bidi(text: &str, only_if_rtl: bool) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    if only_if_rtl && !text.chars().any(is_rtl_char) {
        return text.to_string();
    }
    let mut chars = text.chars();
    let first = chars.next();
    let last = chars.last();
    if first == Some(FSI) && last == Some(PDI) {
        return text.to_string();
    }
    format!("{FSI}{text}{PDI}")
}

/// Simplified BiDi direction classes: L, R, or N (neutral).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    L,
    R,
    N,
}

fn char_direction(c: char) -> Direction {
    match bidi_class(c) {
        unicode_bidi::BidiClass::R | unicode_bidi::BidiClass::AL | unicode_bidi::BidiClass::AN => {
            Direction::R
        }
        unicode_bidi::BidiClass::L => Direction::L,
        _ => Direction::N,
    }
}

/// Direction from the first strong character in *word*.
fn word_direction(word: &str) -> Direction {
    word.chars()
        .map(char_direction)
        .find(|d| *d != Direction::N)
        .unwrap_or(Direction::N)
}

/// UAX #9 rules P2/P3: base direction from first strong character.
fn paragraph_base_direction(text: &str) -> Direction {
    text.chars()
        .map(char_direction)
        .find(|d| *d != Direction::N)
        .unwrap_or(Direction::L)
}

/// Perform UAX #9 word-level reordering (L2 reversal rule).
fn do_reorder(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let base_level: usize = usize::from(paragraph_base_direction(text) == Direction::R);

    let levels: Vec<usize> = words
        .iter()
        .map(|word| match word_direction(word) {
            Direction::R => 1,
            Direction::L => {
                if base_level == 1 {
                    2
                } else {
                    0
                }
            }
            Direction::N => base_level,
        })
        .collect();
    if levels.is_empty() {
        return text.to_string();
    }

    let mut indices: Vec<usize> = (0..words.len()).collect();
    for level in (1..=*levels.iter().max().unwrap_or(&0)).rev() {
        let mut i = 0;
        while i < indices.len() {
            if levels[indices[i]] >= level {
                let mut j = i;
                while j < indices.len() && levels[indices[j]] >= level {
                    j += 1;
                }
                indices[i..j].reverse();
                i = j;
            } else {
                i += 1;
            }
        }
    }

    indices
        .iter()
        .map(|&idx| words[idx])
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Raw UAX #9 line reorder — always applies (the `bidi_mode` gate lives in
/// Python so it can consult user settings).
#[pyfunction]
fn reorder_line(text: &str) -> String {
    do_reorder(text)
}

/// Pre-wrap RTL text for multi-line display with reordering applied.
/// Raw version — the `bidi_mode` gate lives in Python.  Mirrors
/// `bidi.wrap_rtl_line` *as if reordering were enabled*.
#[pyfunction]
fn wrap_line(text: &str, width: i64) -> String {
    if text.is_empty() || !text.chars().any(is_rtl_char) {
        return text.to_string();
    }

    let char_count = text.chars().count() as i64;
    if char_count <= width || width <= 0 {
        return do_reorder(text);
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut lines: Vec<String> = Vec::new();
    let mut current_words: Vec<&str> = Vec::new();
    let mut current_len: i64 = 0;

    for word in words {
        let word_len = word.chars().count() as i64;
        let needed = if current_words.is_empty() {
            word_len
        } else {
            current_len + 1 + word_len
        };
        if needed <= width {
            current_words.push(word);
            current_len = needed;
        } else {
            if !current_words.is_empty() {
                lines.push(current_words.join(" "));
            }
            current_words = vec![word];
            current_len = word_len;
        }
    }
    if !current_words.is_empty() {
        lines.push(current_words.join(" "));
    }

    lines
        .iter()
        .map(|line| do_reorder(line))
        .collect::<Vec<String>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Transliteration (mirrors ytm_player/utils/transliteration.py)
// ---------------------------------------------------------------------------

/// Transliterate non-ASCII text to ASCII via the anyascii Rust crate.
/// Mirrors `transliteration.transliterate_line` (minus its Python-side
/// ASCII fast path and lru_cache, which are kept in Python).
#[pyfunction]
fn transliterate(text: &str) -> String {
    any_ascii::any_ascii(text)
}

/// Return True if *text* is non-empty and contains any non-ASCII characters.
/// Mirrors `ytm_player.utils.transliteration.has_non_ascii`.
#[pyfunction]
fn has_non_ascii(text: &str) -> bool {
    !text.is_empty() && !text.is_ascii()
}

// ---------------------------------------------------------------------------
// Diagnostics (mirrors ytm_player/utils/doctor.py and logging.py)
// ---------------------------------------------------------------------------

/// Redact recognized auth headers/tokens and HTTP(S)/TCP URLs from
/// diagnostic text. Mirrors `ytm_player.utils.doctor._redact` exactly:
/// each header pattern keeps its prefix and masks the rest of the line,
/// then whole URLs are replaced with `[URL REDACTED]`.
#[pyfunction]
fn redact(text: &str) -> String {
    let mut out = text.to_string();
    for re in DIAG_REDACT_RES.iter() {
        out = re.replace_all(&out, "${1}[REDACTED]").into_owned();
    }
    DIAG_URL_RE.replace_all(&out, "[URL REDACTED]").into_owned()
}

/// Mirror `pathlib.PurePath.name` for the platform's path syntax: split on
/// separators, drop empty and `.` components, and return the last remaining
/// component ("" when none). `..` is kept, matching pathlib.
fn py_path_name(path: &str) -> String {
    let separators: &[char] = if cfg!(windows) { &['/', '\\'] } else { &['/'] };
    let mut name = String::new();
    for component in path.split(separators) {
        if component.is_empty() || component == "." {
            continue;
        }
        name = component.to_string();
    }
    name
}

/// True when *argv* is a real ytm-player entry point. Mirrors
/// `ytm_player.utils.doctor._argv_is_ytm`: the `ytm` console script
/// (interpreter in argv[0], script path in argv[1]) or `python -m
/// ytm_player`; never an unrelated process whose cmdline merely mentions
/// a ytm-player path.
#[pyfunction]
fn argv_is_ytm(argv: Vec<String>) -> bool {
    let head_names: Vec<String> = argv.iter().take(2).map(|a| py_path_name(a)).collect();
    head_names.iter().any(|name| name == "ytm") || argv.iter().any(|a| a == "ytm_player")
}

/// Filter already-read log lines down to those at or above *min_level* and
/// return the last *n* joined together. Mirrors the pure filtering core of
/// `ytm_player.utils.logging.get_recent_log_lines` (file I/O stays in
/// Python): lines keep their trailing newlines, unknown levels apply no
/// filter, and the tail slice follows Python's `lines[-n:]` semantics.
#[pyfunction(signature = (lines, n, min_level = None))]
fn filter_log_lines(lines: Vec<String>, n: i64, min_level: Option<&str>) -> String {
    let mut kept = lines;
    if let Some(level) = min_level {
        const ORDER: [&str; 5] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
        let upper = level.to_uppercase();
        if let Some(threshold) = ORDER.iter().position(|o| *o == upper) {
            let wanted: Vec<String> = ORDER[threshold..]
                .iter()
                .map(|w| format!("[{w}]"))
                .collect();
            kept.retain(|line| wanted.iter().any(|w| line.contains(w.as_str())));
        }
    }
    // Mirror Python's `lines[-n:]`: positive n keeps the last n, zero keeps
    // everything (`lines[-0:]` is `lines[0:]`), negative n drops the leading
    // -n entries (empty when -n exceeds the length).
    let len = kept.len();
    let start = if n > 0 {
        len.saturating_sub(n as usize)
    } else if n == 0 {
        0
    } else {
        ((-n) as i128).min(len as i128) as usize
    };
    kept[start..].concat()
}

// ---------------------------------------------------------------------------
// yt-dlp option normalisation (mirrors ytm_player/services/yt_dlp_options.py)
// ---------------------------------------------------------------------------

/// Split a string by commas/whitespace and drop empties. Mirrors
/// `yt_dlp_options._split_csv_or_space`.
fn split_csv_or_space(value: &str) -> Vec<String> {
    value
        .replace(',', " ")
        .split_whitespace()
        .map(String::from)
        .collect()
}

/// Parse a `runtime[:path]` token into (name, optional path). Mirrors
/// `yt_dlp_options._parse_runtime_token`: an empty/whitespace spec or an
/// empty runtime name yields None; a colon-delimited path is kept only when
/// it is non-blank after stripping.
fn parse_runtime_token(spec: &str) -> Option<(String, Option<String>)> {
    let token = spec.trim();
    if token.is_empty() {
        return None;
    }
    let (runtime, path) = match token.split_once(':') {
        Some((head, tail)) => (head, Some(tail)),
        None => (token, None),
    };
    let name = runtime.trim().to_lowercase();
    if name.is_empty() {
        return None;
    }
    match path.map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => Some((name, Some(p.to_string()))),
        None => Some((name, None)),
    }
}

/// Return a yt-dlp compatible remote_components list, or None when unset or
/// blank. Mirrors `yt_dlp_options.normalize_remote_components`: a string is
/// split on commas/whitespace; any other non-None value is iterated and each
/// item is coerced with str() then stripped (empty results dropped).
#[pyfunction]
fn normalize_remote_components<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    if value.is_none() {
        return Ok(None);
    }
    let parts: Vec<String> = if let Ok(s) = value.extract::<String>() {
        split_csv_or_space(&s)
    } else {
        let mut parts = Vec::new();
        for item in value.try_iter()? {
            let item = item?;
            let text = item.str()?.to_str()?.trim().to_string();
            if !text.is_empty() {
                parts.push(text);
            }
        }
        parts
    };
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(PyList::new(py, parts)?))
}

/// Return a yt-dlp compatible js_runtimes dict, or None when unset or blank.
/// Mirrors `yt_dlp_options.normalize_js_runtimes`: a dict maps
/// lowercased/stripped names to their config sub-dicts verbatim (non-dict
/// configs become {}); a string or iterable of `runtime[:path]` tokens is
/// parsed into {"name": {"path": ...}} entries. Insertion order and the
/// later-duplicate-wins behaviour match Python dict semantics.
#[pyfunction]
fn normalize_js_runtimes<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if value.is_none() {
        return Ok(None);
    }
    let result = PyDict::new(py);

    if let Ok(dict) = value.downcast::<PyDict>() {
        for (runtime, config) in dict.iter() {
            let name = runtime.str()?.to_str()?.trim().to_lowercase();
            if name.is_empty() {
                continue;
            }
            if config.is_instance_of::<PyDict>() {
                result.set_item(name, config)?;
            } else {
                result.set_item(name, PyDict::new(py))?;
            }
        }
    } else {
        let specs: Vec<String> = if let Ok(s) = value.extract::<String>() {
            split_csv_or_space(&s)
        } else {
            let mut specs = Vec::new();
            for item in value.try_iter()? {
                let item = item?;
                specs.push(item.str()?.to_str()?.to_string());
            }
            specs
        };
        for spec in specs {
            if let Some((name, path)) = parse_runtime_token(&spec) {
                let config = PyDict::new(py);
                if let Some(p) = path {
                    config.set_item("path", p)?;
                }
                result.set_item(name, config)?;
            }
        }
    }

    if result.is_empty() {
        return Ok(None);
    }
    Ok(Some(result))
}

// ---------------------------------------------------------------------------
// IPC single-instance guard (mirrors ytm_player/ipc.py)
// ---------------------------------------------------------------------------

/// Mirror the lenient entry-point heuristic of
/// `ytm_player.ipc._looks_like_ytm`: any argument whose
/// separator-agnostic basename is `ytm` or starts with `ytm.`, or that
/// contains `ytm_player` anywhere, counts as ours; a single-element argv
/// whose basename starts with `python` also counts (the Windows
/// image-name-only probe cannot rule out `python -m ytm_player`). Errs on
/// the side of True — a false "ours" only preserves the block behavior,
/// while a false "not ours" could steal a live TUI's PID file.
#[pyfunction]
fn looks_like_ytm(argv: Vec<String>) -> bool {
    fn basename_lower(arg: &str) -> String {
        arg.replace('\\', "/")
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_lowercase()
    }

    for arg in &argv {
        let name = basename_lower(arg);
        if name == "ytm" || name.starts_with("ytm.") || arg.contains("ytm_player") {
            return true;
        }
    }
    if argv.len() == 1 {
        let name = basename_lower(&argv[0]);
        if name.starts_with("python") {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Keymap sequence matching (mirrors ytm_player/config/keymap.py)
// ---------------------------------------------------------------------------

/// Native matcher for the KeyMap binding table.
///
/// Mirrors the exact/pending/no-match semantics of `KeyMap.match`:
///
/// - `("exact", Some(action))` — the pressed sequence is itself a binding.
///   Exact matches win even when the sequence is also a proper prefix of a
///   longer binding (the Python reference checks `key_sequence in bindings`
///   before scanning prefixes).
/// - `("pending", None)` — the sequence is a proper prefix of at least one
///   longer binding, so the app should buffer it and wait for more keys.
///   The empty sequence is pending whenever any binding exists (every
///   `bound_seq[:0] == ()`), matching the Python fallback's slice
///   comparison.
/// - `("no_match", None)` — neither, so the app resets its key buffer.
///
/// The Python fallback scans the whole bindings dict for the pending check
/// on every keystroke; here both lookups are hash queries against tables
/// precomputed once per binding-table change.
///
/// *bindings* is the `KeyMap.bindings` dict mapping sequences (tuples of
/// key-name strings) to action names. Values may be `Action` members — the
/// str-enum extracts as its value string.
#[pyclass(frozen)]
struct KeyMatcher {
    /// Bound sequence → action name.
    exact: HashMap<Vec<String>, String>,
    /// Every proper prefix (including the empty prefix) of every bound
    /// sequence — the sequences that can still grow into a binding.
    prefixes: HashSet<Vec<String>>,
}

#[pymethods]
impl KeyMatcher {
    #[new]
    fn new(bindings: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut exact: HashMap<Vec<String>, String> = HashMap::new();
        let mut prefixes: HashSet<Vec<String>> = HashSet::new();
        for (seq_obj, action_obj) in bindings.iter() {
            let seq: Vec<String> = seq_obj.extract()?;
            let action: String = action_obj.extract()?;
            for end in 0..seq.len() {
                prefixes.insert(seq[..end].to_vec());
            }
            // Last write wins, like the Python dict the table came from.
            exact.insert(seq, action);
        }
        Ok(Self { exact, prefixes })
    }

    /// Classify *seq* against the binding table.
    /// Returns ("exact", Some(action_name)) | ("pending", None)
    /// | ("no_match", None) — the action-name form of `KeyMap.match`.
    fn match_keys(&self, seq: Vec<String>) -> (String, Option<String>) {
        if let Some(action) = self.exact.get(&seq) {
            return ("exact".to_string(), Some(action.clone()));
        }
        if self.prefixes.contains(&seq) {
            return ("pending".to_string(), None);
        }
        ("no_match".to_string(), None)
    }
}

// ---------------------------------------------------------------------------
// Key-name normalisation (mirrors app/_keys.py's _normalize_key)
// ---------------------------------------------------------------------------

/// CPython's `str.isspace()` acceptance set for a single character.
/// Rust's `char::is_whitespace()` (the Unicode `White_Space` property)
/// would disagree on `\x1c`–`\x1f`, which CPython accepts, so the set is
/// spelled out instead of delegating.
fn py_isspace(c: char) -> bool {
    matches!(
        c,
        '\t' | '\n'
            | '\u{0b}'
            | '\u{0c}'
            | '\r'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{1f}'
            | ' '
            | '\u{85}'
            | '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

/// Convert a Textual key event's primitive components into the string
/// format used by `KeyMap`.
///
/// Mirrors `_normalize_key` from `app/_keys.py`, decomposed into the
/// event's primitives so the engine itself is pure: the key name, the
/// optional character, and the character's printability. Textual key
/// names like `ctrl+r` become `C-r`, `shift+tab` becomes `S-tab`,
/// `alt+v` becomes `M-v`; a single-code-point printable non-whitespace
/// character is itself (Textual names punctuation symbolically — `>`
/// arrives as `greater_than_sign` — while the keymap binds the literal
/// character; whitespace keeps its name); the special names below are
/// remapped and everything else passes through unchanged.
///
/// Faithful to the reference's gates: `len(character) == 1` counts code
/// points (a two-code-point decomposed character falls through to the
/// name mapping, exactly like Python's `len`), printability is the
/// caller-supplied `Key.is_printable`, and `isspace` is CPython's set
/// (see `py_isspace`), not the Unicode property.
#[pyfunction]
#[pyo3(signature = (key, character, is_printable))]
fn normalize_key(key: &str, character: Option<&str>, is_printable: bool) -> String {
    if let Some(rest) = key.strip_prefix("ctrl+") {
        return format!("C-{rest}");
    }
    if let Some(rest) = key.strip_prefix("shift+") {
        return format!("S-{rest}");
    }
    if let Some(rest) = key.strip_prefix("alt+") {
        return format!("M-{rest}");
    }

    // `character is not None and len(character) == 1 and is_printable
    //  and not character.isspace()`
    if let Some(ch) = character {
        let mut chars = ch.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            if is_printable && !py_isspace(c) {
                return c.to_string();
            }
        }
    }

    // Textual's special key names remapped to the keymap's names;
    // everything else passes through as-is.
    let mapped = match key {
        "up" => Some("up"),
        "down" => Some("down"),
        "left" => Some("left"),
        "right" => Some("right"),
        "home" => Some("home"),
        "end" => Some("end"),
        "pageup" => Some("page_up"),
        "pagedown" => Some("page_down"),
        "page_up" => Some("page_up"),
        "page_down" => Some("page_down"),
        "backspace" => Some("backspace"),
        "delete" => Some("delete"),
        "tab" => Some("tab"),
        "enter" => Some("enter"),
        "return" => Some("enter"),
        "escape" => Some("escape"),
        "plus" => Some("+"),
        "minus" => Some("-"),
        "equals" => Some("="),
        "question_mark" => Some("?"),
        "slash" => Some("/"),
        _ => None,
    };
    match mapped {
        Some(m) => m.to_string(),
        None => key.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Playback queue engine (mirrors ytm_player/services/queue.py)
// ---------------------------------------------------------------------------

/// Repeat-mode codes shared with the Python `QueueManager` facade:
/// 0 = off, 1 = all, 2 = one — the `RepeatMode` member order.
const REPEAT_OFF: u8 = 0;
const REPEAT_ALL: u8 = 1;
const REPEAT_ONE: u8 = 2;

/// Native playback-order engine backing `QueueManager`.
///
/// Owns the queue's ordering state — the track list, one entry id per
/// occurrence, the shuffle order, the current position, the repeat/shuffle
/// flags and the removed-current bookkeeping — and mirrors the pure-Python
/// `_QueueCore` fallback operation for operation, including the exact order
/// in which random draws happen (the parity tests rely on both).
///
/// Randomness always goes through the Python `random` module (the default
/// `rng`) rather than an in-crate generator, so `random.seed(...)` and
/// monkeypatched `random.shuffle` / `random.randint` / `random.randrange`
/// behave exactly like the fallback.
///
/// `QueueManager` holds its `threading.Lock` around every call; nothing here
/// blocks. Tracks are stored as owned references to the caller's dicts and
/// handed back unchanged — the core never inspects or copies them (except
/// reading `video_id` for radio dedup, mirroring the fallback).
#[pyclass]
struct QueueCore {
    /// Track objects in insertion order.
    tracks: Vec<Py<PyAny>>,
    /// One internal id per queue occurrence, parallel to `tracks`.
    entry_ids: Vec<i64>,
    next_entry_id: i64,
    generation: i64,
    /// A removed audible occurrence can finish outside the queue; keep its
    /// successor by occurrence, never pretend that successor is playing.
    removed_current: bool,
    removed_next_id: Option<i64>,
    current_index: isize,
    repeat: u8,
    shuffle: bool,
    /// When shuffling, maps playback positions to indices in `tracks`.
    shuffle_order: Vec<usize>,
    shuffle_position: isize,
    /// Random source: the Python `random` module. Held as the module object
    /// so attribute patches (`random.shuffle = ...`) apply here too.
    rng: Py<PyAny>,
}

impl QueueCore {
    /// Current index into `tracks`, resolving the shuffle indirection.
    /// (The pre-port `_real_index` helper — does NOT consult
    /// `removed_current`; callers that must guard it.)
    fn current_real_index(&self) -> isize {
        if self.shuffle
            && self.shuffle_position >= 0
            && (self.shuffle_position as usize) < self.shuffle_order.len()
        {
            self.shuffle_order[self.shuffle_position as usize] as isize
        } else {
            self.current_index
        }
    }

    /// Mint `count` fresh entry ids.
    fn new_entry_ids(&mut self, count: usize) -> Vec<i64> {
        let start = self.next_entry_id + 1;
        self.next_entry_id += count as i64;
        (start..start + count as i64).collect()
    }

    /// Visible position of the removed occurrence's successor, or the end.
    ///
    /// Mirrors `_removed_next_position`, including its unguarded
    /// `list.index` semantics: a shuffle order that no longer contains the
    /// successor's real index raises `ValueError` exactly like Python.
    fn removed_next_position(&self) -> PyResult<usize> {
        if let Some(id) = self.removed_next_id {
            if let Some(real) = self.entry_ids.iter().position(|&e| e == id) {
                if self.shuffle {
                    return match self.shuffle_order.iter().position(|&r| r == real) {
                        Some(pos) => Ok(pos),
                        None => Err(PyValueError::new_err(format!("{real} is not in list"))),
                    };
                }
                return Ok(real);
            }
        }
        Ok(self.tracks.len())
    }

    /// Rebuild the shuffle order, optionally keeping the current track first.
    fn rebuild_shuffle(&mut self, py: Python, keep_current: bool) -> PyResult<()> {
        let mut indices: Vec<usize> = (0..self.tracks.len()).collect();
        let current_real = self.current_real_index();

        if keep_current
            && !self.removed_current
            && current_real >= 0
            && (current_real as usize) < self.tracks.len()
        {
            let current_real = current_real as usize;
            if let Some(pos) = indices.iter().position(|&i| i == current_real) {
                indices.remove(pos);
            }
            self.rng_shuffle(py, &mut indices)?;
            self.shuffle_order = Vec::with_capacity(self.tracks.len());
            self.shuffle_order.push(current_real);
            self.shuffle_order.extend_from_slice(&indices);
            self.shuffle_position = 0;
        } else {
            self.rng_shuffle(py, &mut indices)?;
            self.shuffle_order = indices;
            self.shuffle_position = -1;
        }
        Ok(())
    }

    /// `random.shuffle` over a Rust vec, routed through the Python rng so
    /// seeded sequences and monkeypatches match the fallback exactly.
    fn rng_shuffle(&self, py: Python, values: &mut Vec<usize>) -> PyResult<()> {
        let list = PyList::new(py, values.iter().copied())?;
        self.rng.bind(py).call_method1("shuffle", (list.clone(),))?;
        values.clear();
        for item in list.iter() {
            values.push(item.extract::<usize>()?);
        }
        Ok(())
    }

    /// `random.randint(a, b)` — both ends inclusive, like Python.
    fn rng_randint(&self, py: Python, a: i64, b: i64) -> PyResult<i64> {
        self.rng.bind(py).call_method1("randint", (a, b))?.extract()
    }

    /// `random.randrange(n)` — the one-argument form.
    fn rng_randrange(&self, py: Python, n: i64) -> PyResult<i64> {
        self.rng.bind(py).call_method1("randrange", (n,))?.extract()
    }

    /// `list.insert` semantics for a `Vec`: out-of-range positions append.
    fn vec_insert<T>(v: &mut Vec<T>, pos: usize, item: T) {
        let pos = pos.min(v.len());
        v.insert(pos, item);
    }

    /// The track at real index `real`, or `None` when out of range.
    fn track_at(&self, py: Python, real: isize) -> Option<Py<PyAny>> {
        if real >= 0 && (real as usize) < self.tracks.len() {
            Some(self.tracks[real as usize].clone_ref(py))
        } else {
            None
        }
    }

    /// Add a track without acquiring anything (mirrors `_add_unlocked`).
    fn add_unlocked(
        &mut self,
        py: Python,
        track: Py<PyAny>,
        position: Option<isize>,
    ) -> PyResult<()> {
        let entry_id = self.new_entry_ids(1)[0];
        if self.removed_current && self.removed_next_id.is_none() {
            self.removed_next_id = Some(entry_id);
        }
        let new_idx: usize;
        let append = match position {
            None => true,
            Some(pos) => pos >= self.tracks.len() as isize,
        };
        if append {
            self.tracks.push(track);
            self.entry_ids.push(entry_id);
            new_idx = self.tracks.len() - 1;
        } else {
            let pos = position.unwrap().max(0) as usize;
            self.tracks.insert(pos, track);
            self.entry_ids.insert(pos, entry_id);
            new_idx = pos;
            // Adjust the current index if we inserted before it.
            if !self.shuffle && pos as isize <= self.current_index {
                self.current_index += 1;
            }
        }

        if self.shuffle {
            // Insert the new track at a random future position in shuffle order.
            let insert_at: isize = if self.shuffle_position >= 0 {
                self.shuffle_position + 1
            } else {
                0
            };
            // Shift existing shuffle indices that are >= new_idx.
            for i in self.shuffle_order.iter_mut() {
                if *i >= new_idx {
                    *i += 1;
                }
            }
            let len_order = self.shuffle_order.len();
            let future_pos: usize = if (insert_at as usize) < len_order {
                self.rng_randint(py, insert_at as i64 + 1, len_order as i64)? as usize
            } else {
                len_order
            };
            Self::vec_insert(&mut self.shuffle_order, future_pos, new_idx);
        }
        Ok(())
    }

    /// Insert at the removed audible row's boundary (mirrors
    /// `_add_next_removed`). Callers guarantee at least one track.
    fn add_next_removed(&mut self, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        let position = self.removed_next_position()?;
        let ids = self.new_entry_ids(tracks.len());
        let count = tracks.len();
        if self.shuffle {
            let start = self.tracks.len();
            self.tracks.extend(tracks);
            self.entry_ids.extend(ids.iter().copied());
            for (offset, new_idx) in (start..start + count).enumerate() {
                Self::vec_insert(&mut self.shuffle_order, position + offset, new_idx);
            }
        } else {
            for (offset, t) in tracks.into_iter().enumerate() {
                Self::vec_insert(&mut self.tracks, position + offset, t);
            }
            for (offset, id) in ids.iter().copied().enumerate() {
                Self::vec_insert(&mut self.entry_ids, position + offset, id);
            }
        }
        self.removed_next_id = Some(ids[0]);
        Ok(())
    }

    /// Append tracks + update the shuffle order (mirrors
    /// `_add_multiple_unlocked`).
    fn add_multiple_unlocked(&mut self, py: Python, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        let was_empty = self.tracks.is_empty();
        let start_idx = self.tracks.len();
        let count = tracks.len();
        self.tracks.extend(tracks);
        let ids = self.new_entry_ids(count);
        self.entry_ids.extend(ids);
        if count > 0 && self.removed_current && self.removed_next_id.is_none() {
            self.removed_next_id = Some(self.entry_ids[start_idx]);
        }
        let mut new_indices: Vec<usize> = (start_idx..start_idx + count).collect();

        if self.shuffle {
            if was_empty {
                // Fresh queue after clear() — full shuffle build instead of
                // piecemeal insertion into an empty list.
                self.rebuild_shuffle(py, false)?;
            } else {
                // Insert new indices at random future positions.
                let insert_after: isize = if self.shuffle_position >= 0 {
                    self.shuffle_position + 1
                } else {
                    0
                };
                self.rng_shuffle(py, &mut new_indices)?;
                for new_idx in new_indices {
                    // len() is re-read per iteration, like the fallback.
                    let pos =
                        self.rng_randint(py, insert_after as i64, self.shuffle_order.len() as i64)?
                            as usize;
                    Self::vec_insert(&mut self.shuffle_order, pos, new_idx);
                }
            }
        }
        Ok(())
    }

    /// Remove the entry at visible/playback `index` (mirrors
    /// `_remove_unlocked`).
    fn remove_unlocked(&mut self, index: isize) {
        if !(index >= 0 && index < self.tracks.len() as isize) {
            return;
        }
        let index = index as usize;
        let order_len = if self.shuffle {
            self.shuffle_order.len()
        } else {
            self.tracks.len()
        };
        // `order[index]` — the shuffle order when shuffling, identity
        // otherwise. In bounds by the outer check because the shuffle order
        // is a permutation of the track indices.
        let real_idx = if self.shuffle {
            self.shuffle_order[index]
        } else {
            index
        };
        let current_pos = if self.shuffle {
            self.shuffle_position
        } else {
            self.current_index
        };
        let is_current = (!self.removed_current && index as isize == current_pos)
            || (self.removed_current && self.removed_next_id == Some(self.entry_ids[real_idx]));
        if is_current {
            self.removed_current = true;
            self.removed_next_id = if index + 1 < order_len {
                let next_real = if self.shuffle {
                    self.shuffle_order[index + 1]
                } else {
                    index + 1
                };
                Some(self.entry_ids[next_real])
            } else {
                None
            };
        }

        if self.shuffle {
            if index >= self.shuffle_order.len() {
                return;
            }
            let real_idx = self.shuffle_order[index];
            self.shuffle_order.remove(index);
            // Shift indices that pointed beyond the removed track.
            for i in self.shuffle_order.iter_mut() {
                if *i > real_idx {
                    *i -= 1;
                }
            }
            self.tracks.remove(real_idx);
            self.entry_ids.remove(real_idx);
            if (index as isize) < self.shuffle_position {
                self.shuffle_position -= 1;
            } else if index as isize == self.shuffle_position {
                // Current track removed; clamp position (mirrors the
                // non-shuffle branch: current() lands on the next track).
                if self.shuffle_position >= self.shuffle_order.len() as isize {
                    self.shuffle_position = self.shuffle_order.len() as isize - 1;
                }
            }
        } else {
            self.tracks.remove(index);
            self.entry_ids.remove(index);
            if (index as isize) < self.current_index {
                self.current_index -= 1;
            } else if index as isize == self.current_index {
                // Current track removed; clamp index.
                if self.current_index >= self.tracks.len() as isize {
                    self.current_index = self.tracks.len() as isize - 1;
                }
            }
        }
    }
}

#[pymethods]
impl QueueCore {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        Ok(Self {
            tracks: Vec::new(),
            entry_ids: Vec::new(),
            next_entry_id: 0,
            generation: 0,
            removed_current: false,
            removed_next_id: None,
            current_index: -1,
            repeat: REPEAT_OFF,
            shuffle: false,
            shuffle_order: Vec::new(),
            shuffle_position: -1,
            rng: py.import("random")?.into_any().unbind(),
        })
    }

    // -- Queue manipulation ---------------------------------------------

    /// Add a track to the queue. `None` position = append to end.
    #[pyo3(signature = (track, position=None))]
    fn add(&mut self, py: Python, track: Py<PyAny>, position: Option<isize>) -> PyResult<()> {
        self.add_unlocked(py, track, position)
    }

    /// Insert a track immediately after the currently playing track.
    fn add_next(&mut self, py: Python, track: Py<PyAny>) -> PyResult<()> {
        if self.removed_current {
            self.add_next_removed(vec![track])?;
            return Ok(());
        }
        if self.shuffle {
            // Insert into tracks and put it next in shuffle order.
            self.tracks.push(track);
            let ids = self.new_entry_ids(1);
            self.entry_ids.extend(ids);
            let new_idx = self.tracks.len() - 1;
            let insert_pos = (self.shuffle_position + 1).max(0) as usize;
            Self::vec_insert(&mut self.shuffle_order, insert_pos, new_idx);
        } else {
            let insert_pos = if self.current_index >= 0 {
                self.current_index + 1
            } else {
                0
            };
            self.add_unlocked(py, track, Some(insert_pos))?;
        }
        Ok(())
    }

    /// Insert *tracks* immediately after the current track, preserving order.
    fn add_next_multiple(&mut self, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        if tracks.is_empty() {
            return Ok(());
        }
        if self.removed_current {
            self.add_next_removed(tracks)?;
            return Ok(());
        }
        let entry_ids = self.new_entry_ids(tracks.len());
        if self.shuffle {
            let start_idx = self.tracks.len();
            let count = tracks.len();
            self.tracks.extend(tracks);
            self.entry_ids.extend(entry_ids);
            let insert_pos = (self.shuffle_position + 1).max(0) as usize;
            for (offset, new_idx) in (start_idx..start_idx + count).enumerate() {
                Self::vec_insert(&mut self.shuffle_order, insert_pos + offset, new_idx);
            }
        } else {
            let insert_pos = if self.current_index >= 0 {
                self.current_index + 1
            } else {
                0
            };
            let insert_pos = (insert_pos.clamp(0, self.tracks.len() as isize)) as usize;
            for (offset, t) in tracks.into_iter().enumerate() {
                Self::vec_insert(&mut self.tracks, insert_pos + offset, t);
            }
            for (offset, id) in entry_ids.into_iter().enumerate() {
                Self::vec_insert(&mut self.entry_ids, insert_pos + offset, id);
            }
        }
        Ok(())
    }

    /// Append multiple tracks to the end of the queue.
    fn add_multiple(&mut self, py: Python, tracks: Vec<Py<PyAny>>) -> PyResult<()> {
        if tracks.is_empty() {
            return Ok(());
        }
        self.add_multiple_unlocked(py, tracks)
    }

    /// Append the tracks whose video_id is not already queued (radio dedup).
    /// Returns the filtered list; the caller decides what to do with it.
    fn filter_not_in_queue(
        &self,
        py: Python,
        candidates: Vec<Py<PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let existing: Vec<Py<PyAny>> = self
            .tracks
            .iter()
            .map(|t| {
                t.bind(py)
                    .call_method1("get", ("video_id",))
                    .map(|v| v.unbind())
            })
            .collect::<PyResult<Vec<Py<PyAny>>>>()?;
        let set = PySet::new(py, existing)?;
        let mut out: Vec<Py<PyAny>> = Vec::new();
        for t in candidates {
            let vid = t.bind(py).call_method1("get", ("video_id",))?;
            if !set.contains(vid)? {
                out.push(t);
            }
        }
        Ok(out)
    }

    /// Remove the track at the given index (in visible/playback order).
    fn remove(&mut self, index: isize) {
        self.remove_unlocked(index);
    }

    /// Remove the one occurrence *entry_id* names. False when no entry
    /// carries that id any more — never falls back to another occurrence.
    fn remove_entry(&mut self, entry_id: i64) -> bool {
        let real_idx = match self.entry_ids.iter().position(|&e| e == entry_id) {
            Some(r) => r,
            None => return false,
        };
        let index = if self.shuffle {
            match self.shuffle_order.iter().position(|&r| r == real_idx) {
                Some(p) => p as isize,
                None => return false,
            }
        } else {
            real_idx as isize
        };
        self.remove_unlocked(index);
        true
    }

    /// Remove all tracks from the queue (bumps the generation).
    fn clear(&mut self) {
        self.generation += 1;
        self.removed_current = false;
        self.removed_next_id = None;
        self.tracks.clear();
        self.entry_ids.clear();
        self.current_index = -1;
        self.shuffle_order.clear();
        self.shuffle_position = -1;
    }

    /// Move a track from one position to another in the visible order.
    #[pyo3(name = "move")]
    fn move_(&mut self, from_idx: isize, to_idx: isize) {
        if from_idx == to_idx {
            return;
        }
        if !(from_idx >= 0 && from_idx < self.tracks.len() as isize) {
            return;
        }
        let to_idx = to_idx.max(0).min(self.tracks.len() as isize - 1);

        if self.shuffle {
            // Move within shuffle order.
            if from_idx >= self.shuffle_order.len() as isize
                || to_idx >= self.shuffle_order.len() as isize
            {
                return;
            }
            let item = self.shuffle_order.remove(from_idx as usize);
            Self::vec_insert(&mut self.shuffle_order, to_idx as usize, item);
            if from_idx == self.shuffle_position {
                self.shuffle_position = to_idx;
            } else if from_idx < self.shuffle_position && to_idx >= self.shuffle_position {
                self.shuffle_position -= 1;
            } else if to_idx <= self.shuffle_position && self.shuffle_position < from_idx {
                self.shuffle_position += 1;
            }
        } else {
            let track = self.tracks.remove(from_idx as usize);
            self.tracks.insert(to_idx as usize, track);
            let id = self.entry_ids.remove(from_idx as usize);
            self.entry_ids.insert(to_idx as usize, id);
            if from_idx == self.current_index {
                self.current_index = to_idx;
            } else if from_idx < self.current_index && to_idx >= self.current_index {
                self.current_index -= 1;
            } else if to_idx <= self.current_index && self.current_index < from_idx {
                self.current_index += 1;
            }
        }
    }

    // -- Playback navigation ---------------------------------------------

    /// Return the currently selected track, or None.
    fn current(&self, py: Python) -> Option<Py<PyAny>> {
        if self.removed_current {
            return None;
        }
        self.track_at(py, self.current_real_index())
    }

    /// Advance to the next track and return it (repeat/shuffle aware).
    fn next_track(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }

        if self.removed_current {
            let mut pos = self.removed_next_position()? as isize;
            if pos == self.tracks.len() as isize {
                if self.repeat != REPEAT_ALL {
                    return Ok(None);
                }
                pos = 0;
            }
            self.removed_current = false;
            self.removed_next_id = None;
            if self.shuffle {
                self.shuffle_position = pos;
            } else {
                self.current_index = pos;
            }
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.repeat == REPEAT_ONE {
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.shuffle {
            let next_pos = self.shuffle_position + 1;
            if next_pos >= self.shuffle_order.len() as isize {
                if self.repeat == REPEAT_ALL {
                    self.rebuild_shuffle(py, false)?;
                    self.shuffle_position = 0;
                } else {
                    return Ok(None);
                }
            } else {
                self.shuffle_position = next_pos;
            }
        } else {
            let next_idx = self.current_index + 1;
            if next_idx >= self.tracks.len() as isize {
                if self.repeat == REPEAT_ALL {
                    self.current_index = 0;
                } else {
                    return Ok(None);
                }
            } else {
                self.current_index = next_idx;
            }
        }

        Ok(self.track_at(py, self.current_real_index()))
    }

    /// Go back to the previous track and return it (repeat/shuffle aware).
    fn previous_track(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }

        if self.removed_current {
            let mut pos = self.removed_next_position()? as isize - 1;
            if pos < 0 {
                if self.repeat != REPEAT_ALL {
                    return Ok(None);
                }
                pos = self.tracks.len() as isize - 1;
            }
            self.removed_current = false;
            self.removed_next_id = None;
            if self.shuffle {
                self.shuffle_position = pos;
            } else {
                self.current_index = pos;
            }
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.repeat == REPEAT_ONE {
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.shuffle {
            let prev_pos = self.shuffle_position - 1;
            if prev_pos < 0 {
                if self.repeat == REPEAT_ALL {
                    self.shuffle_position = self.shuffle_order.len() as isize - 1;
                } else {
                    return Ok(None);
                }
            } else {
                self.shuffle_position = prev_pos;
            }
        } else {
            let prev_idx = self.current_index - 1;
            if prev_idx < 0 {
                if self.repeat == REPEAT_ALL {
                    self.current_index = self.tracks.len() as isize - 1;
                } else {
                    return Ok(None);
                }
            } else {
                self.current_index = prev_idx;
            }
        }

        Ok(self.track_at(py, self.current_real_index()))
    }

    /// Return the next track WITHOUT advancing the position (prefetch).
    fn peek_next(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }

        if self.removed_current {
            let mut pos = self.removed_next_position()? as isize;
            if pos == self.tracks.len() as isize {
                if self.repeat != REPEAT_ALL {
                    return Ok(None);
                }
                pos = 0;
            }
            let real = if self.shuffle {
                self.shuffle_order[pos as usize] as isize
            } else {
                pos
            };
            return Ok(self.track_at(py, real));
        }

        if self.repeat == REPEAT_ONE {
            return Ok(self.track_at(py, self.current_real_index()));
        }

        if self.shuffle {
            let next_pos = self.shuffle_position + 1;
            if next_pos >= self.shuffle_order.len() as isize {
                // End of the shuffle order. Even under RepeatMode.ALL we
                // can't predict the next reshuffle — nothing to prefetch.
                return Ok(None);
            }
            let real_idx = self.shuffle_order[next_pos as usize] as isize;
            return Ok(self.track_at(py, real_idx));
        }

        let next_idx = self.current_index + 1;
        if next_idx >= self.tracks.len() as isize {
            if self.repeat == REPEAT_ALL {
                return Ok(self.track_at(py, 0));
            }
            return Ok(None);
        }
        Ok(self.track_at(py, next_idx))
    }

    // -- Repeat / Shuffle -------------------------------------------------

    /// Set the repeat mode from a `RepeatMode` code (0 off, 1 all, 2 one).
    fn set_repeat(&mut self, code: u8) {
        self.repeat = code;
    }

    /// Cycle through repeat modes: OFF -> ALL -> ONE -> OFF. Returns the
    /// new code.
    fn cycle_repeat(&mut self) -> u8 {
        self.repeat = match self.repeat {
            REPEAT_OFF => REPEAT_ALL,
            REPEAT_ALL => REPEAT_ONE,
            _ => REPEAT_OFF,
        };
        self.repeat
    }

    /// Toggle shuffle mode on or off.
    fn toggle_shuffle(&mut self, py: Python) -> PyResult<()> {
        // Resolve the playing occurrence under the mode being left. Once
        // the flag flips, current_real_index() reads the other mode's
        // pointer, and current_index is stale after next/previous under
        // shuffle.
        let real = self.current_real_index();
        self.shuffle = !self.shuffle;
        if self.shuffle {
            self.rebuild_shuffle(py, true)?;
        } else {
            // Exiting shuffle: the real index becomes the current position.
            self.current_index = if real >= 0 && (real as usize) < self.tracks.len() {
                real
            } else {
                -1
            };
            self.shuffle_order.clear();
            self.shuffle_position = -1;
        }
        Ok(())
    }

    // -- Random / Radio ---------------------------------------------------

    /// Pick and jump to a random track from the queue.
    fn play_random(&mut self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        if self.tracks.is_empty() {
            return Ok(None);
        }
        self.removed_current = false;
        self.removed_next_id = None;
        let idx = self.rng_randrange(py, self.tracks.len() as i64)? as usize;
        if self.shuffle {
            // Find or set position in shuffle order.
            self.shuffle_position = match self.shuffle_order.iter().position(|&r| r == idx) {
                Some(pos) => pos as isize,
                None => 0,
            };
        } else {
            self.current_index = idx as isize;
        }
        Ok(self.track_at(py, self.current_real_index()))
    }

    // -- Jumps -------------------------------------------------------------

    /// Resolve and select an occurrence atomically, including under shuffle.
    fn jump_to_entry(&mut self, py: Python, entry_id: i64) -> PyResult<Option<Py<PyAny>>> {
        let real = match self.entry_ids.iter().position(|&e| e == entry_id) {
            Some(r) => r,
            None => return Ok(None),
        };
        self.removed_current = false;
        self.removed_next_id = None;
        self.current_index = real as isize;
        if self.shuffle {
            // Mirrors the fallback's unguarded `list.index`: a real index
            // missing from the shuffle order raises ValueError.
            let pos = match self.shuffle_order.iter().position(|&r| r == real) {
                Some(p) => p,
                None => return Err(PyValueError::new_err(format!("{real} is not in list"))),
            };
            self.shuffle_position = pos as isize;
        }
        Ok(self.track_at(py, real as isize))
    }

    /// Jump to a specific index in the visible order and return the track.
    fn jump_to(&mut self, py: Python, index: isize) -> Option<Py<PyAny>> {
        if !(index >= 0 && index < self.tracks.len() as isize) {
            return None;
        }
        self.removed_current = false;
        self.removed_next_id = None;
        if self.shuffle {
            if !(index >= 0 && index < self.shuffle_order.len() as isize) {
                return None;
            }
            self.shuffle_position = index;
            // Keep current_index in sync as a safety fallback.
            self.current_index = self.shuffle_order[index as usize] as isize;
        } else {
            self.current_index = index;
        }
        self.track_at(py, self.current_real_index())
    }

    /// Jump to a track by its index in the internal track list.
    fn jump_to_real(&mut self, py: Python, real_index: isize) -> Option<Py<PyAny>> {
        if !(real_index >= 0 && real_index < self.tracks.len() as isize) {
            return None;
        }
        self.removed_current = false;
        self.removed_next_id = None;
        if self.shuffle {
            match self
                .shuffle_order
                .iter()
                .position(|&r| r == real_index as usize)
            {
                Some(pos) => self.shuffle_position = pos as isize,
                None => {
                    // Track not in shuffle order — insert it right after the
                    // current position so playback can continue from here.
                    let insert_pos = (self.shuffle_position + 1).max(0) as usize;
                    Self::vec_insert(&mut self.shuffle_order, insert_pos, real_index as usize);
                    self.shuffle_position = insert_pos as isize;
                }
            }
        } else {
            self.current_index = real_index;
        }
        self.track_at(py, self.current_real_index())
    }

    // -- State accessors ---------------------------------------------------

    /// Lifetime of this queue; ordinary navigation/insertion keeps it.
    fn generation(&self) -> i64 {
        self.generation
    }

    /// Index of the currently playing track in the visible order.
    fn current_index(&self) -> isize {
        if self.removed_current {
            return -1;
        }
        if self.shuffle {
            return self.shuffle_position;
        }
        self.current_index
    }

    fn length(&self) -> usize {
        self.tracks.len()
    }

    fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    fn repeat_code(&self) -> u8 {
        self.repeat
    }

    fn shuffle_enabled(&self) -> bool {
        self.shuffle
    }

    /// Current index into `tracks` regardless of shuffle mode.
    fn real_index(&self) -> isize {
        if self.removed_current {
            return -1;
        }
        self.current_real_index()
    }

    /// Number of tracks remaining after the current position.
    fn remaining_tracks(&self) -> PyResult<i64> {
        if self.removed_current {
            return Ok(self.tracks.len() as i64 - self.removed_next_position()? as i64);
        }
        if self.shuffle {
            return Ok(self.shuffle_order.len() as i64 - self.shuffle_position as i64 - 1);
        }
        Ok(self.tracks.len() as i64 - self.current_index as i64 - 1)
    }

    /// Tracks in the current playback order (a Python tuple, like the
    /// fallback's property).
    fn tracks(&self, py: Python) -> PyResult<Py<PyAny>> {
        let items: Vec<Py<PyAny>> = if self.shuffle {
            self.shuffle_order
                .iter()
                .map(|&i| self.tracks[i].clone_ref(py))
                .collect()
        } else {
            self.tracks.iter().map(|t| t.clone_ref(py)).collect()
        };
        Ok(PyTuple::new(py, items)?.into_any().unbind())
    }

    /// `(entry_id, track)` pairs in the current playback order.
    fn entries(&self, py: Python) -> PyResult<Py<PyAny>> {
        let mut items: Vec<Py<PyAny>> = Vec::with_capacity(self.entry_ids.len());
        if self.shuffle {
            for &i in &self.shuffle_order {
                let pair = (self.entry_ids[i], self.tracks[i].clone_ref(py))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                items.push(pair);
            }
        } else {
            for i in 0..self.entry_ids.len() {
                let pair = (self.entry_ids[i], self.tracks[i].clone_ref(py))
                    .into_pyobject(py)?
                    .into_any()
                    .unbind();
                items.push(pair);
            }
        }
        Ok(PyTuple::new(py, items)?.into_any().unbind())
    }

    // -- Test seam ---------------------------------------------------------

    /// Overwrite the shuffle order/position wholesale. No validation —
    /// exists so tests can pin a deterministic playback order on both the
    /// native and the pure-Python core.
    fn force_shuffle_state(&mut self, order: Vec<usize>, position: isize) {
        self.shuffle = true;
        self.shuffle_order = order;
        self.shuffle_position = position;
    }
}

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
fn filter_sort_tracks(
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

// ---------------------------------------------------------------------------
// Session-state validation engine (mirrors ytm_player/app/_session.py)
// ---------------------------------------------------------------------------

/// `int(max(0, min(100, v)))` for a value already known to be a non-bool
/// int. Ints beyond i64 still clamp by sign, exactly like CPython's
/// comparison-based `min`/`max` (min(100, huge) == 100, max(0, -huge) == 0).
fn clamp_volume_int(v: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(i) = v.extract::<i64>() {
        return Ok(i.clamp(0, 100));
    }
    let negative = v
        .str()?
        .to_str()
        .map(|s| s.starts_with('-'))
        .unwrap_or(false);
    Ok(if negative { 0 } else { 100 })
}

/// The same clamp for a float, transcribing Python's comparison order so
/// NaN maps to 100 exactly like `max(0, min(100, nan))` does
/// (`min(a, b)` keeps `a` unless `b < a`; NaN comparisons are always
/// false, so `min(100, nan)` is 100 and `min(100, -inf)` is -inf).
fn clamp_volume_float(f: f64) -> i64 {
    let m = if f < 100.0 { f } else { 100.0 };
    let m = if m > 0.0 { m } else { 0.0 };
    m.trunc() as i64
}

/// Apply the Python volume gate: a non-bool int/float clamps into
/// 0..=100; anything else fails (the caller substitutes the raw default).
/// Exact type checks suffice because `json.loads` never produces
/// subclasses.
fn try_clamp_volume(v: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    // bool is checked first: it passes PyLong_Check but fails the gate.
    if v.is_instance_of::<PyBool>() {
        return Ok(None);
    }
    if v.downcast::<PyFloat>().is_ok() {
        let f: f64 = v.extract()?;
        return Ok(Some(clamp_volume_float(f)));
    }
    if v.is_instance_of::<PyInt>() {
        return clamp_volume_int(v).map(Some);
    }
    Ok(None)
}

/// `isinstance(x, int)` as an i64 — bools count (they are ints in
/// Python, so a stored `true` queue_index validates as index 1, matching
/// the fallback's `isinstance(saved_index, int)` check).
fn python_int_value(x: &Bound<'_, PyAny>) -> Option<i64> {
    if let Ok(b) = x.downcast::<PyBool>() {
        return Some(i64::from(b.is_true()));
    }
    x.extract::<i64>().ok()
}

/// Validate and coerce a parsed session.json object in one pass.
///
/// Transcribes the pure validation half of
/// `SessionMixin._restore_session_state` from `app/_session.py`: every
/// isinstance gate, clamp and fallback applied to the raw parsed state
/// before any app object is touched. The mixin reads the returned dict
/// and only *applies* the values — all decisions are made here,
/// identically on either backend.
///
/// Returned keys:
///
/// - `volume` — clamped int, or the raw *default_volume* when the stored
///   value's type is invalid (the fallback returns the raw default
///   unclamped, so an absent key with a non-numeric default round-trips).
/// - `repeat` — "off" / "all" / "one"; anything else degrades to "off".
/// - `shuffle` — truthiness of the stored `shuffle`.
/// - `queue_tracks` — `normalize_tracks()` of the stored list when it is
///   a truthy list, else `[]`.
/// - `queue_index` — the stored index when it is an int (bool included)
///   inside the normalized range, else None.
/// - `queue_context_id` — non-empty str or None.
/// - `sidebar_per_page` — the stored dict itself (no copy) when truthy,
///   else None.
/// - `first_run_hint_shown` / `mpris_hint_shown` — truthiness.
/// - `has_transliteration` + `transliteration_enabled` — key presence
///   and the RAW stored value (the fallback assigns it unconverted).
/// - `resume` — the stored resume dict (the original object, not a copy:
///   the mixin copies for `_loaded_resume` and reads position /
///   playlist_id from it) when it is a dict with a truthy `video_id`,
///   else None.
#[pyfunction]
fn coerce_session_state<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    default_volume: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // -- volume ------------------------------------------------------------
    // `state.get("volume", default)` then the isinstance gate: a non-bool
    // int/float clamps into 0..=100, anything else (including an absent
    // key with a non-numeric default) yields the raw default.
    let volume_clamped: Option<i64> = match state.get_item("volume")? {
        Some(ref v) => try_clamp_volume(v)?,
        None => try_clamp_volume(default_volume)?,
    };

    // -- repeat mode ---------------------------------------------------------
    let repeat: String = match state.get_item("repeat")? {
        Some(ref v) => match v.downcast::<PyString>() {
            Ok(s) => {
                let t = s.to_str()?;
                if t == "off" || t == "all" || t == "one" {
                    t.to_string()
                } else {
                    "off".to_string()
                }
            }
            Err(_) => "off".to_string(),
        },
        None => "off".to_string(),
    };

    // -- queue -----------------------------------------------------------------
    let (queue_tracks, queue_index): (Bound<'py, PyList>, Option<i64>) =
        match state.get_item("queue_tracks")? {
            Some(ref v) if v.is_truthy()? && v.is_instance_of::<PyList>() => {
                let normalized = normalize_tracks(py, v.downcast::<PyList>().unwrap())?;
                let len = normalized.len() as i64;
                // `state.get("queue_index", 0)` gated by
                // `isinstance(saved_index, int) and 0 <= saved_index < len`.
                let idx = match state.get_item("queue_index")? {
                    Some(ref x) => python_int_value(x).filter(|i| *i >= 0 && *i < len),
                    None => {
                        if len > 0 {
                            Some(0)
                        } else {
                            None
                        }
                    }
                };
                (normalized, idx)
            }
            _ => (PyList::empty(py), None),
        };

    // -- shuffle -----------------------------------------------------------------
    let shuffle: bool = match state.get_item("shuffle")? {
        Some(v) => v.is_truthy()?,
        None => false,
    };

    // -- collection context ----------------------------------------------------
    let queue_context_id: Option<String> = match state.get_item("queue_context_id")? {
        Some(ref v) => match v.downcast::<PyString>() {
            Ok(s) => {
                let t = s.to_str()?;
                if t.is_empty() {
                    None
                } else {
                    Some(t.to_string())
                }
            }
            Err(_) => None,
        },
        None => None,
    };

    // -- sidebar state ----------------------------------------------------------
    let sidebar_per_page: Option<Bound<'py, PyAny>> = match state.get_item("sidebar_per_page")? {
        Some(ref v) if v.is_truthy()? && v.is_instance_of::<PyDict>() => Some(v.clone()),
        _ => None,
    };

    // -- hint flags ------------------------------------------------------------
    let first_run_hint_shown: bool = match state.get_item("first_run_hint_shown")? {
        Some(v) => v.is_truthy()?,
        None => false,
    };
    let mpris_hint_shown: bool = match state.get_item("mpris_hint_shown")? {
        Some(v) => v.is_truthy()?,
        None => false,
    };

    // -- transliteration toggle (raw value, assigned unconverted) ---------------
    let has_transliteration = state.contains("transliteration_enabled")?;
    let transliteration_enabled: Option<Bound<'py, PyAny>> =
        state.get_item("transliteration_enabled")?;

    // -- resume staging -----------------------------------------------------------
    // The original dict (not a copy): the mixin copies it for
    // `_loaded_resume` and reads position / playlist_id from it directly.
    let resume: Option<Bound<'py, PyAny>> = match state.get_item("resume")? {
        Some(ref r) if r.is_instance_of::<PyDict>() => {
            let has_video_id = match r.downcast::<PyDict>().unwrap().get_item("video_id")? {
                Some(v) => v.is_truthy()?,
                None => false,
            };
            if has_video_id {
                Some(r.clone())
            } else {
                None
            }
        }
        _ => None,
    };

    // -- assemble ------------------------------------------------------------
    let out = PyDict::new(py);
    match volume_clamped {
        Some(i) => {
            out.set_item("volume", i)?;
        }
        None => {
            out.set_item("volume", default_volume.clone())?;
        }
    }
    out.set_item("repeat", repeat.as_str())?;
    out.set_item("shuffle", shuffle)?;
    out.set_item("queue_tracks", queue_tracks)?;
    match queue_index {
        Some(i) => {
            out.set_item("queue_index", i)?;
        }
        None => {
            out.set_item("queue_index", pyo3::types::PyNone::get(py))?;
        }
    }
    match &queue_context_id {
        Some(s) => {
            out.set_item("queue_context_id", s.as_str())?;
        }
        None => {
            out.set_item("queue_context_id", pyo3::types::PyNone::get(py))?;
        }
    }
    match &sidebar_per_page {
        Some(d) => {
            out.set_item("sidebar_per_page", d.clone())?;
        }
        None => {
            out.set_item("sidebar_per_page", pyo3::types::PyNone::get(py))?;
        }
    }
    out.set_item("first_run_hint_shown", first_run_hint_shown)?;
    out.set_item("mpris_hint_shown", mpris_hint_shown)?;
    out.set_item("has_transliteration", has_transliteration)?;
    match transliteration_enabled {
        Some(v) => {
            out.set_item("transliteration_enabled", v)?;
        }
        None => {
            out.set_item("transliteration_enabled", pyo3::types::PyNone::get(py))?;
        }
    }
    match resume {
        Some(r) => {
            out.set_item("resume", r)?;
        }
        None => {
            out.set_item("resume", pyo3::types::PyNone::get(py))?;
        }
    }
    Ok(out)
}

/// Linear scan for the first track whose `video_id` equals *video_id* —
/// the resume-point lookup from `SessionMixin._restore_session_state`,
/// run over the restored (possibly shuffled) queue. Comparison uses
/// Python `==` semantics (the stored resume `video_id` can be any truthy
/// JSON value), and non-dict entries are skipped where the Python
/// fallback would raise — queue entries are always normalized dicts, and
/// the engine staying total keeps one malformed entry from taking the
/// restore down with it.
#[pyfunction]
fn find_track_index(
    tracks: &Bound<'_, PyAny>,
    video_id: &Bound<'_, PyAny>,
) -> PyResult<Option<usize>> {
    for (i, item) in tracks.try_iter()?.enumerate() {
        let item = item?;
        if let Ok(dict) = item.downcast::<PyDict>() {
            if let Some(vid) = dict.get_item("video_id")? {
                if vid.eq(video_id)? {
                    return Ok(Some(i));
                }
            }
        }
    }
    Ok(None)
}

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
fn shallow_copy_dict<'py>(
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
fn stamp_track<'py>(
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
fn occurrence_key<'py>(py: Python<'py>, track: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
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
fn with_accepted_play<'py>(
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
fn with_accepted_local_play<'py>(
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
fn park_pending_play<'py>(
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
fn merge_pending_account_plays<'py>(
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
fn enrich_track<'py>(
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
fn compose_all_rows<'py>(
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

// ---------------------------------------------------------------------------
// Navigation stack engine (mirrors ytm_player/app/_navigation.py)
// ---------------------------------------------------------------------------

/// One back/forward stack entry: `(page_name, kwargs)`. The kwargs dict is
/// an opaque payload carried by reference — a popped entry's dict flows to
/// the mounted page exactly like the fallback's, while the snapshots built
/// along the way are fresh dicts.
type NavEntry = (String, Py<PyDict>);

/// Keep the last *max* entries, mirroring Python's `stack[-max:]`. A *max*
/// of 0 keeps everything, exactly like `x[-0:]` (== `x[0:]`).
fn cap_stack<T>(stack: &mut Vec<T>, max: usize) {
    if max > 0 && stack.len() > max {
        stack.drain(..stack.len() - max);
    }
}

/// `dict(current_kwargs)` + `.update(cache.get(current_page, {}))` — the
/// snapshot of the page being left that gets pushed onto a stack: a fresh
/// dict of the constructor kwargs with the page's cached nav state applied
/// over them. The `update` call goes through the method (not a Rust-side
/// item copy) so any mapping value behaves exactly like the fallback's
/// `.update` — cached nav states are always dicts.
fn leaving_snapshot<'py>(
    py: Python<'py>,
    current_kwargs: &Bound<'py, PyDict>,
    cache: &Bound<'py, PyDict>,
    current_page: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let snap = shallow_copy_dict(py, current_kwargs)?;
    if let Some(state) = cache.get_item(current_page)? {
        snap.call_method1("update", (state,))?;
    }
    Ok(snap)
}

/// Assemble the engine outcome dict. The new state is included on every
/// action — a no-op still returns the (cache-refreshed, possibly popped)
/// stacks, because the cache refresh and any stack pop happen before the
/// request is validated and the caller must always apply them.
#[allow(clippy::too_many_arguments)]
fn nav_outcome<'py>(
    py: Python<'py>,
    action: &str,
    reason: Option<&str>,
    page: &str,
    kwargs: Py<PyDict>,
    nav: Vec<NavEntry>,
    forward: Vec<NavEntry>,
    cache: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("action", action)?;
    match reason {
        Some(reason) => {
            out.set_item("reason", reason)?;
        }
        None => {
            out.set_item("reason", pyo3::types::PyNone::get(py))?;
        }
    }
    out.set_item("page", page)?;
    out.set_item("kwargs", kwargs)?;
    out.set_item("nav_stack", PyList::new(py, nav)?)?;
    out.set_item("forward_stack", PyList::new(py, forward)?)?;
    out.set_item("page_state_cache", cache)?;
    Ok(out)
}

/// Resolve one navigation request against the back/forward stacks — the
/// pure stack machine behind `NavigationMixin.navigate_to` from
/// `app/_navigation.py`, transcribed verbatim from its bookkeeping.
///
/// Mirrors the pure-Python reference (`resolve_navigation` in
/// `_navigation.py`) case for case:
///
/// - The leaving page's fresh *current_page_state* (Python `None` = no
///   mounted page / no `get_nav_state`: the cache is left alone) is
///   folded into a copy of *page_state_cache* first — truthy state
///   replaces the cached entry, empty state drops it — because the
///   stack snapshots below read that cache back.
/// - `back` pops the nav stack (pushing the current page's snapshot onto
///   the forward stack); `forward` does the reverse. `back` with an empty
///   stack goes to `library`; `forward` with an empty stack is a silent
///   no-op.
/// - Clicking the page you're already on (empty *kwargs*) pops back
///   instead; with nothing to pop it's a silent no-op.
/// - Any non-back/forward navigation pushes the current page's snapshot
///   onto the nav stack and invalidates the forward history — you can't
///   redo a future you didn't take.
/// - A switch with empty kwargs restores the target's cached state.
/// - A page not in *page_names* is a no-op (after the back/forward
///   handling, so a dead name popped from a stack is caught here too).
///
/// Inputs are never mutated; the returned `nav_stack` / `forward_stack` /
/// `page_state_cache` are fresh containers whose payload dicts are the
/// input objects themselves (snapshots and restored kwargs are new dicts,
/// exactly like the fallback). The caller passes `page_names` /
/// `max_stack` so the Python module's constants stay the single source
/// of truth.
///
/// Returns `{action, reason, page, kwargs, nav_stack, forward_stack,
/// page_state_cache}`: `action` is `"switch"` or `"noop"`, `reason` is
/// `"unknown_page"` / `"no_forward_history"` / `"same_page_no_history"`
/// for a no-op and None for a switch.
#[pyfunction(signature = (
    current_page,
    current_kwargs,
    nav_stack,
    forward_stack,
    page_state_cache,
    current_page_state,
    page_name,
    kwargs,
    page_names,
    max_stack,
))]
#[allow(clippy::too_many_arguments)]
fn resolve_navigation<'py>(
    py: Python<'py>,
    current_page: &str,
    current_kwargs: &Bound<'py, PyDict>,
    nav_stack: Vec<NavEntry>,
    forward_stack: Vec<NavEntry>,
    page_state_cache: &Bound<'py, PyDict>,
    current_page_state: &Bound<'py, PyAny>,
    page_name: &str,
    kwargs: &Bound<'py, PyDict>,
    page_names: Vec<String>,
    max_stack: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let mut page_name = page_name.to_string();
    let mut kwargs = kwargs.clone().unbind();

    // -- refresh the leaving page's cached nav state ------------------------
    // A truthy fresh state replaces the cached entry; an empty one means
    // the user left whatever the cache remembers — drop it rather than
    // restore it later. None (no mounted page / no get_nav_state) leaves
    // the cache alone.
    let cache = shallow_copy_dict(py, page_state_cache)?;
    if !current_page.is_empty() && !current_page_state.is_none() {
        if current_page_state.is_truthy()? {
            cache.set_item(current_page, current_page_state)?;
        } else if cache.contains(current_page)? {
            cache.del_item(current_page)?;
        }
    }

    let mut nav = nav_stack;
    let mut forward = forward_stack;

    // -- handle "back" / "forward" navigation via stacks --------------------
    let is_back = page_name == "back";
    let is_forward = page_name == "forward";
    if is_back {
        if let Some((prev_page, prev_kwargs)) = nav.pop() {
            // Push the page we're leaving onto the forward stack so the
            // user can come back to it via forward.
            if !current_page.is_empty() {
                let snap = leaving_snapshot(py, current_kwargs, &cache, current_page)?;
                forward.push((current_page.to_string(), snap.unbind()));
                cap_stack(&mut forward, max_stack);
            }
            page_name = prev_page;
            kwargs = prev_kwargs;
        } else {
            page_name = "library".to_string();
        }
    } else if is_forward {
        if let Some((next_page, next_kwargs)) = forward.pop() {
            // Push the page we're leaving back onto the nav stack so the
            // user can return to it via back.
            if !current_page.is_empty() {
                let snap = leaving_snapshot(py, current_kwargs, &cache, current_page)?;
                nav.push((current_page.to_string(), snap.unbind()));
                cap_stack(&mut nav, max_stack);
            }
            page_name = next_page;
            kwargs = next_kwargs;
        } else {
            // No forward history — silently no-op.
            return nav_outcome(
                py,
                "noop",
                Some("no_forward_history"),
                &page_name,
                kwargs,
                nav,
                forward,
                cache,
            );
        }
    }

    if !page_names.contains(&page_name) {
        return nav_outcome(
            py,
            "noop",
            Some("unknown_page"),
            &page_name,
            kwargs,
            nav,
            forward,
            cache,
        );
    }

    // -- same-page click pops back instead ----------------------------------
    let mut same_page_back = false;
    if page_name == current_page && !kwargs.bind(py).is_truthy()? {
        if let Some((prev_page, prev_kwargs)) = nav.pop() {
            page_name = prev_page;
            kwargs = prev_kwargs;
            same_page_back = true;
        } else {
            return nav_outcome(
                py,
                "noop",
                Some("same_page_no_history"),
                &page_name,
                kwargs,
                nav,
                forward,
                cache,
            );
        }
    }

    // -- push current page onto the nav stack before switching --------------
    // Skip for back / forward / same-page-back — those already managed the
    // stacks above (or popped, in the same-page case).
    if !is_back
        && !is_forward
        && !same_page_back
        && !current_page.is_empty()
        && current_page != page_name
    {
        let snap = leaving_snapshot(py, current_kwargs, &cache, current_page)?;
        nav.push((current_page.to_string(), snap.unbind()));
        cap_stack(&mut nav, max_stack);
        // Browser semantics: any non-back/forward navigation invalidates
        // the forward history (you can't redo a future you didn't take).
        forward.clear();
    }

    // -- restore cached state when no explicit kwargs were given ------------
    if !kwargs.bind(py).is_truthy()? {
        if let Some(state) = cache.get_item(page_name.as_str())? {
            // `dict(cache[page_name])` — a fresh shallow copy; the cached
            // entry itself stays in the cache.
            let restored = PyDict::new(py);
            restored.call_method1("update", (state,))?;
            kwargs = restored.unbind();
        }
    }

    nav_outcome(py, "switch", None, &page_name, kwargs, nav, forward, cache)
}

/// The dead-playlist predicate of the purge: a Library entry carries
/// `playlist_id`, a playlist context page (opened from search/browse)
/// carries `context_type` + `context_id`. Set membership goes through the
/// Python set object (hash/eq semantics), so `None` and missing values
/// match nothing, exactly like the fallback's `in dead_ids`.
fn nav_entry_is_dead(
    py: Python<'_>,
    dead_ids: &Bound<'_, PySet>,
    kw: &Bound<'_, PyDict>,
) -> PyResult<bool> {
    let _ = py;
    if let Some(playlist_id) = kw.get_item("playlist_id")? {
        if dead_ids.contains(playlist_id)? {
            return Ok(true);
        }
    }
    if let Some(context_type) = kw.get_item("context_type")? {
        if context_type.eq("playlist")? {
            if let Some(context_id) = kw.get_item("context_id")? {
                if dead_ids.contains(context_id)? {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Drop every navigation reference to the playlists in *dead_ids* — the
/// pure half of `NavigationMixin._purge_playlist_nav_state` from
/// `app/_navigation.py`. The `library` cache entry and the stack entries
/// naming a deleted playlist (via `playlist_id` — Library pages — or
/// `context_type` + `context_id` — playlist context pages) plus the
/// bare-Library fallback id must all go, or a bare footer-nav to Library
/// would reopen the deleted playlist.
///
/// Mirrors the pure-Python reference (`purge_dead_playlist_refs` in
/// `_navigation.py`) case for case; inputs are never mutated. Returns
/// `{active_library_playlist_id, nav_stack, forward_stack,
/// page_state_cache}`.
#[pyfunction]
fn purge_dead_playlist_refs<'py>(
    py: Python<'py>,
    dead_ids: &Bound<'py, PySet>,
    active_library_playlist_id: &Bound<'py, PyAny>,
    nav_stack: Vec<NavEntry>,
    forward_stack: Vec<NavEntry>,
    page_state_cache: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    // LibraryPage.on_mount falls back to this id when mounted bare. The
    // membership test goes through the Python set (hash/eq semantics), so
    // `None in dead_ids` matches nothing and a non-dead id is carried
    // through unchanged — the original object, like the fallback.
    let active: Option<Py<PyAny>> = if dead_ids.contains(active_library_playlist_id)? {
        None
    } else {
        Some(active_library_playlist_id.clone().unbind())
    };

    let cache = shallow_copy_dict(py, page_state_cache)?;
    if let Some(lib_state) = cache.get_item("library")? {
        if lib_state.is_truthy()? {
            // `lib_state.get("playlist_id") in dead_ids` — the get() call
            // goes through the attribute, like the fallback.
            let playlist_id = lib_state.call_method1("get", ("playlist_id",))?;
            if dead_ids.contains(playlist_id)? {
                cache.del_item("library")?;
            }
        }
    }

    let mut nav: Vec<NavEntry> = Vec::with_capacity(nav_stack.len());
    for (name, kw) in &nav_stack {
        if !nav_entry_is_dead(py, dead_ids, kw.bind(py))? {
            nav.push((name.clone(), kw.clone_ref(py)));
        }
    }
    let mut forward: Vec<NavEntry> = Vec::with_capacity(forward_stack.len());
    for (name, kw) in &forward_stack {
        if !nav_entry_is_dead(py, dead_ids, kw.bind(py))? {
            forward.push((name.clone(), kw.clone_ref(py)));
        }
    }

    let out = PyDict::new(py);
    match active {
        Some(id) => {
            out.set_item("active_library_playlist_id", id)?;
        }
        None => {
            out.set_item("active_library_playlist_id", pyo3::types::PyNone::get(py))?;
        }
    }
    out.set_item("nav_stack", PyList::new(py, nav)?)?;
    out.set_item("forward_stack", PyList::new(py, forward)?)?;
    out.set_item("page_state_cache", cache)?;
    Ok(out)
}

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
fn is_event_shelf(py: Python, shelf: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(shelf_title(py, shelf)?.contains(": "))
}

/// The sort key behind the country-chart pill order: a tuple
/// `(rank, title)` where *rank* is the index of the first `_CHART_PRIORITY`
/// prefix the title starts with (99 for none of them) and *title* is the
/// `str()`-ified shelf title that breaks ties.
///
/// Mirrors `_chart_sort_key` from `ui/pages/browse.py`.
#[pyfunction]
fn chart_sort_key(py: Python, shelf: &Bound<'_, PyAny>) -> PyResult<(i64, String)> {
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
fn split_events_and_charts<'py>(
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
fn is_playlist_row(item: &Bound<'_, PyAny>) -> PyResult<bool> {
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
fn playlist_rows<'py>(
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
fn field_get<'py>(
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
fn first_truthy_get<'py>(
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

/// Python's `value in ("a", "b")`: left-to-right `==` comparisons against
/// the strings, with the result taken by truthiness — so str subclasses
/// and `__eq__`-overriding objects classify exactly like the reference.
/// An exact `str` takes the contents fast path (`str.__eq__` is C-level
/// equality, no dispatch).
fn eq_any_str<'py>(value: &Bound<'py, PyAny>, options: &[&str]) -> PyResult<bool> {
    if value.is_exact_instance_of::<PyString>() {
        let s = value.downcast::<PyString>().unwrap();
        if let Ok(contents) = s.to_str() {
            return Ok(options.contains(&contents));
        }
    }
    for option in options {
        if value.eq(*option)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// `str(obj)` for label interpolation — the f-string conversion the
/// reference applies to title/year/subscribers/author/count values.
fn py_str(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.str()?.to_string_lossy().into_owned())
}

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
fn format_search_result(py: Python, item: &Bound<'_, PyAny>) -> PyResult<String> {
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
