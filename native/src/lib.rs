use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use regex::Regex;
use std::collections::{HashMap, HashSet};
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

        let raw_dur = get_optional_any(track, "duration_seconds")?
            .or_else(|| get_optional_any(track, "duration").unwrap_or(None))
            .or_else(|| get_optional_any(track, "length").unwrap_or(None));
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
