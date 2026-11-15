//! Native cores for `src/ytm_player/utils/formatting.py` — duration/
//! count/size/ago formatting, ID extraction, track normalization and
//! shelf-title cleaning.

use crate::ui::pages::help::py_capitalize;
use crate::utils::pycompat::get_optional_any;
use crate::utils::pycompat::get_optional_string;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use regex::Regex;
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

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

// ---------------------------------------------------------------------------
// Pure formatting functions
// ---------------------------------------------------------------------------

/// Format seconds as H:MM:SS or M:SS. Negative values clamp to 0.
#[pyfunction]
pub(crate) fn format_duration(seconds: i64) -> String {
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
pub(crate) fn truncate(text: &str, max_len: isize) -> String {
    truncate_str(text, max_len)
}

/// The `truncate` body, shared with `format_search_result` so the label
/// composer truncates through exactly one definition.
pub(crate) fn truncate_str(text: &str, max_len: isize) -> String {
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
pub(crate) fn format_count(n: i64) -> String {
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
pub(crate) fn format_size(bytes_val: i64) -> String {
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
pub(crate) fn get_video_id(_py: Python, track: &Bound<'_, PyDict>) -> String {
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
pub(crate) fn is_valid_video_id(vid: &str) -> bool {
    VALID_VIDEO_ID_RE.is_match(vid)
}

/// Strip the `VL` prefix from a playlist ID if present.
#[pyfunction]
pub(crate) fn strip_vl_prefix(playlist_id: &str) -> String {
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
pub(crate) fn clean_shelf_title(
    _py: Python,
    raw: &str,
    region_names: &Bound<'_, PyList>,
) -> String {
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

/// Build the metadata subtitle line for a playlist header.
/// Format: `Owner · Privacy · Year · N tracks`
/// *year* is a PyAny that may be None, int, or str. The privacy runs
/// through `py_capitalize` — a faithful `str.capitalize()`, the same
/// call the pure-Python fallback makes (the ported `capitalize_first`
/// helper this used to call uppercased the first character instead of
/// titlecasing it, diverging from the fallback on `ß`-shaped inputs
/// the ASCII-only production values never hit).
#[pyfunction]
pub(crate) fn build_playlist_subtitle(
    owner: &str,
    privacy: &str,
    year: &Bound<'_, PyAny>,
    track_count: i64,
) -> PyResult<String> {
    let mut parts: Vec<String> = vec![owner.to_string()];
    if !privacy.is_empty() {
        parts.push(py_capitalize(privacy));
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
pub(crate) fn extract_duration(_py: Python, track: &Bound<'_, PyDict>) -> i64 {
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
pub(crate) fn extract_artist(py: Python, track: &Bound<'_, PyDict>) -> String {
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
pub(crate) fn normalize_tracks<'py>(
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
pub(crate) fn sanitize_title_for_lyric_lookup(title: &str, artist: Option<&str>) -> String {
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
pub(crate) fn format_ago(timestamp_seconds: f64) -> String {
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
