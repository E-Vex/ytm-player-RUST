use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyIterator, PyList, PySet, PyString, PyTuple,
};
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
    // Count/sequence key-input state machine (app/_keys.py on_key
    // equivalent): the count-prefix accumulation and sequence-buffer
    // resolution behind every keystroke — the digit gate, the
    // exact/pending/no-match classification with the 1000 count cap, and
    // the buffer resets.
    module.add_function(pyo3::wrap_pyfunction!(feed_key, module)?)?;

    // Action dispatch engine (app/_keys.py _handle_action equivalent):
    // the dispatch-table walk (which branch handles an action, and the
    // page the navigations target), the pane decision behind the routed
    // navigation actions, the CURRENT_CONTEXT album-target extraction,
    // and the TOGGLE_SHUFFLE lock gate.
    module.add_function(pyo3::wrap_pyfunction!(dispatch_route, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(route_navigation_action, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(current_context_target, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(shuffle_lock_blocked, module)?)?;

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

    // Sidebar pane-routing state machine (app/_sidebar.py equivalent):
    // the three-pane focus router — per-page playlist-sidebar visibility
    // (read/toggle/ensure), the visible-pane list, pane resolution from
    // the focused widget's ancestor chain, the focus gate, and the
    // left/right/cycle movement decisions behind Ctrl+w h / l / w.
    module.add_function(pyo3::wrap_pyfunction!(sidebar_visible_for_page, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(toggle_sidebar_state, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ensure_sidebar_visible, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(visible_panes, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(pane_of_focused, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(pane_focus_target, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(next_pane_left, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(next_pane_right, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(next_pane_cycle, module)?)?;

    // Playlist panel view engine (ui/sidebars/playlist_sidebar.py
    // equivalent): the label/count view model behind the sidebar's
    // LibraryPanel — the row label and footer count label, the `/`-filter
    // pass, the playlist-id matching behind the optimistic sidebar
    // mutations (removal / count updates / auto-select / the Liked Music
    // exclusion), the count arithmetic, and the info-bar title read.
    module.add_function(pyo3::wrap_pyfunction!(format_panel_item, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(panel_count_label, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(filter_panel_items, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(matches_removal, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(matches_count_update, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(bump_playlist_count, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(selection_title, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(is_liked_music_playlist, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(matches_auto_select, module)?)?;

    // Context-page view-model assembly (ui/pages/context.py equivalent):
    // the header/discography view model behind the album, artist and
    // playlist detail pages — the album and playlist header compositions
    // (title, subtitle, unavailable count, the playlist's owned flag and
    // remaining-tail fetch gate), the artist discography (top songs,
    // the albums+singles concatenation, subscribers and the
    // similar-artists bar), and the see-all section selection behind
    // the discography expansion.
    module.add_function(pyo3::wrap_pyfunction!(album_header, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(playlist_header, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(artist_discography, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(expandable_sections, module)?)?;

    // Play-history row engine (ytm_player/services/history.py
    // equivalent): the skip gate and insert-row assembly behind
    // HistoryManager.log_play, and the stored-vs-new listened-seconds
    // decision (the int coercion, no-write gate and delta) behind
    // update_play_listened_seconds.
    module.add_function(pyo3::wrap_pyfunction!(play_log_row, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(listened_delta, module)?)?;

    // Command palette catalog (app/_commands.py equivalent): the
    // (name, help) catalog behind the command provider — the data both
    // the discovery list and the query search dispatch over, with the
    // per-entry callback resolution staying on the Python side next to
    // the Textual hit machinery.
    module.add_function(pyo3::wrap_pyfunction!(command_catalog, module)?)?;

    // Stats aggregation payload (ytm_player/services/history.py
    // equivalent): the five-key listening-stats dict assembled behind
    // HistoryManager.get_stats from the three scalar aggregates and
    // the two top lists.
    module.add_function(pyo3::wrap_pyfunction!(stats_payload, module)?)?;

    // History-read row materialisation (ytm_player/services/history.py
    // and cli.py equivalents): the [dict(row) for row in rows]
    // comprehension behind every history read — the getters'
    // fetchall() lists and the CLI's _query_history alike — converted
    // through CPython's own dict constructor.
    module.add_function(pyo3::wrap_pyfunction!(row_dicts, module)?)?;

    // CLI stats aggregation payload (cli.py equivalent): the four-key
    // listening-stats dict assembled behind the standalone `ytm stats`
    // command, with the top-tracks list rebuilt from its row tuples'
    // first four fields.
    module.add_function(pyo3::wrap_pyfunction!(cli_stats_payload, module)?)?;

    // IPC dispatch engine (app/_ipc.py _handle_ipc_command equivalent):
    // the dispatch-table walk behind the CLI-shells command router —
    // which branch of the match handles each command string, and the
    // rating the like/dislike/unlike branches rate with.
    module.add_function(pyo3::wrap_pyfunction!(ipc_dispatch_route, module)?)?;

    // IPC seek-offset parser (app/_ipc.py _ipc_seek equivalent): the
    // classification of one offset value into the seek it names —
    // relative (+/-seconds), clock (mm:ss / hh:mm:ss) and absolute
    // seconds, with the missing/invalid verdicts the response carries.
    module.add_function(pyo3::wrap_pyfunction!(ipc_seek_offset, module)?)?;

    // CLI JSON serialisation (cli.py _json_output equivalent): the
    // json.dumps(data, indent=..., default=str) text behind every
    // JSON-emitting CLI command — the ensure_ascii escaping, the
    // float/int reprs, the indent-aware layout, the key coercion and
    // the circular-reference guard.
    module.add_function(pyo3::wrap_pyfunction!(json_dumps, module)?)?;

    // Doctor report assembly (utils/doctor.py gather_diagnostics
    // equivalent): the multi-section `ytm doctor` text — the banners,
    // the aligned identity and paths labels, the crash-file section,
    // the final join and the redaction pass — from the probes'
    // results.
    module.add_function(pyo3::wrap_pyfunction!(doctor_report, module)?)?;

    // Doctor probe engines (utils/doctor.py equivalents): the
    // mpv-warning tail behind the Recent mpv warnings/errors probe, and
    // the crash-staleness verdict behind the Most recent crash file
    // section — the metadata-header scan, the PEP 440 comparison of the
    // recorded version against the installed one, and the note assembly.
    module.add_function(pyo3::wrap_pyfunction!(recent_mpv_lines, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(crash_staleness_note, module)?)?;
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

/// The start index of Python's `seq[-n:]` tail slice: a positive *n*
/// keeps the last n, zero keeps everything (`seq[-0:]` is `seq[0:]`),
/// and a negative n drops the leading -n entries (empty when that
/// exceeds the length).
fn py_tail_start(len: usize, n: i64) -> usize {
    if n > 0 {
        len.saturating_sub(n as usize)
    } else if n == 0 {
        0
    } else {
        // unsigned_abs rather than -n: negating i64::MIN overflows.
        (n.unsigned_abs() as i128).min(len as i128) as usize
    }
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
    // Python's `lines[-n:]` (see py_tail_start).
    kept[py_tail_start(kept.len(), n)..].concat()
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
// Count/sequence key-input state machine (mirrors app/_keys.py's on_key)
// ---------------------------------------------------------------------------

/// The count-prefix safety cap, `app/_keys.py`'s `_MAX_KEY_COUNT`.
const MAX_KEY_COUNT: i64 = 1000;

/// `str.isdecimal()` acceptance set: General_Category Nd, as ranges
/// generated from CPython 3.12's Unicode data (Unicode 15.0). Every range
/// is a whole number of consecutive decades starting at value 0, so the
/// decimal value of a code point is `(cp - start) % 10` — the property
/// CPython's `int()` relies on for non-ASCII decimal digits
/// (`int('٣') == 3`). Newer CPython releases may add Nd ranges for new
/// scripts (Unicode 16 added Kirat Rai & co.); those digits would fall
/// through to the keymap here until the table is regenerated, a divergence
/// bounded to scripts added after the generating Unicode version.
static ND_RANGES: &[(u32, u32)] = &[
    (0x0030, 0x0039),
    (0x0660, 0x0669),
    (0x06F0, 0x06F9),
    (0x07C0, 0x07C9),
    (0x0966, 0x096F),
    (0x09E6, 0x09EF),
    (0x0A66, 0x0A6F),
    (0x0AE6, 0x0AEF),
    (0x0B66, 0x0B6F),
    (0x0BE6, 0x0BEF),
    (0x0C66, 0x0C6F),
    (0x0CE6, 0x0CEF),
    (0x0D66, 0x0D6F),
    (0x0DE6, 0x0DEF),
    (0x0E50, 0x0E59),
    (0x0ED0, 0x0ED9),
    (0x0F20, 0x0F29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17E0, 0x17E9),
    (0x1810, 0x1819),
    (0x1946, 0x194F),
    (0x19D0, 0x19D9),
    (0x1A80, 0x1A89),
    (0x1A90, 0x1A99),
    (0x1B50, 0x1B59),
    (0x1BB0, 0x1BB9),
    (0x1C40, 0x1C49),
    (0x1C50, 0x1C59),
    (0xA620, 0xA629),
    (0xA8D0, 0xA8D9),
    (0xA900, 0xA909),
    (0xA9D0, 0xA9D9),
    (0xA9F0, 0xA9F9),
    (0xAA50, 0xAA59),
    (0xABF0, 0xABF9),
    (0xFF10, 0xFF19),
    (0x104A0, 0x104A9),
    (0x10D30, 0x10D39),
    (0x11066, 0x1106F),
    (0x110F0, 0x110F9),
    (0x11136, 0x1113F),
    (0x111D0, 0x111D9),
    (0x112F0, 0x112F9),
    (0x11450, 0x11459),
    (0x114D0, 0x114D9),
    (0x11650, 0x11659),
    (0x116C0, 0x116C9),
    (0x11730, 0x11739),
    (0x118E0, 0x118E9),
    (0x11950, 0x11959),
    (0x11C50, 0x11C59),
    (0x11D50, 0x11D59),
    (0x11DA0, 0x11DA9),
    (0x11F50, 0x11F59),
    (0x16A60, 0x16A69),
    (0x16AC0, 0x16AC9),
    (0x16B50, 0x16B59),
    (0x1D7CE, 0x1D7FF),
    (0x1E140, 0x1E149),
    (0x1E2F0, 0x1E2F9),
    (0x1E4F0, 0x1E4F9),
    (0x1E950, 0x1E959),
    (0x1FBF0, 0x1FBF9),
];

/// The `str.isdigit()`-but-not-`str.isdecimal()` supplement
/// (Numeric_Type=Digit: superscripts, subscripts, circled digits, …), as
/// ranges generated from CPython 3.12's Unicode data. These satisfy the
/// reference's `key.isdigit()` count-accumulation gate but are rejected by
/// `int()` — the latent `'²'`+`'j'` ValueError the engine reproduces
/// faithfully.
static DIGIT_ONLY_RANGES: &[(u32, u32)] = &[
    (0x00B2, 0x00B3),
    (0x00B9, 0x00B9),
    (0x1369, 0x1371),
    (0x19DA, 0x19DA),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247C),
    (0x2488, 0x2490),
    (0x24EA, 0x24EA),
    (0x24F5, 0x24FD),
    (0x24FF, 0x24FF),
    (0x2776, 0x277E),
    (0x2780, 0x2788),
    (0x278A, 0x2792),
    (0x10A40, 0x10A43),
    (0x10E60, 0x10E68),
    (0x11052, 0x1105A),
    (0x1F100, 0x1F10A),
];

/// Index of the range containing *cp*, in a sorted non-overlapping table.
fn range_index(cp: u32, ranges: &[(u32, u32)]) -> Option<usize> {
    // First range whose high bound is >= cp.
    let idx = ranges.partition_point(|&(_lo, hi)| hi < cp);
    match ranges.get(idx) {
        Some(&(lo, _hi)) if lo <= cp => Some(idx),
        _ => None,
    }
}

/// The decimal value of an Nd digit (0-9), like CPython's `int()` reads it.
fn nd_digit_value(c: char) -> Option<u32> {
    let cp = c as u32;
    let idx = range_index(cp, ND_RANGES)?;
    Some((cp - ND_RANGES[idx].0) % 10)
}

/// CPython's `str.isdigit()` for a single character: Nd (decimal) or the
/// Numeric_Type=Digit supplement. Rust's `char::is_numeric()` would
/// disagree (it also accepts Numeric_Type=Numeric, like Roman numerals),
/// so the acceptance set is spelled out as tables instead.
fn py_is_digit_char(c: char) -> bool {
    let cp = c as u32;
    nd_digit_value(c).is_some() || range_index(cp, DIGIT_ONLY_RANGES).is_some()
}

/// CPython's `str.isdigit()` on a whole string: non-empty with every
/// character a digit.
fn py_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(py_is_digit_char)
}

/// `int(str)` for a count buffer: decimal (Nd) digits of any script, the
/// only characters the digit gate can have accumulated. A digit-classified
/// non-decimal character (Numeric_Type=Digit, e.g. `'²'`) raises
/// `ValueError` with CPython's message — the reference's latent crash,
/// transcribed faithfully. Saturates far above the 1000 cap where the
/// reference would build an exact bigint.
fn parse_count(s: &str) -> PyResult<i64> {
    if s.is_empty() {
        return Err(PyValueError::new_err(
            "invalid literal for int() with base 10: ''",
        ));
    }
    let mut acc: i128 = 0;
    for c in s.chars() {
        match nd_digit_value(c) {
            Some(d) => acc = acc.saturating_mul(10).saturating_add(i128::from(d)),
            None => {
                return Err(PyValueError::new_err(format!(
                    "invalid literal for int() with base 10: '{s}'"
                )))
            }
        }
    }
    Ok(acc.min(i64::MAX as i128) as i64)
}

/// Assemble one engine outcome map: fresh in every field, like every
/// engine outcome — the buffers are new objects the caller assigns over
/// the app's attributes, inputs are never mutated. Event suppression
/// follows the outcome kind, exactly like the reference's branches: the
/// count digit and every matched sequence prevent the default, only a
/// resolved match (exact or pending) also stops propagation, and no
/// match lets the event flow on untouched.
fn key_outcome<'py>(
    py: Python<'py>,
    outcome: &str,
    keys: &Bound<'py, PyList>,
    count: &str,
    dispatch_count: Option<i64>,
    action: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (prevent_default, stop) = match outcome {
        "count_digit" => (true, false),
        "exact" | "pending" => (true, true),
        _ => (false, false),
    };
    let out = PyDict::new(py);
    out.set_item("outcome", outcome)?;
    out.set_item("prevent_default", prevent_default)?;
    out.set_item("stop", stop)?;
    out.set_item("key_buffer", keys.clone())?;
    out.set_item("count_buffer", count)?;
    match dispatch_count {
        Some(n) => out.set_item("count", n)?,
        None => out.set_item("count", py.None())?,
    }
    match action {
        Some(a) => out.set_item("action", a)?,
        None => out.set_item("action", py.None())?,
    }
    Ok(out)
}

/// The count/sequence buffer state machine behind `on_key` from
/// `app/_keys.py` — the dispatch glue between key-name normalisation and
/// `_handle_action`.
///
/// Mirrors the pure-Python reference (`_feed_key`): a digit key with an
/// empty sequence buffer accumulates the count prefix (prevent_default
/// only); anything else is appended to the sequence buffer and classified
/// by `keymap.match` — exact dispatches with the parsed count (default 1,
/// capped at 1000) and clears both buffers, pending keeps the grown
/// sequence waiting for more keys, and no match clears both buffers while
/// letting the event flow on. The classification result is read from the
/// `MatchResult` member's string `.value`; anything the reference's `==`
/// would reject — a mock without a string value — reads as no_match, never
/// an error, exactly like the fallback's else branch.
///
/// The outcome map carries `{"outcome": "count_digit" | "exact" |
/// "pending" | "no_match", "prevent_default": bool, "stop": bool,
/// "key_buffer": list, "count_buffer": str, "count": int | None,
/// "action": Action | None}`. Buffer items pass through as the objects
/// they are — the engine never re-types them — and `key_buffer`'s
/// emptiness is the reference's `not self._key_buffer`.
#[pyfunction]
fn feed_key<'py>(
    py: Python<'py>,
    key: &str,
    key_buffer: &Bound<'py, PyList>,
    count_buffer: &str,
    keymap: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // Digit handling: accumulate the count prefix if no keys are buffered.
    if py_isdigit(key) && key_buffer.is_empty() {
        let fresh = PyList::new(py, key_buffer.iter().collect::<Vec<_>>())?;
        return key_outcome(
            py,
            "count_digit",
            &fresh,
            &format!("{count_buffer}{key}"),
            None,
            None,
        );
    }

    // Append to the sequence buffer and classify through the keymap —
    // `result, action = self.keymap.match(tuple(self._key_buffer))`.
    let key_py = PyString::new(py, key);
    let grown: Vec<Bound<'py, PyAny>> = key_buffer
        .iter()
        .chain(std::iter::once(key_py.clone().into_any()))
        .collect();
    let sequence = PyTuple::new(py, grown.iter())?;
    let matched = keymap.call_method1("match", (sequence,))?;
    // Unpack exactly like Python's two-name assignment: any 2-iterable
    // (a tuple in production, a list or custom sequence in a stub) is
    // accepted, with CPython's unpack ValueErrors otherwise.
    let mut items = matched.try_iter()?;
    let result = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 2, got 0)",
            ))
        }
    };
    let action = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 2, got 1)",
            ))
        }
    };
    if items.next().is_some() {
        return Err(PyValueError::new_err(
            "too many values to unpack (expected 2)",
        ));
    }
    let action = if action.is_none() { None } else { Some(action) };
    let kind = match result
        .getattr("value")
        .ok()
        .and_then(|value| value.extract::<String>().ok())
        .as_deref()
    {
        Some("exact") => "exact",
        Some("pending") => "pending",
        _ => "no_match",
    };

    if kind == "exact" {
        let count = if count_buffer.is_empty() {
            1
        } else {
            parse_count(count_buffer)?
        };
        let count = count.min(MAX_KEY_COUNT);
        return key_outcome(
            py,
            "exact",
            &PyList::empty(py),
            "",
            Some(count),
            action.as_ref(),
        );
    }
    if kind == "pending" {
        let grown_list = PyList::new(py, grown)?;
        return key_outcome(py, "pending", &grown_list, count_buffer, None, None);
    }
    key_outcome(py, "no_match", &PyList::empty(py), "", None, None)
}

// ---------------------------------------------------------------------------
// Action dispatch engine (mirrors app/_keys.py's _handle_action routing)
// ---------------------------------------------------------------------------

/// Assemble one dispatch outcome map: fresh in every field, like every
/// engine outcome. *branch* and *page* pass through as the matched
/// row's own objects — the engine never re-types them — so a stub table
/// returns its stub tokens exactly like the fallback.
fn dispatch_outcome<'py>(
    py: Python<'py>,
    branch: &Bound<'py, PyAny>,
    page: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("branch", branch)?;
    out.set_item("page", page)?;
    Ok(out)
}

/// Unpack one dispatch-table row exactly like the fallback's
/// `for member, branch, page in table`: any 3-iterable works (a tuple
/// in production, a generator or `__getitem__` sequence in a stub),
/// with CPython's unpack TypeErrors and ValueErrors otherwise — the
/// not-enough/too-many counts are the items actually yielded, and the
/// "cannot unpack non-iterable" rewrite only applies to a type with no
/// iteration support at all (an `__iter__` that itself raises keeps its
/// own error, like CPython's `tp_iter` check).
fn unpack_dispatch_row<'py>(
    row: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let py = row.py();
    let mut items = match row.try_iter() {
        Ok(items) => items,
        Err(err) => {
            if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                // `type(row).__iter__` present and non-None mirrors
                // tp_iter being filled; anything else is CPython's
                // non-iterable rewrite.
                let type_iter = row
                    .get_type()
                    .getattr("__iter__")
                    .ok()
                    .filter(|attr| !attr.is_none());
                if type_iter.is_none() {
                    let type_name = row
                        .get_type()
                        .name()
                        .map(|name| name.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "cannot unpack non-iterable {type_name} object"
                    )));
                }
            }
            return Err(err);
        }
    };
    let member = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 3, got 0)",
            ))
        }
    };
    let branch = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 3, got 1)",
            ))
        }
    };
    let page = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 3, got 2)",
            ))
        }
    };
    if items.next().is_some() {
        return Err(PyValueError::new_err(
            "too many values to unpack (expected 3)",
        ));
    }
    Ok((member, branch, page))
}

/// The dispatch table behind `_handle_action` from `app/_keys.py` —
/// which branch of the dispatch handles an action, and for the page
/// navigations which page it targets.
///
/// Mirrors the pure-Python reference (`_dispatch_route`): `None`
/// dispatches nothing, each row of *table* is compared in order with
/// Python `==` (subject on the left — exactly like the reference
/// `match`'s lazily evaluated value patterns, first equal match wins,
/// so a plain `"play_pause"` string dispatches like the str-enum
/// member) and anything no row claims is unhandled. *table* is the
/// module's own ordered `(Action, branch, page)` tuple in production;
/// a stub table drives the walk identically. The outcome map carries
/// `{"branch": str, "page": str | None}` — fresh in every field, with
/// the matched row's branch/page objects passed through unchanged —
/// and neither *action* nor *table* is mutated.
#[pyfunction]
#[pyo3(signature = (action, table))]
fn dispatch_route<'py>(
    py: Python<'py>,
    action: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if action.is_none() {
        let branch = PyString::new(py, "none");
        let none = py.None().into_bound(py);
        return dispatch_outcome(py, &branch, &none);
    }
    for row in table.try_iter()? {
        let (member, branch, page) = unpack_dispatch_row(&row?)?;
        if action.eq(&member)? {
            return dispatch_outcome(py, &branch, &page);
        }
    }
    let branch = PyString::new(py, "unhandled");
    let none = py.None().into_bound(py);
    dispatch_outcome(py, &branch, &none)
}

/// The pane decision behind `_route_navigation_action` from
/// `app/_keys.py` — which pane a routed navigation action targets.
///
/// Mirrors the pure-Python reference (`_navigation_action_route`): the
/// Playlists sidebar's widget while that pane holds focus and the
/// action is in *sidebar_actions*, the lyrics pane's widget while it
/// holds focus and the action is in *lyrics_actions*, and the page
/// otherwise. Membership goes through Python's `in` (hash + `==`), so
/// the caller's frozensets — or a stub's sets — decide exactly like
/// the fallback, and a plain string action routes like its member. The
/// reference's `if`/`elif` collapses to two independent gates: the
/// `elif` is only skipped when the first gate holds, which already
/// means the pane is Playlists and the lyrics gate could never fire.
/// Returns `"sidebar" | "lyrics" | "page"`.
#[pyfunction]
#[pyo3(signature = (pane, action, sidebar_actions, lyrics_actions))]
fn route_navigation_action<'py>(
    pane: &str,
    action: &Bound<'py, PyAny>,
    sidebar_actions: &Bound<'py, PyAny>,
    lyrics_actions: &Bound<'py, PyAny>,
) -> PyResult<String> {
    if pane == PANE_PLAYLISTS && sidebar_actions.contains(action)? {
        return Ok("sidebar".to_string());
    }
    if pane == PANE_LYRICS && lyrics_actions.contains(action)? {
        return Ok("lyrics".to_string());
    }
    Ok("page".to_string())
}

/// The album-target extraction behind the CURRENT_CONTEXT dispatch
/// branch of `_handle_action` from `app/_keys.py`.
///
/// Mirrors the pure-Python reference (`_current_context_target`): a
/// falsy *track* (nothing playing — an empty dict counts) can only
/// notify; otherwise the album id is `track.get("album_id")`, falling
/// back to `track.get("album")["id"]` when the former is falsy and the
/// album is a dict (`isinstance`, so a dict subclass passes). A truthy
/// id navigates; anything else notifies. The id passes through as the
/// object it is. Returns `{"outcome": "navigate", "album_id": <id>}` or
/// `{"outcome": "notify", "reason": "no_track" | "no_album"}`.
#[pyfunction]
fn current_context_target<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if !track.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("outcome", "notify")?;
        out.set_item("reason", "no_track")?;
        return Ok(out);
    }
    let mut album_id = track.call_method1("get", ("album_id",))?;
    if !album_id.is_truthy()? {
        let album = track.call_method1("get", ("album",))?;
        if album.is_instance_of::<PyDict>() {
            album_id = album.call_method1("get", ("id",))?;
        }
    }
    if album_id.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("outcome", "navigate")?;
        out.set_item("album_id", &album_id)?;
        return Ok(out);
    }
    let out = PyDict::new(py);
    out.set_item("outcome", "notify")?;
    out.set_item("reason", "no_album")?;
    Ok(out)
}

/// The Shuffle-lock gate behind the TOGGLE_SHUFFLE dispatch branch of
/// `_handle_action` from `app/_keys.py`.
///
/// Mirrors the pure-Python reference (`_shuffle_lock_blocked`): with a
/// falsy context id there is nothing locked (and *prefs* is never
/// consulted — the reference's `ctx and ...` short-circuits the same
/// way); otherwise `prefs.get(context_id)` decides with Python
/// truthiness. *prefs* is read duck-typed, so the ShufflePreferences
/// service, a mock, or a plain dict all behave identically.
#[pyfunction]
fn shuffle_lock_blocked(context_id: &Bound<'_, PyAny>, prefs: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !context_id.is_truthy()? {
        return Ok(false);
    }
    let value = prefs.call_method1("get", (context_id,))?;
    value.is_truthy()
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

// ---------------------------------------------------------------------------
// Sidebar pane-routing state machine (mirrors ytm_player/app/_sidebar.py)
// ---------------------------------------------------------------------------

/// The pane identifiers, in the layout's left-to-right order
/// `[playlists | content | lyrics]` — the Python module's `PANE_*`
/// constants.
const PANE_PLAYLISTS: &str = "playlists";
const PANE_CONTENT: &str = "content";
const PANE_LYRICS: &str = "lyrics";

/// The DOM ids of the three pane containers; a focused widget's ancestor
/// chain is scanned for these, exactly like the fallback's `node_id ==`
/// comparisons.
const ID_PLAYLIST_SIDEBAR: &str = "playlist-sidebar";
const ID_LYRICS_SIDEBAR: &str = "lyrics-sidebar";
const ID_MAIN_CONTENT: &str = "main-content";

/// The page key a sidebar-visibility read/toggle/ensure resolves to: an
/// empty page (no page mounted yet) means Library, exactly like the
/// fallback's `page or "library"`.
fn sidebar_page_key(page: &str) -> &str {
    if page.is_empty() {
        "library"
    } else {
        page
    }
}

/// `per_page.get(key, default)` gated by Python truthiness — the current
/// visibility flag for the read/toggle/ensure decisions. A stored value
/// keeps its own truthiness (a hand-edited session.json may hold `1` or
/// `""`), and a missing entry reads the default's.
fn sidebar_page_visible(
    per_page: &Bound<'_, PyDict>,
    key: &str,
    default: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    match per_page.get_item(key)? {
        Some(value) => value.is_truthy(),
        None => default.is_truthy(),
    }
}

/// The playlist-sidebar visibility stored for a page — the read behind
/// `SidebarMixin._is_playlist_sidebar_visible` (and the toggle/ensure
/// bookkeeping) from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`sidebar_visible_for_page`): an
/// empty *page* (no page mounted yet) reads Library. The stored value —
/// or *default* when the page has no entry — is returned raw, so a truthy
/// non-bool stored value (a hand-edited session.json) flows on with its
/// own truthiness, like the reference's `.get`.
#[pyfunction]
fn sidebar_visible_for_page<'py>(
    per_page: &Bound<'py, PyDict>,
    page: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match per_page.get_item(sidebar_page_key(page))? {
        Some(value) => Ok(value),
        None => Ok(default.clone()),
    }
}

/// Flip the playlist-sidebar visibility stored for a page — the pure half
/// of `SidebarMixin._toggle_playlist_sidebar` from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`toggle_sidebar_state`): an empty
/// *page* means Library; the new state is the negation of the current one
/// (*default* when the page has no entry, Python truthiness deciding) and
/// lands in a fresh copy of *per_page* — inputs are never mutated.
/// Returns `{"visible": bool, "sidebar_per_page": dict}`.
#[pyfunction]
fn toggle_sidebar_state<'py>(
    py: Python<'py>,
    per_page: &Bound<'py, PyDict>,
    page: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let key = sidebar_page_key(page);
    let new_state = !sidebar_page_visible(per_page, key, default)?;
    let fresh = shallow_copy_dict(py, per_page)?;
    fresh.set_item(key, new_state)?;
    let out = PyDict::new(py);
    out.set_item("visible", new_state)?;
    out.set_item("sidebar_per_page", fresh)?;
    Ok(out)
}

/// Make the playlist-sidebar entry for a page visible, if it isn't — the
/// pure half of `SidebarMixin._ensure_playlist_sidebar_visible` from
/// `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`ensure_sidebar_visible`): when the
/// current value (*default* for a missing entry, Python truthiness
/// deciding) is falsy, a fresh copy of *per_page* gets `True` written for
/// the page; otherwise the map is returned unchanged (still a fresh copy,
/// like every engine outcome). Returns `{"changed": bool,
/// "sidebar_per_page": dict}`.
#[pyfunction]
fn ensure_sidebar_visible<'py>(
    py: Python<'py>,
    per_page: &Bound<'py, PyDict>,
    page: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let key = sidebar_page_key(page);
    let changed = !sidebar_page_visible(per_page, key, default)?;
    let fresh = shallow_copy_dict(py, per_page)?;
    if changed {
        fresh.set_item(key, true)?;
    }
    let out = PyDict::new(py);
    out.set_item("changed", changed)?;
    out.set_item("sidebar_per_page", fresh)?;
    Ok(out)
}

/// The visible panes in left-to-right order — the list behind
/// `SidebarMixin._visible_panes` (and the cycle movement) from
/// `app/_sidebar.py`. Content is always present; the Playlists sidebar
/// leads when its per-page visibility says so and the lyrics pane closes
/// the row while it is open. Both flags are read with Python truthiness
/// (`if flag:`), so a truthy non-bool visibility — the raw stored value
/// from a hand-edited session.json — includes its pane, like the
/// fallback.
#[pyfunction]
fn visible_panes<'py>(
    playlist_sidebar_visible: &Bound<'py, PyAny>,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<Vec<String>> {
    Ok(visible_panes_list(
        playlist_sidebar_visible.is_truthy()?,
        lyrics_open.is_truthy()?,
    )
    .into_iter()
    .map(String::from)
    .collect())
}

/// The shared body of `visible_panes` — panes as static strs so the
/// movement decisions can index into them without reallocation.
fn visible_panes_list(playlist_sidebar_visible: bool, lyrics_open: bool) -> Vec<&'static str> {
    let mut panes: Vec<&'static str> = Vec::with_capacity(3);
    if playlist_sidebar_visible {
        panes.push(PANE_PLAYLISTS);
    }
    panes.push(PANE_CONTENT);
    if lyrics_open {
        panes.push(PANE_LYRICS);
    }
    panes
}

/// `getattr(node, name, None)` — None for a missing attribute or an
/// explicit None value, so a widget without the attribute reads as
/// id-less / parent-less. Any other error propagates, exactly like the
/// fallback's three-argument `getattr`.
fn attr_or_none<'py>(node: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    match node.getattr(name) {
        Ok(value) => Ok(if value.is_none() { None } else { Some(value) }),
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(node.py()) => Ok(None),
        Err(err) => Err(err),
    }
}

/// The pane the focused widget lives in — the pure half of
/// `SidebarMixin._current_pane` from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`pane_of_focused`): walk the
/// focused widget's parent chain (duck-typed `.id` / `.parent` attribute
/// reads, matching the fallback's `getattr(node, ..., None)`) looking for
/// a known pane container (`playlist-sidebar` / `lyrics-sidebar` /
/// `main-content`) and return its pane. The id comparisons go through
/// Python `==`, so a node whose id object defines its own equality
/// resolves exactly like it does on the fallback. A chain with no known
/// container means content — the `_active_pane` hint fallback only
/// applies when nothing is focused, which the mixin handles before
/// calling.
#[pyfunction]
fn pane_of_focused(focused: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut node: Option<Bound<'_, PyAny>> = Some(focused.clone());
    while let Some(current) = node {
        if let Some(node_id) = attr_or_none(&current, "id")? {
            if node_id.eq(ID_PLAYLIST_SIDEBAR)? {
                return Ok(PANE_PLAYLISTS.to_string());
            }
            if node_id.eq(ID_LYRICS_SIDEBAR)? {
                return Ok(PANE_LYRICS.to_string());
            }
            if node_id.eq(ID_MAIN_CONTENT)? {
                return Ok(PANE_CONTENT.to_string());
            }
        }
        node = attr_or_none(&current, "parent")?;
    }
    Ok(PANE_CONTENT.to_string())
}

/// Whether a pane can take keyboard focus, and what to focus — the gate
/// behind `SidebarMixin._focus_pane` from `app/_sidebar.py`.
///
/// Mirrors the pure-Python reference (`pane_focus_target`): the Playlists
/// pane is always focusable (focusing it auto-shows the sidebar), the
/// lyrics pane only while it is open (*lyrics_open* read with Python
/// truthiness), and anything else — the content pane or an unknown pane
/// name — normalises to content. Returns None when the pane cannot take
/// focus (lyrics while closed), meaning "no move".
#[pyfunction]
fn pane_focus_target<'py>(pane: &str, lyrics_open: &Bound<'py, PyAny>) -> PyResult<Option<String>> {
    if pane == PANE_PLAYLISTS {
        Ok(Some(PANE_PLAYLISTS.to_string()))
    } else if pane == PANE_LYRICS {
        if lyrics_open.is_truthy()? {
            Ok(Some(PANE_LYRICS.to_string()))
        } else {
            Ok(None)
        }
    } else {
        Ok(Some(PANE_CONTENT.to_string()))
    }
}

/// Ctrl+w h — the pane left of the current one. Reaching the Playlists
/// sidebar from the content pane auto-shows it (via the focus gate), so
/// left never fails; lyrics steps to content. Any other pane also goes to
/// Playlists, like the reference's else.
///
/// Mirrors `next_pane_left` from `app/_sidebar.py`.
#[pyfunction]
fn next_pane_left(current_pane: &str) -> String {
    if current_pane == PANE_LYRICS {
        PANE_CONTENT.to_string()
    } else {
        PANE_PLAYLISTS.to_string()
    }
}

/// Ctrl+w l — the pane right of the current one, or None at the edge.
/// Playlists steps to content; content enters the lyrics pane only while
/// it is open (*lyrics_open* read with Python truthiness); lyrics (or an
/// unknown pane) is already at the rightmost reachable position — None
/// means "stay put", and the caller skips the focus move entirely.
///
/// Mirrors `next_pane_right` from `app/_sidebar.py`.
#[pyfunction]
fn next_pane_right<'py>(
    current_pane: &str,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    if current_pane == PANE_PLAYLISTS {
        Ok(Some(PANE_CONTENT.to_string()))
    } else if current_pane == PANE_CONTENT && lyrics_open.is_truthy()? {
        Ok(Some(PANE_LYRICS.to_string()))
    } else {
        Ok(None)
    }
}

/// Ctrl+w w — the next visible pane after the current one, wrapping. The
/// cycle runs through `visible_panes` left-to-right; a current pane that
/// isn't visible (hidden lyrics, an unknown name) is treated as content,
/// so the cycle always has a starting point — content is always in the
/// list. Both visibility flags are read with Python truthiness, like
/// `visible_panes`.
///
/// Mirrors `next_pane_cycle` from `app/_sidebar.py`.
#[pyfunction]
fn next_pane_cycle<'py>(
    current_pane: &str,
    playlist_sidebar_visible: &Bound<'py, PyAny>,
    lyrics_open: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let panes = visible_panes_list(
        playlist_sidebar_visible.is_truthy()?,
        lyrics_open.is_truthy()?,
    );
    let current = if panes.contains(&current_pane) {
        current_pane
    } else {
        PANE_CONTENT
    };
    // Content is always in the list, so the fallback above always finds
    // a starting point.
    let idx = panes
        .iter()
        .position(|pane| *pane == current)
        .expect("content pane is always visible");
    Ok(panes[(idx + 1) % panes.len()].to_string())
}

// ---------------------------------------------------------------------------
// Playlist panel view engine (mirrors ui/sidebars/playlist_sidebar.py)
// ---------------------------------------------------------------------------

/// Python's `str.lower()`: the Unicode section-3.13 default lowercase
/// mapping, which lowercases the Greek capital sigma contextually — a
/// word-final `Σ` becomes `ς`, a mid-word one `σ`. Rust's
/// `str::to_lowercase` implements the same mapping (including the final
/// sigma special case), so it agrees with CPython case for case; the
/// per-code-point `char::to_lowercase` does not (it never applies the
/// context rule) and must not be used here.
fn python_lower(s: &str) -> String {
    s.to_lowercase()
}

/// Python's `str.strip()`: trims the CPython `isspace` set (see
/// `py_isspace`) from both ends.
fn python_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// `item.get("playlistId") or item.get("browseId", *browse_default*)` —
/// the shared left half of the panel's three playlist-id reads. A truthy
/// `playlistId` wins outright; otherwise the `browseId` read flows on raw
/// (its default substituting only when the key is missing — a present
/// `None` passes through), so each predicate sees exactly the value the
/// reference's `or`-chain leaves it.
fn pid_or_browse<'py>(
    py: Python<'py>,
    item: &Bound<'py, PyAny>,
    browse_default: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let playlist_id = field_get(py, item, "playlistId", None)?;
    if playlist_id.is_truthy()? {
        return Ok(playlist_id);
    }
    let default = browse_default.map(|d| PyString::new(py, d).into_any());
    field_get(py, item, "browseId", default)
}

/// The row label for one panel item.
///
/// Mirrors `LibraryPanel._format_item` from
/// `ui/sidebars/playlist_sidebar.py`: the title is `title` falling back
/// to `name` (`"Unknown"` when both keys are missing — the inner default
/// is evaluated before the outer lookup, exactly like the reference's
/// eager argument evaluation, and a present-but-`None` title reads as
/// `None`), and a count that is present and not `None` — `0` included,
/// `is not None` is not a truthiness gate — appends
/// `" (<count> tracks)"` with both values interpolated via `str()`.
/// Without a count the title is returned **raw**, whatever object it is,
/// exactly like the reference's bare `return title`.
///
/// Field reads resolve exactly like the reference's `.get` calls (see
/// `field_get`); a `.get`-less item reads as field-less where the
/// reference would raise — the engine stays total.
#[pyfunction]
fn format_panel_item(py: Python, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let unknown = PyString::new(py, "Unknown").into_any();
    let name = field_get(py, item, "name", Some(unknown))?;
    let title = field_get(py, item, "title", Some(name))?;
    let count = field_get(py, item, "count", None)?;
    if !count.is_none() {
        let label = format!("{} ({} tracks)", py_str(&title)?, py_str(&count)?);
        return Ok(PyString::new(py, &label).into_any().unbind());
    }
    Ok(title.unbind())
}

/// The footer count label: `"N item(s)"` when every item is shown,
/// `"shown/total"` once the filter hides some.
#[pyfunction]
fn panel_count_label(shown: i64, total: i64) -> String {
    if shown == total {
        let plural = if total == 1 { "" } else { "s" };
        format!("{total} item{plural}")
    } else {
        format!("{shown}/{total}")
    }
}

/// The `/`-filter pass: the query stripped and lowercased, then the items
/// (a fresh list of the same objects) whose lowercased row label contains
/// it; an empty query resets the view to a fresh copy of every item.
///
/// Mirrors `LibraryPanel.on_input_changed` from
/// `ui/sidebars/playlist_sidebar.py`. Lowercasing is Python's
/// per-code-point mapping on both sides of the comparison (see
/// `python_lower`), and containment is code-point substring, so the two
/// backends agree on every label/query pair. Where the reference would
/// raise — a non-dict item's `.get`, or a non-string raw title hitting
/// `.lower()` — the engine stays total and simply never matches the row.
#[pyfunction]
fn filter_panel_items(
    py: Python,
    items: &Bound<'_, PyList>,
    raw_query: &str,
) -> PyResult<Py<PyList>> {
    // query = raw_query.strip().lower()
    let query = python_lower(python_strip(raw_query));
    let kept: Vec<Bound<'_, PyAny>> = if query.is_empty() {
        items.iter().collect()
    } else {
        let mut kept: Vec<Bound<'_, PyAny>> = Vec::with_capacity(items.len());
        for item in items.iter() {
            // The reference calls `.get` on every item; a non-dict row
            // would raise there, so the engine never matches it (the
            // `filter_sort_tracks` precedent). Dict subclasses keep going
            // and steer through their own `get`.
            if item.downcast::<PyDict>().is_err() {
                continue;
            }
            let label = format_panel_item(py, &item)?;
            let label_bound = label.into_bound(py);
            // A raw non-string title (no count) would raise on `.lower()`;
            // it never matches, like the reference's AttributeError.
            if let Ok(label_str) = label_bound.extract::<String>() {
                if python_lower(&label_str).contains(&query) {
                    kept.push(item);
                }
            }
        }
        kept
    };
    Ok(PyList::new(py, kept)?.unbind())
}

/// `LibraryPanel.remove_item`'s predicate: the item's playlist id
/// (`playlistId` or `browseId`, `""` when both read falsy/missing)
/// equals the target or its `VL`-prefixed form.
#[pyfunction]
fn matches_removal(py: Python, item: &Bound<'_, PyAny>, playlist_id: &str) -> PyResult<bool> {
    let pid = pid_or_browse(py, item, Some(""))?;
    let vl = format!("VL{playlist_id}");
    eq_any_str(&pid, &[playlist_id, vl.as_str()])
}

/// `LibraryPanel.update_item_count`'s predicate: the item's playlist id
/// (`playlistId` or `browseId` or `""`) equals the target, its
/// `VL`-prefixed form, its `VL`-stripped form, or prefixes the target —
/// tolerant in **both** `VL` directions, unlike the removal predicate.
#[pyfunction]
fn matches_count_update(py: Python, item: &Bound<'_, PyAny>, playlist_id: &str) -> PyResult<bool> {
    let empty = PyString::new(py, "").into_any();
    let raw = pid_or_browse(py, item, Some(""))?;
    // … or "" — the trailing operand maps a falsy read to "".
    let pid = if raw.is_truthy()? { raw } else { empty };
    // pid == target or pid == f"VL{target}" — Python `==`, so exotic pid
    // objects steer like the reference.
    let vl_target = format!("VL{playlist_id}");
    if eq_any_str(&pid, &[playlist_id, vl_target.as_str()])? {
        return Ok(true);
    }
    // f"VL{pid}" == playlist_id — the f-string is always a plain str.
    if format!("VL{}", py_str(&pid)?) == playlist_id {
        return Ok(true);
    }
    // pid == playlist_id.removeprefix("VL")
    let stripped = playlist_id.strip_prefix("VL").unwrap_or(playlist_id);
    eq_any_str(&pid, &[stripped])
}

/// `int(x)` for the shapes a cached count can take: bools map to 0/1,
/// ints pass through (a Python int beyond i64 clamps by sign, exactly
/// like the ported volume clamp), floats truncate toward zero, and a str
/// parses as a decimal integer (sign, digits, underscores between digits,
/// surrounding whitespace — see `parse_python_decimal_int`). Where
/// `int()` would raise — NaN/inf floats, non-numeric strings, other
/// types — the coercion reads as None and the caller no-ops (the engine
/// stays total; the loader only ever stores int-or-None counts, and both
/// `update_item_count` call sites wrap the call in `except Exception`).
fn python_int_coerce(value: &Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    // bool is checked first: it passes the int extraction but maps to
    // 0/1, like int(True) == 1.
    if let Ok(b) = value.downcast::<PyBool>() {
        return Ok(Some(i64::from(b.is_true())));
    }
    if let Ok(i) = value.extract::<i64>() {
        return Ok(Some(i));
    }
    if value.is_instance_of::<PyInt>() {
        // A Python int beyond i64: clamps by sign, like CPython's
        // comparison-based max(0, huge + delta) staying huge.
        let negative = value.str()?.to_str()?.starts_with('-');
        return Ok(Some(if negative { i64::MIN } else { i64::MAX }));
    }
    if let Ok(f) = value.extract::<f64>() {
        // int(float) truncates toward zero; at or beyond the i64 range
        // (where Python's int() is exact and ours is not) the count
        // no-ops.
        if f.is_finite() && f.abs() < 9.223_372_036_854_776e18 {
            return Ok(Some(f.trunc() as i64));
        }
        return Ok(None);
    }
    if let Ok(s) = value.extract::<String>() {
        return Ok(parse_python_decimal_int(&s));
    }
    Ok(None)
}

/// `int(str)` for the decimal spellings a hand-stored count string can
/// take: surrounding whitespace, an optional sign, digits with
/// underscores only between them. Python also accepts non-ASCII digits
/// and other `int()` spellings; those (like every string `int()` would
/// reject) read as None so the caller no-ops where the fallback would
/// raise.
fn parse_python_decimal_int(s: &str) -> Option<i64> {
    let trimmed = python_strip(s);
    let mut chars = trimmed.chars().peekable();
    let mut digits = String::with_capacity(trimmed.len());
    if matches!(chars.peek(), Some('+')) {
        chars.next();
    } else if matches!(chars.peek(), Some('-')) {
        digits.push('-');
        chars.next();
    }
    let mut last_was_digit = false;
    let mut any_digit = false;
    for c in chars {
        if c.is_ascii_digit() {
            digits.push(c);
            last_was_digit = true;
            any_digit = true;
        } else if c == '_' && last_was_digit {
            // An underscore must sit between digits ("1_0" is 10,
            // "1__0" and "1_" are rejected, like int()).
            last_was_digit = false;
        } else {
            return None;
        }
    }
    if !any_digit || !last_was_digit {
        return None;
    }
    match digits.parse::<i128>() {
        Ok(v) => Some(v.clamp(i64::MIN as i128, i64::MAX as i128) as i64),
        // A digit string beyond i128 clamps by sign, like the huge-int
        // branch of python_int_coerce.
        Err(_) => Some(if digits.starts_with('-') {
            i64::MIN
        } else {
            i64::MAX
        }),
    }
}

/// The optimistic count arithmetic: `None` (unknown, or a missing count
/// that reads as `None`) stays `None` so the caller leaves the count for
/// the next library reload; otherwise `max(0, int(current) + delta)` —
/// never negative, saturating so a near-i64 count clamps instead of
/// wrapping. An `int()`-proof current (see `python_int_coerce`) also
/// reads as `None`: the engine no-ops where the fallback would raise.
#[pyfunction]
fn bump_playlist_count(py: Python, current: &Bound<'_, PyAny>, delta: i64) -> PyResult<Py<PyAny>> {
    if current.is_none() {
        return Ok(py.None());
    }
    match python_int_coerce(current)? {
        Some(v) => {
            let bumped = (v as i128 + delta as i128).max(0).min(i64::MAX as i128) as i64;
            Ok(bumped.into_pyobject(py)?.into_any().unbind())
        }
        None => Ok(py.None()),
    }
}

/// The info-bar read behind `SelectionChanged`: `title` falling back to
/// `name`, `""` when both keys are missing — *not* the label composer's
/// `"Unknown"` — and returned raw. The inner default is evaluated before
/// the outer lookup, like the reference's eager argument evaluation.
#[pyfunction]
fn selection_title(py: Python, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let empty = PyString::new(py, "").into_any();
    let name = field_get(py, item, "name", Some(empty))?;
    Ok(field_get(py, item, "title", Some(name))?.unbind())
}

/// Whether the item is the account's auto-generated Liked Music playlist
/// (id `LM`) — the row `ensure_loaded` filters out because it is already
/// a pinned nav item.
#[pyfunction]
fn is_liked_music_playlist(py: Python, item: &Bound<'_, PyAny>) -> PyResult<bool> {
    // (item.get("playlistId") or item.get("browseId", "")) == "LM"
    let pid = pid_or_browse(py, item, Some(""))?;
    eq_any_str(&pid, &["LM"])
}

/// `PlaylistSidebar.auto_select_playlist`'s predicate: the item's playlist
/// id (`playlistId` or `browseId`, no `""` default — a missing `browseId`
/// reads as `None`) equals the target exactly, with no `VL` tolerance.
#[pyfunction]
fn matches_auto_select(py: Python, item: &Bound<'_, PyAny>, playlist_id: &str) -> PyResult<bool> {
    let pid = pid_or_browse(py, item, None)?;
    eq_any_str(&pid, &[playlist_id])
}

// ---------------------------------------------------------------------------
// Context-page view-model assembly (mirrors ui/pages/context.py)
// ---------------------------------------------------------------------------

/// `"<n> track(s)"` — the count segment of every context subtitle, with
/// the reference's `!= 1` plural rule (0 and 2 both pluralise).
fn track_count_label(n: i64) -> String {
    if n != 1 {
        format!("{n} tracks")
    } else {
        format!("{n} track")
    }
}

/// The normalized track list for a raw `tracks` value, plus that raw
/// value's `len()`. A raw value that is not a list — where the fallback's
/// `normalize_tracks` would raise and the ported native one rejects at
/// the boundary — reads as empty with an unknown count, so the engine
/// stays total and the caller's unavailable/remaining decisions degrade
/// to "nothing hidden" / "nothing more to fetch".
fn normalized_tracks_and_len<'py>(
    py: Python<'py>,
    raw_tracks: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyList>, Option<i64>)> {
    match raw_tracks.downcast::<PyList>() {
        Ok(list) => {
            let len = list.len() as i64;
            Ok((normalize_tracks(py, list)?, Some(len)))
        }
        Err(_) => Ok((PyList::empty(py), None)),
    }
}

/// Python's `value > n` for the numeric shapes a stored `trackCount` can
/// take: bools compare as 0/1, ints as themselves (a Python int beyond
/// i64 clamps by sign, which only ever decides against realistic
/// counts), floats as their f64. Any other type — where the reference's
/// `>` would raise `TypeError` — reads as `None`, so the caller skips
/// the gated decision instead of raising.
fn numeric_gt(value: &Bound<'_, PyAny>, n: i64) -> Option<bool> {
    if let Ok(b) = value.downcast::<PyBool>() {
        return Some(i64::from(b.is_true()) > n);
    }
    if let Ok(i) = value.extract::<i64>() {
        return Some(i > n);
    }
    if value.is_instance_of::<PyInt>() {
        // A Python int beyond the i64 range: only its sign can decide.
        let negative = value.str().ok()?.to_str().ok()?.starts_with('-');
        return Some(!negative);
    }
    if let Ok(f) = value.extract::<f64>() {
        return Some(f > n as f64);
    }
    None
}

/// `section.get("results", [])` for one of the artist page's sections —
/// `None` when *data* has no such section, the section is not a dict, or
/// its `results` is not a list (each a raise in the fallback's
/// concatenation; the engine stays total and the section contributes no
/// rows).
fn section_results_opt<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Option<Bound<'py, PyList>>> {
    let section = field_get(py, data, key, None)?;
    if !section.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    let empty = PyList::empty(py).into_any();
    let results = field_get(py, &section, "results", Some(empty))?;
    Ok(match results.downcast::<PyList>() {
        Ok(list) => Some(list.clone()),
        Err(_) => None,
    })
}

/// The sections of a `get_artist` payload that carry a "see all" page —
/// a dict section with truthy `browseId` and `params` — in the
/// albums-then-singles order the reference's comprehension produces. The
/// returned sections are the original objects, so the caller's
/// `section["results"] = full` swap lands in the fetched payload.
///
/// Mirrors the selection half of
/// `ContextPage._expand_artist_discography` from `ui/pages/context.py`;
/// the concurrent `get_artist_albums` fetch stays in Python (it is I/O).
#[pyfunction]
fn expandable_sections<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut sections: Vec<Bound<'py, PyAny>> = Vec::new();
    for key in ["albums", "singles"] {
        let section = field_get(py, data, key, None)?;
        if !section.is_instance_of::<PyDict>() {
            continue;
        }
        let browse_id = field_get(py, &section, "browseId", None)?;
        if !browse_id.is_truthy()? {
            continue;
        }
        let params = field_get(py, &section, "params", None)?;
        if !params.is_truthy()? {
            continue;
        }
        sections.push(section);
    }
    PyList::new(py, sections)
}

/// The album page's view model — the pure core of
/// `ContextPage._build_album` from `ui/pages/context.py`.
///
/// Returned keys:
///
/// - `title` — `data.get("title", "Unknown Album")`, raw.
/// - `subtitle` — `"<artist> · <year> · <n> track(s)"`: the artist
///   through the shared `extract_artist`, the year segment only when
///   truthy and interpolated via `str()`, the count through the shared
///   plural rule.
/// - `tracks` — `normalize_tracks(data.get("tracks", []))`.
/// - `unavailable` — `len(raw) - len(normalized)`, the count of
///   unplayable entries the header reports as hidden; `None` when the
///   raw value is not a list (the header's label is truthiness-gated,
///   so `None` hides it where the fallback would have raised).
#[pyfunction]
fn album_header<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let unknown_album = PyString::new(py, "Unknown Album").into_any();
    let title = field_get(py, data, "title", Some(unknown_album))?;

    // extract_artist(data) — the shared native engine. A non-dict data
    // (whose field reads above already degraded to their defaults)
    // reads as Unknown, like the search engine's album branch.
    let artist = match data.downcast::<PyDict>() {
        Ok(dict) => extract_artist(py, dict),
        Err(_) => "Unknown".to_string(),
    };

    let year = field_get(py, data, "year", Some(PyString::new(py, "").into_any()))?;
    let empty_tracks = PyList::empty(py).into_any();
    let raw_tracks = field_get(py, data, "tracks", Some(empty_tracks))?;
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    let track_count = tracks.len() as i64;

    // subtitle_parts = [artist_name]; if year: append str(year); append
    // f"{n} track(s)" — joined on " · ".
    let mut parts: Vec<String> = vec![artist];
    if year.is_truthy()? {
        parts.push(py_str(&year)?);
    }
    parts.push(track_count_label(track_count));

    let out = PyDict::new(py);
    out.set_item("title", title)?;
    out.set_item("subtitle", parts.join(" \u{b7} "))?;
    out.set_item("tracks", tracks)?;
    match raw_len {
        Some(len) => out.set_item("unavailable", len - track_count)?,
        None => out.set_item("unavailable", pyo3::types::PyNone::get(py))?,
    }
    Ok(out)
}

/// The playlist page's view model — the pure core of
/// `ContextPage._build_playlist` from `ui/pages/context.py`.
///
/// Returned keys:
///
/// - `title` — `data.get("title", "Unknown Playlist")`, raw.
/// - `subtitle` — `"<owner> · <n> track(s)"` plus `" (loading <total>
///   total…)"` when the stored `trackCount` is truthy and numerically
///   above the normalized count (the owner is `author.get("name",
///   "Unknown")` for a dict `author`, `str(author)` otherwise).
/// - `tracks` — `normalize_tracks(data.get("tracks", []))`.
/// - `unavailable` — `len(raw) - len(normalized)`; `None` when the raw
///   value is not a list.
/// - `owned` — truthiness of `data.get("owned", False)` (the header's
///   Add-to-Library gate).
/// - `fetch_remaining` — `trackCount > len(raw)`, the gate behind the
///   background fetch of the playlist's tail.
/// - `raw_count` — `len(raw)`, the `already_have` the tail fetch
///   resumes from; 0 whenever the gate above is closed.
#[pyfunction]
fn playlist_header<'py>(py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let unknown_playlist = PyString::new(py, "Unknown Playlist").into_any();
    let title = field_get(py, data, "title", Some(unknown_playlist))?;

    // owner = author.get("name", "Unknown") if isinstance(author, dict)
    //         else str(author)
    let empty_author = PyDict::new(py).into_any();
    let author = field_get(py, data, "author", Some(empty_author))?;
    let owner = if author.is_instance_of::<PyDict>() {
        let unknown_name = PyString::new(py, "Unknown").into_any();
        py_str(&field_get(py, &author, "name", Some(unknown_name))?)?
    } else {
        py_str(&author)?
    };

    let empty_tracks = PyList::empty(py).into_any();
    let raw_tracks = field_get(py, data, "tracks", Some(empty_tracks))?;
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    let track_count = tracks.len() as i64;

    // total_count = data.get("trackCount") or track_count — the suffix
    // and the remaining-fetch gate only ever fire for a truthy stored
    // count that is numerically above the comparison count; a stored
    // count Python's `>` would reject reads as uncomparable and skips
    // both (the engine stays total).
    let raw_total = field_get(py, data, "trackCount", None)?;
    let total = if raw_total.is_truthy()? {
        Some(raw_total)
    } else {
        None
    };

    let mut subtitle = format!("{owner} \u{b7} {}", track_count_label(track_count));
    let mut fetch_remaining = false;
    if let Some(ref total_value) = total {
        // if total_count > track_count: subtitle += f" (loading {total} total…)"
        if numeric_gt(total_value, track_count) == Some(true) {
            subtitle.push_str(&format!(
                " (loading {} total\u{2026})",
                py_str(total_value)?
            ));
        }
        // if total_count > len(raw_tracks): fetch the tail
        if let Some(raw_count) = raw_len {
            fetch_remaining = numeric_gt(total_value, raw_count) == Some(true);
        }
    }

    // owned = data.get("owned", False) — the gate is truthiness and the
    // default False is falsy like the None a missing key reads as, so
    // one truthiness read covers both.
    let owned = field_get(py, data, "owned", None)?.is_truthy()?;

    let out = PyDict::new(py);
    out.set_item("title", title)?;
    out.set_item("subtitle", subtitle)?;
    out.set_item("tracks", tracks)?;
    match raw_len {
        Some(len) => out.set_item("unavailable", len - track_count)?,
        None => out.set_item("unavailable", pyo3::types::PyNone::get(py))?,
    }
    out.set_item("owned", owned)?;
    out.set_item("fetch_remaining", fetch_remaining)?;
    out.set_item("raw_count", raw_len.unwrap_or(0))?;
    Ok(out)
}

/// The artist page's view model — the pure core of
/// `ContextPage._build_artist` from `ui/pages/context.py`, the
/// discography assembly at the heart of the context page.
///
/// Returned keys:
///
/// - `name` — `data.get("name", "Unknown Artist")`, raw.
/// - `subscriber_text` — `f"{subs} subscribers"` for a truthy
///  `subscribers` value (interpolated via `str()`), `""` otherwise.
/// - `top_songs` — `normalize_tracks(songs.get("results", []))` for a
///   dict `songs` section, `[]` otherwise.
/// - `songs_browse_id` — `songs.get("browseId")` for a dict `songs`
///   section (the playlist holding the artist's full song list, fetched
///   in the background), `None` otherwise.
/// - `albums` — the albums section's `results` concatenated after the
///   singles section's — the discography rows, original objects in
///   feed order.
/// - `related_text` — `"Similar Artists: " + " · ".join(truthy names)`
///   over the first 8 related artists (a dict row's `name` with its
///   truthiness gating the join, `str(row)` for anything else);
///   `""` when there are no truthy `related` results.
#[pyfunction]
fn artist_discography<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let unknown_artist = PyString::new(py, "Unknown Artist").into_any();
    let name = field_get(py, data, "name", Some(unknown_artist))?;

    // subscriber_text = f"{subs} subscribers" if subs else ""
    let subs = field_get(
        py,
        data,
        "subscribers",
        Some(PyString::new(py, "").into_any()),
    )?;
    let subscriber_text = if subs.is_truthy()? {
        format!("{} subscribers", py_str(&subs)?)
    } else {
        String::new()
    };

    // songs_section = data.get("songs", {}); top_songs and browseId are
    // read through the isinstance gate.
    let empty_section = PyDict::new(py).into_any();
    let songs = field_get(py, data, "songs", Some(empty_section.clone()))?;
    let (top_songs, songs_browse_id) = if songs.is_instance_of::<PyDict>() {
        let empty_results = PyList::empty(py).into_any();
        let results = field_get(py, &songs, "results", Some(empty_results))?;
        let top = normalized_tracks_and_len(py, &results)?.0;
        let browse_id = field_get(py, &songs, "browseId", None)?;
        (top, browse_id)
    } else {
        (PyList::empty(py), py.None().into_bound(py))
    };

    // all_albums = albums_list + singles_list
    let mut album_rows: Vec<Bound<'_, PyAny>> = Vec::new();
    for key in ["albums", "singles"] {
        if let Some(list) = section_results_opt(py, data, key)? {
            album_rows.extend(list.iter());
        }
    }

    // related_text = "Similar Artists: " + " · ".join(names of the first
    // 8 related artists, keeping only truthy ones) — gated on truthy
    // results; a truthy non-list (where the fallback's [:8] would
    // raise) contributes no text.
    let related = field_get(py, data, "related", Some(empty_section))?;
    let mut related_text = String::new();
    if related.is_instance_of::<PyDict>() {
        let empty_results = PyList::empty(py).into_any();
        let results = field_get(py, &related, "results", Some(empty_results))?;
        if results.is_truthy()? {
            if let Ok(list) = results.downcast::<PyList>() {
                let mut names: Vec<String> = Vec::new();
                for entry in list.iter().take(8) {
                    // a.get("name", "") if isinstance(a, dict) else str(a)
                    // — the join's `if n` filter runs on the value the
                    // comprehension produced: the raw name for dict rows
                    // (stringified only where the fallback's join would
                    // raise on a non-str), the already-stringified row
                    // for everything else.
                    let name_str = if entry.is_instance_of::<PyDict>() {
                        let name_value =
                            field_get(py, &entry, "name", Some(PyString::new(py, "").into_any()))?;
                        if name_value.is_truthy()? {
                            Some(py_str(&name_value)?)
                        } else {
                            None
                        }
                    } else {
                        let s = py_str(&entry)?;
                        if !s.is_empty() {
                            Some(s)
                        } else {
                            None
                        }
                    };
                    if let Some(s) = name_str {
                        names.push(s);
                    }
                }
                related_text = format!("Similar Artists: {}", names.join(" \u{b7} "));
            }
        }
    }

    let out = PyDict::new(py);
    out.set_item("name", name)?;
    out.set_item("subscriber_text", subscriber_text)?;
    out.set_item("top_songs", top_songs)?;
    out.set_item("songs_browse_id", songs_browse_id)?;
    out.set_item("albums", PyList::new(py, album_rows)?)?;
    out.set_item("related_text", related_text)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Play-history row engine (mirrors ytm_player/services/history.py)
// ---------------------------------------------------------------------------

/// `int(value)` — CPython's own builtin, dispatched directly.
///
/// The history row engine coerces caller-wide values (SQLite cells,
/// configured thresholds, track duration fields), where `int()`'s
/// acceptance set is larger than any transcription wants to spell out:
/// float truncation, decimal strings with signs, underscores, surrounding
/// whitespace and non-ASCII digits, `__int__`/`__index__` dispatch,
/// bigints beyond i64 — and the exact `ValueError`/`TypeError` messages
/// where the reference would raise. The coercion runs once per logged
/// play or finalised listen, as far from a hot path as the extension
/// gets, so exactness costs nothing measurable.
fn builtin_int<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let int_type = py.import("builtins")?.getattr("int")?;
    int_type.call1((value,))
}

/// The duration chain behind `log_play`'s row assembly — a duck-typed
/// transcription of `extract_duration` from `utils/formatting.py`, every
/// read through the track's own `get` method so a dict subclass
/// overriding it steers both backends identically.
///
/// `duration_seconds` is coerced through `int()` whenever present (the
/// reference's `is not None` gate — `False` and `""` both coerce), with
/// the `(ValueError, TypeError)` swallow: a numeric string parses, a
/// `"M:SS"` spelling falls through, a float truncates. `duration` and
/// `length` accept any int — bools included, the object itself returned
/// (`True` stays `True`, exactly like the fallback) — and
/// `":"`-bearing strings, split and recombined with Python arithmetic
/// (`int(a) * 60 + int(b)` for `M:SS`, the `H:MM:SS` three-part form,
/// a `ValueError` from either parse swallowed, everything else passing
/// through). No match yields 0. Where a read raises — a track without
/// `get`, or a `get` that raises — the error surfaces exactly where the
/// fallback's would.
fn play_extract_duration<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // dur = track.get("duration_seconds") — duck-typed, no default.
    let dur = track.call_method1("get", ("duration_seconds",))?;
    if !dur.is_none() {
        // return int(dur) — except (ValueError, TypeError): pass
        match builtin_int(py, &dur) {
            Ok(coerced) => return Ok(coerced),
            Err(err)
                if err.is_instance_of::<PyValueError>(py)
                    || err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) => {}
            Err(err) => return Err(err),
        }
    }
    for key in ["duration", "length"] {
        // dur = track.get(key)
        let dur = track.call_method1("get", (key,))?;
        // if isinstance(dur, int): return dur — the object itself, bools
        // included (the reference returns True, never 1).
        if dur.is_instance_of::<PyInt>() {
            return Ok(dur);
        }
        // if isinstance(dur, str) and ":" in dur
        if dur.is_instance_of::<PyString>() && dur.contains(":")? {
            // parts = dur.split(":") — the str protocol, subclass
            // overrides included.
            let parts = dur.call_method1("split", (":",))?;
            if let Some(seconds) = duration_parts_seconds(py, &parts)? {
                return Ok(seconds);
            }
        }
    }
    Ok(0i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any())
}

/// `int(parts[0]) * 60 + int(parts[1])` (and the `H:MM:SS` three-part
/// form) over a split duration string — Python's own arithmetic through
/// the number protocol, exact at any magnitude. A `ValueError` from
/// either `int()` parse maps to None: the reference's
/// `except ValueError: pass` falls through to the next duration key.
/// Everything else — a split override returning something unindexable,
/// a `TypeError` from a non-string part — propagates, exactly like the
/// fallback.
fn duration_parts_seconds<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // len(parts) == 2 / == 3 — PyObject_Size, raising like len() would.
    // Any other length (a "1:2:3:4" spelling, say) matches neither of the
    // reference's two if-statements and falls through to the next
    // duration key without parsing anything. The parses and arithmetic
    // live in the helper so a `?` there returns from *it*, landing in
    // the ValueError match below — not straight out of this function.
    let total = match parts.len()? {
        2 => combine_duration_parts(py, parts, 2),
        3 => combine_duration_parts(py, parts, 3),
        _ => return Ok(None),
    };
    match total {
        Ok(value) => Ok(Some(value)),
        Err(err) if err.is_instance_of::<PyValueError>(py) => Ok(None),
        Err(err) => Err(err),
    }
}

/// The pieces of one split duration string, parsed and recombined —
/// `int(parts[0]) * 60 + int(parts[1])` for `M:SS`, the `H:MM:SS`
/// three-part form for a length of 3. Python's own arithmetic through
/// the number protocol, exact at any magnitude.
fn combine_duration_parts<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyAny>,
    len: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let first = builtin_int(py, &parts.get_item(0)?)?;
    let second = builtin_int(py, &parts.get_item(1)?)?;
    if len == 2 {
        return first.mul(60i64)?.add(&second);
    }
    let third = builtin_int(py, &parts.get_item(2)?)?;
    first.mul(3600i64)?.add(&second.mul(60i64)?)?.add(&third)
}

/// The row assembly behind `HistoryManager.log_play` — the pure core of
/// `services/history.py`'s play logging.
///
/// Mirrors the pure-Python reference (`_play_log_row`): the skip gate
/// comes first — `listened_seconds <= min_listen_seconds` through
/// Python's own `<=` (bools compare as ints, floats numerically, and the
/// same `TypeError` where Python's comparison would reject the pair) —
/// returning None for every listen the reference would drop before
/// touching anything else. Otherwise the insert row is assembled from
/// the track: `video_id` through `[]` (the reference's one indexing
/// read, a missing key raising KeyError), `title` / `artist` / `album`
/// through duck-typed `.get` with the `""` defaults, and `duration`
/// through the duration chain above. Every value passes through as the
/// object it is — `listened_seconds` and `source` included — into a
/// fresh map, and the inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (track, listened_seconds, source, min_listen_seconds))]
fn play_log_row<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
    listened_seconds: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
    min_listen_seconds: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // if listened_seconds <= min_listen_seconds: return None
    if listened_seconds.le(min_listen_seconds)? {
        return Ok(None);
    }
    // video_id = track["video_id"] — the [] read, KeyError wart included.
    let video_id = track.get_item("video_id")?;
    // title/artist/album = track.get(key, "") — duck-typed .get with the
    // "" defaults; an object without .get raises the reference's
    // AttributeError.
    let empty: Bound<'py, PyAny> = PyString::new(py, "").into_any();
    let title = track.call_method1("get", ("title", empty.clone()))?;
    let artist = track.call_method1("get", ("artist", empty.clone()))?;
    let album = track.call_method1("get", ("album", empty))?;
    // duration = extract_duration(track)
    let duration = play_extract_duration(py, track)?;
    let out = PyDict::new(py);
    out.set_item("video_id", video_id)?;
    out.set_item("title", title)?;
    out.set_item("artist", artist)?;
    out.set_item("album", album)?;
    out.set_item("duration", duration)?;
    out.set_item("listened_seconds", listened_seconds)?;
    out.set_item("source", source)?;
    Ok(Some(out))
}

/// The listened-seconds update decision behind
/// `HistoryManager.update_play_listened_seconds`.
///
/// Mirrors the pure-Python reference (`_listened_delta`): the stored
/// value is coerced with `int(old_value or 0)` — Python truthiness
/// picking the 0 for every falsy cell (None, 0, 0.0, "", False, empty
/// containers), the builtin `int()` for everything else, with its exact
/// `ValueError`/`TypeError` where the reference's coercion would raise
/// (both propagate identically out of the caller). A listen that does
/// not strictly exceed the stored seconds writes nothing (None).
/// Otherwise the delta — `listened_seconds - old_seconds` through
/// Python's own subtraction — comes back as the object the reference's
/// arithmetic would produce, exact at any magnitude.
#[pyfunction]
#[pyo3(signature = (old_value, listened_seconds))]
fn listened_delta<'py>(
    py: Python<'py>,
    old_value: &Bound<'py, PyAny>,
    listened_seconds: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // old_seconds = int(old_value or 0)
    let old_seconds = if old_value.is_truthy()? {
        builtin_int(py, old_value)?
    } else {
        0i64.into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any()
    };
    // if listened_seconds <= old_seconds: return
    if listened_seconds.le(&old_seconds)? {
        return Ok(None);
    }
    // delta = listened_seconds - old_seconds
    Ok(Some(listened_seconds.sub(&old_seconds)?))
}

/// The command palette catalog behind `YTMCommandProvider` from
/// `app/_commands.py` — the pure core of the command provider.
///
/// Mirrors the pure-Python reference (`_command_catalog`): the
/// `(name, help)` pair for every ytm-player command, in dispatch
/// order — the data both the discovery list and the query search
/// walk. The names are the palette's dispatch keys (the Python side
/// resolves each to its host callback through `_command_callback`),
/// so they must stay unique and in sync with the resolvers. A fresh
/// list of fresh tuples comes back on every call — nothing is cached,
/// so no shared mutable state can ever leak between calls.
#[pyfunction]
fn command_catalog() -> Vec<(String, String)> {
    vec![(
        "Theme: Set Current as Default".to_string(),
        "Save the active theme to config.toml".to_string(),
    )]
}

/// The aggregate listening-stats payload behind
/// `HistoryManager.get_stats` — the pure core of the stats read.
///
/// Mirrors the pure-Python reference (`_stats_payload`): the five-key
/// dict assembled from the three scalar aggregates and the two top
/// lists. Every value passes through as the object it is — the scalar
/// counts straight from the SQLite rows, the top lists the very list
/// objects the caller fetched — into a fresh map, in the reference's
/// insertion order (the key order the JSON consumers' output
/// preserves). The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (total_plays, total_listen_time, unique_tracks, top_tracks, top_artists))]
fn stats_payload<'py>(
    py: Python<'py>,
    total_plays: &Bound<'py, PyAny>,
    total_listen_time: &Bound<'py, PyAny>,
    unique_tracks: &Bound<'py, PyAny>,
    top_tracks: &Bound<'py, PyAny>,
    top_artists: &Bound<'py, PyAny>,
) -> Bound<'py, PyDict> {
    let out = PyDict::new(py);
    // Only str keys are inserted — always hashable, so the store cannot
    // fail; each value is stored as the object it is, never coerced.
    out.set_item("total_plays", total_plays)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("total_listen_time", total_listen_time)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("unique_tracks", unique_tracks)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("top_tracks", top_tracks)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("top_artists", top_artists)
        .expect("dict insertion with a str key cannot fail");
    out
}

/// The row materialisation behind every history read — the
/// `[dict(row) for row in rows]` list comprehension the history
/// getters (and the CLI's `_query_history`) each ran inline.
///
/// Mirrors the pure-Python reference (`_row_dicts`): *rows* is walked
/// through Python's iteration protocol (a non-iterable raises
/// CPython's own `TypeError`; an `__iter__` that itself raises keeps
/// its own error) and each yielded row is converted through CPython's
/// own `dict` constructor — fetched from `builtins` and called with
/// the row as its single argument, so every acceptance shape and every
/// rejection message comes free: an exact dict is copied, anything
/// with a `keys` method is read through the mapping protocol
/// (`sqlite3.Row` — the production shape — and duck-typed rows
/// alike, a `keys`/`__getitem__` override on a dict subclass
/// steering the conversion), any other iterable is consumed as
/// key/value pairs, and everything else raises the constructor's own
/// `TypeError`. The result is a fresh list of fresh dicts — the input
/// rows are never mutated, and a dict row is copied, never passed
/// through by identity.
#[pyfunction]
#[pyo3(signature = (rows))]
fn row_dicts<'py>(py: Python<'py>, rows: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    // dict(row) — CPython's own constructor, fetched from builtins at
    // call time exactly like the comprehension's global lookup, so the
    // conversion semantics (mapping protocol, pair iteration, dict
    // copy, and every TypeError message) are the interpreter's own.
    let dict_ctor = PyModule::import(py, "builtins")?.getattr("dict")?;
    let out = PyList::empty(py);
    for row in rows.try_iter()? {
        let converted = dict_ctor.call1((row?,))?;
        // dict(...) always returns an exact dict, never a subclass.
        out.append(converted)?;
    }
    Ok(out)
}

/// The aggregate listening-stats payload behind the CLI's standalone
/// `ytm stats` command — the pure core of the stats read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cli_stats_payload`): the
/// four-key dict assembled from the three scalar aggregates and the
/// top-tracks list, in the reference's insertion order (the key order
/// the JSON output preserves — `total_plays`, `total_seconds`,
/// `unique_tracks`, `top_tracks`). The three scalars pass through as
/// the objects they are — the counts straight from the SQLite
/// `fetchone()` cells. The top-tracks list is rebuilt by walking
/// *top_tracks* through Python's iteration protocol and reading each
/// row's first four fields through `r[0]` … `r[3]` subscript semantics
/// (`PyObject_GetItem`, exactly what the comprehension's
/// `BINARY_SUBSCR` compiles to): a tuple row yields `IndexError` when
/// shorter than four, a dict row `KeyError` on the int key, a
/// duck-typed `__getitem__` row steers both backends identically, and
/// a longer row's extras are simply never read — each row becoming a
/// fresh `{"video_id", "title", "artist", "play_count"}` map in that
/// order. The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (total_plays, total_seconds, unique_tracks, top_tracks))]
fn cli_stats_payload<'py>(
    py: Python<'py>,
    total_plays: &Bound<'py, PyAny>,
    total_seconds: &Bound<'py, PyAny>,
    unique_tracks: &Bound<'py, PyAny>,
    top_tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // Only str keys are inserted — always hashable, so the store cannot
    // fail; each scalar is stored as the object it is, never coerced.
    out.set_item("total_plays", total_plays)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("total_seconds", total_seconds)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("unique_tracks", unique_tracks)
        .expect("dict insertion with a str key cannot fail");
    // [{"video_id": r[0], "title": r[1], "artist": r[2],
    //   "play_count": r[3]} for r in top_tracks]
    let items = PyList::empty(py);
    for row in top_tracks.try_iter()? {
        let row = row?;
        let item = PyDict::new(py);
        for (key, index) in [
            ("video_id", 0usize),
            ("title", 1usize),
            ("artist", 2usize),
            ("play_count", 3usize),
        ] {
            // r[index] — PyObject_GetItem: IndexError on a short
            // sequence, KeyError(0) on a dict row, whatever a
            // duck-typed __getitem__ decides.
            item.set_item(key, row.get_item(index)?)?;
        }
        items.append(item)?;
    }
    out.set_item("top_tracks", items)
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The dispatch table behind `IPCMixin._handle_ipc_command` from
/// `app/_ipc.py` — which branch of the match handles each IPC command
/// string, and the rating the like/dislike/unlike branches rate with.
///
/// Mirrors the pure-Python reference (`_ipc_dispatch_route`): each row
/// of *table* is compared in order with Python `==` (subject on the
/// left — exactly like the reference `match`'s lazily evaluated
/// literal value patterns, which compile to `command == literal`
/// comparisons, first equal match wins) and anything no row claims is
/// unknown. *table* is the module's own ordered
/// `(command, branch, rating)` tuple in production; a stub table
/// drives the walk identically, with row unpacking raising CPython's
/// own for-loop errors (`ValueError` for wrong-length rows, the
/// `cannot unpack non-iterable …` `TypeError` rewrite only for a type
/// with no iteration support at all). The outcome map carries
/// `{"branch": str, "rating": str | None}` — fresh in every field,
/// with the matched row's branch/rating objects passed through
/// unchanged — and neither *command* nor *table* is mutated.
#[pyfunction]
#[pyo3(signature = (command, table))]
fn ipc_dispatch_route<'py>(
    py: Python<'py>,
    command: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    for row in table.try_iter()? {
        let (member, branch, rating) = unpack_dispatch_row(&row?)?;
        // command == member — subject on the left, exactly like the
        // reference match's compiled value patterns.
        if command.eq(&member)? {
            let out = PyDict::new(py);
            out.set_item("branch", branch)?;
            out.set_item("rating", rating)?;
            return Ok(out);
        }
    }
    let out = PyDict::new(py);
    out.set_item("branch", "unknown")?;
    out.set_item("rating", py.None())?;
    Ok(out)
}

/// `float(value)` — the builtin fetched from `builtins` at call time,
/// exactly like the reference's global lookup, so the conversion
/// semantics (and any mutation of `builtins.float`) are the
/// interpreter's own: the string grammar with its whitespace,
/// underscores and `inf`/`nan` spellings, `__float__`/`__index__`
/// coercion, and CPython's own `ValueError`/`TypeError` shapes.
fn builtin_float<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let float_type = py.import("builtins")?.getattr("float")?;
    float_type.call1((value,))
}

/// A fresh `{"kind": …}` outcome row for the seek parser, with the
/// branch's payload (`seconds` for the parsed kinds, `offset` for the
/// invalid ones) passed through as the object it is.
fn seek_outcome<'py>(
    py: Python<'py>,
    kind: &str,
    payload: Option<(&str, &Bound<'py, PyAny>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("kind", kind)?;
    if let Some((key, value)) = payload {
        out.set_item(key, value)?;
    }
    Ok(out)
}

/// The seek-offset parser behind `IPCMixin._ipc_seek` from
/// `app/_ipc.py` — the classification of one offset value into the
/// seek it names.
///
/// Mirrors the pure-Python reference (`_ipc_seek_offset`): the offset
/// is the raw `args.get("offset", "")` value, whatever object it is. A
/// falsy one (the missing key's `""` default, `None`, `0`, `0.0`,
/// `False`, every empty container) is missing — Python truthiness,
/// gated before any attribute is touched. A truthy one is probed with
/// `startswith("+")` / `startswith("-")` — both through the instance's
/// own method lookup, short-circuited exactly like the reference's
/// `or` (a value claiming `+` never probes `-`), so a stub's override
/// steers both backends identically and a non-str value raises
/// CPython's own `AttributeError` right where the reference's first
/// call would (`bytes` its `TypeError` from the str-prefixed call). A
/// signed value is relative — `float(offset)` through the builtin
/// fetched at call time, its `ValueError` mapping to the
/// invalid-offset verdict while every other error keeps its own shape.
/// A value containing `":"` (the containment protocol on the right
/// operand) is a clock — `split(":")` outside the reference's `try` so
/// a raising split keeps its error, then `len(parts)` inside it, then
/// the two- and three-part parses with `int()` in the reference's
/// evaluation order and Python's own `*`/`+` arithmetic
/// (`int(a)*60 + int(b)`, `int(a)*3600 + int(b)*60 + int(c)` — exact
/// at any magnitude, a Unicode-digit or underscore-bearing part parsing
/// exactly like it would at the call site); any `ValueError` from the
/// parses maps to the invalid-time verdict while a part's own
/// `TypeError` propagates, and `float(total)` sits outside the `try` so
/// a total too large for a float keeps its `OverflowError`. Everything
/// else is absolute seconds — `float(offset)` with the same
/// `ValueError` verdict. The outcome is a fresh `{"kind": …}` map —
/// `"missing"`, `"relative"`/`"clock"`/`"absolute"` with the parsed
/// `seconds` object, or `"invalid_offset"`/`"invalid_time"` carrying
/// the raw offset for the response message — and the input is never
/// mutated.
#[pyfunction]
#[pyo3(signature = (offset))]
fn ipc_seek_offset<'py>(
    py: Python<'py>,
    offset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // if not offset_str: -> the missing-offset verdict, before any
    // attribute is touched.
    if !offset.is_truthy()? {
        return seek_outcome(py, "missing", None);
    }
    // offset_str.startswith("+") or offset_str.startswith("-")
    let startswith = offset.getattr("startswith")?;
    let relative =
        startswith.call1(("+",))?.is_truthy()? || startswith.call1(("-",))?.is_truthy()?;
    if relative {
        // seconds = float(offset_str) — except ValueError
        return match builtin_float(py, offset) {
            Ok(seconds) => seek_outcome(py, "relative", Some(("seconds", &seconds))),
            Err(err) if err.is_instance_of::<PyValueError>(py) => {
                seek_outcome(py, "invalid_offset", Some(("offset", offset)))
            }
            Err(err) => Err(err),
        };
    }
    // ":" in offset_str — the containment protocol on the right operand.
    if offset.contains(":")? {
        // parts = offset_str.split(":") — outside the reference's try,
        // so a raising split keeps its own error.
        let parts = offset.getattr("split")?.call1((":",))?;
        // The try body: len(parts) (inside it, exactly like the
        // reference), then the int() parses in the reference's
        // evaluation order with Python's own arithmetic. Any
        // ValueError inside maps to the invalid-time verdict; the
        // helper's `?` returns land in the match below, not straight
        // out of this function.
        let total = match (|| -> PyResult<Option<Bound<'py, PyAny>>> {
            match parts.len()? {
                2 => {
                    // int(parts[0]) * 60 + int(parts[1])
                    let first = builtin_int(py, &parts.get_item(0)?)?;
                    let head = first.mul(60i64)?;
                    let second = builtin_int(py, &parts.get_item(1)?)?;
                    Ok(Some(head.add(&second)?))
                }
                3 => {
                    // int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    let first = builtin_int(py, &parts.get_item(0)?)?;
                    let head = first.mul(3600i64)?;
                    let second = builtin_int(py, &parts.get_item(1)?)?;
                    let mid = head.add(&second.mul(60i64)?)?;
                    let third = builtin_int(py, &parts.get_item(2)?)?;
                    Ok(Some(mid.add(&third)?))
                }
                // The reference's else-return, signalled rather than
                // raised so it cannot be confused with a ValueError.
                _ => Ok(None),
            }
        })() {
            Ok(Some(total)) => total,
            Ok(None) => return seek_outcome(py, "invalid_time", Some(("offset", offset))),
            Err(err) if err.is_instance_of::<PyValueError>(py) => {
                return seek_outcome(py, "invalid_time", Some(("offset", offset)))
            }
            Err(err) => return Err(err),
        };
        // float(total) — outside the reference's try: a total too large
        // for a float keeps its own OverflowError.
        let seconds = builtin_float(py, &total)?;
        return seek_outcome(py, "clock", Some(("seconds", &seconds)));
    }
    // seconds = float(offset_str) — except ValueError
    match builtin_float(py, offset) {
        Ok(seconds) => seek_outcome(py, "absolute", Some(("seconds", &seconds))),
        Err(err) if err.is_instance_of::<PyValueError>(py) => {
            seek_outcome(py, "invalid_offset", Some(("offset", offset)))
        }
        Err(err) => Err(err),
    }
}

// ---------------------------------------------------------------------------
// CLI JSON serialisation (cli.py)
// ---------------------------------------------------------------------------

/// Python's `float.__repr__` — the shortest round-trip digits rendered
/// under CPython's layout rules: fixed notation while the decimal point
/// sits within `1e-4 <= |v| < 1e16`, exponential outside with a sign
/// and at least two exponent digits (`1e+16`, `1.5e-07`), `0.0`/`-0.0`
/// for the zeros, and the specials spelled `NaN` / `Infinity` /
/// `-Infinity` (the `allow_nan=True` spellings). The shortest digits
/// come from Rust's own `{:e}` formatting, which produces the same
/// unique shortest correctly-rounding digit sequence CPython's dtoa
/// does — except where the exact value sits *exactly* between two
/// shortest candidates: those ties round away from zero under Rust's
/// formatter and half-even under CPython's dtoa, and a tie can only
/// carry when the digit run ends odd (a tied pair's even neighbour is
/// the run one digit down), so those — and only those — ask the
/// interpreter's own repr. Every finite double serialises exactly like
/// `repr()` would render it.
fn python_float_repr(py: Python<'_>, v: f64) -> PyResult<String> {
    if v.is_nan() {
        return Ok("NaN".to_string());
    }
    if v.is_infinite() {
        return Ok(if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string());
    }
    if v == 0.0 {
        return Ok(if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_string());
    }
    let neg = v.is_sign_negative();
    // "d[.ddd]e[-]X" — digits and the base-10 exponent of |v|.
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp_part) = sci.split_once('e').expect("exponential spelling");
    let exp: i32 = exp_part.parse().expect("exponent digits");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // An exact midpoint between two shortest candidates rounds away
    // from zero here but half-even under CPython's dtoa — a tie only
    // carries when the digit run ends odd (its even neighbour is the
    // run one digit down), so those — and only those — ask the
    // interpreter's own repr.
    if digits
        .as_bytes()
        .last()
        .is_some_and(|b| (b - b'0') % 2 == 1)
    {
        let repr = PyFloat::new(py, v).repr()?;
        return repr.extract::<String>();
    }
    // decpt — the decimal point's position relative to the digit run
    // (digits "12345", exp 2 -> 123.45, decpt 3).
    let decpt = exp + 1;
    let n = digits.len() as i32;
    let mut out = String::with_capacity(digits.len() + 8);
    if neg {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        // Exponential: d[.ddd]e±XX, at least two exponent digits.
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        let ea = e.unsigned_abs();
        if ea < 10 {
            out.push('0');
        }
        out.push_str(&ea.to_string());
    } else if decpt <= 0 {
        // 0.000ddd
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt >= n {
        // ddd000.0
        out.push_str(&digits);
        for _ in 0..(decpt - n) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        // dd.ddd
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    Ok(out)
}

/// The four-hex `\uXXXX` escape, lowercase digits — CPython's own
/// `Py_hexdigits` table.
fn json_push_u_escape(out: &mut String, c: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('\\');
    out.push('u');
    out.push(HEX[((c >> 12) & 0xf) as usize] as char);
    out.push(HEX[((c >> 8) & 0xf) as usize] as char);
    out.push(HEX[((c >> 4) & 0xf) as usize] as char);
    out.push(HEX[(c & 0xf) as usize] as char);
}

/// The `ensure_ascii` escaping — CPython's `S_CHAR` rule: the printable
/// ASCII run 0x20..=0x7e (bar `"` and `\`) passes through, the seven
/// two-character escapes `\\ \" \b \t \n \f \r` render their short
/// forms, DEL included in everything else becoming a lowercase
/// four-hex `\uXXXX`, and an astral code point its UTF-16 surrogate
/// pair.
fn json_escape_codepoint_into(out: &mut String, c: u32) {
    match c {
        0x22 => out.push_str("\\\""),
        0x5c => out.push_str("\\\\"),
        0x08 => out.push_str("\\b"),
        0x09 => out.push_str("\\t"),
        0x0a => out.push_str("\\n"),
        0x0c => out.push_str("\\f"),
        0x0d => out.push_str("\\r"),
        0x20..=0x7e => out.push(c as u8 as char),
        c if c >= 0x10000 => {
            let v = c - 0x10000;
            json_push_u_escape(out, 0xd800 + (v >> 10));
            json_push_u_escape(out, 0xdc00 + (v & 0x3ff));
        }
        _ => json_push_u_escape(out, c),
    }
}

/// Escape a `&str`'s code points — the fast lane for ASCII runs with
/// nothing to escape, the per-code-point walk otherwise.
fn json_escape_str_into(out: &mut String, s: &str) {
    let plain = s.is_ascii()
        && !s
            .bytes()
            .any(|b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
    out.push('"');
    if plain {
        out.push_str(s);
    } else {
        for c in s.chars() {
            json_escape_codepoint_into(out, c as u32);
        }
    }
    out.push('"');
}

/// Escape one Python string value: the fast lane over its UTF-8
/// content, and — where the string carries lone surrogates, which have
/// no UTF-8 spelling — the WTF-8 bytes of its `surrogatepass` encoding,
/// so a `\ud800` in the payload escapes exactly like the reference's
/// code-point walk would.
fn json_escape_pystring_into(out: &mut String, value: &Bound<'_, PyString>) -> PyResult<()> {
    if let Ok(s) = value.to_str() {
        json_escape_str_into(out, s);
        return Ok(());
    }
    // str.encode("utf-8", "surrogatepass") — the content as WTF-8.
    let bytes = value.call_method1("encode", ("utf-8", "surrogatepass"))?;
    let bytes = bytes
        .downcast::<PyBytes>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("encode did not return bytes"))?;
    let data = bytes.as_bytes();
    out.push('"');
    let mut i = 0usize;
    while i < data.len() {
        // One WTF-8 sequence — plain UTF-8 shapes plus the surrogate
        // range encoded as three-byte sequences.
        let b = data[i];
        let c = if b < 0x80 {
            i += 1;
            b as u32
        } else if b >> 5 == 0b110 {
            i += 2;
            ((b & 0x1f) as u32) << 6 | (data[i - 1] & 0x3f) as u32
        } else if b >> 4 == 0b1110 {
            i += 3;
            ((b & 0x0f) as u32) << 12
                | ((data[i - 2] & 0x3f) as u32) << 6
                | (data[i - 1] & 0x3f) as u32
        } else {
            i += 4;
            ((b & 0x07) as u32) << 18
                | ((data[i - 3] & 0x3f) as u32) << 12
                | ((data[i - 2] & 0x3f) as u32) << 6
                | (data[i - 1] & 0x3f) as u32
        };
        json_escape_codepoint_into(out, c);
    }
    out.push('"');
    Ok(())
}

/// `int.__repr__(value)` — the raw-storage digits, never a subclass's
/// own repr (an IntEnum member renders as its number). i64 covers the
/// whole range Rust can format directly; anything wider goes through
/// the interpreter's own repr.
fn int_repr_string(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    match value.extract::<i64>() {
        Ok(v) => Ok(v.to_string()),
        Err(_) => {
            let repr = py
                .import("builtins")?
                .getattr("int")?
                .getattr("__repr__")?
                .call1((value,))?;
            repr.extract::<String>()
        }
    }
}

/// The key coercion + escaping behind one dict key: `str` keys pass
/// through, `float` keys render through `float.__repr__`, `int` keys
/// through `int.__repr__` (an IntEnum key renders as its numeric
/// digits, never its member name), `True`/`False`/`None` as
/// `true`/`false`/`null`, anything else CPython's own `TypeError`. The
/// coerced text is then escaped like any other string — the digit and
/// repr spellings are all plain ASCII, but the walk stays uniform.
fn write_dict_key_into(out: &mut String, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(s) = key.downcast::<PyString>() {
        json_escape_pystring_into(out, s)?;
        return Ok(());
    }
    let text = if key.is_instance_of::<PyFloat>() {
        python_float_repr(py, key.extract::<f64>()?)?
    } else if key.is_none() {
        "null".to_string()
    } else if key.is_instance_of::<PyBool>() {
        if key.extract::<bool>()? {
            "true"
        } else {
            "false"
        }
        .to_string()
    } else if key.is_instance_of::<PyInt>() {
        int_repr_string(py, key)?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "keys must be str, int, float, bool or None, not {}",
            key.get_type().name()?
        )));
    };
    json_escape_str_into(out, &text);
    Ok(())
}

/// Unpack one two-item iterable — `a, b = item` with CPython's own
/// unpack errors: the not-enough/too-many counts are the items actually
/// yielded, and the "cannot unpack non-iterable" rewrite only applies
/// to a type with no iteration support at all (an `__iter__` that
/// itself raises keeps its own error, like CPython's `tp_iter` check).
/// Serves the JSON encoder's `items()` pairs and the doctor report's
/// crash-file `(path, content)` pair alike.
fn unpack_two<'py>(pair: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let py = pair.py();
    let mut items = match pair.try_iter() {
        Ok(items) => items,
        Err(err) => {
            if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                let type_iter = pair
                    .get_type()
                    .getattr("__iter__")
                    .ok()
                    .filter(|attr| !attr.is_none());
                if type_iter.is_none() {
                    let type_name = pair
                        .get_type()
                        .name()
                        .map(|name| name.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "cannot unpack non-iterable {type_name} object"
                    )));
                }
            }
            return Err(err);
        }
    };
    let key = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 2, got 0)",
            ))
        }
    };
    let value = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 2, got 1)",
            ))
        }
    };
    if items.next().is_some() {
        return Err(PyValueError::new_err(
            "too many values to unpack (expected 2)",
        ));
    }
    Ok((key, value))
}

/// One open frame of the JSON encode — a list/array, a dict/object, or
/// the default-conversion sentinel that holds a converted object's
/// marker until its subtree finishes encoding (CPython's own marker
/// lifetime around the `default()` call). The traversal is iterative,
/// so no Rust recursion can overflow: a payload nested past CPython's
/// own recursion budget encodes where the reference would raise
/// `RecursionError`.
enum JsonFrame<'py> {
    List {
        iter: Bound<'py, PyIterator>,
        first: bool,
        id: usize,
    },
    Dict {
        iter: Bound<'py, PyIterator>,
        first: bool,
        id: usize,
    },
    Default {
        id: usize,
    },
}

/// The `json.dumps(data, indent=..., default=str)` text behind
/// `cli.py`'s `_json_output` — the serialisation every JSON-emitting
/// CLI command prints.
///
/// Mirrors the pure-Python reference (`_json_dumps`) — which is the
/// stdlib call itself — with every other parameter at its default:
/// `ensure_ascii=True` (the `S_CHAR` escaping above), the separators
/// `(', ', ': ')` compact and `(',', ': ')` with the newline+indent
/// layout when *indent* is given (a negative indent collapsing to `""`
/// exactly like the reference's `' ' * n` multiplication; an absurd
/// width raises `MemoryError` rather than building a multi-gigabyte
/// pad), `sort_keys=False` (insertion order preserved), `allow_nan=True`
/// (`NaN`/`Infinity`/`-Infinity`), and `check_circular=True` — the
/// ancestor-chain guard reporting `Circular reference detected` for a
/// cycle while shared non-cyclic references encode on both sides.
///
/// The value dispatch follows the reference's order — the consts, then
/// `str`, `int` (through `int.__repr__`, so an IntEnum member renders
/// as its number and bigints keep their full digits), `float` (through
/// `float.__repr__`, see `python_float_repr`), `list`/`tuple`, `dict`,
/// and then the default call: `str(o)` fetched from `builtins` at call
/// time and invoked exactly like the reference's `default=str` would,
/// its result re-encoded through the same dispatch with the original
/// object's marker held for the conversion's lifetime. Lists iterate
/// through Python's iteration protocol and dicts through their `items`
/// method — the calls both CPython encoders make — so subclasses steer
/// both backends identically; the empty containers render `[]`/`{}`
/// from the truthiness gate before any marker is taken. Dict keys
/// coerce through the reference's rule (see `write_dict_key_into`)
/// with CPython's own `TypeError` for anything else; an `items()` pair
/// that is not a 2-tuple raises the pure-Python fallback's unpack
/// `ValueError`. The inputs are never mutated, and the traversal holds
/// no Rust recursion, so deeply nested payloads encode where the
/// reference's recursive walk would hit the interpreter's recursion
/// budget.
#[pyfunction]
#[pyo3(signature = (data, indent = None))]
fn json_dumps<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indent: Option<i64>,
) -> PyResult<String> {
    if indent.is_some_and(|n| !(-1_000_000..=1_000_000).contains(&n)) {
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "indent is too large",
        ));
    }
    let indent_pad = " ".repeat(indent.map_or(0, |n| n.max(0) as usize));
    let pretty = indent.is_some();

    let mut out = String::new();
    // The ancestor-chain ids behind the circular-reference guard —
    // inserted when a container opens, removed when it closes.
    let mut markers: HashSet<usize> = HashSet::new();
    let mut stack: Vec<JsonFrame<'py>> = Vec::new();
    let mut current: Option<Bound<'py, PyAny>> = Some(data.clone());

    loop {
        if let Some(value) = current.take() {
            // The value dispatch, in the reference's order.
            if value.is_none() {
                out.push_str("null");
            } else if value.is_instance_of::<PyBool>() {
                out.push_str(if value.extract::<bool>()? {
                    "true"
                } else {
                    "false"
                });
            } else if let Ok(s) = value.downcast::<PyString>() {
                json_escape_pystring_into(&mut out, s)?;
            } else if value.is_instance_of::<PyInt>() {
                out.push_str(&int_repr_string(py, &value)?);
            } else if value.is_instance_of::<PyFloat>() {
                out.push_str(&python_float_repr(py, value.extract::<f64>()?)?);
            } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
                // if not lst: '[]' — the truthiness gate before any
                // marker is taken.
                if !value.is_truthy()? {
                    out.push_str("[]");
                } else {
                    let id = value.as_ptr() as usize;
                    if markers.contains(&id) {
                        return Err(PyValueError::new_err("Circular reference detected"));
                    }
                    markers.insert(id);
                    out.push('[');
                    if pretty {
                        out.push('\n');
                        out.push_str(&indent_pad.repeat(stack.len() + 1));
                    }
                    stack.push(JsonFrame::List {
                        iter: value.try_iter()?,
                        first: true,
                        id,
                    });
                }
            } else if value.is_instance_of::<PyDict>() {
                if !value.is_truthy()? {
                    out.push_str("{}");
                } else {
                    let id = value.as_ptr() as usize;
                    if markers.contains(&id) {
                        return Err(PyValueError::new_err("Circular reference detected"));
                    }
                    markers.insert(id);
                    out.push('{');
                    if pretty {
                        out.push('\n');
                        out.push_str(&indent_pad.repeat(stack.len() + 1));
                    }
                    // items() through the mapping protocol — the method
                    // call both CPython encoders make.
                    let items = value.getattr("items")?.call0()?;
                    stack.push(JsonFrame::Dict {
                        iter: items.try_iter()?,
                        first: true,
                        id,
                    });
                }
            } else {
                // default(o) — str, fetched from builtins at call time
                // exactly like the reference's global lookup and called
                // as its default would call it. The sentinel below pops
                // the original object's marker off once the converted
                // value's subtree finishes encoding — CPython's own
                // marker lifetime around the default call.
                let id = value.as_ptr() as usize;
                if markers.contains(&id) {
                    return Err(PyValueError::new_err("Circular reference detected"));
                }
                markers.insert(id);
                let converted = py.import("builtins")?.getattr("str")?.call1((value,))?;
                stack.push(JsonFrame::Default { id });
                current = Some(converted);
            }
            continue;
        }
        // No pending value — pull the next item from the top frame, or
        // close the frame when its iteration is exhausted.
        let depth = stack.len();
        enum Pull<'py> {
            Item(Bound<'py, PyAny>),
            Close { id: usize, closer: Option<u8> },
        }
        let pull = match stack.last_mut() {
            None => break,
            Some(JsonFrame::List { iter, first, id }) => match iter.next() {
                Some(Ok(item)) => {
                    if !*first {
                        if pretty {
                            out.push(',');
                            out.push('\n');
                            out.push_str(&indent_pad.repeat(depth));
                        } else {
                            out.push_str(", ");
                        }
                    }
                    *first = false;
                    Pull::Item(item)
                }
                Some(Err(err)) => return Err(err),
                None => Pull::Close {
                    id: *id,
                    closer: Some(b']'),
                },
            },
            Some(JsonFrame::Dict { iter, first, id }) => match iter.next() {
                Some(Ok(pair)) => {
                    let (key, value) = unpack_two(&pair)?;
                    if !*first {
                        if pretty {
                            out.push(',');
                            out.push('\n');
                            out.push_str(&indent_pad.repeat(depth));
                        } else {
                            out.push_str(", ");
                        }
                    }
                    *first = false;
                    write_dict_key_into(&mut out, py, &key)?;
                    out.push_str(": ");
                    Pull::Item(value)
                }
                Some(Err(err)) => return Err(err),
                None => Pull::Close {
                    id: *id,
                    closer: Some(b'}'),
                },
            },
            Some(JsonFrame::Default { id }) => Pull::Close {
                id: *id,
                closer: None,
            },
        };
        match pull {
            Pull::Item(value) => current = Some(value),
            Pull::Close { id, closer } => {
                if let Some(closer) = closer {
                    if pretty {
                        out.push('\n');
                        out.push_str(&indent_pad.repeat(depth - 1));
                    }
                    out.push(closer as char);
                }
                markers.remove(&id);
                stack.pop();
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Doctor report assembly (utils/doctor.py)
// ---------------------------------------------------------------------------

/// The report assembly behind `gather_diagnostics` from
/// `utils/doctor.py` — the `ytm doctor` text users paste into issues.
///
/// Mirrors the pure-Python reference (`_doctor_report`): the
/// multi-section report built from the probes' results — the banner
/// and the aligned identity labels (`Version:` / `Python:  ` /
/// `OS:      ` / `Machine:`), the paths block with its own alignment
/// (`config:` through `crashes:`), the probe lines appended raw, the
/// empty-log placeholder (a falsy *recent_errors* — the reference's
/// `filtered if filtered else` — substitutes `(no warnings or errors
/// logged)`), the crash-file section (`None` is the no-crash-files
/// marker; a `(path, content)` pair unpacks into the `From:` line, the
/// content, and the staleness note when *crash_note* is not `None` —
/// an empty-string note still lands, `is not None` is not a truthiness
/// gate), the hooks section with its trailing note paragraph, and the
/// final `"\n"` join run through the redaction pass (the same `redact`
/// engine the Python side binds to).
///
/// Every interpolation runs through `str()` exactly like the
/// reference's f-strings — `Path` objects render as their path text,
/// the version triple as its numbers, and anything with a `__str__`
/// through its own. The inputs are never mutated.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the report's 22 slots — one parameter per line the reference appends
#[pyo3(signature = (version, py_major, py_minor, py_micro, system, release, machine, mpv_status, libmpv_status, config_path, theme_path, session_path, log_path, crash_dir, process_status, mpris_status, recent_errors, recent_mpv, faulthandler, crash, crash_note, hooks))]
fn doctor_report<'py>(
    version: &Bound<'py, PyAny>,
    py_major: &Bound<'py, PyAny>,
    py_minor: &Bound<'py, PyAny>,
    py_micro: &Bound<'py, PyAny>,
    system: &Bound<'py, PyAny>,
    release: &Bound<'py, PyAny>,
    machine: &Bound<'py, PyAny>,
    mpv_status: &Bound<'py, PyAny>,
    libmpv_status: &Bound<'py, PyAny>,
    config_path: &Bound<'py, PyAny>,
    theme_path: &Bound<'py, PyAny>,
    session_path: &Bound<'py, PyAny>,
    log_path: &Bound<'py, PyAny>,
    crash_dir: &Bound<'py, PyAny>,
    process_status: &Bound<'py, PyAny>,
    mpris_status: &Bound<'py, PyAny>,
    recent_errors: &Bound<'py, PyAny>,
    recent_mpv: &Bound<'py, PyAny>,
    faulthandler: &Bound<'py, PyAny>,
    crash: &Bound<'py, PyAny>,
    crash_note: &Bound<'py, PyAny>,
    hooks: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let mut sections: Vec<String> = Vec::with_capacity(40);

    sections.push("=== ytm-player diagnostics ===".to_string());
    sections.push(format!("Version: {}", py_str(version)?));
    sections.push(format!(
        "Python:  {}.{}.{}",
        py_str(py_major)?,
        py_str(py_minor)?,
        py_str(py_micro)?
    ));
    sections.push(format!("OS:      {} {}", py_str(system)?, py_str(release)?));
    sections.push(format!("Machine: {}", py_str(machine)?));
    sections.push(py_str(mpv_status)?);
    sections.push(py_str(libmpv_status)?);

    sections.push(String::new());
    sections.push("=== Paths ===".to_string());
    sections.push(format!("config:   {}", py_str(config_path)?));
    sections.push(format!("theme:    {}", py_str(theme_path)?));
    sections.push(format!("session:  {}", py_str(session_path)?));
    sections.push(format!("log:      {}", py_str(log_path)?));
    sections.push(format!("crashes:  {}", py_str(crash_dir)?));

    sections.push(String::new());
    sections.push("=== Process status ===".to_string());
    sections.push(py_str(process_status)?);

    sections.push(String::new());
    sections.push("=== MPRIS / media keys ===".to_string());
    sections.push(py_str(mpris_status)?);

    sections.push(String::new());
    sections.push("=== Recent ERROR/WARNING (last 20) ===".to_string());
    // filtered if filtered else "(no warnings or errors logged)" —
    // truthiness, not a None check.
    let errors_line = if recent_errors.is_truthy()? {
        py_str(recent_errors)?
    } else {
        "(no warnings or errors logged)".to_string()
    };
    sections.push(errors_line);

    sections.push(String::new());
    sections.push("=== Recent mpv warnings/errors ===".to_string());
    sections.push(py_str(recent_mpv)?);

    sections.push(String::new());
    sections.push("=== Most recent faulthandler trace ===".to_string());
    sections.push(py_str(faulthandler)?);

    sections.push(String::new());
    sections.push("=== Most recent crash file ===".to_string());
    if crash.is_none() {
        sections.push("(no crash files found)".to_string());
    } else {
        // path, content = crash — the reference's tuple unpacking.
        let (path, content) = unpack_two(crash)?;
        sections.push(format!("From: {}", py_str(&path)?));
        sections.push(py_str(&content)?);
        // if note is not None — not a truthiness gate: an empty-string
        // note still lands.
        if !crash_note.is_none() {
            sections.push(py_str(crash_note)?);
        }
    }

    sections.push(String::new());
    sections.push("=== Active hooks ===".to_string());
    sections.push(py_str(hooks)?);
    sections.push(String::new());
    let note = "Note: this section reflects the doctor subcommand's own process. ".to_string()
        + "Hooks are installed inside the long-running ytm TUI process — "
        + "their effect is visible above (crashes/, faulthandler.log).";
    sections.push(note);

    // "\n".join(sections) — then the redaction pass, the same engine
    // the Python side's _redact binds to.
    Ok(redact(&sections.join("\n")))
}

// ---------------------------------------------------------------------------
// Doctor probe engines (mirror ytm_player/utils/doctor.py)
// ---------------------------------------------------------------------------

/// The last *n* lines containing `mpv[`, joined together. Mirrors the pure
/// filtering core of `ytm_player.utils.doctor._recent_mpv_lines` (the
/// existence check and the file read stay in Python): the `mpv[` match is
/// a substring test, not a prefix one; lines keep their trailing newlines
/// so the join reproduces them; the empty match list returns the
/// no-warnings marker before any slice is taken; and the tail follows
/// Python's `matches[-n:]` semantics (see `py_tail_start`).
#[pyfunction]
fn recent_mpv_lines(lines: Vec<String>, n: i64) -> String {
    let matches: Vec<&str> = lines
        .iter()
        .map(String::as_str)
        .filter(|ln| ln.contains("mpv["))
        .collect();
    if matches.is_empty() {
        return "(no mpv warnings/errors logged)".to_string();
    }
    matches[py_tail_start(matches.len(), n)..].concat()
}

/// The metadata keys `write_crash_file()` stamps at the top of every crash
/// file (`logging._crash_metadata_header`). Only these count as header
/// lines — a traceback line that happens to be `Word: value` shaped (e.g.
/// `ValueError: x`) must not be mistaken for crash metadata. Mirrors
/// `doctor._CRASH_META_KEYS` (case-sensitive membership).
const CRASH_META_KEYS: [&str; 4] = ["version", "time", "python", "platform"];

/// The `(\w+):\s+(\S+)` line shape of the crash-metadata header, with
/// CPython's `\s`/`\S` set spelled out (`\s` is the str.isspace set —
/// White_Space plus `\x1c`–`\x1f`, which the crate's own `\s` disagrees
/// on; `\S` is its exact complement). The key class stays `\w`: where
/// Python's and the crate's word-char sets differ the divergence provably
/// cannot change the verdict — a key in `CRASH_META_KEYS` is pure ASCII,
/// where both sets agree, and any divergent word run is left without its
/// colon on one side, ending the header scan on both.
static CRASH_META_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(\w+):[\s\x{1c}-\x{1f}]+([^\s\x{1c}-\x{1f}]+)").expect("valid crash meta regex")
});

/// The PEP 440 version grammar from `packaging`, ASCII-scoped and
/// case-insensitive exactly like its `(?a:...)` groups under
/// `re.IGNORECASE`: every word spelling is per-character cased (`AlPhA` is
/// a valid pre-release marker), the local alphabet is `[a-zA-Z0-9]` — the
/// `(?a:)` scoping rejects the Unicode folds (`ſ` U+017F, the Kelvin sign
/// U+212A) the crate's own case-insensitive mode would admit — and the
/// wrapping whitespace is CPython's `\s` (see `CRASH_META_RE`). Anchored at
/// both ends like the reference's `fullmatch`, with group names mirroring
/// packaging's own.
static PEP440_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^[\s\x{1c}-\x{1f}]*[vV]?",
        r"(?:",
        r"(?:(?P<epoch>[0-9]+)!)?",
        r"(?P<release>[0-9]+(?:\.[0-9]+)*)",
        r"(?:[._-]?(?P<pre_l>[aA][lL][pP][hH][aA]|[aA]|[bB][eE][tT][aA]|[bB]",
        r"|[pP][rR][eE][vV][iI][eE][wW]|[pP][rR][eE]|[cC]|[rR][cC])",
        r"[._-]?(?P<pre_n>[0-9]+)?)?",
        r"(?:-(?P<post_n1>[0-9]+)|[._-]?(?P<post_l>[pP][oO][sS][tT]|[rR][eE][vV]|[rR])",
        r"[._-]?(?P<post_n2>[0-9]+)?)?",
        r"(?:[._-]?(?P<dev_l>[dD][eE][vV])[._-]?(?P<dev_n>[0-9]+)?)?",
        r")",
        r"(?:\+(?P<local>[a-zA-Z0-9]+(?:[._-][a-zA-Z0-9]+)*))?",
        r"[\s\x{1c}-\x{1f}]*$",
    ))
    .expect("valid pep440 regex")
});

/// CPython's default `sys.int_max_str_digits`: the length above which the
/// int(str) conversion raises a plain ValueError. `packaging` parses
/// version components through int(), so a numeric component longer than
/// this is a version the reference cannot compare — the engine enforces
/// the same default limit so both backends agree on the no-verdict
/// outcome for such strings.
const INT_MAX_STR_DIGITS: usize = 4300;

/// A non-negative integer of unbounded width, held as its normalized
/// decimal digits (leading zeros stripped; the empty string is zero).
/// PEP 440 epochs and release segments can exceed every fixed-width
/// integer, so the comparison is numeric on the digit run itself: the
/// longer run is the larger number, equal lengths compare digit by digit.
#[derive(PartialEq, Eq)]
struct BigNum(String);

impl BigNum {
    fn zero() -> Self {
        BigNum(String::new())
    }

    /// Normalize a matched digit run, or None when it exceeds CPython's
    /// int-conversion digit limit (see `INT_MAX_STR_DIGITS`).
    fn from_digits(digits: &str) -> Option<Self> {
        if digits.len() > INT_MAX_STR_DIGITS {
            return None;
        }
        Some(BigNum(digits.trim_start_matches('0').to_string()))
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
}

impl Ord for BigNum {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_empty(), other.0.is_empty()) {
            // Both zero.
            (true, true) => std::cmp::Ordering::Equal,
            // Zero is smaller than any other number.
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => self
                .0
                .len()
                .cmp(&other.0.len())
                .then_with(|| self.0.cmp(&other.0)),
        }
    }
}

impl PartialOrd for BigNum {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The pre-release rank from packaging's `_PRE_RANK`: a=0, b=1, rc=2 (the
/// marker spellings alpha/beta/c/pre/preview all normalise onto these).
fn pep440_pre_rank(letter: &str) -> i8 {
    match letter.to_lowercase().as_str() {
        "a" => 0,
        "b" => 1,
        _ => 2,
    }
}

/// The flat pre/post/dev comparison suffix from packaging's `_cmpkey`,
/// compared as the tuple it is: (pre_rank, pre_n, post_rank, post_n,
/// dev_rank, dev_n).
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Pep440Suffix {
    pre_rank: i8,
    pre_n: BigNum,
    post_rank: u8,
    post_n: BigNum,
    dev_rank: u8,
    dev_n: BigNum,
}

/// One local-version segment. Mirrors packaging's `(n, "")` /
/// `(NegativeInfinity, s)` segment pairs: a string segment sorts before
/// every numeric segment, strings compare lexicographically and numbers
/// numerically.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Pep440LocalSeg {
    Str(String),
    Num(BigNum),
}

/// The PEP 440 comparison key, in packaging's `_cmpkey` order: epoch,
/// release (trailing zeros stripped, so `1.0.0 == 1`), the pre/post/dev
/// suffix, then the local segments — absent sorting before present, and
/// the segment list comparing as the tuple it is.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Pep440Key {
    epoch: BigNum,
    release: Vec<BigNum>,
    suffix: Pep440Suffix,
    local: Option<Vec<Pep440LocalSeg>>,
}

/// Parse a version string into its PEP 440 comparison key, or None when
/// the string is not a valid version — or one of its numbers exceeds
/// CPython's int-conversion digit limit (see `INT_MAX_STR_DIGITS`).
fn parse_pep440(text: &str) -> Option<Pep440Key> {
    let caps = PEP440_RE.captures(text)?;

    let epoch = BigNum::from_digits(caps.name("epoch").map(|m| m.as_str()).unwrap_or("0"))?;
    let mut release: Vec<BigNum> = caps
        .name("release")?
        .as_str()
        .split('.')
        .map(BigNum::from_digits)
        .collect::<Option<Vec<_>>>()?;
    // Strip trailing zeros: 1.0.0 compares equal to 1.
    while release.last().is_some_and(BigNum::is_zero) {
        release.pop();
    }

    // Each numeric group is Option<Option<BigNum>>: the outer Option is
    // "did the group match", the inner is the digit-limit-checked number.
    // A matched group whose digits exceed the limit invalidates the whole
    // version (the reference's int() raises through), an unmatched one
    // simply carries no number.
    let pre_l = caps.name("pre_l").map(|m| m.as_str());
    let pre_n = caps.name("pre_n").map(|m| BigNum::from_digits(m.as_str()));
    let post_n1 = caps
        .name("post_n1")
        .map(|m| BigNum::from_digits(m.as_str()));
    let post_l = caps.name("post_l").map(|m| m.as_str());
    let post_n2 = caps
        .name("post_n2")
        .map(|m| BigNum::from_digits(m.as_str()));
    let dev_l = caps.name("dev_l").map(|m| m.as_str());
    let dev_n = caps.name("dev_n").map(|m| BigNum::from_digits(m.as_str()));

    let post_matched = post_n1.is_some() || post_l.is_some();
    let dev_matched = dev_l.is_some();

    // The pre rank: a dev-only release (no pre, no post, dev present)
    // sorts before every alpha; a release without a pre sorts after every
    // rc; otherwise the marker's rank with its number (an implicit 0
    // when the digits are absent).
    let (pre_rank, pre_num) = if let Some(letter) = pre_l {
        let number = match pre_n {
            Some(number) => number?,
            None => BigNum::zero(),
        };
        (pep440_pre_rank(letter), number)
    } else if !post_matched && dev_matched {
        (-1, BigNum::zero())
    } else {
        (3, BigNum::zero())
    };

    // post: `-N` (post_n1) and `postN`/`post`/`rev`/`r` (post_l+post_n2)
    // both normalise to a number, again with the implicit 0.
    let post_num = if post_matched {
        match post_n1.or(post_n2) {
            Some(number) => number?,
            None => BigNum::zero(),
        }
    } else {
        BigNum::zero()
    };

    let dev_num = match dev_n {
        Some(number) => number?,
        None => BigNum::zero(),
    };

    let suffix = Pep440Suffix {
        pre_rank,
        pre_n: pre_num,
        post_rank: u8::from(post_matched),
        post_n: post_num,
        dev_rank: u8::from(!dev_matched),
        dev_n: dev_num,
    };

    let local = match caps.name("local") {
        None => None,
        Some(m) => Some(
            m.as_str()
                .split(['.', '_', '-'])
                .map(|part| {
                    if part.bytes().all(|b| b.is_ascii_digit()) {
                        // int(part): a digit run beyond the int-conversion
                        // limit invalidates the version.
                        BigNum::from_digits(part).map(Pep440LocalSeg::Num)
                    } else {
                        Some(Pep440LocalSeg::Str(part.to_lowercase()))
                    }
                })
                .collect::<Option<Vec<Pep440LocalSeg>>>()?,
        ),
    };

    Some(Pep440Key {
        epoch,
        release,
        suffix,
        local,
    })
}

/// Split exactly like CPython's `str.splitlines`: on every line boundary
/// it accepts (`\n`, `\r\n` as one, `\r`, `\v`, `\f`, `\x1c`, `\x1d`,
/// `\x1e`, `\x85`, `\u{2028}`, `\u{2029}`) — a wider set than the
/// crate's own `str::splitlines` — with a trailing boundary producing no
/// trailing empty line, and an empty string producing no lines at all.
fn py_splitlines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '\n' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => lines.push(std::mem::take(&mut current)),
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                lines.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines
}

/// Warn when the most recent crash predates the running build. Mirrors
/// `ytm_player.utils.doctor._crash_staleness_note`: only the contiguous
/// known `key: value` metadata header just after the `=== label ===`
/// banner is parsed (a blank line ends it; an unknown or non-key-shaped
/// line ends it — so a traceback's `ValueError: x` is never mistaken for
/// metadata, and a `version:` line buried in the body is never read); a
/// missing version line earns the predates-stamping note; the `unknown`
/// sentinel earns none; and otherwise a PEP 440 comparison decides
/// between the stale-build note and no note. Any version that fails to
/// parse — including a numeric component longer than CPython's
/// int-conversion digit limit — earns no note.
#[pyfunction]
fn crash_staleness_note(content: &str, current_version: &str) -> Option<String> {
    let lines = py_splitlines(content);
    let start = usize::from(lines.first().is_some_and(|l| l.starts_with("===")));
    let mut crash_version: Option<&str> = None;
    for line in &lines[start..] {
        // A blank line (nothing but CPython whitespace) ends the header.
        if line.chars().all(py_isspace) {
            break;
        }
        let meta = CRASH_META_RE.captures(line).and_then(|caps| {
            let key = caps.get(1)?.as_str();
            let value = caps.get(2)?.as_str();
            Some((key, value))
        });
        let Some((key, value)) = meta else {
            break; // not a key: value line -> the header is over
        };
        if !CRASH_META_KEYS.contains(&key) {
            break; // not a known header key (e.g. a traceback line)
        }
        if key == "version" {
            crash_version = Some(value);
            break;
        }
    }

    let Some(crash_version) = crash_version else {
        return Some(
            "⚠ This crash file predates crash-version stamping (no version recorded). If it's \
             from an older build it may already be fixed — reproduce on the current version \
             before reporting."
                .to_string(),
        );
    };
    if crash_version == "unknown" {
        return None;
    }
    let (crash_key, current_key) = (parse_pep440(crash_version), parse_pep440(current_version));
    if crash_key? < current_key? {
        return Some(format!(
            "⚠ This crash was recorded by ytm-player {crash_version}, older than the installed \
             {current_version} — it may already be fixed. Reproduce on the current version \
             before reporting."
        ));
    }
    None
}
