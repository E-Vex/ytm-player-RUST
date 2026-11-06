use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::{
    PyImportError, PyIndexError, PyKeyError, PyOSError, PyRecursionError, PyTypeError,
    PyUnicodeDecodeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;
use pyo3::types::{
    PyBool, PyBytes, PyDict, PyFloat, PyInt, PyIterator, PyList, PySet, PySlice, PyString, PyTuple,
};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};
use unicode_bidi::bidi_class;

/// Regex for validating YouTube video IDs.
static VALID_VIDEO_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9_-]{1,64}$").expect("valid video id regex"));

/// A YouTube channel ID — `UC` + 22 URL-safe base64 characters, the
/// reference's `re.fullmatch` anchored at both ends (a trailing newline
/// cannot match the class, so the anchored form is the fullmatch).
static CHANNEL_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^UC[A-Za-z0-9_-]{22}$").expect("valid channel id regex"));

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
    // The chart-region table (the same module's CHART_REGIONS): the
    // 68 (code, name) pairs — ZZ first, then alphabetical by name —
    // behind the region picker, the charts queries and the
    // shelf-title country suffix strip.
    module.add_function(pyo3::wrap_pyfunction!(chart_regions, module)?)?;

    // BiDi text utilities (ytm_player.utils.bidi equivalents).
    module.add_function(pyo3::wrap_pyfunction!(has_rtl, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(isolate_bidi, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(reorder_line, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(wrap_line, module)?)?;
    // The gated public wrappers behind the render path — the hot
    // per-fragment line reorder and the lyrics sidebar's multi-line
    // pre-wrap, each with the empty/no-RTL passthrough and the
    // bidi_mode verdict folded into one native call (the cached
    // settings read stays Python-side, passed in as a bool).
    module.add_function(pyo3::wrap_pyfunction!(reorder_rtl_line, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(wrap_rtl_line, module)?)?;
    // The fused per-cell render engines — the whole
    // reorder/truncate/isolate (and wrap/isolate) chains the render
    // sites used to assemble through three separate calls, folded into
    // one native call per rendered cell or lyric line (the cached
    // settings read stays Python-side, passed in as a bool).
    module.add_function(pyo3::wrap_pyfunction!(render_track_cell, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(render_bar_cell, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(render_lyric_line, module)?)?;

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
    // The configured-extras layering (the same module's
    // apply_configured_yt_dlp_options core): the opts mutation pass
    // behind StreamResolver._build_ydl_opts' second half — the
    // cookiefile write, the CA bundle's missing-file warning and
    // no-certifi compat_opts conversion, and the remote_components /
    // js_runtimes writes — with the file probe and the logger.warning
    // call supplied as thunks, invoked exactly where the reference
    // performed them, and the settings reads and their normalisations
    // staying Python-side in the caller.
    module.add_function(pyo3::wrap_pyfunction!(yt_dlp_configured_opts, module)?)?;

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
    // Track-table row-assembly engines (the same module's equivalents):
    // the per-row (row_key, cells) composition behind _add_track_row —
    // the mark glyph, the original-position number, the BiDi-isolated
    // title/artist/album cells and the duration text — and the fused
    // view engine behind every table rebuild (load_tracks' row walk
    // and _rebuild_view + _reload_sorted's filter/sort/assembly chain,
    // folded into one call answering the positions and the composed
    // rows together), with the add_row calls, the cursor/scroll
    // preservation and the playing-row highlight staying Python-side
    // in the widget, and the cached bidi_mode verdict read once per
    // rebuild Python-side.
    module.add_function(pyo3::wrap_pyfunction!(track_table_row_values, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(track_table_view_rows, module)?)?;

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

    // Search-page label glue (ui/pages/search.py equivalent): the
    // item-to-label composer behind the Albums/Artists/Playlists panels
    // of the search results area, the bottom-of-panel count label, the
    // mode indicator beside the input, the suggestion-overlay pick
    // decision, the recent-searches label pass, and the _search_all
    // categorization bucketing one unfiltered search into the four
    // panels.
    module.add_function(pyo3::wrap_pyfunction!(format_search_result, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_panel_count_label, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_mode_label, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_suggestion_pick, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(recent_search_labels, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(categorize_search_results, module)?)?;

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
    // similar-artists bar), the see-all section selection behind the
    // discography expansion, and the artist-songs worker merge's
    // decision core — the first-batch plan and the remaining-fetch
    // decision behind `_fetch_full_artist_songs`'s swap of the artist's
    // full song list into the top-songs table, plus the tail plan for
    // the chained remaining-tracks fetch.
    module.add_function(pyo3::wrap_pyfunction!(album_header, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(playlist_header, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(artist_discography, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(expandable_sections, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(artist_songs_first_batch, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(artist_songs_tail, module)?)?;

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

    // CLI history-query statement engines (cli.py equivalents): the
    // two statement texts behind the standalone `ytm history` and
    // `ytm history search` reads — every column, newest first,
    // bounded by the caller's limit, each with the limit passed
    // through as the object it is into the parameter tuple. The
    // connect/row-factory/execute/fetch glue and the error exits
    // stay Python-side in _query_history.
    module.add_function(pyo3::wrap_pyfunction!(cli_history_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cli_search_history_select, module)?)?;

    // CLI cache-status engines (cli.py equivalents): the six-key status
    // payload behind the standalone `ytm cache status --json` command,
    // the four aligned human lines it echoes without --json — both from
    // the cache directory and the walk's three aggregates — and the
    // directory walk that produces them, with the exists/iterdir/
    // is_file/stat operations supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(cache_status_payload, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cache_status_text, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cache_status_walk, module)?)?;

    // CLI cache-clear walk (cli.py equivalent): the directory walk
    // behind the standalone `ytm cache clear` command — which entries
    // of the cache directory are unlinked and how many were, with the
    // exists/iterdir/is_file/unlink operations supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(cache_clear_walk, module)?)?;

    // CLI config editor resolution (cli.py equivalent): the plan behind
    // the standalone `ytm config` command — which editor opens the
    // config file, whether Windows shell-opens it instead, and the
    // no-EDITOR no-xdg-open manual-open error.
    module.add_function(pyo3::wrap_pyfunction!(config_editor_plan, module)?)?;

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

    // IPC request gate (ipc.py _ipc_request_gate equivalent): the
    // key-exchange body behind IPCServer._client_connected — the TCP
    // transport's token gate (the presented token compared through the
    // module's own secrets.compare_digest name over both sides' utf-8
    // bytes) and the command whitelist (the containment protocol over
    // the module's own _VALID_COMMANDS frozenset), with the args
    // coercion the dispatch runs with. The framing, the JSON parse,
    // the handler await and the response writes stay Python-side in
    // the server.
    module.add_function(pyo3::wrap_pyfunction!(ipc_request_gate, module)?)?;

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
    // mpv-warning tail behind the Recent mpv warnings/errors probe, the
    // `Fatal Python error:` tail extraction behind the Most recent
    // faulthandler trace probe, and the crash-staleness verdict behind
    // the Most recent crash file section — the metadata-header scan, the
    // PEP 440 comparison of the recorded version against the installed
    // one, and the note assembly.
    module.add_function(pyo3::wrap_pyfunction!(recent_mpv_lines, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(faulthandler_tail, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(crash_staleness_note, module)?)?;

    // Doctor /proc and subprocess probe engines (utils/doctor.py
    // equivalents): the Process status section's ytm-running verdict —
    // the /proc walk, the cmdline decode and entry-point match, the
    // VmRSS parse, the own-pid exclusion — and the `mpv:` version line
    // behind the mpv status probe, both with their side-effecting
    // probes (/proc reads, shutil.which, the --version subprocess run)
    // supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(running_status_text, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(mpv_version_text, module)?)?;

    // Doctor import-probe engines (utils/doctor.py equivalents): the
    // libmpv loadability verdict — the player-module import and the
    // last-line render of its failure message — and the MPRIS
    // availability verdict — the platform gate, the system render and
    // the mpris-module import's BUS_NAME/DBUS_AVAILABLE pair — with the
    // imports themselves supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(libmpv_status_text, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(mpris_status_text, module)?)?;

    // Theme resolution engines (ui/theme.py and app/_app.py equivalents):
    // the Rich-safe ANSI token translation, the 21-field ThemeColors
    // resolution from CSS variables behind from_css_variables, the
    // app-variable overlay + theme.toml override merge behind
    // get_css_variables, and the 21-field rebuild from a Textual Theme's
    // raw colours behind watch_theme — the three sites of the documented
    // accent → primary → DEFAULT_LYRIC_CURRENT lyric-current chain.
    module.add_function(pyo3::wrap_pyfunction!(rich_safe_color, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(theme_fields_from_css, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(app_css_variables, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(theme_fields_from_theme, module)?)?;

    // Spotify-import matching cores (services/spotify_import.py
    // equivalents): the playlist/album ID pull behind the import URL
    // parse, and the spotipy item normaliser behind every track the
    // Web-API path ingests.
    module.add_function(pyo3::wrap_pyfunction!(extract_playlist_id, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(parse_spotipy_item, module)?)?;

    // Settings TOML value formatting (config/settings.py equivalent):
    // the value-to-TOML-literal renderer behind every line
    // Settings.save writes.
    module.add_function(pyo3::wrap_pyfunction!(format_toml_value, module)?)?;

    // Keymap load plan (config/keymap.py _load_from_dict equivalent):
    // the section walk, action validation and binding classification
    // behind keymap.toml loading, applied by the Python caller as it
    // replays the plan.
    module.add_function(pyo3::wrap_pyfunction!(keymap_load_plan, module)?)?;

    // Actions-popup menu builder (ui/popups/actions.py equivalent): the
    // context-menu assembly behind every track/album/artist/playlist
    // action menu — the per-type catalog walk with the queue-membership
    // swap, the playlist-view and search-source gates, and the
    // like/subscribe label flips.
    module.add_function(pyo3::wrap_pyfunction!(build_actions, module)?)?;

    // Progress-bar render geometry (ui/widgets/progress_bar.py
    // equivalents): the progress fraction, the time-prefix/suffix/
    // bar-width metrics, the filled/empty/marker cell computation behind
    // every bar render tick, and the click-to-seek column mapping.
    module.add_function(pyo3::wrap_pyfunction!(progress_fraction, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(progress_bar_metrics, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(progress_bar_cells, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(progress_x_to_seconds, module)?)?;

    // Synced-lyrics LRC parser (ui/sidebars/lyrics_sidebar.py
    // equivalent): the timestamp extraction behind every string-lyrics
    // load — the [mm:ss(.fff)] match at each line's start, the
    // fractional-second conversion, the timestamp stripping and the
    // stable timestamp sort.
    module.add_function(pyo3::wrap_pyfunction!(parse_synced_lyrics, module)?)?;
    // LRCLIB lookup glue (services/lrclib.py equivalent): the
    // query-params assembly behind get_synced_lyrics' request build —
    // the track_name/artist_name pair in the reference's insertion
    // order plus the identity-gated duration through the interpreter's
    // own int() and str() — and the syncedLyrics extraction behind its
    // parse step — the mapping's own get plus the `or None`
    // truthiness gate. The title sanitizer is the already-ported
    // formatting engine; the requests fetch and the resp.json() decode
    // stay Python-side.
    module.add_function(pyo3::wrap_pyfunction!(lrclib_query_params, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(lrclib_synced_lyrics, module)?)?;

    // Update-check version comparison (services/update_check.py
    // equivalent): the strictly-newer PEP 440 verdict behind
    // check_for_update's cache-hit and post-fetch decisions, on the same
    // packaging grammar shape the installed release compiled (the fold
    // flag the Python side probes once and passes down).
    module.add_function(pyo3::wrap_pyfunction!(pep440_is_newer, module)?)?;

    // Help-page rows engine (ui/pages/help.py equivalents): the
    // keybinding-reference row assembly (the action-value title-casing
    // through CPython's own str.title semantics, the key-sequence join,
    // the "(unbound)" placeholder) and the filtered walk (the
    // lowercased-then-stripped needle, the haystack containment, the
    // category-header insertion) behind the searchable help table.
    module.add_function(pyo3::wrap_pyfunction!(help_page_rows, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(help_page_visible_rows, module)?)?;

    // Playlist-picker view model (ui/popups/playlist_picker.py
    // equivalents): the recent-list move-to-front bookkeeping behind
    // _record_recent, the recency sort behind _sort_by_recent, the
    // filtered visible-row walk behind _populate_list and the
    // playlist-count status label.
    module.add_function(pyo3::wrap_pyfunction!(playlist_record_recent, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(playlist_picker_sort, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(playlist_picker_rows, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(playlist_picker_count_label, module)?)?;

    // Queue-page view & mutation engines (ui/pages/queue.py
    // equivalents): the now-playing header view model shared by
    // _update_current_track and _refresh_queue, the radio-seed source
    // label behind _update_queue_source, the footer text behind
    // _update_footer, and the optimistic-mutation decisions behind
    // _remove_selected and _move_track (the stale-table queue-position
    // refusal, the post-remove cursor re-landing, and the clamped
    // count-multiplied reorder destination).
    module.add_function(pyo3::wrap_pyfunction!(queue_now_playing, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(queue_source_view, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(queue_footer_text, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(queue_remove_target, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(queue_post_remove_cursor, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(queue_move_target, module)?)?;

    // Album-art render engines (ui/widgets/album_art.py equivalents): the
    // WCAG colour math behind the placeholder's note glyph (the relative
    // luminance of a six-digit hex colour, the contrast ratio, and the
    // highest-contrast candidate picker with the first-wins tie rule), the
    // half-block cell extraction from the resized thumbnail's raw RGB
    // buffer behind every image render, and the placeholder/empty
    // row-segment plans behind the widget's two no-image renders — with
    // the Pillow decode, the LANCZOS resize, the download and the Rich
    // Text assembly staying Python-side next to the I/O.
    module.add_function(pyo3::wrap_pyfunction!(wcag_relative_luminance, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(wcag_contrast_ratio, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(wcag_readable_on, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(album_art_half_block_cells, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(album_art_placeholder_rows, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(album_art_empty_rows, module)?)?;

    // Shuffle-preferences LRU engine (services/shuffle_prefs.py
    // equivalent): the ordered {context_id: shuffle} map — the get
    // promotion, the set with its move_to_end and evict-while-over walk,
    // the clear, the hand-edited-file ingestion (the dict gate, the
    // str-key filter, the bool(v) truthiness coercion and the for k, v
    // unpacking contract), and the lock-held snapshot +
    // json.dumps(..., indent=2) serialisation behind _save — with the
    // thread lock and the logging staying Python-side in the facade.
    module.add_class::<ShufflePrefsCore>()?;
    // The prefs file's disk I/O cores (also services/shuffle_prefs.py):
    // the read behind _load — the path's own read_text driven from
    // native, the JSON parse through the ported CPython-grammar scanner
    // with the interpreter's own json.loads rendering the exact
    // rejection — and the atomic write behind _save — the parent mkdir,
    // the pathlib with_suffix tmp sibling, the UTF-8 write and the
    // os.replace cutover, each driven through the path object's own
    // protocol so every error shape stays CPython's.
    module.add_function(pyo3::wrap_pyfunction!(shuffle_prefs_read_file, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(shuffle_prefs_write_file, module)?)?;

    // Album-art cache LRU engine (ui/widgets/album_art.py equivalent):
    // the ordered URL→Text map behind the module-level art cache — the
    // hit promotion of the cache-check path, the insert that keeps an
    // existing key's position (OrderedDict __setitem__ semantics — this
    // engine never promotes on insert) with its single-pop overflow
    // eviction, and the guarded delete of the resize invalidation, the
    // renders themselves held as the Python objects they are.
    module.add_class::<ArtCacheCore>()?;

    // Mutation-failure classifier (services/ytmusic.py equivalent): the
    // exception-to-kind mapping behind every mutation call's except arm
    // — the reference's isinstance chain in its exact order (KeyError,
    // the two timeout classes, requests' RequestException,
    // YTMusicUserError, YTMusicServerError) with the classes resolved
    // from the running interpreter at call time, and the
    // YTMusicServerError arm's HTTP-status prefix match (CPython's \d
    // and \b semantics on str patterns, Unicode digits and their int()
    // values included) deciding auth_expired from server_error.
    module.add_function(pyo3::wrap_pyfunction!(classify_mutation_failure, module)?)?;

    // YTMusic retry/renewal bookkeeping engines (services/ytmusic.py
    // equivalents): the auth-expiry sniff behind _is_auth_expired (the
    // gate every expired-session retry consults before renewing — the
    // YTMusicServerError arm delegating to the module's own classifier,
    // the KeyError arm's sign-in marker sniff over the lowered
    // message), the renewal-cooldown gate and stamp behind
    // _renewed_client_for (the remaining-cooldown arithmetic and the
    // failed-renewal stamp, both against the module's own clock and
    // cooldown constant), and the consecutive-failure step behind
    // _run's except arm (the counter bump and the reinit threshold
    // verdict against the module's own cap) — with the locks, the
    // renewal itself, the client rebuilds and the logging staying
    // Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(ytm_auth_expired, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_renewal_cooldown_gate, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_renewal_cooldown_stamp, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_api_failure_step, module)?)?;

    // YTMusic browse-wrapper engines (services/ytmusic.py equivalents):
    // the read wrappers' pure cores — the watch-playlist kwargs
    // assembly behind get_watch_playlist, the tracks extraction the
    // liked-songs / watch-playlist / radio-fetch / playlist-radio and
    // trending/liked-mix sites share, the list-gated playlist-tracks
    // extraction behind get_chart_shelf_tracks, the deduplicated radio
    // seed pool behind get_radio, the six-source rotation behind
    // get_discovery_mix, the chart-shelf assembly and its round-robin
    // shelf pick behind the charts mix, the two playable filters
    // behind the discovery mixes, the explore new-releases extraction
    // behind get_new_releases, and the sort-order params lookup
    // behind get_playlist — with the client calls, the patch-window
    // gate, the renewal machinery and the RNG-driven shuffle/sample
    // staying Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(ytm_watch_playlist_kwargs, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_result_tracks, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_playlist_tracks, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_radio_seed_pool, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_discovery_source_order, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_chart_shelves, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_chart_shelf_index, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ytm_discovery_playable_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_chart_playable_tracks, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_explore_new_releases, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(ytm_playlist_order_params, module)?)?;

    // Playback-bar track-info width engine (ui/playback_bar.py
    // equivalent): the halves-and-thirds width split behind the
    // one-line title/artist/album render — the max(10, width - 30)
    // budget, the min(len, //2) and min(len, //3) fragment widths, and
    // the remainder-minus-separators album width with its max(0, ...)
    // clamp — with the Rich Text assembly, the BiDi reorder/isolate
    // calls and the state icon staying Python-side in the widget.
    module.add_function(pyo3::wrap_pyfunction!(track_info_widths, module)?)?;

    // Audio-cache LRU eviction engine (services/cache.py equivalent):
    // the eviction walk behind CacheManager.evict — which entries fall
    // out of an over-budget cache, LRU-first, the budget re-checked
    // before each entry (the entry that brings the running total
    // exactly to the limit is the last evicted) and the total
    // decremented through the reference's `total -= file_size`
    // augmented assignment — answering the (video_id, file_path,
    // file_size) triples to unlink in eviction order, with the SUM
    // read, the file unlinking, the index DELETE and the commit
    // staying Python-side in the manager.
    module.add_function(pyo3::wrap_pyfunction!(eviction_plan, module)?)?;

    // Audio-cache maintenance engines (services/cache.py equivalents):
    // the statement bodies and the bookkeeping around the eviction
    // walk — the schema DDL script behind init's executescript, the
    // hit-counter batched-commit gate behind get's last_accessed
    // UPDATE (the module's own _COMMIT_EVERY read at call time, a
    // patched constant included), the four-key usage summary behind
    // get_status with the MB → bytes budget and the cache-dir
    // str(), the INSERT OR REPLACE statement and its four-value
    // bindings behind _index, the plan-triples → IN-clause DELETE
    // behind evict's index cleanup, and the wipe's bare DELETE
    // behind clear — with the connection handling, the row reads,
    // the file unlinking and the commits staying Python-side in the
    // manager.
    module.add_function(pyo3::wrap_pyfunction!(cache_schema, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cache_hit_commit_plan, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cache_status_summary, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        cache_index_insert_statement,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        cache_index_delete_statement,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        cache_index_delete_all_statement,
        module
    )?)?;

    // History row-assembly peripherals (services/history.py
    // equivalents): the remaining pure cores around the four ported
    // history engines — the autocomplete string list behind
    // get_search_suggestions ([row["query"] for row in rows]), the
    // played-video-id set behind get_played_video_ids
    // ({row["video_id"] for row in rows} — CPython set-insertion
    // semantics: hash-first, identity-or-== membership, the first of
    // equal values the one kept), and the two parameter tuples
    // log_play binds (the seven-value play_history INSERT tuple and
    // the four-value play_stats upsert tuple, each read through the
    // reference's left-to-right subscript order) — with the SQL
    // itself staying Python-side in the manager.
    module.add_function(pyo3::wrap_pyfunction!(suggestion_strings, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(played_video_id_set, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_insert_params, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_stats_params, module)?)?;

    // Auth session-record engines (services/auth.py equivalents): the
    // trust machinery around auth.json and account.json — the record
    // parser (the CPython JSON grammar over the decoded bytes, its
    // rejections the engine's None, past-budget nesting the same
    // RecursionError the reference raises), the recorded-identity
    // validation (the False-singleton gate, the channel-id fullmatch,
    // the Unicode-digit slot through the interpreter's own int()), the
    // ownership snapshot (the auth hash and the record's revision
    // bound to exactly those bytes), the stream-jar vouching decision
    // tree, the account_menu channel-id extraction (the defensive walk
    // that fails closed), the canonical header serialisation with its
    // candidate-slot variant, the record assembly with its fresh
    // revision, and the SHA-256 hex primitive; the manual-paste header
    // normalization (the ^[E/alternating/standard paste formats, the
    // Chrome decoded-protobuf block stripping and the pseudo-header
    // drops); and the stream cookiejar builder (the youtube.com/
    // google.com domain filter with its control-char gate, and the
    // yt-dlp Netscape serialisation with the jar's own cookie
    // ordering) — with the file reads, the session locking and the
    // atomic writes staying Python-side in the manager.
    module.add_function(pyo3::wrap_pyfunction!(parse_record, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(identity_from_record, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(owner_of, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(vouch_failure, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        channel_id_from_account_menu,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(serialize_headers, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(candidate_payload, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(build_record, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(sha256_hex, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(normalize_raw_headers, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(build_stream_cookiejar, module)?)?;

    // Region-picker filter engine (ui/popups/country_picker.py
    // equivalent): the case-insensitive code-or-name substring pass
    // behind the chart-region picker's filter input — the query
    // stripped and lowercased, an empty query passing the region list
    // through unchanged (the same object the reference hands back),
    // and every kept row re-packed as a fresh (code, name) pair of the
    // same string objects.
    module.add_function(pyo3::wrap_pyfunction!(filter_regions, module)?)?;

    // Terminal orientation gate (utils/terminal.py equivalent): the
    // aspect-ratio verdict behind the horizontal/vertical layout
    // decision — the rows-zero guard, the true division and the 2.3
    // threshold all through the interpreter's own operators, so every
    // numeric shape and every error matches the reference exactly.
    module.add_function(pyo3::wrap_pyfunction!(get_orientation, module)?)?;

    // History SQL-body engines (services/history.py equivalents): the
    // nineteen statement texts and parameter tuples behind the
    // HistoryManager's reads and writes — the schema DDL script and
    // the retention prune behind init, the search-history upsert /
    // selects / delete, the two log_play writes' statement texts
    // (their parameter tuples are the play_insert_params /
    // play_stats_params engines), the listened-seconds read and its
    // two updates, the play-history and recently-played selects, the
    // played-video-id read, the three aggregate reads behind
    // get_stats, and the two top-list selects. Each engine returns
    // the statement exactly as the fallback spells it (leading
    // newline and indentation included) with the bound values passed
    // through as the objects they are; the connection handling, the
    // executescript/execute/commit calls and the fetches stay
    // Python-side in the manager.
    module.add_function(pyo3::wrap_pyfunction!(history_schema, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(history_retention_delete, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_upsert, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_history_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_suggestion_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(search_history_delete, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_history_insert_sql, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_stats_upsert_sql, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_listened_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_listened_update, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_stats_listened_update, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(play_history_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(recently_played_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(played_video_ids_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(stats_total_plays_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        stats_total_listen_time_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(stats_unique_tracks_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(top_tracks_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(top_artists_select, module)?)?;

    // Stream resolver-glue engines (services/stream.py equivalents): the
    // yt-dlp resolver's pure cores — the expiry gate and the available_at
    // wait gate (both reading the module's own `time` name, a frozen or
    // patched clock included), the quality-input gate behind the
    // resolver's setter, the base yt-dlp options dict behind
    // _build_ydl_opts, the extract_info → StreamInfo field assembly
    // behind _try_resolve, the DownloadError hint, and the cache
    // maintenance walks — the expired-entry prune behind _put_cache
    // and prune_expired, and the over-cap eviction of the oldest
    // entries by expires_at behind _put_cache's second phase (the cap
    // read through the module's own _CACHE_MAX_SIZE name, a patched
    // cap included). The YoutubeDL construction, the cookiejar
    // vouching, the network extraction, the cache state and the
    // pending-future bookkeeping stay Python-side in the resolver.
    module.add_function(pyo3::wrap_pyfunction!(is_stale, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(wait_seconds, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(validate_quality, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(stream_base_opts, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(stream_info_fields, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(download_error_hint, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(stream_cache_expired_ids, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(stream_cache_cap_evictions, module)?)?;

    // Stream yt-dlp instance-gate engines (services/stream.py
    // equivalents): the acquisition loop around the extract_info
    // sites' remaining pure cores — the cookiejar-signature reuse
    // gate behind _get_ydl's locked reuse arm (the cached instance's
    // baked signature against the signature the resolver should be
    // using right now, through CPython's own ==) and the mid-build
    // publication verdict behind _get_ydl's second locked section
    // (adopt another thread's instance, retry after a mid-build
    // reset, or install the built one). The signature probe, the
    // locks, the YoutubeDL construction and the _active_resolves
    // accounting stay Python-side in the resolver.
    module.add_function(pyo3::wrap_pyfunction!(
        stream_ydl_cookiejar_current,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(stream_ydl_publish_verdict, module)?)?;

    // Download-service engines (services/download.py equivalents): the
    // base yt-dlp options dict behind DownloadService._build_opts (the
    // format selector, the staged outtmpl, the quiet flags and the
    // FFmpeg-audio postprocessor) and the five probe filenames a
    // completed download is looked up under (the extension order
    // behind _download_sync's publication walk and get_path's lookup).
    // The staging, the yt-dlp invocation, the atomic publication and
    // the thread ownership stay Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(download_base_opts, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(download_candidates, module)?)?;

    // Player glue engines (services/player.py equivalents): the
    // Homebrew libmpv discovery walk behind the import-time brew-prefix
    // probe (the pattern pair through sys.platform, the prefix list
    // through the module's own _BREW_LIB_DIRS name, the
    // glob/sorted/first-hit walk with its per-prefix OSError swallow),
    // the end-file verdict behind _end_file_data's event matching (the
    // instance staleness, the loading park, the entry/track/reason
    // match and the reason fork), the position-dispatch throttle behind
    // _on_time_pos_change (the module clock minus the last dispatch
    // against the 0.4 window), the seek clamp behind
    // _clamp_seek_target, and the command-supersession gate behind
    // _dispatch's still-current check and _finish_load's acceptance.
    // The locks, the pending-event buffer, the payload assembly, the
    // dispatch and the mpv calls stay Python-side in the player.
    module.add_function(pyo3::wrap_pyfunction!(player_find_libmpv, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(player_end_file_verdict, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(player_clamp_seek, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(player_clamp_volume, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(player_position_due, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(player_command_current, module)?)?;
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

/// The chart-region table behind the region picker, the charts queries
/// and the shelf-title country suffix strip — the `CHART_REGIONS`
/// constant of `ytm_player.services.regions`.
///
/// The 68 entries in the reference's exact order: `ZZ` (Global) first
/// as the default, then alphabetical by display name — the 62 codes
/// YouTube's `get_charts` `countries.options` advertises plus the six
/// historically-supported extras (Hong Kong, Malaysia, Singapore,
/// Taiwan, Thailand, Vietnam). The engine answers the pairs as a list
/// of `(code, name)` string tuples; the module wraps it into the
/// tuple-of-tuples shape the constant spells.
#[pyfunction]
fn chart_regions() -> Vec<(String, String)> {
    const REGIONS: &[(&str, &str)] = &[
        ("ZZ", "Global"),
        ("AR", "Argentina"),
        ("AU", "Australia"),
        ("AT", "Austria"),
        ("BE", "Belgium"),
        ("BO", "Bolivia"),
        ("BR", "Brazil"),
        ("CA", "Canada"),
        ("CL", "Chile"),
        ("CO", "Colombia"),
        ("CR", "Costa Rica"),
        ("CZ", "Czechia"),
        ("DK", "Denmark"),
        ("DO", "Dominican Republic"),
        ("EC", "Ecuador"),
        ("EG", "Egypt"),
        ("SV", "El Salvador"),
        ("EE", "Estonia"),
        ("FI", "Finland"),
        ("FR", "France"),
        ("DE", "Germany"),
        ("GT", "Guatemala"),
        ("HN", "Honduras"),
        ("HK", "Hong Kong"),
        ("HU", "Hungary"),
        ("IS", "Iceland"),
        ("IN", "India"),
        ("ID", "Indonesia"),
        ("IE", "Ireland"),
        ("IL", "Israel"),
        ("IT", "Italy"),
        ("JP", "Japan"),
        ("KE", "Kenya"),
        ("LU", "Luxembourg"),
        ("MY", "Malaysia"),
        ("MX", "Mexico"),
        ("NL", "Netherlands"),
        ("NZ", "New Zealand"),
        ("NI", "Nicaragua"),
        ("NG", "Nigeria"),
        ("NO", "Norway"),
        ("PA", "Panama"),
        ("PY", "Paraguay"),
        ("PE", "Peru"),
        ("PL", "Poland"),
        ("PT", "Portugal"),
        ("RO", "Romania"),
        ("RU", "Russia"),
        ("SA", "Saudi Arabia"),
        ("RS", "Serbia"),
        ("SG", "Singapore"),
        ("ZA", "South Africa"),
        ("KR", "South Korea"),
        ("ES", "Spain"),
        ("SE", "Sweden"),
        ("CH", "Switzerland"),
        ("TW", "Taiwan"),
        ("TZ", "Tanzania"),
        ("TH", "Thailand"),
        ("TR", "Turkey"),
        ("UG", "Uganda"),
        ("UA", "Ukraine"),
        ("AE", "United Arab Emirates"),
        ("GB", "United Kingdom"),
        ("US", "United States"),
        ("UY", "Uruguay"),
        ("VN", "Vietnam"),
        ("ZW", "Zimbabwe"),
    ];
    REGIONS
        .iter()
        .map(|(code, name)| (code.to_string(), name.to_string()))
        .collect()
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
    isolate_bidi_impl(text, only_if_rtl)
}

/// The isolate wrap shared by the `isolate_bidi` export and the fused
/// per-cell render engines — one definition so the export and every
/// render chain isolates through exactly the same body: the empty and
/// pure-LTR (when *only_if_rtl*) passthroughs, the FSI...PDI
/// double-wrap guard, and the wrap itself.
fn isolate_bidi_impl(text: &str, only_if_rtl: bool) -> String {
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

/// The wrap-then-reorder pipeline behind both wrap engines — the
/// reorder-enabled body of `bidi.wrap_rtl_line`: the
/// `len(text) <= width or width <= 0` short-circuit answering the bare
/// line reorder, the whitespace word walk packing `needed <= width`
/// runs (`needed` = the word alone for a fresh line, the running length
/// plus one plus the word otherwise — the reference's `+ 1 +` join
/// budget), an over-wide word starting its own line with the pending
/// run flushed first, and the join with every line reordered through
/// `_do_reorder` exactly like the reference's generator join. Called
/// only after the caller's empty/no-RTL gate.
fn wrap_line_impl(text: &str, width: i64) -> String {
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

/// Pre-wrap RTL text for multi-line display with reordering applied.
/// Raw version — the `bidi_mode` gate lives in Python.  Mirrors
/// `bidi.wrap_rtl_line` *as if reordering were enabled*.
#[pyfunction]
fn wrap_line(text: &str, width: i64) -> String {
    if text.is_empty() || !text.chars().any(is_rtl_char) {
        return text.to_string();
    }
    wrap_line_impl(text, width)
}

/// The gated line reorder shared by the `reorder_rtl_line` export and
/// the fused per-cell render engines: the disabled flag (the cached
/// `bidi_mode` verdict, read Python-side), empty text and text without
/// any RTL script character all answer *text* unchanged — the
/// passthrough that keeps `bidi_mode="auto"` and `"passthrough"` free
/// of any transformation; otherwise the UAX #9 L2 word-level reversal
/// of `do_reorder`.
fn gated_reorder(text: &str, reorder_enabled: bool) -> String {
    if !reorder_enabled || text.is_empty() || !text.chars().any(is_rtl_char) {
        return text.to_string();
    }
    do_reorder(text)
}

/// The gated line reorder every rendered RTL fragment funnels through —
/// the public `reorder_rtl_line` of `bidi.py` with the `bidi_mode` gate
/// folded in. The track-table cells, the playback-bar fragments and the
/// lyric lines all pass through here once per render.
///
/// Mirrors the pure-Python reference: empty text, text without any RTL
/// script character (the `has_rtl` scan) and a disabled reorder flag
/// (the cached `bidi_mode` verdict, read Python-side so user settings
/// keep steering both backends) all answer *text* unchanged — the
/// passthrough that keeps `bidi_mode="auto"` and `"passthrough"`
/// free of any transformation; otherwise the UAX #9 L2 word-level
/// reversal of `_do_reorder`. The gate order differs from the
/// reference's statement order (the flag read before the RTL scan) but
/// no observable behaviour does: neither test has a side effect, and
/// every gate answers the same *text* back.
#[pyfunction]
#[pyo3(signature = (text, reorder_enabled))]
fn reorder_rtl_line(text: &str, reorder_enabled: bool) -> String {
    gated_reorder(text, reorder_enabled)
}

/// The gated multi-line pre-wrap behind the lyrics sidebar's line
/// rendering — the public `wrap_rtl_line` of `bidi.py` with the
/// `bidi_mode` gate folded in.
///
/// Mirrors the pure-Python reference: the same empty/no-RTL/disabled
/// passthrough as the line reorder, then the wrap-then-reorder pipeline
/// of `wrap_line_impl` (the width short-circuit, the word walk, the
/// per-line reorder join). The reorder-enabled flag is read Python-side
/// (the cached `bidi_mode` verdict) and travels in as a plain bool, so
/// one native call renders a whole gated wrap.
#[pyfunction]
#[pyo3(signature = (text, width, reorder_enabled))]
fn wrap_rtl_line(text: &str, width: i64, reorder_enabled: bool) -> String {
    if !reorder_enabled || text.is_empty() || !text.chars().any(is_rtl_char) {
        return text.to_string();
    }
    wrap_line_impl(text, width)
}

/// The fused per-cell engine behind the track-table render sites —
/// `isolate_bidi(reorder_rtl_line(text))`, the chain `_add_track_row`
/// and `_highlight_playing`'s restore-cells assembled per
/// title/artist/album cell, folded into one native call.
///
/// Mirrors the chain exactly: the gated line reorder (the disabled /
/// empty / no-RTL passthroughs, then the UAX #9 L2 word-level
/// reversal), then the isolate wrap with its own empty / pure-LTR /
/// already-wrapped passthroughs — the wrap the chain applies AFTER the
/// reorder so the FSI/PDI pair brackets the words the terminal's BiDi
/// resolver sees. Only the cached `bidi_mode` verdict stays
/// Python-side, travelling in as a plain bool.
#[pyfunction]
#[pyo3(signature = (text, reorder_enabled))]
fn render_track_cell(text: &str, reorder_enabled: bool) -> String {
    isolate_bidi_impl(&gated_reorder(text, reorder_enabled), true)
}

/// The fused per-cell engine behind the playback bar's track-info row —
/// `isolate_bidi(truncate(reorder_rtl_line(text), width))`, the chain
/// `_TrackInfo.render` assembled per title/artist/album fragment,
/// folded into one native call.
///
/// Mirrors the chain exactly and in the chain's own order: the gated
/// line reorder, then `formatting.truncate` over the REORDERED text
/// (the reference truncates the reordered fragment — the ellipsis
/// lands at the display edge, and a `width < 1` answers the empty
/// string), then the isolate wrap — applied after the truncate so the
/// closing PDI is never cut off by the width budget. Only the cached
/// `bidi_mode` verdict stays Python-side, travelling in as a plain
/// bool; the width-splitting engine (`track_info_widths`) stays its
/// own call so the reference's `len()`-based split keeps steering both
/// backends.
#[pyfunction]
#[pyo3(signature = (text, width, reorder_enabled))]
fn render_bar_cell(text: &str, width: isize, reorder_enabled: bool) -> String {
    isolate_bidi_impl(
        &truncate_str(&gated_reorder(text, reorder_enabled), width),
        true,
    )
}

/// The fused line engine behind the lyrics sidebar's two views —
/// `isolate_bidi(wrap_rtl_line(text, width))`, the chain
/// `_build_synced_view` and `_build_unsynced_view` assembled per lyric
/// line, folded into one native call.
///
/// Mirrors the chain exactly: the gated multi-line pre-wrap (the same
/// disabled / empty / no-RTL passthroughs, then the wrap-then-reorder
/// pipeline of `wrap_line_impl` — the width short-circuit, the word
/// walk, the per-line reorder join), then the isolate wrap around the
/// WHOLE wrapped string — one FSI...PDI pair bracketing every wrapped
/// line, exactly where the reference's outer call puts it. Only the
/// cached `bidi_mode` verdict stays Python-side, travelling in as a
/// plain bool.
#[pyfunction]
#[pyo3(signature = (text, width, reorder_enabled))]
fn render_lyric_line(text: &str, width: i64, reorder_enabled: bool) -> String {
    let wrapped = if !reorder_enabled || text.is_empty() || !text.chars().any(is_rtl_char) {
        text.to_string()
    } else {
        wrap_line_impl(text, width)
    };
    isolate_bidi_impl(&wrapped, true)
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

/// The configured-extras layering — `yt_dlp_options.py`'s
/// `apply_configured_yt_dlp_options` core, the second half of
/// `StreamResolver._build_ydl_opts`'s opts build (the first half is
/// `stream_base_opts`).
///
/// Mirrors the reference's mutation sequence branch for branch, over
/// the four already-normalised setting values the caller read in the
/// reference's own order (`normalize_cookiefile` /
/// `normalize_cafile` / `normalize_remote_components` /
/// `normalize_js_runtimes` — the path expanders stay Python-side;
/// the latter two are themselves native engines):
///
/// - a truthy *cookies_file* (CPython's own truthiness protocol)
///   writes `opts["cookiefile"]` through the subscript protocol, the
///   value the very object it is;
/// - a truthy *ca_bundle* first probes the file through the
///   zero-argument *is_file* thunk — invoked exactly where the
///   reference's `Path(ca_bundle).is_file()` ran, so a falsy bundle
///   never probes and a raising probe propagates before any mutation
///   — and warns through the *warn* thunk when it came back falsy,
///   the reference's exact `("Configured [yt_dlp] ca_bundle does not
///   exist: %s", ca_bundle)` argument pair; then the `compat_opts`
///   read through the mapping's own `.get` (a fresh empty list as the
///   default) copied through the interpreter's own `list()` — so the
///   stored list object is never mutated — the `"no-certifi"` gate
///   through CPython's own containment, the survivor appended through
///   the list's own `.append`, and the result written back through
///   `set(compat)` — the interpreter's own set constructor, so an
///   unhashable member raises the constructor's own `TypeError` at
///   exactly the reference's point (after the missing-file warning,
///   which the thunk timing preserves);
/// - a truthy *remote_components* warns through the *warn* thunk with
///   the reference's exact argument pair — the second argument the
///   `", "`-join through the str's own `.join` method — then writes
///   `opts["remote_components"]` as the very object it is;
/// - a truthy *js_runtimes* writes `opts["js_runtimes"]` as the very
///   object it is.
///
/// The *warn* thunk is `logger.warning` itself in production, so the
/// warnings fire at exactly the reference's points in the sequence —
/// interleaved with the mutations, not replayed after them. The
/// settings-object attribute reads and their normalisations stay
/// Python-side in `apply_configured_yt_dlp_options`, evaluated up
/// front in the reference's own read order; their interleaving with
/// the mutation sequence is not observable (the reads are
/// side-effect-free dataclass field accesses). The answer is *opts*
/// itself, the object it is.
#[allow(clippy::too_many_arguments)] // the reference's four extras, plus the two thunk seams
#[pyfunction]
#[pyo3(signature = (opts, cookies_file, ca_bundle, is_file, warn, remote_components, js_runtimes))]
fn yt_dlp_configured_opts<'py>(
    py: Python<'py>,
    opts: &Bound<'py, PyAny>,
    cookies_file: &Bound<'py, PyAny>,
    ca_bundle: &Bound<'py, PyAny>,
    is_file: &Bound<'py, PyAny>,
    warn: &Bound<'py, PyAny>,
    remote_components: &Bound<'py, PyAny>,
    js_runtimes: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if cookies_file.is_truthy()? {
        opts.set_item("cookiefile", cookies_file)?;
    }

    if ca_bundle.is_truthy()? {
        // if not Path(ca_bundle).is_file(): logger.warning(...)
        let file_ok = is_file.call0()?;
        if !file_ok.is_truthy()? {
            warn.call1((
                "Configured [yt_dlp] ca_bundle does not exist: %s",
                ca_bundle,
            ))?;
        }
        // compat = list(opts.get("compat_opts", []))
        let empty = PyList::empty(py).into_any();
        let raw_compat = opts.call_method1("get", ("compat_opts", empty))?;
        let list_type = py.import("builtins")?.getattr("list")?;
        let compat = list_type.call1((raw_compat,))?;
        // if "no-certifi" not in compat: compat.append("no-certifi")
        if !compat.contains("no-certifi")? {
            compat.call_method1("append", ("no-certifi",))?;
        }
        // opts["compat_opts"] = set(compat)
        let set_type = py.import("builtins")?.getattr("set")?;
        let compat_set = set_type.call1((compat,))?;
        opts.set_item("compat_opts", compat_set)?;
    }

    if remote_components.is_truthy()? {
        // logger.warning("...: %s", ", ".join(remote_components))
        let joiner = PyString::new(py, ", ");
        let joined = joiner.call_method1("join", (remote_components,))?;
        warn.call1((
            "yt-dlp remote_components is enabled; this allows remote JavaScript component downloads: %s",
            joined,
        ))?;
        opts.set_item("remote_components", remote_components)?;
    }

    if js_runtimes.is_truthy()? {
        opts.set_item("js_runtimes", js_runtimes)?;
    }

    Ok(opts.clone())
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
        return Err(PyValueError::new_err(unpack_too_many_message(
            py, &matched, 2,
        )));
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

/// CPython's UNPACK_SEQUENCE "too many values" message for *obj* being
/// unpacked into *expected* names. 3.14 reports the actual count, but
/// only through the exact-type fast path — an exact list, tuple or dict
/// whose size exceeds the arity; every other iterable (a subclass, a
/// generator, a custom sequence) and every earlier interpreter keeps
/// the sizeless spelling. Mirrors the error arm of `unpack_iterable`
/// in Python/ceval.c, so the native engines raise byte-identical
/// messages on every interpreter the abi3 wheel runs on.
fn unpack_too_many_message<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    expected: usize,
) -> String {
    let reported = if py.version_info() < (3, 14) {
        None
    } else if let Ok(list) = obj.downcast_exact::<PyList>() {
        Some(list.len())
    } else if let Ok(tuple) = obj.downcast_exact::<PyTuple>() {
        Some(tuple.len())
    } else if let Ok(dict) = obj.downcast_exact::<PyDict>() {
        Some(dict.len())
    } else {
        None
    };
    match reported {
        Some(got) if got > expected => {
            format!("too many values to unpack (expected {expected}, got {got})")
        }
        _ => format!("too many values to unpack (expected {expected})"),
    }
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
        return Err(PyValueError::new_err(unpack_too_many_message(py, row, 3)));
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

    /// The fused radio append — `QueueManager.set_radio_tracks`'s
    /// filter-then-add orchestration folded into one core operation:
    /// the `filter_not_in_queue` dedup (the video_id set of the queued
    /// tracks, the candidates read through their own `get`) and, when
    /// anything survives it, the `add_multiple` append — under the one
    /// lock section the facade already holds, with no Python list
    /// crossing the boundary in between. Returns how many tracks were
    /// appended; 0 leaves the queue untouched (no entry ids minted, no
    /// shuffle rebuild, no removed-current bookkeeping), matching the
    /// reference's `if new_tracks:` gate. Duplicates WITHIN *candidates*
    /// pass exactly like the reference's two-step compose — the dedup
    /// only consults the tracks already queued when the filter ran.
    fn add_radio_tracks(&mut self, py: Python, candidates: Vec<Py<PyAny>>) -> PyResult<usize> {
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
        let mut fresh: Vec<Py<PyAny>> = Vec::new();
        for t in candidates {
            let vid = t.bind(py).call_method1("get", ("video_id",))?;
            if !set.contains(vid)? {
                fresh.push(t);
            }
        }
        let added = fresh.len();
        if added > 0 {
            self.add_multiple_unlocked(py, fresh)?;
        }
        Ok(added)
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
fn track_table_row_values<'py>(
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
fn track_table_view_rows(
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

/// The bottom-of-panel count label behind `SearchResultPanel.load_items`
/// from `ui/pages/search.py`: `"No results"` for an empty panel,
/// `"<count> result"` for exactly one and `"<count> results"` from two
/// on (the `!= 1` plural rule).
#[pyfunction]
fn search_panel_count_label(count: i64) -> String {
    if count == 0 {
        return "No results".to_string();
    }
    let plural = if count != 1 { "s" } else { "" };
    format!("{count} result{plural}")
}

/// The search-mode indicator label — the `\u{25b6} Music` / `\u{25b6} All`
/// display beside the search input, wrapped in the theme colour markup.
///
/// Mirrors the inline f-string shared by `SearchPage.compose`,
/// `SearchPage.on_mount` and `SearchPage._toggle_search_mode` from
/// `ui/pages/search.py`: `"music"` renders the music-only label,
/// anything else the all-results one.
#[pyfunction]
fn search_mode_label(mode: &str, color: &str) -> String {
    if mode == "music" {
        format!("[{color}]\u{25b6}[/{color}] Music")
    } else {
        format!("[{color}]\u{25b6}[/{color}] All")
    }
}

/// The suggestion-overlay pick decision behind the suggestion-list's
/// `ListView.Selected` handling from `ui/pages/search.py`: a `None`
/// index (no row highlighted) or an index outside the suggestion list
/// — a stale selection arriving after the overlay was repopulated —
/// answers `None`; a valid index answers that suggestion, the raw
/// stored row object.
#[pyfunction]
#[pyo3(signature = (suggestions, index))]
fn search_suggestion_pick<'py>(
    suggestions: &Bound<'py, PyList>,
    index: Option<i64>,
) -> Option<Bound<'py, PyAny>> {
    // idx is not None and 0 <= idx < len(suggestions)
    let index = index?;
    if index < 0 || index >= suggestions.len() as i64 {
        return None;
    }
    Some(
        suggestions
            .get_item(index as usize)
            .expect("index bounds-checked above"),
    )
}

/// The recent-searches label pass behind `SearchPage._load_recent_searches`
/// from `ui/pages/search.py`: each history row becomes its suggestion
/// label — the row's `query` value, raw — and an empty history reads as
/// an empty list so the caller skips the overlay update.
///
/// `entry["query"]` resolves through the full subscription protocol,
/// so a dict subclass's `__getitem__` steers exactly like the
/// reference. Where the reference would raise `KeyError` — a row
/// without a `query` key (unreachable through the history service's
/// SQL-materialised rows, which always carry the column) — the engine
/// stays total and skips the row instead.
#[pyfunction]
fn recent_search_labels<'py>(
    py: Python<'py>,
    entries: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let labels = PyList::empty(py);
    for entry in entries.iter() {
        match entry.get_item("query") {
            Ok(query) => labels.append(query)?,
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyKeyError>(py) => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(labels)
}

/// The `_search_all` categorization — the bucketing of one unfiltered
/// search into the results area's four panels, from
/// `SearchPage._search_all` in `ui/pages/search.py`.
///
/// Each item's `resultType` (defaulting to `""`, the nested default
/// evaluated before the outer lookup exactly like the reference)
/// picks its bucket through Python `==` semantics (see `eq_any_str`):
/// `song` and `video` go to `songs`, `album` to `albums`, `artist` to
/// `artists` and `playlist` to `playlists`; everything else (the
/// empty default included) is dropped. The returned dict always
/// carries the four buckets — the empty ones as empty lists — in the
/// reference's insertion order, and each bucket holds the item
/// objects themselves in arrival order. The input list is never
/// mutated.
///
/// Field reads resolve exactly like the reference's `.get` calls, so
/// dict subclasses steer the bucketing like they steer the reference.
/// Where the reference would raise — a `.get`-less item
/// (`AttributeError`) — the engine stays total: the item reads as
/// type-less and is dropped, the same stance the label engine takes.
#[pyfunction]
fn categorize_search_results<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyDict>> {
    let songs = PyList::empty(py);
    let albums = PyList::empty(py);
    let artists = PyList::empty(py);
    let playlists = PyList::empty(py);
    let empty = PyString::new(py, "").into_any();

    for item in raw.iter() {
        // result_type = item.get("resultType", "")
        let result_type = field_get(py, &item, "resultType", Some(empty.clone()))?;
        if eq_any_str(&result_type, &["song", "video"])? {
            songs.append(&item)?;
        } else if eq_any_str(&result_type, &["album"])? {
            albums.append(&item)?;
        } else if eq_any_str(&result_type, &["artist"])? {
            artists.append(&item)?;
        } else if eq_any_str(&result_type, &["playlist"])? {
            playlists.append(&item)?;
        }
    }

    let results = PyDict::new(py);
    results.set_item("songs", songs)?;
    results.set_item("albums", albums)?;
    results.set_item("artists", artists)?;
    results.set_item("playlists", playlists)?;
    Ok(results)
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

/// The closed first-batch plan — nothing to load, no tail to chain —
/// for the worker's early-return gates (an empty raw batch, a batch
/// normalization dropped everything from) and for the shapes the
/// reference could not read at all (a non-dict `data`, a truthy
/// non-list `tracks`).
fn artist_songs_closed_plan(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("tracks", PyList::empty(py))?;
    out.set_item("load", false)?;
    out.set_item("fetch_remaining", false)?;
    out.set_item("raw_count", 0)?;
    Ok(out)
}

/// The first-batch plan behind the context-page worker merge — the pure
/// decision core of the first half of
/// `ContextPage._fetch_full_artist_songs` from `ui/pages/context.py`,
/// the background fetch that swaps the artist's full song list (the
/// playlist behind `songs.browseId`, fetched because ytmusicapi
/// truncates artist top songs) into the top-songs table.
///
/// Mirrors the reference:
///
/// ```python
/// raw_tracks = data.get("tracks", []) if isinstance(data, dict) else []
/// if not raw_tracks: return
/// full_tracks = normalize_tracks(raw_tracks)
/// if not full_tracks: return
/// ```
///
/// Returned keys:
///
/// - `tracks` — `normalize_tracks(raw_tracks)`, the batch to swap into
///   the top-songs table; `[]` whenever the swap gate below is closed
///   (the caller leaves the existing truncated list in place).
/// - `load` — the swap gate: the batch only loads when the raw list is
///   non-empty AND normalization kept something (an all-unplayable
///   batch swaps nothing in).
/// - `fetch_remaining` — the remaining-fetch decision, the merge's
///   core: the chained `get_playlist_remaining` tail fetch runs only
///   when the batch loaded AND the stored `trackCount` does not
///   already cover it — a missing `trackCount` (or an explicit `None`)
///   falls back to attempting the call, with empty/error there reading
///   as "nothing more to fetch"; a count at or below the raw batch
///   length closes the gate.
/// - `raw_count` — `len(raw_tracks)`, the `already_have` the tail
///   fetch resumes from; 0 whenever the tail gate is closed.
///
/// Field reads resolve exactly like the reference's `.get` calls, so
/// dict subclasses steer the plan like they steer the reference. Where
/// the reference would raise — a truthy non-list `tracks` (its
/// `normalize_tracks` iteration) or a stored `trackCount` Python's
/// `<=` would reject (`TypeError`, both unreachable through the
/// service's dict-shaped, int-counted responses) — the engine stays
/// total and reads the batch as closed, the same stance the header
/// engines take.
#[pyfunction]
fn artist_songs_first_batch<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // raw_tracks = data.get("tracks", []) if isinstance(data, dict) else []
    let raw_tracks = if data.is_instance_of::<PyDict>() {
        let empty = PyList::empty(py).into_any();
        field_get(py, data, "tracks", Some(empty))?
    } else {
        PyList::empty(py).into_any()
    };
    // if not raw_tracks: return — no swap, no tail fetch.
    if !raw_tracks.is_truthy()? {
        return artist_songs_closed_plan(py);
    }
    let (tracks, raw_len) = normalized_tracks_and_len(py, &raw_tracks)?;
    // if not full_tracks: return — same gate one step later. A truthy
    // non-list raw (where the reference's normalize_tracks would raise
    // in the iteration) reads as no tracks here.
    let raw_count = match raw_len {
        Some(len) if !tracks.is_empty() => len,
        _ => return artist_songs_closed_plan(py),
    };

    // The remaining-fetch decision:
    //   total_count = data.get("trackCount")
    //   if total_count is not None and total_count <= len(raw_tracks):
    //       return
    // — a `None` (missing or explicit) falls back to attempting the
    // call; any comparable count above the batch length chains the
    // tail; anything else closes the gate. `data` is a dict or dict
    // subclass on every path that reaches here (a non-dict read as an
    // empty batch above), so the `trackCount` read steers exactly like
    // the reference's.
    let total = field_get(py, data, "trackCount", None)?;
    let fetch_remaining = if total.is_none() {
        true
    } else {
        // numeric_gt(total, raw_count) == Some(true) is exactly
        // `total > raw_count` — the one comparable outcome whose `<=`
        // reads False and the tail chains. Some(false) is
        // `total <= raw_count` (gate closed), and an uncomparable
        // value — where the reference's `<=` would raise TypeError —
        // closes the gate too, the header engines' stance.
        numeric_gt(&total, raw_count) == Some(true)
    };

    let out = PyDict::new(py);
    out.set_item("tracks", tracks)?;
    out.set_item("load", true)?;
    out.set_item("fetch_remaining", fetch_remaining)?;
    out.set_item("raw_count", raw_count)?;
    Ok(out)
}

/// The tail plan behind the context-page worker merge — the pure
/// decision core of the second half of
/// `ContextPage._fetch_full_artist_songs` from `ui/pages/context.py`:
/// what the chained `get_playlist_remaining` fetch appends to the
/// top-songs table.
///
/// Mirrors the reference:
///
/// ```python
/// if not remaining: return
/// remaining_tracks = normalize_tracks(remaining)
/// if not remaining_tracks: return
/// ```
///
/// Returned keys:
///
/// - `tracks` — `normalize_tracks(remaining)`, the tail rows to
///   append; `[]` whenever the append gate below is closed.
/// - `append` — the append gate: the tail only lands when the fetched
///   remaining list is non-empty AND normalization kept something (an
///   all-unplayable tail is dropped, the batch stays as it loaded).
///
/// A truthy non-list `remaining` (where the reference's
/// `normalize_tracks` iteration would raise — unreachable through
/// `get_playlist_remaining`, which slices a list) reads as no rows —
/// the engine stays total, the same stance the first-batch engine
/// takes.
#[pyfunction]
fn artist_songs_tail<'py>(
    py: Python<'py>,
    remaining: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // if not remaining: return — nothing to append.
    if !remaining.is_truthy()? {
        let out = PyDict::new(py);
        out.set_item("tracks", PyList::empty(py))?;
        out.set_item("append", false)?;
        return Ok(out);
    }
    let (tracks, _) = normalized_tracks_and_len(py, remaining)?;
    let append = !tracks.is_empty();
    let out = PyDict::new(py);
    out.set_item("tracks", tracks)?;
    out.set_item("append", append)?;
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

/// The play-history select behind the CLI's standalone `ytm history`
/// command — the pure half of the history read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cli_history_select`): the
/// statement text exactly as the fallback spells it — every column,
/// newest play first, bounded by the caller's limit — and `limit`
/// passed through as the object it is into the parameter tuple, packed
/// as the `(sql, params)` pair the reference's `_query_history` hands
/// `execute`.
#[pyfunction]
#[pyo3(signature = (limit))]
fn cli_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "SELECT * FROM play_history ORDER BY played_at DESC LIMIT ?".to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The search-history select behind the CLI's standalone
/// `ytm history search` command — the pure half of the search-history
/// read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cli_search_history_select`):
/// the statement text exactly as the fallback spells it — every
/// column, most recently searched first, bounded by the caller's
/// limit — and `limit` passed through as the object it is into the
/// parameter tuple, packed as the `(sql, params)` pair the reference's
/// `_query_history` hands `execute`.
#[pyfunction]
#[pyo3(signature = (limit))]
fn cli_search_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "SELECT * FROM search_history ORDER BY last_searched DESC LIMIT ?".to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The status payload behind the CLI's standalone `ytm cache status`
/// command — the pure core of the status read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cache_status_payload`): the
/// six-key dict assembled from the cache directory and the walk's
/// three aggregates, in the reference's insertion order (the key order
/// the JSON output preserves — `cache_dir`, `file_count`,
/// `total_bytes`, `total_human`, `db_size_bytes`, `db_size_human`).
/// `str(cache_dir)` renders the directory exactly like the reference's
/// dict literal (a `Path` as its path text); `file_count`,
/// `total_bytes` and `db_size` pass through as the objects they are —
/// the count and the sums straight from the directory walk — and the
/// two human spellings come from the same `format_size` engine the
/// Python side's `from …formatting import format_size` resolves to
/// when the extension is built, applied through the same conversion.
/// The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (cache_dir, file_count, total_bytes, db_size))]
fn cache_status_payload<'py>(
    py: Python<'py>,
    cache_dir: &Bound<'py, PyAny>,
    file_count: &Bound<'py, PyAny>,
    total_bytes: &Bound<'py, PyAny>,
    db_size: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // str(cache_dir) — rendered like the reference's dict literal.
    out.set_item("cache_dir", py_str(cache_dir)?)
        .expect("dict insertion with a str key cannot fail");
    // Each aggregate is stored as the object it is, never coerced.
    out.set_item("file_count", file_count)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("total_bytes", total_bytes)
        .expect("dict insertion with a str key cannot fail");
    // format_size(total_bytes) — the same engine the Python side binds
    // to, through the same conversion its own call would perform.
    out.set_item("total_human", format_size(total_bytes.extract::<i64>()?))
        .expect("dict insertion with a str key cannot fail");
    out.set_item("db_size_bytes", db_size)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("db_size_human", format_size(db_size.extract::<i64>()?))
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The human-readable status text behind the CLI's standalone
/// `ytm cache status` command — the four aligned lines the command
/// echoes when `--json` is absent, joined by the newlines the four
/// `click.echo` calls appended (the caller's own echo supplies the
/// trailing one).
///
/// Mirrors the pure-Python reference (`_cache_status_text`): the
/// `Cache directory:` line renders `str(cache_dir)` exactly like the
/// reference's f-string (a `Path` as its path text), `Cached files:`
/// interpolates the count object through `str()` as it is, and the
/// two size lines come from the same `format_size` engine the Python
/// side binds to, through the same conversion its own calls would
/// perform. The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (cache_dir, file_count, total_bytes, db_size))]
fn cache_status_text(
    cache_dir: &Bound<'_, PyAny>,
    file_count: &Bound<'_, PyAny>,
    total_bytes: &Bound<'_, PyAny>,
    db_size: &Bound<'_, PyAny>,
) -> PyResult<String> {
    Ok(format!(
        "Cache directory: {}\nCached files:    {}\nTotal size:      {}\nDatabase size:   {}",
        py_str(cache_dir)?,
        py_str(file_count)?,
        format_size(total_bytes.extract::<i64>()?),
        format_size(db_size.extract::<i64>()?),
    ))
}

/// The directory walk behind the CLI's standalone `ytm cache clear`
/// command — the pure core of the delete in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cache_clear_walk`): when the
/// *dir_exists* thunk (the reference's `CACHE_DIR.exists()`) reports the
/// directory, every entry the *list_entries* thunk yields (the
/// reference's `CACHE_DIR.iterdir()`) that the *is_file* thunk accepts
/// is handed to the *unlink* thunk and counted — subdirectories and
/// everything that is not a plain file are left untouched — and the
/// walk returns the count the reference's `removed` accumulates. Each
/// thunk is invoked exactly where the reference performs its own
/// operation — the existence probe once up front, the listing only
/// when the directory exists, the file test once per entry in
/// iteration order, the unlink only for the entries that pass it — so
/// the filesystem side effects stay Python's (a `Path`, a `Path`
/// subclass or a stub steering both backends identically) while the
/// walk's control flow runs native. An error from any thunk propagates
/// exactly where the reference's own operation would raise it, leaving
/// the count behind it unreturned; the count is a fresh `int`; the
/// inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (dir_exists, list_entries, is_file, unlink))]
fn cache_clear_walk(
    dir_exists: &Bound<'_, PyAny>,
    list_entries: &Bound<'_, PyAny>,
    is_file: &Bound<'_, PyAny>,
    unlink: &Bound<'_, PyAny>,
) -> PyResult<u64> {
    let mut removed: u64 = 0;
    if dir_exists.call0()?.is_truthy()? {
        for entry in list_entries.call0()?.try_iter()? {
            let entry = entry?;
            if is_file.call1((&entry,))?.is_truthy()? {
                unlink.call1((&entry,))?;
                removed += 1;
            }
        }
    }
    Ok(removed)
}

/// The directory walk behind the CLI's standalone `ytm cache status`
/// command — the pure core of the status read in `cli.py`.
///
/// Mirrors the pure-Python reference (`_cache_status_walk`): when the
/// *dir_exists* thunk (the reference's `CACHE_DIR.exists()`) reports the
/// directory, every entry the *list_entries* thunk yields (the
/// reference's `CACHE_DIR.iterdir()`) that the *is_file* thunk accepts
/// contributes its *stat_size* thunk result (the reference's
/// `f.stat().st_size`) to the running totals — subdirectories and
/// everything that is not a plain file are left out — and the walk
/// returns the pair the reference's two accumulators build, the count
/// first like the reference's return. Each thunk is invoked exactly
/// where the reference performs its own operation — the existence probe
/// once up front, the listing only when the directory exists, the file
/// test once per entry in iteration order, the size read only for the
/// entries that pass it — so the filesystem side effects stay Python's
/// (a `Path`, a `Path` subclass or a stub steering both backends
/// identically) while the walk's control flow runs native. The size
/// read precedes the count increment exactly like the reference's
/// statement order, so an error from it leaves that entry uncounted
/// behind it; both accumulations run through CPython's own in-place add
/// (`operator.iadd` — `PyNumber_InPlaceAdd`), so the int + int case
/// stays an exact int whatever the sizes, and a stub's `st_size` that
/// returns something exotic follows the same numeric protocol the
/// reference's `+=` would (a float sum stays a float, a bigint stays
/// unbounded). An error from any thunk propagates exactly where the
/// reference's own operation would raise it, leaving the pair
/// unreturned; the pair is a fresh tuple; the inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (dir_exists, list_entries, is_file, stat_size))]
fn cache_status_walk<'py>(
    py: Python<'py>,
    dir_exists: &Bound<'py, PyAny>,
    list_entries: &Bound<'py, PyAny>,
    is_file: &Bound<'py, PyAny>,
    stat_size: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    // total_bytes = 0 / file_count = 0 — the reference's starting ints,
    // kept as the Python objects the additions build (an unbounded sum
    // renders exactly like the reference's, never a fixed-width
    // coercion).
    let mut total_bytes: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    let mut file_count: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    // operator.iadd — CPython's own += (PyNumber_InPlaceAdd), so both
    // accumulations follow the reference's numeric protocol exactly.
    let iadd = py.import("operator")?.getattr("iadd")?;
    // file_count += 1 — the reference's literal increment, pre-built.
    let one: Bound<'py, PyAny> = 1i64.into_pyobject(py)?.into_any();
    if dir_exists.call0()?.is_truthy()? {
        for entry in list_entries.call0()?.try_iter()? {
            let entry = entry?;
            if is_file.call1((&entry,))?.is_truthy()? {
                // total_bytes += f.stat().st_size — the size read first,
                // its result accumulated through the in-place add.
                let size = stat_size.call1((&entry,))?;
                total_bytes = iadd.call1((&total_bytes, &size))?;
                // file_count += 1 — then the count increment, exactly the
                // reference's statement order.
                file_count = iadd.call1((&file_count, &one))?;
            }
        }
    }
    // return file_count, total_bytes — the reference's pair order.
    Ok((file_count, total_bytes))
}

/// The editor resolution behind the CLI's standalone `ytm config`
/// command — the pure core of the config open in `cli.py`.
///
/// Mirrors the pure-Python reference (`_config_editor_plan`): a truthy
/// *editor_env* (the `EDITOR` environment value — an empty string is
/// falsy and falls through, exactly like the reference's `if editor:`)
/// opens the config *file* with that editor; otherwise a truthy
/// *is_win32* (the reference's `sys.platform == "win32"`, computed by
/// the caller) shell-opens the file (`os.startfile`, whose call stays
/// in the caller); otherwise the *resolve_xdg* thunk — the reference's
/// `shutil.which("xdg-open")` probe, passed as a zero-argument
/// callable and invoked exactly here, exactly once, only on this
/// branch, so the PATH scan never runs when an `EDITOR` is set — is
/// asked for `xdg-open`, whose result opens the config *directory*.
/// The `which` verdict is an `is None` check, not a truthiness one —
/// an empty-string path is still the editor — and a `None` result is
/// the manual-open error, its message naming the directory through
/// `str()` exactly like the reference's f-string. The editor value
/// passes through as the object it is; the targets render through
/// `str()` exactly like the reference's assignments; the plan is a
/// fresh map; the inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (editor_env, is_win32, resolve_xdg, config_file, config_dir))]
fn config_editor_plan<'py>(
    py: Python<'py>,
    editor_env: &Bound<'py, PyAny>,
    is_win32: &Bound<'py, PyAny>,
    resolve_xdg: &Bound<'py, PyAny>,
    config_file: &Bound<'py, PyAny>,
    config_dir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if editor_env.is_truthy()? {
        out.set_item("kind", "editor")?;
        out.set_item("editor", editor_env)?;
        out.set_item("target", py_str(config_file)?)?;
    } else if is_win32.is_truthy()? {
        out.set_item("kind", "startfile")?;
        out.set_item("target", py_str(config_file)?)?;
    } else {
        // shutil.which("xdg-open") — the probe, invoked exactly here
        // and exactly once, only on the no-EDITOR no-Windows branch.
        let editor = resolve_xdg.call0()?;
        let target = py_str(config_dir)?;
        // `if editor is None` — not a truthiness gate: an empty-string
        // path is still the editor.
        if editor.is_none() {
            out.set_item("kind", "error")?;
            out.set_item(
                "message",
                format!(
                    "No $EDITOR set and xdg-open not found. Open manually: {}",
                    py_str(config_dir)?
                ),
            )?;
        } else {
            out.set_item("kind", "editor")?;
            out.set_item("editor", editor)?;
            out.set_item("target", target)?;
        }
    }
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
        return Err(PyValueError::new_err(unpack_too_many_message(
            pair.py(),
            pair,
            2,
        )));
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
    json_encode(py, data, indent_pad, pretty, false, false)
}

/// `sorted(pairs)` with CPython's own tuple ordering — the rule the
/// reference's `sort_keys` arm applies to a dict's `items()`: tuple `a <
/// b` reads the first position through `==` (falling to the second only
/// between equal keys, so equal keys compare their values), every
/// ordering decision made through `<` exactly like `list.sort` (an
/// unorderable pair raising `<`'s own `TypeError`), and pairs equal
/// under neither `<` keeping their input order — `sorted` is stable.
fn py_sorted_pairs<'py>(
    mut pairs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    // tuple a < b — key equality first, the value `<` only between
    // equal keys, the key `<` otherwise.
    fn tuple_lt(
        a: &(Bound<'_, PyAny>, Bound<'_, PyAny>),
        b: &(Bound<'_, PyAny>, Bound<'_, PyAny>),
    ) -> PyResult<bool> {
        if a.0.eq(&b.0)? {
            a.1.lt(&b.1)
        } else {
            a.0.lt(&b.0)
        }
    }
    let mut sort_err: Option<PyErr> = None;
    pairs.sort_by(|a, b| {
        if sort_err.is_some() {
            // The outcome no longer matters — the error returns anyway.
            return std::cmp::Ordering::Equal;
        }
        match (tuple_lt(a, b), tuple_lt(b, a)) {
            (Ok(true), _) => std::cmp::Ordering::Less,
            (Ok(false), Ok(true)) => std::cmp::Ordering::Greater,
            (Ok(_), Ok(_)) => std::cmp::Ordering::Equal,
            (Err(err), _) | (_, Err(err)) => {
                sort_err = Some(err);
                std::cmp::Ordering::Equal
            }
        }
    });
    match sort_err {
        Some(err) => Err(err),
        None => Ok(pairs),
    }
}

/// The shared encode core behind `json_dumps` and the auth engines'
/// canonical serialiser — the same iterative CPython-encoder walk, with
/// the `sort_keys` and default-less arms parameterised: `sort_keys`
/// collects the dict's `items()` and orders them through CPython's own
/// tuple comparison (see `py_sorted_pairs`) instead of keeping
/// insertion order, and `strict` replaces the `default=str` conversion
/// with the reference's own `TypeError` for the unserialisable.
fn json_encode<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indent_pad: String,
    pretty: bool,
    sort_keys: bool,
    strict: bool,
) -> PyResult<String> {
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
                    // call both CPython encoders make — with sort_keys
                    // collecting the pairs and ordering them through
                    // CPython's own tuple comparison (see
                    // py_sorted_pairs) exactly like the reference
                    // encoder's `sorted(dct.items())`, the ordered pairs
                    // re-listed for the shared iteration.
                    let items = value.getattr("items")?.call0()?;
                    let iter = if sort_keys {
                        let mut pairs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
                        for pair in items.try_iter()? {
                            pairs.push(unpack_two(&pair?)?);
                        }
                        let pairs = py_sorted_pairs(pairs)?;
                        PyList::new(py, pairs)?.try_iter()?
                    } else {
                        items.try_iter()?
                    };
                    stack.push(JsonFrame::Dict {
                        iter,
                        first: true,
                        id,
                    });
                }
            } else if strict {
                // No default: the reference encoder's own TypeError for
                // the unserialisable.
                return Err(PyTypeError::new_err(format!(
                    "Object of type {} is not JSON serializable",
                    value.get_type().name()?
                )));
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

/// The `Fatal Python error:` tail extraction behind the Most recent
/// faulthandler trace section of `ytm doctor`. Mirrors the pure core of
/// `ytm_player.utils.doctor._recent_faulthandler` (the existence check
/// and the file read stay in Python): a text that strips to nothing —
/// every character in CPython's `str.isspace` set, the empty string
/// included, so `\x1c`-`\x1f` count where Rust's own `is_whitespace`
/// would stop at White_Space — returns the no-signals marker; otherwise
/// the *last* occurrence of `Fatal Python error:` (`str.rfind`, so the
/// last fatal block wins when the log holds several) opens the returned
/// slice, which runs to the end of the text; and a text without the
/// marker falls back to its last 4000 *characters* — `text[-4000:]`,
/// which is the whole text at or below 4000 and cuts on a character
/// boundary above it (Python slices by code point, not byte, so the
/// 4000-character tail of a multi-byte text can span well over 4000
/// bytes).
#[pyfunction]
fn faulthandler_tail(text: &str) -> String {
    // not text.strip() — empty after stripping iff every character is
    // CPython whitespace (an empty text included, all() on nothing).
    if text.chars().all(py_isspace) {
        return "(faulthandler.log is empty — no fatal signals captured)".to_string();
    }
    match text.rfind("Fatal Python error:") {
        // text[last_idx:] — from the last marker occurrence to the end.
        Some(idx) => text[idx..].to_string(),
        None => {
            // text[-4000:] — the last 4000 characters, the whole text at
            // or below 4000 of them.
            let char_count = text.chars().count();
            if char_count <= 4000 {
                return text.to_string();
            }
            let start = text
                .char_indices()
                .nth(char_count - 4000)
                .map(|(i, _)| i)
                .unwrap_or(0);
            text[start..].to_string()
        }
    }
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

/// The PEP 440 version grammar from `packaging`, in the two shapes its
/// releases have shipped, selected by the `admit_unicode_folds` probe
/// the Python side passes (see `crash_staleness_note`): every word
/// spelling is per-character cased (`AlPhA` is a valid pre-release
/// marker), and the wrapping whitespace is CPython's `\s` (see
/// `CRASH_META_RE`). Anchored at both ends like the reference's
/// `fullmatch`, with group names mirroring packaging's own.
///
/// The two shapes differ only in the Unicode case folds:
///
/// - `PEP440_FOLD_RE` matches the grammar `packaging` compiled through
///   26.0 — plain `re.IGNORECASE` on a Unicode pattern, no ASCII
///   scoping — which admits the simple case folds wherever a letter
///   can appear: the long s `ſ` (U+017F) wherever an s/S can, the
///   dotted and dotless İ/ı (U+0130/U+0131) wherever an i/I can, and
///   the Kelvin sign `K` (U+212A) wherever a k/K can. `1.0+ſ`,
///   `1.0+K`, `1.0prevıew` and `1.0poſt` are valid versions on those
///   releases.
/// - `PEP440_ASCII_RE` matches packaging 26.1+ — the groups are
///   `(?a:...)`-scoped and the local alphabet ASCII — so the same
///   fold spellings are InvalidVersion, exactly the verdicts the
///   pure-Python fallback's `Version(...)` returns on those releases.
///
/// Only the `preview` word's i, the `post` word's s and the local
/// alphabet can ever see a fold (no other grammar position spells an
/// i, s or k), which is why the two grammars differ in exactly those
/// three places. The comparison key the parse builds never changes:
/// on the ASCII grammar a fold spelling cannot appear, and on the
/// fold grammar it ranks by its raw characters through the same slot
/// tuples both grammars share.
static PEP440_FOLD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^[\s\x{1c}-\x{1f}]*[vV]?",
        r"(?:",
        r"(?:(?P<epoch>[0-9]+)!)?",
        r"(?P<release>[0-9]+(?:\.[0-9]+)*)",
        r"(?:[._-]?(?P<pre_l>[aA][lL][pP][hH][aA]|[aA]|[bB][eE][tT][aA]|[bB]",
        r"|[pP][rR][eE][vV][iI\x{130}\x{131}][eE][wW]|[pP][rR][eE]|[cC]|[rR][cC])",
        r"[._-]?(?P<pre_n>[0-9]+)?)?",
        r"(?:-(?P<post_n1>[0-9]+)|[._-]?(?P<post_l>[pP][oO][sS\x{17f}][tT]|[rR][eE][vV]|[rR])",
        r"[._-]?(?P<post_n2>[0-9]+)?)?",
        r"(?:[._-]?(?P<dev_l>[dD][eE][vV])[._-]?(?P<dev_n>[0-9]+)?)?",
        r")",
        r"(?:\+(?P<local>[a-zA-Z0-9\x{130}\x{131}\x{17f}\x{212a}]+(?:[._-][a-zA-Z0-9\x{130}\x{131}\x{17f}\x{212a}]+)*))?",
        r"[\s\x{1c}-\x{1f}]*$",
    ))
    .expect("valid pep440 regex")
});

/// The ASCII-scoped shape of the same grammar — packaging 26.1+ (see
/// `PEP440_FOLD_RE` for the shared structure and the fold inventory).
static PEP440_ASCII_RE: LazyLock<Regex> = LazyLock::new(|| {
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
    .expect("valid pep440 ascii regex")
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

/// packaging's `_LETTER_NORMALIZATION`, applied after `str.lower()`
/// exactly like `_parse_letter_version`: the alternate spellings collapse
/// onto the canonical marker letters (alpha→a, beta→b, c/pre/preview→rc,
/// rev/r→post), and a spelling the lower() left untouched — an exotic
/// Unicode fold like `prevıew` or `poſt`, which no map key equals — stays
/// raw and compares as its own string, exactly the tuple the reference
/// builds.
fn pep440_letter(letter: &str) -> String {
    match letter.to_lowercase().as_str() {
        "alpha" => "a".to_string(),
        "beta" => "b".to_string(),
        "c" | "pre" | "preview" => "rc".to_string(),
        "rev" | "r" => "post".to_string(),
        raw => raw.to_string(),
    }
}

/// One pre/post/dev slot of the comparison key. packaging's `_cmpkey`
/// fills each slot with `NegativeInfinity`, the `(letter, number)` tuple
/// or `Infinity`, and the slots compare in exactly that order — the
/// enum's derived ordering reproduces it: absent-before-everything, the
/// `(letter, number)` pairs comparing as tuples (the letter as its
/// post-lower() post-map string, so an exotic fold spelling like
/// `prevıew` ranks between `b` and `rc` by its raw characters, and
/// `poſt` sorts above `post`), absent-after-everything.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
enum Pep440Slot {
    NegInf,
    Seg(String, BigNum),
    Inf,
}

/// The pre/post/dev comparison suffix in packaging's `_cmpkey` order,
/// each slot one `Pep440Slot`.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct Pep440Suffix {
    pre: Pep440Slot,
    post: Pep440Slot,
    dev: Pep440Slot,
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

/// How a PEP 440 parse failed. The distinction matters to callers that
/// catch only `InvalidVersion` (an `InvalidVersion` subclass): the
/// grammar's rejection is that exception, while a numeric component past
/// CPython's int-conversion digit limit is the plain `ValueError` the
/// constructor's own `int()` raises — which those callers let propagate.
enum Pep440Err {
    /// The grammar rejected the string — packaging raises `InvalidVersion`.
    Invalid,
    /// A numeric component exceeds CPython's int-conversion digit limit
    /// (see `INT_MAX_STR_DIGITS`) — the constructor's `int()` raises a
    /// plain `ValueError`, not an `InvalidVersion`.
    IntLimit,
}

/// `BigNum::from_digits` with the failure reason kept: `None` (the digit
/// run exceeds CPython's int-conversion limit) becomes `Pep440Err::IntLimit`.
fn big_num(digits: &str) -> Result<BigNum, Pep440Err> {
    BigNum::from_digits(digits).ok_or(Pep440Err::IntLimit)
}

/// Parse a version string into its PEP 440 comparison key, keeping the
/// failure reason: `Err(Pep440Err::Invalid)` when the grammar rejects the
/// string, `Err(Pep440Err::IntLimit)` when a numeric component exceeds
/// CPython's int-conversion digit limit (see `INT_MAX_STR_DIGITS`).
/// *admit_unicode_folds* selects the grammar shape (see `PEP440_FOLD_RE`).
fn parse_pep440_err(text: &str, admit_unicode_folds: bool) -> Result<Pep440Key, Pep440Err> {
    let caps = if admit_unicode_folds {
        PEP440_FOLD_RE.captures(text)
    } else {
        PEP440_ASCII_RE.captures(text)
    };
    let caps = caps.ok_or(Pep440Err::Invalid)?;

    let epoch = big_num(caps.name("epoch").map(|m| m.as_str()).unwrap_or("0"))?;
    let mut release: Vec<BigNum> = caps
        .name("release")
        .ok_or(Pep440Err::Invalid)?
        .as_str()
        .split('.')
        .map(big_num)
        .collect::<Result<Vec<_>, _>>()?;
    // Strip trailing zeros: 1.0.0 compares equal to 1.
    while release.last().is_some_and(BigNum::is_zero) {
        release.pop();
    }

    // Each numeric group is Option<Result<BigNum, Pep440Err>>: the outer
    // Option is "did the group match", the inner is the digit-limit-checked
    // number. A matched group whose digits exceed the limit invalidates the
    // whole version (the reference's int() raises through), an unmatched one
    // simply carries no number.
    let pre_l = caps.name("pre_l").map(|m| m.as_str());
    let pre_n = caps.name("pre_n").map(|m| big_num(m.as_str()));
    let post_n1 = caps.name("post_n1").map(|m| big_num(m.as_str()));
    let post_l = caps.name("post_l").map(|m| m.as_str());
    let post_n2 = caps.name("post_n2").map(|m| big_num(m.as_str()));
    let dev_l = caps.name("dev_l").map(|m| m.as_str());
    let dev_n = caps.name("dev_n").map(|m| big_num(m.as_str()));

    let post_matched = post_n1.is_some() || post_l.is_some();
    let dev_matched = dev_l.is_some();

    // The pre slot: a dev-only release (no pre, no post, dev present)
    // sorts before every alpha; a release without a pre sorts after every
    // rc; otherwise the marker's (letter, number) pair — the letter through
    // packaging's lower+normalise map, an implicit 0 when the digits are
    // absent — comparing as the tuple it is, so an exotic fold spelling
    // like `prevıew` ranks by its raw characters, between `b` and `rc`.
    let pre = if let Some(letter) = pre_l {
        let number = match pre_n {
            Some(number) => number?,
            None => BigNum::zero(),
        };
        Pep440Slot::Seg(pep440_letter(letter), number)
    } else if !post_matched && dev_matched {
        Pep440Slot::NegInf
    } else {
        Pep440Slot::Inf
    };

    // post: `-N` (post_n1) and `postN`/`post`/`rev`/`r` (post_l+post_n2)
    // both normalise to the ("post", N) pair — the `-N` shape's letter is
    // exactly "post" — again with the implicit 0; no post segment sorts
    // before every one that has it.
    let post = if post_matched {
        let number = match post_n1.or(post_n2) {
            Some(number) => number?,
            None => BigNum::zero(),
        };
        let letter = match post_l {
            Some(letter) => pep440_letter(letter),
            None => "post".to_string(),
        };
        Pep440Slot::Seg(letter, number)
    } else {
        Pep440Slot::NegInf
    };

    // dev: the ("dev", N) pair with the implicit 0; no dev segment sorts
    // after every one that has it.
    let dev = match dev_l {
        Some(letter) => {
            let number = match dev_n {
                Some(number) => number?,
                None => BigNum::zero(),
            };
            Pep440Slot::Seg(pep440_letter(letter), number)
        }
        None => Pep440Slot::Inf,
    };

    let suffix = Pep440Suffix { pre, post, dev };

    let local = match caps.name("local") {
        None => None,
        Some(m) => Some(
            m.as_str()
                .split(['.', '_', '-'])
                .map(|part| {
                    if part.bytes().all(|b| b.is_ascii_digit()) {
                        // int(part): a digit run beyond the int-conversion
                        // limit invalidates the version.
                        big_num(part).map(Pep440LocalSeg::Num)
                    } else {
                        Ok(Pep440LocalSeg::Str(part.to_lowercase()))
                    }
                })
                .collect::<Result<Vec<Pep440LocalSeg>, _>>()?,
        ),
    };

    Ok(Pep440Key {
        epoch,
        release,
        suffix,
        local,
    })
}

/// Parse a version string into its PEP 440 comparison key, or None when
/// the string is not a valid version — or one of its numbers exceeds
/// CPython's int-conversion digit limit (see `INT_MAX_STR_DIGITS`); both
/// failure kinds read the same to the crash-staleness verdict (no note).
/// *admit_unicode_folds* selects the grammar shape (see `PEP440_FOLD_RE`).
fn parse_pep440(text: &str, admit_unicode_folds: bool) -> Option<Pep440Key> {
    parse_pep440_err(text, admit_unicode_folds).ok()
}

/// Split exactly like CPython's `str.split()` with no separator: on
/// runs of the `isspace` set (see `py_isspace` — a wider set than the
/// crate's own `split_whitespace`, which stops at the Unicode
/// `White_Space` property and misses `\x1c`–`\x1f`), producing no
/// empty pieces — leading and trailing whitespace yields nothing, and
/// an empty string yields no pieces at all.
fn py_split_none(s: &str) -> Vec<&str> {
    s.split(py_isspace)
        .filter(|piece| !piece.is_empty())
        .collect()
}

/// `str.strip()` with no arguments — trimmed of the `isspace` set's
/// leading and trailing characters (see `py_isspace` — a wider set than
/// the crate's own `str::trim`, which stops at the Unicode
/// `White_Space` property and misses `\x1c`–`\x1f`).
fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
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
///
/// *admit_unicode_folds* steers the PEP 440 grammar to the shape the
/// installed `packaging` compiled — the Python side probes it once
/// (`Version("1.0+ſ")`: parseable through packaging 26.0, InvalidVersion
/// from 26.1's `(?a:...)` scoping on) and passes the verdict here, so
/// the engine's accept/reject decisions track the reference the
/// fallback itself calls, on every `packaging` the dependency range
/// (`>=21.0`, uncapped) admits.
#[pyfunction]
#[pyo3(signature = (content, current_version, admit_unicode_folds))]
fn crash_staleness_note(
    content: &str,
    current_version: &str,
    admit_unicode_folds: bool,
) -> Option<String> {
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
    let (crash_key, current_key) = (
        parse_pep440(crash_version, admit_unicode_folds),
        parse_pep440(current_version, admit_unicode_folds),
    );
    if crash_key? < current_key? {
        return Some(format!(
            "⚠ This crash was recorded by ytm-player {crash_version}, older than the installed \
             {current_version} — it may already be fixed. Reproduce on the current version \
             before reporting."
        ));
    }
    None
}

/// The process-status text behind the Process status section of
/// `ytm doctor` — the pure core of `_running_status` in
/// `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_running_status_text`): the
/// /proc walk, with every side-effecting probe passed in as a callable
/// the engine invokes exactly where the reference performs it — the
/// *proc_exists* thunk (the reference's `Path("/proc").exists()`), the
/// *proc_entries* thunk (the reference's `proc_root.iterdir()`), the
/// per-entry *read_cmdline* and *read_status* thunks (the reference's
/// `(entry / "cmdline").read_bytes()` and `(entry / "status")
/// .read_text(encoding="utf-8", errors="replace")`), and the *own_pid*
/// thunk (the reference's `os.getpid()`, evaluated once per collected
/// pid by the reference's list-comprehension filter — so zero times
/// when nothing was collected, once per match when something was).
///
/// The walk itself: a missing /proc is the cannot-determine marker;
/// every entry whose `.name` attribute's own `isdigit` accepts —
/// CPython's Numeric_Type=Digit-or-Decimal set, which the crate carries
/// no table for, so the verdict runs through the name object's own
/// method, str subclasses included — gets its cmdline read (an
/// `OSError` skips the entry, exactly the reference's swallow; anything
/// else propagates), NUL bytes replaced by spaces, the result decoded
/// like `decode("utf-8", errors="replace")` — one U+FFFD per maximal
/// invalid subpart, the same subpart boundaries CPython's replace
/// handler draws — and split like `str.split()` (the CPython whitespace
/// set, see `py_split_none`); an empty argv skips the entry, a
/// non-ytm argv skips it through the same `argv_is_ytm` engine the
/// Python side's module global resolves to (its own parity tests lock
/// the two spellings together). A match gets its status read (the
/// `OSError`-only swallow again), the first `VmRSS:` line's second
/// split word coerced through CPython's own `int()` — the acceptance
/// set (signed, underscored and non-ASCII digit spellings, the
/// 4300-digit limit) and the `ValueError` swallows matching the
/// reference's exactly, a missing or unparseable word leaving the
/// reference's 0 (see `vmrss_word_kb`) — and the pair `int(entry.name)`
/// builds appended, the pid conversion running through the builtin too
/// so an isdigit-accepted name int() rejects (a superscript digit, say)
/// raises exactly at the reference's append. The filter drops the
/// entry whose pid compares equal to the thunk's (CPython's own `!=`,
/// so an int own-pid compares numerically and anything else is never
/// equal); no survivors is `ytm not running`, otherwise the survivors
/// render as `PID <pid> (RSS <rss // 1024> MB)` parts — the floor
/// division through the int object's own `__floordiv__`, so an
/// unbounded VmRSS stays exact — joined after `ytm running: `.
///
/// The cmdline thunk's result is read as `bytes` and the status
/// thunk's as `str` — the shapes `read_bytes()` and `read_text()`
/// return — and both stay unmutated. The pid and RSS objects are the
/// objects `int()` made; everything else is fresh. The inputs are
/// never mutated.
#[pyfunction]
#[pyo3(signature = (proc_exists, proc_entries, read_cmdline, read_status, own_pid))]
fn running_status_text<'py>(
    py: Python<'py>,
    proc_exists: &Bound<'py, PyAny>,
    proc_entries: &Bound<'py, PyAny>,
    read_cmdline: &Bound<'py, PyAny>,
    read_status: &Bound<'py, PyAny>,
    own_pid: &Bound<'py, PyAny>,
) -> PyResult<String> {
    if !proc_exists.call0()?.is_truthy()? {
        return Ok("(cannot determine — /proc not available on this OS)".to_string());
    }
    // rss_kb = 0 — the reference's starting int, swapped in whenever the
    // VmRSS parse yields nothing.
    let zero_kb: Bound<'py, PyAny> = 0i64.into_pyobject(py)?.into_any();
    // (pid, rss) — both stay the Python objects int() made them, so a
    // pid or VmRSS beyond i64 renders exactly like the reference's
    // f-strings instead of overflowing a fixed-width coercion.
    let mut pids: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for entry in proc_entries.call0()?.try_iter()? {
        let entry = entry?;
        // entry.name — the attribute read the reference performs.
        let name = entry.getattr("name")?;
        // entry.name.isdigit() — through the name object's own method
        // (see the doc comment above).
        if !name.call_method0("isdigit")?.is_truthy()? {
            continue;
        }
        // cmdline_path.read_bytes() — the OSError-only swallow.
        let cmdline_raw = match read_cmdline.call1((&entry,)) {
            Ok(raw) => raw,
            Err(err) if err.is_instance_of::<PyOSError>(py) => continue,
            Err(err) => return Err(err),
        };
        // replace(b"\x00", b" ") then decode("utf-8", errors="replace").
        let raw_bytes: Vec<u8> = cmdline_raw.extract()?;
        let spaced: Vec<u8> = raw_bytes
            .iter()
            .map(|&b| if b == 0 { b' ' } else { b })
            .collect();
        let cmdline = String::from_utf8_lossy(&spaced);
        // cmdline.split() — str.split() with no separator.
        let argv: Vec<String> = py_split_none(&cmdline)
            .into_iter()
            .map(str::to_string)
            .collect();
        if argv.is_empty() {
            continue;
        }
        // _argv_is_ytm(argv) — the same engine the Python side's module
        // global resolves to.
        if !argv_is_ytm(argv) {
            continue;
        }
        // (entry / "status").read_text(...) — the OSError-only swallow
        // again.
        let status_text = match read_status.call1((&entry,)) {
            Ok(text) => text,
            Err(err) if err.is_instance_of::<PyOSError>(py) => continue,
            Err(err) => return Err(err),
        };
        let status: String = status_text.extract()?;
        // The VmRSS scan: the first VmRSS-prefixed line decides, and the
        // break is unconditional — a later line never gets a say.
        let mut rss_kb: Option<Bound<'py, PyAny>> = None;
        for line in py_splitlines(&status) {
            if line.starts_with("VmRSS:") {
                rss_kb = Some(vmrss_word_kb(py, &line)?);
                break;
            }
        }
        // int(entry.name) — through the builtin, at the reference's
        // append-time position.
        let pid = builtin_int(py, &name)?;
        let rss = rss_kb.unwrap_or_else(|| zero_kb.clone());
        pids.push((pid, rss));
    }
    // pids = [(pid, rss) for pid, rss in pids if pid != os.getpid()] —
    // the reference's comprehension evaluates the getpid call once per
    // collected pid, so the thunk is invoked per element here too.
    let mut kept: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for (pid, rss) in pids {
        let own = own_pid.call0()?;
        // pid != own — CPython's own != through PyObject_RichCompare.
        let differs = pid.rich_compare(&own, CompareOp::Ne)?.is_truthy()?;
        if differs {
            kept.push((pid, rss));
        }
    }
    if kept.is_empty() {
        return Ok("ytm not running".to_string());
    }
    // parts = [f"PID {pid} (RSS {rss // 1024} MB)" ...] — the floor
    // division through the int object's own __floordiv__, the
    // interpolation through str() exactly like the reference's f-strings.
    let mut parts: Vec<String> = Vec::with_capacity(kept.len());
    for (pid, rss) in &kept {
        let mb = rss.call_method1("__floordiv__", (1024,))?;
        parts.push(format!("PID {} (RSS {} MB)", py_str(pid)?, py_str(&mb)?));
    }
    Ok("ytm running: ".to_string() + &parts.join(", "))
}

/// `int(line.split()[1])` — the VmRSS value word — with the reference's
/// `(IndexError, ValueError)` swallow back to 0: a line whose split
/// yields fewer than two words is the IndexError arm, a word CPython's
/// `int()` rejects is the ValueError one, and anything `int()` accepts
/// is the value — the object itself, unbounded, never a fixed-width
/// coercion (see `builtin_int`).
fn vmrss_word_kb<'py>(py: Python<'py>, line: &str) -> PyResult<Bound<'py, PyAny>> {
    let words = py_split_none(line);
    let Some(word) = words.get(1) else {
        // IndexError — line.split()[1] on a one-word (or empty) line.
        return Ok(0i64.into_pyobject(py)?.into_any());
    };
    let word_obj = PyString::new(py, word);
    match builtin_int(py, &word_obj) {
        Ok(value) => Ok(value),
        // ValueError — a word int() cannot parse ("5,368", "N/A", a run
        // beyond the 4300-digit limit, ...).
        Err(err) if err.is_instance_of::<PyValueError>(py) => {
            Ok(0i64.into_pyobject(py)?.into_any())
        }
        Err(err) => Err(err),
    }
}

/// The `mpv:` version line behind the mpv status probe of `ytm doctor`
/// — the pure core of `_mpv_version` in `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_mpv_version_text`): the
/// *which_mpv* thunk — the reference's `shutil.which("mpv")` PATH
/// probe, invoked exactly once, exactly up front — and a falsy result
/// (an unset `None` or an empty string alike, the reference's
/// `if not mpv_bin:`) is the NOT FOUND marker, the *run_version* thunk
/// never invoked; otherwise the *run_version* thunk — the reference's
/// `subprocess.run([mpv_bin, "--version"], ...)` — runs with the
/// resolved binary, and the version line comes from its result: the
/// `stdout or stderr or ""` or-chain picks the first truthy output —
/// `stderr` read only when `stdout` is falsy, exactly like the
/// reference's or-chain, so a result whose `stderr` attribute raises
/// never raises when `stdout` already satisfied it — the chosen
/// output's own `splitlines()` supplies the lines (CPython's line
/// boundaries, `\x85` and `\u{2028}` included, subclasses included),
/// the `[0:1]` slice the first of them, the `unknown` stand-in when no
/// line made it, and the f-string the `mpv: ` prefix. An `OSError` or
/// `subprocess.SubprocessError` raised anywhere in the run or the
/// parse — the whole span the reference's try block covers — is the
/// failed-to-execute marker (see `is_subprocess_error`); anything else
/// propagates. The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (which_mpv, run_version))]
fn mpv_version_text<'py>(
    py: Python<'py>,
    which_mpv: &Bound<'py, PyAny>,
    run_version: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let mpv_bin = which_mpv.call0()?;
    if !mpv_bin.is_truthy()? {
        return Ok("mpv: NOT FOUND in PATH".to_string());
    }
    match mpv_version_attempt(py, run_version, &mpv_bin) {
        Ok(line) => Ok(line),
        // except (OSError, subprocess.SubprocessError) — the reference's
        // clause, both arms through CPython's own exception matching.
        Err(err) if err.is_instance_of::<PyOSError>(py) || is_subprocess_error(py, &err)? => {
            Ok("mpv: failed to execute".to_string())
        }
        Err(err) => Err(err),
    }
}

/// The run-and-parse half of the mpv version probe — everything the
/// reference's try block covers, so an `OSError` or
/// `subprocess.SubprocessError` from any of it lands in the
/// failed-to-execute arm of the caller.
fn mpv_version_attempt<'py>(
    py: Python<'py>,
    run_version: &Bound<'py, PyAny>,
    mpv_bin: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let out = run_version.call1((mpv_bin,))?;
    // out.stdout — always read; out.stderr — only when stdout is falsy,
    // exactly like the reference's or-chain.
    let stdout = out.getattr("stdout")?;
    let chosen = if stdout.is_truthy()? {
        stdout
    } else {
        let stderr = out.getattr("stderr")?;
        if stderr.is_truthy()? {
            stderr
        } else {
            // "" — the or-chain's literal; its splitlines() is always
            // [], so the [0:1] slice is empty and the marker is unknown.
            return Ok("mpv: unknown".to_string());
        }
    };
    // .splitlines() — the chosen output's own method.
    let lines = chosen.call_method0("splitlines")?;
    // [0:1] — the slice, then its truthiness, then first_line[0].
    let first_line = lines.get_item(PySlice::new(py, 0, 1, 1))?;
    if first_line.is_truthy()? {
        let first = first_line.get_item(0)?;
        Ok(format!("mpv: {}", py_str(&first)?))
    } else {
        Ok("mpv: unknown".to_string())
    }
}

/// `isinstance(err, subprocess.SubprocessError)` — the second arm of
/// the mpv probe's except clause, through the module's own exception
/// root (`subprocess.TimeoutExpired` and `CalledProcessError` inherit
/// it, so the reference's clause catches them too).
fn is_subprocess_error(py: Python<'_>, err: &PyErr) -> PyResult<bool> {
    let subprocess_error = py.import("subprocess")?.getattr("SubprocessError")?;
    Ok(err.is_instance(py, &subprocess_error))
}

/// The libmpv status line behind the libmpv loadability probe of
/// `ytm doctor` — the pure core of `_libmpv_status` in
/// `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_libmpv_status_text`): the
/// *import_player* thunk — the reference's
/// `from ytm_player.services.player import _IMPORT_ERROR_MSG` — is
/// invoked exactly once, exactly up front, and an `ImportError` it
/// raises (whatever arm produced it: the module's own import failing,
/// or the name being absent because the mpv import succeeded) is the
/// OK marker, the library loadable; otherwise the message the thunk
/// returns is rendered through `str()` exactly like the reference's
/// `str(_IMPORT_ERROR_MSG)`, its own `splitlines()` supplies the lines
/// (CPython's line boundaries, `\x85` and `\u{2028}` included), the
/// `[-1]` index the last of them (an `IndexError` when none made it,
/// propagating exactly like the reference's — the except clause covers
/// `ImportError` only), and the `strip()`-ed line (Python's whitespace
/// set, see `py_strip`) interpolates into the NOT LOADABLE message.
/// Everything the reference's try block covered stays under the same
/// single `except ImportError` here — a message object whose own
/// `__str__` raises `ImportError` is the OK marker too (see
/// `libmpv_status_attempt`), exactly like the reference's. The inputs
/// are never mutated.
#[pyfunction]
#[pyo3(signature = (import_player))]
fn libmpv_status_text<'py>(py: Python<'py>, import_player: &Bound<'py, PyAny>) -> PyResult<String> {
    match libmpv_status_attempt(import_player) {
        Ok(text) => Ok(text),
        // except ImportError — the reference's only clause, matched
        // through the exception type itself.
        Err(err) if err.is_instance_of::<PyImportError>(py) => Ok("libmpv: OK".to_string()),
        Err(err) => Err(err),
    }
}

/// The import-and-render half of the libmpv probe — everything the
/// reference's try block covers, so an `ImportError` from any of it
/// (the thunk itself, or a message object's own `__str__`) lands in the
/// OK arm of the caller, and anything else propagates.
fn libmpv_status_attempt(import_player: &Bound<'_, PyAny>) -> PyResult<String> {
    // from ytm_player.services.player import _IMPORT_ERROR_MSG — the
    // thunk, exactly once, exactly up front.
    let import_error_msg = import_player.call0()?;
    // str(_IMPORT_ERROR_MSG) — the reference's conversion, inside its
    // try block, so an ImportError it raises is the OK marker.
    let text = py_str(&import_error_msg)?;
    // .splitlines() — CPython's line boundaries (see py_splitlines).
    let lines = py_splitlines(&text);
    // [-1] — the last line; an empty list is the reference's own
    // IndexError (the except clause covers ImportError only), reproduced
    // through CPython's own message.
    let last = lines
        .last()
        .ok_or_else(|| PyIndexError::new_err("list index out of range"))?;
    // first.strip() — Python's whitespace set (see py_strip).
    Ok(format!("libmpv: NOT LOADABLE ({})", py_strip(last)))
}

/// The MPRIS status line behind the MPRIS availability probe of
/// `ytm doctor` — the pure core of `_mpris_status` in
/// `utils/doctor.py`.
///
/// Mirrors the pure-Python reference (`_mpris_status_text`): a falsy
/// *is_linux* (the reference's `sys.platform != "linux"`, computed by
/// the caller) is the n/a marker, the *system_name* thunk — the
/// reference's `platform.system()` — invoked exactly here, exactly
/// once, only on this branch, so the platform probe never runs on
/// Linux — rendering into the message through `str()` exactly like the
/// reference's f-string; otherwise the *import_mpris* thunk — the
/// reference's `from ytm_player.services.mpris import BUS_NAME,
/// DBUS_AVAILABLE` — supplies the pair, unpacked with CPython's own
/// `UNPACK_SEQUENCE(2)` semantics (see `unpack_pair`), and a truthy
/// `DBUS_AVAILABLE` element is the available line interpolating the
/// `BUS_NAME` element through `str()` exactly like the reference's
/// f-string, a falsy one the UNAVAILABLE fix text. An error from
/// either thunk propagates exactly where the reference's own operation
/// would raise it — the reference has no try around the import, so an
/// `ImportError` surfaces instead of becoming the unavailable marker.
/// The inputs are never mutated.
#[pyfunction]
#[pyo3(signature = (is_linux, system_name, import_mpris))]
fn mpris_status_text<'py>(
    py: Python<'py>,
    is_linux: &Bound<'py, PyAny>,
    system_name: &Bound<'py, PyAny>,
    import_mpris: &Bound<'py, PyAny>,
) -> PyResult<String> {
    if !is_linux.is_truthy()? {
        // platform.system() — invoked exactly here, exactly once, only
        // on this branch, exactly like the reference's f-string.
        let system = system_name.call0()?;
        return Ok(format!(
            "MPRIS: n/a ({} uses its native media integration)",
            py_str(&system)?
        ));
    }
    // from ytm_player.services.mpris import BUS_NAME, DBUS_AVAILABLE —
    // the thunk supplies the pair; the unpack is CPython's own (see
    // unpack_pair).
    let pair = import_mpris.call0()?;
    let (bus_name, dbus_available) = unpack_pair(py, &pair)?;
    if dbus_available.is_truthy()? {
        Ok(format!(
            "MPRIS: available (dbus-fast OK; registers as {})",
            py_str(&bus_name)?
        ))
    } else {
        Ok(
            "MPRIS: UNAVAILABLE — dbus-fast is missing or failed to load (it's a \
             Linux core dependency, so this install is broken/partial; an \
             incompatible dbus-fast build logs the failure to ytm.log — see the \
             recent-errors section below). Fix: reinstall ytm-player \
             (pip install --force-reinstall ytm-player), or pip install dbus-fast."
                .to_string(),
        )
    }
}

/// `a, b = obj` — CPython's `UNPACK_SEQUENCE(2)` exactly: the
/// exact-tuple and exact-list fast paths (a *subclass* of either does
/// not take them — its own `__iter__` runs, exactly like the opcode's
/// slow path), otherwise the iterator protocol with the opcode's own
/// error shaping — `ValueError: not enough values to unpack (expected
/// 2, got N)` when the iterator ends early, `ValueError: too many
/// values to unpack (expected 2)` when it keeps going (on 3.14 the
/// exact types carry the actual count — see `unpack_too_many_message`),
/// and a non-iterable reporting `TypeError: cannot unpack non-iterable <type>
/// object`, the opcode's reformat of `GetIter`'s own TypeError.
fn unpack_pair<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    // PyTuple_CheckExact(seq) && PyTuple_GET_SIZE(seq) == 2 — the fast
    // path reads the items directly, no __iter__ dispatch.
    if let Ok(tuple) = obj.downcast_exact::<PyTuple>() {
        if tuple.len() == 2 {
            return Ok((tuple.get_item(0)?, tuple.get_item(1)?));
        }
    }
    // PyList_CheckExact(seq) && PyList_GET_SIZE(seq) == 2 — the same.
    if let Ok(list) = obj.downcast_exact::<PyList>() {
        if list.len() == 2 {
            return Ok((list.get_item(0)?, list.get_item(1)?));
        }
    }
    // The slow path — PyObject_GetIter, with the opcode's reformat of a
    // non-iterable's TypeError.
    let mut iter = match obj.try_iter() {
        Ok(iter) => iter,
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(py) {
                let type_name = obj.get_type().name()?;
                return Err(PyTypeError::new_err(format!(
                    "cannot unpack non-iterable {type_name} object"
                )));
            }
            return Err(err);
        }
    };
    // Collect exactly two items, counting the successes for the
    // not-enough message.
    let mut items: [Option<Bound<'py, PyAny>>; 2] = [None, None];
    let mut got = 0usize;
    for slot in &mut items {
        match iter.next() {
            Some(Ok(item)) => {
                *slot = Some(item);
                got += 1;
            }
            Some(Err(err)) => return Err(err),
            None => break, // StopIteration before the pair was filled
        }
    }
    let [first, second] = items;
    match (first, second) {
        (Some(a), Some(b)) => {
            // The extra-item probe: a third item is too many, the
            // iterator ending is the reference's success.
            match iter.next() {
                Some(Ok(_)) => Err(PyValueError::new_err(unpack_too_many_message(py, obj, 2))),
                Some(Err(err)) => Err(err),
                None => Ok((a, b)),
            }
        }
        _ => Err(PyValueError::new_err(format!(
            "not enough values to unpack (expected 2, got {got})"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Theme resolution engines (mirror ytm_player/ui/theme.py and app/_app.py)
// ---------------------------------------------------------------------------

/// Translate a Textual-only ANSI color token into a Rich-parseable name.
/// Mirrors `ytm_player.ui.theme.rich_safe_color`: the `ansi_` prefix is
/// stripped (five ASCII bytes), anything else passes through unchanged.
#[pyfunction]
fn rich_safe_color(color: &str) -> String {
    color.strip_prefix("ansi_").unwrap_or(color).to_string()
}

/// `variables.get(key, fallback)` with the fallback as a fresh Python
/// string — a present key yields the original object, not a copy, exactly
/// like the reference's dict lookups.
fn css_get_or<'py>(
    variables: &Bound<'py, PyDict>,
    key: &str,
    fallback: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match variables.get_item(key)? {
        Some(v) => Ok(v),
        None => Ok(PyString::new(variables.py(), fallback).into_any()),
    }
}

/// `variables.get(key, default)` with the default already built.
fn get_or<'py>(
    variables: &Bound<'py, PyDict>,
    key: &str,
    default: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match variables.get_item(key)? {
        Some(v) => Ok(v),
        None => Ok(default),
    }
}

/// The 21 ThemeColors field values resolved from a CSS-variables dict.
/// Mirrors the pure core of `ThemeColors.from_css_variables`
/// (`ytm_player.ui.theme`): each of the twelve base fields is its CSS
/// variable (note `muted_text` reads `text-muted`) or the dataclass
/// default; the nine app-specific fields fall back one level deeper —
/// the derived-from variable first, that field's own dataclass default
/// last, so the shared keys (`surface`, `text`, `text-muted`, `primary`)
/// carry each field's own innermost constant rather than the base
/// field's; and `lyrics_current` runs the documented three-level chain
/// lyrics-current → accent → primary → DEFAULT_LYRIC_CURRENT. A present
/// value passes through as the object it is — the ANSI-token
/// normalisation is NOT applied here, exactly as the reference's mapping
/// doesn't apply it: `ThemeColors.__post_init__` runs it after
/// construction, on whichever backend built the fields.
#[pyfunction]
fn theme_fields_from_css<'py>(
    py: Python<'py>,
    variables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    // The twelve base colours: (field, CSS variable, dataclass default).
    const BASE: [(&str, &str, &str); 12] = [
        ("background", "background", "#0f0f0f"),
        ("foreground", "foreground", "#ffffff"),
        ("primary", "primary", "#ff0000"),
        ("secondary", "secondary", "#aaaaaa"),
        ("accent", "accent", "#ff4e45"),
        ("success", "success", "#2ecc71"),
        ("warning", "warning", "#f39c12"),
        ("error", "error", "#e74c3c"),
        ("surface", "surface", "#1a1a1a"),
        ("border", "border", "#333333"),
        ("muted_text", "text-muted", "#999999"),
        ("text", "text", "#ffffff"),
    ];
    let fields = PyDict::new(py);
    for (name, css_var, default) in BASE {
        fields.set_item(name, css_get_or(variables, css_var, default)?)?;
    }

    // The nine app-specific fields, each its own nested chain — the
    // derived-from key's value first (or that field's own default), the
    // field's own innermost constant at the bottom.
    fields.set_item(
        "playback_bar_bg",
        get_or(
            variables,
            "playback-bar-bg",
            css_get_or(variables, "surface", "#1a1a1a")?,
        )?,
    )?;
    fields.set_item(
        "active_tab",
        get_or(
            variables,
            "active-tab",
            css_get_or(variables, "text", "#ffffff")?,
        )?,
    )?;
    fields.set_item(
        "inactive_tab",
        get_or(
            variables,
            "inactive-tab",
            css_get_or(variables, "text-muted", "#999999")?,
        )?,
    )?;
    fields.set_item(
        "selected_item",
        get_or(
            variables,
            "selected-item",
            css_get_or(variables, "surface", "#2a2a2a")?,
        )?,
    )?;
    fields.set_item(
        "progress_filled",
        get_or(
            variables,
            "progress-filled",
            css_get_or(variables, "primary", "#ff0000")?,
        )?,
    )?;
    fields.set_item(
        "progress_empty",
        get_or(
            variables,
            "progress-empty",
            css_get_or(variables, "surface", "#555555")?,
        )?,
    )?;
    fields.set_item(
        "lyrics_played",
        get_or(
            variables,
            "lyrics-played",
            css_get_or(variables, "text-muted", "#999999")?,
        )?,
    )?;
    // The three-level chain: lyrics-current → accent → primary →
    // DEFAULT_LYRIC_CURRENT (the dataclass default of the field itself).
    fields.set_item(
        "lyrics_current",
        get_or(
            variables,
            "lyrics-current",
            get_or(
                variables,
                "accent",
                css_get_or(variables, "primary", "#ff4e45")?,
            )?,
        )?,
    )?;
    fields.set_item(
        "lyrics_upcoming",
        get_or(
            variables,
            "lyrics-upcoming",
            css_get_or(variables, "text", "#aaaaaa")?,
        )?,
    )?;

    Ok(fields)
}

/// The app's CSS-variables overlay on Textual's theme variables. Mirrors
/// the pure core of `YTMPlayerApp.get_css_variables`
/// (`ytm_player.app._app`): the nine app-specific variables default to
/// values derived from the base palette — all evaluated from the
/// pre-patch state, exactly as the reference's dict literal evaluates
/// them before any insertion, each carrying its own innermost constant —
/// and each is inserted only when the theme hasn't set it; then the
/// theme.toml overrides layer on top — every ThemeColors field name
/// mapped to its CSS dash-case variable, in the reference's literal
/// order, user values winning over everything. The `variables` dict is
/// patched in place and returned — the same object the reference mutates
/// — so existing keys keep their position and new ones append in
/// app-default order.
#[pyfunction]
fn app_css_variables<'py>(
    variables: &Bound<'py, PyDict>,
    colors: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let defaults: [(&str, Bound<'py, PyAny>); 9] = [
        (
            "playback-bar-bg",
            css_get_or(variables, "surface", "#1a1a1a")?,
        ),
        ("active-tab", css_get_or(variables, "text", "#ffffff")?),
        (
            "inactive-tab",
            css_get_or(variables, "text-muted", "#999999")?,
        ),
        (
            "selected-item",
            css_get_or(variables, "surface", "#2a2a2a")?,
        ),
        (
            "progress-filled",
            css_get_or(variables, "primary", "#ff0000")?,
        ),
        (
            "progress-empty",
            css_get_or(variables, "surface", "#555555")?,
        ),
        (
            "lyrics-played",
            css_get_or(variables, "text-muted", "#999999")?,
        ),
        (
            "lyrics-current",
            get_or(
                variables,
                "accent",
                css_get_or(variables, "primary", "#ff4e45")?,
            )?,
        ),
        ("lyrics-upcoming", css_get_or(variables, "text", "#aaaaaa")?),
    ];
    for (key, default) in &defaults {
        // `key not in variables` / `variables[key] = default` — the
        // protocol operations the reference spells, so a dict subclass
        // steering __contains__/__setitem__ behaves exactly as it would
        // there (the default computations above stay non-dispatching
        // `.get`s, like the reference's).
        if !variables.as_any().contains(*key)? {
            variables.as_any().set_item(*key, default.clone())?;
        }
    }

    // theme.toml overrides — `if colors:` gates on the object's
    // truthiness (an empty table is no overrides), and each of the 21
    // field names maps to its CSS variable in the reference's literal
    // order. The membership test and the item read are the Python
    // protocol operations the reference spells (`in`, `[]`), so a
    // non-dict mapping would behave exactly as it would there.
    if colors.is_truthy()? {
        let field_to_css: [(&str, &str); 21] = [
            ("background", "background"),
            ("foreground", "foreground"),
            ("primary", "primary"),
            ("secondary", "secondary"),
            ("accent", "accent"),
            ("success", "success"),
            ("warning", "warning"),
            ("error", "error"),
            ("surface", "surface"),
            ("border", "border"),
            ("text", "text"),
            ("muted_text", "text-muted"),
            ("playback_bar_bg", "playback-bar-bg"),
            ("active_tab", "active-tab"),
            ("inactive_tab", "inactive-tab"),
            ("selected_item", "selected-item"),
            ("progress_filled", "progress-filled"),
            ("progress_empty", "progress-empty"),
            ("lyrics_played", "lyrics-played"),
            ("lyrics_current", "lyrics-current"),
            ("lyrics_upcoming", "lyrics-upcoming"),
        ];
        for (field_name, css_name) in &field_to_css {
            if colors.contains(*field_name)? {
                let value = colors.get_item(*field_name)?;
                variables.as_any().set_item(*css_name, value)?;
            }
        }
    }

    Ok(variables.clone())
}

/// The 21 ThemeColors field values from a Textual Theme's raw colours.
/// Mirrors the pure core of `YTMPlayerApp.watch_theme`'s ThemeColors
/// rebuild (`ytm_player.app._app`): the nine raw Theme attributes arrive
/// as plain values (the attribute reads stay in Python, in the
/// reference's first-read order), and every field applies the
/// reference's own `or` chain — a falsy raw value (None or empty) falls
/// back to its per-site constant, which differs between sites sharing a
/// source (`foreground` yields `#ffffff` for itself but `#aaaaaa` for
/// `lyrics_upcoming`; `surface` yields `#1a1a1a`, `#333333` and
/// `#555555` at its three sites); `progress_filled` keeps the raw
/// `primary` as its default even when falsy; and `lyrics_current`'s
/// default side runs the raw accent → primary → DEFAULT_LYRIC_CURRENT
/// chain. The `variables` dict is the Theme's own; a present
/// app-specific key wins over every fallback. As in
/// `theme_fields_from_css`, the values pass through unnormalised —
/// `__post_init__` runs after construction.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the Theme's nine raw colours — one parameter per attribute the reference reads
#[pyo3(signature = (primary, background, foreground, secondary, accent, success, warning, error, surface, variables))]
fn theme_fields_from_theme<'py>(
    py: Python<'py>,
    primary: &Bound<'py, PyAny>,
    background: &Bound<'py, PyAny>,
    foreground: &Bound<'py, PyAny>,
    secondary: &Bound<'py, PyAny>,
    accent: &Bound<'py, PyAny>,
    success: &Bound<'py, PyAny>,
    warning: &Bound<'py, PyAny>,
    error: &Bound<'py, PyAny>,
    surface: &Bound<'py, PyAny>,
    variables: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    /// `value or fallback` — CPython truthiness, the original object when
    /// truthy, a fresh Python string otherwise.
    fn or_fallback<'py>(value: &Bound<'py, PyAny>, fallback: &str) -> PyResult<Bound<'py, PyAny>> {
        if value.is_truthy()? {
            Ok(value.clone())
        } else {
            Ok(PyString::new(value.py(), fallback).into_any())
        }
    }

    let fields = PyDict::new(py);
    fields.set_item("primary", primary.clone())?;
    fields.set_item("background", or_fallback(background, "#0f0f0f")?)?;
    fields.set_item("foreground", or_fallback(foreground, "#ffffff")?)?;
    fields.set_item("secondary", or_fallback(secondary, "#aaaaaa")?)?;
    fields.set_item("accent", or_fallback(accent, "#ff4e45")?)?;
    fields.set_item("success", or_fallback(success, "#2ecc71")?)?;
    fields.set_item("warning", or_fallback(warning, "#f39c12")?)?;
    fields.set_item("error", or_fallback(error, "#e74c3c")?)?;
    fields.set_item("surface", or_fallback(surface, "#1a1a1a")?)?;
    fields.set_item("text", or_fallback(foreground, "#ffffff")?)?;
    fields.set_item("muted_text", or_fallback(secondary, "#999999")?)?;
    fields.set_item("border", or_fallback(surface, "#333333")?)?;
    fields.set_item(
        "playback_bar_bg",
        get_or(
            variables,
            "playback-bar-bg",
            or_fallback(surface, "#1a1a1a")?,
        )?,
    )?;
    fields.set_item(
        "active_tab",
        get_or(variables, "active-tab", or_fallback(foreground, "#ffffff")?)?,
    )?;
    fields.set_item(
        "inactive_tab",
        get_or(
            variables,
            "inactive-tab",
            or_fallback(secondary, "#999999")?,
        )?,
    )?;
    fields.set_item(
        "selected_item",
        get_or(variables, "selected-item", or_fallback(surface, "#2a2a2a")?)?,
    )?;
    fields.set_item(
        "progress_filled",
        get_or(variables, "progress-filled", primary.clone())?,
    )?;
    fields.set_item(
        "progress_empty",
        get_or(
            variables,
            "progress-empty",
            or_fallback(surface, "#555555")?,
        )?,
    )?;
    fields.set_item(
        "lyrics_played",
        get_or(
            variables,
            "lyrics-played",
            or_fallback(secondary, "#999999")?,
        )?,
    )?;
    // The default side runs the raw chain — `t.accent or t.primary or
    // DEFAULT_LYRIC_CURRENT` — note it is the raw accent and primary
    // here, not the `#ff4e45`/or-fallback values the fields above took.
    fields.set_item(
        "lyrics_current",
        get_or(
            variables,
            "lyrics-current",
            if accent.is_truthy()? {
                accent.clone()
            } else if primary.is_truthy()? {
                primary.clone()
            } else {
                PyString::new(py, "#ff4e45").into_any()
            },
        )?,
    )?;
    fields.set_item(
        "lyrics_upcoming",
        get_or(
            variables,
            "lyrics-upcoming",
            or_fallback(foreground, "#aaaaaa")?,
        )?,
    )?;

    Ok(fields)
}

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
fn extract_playlist_id(url: &str) -> String {
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
fn parse_spotipy_item<'py>(
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

// ---------------------------------------------------------------------------
// Settings TOML value formatting
// ---------------------------------------------------------------------------

/// The `json.dumps(value, ensure_ascii=False)` string escaping behind
/// `_format_toml_value`'s str branch in `config/settings.py`, with the
/// reference's post-`.replace("\x7f", "\\u007f")` folded in (DEL can
/// appear only inside the payload, so per-code-point is equivalent):
/// the two always-escaped characters, the five short control forms,
/// every other code point below 0x20 its lowercase four-hex `\u00XX`
/// form, and everything else — DEL included, the whole non-ASCII range
/// included — passing through literally.
fn toml_json_escape_into(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0a}' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0d}' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => json_push_u_escape(out, c as u32),
            '\u{7f}' => out.push_str("\\u007f"),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Render one settings value as a TOML literal. Mirrors
/// `config/settings.py`'s `_format_toml_value` match, arm for arm and
/// in the reference's order: `bool` before `int` (bool is an int
/// subclass — True renders `true`, never `1`), `int` through
/// `str(value)`, `str` through `json.dumps(value, ensure_ascii=False)`
/// plus the DEL escape, `list` as the bracketed comma-space join of the
/// recursively rendered elements (an empty list is `[]`), and anything
/// else through `repr(value)` — the str and repr renders running
/// through CPython's own operations, so int/str subclasses and exotic
/// objects render exactly as the reference does.
///
/// A string carrying lone surrogates — which Rust strings cannot spell,
/// and which only a programmatically-built settings field could hold,
/// a TOML file being valid Unicode — delegates its whole branch to
/// CPython's own `json.dumps` (the reference implementation itself),
/// so even that corner answers identically.
#[pyfunction]
fn format_toml_value<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyString>> {
    // bool() — first, so True/False never render as 1/0.
    if value.is_instance_of::<PyBool>() {
        return Ok(PyString::new(
            py,
            if value.is_truthy()? { "true" } else { "false" },
        ));
    }
    // int() — str(value), the interpreter's own render.
    if value.is_instance_of::<PyInt>() {
        return value.str();
    }
    // str() — json.dumps(ensure_ascii=False) + the DEL escape.
    if let Ok(s) = value.downcast::<PyString>() {
        if let Ok(text) = s.to_str() {
            let mut out = String::with_capacity(text.len() + 2);
            toml_json_escape_into(&mut out, text);
            return Ok(PyString::new(py, &out));
        }
        // Lone surrogates: the reference's own serialiser, verbatim.
        let kwargs = PyDict::new(py);
        kwargs.set_item("ensure_ascii", false)?;
        let dumped = py
            .import("json")?
            .getattr("dumps")?
            .call((s,), Some(&kwargs))?;
        let replaced = dumped.call_method1("replace", ("\u{7f}", "\\u007f"))?;
        return Ok(replaced.downcast_into::<PyString>()?);
    }
    // list() — "[a, b]" over the recursive renders.
    if let Ok(list) = value.downcast::<PyList>() {
        let mut items = String::new();
        let mut rendered: Vec<Bound<'py, PyString>> = Vec::with_capacity(list.len());
        let mut all_text = true;
        for (i, element) in list.iter().enumerate() {
            let r = format_toml_value(py, &element)?;
            if i > 0 {
                items.push_str(", ");
            }
            match r.to_str() {
                Ok(text) => items.push_str(text),
                Err(_) => all_text = false,
            }
            rendered.push(r);
        }
        if all_text {
            return Ok(PyString::new(py, &format!("[{items}]")));
        }
        // An element's rendering carries a lone surrogate: join through
        // CPython so the result carries it too.
        let sep = PyString::new(py, ", ");
        let joined = sep.call_method1("join", (PyList::new(py, rendered)?,))?;
        let parts = [
            PyString::new(py, "[").into_any(),
            joined,
            PyString::new(py, "]").into_any(),
        ];
        let out = PyString::new(py, "")
            .call_method1("join", (PyList::new(py, parts)?,))?
            .downcast_into::<PyString>()?;
        return Ok(out);
    }
    // The catch-all: repr(value).
    value.repr()
}

// ---------------------------------------------------------------------------
// Keymap load plan
// ---------------------------------------------------------------------------

/// `parse_key_sequence` — `tuple(raw.strip().split())` with CPython's
/// whitespace set (see `py_isspace`) on both the strip and the split.
fn parse_key_sequence_py<'py>(py: Python<'py>, raw: &str) -> PyResult<Bound<'py, PyTuple>> {
    let parts: Vec<Bound<'py, PyString>> = py_split_none(py_strip(raw))
        .into_iter()
        .map(|part| PyString::new(py, part))
        .collect();
    PyTuple::new(py, parts)
}

/// Build the load plan behind `KeyMap._load_from_dict` — the section
/// walk, action validation and binding classification of the
/// keymap.toml merge, with the mutations themselves (and the native
/// matcher invalidation) applied by the Python caller as it replays the
/// plan.
///
/// Mirrors the reference: every top-level value that is a dict is walked
/// in insertion order (a non-dict value is skipped, exactly the
/// reference's `isinstance` gate), and each section entry resolves to
/// one plan step — `("unknown", name)` when `Action(name)` raises
/// ValueError (the enum's own `__call__` performs the lookup, so the
/// value matching and any TypeError from an unhashable name are the
/// interpreter's own, the latter propagating exactly where the
/// reference's `except ValueError` lets it), `("invalid", name,
/// type_name)` when the binding value is neither a string nor a list of
/// strings (`type(keys).__name__` carried for the warning), or
/// `("bind", action, sequences)` with the `Action(name)` member and
/// each key string split into its `tuple(raw.strip().split())`
/// sequence. The plan preserves the reference's order, so the warnings
/// interleave with the applications that follow them exactly as the
/// reference emits them.
///
/// The data is the tomllib-parsed mapping (top-level dict of section
/// dicts with string keys); key strings are the valid-Unicode strs a
/// TOML file produces.
#[pyfunction]
fn keymap_load_plan<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    action_cls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let data = data
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("keymap data must be a dict"))?;
    let plan = PyList::empty(py);
    for section_value in data.values() {
        // `if not isinstance(section, dict): continue`
        let Ok(section) = section_value.downcast::<PyDict>() else {
            continue;
        };
        for (name, keys) in section.iter() {
            // `try: action = Action(action_name) except ValueError:`
            let action = match action_cls.call1((name.clone(),)) {
                Ok(action) => action,
                Err(err) if err.is_instance_of::<PyValueError>(py) => {
                    let step = PyTuple::new(py, [PyString::new(py, "unknown").into_any(), name])?;
                    plan.append(step)?;
                    continue;
                }
                Err(err) => return Err(err),
            };

            // `_is_valid_binding`: a str, or a list of strs.
            let sequences: Vec<Bound<'py, PyTuple>> = if keys.is_instance_of::<PyString>() {
                let raw = keys.downcast::<PyString>().unwrap();
                vec![parse_key_sequence_py(py, raw.to_str()?)?]
            } else if let Ok(list) = keys.downcast::<PyList>() {
                let mut seqs = Vec::with_capacity(list.len());
                let mut valid = true;
                for element in list.iter() {
                    if !element.is_instance_of::<PyString>() {
                        valid = false;
                        break;
                    }
                    let raw = element.downcast::<PyString>().unwrap();
                    seqs.push(parse_key_sequence_py(py, raw.to_str()?)?);
                }
                if !valid {
                    let step = PyTuple::new(
                        py,
                        [
                            PyString::new(py, "invalid").into_any(),
                            name,
                            keys.get_type().name()?.into_any(),
                        ],
                    )?;
                    plan.append(step)?;
                    continue;
                }
                seqs
            } else {
                let step = PyTuple::new(
                    py,
                    [
                        PyString::new(py, "invalid").into_any(),
                        name,
                        keys.get_type().name()?.into_any(),
                    ],
                )?;
                plan.append(step)?;
                continue;
            };

            let step = PyTuple::new(
                py,
                [
                    PyString::new(py, "bind").into_any(),
                    action,
                    PyList::new(py, sequences)?.into_any(),
                ],
            )?;
            plan.append(step)?;
        }
    }
    Ok(plan)
}

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
fn build_actions<'py>(
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

// ---------------------------------------------------------------------------
// Progress-bar render geometry
// ---------------------------------------------------------------------------

/// Python's two-argument `max(a, b)`: `b` displaces `a` only when
/// strictly greater, so a NaN operand — every comparison false — keeps
/// `a`, exactly CPython's `max(0.0, float("nan")) == 0.0`.
fn py_max2(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's two-argument `min(a, b)`: the `py_max2` mirror.
fn py_min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python's `int()` on the float shapes the bar geometry multiplies out:
/// truncation toward zero (`int(-0.9) == 0`, `int(2.7) == 2`). A NaN —
/// which the reference's own `int()` would raise on, and which mpv's
/// time-pos never produces — reads as 0 so the engine stays total;
/// magnitudes beyond i64 clamp by sign, far past any terminal width or
/// track length.
fn py_int_trunc(f: f64) -> i64 {
    if f.is_nan() {
        0
    } else {
        f as i64
    }
}

/// The progress fraction behind `PlaybackProgress.progress` — the pure
/// core of the property from `ui/widgets/progress_bar.py`: `0.0` for a
/// non-positive duration, otherwise `position / duration` clamped to
/// `[0.0, 1.0]` through CPython's own two-argument `min`/`max` bounds
/// (see `py_min2`/`py_max2`; a NaN duration fails the `<= 0` gate on
/// both backends and flows into the clamp, where the bounds keep it at
/// `0.0` — CPython's answer too).
#[pyfunction]
fn progress_fraction(position: f64, duration: f64) -> f64 {
    if duration <= 0.0 {
        return 0.0;
    }
    py_min2(1.0, py_max2(0.0, position / duration))
}

/// The metrics behind `PlaybackProgress`'s bar layout — the pure core of
/// the widget's `_bar_metrics` from `ui/widgets/progress_bar.py`: the
/// elapsed/total time labels (`format_duration(int(...))`, the pending
/// scroll-seek preview standing in for the position) and the bar width
/// left over once both labels' code-point lengths are reserved from the
/// widget width, floored at 0.
#[pyfunction]
#[pyo3(signature = (position, duration, preview, width))]
fn progress_bar_metrics(
    position: f64,
    duration: f64,
    preview: Option<f64>,
    width: i64,
) -> (String, String, i64) {
    // pos = self.position if self._preview_position is None else
    // self._preview_position
    let pos = preview.unwrap_or(position);
    let time_prefix = format!(" {} ", format_duration(py_int_trunc(pos)));
    let time_suffix = format!(" {} ", format_duration(py_int_trunc(duration)));
    // len() counts code points; format_duration's renders are ASCII-only,
    // so chars().count() is CPython's len() exactly.
    let reserved = (time_prefix.chars().count() + time_suffix.chars().count()) as i64;
    (time_prefix, time_suffix, (width - reserved).max(0))
}

/// The cell computation behind `PlaybackProgress.render` — the pure core
/// of the bar composition from `ui/widgets/progress_bar.py`: the filled
/// and empty cell counts (`int(bar_width * progress)`, truncation toward
/// zero) and the scroll-seek preview's marker column — `-1` when no
/// preview is pending or the bar has no width, otherwise
/// `min(int(bar_width * preview_frac), bar_width - 1)` with
/// `preview_frac` the preview position clamped against the duration
/// exactly like the reference (the `duration > 0` ternary, then
/// CPython's two-argument `min`/`max` bounds).
#[pyfunction]
#[pyo3(signature = (bar_width, progress, preview, duration))]
fn progress_bar_cells(
    bar_width: i64,
    progress: f64,
    preview: Option<f64>,
    duration: f64,
) -> (i64, i64, i64) {
    let filled_count = py_int_trunc(bar_width as f64 * progress);
    let empty_count = bar_width - filled_count;
    let marker_col = match preview {
        // if self._preview_position is not None and bar_width > 0
        Some(preview_position) if bar_width > 0 => {
            // preview_frac = min(1.0, max(0.0, preview / duration))
            //                if duration > 0 else 0.0
            let preview_frac = if duration > 0.0 {
                py_min2(1.0, py_max2(0.0, preview_position / duration))
            } else {
                0.0
            };
            py_int_trunc(bar_width as f64 * preview_frac).min(bar_width - 1)
        }
        _ => -1,
    };
    (filled_count, empty_count, marker_col)
}

/// The click-to-seek mapping behind `PlaybackProgress`'s click handling —
/// the pure core of the widget's `_x_to_seconds` from
/// `ui/widgets/progress_bar.py`: the clicked column, translated through
/// the same metrics the render uses (the time prefix eats the leading
/// columns), clamped to `[0, bar_width - 1]` and scaled to
/// `fraction * duration`. `None` — no seek — for a non-positive duration
/// or a bar with no width, exactly the reference's gates.
#[pyfunction]
#[pyo3(signature = (x, position, duration, preview, width))]
fn progress_x_to_seconds(
    x: i64,
    position: f64,
    duration: f64,
    preview: Option<f64>,
    width: i64,
) -> Option<f64> {
    // if self.duration <= 0: return None
    if duration <= 0.0 {
        return None;
    }
    let (time_prefix, _time_suffix, bar_width) =
        progress_bar_metrics(position, duration, preview, width);
    // if bar_width <= 0: return None
    if bar_width <= 0 {
        return None;
    }
    // prefix_len = len(f" {format_duration(int(pos))} ") — the prefix's
    // own code-point length.
    let prefix_len = time_prefix.chars().count() as i64;
    let mut bar_x = x - prefix_len;
    if bar_x < 0 {
        bar_x = 0;
    }
    if bar_x >= bar_width {
        bar_x = bar_width - 1;
    }
    // fraction = bar_x / bar_width — true division, then the scale.
    Some((bar_x as f64 / bar_width as f64) * duration)
}

// ---------------------------------------------------------------------------
// Synced-lyrics LRC parser
// ---------------------------------------------------------------------------

/// The LRC timestamp shape the synced-lyrics parser strips: `[mm:ss]`,
/// `[mm:ss.f]`, `[mm:ss.ff]` or `[mm:ss.fff]` — 1–2 digits, a colon,
/// exactly 2 digits, an optional `.` plus 1–3 digits. Both `re` and the
/// regex crate read `\d` as Unicode Nd (any script's decimal digits),
/// so exotic-script digits parse identically on both backends — and
/// `int()` reads them too (see `nd_digit_value`).
static LRC_TIMESTAMP_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]").expect("valid LRC timestamp regex")
});

/// The same pattern anchored — the `re.match` half of the reference,
/// which accepts a timestamp only at the (stripped) line's start.
static LRC_TIMESTAMP_START_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]").expect("valid LRC start regex")
});

/// `int()` over a run of Nd digits the regex has already guaranteed —
/// the only characters the pattern's digit groups can hold. Saturating,
/// far above anything the 1–3 digit groups can spell.
fn nd_run_value(digits: &str) -> i64 {
    let mut acc: i64 = 0;
    for c in digits.chars() {
        if let Some(d) = nd_digit_value(c) {
            acc = acc.saturating_mul(10).saturating_add(i64::from(d));
        }
    }
    acc
}

/// The synced-lyrics parse behind `LyricsSidebar`'s string-lyrics path —
/// the pure core of `_parse_synced_lyrics` from
/// `ui/sidebars/lyrics_sidebar.py`. Each line of *raw* (CPython's
/// `splitlines` boundaries — see `py_splitlines`) whose stripped form
/// starts with an LRC timestamp contributes one `(timestamp, text)`
/// pair: minutes and seconds through `int()`, the fractional part as
/// `int(digits) / 10 ** len(digits)` (`.5`, `.50` and `.500` are all a
/// half), and the text the line becomes once every timestamp-shaped
/// span is stripped from the ORIGINAL line and the remainder
/// `strip()`ped. The pairs come back stably sorted by timestamp —
/// equal timestamps keep their line order, like the reference's
/// `list.sort(key=...)`.
#[pyfunction]
fn parse_synced_lyrics(raw: &str) -> Vec<(f64, String)> {
    let mut lines: Vec<(f64, String)> = Vec::new();
    for line in py_splitlines(raw) {
        let stripped = python_strip(&line);
        let Some(m) = LRC_TIMESTAMP_START_RE.captures(stripped) else {
            continue;
        };
        let minutes = nd_run_value(m.get(1).map(|g| g.as_str()).unwrap_or(""));
        let seconds = nd_run_value(m.get(2).map(|g| g.as_str()).unwrap_or(""));
        let frac = match m.get(3) {
            // int(centis) / 10 ** len(centis) — len() counts code
            // points, each Nd digit one.
            Some(centis) => {
                let run = centis.as_str();
                let digits = run.chars().count();
                nd_run_value(run) as f64 / 10f64.powi(digits as i32)
            }
            None => 0.0,
        };
        let ts = (minutes * 60 + seconds) as f64 + frac;
        let text = python_strip(&LRC_TIMESTAMP_RE.replace_all(&line, "")).to_string();
        lines.push((ts, text));
    }
    // list.sort(key=lambda x: x[0]) — stable, and every timestamp is
    // finite (minutes/seconds are 1–2 digit fields), so total_cmp is
    // the reference's ordering exactly.
    lines.sort_by(|a, b| a.0.total_cmp(&b.0));
    lines
}

/// The query-params assembly behind the LRCLIB lookup — the pure core
/// of `get_synced_lyrics`' request build in `services/lrclib.py`.
///
/// Mirrors the pure-Python transcription (`_query_params`): the fresh
/// dict with `track_name` (the sanitized title, passed through as the
/// object it is) and `artist_name` (the artist, likewise) in the
/// reference's insertion order, then — only when *duration_seconds* is
/// not `None`, an identity test so a falsy-but-present `0` still
/// travels — the `duration` key set to
/// `str(int(duration_seconds))`, the `int()` through the interpreter's
/// own (so a float truncates, a bool coerces, a numeric string parses
/// and a bad one raises `int()`'s own `ValueError`) and the `str()`
/// through the object's own render. The title sanitizer itself is the
/// already-ported formatting engine and stays at the caller; the
/// `requests` fetch stays Python-side.
#[pyfunction]
fn lrclib_query_params<'py>(
    py: Python<'py>,
    clean_title: &Bound<'py, PyAny>,
    artist: &Bound<'py, PyAny>,
    duration_seconds: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let params = PyDict::new(py);
    params.set_item("track_name", clean_title)?;
    params.set_item("artist_name", artist)?;
    // if duration_seconds is not None — the identity gate, never
    // truthiness (a 0 duration still travels).
    if !duration_seconds.is_none() {
        // str(int(duration_seconds)) — int() through the interpreter's
        // own, str() through the result's own render.
        let coerced = builtin_int(py, duration_seconds)?;
        let rendered = coerced.str()?;
        params.set_item("duration", rendered)?;
    }
    Ok(params)
}

/// The synced-lyrics extraction behind the LRCLIB response — the pure
/// core of `get_synced_lyrics`' parse step in `services/lrclib.py`.
///
/// Mirrors the pure-Python transcription (`_synced_lyrics`):
/// `data.get("syncedLyrics")` — the member read through the mapping's
/// own `get`, so a dict subclass override steers both backends
/// identically and a payload without `get` raises the call's own
/// `AttributeError` — then the `or None` truthiness gate: a falsy
/// member (the empty string, `None`, anything falsy a stub might
/// answer) becomes `None`, a truthy one passes through as the object
/// it is. The `resp.json()` decode stays Python-side (requests' own
/// decoder and its exception shapes).
#[pyfunction]
fn lrclib_synced_lyrics<'py>(data: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // data.get("syncedLyrics") — through the mapping's own get.
    let value = data.call_method1("get", ("syncedLyrics",))?;
    // `or None` — Python truthiness decides.
    if value.is_truthy()? {
        Ok(value)
    } else {
        Ok(data.py().None().into_bound(data.py()))
    }
}

// ---------------------------------------------------------------------------
// Update-check version comparison (mirrors services/update_check.py)
// ---------------------------------------------------------------------------

/// The strictly-newer PEP 440 verdict behind `check_for_update` from
/// `services/update_check.py` — the pure core of `_is_newer`.
///
/// Mirrors the pure-Python reference: `Version(latest) > Version(current)`
/// with only `InvalidVersion` caught (returning `False`). The parse and
/// the comparison are the shared PEP 440 machinery the crash-staleness
/// engine already uses (`parse_pep440_err` → the `Pep440Key` ordering),
/// so every accept/reject and every ordering decision the reference makes
/// comes free — epochs, trailing-zero releases, the pre/post/dev slot
/// tuples, local segments.
///
/// The two failure kinds differ exactly as the reference's single
/// `except InvalidVersion` makes them:
///
/// - a grammar rejection is `InvalidVersion` — the reference catches it
///   and returns `False`, so the engine answers `Ok(false)` directly;
/// - a numeric component past CPython's int-conversion digit limit is
///   the plain `ValueError` the constructor's own `int()` raises (NOT an
///   `InvalidVersion`) — the reference lets it propagate out of
///   `_is_newer` into `check_for_update`'s caller, so the engine raises
///   `PyValueError` with CPython's own message shape and the binding
///   passes it through untouched.
///
/// `Version(latest)` evaluates before `Version(current)` in the
/// reference's expression, so a bad *latest* never even parses *current*:
/// the engine mirrors the order (an over-limit latest raises before the
/// current string is looked at, an invalid latest short-circuits to
/// `Ok(false)` the same way).
///
/// *admit_unicode_folds* steers the PEP 440 grammar to the shape the
/// installed `packaging` compiled — the Python side probes it once
/// (`Version("1.0+ſ")`: parseable through packaging 26.0, InvalidVersion
/// from 26.1's `(?a:...)` scoping on) and passes the verdict here, so
/// the engine tracks the reference the fallback itself calls, on every
/// `packaging` the dependency range (`>=21.0`, uncapped) admits.
#[pyfunction]
#[pyo3(signature = (latest, current, admit_unicode_folds))]
fn pep440_is_newer(latest: &str, current: &str, admit_unicode_folds: bool) -> PyResult<bool> {
    let latest_key = match parse_pep440_err(latest, admit_unicode_folds) {
        Ok(key) => key,
        // except InvalidVersion: return False — the grammar's rejection
        // never reaches the comparison.
        Err(Pep440Err::Invalid) => return Ok(false),
        // The plain ValueError CPython's int() raises for an over-limit
        // component — not an InvalidVersion, so it propagates.
        Err(Pep440Err::IntLimit) => return Err(int_limit_error(latest)),
    };
    let current_key = match parse_pep440_err(current, admit_unicode_folds) {
        Ok(key) => key,
        Err(Pep440Err::Invalid) => return Ok(false),
        Err(Pep440Err::IntLimit) => return Err(int_limit_error(current)),
    };
    Ok(latest_key > current_key)
}

/// The `ValueError` CPython's own `int()` raises for a digit run past the
/// conversion limit, in CPython's message shape — the digit count the
/// reference's message reports is the offending operand's longest digit
/// run (the component the constructor was converting when it raised).
fn int_limit_error(version: &str) -> PyErr {
    let digits = version
        .split(|c: char| !c.is_ascii_digit())
        .map(str::len)
        .max()
        .unwrap_or(0);
    PyValueError::new_err(format!(
        "Exceeds the limit ({INT_MAX_STR_DIGITS} digits) for integer string conversion: value \
         has {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
    ))
}

// ---------------------------------------------------------------------------
// Help-page rows engine (mirrors ui/pages/help.py)
// ---------------------------------------------------------------------------

/// The titlecase-letter category (Lt) — the 31 characters whose UnicodeData
/// titlecase field is the mixed-case digraph form (ǅ-style). CPython's
/// `ISTITLE` tests exactly this category, and `str.title()`'s
/// previous-is-cased check admits them (see `py_is_cased`); Rust's
/// `char::is_uppercase`/`is_lowercase` (the Uppercase/Lowercase derived
/// properties: Lu+Other_Uppercase / Ll+Other_Lowercase) do not, so the
/// category is spelled out. The set has been stable across every Unicode
/// version CPython 3.10–3.14 ships.
fn is_titlecase_letter(c: char) -> bool {
    matches!(
        c,
        '\u{01C5}'
            | '\u{01C8}'
            | '\u{01CB}'
            | '\u{01F2}'
            | '\u{1F88}'..='\u{1F8F}'
            | '\u{1F98}'..='\u{1F9F}'
            | '\u{1FA8}'..='\u{1FAF}'
            | '\u{1FBC}'
            | '\u{1FCC}'
            | '\u{1FFC}'
    )
}

/// CPython's "cased" test as `str.title()` applies it: the Uppercase
/// derived property (Lu + Other_Uppercase — Rust's `char::is_uppercase`,
/// which is why Roman numerals and circled letters count) ∪ the Lowercase
/// derived property (Ll + Other_Lowercase — `char::is_lowercase`, which
/// is why modifier letters like `ʰ` and ordinal indicators like `ª`
/// count) ∪ the titlecase letters (Lt — see `is_titlecase_letter`).
fn py_is_cased(c: char) -> bool {
    c.is_uppercase() || c.is_lowercase() || is_titlecase_letter(c)
}

/// The Case_Ignorable property: characters a final-sigma scan steps over.
/// Case_Ignorable = Word_Break ∈ {MidLetter, MidNumLet, Single_Quote} ∪
/// general categories {Mn, Me, Cf, Lm}. The general categories come from
/// the regex crate's Unicode tables (`unicode-gencat`); the word-break
/// characters and the modifier symbols the property also claims are the
/// explicit list (verified against CPython's own tables char by char —
/// every member below is Case_Ignorable on every Unicode release CPython
/// 3.10–3.14 ships). A newly-scripted combining mark the regex tables
/// know but an older CPython does not is the one place the two sides can
/// disagree, and only for characters younger than the interpreter's
/// Unicode — never for the ASCII identifiers the help rows carry.
static CASE_IGNORABLE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^[\u{0027}\u{002e}\u{003a}\u{005e}\u{0060}\u{00a8}\u{00af}\u{00b4}",
        r"\u{00b7}\u{00b8}\u{02c2}\u{02c3}\u{02c4}\u{02c5}\u{02d2}\u{02d3}",
        r"\u{02d4}\u{02d5}\u{02d6}\u{02d7}\u{02d8}\u{02d9}\u{02da}\u{02db}",
        r"\u{02dc}\u{02dd}\u{02de}\u{02df}\u{02e5}\u{02e6}\u{02e7}\u{02e8}",
        r"\u{02e9}\u{02ea}\u{02eb}\u{02ed}\u{02ef}\u{02f0}\u{02f1}\u{02f2}",
        r"\u{02f3}\u{02f4}\u{02f5}\u{02f6}\u{02f7}\u{02f8}\u{02f9}\u{02fa}",
        r"\u{02fb}\u{02fc}\u{02fd}\u{02fe}\u{02ff}\u{0375}\u{0384}\u{0385}",
        r"\u{0387}\u{055f}\u{05f4}\u{0888}\u{1fbd}\u{1fbf}\u{1fc0}\u{1fc1}",
        r"\u{1fcd}\u{1fce}\u{1fcf}\u{1fdd}\u{1fde}\u{1fdf}\u{1fed}\u{1fee}",
        r"\u{1fef}\u{1ffd}\u{1ffe}\u{2018}\u{2019}\u{2024}\u{2027}\u{309b}",
        r"\u{309c}\u{a700}\u{a701}\u{a702}\u{a703}\u{a704}\u{a705}\u{a706}",
        r"\u{a707}\u{a708}\u{a709}\u{a70a}\u{a70b}\u{a70c}\u{a70d}\u{a70e}",
        r"\u{a70f}\u{a710}\u{a711}\u{a712}\u{a713}\u{a714}\u{a715}\u{a716}",
        r"\u{a720}\u{a721}\u{a789}\u{a78a}\u{ab5b}\u{ab6a}\u{ab6b}\u{fbb2}",
        r"\u{fbb3}\u{fbb4}\u{fbb5}\u{fbb6}\u{fbb7}\u{fbb8}\u{fbb9}\u{fbba}",
        r"\u{fbbb}\u{fbbc}\u{fbbd}\u{fbbe}\u{fbbf}\u{fbc0}\u{fbc1}\u{fbc2}",
        r"\u{fe13}\u{fe52}\u{fe55}\u{ff07}\u{ff0e}\u{ff1a}\u{ff3e}\u{ff40}",
        r"\u{ffe3}\u{1f3fb}\u{1f3fc}\u{1f3fd}\u{1f3fe}\u{1f3ff}",
        r"\p{Mn}\p{Me}\p{Cf}\p{Lm}]$",
    ))
    .expect("valid case-ignorable class")
});

/// Case_Ignorable for a single character (see `CASE_IGNORABLE_RE`).
fn case_ignorable(c: char) -> bool {
    let mut buf = [0u8; 4];
    CASE_IGNORABLE_RE.is_match(c.encode_utf8(&mut buf))
}

/// Every code point whose titlecase mapping differs from its uppercase
/// mapping — the table `py_title` needs, because Rust's stable `std` has
/// no `to_titlecase`. For every character NOT in this table the titlecase
/// mapping equals the uppercase mapping (verified exhaustively against
/// CPython: 135 divergences), which Rust's `char::to_uppercase` — the
/// full Unicode uppercase mapping, multi-character expansions included —
/// implements with the same mappings CPython does.
///
/// The members, all stable since the Unicode versions CPython 3.10–3.14
/// ship: the sharp s (`ß` → `Ss`, the SpecialCasing title entry); the
/// Latin digraph families (`Ǆ`/`ǅ`/`ǆ` and their LJ/NJ/DZ partners —
/// every spelling titles onto the mixed-case form); the Armenian
/// ligatures (`ﬗ` → `Եւ`, `ﬓ`–`ﬗ` → mixed-case pairs); the Latin
/// ligatures (`ﬀ` → `Ff` … `ﬆ` → `St`); the Georgian mkhedruli letters
/// (title is the character itself while uppercase is the Unicode 11
/// Mtavruli capital — `ა` stays `ა`, unlike its `.upper()`); and the
/// Greek ypogegrammeni family (`ᾀ` → `ᾈ`-shaped titles, the ᾳ/ῃ/ῳ
/// trio → `ᾼ`-shaped single-char titles, and the circumflex-perispomeni
/// trio → two- and three-character titles).
static TITLE_DIVERGENCES: &[(char, &str)] = &[
    ('\u{00DF}', "Ss"),
    ('\u{01C4}', "\u{01C5}"),
    ('\u{01C5}', "\u{01C5}"),
    ('\u{01C6}', "\u{01C5}"),
    ('\u{01C7}', "\u{01C8}"),
    ('\u{01C8}', "\u{01C8}"),
    ('\u{01C9}', "\u{01C8}"),
    ('\u{01CA}', "\u{01CB}"),
    ('\u{01CB}', "\u{01CB}"),
    ('\u{01CC}', "\u{01CB}"),
    ('\u{01F1}', "\u{01F2}"),
    ('\u{01F2}', "\u{01F2}"),
    ('\u{01F3}', "\u{01F2}"),
    ('\u{0587}', "\u{0535}\u{0582}"),
    ('\u{10D0}', "\u{10D0}"),
    ('\u{10D1}', "\u{10D1}"),
    ('\u{10D2}', "\u{10D2}"),
    ('\u{10D3}', "\u{10D3}"),
    ('\u{10D4}', "\u{10D4}"),
    ('\u{10D5}', "\u{10D5}"),
    ('\u{10D6}', "\u{10D6}"),
    ('\u{10D7}', "\u{10D7}"),
    ('\u{10D8}', "\u{10D8}"),
    ('\u{10D9}', "\u{10D9}"),
    ('\u{10DA}', "\u{10DA}"),
    ('\u{10DB}', "\u{10DB}"),
    ('\u{10DC}', "\u{10DC}"),
    ('\u{10DD}', "\u{10DD}"),
    ('\u{10DE}', "\u{10DE}"),
    ('\u{10DF}', "\u{10DF}"),
    ('\u{10E0}', "\u{10E0}"),
    ('\u{10E1}', "\u{10E1}"),
    ('\u{10E2}', "\u{10E2}"),
    ('\u{10E3}', "\u{10E3}"),
    ('\u{10E4}', "\u{10E4}"),
    ('\u{10E5}', "\u{10E5}"),
    ('\u{10E6}', "\u{10E6}"),
    ('\u{10E7}', "\u{10E7}"),
    ('\u{10E8}', "\u{10E8}"),
    ('\u{10E9}', "\u{10E9}"),
    ('\u{10EA}', "\u{10EA}"),
    ('\u{10EB}', "\u{10EB}"),
    ('\u{10EC}', "\u{10EC}"),
    ('\u{10ED}', "\u{10ED}"),
    ('\u{10EE}', "\u{10EE}"),
    ('\u{10EF}', "\u{10EF}"),
    ('\u{10F0}', "\u{10F0}"),
    ('\u{10F1}', "\u{10F1}"),
    ('\u{10F2}', "\u{10F2}"),
    ('\u{10F3}', "\u{10F3}"),
    ('\u{10F4}', "\u{10F4}"),
    ('\u{10F5}', "\u{10F5}"),
    ('\u{10F6}', "\u{10F6}"),
    ('\u{10F7}', "\u{10F7}"),
    ('\u{10F8}', "\u{10F8}"),
    ('\u{10F9}', "\u{10F9}"),
    ('\u{10FA}', "\u{10FA}"),
    ('\u{10FD}', "\u{10FD}"),
    ('\u{10FE}', "\u{10FE}"),
    ('\u{10FF}', "\u{10FF}"),
    ('\u{1F80}', "\u{1F88}"),
    ('\u{1F81}', "\u{1F89}"),
    ('\u{1F82}', "\u{1F8A}"),
    ('\u{1F83}', "\u{1F8B}"),
    ('\u{1F84}', "\u{1F8C}"),
    ('\u{1F85}', "\u{1F8D}"),
    ('\u{1F86}', "\u{1F8E}"),
    ('\u{1F87}', "\u{1F8F}"),
    ('\u{1F88}', "\u{1F88}"),
    ('\u{1F89}', "\u{1F89}"),
    ('\u{1F8A}', "\u{1F8A}"),
    ('\u{1F8B}', "\u{1F8B}"),
    ('\u{1F8C}', "\u{1F8C}"),
    ('\u{1F8D}', "\u{1F8D}"),
    ('\u{1F8E}', "\u{1F8E}"),
    ('\u{1F8F}', "\u{1F8F}"),
    ('\u{1F90}', "\u{1F98}"),
    ('\u{1F91}', "\u{1F99}"),
    ('\u{1F92}', "\u{1F9A}"),
    ('\u{1F93}', "\u{1F9B}"),
    ('\u{1F94}', "\u{1F9C}"),
    ('\u{1F95}', "\u{1F9D}"),
    ('\u{1F96}', "\u{1F9E}"),
    ('\u{1F97}', "\u{1F9F}"),
    ('\u{1F98}', "\u{1F98}"),
    ('\u{1F99}', "\u{1F99}"),
    ('\u{1F9A}', "\u{1F9A}"),
    ('\u{1F9B}', "\u{1F9B}"),
    ('\u{1F9C}', "\u{1F9C}"),
    ('\u{1F9D}', "\u{1F9D}"),
    ('\u{1F9E}', "\u{1F9E}"),
    ('\u{1F9F}', "\u{1F9F}"),
    ('\u{1FA0}', "\u{1FA8}"),
    ('\u{1FA1}', "\u{1FA9}"),
    ('\u{1FA2}', "\u{1FAA}"),
    ('\u{1FA3}', "\u{1FAB}"),
    ('\u{1FA4}', "\u{1FAC}"),
    ('\u{1FA5}', "\u{1FAD}"),
    ('\u{1FA6}', "\u{1FAE}"),
    ('\u{1FA7}', "\u{1FAF}"),
    ('\u{1FA8}', "\u{1FA8}"),
    ('\u{1FA9}', "\u{1FA9}"),
    ('\u{1FAA}', "\u{1FAA}"),
    ('\u{1FAB}', "\u{1FAB}"),
    ('\u{1FAC}', "\u{1FAC}"),
    ('\u{1FAD}', "\u{1FAD}"),
    ('\u{1FAE}', "\u{1FAE}"),
    ('\u{1FAF}', "\u{1FAF}"),
    ('\u{1FB2}', "\u{1FBA}\u{0345}"),
    ('\u{1FB3}', "\u{1FBC}"),
    ('\u{1FB4}', "\u{0386}\u{0345}"),
    ('\u{1FB7}', "\u{0391}\u{0342}\u{0345}"),
    ('\u{1FBC}', "\u{1FBC}"),
    ('\u{1FC2}', "\u{1FCA}\u{0345}"),
    ('\u{1FC3}', "\u{1FCC}"),
    ('\u{1FC4}', "\u{0389}\u{0345}"),
    ('\u{1FC7}', "\u{0397}\u{0342}\u{0345}"),
    ('\u{1FCC}', "\u{1FCC}"),
    ('\u{1FF2}', "\u{1FFA}\u{0345}"),
    ('\u{1FF3}', "\u{1FFC}"),
    ('\u{1FF4}', "\u{038F}\u{0345}"),
    ('\u{1FF7}', "\u{03A9}\u{0342}\u{0345}"),
    ('\u{1FFC}', "\u{1FFC}"),
    ('\u{FB00}', "Ff"),
    ('\u{FB01}', "Fi"),
    ('\u{FB02}', "Fl"),
    ('\u{FB03}', "Ffi"),
    ('\u{FB04}', "Ffl"),
    ('\u{FB05}', "St"),
    ('\u{FB06}', "St"),
    ('\u{FB13}', "\u{0544}\u{0576}"),
    ('\u{FB14}', "\u{0544}\u{0565}"),
    ('\u{FB15}', "\u{0544}\u{056B}"),
    ('\u{FB16}', "\u{054E}\u{0576}"),
    ('\u{FB17}', "\u{0544}\u{056D}"),
];

/// CPython's `str.title()`: the titlecase mapping of a character that
/// follows an uncased one, the full lowercase mapping of one that follows
/// a cased one — "cased" being the property `py_is_cased` tests on the
/// ORIGINAL character, the immediately preceding one (no case-ignorable
/// skipping: a combining mark between two letters breaks the run just
/// like a digit does, so `a◌́Σ` titles the sigma, never lowers it).
///
/// Two properties CPython's tables carry that per-character Rust helpers
/// alone do not, both reproduced exactly:
///
/// - the titlecase mapping is the TITLECASE mapping — `ß` titles to
///   `Ss` (not the `SS` its uppercase gives), `ǳ` to `ǲ`, `ﬀ` to `Ff`,
///   Georgian mkhedruli to itself (not its Mtavruli uppercase). The
///   divergences from uppercase are `TITLE_DIVERGENCES`; everywhere else
///   title equals uppercase, which `char::to_uppercase` (the full
///   mapping, multi-character expansions like `ǰ` → `J̌` included)
///   supplies;
/// - the lowercase path applies the final-sigma rule: a capital sigma
///   lowered with no cased character after it (skipping the
///   Case_Ignorable set — see `case_ignorable`) becomes `ς`, one with a
///   cased character ahead stays `σ` — `ΑΣΣ`.title() is `Ασς` exactly.
///   Sigma is the only character whose lowercase mapping is contextual,
///   so every other character goes through `char::to_lowercase` (the
///   full mapping — `İ` lowers to `i` + combining dot above).
///
/// The production input is the help page's action values —
/// `_`-separated ASCII identifiers (`play_pause`.title() →
/// `Play_Pause`... after the caller's `_`→space replacement,
/// `Play Pause`) — but the engine is a faithful `str.title()` for any
/// string, pinned by the parity zoo.
fn py_title(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut prev_cased = false;
    for (i, &ch) in chars.iter().enumerate() {
        if prev_cased {
            if ch == '\u{03A3}' {
                // Final sigma: preceded by a cased character (the
                // lowercase path guarantees it), ς unless a cased
                // character follows once the ignorable set is skipped.
                let mut j = i + 1;
                let mut followed_by_cased = false;
                while j < chars.len() {
                    if case_ignorable(chars[j]) {
                        j += 1;
                        continue;
                    }
                    followed_by_cased = py_is_cased(chars[j]);
                    break;
                }
                out.push(if followed_by_cased {
                    '\u{03C3}'
                } else {
                    '\u{03C2}'
                });
            } else {
                // The full lowercase mapping (İ → i + U+0307 included).
                out.extend(ch.to_lowercase());
            }
        } else {
            // The titlecase mapping: the divergence table first, the
            // uppercase mapping (identical to title everywhere else)
            // otherwise.
            match TITLE_DIVERGENCES.binary_search_by(|probe| probe.0.cmp(&ch)) {
                Ok(idx) => out.push_str(TITLE_DIVERGENCES[idx].1),
                Err(_) => out.extend(ch.to_uppercase()),
            }
        }
        // Casedness of the ORIGINAL character decides the next step —
        // not of anything the mapping emitted.
        prev_cased = py_is_cased(ch);
    }
    out
}

/// The all-rows list behind `HelpPage._build_rows` — the pure core of
/// the help page's row assembly.
///
/// Mirrors the pure-Python reference: each spec entry
/// `(category, action_value, description, key_strs)` becomes the row
/// `(category, name, keys_str, description)` — the name through the
/// `action_value.replace("_", " ").title()` pipeline (`py_title`, the
/// `_format_action_name` body), the keys as the `" / "` join of the
/// formatted key strings, or `"(unbound)"` when the action has no
/// binding. The keymap walk that produces the spec (the
/// `get_keys_for_action`/`format_key` calls over `ACTION_CATEGORIES`
/// and `ACTION_DESCRIPTIONS`) stays on the Python side next to the
/// objects it reads; the description arrives already defaulted (`""`
/// for an undocumented action), exactly as the reference's
/// `ACTION_DESCRIPTIONS.get(action, "")` produced it.
#[pyfunction]
fn help_page_rows(spec: &Bound<'_, PyList>) -> PyResult<Vec<(String, String, String, String)>> {
    let mut rows: Vec<(String, String, String, String)> = Vec::with_capacity(spec.len());
    for entry in spec.iter() {
        let (category, action_value, description, key_strs): (String, String, String, Vec<String>) =
            entry.extract()?;
        let name = py_title(&action_value.replace('_', " "));
        let keys_str = if key_strs.is_empty() {
            "(unbound)".to_string()
        } else {
            key_strs.join(" / ")
        };
        rows.push((category, name, keys_str, description));
    }
    Ok(rows)
}

/// One visible-row emission of the help walk: `(is_header, category,
/// name, keys, description)` — headers carry `(True, category, "", "",
/// "")`, data rows the row itself.
type HelpVisibleRow = (bool, String, String, String, String);

/// The visible-row walk behind `HelpPage._populate_table` — the pure
/// core of the help page's filtering.
///
/// Mirrors the pure-Python reference: the needle is
/// `filter_text.lower().strip()` (lowercased FIRST, then stripped — the
/// opposite order of the playlist picker's query), an empty needle
/// filters nothing, and a non-empty one matches a row only when it is a
/// substring of the lowercased `"{category} {name} {keys} {desc}"`
/// haystack. A category header is emitted whenever the category changes
/// — starting from `""` (so the first category always earns one, and an
/// empty-string category never does), and re-earned if a category
/// recurs after another (a row list the reference would walk the same
/// way, whatever its caller produced). Headers come back as
/// `(True, category, "", "", "")` and data rows as
/// `(False, category, name, keys, description)`; the Textual markup
/// (`[b]{category}[/b]`) and the DataTable calls stay on the Python side.
#[pyfunction]
#[pyo3(signature = (all_rows, filter_text))]
fn help_page_visible_rows(
    all_rows: &Bound<'_, PyList>,
    filter_text: &str,
) -> PyResult<Vec<HelpVisibleRow>> {
    // filter_text.lower().strip() — lower first, then strip.
    let lowered = python_lower(filter_text);
    let needle = python_strip(&lowered);
    let mut out: Vec<HelpVisibleRow> = Vec::new();
    let mut current_category = String::new();
    for row in all_rows.iter() {
        let (category, name, keys, desc): (String, String, String, String) = row.extract()?;
        // `if needle:` — an empty (or whitespace-only) needle filters
        // nothing; every row passes.
        if !needle.is_empty() {
            let haystack = python_lower(&format!("{category} {name} {keys} {desc}"));
            if !haystack.contains(needle) {
                continue;
            }
        }
        if category != current_category {
            current_category.clone_from(&category);
            out.push((
                true,
                category.clone(),
                String::new(),
                String::new(),
                String::new(),
            ));
        }
        out.push((false, category, name, keys, desc));
    }
    Ok(out)
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
fn playlist_record_recent<'py>(
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
fn playlist_picker_sort<'py>(
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
fn playlist_picker_rows<'py>(
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
fn playlist_picker_count_label(count: usize) -> String {
    let plural = if count == 1 { "" } else { "s" };
    format!("{count} playlist{plural}")
}

// ---------------------------------------------------------------------------
// Queue-page view & mutation engines (mirrors ui/pages/queue.py)
// ---------------------------------------------------------------------------

/// The now-playing header's view model — the pure core shared by
/// `QueuePage._update_current_track` and `QueuePage._refresh_queue`.
///
/// Mirrors the pure-Python reference: a falsy track (the `None` a queue
/// with nothing selected reports, an empty dict never occurring in
/// practice) hides the header and answers `None`; otherwise the dict
/// `{"title_label": ..., "artist": ..., "video_id": ...}` where
/// `title_label` is `f"Now Playing: {title}"` — the title read as
/// `track.get("title", "Unknown")` (absent key `"Unknown"`, a present
/// value whatever it is, `str()`'d by the f-string) — `artist` is
/// `track.get("artist", "Unknown")` raw (the object the Label mounts),
/// and `video_id` is `track.get("video_id", "")` raw, the value
/// `set_playing` receives to light the playing row. Field reads resolve
/// exactly like the reference's `.get` calls (see `field_get`); a
/// `.get`-less track reads as field-less where the reference would
/// raise — the engine stays total. The header mounts and the
/// `set_playing` call stay on the Python side.
#[pyfunction(signature = (track=None))]
fn queue_now_playing<'py>(
    py: Python<'py>,
    track: Option<Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    // `if current_track:` — Python truthiness, not an is-None test.
    let track = match track {
        Some(track) if track.is_truthy()? => track,
        _ => return Ok(None),
    };
    let unknown_title = PyString::new(py, "Unknown").into_any();
    let title = field_get(py, &track, "title", Some(unknown_title))?;
    let unknown_artist = PyString::new(py, "Unknown").into_any();
    let artist = field_get(py, &track, "artist", Some(unknown_artist))?;
    let empty_id = PyString::new(py, "").into_any();
    let video_id = field_get(py, &track, "video_id", Some(empty_id))?;

    let out = PyDict::new(py);
    out.set_item("title_label", format!("Now Playing: {}", py_str(&title)?))?;
    out.set_item("artist", artist)?;
    out.set_item("video_id", video_id)?;
    Ok(Some(out))
}

/// The seed-source label's view model — the pure core of
/// `QueuePage._update_queue_source`.
///
/// Mirrors the pure-Python reference: the label stays hidden (`None`)
/// when there are no radio seeds or the `show_queue_source` setting is
/// off; otherwise each seed's title — `seed.get("title", "Unknown")`,
/// absent key `"Unknown"`, a present value whatever it is — feeds both
/// the one-line summary (`", ".join(titles)` for up to three seeds,
/// `"{titles[0]}, {titles[1]} + {n} more"` from the fourth on) and the
/// tooltip's bulleted `"Generated from:"` block. Titles render through
/// `str()`: the tooltip's and the four-plus-summary's f-strings always
/// did, and the up-to-three join — which would raise on a non-str title
/// — reads the same rendering, so the engine stays total where the
/// reference would crash on junk seeds. Returned as `(text, tooltip)`;
/// the Static update and the tooltip assignment stay on the Python side.
#[pyfunction]
#[pyo3(signature = (seeds, show))]
fn queue_source_view(
    py: Python<'_>,
    seeds: Option<Bound<'_, PyList>>,
    show: bool,
) -> PyResult<Option<(String, String)>> {
    // `if seeds and show:` — an empty list hides the label like None does.
    let seeds = match seeds {
        Some(seeds) if show && !seeds.is_empty() => seeds,
        _ => return Ok(None),
    };
    let unknown = PyString::new(py, "Unknown").into_any();
    let mut titles: Vec<String> = Vec::with_capacity(seeds.len());
    for seed in seeds.iter() {
        let title = field_get(py, &seed, "title", Some(unknown.clone()))?;
        titles.push(py_str(&title)?);
    }
    let summary = if titles.len() <= 3 {
        titles.join(", ")
    } else {
        format!("{}, {} + {} more", titles[0], titles[1], titles.len() - 2)
    };
    let text = format!("Generated from: {summary}");
    let tooltip = format!(
        "Generated from:\n{}",
        titles
            .iter()
            .map(|title| format!("  \u{2022} {title}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(Some((text, tooltip)))
}

/// The footer bar's text — the pure core of `QueuePage._update_footer`:
/// `f"Tracks: {count}"` over the queue's length, so `"Tracks: 0"` for
/// the emptied queue reads exactly as the reference's f-string renders
/// it.
#[pyfunction]
fn queue_footer_text(count: i64) -> String {
    format!("Tracks: {count}")
}

/// The queue-position decision behind `QueuePage._remove_selected` —
/// the pure half of the queue page's remove mutation.
///
/// Mirrors the pure-Python reference: the highlighted row's
/// original-order index names a removable queue position only when it
/// is present and `0 <= index < queue_length` (a stale table — one the
/// queue shrank underneath without a refresh — refuses rather than
/// removing the wrong entry); `None` means "no removal". The
/// `queue.remove` call, the page refresh and the cursor re-landing stay
/// on the Python side.
#[pyfunction]
#[pyo3(signature = (selected_index, queue_length))]
fn queue_remove_target(selected_index: Option<i64>, queue_length: i64) -> Option<i64> {
    match selected_index {
        Some(index) if (0..queue_length).contains(&index) => Some(index),
        _ => None,
    }
}

/// The cursor re-landing behind `QueuePage._remove_selected`'s refresh —
/// the pure half of the remove mutation's optimistic UI update.
///
/// Mirrors the pure-Python reference: after the removal and the refresh
/// the row that took the removed entry's place in the displayed order
/// gets the cursor — `min(visible_row, row_count - 1)`, the visible row
/// the cursor sat on before the mutation, clamped into the refreshed
/// table's bounds — and `None` (cursor untouched) when the refreshed
/// table has no rows left. The `move_cursor` call stays on the Python
/// side.
#[pyfunction]
#[pyo3(signature = (visible_row, row_count))]
fn queue_post_remove_cursor(visible_row: i64, row_count: i64) -> Option<i64> {
    if row_count <= 0 {
        return None;
    }
    Some(visible_row.min(row_count - 1))
}

/// The reorder destination behind `QueuePage._move_track` — the pure
/// half of the queue page's move mutation.
///
/// Mirrors the pure-Python reference: no highlighted row (`None`) or a
/// destination equal to the source — including the clamped-at-the-bounds
/// no-ops a too-large count produces — answers `None` (nothing moves);
/// otherwise the destination is `max(0, min(from + direction * count,
/// queue_length - 1))`: the count-multiplied queue positions up or
/// down, clamped to the queue bounds so `15J` near the end simply lands
/// at the bottom. The arithmetic saturates where Python's
/// arbitrary-precision ints cannot overflow — the clamp makes the two
/// agree for every input. The `clear_sort` call, the `queue.move` and
/// the page refresh stay on the Python side.
#[pyfunction]
#[pyo3(signature = (from_index, direction, count, queue_length))]
fn queue_move_target(
    from_index: Option<i64>,
    direction: i64,
    count: i64,
    queue_length: i64,
) -> Option<i64> {
    let from = from_index?;
    // max(0, min(from + direction * count, queue_length - 1)).
    let destination = from
        .saturating_add(direction.saturating_mul(count))
        .min(queue_length - 1)
        .max(0);
    if destination == from {
        return None;
    }
    Some(destination)
}

// ---------------------------------------------------------------------------
// Album-art render engines
// ---------------------------------------------------------------------------

/// Python's `str.strip()`: trims the characters CPython's `str.isspace()`
/// accepts from both ends. Rust's `str::trim()` (the Unicode `White_Space`
/// property) would disagree on `\x1c`–`\x1f`, so the acceptance set is
/// `py_isspace`'s, not the property's.
fn py_str_strip(s: &str) -> &str {
    s.trim_matches(|c: char| py_isspace(c))
}

/// The WCAG relative luminance behind `AlbumArt`'s placeholder note-glyph
/// colour pick — the pure core of `_relative_luminance` from
/// `ui/widgets/album_art.py`: `None` for anything that is not a
/// six-digit hex colour (after `strip()` and `lstrip("#")` — every
/// leading `#`, so `##rrggbb` parses too; the length counts code
/// points), otherwise the sRGB linearisation
/// `0.2126 R' + 0.7152 G' + 0.0722 B'` with each channel either
/// `c / 12.92` below the 0.04045 knee or `((c + 0.055) / 1.055) ** 2.4`
/// above it.
///
/// The channel parse mirrors CPython's `int(s, 16)` on the two-character
/// slices: the pair is stripped with `py_str_strip` first (so
/// `"#1 2345"`'s `"1 "` pair parses to 1, exactly the reference), signs
/// are honoured (`"-f"` reads as -15 — the reference's `int()` does too,
/// and the negative channel simply flows into the linearisation), and
/// everything else fails to `None`. With the per-pair strip the engine
/// agrees with CPython on every two-character input: the forms `int()`
/// additionally accepts beyond that (the `0x` prefix, underscores
/// between digits) cannot fit in two characters, and non-ASCII digits
/// are base-10-only in CPython, so base 16 rejects them both sides.
#[pyfunction]
fn wcag_relative_luminance(color: &str) -> Option<f64> {
    let hex_str = py_str_strip(color).trim_start_matches('#');
    let chars: Vec<char> = hex_str.chars().collect();
    if chars.len() != 6 {
        return None;
    }
    let mut channels = [0.0f64; 3];
    for (i, channel) in channels.iter_mut().enumerate() {
        let pair: String = chars[i * 2..i * 2 + 2].iter().collect();
        match i32::from_str_radix(py_str_strip(&pair), 16) {
            Ok(value) => *channel = value as f64 / 255.0,
            Err(_) => return None,
        }
    }
    let mut linear = [0.0f64; 3];
    for (linearized, &c) in linear.iter_mut().zip(channels.iter()) {
        *linearized = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
    }
    Some(0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2])
}

/// The WCAG contrast ratio behind the note-glyph pick — the pure core of
/// `_contrast_ratio` from `ui/widgets/album_art.py`:
/// `(lighter + 0.05) / (darker + 0.05)` with lighter/darker taken
/// through CPython's own two-argument `max`/`min` (see `py_max2` /
/// `py_min2`), so the NaN corners answer exactly the reference's way.
#[pyfunction]
fn wcag_contrast_ratio(luminance_a: f64, luminance_b: f64) -> f64 {
    let lighter = py_max2(luminance_a, luminance_b);
    let darker = py_min2(luminance_a, luminance_b);
    (lighter + 0.05) / (darker + 0.05)
}

/// The note-glyph colour pick behind `AlbumArt._render_placeholder` —
/// the pure core of `_readable_on` from `ui/widgets/album_art.py`: the
/// candidate with the highest contrast ratio against the background,
/// with the first candidate kept whenever the comparison is impossible
/// (the background or any candidate is not a six-digit hex colour) and
/// on ties (only a strictly greater ratio displaces the incumbent, which
/// starts as the first candidate at ratio 0.0 — so the first candidate
/// always scores in, and an equal later ratio never does).
///
/// The candidate list mirrors the reference's `(first, *others)` tuple;
/// the empty list is a caller error the reference cannot even express
/// (its `first` is a required argument) and reads as the empty string.
#[pyfunction]
fn wcag_readable_on(bg_hex: &str, candidates: Vec<String>) -> String {
    let background = wcag_relative_luminance(bg_hex);
    let luminances: Vec<Option<f64>> = candidates
        .iter()
        .map(|candidate| wcag_relative_luminance(candidate))
        .collect();
    let keep_first = || candidates.first().cloned().unwrap_or_default();
    let Some(background) = background else {
        return keep_first();
    };
    if luminances.iter().any(|luminance| luminance.is_none()) || candidates.is_empty() {
        return keep_first();
    }
    let mut best = 0usize;
    let mut best_ratio = 0.0f64;
    for (i, luminance) in luminances.iter().enumerate() {
        let ratio = wcag_contrast_ratio(luminance.unwrap(), background);
        if ratio > best_ratio {
            best = i;
            best_ratio = ratio;
        }
    }
    candidates[best].clone()
}

/// One half-block character cell: the top pixel's RGB followed by the
/// bottom pixel's RGB — the foreground and background of the `▀` glyph
/// the render styles it with.
type HalfBlockCell = (u8, u8, u8, u8, u8, u8);

/// The half-block cell extraction behind `AlbumArt._image_to_half_blocks`
/// — the pure core of the thumbnail render from `ui/widgets/album_art.py`
/// (the pixel walk the reference's `getpixel` loop performs): one
/// `(top_rgb, bottom_rgb)` pair per character cell, row-major, read from
/// the resized image's raw RGB buffer (`PIL.Image.tobytes()` on an
/// RGB-mode image — `getpixel((x, y))` is exactly
/// `bytes[(y * width + x) * 3 : + 3]`, so the buffer walk and the
/// reference's per-pixel walk agree cell for cell). Each cell joins the
/// pixel rows `2 * row` (top) and `2 * row + 1` (bottom) at column
/// `col`.
///
/// A buffer whose length is not exactly `width * height * 2 * 3` raises
/// `ValueError` on both backends — the reference's own length gate. A
/// non-positive width or height reads as no cells once the gate passes
/// (a zero-area buffer), matching the reference's empty `range()` walks.
#[pyfunction]
fn album_art_half_block_cells(
    pixels: Vec<u8>,
    width: i64,
    height: i64,
) -> PyResult<Vec<HalfBlockCell>> {
    // width * height * 2 * 3 in Python's arbitrary-precision ints; in
    // i128 with overflow checked — an overflowing product can never
    // equal a real buffer length, so it reads as the gate's ValueError
    // exactly like the reference's astronomically-large product does.
    let expected = (width as i128)
        .checked_mul(height as i128)
        .and_then(|product| product.checked_mul(6));
    let matches = matches!(expected, Some(e) if e >= 0 && e == pixels.len() as i128);
    if !matches {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "pixel buffer is {} bytes; {}x{} half-block cells need {}",
            pixels.len(),
            width,
            height,
            width
                .checked_mul(height)
                .and_then(|v| v.checked_mul(6))
                .map(|v| v.to_string())
                .unwrap_or_else(|| "more than any buffer".to_string())
        )));
    }
    let pixel_w = width;
    let mut cells = Vec::with_capacity((width.max(0) * height.max(0)).max(0) as usize);
    for row in 0..height.max(0) {
        for col in 0..pixel_w.max(0) {
            let top = ((row * 2) * pixel_w + col) * 3;
            let bottom = ((row * 2 + 1) * pixel_w + col) * 3;
            cells.push((
                pixels[top as usize],
                pixels[top as usize + 1],
                pixels[top as usize + 2],
                pixels[bottom as usize],
                pixels[bottom as usize + 1],
                pixels[bottom as usize + 2],
            ));
        }
    }
    Ok(cells)
}

/// Python's `//` on the shapes the row plans divide: floor division
/// (`-1 // 2 == -1`), which i64 truncation is not.
fn py_floor_div2(a: i64) -> i64 {
    a.div_euclid(2)
}

/// Python's `string * n`: `n <= 0` reads as the empty string.
fn py_repeat(s: &str, n: i64) -> String {
    s.repeat(n.max(0) as usize)
}

/// The placeholder-box row plan behind `AlbumArt._render_placeholder` —
/// the pure core of the widget's no-thumbnail render from
/// `ui/widgets/album_art.py`: per row, the (text, style) segments to
/// append. Row 0 is the lower-half-block strip (`▄` × width, `accent on
/// bg`); the last row is the upper-half-block strip (`▀` × width, the
/// same style) — checked in that order, so a one-row box is all
/// lower-half blocks and a two-row box never shows a middle; the middle
/// row (`height // 2`, floor division) is the note row — `█` ×
/// `(width - 1) // 2` (floor division, so width 0 pads -1 → the empty
/// string, like Python's negative repeat), the `♫` glyph as
/// `bold {note_color} on {accent}`, then `█` × the rest; every other row
/// is `█` × width in the accent. The note colour itself stays
/// Python-side — it is the `wcag_readable_on` pick over the theme's
/// foreground/background.
#[pyfunction]
#[pyo3(signature = (width, height, accent, bg, note_color))]
fn album_art_placeholder_rows(
    width: i64,
    height: i64,
    accent: &str,
    bg: &str,
    note_color: &str,
) -> Vec<Vec<(String, String)>> {
    let mut rows: Vec<Vec<(String, String)>> = Vec::new();
    for row in 0..height.max(0) {
        let segments: Vec<(String, String)> = if row == 0 {
            vec![(py_repeat("\u{2584}", width), format!("{accent} on {bg}"))]
        } else if row == height - 1 {
            vec![(py_repeat("\u{2580}", width), format!("{accent} on {bg}"))]
        } else if row == py_floor_div2(height) {
            let pad = py_floor_div2(width - 1);
            vec![
                (py_repeat("\u{2588}", pad), accent.to_string()),
                (
                    "\u{266b}".to_string(),
                    format!("bold {note_color} on {accent}"),
                ),
                (py_repeat("\u{2588}", width - pad - 1), accent.to_string()),
            ]
        } else {
            vec![(py_repeat("\u{2588}", width), accent.to_string())]
        };
        rows.push(segments);
    }
    rows
}

/// The empty-state row plan behind `AlbumArt._render_empty` — the pure
/// core of the widget's no-track render from
/// `ui/widgets/album_art.py`: per row, the (text, style) segments to
/// append, every one styled with the muted border colour. The middle row
/// (`height // 2`, floor division — row 0 itself for a one-row box) is
/// the note row: `width - 1) // 2` spaces (floor division, so width 0
/// pads -1 → the empty string, like Python's negative repeat), the `♫`
/// glyph, then the remaining spaces; every other row is width spaces.
#[pyfunction]
#[pyo3(signature = (width, height, muted))]
fn album_art_empty_rows(width: i64, height: i64, muted: &str) -> Vec<Vec<(String, String)>> {
    let mut rows: Vec<Vec<(String, String)>> = Vec::new();
    for row in 0..height.max(0) {
        let segments: Vec<(String, String)> = if row == py_floor_div2(height) {
            let pad = py_floor_div2(width - 1);
            vec![
                (py_repeat(" ", pad), muted.to_string()),
                ("\u{266b}".to_string(), muted.to_string()),
                (py_repeat(" ", width - pad - 1), muted.to_string()),
            ]
        } else {
            vec![(py_repeat(" ", width), muted.to_string())]
        };
        rows.push(segments);
    }
    rows
}

// ── LRU bookkeeping engines (shuffle prefs + album-art cache) ───────────

/// CPython's `\w` character set for `str` patterns, as a one-character
/// test: the alphanumeric union behind `str.isalnum()` — the letter
/// categories `L*` and every numeric category `N*` (decimal `Nd`, letter
/// numbers `Nl` like the Roman numerals, and other numerics `No` like
/// the circled digits and vulgar fractions, the sets CPython's
/// ISALPHA/ISDECIMAL/ISDIGIT/ISNUMERIC union admits) — plus the
/// underscore. Deliberately NOT the regex crate's own `\w` (which adds
/// the combining marks `M*`, the connector punctuation `Pc` and the
/// join-control characters): the mutation-failure classifier's HTTP
/// status boundary must break exactly where Python's breaks — a
/// combining mark after the third digit is a boundary (Python: not a
/// word character), while a circled digit or a fraction after it is not
/// (Python: alphanumeric).
static PY_WORD_CHAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[\p{L}\p{N}_]$").expect("python \\w character set"));

/// One-character `\w` test for the classifier's word boundary — see
/// `PY_WORD_CHAR_RE`.
fn py_is_word_char(c: char) -> bool {
    let mut buf = [0u8; 4];
    PY_WORD_CHAR_RE.is_match(c.encode_utf8(&mut buf))
}

/// Position of *key* in *entries* under CPython's own dict-lookup rule:
/// the identity shortcut first (`stored is key`), then `==` (so keys
/// with no consistent `__eq__` — float NaN looked up through its own
/// object — still hit exactly the way the hash bucket's shortcut makes
/// them). The caller hashes *key* up front, so unhashable lookups raise
/// the reference's `TypeError` before any comparison runs — the order
/// the dict/OrderedDict protocol methods the engines replace follow. A
/// linear walk replaces the hash bucket scan: the same hit for every
/// key whose `__hash__`/`__eq__` honour the Python data model (the
/// documented contract those methods assume), and the walk preserves
/// the insertion order the OrderedDict engines are about.
fn py_key_position<'py, V>(
    py: Python<'py>,
    entries: &[(Py<PyAny>, V)],
    key: &Bound<'py, PyAny>,
) -> PyResult<Option<usize>> {
    for (i, (stored, _)) in entries.iter().enumerate() {
        let stored = stored.bind(py);
        if std::ptr::eq(stored.as_ptr(), key.as_ptr()) {
            return Ok(Some(i));
        }
        if stored.eq(key)? {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// The `for k, v in seq` unpacking of the shuffle-preferences ingestion
/// comprehension: a 2-tuple yields its halves; anything else goes through
/// CPython's UNPACK_SEQUENCE contract — the iterator protocol for
/// non-tuples (so a two-character string unpacks into its characters),
/// `ValueError` with CPython's own not-enough/too-many message for a
/// wrong-shaped yield, and the iterator protocol's own `TypeError` for
/// an uniterable item.
fn py_unpack_pair<'py>(
    item: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let unpack_not_enough = |got: usize| {
        PyValueError::new_err(format!(
            "not enough values to unpack (expected 2, got {got})"
        ))
    };
    if let Ok(tuple) = item.downcast::<PyTuple>() {
        return match tuple.len() {
            2 => Ok((tuple.get_item(0)?, tuple.get_item(1)?)),
            n if n > 2 => Err(PyValueError::new_err(unpack_too_many_message(
                item.py(),
                item,
                2,
            ))),
            n => Err(unpack_not_enough(n)),
        };
    }
    let mut got: Vec<Bound<'py, PyAny>> = Vec::new();
    for value in item.try_iter()? {
        got.push(value?);
        if got.len() > 2 {
            break;
        }
    }
    match got.len() {
        2 => Ok((got.remove(0), got.remove(0))),
        n if n > 2 => Err(PyValueError::new_err(unpack_too_many_message(
            item.py(),
            item,
            2,
        ))),
        n => Err(unpack_not_enough(n)),
    }
}

/// The LRU bookkeeping engine behind
/// `services/shuffle_prefs.py`'s `ShufflePreferences` — the ordered
/// `{context_id: shuffle}` map. Mirrors the pure-Python reference's
/// `OrderedDict` state exactly:
///
/// - `get` — the lookup key hashed first (`dict.get`'s own TypeError for
///   an unhashable), then the identity-or-`==` scan; a hit promotes the
///   entry to most-recent (the reference's `move_to_end`) and returns
///   its value, a miss leaves the state untouched and answers `None`.
/// - `set` — the key hashed first (`__setitem__`'s own TypeError), an
///   existing entry's value updated in place (the stored key object kept,
///   never replaced — the reference's assignment semantics) and then
///   promoted to most-recent (the reference's explicit `move_to_end`),
///   a fresh key appended at the end; then the eviction walk —
///   `while len > max: popitem(last=False)` — the oldest entries dropped
///   until the cap holds.
/// - `clear` — the state emptied.
/// - `load` — the hand-edited-file ingestion: a non-dict payload leaves
///   the state untouched (the reference's `isinstance(data, dict)`
///   gate), a dict is walked through its own `items()` method (a
///   subclass override steers this walk exactly like it steered the
///   comprehension) with the non-str keys dropped (`isinstance(k, str)`),
///   every surviving value coerced through Python truthiness (`bool(v)`
///   — `is_truthy`, CPython's own `__bool__`/`__len__` protocol), each
///   item unpacked through the `for k, v in ...` contract (see
///   `py_unpack_pair`), and the whole mapping built before the state is
///   swapped — a mid-walk failure leaves the old state in place, exactly
///   like the reference's partially-built (and then discarded)
///   OrderedDict.
/// - `dump_json` — the lock-held snapshot + serialisation of the
///   reference's `_save` (`dict(self._prefs)` then
///   `json.dumps(data, indent=2)`): the LRU order rendered through the
///   already-ported `json_dumps` engine with the reference's `indent=2`,
///   so every key coercion (int keys stringified, bool/None keys
///   rendered as the JSON encoder does) and every escape is CPython's
///   own; a non-coercible key raises the encoder's own `TypeError`.
///
/// The thread lock and the file I/O stay in the Python facade; the
/// cap is the reference's `_MAX_ENTRIES` (1000). A negative cap clamps
/// to zero — the reference's own negative-cap loop would pop from an
/// empty dict and crash with KeyError, an input the constant never
/// produces.
#[pyclass]
struct ShufflePrefsCore {
    /// The ordered `(key, value)` pairs, least-recent first.
    entries: Vec<(Py<PyAny>, bool)>,
    /// The entry cap (`_MAX_ENTRIES`).
    max_entries: usize,
}

#[pymethods]
impl ShufflePrefsCore {
    #[new]
    fn new(max_entries: i64) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(0) as usize,
        }
    }

    /// The saved shuffle state for *context_id*, or `None` — promoted to
    /// most-recent on a hit (the reference's `move_to_end`).
    fn get<'py>(
        &mut self,
        py: Python<'py>,
        context_id: &Bound<'py, PyAny>,
    ) -> PyResult<Option<bool>> {
        // dict.get hashes the lookup key before anything else.
        let _ = context_id.hash()?;
        match py_key_position(py, &self.entries, context_id)? {
            Some(i) => {
                let value = self.entries[i].1;
                // move_to_end — the hit becomes most-recent.
                let entry = self.entries.remove(i);
                self.entries.push(entry);
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Persist *shuffle* for *context_id* — the entry promoted to
    /// most-recent, the overflow evicted oldest-first.
    fn set<'py>(
        &mut self,
        py: Python<'py>,
        context_id: &Bound<'py, PyAny>,
        shuffle: bool,
    ) -> PyResult<()> {
        // __setitem__ hashes the key before anything else.
        let _ = context_id.hash()?;
        match py_key_position(py, &self.entries, context_id)? {
            Some(i) => {
                // The assignment updates the value in place (the stored
                // key object kept), then move_to_end promotes the entry.
                self.entries[i].1 = shuffle;
                let entry = self.entries.remove(i);
                self.entries.push(entry);
            }
            None => self.entries.push((context_id.clone().unbind(), shuffle)),
        }
        // while len(self._prefs) > _MAX_ENTRIES: popitem(last=False)
        while self.entries.len() > self.max_entries {
            if self.entries.is_empty() {
                break;
            }
            self.entries.remove(0);
        }
        Ok(())
    }

    /// Empty the state (`clear`).
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Ingest the parsed prefs file (see the class docs). Not a dict →
    /// no state change.
    fn load<'py>(&mut self, data: &Bound<'py, PyAny>) -> PyResult<()> {
        if !data.is_instance_of::<PyDict>() {
            return Ok(());
        }
        // The comprehension's data.items() — the method call, so a
        // subclass override steers the walk.
        let items = data.call_method0("items")?;
        let mut fresh: Vec<(Py<PyAny>, bool)> = Vec::new();
        for item in items.try_iter()? {
            let item = item?;
            let (k, v) = py_unpack_pair(&item)?;
            // if isinstance(k, str) — the non-str keys are dropped
            // before the value is ever coerced.
            if !k.is_instance_of::<PyString>() {
                continue;
            }
            // bool(v) — Python truthiness.
            fresh.push((k.clone().unbind(), v.is_truthy()?));
        }
        // The comprehension built the whole mapping before the
        // assignment; a failure above leaves the old state intact.
        self.entries = fresh;
        Ok(())
    }

    /// The `_save` snapshot + serialisation: the LRU order rendered as
    /// `json.dumps(dict(self._prefs), indent=2)` renders it (through the
    /// ported JSON encoder, so every key coercion and escape is
    /// CPython's own).
    fn dump_json(&self, py: Python) -> PyResult<String> {
        let snapshot = PyDict::new(py);
        for (k, v) in &self.entries {
            snapshot.set_item(k.bind(py), *v)?;
        }
        json_dumps(py, snapshot.as_any(), Some(2))
    }

    /// The number of live entries.
    fn length(&self) -> usize {
        self.entries.len()
    }
}

/// The read step behind `ShufflePreferences._load` — the pure core of
/// the prefs file's disk I/O.
///
/// Mirrors the pure-Python transcription (`_read_prefs_file`):
/// `path.read_text(encoding="utf-8")` — the read driven through the
/// *path* object's own method, so a `Path`, a `Path` subclass or a stub
/// steers both backends identically and every filesystem error shape
/// (the `OSError` subclasses with their filename, the
/// `UnicodeDecodeError` of an undecodable file) is CPython's own — and
/// then `json.loads` over the read result through the ported
/// CPython-grammar scanner. A scanner rejection — and every non-str
/// result a stub's `read_text` might answer — falls through to the
/// interpreter's own `json.loads` call over the same object, the same
/// fallback the auth record parser uses for its exotic-decode path, so
/// the error the caller sees (the `JSONDecodeError` class, message,
/// `pos`/`lineno`/`colno`, or the `TypeError` of an unparseable shape)
/// is exactly the reference's own; the scanner's accept/reject boundary
/// mirrors `json.loads`'s, so the re-ask raises where the reference
/// raises and the parity matrix pins the two together. The exists gate
/// and the logging stay in the facade; the returned value passes
/// through as the object the parse built.
#[pyfunction]
fn shuffle_prefs_read_file<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // self._path.read_text(encoding="utf-8") — through the object's
    // own method, the keyword spelled the reference's way.
    let kwargs = PyDict::new(py);
    kwargs.set_item("encoding", "utf-8")?;
    let text = path.getattr("read_text")?.call((), Some(&kwargs))?;
    // json.loads(text) — the ported scanner over a str, the
    // interpreter's own json.loads for every other shape.
    match text.downcast::<PyString>() {
        Ok(text) => match json_loads(py, text) {
            Ok(value) => Ok(value),
            Err(JsonReject::Py(err)) => Err(err),
            Err(JsonReject::Reject) => py.import("json")?.getattr("loads")?.call1((text,)),
        },
        Err(_) => py.import("json")?.getattr("loads")?.call1((text,)),
    }
}

/// The write step behind `ShufflePreferences._save` — the pure core of
/// the prefs file's disk I/O.
///
/// Mirrors the pure-Python transcription (`_write_prefs_file`), every
/// operation driven through the *path* object's own protocol so a
/// `Path`, a `Path` subclass or a stub steers both backends
/// identically and every error shape is CPython's own:
///
/// - `path.parent.mkdir(parents=True, exist_ok=True)` — the parent
///   property read, then the directory creation through the parent's
///   own method with both keywords spelled the reference's way.
/// - `path.with_suffix(path.suffix + ".tmp")` — the suffix read
///   through the property, the `+ ".tmp"` through CPython's own
///   operator (`operator.add`, so a str subclass's overridden
///   `__add__` steers both backends identically), and the sibling
///   built through the path's own `with_suffix` (its `ValueError`
///   shapes for the degenerate names included).
/// - `tmp.write_text(text, encoding="utf-8")` — the serialised state
///   written through the sibling's own method, the text passed through
///   as the object it is.
/// - `tmp.replace(path)` — the atomic cutover through the sibling's
///   own `os.replace` delegation, the final path passed through as
///   the object it is.
///
/// The snapshot + serialisation (the core's `dump_json`) and the
/// logging stay in the facade; the statement order is the reference's
/// own, so a failure leaves exactly the partial state the reference's
/// would.
#[pyfunction]
fn shuffle_prefs_write_file<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    text: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // self._path.parent.mkdir(parents=True, exist_ok=True)
    let parent = path.getattr("parent")?;
    let mkdir_kwargs = PyDict::new(py);
    mkdir_kwargs.set_item("parents", true)?;
    mkdir_kwargs.set_item("exist_ok", true)?;
    parent.getattr("mkdir")?.call((), Some(&mkdir_kwargs))?;
    // tmp = self._path.with_suffix(self._path.suffix + ".tmp")
    let suffix = path.getattr("suffix")?;
    let add = py.import("operator")?.getattr("add")?;
    let new_suffix = add.call1((suffix, ".tmp"))?;
    let tmp = path.getattr("with_suffix")?.call1((new_suffix,))?;
    // tmp.write_text(text, encoding="utf-8")
    let write_kwargs = PyDict::new(py);
    write_kwargs.set_item("encoding", "utf-8")?;
    tmp.getattr("write_text")?
        .call((text,), Some(&write_kwargs))?;
    // tmp.replace(self._path)
    tmp.getattr("replace")?.call1((path,))?;
    Ok(())
}

/// The LRU bookkeeping engine behind the album-art widget's module-level
/// URL→render cache — the ordered map of thumbnail URLs to the assembled
/// Rich `Text` objects. The values are held as the Python objects they
/// are; the bookkeeping mirrors the reference's `OrderedDict` exactly,
/// which differs from the shuffle-prefs engine in exactly the ways the
/// two Python originals differ:
///
/// - `get` fuses the reference's cache-check path — `url in _ART_CACHE`
///   plus `move_to_end` plus the value read — into one promotion: the
///   lookup key hashed first (`in`'s own TypeError for an unhashable),
///   then the identity-or-`==` scan; a hit promotes the URL to
///   most-recent and hands back the cached render, a miss leaves the
///   state untouched. (The cache never holds `None` — the renderer
///   always returns a `Text` — so `None` answers a miss.)
/// - `insert` fuses the reference's store path — `_ART_CACHE[url] =
///   rendered` plus the overflow check — with the `OrderedDict`
///   `__setitem__` semantics: an existing key's value is updated in
///   place and its position KEPT (this engine never promotes on insert;
///   the shuffle engine does), a fresh key appended at the end; then the
///   reference's single `if len > max: popitem(last=False)` — one
///   oldest-entry eviction, the `if` shape rather than the shuffle
///   engine's `while` walk. The evicted pair is returned (or `None`
///   when nothing was evicted) so tests can pin which render fell out.
/// - `remove` fuses the resize invalidation's `if url in _ART_CACHE:
///   del _ART_CACHE[url]` — the key hashed first, deleted if present,
///   answering whether it was.
/// - `clear` and `length` complete the protocol.
///
/// The lock-free single-threaded contract is the reference's own (the
/// cache lives and dies with the event loop's worker exclusivity); the
/// cap is `_ART_CACHE_MAX` (20). A negative cap clamps to zero, the
/// unreachable corner of the constant's own.
#[pyclass]
struct ArtCacheCore {
    /// The ordered `(url, render)` pairs, least-recent first.
    entries: Vec<(Py<PyAny>, Py<PyAny>)>,
    /// The entry cap (`_ART_CACHE_MAX`).
    max_entries: usize,
}

#[pymethods]
impl ArtCacheCore {
    #[new]
    fn new(max_entries: i64) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(0) as usize,
        }
    }

    /// The cached render for *url* — promoted to most-recent on a hit.
    fn get<'py>(
        &mut self,
        py: Python<'py>,
        url: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        // `url in _ART_CACHE` hashes the key before anything else.
        let _ = url.hash()?;
        match py_key_position(py, &self.entries, url)? {
            Some(i) => {
                let value = self.entries[i].1.clone_ref(py);
                // move_to_end — the hit becomes most-recent.
                let entry = self.entries.remove(i);
                self.entries.push(entry);
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Store *value* for *url* — the existing position kept, the
    /// overflow evicted oldest-first — answering the evicted pair.
    fn insert<'py>(
        &mut self,
        py: Python<'py>,
        url: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Option<(Py<PyAny>, Py<PyAny>)>> {
        // `od[url] = value` hashes the key before anything else.
        let _ = url.hash()?;
        match py_key_position(py, &self.entries, url)? {
            Some(i) => {
                // __setitem__ on an existing key: the value updated in
                // place, the position kept — never a promotion.
                self.entries[i].1 = value.clone().unbind();
            }
            None => self
                .entries
                .push((url.clone().unbind(), value.clone().unbind())),
        }
        // if len(_ART_CACHE) > _ART_CACHE_MAX: popitem(last=False) — the
        // single-pop shape of the reference.
        if self.entries.len() > self.max_entries {
            let evicted = self.entries.remove(0);
            return Ok(Some(evicted));
        }
        Ok(None)
    }

    /// Delete *url* if cached, answering whether it was (the resize
    /// invalidation's guarded delete).
    fn remove<'py>(&mut self, py: Python<'py>, url: &Bound<'py, PyAny>) -> PyResult<bool> {
        // The `in` probe hashes the key before anything else.
        let _ = url.hash()?;
        match py_key_position(py, &self.entries, url)? {
            Some(i) => {
                self.entries.remove(i);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Empty the cache.
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// The number of cached renders.
    fn length(&self) -> usize {
        self.entries.len()
    }
}

// ── Mutation-failure classifier ────────────────────────────────────────

/// The `_HTTP_STATUS_RE.match(str(exc))` of the mutation-failure
/// classifier — the pure core of `_classify_mutation_failure`'s
/// YTMusicServerError arm: the literal prefix `Server returned HTTP `
/// (ASCII, anchored at the start of the message like the reference's
/// `^`), then exactly three Unicode decimal digits (CPython's `\d` on
/// `str` patterns — category Nd, any script, so Arabic-Indic ٤٠١ and
/// Devanagari ४०३ are three-digit codes), then the word boundary `\b`:
/// the end of the message, or a next character that is not CPython's
/// `\w` (alphanumeric-or-underscore — see `py_is_word_char`; a combining
/// mark after the digits is a boundary, a circled digit or a vulgar
/// fraction is not, exactly where Python's own `\b` breaks). Fewer
/// than three digits, or a fourth word character, fails the match —
/// the fixed-count group has no shorter retry under the anchored
/// literal prefix. The three digits then parse through `int()`'s own
/// decimal-digit acceptance (every Nd trio is a valid int — ٤٠١ is
/// 401), so a Unicode-digit 401/403 classifies auth_expired exactly
/// like an ASCII one.
fn server_http_status(message: &str) -> Option<i64> {
    let rest = message.strip_prefix("Server returned HTTP ")?;
    let mut value: i64 = 0;
    let mut chars = rest.chars();
    // Three `\d`s — fewer available is the match failure.
    for _ in 0..3 {
        let c = chars.next()?;
        value = value * 10 + i64::from(nd_digit_value(c)?);
    }
    // `\b` — the end of the message or the next character is not a
    // word character.
    match chars.next() {
        None => Some(value),
        Some(c) if !py_is_word_char(c) => Some(value),
        Some(_) => None,
    }
}

/// The mutation-failure classifier behind `services/ytmusic.py`'s
/// `_classify_mutation_failure` — the pure core that maps an exception
/// raised by a mutation call to its `MutationResult` kind, never
/// "success".
///
/// Mirrors the reference's isinstance chain in its exact order
/// (`KeyError` → parser drift is a server-side problem; the two timeout
/// classes → no confirmation in time, the write may have landed;
/// `requests`' `RequestException` → network; `YTMusicUserError` → no
/// auth set up; `YTMusicServerError` → the HTTP status parsed out of
/// the message, 401/403 an expired session, anything else a
/// server-side problem), with every class resolved from the running
/// interpreter at call time — `asyncio.TimeoutError` is whatever that
/// attribute is on the running Python (the builtin `TimeoutError` on
/// 3.11+, asyncio's own class before), `requests.exceptions` and
/// `ytmusicapi.exceptions` the very modules the reference imported at
/// its own module scope. The YTMusicServerError arm renders the
/// exception through `str()` and matches the HTTP-status prefix (see
/// `server_http_status`); anything else falls through to
/// "server_error", the reference's own unreachable-in-practice safety
/// default. The inputs are never mutated.
#[pyfunction]
fn classify_mutation_failure(py: Python, exc: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    // isinstance(exc, KeyError) — parser drift classifies server_error.
    let builtins = py.import("builtins")?;
    let key_error = builtins.getattr("KeyError")?;
    if exc.is_instance(&key_error)? {
        return Ok("server_error");
    }
    // isinstance(exc, (asyncio.TimeoutError, requests.ReadTimeout))
    let requests_exceptions = py.import("requests.exceptions")?;
    let asyncio_timeout = py.import("asyncio")?.getattr("TimeoutError")?;
    let read_timeout = requests_exceptions.getattr("ReadTimeout")?;
    if exc.is_instance(&asyncio_timeout)? || exc.is_instance(&read_timeout)? {
        return Ok("timeout");
    }
    // isinstance(exc, requests.RequestException)
    let request_exception = requests_exceptions.getattr("RequestException")?;
    if exc.is_instance(&request_exception)? {
        return Ok("network");
    }
    // isinstance(exc, YTMusicUserError)
    let ytmusic_exceptions = py.import("ytmusicapi.exceptions")?;
    let user_error = ytmusic_exceptions.getattr("YTMusicUserError")?;
    if exc.is_instance(&user_error)? {
        return Ok("auth_required");
    }
    // isinstance(exc, YTMusicServerError) — the HTTP status prefix
    // decides auth_expired vs server_error.
    let server_error = ytmusic_exceptions.getattr("YTMusicServerError")?;
    if exc.is_instance(&server_error)? {
        // str(exc) — rendered through CPython's own __str__ (a raising
        // __str__ propagates, exactly like the reference's str() call);
        // the lossy read replaces lone surrogates with U+FFFD, which
        // moves no boundary (a surrogate is not \w, and neither is the
        // replacement character — the only positions the match reads).
        let message = exc.str()?.to_string_lossy().into_owned();
        if let Some(status) = server_http_status(&message) {
            if status == 401 || status == 403 {
                return Ok("auth_expired");
            }
        }
        return Ok("server_error");
    }
    // The reference's unreachable-in-practice safety default.
    Ok("server_error")
}

/// The ytmusic module's own clock — `time.monotonic()` read through the
/// module's `time` attribute, exactly like the reference's global
/// lookup at call time. A test (or any caller) that swaps
/// `ytm_player.services.ytmusic`'s `time` for a frozen namespace steers
/// the native engines exactly like the fallback: the clock the
/// reference would read is the clock the engines read.
fn ytmusic_module_monotonic(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let ytmusic = PyModule::import(py, "ytm_player.services.ytmusic")?;
    let time_module = ytmusic.getattr("time")?;
    let clock = time_module.getattr("monotonic")?;
    clock.call0()
}

/// The auth-expiry sniff behind `services/ytmusic.py`'s
/// `_is_auth_expired` — the gate every expired-session retry
/// (`_call`'s except arm and `get_playlist`'s ordered branch)
/// consults before renewing.
///
/// Mirrors the reference's isinstance chain in its exact order: a
/// `YTMusicServerError` (the class resolved from the running
/// interpreter, the very module the reference imported at its own
/// module scope) delegates to the module's own
/// `_classify_mutation_failure` — resolved through the module
/// attribute at call time, so whichever classifier the module is
/// running (the native engine or the pure-Python transcription)
/// steers this engine identically — and answers whether it classified
/// "auth_expired" (CPython's own `==`); a `KeyError` renders through
/// `str()`, lowercases through the str's own `.lower()` method and
/// answers whether the lowered message contains any of the three
/// sign-in markers — `signinendpoint`, `sign in to`, or the
/// `'logged_in', 'value': '0'` fragment — each containment through
/// CPython's own `in` (the substring protocol on the message object
/// itself, never a re-encoded copy); everything else answers `False`.
#[pyfunction]
#[pyo3(signature = (exc))]
fn ytm_auth_expired(py: Python, exc: &Bound<'_, PyAny>) -> PyResult<bool> {
    // isinstance(exc, YTMusicServerError) — the classifier verdict.
    let ytmusic_exceptions = py.import("ytmusicapi.exceptions")?;
    let server_error = ytmusic_exceptions.getattr("YTMusicServerError")?;
    if exc.is_instance(&server_error)? {
        let classifier = PyModule::import(py, "ytm_player.services.ytmusic")?
            .getattr("_classify_mutation_failure")?;
        let kind = classifier.call1((exc,))?;
        return kind.eq("auth_expired");
    }
    // isinstance(exc, KeyError) — the sign-in marker sniff over the
    // lowered message.
    let builtins = py.import("builtins")?;
    let key_error = builtins.getattr("KeyError")?;
    if exc.is_instance(&key_error)? {
        let message = exc.str()?.call_method0("lower")?;
        let marker_1 = PyString::new(py, "signinendpoint").into_any();
        if message.contains(&marker_1)? {
            return Ok(true);
        }
        let marker_2 = PyString::new(py, "sign in to").into_any();
        if message.contains(&marker_2)? {
            return Ok(true);
        }
        let marker_3 = PyString::new(py, "'logged_in', 'value': '0'").into_any();
        return message.contains(&marker_3);
    }
    Ok(false)
}

/// The renewal-cooldown gate — `services/ytmusic.py`'s
/// `_renewed_client_for` cooldown check, the bookkeeping that keeps a
/// failed automatic renewal from being retried (with its browser
/// cookie extraction and account probes) until the cooldown has
/// elapsed.
///
/// Mirrors the reference's two expressions exactly: the clock read
/// through the module's own `time` name first (see
/// `ytmusic_module_monotonic`), then `remaining = retry_after - now`
/// through CPython's own `-` (its `__sub__` protocol — a non-numeric
/// *retry_after* raises the operator's own `TypeError`), then the
/// `> 0` verdict through CPython's own comparison. The answer is the
/// (blocked, remaining) pair — *remaining* the very object the
/// subtraction produced, for the caller's `%.0fs` log line — with the
/// lock, the renewal itself and the raise staying Python-side in the
/// caller.
#[pyfunction]
#[pyo3(signature = (retry_after))]
fn ytm_renewal_cooldown_gate<'py>(
    py: Python<'py>,
    retry_after: &Bound<'py, PyAny>,
) -> PyResult<(bool, Bound<'py, PyAny>)> {
    let now = ytmusic_module_monotonic(py)?;
    let remaining = retry_after.sub(&now)?;
    let zero = 0i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let blocked = remaining.gt(&zero)?;
    Ok((blocked, remaining))
}

/// The renewal-cooldown stamp — `services/ytmusic.py`'s
/// `_renewed_client_for` failure arm, the bookkeeping a failed
/// automatic renewal writes so the gate above blocks the retries
/// until the cooldown has elapsed.
///
/// Mirrors the reference's single expression exactly: the clock read
/// through the module's own `time` name, then `now +
/// _RENEWAL_COOLDOWN_SECONDS` through CPython's own `+`, with the
/// cooldown constant read through the module's own attribute — the
/// very constant the fallback's own expression runs against, a
/// patched constant included. The stamp is whatever object the
/// addition produced; the assignment stays Python-side in the caller.
#[pyfunction]
fn ytm_renewal_cooldown_stamp<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let now = ytmusic_module_monotonic(py)?;
    let cooldown = PyModule::import(py, "ytm_player.services.ytmusic")?
        .getattr("_RENEWAL_COOLDOWN_SECONDS")?;
    now.add(&cooldown)
}

/// The consecutive-failure step — `services/ytmusic.py`'s `_run`
/// except arm, the retry bookkeeping that counts the expected API
/// failures and re-creates the YTMusic client once a streak of them
/// suggests the session has gone stale.
///
/// Mirrors the reference's two expressions exactly: the bump through
/// CPython's own `+` (its `__add__` protocol — a non-numeric count
/// raises the operator's own `TypeError`), then the `>=` verdict
/// through CPython's own comparison against the module's own
/// `_MAX_API_FAILURES_BEFORE_REINIT` — read through the module
/// attribute, so a patched threshold steers both backends
/// identically. The answer is the (bumped, due) pair — the bumped
/// count for the caller's own assignment and log line, the verdict
/// naming whether the reinit branch runs — with the log, the
/// `_client_init_lock` hold, the client clear and the post-reinit
/// reset to 0 staying Python-side in the caller, at the reference's
/// own points.
#[pyfunction]
#[pyo3(signature = (consecutive_failures))]
fn ytm_api_failure_step<'py>(
    py: Python<'py>,
    consecutive_failures: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let bumped = consecutive_failures.add(&one)?;
    let threshold = PyModule::import(py, "ytm_player.services.ytmusic")?
        .getattr("_MAX_API_FAILURES_BEFORE_REINIT")?;
    let due = bumped.ge(&threshold)?;
    Ok((bumped, due))
}

// ── YTMusic browse-wrapper engines (services/ytmusic.py) ───────────────

/// The kwargs dict behind `YTMusicService.get_watch_playlist` — the
/// pure core of the wrapper's call assembly from `services/ytmusic.py`.
///
/// Mirrors the pure-Python reference (`_watch_playlist_kwargs`): the
/// conditional members in the reference's insertion order — `limit`
/// always; `videoId` / `playlistId` only when the caller passed them,
/// the gate an `is None` identity check (a falsy but non-None id still
/// travels); `radio` only when truthy, and the literal `True` travels,
/// never the flag object. Every value passes through as the object it
/// is; the both-None early return and the `**` unpacking stay
/// Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (video_id, playlist_id, limit, radio))]
fn ytm_watch_playlist_kwargs<'py>(
    py: Python<'py>,
    video_id: &Bound<'py, PyAny>,
    playlist_id: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
    radio: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("limit", limit)?;
    if !video_id.is_none() {
        kwargs.set_item("videoId", video_id)?;
    }
    if !playlist_id.is_none() {
        kwargs.set_item("playlistId", playlist_id)?;
    }
    if radio.is_truthy()? {
        kwargs.set_item("radio", true)?;
    }
    Ok(kwargs)
}

/// The `tracks` member of a watch / liked-songs payload — the pure
/// core of the tracks extraction `get_liked_songs`,
/// `get_watch_playlist`, `get_radio`'s per-seed fetch,
/// `get_playlist_radio` and the discovery trending/liked mixes share
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_result_tracks`): over the
/// subclass-aware `isinstance(result, dict)` gate the member is read
/// through the mapping's own `get` with a fresh empty list as the
/// default (the default list built every call, exactly like the
/// reference's literal); a non-dict answers a fresh empty list,
/// never the input itself, and a dict passes its `tracks` object
/// through as the object it is.
#[pyfunction]
#[pyo3(signature = (result))]
fn ytm_result_tracks<'py>(result: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = result.py();
    if result.is_instance_of::<PyDict>() {
        let default = PyList::empty(py).into_any();
        return result.getattr("get")?.call(("tracks", default), None);
    }
    Ok(PyList::empty(py).into_any())
}

/// The `tracks` member of a playlist payload, list-gated — the pure
/// core of `get_chart_shelf_tracks`'s extraction
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_playlist_tracks`): over the
/// subclass-aware `isinstance(playlist, dict)` gate the member is read
/// through the mapping's own `get` with a fresh empty list default,
/// then the reference's `or []` (a falsy member — `None` where the
/// payload carries the key's null shape — answers a fresh empty
/// list), then the subclass-aware `isinstance(raw, list)` gate: only
/// a list passes through, as the object it is; every other member
/// shape — a dict, a string, a generator — answers a fresh empty
/// list.
#[pyfunction]
#[pyo3(signature = (playlist))]
fn ytm_playlist_tracks<'py>(playlist: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = playlist.py();
    if playlist.is_instance_of::<PyDict>() {
        let default = PyList::empty(py).into_any();
        let raw = playlist.getattr("get")?.call(("tracks", default), None)?;
        let raw = if raw.is_truthy()? {
            raw
        } else {
            PyList::empty(py).into_any()
        };
        if raw.is_instance_of::<PyList>() {
            return Ok(raw);
        }
    }
    Ok(PyList::empty(py).into_any())
}

/// The deduplicated pool of radio tracks across all seeds' fetches —
/// the pure core of `YTMusicService.get_radio`'s mix assembly
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_radio_seed_pool`): the
/// per-seed result lists walked through Python's iteration protocol
/// (a non-iterable raises CPython's own `TypeError`; an `__iter__`
/// that itself raises keeps its own error), each list walked in
/// turn, and each track read its id through the mapping's own `get`
/// — `videoId` first, `video_id` with the fresh `""` default only
/// when the first read answers falsy (the reference's `or`, so the
/// second attribute lookup and call happen only then; a track
/// without a `get` attribute raises the attribute protocol's own
/// `AttributeError` at exactly the lookup the reference would have
/// made) — and the track kept when that id is truthy and not yet
/// seen: the set's own containment and add — the id hashed first
/// (an unhashable id raises the hash protocol's own `TypeError`),
/// the identity-or-`==` probe deciding membership, the FIRST of
/// equal ids the one that keeps its track. Kept tracks pass through
/// as the objects they are, in first-sight order; the shuffle, the
/// limit trim and the normalisation stay Python-side in the wrapper,
/// next to the RNG state.
#[pyfunction]
#[pyo3(signature = (results))]
fn ytm_radio_seed_pool<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let pool = PyList::empty(py);
    let seen = PySet::empty(py)?;
    for tracks in results.try_iter()? {
        for track in tracks?.try_iter()? {
            let track = track?;
            let first = track.getattr("get")?.call1(("videoId",))?;
            let vid = if first.is_truthy()? {
                first
            } else {
                track.getattr("get")?.call(("video_id", ""), None)?
            };
            // if vid and vid not in seen_ids — the truthiness gate
            // first, the containment (and its hash) only behind it.
            if vid.is_truthy()? && !seen.contains(&vid)? {
                seen.add(&vid)?;
                pool.append(track)?;
            }
        }
    }
    Ok(pool)
}

/// The six discovery sources' rotation order — the pure core of
/// `YTMusicService.get_discovery_mix`'s source cycling
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_discovery_source_order`):
/// the `(last % 6) + 1` start (0 → 1 first, 5 → 6 first, 6 wraps to
/// 1) and the six-entry rotation that follows it —
/// `(start + i - 1) % 6 + 1` for `i` over `range(6)`, left-associative
/// exactly like the reference spells it — every operator through
/// CPython's own, so a non-int last source (a float, a bool) steers
/// both backends identically.
#[pyfunction]
#[pyo3(signature = (last_source))]
fn ytm_discovery_source_order<'py>(
    py: Python<'py>,
    last_source: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let six = 6i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let start = last_source.rem(&six)?.add(&one)?;
    let out = PyList::empty(py);
    for i in 0i64..6 {
        let i = i
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any();
        out.append(start.add(&i)?.sub(&one)?.rem(&six)?.add(&one)?)?;
    }
    Ok(out)
}

/// The chart shelves of a `get_charts` payload — the pure core of the
/// discovery charts mix's shelf assembly (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_chart_shelves`): the `daily`,
/// `weekly` and `videos` members read through the mapping's own `get`
/// — one lookup per key, in that order, a missing member skipped —
/// each gated by the subclass-aware `isinstance(arr, list)`, and each
/// survivor's items filtered by the subclass-aware
/// `isinstance(s, dict)` into the flat shelf list. The dict shelves
/// pass through as the objects they are; the payload is never
/// mutated.
#[pyfunction]
#[pyo3(signature = (charts))]
fn ytm_chart_shelves<'py>(
    py: Python<'py>,
    charts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let shelves = PyList::empty(py);
    for key in ["daily", "weekly", "videos"] {
        let arr = charts.getattr("get")?.call1((key,))?;
        if arr.is_instance_of::<PyList>() {
            for s in arr.try_iter()? {
                let s = s?;
                if s.is_instance_of::<PyDict>() {
                    shelves.append(s)?;
                }
            }
        }
    }
    Ok(shelves)
}

/// The next chart shelf's position in the rotation — the pure core of
/// the discovery charts mix's shelf pick (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_chart_shelf_index`): the
/// `(last + 1) % count` round-robin step, both operators through
/// CPython's own — so a zero count raises the modulo's own
/// `ZeroDivisionError` exactly like the reference would (the mix
/// guards with `if not shelves` before it ever reaches the pick).
#[pyfunction]
#[pyo3(signature = (last_index, shelf_count))]
fn ytm_chart_shelf_index<'py>(
    py: Python<'py>,
    last_index: &Bound<'py, PyAny>,
    shelf_count: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    last_index.add(&one)?.rem(shelf_count)
}

/// The playable track dicts of a discovery payload — the pure core of
/// the playable filter the trending, home, liked, artist and history
/// mixes run over their payloads (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_discovery_playable_tracks`):
/// *items* walked through Python's iteration protocol, and only dict
/// items (the subclass-aware `isinstance` gate, short-circuited
/// before any `get` call exactly like the reference's `and`) whose
/// `videoId` member — read through the mapping's own `get` — is
/// truthy pass through, as the objects they are, in the payload's
/// order. The input is never mutated.
#[pyfunction]
#[pyo3(signature = (items))]
fn ytm_discovery_playable_tracks<'py>(
    py: Python<'py>,
    items: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for t in items.try_iter()? {
        let t = t?;
        if t.is_instance_of::<PyDict>() && t.getattr("get")?.call1(("videoId",))?.is_truthy()? {
            out.append(t)?;
        }
    }
    Ok(out)
}

/// The playable track dicts of a chart shelf's track list — the
/// charts mix's own playable filter, the two-key spelling the chart
/// shelves' mixed payload shapes need (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_chart_playable_tracks`):
/// *tracks* walked through Python's iteration protocol, and only
/// dict items (the subclass-aware `isinstance` gate) pass whose id
/// read — `videoId` first through the mapping's own `get`,
/// `video_id` with the fresh `""` default only when the first read
/// answers falsy (the reference's `or`) — is truthy. Kept tracks
/// pass through as the objects they are, in the list's order.
#[pyfunction]
#[pyo3(signature = (tracks))]
fn ytm_chart_playable_tracks<'py>(
    py: Python<'py>,
    tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for t in tracks.try_iter()? {
        let t = t?;
        if t.is_instance_of::<PyDict>() {
            let first = t.getattr("get")?.call1(("videoId",))?;
            let vid = if first.is_truthy()? {
                first
            } else {
                t.getattr("get")?.call(("video_id", ""), None)?
            };
            if vid.is_truthy()? {
                out.append(t)?;
            }
        }
    }
    Ok(out)
}

/// The new-releases member of an explore payload — the pure core of
/// `YTMusicService.get_new_releases` (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_explore_new_releases`): over
/// the subclass-aware `isinstance(result, dict)` gate the
/// `new_releases` member is read through the mapping's own `get`
/// with a fresh empty list default and passed through — as the list
/// object it is — only when the subclass-aware
/// `isinstance(releases, list)` gate holds; every other shape
/// answers a fresh empty list.
#[pyfunction]
#[pyo3(signature = (result))]
fn ytm_explore_new_releases<'py>(
    py: Python<'py>,
    result: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if result.is_instance_of::<PyDict>() {
        let default = PyList::empty(py).into_any();
        let releases = result
            .getattr("get")?
            .call(("new_releases", default), None)?;
        if releases.is_instance_of::<PyList>() {
            return Ok(releases);
        }
    }
    Ok(PyList::empty(py).into_any())
}

/// The sort-params token for a `get_playlist` order name — the pure
/// core of `YTMusicService.get_playlist`'s order resolution
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_playlist_order_params`): the
/// `order or ""` truthiness gate (`None` and every other falsy
/// spelling read as the default order) and the `_ORDER_PARAMS`
/// table lookup — the table read from the service class at call time
/// (see `YTMusicService._ORDER_PARAMS`), so a patched table steers
/// both backends identically — answering the a_to_z / z_to_a /
/// recently_added browse `params` tokens, or `None` for the
/// playlist's server-side default. The ordered-fetch patch window
/// and the renewal machinery stay Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (order))]
fn ytm_playlist_order_params<'py>(
    py: Python<'py>,
    order: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let table = py
        .import("ytm_player.services.ytmusic")?
        .getattr("YTMusicService")?
        .getattr("_ORDER_PARAMS")?;
    let key = if order.is_truthy()? {
        order.clone()
    } else {
        PyString::new(py, "").into_any()
    };
    table.getattr("get")?.call1((key,))
}

// ── Playback-bar track-info width engine ───────────────────────────────

/// The width-splitting engine behind the playback bar's `_TrackInfo`
/// render — the pure core of the one-line title/artist/album split from
/// `ui/playback_bar.py`: the available width for the three text
/// fragments and the per-fragment truncation widths.
///
/// Mirrors the reference's arithmetic exactly: `max_w = max(10, bar_width
/// - 30)` (the reserved tail — volume, repeat, shuffle, heart — plus the
/// state icon's 3 columns, floored at 10 for the narrowest terminals);
/// `title_w = min(len(title), max_w // 2)` and `artist_w =
/// min(len(artist), max_w // 3)` (the halves-and-thirds split of the
/// reference, floor division — `max_w` is at least 10, so the division
/// never goes negative); and `album_w = max_w - title_w - artist_w - 8`
/// (the two em-dash separators — 3 columns each — and the icon again),
/// clamped at the reference's own `max(0, album_w)` at the truncate
/// call. The album's own length never enters the split — the reference
/// gives the album whatever remains, no matter how long it is — so the
/// engine takes no album; the lens are CPython's `len()` — code points,
/// computed on the original (pre-reorder, pre-truncation) strings
/// exactly like the reference computes them. The Rich `Text` assembly,
/// the BiDi reorder/isolate calls and the state icon stay Python-side
/// in the widget.
#[pyfunction]
#[pyo3(signature = (bar_width, title, artist))]
fn track_info_widths(bar_width: i64, title: &str, artist: &str) -> (i64, i64, i64) {
    // max_w = max(10, self.size.width - 30) — floored at 10.
    let max_w = 10.max(bar_width - 30);
    // len() counts code points.
    let title_len = title.chars().count() as i64;
    let artist_len = artist.chars().count() as i64;
    // title_w = min(len(title), max_w // 2); artist_w = min(len(artist),
    // max_w // 3) — floor division on the at-least-10 budget.
    let title_w = title_len.min(max_w / 2);
    let artist_w = artist_len.min(max_w / 3);
    // album_w = max_w - title_w - artist_w - 8 — the reference's
    // max(0, album_w) clamp at the truncate call site.
    let album_w = (max_w - title_w - artist_w - 8).max(0);
    (title_w, artist_w, album_w)
}

// ── Audio-cache LRU eviction engine ────────────────────────────────────

/// The `total -= file_size` of the eviction walk — CPython's in-place
/// subtraction, replicated protocol step for protocol step. The
/// augmented assignment reads the `__isub__` slot from the operand's
/// TYPE (an instance-attribute `__isub__` never runs — the operator
/// protocol reads the type) and uses its result unless it answers
/// `NotImplemented`; otherwise the binary subtraction runs — the very
/// `PyNumber_Subtract` the reference's `x - y` would run, subclass
/// reflection and all — and its result is what the assignment stores
/// back. Ints carry no `__isub__` (the production walk's int total,
/// straight from SQLite's SUM, always takes the binary path), and
/// where every slot misses, the binary failure's canonical
/// "unsupported operand type(s) for -" message is respelled with the
/// in-place protocol's own `-=` — the one shape the binary API cannot
/// spell.
fn py_inplace_sub<'py>(
    lhs: &Bound<'py, PyAny>,
    rhs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = lhs.py();
    let not_implemented = py.NotImplemented();
    if let Ok(in_place) = lhs.get_type().getattr("__isub__") {
        let result = in_place.call1((lhs, rhs))?;
        if !result.is(&not_implemented) {
            return Ok(result);
        }
    }
    match lhs.sub(rhs) {
        Ok(result) => Ok(result),
        Err(err) => {
            if err.is_instance_of::<PyTypeError>(py) {
                let message = err.value(py).str()?.to_string_lossy().into_owned();
                if let Some(rest) = message.strip_prefix("unsupported operand type(s) for -: ") {
                    return Err(PyTypeError::new_err(format!(
                        "unsupported operand type(s) for -=: {rest}"
                    )));
                }
            }
            Err(err)
        }
    }
}

/// The LRU eviction walk behind `CacheManager.evict` from
/// `services/cache.py` — the pure core of the audio cache's overflow
/// handling.
///
/// Mirrors the pure-Python reference (`_eviction_plan`): *rows* is
/// walked in the caller's LRU order (oldest access first — the order
/// the reference's `ORDER BY last_accessed ASC` fetched), and before
/// each row the running total is compared against the budget —
/// `total_size <= max_bytes` through CPython's own `<=`, its
/// `TypeError` for an incomparable pair propagating — so an entry is
/// evicted only while the total is still over budget, exactly the
/// reference's check-then-evict shape: the entry that brings the total
/// exactly to the budget is the LAST evicted (the next check breaks).
/// The iteration begins before the first budget check — the reference's
/// `for` fetched the first row before its `if` ran — so a
/// non-iterable *rows* raises the iteration protocol's own `TypeError`
/// even when the total already fits. Each evicted row is read through
/// CPython's own subscript in the reference's read order —
/// `file_path` (the unlink's read), then `video_id` (the append's),
/// then `file_size` (the subtraction's) — and the three values pass
/// through into the returned `(video_id, file_path, file_size)` triple
/// as the objects they are; the running total is decremented through
/// the reference's augmented assignment (`total -= file_size`, see
/// `py_inplace_sub`). The result is a fresh list of fresh triples in
/// eviction order; the input rows are never mutated, and no file is
/// touched — the plan is computed before the first unlink (the
/// reference unlinked mid-walk; with SQLite rows, the only production
/// input, no read can fail, and the index — the source of truth — is
/// untouched either way), with the unlinking, the index DELETE and the
/// commit staying in the manager.
#[pyfunction]
#[pyo3(signature = (total_size, max_bytes, rows))]
fn eviction_plan<'py>(
    py: Python<'py>,
    total_size: &Bound<'py, PyAny>,
    max_bytes: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let mut total = total_size.clone();
    let out = PyList::empty(py);
    for row in rows.try_iter()? {
        let row = row?;
        // if total <= max_bytes: break — the check before each entry.
        if total.le(max_bytes)? {
            break;
        }
        // The reference's read order: the unlink's file_path, the
        // append's video_id, the subtraction's file_size. The triple
        // and the subtraction share the read object — a clone here is
        // the same Python object, never a copy.
        let file_path = row.get_item("file_path")?;
        let video_id = row.get_item("video_id")?;
        let file_size = row.get_item("file_size")?;
        let triple = PyTuple::new(py, [video_id, file_path, file_size.clone()])?;
        out.append(triple)?;
        // total -= file_size — the augmented assignment's own semantics.
        total = py_inplace_sub(&total, &file_size)?;
    }
    Ok(out)
}

// ── Audio-cache maintenance engines (services/cache.py) ───────────────

/// The schema DDL script behind `CacheManager.init` — the pure half
/// of the cache index's creation step (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_cache_schema`): the module's
/// `_SCHEMA` text — the single `cache_index` table with its
/// `video_id` primary key, its `file_path` / `file_size` / `format`
/// columns and its `cached_at` / `last_accessed` datetime defaults —
/// handed to `executescript` verbatim, the leading newline and
/// indentation the fallback spells included. The connection handling
/// stays Python-side in the manager.
#[pyfunction]
fn cache_schema() -> String {
    "\nCREATE TABLE IF NOT EXISTS cache_index (\n    video_id TEXT PRIMARY KEY,\n    file_path TEXT NOT NULL,\n    file_size INTEGER NOT NULL,\n    format TEXT,\n    cached_at TEXT DEFAULT (datetime('now')),\n    last_accessed TEXT DEFAULT (datetime('now'))\n);\n".to_string()
}

/// The hit-counter batched-commit gate behind `CacheManager.get` —
/// the pure core of the hit path's tail (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_hit_commit_plan`): the count
/// incremented through CPython's own `+` and the sum compared against
/// the module's own `_COMMIT_EVERY` — read at call time from
/// `ytm_player.services.cache`, so a patched constant steers both
/// backends identically (the same pattern as
/// `stream_cache_max_size`) — a sum that reaches the constant
/// answering `(0, True)`: the batched commit is due and the stored
/// count resets, exactly the reference's commit-then-reset; a sum
/// still short answering `(sum, False)`. The `last_accessed` UPDATE
/// around the engine and the commit itself stay Python-side in the
/// manager.
#[pyfunction]
#[pyo3(signature = (hit_count))]
fn cache_hit_commit_plan<'py>(
    py: Python<'py>,
    hit_count: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let commit_every =
        PyModule::import(py, "ytm_player.services.cache")?.getattr("_COMMIT_EVERY")?;
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let count = hit_count.add(&one)?;
    if count.ge(&commit_every)? {
        let zero = 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any();
        return Ok((zero, true));
    }
    Ok((count, false))
}

/// The usage summary behind `CacheManager.get_status` — the pure core
/// of the status read (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_status_summary`): the
/// four-key payload in the reference's insertion order — the two
/// aggregates passed through as the objects the row reads answered,
/// the MB → bytes budget computed through CPython's own
/// `max_size_mb * 1024 * 1024` (both multiplications, their errors
/// included), and the cache directory rendered through CPython's own
/// `str()` (a str input passes through as the same object) — into a
/// fresh dict. The row reads and the aggregate assert stay
/// Python-side in the manager.
#[pyfunction]
#[pyo3(signature = (total_size, file_count, max_size_mb, cache_dir))]
fn cache_status_summary<'py>(
    py: Python<'py>,
    total_size: &Bound<'py, PyAny>,
    file_count: &Bound<'py, PyAny>,
    max_size_mb: &Bound<'py, PyAny>,
    cache_dir: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let k1024 = 1024i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let out = PyDict::new(py);
    out.set_item("total_size", total_size)?;
    out.set_item("file_count", file_count)?;
    out.set_item("max_size", max_size_mb.mul(&k1024)?.mul(&k1024)?)?;
    out.set_item("cache_dir", cache_dir.str()?)?;
    Ok(out)
}

/// The statement and bindings behind `CacheManager._index` — the
/// pure half of the index write (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_index_insert_statement`):
/// the `INSERT OR REPLACE` statement text — exactly as the fallback
/// spells it, leading newline and indentation included — and the
/// four-value parameter tuple: `video_id` / `file_size` / `format`
/// passed through as the objects they are, the path rendered through
/// CPython's own `str()` (a raising `__str__` keeps its own error,
/// exactly like the reference's `str()` call). The execute and the
/// commit stay Python-side in the manager.
#[pyfunction]
#[pyo3(signature = (video_id, path, file_size, format))]
fn cache_index_insert_statement<'py>(
    py: Python<'py>,
    video_id: &Bound<'py, PyAny>,
    path: &Bound<'py, PyAny>,
    file_size: &Bound<'py, PyAny>,
    format: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let params = PyTuple::new(
        py,
        [
            video_id.clone(),
            path.str()?.into_any(),
            file_size.clone(),
            format.clone(),
        ],
    )?;
    Ok((
        "\n        INSERT OR REPLACE INTO cache_index\n            (video_id, file_path, file_size, format)\n        VALUES (?, ?, ?, ?)\n        ".to_string(),
        params,
    ))
}

/// The statement and bindings behind `CacheManager.evict`'s index
/// cleanup — the pure half of the post-unlink write
/// (`services/cache.py`).
///
/// Mirrors the pure-Python reference (`_index_delete_statement`):
/// the video ids extracted from the plan's `(video_id, file_path,
/// file_size)` triples — each entry unpacked through the
/// reference's own `(video_id, _, _)` contract (any 3-iterable
/// works, a wrong-shaped entry raising CPython's own not-enough /
/// too-many `ValueError` and a non-iterable the unpack's own
/// `TypeError` rewrite; see `unpack_dispatch_row`) — the ids passing
/// through as the objects they are, in plan order — and the
/// placeholder-joined `IN` clause packed as the `(sql, ids)` pair
/// the reference's execute call takes: n ids spell n question
/// marks comma-joined, so an empty plan spells an `IN ()` the
/// manager's `if evicted:` gate never sends. The unlink loop and
/// the commit stay Python-side in the manager.
#[pyfunction]
#[pyo3(signature = (evicted))]
fn cache_index_delete_statement<'py>(
    py: Python<'py>,
    evicted: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let mut ids: Vec<Bound<'py, PyAny>> = Vec::new();
    for triple in evicted.try_iter()? {
        let (video_id, _file_path, _file_size) = unpack_dispatch_row(&triple?)?;
        ids.push(video_id);
    }
    let placeholders = std::iter::repeat_n("?", ids.len())
        .collect::<Vec<_>>()
        .join(",");
    let sql = format!("DELETE FROM cache_index WHERE video_id IN ({placeholders})");
    let out = PyList::new(py, ids).expect("list of ids cannot fail");
    Ok((sql, out))
}

/// The statement behind `CacheManager.clear`'s index wipe — the pure
/// half of the wipe's index write (`services/cache.py`).
///
/// Mirrors the pure-Python reference
/// (`_index_delete_all_statement`): the text handed to the
/// reference's execute call, verbatim. The file unlink walk and the
/// commit stay Python-side in the manager.
#[pyfunction]
fn cache_index_delete_all_statement() -> &'static str {
    "DELETE FROM cache_index"
}

// ── History row-assembly peripherals ───────────────────────────────────

/// The autocomplete string list behind
/// `HistoryManager.get_search_suggestions` — the pure core of the
/// suggestion read.
///
/// Mirrors the pure-Python reference (`_suggestion_strings`): the
/// `[row["query"] for row in rows]` comprehension — *rows* walked
/// through Python's iteration protocol (a non-iterable raises
/// CPython's own `TypeError`; an `__iter__` that itself raises keeps
/// its own error), each yielded row read through CPython's own
/// subscript (`row["query"]`, its `KeyError` wart included), and the
/// read values collected — as the objects they are — into a fresh
/// list. The input rows are never mutated.
#[pyfunction]
#[pyo3(signature = (rows))]
fn suggestion_strings<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for row in rows.try_iter()? {
        out.append(row?.get_item("query")?)?;
    }
    Ok(out)
}

/// The played-video-id set behind `HistoryManager.get_played_video_ids`
/// — the pure core of the Recently-Played filter read.
///
/// Mirrors the pure-Python reference (`_played_video_id_set`): the
/// `{row["video_id"] for row in rows}` set comprehension — *rows*
/// walked through Python's iteration protocol (a non-iterable raises
/// CPython's own `TypeError`; an `__iter__` that itself raises keeps
/// its own error), each yielded row read through CPython's own
/// subscript (`row["video_id"]`, its `KeyError` wart included), and
/// each read value inserted — as the object it is — into a fresh set
/// through CPython's own set-add: the value hashed first (an
/// unhashable value raises the hash protocol's own `TypeError`), then
/// the identity-or-`==` probe deciding membership, the FIRST inserted
/// of equal values the one the set keeps (`True` and `1` alias one
/// entry — whichever appeared first survives; the very same NaN
/// object collapses with itself, two distinct NaNs are two entries).
/// The input rows are never mutated.
#[pyfunction]
#[pyo3(signature = (rows))]
fn played_video_id_set<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PySet>> {
    let out = PySet::empty(py)?;
    for row in rows.try_iter()? {
        out.add(row?.get_item("video_id")?)?;
    }
    Ok(out)
}

/// The parameter tuple the play-history INSERT binds — the row
/// assembly behind `HistoryManager.log_play`'s first write.
///
/// Mirrors the pure-Python reference (`_play_insert_params`): the
/// seven values the `INSERT INTO play_history` statement binds, read
/// from the engine row through CPython's own subscript in the
/// reference's left-to-right order — `video_id`, `title`, `artist`,
/// `album`, `duration`, `listened_seconds`, `source` (a missing key's
/// `KeyError` propagating from exactly the read that missed) — and
/// passed through as the objects they are into a fresh tuple. The
/// input row is never mutated.
#[pyfunction]
#[pyo3(signature = (row))]
fn play_insert_params<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            row.get_item("video_id")?,
            row.get_item("title")?,
            row.get_item("artist")?,
            row.get_item("album")?,
            row.get_item("duration")?,
            row.get_item("listened_seconds")?,
            row.get_item("source")?,
        ],
    )
}

/// The parameter tuple the play-stats upsert binds — the row assembly
/// behind `HistoryManager.log_play`'s second write.
///
/// Mirrors the pure-Python reference (`_play_stats_params`): the four
/// values the `INSERT INTO play_stats ... ON CONFLICT` statement binds
/// — `video_id`, `title`, `artist`, `listened_seconds` — read through
/// CPython's own subscript in the reference's left-to-right order and
/// passed through as the objects they are into a fresh tuple. The
/// input row is never mutated.
#[pyfunction]
#[pyo3(signature = (row))]
fn play_stats_params<'py>(
    py: Python<'py>,
    row: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(
        py,
        [
            row.get_item("video_id")?,
            row.get_item("title")?,
            row.get_item("artist")?,
            row.get_item("listened_seconds")?,
        ],
    )
}

// ---------------------------------------------------------------------------
// Auth session-record engines (services/auth.py)
// ---------------------------------------------------------------------------

/// `hashlib.sha256(payload).hexdigest()` — the hash primitive behind
/// every session-record engine. Bytes (and their subclasses) digest
/// through the Rust implementation of the same SHA-256; every other
/// shape falls back to the interpreter's own
/// `hashlib.sha256(payload).hexdigest()` call — a bytearray or
/// memoryview hashes identically, a str payload raises hashlib's own
/// `TypeError` ("Unicode-objects must be encoded before hashing")
/// exactly like the reference's call would.
fn sha256_hex_impl(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(bytes) = payload.downcast::<PyBytes>() {
        let digest = Sha256::digest(bytes.as_bytes());
        return Ok(hex_lower(&digest));
    }
    let hashlib = py.import("hashlib")?;
    let digest = hashlib.getattr("sha256")?.call1((payload,))?;
    digest.getattr("hexdigest")?.call0()?.extract::<String>()
}

/// Lowercase hex — `bytes.hex()`'s own spelling of a digest.
fn hex_lower(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for b in data {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// A decode rejection — the two failure shapes `json.loads` surfaces
/// through the engines. Every `JSONDecodeError` the scanner can raise
/// collapses to `Reject`: the engines swallow the message (the
/// reference's except arm answers `None`), so the accept/reject
/// boundary is the whole observable contract and CPython's positions
/// and spellings never surface. `Py` carries the errors the reference
/// lets propagate — the `RecursionError` of past-budget nesting, an
/// `int()` conversion's own `ValueError`.
enum JsonReject {
    /// A `json.JSONDecodeError`-shaped rejection.
    Reject,
    /// A propagating Python error raised during object construction.
    Py(PyErr),
}

/// One decoded container frame — the explicit stack that keeps the
/// walk free of Rust recursion: a payload nested past CPython's own
/// recursion budget raises the interpreter's `RecursionError` through
/// the depth counter instead of overflowing the Rust stack.
enum JsonLoadFrame {
    Array {
        items: Vec<Py<PyAny>>,
    },
    Object {
        pairs: Vec<(Py<PyAny>, Py<PyAny>)>,
        /// The key whose value the scanner sits between `:` and.
        pending_key: Option<Py<PyAny>>,
    },
}

/// `json.loads(s)` over a decoded str's code points — CPython's own
/// scanner grammar: the four whitespace characters (` `, `\t`, `\n`,
/// `\r`) skipped between tokens; objects and arrays with their
/// delimiter rules (a trailing comma rejected, an object key always a
/// string); strings through the escape table (`\" \\ \/ \b \f \n \r
/// \t`, `\uXXXX` with surrogate pairing — a high surrogate followed by
/// a `\u` escape combines only with a low one, every other shape
/// keeping its lone surrogate exactly like the scanner's slice-peek
/// rule) and raw control characters rejected; the literals
/// null/true/false/NaN/Infinity/-Infinity prefix-matched like the
/// scanner's own slice equality, so trailing junk rejects at the
/// delimiter exactly where the reference's does; numbers through the
/// leading-zero rule with digit-required fraction and exponent,
/// arbitrary-precision integers built through the interpreter's own
/// `int()` and floats through the same correctly-rounded decimal
/// conversion (an absurd exponent overflowing to `inf` silently);
/// duplicate object keys keeping the first position with the last
/// value through the dict store's own replace; and nesting counted
/// against `sys.getrecursionlimit()` with the same `RecursionError`
/// the reference's decoder raises for a past-budget payload — the
/// reference's exact borderline sits at the interpreter's live
/// recursion budget (the limit minus the frames already on the stack),
/// so an input nested within a frame or two of the limit may accept
/// here where the reference raises; every realistic depth answers
/// identically.
fn json_loads<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
) -> Result<Bound<'py, PyAny>, JsonReject> {
    // The text's code points — the fast lane for every str (its UTF-8
    // content), the surrogatepass round trip only for a str carrying
    // lone surrogates (which a plain bytes.decode never produces; only
    // a decode override could return one).
    let cps: Vec<u32> = match text.to_str() {
        Ok(s) => s.chars().map(|c| c as u32).collect(),
        Err(_) => {
            let bytes = text
                .call_method1("encode", ("utf-8", "surrogatepass"))
                .map_err(JsonReject::Py)?;
            let bytes = bytes
                .downcast::<PyBytes>()
                .map_err(|_| JsonReject::Py(PyTypeError::new_err("encode did not return bytes")))?;
            wtf8_code_points(bytes.as_bytes())
        }
    };
    let limit: usize = py
        .import("sys")
        .and_then(|sys| sys.getattr("getrecursionlimit"))
        .and_then(|f| f.call0())
        .and_then(|v| v.extract::<usize>())
        .map_err(JsonReject::Py)?;
    let int_type = py
        .import("builtins")
        .and_then(|b| b.getattr("int"))
        .map_err(JsonReject::Py)?;
    let mut scanner = JsonScanner {
        py,
        cps,
        i: 0,
        depth: 0,
        limit,
        int_type,
    };
    scanner.scan()
}

/// The WTF-8 code-point reader — plain UTF-8 shapes plus the surrogate
/// range as three-byte sequences, the mirror of the writer the
/// `ensure_ascii` escaping already uses.
fn wtf8_code_points(data: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
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
        out.push(c);
    }
    out
}

/// The scanner state — the code points, the cursor, the nesting depth
/// against the interpreter's recursion limit, and the `int` type for
/// arbitrary-precision number construction.
struct JsonScanner<'py> {
    py: Python<'py>,
    cps: Vec<u32>,
    i: usize,
    depth: usize,
    limit: usize,
    int_type: Bound<'py, PyAny>,
}

impl<'py> JsonScanner<'py> {
    /// The scanner's whitespace: exactly ` `, `\t`, `\n`, `\r`.
    fn ws(&mut self) {
        while self.i < self.cps.len() && matches!(self.cps[self.i], 0x20 | 0x09 | 0x0a | 0x0d) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u32> {
        self.cps.get(self.i).copied()
    }

    /// The scanner's literal match — a slice-equality prefix check that
    /// consumes the word only when every character lines up, trailing
    /// characters left for the delimiter rules to reject.
    fn literal(&mut self, word: &[u8]) -> bool {
        if self.i + word.len() > self.cps.len() {
            return false;
        }
        for (k, b) in word.iter().enumerate() {
            if self.cps[self.i + k] != *b as u32 {
                return false;
            }
        }
        self.i += word.len();
        true
    }

    /// `Py_EnterRecursiveCall`'s counter — the nesting budget.
    fn enter(&mut self) -> Result<(), JsonReject> {
        self.depth += 1;
        if self.depth > self.limit {
            return Err(JsonReject::Py(PyRecursionError::new_err(
                "maximum recursion depth exceeded while decoding a JSON object",
            )));
        }
        Ok(())
    }

    /// Four hex digits — the `\uXXXX` escape's own validation.
    fn hex4(&mut self) -> Result<u32, JsonReject> {
        if self.i + 4 > self.cps.len() {
            return Err(JsonReject::Reject);
        }
        let mut v = 0u32;
        for k in 0..4 {
            let c = self.cps[self.i + k];
            let d = match c {
                0x30..=0x39 => c - 0x30,
                0x41..=0x46 => c - 0x41 + 10,
                0x61..=0x66 => c - 0x61 + 10,
                _ => return Err(JsonReject::Reject),
            };
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    /// One JSON string — the cursor sits on the opening quote. Escapes
    /// through the scanner's table, `\uXXXX` with the pairing rule (a
    /// high surrogate followed by a `\u` escape peeks the second; only
    /// a low surrogate combines, anything else leaves the first escape
    /// standing and the second scanned as its own), raw control
    /// characters rejected, lone surrogates kept. The built str takes
    /// the fast lane when every code point is a character and the
    /// interpreter's own `surrogatepass` decode of the WTF-8 bytes
    /// otherwise, so a lone `\ud800` escapes into the result exactly
    /// like the reference's scanner would build it.
    fn string(&mut self) -> Result<Bound<'py, PyString>, JsonReject> {
        let py = self.py;
        self.i += 1; // the opening quote
        let mut out: Vec<u32> = Vec::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return Err(JsonReject::Reject), // unterminated
            };
            match c {
                0x22 => {
                    self.i += 1;
                    break;
                }
                0x5c => {
                    self.i += 1;
                    let esc = match self.peek() {
                        Some(e) => e,
                        None => return Err(JsonReject::Reject),
                    };
                    match esc {
                        0x22 | 0x5c | 0x2f => {
                            out.push(esc);
                            self.i += 1;
                        }
                        0x62 => {
                            out.push(0x08);
                            self.i += 1;
                        }
                        0x66 => {
                            out.push(0x0c);
                            self.i += 1;
                        }
                        0x6e => {
                            out.push(0x0a);
                            self.i += 1;
                        }
                        0x72 => {
                            out.push(0x0d);
                            self.i += 1;
                        }
                        0x74 => {
                            out.push(0x09);
                            self.i += 1;
                        }
                        0x75 => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            let mut uni = hi;
                            if (0xd800..=0xdbff).contains(&hi)
                                && self.i + 1 < self.cps.len()
                                && self.cps[self.i] == 0x5c
                                && self.cps[self.i + 1] == 0x75
                            {
                                // The scanner's peek: `s[end:end+2] == '\\u'`.
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..=0xdfff).contains(&lo) {
                                    uni = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    // Not a low surrogate: the first escape
                                    // stands alone, the second is its own.
                                    self.i = save;
                                }
                            }
                            out.push(uni);
                        }
                        _ => return Err(JsonReject::Reject), // invalid escape
                    }
                }
                c if c < 0x20 => return Err(JsonReject::Reject), // control character
                c => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
        if out.iter().all(|&c| char::from_u32(c).is_some()) {
            let s: String = out.iter().filter_map(|&c| char::from_u32(c)).collect();
            Ok(PyString::new(py, &s))
        } else {
            let mut bytes: Vec<u8> = Vec::with_capacity(out.len());
            for &c in &out {
                match char::from_u32(c) {
                    Some(ch) => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    None => {
                        // A lone surrogate as its three-byte WTF-8 shape.
                        bytes.push(0xed);
                        bytes.push(0xa0 | (((c >> 6) & 0x1f) as u8));
                        bytes.push(0x80 | (c & 0x3f) as u8);
                    }
                }
            }
            let b = PyBytes::new(py, &bytes);
            let s = b
                .call_method1("decode", ("utf-8", "surrogatepass"))
                .map_err(JsonReject::Py)?;
            s.downcast_into::<PyString>()
                .map_err(|_| JsonReject::Py(PyTypeError::new_err("decode did not return str")))
        }
    }

    /// One object key with its colon — the cursor sits before the key.
    fn object_key(&mut self) -> Result<Bound<'py, PyString>, JsonReject> {
        self.ws();
        if self.peek() != Some(0x22) {
            return Err(JsonReject::Reject); // expecting a property name
        }
        let key = self.string()?;
        self.ws();
        if self.peek() != Some(0x3a) {
            return Err(JsonReject::Reject); // expecting ':'
        }
        self.i += 1;
        Ok(key)
    }

    /// One number — the grammar's own shape: an optional sign, `0` or a
    /// non-zero-leading digit run, a digit-required fraction, a
    /// digit-required exponent. Integers build through the
    /// interpreter's own `int()` (arbitrary precision, `-0` the int 0);
    /// floats through the same correctly-rounded decimal conversion
    /// (`1e999` the float `inf`, never an error).
    fn number(&mut self) -> Result<Bound<'py, PyAny>, JsonReject> {
        let start = self.i;
        if self.peek() == Some(0x2d) {
            self.i += 1;
        }
        match self.peek() {
            Some(0x30) => self.i += 1,
            Some(0x31..=0x39) => {
                while matches!(self.peek(), Some(0x30..=0x39)) {
                    self.i += 1;
                }
            }
            _ => return Err(JsonReject::Reject),
        }
        let mut is_float = false;
        if self.peek() == Some(0x2e) {
            self.i += 1;
            if !matches!(self.peek(), Some(0x30..=0x39)) {
                return Err(JsonReject::Reject);
            }
            while matches!(self.peek(), Some(0x30..=0x39)) {
                self.i += 1;
            }
            is_float = true;
        }
        if matches!(self.peek(), Some(0x65) | Some(0x45)) {
            self.i += 1;
            if matches!(self.peek(), Some(0x2b) | Some(0x2d)) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(0x30..=0x39)) {
                return Err(JsonReject::Reject);
            }
            while matches!(self.peek(), Some(0x30..=0x39)) {
                self.i += 1;
            }
            is_float = true;
        }
        let token: String = self.cps[start..self.i]
            .iter()
            .map(|&c| char::from_u32(c).unwrap_or('\u{fffd}'))
            .collect();
        let py = self.py;
        if is_float {
            let v: f64 = token.parse().map_err(|_| JsonReject::Reject)?;
            Ok(PyFloat::new(py, v).into_any())
        } else {
            Ok(self.int_type.call1((&token,)).map_err(JsonReject::Py)?)
        }
    }

    /// The document walk — produce a value, consume it into the frame
    /// stack, close containers on their delimiters, and answer the
    /// single top-level value with the trailing-garbage rejection.
    fn scan(&mut self) -> Result<Bound<'py, PyAny>, JsonReject> {
        let py = self.py;
        let mut stack: Vec<JsonLoadFrame> = Vec::new();
        let mut current: Option<Bound<'py, PyAny>> = None;
        loop {
            // ── produce ── a value, an opened container, or a finished
            // container's aggregate.
            if current.is_none() {
                self.ws();
                let c = match self.peek() {
                    Some(c) => c,
                    None => return Err(JsonReject::Reject), // expecting a value
                };
                match c {
                    0x7b => {
                        self.i += 1;
                        self.enter()?;
                        self.ws();
                        if self.peek() == Some(0x7d) {
                            self.i += 1;
                            self.depth -= 1;
                            current = Some(PyDict::new(py).into_any());
                        } else {
                            let key = self.object_key()?;
                            stack.push(JsonLoadFrame::Object {
                                pairs: Vec::new(),
                                pending_key: Some(key.unbind().into_any()),
                            });
                        }
                    }
                    0x5b => {
                        self.i += 1;
                        self.enter()?;
                        self.ws();
                        if self.peek() == Some(0x5d) {
                            self.i += 1;
                            self.depth -= 1;
                            current = Some(PyList::empty(py).into_any());
                        } else {
                            stack.push(JsonLoadFrame::Array { items: Vec::new() });
                        }
                    }
                    0x22 => {
                        let s = self.string()?;
                        current = Some(s.into_any());
                    }
                    0x74 => {
                        if self.literal(b"true") {
                            current = Some(PyBool::new(py, true).to_owned().into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x66 => {
                        if self.literal(b"false") {
                            current = Some(PyBool::new(py, false).to_owned().into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x6e => {
                        if self.literal(b"null") {
                            current = Some(py.None().into_bound(py));
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x4e => {
                        if self.literal(b"NaN") {
                            current = Some(PyFloat::new(py, f64::NAN).into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x49 => {
                        if self.literal(b"Infinity") {
                            current = Some(PyFloat::new(py, f64::INFINITY).into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x2d => {
                        if self.cps.get(self.i + 1) == Some(&0x49) {
                            if self.literal(b"-Infinity") {
                                current = Some(PyFloat::new(py, f64::NEG_INFINITY).into_any());
                            } else {
                                return Err(JsonReject::Reject);
                            }
                        } else {
                            let v = self.number()?;
                            current = Some(v);
                        }
                    }
                    0x30..=0x39 => {
                        let v = self.number()?;
                        current = Some(v);
                    }
                    _ => return Err(JsonReject::Reject),
                }
            }
            // ── consume ── the produced value into the stack, or answer
            // it at the top level.
            if let Some(value) = current.take() {
                if stack.is_empty() {
                    self.ws();
                    if self.i != self.cps.len() {
                        return Err(JsonReject::Reject); // extra data
                    }
                    return Ok(value);
                }
                let is_object = matches!(stack.last(), Some(JsonLoadFrame::Object { .. }));
                if is_object {
                    let Some(JsonLoadFrame::Object { pairs, pending_key }) = stack.last_mut()
                    else {
                        unreachable!("frame kind checked above");
                    };
                    let key = pending_key.take().expect("object frame awaits its value");
                    pairs.push((key, value.unbind()));
                } else {
                    let Some(JsonLoadFrame::Array { items }) = stack.last_mut() else {
                        unreachable!("frame kind checked above");
                    };
                    items.push(value.unbind());
                }
                self.ws();
                match self.peek() {
                    Some(0x2c) => {
                        self.i += 1;
                        if is_object {
                            let key = self.object_key()?;
                            if let Some(JsonLoadFrame::Object { pending_key, .. }) =
                                stack.last_mut()
                            {
                                *pending_key = Some(key.unbind().into_any());
                            }
                        }
                    }
                    Some(0x5d) | Some(0x7d) => {
                        let closing_array = self.peek() == Some(0x5d);
                        if closing_array == is_object {
                            return Err(JsonReject::Reject); // wrong closer
                        }
                        self.i += 1;
                        self.depth -= 1;
                        let frame = stack.pop().expect("frame exists");
                        current = Some(match frame {
                            JsonLoadFrame::Array { items } => {
                                PyList::new(py, items).map_err(JsonReject::Py)?.into_any()
                            }
                            JsonLoadFrame::Object { pairs, .. } => {
                                let d = PyDict::new(py);
                                // The dict store's own replace: a repeated
                                // key keeps its first position and takes
                                // the last value.
                                for (k, v) in pairs {
                                    d.set_item(k, v).map_err(JsonReject::Py)?;
                                }
                                d.into_any()
                            }
                        });
                    }
                    _ => return Err(JsonReject::Reject), // expecting a delimiter
                }
            }
        }
    }
}

/// The pure core of `services/auth.py`'s `_parse_record` — shared by
/// the exported engine and the composite engines below.
///
/// Mirrors the pure-Python reference: `raw.decode("utf-8")` through the
/// object's own decode method (a `UnicodeDecodeError` is the engine's
/// `None`; a non-decodable payload's `AttributeError` propagates),
/// `json.loads` over the decoded text (the scanner above — every
/// `JSONDecodeError` shape the engine's `None`), then the schema gates:
/// the top value must be a dict and its `schema_version` must compare
/// equal to the constant through CPython's own `==` — so `1.0` and
/// `True` pass exactly like they do in the reference. A decode override
/// returning a non-str falls through to the interpreter's own
/// `json.loads` call so that exotic path keeps the reference's exact
/// behavior, its `JSONDecodeError`/`UnicodeDecodeError` answered with
/// `None` and everything else propagating.
fn parse_record_impl<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if raw.is_none() {
        return Ok(None);
    }
    let decoded = match raw.getattr("decode").and_then(|d| d.call1(("utf-8",))) {
        Ok(d) => d,
        Err(err) => {
            if err.is_instance_of::<PyUnicodeDecodeError>(py) {
                return Ok(None);
            }
            return Err(err);
        }
    };
    let value = match decoded.downcast::<PyString>() {
        Ok(text) => match json_loads(py, text) {
            Ok(v) => v,
            Err(JsonReject::Reject) => return Ok(None),
            Err(JsonReject::Py(err)) => return Err(err),
        },
        Err(_) => {
            let json_module = py.import("json")?;
            let loads = json_module.getattr("loads")?;
            let jde_class = json_module.getattr("JSONDecodeError")?;
            match loads.call1((decoded.clone(),)) {
                Ok(v) => v,
                Err(err) => {
                    let is_json_error = jde_class.is_instance(err.value(py)).unwrap_or(false);
                    if is_json_error || err.is_instance_of::<PyUnicodeDecodeError>(py) {
                        return Ok(None);
                    }
                    return Err(err);
                }
            }
        }
    };
    if !value.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    // data.get("schema_version") — the dict's own get, its None the
    // missing-key answer, compared through CPython's `!=`.
    let schema = value.getattr("get")?.call1(("schema_version",))?;
    if schema.ne(1i64)? {
        return Ok(None);
    }
    Ok(Some(value))
}

/// The account.json record parser — the engine behind every read of
/// the sign-in sidecar.
///
/// Mirrors the pure-Python reference (`_parse_record`): account.json
/// bytes as a dict — or `None` when the file is missing (the `None`
/// passthrough), malformed, of another schema, or undecodable — with
/// the JSON grammar, the schema gate and the exact comparison quirks
/// described on `parse_record_impl`. The file read stays Python-side
/// in the manager.
#[pyfunction]
#[pyo3(signature = (raw))]
fn parse_record<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    parse_record_impl(py, raw)
}

/// The recorded-identity validation — the engine behind the renewal
/// gate that ties an automatic refresh to the account `ytm setup`
/// chose.
///
/// Mirrors the pure-Python reference (`_identity_from_record`): the
/// `verified` field read through CPython's own subscript and gated by
/// identity with the `False` singleton — `is` semantics, so a `0` or
/// any other falsy-but-not-`False` object passes exactly like it does
/// in the reference; the `channel_id` read next, accepted only when it
/// is a str whose whole value matches the channel-id pattern
/// (a str carrying lone surrogates cannot match the class, answering
/// like the reference's fullmatch); the `x-goog-authuser` slot read
/// last, accepted only when it is a str of digits under CPython's own
/// `str.isdigit()` rule (Unicode digits included) and converted
/// through the interpreter's own `int()` — so `int('٣')` keeps its
/// value and an isdigit-true, int-false spelling (a superscript digit)
/// raises `int()`'s own `ValueError` exactly like the reference. The
/// result is the `(slot, channel_id)` pair with both objects passed
/// through as they are, or `None`.
#[pyfunction]
#[pyo3(signature = (record))]
fn identity_from_record<'py>(
    py: Python<'py>,
    record: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if record.is_none() {
        return Ok(None);
    }
    let get = |key: &str| -> PyResult<Bound<'py, PyAny>> { record.getattr("get")?.call1((key,)) };
    let verified = get("verified")?;
    let false_singleton = PyBool::new(py, false).to_owned();
    if verified.is(&false_singleton) {
        return Ok(None);
    }
    let channel_id = get("channel_id")?;
    let slot = get("x-goog-authuser")?;
    let channel_ok = match channel_id.downcast::<PyString>() {
        Ok(s) => s
            .to_str()
            .map(|t| CHANNEL_ID_RE.is_match(t))
            .unwrap_or(false),
        Err(_) => false,
    };
    if !channel_ok {
        return Ok(None);
    }
    let slot_ok = match slot.downcast::<PyString>() {
        Ok(s) => s.to_str().map(py_isdigit).unwrap_or(false),
        Err(_) => false,
    };
    if !slot_ok {
        return Ok(None);
    }
    let slot_int = py.import("builtins")?.getattr("int")?.call1((slot,))?;
    Ok(Some(PyTuple::new(py, [slot_int, channel_id.clone()])?))
}

/// The ownership snapshot — the engine behind the commit guard that
/// refuses an automatic renewal when the sign-in files changed while
/// it was being renewed.
///
/// Mirrors the pure-Python reference (`_owner_of`): no auth.json means
/// no owner (`None`); otherwise the record parses, the auth bytes hash
/// through the SHA-256 primitive, and the record's `revision` is
/// answered only when the record is truthy and its `auth_sha256`
/// compares equal to that hash through CPython's own `==` — and only
/// when the revision itself is a str. The result is the
/// `(auth_sha256, revision)` pair (the revision `None` when absent),
/// the hash a fresh hex str.
#[pyfunction]
#[pyo3(signature = (auth_bytes, record_raw))]
fn owner_of<'py>(
    py: Python<'py>,
    auth_bytes: &Bound<'py, PyAny>,
    record_raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if auth_bytes.is_none() {
        return Ok(None);
    }
    let record = parse_record_impl(py, record_raw)?;
    let sha = PyString::new(py, &sha256_hex_impl(py, auth_bytes)?);
    let revision = match &record {
        Some(record) if record.is_truthy()? => {
            let recorded = record.getattr("get")?.call1(("auth_sha256",))?;
            if recorded.eq(&sha)? {
                let revision = record.getattr("get")?.call1(("revision",))?;
                if revision.is_instance_of::<PyString>() {
                    revision
                } else {
                    py.None().into_bound(py)
                }
            } else {
                py.None().into_bound(py)
            }
        }
        _ => py.None().into_bound(py),
    };
    Ok(Some(PyTuple::new(py, [sha.into_any(), revision])?))
}

/// The stream-jar vouching decision tree — the engine behind
/// `read_vouched_stream_jar`'s refusal reasons.
///
/// Mirrors the pure-Python reference (`_vouch_failure`) read for read:
/// a jar without data (missing and unreadable alike) is unreadable; an
/// auth.json error is unreadable; missing auth.json has no trust; an
/// account.json error renders through the interpreter's own `str()`
/// into its refusal reason; a missing record is the accepted legacy
/// shape; a record that does not parse is malformed; a record whose
/// `auth_sha256` does not compare equal to the hash of the auth bytes
/// (CPython's own `!=`) does not describe them; a record without the
/// `stream_sha256` key is the other accepted legacy shape; a null hash
/// means the sign-in has no jar; and a hash that does not compare equal
/// to the jar's is a mismatch. `None` vouches for the jar. The file
/// reads (and their stat triples) stay Python-side in the caller.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the three reads' flat (data, error) pairs
#[pyo3(signature = (auth_data, auth_error, record_data, record_error, jar_data, jar_error))]
fn vouch_failure<'py>(
    py: Python<'py>,
    auth_data: &Bound<'py, PyAny>,
    auth_error: &Bound<'py, PyAny>,
    record_data: &Bound<'py, PyAny>,
    record_error: &Bound<'py, PyAny>,
    jar_data: &Bound<'py, PyAny>,
    jar_error: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    // The jar's error component is never read — the reference gates on
    // the jar's data alone (missing and unreadable alike).
    let _ = jar_error;
    if jar_data.is_none() {
        return Ok(Some("the stream cookie file is unreadable".to_string()));
    }
    if !auth_error.is_none() {
        return Ok(Some("auth.json is unreadable".to_string()));
    }
    if auth_data.is_none() {
        return Ok(Some("auth.json is missing".to_string()));
    }
    if !record_error.is_none() {
        let rendered = py
            .import("builtins")?
            .getattr("str")?
            .call1((record_error,))?;
        let rendered: String = rendered.extract()?;
        return Ok(Some(format!("account.json cannot be read ({rendered})")));
    }
    if record_data.is_none() {
        return Ok(None);
    }
    let Some(record) = parse_record_impl(py, record_data)? else {
        return Ok(Some("account.json is malformed".to_string()));
    };
    let auth_sha = PyString::new(py, &sha256_hex_impl(py, auth_data)?);
    let recorded_sha = record.getattr("get")?.call1(("auth_sha256",))?;
    if recorded_sha.ne(&auth_sha)? {
        return Ok(Some(
            "account.json does not describe the current auth.json".to_string(),
        ));
    }
    if !record.contains("stream_sha256")? {
        return Ok(None);
    }
    // record_dict["stream_sha256"] — the plain subscript the `in`
    // check just vouched for.
    let expected = record.get_item("stream_sha256")?;
    if expected.is_none() {
        return Ok(Some(
            "the saved sign-in has no stream cookie file".to_string(),
        ));
    }
    let jar_sha = PyString::new(py, &sha256_hex_impl(py, jar_data)?);
    if expected.ne(&jar_sha)? {
        return Ok(Some(
            "the stream cookie file does not match the saved sign-in".to_string(),
        ));
    }
    Ok(None)
}

/// The account_menu channel-id extraction — the engine behind the
/// setup probe that learns which YouTube channel a browser slot's
/// session belongs to.
///
/// Mirrors the pure-Python reference
/// (`_channel_id_from_account_menu`): the lookup chain through
/// CPython's own subscript with the reference's caught arms
/// (`KeyError`, `IndexError`, `TypeError`) answered with `None` and
/// every other error kept; the header gate (a non-dict menu, or an
/// `activeAccountHeaderRenderer` missing from the header's own
/// `in`); the items gate (exactly a list); then the defensive walk —
/// each item's `compactLinkRenderer` fetched through the dict's own
/// `get` when the item is a dict, the ACCOUNT_BOX icon, the
/// navigation endpoint's browse endpoint, the browse id validated by
/// the channel-id pattern, and the page type read through the
/// reference's `(configs or {}).get(..., {}).get("pageType")` chain —
/// the id check and the `==` compared with CPython's own operators,
/// the page-type `==` evaluated only when the id matched (the
/// reference's `and` chain short-circuits before it). Exactly one
/// matching link answers its browse id; zero or several answer `None`
/// — the caller fails closed.
#[pyfunction]
#[pyo3(signature = (response))]
fn channel_id_from_account_menu<'py>(
    py: Python<'py>,
    response: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyString>>> {
    let chain: PyResult<Option<Bound<'py, PyAny>>> = (|| {
        let menu = response
            .get_item("actions")?
            .get_item(0)?
            .get_item("openPopupAction")?
            .get_item("popup")?
            .get_item("multiPageMenuRenderer")?;
        if !menu.is_instance_of::<PyDict>() {
            return Ok(None);
        }
        let header = menu.get_item("header")?;
        if !header.contains("activeAccountHeaderRenderer")? {
            return Ok(None);
        }
        Ok(Some(
            menu.get_item("sections")?
                .get_item(0)?
                .get_item("multiPageMenuSectionRenderer")?
                .get_item("items")?,
        ))
    })();
    let items = match chain {
        Ok(Some(items)) => items,
        Ok(None) => return Ok(None),
        Err(err) => {
            if err.is_instance_of::<PyKeyError>(py)
                || err.is_instance_of::<PyIndexError>(py)
                || err.is_instance_of::<PyTypeError>(py)
            {
                return Ok(None);
            }
            return Err(err);
        }
    };
    if !items.is_instance_of::<PyList>() {
        return Ok(None);
    }
    let mut found: Vec<Bound<'py, PyString>> = Vec::new();
    for item in items.try_iter()? {
        let item = item?;
        let link = if item.is_instance_of::<PyDict>() {
            item.getattr("get")?.call1(("compactLinkRenderer",))?
        } else {
            py.None().into_bound(py)
        };
        if !link.is_instance_of::<PyDict>() {
            continue;
        }
        let icon = link.getattr("get")?.call1(("icon",))?;
        if !icon.is_instance_of::<PyDict>()
            || icon
                .getattr("get")?
                .call1(("iconType",))?
                .ne("ACCOUNT_BOX")?
        {
            continue;
        }
        let endpoint = link.getattr("get")?.call1(("navigationEndpoint",))?;
        let browse = if endpoint.is_instance_of::<PyDict>() {
            endpoint.getattr("get")?.call1(("browseEndpoint",))?
        } else {
            py.None().into_bound(py)
        };
        if !browse.is_instance_of::<PyDict>() {
            continue;
        }
        let browse_id = browse.getattr("get")?.call1(("browseId",))?;
        let configs = browse
            .getattr("get")?
            .call1(("browseEndpointContextSupportedConfigs",))?;
        let configs = if configs.is_truthy()? {
            configs
        } else {
            PyDict::new(py).into_any()
        };
        let music = configs
            .getattr("get")?
            .call1(("browseEndpointContextMusicConfig", PyDict::new(py)))?;
        let page_type = music.getattr("get")?.call1(("pageType",))?;
        if let Ok(id) = browse_id.downcast::<PyString>() {
            if id
                .to_str()
                .map(|t| CHANNEL_ID_RE.is_match(t))
                .unwrap_or(false)
                && page_type.eq("MUSIC_PAGE_TYPE_USER_CHANNEL")?
            {
                found.push(id.clone());
            }
        }
    }
    if found.len() != 1 {
        return Ok(None);
    }
    Ok(Some(found.into_iter().next().expect("len checked above")))
}

/// The canonical auth.json serialisation — the one byte spelling of a
/// header dict, probed and committed as is.
///
/// Mirrors the pure-Python reference (`_serialize_headers`):
/// `json.dumps(headers, ensure_ascii=True, indent=4, sort_keys=True)`
/// — the strict encoder (the reference's own `TypeError` for an
/// unserialisable value or an unorderable key), the sorted-key layout
/// through CPython's own tuple comparison, the four-space indent with
/// the `(',', ': ')` separators, and the ASCII escaping — encoded to
/// UTF-8 bytes (the ASCII-only text's own encoding).
#[pyfunction]
#[pyo3(signature = (headers))]
fn serialize_headers(py: Python<'_>, headers: &Bound<'_, PyAny>) -> PyResult<Py<PyBytes>> {
    let text = json_encode(py, headers, "    ".to_string(), true, true, true)?;
    Ok(PyBytes::new(py, text.as_bytes()).unbind())
}

/// The candidate auth.json bytes for one browser slot — the payload
/// the setup probe writes and commits unchanged.
///
/// Mirrors the pure-Python reference (`_candidate_payload`):
/// `{**base_headers, "x-goog-authuser": str(slot)}` — the dict merge
/// (a dict walked through its own storage, any other mapping through
/// its `keys()`, a repeated key keeping its position with the new
/// value), the slot rendered through the interpreter's own `str()`,
/// then the canonical serialisation above.
#[pyfunction]
#[pyo3(signature = (base_headers, slot))]
fn candidate_payload(
    py: Python<'_>,
    base_headers: &Bound<'_, PyAny>,
    slot: &Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    let merged = PyDict::new(py);
    if base_headers.is_instance_of::<PyDict>() {
        let base = base_headers
            .downcast::<PyDict>()
            .expect("isinstance-checked dict");
        for (key, value) in base.iter() {
            merged.set_item(key, value)?;
        }
    } else {
        let keys = base_headers.getattr("keys")?.call0()?;
        for key in keys.try_iter()? {
            let key = key?;
            merged.set_item(&key, base_headers.get_item(&key)?)?;
        }
    }
    let slot_str = py.import("builtins")?.getattr("str")?.call1((slot,))?;
    merged.set_item("x-goog-authuser", slot_str)?;
    let text = json_encode(py, &merged, "    ".to_string(), true, true, true)?;
    Ok(PyBytes::new(py, text.as_bytes()).unbind())
}

/// The account.json record assembly — the dict the commit writes for
/// exactly the auth bytes it publishes.
///
/// Mirrors the pure-Python reference (`_build_record`): the fixed
/// insertion order (`schema_version`, `x-goog-authuser` through the
/// interpreter's own `str()`, `channel_id` and `name` as the objects
/// they are, `handle` through the truthiness gate — a falsy handle
/// becomes `None` — `verified` as passed, the auth bytes' SHA-256 hex,
/// the stream jar's hash as passed, and a fresh `revision` from the
/// interpreter's own `os.urandom(16).hex()`). The atomic write itself
/// stays Python-side in the commit.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the record's seven fields, one parameter each
#[pyo3(signature = (slot, name, handle, channel_id, verified, auth_payload, stream_sha256))]
fn build_record<'py>(
    py: Python<'py>,
    slot: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    handle: &Bound<'py, PyAny>,
    channel_id: &Bound<'py, PyAny>,
    verified: &Bound<'py, PyAny>,
    auth_payload: &Bound<'py, PyAny>,
    stream_sha256: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("schema_version", 1i64)?;
    let slot_str = py.import("builtins")?.getattr("str")?.call1((slot,))?;
    out.set_item("x-goog-authuser", slot_str)?;
    out.set_item("channel_id", channel_id)?;
    out.set_item("name", name)?;
    let handle_value = if handle.is_truthy()? {
        handle.clone()
    } else {
        py.None().into_bound(py)
    };
    out.set_item("handle", handle_value)?;
    out.set_item("verified", verified)?;
    let sha = sha256_hex_impl(py, auth_payload)?;
    out.set_item("auth_sha256", sha)?;
    out.set_item("stream_sha256", stream_sha256)?;
    let urandom = py.import("os")?.getattr("urandom")?.call1((16,))?;
    let revision = urandom.getattr("hex")?.call0()?;
    out.set_item("revision", revision)?;
    Ok(out)
}

/// The SHA-256 hex primitive — `hashlib.sha256(payload).hexdigest()`
/// as a standalone engine (see `sha256_hex_impl` for the shape rules).
#[pyfunction]
#[pyo3(signature = (payload))]
fn sha256_hex(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<String> {
    sha256_hex_impl(py, payload)
}

// ── Manual-paste header normalization (services/auth.py) ─────────────────

/// The HTTP/2 pseudo-headers Chrome DevTools prefixes its request
/// listings with — dropped from every paste format (mirrors
/// `auth._PSEUDO_HEADERS`, exact string membership).
const PSEUDO_HEADERS: [&str; 5] = [":authority", ":method", ":path", ":scheme", ":status"];

/// Drop Chrome's decoded-protobuf annotation for ``x-client-data``
/// (mirrors `auth._strip_decoded_blocks`): every ``Decoded:`` line
/// with a matching bare ``}`` line later in the input drops both ends
/// and everything between; an unterminated block is left untouched
/// (dropping to end-of-input would discard real headers, and the
/// caller degrades better on a parity shift than on missing lines).
/// Both markers compare after `str.strip()` (CPython's own whitespace
/// set — see `py_strip`).
fn strip_decoded_blocks(lines: Vec<&str>) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if py_strip(lines[i]) != "Decoded:" {
            out.push(lines[i]);
            i += 1;
            continue;
        }
        // next(j for j in range(i + 1, len(lines)) if … == "}") — the
        // first closing brace strictly after the opener.
        match (i + 1..lines.len()).find(|&j| py_strip(lines[j]) == "}") {
            None => {
                out.push(lines[i]);
                i += 1;
            }
            Some(close) => i = close + 1,
        }
    }
    out
}

/// The manual-paste header pre-processing — `services/auth.py`'s
/// `_normalize_raw_headers`, behind `_setup_manual`'s paste loop.
///
/// Mirrors the reference's three paste formats read for read: the
/// single-line terminal paste split on its escape separator (the
/// literal ``^[E`` winning over the real ``ESC E`` over the bare
/// ``ESC``, whatever the paste actually carried), paired two-by-two
/// into ``Name: Value`` lines with the empty names and HTTP/2
/// pseudo-headers dropped; the multi-line walk with blank lines
/// filtered out (blank after CPython's own `str.strip()` — a wider
/// set than the crate's `trim`) and Chrome's decoded-protobuf blocks
/// stripped, classified as alternating name/value lines when more
/// than two lines survive and fewer than a fifth contain ``": "`` —
/// the same `len * 0.2` float threshold — paired two-by-two with only
/// the pseudo-headers dropped (an empty name still pairs here, the
/// reference's asymmetry); and the standard ``Name: Value`` pass that
/// keeps every surviving line stripped, minus the leading-colon
/// pseudo-headers. Every strip is `py_strip`, every join a plain
/// ``\n`` join (the empty list answering the empty string).
#[pyfunction]
#[pyo3(signature = (raw))]
fn normalize_raw_headers(raw: &str) -> String {
    let has_caret = raw.contains("^[E");
    let has_esc_e = raw.contains("\u{1b}E");
    if has_caret || has_esc_e || raw.contains('\u{1b}') {
        // "^[E" if "^[E" in raw else ("\x1bE" if … else "\x1b")
        let sep = if has_caret {
            "^[E"
        } else if has_esc_e {
            "\u{1b}E"
        } else {
            "\u{1b}"
        };
        let parts: Vec<&str> = raw.split(sep).collect();
        let mut lines: Vec<String> = Vec::new();
        let mut i = 0;
        while i + 1 < parts.len() {
            let name = py_strip(parts[i]);
            let value = py_strip(parts[i + 1]);
            i += 2;
            if name.is_empty() || PSEUDO_HEADERS.contains(&name) {
                continue;
            }
            lines.push(format!("{name}: {value}"));
        }
        return lines.join("\n");
    }

    let raw_lines = strip_decoded_blocks(
        raw.split('\n')
            .filter(|line| !py_strip(line).is_empty())
            .collect(),
    );
    let colon_lines = raw_lines.iter().filter(|line| line.contains(": ")).count();
    // len(raw_lines) > 2 and colon_lines < len(raw_lines) * 0.2 — the
    // int-times-float product and the int-vs-float comparison both at
    // f64 width, exactly like the reference's.
    let is_alternating = raw_lines.len() > 2 && (colon_lines as f64) < raw_lines.len() as f64 * 0.2;

    if is_alternating {
        let mut lines: Vec<String> = Vec::new();
        let mut i = 0;
        while i + 1 < raw_lines.len() {
            let name = py_strip(raw_lines[i]);
            let value = py_strip(raw_lines[i + 1]);
            i += 2;
            if PSEUDO_HEADERS.contains(&name) {
                continue;
            }
            lines.push(format!("{name}: {value}"));
        }
        return lines.join("\n");
    }

    let mut result: Vec<&str> = Vec::new();
    for line in &raw_lines {
        let stripped = py_strip(line);
        if stripped.starts_with(':') {
            continue;
        }
        result.push(stripped);
    }
    result.join("\n")
}

// ── Stream cookiejar builder (services/auth.py) ──────────────────────────

/// yt-dlp's own Netscape jar header — `YoutubeDLCookieJar._HEADER`. The
/// reference saves through yt-dlp's subclass, not stdlib's
/// `MozillaCookieJar`, so this two-line header (not stdlib's four-line
/// `NETSCAPE_HEADER_TEXT`) is what the file starts with.
const YTDLP_JAR_HEADER: &str =
    "# Netscape HTTP Cookie File\n# This file is generated by yt-dlp.  Do not edit.\n\n";

/// `CookieJar.__iter__` — stdlib `deepvalues` over the nested
/// `{domain: {path: {name: cookie}}}` store: depth-first, each level's
/// entries in the running interpreter's own order — plain insertion
/// order on 3.11+, and the `sorted(mapping.keys())` walk while 3.10
/// still sorted (gh-86232 changed `deepvalues` mid-3.11 development),
/// so the jar bytes always match what the fallback of the interpreter
/// that loaded the engine would have written. The walk recurses into
/// the store's own dict levels only — the cookie objects it reaches
/// never carry `.items` the way the reference's `obj.items` probe
/// requires of a nested mapping, so both walks agree on every store
/// this engine can build.
fn jar_cookies<'py>(
    py: Python<'py>,
    store: &Bound<'py, PyDict>,
    sorted_fn: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut out = Vec::new();
    jar_walk_level(py, store, py.version_info() < (3, 11), sorted_fn, &mut out)?;
    Ok(out)
}

fn jar_walk_level<'py>(
    py: Python<'py>,
    level: &Bound<'py, PyDict>,
    sorted_jar: bool,
    sorted_fn: &Bound<'py, PyAny>,
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if sorted_jar {
        // values = map(adict.get, sorted(adict.keys())) — the keys
        // ordered through CPython's own sorted (its TypeError for an
        // unorderable pair kept), each read through the dict's own
        // subscript; a key vanishing mid-walk reads as None, which
        // deepvalues then yields as a value — reproduced here.
        let keys = level.getattr("keys")?.call0()?;
        let sorted = sorted_fn.call1((keys,))?;
        for key in sorted.try_iter()? {
            let key = key?;
            match level.get_item(&key)? {
                Some(value) => jar_walk_value(py, value, sorted_jar, sorted_fn, out)?,
                None => out.push(py.None().into_bound(py)),
            }
        }
    } else {
        // list(mapping.values()) — each level in plain insertion order.
        for (_, value) in level.iter() {
            jar_walk_value(py, value, sorted_jar, sorted_fn, out)?;
        }
    }
    Ok(())
}

fn jar_walk_value<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
    sorted_jar: bool,
    sorted_fn: &Bound<'py, PyAny>,
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if let Ok(sub) = value.downcast::<PyDict>() {
        jar_walk_level(py, sub, sorted_jar, sorted_fn, out)?;
    } else {
        out.push(value);
    }
    Ok(())
}

/// The wide stream cookiejar builder — `services/auth.py`'s
/// `_build_stream_cookiejar`, behind `_commit_session`'s first write.
///
/// Mirrors the reference read for read. The filter loop walks *jar*
/// through Python's iteration protocol, each cookie's domain through
/// its own `lstrip(".")`, the exact-family membership
/// (`bare in ("youtube.com", "google.com")` — CPython's own `in`, so a
/// confusable like `notyoutube.com` fails the equality before the
/// suffix arm admits it) and the suffix arm through the object's own
/// `endswith` tuple call; a kept cookie's value read through the
/// reference's `or ""` truthiness gate and its name and value walked
/// through the `ch in value` containment of the control-char gate
/// (`"\t"`/`"\n"`/`"\r"`, name first — an embedded control character
/// would corrupt the line-oriented file). Survivors enter the jar's
/// own nested `{domain: {path: {name: cookie}}}` store — plain dicts,
/// so the reference `set_cookie`'s CPython semantics hold: hash/eq
/// membership, the first position of a repeated key with its last
/// object.
///
/// The save is yt-dlp's `YoutubeDLCookieJar.save` (the subclass the
/// reference instantiates), `ignore_discard`/`ignore_expires` both on
/// so neither gate ever bites: first the session-cookie pass that
/// assigns `expires = 0` to every kept cookie still carrying `None`
/// (the caller's own Cookie objects, mutated exactly like the
/// reference's shared references), then the line pass — the domain
/// through its own `startswith(".")`, the path, the secure flag's
/// truthiness, the expires through the interpreter's own `str()`
/// (never `None` after the first pass, so a session cookie renders
/// `"0"`), the name/value pair swapped when the value is `None`
/// (`Set-Cookie: foo`'s Netscape spelling), all seven columns joined
/// by `"\t".join` — CPython's own join, its TypeError for a non-str
/// column included — and the line written through `'{}\n'.format`.
/// The jar's cookie order is the interpreter's own `deepvalues` walk
/// (see `jar_cookies`), the buffer assembled by `"".join` and encoded
/// through the text object's own `encode("utf-8")` — its
/// `UnicodeEncodeError` shape kept. The `yt_dlp.cookies` import and
/// the StringIO buffer stay Python-side in the fallback only.
#[pyfunction]
#[pyo3(signature = (jar))]
fn build_stream_cookiejar<'py>(py: Python<'py>, jar: &Bound<'py, PyAny>) -> PyResult<Py<PyBytes>> {
    let dot = PyString::new(py, ".");
    let empty_str = PyString::new(py, "");
    let exact_domains = PyTuple::new(py, ["youtube.com", "google.com"])?.into_any();
    let suffix_domains = PyTuple::new(py, [".youtube.com", ".google.com"])?;
    let builtins = py.import("builtins")?;
    let str_fn = builtins.getattr("str")?;
    let sorted_fn = builtins.getattr("sorted")?;

    // stream_jar = YoutubeDLCookieJar() — the nested store starts
    // empty, exactly like the fresh jar the reference builds.
    let store = PyDict::new(py);

    for cookie in jar.try_iter()? {
        let cookie = cookie?;
        // bare = cookie.domain.lstrip(".")
        let domain = cookie.getattr("domain")?;
        let bare = domain.getattr("lstrip")?.call1((dot.clone(),))?;
        // if bare in ("youtube.com", "google.com") or bare.endswith(…)
        let keep = exact_domains.contains(&bare)?
            || bare
                .getattr("endswith")?
                .call1((suffix_domains.clone(),))?
                .is_truthy()?;
        if !keep {
            continue;
        }
        // value = cookie.value or ""
        let value = cookie.getattr("value")?;
        let checked = if value.is_truthy()? {
            value
        } else {
            empty_str.clone().into_any()
        };
        // if _has_control_chars(cookie.name, value): continue — name
        // first, then the gated value, each against \t/\n/\r through
        // CPython's own `in`.
        let name = cookie.getattr("name")?;
        let mut control = false;
        for v in [&name, &checked] {
            for ch in ["\t", "\n", "\r"] {
                if v.contains(PyString::new(py, ch))? {
                    control = true;
                    break;
                }
            }
            if control {
                break;
            }
        }
        if control {
            continue;
        }
        // set_cookie(cookie) — the reference's own store walk, each
        // level created on miss and the (domain, path) attrs read
        // again exactly where set_cookie reads them.
        let cookie_domain = cookie.getattr("domain")?;
        if !store.contains(&cookie_domain)? {
            store.set_item(&cookie_domain, PyDict::new(py))?;
        }
        let paths = store
            .get_item(&cookie_domain)?
            .expect("the domain was just inserted or already present");
        let paths = paths
            .downcast::<PyDict>()
            .expect("set_cookie only ever stores dicts");
        let cookie_path = cookie.getattr("path")?;
        if !paths.contains(&cookie_path)? {
            paths.set_item(&cookie_path, PyDict::new(py))?;
        }
        let names = paths
            .get_item(&cookie_path)?
            .expect("the path was just inserted or already present");
        let names = names
            .downcast::<PyDict>()
            .expect("set_cookie only ever stores dicts");
        let cookie_name = cookie.getattr("name")?;
        names.set_item(&cookie_name, cookie)?;
    }

    // The session-cookie pass — every kept cookie still carrying
    // expires None is assigned the int 0, mutating the caller's own
    // objects exactly like the reference's shared references.
    for cookie in jar_cookies(py, &store, &sorted_fn)? {
        if cookie.getattr("expires")?.is_none() {
            cookie.setattr("expires", 0i64)?;
        }
    }

    // The line pass — yt-dlp's _HEADER, then one seven-column line per
    // cookie in the jar's own order.
    let tab = PyString::new(py, "\t");
    let line_format = PyString::new(py, "{}\n");
    let mut parts: Vec<Bound<'py, PyAny>> = Vec::new();
    parts.push(PyString::new(py, YTDLP_JAR_HEADER).into_any());
    for cookie in jar_cookies(py, &store, &sorted_fn)? {
        let domain = cookie.getattr("domain")?;
        let domain_dot = if domain
            .getattr("startswith")?
            .call1((dot.clone(),))?
            .is_truthy()?
        {
            "TRUE"
        } else {
            "FALSE"
        };
        let path = cookie.getattr("path")?;
        let secure = if cookie.getattr("secure")?.is_truthy()? {
            "TRUE"
        } else {
            "FALSE"
        };
        // str_or_none(cookie.expires, default="") — never None after
        // the session-cookie pass, so the interpreter's own str().
        let expires = cookie.getattr("expires")?;
        let expires_col = if expires.is_none() {
            empty_str.clone().into_any()
        } else {
            str_fn.call1((expires,))?
        };
        // name, value = cookie.name, cookie.value — the value read
        // after the name, a None value swapping the pair.
        let name = cookie.getattr("name")?;
        let value = cookie.getattr("value")?;
        let (name_col, value_col) = if value.is_none() {
            (empty_str.clone().into_any(), name)
        } else {
            (name, value)
        };
        let cols = PyTuple::new(
            py,
            [
                domain,
                PyString::new(py, domain_dot).into_any(),
                path,
                PyString::new(py, secure).into_any(),
                expires_col,
                name_col,
                value_col,
            ],
        )?;
        let joined = tab.getattr("join")?.call1((cols,))?;
        parts.push(line_format.getattr("format")?.call1((joined,))?);
    }
    // The StringIO's accumulated text, assembled and encoded through
    // the str object's own methods.
    let text = empty_str
        .getattr("join")?
        .call1((PyList::new(py, parts)?,))?;
    let encoded = text
        .getattr("encode")?
        .call1((PyString::new(py, "utf-8"),))?;
    let bytes = encoded
        .downcast_into::<PyBytes>()
        .expect("str.encode returns bytes");
    Ok(bytes.unbind())
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
fn filter_regions<'py>(regions: &Bound<'py, PyAny>, query: &str) -> PyResult<Bound<'py, PyAny>> {
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

// ── Terminal orientation gate (utils/terminal.py) ────────────────────────

/// The terminal orientation verdict — `utils/terminal.py`'s
/// `get_orientation`, the aspect-ratio gate behind the app's
/// horizontal/vertical layout decision.
///
/// Mirrors the reference's one expression exactly, every operator the
/// interpreter's own: `rows == 0` through CPython's `==` (its
/// `__eq__` protocol, a non-int rows simply not equal), `cols / rows`
/// through CPython's true division (arbitrary-precision ints divided
/// correctly-rounded — a float on every path — its TypeError for a
/// non-numeric pair kept), and the `> 2.3` comparison through
/// CPython's `>` against the very double the reference's literal
/// compiles to. `"horizontal"` when rows is zero or the ratio exceeds
/// the threshold, `"vertical"` otherwise.
#[pyfunction]
#[pyo3(signature = (cols, rows))]
fn get_orientation<'py>(
    py: Python<'py>,
    cols: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<&'static str> {
    let zero = 0i64.into_py_any(py)?;
    if rows.eq(&zero)? {
        return Ok("horizontal");
    }
    let ratio = cols.div(rows)?;
    let threshold = PyFloat::new(py, 2.3).into_any();
    if ratio.gt(&threshold)? {
        Ok("horizontal")
    } else {
        Ok("vertical")
    }
}

// ── History SQL bodies (services/history.py) ──────────────────────────────

/// The schema DDL script behind `HistoryManager.init`'s
/// `executescript` — the pure half of the database's first write.
///
/// Mirrors the pure-Python reference (`_history_schema`, returning the
/// module's `_SCHEMA` constant): the search-history table with its
/// unique query index, the play-history table and the play-stats
/// table, leading newline and zero-indent lines included — the exact
/// text the fallback returns, so a schema executes identically on
/// either backend.
#[pyfunction]
#[pyo3(signature = ())]
fn history_schema() -> String {
    "
CREATE TABLE IF NOT EXISTS search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    filter_mode TEXT DEFAULT 'music',
    result_count INTEGER,
    search_count INTEGER DEFAULT 1,
    first_searched TEXT DEFAULT (datetime('now')),
    last_searched TEXT DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_search_query
    ON search_history(query, filter_mode);

CREATE TABLE IF NOT EXISTS play_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL,
    title TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    duration_seconds INTEGER,
    listened_seconds INTEGER,
    source TEXT,
    played_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS play_stats (
    video_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT,
    play_count INTEGER DEFAULT 0,
    total_listened_seconds INTEGER DEFAULT 0,
    last_played TEXT
);
"
    .to_string()
}

/// The retention prune behind `HistoryManager.init`'s post-schema
/// write — the statement and the bound retention limit.
///
/// Mirrors the pure-Python reference (`_history_retention_delete`):
/// the statement text exactly as the fallback spells it (the
/// reference's two adjacent literals joined into the one line they
/// concatenate to), and the `max_history` value passed through as the
/// object it is into the parameter tuple — `(sql, (max_history,))`,
/// the shape `aiosqlite`'s `execute(*stmt)` unpacks.
#[pyfunction]
#[pyo3(signature = (max_history))]
fn history_retention_delete<'py>(
    py: Python<'py>,
    max_history: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "DELETE FROM play_history WHERE rowid NOT IN (SELECT rowid FROM play_history ORDER BY played_at DESC LIMIT ?)"
        .to_string();
    let params = PyTuple::new(py, [max_history])?;
    Ok((sql, params))
}

/// The search-history upsert behind `HistoryManager.log_search` — the
/// statement and the three bound values.
///
/// Mirrors the pure-Python reference (`_search_upsert`): the statement
/// text exactly as the fallback spells it (leading newline and
/// eight-space indentation included — the reference's method-body
/// spelling re-indented to the engine's module-level home, a spelling
/// SQLite parses identically), and `query` / `filter_mode` /
/// `result_count` passed through as the objects they are into the
/// parameter tuple.
#[pyfunction]
#[pyo3(signature = (query, filter_mode, result_count))]
fn search_upsert<'py>(
    py: Python<'py>,
    query: &Bound<'py, PyAny>,
    filter_mode: &Bound<'py, PyAny>,
    result_count: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        INSERT INTO search_history (query, filter_mode, result_count)
        VALUES (?, ?, ?)
        ON CONFLICT(query, filter_mode) DO UPDATE SET
            search_count = search_count + 1,
            result_count = excluded.result_count,
            last_searched = datetime('now')
        "
    .to_string();
    let params = PyTuple::new(py, [query, filter_mode, result_count])?;
    Ok((sql, params))
}

/// The recent-searches select behind `HistoryManager.get_search_history`.
///
/// Mirrors the pure-Python reference (`_search_history_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is into the parameter tuple.
#[pyfunction]
#[pyo3(signature = (limit))]
fn search_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT query, filter_mode, result_count, search_count,
               first_searched, last_searched
        FROM search_history
        ORDER BY last_searched DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The autocomplete select behind
/// `HistoryManager.get_search_suggestions`.
///
/// Mirrors the pure-Python reference (`_search_suggestion_select`):
/// the statement text exactly as the fallback spells it, and `prefix`
/// / `limit` passed through as the objects they are into the
/// parameter tuple.
#[pyfunction]
#[pyo3(signature = (prefix, limit))]
fn search_suggestion_select<'py>(
    py: Python<'py>,
    prefix: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT query
        FROM search_history
        WHERE query LIKE ? || '%'
        ORDER BY search_count DESC, last_searched DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [prefix, limit])?;
    Ok((sql, params))
}

/// The clear-all statement behind
/// `HistoryManager.clear_search_history`.
///
/// Mirrors the pure-Python reference (`_search_history_delete`): the
/// bare statement text, no parameter tuple — the shape the reference
/// hands `execute` as its single argument.
#[pyfunction]
#[pyo3(signature = ())]
fn search_history_delete() -> String {
    "DELETE FROM search_history".to_string()
}

/// The play-history INSERT statement text behind
/// `HistoryManager.log_play`'s first write.
///
/// Mirrors the pure-Python reference (`_play_history_insert_sql`):
/// the statement text exactly as the fallback spells it. The seven
/// bound values are the `play_insert_params` engine's tuple (ported
/// earlier), so this engine returns the statement alone — the
/// reference's `execute(sql, params)` two-argument shape stays the
/// caller's to build.
#[pyfunction]
#[pyo3(signature = ())]
fn play_history_insert_sql() -> String {
    "
        INSERT INTO play_history
            (video_id, title, artist, album, duration_seconds,
             listened_seconds, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        "
    .to_string()
}

/// The play-stats upsert statement text behind
/// `HistoryManager.log_play`'s second write.
///
/// Mirrors the pure-Python reference (`_play_stats_upsert_sql`): the
/// statement text exactly as the fallback spells it. The four bound
/// values are the `play_stats_params` engine's tuple (ported
/// earlier), so this engine returns the statement alone.
#[pyfunction]
#[pyo3(signature = ())]
fn play_stats_upsert_sql() -> String {
    "
        INSERT INTO play_stats (video_id, title, artist, play_count,
                                total_listened_seconds, last_played)
        VALUES (?, ?, ?, 1, ?, datetime('now'))
        ON CONFLICT(video_id) DO UPDATE SET
            title = excluded.title,
            artist = excluded.artist,
            play_count = play_count + 1,
            total_listened_seconds = total_listened_seconds + excluded.total_listened_seconds,
            last_played = datetime('now')
        "
    .to_string()
}

/// The stored-listening read behind
/// `HistoryManager.update_play_listened_seconds` — the statement and
/// the bound row id.
///
/// Mirrors the pure-Python reference (`_play_listened_select`): the
/// statement text exactly as the fallback spells it, and `play_id`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (play_id))]
fn play_listened_select<'py>(
    py: Python<'py>,
    play_id: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "SELECT video_id, listened_seconds FROM play_history WHERE id = ?".to_string();
    let params = PyTuple::new(py, [play_id])?;
    Ok((sql, params))
}

/// The finalised-listen update behind
/// `HistoryManager.update_play_listened_seconds` — the statement and
/// the two bound values.
///
/// Mirrors the pure-Python reference (`_play_listened_update`): the
/// statement text exactly as the fallback spells it, and
/// `listened_seconds` / `play_id` passed through as the objects they
/// are, in the statement's placeholder order.
#[pyfunction]
#[pyo3(signature = (listened_seconds, play_id))]
fn play_listened_update<'py>(
    py: Python<'py>,
    listened_seconds: &Bound<'py, PyAny>,
    play_id: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "UPDATE play_history SET listened_seconds = ? WHERE id = ?".to_string();
    let params = PyTuple::new(py, [listened_seconds, play_id])?;
    Ok((sql, params))
}

/// The stats-side listen accrual behind
/// `HistoryManager.update_play_listened_seconds` — the statement and
/// the two bound values.
///
/// Mirrors the pure-Python reference
/// (`_play_stats_listened_update`): the statement text exactly as the
/// fallback spells it, and `delta` / `video_id` passed through as the
/// objects they are, in the statement's placeholder order.
#[pyfunction]
#[pyo3(signature = (delta, video_id))]
fn play_stats_listened_update<'py>(
    py: Python<'py>,
    delta: &Bound<'py, PyAny>,
    video_id: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        UPDATE play_stats
        SET total_listened_seconds = total_listened_seconds + ?,
            last_played = datetime('now')
        WHERE video_id = ?
        "
    .to_string();
    let params = PyTuple::new(py, [delta, video_id])?;
    Ok((sql, params))
}

/// The play-history select behind `HistoryManager.get_play_history`.
///
/// Mirrors the pure-Python reference (`_play_history_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
fn play_history_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT video_id, title, artist, album, duration_seconds,
               listened_seconds, source, played_at
        FROM play_history
        ORDER BY played_at DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The deduplicated recently-played select behind
/// `HistoryManager.get_recently_played` — the one min/max-aggregate
/// statement whose bare columns read from the max row.
///
/// Mirrors the pure-Python reference (`_recently_played_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
fn recently_played_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT video_id, title, artist, album, duration_seconds,
               MAX(played_at) AS played_at
        FROM play_history
        GROUP BY video_id
        ORDER BY played_at DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The played-video-id read behind
/// `HistoryManager.get_played_video_ids` — the Recently-Played
/// filter's DISTINCT walk.
///
/// Mirrors the pure-Python reference (`_played_video_ids_select`):
/// the bare statement text, no parameter tuple — the shape the
/// reference hands `execute` as its single argument.
#[pyfunction]
#[pyo3(signature = ())]
fn played_video_ids_select() -> String {
    "SELECT DISTINCT video_id FROM play_history".to_string()
}

/// The total-plays aggregate behind `HistoryManager.get_stats`.
///
/// Mirrors the pure-Python reference (`_stats_total_plays_select`):
/// the bare statement text, no parameter tuple.
#[pyfunction]
#[pyo3(signature = ())]
fn stats_total_plays_select() -> String {
    "SELECT COUNT(*) AS total_plays FROM play_history".to_string()
}

/// The total-listen-time aggregate behind
/// `HistoryManager.get_stats`.
///
/// Mirrors the pure-Python reference
/// (`_stats_total_listen_time_select`): the bare statement text, no
/// parameter tuple.
#[pyfunction]
#[pyo3(signature = ())]
fn stats_total_listen_time_select() -> String {
    "SELECT COALESCE(SUM(listened_seconds), 0) AS s FROM play_history".to_string()
}

/// The unique-tracks aggregate behind `HistoryManager.get_stats`.
///
/// Mirrors the pure-Python reference (`_stats_unique_tracks_select`):
/// the bare statement text, no parameter tuple.
#[pyfunction]
#[pyo3(signature = ())]
fn stats_unique_tracks_select() -> String {
    "SELECT COUNT(*) AS c FROM play_stats".to_string()
}

/// The most-played-tracks select behind
/// `HistoryManager.get_top_tracks`.
///
/// Mirrors the pure-Python reference (`_top_tracks_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
fn top_tracks_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT video_id, title, artist, play_count,
               total_listened_seconds, last_played
        FROM play_stats
        ORDER BY play_count DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

/// The most-played-artists select behind
/// `HistoryManager.get_top_artists` — the artist-grouped ranking with
/// its NULL/blank filter.
///
/// Mirrors the pure-Python reference (`_top_artists_select`): the
/// statement text exactly as the fallback spells it, and `limit`
/// passed through as the object it is.
#[pyfunction]
#[pyo3(signature = (limit))]
fn top_artists_select<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyTuple>)> {
    let sql = "
        SELECT artist,
               SUM(play_count) AS play_count,
               SUM(total_listened_seconds) AS total_listened_seconds
        FROM play_stats
        WHERE artist IS NOT NULL AND artist != ''
        GROUP BY artist
        ORDER BY play_count DESC
        LIMIT ?
        "
    .to_string();
    let params = PyTuple::new(py, [limit])?;
    Ok((sql, params))
}

// ── Stream resolver glue (services/stream.py) ─────────────────────────────

/// The stream module's own clock — `time.time()` read through the
/// module's `time` attribute, exactly like the reference's global
/// lookup at call time. A test (or any caller) that swaps
/// `ytm_player.services.stream`'s `time` for a frozen namespace steers
/// the native engine exactly like the fallback: the clock the
/// reference would read is the clock the engine reads.
fn stream_module_clock(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let stream = PyModule::import(py, "ytm_player.services.stream")?;
    let time_module = stream.getattr("time")?;
    let clock = time_module.getattr("time")?;
    clock.call0()
}

/// The stream module's `QUALITY_FORMATS` table, read through the
/// module attribute — the very dict the fallback's own lookups run
/// against, a patched table included.
fn stream_quality_formats(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.stream")?.getattr("QUALITY_FORMATS")
}

/// The stream module's `_CACHE_MAX_SIZE` cap, read through the module
/// attribute — the very constant the fallback's own over-cap gate and
/// excess arithmetic run against, a patched cap included.
fn stream_cache_max_size(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.stream")?.getattr("_CACHE_MAX_SIZE")
}

/// `float(info.get(key) or 0)` — the truthiness gate feeding the
/// interpreter's own `float()` (see `builtin_float`), so the string
/// grammar, `__float__` dispatch and error shapes all come free.
fn truthy_or_float_zero<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let raw = field_get(py, info, key, None)?;
    let inner = if raw.is_truthy()? {
        raw
    } else {
        0i64.into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any()
    };
    builtin_float(py, &inner)
}

/// The cached-URL staleness gate — `services/stream.py`'s `_is_stale`,
/// applied by every read path (`_get_cached`, `is_expired`).
///
/// Mirrors the reference's one expression exactly: the clock read
/// through the module's own `time` name first (the left operand), the
/// `_EXPIRY_BUFFER_SECONDS` (300) subtraction through CPython's `-`
/// (its `__sub__` protocol — a non-numeric *expires_at* raises the
/// operator's own `TypeError`), and the `>=` through CPython's own
/// comparison, the verdict the comparison's truthiness. A cached URL
/// is stale once the clock has passed, or is within the buffer of,
/// its expiry — so playback is never handed a URL that dies
/// mid-stream.
#[pyfunction]
#[pyo3(signature = (expires_at))]
fn is_stale<'py>(py: Python<'py>, expires_at: &Bound<'py, PyAny>) -> PyResult<bool> {
    let now = stream_module_clock(py)?;
    let buffer = 300i64.into_py_any(py)?;
    let deadline = expires_at.sub(&buffer)?;
    now.ge(&deadline)
}

/// The available_at wait gate — the pure half of `StreamInfo.wait_seconds`
/// (`services/stream.py`), the pre-roll-ad wait `resolve()` sleeps through.
///
/// Mirrors the reference's `max(0.0, available_at - time.time())`:
/// the subtraction (through CPython's `-`, its `__sub__` protocol) is
/// evaluated first — the clock read through the module's own `time`
/// name on its right — and `max` walks its two operands exactly like
/// the reference's call: the `0.0` literal starts as the running
/// maximum and the subtraction replaces it only when it compares
/// strictly greater (CPython's `>`), so a NaN or negative wait answers
/// the `0.0` arm and an equal wait keeps the literal's value. The
/// answer is whichever operand won — the subtraction's own object
/// when it won, a fresh `0.0` float otherwise.
#[pyfunction]
#[pyo3(signature = (available_at))]
fn wait_seconds<'py>(
    py: Python<'py>,
    available_at: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let now = stream_module_clock(py)?;
    let elapsed = available_at.sub(&now)?;
    let zero = PyFloat::new(py, 0.0).into_any();
    if elapsed.gt(&zero)? {
        Ok(elapsed)
    } else {
        Ok(zero)
    }
}

/// The quality-input gate — `services/stream.py`'s
/// `_validate_quality`, the validation the resolver's `quality`
/// setter raises through.
///
/// Mirrors the reference: the membership test through CPython's own
/// `in` (the containment slot — hash/eq membership over the module's
/// own `QUALITY_FORMATS` table, read through the module attribute so a
/// patched table steers both backends identically), and the rejection
/// a `ValueError` carrying the reference's f-string message — the
/// value interpolated through `str()` and the tier list through the
/// interpreter's own `list()` then `str()` (the setter's value is
/// typed `str`, so the interpolation spells the same text either way).
#[pyfunction]
#[pyo3(signature = (value))]
fn validate_quality(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let table = stream_quality_formats(py)?;
    if table.contains(value)? {
        return Ok(());
    }
    // f"Unknown quality '{value}'. Choose from: {list(QUALITY_FORMATS)}"
    let list_type = py.import("builtins")?.getattr("list")?;
    let keys = list_type.call1((table,))?;
    let mut message = String::from("Unknown quality '");
    message.push_str(&py_str(value)?);
    message.push_str("'. Choose from: ");
    message.push_str(&py_str(&keys)?);
    Err(PyValueError::new_err(message))
}

/// The base yt-dlp options dict — `services/stream.py`'s
/// `_base_ydl_opts`, the pure half of `StreamResolver._build_ydl_opts`.
///
/// Mirrors the reference's dict literal key for key, insertion order
/// included: the `format` selector through the module's own
/// `QUALITY_FORMATS` table — `.get(quality, QUALITY_FORMATS["high"])`,
/// the default read eagerly before the lookup runs, the lookup the
/// dict's own `.get` method (hash/eq membership, a patched table
/// included) — the `quiet`/`no_warnings`/`noplaylist`/`skip_download`
/// `True` flags, the `logger` object passed through as the very
/// object the caller built, the `extract_flat`/`writeinfojson`/
/// `writethumbnail` `False` flags, and the nested `extractor_args`
/// mapping (`{"youtube": {"player_client": ["default", "android"]}}`,
/// fresh containers every call). The configured extras — `cookiefile`,
/// the CA bundle's `compat_opts`, `remote_components`, `js_runtimes`
/// — stay in the caller's `apply_configured_yt_dlp_options` pass.
#[pyfunction]
#[pyo3(signature = (quality, logger))]
fn stream_base_opts<'py>(
    py: Python<'py>,
    quality: &Bound<'py, PyAny>,
    logger: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let table = stream_quality_formats(py)?;
    // QUALITY_FORMATS["high"] — evaluated before .get runs, exactly
    // like the reference's argument evaluation.
    let default = table.get_item("high")?;
    // QUALITY_FORMATS.get(quality, default) — the dict's own method.
    let format = table.call_method1("get", (quality, default))?;

    let opts = PyDict::new(py);
    opts.set_item("format", format)?;
    opts.set_item("quiet", true)?;
    opts.set_item("no_warnings", true)?;
    opts.set_item("logger", logger)?;
    opts.set_item("extract_flat", false)?;
    opts.set_item("noplaylist", true)?;
    opts.set_item("skip_download", true)?;
    opts.set_item("writeinfojson", false)?;
    opts.set_item("writethumbnail", false)?;
    let player_client = PyList::new(py, ["default", "android"])?;
    let youtube = PyDict::new(py);
    youtube.set_item("player_client", player_client)?;
    let extractor_args = PyDict::new(py);
    extractor_args.set_item("youtube", youtube)?;
    opts.set_item("extractor_args", extractor_args)?;
    Ok(opts)
}

/// The extract_info → StreamInfo field assembly —
/// `services/stream.py`'s `_stream_info_fields`, the pure half of
/// `StreamResolver._try_resolve`.
///
/// Mirrors the reference's post-extraction walk read for read: the
/// direct `url` read (`info.get("url", "")`), the `available_at`
/// read through the `float(... or 0)` truthiness gate (the
/// interpreter's own `float()` — string grammar, `__float__`
/// dispatch and error shapes included), the `requested_formats`
/// fallback when the direct URL is falsy (the list read through its
/// own `or []` gate, each format probed with the reference's
/// `vcodec == "none" or acodec != "none"` condition — both
/// comparisons through CPython's own operators — the first match's
/// `url` and `available_at` replacing the direct reads, the walk
/// stopping at the `break`), and the `None` verdict when no stream
/// URL survived. The metadata reads follow the reference's own
/// gates: `acodec` through `.get("acodec", "unknown")`, `audio_ext`
/// through the `audio_ext`-or-`ext` chain (a falsy-but-present `ext`
/// passes through as the object it is, exactly like the reference's
/// `or` arm), `abr` through the `abr`-or-`tbr` chain with the
/// interpreter's own `int()` over the winner (float truncation,
/// decimal strings, `__int__`/`__index__` dispatch — a `0` int when
/// both miss), `duration` the same over its own single-key chain,
/// and `thumbnail` the bare `.get` read. The format name is the
/// acodec unless it compares equal to `"none"` (CPython's `!=`), in
/// which case the audio extension; the expiry is the module clock
/// plus the cache TTL (the interpreter's own `+`). The answer is the
/// eight `StreamInfo` field values in the dataclass's declared
/// order — `url`, `video_id`, `format`, `bitrate`, `duration`,
/// `expires_at`, `thumbnail_url`, `available_at` — with every value
/// passed through as the object it is; the `StreamInfo` construction
/// stays Python-side in the caller. Where the reference would raise —
/// a `.get`-less info or format object — the field reads stay total
/// (the `format_search_result` precedent); the info dict is yt-dlp's
/// own dict on every production path.
#[pyfunction]
#[pyo3(signature = (info, video_id))]
fn stream_info_fields<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
    video_id: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // stream_url = info.get("url", "")
    let empty = PyString::new(py, "").into_any();
    let mut stream_url = field_get(py, info, "url", Some(empty))?;
    // available_at = float(info.get("available_at") or 0)
    let mut available_at = truthy_or_float_zero(py, info, "available_at")?;

    if !stream_url.is_truthy()? {
        // formats = info.get("requested_formats") or []
        let formats = match first_truthy_get(py, info, &["requested_formats"])? {
            Some(value) => value,
            None => PyList::empty(py).into_any(),
        };
        for fmt in formats.try_iter()? {
            let fmt = fmt?;
            // fmt.get("vcodec") == "none" or fmt.get("acodec") != "none"
            let vcodec = field_get(py, &fmt, "vcodec", None)?;
            let acodec = field_get(py, &fmt, "acodec", None)?;
            let vcodec_none = vcodec.eq("none")?;
            let acodec_not_none = acodec.ne("none")?;
            if vcodec_none || acodec_not_none {
                stream_url = field_get(py, &fmt, "url", Some(PyString::new(py, "").into_any()))?;
                available_at = truthy_or_float_zero(py, &fmt, "available_at")?;
                break;
            }
        }
    }

    if !stream_url.is_truthy()? {
        return Ok(None);
    }

    // acodec = info.get("acodec", "unknown")
    let acodec = field_get(
        py,
        info,
        "acodec",
        Some(PyString::new(py, "unknown").into_any()),
    )?;
    // audio_ext = info.get("audio_ext") or info.get("ext", "unknown")
    let audio_ext = match first_truthy_get(py, info, &["audio_ext"])? {
        Some(value) => value,
        None => field_get(
            py,
            info,
            "ext",
            Some(PyString::new(py, "unknown").into_any()),
        )?,
    };
    // abr = int(info.get("abr") or info.get("tbr") or 0)
    let abr = match first_truthy_get(py, info, &["abr", "tbr"])? {
        Some(value) => builtin_int(py, &value)?,
        None => 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any(),
    };
    // duration = int(info.get("duration") or 0)
    let duration = match first_truthy_get(py, info, &["duration"])? {
        Some(value) => builtin_int(py, &value)?,
        None => 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any(),
    };
    // thumbnail = info.get("thumbnail")
    let thumbnail = field_get(py, info, "thumbnail", None)?;
    // fmt_name = acodec if acodec != "none" else audio_ext
    let fmt_name = if acodec.ne("none")? {
        acodec
    } else {
        audio_ext
    };
    // expires_at = time.time() + _CACHE_TTL_SECONDS
    let now = stream_module_clock(py)?;
    let ttl = 18000i64.into_py_any(py)?;
    let expires_at = now.add(&ttl)?;

    Ok(Some(PyTuple::new(
        py,
        [
            stream_url,
            video_id.clone(),
            fmt_name,
            abr,
            duration,
            expires_at,
            thumbnail,
            available_at,
        ],
    )?))
}

/// The DownloadError hint — `services/stream.py`'s
/// `_download_error_hint`, the diagnostic the resolver's error
/// handler appends when a signed-in resolve failed looking
/// auth-related.
///
/// Mirrors the reference's gate: the hint only when the session
/// cookiejar is in use AND the error message matches the
/// `sign in|login_required|confirm you` pattern case-insensitively
/// (the regex crate's Unicode-aware `(?i)` — simple case folding over
/// the same ASCII keywords `re.I` folds, identical on every practical
/// message), the exact hint text with its leading blank and the
/// `ytm setup` advice, and the empty string otherwise. The
/// `_session_cookiejar_signature()` probe and the `str(exc)`
/// conversion stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (message, session_cookiejar_active))]
fn download_error_hint(message: &str, session_cookiejar_active: bool) -> String {
    if session_cookiejar_active && DOWNLOAD_HINT_RE.is_match(message) {
        " (stream cookiejar may be stale/revoked — try `ytm setup` to refresh it)".to_string()
    } else {
        String::new()
    }
}

/// The auth-related error-text pattern behind the DownloadError hint —
/// `re.search(r"sign in|login_required|confirm you", …, re.I)`'s
/// three literal branches, case-insensitive.
static DOWNLOAD_HINT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)sign in|login_required|confirm you").expect("valid download hint regex")
});

/// The expired-entry walk — `services/stream.py`'s
/// `_cache_expired_ids`, the prune phase behind the resolver cache's
/// maintenance (`_put_cache` and `prune_expired`).
///
/// Mirrors the reference's comprehension exactly: the clock read
/// through the module's own `time` name first (see
/// `stream_module_clock`), then one pass over `cache.items()` — the
/// method call, so a subclass override steers the walk — in the
/// mapping's own order, each entry unpacked as the reference's `for
/// vid, info in …` pair (see `py_unpack_pair`), each `expires_at` read
/// through CPython's attribute protocol and compared with the `>=`
/// through CPython's own rich comparison, the reference's operand
/// order — so every numeric shape and every error matches the
/// reference. The ids answer in the cache's iteration order; the dict
/// state stays Python-side (the caller holds the lock and performs
/// the deletions).
#[pyfunction]
#[pyo3(signature = (cache))]
fn stream_cache_expired_ids<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let now = stream_module_clock(py)?;
    let items = cache.call_method0("items")?;
    let mut expired: Vec<Bound<'py, PyAny>> = Vec::new();
    for entry in items.try_iter()? {
        let entry = entry?;
        let (vid, info) = py_unpack_pair(&entry)?;
        let expires_at = info.getattr("expires_at")?;
        if now.ge(&expires_at)? {
            expired.push(vid);
        }
    }
    Ok(PyList::new(py, expired).expect("list of ids cannot fail"))
}

/// The over-cap eviction walk — `services/stream.py`'s
/// `_cache_cap_evictions`, the second phase of `_put_cache`'s
/// maintenance, run once the expired entries are pruned.
///
/// Mirrors the reference exactly: the cap read through the module's
/// own `_CACHE_MAX_SIZE` attribute (a patched cap steers both
/// backends identically), the `len(cache) > cap` gate through
/// CPython's own comparison — within the cap answers the empty list —
/// then `sorted(cache, key=lambda vid: cache[vid].expires_at)`: the
/// key read through the subscript and attribute protocols once per
/// entry over the cache's own iteration order, the ranking through
/// CPython's own `<` on the key results (exactly the comparisons
/// `list.sort` runs), the stability the reference's `sorted()`
/// guarantees. Then `excess = len(cache) - cap` through CPython's own
/// `-` and the `sorted_ids[:excess]` slice through CPython's own
/// slice protocol — so every numeric shape and every error matches
/// the reference, operand orders included. The ids answer in eviction
/// order; the dict state stays Python-side.
#[pyfunction]
#[pyo3(signature = (cache))]
fn stream_cache_cap_evictions<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let cap = stream_cache_max_size(py)?;
    // len(cache) — through CPython's own len(), as the int object the
    // reference's comparisons and subtraction run against.
    let size = cache
        .len()?
        .into_pyobject(py)
        .expect("len() -> int is infallible")
        .into_any();
    // if len(cache) <= _CACHE_MAX_SIZE: return [] — the inverted
    // over-cap gate.
    if !size.gt(&cap)? {
        return Ok(PyList::empty(py));
    }
    // sorted(cache, key=…) — the decorate pass: cache[vid].expires_at
    // once per entry, in the cache's own iteration order.
    let mut keyed: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for vid in cache.try_iter()? {
        let vid = vid?;
        let expires_at = cache.get_item(&vid)?.getattr("expires_at")?;
        keyed.push((vid, expires_at));
    }
    // list.sort's `<`-only ordering over the key results, stable —
    // the first comparison error returns exactly like the reference's
    // sorted() would raise it.
    let mut sort_err: Option<PyErr> = None;
    keyed.sort_by(|a, b| {
        if sort_err.is_some() {
            // The outcome no longer matters — the error returns anyway.
            return std::cmp::Ordering::Equal;
        }
        match (a.1.lt(&b.1), b.1.lt(&a.1)) {
            (Ok(true), _) => std::cmp::Ordering::Less,
            (Ok(false), Ok(true)) => std::cmp::Ordering::Greater,
            (Ok(_), Ok(_)) => std::cmp::Ordering::Equal,
            (Err(err), _) | (_, Err(err)) => {
                sort_err = Some(err);
                std::cmp::Ordering::Equal
            }
        }
    });
    if let Some(err) = sort_err {
        return Err(err);
    }
    // sorted_ids — the fresh list sorted() answers.
    let ids: Vec<Bound<'py, PyAny>> = keyed.into_iter().map(|(vid, _)| vid).collect();
    let sorted_ids = PyList::new(py, ids).expect("list of ids cannot fail");
    // excess = len(cache) - cap, then sorted_ids[:excess] — both
    // through CPython's own protocols, a non-int excess raising the
    // slice's own TypeError exactly like the reference.
    let excess = size.sub(&cap)?;
    let slice_type = PyModule::import(py, "builtins")?.getattr("slice")?;
    let window = slice_type.call1((py.None(), excess, py.None()))?;
    // The slice through the getitem protocol — sorted_ids[window], the
    // reference's sorted_ids[:excess] spelled as the slice object it
    // compiles to.
    let evictions = sorted_ids.as_any().get_item(&window)?;
    Ok(evictions
        .downcast_into::<PyList>()
        .expect("a slice of a list is a list"))
}

// ── Download service (services/download.py) ──────────────────────────────

/// The base yt-dlp options dict behind `DownloadService._build_opts` —
/// the pure half of a download's yt-dlp configuration.
///
/// Mirrors the pure-Python reference (`_base_opts`): the nine-key dict
/// in the reference's insertion order — the audio format selector
/// (`bestaudio[ext=webm]/bestaudio/best`, the module's
/// `_DOWNLOAD_FORMAT`), the staged `outtmpl` passed through as the very
/// object the caller assembled, the quiet/no-warning/noplaylist
/// `True` flags, the `extract_flat`/`writethumbnail`/`writeinfojson`
/// `False` flags, and the one-entry `postprocessors` list (the
/// `FFmpegExtractAudio` postprocessor with its opus codec and 128
/// quality, fresh containers every call). The configured extras stay
/// in the caller's `apply_configured_yt_dlp_options` pass; the
/// staging, the yt-dlp invocation and the atomic publication stay
/// Python-side in the service.
#[pyfunction]
#[pyo3(signature = (output_path))]
fn download_base_opts<'py>(py: Python<'py>, output_path: &Bound<'py, PyAny>) -> Bound<'py, PyDict> {
    let opts = PyDict::new(py);
    // Only str keys are inserted and only str/bool/object values —
    // always storable, so the insertions cannot fail.
    opts.set_item("format", "bestaudio[ext=webm]/bestaudio/best")
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("outtmpl", output_path)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("quiet", true)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("no_warnings", true)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("noplaylist", true)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("extract_flat", false)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("writethumbnail", false)
        .expect("dict insertion with a str key cannot fail");
    opts.set_item("writeinfojson", false)
        .expect("dict insertion with a str key cannot fail");
    let postprocessor = PyDict::new(py);
    postprocessor
        .set_item("key", "FFmpegExtractAudio")
        .expect("dict insertion with a str key cannot fail");
    postprocessor
        .set_item("preferredcodec", "opus")
        .expect("dict insertion with a str key cannot fail");
    postprocessor
        .set_item("preferredquality", "128")
        .expect("dict insertion with a str key cannot fail");
    let postprocessors = PyList::new(py, [postprocessor]).expect("list of one dict cannot fail");
    opts.set_item("postprocessors", postprocessors)
        .expect("dict insertion with a str key cannot fail");
    opts
}

/// The five probe filenames a completed download is looked up under —
/// the extension order behind `DownloadService._download_sync`'s
/// publication walk and `get_path`'s lookup.
///
/// Mirrors the pure-Python reference (`_candidate_names`): the
/// `<video_id>.<ext>` names built over the reference's own probe order
/// — `opus` first (the postprocessor's target), then `webm`, `m4a`,
/// `mp3` and `ogg` — answering as a fresh tuple of fresh strings. The
/// video id is typed `str` end to end in the service; a non-str id
/// raises the argument extraction's own `TypeError` where the
/// reference's f-string would have stringified it.
#[pyfunction]
#[pyo3(signature = (video_id))]
fn download_candidates<'py>(py: Python<'py>, video_id: &str) -> Bound<'py, PyTuple> {
    let names = ["opus", "webm", "m4a", "mp3", "ogg"].map(|ext| format!("{video_id}.{ext}"));
    PyTuple::new(py, names).expect("tuple of five strings cannot fail")
}

// ---------------------------------------------------------------------------
// Player glue engines (mirror ytm_player/services/player.py)
// ---------------------------------------------------------------------------

/// The player module's own monotonic clock — `time.monotonic()` read
/// through the module's `time` attribute, exactly like the reference's
/// global lookup at call time. A test (or any caller) that swaps
/// `ytm_player.services.player`'s `time` for a frozen namespace steers
/// the native engine exactly like the fallback: the clock the reference
/// would read is the clock the engine reads.
fn player_module_clock(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let player = PyModule::import(py, "ytm_player.services.player")?;
    let time_module = player.getattr("time")?;
    let clock = time_module.getattr("monotonic")?;
    clock.call0()
}

/// The Homebrew libmpv discovery walk — `services/player.py`'s
/// `_find_brew_libmpv`, the import-time glue that finds libmpv in the
/// brew prefixes and the user-local prefix ctypes.util.find_library
/// cannot see (#90, #101, #104).
///
/// Mirrors the pure-Python reference exactly: the pattern pair selected
/// by `sys.platform` read through the interpreter's own `sys` module
/// (the `==` CPython's own, so a patched platform steers both backends
/// identically — the tests exercise the darwin pair off-darwin), the
/// prefix list read through the player module's own `_BREW_LIB_DIRS`
/// name (a patched list included), then one pass over the prefixes —
/// each `is_dir()` and `glob(pattern)` called through the prefix
/// object's own protocol, each hit list ordered through CPython's own
/// `sorted` (Path's ordering and its comparison errors are the
/// interpreter's), the pattern order the reference's (`libmpv.so`
/// before `libmpv.so.*`, the unversioned name preferred), the first
/// hit stringified through `str()` — with every per-prefix probe's
/// `OSError` swallowed exactly like the reference's
/// `except OSError: continue` and every other error propagated. The
/// first surviving hit's path is answered, or `None` when no prefix
/// produced one. The `ctypes.util.find_library("mpv")` gate around the
/// reference's call site stays Python-side in the module import.
#[pyfunction]
fn player_find_libmpv(py: Python<'_>) -> PyResult<Option<String>> {
    // sys.platform — read through the interpreter's own sys module, the
    // reference's own global lookup.
    let platform = PyModule::import(py, "sys")?.getattr("platform")?;
    let patterns: [&str; 2] = if platform.eq("darwin")? {
        ["libmpv.dylib", "libmpv.*.dylib"]
    } else {
        ["libmpv.so", "libmpv.so.*"]
    };
    let dirs = PyModule::import(py, "ytm_player.services.player")?.getattr("_BREW_LIB_DIRS")?;
    let sorted_fn = py.import("builtins")?.getattr("sorted")?;
    for lib_dir in dirs.try_iter()? {
        let lib_dir = lib_dir?;
        // The reference's per-prefix body — every probe inside the one
        // `except OSError: continue` arm.
        let probe: PyResult<Option<String>> = (|| {
            if !lib_dir.call_method0("is_dir")?.is_truthy()? {
                return Ok(None);
            }
            for pattern in patterns {
                // hits = sorted(lib_dir.glob(pattern))
                let hits = sorted_fn.call1((lib_dir.call_method1("glob", (pattern,))?,))?;
                if hits.is_truthy()? {
                    // return str(hits[0])
                    let first = hits.get_item(0)?;
                    return Ok(Some(first.str()?.extract::<String>()?));
                }
            }
            Ok(None)
        })();
        match probe {
            Ok(Some(path)) => return Ok(Some(path)),
            Ok(None) => continue,
            // except OSError: continue — the subclass-aware check, so
            // every OSError shape the filesystem probes raise is
            // swallowed exactly like the reference's handler.
            Err(err) if err.is_instance_of::<PyOSError>(py) => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(None)
}

/// The end-file verdict — `services/player.py`'s `_end_file_verdict`,
/// the event-matching gate behind `Player._end_file_data`'s handling of
/// mpv's end-file callbacks.
///
/// Mirrors the reference's checks in their exact order, every one
/// through CPython's own protocol: the instance identity test first
/// (`is`, pointer identity — an end-file from a discarded or recovered
/// instance is never the live player's, answered `"drop"`), then the
/// loading gate (`_loading` truthiness — a loadfile reply still in
/// flight parks the event instead of consuming it, answered `"park"`),
/// then the four-way match answered `"ignore"` when any conjunct fails
/// — the `isinstance(entry_id, int)` type gate (subclass-aware, so a
/// `bool` passes exactly like CPython's isinstance), the
/// `entry_id != current_entry_id` through CPython's own `!=`, the
/// `current_track is None` identity check, and the
/// `reason not in (0, 4)` membership spelled as tuple.__contains__'s
/// own walk (CPython's `==` against each element, the searched object
/// the left operand, stopping at the first hit) — and, everything
/// passed, the `reason == 4` fork that separates mpv's error end
/// (reason 4, answered `"error"`) from the natural end (reason 0,
/// answered `"track_end"`). The short-circuit order is the
/// reference's, so every error matches too. The locks, the
/// pending-event buffer, the payload assembly and the dispatch stay
/// Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (instance, current_instance, loading, entry_id, current_entry_id, current_track, reason))]
fn player_end_file_verdict<'py>(
    instance: &Bound<'py, PyAny>,
    current_instance: &Bound<'py, PyAny>,
    loading: &Bound<'py, PyAny>,
    entry_id: &Bound<'py, PyAny>,
    current_entry_id: &Bound<'py, PyAny>,
    current_track: &Bound<'py, PyAny>,
    reason: &Bound<'py, PyAny>,
) -> PyResult<&'static str> {
    // instance is not self._mpv — pointer identity, CPython's `is`.
    if !instance.is(current_instance) {
        return Ok("drop");
    }
    // if self._loading — truthiness, the reference's own gate.
    if loading.is_truthy()? {
        return Ok("park");
    }
    // The reference's four-conjunct match, short-circuited in its
    // order: not isinstance(entry_id, int) or entry_id != current or
    // current_track is None or reason not in (0, 4).
    let matched = entry_id.is_instance_of::<PyInt>()
        && !entry_id.ne(current_entry_id)?
        && !current_track.is_none()
        && (reason.eq(0i64)? || reason.eq(4i64)?);
    if !matched {
        return Ok("ignore");
    }
    // if reason == 4 — the error fork, through CPython's own ==.
    if reason.eq(4i64)? {
        return Ok("error");
    }
    Ok("track_end")
}

/// The seek-target clamp — `services/player.py`'s `_clamp_seek`, the
/// pure half of `Player._clamp_seek_target` behind `seek` and
/// `seek_absolute`.
///
/// Mirrors the reference exactly: the `duration > 0` gate through
/// CPython's own `>` (the int literal `0` the reference's right
/// operand — an unknown duration, `0.0` or negative, leaves the target
/// uncapped; a NaN duration compares False and leaves it uncapped
/// too), then `min(target, duration)` walked exactly like CPython's
/// own `min` call — the target starts as the running minimum and the
/// duration replaces it only when it compares strictly less (so a NaN
/// on either side keeps the target) — and finally `max(0.0, target)`
/// walked the same way — the `0.0` literal starts as the running
/// maximum and the target replaces it only when it compares strictly
/// greater (so a NaN target clamps to the `0.0` arm). The operands
/// pass through as the objects they are; the `self.duration` read
/// stays Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (target, duration))]
fn player_clamp_seek<'py>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    duration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut clamped = target.clone();
    // if duration > 0 — through CPython's own >, the int literal the
    // right operand.
    if duration.gt(0i64)? {
        // min(target, duration) — CPython's min walk: the duration
        // takes over only when it compares strictly less than the
        // running minimum.
        if duration.lt(target)? {
            clamped = duration.clone();
        }
    }
    // max(0.0, clamped) — CPython's max walk: the 0.0 literal starts
    // as the running maximum; the clamped target replaces it only
    // when it compares strictly greater.
    let zero = PyFloat::new(py, 0.0).into_any();
    if clamped.gt(&zero)? {
        Ok(clamped)
    } else {
        Ok(zero)
    }
}

/// The volume clamp — `services/player.py`'s `set_volume` clamp, the
/// `[0, 100]` level gate behind every volume write (the direct `set`
/// and the relative `change_volume` that funnels into it).
///
/// Mirrors the reference's `max(0, min(100, level))` exactly: the
/// inner `min` walked like CPython's own call — the int literal `100`
/// starts as the running minimum and the level takes over only when
/// it compares strictly less — then the outer `max` walked the same
/// way — the int literal `0` starts as the running maximum and the
/// clamped level replaces it only when it compares strictly greater —
/// so boundary values answer the literal arms and a non-numeric level
/// raises the comparison's own `TypeError`. The mpv assignment and the
/// VOLUME_CHANGE dispatch stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (level))]
fn player_clamp_volume<'py>(
    py: Python<'py>,
    level: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // min(100, level) — CPython's min walk: the level takes over only
    // when it compares strictly less than the running minimum.
    let hundred = 100i64.into_pyobject(py)?.into_any();
    let clamped = if level.lt(&hundred)? {
        level.clone()
    } else {
        hundred
    };
    // max(0, clamped) — CPython's max walk: the clamped level replaces
    // the running maximum only when it compares strictly greater.
    let zero = 0i64.into_pyobject(py)?.into_any();
    if clamped.gt(&zero)? {
        Ok(clamped)
    } else {
        Ok(zero)
    }
}

/// The position-dispatch throttle — `services/player.py`'s
/// `_position_due`, the gate behind `_on_time_pos_change`'s
/// POSITION_CHANGE dispatches (lyrics + UI, ~2 per second).
///
/// Mirrors the reference exactly: the clock read through the module's
/// own `time` name first (see `player_module_clock` — a patched or
/// frozen clock steers both backends identically), then
/// `now - last_dispatch` through CPython's own `-` and the
/// `< 0.4` comparison against the float window through CPython's own
/// `<` — so the boundary (`elapsed == 0.4`) is due, a younger elapsed
/// is throttled, and a non-numeric last-dispatch raises the
/// subtraction's own `TypeError`. Answers `None` when the dispatch is
/// throttled, and otherwise the very `now` the gate read — the value
/// the caller stamps as the new last-dispatch and owns until the next
/// due dispatch. The `value is not None` gate and the dispatch stay
/// Python-side in the observer.
#[pyfunction]
#[pyo3(signature = (last_dispatch))]
fn player_position_due<'py>(
    py: Python<'py>,
    last_dispatch: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let now = player_module_clock(py)?;
    let elapsed = now.sub(last_dispatch)?;
    let window = PyFloat::new(py, 0.4).into_any();
    if elapsed.lt(&window)? {
        Ok(None)
    } else {
        Ok(Some(now))
    }
}

/// The command-supersession gate — `services/player.py`'s
/// `_command_current`, the predicate behind `_dispatch`'s
/// `still_current` closure and `_finish_load`'s acceptance check.
///
/// Mirrors the reference exactly: a `None` command (the dispatch
/// default — events a caller dispatched without a play attempt to
/// vouch for) is unconditionally current, answered `True`; otherwise
/// `command == generation` through CPython's own `==` (equal-valued
/// non-identical objects match) and `current_track is not None`
/// identity — so a superseded play attempt's callbacks are dropped
/// even once scheduled, and a stop that cleared the track invalidates
/// a still-matching command's queued dispatches. `_finish_load`'s
/// acceptance check runs the same predicate — its command is never
/// `None`, so the first arm never fires there. The locks and the
/// state they guard stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (command, generation, current_track))]
fn player_command_current(
    command: &Bound<'_, PyAny>,
    generation: &Bound<'_, PyAny>,
    current_track: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if command.is_none() {
        return Ok(true);
    }
    Ok(command.eq(generation)? && !current_track.is_none())
}

// ---------------------------------------------------------------------------
// IPC request gate (mirror ytm_player/ipc.py)
// ---------------------------------------------------------------------------

/// A fresh `{"verdict": "reject", "error": …}` outcome row for the
/// request gate, carrying the response error the reference spells.
fn ipc_gate_reject<'py>(py: Python<'py>, error: &str) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("verdict", "reject")?;
    out.set_item("error", error)?;
    Ok(out)
}

/// The ipc module's own `secrets` name — the `compare_digest` the
/// reference's global lookup would call, read through the module
/// attribute at call time. A test (or any caller) that swaps
/// `ytm_player.ipc`'s `secrets` for a stub namespace steers the native
/// engine exactly like the fallback: the digest the reference would
/// call is the digest the engine calls.
fn ipc_module_compare_digest(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.ipc")?
        .getattr("secrets")?
        .getattr("compare_digest")
}

/// The request gate — `ipc.py`'s `_ipc_request_gate`, the
/// key-exchange body behind `IPCServer._client_connected`: the token
/// gate and command whitelist a parsed request must pass before the
/// handler dispatches with its args.
///
/// Mirrors the reference's three gates in their exact order, every
/// read through CPython's own protocol. On the TCP transport (*token*
/// not `None` — identity, not truthiness, so an empty-string token
/// still gates; the Unix-socket transport passes `None` and its 0600
/// socket perms are the gate), the request's `token` member —
/// `request.get("token")` through the mapping's own method — must be
/// a `str` (the subclass-aware isinstance gate, short-circuited
/// before any digest, exactly like the reference's `or`) comparing
/// equal to the server's token through `secrets.compare_digest` over
/// both sides' `.encode("utf-8")` — the presented side encoded first,
/// the digest fetched from the module's own `secrets` name at call
/// time (a patched namespace steers both backends identically) — and
/// a missing or non-str member, or any mismatch, rejects with the
/// reference's `authentication failed`. `command` —
/// `request.get("command", "")` — must then be a `str` in
/// *valid_commands* (the containment protocol over the module's own
/// `_VALID_COMMANDS` frozenset, passed through so a patched whitelist
/// steers both backends identically), and anything else rejects with
/// the reference's f-string — the raw command object interpolated
/// through `str()`. `args` — `request.get("args", {})`, the fresh
/// default dict built before the lookup runs — passes through as the
/// object it is when it is a dict (the subclass-aware check) and is
/// another fresh dict otherwise. The accept verdict carries the
/// command and the args as the objects the handler dispatches with;
/// the framing, the JSON parse, the handler await and the response
/// writes stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (request, token, valid_commands))]
fn ipc_request_gate<'py>(
    py: Python<'py>,
    request: &Bound<'py, PyAny>,
    token: &Bound<'py, PyAny>,
    valid_commands: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let get = request.getattr("get")?;
    // if token is not None — identity, not truthiness.
    if !token.is_none() {
        // presented = request.get("token")
        let presented = get.call1(("token",))?;
        // not isinstance(presented, str) — short-circuited before the
        // digest, exactly like the reference's `or`.
        if !presented.is_instance_of::<PyString>() {
            return ipc_gate_reject(py, "authentication failed");
        }
        // secrets.compare_digest(presented.encode("utf-8"),
        //                         token.encode("utf-8")) — the
        // presented side encoded first, the digest read through the
        // ipc module's own secrets name.
        let presented_bytes = presented.getattr("encode")?.call1(("utf-8",))?;
        let token_bytes = token.getattr("encode")?.call1(("utf-8",))?;
        let compare_digest = ipc_module_compare_digest(py)?;
        if !compare_digest
            .call1((presented_bytes, token_bytes))?
            .is_truthy()?
        {
            return ipc_gate_reject(py, "authentication failed");
        }
    }
    // command = request.get("command", "")
    let command = get.call1(("command", ""))?;
    // not isinstance(command, str) or command not in valid_commands
    if !command.is_instance_of::<PyString>() || !valid_commands.contains(&command)? {
        // f"unknown command: {command}" — the object interpolated
        // through str().
        return ipc_gate_reject(py, &format!("unknown command: {}", py_str(&command)?));
    }
    // args = request.get("args", {}) — the fresh default built before
    // the lookup runs, exactly like the reference's argument
    // evaluation.
    let args_default = PyDict::new(py).into_any();
    let args_raw = get.call1(("args", &args_default))?;
    let args = if args_raw.is_instance_of::<PyDict>() {
        args_raw
    } else {
        PyDict::new(py).into_any()
    };
    let out = PyDict::new(py);
    out.set_item("verdict", "accept")?;
    out.set_item("command", command)?;
    out.set_item("args", args)?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// yt-dlp instance gates (mirror the resolver's acquisition loop in
// ytm_player/services/stream.py)
// ---------------------------------------------------------------------------

/// The cookiejar-signature reuse gate — `services/stream.py`'s
/// `_ydl_cookiejar_current`, the gate behind `_get_ydl`'s locked
/// reuse arm.
///
/// Mirrors the reference's single comparison through CPython's own
/// `==` (the `__eq__` protocol — identity-matched singletons, the
/// element-wise tuple walk, reflected lookups and every error shape
/// the interpreter's): `None` matches `None` — an instance built
/// anonymous while streaming is still anonymous is reused — and the
/// signature tuples match element-wise, so a change to any of the
/// three files the signature stats (the stream cookiejar rewritten by
/// `ytm setup` or a mid-session auto-refresh, but also auth.json or
/// account.json replaced on its own) answers False and the caller
/// discards the instance and rebuilds one that loads the new cookies.
/// The signature probe itself (the settings read and the three file
/// stats — `_session_cookiejar_signature`, evaluated by the caller
/// only when an instance exists) and the instance state stay
/// Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (cached_sig, current_sig))]
fn stream_ydl_cookiejar_current(
    cached_sig: &Bound<'_, PyAny>,
    current_sig: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    cached_sig.eq(current_sig)
}

/// The mid-build publication verdict — `services/stream.py`'s
/// `_ydl_publish_verdict`, the gate behind `_get_ydl`'s second locked
/// section, consulted after the unlocked `_build_ydl_opts()` pass.
///
/// Mirrors the reference's two checks in order: the
/// `instance is not None` identity test (another thread's build
/// published an instance while this caller's opts were building —
/// adopt the winner's and never construct a duplicate) answered
/// `"adopt"`, then the `current_generation != snapshot_generation`
/// inequality through CPython's own `!=` (a `_reset_ydl()` landing
/// while the opts built invalidates them — pre-reset settings are
/// baked in — so the loop discards the just-built opts and re-reads
/// current settings) answered `"retry"`, and otherwise the
/// `"install"` verdict that publishes the built instance with the
/// signature it vouched for. The locks, the `_active_resolves`
/// accounting, the YoutubeDL construction and the cached instance
/// fields stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (instance, snapshot_generation, current_generation))]
fn stream_ydl_publish_verdict(
    instance: &Bound<'_, PyAny>,
    snapshot_generation: &Bound<'_, PyAny>,
    current_generation: &Bound<'_, PyAny>,
) -> PyResult<&'static str> {
    if !instance.is_none() {
        return Ok("adopt");
    }
    if current_generation.ne(snapshot_generation)? {
        return Ok("retry");
    }
    Ok("install")
}
