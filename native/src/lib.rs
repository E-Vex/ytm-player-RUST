//! Native (Rust/PyO3) extension for ytm-player: the ported computation
//! cores of the Python package, exposed to Python as `_ytm_player_native`.
//!
//! The module tree mirrors the Python package layout — each module hosts
//! the engines ported from the `src/ytm_player/` module of the same name
//! (each file's `//!` header lists its exact coverage). Every `#[pyfunction]`
//! and `#[pyclass]` is registered by the `#[pymodule]` at the bottom of this
//! file, using qualified paths.

use pyo3::prelude::*;

pub(crate) mod app;
pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod ipc;
pub(crate) mod json;
pub(crate) mod services;
pub(crate) mod ui;
pub(crate) mod utils;

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

#[pymodule]
fn _ytm_player_native(_py: Python, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::format_duration,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(utils::formatting::truncate, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::format_count,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::format_size,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::get_video_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::is_valid_video_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::extract_artist,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::extract_duration,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::normalize_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::strip_vl_prefix,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::build_playlist_subtitle,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::sanitize_title_for_lyric_lookup,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::format_ago,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::formatting::clean_shelf_title,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::regions::normalise_region,
        module
    )?)?;
    // The chart-region table (the same module's CHART_REGIONS): the
    // 68 (code, name) pairs — ZZ first, then alphabetical by name —
    // behind the region picker, the charts queries and the
    // shelf-title country suffix strip.
    module.add_function(pyo3::wrap_pyfunction!(
        services::regions::chart_regions,
        module
    )?)?;

    // BiDi text utilities (ytm_player.utils.bidi equivalents).
    module.add_function(pyo3::wrap_pyfunction!(utils::bidi::has_rtl, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(utils::bidi::isolate_bidi, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(utils::bidi::reorder_line, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(utils::bidi::wrap_line, module)?)?;
    // The gated public wrappers behind the render path — the hot
    // per-fragment line reorder and the lyrics sidebar's multi-line
    // pre-wrap, each with the empty/no-RTL passthrough and the
    // bidi_mode verdict folded into one native call (the cached
    // settings read stays Python-side, passed in as a bool).
    module.add_function(pyo3::wrap_pyfunction!(
        utils::bidi::reorder_rtl_line,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(utils::bidi::wrap_rtl_line, module)?)?;
    // The fused per-cell render engines — the whole
    // reorder/truncate/isolate (and wrap/isolate) chains the render
    // sites used to assemble through three separate calls, folded into
    // one native call per rendered cell or lyric line (the cached
    // settings read stays Python-side, passed in as a bool).
    module.add_function(pyo3::wrap_pyfunction!(
        utils::bidi::render_track_cell,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::bidi::render_bar_cell,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::bidi::render_lyric_line,
        module
    )?)?;
    // The fused whole-list walk behind the lyrics sidebar's two views —
    // every line's render / RTL classification / transliteration in one
    // native call instead of two-to-four per line (the cached settings
    // read stays Python-side, passed in as a bool).
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::lyrics_view_rows,
        module
    )?)?;
    // The YTM timed-lyrics line conversion feeding the synced view —
    // the LyricLine-object walk behind _process_lyrics' synced arm
    // (the hasattr gate, the / 1000.0 seconds through the
    // interpreter's own true division, the text default), in one
    // native call for the whole list; the hasTimestamps gate and the
    // isinstance narrowing stay Python-side in the sidebar.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::convert_lyric_lines,
        module
    )?)?;
    // The synced view's scroll math — the current-line bisect behind
    // _on_position_change (CPython's own < comparisons, so exotic
    // timestamp objects steer the search like they steer the
    // reference), the restyle pass behind the current_line_index
    // reactive (which rows change class, to which of
    // played/current/upcoming), the follow-mode verdict (still on, or
    // resumed once the manual-scroll pause has lapsed — the monotonic
    // clock and the resume delay travel in as plain values) and the
    // centering target (max(0, y - viewport//2 + h//2)); the reactive
    // gates, the geometry reads and the widget calls stay Python-side.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::lyrics_line_for_position,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::lyrics_highlight_changes,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::lyrics_scroll_follows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::lyrics_scroll_target,
        module
    )?)?;

    // Transliteration (ytm_player.utils.transliteration equivalent).
    module.add_function(pyo3::wrap_pyfunction!(
        utils::transliteration::transliterate,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::transliteration::has_non_ascii,
        module
    )?)?;

    // Diagnostics (ytm_player.utils.doctor / logging equivalents).
    module.add_function(pyo3::wrap_pyfunction!(utils::doctor::redact, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(utils::doctor::argv_is_ytm, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::logging::filter_log_lines,
        module
    )?)?;

    // yt-dlp option normalisation (ytm_player.services.yt_dlp_options
    // equivalents). The path expanders (normalize_cookiefile /
    // normalize_cafile) stay in Python: pathlib's expanduser handles
    // `~otheruser` via the pwd database, which has no std equivalent.
    module.add_function(pyo3::wrap_pyfunction!(
        services::yt_dlp_options::normalize_remote_components,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::yt_dlp_options::normalize_js_runtimes,
        module
    )?)?;
    // The configured-extras layering (the same module's
    // apply_configured_yt_dlp_options core): the opts mutation pass
    // behind StreamResolver._build_ydl_opts' second half — the
    // cookiefile write, the CA bundle's missing-file warning and
    // no-certifi compat_opts conversion, and the remote_components /
    // js_runtimes writes — with the file probe and the logger.warning
    // call supplied as thunks, invoked exactly where the reference
    // performed them, and the settings reads and their normalisations
    // staying Python-side in the caller.
    module.add_function(pyo3::wrap_pyfunction!(
        services::yt_dlp_options::yt_dlp_configured_opts,
        module
    )?)?;

    // IPC single-instance guard (ytm_player.ipc equivalent).
    module.add_function(pyo3::wrap_pyfunction!(ipc::looks_like_ytm, module)?)?;

    // Keymap sequence matching (ytm_player.config.keymap equivalent).
    module.add_class::<config::keymap::KeyMatcher>()?;
    // Key-name normalisation (app/_keys.py equivalent): the
    // Textual-key-to-keymap-name translation that feeds sequence
    // matching — modifier prefixes, printable-character preference,
    // and the special-name remap.
    module.add_function(pyo3::wrap_pyfunction!(app::keys::normalize_key, module)?)?;
    // Count/sequence key-input state machine (app/_keys.py on_key
    // equivalent): the count-prefix accumulation and sequence-buffer
    // resolution behind every keystroke — the digit gate, the
    // exact/pending/no-match classification with the 1000 count cap, and
    // the buffer resets.
    module.add_function(pyo3::wrap_pyfunction!(app::keys::feed_key, module)?)?;

    // Action dispatch engine (app/_keys.py _handle_action equivalent):
    // the dispatch-table walk (which branch handles an action, and the
    // page the navigations target), the pane decision behind the routed
    // navigation actions, the CURRENT_CONTEXT album-target extraction,
    // and the TOGGLE_SHUFFLE lock gate.
    module.add_function(pyo3::wrap_pyfunction!(app::keys::dispatch_route, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::keys::route_navigation_action,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::keys::current_context_target,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::keys::shuffle_lock_blocked,
        module
    )?)?;

    // Playback queue engine (ytm_player.services.queue equivalent).
    module.add_class::<services::queue::QueueCore>()?;

    // Track-table view engine (ui/widgets/track_table.py equivalent):
    // the filter + sort pass that computes the visible row order.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::filter_sort_tracks,
        module
    )?)?;
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
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_row_values,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_view_rows,
        module
    )?)?;
    // Track-table rendering-integration engines (the same module's
    // equivalents): the load/append stamping pass (the dict copy +
    // `_original_index` stamp per track), the marked-rows status line,
    // the selection-info label, the post-removal renumbering with the
    // rebuilt visible map and the refreshed `#` cells, the header-click
    // sort toggle, and the column keys mapping the composed cells onto
    // the table's own columns — with the DataTable `add_row`/
    // `update_cell` calls and the cursor preservation staying
    // Python-side in the widget.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_stamp_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_mark_status,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_selection_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_reindex_after_removal,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_sort_state,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_column_keys,
        module
    )?)?;
    // Track-table mark/range selection engines (the same module's
    // equivalents): the visible-run core behind the shift-click extend
    // and the keyboard range marks, the shift-click extend decision and
    // the range-mode marks themselves, the toggle's symmetric
    // difference, the post-removal selection shift (marks, baseline,
    // anchor and range mode with the generation-bump flag), the
    // marked-tracks walk in displayed order, the hidden-marks count,
    // and the occurrence-key carry that resolves a selection across a
    // background reload — with the mark-cell repainting, the cursor
    // placement and the selection-generation arithmetic staying
    // Python-side in the widget.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_visible_run,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_extend_marks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_range_marks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_toggle_marks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_shift_selection,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_marked_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_hidden_marked_count,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::track_table::track_table_carry_selection_keys,
        module
    )?)?;

    // Session-state validation engine (app/_session.py equivalent):
    // the one-pass validation + coercion of the parsed session.json
    // object, the resume-point scan over the restored queue, and the
    // resume-staging plan — which track comes back, from which
    // playlist, at which position (the float() coercion through
    // CPython's own call).
    module.add_function(pyo3::wrap_pyfunction!(
        app::session::coerce_session_state,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::session::find_track_index,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::session::resume_staging_plan,
        module
    )?)?;

    // Recently Played history-merge engine (ui/pages/recently_played.py
    // equivalent): the account-history cache view model — occurrence
    // stamping/identity, accepted-play supersession, pending-play parking
    // and merging, and the All-tab local+account composition — plus the
    // page's own display tails: the empty-state copy pick, the footer
    // line and the nav-state remember-gates.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::stamp_track,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::occurrence_key,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::with_accepted_play,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::with_accepted_local_play,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::park_pending_play,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::merge_pending_account_plays,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::enrich_track,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::compose_all_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::recent_empty_message,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::recent_footer_text,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::recently_played::recent_nav_state,
        module
    )?)?;

    // Navigation stack engine (app/_navigation.py equivalent): the
    // back/forward stack machine behind page navigation — snapshotting
    // the page being left, the stack caps, the same-page-back shortcut,
    // forward-history invalidation and cached-state restore — plus the
    // purge that drops every stack/cache reference to deleted playlists.
    module.add_function(pyo3::wrap_pyfunction!(
        app::navigation::resolve_navigation,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::navigation::purge_dead_playlist_refs,
        module
    )?)?;

    // Browse home-feed shelf classification (ui/pages/browse.py
    // equivalent): the event-vs-chart partition with priority pill
    // ordering behind the Charts tab, and the playlist-row extraction
    // with per-id dedup behind the Playlists tab.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::is_event_shelf,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::chart_sort_key,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::split_events_and_charts,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::is_playlist_row,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::playlist_rows,
        module
    )?)?;
    // Browse rendering path (the same module's equivalent): the
    // item-routing decision core behind navigate_browse_item, the For
    // You shelf view model (the title plus up-to-eight (display, item)
    // rows), the Charts tab's raw-shelf extraction, pill-row view
    // model, country line and event-row visibility gate, and the
    // Releases / Playlists / Subscriptions row labels — the per-section
    // render passes, with the widget mounting and the worker
    // orchestration staying Python-side.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::browse_item_route,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::foryou_shelf_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::charts_raw_shelves,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::charts_pill_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::charts_country_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::charts_event_row_visible,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::new_release_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::playlist_row_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::browse::subscription_label,
        module
    )?)?;

    // Search-page label glue (ui/pages/search.py equivalent): the
    // item-to-label composer behind the Albums/Artists/Playlists panels
    // of the search results area, the bottom-of-panel count label, the
    // mode indicator beside the input, the suggestion-overlay pick
    // decision, the recent-searches label pass, the _search_all
    // categorization bucketing one unfiltered search into the four
    // panels, the result-item navigation decisions behind the panel
    // click handlers (the left-click context-page pick, the right-click
    // item type, the subscribe-id and copy-link resolutions), the
    // music-mode worker merge classifying the gathered concurrent
    // fetches, and the nav-state remember-gates behind get_nav_state.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::format_search_result,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_panel_count_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_mode_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_suggestion_pick,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::recent_search_labels,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::categorize_search_results,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_item_navigation,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_panel_item_type,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_subscribe_browse_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_browse_link,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::music_search_results,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::search::search_nav_state,
        module
    )?)?;

    // Sidebar pane-routing state machine (app/_sidebar.py equivalent):
    // the three-pane focus router — per-page playlist-sidebar visibility
    // (read/toggle/ensure), the visible-pane list, pane resolution from
    // the focused widget's ancestor chain, the focus gate, and the
    // left/right/cycle movement decisions behind Ctrl+w h / l / w.
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::sidebar_visible_for_page,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::toggle_sidebar_state,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::ensure_sidebar_visible,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(app::sidebar::visible_panes, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::pane_of_focused,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::pane_focus_target,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::next_pane_left,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::next_pane_right,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::sidebar::next_pane_cycle,
        module
    )?)?;

    // Playlist panel view engine (ui/sidebars/playlist_sidebar.py
    // equivalent): the label/count view model behind the sidebar's
    // LibraryPanel — the row label and footer count label, the `/`-filter
    // pass, the playlist-id matching behind the optimistic sidebar
    // mutations (removal / count updates / auto-select / the Liked Music
    // exclusion), the count arithmetic, and the info-bar title read.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::format_panel_item,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::panel_count_label,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::filter_panel_items,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::matches_removal,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::matches_count_update,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::bump_playlist_count,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::selection_title,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::is_liked_music_playlist,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::matches_auto_select,
        module
    )?)?;

    // Context-page view-model assembly (ui/pages/context.py equivalent):
    // the header/discography view model behind the album, artist and
    // playlist detail pages — the album and playlist header compositions
    // (title, subtitle, unavailable count, the playlist's owned flag and
    // remaining-tail fetch gate), the artist discography (top songs,
    // the albums+singles concatenation, subscribers and the
    // similar-artists bar), the see-all section selection behind the
    // discography expansion, the artist-songs worker merge's decision
    // core — the first-batch plan and the remaining-fetch decision
    // behind `_fetch_full_artist_songs`'s swap of the artist's full
    // song list into the top-songs table, plus the tail plan for the
    // chained remaining-tracks fetch — the worker-end failure copies
    // (the log-pointing failure message and the empty-load branch's
    // message pick), the artist album-list row assembly, and the
    // header-button resolutions (the Add-to-Library id chain with its
    // VL strip, the Start-Radio item assembly) plus the discography
    // swap applying the fetched see-all lists.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::album_header,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::playlist_header,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::artist_discography,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::expandable_sections,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::artist_songs_first_batch,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::artist_songs_tail,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::load_failure_message,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::empty_load_message,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::library_playlist_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::radio_item,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::apply_expanded_sections,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::context::artist_album_rows,
        module
    )?)?;

    // Play-history row engine (ytm_player/services/history.py
    // equivalent): the skip gate and insert-row assembly behind
    // HistoryManager.log_play, and the stored-vs-new listened-seconds
    // decision (the int coercion, no-write gate and delta) behind
    // update_play_listened_seconds.
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_log_row,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::listened_delta,
        module
    )?)?;

    // Command palette catalog (app/_commands.py equivalent): the
    // (name, help) catalog behind the command provider — the data both
    // the discovery list and the query search dispatch over, with the
    // per-entry callback resolution staying on the Python side next to
    // the Textual hit machinery.
    module.add_function(pyo3::wrap_pyfunction!(
        app::commands::command_catalog,
        module
    )?)?;

    // Stats aggregation payload (ytm_player/services/history.py
    // equivalent): the five-key listening-stats dict assembled behind
    // HistoryManager.get_stats from the three scalar aggregates and
    // the two top lists.
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::stats_payload,
        module
    )?)?;

    // History-read row materialisation (ytm_player/services/history.py
    // and cli.py equivalents): the [dict(row) for row in rows]
    // comprehension behind every history read — the getters'
    // fetchall() lists and the CLI's _query_history alike — converted
    // through CPython's own dict constructor.
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::row_dicts,
        module
    )?)?;

    // CLI stats aggregation payload (cli.py equivalent): the four-key
    // listening-stats dict assembled behind the standalone `ytm stats`
    // command, with the top-tracks list rebuilt from its row tuples'
    // first four fields.
    module.add_function(pyo3::wrap_pyfunction!(cli::cli_stats_payload, module)?)?;

    // CLI history-query statement engines (cli.py equivalents): the
    // two statement texts behind the standalone `ytm history` and
    // `ytm history search` reads — every column, newest first,
    // bounded by the caller's limit, each with the limit passed
    // through as the object it is into the parameter tuple. The
    // connect/row-factory/execute/fetch glue and the error exits
    // stay Python-side in _query_history.
    module.add_function(pyo3::wrap_pyfunction!(cli::cli_history_select, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        cli::cli_search_history_select,
        module
    )?)?;

    // CLI cache-status engines (cli.py equivalents): the six-key status
    // payload behind the standalone `ytm cache status --json` command,
    // the four aligned human lines it echoes without --json — both from
    // the cache directory and the walk's three aggregates — and the
    // directory walk that produces them, with the exists/iterdir/
    // is_file/stat operations supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(cli::cache_status_payload, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cli::cache_status_text, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(cli::cache_status_walk, module)?)?;

    // CLI cache-clear walk (cli.py equivalent): the directory walk
    // behind the standalone `ytm cache clear` command — which entries
    // of the cache directory are unlinked and how many were, with the
    // exists/iterdir/is_file/unlink operations supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(cli::cache_clear_walk, module)?)?;

    // CLI config editor resolution (cli.py equivalent): the plan behind
    // the standalone `ytm config` command — which editor opens the
    // config file, whether Windows shell-opens it instead, and the
    // no-EDITOR no-xdg-open manual-open error.
    module.add_function(pyo3::wrap_pyfunction!(cli::config_editor_plan, module)?)?;

    // IPC dispatch engine (app/_ipc.py _handle_ipc_command equivalent):
    // the dispatch-table walk behind the CLI-shells command router —
    // which branch of the match handles each command string, and the
    // rating the like/dislike/unlike branches rate with.
    module.add_function(pyo3::wrap_pyfunction!(
        app::ipc::ipc_dispatch_route,
        module
    )?)?;

    // IPC seek-offset parser (app/_ipc.py _ipc_seek equivalent): the
    // classification of one offset value into the seek it names —
    // relative (+/-seconds), clock (mm:ss / hh:mm:ss) and absolute
    // seconds, with the missing/invalid verdicts the response carries.
    module.add_function(pyo3::wrap_pyfunction!(app::ipc::ipc_seek_offset, module)?)?;

    // IPC request gate (ipc.py _ipc_request_gate equivalent): the
    // key-exchange body behind IPCServer._client_connected — the TCP
    // transport's token gate (the presented token compared through the
    // module's own secrets.compare_digest name over both sides' utf-8
    // bytes) and the command whitelist (the containment protocol over
    // the module's own _VALID_COMMANDS frozenset), with the args
    // coercion the dispatch runs with. The framing, the JSON parse,
    // the handler await and the response writes stay Python-side in
    // the server.
    module.add_function(pyo3::wrap_pyfunction!(ipc::ipc_request_gate, module)?)?;

    // CLI JSON serialisation (cli.py _json_output equivalent): the
    // json.dumps(data, indent=..., default=str) text behind every
    // JSON-emitting CLI command — the ensure_ascii escaping, the
    // float/int reprs, the indent-aware layout, the key coercion and
    // the circular-reference guard.
    module.add_function(pyo3::wrap_pyfunction!(json::json_dumps, module)?)?;

    // Doctor report assembly (utils/doctor.py gather_diagnostics
    // equivalent): the multi-section `ytm doctor` text — the banners,
    // the aligned identity and paths labels, the crash-file section,
    // the final join and the redaction pass — from the probes'
    // results.
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::doctor_report,
        module
    )?)?;

    // Doctor probe engines (utils/doctor.py equivalents): the
    // mpv-warning tail behind the Recent mpv warnings/errors probe, the
    // `Fatal Python error:` tail extraction behind the Most recent
    // faulthandler trace probe, and the crash-staleness verdict behind
    // the Most recent crash file section — the metadata-header scan, the
    // PEP 440 comparison of the recorded version against the installed
    // one, and the note assembly.
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::recent_mpv_lines,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::faulthandler_tail,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::crash_staleness_note,
        module
    )?)?;

    // Doctor /proc and subprocess probe engines (utils/doctor.py
    // equivalents): the Process status section's ytm-running verdict —
    // the /proc walk, the cmdline decode and entry-point match, the
    // VmRSS parse, the own-pid exclusion — and the `mpv:` version line
    // behind the mpv status probe, both with their side-effecting
    // probes (/proc reads, shutil.which, the --version subprocess run)
    // supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::running_status_text,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::mpv_version_text,
        module
    )?)?;

    // Doctor import-probe engines (utils/doctor.py equivalents): the
    // libmpv loadability verdict — the player-module import and the
    // last-line render of its failure message — and the MPRIS
    // availability verdict — the platform gate, the system render and
    // the mpris-module import's BUS_NAME/DBUS_AVAILABLE pair — with the
    // imports themselves supplied as thunks.
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::libmpv_status_text,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        utils::doctor::mpris_status_text,
        module
    )?)?;

    // Theme resolution engines (ui/theme.py and app/_app.py equivalents):
    // the Rich-safe ANSI token translation, the 21-field ThemeColors
    // resolution from CSS variables behind from_css_variables, the
    // app-variable overlay + theme.toml override merge behind
    // get_css_variables, and the 21-field rebuild from a Textual Theme's
    // raw colours behind watch_theme — the three sites of the documented
    // accent → primary → DEFAULT_LYRIC_CURRENT lyric-current chain.
    module.add_function(pyo3::wrap_pyfunction!(ui::theme::rich_safe_color, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::theme::theme_fields_from_css,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::theme::app_css_variables,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::theme::theme_fields_from_theme,
        module
    )?)?;

    // Spotify-import matching cores (services/spotify_import.py
    // equivalents): the playlist/album ID pull behind the import URL
    // parse, the spotipy item normaliser behind every track the
    // Web-API path ingests, the weighted fuzzy score and the
    // score-sort-threshold classification behind the per-track matcher,
    // the per-page item walk behind the spotipy pagination loop, the
    // track-row walk behind the scraper fallback, the candidate display
    // row behind the interactive prompt, the confirmed/video-ids/
    // skipped accounting behind the import flow's collection step, and
    // the match-type bucket walk behind the result categorisation.
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::extract_playlist_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::parse_spotipy_item,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_fuzzy_score,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_match_classification,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_page_items,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_scraper_track_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_display_candidate,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_summarize_results,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::spotify_import::spotify_bucket_results,
        module
    )?)?;

    // Settings TOML value formatting (config/settings.py equivalent):
    // the value-to-TOML-literal renderer behind every line
    // Settings.save writes.
    module.add_function(pyo3::wrap_pyfunction!(
        config::settings::format_toml_value,
        module
    )?)?;

    // Keymap load plan (config/keymap.py _load_from_dict equivalent):
    // the section walk, action validation and binding classification
    // behind keymap.toml loading, applied by the Python caller as it
    // replays the plan.
    module.add_function(pyo3::wrap_pyfunction!(
        config::keymap::keymap_load_plan,
        module
    )?)?;

    // Path table (config/paths.py equivalent): the platform-aware
    // computation behind every filesystem constant the module exports —
    // the win32 APPDATA/LOCALAPPDATA bases with their home-tree
    // fallbacks, the XDG base gates (empty reads as unset) and the
    // derived files under CONFIG_DIR; a pure function of
    // (platform, home, env) so the win32 branch is exercisable from
    // any host. ensure_dirs/secure_chmod's I/O stays Python-side.
    module.add_function(pyo3::wrap_pyfunction!(config::paths::paths_core, module)?)?;

    // Actions-popup menu builder (ui/popups/actions.py equivalent): the
    // context-menu assembly behind every track/album/artist/playlist
    // action menu — the per-type catalog walk with the queue-membership
    // swap, the playlist-view and search-source gates, and the
    // like/subscribe label flips.
    module.add_function(pyo3::wrap_pyfunction!(ui::popups::build_actions, module)?)?;
    // The actions popup's title derivation (the same module's
    // _title_text): the title/name/artist or-chain with the capitalized
    // item type behind it, and the 34-character truncation — through
    // CPython's own len/slice/add protocol so exotic names answer
    // exactly like the reference.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::actions_popup_title,
        module
    )?)?;
    // The confirm / input / create-playlist popup cores (the same
    // package's remaining pure computations): the confirm popup's
    // button-id verdict, the input popup's submitted-value
    // normalization, the create-playlist popup's chrome-label pair and
    // its submit normalization (strip + str() with the empty-name
    // gate).
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::confirm_popup_button_result,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::input_popup_submitted_value,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::create_playlist_chrome_labels,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::create_playlist_submit_values,
        module
    )?)?;

    // The header-bar / selection-info-bar / track-filter cores (the
    // three remaining ui/ modules' pure computations): the header bar's
    // click routing and lyrics-toggle class choice, the selection bar's
    // truncation width (a zero widget width reads the 80-column
    // default, the padding-reserved floor is 10), and the track
    // filter's input-ownership gate and escape-clear verdict.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::header_bar::header_bar_click_route,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::header_bar::header_bar_lyrics_class,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::selection_info_bar::selection_info_visible_width,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::track_filter::track_filter_owns_input,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::track_filter::track_filter_escape_clears,
        module
    )?)?;

    // Progress-bar render geometry (ui/widgets/progress_bar.py
    // equivalents): the progress fraction, the time-prefix/suffix/
    // bar-width metrics, the filled/empty/marker cell computation behind
    // every bar render tick, and the click-to-seek column mapping.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::progress_bar::progress_fraction,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::progress_bar::progress_bar_metrics,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::progress_bar::progress_bar_cells,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::progress_bar::progress_x_to_seconds,
        module
    )?)?;

    // Synced-lyrics LRC parser (ui/sidebars/lyrics_sidebar.py
    // equivalent): the timestamp extraction behind every string-lyrics
    // load — the [mm:ss(.fff)] match at each line's start, the
    // fractional-second conversion, the timestamp stripping and the
    // stable timestamp sort.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::sidebars::parse_synced_lyrics,
        module
    )?)?;
    // LRCLIB lookup glue (services/lrclib.py equivalent): the
    // query-params assembly behind get_synced_lyrics' request build —
    // the track_name/artist_name pair in the reference's insertion
    // order plus the identity-gated duration through the interpreter's
    // own int() and str() — and the syncedLyrics extraction behind its
    // parse step — the mapping's own get plus the `or None`
    // truthiness gate. The title sanitizer is the already-ported
    // formatting engine; the requests fetch and the resp.json() decode
    // stay Python-side.
    module.add_function(pyo3::wrap_pyfunction!(
        services::lrclib::lrclib_query_params,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::lrclib::lrclib_synced_lyrics,
        module
    )?)?;

    // Update-check version comparison (services/update_check.py
    // equivalent): the strictly-newer PEP 440 verdict behind
    // check_for_update's cache-hit and post-fetch decisions, on the same
    // packaging grammar shape the installed release compiled (the fold
    // flag the Python side probes once and passes down).
    module.add_function(pyo3::wrap_pyfunction!(
        services::update_check::pep440_is_newer,
        module
    )?)?;

    // Help-page rows engine (ui/pages/help.py equivalents): the
    // keybinding-reference row assembly (the action-value title-casing
    // through CPython's own str.title semantics, the key-sequence join,
    // the "(unbound)" placeholder) and the filtered walk (the
    // lowercased-then-stripped needle, the haystack containment, the
    // category-header insertion) behind the searchable help table.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::help::help_page_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::help::help_page_visible_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::help::help_page_rows_for_keymap,
        module
    )?)?;

    // Playlist-picker view model (ui/popups/playlist_picker.py
    // equivalents): the recent-list move-to-front bookkeeping behind
    // _record_recent, the recency sort behind _sort_by_recent, the
    // filtered visible-row walk behind _populate_list and the
    // playlist-count status label.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::playlist_record_recent,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::playlist_picker_sort,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::playlist_picker_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::popups::playlist_picker_count_label,
        module
    )?)?;

    // Queue-page view & mutation engines (ui/pages/queue.py
    // equivalents): the now-playing header view model shared by
    // _update_current_track and _refresh_queue, the radio-seed source
    // label behind _update_queue_source, the footer text behind
    // _update_footer, and the optimistic-mutation decisions behind
    // _remove_selected and _move_track (the stale-table queue-position
    // refusal, the post-remove cursor re-landing, and the clamped
    // count-multiplied reorder destination).
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::queue::queue_now_playing,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::queue::queue_source_view,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::queue::queue_footer_text,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::queue::queue_remove_target,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::queue::queue_post_remove_cursor,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::queue::queue_move_target,
        module
    )?)?;

    // Album-art render engines (ui/widgets/album_art.py equivalents): the
    // WCAG colour math behind the placeholder's note glyph (the relative
    // luminance of a six-digit hex colour, the contrast ratio, and the
    // highest-contrast candidate picker with the first-wins tie rule), the
    // half-block cell extraction from the resized thumbnail's raw RGB
    // buffer behind every image render, and the placeholder/empty
    // row-segment plans behind the widget's two no-image renders — with
    // the Pillow decode, the LANCZOS resize, the download and the Rich
    // Text assembly staying Python-side next to the I/O.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::album_art::wcag_relative_luminance,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::album_art::wcag_contrast_ratio,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::album_art::wcag_readable_on,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::album_art::album_art_half_block_cells,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::album_art::album_art_placeholder_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::widgets::album_art::album_art_empty_rows,
        module
    )?)?;

    // Shuffle-preferences LRU engine (services/shuffle_prefs.py
    // equivalent): the ordered {context_id: shuffle} map — the get
    // promotion, the set with its move_to_end and evict-while-over walk,
    // the clear, the hand-edited-file ingestion (the dict gate, the
    // str-key filter, the bool(v) truthiness coercion and the for k, v
    // unpacking contract), and the lock-held snapshot +
    // json.dumps(..., indent=2) serialisation behind _save — with the
    // thread lock and the logging staying Python-side in the facade.
    module.add_class::<services::shuffle_prefs::ShufflePrefsCore>()?;
    // The prefs file's disk I/O cores (also services/shuffle_prefs.py):
    // the read behind _load — the path's own read_text driven from
    // native, the JSON parse through the ported CPython-grammar scanner
    // with the interpreter's own json.loads rendering the exact
    // rejection — and the atomic write behind _save — the parent mkdir,
    // the pathlib with_suffix tmp sibling, the UTF-8 write and the
    // os.replace cutover, each driven through the path object's own
    // protocol so every error shape stays CPython's.
    module.add_function(pyo3::wrap_pyfunction!(
        services::shuffle_prefs::shuffle_prefs_read_file,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::shuffle_prefs::shuffle_prefs_write_file,
        module
    )?)?;

    // Album-art cache LRU engine (ui/widgets/album_art.py equivalent):
    // the ordered URL→Text map behind the module-level art cache — the
    // hit promotion of the cache-check path, the insert that keeps an
    // existing key's position (OrderedDict __setitem__ semantics — this
    // engine never promotes on insert) with its single-pop overflow
    // eviction, and the guarded delete of the resize invalidation, the
    // renders themselves held as the Python objects they are.
    module.add_class::<ui::widgets::album_art::ArtCacheCore>()?;

    // Mutation-failure classifier (services/ytmusic.py equivalent): the
    // exception-to-kind mapping behind every mutation call's except arm
    // — the reference's isinstance chain in its exact order (KeyError,
    // the two timeout classes, requests' RequestException,
    // YTMusicUserError, YTMusicServerError) with the classes resolved
    // from the running interpreter at call time, and the
    // YTMusicServerError arm's HTTP-status prefix match (CPython's \d
    // and \b semantics on str patterns, Unicode digits and their int()
    // values included) deciding auth_expired from server_error.
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::classify_mutation_failure,
        module
    )?)?;

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
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_auth_expired,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_renewal_cooldown_gate,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_renewal_cooldown_stamp,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_api_failure_step,
        module
    )?)?;

    // YTMusic patch-window bookkeeping engines (services/ytmusic.py
    // equivalents): the in-flight counter's admission and release
    // steps behind _call's gate — the += 1 that admits a normal
    // client call past the patch-window protocol's wait, and the -= 1
    // with its drained verdict that releases it, the verdict naming
    // whether the counter reached zero so the caller sets the
    // _no_inflight event a parked patch window waits on — with the
    // _no_patch wait, the event clear/set and the counter assignments
    // staying Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_patch_window_enter,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_patch_window_exit,
        module
    )?)?;

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
    // gate and the renewal machinery staying Python-side in the
    // service.
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_watch_playlist_kwargs,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_result_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_playlist_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_radio_seed_pool,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_discovery_source_order,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_chart_shelves,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_chart_shelf_index,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_discovery_playable_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_chart_playable_tracks,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_explore_new_releases,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_playlist_order_params,
        module
    )?)?;

    // YTMusic RNG-driven selection engines (services/ytmusic.py
    // equivalents): the radio mix's multi-seed shuffle and limit trim
    // behind get_radio's result assembly, the three-track seed sample
    // every discovery mix source draws, and the artist mix's random
    // pick — the RNG itself always flowing through the Python
    // `random` module (the QueueCore engines' own rule), so seeded
    // sequences and monkeypatched draw functions steer both backends
    // identically — with the client calls, the gather and the
    // normalisation staying Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_radio_mix,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_sample_seeds,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::ytmusic::ytm_pick_artist,
        module
    )?)?;

    // Playback-bar track-info width engine (ui/playback_bar.py
    // equivalent): the halves-and-thirds width split behind the
    // one-line title/artist/album render — the max(10, width - 30)
    // budget, the min(len, //2) and min(len, //3) fragment widths, and
    // the remainder-minus-separators album width with its max(0, ...)
    // clamp — with the Rich Text assembly and the theme styling
    // staying Python-side in the widget.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::playback_bar::track_info_widths,
        module
    )?)?;
    // The fused snippet engine over the same render — the whole
    // content walk (the state icon, the width split, the three
    // BiDi-rendered fragments and their em-dash separators, or the
    // no-track placeholder) folded into one native call that answers
    // the (kind, text) snippet sequence the widget styles by kind,
    // with only the cached bidi_mode verdict read Python-side.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::playback_bar::track_info_snippets,
        module
    )?)?;

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
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::eviction_plan,
        module
    )?)?;

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
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::cache_schema,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::cache_hit_commit_plan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::cache_status_summary,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::cache_index_insert_statement,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::cache_index_delete_statement,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::cache::cache_index_delete_all_statement,
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
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::suggestion_strings,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::played_video_id_set,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_insert_params,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_stats_params,
        module
    )?)?;

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
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::parse_record,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::identity_from_record,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(services::auth::owner_of, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::vouch_failure,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::channel_id_from_account_menu,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::serialize_headers,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::candidate_payload,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::build_record,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(services::auth::sha256_hex, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::normalize_raw_headers,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::auth::build_stream_cookiejar,
        module
    )?)?;

    // Region-picker filter engine (ui/popups/country_picker.py
    // equivalent): the case-insensitive code-or-name substring pass
    // behind the chart-region picker's filter input — the query
    // stripped and lowercased, an empty query passing the region list
    // through unchanged (the same object the reference hands back),
    // and every kept row re-packed as a fresh (code, name) pair of the
    // same string objects.
    module.add_function(pyo3::wrap_pyfunction!(ui::popups::filter_regions, module)?)?;

    // Terminal orientation gate (utils/terminal.py equivalent): the
    // aspect-ratio verdict behind the horizontal/vertical layout
    // decision — the rows-zero guard, the true division and the 2.3
    // threshold all through the interpreter's own operators, so every
    // numeric shape and every error matches the reference exactly.
    module.add_function(pyo3::wrap_pyfunction!(
        utils::terminal::get_orientation,
        module
    )?)?;

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
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::history_schema,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::history_retention_delete,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::search_upsert,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::search_history_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::search_suggestion_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::search_history_delete,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_history_insert_sql,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_stats_upsert_sql,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_listened_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_listened_update,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_stats_listened_update,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::play_history_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::recently_played_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::played_video_ids_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::stats_total_plays_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::stats_total_listen_time_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::stats_unique_tracks_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::top_tracks_select,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::history::top_artists_select,
        module
    )?)?;

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
    // cap included) — plus the retry-delay iteration behind
    // _resolve_sync's retry loop (the (attempt, delay) schedule walk,
    // the schedule read through the module's own _RETRY_DELAYS name,
    // a patched schedule included). The YoutubeDL construction, the
    // cookiejar vouching, the network extraction, the sleeps and the
    // cache state stay Python-side in the resolver.
    module.add_function(pyo3::wrap_pyfunction!(services::stream::is_stale, module)?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::wait_seconds,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::validate_quality,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_base_opts,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_info_fields,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::download_error_hint,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_cache_expired_ids,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_cache_cap_evictions,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_retry_delays,
        module
    )?)?;

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
        services::stream::stream_ydl_cookiejar_current,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_ydl_publish_verdict,
        module
    )?)?;

    // Stream pending-future engines (services/stream.py equivalents):
    // the request-dedup bookkeeping around _resolve's shared futures
    // — the join verdict behind _current_pending (the in-flight
    // resolve a same-generation caller joins instead of extracting
    // again, the entry's generation compared through CPython's own
    // !=), the drop verdict behind the finally-block slot cleanup
    // (the slot dropped only when it still holds this owner's very
    // future, by identity), and the publication gate behind all
    // three of the shared future's publications (set_result,
    // set_exception and the owner-cancelled cancel, gated on not
    // future.done() through the future's own method protocol and the
    // interpreter's own truthiness). The _pending dict state, the
    // awaits, the publications and the slot mutations stay
    // Python-side in the resolver.
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_pending_join_verdict,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_pending_drop_verdict,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::stream::stream_pending_publication_gate,
        module
    )?)?;

    // Download-service engines (services/download.py equivalents): the
    // base yt-dlp options dict behind DownloadService._build_opts (the
    // format selector, the staged outtmpl, the quiet flags and the
    // FFmpeg-audio postprocessor) and the five probe filenames a
    // completed download is looked up under (the extension order
    // behind _download_sync's publication walk and get_path's lookup).
    // The staging, the yt-dlp invocation, the atomic publication and
    // the thread ownership stay Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(
        services::download::download_base_opts,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::download::download_candidates,
        module
    )?)?;

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
    // The event-dispatch arm behind _dispatch's per-callback walk
    // (which of the six arms — the two context-aware schedules, the two
    // bare schedules, the no-loop async drop, the inline sync run — a
    // callback takes) and the mpv log-event level routing behind
    // _on_mpv_log (the four mpv level strings onto the logging
    // constants read through the player module's own logging name, any
    // other string onto DEBUG). The locks, the pending-event buffer,
    // the payload assembly, the scheduling calls, the callbacks, the
    // log calls and the mpv calls stay Python-side in the player.
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_find_libmpv,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_end_file_verdict,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_clamp_seek,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_clamp_volume,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_position_due,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_command_current,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_dispatch_arm,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::player::player_mpv_log_level,
        module
    )?)?;

    // Last.fm scrobble engine (services/lastfm.py equivalent): the
    // scrobble-threshold gate behind check_scrobble's periodic poll —
    // whether the playback position has crossed the Last.fm spec's
    // threshold (50% of the track or 4 minutes, whichever is less):
    // the no-duration gate `duration <= 0` answering the bare cap,
    // otherwise min(duration * percent, cap) walked like CPython's own
    // min (the tie answering the product, the reference's first
    // argument), then the `position >= threshold` verdict — every
    // comparison and the product through CPython's own protocols, and
    // both constants read through the lastfm module's own names at
    // call time so patched values steer both backends identically. The
    // connection/current-track/scrobbled state, the duration read out
    // of the current track and the scrobble submission stay
    // Python-side in the service.
    module.add_function(pyo3::wrap_pyfunction!(
        services::lastfm::lastfm_scrobble_due,
        module
    )?)?;

    // Playback coordination engines (app/_playback.py equivalents): the
    // playback-failure policy behind _handle_play_failure (the
    // repeat-one stop, the below-threshold queue advance with the next
    // track pulled through the queue's own bound method exactly when
    // the reference's arm consults it, the end-of-queue toast, and the
    // escalation whose resolver reset is gated on the failure kind and
    // the resolver's truthiness — the cap read through the playback
    // module's own _MAX_CONSECUTIVE_FAILURES name), and the
    // history-report poll step shared by the two report timer chains
    // (the listen-threshold gate with its boundary kept on the poll
    // side, the live-track reset of the idle count, and the two-poll
    // stability window's stop). The failure counter, the queue state,
    // the resolver reset, the timer re-arms and the report workers stay
    // Python-side in the mixin.
    module.add_function(pyo3::wrap_pyfunction!(
        app::playback::playback_failure_policy,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::playback::playback_history_report_step,
        module
    )?)?;

    // Track-action engines (app/_track_actions.py equivalents): the
    // occurrence gate behind the actions popup's queue-entry naming
    // (_queue_entry_id — only the Queue page's table keys its rows by
    // entry id), the first-artist and album browse-ID extractions
    // behind the artist/album navigation arms (_first_artist_id and
    // _album_browse_id with their read-order-true priority chains),
    // the ActionsPopup answer classification (_handle_action_result's
    // dispatch — which handler each picked action routes to), the
    // entity-action dispatch (_dispatch_entity_action's route table —
    // which method each (action, item type) pair answers, the entity
    // id/name extraction, the missing-id early-outs, and the
    // artist/album navigation targets), the like-toggle rating
    // resolution, the Remove-from-Queue plan (the occurrence arm, the
    // stale warning, and the first-copy-by-video-id walk), the
    // in-queue verdict behind the popup's Add/Remove-from-Queue swap,
    // and the artist-songs merge plan behind the background fetch's
    // enrich-and-append pass. The artist/album popup-view assembly
    // rides with them: the artist-popup gate
    // (_open_actions_for_artist's no-info/one-valid/picker
    // classification), the artist picker's rows, the artist popup's
    // item assembly (the browse-id gate and the {"browseId",
    // "artist"} dict), and the album popup's item assembly (the
    // album-id gate, the album-name resolution and the artists
    // attachment). The popups, the notifications, the
    // awaited workers, the queue mutations and the navigation calls
    // stay Python-side in the mixin.
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::first_artist_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::album_browse_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::queue_entry_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::track_action_route,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::entity_action_route,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::like_toggle_rating,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::remove_from_queue_plan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::in_queue_verdict,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::artist_songs_merge_plan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::artist_popup_plan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::artist_picker_rows,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::artist_popup_item,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        app::track_actions::album_popup_item,
        module
    )?)?;

    // Library-page engines (ui/pages/library.py equivalents): the load
    // view model behind load_playlist's post-fetch walk (the
    // title/owner/privacy/year reads, the normalized track list, the
    // subtitle with its loading suffix — composed through the shared
    // build_playlist_subtitle, read through the formatting module's
    // own name — the unavailable count, and the tail-fetch gate), the
    // Shuffle-lock header button's label/classes pair shared by the
    // load header, the edit-refresh header and the toggle's in-place
    // update, the lock-toggle plan (the new lock state and the
    // force-shuffle gate), and the cursor-restore pair (the
    // playing-track id resolution through the player-then-queue
    // fallback chain, and the row the cursor lands on — the saved nav
    // row in bounds, else the first currently-playing row). The
    // workers, the header mounts, the table loads and the cursor moves
    // stay Python-side in the page.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::library::library_playlist_view,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::library::shuffle_lock_button,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::library::shuffle_lock_toggle_plan,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::library::playing_video_id,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::library::cursor_restore_row,
        module
    )?)?;

    // The liked-songs page's decision cores (the last page without
    // native engines of its own): the empty-state copy
    // (`liked_empty_state` — the load-failed flag picking the failure
    // message that points at the log over the genuine empty-state
    // copy), the footer text (`liked_footer_text` — the count through
    // CPython's own str, the loading-more suffix under the truthiness
    // gate), the restore-cursor bounds gate
    // (`liked_cursor_restore_row` — the saved nav row in the half-open
    // bounds, else None), the nav-state gate (`liked_nav_state` — the
    // row remembered only when not None and > 0) and the Start-Radio
    // seed draw (`radio_seed_sample` — the five-track sample the
    // recently-played page's own Start Radio shares, routed through
    // the Python random module so a seed or a monkeypatched sample
    // steers both backends identically). The workers, the table
    // loads, the tail fetch and the radio fetch stay Python-side in
    // the pages.
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::liked_songs::liked_empty_state,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::liked_songs::liked_footer_text,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::liked_songs::liked_cursor_restore_row,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::liked_songs::liked_nav_state,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        ui::pages::liked_songs::radio_seed_sample,
        module
    )?)?;

    // Discord presence engine (services/discord_rpc.py equivalent):
    // the presence-kwargs assembly behind update's rpc.update(**kwargs)
    // — the seven-key dict in the reference's own key order:
    // activity_type passed through as the very object handed in (the
    // ActivityType.LISTENING member or None, the pypresence import
    // gate staying Python-side), details/state/large_text the
    // `(x or "…")[:128]` arms through the truthiness gate and the
    // value's own __getitem__, large_image the bare unsliced `or`
    // arm, start the interpreter's own int(start_time), and end —
    // appended only when `duration > 0` through CPython's own `>` —
    // the interpreter's own int(start_time + duration). The clock
    // read and the _start_time state write, the pypresence import,
    // the reconnect loop and the RPC call stay Python-side in the
    // service.
    module.add_function(pyo3::wrap_pyfunction!(
        services::discord_rpc::discord_presence_kwargs,
        module
    )?)?;

    // The platform media-integration cores (the four remaining modules
    // with portable pure computations): the MPRIS volume clamp and
    // D-Bus-echo gate (services/mpris.py), the macOS event tap's
    // media-key classification (the data1 bit unpack, the key-down
    // gate and the key-code map, in services/macos_eventtap.py), the
    // macOS Now Playing state choice and microseconds→seconds
    // conversion (services/macos_media.py), and the MPRIS callback
    // unit conversions — the seek/position microseconds→seconds true
    // division and the volume-fraction→percent round (CPython's own
    // ties-to-even) — in app/_mpris.py. The D-Bus / pyobjc / pynput
    // I/O, the Variant construction, the event-tap attribute
    // extraction and the status normalisation stay Python-side in
    // their modules.
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::mpris_clamp_volume,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::mpris_volume_is_echo,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::macos_event_tap_action,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::macos_playback_state,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::macos_now_playing_seconds,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::mpris_us_to_seconds,
        module
    )?)?;
    module.add_function(pyo3::wrap_pyfunction!(
        services::platform_media::mpris_volume_percent,
        module
    )?)?;
    Ok(())
}
