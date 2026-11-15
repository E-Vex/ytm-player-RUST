//! Native cores for `src/ytm_player/ui/playback_bar.py` — track-info
//! widths/snippets and the bar cell renderer.

use crate::utils::bidi::gated_reorder;
use crate::utils::bidi::isolate_bidi_impl;
use crate::utils::formatting::truncate_str;
use pyo3::prelude::*;

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
/// exactly like the reference computes them. The Rich `Text` assembly
/// and the theme styling stay Python-side in the widget — the fused
/// snippet engine below now runs this split inside its own walk.
#[pyfunction]
#[pyo3(signature = (bar_width, title, artist))]
pub(crate) fn track_info_widths(bar_width: i64, title: &str, artist: &str) -> (i64, i64, i64) {
    track_info_widths_impl(bar_width, title, artist)
}

/// The width-splitting core shared by the `track_info_widths` engine
/// and the fused snippet assembly below — the plain arithmetic the
/// public engine answers and the snippet engine runs in-place for its
/// own fragment widths.
fn track_info_widths_impl(bar_width: i64, title: &str, artist: &str) -> (i64, i64, i64) {
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

/// One playback-bar snippet — the `(kind, text)` pair the widget styles
/// by kind: the leading state icon (`"playing"` / `"paused"` /
/// `"stopped"`), the title/artist/album fragments (`"title"` /
/// `"artist"` / `"album"`), the em-dash `"sep"` separators between
/// them, and the `"empty"` no-track placeholder.
type TrackInfoSnippet = (&'static str, String);

/// The em-dash separator the reference appends between the fragments —
/// `" \u{2014} "` (one space, em dash, one space).
const TRACK_INFO_SEP: &str = " \u{2014} ";

/// The fused snippet engine behind the playback bar's `_TrackInfo`
/// render — `ui/playback_bar.py`'s `_track_info_snippets`, the content
/// walk the render used to assemble snippet by snippet.
///
/// Mirrors the reference's walk exactly and in its own order: the
/// three-way state-icon branch first (`is_playing and not is_paused` →
/// the play triangle, `elif is_paused` → the pause bars, else the
/// stopped square — each wrapped in the reference's single spaces),
/// then the `if title:` fork. A falsy title walks the `"empty"` arm —
/// the `"No track playing"` placeholder and nothing else, the
/// artist/album never consulted. A truthy title splits the fragment
/// widths through the width-splitting engine's own core
/// (`track_info_widths_impl` — the lens on the ORIGINAL strings, before
/// any reorder or truncation), renders the title through the
/// `render_bar_cell` chain (the gated reorder, the truncation of the
/// REORDERED fragment, the isolate wrap — `width < 1` answering the
/// empty string), and then, each behind its own truthiness gate, the
/// artist and the album: the em-dash separator FIRST and the fragment
/// second, so an absent artist or album skips its separator too. The
/// kind tags travel with the texts; the Rich `Text` appends and the
/// theme colours each kind is styled with stay Python-side in the
/// widget. Only the cached `bidi_mode` verdict stays Python-side as
/// well, travelling in as a plain bool (read once per render instead
/// of once per fragment).
#[pyfunction]
#[pyo3(signature = (bar_width, title, artist, album, is_playing, is_paused, reorder_enabled))]
pub(crate) fn track_info_snippets(
    bar_width: i64,
    title: &str,
    artist: &str,
    album: &str,
    is_playing: bool,
    is_paused: bool,
    reorder_enabled: bool,
) -> Vec<TrackInfoSnippet> {
    let mut snippets: Vec<TrackInfoSnippet> = Vec::with_capacity(6);
    // The state icon — the reference's three-way branch, its own
    // order: playing wins only while not also paused; a paused flag
    // alone is enough for the paused icon.
    if is_playing && !is_paused {
        snippets.push(("playing", " \u{25b6} ".to_string()));
    } else if is_paused {
        snippets.push(("paused", " \u{23f8} ".to_string()));
    } else {
        snippets.push(("stopped", " \u{25a0} ".to_string()));
    }
    // if title — the truthiness gate over the plain str the reactive
    // feeds the render.
    if !title.is_empty() {
        let (title_w, artist_w, album_w) = track_info_widths_impl(bar_width, title, artist);
        snippets.push(("title", bar_cell(title, title_w as isize, reorder_enabled)));
        if !artist.is_empty() {
            snippets.push(("sep", TRACK_INFO_SEP.to_string()));
            snippets.push((
                "artist",
                bar_cell(artist, artist_w as isize, reorder_enabled),
            ));
        }
        if !album.is_empty() {
            snippets.push(("sep", TRACK_INFO_SEP.to_string()));
            snippets.push(("album", bar_cell(album, album_w as isize, reorder_enabled)));
        }
    } else {
        snippets.push(("empty", "No track playing".to_string()));
    }
    snippets
}

/// The `render_bar_cell` chain (the gated line reorder, the truncation
/// of the reordered fragment, the isolate wrap) with the wrapper's
/// falsy passthrough folded in, so the fused snippet engine renders
/// each fragment exactly like the per-cell engine the widget's render
/// used to reach for. Only the cached `bidi_mode` verdict stays
/// Python-side, travelling in as a plain bool.
fn bar_cell(text: &str, width: isize, reorder_enabled: bool) -> String {
    if text.is_empty() {
        text.to_string()
    } else {
        isolate_bidi_impl(
            &truncate_str(&gated_reorder(text, reorder_enabled), width),
            true,
        )
    }
}
