//! Native cores for `src/ytm_player/ui/playback_bar.py` — track-info
//! widths/snippets, the bar cell renderer, and the chrome engines (the
//! volume/repeat/shuffle/heart/footer-button render mappings, the
//! footer click routing, the active-page verdict and the bar's own
//! right-click track pick).

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

// ── The playback-bar chrome engines ─────────────────────────────────────
//
// The render mappings behind the bar's small always-visible widgets —
// the volume display, the repeat and shuffle indicators, the like
// heart and the footer buttons — plus the two click decisions (the
// footer's action routing and the bar's own right-click track pick).
// Every engine answers the content/kind pair the widget styles; the
// Rich `Text` assembly, the theme reads and the DOM/event plumbing
// stay Python-side in the widgets.

/// The volume display's content — the pure core of `_VolumeDisplay`'s
/// render from `ui/playback_bar.py`: the speaker glyph, the volume
/// right-aligned into three columns, and the percent sign.
///
/// Mirrors the reference's f-string exactly: `{volume:>3}` is
/// CPython's own `format(volume, ">3")` — the very builtin call the
/// f-string field compiles down to, called here on the very object
/// the reactive holds — so every formattable shape answers the
/// interpreter's own rendering (ints padded to 3 columns, wider
/// numbers left un-padded, floats keeping their own digits), and an
/// object without a `__format__` raises its own `TypeError`. The
/// widget keeps only the secondary colour.
#[pyfunction]
#[pyo3(signature = (volume))]
pub(crate) fn volume_display_text<'py>(
    py: Python<'py>,
    volume: &Bound<'py, PyAny>,
) -> PyResult<String> {
    // format(volume, ">3") — through the builtins `format` so the
    // conversion is the interpreter's own.
    let formatted: String = py
        .import("builtins")?
        .getattr("format")?
        .call1((volume, ">3"))?
        .extract()?;
    let mut text = String::from(" \u{1f50a} ");
    text.push_str(&formatted);
    text.push('%');
    Ok(text)
}

/// The repeat button's content pair — the pure core of `_RepeatButton`'s
/// render from `ui/playback_bar.py`: which of the three mode pairs the
/// indicator shows.
///
/// Mirrors the reference's own comparisons — `mode == RepeatMode.ALL`
/// then `mode == RepeatMode.ONE`, CPython's own `==` against the very
/// enum members the module exports (read from `ytm_player.services.queue`
/// at call time, so a patched member steers both backends identically —
/// the order-pairs precedent) — answering `("all", "🔁 all")`,
/// `("one", "🔂 one")` or `("off", "🔁 off")`; the kind carries the
/// style the widget applies. The Rich `Text` assembly and the theme
/// colours stay Python-side in the button.
#[pyfunction]
#[pyo3(signature = (mode))]
pub(crate) fn repeat_button_pair<'py>(
    py: Python<'py>,
    mode: &Bound<'py, PyAny>,
) -> PyResult<(&'static str, &'static str)> {
    let repeat_mode = py
        .import("ytm_player.services.queue")?
        .getattr("RepeatMode")?;
    let member_all = repeat_mode.getattr("ALL")?;
    let member_one = repeat_mode.getattr("ONE")?;
    if mode.eq(&member_all)? {
        Ok(("all", "\u{1f501} all"))
    } else if mode.eq(&member_one)? {
        Ok(("one", "\u{1f502} one"))
    } else {
        Ok(("off", "\u{1f501} off"))
    }
}

/// The shuffle button's content pair — the pure core of `_ShuffleButton`'s
/// render from `ui/playback_bar.py`: the on pair (trailing space and
/// all) or the off pair, picked by the flag's own truthiness — the
/// reference's `if self.shuffle_on:` — so every falsy shape answers
/// the off pair exactly like the reference. The Rich `Text` assembly
/// and the theme colours stay Python-side in the button.
#[pyfunction]
#[pyo3(signature = (shuffle_on))]
pub(crate) fn shuffle_button_pair<'py>(
    shuffle_on: &Bound<'py, PyAny>,
) -> PyResult<(&'static str, &'static str)> {
    if shuffle_on.is_truthy()? {
        Ok(("on", "\u{1f500} on "))
    } else {
        Ok(("off", "\u{1f500} off"))
    }
}

/// The heart button's content pair — the pure core of `_HeartButton`'s
/// render from `ui/playback_bar.py`: the like verdict and the glyph.
///
/// Mirrors the reference's own comparison — `like_status == "LIKE"`,
/// CPython's own `==` on the string literal, so a non-str status (or a
/// lowercase "like") answers the unliked pair exactly like the
/// reference — answering `("liked", " ❤ ")` or `("unliked", " ❤ ")`:
/// the glyph text is the same on both arms; the kind carries the bold
/// accent vs muted colour. The Rich `Text` assembly stays Python-side
/// in the button.
#[pyfunction]
#[pyo3(signature = (like_status))]
pub(crate) fn heart_button_pair<'py>(
    like_status: &Bound<'py, PyAny>,
) -> PyResult<(&'static str, &'static str)> {
    if like_status.eq("LIKE")? {
        Ok(("liked", " \u{2764} "))
    } else {
        Ok(("unliked", " \u{2764} "))
    }
}

/// The footer button's style kind — the pure core of `_FooterButton`'s
/// render from `ui/playback_bar.py`: the three-way style choice over
/// the two flags, the reference's own order — an active button renders
/// bold primary however dimmed, a dimmed one renders dim, anything
/// else muted — every gate the flag's own truthiness, the reference's
/// own `if self.is_active:` / `if self.is_dimmed:`. The label is the
/// caller's own; the Rich `Text` assembly and the theme colours stay
/// Python-side in the button.
#[pyfunction]
#[pyo3(signature = (is_active, is_dimmed))]
pub(crate) fn footer_button_style_kind<'py>(
    is_active: &Bound<'py, PyAny>,
    is_dimmed: &Bound<'py, PyAny>,
) -> PyResult<&'static str> {
    if is_active.is_truthy()? {
        Ok("active")
    } else if is_dimmed.is_truthy()? {
        Ok("dimmed")
    } else {
        Ok("muted")
    }
}

/// The footer button's action routing — the pure core of
/// `_FooterButton.on_click` from `ui/playback_bar.py`: the `match`
/// statement's verdict, its own case order.
///
/// Mirrors the reference's match exactly: the seven page actions
/// (`help` / `library` / `search` / `queue` / `browse` /
/// `liked_songs` / `recently_played`) route to the caller's page
/// navigation — the caller passes its own action on as the page name —
/// `play_pause`, `prev` and `next` route to the transport handlers,
/// `spotify_import` to the import popup, and any other action routes
/// `none` (the reference's match falls through and nothing runs).
/// Every comparison is CPython's own `==` on the string literal — the
/// protocol a match statement's literal patterns evaluate through. The
/// `await`ed handlers, the screen push and the event stopping stay
/// Python-side in the button.
#[pyfunction]
#[pyo3(signature = (action))]
pub(crate) fn footer_button_action_route<'py>(
    action: &Bound<'py, PyAny>,
) -> PyResult<&'static str> {
    const PAGES: [&str; 7] = [
        "help",
        "library",
        "search",
        "queue",
        "browse",
        "liked_songs",
        "recently_played",
    ];
    for page in PAGES {
        if action.eq(page)? {
            return Ok("page");
        }
    }
    if action.eq("play_pause")? {
        return Ok("play_pause");
    }
    if action.eq("prev")? {
        return Ok("prev");
    }
    if action.eq("next")? {
        return Ok("next");
    }
    if action.eq("spotify_import")? {
        return Ok("spotify_import");
    }
    Ok("none")
}

/// The footer button's active-page verdict — the pure core of
/// `FooterBar.set_active_page` from `ui/playback_bar.py`: whether the
/// button the loop is visiting names the page that just became
/// active — `action == page_name`, CPython's own `==` on the two
/// names, so a `None` or non-str page name answers `False` exactly
/// like the reference. The `query_one` walk and the per-button
/// `is_active` write stay Python-side in the bar.
#[pyfunction]
#[pyo3(signature = (action, page_name))]
pub(crate) fn footer_button_active_verdict<'py>(
    action: &Bound<'py, PyAny>,
    page_name: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    action.eq(page_name)
}

/// The playback bar's right-click track pick — the pure core of
/// `PlaybackBar.on_click` from `ui/playback_bar.py`: the track whose
/// actions a right-click on the bar opens.
///
/// Mirrors the reference exactly, every gate CPython's own protocol:
/// the right-button gate first — `event.button != 3`, CPython's own
/// `!=`, a non-3 button answering `None` — then the player-then-queue
/// fallback chain — a truthy player track answers the very object the
/// reference posts (only then does a truthy queue track answer its
/// own), and every falsy shape answers `None`, the no-message arm the
/// reference's `if track:` gate leaves. The player/queue attribute
/// reads and the message construction stay Python-side in the bar.
#[pyfunction]
#[pyo3(signature = (button, player_track, queue_track))]
pub(crate) fn playback_bar_click_track<'py>(
    button: &Bound<'py, PyAny>,
    player_track: &Bound<'py, PyAny>,
    queue_track: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    // if event.button != 3: return — the gate, CPython's own !=.
    if button.ne(3i64)? {
        return Ok(None);
    }
    if player_track.is_truthy()? {
        Ok(Some(player_track.clone()))
    } else if queue_track.is_truthy()? {
        Ok(Some(queue_track.clone()))
    } else {
        Ok(None)
    }
}
