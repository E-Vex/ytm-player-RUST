//! Native cores for `src/ytm_player/utils/bidi.py` — BiDi
//! isolation/reordering/wrapping plus the track/bar/lyric cell
//! renderers.

use crate::utils::formatting::truncate_str;
use pyo3::prelude::*;
use unicode_bidi::bidi_class;

// ---------------------------------------------------------------------------
// BiDi text utilities (mirrors ytm_player/utils/bidi.py)
// ---------------------------------------------------------------------------

/// Unicode directional isolate marks (UAX #9, since Unicode 6.3).
const FSI: char = '\u{2068}'; // FIRST STRONG ISOLATE

const PDI: char = '\u{2069}'; // POP DIRECTIONAL ISOLATE

/// Matches any character from RTL scripts (Arabic, Hebrew, Thaana, Syriac,
/// N'Ko).  Mirrors the `_RTL_RE` range set in `bidi.py` exactly.
pub(crate) fn is_rtl_char(c: char) -> bool {
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
pub(crate) fn has_rtl(text: &str) -> bool {
    text.chars().any(is_rtl_char)
}

/// Wrap *text* in FSI...PDI so its directional context cannot leak.
///
/// Matches `bidi.isolate_bidi`: skips empty text, skips pure-LTR text when
/// *only_if_rtl* is set, and never double-wraps.
#[pyfunction(signature = (text, *, only_if_rtl = true))]
pub(crate) fn isolate_bidi(text: &str, only_if_rtl: bool) -> String {
    isolate_bidi_impl(text, only_if_rtl)
}

/// The isolate wrap shared by the `isolate_bidi` export and the fused
/// per-cell render engines — one definition so the export and every
/// render chain isolates through exactly the same body: the empty and
/// pure-LTR (when *only_if_rtl*) passthroughs, the FSI...PDI
/// double-wrap guard, and the wrap itself.
pub(crate) fn isolate_bidi_impl(text: &str, only_if_rtl: bool) -> String {
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
pub(crate) fn reorder_line(text: &str) -> String {
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
pub(crate) fn wrap_line(text: &str, width: i64) -> String {
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
pub(crate) fn gated_reorder(text: &str, reorder_enabled: bool) -> String {
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
pub(crate) fn reorder_rtl_line(text: &str, reorder_enabled: bool) -> String {
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
pub(crate) fn wrap_rtl_line(text: &str, width: i64, reorder_enabled: bool) -> String {
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
pub(crate) fn render_track_cell(text: &str, reorder_enabled: bool) -> String {
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
pub(crate) fn render_bar_cell(text: &str, width: isize, reorder_enabled: bool) -> String {
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
pub(crate) fn render_lyric_line(text: &str, width: i64, reorder_enabled: bool) -> String {
    render_lyric_line_impl(text, width, reorder_enabled)
}

/// The render-lyric-line body shared by the single-line export and the
/// fused view-rows engine below — one definition so the per-line call
/// and the whole-view walk render through exactly the same pipeline:
/// the gated multi-line pre-wrap (the disabled / empty / no-RTL
/// passthroughs, then the wrap-then-reorder pipeline of
/// `wrap_line_impl` — the width short-circuit, the word walk, the
/// per-line reorder join), then the isolate wrap around the WHOLE
/// wrapped string — one FSI...PDI pair bracketing every wrapped line.
pub(crate) fn render_lyric_line_impl(text: &str, width: i64, reorder_enabled: bool) -> String {
    let wrapped = if !reorder_enabled || text.is_empty() || !text.chars().any(is_rtl_char) {
        text.to_string()
    } else {
        wrap_line_impl(text, width)
    };
    isolate_bidi_impl(&wrapped, true)
}
