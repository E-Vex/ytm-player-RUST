//! Native cores for `src/ytm_player/ui/widgets/album_art.py` — WCAG
//! contrast, half-block cell assembly, placeholder rows and the
//! `ArtCacheCore` bridge class.

use crate::services::ytmusic::py_key_position;
use crate::ui::popups::py_str_strip;
use crate::ui::widgets::progress_bar::py_max2;
use crate::ui::widgets::progress_bar::py_min2;
use pyo3::prelude::*;

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
pub(crate) fn wcag_relative_luminance(color: &str) -> Option<f64> {
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
pub(crate) fn wcag_contrast_ratio(luminance_a: f64, luminance_b: f64) -> f64 {
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
pub(crate) fn wcag_readable_on(bg_hex: &str, candidates: Vec<String>) -> String {
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
pub(crate) fn album_art_half_block_cells(
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
pub(crate) fn album_art_placeholder_rows(
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
pub(crate) fn album_art_empty_rows(
    width: i64,
    height: i64,
    muted: &str,
) -> Vec<Vec<(String, String)>> {
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
pub(crate) struct ArtCacheCore {
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
