//! Native cores for the sidebars — `ui/sidebars/lyrics_sidebar.py`
//! (LRC parsing, lyric-line conversion and view rows) and
//! `ui/sidebars/playlist_sidebar.py` (panel items, count updates,
//! auto-select matching).

use crate::app::keys::nd_digit_value;
use crate::ui::pages::browse::field_get;
use crate::utils::bidi::is_rtl_char;
use crate::utils::bidi::render_lyric_line_impl;
use crate::utils::doctor::py_splitlines;
use crate::utils::pycompat::py_isspace;
use crate::utils::pycompat::py_str;
use crate::utils::pycompat::python_lower;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString};
use regex::Regex;
use std::sync::LazyLock;

/// The fused view-rows engine behind the lyrics sidebar's two views —
/// the per-line walk `_build_synced_view` and `_build_unsynced_view`
/// assembled through separate `render_lyric_line`, `has_rtl`,
/// `has_non_ascii` and `transliterate_line` calls (two per line, four
/// under the transliteration toggle), folded into one native call for
/// the whole lyric list.
///
/// Each line answers its `(display, is_rtl, translit)` row exactly as
/// the reference's loop composes it: the display is the
/// `render_lyric_line` chain (the gated wrap, then the FSI...PDI
/// isolate — pure-LTR lines pass through unwrapped and unwrapped-in),
/// the RTL flag the `text and has_rtl(text)` verdict (an empty line
/// carries no class), and the transliteration present only when the
/// toggle is on AND the line is non-empty non-ASCII (the
/// `has_non_ascii` gate) — the `anyascii` rendering the
/// `transliterate_line` fallback produces, the same table this crate
/// embeds. A falsy non-string line renders as a blank row where the
/// reference would pass the falsy value through to a blank `Static` —
/// the two view builders only ever produce strings — and a truthy
/// non-string raises the reference's own `TypeError`. Only the cached
/// `bidi_mode` verdict stays Python-side, travelling in as a plain
/// bool read once per rebuild instead of once per line; the widget
/// mounting, the class lists and the auto-scroll stay Python-side in
/// the sidebar.
#[pyfunction]
#[pyo3(signature = (lines, wrap_width, transliteration_enabled, reorder_enabled))]
pub(crate) fn lyrics_view_rows(
    lines: &Bound<'_, PyList>,
    wrap_width: i64,
    transliteration_enabled: bool,
    reorder_enabled: bool,
) -> PyResult<Vec<(String, bool, Option<String>)>> {
    let mut rows: Vec<(String, bool, Option<String>)> = Vec::with_capacity(lines.len());
    for item in lines.iter() {
        // The reference's falsy passthroughs: an empty (or otherwise
        // falsy) line renders blank, carries no RTL class and no
        // transliteration — the single "" spelling covers both the
        // empty string the builders produce and the falsy values they
        // never do.
        if !item.is_truthy()? {
            rows.push((String::new(), false, None));
            continue;
        }
        let text: String = item
            .extract()
            .map_err(|_| PyTypeError::new_err("expected a string line"))?;
        let display = render_lyric_line_impl(&text, wrap_width, reorder_enabled);
        let is_rtl = !text.is_empty() && text.chars().any(is_rtl_char);
        let translit = if transliteration_enabled && !text.is_empty() && !text.is_ascii() {
            Some(any_ascii::any_ascii(&text))
        } else {
            None
        };
        rows.push((display, is_rtl, translit));
    }
    Ok(rows)
}

/// The YTM timed-lyrics line conversion — `lyrics_sidebar`'s
/// `_convert_lyric_lines`, the LyricLine-object walk behind
/// `_process_lyrics`'s synced arm. ytmusicapi's `timestamps=True` shape
/// (`hasTimestamps` plus a `lyrics` list of `LyricLine` dataclass
/// objects — `start_time` in milliseconds, `text`) becomes the
/// `(timestamp_seconds, text)` list the synced view consumes.
///
/// Mirrors the reference's list comprehension exactly. The
/// `hasattr(entry, "start_time")` gate is its own getattr read — the
/// value discarded, only an `AttributeError` answering False (a
/// property raising anything else propagates, like the reference's own
/// `hasattr`); a kept entry's `start_time` is then read again by the
/// expression itself (the reference's second read, so a counting
/// property observes both), and divided through CPython's own true
/// division — an int `start_time` answers the float seconds, a float
/// or bool passes through the same `__truediv__` the reference would
/// call, a non-numeric one raises the division's own `TypeError`. The
/// text is `getattr(entry, "text", "")` — only the missing-attribute
/// `AttributeError` swallowed to the `""` default, any other error
/// surfacing, the value passing through as the very object it is
/// (never coerced; the view builders' own gates handle the odd shapes
/// exactly as the reference's would). The `(seconds, text)` pairs land
/// in a fresh list in iteration order — no sort, unlike the LRC parse
/// feeding the same view. The `hasTimestamps` gate and the
/// `isinstance(..., list)` narrowing stay Python-side in the sidebar,
/// as does the empty-conversion fallback to the string-lyrics path.
#[pyfunction]
#[pyo3(signature = (entries))]
pub(crate) fn convert_lyric_lines<'py>(
    py: Python<'py>,
    entries: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    // 1000.0 — the reference's float literal, the right operand of the
    // expression's true division.
    let thousand: Bound<'py, PyAny> = 1000.0f64
        .into_pyobject(py)
        .expect("float conversion is infallible")
        .into_any();
    let out = PyList::empty(py);
    for entry in entries.iter() {
        // hasattr(entry, "start_time") — the gate's own read, its value
        // discarded; only an AttributeError answers False.
        match entry.getattr("start_time") {
            Ok(_) => {}
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => continue,
            Err(err) => return Err(err),
        }
        // entry.start_time / 1000.0 — the expression's own read (the
        // reference's second), then the interpreter's true division.
        let start_time = entry.getattr("start_time")?;
        let seconds = start_time.div(&thousand)?;
        // getattr(entry, "text", "") — the AttributeError fallback arm.
        let text = match entry.getattr("text") {
            Ok(text) => text,
            Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                "".into_pyobject(py)?.into_any()
            }
            Err(err) => return Err(err),
        };
        out.append((seconds, text))?;
    }
    Ok(out)
}

/// Python's `value in ("a", "b")`: left-to-right `==` comparisons against
/// the strings, with the result taken by truthiness — so str subclasses
/// and `__eq__`-overriding objects classify exactly like the reference.
/// An exact `str` takes the contents fast path (`str.__eq__` is C-level
/// equality, no dispatch).
pub(crate) fn eq_any_str<'py>(value: &Bound<'py, PyAny>, options: &[&str]) -> PyResult<bool> {
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

/// Python's `str.strip()`: trims the CPython `isspace` set (see
/// `py_isspace`) from both ends.
pub(crate) fn python_strip(s: &str) -> &str {
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
pub(crate) fn format_panel_item(py: Python, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
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
pub(crate) fn panel_count_label(shown: i64, total: i64) -> String {
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
pub(crate) fn filter_panel_items(
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
pub(crate) fn matches_removal(
    py: Python,
    item: &Bound<'_, PyAny>,
    playlist_id: &str,
) -> PyResult<bool> {
    let pid = pid_or_browse(py, item, Some(""))?;
    let vl = format!("VL{playlist_id}");
    eq_any_str(&pid, &[playlist_id, vl.as_str()])
}

/// `LibraryPanel.update_item_count`'s predicate: the item's playlist id
/// (`playlistId` or `browseId` or `""`) equals the target, its
/// `VL`-prefixed form, its `VL`-stripped form, or prefixes the target —
/// tolerant in **both** `VL` directions, unlike the removal predicate.
#[pyfunction]
pub(crate) fn matches_count_update(
    py: Python,
    item: &Bound<'_, PyAny>,
    playlist_id: &str,
) -> PyResult<bool> {
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
pub(crate) fn bump_playlist_count(
    py: Python,
    current: &Bound<'_, PyAny>,
    delta: i64,
) -> PyResult<Py<PyAny>> {
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
pub(crate) fn selection_title(py: Python, item: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let empty = PyString::new(py, "").into_any();
    let name = field_get(py, item, "name", Some(empty))?;
    Ok(field_get(py, item, "title", Some(name))?.unbind())
}

/// Whether the item is the account's auto-generated Liked Music playlist
/// (id `LM`) — the row `ensure_loaded` filters out because it is already
/// a pinned nav item.
#[pyfunction]
pub(crate) fn is_liked_music_playlist(py: Python, item: &Bound<'_, PyAny>) -> PyResult<bool> {
    // (item.get("playlistId") or item.get("browseId", "")) == "LM"
    let pid = pid_or_browse(py, item, Some(""))?;
    eq_any_str(&pid, &["LM"])
}

/// `PlaylistSidebar.auto_select_playlist`'s predicate: the item's playlist
/// id (`playlistId` or `browseId`, no `""` default — a missing `browseId`
/// reads as `None`) equals the target exactly, with no `VL` tolerance.
#[pyfunction]
pub(crate) fn matches_auto_select(
    py: Python,
    item: &Bound<'_, PyAny>,
    playlist_id: &str,
) -> PyResult<bool> {
    let pid = pid_or_browse(py, item, None)?;
    eq_any_str(&pid, &[playlist_id])
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
pub(crate) fn parse_synced_lyrics(raw: &str) -> Vec<(f64, String)> {
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
