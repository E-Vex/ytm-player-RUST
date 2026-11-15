//! Native cores for `src/ytm_player/app/_keys.py` — the key feed, count
//! prefix and dispatch engines (incl. navigation routing and the
//! shuffle-lock gate).

use crate::app::sidebar::PANE_LYRICS;
use crate::app::sidebar::PANE_PLAYLISTS;
use crate::utils::pycompat::py_isspace;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

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
pub(crate) fn normalize_key(key: &str, character: Option<&str>, is_printable: bool) -> String {
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
pub(crate) fn nd_digit_value(c: char) -> Option<u32> {
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
pub(crate) fn py_isdigit(s: &str) -> bool {
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
pub(crate) fn feed_key<'py>(
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
pub(crate) fn unpack_too_many_message<'py>(
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
pub(crate) fn unpack_dispatch_row<'py>(
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
pub(crate) fn dispatch_route<'py>(
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
pub(crate) fn route_navigation_action<'py>(
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
pub(crate) fn current_context_target<'py>(
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
pub(crate) fn shuffle_lock_blocked(
    context_id: &Bound<'_, PyAny>,
    prefs: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !context_id.is_truthy()? {
        return Ok(false);
    }
    let value = prefs.call_method1("get", (context_id,))?;
    value.is_truthy()
}
