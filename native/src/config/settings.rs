//! Native cores for `src/ytm_player/config/settings.py` — TOML value
//! formatting for config file writes.

use crate::json::json_push_u_escape;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyString};

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
pub(crate) fn format_toml_value<'py>(
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
