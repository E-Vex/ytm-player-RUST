//! Cross-module CPython/PyO3 semantics helpers — `str.lower`/isspace
//! semantics and `str()` coercion. No direct Python counterpart;
//! consumed throughout the crate.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// CPython string semantics
// ---------------------------------------------------------------------------

/// CPython's `str.isspace()` acceptance set for a single character.
/// Rust's `char::is_whitespace()` (the Unicode `White_Space` property)
/// would disagree on `\x1c`–`\x1f`, which CPython accepts, so the set is
/// spelled out instead of delegating.
pub(crate) fn py_isspace(c: char) -> bool {
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

/// `str(obj)` for label interpolation — the f-string conversion the
/// reference applies to title/year/subscribers/author/count values.
pub(crate) fn py_str(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.str()?.to_string_lossy().into_owned())
}

// ---------------------------------------------------------------------------
// CPython case mapping
// ---------------------------------------------------------------------------

/// Python's `str.lower()`: the Unicode section-3.13 default lowercase
/// mapping, which lowercases the Greek capital sigma contextually — a
/// word-final `Σ` becomes `ς`, a mid-word one `σ`. Rust's
/// `str::to_lowercase` implements the same mapping (including the final
/// sigma special case), so it agrees with CPython case for case; the
/// per-code-point `char::to_lowercase` does not (it never applies the
/// context rule) and must not be used here.
pub(crate) fn python_lower(s: &str) -> String {
    s.to_lowercase()
}
