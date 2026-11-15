//! Native cores for `src/ytm_player/utils/transliteration.py` —
//! any-ascii transliteration and the non-ASCII probe.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Transliteration (mirrors ytm_player/utils/transliteration.py)
// ---------------------------------------------------------------------------

/// Transliterate non-ASCII text to ASCII via the anyascii Rust crate.
/// Mirrors `transliteration.transliterate_line` (minus its Python-side
/// ASCII fast path and lru_cache, which are kept in Python).
#[pyfunction]
pub(crate) fn transliterate(text: &str) -> String {
    any_ascii::any_ascii(text)
}

/// Return True if *text* is non-empty and contains any non-ASCII characters.
/// Mirrors `ytm_player.utils.transliteration.has_non_ascii`.
#[pyfunction]
pub(crate) fn has_non_ascii(text: &str) -> bool {
    !text.is_empty() && !text.is_ascii()
}
