//! Native cores for `src/ytm_player/ui/selection_info_bar.py` — the
//! selection info bar's visible-width computation.

use pyo3::prelude::*;

/// The render width behind `SelectionInfoBar`'s text truncation — the
/// pure core of `watch_text` from `ui/selection_info_bar.py`: a zero
/// widget width (the bar not yet laid out) reads the 80-column
/// default; any other width reserves the bar's 1-column padding on
/// each side but never fewer than 10 columns (`max(width - 2, 10)`,
/// CPython's own two-argument `max` — a negative width, truthy like
/// every nonzero int, keeps the 10 floor). The truncation itself is
/// the already-ported formatting engine; the Static update and the
/// untruncated fallback path stay Python-side in the bar.
#[pyfunction]
pub(crate) fn selection_info_visible_width(width: i64) -> i64 {
    // `max(width - 2, 10) if width else 80` — the truthiness gate is
    // `width != 0` for the int the widget width is.
    if width != 0 {
        if width - 2 > 10 {
            width - 2
        } else {
            10
        }
    } else {
        80
    }
}
