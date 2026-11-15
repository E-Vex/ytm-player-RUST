//! Native cores for `src/ytm_player/ui/track_filter.py` — input
//! ownership and the escape-clears rule.

use pyo3::prelude::*;

/// The input-ownership gate behind the filter wiring — the pure core
/// of `on_input_changed` / `on_input_submitted` from
/// `ui/track_filter.py`: only the shared `#track-filter` Input's
/// events are the filter's (`Input.id` is `str | None`; a widget with
/// no id, or any other id, is not ours — the confirm-popup verdict
/// precedent). The Input reads and writes, the table calls and the
/// focus moves stay Python-side in the host page.
#[pyfunction]
#[pyo3(signature = (input_id))]
pub(crate) fn track_filter_owns_input(input_id: Option<&str>) -> bool {
    input_id == Some("track-filter")
}

/// The escape verdict behind the filter's key handler — the pure core
/// of `on_key` from `ui/track_filter.py`: Escape clears the filter
/// (and hides the input, refocuses the table) exactly when the key is
/// `"escape"` and the filter input is still visible; every other key,
/// or an Escape after Enter hid the input (the keep-filter path),
/// leaves the page alone. The event stopping, the clear, the class
/// removal and the refocus stay Python-side in the host page.
#[pyfunction]
pub(crate) fn track_filter_escape_clears(key: &str, filter_visible: bool) -> bool {
    key == "escape" && filter_visible
}
