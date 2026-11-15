//! Native cores for `src/ytm_player/ui/header_bar.py` — click routes and
//! the lyrics pill class.

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// The header-bar / selection-info-bar / track-filter cores
// ---------------------------------------------------------------------------

/// The click routing behind `HeaderBar`'s pointer routing — the pure
/// core of `on_click` from `ui/header_bar.py`: the clicked widget's id
/// maps onto one of the four toggle routes — `"toggle-playlist"` →
/// `"playlist"`, `"toggle-lyrics"` → `"lyrics"`, `"back-btn"` →
/// `"back"`, `"forward-btn"` → `"forward"` — and any other id, or a
/// widget with no id at all (`Widget.id` is `str | None`; `None ==
/// "toggle-playlist"` is `False`, CPython's own `==` on the None
/// literal — the confirm-popup verdict precedent), routes `"none"`.
/// The message construction, the event stopping and the widget-object
/// None gate stay Python-side in the bar.
#[pyfunction]
#[pyo3(signature = (target_id))]
pub(crate) fn header_bar_click_route(target_id: Option<&str>) -> &'static str {
    match target_id {
        Some("toggle-playlist") => "playlist",
        Some("toggle-lyrics") => "lyrics",
        Some("back-btn") => "back",
        Some("forward-btn") => "forward",
        _ => "none",
    }
}

/// The class choice behind `HeaderBar`'s lyrics-toggle render — the
/// pure core of `_apply_lyrics_classes` from `ui/header_bar.py`:
/// `"dimmed"` when the lyrics pane is dimmed (open without a synced
/// payload), `"active"` when it is open and lit, `None` when it is
/// closed — the caller removes both classes first and then adds the
/// answer, exactly the reference's remove-then-add. The DOM reads and
/// writes stay Python-side in the bar.
#[pyfunction]
#[pyo3(signature = (is_dimmed, is_on))]
pub(crate) fn header_bar_lyrics_class(is_dimmed: bool, is_on: bool) -> Option<&'static str> {
    if is_dimmed {
        Some("dimmed")
    } else if is_on {
        Some("active")
    } else {
        None
    }
}
