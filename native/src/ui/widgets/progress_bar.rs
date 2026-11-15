//! Native cores for `src/ytm_player/ui/widgets/progress_bar.py` —
//! progress metrics/cells and x-to-seconds mapping.

use crate::utils::formatting::format_duration;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Progress-bar render geometry
// ---------------------------------------------------------------------------

/// Python's two-argument `max(a, b)`: `b` displaces `a` only when
/// strictly greater, so a NaN operand — every comparison false — keeps
/// `a`, exactly CPython's `max(0.0, float("nan")) == 0.0`.
pub(crate) fn py_max2(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's two-argument `min(a, b)`: the `py_max2` mirror.
pub(crate) fn py_min2(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python's `int()` on the float shapes the bar geometry multiplies out:
/// truncation toward zero (`int(-0.9) == 0`, `int(2.7) == 2`). A NaN —
/// which the reference's own `int()` would raise on, and which mpv's
/// time-pos never produces — reads as 0 so the engine stays total;
/// magnitudes beyond i64 clamp by sign, far past any terminal width or
/// track length.
fn py_int_trunc(f: f64) -> i64 {
    if f.is_nan() {
        0
    } else {
        f as i64
    }
}

/// The progress fraction behind `PlaybackProgress.progress` — the pure
/// core of the property from `ui/widgets/progress_bar.py`: `0.0` for a
/// non-positive duration, otherwise `position / duration` clamped to
/// `[0.0, 1.0]` through CPython's own two-argument `min`/`max` bounds
/// (see `py_min2`/`py_max2`; a NaN duration fails the `<= 0` gate on
/// both backends and flows into the clamp, where the bounds keep it at
/// `0.0` — CPython's answer too).
#[pyfunction]
pub(crate) fn progress_fraction(position: f64, duration: f64) -> f64 {
    if duration <= 0.0 {
        return 0.0;
    }
    py_min2(1.0, py_max2(0.0, position / duration))
}

/// The metrics behind `PlaybackProgress`'s bar layout — the pure core of
/// the widget's `_bar_metrics` from `ui/widgets/progress_bar.py`: the
/// elapsed/total time labels (`format_duration(int(...))`, the pending
/// scroll-seek preview standing in for the position) and the bar width
/// left over once both labels' code-point lengths are reserved from the
/// widget width, floored at 0.
#[pyfunction]
#[pyo3(signature = (position, duration, preview, width))]
pub(crate) fn progress_bar_metrics(
    position: f64,
    duration: f64,
    preview: Option<f64>,
    width: i64,
) -> (String, String, i64) {
    // pos = self.position if self._preview_position is None else
    // self._preview_position
    let pos = preview.unwrap_or(position);
    let time_prefix = format!(" {} ", format_duration(py_int_trunc(pos)));
    let time_suffix = format!(" {} ", format_duration(py_int_trunc(duration)));
    // len() counts code points; format_duration's renders are ASCII-only,
    // so chars().count() is CPython's len() exactly.
    let reserved = (time_prefix.chars().count() + time_suffix.chars().count()) as i64;
    (time_prefix, time_suffix, (width - reserved).max(0))
}

/// The cell computation behind `PlaybackProgress.render` — the pure core
/// of the bar composition from `ui/widgets/progress_bar.py`: the filled
/// and empty cell counts (`int(bar_width * progress)`, truncation toward
/// zero) and the scroll-seek preview's marker column — `-1` when no
/// preview is pending or the bar has no width, otherwise
/// `min(int(bar_width * preview_frac), bar_width - 1)` with
/// `preview_frac` the preview position clamped against the duration
/// exactly like the reference (the `duration > 0` ternary, then
/// CPython's two-argument `min`/`max` bounds).
#[pyfunction]
#[pyo3(signature = (bar_width, progress, preview, duration))]
pub(crate) fn progress_bar_cells(
    bar_width: i64,
    progress: f64,
    preview: Option<f64>,
    duration: f64,
) -> (i64, i64, i64) {
    let filled_count = py_int_trunc(bar_width as f64 * progress);
    let empty_count = bar_width - filled_count;
    let marker_col = match preview {
        // if self._preview_position is not None and bar_width > 0
        Some(preview_position) if bar_width > 0 => {
            // preview_frac = min(1.0, max(0.0, preview / duration))
            //                if duration > 0 else 0.0
            let preview_frac = if duration > 0.0 {
                py_min2(1.0, py_max2(0.0, preview_position / duration))
            } else {
                0.0
            };
            py_int_trunc(bar_width as f64 * preview_frac).min(bar_width - 1)
        }
        _ => -1,
    };
    (filled_count, empty_count, marker_col)
}

/// The click-to-seek mapping behind `PlaybackProgress`'s click handling —
/// the pure core of the widget's `_x_to_seconds` from
/// `ui/widgets/progress_bar.py`: the clicked column, translated through
/// the same metrics the render uses (the time prefix eats the leading
/// columns), clamped to `[0, bar_width - 1]` and scaled to
/// `fraction * duration`. `None` — no seek — for a non-positive duration
/// or a bar with no width, exactly the reference's gates.
#[pyfunction]
#[pyo3(signature = (x, position, duration, preview, width))]
pub(crate) fn progress_x_to_seconds(
    x: i64,
    position: f64,
    duration: f64,
    preview: Option<f64>,
    width: i64,
) -> Option<f64> {
    // if self.duration <= 0: return None
    if duration <= 0.0 {
        return None;
    }
    let (time_prefix, _time_suffix, bar_width) =
        progress_bar_metrics(position, duration, preview, width);
    // if bar_width <= 0: return None
    if bar_width <= 0 {
        return None;
    }
    // prefix_len = len(f" {format_duration(int(pos))} ") — the prefix's
    // own code-point length.
    let prefix_len = time_prefix.chars().count() as i64;
    let mut bar_x = x - prefix_len;
    if bar_x < 0 {
        bar_x = 0;
    }
    if bar_x >= bar_width {
        bar_x = bar_width - 1;
    }
    // fraction = bar_x / bar_width — true division, then the scale.
    Some((bar_x as f64 / bar_width as f64) * duration)
}
