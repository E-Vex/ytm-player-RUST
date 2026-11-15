//! Native cores for `src/ytm_player/utils/logging.py` — log-line
//! filtering (argv matching, tail starts).

use crate::utils::doctor::py_tail_start;
use pyo3::prelude::*;

/// Filter already-read log lines down to those at or above *min_level* and
/// return the last *n* joined together. Mirrors the pure filtering core of
/// `ytm_player.utils.logging.get_recent_log_lines` (file I/O stays in
/// Python): lines keep their trailing newlines, unknown levels apply no
/// filter, and the tail slice follows Python's `lines[-n:]` semantics.
#[pyfunction(signature = (lines, n, min_level = None))]
pub(crate) fn filter_log_lines(lines: Vec<String>, n: i64, min_level: Option<&str>) -> String {
    let mut kept = lines;
    if let Some(level) = min_level {
        const ORDER: [&str; 5] = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
        let upper = level.to_uppercase();
        if let Some(threshold) = ORDER.iter().position(|o| *o == upper) {
            let wanted: Vec<String> = ORDER[threshold..]
                .iter()
                .map(|w| format!("[{w}]"))
                .collect();
            kept.retain(|line| wanted.iter().any(|w| line.contains(w.as_str())));
        }
    }
    // Python's `lines[-n:]` (see py_tail_start).
    kept[py_tail_start(kept.len(), n)..].concat()
}
