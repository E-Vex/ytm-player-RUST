//! Native cores for `src/ytm_player/utils/terminal.py` — the terminal
//! orientation probe.

use pyo3::conversion::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyFloat;

// ── Terminal orientation gate (utils/terminal.py) ────────────────────────

/// The terminal orientation verdict — `utils/terminal.py`'s
/// `get_orientation`, the aspect-ratio gate behind the app's
/// horizontal/vertical layout decision.
///
/// Mirrors the reference's one expression exactly, every operator the
/// interpreter's own: `rows == 0` through CPython's `==` (its
/// `__eq__` protocol, a non-int rows simply not equal), `cols / rows`
/// through CPython's true division (arbitrary-precision ints divided
/// correctly-rounded — a float on every path — its TypeError for a
/// non-numeric pair kept), and the `> 2.3` comparison through
/// CPython's `>` against the very double the reference's literal
/// compiles to. `"horizontal"` when rows is zero or the ratio exceeds
/// the threshold, `"vertical"` otherwise.
#[pyfunction]
#[pyo3(signature = (cols, rows))]
pub(crate) fn get_orientation<'py>(
    py: Python<'py>,
    cols: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<&'static str> {
    let zero = 0i64.into_py_any(py)?;
    if rows.eq(&zero)? {
        return Ok("horizontal");
    }
    let ratio = cols.div(rows)?;
    let threshold = PyFloat::new(py, 2.3).into_any();
    if ratio.gt(&threshold)? {
        Ok("horizontal")
    } else {
        Ok("vertical")
    }
}
