//! Native cores for `src/ytm_player/app/_ipc.py` — IPC dispatch routing
//! and seek-offset computation.

use crate::app::keys::unpack_dispatch_row;
use crate::services::history::builtin_int;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The dispatch table behind `IPCMixin._handle_ipc_command` from
/// `app/_ipc.py` — which branch of the match handles each IPC command
/// string, and the rating the like/dislike/unlike branches rate with.
///
/// Mirrors the pure-Python reference (`_ipc_dispatch_route`): each row
/// of *table* is compared in order with Python `==` (subject on the
/// left — exactly like the reference `match`'s lazily evaluated
/// literal value patterns, which compile to `command == literal`
/// comparisons, first equal match wins) and anything no row claims is
/// unknown. *table* is the module's own ordered
/// `(command, branch, rating)` tuple in production; a stub table
/// drives the walk identically, with row unpacking raising CPython's
/// own for-loop errors (`ValueError` for wrong-length rows, the
/// `cannot unpack non-iterable …` `TypeError` rewrite only for a type
/// with no iteration support at all). The outcome map carries
/// `{"branch": str, "rating": str | None}` — fresh in every field,
/// with the matched row's branch/rating objects passed through
/// unchanged — and neither *command* nor *table* is mutated.
#[pyfunction]
#[pyo3(signature = (command, table))]
pub(crate) fn ipc_dispatch_route<'py>(
    py: Python<'py>,
    command: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    for row in table.try_iter()? {
        let (member, branch, rating) = unpack_dispatch_row(&row?)?;
        // command == member — subject on the left, exactly like the
        // reference match's compiled value patterns.
        if command.eq(&member)? {
            let out = PyDict::new(py);
            out.set_item("branch", branch)?;
            out.set_item("rating", rating)?;
            return Ok(out);
        }
    }
    let out = PyDict::new(py);
    out.set_item("branch", "unknown")?;
    out.set_item("rating", py.None())?;
    Ok(out)
}

/// `float(value)` — the builtin fetched from `builtins` at call time,
/// exactly like the reference's global lookup, so the conversion
/// semantics (and any mutation of `builtins.float`) are the
/// interpreter's own: the string grammar with its whitespace,
/// underscores and `inf`/`nan` spellings, `__float__`/`__index__`
/// coercion, and CPython's own `ValueError`/`TypeError` shapes.
pub(crate) fn builtin_float<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let float_type = py.import("builtins")?.getattr("float")?;
    float_type.call1((value,))
}

/// A fresh `{"kind": …}` outcome row for the seek parser, with the
/// branch's payload (`seconds` for the parsed kinds, `offset` for the
/// invalid ones) passed through as the object it is.
fn seek_outcome<'py>(
    py: Python<'py>,
    kind: &str,
    payload: Option<(&str, &Bound<'py, PyAny>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("kind", kind)?;
    if let Some((key, value)) = payload {
        out.set_item(key, value)?;
    }
    Ok(out)
}

/// The seek-offset parser behind `IPCMixin._ipc_seek` from
/// `app/_ipc.py` — the classification of one offset value into the
/// seek it names.
///
/// Mirrors the pure-Python reference (`_ipc_seek_offset`): the offset
/// is the raw `args.get("offset", "")` value, whatever object it is. A
/// falsy one (the missing key's `""` default, `None`, `0`, `0.0`,
/// `False`, every empty container) is missing — Python truthiness,
/// gated before any attribute is touched. A truthy one is probed with
/// `startswith("+")` / `startswith("-")` — both through the instance's
/// own method lookup, short-circuited exactly like the reference's
/// `or` (a value claiming `+` never probes `-`), so a stub's override
/// steers both backends identically and a non-str value raises
/// CPython's own `AttributeError` right where the reference's first
/// call would (`bytes` its `TypeError` from the str-prefixed call). A
/// signed value is relative — `float(offset)` through the builtin
/// fetched at call time, its `ValueError` mapping to the
/// invalid-offset verdict while every other error keeps its own shape.
/// A value containing `":"` (the containment protocol on the right
/// operand) is a clock — `split(":")` outside the reference's `try` so
/// a raising split keeps its error, then `len(parts)` inside it, then
/// the two- and three-part parses with `int()` in the reference's
/// evaluation order and Python's own `*`/`+` arithmetic
/// (`int(a)*60 + int(b)`, `int(a)*3600 + int(b)*60 + int(c)` — exact
/// at any magnitude, a Unicode-digit or underscore-bearing part parsing
/// exactly like it would at the call site); any `ValueError` from the
/// parses maps to the invalid-time verdict while a part's own
/// `TypeError` propagates, and `float(total)` sits outside the `try` so
/// a total too large for a float keeps its `OverflowError`. Everything
/// else is absolute seconds — `float(offset)` with the same
/// `ValueError` verdict. The outcome is a fresh `{"kind": …}` map —
/// `"missing"`, `"relative"`/`"clock"`/`"absolute"` with the parsed
/// `seconds` object, or `"invalid_offset"`/`"invalid_time"` carrying
/// the raw offset for the response message — and the input is never
/// mutated.
#[pyfunction]
#[pyo3(signature = (offset))]
pub(crate) fn ipc_seek_offset<'py>(
    py: Python<'py>,
    offset: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // if not offset_str: -> the missing-offset verdict, before any
    // attribute is touched.
    if !offset.is_truthy()? {
        return seek_outcome(py, "missing", None);
    }
    // offset_str.startswith("+") or offset_str.startswith("-")
    let startswith = offset.getattr("startswith")?;
    let relative =
        startswith.call1(("+",))?.is_truthy()? || startswith.call1(("-",))?.is_truthy()?;
    if relative {
        // seconds = float(offset_str) — except ValueError
        return match builtin_float(py, offset) {
            Ok(seconds) => seek_outcome(py, "relative", Some(("seconds", &seconds))),
            Err(err) if err.is_instance_of::<PyValueError>(py) => {
                seek_outcome(py, "invalid_offset", Some(("offset", offset)))
            }
            Err(err) => Err(err),
        };
    }
    // ":" in offset_str — the containment protocol on the right operand.
    if offset.contains(":")? {
        // parts = offset_str.split(":") — outside the reference's try,
        // so a raising split keeps its own error.
        let parts = offset.getattr("split")?.call1((":",))?;
        // The try body: len(parts) (inside it, exactly like the
        // reference), then the int() parses in the reference's
        // evaluation order with Python's own arithmetic. Any
        // ValueError inside maps to the invalid-time verdict; the
        // helper's `?` returns land in the match below, not straight
        // out of this function.
        let total = match (|| -> PyResult<Option<Bound<'py, PyAny>>> {
            match parts.len()? {
                2 => {
                    // int(parts[0]) * 60 + int(parts[1])
                    let first = builtin_int(py, &parts.get_item(0)?)?;
                    let head = first.mul(60i64)?;
                    let second = builtin_int(py, &parts.get_item(1)?)?;
                    Ok(Some(head.add(&second)?))
                }
                3 => {
                    // int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                    let first = builtin_int(py, &parts.get_item(0)?)?;
                    let head = first.mul(3600i64)?;
                    let second = builtin_int(py, &parts.get_item(1)?)?;
                    let mid = head.add(&second.mul(60i64)?)?;
                    let third = builtin_int(py, &parts.get_item(2)?)?;
                    Ok(Some(mid.add(&third)?))
                }
                // The reference's else-return, signalled rather than
                // raised so it cannot be confused with a ValueError.
                _ => Ok(None),
            }
        })() {
            Ok(Some(total)) => total,
            Ok(None) => return seek_outcome(py, "invalid_time", Some(("offset", offset))),
            Err(err) if err.is_instance_of::<PyValueError>(py) => {
                return seek_outcome(py, "invalid_time", Some(("offset", offset)))
            }
            Err(err) => return Err(err),
        };
        // float(total) — outside the reference's try: a total too large
        // for a float keeps its own OverflowError.
        let seconds = builtin_float(py, &total)?;
        return seek_outcome(py, "clock", Some(("seconds", &seconds)));
    }
    // seconds = float(offset_str) — except ValueError
    match builtin_float(py, offset) {
        Ok(seconds) => seek_outcome(py, "absolute", Some(("seconds", &seconds))),
        Err(err) if err.is_instance_of::<PyValueError>(py) => {
            seek_outcome(py, "invalid_offset", Some(("offset", offset)))
        }
        Err(err) => Err(err),
    }
}
