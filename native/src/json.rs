//! JSON codec: strict `json.dumps` / `json_loads` engines with
//! CPython-compatible number, string and sort handling. Backs the
//! `ytm --json` payloads (`src/ytm_player/cli.py`) and the auth record
//! parser (`src/ytm_player/services/auth.py`).

use crate::app::keys::unpack_too_many_message;
use crate::services::auth::JsonLoadFrame;
use crate::services::auth::JsonReject;
use pyo3::exceptions::{PyRecursionError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyIterator, PyList, PyString, PyTuple};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// CLI JSON serialisation (cli.py)
// ---------------------------------------------------------------------------

/// Python's `float.__repr__` — the shortest round-trip digits rendered
/// under CPython's layout rules: fixed notation while the decimal point
/// sits within `1e-4 <= |v| < 1e16`, exponential outside with a sign
/// and at least two exponent digits (`1e+16`, `1.5e-07`), `0.0`/`-0.0`
/// for the zeros, and the specials spelled `NaN` / `Infinity` /
/// `-Infinity` (the `allow_nan=True` spellings). The shortest digits
/// come from Rust's own `{:e}` formatting, which produces the same
/// unique shortest correctly-rounding digit sequence CPython's dtoa
/// does — except where the exact value sits *exactly* between two
/// shortest candidates: those ties round away from zero under Rust's
/// formatter and half-even under CPython's dtoa, and a tie can only
/// carry when the digit run ends odd (a tied pair's even neighbour is
/// the run one digit down), so those — and only those — ask the
/// interpreter's own repr. Every finite double serialises exactly like
/// `repr()` would render it.
fn python_float_repr(py: Python<'_>, v: f64) -> PyResult<String> {
    if v.is_nan() {
        return Ok("NaN".to_string());
    }
    if v.is_infinite() {
        return Ok(if v > 0.0 { "Infinity" } else { "-Infinity" }.to_string());
    }
    if v == 0.0 {
        return Ok(if v.is_sign_negative() { "-0.0" } else { "0.0" }.to_string());
    }
    let neg = v.is_sign_negative();
    // "d[.ddd]e[-]X" — digits and the base-10 exponent of |v|.
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp_part) = sci.split_once('e').expect("exponential spelling");
    let exp: i32 = exp_part.parse().expect("exponent digits");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // An exact midpoint between two shortest candidates rounds away
    // from zero here but half-even under CPython's dtoa — a tie only
    // carries when the digit run ends odd (its even neighbour is the
    // run one digit down), so those — and only those — ask the
    // interpreter's own repr.
    if digits
        .as_bytes()
        .last()
        .is_some_and(|b| (b - b'0') % 2 == 1)
    {
        let repr = PyFloat::new(py, v).repr()?;
        return repr.extract::<String>();
    }
    // decpt — the decimal point's position relative to the digit run
    // (digits "12345", exp 2 -> 123.45, decpt 3).
    let decpt = exp + 1;
    let n = digits.len() as i32;
    let mut out = String::with_capacity(digits.len() + 8);
    if neg {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        // Exponential: d[.ddd]e±XX, at least two exponent digits.
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        let ea = e.unsigned_abs();
        if ea < 10 {
            out.push('0');
        }
        out.push_str(&ea.to_string());
    } else if decpt <= 0 {
        // 0.000ddd
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else if decpt >= n {
        // ddd000.0
        out.push_str(&digits);
        for _ in 0..(decpt - n) {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        // dd.ddd
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    }
    Ok(out)
}

/// The four-hex `\uXXXX` escape, lowercase digits — CPython's own
/// `Py_hexdigits` table.
pub(crate) fn json_push_u_escape(out: &mut String, c: u32) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push('\\');
    out.push('u');
    out.push(HEX[((c >> 12) & 0xf) as usize] as char);
    out.push(HEX[((c >> 8) & 0xf) as usize] as char);
    out.push(HEX[((c >> 4) & 0xf) as usize] as char);
    out.push(HEX[(c & 0xf) as usize] as char);
}

/// The `ensure_ascii` escaping — CPython's `S_CHAR` rule: the printable
/// ASCII run 0x20..=0x7e (bar `"` and `\`) passes through, the seven
/// two-character escapes `\\ \" \b \t \n \f \r` render their short
/// forms, DEL included in everything else becoming a lowercase
/// four-hex `\uXXXX`, and an astral code point its UTF-16 surrogate
/// pair.
fn json_escape_codepoint_into(out: &mut String, c: u32) {
    match c {
        0x22 => out.push_str("\\\""),
        0x5c => out.push_str("\\\\"),
        0x08 => out.push_str("\\b"),
        0x09 => out.push_str("\\t"),
        0x0a => out.push_str("\\n"),
        0x0c => out.push_str("\\f"),
        0x0d => out.push_str("\\r"),
        0x20..=0x7e => out.push(c as u8 as char),
        c if c >= 0x10000 => {
            let v = c - 0x10000;
            json_push_u_escape(out, 0xd800 + (v >> 10));
            json_push_u_escape(out, 0xdc00 + (v & 0x3ff));
        }
        _ => json_push_u_escape(out, c),
    }
}

/// Escape a `&str`'s code points — the fast lane for ASCII runs with
/// nothing to escape, the per-code-point walk otherwise.
fn json_escape_str_into(out: &mut String, s: &str) {
    let plain = s.is_ascii()
        && !s
            .bytes()
            .any(|b| b == b'"' || b == b'\\' || b < 0x20 || b == 0x7f);
    out.push('"');
    if plain {
        out.push_str(s);
    } else {
        for c in s.chars() {
            json_escape_codepoint_into(out, c as u32);
        }
    }
    out.push('"');
}

/// Escape one Python string value: the fast lane over its UTF-8
/// content, and — where the string carries lone surrogates, which have
/// no UTF-8 spelling — the WTF-8 bytes of its `surrogatepass` encoding,
/// so a `\ud800` in the payload escapes exactly like the reference's
/// code-point walk would.
fn json_escape_pystring_into(out: &mut String, value: &Bound<'_, PyString>) -> PyResult<()> {
    if let Ok(s) = value.to_str() {
        json_escape_str_into(out, s);
        return Ok(());
    }
    // str.encode("utf-8", "surrogatepass") — the content as WTF-8.
    let bytes = value.call_method1("encode", ("utf-8", "surrogatepass"))?;
    let bytes = bytes
        .downcast::<PyBytes>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("encode did not return bytes"))?;
    let data = bytes.as_bytes();
    out.push('"');
    let mut i = 0usize;
    while i < data.len() {
        // One WTF-8 sequence — plain UTF-8 shapes plus the surrogate
        // range encoded as three-byte sequences.
        let b = data[i];
        let c = if b < 0x80 {
            i += 1;
            b as u32
        } else if b >> 5 == 0b110 {
            i += 2;
            ((b & 0x1f) as u32) << 6 | (data[i - 1] & 0x3f) as u32
        } else if b >> 4 == 0b1110 {
            i += 3;
            ((b & 0x0f) as u32) << 12
                | ((data[i - 2] & 0x3f) as u32) << 6
                | (data[i - 1] & 0x3f) as u32
        } else {
            i += 4;
            ((b & 0x07) as u32) << 18
                | ((data[i - 3] & 0x3f) as u32) << 12
                | ((data[i - 2] & 0x3f) as u32) << 6
                | (data[i - 1] & 0x3f) as u32
        };
        json_escape_codepoint_into(out, c);
    }
    out.push('"');
    Ok(())
}

/// `int.__repr__(value)` — the raw-storage digits, never a subclass's
/// own repr (an IntEnum member renders as its number). i64 covers the
/// whole range Rust can format directly; anything wider goes through
/// the interpreter's own repr.
fn int_repr_string(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    match value.extract::<i64>() {
        Ok(v) => Ok(v.to_string()),
        Err(_) => {
            let repr = py
                .import("builtins")?
                .getattr("int")?
                .getattr("__repr__")?
                .call1((value,))?;
            repr.extract::<String>()
        }
    }
}

/// The key coercion + escaping behind one dict key: `str` keys pass
/// through, `float` keys render through `float.__repr__`, `int` keys
/// through `int.__repr__` (an IntEnum key renders as its numeric
/// digits, never its member name), `True`/`False`/`None` as
/// `true`/`false`/`null`, anything else CPython's own `TypeError`. The
/// coerced text is then escaped like any other string — the digit and
/// repr spellings are all plain ASCII, but the walk stays uniform.
fn write_dict_key_into(out: &mut String, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(s) = key.downcast::<PyString>() {
        json_escape_pystring_into(out, s)?;
        return Ok(());
    }
    let text = if key.is_instance_of::<PyFloat>() {
        python_float_repr(py, key.extract::<f64>()?)?
    } else if key.is_none() {
        "null".to_string()
    } else if key.is_instance_of::<PyBool>() {
        if key.extract::<bool>()? {
            "true"
        } else {
            "false"
        }
        .to_string()
    } else if key.is_instance_of::<PyInt>() {
        int_repr_string(py, key)?
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "keys must be str, int, float, bool or None, not {}",
            key.get_type().name()?
        )));
    };
    json_escape_str_into(out, &text);
    Ok(())
}

/// Unpack one two-item iterable — `a, b = item` with CPython's own
/// unpack errors: the not-enough/too-many counts are the items actually
/// yielded, and the "cannot unpack non-iterable" rewrite only applies
/// to a type with no iteration support at all (an `__iter__` that
/// itself raises keeps its own error, like CPython's `tp_iter` check).
/// Serves the JSON encoder's `items()` pairs and the doctor report's
/// crash-file `(path, content)` pair alike.
pub(crate) fn unpack_two<'py>(
    pair: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let py = pair.py();
    let mut items = match pair.try_iter() {
        Ok(items) => items,
        Err(err) => {
            if err.is_instance_of::<pyo3::exceptions::PyTypeError>(py) {
                let type_iter = pair
                    .get_type()
                    .getattr("__iter__")
                    .ok()
                    .filter(|attr| !attr.is_none());
                if type_iter.is_none() {
                    let type_name = pair
                        .get_type()
                        .name()
                        .map(|name| name.to_string())
                        .unwrap_or_else(|_| "unknown".to_string());
                    return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                        "cannot unpack non-iterable {type_name} object"
                    )));
                }
            }
            return Err(err);
        }
    };
    let key = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 2, got 0)",
            ))
        }
    };
    let value = match items.next() {
        Some(item) => item?,
        None => {
            return Err(PyValueError::new_err(
                "not enough values to unpack (expected 2, got 1)",
            ))
        }
    };
    if items.next().is_some() {
        return Err(PyValueError::new_err(unpack_too_many_message(
            pair.py(),
            pair,
            2,
        )));
    }
    Ok((key, value))
}

/// One open frame of the JSON encode — a list/array, a dict/object, or
/// the default-conversion sentinel that holds a converted object's
/// marker until its subtree finishes encoding (CPython's own marker
/// lifetime around the `default()` call). The traversal is iterative,
/// so no Rust recursion can overflow: a payload nested past CPython's
/// own recursion budget encodes where the reference would raise
/// `RecursionError`.
enum JsonFrame<'py> {
    List {
        iter: Bound<'py, PyIterator>,
        first: bool,
        id: usize,
    },
    Dict {
        iter: Bound<'py, PyIterator>,
        first: bool,
        id: usize,
    },
    Default {
        id: usize,
    },
}

/// The `json.dumps(data, indent=..., default=str)` text behind
/// `cli.py`'s `_json_output` — the serialisation every JSON-emitting
/// CLI command prints.
///
/// Mirrors the pure-Python reference (`_json_dumps`) — which is the
/// stdlib call itself — with every other parameter at its default:
/// `ensure_ascii=True` (the `S_CHAR` escaping above), the separators
/// `(', ', ': ')` compact and `(',', ': ')` with the newline+indent
/// layout when *indent* is given (a negative indent collapsing to `""`
/// exactly like the reference's `' ' * n` multiplication; an absurd
/// width raises `MemoryError` rather than building a multi-gigabyte
/// pad), `sort_keys=False` (insertion order preserved), `allow_nan=True`
/// (`NaN`/`Infinity`/`-Infinity`), and `check_circular=True` — the
/// ancestor-chain guard reporting `Circular reference detected` for a
/// cycle while shared non-cyclic references encode on both sides.
///
/// The value dispatch follows the reference's order — the consts, then
/// `str`, `int` (through `int.__repr__`, so an IntEnum member renders
/// as its number and bigints keep their full digits), `float` (through
/// `float.__repr__`, see `python_float_repr`), `list`/`tuple`, `dict`,
/// and then the default call: `str(o)` fetched from `builtins` at call
/// time and invoked exactly like the reference's `default=str` would,
/// its result re-encoded through the same dispatch with the original
/// object's marker held for the conversion's lifetime. Lists iterate
/// through Python's iteration protocol and dicts through their `items`
/// method — the calls both CPython encoders make — so subclasses steer
/// both backends identically; the empty containers render `[]`/`{}`
/// from the truthiness gate before any marker is taken. Dict keys
/// coerce through the reference's rule (see `write_dict_key_into`)
/// with CPython's own `TypeError` for anything else; an `items()` pair
/// that is not a 2-tuple raises the pure-Python fallback's unpack
/// `ValueError`. The inputs are never mutated, and the traversal holds
/// no Rust recursion, so deeply nested payloads encode where the
/// reference's recursive walk would hit the interpreter's recursion
/// budget.
#[pyfunction]
#[pyo3(signature = (data, indent = None))]
pub(crate) fn json_dumps<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indent: Option<i64>,
) -> PyResult<String> {
    if indent.is_some_and(|n| !(-1_000_000..=1_000_000).contains(&n)) {
        return Err(pyo3::exceptions::PyMemoryError::new_err(
            "indent is too large",
        ));
    }
    let indent_pad = " ".repeat(indent.map_or(0, |n| n.max(0) as usize));
    let pretty = indent.is_some();
    json_encode(py, data, indent_pad, pretty, false, false)
}

/// `sorted(pairs)` with CPython's own tuple ordering — the rule the
/// reference's `sort_keys` arm applies to a dict's `items()`: tuple `a <
/// b` reads the first position through `==` (falling to the second only
/// between equal keys, so equal keys compare their values), every
/// ordering decision made through `<` exactly like `list.sort` (an
/// unorderable pair raising `<`'s own `TypeError`), and pairs equal
/// under neither `<` keeping their input order — `sorted` is stable.
fn py_sorted_pairs<'py>(
    mut pairs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
) -> PyResult<Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    // tuple a < b — key equality first, the value `<` only between
    // equal keys, the key `<` otherwise.
    fn tuple_lt(
        a: &(Bound<'_, PyAny>, Bound<'_, PyAny>),
        b: &(Bound<'_, PyAny>, Bound<'_, PyAny>),
    ) -> PyResult<bool> {
        if a.0.eq(&b.0)? {
            a.1.lt(&b.1)
        } else {
            a.0.lt(&b.0)
        }
    }
    let mut sort_err: Option<PyErr> = None;
    pairs.sort_by(|a, b| {
        if sort_err.is_some() {
            // The outcome no longer matters — the error returns anyway.
            return std::cmp::Ordering::Equal;
        }
        match (tuple_lt(a, b), tuple_lt(b, a)) {
            (Ok(true), _) => std::cmp::Ordering::Less,
            (Ok(false), Ok(true)) => std::cmp::Ordering::Greater,
            (Ok(_), Ok(_)) => std::cmp::Ordering::Equal,
            (Err(err), _) | (_, Err(err)) => {
                sort_err = Some(err);
                std::cmp::Ordering::Equal
            }
        }
    });
    match sort_err {
        Some(err) => Err(err),
        None => Ok(pairs),
    }
}

/// The shared encode core behind `json_dumps` and the auth engines'
/// canonical serialiser — the same iterative CPython-encoder walk, with
/// the `sort_keys` and default-less arms parameterised: `sort_keys`
/// collects the dict's `items()` and orders them through CPython's own
/// tuple comparison (see `py_sorted_pairs`) instead of keeping
/// insertion order, and `strict` replaces the `default=str` conversion
/// with the reference's own `TypeError` for the unserialisable.
pub(crate) fn json_encode<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indent_pad: String,
    pretty: bool,
    sort_keys: bool,
    strict: bool,
) -> PyResult<String> {
    let mut out = String::new();
    // The ancestor-chain ids behind the circular-reference guard —
    // inserted when a container opens, removed when it closes.
    let mut markers: HashSet<usize> = HashSet::new();
    let mut stack: Vec<JsonFrame<'py>> = Vec::new();
    let mut current: Option<Bound<'py, PyAny>> = Some(data.clone());

    loop {
        if let Some(value) = current.take() {
            // The value dispatch, in the reference's order.
            if value.is_none() {
                out.push_str("null");
            } else if value.is_instance_of::<PyBool>() {
                out.push_str(if value.extract::<bool>()? {
                    "true"
                } else {
                    "false"
                });
            } else if let Ok(s) = value.downcast::<PyString>() {
                json_escape_pystring_into(&mut out, s)?;
            } else if value.is_instance_of::<PyInt>() {
                out.push_str(&int_repr_string(py, &value)?);
            } else if value.is_instance_of::<PyFloat>() {
                out.push_str(&python_float_repr(py, value.extract::<f64>()?)?);
            } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
                // if not lst: '[]' — the truthiness gate before any
                // marker is taken.
                if !value.is_truthy()? {
                    out.push_str("[]");
                } else {
                    let id = value.as_ptr() as usize;
                    if markers.contains(&id) {
                        return Err(PyValueError::new_err("Circular reference detected"));
                    }
                    markers.insert(id);
                    out.push('[');
                    if pretty {
                        out.push('\n');
                        out.push_str(&indent_pad.repeat(stack.len() + 1));
                    }
                    stack.push(JsonFrame::List {
                        iter: value.try_iter()?,
                        first: true,
                        id,
                    });
                }
            } else if value.is_instance_of::<PyDict>() {
                if !value.is_truthy()? {
                    out.push_str("{}");
                } else {
                    let id = value.as_ptr() as usize;
                    if markers.contains(&id) {
                        return Err(PyValueError::new_err("Circular reference detected"));
                    }
                    markers.insert(id);
                    out.push('{');
                    if pretty {
                        out.push('\n');
                        out.push_str(&indent_pad.repeat(stack.len() + 1));
                    }
                    // items() through the mapping protocol — the method
                    // call both CPython encoders make — with sort_keys
                    // collecting the pairs and ordering them through
                    // CPython's own tuple comparison (see
                    // py_sorted_pairs) exactly like the reference
                    // encoder's `sorted(dct.items())`, the ordered pairs
                    // re-listed for the shared iteration.
                    let items = value.getattr("items")?.call0()?;
                    let iter = if sort_keys {
                        let mut pairs: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
                        for pair in items.try_iter()? {
                            pairs.push(unpack_two(&pair?)?);
                        }
                        let pairs = py_sorted_pairs(pairs)?;
                        PyList::new(py, pairs)?.try_iter()?
                    } else {
                        items.try_iter()?
                    };
                    stack.push(JsonFrame::Dict {
                        iter,
                        first: true,
                        id,
                    });
                }
            } else if strict {
                // No default: the reference encoder's own TypeError for
                // the unserialisable.
                return Err(PyTypeError::new_err(format!(
                    "Object of type {} is not JSON serializable",
                    value.get_type().name()?
                )));
            } else {
                // default(o) — str, fetched from builtins at call time
                // exactly like the reference's global lookup and called
                // as its default would call it. The sentinel below pops
                // the original object's marker off once the converted
                // value's subtree finishes encoding — CPython's own
                // marker lifetime around the default call.
                let id = value.as_ptr() as usize;
                if markers.contains(&id) {
                    return Err(PyValueError::new_err("Circular reference detected"));
                }
                markers.insert(id);
                let converted = py.import("builtins")?.getattr("str")?.call1((value,))?;
                stack.push(JsonFrame::Default { id });
                current = Some(converted);
            }
            continue;
        }
        // No pending value — pull the next item from the top frame, or
        // close the frame when its iteration is exhausted.
        let depth = stack.len();
        enum Pull<'py> {
            Item(Bound<'py, PyAny>),
            Close { id: usize, closer: Option<u8> },
        }
        let pull = match stack.last_mut() {
            None => break,
            Some(JsonFrame::List { iter, first, id }) => match iter.next() {
                Some(Ok(item)) => {
                    if !*first {
                        if pretty {
                            out.push(',');
                            out.push('\n');
                            out.push_str(&indent_pad.repeat(depth));
                        } else {
                            out.push_str(", ");
                        }
                    }
                    *first = false;
                    Pull::Item(item)
                }
                Some(Err(err)) => return Err(err),
                None => Pull::Close {
                    id: *id,
                    closer: Some(b']'),
                },
            },
            Some(JsonFrame::Dict { iter, first, id }) => match iter.next() {
                Some(Ok(pair)) => {
                    let (key, value) = unpack_two(&pair)?;
                    if !*first {
                        if pretty {
                            out.push(',');
                            out.push('\n');
                            out.push_str(&indent_pad.repeat(depth));
                        } else {
                            out.push_str(", ");
                        }
                    }
                    *first = false;
                    write_dict_key_into(&mut out, py, &key)?;
                    out.push_str(": ");
                    Pull::Item(value)
                }
                Some(Err(err)) => return Err(err),
                None => Pull::Close {
                    id: *id,
                    closer: Some(b'}'),
                },
            },
            Some(JsonFrame::Default { id }) => Pull::Close {
                id: *id,
                closer: None,
            },
        };
        match pull {
            Pull::Item(value) => current = Some(value),
            Pull::Close { id, closer } => {
                if let Some(closer) = closer {
                    if pretty {
                        out.push('\n');
                        out.push_str(&indent_pad.repeat(depth - 1));
                    }
                    out.push(closer as char);
                }
                markers.remove(&id);
                stack.pop();
            }
        }
    }
    Ok(out)
}

/// `json.loads(s)` over a decoded str's code points — CPython's own
/// scanner grammar: the four whitespace characters (` `, `\t`, `\n`,
/// `\r`) skipped between tokens; objects and arrays with their
/// delimiter rules (a trailing comma rejected, an object key always a
/// string); strings through the escape table (`\" \\ \/ \b \f \n \r
/// \t`, `\uXXXX` with surrogate pairing — a high surrogate followed by
/// a `\u` escape combines only with a low one, every other shape
/// keeping its lone surrogate exactly like the scanner's slice-peek
/// rule) and raw control characters rejected; the literals
/// null/true/false/NaN/Infinity/-Infinity prefix-matched like the
/// scanner's own slice equality, so trailing junk rejects at the
/// delimiter exactly where the reference's does; numbers through the
/// leading-zero rule with digit-required fraction and exponent,
/// arbitrary-precision integers built through the interpreter's own
/// `int()` and floats through the same correctly-rounded decimal
/// conversion (an absurd exponent overflowing to `inf` silently);
/// duplicate object keys keeping the first position with the last
/// value through the dict store's own replace; and nesting counted
/// against `sys.getrecursionlimit()` with the same `RecursionError`
/// the reference's decoder raises for a past-budget payload — the
/// reference's exact borderline sits at the interpreter's live
/// recursion budget (the limit minus the frames already on the stack),
/// so an input nested within a frame or two of the limit may accept
/// here where the reference raises; every realistic depth answers
/// identically.
pub(crate) fn json_loads<'py>(
    py: Python<'py>,
    text: &Bound<'py, PyString>,
) -> Result<Bound<'py, PyAny>, JsonReject> {
    // The text's code points — the fast lane for every str (its UTF-8
    // content), the surrogatepass round trip only for a str carrying
    // lone surrogates (which a plain bytes.decode never produces; only
    // a decode override could return one).
    let cps: Vec<u32> = match text.to_str() {
        Ok(s) => s.chars().map(|c| c as u32).collect(),
        Err(_) => {
            let bytes = text
                .call_method1("encode", ("utf-8", "surrogatepass"))
                .map_err(JsonReject::Py)?;
            let bytes = bytes
                .downcast::<PyBytes>()
                .map_err(|_| JsonReject::Py(PyTypeError::new_err("encode did not return bytes")))?;
            wtf8_code_points(bytes.as_bytes())
        }
    };
    let limit: usize = py
        .import("sys")
        .and_then(|sys| sys.getattr("getrecursionlimit"))
        .and_then(|f| f.call0())
        .and_then(|v| v.extract::<usize>())
        .map_err(JsonReject::Py)?;
    let int_type = py
        .import("builtins")
        .and_then(|b| b.getattr("int"))
        .map_err(JsonReject::Py)?;
    let mut scanner = JsonScanner {
        py,
        cps,
        i: 0,
        depth: 0,
        limit,
        int_type,
    };
    scanner.scan()
}

/// The WTF-8 code-point reader — plain UTF-8 shapes plus the surrogate
/// range as three-byte sequences, the mirror of the writer the
/// `ensure_ascii` escaping already uses.
fn wtf8_code_points(data: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0usize;
    while i < data.len() {
        let b = data[i];
        let c = if b < 0x80 {
            i += 1;
            b as u32
        } else if b >> 5 == 0b110 {
            i += 2;
            ((b & 0x1f) as u32) << 6 | (data[i - 1] & 0x3f) as u32
        } else if b >> 4 == 0b1110 {
            i += 3;
            ((b & 0x0f) as u32) << 12
                | ((data[i - 2] & 0x3f) as u32) << 6
                | (data[i - 1] & 0x3f) as u32
        } else {
            i += 4;
            ((b & 0x07) as u32) << 18
                | ((data[i - 3] & 0x3f) as u32) << 12
                | ((data[i - 2] & 0x3f) as u32) << 6
                | (data[i - 1] & 0x3f) as u32
        };
        out.push(c);
    }
    out
}

/// The scanner state — the code points, the cursor, the nesting depth
/// against the interpreter's recursion limit, and the `int` type for
/// arbitrary-precision number construction.
struct JsonScanner<'py> {
    py: Python<'py>,
    cps: Vec<u32>,
    i: usize,
    depth: usize,
    limit: usize,
    int_type: Bound<'py, PyAny>,
}

impl<'py> JsonScanner<'py> {
    /// The scanner's whitespace: exactly ` `, `\t`, `\n`, `\r`.
    fn ws(&mut self) {
        while self.i < self.cps.len() && matches!(self.cps[self.i], 0x20 | 0x09 | 0x0a | 0x0d) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u32> {
        self.cps.get(self.i).copied()
    }

    /// The scanner's literal match — a slice-equality prefix check that
    /// consumes the word only when every character lines up, trailing
    /// characters left for the delimiter rules to reject.
    fn literal(&mut self, word: &[u8]) -> bool {
        if self.i + word.len() > self.cps.len() {
            return false;
        }
        for (k, b) in word.iter().enumerate() {
            if self.cps[self.i + k] != *b as u32 {
                return false;
            }
        }
        self.i += word.len();
        true
    }

    /// `Py_EnterRecursiveCall`'s counter — the nesting budget.
    fn enter(&mut self) -> Result<(), JsonReject> {
        self.depth += 1;
        if self.depth > self.limit {
            return Err(JsonReject::Py(PyRecursionError::new_err(
                "maximum recursion depth exceeded while decoding a JSON object",
            )));
        }
        Ok(())
    }

    /// Four hex digits — the `\uXXXX` escape's own validation.
    fn hex4(&mut self) -> Result<u32, JsonReject> {
        if self.i + 4 > self.cps.len() {
            return Err(JsonReject::Reject);
        }
        let mut v = 0u32;
        for k in 0..4 {
            let c = self.cps[self.i + k];
            let d = match c {
                0x30..=0x39 => c - 0x30,
                0x41..=0x46 => c - 0x41 + 10,
                0x61..=0x66 => c - 0x61 + 10,
                _ => return Err(JsonReject::Reject),
            };
            v = v * 16 + d;
        }
        self.i += 4;
        Ok(v)
    }

    /// One JSON string — the cursor sits on the opening quote. Escapes
    /// through the scanner's table, `\uXXXX` with the pairing rule (a
    /// high surrogate followed by a `\u` escape peeks the second; only
    /// a low surrogate combines, anything else leaves the first escape
    /// standing and the second scanned as its own), raw control
    /// characters rejected, lone surrogates kept. The built str takes
    /// the fast lane when every code point is a character and the
    /// interpreter's own `surrogatepass` decode of the WTF-8 bytes
    /// otherwise, so a lone `\ud800` escapes into the result exactly
    /// like the reference's scanner would build it.
    fn string(&mut self) -> Result<Bound<'py, PyString>, JsonReject> {
        let py = self.py;
        self.i += 1; // the opening quote
        let mut out: Vec<u32> = Vec::new();
        loop {
            let c = match self.peek() {
                Some(c) => c,
                None => return Err(JsonReject::Reject), // unterminated
            };
            match c {
                0x22 => {
                    self.i += 1;
                    break;
                }
                0x5c => {
                    self.i += 1;
                    let esc = match self.peek() {
                        Some(e) => e,
                        None => return Err(JsonReject::Reject),
                    };
                    match esc {
                        0x22 | 0x5c | 0x2f => {
                            out.push(esc);
                            self.i += 1;
                        }
                        0x62 => {
                            out.push(0x08);
                            self.i += 1;
                        }
                        0x66 => {
                            out.push(0x0c);
                            self.i += 1;
                        }
                        0x6e => {
                            out.push(0x0a);
                            self.i += 1;
                        }
                        0x72 => {
                            out.push(0x0d);
                            self.i += 1;
                        }
                        0x74 => {
                            out.push(0x09);
                            self.i += 1;
                        }
                        0x75 => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            let mut uni = hi;
                            if (0xd800..=0xdbff).contains(&hi)
                                && self.i + 1 < self.cps.len()
                                && self.cps[self.i] == 0x5c
                                && self.cps[self.i + 1] == 0x75
                            {
                                // The scanner's peek: `s[end:end+2] == '\\u'`.
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..=0xdfff).contains(&lo) {
                                    uni = 0x10000 + ((hi - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    // Not a low surrogate: the first escape
                                    // stands alone, the second is its own.
                                    self.i = save;
                                }
                            }
                            out.push(uni);
                        }
                        _ => return Err(JsonReject::Reject), // invalid escape
                    }
                }
                c if c < 0x20 => return Err(JsonReject::Reject), // control character
                c => {
                    out.push(c);
                    self.i += 1;
                }
            }
        }
        if out.iter().all(|&c| char::from_u32(c).is_some()) {
            let s: String = out.iter().filter_map(|&c| char::from_u32(c)).collect();
            Ok(PyString::new(py, &s))
        } else {
            let mut bytes: Vec<u8> = Vec::with_capacity(out.len());
            for &c in &out {
                match char::from_u32(c) {
                    Some(ch) => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                    None => {
                        // A lone surrogate as its three-byte WTF-8 shape.
                        bytes.push(0xed);
                        bytes.push(0xa0 | (((c >> 6) & 0x1f) as u8));
                        bytes.push(0x80 | (c & 0x3f) as u8);
                    }
                }
            }
            let b = PyBytes::new(py, &bytes);
            let s = b
                .call_method1("decode", ("utf-8", "surrogatepass"))
                .map_err(JsonReject::Py)?;
            s.downcast_into::<PyString>()
                .map_err(|_| JsonReject::Py(PyTypeError::new_err("decode did not return str")))
        }
    }

    /// One object key with its colon — the cursor sits before the key.
    fn object_key(&mut self) -> Result<Bound<'py, PyString>, JsonReject> {
        self.ws();
        if self.peek() != Some(0x22) {
            return Err(JsonReject::Reject); // expecting a property name
        }
        let key = self.string()?;
        self.ws();
        if self.peek() != Some(0x3a) {
            return Err(JsonReject::Reject); // expecting ':'
        }
        self.i += 1;
        Ok(key)
    }

    /// One number — the grammar's own shape: an optional sign, `0` or a
    /// non-zero-leading digit run, a digit-required fraction, a
    /// digit-required exponent. Integers build through the
    /// interpreter's own `int()` (arbitrary precision, `-0` the int 0);
    /// floats through the same correctly-rounded decimal conversion
    /// (`1e999` the float `inf`, never an error).
    fn number(&mut self) -> Result<Bound<'py, PyAny>, JsonReject> {
        let start = self.i;
        if self.peek() == Some(0x2d) {
            self.i += 1;
        }
        match self.peek() {
            Some(0x30) => self.i += 1,
            Some(0x31..=0x39) => {
                while matches!(self.peek(), Some(0x30..=0x39)) {
                    self.i += 1;
                }
            }
            _ => return Err(JsonReject::Reject),
        }
        let mut is_float = false;
        if self.peek() == Some(0x2e) {
            self.i += 1;
            if !matches!(self.peek(), Some(0x30..=0x39)) {
                return Err(JsonReject::Reject);
            }
            while matches!(self.peek(), Some(0x30..=0x39)) {
                self.i += 1;
            }
            is_float = true;
        }
        if matches!(self.peek(), Some(0x65) | Some(0x45)) {
            self.i += 1;
            if matches!(self.peek(), Some(0x2b) | Some(0x2d)) {
                self.i += 1;
            }
            if !matches!(self.peek(), Some(0x30..=0x39)) {
                return Err(JsonReject::Reject);
            }
            while matches!(self.peek(), Some(0x30..=0x39)) {
                self.i += 1;
            }
            is_float = true;
        }
        let token: String = self.cps[start..self.i]
            .iter()
            .map(|&c| char::from_u32(c).unwrap_or('\u{fffd}'))
            .collect();
        let py = self.py;
        if is_float {
            let v: f64 = token.parse().map_err(|_| JsonReject::Reject)?;
            Ok(PyFloat::new(py, v).into_any())
        } else {
            Ok(self.int_type.call1((&token,)).map_err(JsonReject::Py)?)
        }
    }

    /// The document walk — produce a value, consume it into the frame
    /// stack, close containers on their delimiters, and answer the
    /// single top-level value with the trailing-garbage rejection.
    fn scan(&mut self) -> Result<Bound<'py, PyAny>, JsonReject> {
        let py = self.py;
        let mut stack: Vec<JsonLoadFrame> = Vec::new();
        let mut current: Option<Bound<'py, PyAny>> = None;
        loop {
            // ── produce ── a value, an opened container, or a finished
            // container's aggregate.
            if current.is_none() {
                self.ws();
                let c = match self.peek() {
                    Some(c) => c,
                    None => return Err(JsonReject::Reject), // expecting a value
                };
                match c {
                    0x7b => {
                        self.i += 1;
                        self.enter()?;
                        self.ws();
                        if self.peek() == Some(0x7d) {
                            self.i += 1;
                            self.depth -= 1;
                            current = Some(PyDict::new(py).into_any());
                        } else {
                            let key = self.object_key()?;
                            stack.push(JsonLoadFrame::Object {
                                pairs: Vec::new(),
                                pending_key: Some(key.unbind().into_any()),
                            });
                        }
                    }
                    0x5b => {
                        self.i += 1;
                        self.enter()?;
                        self.ws();
                        if self.peek() == Some(0x5d) {
                            self.i += 1;
                            self.depth -= 1;
                            current = Some(PyList::empty(py).into_any());
                        } else {
                            stack.push(JsonLoadFrame::Array { items: Vec::new() });
                        }
                    }
                    0x22 => {
                        let s = self.string()?;
                        current = Some(s.into_any());
                    }
                    0x74 => {
                        if self.literal(b"true") {
                            current = Some(PyBool::new(py, true).to_owned().into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x66 => {
                        if self.literal(b"false") {
                            current = Some(PyBool::new(py, false).to_owned().into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x6e => {
                        if self.literal(b"null") {
                            current = Some(py.None().into_bound(py));
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x4e => {
                        if self.literal(b"NaN") {
                            current = Some(PyFloat::new(py, f64::NAN).into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x49 => {
                        if self.literal(b"Infinity") {
                            current = Some(PyFloat::new(py, f64::INFINITY).into_any());
                        } else {
                            return Err(JsonReject::Reject);
                        }
                    }
                    0x2d => {
                        if self.cps.get(self.i + 1) == Some(&0x49) {
                            if self.literal(b"-Infinity") {
                                current = Some(PyFloat::new(py, f64::NEG_INFINITY).into_any());
                            } else {
                                return Err(JsonReject::Reject);
                            }
                        } else {
                            let v = self.number()?;
                            current = Some(v);
                        }
                    }
                    0x30..=0x39 => {
                        let v = self.number()?;
                        current = Some(v);
                    }
                    _ => return Err(JsonReject::Reject),
                }
            }
            // ── consume ── the produced value into the stack, or answer
            // it at the top level.
            if let Some(value) = current.take() {
                if stack.is_empty() {
                    self.ws();
                    if self.i != self.cps.len() {
                        return Err(JsonReject::Reject); // extra data
                    }
                    return Ok(value);
                }
                let is_object = matches!(stack.last(), Some(JsonLoadFrame::Object { .. }));
                if is_object {
                    let Some(JsonLoadFrame::Object { pairs, pending_key }) = stack.last_mut()
                    else {
                        unreachable!("frame kind checked above");
                    };
                    let key = pending_key.take().expect("object frame awaits its value");
                    pairs.push((key, value.unbind()));
                } else {
                    let Some(JsonLoadFrame::Array { items }) = stack.last_mut() else {
                        unreachable!("frame kind checked above");
                    };
                    items.push(value.unbind());
                }
                self.ws();
                match self.peek() {
                    Some(0x2c) => {
                        self.i += 1;
                        if is_object {
                            let key = self.object_key()?;
                            if let Some(JsonLoadFrame::Object { pending_key, .. }) =
                                stack.last_mut()
                            {
                                *pending_key = Some(key.unbind().into_any());
                            }
                        }
                    }
                    Some(0x5d) | Some(0x7d) => {
                        let closing_array = self.peek() == Some(0x5d);
                        if closing_array == is_object {
                            return Err(JsonReject::Reject); // wrong closer
                        }
                        self.i += 1;
                        self.depth -= 1;
                        let frame = stack.pop().expect("frame exists");
                        current = Some(match frame {
                            JsonLoadFrame::Array { items } => {
                                PyList::new(py, items).map_err(JsonReject::Py)?.into_any()
                            }
                            JsonLoadFrame::Object { pairs, .. } => {
                                let d = PyDict::new(py);
                                // The dict store's own replace: a repeated
                                // key keeps its first position and takes
                                // the last value.
                                for (k, v) in pairs {
                                    d.set_item(k, v).map_err(JsonReject::Py)?;
                                }
                                d.into_any()
                            }
                        });
                    }
                    _ => return Err(JsonReject::Reject), // expecting a delimiter
                }
            }
        }
    }
}
