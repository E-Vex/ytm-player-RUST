//! Native cores for `src/ytm_player/config/keymap.py` — the `KeyMatcher`
//! bridge class and the keymap load plan.

use crate::services::auth::py_strip;
use crate::utils::doctor::py_split_none;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Keymap sequence matching (mirrors ytm_player/config/keymap.py)
// ---------------------------------------------------------------------------

/// Native matcher for the KeyMap binding table.
///
/// Mirrors the exact/pending/no-match semantics of `KeyMap.match`:
///
/// - `("exact", Some(action))` — the pressed sequence is itself a binding.
///   Exact matches win even when the sequence is also a proper prefix of a
///   longer binding (the Python reference checks `key_sequence in bindings`
///   before scanning prefixes).
/// - `("pending", None)` — the sequence is a proper prefix of at least one
///   longer binding, so the app should buffer it and wait for more keys.
///   The empty sequence is pending whenever any binding exists (every
///   `bound_seq[:0] == ()`), matching the Python fallback's slice
///   comparison.
/// - `("no_match", None)` — neither, so the app resets its key buffer.
///
/// The Python fallback scans the whole bindings dict for the pending check
/// on every keystroke; here both lookups are hash queries against tables
/// precomputed once per binding-table change.
///
/// *bindings* is the `KeyMap.bindings` dict mapping sequences (tuples of
/// key-name strings) to action names. Values may be `Action` members — the
/// str-enum extracts as its value string.
#[pyclass(frozen)]
pub(crate) struct KeyMatcher {
    /// Bound sequence → action name.
    exact: HashMap<Vec<String>, String>,
    /// Every proper prefix (including the empty prefix) of every bound
    /// sequence — the sequences that can still grow into a binding.
    prefixes: HashSet<Vec<String>>,
}

#[pymethods]
impl KeyMatcher {
    #[new]
    fn new(bindings: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut exact: HashMap<Vec<String>, String> = HashMap::new();
        let mut prefixes: HashSet<Vec<String>> = HashSet::new();
        for (seq_obj, action_obj) in bindings.iter() {
            let seq: Vec<String> = seq_obj.extract()?;
            let action: String = action_obj.extract()?;
            for end in 0..seq.len() {
                prefixes.insert(seq[..end].to_vec());
            }
            // Last write wins, like the Python dict the table came from.
            exact.insert(seq, action);
        }
        Ok(Self { exact, prefixes })
    }

    /// Classify *seq* against the binding table.
    /// Returns ("exact", Some(action_name)) | ("pending", None)
    /// | ("no_match", None) — the action-name form of `KeyMap.match`.
    fn match_keys(&self, seq: Vec<String>) -> (String, Option<String>) {
        if let Some(action) = self.exact.get(&seq) {
            return ("exact".to_string(), Some(action.clone()));
        }
        if self.prefixes.contains(&seq) {
            return ("pending".to_string(), None);
        }
        ("no_match".to_string(), None)
    }
}

// ---------------------------------------------------------------------------
// Keymap load plan
// ---------------------------------------------------------------------------

/// `parse_key_sequence` — `tuple(raw.strip().split())` with CPython's
/// whitespace set (see `py_isspace`) on both the strip and the split.
fn parse_key_sequence_py<'py>(py: Python<'py>, raw: &str) -> PyResult<Bound<'py, PyTuple>> {
    let parts: Vec<Bound<'py, PyString>> = py_split_none(py_strip(raw))
        .into_iter()
        .map(|part| PyString::new(py, part))
        .collect();
    PyTuple::new(py, parts)
}

/// Build the load plan behind `KeyMap._load_from_dict` — the section
/// walk, action validation and binding classification of the
/// keymap.toml merge, with the mutations themselves (and the native
/// matcher invalidation) applied by the Python caller as it replays the
/// plan.
///
/// Mirrors the reference: every top-level value that is a dict is walked
/// in insertion order (a non-dict value is skipped, exactly the
/// reference's `isinstance` gate), and each section entry resolves to
/// one plan step — `("unknown", name)` when `Action(name)` raises
/// ValueError (the enum's own `__call__` performs the lookup, so the
/// value matching and any TypeError from an unhashable name are the
/// interpreter's own, the latter propagating exactly where the
/// reference's `except ValueError` lets it), `("invalid", name,
/// type_name)` when the binding value is neither a string nor a list of
/// strings (`type(keys).__name__` carried for the warning), or
/// `("bind", action, sequences)` with the `Action(name)` member and
/// each key string split into its `tuple(raw.strip().split())`
/// sequence. The plan preserves the reference's order, so the warnings
/// interleave with the applications that follow them exactly as the
/// reference emits them.
///
/// The data is the tomllib-parsed mapping (top-level dict of section
/// dicts with string keys); key strings are the valid-Unicode strs a
/// TOML file produces.
#[pyfunction]
pub(crate) fn keymap_load_plan<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    action_cls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let data = data
        .downcast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("keymap data must be a dict"))?;
    let plan = PyList::empty(py);
    for section_value in data.values() {
        // `if not isinstance(section, dict): continue`
        let Ok(section) = section_value.downcast::<PyDict>() else {
            continue;
        };
        for (name, keys) in section.iter() {
            // `try: action = Action(action_name) except ValueError:`
            let action = match action_cls.call1((name.clone(),)) {
                Ok(action) => action,
                Err(err) if err.is_instance_of::<PyValueError>(py) => {
                    let step = PyTuple::new(py, [PyString::new(py, "unknown").into_any(), name])?;
                    plan.append(step)?;
                    continue;
                }
                Err(err) => return Err(err),
            };

            // `_is_valid_binding`: a str, or a list of strs.
            let sequences: Vec<Bound<'py, PyTuple>> = if keys.is_instance_of::<PyString>() {
                let raw = keys.downcast::<PyString>().unwrap();
                vec![parse_key_sequence_py(py, raw.to_str()?)?]
            } else if let Ok(list) = keys.downcast::<PyList>() {
                let mut seqs = Vec::with_capacity(list.len());
                let mut valid = true;
                for element in list.iter() {
                    if !element.is_instance_of::<PyString>() {
                        valid = false;
                        break;
                    }
                    let raw = element.downcast::<PyString>().unwrap();
                    seqs.push(parse_key_sequence_py(py, raw.to_str()?)?);
                }
                if !valid {
                    let step = PyTuple::new(
                        py,
                        [
                            PyString::new(py, "invalid").into_any(),
                            name,
                            keys.get_type().name()?.into_any(),
                        ],
                    )?;
                    plan.append(step)?;
                    continue;
                }
                seqs
            } else {
                let step = PyTuple::new(
                    py,
                    [
                        PyString::new(py, "invalid").into_any(),
                        name,
                        keys.get_type().name()?.into_any(),
                    ],
                )?;
                plan.append(step)?;
                continue;
            };

            let step = PyTuple::new(
                py,
                [
                    PyString::new(py, "bind").into_any(),
                    action,
                    PyList::new(py, sequences)?.into_any(),
                ],
            )?;
            plan.append(step)?;
        }
    }
    Ok(plan)
}
