//! Native cores for `src/ytm_player/services/shuffle_prefs.py` — the
//! `ShufflePrefsCore` bridge class and the shuffle-lock prefs file IO.

use crate::json::json_dumps;
use crate::json::json_loads;
use crate::services::auth::JsonReject;
use crate::services::ytmusic::py_key_position;
use crate::services::ytmusic::py_unpack_pair;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

/// The LRU bookkeeping engine behind
/// `services/shuffle_prefs.py`'s `ShufflePreferences` — the ordered
/// `{context_id: shuffle}` map. Mirrors the pure-Python reference's
/// `OrderedDict` state exactly:
///
/// - `get` — the lookup key hashed first (`dict.get`'s own TypeError for
///   an unhashable), then the identity-or-`==` scan; a hit promotes the
///   entry to most-recent (the reference's `move_to_end`) and returns
///   its value, a miss leaves the state untouched and answers `None`.
/// - `set` — the key hashed first (`__setitem__`'s own TypeError), an
///   existing entry's value updated in place (the stored key object kept,
///   never replaced — the reference's assignment semantics) and then
///   promoted to most-recent (the reference's explicit `move_to_end`),
///   a fresh key appended at the end; then the eviction walk —
///   `while len > max: popitem(last=False)` — the oldest entries dropped
///   until the cap holds.
/// - `clear` — the state emptied.
/// - `load` — the hand-edited-file ingestion: a non-dict payload leaves
///   the state untouched (the reference's `isinstance(data, dict)`
///   gate), a dict is walked through its own `items()` method (a
///   subclass override steers this walk exactly like it steered the
///   comprehension) with the non-str keys dropped (`isinstance(k, str)`),
///   every surviving value coerced through Python truthiness (`bool(v)`
///   — `is_truthy`, CPython's own `__bool__`/`__len__` protocol), each
///   item unpacked through the `for k, v in ...` contract (see
///   `py_unpack_pair`), and the whole mapping built before the state is
///   swapped — a mid-walk failure leaves the old state in place, exactly
///   like the reference's partially-built (and then discarded)
///   OrderedDict.
/// - `dump_json` — the lock-held snapshot + serialisation of the
///   reference's `_save` (`dict(self._prefs)` then
///   `json.dumps(data, indent=2)`): the LRU order rendered through the
///   already-ported `json_dumps` engine with the reference's `indent=2`,
///   so every key coercion (int keys stringified, bool/None keys
///   rendered as the JSON encoder does) and every escape is CPython's
///   own; a non-coercible key raises the encoder's own `TypeError`.
///
/// The thread lock and the file I/O stay in the Python facade; the
/// cap is the reference's `_MAX_ENTRIES` (1000). A negative cap clamps
/// to zero — the reference's own negative-cap loop would pop from an
/// empty dict and crash with KeyError, an input the constant never
/// produces.
#[pyclass]
pub(crate) struct ShufflePrefsCore {
    /// The ordered `(key, value)` pairs, least-recent first.
    entries: Vec<(Py<PyAny>, bool)>,
    /// The entry cap (`_MAX_ENTRIES`).
    max_entries: usize,
}

#[pymethods]
impl ShufflePrefsCore {
    #[new]
    fn new(max_entries: i64) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(0) as usize,
        }
    }

    /// The saved shuffle state for *context_id*, or `None` — promoted to
    /// most-recent on a hit (the reference's `move_to_end`).
    fn get<'py>(
        &mut self,
        py: Python<'py>,
        context_id: &Bound<'py, PyAny>,
    ) -> PyResult<Option<bool>> {
        // dict.get hashes the lookup key before anything else.
        let _ = context_id.hash()?;
        match py_key_position(py, &self.entries, context_id)? {
            Some(i) => {
                let value = self.entries[i].1;
                // move_to_end — the hit becomes most-recent.
                let entry = self.entries.remove(i);
                self.entries.push(entry);
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Persist *shuffle* for *context_id* — the entry promoted to
    /// most-recent, the overflow evicted oldest-first.
    fn set<'py>(
        &mut self,
        py: Python<'py>,
        context_id: &Bound<'py, PyAny>,
        shuffle: bool,
    ) -> PyResult<()> {
        // __setitem__ hashes the key before anything else.
        let _ = context_id.hash()?;
        match py_key_position(py, &self.entries, context_id)? {
            Some(i) => {
                // The assignment updates the value in place (the stored
                // key object kept), then move_to_end promotes the entry.
                self.entries[i].1 = shuffle;
                let entry = self.entries.remove(i);
                self.entries.push(entry);
            }
            None => self.entries.push((context_id.clone().unbind(), shuffle)),
        }
        // while len(self._prefs) > _MAX_ENTRIES: popitem(last=False)
        while self.entries.len() > self.max_entries {
            if self.entries.is_empty() {
                break;
            }
            self.entries.remove(0);
        }
        Ok(())
    }

    /// Empty the state (`clear`).
    fn clear(&mut self) {
        self.entries.clear();
    }

    /// Ingest the parsed prefs file (see the class docs). Not a dict →
    /// no state change.
    fn load<'py>(&mut self, data: &Bound<'py, PyAny>) -> PyResult<()> {
        if !data.is_instance_of::<PyDict>() {
            return Ok(());
        }
        // The comprehension's data.items() — the method call, so a
        // subclass override steers the walk.
        let items = data.call_method0("items")?;
        let mut fresh: Vec<(Py<PyAny>, bool)> = Vec::new();
        for item in items.try_iter()? {
            let item = item?;
            let (k, v) = py_unpack_pair(&item)?;
            // if isinstance(k, str) — the non-str keys are dropped
            // before the value is ever coerced.
            if !k.is_instance_of::<PyString>() {
                continue;
            }
            // bool(v) — Python truthiness.
            fresh.push((k.clone().unbind(), v.is_truthy()?));
        }
        // The comprehension built the whole mapping before the
        // assignment; a failure above leaves the old state intact.
        self.entries = fresh;
        Ok(())
    }

    /// The `_save` snapshot + serialisation: the LRU order rendered as
    /// `json.dumps(dict(self._prefs), indent=2)` renders it (through the
    /// ported JSON encoder, so every key coercion and escape is
    /// CPython's own).
    fn dump_json(&self, py: Python) -> PyResult<String> {
        let snapshot = PyDict::new(py);
        for (k, v) in &self.entries {
            snapshot.set_item(k.bind(py), *v)?;
        }
        json_dumps(py, snapshot.as_any(), Some(2))
    }

    /// The number of live entries.
    fn length(&self) -> usize {
        self.entries.len()
    }
}

/// The read step behind `ShufflePreferences._load` — the pure core of
/// the prefs file's disk I/O.
///
/// Mirrors the pure-Python transcription (`_read_prefs_file`):
/// `path.read_text(encoding="utf-8")` — the read driven through the
/// *path* object's own method, so a `Path`, a `Path` subclass or a stub
/// steers both backends identically and every filesystem error shape
/// (the `OSError` subclasses with their filename, the
/// `UnicodeDecodeError` of an undecodable file) is CPython's own — and
/// then `json.loads` over the read result through the ported
/// CPython-grammar scanner. A scanner rejection — and every non-str
/// result a stub's `read_text` might answer — falls through to the
/// interpreter's own `json.loads` call over the same object, the same
/// fallback the auth record parser uses for its exotic-decode path, so
/// the error the caller sees (the `JSONDecodeError` class, message,
/// `pos`/`lineno`/`colno`, or the `TypeError` of an unparseable shape)
/// is exactly the reference's own; the scanner's accept/reject boundary
/// mirrors `json.loads`'s, so the re-ask raises where the reference
/// raises and the parity matrix pins the two together. The exists gate
/// and the logging stay in the facade; the returned value passes
/// through as the object the parse built.
#[pyfunction]
pub(crate) fn shuffle_prefs_read_file<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // self._path.read_text(encoding="utf-8") — through the object's
    // own method, the keyword spelled the reference's way.
    let kwargs = PyDict::new(py);
    kwargs.set_item("encoding", "utf-8")?;
    let text = path.getattr("read_text")?.call((), Some(&kwargs))?;
    // json.loads(text) — the ported scanner over a str, the
    // interpreter's own json.loads for every other shape.
    match text.downcast::<PyString>() {
        Ok(text) => match json_loads(py, text) {
            Ok(value) => Ok(value),
            Err(JsonReject::Py(err)) => Err(err),
            Err(JsonReject::Reject) => py.import("json")?.getattr("loads")?.call1((text,)),
        },
        Err(_) => py.import("json")?.getattr("loads")?.call1((text,)),
    }
}

/// The write step behind `ShufflePreferences._save` — the pure core of
/// the prefs file's disk I/O.
///
/// Mirrors the pure-Python transcription (`_write_prefs_file`), every
/// operation driven through the *path* object's own protocol so a
/// `Path`, a `Path` subclass or a stub steers both backends
/// identically and every error shape is CPython's own:
///
/// - `path.parent.mkdir(parents=True, exist_ok=True)` — the parent
///   property read, then the directory creation through the parent's
///   own method with both keywords spelled the reference's way.
/// - `path.with_suffix(path.suffix + ".tmp")` — the suffix read
///   through the property, the `+ ".tmp"` through CPython's own
///   operator (`operator.add`, so a str subclass's overridden
///   `__add__` steers both backends identically), and the sibling
///   built through the path's own `with_suffix` (its `ValueError`
///   shapes for the degenerate names included).
/// - `tmp.write_text(text, encoding="utf-8")` — the serialised state
///   written through the sibling's own method, the text passed through
///   as the object it is.
/// - `tmp.replace(path)` — the atomic cutover through the sibling's
///   own `os.replace` delegation, the final path passed through as
///   the object it is.
///
/// The snapshot + serialisation (the core's `dump_json`) and the
/// logging stay in the facade; the statement order is the reference's
/// own, so a failure leaves exactly the partial state the reference's
/// would.
#[pyfunction]
pub(crate) fn shuffle_prefs_write_file<'py>(
    py: Python<'py>,
    path: &Bound<'py, PyAny>,
    text: &Bound<'py, PyAny>,
) -> PyResult<()> {
    // self._path.parent.mkdir(parents=True, exist_ok=True)
    let parent = path.getattr("parent")?;
    let mkdir_kwargs = PyDict::new(py);
    mkdir_kwargs.set_item("parents", true)?;
    mkdir_kwargs.set_item("exist_ok", true)?;
    parent.getattr("mkdir")?.call((), Some(&mkdir_kwargs))?;
    // tmp = self._path.with_suffix(self._path.suffix + ".tmp")
    let suffix = path.getattr("suffix")?;
    let add = py.import("operator")?.getattr("add")?;
    let new_suffix = add.call1((suffix, ".tmp"))?;
    let tmp = path.getattr("with_suffix")?.call1((new_suffix,))?;
    // tmp.write_text(text, encoding="utf-8")
    let write_kwargs = PyDict::new(py);
    write_kwargs.set_item("encoding", "utf-8")?;
    tmp.getattr("write_text")?
        .call((text,), Some(&write_kwargs))?;
    // tmp.replace(self._path)
    tmp.getattr("replace")?.call1((path,))?;
    Ok(())
}
