//! Native cores for `src/ytm_player/services/yt_dlp_options.py` — the
//! configured-opts normalization (remote components, JS runtimes).

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};

// ---------------------------------------------------------------------------
// yt-dlp option normalisation (mirrors ytm_player/services/yt_dlp_options.py)
// ---------------------------------------------------------------------------

/// Split a string by commas/whitespace and drop empties. Mirrors
/// `yt_dlp_options._split_csv_or_space`.
fn split_csv_or_space(value: &str) -> Vec<String> {
    value
        .replace(',', " ")
        .split_whitespace()
        .map(String::from)
        .collect()
}

/// Parse a `runtime[:path]` token into (name, optional path). Mirrors
/// `yt_dlp_options._parse_runtime_token`: an empty/whitespace spec or an
/// empty runtime name yields None; a colon-delimited path is kept only when
/// it is non-blank after stripping.
fn parse_runtime_token(spec: &str) -> Option<(String, Option<String>)> {
    let token = spec.trim();
    if token.is_empty() {
        return None;
    }
    let (runtime, path) = match token.split_once(':') {
        Some((head, tail)) => (head, Some(tail)),
        None => (token, None),
    };
    let name = runtime.trim().to_lowercase();
    if name.is_empty() {
        return None;
    }
    match path.map(str::trim).filter(|p| !p.is_empty()) {
        Some(p) => Some((name, Some(p.to_string()))),
        None => Some((name, None)),
    }
}

/// Return a yt-dlp compatible remote_components list, or None when unset or
/// blank. Mirrors `yt_dlp_options.normalize_remote_components`: a string is
/// split on commas/whitespace; any other non-None value is iterated and each
/// item is coerced with str() then stripped (empty results dropped).
#[pyfunction]
pub(crate) fn normalize_remote_components<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyList>>> {
    if value.is_none() {
        return Ok(None);
    }
    let parts: Vec<String> = if let Ok(s) = value.extract::<String>() {
        split_csv_or_space(&s)
    } else {
        let mut parts = Vec::new();
        for item in value.try_iter()? {
            let item = item?;
            let text = item.str()?.to_str()?.trim().to_string();
            if !text.is_empty() {
                parts.push(text);
            }
        }
        parts
    };
    if parts.is_empty() {
        return Ok(None);
    }
    Ok(Some(PyList::new(py, parts)?))
}

/// Return a yt-dlp compatible js_runtimes dict, or None when unset or blank.
/// Mirrors `yt_dlp_options.normalize_js_runtimes`: a dict maps
/// lowercased/stripped names to their config sub-dicts verbatim (non-dict
/// configs become {}); a string or iterable of `runtime[:path]` tokens is
/// parsed into {"name": {"path": ...}} entries. Insertion order and the
/// later-duplicate-wins behaviour match Python dict semantics.
#[pyfunction]
pub(crate) fn normalize_js_runtimes<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    if value.is_none() {
        return Ok(None);
    }
    let result = PyDict::new(py);

    if let Ok(dict) = value.downcast::<PyDict>() {
        for (runtime, config) in dict.iter() {
            let name = runtime.str()?.to_str()?.trim().to_lowercase();
            if name.is_empty() {
                continue;
            }
            if config.is_instance_of::<PyDict>() {
                result.set_item(name, config)?;
            } else {
                result.set_item(name, PyDict::new(py))?;
            }
        }
    } else {
        let specs: Vec<String> = if let Ok(s) = value.extract::<String>() {
            split_csv_or_space(&s)
        } else {
            let mut specs = Vec::new();
            for item in value.try_iter()? {
                let item = item?;
                specs.push(item.str()?.to_str()?.to_string());
            }
            specs
        };
        for spec in specs {
            if let Some((name, path)) = parse_runtime_token(&spec) {
                let config = PyDict::new(py);
                if let Some(p) = path {
                    config.set_item("path", p)?;
                }
                result.set_item(name, config)?;
            }
        }
    }

    if result.is_empty() {
        return Ok(None);
    }
    Ok(Some(result))
}

/// The configured-extras layering — `yt_dlp_options.py`'s
/// `apply_configured_yt_dlp_options` core, the second half of
/// `StreamResolver._build_ydl_opts`'s opts build (the first half is
/// `stream_base_opts`).
///
/// Mirrors the reference's mutation sequence branch for branch, over
/// the four already-normalised setting values the caller read in the
/// reference's own order (`normalize_cookiefile` /
/// `normalize_cafile` / `normalize_remote_components` /
/// `normalize_js_runtimes` — the path expanders stay Python-side;
/// the latter two are themselves native engines):
///
/// - a truthy *cookies_file* (CPython's own truthiness protocol)
///   writes `opts["cookiefile"]` through the subscript protocol, the
///   value the very object it is;
/// - a truthy *ca_bundle* first probes the file through the
///   zero-argument *is_file* thunk — invoked exactly where the
///   reference's `Path(ca_bundle).is_file()` ran, so a falsy bundle
///   never probes and a raising probe propagates before any mutation
///   — and warns through the *warn* thunk when it came back falsy,
///   the reference's exact `("Configured [yt_dlp] ca_bundle does not
///   exist: %s", ca_bundle)` argument pair; then the `compat_opts`
///   read through the mapping's own `.get` (a fresh empty list as the
///   default) copied through the interpreter's own `list()` — so the
///   stored list object is never mutated — the `"no-certifi"` gate
///   through CPython's own containment, the survivor appended through
///   the list's own `.append`, and the result written back through
///   `set(compat)` — the interpreter's own set constructor, so an
///   unhashable member raises the constructor's own `TypeError` at
///   exactly the reference's point (after the missing-file warning,
///   which the thunk timing preserves);
/// - a truthy *remote_components* warns through the *warn* thunk with
///   the reference's exact argument pair — the second argument the
///   `", "`-join through the str's own `.join` method — then writes
///   `opts["remote_components"]` as the very object it is;
/// - a truthy *js_runtimes* writes `opts["js_runtimes"]` as the very
///   object it is.
///
/// The *warn* thunk is `logger.warning` itself in production, so the
/// warnings fire at exactly the reference's points in the sequence —
/// interleaved with the mutations, not replayed after them. The
/// settings-object attribute reads and their normalisations stay
/// Python-side in `apply_configured_yt_dlp_options`, evaluated up
/// front in the reference's own read order; their interleaving with
/// the mutation sequence is not observable (the reads are
/// side-effect-free dataclass field accesses). The answer is *opts*
/// itself, the object it is.
#[allow(clippy::too_many_arguments)] // the reference's four extras, plus the two thunk seams
#[pyfunction]
#[pyo3(signature = (opts, cookies_file, ca_bundle, is_file, warn, remote_components, js_runtimes))]
pub(crate) fn yt_dlp_configured_opts<'py>(
    py: Python<'py>,
    opts: &Bound<'py, PyAny>,
    cookies_file: &Bound<'py, PyAny>,
    ca_bundle: &Bound<'py, PyAny>,
    is_file: &Bound<'py, PyAny>,
    warn: &Bound<'py, PyAny>,
    remote_components: &Bound<'py, PyAny>,
    js_runtimes: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if cookies_file.is_truthy()? {
        opts.set_item("cookiefile", cookies_file)?;
    }

    if ca_bundle.is_truthy()? {
        // if not Path(ca_bundle).is_file(): logger.warning(...)
        let file_ok = is_file.call0()?;
        if !file_ok.is_truthy()? {
            warn.call1((
                "Configured [yt_dlp] ca_bundle does not exist: %s",
                ca_bundle,
            ))?;
        }
        // compat = list(opts.get("compat_opts", []))
        let empty = PyList::empty(py).into_any();
        let raw_compat = opts.call_method1("get", ("compat_opts", empty))?;
        let list_type = py.import("builtins")?.getattr("list")?;
        let compat = list_type.call1((raw_compat,))?;
        // if "no-certifi" not in compat: compat.append("no-certifi")
        if !compat.contains("no-certifi")? {
            compat.call_method1("append", ("no-certifi",))?;
        }
        // opts["compat_opts"] = set(compat)
        let set_type = py.import("builtins")?.getattr("set")?;
        let compat_set = set_type.call1((compat,))?;
        opts.set_item("compat_opts", compat_set)?;
    }

    if remote_components.is_truthy()? {
        // logger.warning("...: %s", ", ".join(remote_components))
        let joiner = PyString::new(py, ", ");
        let joined = joiner.call_method1("join", (remote_components,))?;
        warn.call1((
            "yt-dlp remote_components is enabled; this allows remote JavaScript component downloads: %s",
            joined,
        ))?;
        opts.set_item("remote_components", remote_components)?;
    }

    if js_runtimes.is_truthy()? {
        opts.set_item("js_runtimes", js_runtimes)?;
    }

    Ok(opts.clone())
}
