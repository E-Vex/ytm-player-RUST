//! Native cores for `src/ytm_player/services/auth.py` — identity
//! records, channel-id extraction, raw-header normalization and the
//! stream cookie jar assembly (SHA-256 candidate payloads).

use crate::app::keys::py_isdigit;
use crate::json::json_encode;
use crate::json::json_loads;
use crate::utils::pycompat::py_isspace;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyUnicodeDecodeError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use regex::Regex;
use sha2::{Digest, Sha256};
use std::sync::LazyLock;

/// A YouTube channel ID — `UC` + 22 URL-safe base64 characters, the
/// reference's `re.fullmatch` anchored at both ends (a trailing newline
/// cannot match the class, so the anchored form is the fullmatch).
static CHANNEL_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^UC[A-Za-z0-9_-]{22}$").expect("valid channel id regex"));

/// `str.strip()` with no arguments — trimmed of the `isspace` set's
/// leading and trailing characters (see `py_isspace` — a wider set than
/// the crate's own `str::trim`, which stops at the Unicode
/// `White_Space` property and misses `\x1c`–`\x1f`).
pub(crate) fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

// ---------------------------------------------------------------------------
// Auth session-record engines (services/auth.py)
// ---------------------------------------------------------------------------

/// `hashlib.sha256(payload).hexdigest()` — the hash primitive behind
/// every session-record engine. Bytes (and their subclasses) digest
/// through the Rust implementation of the same SHA-256; every other
/// shape falls back to the interpreter's own
/// `hashlib.sha256(payload).hexdigest()` call — a bytearray or
/// memoryview hashes identically, a str payload raises hashlib's own
/// `TypeError` ("Unicode-objects must be encoded before hashing")
/// exactly like the reference's call would.
fn sha256_hex_impl(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(bytes) = payload.downcast::<PyBytes>() {
        let digest = Sha256::digest(bytes.as_bytes());
        return Ok(hex_lower(&digest));
    }
    let hashlib = py.import("hashlib")?;
    let digest = hashlib.getattr("sha256")?.call1((payload,))?;
    digest.getattr("hexdigest")?.call0()?.extract::<String>()
}

/// Lowercase hex — `bytes.hex()`'s own spelling of a digest.
fn hex_lower(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for b in data {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// A decode rejection — the two failure shapes `json.loads` surfaces
/// through the engines. Every `JSONDecodeError` the scanner can raise
/// collapses to `Reject`: the engines swallow the message (the
/// reference's except arm answers `None`), so the accept/reject
/// boundary is the whole observable contract and CPython's positions
/// and spellings never surface. `Py` carries the errors the reference
/// lets propagate — the `RecursionError` of past-budget nesting, an
/// `int()` conversion's own `ValueError`.
pub(crate) enum JsonReject {
    /// A `json.JSONDecodeError`-shaped rejection.
    Reject,
    /// A propagating Python error raised during object construction.
    Py(PyErr),
}

/// One decoded container frame — the explicit stack that keeps the
/// walk free of Rust recursion: a payload nested past CPython's own
/// recursion budget raises the interpreter's `RecursionError` through
/// the depth counter instead of overflowing the Rust stack.
pub(crate) enum JsonLoadFrame {
    Array {
        items: Vec<Py<PyAny>>,
    },
    Object {
        pairs: Vec<(Py<PyAny>, Py<PyAny>)>,
        /// The key whose value the scanner sits between `:` and.
        pending_key: Option<Py<PyAny>>,
    },
}

/// The pure core of `services/auth.py`'s `_parse_record` — shared by
/// the exported engine and the composite engines below.
///
/// Mirrors the pure-Python reference: `raw.decode("utf-8")` through the
/// object's own decode method (a `UnicodeDecodeError` is the engine's
/// `None`; a non-decodable payload's `AttributeError` propagates),
/// `json.loads` over the decoded text (the scanner above — every
/// `JSONDecodeError` shape the engine's `None`), then the schema gates:
/// the top value must be a dict and its `schema_version` must compare
/// equal to the constant through CPython's own `==` — so `1.0` and
/// `True` pass exactly like they do in the reference. A decode override
/// returning a non-str falls through to the interpreter's own
/// `json.loads` call so that exotic path keeps the reference's exact
/// behavior, its `JSONDecodeError`/`UnicodeDecodeError` answered with
/// `None` and everything else propagating.
fn parse_record_impl<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if raw.is_none() {
        return Ok(None);
    }
    let decoded = match raw.getattr("decode").and_then(|d| d.call1(("utf-8",))) {
        Ok(d) => d,
        Err(err) => {
            if err.is_instance_of::<PyUnicodeDecodeError>(py) {
                return Ok(None);
            }
            return Err(err);
        }
    };
    let value = match decoded.downcast::<PyString>() {
        Ok(text) => match json_loads(py, text) {
            Ok(v) => v,
            Err(JsonReject::Reject) => return Ok(None),
            Err(JsonReject::Py(err)) => return Err(err),
        },
        Err(_) => {
            let json_module = py.import("json")?;
            let loads = json_module.getattr("loads")?;
            let jde_class = json_module.getattr("JSONDecodeError")?;
            match loads.call1((decoded.clone(),)) {
                Ok(v) => v,
                Err(err) => {
                    let is_json_error = jde_class.is_instance(err.value(py)).unwrap_or(false);
                    if is_json_error || err.is_instance_of::<PyUnicodeDecodeError>(py) {
                        return Ok(None);
                    }
                    return Err(err);
                }
            }
        }
    };
    if !value.is_instance_of::<PyDict>() {
        return Ok(None);
    }
    // data.get("schema_version") — the dict's own get, its None the
    // missing-key answer, compared through CPython's `!=`.
    let schema = value.getattr("get")?.call1(("schema_version",))?;
    if schema.ne(1i64)? {
        return Ok(None);
    }
    Ok(Some(value))
}

/// The account.json record parser — the engine behind every read of
/// the sign-in sidecar.
///
/// Mirrors the pure-Python reference (`_parse_record`): account.json
/// bytes as a dict — or `None` when the file is missing (the `None`
/// passthrough), malformed, of another schema, or undecodable — with
/// the JSON grammar, the schema gate and the exact comparison quirks
/// described on `parse_record_impl`. The file read stays Python-side
/// in the manager.
#[pyfunction]
#[pyo3(signature = (raw))]
pub(crate) fn parse_record<'py>(
    py: Python<'py>,
    raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    parse_record_impl(py, raw)
}

/// The recorded-identity validation — the engine behind the renewal
/// gate that ties an automatic refresh to the account `ytm setup`
/// chose.
///
/// Mirrors the pure-Python reference (`_identity_from_record`): the
/// `verified` field read through CPython's own subscript and gated by
/// identity with the `False` singleton — `is` semantics, so a `0` or
/// any other falsy-but-not-`False` object passes exactly like it does
/// in the reference; the `channel_id` read next, accepted only when it
/// is a str whose whole value matches the channel-id pattern
/// (a str carrying lone surrogates cannot match the class, answering
/// like the reference's fullmatch); the `x-goog-authuser` slot read
/// last, accepted only when it is a str of digits under CPython's own
/// `str.isdigit()` rule (Unicode digits included) and converted
/// through the interpreter's own `int()` — so `int('٣')` keeps its
/// value and an isdigit-true, int-false spelling (a superscript digit)
/// raises `int()`'s own `ValueError` exactly like the reference. The
/// result is the `(slot, channel_id)` pair with both objects passed
/// through as they are, or `None`.
#[pyfunction]
#[pyo3(signature = (record))]
pub(crate) fn identity_from_record<'py>(
    py: Python<'py>,
    record: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if record.is_none() {
        return Ok(None);
    }
    let get = |key: &str| -> PyResult<Bound<'py, PyAny>> { record.getattr("get")?.call1((key,)) };
    let verified = get("verified")?;
    let false_singleton = PyBool::new(py, false).to_owned();
    if verified.is(&false_singleton) {
        return Ok(None);
    }
    let channel_id = get("channel_id")?;
    let slot = get("x-goog-authuser")?;
    let channel_ok = match channel_id.downcast::<PyString>() {
        Ok(s) => s
            .to_str()
            .map(|t| CHANNEL_ID_RE.is_match(t))
            .unwrap_or(false),
        Err(_) => false,
    };
    if !channel_ok {
        return Ok(None);
    }
    let slot_ok = match slot.downcast::<PyString>() {
        Ok(s) => s.to_str().map(py_isdigit).unwrap_or(false),
        Err(_) => false,
    };
    if !slot_ok {
        return Ok(None);
    }
    let slot_int = py.import("builtins")?.getattr("int")?.call1((slot,))?;
    Ok(Some(PyTuple::new(py, [slot_int, channel_id.clone()])?))
}

/// The ownership snapshot — the engine behind the commit guard that
/// refuses an automatic renewal when the sign-in files changed while
/// it was being renewed.
///
/// Mirrors the pure-Python reference (`_owner_of`): no auth.json means
/// no owner (`None`); otherwise the record parses, the auth bytes hash
/// through the SHA-256 primitive, and the record's `revision` is
/// answered only when the record is truthy and its `auth_sha256`
/// compares equal to that hash through CPython's own `==` — and only
/// when the revision itself is a str. The result is the
/// `(auth_sha256, revision)` pair (the revision `None` when absent),
/// the hash a fresh hex str.
#[pyfunction]
#[pyo3(signature = (auth_bytes, record_raw))]
pub(crate) fn owner_of<'py>(
    py: Python<'py>,
    auth_bytes: &Bound<'py, PyAny>,
    record_raw: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if auth_bytes.is_none() {
        return Ok(None);
    }
    let record = parse_record_impl(py, record_raw)?;
    let sha = PyString::new(py, &sha256_hex_impl(py, auth_bytes)?);
    let revision = match &record {
        Some(record) if record.is_truthy()? => {
            let recorded = record.getattr("get")?.call1(("auth_sha256",))?;
            if recorded.eq(&sha)? {
                let revision = record.getattr("get")?.call1(("revision",))?;
                if revision.is_instance_of::<PyString>() {
                    revision
                } else {
                    py.None().into_bound(py)
                }
            } else {
                py.None().into_bound(py)
            }
        }
        _ => py.None().into_bound(py),
    };
    Ok(Some(PyTuple::new(py, [sha.into_any(), revision])?))
}

/// The stream-jar vouching decision tree — the engine behind
/// `read_vouched_stream_jar`'s refusal reasons.
///
/// Mirrors the pure-Python reference (`_vouch_failure`) read for read:
/// a jar without data (missing and unreadable alike) is unreadable; an
/// auth.json error is unreadable; missing auth.json has no trust; an
/// account.json error renders through the interpreter's own `str()`
/// into its refusal reason; a missing record is the accepted legacy
/// shape; a record that does not parse is malformed; a record whose
/// `auth_sha256` does not compare equal to the hash of the auth bytes
/// (CPython's own `!=`) does not describe them; a record without the
/// `stream_sha256` key is the other accepted legacy shape; a null hash
/// means the sign-in has no jar; and a hash that does not compare equal
/// to the jar's is a mismatch. `None` vouches for the jar. The file
/// reads (and their stat triples) stay Python-side in the caller.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the three reads' flat (data, error) pairs
#[pyo3(signature = (auth_data, auth_error, record_data, record_error, jar_data, jar_error))]
pub(crate) fn vouch_failure<'py>(
    py: Python<'py>,
    auth_data: &Bound<'py, PyAny>,
    auth_error: &Bound<'py, PyAny>,
    record_data: &Bound<'py, PyAny>,
    record_error: &Bound<'py, PyAny>,
    jar_data: &Bound<'py, PyAny>,
    jar_error: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    // The jar's error component is never read — the reference gates on
    // the jar's data alone (missing and unreadable alike).
    let _ = jar_error;
    if jar_data.is_none() {
        return Ok(Some("the stream cookie file is unreadable".to_string()));
    }
    if !auth_error.is_none() {
        return Ok(Some("auth.json is unreadable".to_string()));
    }
    if auth_data.is_none() {
        return Ok(Some("auth.json is missing".to_string()));
    }
    if !record_error.is_none() {
        let rendered = py
            .import("builtins")?
            .getattr("str")?
            .call1((record_error,))?;
        let rendered: String = rendered.extract()?;
        return Ok(Some(format!("account.json cannot be read ({rendered})")));
    }
    if record_data.is_none() {
        return Ok(None);
    }
    let Some(record) = parse_record_impl(py, record_data)? else {
        return Ok(Some("account.json is malformed".to_string()));
    };
    let auth_sha = PyString::new(py, &sha256_hex_impl(py, auth_data)?);
    let recorded_sha = record.getattr("get")?.call1(("auth_sha256",))?;
    if recorded_sha.ne(&auth_sha)? {
        return Ok(Some(
            "account.json does not describe the current auth.json".to_string(),
        ));
    }
    if !record.contains("stream_sha256")? {
        return Ok(None);
    }
    // record_dict["stream_sha256"] — the plain subscript the `in`
    // check just vouched for.
    let expected = record.get_item("stream_sha256")?;
    if expected.is_none() {
        return Ok(Some(
            "the saved sign-in has no stream cookie file".to_string(),
        ));
    }
    let jar_sha = PyString::new(py, &sha256_hex_impl(py, jar_data)?);
    if expected.ne(&jar_sha)? {
        return Ok(Some(
            "the stream cookie file does not match the saved sign-in".to_string(),
        ));
    }
    Ok(None)
}

/// The account_menu channel-id extraction — the engine behind the
/// setup probe that learns which YouTube channel a browser slot's
/// session belongs to.
///
/// Mirrors the pure-Python reference
/// (`_channel_id_from_account_menu`): the lookup chain through
/// CPython's own subscript with the reference's caught arms
/// (`KeyError`, `IndexError`, `TypeError`) answered with `None` and
/// every other error kept; the header gate (a non-dict menu, or an
/// `activeAccountHeaderRenderer` missing from the header's own
/// `in`); the items gate (exactly a list); then the defensive walk —
/// each item's `compactLinkRenderer` fetched through the dict's own
/// `get` when the item is a dict, the ACCOUNT_BOX icon, the
/// navigation endpoint's browse endpoint, the browse id validated by
/// the channel-id pattern, and the page type read through the
/// reference's `(configs or {}).get(..., {}).get("pageType")` chain —
/// the id check and the `==` compared with CPython's own operators,
/// the page-type `==` evaluated only when the id matched (the
/// reference's `and` chain short-circuits before it). Exactly one
/// matching link answers its browse id; zero or several answer `None`
/// — the caller fails closed.
#[pyfunction]
#[pyo3(signature = (response))]
pub(crate) fn channel_id_from_account_menu<'py>(
    py: Python<'py>,
    response: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyString>>> {
    let chain: PyResult<Option<Bound<'py, PyAny>>> = (|| {
        let menu = response
            .get_item("actions")?
            .get_item(0)?
            .get_item("openPopupAction")?
            .get_item("popup")?
            .get_item("multiPageMenuRenderer")?;
        if !menu.is_instance_of::<PyDict>() {
            return Ok(None);
        }
        let header = menu.get_item("header")?;
        if !header.contains("activeAccountHeaderRenderer")? {
            return Ok(None);
        }
        Ok(Some(
            menu.get_item("sections")?
                .get_item(0)?
                .get_item("multiPageMenuSectionRenderer")?
                .get_item("items")?,
        ))
    })();
    let items = match chain {
        Ok(Some(items)) => items,
        Ok(None) => return Ok(None),
        Err(err) => {
            if err.is_instance_of::<PyKeyError>(py)
                || err.is_instance_of::<PyIndexError>(py)
                || err.is_instance_of::<PyTypeError>(py)
            {
                return Ok(None);
            }
            return Err(err);
        }
    };
    if !items.is_instance_of::<PyList>() {
        return Ok(None);
    }
    let mut found: Vec<Bound<'py, PyString>> = Vec::new();
    for item in items.try_iter()? {
        let item = item?;
        let link = if item.is_instance_of::<PyDict>() {
            item.getattr("get")?.call1(("compactLinkRenderer",))?
        } else {
            py.None().into_bound(py)
        };
        if !link.is_instance_of::<PyDict>() {
            continue;
        }
        let icon = link.getattr("get")?.call1(("icon",))?;
        if !icon.is_instance_of::<PyDict>()
            || icon
                .getattr("get")?
                .call1(("iconType",))?
                .ne("ACCOUNT_BOX")?
        {
            continue;
        }
        let endpoint = link.getattr("get")?.call1(("navigationEndpoint",))?;
        let browse = if endpoint.is_instance_of::<PyDict>() {
            endpoint.getattr("get")?.call1(("browseEndpoint",))?
        } else {
            py.None().into_bound(py)
        };
        if !browse.is_instance_of::<PyDict>() {
            continue;
        }
        let browse_id = browse.getattr("get")?.call1(("browseId",))?;
        let configs = browse
            .getattr("get")?
            .call1(("browseEndpointContextSupportedConfigs",))?;
        let configs = if configs.is_truthy()? {
            configs
        } else {
            PyDict::new(py).into_any()
        };
        let music = configs
            .getattr("get")?
            .call1(("browseEndpointContextMusicConfig", PyDict::new(py)))?;
        let page_type = music.getattr("get")?.call1(("pageType",))?;
        if let Ok(id) = browse_id.downcast::<PyString>() {
            if id
                .to_str()
                .map(|t| CHANNEL_ID_RE.is_match(t))
                .unwrap_or(false)
                && page_type.eq("MUSIC_PAGE_TYPE_USER_CHANNEL")?
            {
                found.push(id.clone());
            }
        }
    }
    if found.len() != 1 {
        return Ok(None);
    }
    Ok(Some(found.into_iter().next().expect("len checked above")))
}

/// The canonical auth.json serialisation — the one byte spelling of a
/// header dict, probed and committed as is.
///
/// Mirrors the pure-Python reference (`_serialize_headers`):
/// `json.dumps(headers, ensure_ascii=True, indent=4, sort_keys=True)`
/// — the strict encoder (the reference's own `TypeError` for an
/// unserialisable value or an unorderable key), the sorted-key layout
/// through CPython's own tuple comparison, the four-space indent with
/// the `(',', ': ')` separators, and the ASCII escaping — encoded to
/// UTF-8 bytes (the ASCII-only text's own encoding).
#[pyfunction]
#[pyo3(signature = (headers))]
pub(crate) fn serialize_headers(
    py: Python<'_>,
    headers: &Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    let text = json_encode(py, headers, "    ".to_string(), true, true, true)?;
    Ok(PyBytes::new(py, text.as_bytes()).unbind())
}

/// The candidate auth.json bytes for one browser slot — the payload
/// the setup probe writes and commits unchanged.
///
/// Mirrors the pure-Python reference (`_candidate_payload`):
/// `{**base_headers, "x-goog-authuser": str(slot)}` — the dict merge
/// (a dict walked through its own storage, any other mapping through
/// its `keys()`, a repeated key keeping its position with the new
/// value), the slot rendered through the interpreter's own `str()`,
/// then the canonical serialisation above.
#[pyfunction]
#[pyo3(signature = (base_headers, slot))]
pub(crate) fn candidate_payload(
    py: Python<'_>,
    base_headers: &Bound<'_, PyAny>,
    slot: &Bound<'_, PyAny>,
) -> PyResult<Py<PyBytes>> {
    let merged = PyDict::new(py);
    if base_headers.is_instance_of::<PyDict>() {
        let base = base_headers
            .downcast::<PyDict>()
            .expect("isinstance-checked dict");
        for (key, value) in base.iter() {
            merged.set_item(key, value)?;
        }
    } else {
        let keys = base_headers.getattr("keys")?.call0()?;
        for key in keys.try_iter()? {
            let key = key?;
            merged.set_item(&key, base_headers.get_item(&key)?)?;
        }
    }
    let slot_str = py.import("builtins")?.getattr("str")?.call1((slot,))?;
    merged.set_item("x-goog-authuser", slot_str)?;
    let text = json_encode(py, &merged, "    ".to_string(), true, true, true)?;
    Ok(PyBytes::new(py, text.as_bytes()).unbind())
}

/// The account.json record assembly — the dict the commit writes for
/// exactly the auth bytes it publishes.
///
/// Mirrors the pure-Python reference (`_build_record`): the fixed
/// insertion order (`schema_version`, `x-goog-authuser` through the
/// interpreter's own `str()`, `channel_id` and `name` as the objects
/// they are, `handle` through the truthiness gate — a falsy handle
/// becomes `None` — `verified` as passed, the auth bytes' SHA-256 hex,
/// the stream jar's hash as passed, and a fresh `revision` from the
/// interpreter's own `os.urandom(16).hex()`). The atomic write itself
/// stays Python-side in the commit.
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the record's seven fields, one parameter each
#[pyo3(signature = (slot, name, handle, channel_id, verified, auth_payload, stream_sha256))]
pub(crate) fn build_record<'py>(
    py: Python<'py>,
    slot: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    handle: &Bound<'py, PyAny>,
    channel_id: &Bound<'py, PyAny>,
    verified: &Bound<'py, PyAny>,
    auth_payload: &Bound<'py, PyAny>,
    stream_sha256: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("schema_version", 1i64)?;
    let slot_str = py.import("builtins")?.getattr("str")?.call1((slot,))?;
    out.set_item("x-goog-authuser", slot_str)?;
    out.set_item("channel_id", channel_id)?;
    out.set_item("name", name)?;
    let handle_value = if handle.is_truthy()? {
        handle.clone()
    } else {
        py.None().into_bound(py)
    };
    out.set_item("handle", handle_value)?;
    out.set_item("verified", verified)?;
    let sha = sha256_hex_impl(py, auth_payload)?;
    out.set_item("auth_sha256", sha)?;
    out.set_item("stream_sha256", stream_sha256)?;
    let urandom = py.import("os")?.getattr("urandom")?.call1((16,))?;
    let revision = urandom.getattr("hex")?.call0()?;
    out.set_item("revision", revision)?;
    Ok(out)
}

/// The SHA-256 hex primitive — `hashlib.sha256(payload).hexdigest()`
/// as a standalone engine (see `sha256_hex_impl` for the shape rules).
#[pyfunction]
#[pyo3(signature = (payload))]
pub(crate) fn sha256_hex(py: Python<'_>, payload: &Bound<'_, PyAny>) -> PyResult<String> {
    sha256_hex_impl(py, payload)
}

// ── Manual-paste header normalization (services/auth.py) ─────────────────

/// The HTTP/2 pseudo-headers Chrome DevTools prefixes its request
/// listings with — dropped from every paste format (mirrors
/// `auth._PSEUDO_HEADERS`, exact string membership).
const PSEUDO_HEADERS: [&str; 5] = [":authority", ":method", ":path", ":scheme", ":status"];

/// Drop Chrome's decoded-protobuf annotation for ``x-client-data``
/// (mirrors `auth._strip_decoded_blocks`): every ``Decoded:`` line
/// with a matching bare ``}`` line later in the input drops both ends
/// and everything between; an unterminated block is left untouched
/// (dropping to end-of-input would discard real headers, and the
/// caller degrades better on a parity shift than on missing lines).
/// Both markers compare after `str.strip()` (CPython's own whitespace
/// set — see `py_strip`).
fn strip_decoded_blocks(lines: Vec<&str>) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if py_strip(lines[i]) != "Decoded:" {
            out.push(lines[i]);
            i += 1;
            continue;
        }
        // next(j for j in range(i + 1, len(lines)) if … == "}") — the
        // first closing brace strictly after the opener.
        match (i + 1..lines.len()).find(|&j| py_strip(lines[j]) == "}") {
            None => {
                out.push(lines[i]);
                i += 1;
            }
            Some(close) => i = close + 1,
        }
    }
    out
}

/// The manual-paste header pre-processing — `services/auth.py`'s
/// `_normalize_raw_headers`, behind `_setup_manual`'s paste loop.
///
/// Mirrors the reference's three paste formats read for read: the
/// single-line terminal paste split on its escape separator (the
/// literal ``^[E`` winning over the real ``ESC E`` over the bare
/// ``ESC``, whatever the paste actually carried), paired two-by-two
/// into ``Name: Value`` lines with the empty names and HTTP/2
/// pseudo-headers dropped; the multi-line walk with blank lines
/// filtered out (blank after CPython's own `str.strip()` — a wider
/// set than the crate's `trim`) and Chrome's decoded-protobuf blocks
/// stripped, classified as alternating name/value lines when more
/// than two lines survive and fewer than a fifth contain ``": "`` —
/// the same `len * 0.2` float threshold — paired two-by-two with only
/// the pseudo-headers dropped (an empty name still pairs here, the
/// reference's asymmetry); and the standard ``Name: Value`` pass that
/// keeps every surviving line stripped, minus the leading-colon
/// pseudo-headers. Every strip is `py_strip`, every join a plain
/// ``\n`` join (the empty list answering the empty string).
#[pyfunction]
#[pyo3(signature = (raw))]
pub(crate) fn normalize_raw_headers(raw: &str) -> String {
    let has_caret = raw.contains("^[E");
    let has_esc_e = raw.contains("\u{1b}E");
    if has_caret || has_esc_e || raw.contains('\u{1b}') {
        // "^[E" if "^[E" in raw else ("\x1bE" if … else "\x1b")
        let sep = if has_caret {
            "^[E"
        } else if has_esc_e {
            "\u{1b}E"
        } else {
            "\u{1b}"
        };
        let parts: Vec<&str> = raw.split(sep).collect();
        let mut lines: Vec<String> = Vec::new();
        let mut i = 0;
        while i + 1 < parts.len() {
            let name = py_strip(parts[i]);
            let value = py_strip(parts[i + 1]);
            i += 2;
            if name.is_empty() || PSEUDO_HEADERS.contains(&name) {
                continue;
            }
            lines.push(format!("{name}: {value}"));
        }
        return lines.join("\n");
    }

    let raw_lines = strip_decoded_blocks(
        raw.split('\n')
            .filter(|line| !py_strip(line).is_empty())
            .collect(),
    );
    let colon_lines = raw_lines.iter().filter(|line| line.contains(": ")).count();
    // len(raw_lines) > 2 and colon_lines < len(raw_lines) * 0.2 — the
    // int-times-float product and the int-vs-float comparison both at
    // f64 width, exactly like the reference's.
    let is_alternating = raw_lines.len() > 2 && (colon_lines as f64) < raw_lines.len() as f64 * 0.2;

    if is_alternating {
        let mut lines: Vec<String> = Vec::new();
        let mut i = 0;
        while i + 1 < raw_lines.len() {
            let name = py_strip(raw_lines[i]);
            let value = py_strip(raw_lines[i + 1]);
            i += 2;
            if PSEUDO_HEADERS.contains(&name) {
                continue;
            }
            lines.push(format!("{name}: {value}"));
        }
        return lines.join("\n");
    }

    let mut result: Vec<&str> = Vec::new();
    for line in &raw_lines {
        let stripped = py_strip(line);
        if stripped.starts_with(':') {
            continue;
        }
        result.push(stripped);
    }
    result.join("\n")
}

// ── Stream cookiejar builder (services/auth.py) ──────────────────────────

/// yt-dlp's own Netscape jar header — `YoutubeDLCookieJar._HEADER`. The
/// reference saves through yt-dlp's subclass, not stdlib's
/// `MozillaCookieJar`, so this two-line header (not stdlib's four-line
/// `NETSCAPE_HEADER_TEXT`) is what the file starts with.
const YTDLP_JAR_HEADER: &str =
    "# Netscape HTTP Cookie File\n# This file is generated by yt-dlp.  Do not edit.\n\n";

/// `CookieJar.__iter__` — stdlib `deepvalues` over the nested
/// `{domain: {path: {name: cookie}}}` store: depth-first, each level's
/// entries in the running interpreter's own order — plain insertion
/// order on 3.11+, and the `sorted(mapping.keys())` walk while 3.10
/// still sorted (gh-86232 changed `deepvalues` mid-3.11 development),
/// so the jar bytes always match what the fallback of the interpreter
/// that loaded the engine would have written. The walk recurses into
/// the store's own dict levels only — the cookie objects it reaches
/// never carry `.items` the way the reference's `obj.items` probe
/// requires of a nested mapping, so both walks agree on every store
/// this engine can build.
fn jar_cookies<'py>(
    py: Python<'py>,
    store: &Bound<'py, PyDict>,
    sorted_fn: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut out = Vec::new();
    jar_walk_level(py, store, py.version_info() < (3, 11), sorted_fn, &mut out)?;
    Ok(out)
}

fn jar_walk_level<'py>(
    py: Python<'py>,
    level: &Bound<'py, PyDict>,
    sorted_jar: bool,
    sorted_fn: &Bound<'py, PyAny>,
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if sorted_jar {
        // values = map(adict.get, sorted(adict.keys())) — the keys
        // ordered through CPython's own sorted (its TypeError for an
        // unorderable pair kept), each read through the dict's own
        // subscript; a key vanishing mid-walk reads as None, which
        // deepvalues then yields as a value — reproduced here.
        let keys = level.getattr("keys")?.call0()?;
        let sorted = sorted_fn.call1((keys,))?;
        for key in sorted.try_iter()? {
            let key = key?;
            match level.get_item(&key)? {
                Some(value) => jar_walk_value(py, value, sorted_jar, sorted_fn, out)?,
                None => out.push(py.None().into_bound(py)),
            }
        }
    } else {
        // list(mapping.values()) — each level in plain insertion order.
        for (_, value) in level.iter() {
            jar_walk_value(py, value, sorted_jar, sorted_fn, out)?;
        }
    }
    Ok(())
}

fn jar_walk_value<'py>(
    py: Python<'py>,
    value: Bound<'py, PyAny>,
    sorted_jar: bool,
    sorted_fn: &Bound<'py, PyAny>,
    out: &mut Vec<Bound<'py, PyAny>>,
) -> PyResult<()> {
    if let Ok(sub) = value.downcast::<PyDict>() {
        jar_walk_level(py, sub, sorted_jar, sorted_fn, out)?;
    } else {
        out.push(value);
    }
    Ok(())
}

/// The wide stream cookiejar builder — `services/auth.py`'s
/// `_build_stream_cookiejar`, behind `_commit_session`'s first write.
///
/// Mirrors the reference read for read. The filter loop walks *jar*
/// through Python's iteration protocol, each cookie's domain through
/// its own `lstrip(".")`, the exact-family membership
/// (`bare in ("youtube.com", "google.com")` — CPython's own `in`, so a
/// confusable like `notyoutube.com` fails the equality before the
/// suffix arm admits it) and the suffix arm through the object's own
/// `endswith` tuple call; a kept cookie's value read through the
/// reference's `or ""` truthiness gate and its name and value walked
/// through the `ch in value` containment of the control-char gate
/// (`"\t"`/`"\n"`/`"\r"`, name first — an embedded control character
/// would corrupt the line-oriented file). Survivors enter the jar's
/// own nested `{domain: {path: {name: cookie}}}` store — plain dicts,
/// so the reference `set_cookie`'s CPython semantics hold: hash/eq
/// membership, the first position of a repeated key with its last
/// object.
///
/// The save is yt-dlp's `YoutubeDLCookieJar.save` (the subclass the
/// reference instantiates), `ignore_discard`/`ignore_expires` both on
/// so neither gate ever bites: first the session-cookie pass that
/// assigns `expires = 0` to every kept cookie still carrying `None`
/// (the caller's own Cookie objects, mutated exactly like the
/// reference's shared references), then the line pass — the domain
/// through its own `startswith(".")`, the path, the secure flag's
/// truthiness, the expires through the interpreter's own `str()`
/// (never `None` after the first pass, so a session cookie renders
/// `"0"`), the name/value pair swapped when the value is `None`
/// (`Set-Cookie: foo`'s Netscape spelling), all seven columns joined
/// by `"\t".join` — CPython's own join, its TypeError for a non-str
/// column included — and the line written through `'{}\n'.format`.
/// The jar's cookie order is the interpreter's own `deepvalues` walk
/// (see `jar_cookies`), the buffer assembled by `"".join` and encoded
/// through the text object's own `encode("utf-8")` — its
/// `UnicodeEncodeError` shape kept. The `yt_dlp.cookies` import and
/// the StringIO buffer stay Python-side in the fallback only.
#[pyfunction]
#[pyo3(signature = (jar))]
pub(crate) fn build_stream_cookiejar<'py>(
    py: Python<'py>,
    jar: &Bound<'py, PyAny>,
) -> PyResult<Py<PyBytes>> {
    let dot = PyString::new(py, ".");
    let empty_str = PyString::new(py, "");
    let exact_domains = PyTuple::new(py, ["youtube.com", "google.com"])?.into_any();
    let suffix_domains = PyTuple::new(py, [".youtube.com", ".google.com"])?;
    let builtins = py.import("builtins")?;
    let str_fn = builtins.getattr("str")?;
    let sorted_fn = builtins.getattr("sorted")?;

    // stream_jar = YoutubeDLCookieJar() — the nested store starts
    // empty, exactly like the fresh jar the reference builds.
    let store = PyDict::new(py);

    for cookie in jar.try_iter()? {
        let cookie = cookie?;
        // bare = cookie.domain.lstrip(".")
        let domain = cookie.getattr("domain")?;
        let bare = domain.getattr("lstrip")?.call1((dot.clone(),))?;
        // if bare in ("youtube.com", "google.com") or bare.endswith(…)
        let keep = exact_domains.contains(&bare)?
            || bare
                .getattr("endswith")?
                .call1((suffix_domains.clone(),))?
                .is_truthy()?;
        if !keep {
            continue;
        }
        // value = cookie.value or ""
        let value = cookie.getattr("value")?;
        let checked = if value.is_truthy()? {
            value
        } else {
            empty_str.clone().into_any()
        };
        // if _has_control_chars(cookie.name, value): continue — name
        // first, then the gated value, each against \t/\n/\r through
        // CPython's own `in`.
        let name = cookie.getattr("name")?;
        let mut control = false;
        for v in [&name, &checked] {
            for ch in ["\t", "\n", "\r"] {
                if v.contains(PyString::new(py, ch))? {
                    control = true;
                    break;
                }
            }
            if control {
                break;
            }
        }
        if control {
            continue;
        }
        // set_cookie(cookie) — the reference's own store walk, each
        // level created on miss and the (domain, path) attrs read
        // again exactly where set_cookie reads them.
        let cookie_domain = cookie.getattr("domain")?;
        if !store.contains(&cookie_domain)? {
            store.set_item(&cookie_domain, PyDict::new(py))?;
        }
        let paths = store
            .get_item(&cookie_domain)?
            .expect("the domain was just inserted or already present");
        let paths = paths
            .downcast::<PyDict>()
            .expect("set_cookie only ever stores dicts");
        let cookie_path = cookie.getattr("path")?;
        if !paths.contains(&cookie_path)? {
            paths.set_item(&cookie_path, PyDict::new(py))?;
        }
        let names = paths
            .get_item(&cookie_path)?
            .expect("the path was just inserted or already present");
        let names = names
            .downcast::<PyDict>()
            .expect("set_cookie only ever stores dicts");
        let cookie_name = cookie.getattr("name")?;
        names.set_item(&cookie_name, cookie)?;
    }

    // The session-cookie pass — every kept cookie still carrying
    // expires None is assigned the int 0, mutating the caller's own
    // objects exactly like the reference's shared references.
    for cookie in jar_cookies(py, &store, &sorted_fn)? {
        if cookie.getattr("expires")?.is_none() {
            cookie.setattr("expires", 0i64)?;
        }
    }

    // The line pass — yt-dlp's _HEADER, then one seven-column line per
    // cookie in the jar's own order.
    let tab = PyString::new(py, "\t");
    let line_format = PyString::new(py, "{}\n");
    let mut parts: Vec<Bound<'py, PyAny>> = Vec::new();
    parts.push(PyString::new(py, YTDLP_JAR_HEADER).into_any());
    for cookie in jar_cookies(py, &store, &sorted_fn)? {
        let domain = cookie.getattr("domain")?;
        let domain_dot = if domain
            .getattr("startswith")?
            .call1((dot.clone(),))?
            .is_truthy()?
        {
            "TRUE"
        } else {
            "FALSE"
        };
        let path = cookie.getattr("path")?;
        let secure = if cookie.getattr("secure")?.is_truthy()? {
            "TRUE"
        } else {
            "FALSE"
        };
        // str_or_none(cookie.expires, default="") — never None after
        // the session-cookie pass, so the interpreter's own str().
        let expires = cookie.getattr("expires")?;
        let expires_col = if expires.is_none() {
            empty_str.clone().into_any()
        } else {
            str_fn.call1((expires,))?
        };
        // name, value = cookie.name, cookie.value — the value read
        // after the name, a None value swapping the pair.
        let name = cookie.getattr("name")?;
        let value = cookie.getattr("value")?;
        let (name_col, value_col) = if value.is_none() {
            (empty_str.clone().into_any(), name)
        } else {
            (name, value)
        };
        let cols = PyTuple::new(
            py,
            [
                domain,
                PyString::new(py, domain_dot).into_any(),
                path,
                PyString::new(py, secure).into_any(),
                expires_col,
                name_col,
                value_col,
            ],
        )?;
        let joined = tab.getattr("join")?.call1((cols,))?;
        parts.push(line_format.getattr("format")?.call1((joined,))?);
    }
    // The StringIO's accumulated text, assembled and encoded through
    // the str object's own methods.
    let text = empty_str
        .getattr("join")?
        .call1((PyList::new(py, parts)?,))?;
    let encoded = text
        .getattr("encode")?
        .call1((PyString::new(py, "utf-8"),))?;
    let bytes = encoded
        .downcast_into::<PyBytes>()
        .expect("str.encode returns bytes");
    Ok(bytes.unbind())
}
