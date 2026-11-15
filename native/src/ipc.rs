//! Native cores for `src/ytm_player/ipc.py` — the Unix-socket request
//! gate (module compare-digest + argv check).

use crate::utils::pycompat::py_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

// ---------------------------------------------------------------------------
// IPC single-instance guard (mirrors ytm_player/ipc.py)
// ---------------------------------------------------------------------------

/// Mirror the lenient entry-point heuristic of
/// `ytm_player.ipc._looks_like_ytm`: any argument whose
/// separator-agnostic basename is `ytm` or starts with `ytm.`, or that
/// contains `ytm_player` anywhere, counts as ours; a single-element argv
/// whose basename starts with `python` also counts (the Windows
/// image-name-only probe cannot rule out `python -m ytm_player`). Errs on
/// the side of True — a false "ours" only preserves the block behavior,
/// while a false "not ours" could steal a live TUI's PID file.
#[pyfunction]
pub(crate) fn looks_like_ytm(argv: Vec<String>) -> bool {
    fn basename_lower(arg: &str) -> String {
        arg.replace('\\', "/")
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_lowercase()
    }

    for arg in &argv {
        let name = basename_lower(arg);
        if name == "ytm" || name.starts_with("ytm.") || arg.contains("ytm_player") {
            return true;
        }
    }
    if argv.len() == 1 {
        let name = basename_lower(&argv[0]);
        if name.starts_with("python") {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// IPC request gate (mirror ytm_player/ipc.py)
// ---------------------------------------------------------------------------

/// A fresh `{"verdict": "reject", "error": …}` outcome row for the
/// request gate, carrying the response error the reference spells.
fn ipc_gate_reject<'py>(py: Python<'py>, error: &str) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("verdict", "reject")?;
    out.set_item("error", error)?;
    Ok(out)
}

/// The ipc module's own `secrets` name — the `compare_digest` the
/// reference's global lookup would call, read through the module
/// attribute at call time. A test (or any caller) that swaps
/// `ytm_player.ipc`'s `secrets` for a stub namespace steers the native
/// engine exactly like the fallback: the digest the reference would
/// call is the digest the engine calls.
fn ipc_module_compare_digest(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.ipc")?
        .getattr("secrets")?
        .getattr("compare_digest")
}

/// The request gate — `ipc.py`'s `_ipc_request_gate`, the
/// key-exchange body behind `IPCServer._client_connected`: the token
/// gate and command whitelist a parsed request must pass before the
/// handler dispatches with its args.
///
/// Mirrors the reference's three gates in their exact order, every
/// read through CPython's own protocol. On the TCP transport (*token*
/// not `None` — identity, not truthiness, so an empty-string token
/// still gates; the Unix-socket transport passes `None` and its 0600
/// socket perms are the gate), the request's `token` member —
/// `request.get("token")` through the mapping's own method — must be
/// a `str` (the subclass-aware isinstance gate, short-circuited
/// before any digest, exactly like the reference's `or`) comparing
/// equal to the server's token through `secrets.compare_digest` over
/// both sides' `.encode("utf-8")` — the presented side encoded first,
/// the digest fetched from the module's own `secrets` name at call
/// time (a patched namespace steers both backends identically) — and
/// a missing or non-str member, or any mismatch, rejects with the
/// reference's `authentication failed`. `command` —
/// `request.get("command", "")` — must then be a `str` in
/// *valid_commands* (the containment protocol over the module's own
/// `_VALID_COMMANDS` frozenset, passed through so a patched whitelist
/// steers both backends identically), and anything else rejects with
/// the reference's f-string — the raw command object interpolated
/// through `str()`. `args` — `request.get("args", {})`, the fresh
/// default dict built before the lookup runs — passes through as the
/// object it is when it is a dict (the subclass-aware check) and is
/// another fresh dict otherwise. The accept verdict carries the
/// command and the args as the objects the handler dispatches with;
/// the framing, the JSON parse, the handler await and the response
/// writes stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (request, token, valid_commands))]
pub(crate) fn ipc_request_gate<'py>(
    py: Python<'py>,
    request: &Bound<'py, PyAny>,
    token: &Bound<'py, PyAny>,
    valid_commands: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let get = request.getattr("get")?;
    // if token is not None — identity, not truthiness.
    if !token.is_none() {
        // presented = request.get("token")
        let presented = get.call1(("token",))?;
        // not isinstance(presented, str) — short-circuited before the
        // digest, exactly like the reference's `or`.
        if !presented.is_instance_of::<PyString>() {
            return ipc_gate_reject(py, "authentication failed");
        }
        // secrets.compare_digest(presented.encode("utf-8"),
        //                         token.encode("utf-8")) — the
        // presented side encoded first, the digest read through the
        // ipc module's own secrets name.
        let presented_bytes = presented.getattr("encode")?.call1(("utf-8",))?;
        let token_bytes = token.getattr("encode")?.call1(("utf-8",))?;
        let compare_digest = ipc_module_compare_digest(py)?;
        if !compare_digest
            .call1((presented_bytes, token_bytes))?
            .is_truthy()?
        {
            return ipc_gate_reject(py, "authentication failed");
        }
    }
    // command = request.get("command", "")
    let command = get.call1(("command", ""))?;
    // not isinstance(command, str) or command not in valid_commands
    if !command.is_instance_of::<PyString>() || !valid_commands.contains(&command)? {
        // f"unknown command: {command}" — the object interpolated
        // through str().
        return ipc_gate_reject(py, &format!("unknown command: {}", py_str(&command)?));
    }
    // args = request.get("args", {}) — the fresh default built before
    // the lookup runs, exactly like the reference's argument
    // evaluation.
    let args_default = PyDict::new(py).into_any();
    let args_raw = get.call1(("args", &args_default))?;
    let args = if args_raw.is_instance_of::<PyDict>() {
        args_raw
    } else {
        PyDict::new(py).into_any()
    };
    let out = PyDict::new(py);
    out.set_item("verdict", "accept")?;
    out.set_item("command", command)?;
    out.set_item("args", args)?;
    Ok(out)
}
