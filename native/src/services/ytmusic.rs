//! Native cores for `src/ytm_player/services/ytmusic.py` — API kwargs
//! adapters, renewal cooldowns, mutation classification and the
//! discovery/chart/radio assemblies.

use crate::app::keys::nd_digit_value;
use crate::app::keys::unpack_too_many_message;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};
use regex::Regex;
use std::sync::LazyLock;

// ── LRU bookkeeping engines (shuffle prefs + album-art cache) ───────────

/// CPython's `\w` character set for `str` patterns, as a one-character
/// test: the alphanumeric union behind `str.isalnum()` — the letter
/// categories `L*` and every numeric category `N*` (decimal `Nd`, letter
/// numbers `Nl` like the Roman numerals, and other numerics `No` like
/// the circled digits and vulgar fractions, the sets CPython's
/// ISALPHA/ISDECIMAL/ISDIGIT/ISNUMERIC union admits) — plus the
/// underscore. Deliberately NOT the regex crate's own `\w` (which adds
/// the combining marks `M*`, the connector punctuation `Pc` and the
/// join-control characters): the mutation-failure classifier's HTTP
/// status boundary must break exactly where Python's breaks — a
/// combining mark after the third digit is a boundary (Python: not a
/// word character), while a circled digit or a fraction after it is not
/// (Python: alphanumeric).
static PY_WORD_CHAR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[\p{L}\p{N}_]$").expect("python \\w character set"));

/// One-character `\w` test for the classifier's word boundary — see
/// `PY_WORD_CHAR_RE`.
fn py_is_word_char(c: char) -> bool {
    let mut buf = [0u8; 4];
    PY_WORD_CHAR_RE.is_match(c.encode_utf8(&mut buf))
}

/// Position of *key* in *entries* under CPython's own dict-lookup rule:
/// the identity shortcut first (`stored is key`), then `==` (so keys
/// with no consistent `__eq__` — float NaN looked up through its own
/// object — still hit exactly the way the hash bucket's shortcut makes
/// them). The caller hashes *key* up front, so unhashable lookups raise
/// the reference's `TypeError` before any comparison runs — the order
/// the dict/OrderedDict protocol methods the engines replace follow. A
/// linear walk replaces the hash bucket scan: the same hit for every
/// key whose `__hash__`/`__eq__` honour the Python data model (the
/// documented contract those methods assume), and the walk preserves
/// the insertion order the OrderedDict engines are about.
pub(crate) fn py_key_position<'py, V>(
    py: Python<'py>,
    entries: &[(Py<PyAny>, V)],
    key: &Bound<'py, PyAny>,
) -> PyResult<Option<usize>> {
    for (i, (stored, _)) in entries.iter().enumerate() {
        let stored = stored.bind(py);
        if std::ptr::eq(stored.as_ptr(), key.as_ptr()) {
            return Ok(Some(i));
        }
        if stored.eq(key)? {
            return Ok(Some(i));
        }
    }
    Ok(None)
}

/// The `for k, v in seq` unpacking of the shuffle-preferences ingestion
/// comprehension: a 2-tuple yields its halves; anything else goes through
/// CPython's UNPACK_SEQUENCE contract — the iterator protocol for
/// non-tuples (so a two-character string unpacks into its characters),
/// `ValueError` with CPython's own not-enough/too-many message for a
/// wrong-shaped yield, and the iterator protocol's own `TypeError` for
/// an uniterable item.
pub(crate) fn py_unpack_pair<'py>(
    item: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let unpack_not_enough = |got: usize| {
        PyValueError::new_err(format!(
            "not enough values to unpack (expected 2, got {got})"
        ))
    };
    if let Ok(tuple) = item.downcast::<PyTuple>() {
        return match tuple.len() {
            2 => Ok((tuple.get_item(0)?, tuple.get_item(1)?)),
            n if n > 2 => Err(PyValueError::new_err(unpack_too_many_message(
                item.py(),
                item,
                2,
            ))),
            n => Err(unpack_not_enough(n)),
        };
    }
    let mut got: Vec<Bound<'py, PyAny>> = Vec::new();
    for value in item.try_iter()? {
        got.push(value?);
        if got.len() > 2 {
            break;
        }
    }
    match got.len() {
        2 => Ok((got.remove(0), got.remove(0))),
        n if n > 2 => Err(PyValueError::new_err(unpack_too_many_message(
            item.py(),
            item,
            2,
        ))),
        n => Err(unpack_not_enough(n)),
    }
}

// ── Mutation-failure classifier ────────────────────────────────────────

/// The `_HTTP_STATUS_RE.match(str(exc))` of the mutation-failure
/// classifier — the pure core of `_classify_mutation_failure`'s
/// YTMusicServerError arm: the literal prefix `Server returned HTTP `
/// (ASCII, anchored at the start of the message like the reference's
/// `^`), then exactly three Unicode decimal digits (CPython's `\d` on
/// `str` patterns — category Nd, any script, so Arabic-Indic ٤٠١ and
/// Devanagari ४०३ are three-digit codes), then the word boundary `\b`:
/// the end of the message, or a next character that is not CPython's
/// `\w` (alphanumeric-or-underscore — see `py_is_word_char`; a combining
/// mark after the digits is a boundary, a circled digit or a vulgar
/// fraction is not, exactly where Python's own `\b` breaks). Fewer
/// than three digits, or a fourth word character, fails the match —
/// the fixed-count group has no shorter retry under the anchored
/// literal prefix. The three digits then parse through `int()`'s own
/// decimal-digit acceptance (every Nd trio is a valid int — ٤٠١ is
/// 401), so a Unicode-digit 401/403 classifies auth_expired exactly
/// like an ASCII one.
fn server_http_status(message: &str) -> Option<i64> {
    let rest = message.strip_prefix("Server returned HTTP ")?;
    let mut value: i64 = 0;
    let mut chars = rest.chars();
    // Three `\d`s — fewer available is the match failure.
    for _ in 0..3 {
        let c = chars.next()?;
        value = value * 10 + i64::from(nd_digit_value(c)?);
    }
    // `\b` — the end of the message or the next character is not a
    // word character.
    match chars.next() {
        None => Some(value),
        Some(c) if !py_is_word_char(c) => Some(value),
        Some(_) => None,
    }
}

/// The mutation-failure classifier behind `services/ytmusic.py`'s
/// `_classify_mutation_failure` — the pure core that maps an exception
/// raised by a mutation call to its `MutationResult` kind, never
/// "success".
///
/// Mirrors the reference's isinstance chain in its exact order
/// (`KeyError` → parser drift is a server-side problem; the two timeout
/// classes → no confirmation in time, the write may have landed;
/// `requests`' `RequestException` → network; `YTMusicUserError` → no
/// auth set up; `YTMusicServerError` → the HTTP status parsed out of
/// the message, 401/403 an expired session, anything else a
/// server-side problem), with every class resolved from the running
/// interpreter at call time — `asyncio.TimeoutError` is whatever that
/// attribute is on the running Python (the builtin `TimeoutError` on
/// 3.11+, asyncio's own class before), `requests.exceptions` and
/// `ytmusicapi.exceptions` the very modules the reference imported at
/// its own module scope. The YTMusicServerError arm renders the
/// exception through `str()` and matches the HTTP-status prefix (see
/// `server_http_status`); anything else falls through to
/// "server_error", the reference's own unreachable-in-practice safety
/// default. The inputs are never mutated.
#[pyfunction]
pub(crate) fn classify_mutation_failure(
    py: Python,
    exc: &Bound<'_, PyAny>,
) -> PyResult<&'static str> {
    // isinstance(exc, KeyError) — parser drift classifies server_error.
    let builtins = py.import("builtins")?;
    let key_error = builtins.getattr("KeyError")?;
    if exc.is_instance(&key_error)? {
        return Ok("server_error");
    }
    // isinstance(exc, (asyncio.TimeoutError, requests.ReadTimeout))
    let requests_exceptions = py.import("requests.exceptions")?;
    let asyncio_timeout = py.import("asyncio")?.getattr("TimeoutError")?;
    let read_timeout = requests_exceptions.getattr("ReadTimeout")?;
    if exc.is_instance(&asyncio_timeout)? || exc.is_instance(&read_timeout)? {
        return Ok("timeout");
    }
    // isinstance(exc, requests.RequestException)
    let request_exception = requests_exceptions.getattr("RequestException")?;
    if exc.is_instance(&request_exception)? {
        return Ok("network");
    }
    // isinstance(exc, YTMusicUserError)
    let ytmusic_exceptions = py.import("ytmusicapi.exceptions")?;
    let user_error = ytmusic_exceptions.getattr("YTMusicUserError")?;
    if exc.is_instance(&user_error)? {
        return Ok("auth_required");
    }
    // isinstance(exc, YTMusicServerError) — the HTTP status prefix
    // decides auth_expired vs server_error.
    let server_error = ytmusic_exceptions.getattr("YTMusicServerError")?;
    if exc.is_instance(&server_error)? {
        // str(exc) — rendered through CPython's own __str__ (a raising
        // __str__ propagates, exactly like the reference's str() call);
        // the lossy read replaces lone surrogates with U+FFFD, which
        // moves no boundary (a surrogate is not \w, and neither is the
        // replacement character — the only positions the match reads).
        let message = exc.str()?.to_string_lossy().into_owned();
        if let Some(status) = server_http_status(&message) {
            if status == 401 || status == 403 {
                return Ok("auth_expired");
            }
        }
        return Ok("server_error");
    }
    // The reference's unreachable-in-practice safety default.
    Ok("server_error")
}

/// The ytmusic module's own clock — `time.monotonic()` read through the
/// module's `time` attribute, exactly like the reference's global
/// lookup at call time. A test (or any caller) that swaps
/// `ytm_player.services.ytmusic`'s `time` for a frozen namespace steers
/// the native engines exactly like the fallback: the clock the
/// reference would read is the clock the engines read.
fn ytmusic_module_monotonic(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let ytmusic = PyModule::import(py, "ytm_player.services.ytmusic")?;
    let time_module = ytmusic.getattr("time")?;
    let clock = time_module.getattr("monotonic")?;
    clock.call0()
}

/// The auth-expiry sniff behind `services/ytmusic.py`'s
/// `_is_auth_expired` — the gate every expired-session retry
/// (`_call`'s except arm and `get_playlist`'s ordered branch)
/// consults before renewing.
///
/// Mirrors the reference's isinstance chain in its exact order: a
/// `YTMusicServerError` (the class resolved from the running
/// interpreter, the very module the reference imported at its own
/// module scope) delegates to the module's own
/// `_classify_mutation_failure` — resolved through the module
/// attribute at call time, so whichever classifier the module is
/// running (the native engine or the pure-Python transcription)
/// steers this engine identically — and answers whether it classified
/// "auth_expired" (CPython's own `==`); a `KeyError` renders through
/// `str()`, lowercases through the str's own `.lower()` method and
/// answers whether the lowered message contains any of the three
/// sign-in markers — `signinendpoint`, `sign in to`, or the
/// `'logged_in', 'value': '0'` fragment — each containment through
/// CPython's own `in` (the substring protocol on the message object
/// itself, never a re-encoded copy); everything else answers `False`.
#[pyfunction]
#[pyo3(signature = (exc))]
pub(crate) fn ytm_auth_expired(py: Python, exc: &Bound<'_, PyAny>) -> PyResult<bool> {
    // isinstance(exc, YTMusicServerError) — the classifier verdict.
    let ytmusic_exceptions = py.import("ytmusicapi.exceptions")?;
    let server_error = ytmusic_exceptions.getattr("YTMusicServerError")?;
    if exc.is_instance(&server_error)? {
        let classifier = PyModule::import(py, "ytm_player.services.ytmusic")?
            .getattr("_classify_mutation_failure")?;
        let kind = classifier.call1((exc,))?;
        return kind.eq("auth_expired");
    }
    // isinstance(exc, KeyError) — the sign-in marker sniff over the
    // lowered message.
    let builtins = py.import("builtins")?;
    let key_error = builtins.getattr("KeyError")?;
    if exc.is_instance(&key_error)? {
        let message = exc.str()?.call_method0("lower")?;
        let marker_1 = PyString::new(py, "signinendpoint").into_any();
        if message.contains(&marker_1)? {
            return Ok(true);
        }
        let marker_2 = PyString::new(py, "sign in to").into_any();
        if message.contains(&marker_2)? {
            return Ok(true);
        }
        let marker_3 = PyString::new(py, "'logged_in', 'value': '0'").into_any();
        return message.contains(&marker_3);
    }
    Ok(false)
}

/// The renewal-cooldown gate — `services/ytmusic.py`'s
/// `_renewed_client_for` cooldown check, the bookkeeping that keeps a
/// failed automatic renewal from being retried (with its browser
/// cookie extraction and account probes) until the cooldown has
/// elapsed.
///
/// Mirrors the reference's two expressions exactly: the clock read
/// through the module's own `time` name first (see
/// `ytmusic_module_monotonic`), then `remaining = retry_after - now`
/// through CPython's own `-` (its `__sub__` protocol — a non-numeric
/// *retry_after* raises the operator's own `TypeError`), then the
/// `> 0` verdict through CPython's own comparison. The answer is the
/// (blocked, remaining) pair — *remaining* the very object the
/// subtraction produced, for the caller's `%.0fs` log line — with the
/// lock, the renewal itself and the raise staying Python-side in the
/// caller.
#[pyfunction]
#[pyo3(signature = (retry_after))]
pub(crate) fn ytm_renewal_cooldown_gate<'py>(
    py: Python<'py>,
    retry_after: &Bound<'py, PyAny>,
) -> PyResult<(bool, Bound<'py, PyAny>)> {
    let now = ytmusic_module_monotonic(py)?;
    let remaining = retry_after.sub(&now)?;
    let zero = 0i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let blocked = remaining.gt(&zero)?;
    Ok((blocked, remaining))
}

/// The renewal-cooldown stamp — `services/ytmusic.py`'s
/// `_renewed_client_for` failure arm, the bookkeeping a failed
/// automatic renewal writes so the gate above blocks the retries
/// until the cooldown has elapsed.
///
/// Mirrors the reference's single expression exactly: the clock read
/// through the module's own `time` name, then `now +
/// _RENEWAL_COOLDOWN_SECONDS` through CPython's own `+`, with the
/// cooldown constant read through the module's own attribute — the
/// very constant the fallback's own expression runs against, a
/// patched constant included. The stamp is whatever object the
/// addition produced; the assignment stays Python-side in the caller.
#[pyfunction]
pub(crate) fn ytm_renewal_cooldown_stamp<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let now = ytmusic_module_monotonic(py)?;
    let cooldown = PyModule::import(py, "ytm_player.services.ytmusic")?
        .getattr("_RENEWAL_COOLDOWN_SECONDS")?;
    now.add(&cooldown)
}

/// The consecutive-failure step — `services/ytmusic.py`'s `_run`
/// except arm, the retry bookkeeping that counts the expected API
/// failures and re-creates the YTMusic client once a streak of them
/// suggests the session has gone stale.
///
/// Mirrors the reference's two expressions exactly: the bump through
/// CPython's own `+` (its `__add__` protocol — a non-numeric count
/// raises the operator's own `TypeError`), then the `>=` verdict
/// through CPython's own comparison against the module's own
/// `_MAX_API_FAILURES_BEFORE_REINIT` — read through the module
/// attribute, so a patched threshold steers both backends
/// identically. The answer is the (bumped, due) pair — the bumped
/// count for the caller's own assignment and log line, the verdict
/// naming whether the reinit branch runs — with the log, the
/// `_client_init_lock` hold, the client clear and the post-reinit
/// reset to 0 staying Python-side in the caller, at the reference's
/// own points.
#[pyfunction]
#[pyo3(signature = (consecutive_failures))]
pub(crate) fn ytm_api_failure_step<'py>(
    py: Python<'py>,
    consecutive_failures: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let bumped = consecutive_failures.add(&one)?;
    let threshold = PyModule::import(py, "ytm_player.services.ytmusic")?
        .getattr("_MAX_API_FAILURES_BEFORE_REINIT")?;
    let due = bumped.ge(&threshold)?;
    Ok((bumped, due))
}

/// The patch-window enter step — `services/ytmusic.py`'s `_call` gate
/// arm, the admission half of the in-flight bookkeeping that keeps
/// every normal client call out of a `get_playlist(order=...)` patch
/// window's way.
///
/// Mirrors the reference's single expression exactly: the counter's
/// `+= 1` through CPython's own `+` (its `__add__` protocol — a
/// non-numeric count raises the operator's own `TypeError`). The
/// bumped count is the very object the addition produced, for the
/// caller's own `_inflight` assignment; the assignment itself and
/// the `_no_inflight` clear that follows it stay Python-side in the
/// caller, exactly at the reference's own points.
#[pyfunction]
#[pyo3(signature = (inflight))]
pub(crate) fn ytm_patch_window_enter<'py>(
    py: Python<'py>,
    inflight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    inflight.add(&one)
}

/// The patch-window exit step — `services/ytmusic.py`'s `_call`
/// finally arm, the release half of the in-flight bookkeeping: a call
/// leaving the gate counts itself out, and the drained verdict names
/// whether the counter reached zero — the moment the caller sets
/// `_no_inflight` so a patch window parked on it can open.
///
/// Mirrors the reference's two expressions exactly: the counter's
/// `-= 1` through CPython's own `-` (its `__sub__` protocol — a
/// non-numeric count raises the operator's own `TypeError`), then
/// the `== 0` verdict through CPython's own comparison. The answer
/// is the (left, drained) pair — *left* the very object the
/// subtraction produced, for the caller's own `_inflight`
/// assignment — with the assignment, the `_no_inflight` set and the
/// zombie-worker limitation note staying Python-side in the caller,
/// at the reference's own points.
#[pyfunction]
#[pyo3(signature = (inflight))]
pub(crate) fn ytm_patch_window_exit<'py>(
    py: Python<'py>,
    inflight: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool)> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let left = inflight.sub(&one)?;
    let zero = 0i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let drained = left.eq(&zero)?;
    Ok((left, drained))
}

// ── YTMusic browse-wrapper engines (services/ytmusic.py) ───────────────

/// The kwargs dict behind `YTMusicService.get_watch_playlist` — the
/// pure core of the wrapper's call assembly from `services/ytmusic.py`.
///
/// Mirrors the pure-Python reference (`_watch_playlist_kwargs`): the
/// conditional members in the reference's insertion order — `limit`
/// always; `videoId` / `playlistId` only when the caller passed them,
/// the gate an `is None` identity check (a falsy but non-None id still
/// travels); `radio` only when truthy, and the literal `True` travels,
/// never the flag object. Every value passes through as the object it
/// is; the both-None early return and the `**` unpacking stay
/// Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (video_id, playlist_id, limit, radio))]
pub(crate) fn ytm_watch_playlist_kwargs<'py>(
    py: Python<'py>,
    video_id: &Bound<'py, PyAny>,
    playlist_id: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
    radio: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("limit", limit)?;
    if !video_id.is_none() {
        kwargs.set_item("videoId", video_id)?;
    }
    if !playlist_id.is_none() {
        kwargs.set_item("playlistId", playlist_id)?;
    }
    if radio.is_truthy()? {
        kwargs.set_item("radio", true)?;
    }
    Ok(kwargs)
}

/// The `tracks` member of a watch / liked-songs payload — the pure
/// core of the tracks extraction `get_liked_songs`,
/// `get_watch_playlist`, `get_radio`'s per-seed fetch,
/// `get_playlist_radio` and the discovery trending/liked mixes share
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_result_tracks`): over the
/// subclass-aware `isinstance(result, dict)` gate the member is read
/// through the mapping's own `get` with a fresh empty list as the
/// default (the default list built every call, exactly like the
/// reference's literal); a non-dict answers a fresh empty list,
/// never the input itself, and a dict passes its `tracks` object
/// through as the object it is.
#[pyfunction]
#[pyo3(signature = (result))]
pub(crate) fn ytm_result_tracks<'py>(result: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = result.py();
    if result.is_instance_of::<PyDict>() {
        let default = PyList::empty(py).into_any();
        return result.getattr("get")?.call(("tracks", default), None);
    }
    Ok(PyList::empty(py).into_any())
}

/// The `tracks` member of a playlist payload, list-gated — the pure
/// core of `get_chart_shelf_tracks`'s extraction
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_playlist_tracks`): over the
/// subclass-aware `isinstance(playlist, dict)` gate the member is read
/// through the mapping's own `get` with a fresh empty list default,
/// then the reference's `or []` (a falsy member — `None` where the
/// payload carries the key's null shape — answers a fresh empty
/// list), then the subclass-aware `isinstance(raw, list)` gate: only
/// a list passes through, as the object it is; every other member
/// shape — a dict, a string, a generator — answers a fresh empty
/// list.
#[pyfunction]
#[pyo3(signature = (playlist))]
pub(crate) fn ytm_playlist_tracks<'py>(
    playlist: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = playlist.py();
    if playlist.is_instance_of::<PyDict>() {
        let default = PyList::empty(py).into_any();
        let raw = playlist.getattr("get")?.call(("tracks", default), None)?;
        let raw = if raw.is_truthy()? {
            raw
        } else {
            PyList::empty(py).into_any()
        };
        if raw.is_instance_of::<PyList>() {
            return Ok(raw);
        }
    }
    Ok(PyList::empty(py).into_any())
}

/// The deduplicated pool of radio tracks across all seeds' fetches —
/// the pure core of `YTMusicService.get_radio`'s mix assembly
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_radio_seed_pool`): the
/// per-seed result lists walked through Python's iteration protocol
/// (a non-iterable raises CPython's own `TypeError`; an `__iter__`
/// that itself raises keeps its own error), each list walked in
/// turn, and each track read its id through the mapping's own `get`
/// — `videoId` first, `video_id` with the fresh `""` default only
/// when the first read answers falsy (the reference's `or`, so the
/// second attribute lookup and call happen only then; a track
/// without a `get` attribute raises the attribute protocol's own
/// `AttributeError` at exactly the lookup the reference would have
/// made) — and the track kept when that id is truthy and not yet
/// seen: the set's own containment and add — the id hashed first
/// (an unhashable id raises the hash protocol's own `TypeError`),
/// the identity-or-`==` probe deciding membership, the FIRST of
/// equal ids the one that keeps its track. Kept tracks pass through
/// as the objects they are, in first-sight order; the shuffle, the
/// limit trim and the normalisation stay Python-side in the wrapper,
/// next to the RNG state.
#[pyfunction]
#[pyo3(signature = (results))]
pub(crate) fn ytm_radio_seed_pool<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let pool = PyList::empty(py);
    let seen = PySet::empty(py)?;
    for tracks in results.try_iter()? {
        for track in tracks?.try_iter()? {
            let track = track?;
            let first = track.getattr("get")?.call1(("videoId",))?;
            let vid = if first.is_truthy()? {
                first
            } else {
                track.getattr("get")?.call(("video_id", ""), None)?
            };
            // if vid and vid not in seen_ids — the truthiness gate
            // first, the containment (and its hash) only behind it.
            if vid.is_truthy()? && !seen.contains(&vid)? {
                seen.add(&vid)?;
                pool.append(track)?;
            }
        }
    }
    Ok(pool)
}

/// The six discovery sources' rotation order — the pure core of
/// `YTMusicService.get_discovery_mix`'s source cycling
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_discovery_source_order`):
/// the `(last % 6) + 1` start (0 → 1 first, 5 → 6 first, 6 wraps to
/// 1) and the six-entry rotation that follows it —
/// `(start + i - 1) % 6 + 1` for `i` over `range(6)`, left-associative
/// exactly like the reference spells it — every operator through
/// CPython's own, so a non-int last source (a float, a bool) steers
/// both backends identically.
#[pyfunction]
#[pyo3(signature = (last_source))]
pub(crate) fn ytm_discovery_source_order<'py>(
    py: Python<'py>,
    last_source: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let six = 6i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    let start = last_source.rem(&six)?.add(&one)?;
    let out = PyList::empty(py);
    for i in 0i64..6 {
        let i = i
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any();
        out.append(start.add(&i)?.sub(&one)?.rem(&six)?.add(&one)?)?;
    }
    Ok(out)
}

/// The chart shelves of a `get_charts` payload — the pure core of the
/// discovery charts mix's shelf assembly (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_chart_shelves`): the `daily`,
/// `weekly` and `videos` members read through the mapping's own `get`
/// — one lookup per key, in that order, a missing member skipped —
/// each gated by the subclass-aware `isinstance(arr, list)`, and each
/// survivor's items filtered by the subclass-aware
/// `isinstance(s, dict)` into the flat shelf list. The dict shelves
/// pass through as the objects they are; the payload is never
/// mutated.
#[pyfunction]
#[pyo3(signature = (charts))]
pub(crate) fn ytm_chart_shelves<'py>(
    py: Python<'py>,
    charts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let shelves = PyList::empty(py);
    for key in ["daily", "weekly", "videos"] {
        let arr = charts.getattr("get")?.call1((key,))?;
        if arr.is_instance_of::<PyList>() {
            for s in arr.try_iter()? {
                let s = s?;
                if s.is_instance_of::<PyDict>() {
                    shelves.append(s)?;
                }
            }
        }
    }
    Ok(shelves)
}

/// The next chart shelf's position in the rotation — the pure core of
/// the discovery charts mix's shelf pick (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_chart_shelf_index`): the
/// `(last + 1) % count` round-robin step, both operators through
/// CPython's own — so a zero count raises the modulo's own
/// `ZeroDivisionError` exactly like the reference would (the mix
/// guards with `if not shelves` before it ever reaches the pick).
#[pyfunction]
#[pyo3(signature = (last_index, shelf_count))]
pub(crate) fn ytm_chart_shelf_index<'py>(
    py: Python<'py>,
    last_index: &Bound<'py, PyAny>,
    shelf_count: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    last_index.add(&one)?.rem(shelf_count)
}

/// The playable track dicts of a discovery payload — the pure core of
/// the playable filter the trending, home, liked, artist and history
/// mixes run over their payloads (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_discovery_playable_tracks`):
/// *items* walked through Python's iteration protocol, and only dict
/// items (the subclass-aware `isinstance` gate, short-circuited
/// before any `get` call exactly like the reference's `and`) whose
/// `videoId` member — read through the mapping's own `get` — is
/// truthy pass through, as the objects they are, in the payload's
/// order. The input is never mutated.
#[pyfunction]
#[pyo3(signature = (items))]
pub(crate) fn ytm_discovery_playable_tracks<'py>(
    py: Python<'py>,
    items: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for t in items.try_iter()? {
        let t = t?;
        if t.is_instance_of::<PyDict>() && t.getattr("get")?.call1(("videoId",))?.is_truthy()? {
            out.append(t)?;
        }
    }
    Ok(out)
}

/// The playable track dicts of a chart shelf's track list — the
/// charts mix's own playable filter, the two-key spelling the chart
/// shelves' mixed payload shapes need (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_chart_playable_tracks`):
/// *tracks* walked through Python's iteration protocol, and only
/// dict items (the subclass-aware `isinstance` gate) pass whose id
/// read — `videoId` first through the mapping's own `get`,
/// `video_id` with the fresh `""` default only when the first read
/// answers falsy (the reference's `or`) — is truthy. Kept tracks
/// pass through as the objects they are, in the list's order.
#[pyfunction]
#[pyo3(signature = (tracks))]
pub(crate) fn ytm_chart_playable_tracks<'py>(
    py: Python<'py>,
    tracks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for t in tracks.try_iter()? {
        let t = t?;
        if t.is_instance_of::<PyDict>() {
            let first = t.getattr("get")?.call1(("videoId",))?;
            let vid = if first.is_truthy()? {
                first
            } else {
                t.getattr("get")?.call(("video_id", ""), None)?
            };
            if vid.is_truthy()? {
                out.append(t)?;
            }
        }
    }
    Ok(out)
}

/// The new-releases member of an explore payload — the pure core of
/// `YTMusicService.get_new_releases` (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_explore_new_releases`): over
/// the subclass-aware `isinstance(result, dict)` gate the
/// `new_releases` member is read through the mapping's own `get`
/// with a fresh empty list default and passed through — as the list
/// object it is — only when the subclass-aware
/// `isinstance(releases, list)` gate holds; every other shape
/// answers a fresh empty list.
#[pyfunction]
#[pyo3(signature = (result))]
pub(crate) fn ytm_explore_new_releases<'py>(
    py: Python<'py>,
    result: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if result.is_instance_of::<PyDict>() {
        let default = PyList::empty(py).into_any();
        let releases = result
            .getattr("get")?
            .call(("new_releases", default), None)?;
        if releases.is_instance_of::<PyList>() {
            return Ok(releases);
        }
    }
    Ok(PyList::empty(py).into_any())
}

/// The sort-params token for a `get_playlist` order name — the pure
/// core of `YTMusicService.get_playlist`'s order resolution
/// (`services/ytmusic.py`).
///
/// Mirrors the pure-Python reference (`_playlist_order_params`): the
/// `order or ""` truthiness gate (`None` and every other falsy
/// spelling read as the default order) and the `_ORDER_PARAMS`
/// table lookup — the table read from the service class at call time
/// (see `YTMusicService._ORDER_PARAMS`), so a patched table steers
/// both backends identically — answering the a_to_z / z_to_a /
/// recently_added browse `params` tokens, or `None` for the
/// playlist's server-side default. The ordered-fetch patch window
/// and the renewal machinery stay Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (order))]
pub(crate) fn ytm_playlist_order_params<'py>(
    py: Python<'py>,
    order: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let table = py
        .import("ytm_player.services.ytmusic")?
        .getattr("YTMusicService")?
        .getattr("_ORDER_PARAMS")?;
    let key = if order.is_truthy()? {
        order.clone()
    } else {
        PyString::new(py, "").into_any()
    };
    table.getattr("get")?.call1((key,))
}

// ── YTMusic RNG-driven selection engines (services/ytmusic.py) ────────

/// The shuffled, trimmed mix behind `YTMusicService.get_radio` — the
/// pure core of the wrapper's result assembly from
/// `services/ytmusic.py`, the RNG-driven half the seed-pool engine's
/// own contract left with the wrapper.
///
/// Mirrors the pure-Python reference (`_radio_mix`): the multi-seed
/// gate `seed_count > 1` through CPython's own `>` (single-seed
/// callers expect the seed track to lead, so their pool keeps its
/// first-sight order), the `random.shuffle(pool)` under it — routed
/// through the Python `random` module exactly like the QueueCore
/// engines route theirs, so `random.seed(...)` before the call and a
/// monkeypatched `random.shuffle` steer the native engine identically
/// to the fallback — shuffling the caller's pool in place, and the
/// `pool[:limit]` trim through CPython's own slice protocol. The
/// answer is the slice the reference passes on; the normalisation
/// stays Python-side in the wrapper.
#[pyfunction]
#[pyo3(signature = (pool, seed_count, limit))]
pub(crate) fn ytm_radio_mix<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
    seed_count: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let one = 1i64
        .into_pyobject(py)
        .expect("int conversion is infallible")
        .into_any();
    if seed_count.gt(&one)? {
        let random = PyModule::import(py, "random")?;
        random.call_method1("shuffle", (pool.clone(),))?;
    }
    // pool[:limit] — the slice through CPython's own protocols, the
    // None/limit/None window the reference's spelling compiles to, a
    // non-int limit raising the slice's own TypeError exactly like
    // the reference.
    let slice_type = PyModule::import(py, "builtins")?.getattr("slice")?;
    let window = slice_type.call1((py.None(), limit.clone(), py.None()))?;
    pool.get_item(&window)
}

/// The three-track seed sample a discovery mix draws — the pure core
/// of the sampling step every `YTMusicService.get_discovery_mix`
/// source shares (the charts, trending, home, liked-songs, artist and
/// history mixes), from `services/ytmusic.py`.
///
/// Mirrors the pure-Python reference (`_sample_seeds`): the draw size
/// `min(3, len(pool))` — the length through CPython's own sequence
/// protocol, so a lengthless pool raises the reference's own `len`
/// `TypeError`, capped at three and at the pool's own length — then
/// `random.sample(pool, k)` routed through the Python `random`
/// module, so `random.seed(...)` before the call and a monkeypatched
/// `random.sample` steer the native engine identically to the
/// fallback. The answer is the sample the module's own `sample`
/// produced, the tracks passing through as the objects they are.
#[pyfunction]
#[pyo3(signature = (pool))]
pub(crate) fn ytm_sample_seeds<'py>(
    py: Python<'py>,
    pool: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let random = PyModule::import(py, "random")?;
    let len = pool.len()?;
    let k = len.min(3);
    random.call_method1("sample", (pool.clone(), k))
}

/// The random artist pick behind the artist discovery mix's seed
/// selection — the pure core of the first draw in
/// `YTMusicService.get_discovery_mix`'s artist source, from
/// `services/ytmusic.py`.
///
/// Mirrors the pure-Python reference (`_pick_artist`):
/// `random.choice(artists)` routed through the Python `random`
/// module, so `random.seed(...)` before the call and a monkeypatched
/// `random.choice` steer the native engine identically to the
/// fallback. The artist passes through as the object it is; the
/// empty-artists guard stays Python-side in the wrapper (the
/// reference's `random.choice` would raise its own `IndexError`
/// on an empty sequence, which the guard keeps unreachable).
#[pyfunction]
#[pyo3(signature = (artists))]
pub(crate) fn ytm_pick_artist<'py>(
    py: Python<'py>,
    artists: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let random = PyModule::import(py, "random")?;
    Ok(random
        .call_method1("choice", (artists.clone(),))?
        .into_any())
}
