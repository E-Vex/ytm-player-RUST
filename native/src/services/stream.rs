//! Native cores for `src/ytm_player/services/stream.py` — resolve opts,
//! retry schedules, staleness/quality gates, cache-cap evictions and the
//! pending-publication verdicts (incl. the download error hint).

use crate::app::ipc::builtin_float;
use crate::services::history::builtin_int;
use crate::services::ytmusic::py_unpack_pair;
use crate::ui::pages::browse::field_get;
use crate::ui::pages::browse::first_truthy_get;
use crate::utils::pycompat::py_str;
use pyo3::conversion::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};
use regex::Regex;
use std::sync::LazyLock;

// ── Stream resolver glue (services/stream.py) ─────────────────────────────

/// The stream module's own clock — `time.time()` read through the
/// module's `time` attribute, exactly like the reference's global
/// lookup at call time. A test (or any caller) that swaps
/// `ytm_player.services.stream`'s `time` for a frozen namespace steers
/// the native engine exactly like the fallback: the clock the
/// reference would read is the clock the engine reads.
fn stream_module_clock(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let stream = PyModule::import(py, "ytm_player.services.stream")?;
    let time_module = stream.getattr("time")?;
    let clock = time_module.getattr("time")?;
    clock.call0()
}

/// The stream module's `QUALITY_FORMATS` table, read through the
/// module attribute — the very dict the fallback's own lookups run
/// against, a patched table included.
fn stream_quality_formats(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.stream")?.getattr("QUALITY_FORMATS")
}

/// The stream module's retry schedule, read through the module
/// attribute — the very `_RETRY_DELAYS` list the fallback's own
/// enumerate walk runs against, a patched schedule included.
fn stream_retry_schedule(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.stream")?.getattr("_RETRY_DELAYS")
}

/// The stream module's `_CACHE_MAX_SIZE` cap, read through the module
/// attribute — the very constant the fallback's own over-cap gate and
/// excess arithmetic run against, a patched cap included.
fn stream_cache_max_size(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.services.stream")?.getattr("_CACHE_MAX_SIZE")
}

/// `float(info.get(key) or 0)` — the truthiness gate feeding the
/// interpreter's own `float()` (see `builtin_float`), so the string
/// grammar, `__float__` dispatch and error shapes all come free.
fn truthy_or_float_zero<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let raw = field_get(py, info, key, None)?;
    let inner = if raw.is_truthy()? {
        raw
    } else {
        0i64.into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any()
    };
    builtin_float(py, &inner)
}

/// The cached-URL staleness gate — `services/stream.py`'s `_is_stale`,
/// applied by every read path (`_get_cached`, `is_expired`).
///
/// Mirrors the reference's one expression exactly: the clock read
/// through the module's own `time` name first (the left operand), the
/// `_EXPIRY_BUFFER_SECONDS` (300) subtraction through CPython's `-`
/// (its `__sub__` protocol — a non-numeric *expires_at* raises the
/// operator's own `TypeError`), and the `>=` through CPython's own
/// comparison, the verdict the comparison's truthiness. A cached URL
/// is stale once the clock has passed, or is within the buffer of,
/// its expiry — so playback is never handed a URL that dies
/// mid-stream.
#[pyfunction]
#[pyo3(signature = (expires_at))]
pub(crate) fn is_stale<'py>(py: Python<'py>, expires_at: &Bound<'py, PyAny>) -> PyResult<bool> {
    let now = stream_module_clock(py)?;
    let buffer = 300i64.into_py_any(py)?;
    let deadline = expires_at.sub(&buffer)?;
    now.ge(&deadline)
}

/// The available_at wait gate — the pure half of `StreamInfo.wait_seconds`
/// (`services/stream.py`), the pre-roll-ad wait `resolve()` sleeps through.
///
/// Mirrors the reference's `max(0.0, available_at - time.time())`:
/// the subtraction (through CPython's `-`, its `__sub__` protocol) is
/// evaluated first — the clock read through the module's own `time`
/// name on its right — and `max` walks its two operands exactly like
/// the reference's call: the `0.0` literal starts as the running
/// maximum and the subtraction replaces it only when it compares
/// strictly greater (CPython's `>`), so a NaN or negative wait answers
/// the `0.0` arm and an equal wait keeps the literal's value. The
/// answer is whichever operand won — the subtraction's own object
/// when it won, a fresh `0.0` float otherwise.
#[pyfunction]
#[pyo3(signature = (available_at))]
pub(crate) fn wait_seconds<'py>(
    py: Python<'py>,
    available_at: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let now = stream_module_clock(py)?;
    let elapsed = available_at.sub(&now)?;
    let zero = PyFloat::new(py, 0.0).into_any();
    if elapsed.gt(&zero)? {
        Ok(elapsed)
    } else {
        Ok(zero)
    }
}

/// The quality-input gate — `services/stream.py`'s
/// `_validate_quality`, the validation the resolver's `quality`
/// setter raises through.
///
/// Mirrors the reference: the membership test through CPython's own
/// `in` (the containment slot — hash/eq membership over the module's
/// own `QUALITY_FORMATS` table, read through the module attribute so a
/// patched table steers both backends identically), and the rejection
/// a `ValueError` carrying the reference's f-string message — the
/// value interpolated through `str()` and the tier list through the
/// interpreter's own `list()` then `str()` (the setter's value is
/// typed `str`, so the interpolation spells the same text either way).
#[pyfunction]
#[pyo3(signature = (value))]
pub(crate) fn validate_quality(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<()> {
    let table = stream_quality_formats(py)?;
    if table.contains(value)? {
        return Ok(());
    }
    // f"Unknown quality '{value}'. Choose from: {list(QUALITY_FORMATS)}"
    let list_type = py.import("builtins")?.getattr("list")?;
    let keys = list_type.call1((table,))?;
    let mut message = String::from("Unknown quality '");
    message.push_str(&py_str(value)?);
    message.push_str("'. Choose from: ");
    message.push_str(&py_str(&keys)?);
    Err(PyValueError::new_err(message))
}

/// The base yt-dlp options dict — `services/stream.py`'s
/// `_base_ydl_opts`, the pure half of `StreamResolver._build_ydl_opts`.
///
/// Mirrors the reference's dict literal key for key, insertion order
/// included: the `format` selector through the module's own
/// `QUALITY_FORMATS` table — `.get(quality, QUALITY_FORMATS["high"])`,
/// the default read eagerly before the lookup runs, the lookup the
/// dict's own `.get` method (hash/eq membership, a patched table
/// included) — the `quiet`/`no_warnings`/`noplaylist`/`skip_download`
/// `True` flags, the `logger` object passed through as the very
/// object the caller built, the `extract_flat`/`writeinfojson`/
/// `writethumbnail` `False` flags, and the nested `extractor_args`
/// mapping (`{"youtube": {"player_client": ["default", "android"]}}`,
/// fresh containers every call). The configured extras — `cookiefile`,
/// the CA bundle's `compat_opts`, `remote_components`, `js_runtimes`
/// — stay in the caller's `apply_configured_yt_dlp_options` pass.
#[pyfunction]
#[pyo3(signature = (quality, logger))]
pub(crate) fn stream_base_opts<'py>(
    py: Python<'py>,
    quality: &Bound<'py, PyAny>,
    logger: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let table = stream_quality_formats(py)?;
    // QUALITY_FORMATS["high"] — evaluated before .get runs, exactly
    // like the reference's argument evaluation.
    let default = table.get_item("high")?;
    // QUALITY_FORMATS.get(quality, default) — the dict's own method.
    let format = table.call_method1("get", (quality, default))?;

    let opts = PyDict::new(py);
    opts.set_item("format", format)?;
    opts.set_item("quiet", true)?;
    opts.set_item("no_warnings", true)?;
    opts.set_item("logger", logger)?;
    opts.set_item("extract_flat", false)?;
    opts.set_item("noplaylist", true)?;
    opts.set_item("skip_download", true)?;
    opts.set_item("writeinfojson", false)?;
    opts.set_item("writethumbnail", false)?;
    let player_client = PyList::new(py, ["default", "android"])?;
    let youtube = PyDict::new(py);
    youtube.set_item("player_client", player_client)?;
    let extractor_args = PyDict::new(py);
    extractor_args.set_item("youtube", youtube)?;
    opts.set_item("extractor_args", extractor_args)?;
    Ok(opts)
}

/// The extract_info → StreamInfo field assembly —
/// `services/stream.py`'s `_stream_info_fields`, the pure half of
/// `StreamResolver._try_resolve`.
///
/// Mirrors the reference's post-extraction walk read for read: the
/// direct `url` read (`info.get("url", "")`), the `available_at`
/// read through the `float(... or 0)` truthiness gate (the
/// interpreter's own `float()` — string grammar, `__float__`
/// dispatch and error shapes included), the `requested_formats`
/// fallback when the direct URL is falsy (the list read through its
/// own `or []` gate, each format probed with the reference's
/// `vcodec == "none" or acodec != "none"` condition — both
/// comparisons through CPython's own operators — the first match's
/// `url` and `available_at` replacing the direct reads, the walk
/// stopping at the `break`), and the `None` verdict when no stream
/// URL survived. The metadata reads follow the reference's own
/// gates: `acodec` through `.get("acodec", "unknown")`, `audio_ext`
/// through the `audio_ext`-or-`ext` chain (a falsy-but-present `ext`
/// passes through as the object it is, exactly like the reference's
/// `or` arm), `abr` through the `abr`-or-`tbr` chain with the
/// interpreter's own `int()` over the winner (float truncation,
/// decimal strings, `__int__`/`__index__` dispatch — a `0` int when
/// both miss), `duration` the same over its own single-key chain,
/// and `thumbnail` the bare `.get` read. The format name is the
/// acodec unless it compares equal to `"none"` (CPython's `!=`), in
/// which case the audio extension; the expiry is the module clock
/// plus the cache TTL (the interpreter's own `+`). The answer is the
/// eight `StreamInfo` field values in the dataclass's declared
/// order — `url`, `video_id`, `format`, `bitrate`, `duration`,
/// `expires_at`, `thumbnail_url`, `available_at` — with every value
/// passed through as the object it is; the `StreamInfo` construction
/// stays Python-side in the caller. Where the reference would raise —
/// a `.get`-less info or format object — the field reads stay total
/// (the `format_search_result` precedent); the info dict is yt-dlp's
/// own dict on every production path.
#[pyfunction]
#[pyo3(signature = (info, video_id))]
pub(crate) fn stream_info_fields<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
    video_id: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    // stream_url = info.get("url", "")
    let empty = PyString::new(py, "").into_any();
    let mut stream_url = field_get(py, info, "url", Some(empty))?;
    // available_at = float(info.get("available_at") or 0)
    let mut available_at = truthy_or_float_zero(py, info, "available_at")?;

    if !stream_url.is_truthy()? {
        // formats = info.get("requested_formats") or []
        let formats = match first_truthy_get(py, info, &["requested_formats"])? {
            Some(value) => value,
            None => PyList::empty(py).into_any(),
        };
        for fmt in formats.try_iter()? {
            let fmt = fmt?;
            // fmt.get("vcodec") == "none" or fmt.get("acodec") != "none"
            let vcodec = field_get(py, &fmt, "vcodec", None)?;
            let acodec = field_get(py, &fmt, "acodec", None)?;
            let vcodec_none = vcodec.eq("none")?;
            let acodec_not_none = acodec.ne("none")?;
            if vcodec_none || acodec_not_none {
                stream_url = field_get(py, &fmt, "url", Some(PyString::new(py, "").into_any()))?;
                available_at = truthy_or_float_zero(py, &fmt, "available_at")?;
                break;
            }
        }
    }

    if !stream_url.is_truthy()? {
        return Ok(None);
    }

    // acodec = info.get("acodec", "unknown")
    let acodec = field_get(
        py,
        info,
        "acodec",
        Some(PyString::new(py, "unknown").into_any()),
    )?;
    // audio_ext = info.get("audio_ext") or info.get("ext", "unknown")
    let audio_ext = match first_truthy_get(py, info, &["audio_ext"])? {
        Some(value) => value,
        None => field_get(
            py,
            info,
            "ext",
            Some(PyString::new(py, "unknown").into_any()),
        )?,
    };
    // abr = int(info.get("abr") or info.get("tbr") or 0)
    let abr = match first_truthy_get(py, info, &["abr", "tbr"])? {
        Some(value) => builtin_int(py, &value)?,
        None => 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any(),
    };
    // duration = int(info.get("duration") or 0)
    let duration = match first_truthy_get(py, info, &["duration"])? {
        Some(value) => builtin_int(py, &value)?,
        None => 0i64
            .into_pyobject(py)
            .expect("int conversion is infallible")
            .into_any(),
    };
    // thumbnail = info.get("thumbnail")
    let thumbnail = field_get(py, info, "thumbnail", None)?;
    // fmt_name = acodec if acodec != "none" else audio_ext
    let fmt_name = if acodec.ne("none")? {
        acodec
    } else {
        audio_ext
    };
    // expires_at = time.time() + _CACHE_TTL_SECONDS
    let now = stream_module_clock(py)?;
    let ttl = 18000i64.into_py_any(py)?;
    let expires_at = now.add(&ttl)?;

    Ok(Some(PyTuple::new(
        py,
        [
            stream_url,
            video_id.clone(),
            fmt_name,
            abr,
            duration,
            expires_at,
            thumbnail,
            available_at,
        ],
    )?))
}

/// The DownloadError hint — `services/stream.py`'s
/// `_download_error_hint`, the diagnostic the resolver's error
/// handler appends when a signed-in resolve failed looking
/// auth-related.
///
/// Mirrors the reference's gate: the hint only when the session
/// cookiejar is in use AND the error message matches the
/// `sign in|login_required|confirm you` pattern case-insensitively
/// (the regex crate's Unicode-aware `(?i)` — simple case folding over
/// the same ASCII keywords `re.I` folds, identical on every practical
/// message), the exact hint text with its leading blank and the
/// `ytm setup` advice, and the empty string otherwise. The
/// `_session_cookiejar_signature()` probe and the `str(exc)`
/// conversion stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (message, session_cookiejar_active))]
pub(crate) fn download_error_hint(message: &str, session_cookiejar_active: bool) -> String {
    if session_cookiejar_active && DOWNLOAD_HINT_RE.is_match(message) {
        " (stream cookiejar may be stale/revoked — try `ytm setup` to refresh it)".to_string()
    } else {
        String::new()
    }
}

/// The auth-related error-text pattern behind the DownloadError hint —
/// `re.search(r"sign in|login_required|confirm you", …, re.I)`'s
/// three literal branches, case-insensitive.
static DOWNLOAD_HINT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)sign in|login_required|confirm you").expect("valid download hint regex")
});

/// The expired-entry walk — `services/stream.py`'s
/// `_cache_expired_ids`, the prune phase behind the resolver cache's
/// maintenance (`_put_cache` and `prune_expired`).
///
/// Mirrors the reference's comprehension exactly: the clock read
/// through the module's own `time` name first (see
/// `stream_module_clock`), then one pass over `cache.items()` — the
/// method call, so a subclass override steers the walk — in the
/// mapping's own order, each entry unpacked as the reference's `for
/// vid, info in …` pair (see `py_unpack_pair`), each `expires_at` read
/// through CPython's attribute protocol and compared with the `>=`
/// through CPython's own rich comparison, the reference's operand
/// order — so every numeric shape and every error matches the
/// reference. The ids answer in the cache's iteration order; the dict
/// state stays Python-side (the caller holds the lock and performs
/// the deletions).
#[pyfunction]
#[pyo3(signature = (cache))]
pub(crate) fn stream_cache_expired_ids<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let now = stream_module_clock(py)?;
    let items = cache.call_method0("items")?;
    let mut expired: Vec<Bound<'py, PyAny>> = Vec::new();
    for entry in items.try_iter()? {
        let entry = entry?;
        let (vid, info) = py_unpack_pair(&entry)?;
        let expires_at = info.getattr("expires_at")?;
        if now.ge(&expires_at)? {
            expired.push(vid);
        }
    }
    Ok(PyList::new(py, expired).expect("list of ids cannot fail"))
}

/// The over-cap eviction walk — `services/stream.py`'s
/// `_cache_cap_evictions`, the second phase of `_put_cache`'s
/// maintenance, run once the expired entries are pruned.
///
/// Mirrors the reference exactly: the cap read through the module's
/// own `_CACHE_MAX_SIZE` attribute (a patched cap steers both
/// backends identically), the `len(cache) > cap` gate through
/// CPython's own comparison — within the cap answers the empty list —
/// then `sorted(cache, key=lambda vid: cache[vid].expires_at)`: the
/// key read through the subscript and attribute protocols once per
/// entry over the cache's own iteration order, the ranking through
/// CPython's own `<` on the key results (exactly the comparisons
/// `list.sort` runs), the stability the reference's `sorted()`
/// guarantees. Then `excess = len(cache) - cap` through CPython's own
/// `-` and the `sorted_ids[:excess]` slice through CPython's own
/// slice protocol — so every numeric shape and every error matches
/// the reference, operand orders included. The ids answer in eviction
/// order; the dict state stays Python-side.
#[pyfunction]
#[pyo3(signature = (cache))]
pub(crate) fn stream_cache_cap_evictions<'py>(
    py: Python<'py>,
    cache: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let cap = stream_cache_max_size(py)?;
    // len(cache) — through CPython's own len(), as the int object the
    // reference's comparisons and subtraction run against.
    let size = cache
        .len()?
        .into_pyobject(py)
        .expect("len() -> int is infallible")
        .into_any();
    // if len(cache) <= _CACHE_MAX_SIZE: return [] — the inverted
    // over-cap gate.
    if !size.gt(&cap)? {
        return Ok(PyList::empty(py));
    }
    // sorted(cache, key=…) — the decorate pass: cache[vid].expires_at
    // once per entry, in the cache's own iteration order.
    let mut keyed: Vec<(Bound<'py, PyAny>, Bound<'py, PyAny>)> = Vec::new();
    for vid in cache.try_iter()? {
        let vid = vid?;
        let expires_at = cache.get_item(&vid)?.getattr("expires_at")?;
        keyed.push((vid, expires_at));
    }
    // list.sort's `<`-only ordering over the key results, stable —
    // the first comparison error returns exactly like the reference's
    // sorted() would raise it.
    let mut sort_err: Option<PyErr> = None;
    keyed.sort_by(|a, b| {
        if sort_err.is_some() {
            // The outcome no longer matters — the error returns anyway.
            return std::cmp::Ordering::Equal;
        }
        match (a.1.lt(&b.1), b.1.lt(&a.1)) {
            (Ok(true), _) => std::cmp::Ordering::Less,
            (Ok(false), Ok(true)) => std::cmp::Ordering::Greater,
            (Ok(_), Ok(_)) => std::cmp::Ordering::Equal,
            (Err(err), _) | (_, Err(err)) => {
                sort_err = Some(err);
                std::cmp::Ordering::Equal
            }
        }
    });
    if let Some(err) = sort_err {
        return Err(err);
    }
    // sorted_ids — the fresh list sorted() answers.
    let ids: Vec<Bound<'py, PyAny>> = keyed.into_iter().map(|(vid, _)| vid).collect();
    let sorted_ids = PyList::new(py, ids).expect("list of ids cannot fail");
    // excess = len(cache) - cap, then sorted_ids[:excess] — both
    // through CPython's own protocols, a non-int excess raising the
    // slice's own TypeError exactly like the reference.
    let excess = size.sub(&cap)?;
    let slice_type = PyModule::import(py, "builtins")?.getattr("slice")?;
    let window = slice_type.call1((py.None(), excess, py.None()))?;
    // The slice through the getitem protocol — sorted_ids[window], the
    // reference's sorted_ids[:excess] spelled as the slice object it
    // compiles to.
    let evictions = sorted_ids.as_any().get_item(&window)?;
    Ok(evictions
        .downcast_into::<PyList>()
        .expect("a slice of a list is a list"))
}

/// The retry-delay iteration — `services/stream.py`'s `_retry_delays`,
/// the `(attempt, delay)` schedule walk behind
/// `StreamResolver._resolve_sync`'s retry loop.
///
/// Mirrors the reference's `list(enumerate(delays))` exactly: the
/// schedule read through the module's own `_RETRY_DELAYS` attribute
/// (a patched schedule steers both backends identically), then one
/// pass over it through the iterator protocol — the schedule's own
/// objects yielded as they come, each paired with its attempt index
/// counting from 0 exactly like the interpreter's own `enumerate` —
/// answering the fresh list of `(attempt, delay)` tuples the resolve
/// loop walks (and breaks out of early on the first success). The
/// sleep gate (`> 0`), the `time.sleep` call and the extraction
/// attempts stay Python-side in the caller.
#[pyfunction]
pub(crate) fn stream_retry_delays<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let schedule = stream_retry_schedule(py)?;
    let mut steps: Vec<Bound<'py, PyAny>> = Vec::new();
    for (attempt, delay) in schedule.try_iter()?.enumerate() {
        let delay = delay?;
        let index = attempt
            .into_pyobject(py)
            .expect("usize -> int is infallible")
            .into_any();
        steps.push(PyTuple::new(py, [index, delay])?.into_any());
    }
    Ok(PyList::new(py, steps).expect("list of pairs cannot fail"))
}

// ---------------------------------------------------------------------------
// yt-dlp instance gates (mirror the resolver's acquisition loop in
// ytm_player/services/stream.py)
// ---------------------------------------------------------------------------

/// The cookiejar-signature reuse gate — `services/stream.py`'s
/// `_ydl_cookiejar_current`, the gate behind `_get_ydl`'s locked
/// reuse arm.
///
/// Mirrors the reference's single comparison through CPython's own
/// `==` (the `__eq__` protocol — identity-matched singletons, the
/// element-wise tuple walk, reflected lookups and every error shape
/// the interpreter's): `None` matches `None` — an instance built
/// anonymous while streaming is still anonymous is reused — and the
/// signature tuples match element-wise, so a change to any of the
/// three files the signature stats (the stream cookiejar rewritten by
/// `ytm setup` or a mid-session auto-refresh, but also auth.json or
/// account.json replaced on its own) answers False and the caller
/// discards the instance and rebuilds one that loads the new cookies.
/// The signature probe itself (the settings read and the three file
/// stats — `_session_cookiejar_signature`, evaluated by the caller
/// only when an instance exists) and the instance state stay
/// Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (cached_sig, current_sig))]
pub(crate) fn stream_ydl_cookiejar_current(
    cached_sig: &Bound<'_, PyAny>,
    current_sig: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    cached_sig.eq(current_sig)
}

/// The mid-build publication verdict — `services/stream.py`'s
/// `_ydl_publish_verdict`, the gate behind `_get_ydl`'s second locked
/// section, consulted after the unlocked `_build_ydl_opts()` pass.
///
/// Mirrors the reference's two checks in order: the
/// `instance is not None` identity test (another thread's build
/// published an instance while this caller's opts were building —
/// adopt the winner's and never construct a duplicate) answered
/// `"adopt"`, then the `current_generation != snapshot_generation`
/// inequality through CPython's own `!=` (a `_reset_ydl()` landing
/// while the opts built invalidates them — pre-reset settings are
/// baked in — so the loop discards the just-built opts and re-reads
/// current settings) answered `"retry"`, and otherwise the
/// `"install"` verdict that publishes the built instance with the
/// signature it vouched for. The locks, the `_active_resolves`
/// accounting, the YoutubeDL construction and the cached instance
/// fields stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (instance, snapshot_generation, current_generation))]
pub(crate) fn stream_ydl_publish_verdict(
    instance: &Bound<'_, PyAny>,
    snapshot_generation: &Bound<'_, PyAny>,
    current_generation: &Bound<'_, PyAny>,
) -> PyResult<&'static str> {
    if !instance.is_none() {
        return Ok("adopt");
    }
    if current_generation.ne(snapshot_generation)? {
        return Ok("retry");
    }
    Ok("install")
}

// ---------------------------------------------------------------------------
// Pending-future bookkeeping (mirror the resolver's request dedup in
// ytm_player/services/stream.py)
// ---------------------------------------------------------------------------

/// The pending-join verdict — `services/stream.py`'s
/// `_pending_join_verdict`, the join gate behind
/// `StreamResolver._current_pending` (and through it `_resolve`'s
/// dedup arm and `prefetch`'s already-resolving check).
///
/// Mirrors the reference's slot read-back exactly: the raw
/// `dict.get` answer it is handed is the None singleton when the
/// slot is empty — no in-flight resolve to join — and otherwise the
/// entry's own `(generation, future)` pair, the entry's generation
/// fetched through CPython's own subscript and compared through
/// CPython's own `!=`, so a slot registered before a `clear_cache()`
/// bumped the generation answers no-join (its result comes from the
/// old resolver setup) while a same-generation slot answers its
/// future — the very object the entry holds, fetched through the
/// same subscript — for the caller's `asyncio.shield`. The slot
/// state (the `_pending` dict and its entries) and the awaits stay
/// Python-side in the resolver.
#[pyfunction]
#[pyo3(signature = (entry, generation))]
pub(crate) fn stream_pending_join_verdict<'py>(
    py: Python<'py>,
    entry: &Bound<'py, PyAny>,
    generation: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if entry.is_none() {
        return Ok(py.None().into_bound(py));
    }
    let entry_generation = entry.get_item(0)?;
    if entry_generation.ne(generation)? {
        return Ok(py.None().into_bound(py));
    }
    entry.get_item(1)
}

/// The pending-drop verdict — `services/stream.py`'s
/// `_pending_drop_verdict`, the identity gate behind `_resolve`'s
/// finally-block slot cleanup.
///
/// Mirrors the reference's short-circuited identity chain exactly:
/// `entry is not None and entry[1] is future` — the None check by
/// identity, the slot's future fetched through CPython's own
/// subscript and compared to the owner's future by identity, never
/// equality. An empty slot is never dropped (nothing to remove), and
/// a slot is dropped only when it still holds this owner's very
/// future — so a resolve from before a `clear_cache()`, whose slot a
/// newer same-video request has taken over, leaves the newer
/// request's entry in place for its own waiters. The `del` itself
/// and the slot state stay Python-side in the resolver.
#[pyfunction]
#[pyo3(signature = (entry, future))]
pub(crate) fn stream_pending_drop_verdict(
    entry: &Bound<'_, PyAny>,
    future: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if entry.is_none() {
        return Ok(false);
    }
    Ok(entry.get_item(1)?.is(future))
}

/// The pending publication gate — `services/stream.py`'s
/// `_pending_publication_gate`, the unresolved gate behind the shared
/// pending future's three publications.
///
/// Mirrors the reference's `not future.done()` exactly: the `done()`
/// read through the future's own method protocol (a real
/// `asyncio.Future` answers a bool; a stand-in answers whatever it
/// answers), then inverted through CPython's own truthiness — the
/// interpreter's `not` over whatever came back. True while the
/// future still wants a resolution, so the owner's `set_result` (the
/// try arm), `set_exception` (the except arm) or `cancel` (the
/// finally arm, for an owner cancelled before the future resolved —
/// a bare `BaseException` bypasses the except arm, and without the
/// cancel concurrent waiters would hang forever on the orphaned
/// future) may land on it; False once the future is resolved,
/// cancelled or exception-set, so a publication never raises
/// `InvalidStateError` over a resolution that already landed. The
/// publications themselves and the future state stay Python-side in
/// the resolver.
#[pyfunction]
#[pyo3(signature = (future))]
pub(crate) fn stream_pending_publication_gate(future: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(!future.call_method0("done")?.is_truthy()?)
}
