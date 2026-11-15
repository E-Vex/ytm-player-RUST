//! Native cores for `src/ytm_player/services/player.py` — libmpv
//! discovery, end-file verdicts, seek/volume clamps and dispatch arms.

use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyInt};

// ---------------------------------------------------------------------------
// Player glue engines (mirror ytm_player/services/player.py)
// ---------------------------------------------------------------------------

/// The player module's own monotonic clock — `time.monotonic()` read
/// through the module's `time` attribute, exactly like the reference's
/// global lookup at call time. A test (or any caller) that swaps
/// `ytm_player.services.player`'s `time` for a frozen namespace steers
/// the native engine exactly like the fallback: the clock the reference
/// would read is the clock the engine reads.
fn player_module_clock(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let player = PyModule::import(py, "ytm_player.services.player")?;
    let time_module = player.getattr("time")?;
    let clock = time_module.getattr("monotonic")?;
    clock.call0()
}

/// The Homebrew libmpv discovery walk — `services/player.py`'s
/// `_find_brew_libmpv`, the import-time glue that finds libmpv in the
/// brew prefixes and the user-local prefix ctypes.util.find_library
/// cannot see (#90, #101, #104).
///
/// Mirrors the pure-Python reference exactly: the pattern pair selected
/// by `sys.platform` read through the interpreter's own `sys` module
/// (the `==` CPython's own, so a patched platform steers both backends
/// identically — the tests exercise the darwin pair off-darwin), the
/// prefix list read through the player module's own `_BREW_LIB_DIRS`
/// name (a patched list included), then one pass over the prefixes —
/// each `is_dir()` and `glob(pattern)` called through the prefix
/// object's own protocol, each hit list ordered through CPython's own
/// `sorted` (Path's ordering and its comparison errors are the
/// interpreter's), the pattern order the reference's (`libmpv.so`
/// before `libmpv.so.*`, the unversioned name preferred), the first
/// hit stringified through `str()` — with every per-prefix probe's
/// `OSError` swallowed exactly like the reference's
/// `except OSError: continue` and every other error propagated. The
/// first surviving hit's path is answered, or `None` when no prefix
/// produced one. The `ctypes.util.find_library("mpv")` gate around the
/// reference's call site stays Python-side in the module import.
#[pyfunction]
pub(crate) fn player_find_libmpv(py: Python<'_>) -> PyResult<Option<String>> {
    // sys.platform — read through the interpreter's own sys module, the
    // reference's own global lookup.
    let platform = PyModule::import(py, "sys")?.getattr("platform")?;
    let patterns: [&str; 2] = if platform.eq("darwin")? {
        ["libmpv.dylib", "libmpv.*.dylib"]
    } else {
        ["libmpv.so", "libmpv.so.*"]
    };
    let dirs = PyModule::import(py, "ytm_player.services.player")?.getattr("_BREW_LIB_DIRS")?;
    let sorted_fn = py.import("builtins")?.getattr("sorted")?;
    for lib_dir in dirs.try_iter()? {
        let lib_dir = lib_dir?;
        // The reference's per-prefix body — every probe inside the one
        // `except OSError: continue` arm.
        let probe: PyResult<Option<String>> = (|| {
            if !lib_dir.call_method0("is_dir")?.is_truthy()? {
                return Ok(None);
            }
            for pattern in patterns {
                // hits = sorted(lib_dir.glob(pattern))
                let hits = sorted_fn.call1((lib_dir.call_method1("glob", (pattern,))?,))?;
                if hits.is_truthy()? {
                    // return str(hits[0])
                    let first = hits.get_item(0)?;
                    return Ok(Some(first.str()?.extract::<String>()?));
                }
            }
            Ok(None)
        })();
        match probe {
            Ok(Some(path)) => return Ok(Some(path)),
            Ok(None) => continue,
            // except OSError: continue — the subclass-aware check, so
            // every OSError shape the filesystem probes raise is
            // swallowed exactly like the reference's handler.
            Err(err) if err.is_instance_of::<PyOSError>(py) => continue,
            Err(err) => return Err(err),
        }
    }
    Ok(None)
}

/// The end-file verdict — `services/player.py`'s `_end_file_verdict`,
/// the event-matching gate behind `Player._end_file_data`'s handling of
/// mpv's end-file callbacks.
///
/// Mirrors the reference's checks in their exact order, every one
/// through CPython's own protocol: the instance identity test first
/// (`is`, pointer identity — an end-file from a discarded or recovered
/// instance is never the live player's, answered `"drop"`), then the
/// loading gate (`_loading` truthiness — a loadfile reply still in
/// flight parks the event instead of consuming it, answered `"park"`),
/// then the four-way match answered `"ignore"` when any conjunct fails
/// — the `isinstance(entry_id, int)` type gate (subclass-aware, so a
/// `bool` passes exactly like CPython's isinstance), the
/// `entry_id != current_entry_id` through CPython's own `!=`, the
/// `current_track is None` identity check, and the
/// `reason not in (0, 4)` membership spelled as tuple.__contains__'s
/// own walk (CPython's `==` against each element, the searched object
/// the left operand, stopping at the first hit) — and, everything
/// passed, the `reason == 4` fork that separates mpv's error end
/// (reason 4, answered `"error"`) from the natural end (reason 0,
/// answered `"track_end"`). The short-circuit order is the
/// reference's, so every error matches too. The locks, the
/// pending-event buffer, the payload assembly and the dispatch stay
/// Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (instance, current_instance, loading, entry_id, current_entry_id, current_track, reason))]
pub(crate) fn player_end_file_verdict<'py>(
    instance: &Bound<'py, PyAny>,
    current_instance: &Bound<'py, PyAny>,
    loading: &Bound<'py, PyAny>,
    entry_id: &Bound<'py, PyAny>,
    current_entry_id: &Bound<'py, PyAny>,
    current_track: &Bound<'py, PyAny>,
    reason: &Bound<'py, PyAny>,
) -> PyResult<&'static str> {
    // instance is not self._mpv — pointer identity, CPython's `is`.
    if !instance.is(current_instance) {
        return Ok("drop");
    }
    // if self._loading — truthiness, the reference's own gate.
    if loading.is_truthy()? {
        return Ok("park");
    }
    // The reference's four-conjunct match, short-circuited in its
    // order: not isinstance(entry_id, int) or entry_id != current or
    // current_track is None or reason not in (0, 4).
    let matched = entry_id.is_instance_of::<PyInt>()
        && !entry_id.ne(current_entry_id)?
        && !current_track.is_none()
        && (reason.eq(0i64)? || reason.eq(4i64)?);
    if !matched {
        return Ok("ignore");
    }
    // if reason == 4 — the error fork, through CPython's own ==.
    if reason.eq(4i64)? {
        return Ok("error");
    }
    Ok("track_end")
}

/// The seek-target clamp — `services/player.py`'s `_clamp_seek`, the
/// pure half of `Player._clamp_seek_target` behind `seek` and
/// `seek_absolute`.
///
/// Mirrors the reference exactly: the `duration > 0` gate through
/// CPython's own `>` (the int literal `0` the reference's right
/// operand — an unknown duration, `0.0` or negative, leaves the target
/// uncapped; a NaN duration compares False and leaves it uncapped
/// too), then `min(target, duration)` walked exactly like CPython's
/// own `min` call — the target starts as the running minimum and the
/// duration replaces it only when it compares strictly less (so a NaN
/// on either side keeps the target) — and finally `max(0.0, target)`
/// walked the same way — the `0.0` literal starts as the running
/// maximum and the target replaces it only when it compares strictly
/// greater (so a NaN target clamps to the `0.0` arm). The operands
/// pass through as the objects they are; the `self.duration` read
/// stays Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (target, duration))]
pub(crate) fn player_clamp_seek<'py>(
    py: Python<'py>,
    target: &Bound<'py, PyAny>,
    duration: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut clamped = target.clone();
    // if duration > 0 — through CPython's own >, the int literal the
    // right operand.
    if duration.gt(0i64)? {
        // min(target, duration) — CPython's min walk: the duration
        // takes over only when it compares strictly less than the
        // running minimum.
        if duration.lt(target)? {
            clamped = duration.clone();
        }
    }
    // max(0.0, clamped) — CPython's max walk: the 0.0 literal starts
    // as the running maximum; the clamped target replaces it only
    // when it compares strictly greater.
    let zero = PyFloat::new(py, 0.0).into_any();
    if clamped.gt(&zero)? {
        Ok(clamped)
    } else {
        Ok(zero)
    }
}

/// The volume clamp — `services/player.py`'s `set_volume` clamp, the
/// `[0, 100]` level gate behind every volume write (the direct `set`
/// and the relative `change_volume` that funnels into it).
///
/// Mirrors the reference's `max(0, min(100, level))` exactly: the
/// inner `min` walked like CPython's own call — the int literal `100`
/// starts as the running minimum and the level takes over only when
/// it compares strictly less — then the outer `max` walked the same
/// way — the int literal `0` starts as the running maximum and the
/// clamped level replaces it only when it compares strictly greater —
/// so boundary values answer the literal arms and a non-numeric level
/// raises the comparison's own `TypeError`. The mpv assignment and the
/// VOLUME_CHANGE dispatch stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (level))]
pub(crate) fn player_clamp_volume<'py>(
    py: Python<'py>,
    level: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // min(100, level) — CPython's min walk: the level takes over only
    // when it compares strictly less than the running minimum.
    let hundred = 100i64.into_pyobject(py)?.into_any();
    let clamped = if level.lt(&hundred)? {
        level.clone()
    } else {
        hundred
    };
    // max(0, clamped) — CPython's max walk: the clamped level replaces
    // the running maximum only when it compares strictly greater.
    let zero = 0i64.into_pyobject(py)?.into_any();
    if clamped.gt(&zero)? {
        Ok(clamped)
    } else {
        Ok(zero)
    }
}

/// The position-dispatch throttle — `services/player.py`'s
/// `_position_due`, the gate behind `_on_time_pos_change`'s
/// POSITION_CHANGE dispatches (lyrics + UI, ~2 per second).
///
/// Mirrors the reference exactly: the clock read through the module's
/// own `time` name first (see `player_module_clock` — a patched or
/// frozen clock steers both backends identically), then
/// `now - last_dispatch` through CPython's own `-` and the
/// `< 0.4` comparison against the float window through CPython's own
/// `<` — so the boundary (`elapsed == 0.4`) is due, a younger elapsed
/// is throttled, and a non-numeric last-dispatch raises the
/// subtraction's own `TypeError`. Answers `None` when the dispatch is
/// throttled, and otherwise the very `now` the gate read — the value
/// the caller stamps as the new last-dispatch and owns until the next
/// due dispatch. The `value is not None` gate and the dispatch stay
/// Python-side in the observer.
#[pyfunction]
#[pyo3(signature = (last_dispatch))]
pub(crate) fn player_position_due<'py>(
    py: Python<'py>,
    last_dispatch: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let now = player_module_clock(py)?;
    let elapsed = now.sub(last_dispatch)?;
    let window = PyFloat::new(py, 0.4).into_any();
    if elapsed.lt(&window)? {
        Ok(None)
    } else {
        Ok(Some(now))
    }
}

/// The command-supersession gate — `services/player.py`'s
/// `_command_current`, the predicate behind `_dispatch`'s
/// `still_current` closure and `_finish_load`'s acceptance check.
///
/// Mirrors the reference exactly: a `None` command (the dispatch
/// default — events a caller dispatched without a play attempt to
/// vouch for) is unconditionally current, answered `True`; otherwise
/// `command == generation` through CPython's own `==` (equal-valued
/// non-identical objects match) and `current_track is not None`
/// identity — so a superseded play attempt's callbacks are dropped
/// even once scheduled, and a stop that cleared the track invalidates
/// a still-matching command's queued dispatches. `_finish_load`'s
/// acceptance check runs the same predicate — its command is never
/// `None`, so the first arm never fires there. The locks and the
/// state they guard stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (command, generation, current_track))]
pub(crate) fn player_command_current(
    command: &Bound<'_, PyAny>,
    generation: &Bound<'_, PyAny>,
    current_track: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if command.is_none() {
        return Ok(true);
    }
    Ok(command.eq(generation)? && !current_track.is_none())
}

/// The event-dispatch arm — `services/player.py`'s `_dispatch_arm`, the
/// branch decision behind `Player._dispatch`'s per-callback walk: which
/// of the six arms a registered callback takes.
///
/// Mirrors the reference's nested branch exactly and in its own order:
/// a live loop first (`loop is not None and not loop.is_closed()` —
/// computed Python-side, the loop object's own `is_closed` probe
/// staying there) — a coroutine callback schedules through
/// `call_soon_threadsafe` under the captured dispatch context when one
/// exists (`"async_ctx"`) or bare otherwise (`"async"`), a sync
/// callback the same pair (`"sync_ctx"` / `"sync"`); a dead loop runs
/// only sync callbacks — a coroutine is dropped with the warning the
/// caller logs (`"drop"`), a sync callback runs inline under the
/// caller's `still_current` gate (`"direct"`). The `call_soon_threadsafe`
/// calls, the context object, the task bookkeeping, the callbacks
/// themselves and every log line stay Python-side in the caller.
#[pyfunction]
#[pyo3(signature = (loop_live, is_coroutine, has_context))]
pub(crate) fn player_dispatch_arm(
    loop_live: bool,
    is_coroutine: bool,
    has_context: bool,
) -> &'static str {
    if loop_live {
        if is_coroutine {
            if has_context {
                "async_ctx"
            } else {
                "async"
            }
        } else if has_context {
            "sync_ctx"
        } else {
            "sync"
        }
    } else if is_coroutine {
        "drop"
    } else {
        "direct"
    }
}

/// The mpv log-event level routing — `services/player.py`'s
/// `_mpv_log_level`, the level map behind `_on_mpv_log`, the log
/// handler the mpv instance is constructed with.
///
/// Mirrors the reference's per-call dict walk exactly: the four mpv
/// level strings map onto Python logging's own constants — `fatal` →
/// `logging.CRITICAL`, `error` → `logging.ERROR`, `warn` →
/// `logging.WARNING`, `info` → `logging.INFO` — and every other string
/// answers `logging.DEBUG`. Each constant is read through the player
/// module's own `logging` name at call time, exactly like the
/// reference's dict construction reads its globals, so a patched or
/// customised module attribute steers both backends identically — the
/// pattern `player_module_clock` established for the module clock. The
/// `logger.log` call — the `"mpv[%s]: %s"` format with the prefix and
/// the `message.rstrip()` — stays Python-side in the handler.
#[pyfunction]
#[pyo3(signature = (level))]
pub(crate) fn player_mpv_log_level<'py>(
    py: Python<'py>,
    level: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let logging = PyModule::import(py, "ytm_player.services.player")?.getattr("logging")?;
    let name = match level {
        "fatal" => "CRITICAL",
        "error" => "ERROR",
        "warn" => "WARNING",
        "info" => "INFO",
        _ => return logging.getattr("DEBUG"),
    };
    logging.getattr(name)
}
