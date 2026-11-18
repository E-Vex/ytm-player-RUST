//! Native cores for `src/ytm_player/app/_playback.py` — the playback
//! failure policy, the history-report poll step, the stale-attempt
//! consume gates, the poll-step failure reset, the track-change
//! notification, the prefetch gate, the queue-refill plan, the
//! local-history finalize and insert-finalize plans, and the
//! history-report timer arm pair.

use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList};

use crate::services::history::builtin_int;
use crate::utils::pycompat::py_str;

// ---------------------------------------------------------------------------
// Playback coordination engines (mirror ytm_player/app/_playback.py)
// ---------------------------------------------------------------------------

/// The playback module's own `_MAX_CONSECUTIVE_FAILURES` cap, read
/// through the module attribute — the very constant the fallback's own
/// threshold arm runs against, a patched value included.
fn playback_max_consecutive_failures(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.app._playback")?.getattr("_MAX_CONSECUTIVE_FAILURES")
}

/// The playback module's own `_YTM_HISTORY_POLL_SECONDS` interval, read
/// through the module attribute — the very constant the fallback's own
/// timer arm runs against, a patched value included.
fn playback_ytm_history_poll_seconds(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    PyModule::import(py, "ytm_player.app._playback")?.getattr("_YTM_HISTORY_POLL_SECONDS")
}

/// The playback-failure policy — `app/_playback.py`'s
/// `_handle_play_failure` decision core, the verdict every failed play
/// runs through once the consecutive-failure counter has been bumped.
///
/// Mirrors the reference's walk exactly, every comparison through
/// CPython's own protocol: the repeat gate `repeat_mode == "one"`
/// first — a failing track in repeat-one must not skip itself forever,
/// so the verdict is `stop` and nothing else is consulted. Then the
/// threshold `consecutive_failures < _MAX_CONSECUTIVE_FAILURES` — the
/// cap read through the playback module's own name at call time (a
/// patched cap steers both backends identically), the comparison
/// CPython's own `<` so a non-numeric count raises the comparison's
/// own `TypeError`. Under the cap, the queue's own bound `next_track`
/// method is pulled — called exactly here, exactly once, the
/// reference's own advance-and-read — and a truthy answer is the
/// `advance` verdict carrying the very track object the queue handed
/// back (the reference's `if next_track:` truthiness, never identity),
/// a falsy one the `end_of_queue` verdict. At or over the cap the
/// verdict is `escalate`, with `reset_resolver` true only when
/// `failure_kind` is not `None` (the identity check, short-circuited
/// before the resolver is touched) and *resolver* is truthy through
/// CPython's own `bool` protocol — the systemic failures get a fresh
/// yt-dlp instance, the no-video-id path does not. The counter bump
/// and reset, the queue state, the resolver reset itself, the toasts
/// and the deferred advance stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (repeat_mode, consecutive_failures, failure_kind, resolver, next_track))]
pub(crate) fn playback_failure_policy<'py>(
    py: Python<'py>,
    repeat_mode: &Bound<'py, PyAny>,
    consecutive_failures: &Bound<'py, PyAny>,
    failure_kind: &Bound<'py, PyAny>,
    resolver: &Bound<'py, PyAny>,
    next_track: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // repeat_mode == "one" — CPython's own ==, the reference's repeat gate.
    if repeat_mode.eq("one")? {
        let out = PyDict::new(py);
        out.set_item("action", "stop")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // consecutive_failures < _MAX_CONSECUTIVE_FAILURES — CPython's own
    // <, the cap read through the playback module's own name.
    let cap = playback_max_consecutive_failures(py)?;
    if consecutive_failures.lt(&cap)? {
        // next_track() — the queue's own bound method, pulled exactly
        // here and exactly once, the reference's advance-and-read.
        let track = next_track.call0()?;
        if track.is_truthy()? {
            let out = PyDict::new(py);
            out.set_item("action", "advance")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("track", track)
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
        let out = PyDict::new(py);
        out.set_item("action", "end_of_queue")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // failure_kind is not None and resolver — the identity check
    // short-circuited before the resolver's truthiness, the reference's
    // own gate order.
    let reset_resolver = !failure_kind.is_none() && resolver.is_truthy()?;
    let out = PyDict::new(py);
    out.set_item("action", "escalate")
        .expect("dict insertion with a str key cannot fail");
    out.set_item("reset_resolver", reset_resolver)
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The history-report poll step — the shared decision core behind
/// `app/_playback.py`'s two report timer chains,
/// `_report_local_play` and `_report_ytm_play`, consulted on every
/// timer tick: whether the play is reportable yet, whether the chain
/// keeps polling, and what idle count the next poll carries.
///
/// Mirrors the reference's walk exactly, every comparison through
/// CPython's own protocol: the threshold gate `listened <=
/// min_listen_seconds` — the boundary kept on the poll side, so a
/// listen exactly at the configured minimum is not yet reportable, and
/// a non-numeric listen raises the comparison's own `TypeError`.
/// Above the threshold the verdict is `report`. At or under it, a
/// live player (`current_track is not None` — the identity check, not
/// truthiness, so only the genuine None clear counts as idle) answers
/// `poll` with the idle count reset to zero; a None current track
/// walks the two-poll stability window — `idle_polls + 1 >= 2`, the
/// bump through CPython's own `+` and the window edge through
/// CPython's own `>=` — answering `stop` once two consecutive polls
/// found the player idle (the window absorbing the transient
/// current_track clear a duplicate mpv end-file causes on natural
/// advance), and `poll` carrying the bumped idle count otherwise (the
/// bumped object itself, the reference's own `idle_polls + 1` result).
/// The position reads, the min-listen resolution, the timer re-arms
/// and the report workers stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (listened, min_listen_seconds, current_track, idle_polls))]
pub(crate) fn playback_history_report_step<'py>(
    py: Python<'py>,
    listened: &Bound<'py, PyAny>,
    min_listen_seconds: &Bound<'py, PyAny>,
    current_track: &Bound<'py, PyAny>,
    idle_polls: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    // listened <= min_listen_seconds — CPython's own <=, the boundary
    // kept on the poll side.
    if listened.le(min_listen_seconds)? {
        // current_track is None — identity, not truthiness.
        if current_track.is_none() {
            // idle_polls + 1 >= 2 — the bump through CPython's own +,
            // the window edge through CPython's own >=.
            let one: Bound<'py, PyAny> = 1i64.into_pyobject(py)?.into_any();
            let bumped = idle_polls.add(&one)?;
            let two: Bound<'py, PyAny> = 2i64.into_pyobject(py)?.into_any();
            if bumped.ge(&two)? {
                let out = PyDict::new(py);
                out.set_item("action", "stop")
                    .expect("dict insertion with a str key cannot fail");
                return Ok(out);
            }
            let out = PyDict::new(py);
            out.set_item("action", "poll")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("idle_polls", bumped)
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
        // A live track resets the idle count — the reference's else arm.
        let out = PyDict::new(py);
        out.set_item("action", "poll")
            .expect("dict insertion with a str key cannot fail");
        out.set_item("idle_polls", 0i64)
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    let out = PyDict::new(py);
    out.set_item("action", "report")
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The stale-attempt consume gate — the shared decision core behind
/// `app/_playback.py`'s `_on_player_error` and `_on_track_end`: whether
/// a player-event payload's attempt token still names the live play, so
/// its handler may consume it.
///
/// Mirrors the reference's four-arm or-chain exactly, every comparison
/// through CPython's own protocol, each arm consulted in the
/// reference's own order: an *attempt* that fails `isinstance(..., int)`
/// (the `PyLong_Check` acceptance set — a `bool` is an `int`, exactly
/// like CPython) is not current; an attempt that no longer equals
/// *play_generation* was superseded by a newer committed play; an
/// attempt at or under the handler's own *handled_attempt* counter was
/// already consumed by an earlier event (a duplicate end-file or a
/// duplicate error for the same play) — the comparison CPython's own
/// `<=`, so a non-numeric counter raises the comparison's own
/// `TypeError`; and an *active_stop_generation* that disagrees with
/// *queue_stop_generation* names a stop that superseded the play. Every
/// other attempt is current: the caller consumes it by writing it back
/// as its own handled counter. The event unpacking, the counter writes,
/// the request-ownership checks and the awaits stay Python-side in the
/// mixin.
#[pyfunction]
#[pyo3(signature = (
    attempt,
    play_generation,
    handled_attempt,
    active_stop_generation,
    queue_stop_generation
))]
pub(crate) fn playback_attempt_gate(
    attempt: &Bound<'_, PyAny>,
    play_generation: &Bound<'_, PyAny>,
    handled_attempt: &Bound<'_, PyAny>,
    active_stop_generation: &Bound<'_, PyAny>,
    queue_stop_generation: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    // isinstance(attempt, int) — the PyLong_Check acceptance set, bools
    // included.
    if !attempt.is_instance_of::<PyInt>() {
        return Ok(false);
    }
    // attempt != play_generation — CPython's own !=.
    if attempt.ne(play_generation)? {
        return Ok(false);
    }
    // attempt <= handled_attempt — CPython's own <=.
    if attempt.le(handled_attempt)? {
        return Ok(false);
    }
    // active_stop_generation != queue_stop_generation — CPython's own !=.
    if active_stop_generation.ne(queue_stop_generation)? {
        return Ok(false);
    }
    Ok(true)
}

/// The poll-step failure reset — `app/_playback.py`'s
/// `_poll_position` decision core: whether the position poll just
/// proved the current play's audio is advancing, the one point the
/// consecutive-failure counter and the recovery marker are cleared.
///
/// Mirrors the reference's and-chain exactly, every comparison through
/// CPython's own protocol, every arm consulted in the reference's own
/// order: `position > 0` (audio actually advancing — TRACK_CHANGE fires
/// before mpv has opened the stream, so only a positive position is
/// proof a play attempt succeeded), the failure-state arm — a truthy
/// *consecutive_failures* count or a *recovery_generation* that is not
/// `None` (the identity check — the one retry marker), so an untouched
/// state skips the rest — then the *player* object's own `is_playing`
/// and `current_attempt`, each read exactly here, exactly when the
/// reference's chain reaches it, so a short-circuited chain never
/// touches the player at all, and `current_attempt == play_generation`
/// (CPython's own `==`) so the advancing audio really is the play this
/// poll tracks. The counter and marker writes, the position read for
/// the bar update, the MPRIS / macOS / Last.fm fan-out and the timers
/// stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (position, consecutive_failures, recovery_generation, player, play_generation))]
pub(crate) fn playback_poll_failure_reset(
    position: &Bound<'_, PyAny>,
    consecutive_failures: &Bound<'_, PyAny>,
    recovery_generation: &Bound<'_, PyAny>,
    player: &Bound<'_, PyAny>,
    play_generation: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    // position > 0 — CPython's own >.
    if !position.gt(0)? {
        return Ok(false);
    }
    // consecutive_failures or recovery_generation is not None — the
    // truthiness read short-circuited before the identity check.
    if !(consecutive_failures.is_truthy()? || !recovery_generation.is_none()) {
        return Ok(false);
    }
    // player.is_playing — read exactly here, exactly when the
    // reference's chain reaches it.
    let is_playing = player.getattr("is_playing")?;
    if !is_playing.is_truthy()? {
        return Ok(false);
    }
    // player.current_attempt == play_generation — CPython's own ==.
    let current_attempt = player.getattr("current_attempt")?;
    current_attempt.eq(play_generation)
}

/// The track-change notification message — `app/_playback.py`'s
/// `_on_track_change` toast assembly: the notification copy a track
/// change shows when `[notifications] enabled` is on.
///
/// Mirrors the reference exactly: the title, artist and album are read
/// through the track's own `get` method (a dict subclass overriding it
/// steers both backends identically) with the reference's own defaults
/// — `"Unknown"` for a missing title or artist, `""` for a missing
/// album — then the configured *fmt* template runs through CPython's
/// own `str.format(title=..., artist=..., album=...)`, so every format
/// spec and conversion is the interpreter's own. A `KeyError` (a
/// placeholder the payload does not name) or a `ValueError` (a
/// malformed template) falls back to the `"{title} — {artist}"`
/// spelling — the em-dash separator and the f-string's own `str()`
/// conversions — exactly the reference's `except (KeyError, ValueError)`
/// arm; anything else propagates. The enabled gate, the timeout read
/// and the `notify` call stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (track, fmt))]
pub(crate) fn playback_notification_message<'py>(
    py: Python<'py>,
    track: &Bound<'py, PyAny>,
    fmt: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // title / artist / album — the track's own get, the reference's
    // own defaults, the reference's own read order.
    let title = track.call_method1("get", ("title", "Unknown"))?;
    let artist = track.call_method1("get", ("artist", "Unknown"))?;
    let album = track.call_method1("get", ("album", ""))?;
    // fmt.format(title=..., artist=..., album=...) — CPython's own
    // format, every spec and conversion the interpreter's own.
    let kwargs = PyDict::new(py);
    kwargs.set_item("title", &title)?;
    kwargs.set_item("artist", &artist)?;
    kwargs.set_item("album", &album)?;
    match fmt.call_method("format", (), Some(&kwargs)) {
        Ok(message) => Ok(message),
        // except (KeyError, ValueError) — the fallback spelling, the
        // f-string's own str() conversions on the very title/artist
        // objects the reads produced.
        Err(err)
            if err.is_instance_of::<PyKeyError>(py) || err.is_instance_of::<PyValueError>(py) =>
        {
            let fallback = format!("{} — {}", py_str(&title)?, py_str(&artist)?);
            Ok(fallback.into_pyobject(py)?.into_any())
        }
        Err(err) => Err(err),
    }
}

/// The prefetch gate — `app/_playback.py`'s `_prefetch_next_track`
/// decision core: which video id the next track's stream URL
/// prefetch runs against, if any.
///
/// Mirrors the reference exactly: a falsy *peeked* queue head answers
/// `None` (nothing to prefetch — the reference's `if next_track:`
/// gate), and the head's own `get` ("video_id", "") is read only for
/// a truthy head, exactly when the reference reads it — a queue whose
/// `peek_next` answers an object without `get` raises its own
/// `AttributeError` here. A falsy id answers `None` (the reference's
/// `if next_id:` gate); a truthy one answers the very object the read
/// produced, the id the resolver's own `prefetch` receives. The
/// `peek_next` call, the resolver guard and the worker scheduling stay
/// Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (peeked))]
pub(crate) fn playback_prefetch_next_id<'py>(
    py: Python<'py>,
    peeked: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // if next_track: — the queue head's truthiness.
    if !peeked.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    // next_track.get("video_id", "") — the head's own get.
    let next_id = peeked.call_method1("get", ("video_id", ""))?;
    // if next_id: — the id's truthiness; the object itself travels on.
    if !next_id.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    Ok(next_id)
}

/// The queue-refill plan — `app/_playback.py`'s `_refill_queue`
/// decision core: whether autoplay tops the queue back up in the
/// background, and the seed tracks that radio fetch runs against.
///
/// Mirrors the reference exactly, every comparison through CPython's
/// own protocol, every arm in the reference's own order. The four
/// gates first: a *repeat_mode* other than `"off"` answers `None`
/// (repeat never drains the queue), a falsy *autoplay_enabled* setting
/// answers `None`, *remaining_tracks* over 3 answers `None` (the
/// comparison CPython's own `>`), and a truthy *extend_running* flag —
/// the caller's own walk of its live `queue_extend` workers — answers
/// `None` so two refills never race. Then the seeds, read off the
/// *queue* object exactly when the reference reads them: `tracks` and
/// `real_index`, the played prefix `tracks[: real_index + 1]` (the
/// bump CPython's own `+`, the slice the interpreter's own
/// `__getitem__`, the `list()` conversion the builtins' own) or a
/// fresh empty list for a negative index, and the last five of that
/// prefix — `played[-5:]`. Empty seeds fall back to the *player*
/// object's own `current_track` (read exactly here, a falsy player
/// answering none), and still-empty seeds answer `None` — no refill.
/// Any other answer is the very seeds list the radio fetch receives.
/// The worker walk, the worker scheduling and the radio fetch stay
/// Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (repeat_mode, autoplay_enabled, remaining_tracks, extend_running, queue, player))]
pub(crate) fn playback_refill_plan<'py>(
    py: Python<'py>,
    repeat_mode: &Bound<'py, PyAny>,
    autoplay_enabled: &Bound<'py, PyAny>,
    remaining_tracks: &Bound<'py, PyAny>,
    extend_running: &Bound<'py, PyAny>,
    queue: &Bound<'py, PyAny>,
    player: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // repeat_mode != "off" — CPython's own !=.
    if repeat_mode.ne("off")? {
        return Ok(py.None().into_bound(py));
    }
    // not autoplay — the setting's truthiness.
    if !autoplay_enabled.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    // remaining_tracks > 3 — CPython's own >.
    if remaining_tracks.gt(3)? {
        return Ok(py.None().into_bound(py));
    }
    // A live queue_extend worker — the caller's own walk, collapsed to
    // one flag.
    if extend_running.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    // all_tracks / current_idx — the queue's own attributes, read
    // exactly here, only once every gate has passed.
    let all_tracks = queue.getattr("tracks")?;
    let current_idx = queue.getattr("real_index")?;
    let played: Bound<'py, PyAny> = if current_idx.ge(0)? {
        // tracks[: current_idx + 1] — the bump CPython's own +, the
        // slice the interpreter's own __getitem__, the materialisation
        // the builtins' own list.
        let stop = current_idx.add(1)?;
        let slice = py
            .import("builtins")?
            .getattr("slice")?
            .call1((py.None(), &stop))?;
        let sliced = all_tracks.get_item(&slice)?;
        py.import("builtins")?.getattr("list")?.call1((&sliced,))?
    } else {
        PyList::new(py, Vec::<Bound<'py, PyAny>>::new())?.into_any()
    };
    // played[-5:] — the last five of the played prefix.
    let tail = py
        .import("builtins")?
        .getattr("slice")?
        .call1((-5i64, py.None()))?;
    let seeds = played.get_item(&tail)?;
    if seeds.is_truthy()? {
        return Ok(seeds);
    }
    // Empty seeds: the player's own current track, read exactly here —
    // the reference's `if self.player else None` guard first.
    let track = if player.is_truthy()? {
        player.getattr("current_track")?
    } else {
        py.None().into_bound(py)
    };
    if !track.is_truthy()? {
        return Ok(py.None().into_bound(py));
    }
    Ok(PyList::new(py, vec![track])?.into_any())
}

/// The local-history finalize plan — `app/_playback.py`'s
/// `_log_local_listen` decision core: how one play's listen duration
/// is written to the local SQLite history when its track ends or is
/// superseded.
///
/// Mirrors the reference's walk exactly, every read through the
/// *claim* object's own attributes, every comparison through CPython's
/// own protocol, in the reference's own order. The ownership gate
/// first: a `None` *claim*, or one whose own `video_id` does not equal
/// *video_id* (CPython's own `==`), is not owned — another play owns
/// the pending row, and the plan's `mark_finalized` is false. An owned
/// claim that is already `finalized` answers `consumed` — a duplicate
/// end-file or a concurrent supersede already wrote the row, and a
/// second write would duplicate it; no listen is even measured. Every
/// other owned claim is marked consumed by the caller (`mark_finalized`
/// true — the reference's own `owned.finalized = True` before anything
/// else, so even a nothing-to-write finish blocks a later finalize).
/// Then the listen, measured exactly like the reference: the *player*
/// object's own `position`, read exactly here, minus *track_start_position*
/// through CPython's own `-`, coerced through the builtins' own `int()`
/// — truncation, `ValueError` on a `nan`, `OverflowError` on an
/// infinite position, exactly the interpreter's own shapes. A listen
/// at or under zero answers `no_write` (mpv reads position 0.0 once
/// idle, so a natural EOF lands here). A positive listen on an owned
/// claim with a `row_id` answers `update_row` carrying that very row
/// id — the insert worker already landed the row; an owned claim whose
/// insert has started answers `stash_pending` — the worker is in
/// flight and the caller parks the duration on the claim for it to
/// apply; everything else answers `direct_log` — the caller's own
/// `log_play` writes the row now. The claim writes, the history calls
/// and the awaits stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (claim, video_id, player, track_start_position))]
pub(crate) fn playback_local_listen_plan<'py>(
    py: Python<'py>,
    claim: &Bound<'py, PyAny>,
    video_id: &Bound<'py, PyAny>,
    player: &Bound<'py, PyAny>,
    track_start_position: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // claim is not None and claim.video_id == video_id — the identity
    // check short-circuited before the claim's own video_id read, the
    // comparison CPython's own ==.
    let owned = !claim.is_none() && claim.getattr("video_id")?.eq(video_id)?;
    if owned {
        // owned.finalized — the claim's own flag.
        let finalized = claim.getattr("finalized")?;
        if finalized.is_truthy()? {
            out.set_item("action", "consumed")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("mark_finalized", false)
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
    }
    // listened = int(player.position - track_start_position) — the
    // player's own position, read exactly here, CPython's own -, the
    // builtins' own int().
    let position = player.getattr("position")?;
    let listened = builtin_int(py, &position.sub(track_start_position)?)?;
    out.set_item("listened", &listened)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("mark_finalized", owned)
        .expect("dict insertion with a str key cannot fail");
    // listened <= 0 — CPython's own <=, the nothing-to-write gate.
    if listened.le(0)? {
        out.set_item("action", "no_write")
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    if owned {
        // owned.row_id is not None — the identity check on the claim's
        // own row id.
        let row_id = claim.getattr("row_id")?;
        if !row_id.is_none() {
            out.set_item("action", "update_row")
                .expect("dict insertion with a str key cannot fail");
            out.set_item("row_id", &row_id)
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
        // owned.insert_started — the claim's own flag.
        let insert_started = claim.getattr("insert_started")?;
        if insert_started.is_truthy()? {
            out.set_item("action", "stash_pending")
                .expect("dict insertion with a str key cannot fail");
            return Ok(out);
        }
    }
    out.set_item("action", "direct_log")
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The insert-finalize plan — `app/_playback.py`'s
/// `_insert_local_history_play` decision core: what the local-history
/// insert worker does once its row lands, and whether the play is
/// still current enough to reflect optimistically.
///
/// Mirrors the reference's walk exactly, every read through the
/// *claim* object's own attributes, every comparison through CPython's
/// own protocol. The claim's own `pending_seconds` is read exactly
/// here: a pending duration means finalize stashed it while this
/// worker was in flight — the plan answers `had_pending` carrying the
/// very value, with `apply_pending` true only when it beats *listened*
/// (CPython's own `>` — a shorter stash keeps the row's own value),
/// and `optimistic_add` always false (the reference returns without
/// the generation check). Without a pending duration the plan answers
/// the generation check instead — the claim's own `generation` equals
/// *play_generation* (CPython's own `==`) — the caller's own
/// optimistic add to the Recently Played page, which runs only while
/// this play is still the current one (generation, not current_track:
/// a duplicate end-file can transiently clear the latter). The claim
/// writes, the history update, the optimistic add and the awaits stay
/// Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (claim, listened, play_generation))]
pub(crate) fn playback_insert_finalize_plan<'py>(
    py: Python<'py>,
    claim: &Bound<'py, PyAny>,
    listened: &Bound<'py, PyAny>,
    play_generation: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    // pending = claim.pending_seconds — the claim's own stash, read
    // exactly here.
    let pending = claim.getattr("pending_seconds")?;
    if !pending.is_none() {
        out.set_item("had_pending", true)
            .expect("dict insertion with a str key cannot fail");
        out.set_item("pending", &pending)
            .expect("dict insertion with a str key cannot fail");
        // pending > listened — CPython's own >.
        out.set_item("apply_pending", pending.gt(listened)?)
            .expect("dict insertion with a str key cannot fail");
        out.set_item("optimistic_add", false)
            .expect("dict insertion with a str key cannot fail");
        return Ok(out);
    }
    // claim.generation == play_generation — CPython's own ==, the
    // still-current gate behind the optimistic add.
    let generation = claim.getattr("generation")?;
    out.set_item("had_pending", false)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("pending", py.None())
        .expect("dict insertion with a str key cannot fail");
    out.set_item("apply_pending", false)
        .expect("dict insertion with a str key cannot fail");
    out.set_item("optimistic_add", generation.eq(play_generation)?)
        .expect("dict insertion with a str key cannot fail");
    Ok(out)
}

/// The min-listen clamp — `app/_playback.py`'s
/// `_history_min_listen_seconds` decision core: the configured
/// minimum seconds a listen must run before it counts as a play.
///
/// Mirrors the reference exactly: the raw configured *value* is
/// coerced through the builtins' own `int()` — float truncation,
/// decimal strings, `__int__`/`__index__` dispatch and the exact
/// `ValueError`/`TypeError` messages where the reference would raise —
/// then clamped through CPython's own `max(0, ...)` comparison
/// (`coerced > 0`), so a negative or zero configuration can never make
/// the gate negative. The settings read stays Python-side in the
/// mixin.
#[pyfunction]
#[pyo3(signature = (value))]
pub(crate) fn playback_history_min_listen_seconds<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // int(value) — the builtins' own coercion.
    let coerced = builtin_int(py, value)?;
    // max(0, coerced) — CPython's own max comparison: coerced > 0.
    if coerced.gt(0)? {
        return Ok(coerced);
    }
    Ok(0i64.into_pyobject(py)?.into_any())
}

/// The history-report timer arm delay — `app/_playback.py`'s
/// `_history_timer_delay` decision core: how long both report timer
/// chains (`_report_local_play` and `_report_ytm_play`) wait before
/// their first poll.
///
/// Mirrors the reference exactly: the already-clamped
/// *min_listen_seconds* is coerced through the builtins' own
/// `float()`, then measured against this module's own
/// `_YTM_HISTORY_POLL_SECONDS` interval — read through the playback
/// module's own name at call time, so a patched interval steers both
/// backends identically — through CPython's own `max` comparison
/// (`poll > fl`). The answer is never zero: a threshold of 0 (count
/// any playback) is valid gating, yet `set_timer(0)` trips a
/// `ZeroDivisionError` inside Textual's timer, so the poll interval is
/// the floor. The `set_timer` scheduling stays Python-side in the
/// mixin.
#[pyfunction]
#[pyo3(signature = (min_listen_seconds))]
pub(crate) fn playback_history_timer_delay<'py>(
    py: Python<'py>,
    min_listen_seconds: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // float(min_listen_seconds) — the builtins' own coercion.
    let fl = py
        .import("builtins")?
        .getattr("float")?
        .call1((min_listen_seconds,))?;
    // _YTM_HISTORY_POLL_SECONDS — this module's own name, read at call
    // time so a patched interval steers both backends identically.
    let poll = playback_ytm_history_poll_seconds(py)?;
    // max(fl, poll) — CPython's own max comparison: poll > fl.
    if poll.gt(&fl)? {
        return Ok(poll);
    }
    Ok(fl)
}
