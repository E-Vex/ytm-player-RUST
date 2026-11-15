//! Native cores for the platform media integrations — `services/mpris.py`,
//! `services/macos_media.py`, `services/macos_eventtap.py` and
//! `app/_mpris.py`: volume clamps/echo gates, event-tap classification,
//! Now Playing state and unit conversions.

use pyo3::exceptions::{PyOverflowError, PyValueError};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// The platform media-integration cores (services/mpris.py,
// services/macos_eventtap.py, services/macos_media.py, app/_mpris.py)
// ---------------------------------------------------------------------------

/// The volume clamp behind every MPRIS volume write — the pure core of
/// `_PlayerInterface`'s `Volume` setter and `set_volume` helper and of
/// `MPRISService.update_volume`'s push, from `services/mpris.py`:
/// `max(0.0, min(1.0, volume))`, the D-Bus `Volume` property's `[0, 1]`
/// domain.
///
/// NaN carries CPython's own argument-order semantics — `min(1.0, nan)`
/// answers `1.0` (the running first argument survives because the
/// comparison is false), which is also Rust `f64::min`'s NaN rule (the
/// other operand wins), and `max(0.0, x)` likewise — so a NaN volume
/// reads `1.0` exactly as the reference's clamp does. The D-Bus
/// property machinery, the callback scheduling and the
/// PropertiesChanged emission stay Python-side in the service.
#[pyfunction]
#[pyo3(signature = (volume))]
pub(crate) fn mpris_clamp_volume(volume: f64) -> f64 {
    0.0f64.max(1.0f64.min(volume))
}

/// The D-Bus-echo gate behind `MPRISService.update_volume` from
/// `services/mpris.py`: a pushed volume whose distance from the value
/// the `Volume` setter already stored (`abs(volume - current) < 1e-6`)
/// is the echo of a D-Bus-initiated set — re-emitting it would
/// ping-pong with the client, so the push is skipped. The comparison
/// is CPython's own strict `<`, so an exactly-`1e-6` distance is not
/// an echo, and a NaN distance (impossible through the clamped setter)
/// compares false. The clamp feeding both operands, the stored-value
/// read and the PropertiesChanged emission stay Python-side in the
/// service.
#[pyfunction]
#[pyo3(signature = (volume, current))]
pub(crate) fn mpris_volume_is_echo(volume: f64, current: f64) -> bool {
    (volume - current).abs() < 1e-6
}

/// The media-key classification behind `_event_action` from
/// `services/macos_eventtap.py` — the pure core of the macOS event
/// tap: the NSEvent `data1` word's bit unpack (the key code in the
/// high 16 bits, the flags in the low 16 with the key state in bits
/// 8-15), the media-key-down state gate (`0xA`), and the key-code →
/// action map (16 play/pause, 17 next, 18 previous, 19 fast-forward →
/// next, 20 rewind → previous).
///
/// The masking arithmetic runs on `i64` two's-complement bits, which
/// agree with CPython's arbitrary-precision `&`/`>>` for every value
/// an `i64` can hold (a negative `data1` masks to the same positive
/// key code both sides); the None gate, the `subtype()` comparison
/// against the raw framework value (not int()-coerced — the reference
/// compares whatever `subtype()` returns), the `int()` coercion of
/// `data1()` and the broad `except` around the attribute extraction
/// stay Python-side in the tap.
#[pyfunction]
#[pyo3(signature = (data1))]
pub(crate) fn macos_event_tap_action(data1: i64) -> Option<&'static str> {
    let key_code = (data1 & 0xFFFF0000) >> 16;
    let key_flags = data1 & 0x0000FFFF;
    let key_state = (key_flags & 0xFF00) >> 8;
    if key_state != 0xA {
        return None;
    }
    match key_code {
        16 => Some("play_pause"),
        17 => Some("next"),
        18 => Some("previous"),
        19 => Some("next"),
        20 => Some("previous"),
        _ => None,
    }
}

/// The playback-state choice behind `MacOSMediaService._playback_state`
/// from `services/macos_media.py`: `"playing"` / `"paused"` /
/// `"stopped"` for the exact status match, and the no-status publish
/// resolving through the service's own playing flag —
/// `_PLAYBACK_STATE_PLAYING if self._is_playing else
/// _PLAYBACK_STATE_PAUSED`.
///
/// The match is exact, not case-insensitive — a `"Playing"` that
/// skipped the caller's `lower()` falls to the playing-flag branch
/// exactly as the reference's `==` does. The answer is the state name,
/// not the framework constant: the caller maps it through the
/// module's constant table, whose entries are `None` without the
/// MediaPlayer bindings — the `setPlaybackState_` write then skips
/// exactly as the reference's `None` returns did. The `setNowPlayingInfo_`
/// call and its broad except stay Python-side in the service.
#[pyfunction]
#[pyo3(signature = (status, is_playing))]
pub(crate) fn macos_playback_state(status: Option<&str>, is_playing: bool) -> &'static str {
    match status {
        Some("playing") => "playing",
        Some("paused") => "paused",
        Some("stopped") => "stopped",
        _ => {
            if is_playing {
                "playing"
            } else {
                "paused"
            }
        }
    }
}

/// The microseconds→seconds conversion behind the macOS Now Playing
/// writes from `services/macos_media.py` — `max(0.0, us / 1_000_000)`
/// — for both `update_metadata`'s `playbackDuration` and
/// `update_position`'s `elapsedPlaybackTime`.
///
/// The division is CPython's own int/int true division for every
/// microsecond count an `i64` can hold below the f64 mantissa limit
/// (2^53 µs ≈ 285 years; beyond it the i64→f64 conversion rounds
/// before the divide where CPython divides the exact integer —
/// unreachable for a track duration or position), and the floor at
/// zero turns a negative count into `0.0` exactly as the reference's
/// `max` does. The `MPMediaItemProperty*` key writes and the publish
/// stay Python-side in the service.
#[pyfunction]
#[pyo3(signature = (microseconds))]
pub(crate) fn macos_now_playing_seconds(microseconds: i64) -> f64 {
    0.0f64.max(microseconds as f64 / 1_000_000.0)
}

/// The microseconds→seconds conversion behind the MPRIS callback unit
/// conversions from `app/_mpris.py` — `us / 1_000_000`, CPython's own
/// int/int true division — for both `_mpris_seek`'s D-Bus `Seek`
/// offset and `_mpris_set_position`'s `SetPosition` position. Neither
/// arm clamps: a D-Bus client's backward seek arrives as a negative
/// offset and must reach the player as negative seconds, exactly as
/// the reference passes it.
///
/// The division matches CPython for every count an `i64` holds below
/// the f64 mantissa limit (2^53 µs ≈ 285 years); beyond it the
/// i64→f64 conversion rounds before the divide where CPython divides
/// the exact integer. The player guards and the seek calls stay
/// Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (microseconds))]
pub(crate) fn mpris_us_to_seconds(microseconds: i64) -> f64 {
    microseconds as f64 / 1_000_000.0
}

/// The volume-fraction→percent conversion behind `_mpris_set_volume`
/// from `app/_mpris.py` — `round(volume * 100)`, CPython's own
/// single-argument `round`: the nearest integer with ties going to
/// the even neighbour (`round(12.5) == 12`, `round(37.5) == 38`), not
/// half-away-from-zero — `f64::round_ties_even` is the same
/// ties-to-even on the same exact float, so a `0.125` volume (whose
/// scaled 12.5 is exactly representable) answers 12 on both sides.
///
/// CPython's two error arms are reproduced with their own exception
/// types and messages: a NaN answers
/// `ValueError("cannot convert float NaN to integer")` and an
/// infinity `OverflowError("cannot convert float infinity to
/// integer")` — unreachable through the `Volume` setter's clamped
/// `[0, 1]` domain, which lands every real write in `[0, 100]`. A
/// finite value whose scaled percent rounds beyond the `i64` range —
/// equally unreachable — saturates where CPython would produce an
/// arbitrary-precision int. The player guard and the `set_volume`
/// call stay Python-side in the mixin.
#[pyfunction]
#[pyo3(signature = (volume))]
pub(crate) fn mpris_volume_percent(volume: f64) -> PyResult<i64> {
    let scaled = volume * 100.0;
    if scaled.is_nan() {
        return Err(PyValueError::new_err("cannot convert float NaN to integer"));
    }
    if scaled.is_infinite() {
        return Err(PyOverflowError::new_err(
            "cannot convert float infinity to integer",
        ));
    }
    let rounded = scaled.round_ties_even();
    if rounded >= i64::MAX as f64 {
        return Ok(i64::MAX);
    }
    if rounded <= i64::MIN as f64 {
        return Ok(i64::MIN);
    }
    Ok(rounded as i64)
}
