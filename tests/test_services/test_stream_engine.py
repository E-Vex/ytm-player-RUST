"""Tests for the stream resolver-glue engines.

The eight pure cores of ``services/stream.py`` run with two
interchangeable backends: the pure-Python transcriptions in the module
and, when the Rust extension is built, the native implementations
bound at the module's tail. ``_is_stale`` is the buffered expiry gate
behind every cache read path; ``_wait_seconds`` is the pre-roll-ad
wait behind ``StreamInfo.wait_seconds``; ``_validate_quality`` is the
gate the resolver's ``quality`` setter raises through;
``_base_ydl_opts`` is the ten-key yt-dlp options dict behind
``_build_ydl_opts`` (before the configured extras);
``_stream_info_fields`` is the extract-info → ``StreamInfo`` field
assembly behind ``_try_resolve``; ``_download_error_hint`` is the
stale-cookiejar diagnostic the DownloadError handler appends;
``_cache_expired_ids`` is the expired-entry walk behind both
``_put_cache``'s and ``prune_expired``'s prune phase;
``_cache_cap_evictions`` is the over-cap eviction walk behind
``_put_cache``'s second phase (the cap read through the module's own
``_CACHE_MAX_SIZE`` name). The three clock-reading engines resolve
``time.time()`` through the module's own ``time`` name at call time —
a frozen or patched clock steers both backends identically, which the
tests below use for determinism and the parity class pins as the
contract.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked and the same frozen clock installed) case for case — error
shapes included — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ytm_player.services.stream import (
    StreamInfo,
    StreamResolver,
    _base_ydl_opts,
    _cache_cap_evictions,
    _cache_expired_ids,
    _download_error_hint,
    _is_stale,
    _stream_info_fields,
    _validate_quality,
    _wait_seconds,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_FROZEN_NOW = 1000.0

_HINT = " (stream cookiejar may be stale/revoked — try `ytm setup` to refresh it)"


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze stream.py's clock at 1000 — the engine's own module-clock
    read makes both backends see the frozen value."""
    from ytm_player.services import stream

    monkeypatch.setattr(stream, "time", SimpleNamespace(time=lambda: _FROZEN_NOW))
    return _FROZEN_NOW


class TestIsStale:
    """The buffered expiry gate."""

    def test_expired_is_stale(self, frozen_clock):
        assert _is_stale(700.0) is True
        assert _is_stale(100.0) is True

    def test_far_future_is_fresh(self, frozen_clock):
        assert _is_stale(5000.0) is False
        assert _is_stale(10**9) is False

    def test_the_buffer_edge_is_stale(self, frozen_clock):
        # 1000 >= 1300 - 300 == 1000 — the boundary is stale (>=).
        assert _is_stale(1300.0) is True
        assert _is_stale(1300) is True  # int arithmetic lands the same

    def test_just_inside_the_buffer_is_fresh(self, frozen_clock):
        assert _is_stale(1300.5) is False
        assert _is_stale(1301.0) is False

    def test_non_numeric_raises_the_subtraction_error(self, frozen_clock):
        with pytest.raises(TypeError, match="unsupported operand"):
            _is_stale("soon")


class TestWaitSeconds:
    """The available_at wait gate."""

    def test_future_waits_the_distance(self, frozen_clock):
        assert _wait_seconds(1030.0) == 30
        assert _wait_seconds(1030) == 30.0
        assert _wait_seconds(1e18) == 1e18 - _FROZEN_NOW

    def test_past_and_present_wait_zero(self, frozen_clock):
        assert _wait_seconds(995.0) == 0.0
        assert _wait_seconds(1000.0) == 0.0
        assert _wait_seconds(-50.0) == 0.0

    def test_nan_clamps_to_zero(self, frozen_clock):
        assert _wait_seconds(float("nan")) == 0.0

    def test_stream_info_wait_seconds_delegates(self, frozen_clock):
        assert (
            StreamInfo(
                url="https://x", video_id="v", format="opus", bitrate=1, duration=1, expires_at=1.0
            ).wait_seconds()
            == 0.0
        )
        assert (
            StreamInfo(
                url="https://x",
                video_id="v",
                format="opus",
                bitrate=1,
                duration=1,
                expires_at=1.0,
                available_at=1002.5,
            ).wait_seconds()
            == 2.5
        )


class TestValidateQuality:
    """The quality-input gate."""

    @pytest.mark.parametrize("tier", ["high", "medium", "low"])
    def test_tiers_pass(self, tier):
        assert _validate_quality(tier) is None

    def test_unknown_quality_raises_the_reference_message(self):
        with pytest.raises(ValueError) as excinfo:
            _validate_quality("bogus")
        assert str(excinfo.value) == (
            "Unknown quality 'bogus'. Choose from: ['high', 'medium', 'low']"
        )

    def test_membership_is_case_sensitive(self):
        with pytest.raises(ValueError, match="Unknown quality"):
            _validate_quality("HIGH")

    def test_the_setter_raises_through_the_engine(self):
        resolver = StreamResolver()
        with pytest.raises(ValueError, match="Unknown quality 'bogus'"):
            resolver.quality = "bogus"
        assert resolver.quality == "high"  # unchanged


class TestBaseOpts:
    """The base yt-dlp options dict."""

    def test_keys_in_the_reference_order(self):
        opts = _base_ydl_opts("high", "LOGGER")
        assert list(opts) == [
            "format",
            "quiet",
            "no_warnings",
            "logger",
            "extract_flat",
            "noplaylist",
            "skip_download",
            "writeinfojson",
            "writethumbnail",
            "extractor_args",
        ]

    @pytest.mark.parametrize(
        ("quality", "selector"),
        [
            ("high", "bestaudio/best[vcodec!=none][acodec!=none]"),
            ("medium", "bestaudio[abr<=128]/bestaudio/best[vcodec!=none][acodec!=none]"),
            ("low", "bestaudio[abr<=64]/bestaudio/best[vcodec!=none][acodec!=none]"),
        ],
    )
    def test_quality_selects_the_format(self, quality, selector):
        opts = _base_ydl_opts(quality, "LOGGER")
        assert opts["format"] == selector

    def test_unknown_quality_falls_back_to_high(self):
        assert (
            _base_ydl_opts("ultra", "LOGGER")["format"]
            == _base_ydl_opts("high", "LOGGER")["format"]
        )

    def test_the_full_dict_matches_the_reference(self):
        logger = object()
        opts = _base_ydl_opts("high", logger)
        assert opts == {
            "format": "bestaudio/best[vcodec!=none][acodec!=none]",
            "quiet": True,
            "no_warnings": True,
            "logger": logger,
            "extract_flat": False,
            "noplaylist": True,
            "skip_download": True,
            "writeinfojson": False,
            "writethumbnail": False,
            "extractor_args": {"youtube": {"player_client": ["default", "android"]}},
        }

    def test_the_logger_object_passes_through(self):
        logger = _logger = object()
        assert _base_ydl_opts("high", logger)["logger"] is logger

    def test_every_call_builds_fresh_containers(self):
        a = _base_ydl_opts("high", "L")
        b = _base_ydl_opts("high", "L")
        assert a is not b
        assert a == b
        assert a["extractor_args"] is not b["extractor_args"]
        assert a["extractor_args"]["youtube"] is not b["extractor_args"]["youtube"]
        assert (
            a["extractor_args"]["youtube"]["player_client"]
            is not (b["extractor_args"]["youtube"]["player_client"])
        )


class TestStreamInfoFields:
    """The extract-info → StreamInfo field assembly."""

    def test_the_full_extraction(self, frozen_clock):
        info = {
            "url": "https://googlevideo.example/a",
            "available_at": 12.5,
            "acodec": "opus",
            "audio_ext": "webm",
            "abr": 128,
            "duration": 200,
            "thumbnail": "https://i.example/t.jpg",
        }
        fields = _stream_info_fields(info, "vid1")
        assert fields == (
            "https://googlevideo.example/a",
            "vid1",
            "opus",
            128,
            200,
            _FROZEN_NOW + 18000,
            "https://i.example/t.jpg",
            12.5,
        )

    def test_fields_construct_the_dataclass(self, frozen_clock):
        info = {"url": "https://a", "acodec": "opus", "abr": 96, "duration": 60}
        info_obj = StreamInfo(*_stream_info_fields(info, "vid2"))
        assert info_obj.url == "https://a"
        assert info_obj.video_id == "vid2"
        assert info_obj.format == "opus"
        assert info_obj.bitrate == 96
        assert info_obj.duration == 60
        assert info_obj.expires_at == _FROZEN_NOW + 18000
        assert info_obj.thumbnail_url is None
        assert info_obj.available_at == 0.0

    def test_video_id_passes_through_as_the_object(self):
        video_id = "vid3"
        fields = _stream_info_fields({"url": "u"}, video_id)
        assert fields[1] is video_id

    def test_requested_formats_fallback(self, frozen_clock):
        info = {
            "url": "",
            "available_at": 999,
            "requested_formats": [
                {"vcodec": "h264", "acodec": "none", "url": "https://wrong"},
                {"vcodec": "none", "url": "https://nested", "available_at": 42},
            ],
            "acodec": "none",
            "ext": "m4a",
            "tbr": 64.5,
        }
        fields = _stream_info_fields(info, "v")
        # The first format the reference's condition accepts supplies the
        # URL and the available_at — the h264+audio-none row is skipped
        # (vcodec != "none" and acodec == "none"), the vcodec-none row wins.
        assert fields[0] == "https://nested"
        assert fields[7] == 42.0
        assert fields[2] == "m4a"  # acodec "none" falls back to the extension
        assert fields[3] == 64  # int(64.5) truncates toward zero

    def test_no_stream_url_anywhere_answers_none(self):
        assert _stream_info_fields({}, "v") is None
        assert _stream_info_fields({"url": ""}, "v") is None
        assert _stream_info_fields({"url": "", "requested_formats": []}, "v") is None
        assert _stream_info_fields({"url": "", "requested_formats": [{"url": ""}]}, "v") is None

    def test_missing_acodec_reads_unknown(self):
        fields = _stream_info_fields({"url": "u"}, "v")
        assert fields[2] == "unknown"

    def test_audio_ext_chain(self):
        # The extension only names the format when the acodec is "none";
        # then audio_ext wins when truthy, otherwise ext, a falsy-but-present
        # ext passing through as the object it is (the reference's or arm).
        assert (
            _stream_info_fields({"url": "u", "acodec": "none", "audio_ext": "webm"}, "v")[2]
            == "webm"
        )
        assert _stream_info_fields({"url": "u", "acodec": "none", "ext": "m4a"}, "v")[2] == "m4a"
        assert _stream_info_fields({"url": "u", "acodec": "none"}, "v")[2] == "unknown"
        assert _stream_info_fields({"url": "u", "acodec": "none", "ext": ""}, "v")[2] == ""
        assert _stream_info_fields({"url": "u"}, "v")[2] == "unknown"

    def test_abr_tbr_chain(self):
        assert _stream_info_fields({"url": "u", "abr": 128, "tbr": 130}, "v")[3] == 128
        assert _stream_info_fields({"url": "u", "tbr": 130}, "v")[3] == 130
        assert _stream_info_fields({"url": "u", "abr": 0, "tbr": 130}, "v")[3] == 130
        assert _stream_info_fields({"url": "u"}, "v")[3] == 0
        assert _stream_info_fields({"url": "u", "abr": "96"}, "v")[3] == 96

    def test_duration_chain(self):
        assert _stream_info_fields({"url": "u", "duration": 200}, "v")[4] == 200
        assert _stream_info_fields({"url": "u", "duration": 0}, "v")[4] == 0
        assert _stream_info_fields({"url": "u"}, "v")[4] == 0
        assert _stream_info_fields({"url": "u", "duration": "200"}, "v")[4] == 200

    def test_a_non_int_duration_keeps_the_reference_error(self):
        with pytest.raises(ValueError, match="invalid literal"):
            _stream_info_fields({"url": "u", "duration": "soon"}, "v")

    def test_available_at_coercions(self):
        assert _stream_info_fields({"url": "u"}, "v")[7] == 0.0
        assert _stream_info_fields({"url": "u", "available_at": None}, "v")[7] == 0.0
        assert _stream_info_fields({"url": "u", "available_at": 0}, "v")[7] == 0.0
        assert _stream_info_fields({"url": "u", "available_at": 3}, "v")[7] == 3.0
        assert _stream_info_fields({"url": "u", "available_at": "1.5"}, "v")[7] == 1.5

    def test_thumbnail_passes_through(self):
        thumb = object()
        assert _stream_info_fields({"url": "u", "thumbnail": thumb}, "v")[6] is thumb
        assert _stream_info_fields({"url": "u"}, "v")[6] is None


class TestDownloadErrorHint:
    """The stale-cookiejar diagnostic."""

    def test_auth_related_errors_with_the_jar_get_the_hint(self):
        assert _download_error_hint("Sign in to confirm you're not a bot", True) == _HINT
        assert _download_error_hint("Please LOGIN_REQUIRED to continue", True) == _HINT
        assert _download_error_hint("confirm you are human", True) == _HINT

    def test_case_insensitive_match(self):
        assert _download_error_hint("PLEASE SIGN IN", True) == _HINT

    def test_anonymous_streaming_never_gets_the_hint(self):
        assert _download_error_hint("Sign in to confirm you're not a bot", False) == ""

    def test_unrelated_errors_get_no_hint(self):
        assert _download_error_hint("video unavailable", True) == ""
        assert _download_error_hint("HTTP Error 403", True) == ""
        assert _download_error_hint("", True) == ""


def _entry(expires_at):
    """A cache value exposing the one attribute the walks read."""
    return SimpleNamespace(expires_at=expires_at)


class TestCacheExpiredIds:
    """The expired-entry walk behind ``_put_cache`` and ``prune_expired``."""

    def test_empty_cache_answers_empty(self, frozen_clock):
        assert _cache_expired_ids({}) == []

    def test_fresh_cache_answers_empty(self, frozen_clock):
        cache = {"a": _entry(1300.0), "b": _entry(5000.0)}
        assert _cache_expired_ids(cache) == []

    def test_expired_answered_in_the_cache_iteration_order(self, frozen_clock):
        cache = {
            "fresh": _entry(1300.0),
            "old": _entry(700.0),
            "edge": _entry(1000.0),  # the boundary is expired (>=)
            "ancient": _entry(-50.0),
            "new": _entry(1000.5),
        }
        assert _cache_expired_ids(cache) == ["old", "edge", "ancient"]

    def test_ids_pass_through_as_the_objects(self, frozen_clock):
        vid = "the-very-id"
        assert _cache_expired_ids({vid: _entry(700.0)})[0] is vid

    def test_int_expires_at_compares_like_a_float(self, frozen_clock):
        assert _cache_expired_ids({"a": _entry(1000)}) == ["a"]
        assert _cache_expired_ids({"a": _entry(1001)}) == []

    def test_walks_a_real_resolver_cache(self, frozen_clock):
        resolver = StreamResolver()
        with resolver._cache_lock:
            resolver._cache["fresh"] = StreamInfo(
                url="https://x",
                video_id="fresh",
                format="opus",
                bitrate=1,
                duration=1,
                expires_at=_FROZEN_NOW + 18000,
            )
            resolver._cache["old"] = StreamInfo(
                url="https://x",
                video_id="old",
                format="opus",
                bitrate=1,
                duration=1,
                expires_at=0.0,
            )
        assert _cache_expired_ids(resolver._cache) == ["old"]

    def test_non_numeric_expires_at_raises_the_comparison_error(self, frozen_clock):
        with pytest.raises(TypeError, match="not supported between instances"):
            _cache_expired_ids({"a": _entry("soon")})


class TestCacheCapEvictions:
    """The over-cap eviction walk behind ``_put_cache``'s second phase."""

    def test_within_cap_answers_empty(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 3)
        cache = {"a": _entry(100.0), "b": _entry(300.0), "c": _entry(200.0)}
        assert _cache_cap_evictions(cache) == []

    def test_over_cap_evicts_the_oldest_by_expires_at(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 2)
        cache = {"a": _entry(100.0), "b": _entry(300.0), "c": _entry(200.0)}
        assert _cache_cap_evictions(cache) == ["a"]

    def test_the_excess_is_exact_and_ordered(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 0)
        cache = {"b": _entry(100.0), "a": _entry(500.0), "a2": _entry(300.0)}
        assert _cache_cap_evictions(cache) == ["b", "a2", "a"]

    def test_ties_keep_the_cache_iteration_order(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 2)
        cache = {
            "z": _entry(900.0),
            "y": _entry(100.0),
            "x": _entry(900.0),
            "w": _entry(100.0),
        }
        # Stable sort: the 100s evict in insertion order (y before w),
        # and only the over-cap excess goes.
        assert _cache_cap_evictions(cache) == ["y", "w"]

    def test_huge_values_rank_after_small_ones(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 2)
        cache = {"a": _entry(10**20), "b": _entry(1.5), "c": _entry(-3)}
        assert _cache_cap_evictions(cache) == ["c"]

    def test_the_cap_is_read_through_the_module_name(self, monkeypatch):
        from ytm_player.services import stream

        cache = {"a": _entry(100.0), "b": _entry(300.0), "c": _entry(200.0)}
        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 3)
        assert _cache_cap_evictions(cache) == []
        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 1)
        assert _cache_cap_evictions(cache) == ["a", "c"]

    def test_a_missing_expires_at_raises_the_attribute_error(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_CACHE_MAX_SIZE", 0)
        with pytest.raises(AttributeError, match="expires_at"):
            _cache_cap_evictions({"a": SimpleNamespace()})


# Each parity case: the engine's arguments as JSON-round-trippable
# lists, keyed by the engine's native name (the Python module-level
# fallback spells the same name with a leading underscore). Both the
# parent and the child run with stream.py's clock frozen at 1000, so
# every clock-derived value is exact on both sides. The cache is
# carried as a {video_id: expires_at} mapping and rebuilt as the
# attribute-bearing objects the walks read; the cap engine's first
# argument is the _CACHE_MAX_SIZE both sides install before the call.
# (expires_at never spells NaN in these cases: the ranking of a NaN
# key is sort-algorithm-dependent, so only totally ordered values are
# parity-pinned.)
_PARITY_CASES = {
    "is_stale": [
        [1300.0],
        [1300.5],
        [1301.0],
        [1300],
        [700.0],
        [100.0],
        [5000.0],
        [1000000000],
        ["soon"],
    ],
    "wait_seconds": [[1030.0], [1030], [995.0], [1000.0], [1000.5], [-50.0], [1e18]],
    "validate_quality": [["high"], ["medium"], ["low"], ["bogus"], [""], [5], [None], [True]],
    "stream_base_opts": [
        ["high", "LOGGER"],
        ["medium", "LOGGER"],
        ["low", "LOGGER"],
        ["ultra", "L"],
    ],
    "stream_info_fields": [
        [
            {
                "url": "https://googlevideo.example/a",
                "available_at": 12.5,
                "acodec": "opus",
                "audio_ext": "webm",
                "abr": 128,
                "duration": 200,
                "thumbnail": "https://i.example/t.jpg",
            },
            "vid1",
        ],
        [
            {
                "url": "",
                "available_at": 999,
                "requested_formats": [
                    {"vcodec": "h264", "acodec": "none", "url": "https://wrong"},
                    {"vcodec": "none", "url": "https://nested", "available_at": 42},
                ],
                "acodec": "none",
                "ext": "m4a",
                "tbr": 64.5,
            },
            "v",
        ],
        [{"url": "", "requested_formats": []}, "v"],
        [{}, "v"],
        [
            {"url": "https://c", "acodec": "none", "audio_ext": 0, "ext": "", "duration": "1:30"},
            "v",
        ],
        [
            {
                "url": "https://d",
                "acodec": "opus",
                "abr": "96",
                "tbr": "130",
                "duration": "200",
                "thumbnail": None,
                "available_at": "1.5",
            },
            "dQw4w9WgXcQ",
        ],
        [
            {
                "url": 0,
                "requested_formats": [{"acodec": "mp4a", "url": "https://e", "available_at": 0}],
                "acodec": "opus",
            },
            "v",
        ],
    ],
    "download_error_hint": [
        ["Sign in to confirm you're not a bot", True],
        ["Please LOGIN_REQUIRED to continue", True],
        ["confirm you are human", True],
        ["Sign in", False],
        ["video unavailable", True],
        ["", True],
        ["", False],
    ],
    "stream_cache_expired_ids": [
        [{}],
        [{"a": 1300.0, "b": 1500.0}],
        [{"old": 700.0, "edge": 1000.0, "new": 1300.0}],
        [{"a": 999.5, "b": 1000, "c": 1000.5}],
        [{"neg": -50.0, "zero": 0}],
        [{"a": "soon"}],
    ],
    "stream_cache_cap_evictions": [
        [128, {}],
        [3, {"a": 100.0, "b": 300.0, "c": 200.0}],
        [2, {"a": 100.0, "b": 300.0, "c": 200.0}],
        [1, {"a": 100.0, "b": 100.0, "c": 100.0}],
        [0, {"b": 100.0, "a": 500.0, "a2": 300.0}],
        [2, {"z": 900.0, "y": 100.0, "x": 900.0, "w": 100.0}],
        [2, {"a": 100000000000000000000, "b": 1.5, "c": -3}],
    ],
}

_CHILD_PROG = """
import json
import sys
from types import SimpleNamespace

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import stream

stream.time = SimpleNamespace(time=lambda: 1000.0)

PY_NAMES = {
    "is_stale": "_is_stale",
    "wait_seconds": "_wait_seconds",
    "validate_quality": "_validate_quality",
    "stream_base_opts": "_base_ydl_opts",
    "stream_info_fields": "_stream_info_fields",
    "download_error_hint": "_download_error_hint",
    "stream_cache_expired_ids": "_cache_expired_ids",
    "stream_cache_cap_evictions": "_cache_cap_evictions",
}


def call_args(name, args):
    if name == "stream_cache_expired_ids":
        return [{vid: SimpleNamespace(expires_at=exp) for vid, exp in args[0].items()}]
    if name == "stream_cache_cap_evictions":
        stream._CACHE_MAX_SIZE = args[0]
        return [{vid: SimpleNamespace(expires_at=exp) for vid, exp in args[1].items()}]
    return args


def outcome(name, args):
    fn = getattr(stream, PY_NAMES[name])
    try:
        result = fn(*call_args(name, args))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if name == "stream_info_fields":
        return {"ok": None if result is None else list(result)}
    if name == "stream_base_opts":
        return {"ok": {"keys": list(result), "dict": {k: v for k, v in result.items() if k != "logger"}}}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {name: [outcome(name, args) for args in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""


# The Python module-level name each native engine replaces (the tail
# rebinding assigns the native function to the fallback's own name).
_PY_NAMES = {
    "is_stale": "_is_stale",
    "wait_seconds": "_wait_seconds",
    "validate_quality": "_validate_quality",
    "stream_base_opts": "_base_ydl_opts",
    "stream_info_fields": "_stream_info_fields",
    "download_error_hint": "_download_error_hint",
    "stream_cache_expired_ids": "_cache_expired_ids",
    "stream_cache_cap_evictions": "_cache_cap_evictions",
}


def _cache_call_args(name, args):
    """The cache rebuilt from the JSON case as the attribute-bearing
    objects the walks read; the cap engine's case also installs its
    cap first (the caller restores it once the engine has run, so the
    module constant never leaks)."""
    from ytm_player.services import stream

    if name == "stream_cache_expired_ids":
        return [{vid: _entry(exp) for vid, exp in args[0].items()}], None
    if name == "stream_cache_cap_evictions":
        saved = stream._CACHE_MAX_SIZE
        stream._CACHE_MAX_SIZE = args[0]
        return [{vid: _entry(exp) for vid, exp in args[1].items()}], saved
    return args, None


def _native_outcomes():
    """The parent-side outcomes, computed against the module's current
    (frozen) clock — the same rebinding the child performs."""

    def outcome(name, args):
        fn = getattr(sys.modules["ytm_player.services.stream"], _PY_NAMES[name])
        call_args, saved_cap = _cache_call_args(name, args)
        try:
            result = fn(*call_args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            if saved_cap is not None:
                from ytm_player.services import stream

                stream._CACHE_MAX_SIZE = saved_cap
        if name == "stream_info_fields":
            return {"ok": None if result is None else list(result)}
        if name == "stream_base_opts":
            return {
                "ok": {
                    "keys": list(result),
                    "dict": {k: v for k, v in result.items() if k != "logger"},
                }
            }
        return {"ok": result}

    return {name: [outcome(name, args) for args in cases] for name, cases in _PARITY_CASES.items()}


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same verdicts, same dicts,
    same tuples, same errors with the same messages — with both sides
    reading the same frozen module clock, so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_resolver_glue_matches_fallback(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "time", SimpleNamespace(time=lambda: _FROZEN_NOW))

        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes()
        assert set(native) == set(fallback) == set(_PARITY_CASES)
        for name, cases in _PARITY_CASES.items():
            for case, n, f in zip(cases, native[name], fallback[name]):
                assert n == f, (
                    f"native and fallback {name} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
