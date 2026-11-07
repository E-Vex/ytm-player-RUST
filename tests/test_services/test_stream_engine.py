"""Tests for the stream resolver-glue engines.

The fourteen pure cores of ``services/stream.py`` run with two
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
``_CACHE_MAX_SIZE`` name); ``_ydl_cookiejar_current`` is the
cookiejar-signature reuse gate behind ``_get_ydl``'s locked reuse arm
(the cached instance's baked signature against the signature the
resolver should be using right now); ``_ydl_publish_verdict`` is the
mid-build publication verdict behind ``_get_ydl``'s second locked
section (adopt another thread's instance, retry after a mid-build
reset, or install the built one); ``_retry_delays`` is the
``(attempt, delay)`` schedule walk behind ``_resolve_sync``'s retry
loop (the schedule read through the module's own ``_RETRY_DELAYS``
name); ``_pending_join_verdict`` is the join gate behind
``_current_pending`` (the in-flight resolve a same-generation caller
joins instead of extracting again); ``_pending_drop_verdict`` is the
identity gate behind ``_resolve``'s finally cleanup (the pending slot
is dropped only when it still holds this owner's very future); and
``_pending_publication_gate`` is the unresolved gate behind the
shared pending future's three publications (``set_result``,
``set_exception`` and the owner-cancelled ``cancel``). The three
clock-reading engines resolve ``time.time()``
through the module's own ``time`` name at call time — a frozen or
patched clock steers both backends identically, which the tests below
use for determinism and the parity class pins as the contract.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked and the same frozen clock installed) case for case — error
shapes included — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import asyncio
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
    _pending_drop_verdict,
    _pending_join_verdict,
    _pending_publication_gate,
    _retry_delays,
    _stream_info_fields,
    _validate_quality,
    _wait_seconds,
    _ydl_cookiejar_current,
    _ydl_publish_verdict,
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


def _json_sig_case(value):
    """The JSON case's signature re-packed as the nested tuples the
    reference builds (a list never equals a tuple, so both sides of the
    parity comparison must spell the same shape)."""
    if isinstance(value, list):
        return tuple(_json_sig_case(item) for item in value)
    return value


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


class TestYdlCookiejarCurrent:
    """The cookiejar-signature reuse gate behind _get_ydl's reuse arm."""

    def test_none_matches_none(self):
        # An instance built anonymous while streaming is still anonymous.
        assert _ydl_cookiejar_current(None, None) is True

    def test_equal_signatures_are_current(self):
        sig = ("/path/jar", (1, 2, 3), None, (5, 6, 7))
        assert _ydl_cookiejar_current(sig, sig) is True
        assert _ydl_cookiejar_current(sig, tuple(sig)) is True

    def test_a_rewritten_jar_is_not_current(self):
        cached = ("/p", (1, 2, 3), (4, 5), (6, 7))
        current = ("/p", (1, 9, 3), (4, 5), (6, 7))
        assert _ydl_cookiejar_current(cached, current) is False

    def test_a_replaced_record_is_not_current(self):
        cached = ("/p", (1, 2, 3), (4, 5), (6, 7))
        assert _ydl_cookiejar_current(cached, ("/p", (1, 2, 3), (4, 5), (6, 8))) is False

    def test_a_changed_jar_path_is_not_current(self):
        cached = ("/old", (1, 2, 3), None, None)
        assert _ydl_cookiejar_current(cached, ("/new", (1, 2, 3), None, None)) is False

    def test_signed_in_now_means_rebuild(self):
        assert _ydl_cookiejar_current(None, ("/p", (1, 2, 3), None, None)) is False

    def test_anonymous_now_means_rebuild(self):
        assert _ydl_cookiejar_current(("/p", (1, 2, 3), None, None), None) is False

    def test_a_list_is_not_the_tuples_it_ressembles(self):
        # CPython == semantics: a list-spelled signature never matches
        # the tuple the reference builds — even with equal contents.
        assert _ydl_cookiejar_current(["/p", [1]], ("/p", (1,))) is False
        assert _ydl_cookiejar_current(("/p", (1,)), ["/p", [1]]) is False

    def test_int_and_float_sigs_match(self):
        assert _ydl_cookiejar_current(("/p", (1, 5), None, None), ("/p", (1, 5.0), None, None)) is (
            True
        )

    def test_the_comparison_is_the_eq_protocol(self):
        class Sig:
            def __init__(self, verdict):
                self.verdict = verdict

            def __eq__(self, other):
                return self.verdict

        # A truthy __eq__ verdict means current — the truthiness of the
        # comparison is the gate, the same one the reference's `if`
        # applies; a falsy one means not.
        assert _ydl_cookiejar_current(Sig("yes"), object())
        assert not _ydl_cookiejar_current(Sig(""), object())


class TestYdlPublishVerdict:
    """The mid-build publication verdict behind _get_ydl's second arm."""

    def test_an_instance_published_meanwhile_is_adopted(self):
        assert _ydl_publish_verdict(object(), 1, 1) == "adopt"
        assert _ydl_publish_verdict("ydl", 1, 99) == "adopt"

    def test_a_mid_build_reset_forces_a_retry(self):
        assert _ydl_publish_verdict(None, 1, 2) == "retry"
        assert _ydl_publish_verdict(None, 0, 1) == "retry"

    def test_the_steady_state_installs(self):
        assert _ydl_publish_verdict(None, 3, 3) == "install"
        assert _ydl_publish_verdict(None, 0, 0) == "install"

    def test_the_instance_check_beats_the_generation_check(self):
        # The identity check runs first: an instance present with a moved
        # generation is still adopted.
        assert _ydl_publish_verdict("ydl", 1, 2) == "adopt"

    def test_generation_equality_is_value_equality(self):
        # current != snapshot through CPython's !=, so equal-valued
        # int/float/bool generations never retry.
        assert _ydl_publish_verdict(None, 1, 1.0) == "install"
        assert _ydl_publish_verdict(None, True, 1) == "install"

    def test_string_generations_compare_by_value(self):
        assert _ydl_publish_verdict(None, "a", "a") == "install"
        assert _ydl_publish_verdict(None, "a", "b") == "retry"

    def test_the_inequality_verdict_is_its_truthiness(self):
        class Weird:
            def __ne__(self, other):
                return "yes"

        class Quiet:
            def __ne__(self, other):
                return ""

        # != through CPython's own protocol: the retry gate is the
        # truthiness of the comparison's result, not the result's type.
        assert _ydl_publish_verdict(None, 1, Weird()) == "retry"
        assert _ydl_publish_verdict(None, 1, Quiet()) == "install"

    def test_a_raising_ne_propagates(self):
        class Boom:
            def __ne__(self, other):
                raise RuntimeError("boom")

        # The current generation is the comparison's subject (the left
        # operand), exactly like the reference's self._ydl_generation.
        with pytest.raises(RuntimeError, match="boom"):
            _ydl_publish_verdict(None, 1, Boom())


class TestRetryDelays:
    """The ``(attempt, delay)`` iteration behind ``_resolve_sync``'s
    retry loop."""

    def test_the_default_schedule_pairs_attempts_with_delays(self):
        assert _retry_delays() == [(0, 0), (1, 1.0), (2, 2.0)]

    def test_the_pairs_carry_the_schedules_own_objects(self):
        # The walk yields the schedule's very objects — enumerate (and
        # the native walk, the same iterator protocol) never copies
        # the delays. Small ints are interned, so the float arms are
        # the meaningful identity pins.
        from ytm_player.services import stream

        steps = _retry_delays()
        assert steps[0][1] is stream._RETRY_DELAYS[0]
        assert steps[1][1] is stream._RETRY_DELAYS[1]
        assert steps[2][1] is stream._RETRY_DELAYS[2]

    def test_a_fresh_list_every_call(self):
        from ytm_player.services import stream

        first = _retry_delays()
        first.append("junk")
        assert _retry_delays() == [(0, 0), (1, 1.0), (2, 2.0)]
        assert stream._RETRY_DELAYS == [0, 1.0, 2.0]  # never mutated

    def test_the_schedule_is_read_through_the_module_name(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_RETRY_DELAYS", [5, 0.5, 0.25, 7])
        assert _retry_delays() == [(0, 5), (1, 0.5), (2, 0.25), (3, 7)]

    def test_an_empty_schedule_answers_no_attempts(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_RETRY_DELAYS", [])
        assert _retry_delays() == []

    def test_the_walk_is_the_iterator_protocol(self, monkeypatch):
        # enumerate runs over any iterable, and both backends walk the
        # schedule through the same protocol — so tuple and string
        # schedules steer identically.
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_RETRY_DELAYS", (9, 0.5))
        assert _retry_delays() == [(0, 9), (1, 0.5)]
        monkeypatch.setattr(stream, "_RETRY_DELAYS", "ab")
        assert _retry_delays() == [(0, "a"), (1, "b")]


class TestResolveSyncRetryWalk:
    """The retry-delay engine inside the resolver's own retry loop, on
    whichever backend is loaded — the attempts made, the waits slept
    between them, the early exit on the first success and the gates
    that never walk the schedule at all."""

    @staticmethod
    def _scripted(monkeypatch, outcomes):
        """A resolver whose ``_try_resolve`` answers the scripted
        *outcomes* (popped left to right, ``None`` once exhausted),
        recording every attempt, every URL and every sleep — the last
        through the module's own ``time`` name swapped for a recording
        sleeper, the steering ``_resolve_sync``'s ``time.sleep`` reads
        through."""
        from ytm_player.services import stream

        resolver = StreamResolver()
        seen: dict[str, list] = {"attempts": [], "urls": [], "sleeps": []}
        monkeypatch.setattr(stream, "time", SimpleNamespace(sleep=seen["sleeps"].append))
        scripted = list(outcomes)

        def fake_try_resolve(url, video_id, attempt):
            seen["attempts"].append(attempt)
            seen["urls"].append(url)
            return scripted.pop(0) if scripted else None

        resolver._try_resolve = fake_try_resolve
        return resolver, seen

    def test_the_full_budget_walks_every_attempt(self, monkeypatch):
        resolver, seen = self._scripted(monkeypatch, [None, None, None])
        assert resolver._resolve_sync("abc12345678") is None
        assert seen["attempts"] == [0, 1, 2]
        assert seen["sleeps"] == [1.0, 2.0]

    def test_every_attempt_hits_the_watch_url(self, monkeypatch):
        resolver, seen = self._scripted(monkeypatch, [None, None, None])
        resolver._resolve_sync("abc12345678")
        assert seen["urls"] == ["https://music.youtube.com/watch?v=abc12345678"] * 3

    def test_a_success_stops_the_walk(self, monkeypatch):
        sentinel = StreamInfo(
            url="https://x",
            video_id="abc12345678",
            format="opus",
            bitrate=1,
            duration=1,
            expires_at=1.0,
        )
        resolver, seen = self._scripted(monkeypatch, [None, sentinel])
        assert resolver._resolve_sync("abc12345678") is sentinel
        assert seen["attempts"] == [0, 1]
        assert seen["sleeps"] == [1.0]  # the third attempt's 2.0 wait never slept

    def test_a_first_try_success_never_sleeps(self, monkeypatch):
        sentinel = StreamInfo(
            url="https://x",
            video_id="abc12345678",
            format="opus",
            bitrate=1,
            duration=1,
            expires_at=1.0,
        )
        resolver, seen = self._scripted(monkeypatch, [sentinel])
        assert resolver._resolve_sync("abc12345678") is sentinel
        assert seen["attempts"] == [0]
        assert seen["sleeps"] == []

    def test_a_patched_schedule_steers_the_resolver(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_RETRY_DELAYS", [0, 0.5])
        resolver, seen = self._scripted(monkeypatch, [None, None])
        assert resolver._resolve_sync("abc12345678") is None
        assert seen["attempts"] == [0, 1]
        assert seen["sleeps"] == [0.5]

    def test_an_empty_schedule_makes_no_attempts(self, monkeypatch):
        from ytm_player.services import stream

        monkeypatch.setattr(stream, "_RETRY_DELAYS", [])
        resolver, seen = self._scripted(monkeypatch, [object()])
        assert resolver._resolve_sync("abc12345678") is None
        assert seen["attempts"] == []
        assert seen["sleeps"] == []

    def test_an_invalid_video_id_never_walks_the_schedule(self, monkeypatch):
        # Rejected before the loop: no attempt, no sleep.
        resolver, seen = self._scripted(monkeypatch, [object()])
        assert resolver._resolve_sync("not a video id!") is None
        assert seen["attempts"] == []
        assert seen["sleeps"] == []


class TestPendingJoinVerdict:
    """The join gate behind ``_current_pending``."""

    def test_an_empty_slot_answers_no_join(self):
        assert _pending_join_verdict(None, 3) is None

    def test_a_same_generation_slot_answers_its_future(self):
        future = object()
        assert _pending_join_verdict((3, future), 3) is future

    def test_a_slot_from_another_generation_answers_no_join(self):
        future = object()
        assert _pending_join_verdict((2, future), 3) is None

    def test_the_generation_comparison_is_cpythons_own_ne(self):
        # 3 == 3.0 and True == 1 through the interpreter's own !=, so
        # those spellings join; 3 != "3" does not.
        future = object()
        assert _pending_join_verdict((3.0, future), 3) is future
        assert _pending_join_verdict((1, future), True) is future
        assert _pending_join_verdict((0, future), False) is future
        assert _pending_join_verdict((3, future), "3") is None

    def test_a_non_subscriptable_entry_raises_the_subscripts_own_error(self):
        with pytest.raises(TypeError):
            _pending_join_verdict(7, 3)

    def test_an_empty_entry_answers_the_index_error(self):
        with pytest.raises(IndexError):
            _pending_join_verdict((), 3)

    def test_a_generation_only_entry_answers_the_index_error(self):
        # entry[0] matched the generation, so the [1] fetch is the one
        # that raises.
        with pytest.raises(IndexError):
            _pending_join_verdict((3,), 3)

    def test_a_mismatched_short_entry_never_touches_the_second_element(self):
        # "a" != "b" short-circuits before entry[1], so the missing
        # second element is never fetched — None, not IndexError.
        assert _pending_join_verdict(("a",), "b") is None


class TestPendingDropVerdict:
    """The identity gate behind ``_resolve``'s finally cleanup."""

    def test_an_empty_slot_is_never_dropped(self):
        assert _pending_drop_verdict(None, object()) is False

    def test_the_owners_own_slot_is_dropped(self):
        future = object()
        assert _pending_drop_verdict((3, future), future) is True

    def test_another_future_never_drops_the_slot(self):
        assert _pending_drop_verdict((3, object()), object()) is False

    def test_the_comparison_is_identity_never_equality(self):
        # Two equal-but-distinct futures: the reference drops the slot
        # only for the very object it installed, so a newer request's
        # equal-looking entry survives the old owner's cleanup.
        class EqualFuture:
            def __eq__(self, other):
                return isinstance(other, EqualFuture)

        future, twin = EqualFuture(), EqualFuture()
        assert future == twin
        assert future is not twin
        assert _pending_drop_verdict((3, future), twin) is False
        assert _pending_drop_verdict((3, future), future) is True

    def test_a_none_future_in_the_slot_matches_only_none_itself(self):
        # (3, None) with the owner's future spelled None — the None
        # singleton is identical to itself, so the slot drops.
        assert _pending_drop_verdict((3, None), None) is True
        assert _pending_drop_verdict((3, None), object()) is False

    def test_a_non_subscriptable_entry_raises_the_subscripts_own_error(self):
        with pytest.raises(TypeError):
            _pending_drop_verdict(7, object())

    def test_an_empty_entry_answers_the_index_error(self):
        with pytest.raises(IndexError):
            _pending_drop_verdict((), object())


class TestPendingPublicationGate:
    """The unresolved gate behind the shared future's publications."""

    @staticmethod
    def _stand_in(answer):
        class StandInFuture:
            def done(self):
                return answer

        return StandInFuture()

    def test_an_unresolved_future_may_be_published(self):
        assert _pending_publication_gate(self._stand_in(False)) is True

    def test_a_resolved_future_may_not(self):
        assert _pending_publication_gate(self._stand_in(True)) is False

    def test_the_gate_is_the_interpreters_own_not_over_done(self):
        # done() answering any truthy object blocks the publication and
        # any falsy one allows it — the truthiness protocol, not a
        # bool check.
        for truthy in (5, [1], "x", 0.5, True):
            assert _pending_publication_gate(self._stand_in(truthy)) is False
        for falsy in (0, [], "", 0.0, False, None):
            assert _pending_publication_gate(self._stand_in(falsy)) is True

    def test_a_raising_done_propagates(self):
        class Raising:
            def done(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _pending_publication_gate(Raising())

    def test_a_non_callable_done_answers_the_calls_own_error(self):
        with pytest.raises(TypeError):
            _pending_publication_gate(SimpleNamespace(done=3))

    def test_real_futures_in_every_state(self):
        async def run():
            loop = asyncio.get_running_loop()
            pending = loop.create_future()
            resolved = loop.create_future()
            resolved.set_result(None)
            cancelled = loop.create_future()
            cancelled.cancel()
            failed = loop.create_future()
            failed.set_exception(RuntimeError("x"))
            gates = (
                _pending_publication_gate(pending),
                _pending_publication_gate(resolved),
                _pending_publication_gate(cancelled),
                _pending_publication_gate(failed),
            )
            failed.exception()  # retrieved: no never-retrieved warning
            return gates

        assert asyncio.run(run()) == (True, False, False, False)


class TestPendingEngineWiring:
    """The resolver's dedup flow consults the module-level engine
    names, so whichever backend is loaded steers the flow identically —
    the routing the shielding, takeover and orphan tests in
    test_stream.py exercise end-to-end on both backends."""

    def test_current_pending_routes_through_the_join_engine(self, monkeypatch):
        from ytm_player.services import stream

        resolver = StreamResolver()
        consults = []
        real = stream._pending_join_verdict

        def spy(entry, generation):
            consults.append((entry, generation))
            return real(entry, generation)

        monkeypatch.setattr(stream, "_pending_join_verdict", spy)

        assert resolver._current_pending("v", 3) is None
        assert consults == [(None, 3)]

        future = object()
        resolver._pending["v"] = (3, future)
        assert resolver._current_pending("v", 3) is future
        assert resolver._current_pending("v", 4) is None
        assert consults == [(None, 3), ((3, future), 3), ((3, future), 4)]

    async def test_the_resolve_flow_consults_the_engines(self, monkeypatch):
        from ytm_player.services import stream

        resolver = StreamResolver()
        gates = []
        drops = []
        real_gate = stream._pending_publication_gate
        real_drop = stream._pending_drop_verdict

        def gate_spy(future):
            verdict = real_gate(future)
            gates.append(verdict)
            return verdict

        def drop_spy(entry, future):
            verdict = real_drop(entry, future)
            drops.append(verdict)
            return verdict

        monkeypatch.setattr(stream, "_pending_publication_gate", gate_spy)
        monkeypatch.setattr(stream, "_pending_drop_verdict", drop_spy)

        info = StreamInfo(
            url="https://x",
            video_id="wiring1",
            format="opus",
            bitrate=1,
            duration=1,
            expires_at=1e9,
        )
        resolver._resolve_sync = lambda video_id: info

        assert await resolver._resolve("wiring1") is info
        # The set_result publication consulted the gate (True, then
        # published); the finally cleanup consulted the drop verdict
        # (True, then deleted) and the cancel gate (False — the future
        # is resolved, so no cancel).
        assert gates == [True, False]
        assert drops == [True]
        assert "wiring1" not in resolver._pending


# Each parity case: the engine's arguments as JSON-round-trippable
# lists, keyed by the engine's native name (the Python module-level
# fallback spells the same name with a leading underscore). Both the
# parent and the child run with stream.py's clock frozen at 1000, so
# every clock-derived value is exact on both sides. The cache is
# carried as a {video_id: expires_at} mapping and rebuilt as the
# attribute-bearing objects the walks read; the cap engine's first
# argument is the _CACHE_MAX_SIZE both sides install before the call;
# the cookiejar-signature engine's cases are carried as (nested) JSON
# lists and rebuilt as the tuples the reference builds, so the
# element-wise tuple comparison — a list never matching a tuple — is
# exercised on both sides. (expires_at never spells NaN in these
# cases: the ranking of a NaN key is sort-algorithm-dependent, so only
# totally ordered values are parity-pinned.)
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
    "stream_ydl_cookiejar_current": [
        [None, None],
        [["/p", [1, 2, 3], None, [5, 6, 7]], ["/p", [1, 2, 3], None, [5, 6, 7]]],
        [["/p", [1, 2, 3], [4, 5], [6, 7]], ["/p", [1, 9, 3], [4, 5], [6, 7]]],
        [["/p", [1, 2, 3], [4, 5], [6, 7]], ["/p", [1, 2, 3], [4, 5], [6, 8]]],
        [["/old", [1, 2, 3], None, None], ["/new", [1, 2, 3], None, None]],
        [["/p", [1, 2, 3], None, None], None],
        [None, ["/p", [1, 2, 3], None, None]],
        [["/p", [1, 5], None, None], ["/p", [1, 5.0], None, None]],
        [[], []],
    ],
    "stream_ydl_publish_verdict": [
        ["ydl", 1, 1],
        ["ydl", 1, 2],
        [None, 1, 2],
        [None, 0, 1],
        [None, 3, 3],
        [None, 0, 0],
        [None, 1, 1.0],
        [None, True, 1],
        [None, "a", "a"],
        [None, "a", "b"],
    ],
    # Each case installs its schedule as the module's _RETRY_DELAYS on
    # both sides before the call (the default shape included); the
    # string case pins the walk's iterator protocol — enumerate runs
    # over any iterable, and so does the native engine.
    "stream_retry_delays": [
        [[0, 1.0, 2.0]],
        [[]],
        [[0]],
        [[5]],
        [[0, 0.5, 0.25, 7]],
        [[2.0, 1]],
        [[True, None, 0.0]],
        ["ab"],
    ],
    # The pending-future engines' cases. The slot entry is carried as
    # a descriptor both sides rebuild: null for the empty slot,
    # [generation, future-descriptor] for an occupied one, a
    # {"nonseq": …} dict for a non-subscriptable stand-in and a
    # {"short": […]} dict for a too-short tuple. The future is its own
    # descriptor: the literal its done() answers, or a
    # {"raise": …} / {"noncallable": …} dict for the protocol's error
    # shapes. The join outcome names which object came back — "slot"
    # for the very future the case built, null for no-join — so
    # identity, not value, is what both sides must agree on; the drop
    # cases' third flag passes the entry's own future (true) or a
    # distinct twin built from the same descriptor (false), pinning
    # the identity comparison on both sides.
    "stream_pending_join_verdict": [
        [None, 3],
        [[3, False], 3],
        [[3, False], 4],
        [[3, False], 3.0],
        [[1, False], True],
        [[0, False], False],
        [[3, True], 4],
        [["a", False], "a"],
        [["a", False], "b"],
        [[True, 0.0], 1],
        [{"nonseq": 7}, 3],
        [{"short": []}, 3],
        [{"short": [3]}, 3],
        [{"short": ["a"]}, "b"],
    ],
    "stream_pending_drop_verdict": [
        [None, False, False],
        [[3, False], False, True],
        [[3, False], False, False],
        [[3, True], True, True],
        [[3, False], True, False],
        [[7, None], None, True],
        [{"short": [3, None]}, None, True],
        [{"short": [3, None]}, False, False],
        [{"nonseq": 7}, False, False],
        [{"short": []}, False, False],
    ],
    "stream_pending_publication_gate": [
        [[False]],
        [[True]],
        [[5]],
        [[0]],
        [[""]],
        [["x"]],
        [[0.5]],
        [[0.0]],
        [[None]],
        [[[1]]],
        [[[]]],
        [[{"raise": "boom"}]],
        [[{"noncallable": 3}]],
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
    "stream_retry_delays": "_retry_delays",
    "stream_ydl_cookiejar_current": "_ydl_cookiejar_current",
    "stream_ydl_publish_verdict": "_ydl_publish_verdict",
    "stream_pending_join_verdict": "_pending_join_verdict",
    "stream_pending_drop_verdict": "_pending_drop_verdict",
    "stream_pending_publication_gate": "_pending_publication_gate",
}


def json_sig(value):
    # The JSON case's signature re-packed as the nested tuples the
    # reference builds (a list never equals a tuple, so both sides
    # must spell the same shape).
    if isinstance(value, list):
        return tuple(json_sig(item) for item in value)
    return value


class StandInFuture:
    # A future stand-in whose done() answers the descriptor's literal
    # (or raises / is non-callable) — the object both sides build from
    # the same JSON case, so identity comparisons are meaningful.
    # Twins built from the same descriptor compare equal, so the drop
    # cases' distinct-twin spelling genuinely separates `is` from `==`.

    def __init__(self, desc):
        self._desc = desc

    def done(self):
        if isinstance(self._desc, dict) and "raise" in self._desc:
            raise RuntimeError(self._desc["raise"])
        return self._desc

    def __eq__(self, other):
        return isinstance(other, StandInFuture) and self._desc == other._desc


def make_future(desc):
    if isinstance(desc, dict) and "noncallable" in desc:
        return SimpleNamespace(done=desc["noncallable"])
    return StandInFuture(desc)


def make_entry(desc):
    if desc is None:
        return None
    if isinstance(desc, dict):
        if "nonseq" in desc:
            return desc["nonseq"]
        return tuple(desc["short"])
    return (desc[0], make_future(desc[1]))


def call_args(name, args):
    if name == "stream_cache_expired_ids":
        return [{vid: SimpleNamespace(expires_at=exp) for vid, exp in args[0].items()}]
    if name == "stream_cache_cap_evictions":
        stream._CACHE_MAX_SIZE = args[0]
        return [{vid: SimpleNamespace(expires_at=exp) for vid, exp in args[1].items()}]
    if name == "stream_retry_delays":
        stream._RETRY_DELAYS = args[0]
        return []
    if name == "stream_ydl_cookiejar_current":
        return [json_sig(args[0]), json_sig(args[1])]
    if name == "stream_pending_drop_verdict":
        entry = make_entry(args[0])
        if args[2] and isinstance(entry, tuple) and len(entry) == 2:
            return [entry, entry[1]]
        return [entry, make_future(args[1])]
    if name == "stream_pending_publication_gate":
        return [make_future(args[0])]
    return args


def outcome(name, args):
    fn = getattr(stream, PY_NAMES[name])
    if name == "stream_pending_join_verdict":
        # Built once here so the identity pin compares against the very
        # object the engine was handed.
        entry = make_entry(args[0])
        try:
            result = fn(entry, args[1])
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if result is None:
            return {"ok": None}
        if isinstance(entry, tuple) and len(entry) == 2 and result is entry[1]:
            return {"ok": "slot"}
        return {"ok": "other"}
    try:
        result = fn(*call_args(name, args))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if name == "stream_info_fields":
        return {"ok": None if result is None else list(result)}
    if name == "stream_base_opts":
        return {"ok": {"keys": list(result), "dict": {k: v for k, v in result.items() if k != "logger"}}}
    if name == "stream_retry_delays":
        return {"ok": [[attempt, delay] for attempt, delay in result]}
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
    "stream_retry_delays": "_retry_delays",
    "stream_ydl_cookiejar_current": "_ydl_cookiejar_current",
    "stream_ydl_publish_verdict": "_ydl_publish_verdict",
    "stream_pending_join_verdict": "_pending_join_verdict",
    "stream_pending_drop_verdict": "_pending_drop_verdict",
    "stream_pending_publication_gate": "_pending_publication_gate",
}


class _StandInFuture:
    """A future stand-in whose ``done()`` answers the descriptor's
    literal (or raises / is non-callable) — the object both sides build
    from the same JSON case, so identity comparisons are meaningful.
    Twins built from the same descriptor compare equal, so the drop
    cases' distinct-twin spelling genuinely separates ``is`` from
    ``==``; the child spells the identical class inside its own
    program."""

    def __init__(self, desc):
        self._desc = desc

    def done(self):
        if isinstance(self._desc, dict) and "raise" in self._desc:
            raise RuntimeError(self._desc["raise"])
        return self._desc

    def __eq__(self, other):
        return isinstance(other, _StandInFuture) and self._desc == other._desc


def _make_future(desc):
    if isinstance(desc, dict) and "noncallable" in desc:
        return SimpleNamespace(done=desc["noncallable"])
    return _StandInFuture(desc)


def _make_entry(desc):
    """The pending slot rebuilt from the JSON case: None for the empty
    slot, a (generation, stand-in-future) pair for an occupied one, a
    non-subscriptable stand-in, or a too-short tuple."""
    if desc is None:
        return None
    if isinstance(desc, dict):
        if "nonseq" in desc:
            return desc["nonseq"]
        return tuple(desc["short"])
    return (desc[0], _make_future(desc[1]))


def _cache_call_args(name, args):
    """The cache rebuilt from the JSON case as the attribute-bearing
    objects the walks read; the cap engine's case also installs its
    cap first and the retry-delay engine's case its schedule (the
    caller restores both once the engine has run, so the module
    constants never leak); the cookiejar-signature engine's cases are
    re-packed as the nested tuples the reference builds; the pending
    engines' cases are rebuilt as the slot/future stand-ins the
    verdicts read (the drop case's third flag handing the entry's own
    future or a distinct twin)."""
    from ytm_player.services import stream

    if name == "stream_cache_expired_ids":
        return [{vid: _entry(exp) for vid, exp in args[0].items()}], None
    if name == "stream_cache_cap_evictions":
        saved = stream._CACHE_MAX_SIZE
        stream._CACHE_MAX_SIZE = args[0]
        return [{vid: _entry(exp) for vid, exp in args[1].items()}], ("_CACHE_MAX_SIZE", saved)
    if name == "stream_retry_delays":
        saved = stream._RETRY_DELAYS
        stream._RETRY_DELAYS = args[0]
        return [], ("_RETRY_DELAYS", saved)
    if name == "stream_ydl_cookiejar_current":
        return [_json_sig_case(args[0]), _json_sig_case(args[1])], None
    if name == "stream_pending_drop_verdict":
        entry = _make_entry(args[0])
        if args[2] and isinstance(entry, tuple) and len(entry) == 2:
            return [entry, entry[1]], None
        return [entry, _make_future(args[1])], None
    if name == "stream_pending_publication_gate":
        return [_make_future(args[0])], None
    return args, None


def _native_outcomes():
    """The parent-side outcomes, computed against the module's current
    (frozen) clock — the same rebinding the child performs."""

    def outcome(name, args):
        fn = getattr(sys.modules["ytm_player.services.stream"], _PY_NAMES[name])
        if name == "stream_pending_join_verdict":
            # Built once here so the identity pin compares against the
            # very object the engine was handed — the same shape the
            # child's outcome spells.
            entry = _make_entry(args[0])
            try:
                result = fn(entry, args[1])
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                return {"error": f"{type(exc).__name__}: {exc}"}
            if result is None:
                return {"ok": None}
            if isinstance(entry, tuple) and len(entry) == 2 and result is entry[1]:
                return {"ok": "slot"}
            return {"ok": "other"}
        call_args, restore = _cache_call_args(name, args)
        try:
            result = fn(*call_args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        finally:
            if restore is not None:
                from ytm_player.services import stream

                setattr(stream, restore[0], restore[1])
        if name == "stream_info_fields":
            return {"ok": None if result is None else list(result)}
        if name == "stream_base_opts":
            return {
                "ok": {
                    "keys": list(result),
                    "dict": {k: v for k, v in result.items() if k != "logger"},
                }
            }
        if name == "stream_retry_delays":
            return {"ok": [[attempt, delay] for attempt, delay in result]}
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
