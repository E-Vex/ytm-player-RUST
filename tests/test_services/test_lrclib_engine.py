"""Tests for the LRCLIB lookup glue engines.

The pure cores behind ``get_synced_lyrics`` in ``services/lrclib.py``:
the query-params assembly ``_query_params`` (the track_name /
artist_name pair in the reference's insertion order, plus the
identity-gated duration through ``str(int(...))``) and the
synced-lyrics extraction ``_synced_lyrics`` (the mapping's own ``get``
plus the ``or None`` truthiness gate). The pure-Python references live
in ``services/lrclib.py``; when the Rust extension is built the same
engines run natively (``_ytm_player_native.lrclib_query_params`` /
``lrclib_synced_lyrics``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
error shapes included, so the two implementations cannot drift.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ytm_player.services import lrclib as _lrclib
from ytm_player.services.lrclib import get_synced_lyrics

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── _query_params ──────────────────────────────────────────────────


class TestQueryParams:
    def test_base_pair_in_insertion_order(self):
        params = _lrclib._query_params("Clean Title", "Artist", None)
        assert params == {"track_name": "Clean Title", "artist_name": "Artist"}
        assert list(params) == ["track_name", "artist_name"]

    def test_duration_none_is_the_identity_gate(self):
        # `is not None`, never truthiness — a 0 duration still travels.
        params = _lrclib._query_params("T", "A", 0)
        assert params["duration"] == "0"

    def test_duration_false_coerces_like_int(self):
        params = _lrclib._query_params("T", "A", False)
        assert params["duration"] == "0"

    def test_duration_float_truncates(self):
        params = _lrclib._query_params("T", "A", 190.7)
        assert params["duration"] == "190"

    def test_duration_negative_float_truncates_toward_zero(self):
        params = _lrclib._query_params("T", "A", -0.9)
        assert params["duration"] == "0"

    def test_duration_numeric_string_parses(self):
        params = _lrclib._query_params("T", "A", "42")
        assert params["duration"] == "42"

    def test_duration_bad_string_raises_ints_value_error(self):
        with pytest.raises(ValueError, match="invalid literal for int"):
            _lrclib._query_params("T", "A", "12.5")

    def test_title_and_artist_pass_through_as_the_objects_they_are(self):
        title = "Title"
        artist = "Artist"
        params = _lrclib._query_params(title, artist, None)
        assert params["track_name"] is title
        assert params["artist_name"] is artist

    def test_empty_strings_travel(self):
        params = _lrclib._query_params("", "", None)
        assert params == {"track_name": "", "artist_name": ""}

    def test_unicode_travels(self):
        params = _lrclib._query_params("歌曲名 (Live)", "歌手", 200)
        assert params["track_name"] == "歌曲名 (Live)"
        assert params["artist_name"] == "歌手"
        assert params["duration"] == "200"


# ── _synced_lyrics ─────────────────────────────────────────────────


class TestSyncedLyrics:
    def test_member_present_passes_through(self):
        lrc = "[00:01.00] line one\n[00:05.00] line two"
        assert _lrclib._synced_lyrics({"syncedLyrics": lrc}) == lrc

    def test_member_passes_through_as_the_object_it_is(self):
        lrc = "[00:01] hi"
        assert _lrclib._synced_lyrics({"syncedLyrics": lrc}) is lrc

    def test_missing_key_answers_none(self):
        assert _lrclib._synced_lyrics({"plainLyrics": "text"}) is None

    def test_explicit_none_answers_none(self):
        assert _lrclib._synced_lyrics({"syncedLyrics": None}) is None

    def test_empty_string_answers_none(self):
        # The `or None` truthiness gate.
        assert _lrclib._synced_lyrics({"syncedLyrics": ""}) is None

    def test_falsy_member_answers_none(self):
        assert _lrclib._synced_lyrics({"syncedLyrics": 0}) is None
        assert _lrclib._synced_lyrics({"syncedLyrics": []}) is None

    def test_dict_subclass_get_steers_the_read(self):
        class MyDict(dict):
            def get(self, key, default=None):
                if key == "syncedLyrics":
                    return "steered"
                return super().get(key, default)

        assert _lrclib._synced_lyrics(MyDict(other=1)) == "steered"

    def test_get_with_default_still_gated_by_truthiness(self):
        class DefaultDict(dict):
            def get(self, key, default=None):
                return super().get(key, "fallback")

        # A missing member answers the get's own default; an existing
        # falsy member (the empty string) is what the default never
        # replaces, and the `or None` gate turns it into None.
        assert _lrclib._synced_lyrics(DefaultDict()) == "fallback"
        assert _lrclib._synced_lyrics(DefaultDict(syncedLyrics="")) is None

    def test_non_mapping_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            _lrclib._synced_lyrics([1, 2, 3])


# ── get_synced_lyrics end-to-end (network mocked) ──────────────────


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestGetSyncedLyricsFlow:
    """The public fetch flow on whichever backend is loaded — the
    sanitized title, the engine-built params and the engine extraction
    around the mocked requests call."""

    async def test_round_trip_with_mocked_fetch(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, headers=None, timeout=None):
            captured.update(url=url, params=params, timeout=timeout)
            return _FakeResponse({"syncedLyrics": "[00:01.00] hi"})

        monkeypatch.setattr(_lrclib.requests, "get", fake_get)
        result = await get_synced_lyrics("Song Name (Official Video)", "Artist", 190.7)
        assert result == "[00:01.00] hi"
        assert captured["url"] == "https://lrclib.net/api/get"
        # The sanitized title + the truncated duration travel.
        assert captured["params"] == {
            "track_name": "Song Name",
            "artist_name": "Artist",
            "duration": "190",
        }
        assert captured["timeout"] == 5

    async def test_none_duration_omits_the_param(self, monkeypatch):
        captured: dict = {}

        def fake_get(url, params=None, headers=None, timeout=None):
            captured["params"] = params
            return _FakeResponse({"syncedLyrics": None})

        monkeypatch.setattr(_lrclib.requests, "get", fake_get)
        assert await get_synced_lyrics("Song", "Artist") is None
        assert captured["params"] == {"track_name": "Song", "artist_name": "Artist"}

    async def test_fetch_failure_answers_none(self, monkeypatch):
        def fake_get(url, params=None, headers=None, timeout=None):
            raise OSError("network down")

        monkeypatch.setattr(_lrclib.requests, "get", fake_get)
        assert await get_synced_lyrics("Song", "Artist") is None

    async def test_empty_synced_lyrics_answers_none(self, monkeypatch):
        monkeypatch.setattr(
            _lrclib.requests, "get", lambda *a, **k: _FakeResponse({"syncedLyrics": ""})
        )
        assert await get_synced_lyrics("Song", "Artist") is None


# ── native vs fallback parity ───────────────────────────────────────

# Each case is a Python expression evaluated against the engines in
# two interpreters: this one (native extension loaded) and a child
# where the extension import is blocked (pure-Python fallback). Their
# reprs must match exactly — the param dicts with their insertion
# order, the extraction verdicts, and the error reprs for the
# int()-rejections and the missing-get payloads.
_PARITY_CASES = [
    "lrclib._query_params('Clean Title', 'Artist', None)",
    "lrclib._query_params('T', 'A', 0)",
    "lrclib._query_params('T', 'A', False)",
    "lrclib._query_params('T', 'A', True)",
    "lrclib._query_params('T', 'A', 190.7)",
    "lrclib._query_params('T', 'A', -0.9)",
    "lrclib._query_params('T', 'A', 42)",
    "lrclib._query_params('T', 'A', '42')",
    "lrclib._query_params('T', 'A', '  7  ')",
    "lrclib._query_params('', '', 1)",
    "lrclib._query_params('歌曲名', '歌手', 200)",
    "lrclib._query_params('T', 'A', '12.5')",
    "lrclib._query_params('T', 'A', 'abc')",
    "lrclib._query_params('T', 'A', object())",
    "lrclib._synced_lyrics({'syncedLyrics': '[00:01] hi'})",
    "lrclib._synced_lyrics({'syncedLyrics': ''})",
    "lrclib._synced_lyrics({'syncedLyrics': None})",
    "lrclib._synced_lyrics({})",
    "lrclib._synced_lyrics({'syncedLyrics': 0})",
    "lrclib._synced_lyrics({'syncedLyrics': []})",
    "lrclib._synced_lyrics({'syncedLyrics': 'text'})",
    "lrclib._synced_lyrics([1, 2, 3])",
    "lrclib._synced_lyrics(None)",
]


def _eval_in_child(expr: str) -> str:
    prog = (
        "import sys\n"
        "sys.modules['ytm_player.utils._ytm_player_native'] = None\n"
        "from ytm_player.services import lrclib\n"
        "try:\n"
        f"    print(repr({expr}))\n"
        "except BaseException as e:\n"
        "    print('ERR', type(e).__name__, repr(str(e)))\n"
    )
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _eval_here(expr: str) -> str:
    try:
        return repr(eval(expr, {"lrclib": _lrclib}))
    except BaseException as e:  # noqa: BLE001 - repr'd for comparison
        return f"ERR {type(e).__name__} {str(e)!r}"


class TestNativeParity:
    """When the Rust extension is built, the native LRCLIB engines must
    answer every case exactly like the pure-Python references run in a
    child interpreter with the extension import blocked — the param
    dicts with their insertion order, the extraction verdicts and the
    error shapes included — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self):
        assert _lrclib._query_params.__module__ != __name__  # sanity: module binding
        for expr in _PARITY_CASES:
            native = _eval_here(expr)
            fallback = _eval_in_child(expr)
            assert native == fallback, f"{expr}: {native!r} != {fallback!r}"
