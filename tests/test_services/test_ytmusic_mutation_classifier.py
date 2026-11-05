"""Tests for the mutation-failure classifier engine.

The engine behind every mutation call's except arm in
``services/ytmusic.py``: the exception-to-kind mapping
``_classify_mutation_failure`` — the reference's isinstance chain in
its exact order (KeyError, the two timeout classes, requests'
``RequestException``, ``YTMusicUserError``, ``YTMusicServerError``),
with the YTMusicServerError arm parsing the HTTP status out of the
message string through CPython's own ``\\d{3}\\b`` semantics on str
patterns (Unicode decimal digits of any script and their ``int()``
values included). The pure-Python reference lives in
``services/ytmusic.py``; when the Rust extension is built the same
engine runs natively (``_ytm_player_native.classify_mutation_failure``).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) exception for
exception, so the two implementations cannot drift apart.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest
import requests.exceptions
from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError

from ytm_player.services.ytmusic import _classify_mutation_failure

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── The isinstance chain, in its exact order ───────────────────────


class TestExceptionChain:
    def test_key_error_is_server_error(self):
        # ytmusicapi parser drift: a server-side problem, never
        # "network".
        assert _classify_mutation_failure(KeyError("playlistPanelRenderer")) == "server_error"

    def test_wait_for_timeout_is_a_timeout(self):
        assert _classify_mutation_failure(asyncio.TimeoutError()) == "timeout"

    def test_builtin_timeout_error_is_a_timeout(self):
        # On 3.11+ asyncio.TimeoutError *is* the builtin TimeoutError —
        # the native engine resolves the class from the running
        # interpreter, so this holds on either spelling. On 3.10 the
        # builtin is a distinct class the reference's chain never
        # lists, so it falls through to the server_error default.
        if sys.version_info >= (3, 11):
            assert _classify_mutation_failure(TimeoutError()) == "timeout"
        else:
            assert _classify_mutation_failure(TimeoutError()) == "server_error"

    def test_read_timeout_is_a_timeout(self):
        assert _classify_mutation_failure(requests.exceptions.ReadTimeout("slow")) == "timeout"

    def test_read_timeout_wins_over_network(self):
        # ReadTimeout is both a Timeout and a RequestException — the
        # chain's order decides, and the timeout arm comes first.
        assert _classify_mutation_failure(requests.exceptions.ReadTimeout("slow")) == "timeout"

    def test_connect_timeout_stays_network(self):
        assert _classify_mutation_failure(requests.exceptions.ConnectTimeout("refused")) == (
            "network"
        )

    def test_connection_error_stays_network(self):
        assert _classify_mutation_failure(requests.exceptions.ConnectionError("refused")) == (
            "network"
        )

    def test_plain_request_exception_is_network(self):
        assert _classify_mutation_failure(requests.exceptions.RequestException("boom")) == (
            "network"
        )

    def test_user_error_is_auth_required(self):
        assert _classify_mutation_failure(YTMusicUserError("unauthorized")) == "auth_required"

    def test_key_error_wins_over_everything(self):
        # A class matching two arms takes the chain's first — KeyError
        # before the ytmusicapi types.
        class KeyedServerError(KeyError, YTMusicServerError):
            pass

        assert _classify_mutation_failure(KeyedServerError("x")) == "server_error"

    def test_unknown_exception_is_server_error(self):
        # The reference's unreachable-in-practice safety default.
        class Weird(BaseException):
            pass

        assert _classify_mutation_failure(Weird("???")) == "server_error"

    def test_plain_oserror_is_server_error(self):
        assert _classify_mutation_failure(OSError("nope")) == "server_error"


# ── The HTTP status prefix — CPython's \d{3}\b on str patterns ─────


class TestHttpStatusPrefix:
    def test_401_is_auth_expired(self):
        exc = YTMusicServerError("Server returned HTTP 401: Unauthorized.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_403_is_auth_expired(self):
        exc = YTMusicServerError("Server returned HTTP 403: Forbidden.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_500_is_server_error(self):
        exc = YTMusicServerError("Server returned HTTP 500: Internal Server Error.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_404_is_server_error(self):
        exc = YTMusicServerError("Server returned HTTP 404: Not Found.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_two_digit_status_does_not_match(self):
        # \d{3} needs exactly three digits; the fixed-count group has
        # no shorter retry under the anchored literal prefix.
        exc = YTMusicServerError("Server returned HTTP 40: Bad Request.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_four_digit_status_does_not_match(self):
        # The fourth digit is a word character — \b fails after three.
        exc = YTMusicServerError("Server returned HTTP 4014: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_three_digits_at_end_of_message_match(self):
        # \b holds at the end of the string.
        exc = YTMusicServerError("Server returned HTTP 401")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_letter_after_digits_breaks_the_boundary(self):
        exc = YTMusicServerError("Server returned HTTP 401a: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_word_char_emoji_after_digits_breaks_the_boundary(self):
        # Emoji are not \w in Python — the boundary holds, 401 parses.
        exc = YTMusicServerError("Server returned HTTP 401🎵: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_alphanumeric_fraction_after_digits_breaks_the_boundary(self):
        # ½ is No — alphanumeric for CPython's \w, so the boundary
        # after the third digit fails and no status is parsed.
        exc = YTMusicServerError("Server returned HTTP 401½: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_circled_digit_after_digits_breaks_the_boundary(self):
        # ① is No too — not a \d (so the group can't extend) but
        # alnum (so the boundary can't hold).
        exc = YTMusicServerError("Server returned HTTP 401①: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_combining_mark_after_digits_keeps_the_boundary(self):
        # A combining mark is NOT \w for CPython (regex \w it is not)
        # — the boundary holds, the status parses.
        exc = YTMusicServerError("Server returned HTTP 401́: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_underscore_after_digits_breaks_the_boundary(self):
        exc = YTMusicServerError("Server returned HTTP 401_: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_lowercase_prefix_does_not_match(self):
        exc = YTMusicServerError("server returned http 401: x")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_leading_space_does_not_match(self):
        exc = YTMusicServerError(" Server returned HTTP 401: x")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_empty_message_is_server_error(self):
        exc = YTMusicServerError("")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_unrelated_message_is_server_error(self):
        exc = YTMusicServerError("Something else went wrong")
        assert _classify_mutation_failure(exc) == "server_error"


class TestUnicodeDigits:
    """CPython's ``\\d`` on str patterns is category Nd — any script —
    and ``int()`` parses every Nd trio. A Unicode-digit 401/403 is an
    auth_expired exactly like an ASCII one."""

    def test_arabic_indic_401_is_auth_expired(self):
        exc = YTMusicServerError("Server returned HTTP ٤٠١: Unauthorized.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_devanagari_403_is_auth_expired(self):
        exc = YTMusicServerError("Server returned HTTP ४०३: Forbidden.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_fullwidth_401_is_auth_expired(self):
        exc = YTMusicServerError("Server returned HTTP ４０１: Unauthorized.\n<body>")
        assert _classify_mutation_failure(exc) == "auth_expired"

    def test_arabic_indic_500_stays_server_error(self):
        exc = YTMusicServerError("Server returned HTTP ٥٠٠: Error.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_non_decimal_digit_run_does_not_match(self):
        # ² is No — \d refuses it (only three Nd available after the
        # prefix, the fourth char breaks nothing because the group
        # already fixed three)… but 401 followed by ² leaves \b intact?
        # No: ² is alnum (\w), so the boundary fails — server_error.
        exc = YTMusicServerError("Server returned HTTP 401²: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"

    def test_circled_digit_is_not_a_digit(self):
        # ① is No, not Nd — \d{3} cannot complete on it.
        exc = YTMusicServerError("Server returned HTTP ④①: Weird.\n<body>")
        assert _classify_mutation_failure(exc) == "server_error"


# ── The suffix table (cascade-site surface) ────────────────────────


class TestSuffixTable:
    def test_every_kind_has_a_distinct_suffix(self):
        from ytm_player.services.ytmusic import mutation_failure_suffix

        kinds = [
            "success",
            "auth_required",
            "auth_expired",
            "network",
            "timeout",
            "rejected",
            "server_error",
            "duplicate",
        ]
        suffixes = {kind: mutation_failure_suffix(kind) for kind in kinds}
        assert suffixes["success"] == ""
        assert len({s for s in suffixes.values() if s}) == len(kinds) - 1


# ── Native parity ──────────────────────────────────────────────────

# The child interpreter runs this program with the case list on stdin
# and prints the reference engine's results as JSON. Each case is a
# [kind, payload] pair; both sides build the very same exception from
# it and classify.
_CHILD_PROG = """
import asyncio
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

import requests.exceptions as rq
from ytmusicapi.exceptions import YTMusicServerError, YTMusicUserError

from ytm_player.services.ytmusic import _classify_mutation_failure

cases = json.load(sys.stdin)


class Weird(BaseException):
    pass


class KeyedServerError(KeyError, YTMusicServerError):
    pass


def build(case):
    kind, payload = case
    if kind == 'server':
        return YTMusicServerError(payload)
    if kind == 'user':
        return YTMusicUserError(payload)
    if kind == 'key':
        return KeyError(payload)
    if kind == 'timeout':
        return asyncio.TimeoutError()
    if kind == 'readtimeout':
        return rq.ReadTimeout(payload)
    if kind == 'connecttimeout':
        return rq.ConnectTimeout(payload)
    if kind == 'connection':
        return rq.ConnectionError(payload)
    if kind == 'request':
        return rq.RequestException(payload)
    if kind == 'weird':
        return Weird(payload)
    if kind == 'plain':
        return Exception(payload)
    if kind == 'oserror':
        return OSError(payload)
    if kind == 'keyedserver':
        return KeyedServerError(payload)
    raise AssertionError(kind)


out = []
for case in cases:
    exc = build(case)
    try:
        out.append({'ok': _classify_mutation_failure(exc)})
    except Exception as exc:
        out.append({'error': type(exc).__name__})
json.dump(out, sys.stdout)
"""

# (kind, payload) pairs the two implementations must agree on: every
# arm of the chain, the boundary matrix (end-of-string, word char,
# emoji, fraction, circled digit, combining mark, underscore), the
# anchored-prefix refusals, and the Unicode-digit trio (Arabic-Indic,
# Devanagari, fullwidth) whose int() values are 401/403/500.
_PARITY_CASES = [
    ("key", "playlistPanelRenderer"),
    ("key", None),
    ("timeout", None),
    ("readtimeout", "slow read"),
    ("connecttimeout", "refused"),
    ("connection", "refused"),
    ("request", "boom"),
    ("user", "unauthorized"),
    ("weird", "???"),
    ("plain", "mystery"),
    ("oserror", "nope"),
    ("keyedserver", "both arms"),
    ("server", "Server returned HTTP 401: Unauthorized.\nbody"),
    ("server", "Server returned HTTP 403: Forbidden.\nbody"),
    ("server", "Server returned HTTP 500: Internal Server Error.\nbody"),
    ("server", "Server returned HTTP 404: Not Found.\nbody"),
    ("server", "Server returned HTTP 40: Bad Request.\nbody"),
    ("server", "Server returned HTTP 4014: Weird.\nbody"),
    ("server", "Server returned HTTP 401"),
    ("server", "Server returned HTTP 401a: Weird.\nbody"),
    ("server", "Server returned HTTP 401🎵: Weird.\nbody"),
    ("server", "Server returned HTTP 401½: Weird.\nbody"),
    ("server", "Server returned HTTP 401①: Weird.\nbody"),
    ("server", "Server returned HTTP 401́: Weird.\nbody"),
    ("server", "Server returned HTTP 401_: Weird.\nbody"),
    ("server", "Server returned HTTP 401²: Weird.\nbody"),
    ("server", "Server returned HTTP ④①: Weird.\nbody"),
    ("server", "server returned http 401: x"),
    ("server", " Server returned HTTP 401: x"),
    ("server", ""),
    ("server", "Something else went wrong"),
    ("server", "Server returned HTTP ٤٠١: Unauthorized.\nbody"),
    ("server", "Server returned HTTP ٤٠٣: Forbidden.\nbody"),
    ("server", "Server returned HTTP ٤٠٠: Bad.\nbody"),
    ("server", "Server returned HTTP ४०३: Forbidden.\nbody"),
    ("server", "Server returned HTTP ４０１: Unauthorized.\nbody"),
    ("server", "Server returned HTTP ４００: Bad.\nbody"),
    ("server", "Server returned HTTP ٥٠٠: Error.\nbody"),
    ("server", "Server returned HTTP ٤٠١: end"),
]


def _native_results(cases: list) -> list:
    """Run whichever engine the module loaded (native when built) over
    *cases*, building the very exceptions the child builds."""

    class Weird(BaseException):
        pass

    class KeyedServerError(KeyError, YTMusicServerError):
        pass

    def build(case):
        kind, payload = case
        if kind == "server":
            return YTMusicServerError(payload)
        if kind == "user":
            return YTMusicUserError(payload)
        if kind == "key":
            return KeyError(payload)
        if kind == "timeout":
            return asyncio.TimeoutError()
        if kind == "readtimeout":
            return requests.exceptions.ReadTimeout(payload)
        if kind == "connecttimeout":
            return requests.exceptions.ConnectTimeout(payload)
        if kind == "connection":
            return requests.exceptions.ConnectionError(payload)
        if kind == "request":
            return requests.exceptions.RequestException(payload)
        if kind == "weird":
            return Weird(payload)
        if kind == "plain":
            return Exception(payload)
        if kind == "oserror":
            return OSError(payload)
        if kind == "keyedserver":
            return KeyedServerError(payload)
        raise AssertionError(kind)

    out = []
    for case in cases:
        exc = build(case)
        try:
            out.append({"ok": _classify_mutation_failure(exc)})
        except Exception as exc:  # pragma: no cover — parity with the child's catch
            out.append({"error": type(exc).__name__})
    return out


def _fallback_results(cases: list) -> list:
    """Run the pure-Python reference engine over *cases* in a child
    interpreter with the native extension import blocked."""
    payload = json.dumps(cases)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native classifier must
    answer every exception exactly like the pure-Python reference run
    in a child interpreter with the extension import blocked — the
    chain's order, the boundary matrix and the Unicode-digit trio
    included — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _native_results(_PARITY_CASES)
        fallback = _fallback_results(_PARITY_CASES)
        for case, native_result, fallback_result in zip(_PARITY_CASES, native, fallback):
            assert native_result == fallback_result, (
                f"native and fallback classifier results differ for case {case!r}"
            )
