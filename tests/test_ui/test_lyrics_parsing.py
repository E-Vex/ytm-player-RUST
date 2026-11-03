"""Tests for the synced-lyrics LRC parser.

The parser behind ``LyricsSidebar``'s string-lyrics path
(``_parse_synced_lyrics`` from ``ui/sidebars/lyrics_sidebar.py``): the
timestamp match at each line's start, the fractional-second conversion
and the stable timestamp sort. The pure-Python reference lives in the
sidebar module; when the Rust extension is built the same engine runs
natively (``_ytm_player_native.parse_synced_lyrics``).

The unit tests below assert the parser's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.sidebars.lyrics_sidebar import _parse_synced_lyrics

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Timestamp extraction ───────────────────────────────────────────


class TestTimestampExtraction:
    def test_basic(self):
        assert _parse_synced_lyrics("[00:12.50] hello") == [(12.5, "hello")]

    def test_fraction_digit_count_decides_the_value(self):
        # .5, .50 and .500 are all a half — int(digits) / 10 ** len(digits).
        assert _parse_synced_lyrics("[00:12.5] a") == [(12.5, "a")]
        assert _parse_synced_lyrics("[00:12.50] b") == [(12.5, "b")]
        assert _parse_synced_lyrics("[00:12.500] c") == [(12.5, "c")]

    def test_two_digit_fraction_is_hundredths(self):
        assert _parse_synced_lyrics("[00:12.05] a") == [(12.05, "a")]

    def test_single_digit_minute(self):
        assert _parse_synced_lyrics("[1:23] x") == [(83.0, "x")]

    def test_large_fields(self):
        # The pattern admits up to 99:99 — 99 minutes, 99 seconds.
        assert _parse_synced_lyrics("[99:99] big") == [(99 * 60 + 99, "big")]

    def test_zero_timestamp(self):
        assert _parse_synced_lyrics("[0:00] zero") == [(0.0, "zero")]

    def test_lines_without_timestamps_are_skipped(self):
        raw = "no timestamps at all\nplain text\n[00:12] kept"
        assert _parse_synced_lyrics(raw) == [(12.0, "kept")]

    def test_timestamp_must_start_the_line(self):
        # re.match semantics: a mid-line timestamp alone is not a lyric.
        assert _parse_synced_lyrics("text [00:12] mid-line only") == []

    def test_malformed_timestamps_are_skipped(self):
        assert _parse_synced_lyrics("[12:3] one-digit second") == []
        assert _parse_synced_lyrics("[123:45] three-digit minute") == []

    def test_empty_input(self):
        assert _parse_synced_lyrics("") == []


# ── Text handling ──────────────────────────────────────────────────


class TestTextHandling:
    def test_padded_lines_are_stripped(self):
        assert _parse_synced_lyrics("  [00:12.00]  padded line  ") == [(12.0, "padded line")]

    def test_every_timestamp_shape_is_stripped_from_the_line(self):
        # re.sub removes all matches, not just the leading one.
        assert _parse_synced_lyrics("[00:12][01:00] both stripped") == [(12.0, "both stripped")]

    def test_empty_lyric_line(self):
        assert _parse_synced_lyrics("[00:12]   ") == [(12.0, "")]

    def test_unicode_digits_parse(self):
        # Both re's \d and int() read Nd digits of any script.
        assert _parse_synced_lyrics("[٠١:٢٣.٥٠] arabic") == [(83.5, "arabic")]


# ── Ordering and line boundaries ───────────────────────────────────


class TestOrderingAndBoundaries:
    def test_output_is_sorted_by_timestamp(self):
        raw = "[00:20] c\n[00:05] a\n[00:12] b"
        assert [text for _, text in _parse_synced_lyrics(raw)] == ["a", "b", "c"]

    def test_equal_timestamps_keep_line_order(self):
        raw = "[00:05] first\n[00:05] second"
        assert [text for _, text in _parse_synced_lyrics(raw)] == ["first", "second"]

    def test_crlf_and_lone_cr_boundaries(self):
        raw = "[00:12] a\r\n[00:20] b\r[00:30] c"
        assert _parse_synced_lyrics(raw) == [(12.0, "a"), (20.0, "b"), (30.0, "c")]

    def test_unicode_line_separator_is_a_boundary(self):
        raw = "[00:12.50]\u2028[00:20.00] after"
        assert _parse_synced_lyrics(raw) == [(12.5, ""), (20.0, "after")]

    def test_trailing_newline_produces_no_empty_pair(self):
        assert _parse_synced_lyrics("[00:12] a\n") == [(12.0, "a")]


# ── Native / fallback parity ───────────────────────────────────────

_LRC_CASES = [
    "",
    "no timestamps at all\nplain text",
    "[00:12.50] hello",
    "[00:12.5] short frac",
    "[00:12.500] three digits",
    "[1:23] single-digit minute",
    "[99:99] big fields",
    "[00:05] b\n[00:12] a",
    "[00:05] x\n[00:05] y",
    "  [00:12.00]  padded line  ",
    "[00:12][01:00] both stripped",
    "text [00:12] mid-line only",
    "[00:12.50]\u2028[00:20.00] sep",
    "[00:12] a\r\n[00:20] b\r",
    "[٠١:٢٣.٥٠] arabic digits",
    "[00:12.999] max frac",
    "[0:00] zero",
    "[]:] junk",
    "[12:3] one-digit second",
    "[123:45] three-digit minute",
    "[00:12]   ",
    "[00:12.05] hundredths",
    "multiple\nlines\n[01:02.005]  padded  \nno match [02:00] here",
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.sidebars.lyrics_sidebar import _parse_synced_lyrics

cases = json.load(sys.stdin)
out = [_parse_synced_lyrics(raw) for raw in cases]
json.dump(out, sys.stdout)
"""


def _native_results(cases: list) -> list:
    """Run whichever engine the module loaded (native when built) over *cases*."""
    return [_parse_synced_lyrics(raw) for raw in cases]


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
    """When the Rust extension is built, the native LRC parser must
    produce exactly the same (timestamp, text) pairs as the pure-Python
    reference for every case — same boundaries, same fractions, same
    Unicode-digit handling, same stable sort — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = json.loads(json.dumps(_native_results(_LRC_CASES)))
        fallback = _fallback_results(_LRC_CASES)
        for i, case in enumerate(_LRC_CASES):
            assert native[i] == fallback[i], (
                f"native and fallback parse results differ for case {i}: {case!r}"
            )
