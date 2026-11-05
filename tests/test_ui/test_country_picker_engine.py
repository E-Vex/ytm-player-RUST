"""Tests for the chart-region picker's filter engine.

``filter_regions`` (ui/popups/country_picker.py) runs with two
interchangeable backends: the pure-Python transcription in the module
and, when the Rust extension is built, the native implementation bound
at the module's tail. The engine: the query stripped and lowercased, an
empty query handing the region list back unchanged (the very same
object), and otherwise the rows whose ISO code or display name contains
the query case-insensitively, re-packed as fresh pairs of the same
string objects.

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case —
the for-unpack ValueError shapes included — so the two implementations
cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.services.regions import CHART_REGIONS
from ytm_player.ui.popups.country_picker import filter_regions

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

REGIONS = (
    ("ZZ", "Global"),
    ("GB", "United Kingdom"),
    ("US", "United States"),
    ("AE", "United Arab Emirates"),
    ("KR", "South Korea"),
)


class TestFilterRegions:
    """The region filter (whichever backend)."""

    def test_empty_query_returns_the_same_object(self):
        assert filter_regions(REGIONS, "") is REGIONS

    def test_whitespace_query_returns_the_same_object(self):
        assert filter_regions(REGIONS, "   \t ") is REGIONS

    def test_matches_on_code(self):
        assert filter_regions(REGIONS, "gb") == (("GB", "United Kingdom"),)

    def test_matches_on_name(self):
        assert filter_regions(REGIONS, "united") == (
            ("GB", "United Kingdom"),
            ("US", "United States"),
            ("AE", "United Arab Emirates"),
        )

    def test_match_is_case_insensitive_substring_on_either_field(self):
        assert filter_regions(REGIONS, "KINGDOM") == (("GB", "United Kingdom"),)
        assert filter_regions(REGIONS, "korea") == (("KR", "South Korea"),)
        assert filter_regions(REGIONS, "z") == (("ZZ", "Global"),)

    def test_query_is_stripped(self):
        assert filter_regions(REGIONS, "  global ") == (("ZZ", "Global"),)

    def test_no_match_is_the_empty_tuple(self):
        assert filter_regions(REGIONS, "atlantis") == ()

    def test_kept_rows_are_fresh_pairs_of_the_same_strings(self):
        result = filter_regions(REGIONS, "global")
        (row,) = result
        assert row == ("ZZ", "Global")
        assert row[0] is REGIONS[0][0]
        assert row[1] is REGIONS[0][1]
        assert row is not REGIONS[0]

    def test_every_chart_region_finds_itself(self):
        for code, name in CHART_REGIONS:
            assert (code, name) in filter_regions(CHART_REGIONS, code.lower())
            assert (code, name) in filter_regions(CHART_REGIONS, name.lower())

    def test_malformed_row_raises_the_unpack_error(self):
        with pytest.raises(ValueError, match="unpack"):
            filter_regions((("US", "United States", "extra"),), "kingdom")
        with pytest.raises(ValueError, match="unpack"):
            filter_regions((("US",),), "kingdom")


# ── Total engine (native only) ────────────────────────────────────────


class TestTotalityEngine:
    """The native engine stays total where the reference would raise.

    A non-string code or name raises AttributeError in the reference's
    ``.lower()``; the native engine degrades instead — the convention the
    ported engines before it follow (``filter_panel_items``) — so these
    tests only run with the extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_string_fields_never_match(self):
        # A query only the non-string field could have matched: the row
        # falls to the other field, which doesn't contain it either.
        assert filter_regions(((7, "United States"),), "7") == ()
        assert filter_regions((("US", 7),), "united") == ()
        assert filter_regions(((7, "United States"), ("US", 7)), "7") == ()

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_string_fields_flow_through_the_empty_query(self):
        regions = ((7, "United States"), ("US", 7))
        # The empty query hands the regions object back untouched, so it
        # never reads a field — no error on either backend.
        assert filter_regions(regions, "") is regions


# ── Native vs fallback parity ─────────────────────────────────────────

# Each parity case: [rows, query] where rows is a list of [code, name].
# Shapes where the pure-Python reference would raise outside the unpack
# contract (non-string fields hitting ``.lower()``) stay out and are
# pinned by TestTotalityEngine above; the for-unpack ValueErrors stay in
# — both backends raise them with the interpreter's own message.
_PARITY_CASES = [
    [[["ZZ", "Global"]], ""],
    [[["ZZ", "Global"]], "   "],
    [
        [["ZZ", "Global"], ["GB", "United Kingdom"], ["US", "United States"]],
        "united",
    ],
    [[["ZZ", "Global"], ["GB", "United Kingdom"]], "GB"],
    [[["ZZ", "Global"], ["GB", "United Kingdom"]], " gb "],
    [[["US", "United States"], ["ES", "Spain"]], "z"],
    [[["US", "United States"]], "Ünïcode"],
    [[["US", "United States"], ["DE", "Germany"]], "germany"],
    # Unicode case mapping: 'İ' lowercases to a two-code-point sequence,
    # sigma and eszett fold per code point on both sides.
    [[["TR", "Türkiye"]], "türk"],
    [[["TR", "Türkiye"]], "TÜRK"],
    [[["GR", "ΣAB"], ["DE", "straße"]], "σab"],
    [[["GR", "ΣAB"], ["DE", "straße"]], "STRASSE"],
    # A malformed row reaches the for-unpack contract on both backends.
    [[["US", "United States", "extra"], ["GB", "United Kingdom"]], "kingdom"],
    [[["US", "United States"], ["GB"]], "kingdom"],
    [[["US", "United States"], ["GB", "United Kingdom"]], "KINGDOM"],
    [[["US", "United States"], ["GB", "United Kingdom"]], "united states of atlantis"],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.country_picker import filter_regions


def outcome(rows, query):
    regions = tuple(tuple(row) for row in rows)
    try:
        return {"ok": [list(map(str, filter_regions(regions, query)))]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
out = [outcome(*case) for case in payload]
json.dump(out, sys.stdout)
"""


def _native_outcome(rows, query):
    try:
        regions = tuple(tuple(row) for row in rows)
        return {"ok": [list(map(str, filter_regions(regions, query)))]}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same accepts, same values,
    same errors with the same messages — for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        payload = json.dumps(_PARITY_CASES)
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        assert len(fallback) == len(_PARITY_CASES)
        for (rows, query), expected in zip(_PARITY_CASES, fallback):
            native = _native_outcome(rows, query)
            assert native == expected, (
                f"native and fallback filter_regions differ for {rows!r} / {query!r}:\n"
                f"native:   {native!r}\n"
                f"fallback: {expected!r}"
            )
