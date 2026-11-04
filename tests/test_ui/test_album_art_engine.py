"""Tests for the album-art render engines.

The engines behind ``AlbumArt``: the WCAG colour math of the
placeholder's note glyph (``_relative_luminance``,
``_contrast_ratio``, ``_readable_on``), the half-block cell extraction
from the resized thumbnail's raw RGB buffer (``_half_block_cells``)
and the two no-image row plans (``_placeholder_rows`` and
``_empty_rows``). The pure-Python references live in
``ui/widgets/album_art.py``; when the Rust extension is built the same
engines run natively (``_ytm_player_native.wcag_relative_luminance`` &
co).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from io import BytesIO

import pytest
from PIL import Image
from rich.color import Color
from rich.style import Style
from rich.text import Text

from ytm_player.ui.widgets.album_art import (
    _BLOCK_BOTTOM,
    _BLOCK_FULL,
    _BLOCK_TOP,
    _NOTE,
    AlbumArt,
    _contrast_ratio,
    _empty_rows,
    _half_block_cells,
    _placeholder_rows,
    _readable_on,
    _relative_luminance,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── _relative_luminance ───────────────────────────────────────────


class TestRelativeLuminance:
    def test_white_is_one(self):
        assert _relative_luminance("#ffffff") == 1.0

    def test_black_is_zero(self):
        assert _relative_luminance("#000000") == 0.0

    def test_pure_red_is_the_red_weight(self):
        # Only the red coefficient survives: 0.2126 * 1.0.
        assert _relative_luminance("#ff0000") == pytest.approx(0.2126)

    def test_pure_green_and_blue_weights(self):
        assert _relative_luminance("#00ff00") == pytest.approx(0.7152)
        assert _relative_luminance("#0000ff") == pytest.approx(0.0722)

    def test_mid_grey_linearises_below_the_channels(self):
        # #808080 has a relative luminance of about 0.216 once sRGB is
        # linearised.
        assert _relative_luminance("#808080") == pytest.approx(0.21586050011389923)

    def test_hash_is_optional(self):
        assert _relative_luminance("fff59b") == _relative_luminance("#fff59b")

    def test_every_leading_hash_is_stripped(self):
        # lstrip("#") strips them all, so ##rrggbb parses too.
        assert _relative_luminance("##fff59b") == _relative_luminance("#fff59b")

    def test_surrounding_whitespace_is_stripped(self):
        assert _relative_luminance("  #fff59b \n") == _relative_luminance("#fff59b")

    @pytest.mark.parametrize(
        "color",
        ["not-a-color", "ansi_default", "#fff", "#12345", "", "red", "#12345g", "日本語日本"],
    )
    def test_non_six_digit_hex_is_none(self, color):
        assert _relative_luminance(color) is None

    def test_non_hex_digits_are_none(self):
        assert _relative_luminance("#zzz000") is None

    def test_whitespace_inside_a_channel_pair_parses_like_int(self):
        # int("1 ", 16) == 1 in CPython, so the pair "1 " is channel 1 —
        # the colour parses with channels (1, 0x23, 0x45) on both
        # backends rather than being rejected.
        channels = [int("1 ", 16) / 255, int("23", 16) / 255, int("45", 16) / 255]
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
        expected = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
        assert _relative_luminance("#1 2345") == pytest.approx(expected)
        assert _relative_luminance("#1 2345") is not None


# ── _contrast_ratio ───────────────────────────────────────────────


class TestContrastRatio:
    def test_black_on_white_is_twenty_one(self):
        assert _contrast_ratio(1.0, 0.0) == 21.0

    def test_symmetric(self):
        assert _contrast_ratio(0.2126, 0.0) == _contrast_ratio(0.0, 0.2126)

    def test_equal_luminances_are_one(self):
        assert _contrast_ratio(0.5, 0.5) == 1.0

    def test_red_on_black_is_5_25(self):
        # (0.2126 + 0.05) / 0.05.
        assert _contrast_ratio(0.2126, 0.0) == pytest.approx(5.252)


# ── _readable_on ──────────────────────────────────────────────────


class TestReadableOnEngine:
    def test_light_background_picks_dark(self):
        assert _readable_on("#fff59b", "#f3edf7", "#070722") == "#070722"

    def test_dark_background_picks_light(self):
        assert _readable_on("#1e3a5f", "#ffffff", "#0f0f0f") == "#ffffff"

    def test_pure_red_picks_black_by_contrast_ratio(self):
        assert _readable_on("#ff0000", "#ffffff", "#000000") == "#000000"

    def test_mid_grey_picks_black(self):
        assert _readable_on("#808080", "#ffffff", "#000000") == "#000000"

    def test_tie_keeps_the_first_candidate(self):
        assert _readable_on("#ffffff", "000000", "#000000") == "000000"

    def test_unparseable_background_keeps_the_first_candidate(self):
        assert _readable_on("not-a-color", "#ffffff", "#000000") == "#ffffff"

    def test_unparseable_candidate_keeps_the_first_candidate(self):
        assert _readable_on("#fff59b", "ansi_default", "#000000") == "ansi_default"
        assert _readable_on("#fff59b", "#ffffff", "ansi_default") == "#ffffff"

    def test_single_candidate_round_trips(self):
        assert _readable_on("#000000", "#ffffff") == "#ffffff"


# ── _half_block_cells ─────────────────────────────────────────────


class TestHalfBlockCells:
    def test_single_cell(self):
        # One column, one character row: the top pixel then the bottom.
        assert _half_block_cells(bytes([1, 2, 3, 4, 5, 6]), 1, 1) == [(1, 2, 3, 4, 5, 6)]

    def test_two_by_two_grid_is_row_major(self):
        # pixel_w = 2, pixel_h = 4 → 24 bytes. Cell (row, col) joins
        # pixels (col, 2*row) and (col, 2*row + 1).
        pixels = bytes(range(24))
        assert _half_block_cells(pixels, 2, 2) == [
            (0, 1, 2, 6, 7, 8),  # row 0, col 0
            (3, 4, 5, 9, 10, 11),  # row 0, col 1
            (12, 13, 14, 18, 19, 20),  # row 1, col 0
            (15, 16, 17, 21, 22, 23),  # row 1, col 1
        ]

    def test_wide_single_row(self):
        # One character row of three cells: top pixels are the first
        # three, bottom pixels the next three.
        pixels = bytes([10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62])
        assert _half_block_cells(pixels, 3, 1) == [
            (10, 11, 12, 40, 41, 42),
            (20, 21, 22, 50, 51, 52),
            (30, 31, 32, 60, 61, 62),
        ]

    @pytest.mark.parametrize(
        ("length", "width", "height"),
        [(5, 1, 1), (0, 1, 1), (24, 2, 3), (6, 2, 1), (18, 4, 1)],
    )
    def test_wrong_size_buffer_raises(self, length, width, height):
        with pytest.raises(ValueError):
            _half_block_cells(bytes(length), width, height)

    def test_zero_area_is_no_cells(self):
        assert _half_block_cells(b"", 0, 0) == []
        assert _half_block_cells(b"", 0, 5) == []
        assert _half_block_cells(b"", 5, 0) == []

    def test_non_positive_dimensions_with_matching_buffer_are_no_cells(self):
        # A zero-length buffer "matches" the degenerate products (any
        # zero dimension); the walks stay empty exactly like range().
        assert _half_block_cells(b"", -1, 0) == []


# ── _placeholder_rows ─────────────────────────────────────────────


class TestPlaceholderRows:
    def test_five_by_five_box(self):
        rows = _placeholder_rows(5, 5, "#ff0000", "#1a1a1a", "#ffffff")
        assert rows == [
            [(_BLOCK_BOTTOM * 5, "#ff0000 on #1a1a1a")],
            [(_BLOCK_FULL * 5, "#ff0000")],
            [
                (_BLOCK_FULL * 2, "#ff0000"),
                (_NOTE, "bold #ffffff on #ff0000"),
                (_BLOCK_FULL * 2, "#ff0000"),
            ],
            [(_BLOCK_FULL * 5, "#ff0000")],
            [(_BLOCK_TOP * 5, "#ff0000 on #1a1a1a")],
        ]

    def test_one_row_box_is_all_lower_half_blocks(self):
        # Row 0 wins the first-row check before the last-row one.
        assert _placeholder_rows(4, 1, "a", "b", "c") == [
            [(_BLOCK_BOTTOM * 4, "a on b")],
        ]

    def test_two_row_box_never_shows_a_middle(self):
        assert _placeholder_rows(4, 2, "a", "b", "c") == [
            [(_BLOCK_BOTTOM * 4, "a on b")],
            [(_BLOCK_TOP * 4, "a on b")],
        ]

    def test_three_row_box_middle_is_the_note_row(self):
        assert _placeholder_rows(4, 3, "a", "b", "c") == [
            [(_BLOCK_BOTTOM * 4, "a on b")],
            [
                (_BLOCK_FULL * 1, "a"),
                (_NOTE, "bold c on a"),
                (_BLOCK_FULL * 2, "a"),
            ],
            [(_BLOCK_TOP * 4, "a on b")],
        ]

    def test_odd_width_pads_left(self):
        # (width - 1) // 2 floors left: 5 → 2 left, 2 right.
        _, middle, _ = _placeholder_rows(5, 3, "a", "b", "c")
        assert middle == [
            (_BLOCK_FULL * 2, "a"),
            (_NOTE, "bold c on a"),
            (_BLOCK_FULL * 2, "a"),
        ]

    def test_width_one_middle_row_is_just_the_note(self):
        _, middle, _ = _placeholder_rows(1, 3, "a", "b", "c")
        assert middle == [
            ("", "a"),
            (_NOTE, "bold c on a"),
            ("", "a"),
        ]

    def test_width_zero_middle_row_floor_divides_negative(self):
        # pad = (0 - 1) // 2 == -1 → the empty string, like Python's
        # negative repeat; the right side is 0 - (-1) - 1 == 0 → also
        # empty. The note glyph alone survives.
        _, middle, _ = _placeholder_rows(0, 3, "a", "b", "c")
        assert middle == [
            ("", "a"),
            (_NOTE, "bold c on a"),
            ("", "a"),
        ]

    def test_non_positive_height_is_no_rows(self):
        assert _placeholder_rows(5, 0, "a", "b", "c") == []
        assert _placeholder_rows(5, -2, "a", "b", "c") == []


# ── _empty_rows ───────────────────────────────────────────────────


class TestEmptyRows:
    def test_three_by_five_note_in_the_middle(self):
        rows = _empty_rows(5, 3, "#333333")
        assert rows == [
            [(" " * 5, "#333333")],
            [
                (" " * 2, "#333333"),
                (_NOTE, "#333333"),
                (" " * 2, "#333333"),
            ],
            [(" " * 5, "#333333")],
        ]

    def test_one_row_box_is_the_note_row(self):
        # height // 2 == 0 for h == 1.
        assert _empty_rows(5, 1, "#333333") == [
            [
                (" " * 2, "#333333"),
                (_NOTE, "#333333"),
                (" " * 2, "#333333"),
            ],
        ]

    def test_width_one_is_just_the_note(self):
        rows = _empty_rows(1, 3, "m")
        assert rows == [
            [(" ", "m")],
            [("", "m"), (_NOTE, "m"), ("", "m")],
            [(" ", "m")],
        ]

    def test_non_positive_height_is_no_rows(self):
        assert _empty_rows(5, 0, "m") == []
        assert _empty_rows(5, -1, "m") == []


# ── _image_to_half_blocks (the buffer-walk refactor) ──────────────


def _reference_half_blocks(img_bytes: bytes, width: int, height: int) -> Text:
    """The pre-engine getpixel walk, kept as the equivalence oracle."""
    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    pixel_w = width
    pixel_h = height * 2
    img = img.resize((pixel_w, pixel_h), Image.Resampling.LANCZOS)

    result = Text()
    for row in range(height):
        if row > 0:
            result.append("\n")
        top_y = row * 2
        bot_y = row * 2 + 1
        for col in range(pixel_w):
            tr, tg, tb = img.getpixel((col, top_y))
            br, bg, bb = img.getpixel((col, bot_y))
            style = Style(
                color=Color.from_rgb(tr, tg, tb),
                bgcolor=Color.from_rgb(br, bg, bb),
            )
            result.append("\u2580", style=style)
    return result


def _cell_styles(text: Text) -> list[tuple[int, int, str]]:
    return [(span.start, span.end, str(span.style)) for span in text.spans]


def _gradient_png(width: int, height: int) -> bytes:
    img = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), (x * 8 % 256, y * 16 % 256, (x * y * 3) % 256))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestImageToHalfBlocks:
    @pytest.mark.parametrize(
        ("width", "height"),
        [(1, 1), (3, 2), (10, 3), (12, 6), (5, 1)],
    )
    def test_matches_the_getpixel_walk(self, width, height):
        img_bytes = _gradient_png(40, 30)

        result = AlbumArt._image_to_half_blocks(img_bytes, width, height)
        reference = _reference_half_blocks(img_bytes, width, height)

        assert result.plain == reference.plain
        assert _cell_styles(result) == _cell_styles(reference)

    def test_renders_half_block_glyphs(self):
        result = AlbumArt._image_to_half_blocks(_gradient_png(8, 8), 3, 2)
        assert result.plain == "\u2580\u2580\u2580\n\u2580\u2580\u2580"

    def test_one_by_one_has_a_single_styled_span(self):
        result = AlbumArt._image_to_half_blocks(_gradient_png(8, 8), 1, 1)
        assert result.plain == "\u2580"
        assert len(result.spans) == 1
        span = result.spans[0]
        assert (span.start, span.end) == (0, 1)
        # The cell's foreground is the top pixel row, the background the
        # bottom one.
        assert span.style.color is not None
        assert span.style.bgcolor is not None
        assert span.style.color != span.style.bgcolor


# ── Native / fallback parity ──────────────────────────────────────

_LUMINANCE_CASES = [
    "#ffffff",
    "#000000",
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#808080",
    "#fff59b",
    "#1e3a5f",
    "#f3edf7",
    "#070722",
    "fff59b",
    "##fff59b",
    "  #fff59b \n",
    "#1 2345",
    "#zzz000",
    "#12345",
    "#fff",
    "",
    "not-a-color",
    "ansi_default",
    "red",
    "#0F0f0F",
    "#-1-1-1",
    "#+1+1+1",
    "日本語日本",
    "# 12345",
    "\t#fff59b\t",
]

_CONTRAST_CASES = [
    (1.0, 0.0),
    (0.0, 1.0),
    (0.2126, 0.0),
    (0.0, 0.2126),
    (0.5, 0.5),
    (0.21586050011389923, 0.0722),
    (0.8893137344423357, 0.01638187637053478),
    (0.0, 0.0),
    (1.0, 1.0),
    (-0.06, 0.2126),
    (0.2126, 0.2126),
    (0.123456789, 0.987654321),
]

_READABLE_CASES = [
    ("#fff59b", ["#f3edf7", "#070722"]),
    ("#1e3a5f", ["#ffffff", "#0f0f0f"]),
    ("#ff0000", ["#ffffff", "#000000"]),
    ("#808080", ["#ffffff", "#000000"]),
    ("#808080", ["#000000", "#ffffff"]),
    ("#ffffff", ["000000", "#000000"]),
    ("fff59b", ["#f3edf7", "#070722"]),
    ("not-a-color", ["#ffffff", "#000000"]),
    ("#fff59b", ["ansi_default", "#000000"]),
    ("#fff59b", ["#ffffff", "ansi_default"]),
    ("#000000", ["#ffffff"]),
    ("#808080", ["#ffffff", "#000000", "#f3edf7", "red"]),
    ("#-1-1-1", ["#ffffff", "#000000"]),
]

_CELLS_CASES = [
    ([1, 2, 3, 4, 5, 6], 1, 1),
    (list(range(24)), 2, 2),
    (list(range(18)), 3, 1),
    (list(range(54)), 3, 3),
    (list(range(120)), 4, 5),
    ([], 0, 0),
    ([], 0, 5),
    ([], 5, 0),
    ([], -1, -3),
    ([], 0, -5),
    (list(range(6)), 1, 1),
    ([0, 0, 0, 255, 255, 255], 1, 1),
    # Wrong-size buffers: both backends must refuse them.
    (list(range(5)), 1, 1),
    ([], 1, 1),
    (list(range(24)), 2, 3),
    (list(range(7)), 1, 1),
]

_PLACEHOLDER_CASES = [
    (5, 5, "#ff0000", "#1a1a1a", "#ffffff"),
    (4, 1, "a", "b", "c"),
    (4, 2, "a", "b", "c"),
    (4, 3, "a", "b", "c"),
    (1, 3, "a", "b", "c"),
    (0, 3, "a", "b", "c"),
    (5, 0, "a", "b", "c"),
    (5, -2, "a", "b", "c"),
    (7, 4, "accent", "bg", "note"),
    (2, 7, "x", "y", "z"),
    (0, 1, "a", "b", "c"),
    (1, 1, "a", "b", "c"),
    (0, 0, "a", "b", "c"),
    (6, 9, "#ff0000", "#1a1a1a", "#070722"),
]

_EMPTY_CASES = [
    (5, 3, "#333333"),
    (5, 1, "m"),
    (1, 3, "m"),
    (1, 1, "m"),
    (0, 3, "m"),
    (0, 1, "m"),
    (5, 0, "m"),
    (5, -1, "m"),
    (7, 4, "muted"),
    (0, 0, "m"),
]

_CASES = {
    "luminance": _LUMINANCE_CASES,
    "contrast": _CONTRAST_CASES,
    "readable": _READABLE_CASES,
    "cells": _CELLS_CASES,
    "placeholder": _PLACEHOLDER_CASES,
    "empty": _EMPTY_CASES,
}

# The child interpreter runs this program with the case list on stdin and
# prints the reference engines' results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets.album_art import (
    _contrast_ratio,
    _empty_rows,
    _half_block_cells,
    _placeholder_rows,
    _readable_on,
    _relative_luminance,
)

cases = json.load(sys.stdin)


def run(fn, *args):
    try:
        return {"ok": fn(*args)}
    except Exception as exc:
        return {"error": type(exc).__name__}


out = {
    "luminance": [run(_relative_luminance, c) for c in cases["luminance"]],
    "contrast": [run(_contrast_ratio, a, b) for a, b in cases["contrast"]],
    "readable": [run(_readable_on, bg, *cands) for bg, cands in cases["readable"]],
    "cells": [run(_half_block_cells, bytes(px), w, h) for px, w, h in cases["cells"]],
    "placeholder": [
        run(_placeholder_rows, w, h, a, b, c) for w, h, a, b, c in cases["placeholder"]
    ],
    "empty": [run(_empty_rows, w, h, m) for w, h, m in cases["empty"]],
}
json.dump(out, sys.stdout)
"""


def _native_results(cases: dict) -> dict:
    """Run whichever engines the module loaded (native when built) over *cases*."""

    def run(fn, *args):
        try:
            return {"ok": fn(*args)}
        except Exception as exc:
            return {"error": type(exc).__name__}

    return {
        "luminance": [run(_relative_luminance, c) for c in cases["luminance"]],
        "contrast": [run(_contrast_ratio, a, b) for a, b in cases["contrast"]],
        "readable": [run(_readable_on, bg, *cands) for bg, cands in cases["readable"]],
        "cells": [run(_half_block_cells, bytes(px), w, h) for px, w, h in cases["cells"]],
        "placeholder": [
            run(_placeholder_rows, w, h, a, b, c) for w, h, a, b, c in cases["placeholder"]
        ],
        "empty": [run(_empty_rows, w, h, m) for w, h, m in cases["empty"]],
    }


def _fallback_results(cases: dict) -> dict:
    """Run the pure-Python reference engines over *cases* in a child
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


def _jsonify(value):
    """Normalise a result the way the child's JSON round-trip does."""
    return json.loads(json.dumps(value))


class TestNativeParity:
    """When the Rust extension is built, the native album-art render
    engines must produce exactly the same luminances, ratios, colour
    picks, cell tuples and row plans as the pure-Python references for
    every case — including the same ValueError refusals — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _jsonify(_native_results(_CASES))
        fallback = _fallback_results(_CASES)
        for group, cases in _CASES.items():
            for i in range(len(cases)):
                assert native[group][i] == fallback[group][i], (
                    f"native and fallback {group} results differ for case {i}: {cases[i]}"
                )
