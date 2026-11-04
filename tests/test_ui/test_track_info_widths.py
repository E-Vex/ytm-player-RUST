"""Tests for the playback-bar track-info width-splitting engine.

The engine behind ``_TrackInfo.render``: the halves-and-thirds width
split behind the one-line title/artist/album render — the
``max(10, bar_width - 30)`` budget, the ``min(len, //2)`` and
``min(len, //3)`` fragment widths, and the remainder-minus-separators
album width with its ``max(0, ...)`` clamp. The pure-Python reference
lives in ``ui/playback_bar.py`` (``_track_info_widths``); when the Rust
extension is built the same engine runs natively
(``_ytm_player_native.track_info_widths``).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.playback_bar import _track_info_widths

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _reference_widths(bar_width: int, title: str, artist: str) -> tuple[int, int, int]:
    """The pre-engine arithmetic, kept here as the oracle: exactly the
    lines the render used to inline."""
    max_w = max(10, bar_width - 30)
    title_w = min(len(title), max_w // 2)
    artist_w = min(len(artist), max_w // 3)
    album_w = max(0, max_w - title_w - artist_w - 8)
    return title_w, artist_w, album_w


# ── The budget floor ──────────────────────────────────────────────


class TestBudgetFloor:
    def test_zero_width_floors_at_ten(self):
        # max(10, 0 - 30) = 10 → halves 5, thirds 3, album 10-5-3-8 < 0
        # → clamped to 0.
        assert _track_info_widths(0, "Title", "Artist") == (5, 3, 0)

    def test_narrow_terminals_all_floor_at_ten(self):
        assert _track_info_widths(1, "Title", "Artist") == (5, 3, 0)
        assert _track_info_widths(20, "Title", "Artist") == (5, 3, 0)
        assert _track_info_widths(39, "Title", "Artist") == (5, 3, 0)
        assert _track_info_widths(40, "Title", "Artist") == (5, 3, 0)

    def test_budget_starts_opening_at_41(self):
        # 41 - 30 = 11 → title min(5, 5), artist min(6, 3), album
        # 11 - 5 - 3 - 8 < 0 → 0.
        assert _track_info_widths(41, "Title", "Artist") == (5, 3, 0)

    def test_empty_strings_floor_to_zero_fragment_widths(self):
        # No title/artist to spend the floored-10 budget on — the
        # album still gets the 2 that remain of it.
        assert _track_info_widths(0, "", "") == (0, 0, 2)

    def test_widths_match_the_reference_arithmetic(self):
        for bar_width in (0, 1, 25, 40, 41, 50, 60, 80, 100, 120, 200, 500):
            for title, artist in (
                ("", ""),
                ("T", ""),
                ("", "A"),
                ("Title", "Artist"),
                ("A much longer title here", "Some artist name"),
                ("日本語のタイトル", "アーティスト"),
            ):
                assert _track_info_widths(bar_width, title, artist) == _reference_widths(
                    bar_width, title, artist
                )


# ── The split on a real budget ────────────────────────────────────


class TestSplit:
    def test_wide_bar_splits_halves_and_thirds(self):
        # max_w = 90 → title 45, artist 30, album 90-45-30-8 = 7.
        assert _track_info_widths(120, "x" * 100, "y" * 100) == (45, 30, 7)

    def test_short_fragments_take_their_own_length(self):
        # min(len, budget): short strings are never padded up.
        assert _track_info_widths(120, "T", "A") == (1, 1, 80)

    def test_album_takes_the_remainder_minus_separators(self):
        # The two em-dash separators (3 columns each) and the icon's
        # other 2 columns are the 8 the album loses.
        assert _track_info_widths(120, "Title", "Artist") == (5, 6, 71)

    def test_album_clamps_at_zero(self):
        # max_w = 20, title 10, artist 6 → 20-10-6-8 = -4 → 0.
        assert _track_info_widths(50, "x" * 100, "y" * 100) == (10, 6, 0)

    def test_lens_are_code_points(self):
        # CJK titles count code points exactly like len() — three
        # characters are three columns to the split (the terminal
        # double-width rendering is the Rich render's business, not the
        # split's — the reference never measured display width).
        assert _track_info_widths(120, "日本語", "émoji") == (3, 5, 74)

    def test_astral_and_combining_characters_count_as_one(self):
        # 🎧 is one code point; e + combining acute is two.
        assert _track_info_widths(120, "🎧", "é") == (1, 2, 79)

    def test_short_title_long_artist(self):
        # The artist's thirds cap is independent of the title's halves.
        assert _track_info_widths(120, "T", "y" * 100) == (1, 30, 51)


# ── The render through the engine ──────────────────────────────────


class TestRender:
    async def test_mounted_render_truncates_through_the_engine(self):
        from textual.app import App
        from textual.widgets import Static

        from ytm_player.ui.playback_bar import _TrackInfo

        class _Host(App):
            def compose(self):
                yield _TrackInfo(id="pb-track-info")
                yield Static(id="sizer")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            info = app.query_one("#pb-track-info", _TrackInfo)
            info.title = "A very long title that will not fit the bar"
            info.artist = "A very long artist name that will not fit either"
            info.album = "An album name"
            info.is_playing = True
            await pilot.pause()

            title_w, artist_w, album_w = _track_info_widths(60, info.title, info.artist)
            render = info.render()
            # The plain text of the render carries the truncated
            # fragments (before isolate_bidi's markers).
            plain = render.plain
            assert "A very long " in plain
            # The engine's word is what the render truncated through.
            from ytm_player.utils.formatting import truncate

            assert truncate(info.title, title_w) in plain
            assert truncate(info.artist, artist_w) in plain
            assert truncate(info.album, album_w) in plain

    async def test_mounted_render_narrow_terminal_clamps_album(self):
        from textual.app import App

        from ytm_player.ui.playback_bar import _TrackInfo

        class _Host(App):
            def compose(self):
                yield _TrackInfo(id="pb-track-info")

        app = _Host()
        async with app.run_test(size=(30, 10)) as pilot:
            info = app.query_one("#pb-track-info", _TrackInfo)
            info.title = "Title"
            info.artist = "Artist"
            info.album = "Album"
            await pilot.pause()

            _, _, album_w = _track_info_widths(30, "Title", "Artist")
            assert album_w == 0
            render = info.render()
            # Zero album width — truncate to nothing, so the album
            # itself never renders.
            assert "Album" not in render.plain


# ── Native parity ──────────────────────────────────────────────────

# The child interpreter runs this program with the case list on stdin
# and prints the reference engine's results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.playback_bar import _track_info_widths

cases = json.load(sys.stdin)
out = [_track_info_widths(w, t, a) for w, t, a in cases]
json.dump(out, sys.stdout)
"""

# (bar_width, title, artist) cases the two implementations must agree
# on: the floor (0 / 1 / 20 / 39 / 40), the opening budget (41+), the
# halves/thirds/remainder split on a wide bar, the short fragments, the
# code-point lens (CJK, astral, combining), the empty strings, and the
# very wide terminal.
_PARITY_CASES = [
    (0, "Title", "Artist"),
    (1, "Title", "Artist"),
    (20, "Title", "Artist"),
    (39, "Title", "Artist"),
    (40, "Title", "Artist"),
    (41, "Title", "Artist"),
    (50, "x" * 100, "y" * 100),
    (60, "A very long title that will not fit", "A very long artist name"),
    (80, "x" * 100, "y" * 100),
    (100, "", ""),
    (100, "T", ""),
    (100, "", "A"),
    (100, "T", "A"),
    (120, "Title", "Artist"),
    (120, "日本語のタイトル", "アーティスト"),
    (120, "🎧🎧🎧", "ééé"),
    (120, "x" * 100, "y" * 100),
    (200, "A much longer title here", "Some artist name"),
    (500, "x" * 400, "y" * 400),
]


def _native_results(cases: list) -> list:
    """Run whichever engine the module loaded (native when built) over
    *cases*."""
    return [_track_info_widths(w, t, a) for w, t, a in cases]


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
    return [tuple(result) for result in json.loads(out.stdout)]


class TestNativeParity:
    """When the Rust extension is built, the native width engine must
    answer every case exactly like the pure-Python reference run in a
    child interpreter with the extension import blocked — the budget
    floor, the clamps and the code-point lens included — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _native_results(_PARITY_CASES)
        fallback = _fallback_results(_PARITY_CASES)
        for case, native_result, fallback_result in zip(_PARITY_CASES, native, fallback):
            assert tuple(native_result) == fallback_result, (
                f"native and fallback track-info widths differ for case {case}"
            )
