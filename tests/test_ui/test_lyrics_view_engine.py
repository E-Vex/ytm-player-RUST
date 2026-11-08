"""Tests for the lyrics-view row engine (per-line render composition).

``_compose_lyric_rows`` is the engine behind ``LyricsSidebar``'s two view
builders: every lyric line's ``(display, is_rtl, translit)`` row — the
BiDi render (gated wrap, then FSI...PDI isolate), the RTL classification
and the transliteration — composed for the whole list in one call. The
pure-Python reference lives in ``ui/sidebars/lyrics_sidebar.py``; when
the Rust extension is built the same engine runs natively
(``_ytm_player_native.lyrics_view_rows``).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart. The wiring tests mount the
sidebar itself and check the rows land in the view the builders
assemble from them.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.sidebars.lyrics_sidebar import _compose_lyric_rows

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

FSI = "\u2068"  # FIRST STRONG ISOLATE
PDI = "\u2069"  # POP DIRECTIONAL ISOLATE


# ── Row composition ─────────────────────────────────────────────────


class TestRowComposition:
    def test_ltr_line_passes_through_unwrapped(self):
        # Pure-LTR lines render as-is: no wrap, no isolate, no flags.
        assert _compose_lyric_rows(["hello world"], 35, False) == [("hello world", False, None)]

    def test_rows_keep_input_order(self):
        lines = ["first", "second", "third"]
        rows = _compose_lyric_rows(lines, 35, False)
        assert [row[0] for row in rows] == lines

    def test_empty_line_is_a_blank_row(self):
        assert _compose_lyric_rows([""], 35, True) == [("", False, None)]

    def test_whitespace_only_line_passes_through(self):
        assert _compose_lyric_rows(["   "], 35, True) == [("   ", False, None)]

    def test_empty_list_answers_empty(self):
        assert _compose_lyric_rows([], 35, False) == []
        assert _compose_lyric_rows([], 35, True) == []

    def test_tuple_sequence_is_accepted(self):
        # The reference iterates any sequence; the engine's contract is
        # the same.
        assert _compose_lyric_rows(("hello", "world"), 35, False) == [
            ("hello", False, None),
            ("world", False, None),
        ]

    def test_every_row_is_a_three_item_triple(self):
        for row in _compose_lyric_rows(["a", "", "שלום", "東京"], 35, True):
            assert isinstance(row, tuple)
            assert len(row) == 3
            assert isinstance(row[0], str)
            assert isinstance(row[1], bool)
            assert row[2] is None or isinstance(row[2], str)

    def test_input_list_is_not_mutated(self):
        lines = ["hello", "שלום", "東京", ""]
        snapshot = list(lines)
        _compose_lyric_rows(lines, 35, True)
        assert lines == snapshot


# ── RTL classification ──────────────────────────────────────────────


class TestRtlClassification:
    def test_rtl_line_is_isolated_and_flagged(self):
        ((display, is_rtl, translit),) = _compose_lyric_rows(
            ["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"], 35, False
        )
        assert is_rtl is True
        assert display == f"{FSI}\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd{PDI}"

    def test_mixed_direction_line_is_flagged_and_isolated(self):
        ((display, is_rtl, _),) = _compose_lyric_rows(
            ["mix English \u05e2\u05d1\u05e8\u05d9\u05ea end"], 35, False
        )
        assert is_rtl is True
        assert display == f"{FSI}mix English \u05e2\u05d1\u05e8\u05d9\u05ea end{PDI}"

    def test_ltr_line_is_not_flagged(self):
        ((_, is_rtl, _),) = _compose_lyric_rows(["plain latin text"], 35, False)
        assert is_rtl is False

    def test_empty_line_is_not_flagged(self):
        ((_, is_rtl, _),) = _compose_lyric_rows([""], 35, False)
        assert is_rtl is False

    def test_cjk_line_is_not_rtl(self):
        # The RTL classification is script-based: CJK is not RTL and
        # renders unwrapped/unisolated.
        ((display, is_rtl, _),) = _compose_lyric_rows(["\u6771\u4eac"], 35, False)
        assert is_rtl is False
        assert display == "\u6771\u4eac"

    def test_arabic_line_is_isolated(self):
        ((display, is_rtl, _),) = _compose_lyric_rows(["\u0645\u0631\u062d\u0628\u0627"], 35, False)
        assert is_rtl is True
        assert display.startswith(FSI)
        assert display.endswith(PDI)


# ── Transliteration ─────────────────────────────────────────────────


class TestTransliteration:
    def test_enabled_cjk_line_gets_a_translit(self):
        ((display, is_rtl, translit),) = _compose_lyric_rows(["\u6771\u4eac"], 35, True)
        assert translit == "DongJing"
        # The display keeps the original script; the translit is a
        # separate row slot.
        assert display == "\u6771\u4eac"
        assert is_rtl is False

    def test_disabled_cjk_line_gets_none(self):
        ((_, _, translit),) = _compose_lyric_rows(["\u6771\u4eac"], 35, False)
        assert translit is None

    def test_ascii_line_gets_none_even_when_enabled(self):
        ((_, _, translit),) = _compose_lyric_rows(["hello world"], 35, True)
        assert translit is None

    def test_empty_line_gets_none_even_when_enabled(self):
        ((_, _, translit),) = _compose_lyric_rows([""], 35, True)
        assert translit is None

    def test_rtl_line_gets_both_flag_and_translit(self):
        ((display, is_rtl, translit),) = _compose_lyric_rows(
            ["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"], 35, True
        )
        assert is_rtl is True
        assert display == f"{FSI}\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd{PDI}"
        assert translit == "slvm 'vlm"

    def test_accented_latin_transliterates(self):
        ((_, _, translit),) = _compose_lyric_rows(["caf\u00e9 ni\u00f1o"], 35, True)
        assert translit == "cafe nino"

    def test_mixed_script_line_transliterates_whole(self):
        ((_, _, translit),) = _compose_lyric_rows(["Hello \u4e16\u754c"], 35, True)
        assert translit == "Hello ShiJie"

    def test_translit_is_ascii(self):
        rows = _compose_lyric_rows(
            [
                "\u6771\u4eac",
                "\ub204\uad6c\ub098",
                "\u041c\u043e\u0441\u043a\u0432\u0430",
                "\u0645\u0631\u062d\u0628\u0627",
            ],
            35,
            True,
        )
        for _, _, translit in rows:
            assert translit is not None
            assert translit.isascii()


# ── Wrap and reorder (bidi_mode="reorder") ──────────────────────────


class TestWrapAndReorder:
    def test_reorder_on_reverses_rtl_word_order(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", True)
        ((display, _, _),) = _compose_lyric_rows(
            ["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"], 35, False
        )
        # UAX #9 L2 word-level reversal under an R base direction.
        assert display == f"{FSI}\u05e2\u05d5\u05dc\u05dd \u05e9\u05dc\u05d5\u05dd{PDI}"

    def test_reorder_off_leaves_rtl_word_order_alone(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", False)
        ((display, _, _),) = _compose_lyric_rows(
            ["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"], 35, False
        )
        assert display == f"{FSI}\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd{PDI}"

    def test_reorder_on_wraps_long_rtl_line(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", True)
        # Ten single-letter Hebrew words under width 10: five fit per
        # wrapped line (2k-1 <= 10 through k=5), each line reversed.
        line = "\u05d0 \u05d1 \u05d2 \u05d3 \u05d4 \u05d5 \u05d6 \u05d7 \u05d8 \u05d9"
        ((display, _, _),) = _compose_lyric_rows([line], 10, False)
        assert (
            display
            == f"{FSI}\u05d4 \u05d3 \u05d2 \u05d1 \u05d0\n\u05d9 \u05d8 \u05d7 \u05d6 \u05d5{PDI}"
        )

    def test_reorder_on_zero_width_reorders_without_wrapping(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", True)
        # width <= 0 short-circuits the word walk to the bare line
        # reorder — one line, no newline in the display.
        line = "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd \u05d7\u05d3\u05e9"
        ((display, _, _),) = _compose_lyric_rows([line], 0, False)
        assert "\n" not in display
        assert (
            display
            == f"{FSI}\u05d7\u05d3\u05e9 \u05e2\u05d5\u05dc\u05dd \u05e9\u05dc\u05d5\u05dd{PDI}"
        )

    def test_ltr_line_never_wraps_even_when_reorder_on(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", True)
        line = "a line of latin words far wider than five columns"
        ((display, is_rtl, _),) = _compose_lyric_rows([line], 5, False)
        assert display == line
        assert is_rtl is False

    def test_mixed_line_wraps_with_each_line_reordered(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", True)
        ((display, _, _),) = _compose_lyric_rows(
            ["mix English \u05e2\u05d1\u05e8\u05d9\u05ea end"], 20, False
        )
        # 22 columns over a 20 budget: "end" starts its own line; the
        # first line's lone RTL word reverses in place.
        assert display == f"{FSI}mix English \u05e2\u05d1\u05e8\u05d9\u05ea\nend{PDI}"

    def test_wrap_and_translit_together(self, monkeypatch):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", True)
        line = "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd \u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"
        ((display, is_rtl, translit),) = _compose_lyric_rows([line], 10, True)
        assert "\n" in display
        assert display.startswith(FSI) and display.endswith(PDI)
        assert is_rtl is True
        # The translit is of the ORIGINAL line, not the wrapped render.
        assert translit == "slvm 'vlm slvm 'vlm"


# ── View-builder wiring ─────────────────────────────────────────────


class TestSyncedViewWiring:
    async def test_synced_view_mounts_rows_in_order(self):
        from textual.app import App, ComposeResult

        from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

        class _Host(App):
            """Minimal host providing the theme variables the CSS needs."""

            def get_css_variables(self) -> dict[str, str]:
                variables = super().get_css_variables()
                variables["lyrics-played"] = "#999999"
                variables["lyrics-current"] = "#ff4e45"
                variables["lyrics-upcoming"] = "#cccccc"
                return variables

            def compose(self) -> ComposeResult:
                yield LyricsSidebar()

        app = _Host()
        async with app.run_test(size=(60, 20)):
            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [
                (0.0, "first line"),
                (5.0, "\u05e9\u05dc\u05d5\u05dd"),
                (10.0, "\u6771\u4eac"),
            ]
            sidebar._transliteration_enabled = True

            sidebar._build_synced_view()

            widgets = sidebar._lyric_widgets
            # One lyric widget per synced line, timestamps preserved.
            assert len(widgets) == 3
            assert [w._timestamp for w in widgets] == [0.0, 5.0, 10.0]
            assert [w._text for w in widgets] == [
                "first line",
                f"{FSI}\u05e9\u05dc\u05d5\u05dd{PDI}",
                "\u6771\u4eac",
            ]
            # Every line starts upcoming; only the Hebrew one is RTL.
            assert all(w.has_class("lyrics-upcoming") for w in widgets)
            assert not widgets[0].has_class("lyrics-rtl")
            assert widgets[1].has_class("lyrics-rtl")
            assert not widgets[2].has_class("lyrics-rtl")
            # The rebuild resets the highlight.
            assert sidebar.current_line_index == -1
            # The scroll area holds the three lines plus a
            # transliteration widget under each non-ASCII line — the
            # Hebrew one and the CJK one both qualify.
            mounted = sidebar.query_one("#ls-scroll").children
            assert len(mounted) == 5
            translit_widgets = [w for w in mounted if w.has_class("lyrics-transliterated")]
            assert len(translit_widgets) == 2
            # Each transliteration directly follows its own line.
            assert mounted.index(translit_widgets[0]) == 2
            assert mounted.index(translit_widgets[1]) == 4

    async def test_synced_view_without_transliteration(self):
        from textual.app import App, ComposeResult

        from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

        class _Host(App):
            """Minimal host providing the theme variables the CSS needs."""

            def get_css_variables(self) -> dict[str, str]:
                variables = super().get_css_variables()
                variables["lyrics-played"] = "#999999"
                variables["lyrics-current"] = "#ff4e45"
                variables["lyrics-upcoming"] = "#cccccc"
                return variables

            def compose(self) -> ComposeResult:
                yield LyricsSidebar()

        app = _Host()
        async with app.run_test(size=(60, 20)):
            sidebar = app.query_one(LyricsSidebar)
            sidebar._synced_lines = [(1.5, "\u6771\u4eac"), (3.0, "hello")]
            sidebar._transliteration_enabled = False

            sidebar._build_synced_view()

            assert len(sidebar._lyric_widgets) == 2
            mounted = sidebar.query_one("#ls-scroll").children
            assert len(mounted) == 2
            assert not any(w.has_class("lyrics-transliterated") for w in mounted)


class TestUnsyncedViewWiring:
    async def test_unsynced_view_mounts_rows_in_order(self):
        from textual.app import App, ComposeResult

        from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

        class _Host(App):
            """Minimal host providing the theme variables the CSS needs."""

            def get_css_variables(self) -> dict[str, str]:
                variables = super().get_css_variables()
                variables["lyrics-played"] = "#999999"
                variables["lyrics-current"] = "#ff4e45"
                variables["lyrics-upcoming"] = "#cccccc"
                return variables

            def compose(self) -> ComposeResult:
                yield LyricsSidebar()

        app = _Host()
        async with app.run_test(size=(60, 20)):
            sidebar = app.query_one(LyricsSidebar)
            sidebar._unsynced_lines = [
                "plain line",
                "\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd",
                "",
            ]
            sidebar._transliteration_enabled = True

            sidebar._build_unsynced_view()

            widgets = sidebar._lyric_widgets
            assert len(widgets) == 3
            # Unsynced lines carry no timestamp.
            assert all(w._timestamp is None for w in widgets)
            assert [w._text for w in widgets] == [
                "plain line",
                f"{FSI}\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd{PDI}",
                "",
            ]
            assert widgets[1].has_class("lyrics-rtl")
            assert not widgets[0].has_class("lyrics-rtl")
            # No non-ASCII-non-RTL line here except the Hebrew, whose
            # transliteration still mounts (it is non-ASCII).
            mounted = sidebar.query_one("#ls-scroll").children
            translit_widgets = [w for w in mounted if w.has_class("lyrics-transliterated")]
            assert len(translit_widgets) == 1

    async def test_unsynced_view_respects_the_toggle(self):
        from textual.app import App, ComposeResult

        from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

        class _Host(App):
            """Minimal host providing the theme variables the CSS needs."""

            def get_css_variables(self) -> dict[str, str]:
                variables = super().get_css_variables()
                variables["lyrics-played"] = "#999999"
                variables["lyrics-current"] = "#ff4e45"
                variables["lyrics-upcoming"] = "#cccccc"
                return variables

            def compose(self) -> ComposeResult:
                yield LyricsSidebar()

        app = _Host()
        async with app.run_test(size=(60, 20)):
            sidebar = app.query_one(LyricsSidebar)
            sidebar._unsynced_lines = ["\u6771\u4eac \u306f \u8cdc\u3084\u304b"]
            sidebar._transliteration_enabled = False

            sidebar._build_unsynced_view()

            mounted = sidebar.query_one("#ls-scroll").children
            assert len(mounted) == 1
            assert not any(w.has_class("lyrics-transliterated") for w in mounted)


# ── Native / fallback parity ────────────────────────────────────────
#
# When the Rust extension is built, the engine above IS the native one;
# each case below is also run in a child interpreter with the extension
# import blocked (pure-Python reference), and the two row lists must
# agree exactly — under both the reorder-off and reorder-on verdicts,
# with the transliteration toggle both ways.

# (lines, wrap_width, transliteration_enabled)
_PARITY_CASES = [
    ([], 35, False),
    ([], 35, True),
    (["hello world", "another line"], 35, False),
    (["hello world"], 35, True),
    ([""], 35, True),
    (["", "hello", ""], 35, False),
    (["   "], 35, True),
    (["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"], 35, False),
    (["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd"], 35, True),
    (["\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"], 20, True),
    (["mix English \u05e2\u05d1\u05e8\u05d9\u05ea end"], 35, False),
    (["mix English \u05e2\u05d1\u05e8\u05d9\u05ea end"], 20, True),
    (["\u05d0 \u05d1 \u05d2 \u05d3 \u05d4 \u05d5 \u05d6 \u05d7 \u05d8 \u05d9"], 10, False),
    (["\u05d0 \u05d1 \u05d2 \u05d3 \u05d4 \u05d5 \u05d6 \u05d7 \u05d8 \u05d9"], 10, True),
    (["\u05e9\u05dc\u05d5\u05dd \u05e2\u05d5\u05dc\u05dd \u05d7\u05d3\u05e9"], 0, True),
    (["\u6771\u4eac\u306f\u3068\u3066\u3082\u8cdc\u3084\u304b\u3067\u3059"], 35, True),
    (["\uc11c\uc6b8\uc758 \ubc24"], 35, True),
    (["caf\u00e9 ni\u00f1o"], 35, True),
    (["\u041f\u0440\u0438\u0432\u0435\u0442 \u043c\u0438\u0440"], 35, True),
    (["\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac"], 35, True),
    (["Hello \u4e16\u754c"], 35, True),
    (["mixed \u65e5\u672c\u8a9e and \u05e2\u05d1\u05e8\u05d9\u05ea line"], 12, True),
    (
        [
            "\u0623\u063a\u0646\u064a\u0629 \u062c\u0645\u064a\u0644\u0629 \u062c\u062f\u0627 \u062c\u062f\u0627 \u062c\u062f\u0627 \u062c\u062f\u0627"
        ],
        8,
        True,
    ),
    (["one two three four five six seven eight nine ten"], 0, False),
    (["a line of latin words far wider than five columns"], 5, True),
    (["first", "second", "third"], 35, False),
    (["\u05e9\u05dc\u05d5\u05dd", "", "\u6771\u4eac", "   ", "end"], 35, True),
]

_CHILD_PROG = """
import json, sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.sidebars import lyrics_sidebar
from ytm_player.utils import bidi

payload = json.load(sys.stdin)
bidi._should_reorder = payload["flag"]

out = []
for case in payload["cases"]:
    lines, width, translit_on = case
    try:
        rows = lyrics_sidebar._compose_lyric_rows(lines, width, translit_on)
        out.append({"ok": [[display, is_rtl, translit] for display, is_rtl, translit in rows]})
    except BaseException as exc:
        out.append({"error": f"{type(exc).__name__}: {exc}"})
json.dump({"flag": payload["flag"], "rows": out}, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native row engine must
    produce exactly the same (display, is_rtl, translit) rows as the
    pure-Python reference — same BiDi renders (under both the
    reorder-off and reorder-on verdicts), same classifications, same
    transliterations and same gates — for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("flag", [False, True])
    def test_native_matches_fallback(self, monkeypatch, flag):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", flag)

        native_rows = []
        for lines, width, translit_on in _PARITY_CASES:
            try:
                rows = _compose_lyric_rows(lines, width, translit_on)
                native_rows.append(
                    {"ok": [[display, is_rtl, translit] for display, is_rtl, translit in rows]}
                )
            except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
                native_rows.append({"error": f"{type(exc).__name__}: {exc}"})

        payload = {"flag": flag, "cases": _PARITY_CASES}
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        assert fallback["flag"] == flag

        for case, native, ref in zip(_PARITY_CASES, native_rows, fallback["rows"]):
            assert native == ref, (
                f"native and fallback lyric rows differ for {case!r} "
                f"(reorder={flag}):\n{native!r}\n{ref!r}"
            )
