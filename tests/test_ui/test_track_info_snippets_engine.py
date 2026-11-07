"""Tests for the playback-bar track-info snippet engine.

The engine behind ``_TrackInfo.render``: the one-line content walk —
the state-icon branch, the title/artist/album fragments truncated
through the ``render_bar_cell`` chain with the widths the
``_track_info_widths`` engine splits, the em-dash separators between
them, and the no-track placeholder — answering the ``(kind, text)``
snippet sequence the render styles by kind. The pure-Python reference
lives in ``ui/playback_bar.py`` (``_track_info_snippets``); when the
Rust extension is built the same engine runs natively
(``_ytm_player_native.track_info_snippets``).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case —
under both the reorder-off and reorder-on BiDi verdicts — so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.playback_bar import _track_info_snippets

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

# The state-icon glyphs, spelled here so the assertions read like the
# bar itself.
_PLAYING_ICON = " \u25b6 "
_PAUSED_ICON = " \u23f8 "
_STOPPED_ICON = " \u25a0 "

# The em-dash separator between the fragments.
_SEP = " \u2014 "


def _kinds(snippets: list[tuple[str, str]]) -> list[str]:
    """The kind sequence of a snippet list — the walk's own shape."""
    return [kind for kind, _ in snippets]


# ── The state-icon branch ──────────────────────────────────────────


class TestStateIcon:
    def test_playing_and_not_paused_is_the_play_icon(self):
        snippets = _track_info_snippets(60, "T", "A", "B", True, False)
        assert snippets[0] == ("playing", _PLAYING_ICON)

    def test_paused_alone_is_the_pause_icon(self):
        snippets = _track_info_snippets(60, "T", "A", "B", False, True)
        assert snippets[0] == ("paused", _PAUSED_ICON)

    def test_playing_and_paused_is_the_pause_icon(self):
        # The branch order: `is_playing and not is_paused` fails first,
        # `elif is_paused` takes over — a paused-but-flagged-playing
        # state shows the pause bars.
        snippets = _track_info_snippets(60, "T", "A", "B", True, True)
        assert snippets[0] == ("paused", _PAUSED_ICON)

    def test_neither_is_the_stopped_icon(self):
        snippets = _track_info_snippets(60, "T", "A", "B", False, False)
        assert snippets[0] == ("stopped", _STOPPED_ICON)


# ── The no-track placeholder ───────────────────────────────────────


class TestNoTrack:
    def test_empty_title_is_the_placeholder(self):
        snippets = _track_info_snippets(60, "", "", "", False, False)
        assert snippets == [("stopped", _STOPPED_ICON), ("empty", "No track playing")]

    def test_artist_and_album_are_never_consulted_without_a_title(self):
        # A falsy title walks the empty arm — the artist and album are
        # not rendered even when set.
        snippets = _track_info_snippets(60, "", "Some Artist", "Some Album", True, False)
        assert snippets == [("playing", _PLAYING_ICON), ("empty", "No track playing")]


# ── The fragment walk ──────────────────────────────────────────────


class TestFragments:
    def test_full_walk_shape(self):
        snippets = _track_info_snippets(120, "Title", "Artist", "Album", True, False)
        assert snippets == [
            ("playing", _PLAYING_ICON),
            ("title", "Title"),
            ("sep", _SEP),
            ("artist", "Artist"),
            ("sep", _SEP),
            ("album", "Album"),
        ]

    def test_long_fragments_truncate_through_the_width_engine(self):
        # bar_width 120 → max_w 90 → title 45, artist 30, album 7.
        snippets = _track_info_snippets(120, "x" * 100, "y" * 100, "z" * 100, True, False)
        by_kind = dict(snippets)
        assert by_kind["title"] == "x" * 44 + "…"
        assert by_kind["artist"] == "y" * 29 + "…"
        assert by_kind["album"] == "z" * 6 + "…"

    def test_absent_artist_skips_its_separator(self):
        snippets = _track_info_snippets(120, "Title", "", "Album", True, False)
        assert _kinds(snippets) == ["playing", "title", "sep", "album"]
        assert dict(snippets)["sep"] == _SEP

    def test_absent_album_skips_its_separator(self):
        snippets = _track_info_snippets(120, "Title", "Artist", "", True, False)
        assert _kinds(snippets) == ["playing", "title", "sep", "artist"]

    def test_only_a_title(self):
        snippets = _track_info_snippets(120, "Title", "", "", True, False)
        assert snippets == [("playing", _PLAYING_ICON), ("title", "Title")]

    def test_zero_album_width_answers_the_empty_fragment(self):
        # bar_width 30 → max_w 10 → title 5, artist 3, album
        # 10-5-3-8 < 0 → 0 — the album truncates to the empty string
        # but its (separator, snippet) pair still walks, exactly like
        # the reference's append of a zero-width render.
        snippets = _track_info_snippets(30, "Title", "Artist", "Album", True, False)
        assert snippets[-1] == ("album", "")
        assert snippets[-2] == ("sep", _SEP)

    def test_cjk_lens_are_code_points(self):
        # The widths split on code points — three CJK characters are
        # three to the split.
        snippets = _track_info_snippets(120, "日本語", "アーティスト", "アルバム", True, False)
        assert dict(snippets)["title"] == "日本語"


# ── The BiDi fragment render ───────────────────────────────────────


class TestBidiRender:
    def test_rtl_fragment_is_isolated(self):
        snippets = _track_info_snippets(120, "שלום עלום", "", "", True, False)
        title = dict(snippets)["title"]
        assert title.startswith("⁨")
        assert title.endswith("⁩")

    def test_pure_ltr_fragment_is_not_isolated(self):
        snippets = _track_info_snippets(120, "Hello", "", "", True, False)
        assert dict(snippets)["title"] == "Hello"

    def test_separators_are_never_isolated(self):
        snippets = _track_info_snippets(120, "T", "A", "B", True, False)
        seps = [text for kind, text in snippets if kind == "sep"]
        # One separator before the artist, one before the album — both
        # the bare em-dash, never the BiDi isolate.
        assert seps == [_SEP, _SEP]


# ── The render through the engine ──────────────────────────────────


class TestRender:
    async def test_mounted_render_walks_the_snippets(self):
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
            info.title = "Song"
            info.artist = "Artist"
            info.album = "Album"
            info.is_playing = True
            await pilot.pause()

            snippets = _track_info_snippets(60, "Song", "Artist", "Album", True, False)
            render = info.render()
            # The render's plain text is exactly the engine's snippet
            # texts, appended in the engine's own order.
            assert render.plain == "".join(text for _, text in snippets)

    async def test_mounted_render_carries_the_pause_icon(self):
        from textual.app import App

        from ytm_player.ui.playback_bar import _TrackInfo

        class _Host(App):
            def compose(self):
                yield _TrackInfo(id="pb-track-info")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            info = app.query_one("#pb-track-info", _TrackInfo)
            info.title = "Song"
            info.is_playing = True
            info.is_paused = True
            await pilot.pause()

            render = info.render()
            assert _PAUSED_ICON in render.plain
            assert _PLAYING_ICON not in render.plain

    async def test_mounted_render_without_a_track(self):
        from textual.app import App

        from ytm_player.ui.playback_bar import _TrackInfo

        class _Host(App):
            def compose(self):
                yield _TrackInfo(id="pb-track-info")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            info = app.query_one("#pb-track-info", _TrackInfo)
            await pilot.pause()

            render = info.render()
            assert render.plain == _STOPPED_ICON + "No track playing"


# ── Native vs fallback parity ──────────────────────────────────────
#
# The snippet engine travels over the same JSON bridge, with the cached
# bidi_mode verdict pinned to the same value in both interpreters (the
# child sets bidi._should_reorder before importing the engine; the
# parent monkeypatches the same global), so the render paths compare
# exactly — both the reorder-off pass and the reorder-on pass.

_PARITY_CASES = [
    # (bar_width, title, artist, album, is_playing, is_paused)
    (60, "Song Title", "Artist Name", "Album", True, False),
    (60, "Song Title", "Artist Name", "Album", True, True),
    (60, "Song Title", "Artist Name", "Album", False, True),
    (60, "Song Title", "Artist Name", "Album", False, False),
    (30, "", "A", "B", False, False),
    (0, "", "", "", False, False),
    (0, "T", "A", "B", False, False),
    (1, "Title", "Artist", "Album", True, False),
    (20, "Title", "Artist", "Album", True, False),
    (30, "Title", "Artist", "Album", True, False),
    (40, "Title", "Artist", "Album", True, False),
    (41, "Title", "Artist", "Album", True, False),
    (50, "x" * 100, "y" * 100, "z" * 100, True, False),
    (80, "x" * 100, "y" * 100, "z" * 100, True, False),
    (120, "x" * 100, "y" * 100, "z" * 100, True, False),
    (120, "Title", "", "Album", True, False),
    (120, "Title", "Artist", "", True, False),
    (120, "Title", "", "", True, False),
    (120, "T", "A", "B", True, False),
    (120, "A very long title that will not fit", "A very long artist name", "", True, False),
    (200, "A much longer title here", "Some artist name", "Album name", False, False),
    (500, "x" * 400, "y" * 400, "z" * 400, True, False),
    # CJK / astral / combining — the code-point lens.
    (100, "日本語のタイトル", "アーティスト", "アルバム", True, False),
    (120, "🎧🎧🎧", "ééé", "𝄞♯", True, False),
    # RTL fragments: the isolate wrap (and, under the reorder flag, the
    # word-level reversal) both apply.
    (60, "שלום עלום", "سلام", "مرحبا", True, False),
    (120, "mix English עברית end", "עברית and English", "مرحبا بالعالم", True, True),
    (120, "RTL without artist", "مرحبا", "", False, False),
]

_CHILD_PROG = """
import json, sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.playback_bar import _track_info_snippets
from ytm_player.utils import bidi

payload = json.load(sys.stdin)
bidi._should_reorder = payload["flag"]

out = {
    "flag": payload["flag"],
    "snippets": [
        [list(snippet) for snippet in _track_info_snippets(*case)]
        for case in payload["cases"]
    ],
}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native snippet engine must
    answer every case exactly like the pure-Python reference run in a
    child interpreter with the extension import blocked — same kinds,
    same texts (the BiDi renders included, under both the reorder-off
    and reorder-on verdicts), same order — so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("flag", [False, True])
    def test_native_matches_fallback(self, monkeypatch, flag):
        from ytm_player.utils import bidi

        monkeypatch.setattr(bidi, "_should_reorder", flag)
        payload = {"flag": flag, "cases": _PARITY_CASES}
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        native = [
            [list(snippet) for snippet in _track_info_snippets(*case)] for case in _PARITY_CASES
        ]
        for case, n, f in zip(_PARITY_CASES, native, fallback["snippets"]):
            assert n == f, (
                f"native and fallback track-info snippets differ for {case!r}:\n"
                f"native={n!r}\nfallback={f!r}"
            )
