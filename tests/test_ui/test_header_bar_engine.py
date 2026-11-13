"""Tests for the header bar's click-routing and lyrics-class engines.

``_click_route`` and ``_lyrics_class`` (ui/header_bar.py) run with two
interchangeable backends: the pure-Python transcriptions in the module
and, when the Rust extension is built, the native implementations
bound at the module's tail. The engines: the four-way click routing
(``"playlist"`` / ``"lyrics"`` / ``"back"`` / ``"forward"`` / none)
behind ``HeaderBar.on_click``, and the three-way lyrics-toggle class
choice (``"dimmed"`` / ``"active"`` / none) behind
``_apply_lyrics_classes``.

The unit tests below assert each engine's contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.header_bar import HeaderBar, _click_route, _lyrics_class

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestClickRoute:
    """The routing verdict (whichever backend)."""

    def test_the_four_toggle_ids_route(self):
        assert _click_route("toggle-playlist") == "playlist"
        assert _click_route("toggle-lyrics") == "lyrics"
        assert _click_route("back-btn") == "back"
        assert _click_route("forward-btn") == "forward"

    def test_other_ids_route_nowhere(self):
        assert _click_route("header-spacer") == "none"
        assert _click_route("toggle-playlist2") == "none"
        assert _click_route("") == "none"

    def test_a_widget_without_an_id_routes_nowhere(self):
        """``Widget.id`` is ``str | None``; a None id never routes."""
        assert _click_route(None) == "none"

    def test_the_match_is_exact_not_case_insensitive(self):
        assert _click_route("TOGGLE-PLAYLIST") == "none"
        assert _click_route("toggle-playlist ") == "none"
        assert _click_route(" back-btn") == "none"


class TestLyricsClass:
    """The lyrics-toggle class choice (whichever backend)."""

    def test_dimmed_wins_over_open(self):
        """The reference checks dimmed first — an open+dimmed pane is dimmed."""
        assert _lyrics_class(True, True) == "dimmed"

    def test_open_and_lit_is_active(self):
        assert _lyrics_class(False, True) == "active"

    def test_closed_is_no_class(self):
        assert _lyrics_class(False, False) is None

    def test_dimmed_but_closed_is_still_dimmed(self):
        """The dimmed flag alone decides — the pane stays dimmed when set."""
        assert _lyrics_class(True, False) == "dimmed"


class _FakeStatic:
    """A query_one stand-in recording class operations."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []

    def add_class(self, *classes: str) -> None:
        self.added.extend(classes)

    def remove_class(self, *classes: str) -> None:
        self.removed.extend(classes)


class _FakeClick:
    """A Click stand-in exposing the widget and recording the stop."""

    def __init__(self, widget: object | None) -> None:
        self.widget = widget
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _FakeWidget:
    def __init__(self, widget_id: str | None) -> None:
        self.id = widget_id


def _make_bar() -> tuple[HeaderBar, list]:
    """A HeaderBar whose posted messages are recorded."""
    bar = HeaderBar()
    posted: list = []
    bar.post_message = posted.append  # type: ignore[method-assign]
    return bar, posted


class TestClickWiring:
    """The bar's click handler routes its verdict into the messages."""

    def test_playlist_toggle_posts_playlist_message(self):
        bar, posted = _make_bar()
        event = _FakeClick(_FakeWidget("toggle-playlist"))
        bar.on_click(event)  # type: ignore[arg-type]
        assert len(posted) == 1
        assert isinstance(posted[0], HeaderBar.TogglePlaylistSidebar)
        assert event.stopped

    def test_lyrics_toggle_posts_lyrics_message(self):
        bar, posted = _make_bar()
        event = _FakeClick(_FakeWidget("toggle-lyrics"))
        bar.on_click(event)  # type: ignore[arg-type]
        assert len(posted) == 1
        assert isinstance(posted[0], HeaderBar.ToggleLyricsSidebar)
        assert event.stopped

    def test_back_button_posts_back_message(self):
        bar, posted = _make_bar()
        event = _FakeClick(_FakeWidget("back-btn"))
        bar.on_click(event)  # type: ignore[arg-type]
        assert len(posted) == 1
        assert isinstance(posted[0], HeaderBar.BackRequested)
        assert event.stopped

    def test_forward_button_posts_forward_message(self):
        bar, posted = _make_bar()
        event = _FakeClick(_FakeWidget("forward-btn"))
        bar.on_click(event)  # type: ignore[arg-type]
        assert len(posted) == 1
        assert isinstance(posted[0], HeaderBar.ForwardRequested)
        assert event.stopped

    def test_a_click_on_anything_else_posts_nothing(self):
        bar, posted = _make_bar()
        event = _FakeClick(_FakeWidget("header-spacer"))
        bar.on_click(event)  # type: ignore[arg-type]
        assert posted == []
        assert not event.stopped

    def test_a_click_without_a_target_widget_posts_nothing(self):
        bar, posted = _make_bar()
        event = _FakeClick(None)
        bar.on_click(event)  # type: ignore[arg-type]
        assert posted == []
        assert not event.stopped


class TestLyricsWiring:
    """The lyrics state setters route their verdict into the classes."""

    @staticmethod
    def _make_bar_with_fake_button() -> tuple[HeaderBar, _FakeStatic]:
        bar = HeaderBar()
        fake = _FakeStatic()
        bar.query_one = lambda *args, **kwargs: fake  # type: ignore[method-assign]
        return bar, fake

    def test_open_and_lit_adds_active(self):
        bar, fake = self._make_bar_with_fake_button()
        bar.set_lyrics_state(True)
        assert fake.removed == ["active", "dimmed"]
        assert fake.added == ["active"]

    def test_dimmed_adds_dimmed_over_active(self):
        bar, fake = self._make_bar_with_fake_button()
        bar.set_lyrics_state(True)
        bar.set_lyrics_dimmed(True)
        assert fake.removed == ["active", "dimmed", "active", "dimmed"]
        assert fake.added == ["active", "dimmed"]

    def test_closing_while_dimmed_stays_dimmed(self):
        """The dimmed flag survives an is_on flip — it wins the choice."""
        bar, fake = self._make_bar_with_fake_button()
        bar.set_lyrics_state(True)
        bar.set_lyrics_dimmed(True)
        bar.set_lyrics_state(False)
        assert fake.added == ["active", "dimmed", "dimmed"]

    def test_fully_closed_adds_nothing(self):
        """A closed, undimmed pane removes both classes and adds none."""
        bar, fake = self._make_bar_with_fake_button()
        bar.set_lyrics_state(False)
        bar.set_lyrics_dimmed(False)
        assert fake.added == []
        # One remove-then-add pass per setter call, none adding a class.
        assert fake.removed == ["active", "dimmed", "active", "dimmed"]


# ── Native vs fallback parity ─────────────────────────────────────────

_ROUTE_CASES = [
    None,
    "toggle-playlist",
    "toggle-lyrics",
    "back-btn",
    "forward-btn",
    "header-spacer",
    "",
    "toggle-playlist ",
    "TOGGLE-PLAYLIST",
    "back",
]

_CLASS_CASES = [
    [False, False],
    [False, True],
    [True, False],
    [True, True],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.header_bar import _click_route, _lyrics_class

payload = json.load(sys.stdin)
out = {
    "routes": [_click_route(case) for case in payload["route_cases"]],
    "classes": [_lyrics_class(dimmed, on) for dimmed, on in payload["class_cases"]],
}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native verdicts must equal
    the pure-Python references' for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self):
        payload = json.dumps({"route_cases": _ROUTE_CASES, "class_cases": _CLASS_CASES})
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native_routes = [_click_route(case) for case in _ROUTE_CASES]
        native_classes = [_lyrics_class(dimmed, on) for dimmed, on in _CLASS_CASES]
        assert native_routes == fallback["routes"]
        assert native_classes == fallback["classes"]
