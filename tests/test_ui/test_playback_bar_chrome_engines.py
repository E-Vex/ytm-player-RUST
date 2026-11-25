"""Tests for the playback bar's chrome engines.

The pure decision cores behind ``ui/playback_bar.py``'s always-visible
chrome — the module's remaining inline render mappings and click
decisions before this port. Eight engines: the volume display's
content (``_volume_display_text`` — the speaker glyph, the volume
right-aligned into three columns through CPython's own
``format(volume, ">3")``, and the percent sign), the repeat button's
mode pair (``_repeat_button_pair`` — the ``RepeatMode`` comparison and
the icon/label pair), the shuffle button's on/off pair
(``_shuffle_button_pair``), the heart button's like verdict
(``_heart_button_pair``), the footer button's style kind
(``_footer_button_style_kind`` — active over dimmed over muted) and
action routing (``_footer_button_action_route`` — the match
statement's verdict over the seven page actions, the transport trio
and the import popup), the footer's active-page verdict
(``_footer_button_active_verdict``) and the bar's right-click track
pick (``_playback_bar_click_track`` — the button-3 gate and the
player-then-queue fallback chain). The engines run with two
interchangeable backends: the pure-Python transcriptions in the
module and, when the Rust extension is built, the native
implementations bound at the module's tail.

The unit tests below assert each engine's contract on whichever
backend is loaded, and the wiring tests pin the widgets' routing: the
chrome renders walk the engines' pairs, the footer button's click
handler dispatches the engine's route to the host's own handlers, the
footer's active-page walk consults the verdict engine, and the bar's
right-click consults the pick engine before posting its message. When
the extension is built, the parity tests additionally compare the
native engines against the pure-Python references (run in a child
interpreter with the extension import blocked) case for case — error
shapes included — so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult

from ytm_player.services.queue import RepeatMode
from ytm_player.ui.playback_bar import (
    PlaybackBar,
    _footer_button_action_route,
    _footer_button_active_verdict,
    _footer_button_style_kind,
    _heart_button_pair,
    _playback_bar_click_track,
    _repeat_button_pair,
    _shuffle_button_pair,
    _volume_display_text,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── The volume display's content ────────────────────────────────────


class TestVolumeDisplayText:
    """``_volume_display_text``'s glyph, padding and percent, on
    whichever backend."""

    def test_a_two_digit_volume_pads_to_three_columns(self):
        assert _volume_display_text(80) == " \U0001f50a  80%"

    def test_a_one_digit_volume_pads_to_three_columns(self):
        assert _volume_display_text(5) == " \U0001f50a   5%"

    def test_zero_pads_like_any_one_digit(self):
        assert _volume_display_text(0) == " \U0001f50a   0%"

    def test_a_three_digit_volume_needs_no_padding(self):
        assert _volume_display_text(100) == " \U0001f50a 100%"

    def test_a_wider_volume_is_left_un_padded(self):
        # 4 columns — the spec's minimum width, never a truncation.
        assert _volume_display_text(1000) == " \U0001f50a 1000%"

    def test_a_negative_volume_keeps_its_sign(self):
        assert _volume_display_text(-7) == " \U0001f50a  -7%"

    def test_a_string_volume_right_aligns_too(self):
        # A formattable shape answers the interpreter's own rendering —
        # a str pads with spaces just like an int.
        assert _volume_display_text("5") == " \U0001f50a   5%"

    def test_a_float_volume_keeps_its_own_digits(self):
        assert _volume_display_text(80.5) == " \U0001f50a 80.5%"

    def test_an_unformattable_volume_raises_the_interpreter_s_own_error(self):
        # ``{x:>3}`` on an object without a ``__format__`` is the
        # interpreter's own TypeError — the very builtin both engines call.
        with pytest.raises(TypeError):
            _volume_display_text(object())


# ── The repeat button's pair ───────────────────────────────────────


class TestRepeatButtonPair:
    """``_repeat_button_pair``'s mode verdict and pairs, on whichever
    backend."""

    def test_all_answers_the_all_pair(self):
        assert _repeat_button_pair(RepeatMode.ALL) == ("all", "\U0001f501 all")

    def test_one_answers_the_one_pair(self):
        assert _repeat_button_pair(RepeatMode.ONE) == ("one", "\U0001f502 one")

    def test_off_answers_the_off_pair(self):
        assert _repeat_button_pair(RepeatMode.OFF) == ("off", "\U0001f501 off")

    def test_the_member_strings_compare_through_their_values(self):
        # RepeatMode is a StrEnum — a plain string carrying the member's
        # value compares equal to the member, exactly like the reference's
        # own `==`.
        assert _repeat_button_pair("all")[0] == "all"
        assert _repeat_button_pair("one")[0] == "one"

    def test_a_non_member_string_answers_off(self):
        assert _repeat_button_pair("ALL")[0] == "off"
        assert _repeat_button_pair("")[0] == "off"

    def test_a_non_str_mode_answers_off(self):
        # `mode == RepeatMode.ALL` is False for every non-str shape.
        assert _repeat_button_pair(None)[0] == "off"
        assert _repeat_button_pair(5)[0] == "off"


# ── The shuffle button's pair ───────────────────────────────────────


class TestShuffleButtonPair:
    """``_shuffle_button_pair``'s truthiness pick, on whichever
    backend."""

    def test_on_answers_the_on_pair_with_its_trailing_space(self):
        assert _shuffle_button_pair(True) == ("on", "\U0001f500 on ")

    def test_off_answers_the_off_pair(self):
        assert _shuffle_button_pair(False) == ("off", "\U0001f500 off")

    def test_every_truthy_shape_answers_on(self):
        assert _shuffle_button_pair(1)[0] == "on"
        assert _shuffle_button_pair("x")[0] == "on"
        assert _shuffle_button_pair([0])[0] == "on"

    def test_every_falsy_shape_answers_off(self):
        assert _shuffle_button_pair(0)[0] == "off"
        assert _shuffle_button_pair("")[0] == "off"
        assert _shuffle_button_pair(None)[0] == "off"
        assert _shuffle_button_pair([])[0] == "off"


# ── The heart button's pair ─────────────────────────────────────────


class TestHeartButtonPair:
    """``_heart_button_pair``'s like verdict, on whichever backend."""

    def test_the_like_status_answers_the_liked_pair(self):
        assert _heart_button_pair("LIKE") == ("liked", " \u2764 ")

    def test_the_glyph_is_the_same_on_both_arms(self):
        # Only the kind — the colour — differs between the arms.
        assert _heart_button_pair("LIKE")[1] == _heart_button_pair("")[1]

    def test_any_other_status_answers_the_unliked_pair(self):
        assert _heart_button_pair("like")[0] == "unliked"
        assert _heart_button_pair("DISLIKE")[0] == "unliked"
        assert _heart_button_pair("INDIFFERENT")[0] == "unliked"
        assert _heart_button_pair("")[0] == "unliked"

    def test_a_non_str_status_answers_the_unliked_pair(self):
        # `like_status == "LIKE"` is False for every non-str shape.
        assert _heart_button_pair(None)[0] == "unliked"
        assert _heart_button_pair(5)[0] == "unliked"


# ── The footer button's style kind ──────────────────────────────────


class TestFooterButtonStyleKind:
    """``_footer_button_style_kind``'s three-way choice, on whichever
    backend."""

    def test_active_wins_over_dimmed(self):
        # The reference checks active first — an active+dimmed button
        # renders active.
        assert _footer_button_style_kind(True, True) == "active"

    def test_dimmed_alone_is_dimmed(self):
        assert _footer_button_style_kind(False, True) == "dimmed"

    def test_neither_is_muted(self):
        assert _footer_button_style_kind(False, False) == "muted"

    def test_every_gate_is_the_flag_s_own_truthiness(self):
        assert _footer_button_style_kind(1, 0) == "active"
        assert _footer_button_style_kind(0, 1) == "dimmed"
        assert _footer_button_style_kind("", "") == "muted"
        assert _footer_button_style_kind(None, "x") == "dimmed"
        assert _footer_button_style_kind("x", None) == "active"


# ── The footer button's action routing ──────────────────────────────


class TestFooterButtonActionRoute:
    """``_footer_button_action_route``'s match verdict, on whichever
    backend."""

    def test_the_seven_page_actions_route_to_page(self):
        for action in (
            "help",
            "library",
            "search",
            "queue",
            "browse",
            "liked_songs",
            "recently_played",
        ):
            assert _footer_button_action_route(action) == "page"

    def test_the_transport_trio_routes_to_its_handlers(self):
        assert _footer_button_action_route("play_pause") == "play_pause"
        assert _footer_button_action_route("prev") == "prev"
        assert _footer_button_action_route("next") == "next"

    def test_the_import_action_routes_to_the_popup(self):
        assert _footer_button_action_route("spotify_import") == "spotify_import"

    def test_any_other_action_routes_nowhere(self):
        # The reference's match falls through — nothing runs.
        assert _footer_button_action_route("Import") == "none"
        assert _footer_button_action_route("library ") == "none"
        assert _footer_button_action_route("") == "none"

    def test_a_non_str_action_routes_nowhere(self):
        assert _footer_button_action_route(None) == "none"
        assert _footer_button_action_route(5) == "none"


# ── The footer's active-page verdict ────────────────────────────────


class TestFooterButtonActiveVerdict:
    """``_footer_button_active_verdict``'s equality, on whichever
    backend."""

    def test_the_same_name_is_active(self):
        assert _footer_button_active_verdict("library", "library") is True

    def test_a_different_name_is_not(self):
        assert _footer_button_active_verdict("library", "search") is False

    def test_a_non_str_page_name_never_matches(self):
        assert _footer_button_active_verdict("library", None) is False
        assert _footer_button_active_verdict(None, None) is True


# ── The bar's right-click track pick ────────────────────────────────


class TestPlaybackBarClickTrack:
    """``_playback_bar_click_track``'s gate and chain, on whichever
    backend."""

    def test_a_truthy_player_track_wins(self):
        track = {"video_id": "p"}
        assert _playback_bar_click_track(3, track, {"video_id": "q"}) is track

    def test_a_falsy_player_track_falls_to_the_queue(self):
        queue_track = {"video_id": "q"}
        assert _playback_bar_click_track(3, None, queue_track) is queue_track
        assert _playback_bar_click_track(3, {}, queue_track) is queue_track

    def test_the_very_object_is_answered_not_a_copy(self):
        player_track = {"video_id": "p"}
        assert _playback_bar_click_track(3, player_track, None) is player_track

    def test_every_falsy_shape_answers_none(self):
        assert _playback_bar_click_track(3, None, None) is None
        assert _playback_bar_click_track(3, {}, {}) is None
        assert _playback_bar_click_track(3, "", 0) is None

    def test_a_non_right_button_answers_none(self):
        # The button-3 gate — every other button is a no-message click.
        assert _playback_bar_click_track(1, {"video_id": "p"}, None) is None
        assert _playback_bar_click_track(2, None, {"video_id": "q"}) is None


# ── The chrome renders through the engines ──────────────────────────


class TestChromeRenders:
    """The mounted widgets' renders walk the engines' pairs."""

    async def test_the_volume_display_renders_the_engine_content(self):
        from ytm_player.ui.playback_bar import _VolumeDisplay

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield _VolumeDisplay(id="pb-volume")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            display = app.query_one("#pb-volume", _VolumeDisplay)
            for volume in (0, 5, 80, 100, 1000):
                display.volume = volume
                await pilot.pause()
                assert display.render().plain == _volume_display_text(volume)

    async def test_the_repeat_button_renders_each_pair(self):
        from ytm_player.ui.playback_bar import _RepeatButton

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield _RepeatButton(id="pb-repeat")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            button = app.query_one("#pb-repeat", _RepeatButton)
            for mode in (RepeatMode.ALL, RepeatMode.ONE, RepeatMode.OFF):
                button.repeat_mode = mode
                await pilot.pause()
                _, text = _repeat_button_pair(mode)
                assert button.render().plain == text

    async def test_the_shuffle_button_renders_each_pair(self):
        from ytm_player.ui.playback_bar import _ShuffleButton

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield _ShuffleButton(id="pb-shuffle")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            button = app.query_one("#pb-shuffle", _ShuffleButton)
            for shuffle_on in (True, False):
                button.shuffle_on = shuffle_on
                await pilot.pause()
                _, text = _shuffle_button_pair(shuffle_on)
                assert button.render().plain == text

    async def test_the_heart_button_renders_the_liked_and_unliked_glyph(self):
        from ytm_player.ui.playback_bar import _HeartButton

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield _HeartButton(id="pb-heart")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            button = app.query_one("#pb-heart", _HeartButton)
            for status in ("LIKE", "like"):
                button.like_status = status
                await pilot.pause()
                _, text = _heart_button_pair(status)
                assert button.render().plain == text

    async def test_the_footer_button_renders_its_label_under_each_kind(self):
        from ytm_player.ui.playback_bar import _FooterButton

        class _Host(App):
            def compose(self) -> ComposeResult:
                yield _FooterButton("Library", "library", id="footer-library")

        app = _Host()
        async with app.run_test(size=(60, 10)) as pilot:
            button = app.query_one("#footer-library", _FooterButton)
            for active, dimmed in ((True, True), (False, True), (False, False)):
                button.is_active = active
                button.is_dimmed = dimmed
                await pilot.pause()
                kind = _footer_button_style_kind(active, dimmed)
                assert button.render().plain == "Library"
                assert kind in ("active", "dimmed", "muted")


# ── The footer button's click routing ──────────────────────────────


class _FakeClick:
    """A Click stand-in recording the stop."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class _RecordingHost:
    """A YTMHostBase stand-in recording every routed handler."""

    def __init__(self) -> None:
        self.pages: list[str] = []
        self.toggles: int = 0
        self.prevs: int = 0
        self.nexts: int = 0
        self.screens: list[object] = []

    async def navigate_to(self, page: str) -> None:
        self.pages.append(page)

    async def _toggle_play_pause(self) -> None:
        self.toggles += 1

    async def _play_previous(self) -> None:
        self.prevs += 1

    async def _play_next(self) -> None:
        self.nexts += 1

    def push_screen(self, screen: object) -> None:
        self.screens.append(screen)


async def _click(action: str) -> tuple[_RecordingHost, _FakeClick]:
    host = _RecordingHost()
    button = SimpleNamespace(app=host, _action=action)
    event = _FakeClick()
    from ytm_player.ui.playback_bar import _FooterButton

    await _FooterButton.on_click(button, event)  # type: ignore[arg-type]
    return host, event


class TestFooterClickRouting:
    """The footer button's click handler dispatches the route engine's
    verdict to the host's own handlers."""

    async def test_a_page_action_navigates_to_the_action_page(self):
        host, event = await _click("library")
        assert host.pages == ["library"]
        assert event.stopped

    async def test_each_page_action_navigates_to_its_own_page(self):
        for action in ("help", "search", "queue", "browse", "liked_songs", "recently_played"):
            host, _ = await _click(action)
            assert host.pages == [action]

    async def test_the_transport_trio_routes_to_its_handlers(self):
        host, _ = await _click("play_pause")
        assert host.toggles == 1 and not host.pages
        host, _ = await _click("prev")
        assert host.prevs == 1 and not host.pages
        host, _ = await _click("next")
        assert host.nexts == 1 and not host.pages

    async def test_the_import_action_pushes_the_import_popup(self, monkeypatch):
        from ytm_player.ui.popups import spotify_import as si

        class _StubPopup:
            pass

        monkeypatch.setattr(si, "SpotifyImportPopup", _StubPopup)
        host, _ = await _click("spotify_import")
        assert len(host.screens) == 1
        assert isinstance(host.screens[0], _StubPopup)
        assert not host.pages

    async def test_an_unknown_action_runs_nothing(self):
        host, event = await _click("unknown")
        assert host.pages == []
        assert host.toggles == 0 and host.prevs == 0 and host.nexts == 0
        assert host.screens == []
        assert event.stopped


# ── The footer's active-page walk ──────────────────────────────────


class TestFooterActivePage:
    """The footer's active-page walk consults the verdict engine."""

    def test_the_walk_marks_only_the_active_page(self):
        from ytm_player.ui.playback_bar import FooterBar

        buttons: dict[str, object] = {}

        class _FooterBar(FooterBar):
            def query_one(self, selector: str, expect_type: object = object) -> object:  # type: ignore[override]
                assert selector.startswith("#footer-")
                return buttons[selector.removeprefix("#footer-")]

        bar = _FooterBar()
        for action in ("library", "search", "browse", "queue", "help"):
            button = SimpleNamespace(is_active=None)
            buttons[action] = button

        bar.set_active_page("library")

        assert buttons["library"].is_active is True  # type: ignore[attr-defined]
        for action in ("search", "browse", "queue", "help"):
            assert buttons[action].is_active is False  # type: ignore[attr-defined]


# ── The bar's right-click wiring ────────────────────────────────────


class _FakeBar:
    """A PlaybackBar stand-in recording the posted messages."""

    def __init__(self, player_track: object, queue_track: object) -> None:
        self.app = SimpleNamespace(
            player=SimpleNamespace(current_track=player_track)
            if player_track is not None
            else None,
            queue=SimpleNamespace(current_track=queue_track),
        )
        self.posted: list[object] = []
        self.TrackRightClicked = PlaybackBar.TrackRightClicked

    def post_message(self, message: object) -> None:
        self.posted.append(message)


class TestBarClickWiring:
    """The bar's right-click handler consults the pick engine before
    posting its message."""

    def test_a_player_track_is_posted(self):
        track = {"video_id": "p"}
        bar = _FakeBar(track, {"video_id": "q"})
        PlaybackBar.on_click(bar, SimpleNamespace(button=3))  # type: ignore[arg-type]
        assert len(bar.posted) == 1
        assert bar.posted[0].track is track

    def test_a_falsy_player_track_falls_to_the_queue_track(self):
        queue_track = {"video_id": "q"}
        bar = _FakeBar({}, queue_track)
        PlaybackBar.on_click(bar, SimpleNamespace(button=3))  # type: ignore[arg-type]
        assert bar.posted[0].track is queue_track

    def test_a_missing_player_reads_the_queue(self):
        queue_track = {"video_id": "q"}
        bar = _FakeBar(None, queue_track)
        PlaybackBar.on_click(bar, SimpleNamespace(button=3))  # type: ignore[arg-type]
        assert bar.posted[0].track is queue_track

    def test_no_track_posts_nothing(self):
        bar = _FakeBar(None, None)
        PlaybackBar.on_click(bar, SimpleNamespace(button=3))  # type: ignore[arg-type]
        assert bar.posted == []

    def test_a_non_right_button_posts_nothing(self):
        bar = _FakeBar({"video_id": "p"}, None)
        PlaybackBar.on_click(bar, SimpleNamespace(button=1))  # type: ignore[arg-type]
        assert bar.posted == []

    def test_a_partial_host_without_a_queue_never_raises(self):
        # A truthy player track never consults the queue — a partial
        # host (a player with no queue attribute) must not raise where
        # the reference's own short-circuit would not.
        track = {"video_id": "p"}
        posted: list[object] = []
        bar = SimpleNamespace(
            app=SimpleNamespace(player=SimpleNamespace(current_track=track)),
            post_message=posted.append,
            TrackRightClicked=PlaybackBar.TrackRightClicked,
        )
        PlaybackBar.on_click(bar, SimpleNamespace(button=3))  # type: ignore[arg-type]
        assert posted and posted[0].track is track


# ── Native vs fallback parity ───────────────────────────────────────
#
# Every engine travels over the JSON bridge. The repeat cases
# round-trip the StrEnum members as their string values — the fallback
# compares those exactly like the native engine compares the members —
# and the volume error case (an object without a __format__) is pinned
# by the unit tests on both sides of the bridge instead (its message
# is the interpreter's own on either backend, the very builtin both
# call).

_VOLUME_PARITY_CASES = [0, 5, 80, 100, 999, 1000, -7, "5", "80", 80.5]

_REPEAT_PARITY_CASES = [
    "all",
    "one",
    "off",
    "ALL",
    "",
    "library",
    None,
    5,
]

_SHUFFLE_PARITY_CASES = [True, False, 0, 1, "", "x", None, []]

_HEART_PARITY_CASES = ["LIKE", "like", "DISLIKE", "INDIFFERENT", "", None, 5]

_STYLE_PARITY_CASES = [
    [True, True],
    [True, False],
    [False, True],
    [False, False],
    [0, 1],
    [1, 0],
    [None, "x"],
    ["x", None],
    ["", ""],
]

_ROUTE_PARITY_CASES = [
    "help",
    "library",
    "search",
    "queue",
    "browse",
    "liked_songs",
    "recently_played",
    "play_pause",
    "prev",
    "next",
    "spotify_import",
    "Import",
    "library ",
    "",
    None,
    5,
]

_ACTIVE_PARITY_CASES = [
    ["library", "library"],
    ["library", "search"],
    ["queue", "queue"],
    ["help", None],
    [None, None],
    ["x", 5],
]

# [button, player_track, queue_track]
_PICK_PARITY_CASES = [
    [3, {"video_id": "p"}, None],
    [3, {"video_id": "p"}, {"video_id": "q"}],
    [3, None, {"video_id": "q"}],
    [3, {}, {"video_id": "q"}],
    [3, {}, {}],
    [3, None, None],
    [3, "plain", None],
    [3, None, "plain"],
    [3, None, ""],
    [1, {"video_id": "p"}, None],
    [2, None, {"video_id": "q"}],
    [0, {}, {}],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.playback_bar import (
    _footer_button_active_verdict,
    _footer_button_action_route,
    _footer_button_style_kind,
    _heart_button_pair,
    _playback_bar_click_track,
    _repeat_button_pair,
    _shuffle_button_pair,
    _volume_display_text,
)


def outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def pick_shape(answer):
    if answer is None:
        return None
    if isinstance(answer, dict):
        return {"type": "dict", "video_id": answer.get("video_id")}
    return {"type": type(answer).__name__, "repr": repr(answer)}


payload = json.load(sys.stdin)
out = {
    "volume": [outcome(_volume_display_text, case) for case in payload["volume"]],
    "repeat": [outcome(_repeat_button_pair, case) for case in payload["repeat"]],
    "shuffle": [outcome(_shuffle_button_pair, case) for case in payload["shuffle"]],
    "heart": [outcome(_heart_button_pair, case) for case in payload["heart"]],
    "style": [outcome(_footer_button_style_kind, *case) for case in payload["style"]],
    "route": [outcome(_footer_button_action_route, case) for case in payload["route"]],
    "active": [
        outcome(_footer_button_active_verdict, *case) for case in payload["active"]
    ],
    "pick": [
        outcome(_playback_bar_click_track, *case) for case in payload["pick"]
    ],
}
# The pick engine answers objects — the posted track's identity — which
# do not survive the JSON bridge; reduce them to their verdict shape so
# the bridge compares the decision, not the serialization.
for pick in out["pick"]:
    if "ok" in pick:
        pick["ok"] = pick_shape(pick["ok"])
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_outcomes() -> dict[str, list]:
    """The parent-side outcomes — every engine over every case, run by
    whichever backend the module loaded (the native one, when the
    parity tests run)."""

    def outcome(fn, *args):
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"ok": result}

    def pick_shape(answer):
        if answer is None:
            return None
        if isinstance(answer, dict):
            return {"type": "dict", "video_id": answer.get("video_id")}
        return {"type": type(answer).__name__, "repr": repr(answer)}

    picks = []
    for button, player_track, queue_track in _PICK_PARITY_CASES:
        pick = outcome(_playback_bar_click_track, button, player_track, queue_track)
        if "ok" in pick:
            pick["ok"] = pick_shape(pick["ok"])
        picks.append(pick)
    return {
        "volume": [outcome(_volume_display_text, case) for case in _VOLUME_PARITY_CASES],
        "repeat": [outcome(_repeat_button_pair, case) for case in _REPEAT_PARITY_CASES],
        "shuffle": [outcome(_shuffle_button_pair, case) for case in _SHUFFLE_PARITY_CASES],
        "heart": [outcome(_heart_button_pair, case) for case in _HEART_PARITY_CASES],
        "style": [outcome(_footer_button_style_kind, *case) for case in _STYLE_PARITY_CASES],
        "route": [outcome(_footer_button_action_route, case) for case in _ROUTE_PARITY_CASES],
        "active": [outcome(_footer_button_active_verdict, *case) for case in _ACTIVE_PARITY_CASES],
        "pick": picks,
    }


def _run_child() -> dict[str, list]:
    """The pure-Python reference outcomes, run in a child interpreter
    with the extension import blocked."""
    payload = {
        "volume": _VOLUME_PARITY_CASES,
        "repeat": _REPEAT_PARITY_CASES,
        "shuffle": _SHUFFLE_PARITY_CASES,
        "heart": _HEART_PARITY_CASES,
        "style": _STYLE_PARITY_CASES,
        "route": _ROUTE_PARITY_CASES,
        "active": _ACTIVE_PARITY_CASES,
        "pick": _PICK_PARITY_CASES,
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same contents, same pairs,
    same kinds, same verdicts, same errors with the same messages — so
    the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_every_engine_matches_the_fallback(self):
        fallback = _run_child()
        native = json.loads(json.dumps(_native_outcomes()))
        for family, cases in [
            ("volume", _VOLUME_PARITY_CASES),
            ("repeat", _REPEAT_PARITY_CASES),
            ("shuffle", _SHUFFLE_PARITY_CASES),
            ("heart", _HEART_PARITY_CASES),
            ("style", _STYLE_PARITY_CASES),
            ("route", _ROUTE_PARITY_CASES),
            ("active", _ACTIVE_PARITY_CASES),
            ("pick", _PICK_PARITY_CASES),
        ]:
            for case, n, f in zip(cases, native[family], fallback[family]):
                assert n == f, (
                    f"native and fallback {family} engine differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
