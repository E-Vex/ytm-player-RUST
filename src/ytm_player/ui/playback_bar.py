"""Always-visible playback status bar and interactive footer.

The module's chrome engines — the volume display's content
(``_volume_display_text``, the speaker glyph, the volume
right-aligned into three columns and the percent sign), the repeat
button's mode pair (``_repeat_button_pair``), the shuffle button's
on/off pair (``_shuffle_button_pair``), the heart button's like
verdict (``_heart_button_pair``), the footer button's style kind
(``_footer_button_style_kind``) and action routing
(``_footer_button_action_route``), the footer's active-page verdict
(``_footer_button_active_verdict``), the bar's right-click track pick
(``_playback_bar_click_track``) and the track-info width/snippet
engines — are dual-implemented: the Rust extension
(``_ytm_player_native``) implements them when it is built, and the
pure-Python transcriptions below run otherwise; the bottom-of-module
seam rebinds the engines to the native functions when the extension
is importable. The widgets below only do styling, DOM and event
plumbing on the engines' answers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget

from ytm_player.services.queue import RepeatMode
from ytm_player.ui.theme import get_theme
from ytm_player.ui.widgets.album_art import AlbumArt
from ytm_player.ui.widgets.progress_bar import PlaybackProgress
from ytm_player.utils.formatting import extract_artist

if TYPE_CHECKING:
    from ytm_player.app._base import YTMHostBase

logger = logging.getLogger(__name__)

# Playback state symbols.
_ICON_PLAYING = "\u25b6"  # Black right-pointing triangle
_ICON_PAUSED = "\u23f8"  # Double vertical bar
_ICON_STOPPED = "\u25a0"  # Black square

_ICON_VOLUME = "\U0001f50a"  # Speaker high volume

_ICON_REPEAT_OFF = "\U0001f501"  # Repeat button
_ICON_REPEAT_ALL = "\U0001f501"  # Same icon, coloured differently
_ICON_REPEAT_ONE = "\U0001f502"  # Repeat single button

_ICON_SHUFFLE_OFF = "\U0001f500"  # Twisted arrows
_ICON_SHUFFLE_ON = "\U0001f500"  # Same, coloured differently


# ── Track info widget ─────────────────────────────────────────────


def _track_info_widths(bar_width: int, title: str, artist: str) -> tuple[int, int, int]:
    """The title/artist/album truncation widths — the width-splitting engine.

    The pure core of the ``_TrackInfo`` render: the available budget for
    the three text fragments and the per-fragment widths the reference
    truncates each through. ``max_w = max(10, bar_width - 30)`` (the
    reserved tail — volume, repeat, shuffle, heart — plus the state
    icon's 3 columns, floored at 10 for the narrowest terminals);
    ``title_w = min(len(title), max_w // 2)`` and
    ``artist_w = min(len(artist), max_w // 3)`` (the halves-and-thirds
    split of the reference, floor division); and
    ``album_w = max(0, max_w - title_w - artist_w - 8)`` (the two
    em-dash separators — 3 columns each — and the icon again, clamped
    at the reference's own ``max(0, album_w)`` at the truncate call).
    The album's own length never enters the split — the reference
    gives the album whatever remains — and the lens are CPython's
    ``len()``: code points, computed on the original (pre-reorder,
    pre-truncation) strings exactly like the reference computes them.

    Implemented by the Rust extension
    (``_ytm_player_native.track_info_widths``) when it is built; this
    pure-Python transcription is the fallback otherwise. The fused
    snippet engine below runs this split inside its own walk.
    """
    max_w = max(10, bar_width - 30)
    title_w = min(len(title), max_w // 2)
    artist_w = min(len(artist), max_w // 3)
    album_w = max(0, max_w - title_w - artist_w - 8)
    return title_w, artist_w, album_w


def _track_info_snippets(
    bar_width: int,
    title: str,
    artist: str,
    album: str,
    is_playing: bool,
    is_paused: bool,
) -> list[tuple[str, str]]:
    """The one-line track-info content walk — the snippet engine.

    The pure half of ``_TrackInfo.render``: the ``(kind, text)``
    snippet sequence the render styles by kind — the state icon
    (``"playing"`` / ``"paused"`` / ``"stopped"``, the three-way
    branch in the reference's own order: playing wins only while not
    also paused), then, when a title is set, the title / artist /
    album fragments (each truncated through the ``render_bar_cell``
    chain with the widths the ``_track_info_widths`` engine splits,
    the em-dash ``"sep"`` separators between them, an absent artist
    or album skipping its separator too), or the ``"No track playing"``
    placeholder (``"empty"``) otherwise — a falsy title never consults
    the artist or the album. The styling — the theme colour each kind
    is appended with — stays with the widget.

    Implemented by the Rust extension
    (``_ytm_player_native.track_info_snippets``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    from ytm_player.utils.bidi import render_bar_cell

    snippets: list[tuple[str, str]] = []
    if is_playing and not is_paused:
        snippets.append(("playing", f" {_ICON_PLAYING} "))
    elif is_paused:
        snippets.append(("paused", f" {_ICON_PAUSED} "))
    else:
        snippets.append(("stopped", f" {_ICON_STOPPED} "))

    if title:
        title_w, artist_w, album_w = _track_info_widths(bar_width, title, artist)
        snippets.append(("title", render_bar_cell(title, title_w)))
        if artist:
            snippets.append(("sep", " \u2014 "))
            snippets.append(("artist", render_bar_cell(artist, artist_w)))
        if album:
            snippets.append(("sep", " \u2014 "))
            snippets.append(("album", render_bar_cell(album, album_w)))
    else:
        snippets.append(("empty", "No track playing"))
    return snippets


class _TrackInfo(Widget):
    """Displays the current track title, artist, and album on a single line."""

    DEFAULT_CSS = """
    _TrackInfo {
        height: 1;
        width: 1fr;
    }
    """

    title: reactive[str] = reactive("")
    artist: reactive[str] = reactive("")
    album: reactive[str] = reactive("")
    is_playing: reactive[bool] = reactive(False)
    is_paused: reactive[bool] = reactive(False)

    def render(self) -> Text:
        theme = get_theme()
        result = Text()

        # The whole content walk — the state icon, the three fragments
        # and their separators (or the no-track placeholder) — runs
        # through the snippet engine (Rust when the extension is built):
        # one call answering the (kind, text) sequence, styled here by
        # kind with the theme's colours. FSI...PDI isolates each RTL
        # fragment so it cannot pull the adjacent widgets (volume,
        # repeat, shuffle) into the RTL BiDi context — the engine
        # applies the isolate AFTER truncate (PDI must not be cut).
        for kind, text in _track_info_snippets(
            self.size.width, self.title, self.artist, self.album, self.is_playing, self.is_paused
        ):
            if kind == "playing":
                result.append(text, style=f"bold {theme.primary}")
            elif kind == "paused":
                result.append(text, style=f"bold {theme.warning}")
            elif kind == "title":
                result.append(text, style=f"bold {theme.foreground}")
            elif kind == "artist":
                result.append(text, style=theme.secondary)
            else:
                # "stopped", "sep", "album" and "empty" all render
                # in the muted colour.
                result.append(text, style=theme.muted_text)

        return result


# ── Interactive control widgets ───────────────────────────────────


def _volume_display_text(volume: Any) -> str:
    """The volume display's content — the ``volume`` engine.

    The pure core of ``_VolumeDisplay.render``: the speaker glyph, the
    volume right-aligned into three columns, and the percent sign —
    ``{volume:>3}`` is CPython's own ``format(volume, ">3")``, the
    very builtin call the f-string field compiles down to, so every
    formattable shape answers the interpreter's own rendering. The
    widget keeps only the secondary colour.

    Implemented by the Rust extension
    (``_ytm_player_native.volume_display_text``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return f" {_ICON_VOLUME} {volume:>3}%"


def _repeat_button_pair(mode: Any) -> tuple[str, str]:
    """The repeat button's content pair — the ``repeat`` engine.

    The pure core of ``_RepeatButton.render``: which of the three mode
    pairs the indicator shows — ``mode == RepeatMode.ALL`` then ``mode
    == RepeatMode.ONE``, CPython's own ``==`` against the very enum
    members the module exports — answering ``(kind, text)`` with the
    kind carrying the style the widget applies.

    Implemented by the Rust extension
    (``_ytm_player_native.repeat_button_pair``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if mode == RepeatMode.ALL:
        return ("all", f"{_ICON_REPEAT_ALL} all")
    if mode == RepeatMode.ONE:
        return ("one", f"{_ICON_REPEAT_ONE} one")
    return ("off", f"{_ICON_REPEAT_OFF} off")


def _shuffle_button_pair(shuffle_on: Any) -> tuple[str, str]:
    """The shuffle button's content pair — the ``shuffle`` engine.

    The pure core of ``_ShuffleButton.render``: the on pair (trailing
    space and all) or the off pair, picked by the flag's own
    truthiness. The kind carries the style the widget applies.

    Implemented by the Rust extension
    (``_ytm_player_native.shuffle_button_pair``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if shuffle_on:
        return ("on", f"{_ICON_SHUFFLE_ON} on ")
    return ("off", f"{_ICON_SHUFFLE_OFF} off")


def _heart_button_pair(like_status: Any) -> tuple[str, str]:
    """The heart button's content pair — the ``heart`` engine.

    The pure core of ``_HeartButton.render``: the like verdict —
    ``like_status == "LIKE"``, CPython's own ``==`` on the string
    literal, so a non-str status answers the unliked pair exactly like
    the reference — answering ``(kind, text)`` with the kind carrying
    the bold accent vs muted colour; the glyph text is the same on
    both arms.

    Implemented by the Rust extension
    (``_ytm_player_native.heart_button_pair``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if like_status == "LIKE":
        return ("liked", f" {_ICON_HEART} ")
    return ("unliked", f" {_ICON_HEART} ")


class _VolumeDisplay(Widget):
    """Volume display — scroll to change volume."""

    DEFAULT_CSS = """
    _VolumeDisplay {
        height: 1;
        width: auto;
        min-width: 9;
    }
    """

    volume: reactive[int] = reactive(80)

    def render(self) -> Text:
        # The content — the speaker glyph, the volume right-aligned into
        # three columns, the percent sign — is resolved by the
        # ``_volume_display_text`` engine (native when the Rust extension
        # is built); the widget keeps only the secondary colour.
        return Text(_volume_display_text(self.volume), style=get_theme().secondary)

    async def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        event.stop()
        app = cast("YTMHostBase", self.app)
        if app.player is not None:
            await app.player.change_volume(5)

    async def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        event.stop()
        app = cast("YTMHostBase", self.app)
        if app.player is not None:
            await app.player.change_volume(-5)


class _RepeatButton(Widget):
    """Clickable repeat mode indicator."""

    DEFAULT_CSS = """
    _RepeatButton {
        height: 1;
        width: auto;
        min-width: 7;
        padding: 0 1;
    }
    _RepeatButton:hover {
        background: $accent 30%;
    }
    """

    repeat_mode: reactive[RepeatMode] = reactive(RepeatMode.OFF)

    def render(self) -> Text:
        theme = get_theme()
        # The mode pair — which icon and label the indicator shows — is
        # resolved by the ``_repeat_button_pair`` engine (native when the
        # Rust extension is built); the widget keeps only the colours.
        kind, text = _repeat_button_pair(self.repeat_mode)
        if kind == "all":
            return Text(text, style=f"bold {theme.primary}")
        if kind == "one":
            return Text(text, style=f"bold {theme.warning}")
        return Text(text, style=theme.muted_text)

    async def on_click(self, event: Click) -> None:
        event.stop()
        app = cast("YTMHostBase", self.app)
        mode = app.queue.cycle_repeat()
        try:
            bar = app.query_one("#playback-bar", PlaybackBar)
            bar.update_repeat(mode)
            app.notify(f"Repeat: {mode.value}", timeout=2)
        except Exception:
            logger.debug("Failed to update repeat mode display on click", exc_info=True)


class _ShuffleButton(Widget):
    """Clickable shuffle indicator.

    Disabled when the current playback context has Shuffle lock on (set
    from the Library page playlist header). Visually dimmed and ignores
    clicks; toggle the lock off in the playlist header to re-enable.
    """

    DEFAULT_CSS = """
    _ShuffleButton {
        height: 1;
        width: auto;
        min-width: 7;
        padding: 0 1;
    }
    _ShuffleButton:hover {
        background: $accent 30%;
    }
    _ShuffleButton.locked {
        text-style: dim;
    }
    _ShuffleButton.locked:hover {
        background: transparent;
    }
    """

    shuffle_on: reactive[bool] = reactive(False)
    locked: reactive[bool] = reactive(False)

    def render(self) -> Text:
        theme = get_theme()
        # The on/off pair — which icon and label the indicator shows — is
        # resolved by the ``_shuffle_button_pair`` engine (native when the
        # Rust extension is built); the widget keeps only the colours.
        kind, text = _shuffle_button_pair(self.shuffle_on)
        if kind == "on":
            return Text(text, style=f"bold {theme.primary}")
        return Text(text, style=theme.muted_text)

    def watch_locked(self, locked: bool) -> None:
        """Apply / remove the .locked class so CSS can dim the widget."""
        if locked:
            self.add_class("locked")
        else:
            self.remove_class("locked")

    async def on_click(self, event: Click) -> None:
        event.stop()
        if self.locked:
            # Shuffle is locked on for this playlist — point the user at the
            # toggle in the playlist header instead of silently no-oping.
            try:
                self.app.notify(
                    "Shuffle is locked for this playlist — toggle Shuffle lock in the playlist header.",
                    severity="warning",
                    timeout=4,
                )
            except Exception:
                pass
            return
        app = cast("YTMHostBase", self.app)
        app.queue.toggle_shuffle()
        app._refresh_queue_page()
        enabled = app.queue.shuffle_enabled
        try:
            bar = app.query_one("#playback-bar", PlaybackBar)
            bar.update_shuffle(enabled)
            state = "on" if enabled else "off"
            app.notify(f"Shuffle: {state}", timeout=2)
        except Exception:
            logger.debug("Failed to update shuffle state display on click", exc_info=True)


# Heart icon for the like indicator. Same character, styled differently
# based on like state (filled accent when liked, muted when not).
_ICON_HEART = "\u2764"  # Heavy Black Heart


class _HeartButton(Widget):
    """Like-state indicator and click-toggle for the currently-playing track."""

    DEFAULT_CSS = """
    _HeartButton {
        height: 1;
        width: 3;
        margin: 0 1;
        content-align: center middle;
    }
    _HeartButton:hover {
        background: $accent 30%;
    }
    """

    # likeStatus value: "LIKE", "DISLIKE", "INDIFFERENT", or "" (unknown).
    like_status: reactive[str] = reactive("")

    def render(self) -> Text:
        theme = get_theme()
        # The like verdict — which colour the heart carries — is resolved
        # by the ``_heart_button_pair`` engine (native when the Rust
        # extension is built); the widget keeps only the colours.
        kind, text = _heart_button_pair(self.like_status)
        if kind == "liked":
            return Text(text, style=f"bold {theme.accent}")
        return Text(text, style=theme.muted_text)

    def on_click(self, event: Click) -> None:
        """Click toggles like via the app's _toggle_like_current."""
        event.stop()
        app = cast("YTMHostBase", self.app)
        try:
            self.run_worker(app._toggle_like_current(), group="toggle-like", exclusive=True)
        except Exception:
            logger.debug("Failed to toggle like from heart click", exc_info=True)


# ── Main playback bar ─────────────────────────────────────────────


def _playback_bar_click_track(button: Any, player_track: Any, queue_track: Any) -> Any:
    """The playback bar's right-click track pick — the ``pick`` engine.

    The pure core of ``PlaybackBar.on_click``: the track whose actions
    a right-click on the bar opens — the right-button gate first
    (``button != 3``, CPython's own ``!=``), then the player-then-queue
    fallback chain (a truthy player track answered first, only then a
    truthy queue track, every falsy shape answering ``None`` — the
    no-message arm the reference's ``if track:`` gate leaves). The
    player/queue attribute reads and the message construction stay
    with the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.playback_bar_click_track``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if button != 3:
        return None
    if player_track:
        return player_track
    if queue_track:
        return queue_track
    return None


class PlaybackBar(Widget):
    """Persistent playback bar showing track info, progress, and controls.

    Layout (2 lines + optional album art):
        Line 1: [art] > Song Title -- Artist Name -- Album       vol  repeat  shuffle
        Line 2: [art]  1:23 [===========>---------] 4:56
    """

    class TrackRightClicked(Message):
        """Emitted when the playback bar area is right-clicked."""

        def __init__(self, track: dict) -> None:
            super().__init__()
            self.track = track

    DEFAULT_CSS = """
    PlaybackBar {
        dock: bottom;
        height: 4;
        background: $playback-bar-bg;
        border-top: solid $border;
    }
    PlaybackBar #pb-outer {
        height: 100%;
        width: 1fr;
    }
    PlaybackBar #pb-art {
        width: 10;
        height: 3;
        margin: 0 1 0 0;
    }
    PlaybackBar #pb-content {
        width: 1fr;
        height: auto;
    }
    PlaybackBar #pb-top-row {
        height: 1;
        width: 1fr;
    }
    PlaybackBar #pb-bottom-row {
        height: 1;
        width: 1fr;
    }
    PlaybackBar #pb-track-info {
        width: 1fr;
    }
    PlaybackBar #pb-progress {
        width: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        from ytm_player.config.settings import get_settings

        settings = get_settings()
        with Horizontal(id="pb-outer"):
            art = AlbumArt(id="pb-art")
            if not settings.ui.album_art:
                art.display = False
            yield art
            with Vertical(id="pb-content"):
                with Horizontal(id="pb-top-row"):
                    yield _TrackInfo(id="pb-track-info")
                    yield _HeartButton(id="pb-heart")
                    yield _VolumeDisplay(id="pb-volume")
                    yield _RepeatButton(id="pb-repeat")
                    yield _ShuffleButton(id="pb-shuffle")
                with Horizontal(id="pb-bottom-row"):
                    yield PlaybackProgress(bar_style=settings.ui.progress_style, id="pb-progress")

    def on_click(self, event: Click) -> None:
        """Right-click on the playback bar opens track actions."""
        # The pick — the button-3 gate and the player-then-queue
        # fallback chain — is resolved by the ``_playback_bar_click_track``
        # engine (native when the Rust extension is built); the message
        # construction stays here. The queue read follows the
        # library-page precedent's defensive shape so a partial host
        # (a player with no queue) never raises where the reference's
        # own short-circuit would not.
        app = cast("YTMHostBase", self.app)
        player = app.player
        player_track = player.current_track if player is not None else None
        queue = getattr(app, "queue", None)
        queue_track = queue.current_track if queue else None
        track = _playback_bar_click_track(event.button, player_track, queue_track)
        if track is not None:
            self.post_message(self.TrackRightClicked(track))

    # ── Public update methods ────────────────────────────────────────

    def update_track(self, track: dict | None) -> None:
        """Update displayed track information."""
        info = self.query_one("#pb-track-info", _TrackInfo)
        art = self.query_one("#pb-art", AlbumArt)

        if track is None:
            info.title = ""
            info.artist = ""
            info.album = ""
            info.is_playing = False
            info.is_paused = False
            art.clear_track()
            return

        info.title = track.get("title", "")
        info.artist = extract_artist(track)
        info.album = track.get("album") or ""
        art.set_track(track.get("thumbnail_url", ""))

    def update_playback_state(self, *, is_playing: bool, is_paused: bool) -> None:
        """Update play/pause state indicators."""
        info = self.query_one("#pb-track-info", _TrackInfo)
        info.is_playing = is_playing
        info.is_paused = is_paused

    def update_position(self, position: float, duration: float | None = None) -> None:
        """Update the progress bar position."""
        progress = self.query_one("#pb-progress", PlaybackProgress)
        progress.update_position(position, duration)

    def update_volume(self, volume: int) -> None:
        """Update the volume display."""
        vol = self.query_one("#pb-volume", _VolumeDisplay)
        vol.volume = volume

    def update_repeat(self, mode: RepeatMode) -> None:
        """Update the repeat mode display."""
        rep = self.query_one("#pb-repeat", _RepeatButton)
        rep.repeat_mode = mode

    def update_shuffle(self, enabled: bool) -> None:
        """Update the shuffle state display."""
        shuf = self.query_one("#pb-shuffle", _ShuffleButton)
        shuf.shuffle_on = enabled

    def refresh_shuffle_lock_state(self) -> None:
        """Re-read shuffle_prefs for the current queue context and dim/un-dim
        the shuffle button accordingly. Called from the Library page when the
        user toggles Shuffle lock or enters a different playlist."""
        try:
            app = cast("YTMHostBase", self.app)
            ctx = app.queue.current_context_id
            locked = bool(app.shuffle_prefs.get(ctx)) if ctx else False
            shuf = self.query_one("#pb-shuffle", _ShuffleButton)
            shuf.locked = locked
        except Exception:
            logger.debug("Failed to refresh shuffle lock state", exc_info=True)

    def update_like_status(self, status: str | None) -> None:
        """Update the heart icon based on the track's likeStatus.

        Accepts 'LIKE', 'DISLIKE', 'INDIFFERENT', None, or an unknown
        string. Anything other than 'LIKE' shows the muted (not-liked)
        state.
        """
        try:
            heart = self.query_one("#pb-heart", _HeartButton)
            heart.like_status = (status or "").upper()
        except Exception:
            logger.debug("Failed to update heart like_status display", exc_info=True)


# ── Interactive footer bar ────────────────────────────────────────


# The seven page actions the footer routes to the caller's page
# navigation (the match statement's first case, its own order).
_PAGE_ACTIONS_FOOTER = (
    "help",
    "library",
    "search",
    "queue",
    "browse",
    "liked_songs",
    "recently_played",
)


def _footer_button_action_route(action: Any) -> str:
    """The footer button's action routing — the ``route`` engine.

    The pure core of ``_FooterButton.on_click``: the match statement's
    verdict — the seven page actions (``help`` / ``library`` /
    ``search`` / ``queue`` / ``browse`` / ``liked_songs`` /
    ``recently_played``) route to the caller's page navigation (the
    caller passes its own action on as the page name), ``play_pause`` /
    ``prev`` / ``next`` route to the transport handlers,
    ``spotify_import`` to the import popup, and any other action routes
    ``none`` — the match falls through and nothing runs. Every
    comparison is CPython's own ``==`` on the string literal, the
    protocol a match statement's literal patterns evaluate through.

    Implemented by the Rust extension
    (``_ytm_player_native.footer_button_action_route``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if action in _PAGE_ACTIONS_FOOTER:
        return "page"
    if action == "play_pause":
        return "play_pause"
    if action == "prev":
        return "prev"
    if action == "next":
        return "next"
    if action == "spotify_import":
        return "spotify_import"
    return "none"


def _footer_button_style_kind(is_active: Any, is_dimmed: Any) -> str:
    """The footer button's style choice — the ``style`` engine.

    The pure core of ``_FooterButton.render``: the three-way style
    choice over the two flags, the reference's own order — an active
    button renders bold primary however dimmed, a dimmed one renders
    dim, anything else muted — every gate the flag's own truthiness.

    Implemented by the Rust extension
    (``_ytm_player_native.footer_button_style_kind``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if is_active:
        return "active"
    if is_dimmed:
        return "dimmed"
    return "muted"


def _footer_button_active_verdict(action: Any, page_name: Any) -> bool:
    """The footer's active-page verdict — the ``active`` engine.

    The pure core of ``FooterBar.set_active_page``: whether the button
    the loop is visiting names the page that just became active —
    ``action == page_name``, CPython's own ``==`` on the two names. The
    ``query_one`` walk and the per-button ``is_active`` write stay with
    the caller.

    Implemented by the Rust extension
    (``_ytm_player_native.footer_button_active_verdict``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    return action == page_name


class _FooterButton(Widget):
    """A clickable footer button."""

    DEFAULT_CSS = """
    _FooterButton {
        height: 1;
        width: auto;
        padding: 0 1;
    }
    _FooterButton:hover {
        background: $accent 30%;
    }
    """

    is_active: reactive[bool] = reactive(False)
    is_dimmed: reactive[bool] = reactive(False)

    def __init__(self, label: str, action: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._action = action

    def render(self) -> Text:
        theme = get_theme()
        # The style kind — active over dimmed over muted, the reference's
        # own order — is resolved by the ``_footer_button_style_kind``
        # engine (native when the Rust extension is built); the widget
        # keeps only the label and the colours.
        kind = _footer_button_style_kind(self.is_active, self.is_dimmed)
        if kind == "active":
            return Text(self._label, style=f"bold {theme.primary}")
        if kind == "dimmed":
            return Text(self._label, style="dim")
        return Text(self._label, style=theme.muted_text)

    async def on_click(self, event: Click) -> None:
        event.stop()
        app = cast("YTMHostBase", self.app)
        # The routing verdict — the match statement's own cases — is
        # resolved by the ``_footer_button_action_route`` engine (native
        # when the Rust extension is built); the awaited handlers, the
        # screen push and the event stopping stay here.
        route = _footer_button_action_route(self._action)
        if route == "page":
            await app.navigate_to(self._action)
        elif route == "play_pause":
            await app._toggle_play_pause()
        elif route == "prev":
            await app._play_previous()
        elif route == "next":
            await app._play_next()
        elif route == "spotify_import":
            from ytm_player.ui.popups.spotify_import import SpotifyImportPopup

            app.push_screen(SpotifyImportPopup())


class FooterBar(Widget):
    """Interactive footer with clickable navigation items."""

    DEFAULT_CSS = """
    FooterBar {
        dock: bottom;
        height: 1;
        background: $background;
    }
    FooterBar #footer-inner {
        height: 1;
        width: 1fr;
    }
    FooterBar #footer-help {
        dock: right;
    }
    """

    # Page actions that get an active-page highlight.
    _PAGE_ACTIONS = {
        "library",
        "search",
        "browse",
        "queue",
        "help",
    }

    def compose(self) -> ComposeResult:
        with Horizontal(id="footer-inner"):
            # Playback controls (icon-only).
            yield _FooterButton("\u23ee", "prev")
            yield _FooterButton("\u23ef", "play_pause")
            yield _FooterButton("\u23ed", "next")
            # Page navigation.
            yield _FooterButton("Library", "library", id="footer-library")
            yield _FooterButton("Search", "search", id="footer-search")
            yield _FooterButton("Browse", "browse", id="footer-browse")
            yield _FooterButton("Queue", "queue", id="footer-queue")
            # Spotify import.
            yield _FooterButton("Import", "spotify_import")
            # Help pushed to far right.
            yield _FooterButton("?", "help", id="footer-help")

    def set_active_page(self, page_name: str) -> None:
        """Highlight the footer button corresponding to the active page."""
        for action in self._PAGE_ACTIONS:
            try:
                btn = self.query_one(f"#footer-{action}", _FooterButton)
                # The active verdict — ``action == page_name`` — is
                # resolved by the ``_footer_button_active_verdict`` engine
                # (native when the Rust extension is built); the query
                # walk and the ``is_active`` write stay here.
                btn.is_active = _footer_button_active_verdict(action, page_name)
            except Exception:
                logger.debug(
                    "Failed to update footer button for action '%s'", action, exc_info=True
                )


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same width-splitting, snippet and
# chrome engines; these bindings override the pure-Python definitions
# above when the extension is importable.  The rebinding must happen
# AFTER the definitions above — assigning the native functions any
# earlier would be clobbered by the ``def`` statements (the lesson that
# produced formatting.py's dead native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_info_widths as _native_track_info_widths,
    )
except ImportError:
    pass
else:

    def _track_info_widths(  # noqa: F811
        bar_width: int, title: str, artist: str
    ) -> tuple[int, int, int]:
        """The title/artist/album truncation widths (native Rust engine)."""
        return _native_track_info_widths(bar_width, title, artist)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        track_info_snippets as _native_track_info_snippets,
    )
except ImportError:
    pass
else:

    def _track_info_snippets(  # noqa: F811
        bar_width: int,
        title: str,
        artist: str,
        album: str,
        is_playing: bool,
        is_paused: bool,
    ) -> list[tuple[str, str]]:
        """The one-line track-info snippets (native Rust engine).

        The whole content walk — the state icon, the width split, the
        three BiDi-rendered fragments and their em-dash separators (or
        the no-track placeholder) — runs inside one native call, with
        only the cached ``bidi_mode`` verdict read here (once per
        render instead of once per fragment).
        """
        from ytm_player.utils.bidi import _get_reorder_enabled

        return _native_track_info_snippets(
            bar_width, title, artist, album, is_playing, is_paused, _get_reorder_enabled()
        )


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        volume_display_text as _native_volume_display_text,
    )
except ImportError:
    pass
else:

    def _volume_display_text(volume: Any) -> str:  # noqa: F811
        """The volume display's content (native Rust engine)."""
        return _native_volume_display_text(volume)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        repeat_button_pair as _native_repeat_button_pair,
    )
except ImportError:
    pass
else:

    def _repeat_button_pair(mode: Any) -> tuple[str, str]:  # noqa: F811
        """The repeat button's content pair (native Rust engine)."""
        return _native_repeat_button_pair(mode)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        shuffle_button_pair as _native_shuffle_button_pair,
    )
except ImportError:
    pass
else:

    def _shuffle_button_pair(shuffle_on: Any) -> tuple[str, str]:  # noqa: F811
        """The shuffle button's content pair (native Rust engine)."""
        return _native_shuffle_button_pair(shuffle_on)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        heart_button_pair as _native_heart_button_pair,
    )
except ImportError:
    pass
else:

    def _heart_button_pair(like_status: Any) -> tuple[str, str]:  # noqa: F811
        """The heart button's content pair (native Rust engine)."""
        return _native_heart_button_pair(like_status)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        footer_button_style_kind as _native_footer_button_style_kind,
    )
except ImportError:
    pass
else:

    def _footer_button_style_kind(is_active: Any, is_dimmed: Any) -> str:  # noqa: F811
        """The footer button's style kind (native Rust engine)."""
        return _native_footer_button_style_kind(is_active, is_dimmed)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        footer_button_action_route as _native_footer_button_action_route,
    )
except ImportError:
    pass
else:

    def _footer_button_action_route(action: Any) -> str:  # noqa: F811
        """The footer button's action routing (native Rust engine)."""
        return _native_footer_button_action_route(action)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        footer_button_active_verdict as _native_footer_button_active_verdict,
    )
except ImportError:
    pass
else:

    def _footer_button_active_verdict(action: Any, page_name: Any) -> bool:  # noqa: F811
        """The footer's active-page verdict (native Rust engine)."""
        return _native_footer_button_active_verdict(action, page_name)


try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        playback_bar_click_track as _native_playback_bar_click_track,
    )
except ImportError:
    pass
else:

    def _playback_bar_click_track(  # noqa: F811
        button: Any, player_track: Any, queue_track: Any
    ) -> Any:
        """The playback bar's right-click track pick (native Rust engine)."""
        return _native_playback_bar_click_track(button, player_track, queue_track)
