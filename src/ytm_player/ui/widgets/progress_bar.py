"""Playback progress bar widget with click-to-seek and scroll-to-seek.

The render geometry — the progress fraction (``_progress_fraction``),
the time-label/bar-width metrics (``_bar_metrics``), the bar cell
computation with the scroll-seek marker (``_bar_cells``) and the
click-to-seek column mapping (``_x_to_seconds``) — is dual-implemented:
the Rust extension (``_ytm_player_native.progress_fraction`` & co.)
implements the engines when it is built, and the pure-Python
transcriptions below run otherwise. The widget itself stays Textual
glue: it feeds its reactive state through the engines and styles the
cells the engines compute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from rich.text import Text
from textual.events import Click, MouseScrollDown, MouseScrollUp
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget

from ytm_player.ui.theme import get_theme
from ytm_player.utils.formatting import format_duration

if TYPE_CHECKING:
    from ytm_player.app._base import YTMHostBase

# Seconds to jump per scroll tick.
_SCROLL_STEP = 3.0

# Delay (seconds) after the last scroll event before the seek fires.
_SCROLL_COMMIT_DELAY = 0.6


def _progress_fraction(position: float, duration: float) -> float:
    """Progress as a fraction between 0.0 and 1.0 — the ``progress`` engine.

    The pure half of ``PlaybackProgress.progress``: ``0.0`` for a
    non-positive duration, otherwise ``position / duration`` clamped to
    ``[0.0, 1.0]``.

    Implemented by the Rust extension
    (``_ytm_player_native.progress_fraction``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    if duration <= 0:
        return 0.0
    return min(1.0, max(0.0, position / duration))


def _bar_metrics(
    position: float,
    duration: float,
    preview_position: float | None,
    width: int,
) -> tuple[str, str, int]:
    """The bar's layout metrics — the ``_bar_metrics`` engine.

    The pure half of the widget's bar layout: ``(time_prefix,
    time_suffix, bar_width)``, where the prefix/suffix are the
    elapsed/total labels (the pending scroll-seek preview standing in
    for the position) and the bar width is what the widget width has
    left once both labels' lengths are reserved, floored at 0.

    Implemented by the Rust extension
    (``_ytm_player_native.progress_bar_metrics``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    pos = position if preview_position is None else preview_position
    elapsed_str = format_duration(int(pos))
    total_str = format_duration(int(duration))
    time_prefix = f" {elapsed_str} "
    time_suffix = f" {total_str} "
    reserved = len(time_prefix) + len(time_suffix)
    bar_width = max(0, width - reserved)
    return time_prefix, time_suffix, bar_width


def _bar_cells(
    bar_width: int,
    progress: float,
    preview_position: float | None,
    duration: float,
) -> tuple[int, int, int]:
    """The bar's cell counts — the ``render`` composition engine.

    The pure half of ``PlaybackProgress.render``: the filled and empty
    cell counts (``int(bar_width * progress)``) and the scroll-seek
    preview's marker column — ``-1`` when no preview is pending or the
    bar has no width, otherwise ``min(int(bar_width * preview_frac),
    bar_width - 1)`` with ``preview_frac`` the preview position clamped
    against the duration.

    Implemented by the Rust extension
    (``_ytm_player_native.progress_bar_cells``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    filled_count = int(bar_width * progress)
    empty_count = bar_width - filled_count
    if preview_position is not None and bar_width > 0:
        preview_frac = min(1.0, max(0.0, preview_position / duration)) if duration > 0 else 0.0
        marker_col = min(int(bar_width * preview_frac), bar_width - 1)
    else:
        marker_col = -1
    return filled_count, empty_count, marker_col


def _x_to_seconds(
    x: int,
    position: float,
    duration: float,
    preview_position: float | None,
    width: int,
) -> float | None:
    """Map a widget-relative column to a seek position — the click engine.

    The pure half of the widget's click-to-seek: the column translated
    through the same metrics the render uses (the time prefix eats the
    leading columns), clamped to ``[0, bar_width - 1]`` and scaled to
    ``fraction * duration``. ``None`` — no seek — for a non-positive
    duration or a bar with no width.

    Implemented by the Rust extension
    (``_ytm_player_native.progress_x_to_seconds``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if duration <= 0:
        return None
    _, _, bar_width = _bar_metrics(position, duration, preview_position, width)
    if bar_width <= 0:
        return None

    pos = position if preview_position is None else preview_position
    elapsed_str = format_duration(int(pos))
    prefix_len = len(f" {elapsed_str} ")

    bar_x = x - prefix_len
    if bar_x < 0:
        bar_x = 0
    if bar_x >= bar_width:
        bar_x = bar_width - 1

    fraction = bar_x / bar_width
    return fraction * duration


class PlaybackProgress(Widget):
    """Displays a playback progress bar with elapsed/total time.

    Renders as: 1:23 [==========>-----------] 4:56

    Supports block style (default) and line style via the ``style`` parameter.
    Clicking anywhere on the bar seeks to that position.  Scrolling up/down
    adjusts a preview marker that commits after a short pause.
    """

    DEFAULT_CSS = """
    PlaybackProgress {
        height: 1;
        width: 1fr;
    }
    """

    position: reactive[float] = reactive(0.0)
    duration: reactive[float] = reactive(0.0)

    BLOCK_FILLED = "\u2588"  # Full block
    BLOCK_EMPTY = "\u2591"  # Light shade
    MARKER = "\u2503"  # Heavy vertical bar (seek preview)
    LINE_FILLED = "\u2501"  # Box heavy horizontal
    LINE_HEAD = "\u25cf"  # Black circle
    LINE_EMPTY = "\u2500"  # Box light horizontal

    def __init__(
        self,
        *,
        bar_style: str = "block",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._bar_style = bar_style

        # Scroll-seek preview state.
        self._preview_position: float | None = None
        self._scroll_timer: Timer | None = None

    @property
    def progress(self) -> float:
        """Progress as a fraction between 0.0 and 1.0."""
        return _progress_fraction(self.position, self.duration)

    # ── Rendering ─────────────────────────────────────────────────

    def render(self) -> Text:
        time_prefix, time_suffix, bar_width = _bar_metrics(
            self.position, self.duration, self._preview_position, self.size.width
        )
        filled_count, empty_count, marker_col = _bar_cells(
            bar_width, self.progress, self._preview_position, self.duration
        )

        theme = get_theme()
        filled_color = theme.progress_filled
        empty_color = theme.progress_empty
        time_color = theme.secondary
        marker_color = theme.foreground

        result = Text()
        result.append(time_prefix, style=time_color)

        if marker_col >= 0:
            # Show a marker at the preview position.
            for i in range(bar_width):
                if i == marker_col:
                    result.append(self.MARKER, style=f"bold {marker_color}")
                elif i < filled_count:
                    result.append(
                        self.BLOCK_FILLED if self._bar_style == "block" else self.LINE_FILLED,
                        style=filled_color,
                    )
                else:
                    result.append(
                        self.BLOCK_EMPTY if self._bar_style == "block" else self.LINE_EMPTY,
                        style=empty_color,
                    )
        else:
            # Normal rendering (no preview).
            if self._bar_style == "line":
                if filled_count > 0:
                    filled_str = self.LINE_FILLED * (filled_count - 1) + self.LINE_HEAD
                else:
                    filled_str = ""
                empty_str = self.LINE_EMPTY * empty_count
            else:
                filled_str = self.BLOCK_FILLED * filled_count
                empty_str = self.BLOCK_EMPTY * empty_count

            result.append(filled_str, style=filled_color)
            result.append(empty_str, style=empty_color)

        result.append(time_suffix, style=time_color)
        return result

    # ── Click to seek ─────────────────────────────────────────────

    def on_click(self, event: Click) -> None:
        """Seek to the clicked position on the bar."""
        event.stop()
        target = _x_to_seconds(
            event.x, self.position, self.duration, self._preview_position, self.size.width
        )
        if target is not None:
            self._seek_to(target)

    # ── Scroll to seek (with preview marker) ──────────────────────

    def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        """Scroll up → fast-forward preview."""
        event.stop()
        self._scroll_adjust(_SCROLL_STEP)

    def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        """Scroll down → rewind preview."""
        event.stop()
        self._scroll_adjust(-_SCROLL_STEP)

    def _scroll_adjust(self, delta: float) -> None:
        """Move the preview marker by *delta* seconds and reset the commit timer."""
        if self.duration <= 0:
            return

        # Initialise preview from current position on first scroll tick.
        if self._preview_position is None:
            self._preview_position = self.position

        self._preview_position = max(0.0, min(self.duration, self._preview_position + delta))
        self.refresh()

        # Reset the commit timer.
        if self._scroll_timer is not None:
            self._scroll_timer.stop()
        self._scroll_timer = self.set_timer(_SCROLL_COMMIT_DELAY, self._commit_scroll)

    def _commit_scroll(self) -> None:
        """Called when scrolling pauses — seek to the preview position."""
        self._scroll_timer = None
        if self._preview_position is not None:
            target = self._preview_position
            self._preview_position = None
            self._seek_to(target)
        self.refresh()

    # ── Helpers ────────────────────────────────────────────────────

    def _seek_to(self, seconds: float) -> None:
        """Tell the app player to seek to an absolute position."""
        app = cast("YTMHostBase", self.app)
        player = app.player
        if player is not None:
            self.call_later(lambda: app.run_worker(player.seek_absolute(seconds)))

    # ── Public API (unchanged) ────────────────────────────────────

    def update_position(self, position: float, duration: float | None = None) -> None:
        """Update the position and optionally the duration."""
        self.position = position
        if duration is not None:
            self.duration = duration


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same render-geometry engines; these
# bindings override the pure-Python definitions above when the extension
# is importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        progress_bar_cells as _native_progress_bar_cells,
    )
    from ytm_player.utils._ytm_player_native import (
        progress_bar_metrics as _native_progress_bar_metrics,
    )
    from ytm_player.utils._ytm_player_native import (
        progress_fraction as _native_progress_fraction,
    )
    from ytm_player.utils._ytm_player_native import (
        progress_x_to_seconds as _native_progress_x_to_seconds,
    )
except ImportError:
    pass
else:

    def _progress_fraction(position: float, duration: float) -> float:  # noqa: F811
        """The progress fraction (native Rust engine)."""
        return _native_progress_fraction(position, duration)

    def _bar_metrics(  # noqa: F811
        position: float,
        duration: float,
        preview_position: float | None,
        width: int,
    ) -> tuple[str, str, int]:
        """The bar layout metrics (native Rust engine)."""
        return _native_progress_bar_metrics(position, duration, preview_position, width)

    def _bar_cells(  # noqa: F811
        bar_width: int,
        progress: float,
        preview_position: float | None,
        duration: float,
    ) -> tuple[int, int, int]:
        """The bar cell counts and marker column (native Rust engine)."""
        return _native_progress_bar_cells(bar_width, progress, preview_position, duration)

    def _x_to_seconds(  # noqa: F811
        x: int,
        position: float,
        duration: float,
        preview_position: float | None,
        width: int,
    ) -> float | None:
        """The click-column to seconds mapping (native Rust engine)."""
        return _native_progress_x_to_seconds(x, position, duration, preview_position, width)
