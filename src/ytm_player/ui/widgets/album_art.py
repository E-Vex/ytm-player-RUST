"""Album art display widget for the terminal.

Downloads thumbnails and renders them as colored Unicode half-blocks (▀).
Falls back to a styled placeholder while loading or if the image cannot
be fetched.

The render engines — the WCAG colour math behind the placeholder's
note glyph (``_relative_luminance``, ``_contrast_ratio`` and
``_readable_on``), the half-block cell extraction from the resized
thumbnail's raw RGB buffer (``_half_block_cells``) and the two
no-image row plans (``_placeholder_rows`` and ``_empty_rows``) — and
the module-level art cache's LRU bookkeeping (``_ArtCacheCore``: the
hit promotion, the insert that keeps an existing key's position, the
overflow eviction, the guarded delete) are dual-implemented: the Rust
extension (``_ytm_player_native.wcag_relative_luminance`` & co.,
``_ytm_player_native.ArtCacheCore``) implements the engines when it
is built, and the pure-Python transcriptions below run
otherwise. The widget itself stays glue: it downloads and decodes the
thumbnail (Pillow), resizes it, and assembles the Rich ``Text`` from
the segments the engines compute.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from io import BytesIO
from typing import Any, Protocol, cast
from urllib.request import urlopen

from PIL import Image
from rich.color import Color
from rich.style import Style
from rich.text import Text
from textual.events import Resize
from textual.reactive import reactive
from textual.widget import Widget

from ytm_player.ui.theme import get_theme

logger = logging.getLogger(__name__)

# Unicode block characters.
_BLOCK_TOP = "\u2580"  # Upper half block ▀
_BLOCK_BOTTOM = "\u2584"  # Lower half block ▄
_BLOCK_FULL = "\u2588"  # Full block █
_NOTE = "\u266b"  # Beamed eighth notes ♫

# Module-level thumbnail cache: URL -> rendered Text. The LRU
# bookkeeping lives in an ``ArtCacheCore`` — the Rust engine from the
# native extension (imported above) when it is built, the pure-Python
# ``_ArtCacheCore`` transcription below otherwise.
_ART_CACHE_MAX = 20

# Try to load the native Rust art-cache engine via PyO3. If the compiled
# extension is unavailable (e.g. during development without a Rust
# toolchain), fall back to the pure-Python core defined below.
try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        ArtCacheCore as _NativeArtCacheCore,
    )
except ImportError:
    _NativeArtCacheCore = None

_DOWNLOAD_TIMEOUT = 5


def _relative_luminance(hex_color: str) -> float | None:
    """WCAG relative luminance of a six-digit hex colour — the engine.

    The pure half of the placeholder's note-glyph contrast math:
    ``None`` for anything that is not a six-digit hex colour (``#rrggbb``
    and ``rrggbb``; terminal-dependent colours have no RGB value the
    terminal is bound to, so they are not guessed at), otherwise the
    sRGB linearisation ``0.2126 R' + 0.7152 G' + 0.0722 B'`` with each
    channel either ``c / 12.92`` below the 0.04045 knee or
    ``((c + 0.055) / 1.055) ** 2.4`` above it.

    Implemented by the Rust extension
    (``_ytm_player_native.wcag_relative_luminance``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    hex_str = hex_color.strip().lstrip("#")
    if len(hex_str) != 6:
        return None
    try:
        channels = [int(hex_str[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return None
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast_ratio(luminance_a: float, luminance_b: float) -> float:
    """The WCAG contrast ratio — the engine.

    The pure half of the note-glyph pick: ``(lighter + 0.05) /
    (darker + 0.05)`` between the two luminances.

    Implemented by the Rust extension
    (``_ytm_player_native.wcag_contrast_ratio``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    lighter, darker = max(luminance_a, luminance_b), min(luminance_a, luminance_b)
    return (lighter + 0.05) / (darker + 0.05)


def _readable_on(bg_hex: str, first: str, *others: str) -> str:
    """Return the candidate colour with the highest contrast ratio against *bg_hex* — the engine.

    The placeholder's note glyph used to hardcode ``theme.foreground`` as
    its colour regardless of the background it sits on, which goes nearly
    invisible on a light or pastel ``primary``. Neither candidate is
    assumed to be the light or the dark one: on a light theme the
    foreground is dark and the background light, the reverse of the
    stock theme. *first* wins ties and is kept whenever the comparison
    isn't possible, i.e. the background or any candidate is not a
    six-digit hex colour (with or without ``#``).

    Implemented by the Rust extension
    (``_ytm_player_native.wcag_readable_on``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    background = _relative_luminance(bg_hex)
    candidates = (first, *others)
    luminances = [_relative_luminance(c) for c in candidates]
    if background is None or any(lum is None for lum in luminances):
        return first
    best, best_ratio = first, 0.0
    for candidate, luminance in zip(candidates, luminances):
        ratio = _contrast_ratio(cast(float, luminance), background)
        if ratio > best_ratio:
            best, best_ratio = candidate, ratio
    return best


def _half_block_cells(
    pixels: bytes, width: int, height: int
) -> list[tuple[int, int, int, int, int, int]]:
    """The per-cell top/bottom RGB pairs — the thumbnail pixel walk engine.

    The pure half of ``_image_to_half_blocks``: one
    ``(top_r, top_g, top_b, bottom_r, bottom_g, bottom_b)`` tuple per
    character cell, row-major, read from the resized thumbnail's raw RGB
    buffer — ``PIL.Image.tobytes()`` on an RGB-mode image, where
    ``getpixel((x, y))`` is exactly ``bytes[(y * width + x) * 3 : + 3]``,
    so this buffer walk agrees with the per-pixel ``getpixel`` walk cell
    for cell. Each cell joins the pixel rows ``2 * row`` (top) and
    ``2 * row + 1`` (bottom) at column ``col``. A buffer whose length
    is not exactly ``width * height * 2 * 3`` raises ``ValueError``.

    Implemented by the Rust extension
    (``_ytm_player_native.album_art_half_block_cells``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    if len(pixels) != width * height * 2 * 3:
        raise ValueError(
            f"pixel buffer is {len(pixels)} bytes; "
            f"{width}x{height} half-block cells need {width * height * 2 * 3}"
        )
    cells: list[tuple[int, int, int, int, int, int]] = []
    for row in range(height):
        top_base = (row * 2) * width
        bottom_base = (row * 2 + 1) * width
        for col in range(width):
            top = (top_base + col) * 3
            bottom = (bottom_base + col) * 3
            cells.append(
                (
                    pixels[top],
                    pixels[top + 1],
                    pixels[top + 2],
                    pixels[bottom],
                    pixels[bottom + 1],
                    pixels[bottom + 2],
                )
            )
    return cells


def _placeholder_rows(
    width: int, height: int, accent: str, bg: str, note_color: str
) -> list[list[tuple[str, str]]]:
    """The placeholder-box row plan — the no-thumbnail render engine.

    The pure half of ``_render_placeholder``: per row, the ``(text,
    style)`` segments to append. Row 0 is the lower-half-block strip
    (``▄`` × width, ``accent on bg``); the last row is the
    upper-half-block strip (``▀`` × width, the same style) — checked in
    that order, so a one-row box is all lower-half blocks and a two-row
    box never shows a middle; the middle row (``height // 2``) is the
    note row — ``█`` × ``(width - 1) // 2``, the ``♫`` glyph as
    ``bold {note_color} on {accent}``, then ``█`` × the rest; every
    other row is ``█`` × width in the accent.

    Implemented by the Rust extension
    (``_ytm_player_native.album_art_placeholder_rows``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    rows: list[list[tuple[str, str]]] = []
    for row in range(height):
        if row == 0:
            rows.append([(_BLOCK_BOTTOM * width, f"{accent} on {bg}")])
        elif row == height - 1:
            rows.append([(_BLOCK_TOP * width, f"{accent} on {bg}")])
        elif row == height // 2:
            pad = (width - 1) // 2
            rows.append(
                [
                    (_BLOCK_FULL * pad, accent),
                    (_NOTE, f"bold {note_color} on {accent}"),
                    (_BLOCK_FULL * (width - pad - 1), accent),
                ]
            )
        else:
            rows.append([(_BLOCK_FULL * width, accent)])
    return rows


def _empty_rows(width: int, height: int, muted: str) -> list[list[tuple[str, str]]]:
    """The empty-state row plan — the no-track render engine.

    The pure half of ``_render_empty``: per row, the ``(text, style)``
    segments to append, every one styled with the muted border colour.
    The middle row (``height // 2`` — row 0 itself for a one-row box) is
    the note row — ``(width - 1) // 2`` spaces, the ``♫`` glyph, then
    the remaining spaces; every other row is width spaces.

    Implemented by the Rust extension
    (``_ytm_player_native.album_art_empty_rows``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    rows: list[list[tuple[str, str]]] = []
    for row in range(height):
        if row == height // 2:
            pad = (width - 1) // 2
            rows.append(
                [
                    (" " * pad, muted),
                    (_NOTE, muted),
                    (" " * (width - pad - 1), muted),
                ]
            )
        else:
            rows.append([(" " * width, muted)])
    return rows


class ArtCacheLike(Protocol):
    """The art-cache LRU-bookkeeping contract both implementations satisfy.

    The widget calls these from its download worker; the renders are
    held as the very objects the renderer assembled (never copied).
    """

    def get(self, url: str) -> Text | None: ...
    def insert(self, url: str, value: Text) -> tuple[str, Text] | None: ...
    def remove(self, url: str) -> bool: ...
    def clear(self) -> None: ...
    def length(self) -> int: ...


def _new_art_cache() -> ArtCacheLike:
    """Build the best available album-art cache core."""
    if _NativeArtCacheCore is not None:
        return _NativeArtCacheCore(_ART_CACHE_MAX)
    return _ArtCacheCore(_ART_CACHE_MAX)


class _ArtCacheCore:
    """Pure-Python album-art cache LRU engine (fallback for the native
    Rust core).

    A transcription of the reference's module-level ``OrderedDict``
    bookkeeping: the entries are ordered least-recent first, ``get``
    fuses the ``in`` probe + ``move_to_end`` + the value read into one
    hit promotion (the cache never holds ``None`` — the renderer always
    returns a ``Text`` — so ``None`` answers a miss), ``insert`` fuses
    the ``__setitem__`` store with the single-pop overflow eviction
    (an existing key's value updated in place, its position KEPT — this
    engine never promotes on insert; the shuffle-preferences engine
    does, because *its* reference promoted — answering the evicted
    pair), and ``remove`` is the resize invalidation's guarded delete.
    """

    def __init__(self, max_entries: int) -> None:
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._max_entries = max_entries

    def get(self, url: Any) -> Any:
        if url in self._cache:
            self._cache.move_to_end(url)
            return self._cache[url]
        return None

    def insert(self, url: Any, value: Any) -> tuple[Any, Any] | None:
        self._cache[url] = value
        if len(self._cache) > self._max_entries:
            return self._cache.popitem(last=False)
        return None

    def remove(self, url: Any) -> bool:
        if url in self._cache:
            del self._cache[url]
            return True
        return False

    def clear(self) -> None:
        self._cache.clear()

    def length(self) -> int:
        return len(self._cache)


# The module-level art cache — the factory picks the native Rust core
# when the extension is built, the pure-Python transcription otherwise.
_ART_CACHE: ArtCacheLike = _new_art_cache()


class AlbumArt(Widget):
    """Displays album art in the terminal using half-block Unicode rendering.

    When Pillow is installed, thumbnails are downloaded and rendered as
    colored ``▀`` characters (two pixel rows per character row).  Without
    Pillow the widget shows a styled placeholder.
    """

    DEFAULT_CSS = """
    AlbumArt {
        width: 12;
        height: 3;
        min-width: 8;
        min-height: 3;
    }
    """

    has_track: reactive[bool] = reactive(False)
    thumbnail_url: reactive[str] = reactive("")

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._rendered: Text | None = None

    # ── Reactive watchers ─────────────────────────────────────────────

    def watch_has_track(self, value: bool) -> None:
        if not value:
            self._rendered = None
        self.refresh()

    def watch_thumbnail_url(self, value: str) -> None:
        """When thumbnail URL changes, fetch and render the image."""
        if value:
            self.run_worker(self._load_thumbnail(value), exclusive=True, group="album-art")
        else:
            self._rendered = None
            self.refresh()

    # ── Thumbnail loading ─────────────────────────────────────────────

    async def _load_thumbnail(self, url: str) -> None:
        """Download thumbnail and convert to half-block art."""
        import asyncio

        # Check cache first — the hit promotes the URL to most-recent.
        cached = _ART_CACHE.get(url)
        if cached is not None:
            self._rendered = cached
            self.refresh()
            return

        try:
            img_bytes = await asyncio.to_thread(self._download, url)
            w = self.size.width
            h = self.size.height
            if w < 2 or h < 1:
                return
            rendered = await asyncio.to_thread(self._image_to_half_blocks, img_bytes, w, h)

            # Cache it.
            _ART_CACHE.insert(url, rendered)

            self._rendered = rendered
            self.refresh()
        except Exception:
            logger.debug("Failed to load album art from %s", url, exc_info=True)
            self._rendered = None
            self.refresh()

    @staticmethod
    def _download(url: str) -> bytes:
        """Download image bytes (runs in thread)."""
        with urlopen(url, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            return resp.read()

    # ── Half-block rendering ──────────────────────────────────────────

    @staticmethod
    def _image_to_half_blocks(img_bytes: bytes, width: int, height: int) -> Text:
        """Convert image to colored half-block characters.

        Each character cell represents 2 vertical pixels using the upper
        half block (▀) with foreground = top pixel, background = bottom pixel.
        """
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        # Each char = 1 pixel wide, 2 pixels tall.
        pixel_w = width
        pixel_h = height * 2
        img = img.resize((pixel_w, pixel_h), Image.Resampling.LANCZOS)

        # The resized RGB image's own bytes — the engine's buffer walk is
        # the getpixel walk (getpixel((x, y)) on an RGB-mode image is
        # exactly bytes[(y * pixel_w + x) * 3 : + 3]).
        cells = _half_block_cells(img.tobytes(), width, height)

        result = Text()
        for row in range(height):
            if row > 0:
                result.append("\n")
            base = row * width
            for col in range(width):
                tr, tg, tb, br, bg, bb = cells[base + col]
                style = Style(
                    color=Color.from_rgb(tr, tg, tb),
                    bgcolor=Color.from_rgb(br, bg, bb),
                )
                result.append("\u2580", style=style)
        return result

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self) -> Text:
        w = self.size.width
        h = self.size.height

        if self._rendered is not None and self.has_track:
            return self._rendered

        if not self.has_track or w < 3 or h < 1:
            return self._render_empty(w, h)

        return self._render_placeholder(w, h)

    def _render_empty(self, w: int, h: int) -> Text:
        """Render an empty/muted placeholder."""
        result = Text()
        muted = get_theme().border
        for i, row in enumerate(_empty_rows(w, h, muted)):
            if i > 0:
                result.append("\n")
            for text, style in row:
                result.append(text, style=style)
        return result

    def _render_placeholder(self, w: int, h: int) -> Text:
        """Render a colored placeholder box with a music note."""
        theme = get_theme()
        accent = theme.primary
        bg = theme.playback_bar_bg
        note_color = _readable_on(accent, theme.foreground, theme.background)

        result = Text()
        for i, row in enumerate(_placeholder_rows(w, h, accent, bg, note_color)):
            if i > 0:
                result.append("\n")
            for text, style in row:
                result.append(text, style=style)

        return result

    def on_resize(self, event: Resize) -> None:
        """Re-render on resize if we have a cached image for the current URL."""
        if not self.has_track or not self.thumbnail_url:
            return
        url = self.thumbnail_url
        # If the URL is in cache, invalidate and re-fetch at new size.
        _ART_CACHE.remove(url)
        self.run_worker(self._load_thumbnail(url), exclusive=True, group="album-art")

    def set_track(self, thumbnail_url: str = "") -> None:
        """Update the widget for a new track."""
        self.thumbnail_url = thumbnail_url
        self.has_track = True

    def clear_track(self) -> None:
        """Clear the current track display."""
        self._rendered = None
        self.has_track = False
        self.thumbnail_url = ""


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same render engines; these bindings
# override the pure-Python definitions above when the extension is
# importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        album_art_empty_rows as _native_album_art_empty_rows,
    )
    from ytm_player.utils._ytm_player_native import (
        album_art_half_block_cells as _native_album_art_half_block_cells,
    )
    from ytm_player.utils._ytm_player_native import (
        album_art_placeholder_rows as _native_album_art_placeholder_rows,
    )
    from ytm_player.utils._ytm_player_native import (
        wcag_contrast_ratio as _native_wcag_contrast_ratio,
    )
    from ytm_player.utils._ytm_player_native import (
        wcag_readable_on as _native_wcag_readable_on,
    )
    from ytm_player.utils._ytm_player_native import (
        wcag_relative_luminance as _native_wcag_relative_luminance,
    )
except ImportError:
    pass
else:

    def _relative_luminance(hex_color: str) -> float | None:  # noqa: F811
        """The WCAG relative luminance (native Rust engine)."""
        return _native_wcag_relative_luminance(hex_color)

    def _contrast_ratio(luminance_a: float, luminance_b: float) -> float:  # noqa: F811
        """The WCAG contrast ratio (native Rust engine)."""
        return _native_wcag_contrast_ratio(luminance_a, luminance_b)

    def _readable_on(bg_hex: str, first: str, *others: str) -> str:  # noqa: F811
        """The highest-contrast candidate colour pick (native Rust engine)."""
        return _native_wcag_readable_on(bg_hex, [first, *others])

    def _half_block_cells(  # noqa: F811
        pixels: bytes, width: int, height: int
    ) -> list[tuple[int, int, int, int, int, int]]:
        """The per-cell top/bottom RGB pairs (native Rust engine)."""
        return _native_album_art_half_block_cells(pixels, width, height)

    def _placeholder_rows(  # noqa: F811
        width: int, height: int, accent: str, bg: str, note_color: str
    ) -> list[list[tuple[str, str]]]:
        """The placeholder-box row plan (native Rust engine)."""
        return _native_album_art_placeholder_rows(width, height, accent, bg, note_color)

    def _empty_rows(  # noqa: F811
        width: int, height: int, muted: str
    ) -> list[list[tuple[str, str]]]:
        """The empty-state row plan (native Rust engine)."""
        return _native_album_art_empty_rows(width, height, muted)
