"""BiDi text utilities for correct RTL display in terminals.

Terminal BiDi support varies widely:

- **Native BiDi** (Konsole, GNOME Terminal/VTE, mlterm): The terminal
  implements UAX #9 and reorders RTL text automatically.  Manual reordering
  would double-reverse the text.
- **No BiDi** (WezTerm default, Alacritty, Ghostty, Foot): Characters are
  placed left-to-right in memory order.  Arabic/Hebrew text in Unicode
  logical order is still *readable* L-to-R (logical order = reading order),
  so manual reordering actually *breaks* it by reversing the sentence.
- **Partial BiDi** (Kitty): Some reordering at the text-run level.

In practice, manual word reordering is almost never correct:
- On BiDi terminals it double-reverses.
- On non-BiDi terminals the logical order is already readable.

The ``bidi_mode`` setting in ``[ui]`` controls behavior:
- ``"auto"`` (default): detect terminal — passthrough for all known terminals.
- ``"reorder"``: force manual word reordering (UAX #9 L2 rule).
- ``"passthrough"``: never reorder, pass text through as-is.
"""

from __future__ import annotations

import re
import unicodedata

# Matches any character from RTL scripts (Arabic, Hebrew, Thaana, Syriac, N'Ko).
_RTL_RE = re.compile(
    r"[\u0590-\u05FF\u0600-\u06FF\u0700-\u074F\u0750-\u077F"
    r"\u0780-\u07BF\u07C0-\u07FF\u08A0-\u08FF"
    r"\uFB1D-\uFB4F\uFB50-\uFDFF\uFE70-\uFEFF]"
)

# Unicode directional isolate marks (UAX #9, since Unicode 6.3).
LRI = "\u2066"  # LEFT-TO-RIGHT ISOLATE
RLI = "\u2067"  # RIGHT-TO-LEFT ISOLATE
FSI = "\u2068"  # FIRST STRONG ISOLATE
PDI = "\u2069"  # POP DIRECTIONAL ISOLATE

# Cached detection result.
_should_reorder: bool | None = None


def has_rtl(text: str) -> bool:
    """Return True if text contains any RTL script characters."""
    return bool(_RTL_RE.search(text))


def _detect_should_reorder() -> bool:
    """Auto-detect whether manual word reordering is needed.

    Returns True only if the terminal is known to lack BiDi support AND
    is known to need manual reordering.  In practice this returns False
    for all known terminals because:
    - BiDi terminals (Konsole, VTE) handle reordering natively.
    - Non-BiDi terminals (WezTerm, Alacritty, etc.) display logical
      order text readably without reordering.
    """
    # Currently no terminal benefits from manual reordering.
    # This function exists so we can add exceptions if one is found.
    return False


def _get_reorder_enabled() -> bool:
    """Check the bidi_mode setting and return whether to reorder."""
    global _should_reorder
    if _should_reorder is not None:
        return _should_reorder

    try:
        from ytm_player.config.settings import get_settings

        mode = get_settings().ui.bidi_mode
    except Exception:
        mode = "auto"

    if mode == "reorder":
        _should_reorder = True
    elif mode == "passthrough":
        _should_reorder = False
    else:  # "auto"
        _should_reorder = _detect_should_reorder()

    return _should_reorder


def reset_bidi_cache() -> None:
    """Reset the cached detection result (useful after settings change)."""
    global _should_reorder
    _should_reorder = None


# ── Internal reorder logic (only used when bidi_mode="reorder") ──────────


def _char_direction(ch: str) -> str:
    """Return simplified BiDi direction: 'R', 'L', or 'N' (neutral)."""
    bidi = unicodedata.bidirectional(ch)
    if bidi in ("R", "AL", "AN"):
        return "R"
    if bidi == "L":
        return "L"
    return "N"


def _word_direction(word: str) -> str:
    """Return direction from the first strong character in *word*."""
    for ch in word:
        d = _char_direction(ch)
        if d != "N":
            return d
    return "N"


def _paragraph_base_direction(text: str) -> str:
    """UAX #9 rules P2/P3: base direction from first strong character."""
    for ch in text:
        d = _char_direction(ch)
        if d != "N":
            return d
    return "L"


def _do_reorder(text: str) -> str:
    """Perform UAX #9 word-level reordering (L2 reversal rule)."""
    words = text.split()
    if not words:
        return text

    base_dir = _paragraph_base_direction(text)
    base_level = 1 if base_dir == "R" else 0

    levels: list[int] = []
    for word in words:
        wd = _word_direction(word)
        if wd == "R":
            levels.append(1)
        elif wd == "L":
            levels.append(2 if base_level == 1 else 0)
        else:
            levels.append(base_level)

    if not levels:
        return text

    max_level = max(levels)
    indices = list(range(len(words)))

    for level in range(max_level, 0, -1):
        i = 0
        while i < len(indices):
            if levels[indices[i]] >= level:
                j = i
                while j < len(indices) and levels[indices[j]] >= level:
                    j += 1
                indices[i:j] = indices[i:j][::-1]
                i = j
            else:
                i += 1

    return " ".join(words[idx] for idx in indices)


# ── Public API ───────────────────────────────────────────────────────────


def reorder_rtl_line(text: str) -> str:
    """Reorder words in a line for RTL display, if enabled.

    When ``bidi_mode`` is ``"auto"`` or ``"passthrough"``, returns the
    text unchanged.  When ``"reorder"``, applies UAX #9 word-level
    reversal.
    """
    if not text or not has_rtl(text):
        return text
    if not _get_reorder_enabled():
        return text
    return _do_reorder(text)


def isolate_bidi(text: str, *, only_if_rtl: bool = True) -> str:
    """Wrap *text* in FSI...PDI so its directional context cannot leak.

    Use for ANY user-supplied string concatenated with other independent
    strings on the same line (table cells, playback bar fragments, lyric
    lines).  Without isolation, RTL text can bleed into adjacent layout
    in some terminals — appearing as duplicated text at the row edge or
    fragments after the volume display in the playback bar.

    FSI auto-detects the inner direction from the first strong character,
    so pure-Latin titles render LTR, pure-Arabic titles render RTL, mixed
    titles work correctly.

    IMPORTANT: Apply ``isolate_bidi`` AFTER ``truncate``.  FSI and PDI
    are zero display columns but each adds 1 to ``len()``, so truncating
    after wrapping would cut off the closing PDI.

    Args:
        text: The text to isolate.
        only_if_rtl: If True (default), skip the wrap for pure-LTR text
            to keep clipboard output clean.  Set False to always wrap.
    """
    if not text:
        return text
    if only_if_rtl and not has_rtl(text):
        return text
    # Don't double-wrap.
    if text.startswith(FSI) and text.endswith(PDI):
        return text
    return f"{FSI}{text}{PDI}"


def wrap_rtl_line(text: str, width: int) -> str:
    """Pre-wrap RTL text for multi-line display, with optional reordering.

    When reordering is disabled (default), simply returns the text as-is
    so the terminal or reader handles directionality naturally.
    """
    if not text or not has_rtl(text):
        return text
    if not _get_reorder_enabled():
        return text

    if len(text) <= width or width <= 0:
        return _do_reorder(text)

    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current_words: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        needed = word_len if not current_words else current_len + 1 + word_len
        if needed <= width:
            current_words.append(word)
            current_len = needed
        else:
            if current_words:
                lines.append(" ".join(current_words))
            current_words = [word]
            current_len = word_len

    if current_words:
        lines.append(" ".join(current_words))

    return "\n".join(_do_reorder(line) for line in lines)


# ── Native (Rust/PyO3) implementation, if available ──────────────────────
# The Rust extension implements the same functions; these bindings override
# the pure-Python definitions above when the extension is importable.


try:
    from ._ytm_player_native import (  # type: ignore[import-not-found]
        has_rtl as _native_has_rtl,
        isolate_bidi as _native_isolate_bidi,
        reorder_line as _native_reorder_line,
        wrap_line as _native_wrap_line,
    )
except ImportError:
    pass
else:

    has_rtl = _native_has_rtl  # type: ignore[assignment]
    isolate_bidi = _native_isolate_bidi  # type: ignore[assignment]
    # Internal symbol is part of this module's test surface.
    _do_reorder = _native_reorder_line  # type: ignore[assignment]

    def reorder_rtl_line(text: str) -> str:  # noqa: F811
        """Reorder words in a line for RTL display, if enabled (native)."""
        if not text or not has_rtl(text):
            return text
        if not _get_reorder_enabled():
            return text
        return _native_reorder_line(text)

    def wrap_rtl_line(text: str, width: int) -> str:  # noqa: F811
        """Pre-wrap RTL text for multi-line display, with reordering (native)."""
        if not text or not has_rtl(text):
            return text
        if not _get_reorder_enabled():
            return text
        return _native_wrap_line(text, width)
