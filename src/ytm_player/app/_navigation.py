"""Page navigation mixin for YTMPlayerApp.

The stack bookkeeping is split out of ``NavigationMixin``: the pure
navigation engine — ``resolve_navigation`` (the back/forward stack
machine: snapshotting the page being left, the 20-entry caps, the
same-page-back shortcut, forward-history invalidation and cached-state
restore) and ``purge_dead_playlist_refs`` (dropping every stack/cache
reference to deleted playlists) — is implemented by the Rust extension
(``_ytm_player_native.resolve_navigation`` /
``_ytm_player_native.purge_dead_playlist_refs``) when it is built and
by the pure-Python transcriptions below otherwise. The mixin only reads
the mounted page's fresh nav state, applies the engine's outcome and
does the widget I/O.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from textual.app import ComposeResult
from textual.containers import Container
from textual.widget import Widget
from textual.widgets import Static

from ytm_player.app._base import YTMHostBase
from ytm_player.app._sidebar import sidebar_visible_for_page
from ytm_player.config import Action
from ytm_player.ui.playback_bar import FooterBar

if TYPE_CHECKING:
    from ytm_player.app._base import PageWidget

logger = logging.getLogger(__name__)

# Valid page names.
PAGE_NAMES = (
    "library",
    "search",
    "context",
    "browse",
    "queue",
    "help",
    "liked_songs",
    "recently_played",
)

_MAX_NAV_STACK = 20


# ── Navigation stack engine ──────────────────────────────────────────


def resolve_navigation(
    current_page: str,
    current_kwargs: dict,
    nav_stack: list[tuple[str, dict]],
    forward_stack: list[tuple[str, dict]],
    page_state_cache: dict[str, dict],
    current_page_state: dict | None,
    page_name: str,
    kwargs: dict,
    page_names: tuple[str, ...] = PAGE_NAMES,
    max_stack: int = _MAX_NAV_STACK,
) -> dict:
    """Resolve one navigation request against the back/forward stacks.

    The pure stack machine behind ``NavigationMixin.navigate_to``,
    transcribed verbatim from its bookkeeping:

    - The leaving page's fresh *current_page_state* (``None`` = no
      mounted page / no ``get_nav_state``: leave the cache alone) is
      folded into *page_state_cache* first — truthy state replaces the
      cached entry, empty state drops it — because the stack snapshots
      below read that cache back.
    - ``back`` pops the nav stack (pushing the current page's snapshot
      onto the forward stack); ``forward`` does the reverse. ``back``
      with an empty stack goes to ``library``; ``forward`` with an empty
      stack is a silent no-op.
    - Clicking the page you're already on (no kwargs) pops back instead.
    - Any non-back/forward navigation pushes the current page's snapshot
      onto the nav stack and invalidates the forward history — you can't
      redo a future you didn't take.
    - A switch with empty kwargs restores the target's cached state.

    Inputs are never mutated. The returned ``nav_stack`` /
    ``forward_stack`` / ``page_state_cache`` are fresh containers whose
    payload dicts are the input objects themselves (snapshots and
    restored kwargs are new dicts, like the mixin's originals).

    Returns ``{"action": "switch" | "noop", "reason": str | None,
    "page": str, "kwargs": dict, "nav_stack": list, "forward_stack":
    list, "page_state_cache": dict}`` — *reason* is ``"unknown_page"``,
    ``"no_forward_history"`` or ``"same_page_no_history"`` for a no-op,
    None for a switch. The new state is returned even on a no-op: the
    cache refresh and any stack pop happen before the request is
    validated, so the caller must always apply it.
    """
    cache = dict(page_state_cache)
    if current_page and current_page_state is not None:
        if current_page_state:
            cache[current_page] = current_page_state
        else:
            # An empty fresh state means the user left whatever the
            # cache remembers — drop it rather than restore it later.
            cache.pop(current_page, None)

    nav = list(nav_stack)
    forward = list(forward_stack)

    def _leaving_snapshot() -> dict:
        snap = dict(current_kwargs)
        snap.update(cache.get(current_page, {}))
        return snap

    def _outcome(action: str, reason: str | None) -> dict:
        return {
            "action": action,
            "reason": reason,
            "page": page_name,
            "kwargs": kwargs,
            "nav_stack": nav,
            "forward_stack": forward,
            "page_state_cache": cache,
        }

    # Handle "back" / "forward" navigation via stacks.
    is_back = page_name == "back"
    is_forward = page_name == "forward"
    if is_back:
        if nav:
            prev_page, prev_kwargs = nav.pop()
            # Push the page we're leaving onto the forward stack so
            # the user can come back to it via forward.
            if current_page:
                forward.append((current_page, _leaving_snapshot()))
                if len(forward) > max_stack:
                    forward = forward[-max_stack:]
            page_name = prev_page
            kwargs = prev_kwargs
        else:
            page_name = "library"
    elif is_forward:
        if forward:
            next_page, next_kwargs = forward.pop()
            # Push the page we're leaving back onto the nav stack so
            # the user can return to it via back.
            if current_page:
                nav.append((current_page, _leaving_snapshot()))
                if len(nav) > max_stack:
                    nav = nav[-max_stack:]
            page_name = next_page
            kwargs = next_kwargs
        else:
            # No forward history — silently no-op.
            return _outcome("noop", "no_forward_history")

    if page_name not in page_names:
        return _outcome("noop", "unknown_page")

    same_page_back = False
    if page_name == current_page and not kwargs:
        # Clicking the same page again goes back to the previous page.
        if nav:
            prev_page, prev_kwargs = nav.pop()
            page_name = prev_page
            kwargs = prev_kwargs
            same_page_back = True
        else:
            return _outcome("noop", "same_page_no_history")

    # Push current page onto the nav stack before switching.
    # Skip for back / forward / same-page-back — those already managed
    # the stacks above (or popped, in the same-page case).
    if (
        not is_back
        and not is_forward
        and not same_page_back
        and current_page
        and current_page != page_name
    ):
        nav.append((current_page, _leaving_snapshot()))
        # Cap stack size.
        if len(nav) > max_stack:
            nav = nav[-max_stack:]
        # Browser semantics: any non-back/forward navigation invalidates
        # the forward history (you can't redo a future you didn't take).
        forward = []

    # Restore cached state when no explicit kwargs were given.
    if not kwargs and page_name in cache:
        kwargs = dict(cache[page_name])

    return _outcome("switch", None)


def purge_dead_playlist_refs(
    dead_ids: set[str],
    active_library_playlist_id: str | None,
    nav_stack: list[tuple[str, dict]],
    forward_stack: list[tuple[str, dict]],
    page_state_cache: dict[str, dict],
) -> dict:
    """Drop every navigation reference to the playlists in *dead_ids*.

    The pure half of ``NavigationMixin._purge_playlist_nav_state``: the
    ``library`` cache entry and the stack entries naming a deleted
    playlist — via ``playlist_id`` (Library pages) or ``context_type`` +
    ``context_id`` (playlist context pages opened from search/browse) —
    plus the bare-Library fallback id must all go, or a bare footer-nav
    to Library would reopen the deleted playlist.

    Inputs are never mutated. Returns ``{"active_library_playlist_id":
    str | None, "nav_stack": list, "forward_stack": list,
    "page_state_cache": dict}``.
    """
    # LibraryPage.on_mount falls back to this id when mounted bare.
    active = None if active_library_playlist_id in dead_ids else active_library_playlist_id

    cache = dict(page_state_cache)
    lib_state = cache.get("library")
    if lib_state and lib_state.get("playlist_id") in dead_ids:
        del cache["library"]

    def _is_dead(kw: dict) -> bool:
        # Library entries carry playlist_id; playlist context pages
        # (opened from search/browse) carry context_type + context_id.
        if kw.get("playlist_id") in dead_ids:
            return True
        return kw.get("context_type") == "playlist" and kw.get("context_id") in dead_ids

    nav = [entry for entry in nav_stack if not _is_dead(entry[1])]
    forward = [entry for entry in forward_stack if not _is_dead(entry[1])]

    return {
        "active_library_playlist_id": active,
        "nav_stack": nav,
        "forward_stack": forward,
        "page_state_cache": cache,
    }


# ── Placeholder page widget ─────────────────────────────────────────


class _PlaceholderPage(Widget):
    """Temporary placeholder shown for pages not yet implemented."""

    DEFAULT_CSS = """
    _PlaceholderPage {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
    }
    """

    def __init__(self, page_name: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._page_name = page_name

    def compose(self) -> ComposeResult:
        yield Static(
            f"\n\n  [{self._page_name.upper()}]\n\n"
            f"  This page is not yet implemented.\n"
            f"  Navigate with: g l (library), g s (search), z (queue), ? (help)\n",
            id="placeholder-text",
        )

    async def handle_action(self, action: Action, count: int = 1) -> None:
        """No-op action handler for placeholder pages."""
        pass


class NavigationMixin(YTMHostBase):
    """Page navigation: navigate_to, _create_page, _get_current_page."""

    @property
    def current_page_name(self) -> str:
        return self._current_page

    async def navigate_to(self, page_name: str, **kwargs: Any) -> None:
        """Swap the content of #main-content to a new page.

        Extra *kwargs* are forwarded to the page constructor (e.g.
        ``context_type`` and ``context_id`` for ContextPage).

        Pass ``page_name="back"`` to pop from the back stack (and push the
        current page onto the forward stack), or ``page_name="forward"``
        to do the reverse.

        All stack/cache bookkeeping happens up front in
        :func:`resolve_navigation` — the native Rust engine when the
        extension is built, the pure-Python transcription otherwise.
        This method reads the mounted page's fresh nav state, applies
        the engine's outcome, and does the widget I/O.
        """
        # Read the current page's nav state FIRST — the engine folds it
        # into the cache before the back/forward stack snapshots read it
        # back, so a stale entry there would restore a previous visit's
        # cursor. None means "no mounted page / no get_nav_state" and
        # leaves the cached entry alone; an empty dict means the page
        # deliberately reports no state and drops it.
        current_page_state: dict | None = None
        if self._current_page:
            current_page = self._get_current_page()
            if current_page is not None and hasattr(current_page, "get_nav_state"):
                current_page_state = current_page.get_nav_state()

        outcome = resolve_navigation(
            self._current_page,
            self._current_page_kwargs,
            self._nav_stack,
            self._forward_stack,
            self._page_state_cache,
            current_page_state,
            page_name,
            kwargs,
            page_names=PAGE_NAMES,
            max_stack=_MAX_NAV_STACK,
        )
        self._nav_stack = outcome["nav_stack"]
        self._forward_stack = outcome["forward_stack"]
        self._page_state_cache = outcome["page_state_cache"]

        if outcome["action"] == "noop":
            if outcome["reason"] == "unknown_page":
                logger.warning("Unknown page: %s", outcome["page"])
            return

        page_name = outcome["page"]
        kwargs = outcome["kwargs"]

        container = self.query_one("#main-content", Container)

        # remove_children and mount are async; must await them.
        await container.remove_children()
        page_widget = self._create_page(page_name, **kwargs)
        await container.mount(page_widget)
        self._current_page = page_name
        self._current_page_kwargs = dict(kwargs)
        # A page swap moves the user back into the content pane; reset the
        # keyboard-focus pane so movement keys drive the new page, not a
        # sidebar that was focused on the previous page.
        self._active_pane = "content"

        # Update footer active page indicator.
        try:
            footer = self.query_one("#app-footer", FooterBar)
            footer.set_active_page(page_name)
        except Exception:
            logger.debug("Failed to update footer active page indicator", exc_info=True)

        # Show/hide the header back/forward buttons based on stack state.
        try:
            from ytm_player.ui.header_bar import HeaderBar

            header = self.query_one("#app-header", HeaderBar)
            header.set_back_visible(bool(self._nav_stack))
            header.set_forward_visible(bool(self._forward_stack))
        except Exception:
            logger.exception("Failed to update header back/forward button visibility")

        # Apply per-page playlist sidebar visibility — through the sidebar
        # engine's read, so it runs native when the extension is built.
        sidebar_visible = sidebar_visible_for_page(
            self._sidebar_per_page, page_name, self._sidebar_default
        )
        self._apply_playlist_sidebar(sidebar_visible)

        # Apply global lyrics sidebar visibility.
        self._apply_lyrics_sidebar(self._lyrics_sidebar_open)

        logger.debug("Navigated to page: %s", page_name)

    def _purge_playlist_nav_state(self, dead_ids: set[str]) -> None:
        """Drop cached/stacked navigation state referencing a deleted playlist.

        Without this, a bare footer-nav to Library (empty kwargs → cached
        state applied) would reopen the deleted playlist. The filtering
        itself runs in :func:`purge_dead_playlist_refs` — the native Rust
        engine when the extension is built, the pure-Python transcription
        otherwise; this method applies its outcome and refreshes the UI.
        """
        outcome = purge_dead_playlist_refs(
            dead_ids,
            self._active_library_playlist_id,
            self._nav_stack,
            self._forward_stack,
            self._page_state_cache,
        )
        self._active_library_playlist_id = outcome["active_library_playlist_id"]
        self._nav_stack = outcome["nav_stack"]
        self._forward_stack = outcome["forward_stack"]
        self._page_state_cache = outcome["page_state_cache"]

        # The purge may have emptied a stack — refresh the header buttons so
        # a stale back/forward control doesn't stay clickable.
        try:
            from ytm_player.ui.header_bar import HeaderBar

            header = self.query_one("#app-header", HeaderBar)
            header.set_back_visible(bool(self._nav_stack))
            header.set_forward_visible(bool(self._forward_stack))
        except Exception:
            logger.exception("Failed to update header back/forward button visibility")

    def _create_page(self, page_name: str, **kwargs: Any) -> Widget:
        """Instantiate the widget for a given page name."""
        from ytm_player.ui.pages.browse import BrowsePage
        from ytm_player.ui.pages.context import ContextPage
        from ytm_player.ui.pages.help import HelpPage
        from ytm_player.ui.pages.library import LibraryPage
        from ytm_player.ui.pages.liked_songs import LikedSongsPage
        from ytm_player.ui.pages.queue import QueuePage
        from ytm_player.ui.pages.recently_played import RecentlyPlayedPage
        from ytm_player.ui.pages.search import SearchPage

        page_map: dict[str, type[Widget]] = {
            "library": LibraryPage,
            "search": SearchPage,
            "context": ContextPage,
            "browse": BrowsePage,
            "queue": QueuePage,
            "help": HelpPage,
            "liked_songs": LikedSongsPage,
            "recently_played": RecentlyPlayedPage,
        }
        page_cls = page_map.get(page_name)
        if page_cls is None:
            return _PlaceholderPage(page_name, id=f"page-{page_name}")
        # ContextPage uses unique IDs because back-to-back navigation between
        # contexts (e.g. album → album) can race the previous instance's
        # removal, causing DuplicateIds when the new one mounts.
        if page_name == "context":
            self._context_seq += 1
            return page_cls(id=f"page-context-{self._context_seq}", **kwargs)
        return page_cls(id=f"page-{page_name}", **kwargs)

    def _get_current_page(self) -> PageWidget | None:
        """Return the currently mounted page widget, or None.

        Returned as a ``PageWidget`` (Protocol) so callers can invoke
        ``handle_action`` and ``get_nav_state`` without ``hasattr``
        gymnastics for the type-checker.  All concrete pages mounted
        in ``#main-content`` implement this protocol.
        """
        try:
            container = self.query_one("#main-content", Container)
            children = list(container.children)
            if not children:
                return None
            return cast("PageWidget", children[0])
        except Exception:
            logger.debug("Failed to get current page", exc_info=True)
            return None


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same engines; these bindings replace
# the pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        purge_dead_playlist_refs as _native_purge_dead_playlist_refs,
    )
    from ytm_player.utils._ytm_player_native import (
        resolve_navigation as _native_resolve_navigation,
    )
except ImportError:
    pass
else:
    resolve_navigation = _native_resolve_navigation  # type: ignore[assignment]
    purge_dead_playlist_refs = _native_purge_dead_playlist_refs  # type: ignore[assignment]
