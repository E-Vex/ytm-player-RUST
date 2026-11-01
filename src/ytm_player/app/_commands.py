"""Custom command palette providers for ytm-player.

The provider's pure core — the command catalog, the ``(name, help)``
list every palette dispatch walks — is dual-implemented: implemented
by the Rust extension (``_ytm_player_native.command_catalog``) when it
is built and by the pure-Python transcription below otherwise.  The
provider itself stays Textual glue: it resolves each catalog entry's
callback on the host and builds the matcher-driven hit objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from textual.command import DiscoveryHit, Hit, Hits, Provider

if TYPE_CHECKING:
    from ytm_player.app._app import YTMPlayerApp


def _command_catalog() -> list[tuple[str, str]]:
    """The command palette catalog: ``(name, help)`` per ytm-player command.

    The pure core of ``YTMCommandProvider`` — the catalog behind both
    the discovery list and the query search, in dispatch order.  Names
    must stay unique (the callback resolver is name-keyed) and in sync
    with ``_command_callback``'s resolvers.

    Implemented by the Rust extension
    (``_ytm_player_native.command_catalog``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    return [
        ("Theme: Set Current as Default", "Save the active theme to config.toml"),
    ]


def _command_callback(host: YTMPlayerApp, name: str) -> Callable[[], None]:
    """Resolve the callback the catalog entry *name* dispatches to on *host*.

    A catalog name without a resolver is a programming error — the
    catalog (on either backend) and this resolver drifted — so it
    raises ``KeyError`` instead of silently skipping the command.
    """
    if name == "Theme: Set Current as Default":
        return host.action_set_current_theme_as_default
    raise KeyError(name)


class YTMCommandProvider(Provider):
    """Custom command provider for ytm-player specific actions."""

    @property
    def _host(self) -> YTMPlayerApp:
        return self.app  # type: ignore[return-value]

    async def discover(self) -> Hits:
        """Yield discovery hits for ytm-player commands."""
        for name, help_text in _command_catalog():
            yield DiscoveryHit(
                name,
                _command_callback(self._host, name),
                help=help_text,
            )

    async def search(self, query: str) -> Hits:
        """Fuzzy search ytm-player commands."""
        matcher = self.matcher(query)
        # Callbacks resolve for every entry up front — exactly as lazy as
        # the reference's list build, before any matching happens.
        commands = [
            (name, _command_callback(self._host, name), help_text)
            for name, help_text in _command_catalog()
        ]
        for name, callback, help_text in commands:
            if (match := matcher.match(name)) > 0:
                yield Hit(
                    match,
                    matcher.highlight(name),
                    callback,
                    help=help_text,
                )


# ── Native (Rust/PyO3) override, if available ──────────────────────────
# The Rust extension implements the same catalog; this binding replaces the
# pure-Python definition above when the extension is importable.  The
# rebinding must happen AFTER the definition above — assigning the native
# function any earlier would be clobbered by the ``def`` statement.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        command_catalog as _native_command_catalog,
    )
except ImportError:
    pass
else:
    _command_catalog = _native_command_catalog  # type: ignore[assignment]
