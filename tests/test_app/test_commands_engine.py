"""Tests for the command palette catalog engine.

``_command_catalog`` is the engine behind ``YTMCommandProvider``: the
``(name, help)`` catalog every palette dispatch walks — the discovery
list and the query search both iterate it, resolving each entry's
callback on the host through ``_command_callback``. The pure-Python
transcription lives in ``app/_commands.py``; when the Rust extension is
built the same engine runs natively
(``_ytm_player_native.command_catalog``).

The unit tests below assert the engine's contract on whichever backend
is loaded, plus the provider glue the port introduced (the per-entry
callback resolution, its eager binding, and the loud failure on
catalog/resolver drift). When the extension is built, the parity test
additionally compares the native catalog against the pure-Python
reference (run in a child interpreter with the extension import
blocked) entry for entry, so the two implementations cannot drift.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from ytm_player.app._commands import YTMCommandProvider, _command_callback, _command_catalog

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_THEME_COMMAND = "Theme: Set Current as Default"
_THEME_HELP = "Save the active theme to config.toml"


class RecordingHost:
    """A host recording every attribute read — pins the eager callback
    binding order (the reference resolved every callback while building
    the command list, before any matching)."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    def __getattr__(self, name: str) -> Any:
        self.reads.append(name)
        return lambda: None


def _make_provider(host: Any) -> YTMCommandProvider:
    screen = MagicMock()
    screen.app = host
    return YTMCommandProvider(screen)


class TestCommandCatalog:
    def test_catalog_lists_the_theme_command(self):
        assert _command_catalog() == [(_THEME_COMMAND, _THEME_HELP)]

    def test_entries_are_name_help_string_pairs(self):
        for entry in _command_catalog():
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            name, help_text = entry
            assert type(name) is str
            assert type(help_text) is str

    def test_names_are_unique(self):
        # The names are the palette's dispatch keys — a duplicate would
        # resolve both entries to the same callback and shadow one.
        names = [name for name, _help in _command_catalog()]
        assert len(names) == len(set(names))

    def test_fresh_list_every_call(self):
        first = _command_catalog()
        second = _command_catalog()
        assert first is not second
        assert first == second
        # The entries are immutable tuples, so sharing them between
        # calls is safe — the fallback's tuple literals are interned
        # constants on CPython; only the list is guaranteed fresh.
        assert isinstance(first[0], tuple)


class TestCommandCallback:
    def test_resolves_the_theme_command_to_the_host_action(self):
        host = MagicMock()
        callback = _command_callback(host, _THEME_COMMAND)
        assert callback is host.action_set_current_theme_as_default

    def test_unknown_name_raises_keyerror(self):
        # A catalog name without a resolver is a programming error (the
        # catalog and the resolver drifted) — it must fail loudly, not
        # silently skip the command.
        with pytest.raises(KeyError, match="No Such Command"):
            _command_callback(MagicMock(), "No Such Command")


class TestProviderGlue:
    async def test_discover_yields_one_hit_per_catalog_entry(self):
        provider = _make_provider(MagicMock())
        hits = [hit async for hit in provider.discover()]
        assert len(hits) == len(_command_catalog())
        assert _THEME_COMMAND in str(hits[0].display)
        assert _THEME_HELP in str(hits[0].help)

    async def test_search_matches_the_theme_command(self):
        provider = _make_provider(MagicMock())
        hits = [hit async for hit in provider.search("theme default")]
        assert len(hits) == 1
        assert hits[0].score > 0

    async def test_search_irrelevant_query_yields_nothing(self):
        provider = _make_provider(MagicMock())
        hits = [hit async for hit in provider.search("zzzzzzzzz")]
        assert hits == []

    async def test_callbacks_bind_eagerly_before_any_matching(self):
        # The reference built the whole (name, callback, help) list —
        # resolving every callback — before the first matcher.match()
        # call. A query that matches nothing still reads the actions.
        host = RecordingHost()
        provider = _make_provider(host)
        hits = [hit async for hit in provider.search("zzzzzzzzz")]
        assert hits == []
        assert host.reads == ["action_set_current_theme_as_default"]

    async def test_discover_binds_the_callback_per_entry(self):
        host = RecordingHost()
        provider = _make_provider(host)
        hits = [hit async for hit in provider.discover()]
        assert len(hits) == 1
        assert host.reads == ["action_set_current_theme_as_default"]

    async def test_catalog_entry_without_resolver_fails_loudly(self, monkeypatch):
        # Simulated drift: a catalog (either backend) naming a command
        # the resolver doesn't know must surface its KeyError through
        # the palette machinery, never silently skip the entry.
        monkeypatch.setattr(
            "ytm_player.app._commands._command_catalog",
            lambda: [("Drifted Command", "help")],
        )
        provider = _make_provider(MagicMock())
        with pytest.raises(KeyError, match="Drifted Command"):
            _ = [hit async for hit in provider.discover()]


# ── Native vs fallback parity ──────────────────────────────────────────

# The child interpreter runs this program and prints the fallback
# catalog as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._commands import _command_catalog

json.dump(_command_catalog(), sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native catalog must carry
    exactly the reference's entries — same names, same help texts, same
    order — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_command_catalog_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        # Tuples become lists on the native side too, for the JSON-shaped
        # comparison (JSON round-trips the fallback's tuples as lists).
        native = [[name, help_text] for name, help_text in _command_catalog()]
        assert native == fallback, (
            f"native and fallback catalogs differ:\nnative:   {native!r}\nfallback: {fallback!r}"
        )
