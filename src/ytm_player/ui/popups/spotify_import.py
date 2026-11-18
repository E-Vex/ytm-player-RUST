"""Spotify playlist import popup — Single (≤100) and Multi (100+) modes.

Single mode: paste one Spotify URL → match → create YTM playlist.
Multi mode: enter combined name + number of parts → paste N URLs →
            match all sequentially → create one combined YTM playlist.

The per-track match verdict is resolved by the import service's
``_match_classification`` engine — implemented by the Rust extension
(``_ytm_player_native.spotify_match_classification``) when it is built,
by the pure-Python transcription in ``services/spotify_import.py``
otherwise. The popup's own flow decisions — the multi-mode setup
cascade, the URL-slot validation, the summary counting, the
create-batch toast verdict, the per-track result arm, the
candidate-row assembly, the input-submission routing table and the
tab-activation phase gate — resolve through the same dual-implemented
engines below (``_ytm_player_native.spotify_import_*``); the popup
owns the search calls, the widget arms and the notifications around
them. No Python-side scoring/sort/threshold walk, counting pass,
flow-verdict cascade, input-routing arm or tab-phase gate remains
here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Click
from textual.message import Message
from textual.widgets import (
    Button,
    Input,
    Label,
    ListItem,
    ListView,
    ProgressBar,
    Static,
)

# NOTE: ytm_player.services.spotify_import is imported lazily inside worker
# methods to avoid pulling in heavy optional deps (thefuzz, spotify_scraper)
# at module load time.
from ytm_player.ui.popups.base import BasePopup
from ytm_player.ui.theme import get_theme
from ytm_player.utils.formatting import extract_artist

logger = logging.getLogger(__name__)

_URL_PATTERN = re.compile(r"https?://open\.spotify\.com/(playlist|album)/")

# Max tracks per add_playlist_items call.
_ADD_BATCH_SIZE = 100


def _candidate_row(idx: int, candidate: dict) -> str:
    """Compose one disambiguation-candidate row — ``_CandidateItem``'s
    label line.

    The title through the mapping's own ``get`` with the ``"?"``
    default, the artist through the formatting module's
    ``extract_artist`` engine, the duration through the candidate's
    ``duration`` field with the ``duration_seconds`` fallback — the
    ``m:ss`` rendering only for a truthy seconds read — and the whole
    line composed through the reference's f-string shape.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_candidate_row``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    title = candidate.get("title", "?")
    artist_str = extract_artist(candidate)
    dur = candidate.get("duration", "")
    if not dur:
        dur_sec = candidate.get("duration_seconds", 0)
        dur = f"{dur_sec // 60}:{dur_sec % 60:02d}" if dur_sec else ""
    dur_part = f" ({dur})" if dur else ""
    return f"  {idx}. {title} — {artist_str}{dur_part}"


def _result_symbol(match_type: object) -> tuple[str, str]:
    """The per-track result-row arm — ``(symbol, theme field)``.

    The display decision both import workers render per matched
    track: an ``EXACT`` verdict the check arm styled ``success``, a
    ``MULTIPLE`` one the question arm styled ``warning``, anything
    else the cross arm styled ``error``. The comparison is CPython's
    own ``==`` against the service's ``MatchType`` members (imported
    at call time, the workers' own lazy-import pattern); the theme
    colour read stays at the call site — ``getattr(get_theme(),
    field)``.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_result_symbol``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    from ytm_player.services.spotify_import import MatchType

    if match_type == MatchType.EXACT:
        return "✓", "success"
    if match_type == MatchType.MULTIPLE:
        return "?", "warning"
    return "✗", "error"


def _multi_part_verdict(name: str, count_str: str) -> tuple[str, int]:
    """The multi-mode setup verdict — ``_generate_url_inputs``' gates.

    The validation cascade in the reference's own order: a falsy
    (stripped) *name* answers ``"name"``; a *count_str* CPython's own
    ``int()`` rejects as a ``ValueError`` answers ``"count"`` (a
    ``TypeError`` escapes to the caller, exactly like the reference's
    ``except ValueError``-only arm); a parsed count outside ``2..=20``
    answers ``"range"``; otherwise the verdict is ``"ok"``. Only the
    ``"ok"`` verdict's count means anything — the failed gates answer
    ``0`` — and the widget reads (the strip), the notifications, the
    focus moves and the input mounting stay at the call site.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_multi_part_verdict``) when it
    is built; this pure-Python transcription is the fallback otherwise.
    """
    if not name:
        return "name", 0
    try:
        count = int(count_str)
    except ValueError:
        return "count", 0
    if count < 2 or count > 20:
        return "range", 0
    return "ok", count


def _url_slot_verdict(values: list[str | None]) -> tuple[str, int] | None:
    """The multi-mode URL-slot verdict — ``_start_multi_import``'s pass.

    The first slot (0-based) whose value is bad, with the badness kind:
    a ``None`` entry — the caller's widget read failed, a missing URL
    input — answers ``("missing", slot)``; a value the module's
    ``_URL_PATTERN`` does not match answers ``("invalid", slot)``. A
    list whose every slot reads a valid playlist/album URL answers
    ``None``. The widget reads and the notify/focus moves stay at the
    call site.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_url_slot_verdict``) when it
    is built; this pure-Python transcription is the fallback otherwise.
    """
    for slot, value in enumerate(values):
        if value is None:
            return "missing", slot
        if not _URL_PATTERN.match(value):
            return "invalid", slot
    return None


def _summary_stats(results: list) -> dict:
    """The summary assembly — ``_show_summary``'s counting pass.

    The three bucket counts (``exact`` / ``fuzzy`` / ``not_found``,
    each a full pass comparing ``r.match_type`` through CPython's own
    ``==`` against the service's ``MatchType`` members, imported at
    call time), the ``total``, the ``matched`` sum, and the video-id
    collection — ``get_video_id(r.selected)`` for every result whose
    ``selected`` is not ``None``, the empty ids dropped — exactly the
    reference's two-pass walk. The status-line rendering and the
    widget arms stay at the call site.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_summary_stats``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    from ytm_player.services.spotify_import import MatchType, get_video_id

    exact = sum(1 for r in results if r.match_type == MatchType.EXACT)
    fuzzy = sum(1 for r in results if r.match_type == MatchType.MULTIPLE)
    not_found = sum(1 for r in results if r.match_type == MatchType.NONE)
    total = len(results)
    video_ids = [get_video_id(r.selected) for r in results if r.selected is not None]
    video_ids = [vid for vid in video_ids if vid]
    return {
        "exact": exact,
        "fuzzy": fuzzy,
        "not_found": not_found,
        "total": total,
        "matched": exact + fuzzy,
        "video_ids": video_ids,
    }


def _batch_outcome(batch_results: list, batch_sizes: list[int]) -> dict:
    """The create-batch toast verdict — ``_do_create``'s final arm.

    Which toast the create flow raises once every add-batch settled:
    no failed batches answers ``"success"``; every batch failed and
    they all failed the same way answers ``"all_failed_single"`` with
    that failure kind riding along (the very first failing result
    object, the reference's ``failure_kinds[0]``); every batch failed
    for more than one reason answers ``"all_failed_mixed"``; anything
    between answers ``"partial"`` with the accepted-track total and the
    failed-batch count. The message wording, the
    ``mutation_failure_suffix`` call and the notify calls stay at the
    call site.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_batch_outcome``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    failed_batches = 0
    failure_kinds: list = []
    added_total = 0
    for result, size in zip(batch_results, batch_sizes):
        if result == "success":
            added_total += size
        else:
            failed_batches += 1
            failure_kinds.append(result)
    batches = len(batch_results)
    if failed_batches == 0:
        return {
            "verdict": "success",
            "kind": None,
            "added_total": added_total,
            "failed_batches": 0,
            "batches": batches,
        }
    if failed_batches == batches:
        if len(set(failure_kinds)) == 1:
            return {
                "verdict": "all_failed_single",
                "kind": failure_kinds[0],
                "added_total": added_total,
                "failed_batches": failed_batches,
                "batches": batches,
            }
        return {
            "verdict": "all_failed_mixed",
            "kind": None,
            "added_total": added_total,
            "failed_batches": failed_batches,
            "batches": batches,
        }
    return {
        "verdict": "partial",
        "kind": None,
        "added_total": added_total,
        "failed_batches": failed_batches,
        "batches": batches,
    }


def _input_route(input_id: str | None, phase: str, value: str) -> dict[str, Any] | None:
    """The input-submission routing table — ``on_input_submitted``'s core.

    Which action an (input-id × phase) pair routes to, with each
    claimed arm's own data decision inside, in the reference's own arm
    order: ``si-url-input`` in the ``url`` phase answers the
    single-URL submit — the value stripped, the one-slot URL pass (the
    same ``_url_slot_verdict`` engine the multi-mode start run
    validates through) deciding between ``"invalid_url"`` (the
    caller's warning toast) and ``"start_single"`` with the stripped
    url riding along; ``si-cred-secret-input``/``si-cred-id-input`` in
    the ``creds`` phase answer ``"save_creds"``/``"focus_secret"``;
    ``si-name-input`` in the ``summary`` phase answers the name submit
    — the value stripped, an empty name answering ``"name_empty"``
    (the caller's warning toast), anything else ``"create_playlist"``
    with the stripped name riding along; ``si-multi-count-input``/
    ``si-multi-name-input`` in the ``multi_setup`` phase answer
    ``"generate_urls"``/``"focus_count"``; and an id starting with
    ``si-multi-url-`` in the ``multi_urls`` phase answers the advance
    walk — the index suffix (every ``si-multi-url-`` occurrence
    replaced away) through CPython's own ``int()``, a ``ValueError``
    answering nothing (the reference's ``except ValueError: pass``),
    a parse answering ``"advance_url"`` with the next input's id
    (``si-multi-url-{idx + 1}``). Anything else — an unknown id, a
    known id in the wrong phase, a widget with no id at all — answers
    ``None``: the reference's chain has no else, so an unmatched
    submission simply does nothing. The notifications, the focus
    moves, the worker starts and the start-button fallback (the focus
    query failing) stay at the call site.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_input_route``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    # ── Single mode ──
    if input_id == "si-url-input" and phase == "url":
        url = value.strip()
        # The single-URL gate is the one-slot URL pass — the same
        # engine the multi-mode start run validates through.
        if _url_slot_verdict([url]) is not None:
            return {"action": "invalid_url"}
        return {"action": "start_single", "url": url}

    if input_id == "si-cred-secret-input" and phase == "creds":
        return {"action": "save_creds"}

    if input_id == "si-cred-id-input" and phase == "creds":
        return {"action": "focus_secret"}

    if input_id == "si-name-input" and phase == "summary":
        name = value.strip()
        if not name:
            return {"action": "name_empty"}
        return {"action": "create_playlist", "name": name}

    # ── Multi mode ──
    if input_id == "si-multi-count-input" and phase == "multi_setup":
        return {"action": "generate_urls"}

    if input_id == "si-multi-name-input" and phase == "multi_setup":
        return {"action": "focus_count"}

    if input_id and input_id.startswith("si-multi-url-") and phase == "multi_urls":
        idx_str = input_id.replace("si-multi-url-", "")
        try:
            next_idx = int(idx_str) + 1
        except ValueError:
            return None
        return {"action": "advance_url", "next_id": f"si-multi-url-{next_idx}"}

    return None


def _tab_activation(tab_id: str) -> dict[str, Any]:
    """The tab-activation phase gate — ``_activate_tab``'s core.

    Which phase and which widget arms activating a tab produces:
    ``tab_id == "single"`` answers the single arm — the phase resets
    to ``"url"``, the single pane shows (the multi pane hides — the
    two displays are always complementary), the status line reads the
    paste prompt, the focus moves to the URL input and no multi state
    resets. Anything else — the multi tab, but also the separator or
    any other label the reference's ``else`` would catch — answers the
    multi arm: the phase resets to ``"multi_setup"``, the multi pane
    shows (the single pane hides), the status line reads the
    split-playlist instructions, the focus moves to the combined-name
    input and the caller resets the multi setup (the URL container
    emptied, the start button hidden, the name and count inputs
    cleared). The mode assignment, the tab-label class toggles, the
    display writes, the status update, the resets and the focus moves
    stay at the call site, applied from the answer in the reference's
    own order.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_import_tab_activation``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if tab_id == "single":
        return {
            "phase": "url",
            "single_visible": True,
            "status": "Paste a Spotify playlist URL and press Enter",
            "focus": "#si-url-input",
            "reset_multi": False,
        }
    return {
        "phase": "multi_setup",
        "single_visible": False,
        "status": (
            "Split your 100+ track playlist into chunks of ≤100 on Spotify,\n"
            "then enter the combined name and number of parts below"
        ),
        "focus": "#si-multi-name-input",
        "reset_multi": True,
    }


# ── Shared list items for results ────────────────────────────────────


class _ResultItem(ListItem):
    """A single track match result shown during progress."""

    def __init__(self, symbol: str, style: str, text: str) -> None:
        super().__init__()
        self._symbol = symbol
        self._style = style
        self._text = text

    def compose(self) -> ComposeResult:
        yield Label(f"[{self._style}]{self._symbol}[/{self._style}] {self._text}")


class _CandidateItem(ListItem):
    """A selectable candidate shown during disambiguation."""

    def __init__(self, index: int, candidate: dict) -> None:
        super().__init__()
        self.candidate = candidate
        self._index = index

    def compose(self) -> ComposeResult:
        yield Label(_candidate_row(self._index, self.candidate))


class _SkipItem(ListItem):
    """The 'Skip' option at the end of the candidate list."""

    def compose(self) -> ComposeResult:
        yield Label("  [dim]Skip this track[/dim]")


# ── Tab button ───────────────────────────────────────────────────────


class _TabClicked(Message):
    """Posted by _TabLabel when clicked."""

    def __init__(self, tab_id: str) -> None:
        super().__init__()
        self.tab_id = tab_id


class _TabLabel(Static):
    """Clickable tab label that toggles active state."""

    DEFAULT_CSS = """
    _TabLabel {
        width: auto;
        height: 1;
        padding: 0 2;
        color: $text-muted;
    }
    _TabLabel.active {
        color: $text;
        text-style: bold underline;
    }
    """

    def __init__(self, label: str, tab_id: str) -> None:
        super().__init__(label)
        self.tab_id = tab_id

    def on_click(self, event: Click) -> None:
        if self.tab_id in ("single", "multi"):
            event.stop()
            self.post_message(_TabClicked(self.tab_id))


# ── Main import popup ────────────────────────────────────────────────


class SpotifyImportPopup(BasePopup[str | None]):
    """Two-tab Spotify import: Single (≤100 tracks) and Multi (100+ tracks)."""

    DEFAULT_CSS = """
    SpotifyImportPopup > Vertical {
        width: 75;
        max-height: 85%;
    }

    /* Title */
    SpotifyImportPopup #si-title {
        text-align: center;
        text-style: bold;
        width: 100%;
        color: $text;
        margin-bottom: 1;
    }

    /* Tab bar */
    SpotifyImportPopup #si-tab-bar {
        height: 1;
        width: 100%;
        margin-bottom: 1;
        align-horizontal: center;
    }

    /* Status */
    SpotifyImportPopup #si-status {
        text-align: center;
        width: 100%;
        color: $text-muted;
        margin-bottom: 1;
    }

    /* Single mode */
    SpotifyImportPopup #si-single-content { }

    SpotifyImportPopup #si-url-input {
        margin-bottom: 1;
    }

    SpotifyImportPopup #si-cred-id-input {
        display: none;
        margin-bottom: 1;
    }

    SpotifyImportPopup #si-cred-id-input.visible {
        display: block;
    }

    SpotifyImportPopup #si-cred-secret-input {
        display: none;
        margin-bottom: 1;
    }

    SpotifyImportPopup #si-cred-secret-input.visible {
        display: block;
    }

    /* Multi mode */
    SpotifyImportPopup #si-multi-content {
        display: none;
    }

    SpotifyImportPopup #si-multi-name-input {
        margin-bottom: 1;
    }

    SpotifyImportPopup #si-multi-count-input {
        margin-bottom: 1;
    }

    SpotifyImportPopup #si-multi-urls {
        max-height: 15;
        margin-bottom: 1;
    }

    SpotifyImportPopup #si-multi-urls Input {
        margin-bottom: 0;
    }

    SpotifyImportPopup #si-multi-start-btn {
        display: none;
        margin-top: 1;
        width: 100%;
    }

    SpotifyImportPopup #si-multi-start-btn.visible {
        display: block;
    }

    /* Shared */
    SpotifyImportPopup #si-progress {
        margin: 0 1 1 1;
    }

    SpotifyImportPopup #si-results {
        height: auto;
        max-height: 20;
        background: $surface;
    }

    SpotifyImportPopup ListItem {
        padding: 0 1;
        height: 1;
    }

    SpotifyImportPopup #si-name-input {
        display: none;
    }

    SpotifyImportPopup #si-name-input.visible {
        display: block;
        margin-top: 1;
    }

    SpotifyImportPopup #si-summary {
        display: none;
    }

    SpotifyImportPopup #si-summary.visible {
        display: block;
        text-align: center;
        width: 100%;
        color: $text;
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._mode: str = "single"  # "single" or "multi"
        # Single-mode phases: url | creds | progress | disambiguate | summary | creating
        # Multi-mode phases: multi_setup | multi_urls | multi_progress | disambiguate | summary | creating
        self._phase: str = "url"
        self._playlist_name: str = ""
        self._results: list = []
        self._video_ids: list[str] = []
        self._pending_url: str = ""
        # Disambiguation state.
        self._disambig_queue: list = []
        self._disambig_index: int = 0
        # Multi-mode state.
        self._multi_name: str = ""
        self._multi_url_count: int = 0
        self._multi_urls: list[str] = []
        self._all_results: list = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Import Spotify Playlist", id="si-title")
            with Horizontal(id="si-tab-bar"):
                yield _TabLabel("Single (≤100)", "single")
                yield _TabLabel("  |  ", "separator")
                yield _TabLabel("Multi (100+)", "multi")
            yield Static("Paste a Spotify playlist URL and press Enter", id="si-status")

            # ── Single mode content ──
            with Vertical(id="si-single-content"):
                yield Input(
                    placeholder="https://open.spotify.com/playlist/...",
                    id="si-url-input",
                )
                yield Input(
                    placeholder="Spotify Client ID",
                    id="si-cred-id-input",
                )
                yield Input(
                    placeholder="Spotify Client Secret",
                    id="si-cred-secret-input",
                    password=True,
                )

            # ── Multi mode content ──
            with Vertical(id="si-multi-content"):
                yield Input(
                    placeholder="Combined playlist name (e.g. My Playlist)",
                    id="si-multi-name-input",
                )
                yield Input(
                    placeholder="Number of split playlists (2-20)",
                    id="si-multi-count-input",
                )
                yield VerticalScroll(id="si-multi-urls")
                yield Button("Start Import", id="si-multi-start-btn", variant="primary")

            # ── Shared widgets ──
            yield ProgressBar(total=100, show_eta=False, id="si-progress")
            yield ListView(id="si-results")
            yield Static("", id="si-summary")
            yield Input(placeholder="Playlist name...", id="si-name-input")

    def on_mount(self) -> None:
        self.query_one("#si-progress", ProgressBar).display = False
        self.query_one("#si-results", ListView).display = False
        self._activate_tab("single")
        self.query_one("#si-url-input", Input).focus()

    # ── Tab switching ────────────────────────────────────────────────

    def on__tab_clicked(self, event: _TabClicked) -> None:
        """Handle tab clicks — only before processing starts."""
        if self._phase in ("url", "multi_setup"):
            self._activate_tab(event.tab_id)

    def _activate_tab(self, tab_id: str) -> None:
        """Switch between single and multi mode."""
        self._mode = tab_id
        arms = _tab_activation(tab_id)

        # Update tab labels.
        for tab in self.query(_TabLabel):
            if tab.tab_id == tab_id:
                tab.add_class("active")
            elif tab.tab_id != "separator":
                tab.remove_class("active")

        # The engine's phase gate, applied in the reference's own order.
        self._phase = arms["phase"]
        single_content = self.query_one("#si-single-content")
        multi_content = self.query_one("#si-multi-content")
        single_content.display = arms["single_visible"]
        multi_content.display = not arms["single_visible"]
        self.query_one("#si-status", Static).update(arms["status"])

        if arms["reset_multi"]:
            # Reset multi URL container.
            self.query_one("#si-multi-urls", VerticalScroll).remove_children()
            self.query_one("#si-multi-start-btn").remove_class("visible")
            self.query_one("#si-multi-name-input", Input).value = ""
            self.query_one("#si-multi-count-input", Input).value = ""
        self.query_one(arms["focus"], Input).focus()

    # ── Input handling ───────────────────────────────────────────────

    def on_input_submitted(self, event: Input.Submitted) -> None:
        route = _input_route(event.input.id, self._phase, event.value)
        if route is None:
            return
        action = route["action"]

        # ── Single mode ──
        if action == "invalid_url":
            self.notify("Invalid Spotify URL", severity="warning")
        elif action == "start_single":
            url = route["url"]
            self._pending_url = url
            self._start_single_import(url)
        elif action == "save_creds":
            self._save_creds_and_retry()
        elif action == "focus_secret":
            self.query_one("#si-cred-secret-input", Input).focus()
        elif action == "name_empty":
            self.notify("Name cannot be empty", severity="warning")
        elif action == "create_playlist":
            self._create_playlist(route["name"])

        # ── Multi mode ──
        elif action == "generate_urls":
            self._generate_url_inputs()
        elif action == "focus_count":
            self.query_one("#si-multi-count-input", Input).focus()
        elif action == "advance_url":
            # User pressed Enter on a URL input — move to next or start.
            try:
                self.query_one(f"#{route['next_id']}", Input).focus()
            except Exception:
                # Last input — focus the start button.
                self.query_one("#si-multi-start-btn", Button).focus()

    # ── Single mode: credential setup ────────────────────────────────

    def _show_creds_prompt(self) -> None:
        """Ask the user for Spotify API credentials."""
        self._phase = "creds"
        status = self.query_one("#si-status", Static)
        status.update(
            "Playlist has >100 tracks. Spotify API credentials needed.\n"
            "Get free credentials at [bold]developer.spotify.com/dashboard[/bold]\n"
            "Enter Client ID and Client Secret below:"
        )
        self.query_one("#si-url-input", Input).display = False
        self.query_one("#si-progress", ProgressBar).display = False

        cred_id = self.query_one("#si-cred-id-input", Input)
        cred_secret = self.query_one("#si-cred-secret-input", Input)
        cred_id.add_class("visible")
        cred_secret.add_class("visible")
        cred_id.focus()

    def _save_creds_and_retry(self) -> None:
        """Save credentials and restart the import."""
        from ytm_player.services.spotify_import import save_spotify_creds

        client_id = self.query_one("#si-cred-id-input", Input).value.strip()
        client_secret = self.query_one("#si-cred-secret-input", Input).value.strip()
        if not client_id or not client_secret:
            self.notify("Both Client ID and Secret are required", severity="warning")
            return

        save_spotify_creds(client_id, client_secret)
        self.query_one("#si-cred-id-input", Input).remove_class("visible")
        self.query_one("#si-cred-secret-input", Input).remove_class("visible")
        self.notify("Credentials saved", severity="information")
        self._start_single_import(self._pending_url)

    # ── Single mode: import flow ─────────────────────────────────────

    def _start_single_import(self, url: str) -> None:
        """Kick off the single-playlist import worker."""
        self._phase = "progress"
        self.query_one("#si-single-content").display = False
        self.query_one("#si-tab-bar").display = False
        self.query_one("#si-progress", ProgressBar).display = True
        self.query_one("#si-results", ListView).display = True
        self.query_one("#si-status", Static).update("Fetching Spotify playlist...")
        self.run_worker(
            self._do_single_import(url),
            name="spotify_import",
            group="spotify-import",
            exclusive=True,
        )

    async def _do_single_import(self, url: str) -> None:
        """Fetch Spotify tracks, then match each against YouTube Music."""
        from ytm_player.services.spotify_import import (
            MatchResult,
            MatchType,
            _bucket_results,
            _match_classification,
            extract_spotify_tracks,
            has_spotify_creds,
        )

        status = self.query_one("#si-status", Static)
        progress_bar = self.query_one("#si-progress", ProgressBar)
        results_list = self.query_one("#si-results", ListView)

        # Step 1: Extract tracks from Spotify.
        try:
            playlist_name, spotify_tracks = await asyncio.to_thread(extract_spotify_tracks, url)
        except Exception as exc:
            color = get_theme().error
            status.update(f"[{color}]Failed to fetch playlist:[/{color}] {exc}")
            return

        if not spotify_tracks:
            color = get_theme().warning
            status.update(f"[{color}]No tracks found in the playlist.[/{color}]")
            return

        self._playlist_name = playlist_name

        # Detect truncation.
        if len(spotify_tracks) == 100 and not has_spotify_creds():
            self.call_later(self._show_creds_prompt)
            return

        total = len(spotify_tracks)
        progress_bar.update(total=total, progress=0)
        status.update(f"Matching on YouTube Music... (0/{total})")

        # Step 2: Match each track.
        results: list[MatchResult] = []
        ytmusic_svc = self.app.ytmusic  # type: ignore[attr-defined]

        for i, sp_track in enumerate(spotify_tracks, 1):
            query = f"{sp_track['name']} {sp_track['artist']}"
            try:
                search_results = await ytmusic_svc.search(query, filter="songs", limit=5)
            except Exception:
                search_results = []

            result = _match_classification(sp_track, search_results)
            results.append(result)

            display_text = f"{sp_track['name']} — {sp_track['artist']}"
            symbol, theme_field = _result_symbol(result.match_type)
            results_list.append(
                _ResultItem(symbol, getattr(get_theme(), theme_field), display_text)
            )

            progress_bar.update(progress=i)
            status.update(f"Matching on YouTube Music... ({i}/{total})")
            results_list.scroll_end(animate=False)

        self._results = results

        uncertain = _bucket_results(results, MatchType.MULTIPLE)
        if uncertain:
            self._disambig_queue = uncertain
            self._disambig_index = 0
            self._start_disambiguation()
        else:
            self._show_summary()

    # ── Multi mode: setup ────────────────────────────────────────────

    def _generate_url_inputs(self) -> None:
        """Validate count and generate URL input fields."""
        name = self.query_one("#si-multi-name-input", Input).value.strip()
        count_str = self.query_one("#si-multi-count-input", Input).value.strip()

        verdict, count = _multi_part_verdict(name, count_str)
        if verdict == "name":
            self.notify("Enter a combined playlist name", severity="warning")
            self.query_one("#si-multi-name-input", Input).focus()
            return
        if verdict == "count":
            self.notify("Enter a valid number", severity="warning")
            return
        if verdict == "range":
            self.notify("Number of parts must be between 2 and 20", severity="warning")
            return

        self._multi_name = name
        self._multi_url_count = count
        self._phase = "multi_urls"

        # Disable setup inputs.
        self.query_one("#si-multi-name-input", Input).disabled = True
        self.query_one("#si-multi-count-input", Input).disabled = True

        # Generate URL inputs.
        url_container = self.query_one("#si-multi-urls", VerticalScroll)
        for i in range(count):
            url_container.mount(
                Input(
                    placeholder=f"Spotify URL for {name}{i + 1}",
                    id=f"si-multi-url-{i}",
                )
            )

        # Show the start button.
        self.query_one("#si-multi-start-btn").add_class("visible")

        status = self.query_one("#si-status", Static)
        status.update(f"Paste {count} Spotify playlist URLs below, then click Start Import")

        # Focus the first URL input.
        self.call_later(lambda: self.query_one("#si-multi-url-0", Input).focus())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the Start Import button for multi mode."""
        if event.button.id == "si-multi-start-btn" and self._phase == "multi_urls":
            self._start_multi_import()

    def _start_multi_import(self) -> None:
        """Validate all URLs and begin multi-playlist import."""
        # Read every slot's value first (a missing input reads None);
        # the engine then decides which slot — if any — is bad.
        values: list[str | None] = []
        for i in range(self._multi_url_count):
            try:
                url_input = self.query_one(f"#si-multi-url-{i}", Input)
                values.append(url_input.value.strip())
            except Exception:
                values.append(None)

        bad = _url_slot_verdict(values)
        if bad is not None:
            kind, slot = bad
            if kind == "missing":
                self.notify(f"Missing URL input #{slot + 1}", severity="error")
            else:
                self.notify(f"Invalid Spotify URL in slot {slot + 1}", severity="warning")
                self.query_one(f"#si-multi-url-{slot}", Input).focus()
            return

        urls: list[str] = [value for value in values if value is not None]

        self._multi_urls = urls
        self._all_results = []

        # Hide multi inputs, show progress.
        self._phase = "multi_progress"
        self._playlist_name = self._multi_name
        self.query_one("#si-multi-content").display = False
        self.query_one("#si-tab-bar").display = False
        self.query_one("#si-progress", ProgressBar).display = True
        self.query_one("#si-results", ListView).display = True

        self.run_worker(
            self._do_multi_import(),
            name="spotify_multi_import",
            group="spotify-import",
            exclusive=True,
        )

    async def _do_multi_import(self) -> None:
        """Process all split playlists sequentially."""
        from ytm_player.services.spotify_import import (
            MatchType,
            _bucket_results,
            _match_classification,
            extract_spotify_tracks,
        )

        status = self.query_one("#si-status", Static)
        progress_bar = self.query_one("#si-progress", ProgressBar)
        results_list = self.query_one("#si-results", ListView)
        ytmusic_svc = self.app.ytmusic  # type: ignore[attr-defined]

        total_urls = len(self._multi_urls)

        for url_idx, url in enumerate(self._multi_urls):
            part_num = url_idx + 1
            status.update(f"[bold]Part {part_num}/{total_urls}:[/bold] Fetching from Spotify...")

            # Extract tracks from this part.
            try:
                part_name, spotify_tracks = await asyncio.to_thread(extract_spotify_tracks, url)
            except Exception as exc:
                color = get_theme().error
                status.update(f"[{color}]Failed to fetch part {part_num}:[/{color}] {exc}")
                return

            if not spotify_tracks:
                results_list.append(
                    _ResultItem(
                        "⚠", get_theme().warning, f"Part {part_num}: no tracks found, skipping"
                    )
                )
                continue

            results_list.append(
                _ResultItem(
                    "▸",
                    "bold",
                    f"Part {part_num}: {part_name} ({len(spotify_tracks)} tracks)",
                )
            )
            results_list.scroll_end(animate=False)

            total_tracks = len(spotify_tracks)
            progress_bar.update(total=total_tracks, progress=0)

            # Match each track.
            for i, sp_track in enumerate(spotify_tracks, 1):
                query = f"{sp_track['name']} {sp_track['artist']}"
                try:
                    search_results = await ytmusic_svc.search(query, filter="songs", limit=5)
                except Exception:
                    search_results = []

                result = _match_classification(sp_track, search_results)
                self._all_results.append(result)

                display_text = f"{sp_track['name']} — {sp_track['artist']}"
                symbol, theme_field = _result_symbol(result.match_type)
                results_list.append(
                    _ResultItem(symbol, getattr(get_theme(), theme_field), display_text)
                )

                progress_bar.update(progress=i)
                status.update(
                    f"[bold]Part {part_num}/{total_urls}:[/bold] Matching... ({i}/{total_tracks})"
                )
                results_list.scroll_end(animate=False)

        # All parts processed. Store combined results.
        self._results = self._all_results

        # Check for uncertain matches.
        uncertain = _bucket_results(self._results, MatchType.MULTIPLE)
        if uncertain:
            self._disambig_queue = uncertain
            self._disambig_index = 0
            self._start_disambiguation()
        else:
            self._show_summary()

    # ── Shared: disambiguation ───────────────────────────────────────

    def _start_disambiguation(self) -> None:
        """Enter the disambiguation phase."""
        self._phase = "disambiguate"
        self.query_one("#si-progress", ProgressBar).display = False
        self._show_current_disambig()

    def _show_current_disambig(self) -> None:
        """Display candidates for the current uncertain match."""
        if self._disambig_index >= len(self._disambig_queue):
            self._show_summary()
            return

        result = self._disambig_queue[self._disambig_index]
        sp = result.spotify_track
        total_uncertain = len(self._disambig_queue)
        current = self._disambig_index + 1

        status = self.query_one("#si-status", Static)
        status.update(
            f"[bold]Uncertain match ({current}/{total_uncertain}):[/bold] "
            f'"{sp["name"]}" by {sp["artist"]}\n'
            f"Select the correct match or skip:"
        )

        results_list = self.query_one("#si-results", ListView)
        results_list.clear()
        results_list.display = True

        for i, candidate in enumerate(result.candidates[:5], 1):
            results_list.append(_CandidateItem(i, candidate))
        results_list.append(_SkipItem())

        results_list.index = 0
        results_list.focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle candidate selection during disambiguation."""
        if self._phase != "disambiguate":
            return

        event.stop()
        result = self._disambig_queue[self._disambig_index]

        if isinstance(event.item, _CandidateItem):
            from ytm_player.services.spotify_import import MatchType

            result.selected = event.item.candidate
            result.match_type = MatchType.EXACT
        elif isinstance(event.item, _SkipItem):
            from ytm_player.services.spotify_import import MatchType

            result.selected = None
            result.match_type = MatchType.NONE

        self._disambig_index += 1
        self._show_current_disambig()

    # ── Shared: summary + create ─────────────────────────────────────

    def _show_summary(self) -> None:
        """Display summary stats and playlist name input."""
        self._phase = "summary"

        stats = _summary_stats(self._results)
        not_found = stats["not_found"]
        total = stats["total"]

        self._video_ids = stats["video_ids"]

        results_list = self.query_one("#si-results", ListView)
        results_list.clear()
        results_list.display = False

        self.query_one("#si-progress", ProgressBar).display = False

        status = self.query_one("#si-status", Static)
        matched = stats["matched"]
        status.update(f"{matched} matched, {not_found} skipped/not found ({total} total)")

        if not self._video_ids:
            summary = self.query_one("#si-summary", Static)
            summary.update("No tracks matched — nothing to create. Press Escape to close.")
            summary.add_class("visible")
            return

        summary = self.query_one("#si-summary", Static)
        summary.update("Enter a playlist name and press Enter to create")
        summary.add_class("visible")

        name_input = self.query_one("#si-name-input", Input)
        name_input.value = self._playlist_name
        name_input.add_class("visible")
        name_input.focus()

    def _create_playlist(self, name: str) -> None:
        """Create the playlist on YouTube Music."""
        self._phase = "creating"
        self.run_worker(
            self._do_create(name), name="create_playlist", group="spotify-import", exclusive=True
        )

    async def _do_create(self, name: str) -> None:
        """Create playlist and add tracks in batches of 100."""
        status = self.query_one("#si-status", Static)
        progress_bar = self.query_one("#si-progress", ProgressBar)

        ytmusic_svc = self.app.ytmusic  # type: ignore[attr-defined]

        try:
            desc = f"Imported from Spotify: {self._playlist_name}"
            status.update(f"Creating '{name}' on YouTube Music...")
            create_status, playlist_id = await ytmusic_svc.create_playlist(name, desc)
            if create_status != "success" or not playlist_id:
                color = get_theme().error
                status.update(f"[{color}]Failed to create playlist.[/{color}]")
                return

            total_ids = len(self._video_ids)
            if total_ids > 0:
                batches = [
                    self._video_ids[i : i + _ADD_BATCH_SIZE]
                    for i in range(0, total_ids, _ADD_BATCH_SIZE)
                ]
                if len(batches) > 1:
                    progress_bar.display = True
                    progress_bar.update(total=len(batches), progress=0)

                from ytm_player.services.ytmusic import mutation_failure_suffix

                # The per-batch answers, collected in batch order; the
                # engine then decides which toast the outcome deserves.
                batch_results: list = []
                batch_sizes = [len(batch) for batch in batches]
                for batch_idx, batch in enumerate(batches, 1):
                    status.update(
                        f"Adding tracks... ({min(batch_idx * _ADD_BATCH_SIZE, total_ids)}/{total_ids})"
                    )
                    # Imports preserve the source playlist faithfully — Spotify
                    # playlists routinely contain the same track twice, so pass
                    # duplicates=True to avoid a whole batch false-failing on a
                    # single repeated track.
                    result = await ytmusic_svc.add_playlist_items(
                        playlist_id, batch, duplicates=True
                    )
                    batch_results.append(result)
                    if len(batches) > 1:
                        progress_bar.update(progress=batch_idx)

                outcome = _batch_outcome(batch_results, batch_sizes)
                if outcome["verdict"] == "success":
                    self.notify(
                        f"Created '{name}' with {total_ids} tracks",
                        severity="information",
                    )
                elif outcome["verdict"] == "all_failed_single":
                    # All batches failed the same way — surface that.
                    suffix = mutation_failure_suffix(outcome["kind"])  # type: ignore[arg-type]
                    self.notify(
                        f"Created '{name}' but couldn't add any tracks — {suffix}",
                        severity="error",
                    )
                elif outcome["verdict"] == "all_failed_mixed":
                    self.notify(
                        f"Created '{name}' but failed to add any tracks "
                        f"({outcome['batches']} batch(es) rejected)",
                        severity="error",
                    )
                else:
                    self.notify(
                        f"Created '{name}' with partial track set: "
                        f"{outcome['added_total']}/{total_ids} added "
                        f"({outcome['failed_batches']} batch(es) failed)",
                        severity="warning",
                    )
            self.dismiss(playlist_id)

        except Exception as exc:
            logger.exception("Failed to create playlist")
            color = get_theme().error
            status.update(f"[{color}]Error:[/{color}] {exc}")

    # ── Cancel ───────────────────────────────────────────────────────

    def action_cancel(self) -> None:
        """Cancel active workers and dismiss the popup.

        Overrides ``BasePopup.action_cancel`` so both Escape and a backdrop
        click cancel in-flight import workers before dismissing.
        """
        for worker in self.workers:
            worker.cancel()
        self.dismiss(self._CANCEL_RESULT)


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same flow engines; these bindings replace
# the pure-Python definitions above when the extension is importable.  The
# rebinding must happen AFTER the definitions above — assigning the native
# functions any earlier would be clobbered by the ``def`` statements.

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        spotify_import_batch_outcome as _native_spotify_import_batch_outcome,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_candidate_row as _native_spotify_import_candidate_row,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_input_route as _native_spotify_import_input_route,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_multi_part_verdict as _native_spotify_import_multi_part_verdict,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_result_symbol as _native_spotify_import_result_symbol,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_summary_stats as _native_spotify_import_summary_stats,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_tab_activation as _native_spotify_import_tab_activation,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_import_url_slot_verdict as _native_spotify_import_url_slot_verdict,
    )
except ImportError:
    pass
else:
    _candidate_row = _native_spotify_import_candidate_row  # type: ignore[assignment]
    _result_symbol = _native_spotify_import_result_symbol  # type: ignore[assignment]
    _multi_part_verdict = _native_spotify_import_multi_part_verdict  # type: ignore[assignment]
    _url_slot_verdict = _native_spotify_import_url_slot_verdict  # type: ignore[assignment]
    _summary_stats = _native_spotify_import_summary_stats  # type: ignore[assignment]
    _batch_outcome = _native_spotify_import_batch_outcome  # type: ignore[assignment]
    _input_route = _native_spotify_import_input_route  # type: ignore[assignment]
    _tab_activation = _native_spotify_import_tab_activation  # type: ignore[assignment]
