"""Spotify playlist import — extract tracks, match on YouTube Music, create playlist.

The matching pipeline's pure cores are dual-implemented: the URL/item
parse (``_extract_playlist_id`` / ``_parse_spotipy_item``), the weighted
fuzzy score (``_fuzzy_score``), the score-sort-threshold classification
(``_match_classification``), the per-page item walk (``_page_items``),
the scraper fallback's track-row walk (``_scraper_track_rows``), the
candidate display row (``_display_candidate``), the confirmed/video-ids/
skipped accounting (``_summarize_results``) and the match-type bucket
walk (``_bucket_results``). Each is implemented by the Rust extension
(``_ytm_player_native.extract_playlist_id`` / ``parse_spotipy_item`` /
``spotify_fuzzy_score`` / ``spotify_match_classification`` /
``spotify_page_items`` / ``spotify_scraper_track_rows`` /
``spotify_display_candidate`` / ``spotify_summarize_results`` /
``spotify_bucket_results``) when it is built and by the pure-Python
transcriptions below otherwise; the spotipy/scraper network calls, the
thread-pool fan-out, the interactive prompts and the playlist creation
stay Python-side in this module.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import click
from ytmusicapi import YTMusic

try:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from thefuzz import fuzz  # type: ignore[reportMissingImports]

    _HAS_SPOTIFY_DEPS = True
except ImportError:
    _HAS_SPOTIFY_DEPS = False

from pathlib import Path

from ytm_player.config.paths import CONFIG_DIR, SECURE_FILE_MODE, SPOTIFY_CREDS_FILE
from ytm_player.utils.formatting import (
    VALID_VIDEO_ID,
    extract_artist,
    extract_duration,
    format_duration,
    get_video_id,
)

logger = logging.getLogger(__name__)

TITLE_MATCH_WEIGHT = 0.6
ARTIST_MATCH_WEIGHT = 0.4
AUTO_MATCH_THRESHOLD = 85


class MatchType(Enum):
    EXACT = "exact"
    MULTIPLE = "multiple"
    NONE = "none"


@dataclass
class MatchResult:
    """Result of matching a Spotify track against YouTube Music."""

    spotify_track: dict
    match_type: MatchType
    candidates: list[dict] = field(default_factory=list)
    selected: dict | None = None


# ── Spotify credential helpers ────────────────────────────────────────


def load_spotify_creds() -> dict[str, str] | None:
    """Load stored Spotify client_id/client_secret, or ``None``."""
    if not SPOTIFY_CREDS_FILE.exists():
        return None
    try:
        data = json.loads(SPOTIFY_CREDS_FILE.read_text(encoding="utf-8"))
        if data.get("client_id") and data.get("client_secret"):
            return data
    except Exception:
        logger.debug("Failed to parse Spotify credentials file", exc_info=True)
    return None


def save_spotify_creds(client_id: str, client_secret: str) -> None:
    """Persist Spotify API credentials."""
    import os

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fd = os.open(
        str(SPOTIFY_CREDS_FILE),
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        SECURE_FILE_MODE,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"client_id": client_id, "client_secret": client_secret}, f, indent=2)


def has_spotify_creds() -> bool:
    """Return True if Spotify API credentials are configured."""
    return load_spotify_creds() is not None


# ── Track extraction ──────────────────────────────────────────────────


def _extract_playlist_id(url: str) -> str:
    """Pull the playlist/album ID from a Spotify URL.

    Implemented by the Rust extension
    (``_ytm_player_native.extract_playlist_id``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    m = re.search(r"(playlist|album)/([a-zA-Z0-9]+)", url)
    return m.group(2) if m else ""


def _parse_spotipy_item(item: dict) -> dict:
    """Normalise a spotipy track item to our internal format.

    Implemented by the Rust extension
    (``_ytm_player_native.parse_spotipy_item``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    track = item.get("track", item)
    if not track or not track.get("name"):
        return {}
    artist_name = extract_artist(track)
    album = track.get("album", {})
    return {
        "name": track.get("name", ""),
        "artist": artist_name,
        "album": album.get("name", "") if isinstance(album, dict) else "",
        "duration_ms": track.get("duration_ms", 0),
    }


def _page_items(results: dict) -> list[dict]:
    """One page's parsed track rows.

    The pure core of ``extract_spotify_tracks_spotipy``'s pagination
    walk: the page's ``items`` member read through the mapping's own
    ``get`` with a fresh empty list default (a missing key answers no
    items), each item normalised through the ``_parse_spotipy_item``
    engine, and each falsy parse — the empty dict the normaliser
    answers for an episode entry or a region-locked track — dropped by
    the ``if parsed:`` truthiness gate. The ``while results`` loop, the
    ``sp.next`` fetch and the ``next`` member gate stay in the wrapper.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_page_items``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    tracks: list[dict] = []
    for item in results.get("items", []):
        parsed = _parse_spotipy_item(item)
        if parsed:
            tracks.append(parsed)
    return tracks


def extract_spotify_tracks_spotipy(url: str) -> tuple[str, list[dict]]:
    """Extract ALL tracks using the Spotify Web API (with pagination).

    Uses stored client credentials from ``~/.config/ytm-player/spotify.json``.
    Raises ``RuntimeError`` if credentials are missing or the request fails.
    """
    import spotipy  # type: ignore[reportMissingImports]
    from spotipy.oauth2 import SpotifyClientCredentials  # type: ignore[reportMissingImports]

    creds = load_spotify_creds()
    if not creds:
        raise RuntimeError("Spotify API credentials not configured")

    sp = spotipy.Spotify(
        auth_manager=SpotifyClientCredentials(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
        )
    )

    playlist_id = _extract_playlist_id(url)
    if not playlist_id:
        raise RuntimeError("Could not parse playlist ID from URL")

    # Determine if this is a playlist or album.
    is_album = "/album/" in url

    if is_album:
        album = sp.album(playlist_id)
        if album is None:
            raise RuntimeError("Spotify API returned no data for album")
        playlist_name = album.get("name", "Imported Album")
        results = album.get("tracks", {})
    else:
        playlist = sp.playlist(playlist_id)
        if playlist is None:
            raise RuntimeError("Spotify API returned no data for playlist")
        playlist_name = playlist.get("name", "Imported Playlist")
        results = playlist.get("tracks", {})

    tracks: list[dict] = []
    while results:
        tracks.extend(_page_items(results))
        # Follow pagination.
        if results.get("next"):
            results = sp.next(results)
        else:
            break

    return playlist_name, tracks


def _scraper_track_rows(playlist: Any) -> list[dict]:
    """The scraper fallback's track rows.

    The pure core of ``extract_spotify_tracks``'s scraper walk: the
    playlist's ``tracks`` walked item by item, each item's ``track``
    read through its attribute, the artist as the ``", "``-joined
    ``name``s of ``track.artists`` with the ``or "Unknown"`` truthiness
    gate behind it (no artists, or a join that answers an empty
    string, reads as ``Unknown``), the album name read only through a
    truthy ``track.album`` (a falsy album answers ``""``), and
    ``name``/``duration_ms`` passing through as the objects they are,
    into a fresh four-key dict per row. The playlist name and
    total-count reads, and the short-pagination warning, stay in the
    wrapper.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_scraper_track_rows``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    tracks: list[dict] = []
    for item in playlist.tracks:
        track = item.track
        artist_name = ", ".join(a.name for a in track.artists) or "Unknown"
        album_name = track.album.name if track.album else ""

        tracks.append(
            {
                "name": track.name,
                "artist": artist_name,
                "album": album_name,
                "duration_ms": track.duration_ms,
            }
        )
    return tracks


def extract_spotify_tracks(url: str) -> tuple[str, list[dict]]:
    """Extract tracks — tries spotipy first, falls back to spotify_scraper.

    Returns:
        Tuple of (playlist_name, list of track dicts with name/artist/album/duration_ms).
    """
    # Try spotipy first (supports full pagination).
    if has_spotify_creds():
        try:
            return extract_spotify_tracks_spotipy(url)
        except Exception as exc:
            logger.warning("spotipy extraction failed, falling back to scraper: %s", exc)

    # Fallback: spotify_scraper. max_tracks=None fetches the whole playlist,
    # not just its first page.
    from spotify_scraper import SpotifyClient  # type: ignore[reportMissingImports]

    client = SpotifyClient()
    try:
        playlist = client.get_playlist(url, max_tracks=None)

        playlist_name = playlist.name or "Imported Playlist"
        track_count = playlist.total_tracks or 0
        tracks = _scraper_track_rows(playlist)

        # Flag the rare case where the scraper's own pagination came up short.
        if track_count and track_count > len(tracks):
            logger.warning(
                "Playlist has %d tracks but scraper returned %d. "
                "Configure Spotify API credentials for full import.",
                track_count,
                len(tracks),
            )

        return playlist_name, tracks
    finally:
        client.close()


def _fuzzy_score(spotify_track: dict, ytm_track: dict) -> int:
    """Compute a fuzzy match score between a Spotify track and a YTM result.

    The pure core of the import matcher's candidate ranking: both sides
    lowercased (the YTM title through the ``or ""`` truthiness gate, the
    YTM artist through the formatting module's ``extract_artist``),
    both ratios through the ``fuzz`` the [spotify] extra provides, and
    the weighted sum — ``int(title_score * TITLE_MATCH_WEIGHT +
    artist_score * ARTIST_MATCH_WEIGHT)`` — with the module's own
    weights, so patched weights steer both backends identically.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_fuzzy_score``) when it is built; this
    pure-Python transcription is the fallback otherwise.
    """
    # fuzz / Console / Progress / etc. are conditionally imported under
    # _HAS_SPOTIFY_DEPS at module top. Callers gate on _HAS_SPOTIFY_DEPS
    # before reaching this code path; Pyright can't narrow through that
    # so we suppress the possibly-unbound warning on each usage site.
    sp_title = spotify_track.get("name", "").lower()
    sp_artist = spotify_track.get("artist", "").lower()

    ytm_title = (ytm_track.get("title", "") or "").lower()
    ytm_artist = extract_artist(ytm_track).lower()

    title_score = fuzz.ratio(sp_title, ytm_title)  # type: ignore[possibly-unbound]
    artist_score = fuzz.ratio(sp_artist, ytm_artist)  # type: ignore[possibly-unbound]

    # Weighted: title matters more but artist is still important.
    return int(title_score * TITLE_MATCH_WEIGHT + artist_score * ARTIST_MATCH_WEIGHT)


_MATCH_MAX_WORKERS = 5


def _match_classification(spotify_track: dict, search_results: list[dict]) -> MatchResult:
    """The score-sort-threshold verdict for one Spotify track's search.

    The pure core of ``_search_and_score``'s post-search walk: the
    empty-results gate first (answering a ``MatchType.NONE`` result
    with the dataclass's own default ``candidates``/``selected``),
    then every candidate scored through the ``_fuzzy_score`` engine,
    the ``(score, candidate)`` pairs sorted best-first — stable, so
    equal scores keep the search's own order — and the leading pair
    gating the verdict: ``best_score >= AUTO_MATCH_THRESHOLD`` answers
    a ``MatchType.EXACT`` result whose ``selected`` is the very
    best-candidate object and whose ``candidates`` is the best-first
    list, every other score a ``MatchType.MULTIPLE`` result over the
    same list. The ``ytmusic.search`` call, the query build and the
    exception arm that answers empty results for a failed search stay
    in the wrapper.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_match_classification``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    if not search_results:
        return MatchResult(
            spotify_track=spotify_track,
            match_type=MatchType.NONE,
        )

    scored = [(_fuzzy_score(spotify_track, c), c) for c in search_results]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_candidate = scored[0]

    if best_score >= AUTO_MATCH_THRESHOLD:
        return MatchResult(
            spotify_track=spotify_track,
            match_type=MatchType.EXACT,
            candidates=[c for _, c in scored],
            selected=best_candidate,
        )

    return MatchResult(
        spotify_track=spotify_track,
        match_type=MatchType.MULTIPLE,
        candidates=[c for _, c in scored],
    )


def _search_and_score(ytmusic: YTMusic, sp_track: dict, index: int) -> tuple[int, MatchResult]:
    """Search YTM for a single Spotify track and return (original_index, result).

    Designed to run inside a thread pool.  All data it touches is either
    thread-local (search_results, scored) or read-only (sp_track), so no
    locking is needed.  The match verdict itself is resolved by the
    ``_match_classification`` engine (native when the Rust extension is
    built, the pure-Python transcription otherwise); this wrapper owns
    the query build, the search call and the failed-search arm.
    """
    query = f"{sp_track['name']} {sp_track['artist']}"
    try:
        search_results = ytmusic.search(query, filter="songs", limit=5)
    except Exception:
        search_results = []

    return index, _match_classification(sp_track, search_results)


def match_tracks(
    ytmusic: YTMusic, spotify_tracks: list[dict], console: Console
) -> list[MatchResult]:
    """Search YTM for each Spotify track and categorize matches."""
    if not _HAS_SPOTIFY_DEPS:
        click.echo("Spotify import requires extra dependencies: pip install ytm-player[spotify]")
        return []

    # Pre-allocate results list so we can slot them back in order.
    results: list[MatchResult | None] = [None] * len(spotify_tracks)

    with Progress(  # type: ignore[possibly-unbound]
        SpinnerColumn(),  # type: ignore[possibly-unbound]
        TextColumn("[progress.description]{task.description}"),  # type: ignore[possibly-unbound]
        BarColumn(),  # type: ignore[possibly-unbound]
        TextColumn("{task.completed}/{task.total}"),  # type: ignore[possibly-unbound]
        console=console,
    ) as progress:
        task = progress.add_task("Searching YouTube Music...", total=len(spotify_tracks))

        with ThreadPoolExecutor(max_workers=_MATCH_MAX_WORKERS) as executor:
            futures = {
                executor.submit(_search_and_score, ytmusic, sp_track, idx): idx
                for idx, sp_track in enumerate(spotify_tracks)
            }

            for future in as_completed(futures):
                idx, match_result = future.result()
                results[idx] = match_result
                progress.advance(task)

    # All slots should be filled; narrow the type for the caller.
    return [r for r in results if r is not None]


def _display_candidate(idx: int, candidate: dict) -> str:
    """Format a single YTM candidate for display.

    The pure core of the import prompt's candidate line: the title read
    through the mapping's own ``get`` with the ``"?"`` default, the
    artist and duration through the formatting module's
    ``extract_artist``/``extract_duration`` engines (the duration
    rendering through ``format_duration`` only for a truthy read), the
    ``resultType`` read gating the `` [type]`` suffix (only a truthy
    type that is not ``"song"`` earns it), and the whole line composed
    through the reference's f-string shape.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_display_candidate``) when it is
    built; this pure-Python transcription is the fallback otherwise.
    """
    title = candidate.get("title", "?")
    artist_str = extract_artist(candidate)
    dur_sec = extract_duration(candidate)
    duration = format_duration(dur_sec) if dur_sec else "?"
    result_type = candidate.get("resultType", "")
    suffix = f" [{result_type}]" if result_type and result_type != "song" else ""
    return f"  {idx}. {title} — {artist_str} ({duration}){suffix}"


def _summarize_results(
    results: list[MatchResult],
) -> tuple[list[MatchResult], list[str], int]:
    """Split match results into confirmed selections, playable IDs, and skips.

    A result is *confirmed* when it has a non-``None`` ``selected`` candidate;
    every other result is *skipped* (``skipped = total - confirmed``). Confirmed
    selections whose extracted video ID is empty are dropped from the returned
    ID list (unplayable) but are NOT counted as skipped — so ``len(video_ids)``
    may be smaller than the confirmed count.

    Returns ``(confirmed_results, video_ids, skipped)``.

    The confirmed/video-ids/skipped accounting itself — the ``is not None``
    identity gate behind the confirmed walk, the ``len(results) -
    len(confirmed)`` difference, and the truthy-selected ``get_video_id``
    reads with the empty-id drop — is implemented by the Rust extension
    (``_ytm_player_native.spotify_summarize_results``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    confirmed = [r for r in results if r.selected is not None]
    skipped = len(results) - len(confirmed)
    video_ids = [get_video_id(r.selected) for r in confirmed if r.selected]
    video_ids = [vid for vid in video_ids if vid]  # Filter empty.
    return confirmed, video_ids, skipped


def _bucket_results(results: list[MatchResult], match_type: MatchType) -> list[MatchResult]:
    """The results whose match type equals *match_type*.

    The pure core of the import flow's result categorisation: the
    exact/multiple/none splits behind ``run_import``'s summary counts,
    its interactive resolution steps and its still-none re-walk — each
    a walk keeping every result whose ``match_type`` compares equal to
    the given member, the results passing through as the very objects
    they are, in the results' own order.

    Implemented by the Rust extension
    (``_ytm_player_native.spotify_bucket_results``) when it is built;
    this pure-Python transcription is the fallback otherwise.
    """
    return [r for r in results if r.match_type == match_type]


def run_import(spotify_url: str, auth_file: Path) -> None:
    """Orchestrate the full interactive Spotify → YTM import flow."""
    if not _HAS_SPOTIFY_DEPS:
        click.echo("Spotify import requires extra dependencies: pip install ytm-player[spotify]")
        return
    console = Console()  # type: ignore[possibly-unbound]

    # Validate URL.
    if not re.match(r"https?://open\.spotify\.com/(playlist|album)/", spotify_url):
        console.print("[red]Invalid Spotify URL.[/red] Expected a playlist or album link.")
        return

    # Step 1: Extract tracks from Spotify.
    console.print()
    with console.status("Fetching Spotify playlist..."):
        try:
            playlist_name, spotify_tracks = extract_spotify_tracks(spotify_url)
        except Exception as exc:
            console.print(f"[red]Failed to fetch Spotify playlist:[/red] {exc}")
            return

    if not spotify_tracks:
        console.print("[yellow]No tracks found in the playlist.[/yellow]")
        return

    console.print(f'Fetched [bold]"{playlist_name}"[/bold] ({len(spotify_tracks)} tracks)')
    console.print()

    # Step 2: Initialize YTM client.
    try:
        from ytm_player.config.settings import get_settings

        brand_account = get_settings().general.brand_account_id or None
        ytmusic = YTMusic(str(auth_file), user=brand_account)
    except Exception as exc:
        console.print(f"[red]Failed to initialize YouTube Music client:[/red] {exc}")
        return

    # Step 3: Search & match.
    results = match_tracks(ytmusic, spotify_tracks, console)

    exact = _bucket_results(results, MatchType.EXACT)
    multiple = _bucket_results(results, MatchType.MULTIPLE)
    none_found = _bucket_results(results, MatchType.NONE)

    console.print()
    console.print("Results:")
    console.print(f"  [green]✓[/green] {len(exact)} exact matches")
    if multiple:
        console.print(f"  [yellow]?[/yellow] {len(multiple)} need your input")
    if none_found:
        console.print(f"  [red]✗[/red] {len(none_found)} not found")
    console.print()

    # Step 4: Resolve ambiguous matches interactively.
    for result in multiple:
        sp = result.spotify_track
        console.print(f'── [bold]"{sp["name"]}"[/bold] by {sp["artist"]} ──')
        console.print("Multiple matches:")
        for i, candidate in enumerate(result.candidates[:5], 1):
            console.print(_display_candidate(i, candidate))
        skip_idx = len(result.candidates[:5]) + 1
        console.print(f"  {skip_idx}. Skip")

        choice = click.prompt(
            "Choice",
            type=int,
            default=1,
            show_default=True,
        )

        if 1 <= choice <= len(result.candidates[:5]):
            result.selected = result.candidates[choice - 1]
            result.match_type = MatchType.EXACT
        else:
            result.match_type = MatchType.NONE
        console.print()

    # Step 5: Resolve "not found" tracks.
    still_none = _bucket_results(results, MatchType.NONE)
    for result in still_none:
        sp = result.spotify_track
        console.print(f'── [bold]"{sp["name"]}"[/bold] by {sp["artist"]} ──')
        console.print("[red]No match found.[/red]")
        console.print("  1. Search with different query")
        console.print("  2. Enter video ID manually")
        console.print("  3. Skip")

        choice = click.prompt("Choice", type=int, default=3, show_default=True)

        if choice == 1:
            custom_query = click.prompt("Search query")
            try:
                custom_results = ytmusic.search(custom_query, filter="songs", limit=5)
            except Exception:
                custom_results = []

            if custom_results:
                for i, candidate in enumerate(custom_results, 1):
                    console.print(_display_candidate(i, candidate))
                skip_idx = len(custom_results) + 1
                console.print(f"  {skip_idx}. Skip")
                pick = click.prompt("Choice", type=int, default=1, show_default=True)
                if 1 <= pick <= len(custom_results):
                    result.selected = custom_results[pick - 1]
                    result.match_type = MatchType.EXACT
            else:
                console.print("[yellow]No results found.[/yellow]")

        elif choice == 2:
            video_id = click.prompt("Video ID").strip()
            if video_id and VALID_VIDEO_ID.match(video_id):
                result.selected = {"videoId": video_id}
                result.match_type = MatchType.EXACT
            elif video_id:
                console.print("[yellow]Invalid video ID format.[/yellow]")

        console.print()

    # Step 6: Collect confirmed track IDs.
    confirmed, video_ids, skipped = _summarize_results(results)

    if not confirmed:
        console.print("[yellow]No tracks to add. Import cancelled.[/yellow]")
        return

    # Step 7: Let user rename the playlist.
    final_name = click.prompt(
        "Playlist name",
        default=playlist_name,
        show_default=True,
    )

    # Step 8: Create the YTM playlist and add tracks.
    console.print()
    with console.status(f'Creating playlist "{final_name}" on YouTube Music...'):
        try:
            playlist_id = ytmusic.create_playlist(
                final_name,
                f"Imported from Spotify: {playlist_name}",
                privacy_status="PRIVATE",
                video_ids=video_ids,
            )

            if not playlist_id or not isinstance(playlist_id, str):
                console.print("[red]Failed to create playlist.[/red]")
                return

        except Exception as exc:
            console.print(f"[red]Failed to create playlist:[/red] {exc}")
            return

    console.print(
        f"[green]✓[/green] Playlist created with {len(video_ids)} tracks"
        + (f" ({skipped} skipped)" if skipped else "")
    )


# ── Native (Rust/PyO3) overrides, if available ──────────────────────────
# The Rust extension implements the same matching cores; these bindings
# override the pure-Python definitions above when the extension is
# importable.  The rebinding must happen AFTER the definitions above —
# assigning the native functions any earlier would be clobbered by the
# ``def`` statements (the lesson that produced formatting.py's dead
# native bindings).

try:
    from ytm_player.utils._ytm_player_native import (  # type: ignore[import-not-found]
        extract_playlist_id as _native_extract_playlist_id,
    )
    from ytm_player.utils._ytm_player_native import (
        parse_spotipy_item as _native_parse_spotipy_item,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_bucket_results as _native_spotify_bucket_results,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_display_candidate as _native_spotify_display_candidate,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_fuzzy_score as _native_spotify_fuzzy_score,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_match_classification as _native_spotify_match_classification,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_page_items as _native_spotify_page_items,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_scraper_track_rows as _native_spotify_scraper_track_rows,
    )
    from ytm_player.utils._ytm_player_native import (
        spotify_summarize_results as _native_spotify_summarize_results,
    )
except ImportError:
    pass
else:
    _extract_playlist_id = _native_extract_playlist_id  # type: ignore[assignment]
    _parse_spotipy_item = _native_parse_spotipy_item  # type: ignore[assignment]
    _fuzzy_score = _native_spotify_fuzzy_score  # type: ignore[assignment]
    _match_classification = _native_spotify_match_classification  # type: ignore[assignment]
    _page_items = _native_spotify_page_items  # type: ignore[assignment]
    _scraper_track_rows = _native_spotify_scraper_track_rows  # type: ignore[assignment]
    _display_candidate = _native_spotify_display_candidate  # type: ignore[assignment]
    _summarize_results = _native_spotify_summarize_results  # type: ignore[assignment]
    _bucket_results = _native_spotify_bucket_results  # type: ignore[assignment]
