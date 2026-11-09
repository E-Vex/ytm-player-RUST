"""Tests for the Spotify-import popup's fused match-classification wiring.

The popup's two import workers — ``_do_single_import`` and
``_do_multi_import`` — resolve every per-track match verdict through the
import service's ``_match_classification`` engine: the native
``_ytm_player_native.spotify_match_classification`` when the Rust
extension is built, the pure-Python transcription in
``services/spotify_import.py`` otherwise. The popup's own duplicate of
the score-sort-threshold walk (its former ``_classify_match``) is gone;
the search call, the display arms, the disambiguation queue and the
playlist creation stay Python-side around the engine call.

The engine's own contract — the empty-results gate, the stable
best-first sort, the inclusive threshold, the candidate/selection
identities — is pinned in ``tests/test_services/test_spotify_import.py``
(``TestMatchClassification`` plus the child-interpreter parity classes);
these tests pin the wiring: the popup's flows really steer through the
service module's engine name (the very binding the native tail swaps
in), the verdicts flow on into the ✓/?/✗ display arms, the
disambiguation queue and the summary, and no local classification walk
remains to drift.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import ListView, Static

from ytm_player.services import spotify_import as si
from ytm_player.services.spotify_import import MatchType
from ytm_player.ui.popups.spotify_import import SpotifyImportPopup

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# The canned catalogue: Spotify tracks and their YTM search candidates.
# Exact pairs (identical lowercased title+artist) score 100 through the
# real scorer; the miss pair scores far below the 85 threshold.
_SP_TRACK_A = {"name": "Song One", "artist": "Artist One"}
_SP_TRACK_B = {"name": "Song Two", "artist": "Artist Two"}
_YTM_MATCH_A = {
    "title": "Song One",
    "artists": [{"name": "Artist One", "id": "a1"}],
    "video_id": "aaaaaaaaaaa",
}
_YTM_MATCH_B = {
    "title": "Song Two",
    "artists": [{"name": "Artist Two", "id": "a2"}],
    "video_id": "bbbbbbbbbbb",
}
_YTM_MISS = {
    "title": "Completely Different",
    "artists": [{"name": "Someone Else", "id": "m1"}],
    "video_id": "ccccccccccc",
}


class _FakeYTMusic:
    """Stands in for the app's YTMusic service: canned per-query search
    answers, or a raise (a ``None`` entry), with every query recorded."""

    def __init__(self, by_query: dict[str, list[dict] | None] | None = None) -> None:
        self._by_query = by_query or {}
        self.queries: list[str] = []

    async def search(self, query: str, *, filter: str = "", limit: int = 0) -> list[dict]:
        self.queries.append(query)
        answer = self._by_query.get(query)
        if answer is None:
            raise RuntimeError("search exploded")
        return answer


class _Host(App):
    """Minimal host exposing the ``ytmusic`` attribute the popup reads."""

    def __init__(self, ytmusic: _FakeYTMusic) -> None:
        super().__init__()
        self.ytmusic = ytmusic

    def compose(self) -> ComposeResult:
        yield Static("host")


class _Spy:
    """Wraps the engine, recording every ``(track, results)`` pair it is
    handed and answering exactly what the wrapped engine answers."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self.calls: list[tuple[dict, list[dict]]] = []

    def __call__(self, spotify_track: dict, search_results: list[dict]) -> Any:
        self.calls.append((spotify_track, search_results))
        return self._engine(spotify_track, search_results)


def _result_symbols(popup: SpotifyImportPopup) -> list[str]:
    """The rendered result-row symbols, in append order."""
    results_list = popup.query_one("#si-results", ListView)
    return [child._symbol for child in results_list.children]


# ── The duplicated walk stays deleted ───────────────────────────────


class TestFusionDriftPins:
    def test_the_local_classification_walk_is_gone(self):
        """The popup no longer carries its own ``_classify_match``; the
        module source holds none of the classification walk's own
        inputs either, so the only scoring/threshold gate is the
        service engine's."""
        assert not hasattr(SpotifyImportPopup, "_classify_match")

        source = inspect.getsource(SpotifyImportPopup)
        assert "_fuzzy_score" not in source, "the popup must not score candidates itself"
        assert "AUTO_MATCH_THRESHOLD" not in source, "the popup must not gate verdicts itself"

    def test_both_workers_bind_the_service_engine(self):
        """Each import worker imports ``_match_classification`` from the
        service module at call time — the very name the native tail
        swaps, so the binding can never go stale."""
        for worker in (SpotifyImportPopup._do_single_import, SpotifyImportPopup._do_multi_import):
            source = inspect.getsource(worker)
            assert "_match_classification" in source
            assert "_match_classification(sp_track, search_results)" in source


# ── Single-mode wiring ──────────────────────────────────────────────


class TestSingleImportWiring:
    async def test_verdicts_flow_through_the_service_engine(self, monkeypatch):
        """The popup's per-track calls land on the service module's
        ``_match_classification`` name — patched here with a spy — with
        the very track dict and search list the flow built, and the
        engine's MatchResult instances flow on into ``_results``."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_A, _SP_TRACK_B]),
        )
        ytm = _FakeYTMusic(
            by_query={
                "Song One Artist One": [_YTM_MATCH_A],
                "Song Two Artist Two": [_YTM_MATCH_B],
            }
        )
        spy = _Spy(si._match_classification)
        monkeypatch.setattr(si, "_match_classification", spy)

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            # The flow asked the search service once per track.
            assert ytm.queries == ["Song One Artist One", "Song Two Artist Two"]
            # Both tracks steered through the module-level engine name.
            assert spy.calls == [
                (_SP_TRACK_A, [_YTM_MATCH_A]),
                (_SP_TRACK_B, [_YTM_MATCH_B]),
            ]
            # The engine's verdicts landed in the popup's results.
            assert [r.match_type for r in popup._results] == [MatchType.EXACT, MatchType.EXACT]
            assert popup._results[0].selected is _YTM_MATCH_A
            assert popup._results[1].selected is _YTM_MATCH_B

    async def test_exact_matches_render_the_check_arms(self, monkeypatch):
        """An EXACT verdict renders its ✓ row; the flow reaches the
        summary phase with the engine's selection collected as a video
        id for creation. (The display rows are asserted with the
        end-of-flow summary no-op'd — the real summary clears the
        list.)"""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_A]),
        )
        ytm = _FakeYTMusic(by_query={"Song One Artist One": [_YTM_MATCH_A]})

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()
            monkeypatch.setattr(popup, "_show_summary", lambda *args: None)

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            assert _result_symbols(popup) == ["✓"]

    async def test_the_summary_collects_the_engines_selection(self, monkeypatch):
        """Unpatched, the all-EXACT flow lands in the summary phase
        with the selected video ids gathered for creation."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_A]),
        )
        ytm = _FakeYTMusic(by_query={"Song One Artist One": [_YTM_MATCH_A]})

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            assert popup._phase == "summary"
            assert popup._disambig_queue == []
            assert popup._video_ids == ["aaaaaaaaaaa"]
            status = popup.query_one("#si-status", Static)
            assert "1 matched" in str(status._Static__content)

    async def test_failed_search_reaches_the_engine_as_no_results(self, monkeypatch):
        """A raising search becomes the engine's empty-list arm — the ✗
        display row and a NONE verdict, not a crash."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_A]),
        )
        ytm = _FakeYTMusic(by_query={"Song One Artist One": None})
        spy = _Spy(si._match_classification)
        monkeypatch.setattr(si, "_match_classification", spy)

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()
            monkeypatch.setattr(popup, "_show_summary", lambda *args: None)

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            # The flow swallowed the search error into an empty list.
            assert spy.calls == [(_SP_TRACK_A, [])]
            assert popup._results[0].match_type is MatchType.NONE
            assert _result_symbols(popup) == ["✗"]

    async def test_uncertain_matches_enter_disambiguation(self, monkeypatch):
        """A below-threshold verdict renders the ? row, queues the
        result for disambiguation, and shows the engine's best-first
        candidate list to pick from."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_B]),
        )
        ytm = _FakeYTMusic(by_query={"Song Two Artist Two": [_YTM_MISS]})

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()
            monkeypatch.setattr(popup, "_start_disambiguation", lambda *args: None)

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            # The ? row rendered before the disambiguation phase took over.
            assert _result_symbols(popup) == ["?"]
            assert popup._results[0].match_type is MatchType.MULTIPLE
            assert popup._results[0].selected is None
            assert popup._results[0].candidates == [_YTM_MISS]
            assert popup._disambig_queue == [popup._results[0]]

    async def test_the_disambiguation_view_shows_the_engines_candidates(self, monkeypatch):
        """Unpatched, the uncertain flow enters the disambiguation phase
        and mounts the engine's candidates plus the Skip row."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_B]),
        )
        ytm = _FakeYTMusic(by_query={"Song Two Artist Two": [_YTM_MISS]})

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            assert popup._phase == "disambiguate"
            from ytm_player.ui.popups.spotify_import import _CandidateItem, _SkipItem

            results_list = popup.query_one("#si-results", ListView)
            candidates = [c for c in results_list.children if isinstance(c, _CandidateItem)]
            assert [c.candidate for c in candidates] == [_YTM_MISS]
            assert any(isinstance(c, _SkipItem) for c in results_list.children)

    async def test_the_verdicts_ride_the_modules_threshold(self, monkeypatch):
        """Patching the service module's ``AUTO_MATCH_THRESHOLD`` flips
        the popup's verdicts — proof the flow holds no threshold of its
        own and the engine reads the module name at call time."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_A]),
        )
        monkeypatch.setattr(si, "AUTO_MATCH_THRESHOLD", 200)
        ytm = _FakeYTMusic(by_query={"Song One Artist One": [_YTM_MATCH_A]})

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            # A perfect 100 score still falls under a patched 200 gate.
            assert popup._results[0].match_type is MatchType.MULTIPLE
            assert popup._phase == "disambiguate"


# ── Multi-mode wiring ───────────────────────────────────────────────


class TestMultiImportWiring:
    async def test_multi_part_verdicts_flow_through_the_service_engine(self, monkeypatch):
        """The multi-mode walk classifies every part's tracks through
        the same module engine name, skips the engine entirely for a
        part with no tracks, and collects the parts' results in order."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: (
                "Part One" if url.endswith("one") else "Part Two",
                [_SP_TRACK_A] if url.endswith("one") else [],
            ),
        )
        ytm = _FakeYTMusic(by_query={"Song One Artist One": [_YTM_MATCH_A]})
        spy = _Spy(si._match_classification)
        monkeypatch.setattr(si, "_match_classification", spy)

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()
            monkeypatch.setattr(popup, "_show_summary", lambda *args: None)

            popup._multi_name = "Combined"
            # _start_multi_import stages the combined name before the
            # worker runs; mirror that pre-state here.
            popup._playlist_name = "Combined"
            popup._multi_urls = [
                "https://open.spotify.com/playlist/one",
                "https://open.spotify.com/playlist/two",
            ]
            await popup._do_multi_import()
            await pilot.pause()

            # Only the part with tracks reached the engine.
            assert spy.calls == [(_SP_TRACK_A, [_YTM_MATCH_A])]
            assert [r.match_type for r in popup._results] == [MatchType.EXACT]
            assert popup._all_results == popup._results
            assert popup._playlist_name == "Combined"
            # Part headers and rows rendered: the loaded part's ✓, the
            # empty part's warning.
            assert _result_symbols(popup) == ["▸", "✓", "⚠"]
            results_list = popup.query_one("#si-results", ListView)
            texts = [child._text for child in results_list.children]
            assert any("Part 1: Part One (1 tracks)" in text for text in texts)
            assert any("Part 2: no tracks found" in text for text in texts)


# ── Native bindings ─────────────────────────────────────────────────


class TestNativeBindings:
    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    async def test_the_popup_answers_through_the_native_engine(self, monkeypatch):
        """End to end with the extension built: the popup's flow and a
        direct call to the native engine answer the same verdict for
        the same inputs — the popup really rides the native engine."""
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks",
            lambda url: ("Source Playlist", [_SP_TRACK_A, _SP_TRACK_B]),
        )
        ytm = _FakeYTMusic(
            by_query={
                "Song One Artist One": [_YTM_MATCH_A, _YTM_MISS],
                "Song Two Artist Two": [_YTM_MISS],
            }
        )

        assert si._match_classification is _ytm_player_native.spotify_match_classification

        app = _Host(ytm)
        async with app.run_test() as pilot:
            popup = SpotifyImportPopup()
            await app.push_screen(popup)
            await pilot.pause()

            await popup._do_single_import("https://open.spotify.com/playlist/xyz")
            await pilot.pause()

            direct_a = si._match_classification(_SP_TRACK_A, [_YTM_MATCH_A, _YTM_MISS])
            direct_b = si._match_classification(_SP_TRACK_B, [_YTM_MISS])

            assert popup._results[0].match_type is direct_a.match_type is MatchType.EXACT
            assert popup._results[0].selected is _YTM_MATCH_A
            assert popup._results[0].candidates == [_YTM_MATCH_A, _YTM_MISS]
            assert popup._results[1].match_type is direct_b.match_type is MatchType.MULTIPLE
            assert popup._results[1].candidates == [_YTM_MISS]
