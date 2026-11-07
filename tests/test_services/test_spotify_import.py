"""Tests for the Spotify import service (Task T16).

``spotify_import`` shipped two real regressions historically (v1.8.0:
``_get_video_id`` ImportError; a batch-count off-by-99) yet had zero tests.
These cover the pure logic functions — credential I/O, URL/item parsing,
fuzzy scoring, per-track search/scoring, ordered match aggregation, the
spotipy/scraper extraction paths, and the confirmed/skipped accounting that
``run_import`` uses to report results.

The interactive ``run_import`` prompt loop itself (click.prompt / rich Console)
is NOT exercised — only the ``_summarize_results`` accounting helper extracted
from it.

Everything is mocked: ytmusicapi is a ``MagicMock``; ``spotipy`` and
``spotify_scraper`` are injected as fake ``sys.modules`` entries so the tests
run even when those optional packages are not installed. No test hits the
network.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
import types
from difflib import SequenceMatcher
from unittest.mock import MagicMock

import pytest

from ytm_player.services import spotify_import as si


@pytest.fixture(autouse=True)
def _fake_fuzz(monkeypatch):
    """Deterministic stand-in for thefuzz, which is a [spotify]-extra dep.

    CI installs only .[dev], so ``si.fuzz`` may be unbound there
    (``raising=False``). ``_fuzzy_score`` uses nothing but ``fuzz.ratio``,
    and SequenceMatcher has the same shape — 100 for identical strings,
    near-0 for disjoint — so the weighting assertions hold in every
    environment, real thefuzz installed or not.
    """

    class _Fuzz:
        @staticmethod
        def ratio(a: str, b: str) -> int:
            return int(round(SequenceMatcher(None, a, b).ratio() * 100))

    monkeypatch.setattr(si, "fuzz", _Fuzz, raising=False)


# ── helpers ───────────────────────────────────────────────────────────


def _sp_track(name: str = "Song A", artist: str = "Artist A") -> dict:
    """A Spotify-side track dict in this module's internal format."""
    return {"name": name, "artist": artist, "album": "Album A", "duration_ms": 200_000}


def _ytm_candidate(
    title: str = "Song A",
    artist: str = "Artist A",
    duration: int = 200,
    video_id: str = "vid00000001",
    result_type: str = "song",
) -> dict:
    """A YouTube Music search-result candidate (artists-list shape)."""
    return {
        "title": title,
        "artists": [{"name": artist}],
        "duration_seconds": duration,
        "videoId": video_id,
        "resultType": result_type,
    }


def _spotipy_item(name: str, artist: str, album: str = "Al", ms: int = 1000) -> dict:
    """A spotipy playlist item (track nested under 'track')."""
    return {
        "track": {
            "name": name,
            "artists": [{"name": artist}],
            "album": {"name": album},
            "duration_ms": ms,
        }
    }


def _make_result(selected: dict | None, match_type=None) -> si.MatchResult:
    return si.MatchResult(
        spotify_track=_sp_track(),
        match_type=match_type or si.MatchType.EXACT,
        selected=selected,
    )


# ── credential helpers ────────────────────────────────────────────────


class TestCredentialHelpers:
    def test_load_returns_none_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", tmp_path / "nope.json")
        assert si.load_spotify_creds() is None

    def test_load_returns_dict_when_valid(self, tmp_path, monkeypatch):
        f = tmp_path / "spotify.json"
        f.write_text(json.dumps({"client_id": "cid", "client_secret": "sec"}), encoding="utf-8")
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", f)
        creds = si.load_spotify_creds()
        assert creds == {"client_id": "cid", "client_secret": "sec"}

    def test_load_returns_none_on_malformed_json(self, tmp_path, monkeypatch):
        f = tmp_path / "spotify.json"
        f.write_text("{ this is not json", encoding="utf-8")
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", f)
        assert si.load_spotify_creds() is None

    def test_load_returns_none_when_secret_missing(self, tmp_path, monkeypatch):
        f = tmp_path / "spotify.json"
        f.write_text(json.dumps({"client_id": "cid"}), encoding="utf-8")
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", f)
        assert si.load_spotify_creds() is None

    def test_save_then_load_round_trips(self, tmp_path, monkeypatch):
        f = tmp_path / "config" / "spotify.json"
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", f)
        monkeypatch.setattr(si, "CONFIG_DIR", tmp_path / "config")
        si.save_spotify_creds("my-id", "my-secret")
        assert f.exists()
        assert si.load_spotify_creds() == {"client_id": "my-id", "client_secret": "my-secret"}

    def test_has_spotify_creds_reflects_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", tmp_path / "missing.json")
        assert si.has_spotify_creds() is False

        f = tmp_path / "spotify.json"
        f.write_text(json.dumps({"client_id": "c", "client_secret": "s"}), encoding="utf-8")
        monkeypatch.setattr(si, "SPOTIFY_CREDS_FILE", f)
        assert si.has_spotify_creds() is True


# ── URL / item parsing ────────────────────────────────────────────────


class TestParsing:
    def test_extract_playlist_id_from_playlist_url(self):
        url = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        assert si._extract_playlist_id(url) == "37i9dQZF1DXcBWIGoYBM5M"

    def test_extract_playlist_id_from_album_url(self):
        url = "https://open.spotify.com/album/1DFixLWuPkv3KT3TnV35m3?si=x"
        assert si._extract_playlist_id(url) == "1DFixLWuPkv3KT3TnV35m3"

    def test_extract_playlist_id_returns_empty_on_garbage(self):
        assert si._extract_playlist_id("https://example.com/nothing") == ""

    def test_parse_spotipy_item_nested_track(self):
        item = _spotipy_item("Title", "Artist", album="TheAlbum", ms=180_000)
        parsed = si._parse_spotipy_item(item)
        assert parsed == {
            "name": "Title",
            "artist": "Artist",
            "album": "TheAlbum",
            "duration_ms": 180_000,
        }

    def test_parse_spotipy_item_direct_track(self):
        # Some payloads have no "track" wrapper (album tracks endpoint).
        item = {
            "name": "Direct",
            "artists": [{"name": "A"}],
            "album": {"name": "Al"},
            "duration_ms": 5,
        }
        parsed = si._parse_spotipy_item(item)
        assert parsed["name"] == "Direct"
        assert parsed["artist"] == "A"

    def test_parse_spotipy_item_empty_when_no_name(self):
        assert si._parse_spotipy_item({"track": {"name": ""}}) == {}
        assert si._parse_spotipy_item({"track": None}) == {}

    def test_parse_spotipy_item_album_not_dict(self):
        item = {"track": {"name": "N", "artists": [{"name": "A"}], "album": None}}
        parsed = si._parse_spotipy_item(item)
        assert parsed["album"] == ""


# ── fuzzy scoring ─────────────────────────────────────────────────────


class TestFuzzyScore:
    def test_identical_title_and_artist_scores_max(self):
        score = si._fuzzy_score(
            _sp_track("Bohemian Rhapsody", "Queen"), _ytm_candidate("Bohemian Rhapsody", "Queen")
        )
        assert score == 100

    def test_title_weighted_more_than_artist(self):
        # Perfect title, wrong artist vs wrong title, perfect artist.
        title_only = si._fuzzy_score(
            _sp_track("Exactly This Title", "Right Artist"),
            _ytm_candidate("Exactly This Title", "zzzzzzzzzzzz"),
        )
        artist_only = si._fuzzy_score(
            _sp_track("Exactly This Title", "Right Artist"),
            _ytm_candidate("qqqqqqqqqqqq", "Right Artist"),
        )
        assert title_only > artist_only, "title weight (0.6) must dominate artist (0.4)"

    def test_disjoint_strings_score_low(self):
        score = si._fuzzy_score(
            _sp_track("aaaaaaaa", "bbbbbbbb"),
            _ytm_candidate("wxyzwxyz", "qpqpqpqp"),
        )
        assert score < si.AUTO_MATCH_THRESHOLD

    def test_missing_ytm_title_does_not_raise(self):
        # ytm_track with None title must be coerced, not crash.
        score = si._fuzzy_score(_sp_track("X", "Y"), {"title": None, "artists": []})
        assert isinstance(score, int)

    def test_the_fuzz_engine_is_read_at_call_time(self, monkeypatch):
        # A patched fuzz steers whichever backend is loaded — the module
        # attribute is the routing point both engines read.
        class _Table:
            @staticmethod
            def ratio(a: str, b: str) -> int:
                return 7 if a and b else 3

        monkeypatch.setattr(si, "fuzz", _Table)
        # Both ratios 7 → int(7 * 0.6 + 7 * 0.4) == 7.
        assert si._fuzzy_score(_sp_track("A", "B"), _ytm_candidate("C", "D")) == 7

    def test_the_weights_are_read_at_call_time(self, monkeypatch):
        # Patched weights steer whichever backend is loaded identically.
        monkeypatch.setattr(si, "TITLE_MATCH_WEIGHT", 1.0)
        monkeypatch.setattr(si, "ARTIST_MATCH_WEIGHT", 0.0)
        sp = _sp_track("Same", "Irrelevant")
        # Title ratio 100 (identical), artist ratio whatever → 100.
        assert si._fuzzy_score(sp, _ytm_candidate("Same", "zzzz")) == 100


# ── _search_and_score ─────────────────────────────────────────────────


class TestSearchAndScore:
    def test_no_results_yields_none_match(self):
        ytm = MagicMock()
        ytm.search.return_value = []
        idx, result = si._search_and_score(ytm, _sp_track(), 3)
        assert idx == 3
        assert result.match_type is si.MatchType.NONE
        assert result.selected is None

    def test_search_exception_is_treated_as_no_results(self):
        ytm = MagicMock()
        ytm.search.side_effect = RuntimeError("network boom")
        idx, result = si._search_and_score(ytm, _sp_track(), 0)
        assert result.match_type is si.MatchType.NONE

    def test_high_score_yields_exact_with_selection(self):
        ytm = MagicMock()
        ytm.search.return_value = [_ytm_candidate("Song A", "Artist A", video_id="perfect")]
        _, result = si._search_and_score(ytm, _sp_track("Song A", "Artist A"), 0)
        assert result.match_type is si.MatchType.EXACT
        assert result.selected is not None
        assert result.selected["videoId"] == "perfect"

    def test_low_score_yields_multiple_without_selection(self):
        ytm = MagicMock()
        ytm.search.return_value = [_ytm_candidate("nothing alike", "different too")]
        _, result = si._search_and_score(ytm, _sp_track("Song A", "Artist A"), 0)
        assert result.match_type is si.MatchType.MULTIPLE
        assert result.selected is None
        assert len(result.candidates) == 1

    def test_candidates_sorted_best_first(self):
        ytm = MagicMock()
        ytm.search.return_value = [
            _ytm_candidate("totally wrong", "totally wrong", video_id="bad"),
            _ytm_candidate("Song A", "Artist A", video_id="good"),
        ]
        _, result = si._search_and_score(ytm, _sp_track("Song A", "Artist A"), 0)
        assert result.match_type is si.MatchType.EXACT
        assert result.candidates[0]["videoId"] == "good"
        assert result.selected["videoId"] == "good"

    def test_query_combines_name_and_artist(self):
        ytm = MagicMock()
        ytm.search.return_value = []
        si._search_and_score(ytm, _sp_track("My Song", "My Artist"), 0)
        called_query = ytm.search.call_args[0][0]
        assert "My Song" in called_query and "My Artist" in called_query


# ── _match_classification (the post-search verdict) ───────────────────


class TestMatchClassification:
    """The classification engine behind _search_and_score's walk, on
    whichever backend is loaded. The scripted _fuzzy_score patches below
    steer both backends: the engine (and the fallback) resolve the
    scorer through the module's own attribute at call time."""

    def _script_scores(self, monkeypatch, table: dict[str, int]):
        def fake_score(_sp, candidate):
            return table[candidate["videoId"]]

        monkeypatch.setattr(si, "_fuzzy_score", fake_score)

    def test_empty_results_answer_none(self):
        result = si._match_classification(_sp_track(), [])
        assert result.match_type is si.MatchType.NONE
        assert result.candidates == []
        assert result.selected is None

    def test_falsy_results_answer_none(self):
        # The gate is truthiness: an empty mapping answers NONE too.
        result = si._match_classification(_sp_track(), {})
        assert result.match_type is si.MatchType.NONE

    def test_best_first_exact_with_identity_selection(self, monkeypatch):
        good = _ytm_candidate("Song A", "Artist A", video_id="good")
        bad = _ytm_candidate("nope", "nope", video_id="bad")
        self._script_scores(monkeypatch, {"good": 100, "bad": 10})
        result = si._match_classification(_sp_track(), [bad, good])
        assert result.match_type is si.MatchType.EXACT
        assert result.selected is good  # the very object, not a copy
        assert result.candidates[0] is good
        assert result.candidates[1] is bad

    def test_threshold_is_inclusive(self, monkeypatch):
        cand = _ytm_candidate("Edge", "Case", video_id="e")
        self._script_scores(monkeypatch, {"e": si.AUTO_MATCH_THRESHOLD})
        result = si._match_classification(_sp_track(), [cand])
        assert result.match_type is si.MatchType.EXACT

    def test_below_threshold_answers_multiple(self, monkeypatch):
        cand = _ytm_candidate("Edge", "Case", video_id="e")
        self._script_scores(monkeypatch, {"e": si.AUTO_MATCH_THRESHOLD - 1})
        result = si._match_classification(_sp_track(), [cand])
        assert result.match_type is si.MatchType.MULTIPLE
        assert result.selected is None
        assert result.candidates == [cand]

    def test_equal_scores_keep_first_seen_order(self, monkeypatch):
        first = _ytm_candidate("Twin", "A", video_id="first")
        second = _ytm_candidate("Twin", "B", video_id="second")
        self._script_scores(monkeypatch, {"first": 90, "second": 90})
        result = si._match_classification(_sp_track(), [first, second])
        # list.sort(key=..., reverse=True) is stable: ties keep input order.
        assert result.candidates[0] is first
        assert result.candidates[1] is second

    def test_spotify_track_passes_through_by_identity(self, monkeypatch):
        sp = _sp_track("Identity", "Check")
        self._script_scores(monkeypatch, {})
        result = si._match_classification(sp, [])
        assert result.spotify_track is sp

    def test_the_scorer_is_read_at_call_time(self, monkeypatch):
        # Patching si._fuzzy_score steers the classification on whichever
        # backend is loaded — the routing point both engines share.
        calls: list[dict] = []

        def fake_score(sp, candidate):
            calls.append(candidate)
            return 100

        monkeypatch.setattr(si, "_fuzzy_score", fake_score)
        candidates = [_ytm_candidate(video_id="x"), _ytm_candidate(video_id="y")]
        result = si._match_classification(_sp_track(), candidates)
        assert result.match_type is si.MatchType.EXACT
        assert calls == candidates


# ── match_tracks (aggregation + ordering) ─────────────────────────────


def _quiet_console():
    from rich.console import Console

    return Console(file=io.StringIO(), force_terminal=False)


class TestMatchTracks:
    def test_returns_empty_when_deps_missing(self, monkeypatch):
        monkeypatch.setattr(si, "_HAS_SPOTIFY_DEPS", False)
        assert si.match_tracks(MagicMock(), [_sp_track()], MagicMock()) == []

    def test_preserves_input_order(self, monkeypatch):
        monkeypatch.setattr(si, "_HAS_SPOTIFY_DEPS", True)
        tracks = [_sp_track(f"Song {i}", f"Artist {i}") for i in range(6)]

        ytm = MagicMock()

        def fake_search(query, **_kw):
            # Echo back a candidate whose title matches the queried song.
            # Earlier tracks sleep LONGER so thread completion order is the
            # reverse of input order — the slot-by-index logic must still
            # return results in input order.
            name = query.rsplit(" ", 2)[0]
            idx = int(name.split()[-1])
            time.sleep((len(tracks) - idx) * 0.01)
            return [_ytm_candidate(name, "x")]

        ytm.search.side_effect = fake_search
        results = si.match_tracks(ytm, tracks, _quiet_console())

        assert len(results) == len(tracks)
        assert [r.spotify_track["name"] for r in results] == [t["name"] for t in tracks]

    def test_categorizes_matches(self, monkeypatch):
        monkeypatch.setattr(si, "_HAS_SPOTIFY_DEPS", True)
        tracks = [_sp_track("Hit", "Star"), _sp_track("Miss", "Nobody")]

        ytm = MagicMock()

        def fake_search(query, **_kw):
            if query.startswith("Hit"):
                return [_ytm_candidate("Hit", "Star")]
            return []  # "Miss" finds nothing

        ytm.search.side_effect = fake_search
        results = si.match_tracks(ytm, tracks, _quiet_console())
        by_name = {r.spotify_track["name"]: r for r in results}
        assert by_name["Hit"].match_type is si.MatchType.EXACT
        assert by_name["Miss"].match_type is si.MatchType.NONE


# ── _display_candidate ────────────────────────────────────────────────


class TestDisplayCandidate:
    def test_formats_title_artist_and_duration(self):
        line = si._display_candidate(1, _ytm_candidate("Track", "Band", duration=125))
        assert "1." in line and "Track" in line and "Band" in line and "2:05" in line

    def test_unknown_duration_renders_question_mark(self):
        cand = {"title": "T", "artists": [{"name": "A"}]}
        line = si._display_candidate(2, cand)
        assert "(?)" in line

    def test_non_song_result_type_gets_suffix(self):
        line = si._display_candidate(1, _ytm_candidate(result_type="video"))
        assert "[video]" in line

    def test_song_result_type_has_no_suffix(self):
        line = si._display_candidate(1, _ytm_candidate(result_type="song"))
        assert "[song]" not in line

    def test_the_line_shape_is_pinned_exactly(self):
        # The reference's f-string: two leading spaces, "<idx>. ", the
        # em dash around the artist, the duration in parens.
        line = si._display_candidate(3, _ytm_candidate("T", "A", duration=65))
        assert line == "  3. T — A (1:05)"

    def test_a_missing_title_reads_the_question_mark_default(self):
        line = si._display_candidate(1, {"artists": [{"name": "A"}]})
        assert line.startswith("  1. ? — A (")


# ── _summarize_results (accounting — off-by-99 / partial-failure) ─────


class TestSummarizeResults:
    def test_empty_results(self):
        confirmed, video_ids, skipped = si._summarize_results([])
        assert confirmed == [] and video_ids == [] and skipped == 0

    def test_all_confirmed(self):
        results = [_make_result({"videoId": f"v{i}"}) for i in range(3)]
        confirmed, video_ids, skipped = si._summarize_results(results)
        assert len(confirmed) == 3
        assert video_ids == ["v0", "v1", "v2"]
        assert skipped == 0

    def test_unselected_counted_as_skipped(self):
        results = [
            _make_result({"videoId": "keep"}),
            _make_result(None, si.MatchType.NONE),
            _make_result(None, si.MatchType.NONE),
        ]
        confirmed, video_ids, skipped = si._summarize_results(results)
        assert video_ids == ["keep"]
        assert skipped == 2

    def test_confirmed_with_empty_video_id_dropped_but_not_skipped(self):
        # Manual video-ID entry could leave an empty id: it drops out of the
        # playable list but is NOT double-counted as a skip.
        results = [
            _make_result({"videoId": "good"}),
            _make_result({"videoId": ""}),
        ]
        confirmed, video_ids, skipped = si._summarize_results(results)
        assert video_ids == ["good"]
        assert len(confirmed) == 2
        assert skipped == 0

    def test_one_skip_out_of_hundred_is_not_off_by_99(self):
        # Guards the historical off-by-99: 100 tracks, exactly one unmatched.
        results = [_make_result({"videoId": f"v{i}"}) for i in range(99)]
        results.append(_make_result(None, si.MatchType.NONE))
        confirmed, video_ids, skipped = si._summarize_results(results)
        assert len(video_ids) == 99, "must add 99, not 1"
        assert skipped == 1, "must skip 1, not 99"

    def test_reads_both_videoid_key_conventions(self):
        results = [
            _make_result({"videoId": "camel"}),
            _make_result({"video_id": "snake"}),
        ]
        _, video_ids, _ = si._summarize_results(results)
        assert video_ids == ["camel", "snake"]

    def test_confirmed_results_pass_through_by_identity(self):
        # The very objects, not copies — run_import mutates them later.
        results = [
            _make_result({"videoId": "a"}),
            _make_result(None, si.MatchType.NONE),
            _make_result({"videoId": "b"}),
        ]
        confirmed, _, _ = si._summarize_results(results)
        assert confirmed[0] is results[0]
        assert confirmed[1] is results[2]

    def test_a_selected_without_a_dict_shape_raises_like_the_reference(self):
        # A truthy non-dict selection is outside the ported contract; the
        # native engine answers its TypeError where the fallback answers
        # the duck-typed AttributeError — both refuse it.
        result = _make_result("not-a-dict")
        with pytest.raises((TypeError, AttributeError)):
            si._summarize_results([result])


# ── _bucket_results (the match-type bucket walk) ─────────────────────


class TestBucketResults:
    def test_buckets_split_by_match_type(self):
        exact = _make_result({"videoId": "a"})
        multiple = _make_result(None, si.MatchType.MULTIPLE)
        none_ = _make_result(None, si.MatchType.NONE)
        results = [exact, none_, multiple]
        assert si._bucket_results(results, si.MatchType.EXACT) == [exact]
        assert si._bucket_results(results, si.MatchType.MULTIPLE) == [multiple]
        assert si._bucket_results(results, si.MatchType.NONE) == [none_]

    def test_results_pass_through_by_identity_in_order(self):
        first = _make_result(None, si.MatchType.NONE)
        second = _make_result(None, si.MatchType.NONE)
        interloper = _make_result({"videoId": "x"})
        bucket = si._bucket_results([first, interloper, second], si.MatchType.NONE)
        assert bucket == [first, second]
        assert bucket[0] is first and bucket[1] is second

    def test_empty_results_answer_empty(self):
        assert si._bucket_results([], si.MatchType.EXACT) == []


# ── _page_items (one page of the spotipy pagination walk) ────────────


class TestPageItems:
    def test_parses_one_pages_items(self):
        page = {"items": [{"track": _item("A", "X")}, {"track": _item("B", "Y")}]}
        rows = si._page_items(page)
        assert [r["name"] for r in rows] == ["A", "B"]

    def test_falsy_parses_are_dropped(self):
        page = {"items": [{"track": None}, {"track": {}}, {"track": _item("V", "A")}]}
        rows = si._page_items(page)
        assert [r["name"] for r in rows] == ["V"]

    def test_a_missing_items_key_answers_no_rows(self):
        assert si._page_items({}) == []
        assert si._page_items({"next": "url"}) == []

    def test_an_empty_items_list_answers_no_rows(self):
        assert si._page_items({"items": []}) == []

    def test_a_non_iterable_items_member_raises_the_iteration_error(self):
        with pytest.raises(TypeError):
            si._page_items({"items": 5})

    def test_the_parse_engine_is_read_at_call_time(self, monkeypatch):
        # Patching _parse_spotipy_item steers the walk on whichever
        # backend is loaded — the routing point both engines share.
        monkeypatch.setattr(si, "_parse_spotipy_item", lambda item: {"name": "patched"})
        rows = si._page_items({"items": [{"track": _item("A", "X")}]})
        assert rows == [{"name": "patched"}]


# ── extract_spotify_tracks_spotipy (Web API path) ─────────────────────


@pytest.fixture
def fake_spotipy(monkeypatch):
    """Inject fake ``spotipy`` + ``spotipy.oauth2`` modules.

    Returns the ``sp`` MagicMock so tests can set ``.playlist`` / ``.album``
    / ``.next`` return values. ``SpotifyClientCredentials`` is a MagicMock so
    call args can be asserted.
    """
    sp = MagicMock(name="spotify-client")
    creds_ctor = MagicMock(name="SpotifyClientCredentials", return_value="auth")

    mod = types.ModuleType("spotipy")
    mod.Spotify = MagicMock(return_value=sp)  # type: ignore[attr-defined]
    oauth = types.ModuleType("spotipy.oauth2")
    oauth.SpotifyClientCredentials = creds_ctor  # type: ignore[attr-defined]
    mod.oauth2 = oauth  # type: ignore[attr-defined]

    monkeypatch.setitem(__import__("sys").modules, "spotipy", mod)
    monkeypatch.setitem(__import__("sys").modules, "spotipy.oauth2", oauth)

    sp._creds_ctor = creds_ctor  # stash for assertions
    sp._Spotify = mod.Spotify
    return sp


class TestExtractSpotipy:
    def test_raises_without_credentials(self, fake_spotipy, monkeypatch):
        monkeypatch.setattr(si, "load_spotify_creds", lambda: None)
        with pytest.raises(RuntimeError, match="credentials not configured"):
            si.extract_spotify_tracks_spotipy("https://open.spotify.com/playlist/abc")

    def test_raises_on_unparseable_url(self, fake_spotipy, monkeypatch):
        monkeypatch.setattr(
            si, "load_spotify_creds", lambda: {"client_id": "c", "client_secret": "s"}
        )
        with pytest.raises(RuntimeError, match="Could not parse playlist ID"):
            si.extract_spotify_tracks_spotipy("https://open.spotify.com/foo/")

    def test_playlist_with_pagination(self, fake_spotipy, monkeypatch):
        monkeypatch.setattr(
            si, "load_spotify_creds", lambda: {"client_id": "c", "client_secret": "s"}
        )
        page1 = {"items": [_spotipy_item("S1", "A1"), _spotipy_item("S2", "A2")], "next": "url2"}
        page2 = {"items": [_spotipy_item("S3", "A3")], "next": None}
        fake_spotipy.playlist.return_value = {"name": "My Playlist", "tracks": page1}
        fake_spotipy.next.return_value = page2

        name, tracks = si.extract_spotify_tracks_spotipy(
            "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"
        )
        assert name == "My Playlist"
        assert [t["name"] for t in tracks] == ["S1", "S2", "S3"]
        fake_spotipy.next.assert_called_once_with(page1)
        fake_spotipy._creds_ctor.assert_called_once_with(client_id="c", client_secret="s")

    def test_album_path(self, fake_spotipy, monkeypatch):
        monkeypatch.setattr(
            si, "load_spotify_creds", lambda: {"client_id": "c", "client_secret": "s"}
        )
        fake_spotipy.album.return_value = {
            "name": "My Album",
            "tracks": {"items": [_spotipy_item("T1", "A1")], "next": None},
        }
        name, tracks = si.extract_spotify_tracks_spotipy(
            "https://open.spotify.com/album/1DFixLWuPkv3KT3TnV35m3"
        )
        assert name == "My Album"
        assert [t["name"] for t in tracks] == ["T1"]
        fake_spotipy.album.assert_called_once()

    def test_raises_when_album_api_returns_none(self, fake_spotipy, monkeypatch):
        monkeypatch.setattr(
            si, "load_spotify_creds", lambda: {"client_id": "c", "client_secret": "s"}
        )
        fake_spotipy.album.return_value = None
        with pytest.raises(RuntimeError, match="no data for album"):
            si.extract_spotify_tracks_spotipy("https://open.spotify.com/album/abc123")


# ── extract_spotify_tracks (dispatcher + scraper fallback) ────────────


def _scraper_track(name: str, artist: str, album: str | None = "Al", ms: int = 1000):
    """A fake ``spotify_scraper.models.track.Track``, attribute-shaped only.

    Built with ``SimpleNamespace`` rather than importing the real dataclass,
    since ``spotify_scraper`` is an optional dependency the test suite must
    not require.
    """
    return types.SimpleNamespace(
        name=name,
        duration_ms=ms,
        artists=(types.SimpleNamespace(name=artist),),
        album=types.SimpleNamespace(name=album) if album else None,
    )


def _scraper_item(name: str, artist: str, album: str | None = "Al", ms: int = 1000):
    """A fake ``spotify_scraper.models.playlist.PlaylistTrack``."""
    return types.SimpleNamespace(track=_scraper_track(name, artist, album, ms))


def _scraper_playlist(name: str, total_tracks: int, tracks: list):
    """A fake ``spotify_scraper.models.playlist.Playlist``."""
    return types.SimpleNamespace(name=name, total_tracks=total_tracks, tracks=tuple(tracks))


def _install_fake_scraper(monkeypatch, playlist):
    """Inject a fake ``spotify_scraper`` module; return the client mock."""
    client = MagicMock(name="scraper-client")
    client.get_playlist.return_value = playlist
    mod = types.ModuleType("spotify_scraper")
    mod.SpotifyClient = MagicMock(return_value=client)  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "spotify_scraper", mod)
    return client


class TestScraperTrackRows:
    def test_rows_carry_the_four_keys_in_order(self):
        playlist = _scraper_playlist("PL", 1, [_scraper_item("S", "A")])
        rows = si._scraper_track_rows(playlist)
        assert rows == [{"name": "S", "artist": "A", "album": "Al", "duration_ms": 1000}]
        assert list(rows[0]) == ["name", "artist", "album", "duration_ms"]

    def test_multiple_artists_join_with_comma_space(self):
        track = types.SimpleNamespace(
            name="Collab",
            duration_ms=1000,
            artists=(
                types.SimpleNamespace(name="First"),
                types.SimpleNamespace(name="Second"),
            ),
            album=types.SimpleNamespace(name="Album"),
        )
        playlist = types.SimpleNamespace(
            name="PL", total_tracks=1, tracks=(types.SimpleNamespace(track=track),)
        )
        assert si._scraper_track_rows(playlist)[0]["artist"] == "First, Second"

    def test_no_artists_answer_unknown(self):
        track = types.SimpleNamespace(name="Solo", duration_ms=1000, artists=(), album=None)
        playlist = types.SimpleNamespace(
            name="PL", total_tracks=1, tracks=(types.SimpleNamespace(track=track),)
        )
        rows = si._scraper_track_rows(playlist)
        assert rows[0]["artist"] == "Unknown"
        assert rows[0]["album"] == ""

    def test_a_falsy_album_answers_the_empty_string(self):
        # A present-but-falsy album (not just None) reads as "".
        track = types.SimpleNamespace(
            name="N", duration_ms=1, artists=(types.SimpleNamespace(name="A"),), album=[]
        )
        playlist = types.SimpleNamespace(
            name="PL", total_tracks=1, tracks=(types.SimpleNamespace(track=track),)
        )
        assert si._scraper_track_rows(playlist)[0]["album"] == ""

    def test_a_truthy_album_without_a_name_raises(self):
        track = types.SimpleNamespace(
            name="N",
            duration_ms=1,
            artists=(types.SimpleNamespace(name="A"),),
            album=types.SimpleNamespace(poster="x"),
        )
        playlist = types.SimpleNamespace(
            name="PL", total_tracks=1, tracks=(types.SimpleNamespace(track=track),)
        )
        with pytest.raises(AttributeError):
            si._scraper_track_rows(playlist)

    def test_a_non_string_artist_name_raises_the_joins_own_error(self):
        track = types.SimpleNamespace(
            name="N", duration_ms=1, artists=(types.SimpleNamespace(name=5),), album=None
        )
        playlist = types.SimpleNamespace(
            name="PL", total_tracks=1, tracks=(types.SimpleNamespace(track=track),)
        )
        with pytest.raises(TypeError):
            si._scraper_track_rows(playlist)

    def test_values_pass_through_as_the_objects_they_are(self):
        track = types.SimpleNamespace(
            name="N",
            duration_ms=180_000,
            artists=(types.SimpleNamespace(name="A"),),
            album=types.SimpleNamespace(name="Al"),
        )
        playlist = types.SimpleNamespace(
            name="PL", total_tracks=1, tracks=(types.SimpleNamespace(track=track),)
        )
        rows = si._scraper_track_rows(playlist)
        assert rows[0]["name"] is track.name
        assert rows[0]["duration_ms"] is track.duration_ms


class TestExtractDispatcher:
    def test_uses_spotipy_when_creds_present(self, monkeypatch):
        monkeypatch.setattr(si, "has_spotify_creds", lambda: True)
        monkeypatch.setattr(
            si, "extract_spotify_tracks_spotipy", lambda url: ("Full PL", [{"name": "x"}])
        )
        name, tracks = si.extract_spotify_tracks("https://open.spotify.com/playlist/x")
        assert name == "Full PL"
        assert tracks == [{"name": "x"}]

    def test_falls_back_to_scraper_when_spotipy_fails(self, monkeypatch):
        monkeypatch.setattr(si, "has_spotify_creds", lambda: True)

        def boom(url):
            raise RuntimeError("spotipy exploded")

        monkeypatch.setattr(si, "extract_spotify_tracks_spotipy", boom)
        client = _install_fake_scraper(
            monkeypatch,
            _scraper_playlist("Scraped", 2, [_scraper_item("S1", "A1"), _scraper_item("S2", "A2")]),
        )
        name, tracks = si.extract_spotify_tracks("https://open.spotify.com/playlist/x")
        assert name == "Scraped"
        assert [t["name"] for t in tracks] == ["S1", "S2"]
        client.close.assert_called_once()

    def test_scraper_used_directly_without_creds(self, monkeypatch):
        monkeypatch.setattr(si, "has_spotify_creds", lambda: False)
        # If spotipy were consulted this would blow up; assert it is not.
        monkeypatch.setattr(
            si,
            "extract_spotify_tracks_spotipy",
            MagicMock(side_effect=AssertionError("spotipy must not be called")),
        )
        client = _install_fake_scraper(
            monkeypatch, _scraper_playlist("S", 1, [_scraper_item("Only", "A")])
        )
        name, tracks = si.extract_spotify_tracks("https://open.spotify.com/playlist/x")
        assert name == "S"
        assert [t["name"] for t in tracks] == ["Only"]
        client.close.assert_called_once()

    def test_fetches_without_a_track_cap(self, monkeypatch):
        """No more hardcoded ~100-track ceiling: max_tracks=None fetches all."""
        monkeypatch.setattr(si, "has_spotify_creds", lambda: False)
        client = _install_fake_scraper(
            monkeypatch, _scraper_playlist("S", 1, [_scraper_item("Only", "A")])
        )
        si.extract_spotify_tracks("https://open.spotify.com/playlist/x")
        client.get_playlist.assert_called_once_with(
            "https://open.spotify.com/playlist/x", max_tracks=None
        )

    def test_truncation_is_logged(self, monkeypatch, caplog):
        import logging

        monkeypatch.setattr(si, "has_spotify_creds", lambda: False)
        _install_fake_scraper(
            monkeypatch, _scraper_playlist("Big", 500, [_scraper_item("One", "A")])
        )
        with caplog.at_level(logging.WARNING, logger=si.logger.name):
            si.extract_spotify_tracks("https://open.spotify.com/playlist/x")
        assert any("scraper returned" in rec.message.lower() for rec in caplog.records)


# ── native vs fallback parity: the matching cores ──────────────────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# The URL zoo: playlist and album links (the two real shapes), query
# suffixes, ids that embed the other marker word, URI-form links (no
# slash — no match), plural paths, case-mismatched markers, markers
# inside the id, empty and non-Spotify URLs, and a URL whose first
# match is the album marker even though a playlist marker follows.
_URL_ZOO = [
    "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
    "https://open.spotify.com/album/1DFixLWuPkv3KT3TnV35m3",
    "https://open.spotify.com/album/1DFixLWuPkv3KT3TnV35m3?si=abc",
    "open.spotify.com/playlist/short",
    "https://open.spotify.com/playlist/albumabc",
    "https://open.spotify.com/playlist/37i9d/album/1DFi",
    "spotify:playlist:37i9dQZF1DXcBWIGoYBM5M",
    "spotify:album:1DFixLWuPkv3KT3TnV35m3",
    "https://open.spotify.com/playlists/37i9dQZF1DXcBWIGoYBM5M",
    "https://open.spotify.com/PLAYLIST/37i9dQZF1DXcBWIGoYBM5M",
    "https://open.spotify.com/Album/1DFixLWuPkv3KT3TnV35m3",
    "https://open.spotify.com/playlist/",
    "https://open.spotify.com/playlist/?si=x",
    "https://open.spotify.com/playlist/a-b_c",
    "https://example.com/nothing",
    "",
    "playlist/no-scheme",
    "album/42",
    "see /playlist/x for more",
    "https://open.spotify.com/album/1DFixLWuPkv3KT3TnV35m3/playlist/zzz",
]


def _item(
    name="Song",
    artists=("Artist",),
    album="Album",
    duration_ms=1000,
    artist=None,
):
    track: dict = {"name": name, "duration_ms": duration_ms}
    if artist is not None:
        track["artist"] = artist
    else:
        track["artists"] = [{"name": a} for a in artists]
    if album is not None:
        track["album"] = {"name": album}
    return track


# The item zoo: the nested-track shape, the direct (album-endpoint)
# shape, the falsy gates (None track, empty track, missing/empty/None
# name), the artist fallbacks (direct artist key, artists list, the
# view-count pseudo-artist exclusion, no artists at all), the album
# shapes (dict without name, name None, non-dict album, absent album),
# and the duration_ms default-vs-None distinction.
_ITEM_ZOO = [
    {"track": _item()},
    {"track": _item(name="Ünïcode Şong", artists=("Årtist", "Othër"), album="Ålbüm")},
    {"track": _item(duration_ms=None)},
    {"track": _item(album=None)},
    _item(name="Direct"),
    _item(name="No Album Field", album=None),
    {"track": None},
    {"track": {}},
    {"track": {"name": ""}},
    {"track": {"name": None}},
    {"track": {"name": 0}},
    {},
    {"track": _item(artist="Direct Artist")},
    {"track": _item(artists=())},
    {"track": {"name": "V", "artists": [{"name": "358K views"}], "views": "358K views"}},
    {
        "track": {
            "name": "V2",
            "artists": [{"name": "Real"}, {"name": "1,234 views"}],
            "views": "1,234 views",
            "duration_ms": 0,
        }
    },
    {"track": {"name": "N", "album": {}}},
    {"track": {"name": "N", "album": {"name": None}}},
    {"track": {"name": "N", "album": "Plain String"}},
    {"track": {"name": "N"}},
    {"track": {"name": "ᴱᵐᵖᵗʸ", "artists": [{"name": ""}, {"name": "K"}]}},
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import spotify_import as si

urls, items = json.load(sys.stdin)
out = {
    "ids": [si._extract_playlist_id(u) for u in urls],
    "parsed": [si._parse_spotipy_item(i) for i in items],
}
json.dump(out, sys.stdout)
"""


class TestMatchingCoresNativeParity:
    """The URL-parse and item-normalise cores, on the child-interpreter
    pattern the other parity tests use: the fallback runs with the
    extension import blocked, so both implementations are compared
    directly instead of trusting whichever one the test process loaded.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_matching_cores_match_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps([_URL_ZOO, _ITEM_ZOO]),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        ids = [si._extract_playlist_id(url) for url in _URL_ZOO]
        for url, nv, fb in zip(_URL_ZOO, ids, fallback["ids"]):
            assert nv == fb, f"playlist ids differ for {url!r}: {nv!r} vs {fb!r}"

        parsed = [si._parse_spotipy_item(item) for item in _ITEM_ZOO]
        for item, nv, fb in zip(_ITEM_ZOO, parsed, fallback["parsed"]):
            assert nv == fb, f"parsed items differ for {item!r}:\n{nv!r}\nvs\n{fb!r}"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_native_engines_really_answer(self):
        """The bindings are live: the module's cores are the native
        functions themselves when the extension is built."""
        from ytm_player.utils._ytm_player_native import (
            extract_playlist_id,
            parse_spotipy_item,
            spotify_bucket_results,
            spotify_display_candidate,
            spotify_fuzzy_score,
            spotify_match_classification,
            spotify_page_items,
            spotify_scraper_track_rows,
            spotify_summarize_results,
        )

        assert si._extract_playlist_id is extract_playlist_id
        assert si._parse_spotipy_item is parse_spotipy_item
        assert si._fuzzy_score is spotify_fuzzy_score
        assert si._match_classification is spotify_match_classification
        assert si._page_items is spotify_page_items
        assert si._scraper_track_rows is spotify_scraper_track_rows
        assert si._display_candidate is spotify_display_candidate
        assert si._summarize_results is spotify_summarize_results
        assert si._bucket_results is spotify_bucket_results

    def test_url_zoo_contracts(self):
        """Pin the shared contract on whichever backend is loaded."""
        assert (
            si._extract_playlist_id("https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M")
            == "37i9dQZF1DXcBWIGoYBM5M"
        )
        # The first marker wins: an album marker inside a playlist id is
        # part of the id, and a URL with both takes the earlier one.
        assert si._extract_playlist_id("https://open.spotify.com/playlist/albumabc") == "albumabc"
        assert si._extract_playlist_id("https://open.spotify.com/album/1DFi/playlist/zzz") == "1DFi"
        # URI form, plural paths and case mismatches have no match.
        assert si._extract_playlist_id("spotify:playlist:37i9dQZF1DXcBWIGoYBM5M") == ""
        assert si._extract_playlist_id("https://open.spotify.com/playlists/abc") == ""
        assert si._extract_playlist_id("https://open.spotify.com/PLAYLIST/abc") == ""
        # An id-less marker is no match; the char class is [a-zA-Z0-9].
        assert si._extract_playlist_id("https://open.spotify.com/playlist/") == ""
        assert si._extract_playlist_id("https://open.spotify.com/playlist/a-b_c") == "a"

    def test_item_zoo_contracts(self):
        """Pin the shared contract on whichever backend is loaded."""
        assert si._parse_spotipy_item({"track": _item()}) == {
            "name": "Song",
            "artist": "Artist",
            "album": "Album",
            "duration_ms": 1000,
        }
        # The falsy gates.
        assert si._parse_spotipy_item({"track": None}) == {}
        assert si._parse_spotipy_item({"track": {}}) == {}
        assert si._parse_spotipy_item({"track": {"name": ""}}) == {}
        # duration_ms: absent is 0, present-None is None (a .get default
        # never replaces a present value).
        assert si._parse_spotipy_item({"track": _item()})["duration_ms"] == 1000
        assert si._parse_spotipy_item(_item())["duration_ms"] == 1000
        assert si._parse_spotipy_item({"track": {"name": "N"}})["duration_ms"] == 0
        # The artist chain: direct key, artists list, view-count runs
        # dropped, no artists at all.
        assert si._parse_spotipy_item({"track": _item(artist="Solo")})["artist"] == "Solo"
        assert si._parse_spotipy_item({"track": _item(artists=("A", "B"))})["artist"] == "A, B"
        assert (
            si._parse_spotipy_item({"track": {"name": "V", "artists": []}})["artist"] == "Unknown"
        )
        # The album chain: only a dict's name is read; a present-None
        # name passes through, a non-dict album is "".
        assert si._parse_spotipy_item({"track": {"name": "N", "album": {}}})["album"] == ""
        assert si._parse_spotipy_item({"track": {"name": "N", "album": "S"}})["album"] == ""


# ── native vs fallback parity: the matching pipeline ──────────────────

# The pipeline parity cases, JSON-round-trippable, keyed by engine. The
# child interpreter runs the pure-Python transcriptions with the
# extension import blocked (and the same deterministic fuzz stand-in the
# autouse fixture installs here); the parent runs the native engines the
# module's tail bound. Cases that raise pin the error shape too.
_PIPELINE_CASES = {
    "fuzzy": [
        # (spotify_track, ytm_candidate) pairs.
        [
            {"name": "Song A", "artist": "Artist A"},
            {"title": "Song A", "artists": [{"name": "Artist A"}]},
        ],
        [
            {"name": "Hypnotize", "artist": "The Notorious B.I.G."},
            {"title": "Hypnotize", "artists": [{"name": "Notorious B.I.G."}]},
        ],
        [
            {"name": "Ünïcode Şong", "artist": "Årtist"},
            {"title": "üNÏCODE şONG", "artists": [{"name": "åRTIST"}]},
        ],
        # The .get("") defaults for a keyless Spotify track.
        [{}, {"title": "Whatever", "artists": [{"name": "X"}]}],
        # The or-"" gate over a present-None title, and extract_artist's
        # no-artists "Unknown".
        [{"name": "X", "artist": "Y"}, {"title": None, "artists": []}],
        # The direct-artist spelling.
        [{"name": "Direct", "artist": "Solo"}, {"title": "Direct", "artist": "Solo"}],
        # extract_artist's view-count pseudo-artist exclusion.
        [
            {"name": "T", "artist": "Real"},
            {
                "title": "T",
                "artists": [{"name": "358K views"}, {"name": "Real"}],
                "views": "358K views",
            },
        ],
        # A non-string title reaches .lower() and raises on both sides.
        [{"name": "N", "artist": "A"}, {"title": 123, "artists": []}],
    ],
    "classification": [
        # (spotify_track, search_results) pairs.
        [{"name": "Song A", "artist": "Artist A"}, []],
        [
            {"name": "Song A", "artist": "Artist A"},
            [{"title": "Song A", "artists": [{"name": "Artist A"}], "videoId": "a"}],
        ],
        [
            {"name": "Song A", "artist": "Artist A"},
            [{"title": "nothing alike", "artists": [{"name": "different"}], "videoId": "b"}],
        ],
        # Best-first ordering with a mixed-quality field.
        [
            {"name": "Song A", "artist": "Artist A"},
            [
                {"title": "nothing alike", "artists": [{"name": "different"}], "videoId": "bad"},
                {"title": "Song A", "artists": [{"name": "Artist A"}], "videoId": "good"},
            ],
        ],
        # Unicode on both sides.
        [
            {"name": "Ünïcode Şong", "artist": "Årtist"},
            [
                {"title": "üNÏCODE şONG", "artists": [{"name": "åRTIST"}], "videoId": "u"},
                {"title": "other", "artists": [{"name": "other"}], "videoId": "o"},
            ],
        ],
        # A non-dict candidate reaches the scorer's .get and raises.
        [{"name": "N", "artist": "A"}, ["not-a-dict"]],
    ],
    "page": [
        # One page dict per case.
        {"items": [{"track": {"name": "A", "artists": [{"name": "X"}], "duration_ms": 1000}}]},
        {"items": []},
        {},
        {
            "items": [
                {"track": None},
                {"track": {}},
                {"track": {"name": "V", "artists": []}},
            ]
        },
        {"items": 5},
        {
            "items": [
                {
                    "track": {
                        "name": "Ünïcode Şong",
                        "artists": [{"name": "Å"}],
                        "duration_ms": None,
                    }
                }
            ]
        },
    ],
    "scraper": [
        # Playlist descriptors: a list of track descriptors each carrying
        # name / artists (a list of names) / album (name or null) /
        # duration_ms. The child and the parent build the same
        # SimpleNamespace stand-ins from them.
        {
            "tracks": [
                {
                    "name": "S",
                    "artists": ["A"],
                    "album": {"name": "Al"},
                    "duration_ms": 1000,
                }
            ]
        },
        {
            "tracks": [
                {
                    "name": "Collab",
                    "artists": ["First", "Second"],
                    "album": {"name": "Album"},
                    "duration_ms": 180_000,
                }
            ]
        },
        {"tracks": [{"name": "Solo", "artists": [], "album": None, "duration_ms": 5}]},
        {
            "tracks": [
                {"name": "One", "artists": ["A"], "album": {"name": "Al"}, "duration_ms": 1},
                {"name": "Two", "artists": ["B"], "album": None, "duration_ms": 2},
            ]
        },
        # A non-string artist name reaches the join and raises on both
        # sides.
        {"tracks": [{"name": "N", "artists": [5], "album": None, "duration_ms": 1}]},
    ],
    "display": [
        # (idx, candidate) pairs.
        [1, {"title": "Track", "artists": [{"name": "Band"}], "duration_seconds": 125}],
        [2, {"title": "T", "artists": [{"name": "A"}]}],
        [
            3,
            {
                "title": "T",
                "artists": [{"name": "A"}],
                "duration_seconds": 65,
                "resultType": "video",
            },
        ],
        [
            0,
            {
                "title": "T",
                "artists": [{"name": "A"}],
                "duration_seconds": 65,
                "resultType": "song",
            },
        ],
        # A non-int idx just interpolates; an hours-long duration renders
        # the H:MM:SS shape.
        ["x", {"title": "T", "artists": [{"name": "A"}], "duration_seconds": 3725}],
        [1, {"artists": [{"name": "A"}]}],
        [1, {"title": "Ünïcode", "artists": [{"name": "Årtist"}], "duration_seconds": 5}],
    ],
    "summarize": [
        # One list of MatchResult descriptors per case.
        [],
        [
            {
                "spotify_track": {"name": "A", "artist": "X"},
                "match_type": "exact",
                "selected": {"videoId": "v0"},
                "candidates": [],
            },
            {
                "spotify_track": {"name": "B", "artist": "Y"},
                "match_type": "exact",
                "selected": {"videoId": "v1"},
                "candidates": [],
            },
        ],
        [
            {
                "spotify_track": {"name": "A", "artist": "X"},
                "match_type": "exact",
                "selected": {"videoId": "keep"},
                "candidates": [],
            },
            {
                "spotify_track": {"name": "B", "artist": "Y"},
                "match_type": "none",
                "selected": None,
                "candidates": [],
            },
        ],
        # A confirmed-but-empty video ID drops out without counting as a
        # skip.
        [
            {
                "spotify_track": {"name": "A", "artist": "X"},
                "match_type": "exact",
                "selected": {"videoId": ""},
                "candidates": [],
            }
        ],
        # Both key conventions.
        [
            {
                "spotify_track": {"name": "A", "artist": "X"},
                "match_type": "exact",
                "selected": {"video_id": "snake"},
                "candidates": [],
            }
        ],
    ],
    "bucket": [
        # (list of MatchResult descriptors, match-type name) pairs.
        [
            [
                {
                    "spotify_track": {"name": "A", "artist": "X"},
                    "match_type": "exact",
                    "selected": {"videoId": "a"},
                    "candidates": [],
                },
                {
                    "spotify_track": {"name": "B", "artist": "Y"},
                    "match_type": "none",
                    "selected": None,
                    "candidates": [],
                },
                {
                    "spotify_track": {"name": "C", "artist": "Z"},
                    "match_type": "multiple",
                    "selected": None,
                    "candidates": [{"title": "c"}],
                },
            ],
            "exact",
        ],
        [
            [
                {
                    "spotify_track": {"name": "A", "artist": "X"},
                    "match_type": "exact",
                    "selected": {"videoId": "a"},
                    "candidates": [],
                },
                {
                    "spotify_track": {"name": "B", "artist": "Y"},
                    "match_type": "none",
                    "selected": None,
                    "candidates": [],
                },
            ],
            "none",
        ],
        [[], "multiple"],
    ],
}

_PIPELINE_CHILD_PROG = """
import json
import sys
import types
from difflib import SequenceMatcher

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import spotify_import as si


class _Fuzz:
    @staticmethod
    def ratio(a, b):
        return int(round(SequenceMatcher(None, a, b).ratio() * 100))


si.fuzz = _Fuzz


def _result_from_desc(d):
    return si.MatchResult(
        spotify_track=d["spotify_track"],
        match_type=getattr(si.MatchType, d["match_type"].upper()),
        candidates=d.get("candidates", []),
        selected=d.get("selected"),
    )


def _desc_result(r):
    return {
        "spotify_track": r.spotify_track,
        "match_type": r.match_type.value,
        "candidates": r.candidates,
        "selected": r.selected,
    }


def _playlist_from_desc(d):
    def _track_from(t):
        return types.SimpleNamespace(track=types.SimpleNamespace(
            name=t["name"],
            artists=tuple(types.SimpleNamespace(name=a) for a in t["artists"]),
            album=types.SimpleNamespace(name=t["album"]["name"]) if t["album"] else None,
            duration_ms=t["duration_ms"],
        ))

    return types.SimpleNamespace(tracks=tuple(_track_from(t) for t in d["tracks"]))


def _outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {
    "fuzzy": [_outcome(si._fuzzy_score, sp, ytm) for sp, ytm in payload["fuzzy"]],
    "classification": [],
    "page": [_outcome(si._page_items, page) for page in payload["page"]],
    "scraper": [
        _outcome(si._scraper_track_rows, _playlist_from_desc(p)) for p in payload["scraper"]
    ],
    "display": [_outcome(si._display_candidate, idx, c) for idx, c in payload["display"]],
    "summarize": [],
    "bucket": [],
}
for sp, candidates in payload["classification"]:
    r = _outcome(si._match_classification, sp, candidates)
    if "ok" in r:
        r = {"ok": _desc_result(r["ok"])}
    out["classification"].append(r)
for results in payload["summarize"]:
    r = _outcome(si._summarize_results, [_result_from_desc(d) for d in results])
    if "ok" in r:
        confirmed, video_ids, skipped = r["ok"]
        r = {
            "ok": {
                "confirmed": [_desc_result(c) for c in confirmed],
                "video_ids": video_ids,
                "skipped": skipped,
            }
        }
    out["summarize"].append(r)
for results, match_type in payload["bucket"]:
    r = _outcome(
        si._bucket_results,
        [_result_from_desc(d) for d in results],
        getattr(si.MatchType, match_type.upper()),
    )
    if "ok" in r:
        r = {"ok": [_desc_result(x) for x in r["ok"]]}
    out["bucket"].append(r)
json.dump(out, sys.stdout)
"""


def _pipeline_result_from_desc(d: dict) -> si.MatchResult:
    return si.MatchResult(
        spotify_track=d["spotify_track"],
        match_type=getattr(si.MatchType, d["match_type"].upper()),
        candidates=d.get("candidates", []),
        selected=d.get("selected"),
    )


def _pipeline_desc_result(r: si.MatchResult) -> dict:
    return {
        "spotify_track": r.spotify_track,
        "match_type": r.match_type.value,
        "candidates": r.candidates,
        "selected": r.selected,
    }


def _pipeline_playlist_from_desc(d: dict):
    def _track_from(t):
        return types.SimpleNamespace(
            track=types.SimpleNamespace(
                name=t["name"],
                artists=tuple(types.SimpleNamespace(name=a) for a in t["artists"]),
                album=types.SimpleNamespace(name=t["album"]["name"]) if t["album"] else None,
                duration_ms=t["duration_ms"],
            )
        )

    return types.SimpleNamespace(tracks=tuple(_track_from(t) for t in d["tracks"]))


def _pipeline_outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


def _native_pipeline_outcomes() -> dict:
    cases = _PIPELINE_CASES
    out = {
        "fuzzy": [_pipeline_outcome(si._fuzzy_score, sp, ytm) for sp, ytm in cases["fuzzy"]],
        "classification": [],
        "page": [_pipeline_outcome(si._page_items, page) for page in cases["page"]],
        "scraper": [
            _pipeline_outcome(si._scraper_track_rows, _pipeline_playlist_from_desc(p))
            for p in cases["scraper"]
        ],
        "display": [
            _pipeline_outcome(si._display_candidate, idx, c) for idx, c in cases["display"]
        ],
        "summarize": [],
        "bucket": [],
    }
    for sp, candidates in cases["classification"]:
        r = _pipeline_outcome(si._match_classification, sp, candidates)
        if "ok" in r:
            r = {"ok": _pipeline_desc_result(r["ok"])}
        out["classification"].append(r)
    for results in cases["summarize"]:
        r = _pipeline_outcome(
            si._summarize_results, [_pipeline_result_from_desc(d) for d in results]
        )
        if "ok" in r:
            confirmed, video_ids, skipped = r["ok"]
            r = {
                "ok": {
                    "confirmed": [_pipeline_desc_result(c) for c in confirmed],
                    "video_ids": video_ids,
                    "skipped": skipped,
                }
            }
        out["summarize"].append(r)
    for results, match_type in cases["bucket"]:
        r = _pipeline_outcome(
            si._bucket_results,
            [_pipeline_result_from_desc(d) for d in results],
            getattr(si.MatchType, match_type.upper()),
        )
        if "ok" in r:
            r = {"ok": [_pipeline_desc_result(x) for x in r["ok"]]}
        out["bucket"].append(r)
    return out


class TestMatchingPipelineNativeParity:
    """The matching-pipeline engines, on the child-interpreter pattern
    the other parity tests use: the fallback runs with the extension
    import blocked (and the same deterministic fuzz stand-in), so both
    implementations are compared directly instead of trusting whichever
    one the test process loaded — verdicts, orders, descriptors and
    error shapes included.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_pipeline_engines_match_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _PIPELINE_CHILD_PROG],
            input=json.dumps(_PIPELINE_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_pipeline_outcomes()
        assert set(native) == set(fallback) == set(_PIPELINE_CASES)
        for engine, cases in _PIPELINE_CASES.items():
            assert len(native[engine]) == len(fallback[engine]) == len(cases)
            for case, n, f in zip(cases, native[engine], fallback[engine]):
                assert n == f, (
                    f"native and fallback {engine} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_classification_sort_is_cpythons_own(self, monkeypatch):
        """The sort the classification engine runs is CPython's own
        stable reverse ordering: ties keep the search's order even when
        the scores come from a patched scorer (routed through the module
        attribute both engines read)."""
        order: list[str] = []

        def fake_score(_sp, candidate):
            order.append(candidate["videoId"])
            return 90

        monkeypatch.setattr(si, "_fuzzy_score", fake_score)
        candidates = [
            _ytm_candidate("T", "A", video_id="first"),
            _ytm_candidate("T", "A", video_id="second"),
            _ytm_candidate("T", "A", video_id="third"),
        ]
        result = si._match_classification(_sp_track(), candidates)
        assert [c["videoId"] for c in result.candidates] == ["first", "second", "third"]
        # Every candidate was scored, in the search's own order.
        assert order == ["first", "second", "third"]
