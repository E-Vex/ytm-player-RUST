"""Tests for the context-page view-model assembly engine.

The engine behind the album, artist and playlist detail pages — the
album and playlist header compositions (``album_header`` /
``playlist_header``: title, subtitle, unavailable count, the playlist's
owned flag and remaining-tail fetch gate) and the artist discography
(``artist_discography``: top songs, the albums+singles concatenation,
subscribers and the similar-artists bar, plus ``expandable_sections``,
the see-all section selection behind the discography expansion). The
pure-Python reference lives in ``ui/pages/context.py``; when the Rust
extension is built the same engines run natively
(``_ytm_player_native``).

The unit tests below assert the engines' contract on whichever backend is
loaded. When the extension is built, the parity tests additionally compare
the native engines against the pure-Python references (run in a child
interpreter with the extension blocked) case for case, so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import Any

import pytest

from ytm_player.ui.pages.context import (
    album_header,
    artist_discography,
    expandable_sections,
    playlist_header,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _track(vid: str, title: str = "T", artist: str = "A") -> dict[str, Any]:
    return {
        "videoId": vid,
        "title": title,
        "artists": [{"name": artist, "id": "UC1"}],
        "duration_seconds": 100,
    }


# ── Album header ──────────────────────────────────────────────────────


class TestAlbumHeader:
    def test_full_view_model(self) -> None:
        data = {
            "title": "Night Drive",
            "artists": [{"name": "Ana"}, {"name": "Bo"}],
            "year": 2023,
            "tracks": [_track("v1"), {"title": "no id"}, _track("v2")],
        }
        view = album_header(data)
        assert view["title"] == "Night Drive"
        assert view["subtitle"] == "Ana, Bo · 2023 · 2 tracks"
        assert [t["video_id"] for t in view["tracks"]] == ["v1", "v2"]
        assert view["unavailable"] == 1

    def test_singular_count(self) -> None:
        view = album_header({"title": "S", "artists": [{"name": "A"}], "tracks": [_track("v1")]})
        assert view["subtitle"] == "A · 1 track"

    def test_zero_tracks_pluralises(self) -> None:
        view = album_header({"title": "Z", "artists": [{"name": "A"}], "tracks": []})
        assert view["subtitle"] == "A · 0 tracks"

    def test_year_segment_only_when_truthy(self) -> None:
        base = {"title": "Y", "artists": [{"name": "A"}], "tracks": []}
        assert album_header({**base, "year": 2021})["subtitle"] == "A · 2021 · 0 tracks"
        assert album_header({**base, "year": "2021"})["subtitle"] == "A · 2021 · 0 tracks"
        # Falsy spellings omit the segment entirely.
        assert album_header({**base, "year": ""})["subtitle"] == "A · 0 tracks"
        assert album_header({**base, "year": 0})["subtitle"] == "A · 0 tracks"
        assert album_header({**base, "year": None})["subtitle"] == "A · 0 tracks"
        assert album_header(dict(base))["subtitle"] == "A · 0 tracks"

    def test_artist_through_extract_artist(self) -> None:
        assert album_header({"title": "A", "artist": "Solo"})["subtitle"].startswith("Solo ·")
        assert album_header({"title": "A"})["subtitle"].startswith("Unknown ·")
        # Multi-artist list joins on ", ".
        view = album_header({"title": "A", "artists": [{"name": "X"}, {"name": "Y"}], "tracks": []})
        assert view["subtitle"] == "X, Y · 0 tracks"

    def test_unknown_title_fallback(self) -> None:
        assert album_header({"tracks": []})["title"] == "Unknown Album"

    def test_present_none_title_flows_raw(self) -> None:
        # .get("title", …) returns the present None; only a MISSING key
        # falls back to "Unknown Album".
        assert album_header({"title": None, "tracks": []})["title"] is None

    def test_missing_tracks_key_reads_empty(self) -> None:
        view = album_header({"title": "M", "artists": [{"name": "A"}]})
        assert view["tracks"] == []
        assert view["unavailable"] == 0

    def test_unavailable_counts_unplayable_entries(self) -> None:
        view = album_header(
            {"title": "U", "artists": [{"name": "A"}], "tracks": [{"title": "a"}, {"title": "b"}]}
        )
        assert view["tracks"] == []
        assert view["unavailable"] == 2

    def test_tracks_are_normalized(self) -> None:
        view = album_header(
            {
                "title": "N",
                "artists": [{"name": "A"}],
                "tracks": [
                    {
                        "videoId": "v1",
                        "title": "T1",
                        "artists": [{"name": "A", "id": "UC1"}],
                        "album": {"name": "Al", "id": "MPRE"},
                        "duration_seconds": 250,
                        "thumbnails": [{"url": "s"}, {"url": "b"}],
                    }
                ],
            }
        )
        (track,) = view["tracks"]
        assert track["video_id"] == "v1"
        assert track["album"] == "Al"
        assert track["album_id"] == "MPRE"
        assert track["duration"] == 250
        assert track["thumbnail_url"] == "b"

    def test_input_not_mutated(self) -> None:
        data = {
            "title": "Night Drive",
            "artists": [{"name": "Ana"}],
            "year": 2023,
            "tracks": [_track("v1"), {"title": "no id"}],
        }
        before = copy.deepcopy(data)
        album_header(data)
        assert data == before


# ── Playlist header ───────────────────────────────────────────────────


class TestPlaylistHeader:
    def test_loading_suffix_and_fetch_gate(self) -> None:
        data = {
            "title": "Big Playlist",
            "author": {"name": "me"},
            "owned": False,
            "trackCount": 8,
            "tracks": [_track("v1"), _track("v2"), {"title": "no id"}],
        }
        view = playlist_header(data)
        assert view["title"] == "Big Playlist"
        assert view["subtitle"] == "me · 2 tracks (loading 8 total…)"
        assert view["unavailable"] == 1
        assert view["owned"] is False
        assert view["fetch_remaining"] is True
        assert view["raw_count"] == 3

    def test_no_suffix_when_track_count_covers_the_raw_list(self) -> None:
        view = playlist_header(
            {
                "title": "P",
                "author": {"name": "me"},
                "trackCount": 2,
                "tracks": [_track("v1"), _track("v2")],
            }
        )
        assert view["subtitle"] == "me · 2 tracks"
        assert view["fetch_remaining"] is False

    def test_suffix_gate_uses_the_normalized_count(self) -> None:
        # trackCount sits between the normalized (2) and raw (3) counts:
        # no suffix (2 is not above 2) but the tail fetch still runs (2
        # is not above 3 either — so neither fires here)…
        view = playlist_header(
            {
                "title": "P",
                "author": {"name": "me"},
                "trackCount": 2,
                "tracks": [_track("v1"), _track("v2"), {"title": "no id"}],
            }
        )
        assert view["subtitle"] == "me · 2 tracks"
        assert view["fetch_remaining"] is False
        # …while a count above the normalized list (but equal to the raw
        # list) suffixes the subtitle without fetching the tail — the
        # raw list already covers the total.
        view = playlist_header(
            {
                "title": "P",
                "author": {"name": "me"},
                "trackCount": 3,
                "tracks": [_track("v1"), _track("v2"), {"title": "no id"}],
            }
        )
        assert view["subtitle"] == "me · 2 tracks (loading 3 total…)"
        assert view["fetch_remaining"] is False

    def test_falsy_track_count_reads_as_absent(self) -> None:
        base = {"title": "P", "author": {"name": "me"}, "tracks": [_track("v1")]}
        assert playlist_header({**base, "trackCount": 0})["subtitle"] == "me · 1 track"
        assert playlist_header({**base, "trackCount": None})["subtitle"] == "me · 1 track"
        assert playlist_header(dict(base))["subtitle"] == "me · 1 track"

    def test_owned_truthiness(self) -> None:
        base = {"title": "P", "author": {"name": "me"}, "tracks": []}
        assert playlist_header({**base, "owned": True})["owned"] is True
        assert playlist_header({**base, "owned": 1})["owned"] is True
        assert playlist_header({**base, "owned": False})["owned"] is False
        assert playlist_header({**base, "owned": ""})["owned"] is False
        assert playlist_header(dict(base))["owned"] is False

    def test_owner_shapes(self) -> None:
        assert playlist_header({"author": {"name": "me"}, "tracks": []})["subtitle"].startswith(
            "me ·"
        )
        # A dict author without a name reads "Unknown".
        assert playlist_header({"author": {}, "tracks": []})["subtitle"].startswith("Unknown ·")
        # A present-None name stringifies, like the reference's f-string.
        assert playlist_header({"author": {"name": None}, "tracks": []})["subtitle"].startswith(
            "None ·"
        )
        # A non-dict author goes through str().
        assert playlist_header({"author": "someone", "tracks": []})["subtitle"].startswith(
            "someone ·"
        )
        assert playlist_header({"author": None, "tracks": []})["subtitle"].startswith("None ·")
        assert playlist_header({"tracks": []})["subtitle"].startswith("Unknown ·")

    def test_singular_count(self) -> None:
        view = playlist_header({"title": "P", "author": {"name": "me"}, "tracks": [_track("v1")]})
        assert view["subtitle"] == "me · 1 track"

    def test_input_not_mutated(self) -> None:
        data = {
            "title": "P",
            "author": {"name": "me"},
            "owned": True,
            "trackCount": 5,
            "tracks": [_track("v1")],
        }
        before = copy.deepcopy(data)
        playlist_header(data)
        assert data == before


# ── Artist discography ────────────────────────────────────────────────


class TestArtistDiscography:
    def test_full_assembly(self) -> None:
        data = {
            "name": "Artist X",
            "subscribers": "1.2M",
            "songs": {"browseId": "VLPLfull", "results": [_track("v1"), _track("v2")]},
            "albums": {"results": [{"title": "LP", "browseId": "MPREb_1"}]},
            "singles": {"results": [{"title": "S1", "browseId": "MPREb_2"}]},
            "related": {"results": [{"name": "R1"}, {"name": "R2"}]},
        }
        view = artist_discography(data)
        assert view["name"] == "Artist X"
        assert view["subscriber_text"] == "1.2M subscribers"
        assert [t["video_id"] for t in view["top_songs"]] == ["v1", "v2"]
        assert view["songs_browse_id"] == "VLPLfull"
        assert view["albums"] == [
            {"title": "LP", "browseId": "MPREb_1"},
            {"title": "S1", "browseId": "MPREb_2"},
        ]
        assert view["related_text"] == "Similar Artists: R1 · R2"

    def test_unknown_name_fallback(self) -> None:
        assert artist_discography({})["name"] == "Unknown Artist"

    def test_subscriber_text_shapes(self) -> None:
        assert artist_discography({"subscribers": "1.2M"})["subscriber_text"] == "1.2M subscribers"
        assert artist_discography({"subscribers": 1200})["subscriber_text"] == "1200 subscribers"
        assert artist_discography({"subscribers": ""})["subscriber_text"] == ""
        assert artist_discography({"subscribers": None})["subscriber_text"] == ""
        assert artist_discography({"subscribers": 0})["subscriber_text"] == ""
        assert artist_discography({})["subscriber_text"] == ""

    def test_albums_concatenate_albums_then_singles(self) -> None:
        album_row = {"title": "LP", "browseId": "MPREb_1"}
        single_row = {"title": "Single", "browseId": "MPREb_2"}
        view = artist_discography(
            {"albums": {"results": [album_row]}, "singles": {"results": [single_row]}}
        )
        # A fresh list of the original row objects, albums first.
        assert view["albums"] == [album_row, single_row]
        assert view["albums"][0] is album_row
        assert view["albums"][1] is single_row

    def test_missing_sections_read_empty(self) -> None:
        view = artist_discography({"name": "N"})
        assert view["top_songs"] == []
        assert view["songs_browse_id"] is None
        assert view["albums"] == []
        assert view["related_text"] == ""

    def test_non_dict_sections_read_empty(self) -> None:
        view = artist_discography(
            {"songs": "nope", "albums": "nope", "singles": 7, "related": None}
        )
        assert view["top_songs"] == []
        assert view["songs_browse_id"] is None
        assert view["albums"] == []
        assert view["related_text"] == ""

    def test_songs_browse_id_present_none(self) -> None:
        # .get("browseId") passes a present None through, like the
        # reference.
        assert (
            artist_discography({"songs": {"browseId": None, "results": []}})["songs_browse_id"]
            is None
        )
        assert artist_discography({"songs": {"results": []}})["songs_browse_id"] is None
        assert (
            artist_discography({"songs": {"browseId": "VLPL", "results": []}})["songs_browse_id"]
            == "VLPL"
        )

    def test_top_songs_are_normalized(self) -> None:
        view = artist_discography({"songs": {"results": [{"title": "no id"}, _track("v1")]}})
        assert [t["video_id"] for t in view["top_songs"]] == ["v1"]

    def test_related_caps_at_eight(self) -> None:
        artists = [{"name": f"R{i}"} for i in range(10)]
        view = artist_discography({"related": {"results": artists}})
        assert view["related_text"] == "Similar Artists: " + " · ".join(f"R{i}" for i in range(8))

    def test_related_filters_falsy_names(self) -> None:
        view = artist_discography(
            {"related": {"results": [{"name": ""}, {"name": "X"}, {"name": None}]}}
        )
        assert view["related_text"] == "Similar Artists: X"

    def test_related_non_dict_rows_stringify(self) -> None:
        view = artist_discography({"related": {"results": ["Solo", 42, {"name": "Duo"}]}})
        assert view["related_text"] == "Similar Artists: Solo · 42 · Duo"

    def test_empty_related_results_stay_empty(self) -> None:
        assert artist_discography({"related": {"results": []}})["related_text"] == ""

    def test_input_not_mutated(self) -> None:
        data = {
            "name": "N",
            "subscribers": "5",
            "songs": {"browseId": "VL", "results": [_track("v1")]},
            "albums": {"results": [{"title": "a"}]},
            "singles": {"results": []},
            "related": {"results": [{"name": "R"}]},
        }
        before = copy.deepcopy(data)
        artist_discography(data)
        assert data == before


# ── Expandable sections ───────────────────────────────────────────────


class TestExpandableSections:
    def test_selects_browseable_sections_in_order(self) -> None:
        albums = {"browseId": "MPAD_1", "params": "p1", "results": []}
        singles = {"browseId": "MPAD_2", "params": "p2", "results": []}
        result = expandable_sections({"albums": albums, "singles": singles})
        # The original section objects, albums first.
        assert result == [albums, singles]
        assert result[0] is albums
        assert result[1] is singles

    def test_skips_sections_without_params_or_browse_id(self) -> None:
        assert expandable_sections({"albums": {"browseId": "MPAD", "results": []}}) == []
        assert expandable_sections({"albums": {"params": "p", "results": []}}) == []
        # Falsy spellings count as absent.
        assert expandable_sections({"albums": {"browseId": "", "params": "p"}}) == []
        assert expandable_sections({"albums": {"browseId": "MPAD", "params": ""}}) == []
        assert expandable_sections({"albums": {"browseId": None, "params": "p"}}) == []
        assert expandable_sections({"albums": {"browseId": "MPAD", "params": None}}) == []

    def test_skips_non_dict_and_missing_sections(self) -> None:
        assert expandable_sections({"albums": "nope", "singles": 7}) == []
        assert expandable_sections({}) == []
        assert expandable_sections({"albums": None, "singles": None}) == []

    def test_singles_only(self) -> None:
        singles = {"browseId": "MPAD_2", "params": "p"}
        assert expandable_sections({"singles": singles}) == [singles]


# ── Dict protocol ─────────────────────────────────────────────────────


class TestDictProtocol:
    def test_field_reads_go_through_get(self) -> None:
        # A dict subclass overriding `.get` must steer the engine exactly
        # like it steers the reference: both go through the method. (The
        # artist read inside album_header goes through the shared
        # extract_artist engine, which reads C-level dict fields natively
        # — the same documented policy as the search engine — so the
        # steering test pins the engine's own field reads.)
        class Overriding(dict):
            def get(self, key: str, default: Any = None) -> Any:
                if key == "title":
                    return "Overridden"
                return super().get(key, default)

        assert album_header(Overriding({"tracks": []}))["title"] == "Overridden"
        assert playlist_header(Overriding({"tracks": []}))["title"] == "Overridden"
        assert artist_discography(Overriding({"songs": {}}))["name"] == "Unknown Artist"

    def test_author_reads_go_through_get(self) -> None:
        class Overriding(dict):
            def get(self, key: str, default: Any = None) -> Any:
                if key == "name":
                    return "Overridden"
                return super().get(key, default)

        view = playlist_header({"author": Overriding({}), "tracks": []})
        assert view["subtitle"].startswith("Overridden ·")


# ── Native-only totality ──────────────────────────────────────────────


class TestTotalityEngine:
    """Where the pure-Python reference would raise, the native engine
    stays total (matching the ported engines before it): a non-list
    ``tracks`` value reads as empty with an unknown count, an
    uncomparable ``trackCount`` skips the gated decisions, and non-list
    section results / related rows contribute nothing."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_list_album_tracks_read_empty(self) -> None:
        view = album_header({"title": "X", "artists": [{"name": "A"}], "tracks": "nope"})
        assert view["tracks"] == []
        assert view["unavailable"] is None

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_list_playlist_tracks_read_empty(self) -> None:
        view = playlist_header(
            {"title": "P", "author": {"name": "me"}, "trackCount": 5, "tracks": "nope"}
        )
        assert view["tracks"] == []
        assert view["unavailable"] is None
        assert view["fetch_remaining"] is False
        assert view["raw_count"] == 0

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_uncomparable_track_count_skips_the_gates(self) -> None:
        # A string trackCount would make the reference's `>` raise
        # TypeError; the engine skips the suffix and the tail fetch.
        view = playlist_header(
            {"title": "P", "author": {"name": "me"}, "trackCount": "500", "tracks": [_track("v1")]}
        )
        assert view["subtitle"] == "me · 1 track"
        assert view["fetch_remaining"] is False

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_truthy_non_list_related_contributes_nothing(self) -> None:
        # The reference's [:8] on a dict would raise TypeError.
        view = artist_discography({"related": {"results": {"a": 1}}})
        assert view["related_text"] == ""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_list_section_results_read_empty(self) -> None:
        # The reference's `albums_list + singles_list` would raise
        # TypeError on a string results value.
        view = artist_discography(
            {"albums": {"results": "nope"}, "singles": {"results": [{"title": "s"}]}}
        )
        assert view["albums"] == [{"title": "s"}]

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_list_songs_results_read_empty(self) -> None:
        view = artist_discography({"songs": {"results": "nope"}})
        assert view["top_songs"] == []
        assert view["songs_browse_id"] is None

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_str_related_name_stringifies(self) -> None:
        # The reference's str.join on an int name would raise TypeError.
        view = artist_discography({"related": {"results": [{"name": 7}, {"name": ""}]}})
        assert view["related_text"] == "Similar Artists: 7"


# ── Native vs fallback parity ─────────────────────────────────────────

# One entry per engine case; the child interpreter below JSON round-trips
# them, so every value is a JSON type. Shapes where the pure-Python
# reference would raise (non-list tracks, uncomparable trackCount,
# non-list related results, non-str related names) stay out of the parity
# cases and are pinned by TestTotalityEngine above.

_PARITY_TRACKS = [
    {
        "videoId": "v1",
        "title": "T1",
        "artists": [{"name": "A", "id": "UC1"}],
        "duration_seconds": 100,
    },
    {
        "videoId": "v2",
        "title": "T2",
        "artists": [{"name": "B"}, {"name": "C"}],
        "album": {"name": "Al", "id": "MPRE"},
        "duration": "3:21",
    },
    {"title": "no id"},
    {
        "videoId": "v3",
        "title": "Vid",
        "artists": [{"name": "A"}, {"name": "358K views"}],
        "views": "358K views",
        "isVideo": True,
    },
    {"videoId": "v4", "length": "2:05"},
    {"video_id": "v5", "title": "T5", "likeStatus": "LIKE", "setVideoId": "sv"},
    {"videoId": "v6", "thumbnails": [{"url": "s"}, {"url": "b"}]},
]

_ALBUM_CASES: list[dict[str, Any]] = [
    {
        "title": "Night Drive",
        "artists": [{"name": "Ana"}, {"name": "Bo"}],
        "year": 2023,
        "tracks": _PARITY_TRACKS,
    },
    {"title": "No Year", "artists": [{"name": "Ana"}], "tracks": _PARITY_TRACKS[:2]},
    {"title": "Y", "artists": [{"name": "Ana"}], "year": "", "tracks": []},
    {"title": "Y", "artists": [{"name": "Ana"}], "year": 0, "tracks": []},
    {"title": "Y", "artists": [{"name": "Ana"}], "year": None, "tracks": []},
    {"title": "Y", "artists": [{"name": "Ana"}], "year": "2021", "tracks": []},
    {"title": "S", "artists": [{"name": "Ana"}], "tracks": _PARITY_TRACKS[:1]},
    {"title": "M", "artists": [{"name": "Ana"}]},
    {"title": "E", "artist": "Solo", "tracks": []},
    {"artists": [{"name": "Ana"}], "tracks": []},
    {"title": None, "artists": [{"name": "Ana"}], "tracks": []},
    {"title": "U", "tracks": []},
    {"title": "B", "artist": "Solo Name", "tracks": []},
    {"title": "Z", "artists": [{"name": "Ana"}], "tracks": [{"title": "a"}, {"title": "b"}]},
    {"title": 42, "artists": [{"name": "Ana"}], "tracks": []},
]

_PLAYLIST_CASES: list[dict[str, Any]] = [
    {
        "title": "Big Playlist",
        "author": {"name": "me"},
        "owned": False,
        "trackCount": 8,
        "tracks": _PARITY_TRACKS,
    },
    {
        "title": "P",
        "author": {"name": "me"},
        "trackCount": 6,
        "tracks": _PARITY_TRACKS[:6],
    },
    {
        "title": "P",
        "author": {"name": "me"},
        "trackCount": 2,
        "tracks": _PARITY_TRACKS[:2] + [{"title": "no id"}],
    },
    {"title": "P", "author": {"name": "me"}, "trackCount": 0, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": None, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": 8.5, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": True, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": 10**30, "tracks": []},
    {"title": "P", "author": "someone", "trackCount": 5, "tracks": []},
    {"title": "P", "author": {}, "trackCount": 5, "tracks": []},
    {"title": "P", "author": {"name": None}, "trackCount": 5, "tracks": []},
    {"title": "P", "author": {"name": ""}, "trackCount": 5, "tracks": []},
    {"title": "P", "author": None, "trackCount": 5, "tracks": []},
    {"title": "P", "owned": True, "trackCount": 5, "tracks": []},
    {"title": "P", "owned": 1, "trackCount": 5, "tracks": []},
    {"title": "P", "owned": "", "trackCount": 5, "tracks": []},
    {"title": "P", "author": {"name": "me"}, "trackCount": 5, "tracks": [_track("v1")]},
]

_ARTIST_CASES: list[dict[str, Any]] = [
    {
        "name": "Artist X",
        "subscribers": "1.2M",
        "songs": {"browseId": "VLPLfull", "results": _PARITY_TRACKS},
        "albums": {"results": [{"title": "LP", "browseId": "MPREb_1"}]},
        "singles": {"results": [{"title": "S1", "browseId": "MPREb_2"}]},
        "related": {"results": [{"name": "R1"}, {"name": "R2"}]},
    },
    {"name": "N", "subscribers": 1200},
    {"name": "N", "subscribers": ""},
    {"name": "N", "subscribers": None},
    {"name": "N", "subscribers": 0},
    {"name": "N"},
    {"name": None, "songs": {}},
    {"name": "N", "songs": "nope"},
    {"name": "N", "songs": {"results": _PARITY_TRACKS}},
    {"name": "N", "songs": {"browseId": None, "results": []}},
    {"name": "N", "songs": {"browseId": "VLPL", "results": []}},
    {
        "name": "N",
        "albums": {"results": [{"title": "a1"}, {"title": "a2"}]},
        "singles": {"results": [{"title": "s1"}]},
    },
    {"name": "N", "albums": {"results": []}, "singles": {"results": []}},
    {"name": "N", "related": {"results": [{"name": f"R{i}"} for i in range(10)]}},
    {"name": "N", "related": {"results": [{"name": ""}, {"name": "X"}, {"name": None}]}},
    {"name": "N", "related": {"results": ["Solo", 42, {"name": "Duo"}, ""]}},
    {"name": "N", "related": {"results": []}},
    {"name": "N", "related": {}},
    {"name": "N", "albums": "nope", "singles": 7, "related": None},
]

_SECTIONS_CASES: list[dict[str, Any]] = [
    {
        "albums": {"browseId": "MPAD_1", "params": "p1", "results": [{"title": "a"}]},
        "singles": {"browseId": "MPAD_2", "params": "p2", "results": []},
    },
    {"albums": {"browseId": "MPAD", "results": []}},
    {"albums": {"params": "p", "results": []}},
    {"albums": {"browseId": "", "params": "p"}},
    {"albums": {"browseId": "MPAD", "params": ""}},
    {"albums": {"browseId": None, "params": "p"}},
    {"albums": {"browseId": "MPAD", "params": None}},
    {"albums": "nope", "singles": 7},
    {},
    {"albums": None, "singles": None},
    {"singles": {"browseId": "MPAD_2", "params": "p"}},
]

_PARITY_CASES: dict[str, list[Any]] = {
    "album": _ALBUM_CASES,
    "playlist": _PLAYLIST_CASES,
    "artist": _ARTIST_CASES,
    "sections": _SECTIONS_CASES,
}

_CHILD_PROG = r"""
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.context import (
    album_header,
    artist_discography,
    expandable_sections,
    playlist_header,
)

data = json.load(sys.stdin)
out = {
    "album": [album_header(c) for c in data["album_cases"]],
    "playlist": [playlist_header(c) for c in data["playlist_cases"]],
    "artist": [artist_discography(c) for c in data["artist_cases"]],
    "sections": [expandable_sections(c) for c in data["sections_cases"]],
}
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_outcomes() -> dict:
    """Run the engines in this process — native when the extension is
    built, the fallback otherwise (in which case the parity test is
    skipped)."""
    return {
        "album": [album_header(c) for c in _ALBUM_CASES],
        "playlist": [playlist_header(c) for c in _PLAYLIST_CASES],
        "artist": [artist_discography(c) for c in _ARTIST_CASES],
        "sections": [expandable_sections(c) for c in _SECTIONS_CASES],
    }


def _fallback_outcomes() -> dict:
    """Run the pure-Python reference engines in a child interpreter with
    the native extension import blocked."""
    payload = json.dumps(
        {
            "album_cases": _ALBUM_CASES,
            "playlist_cases": _PLAYLIST_CASES,
            "artist_cases": _ARTIST_CASES,
            "sections_cases": _SECTIONS_CASES,
        }
    )
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native view-model assembly
    must produce exactly the same outcomes as the pure-Python references
    — the same titles, subtitles, unavailable counts, owned flags and
    fetch gates, the same top songs, discography rows and
    similar-artist bars, and the same see-all section selections — for
    every case, so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engines_match_fallback(self) -> None:
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert native.keys() == fallback.keys()
        for key, cases in _PARITY_CASES.items():
            for case, got, want in zip(cases, native[key], fallback[key]):
                assert got == want, (
                    f"native and fallback {key} outcomes differ for case {case!r}: "
                    f"native={got!r} fallback={want!r}"
                )
