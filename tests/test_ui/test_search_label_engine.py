"""Tests for the search-result label engine.

``_format_search_result`` is the item-to-label composer behind the
Albums/Artists/Playlists panels of the search results area: the type tag
(``resultType`` falling back to ``category``) picks the branch, the title is
the first truthy of ``title``/``name``/``artist``, and every branch
truncates to 60 characters. The pure-Python reference lives in
``ui/pages/search.py``; when the Rust extension is built the same engine
runs natively (``_ytm_player_native.format_search_result``).

The unit tests below assert the engine's contract on whichever backend is
loaded. When the extension is built, the parity test additionally compares
the native engine against the pure-Python reference (run in a child
interpreter with the extension import blocked) case for case, so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from ytm_player.ui.pages.search import _format_search_result

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── Album labels ──────────────────────────────────────────────────────


class TestAlbumLabels:
    def test_artist_and_year(self) -> None:
        item = {"resultType": "album", "title": "Moon", "artist": "Kero", "year": 2023}
        assert _format_search_result(item) == "Moon - Kero (2023)"

    def test_artists_list_is_joined(self) -> None:
        item = {
            "resultType": "album",
            "title": "X",
            "artists": [{"name": "A"}, {"name": "B"}],
            "year": "1994",
        }
        assert _format_search_result(item) == "X - A, B (1994)"

    @pytest.mark.parametrize("year", [None, "", 0])
    def test_falsy_year_omits_the_suffix(self, year: Any) -> None:
        item = {"resultType": "album", "title": "Moon", "artist": "Kero", "year": year}
        assert _format_search_result(item) == "Moon - Kero"

    def test_string_zero_year_is_truthy(self) -> None:
        # "0" is a truthy string, unlike the int 0 — the suffix appears.
        item = {"resultType": "album", "title": "Moon", "artist": "Kero", "year": "0"}
        assert _format_search_result(item) == "Moon - Kero (0)"

    def test_missing_year_omits_the_suffix(self) -> None:
        item = {"resultType": "album", "title": "Moon", "artist": "Kero"}
        assert _format_search_result(item) == "Moon - Kero"

    def test_plural_type_tag(self) -> None:
        item = {"resultType": "albums", "title": "Moon", "artist": "Kero", "year": 2023}
        assert _format_search_result(item) == "Moon - Kero (2023)"

    def test_artist_falls_back_to_unknown(self) -> None:
        item = {"resultType": "album", "title": "Moon"}
        assert _format_search_result(item) == "Moon - Unknown"

    def test_view_count_pseudo_artist_is_dropped(self) -> None:
        # The shared extract_artist semantics carry over: an id-less run
        # matching the track's `views` field is not an artist.
        item = {
            "resultType": "album",
            "title": "Vid",
            "artists": [{"name": "Real", "id": "UC1"}, {"name": "358K views", "id": None}],
            "views": "358K views",
        }
        assert _format_search_result(item) == "Vid - Real"


# ── Artist labels ─────────────────────────────────────────────────────


class TestArtistLabels:
    def test_subscribers_suffix_uses_two_spaces(self) -> None:
        item = {"resultType": "artist", "title": "Rex", "subscribers": "2.5M"}
        assert _format_search_result(item) == "Rex  [2.5M]"

    def test_int_subscribers(self) -> None:
        item = {"resultType": "artist", "title": "Rex", "subscribers": 2500000}
        assert _format_search_result(item) == "Rex  [2500000]"

    @pytest.mark.parametrize("subs", [None, "", 0])
    def test_falsy_subscribers_omits_the_suffix(self, subs: Any) -> None:
        item = {"resultType": "artist", "title": "Rex", "subscribers": subs}
        assert _format_search_result(item) == "Rex"

    def test_plural_type_tag(self) -> None:
        item = {"resultType": "artists", "title": "Rex", "subscribers": "2.5M"}
        assert _format_search_result(item) == "Rex  [2.5M]"


# ── Playlist labels ───────────────────────────────────────────────────


class TestPlaylistLabels:
    def test_author_and_item_count(self) -> None:
        item = {"resultType": "playlist", "title": "Mix", "author": "Alice", "itemCount": 12}
        assert _format_search_result(item) == "Mix by Alice (12 tracks)"

    def test_count_falls_back_to_the_count_key(self) -> None:
        item = {"category": "playlist", "title": "Mix", "count": 3}
        assert _format_search_result(item) == "Mix (3 tracks)"

    def test_present_but_falsy_item_count_beats_the_default(self) -> None:
        # `item.get("itemCount", item.get("count", ""))`: the key is
        # present, so the (falsy) 0 wins over the truthy `count` default —
        # the eager-default evaluation the reference performs.
        item = {"resultType": "playlist", "title": "Mix", "author": "A", "itemCount": 0, "count": 5}
        assert _format_search_result(item) == "Mix by A"

    @pytest.mark.parametrize("count", [None, "", 0])
    def test_falsy_count_omits_the_suffix(self, count: Any) -> None:
        item = {"resultType": "playlist", "title": "Mix", "author": "Alice", "itemCount": count}
        assert _format_search_result(item) == "Mix by Alice"

    def test_author_and_count_together(self) -> None:
        item = {"resultType": "playlists", "title": "Mix", "author": "Al", "count": "7"}
        assert _format_search_result(item) == "Mix by Al (7 tracks)"


# ── Type tag and title resolution ─────────────────────────────────────


class TestTypeTagAndTitle:
    def test_category_fallback(self) -> None:
        item = {"category": "artist", "title": "Rex"}
        assert _format_search_result(item) == "Rex"

    def test_result_type_wins_over_category(self) -> None:
        item = {"resultType": "album", "category": "artist", "title": "T", "artist": "A"}
        assert _format_search_result(item) == "T - A"

    def test_empty_result_type_is_the_bare_title(self) -> None:
        item = {"resultType": "", "title": "T"}
        assert _format_search_result(item) == "T"

    def test_unknown_result_type_is_the_bare_title(self) -> None:
        # Song/video results have no branch of their own.
        item = {"resultType": "song", "title": "T", "artist": "Ignored"}
        assert _format_search_result(item) == "T"

    def test_non_string_result_type_is_the_bare_title(self) -> None:
        item = {"resultType": 42, "title": "T"}
        assert _format_search_result(item) == "T"

    def test_title_chain_title_name_artist_unknown(self) -> None:
        assert _format_search_result({"title": "T"}) == "T"
        assert _format_search_result({"name": "N"}) == "N"
        assert _format_search_result({"artist": "A"}) == "A"
        assert _format_search_result({"title": "", "name": "N"}) == "N"
        assert _format_search_result({"title": None, "name": "", "artist": "A"}) == "A"
        assert _format_search_result({}) == "Unknown"

    def test_falsy_title_values_fall_through(self) -> None:
        # Empty/None/0 titles are skipped by the or-chain, exactly like
        # Python truthiness.
        item = {"resultType": "song", "title": 0, "name": "N"}
        assert _format_search_result(item) == "N"

    def test_long_titles_truncate_to_60(self) -> None:
        item = {"resultType": "song", "title": "T" * 70}
        assert _format_search_result(item) == "T" * 59 + "…"

    def test_truncation_counts_code_points(self) -> None:
        # A 61-code-point title truncates to 59 code points + the
        # ellipsis, never slicing a multi-byte character in half.
        item = {"resultType": "song", "title": "é" * 61}
        assert _format_search_result(item) == "é" * 59 + "…"

    def test_label_truncates_after_the_suffix(self) -> None:
        item = {"resultType": "album", "title": "T" * 70, "artist": "A", "year": 2023}
        label = _format_search_result(item)
        assert len(label) == 60
        assert label.endswith("…")


# ── Dict protocol ─────────────────────────────────────────────────────


class TestDictProtocol:
    def test_field_reads_go_through_get(self) -> None:
        # A dict subclass overriding `.get` must steer the engine exactly
        # like it steers the reference: both go through the method.
        class Overriding(dict):
            def get(self, key, default=None):
                if key == "year":
                    return 1999
                return super().get(key, default)

        item = Overriding(resultType="album", title="T", artist="A")
        assert _format_search_result(item) == "T - A (1999)"

    def test_input_is_not_mutated(self) -> None:
        item = {
            "resultType": "album",
            "title": "Moon",
            "artists": [{"name": "A"}],
            "year": 2023,
        }
        snapshot = json.dumps(item, sort_keys=True)
        _format_search_result(item)
        assert json.dumps(item, sort_keys=True) == snapshot


# ── Total engine (native only) ────────────────────────────────────────


class TestTotalityEngine:
    """The native engine stays total where the reference would raise.

    A `.get`-less item raises `AttributeError` in the reference; a truthy
    non-string bare title raises `TypeError` inside `truncate`. The native
    engine degrades instead — the convention the ported engines before it
    follow — so these tests only run with the extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_dict_item_degrades_to_unknown(self) -> None:
        assert _format_search_result(42) == "Unknown"
        assert _format_search_result(None) == "Unknown"
        assert _format_search_result(["not", "a", "dict"]) == "Unknown"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_string_bare_title_is_stringified(self) -> None:
        # The branch's f-strings already stringify titles; the bare-title
        # branch now does too, instead of raising on len().
        item = {"resultType": "song", "title": 42}
        assert _format_search_result(item) == "42"


# ── Native vs fallback parity ─────────────────────────────────────────

# One entry per engine case; the child interpreter below JSON round-trips
# them, so every value is a JSON type.
_LABEL_CASES: list[dict[str, Any]] = [
    # Album branch
    {"resultType": "album", "title": "Moon", "artist": "Kero", "year": 2023},
    {
        "resultType": "album",
        "title": "X",
        "artists": [{"name": "A"}, {"name": "B"}],
        "year": "1994",
    },
    {"resultType": "album", "title": "NoYear", "artist": "A"},
    {"resultType": "album", "title": "FalsyYear", "artist": "A", "year": 0},
    {"resultType": "album", "title": "NoneYear", "artist": "A", "year": None},
    {"resultType": "albums", "title": "Plural", "artist": "A", "year": 2000},
    {"resultType": "album", "title": "NoArtist"},
    {"resultType": "album", "name": "NamedAlbum", "artist": "A"},
    {
        "resultType": "album",
        "title": "Vid",
        "artists": [{"name": "Real", "id": "UC1"}, {"name": "358K views", "id": None}],
        "views": "358K views",
    },
    {"resultType": "album", "artists": ["plain", 7]},
    {"resultType": "album", "title": "T" * 70, "artist": "A"},
    # Artist branch
    {"resultType": "artist", "title": "Rex", "subscribers": "2.5M"},
    {"resultType": "artist", "title": "Int", "subscribers": 2500000},
    {"resultType": "artist", "title": "NoSubs"},
    {"resultType": "artist", "title": "FalsySubs", "subscribers": 0},
    {"resultType": "artists", "title": "Plural", "subscribers": "1"},
    # Playlist branch
    {"resultType": "playlist", "title": "Mix", "author": "Alice", "itemCount": 12},
    {"resultType": "playlist", "title": "Mix", "count": 3},
    {"resultType": "playlist", "title": "Mix", "author": "A", "itemCount": 0, "count": 5},
    {"resultType": "playlist", "title": "Mix", "itemCount": None},
    {"resultType": "playlists", "title": "Plural", "author": "Al", "count": "7"},
    {"resultType": "playlist", "title": "NoAuthor", "itemCount": 9},
    # Type tag / title resolution
    {"category": "artist", "title": "Cat"},
    {"resultType": "album", "category": "artist", "title": "T", "artist": "A"},
    {"resultType": "", "title": "Empty"},
    {"resultType": "song", "title": "Bare"},
    {"resultType": "video", "title": "Bare", "artist": "Ignored"},
    {"resultType": 42, "title": "IntType"},
    {"title": "OnlyTitle"},
    {"name": "OnlyName"},
    {"artist": "OnlyArtist"},
    {"title": "", "name": "NameAfterEmpty"},
    {"title": None, "name": "", "artist": "ArtistLast"},
    {},
    {"resultType": "song", "title": "T" * 70},
    {"resultType": "song", "title": "é" * 61},
    {
        "resultType": "song",
        "title": "日本語のタイトルがとても長い場合にはここで切り詰められるはずだ",
    },
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.search import _format_search_result

cases = json.load(sys.stdin)
json.dump([_format_search_result(c) for c in cases], sys.stdout, ensure_ascii=False)
"""


def _fallback_labels() -> list[str]:
    """Run the pure-Python reference engine in a child interpreter with
    the native extension import blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(_LABEL_CASES, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native label engine must
    produce exactly the same labels as the pure-Python reference for every
    case — every branch, the title chain, every falsy-value rule, the
    eager-default evaluation, and the truncation boundaries — so the two
    implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_labels_match_fallback(self) -> None:
        native = [_format_search_result(case) for case in _LABEL_CASES]
        fallback = _fallback_labels()
        assert native == fallback, (
            "native and fallback search-label outcomes differ:\n"
            f"native:   {native!r}\n"
            f"fallback: {fallback!r}"
        )
