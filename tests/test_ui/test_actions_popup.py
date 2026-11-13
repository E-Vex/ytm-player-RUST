"""Tests for the actions popup's menu builder and title engine."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.popups.actions import _build_actions, _derive_title

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _action_ids(item, item_type="track", in_playlist=False):
    return [a[0] for a in _build_actions(item, item_type, in_playlist=in_playlist)]


# ── go_to_album filtering ────────────────────────────────────────────


def test_go_to_album_with_string_album_no_album_id():
    """Regression: album as string must not crash (was AttributeError)."""
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album": "Some Album Name",
    }
    assert "go_to_album" not in _action_ids(item)


def test_go_to_album_with_album_id():
    """go_to_album present when album_id exists."""
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album": "Some Album",
        "album_id": "MPREb_abc123",
    }
    assert "go_to_album" in _action_ids(item)


def test_go_to_album_with_dict_album():
    """go_to_album present when album is a dict with 'id'."""
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album": {"name": "Album", "id": "album123"},
    }
    assert "go_to_album" in _action_ids(item)


def test_go_to_album_dict_no_id():
    """go_to_album filtered when album dict has no 'id'."""
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album": {"name": "Album"},
    }
    assert "go_to_album" not in _action_ids(item)


def test_go_to_album_none():
    """go_to_album filtered when album is None."""
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album": None,
    }
    assert "go_to_album" not in _action_ids(item)


# ── go_to_artist filtering ───────────────────────────────────────────


def test_go_to_artist_no_artist_info():
    """go_to_artist filtered when no artist data."""
    item = {"title": "Test", "artists": [], "album_id": "a1"}
    assert "go_to_artist" not in _action_ids(item)


def test_go_to_artist_with_artists_list():
    """go_to_artist present when artists list is populated."""
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album_id": "a1",
    }
    assert "go_to_artist" in _action_ids(item)


def test_go_to_artist_with_artist_string():
    """go_to_artist present when 'artist' string exists (no 'artists' list)."""
    item = {"title": "Test", "artists": [], "artist": "Someone", "album_id": "a1"}
    assert "go_to_artist" in _action_ids(item)


# ── Label swaps ───────────────────────────────────────────────────────


def test_liked_track_shows_unlike():
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album_id": "a1",
        "likeStatus": "LIKE",
    }
    actions = dict(_build_actions(item, "track"))
    assert actions["toggle_like"] == "Unlike"


def test_unliked_track_shows_like():
    item = {
        "title": "Test",
        "artists": [{"name": "A", "id": "1"}],
        "album_id": "a1",
    }
    actions = dict(_build_actions(item, "track"))
    assert actions["toggle_like"] == "Like"


def test_subscribed_artist_shows_unsubscribe():
    item = {"name": "Artist", "subscribed": True}
    actions = dict(_build_actions(item, "artist"))
    assert actions["toggle_subscribe"] == "Unsubscribe"


# ── Item types ────────────────────────────────────────────────────────


def test_unknown_type_falls_back_to_track():
    item = {"title": "Test", "artists": [{"name": "A", "id": "1"}], "album_id": "a1"}
    ids = _action_ids(item, "unknown_type")
    assert "play" in ids
    assert "play_all" not in ids


# ── Remove from playlist filtering ────────────────────────────────────


def test_remove_from_playlist_shown_when_in_playlist():
    item = {"title": "Test", "artists": [{"name": "A", "id": "1"}], "album_id": "a1"}
    ids = _action_ids(item, "track", in_playlist=True)
    assert "remove_from_playlist" in ids


def test_remove_from_playlist_hidden_when_not_in_playlist():
    item = {"title": "Test", "artists": [{"name": "A", "id": "1"}], "album_id": "a1"}
    ids = _action_ids(item, "track", in_playlist=False)
    assert "remove_from_playlist" not in ids


# ── Queue-membership swap ─────────────────────────────────────────────


def test_add_to_queue_swapped_when_in_queue():
    item = {"title": "Test", "artists": [{"name": "A", "id": "1"}], "album_id": "a1"}
    actions = dict(_build_actions(item, "track", in_queue=True))
    assert "add_to_queue" not in actions
    assert actions["remove_from_queue"] == "Remove from Queue"


def test_add_to_queue_kept_when_not_in_queue():
    item = {"title": "Test", "artists": [{"name": "A", "id": "1"}], "album_id": "a1"}
    actions = dict(_build_actions(item, "track", in_queue=False))
    assert actions["add_to_queue"] == "Add to Queue"
    assert "remove_from_queue" not in actions


# ── Title derivation ─────────────────────────────────────────────────


class TestDeriveTitle:
    """The title engine (whichever backend)."""

    def test_title_field_wins(self):
        assert _derive_title({"title": "Song", "name": "N", "artist": "A"}, "track") == "Song"

    def test_name_field_when_title_missing(self):
        assert _derive_title({"name": "Playlist"}, "playlist") == "Playlist"

    def test_artist_field_when_title_and_name_missing(self):
        assert _derive_title({"artist": "Someone"}, "track") == "Someone"

    def test_falsy_fields_fall_through(self):
        assert _derive_title({"title": "", "name": None, "artist": "A"}, "track") == "A"
        assert _derive_title({"title": None, "name": "N"}, "album") == "N"

    def test_all_falsy_answers_the_capitalized_item_type(self):
        assert _derive_title({}, "track") == "Track"
        assert _derive_title({"title": ""}, "album") == "Album"
        assert _derive_title({"title": "", "name": "", "artist": ""}, "artist") == "Artist"
        assert _derive_title({}, "playlist") == "Playlist"

    def test_34_characters_fit_untouched(self):
        name = "x" * 34
        assert _derive_title({"title": name}, "track") == name

    def test_a_fitting_name_is_returned_as_the_same_object(self):
        name = "y" * 34
        assert _derive_title({"title": name}, "track") is name

    def test_35_characters_truncate_to_31_plus_ellipsis(self):
        assert _derive_title({"title": "x" * 35}, "track") == "x" * 31 + "..."
        assert len(_derive_title({"title": "x" * 35}, "track")) == 34

    def test_the_truncation_boundary_is_strict(self):
        for n in (31, 32, 33, 34):
            assert _derive_title({"title": "x" * n}, "track") == "x" * n
        for n in (35, 36, 100):
            assert _derive_title({"title": "x" * n}, "track") == "x" * 31 + "..."

    def test_length_counts_code_points_not_bytes(self):
        assert _derive_title({"title": "Ä" * 35}, "track") == "Ä" * 31 + "..."
        assert _derive_title({"title": "𝄞" * 35}, "track") == "𝄞" * 31 + "..."
        # 34 code points of a wide character still fits.
        assert _derive_title({"title": "𝄞" * 34}, "track") == "𝄞" * 34

    def test_multibyte_names_truncate_mid_string_correctly(self):
        # 45 code points: the first 31 ("日本語" * 10 + "日") plus "...".
        assert _derive_title({"title": "日本語" * 15}, "track") == "日本語" * 10 + "日..."


class TestDeriveTitleProtocolShapes:
    """The truncation runs through the reference's own protocol — a
    str subclass is subscripted and added exactly like the reference
    subscripts and adds it.
    """

    def test_a_str_subclass_truncates_like_a_plain_str(self):
        class _PlainSub(str):
            pass

        assert _derive_title({"title": _PlainSub("y" * 35)}, "track") == "y" * 31 + "..."
        assert _derive_title({"title": _PlainSub("y" * 34)}, "track") == "y" * 34

    def test_an_overriding_subclass_sees_the_reference_slice(self):
        class _SlicingSub(str):
            def __getitem__(self, key):
                assert key.start is None, f"slice start is {key.start!r}"
                assert key.stop == 31, f"slice stop is {key.stop!r}"
                assert key.step is None, f"slice step is {key.step!r}"
                return "SLICED"

        assert _derive_title({"title": _SlicingSub("z" * 40)}, "track") == "SLICED..."

    def test_a_short_sized_exotic_rides_home_untouched(self):
        # len(dict) <= 34: the reference returns the dict itself.
        assert _derive_title({"title": {"k": 1}}, "track") == {"k": 1}


class TestTotalityEngine:
    """The native engine stays total where the reference would raise.

    A ``.get``-less item raises AttributeError in the reference's
    ``item.get(...)``; the native engine reads it as field-less
    instead — the ``build_actions`` convention — so these tests only
    run with the extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_getless_item_reads_as_fieldless(self):
        assert _derive_title(42, "track") == "Track"
        assert _derive_title(object(), "album") == "Album"


# ── Native / fallback parity ──────────────────────────────────────────

_ARTISTS = [{"name": "A", "id": "1"}]

# (item, item_type, in_queue, in_playlist, source)
_ACTION_CASES = [
    ({}, "track", False, False, "default"),
    ({"artists": _ARTISTS, "album_id": "MPREb_1"}, "track", False, False, "default"),
    ({"artist": "Solo", "album": {"id": "a1", "name": "X"}}, "track", False, False, "default"),
    ({"artists": _ARTISTS, "likeStatus": "LIKE"}, "track", False, False, "default"),
    ({"artists": _ARTISTS, "liked": True}, "track", False, False, "default"),
    ({"artists": _ARTISTS, "likeStatus": "DISLIKE"}, "track", False, False, "default"),
    ({"artists": _ARTISTS, "likeStatus": None, "liked": False}, "track", False, False, "default"),
    ({}, "track", True, False, "default"),
    ({"artists": _ARTISTS}, "track", False, True, "default"),
    ({"artists": _ARTISTS}, "track", True, True, "default"),
    ({}, "album", False, False, "default"),
    ({"artists": _ARTISTS}, "album", False, False, "default"),
    ({}, "artist", False, False, "default"),
    ({"subscribed": True}, "artist", False, False, "default"),
    ({"subscribed": False}, "artist", False, False, "default"),
    ({"subscribed": 1}, "artist", False, False, "default"),
    ({"title": "P"}, "playlist", False, False, "default"),
    ({"title": "P"}, "playlist", False, False, "search"),
    ({"title": "P"}, "playlist", True, False, "default"),
    ({}, "bogus", False, False, "default"),
    ({"album": "string album", "artists": _ARTISTS}, "track", False, False, "default"),
    ({"album": {"name": "no id"}, "artists": _ARTISTS}, "track", False, False, "default"),
    ({"artists": [], "artist": ""}, "track", False, False, "default"),
    ({"artists": [], "artist": "Named"}, "track", False, False, "default"),
    ({"album_id": 0, "album": {"id": "x"}, "artists": _ARTISTS}, "track", False, False, "default"),
    ({"album_id": "MPREb_x"}, "track", False, False, "default"),
    ({"likeStatus": "LIKE", "subscribed": True}, "artist", False, False, "default"),
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.actions import _build_actions

cases = json.load(sys.stdin)
out = [
    _build_actions(item, item_type, in_queue=in_queue, in_playlist=in_playlist, source=source)
    for item, item_type, in_queue, in_playlist, source in cases
]
json.dump(out, sys.stdout)
"""


def _native_results(cases: list) -> list:
    """Run whichever builder the module loaded (native when built) over *cases*."""
    return [
        _build_actions(item, item_type, in_queue=in_queue, in_playlist=in_playlist, source=source)
        for item, item_type, in_queue, in_playlist, source in cases
    ]


def _fallback_results(cases: list) -> list:
    """Run the pure-Python reference builder over *cases* in a child
    interpreter with the native extension import blocked."""
    payload = json.dumps(cases)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native menu builder must
    produce exactly the same (action_id, label) lists as the
    pure-Python reference for every case — same catalogs, same gates,
    same label flips — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = json.loads(json.dumps(_native_results(_ACTION_CASES)))
        fallback = _fallback_results(_ACTION_CASES)
        for i, case in enumerate(_ACTION_CASES):
            assert native[i] == fallback[i], (
                f"native and fallback menus differ for case {i}: {case!r}"
            )


# (item, item_type) pairs for the title engine's parity run. Shapes where
# the pure-Python reference would raise outside the len/slice/add
# contract (.get-less items) stay out and are pinned by TestTotalityEngine
# above; the len()/[:31] + "..." error shapes stay in — both backends
# raise them with the interpreter's own message.
_TITLE_CASES = [
    ({}, "track"),
    ({}, "album"),
    ({}, "artist"),
    ({}, "playlist"),
    ({}, "bogus type"),
    ({"title": "Hello"}, "track"),
    ({"title": "", "name": "Name"}, "track"),
    ({"title": None, "artist": "Art"}, "track"),
    ({"name": "N", "artist": "A"}, "album"),
    ({"title": "T", "name": "N", "artist": "A"}, "artist"),
    ({"title": "x" * 31}, "track"),
    ({"title": "x" * 32}, "track"),
    ({"title": "x" * 34}, "track"),
    ({"title": "x" * 35}, "track"),
    ({"title": "x" * 100}, "track"),
    ({"title": "Ä" * 35}, "track"),
    ({"title": "𝄞" * 35}, "track"),
    ({"title": "日本語" * 15}, "track"),
    ({"title": 0, "name": "N"}, "track"),
    ({"title": {}, "name": "N"}, "track"),
    ({"title": {"k": 1}}, "track"),
    ({"title": [1, 2, 3]}, "track"),
    ({"title": [1] * 34}, "track"),
    ({"title": 42}, "track"),
    ({"title": [1] * 40}, "track"),
    # The capitalize fallback's own parity: the digraph titlecase, the
    # ligatures, the final-sigma rule and its uncased-first-char shape —
    # characters stable across every Unicode release CPython 3.10-3.15
    # ships (the py_title caveat).
    ({}, "ßtring"),
    ({}, "ǳabc"),
    ({}, "ﬀx"),
    ({}, "ΑΣΣ"),
    ({}, "ABCΣ"),
    ({}, "aΣb"),
    ({}, "1Σ"),
    ({}, "A\u0301Σ"),
    ({}, "İ"),
    ({}, "ABC"),
]

_TITLE_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.popups.actions import _derive_title


def outcome(item, item_type):
    try:
        return {"ok": _derive_title(item, item_type)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


payload = json.load(sys.stdin)
out = [outcome(item, item_type) for item, item_type in payload]
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_title_outcome(item, item_type):
    try:
        return {"ok": _derive_title(item, item_type)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestTitleNativeParity:
    """When the Rust extension is built, the native title engine must
    answer exactly like the pure-Python reference for every case — the
    or-chain, the capitalize fallback, the truncation and the exotic
    len/slice/add error shapes included — so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        payload = json.dumps(_TITLE_CASES, ensure_ascii=False)
        result = subprocess.run(
            [sys.executable, "-c", _TITLE_CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        assert len(fallback) == len(_TITLE_CASES)
        for (item, item_type), expected in zip(_TITLE_CASES, fallback):
            native = _native_title_outcome(item, item_type)
            assert native == expected, (
                f"native and fallback titles differ for {item!r} / {item_type!r}:\n"
                f"native:   {native!r}\n"
                f"fallback: {expected!r}"
            )
