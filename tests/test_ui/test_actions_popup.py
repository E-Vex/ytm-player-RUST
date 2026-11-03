"""Tests for the actions popup _build_actions function."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.popups.actions import _build_actions


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


# ── Native / fallback parity ──────────────────────────────────────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

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
