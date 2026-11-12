"""Tests for the track-action engines.

The pure decision cores behind ``app/_track_actions.py`` — the largest
module in the app package, now with its decision surface running
natively when the Rust extension is built. Thirteen engines: the ID
extractions (``_first_artist_id``, ``_album_browse_id``), the
occurrence gate (``_queue_entry_id`` — only the Queue page's table
keys its rows by entry id), the ActionsPopup answer classification
(``_track_action_route`` — which handler each picked action routes
to), the entity-action dispatch (``_entity_action_route`` — which
method each (action, item type) pair answers, the entity id/name
extraction, the missing-id early-outs, and the artist/album
navigation targets), the like-toggle rating (``_like_toggle_rating``),
the Remove-from-Queue plan (``_remove_from_queue_plan`` — the
occurrence arm, the stale warning, the first-copy-by-video-id walk),
the in-queue verdict (``_in_queue_verdict`` — the popup's
Add/Remove-from-Queue swap), the artist-songs merge plan
(``_artist_songs_merge_plan`` — the background fetch's
enrich-and-append pass), and the artist/album popup-view assembly
(``_artist_popup_plan``'s gate, ``_artist_picker_rows``'s picker,
``_artist_popup_item``'s and ``_album_popup_item``'s item dicts). The
engines run with two interchangeable backends: the pure-Python
transcriptions in the module and, when the Rust extension is built,
the native implementations bound at the module's tail.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case — error shapes included — so the two
implementations cannot drift apart. The wiring tests pin the mixin's
routing: the popup consults the in-queue verdict with the queue's own
tracks and steers its action list, the result callback consults the
route engine with the popup's answer and the like-rating engine with
the very track, the removal consults the plan engine with the host's
own values and steers the removals and toasts, the entity dispatch
consults the route engine and awaits the engine's own ids, and the
artist-songs background fetch consults the merge plan and applies its
very objects. The existing suites in test_artist_actions.py,
test_queue_refresh_and_identity.py, test_enqueue_actions.py and
test_e_playback_ownership.py keep exercising the flows end to end on
both backends.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from ytm_player.app import _track_actions as track_actions
from ytm_player.app._track_actions import (
    TrackActionsMixin,
    _album_browse_id,
    _album_popup_item,
    _artist_picker_rows,
    _artist_popup_item,
    _artist_popup_plan,
    _artist_songs_merge_plan,
    _entity_action_route,
    _first_artist_id,
    _in_queue_verdict,
    _like_toggle_rating,
    _queue_entry_id,
    _remove_from_queue_plan,
    _track_action_route,
)
from ytm_player.utils.formatting import get_video_id

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _track(video_id: str = "v", **extra: object) -> dict:
    track = {"video_id": video_id, "title": "T"}
    track.update(extra)
    return track


# ── The first-artist ID extraction ───────────────────────────────────────


class TestFirstArtistId:
    """``_first_artist_id``'s gate and answer, on whichever backend."""

    def test_the_first_artists_browse_id_wins(self):
        assert _first_artist_id({"artists": [{"id": "UC1", "name": "A"}]}) == "UC1"

    def test_a_falsy_id_falls_through_to_the_browse_id(self):
        assert _first_artist_id({"artists": [{"id": "", "browseId": "BR1"}]}) == "BR1"

    def test_a_missing_browse_id_answers_the_default(self):
        assert _first_artist_id({"artists": [{"name": "A"}]}) == ""

    def test_only_the_first_artist_is_consulted(self):
        assert _first_artist_id({"artists": [{"name": "A"}, {"id": "UC2"}]}) == ""

    def test_a_falsy_id_zero_falls_through(self):
        assert _first_artist_id({"artists": [{"id": 0, "browseId": "BR2"}]}) == "BR2"

    def test_a_truthy_non_string_id_passes_through(self):
        assert _first_artist_id({"artists": [{"id": 7}]}) == 7

    def test_a_non_list_artists_value_answers_empty(self):
        assert _first_artist_id({"artists": "UC1"}) == ""

    def test_a_none_artists_value_answers_empty(self):
        assert _first_artist_id({"artists": None}) == ""

    def test_an_empty_artists_list_answers_empty(self):
        assert _first_artist_id({"artists": []}) == ""

    def test_a_missing_artists_key_answers_empty(self):
        assert _first_artist_id({}) == ""

    def test_a_non_dict_first_artist_answers_empty(self):
        assert _first_artist_id({"artists": ["UC1"]}) == ""


# ── The album browse-ID extraction ───────────────────────────────────────


class TestAlbumBrowseId:
    """``_album_browse_id``'s priority chain, on whichever backend."""

    def test_the_album_id_outranks_the_nested_and_top_level_ids(self):
        source = {"album_id": "A1", "album": {"id": "A2"}, "browseId": "BR"}
        assert _album_browse_id(source) == "A1"

    def test_the_nested_id_outranks_the_top_level_browse_id(self):
        assert _album_browse_id({"album": {"id": "A2"}, "browseId": "BR"}) == "A2"

    def test_the_browse_id_answers_when_nothing_else_does(self):
        assert _album_browse_id({"browseId": "BR"}) == "BR"

    def test_a_non_dict_album_is_never_nested_into(self):
        assert _album_browse_id({"album": "Greatest Hits"}) == ""

    def test_every_falsy_answer_is_the_empty_string(self):
        assert _album_browse_id({"album_id": "", "album": {"id": ""}, "browseId": 0}) == ""

    def test_a_missing_album_key_answers_through_the_rest_of_the_chain(self):
        assert _album_browse_id({"album_id": None, "browseId": "BR"}) == "BR"

    def test_an_empty_source_answers_empty(self):
        assert _album_browse_id({}) == ""


# ── The occurrence gate ──────────────────────────────────────────────────


class TestQueueEntryId:
    """``_queue_entry_id``'s gate, on whichever backend."""

    def test_a_queue_page_row_with_an_int_key_names_the_entry(self):
        assert _queue_entry_id(True, 5) == 5

    def test_the_key_object_itself_rides_home(self):
        class MyInt(int):
            pass

        key = MyInt(5)
        assert _queue_entry_id(True, key) is key

    def test_a_non_queue_table_never_names_an_occurrence(self):
        assert _queue_entry_id(False, 5) is None

    def test_a_non_int_key_never_names_an_occurrence(self):
        assert _queue_entry_id(True, "5") is None

    def test_no_key_never_names_an_occurrence(self):
        assert _queue_entry_id(True, None) is None

    def test_a_bool_key_counts_as_an_int(self):
        # CPython's own subclass rule — isinstance(True, int) holds.
        assert _queue_entry_id(True, True) is True

    def test_a_truthy_non_bool_from_queue_still_gates(self):
        assert _queue_entry_id(1, 9) == 9
        assert _queue_entry_id(0, 9) is None

    def test_a_float_key_is_not_an_int(self):
        assert _queue_entry_id(True, 5.0) is None

    def test_the_staticmethod_surface_delegates_to_the_engine(self):
        assert TrackActionsMixin._queue_entry_id(True, 7) == 7
        assert TrackActionsMixin._queue_entry_id(False, 7) is None


# ── The ActionsPopup answer classification ───────────────────────────────


class TestTrackActionRoute:
    """``_track_action_route``'s twelve arms, on whichever backend."""

    @pytest.mark.parametrize(
        ("action_id", "handler"),
        [
            ("add_to_playlist", "add_to_playlist"),
            ("play", "play"),
            ("download", "download"),
            ("play_next", "play_next"),
            ("add_to_queue", "add_to_queue"),
            ("remove_from_queue", "remove_from_queue"),
            ("start_radio", "start_radio"),
            ("go_to_artist", "go_to_artist"),
            ("go_to_album", "go_to_album"),
            ("toggle_like", "toggle_like"),
            ("copy_link", "copy_link"),
            ("remove_from_playlist", "remove_from_playlist"),
        ],
    )
    def test_each_known_action_routes_to_its_own_handler(self, action_id, handler):
        assert _track_action_route(action_id) == {"handler": handler}

    def test_a_dismissed_popup_routes_to_nothing(self):
        assert _track_action_route(None) == {"handler": "none"}

    def test_an_unknown_action_is_unhandled(self):
        assert _track_action_route("nonsense") == {"handler": "unhandled"}

    def test_the_empty_string_is_unhandled(self):
        assert _track_action_route("") == {"handler": "unhandled"}


# ── The entity-action dispatch ───────────────────────────────────────────


class TestEntityActionRoute:
    """``_entity_action_route``'s arms, on whichever backend."""

    def test_play_all_on_an_album_routes_to_play_album(self):
        route = _entity_action_route("play_all", {"browseId": "ALB", "title": "A"}, "album")
        assert route == {
            "handled": True,
            "handler": "_play_album",
            "shuffle": False,
            "entity_id": "ALB",
            "entity_name": "A",
        }

    def test_shuffle_play_carries_the_shuffle_flag(self):
        route = _entity_action_route("shuffle_play", {"browseId": "ALB"}, "album")
        assert route["handler"] == "_play_album"
        assert route["shuffle"] is True

    def test_play_all_on_a_playlist_routes_to_play_playlist(self):
        route = _entity_action_route("play_all", {"playlistId": "PL"}, "playlist")
        assert route == {
            "handled": True,
            "handler": "_play_playlist",
            "shuffle": False,
            "entity_id": "PL",
            "entity_name": "Unknown",
        }

    def test_the_entity_id_reads_all_three_keys_in_priority_order(self):
        assert _entity_action_route("play_all", {"album_id": "A3"}, "album")["entity_id"] == "A3"
        assert (
            _entity_action_route("play_all", {"playlistId": "P1"}, "playlist")["entity_id"] == "P1"
        )
        assert _entity_action_route("play_all", {"browseId": "B1"}, "album")["entity_id"] == "B1"

    def test_the_entity_name_reads_its_own_chain(self):
        item = {"album_id": "A", "title": "T", "name": "N", "artist": "Ar"}
        assert _entity_action_route("play_all", item, "album")["entity_name"] == "T"
        del item["title"]
        assert _entity_action_route("play_all", item, "album")["entity_name"] == "N"
        del item["name"]
        assert _entity_action_route("play_all", item, "album")["entity_name"] == "Ar"
        del item["artist"]
        assert _entity_action_route("play_all", item, "album")["entity_name"] == "Unknown"

    def test_play_all_without_an_id_notifies(self):
        assert _entity_action_route("play_all", {}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_shuffle_play_on_a_playlist_without_an_id_notifies(self):
        assert _entity_action_route("shuffle_play", {"browseId": ""}, "playlist") == {
            "handled": True,
            "handler": "notify",
        }

    def test_play_all_on_an_artist_is_unhandled(self):
        assert _entity_action_route("play_all", {"browseId": "X"}, "artist") == {
            "handled": False,
            "handler": "unhandled",
        }

    def test_add_to_queue_routes_albums_and_playlists(self):
        assert _entity_action_route("add_to_queue", {"browseId": "ALB", "title": "A"}, "album") == {
            "handled": True,
            "handler": "_add_album_to_queue",
            "entity_id": "ALB",
            "entity_name": "A",
        }
        assert _entity_action_route(
            "add_to_queue", {"playlistId": "PL", "artist": "Ar"}, "playlist"
        ) == {
            "handled": True,
            "handler": "_add_playlist_to_queue",
            "entity_id": "PL",
            "entity_name": "Ar",
        }

    def test_add_to_queue_without_an_id_notifies(self):
        assert _entity_action_route("add_to_queue", {}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_add_to_queue_on_an_artist_is_unhandled(self):
        assert _entity_action_route("add_to_queue", {"browseId": "X"}, "artist") == {
            "handled": False,
            "handler": "unhandled",
        }

    def test_play_next_routes_albums_and_playlists(self):
        assert _entity_action_route("play_next", {"browseId": "ALB", "title": "A"}, "album") == {
            "handled": True,
            "handler": "_add_album_next",
            "entity_id": "ALB",
            "entity_name": "A",
        }
        assert _entity_action_route("play_next", {"playlistId": "PL"}, "playlist") == {
            "handled": True,
            "handler": "_add_playlist_next",
            "entity_id": "PL",
            "entity_name": "Unknown",
        }

    def test_play_next_without_an_id_notifies(self):
        assert _entity_action_route("play_next", {"browseId": ""}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_play_next_on_a_track_is_unhandled(self):
        assert _entity_action_route("play_next", {"browseId": "X"}, "track") == {
            "handled": False,
            "handler": "unhandled",
        }

    def test_start_radio_on_an_artist_routes_with_the_entity_id(self):
        assert _entity_action_route("start_radio", {"browseId": "UC1"}, "artist") == {
            "handled": True,
            "handler": "_start_artist_radio",
            "entity_id": "UC1",
        }

    def test_start_radio_on_an_artist_without_an_id_notifies(self):
        assert _entity_action_route("start_radio", {}, "artist") == {
            "handled": True,
            "handler": "notify",
        }

    def test_start_radio_on_a_playlist_has_no_id_gate(self):
        # The reference passes the item itself — no id needed.
        assert _entity_action_route("start_radio", {"browseId": "PL"}, "playlist") == {
            "handled": True,
            "handler": "_start_playlist_radio",
        }

    def test_start_radio_on_an_album_is_unhandled(self):
        assert _entity_action_route("start_radio", {"browseId": "X"}, "album") == {
            "handled": False,
            "handler": "unhandled",
        }

    def test_play_top_songs_routes_artists(self):
        assert _entity_action_route("play_top_songs", {"browseId": "UC1"}, "artist") == {
            "handled": True,
            "handler": "_play_artist_top_songs",
            "entity_id": "UC1",
        }

    def test_play_top_songs_without_an_id_notifies(self):
        assert _entity_action_route("play_top_songs", {}, "artist") == {
            "handled": True,
            "handler": "notify",
        }

    def test_play_top_songs_on_a_playlist_is_unhandled(self):
        assert _entity_action_route("play_top_songs", {"browseId": "X"}, "playlist") == {
            "handled": False,
            "handler": "unhandled",
        }

    def test_go_to_artist_with_listed_artists_uses_the_first_artist(self):
        route = _entity_action_route(
            "go_to_artist",
            {"artists": [{"id": "UC1"}, {"id": "UC2"}], "browseId": "B"},
            "playlist",
        )
        assert route == {"handled": True, "handler": "_navigate_artist", "artist_id": "UC1"}

    def test_view_similar_shares_the_go_to_artist_walk(self):
        route = _entity_action_route("view_similar", {"artists": [{"browseId": "BR"}]}, "album")
        assert route == {"handled": True, "handler": "_navigate_artist", "artist_id": "BR"}

    def test_go_to_artist_with_unusable_artists_falls_back_to_the_entity_id(self):
        # The artists list is empty — an artist item answers through its
        # own entity id.
        route = _entity_action_route("go_to_artist", {"browseId": "UC9"}, "artist")
        assert route == {"handled": True, "handler": "_navigate_artist", "artist_id": "UC9"}

    def test_go_to_artist_on_a_non_artist_item_without_artists_notifies(self):
        assert _entity_action_route("go_to_artist", {"browseId": "UC9"}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_go_to_artist_with_artists_missing_their_ids_notifies(self):
        assert _entity_action_route("view_similar", {"artists": [{"name": "A"}]}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_go_to_artist_with_a_non_list_artists_value_uses_the_entity_id(self):
        route = _entity_action_route(
            "go_to_artist", {"artists": "nope", "browseId": "UC7"}, "artist"
        )
        assert route == {"handled": True, "handler": "_navigate_artist", "artist_id": "UC7"}

    def test_go_to_album_uses_the_album_browse_id(self):
        route = _entity_action_route("go_to_album", {"album_id": "AL1"}, "track")
        assert route == {"handled": True, "handler": "_navigate_album", "album_id": "AL1"}

    def test_go_to_album_without_an_album_id_notifies(self):
        assert _entity_action_route("go_to_album", {}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_add_to_library_routes_albums(self):
        assert _entity_action_route(
            "add_to_library", {"browseId": "ALB", "title": "A"}, "album"
        ) == {
            "handled": True,
            "handler": "_add_album_to_library",
            "entity_id": "ALB",
            "entity_name": "A",
        }

    def test_add_to_library_without_an_id_notifies(self):
        assert _entity_action_route("add_to_library", {"browseId": ""}, "album") == {
            "handled": True,
            "handler": "notify",
        }

    def test_add_to_library_on_an_artist_is_unhandled(self):
        assert _entity_action_route("add_to_library", {"browseId": "X"}, "artist") == {
            "handled": False,
            "handler": "unhandled",
        }

    @pytest.mark.parametrize("action_id", ["copy_link", "toggle_subscribe", "edit", "nonsense"])
    def test_actions_no_arm_claims_are_unhandled(self, action_id):
        assert _entity_action_route(action_id, {"browseId": "X"}, "album") == {
            "handled": False,
            "handler": "unhandled",
        }


# ── The like-toggle rating ───────────────────────────────────────────────


class TestLikeToggleRating:
    """``_like_toggle_rating``'s verdict, on whichever backend."""

    def test_a_liked_track_toggles_to_indifferent(self):
        assert _like_toggle_rating({"likeStatus": "LIKE"}) == {
            "is_liked": True,
            "rating": "INDIFFERENT",
            "label": "Unliked",
        }

    def test_an_unrated_track_toggles_to_like(self):
        assert _like_toggle_rating({}) == {
            "is_liked": False,
            "rating": "LIKE",
            "label": "Liked",
        }

    def test_the_liked_flag_alone_carries_the_verdict(self):
        assert _like_toggle_rating({"likeStatus": "INDIFFERENT", "liked": True}) == {
            "is_liked": True,
            "rating": "INDIFFERENT",
            "label": "Unliked",
        }

    def test_the_like_status_outranks_a_falsy_liked_flag(self):
        assert (
            _like_toggle_rating({"likeStatus": "LIKE", "liked": False})["rating"] == "INDIFFERENT"
        )

    def test_a_truthy_non_bool_liked_flag_carries_the_verdict(self):
        verdict = _like_toggle_rating({"liked": 1})
        assert verdict["is_liked"] == 1  # the raw or-result, CPython's own
        assert verdict["rating"] == "INDIFFERENT"
        assert verdict["label"] == "Unliked"

    def test_a_falsy_liked_flag_answers_like(self):
        verdict = _like_toggle_rating({"liked": 0})
        assert verdict["is_liked"] == 0
        assert verdict["rating"] == "LIKE"

    def test_a_disliked_track_toggles_to_like(self):
        assert _like_toggle_rating({"likeStatus": "DISLIKE"}) == {
            "is_liked": False,
            "rating": "LIKE",
            "label": "Liked",
        }


# ── The Remove-from-Queue plan ───────────────────────────────────────────


class TestRemoveFromQueuePlan:
    """``_remove_from_queue_plan``'s arms, on whichever backend."""

    def test_an_entry_id_present_names_the_occurrence(self):
        assert _remove_from_queue_plan(False, 3, [], "v") == {"arm": "entry"}

    def test_from_queue_with_an_entry_id_is_the_entry_arm_too(self):
        assert _remove_from_queue_plan(True, 3, [], "v") == {"arm": "entry"}

    def test_a_falsy_entry_id_still_names_the_occurrence(self):
        # 0 is not None — the occurrence is named even with a falsy id.
        assert _remove_from_queue_plan(True, 0, [], "") == {"arm": "entry"}

    def test_from_queue_without_an_id_is_stale(self):
        assert _remove_from_queue_plan(True, None, [], "v") == {"arm": "stale"}

    def test_without_an_occurrence_and_a_video_id_nothing_happens(self):
        assert _remove_from_queue_plan(False, None, [{"video_id": "v"}], "") == {"arm": "none"}

    def test_the_first_copy_by_video_id_goes(self):
        tracks = [{"video_id": "x"}, {"video_id": "v"}, {"video_id": "v"}]
        assert _remove_from_queue_plan(False, None, tracks, "v") == {"arm": "remove_at", "index": 1}

    def test_a_match_at_the_front_is_index_zero(self):
        assert _remove_from_queue_plan(False, None, [{"video_id": "v"}], "v") == {
            "arm": "remove_at",
            "index": 0,
        }

    def test_a_row_without_a_video_id_never_matches(self):
        assert _remove_from_queue_plan(False, None, [{"video_id": None}], "v") == {"arm": "none"}

    def test_no_match_answers_none(self):
        assert _remove_from_queue_plan(False, None, [{"video_id": "x"}], "v") == {"arm": "none"}

    def test_an_empty_queue_answers_none(self):
        assert _remove_from_queue_plan(False, None, [], "v") == {"arm": "none"}

    def test_cpythons_own_equality_decides_the_match(self):
        # 1 == True under CPython's own int/bool unification.
        assert _remove_from_queue_plan(False, None, [{"video_id": 1}], True) == {
            "arm": "remove_at",
            "index": 0,
        }

    def test_a_raising_row_read_propagates(self):
        class Boom:
            def get(self, *_args: object) -> object:
                raise RuntimeError("row read failed")

        with pytest.raises(RuntimeError, match="row read failed"):
            _remove_from_queue_plan(False, None, [Boom()], "v")


# ── The in-queue verdict ─────────────────────────────────────────────────


class TestInQueueVerdict:
    """``_in_queue_verdict``'s walk, on whichever backend."""

    def test_a_queued_track_is_in_the_queue(self):
        assert _in_queue_verdict("v", [{"video_id": "v"}]) is True

    def test_the_walk_finds_any_matching_row(self):
        assert _in_queue_verdict("v", [{"video_id": "x"}, {"video_id": "v"}]) is True

    def test_a_track_not_queued_is_not_in_the_queue(self):
        assert _in_queue_verdict("v", [{"video_id": "x"}]) is False

    def test_an_empty_queue_answers_false(self):
        assert _in_queue_verdict("v", []) is False

    def test_a_falsy_video_id_answers_false_without_touching_the_queue(self):
        class Boom:
            def __iter__(self) -> object:
                raise RuntimeError("queue touched")

        assert _in_queue_verdict("", [Boom()]) is False
        assert _in_queue_verdict(None, [Boom()]) is False

    def test_a_row_without_a_video_id_never_matches(self):
        assert _in_queue_verdict("v", [{"title": "no vid"}]) is False

    def test_cpythons_own_equality_decides_the_verdict(self):
        assert _in_queue_verdict(True, [{"video_id": 1}]) is True

    def test_the_comparison_is_case_sensitive(self):
        assert _in_queue_verdict("v", [{"video_id": "V"}]) is False


# ── The artist-songs merge plan ──────────────────────────────────────────


class TestArtistSongsMergePlan:
    """``_artist_songs_merge_plan``'s enrich-and-append pass, on whichever
    backend."""

    def test_the_plan_enriches_and_appends(self):
        all_tracks = [
            {"video_id": "s1", "duration": 210},
            {"video_id": "s2"},
            {"video_id": "s3"},
        ]
        initial = [{"video_id": "s1"}]
        queue_track = {"video_id": "s1"}

        plan = _artist_songs_merge_plan(all_tracks, initial, [queue_track])

        assert plan["updates"] == [(queue_track, {"video_id": "s1", "duration": 210})]
        assert plan["appends"] == [{"video_id": "s2"}, {"video_id": "s3"}]

    def test_the_updates_carry_the_very_objects(self):
        queue_track = {"video_id": "s1"}
        full_track = {"video_id": "s1", "duration": 210}
        plan = _artist_songs_merge_plan([full_track], [{"video_id": "s1"}], [queue_track])
        assert plan["updates"][0][0] is queue_track
        assert plan["updates"][0][1] is full_track

    def test_a_duplicate_video_id_keeps_the_last_full_track(self):
        # The dict build is last-wins, exactly the reference's own.
        first, last = {"video_id": "s1", "title": "old"}, {"video_id": "s1", "title": "new"}
        queue_track = {"video_id": "s1"}
        plan = _artist_songs_merge_plan([first, last], [], [queue_track])
        assert plan["updates"] == [(queue_track, last)]

    def test_duplicates_within_the_fetch_are_all_appended(self):
        # The append pass checks membership against the initial batch
        # only — both copies of a duplicated fetch ride home.
        dup_a, dup_b = {"video_id": "s1", "title": "a"}, {"video_id": "s1", "title": "b"}
        plan = _artist_songs_merge_plan([dup_a, dup_b], [{"video_id": "other"}], [])
        assert plan["appends"] == [dup_a, dup_b]

    def test_a_track_without_a_video_id_is_appended(self):
        # The index filter skips it, but the append walk only checks the
        # initial batch's ids — "" is not among them, so it rides home.
        no_vid = {"title": "no vid"}
        plan = _artist_songs_merge_plan([no_vid], [{"video_id": "s1"}], [])
        assert plan["appends"] == [no_vid]

    def test_an_initial_track_without_a_video_id_parks_the_empty_id(self):
        # "" lands in the initial set, so a fetched track without a
        # video id is NOT appended.
        no_vid = {"title": "no vid"}
        plan = _artist_songs_merge_plan([no_vid], [{}], [])
        assert plan["appends"] == []

    def test_a_queue_track_not_in_the_full_list_is_not_enriched(self):
        plan = _artist_songs_merge_plan(
            [{"video_id": "s1"}], [{"video_id": "s1"}], [{"video_id": "q"}]
        )
        assert plan["updates"] == []

    def test_empty_inputs_answer_an_empty_plan(self):
        assert _artist_songs_merge_plan([], [], []) == {"updates": [], "appends": []}

    def test_an_unhashable_video_id_raises_the_dict_build_error(self):
        with pytest.raises(TypeError, match="unhashable type"):
            _artist_songs_merge_plan([{"video_id": ["not-hashable"]}], [], [])

    def test_an_unhashable_initial_id_raises_the_set_build_error(self):
        with pytest.raises(TypeError, match="unhashable type"):
            _artist_songs_merge_plan([], [{"video_id": ["not-hashable"]}], [])


# ── The artist-popup gate ───────────────────────────────────────────


class TestArtistPopupPlan:
    """``_artist_popup_plan``'s classification, on whichever backend."""

    def test_a_missing_artists_key_answers_none(self):
        assert _artist_popup_plan({"title": "T"}) == {"arm": "none"}

    def test_a_none_artists_value_answers_none(self):
        assert _artist_popup_plan({"artists": None}) == {"arm": "none"}

    def test_an_empty_artists_list_answers_none(self):
        assert _artist_popup_plan({"artists": []}) == {"arm": "none"}

    def test_a_non_list_artists_value_answers_none(self):
        assert _artist_popup_plan({"artists": "UC1"}) == {"arm": "none"}

    def test_artists_without_any_id_answer_none(self):
        assert _artist_popup_plan({"artists": [{"name": "A"}, {"name": "B"}]}) == {"arm": "none"}

    def test_an_empty_id_does_not_count_as_valid(self):
        assert _artist_popup_plan({"artists": [{"name": "A", "id": ""}]}) == {"arm": "none"}

    def test_a_browse_id_counts_as_valid(self):
        plan = _artist_popup_plan({"artists": [{"name": "A", "browseId": "BR"}]})
        assert plan["arm"] == "single"
        assert plan["artist"] == {"name": "A", "browseId": "BR"}
        assert plan["valid"] == [{"name": "A", "browseId": "BR"}]

    def test_exactly_one_valid_artist_answers_single(self):
        a, b = {"name": "A", "id": "UC1"}, {"name": "B"}
        plan = _artist_popup_plan({"artists": [a, b]})
        assert plan == {"arm": "single", "valid": [a], "artist": a}

    def test_several_valid_artists_answer_pick(self):
        a, b = {"name": "A", "id": "UC1"}, {"name": "B", "id": "UC2"}
        plan = _artist_popup_plan({"artists": [a, {"name": "X"}, b]})
        assert plan == {"arm": "pick", "valid": [a, b]}

    def test_the_valid_list_keeps_the_artists_own_order(self):
        a, b = {"name": "A", "id": "UC1"}, {"name": "B", "browseId": "BR2"}
        plan = _artist_popup_plan({"artists": [b, a]})
        assert plan["valid"] == [b, a]


# ── The artist picker's rows ────────────────────────────────────────


class TestArtistPickerRows:
    """``_artist_picker_rows``'s assembly, on whichever backend."""

    def test_one_pair_per_artist_in_order(self):
        rows = _artist_picker_rows([{"name": "A"}, {"name": "B"}, {"name": "C"}])
        assert rows == [("0", "A"), ("1", "B"), ("2", "C")]

    def test_a_missing_name_answers_the_default(self):
        assert _artist_picker_rows([{"id": "UC1"}]) == [("0", "Unknown Artist")]

    def test_a_falsy_name_answers_itself(self):
        assert _artist_picker_rows([{"name": ""}]) == [("0", "")]

    def test_the_name_rides_home_verbatim(self):
        assert _artist_picker_rows([{"name": 42}]) == [("0", 42)]

    def test_an_empty_artists_list_answers_no_rows(self):
        assert _artist_picker_rows([]) == []


# ── The artist popup's item ─────────────────────────────────────────


class TestArtistPopupItem:
    """``_artist_popup_item``'s gate and assembly, on whichever backend."""

    def test_the_id_outranks_the_browse_id(self):
        verdict = _artist_popup_item({"id": "UC1", "browseId": "BR", "name": "A"})
        assert verdict == {
            "arm": "item",
            "browse_id": "UC1",
            "item": {"browseId": "UC1", "artist": "A"},
        }

    def test_a_falsy_id_falls_through_to_the_browse_id(self):
        verdict = _artist_popup_item({"id": "", "browseId": "BR", "name": "A"})
        assert verdict["browse_id"] == "BR"
        assert verdict["item"] == {"browseId": "BR", "artist": "A"}

    def test_a_missing_browse_id_answers_the_default_and_gates(self):
        assert _artist_popup_item({"name": "A"}) == {"arm": "none"}

    def test_an_empty_browse_id_gates_too(self):
        assert _artist_popup_item({"id": "", "browseId": ""}) == {"arm": "none"}

    def test_a_missing_name_answers_the_default(self):
        verdict = _artist_popup_item({"id": "UC1"})
        assert verdict["item"] == {"browseId": "UC1", "artist": "Unknown Artist"}

    def test_the_name_rides_home_verbatim(self):
        verdict = _artist_popup_item({"id": "UC1", "name": 7})
        assert verdict["item"] == {"browseId": "UC1", "artist": 7}


# ── The album popup's item ──────────────────────────────────────────


class TestAlbumPopupItem:
    """``_album_popup_item``'s gate and assembly, on whichever backend."""

    def test_the_album_id_outranks_the_nested_album_id(self):
        verdict = _album_popup_item({"album_id": "ALB1", "album": {"id": "ALB2", "name": "N"}})
        assert verdict == {
            "arm": "item",
            "album_id": "ALB1",
            "item": {"browseId": "ALB1", "title": "N"},
        }

    def test_a_missing_album_id_falls_through_to_the_nested_id(self):
        verdict = _album_popup_item({"album": {"id": "ALB2", "name": "N"}})
        assert verdict["album_id"] == "ALB2"
        assert verdict["item"] == {"browseId": "ALB2", "title": "N"}

    def test_a_string_album_names_the_item(self):
        verdict = _album_popup_item({"album": "My Album"})
        assert verdict["arm"] == "none"  # a string album carries no id

    def test_an_album_id_with_a_string_album(self):
        verdict = _album_popup_item({"album_id": "ALB1", "album": "My Album"})
        assert verdict["item"] == {"browseId": "ALB1", "title": "My Album"}

    def test_a_non_dict_album_contributes_no_nested_id(self):
        verdict = _album_popup_item({"album": "plain", "browseId": "BR"})
        assert verdict == {"arm": "none"}  # the reference reads no top-level browseId here

    def test_a_missing_album_id_gates(self):
        assert _album_popup_item({"title": "T"}) == {"arm": "none"}

    def test_an_empty_nested_id_gates(self):
        assert _album_popup_item({"album": {"id": "", "name": "N"}}) == {"arm": "none"}

    def test_a_missing_album_name_answers_the_default(self):
        verdict = _album_popup_item({"album": {"id": "ALB"}})
        assert verdict["item"] == {"browseId": "ALB", "title": "Unknown Album"}

    def test_a_truthy_artists_list_is_attached_verbatim(self):
        artists = [{"name": "A", "id": "UC1"}]
        verdict = _album_popup_item({"album": {"id": "ALB"}, "artists": artists})
        assert verdict["item"]["artists"] is artists

    def test_a_falsy_artists_list_is_not_attached(self):
        verdict = _album_popup_item({"album": {"id": "ALB"}, "artists": []})
        assert "artists" not in verdict["item"]


# ── Error shapes ─────────────────────────────────────────────────────────


class TestEngineErrorShapes:
    """The engines' errors stay CPython's own, on whichever backend."""

    def test_the_entity_route_needs_a_mapping_item(self):
        with pytest.raises(AttributeError):
            _entity_action_route("play_all", 5, "album")

    def test_a_raising_queue_row_read_propagates_through_the_in_queue_walk(self):
        class Boom:
            def get(self, *_args: object) -> object:
                raise RuntimeError("row read failed")

        with pytest.raises(RuntimeError, match="row read failed"):
            _in_queue_verdict("v", [Boom()])

    def test_a_raising_track_read_propagates_through_the_like_rating(self):
        class Boom:
            def get(self, *_args: object) -> object:
                raise RuntimeError("track read failed")

        with pytest.raises(RuntimeError, match="track read failed"):
            _like_toggle_rating(Boom())

    def test_the_artist_popup_plan_needs_a_mapping_track(self):
        with pytest.raises(AttributeError):
            _artist_popup_plan(5)

    def test_the_album_popup_item_needs_a_mapping_track(self):
        with pytest.raises(AttributeError):
            _album_popup_item(5)

    def test_the_artist_popup_item_needs_a_mapping_artist(self):
        with pytest.raises(AttributeError):
            _artist_popup_item(5)


# ── Wiring: the actions popup ────────────────────────────────────────────


def _popup_host(tracks: list | None = None) -> MagicMock:
    """A mocked host whose popup-open helper is the real one."""
    host = MagicMock()
    host.queue = MagicMock()
    host.queue.tracks = tracks if tracks is not None else []
    host._refresh_queue_page = MagicMock()
    host.notify = MagicMock()
    host.run_worker = MagicMock()
    host._open_actions_for_track = types.MethodType(TrackActionsMixin._open_actions_for_track, host)
    return host


def _confirm(host: MagicMock, track: dict, **kwargs: object) -> object:
    """Open the popup for *track* and return its result callback."""
    host._open_actions_for_track(track, **kwargs)
    return host.push_screen.call_args.args[1]


def _popups_action_ids(host: MagicMock) -> list[str]:
    popup = host.push_screen.call_args.args[0]
    return [action_id for action_id, _label in popup._actions]


class TestPopupWiring:
    """``_open_actions_for_track`` routes through the engines and keeps
    its own widget calls."""

    def test_the_in_queue_engine_is_consulted_with_the_hosts_own_values(self, monkeypatch):
        calls: list[tuple] = []

        def recording_engine(video_id, queue_tracks):
            calls.append((video_id, queue_tracks))
            return False

        monkeypatch.setattr(track_actions, "_in_queue_verdict", recording_engine)
        tracks: list = [{"video_id": "v"}]
        host = _popup_host(tracks)
        _confirm(host, _track("v"))

        assert calls == [("v", tracks)]  # the very list object

    def test_an_in_queue_track_swaps_the_popups_queue_action(self):
        host = _popup_host([{"video_id": "v"}])
        _confirm(host, _track("v"))

        ids = _popups_action_ids(host)
        assert "remove_from_queue" in ids
        assert "add_to_queue" not in ids

    def test_a_track_not_queued_keeps_the_add_action(self):
        host = _popup_host([{"video_id": "other"}])
        _confirm(host, _track("v"))

        ids = _popups_action_ids(host)
        assert "add_to_queue" in ids
        assert "remove_from_queue" not in ids

    def test_the_route_engine_is_consulted_with_the_popups_answer(self, monkeypatch):
        calls: list = []

        def recording_engine(action_id):
            calls.append(action_id)
            return {"handler": "play"}

        monkeypatch.setattr(track_actions, "_track_action_route", recording_engine)
        host = _popup_host()
        confirm = _confirm(host, _track("v"))

        confirm("play")

        assert calls == ["play"]
        host.run_worker.assert_called_once()

    def test_an_unhandled_answer_does_nothing(self):
        host = _popup_host()
        confirm = _confirm(host, _track("v"))

        confirm("nonsense")

        host.run_worker.assert_not_called()
        host.notify.assert_not_called()

    def test_a_dismissed_popup_does_nothing(self):
        host = _popup_host()
        confirm = _confirm(host, _track("v"))

        confirm(None)

        host.run_worker.assert_not_called()
        host.notify.assert_not_called()

    async def test_the_like_rating_engine_is_consulted_on_toggle_like(self, monkeypatch):
        calls: list = []

        def recording_engine(track):
            calls.append(track)
            return {"is_liked": False, "rating": "LIKE", "label": "Liked"}

        monkeypatch.setattr(track_actions, "_like_toggle_rating", recording_engine)
        host = _popup_host()
        host.ytmusic = AsyncMock()
        host.ytmusic.rate_song = AsyncMock(return_value="success")
        track = _track("v")
        confirm = _confirm(host, track)

        confirm("toggle_like")
        await host.run_worker.call_args.args[0]

        assert calls == [track]  # the very track object
        host.ytmusic.rate_song.assert_awaited_once_with("v", "LIKE")
        assert track["likeStatus"] == "LIKE"
        host.notify.assert_called_once_with("Liked", timeout=2)

    async def test_the_toggle_like_failure_toasts_the_error(self):
        host = _popup_host()
        host.ytmusic = AsyncMock()
        host.ytmusic.rate_song = AsyncMock(return_value="failure")
        confirm = _confirm(host, _track("v"))

        confirm("toggle_like")
        await host.run_worker.call_args.args[0]

        assert host.notify.call_args.kwargs.get("severity") == "error"
        assert "Couldn't" in host.notify.call_args.args[0]

    def test_the_go_to_artist_arm_navigates_with_the_engines_id(self):
        host = _popup_host()
        track = _track("v", artists=[{"id": "UC1", "name": "A"}])
        confirm = _confirm(host, track)

        confirm("go_to_artist")

        host.run_worker.assert_called_once()

    def test_the_remove_from_queue_arm_consults_the_removal(self):
        host = _popup_host()
        host._remove_from_queue = MagicMock()
        confirm = _confirm(host, _track("v"), queue_entry_id=7, from_queue=True)

        confirm("remove_from_queue")

        host._remove_from_queue.assert_called_once_with(
            host._remove_from_queue.call_args.args[0], 7, from_queue=True
        )


# ── Wiring: the removal ──────────────────────────────────────────────────


class TestRemovalWiring:
    """``_remove_from_queue`` consults the plan engine with the host's own
    values and steers the removals and toasts by its arm."""

    def _host(self) -> MagicMock:
        host = MagicMock()
        host.queue = MagicMock()
        host.queue.tracks = []
        host._refresh_queue_page = MagicMock()
        host.notify = MagicMock()
        host._remove_from_queue = types.MethodType(TrackActionsMixin._remove_from_queue, host)
        return host

    def test_the_plan_engine_is_consulted_with_the_hosts_own_values(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(from_queue, queue_entry_id, queue_tracks, video_id):
            calls.append(
                {
                    "from_queue": from_queue,
                    "queue_entry_id": queue_entry_id,
                    "queue_tracks": queue_tracks,
                    "video_id": video_id,
                }
            )
            return {"arm": "none"}

        monkeypatch.setattr(track_actions, "_remove_from_queue_plan", recording_engine)
        host = self._host()
        tracks: list = [{"video_id": "x"}]
        host.queue.tracks = tracks
        track = _track("v")

        host._remove_from_queue(track, 4, from_queue=True)

        assert calls == [
            {
                "from_queue": True,
                "queue_entry_id": 4,
                "queue_tracks": tracks,  # the very list object
                "video_id": get_video_id(track),
            }
        ]
        host.notify.assert_not_called()

    def test_the_entry_arm_removes_the_occurrence(self, monkeypatch):
        monkeypatch.setattr(
            track_actions, "_remove_from_queue_plan", lambda *args: {"arm": "entry"}
        )
        host = self._host()
        host.queue.remove_entry = MagicMock(return_value=True)

        host._remove_from_queue(_track("v"), 3)

        host.queue.remove_entry.assert_called_once_with(3)
        host._refresh_queue_page.assert_called_once()
        host.notify.assert_called_once_with("Removed from queue", timeout=2)

    def test_the_entry_arm_warns_when_the_occurrence_is_gone(self, monkeypatch):
        monkeypatch.setattr(
            track_actions, "_remove_from_queue_plan", lambda *args: {"arm": "entry"}
        )
        host = self._host()
        host.queue.remove_entry = MagicMock(return_value=False)

        host._remove_from_queue(_track("v"), 3)

        host._refresh_queue_page.assert_called_once()
        host.notify.assert_called_once_with(
            "That queue entry is no longer present", severity="warning", timeout=3
        )

    def test_the_stale_arm_warns_without_touching_the_queue(self, monkeypatch):
        monkeypatch.setattr(
            track_actions, "_remove_from_queue_plan", lambda *args: {"arm": "stale"}
        )
        host = self._host()

        host._remove_from_queue(_track("v"), None, from_queue=True)

        host.queue.remove_entry.assert_not_called()
        host.queue.remove.assert_not_called()
        host._refresh_queue_page.assert_called_once()
        host.notify.assert_called_once_with(
            "That queue entry is no longer present", severity="warning", timeout=3
        )

    def test_the_remove_at_arm_removes_the_indexed_row(self, monkeypatch):
        monkeypatch.setattr(
            track_actions, "_remove_from_queue_plan", lambda *args: {"arm": "remove_at", "index": 2}
        )
        host = self._host()

        host._remove_from_queue(_track("v"), None)

        host.queue.remove.assert_called_once_with(2)
        host._refresh_queue_page.assert_called_once()
        host.notify.assert_called_once_with("Removed from queue", timeout=2)


# ── Wiring: the entity dispatch ──────────────────────────────────────────


class TestEntityDispatchWiring:
    """``_dispatch_entity_action`` consults the route engine and awaits
    the engine's own ids."""

    def _host(self) -> MagicMock:
        host = MagicMock()
        host.notify = MagicMock()
        host._play_album = AsyncMock()
        host._play_playlist = AsyncMock()
        host._add_album_to_queue = AsyncMock()
        host._add_playlist_to_queue = AsyncMock()
        host._add_album_next = AsyncMock()
        host._add_playlist_next = AsyncMock()
        host._start_artist_radio = AsyncMock()
        host._start_playlist_radio = AsyncMock()
        host._play_artist_top_songs = AsyncMock()
        host._add_album_to_library = AsyncMock()
        host.navigate_to = AsyncMock()
        return host

    async def test_the_route_engine_is_consulted_with_the_actions_own_values(self, monkeypatch):
        calls: list[tuple] = []

        def recording_engine(action_id, item, item_type):
            calls.append((action_id, item, item_type))
            return {
                "handled": True,
                "handler": "_play_album",
                "shuffle": True,
                "entity_id": "E",
                "entity_name": "N",
            }

        monkeypatch.setattr(track_actions, "_entity_action_route", recording_engine)
        host = self._host()
        item = {"browseId": "ALB", "title": "A"}

        result = await TrackActionsMixin._dispatch_entity_action(
            host, "shuffle_play", item, "album"
        )

        assert result is True
        assert calls == [("shuffle_play", item, "album")]  # the very item object
        host._play_album.assert_awaited_once_with("E", "N", shuffle=True)

    async def test_the_unhandled_route_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            track_actions,
            "_entity_action_route",
            lambda *args: {"handled": False, "handler": "unhandled"},
        )
        host = self._host()

        result = await TrackActionsMixin._dispatch_entity_action(
            host, "copy_link", {"browseId": "X"}, "album"
        )

        assert result is False
        host.notify.assert_not_called()

    async def test_the_notify_route_toasts_and_is_handled(self, monkeypatch):
        monkeypatch.setattr(
            track_actions,
            "_entity_action_route",
            lambda *args: {"handled": True, "handler": "notify"},
        )
        host = self._host()

        result = await TrackActionsMixin._dispatch_entity_action(host, "play_all", {}, "album")

        assert result is True
        host.notify.assert_called_once_with("No ID available", severity="error", timeout=2)

    async def test_the_navigate_artist_route_awaits_the_engines_target(self, monkeypatch):
        monkeypatch.setattr(
            track_actions,
            "_entity_action_route",
            lambda *args: {"handled": True, "handler": "_navigate_artist", "artist_id": "UC1"},
        )
        host = self._host()

        result = await TrackActionsMixin._dispatch_entity_action(host, "go_to_artist", {}, "album")

        assert result is True
        host.navigate_to.assert_awaited_once_with(
            "context", context_type="artist", context_id="UC1"
        )

    async def test_the_navigate_album_route_awaits_the_engines_target(self, monkeypatch):
        monkeypatch.setattr(
            track_actions,
            "_entity_action_route",
            lambda *args: {"handled": True, "handler": "_navigate_album", "album_id": "AL1"},
        )
        host = self._host()

        result = await TrackActionsMixin._dispatch_entity_action(host, "go_to_album", {}, "track")

        assert result is True
        host.navigate_to.assert_awaited_once_with("context", context_type="album", context_id="AL1")

    async def test_the_start_playlist_radio_route_passes_the_item_itself(self, monkeypatch):
        monkeypatch.setattr(
            track_actions,
            "_entity_action_route",
            lambda *args: {"handled": True, "handler": "_start_playlist_radio"},
        )
        host = self._host()
        item = {"browseId": "PL"}

        result = await TrackActionsMixin._dispatch_entity_action(
            host, "start_radio", item, "playlist"
        )

        assert result is True
        host._start_playlist_radio.assert_awaited_once_with(item)

    async def test_the_add_to_library_route_awaits_with_the_engines_ids(self, monkeypatch):
        monkeypatch.setattr(
            track_actions,
            "_entity_action_route",
            lambda *args: {
                "handled": True,
                "handler": "_add_album_to_library",
                "entity_id": "E",
                "entity_name": "N",
            },
        )
        host = self._host()

        result = await TrackActionsMixin._dispatch_entity_action(
            host, "add_to_library", {}, "album"
        )

        assert result is True
        host._add_album_to_library.assert_awaited_once_with("E", "N")


# ── Wiring: the artist-songs merge ───────────────────────────────────────


class TestMergePlanWiring:
    """``_fetch_remaining_artist_songs`` consults the merge plan engine
    with the normalized lists and applies its very objects."""

    def _host(self, playlist_payload: dict) -> MagicMock:
        host = MagicMock()
        host.ytmusic = MagicMock()
        host.ytmusic.get_playlist = AsyncMock(return_value=playlist_payload)
        host.queue = MagicMock()
        host.queue.tracks = []
        host.queue.generation = 1
        host.queue.add = MagicMock()
        host._refresh_queue_page = MagicMock()
        return host

    async def test_the_plan_engine_is_consulted_with_the_normalized_lists(self, monkeypatch):
        calls: list[tuple] = []

        def recording_engine(all_tracks, initial_tracks, queue_tracks):
            calls.append((all_tracks, initial_tracks, queue_tracks))
            return {"updates": [], "appends": []}

        monkeypatch.setattr(track_actions, "_artist_songs_merge_plan", recording_engine)
        queue_track = {"video_id": "s1"}
        host = self._host({"tracks": [{"videoId": "s1", "duration_seconds": 210}]})
        host.queue.tracks = [queue_track]
        initial = [{"video_id": "s1"}]

        await TrackActionsMixin._fetch_remaining_artist_songs(host, "VLPLfull", initial)

        assert len(calls) == 1
        all_tracks, initial_arg, queue_tracks = calls[0]
        # The fetch's raw payloads arrive normalized — the full track shape.
        assert [t["video_id"] for t in all_tracks] == ["s1"]
        assert [t["duration"] for t in all_tracks] == [210]
        assert initial_arg == initial
        assert queue_tracks == [queue_track]
        host.queue.add.assert_not_called()

    async def test_the_updates_and_appends_are_applied(self, monkeypatch):
        queue_track = {"video_id": "s1"}
        full_track = {"video_id": "s1", "duration": 210}
        appended = {"video_id": "s2"}
        monkeypatch.setattr(
            track_actions,
            "_artist_songs_merge_plan",
            lambda *args: {"updates": [(queue_track, full_track)], "appends": [appended]},
        )
        host = self._host({"tracks": []})

        await TrackActionsMixin._fetch_remaining_artist_songs(
            host, "VLPLfull", [{"video_id": "s1"}]
        )

        assert queue_track["duration"] == 210  # the in-place update hit the queue's own dict
        host.queue.add.assert_called_once_with(appended)
        host._refresh_queue_page.assert_called_once()


# ── Wiring: the artist/album popup-view assembly ─────────────────────


class TestArtistAlbumPopupWiring:
    """The artist/album popup-open helpers consult the popup-view
    engines and steer their own widget calls."""

    def test_the_plan_engine_is_consulted_with_the_very_track(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(track):
            calls.append(track)
            return {"arm": "none"}

        monkeypatch.setattr(track_actions, "_artist_popup_plan", recording_engine)
        host = MagicMock()
        track = {"artists": [{"name": "A", "id": "UC1"}]}

        TrackActionsMixin._open_actions_for_artist(host, track)

        assert calls == [track]
        host.notify.assert_called_once_with(
            "No artist info available.", severity="warning", timeout=2
        )
        host.push_screen.assert_not_called()

    def test_the_single_arm_opens_the_actions_popup_for_that_artist(self):
        host = MagicMock()
        a = {"name": "A", "id": "UC1"}

        TrackActionsMixin._open_actions_for_artist(host, {"artists": [a]})

        host._show_artist_actions.assert_called_once_with(a)
        host.notify.assert_not_called()

    def test_the_pick_arm_opens_the_picker_with_the_valid_list(self):
        host = MagicMock()
        a, b = {"name": "A", "id": "UC1"}, {"name": "B", "id": "UC2"}

        TrackActionsMixin._open_actions_for_artist(host, {"artists": [a, b]})

        host._show_artist_picker.assert_called_once()
        args = host._show_artist_picker.call_args.args
        assert args[0] == {"artists": [a, b]}
        assert args[1] == [a, b]

    def test_the_picker_rides_the_rows_engine(self, monkeypatch):
        calls: list[list] = []

        def recording_engine(artists):
            calls.append(artists)
            return [("9", "Picked")]

        monkeypatch.setattr(track_actions, "_artist_picker_rows", recording_engine)
        host = MagicMock()
        artists = [{"name": "A", "id": "UC1"}]

        TrackActionsMixin._show_artist_picker(host, {"artists": artists}, artists)

        assert calls == [artists]
        popup, _callback = host.push_screen.call_args.args
        assert popup._actions == [("9", "Picked")]
        assert popup.item == {"title": "Select Artist"}

    def test_the_artist_actions_ride_the_item_engine(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(artist):
            calls.append(artist)
            return {
                "arm": "item",
                "browse_id": "UC1",
                "item": {"browseId": "UC1", "artist": "A"},
            }

        monkeypatch.setattr(track_actions, "_artist_popup_item", recording_engine)
        host = MagicMock()
        artist = {"id": "UC1", "name": "A"}

        TrackActionsMixin._show_artist_actions(host, artist)

        assert calls == [artist]
        popup, _callback = host.push_screen.call_args.args
        assert popup.item == {"browseId": "UC1", "artist": "A"}
        assert popup.item_type == "artist"

    def test_the_artist_actions_with_no_browse_id_push_nothing(self):
        host = MagicMock()

        TrackActionsMixin._show_artist_actions(host, {"name": "A"})

        host.push_screen.assert_not_called()

    def test_the_back_entry_extends_the_artists_action_list(self):
        from ytm_player.ui.popups.actions import ARTIST_ACTIONS

        host = MagicMock()
        # The picker's callback must reach the real actions helper, not a
        # mock — it is the one that extends ARTIST_ACTIONS with _back.
        host._show_artist_actions = types.MethodType(TrackActionsMixin._show_artist_actions, host)
        track = {"artists": [{"name": "A", "id": "UC1"}]}
        artists = track["artists"]

        TrackActionsMixin._show_artist_picker(host, track, artists)
        _on_pick = host.push_screen.call_args.args[1]
        _on_pick("0")

        popup, _callback = host.push_screen.call_args.args
        assert popup._actions == list(ARTIST_ACTIONS) + [("_back", "← Back")]

    def test_the_album_item_engine_is_consulted_with_the_very_track(self, monkeypatch):
        calls: list[dict] = []

        def recording_engine(track):
            calls.append(track)
            return {"arm": "none"}

        monkeypatch.setattr(track_actions, "_album_popup_item", recording_engine)
        host = MagicMock()
        track = {"album": {"id": "ALB"}}

        TrackActionsMixin._open_actions_for_album(host, track)

        assert calls == [track]
        host.notify.assert_called_once_with(
            "No album info available.", severity="warning", timeout=2
        )
        host.push_screen.assert_not_called()

    def test_the_album_popup_carries_the_engines_item(self):
        host = MagicMock()
        artists = [{"name": "A", "id": "UC1"}]

        TrackActionsMixin._open_actions_for_album(
            host, {"album": {"id": "ALB", "name": "N"}, "artists": artists}
        )

        popup, _callback = host.push_screen.call_args.args
        assert popup.item == {"browseId": "ALB", "title": "N", "artists": artists}
        assert popup.item_type == "album"


# ── Native parity ────────────────────────────────────────────────────────

# First-artist cases: the source dicts, scalars only.
_FIRST_ARTIST_PARITY_CASES = [
    {},
    {"artists": None},
    {"artists": "UC1"},
    {"artists": []},
    {"artists": ["UC1"]},
    {"artists": [{"name": "A"}]},
    {"artists": [{"id": "UC1", "name": "A"}]},
    {"artists": [{"id": "", "browseId": "BR1"}]},
    {"artists": [{"id": "", "browseId": ""}]},
    {"artists": [{"id": 0, "browseId": "BR2"}]},
    {"artists": [{"id": 7}]},
    {"artists": [{"name": "A"}, {"id": "UC2"}]},
    {"artists": [{"id": "UC1"}, {"id": "UC2"}]},
    {"artists": [{"browseId": "BR3"}]},
]

# Album-browse cases: the source dicts.
_ALBUM_BROWSE_PARITY_CASES = [
    {},
    {"album_id": "A1"},
    {"album": {"id": "A2"}},
    {"album": "plain string"},
    {"browseId": "BR1"},
    {"album_id": "A1", "album": {"id": "A2"}, "browseId": "BR"},
    {"album_id": "", "album": {"id": "A2"}, "browseId": "BR"},
    {"album_id": None, "album": {"id": None}, "browseId": "BR"},
    {"album_id": 0, "album": {"id": 0}, "browseId": 0},
    {"album": {"id": "A2"}, "browseId": "BR"},
    {"album": {"id": ""}, "browseId": "BR"},
    {"album_id": "A1", "browseId": "BR"},
]

# Occurrence-gate cases: [from_queue, key].
_QUEUE_ENTRY_PARITY_CASES = [
    [True, 5],
    [False, 5],
    [True, "5"],
    [True, None],
    [True, True],
    [True, False],
    [1, 7],
    [0, 7],
    [True, 5.0],
    [True, [1]],
]

# Popup-answer cases: the action ids, None included.
_TRACK_ROUTE_PARITY_CASES = [
    None,
    "add_to_playlist",
    "play",
    "download",
    "play_next",
    "add_to_queue",
    "remove_from_queue",
    "start_radio",
    "go_to_artist",
    "go_to_album",
    "toggle_like",
    "copy_link",
    "remove_from_playlist",
    "nonsense",
    "",
]

# Entity-route cases: [action_id, item, item_type].
_ENTITY_ROUTE_PARITY_CASES = [
    ["play_all", {"browseId": "ALB", "title": "A"}, "album"],
    ["shuffle_play", {"browseId": "ALB", "title": "A"}, "album"],
    ["play_all", {"browseId": "PL"}, "playlist"],
    ["shuffle_play", {"playlistId": "PL2", "name": "N"}, "playlist"],
    ["play_all", {"browseId": "X"}, "artist"],
    ["play_all", {}, "album"],
    ["play_all", {"album_id": "A3"}, "album"],
    ["shuffle_play", {"browseId": ""}, "playlist"],
    ["add_to_queue", {"browseId": "ALB", "title": "A"}, "album"],
    ["add_to_queue", {"playlistId": "PL", "artist": "Ar"}, "playlist"],
    ["add_to_queue", {}, "album"],
    ["add_to_queue", {"browseId": "X"}, "artist"],
    ["play_next", {"browseId": "ALB", "title": "A"}, "album"],
    ["play_next", {"playlistId": "PL"}, "playlist"],
    ["play_next", {"browseId": ""}, "album"],
    ["play_next", {"browseId": "X"}, "track"],
    ["start_radio", {"browseId": "UC1"}, "artist"],
    ["start_radio", {"browseId": ""}, "artist"],
    ["start_radio", {"browseId": "PL"}, "playlist"],
    ["start_radio", {"browseId": "X"}, "album"],
    ["play_top_songs", {"browseId": "UC1"}, "artist"],
    ["play_top_songs", {}, "artist"],
    ["play_top_songs", {"browseId": "X"}, "playlist"],
    ["go_to_artist", {"artists": [{"id": "UC1", "name": "A"}], "title": "T"}, "album"],
    ["view_similar", {"artists": [{"name": "A"}]}, "album"],
    ["go_to_artist", {"artists": []}, "artist"],
    ["go_to_artist", {"browseId": "UC9"}, "artist"],
    ["go_to_artist", {"browseId": "UC9"}, "album"],
    ["go_to_artist", {"artists": [{"id": "UC1"}, {"id": "UC2"}], "browseId": "B"}, "playlist"],
    ["view_similar", {"artists": "notalist", "browseId": "UC7"}, "artist"],
    ["go_to_album", {"album_id": "AL1"}, "track"],
    ["go_to_album", {"album": {"id": "AL2"}}, "album"],
    ["go_to_album", {}, "album"],
    ["go_to_album", {"browseId": "BR"}, "playlist"],
    ["add_to_library", {"browseId": "ALB", "title": "A"}, "album"],
    ["add_to_library", {"browseId": ""}, "album"],
    ["add_to_library", {"browseId": "X"}, "artist"],
    ["copy_link", {"browseId": "X"}, "album"],
    ["toggle_subscribe", {"browseId": "X"}, "artist"],
    ["nonsense", {"browseId": "X"}, "album"],
    ["play_all", {"album_id": "A1", "name": "N"}, "album"],
    ["play_all", {"album_id": "A1", "artist": "Ar"}, "album"],
    ["play_all", {"album_id": "A1"}, "album"],
    ["play_all", 5, "album"],
]

# Like-rating cases: the track dicts.
_LIKE_PARITY_CASES = [
    {},
    {"likeStatus": "LIKE"},
    {"likeStatus": "DISLIKE"},
    {"liked": True},
    {"liked": 1},
    {"liked": 0},
    {"liked": None},
    {"liked": "yes"},
    {"likeStatus": "LIKE", "liked": False},
    {"likeStatus": "INDIFFERENT", "liked": True},
    {"likeStatus": None, "liked": False},
]

# Removal-plan cases: [from_queue, entry_id, tracks, video_id].
_REMOVE_PLAN_PARITY_CASES = [
    [True, 3, [{"video_id": "v"}], "v"],
    [False, 3, [{"video_id": "v"}], "v"],
    [True, None, [{"video_id": "v"}], "v"],
    [True, 0, [], ""],
    [False, None, [{"video_id": "v"}], "v"],
    [False, None, [{"video_id": "x"}, {"video_id": "v"}], "v"],
    [False, None, [{"video_id": "v"}], "v"],
    [False, None, [{"video_id": "x"}], "v"],
    [False, None, [{"video_id": "x"}], ""],
    [False, None, [], "v"],
    [False, None, [{"video_id": None}], "v"],
    [False, None, [{"video_id": 1}], True],
    [False, None, [{"title": "no vid"}], "v"],
]

# In-queue cases: [video_id, tracks].
_IN_QUEUE_PARITY_CASES = [
    ["", [{"video_id": "v"}]],
    [None, [{"video_id": None}]],
    ["v", [{"video_id": "v"}]],
    ["v", [{"video_id": "x"}, {"video_id": "v"}]],
    ["v", [{"video_id": "x"}]],
    ["v", []],
    ["v", [{"video_id": None}]],
    ["v", [{"title": "no vid"}]],
    [True, [{"video_id": 1}]],
    ["v", [{"video_id": "V"}]],
]

# Merge-plan cases: [all_tracks, initial_tracks, queue_tracks].
_MERGE_PARITY_CASES = [
    [
        [{"video_id": "s1", "duration": 210}, {"video_id": "s2"}, {"video_id": "s3"}],
        [{"video_id": "s1"}],
        [{"video_id": "s1"}],
    ],
    [
        [{"video_id": "s1", "title": "old"}, {"video_id": "s1", "title": "new"}],
        [],
        [{"video_id": "s1"}],
    ],
    [
        [{"video_id": "s1", "title": "a"}, {"video_id": "s1", "title": "b"}],
        [{"video_id": "other"}],
        [],
    ],
    [
        [{"title": "no vid"}],
        [{"video_id": "s1"}],
        [],
    ],
    [
        [{"title": "no vid"}],
        [{}],
        [],
    ],
    [
        [{"video_id": "s1"}],
        [{"video_id": "s1"}],
        [{"video_id": "q"}],
    ],
    [[[], []], [[], []], [[], []]],  # placeholder replaced below
    [
        [{"video_id": ["not-hashable"]}],
        [],
        [],
    ],
    [
        [],
        [{"video_id": ["not-hashable"]}],
        [],
    ],
]
# The empty-everything case, spelled plainly.
_MERGE_PARITY_CASES[6] = [[], [], []]

# Artist-popup-gate cases: the track dicts.
_ARTIST_PLAN_PARITY_CASES = [
    {},
    {"artists": None},
    {"artists": []},
    {"artists": "UC1"},
    {"artists": [{"name": "A"}]},
    {"artists": [{"name": "A"}, {"id": "UC2"}]},
    {"artists": [{"id": "UC1"}]},
    {"artists": [{"id": "", "browseId": "BR"}]},
    {"artists": [{"browseId": 0}]},
    {"artists": [{"id": 0, "browseId": "BR"}]},
    {"artists": [{"id": "UC1"}, {"id": "UC2"}]},
    {"artists": [{"name": "A", "id": "UC1"}, {"name": "B"}]},
    {"artists": ["not-a-dict"]},
]

# Artist-picker-row cases: the artist lists.
_ARTIST_ROWS_PARITY_CASES = [
    [],
    [{"name": "A"}],
    [{"id": "UC1"}],
    [{"name": ""}],
    [{"name": 42}],
    [{"name": "A"}, {"name": "B"}],
]

# Artist-item cases: the artist dicts.
_ARTIST_ITEM_PARITY_CASES = [
    {},
    {"id": "UC1"},
    {"id": "", "browseId": "BR"},
    {"browseId": ""},
    {"id": "", "browseId": ""},
    {"id": 7},
    {"id": "UC1", "name": "A"},
    {"name": "A"},
    {"id": "UC1", "name": 7},
    {"id": "", "name": "X"},
]

# Album-item cases: the track dicts.
_ALBUM_ITEM_PARITY_CASES = [
    {},
    {"album_id": "A1"},
    {"album": {"id": "A2"}},
    {"album": "str"},
    {"album_id": "A1", "album": {"id": "A2"}},
    {"album_id": "", "album": {"id": ""}},
    {"album": {"id": "A2", "name": "N"}},
    {"album_id": "A1", "album": "My Album"},
    {"album": {"id": "ALB"}, "artists": []},
    {"album": {"id": "ALB"}, "artists": [{"name": "A"}]},
    {"album": {"id": "ALB"}, "artists": None},
    {"album_id": 0, "album": {"id": 0}},
    {"album": None},
    {"album": 5, "album_id": "A1"},
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app import _track_actions as ta


def outcome(fn, *args):
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": result}


payload = json.load(sys.stdin)
json.dump(
    {
        "first_artist": [
            outcome(ta._first_artist_id, case) for case in payload["first_artist"]
        ],
        "album_browse": [
            outcome(ta._album_browse_id, case) for case in payload["album_browse"]
        ],
        "queue_entry": [
            outcome(ta._queue_entry_id, *case) for case in payload["queue_entry"]
        ],
        "track_route": [
            outcome(ta._track_action_route, case) for case in payload["track_route"]
        ],
        "entity_route": [
            outcome(ta._entity_action_route, *case) for case in payload["entity_route"]
        ],
        "like": [outcome(ta._like_toggle_rating, case) for case in payload["like"]],
        "remove_plan": [
            outcome(ta._remove_from_queue_plan, *case) for case in payload["remove_plan"]
        ],
        "in_queue": [
            outcome(ta._in_queue_verdict, *case) for case in payload["in_queue"]
        ],
        "merge": [
            outcome(ta._artist_songs_merge_plan, *case) for case in payload["merge"]
        ],
        "artist_plan": [
            outcome(ta._artist_popup_plan, case) for case in payload["artist_plan"]
        ],
        "artist_rows": [
            outcome(ta._artist_picker_rows, case) for case in payload["artist_rows"]
        ],
        "artist_item": [
            outcome(ta._artist_popup_item, case) for case in payload["artist_item"]
        ],
        "album_item": [
            outcome(ta._album_popup_item, case) for case in payload["album_item"]
        ],
    },
    sys.stdout,
)
"""


def _native_outcomes():
    """The parent-side outcomes — every engine over every case, run by
    whichever backend the module loaded (the native one, when the parity
    tests run)."""

    def outcome(fn, *args):
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"ok": result}

    return {
        "first_artist": [outcome(_first_artist_id, case) for case in _FIRST_ARTIST_PARITY_CASES],
        "album_browse": [outcome(_album_browse_id, case) for case in _ALBUM_BROWSE_PARITY_CASES],
        "queue_entry": [outcome(_queue_entry_id, *case) for case in _QUEUE_ENTRY_PARITY_CASES],
        "track_route": [outcome(_track_action_route, case) for case in _TRACK_ROUTE_PARITY_CASES],
        "entity_route": [
            outcome(_entity_action_route, *case) for case in _ENTITY_ROUTE_PARITY_CASES
        ],
        "like": [outcome(_like_toggle_rating, case) for case in _LIKE_PARITY_CASES],
        "remove_plan": [
            outcome(_remove_from_queue_plan, *case) for case in _REMOVE_PLAN_PARITY_CASES
        ],
        "in_queue": [outcome(_in_queue_verdict, *case) for case in _IN_QUEUE_PARITY_CASES],
        "merge": [outcome(_artist_songs_merge_plan, *case) for case in _MERGE_PARITY_CASES],
        "artist_plan": [outcome(_artist_popup_plan, case) for case in _ARTIST_PLAN_PARITY_CASES],
        "artist_rows": [outcome(_artist_picker_rows, case) for case in _ARTIST_ROWS_PARITY_CASES],
        "artist_item": [outcome(_artist_popup_item, case) for case in _ARTIST_ITEM_PARITY_CASES],
        "album_item": [outcome(_album_popup_item, case) for case in _ALBUM_ITEM_PARITY_CASES],
    }


def _run_child():
    """The pure-Python reference outcomes, run in a child interpreter
    with the extension import blocked."""
    payload = {
        "first_artist": _FIRST_ARTIST_PARITY_CASES,
        "album_browse": _ALBUM_BROWSE_PARITY_CASES,
        "queue_entry": _QUEUE_ENTRY_PARITY_CASES,
        "track_route": _TRACK_ROUTE_PARITY_CASES,
        "entity_route": _ENTITY_ROUTE_PARITY_CASES,
        "like": _LIKE_PARITY_CASES,
        "remove_plan": _REMOVE_PLAN_PARITY_CASES,
        "in_queue": _IN_QUEUE_PARITY_CASES,
        "merge": _MERGE_PARITY_CASES,
        "artist_plan": _ARTIST_PLAN_PARITY_CASES,
        "artist_rows": _ARTIST_ROWS_PARITY_CASES,
        "artist_item": _ARTIST_ITEM_PARITY_CASES,
        "album_item": _ALBUM_ITEM_PARITY_CASES,
    }
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same routes, same plans,
    same verdicts, same errors with the same messages — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_every_engine_matches_the_fallback(self):
        fallback = _run_child()
        # The JSON round-trip normalizes the native side the same way the
        # child's output was normalized (tuples ride home from the merge
        # plan as lists, exactly like the reference's own tuples do).
        native = json.loads(json.dumps(_native_outcomes()))
        for family, cases in [
            ("first_artist", _FIRST_ARTIST_PARITY_CASES),
            ("album_browse", _ALBUM_BROWSE_PARITY_CASES),
            ("queue_entry", _QUEUE_ENTRY_PARITY_CASES),
            ("track_route", _TRACK_ROUTE_PARITY_CASES),
            ("entity_route", _ENTITY_ROUTE_PARITY_CASES),
            ("like", _LIKE_PARITY_CASES),
            ("remove_plan", _REMOVE_PLAN_PARITY_CASES),
            ("in_queue", _IN_QUEUE_PARITY_CASES),
            ("merge", _MERGE_PARITY_CASES),
            ("artist_plan", _ARTIST_PLAN_PARITY_CASES),
            ("artist_rows", _ARTIST_ROWS_PARITY_CASES),
            ("artist_item", _ARTIST_ITEM_PARITY_CASES),
            ("album_item", _ALBUM_ITEM_PARITY_CASES),
        ]:
            for case, n, f in zip(cases, native[family], fallback[family]):
                assert n == f, (
                    f"native and fallback {family} engine differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
