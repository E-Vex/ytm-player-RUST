"""Tests for the ytmusic browse-wrapper engines.

The pure cores behind ``services/ytmusic.py``'s read wrappers: the
watch-playlist kwargs assembly ``_watch_playlist_kwargs``, the
result-tracks extraction ``_result_tracks`` (shared by
``get_liked_songs``, ``get_watch_playlist``, ``get_radio``'s per-seed
fetch, ``get_playlist_radio`` and the discovery trending/liked
mixes), the list-gated playlist-tracks extraction
``_playlist_tracks`` (behind ``get_chart_shelf_tracks``), the
deduplicated radio seed pool ``_radio_seed_pool`` (behind
``get_radio``), the six-source rotation ``_discovery_source_order``
and the chart-shelf assembly ``_chart_shelves`` plus its round-robin
shelf pick ``_chart_shelf_index`` (behind ``get_discovery_mix``'s
charts mix), the playable filters ``_discovery_playable_tracks`` /
``_chart_playable_tracks``, the explore new-releases extraction
``_explore_new_releases`` (behind ``get_new_releases``) and the
playlist sort-order lookup ``_playlist_order_params`` (behind
``get_playlist``). The pure-Python transcriptions live in
``services/ytmusic.py``; when the Rust extension is built the same
engines run natively
(``_ytm_player_native.ytm_watch_playlist_kwargs`` and friends).

The unit tests below assert each engine's contract on whichever
backend is loaded — the member gates, the read orders, the
pass-through identity and the protocol errors. When the extension is
built, the parity tests additionally compare the native engines
against the pure-Python references (run in a child interpreter with
the extension import blocked) case for case — including the error
shapes, where CPython's own messages must surface identically — so
the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.services.ytmusic import (
    YTMusicService,
    _chart_playable_tracks,
    _chart_shelf_index,
    _chart_shelves,
    _discovery_playable_tracks,
    _discovery_source_order,
    _explore_new_releases,
    _playlist_order_params,
    _playlist_tracks,
    _radio_seed_pool,
    _result_tracks,
    _watch_playlist_kwargs,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class LogGet:
    """A track-shaped object recording the keys its ``get`` reads."""

    def __init__(self, answers: dict, log: list, boom: tuple = ()) -> None:
        self._answers = answers
        self._log = log
        self._boom = boom

    def get(self, key: str, default: object = None) -> object:
        self._log.append(key)
        if key in self._boom:
            raise RuntimeError("boom")
        return self._answers.get(key, default)


class BoomGetDict(dict):
    """A real dict whose ``get`` raises — error pass-through for the
    gated filters."""

    def get(self, key: str, default: object = None) -> object:
        raise RuntimeError("boom")


class LogGetDict(dict):
    """A dict recording the keys its ``get`` reads, answering per key."""

    def __init__(self, answers: dict, log: list, boom: tuple = ()) -> None:
        super().__init__()
        self._answers = answers
        self._log = log
        self._boom = boom

    def get(self, key: str, default: object = None) -> object:
        self._log.append(key)
        if key in self._boom:
            raise RuntimeError("boom")
        return self._answers.get(key, default)


class NoGet:
    """An object with no ``get`` attribute at all."""


# ── The watch-playlist kwargs assembly ─────────────────────────────────


class TestWatchPlaylistKwargs:
    def test_all_four_members_in_insertion_order(self):
        kwargs = _watch_playlist_kwargs("v1", "PL1", 25, True)
        assert kwargs == {"limit": 25, "videoId": "v1", "playlistId": "PL1", "radio": True}
        assert list(kwargs) == ["limit", "videoId", "playlistId", "radio"]

    def test_none_ids_leave_their_members_out(self):
        assert _watch_playlist_kwargs(None, None, 25, False) == {"limit": 25}

    def test_falsy_but_not_none_ids_still_travel(self):
        kwargs = _watch_playlist_kwargs("", "", 0, False)
        assert kwargs == {"limit": 0, "videoId": "", "playlistId": ""}

    def test_radio_member_is_the_literal_true_never_the_flag(self):
        assert _watch_playlist_kwargs("v", None, 25, "yes")["radio"] is True
        assert _watch_playlist_kwargs("v", None, 25, 1)["radio"] is True
        assert "radio" not in _watch_playlist_kwargs("v", None, 25, "")
        assert "radio" not in _watch_playlist_kwargs("v", None, 25, 0)
        assert "radio" not in _watch_playlist_kwargs("v", None, 25, None)
        assert "radio" not in _watch_playlist_kwargs("v", None, 25, [])

    def test_values_pass_through_as_the_objects_they_are(self):
        limit = object()
        video_id = object()
        playlist_id = object()
        kwargs = _watch_playlist_kwargs(video_id, playlist_id, limit, False)
        assert kwargs["limit"] is limit
        assert kwargs["videoId"] is video_id
        assert kwargs["playlistId"] is playlist_id

    def test_fresh_dict_every_call(self):
        assert _watch_playlist_kwargs(None, None, 25, False) is not (
            _watch_playlist_kwargs(None, None, 25, False)
        )


# ── The result-tracks extraction ──────────────────────────────────────


class TestResultTracks:
    def test_the_member_list_passes_through_as_it_is(self):
        tracks = [{"videoId": "v1"}]
        assert _result_tracks({"tracks": tracks}) is tracks

    def test_missing_member_answers_a_fresh_empty_list(self):
        result = _result_tracks({"other": 1})
        assert result == []
        assert result is not _result_tracks({"other": 1})

    def test_non_dict_answers_a_fresh_empty_list(self):
        for payload in ("str", None, 5, [], [{"tracks": []}]):
            result = _result_tracks(payload)
            assert result == []
            assert result is not _result_tracks(payload)

    def test_a_mapping_without_get_never_raises(self):
        # The isinstance gate short-circuits before any get call — a
        # non-dict mapping answers [] without touching its methods.
        class Mapping:
            def get(self, key, default=None):
                raise AssertionError("never called")

        assert _result_tracks(Mapping()) == []

    def test_dict_subclasses_pass_the_gate(self):
        class Sub(dict):
            pass

        tracks = [1, 2]
        assert _result_tracks(Sub(tracks=tracks)) is tracks


# ── The list-gated playlist-tracks extraction ─────────────────────────


class TestPlaylistTracks:
    def test_the_member_list_passes_through_as_it_is(self):
        tracks = [{"videoId": "v1"}]
        assert _playlist_tracks({"tracks": tracks}) is tracks

    def test_falsy_members_answer_a_fresh_empty_list(self):
        for member in (None, [], 0, ""):
            assert _playlist_tracks({"tracks": member}) == []

    def test_non_list_truthy_members_answer_a_fresh_empty_list(self):
        class Gen:
            def __iter__(self):
                yield 1

        for member in ("tracks", {"a": 1}, Gen()):
            assert _playlist_tracks({"tracks": member}) == []

    def test_non_dict_answers_a_fresh_empty_list(self):
        for payload in ("str", None, 5, []):
            assert _playlist_tracks(payload) == []

    def test_list_subclasses_pass_the_gate(self):
        class Sub(list):
            pass

        tracks = Sub([1, 2])
        assert _playlist_tracks({"tracks": tracks}) is tracks

    def test_dict_subclasses_pass_the_dict_gate(self):
        class Sub(dict):
            pass

        assert _playlist_tracks(Sub(tracks=[1])) == [1]


# ── The radio seed pool ───────────────────────────────────────────────


class TestRadioSeedPool:
    def test_first_occurrence_wins_in_first_sight_order(self):
        t1, t2, t3 = {"videoId": "a"}, {"videoId": "a"}, {"videoId": "b"}
        pool = _radio_seed_pool([[t1, t2], [t3]])
        assert pool == [t1, t3]
        assert pool[0] is t1
        assert pool[1] is t3

    def test_video_id_falls_back_when_video_id_key_is_missing_or_falsy(self):
        kept = _radio_seed_pool(
            [
                [
                    {"videoId": None, "video_id": "x"},
                    {"videoId": "", "video_id": "y"},
                    {"video_id": "z"},
                ]
            ]
        )
        assert [t["video_id"] for t in kept] == ["x", "y", "z"]

    def test_falsy_ids_drop_the_track(self):
        assert _radio_seed_pool([[{"videoId": None, "video_id": ""}, {"videoId": ""}, {}]]) == []

    def test_the_second_read_happens_only_when_the_first_answers_falsy(self):
        log: list[str] = []
        first = LogGet({"videoId": "a"}, log)
        second = LogGet({}, log)
        third = LogGet({"videoId": 0, "video_id": "b"}, log)
        assert _radio_seed_pool([[first, second, third]]) == [first, third]
        assert log == ["videoId", "videoId", "video_id", "videoId", "video_id"]

    def test_a_track_without_get_raises_the_attribute_error(self):
        with pytest.raises(AttributeError):
            _radio_seed_pool([[NoGet()]])

    def test_non_iterable_results_raise_typeerror(self):
        with pytest.raises(TypeError):
            _radio_seed_pool([5])

    def test_a_string_results_iterates_its_characters(self):
        # CPython iterates the str; each char's .get raises.
        with pytest.raises(AttributeError):
            _radio_seed_pool(["ab"])

    def test_unhashable_truthy_id_raises_typeerror(self):
        with pytest.raises(TypeError):
            _radio_seed_pool([[{"videoId": [1]}]])

    def test_a_raising_get_keeps_its_own_error(self):
        log: list[str] = []
        track = LogGet({}, log, boom=("videoId",))
        with pytest.raises(RuntimeError, match="boom"):
            _radio_seed_pool([[track]])

    def test_generator_results_are_consumed(self):
        def seeds():
            yield [{"videoId": "a"}]
            yield [{"videoId": "b"}]

        assert _radio_seed_pool(seeds()) == [{"videoId": "a"}, {"videoId": "b"}]

    def test_empty_shapes(self):
        assert _radio_seed_pool([]) == []
        assert _radio_seed_pool([[], []]) == []


# ── The discovery source rotation ──────────────────────────────────────


class TestDiscoverySourceOrder:
    def test_zero_starts_at_one(self):
        assert _discovery_source_order(0) == [1, 2, 3, 4, 5, 6]

    def test_every_offset_rotates_by_one(self):
        assert _discovery_source_order(1) == [2, 3, 4, 5, 6, 1]
        assert _discovery_source_order(5) == [6, 1, 2, 3, 4, 5]

    def test_the_rotation_wraps_past_six(self):
        assert _discovery_source_order(6) == [1, 2, 3, 4, 5, 6]
        assert _discovery_source_order(11) == [6, 1, 2, 3, 4, 5]

    def test_negative_last_source_uses_pythons_modulo(self):
        assert _discovery_source_order(-1) == [6, 1, 2, 3, 4, 5]
        assert _discovery_source_order(-7) == [6, 1, 2, 3, 4, 5]

    def test_bools_and_floats_steer_through_the_operators(self):
        assert _discovery_source_order(True) == [2, 3, 4, 5, 6, 1]
        assert _discovery_source_order(False) == [1, 2, 3, 4, 5, 6]
        assert _discovery_source_order(0.0) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


# ── The chart-shelf assembly and its round-robin pick ──────────────────


class TestChartShelves:
    def test_empty_payload_answers_no_shelves(self):
        assert _chart_shelves({}) == []

    def test_the_three_keys_keep_their_order(self):
        daily, weekly, videos = {"playlistId": "d1"}, {"playlistId": "w1"}, {"playlistId": "v1"}
        shelves = _chart_shelves({"daily": [daily], "weekly": [weekly], "videos": [videos]})
        assert shelves == [daily, weekly, videos]
        assert shelves[0] is daily

    def test_non_list_members_are_skipped(self):
        for member in (None, "x", {"a": 1}, 5):
            assert _chart_shelves({"daily": member}) == []

    def test_non_dict_shelf_items_are_dropped(self):
        assert _chart_shelves({"daily": [{"a": 1}, 5, None, "x"]}) == [{"a": 1}]

    def test_dict_and_list_subclasses_pass_their_gates(self):
        class SubDict(dict):
            pass

        class SubList(list):
            pass

        shelf = SubDict(a=1)
        assert _chart_shelves({"daily": SubList([shelf])}) == [shelf]

    def test_a_payload_without_get_raises_the_attribute_error(self):
        with pytest.raises(AttributeError):
            _chart_shelves(NoGet())

    def test_generator_members_fail_the_list_gate(self):
        assert _chart_shelves({"daily": iter([])}) == []


class TestChartShelfIndex:
    def test_the_round_robin_steps_by_one(self):
        assert _chart_shelf_index(0, 4) == 1
        assert _chart_shelf_index(3, 4) == 0
        assert _chart_shelf_index(2, 1) == 0

    def test_zero_count_raises_the_modulos_own_error(self):
        # CPython's own wording for the message differs across the
        # supported range (3.10 and 3.15 spell it "integer division or
        # modulo by zero", 3.12 "integer modulo by zero"); the contract
        # is the modulo's own ZeroDivisionError surfacing verbatim, so
        # both historical spellings match.
        with pytest.raises(ZeroDivisionError, match=r"integer (?:division or )?modulo by zero"):
            _chart_shelf_index(2, 0)

    def test_floats_and_bools_steer_through_the_operators(self):
        assert _chart_shelf_index(2.5, 4.0) == 3.5
        assert _chart_shelf_index(True, 2) == 0


# ── The playable filters ──────────────────────────────────────────────


class TestDiscoveryPlayableTracks:
    def test_only_dicts_with_a_truthy_video_id_are_kept(self):
        kept = {"videoId": "a"}
        dropped = [{"videoId": ""}, {"videoId": None}, {"videoId": 0}, {}, 5, None, "s", []]
        assert _discovery_playable_tracks([kept, *dropped]) == [kept]

    def test_kept_tracks_pass_through_in_order(self):
        t1, t2 = {"videoId": "a"}, {"videoId": "b"}
        assert _discovery_playable_tracks([t1, t2]) == [t1, t2]

    def test_non_dicts_never_have_their_get_called(self):
        log: list[str] = []
        stub = LogGet({"videoId": "a"}, log)  # not a dict: the gate drops it first
        assert _discovery_playable_tracks([stub]) == []
        assert log == []

    def test_dict_subclasses_pass_the_gate(self):
        class Sub(dict):
            pass

        assert _discovery_playable_tracks([Sub(videoId="a")]) == [Sub(videoId="a")]

    def test_a_raising_get_keeps_its_own_error(self):
        with pytest.raises(RuntimeError, match="boom"):
            _discovery_playable_tracks([BoomGetDict()])

    def test_non_iterable_items_raise_typeerror(self):
        with pytest.raises(TypeError):
            _discovery_playable_tracks(5)

    def test_generator_items_are_consumed(self):
        def items():
            yield {"videoId": "a"}
            yield {"videoId": ""}

        assert _discovery_playable_tracks(items()) == [{"videoId": "a"}]


class TestChartPlayableTracks:
    def test_both_id_keys_are_accepted(self):
        kept = [{"videoId": "a"}, {"video_id": "b"}, {"videoId": None, "video_id": "c"}]
        dropped = [{"videoId": ""}, {"video_id": ""}, {}, 5, "s"]
        assert _chart_playable_tracks([*kept, *dropped]) == kept

    def test_the_second_read_happens_only_when_the_first_answers_falsy(self):
        log: list[str] = []
        first = LogGetDict({"videoId": "a"}, log)
        second = LogGetDict({}, log)
        third = LogGetDict({"videoId": 0, "video_id": "b"}, log)
        assert _chart_playable_tracks([first, second, third]) == [first, third]
        assert log == ["videoId", "videoId", "video_id", "videoId", "video_id"]

    def test_non_dicts_are_dropped_before_any_get(self):
        stub = LogGet({"videoId": "a"}, [])
        assert _chart_playable_tracks([stub]) == []

    def test_a_raising_get_keeps_its_own_error(self):
        with pytest.raises(RuntimeError, match="boom"):
            _chart_playable_tracks([BoomGetDict()])


# ── The explore new-releases extraction ───────────────────────────────


class TestExploreNewReleases:
    def test_the_member_list_passes_through_as_it_is(self):
        releases = [{"title": "Fresh"}]
        assert _explore_new_releases({"new_releases": releases}) is releases

    def test_non_list_members_answer_a_fresh_empty_list(self):
        for member in (None, "x", {"a": 1}, [], 0):
            assert _explore_new_releases({"new_releases": member}) == []

    def test_non_dicts_answer_a_fresh_empty_list(self):
        for payload in ("str", None, 5, [1]):
            assert _explore_new_releases(payload) == []

    def test_dict_subclasses_pass_both_gates(self):
        class Sub(dict):
            pass

        class SubList(list):
            pass

        releases = SubList([1])
        assert _explore_new_releases(Sub(new_releases=releases)) is releases


# ── The playlist sort-order lookup ────────────────────────────────────


class TestPlaylistOrderParams:
    def test_the_three_order_names_answer_their_tokens(self):
        assert _playlist_order_params("a_to_z") == "ggMGKgQIARAA"
        assert _playlist_order_params("z_to_a") == "ggMGKgQIARAB"
        assert _playlist_order_params("recently_added") == "ggMGKgQIABAB"

    def test_unknown_and_falsy_orders_answer_none(self):
        for order in (None, "", 0, False, "A_TO_Z", "bogus", "recently_added "):
            assert _playlist_order_params(order) is None

    def test_a_patched_table_steers_whichever_backend_is_loaded(self):
        original = YTMusicService._ORDER_PARAMS
        try:
            YTMusicService._ORDER_PARAMS = {"custom": "custom_params"}
            assert _playlist_order_params("custom") == "custom_params"
            assert _playlist_order_params("a_to_z") is None
        finally:
            YTMusicService._ORDER_PARAMS = original


# ── Native parity ──────────────────────────────────────────────────────
# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable: the duck-typed shapes are
# spelled out as specs both sides build identically (the "logget" shape
# records its own reads into a shared log both sides render), and the
# outcomes are rendered back through a canonical walk (objects by their
# class tag, the logget shape by its log) so the two backends' answers
# compare case for case — error messages included.

_PARITY_CASES = [
    # ── _watch_playlist_kwargs ──
    {"engine": "watch_kwargs", "args": [None, None, 25, False]},
    {"engine": "watch_kwargs", "args": ["v1", "PL1", 25, True]},
    {"engine": "watch_kwargs", "args": ["", "", 0, 1]},
    {"engine": "watch_kwargs", "args": [None, "PL", None, ""]},
    {"engine": "watch_kwargs", "args": [[1, 2], {"a": 1}, 25, 0]},
    {"engine": "watch_kwargs", "args": [None, None, 25, "surprise"]},
    # ── _result_tracks ──
    {"engine": "result_tracks", "args": [{"tracks": [1, 2]}]},
    {"engine": "result_tracks", "args": [{"tracks": {}}]},
    {"engine": "result_tracks", "args": [{"other": 1}]},
    {"engine": "result_tracks", "args": ["str"]},
    {"engine": "result_tracks", "args": [None]},
    {"engine": "result_tracks", "args": [5]},
    {"engine": "result_tracks", "args": [[]]},
    {"engine": "result_tracks", "args": [{"t": "dictsub", "data": {"tracks": [1]}}]},
    {"engine": "result_tracks", "args": [{"t": "loggetdict"}]},
    # ── _playlist_tracks ──
    {"engine": "playlist_tracks", "args": [{"tracks": [1, 2]}]},
    {"engine": "playlist_tracks", "args": [{"tracks": []}]},
    {"engine": "playlist_tracks", "args": [{"tracks": None}]},
    {"engine": "playlist_tracks", "args": [{"tracks": "s"}]},
    {"engine": "playlist_tracks", "args": [{"tracks": {"a": 1}}]},
    {"engine": "playlist_tracks", "args": [{"tracks": {"t": "listsub", "items": [1]}}]},
    {"engine": "playlist_tracks", "args": [{}]},
    {"engine": "playlist_tracks", "args": ["x"]},
    {"engine": "playlist_tracks", "args": [5]},
    {"engine": "playlist_tracks", "args": [{"t": "dictsub", "data": {"tracks": [7]}}]},
    # ── _radio_seed_pool ──
    {"engine": "radio_seed_pool", "args": [[[]]]},
    {"engine": "radio_seed_pool", "args": [[[], []]]},
    {
        "engine": "radio_seed_pool",
        "args": [[[{"videoId": "a"}], [{"videoId": "a"}, {"videoId": "b"}]]],
    },
    {
        "engine": "radio_seed_pool",
        "args": [
            [
                [
                    {"videoId": None, "video_id": "x"},
                    {"videoId": "", "video_id": "y"},
                    {"video_id": "z"},
                    {"videoId": 0, "video_id": ""},
                    {},
                ]
            ]
        ],
    },
    {"engine": "radio_seed_pool", "args": [[[{"videoId": [1]}]]]},
    {"engine": "radio_seed_pool", "args": [[[{"t": "logget", "answers": {"videoId": "a"}}]]]},
    {
        "engine": "radio_seed_pool",
        "args": [
            [
                [
                    {"t": "logget", "answers": {"videoId": None, "video_id": "b"}},
                    {"t": "logget", "answers": {"videoId": "c"}},
                ]
            ]
        ],
    },
    {"engine": "radio_seed_pool", "args": [[[{"t": "noget"}]]]},
    {"engine": "radio_seed_pool", "args": [[5]]},
    {"engine": "radio_seed_pool", "args": ["ab"]},
    {
        "engine": "radio_seed_pool",
        "args": [{"t": "gen", "items": [[{"videoId": "a"}], [{"videoId": "b"}]]}],
    },
    {"engine": "radio_seed_pool", "args": [[[{"t": "boomget"}]]]},
    # ── _discovery_source_order ──
    {"engine": "discovery_source_order", "args": [0]},
    {"engine": "discovery_source_order", "args": [1]},
    {"engine": "discovery_source_order", "args": [5]},
    {"engine": "discovery_source_order", "args": [6]},
    {"engine": "discovery_source_order", "args": [11]},
    {"engine": "discovery_source_order", "args": [-1]},
    {"engine": "discovery_source_order", "args": [-7]},
    {"engine": "discovery_source_order", "args": [1.5]},
    {"engine": "discovery_source_order", "args": [True]},
    {"engine": "discovery_source_order", "args": [False]},
    # ── _chart_shelves ──
    {"engine": "chart_shelves", "args": [{}]},
    {"engine": "chart_shelves", "args": [{"daily": [{"playlistId": "d"}]}]},
    {
        "engine": "chart_shelves",
        "args": [
            {
                "daily": [{"a": 1}, 5],
                "weekly": None,
                "videos": [{"v": 1}, "x"],
                "other": [{"never": 1}],
            }
        ],
    },
    {"engine": "chart_shelves", "args": [{"daily": "x", "weekly": {"a": 1}, "videos": 5}]},
    {"engine": "chart_shelves", "args": [{"daily": {"t": "gen", "items": [{"g": 1}]}}]},
    {"engine": "chart_shelves", "args": [{"t": "chartslogget"}]},
    {"engine": "chart_shelves", "args": [{"t": "noget"}]},
    {"engine": "chart_shelves", "args": [{"daily": {"t": "listsub", "items": [{"s": 1}]}}]},
    # ── _chart_shelf_index ──
    {"engine": "chart_shelf_index", "args": [0, 4]},
    {"engine": "chart_shelf_index", "args": [3, 4]},
    {"engine": "chart_shelf_index", "args": [5, 4]},
    {"engine": "chart_shelf_index", "args": [2, 1]},
    {"engine": "chart_shelf_index", "args": [2, 0]},
    {"engine": "chart_shelf_index", "args": [2.5, 4.0]},
    {"engine": "chart_shelf_index", "args": [True, 2]},
    # ── _discovery_playable_tracks ──
    {
        "engine": "discovery_playable_tracks",
        "args": [
            [
                {"videoId": "a"},
                {"videoId": ""},
                {"videoId": None},
                {"videoId": 0},
                {},
                5,
                None,
                "s",
                [],
            ]
        ],
    },
    {"engine": "discovery_playable_tracks", "args": [[]]},
    {"engine": "discovery_playable_tracks", "args": [5]},
    {
        "engine": "discovery_playable_tracks",
        "args": [{"t": "gen", "items": [{"videoId": "a"}, {"videoId": ""}]}],
    },
    {
        "engine": "discovery_playable_tracks",
        "args": [[{"t": "logget", "answers": {"videoId": "a"}}]],
    },
    {"engine": "discovery_playable_tracks", "args": [[{"t": "boomget"}]]},
    # ── _chart_playable_tracks ──
    {
        "engine": "chart_playable_tracks",
        "args": [
            [
                {"videoId": "a"},
                {"videoId": None, "video_id": "b"},
                {"video_id": "c"},
                {"videoId": "", "video_id": ""},
                {},
                5,
                "s",
            ]
        ],
    },
    {
        "engine": "chart_playable_tracks",
        "args": [
            [
                {"t": "logget", "answers": {"videoId": None, "video_id": "b"}},
                {"t": "logget", "answers": {"videoId": "c"}},
            ]
        ],
    },
    {"engine": "chart_playable_tracks", "args": [[[{"t": "noget"}]]]},
    {"engine": "chart_playable_tracks", "args": [[[{"t": "boomget"}]]]},
    {"engine": "chart_playable_tracks", "args": [[5]]},
    # ── _explore_new_releases ──
    {"engine": "explore_new_releases", "args": [{"new_releases": [1, 2]}]},
    {"engine": "explore_new_releases", "args": [{"new_releases": "x"}]},
    {"engine": "explore_new_releases", "args": [{"new_releases": {}}]},
    {"engine": "explore_new_releases", "args": [{"new_releases": {"t": "listsub", "items": [1]}}]},
    {"engine": "explore_new_releases", "args": [{"other": []}]},
    {"engine": "explore_new_releases", "args": ["str"]},
    {"engine": "explore_new_releases", "args": [None]},
    {"engine": "explore_new_releases", "args": [{"t": "dictsub", "data": {"new_releases": [9]}}]},
    # ── _playlist_order_params ──
    {"engine": "playlist_order_params", "args": ["a_to_z"]},
    {"engine": "playlist_order_params", "args": ["z_to_a"]},
    {"engine": "playlist_order_params", "args": ["recently_added"]},
    {"engine": "playlist_order_params", "args": [None]},
    {"engine": "playlist_order_params", "args": [""]},
    {"engine": "playlist_order_params", "args": ["bogus"]},
    {"engine": "playlist_order_params", "args": [0]},
    {"engine": "playlist_order_params", "args": [True]},
]


_KNOWN_KINDS = {
    "dictsub",
    "listsub",
    "gen",
    "logget",
    "boomget",
    "chartslogget",
    "noget",
    "loggetdict",
}


def _build(spec, log):
    """Build the shape a spec spells — identically in this process and
    the child interpreter — recursing into plain containers so nested
    members can carry duck-typed shapes too."""
    if isinstance(spec, list):
        return [_build(item, log) for item in spec]
    if not isinstance(spec, dict):
        return spec
    kind = spec.get("t")
    if kind == "dictsub":
        return {k: _build(v, log) for k, v in spec["data"].items()}
    if kind == "listsub":
        return [_build(v, log) for v in spec["items"]]
    if kind == "gen":
        return iter([_build(v, log) for v in spec["items"]])
    if kind == "logget":

        class LogGet:
            def __init__(self, answers, log):
                self._answers = answers
                self._log = log

            def get(self, key, default=None):
                self._log.append(key)
                return self._answers.get(key, default)

        return LogGet(dict(spec["answers"]), log)
    if kind == "boomget":

        class BoomGet:
            def get(self, key, default=None):
                raise RuntimeError("boom")

        return BoomGet()
    if kind == "chartslogget":

        class ChartsLogGet:
            def __init__(self, log):
                self._log = log

            def get(self, key, default=None):
                self._log.append(key)
                if key == "daily":
                    return [{"d": 1}]
                if key == "weekly":
                    return []
                return None

        return ChartsLogGet(log)
    if kind == "noget":
        return object()
    if kind == "loggetdict":

        class LogGetDict(dict):
            def __init__(self, log):
                super().__init__(tracks=[1])
                self._log = log

            def get(self, key, default=None):
                self._log.append(key)
                return super().get(key, default)

        return LogGetDict(log)
    if kind in _KNOWN_KINDS:
        raise AssertionError(kind)
    return {k: _build(v, log) for k, v in spec.items()}


def _render(value):
    """Render an outcome canonically — JSON shapes by value, the
    duck-typed shapes by their log or class tag."""
    if isinstance(value, dict):
        if hasattr(value, "_log"):
            return {"__logget__": True}
        return {str(k): _render(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_render(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"__obj__": type(value).__name__}


def _outcome(case: dict) -> dict:
    log: list = []
    args = [_build(spec, log) for spec in case["args"]]
    engine = {
        "watch_kwargs": _watch_playlist_kwargs,
        "result_tracks": _result_tracks,
        "playlist_tracks": _playlist_tracks,
        "radio_seed_pool": _radio_seed_pool,
        "discovery_source_order": _discovery_source_order,
        "chart_shelves": _chart_shelves,
        "chart_shelf_index": _chart_shelf_index,
        "discovery_playable_tracks": _discovery_playable_tracks,
        "chart_playable_tracks": _chart_playable_tracks,
        "explore_new_releases": _explore_new_releases,
        "playlist_order_params": _playlist_order_params,
    }[case["engine"]]
    try:
        return {"ok": _render(engine(*args)), "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


# The child interpreter runs this program with the case matrix on the
# stdin and prints the reference engines' outcomes as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.ytmusic import (
    _chart_playable_tracks,
    _chart_shelf_index,
    _chart_shelves,
    _discovery_playable_tracks,
    _discovery_source_order,
    _explore_new_releases,
    _playlist_order_params,
    _playlist_tracks,
    _radio_seed_pool,
    _result_tracks,
    _watch_playlist_kwargs,
)


def build(spec, log):
    if isinstance(spec, list):
        return [build(item, log) for item in spec]
    if not isinstance(spec, dict):
        return spec
    kind = spec.get('t')
    if kind == 'dictsub':
        return {k: build(v, log) for k, v in spec['data'].items()}
    if kind == 'listsub':
        return [build(v, log) for v in spec['items']]
    if kind == 'gen':
        return iter([build(v, log) for v in spec['items']])
    if kind == 'logget':
        class LogGet:
            def __init__(self, answers, log):
                self._answers = answers
                self._log = log

            def get(self, key, default=None):
                self._log.append(key)
                return self._answers.get(key, default)

        return LogGet(dict(spec['answers']), log)
    if kind == 'boomget':
        class BoomGet:
            def get(self, key, default=None):
                raise RuntimeError('boom')

        return BoomGet()
    if kind == 'chartslogget':
        class ChartsLogGet:
            def __init__(self, log):
                self._log = log

            def get(self, key, default=None):
                self._log.append(key)
                if key == 'daily':
                    return [{'d': 1}]
                if key == 'weekly':
                    return []
                return None

        return ChartsLogGet(log)
    if kind == 'noget':
        return object()
    if kind == 'loggetdict':
        class LogGetDict(dict):
            def __init__(self, log):
                super().__init__(tracks=[1])
                self._log = log

            def get(self, key, default=None):
                self._log.append(key)
                return super().get(key, default)

        return LogGetDict(log)
    if kind in {'dictsub', 'listsub', 'gen', 'logget', 'boomget', 'chartslogget', 'noget', 'loggetdict'}:
        raise AssertionError(kind)
    return {k: build(v, log) for k, v in spec.items()}


def render(value):
    if isinstance(value, dict):
        if hasattr(value, '_log'):
            return {'__logget__': True}
        return {str(k): render(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [render(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {'__obj__': type(value).__name__}


ENGINES = {
    'watch_kwargs': _watch_playlist_kwargs,
    'result_tracks': _result_tracks,
    'playlist_tracks': _playlist_tracks,
    'radio_seed_pool': _radio_seed_pool,
    'discovery_source_order': _discovery_source_order,
    'chart_shelves': _chart_shelves,
    'chart_shelf_index': _chart_shelf_index,
    'discovery_playable_tracks': _discovery_playable_tracks,
    'chart_playable_tracks': _chart_playable_tracks,
    'explore_new_releases': _explore_new_releases,
    'playlist_order_params': _playlist_order_params,
}


def outcome(case):
    log = []
    args = [build(spec, log) for spec in case['args']]
    try:
        return {'ok': render(ENGINES[case['engine']](*args)), 'log': log}
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'log': log}


payload = json.load(sys.stdin)
json.dump([outcome(case) for case in payload], sys.stdout)
"""


def _fallback_outcomes() -> list:
    """Run the pure-Python reference engines over the parity cases in
    a child interpreter with the native extension import blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(_PARITY_CASES),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _native_outcomes() -> list:
    return [_outcome(case) for case in _PARITY_CASES]


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references run in a child interpreter
    with the extension import blocked — same gates, same read orders
    (the logget shapes' logs included), same pass-throughs and same
    errors with the same messages — for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _native_outcomes()
        fallback = _fallback_outcomes()
        assert len(native) == len(fallback)
        for case, n, f in zip(_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
