"""Tests for the session-state validation engine.

``coerce_session_state`` is the engine behind
``SessionMixin._restore_session_state``: the one-pass validation and
coercion of the raw parsed session.json object — every isinstance gate,
clamp and fallback decided before any app state is touched. The
pure-Python reference lives in ``app/_session.py``; when the Rust
extension is built the same engine runs natively
(``_ytm_player_native.coerce_session_state``). ``find_track_index`` is
the resume-point scan over the restored queue, with the same dual
implementation.

The unit tests below assert the engine's contract on whichever backend is
loaded. When the extension is built, the parity tests additionally compare
the native engine against the pure-Python reference (run in a child
interpreter with the extension import blocked) case for case, so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.app._session import coerce_session_state, find_track_index

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestVolume:
    def test_absent_key_clamps_the_default(self):
        assert coerce_session_state({}, 80)["volume"] == 80

    def test_absent_key_clamps_an_out_of_range_default(self):
        # The .get() default runs through the same isinstance gate, so an
        # absent key with a huge configured default clamps to 100.
        assert coerce_session_state({}, 300)["volume"] == 100

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            (50, 50),
            (0, 0),
            (100, 100),
            (101, 100),
            (500, 100),
            (-1, 0),
            (-500, 0),
            (79.9, 79),
            (100.7, 100),
            (-0.5, 0),
            (10**30, 100),
            (-(10**30), 0),
        ],
    )
    def test_numeric_values_clamp(self, stored, expected):
        assert coerce_session_state({"volume": stored}, 80)["volume"] == expected

    @pytest.mark.parametrize(
        "stored", [True, False, "80", "", None, [80], {"v": 80}, {"volume": 80}]
    )
    def test_invalid_values_fall_back_to_the_raw_default(self, stored):
        # The raw default is returned unclamped — only values that pass
        # the isinstance gate get clamped (faithful to the original).
        assert coerce_session_state({"volume": stored}, 250)["volume"] == 250

    def test_float_nan_clamps_to_100(self):
        # min(100, nan) is 100 in Python (NaN comparisons are false), so a
        # NaN volume clamps to 100 rather than poisoning the clamp.
        assert coerce_session_state({"volume": float("nan")}, 80)["volume"] == 100

    @pytest.mark.parametrize("stored", [float("inf"), float("-inf")])
    def test_float_infinities_clamp(self, stored):
        expected = 100 if stored > 0 else 0
        assert coerce_session_state({"volume": stored}, 80)["volume"] == expected


class TestRepeat:
    @pytest.mark.parametrize("value", ["off", "all", "one"])
    def test_valid_members_pass_through(self, value):
        assert coerce_session_state({"repeat": value}, 80)["repeat"] == value

    @pytest.mark.parametrize(
        "value", ["OFF", "bogus", "", None, 5, True, False, 1.0, [1], {"x": 1}]
    )
    def test_invalid_values_fall_back_to_off(self, value):
        assert coerce_session_state({"repeat": value}, 80)["repeat"] == "off"

    def test_absent_key_defaults_to_off(self):
        assert coerce_session_state({}, 80)["repeat"] == "off"


class TestQueue:
    def test_absent_key(self):
        c = coerce_session_state({}, 80)
        assert c["queue_tracks"] == []
        assert c["queue_index"] is None

    def test_empty_list(self):
        c = coerce_session_state({"queue_tracks": []}, 80)
        assert c["queue_tracks"] == []
        assert c["queue_index"] is None

    @pytest.mark.parametrize("stored", [{}, "x", 5, None, True])
    def test_non_list_values_are_ignored(self, stored):
        c = coerce_session_state({"queue_tracks": stored}, 80)
        assert c["queue_tracks"] == []
        assert c["queue_index"] is None

    def test_tracks_are_normalized(self):
        c = coerce_session_state(
            {"queue_tracks": [{"videoId": "abc", "title": "X"}, {"title": "no id"}]}, 80
        )
        assert c["queue_tracks"] == [
            {
                "video_id": "abc",
                "title": "X",
                "artist": "Unknown",
                "artists": [],
                "album": "",
                "album_id": None,
                "duration": None,
                "thumbnail_url": None,
                "is_video": False,
                "likeStatus": None,
                "setVideoId": None,
            }
        ]

    def test_absent_index_validates_as_zero_when_tracks_exist(self):
        c = coerce_session_state({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}]}, 80)
        assert c["queue_index"] == 0

    @pytest.mark.parametrize(
        ("index", "expected"),
        [
            (0, 0),
            (1, 1),
            (2, None),
            (-1, None),
            ("1", None),
            (None, None),
            (1.0, None),
            ([1], None),
        ],
    )
    def test_index_validation(self, index, expected):
        state = {"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": index}
        assert coerce_session_state(state, 80)["queue_index"] == expected

    def test_bool_index_counts_as_int(self):
        # bool is an int subclass in Python, so True validates as index 1 —
        # a faithful transcription of the original isinstance check.
        state = {"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": True}
        assert coerce_session_state(state, 80)["queue_index"] == 1

    def test_index_range_uses_the_normalized_length(self):
        # The second entry has no video id and is dropped by
        # normalize_tracks, so index 1 no longer fits.
        state = {"queue_tracks": [{"videoId": "a"}, {"title": "no id"}], "queue_index": 1}
        assert coerce_session_state(state, 80)["queue_index"] is None

    def test_all_entries_dropped_yields_empty_list(self):
        c = coerce_session_state({"queue_tracks": [{"title": "no id"}]}, 80)
        assert c["queue_tracks"] == []
        assert c["queue_index"] is None


class TestScalars:
    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            (True, True),
            (1, True),
            ("yes", True),
            (False, False),
            (0, False),
            ("", False),
            (None, False),
        ],
    )
    def test_shuffle_truthiness(self, stored, expected):
        assert coerce_session_state({"shuffle": stored}, 80)["shuffle"] is expected

    def test_shuffle_absent_defaults_false(self):
        assert coerce_session_state({}, 80)["shuffle"] is False

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            ("PL123", "PL123"),
            ("", None),
            (None, None),
            (5, None),
            (True, None),
            ([], None),
        ],
    )
    def test_queue_context_id(self, stored, expected):
        assert coerce_session_state({"queue_context_id": stored}, 80)["queue_context_id"] == (
            expected
        )

    def test_sidebar_dict_is_returned_by_reference(self):
        sidebar = {"library": 0}
        c = coerce_session_state({"sidebar_per_page": sidebar}, 80)
        assert c["sidebar_per_page"] is sidebar

    @pytest.mark.parametrize("stored", [{}, None, "x", 5, []])
    def test_invalid_sidebar_is_none(self, stored):
        assert coerce_session_state({"sidebar_per_page": stored}, 80)["sidebar_per_page"] is None

    @pytest.mark.parametrize(
        ("stored", "expected"),
        [
            (True, True),
            ("false", True),  # any non-empty string is truthy — faithful wart
            (1, True),
            (False, False),
            (0, False),
            ("", False),
            (None, False),
            ([], False),
        ],
    )
    def test_hint_flag_truthiness(self, stored, expected):
        c = coerce_session_state({"first_run_hint_shown": stored, "mpris_hint_shown": stored}, 80)
        assert c["first_run_hint_shown"] is expected
        assert c["mpris_hint_shown"] is expected

    def test_hint_flags_absent_default_false(self):
        c = coerce_session_state({}, 80)
        assert c["first_run_hint_shown"] is False
        assert c["mpris_hint_shown"] is False

    def test_transliteration_absent(self):
        c = coerce_session_state({}, 80)
        assert c["has_transliteration"] is False
        assert c["transliteration_enabled"] is None

    def test_transliteration_raw_value_passes_through(self):
        # The stored value is assigned unconverted (faithful to the
        # original), so even a non-bool survives verbatim.
        c = coerce_session_state({"transliteration_enabled": "legacy"}, 80)
        assert c["has_transliteration"] is True
        assert c["transliteration_enabled"] == "legacy"

    def test_transliteration_false_value_is_present(self):
        c = coerce_session_state({"transliteration_enabled": False}, 80)
        assert c["has_transliteration"] is True
        assert c["transliteration_enabled"] is False


class TestResume:
    def test_valid_resume_is_returned_by_reference(self):
        resume = {"video_id": "abc", "position": 42.5, "playlist_id": "PL1"}
        c = coerce_session_state({"resume": resume}, 80)
        assert c["resume"] is resume

    @pytest.mark.parametrize(
        "stored",
        [
            None,
            "x",
            5,
            [],
            {},
            {"video_id": ""},
            {"video_id": None},
            {"position": 42.5},  # no video_id at all
        ],
    )
    def test_invalid_resume_is_none(self, stored):
        assert coerce_session_state({"resume": stored}, 80)["resume"] is None

    def test_truthy_non_string_video_id_is_kept(self):
        # The gate is plain truthiness; the scan comparison later decides.
        c = coerce_session_state({"resume": {"video_id": 5}}, 80)
        assert c["resume"] == {"video_id": 5}

    def test_absent_key_is_none(self):
        assert coerce_session_state({}, 80)["resume"] is None


class TestFindTrackIndex:
    TRACKS = [
        {"video_id": "a", "title": "A"},
        {"video_id": "b", "title": "B"},
        {"video_id": "a", "title": "A again"},
    ]

    def test_found(self):
        assert find_track_index(self.TRACKS, "b") == 1

    def test_first_match_wins(self):
        assert find_track_index(self.TRACKS, "a") == 0

    def test_not_found(self):
        assert find_track_index(self.TRACKS, "zzz") is None

    def test_empty_queue(self):
        assert find_track_index([], "a") is None

    def test_tuple_input(self):
        # QueueManager.tracks returns a tuple.
        assert find_track_index(tuple(self.TRACKS), "b") == 1

    def test_non_string_video_id_never_matches_string_ids(self):
        assert find_track_index(self.TRACKS, 5) is None

    def test_numeric_track_id_matches_numeric_lookup(self):
        assert find_track_index([{"video_id": 5}], 5) == 0

    def test_missing_video_id_key(self):
        assert find_track_index([{"title": "none"}], "a") is None


# ── Native / fallback parity ────────────────────────────────────────────
#
# Cases model session.json as json.loads produces it: dicts with str keys,
# JSON scalars. Track entries keep string ids — the normalize_tracks
# implementations' handling of non-dict rows / non-string ids is pinned by
# the formatting tests, not re-litigated here.

_PARITY_STATES = [
    # Empty / minimal states with a spread of defaults.
    ({}, 80),
    ({}, 0),
    ({}, 300),
    ({}, -5),
    # Volume: in-range, clamped, truncated, invalid, degenerate floats.
    ({"volume": 50}, 80),
    ({"volume": 0}, 80),
    ({"volume": 100}, 80),
    ({"volume": 101}, 80),
    ({"volume": -1}, 80),
    ({"volume": 79.9}, 80),
    ({"volume": -0.5}, 80),
    ({"volume": 100.7}, 80),
    ({"volume": True}, 80),
    ({"volume": False}, 80),
    ({"volume": "80"}, 80),
    ({"volume": ""}, 80),
    ({"volume": None}, 80),
    ({"volume": [80]}, 80),
    ({"volume": {"v": 80}}, 80),
    ({"volume": 10**30}, 80),
    ({"volume": -(10**30)}, 80),
    ({"volume": "bad"}, 250),  # invalid stored value → the raw default round-trips
    # Repeat: members, casing, garbage, non-strings.
    *(
        ({"repeat": value}, 80)
        for value in ["off", "all", "one", "OFF", "All", "bogus", "", None, 5, True, [1]]
    ),
    # Queue: shapes, index validation, normalization interplay.
    ({"queue_tracks": []}, 80),
    ({"queue_tracks": {}}, 80),
    ({"queue_tracks": "x"}, 80),
    ({"queue_tracks": None}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}]}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": 0}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": 1}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": 2}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": -1}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": "1"}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": None}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": True}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": 1.0}, 80),
    ({"queue_tracks": [{"videoId": "a"}, {"videoId": "b"}], "queue_index": 10**30}, 80),
    (
        {
            "queue_tracks": [
                {"videoId": "a", "title": "A", "artists": [{"name": "X", "id": "1"}]},
                {"title": "dropped"},
                {"videoId": "c", "duration": "3:30"},
            ],
            "queue_index": 2,
        },
        80,
    ),
    # Shuffle / context / sidebar / hints / transliteration.
    *(({"shuffle": v}, 80) for v in [True, False, 1, 0, "x", "", None, []]),
    *(({"queue_context_id": v}, 80) for v in ["PL1", "", None, 5, True, []]),
    *(({"sidebar_per_page": v}, 80) for v in [{"library": 0}, {}, None, "x", 5, []]),
    *(
        ({"first_run_hint_shown": v, "mpris_hint_shown": v}, 80)
        for v in [True, False, 1, 0, "false", "", None]
    ),
    ({"transliteration_enabled": True}, 80),
    ({"transliteration_enabled": False}, 80),
    ({"transliteration_enabled": "legacy"}, 80),
    ({"transliteration_enabled": None}, 80),
    # Resume gates.
    *(
        ({"resume": v}, 80)
        for v in [
            None,
            "x",
            5,
            [],
            {},
            {"video_id": ""},
            {"video_id": None},
            {"position": 42.5},
            {"video_id": "abc", "position": 42.5, "playlist_id": "PL1"},
            {"video_id": 5, "position": "12.5"},
            {"video_id": "abc", "position": "not a number"},
            {"video_id": "abc", "position": None},
            {"video_id": "abc", "position": [1]},
            {"video_id": "abc"},
        ]
    ),
    # A realistic full state, the way _save_session_state writes it.
    (
        {
            "schema_version": 1,
            "volume": 64,
            "repeat": "all",
            "shuffle": True,
            "queue_tracks": [
                {
                    "video_id": "dQw4w9WgXcQ",
                    "title": "Never Gonna Give You Up",
                    "artist": "Rick Astley",
                    "artists": [{"name": "Rick Astley", "id": "1"}],
                    "album": "Whenever You Need Somebody",
                    "album_id": "MPREb_albxzGmBOy",
                    "duration": 213,
                    "thumbnail_url": None,
                    "is_video": False,
                    "likeStatus": "INDIFFERENT",
                    "setVideoId": None,
                },
                {
                    "video_id": "abc",
                    "title": "夜に駆ける",
                    "artist": "YOASOBI",
                    "artists": [],
                    "album": "",
                    "album_id": None,
                    "duration": None,
                    "thumbnail_url": None,
                    "is_video": False,
                    "likeStatus": None,
                    "setVideoId": None,
                },
            ],
            "queue_index": 1,
            "queue_context_id": "PL123",
            "resume": {"video_id": "abc", "position": 42.5, "playlist_id": "PL123"},
            "sidebar_per_page": {"library": 0, "search": 2},
            "transliteration_enabled": True,
            "first_run_hint_shown": True,
            "mpris_hint_shown": False,
        },
        80,
    ),
]

_PARITY_SCANS = [
    ([], "a"),
    ([{"video_id": "a"}], "a"),
    ([{"video_id": "a"}], "b"),
    ([{"video_id": "a"}, {"video_id": "b"}, {"video_id": "a"}], "a"),
    ([{"video_id": "a"}, {"video_id": "b"}, {"video_id": "a"}], "b"),
    ([{"video_id": "a"}, {"video_id": "b"}], 5),
    ([{"video_id": 5}], 5),
    ([{"title": "no id"}], "a"),
    ([{"video_id": None}], "a"),
    ([{"video_id": "夜に駆ける"}], "夜に駆ける"),
]

# The child interpreter runs this program with the case list on stdin and
# prints the reference engine's outputs as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.app._session import coerce_session_state, find_track_index

payload = json.load(sys.stdin)
states = [
    coerce_session_state(state, default_volume)
    for state, default_volume in payload["states"]
]
indices = [
    find_track_index(tracks, video_id)
    for tracks, video_id in payload["scans"]
]
json.dump({"states": states, "indices": indices}, sys.stdout)
"""


def _fallback_results() -> dict:
    """Run the pure-Python reference engine over the parity cases in a
    child interpreter with the native extension import blocked."""
    payload = json.dumps({"states": _PARITY_STATES, "scans": _PARITY_SCANS})
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same gates, same clamps, same
    fallbacks — for every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        fallback = _fallback_results()
        for i, (state, default_volume) in enumerate(_PARITY_STATES):
            native = coerce_session_state(state, default_volume)
            assert native == fallback["states"][i], (
                f"native and fallback coerced state differ for {state!r} "
                f"with default_volume={default_volume!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_find_track_index_matches_fallback(self):
        fallback = _fallback_results()
        for i, (tracks, video_id) in enumerate(_PARITY_SCANS):
            native = find_track_index(tracks, video_id)
            assert native == fallback["indices"][i], (
                f"native and fallback scan results differ for "
                f"tracks={tracks!r}, video_id={video_id!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_nan_volume_matches_fallback(self):
        """NaN clamps to 100 on both backends (json.dumps writes NaN, so
        this case can't live in the JSON-shared matrix above)."""
        payload = json.dumps({"states": [[{"volume": float("nan")}, 80]], "scans": []})
        out = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(out.stdout)["states"][0]["volume"]
        native = coerce_session_state({"volume": float("nan")}, 80)["volume"]
        assert native == fallback == 100
