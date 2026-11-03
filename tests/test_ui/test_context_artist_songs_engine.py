"""Tests for the artist-songs worker-merge engines.

The decision core of the context-page worker merge —
``ContextPage._fetch_full_artist_songs``, the background worker that
swaps the artist's full song list (the playlist behind
``songs.browseId``) into the top-songs table and chains the
remaining-tracks fetch after it: ``artist_songs_first_batch`` (the
first-batch plan — the normalized batch, the swap gate, the
remaining-fetch decision behind the ``trackCount``-vs-batch-length
check, and the ``already_have`` offset) and ``artist_songs_tail``
(the append plan for the chained fetch). The pure-Python references
live in ``ui/pages/context.py``; when the Rust extension is built the
same engines run natively
(``_ytm_player_native.artist_songs_first_batch`` /
``_ytm_player_native.artist_songs_tail``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case,
so the two implementations cannot drift apart — over the reachable
shapes. The deliberate native-only degradations (an uncomparable
``trackCount`` closing the tail gate where the reference raises
``TypeError``; a truthy non-list ``tracks``/``remaining`` reading as
closed where the reference raises in the normalization) are pinned by
the native-only ``TestTotalityEngine`` instead, the same stance the
label and queue engines' suites take.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from typing import Any

import pytest
from textual.app import App, ComposeResult

from ytm_player.ui.pages.context import (
    ContextPage,
    artist_songs_first_batch,
    artist_songs_tail,
)
from ytm_player.ui.widgets.track_table import TrackTable

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _raw_track(n: int) -> dict[str, Any]:
    """One playable raw track — the ytmusicapi playlist-row shape."""
    return {
        "videoId": f"v{n}",
        "title": f"T{n}",
        "artists": [{"name": "A", "id": "UC1"}],
        "album": {"name": "Al", "id": "MPRE"},
        "duration_seconds": 100,
        "thumbnails": [],
    }


def _unplayable(n: int) -> dict[str, Any]:
    """A raw row without a video id — dropped by normalization."""
    return {"title": f"No Video {n}", "artists": [{"name": "A", "id": "UC1"}]}


def _closed_plan() -> dict[str, Any]:
    """The plan every early-return gate answers: nothing loads, no tail."""
    return {"tracks": [], "load": False, "fetch_remaining": False, "raw_count": 0}


# ── First-batch plan ─────────────────────────────────────────────────


class TestFirstBatchPlan:
    def test_a_count_above_the_batch_chains_the_tail(self) -> None:
        plan = artist_songs_first_batch({"tracks": [_raw_track(0), _raw_track(1)], "trackCount": 5})
        assert plan["load"] is True
        assert plan["fetch_remaining"] is True
        assert plan["raw_count"] == 2
        assert [t["video_id"] for t in plan["tracks"]] == ["v0", "v1"]

    def test_a_covered_playlist_closes_the_tail_gate(self) -> None:
        # trackCount == the batch length: the tail is already here.
        plan = artist_songs_first_batch({"tracks": [_raw_track(0), _raw_track(1)], "trackCount": 2})
        assert plan["load"] is True
        assert plan["fetch_remaining"] is False
        assert plan["raw_count"] == 2

    def test_a_count_below_the_batch_closes_the_tail_gate(self) -> None:
        plan = artist_songs_first_batch({"tracks": [_raw_track(0)], "trackCount": 0})
        assert plan["load"] is True
        assert plan["fetch_remaining"] is False

    def test_a_missing_count_falls_back_to_attempting_the_call(self) -> None:
        # The comment in the worker: "If trackCount isn't exposed, fall
        # back to attempting the call and treating empty/error as
        # 'nothing more to fetch'."
        plan = artist_songs_first_batch({"tracks": [_raw_track(0)]})
        assert plan["load"] is True
        assert plan["fetch_remaining"] is True
        assert plan["raw_count"] == 1

    def test_an_explicit_none_count_falls_back_to_attempting_the_call(self) -> None:
        # `is not None` — not truthiness — gates the reference.
        plan = artist_songs_first_batch({"tracks": [_raw_track(0)], "trackCount": None})
        assert plan["fetch_remaining"] is True

    def test_the_tail_resumes_from_the_raw_length(self) -> None:
        # already_have is len(raw), not len(normalized): a dropped
        # (unplayable) row still counts toward the offset the tail
        # fetch resumes from.
        raw = [_raw_track(0), _unplayable(1), _unplayable(2)]
        plan = artist_songs_first_batch({"tracks": raw, "trackCount": 7})
        assert plan["load"] is True
        assert len(plan["tracks"]) == 1
        assert plan["raw_count"] == 3

    def test_an_empty_batch_loads_nothing_and_chains_nothing(self) -> None:
        assert artist_songs_first_batch({"tracks": [], "trackCount": 9}) == _closed_plan()

    def test_a_missing_tracks_key_loads_nothing(self) -> None:
        # get_playlist's graceful-degrade shape: {} on failure.
        assert artist_songs_first_batch({}) == _closed_plan()

    def test_an_all_unplayable_batch_loads_nothing(self) -> None:
        # Normalization dropped every row: leave the truncated list.
        plan = artist_songs_first_batch({"tracks": [_unplayable(0), _unplayable(1)]})
        assert plan == _closed_plan()

    def test_a_non_dict_payload_reads_as_an_empty_batch(self) -> None:
        # The worker's isinstance guard: anything the service did not
        # shape as a dict answers the closed plan.
        for data in (None, 42, "playlist", ["not", "a", "dict"]):
            assert artist_songs_first_batch(data) == _closed_plan()

    def test_the_plan_carries_its_keys_in_order(self) -> None:
        plan = artist_songs_first_batch({"tracks": [_raw_track(0)], "trackCount": 2})
        assert list(plan) == ["tracks", "load", "fetch_remaining", "raw_count"]

    def test_dict_subclasses_steer_the_count_read(self) -> None:
        class Overriding(dict):
            def get(self, key, default=None):
                if key == "trackCount":
                    return 99
                return super().get(key, default)

        plan = artist_songs_first_batch(Overriding(tracks=[_raw_track(0)], trackCount=1))
        assert plan["load"] is True
        # The synthetic 99 above the batch chains the tail, overriding
        # the stored 1 that would have closed it.
        assert plan["fetch_remaining"] is True

    def test_dict_subclasses_steer_the_tracks_read(self) -> None:
        class Overriding(dict):
            def get(self, key, default=None):
                if key == "tracks":
                    return [_raw_track(0)]
                return super().get(key, default)

        plan = artist_songs_first_batch(Overriding(trackCount=1))
        assert plan["load"] is True
        assert [t["video_id"] for t in plan["tracks"]] == ["v0"]
        # The stored count covers the steered batch: no tail.
        assert plan["fetch_remaining"] is False


# ── Tail plan ────────────────────────────────────────────────────────


class TestTailPlan:
    def test_a_non_empty_tail_appends_its_normalized_rows(self) -> None:
        plan = artist_songs_tail([_raw_track(3), _raw_track(4)])
        assert plan["append"] is True
        assert [t["video_id"] for t in plan["tracks"]] == ["v3", "v4"]

    def test_an_empty_tail_appends_nothing(self) -> None:
        assert artist_songs_tail([]) == {"tracks": [], "append": False}

    def test_an_all_unplayable_tail_appends_nothing(self) -> None:
        assert artist_songs_tail([_unplayable(0), _unplayable(1)]) == {
            "tracks": [],
            "append": False,
        }

    def test_the_tail_plan_carries_its_keys_in_order(self) -> None:
        plan = artist_songs_tail([_raw_track(0)])
        assert list(plan) == ["tracks", "append"]


# ── Total engine (native only) ──────────────────────────────────────


class TestTotalityEngine:
    """The native engines stay total where the references would raise.

    A stored ``trackCount`` Python's ``<=`` would reject raises
    ``TypeError`` in the reference; a truthy non-list ``tracks`` (or
    ``remaining``) raises in the reference's ``normalize_tracks``
    iteration. The native engines degrade instead — the convention the
    ported engines before it follow — so these tests only run with the
    extension built.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_an_uncomparable_track_count_closes_the_tail_gate(self) -> None:
        plan = _ytm_player_native.artist_songs_first_batch(
            {"tracks": [_raw_track(0), _raw_track(1)], "trackCount": "25"}
        )
        assert plan["load"] is True
        assert plan["fetch_remaining"] is False

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_truthy_non_list_tracks_value_reads_as_an_empty_batch(self) -> None:
        plan = _ytm_player_native.artist_songs_first_batch({"tracks": "v0v1", "trackCount": 9})
        assert plan == _closed_plan()

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_a_truthy_non_list_tail_reads_as_no_rows(self) -> None:
        plan = _ytm_player_native.artist_songs_tail("v3v4")
        assert plan == {"tracks": [], "append": False}


# ── Native vs fallback parity ────────────────────────────────────────

# One entry per engine case; the child interpreter below JSON
# round-trips them, so every value is a JSON type. The payload sticks
# to the reachable shapes — the deliberate native-only degradations
# (the uncomparable count, the non-list tracks/remaining) are pinned
# by TestTotalityEngine instead, exactly like the search engines' suite.

_FIRST_BATCH_CASES: list[Any] = [
    {"tracks": [_raw_track(0), _raw_track(1)], "trackCount": 5},
    {"tracks": [_raw_track(0), _raw_track(1)], "trackCount": 2},
    {"tracks": [_raw_track(0), _raw_track(1), _raw_track(2)], "trackCount": 2},
    {"tracks": [_raw_track(0)], "trackCount": 0},
    {"tracks": [_raw_track(0)]},
    {"tracks": [_raw_track(0)], "trackCount": None},
    {"tracks": [_raw_track(0), _raw_track(1)], "trackCount": 2.5},
    {"tracks": [_raw_track(0), _unplayable(1), _unplayable(2)], "trackCount": 7},
    {"tracks": [_raw_track(0), _raw_track(1), _raw_track(2), _unplayable(3)]},
    {"tracks": []},
    {},
    {"tracks": [_unplayable(0), _unplayable(1)], "trackCount": 9},
    None,
    42,
    "playlist",
    [["not", "a", "dict"]],
]

_TAIL_CASES: list[Any] = [
    [_raw_track(3), _raw_track(4)],
    [],
    [_unplayable(0)],
    [_raw_track(0), _unplayable(1), _raw_track(2)],
    [_raw_track(0), _raw_track(1), _raw_track(2), _raw_track(3)],
    None,
    0,
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.pages.context import artist_songs_first_batch, artist_songs_tail

payload = json.load(sys.stdin)

out = {
    "first_batch": [
        artist_songs_first_batch(data) for data in payload["first_batch_cases"]
    ],
    "tail": [artist_songs_tail(remaining) for remaining in payload["tail_cases"]],
}
json.dump(out, sys.stdout, ensure_ascii=False)
"""


def _native_results(payload: dict) -> dict:
    """Run whichever engines the module loaded (native when built) over
    the payload."""
    return {
        "first_batch": [artist_songs_first_batch(data) for data in payload["first_batch_cases"]],
        "tail": [artist_songs_tail(remaining) for remaining in payload["tail_cases"]],
    }


def _fallback_results(payload: dict) -> dict:
    """Run the pure-Python reference engines over the payload in a child
    interpreter with the native extension import blocked."""
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native artist-songs engines
    must produce exactly the same plans as the pure-Python references
    for every case — every branch of the remaining-fetch decision, both
    empty gates, the raw-length offset and every normalization outcome —
    so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self) -> None:
        payload = {
            "first_batch_cases": _FIRST_BATCH_CASES,
            "tail_cases": _TAIL_CASES,
        }
        native = _native_results(payload)
        fallback = _fallback_results(payload)
        assert native == fallback, (
            "native and fallback artist-songs plans differ:\n"
            f"native:   {native!r}\n"
            f"fallback: {fallback!r}"
        )


# ── The worker merge, through the real Textual loop ─────────────────


class _SongsYTMusic:
    """An artist whose songs playlist (and its tail) are fixed up front,
    with the tail fetches recorded."""

    def __init__(self, playlist_payload: dict[str, Any], remaining: list[dict[str, Any]]) -> None:
        self.playlist_payload = playlist_payload
        self.remaining = remaining
        self.remaining_calls: list[tuple[str, int]] = []

    async def get_artist(self, browse_id: str) -> dict[str, Any]:
        # ytmusicapi truncates the artist's top songs to a handful —
        # the full list lives behind songs.browseId as a playlist.
        return {
            "name": "Artist X",
            "songs": {"browseId": "VLPLsongs", "results": [_raw_track(0), _raw_track(1)]},
        }

    async def get_playlist(self, playlist_id: str, limit=None, order=None) -> dict[str, Any]:
        assert playlist_id == "VLPLsongs"
        return self.playlist_payload

    async def get_playlist_remaining(
        self, playlist_id: str, already_have: int, order=None
    ) -> list[dict[str, Any]]:
        self.remaining_calls.append((playlist_id, already_have))
        return self.remaining


class _Host(App[None]):
    def __init__(self, ytmusic: _SongsYTMusic) -> None:
        super().__init__()
        self.ytmusic = ytmusic

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield ContextPage("artist", "UC_x", id="context")


async def _settled_table(app: _Host, pilot) -> TrackTable:
    await asyncio.wait_for(app.workers.wait_for_complete(), 5)
    await pilot.pause()
    return app.query_one("#context-tracks", TrackTable)


class TestWorkerMergeGlue:
    """The engines' decisions reaching the worker's fetches and the
    top-songs table through the real Textual loop."""

    async def test_a_count_above_the_batch_chains_the_tail_from_the_raw_offset(self) -> None:
        # 4 raw rows, one unplayable: the batch swaps 3 in, the tail
        # resumes from 4 (len(raw), not the normalized 3).
        ytmusic = _SongsYTMusic(
            {
                "trackCount": 8,
                "tracks": [_raw_track(0), _unplayable(1), _raw_track(2), _raw_track(3)],
            },
            [_raw_track(4), _raw_track(5), _raw_track(6), _raw_track(7)],
        )
        async with _Host(ytmusic).run_test() as pilot:
            table = await _settled_table(pilot.app, pilot)

            assert table.row_count == 7  # 3 playable batch + 4 tail
            assert ytmusic.remaining_calls == [("VLPLsongs", 4)]

    async def test_a_covered_playlist_never_fetches_the_tail(self) -> None:
        ytmusic = _SongsYTMusic(
            {"trackCount": 3, "tracks": [_raw_track(0), _raw_track(1), _raw_track(2)]},
            [_raw_track(3)],
        )
        async with _Host(ytmusic).run_test() as pilot:
            table = await _settled_table(pilot.app, pilot)

            assert table.row_count == 3
            assert ytmusic.remaining_calls == []

    async def test_a_missing_count_falls_back_to_attempting_the_tail(self) -> None:
        # No trackCount: the tail fetch is attempted anyway, and its
        # empty answer reads as "nothing more to fetch".
        ytmusic = _SongsYTMusic({"tracks": [_raw_track(0), _raw_track(1)]}, [])
        async with _Host(ytmusic).run_test() as pilot:
            table = await _settled_table(pilot.app, pilot)

            assert table.row_count == 2
            assert ytmusic.remaining_calls == [("VLPLsongs", 2)]

    async def test_an_empty_batch_leaves_the_truncated_list_in_place(self) -> None:
        # get_playlist degraded to {}: the swap never happens, the tail
        # never chains, the original top songs stay.
        ytmusic = _SongsYTMusic({}, [_raw_track(3)])
        async with _Host(ytmusic).run_test() as pilot:
            table = await _settled_table(pilot.app, pilot)

            assert table.row_count == 2
            assert ytmusic.remaining_calls == []

    async def test_an_all_unplayable_batch_leaves_the_truncated_list_in_place(self) -> None:
        ytmusic = _SongsYTMusic(
            {"trackCount": 9, "tracks": [_unplayable(0), _unplayable(1)]},
            [_raw_track(3)],
        )
        async with _Host(ytmusic).run_test() as pilot:
            table = await _settled_table(pilot.app, pilot)

            assert table.row_count == 2
            assert ytmusic.remaining_calls == []
