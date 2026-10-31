"""Tests for ytm_player.services.queue.QueueManager."""

import random

import pytest

from ytm_player.services.queue import QueueManager, RepeatMode


class TestEmptyQueue:
    def test_is_empty(self, queue_manager):
        assert queue_manager.is_empty
        assert queue_manager.length == 0

    def test_current_is_none(self, queue_manager):
        assert queue_manager.current() is None

    def test_next_is_none(self, queue_manager):
        assert queue_manager.next_track() is None

    def test_previous_is_none(self, queue_manager):
        assert queue_manager.previous_track() is None

    def test_tracks_empty(self, queue_manager):
        assert queue_manager.tracks == ()


class TestSingleTrack:
    def test_add_and_current(self, queue_manager, sample_track):
        queue_manager.add(sample_track)
        assert queue_manager.length == 1
        assert not queue_manager.is_empty
        # Must jump to index 0 first.
        queue_manager.jump_to(0)
        assert queue_manager.current() == sample_track

    def test_repeat_off_next_returns_none(self, queue_manager, sample_track):
        queue_manager.add(sample_track)
        queue_manager.jump_to(0)
        # Next should return None (only 1 track, repeat off).
        assert queue_manager.next_track() is None

    def test_repeat_one(self, queue_manager, sample_track):
        queue_manager.add(sample_track)
        queue_manager.jump_to(0)
        queue_manager.set_repeat(RepeatMode.ONE)
        assert queue_manager.next_track() == sample_track
        assert queue_manager.next_track() == sample_track

    def test_repeat_all_wraps(self, queue_manager, sample_track):
        queue_manager.add(sample_track)
        queue_manager.jump_to(0)
        queue_manager.set_repeat(RepeatMode.ALL)
        assert queue_manager.next_track() == sample_track


class TestMultipleTracks:
    def test_forward_navigation(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        assert queue_manager.current()["video_id"] == "vid_01"
        assert queue_manager.next_track()["video_id"] == "vid_02"
        assert queue_manager.next_track()["video_id"] == "vid_03"

    def test_backward_navigation(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(2)
        assert queue_manager.current()["video_id"] == "vid_03"
        assert queue_manager.previous_track()["video_id"] == "vid_02"
        assert queue_manager.previous_track()["video_id"] == "vid_01"

    def test_no_wrap_without_repeat(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(4)
        assert queue_manager.next_track() is None

    def test_wrap_with_repeat_all(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.set_repeat(RepeatMode.ALL)
        queue_manager.jump_to(4)
        track = queue_manager.next_track()
        assert track["video_id"] == "vid_01"

    def test_backward_wrap_with_repeat_all(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.set_repeat(RepeatMode.ALL)
        queue_manager.jump_to(0)
        track = queue_manager.previous_track()
        assert track["video_id"] == "vid_05"


class TestShuffle:
    def test_toggle_shuffle_on_off(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)

        queue_manager.toggle_shuffle()
        assert queue_manager.shuffle_enabled
        # Current track should remain the same.
        assert queue_manager.current()["video_id"] == "vid_01"

        queue_manager.toggle_shuffle()
        assert not queue_manager.shuffle_enabled

    def test_shuffle_plays_all_tracks(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        seen = {queue_manager.current()["video_id"]}
        for _ in range(4):
            t = queue_manager.next_track()
            assert t is not None
            seen.add(t["video_id"])

        assert len(seen) == 5

    def test_add_while_shuffled(self, queue_manager, sample_tracks, sample_track):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        queue_manager.add(sample_track)
        assert queue_manager.length == 6


class TestOperations:
    def test_add_at_position(self, queue_manager, sample_tracks, sample_track):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.add(sample_track, position=2)
        assert queue_manager.length == 6
        queue_manager.jump_to(2)
        assert queue_manager.current()["video_id"] == sample_track["video_id"]

    def test_clear(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.clear()
        assert queue_manager.is_empty
        assert queue_manager.current() is None

    def test_remove(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.remove(1)
        assert queue_manager.length == 4
        queue_manager.jump_to(1)
        assert queue_manager.current()["video_id"] == "vid_03"

    def test_jump_to_valid(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        result = queue_manager.jump_to(3)
        assert result["video_id"] == "vid_04"

    def test_jump_to_invalid(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        assert queue_manager.jump_to(10) is None
        assert queue_manager.jump_to(-1) is None

    def test_cycle_repeat(self, queue_manager):
        assert queue_manager.repeat_mode == RepeatMode.OFF
        assert queue_manager.cycle_repeat() == RepeatMode.ALL
        assert queue_manager.cycle_repeat() == RepeatMode.ONE
        assert queue_manager.cycle_repeat() == RepeatMode.OFF

    def test_add_multiple(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        assert queue_manager.length == 5

    def test_move(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.move(0, 4)
        queue_manager.jump_to(0)
        assert queue_manager.current()["video_id"] == "vid_02"
        queue_manager.jump_to(4)
        assert queue_manager.current()["video_id"] == "vid_01"


class TestShuffleAdvanced:
    """Shuffle-mode tests for remove, move, add_next, previous_track, and wrap."""

    def test_remove_in_shuffle_mode(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        # Advance once so we're at position 1.
        next_t = queue_manager.next_track()
        assert next_t is not None
        current_vid = next_t["video_id"]

        # Remove position 0 (already played) — current should stay the same.
        queue_manager.remove(0)
        assert queue_manager.length == 4
        assert queue_manager.current()["video_id"] == current_vid

    def test_move_in_shuffle_mode(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        current_vid = queue_manager.current()["video_id"]
        # Move current track (position 0) to position 3.
        queue_manager.move(0, 3)
        # After move, current should still point at the same track.
        assert queue_manager.current()["video_id"] == current_vid

    def test_add_next_in_shuffle_mode(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        extra = {
            "video_id": "extra_1",
            "title": "Extra",
            "artist": "X",
            "artists": [],
            "album": "",
            "album_id": None,
            "duration": 100,
            "thumbnail_url": None,
            "is_video": False,
        }
        queue_manager.add_next(extra)
        assert queue_manager.length == 6
        # The next track should be the one we just added.
        nxt = queue_manager.next_track()
        assert nxt["video_id"] == "extra_1"

    def test_previous_in_shuffle_mode(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        first_vid = queue_manager.current()["video_id"]
        second = queue_manager.next_track()
        assert second is not None

        prev = queue_manager.previous_track()
        assert prev is not None
        assert prev["video_id"] == first_vid

    def test_previous_at_start_no_repeat(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        assert queue_manager.previous_track() is None

    def test_next_wraps_with_repeat_all_shuffle(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        queue_manager.set_repeat(RepeatMode.ALL)

        # Exhaust all tracks.
        seen = {queue_manager.current()["video_id"]}
        for _ in range(4):
            t = queue_manager.next_track()
            assert t is not None
            seen.add(t["video_id"])
        assert len(seen) == 5

        # Next should wrap (rebuild shuffle) and return a valid track.
        wrap = queue_manager.next_track()
        assert wrap is not None
        assert wrap["video_id"] in {t["video_id"] for t in sample_tracks}

    def test_next_end_no_repeat_shuffle(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        # Exhaust all tracks.
        for _ in range(4):
            queue_manager.next_track()

        # No repeat → None at end.
        assert queue_manager.next_track() is None

    def test_play_random(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        result = queue_manager.play_random()
        assert result is not None
        assert result["video_id"] in {t["video_id"] for t in sample_tracks}

    def test_play_random_empty(self, queue_manager):
        assert queue_manager.play_random() is None


class TestRemoveEdgeCases:
    """Edge cases for QueueManager.remove() — issue #22."""

    def test_remove_currently_playing(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(2)
        assert queue_manager.current()["video_id"] == "vid_03"

        queue_manager.remove(2)
        assert queue_manager.length == 4
        # The audible entry finishes outside the queue; EOF selects its successor.
        assert queue_manager.current() is None
        assert queue_manager.next_track()["video_id"] == "vid_04"

    def test_remove_last_track_empties_queue(self, queue_manager):
        from tests.conftest import _make_track

        track = _make_track("only", "Only Track", "Solo Artist", 100)
        queue_manager.add(track)
        queue_manager.jump_to(0)
        assert queue_manager.current()["video_id"] == "only"

        queue_manager.remove(0)
        assert queue_manager.is_empty
        assert queue_manager.current() is None

    def test_remove_before_current_adjusts_index(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(3)
        assert queue_manager.current()["video_id"] == "vid_04"

        queue_manager.remove(1)
        assert queue_manager.length == 4
        # Current track should still be vid_04 (index shifted down).
        assert queue_manager.current()["video_id"] == "vid_04"

    def test_remove_after_current_keeps_index(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(1)
        assert queue_manager.current()["video_id"] == "vid_02"

        queue_manager.remove(3)
        assert queue_manager.length == 4
        assert queue_manager.current()["video_id"] == "vid_02"


class TestRemoveShuffleCurrentTrack:
    """remove() of the current track under shuffle must match the
    non-shuffle contract: finish the removed song, then select its successor."""

    def _shuffled_order(self, queue_manager, sample_tracks) -> list[str]:
        """Populate, enable shuffle, return video_ids in shuffle order."""
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        return [t["video_id"] for t in queue_manager.tracks]

    def test_remove_current_lands_on_next_shuffle_track(self, queue_manager, sample_tracks):
        order = self._shuffled_order(queue_manager, sample_tracks)
        queue_manager.next_track()
        assert queue_manager.current()["video_id"] == order[1]

        queue_manager.remove(1)
        assert queue_manager.current() is None
        assert queue_manager.next_track()["video_id"] == order[2]

    def test_remove_current_at_end_does_not_replay_previous(self, queue_manager, sample_tracks):
        order = self._shuffled_order(queue_manager, sample_tracks)
        queue_manager.jump_to(len(order) - 1)

        queue_manager.remove(len(order) - 1)
        assert queue_manager.current() is None
        assert queue_manager.next_track() is None

    def test_remove_before_current_keeps_current_track(self, queue_manager, sample_tracks):
        order = self._shuffled_order(queue_manager, sample_tracks)
        queue_manager.jump_to(2)

        queue_manager.remove(0)
        assert queue_manager.current()["video_id"] == order[2]

    def test_remove_after_current_keeps_current_track(self, queue_manager, sample_tracks):
        order = self._shuffled_order(queue_manager, sample_tracks)
        queue_manager.jump_to(1)

        queue_manager.remove(3)
        assert queue_manager.current()["video_id"] == order[1]

    def test_remove_only_track_shuffle_empties_queue(self, queue_manager):
        from tests.conftest import _make_track

        queue_manager.add(_make_track("only", "Only Track", "Solo Artist", 100))
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()

        queue_manager.remove(0)
        assert queue_manager.is_empty
        assert queue_manager.current() is None
        assert queue_manager.current_index == -1


class TestRemovedAudibleBoundary:
    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize("repeat", [RepeatMode.OFF, RepeatMode.ALL, RepeatMode.ONE])
    @pytest.mark.parametrize("count,position", [(1, 0), (4, 0), (4, 1), (4, 3)])
    def test_finish_then_next(self, queue_manager, shuffle, repeat, count, position):
        q = queue_manager
        # Duplicate values and even shared dicts cannot define occurrence identity.
        same = {"video_id": "A"}
        q.add_multiple([same] * count)
        if shuffle:
            q.toggle_shuffle()
        q.set_repeat(repeat)
        q.jump_to(position)
        before = q.entries
        q.remove_entry(before[position][0])
        assert q.current_track is None
        assert q.current_index == -1
        expected = before[position + 1][0] if position + 1 < count else None
        if expected is None and count > 1 and repeat == RepeatMode.ALL:
            expected = before[0][0]
        next_track = q.next_track()
        if expected is None:
            assert next_track is None
        else:
            assert next_track is same
            assert q.entries[q.current_index][0] == expected

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_removing_successor_again_keeps_following_survivor(self, queue_manager, shuffle):
        q = queue_manager
        q.add_multiple([{"video_id": str(i)} for i in range(5)])
        if shuffle:
            q.toggle_shuffle()
        q.jump_to(1)
        before = q.entries
        q.remove_entry(before[1][0])
        q.remove_entry(before[2][0])
        assert q.peek_next() is before[3][1]
        assert q.next_track() is before[3][1]

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_play_next_insert_at_removed_boundary(self, queue_manager, shuffle):
        q = queue_manager
        q.add_multiple([{"video_id": str(i)} for i in range(4)])
        if shuffle:
            q.toggle_shuffle()
        q.jump_to(1)
        before = q.entries
        q.remove_entry(before[1][0])
        new = [{"video_id": "X"}, {"video_id": "Y"}]
        q.add_next_multiple(new)
        assert q.next_track() is new[0]
        assert q.next_track() is new[1]
        assert q.next_track() is before[2][1]

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_append_after_removed_last_starts_new_tail(self, queue_manager, shuffle):
        q = queue_manager
        q.add_multiple([{"video_id": "A"}, {"video_id": "B"}])
        if shuffle:
            q.toggle_shuffle()
        q.jump_to(1)
        q.remove(1)
        tail = {"video_id": "C"}
        q.add(tail)
        assert q.next_track() is tail

    def test_reorder_keeps_the_surviving_successor_identity(self, queue_manager):
        q = queue_manager
        q.add_multiple([{"video_id": str(i)} for i in range(5)])
        q.jump_to(1)
        successor = q.entries[2]
        q.remove(1)
        q.move(1, 3)
        q.toggle_shuffle()
        assert q.next_track() is successor[1]
        assert q.entries[q.current_index][0] == successor[0]

    def test_clear_only_changes_lifetime(self, queue_manager):
        q = queue_manager
        generation = q.generation
        q.add_multiple([{"video_id": "A"}, {"video_id": "A"}])
        q.jump_to(0)
        q.next_track()
        q.toggle_shuffle()
        q.move(0, 1)
        q.remove(0)
        assert q.generation == generation
        q.clear()
        assert q.generation == generation + 1


class TestRadioTracks:
    def test_set_radio_tracks_deduplicates(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)

        radio = [
            {"video_id": "vid_01", "title": "Dup"},
            {"video_id": "vid_new", "title": "New Track"},
        ]
        queue_manager.set_radio_tracks(radio)
        assert queue_manager.length == 6  # 5 + 1 new

    def test_set_radio_tracks_all_new(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)

        radio = [
            {"video_id": "new_1", "title": "New 1"},
            {"video_id": "new_2", "title": "New 2"},
        ]
        queue_manager.set_radio_tracks(radio)
        assert queue_manager.length == 7

    def test_set_radio_tracks_empty_shuffle_rebuilds(self, queue_manager, sample_tracks):
        """Radio tracks into an empty shuffled queue must rebuild the shuffle
        order — a missed rebuild leaves the playback order empty (regression)."""
        queue_manager.toggle_shuffle()  # shuffle on while queue is still empty
        assert queue_manager.shuffle_enabled

        queue_manager.set_radio_tracks(sample_tracks)

        # Under shuffle the playback order follows the shuffle order; a
        # queue that skipped the rebuild would show no tracks at all.
        assert [t["video_id"] for t in queue_manager.tracks] != []
        assert sorted(t["video_id"] for t in queue_manager.tracks) == sorted(
            t["video_id"] for t in sample_tracks
        )


class TestQueuePositionTracking:
    """Tests for real_index and remaining_tracks properties."""

    def test_real_index_without_shuffle(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(2)
        assert queue_manager.real_index == 2

    def test_real_index_with_shuffle_resolves_to_tracks_index(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.toggle_shuffle()
        queue_manager.jump_to(0)
        assert 0 <= queue_manager.real_index < len(sample_tracks)

    def test_remaining_tracks_without_shuffle(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(2)
        assert queue_manager.remaining_tracks == 2  # tracks 3 and 4

    def test_remaining_tracks_with_shuffle(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.toggle_shuffle()
        queue_manager.jump_to(0)
        assert queue_manager.remaining_tracks == len(sample_tracks) - 1


class TestContextId:
    def test_default_is_none(self, queue_manager):
        assert queue_manager.current_context_id is None

    def test_set_and_read_back(self, queue_manager):
        queue_manager.set_context("PLABCD")
        assert queue_manager.current_context_id == "PLABCD"

    def test_set_to_none_clears(self, queue_manager):
        queue_manager.set_context("PLABCD")
        queue_manager.set_context(None)
        assert queue_manager.current_context_id is None

    def test_clear_does_not_affect_context(self, queue_manager, sample_track):
        # clear() resets tracks/shuffle state but does NOT reset context_id —
        # the context only changes via explicit set_context() at the
        # playback-start sites.
        queue_manager.set_context("PLABCD")
        queue_manager.add(sample_track)
        queue_manager.clear()
        assert queue_manager.current_context_id == "PLABCD"


class TestRadioSeeds:
    def test_default_is_none(self, queue_manager):
        assert queue_manager.radio_seeds is None

    def test_set_and_read_back(self, queue_manager):
        seeds = [{"title": "Song A"}, {"title": "Song B"}]
        queue_manager.radio_seeds = seeds
        assert queue_manager.radio_seeds is seeds

    def test_clear_resets_to_none(self, queue_manager, sample_track):
        queue_manager.radio_seeds = [{"title": "Song A"}]
        queue_manager.add(sample_track)
        queue_manager.clear()
        assert queue_manager.radio_seeds is None


class TestPeekNext:
    """peek_next is prefetch-critical: it must never advance the position."""

    def test_empty_queue_is_none(self, queue_manager):
        assert queue_manager.peek_next() is None

    def test_returns_next_without_advancing(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)
        assert queue_manager.peek_next() == sample_tracks[1]
        # Position must be unchanged after peeking.
        assert queue_manager.current() == sample_tracks[0]

    def test_last_track_repeat_off_is_none(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(len(sample_tracks) - 1)
        assert queue_manager.peek_next() is None

    def test_last_track_repeat_all_wraps_to_first(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(len(sample_tracks) - 1)
        queue_manager.set_repeat(RepeatMode.ALL)
        assert queue_manager.peek_next() == sample_tracks[0]

    def test_repeat_one_returns_current(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(2)
        queue_manager.set_repeat(RepeatMode.ONE)
        assert queue_manager.peek_next() == sample_tracks[2]

    def test_shuffle_end_of_order_is_none(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        # Jump to the final shuffle position — no known next to prefetch.
        queue_manager.jump_to(len(sample_tracks) - 1)
        assert queue_manager.peek_next() is None

    def test_shuffle_mid_order_returns_a_track(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        queue_manager.jump_to(0)
        nxt = queue_manager.peek_next()
        assert nxt in sample_tracks


def _extra(video_id: str) -> dict:
    """A minimal standardized track dict for add_next_multiple tests."""
    return {
        "video_id": video_id,
        "title": video_id,
        "artist": "X",
        "artists": [],
        "album": "",
        "album_id": None,
        "duration": 100,
        "thumbnail_url": None,
        "is_video": False,
    }


class TestAddNextMultiple:
    """add_next_multiple: insert N tracks right after the current track, in order."""

    def test_empty_input_is_noop(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)
        queue_manager.add_next_multiple([])
        assert queue_manager.length == 5

    def test_empty_queue_inserts_at_front_in_order(self, queue_manager):
        queue_manager.add_next_multiple([_extra("x1"), _extra("x2")])
        assert queue_manager.length == 2
        assert [t["video_id"] for t in queue_manager.tracks] == ["x1", "x2"]

    def test_mid_queue_inserts_after_current_in_order(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(1)  # current = vid_02
        queue_manager.add_next_multiple([_extra("x1"), _extra("x2")])
        vids = [t["video_id"] for t in queue_manager.tracks]
        assert vids == ["vid_01", "vid_02", "x1", "x2", "vid_03", "vid_04", "vid_05"]
        # Current track is unaffected by the insertion after it.
        assert queue_manager.current()["video_id"] == "vid_02"

    def test_next_plays_inserted_tracks_in_order(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)  # current = vid_01
        queue_manager.add_next_multiple([_extra("x1"), _extra("x2")])
        assert queue_manager.next_track()["video_id"] == "x1"
        assert queue_manager.next_track()["video_id"] == "x2"
        assert queue_manager.next_track()["video_id"] == "vid_02"

    def test_order_preserved_for_many(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(2)
        queue_manager.add_next_multiple([_extra(f"x{i}") for i in range(5)])
        played = [queue_manager.next_track()["video_id"] for _ in range(5)]
        assert played == ["x0", "x1", "x2", "x3", "x4"]

    def test_shuffle_inserts_next_in_order(self, queue_manager, sample_tracks):
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        queue_manager.add_next_multiple([_extra("x1"), _extra("x2")])
        assert queue_manager.length == 7
        # Under shuffle the inserted tracks are the next N to play, in order.
        assert queue_manager.next_track()["video_id"] == "x1"
        assert queue_manager.next_track()["video_id"] == "x2"

    def test_duplicates_inserted_as_is(self, queue_manager, sample_tracks):
        # No dedup, consistent with add_next().
        queue_manager.add_multiple(sample_tracks)
        queue_manager.jump_to(0)
        queue_manager.add_next_multiple([sample_tracks[0]])  # vid_01 already present
        assert queue_manager.length == 6
        assert queue_manager.next_track()["video_id"] == "vid_01"


class TestShuffleOffKeepsThePlayingOccurrence:
    """Turning shuffle off must keep the track reached under shuffle.

    Only ``jump_to`` kept ``_current_index`` in sync while shuffled, so after
    ``next_track``/``previous_track`` the toggle used to snap back to the
    track that was playing when shuffle went on.
    """

    def _same_song(self, n: int) -> list[dict]:
        return [
            {"video_id": "same", "title": "Same", "artist": "A", "duration": 100} for _ in range(n)
        ]

    def test_shuffle_off_keeps_the_track_reached_by_next(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        reached = queue_manager.next_track()
        assert reached is not None

        queue_manager.toggle_shuffle()

        assert queue_manager.current_track is reached
        position = sample_tracks.index(reached)
        assert queue_manager.current_index == position
        following = sample_tracks[position + 1] if position + 1 < len(sample_tracks) else None
        assert queue_manager.next_track() is following

    def test_shuffle_off_keeps_the_track_reached_by_previous(self, queue_manager, sample_tracks):
        for t in sample_tracks:
            queue_manager.add(t)
        queue_manager.jump_to(0)
        queue_manager.toggle_shuffle()
        queue_manager.next_track()
        queue_manager.next_track()
        reached = queue_manager.previous_track()
        assert reached is not None

        queue_manager.toggle_shuffle()

        assert queue_manager.current_track is reached
        assert queue_manager.current_index == sample_tracks.index(reached)

    def test_shuffle_off_keeps_the_occurrence_among_identical_songs(self):
        """Asserted by entry id: the same song six times, random walks, then off."""
        for seed in range(40):
            rng = random.Random(seed)
            queue = QueueManager()
            queue.add_multiple(self._same_song(6))
            queue.jump_to(rng.randrange(6))
            queue.toggle_shuffle()
            for _ in range(rng.randrange(1, 12)):
                step = rng.choice(("next", "previous", "jump"))
                if step == "next":
                    queue.next_track()
                elif step == "previous":
                    queue.previous_track()
                else:
                    queue.jump_to(rng.randrange(6))
            playing_id, playing = queue.entries[queue.current_index]

            queue.toggle_shuffle()

            entry_id, track = queue.entries[queue.current_index]
            assert entry_id == playing_id, f"seed {seed}"
            assert track is playing and queue.current_track is playing


# ── Native/fallback parity ──────────────────────────────────────────────
#
# When the Rust extension is built, QueueManager runs the native QueueCore;
# otherwise the pure-Python _QueueCore transcription. Both cores draw
# randomness through the global ``random`` module, so seeding it
# identically makes even shuffled playback orders reproducible — and the
# two implementations comparable byte-for-byte. Each case below is a
# script run against a fresh QueueManager in this interpreter (native
# core) and in a child interpreter with the extension import blocked
# (fallback core); the dumped observable state must match exactly.

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_DUMP = """(
    [(i, x["video_id"]) for i, x in q.entries],
    q.current_index,
    q.real_index,
    q.length,
    q.remaining_tracks,
    q.shuffle_enabled,
    q.repeat_mode.value,
    q.generation,
    (q.current() or {}).get("video_id"),
    (q.peek_next() or {}).get("video_id"),
)"""

_PARITY_CASES: list[tuple[int, str]] = [
    # Deterministic navigation, no randomness at all.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(5)]); q.jump_to(2); "
        "q.next_track(); q.previous_track(); q.next_track(); q.next_track()",
    ),
    # Exhaustion then repeat-all wraparound.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(3)]); q.jump_to(0); "
        "q.set_repeat(RepeatMode.ALL); [q.next_track() for _ in range(7)]",
    ),
    # Repeat-one keeps returning the same track.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(3)]); q.jump_to(1); "
        "q.set_repeat(RepeatMode.ONE); [q.next_track() for _ in range(3)]; q.previous_track()",
    ),
    # Seeded full shuffle, a walk through it, then unshuffling back.
    (
        42,
        "q.add_multiple([t(f'v{i}') for i in range(8)]); q.jump_to(3); q.toggle_shuffle(); "
        "[q.next_track() for _ in range(4)]; q.toggle_shuffle()",
    ),
    # Shuffle wrap with repeat-all exercises the reshuffle path.
    (
        7,
        "q.add_multiple([t(f'v{i}') for i in range(5)]); q.toggle_shuffle(); "
        "q.set_repeat(RepeatMode.ALL); [q.next_track() for _ in range(12)]",
    ),
    # Removing the current row: removed-current bookkeeping, then EOF picks
    # the successor.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(5)]); q.jump_to(2); q.remove(2); "
        "q.current(); q.next_track(); q.next_track()",
    ),
    # Removals under a seeded shuffle order.
    (
        99,
        "q.add_multiple([t(f'v{i}') for i in range(6)]); q.toggle_shuffle(); "
        "q.jump_to(2); q.remove(1); q.remove(3); q.next_track(); q.previous_track()",
    ),
    # add_next / add_next_multiple / positioned add under shuffle.
    (
        5,
        "q.add_multiple([t(f'v{i}') for i in range(4)]); q.toggle_shuffle(); q.jump_to(1); "
        "q.add_next(t('nx')); q.add_next_multiple([t('m1'), t('m2')]); "
        "q.add(t('ins'), position=1); [q.next_track() for _ in range(3)]",
    ),
    # add_next while the audible row was just removed: splice at the boundary.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(4)]); q.jump_to(1); q.remove(1); "
        "q.add_next(t('after')); q.add_next_multiple([t('b1'), t('b2')]); "
        "q.next_track(); q.next_track()",
    ),
    # Moves, removal by entry id, and the three jump flavours.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(5)]); q.jump_to(2); q.move(0, 3); "
        "ids = [i for i, _ in q.entries]; q.remove_entry(ids[4]); q.jump_to_entry(ids[1]); "
        "q.jump_to_real(0); q.remove(0)",
    ),
    # play_random + radio dedup (set_radio_tracks skips known video ids).
    (
        11,
        "q.add_multiple([t(f'v{i}') for i in range(5)]); q.play_random(); "
        "q.set_radio_tracks([t('v2'), t('r1'), t('r2')]); q.play_random()",
    ),
    # clear bumps the generation; refill rebuilds the shuffle order.
    (
        3,
        "q.add_multiple([t(f'v{i}') for i in range(4)]); q.clear(); "
        "q.add_multiple([t(f'n{i}') for i in range(3)]); q.jump_to(0); q.toggle_shuffle()",
    ),
    # cycle_repeat walks OFF -> ALL -> ONE -> OFF -> ALL.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(2)]); q.jump_to(0); "
        "[q.cycle_repeat() for _ in range(5)]; q.next_track()",
    ),
    # previous at the shuffle start with repeat-all wraps to the end;
    # peek_next reports the wrap-around successor without advancing.
    (
        8,
        "q.add_multiple([t(f'v{i}') for i in range(4)]); q.toggle_shuffle(); "
        "q.set_repeat(RepeatMode.ALL); q.jump_to(0); q.previous_track(); q.peek_next()",
    ),
    # The deterministic-shuffle test seam behaves the same on both cores.
    (
        0,
        "q.add_multiple([t(f'v{i}') for i in range(4)]); "
        "q._force_shuffle_state([2, 0, 3, 1], -1); q.next_track(); q.remove(0)",
    ),
]


def _native_run(seed: int, script: str) -> str:
    """Run *script* against a native-core QueueManager in this interpreter."""
    ns: dict = {
        "q": QueueManager(),
        "t": lambda v: {"video_id": v, "title": v},
        "RepeatMode": RepeatMode,
    }
    random.seed(seed)
    exec(script, ns)  # noqa: S307 - fixed test scripts, not user input
    return repr(eval(_DUMP, ns))  # noqa: S307


def _fallback_run(seed: int, script: str) -> str:
    """Run *script* against a fallback-core QueueManager in a child
    interpreter with the native extension import blocked."""
    import subprocess
    import sys

    prog = (
        "import sys\n"
        "sys.modules['ytm_player.utils._ytm_player_native'] = None\n"
        "import random\n"
        f"random.seed({seed})\n"
        "from ytm_player.services.queue import QueueManager, RepeatMode\n"
        "q = QueueManager()\n"
        "def t(v):\n"
        "    return {'video_id': v, 'title': v}\n"
        f"{script}\n"
        f"print(repr({_DUMP}))\n"
    )
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True)
    return out.stdout.strip()


class TestNativeParity:
    """When the Rust extension is built, the native QueueCore must be
    observably identical to the pure-Python fallback — same tracks, same
    entry ids, same positions, same seeded shuffle orders — after every
    scripted operation sequence, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("seed,script", _PARITY_CASES)
    def test_native_matches_fallback(self, seed, script):
        assert _native_run(seed, script) == _fallback_run(seed, script)

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback_on_identical_songs(self):
        """The occurrence-preserving walk from the shuffle-off test, pinned
        across backends: the same dict object queued six times, seeded
        random walks, then leaving shuffle must land on the same entry."""
        script = (
            "rng = random.Random(6)\n"
            "same = t('same')\n"
            "q.add_multiple([same for _ in range(6)])\n"
            "q.jump_to(rng.randrange(6))\n"
            "q.toggle_shuffle()\n"
            "for _ in range(rng.randrange(1, 12)):\n"
            "    step = rng.choice(('next', 'previous', 'jump'))\n"
            "    if step == 'next':\n"
            "        q.next_track()\n"
            "    elif step == 'previous':\n"
            "        q.previous_track()\n"
            "    else:\n"
            "        q.jump_to(rng.randrange(6))\n"
            "playing_id, playing = q.entries[q.current_index]\n"
            "q.toggle_shuffle()\n"
            "assert q.entries[q.current_index][0] == playing_id\n"
            "assert q.current_track is playing\n"
        )
        ns: dict = {
            "q": QueueManager(),
            "t": lambda v: {"video_id": v, "title": v},
            "random": random,
        }
        random.seed(1234)
        exec(script, ns)
        assert repr(eval(_DUMP, ns)) == _fallback_run(1234, script)

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_monkeypatched_random_shuffle_steers_both_cores(self, monkeypatch):
        """Patching random.shuffle (as the playback-ownership tests do) must
        steer the native core exactly like the fallback: the native core
        calls back into the random module rather than its own generator."""
        script = (
            "q.add_multiple([t(c) for c in 'ABCDE'])\n"
            "q.toggle_shuffle()\n"
            "[q.next_track() for _ in range(4)]\n"
        )
        monkeypatch.setattr(random, "shuffle", lambda items: items.reverse())
        native_ns: dict = {"q": QueueManager(), "t": lambda v: {"video_id": v, "title": v}}
        exec(script, native_ns)
        # With the reversed shuffle the visible order is E..A and playback
        # has walked E, D, C, B.
        assert [x["video_id"] for x in native_ns["q"].tracks] == list("EDCBA")
        native_dump = repr(eval(_DUMP, native_ns))

        fallback_script = "import random\nrandom.shuffle = lambda items: items.reverse()\n" + script
        assert native_dump == _fallback_run(0, fallback_script)
