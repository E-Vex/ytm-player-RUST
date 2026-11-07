"""Tests for the ytmusic RNG-driven selection engines.

The RNG-driven selection cores behind ``services/ytmusic.py``'s
radio and discovery mixes: the radio mix ``_radio_mix`` (the
multi-seed shuffle gate and the ``pool[:limit]`` trim behind
``get_radio``'s result assembly), the three-track seed sample
``_sample_seeds`` (the ``random.sample(pool, min(3, len(pool)))``
draw every discovery mix source makes) and the random artist pick
``_pick_artist`` (the ``random.choice(artists)`` behind the artist
mix's first draw). The pure-Python transcriptions live in
``services/ytmusic.py``; when the Rust extension is built the same
engines run natively
(``_ytm_player_native.ytm_radio_mix`` /
``_ytm_player_native.ytm_sample_seeds`` /
``_ytm_player_native.ytm_pick_artist``).

Both backends route their randomness through the Python ``random``
module — the QueueCore engines' own rule — so ``random.seed(...)``
before a call and monkeypatched draw functions steer them
identically. The unit tests below assert each engine's contract on
whichever backend is loaded; when the extension is built, the
parity tests additionally compare the native engines against the
pure-Python references (run in a child interpreter with the
extension import blocked) case for case under a shared seed —
including the in-place pool mutation and the error shapes, where
CPython's own messages must surface identically — so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from unittest.mock import patch

import pytest

from tests.conftest import make_ytmusic_service
from ytm_player.services.ytmusic import (
    _pick_artist,
    _radio_mix,
    _sample_seeds,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _ids(tracks: list) -> list:
    return [t["id"] if isinstance(t, dict) else t for t in tracks]


# ── The radio mix ───────────────────────────────────────────────────────


class TestRadioMix:
    def test_single_seed_keeps_first_sight_order(self):
        """seed_count of 1 (or less) never shuffles — single-seed callers
        expect the seed track to lead."""
        pool = [{"id": i} for i in range(6)]
        random.seed(42)
        out = _radio_mix(pool, 1, 4)
        assert _ids(out) == [0, 1, 2, 3]
        # The pool itself is untouched too — the shuffle never ran.
        assert _ids(pool) == list(range(6))

    def test_zero_seed_count_does_not_shuffle(self):
        pool = [{"id": i} for i in range(4)]
        out = _radio_mix(pool, 0, 4)
        assert _ids(out) == [0, 1, 2, 3]

    def test_multi_seed_shuffles_the_pool_in_place(self):
        """seed_count of 2+ shuffles the caller's pool itself, exactly
        like the reference's in-place random.shuffle."""
        pool = [{"id": i} for i in range(12)]
        random.seed(1234)
        expected = list(range(12))
        random.shuffle(expected)
        random.seed(1234)
        _radio_mix(pool, 2, 12)
        assert _ids(pool) == expected

    def test_multi_seed_trims_after_the_shuffle(self):
        """The slice is taken from the shuffled pool — the trim the
        reference spells pool[:limit]."""
        pool = [{"id": i} for i in range(12)]
        random.seed(1234)
        expected = list(range(12))
        random.shuffle(expected)
        random.seed(1234)
        out = _radio_mix(pool, 2, 5)
        assert _ids(out) == expected[:5]

    def test_seeded_sequences_match_the_module_stream(self):
        """The shuffle draws from the random module's own stream: the
        same seed answers the same mix, and matches a direct
        random.shuffle of the same pool."""
        pool_a = [{"id": i} for i in range(10)]
        pool_b = [{"id": i} for i in range(10)]
        random.seed(99)
        first = _ids(_radio_mix(pool_a, 2, 10))
        random.seed(99)
        second = _ids(_radio_mix(pool_b, 2, 10))
        assert first == second
        random.seed(99)
        expected = list(range(10))
        random.shuffle(expected)
        assert first == expected

    def test_trim_semantics_are_the_reference_slice(self):
        """The trim is CPython's own pool[:limit] — a limit past the end
        answers the whole pool, zero answers empty and a negative
        limit answers the tail-cut slice."""
        pool = [{"id": i} for i in range(5)]
        assert _ids(_radio_mix(pool, 1, 10)) == [0, 1, 2, 3, 4]
        assert _radio_mix(pool, 1, 0) == []
        assert _ids(_radio_mix(pool, 1, -2)) == [0, 1, 2]

    def test_items_pass_through_as_the_objects_they_are(self):
        pool = [{"id": i} for i in range(4)]
        out = _radio_mix(pool, 1, 3)
        assert out[0] is pool[0]
        assert out[2] is pool[2]

    def test_non_sequence_pool_raises_the_shuffle_error(self):
        with pytest.raises(TypeError):
            _radio_mix(5, 2, 3)

    def test_non_int_limit_raises_the_slice_error(self):
        with pytest.raises(TypeError):
            _radio_mix([{"id": 1}], 1, "x")  # type: ignore[arg-type]


# ── The discovery seed sample ───────────────────────────────────────────


class TestSampleSeeds:
    def test_draws_at_most_three(self):
        pool = [{"id": i} for i in range(10)]
        random.seed(7)
        out = _sample_seeds(pool)
        assert len(out) == 3
        assert all(item in pool for item in out)

    def test_short_pool_answers_every_track_it_has(self):
        assert _sample_seeds([{"id": 0}]) == [{"id": 0}]
        random.seed(3)
        out = _sample_seeds([{"id": 0}, {"id": 1}])
        assert len(out) == 2

    def test_empty_pool_answers_empty(self):
        assert _sample_seeds([]) == []

    def test_seeded_sequences_match_the_module_stream(self):
        pool_a = [{"id": i} for i in range(9)]
        pool_b = [{"id": i} for i in range(9)]
        random.seed(5)
        first = _ids(_sample_seeds(pool_a))
        random.seed(5)
        second = _ids(_sample_seeds(pool_b))
        assert first == second
        random.seed(5)
        assert first == _ids(random.sample(pool_b, 3))

    def test_items_pass_through_as_the_objects_they_are(self):
        pool = [{"id": i} for i in range(6)]
        random.seed(11)
        out = _sample_seeds(pool)
        assert all(item in pool for item in out)
        assert len({id(item) for item in out}) == 3

    def test_lengthless_pool_raises_the_len_error(self):
        with pytest.raises(TypeError):
            _sample_seeds(5)


# ── The artist pick ─────────────────────────────────────────────────────


class TestPickArtist:
    def test_answers_one_of_the_artists(self):
        artists = [{"artist": f"A{i}"} for i in range(5)]
        random.seed(3)
        picked = _pick_artist(artists)
        assert picked in artists

    def test_single_artist_answers_that_artist(self):
        only = {"artist": "Only"}
        assert _pick_artist([only]) is only

    def test_seeded_sequences_match_the_module_stream(self):
        artists = [{"artist": f"A{i}"} for i in range(6)]
        random.seed(21)
        first = _pick_artist(artists)
        random.seed(21)
        second = _pick_artist(artists)
        assert first is second
        random.seed(21)
        assert first is artists[random.choice(range(6))]

    def test_empty_artists_raises_the_choice_error(self):
        with pytest.raises(IndexError):
            _pick_artist([])

    def test_non_sequence_raises_the_len_error(self):
        with pytest.raises(TypeError):
            _pick_artist(5)


# ── Wrapper-level steering ──────────────────────────────────────────────


class TestWrapperSteering:
    async def test_monkeypatched_shuffle_steers_get_radio(self):
        """Patching random.shuffle (as the queue parity suite does for
        the QueueCore) must steer the whole get_radio chain — wrapper
        to engine to module — identically on either backend."""
        svc = make_ytmusic_service()
        raw_a = [{"videoId": f"a{i}", "title": f"A{i}"} for i in range(4)]
        raw_b = [{"videoId": f"b{i}", "title": f"B{i}"} for i in range(4)]

        async def mock_call(func, **kwargs):
            vid = kwargs.get("videoId", "")
            return {"tracks": raw_a if vid == "seedA" else raw_b}

        original_shuffle = random.shuffle
        random.shuffle = lambda items: items.reverse()
        try:
            with patch.object(svc, "_call", side_effect=mock_call):
                result = await svc.get_radio(["seedA", "seedB"], limit=4)
        finally:
            random.shuffle = original_shuffle
        # Pool in first-sight order is a0..a3, b0..b3; the reversed
        # shuffle answers b3, b2, b1, b0 first — the trim keeps four.
        assert [t["video_id"] for t in result] == ["b3", "b2", "b1", "b0"]

    async def test_monkeypatched_sample_steers_get_discovery_mix(self):
        """Patching random.sample must steer the discovery trending
        mix's seed draw through the engine chain on either backend."""
        svc = make_ytmusic_service()
        svc._last_discovery_source = 1  # → starts at source 2 (trending)

        async def mock_call(func, **kwargs):
            if func is svc.client.get_explore:
                return {"trending": {"playlist": "PL1"}}
            return {"tracks": [{"videoId": f"t{i}", "title": f"T{i}"} for i in range(6)]}

        original_sample = random.sample
        random.sample = lambda pool, k: pool[:k]
        try:
            with patch.object(svc, "_call", side_effect=mock_call):
                seeds, label = await svc.get_discovery_mix()
        finally:
            random.sample = original_sample
        assert label == "Trending"
        assert [s["videoId"] for s in seeds] == ["t0", "t1", "t2"]


# ── Native/fallback parity ──────────────────────────────────────────────

# The case matrices below are shared with the child interpreter as JSON,
# so every argument is spelled as a JSON-safe spec the child rebuilds.
#
# Each case seeds the random module before the engine runs — on both
# sides — so the native engine and the pure-Python reference draw from
# identical streams. `mutates` marks the engines whose first argument
# the reference mutates in place (the radio mix's shuffle); the pool's
# post-call shape is part of the compared outcome.
_PARITY_CASES = [
    # radio_mix — the single-seed arm keeps first-sight order.
    {"engine": "radio_mix", "seed": 1234, "mutates": True, "args": [{"t": "objs", "n": 8}, 1, 5]},
    # radio_mix — the multi-seed arm shuffles, then trims.
    {"engine": "radio_mix", "seed": 1234, "mutates": True, "args": [{"t": "objs", "n": 8}, 2, 5]},
    {"engine": "radio_mix", "seed": 42, "mutates": True, "args": [{"t": "objs", "n": 8}, 2, 5]},
    # radio_mix — a zero seed count never shuffles.
    {"engine": "radio_mix", "seed": 99, "mutates": True, "args": [{"t": "objs", "n": 4}, 0, 3]},
    # radio_mix — the trim past the end answers the whole pool.
    {"engine": "radio_mix", "seed": 7, "mutates": True, "args": [{"t": "objs", "n": 3}, 5, 10]},
    # radio_mix — a zero and a negative limit answer the slice's own
    # shapes (empty, tail-cut).
    {"engine": "radio_mix", "seed": 5, "mutates": True, "args": [{"t": "objs", "n": 6}, 2, 0]},
    {"engine": "radio_mix", "seed": 5, "mutates": True, "args": [{"t": "objs", "n": 6}, 2, -2]},
    # radio_mix — plain ints shuffle as any mutable sequence would.
    {"engine": "radio_mix", "seed": 7, "mutates": True, "args": [{"t": "ints", "n": 8}, 2, 4]},
    # radio_mix — a single-seed tuple pool answers a tuple slice.
    {"engine": "radio_mix", "seed": 1, "mutates": True, "args": [{"t": "tuple", "n": 4}, 1, 2]},
    # radio_mix — a multi-seed tuple pool raises the shuffle's own
    # TypeError; a non-int limit raises the slice's own.
    {"engine": "radio_mix", "seed": 1, "mutates": True, "args": [{"t": "tuple", "n": 4}, 2, 2]},
    {"engine": "radio_mix", "seed": 1, "mutates": True, "args": [{"t": "objs", "n": 4}, 1, "x"]},
    {"engine": "radio_mix", "seed": 1, "mutates": True, "args": [5, 2, 3]},
    # sample_seeds — the three-draw, the short pools, the empty pool.
    {"engine": "sample_seeds", "seed": 1234, "args": [{"t": "objs", "n": 10}]},
    {"engine": "sample_seeds", "seed": 8, "args": [{"t": "objs", "n": 2}]},
    {"engine": "sample_seeds", "seed": 8, "args": [{"t": "objs", "n": 1}]},
    {"engine": "sample_seeds", "seed": 8, "args": [{"t": "objs", "n": 0}]},
    # sample_seeds — plain ints and a tuple pool.
    {"engine": "sample_seeds", "seed": 77, "args": [{"t": "ints", "n": 7}]},
    {"engine": "sample_seeds", "seed": 77, "args": [{"t": "tuple", "n": 4}]},
    # sample_seeds — a lengthless pool raises the len TypeError.
    {"engine": "sample_seeds", "seed": 1, "args": [5]},
    {"engine": "sample_seeds", "seed": 1, "args": [{"t": "noget"}]},
    # pick_artist — the draw, the singleton, the errors.
    {"engine": "pick_artist", "seed": 1234, "args": [{"t": "objs", "n": 5}]},
    {"engine": "pick_artist", "seed": 1234, "args": [{"t": "objs", "n": 1}]},
    {"engine": "pick_artist", "seed": 77, "args": [{"t": "ints", "n": 3}]},
    {"engine": "pick_artist", "seed": 1, "args": [{"t": "tuple", "n": 0}]},
    {"engine": "pick_artist", "seed": 1, "args": [5]},
]


def _build(spec):
    """Rebuild a JSON-safe case argument as its live object."""
    if isinstance(spec, list):
        return [_build(item) for item in spec]
    if not isinstance(spec, dict):
        return spec
    kind = spec.get("t")
    if kind == "objs":
        return [{"id": i} for i in range(spec["n"])]
    if kind == "ints":
        return list(range(spec["n"]))
    if kind == "tuple":
        return tuple(range(spec["n"]))
    if kind == "noget":
        return object()
    raise AssertionError(kind)


def _render(value):
    """Render an engine outcome as a JSON-safe shape — identity by
    content for the id-dicts, structure for containers, the type name
    for anything else."""
    if isinstance(value, dict):
        return {"id": value["id"]}
    if isinstance(value, tuple):
        return ["__tuple__", [_render(v) for v in value]]
    if isinstance(value, list):
        return [_render(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"__obj__": type(value).__name__}


_ENGINES = {
    "radio_mix": _radio_mix,
    "sample_seeds": _sample_seeds,
    "pick_artist": _pick_artist,
}


def _outcome(case) -> dict:
    """Run one case against whichever engines this interpreter loaded."""
    args = [_build(spec) for spec in case["args"]]
    random.seed(case["seed"])
    try:
        out = _ENGINES[case["engine"]](*args)
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}
    result = {"ok": _render(out)}
    if case.get("mutates"):
        result["after"] = _render(args[0])
    return result


# The child interpreter runs this program with the case matrix on the
# stdin and prints the reference engines' outcomes as JSON — the native
# extension import blocked, the random module seeded per case, so the
# pure-Python references draw from the same streams.
_CHILD_PROG = """
import json
import random
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.ytmusic import (
    _pick_artist,
    _radio_mix,
    _sample_seeds,
)


def build(spec):
    if isinstance(spec, list):
        return [build(item) for item in spec]
    if not isinstance(spec, dict):
        return spec
    kind = spec.get('t')
    if kind == 'objs':
        return [{'id': i} for i in range(spec['n'])]
    if kind == 'ints':
        return list(range(spec['n']))
    if kind == 'tuple':
        return tuple(range(spec['n']))
    if kind == 'noget':
        return object()
    raise AssertionError(kind)


def render(value):
    if isinstance(value, dict):
        return {'id': value['id']}
    if isinstance(value, tuple):
        return ['__tuple__', [render(v) for v in value]]
    if isinstance(value, list):
        return [render(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {'__obj__': type(value).__name__}


ENGINES = {
    'radio_mix': _radio_mix,
    'sample_seeds': _sample_seeds,
    'pick_artist': _pick_artist,
}


def outcome(case):
    args = [build(spec) for spec in case['args']]
    random.seed(case['seed'])
    try:
        out = ENGINES[case['engine']](*args)
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}
    result = {'ok': render(out)}
    if case.get('mutates'):
        result['after'] = render(args[0])
    return result


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
    with the extension import blocked — same gates, same trims, the
    same draws under the same seeds (the in-place pool mutation
    included) and the same errors with the same messages — for every
    case, so the two implementations cannot drift."""

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

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_monkeypatched_draws_steer_the_native_engines(self):
        """Patching random.shuffle / random.sample / random.choice must
        steer the native engines exactly like the fallback: they call
        back into the random module rather than any generator of their
        own."""
        original = (random.shuffle, random.sample, random.choice)
        random.shuffle = lambda items: items.reverse()
        random.sample = lambda pool, k: pool[:k]
        random.choice = lambda seq: seq[0]
        try:
            pool = [{"id": i} for i in range(6)]
            out = _radio_mix(pool, 2, 3)
            assert [t["id"] for t in out] == [5, 4, 3]
            assert [t["id"] for t in pool] == [5, 4, 3, 2, 1, 0]
            assert _ids(_sample_seeds([{"id": i} for i in range(6)])) == [0, 1, 2]
            assert _pick_artist([{"id": 9}, {"id": 8}]) == {"id": 9}
        finally:
            random.shuffle, random.sample, random.choice = original
