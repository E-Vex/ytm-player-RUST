"""Concurrency tests for YTMusicService.

The get_playlist(order=...) path monkey-patches client._send_request to
inject sort params.  Two concurrent calls would stack patches and fail to
fully restore the original — meaning a third call could see a stale
patched _send_request.

The patch-window bookkeeping engines — the in-flight counter's admission
(``_patch_window_enter``) and release (``_patch_window_exit``) steps behind
``_call``'s gate and finally arms — are unit-tested on whichever backend is
loaded, wired through ``_call`` itself, and (when the Rust extension is
built) compared against the pure-Python references run in a child
interpreter with the extension import blocked, error shapes included, so
the two implementations cannot drift.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import make_ytmusic_service
from ytm_player.services.ytmusic import _patch_window_enter, _patch_window_exit

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


@pytest.fixture
def ytmusic_service():
    """Construct YTMusicService with a fake client (bypasses __init__)."""
    fake_client = MagicMock()
    # Real send_request impl we want to be restored.
    fake_client._original_send = lambda endpoint, body, *a, **kw: {"contents": []}
    fake_client._send_request = fake_client._original_send

    # get_playlist returns minimal valid response.
    def fake_get_playlist(playlist_id, **kwargs):
        return {"tracks": []}

    fake_client.get_playlist = fake_get_playlist
    return make_ytmusic_service(_ytm=fake_client)


class TestSendRequestRaceCondition:
    """C3: concurrent get_playlist(order=...) must not corrupt _send_request."""

    async def test_concurrent_order_calls_restore_original_send_request(self, ytmusic_service):
        """Run two get_playlist(order=...) calls concurrently and assert
        client._send_request is the ORIGINAL function after both complete.

        The bug: each call captures original_send = client._send_request
        BEFORE patching.  If call A patches first, call B captures the
        patched function as 'original'.  When both finish, the restore
        in B's finally block sets _send_request to A's patched function,
        not the true original.
        """
        client = ytmusic_service._ytm
        true_original = client._send_request

        # Make get_playlist take a measurable amount of time so calls overlap.
        # (Patch _run — the ordered path bypasses the gated _call.)
        async def fake_call(func, *args, **kwargs):
            # Simulate the work the real _run would do on a thread.
            await asyncio.sleep(0.05)
            return {"tracks": []}

        with patch.object(ytmusic_service, "_run", side_effect=fake_call):
            # Run two concurrent calls with order= so both trigger the
            # monkey-patch path.
            await asyncio.gather(
                ytmusic_service.get_playlist("PL1", order="a_to_z"),
                ytmusic_service.get_playlist("PL2", order="recently_added"),
            )

        assert client._send_request is true_original, (
            "After concurrent get_playlist(order=...) calls, client._send_request "
            "must be restored to the true original. Got a stacked/leaked patch instead."
        )


class TestPatchWindowIsolation:
    """A patch window must be exclusive against ALL other client calls —
    a concurrent "browse" request would otherwise get the sort params
    injected into its body."""

    async def test_normal_call_during_patch_window_is_not_polluted(self, ytmusic_service):
        """A normal call issued WHILE an ordered fetch is mid-flight must
        wait out the window and send an un-patched body."""
        client = ytmusic_service._ytm
        recorded: list[tuple[str, dict]] = []

        def real_send(endpoint, body, *a, **kw):
            recorded.append((endpoint, dict(body)))
            return {"contents": []}

        client._send_request = real_send

        entered = threading.Event()
        release = threading.Event()

        def fake_get_playlist(playlist_id, **kwargs):
            # Emulate ytmusicapi: route a browse request through _send_request,
            # then hold the window open until the test releases it.
            client._send_request("browse", {"browseId": playlist_id})
            entered.set()
            release.wait(timeout=5)
            return {"tracks": []}

        client.get_playlist = fake_get_playlist

        def normal_browse():
            client._send_request("browse", {"browseId": "NORMAL"})

        ordered = asyncio.create_task(ytmusic_service.get_playlist("PL1", order="a_to_z"))
        await asyncio.to_thread(entered.wait, 5)  # window open, fetch mid-flight
        normal = asyncio.create_task(ytmusic_service._call(normal_browse))
        await asyncio.sleep(0.1)  # pre-fix, the normal call runs here — patched
        release.set()
        await asyncio.gather(ordered, normal)

        ordered_bodies = [b for e, b in recorded if b.get("browseId") == "PL1"]
        normal_bodies = [b for e, b in recorded if b.get("browseId") == "NORMAL"]
        assert ordered_bodies and "params" in ordered_bodies[0]  # sanity: patch worked
        assert normal_bodies == [{"browseId": "NORMAL"}], (
            "sort params leaked into a concurrent unrelated browse request"
        )

    async def test_patch_window_waits_for_inflight_calls(self, ytmusic_service):
        """An ordered fetch must not patch _send_request while a normal call
        that captured the un-patched client is still in flight."""
        client = ytmusic_service._ytm
        recorded: list[tuple[str, dict]] = []

        def real_send(endpoint, body, *a, **kw):
            recorded.append((endpoint, dict(body)))
            return {"contents": []}

        client._send_request = real_send

        started = threading.Event()
        normal_release = threading.Event()
        ordered_release = threading.Event()

        def fake_get_playlist(playlist_id, **kwargs):
            client._send_request("browse", {"browseId": playlist_id})
            ordered_release.wait(timeout=5)  # hold the window open
            return {"tracks": []}

        client.get_playlist = fake_get_playlist

        def slow_normal():
            started.set()
            normal_release.wait(timeout=5)  # ordered call arrives while we sleep
            client._send_request("browse", {"browseId": "NORMAL"})

        normal = asyncio.create_task(ytmusic_service._call(slow_normal))
        await asyncio.to_thread(started.wait, 5)
        ordered = asyncio.create_task(ytmusic_service.get_playlist("PL1", order="a_to_z"))
        # Pre-fix the patch lands here, mid-normal-call; post-fix the ordered
        # call is parked waiting for the normal call to drain.
        await asyncio.sleep(0.1)
        normal_release.set()
        await normal  # normal sends its browse body now
        ordered_release.set()
        await ordered

        normal_bodies = [b for e, b in recorded if b.get("browseId") == "NORMAL"]
        assert normal_bodies == [{"browseId": "NORMAL"}], (
            "patch window opened while a normal call was in flight"
        )


# ── The patch-window bookkeeping engines ──────────────────────────────


class TestPatchWindowBookkeeping:
    """The in-flight admission/release engines behind ``_call``'s gate,
    on whichever backend is loaded."""

    def test_enter_bumps_the_count(self):
        assert _patch_window_enter(0) == 1
        assert _patch_window_enter(1) == 2
        assert _patch_window_enter(5) == 6

    def test_exit_steps_down_without_draining_above_zero(self):
        assert _patch_window_exit(3) == (2, False)
        assert _patch_window_exit(2) == (1, False)

    def test_exit_at_one_drains(self):
        assert _patch_window_exit(1) == (0, True)

    def test_exit_below_zero_keeps_counting(self):
        # The reference has no floor: a stray exit below zero keeps
        # subtracting (and never names the drain) — pinned so the
        # engines cannot silently add a clamp.
        assert _patch_window_exit(0) == (-1, False)
        assert _patch_window_exit(-1) == (-2, False)

    def test_the_bump_is_cpythons_own_addition(self):
        assert _patch_window_enter(2.5) == 3.5
        assert _patch_window_enter(True) == 2
        assert _patch_window_enter(False) == 1
        assert _patch_window_enter(-1) == 0
        assert _patch_window_enter(10**20) == 10**20 + 1

    def test_the_step_is_cpythons_own_subtraction(self):
        assert _patch_window_exit(2.5) == (1.5, False)
        assert _patch_window_exit(1.5) == (0.5, False)
        assert _patch_window_exit(True) == (0, True)
        assert _patch_window_exit(False) == (-1, False)
        assert _patch_window_exit(10**20) == (10**20 - 1, False)

    def test_non_numeric_counts_raise_the_operators_own_type_error(self):
        with pytest.raises(TypeError, match="unsupported operand type"):
            _patch_window_enter(None)
        with pytest.raises(TypeError, match="unsupported operand type"):
            _patch_window_exit(None)
        with pytest.raises(TypeError, match="unsupported operand type"):
            _patch_window_exit("2")
        with pytest.raises(TypeError, match="can only concatenate"):
            _patch_window_enter("2")
        with pytest.raises(TypeError, match="can only concatenate"):
            _patch_window_enter([1])


class TestCallInflightBookkeeping:
    """The engines wired through ``_call`` itself: a call counts itself
    in at the gate and out in the finally, and the event that parks a
    patch window sets only when the last concurrent call drains."""

    async def test_a_call_counts_itself_in_and_out(self, ytmusic_service):
        started = threading.Event()
        release = threading.Event()

        def slow_normal():
            assert ytmusic_service._inflight == 1
            assert not ytmusic_service._no_inflight.is_set()
            started.set()
            release.wait(timeout=5)
            return 42

        task = asyncio.create_task(ytmusic_service._call(slow_normal))
        await asyncio.to_thread(started.wait, 5)
        assert ytmusic_service._inflight == 1
        assert not ytmusic_service._no_inflight.is_set()
        release.set()
        assert await task == 42
        assert ytmusic_service._inflight == 0
        assert ytmusic_service._no_inflight.is_set()

    async def test_a_failing_call_still_counts_itself_out(self, ytmusic_service):
        def boom():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await ytmusic_service._call(boom)
        assert ytmusic_service._inflight == 0
        assert ytmusic_service._no_inflight.is_set()

    async def test_the_event_sets_only_when_the_last_call_drains(self, ytmusic_service):
        entered = [threading.Event(), threading.Event()]
        release = threading.Event()

        def make_call(tag):
            def run():
                entered[tag].set()
                release.wait(timeout=5)
                return tag

            return run

        first = asyncio.create_task(ytmusic_service._call(make_call(0)))
        second = asyncio.create_task(ytmusic_service._call(make_call(1)))
        await asyncio.to_thread(entered[0].wait, 5)
        await asyncio.to_thread(entered[1].wait, 5)
        assert ytmusic_service._inflight == 2
        assert not ytmusic_service._no_inflight.is_set()
        release.set()
        results = await asyncio.gather(first, second)
        assert sorted(results) == [0, 1]
        # Whichever call left first stepped down to 1 without setting
        # the event; only the last one's exit drained to zero.
        assert ytmusic_service._inflight == 0
        assert ytmusic_service._no_inflight.is_set()


# ── native vs fallback parity ─────────────────────────────────────────

# Each case travels over a JSON bridge to a child interpreter with the
# extension import blocked; the engines are pure arithmetic over their
# single argument, so the cases span the numeric shapes (int, float,
# bool, BigInt, negative), the drained boundary, and the non-numeric
# shapes whose TypeError messages the operators themselves spell.
_PARITY_CASES = {
    "enter": [0, 1, 2, 5, -1, 2.5, True, False, 10**20, "2", None, [1]],
    "exit": [3, 1, 0, 2, -1, 2.5, 1.5, True, False, 10**20, "2", None, [1]],
}


def _outcome(name, arg):
    """One case's outcome on whichever backend this interpreter loaded."""
    try:
        if name == "enter":
            result = _patch_window_enter(arg)
        elif name == "exit":
            result = _patch_window_exit(arg)
        else:
            raise AssertionError(name)
    except BaseException as exc:  # noqa: BLE001 - repr'd for comparison
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, tuple):
        return {"ok": list(result)}
    return {"ok": result}


_CHILD_PROG = """
import json, sys

sys.modules["ytm_player.utils._ytm_player_native"] = None

from ytm_player.services import ytmusic


def outcome(name, arg):
    try:
        if name == "enter":
            result = ytmusic._patch_window_enter(arg)
        elif name == "exit":
            result = ytmusic._patch_window_exit(arg)
        else:
            raise AssertionError(name)
    except BaseException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, tuple):
        return {"ok": list(result)}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {name: [outcome(name, arg) for arg in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the patch-window bookkeeping
    engines must answer every case exactly like the pure-Python
    references run in a child interpreter with the extension import
    blocked — same counts, same drained verdicts, same errors with the
    same messages — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_bookkeeping_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = {
            name: [_outcome(name, arg) for arg in cases] for name, cases in _PARITY_CASES.items()
        }
        assert set(native) == set(fallback) == set(_PARITY_CASES)
        for name, cases in _PARITY_CASES.items():
            for case, n, f in zip(cases, native[name], fallback[name]):
                assert n == f, (
                    f"native and fallback {name} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
