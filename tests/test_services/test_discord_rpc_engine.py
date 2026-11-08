"""Tests for the Discord presence-kwargs engine.

The pure core behind ``services/discord_rpc.py``'s presence refresh:
``_presence_kwargs``, the seven-key dict ``update`` hands to
``rpc.update(**kwargs)`` — the fallback-text arms (a falsy
title/artist/album answers ``Unknown Track`` / ``Unknown Artist`` /
``YouTube Music``), the 128-char cuts on the three text fields
(``large_image`` passes through unsliced), the ``int(start_time)``
timestamp, and the ``int(start_time + duration)`` end edge appended
only when ``duration > 0``. The engine runs with two interchangeable
backends: the pure-Python transcription in the module and, when the
Rust extension is built, the native implementation bound at the
module's tail. Every truthiness gate, slice, comparison, addition and
``int()`` cast goes through CPython's own protocols, so the bool
spellings and the error shapes — including the order the dict's own
arms surface them — stay the interpreter's.

The unit tests below assert the engine's contract on whichever
backend is loaded. When the extension is built, the parity tests
additionally compare the native engine against the pure-Python
reference (run in a child interpreter with the extension import
blocked) case for case — error shapes, the key order and the unsliced
protocol spellings included — so the two implementations cannot drift
apart. The wiring tests pin ``update``'s routing: the frozen clock
steers the start time, the engine's dict is the very dict the RPC
receives, and a patched engine name steers the assembly exactly like
the module tail's native binding does.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ytm_player.services import discord_rpc as discord_module
from ytm_player.services.discord_rpc import DiscordRPC, _presence_kwargs

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

_FROZEN_NOW = 2_000.0

_SIX_KEYS = ["activity_type", "details", "state", "large_image", "large_text", "start"]


# ── The assembly arms ─────────────────────────────────────────────────────


class TestPlainAssembly:
    def test_the_full_dict(self):
        kwargs = _presence_kwargs("Song", "Artist", "Album", 180, "https://t/1.jpg", 1000.7, "l")
        assert kwargs == {
            "activity_type": "l",
            "details": "Song",
            "state": "Artist",
            "large_image": "https://t/1.jpg",
            "large_text": "Album",
            "start": 1000,
            "end": 1180,
        }

    def test_start_truncates_toward_zero(self):
        assert _presence_kwargs("T", "A", "", 0, "", 1000.9, None)["start"] == 1000
        assert _presence_kwargs("T", "A", "", 0, "", -5.5, None)["start"] == -5

    def test_the_key_order_is_the_reference_s_own(self):
        assert list(_presence_kwargs("T", "A", "", 180, "", 1.0, None)) == [*_SIX_KEYS, "end"]
        assert list(_presence_kwargs("T", "A", "", 0, "", 1.0, None)) == _SIX_KEYS

    def test_activity_type_is_the_very_object(self):
        marker = object()
        kwargs = _presence_kwargs("T", "A", "", 0, "", 1.0, marker)
        assert kwargs["activity_type"] is marker


class TestFallbackArms:
    def test_empty_strings_answer_the_fallbacks(self):
        kwargs = _presence_kwargs("", "", "", 0, "", 1000.0, None)
        assert kwargs["details"] == "Unknown Track"
        assert kwargs["state"] == "Unknown Artist"
        assert kwargs["large_image"] == "ytm_icon"
        assert kwargs["large_text"] == "YouTube Music"

    def test_none_answers_the_fallbacks_too(self):
        kwargs = _presence_kwargs(None, None, None, 0, None, 1000.0, None)
        assert kwargs["details"] == "Unknown Track"
        assert kwargs["state"] == "Unknown Artist"
        assert kwargs["large_image"] == "ytm_icon"
        assert kwargs["large_text"] == "YouTube Music"

    def test_a_truthy_thumbnail_passes_through_unsliced(self):
        long_url = "u" * 300
        kwargs = _presence_kwargs("T", "A", "", 0, long_url, 1.0, None)
        assert kwargs["large_image"] == long_url


class TestTruncation:
    def test_a_200_char_title_cuts_at_128(self):
        title = "x" * 200
        kwargs = _presence_kwargs(title, "A", "", 0, "", 1.0, None)
        assert kwargs["details"] == title[:128]
        assert len(kwargs["details"]) == 128

    def test_exactly_128_stays_whole(self):
        title = "x" * 128
        assert _presence_kwargs(title, "", "", 0, "", 1.0, None)["details"] == title

    def test_129_cuts_to_128(self):
        title = "x" * 129
        assert len(_presence_kwargs(title, "", "", 0, "", 1.0, None)["details"]) == 128

    def test_the_cut_hits_state_and_large_text_too(self):
        artist = "y" * 200
        album = "z" * 200
        kwargs = _presence_kwargs("T", artist, album, 0, "", 1.0, None)
        assert kwargs["state"] == artist[:128]
        assert kwargs["large_text"] == album[:128]


class TestEndGate:
    def test_zero_duration_leaves_end_absent(self):
        assert "end" not in _presence_kwargs("T", "A", "", 0, "", 1.0, None)

    def test_negative_duration_leaves_end_absent(self):
        assert "end" not in _presence_kwargs("T", "A", "", -10, "", 1.0, None)

    def test_a_positive_duration_spells_the_far_edge(self):
        # int(100.25 + 180.5) == int(280.75) == 280.
        kwargs = _presence_kwargs("T", "A", "", 180.5, "", 100.25, None)
        assert kwargs["end"] == 280

    def test_true_duration_is_a_positive_one(self):
        # True > 0 is True, and 100.0 + True == 101.0 — the bool
        # spellings stay CPython's.
        kwargs = _presence_kwargs("T", "A", "", True, "", 100.0, None)
        assert kwargs["end"] == 101


class TestIntSpellings:
    def test_a_numeric_string_start_parses(self):
        assert _presence_kwargs("T", "A", "", 0, "", "1000", None)["start"] == 1000


class TestErrorShapes:
    """Each arm surfaces its own error first, in the dict's own order."""

    def test_a_truthy_non_subscriptable_title(self):
        with pytest.raises(TypeError, match="'int' object is not subscriptable"):
            _presence_kwargs(123, "A", "", 0, "", 1.0, None)

    def test_the_title_error_precedes_the_artist_error(self):
        with pytest.raises(TypeError, match="'int' object is not subscriptable"):
            _presence_kwargs(123, 456, "", 0, "", 1.0, None)

    def test_a_none_start_time_raises_ints_own_error(self):
        with pytest.raises(TypeError, match=r"int\(\) argument must be a string"):
            _presence_kwargs("T", "A", "", 0, "", None, None)

    def test_a_non_numeric_string_start_raises_valueerror(self):
        with pytest.raises(ValueError, match="invalid literal for int()"):
            _presence_kwargs("T", "A", "", 0, "", "abc", None)

    def test_a_string_duration_raises_the_gt_error(self):
        with pytest.raises(TypeError, match="'>' not supported"):
            _presence_kwargs("T", "A", "", "180", "", 1.0, None)

    def test_the_album_error_precedes_the_start_error(self):
        # large_text (789[:128]) is assembled before int(None).
        with pytest.raises(TypeError, match="'int' object is not subscriptable"):
            _presence_kwargs("T", "A", 789, 0, "", None, None)

    def test_the_start_error_precedes_the_end_gate(self):
        # int(None) raises before the duration > 0 comparison.
        with pytest.raises(TypeError, match=r"int\(\) argument must be a string"):
            _presence_kwargs("T", "A", "", "180", "", None, None)


# ── The update wiring ─────────────────────────────────────────────────────


def _install_fake_pypresence(monkeypatch):
    class FakePresence:
        instances: list[FakePresence] = []

        def __init__(self, client_id: str) -> None:
            self.client_id = client_id
            self.updates: list[dict] = []
            FakePresence.instances.append(self)

        async def connect(self) -> None:
            pass

        def close(self) -> None:
            pass

        async def update(self, **kwargs) -> None:
            self.updates.append(kwargs)

        async def clear(self) -> None:
            pass

    fake = ModuleType("pypresence")
    fake.AioPresence = FakePresence
    fake.ActivityType = SimpleNamespace(LISTENING="listening")
    monkeypatch.setitem(sys.modules, "pypresence", fake)
    return FakePresence


@pytest.fixture
def frozen_clock(monkeypatch):
    """Freeze discord_rpc's clock at 2000 — the module's own time
    read steers the start time on both backends identically."""
    monkeypatch.setattr(discord_module, "time", SimpleNamespace(time=lambda: _FROZEN_NOW))
    return _FROZEN_NOW


class TestUpdateWiring:
    async def test_update_hands_the_engine_s_dict_to_the_rpc(self, monkeypatch, frozen_clock):
        fake_presence = _install_fake_pypresence(monkeypatch)
        rpc = DiscordRPC(client_id="123")
        assert await rpc.connect() is True

        await rpc.update(
            title="Song",
            artist="Artist",
            album="Album",
            duration=180,
            position=30.5,
            thumbnail_url="https://t/1.jpg",
        )

        # start = int(2000 - 30.5) = int(1969.5); end = int(1969.5 + 180).
        assert fake_presence.instances[0].updates == [
            {
                "activity_type": "listening",
                "details": "Song",
                "state": "Artist",
                "large_image": "https://t/1.jpg",
                "large_text": "Album",
                "start": 1969,
                "end": 2149,
            }
        ]

    async def test_the_truncation_flows_end_to_end(self, monkeypatch, frozen_clock):
        fake_presence = _install_fake_pypresence(monkeypatch)
        rpc = DiscordRPC(client_id="123")
        await rpc.connect()

        await rpc.update(title="x" * 200, artist="Artist", duration=0)

        presence = fake_presence.instances[0].updates[0]
        assert presence["details"] == "x" * 128
        assert "end" not in presence
        assert presence["start"] == 2000

    async def test_a_patched_engine_name_steers_the_assembly(self, monkeypatch, frozen_clock):
        fake_presence = _install_fake_pypresence(monkeypatch)
        calls: list[tuple] = []
        sentinel = {"details": "sentinel"}

        def recording_engine(*args):
            calls.append(args)
            return sentinel

        monkeypatch.setattr(discord_module, "_presence_kwargs", recording_engine)

        rpc = DiscordRPC(client_id="123")
        await rpc.connect()
        await rpc.update(title="Song", artist="Artist", album="Album", duration=180)

        # (title, artist, album, duration, thumbnail_url, start_time,
        # activity_type) — the frozen clock's 2000.0 start and the fake
        # enum's "listening".
        assert calls == [("Song", "Artist", "Album", 180, "", 2000.0, "listening")]
        assert fake_presence.instances[0].updates == [sentinel]


# ── Native parity ─────────────────────────────────────────────────────────

# Each case: [title, artist, album, duration, thumbnail_url, start_time,
# activity_type] — the fields either JSON scalars or stand-in descriptors
# ({"nonsub": …} for a truthy non-subscriptable). The outcome is the
# kwargs dict as ordered (key, value) pairs, or the error's own shape.
_LONG = "q" * 200
_PARITY_CASES = [
    ["Song", "Artist", "Album", 180, "https://t/1.jpg", 1000.7, "listening"],
    ["", "", "", 0, "", 1000.0, None],
    [None, None, None, 0, None, 1000.0, None],
    [_LONG, _LONG, _LONG, 0, "u" * 300, 1000.0, None],
    ["Song", "Artist", "Album", -10, "u", 1000.0, None],
    ["Song", "Artist", "Album", 180.5, "u", 100.25, None],
    ["Song", "Artist", "Album", True, "u", 100.0, None],
    ["Song", "Artist", "Album", 0, "u", "1000", None],
    ["Song", "Artist", "Album", 180, "u", 0.0, "listening"],
    ["Song", "Artist", "Album", 180, "u", -5.5, None],
    [list(range(400)), "Artist", "Album", 0, "u", 1.0, None],
    [{"nonsub": 123}, "Artist", "Album", 0, "u", 1.0, None],
    ["Song", {"nonsub": 456}, "Album", 0, "u", 1.0, None],
    ["Song", "Artist", {"nonsub": 789}, 0, "u", 1.0, None],
    ["Song", "Artist", {"nonsub": 789}, 0, "u", None, None],
    ["Song", "Artist", "Album", "180", "u", 1.0, None],
    ["Song", "Artist", "Album", "180", "u", None, None],
    ["Song", "Artist", "Album", 0, "u", "abc", None],
    ["Song", "Artist", "Album", 0, "u", None, None],
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import discord_rpc


def make_arg(desc):
    if isinstance(desc, dict) and "nonsub" in desc:
        return desc["nonsub"]
    return desc


def outcome(case):
    args = [make_arg(value) for value in case]
    try:
        result = discord_rpc._presence_kwargs(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"ok": [[key, value] for key, value in result.items()]}


payload = json.load(sys.stdin)
out = [outcome(case) for case in payload]
json.dump(out, sys.stdout)
"""


def _make_arg(desc):
    """The argument stand-in rebuilt from the JSON case — the same
    shape the child spells inside its own program."""
    if isinstance(desc, dict) and "nonsub" in desc:
        return desc["nonsub"]
    return desc


def _native_outcomes():
    """The parent-side outcomes against the loaded (native) engine."""

    def outcome(case):
        args = [_make_arg(value) for value in case]
        try:
            result = discord_module._presence_kwargs(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"ok": [[key, value] for key, value in result.items()]}

    return [outcome(case) for case in _PARITY_CASES]


class TestNativeParity:
    """When the Rust extension is built, the native engine must answer
    exactly like the pure-Python reference — same dicts in the same
    key order, same errors with the same messages — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_presence_assembly_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes()
        for case, n, f in zip(_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback _presence_kwargs differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
