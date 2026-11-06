"""Tests for the shuffle-preferences LRU engine.

The engine behind ``ShufflePreferences``: the ordered
``{context_id: shuffle}`` map — the get promotion, the set with its
move_to_end and evict-while-over walk, the hand-edited-file ingestion
(the dict gate, the str-key filter, the bool(v) truthiness coercion)
and the snapshot + ``json.dumps(..., indent=2)`` serialisation behind
``_save`` — plus the disk-I/O engines ``_read_prefs_file`` /
``_write_prefs_file`` behind ``_load`` / ``_save`` (the file read +
``json.loads`` parse, and the parent-mkdir / tmp-sibling / write /
replace atomic write). The pure-Python references live in
``services/shuffle_prefs.py`` (``_ShufflePrefsCore`` and the module
transcriptions); when the Rust extension is built the same engines run
natively (``_ytm_player_native.ShufflePrefsCore``,
``shuffle_prefs_read_file`` and ``shuffle_prefs_write_file``).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) operation for
operation, so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.services import shuffle_prefs as _sp
from ytm_player.services.shuffle_prefs import _MAX_ENTRIES, ShufflePreferences

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _make_core(max_entries: int):
    """Build the engine under test — the native Rust core when the
    extension is built, the pure-Python transcription otherwise."""
    if _HAVE_NATIVE:
        return _ytm_player_native.ShufflePrefsCore(max_entries)
    return _sp._ShufflePrefsCore(max_entries)


# ── get / set / clear ─────────────────────────────────────────────


class TestSetGet:
    def test_set_then_get_roundtrip(self):
        core = _make_core(10)
        core.set("PLABCD", True)
        assert core.get("PLABCD") is True

    def test_set_false_roundtrip(self):
        core = _make_core(10)
        core.set("MPREb_xyz", False)
        assert core.get("MPREb_xyz") is False

    def test_overwrite_updates_in_place(self):
        core = _make_core(10)
        core.set("PLABCD", True)
        core.set("PLABCD", False)
        assert core.get("PLABCD") is False
        assert core.length() == 1

    def test_miss_is_none(self):
        core = _make_core(10)
        core.set("real", True)
        assert core.get("never-set") is None

    def test_clear_empties(self):
        core = _make_core(10)
        core.set("a", True)
        core.set("b", False)
        core.clear()
        assert core.length() == 0
        assert core.get("a") is None


# ── LRU order ─────────────────────────────────────────────────────


class TestLruOrder:
    def test_set_promotes_existing_to_most_recent(self):
        core = _make_core(3)
        core.set("a", True)
        core.set("b", True)
        core.set("c", True)
        # Re-set "a" — it becomes most-recent, so "b" is now oldest.
        core.set("a", False)
        core.set("d", True)  # evicts b
        assert core.get("b") is None
        assert core.get("a") is False
        assert core.get("c") is True
        assert core.get("d") is True

    def test_get_promotes_hit_to_most_recent(self):
        core = _make_core(3)
        core.set("a", True)
        core.set("b", True)
        core.set("c", True)
        # Touch "a" — it becomes most-recent, so "b" is now oldest.
        assert core.get("a") is True
        core.set("d", True)  # evicts b
        assert core.get("b") is None
        assert core.get("a") is True

    def test_miss_does_not_reorder(self):
        core = _make_core(3)
        core.set("a", True)
        core.set("b", True)
        core.set("c", True)
        assert core.get("nope") is None
        core.set("d", True)  # still evicts a — the miss promoted nothing
        assert core.get("a") is None
        assert core.get("b") is True

    def test_evict_while_over_walks(self):
        core = _make_core(2)
        # One set that jumps the cap by more than one entry: the walk
        # evicts until the cap holds (the reference's while, not a
        # single if).
        for i in range(5):
            core.set(f"k{i}", True)
        assert core.length() == 2
        assert core.get("k3") is True
        assert core.get("k4") is True

    def test_full_cap_flow_matches_the_reference_constant(self):
        # The full-scale flow the facade runs: cap at _MAX_ENTRIES.
        core = _make_core(_MAX_ENTRIES)
        for i in range(_MAX_ENTRIES + 5):
            core.set(f"key-{i}", i % 2 == 0)
        for i in range(5):
            assert core.get(f"key-{i}") is None
        assert core.get(f"key-{_MAX_ENTRIES + 4}") is not None

    def test_lru_order_survives_the_json_roundtrip(self):
        # The dump order IS the LRU order — least-recent first.
        core = _make_core(10)
        core.set("a", True)
        core.set("b", True)
        core.set("c", False)
        core.get("a")  # promote a
        assert json.loads(core.dump_json()) == {"b": True, "c": False, "a": True}


# ── load — the hand-edited-file ingestion ──────────────────────────


class TestLoad:
    def test_non_dict_leaves_state_untouched(self):
        core = _make_core(10)
        core.set("a", True)
        core.load(["not", "a", "dict"])
        core.load("string")
        core.load(None)
        assert core.get("a") is True
        assert core.length() == 1

    def test_ingestion_replaces_state(self):
        core = _make_core(10)
        core.set("old", True)
        core.load({"new": False})
        assert core.get("old") is None
        assert core.get("new") is False

    def test_non_string_keys_are_dropped(self):
        core = _make_core(10)
        core.load({1: True, "good": True, ("t",): True, None: True})
        assert core.length() == 1
        assert core.get("good") is True

    def test_values_coerce_through_python_truthiness(self):
        core = _make_core(10)
        core.load(
            {
                "truthy-string": "false",  # a non-empty string is truthy!
                "empty-string": "",
                "zero": 0,
                "one": 1,
                "empty-list": [],
                "list": [0],
                "empty-dict": {},
                "none": None,
                "float-zero": 0.0,
                "float": 0.5,
            }
        )
        assert core.get("truthy-string") is True
        assert core.get("empty-string") is False
        assert core.get("zero") is False
        assert core.get("one") is True
        assert core.get("empty-list") is False
        assert core.get("list") is True
        assert core.get("empty-dict") is False
        assert core.get("none") is False
        assert core.get("float-zero") is False
        assert core.get("float") is True

    def test_non_str_key_skips_before_its_value_is_coerced(self):
        # The comprehension's filter runs before bool(v) — a value whose
        # truthiness raises under a non-str key never runs at all.
        class Raising:
            def __bool__(self) -> bool:
                raise RuntimeError("coerced")

        core = _make_core(10)
        core.load({1: Raising(), "good": True})
        assert core.get("good") is True

    def test_mid_walk_failure_leaves_old_state_intact(self):
        class Raising:
            def __bool__(self) -> bool:
                raise RuntimeError("coerced")

        core = _make_core(10)
        core.set("old", True)
        with pytest.raises(RuntimeError):
            core.load({"a": True, "bad": Raising(), "b": False})
        # The whole mapping is built before the swap — the old state
        # survives a mid-walk failure exactly like the reference's
        # partially-built (and then discarded) OrderedDict.
        assert core.get("old") is True
        assert core.get("a") is None

    def test_duplicate_keys_keep_the_last(self):
        # Not reachable through json.loads (which already resolves
        # duplicates), but the dict contract the engine ingests.
        core = _make_core(10)
        core.load({"a": True})
        assert core.get("a") is True


# ── dump_json — the _save serialisation ───────────────────────────


class TestDumpJson:
    def test_empty_state_dumps_the_empty_object(self):
        core = _make_core(10)
        assert core.dump_json() == "{}"

    def test_matches_stdlib_json_dumps_indent_2(self):
        core = _make_core(10)
        core.set("PLABCD", True)
        core.set("PLEFGH", False)
        assert core.dump_json() == json.dumps({"PLABCD": True, "PLEFGH": False}, indent=2)

    def test_escaping_matches_stdlib(self):
        # Keys with quotes, backslashes, control characters, non-ASCII
        # (ensure_ascii escapes) and astral-plane characters.
        keys = [
            'quote"key',
            "back\\slash",
            "new\nline",
            "tab\tkey",
            "日本語",
            "emoji🎧",
            "plain",
        ]
        core = _make_core(len(keys))
        for i, key in enumerate(keys):
            core.set(key, i % 2 == 0)
        expected = {key: i % 2 == 0 for i, key in enumerate(keys)}
        assert core.dump_json() == json.dumps(expected, indent=2)

    def test_lone_surrogate_key_escalates_like_stdlib(self):
        # A hand-edited file can carry a lone surrogate through
        # json.loads; the engine stores it, hits it and serialises it
        # back exactly the way CPython's ensure_ascii encoder does.
        core = _make_core(10)
        key = "\ud800"
        core.set(key, True)
        assert core.get(key) is True
        assert core.dump_json() == json.dumps({key: True}, indent=2)

    def test_int_key_stringifies_like_the_json_encoder(self):
        # The OrderedDict the reference replaced accepted any hashable
        # key; json.dumps coerces the numeric ones. The engine keeps
        # that contract: an int key dumps as its string form (and does
        # not collide with the same-named str key on the way back).
        core = _make_core(10)
        core.set(5, True)
        assert core.get(5) is True
        assert core.get("5") is None
        assert core.dump_json() == json.dumps({5: True}, indent=2)


# ── The OrderedDict lookup contract ────────────────────────────────


class TestLookupContract:
    def test_unhashable_get_raises_type_error(self):
        core = _make_core(10)
        with pytest.raises(TypeError):
            core.get(["unhashable"])

    def test_unhashable_set_raises_type_error(self):
        core = _make_core(10)
        with pytest.raises(TypeError):
            core.set(["unhashable"], True)
        assert core.length() == 0

    def test_identity_shortcut_hits_nan(self):
        nan = float("nan")
        core = _make_core(10)
        core.set(nan, True)
        assert core.get(nan) is True

    def test_two_distinct_nans_are_two_entries(self):
        core = _make_core(10)
        core.set(float("nan"), True)
        core.set(float("nan"), False)
        assert core.length() == 2

    def test_true_and_one_alias_one_entry(self):
        core = _make_core(10)
        core.set(True, True)
        core.set(1, False)
        assert core.length() == 1
        assert core.get(1) is False
        assert core.get(True) is False


# ── The facade over the engine ─────────────────────────────────────


class TestFacade:
    def test_save_writes_the_engine_serialisation(self, tmp_path):
        path = tmp_path / "prefs.json"
        prefs = ShufflePreferences(path)
        prefs.set("PLABCD", True)
        prefs.set("PLEFGH", False)
        assert path.read_text(encoding="utf-8") == json.dumps(
            {"PLABCD": True, "PLEFGH": False}, indent=2
        )

    def test_hand_edited_payload_loads_through_the_engine(self, tmp_path):
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"a": "truthy", "b": 0, "c": None, "d": []}), encoding="utf-8")
        prefs = ShufflePreferences(path)
        assert prefs.get("a") is True
        assert prefs.get("b") is False
        assert prefs.get("c") is False
        assert prefs.get("d") is False

    def test_engine_roundtrip_survives_a_new_instance(self, tmp_path):
        path = tmp_path / "prefs.json"
        a = ShufflePreferences(path)
        a.set("PLABCD", True)
        a.set("PLEFGH", False)
        b = ShufflePreferences(path)
        assert b.get("PLABCD") is True
        assert b.get("PLEFGH") is False


# ── Native parity ──────────────────────────────────────────────────

# The child interpreter runs this program with the case list on stdin
# and prints the reference engine's results as JSON. Each case is a
# list of operations against a fresh core of the given cap:
# ["set", key, value] / ["get", key] / ["clear"] / ["load", dict] /
# ["length"] / ["dump"].
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.shuffle_prefs import _ShufflePrefsCore

cases = json.load(sys.stdin)
out = []
for case in cases:
    cap, ops = case
    core = _ShufflePrefsCore(cap)
    steps = []
    for op in ops:
        kind = op[0]
        try:
            if kind == 'set':
                core.set(op[1], op[2])
                steps.append({'ok': None})
            elif kind == 'get':
                steps.append({'ok': core.get(op[1])})
            elif kind == 'clear':
                core.clear()
                steps.append({'ok': None})
            elif kind == 'load':
                core.load(op[1])
                steps.append({'ok': None})
            elif kind == 'length':
                steps.append({'ok': core.length()})
            elif kind == 'dump':
                steps.append({'ok': core.dump_json()})
            else:
                raise AssertionError(kind)
        except Exception as exc:
            steps.append({'error': type(exc).__name__})
    out.append(steps)
json.dump(out, sys.stdout)
"""


def _native_results(cases: list) -> list:
    """Run whichever engine the module loaded (native when built) over
    *cases*."""

    def run_case(case):
        cap, ops = case
        core = _make_core(cap)
        steps = []
        for op in ops:
            kind = op[0]
            try:
                if kind == "set":
                    core.set(op[1], op[2])
                    steps.append({"ok": None})
                elif kind == "get":
                    steps.append({"ok": core.get(op[1])})
                elif kind == "clear":
                    core.clear()
                    steps.append({"ok": None})
                elif kind == "load":
                    core.load(op[1])
                    steps.append({"ok": None})
                elif kind == "length":
                    steps.append({"ok": core.length()})
                elif kind == "dump":
                    steps.append({"ok": core.dump_json()})
                else:
                    raise AssertionError(kind)
            except Exception as exc:
                steps.append({"error": type(exc).__name__})
        return steps

    return [run_case(case) for case in cases]


def _fallback_results(cases: list) -> list:
    """Run the pure-Python reference engine over *cases* in a child
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


# Op sequences the two implementations must agree on, case for case:
# the plain roundtrip; the overwrite-promotion; the get-promotion
# steering the eviction; the miss leaving the order alone; the cap walk
# (jumping the cap by more than one entry); the clear; the ingestion
# replace; the truthiness coercion set; the non-str-key drop; the
# serialisation (plain, unicode-escaped, escaped-quotes, surrogate,
# empty); the int-key aliasing; the duplicate-key last-wins; and the
# unhashable-key refusals.
_PARITY_CASES: list = [
    [10, [["set", "a", True], ["get", "a"], ["length"], ["dump"]]],
    [10, [["set", "a", True], ["set", "b", False], ["set", "a", False], ["dump"]]],
    [
        3,
        [
            ["set", "a", True],
            ["set", "b", True],
            ["set", "c", True],
            ["get", "a"],
            ["set", "d", True],
            ["get", "b"],
            ["get", "a"],
            ["get", "c"],
            ["get", "d"],
            ["length"],
        ],
    ],
    [
        3,
        [
            ["set", "a", True],
            ["set", "b", True],
            ["set", "c", True],
            ["get", "nope"],
            ["set", "d", True],
            ["get", "a"],
            ["get", "b"],
        ],
    ],
    [2, [["set", f"k{i}", i % 2 == 0] for i in range(5)] + [["length"], ["dump"]]],
    [10, [["set", "a", True], ["set", "b", True], ["clear"], ["length"], ["dump"]]],
    [10, [["set", "old", True], ["load", {"new": False}], ["get", "old"], ["get", "new"]]],
    [
        10,
        [
            [
                "load",
                {
                    "truthy-string": "false",
                    "empty-string": "",
                    "zero": 0,
                    "one": 1,
                    "empty-list": [],
                    "list": [0],
                    "empty-dict": {},
                    "none": None,
                    "float-zero": 0.0,
                    "float": 0.5,
                },
            ],
            ["get", "truthy-string"],
            ["get", "empty-string"],
            ["get", "zero"],
            ["get", "one"],
            ["get", "empty-list"],
            ["get", "list"],
            ["get", "empty-dict"],
            ["get", "none"],
            ["get", "float-zero"],
            ["get", "float"],
        ],
    ],
    [10, [["load", {"good": True, "weird": 1}], ["get", "good"], ["length"]]],
    [10, [["set", 'quote"key', True], ["set", "back\\slash", False], ["dump"]]],
    [10, [["set", "日本語", True], ["set", "emoji🎧", False], ["dump"]]],
    [10, [["set", "\ud800", True], ["get", "\ud800"], ["dump"]]],
    [10, [["length"], ["dump"]]],
    [10, [["set", "a", True], ["get", 5], ["set", 5, True], ["get", 5], ["get", "5"], ["dump"]]],
    [10, [["load", {"a": True}], ["get", "a"]]],
    [10, [["set", "a", True], ["load", {"b": False}], ["length"], ["dump"]]],
]


class TestNativeParity:
    """When the Rust extension is built, the native LRU engine must
    answer every operation exactly like the pure-Python reference run
    in a child interpreter with the extension import blocked — the
    promotions, the evictions, the ingestions, the serialisations and
    the unhashable refusals included — so the two implementations
    cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _native_results(_PARITY_CASES)
        fallback = _fallback_results(_PARITY_CASES)
        for case, native_steps, fallback_steps in zip(_PARITY_CASES, native, fallback):
            assert native_steps == fallback_steps, (
                f"native and fallback shuffle-prefs results differ for case {case}"
            )


# ── file engines: read + parse, atomic write ─────────────────────────


class TestReadPrefsFile:
    """The `_read_prefs_file` engine — the file read + `json.loads`
    parse behind `ShufflePreferences._load`, on whichever backend is
    loaded (the native `shuffle_prefs_read_file` when the extension is
    built, the pure-Python transcription otherwise)."""

    def test_parses_dict_payload(self, tmp_path):
        target = tmp_path / "shuffle_prefs.json"
        target.write_text('{\n  "PLabc": true\n}', encoding="utf-8")
        assert _sp._read_prefs_file(target) == {"PLabc": True}

    def test_parses_non_dict_json_shapes(self, tmp_path):
        # The engine answers whatever json.loads answers; the dict gate
        # lives in the core's ingestion, not in the read.
        for text, expected in [
            ("[1, 2]", [1, 2]),
            ('"plain"', "plain"),
            ("null", None),
            ("3.5", 3.5),
            ("42", 42),
            ("NaN", float("nan")),
            ("Infinity", float("inf")),
            ("-Infinity", float("-inf")),
            ('  {"k": "v"}  ', {"k": "v"}),
            ("1e999", float("inf")),
        ]:
            target = tmp_path / "prefs.json"
            target.write_text(text, encoding="utf-8")
            got = _sp._read_prefs_file(target)
            if isinstance(got, float) and got != got:
                assert expected != expected  # NaN identity
            else:
                assert got == expected, text

    def test_parses_unicode_payload(self, tmp_path):
        target = tmp_path / "prefs.json"
        target.write_text('{"日本語": true, "emoji🎧": false}', encoding="utf-8")
        assert _sp._read_prefs_file(target) == {"日本語": True, "emoji🎧": False}

    def test_bad_json_raises_stdlib_shaped_decode_error(self, tmp_path):
        import json as _json

        for text in ["{bad}", "", "[1,", "{'x': 1}", '{"a": }', "tru"]:
            target = tmp_path / "prefs.json"
            target.write_text(text, encoding="utf-8")
            with pytest.raises(_json.JSONDecodeError) as engine_err:
                _sp._read_prefs_file(target)
            with pytest.raises(_json.JSONDecodeError) as stdlib_err:
                _json.loads(text)
            assert str(engine_err.value) == str(stdlib_err.value), text
            assert engine_err.value.pos == stdlib_err.value.pos, text

    def test_invalid_utf8_raises_unicode_decode_error(self, tmp_path):
        target = tmp_path / "prefs.json"
        target.write_bytes(b'{"a": "\xff"}')
        with pytest.raises(UnicodeDecodeError):
            _sp._read_prefs_file(target)

    def test_missing_file_raises_with_filename(self, tmp_path):

        target = tmp_path / "missing.json"
        with pytest.raises(OSError) as err:
            _sp._read_prefs_file(target)
        assert err.value.filename == str(target)

    def test_directory_path_raises_os_error(self, tmp_path):
        with pytest.raises(OSError):
            _sp._read_prefs_file(tmp_path)

    def test_stub_read_text_answering_bytes_parses_like_json_loads(self):
        # The transcription is json.loads(path.read_text(...)) — a stub
        # whose read_text answers bytes keeps json.loads' bytes lane on
        # both backends.
        class StubPath:
            def read_text(self, encoding=None, **kwargs):
                assert encoding == "utf-8"
                return b'{"a": 1}'

        assert _sp._read_prefs_file(StubPath()) == {"a": 1}

    def test_stub_read_text_error_propagates(self):
        class StubPath:
            def read_text(self, encoding=None, **kwargs):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            _sp._read_prefs_file(StubPath())


class TestWritePrefsFile:
    """The `_write_prefs_file` engine — the atomic write behind
    `ShufflePreferences._save`, on whichever backend is loaded (the
    native `shuffle_prefs_write_file` when the extension is built, the
    pure-Python transcription otherwise)."""

    def test_writes_content_and_leaves_no_tmp(self, tmp_path):
        target = tmp_path / "shuffle_prefs.json"
        _sp._write_prefs_file(target, '{\n  "a": true\n}')
        assert target.read_text(encoding="utf-8") == '{\n  "a": true\n}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ["shuffle_prefs.json"]

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "prefs.json"
        _sp._write_prefs_file(target, "1")
        assert target.read_text(encoding="utf-8") == "1"

    def test_overwrites_existing(self, tmp_path):
        target = tmp_path / "prefs.json"
        _sp._write_prefs_file(target, "first")
        _sp._write_prefs_file(target, "second")
        assert target.read_text(encoding="utf-8") == "second"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["prefs.json"]

    def test_tmp_sibling_is_suffix_plus_tmp(self, tmp_path):
        # The sibling is <name>.json.tmp (the path's own suffix kept),
        # pinned by blocking that exact name with a directory: the
        # write must fail against it, proving which sibling it used.
        target = tmp_path / "prefs.json"
        (tmp_path / "prefs.json.tmp").mkdir()
        with pytest.raises(OSError):
            _sp._write_prefs_file(target, "x")

    def test_edge_names_get_their_tmp_sibling(self, tmp_path):
        for name in ("no_suffix", ".hidden", "a.b.c"):
            case_dir = tmp_path / name.replace(".", "_")
            target = case_dir / name
            _sp._write_prefs_file(target, "x")
            assert target.read_text(encoding="utf-8") == "x"
            # Only the final file remains — the .tmp sibling (whatever
            # pathlib derived it to be) was consumed by the replace.
            assert [p.name for p in sorted(case_dir.iterdir())] == [name]

    def test_write_error_propagates(self, tmp_path):
        target = tmp_path / "prefs.json"
        (tmp_path / "prefs.json.tmp").mkdir()
        with pytest.raises(OSError):
            _sp._write_prefs_file(target, "x")
        # The final file was never created.
        assert not target.exists()

    def test_drives_the_path_objects_own_protocol(self, tmp_path):
        # The engine drives read/write through the path object's own
        # methods, so a recording Path subclass steers both backends
        # identically — the same contract the cache walk engines keep.
        from pathlib import Path

        calls: list[str] = []

        class RecordingPath(Path):
            def read_text(self, encoding=None, **kwargs):
                calls.append("read_text")
                return Path.read_text(self, encoding=encoding, **kwargs)

            def write_text(self, data, encoding=None, **kwargs):
                calls.append("write_text")
                return Path.write_text(self, data, encoding=encoding, **kwargs)

            def replace(self, target):
                calls.append("replace")
                return Path.replace(self, target)

        target = RecordingPath(tmp_path / "prefs.json")
        _sp._write_prefs_file(target, '{"a": true}')
        assert "write_text" in calls
        assert "replace" in calls
        assert target.read_text(encoding="utf-8") == '{"a": true}'
        assert _sp._read_prefs_file(target) == {"a": True}
        assert "read_text" in calls

    def test_stub_with_suffix_error_propagates(self):
        class StubParent:
            def mkdir(self, **kwargs):
                return None

        class StubPath:
            parent = StubParent()

            @property
            def suffix(self):
                return ".json"

            def with_suffix(self, new_suffix):
                raise ValueError("Invalid suffix %r" % (new_suffix,))

        with pytest.raises(ValueError, match="Invalid suffix"):
            _sp._write_prefs_file(StubPath(), "x")


# ── file engines: native vs fallback parity ──────────────────────────

# Op sequences the two file-engine implementations must agree on, case
# for case, over a fresh directory each: the read across the JSON
# payload shapes (dicts, non-dicts, the NaN/Infinity spellings, the
# big-int and inf exponents, padded whitespace, unicode), the read
# rejections (bad JSON shapes, an empty file, invalid UTF-8 bytes, a
# directory, a missing file) and the write (plain, nested parents,
# suffixless/hidden/multi-dot names, the overwrite, the tmp sibling
# blocked by a directory).
_FILE_PARITY_CASES: list = [
    [["preset", "prefs.json", '{"a": true}'], ["read", "prefs.json"]],
    [["preset", "prefs.json", "[1, 2]"], ["read", "prefs.json"]],
    [["preset", "prefs.json", '"str"'], ["read", "prefs.json"]],
    [["preset", "prefs.json", "null"], ["read", "prefs.json"]],
    [["preset", "prefs.json", "NaN"], ["read", "prefs.json"]],
    [["preset", "prefs.json", "Infinity"], ["read", "prefs.json"]],
    [["preset", "prefs.json", "-Infinity"], ["read", "prefs.json"]],
    [["preset", "prefs.json", "1e999"], ["read", "prefs.json"]],
    [["preset", "prefs.json", "123456789012345678901234567890"], ["read", "prefs.json"]],
    [["preset", "prefs.json", '  {"k": "v"}  '], ["read", "prefs.json"]],
    [["preset", "prefs.json", '{"日本語": true, "emoji🎧": false}'], ["read", "prefs.json"]],
    [["preset", "prefs.json", '{"a": "\\u00e9"}'], ["read", "prefs.json"]],
    [["preset", "prefs.json", "{bad}"], ["read", "prefs.json"]],
    [["preset", "prefs.json", ""], ["read", "prefs.json"]],
    [["preset", "prefs.json", "[1,"], ["read", "prefs.json"]],
    [["preset", "prefs.json", "{'x': 1}"], ["read", "prefs.json"]],
    [["preset", "prefs.json", '{"a": }'], ["read", "prefs.json"]],
    [["preset", "prefs.json", "<undecodable>"], ["read", "prefs.json"]],
    [["preset", "<dir>", ""], ["read", "<dir>"]],
    [["read", "missing.json"]],
    [
        ["preset", "prefs.json", "old"],
        ["write", "prefs.json", "new"],
        ["read", "prefs.json"],
        ["tree"],
    ],
    [["write", "nested/deeper/prefs.json", "1"], ["tree"]],
    [["write", "no_suffix", "x"], ["tree"]],
    [["write", ".hidden", "x"], ["tree"]],
    [["write", "a.b.c", "x"], ["tree"]],
    [["preset", "prefs.json.tmp", "<dir>"], ["write", "prefs.json", "x"], ["tree"]],
]

_FILE_CHILD_PROG = """
import json, pathlib, sys
sys.modules["ytm_player.utils._ytm_player_native"] = None
from ytm_player.services.shuffle_prefs import _read_prefs_file, _write_prefs_file

root = pathlib.Path(sys.argv[1])
out = []
for op in json.load(sys.stdin):
    kind = op[0]
    if kind == "preset":
        name, payload = op[1], op[2]
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if payload == "<dir>":
            path.mkdir(parents=True, exist_ok=True)
        elif payload == "<undecodable>":
            path.write_bytes('{"a": "\\udcff"}'.encode("utf-8", "surrogateescape"))
        else:
            path.write_text(payload, encoding="utf-8")
        out.append(["preset"])
    elif kind == "read":
        try:
            out.append(["ok", repr(_read_prefs_file(root / op[1]))])
        except BaseException as exc:
            out.append(["err", type(exc).__name__, str(exc).replace(str(root), "<root>")])
    elif kind == "write":
        try:
            _write_prefs_file(root / op[1], op[2])
            out.append(["ok"])
        except BaseException as exc:
            out.append(["err", type(exc).__name__, str(exc).replace(str(root), "<root>")])
    elif kind == "tree":
        entries = []
        for path in sorted(root.rglob("*")):
            rel = str(path.relative_to(root))
            if path.is_dir():
                entries.append([rel, "<dir>"])
            else:
                data = path.read_bytes()
                try:
                    entries.append([rel, data.decode("utf-8")])
                except UnicodeDecodeError:
                    entries.append([rel, repr(data)])
        out.append(["tree", entries])
    else:
        raise AssertionError(kind)
json.dump(out, sys.stdout)
"""


def _run_file_cases_native(cases: list, root) -> list:
    """Run *cases* against the native file engines in-process, each
    case's ops recorded exactly like the child program records them."""

    def materialize(path, payload):
        from pathlib import Path

        path = Path(root) / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if payload == "<dir>":
            path.mkdir(parents=True, exist_ok=True)
        elif payload == "<undecodable>":
            path.write_bytes('{"a": "\udcff"}'.encode("utf-8", "surrogateescape"))
        else:
            path.write_text(payload, encoding="utf-8")

    out = []
    for op in cases:
        kind = op[0]
        if kind == "preset":
            materialize(op[1], op[2])
            out.append(["preset"])
        elif kind == "read":
            try:
                out.append(["ok", repr(_ytm_player_native.shuffle_prefs_read_file(root / op[1]))])
            except BaseException as exc:
                out.append(["err", type(exc).__name__, str(exc).replace(str(root), "<root>")])
        elif kind == "write":
            try:
                _ytm_player_native.shuffle_prefs_write_file(root / op[1], op[2])
                out.append(["ok"])
            except BaseException as exc:
                out.append(["err", type(exc).__name__, str(exc).replace(str(root), "<root>")])
        elif kind == "tree":
            entries = []
            for path in sorted((root).rglob("*")):
                rel = str(path.relative_to(root))
                if path.is_dir():
                    entries.append([rel, "<dir>"])
                else:
                    data = path.read_bytes()
                    try:
                        entries.append([rel, data.decode("utf-8")])
                    except UnicodeDecodeError:
                        entries.append([rel, repr(data)])
            out.append(["tree", entries])
        else:
            raise AssertionError(kind)
    return out


def _run_file_cases_fallback(cases: list, root) -> list:
    """Run *cases* against the pure-Python transcriptions in a child
    interpreter with the native extension import blocked."""
    result = subprocess.run(
        [sys.executable, "-c", _FILE_CHILD_PROG, str(root)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


class TestFileEngineParity:
    """When the Rust extension is built, the native file engines must
    answer every case exactly like the pure-Python transcriptions run
    in a child interpreter with the extension import blocked — the
    parsed payloads, the rejections with their error types and
    messages, the resulting file trees and the tmp-sibling blocking
    included — so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_file_engines_match_fallback(self, tmp_path):
        import shutil

        native_root = tmp_path / "native"
        fallback_root = tmp_path / "fallback"
        for case in _FILE_PARITY_CASES:
            native_root.mkdir()
            fallback_root.mkdir()
            try:
                native = _run_file_cases_native(case, native_root)
                fallback = _run_file_cases_fallback(case, fallback_root)
            finally:
                shutil.rmtree(native_root)
                shutil.rmtree(fallback_root)
            assert native == fallback, f"file-engine results differ for {case}"
