"""Tests for the CLI JSON serialisation engine.

``_json_dumps`` is the engine behind every JSON-emitting CLI command
(``ytm now``, ``ytm status``, ``ytm queue``, ``ytm search``,
``ytm history``, ``ytm stats``): the ``json.dumps(data, indent=...,
default=str)`` text ``_json_output`` prints. The pure-Python
transcription lives in ``cli.py``; when the Rust extension is built the
same engine runs natively (``_ytm_player_native.json_dumps``).

The unit tests below assert the engine's contract on whichever backend
is loaded, plus the ``_json_output`` glue (the compact flag, the pretty
default, the trailing-newline echo). When the extension is built, the
parity test additionally compares the native engine against the
pure-Python reference (run in a child interpreter with the extension
import blocked) case for case — including the error shapes, where
CPython's own messages must surface identically — so the two
implementations cannot drift apart, and the native-only class exercises
the shapes the JSON transport cannot carry (the randomized float zoo,
the lone surrogate, the deep-nesting totality and the default-call
marker lifetime).
"""

from __future__ import annotations

import datetime
import json
import random
import struct
import subprocess
import sys
from enum import IntEnum

import pytest
from click.testing import CliRunner

from ytm_player.cli import _json_dumps, main

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class Color(IntEnum):
    RED = 1


class MyList(list):
    pass


class MyDict(dict):
    def items(self):
        return [("z", 99)]


class MyInt(int):
    def __repr__(self):
        return "MYINT"


class MyStr(str):
    pass


class MyFloat(float):
    def __repr__(self):
        return "MYFLOAT"


class TestScalars:
    def test_null_bools(self):
        assert _json_dumps(None) == "null"
        assert _json_dumps([True, False, None]) == "[true, false, null]"

    def test_ints(self):
        assert _json_dumps(0) == "0"
        assert _json_dumps(-5) == "-5"
        assert _json_dumps(10**30) == "1" + "0" * 30  # bigints keep every digit

    def test_int_subclass_renders_as_its_number(self):
        # int.__repr__, never the subclass's own — the IntEnum comment.
        assert _json_dumps(Color.RED) == "1"
        assert _json_dumps({"c": Color.RED}) == '{"c": 1}'
        assert _json_dumps(MyInt(5)) == "5"

    def test_floats(self):
        assert _json_dumps(1.0) == "1.0"
        assert _json_dumps(90.0) == "90.0"
        assert _json_dumps(-0.0) == "-0.0"
        assert _json_dumps(0.1) == "0.1"
        assert _json_dumps(123.456) == "123.456"
        assert _json_dumps(1e15) == "1000000000000000.0"
        assert _json_dumps(1e16) == "1e+16"
        assert _json_dumps(1e-4) == "0.0001"
        assert _json_dumps(1e-5) == "1e-05"
        assert _json_dumps(1.5e-7) == "1.5e-07"
        assert _json_dumps(5e-324) == "5e-324"
        assert _json_dumps(1e308) == "1e+308"
        assert _json_dumps(0.30000000000000004) == "0.30000000000000004"

    def test_float_specials(self):
        assert _json_dumps(float("nan")) == "NaN"
        assert _json_dumps(float("inf")) == "Infinity"
        assert _json_dumps(float("-inf")) == "-Infinity"

    def test_float_subclass_renders_as_its_number(self):
        assert _json_dumps(MyFloat(1.5)) == "1.5"

    def test_float_tie_rounds_half_even(self):
        # The exact-midpoint doubles whose two shortest spellings both
        # round-trip — CPython's dtoa picks the even-ending one.
        assert _json_dumps(2163015292276093.2) == "2163015292276093.2"
        assert _json_dumps(79736654567269.12) == "79736654567269.12"
        assert _json_dumps(-1862136454571465.2) == "-1862136454571465.2"

    def test_str_escaping(self):
        assert _json_dumps("plain") == '"plain"'
        assert _json_dumps('a"b\\c') == '"a\\"b\\\\c"'
        assert _json_dumps("\b\t\n\f\r") == '"\\b\\t\\n\\f\\r"'
        assert _json_dumps("\x00\x1f") == '"\\u0000\\u001f"'
        assert _json_dumps("\x7f") == '"\\u007f"'  # DEL escapes too
        assert _json_dumps("k€\U0001f600") == '"k\\u20ac\\ud83d\\ude00"'
        assert _json_dumps("\U0001d11e") == '"\\ud834\\udd1e"'  # surrogate pair
        assert _json_dumps("ключ") == '"\\u043a\\u043b\\u044e\\u0447"'

    def test_str_subclass_renders_its_content(self):
        assert _json_dumps(MyStr("hi")) == '"hi"'


class TestContainers:
    def test_lists_and_tuples(self):
        assert _json_dumps([1, 2]) == "[1, 2]"
        assert _json_dumps((1, 2)) == "[1, 2]"
        assert _json_dumps(MyList([1, 2])) == "[1, 2]"
        assert _json_dumps([]) == "[]"
        assert _json_dumps(()) == "[]"

    def test_dicts(self):
        assert _json_dumps({"a": 1}) == '{"a": 1}'
        assert _json_dumps({}) == "{}"
        assert _json_dumps({"z": 1, "a": 2, "m": 3}) == '{"z": 1, "a": 2, "m": 3}'
        assert (
            _json_dumps({"ключ": "значение"})
            == '{"\\u043a\\u043b\\u044e\\u0447": "\\u0437\\u043d\\u0430\\u0447\\u0435\\u043d\\u0438\\u0435"}'
        )

    def test_dict_subclass_drives_items(self):
        assert _json_dumps(MyDict({"a": 1})) == '{"z": 99}'

    def test_nested(self):
        assert _json_dumps({"a": [1, {"b": []}]}) == '{"a": [1, {"b": []}]}'

    def test_key_coercion(self):
        assert _json_dumps({1: "x"}) == '{"1": "x"}'
        assert (
            _json_dumps({1.5: "x", True: "y", None: "z"})
            == '{"1.5": "x", "true": "y", "null": "z"}'
        )
        assert _json_dumps({10**40: "x"}) == '{"1' + "0" * 40 + '": "x"}'
        assert _json_dumps({Color.RED: "x"}) == '{"1": "x"}'
        assert _json_dumps({MyInt(7): "x"}) == '{"7": "x"}'
        assert _json_dumps({MyStr("k"): 1}) == '{"k": 1}'

    def test_special_float_keys(self):
        assert _json_dumps({float("nan"): 1, float("inf"): 2}) == '{"NaN": 1, "Infinity": 2}'

    def test_bad_key_raises(self):
        with pytest.raises(TypeError, match="keys must be str, int, float, bool or None"):
            _json_dumps({1j: "x"})


class TestIndent:
    def test_pretty_layout(self):
        assert _json_dumps({"a": [1, 2]}, indent=2) == '{\n  "a": [\n    1,\n    2\n  ]\n}'

    def test_empty_containers_stay_bare(self):
        assert _json_dumps([], indent=2) == "[]"
        assert _json_dumps({}, indent=2) == "{}"
        assert _json_dumps({"a": []}, indent=2) == '{\n  "a": []\n}'

    def test_indent_variants(self):
        assert _json_dumps([1], indent=0) == "[\n1\n]"
        assert _json_dumps([[1]], indent=4) == "[\n    [\n        1\n    ]\n]"

    def test_negative_indent_collapses(self):
        assert _json_dumps([1], indent=-5) == "[\n1\n]"


class TestDefault:
    def test_sets_bytes_datetimes_objects(self):
        assert _json_dumps({1, 2}) == '"{1, 2}"'
        assert _json_dumps(b"ab") == "\"b'ab'\""
        assert _json_dumps(datetime.datetime(2026, 1, 2, 3, 4, 5)) == '"2026-01-02 03:04:05"'
        assert _json_dumps(3 + 4j) == '"(3+4j)"'
        plain = object()
        assert _json_dumps(plain) == json.dumps(plain, default=str)

    def test_default_result_reencodes(self):
        # The converted value goes through the same dispatch — a
        # container-shaped conversion encodes as a container.
        class Box:
            def __str__(self):
                return "[1, 2]"

        assert _json_dumps(Box()) == '"[1, 2]"'

    def test_circular_reference(self):
        a = []
        a.append(a)
        with pytest.raises(ValueError, match="Circular reference detected"):
            _json_dumps(a)

        d = {}
        d["self"] = d
        with pytest.raises(ValueError, match="Circular reference detected"):
            _json_dumps(d)

    def test_shared_non_cyclic_references_encode(self):
        shared = {"x": 1}
        assert _json_dumps([shared, [shared]]) == '[{"x": 1}, [{"x": 1}]]'
        box = object()
        assert _json_dumps([box, box]) == json.dumps([box, box], default=str)

    def test_repeated_object_after_conversion(self):
        # The default-path marker is released once the conversion's
        # subtree finishes — the same object converts twice.
        box = object()
        twice = _json_dumps([box, box])
        assert twice == json.dumps([box, box], default=str)


class TestGlue:
    """``_json_output`` through the CLI's own command surface."""

    def test_pretty_default_two_space_indent(self, tmp_path, monkeypatch):
        db = tmp_path / "h.db"
        import sqlite3

        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE play_history (id INTEGER PRIMARY KEY, video_id TEXT, "
            "title TEXT, artist TEXT, album TEXT, duration_seconds INTEGER, "
            "listened_seconds INTEGER, source TEXT, played_at TEXT)"
        )
        conn.execute(
            "INSERT INTO play_history (video_id, title, artist, played_at) "
            "VALUES ('v1', 'T', 'A', '2026-01-01')"
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        result = CliRunner().invoke(main, ["history", "--limit", "1"])
        assert result.exit_code == 0, result.output
        assert result.output == (
            "[\n"
            "  {\n"
            '    "id": 1,\n'
            '    "video_id": "v1",\n'
            '    "title": "T",\n'
            '    "artist": "A",\n'
            '    "album": null,\n'
            '    "duration_seconds": null,\n'
            '    "listened_seconds": null,\n'
            '    "source": null,\n'
            '    "played_at": "2026-01-01"\n'
            "  }\n"
            "]\n"
        )

    def test_compact_flag(self, tmp_path, monkeypatch):
        import sqlite3

        db = tmp_path / "h.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE play_history (id INTEGER PRIMARY KEY, video_id TEXT, "
            "title TEXT, artist TEXT, album TEXT, duration_seconds INTEGER, "
            "listened_seconds INTEGER, source TEXT, played_at TEXT)"
        )
        conn.execute(
            "INSERT INTO play_history (video_id, title, artist, played_at) "
            "VALUES ('v1', 'T', 'A', '2026-01-01')"
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr("ytm_player.cli.HISTORY_DB", db)
        result = CliRunner().invoke(main, ["history", "--json", "--limit", "1"])
        assert result.exit_code == 0, result.output
        assert result.output.startswith('[{"id": 1,')
        assert "\n" not in result.output.strip()

    def test_echo_adds_the_trailing_newline(self):
        assert _json_dumps({"a": 1}) == '{"a": 1}'
        assert not _json_dumps({"a": 1}).endswith("\n")


# ── Native vs fallback parity ──────────────────────────────────────────

# The matrix below is shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable.
_JSON_PARITY_CASES = [
    None,
    True,
    False,
    0,
    -5,
    10**30,
    1.0,
    -0.0,
    0.1,
    90.5,
    123.456,
    1e15,
    1e16,
    1e-4,
    1e-5,
    1.5e-7,
    5e-324,
    1e308,
    0.30000000000000004,
    2163015292276093.2,
    79736654567269.12,
    "",
    "plain",
    'a"b\\c\nd\te\bf\fg\rh',
    "\x00\x1f\x7f",
    "k€\U0001f600",
    "ключ",
    [],
    {},
    [1, 2, [3, {"a": []}]],
    {"a": 1, "b": [1, 2]},
    {"z": 1, "a": 2, "m": 3},
    {1: "x", 1.5: None, None: {}},
    {10**40: "big"},
    [[[[[1]]]]],
    {"nested": {"deeper": {"deepest": [True, False, None]}}},
]

_JSON_PARITY_INDENTS = [None, 0, 2, 4]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _json_dumps

payload = json.load(sys.stdin)
json.dump([_json_dumps(data, indent=indent) for data, indent in payload], sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native engine must render
    exactly like the pure-Python reference — byte for byte, every
    indent mode — for every case, so the two implementations cannot
    drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_json_dumps_matches_fallback(self):
        cases = [(data, indent) for data in _JSON_PARITY_CASES for indent in _JSON_PARITY_INDENTS]
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_json_dumps(data, indent=indent) for data, indent in cases]
        for (data, indent), n, f in zip(cases, native, fallback):
            assert n == f, (
                f"native and fallback serialisations differ for {data!r} "
                f"indent={indent!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_error_shapes_match(self):
        # The error matrix: keys CPython rejects, and the cycle. The
        # child builds the cyclic payload itself — it has no JSON
        # spelling.
        prog = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _json_dumps


def outcome(fn):
    try:
        return {"ok": fn()}
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


outcomes = [
    outcome(lambda: _json_dumps({complex(1, 2): "x"})),
    outcome(lambda: _json_dumps({(1, 2): "x"})),
    outcome(lambda: _json_dumps({frozenset({"a"}): "x"})),
]
cyclic = []
cyclic.append(cyclic)
outcomes.append(outcome(lambda: _json_dumps(cyclic)))
cyc_dict = {}
cyc_dict["self"] = cyc_dict
outcomes.append(outcome(lambda: _json_dumps(cyc_dict)))
json.dump(outcomes, sys.stdout)
"""
        result = subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, check=True
        )
        fallback = json.loads(result.stdout)

        def outcome(fn):
            try:
                return {"ok": fn()}
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                return {"error": f"{type(exc).__name__}: {exc}"}

        cyclic = []
        cyclic.append(cyclic)
        cyc_dict = {}
        cyc_dict["self"] = cyc_dict
        native = [
            outcome(lambda: _json_dumps({complex(1, 2): "x"})),
            outcome(lambda: _json_dumps({(1, 2): "x"})),
            outcome(lambda: _json_dumps({frozenset({"a"}): "x"})),
            outcome(lambda: _json_dumps(cyclic)),
            outcome(lambda: _json_dumps(cyc_dict)),
        ]
        assert native == fallback


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
class TestNativeOnly:
    """The shapes the JSON-transport parity matrix cannot carry, pinned
    directly against the stdlib on this interpreter.
    """

    def test_absurd_indent_raises_memory_error(self):
        # The native engine's totality guard: an indent too wide to pad
        # raises immediately instead of building a multi-gigabyte pad.
        # (The stdlib fallback's own behaviour at this width is
        # machine-dependent — allocation or MemoryError — so this pin is
        # native-only.)
        with pytest.raises(MemoryError):
            _json_dumps([1], indent=10**9)

    def test_randomized_float_zoo(self):
        # Random bit patterns (subnormals, specials, huge, tiny), the
        # arithmetic combinations, and the halfway ties — every float
        # must serialise exactly like json.dumps would.
        rng = random.Random(20261102)
        for _ in range(4000):
            v = struct.unpack("<d", struct.pack("<Q", rng.getrandbits(64)))[0]
            assert _json_dumps(v) == json.dumps(v, default=str), v.hex()
        for _ in range(1000):
            v = rng.uniform(-1e6, 1e6) * 10 ** rng.randint(-30, 30)
            assert _json_dumps(v) == json.dumps(v, default=str), v.hex()
        for i in range(500):
            v = float(rng.getrandbits(40)) + 0.5
            assert _json_dumps(v) == json.dumps(v, default=str), v.hex()

    def test_lone_surrogate_escapes(self):
        # A lone surrogate has no UTF-8 spelling, but the reference's
        # code-point walk escapes it as \\ud800 all the same.
        assert _json_dumps("\ud800") == json.dumps("\ud800")
        assert _json_dumps(["\ud800", "\udfff"]) == json.dumps(["\ud800", "\udfff"])
        assert _json_dumps("a\ud800b") == json.dumps("a\ud800b")

    def test_deep_nesting_is_total(self):
        # The iterative traversal holds no Rust recursion — a payload
        # far past the interpreter's own recursion budget encodes.
        deep = 1
        for _ in range(5000):
            deep = [deep]
        text = _json_dumps(deep)
        assert text.startswith("[" * 100)
        assert text.endswith("]" * 100)
        assert len(text) == 2 * 5000 + 1  # the brackets plus the 1

    def test_default_marker_held_during_conversion(self):
        # A str() whose result re-encloses its own object is the cycle
        # the reference reports — the conversion's marker is held for
        # the converted subtree's lifetime, exactly like the recursive
        # call. (The trapping class keeps builtins.str a type, which
        # the stdlib encoder's own isinstance checks require, and the
        # assertion works on args — every str() in flight is the trap
        # itself while the patch is live.)
        import builtins

        class TrappingStr:
            def __new__(cls, o):
                return [o]

        obj = object()
        real_str = builtins.str
        caught = None
        try:
            builtins.str = TrappingStr
            try:
                _json_dumps(obj)
            except ValueError as exc:
                caught = exc
        finally:
            builtins.str = real_str
        assert caught is not None
        assert caught.args == ("Circular reference detected",)
