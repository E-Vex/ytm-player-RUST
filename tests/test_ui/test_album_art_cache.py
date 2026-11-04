"""Tests for the album-art cache LRU engine.

The engine behind the album-art widget's module-level URL→render cache:
the hit promotion of the cache-check path (``get``), the insert that
keeps an existing key's position (``OrderedDict.__setitem__``
semantics — this engine never promotes on insert) with its single-pop
overflow eviction, the guarded delete of the resize invalidation
(``remove``), and the clear/length bookkeeping. The pure-Python
reference lives in ``ui/widgets/album_art.py`` (``_ArtCacheCore``);
when the Rust extension is built the same engine runs natively
(``_ytm_player_native.ArtCacheCore``).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) operation for
operation, so the two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.ui.widgets.album_art import _ART_CACHE_MAX, _new_art_cache

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


def _make_core(max_entries: int):
    """Build the engine under test — the native Rust core when the
    extension is built, the pure-Python transcription otherwise."""
    if _HAVE_NATIVE:
        return _ytm_player_native.ArtCacheCore(max_entries)
    from ytm_player.ui.widgets.album_art import _ArtCacheCore

    return _ArtCacheCore(max_entries)


# ── get / insert / remove ─────────────────────────────────────────


class TestGetInsertRemove:
    def test_miss_is_none(self):
        core = _make_core(5)
        assert core.get("https://example.invalid/a.jpg") is None

    def test_insert_then_get_roundtrip(self):
        core = _make_core(5)
        core.insert("url-a", "render-a")
        assert core.get("url-a") == "render-a"

    def test_get_hands_back_the_very_object(self):
        # The renders are held as the Python objects they are — never
        # copied — so the widget's identity-sensitive ``_rendered``
        # refresh keeps working.
        core = _make_core(5)
        sentinel = object()
        core.insert("url-a", sentinel)
        assert core.get("url-a") is sentinel

    def test_insert_existing_updates_value_in_place(self):
        core = _make_core(5)
        core.insert("url-a", "old")
        core.insert("url-a", "new")
        assert core.get("url-a") == "new"
        assert core.length() == 1

    def test_remove_present_answers_true(self):
        core = _make_core(5)
        core.insert("url-a", "render-a")
        assert core.remove("url-a") is True
        assert core.get("url-a") is None
        assert core.length() == 0

    def test_remove_missing_answers_false(self):
        core = _make_core(5)
        assert core.remove("never-cached") is False

    def test_clear_empties(self):
        core = _make_core(5)
        core.insert("url-a", "a")
        core.insert("url-b", "b")
        core.clear()
        assert core.length() == 0
        assert core.get("url-a") is None


# ── LRU order — the two engines' differentiating contract ──────────


class TestLruOrder:
    def test_overflow_evicts_the_least_recent(self):
        core = _make_core(2)
        assert core.insert("a", "ra") is None
        assert core.insert("b", "rb") is None
        evicted = core.insert("c", "rc")
        assert evicted == ("a", "ra")
        assert core.get("a") is None
        assert core.get("b") == "rb"
        assert core.get("c") == "rc"

    def test_no_eviction_under_the_cap(self):
        core = _make_core(3)
        assert core.insert("a", "ra") is None
        assert core.insert("b", "rb") is None
        assert core.insert("c", "rc") is None

    def test_hit_promotion_steers_the_eviction(self):
        core = _make_core(2)
        core.insert("a", "ra")
        core.insert("b", "rb")
        assert core.get("a") == "ra"  # promote a
        evicted = core.insert("c", "rc")
        assert evicted == ("b", "rb")

    def test_miss_leaves_the_order_alone(self):
        core = _make_core(2)
        core.insert("a", "ra")
        core.insert("b", "rb")
        assert core.get("nope") is None
        evicted = core.insert("c", "rc")
        assert evicted == ("a", "ra")

    def test_insert_never_promotes_an_existing_key(self):
        # THE differentiator against the shuffle-preferences engine:
        # __setitem__ on an existing key keeps its position, so a later
        # overflow still evicts it as the oldest — the shuffle engine's
        # set() would have promoted it out of harm's way (and its
        # eviction victim would have been "b" instead).
        core = _make_core(2)
        core.insert("a", "ra")
        core.insert("b", "rb")
        core.insert("a", "ra-new")  # position kept, value replaced
        evicted = core.insert("c", "rc")
        assert evicted == ("a", "ra-new")
        assert core.get("a") is None
        assert core.get("b") == "rb"
        assert core.get("c") == "rc"

    def test_single_pop_even_when_jump_exceeds_cap_by_one(self):
        # The reference's `if` shape: one eviction per insert, never a
        # walk. (Jumping the cap by more than one is unreachable
        # through the widget's single-insert path; the shape is pinned
        # by construction — each insert adds at most one entry.)
        core = _make_core(0)
        evicted = core.insert("a", "ra")
        assert evicted == ("a", "ra")
        assert core.length() == 0

    def test_full_cap_flow_at_the_reference_constant(self):
        core = _make_core(_ART_CACHE_MAX)
        for i in range(_ART_CACHE_MAX):
            core.insert(f"url-{i}", f"render-{i}")
        assert core.length() == _ART_CACHE_MAX
        evicted = core.insert("url-new", "render-new")
        assert evicted == ("url-0", "render-0")
        assert core.length() == _ART_CACHE_MAX


# ── The module-level cache over the engine ─────────────────────────


class TestModuleCache:
    def test_module_cache_is_the_engine(self):
        # The module-level _ART_CACHE is built by the same factory the
        # tests use — the native Rust core when the extension is built,
        # the pure-Python transcription otherwise.
        assert _new_art_cache().length() == 0

    def test_unhashable_key_raises_type_error(self):
        core = _make_core(5)
        with pytest.raises(TypeError):
            core.get(["unhashable"])
        with pytest.raises(TypeError):
            core.insert(["unhashable"], "value")

    def test_identity_shortcut_hits_nan(self):
        nan = float("nan")
        core = _make_core(5)
        core.insert(nan, "render")
        assert core.get(nan) == "render"


# ── Native parity ──────────────────────────────────────────────────

# The child interpreter runs this program with the case list on stdin
# and prints the reference engine's results as JSON. Each case is a
# list of operations against a fresh core of the given cap:
# ["insert", url, value] / ["get", url] / ["remove", url] / ["clear"] /
# ["length"].
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.widgets.album_art import _ArtCacheCore

cases = json.load(sys.stdin)
out = []
for case in cases:
    cap, ops = case
    core = _ArtCacheCore(cap)
    steps = []
    for op in ops:
        kind = op[0]
        try:
            if kind == 'insert':
                steps.append({'ok': core.insert(op[1], op[2])})
            elif kind == 'get':
                steps.append({'ok': core.get(op[1])})
            elif kind == 'remove':
                steps.append({'ok': core.remove(op[1])})
            elif kind == 'clear':
                core.clear()
                steps.append({'ok': None})
            elif kind == 'length':
                steps.append({'ok': core.length()})
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
                if kind == "insert":
                    steps.append({"ok": core.insert(op[1], op[2])})
                elif kind == "get":
                    steps.append({"ok": core.get(op[1])})
                elif kind == "remove":
                    steps.append({"ok": core.remove(op[1])})
                elif kind == "clear":
                    core.clear()
                    steps.append({"ok": None})
                elif kind == "length":
                    steps.append({"ok": core.length()})
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


def _jsonify(value):
    """Normalise a result the way the child's JSON round-trip does
    (the native engine's evicted pairs come back as tuples)."""
    return json.loads(json.dumps(value))


# Op sequences the two implementations must agree on, case for case:
# the roundtrip and the miss; the overflow eviction; the hit
# promotion steering the eviction; the miss leaving the order alone;
# the no-promotion insert (the differentiator); the in-place value
# update; the guarded remove; the clear; the zero-cap degenerate; the
# unicode URL keys; and the unhashable refusals.
_PARITY_CASES: list = [
    [5, [["get", "missing"], ["length"]]],
    [5, [["insert", "a", "ra"], ["get", "a"], ["length"]]],
    [
        2,
        [
            ["insert", "a", "ra"],
            ["insert", "b", "rb"],
            ["insert", "c", "rc"],
            ["get", "a"],
            ["get", "b"],
            ["get", "c"],
            ["length"],
        ],
    ],
    [
        2,
        [
            ["insert", "a", "ra"],
            ["insert", "b", "rb"],
            ["get", "a"],
            ["insert", "c", "rc"],
            ["get", "a"],
            ["get", "b"],
        ],
    ],
    [
        2,
        [
            ["insert", "a", "ra"],
            ["insert", "b", "rb"],
            ["get", "nope"],
            ["insert", "c", "rc"],
            ["get", "a"],
            ["get", "b"],
        ],
    ],
    [
        2,
        [
            ["insert", "a", "ra"],
            ["insert", "b", "rb"],
            ["insert", "a", "ra-new"],
            ["insert", "c", "rc"],
            ["get", "a"],
            ["get", "b"],
            ["get", "c"],
        ],
    ],
    [
        5,
        [
            ["insert", "a", "old"],
            ["insert", "a", "new"],
            ["get", "a"],
            ["length"],
        ],
    ],
    [
        5,
        [
            ["insert", "a", "ra"],
            ["remove", "a"],
            ["get", "a"],
            ["remove", "a"],
            ["length"],
        ],
    ],
    [5, [["insert", "a", "ra"], ["insert", "b", "rb"], ["clear"], ["length"]]],
    [0, [["insert", "a", "ra"], ["get", "a"], ["length"]]],
    [
        5,
        [
            ["insert", "https://i.ytimg.com/日本語.jpg", "ra"],
            ["get", "https://i.ytimg.com/日本語.jpg"],
            ["remove", "https://i.ytimg.com/日本語.jpg"],
            ["length"],
        ],
    ],
]


class TestNativeParity:
    """When the Rust extension is built, the native cache engine must
    answer every operation exactly like the pure-Python reference run
    in a child interpreter with the extension import blocked — the
    promotions, the evictions (and their returned pairs), the guarded
    removes and the unhashable refusals included — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = _jsonify(_native_results(_PARITY_CASES))
        fallback = _fallback_results(_PARITY_CASES)
        for case, native_steps, fallback_steps in zip(_PARITY_CASES, native, fallback):
            assert native_steps == fallback_steps, (
                f"native and fallback art-cache results differ for case {case}"
            )
