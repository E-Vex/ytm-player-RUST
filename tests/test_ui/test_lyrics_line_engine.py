"""Tests for the YTM timed-lyrics line-conversion engine.

``_convert_lyric_lines`` is the engine behind ``LyricsSidebar``'s
``_process_lyrics`` synced arm: ytmusicapi's ``timestamps=True`` shape
(``hasTimestamps`` plus a ``lyrics`` list of ``LyricLine`` dataclass
objects — ``start_time`` in milliseconds, ``text``) converted into the
``(timestamp_seconds, text)`` list the synced view consumes. The
pure-Python reference lives in ``ui/sidebars/lyrics_sidebar.py``; when
the Rust extension is built the same engine runs natively
(``_ytm_player_native.convert_lyric_lines``).

The unit tests below assert the engine's contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engine against the pure-Python reference (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart. The wiring tests mount the
sidebar itself and check the synced view lands assembled from the
converted pairs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from ytmusicapi.models import LyricLine

from ytm_player.ui.sidebars.lyrics_sidebar import _convert_lyric_lines

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class _Attrs:
    """A stand-in exposing exactly the names given — every other
    attribute answers ``AttributeError``, like a dataclass missing the
    field (the shapes ytmusicapi's ``LyricLine`` never produces but the
    conversion's gates must still answer for)."""

    def __init__(self, **attrs: Any) -> None:
        self._attrs = attrs

    def __getattr__(self, name: str) -> Any:
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name) from None


class _CountingStart:
    """A ``start_time`` property answering ``1000 * read_count`` — each
    read observable through both the answer and the counter, pinning
    the reference's two reads (the ``hasattr`` gate's own, then the
    expression's)."""

    text = "counted"

    def __init__(self) -> None:
        self.reads = 0

    @property
    def start_time(self) -> Any:
        self.reads += 1
        return 1000 * self.reads


class _StartRaises:
    """A ``start_time`` property that raises — pinning the gate's
    AttributeError-only swallow (an AttributeError skips the entry,
    anything else propagates, like the reference's own ``hasattr``)."""

    text = "boom text"

    def __init__(self, exc: type[BaseException]) -> None:
        self._exc = exc

    @property
    def start_time(self) -> Any:
        raise self._exc("boom")


class _TextRaises:
    """A ``text`` property that raises — pinning the default's
    AttributeError-only swallow (an AttributeError reads as the `""`
    default, anything else propagates, like ``getattr``'s own)."""

    start_time = 500

    def __init__(self, exc: type[BaseException]) -> None:
        self._exc = exc

    @property
    def text(self) -> Any:
        raise self._exc("boom")


def _line(start_time: Any, text: Any = "line") -> _Attrs:
    """A plain attribute-carrying entry, the dataclass shape minus the
    extra fields."""
    return _Attrs(start_time=start_time, text=text)


# ── The canonical shape ────────────────────────────────────────────


class TestConversion:
    def test_the_real_ytmusicapi_dataclass(self):
        lines = [
            LyricLine(text="I was a liar", start_time=9200, end_time=10630, id=1),
            LyricLine(text="I gave in to the fire", start_time=10630, end_time=13000, id=2),
        ]
        assert _convert_lyric_lines(lines) == [
            (9.2, "I was a liar"),
            (10.63, "I gave in to the fire"),
        ]

    def test_milliseconds_become_float_seconds(self):
        pairs = _convert_lyric_lines(
            [_line(0, "zero"), _line(1, "one ms"), _line(1500, "1.5s"), _line(9200, "9.2s")]
        )
        assert [ts for ts, _text in pairs] == [0.0, 0.001, 1.5, 9.2]
        # The seconds are floats — CPython's own int / 1000.0.
        assert all(isinstance(ts, float) for ts, _text in pairs)

    def test_float_and_negative_starts_divide_through_cpython(self):
        pairs = _convert_lyric_lines([_line(9200.5, "float"), _line(-9200, "negative")])
        assert pairs[0][0] == 9200.5 / 1000.0
        assert pairs[1][0] == -9200 / 1000.0

    def test_bool_start_time_is_an_int_to_the_division(self):
        # bool is an int subclass: True / 1000.0 answers 0.001.
        ((seconds, _text),) = _convert_lyric_lines([_line(True, "bool")])
        assert seconds == 0.001

    def test_the_seconds_are_cpythons_own_division(self):
        # The exact float the reference's / 1000.0 produces — never a
        # reimplementation's rounding.
        ((seconds, _text),) = _convert_lyric_lines([_line(9200)])
        assert seconds == 9200 / 1000.0
        assert repr(seconds) == repr(9200 / 1000.0)

    def test_order_is_preserved_without_sorting(self):
        # Unlike the LRC parse feeding the same view, no sort happens.
        pairs = _convert_lyric_lines([_line(20000, "late"), _line(100, "early")])
        assert [text for _ts, text in pairs] == ["late", "early"]
        assert [ts for ts, _text in pairs] == [20.0, 0.1]

    def test_empty_list_answers_empty(self):
        assert _convert_lyric_lines([]) == []

    def test_simple_namespace_entries(self):
        entries = [SimpleNamespace(start_time=9200, text="ns line")]
        assert _convert_lyric_lines(entries) == [(9.2, "ns line")]


# ── The hasattr gate ───────────────────────────────────────────────


class TestStartGate:
    def test_entries_without_start_time_are_skipped(self):
        entries: list[Any] = [
            _Attrs(text="no start"),
            _line(100, "kept"),
            42,  # an int carries no start_time attribute
            "a string",
            None,
            [1, 2],
            {"start_time": 1, "text": "a dict is not attribute-shaped"},
            _line(200, "kept too"),
        ]
        pairs = _convert_lyric_lines(entries)
        assert pairs == [(0.1, "kept"), (0.2, "kept too")]

    def test_explicit_none_start_time_is_kept_by_the_gate(self):
        # The gate is hasattr, never truthiness: an explicit None
        # attribute exists — the division then raises its own TypeError.
        with pytest.raises(TypeError):
            _convert_lyric_lines([_Attrs(start_time=None, text="none start")])

    def test_the_gate_reads_start_time_twice(self):
        counter = _CountingStart()
        ((seconds, text),) = _convert_lyric_lines([counter])
        # The hasattr read, then the expression's own read.
        assert counter.reads == 2
        assert seconds == 2.0
        assert text == "counted"

    def test_start_property_raising_attribute_error_skips(self):
        pairs = _convert_lyric_lines([_StartRaises(AttributeError), _line(100, "kept")])
        assert pairs == [(0.1, "kept")]

    def test_start_property_raising_other_errors_propagates(self):
        with pytest.raises(ValueError, match="boom"):
            _convert_lyric_lines([_StartRaises(ValueError)])

    def test_non_numeric_start_time_raises_the_divisions_own_error(self):
        with pytest.raises(TypeError):
            _convert_lyric_lines([_Attrs(start_time="9200", text="str start")])


# ── The text read ──────────────────────────────────────────────────


class TestTextRead:
    def test_missing_text_reads_as_empty_string(self):
        assert _convert_lyric_lines([_Attrs(start_time=100)]) == [(0.1, "")]

    def test_text_passes_through_as_the_very_object(self):
        # Never coerced: the attribute value is the very object.
        sentinel = object()
        ((_, text),) = _convert_lyric_lines([_Attrs(start_time=100, text=sentinel)])
        assert text is sentinel

    def test_non_string_text_passes_through_uncoerced(self):
        pairs = _convert_lyric_lines([_line(100, 42), _line(200, None), _line(300, True)])
        assert pairs == [(0.1, 42), (0.2, None), (0.3, True)]

    def test_text_property_raising_attribute_error_reads_empty(self):
        assert _convert_lyric_lines([_TextRaises(AttributeError)]) == [(0.5, "")]

    def test_text_property_raising_other_errors_propagates(self):
        with pytest.raises(ValueError, match="boom"):
            _convert_lyric_lines([_TextRaises(ValueError)])

    def test_unicode_text_passes_through_untouched(self):
        pairs = _convert_lyric_lines([_line(500, "שלום עולם"), _line(900, "東京")])
        assert [text for _ts, text in pairs] == ["שלום עולם", "東京"]


# ── Container handling ─────────────────────────────────────────────


class TestContainers:
    def test_input_list_is_not_mutated(self):
        entries = [_line(100, "a"), _Attrs(text="skipped"), _line(200, "b")]
        snapshot = list(entries)
        _convert_lyric_lines(entries)
        assert entries == snapshot

    def test_tuple_input_is_accepted(self):
        assert _convert_lyric_lines((_line(100, "tuple"),)) == [(0.1, "tuple")]

    def test_generator_input_is_consumed(self):
        entries = iter([_line(100, "gen"), _Attrs(text="skipped")])
        assert _convert_lyric_lines(entries) == [(0.1, "gen")]


# ── _process_lyrics wiring ─────────────────────────────────────────


def _host_app():
    from textual.app import App, ComposeResult

    from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

    class _Host(App):
        """Minimal host providing the theme variables the CSS needs."""

        def get_css_variables(self) -> dict[str, str]:
            variables = super().get_css_variables()
            variables["lyrics-played"] = "#999999"
            variables["lyrics-current"] = "#ff4e45"
            variables["lyrics-upcoming"] = "#cccccc"
            return variables

        def compose(self) -> ComposeResult:
            yield LyricsSidebar()

    return _Host()


class TestProcessLyricsWiring:
    async def test_timed_lyrics_mount_the_synced_view(self):
        app = _host_app()
        async with app.run_test(size=(60, 20)):
            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar_widget = app.query_one(LyricsSidebar)
            data = {
                "hasTimestamps": True,
                "lyrics": [
                    LyricLine(text="first", start_time=0, end_time=1000, id=1),
                    LyricLine(text="second", start_time=5000, end_time=6000, id=2),
                ],
                "source": "Source: Musixmatch",
            }

            sidebar_widget._process_lyrics(data)

            assert sidebar_widget._is_synced is True
            assert sidebar_widget._synced_lines == [(0.0, "first"), (5.0, "second")]
            assert sidebar_widget._synced_timestamps == [0.0, 5.0]
            assert sidebar_widget._unsynced_lines == []
            # The view mounted from the converted pairs, timestamps kept.
            widgets = sidebar_widget._lyric_widgets
            assert [w._timestamp for w in widgets] == [0.0, 5.0]
            assert [w._text for w in widgets] == ["first", "second"]
            # The scroll is showing; the status label is hidden.
            assert app.query_one("#ls-scroll").display is True
            assert app.query_one("#ls-status").display is False

    async def test_empty_conversion_falls_through_to_the_string_path(self):
        # hasTimestamps + a list whose entries all lack start_time:
        # the conversion answers [] and the string path finds no str
        # lyrics either — the "No lyrics available." status.
        app = _host_app()
        async with app.run_test(size=(60, 20)):
            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar_widget = app.query_one(LyricsSidebar)
            data = {
                "hasTimestamps": True,
                "lyrics": [_Attrs(text="no start"), {"start_time": 1, "text": "dict"}],
            }

            sidebar_widget._process_lyrics(data)

            assert sidebar_widget._is_synced is False
            assert sidebar_widget._synced_lines == []
            assert app.query_one("#ls-status").display is True

    async def test_plain_string_lyrics_still_route_to_the_lrc_parse(self):
        app = _host_app()
        async with app.run_test(size=(60, 20)):
            from ytm_player.ui.sidebars.lyrics_sidebar import LyricsSidebar

            sidebar_widget = app.query_one(LyricsSidebar)
            sidebar_widget._process_lyrics({"hasTimestamps": False, "lyrics": "[00:12] hello"})

            assert sidebar_widget._is_synced is True
            assert sidebar_widget._synced_lines == [(12.0, "hello")]


# ── Native / fallback parity ────────────────────────────────────────
#
# When the Rust extension is built, the engine above IS the native one;
# each case below is also run in a child interpreter with the extension
# import blocked (pure-Python reference), and the two pair lists must
# agree exactly — pairs compared through their reprs (the seconds are
# CPython's own division result, the text the very attribute values),
# errors through their type-and-message spelling.

# The descriptor language: JSON-safe spellings of the stand-in entries
# above, rebuilt identically on both sides of the bridge.
_EXCS = {"AttributeError": AttributeError, "ValueError": ValueError}


def _build_entries(descriptors: list[dict]) -> list[Any]:
    """Rebuild the stand-in entries from a JSON descriptor list."""
    entries: list[Any] = []
    for descriptor in descriptors:
        kind = descriptor["kind"]
        if kind == "attrs":
            entries.append(_Attrs(**descriptor["attrs"]))
        elif kind == "namespace":
            entries.append(SimpleNamespace(**descriptor["attrs"]))
        elif kind == "dataclass":
            entries.append(
                LyricLine(
                    text=descriptor["text"],
                    start_time=descriptor["start_time"],
                    end_time=0,
                    id=0,
                )
            )
        elif kind == "counting":
            entries.append(_CountingStart())
        elif kind == "prop_start":
            entries.append(_StartRaises(_EXCS[descriptor["exc"]]))
        elif kind == "prop_text":
            entries.append(_TextRaises(_EXCS[descriptor["exc"]]))
        elif kind == "raw":
            entries.append(descriptor["value"])
        else:  # pragma: no cover - the descriptor language is closed
            raise AssertionError(f"unknown descriptor kind: {kind}")
    return entries


def _describe(pairs: list[tuple[Any, Any]]) -> list[str]:
    return [repr(pair) for pair in pairs]


_PARITY_CASES: list[list[dict]] = [
    # The canonical ytmusicapi shape.
    [
        {"kind": "dataclass", "text": "I was a liar", "start_time": 9200},
        {"kind": "dataclass", "text": "I gave in to the fire", "start_time": 10630},
    ],
    # Millisecond values through the division.
    [
        {"kind": "attrs", "attrs": {"start_time": 0, "text": "zero"}},
        {"kind": "attrs", "attrs": {"start_time": 1, "text": "one ms"}},
        {"kind": "attrs", "attrs": {"start_time": 1500, "text": "1.5s"}},
        {"kind": "attrs", "attrs": {"start_time": 9200, "text": "9.2s"}},
    ],
    # Order preserved — no sort, unlike the LRC parse.
    [
        {"kind": "attrs", "attrs": {"start_time": 20000, "text": "late"}},
        {"kind": "attrs", "attrs": {"start_time": 100, "text": "early"}},
    ],
    # The hasattr gate: keeps and skips interleaved.
    [
        {"kind": "attrs", "attrs": {"text": "no start"}},
        {"kind": "attrs", "attrs": {"start_time": 100, "text": "kept"}},
        {"kind": "raw", "value": 42},
        {"kind": "raw", "value": "a string"},
        {"kind": "raw", "value": None},
        {"kind": "raw", "value": [1, 2]},
        {"kind": "raw", "value": {"start_time": 1, "text": "a dict"}},
        {"kind": "attrs", "attrs": {"start_time": 200, "text": "kept too"}},
    ],
    # Everything skipped answers [].
    [
        {"kind": "attrs", "attrs": {"text": "a"}},
        {"kind": "raw", "value": 7},
        {"kind": "raw", "value": {"start_time": 1, "text": "dict"}},
    ],
    # Missing text reads as the "" default.
    [{"kind": "attrs", "attrs": {"start_time": 100}}],
    # Odd start_time spellings: float, negative, bool, big int.
    [
        {"kind": "attrs", "attrs": {"start_time": 9200.5, "text": "float"}},
        {"kind": "attrs", "attrs": {"start_time": -9200, "text": "negative"}},
        {"kind": "attrs", "attrs": {"start_time": True, "text": "bool"}},
        {"kind": "attrs", "attrs": {"start_time": 1180591620717411303424, "text": "2**70"}},
    ],
    # Non-numeric start_times raise the division's own TypeError.
    [{"kind": "attrs", "attrs": {"start_time": "9200", "text": "str start"}}],
    [{"kind": "attrs", "attrs": {"start_time": None, "text": "none start"}}],
    # Text passthrough: non-string values never coerced.
    [
        {"kind": "attrs", "attrs": {"start_time": 100, "text": 42}},
        {"kind": "attrs", "attrs": {"start_time": 200, "text": None}},
        {"kind": "attrs", "attrs": {"start_time": 300, "text": True}},
    ],
    # Unicode text untouched.
    [
        {"kind": "attrs", "attrs": {"start_time": 500, "text": "שלום עולם"}},
        {"kind": "attrs", "attrs": {"start_time": 900, "text": "東京"}},
    ],
    # The SimpleNamespace shape.
    [{"kind": "namespace", "attrs": {"start_time": 9200, "text": "ns line"}}],
    # The two-read gate: the counting property observes both reads.
    [{"kind": "counting"}],
    # Property raises: AttributeError swallows, ValueError propagates.
    [
        {"kind": "prop_start", "exc": "AttributeError"},
        {"kind": "attrs", "attrs": {"start_time": 100, "text": "kept"}},
    ],
    [{"kind": "prop_start", "exc": "ValueError"}],
    [{"kind": "prop_text", "exc": "AttributeError"}],
    [{"kind": "prop_text", "exc": "ValueError"}],
    # An error mid-walk discards the whole walk's pairs.
    [
        {"kind": "attrs", "attrs": {"start_time": 100, "text": "first"}},
        {"kind": "attrs", "attrs": {"start_time": "bad", "text": "str start"}},
    ],
    # The empty list.
    [],
]

_CHILD_PROG = """
import json, sys
from types import SimpleNamespace

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytmusicapi.models import LyricLine

from ytm_player.ui.sidebars.lyrics_sidebar import _convert_lyric_lines


class _Attrs:
    def __init__(self, **attrs):
        self._attrs = attrs

    def __getattr__(self, name):
        try:
            return self._attrs[name]
        except KeyError:
            raise AttributeError(name) from None


class _CountingStart:
    text = "counted"

    def __init__(self):
        self.reads = 0

    @property
    def start_time(self):
        self.reads += 1
        return 1000 * self.reads


class _StartRaises:
    text = "boom text"

    def __init__(self, exc):
        self._exc = exc

    @property
    def start_time(self):
        raise self._exc("boom")


class _TextRaises:
    start_time = 500

    def __init__(self, exc):
        self._exc = exc

    @property
    def text(self):
        raise self._exc("boom")


_EXCS = {"AttributeError": AttributeError, "ValueError": ValueError}


def _build_entries(descriptors):
    entries = []
    for descriptor in descriptors:
        kind = descriptor["kind"]
        if kind == "attrs":
            entries.append(_Attrs(**descriptor["attrs"]))
        elif kind == "namespace":
            entries.append(SimpleNamespace(**descriptor["attrs"]))
        elif kind == "dataclass":
            entries.append(
                LyricLine(
                    text=descriptor["text"],
                    start_time=descriptor["start_time"],
                    end_time=0,
                    id=0,
                )
            )
        elif kind == "counting":
            entries.append(_CountingStart())
        elif kind == "prop_start":
            entries.append(_StartRaises(_EXCS[descriptor["exc"]]))
        elif kind == "prop_text":
            entries.append(_TextRaises(_EXCS[descriptor["exc"]]))
        elif kind == "raw":
            entries.append(descriptor["value"])
        else:
            raise AssertionError(f"unknown descriptor kind: {kind}")
    return entries


cases = json.load(sys.stdin)
out = []
for case in cases:
    entries = _build_entries(case)
    try:
        pairs = _convert_lyric_lines(entries)
        out.append([repr(pair) for pair in pairs])
    except BaseException as exc:
        out.append({"error": f"{type(exc).__name__}: {exc}"})
json.dump(out, sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native conversion engine
    must produce exactly the same (seconds, text) pairs as the
    pure-Python reference — same hasattr gate (and its two reads per
    kept entry), same division results and error shapes, same text
    defaults and pass-throughs, same order — for every case, so the
    two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native_results = []
        for case in _PARITY_CASES:
            try:
                pairs = _convert_lyric_lines(_build_entries(case))
                native_results.append(_describe(pairs))
            except BaseException as exc:  # noqa: BLE001 - parity-pinned error shape
                native_results.append({"error": f"{type(exc).__name__}: {exc}"})

        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        for case, native, ref in zip(_PARITY_CASES, native_results, fallback):
            assert native == ref, (
                f"native and fallback lyric-line conversions differ for {case!r}:"
                f"\n{native!r}\n{ref!r}"
            )


class TestNativeBindings:
    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_native_engine_really_answers(self):
        """The binding is live: the sidebar's conversion runs through
        the native engine behind its wrapper when the extension is
        built."""
        from ytm_player.ui.sidebars import lyrics_sidebar
        from ytm_player.utils._ytm_player_native import convert_lyric_lines

        lines = [
            LyricLine(text="I was a liar", start_time=9200, end_time=10630, id=1),
            LyricLine(text="", start_time=0, end_time=100, id=2),
        ]
        # The wrapper answers exactly what the native engine answers
        # for the materialised list.
        assert lyrics_sidebar._convert_lyric_lines(lines) == convert_lyric_lines(list(lines))
        # The module-level name is the wrapper; the native engine is
        # what it calls — a tuple input rides the same materialisation
        # the engine itself would reject.
        assert callable(convert_lyric_lines)
        assert lyrics_sidebar._convert_lyric_lines(tuple(lines)) == [
            (9.2, "I was a liar"),
            (0.0, ""),
        ]
