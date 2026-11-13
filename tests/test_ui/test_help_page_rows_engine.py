"""Tests for the help-page rows engines.

The engines behind ``HelpPage``: the all-rows list — both halves: the
row-formatting spec engine (``_rows_from_spec``, the action-value
title-casing, the key-sequence join, the ``(unbound)`` placeholder) and
the fused keymap walk (``_rows_for_keymap``, the ``_build_rows`` engine
— the category walk, the description lookups and the binding-table
scan folded in) — and the filtered walk (``_visible_rows``, the
``_populate_table`` core — the lowercased-then-stripped needle, the
haystack containment, the category-header insertion). The pure-Python
references live in ``ui/pages/help.py``; when the Rust extension is
built the same engines run natively
(``_ytm_player_native.help_page_rows`` & co).

The unit tests below assert the engines' contract on whichever backend
is loaded. When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in a
child interpreter with the extension import blocked) case for case, so
the two implementations cannot drift apart.

The title-casing zoo sticks to characters whose case data has been
stable across every Unicode release CPython 3.10–3.14 ships (and the
Rust tables): the native engine tracks the Rust toolchain's Unicode and
the fallback the interpreter's, so spellings whose mappings were added
or changed in between (the post-15.0 Latin capitals, the recategorized
glottal letters) are out of scope — the production input is the
``[a-z_]+`` action-value identifier set.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from textual.app import App, ComposeResult
from textual.widgets import DataTable

from ytm_player.config.keymap import (
    DEFAULT_BINDINGS,
    Action,
    KeyMap,
    parse_key_sequence,
)
from ytm_player.ui.pages.help import (
    ACTION_CATEGORIES,
    ACTION_DESCRIPTIONS,
    HelpPage,
    _rows_for_keymap,
    _rows_from_spec,
    _title_case_action_value,
    _visible_rows,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── _rows_from_spec ─────────────────────────────────────────────────


class TestRowsFromSpec:
    def test_basic_row(self):
        rows = _rows_from_spec([("Playback", "play_pause", "Toggle play/pause", ["space", "p"])])
        assert rows == [("Playback", "Play Pause", "space / p", "Toggle play/pause")]

    def test_unbound_action(self):
        rows = _rows_from_spec([("Pages", "discovery_mix", "Start discovery", [])])
        assert rows == [("Pages", "Discovery Mix", "(unbound)", "Start discovery")]

    def test_single_key_has_no_join(self):
        rows = _rows_from_spec([("Navigation", "move_down", "Move cursor down", ["j"])])
        assert rows[0][2] == "j"

    def test_multi_word_action_values(self):
        rows = _rows_from_spec(
            [
                ("Navigation", "focus_pane_cycle", "Cycle focus", ["ctrl+w"]),
                ("Sorting", "reverse_sort", "Reverse sort order", ["R", "shift+r"]),
            ]
        )
        assert rows[0][1] == "Focus Pane Cycle"
        assert rows[1][1] == "Reverse Sort"

    def test_description_passes_through_undefaulted(self):
        """The description arrives already defaulted — an empty string for
        an undocumented action flows through unchanged."""
        rows = _rows_from_spec([("Actions", "some_action", "", ["x"])])
        assert rows[0][3] == ""

    def test_empty_spec_yields_no_rows(self):
        assert _rows_from_spec([]) == []

    def test_row_order_follows_spec_order(self):
        spec = [
            ("B", "b_action", "d2", ["k2"]),
            ("A", "a_action", "d1", ["k1"]),
        ]
        rows = _rows_from_spec(spec)
        assert [r[0] for r in rows] == ["B", "A"]

    def test_double_underscore_becomes_two_spaces(self):
        rows = _rows_from_spec([("C", "a__b", "d", [])])
        assert rows[0][1] == "A  B"


# ── title-casing pipeline (the ``_title_case_action_value`` core) ────


class TestTitleCasing:
    """The ``replace("_", " ").title()`` name pipeline, through the rows
    engine's own path — CPython's ``str.title()`` semantics carried over."""

    ROW_CASES = [
        # ASCII shapes: word-initial capitals after any uncased char.
        ("play_pause", "Play Pause"),
        ("PLAY_PAUSE", "Play Pause"),
        ("a1b", "A1B"),  # a digit is uncased — the b after it is word-initial
        ("it's", "It'S"),  # apostrophe is uncased — the s after it is word-initial
        ("it_s", "It S"),  # the underscore became a space before the title step
        ("abc", "Abc"),
        # The sharp s: titlecased (Ss), not uppercased (SS).
        ("ß", "Ss"),
        ("a_ß", "A Ss"),
        ("straße", "Straße"),  # ß lowered mid-word stays
        # Digraph families: every spelling titles onto the mixed-case form.
        ("Ǆ", "ǅ"),
        ("ǅ", "ǅ"),
        ("ǆ", "ǅ"),
        ("ǳ", "ǲ"),
        ("Ǳ", "ǲ"),
        ("ǅaǅ", "ǅaǆ"),  # lowered mid-word: the Lt form lowers to ǆ
        # Ligatures title to mixed-case expansions.
        ("ﬀ", "Ff"),
        ("ﬁ", "Fi"),
        ("ﬃ", "Ffi"),
        ("ﬆ", "St"),
        # Greek: the capital-sigma final form applies in the lowercase
        # path, with the Case_Ignorable set skipped ahead.
        ("aΣ", "Aς"),
        ("aΣb", "Aσb"),
        ("ΣΣΣ", "Σσς"),
        ("ΑΣΣ", "Ασς"),
        ("aΣ·b", "Aσ·B"),  # middle dot is case-ignorable — b is still ahead
        ("aΣ\u0301b", "Aσ́B"),  # so is a combining mark
        # A character that is BOTH cased and case-ignorable (ʰ is Lm
        # with Other_Lowercase): the sigma's lowercase path sees it as
        # the preceding cased character, but the Final_Sigma scan steps
        # over it — an uncased character beyond (1, ·) un-finals the
        # sigma, a cased one (a) keeps it final.
        ("1ʰΣ", "1ʰσ"),
        ("·ʰΣ", "·ʰσ"),
        ("aʰΣ", "Aʰς"),
        ("1ʰΣb", "1ʰσb"),
        # Ypogegrammeni titles: the single-char capitals.
        ("ᾳ", "ᾼ"),
        ("ᾀ", "ᾈ"),
        # Multi-character titlecase and lowercase expansions.
        ("ǰa", "J̌a"),
        ("ΐa", "Ϊ́a"),
        ("aİ", "Ai̇"),  # İ lowers to i + combining dot above
        # Cased-but-not-letters: modifier letters and ordinals keep the
        # next letter lowercase; uncased symbols don't.
        ("ʰa", "ʰa"),
        ("ªb", "ªb"),
        ("№a", "№A"),
        # Roman numerals are Other_Uppercase: cased, but title is itself.
        ("Ⅻa", "Ⅻa"),
        ("ⅷa", "Ⅷa"),
        # Uncased scripts pass through untouched.
        ("中a", "中A"),
        ("文a", "文A"),
        # Georgian mkhedruli titles to itself, not the Mtavruli capital.
        ("aა", "Aა"),
        # Micro sign titles to the Greek capital mu.
        ("µa", "Μa"),
        ("ǟ", "Ǟ"),
        ("aǟ", "Aǟ"),
    ]

    @pytest.mark.parametrize("value, expected_name", ROW_CASES)
    def test_title_case_pipeline(self, value, expected_name):
        assert _title_case_action_value(value) == expected_name

    @pytest.mark.parametrize("value, expected_name", ROW_CASES)
    def test_pipeline_through_the_rows_engine(self, value, expected_name):
        rows = _rows_from_spec([("Cat", value, "d", ["k"])])
        assert rows[0][1] == expected_name

    def test_underscore_replacement_precedes_title_casing(self):
        """``play_pause`` titles as ``Play Pause`` — the underscore is a
        word boundary only after its replacement, so the pipeline is
        replace-then-title, exactly the reference's expression order."""
        assert _title_case_action_value("play_pause") == "Play Pause"


# ── _visible_rows ───────────────────────────────────────────────────


_HEADER = (True, "", "", "", "")


class TestVisibleRows:
    def _rows(self):
        return [
            ("Playback", "Play Pause", "space", "Toggle play/pause"),
            ("Playback", "Next Track", "n", "Skip to next track"),
            ("Navigation", "Move Down", "j", "Move cursor down"),
        ]

    def test_no_filter_shows_every_row_with_headers(self):
        out = _visible_rows(self._rows(), "")
        assert out == [
            (True, "Playback", "", "", ""),
            (False, "Playback", "Play Pause", "space", "Toggle play/pause"),
            (False, "Playback", "Next Track", "n", "Skip to next track"),
            (True, "Navigation", "", "", ""),
            (False, "Navigation", "Move Down", "j", "Move cursor down"),
        ]

    def test_whitespace_only_filter_filters_nothing(self):
        """A needle of only whitespace strips to the empty string — the
        reference's ``if needle:`` then filters nothing."""
        assert _visible_rows(self._rows(), "   \t ") == _visible_rows(self._rows(), "")

    def test_filter_matches_case_insensitively_on_name(self):
        out = _visible_rows(self._rows(), "NEXT")
        assert out == [
            (True, "Playback", "", "", ""),
            (False, "Playback", "Next Track", "n", "Skip to next track"),
        ]

    def test_filter_matches_on_category(self):
        out = _visible_rows(self._rows(), "navigation")
        assert out == [
            (True, "Navigation", "", "", ""),
            (False, "Navigation", "Move Down", "j", "Move cursor down"),
        ]

    def test_filter_matches_on_keys(self):
        out = _visible_rows(self._rows(), "space")
        assert out == [
            (True, "Playback", "", "", ""),
            (False, "Playback", "Play Pause", "space", "Toggle play/pause"),
        ]

    def test_filter_matches_on_description(self):
        out = _visible_rows(self._rows(), "cursor")
        assert [r[1] for r in out if not r[0]] == ["Navigation"]

    def test_filter_needle_is_stripped(self):
        out = _visible_rows(self._rows(), "  next  ")
        assert [r for r in out if not r[0]] == [
            (False, "Playback", "Next Track", "n", "Skip to next track"),
        ]

    def test_needle_lowered_before_stripping(self):
        """The reference lowercases first (``.lower().strip()``): a needle
        whose lowering expands (İ → i + combining dot) keeps the expansion
        through the strip, and matches the same expansion in the
        lowercased haystack."""
        rows = [("C", "aİ", "k", "d")]
        out = _visible_rows(rows, "İ")
        assert out == [
            (True, "C", "", "", ""),
            (False, "C", "aİ", "k", "d"),
        ]

    def test_no_match_yields_nothing(self):
        assert _visible_rows(self._rows(), "zzz-not-there") == []

    def test_match_spans_fields(self):
        """The haystack joins the fields with spaces — a needle may span
        the seam between two of them."""
        rows = [("A", "X", "Y", "d")]
        assert _visible_rows(rows, "x y") == [
            (True, "A", "", "", ""),
            (False, "A", "X", "Y", "d"),
        ]

    def test_recurring_category_earns_its_header_again(self):
        rows = [
            ("A", "one", "k", "d"),
            ("B", "two", "k", "d"),
            ("A", "three", "k", "d"),
        ]
        out = _visible_rows(rows, "")
        assert [r[0] for r in out] == [True, False, True, False, True, False]

    def test_empty_category_never_earns_a_header(self):
        """The walk's category starts at ``""`` — a row whose category IS
        the empty string never triggers the change test."""
        rows = [("", "one", "k", "d"), ("", "two", "k", "d")]
        out = _visible_rows(rows, "")
        assert [r[0] for r in out] == [False, False]

    def test_filter_respects_category_changes(self):
        """Headers only appear for rows that survived the filter."""
        rows = [
            ("A", "one", "k", "d"),
            ("B", "two", "k", "d"),
            ("B", "three", "k", "d"),
        ]
        out = _visible_rows(rows, "three")
        assert out == [
            (True, "B", "", "", ""),
            (False, "B", "three", "k", "d"),
        ]

    def test_empty_rows_yields_nothing(self):
        assert _visible_rows([], "") == []
        assert _visible_rows([], "filter") == []


# ── _rows_for_keymap (the fused ``_build_rows`` engine) ────────────────────


class TestRowsForKeymap:
    """The fused all-rows engine: the keymap walk (the binding-table
    scan, the description lookups, the category walk) and the row
    formatting folded into one call — pinned on whichever backend is
    loaded.
    """

    @staticmethod
    def _keymap(pairs):
        return KeyMap(bindings={tuple(seq): Action(value) for seq, value in pairs})

    def test_basic_walk(self):
        keymap = self._keymap(
            [
                (("space",), "play_pause"),
                (("j",), "move_down"),
                (("down",), "move_down"),
            ]
        )
        categories = {
            "Playback": [Action.PLAY_PAUSE],
            "Navigation": [Action.MOVE_DOWN, Action.TOGGLE_SEARCH_MODE],
        }
        descriptions = {
            Action.PLAY_PAUSE: "Toggle play/pause",
            Action.MOVE_DOWN: "Move cursor down",
        }
        rows = _rows_for_keymap(keymap, categories, descriptions)
        assert rows == [
            ("Playback", "Play Pause", "space", "Toggle play/pause"),
            ("Navigation", "Move Down", "j / down", "Move cursor down"),
            ("Navigation", "Toggle Search Mode", "(unbound)", ""),
        ]

    def test_keys_follow_binding_table_order(self):
        """``get_keys_for_action`` answers in binding-table insertion
        order — the join keeps that order whatever the sequences."""
        keymap = self._keymap(
            [
                (("down",), "move_down"),  # bound FIRST
                (("j",), "move_down"),
                (("C-n",), "move_down"),
            ]
        )
        rows = _rows_for_keymap(keymap, {"N": [Action.MOVE_DOWN]}, {})
        assert rows[0][2] == "down / j / C-n"

    def test_multi_key_sequence_is_space_joined(self):
        """``format_key`` is ``" ".join(seq)`` — a two-key sequence
        keeps its inner space, distinct from the ``" / "`` join."""
        keymap = self._keymap(
            [
                (("C-w", "h"), "focus_pane_left"),
                (("g", "space"), "current_context"),
            ]
        )
        rows = _rows_for_keymap(keymap, {"N": [Action.FOCUS_PANE_LEFT, Action.CURRENT_CONTEXT]}, {})
        assert rows[0][2] == "C-w h"
        assert rows[1][2] == "g space"

    def test_unbound_and_undocumented(self):
        rows = _rows_for_keymap(self._keymap([]), {"A": [Action.QUIT]}, {})
        assert rows == [("A", "Quit", "(unbound)", "")]

    def test_empty_categories_yield_no_rows(self):
        assert _rows_for_keymap(self._keymap([(("space",), "play_pause")]), {}, {}) == []

    def test_description_for_uncategorised_action_is_never_read(self):
        """A description for an action no category lists is unread —
        no row, nothing observable through the output."""
        keymap = self._keymap([(("space",), "play_pause")])
        rows = _rows_for_keymap(keymap, {"A": [Action.PLAY_PAUSE]}, {Action.HELP: "Show help"})
        assert rows == [("A", "Play Pause", "space", "")]

    def test_binding_for_uncategorised_action_is_never_read(self):
        keymap = self._keymap(
            [
                (("space",), "play_pause"),
                (("?",), "help"),
            ]
        )
        rows = _rows_for_keymap(keymap, {"A": [Action.PLAY_PAUSE]}, {})
        assert rows == [("A", "Play Pause", "space", "")]

    def test_repeated_action_earns_two_rows(self):
        """The reference walk emits one row per category-list occurrence
        — both read the same binding group."""
        keymap = self._keymap([(("space",), "play_pause")])
        rows = _rows_for_keymap(keymap, {"A": [Action.PLAY_PAUSE], "B": [Action.PLAY_PAUSE]}, {})
        assert rows == [
            ("A", "Play Pause", "space", ""),
            ("B", "Play Pause", "space", ""),
        ]

    def test_fused_equals_the_reference_composition(self):
        """The fused engine must answer exactly what the reference chain
        did: the hand-rolled spec walk, then the spec engine — on the
        compiled defaults, the production shape."""
        keymap = self._keymap(_default_keymap_pairs())
        spec = []
        for category, actions in ACTION_CATEGORIES.items():
            for action in actions:
                description = ACTION_DESCRIPTIONS.get(action, "")
                key_strs = [keymap.format_key(seq) for seq in keymap.get_keys_for_action(action)]
                spec.append((category, action.value, description, key_strs))
        expected = _rows_from_spec(spec)
        rows = _rows_for_keymap(keymap, ACTION_CATEGORIES, ACTION_DESCRIPTIONS)
        assert rows == expected
        assert ("Playback", "Play Pause", "space", "Toggle play/pause") in rows
        assert (
            "Navigation",
            "Focus Pane Left",
            "C-w h",
            "Focus the Playlists sidebar (vim window: left)",
        ) in rows


# ── Native / fallback parity ─────────────────────────────────────────

_ROW_SPECS = [
    ("Playback", "play_pause", "Toggle play/pause", ["space", "p"]),
    ("Playback", "next_track", "Skip to next track", ["n"]),
    ("Navigation", "move_down", "Move cursor down", ["j", "down"]),
    ("Navigation", "discovery_mix", "Start discovery roulette", []),
    ("Sorting", "reverse_sort", "Reverse sort order", ["R"]),
] + [
    (  # the title-casing zoo, one row per case
        "Zoo",
        value,
        "d",
        ["k"],
    )
    for value, _ in TestTitleCasing.ROW_CASES
]

_ALL_ROWS = _rows_from_spec(_ROW_SPECS)

_FILTER_TEXTS = [
    "",
    "   ",
    "next",
    "NEXT",
    "  next  ",
    "play",
    "space",
    "cursor",
    "zzz-not-there",
    "İ",
    "Σ",
    "navigation",
    "play pause",
]


def _default_keymap_pairs():
    """(sequence, action value) pairs in the compiled defaults' own
    order — the insertion order ``_load_defaults`` binds them in."""
    return [
        (parse_key_sequence(seq), value) for value, seqs in DEFAULT_BINDINGS.items() for seq in seqs
    ]


_KEYMAP_CASES = [
    # The compiled defaults over the real categories and descriptions
    # — the production shape, end to end.
    {
        "bindings": [[list(seq), value] for seq, value in _default_keymap_pairs()],
        "categories": [
            [category, [a.value for a in actions]]
            for category, actions in ACTION_CATEGORIES.items()
        ],
        "descriptions": [[a.value, d] for a, d in ACTION_DESCRIPTIONS.items()],
    },
    # The hand-built shapes: binding-table order, the multi-key join,
    # unbound + undocumented, the unread extras, the repeated action.
    {
        "bindings": [
            [["down"], "move_down"],
            [["C-w", "h"], "focus_pane_left"],
            [["j"], "move_down"],
            [["?"], "help"],
            [["g", "space"], "current_context"],
        ],
        "categories": [
            ["Playback", ["play_pause"]],
            ["Navigation", ["move_down", "focus_pane_left", "toggle_search_mode"]],
            ["Pages", ["help", "current_context", "play_pause"]],
        ],
        "descriptions": [
            ["play_pause", "Toggle play/pause"],
            ["move_down", "Move cursor down"],
            ["focus_pane_left", "Focus the Playlists sidebar (vim window: left)"],
        ],
    },
    # An empty binding table: every row unbound.
    {
        "bindings": [],
        "categories": [["Pages", ["help", "queue"]]],
        "descriptions": [["help", "Show this help page"]],
    },
    # No categories at all: no rows, whatever the table holds.
    {
        "bindings": [[["space"], "play_pause"]],
        "categories": [],
        "descriptions": [],
    },
]

# The child interpreter runs this program with the case lists on stdin
# and prints the reference engines' results as JSON.
_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.config.keymap import Action, KeyMap
from ytm_player.ui.pages.help import _rows_for_keymap, _rows_from_spec, _visible_rows

payload = json.load(sys.stdin)


def _keymap_case(case):
    keymap = KeyMap(bindings={tuple(seq): Action(value) for seq, value in case["bindings"]})
    categories = {category: [Action(v) for v in actions] for category, actions in case["categories"]}
    descriptions = {Action(v): d for v, d in case["descriptions"]}
    return _rows_for_keymap(keymap, categories, descriptions)


out = {
    "rows": [_rows_from_spec(spec) for spec in payload["specs"]],
    "visible": [
        _visible_rows(payload["all_rows"], text) for text in payload["filter_texts"]
    ],
    "fused": [_keymap_case(case) for case in payload["keymap_cases"]],
}
json.dump(out, sys.stdout)
"""


def _native_results(payload: dict) -> dict:
    """Run whichever engines the module loaded (native when built) over
    the payload."""
    from ytm_player.config.keymap import Action, KeyMap

    def _keymap_case(case):
        keymap = KeyMap(bindings={tuple(seq): Action(value) for seq, value in case["bindings"]})
        categories = {category: [Action(v) for v in acts] for category, acts in case["categories"]}
        descriptions = {Action(v): d for v, d in case["descriptions"]}
        return _rows_for_keymap(keymap, categories, descriptions)

    return {
        "rows": [_rows_from_spec(spec) for spec in payload["specs"]],
        "visible": [_visible_rows(payload["all_rows"], text) for text in payload["filter_texts"]],
        "fused": [_keymap_case(case) for case in payload["keymap_cases"]],
    }


def _fallback_results(payload: dict) -> dict:
    """Run the pure-Python reference engines over the payload in a child
    interpreter with the native extension import blocked."""
    out = subprocess.run(
        [sys.executable, "-c", _CHILD_PROG],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


class TestNativeParity:
    """When the Rust extension is built, the native help-row engines must
    produce exactly the same rows and filtered walks as the pure-Python
    references for every case, so the two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        payload = {
            "specs": [_ROW_SPECS, _ROW_SPECS[:2], []],
            "all_rows": _ALL_ROWS,
            "filter_texts": _FILTER_TEXTS,
            "keymap_cases": _KEYMAP_CASES,
        }
        native = json.loads(json.dumps(_native_results(payload)))
        fallback = _fallback_results(payload)
        for i, spec in enumerate(payload["specs"]):
            assert native["rows"][i] == fallback["rows"][i], (
                f"native and fallback rows differ for spec {i}: {spec}"
            )
        for i, text in enumerate(_FILTER_TEXTS):
            assert native["visible"][i] == fallback["visible"][i], (
                f"native and fallback visible walks differ for filter {text!r}"
            )
        for i, case in enumerate(payload["keymap_cases"]):
            assert native["fused"][i] == fallback["fused"][i], (
                f"native and fallback fused walks differ for keymap case {i}: {case}"
            )


# ── Mounted widget integration ───────────────────────────────────────


class _HelpHost(App):
    """Minimal host app mounting a HelpPage through the real Textual loop."""

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables["selected-item"] = "#3a3a3a"
        return variables

    def compose(self) -> ComposeResult:
        yield HelpPage()


@pytest.fixture(autouse=True)
def _default_keymap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the keymap to the compiled defaults so the test environment's
    user config cannot steer the row content."""
    monkeypatch.setattr("ytm_player.ui.pages.help.get_keymap", lambda: KeyMap.load())


class TestHelpPageWidget:
    """The HelpPage glue: the engines' output reaching the DataTable —
    rows built from the real keymap, headers between categories, and the
    filter reactive re-populating through the engine walk."""

    async def test_rows_built_from_the_real_keymap(self):
        async with _HelpHost().run_test() as pilot:
            page = pilot.app.query_one(HelpPage)
            table = page.query_one("#help-table", DataTable)

            # Every documented action earned a row, and the category
            # headers sit between the categories.
            assert table.row_count > 0
            assert len(page._all_rows) >= 50
            header_keys = [k.value for k in table.rows]
            assert any(k.startswith("header_") for k in header_keys)
            assert "header_Playback" in header_keys

            first = page._all_rows[0]
            assert first[0] == "Playback"
            assert first[1] == "Play Pause"

    async def test_filter_reduces_the_table_through_the_engine(self):
        async with _HelpHost().run_test() as pilot:
            page = pilot.app.query_one(HelpPage)
            table = page.query_one("#help-table", DataTable)
            full_count = table.row_count

            page.filter_text = "volume"
            await pilot.pause()

            # Exactly the rows whose haystack mentions the needle survive,
            # plus the headers the walk re-emits.
            matching = sum(
                1
                for category, name, keys, desc in page._all_rows
                if "volume" in f"{category} {name} {keys} {desc}".lower()
            )
            assert matching > 0
            headers = len(
                {
                    category
                    for category, name, keys, desc in page._all_rows
                    if "volume" in f"{category} {name} {keys} {desc}".lower()
                }
            )
            assert table.row_count == matching + headers
            assert table.row_count < full_count

            # And the empty needle restores everything.
            page.filter_text = ""
            await pilot.pause()
            assert table.row_count == full_count
