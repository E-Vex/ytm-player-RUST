"""The theme resolution engines — contracts on whichever backend is loaded.

The three resolution sites of the documented invariant chain —
``ThemeColors.from_css_variables`` (``_theme_fields_from_css``),
``YTMPlayerApp.get_css_variables`` (``_app_css_variables``) and
``YTMPlayerApp.watch_theme`` (``_theme_fields_from_theme``) — plus the
``rich_safe_color`` token translation every construction path normalises
through. The contract tests run against whichever backend is loaded
(native when the extension is built, the pure-Python transcription
otherwise); the parity tests compare the two backends over JSON zoos in
a child interpreter with the extension blocked.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from unittest.mock import patch

import pytest

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


# ── rich_safe_color ────────────────────────────────────────────────────


class TestRichSafeColorEngine:
    def test_strips_the_ansi_prefix(self):
        from ytm_player.ui.theme import rich_safe_color

        assert rich_safe_color("ansi_cyan") == "cyan"
        assert rich_safe_color("ansi_bright_red") == "bright_red"

    def test_passthrough_without_the_prefix(self):
        from ytm_player.ui.theme import rich_safe_color

        for value in ("#ff0000", "red", "default", "", "ansi"):
            assert rich_safe_color(value) == value

    def test_the_bare_prefix_strips_to_empty(self):
        from ytm_player.ui.theme import rich_safe_color

        assert rich_safe_color("ansi_") == ""


# ── _theme_fields_from_css ─────────────────────────────────────────────


class TestThemeFieldsFromCss:
    @staticmethod
    def fields(variables: dict) -> dict:
        from ytm_player.ui.theme import _theme_fields_from_css

        return dict(_theme_fields_from_css(variables))

    def test_all_absent_yields_each_fields_own_default(self):
        # The nine app-specific fields bottom out at THEIR dataclass
        # defaults, not the base fields' — selected-item's surface chain
        # ends at #2a2a2a (surface's own default is #1a1a1a), and
        # lyrics-upcoming's text chain at #aaaaaa (text's own is #ffffff).
        f = self.fields({})
        assert f["background"] == "#0f0f0f"
        assert f["muted_text"] == "#999999"
        assert f["selected_item"] == "#2a2a2a"
        assert f["progress_empty"] == "#555555"
        assert f["lyrics_upcoming"] == "#aaaaaa"
        assert f["lyrics_current"] == "#ff4e45"  # DEFAULT_LYRIC_CURRENT

    def test_base_variables_win(self):
        f = self.fields({"background": "#111111", "text-muted": "#222222"})
        assert f["background"] == "#111111"
        assert f["muted_text"] == "#222222"

    def test_app_specific_keys_win_over_derived(self):
        f = self.fields({"surface": "#333333", "selected-item": "#444444"})
        assert f["selected_item"] == "#444444"
        assert f["playback_bar_bg"] == "#333333"  # derived from surface
        assert f["surface"] == "#333333"

    def test_lyrics_current_three_level_chain(self):
        # lyrics-current -> accent -> primary -> DEFAULT_LYRIC_CURRENT.
        assert self.fields({})["lyrics_current"] == "#ff4e45"
        assert self.fields({"primary": "#p"})["lyrics_current"] == "#p"
        assert self.fields({"primary": "#p", "accent": "#a"})["lyrics_current"] == "#a"
        assert self.fields({"lyrics-current": "#lc", "accent": "#a"})["lyrics_current"] == "#lc"

    def test_progress_filled_falls_back_to_primary(self):
        assert self.fields({"primary": "#pp"})["progress_filled"] == "#pp"
        assert self.fields({})["progress_filled"] == "#ff0000"

    def test_present_values_pass_through_as_the_objects_they_are(self):
        # The engine resolves; the ANSI normalisation is __post_init__'s,
        # applied after construction — an ansi_ token or a non-str value
        # comes out untouched.
        f = self.fields({"accent": "ansi_cyan", "surface": 5})
        assert f["accent"] == "ansi_cyan"
        assert f["surface"] == 5

    def test_present_none_passes_through(self):
        # dict.get semantics: a present-but-None value is the value.
        assert self.fields({"background": None})["background"] is None

    def test_every_field_is_resolved(self):
        from ytm_player.ui.theme import ThemeColors

        assert set(self.fields({})) == {f.name for f in dataclasses.fields(ThemeColors)}

    def test_from_css_variables_builds_the_dataclass_normalised(self):
        from ytm_player.ui.theme import ThemeColors

        tc = ThemeColors.from_css_variables({"accent": "ansi_cyan"})
        # __post_init__ normalises the token AFTER the resolution.
        assert tc.accent == "cyan"
        assert tc.lyrics_current == "cyan"


# ── _app_css_variables ─────────────────────────────────────────────────


class TestAppCssVariables:
    @staticmethod
    def overlay(variables: dict, colors: dict | None) -> dict:
        from ytm_player.app._app import _app_css_variables

        return _app_css_variables(variables, colors)

    def test_nine_defaults_inserted_in_order(self):
        out = self.overlay(
            {"surface": "#s", "text": "#t", "text-muted": "#m", "primary": "#p"}, None
        )
        assert list(out) == [
            "surface",
            "text",
            "text-muted",
            "primary",
            "playback-bar-bg",
            "active-tab",
            "inactive-tab",
            "selected-item",
            "progress-filled",
            "progress-empty",
            "lyrics-played",
            "lyrics-current",
            "lyrics-upcoming",
        ]
        assert out["playback-bar-bg"] == "#s"
        assert out["active-tab"] == "#t"
        assert out["inactive-tab"] == "#m"
        assert out["selected-item"] == "#s"
        assert out["progress-filled"] == "#p"
        assert out["progress-empty"] == "#s"

    def test_empty_base_yields_the_hardcoded_constants(self):
        out = self.overlay({}, None)
        assert out["playback-bar-bg"] == "#1a1a1a"
        assert out["selected-item"] == "#2a2a2a"
        assert out["progress-empty"] == "#555555"
        assert out["lyrics-current"] == "#ff4e45"  # DEFAULT_LYRIC_CURRENT
        assert out["lyrics-upcoming"] == "#aaaaaa"

    def test_lyrics_current_chain(self):
        assert self.overlay({}, None)["lyrics-current"] == "#ff4e45"
        assert self.overlay({"primary": "#p"}, None)["lyrics-current"] == "#p"
        assert self.overlay({"accent": "#a"}, None)["lyrics-current"] == "#a"
        assert self.overlay({"lyrics-current": "#lc"}, None)["lyrics-current"] == "#lc"

    def test_existing_app_keys_are_left_alone(self):
        out = self.overlay({"selected-item": "#keep", "surface": "#s"}, None)
        assert out["selected-item"] == "#keep"

    def test_defaults_computed_from_the_pre_patch_state(self):
        # All nine defaults see the base variables as they were — the
        # app keys the overlay itself inserts never feed back.
        out = self.overlay({"text": "#t"}, None)
        assert out["active-tab"] == "#t"
        assert out["lyrics-upcoming"] == "#t"

    def test_overrides_win_over_everything(self):
        out = self.overlay(
            {"surface": "#s", "lyrics-current": "#lc"},
            {"primary": "#op", "lyrics_current": "#oly", "muted_text": "#om"},
        )
        assert out["primary"] == "#op"
        assert out["lyrics-current"] == "#oly"
        assert out["text-muted"] == "#om"
        assert out["surface"] == "#s"  # not overridden

    def test_empty_or_missing_colors_is_no_override(self):
        for colors in (None, {}):
            assert self.overlay({"surface": "#s"}, colors) == self.overlay({"surface": "#s"}, None)

    def test_patched_in_place_and_returned_as_the_same_object(self):
        variables = {"surface": "#s"}
        out = self.overlay(variables, {"primary": "#p"})
        assert out is variables

    def test_override_values_pass_through_as_the_objects_they_are(self):
        assert self.overlay({}, {"primary": 7})["primary"] == 7


# ── _theme_fields_from_theme ───────────────────────────────────────────


class TestThemeFieldsFromTheme:
    @staticmethod
    def fields(raws: list, variables: dict) -> dict:
        from ytm_player.app._app import _theme_fields_from_theme

        return dict(_theme_fields_from_theme(*raws, variables))

    NINES = [None] * 9

    def test_all_falsy_raws_hit_every_fallback(self):
        f = self.fields(self.NINES, {})
        assert f["primary"] is None  # raw, no fallback
        assert f["background"] == "#0f0f0f"
        assert f["foreground"] == "#ffffff"
        assert f["secondary"] == "#aaaaaa"
        assert f["accent"] == "#ff4e45"
        assert f["success"] == "#2ecc71"
        assert f["warning"] == "#f39c12"
        assert f["error"] == "#e74c3c"
        assert f["surface"] == "#1a1a1a"

    def test_per_site_constants_differ_for_shared_sources(self):
        # foreground feeds itself (#ffffff), lyrics_upcoming (#aaaaaa);
        # surface feeds itself (#1a1a1a), border (#333333) and
        # progress_empty (#555555); secondary feeds itself (#aaaaaa) and
        # muted_text/lyrics_played/inactive_tab (#999999).
        f = self.fields(self.NINES, {})
        assert f["text"] == "#ffffff"
        assert f["muted_text"] == "#999999"
        assert f["border"] == "#333333"
        assert f["lyrics_upcoming"] == "#aaaaaa"
        assert f["lyrics_played"] == "#999999"

    def test_empty_strings_are_falsy_too(self):
        f = self.fields([""] * 9, {})
        assert f["background"] == "#0f0f0f"
        assert f["primary"] == ""  # raw passthrough

    def test_progress_filled_keeps_the_raw_primary_even_when_falsy(self):
        # The only field whose default is a raw value with NO or-chain.
        assert self.fields(self.NINES, {})["progress_filled"] is None
        assert self.fields(["#p", *([None] * 8)], {})["progress_filled"] == "#p"

    def test_lyrics_current_raw_chain(self):
        # accent or primary or DEFAULT — the raw values, not the
        # or-fallback constants the fields themselves took.
        f = self.fields(["#p", None, None, None, None, None, None, None, None], {})
        assert f["lyrics_current"] == "#p"
        f = self.fields([None, None, None, None, "#a", None, None, None, None], {})
        assert f["lyrics_current"] == "#a"
        f = self.fields(self.NINES, {})
        assert f["lyrics_current"] == "#ff4e45"
        # accent None + primary truthy: the accent FIELD takes #ff4e45 but
        # lyrics_current follows primary.
        f = self.fields(["#p", None, None, None, None, None, None, None, None], {})
        assert f["accent"] == "#ff4e45"
        assert f["lyrics_current"] == "#p"

    def test_theme_variables_win_for_app_specific_fields(self):
        f = self.fields(
            [None] * 8 + ["#surf"],
            {
                "playback-bar-bg": "#pbbg",
                "lyrics-current": "#lc",
                "selected-item": None,
            },
        )
        assert f["playback_bar_bg"] == "#pbbg"
        assert f["lyrics_current"] == "#lc"
        assert f["selected_item"] is None  # present-but-None wins
        assert f["active_tab"] == "#ffffff"  # missing -> or-chain

    def test_derived_fields_use_the_or_chains_of_their_site(self):
        f = self.fields(["#p", "#bg", "#fg", "#sec", None, None, None, None, "#surf"], {})
        assert f["playback_bar_bg"] == "#surf"
        assert f["active_tab"] == "#fg"
        assert f["inactive_tab"] == "#sec"
        assert f["border"] == "#surf"
        assert f["muted_text"] == "#sec"
        assert f["text"] == "#fg"


# ── Native vs fallback parity ──────────────────────────────────────────

# JSON-round-trippable input zoos: variables dicts for the CSS resolver,
# (variables, colors) pairs for the overlay, (nine raws, variables) pairs
# for the Theme resolver, and plain strings for the token translation.
_CSS_ZOO = [
    {},
    {"background": "#111111"},
    {"text-muted": "#222222"},
    {"accent": "#ff1122"},
    {"primary": "#pp", "accent": "#aa"},
    {"lyrics-current": "#lc"},
    {"lyrics-current": "#lc", "accent": "#a", "primary": "#p"},
    {"surface": "#s", "selected-item": "#sel"},
    {"progress-filled": "#pf"},
    {"lyrics-upcoming": "#lu"},
    {"playback-bar-bg": "#pbbg", "active-tab": "#at", "inactive-tab": "#it"},
    {"background": None, "surface": None},
    {"accent": "", "primary": ""},
    {"accent": "ansi_cyan", "primary": "ansi_bright_red"},
    {"background": 5, "surface": ["x"]},
    {"border": "#b", "error": "#e", "success": "#su", "warning": "#w", "secondary": "#se"},
    {"foreground": "#fg", "text": "#tx"},
    {
        k: f"#{i}"
        for i, k in enumerate(
            [
                "background",
                "foreground",
                "primary",
                "secondary",
                "accent",
                "success",
                "warning",
                "error",
                "surface",
                "border",
                "text-muted",
                "text",
                "playback-bar-bg",
                "active-tab",
                "inactive-tab",
                "selected-item",
                "progress-filled",
                "progress-empty",
                "lyrics-played",
                "lyrics-current",
                "lyrics-upcoming",
            ]
        )
    },
]

_APP_ZOO = [
    [{}, None],
    [{}, {}],
    [{"surface": "#s"}, None],
    [{"text": "#t"}, None],
    [{"text-muted": "#m"}, None],
    [{"primary": "#p"}, None],
    [{"accent": "#a", "primary": "#p"}, None],
    [{"lyrics-current": "#lc"}, None],
    [
        {"surface": "#s", "text": "#t", "text-muted": "#m", "primary": "#p"},
        {"primary": "#op", "lyrics_current": "#oly"},
    ],
    [{"surface": "#s"}, {"muted_text": "#om", "playback_bar_bg": "#opb"}],
    [{"lyrics-current": "#lc"}, {"lyrics_current": None}],
    [{"surface": "#s"}, {"primary": 7, "text": ["a"]}],
    [{}, {"not": 1, "a": 2, "field": 3}],
    [
        {
            k: f"#{i}"
            for i, k in enumerate(
                [
                    "background",
                    "foreground",
                    "primary",
                    "secondary",
                    "accent",
                    "success",
                    "warning",
                    "error",
                    "surface",
                    "border",
                    "text",
                    "text-muted",
                    "playback-bar-bg",
                    "active-tab",
                    "inactive-tab",
                    "selected-item",
                    "progress-filled",
                    "progress-empty",
                    "lyrics-played",
                    "lyrics-current",
                    "lyrics-upcoming",
                ]
            )
        },
        {
            "background": "#ob",
            "foreground": "#of",
            "primary": "#op",
            "secondary": "#os",
            "accent": "#oa",
            "success": "#osu",
            "warning": "#ow",
            "error": "#oe",
            "surface": "#osu2",
            "border": "#ob2",
            "text": "#ot",
            "muted_text": "#om",
            "playback_bar_bg": "#opb",
            "active_tab": "#oat",
            "inactive_tab": "#oit",
            "selected_item": "#osi",
            "progress_filled": "#opf",
            "progress_empty": "#ope",
            "lyrics_played": "#olp",
            "lyrics_current": "#olc",
            "lyrics_upcoming": "#olu",
        },
    ],
]

_THEME_ZOO = [
    [[None] * 9, {}],
    [[""] * 9, {}],
    [["#p", "#bg", "#fg", "#sec", "#a", "#su", "#w", "#e", "#surf"], {}],
    [["#p", None, None, None, None, None, None, None, None], {}],
    [[None, None, None, None, "#a", None, None, None, None], {}],
    [[None, None, None, None, None, None, None, None, "#surf"], {}],
    [
        [None] * 9,
        {
            "playback-bar-bg": "#pbbg",
            "active-tab": "#at",
            "inactive-tab": "#it",
            "selected-item": "#sel",
            "progress-filled": "#pf",
            "progress-empty": "#pe",
            "lyrics-played": "#lp",
            "lyrics-current": "#lc",
            "lyrics-upcoming": "#lu",
        },
    ],
    [[None] * 9, {"selected-item": None, "lyrics-current": ""}],
    [["", "#bg", "#fg", "#sec", "", "", "", "", ""], {"lyrics-current": "#lc"}],
    [["#p", "#bg", "#fg", "#sec", "#a", "#su", "#w", "#e", "#surf"], {"progress-filled": 9}],
]

_COLOR_ZOO = ["ansi_cyan", "ansi_bright_red", "#ff0000", "red", "default", "", "ansi", "ansi_"]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.ui.theme import _theme_fields_from_css, rich_safe_color
from ytm_player.app._app import _app_css_variables, _theme_fields_from_theme

payload = json.load(sys.stdin)
out = {
    'css': [dict(_theme_fields_from_css(v)) for v in payload['css']],
    'app': [dict(_app_css_variables(dict(v), c)) for v, c in payload['app']],
    'theme': [dict(_theme_fields_from_theme(*raws, v)) for raws, v in payload['theme']],
    'color': [rich_safe_color(s) for s in payload['color']],
}
json.dump(out, sys.stdout)
"""


def _diff_report(nv: dict, fb: dict) -> str:
    keys = set(nv) | set(fb)
    diffs = {
        k: (nv.get(k, "<missing>"), fb.get(k, "<missing>"))
        for k in sorted(keys)
        if nv.get(k, "<missing>") != fb.get(k, "<missing>")
    }
    return f"native:   {nv!r}\nfallback: {fb!r}\nfields differing: {diffs!r}"


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
class TestThemeEngineNativeParity:
    def test_engines_match_fallback_over_the_zoos(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(
                {
                    "css": _CSS_ZOO,
                    "app": _APP_ZOO,
                    "theme": _THEME_ZOO,
                    "color": _COLOR_ZOO,
                }
            ),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        from ytm_player.app._app import _app_css_variables, _theme_fields_from_theme
        from ytm_player.ui.theme import _theme_fields_from_css, rich_safe_color

        native_css = [dict(_theme_fields_from_css(dict(v))) for v in _CSS_ZOO]
        for variables, nv, fb in zip(_CSS_ZOO, native_css, fallback["css"]):
            assert nv == fb, (
                f"css fields differ for variables={variables!r}:\n{_diff_report(nv, fb)}"
            )

        native_app = [dict(_app_css_variables(dict(v), c)) for v, c in _APP_ZOO]
        for (variables, colors), nv, fb in zip(_APP_ZOO, native_app, fallback["app"]):
            assert nv == fb, (
                f"app overlay differs for variables={variables!r} colors={colors!r}:\n"
                f"{_diff_report(nv, fb)}"
            )

        native_theme = [dict(_theme_fields_from_theme(*raws, v)) for raws, v in _THEME_ZOO]
        for (raws, variables), nv, fb in zip(_THEME_ZOO, native_theme, fallback["theme"]):
            assert nv == fb, (
                f"theme fields differ for raws={raws!r} variables={variables!r}:\n"
                f"{_diff_report(nv, fb)}"
            )

        native_color = [rich_safe_color(s) for s in _COLOR_ZOO]
        assert native_color == fallback["color"]

    def test_from_css_variables_matches_fallback_through_the_classmethod(self):
        # The public classmethod on the native backend vs a child
        # interpreter on the fallback, with theme.toml neutralised on both
        # sides so only the resolution is compared.
        child = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from unittest.mock import patch

from ytm_player.ui.theme import ThemeColors

payload = json.load(sys.stdin)
with patch.object(ThemeColors, '_apply_toml_overrides', lambda self: None):
    out = [dict(vars(ThemeColors.from_css_variables(v))) for v in payload['css']]
json.dump(out, sys.stdout)
"""
        result = subprocess.run(
            [sys.executable, "-c", child],
            input=json.dumps({"css": _CSS_ZOO}),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)

        from ytm_player.ui.theme import ThemeColors

        with patch.object(ThemeColors, "_apply_toml_overrides", lambda self: None):
            native = [dict(vars(ThemeColors.from_css_variables(dict(v)))) for v in _CSS_ZOO]
        for variables, nv, fb in zip(_CSS_ZOO, native, fallback):
            assert nv == fb, (
                f"from_css_variables differs for {variables!r}:\n{_diff_report(nv, fb)}"
            )
