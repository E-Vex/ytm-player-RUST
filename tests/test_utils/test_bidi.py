"""Tests for BiDi utilities (bidi.py).

With bidi_mode="auto" (default), reorder_rtl_line and wrap_rtl_line
pass text through unchanged — reordering is disabled because it causes
double-reversal on BiDi terminals and wrong reading order on non-BiDi
terminals.

Tests for the internal _do_reorder function verify the UAX #9 logic
still works when bidi_mode="reorder" is explicitly set.
"""

import sys

import pytest

import ytm_player.utils.bidi as bidi
from ytm_player.utils.bidi import (
    FSI,
    PDI,
    _do_reorder,
    has_rtl,
    isolate_bidi,
    reorder_rtl_line,
    reset_bidi_cache,
    wrap_rtl_line,
)

# ── has_rtl ──────────────────────────────────────────────────────────


class TestHasRtl:
    def test_pure_english(self):
        assert has_rtl("Hello World") is False

    def test_pure_arabic(self):
        assert has_rtl("مرحبا بالعالم") is True

    def test_pure_hebrew(self):
        assert has_rtl("שלום עולם") is True

    def test_mixed_arabic_english(self):
        assert has_rtl("Hello عالم") is True

    def test_empty_string(self):
        assert has_rtl("") is False

    def test_numbers_only(self):
        assert has_rtl("12345") is False

    def test_arabic_presentation_forms(self):
        # U+FE70 = Arabic Presentation Form-B
        assert has_rtl("\ufe70") is True

    def test_emoji_no_rtl(self):
        assert has_rtl("\U0001f3b5 Music") is False


# ── Default mode (auto/passthrough): text passes through unchanged ───


class TestDefaultPassthrough:
    """With bidi_mode='auto', reorder functions return text unchanged."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        reset_bidi_cache()
        yield
        reset_bidi_cache()

    def test_reorder_returns_unchanged(self):
        text = "يا ليل يا عين"
        assert reorder_rtl_line(text) == text

    def test_wrap_returns_unchanged(self):
        text = "يا ليل يا عين"
        assert wrap_rtl_line(text, 80) == text

    def test_reorder_pure_english_unchanged(self):
        assert reorder_rtl_line("Hello World") == "Hello World"

    def test_reorder_empty(self):
        assert reorder_rtl_line("") == ""

    def test_wrap_empty(self):
        assert wrap_rtl_line("", 80) == ""

    def test_wrap_pure_english_unchanged(self):
        assert wrap_rtl_line("Hello World", 80) == "Hello World"


# ── _do_reorder (internal UAX #9 logic) ─────────────────────────────


class TestDoReorderPureRtl:
    def test_two_arabic_words_reversed(self):
        assert _do_reorder("عيون القلب") == "القلب عيون"

    def test_four_arabic_words(self):
        assert _do_reorder("يا ليل يا عين") == "عين يا ليل يا"

    def test_arabic_with_tashkeel(self):
        assert _do_reorder("بِسْمِ اللَّهِ") == "اللَّهِ بِسْمِ"

    def test_hebrew(self):
        assert _do_reorder("שלום עולם") == "עולם שלום"


class TestDoReorderMixed:
    def test_rtl_base_with_english_suffix(self):
        assert _do_reorder("حبيبي (Remix)") == "(Remix) حبيبي"

    def test_ltr_base_with_arabic_suffix(self):
        assert _do_reorder("Beautiful Day - محمد حماقي") == "Beautiful Day - حماقي محمد"

    def test_rtl_base_with_embedded_english(self):
        assert _do_reorder("كلمات DJ Khaled أغنية") == "أغنية DJ Khaled كلمات"

    def test_rtl_base_feat(self):
        assert _do_reorder("فيروز feat. Rahbani") == "feat. Rahbani فيروز"

    def test_rtl_with_number(self):
        assert _do_reorder("أغنية رقم 3") == "3 رقم أغنية"

    def test_rtl_with_dash(self):
        assert _do_reorder("عمرو دياب - تملي معاك") == "معاك تملي - دياب عمرو"

    def test_ltr_base_keeps_ltr_order(self):
        assert _do_reorder("Song by فنان العرب is great") == "Song by العرب فنان is great"

    def test_multiple_ltr_blocks_in_rtl(self):
        assert _do_reorder("كلمة Hello كلمة World كلمة") == "كلمة World كلمة Hello كلمة"


class TestDoReorderEdgeCases:
    def test_arabic_with_parentheses(self):
        assert _do_reorder("أنشودة (رائعة)") == "(رائعة) أنشودة"

    def test_emoji_with_arabic(self):
        assert _do_reorder("🎵 أغنية جميلة") == "جميلة أغنية 🎵"

    def test_arabic_comma(self):
        assert _do_reorder("أحمد، محمد") == "محمد أحمد،"

    def test_arabic_question_mark(self):
        assert _do_reorder("ما هذا؟") == "هذا؟ ما"

    def test_empty_string(self):
        assert _do_reorder("") == ""

    def test_whitespace_only(self):
        assert _do_reorder("   ") == "   "

    def test_single_word(self):
        assert _do_reorder("مرحبا") == "مرحبا"


# ── isolate_bidi ─────────────────────────────────────────────────────


class TestIsolateBidi:
    """isolate_bidi wraps RTL-containing text in FSI...PDI to prevent
    BiDi bleed across visual chunks (table cells, playback bar fragments)."""

    def test_arabic_wrapped(self):
        assert isolate_bidi("مرحبا") == f"{FSI}مرحبا{PDI}"

    def test_hebrew_wrapped(self):
        assert isolate_bidi("שלום") == f"{FSI}שלום{PDI}"

    def test_pure_latin_unchanged_by_default(self):
        assert isolate_bidi("Hello") == "Hello"

    def test_pure_latin_wrapped_when_forced(self):
        assert isolate_bidi("Hello", only_if_rtl=False) == f"{FSI}Hello{PDI}"

    def test_empty_unchanged(self):
        assert isolate_bidi("") == ""

    def test_idempotent(self):
        once = isolate_bidi("مرحبا")
        twice = isolate_bidi(once)
        assert once == twice

    def test_mixed_content_wrapped(self):
        result = isolate_bidi("Song مرحبا")
        assert result.startswith(FSI)
        assert result.endswith(PDI)
        assert "مرحبا" in result

    def test_zero_cell_width_via_rich(self):
        """The contract: isolation marks must not consume display columns."""
        from rich.cells import cell_len

        plain = "مرحبا"
        wrapped = isolate_bidi(plain)
        assert cell_len(wrapped) == cell_len(plain)


# ── Render-site enforcement (regression guard) ───────────────────────


class TestIsolateBidiCallSites:
    """Verify that every render site that displays user-supplied text
    calls isolate_bidi.  This guards against re-introducing the BiDi
    bleed bug where Arabic titles appear duplicated at row edges or
    bleed into adjacent widgets in the playback bar.

    If you move or rename these render sites, update this test —
    do NOT just delete entries without ensuring isolation is preserved.
    """

    REQUIRED_SITES = {
        "src/ytm_player/ui/widgets/track_table.py": "isolate_bidi",
        "src/ytm_player/ui/playback_bar.py": "isolate_bidi",
        "src/ytm_player/ui/sidebars/lyrics_sidebar.py": "isolate_bidi",
    }

    def test_render_sites_call_isolate_bidi(self):
        import pathlib

        repo_root = pathlib.Path(__file__).parent.parent.parent
        for rel_path, token in self.REQUIRED_SITES.items():
            full_path = repo_root / rel_path
            assert full_path.exists(), f"{rel_path} does not exist"
            text = full_path.read_text(encoding="utf-8")
            assert token in text, (
                f"{rel_path} must call {token} to prevent RTL BiDi bleed. "
                "Wrap user text fragments (title, artist, album, lyrics) "
                "with isolate_bidi() AFTER truncate()."
            )


# ── Gated reorder (bidi_mode="reorder" forced on) ────────────────────


class TestGatedReorderEnabled:
    """The public reorder_rtl_line/wrap_rtl_line with the cached
    ``bidi_mode`` verdict forced on — the render-path behaviour every
    fragment funnels through when a user opts into reordering."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        reset_bidi_cache()
        bidi._should_reorder = True
        yield
        reset_bidi_cache()

    def test_reorder_reverses_rtl_words(self):
        assert reorder_rtl_line("عيون القلب") == "القلب عيون"

    def test_reorder_mixed_keeps_ltr_blocks(self):
        assert reorder_rtl_line("كلمات DJ Khaled أغنية") == "أغنية DJ Khaled كلمات"

    def test_reorder_ltr_text_unchanged(self):
        assert reorder_rtl_line("Hello World") == "Hello World"

    def test_reorder_empty_unchanged(self):
        assert reorder_rtl_line("") == ""

    def test_reorder_falsy_passthrough(self):
        # The falsy gate runs before any native call, so None (and any
        # other falsy input a defensive caller might pass) passes
        # through exactly like the pure-Python reference.
        assert reorder_rtl_line(None) is None  # type: ignore[arg-type]

    def test_wrap_short_line_reorders_without_wrapping(self):
        # len(text) <= width answers the bare line reorder.
        assert wrap_rtl_line("عيون القلب", 80) == "القلب عيون"

    def test_wrap_zero_width_reorders_without_wrapping(self):
        # width <= 0 answers the bare line reorder too.
        assert wrap_rtl_line("عيون القلب", 0) == "القلب عيون"

    def test_wrap_negative_width_reorders_without_wrapping(self):
        assert wrap_rtl_line("عيون القلب", -3) == "القلب عيون"

    def test_wrap_splits_over_wide_lines(self):
        assert wrap_rtl_line("يا ليل يا عين طويل", 9) == "يا ليل يا\nطويل عين"

    def test_wrap_over_wide_word_gets_its_own_line(self):
        assert wrap_rtl_line("مرحبا", 2) == "مرحبا"

    def test_wrap_ltr_text_unchanged(self):
        assert wrap_rtl_line("Hello World", 4) == "Hello World"


class TestGatedReorderDisabled:
    """The same public functions with the cached verdict forced off —
    the passthrough every fragment answers when reordering is disabled
    (the default ``bidi_mode="auto"``), RTL text included."""

    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        reset_bidi_cache()
        bidi._should_reorder = False
        yield
        reset_bidi_cache()

    def test_reorder_rtl_unchanged(self):
        assert reorder_rtl_line("عيون القلب") == "عيون القلب"

    def test_wrap_rtl_unchanged(self):
        assert wrap_rtl_line("يا ليل يا عين طويل", 9) == "يا ليل يا عين طويل"


# ── native vs fallback parity ────────────────────────────────────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
class TestGatedEngineContract:
    """The gated native engines taken directly — the empty/no-RTL/disabled
    passthroughs and the folded ``bidi_mode`` verdict, independent of the
    Python wrappers that read the cached setting."""

    def test_reorder_passthrough_when_disabled(self):
        assert _ytm_player_native.reorder_rtl_line("عيون القلب", False) == "عيون القلب"

    def test_reorder_passthrough_without_rtl(self):
        assert _ytm_player_native.reorder_rtl_line("Hello World", True) == "Hello World"

    def test_reorder_passthrough_empty(self):
        assert _ytm_player_native.reorder_rtl_line("", True) == ""

    def test_reorder_reverses_when_enabled(self):
        assert _ytm_player_native.reorder_rtl_line("عيون القلب", True) == "القلب عيون"

    def test_wrap_passthrough_when_disabled(self):
        assert _ytm_player_native.wrap_rtl_line("يا ليل يا عين طويل", 9, False) == (
            "يا ليل يا عين طويل"
        )

    def test_wrap_passthrough_without_rtl(self):
        assert _ytm_player_native.wrap_rtl_line("Hello World", 4, True) == "Hello World"

    def test_wrap_passthrough_empty(self):
        assert _ytm_player_native.wrap_rtl_line("", 4, True) == ""

    def test_wrap_pipeline_when_enabled(self):
        assert _ytm_player_native.wrap_rtl_line("يا ليل يا عين طويل", 9, True) == (
            "يا ليل يا\nطويل عين"
        )

    def test_wrap_short_line_becomes_bare_reorder(self):
        assert _ytm_player_native.wrap_rtl_line("عيون القلب", 80, True) == "القلب عيون"


# Op sequences the two implementations must agree on, case for case:
# the reorder across pure-RTL / mixed / neutral / empty / whitespace
# inputs with the gate both on and off, and the wrap across the width
# short-circuit edges (0, negative, exact-len, len+1), the join-budget
# splits, the over-wide-word own line and the per-line reorder join.
_PARITY_CASES: list = [
    ["reorder", "عيون القلب", 0, True],
    ["reorder", "يا ليل يا عين", 0, True],
    ["reorder", "بِسْمِ اللَّهِ", 0, True],
    ["reorder", "שלום עולם", 0, True],
    ["reorder", "Beautiful Day - محمد حماقي", 0, True],
    ["reorder", "كلمات DJ Khaled أغنية", 0, True],
    ["reorder", "فيروز feat. Rahbani", 0, True],
    ["reorder", "أغنية رقم 3", 0, True],
    ["reorder", "أنشودة (رائعة)", 0, True],
    ["reorder", "🎵 أغنية جميلة", 0, True],
    ["reorder", "أحمد، محمد", 0, True],
    ["reorder", "مرحبا", 0, True],
    ["reorder", "Hello World", 0, True],
    ["reorder", "12345", 0, True],
    ["reorder", "\U0001f3b5 Music", 0, True],
    ["reorder", "", 0, True],
    ["reorder", "   ", 0, True],
    ["reorder", "عيون القلب", 0, False],
    ["reorder", "يا ليل يا عين", 0, False],
    ["reorder", "Hello World", 0, False],
    ["wrap", "يا ليل يا عين طويل", 9, True],
    ["wrap", "يا ليل يا عين طويل", 9, False],
    ["wrap", "أغنية جميلة جدا جدا جدا", 5, True],
    ["wrap", "مرحبا", 2, True],
    ["wrap", "مرحبا", 3, True],
    ["wrap", "مرحبا", 4, True],
    ["wrap", "مرحبا", 80, True],
    ["wrap", "مرحبا", 0, True],
    ["wrap", "مرحبا", -3, True],
    ["wrap", "عيون القلب", 0, True],
    ["wrap", "عيون القلب", -1, True],
    ["wrap", "Hello عالم", 4, True],
    ["wrap", "ش ك ي", 1, True],
    ["wrap", "Hello World", 4, True],
    ["wrap", "  ", 1, True],
    ["wrap", "", 3, True],
]

_CHILD_PROG = """
import json, sys
sys.modules["ytm_player.utils._ytm_player_native"] = None
import ytm_player.utils.bidi as bidi
out = []
for op, text, width, enabled in json.load(sys.stdin):
    bidi._should_reorder = enabled
    if op == "reorder":
        out.append(bidi.reorder_rtl_line(text))
    else:
        out.append(bidi.wrap_rtl_line(text, width))
print(json.dumps(out, ensure_ascii=False))
"""


def _run_cases(cases: list, in_child: bool) -> list:
    """Run *cases* against the public gated functions — in a child
    interpreter with the extension import blocked (the pure-Python
    fallback) when *in_child*, in-process (the native-backed module)
    otherwise."""
    import json
    import subprocess

    if in_child:
        out = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(cases),
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(out.stdout)
    results = []
    saved = bidi._should_reorder
    try:
        for op, text, width, enabled in cases:
            bidi._should_reorder = enabled
            if op == "reorder":
                results.append(bidi.reorder_rtl_line(text))
            else:
                results.append(bidi.wrap_rtl_line(text, width))
    finally:
        bidi._should_reorder = saved
        reset_bidi_cache()
    return results


@pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
class TestNativeParity:
    """When the Rust extension is built, the gated public functions must
    answer every case exactly like the pure-Python fallback run in a
    child interpreter with the extension import blocked — the passthrough
    gates, the reorder reversals and the wrap splits included — so the
    two implementations cannot drift.
    """

    def test_gated_functions_match_fallback(self):
        native = _run_cases(_PARITY_CASES, in_child=False)
        fallback = _run_cases(_PARITY_CASES, in_child=True)
        for case, native_result, fallback_result in zip(_PARITY_CASES, native, fallback):
            assert native_result == fallback_result, (
                f"native and fallback bidi results differ for {case}: "
                f"{native_result!r} != {fallback_result!r}"
            )
