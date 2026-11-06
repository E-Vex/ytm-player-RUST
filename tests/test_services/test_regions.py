"""Tests for the YouTube Music chart-region list."""

from __future__ import annotations

import pytest


class TestChartRegions:
    def test_module_imports(self):
        from ytm_player.services import regions

        assert hasattr(regions, "CHART_REGIONS")

    def test_regions_is_tuple_of_pairs(self):
        from ytm_player.services.regions import CHART_REGIONS

        assert isinstance(CHART_REGIONS, tuple)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in CHART_REGIONS)

    def test_codes_are_uppercase_2char(self):
        from ytm_player.services.regions import CHART_REGIONS

        for code, name in CHART_REGIONS:
            assert isinstance(code, str)
            assert isinstance(name, str)
            assert code.isupper(), f"non-uppercase code: {code!r}"
            assert len(code) == 2, f"non-2-char code: {code!r}"

    def test_names_non_empty(self):
        from ytm_player.services.regions import CHART_REGIONS

        for code, name in CHART_REGIONS:
            assert name.strip(), f"empty name for code {code!r}"

    def test_no_duplicate_codes(self):
        from ytm_player.services.regions import CHART_REGIONS

        codes = [c for c, _ in CHART_REGIONS]
        assert len(codes) == len(set(codes)), "duplicate codes present"

    def test_us_present(self):
        """US is the default region in settings; must always be available."""
        from ytm_player.services.regions import CHART_REGIONS

        codes = [c for c, _ in CHART_REGIONS]
        assert "US" in codes

    def test_global_first_then_alphabetical(self):
        """Global ("ZZ") sits at position 0 as the default; rest alphabetical."""
        from ytm_player.services.regions import CHART_REGIONS

        assert CHART_REGIONS[0] == ("ZZ", "Global")
        rest = [name for _, name in CHART_REGIONS[1:]]
        assert rest == sorted(rest), "non-global regions not alphabetical by name"

    def test_minimum_size(self):
        """YouTube's advertised list is 62 codes (including ZZ = Global);
        we add 6 historically-supported ones outside that list. Total ≥ 60."""
        from ytm_player.services.regions import CHART_REGIONS

        assert len(CHART_REGIONS) >= 60

    def test_spain_present(self):
        """Spain was reported missing in issue #73 (user tried locale-style ES-ES)."""
        from ytm_player.services.regions import CHART_REGIONS

        codes = [c for c, _ in CHART_REGIONS]
        assert "ES" in codes


class TestNormaliseRegion:
    def test_two_letter_passthrough(self):
        from ytm_player.services.regions import normalise_region

        assert normalise_region("ES") == "ES"
        assert normalise_region("US") == "US"

    def test_lowercase_uppercased(self):
        from ytm_player.services.regions import normalise_region

        assert normalise_region("es") == "ES"
        assert normalise_region("gb") == "GB"

    def test_locale_dash_stripped(self):
        """Locale territory, not language, determines the chart region."""
        from ytm_player.services.regions import normalise_region

        assert normalise_region("ES-ES") == "ES"
        assert normalise_region("ES-MX") == "MX"
        assert normalise_region("en-GB") == "GB"

    def test_locale_underscore_stripped(self):
        """es_ES (POSIX locale) → ES."""
        from ytm_player.services.regions import normalise_region

        assert normalise_region("es_ES") == "ES"
        assert normalise_region("en_GB") == "GB"

    @pytest.mark.parametrize("value", ["en-GB", "en_GB", " EN_gb "])
    async def test_charts_receives_territory_not_language(self, ytmusic_service, value):
        ytmusic_service._ytm.get_charts.return_value = {"songs": []}

        assert await ytmusic_service.get_charts(value) == {"songs": []}
        ytmusic_service._ytm.get_charts.assert_called_once_with(country="GB")

    def test_existing_bare_values_are_preserved(self):
        from ytm_player.services.regions import CHART_REGIONS, normalise_region

        for code, name in CHART_REGIONS:
            assert normalise_region(code.lower()) == code
            assert normalise_region(name) == name.upper()
        assert normalise_region("auto") == "AUTO"
        assert normalise_region("  ") == ""

    def test_whitespace_trimmed(self):
        from ytm_player.services.regions import normalise_region

        assert normalise_region("  es-ES  ") == "ES"

    def test_empty_returns_empty(self):
        from ytm_player.services.regions import normalise_region

        assert normalise_region("") == ""


class TestChartRegionsEngine:
    """The table on whichever backend is loaded — the native
    ``chart_regions`` engine wrapped into the tuple-of-tuples shape when
    the extension is built, the pure-Python literal otherwise."""

    def test_exact_length_is_68(self):
        """62 advertised codes (incl. ZZ) + 6 historically-supported extras."""
        from ytm_player.services.regions import CHART_REGIONS

        assert len(CHART_REGIONS) == 68

    def test_entries_are_str_pairs(self):
        from ytm_player.services.regions import CHART_REGIONS

        for code, name in CHART_REGIONS:
            assert type(code) is str
            assert type(name) is str

    def test_known_anchors(self):
        from ytm_player.services.regions import CHART_REGIONS

        table = dict(CHART_REGIONS)
        assert table["ES"] == "Spain"
        assert table["US"] == "United States"
        assert table["JP"] == "Japan"
        assert table["MY"] == "Malaysia"

    def test_extras_outside_the_advertised_list_present(self):
        """HK/MY/SG/TW/TH/VN — historically returned chart data even
        though YouTube's advertised list omits them."""
        from ytm_player.services.regions import CHART_REGIONS

        codes = {c for c, _ in CHART_REGIONS}
        assert {"HK", "MY", "SG", "TW", "TH", "VN"} <= codes


# ── native vs fallback parity ────────────────────────────────────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestNativeParity:
    """When the Rust extension is built, the native chart-region table
    must answer exactly the literal the pure-Python fallback binds —
    every pair, in the reference's order — so the two spellings cannot
    drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_table_matches_fallback_literal(self):
        import json
        import subprocess
        import sys

        prog = (
            "import sys; "
            "sys.modules['ytm_player.utils._ytm_player_native'] = None; "
            "from ytm_player.services.regions import CHART_REGIONS; "
            "import json; print(json.dumps(CHART_REGIONS))"
        )
        out = subprocess.run(
            [sys.executable, "-c", prog], capture_output=True, text=True, check=True
        )
        fallback = [tuple(pair) for pair in json.loads(out.stdout)]
        native = list(_ytm_player_native.chart_regions())
        assert native == fallback
        # And the module binding is the wrapped native table.
        from ytm_player.services.regions import CHART_REGIONS

        assert tuple(native) == CHART_REGIONS
