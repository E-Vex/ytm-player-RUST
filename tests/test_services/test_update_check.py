"""Tests for services/update_check.py — the PyPI version probe."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
from packaging.version import InvalidVersion, Version

from ytm_player.services.update_check import (
    _is_newer,
    check_for_update,
)

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestIsNewer:
    def test_newer_patch(self):
        assert _is_newer("1.6.1", "1.6.0") is True

    def test_newer_minor(self):
        assert _is_newer("1.7.0", "1.6.5") is True

    def test_same_version(self):
        assert _is_newer("1.6.0", "1.6.0") is False

    def test_older(self):
        assert _is_newer("1.5.9", "1.6.0") is False

    def test_unparseable_returns_false(self):
        assert _is_newer("garbage", "1.6.0") is False

    def test_double_digit_minor(self):
        # Trivially correct under tuple-int compare too, but this is the
        # canonical "lex compare gets it wrong" test that motivated the
        # switch to packaging.version.Version.
        assert _is_newer("1.10.0", "1.9.0") is True
        assert _is_newer("1.9.0", "1.10.0") is False

    def test_post_release_newer_than_release(self):
        # 1.6.0.post1 IS strictly newer than 1.6.0 under PEP 440. The
        # old hand-rolled parser couldn't see this — `.post1` had no
        # numeric prefix so the chunk parsed to () and _is_newer
        # returned False. packaging.version.Version handles it.
        assert _is_newer("1.6.0.post1", "1.6.0") is True

    def test_pre_release_older_than_release(self):
        assert _is_newer("1.6.0rc1", "1.6.0") is False
        assert _is_newer("1.6.0", "1.6.0rc1") is True


class TestCacheShape:
    """A wrong-shape update_check.json is discarded, never raised on."""

    def test_list_cache_is_ignored(self, tmp_path):
        cache = tmp_path / "update_check.json"
        cache.write_text("[]", encoding="utf-8")
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value="1.7.0",
        ) as fetch:
            result = check_for_update("1.6.0", cache)
        assert result == "1.7.0"
        fetch.assert_called_once()

    def test_bad_checked_at_is_ignored(self, tmp_path):
        cache = tmp_path / "update_check.json"
        cache.write_text(
            json.dumps({"checked_at": "bad", "latest": "9.9.9"}),
            encoding="utf-8",
        )
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value="1.7.0",
        ) as fetch:
            result = check_for_update("1.6.0", cache)
        assert result == "1.7.0"
        fetch.assert_called_once()

    def test_non_string_latest_is_ignored(self, tmp_path):
        cache = tmp_path / "update_check.json"
        cache.write_text(
            json.dumps({"checked_at": time.time(), "latest": 5}),
            encoding="utf-8",
        )
        with patch("ytm_player.services.update_check._fetch_latest_from_pypi") as fetch:
            result = check_for_update("1.6.0", cache)
        assert result is None
        fetch.assert_not_called()


class TestCheckForUpdate:
    def test_cache_within_24h_skips_network(self, tmp_path):
        cache = tmp_path / "update_check.json"
        cache.write_text(
            json.dumps({"checked_at": time.time(), "latest": "1.7.0"}),
            encoding="utf-8",
        )
        with patch("ytm_player.services.update_check._fetch_latest_from_pypi") as fetch:
            result = check_for_update("1.6.0", cache)
        assert result == "1.7.0"
        fetch.assert_not_called()

    def test_cache_within_24h_no_update_returns_none(self, tmp_path):
        cache = tmp_path / "update_check.json"
        cache.write_text(
            json.dumps({"checked_at": time.time(), "latest": "1.6.0"}),
            encoding="utf-8",
        )
        result = check_for_update("1.6.0", cache)
        assert result is None

    def test_stale_cache_triggers_fetch(self, tmp_path):
        cache = tmp_path / "update_check.json"
        cache.write_text(
            json.dumps({"checked_at": 0, "latest": "1.5.0"}),
            encoding="utf-8",
        )
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value="1.7.0",
        ):
            result = check_for_update("1.6.0", cache)
        assert result == "1.7.0"
        # Cache should now be updated.
        new_cache = json.loads(cache.read_text(encoding="utf-8"))
        assert new_cache["latest"] == "1.7.0"

    def test_no_cache_file_triggers_fetch(self, tmp_path):
        cache = tmp_path / "missing.json"
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value="1.7.0",
        ):
            result = check_for_update("1.6.0", cache)
        assert result == "1.7.0"
        assert cache.exists()

    def test_network_failure_returns_none(self, tmp_path):
        cache = tmp_path / "missing.json"
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value=None,
        ):
            result = check_for_update("1.6.0", cache)
        assert result is None
        # Cache NOT written on failure.
        assert not cache.exists()

    def test_pypi_returns_older_version(self, tmp_path):
        """Should not surface an "update" if PyPI has an older version (e.g. yanked)."""
        cache = tmp_path / "missing.json"
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value="1.5.0",
        ):
            result = check_for_update("1.6.0", cache)
        assert result is None
        # Cache IS written (we successfully fetched) — just nothing to surface.
        assert cache.exists()

    def test_clock_skew_triggers_fetch(self, tmp_path):
        """checked_at in the future (clock went backwards) → re-fetch."""
        cache = tmp_path / "update_check.json"
        cache.write_text(
            json.dumps({"checked_at": time.time() + 86400, "latest": "1.5.0"}),
            encoding="utf-8",
        )
        with patch(
            "ytm_player.services.update_check._fetch_latest_from_pypi",
            return_value="1.7.0",
        ):
            result = check_for_update("1.6.0", cache)
        assert result == "1.7.0"


# ── PEP 440 comparison contract ──────────────────────────────────────


def _packaging_verdict(latest: str, current: str) -> bool:
    """The installed packaging's own strictly-newer verdict — the engine's
    contract for spellings whose accept/reject depends on the grammar
    shape the installed release compiled."""
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return False


class TestIsNewerPep440:
    """The strictly-newer verdict over the wider PEP 440 space."""

    def test_epoch_orders(self):
        assert _is_newer("2!1.0", "1!9.9") is True  # epoch dominates the release
        assert _is_newer("1!2.0", "2!1.0") is False
        assert _is_newer("!1.0", "1.0") is False  # empty epoch == no epoch

    def test_local_segments_order(self):
        assert _is_newer("2.1.0+build.2", "2.1.0+build.1") is True
        assert _is_newer("2.1.0+2", "2.1.0+abc") is True  # str sorts before num
        assert _is_newer("2.1.0", "2.1.0+local") is False  # absent local first
        assert _is_newer("2.1.0+5.1", "2.1.0+5") is True

    def test_trailing_zero_release_equality(self):
        assert _is_newer("2.1.0.0", "2.1") is False
        assert _is_newer("1.0.0.0.1", "1.0.0.0.1") is False

    def test_pre_post_dev_ordering(self):
        assert _is_newer("2.1.0rc1", "2.1.0b2") is True
        assert _is_newer("2.1.0.dev5", "2.1.0a1") is False  # dev sorts before alpha
        assert _is_newer("1.0a2.post2.dev3", "1.0") is False
        assert _is_newer("1.0.post1.dev1", "1.0.post1") is False

    def test_v_prefix_is_accepted(self):
        assert _is_newer("v1.7.0", "1.6.9") is True

    def test_both_invalid_is_false(self):
        assert _is_newer("garbage", "also garbage") is False

    def test_invalid_current_is_false(self):
        assert _is_newer("1.7.0", "not a version") is False

    def test_over_limit_latest_raises_plain_valueerror(self):
        """A numeric component past CPython's int-conversion digit limit is
        the constructor's own plain ValueError — NOT an InvalidVersion —
        so the reference's ``except InvalidVersion`` lets it propagate."""
        with pytest.raises(ValueError) as excinfo:
            _is_newer("1." + "0" * 5000, "1.0")
        assert not isinstance(excinfo.value, InvalidVersion)

    def test_over_limit_current_raises_plain_valueerror(self):
        with pytest.raises(ValueError) as excinfo:
            _is_newer("1.0", "1." + "0" * 5000)
        assert not isinstance(excinfo.value, InvalidVersion)

    def test_over_limit_local_segment_raises_plain_valueerror(self):
        with pytest.raises(ValueError) as excinfo:
            _is_newer("1.0+" + "9" * 5000, "1.0")
        assert not isinstance(excinfo.value, InvalidVersion)

    def test_invalid_latest_short_circuits_before_current_is_parsed(self):
        """``Version(latest)`` evaluates first: an invalid latest answers
        False without ever parsing the current string — even one whose
        parse would have raised the digit-limit ValueError."""
        assert _is_newer("garbage", "1." + "0" * 5000) is False

    def test_over_limit_latest_raises_before_current_is_parsed(self):
        """Dually, an over-limit latest raises before an invalid current
        could answer False."""
        with pytest.raises(ValueError):
            _is_newer("1." + "0" * 5000, "garbage")


# Unicode-fold spellings: through packaging 26.0 the grammar (plain
# re.IGNORECASE on a Unicode pattern) admits them wherever a letter can
# appear; 26.1+ scopes the groups (?a:...) and rejects them. The verdict
# for such spellings therefore follows the installed release — the same
# contract the crash-staleness engine carries.
_FOLD_CASES = [
    ("1.0+\u017f", "1.0"),
    ("1.0+\u017f", "1.0+s"),
    ("1.0+\u212a", "1.0+k"),
    ("1.0+\u0130", "1.0+i"),
    ("1.0+\u0131", "1.0+i"),
    ("1.0po\u017ft1", "1.0.post1"),
    ("2.1.0prev\u0131ew1", "2.1.0rc1"),
    ("2.1.0prev\u0131ew1", "2.1.0b1"),
    ("2.1.0prev\u0130ew1", "2.1.0preview1"),
]


class TestIsNewerFolds:
    """Fold spellings follow the installed packaging's grammar."""

    @pytest.mark.parametrize("latest, current", _FOLD_CASES)
    def test_fold_verdicts_match_the_installed_packaging(self, latest, current):
        assert _is_newer(latest, current) is _packaging_verdict(latest, current)

    @pytest.mark.parametrize("latest, current", _FOLD_CASES)
    def test_fold_verdicts_are_symmetric(self, latest, current):
        """Either way round, the verdict is the installed packaging's."""
        assert _is_newer(current, latest) is _packaging_verdict(current, latest)


# ── Native / fallback parity ─────────────────────────────────────────

# (latest, current) pairs, JSON-round-trippable. The matrix covers the
# release/pre/post/dev/local ordering space, epochs, trailing zeros, the
# fold spellings, invalid spellings on either side, and the digit-limit
# overflows (whose ValueError both backends raise).
_IS_NEWER_PARITY_CASES = [
    ("1.6.1", "1.6.0"),
    ("1.6.0", "1.6.0"),
    ("1.5.9", "1.6.0"),
    ("1.10.0", "1.9.0"),
    ("1.9.0", "1.10.0"),
    ("1.6.0.post1", "1.6.0"),
    ("1.6.0rc1", "1.6.0"),
    ("1.6.0", "1.6.0rc1"),
    ("1!2.0", "2!1.0"),
    ("2!1.0", "1!9.9"),
    ("!1.0", "1.0"),
    ("1.0.0-1", "1.0"),
    ("1.0-1", "1.0-po\u017ft"),
    ("1.0a2.post2.dev3", "1.0"),
    ("2.1.0+build.1", "2.1.0+build.2"),
    ("2.1.0+9", "2.1.0+10"),
    ("2.1.0+2", "2.1.0+abc"),
    ("2.1.0", "2.1.0+local"),
    ("007.0002", "7.3"),
    ("2.1.0.0", "2.1"),
    ("1.0.0.0.1", "1.0.0.0.1"),
    ("v1.7.0", "1.6.9"),
    ("2.1.0alpha2", "2.1.0b1"),
    ("2.1.0AlPhA1", "2.1.0"),
    ("2.1.0preview3", "2.1.0rc1"),
    ("2.1.0_c3", "2.1.0rc3"),
    ("2.1.0.dev5", "2.1.0a1"),
    ("2.1.0dev", "2.1.0"),
    ("garbage", "1.6.0"),
    ("1.7.0", "not a version"),
    ("garbage", "also garbage"),
    ("garbage", "1." + "0" * 5000),
    ("1." + "0" * 5000, "1.0"),
    ("1.0", "1." + "0" * 5000),
    ("1." + "0" * 5000, "garbage"),
    ("1.0+" + "9" * 5000, "1.0"),
    ("1.0rc" + "9" * 5000, "1.0"),
    # Unicode-fold spellings — the verdict follows the installed
    # packaging on both backends (the native engine probes it).
    ("1.0+\u017f", "1.0"),
    ("1.0+\u017f", "1.0+s"),
    ("1.0+\u212a", "1.0+k"),
    ("1.0+\u0130", "1.0+i"),
    ("1.0+\u0131", "1.0+i"),
    ("1.0po\u017ft1", "1.0.post1"),
    ("2.1.0prev\u0131ew1", "2.1.0rc1"),
    ("2.1.0prev\u0131ew1", "2.1.0b1"),
    ("2.1.0prev\u0130ew1", "2.1.0preview1"),
]

# The child interpreter runs this program with the case list on stdin and
# prints the fallback engine's verdicts (or the ValueError marker for the
# digit-limit overflows) as JSON.
_IS_NEWER_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services.update_check import _is_newer

cases = json.load(sys.stdin)

out = []
for latest, current in cases:
    try:
        out.append(_is_newer(latest, current))
    except ValueError:
        out.append({"error": "ValueError"})
json.dump(out, sys.stdout)
"""


class TestIsNewerNativeParity:
    """When the Rust extension is built, the native strictly-newer engine
    must produce exactly the same verdicts (and the same digit-limit
    ValueErrors) as the pure-Python reference for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_native_matches_fallback(self):
        native = []
        for latest, current in _IS_NEWER_PARITY_CASES:
            try:
                native.append(_is_newer(latest, current))
            except ValueError:
                native.append({"error": "ValueError"})
        native = json.loads(json.dumps(native))

        fallback = json.loads(
            subprocess.run(
                [sys.executable, "-c", _IS_NEWER_CHILD_PROG],
                input=json.dumps(_IS_NEWER_PARITY_CASES),
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )

        for i, case in enumerate(_IS_NEWER_PARITY_CASES):
            assert native[i] == fallback[i], (
                f"native and fallback _is_newer verdicts differ for case {i}: {case}"
            )

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_both_packaging_grammar_shapes_are_reachable(self):
        """The engine's fold flag selects between the two grammars
        packaging has shipped, regardless of which one is installed.

        With ``admit_unicode_folds=True`` (packaging <= 26.0's unscoped
        ``re.IGNORECASE``) the fold spellings parse and order by their raw
        characters; with ``False`` (26.1+ ``(?a:...)`` scoping) the same
        spellings are InvalidVersion — a False verdict. Every ASCII
        spelling gives the identical verdict either way: the grammars
        differ in exactly the positions a fold can reach, none of which
        an ASCII spelling touches.
        """
        from ytm_player.utils._ytm_player_native import pep440_is_newer

        ascii_cases = [
            (latest, current)
            for latest, current in _IS_NEWER_PARITY_CASES
            if latest.isascii() and current.isascii() and len(latest) < 100 and len(current) < 100
        ]
        assert ascii_cases, "the parity matrix must carry ASCII spellings"
        for latest, current in ascii_cases:
            assert pep440_is_newer(latest, current, True) == pep440_is_newer(
                latest, current, False
            ), (latest, current)

        # With the fold grammar, the fold spellings parse and order by
        # their raw characters: a fold local beats the plain local's
        # absence, and the exotic pre spellings rank below rc.
        assert pep440_is_newer("1.0+\u017f", "1.0", True) is True
        assert pep440_is_newer("1.0+\u017f", "1.0+s", True) is True  # ſ > s raw
        assert pep440_is_newer("2.1.0prev\u0131ew1", "2.1.0rc1", True) is False
        assert pep440_is_newer("2.1.0prev\u0131ew1", "2.1.0b1", True) is True
        # With the ASCII grammar they are InvalidVersion — a False verdict.
        assert pep440_is_newer("1.0+\u017f", "1.0", False) is False
        assert pep440_is_newer("1.0+\u017f", "1.0+s", False) is False
        assert pep440_is_newer("2.1.0prev\u0131ew1", "2.1.0rc1", False) is False
        assert pep440_is_newer("2.1.0prev\u0131ew1", "2.1.0b1", False) is False

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_production_wrapper_probe_steers_the_engine(self):
        """The production ``_is_newer`` wrapper probes the installed
        packaging once and passes the verdict down, so it answers exactly
        what the raw engine with the probed grammar shape answers."""
        from ytm_player.services.update_check import _packaging_admits_unicode_folds
        from ytm_player.utils._ytm_player_native import pep440_is_newer

        folds = _packaging_admits_unicode_folds()
        for latest, current in _FOLD_CASES + [
            ("1.7.0", "1.6.0"),
            ("garbage", "1.6.0"),
            ("1.6.0.post1", "1.6.0"),
        ]:
            expected = pep440_is_newer(latest, current, folds)
            assert _is_newer(latest, current) == expected, (latest, current)
