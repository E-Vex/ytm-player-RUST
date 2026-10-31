"""Tests for the transliteration utility module."""

from importlib.util import find_spec
from unittest.mock import patch

import pytest

from ytm_player.utils.transliteration import has_non_ascii, transliterate_line

_HAVE_NATIVE = find_spec("ytm_player.utils._ytm_player_native") is not None

# ── has_non_ascii ─────────────────────────────────────────────────────────────


def test_has_non_ascii_pure_latin():
    assert has_non_ascii("Hello world") is False


def test_has_non_ascii_empty():
    assert has_non_ascii("") is False


def test_has_non_ascii_japanese():
    assert has_non_ascii("東京") is True


def test_has_non_ascii_korean():
    assert has_non_ascii("서울") is True


def test_has_non_ascii_mixed():
    assert has_non_ascii("Hello 世界") is True


def test_has_non_ascii_ascii_punctuation():
    assert has_non_ascii("rock & roll!") is False


# ── transliterate_line ────────────────────────────────────────────────────────


def test_transliterate_empty():
    assert transliterate_line("") == ""


def test_transliterate_pure_ascii_passthrough():
    assert transliterate_line("Hello world") == "Hello world"


def test_transliterate_japanese():
    result = transliterate_line("東京")
    assert result.isascii()
    assert len(result) > 0


def test_transliterate_korean():
    result = transliterate_line("서울")
    assert result.isascii()
    assert len(result) > 0


def test_transliterate_chinese():
    result = transliterate_line("北京")
    assert result.isascii()
    assert len(result) > 0


def test_transliterate_cyrillic():
    result = transliterate_line("Москва")
    assert result.isascii()
    assert len(result) > 0


def test_transliterate_mixed_keeps_ascii():
    result = transliterate_line("Hello 世界")
    assert result.isascii()
    assert "Hello" in result


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_has_non_ascii_accented_latin():
    assert has_non_ascii("caf\u00e9") is True
    assert has_non_ascii("ni\u00f1o") is True


def test_has_non_ascii_whitespace_only():
    assert has_non_ascii("   ") is False
    assert has_non_ascii("\t\n") is False


def test_transliterate_accented_latin():
    result = transliterate_line("caf\u00e9")
    assert result.isascii()
    assert "caf" in result


def test_transliterate_arabic():
    result = transliterate_line("\u0645\u0631\u062d\u0628\u0627")
    assert result.isascii()
    assert len(result) > 0


def test_transliterate_cache_hit():
    """Verify lru_cache returns the same object on repeated calls."""
    transliterate_line.cache_clear()
    result1 = transliterate_line("東京")
    result2 = transliterate_line("東京")
    assert result1 is result2
    info = transliterate_line.cache_info()
    assert info.hits >= 1
    assert info.misses == 1
    transliterate_line.cache_clear()


def test_transliterate_ascii_returns_same_object():
    """Pure ASCII input returns the exact same string object (no copy)."""
    transliterate_line.cache_clear()
    s = "Hello world"
    assert transliterate_line(s) is s
    transliterate_line.cache_clear()


# ── Graceful fallback ─────────────────────────────────────────────────────────


def test_transliterate_fallback_without_anyascii():
    # When the native backend is installed it doesn't consult the
    # ``anyascii`` Python package at all, so the import-failure fallback
    # can only be exercised without it.
    if _HAVE_NATIVE:
        pytest.skip("native backend is installed; anyascii fallback unused")
    # Clear lru_cache so the patched import takes effect
    transliterate_line.cache_clear()
    with patch.dict("sys.modules", {"anyascii": None}):
        result = transliterate_line("東京")
        assert result == "東京"
    transliterate_line.cache_clear()
