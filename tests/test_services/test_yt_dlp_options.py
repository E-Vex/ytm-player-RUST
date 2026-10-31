import logging
from pathlib import Path

import pytest

from ytm_player.config.settings import YtDlpSettings
from ytm_player.services.yt_dlp_options import (
    apply_configured_yt_dlp_options,
    normalize_cafile,
    normalize_cookiefile,
    normalize_js_runtimes,
    normalize_remote_components,
)


def test_normalize_cookiefile_expands_home():
    got = normalize_cookiefile("~/cookies.txt")
    assert got is not None
    # Use Path semantics so this works on Windows (backslash) too.
    p = Path(got)
    assert p.name == "cookies.txt"
    assert p.is_absolute()


def test_normalize_cookiefile_empty_none():
    assert normalize_cookiefile("") is None
    assert normalize_cookiefile(None) is None


def test_normalize_remote_components_string_to_list():
    assert normalize_remote_components("ejs:github, ejs:npm") == ["ejs:github", "ejs:npm"]


def test_normalize_remote_components_empty_and_none():
    assert normalize_remote_components("") is None
    assert normalize_remote_components([]) is None
    assert normalize_remote_components(None) is None


def test_normalize_js_runtimes_string_to_dict():
    assert normalize_js_runtimes("bun") == {"bun": {}}


def test_normalize_js_runtimes_with_paths():
    assert normalize_js_runtimes("bun:/path/to/bun node:/path/to/node") == {
        "bun": {"path": "/path/to/bun"},
        "node": {"path": "/path/to/node"},
    }


def test_normalize_js_runtimes_list_to_dict():
    assert normalize_js_runtimes(["bun", "node:/opt/node"]) == {
        "bun": {},
        "node": {"path": "/opt/node"},
    }


def test_normalize_js_runtimes_empty_and_none():
    assert normalize_js_runtimes("") is None
    assert normalize_js_runtimes([]) is None
    assert normalize_js_runtimes(None) is None


def test_apply_configured_options_converts_all_fields():
    settings = YtDlpSettings(
        cookies_file="~/cookies.txt",
        remote_components="ejs:github",
        js_runtimes="bun:/tmp/bun",
    )

    opts = apply_configured_yt_dlp_options({"quiet": True}, settings)

    assert opts["quiet"] is True
    assert Path(opts["cookiefile"]).name == "cookies.txt"
    assert opts["remote_components"] == ["ejs:github"]
    assert opts["js_runtimes"] == {"bun": {"path": "/tmp/bun"}}


def test_apply_configured_options_warns_when_remote_components_enabled(caplog):
    settings = YtDlpSettings(remote_components="ejs:github")
    with caplog.at_level(logging.WARNING):
        opts = apply_configured_yt_dlp_options({"quiet": True}, settings)

    assert opts["remote_components"] == ["ejs:github"]
    assert "remote_components is enabled" in caplog.text


def test_apply_configured_options_skips_unset_fields():
    settings = YtDlpSettings()
    opts = apply_configured_yt_dlp_options({"quiet": True}, settings)
    assert opts == {"quiet": True}


def test_normalize_cafile_expands_home():
    got = normalize_cafile("~/certs/corporate.pem")
    assert got is not None
    p = Path(got)
    assert p.name == "corporate.pem"
    assert p.is_absolute()


def test_normalize_cafile_empty_and_none():
    assert normalize_cafile("") is None
    assert normalize_cafile(None) is None


def test_apply_configured_options_sets_cafile():
    settings = YtDlpSettings(ca_bundle="~/certs/corporate.pem")
    opts = apply_configured_yt_dlp_options({"quiet": True}, settings)
    # ca_bundle triggers no-certifi so yt-dlp uses system/SSL_CERT_FILE certs.
    assert "cafile" not in opts
    assert opts["compat_opts"] == {"no-certifi"}
    assert opts["quiet"] is True


def test_apply_configured_options_omits_cafile_when_unset():
    settings = YtDlpSettings()
    opts = apply_configured_yt_dlp_options({"quiet": True}, settings)
    assert "cafile" not in opts


# ── native vs fallback parity ────────────────────────────────────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

# Each case is a Python expression evaluated against
# ytm_player.services.yt_dlp_options in two interpreters: this one (native
# extension loaded) and a child where the extension import is blocked
# (pure-Python fallback). Their reprs must match exactly.
_PARITY_CASES = [
    "normalize_remote_components('ejs:github, ejs:npm')",
    "normalize_remote_components('  a ,, b\\tc  ')",
    "normalize_remote_components([' a ', 'b'])",
    "normalize_remote_components((' a ', 2, None))",
    "normalize_remote_components('')",
    "normalize_remote_components([])",
    "normalize_remote_components(None)",
    "normalize_js_runtimes('bun')",
    "normalize_js_runtimes('bun:/path/to/bun node:/path/to/node')",
    "normalize_js_runtimes(['bun', 'node:/opt/node'])",
    # Duplicate names: later token wins, first-insert position is kept.
    "normalize_js_runtimes('Deno:/a BUN:/b deno:/c')",
    # Trailing colon with no path, blank runtime, whitespace-only input.
    "normalize_js_runtimes('bun:')",
    "normalize_js_runtimes(':')",
    "normalize_js_runtimes('  ')",
    "normalize_js_runtimes('')",
    "normalize_js_runtimes([])",
    "normalize_js_runtimes(None)",
    # Dict branch: names are stripped+lowercased, non-dict configs become {},
    # blank names are skipped, and dict configs pass through unchanged.
    "normalize_js_runtimes({'Bun': {'path': '/x'}, ' Deno ': 'notadict', '': {}})",
    "normalize_js_runtimes({'bun': {'path': '/x'}})",
    "normalize_js_runtimes({})",
]


def _fallback_eval(expr: str) -> str:
    """Evaluate *expr* in a child interpreter with the native extension
    blocked, forcing the pure-Python fallbacks."""
    import subprocess
    import sys

    prog = (
        "import sys; "
        "sys.modules['ytm_player.utils._ytm_player_native'] = None; "
        "from ytm_player.services.yt_dlp_options import *; "
        f"print(repr({expr}))"
    )
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True)
    return out.stdout.strip()


class TestNativeParity:
    """When the Rust extension is built, the ported normalizers must produce
    byte-for-byte the same output as the pure-Python fallback.

    The fallback runs in a child interpreter with the extension import
    blocked, so both implementations are compared directly instead of
    trusting whichever one the test process happened to load.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize("expr", _PARITY_CASES)
    def test_native_matches_fallback(self, expr):
        import ytm_player.services.yt_dlp_options as opts

        fallback = _fallback_eval(expr)
        native = repr(eval(expr, vars(opts)))
        assert native == fallback

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_config_subdict_passes_through_by_identity(self):
        """The dict branch keeps the caller's config sub-dicts verbatim."""
        cfg = {"path": "/x"}
        out = _ytm_player_native.normalize_js_runtimes({"bun": cfg})
        assert out is not None
        assert out["bun"] is cfg

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_non_iterable_rejected_like_fallback(self):
        """A non-iterable, non-str value raises TypeError on both backends
        (the fallback raises from its list comprehension, the native path
        from Python's own iteration protocol)."""
        with pytest.raises(TypeError):
            normalize_remote_components(5)
        with pytest.raises(TypeError):
            _ytm_player_native.normalize_remote_components(5)
