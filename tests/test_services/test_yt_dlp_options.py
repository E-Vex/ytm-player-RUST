import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from ytm_player.config.settings import YtDlpSettings
from ytm_player.services.yt_dlp_options import (
    _configured_ydl_opts,
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


# ── The configured-extras layering engine ───────────────────────


class TestConfiguredOptsEngine:
    """The extras-layering core on whichever backend is loaded."""

    @staticmethod
    def _thunks(file_verdict=True):
        log = []

        def is_file():
            log.append("is_file")
            return file_verdict

        def warn(message, *args):
            log.append(("warn", message, args))

        return is_file, warn, log

    def test_no_extras_leaves_opts_untouched(self):
        is_file, warn, log = self._thunks()
        opts = {"quiet": True}
        result = _configured_ydl_opts(opts, None, None, is_file, warn, None, None)
        assert result == {"quiet": True}
        assert result is opts
        # Neither thunk is ever invoked without a configured extra.
        assert log == []

    def test_cookies_file_written_as_the_object_it_is(self):
        is_file, warn, log = self._thunks()
        result = _configured_ydl_opts({}, "/tmp/cookies.txt", None, is_file, warn, None, None)
        assert result["cookiefile"] == "/tmp/cookies.txt"
        assert log == []

    @pytest.mark.parametrize("falsy", [None, ""])
    def test_falsy_cookies_file_not_written(self, falsy):
        is_file, warn, log = self._thunks()
        result = _configured_ydl_opts({}, falsy, None, is_file, warn, None, None)
        assert "cookiefile" not in result
        assert log == []

    def test_ca_bundle_existing_file_sets_compat_without_warning(self):
        is_file, warn, log = self._thunks(file_verdict=True)
        result = _configured_ydl_opts({}, None, "/ca.pem", is_file, warn, None, None)
        assert result["compat_opts"] == {"no-certifi"}
        assert log == ["is_file"]

    def test_ca_bundle_missing_file_warns_then_sets_compat(self):
        is_file, warn, log = self._thunks(file_verdict=False)
        result = _configured_ydl_opts({}, None, "/ca.pem", is_file, warn, None, None)
        assert result["compat_opts"] == {"no-certifi"}
        assert log == [
            "is_file",
            ("warn", "Configured [yt_dlp] ca_bundle does not exist: %s", ("/ca.pem",)),
        ]

    def test_falsy_ca_bundle_never_probes(self):
        is_file, warn, log = self._thunks(file_verdict=False)
        result = _configured_ydl_opts({}, None, "", is_file, warn, None, None)
        assert "compat_opts" not in result
        assert log == []

    def test_existing_compat_list_copied_not_mutated(self):
        is_file, warn, _ = self._thunks()
        stored = ["legacy"]
        opts = {"compat_opts": stored}
        result = _configured_ydl_opts(opts, None, "/ca.pem", is_file, warn, None, None)
        assert result["compat_opts"] == {"legacy", "no-certifi"}
        # The stored list object is never mutated — the copy list() made.
        assert stored == ["legacy"]

    def test_compat_already_present_not_duplicated_but_still_converted(self):
        is_file, warn, _ = self._thunks()
        result = _configured_ydl_opts(
            {"compat_opts": ["no-certifi", "legacy"]}, None, "/ca.pem", is_file, warn, None, None
        )
        assert result["compat_opts"] == {"no-certifi", "legacy"}

    @pytest.mark.parametrize("shape", [("legacy",), {"legacy"}, ["legacy"]])
    def test_compat_iterable_shapes_all_accepted(self, shape):
        is_file, warn, _ = self._thunks()
        result = _configured_ydl_opts(
            {"compat_opts": shape}, None, "/ca.pem", is_file, warn, None, None
        )
        assert result["compat_opts"] == {"legacy", "no-certifi"}

    def test_remote_components_warned_and_stored_as_the_same_object(self):
        is_file, warn, log = self._thunks()
        remote = ["ejs:github", "ejs:npm"]
        result = _configured_ydl_opts({}, None, None, is_file, warn, remote, None)
        assert result["remote_components"] is remote
        assert log == [
            (
                "warn",
                "yt-dlp remote_components is enabled; this allows remote JavaScript "
                "component downloads: %s",
                ("ejs:github, ejs:npm",),
            )
        ]

    def test_empty_remote_components_not_stored(self):
        is_file, warn, log = self._thunks()
        result = _configured_ydl_opts({}, None, None, is_file, warn, [], None)
        assert "remote_components" not in result
        assert log == []

    def test_js_runtimes_stored_as_the_same_object(self):
        is_file, warn, log = self._thunks()
        js = {"bun": {"path": "/x"}}
        result = _configured_ydl_opts({}, None, None, is_file, warn, None, js)
        assert result["js_runtimes"] is js
        assert log == []

    def test_empty_js_runtimes_not_stored(self):
        is_file, warn, _ = self._thunks()
        result = _configured_ydl_opts({}, None, None, is_file, warn, None, {})
        assert "js_runtimes" not in result

    def test_warnings_fire_in_the_reference_order(self):
        is_file, warn, log = self._thunks(file_verdict=False)
        _configured_ydl_opts({}, None, "/ca.pem", is_file, warn, ["ejs:github"], None)
        assert [entry if isinstance(entry, str) else entry[1] for entry in log] == [
            "is_file",
            "Configured [yt_dlp] ca_bundle does not exist: %s",
            "yt-dlp remote_components is enabled; this allows remote JavaScript "
            "component downloads: %s",
        ]

    def test_unhashable_compat_member_raises_after_the_warning(self):
        # The set() conversion's own TypeError — and the thunk timing
        # contract: the missing-file warning has already fired by then.
        is_file, warn, log = self._thunks(file_verdict=False)
        with pytest.raises(TypeError, match="unhashable type"):
            _configured_ydl_opts({"compat_opts": [[]]}, None, "/ca.pem", is_file, warn, None, None)
        assert log == [
            "is_file",
            ("warn", "Configured [yt_dlp] ca_bundle does not exist: %s", ("/ca.pem",)),
        ]

    def test_raising_file_probe_propagates_before_any_mutation(self):
        def is_file():
            raise RuntimeError("probe boom")

        warn = lambda *a: pytest.fail("warn must not run")  # noqa: E731
        opts = {}
        with pytest.raises(RuntimeError, match="probe boom"):
            _configured_ydl_opts(opts, None, "/ca.pem", is_file, warn, None, None)
        assert "compat_opts" not in opts

    def test_raising_warn_propagates_before_the_compat_read(self):
        def is_file():
            return False

        def warn(message, *args):
            raise OSError("log sink gone")

        opts = {}
        with pytest.raises(OSError, match="log sink gone"):
            _configured_ydl_opts(opts, None, "/ca.pem", is_file, warn, None, None)
        assert "compat_opts" not in opts

    def test_opts_get_steers_the_compat_read(self):
        is_file, warn, _ = self._thunks()

        class MyDict(dict):
            def get(self, key, default=None):
                if key == "compat_opts":
                    return ["steered"]
                return super().get(key, default)

        result = _configured_ydl_opts(MyDict(), None, "/ca.pem", is_file, warn, None, None)
        assert result["compat_opts"] == {"steered", "no-certifi"}

    def test_opts_setitem_protocol_used_for_the_writes(self):
        is_file, warn, _ = self._thunks()

        class Recording(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.writes = []

            def __setitem__(self, key, value):
                self.writes.append(key)
                super().__setitem__(key, value)

        opts = Recording()
        result = _configured_ydl_opts(opts, "/c.txt", "/ca.pem", is_file, warn, ["r"], {"bun": {}})
        assert opts.writes == ["cookiefile", "compat_opts", "remote_components", "js_runtimes"]
        assert result is opts

    def test_non_mapping_opts_untouched_without_extras(self):
        is_file, warn, log = self._thunks()
        result = _configured_ydl_opts(5, None, None, is_file, warn, None, None)
        assert result == 5
        assert log == []

    def test_non_mapping_opts_with_cookies_raises_type_error(self):
        is_file, warn, _ = self._thunks()
        with pytest.raises(TypeError, match="does not support item assignment"):
            _configured_ydl_opts(5, "/c.txt", None, is_file, warn, None, None)

    def test_non_str_remote_member_raises_from_the_join(self):
        is_file, warn, _ = self._thunks()
        opts = {}
        with pytest.raises(TypeError, match="expected str instance"):
            _configured_ydl_opts(opts, None, None, is_file, warn, [1, 2], None)
        assert "remote_components" not in opts


class TestApplyConfiguredWrapper:
    """The public wrapper's thunk plumbing on whichever backend is loaded."""

    def test_existing_ca_bundle_does_not_warn(self, tmp_path, caplog):
        bundle = tmp_path / "corporate.pem"
        bundle.write_text("FAKE BUNDLE")
        settings = YtDlpSettings(ca_bundle=str(bundle))
        with caplog.at_level(logging.WARNING):
            opts = apply_configured_yt_dlp_options({"quiet": True}, settings)
        assert opts["compat_opts"] == {"no-certifi"}
        assert "ca_bundle does not exist" not in caplog.text

    def test_missing_ca_bundle_warns_once(self, caplog):
        settings = YtDlpSettings(ca_bundle="/nonexistent/ca.pem")
        with caplog.at_level(logging.WARNING):
            opts = apply_configured_yt_dlp_options({"quiet": True}, settings)
        assert opts["compat_opts"] == {"no-certifi"}
        assert caplog.text.count("ca_bundle does not exist: /nonexistent/ca.pem") == 1


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


# ── the extras-layering engine's native vs fallback parity ──────────
#
# The engine takes thunks, so the parity matrix travels over a JSON
# bridge (the cache-engine pattern): each case names the opts payload,
# the four setting values, the file probe's verdict (or "raise") and
# whether the warn thunk raises; the runner — one copy here for the
# native engine, one embedded in the child for the pure-Python
# transcription — records the result's repr, the same-object verdicts,
# every thunk call in order and any error's shape.

_LAYERING_CASES = [
    # Baseline: nothing configured — opts untouched, no thunk called.
    {
        "opts": {"quiet": True},
        "cookies": None,
        "ca": None,
        "file": False,
        "remote": None,
        "js": None,
    },
    # cookies_file: set, empty (falsy → never written).
    {
        "opts": {},
        "cookies": "/tmp/cookies.txt",
        "ca": None,
        "file": False,
        "remote": None,
        "js": None,
    },
    {"opts": {}, "cookies": "", "ca": None, "file": False, "remote": None, "js": None},
    # CA bundle: existing file (no warn), missing file (warn), falsy
    # (never probed — the empty call log is the pin).
    {"opts": {}, "cookies": None, "ca": "/ca.pem", "file": True, "remote": None, "js": None},
    {"opts": {}, "cookies": None, "ca": "/ca.pem", "file": False, "remote": None, "js": None},
    {"opts": {}, "cookies": None, "ca": "", "file": False, "remote": None, "js": None},
    # compat_opts shapes: empty, populated, already carrying no-certifi,
    # a string (list() explodes it to characters), None (list()'s own
    # TypeError) and an unhashable member (set()'s own TypeError) —
    # the last two AFTER the missing-file warning, the thunk-timing pin.
    {
        "opts": {"compat_opts": []},
        "cookies": None,
        "ca": "/ca.pem",
        "file": True,
        "remote": None,
        "js": None,
    },
    {
        "opts": {"compat_opts": ["legacy"]},
        "cookies": None,
        "ca": "/ca.pem",
        "file": True,
        "remote": None,
        "js": None,
    },
    {
        "opts": {"compat_opts": ["no-certifi"]},
        "cookies": None,
        "ca": "/ca.pem",
        "file": False,
        "remote": None,
        "js": None,
    },
    {
        "opts": {"compat_opts": "abc"},
        "cookies": None,
        "ca": "/ca.pem",
        "file": True,
        "remote": None,
        "js": None,
    },
    {
        "opts": {"compat_opts": None},
        "cookies": None,
        "ca": "/ca.pem",
        "file": False,
        "remote": None,
        "js": None,
    },
    {
        "opts": {"compat_opts": [[]]},
        "cookies": None,
        "ca": "/ca.pem",
        "file": False,
        "remote": None,
        "js": None,
    },
    # A raising file probe propagates before any mutation.
    {"opts": {}, "cookies": None, "ca": "/ca.pem", "file": "raise", "remote": None, "js": None},
    # remote_components: non-empty (warn + same-object write), empty
    # (falsy → neither), and a non-str member (the join's own TypeError
    # before the write).
    {"opts": {}, "cookies": None, "ca": None, "file": False, "remote": ["ejs:github"], "js": None},
    {"opts": {}, "cookies": None, "ca": None, "file": False, "remote": ["a", "b", "c"], "js": None},
    {"opts": {}, "cookies": None, "ca": None, "file": False, "remote": [], "js": None},
    {"opts": {}, "cookies": None, "ca": None, "file": False, "remote": [1, 2], "js": None},
    # js_runtimes: non-empty (same-object write) and empty (falsy).
    {
        "opts": {},
        "cookies": None,
        "ca": None,
        "file": False,
        "remote": None,
        "js": {"bun": {"path": "/x"}},
    },
    {"opts": {}, "cookies": None, "ca": None, "file": False, "remote": None, "js": {}},
    # Everything at once — the writes in the reference's order.
    {
        "opts": {"quiet": 1, "compat_opts": ["x"]},
        "cookies": "/c",
        "ca": "/ca",
        "file": False,
        "remote": ["r"],
        "js": {"bun": {}},
    },
    # A raising warn thunk: the first warning's own error, before the
    # compat read.
    {
        "opts": {},
        "cookies": None,
        "ca": "/ca.pem",
        "file": False,
        "remote": None,
        "js": None,
        "warn": "raise",
    },
    # Non-mapping opts: untouched without extras, the subscript
    # protocol's own TypeError with one.
    {"opts": 5, "cookies": None, "ca": None, "file": False, "remote": None, "js": None},
    {"opts": 5, "cookies": "/c.txt", "ca": None, "file": False, "remote": None, "js": None},
]


def _run_layering_case(case):
    """Run one case against whichever engine this interpreter loaded,
    recording the result, the identity verdicts and the thunk call log."""
    opts = case["opts"]
    file_verdict = case["file"]
    warn_mode = case.get("warn", "record")
    calls = []

    def is_file():
        calls.append("is_file")
        if file_verdict == "raise":
            raise RuntimeError("probe boom")
        return file_verdict

    def warn(message, *args):
        calls.append(["warn", message, list(args)])
        if warn_mode == "raise":
            raise OSError("log sink gone")

    try:
        result = _configured_ydl_opts(
            opts, case["cookies"], case["ca"], is_file, warn, case["remote"], case["js"]
        )
        outcome = {
            # The canonical form keeps dict key order but sorts sets —
            # a set's repr order is the process's own hash seed, which
            # differs between this interpreter and the child.
            "result": _canon(result),
            "same_opts": result is opts,
            "calls": calls,
        }
        if isinstance(result, dict):
            if case["remote"]:
                outcome["remote_same"] = result.get("remote_components") is case["remote"]
            if case["js"]:
                outcome["js_same"] = result.get("js_runtimes") is case["js"]
    except BaseException as exc:  # noqa: BLE001 - repr'd for comparison
        outcome = {"error": f"{type(exc).__name__}: {exc}", "calls": calls}
    return outcome


def _canon(obj):
    """A hash-seed-independent canonical form: dict key order kept
    (insertion order is part of the contract), set members sorted by
    their canonical repr, scalars through repr()."""
    if isinstance(obj, dict):
        return {"d": [[_canon(k), _canon(v)] for k, v in obj.items()]}
    if isinstance(obj, (set, frozenset)):
        return {"s": sorted((_canon(v) for v in obj), key=repr)}
    if isinstance(obj, (list, tuple)):
        return {"l": [_canon(v) for v in obj]}
    return {"v": repr(obj)}


_LAYERING_CHILD_PROG = """
import json, sys
sys.modules["ytm_player.utils._ytm_player_native"] = None
from ytm_player.services.yt_dlp_options import _configured_ydl_opts

def _canon(obj):
    if isinstance(obj, dict):
        return {"d": [[_canon(k), _canon(v)] for k, v in obj.items()]}
    if isinstance(obj, (set, frozenset)):
        return {"s": sorted((_canon(v) for v in obj), key=repr)}
    if isinstance(obj, (list, tuple)):
        return {"l": [_canon(v) for v in obj]}
    return {"v": repr(obj)}

def run(case):
    opts = case["opts"]
    file_verdict = case["file"]
    warn_mode = case.get("warn", "record")
    calls = []

    def is_file():
        calls.append("is_file")
        if file_verdict == "raise":
            raise RuntimeError("probe boom")
        return file_verdict

    def warn(message, *args):
        calls.append(["warn", message, list(args)])
        if warn_mode == "raise":
            raise OSError("log sink gone")

    try:
        result = _configured_ydl_opts(
            opts, case["cookies"], case["ca"], is_file, warn, case["remote"], case["js"]
        )
        outcome = {
            "result": _canon(result),
            "same_opts": result is opts,
            "calls": calls,
        }
        if isinstance(result, dict):
            if case["remote"]:
                outcome["remote_same"] = result.get("remote_components") is case["remote"]
            if case["js"]:
                outcome["js_same"] = result.get("js_runtimes") is case["js"]
    except BaseException as exc:
        outcome = {"error": f"{type(exc).__name__}: {exc}", "calls": calls}
    return outcome

payload = json.load(sys.stdin)
json.dump([run(case) for case in payload], sys.stdout)
"""


class TestLayeringNativeParity:
    """When the Rust extension is built, the extras-layering engine must
    answer every matrix case exactly like the pure-Python transcription
    run in a child interpreter with the extension import blocked — the
    mutated opts (repr'd, key order included), the same-object verdicts,
    every thunk call in order, and the error shapes — so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_engine_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _LAYERING_CHILD_PROG],
            input=json.dumps(_LAYERING_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_run_layering_case(case) for case in _LAYERING_CASES]
        for case, n, f in zip(_LAYERING_CASES, native, fallback):
            assert n == f, f"{case}: {n!r} != {f!r}"
