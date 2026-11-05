"""Tests for Settings TOML serialization, atomic writes, and load behavior."""

from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    # Python 3.10 backport via PyPI
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

from ytm_player.config.settings import Settings


class TestDefaultValues:
    def test_defaults(self):
        s = Settings()
        assert s.general.startup_page == "library"
        assert s.playback.audio_quality == "high"
        assert s.playback.default_volume == 80
        assert s.playback.autoplay is True
        assert s.search.default_mode == "music"
        assert s.yt_dlp.use_session_cookies is False
        assert s.cache.enabled is True
        assert s.cache.max_size_mb == 1024
        assert s.ui.theme == "ytm-dark"
        assert s.ui.album_art is True
        assert s.notifications.enabled is True
        assert s.mpris.enabled is True


def test_ui_settings_theme_default():
    """theme defaults to the bundled ytm-dark Textual theme."""
    from ytm_player.config.settings import UISettings

    ui = UISettings()
    assert ui.theme == "ytm-dark"


def test_settings_load_respects_ui_theme(tmp_config_dir):
    """theme in [ui] is honoured as the declarative default."""
    config_file = tmp_config_dir / "config.toml"
    config_file.write_text('[ui]\ntheme = "catppuccin-mocha"\n', encoding="utf-8")

    s = Settings.load(config_file)

    assert s.ui.theme == "catppuccin-mocha"


def test_ui_settings_show_selection_info_default():
    """show_selection_info defaults to True."""
    from ytm_player.config.settings import UISettings

    ui = UISettings()
    assert ui.show_selection_info is True


def test_settings_load_respects_show_selection_info_false(tmp_config_dir):
    """show_selection_info = false in config.toml is honoured."""
    config_file = tmp_config_dir / "config.toml"
    config_file.write_text("[ui]\nshow_selection_info = false\n", encoding="utf-8")
    s = Settings.load(config_file)
    assert s.ui.show_selection_info is False


def test_ui_settings_show_queue_source_default():
    """show_queue_source defaults to True."""
    from ytm_player.config.settings import UISettings

    ui = UISettings()
    assert ui.show_queue_source is True


def test_settings_load_respects_show_queue_source_false(tmp_config_dir):
    """show_queue_source = false in config.toml is honoured."""
    config_file = tmp_config_dir / "config.toml"
    config_file.write_text("[ui]\nshow_queue_source = false\n", encoding="utf-8")
    s = Settings.load(config_file)
    assert s.ui.show_queue_source is False


class TestSaveLoadRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            "{title}\n{artist}",
            "\r\n\t\b\f",
            "".join(chr(code) for code in range(32)) + "\x7f",
            '"quoted" \\ path \\n literal',
            "Björk 日本語 🎵 \u0085\u2028\u2029",
            '"\n[ui]\ntheme = "unexpected"\n#',
            "",
        ],
    )
    def test_string_values_round_trip_without_reset(self, tmp_path, value):
        path = tmp_path / "config.toml"
        original = Settings()
        original.notifications.format = value
        original.ui.theme = "nord"
        original.playback.default_volume = 37
        original.save(path)

        # Parse directly too: Settings.load recovers invalid TOML with defaults.
        with path.open("rb") as source:
            assert tomllib.load(source)["notifications"]["format"] == value
        loaded = Settings.load(path)
        assert loaded.notifications.format == value
        assert loaded.ui.theme == "nord"
        assert loaded.playback.default_volume == 37
        assert not path.with_suffix(".toml.bak").exists()

    def test_list_strings_use_the_same_escaping(self):
        from ytm_player.config.settings import _format_toml_value

        values = ["a\nb", "\x00\x7f", '"quoted"', "🎵", "C:\\path"]
        assert tomllib.loads(f"values = {_format_toml_value(values)}")["values"] == values

    def test_round_trip(self, tmp_config_dir):
        path = tmp_config_dir / "config.toml"

        original = Settings()
        original.playback.default_volume = 50
        original.general.startup_page = "search"
        original.cache.max_size_mb = 2048
        original.ui.theme = "textual-dark"
        original.save(path)

        loaded = Settings.load(path)
        assert loaded.playback.default_volume == 50
        assert loaded.general.startup_page == "search"
        assert loaded.cache.max_size_mb == 2048
        assert loaded.ui.theme == "textual-dark"
        # Other defaults preserved.
        assert loaded.playback.audio_quality == "high"
        assert loaded.ui.album_art is True

    def test_save_creates_file(self, tmp_config_dir):
        path = tmp_config_dir / "new_config.toml"
        assert not path.exists()
        Settings().save(path)
        assert path.exists()


class TestPartialToml:
    def test_partial_preserves_defaults(self, tmp_config_dir):
        path = tmp_config_dir / "config.toml"
        path.write_text("[playback]\ndefault_volume = 42\n")

        loaded = Settings.load(path)
        assert loaded.playback.default_volume == 42
        # All other fields should be defaults.
        assert loaded.playback.audio_quality == "high"
        assert loaded.general.startup_page == "library"

    def test_missing_file_creates_default(self, tmp_config_dir):
        path = tmp_config_dir / "nonexistent.toml"
        loaded = Settings.load(path)
        assert loaded.playback.default_volume == 80
        assert path.exists()  # Default file should be created.


class TestAtomicWrites:
    """Settings.save must use atomic writes — no partial files left
    on the user's disk if the process is killed mid-write."""

    def test_save_does_not_leave_temp_file_on_success(self, tmp_path):
        config_path = tmp_path / "config.toml"
        s = Settings()
        s.save(config_path)

        # No leftover .tmp file.
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == [], f"Atomic write left temp files: {leftover}"

        # The actual config file should exist and be valid TOML.
        assert config_path.exists()
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert "general" in data

    def test_save_uses_replace_not_direct_write(self, tmp_path, monkeypatch):
        """The implementation should call os.replace (atomic) at some
        point during save, not just path.write_text."""
        import os

        config_path = tmp_path / "config.toml"
        replace_calls: list = []
        original_replace = os.replace

        def tracking_replace(src, dst):
            replace_calls.append((str(src), str(dst)))
            original_replace(src, dst)

        monkeypatch.setattr("os.replace", tracking_replace)

        s = Settings()
        s.save(config_path)

        # At least one os.replace into the final config_path.
        targets = [dst for _src, dst in replace_calls]
        assert str(config_path) in targets, (
            f"expected os.replace into {config_path}, got replace calls: {replace_calls}"
        )

    def test_save_falls_back_to_direct_write_when_replace_is_denied(self, tmp_path, monkeypatch):
        """Windows may deny os.replace when config.toml is held by another process."""
        import os

        config_path = tmp_path / "config.toml"
        config_path.write_text("stale", encoding="utf-8")

        def denied_replace(_src, _dst):
            raise PermissionError("access denied")

        monkeypatch.setattr(os, "replace", denied_replace)

        s = Settings()
        s.ui.theme = "textual-dark"
        s.save(config_path)

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        assert data["ui"]["theme"] == "textual-dark"
        assert not config_path.with_suffix(config_path.suffix + ".tmp").exists()


# ── the TOML value renderer: contracts + native/fallback parity ────────

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None

# The value zoo: every match arm and its edges — bools (before the int
# arm they'd otherwise fall into), ints (including the widths no fixed
# integer holds), strings (the control-character set, DEL, quotes and
# backslashes, the non-ASCII range, the JSON/TOML-shared escape forms),
# lists (empty, nested, mixed), and the repr catch-all (floats, None,
# dicts — the shapes a programmatically-built field could carry).
_VALUE_ZOO = [
    True,
    False,
    0,
    1,
    -5,
    80,
    10**30,
    -(10**30),
    "",
    "plain",
    'say "hi" \\ done',
    "\r\n\t\b\f",
    "".join(chr(code) for code in range(32)) + "\x7f",
    "DEL \x7f twice \x7f",
    "Björk 日本語 🎵 \u0085\u2028\u2029",
    '"\n[ui]\ntheme = "unexpected"\n#',
    "C:\\path\\to\\file",
    [],
    ["a\nb", "\x00\x7f", '"quoted"', "🎵", "C:\\path"],
    ["outer", ["nested", 2, True, ["deeper"]]],
    [True, False, 0, -1],
    1.5,
    0.1,
    None,
    {"a": 1},
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.config.settings import _format_toml_value


def revive(v):
    # JSON round-trips collapse the zoo's None marker; rebuild the exact
    # shapes (None -> the {"$none": True} marker -> None).
    if isinstance(v, dict) and "$none" in v:
        return None
    if isinstance(v, list):
        return [revive(x) for x in v]
    return v


values = json.load(sys.stdin)
json.dump([_format_toml_value(revive(v)) for v in values], sys.stdout)
"""


class TestFormatTomlValueEngine:
    """The value-to-TOML-literal renderer, on whichever backend is
    loaded, plus the child-interpreter parity matrix when the extension
    is built."""

    def test_bool_renders_before_the_int_arm(self):
        from ytm_player.config.settings import _format_toml_value

        assert _format_toml_value(True) == "true"
        assert _format_toml_value(False) == "false"

    def test_int_renders_through_str(self):
        from ytm_player.config.settings import _format_toml_value

        assert _format_toml_value(80) == "80"
        assert _format_toml_value(-5) == "-5"
        assert _format_toml_value(10**30) == str(10**30)

    def test_string_renders_json_escaped_with_del(self):
        import json as json_mod

        from ytm_player.config.settings import _format_toml_value

        for s in ["", "plain", 'q "\\\\', "\n\t\r\x0b\x0c", "\x7f", "ü🎵"]:
            expected = json_mod.dumps(s, ensure_ascii=False).replace("\x7f", "\\u007f")
            assert _format_toml_value(s) == expected

    def test_list_renders_bracketed_join(self):
        from ytm_player.config.settings import _format_toml_value

        assert _format_toml_value([]) == "[]"
        assert _format_toml_value(["a", "b"]) == '["a", "b"]'
        assert _format_toml_value([["x", 1], True]) == '[["x", 1], true]'

    def test_catch_all_renders_repr(self):
        from ytm_player.config.settings import _format_toml_value

        assert _format_toml_value(None) == "None"
        assert _format_toml_value(1.5) == "1.5"
        assert _format_toml_value({"a": 1}) == repr({"a": 1})

    def test_every_rendered_list_stays_parseable_toml(self):
        # The renderer's whole job: whatever it emits for the list-y zoo
        # members re-parses to the same value.
        from ytm_player.config.settings import _format_toml_value

        for value in _VALUE_ZOO:
            if isinstance(value, list):
                rendered = _format_toml_value(value)
                assert tomllib.loads(f"values = {rendered}")["values"] == value

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_renderer_matches_fallback(self):
        import json as json_mod
        import subprocess

        from ytm_player.config.settings import _format_toml_value

        payload = json_mod.dumps([{"$none": True} if v is None else v for v in _VALUE_ZOO])
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json_mod.loads(result.stdout)
        native = [_format_toml_value(v) for v in _VALUE_ZOO]
        for value, nv, fb in zip(_VALUE_ZOO, native, fallback):
            assert nv == fb, f"renders differ for {value!r}:\nnative:   {nv!r}\nfallback: {fb!r}"

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_the_native_engine_really_answers(self):
        """The binding is live: the module's renderer is the native
        function itself when the extension is built."""
        from ytm_player.config.settings import _format_toml_value
        from ytm_player.utils._ytm_player_native import format_toml_value

        assert _format_toml_value is format_toml_value

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_settings_save_runs_the_native_renderer(self):
        """Through the public save(): the emitted file is byte-identical
        to the fallback's (child-interpreter) render of the same
        settings."""
        import subprocess
        import tempfile
        from pathlib import Path

        original = Settings()
        original.notifications.format = "Dêl \x7f 'fmt' \n"
        original.ui.theme = "ünïcode 🎵"
        original.yt_dlp.remote_components = ["a b", '"q"', "C:\\p"]
        original.playback.default_volume = 37

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            original.save(path)
            native_text = path.read_text(encoding="utf-8")

        child = (
            "import sys, tempfile, json\n"
            "from pathlib import Path\n"
            "sys.modules['ytm_player.utils._ytm_player_native'] = None\n"
            "from ytm_player.config.settings import Settings\n"
            "s = Settings()\n"
            "s.notifications.format = 'Dêl \\x7f \\'fmt\\' \\n'\n"
            "s.ui.theme = 'ünïcode 🎵'\n"
            "s.yt_dlp.remote_components = ['a b', '\"q\"', 'C:\\\\p']\n"
            "s.playback.default_volume = 37\n"
            "with tempfile.TemporaryDirectory() as tmp:\n"
            "    p = Path(tmp) / 'config.toml'\n"
            "    s.save(p)\n"
            "    sys.stdout.write(p.read_text(encoding='utf-8'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", child],
            capture_output=True,
            text=True,
            check=True,
        )
        assert native_text == result.stdout
        # And the round trip still holds.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.toml"
            original.save(path)
            loaded = Settings.load(path)
        assert loaded.notifications.format == original.notifications.format
        assert loaded.ui.theme == original.ui.theme
        assert loaded.yt_dlp.remote_components == original.yt_dlp.remote_components
