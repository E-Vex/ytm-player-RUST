"""Tests for `ytm config` editor invocation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from ytm_player.cli import _config_editor_plan, main

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Redirect CONFIG_DIR / CONFIG_FILE to a tmp dir so the test never
    writes to the developer's real config."""
    config_dir = tmp_path / "config"
    config_file = config_dir / "config.toml"
    monkeypatch.setattr("ytm_player.cli.CONFIG_DIR", config_dir)
    monkeypatch.setattr("ytm_player.cli.CONFIG_FILE", config_file)
    return config_dir, config_file


def _patch_subprocess(monkeypatch) -> dict[str, Any]:
    """Intercept subprocess.run so the test never actually launches an editor."""
    captured: dict[str, Any] = {}

    def fake_run(args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr("ytm_player.cli.subprocess.run", fake_run)
    return captured


def _counting_thunk(value: Any) -> tuple[list[int], Any]:
    """A resolve_xdg thunk that records every call."""
    calls: list[int] = []
    return calls, (lambda: (calls.append(1), value)[1])


class TestConfigEditorPlan:
    """The editor resolution behind ``ytm config`` — on whichever backend
    is loaded."""

    def test_editor_env_opens_the_config_file(self):
        calls, thunk = _counting_thunk("/usr/bin/xdg-open")
        plan = _config_editor_plan("vim", False, thunk, Path("/c/config.toml"), Path("/c/dir"))
        assert plan == {"kind": "editor", "editor": "vim", "target": "/c/config.toml"}
        # The PATH probe never runs when an EDITOR is set — the reference
        # only resolves xdg-open on the fallback branch.
        assert calls == []

    def test_editor_with_args_is_passed_through_unsplit(self):
        # The engine passes the EDITOR value through as it is — the
        # shlex tokenisation is the caller's, at the subprocess call.
        plan = _config_editor_plan("code -w", False, lambda: None, "/c/f", "/c/d")
        assert plan == {"kind": "editor", "editor": "code -w", "target": "/c/f"}

    def test_empty_editor_env_falls_through(self):
        # An empty EDITOR is falsy — the reference's `if editor:` — so
        # the xdg branch runs.
        calls, thunk = _counting_thunk("/usr/bin/xdg-open")
        plan = _config_editor_plan("", False, thunk, Path("/c/config.toml"), Path("/c/dir"))
        assert plan == {"kind": "editor", "editor": "/usr/bin/xdg-open", "target": "/c/dir"}
        assert len(calls) == 1  # exactly one probe

    def test_win32_shell_opens_the_config_file(self):
        calls, thunk = _counting_thunk("/usr/bin/xdg-open")
        plan = _config_editor_plan(None, True, thunk, Path("/c/config.toml"), Path("/c/dir"))
        assert plan == {"kind": "startfile", "target": "/c/config.toml"}
        assert calls == []

    def test_empty_editor_env_with_win32_still_shell_opens(self):
        plan = _config_editor_plan("", True, lambda: None, "/c/config.toml", "/c/dir")
        assert plan == {"kind": "startfile", "target": "/c/config.toml"}

    def test_xdg_open_opens_the_config_directory(self):
        calls, thunk = _counting_thunk("/usr/bin/xdg-open")
        plan = _config_editor_plan(None, False, thunk, Path("/c/config.toml"), Path("/c/dir"))
        assert plan == {"kind": "editor", "editor": "/usr/bin/xdg-open", "target": "/c/dir"}
        assert len(calls) == 1

    def test_no_editor_no_xdg_is_the_manual_open_error(self):
        calls, thunk = _counting_thunk(None)
        plan = _config_editor_plan(None, False, thunk, Path("/c/config.toml"), Path("/c/my dir"))
        assert plan == {
            "kind": "error",
            "message": "No $EDITOR set and xdg-open not found. Open manually: /c/my dir",
        }
        assert len(calls) == 1

    def test_empty_string_xdg_result_is_still_the_editor(self):
        # The which verdict is `is None`, not a truthiness gate — an
        # empty-string path (an absurd but possible stub result) is the
        # editor, not the error.
        plan = _config_editor_plan(None, False, lambda: "", "/c/f", "/c/d")
        assert plan == {"kind": "editor", "editor": "", "target": "/c/d"}

    def test_editor_value_passes_through_as_the_object_it_is(self):
        class EditorStr(str):
            pass

        editor = EditorStr("vim")
        plan = _config_editor_plan(editor, False, lambda: None, "/c/f", "/c/d")
        assert plan["editor"] is editor

    def test_fresh_map_every_call(self):
        args = ("vim", False, lambda: None, "/c/f", "/c/d")
        first = _config_editor_plan(*args)
        second = _config_editor_plan(*args)
        assert first is not second
        assert first == second


class TestConfigCommandFallbacks:
    """The resolution glue beyond $EDITOR, through the CLI runner."""

    def test_xdg_open_opens_config_dir_when_no_editor(self, isolated_config, monkeypatch):
        config_dir, _ = isolated_config
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setattr(
            shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None
        )
        captured = _patch_subprocess(monkeypatch)

        result = CliRunner().invoke(main, ["config"])

        assert result.exit_code == 0, result.output
        assert captured["args"] == ["/usr/bin/xdg-open", str(config_dir)]

    def test_no_editor_no_xdg_exits_with_manual_open_error(self, isolated_config, monkeypatch):
        config_dir, _ = isolated_config
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        captured = _patch_subprocess(monkeypatch)

        result = CliRunner().invoke(main, ["config"])

        assert result.exit_code == 1
        assert f"No $EDITOR set and xdg-open not found. Open manually: {config_dir}" in (
            result.output
        )
        assert "args" not in captured  # nothing was ever launched

    def test_win32_shell_opens_the_config_file(self, isolated_config, monkeypatch):
        _, config_file = isolated_config
        monkeypatch.delenv("EDITOR", raising=False)
        monkeypatch.setattr(sys, "platform", "win32")
        startfile: dict[str, Any] = {}

        def fake_startfile(path):
            startfile["path"] = path

        monkeypatch.setattr(os, "startfile", fake_startfile, raising=False)
        captured = _patch_subprocess(monkeypatch)

        result = CliRunner().invoke(main, ["config"])

        assert result.exit_code == 0, result.output
        assert startfile["path"] == str(config_file)
        assert "args" not in captured  # os.startfile replaced subprocess entirely


# ── Native vs fallback parity ──────────────────────────────────────────

# (editor_env, is_win32, xdg, config_file, config_dir) tuples, shared
# with the child interpreter as JSON — both sides wrap the xdg value in
# a thunk, and the thunk's call count comes back with the plan so the
# probe-timing contract (never on the EDITOR branch, never on win32,
# exactly once on the xdg branch) is compared too.
_EDITOR_PARITY_CASES = [
    ("vim", False, None, "/c/config.toml", "/c/dir"),
    ("code -w", False, "/usr/bin/xdg-open", "/c/config.toml", "/c/dir"),
    ("'/opt/My Editor/bin/ed' --no-fork", False, None, "/c/config.toml", "/c/dir"),
    ("code 'unclosed", False, None, "/c/config.toml", "/c/dir"),
    ("", False, "/usr/bin/xdg-open", "/c/config.toml", "/c/dir"),
    ("", False, None, "/c/config.toml", "/c/dir"),
    ("  ", False, None, "/c/config.toml", "/c/dir"),  # whitespace EDITOR is truthy
    (None, True, None, "/c/config.toml", "/c/dir"),  # win32, probe never runs
    (None, True, "/usr/bin/xdg-open", "/c/config.toml", "/c/dir"),
    ("", True, None, "/c/config.toml", "/c/dir"),  # empty EDITOR + win32
    (None, False, "/usr/bin/xdg-open", "/c/config.toml", "/c/dir"),
    (None, False, None, "/c/config.toml", "/c/dir"),  # the manual-open error
    (None, False, "", "/c/config.toml", "/c/dir"),  # empty-string xdg is the editor
    (None, 1, None, "/c/config.toml", "/c/dir"),  # a truthy non-bool win32 flag
    (None, 0, "/usr/bin/xdg-open", "/c/config.toml", "/c/dir"),  # falsy win32 flag
    ("éditeur", False, None, "/ünï/config.toml", "/ünï/dir"),
    (None, False, None, "/c/f.toml", "/c/dir with spaces"),
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _config_editor_plan


def outcome(editor_env, is_win32, xdg, config_file, config_dir):
    calls = []
    def resolve_xdg():
        calls.append(1)
        return xdg
    try:
        plan = _config_editor_plan(editor_env, is_win32, resolve_xdg, config_file, config_dir)
        return {"ok": plan, "calls": len(calls)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


cases = json.load(sys.stdin)
json.dump([outcome(*case) for case in cases], sys.stdout)
"""


class TestNativeParity:
    """When the Rust extension is built, the native engine must decide
    exactly like the pure-Python reference — same plans, same error
    shapes, same probe timing — for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_config_editor_plan_matches_fallback(self):
        def outcome(case):
            editor_env, is_win32, xdg, config_file, config_dir = case
            calls = []

            def resolve_xdg():
                calls.append(1)
                return xdg

            try:
                plan = _config_editor_plan(
                    editor_env, is_win32, resolve_xdg, config_file, config_dir
                )
                return {"ok": plan, "calls": len(calls)}
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                return {"error": f"{type(exc).__name__}: {exc}"}

        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_EDITOR_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [outcome(case) for case in _EDITOR_PARITY_CASES]
        for case, n, f in zip(_EDITOR_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback editor plans differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # The probe-timing contract, pinned by the compared call counts:
        # never on the EDITOR branch, never on win32, exactly once on xdg.
        for case, n in zip(_EDITOR_PARITY_CASES, native):
            editor_env, is_win32, xdg, _, _ = case
            if editor_env or is_win32:
                assert n["calls"] == 0, f"probe ran on a non-xdg branch: {case!r}"
            else:
                assert n["calls"] == 1, f"probe did not run exactly once: {case!r}"


def test_editor_with_args_is_tokenised(isolated_config, monkeypatch):
    """Regression: EDITOR="code -w" must split into ["code", "-w"], not be
    passed as a single argv entry — otherwise subprocess.run would look
    for an executable literally named "code -w" and fail."""
    _, config_file = isolated_config
    monkeypatch.setenv("EDITOR", "code -w")
    captured = _patch_subprocess(monkeypatch)

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0, result.output
    assert captured["args"] == ["code", "-w", str(config_file)]


def test_single_word_editor_unchanged(isolated_config, monkeypatch):
    """EDITOR="vim" stays a single argv entry, no spurious tokenisation."""
    _, config_file = isolated_config
    monkeypatch.setenv("EDITOR", "vim")
    captured = _patch_subprocess(monkeypatch)

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0, result.output
    assert captured["args"] == ["vim", str(config_file)]


def test_editor_with_quoted_path(isolated_config, monkeypatch):
    """shlex handles quoted segments — EDITOR='/path/with spaces/ed -x' is
    parsed as ["/path/with spaces/ed", "-x"], not split on the inner space."""
    _, config_file = isolated_config
    monkeypatch.setenv("EDITOR", "'/opt/My Editor/bin/ed' --no-fork")
    captured = _patch_subprocess(monkeypatch)

    result = CliRunner().invoke(main, ["config"])

    assert result.exit_code == 0, result.output
    assert captured["args"] == [
        "/opt/My Editor/bin/ed",
        "--no-fork",
        str(config_file),
    ]


def test_malformed_editor_quoting_does_not_crash(isolated_config, monkeypatch):
    """shlex.split raises ValueError on unbalanced quotes — must be caught
    and routed through _error so the CLI exits cleanly instead of crashing."""
    _, _ = isolated_config
    # Unmatched single quote — shlex.split raises ValueError("No closing quotation").
    monkeypatch.setenv("EDITOR", "code 'unclosed")

    # subprocess.run should never be reached; patch it so a regression that
    # bypassed the ValueError catch would still not actually launch anything.
    captured = _patch_subprocess(monkeypatch)

    result = CliRunner().invoke(main, ["config"])

    # _error calls sys.exit(1).
    assert result.exit_code != 0
    assert "Failed to open editor" in result.output
    assert "args" not in captured
