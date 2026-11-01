"""Tests for utils.doctor — diagnostic gathering for `ytm doctor`."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestDiagnosticUrlRedaction:
    @pytest.mark.parametrize(
        "url",
        [
            "https://media.invalid/play?ip=192.0.2.42&sig=fake-signature&pot=fake-token",
            "HTTP://media.invalid/play?ip=2001%3Adb8%3A%3A42&signature=fake-signature",
            "tcp://[2001:db8::42]:443",
            "TcP://192.0.2.42:443",
            "https://media.invalid/a(b)?unknown=fake-secret",
        ],
    )
    @pytest.mark.parametrize("quotes", ["", "'", '"'])
    def test_removes_whole_url_preserving_error_text(self, url, quotes):
        from ytm_player.utils.doctor import _redact

        message = f"HTTP 403 opening {quotes}{url}{quotes} failed\nnext diagnostic line"
        assert _redact(message) == (
            f"HTTP 403 opening {quotes}[URL REDACTED]{quotes} failed\nnext diagnostic line"
        )

    def test_multiple_urls_and_header_redaction(self):
        from ytm_player.utils.doctor import _redact

        assert _redact(
            "https://one.invalid/?secret=one http://two.invalid/?secret=two\n"
            "Authorization: Bearer fake-auth\nCookie: SAPISID=fake-cookie\n"
            "Bearer fake-bearer\nstatus: 403; retry unavailable"
        ) == (
            "[URL REDACTED] [URL REDACTED]\nAuthorization: [REDACTED]\n"
            "Cookie: [REDACTED]\nBearer [REDACTED]\nstatus: 403; retry unavailable"
        )

    def test_normal_text_is_unchanged(self):
        from ytm_player.utils.doctor import _redact

        text = "Version: 2.1.0\nHTTP 403\nmpv: available\nconfig: /tmp/ytm/config.toml"
        assert _redact(text) == text

    def test_public_report_redacts_synthetic_logs_and_crash_without_rewriting(
        self, monkeypatch, tmp_path
    ):
        from ytm_player.config import paths
        from ytm_player.utils import doctor
        from ytm_player.utils import logging as log_utils

        for name in ("CONFIG_FILE", "THEME_FILE", "SESSION_STATE_FILE", "LOG_FILE"):
            monkeypatch.setattr(paths, name, tmp_path / name)
        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        monkeypatch.setattr(paths, "CRASH_DIR", crash_dir)
        for name in ("_mpv_version", "_libmpv_status", "_running_status", "_mpris_status"):
            monkeypatch.setattr(doctor, name, lambda: "synthetic collector")
        monkeypatch.setattr(log_utils, "list_active_hooks", lambda: "synthetic hooks")
        url = "https://media.invalid/play?ip=192.0.2.42&sig=fake-signature&pot=fake-token"
        payload = f"[WARNING] mpv[file]: HTTP 403 opening '{url}' failed\n"
        paths.LOG_FILE.write_text(payload, encoding="utf-8")
        crash = crash_dir / "ytm-crash-20260908-000000-000000.log"
        crash.write_text(f"=== Crash ===\nversion: 2.0.0\n\n{payload}", encoding="utf-8")
        fault = crash_dir / "faulthandler.log"
        fault.write_text(f"Fatal Python error: synthetic\n{payload}", encoding="utf-8")
        before = {p: p.read_bytes() for p in (paths.LOG_FILE, crash, fault)}

        report = doctor.gather_diagnostics()

        assert report.count("[URL REDACTED]") == 4  # log, mpv, faulthandler, crash
        assert "HTTP 403" in report
        assert "=== Paths ===" in report
        assert "synthetic hooks" in report
        for private in ("192.0.2.42", "fake-signature", "fake-token", "media.invalid"):
            assert private not in report
        assert {p: p.read_bytes() for p in before} == before


class TestGatherDiagnosticsExisting:
    """v1 sections must still work."""

    def test_includes_version(self):
        from ytm_player.utils.doctor import gather_diagnostics

        report = gather_diagnostics()
        from ytm_player import __version__

        assert __version__ in report

    def test_includes_python_version(self):
        import sys

        from ytm_player.utils.doctor import gather_diagnostics

        report = gather_diagnostics()
        assert f"{sys.version_info.major}.{sys.version_info.minor}" in report

    def test_includes_platform(self):
        import platform

        from ytm_player.utils.doctor import gather_diagnostics

        report = gather_diagnostics()
        assert platform.system() in report


class TestGatherDiagnosticsV2:
    """v2 must include 8 sections in order, with redaction."""

    def test_section_headers_present(self):
        from ytm_player.utils.doctor import gather_diagnostics

        report = gather_diagnostics()
        assert "=== ytm-player diagnostics ===" in report
        assert "=== Paths ===" in report
        assert "=== Process status ===" in report
        assert "=== MPRIS / media keys ===" in report
        assert "=== Recent ERROR/WARNING (last 20) ===" in report
        assert "=== Recent mpv warnings/errors ===" in report
        assert "=== Most recent faulthandler trace ===" in report
        assert "=== Most recent crash file ===" in report
        assert "=== Active hooks ===" in report

    def test_section_order(self):
        from ytm_player.utils.doctor import gather_diagnostics

        report = gather_diagnostics()
        order = [
            "=== ytm-player diagnostics ===",
            "=== Paths ===",
            "=== Process status ===",
            "=== MPRIS / media keys ===",
            "=== Recent ERROR/WARNING (last 20) ===",
            "=== Recent mpv warnings/errors ===",
            "=== Most recent faulthandler trace ===",
            "=== Most recent crash file ===",
            "=== Active hooks ===",
        ]
        positions = [report.index(h) for h in order]
        assert positions == sorted(positions), f"Sections out of order: {positions}"

    def test_redacts_authorization_header(self, monkeypatch, tmp_path: Path):
        from ytm_player.config import paths
        from ytm_player.utils.doctor import gather_diagnostics

        log = tmp_path / "ytm.log"
        log.write_text("2026-04-30 [WARNING] foo: Authorization: Bearer abc123secret\n")
        monkeypatch.setattr(paths, "LOG_FILE", log)

        report = gather_diagnostics()
        assert "abc123secret" not in report
        assert "[REDACTED]" in report

    def test_redacts_cookie_header(self, monkeypatch, tmp_path: Path):
        from ytm_player.config import paths
        from ytm_player.utils.doctor import gather_diagnostics

        log = tmp_path / "ytm.log"
        log.write_text("2026-04-30 [WARNING] foo: Cookie: SAPISID=secret\n")
        monkeypatch.setattr(paths, "LOG_FILE", log)

        report = gather_diagnostics()
        assert "SAPISID=secret" not in report

    def test_mpv_section_filters_for_mpv_prefix(self, monkeypatch, tmp_path: Path):
        from ytm_player.config import paths
        from ytm_player.utils.doctor import gather_diagnostics

        log = tmp_path / "ytm.log"
        log.write_text(
            "2026-04-30 [WARNING] ytm_player: regular warning\n"
            "2026-04-30 [WARNING] ytm_player.services.player: mpv[ao]: format mismatch\n"
            "2026-04-30 [ERROR] ytm_player.services.player: mpv[file]: cannot open\n"
        )
        monkeypatch.setattr(paths, "LOG_FILE", log)

        report = gather_diagnostics()
        start = report.index("=== Recent mpv warnings/errors ===")
        end = report.index("=== Most recent faulthandler trace ===")
        mpv_section = report[start:end]
        assert "mpv[ao]: format mismatch" in mpv_section
        assert "mpv[file]: cannot open" in mpv_section
        assert "regular warning" not in mpv_section

    def test_faulthandler_section_shows_last_block_when_present(self, monkeypatch, tmp_path: Path):
        from ytm_player.config import paths
        from ytm_player.utils.doctor import gather_diagnostics

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        fh = crash_dir / "faulthandler.log"
        fh.write_text(
            "Fatal Python error: Segmentation fault\n\n"
            "Current thread 0x0 (most recent call first):\n"
            "  File 'a.py', line 1 in foo\n"
        )
        monkeypatch.setattr(paths, "CRASH_DIR", crash_dir)

        report = gather_diagnostics()
        start = report.index("=== Most recent faulthandler trace ===")
        end = report.index("=== Most recent crash file ===")
        section = report[start:end]
        assert "Fatal Python error: Segmentation fault" in section

    def test_faulthandler_section_when_absent(self, monkeypatch, tmp_path: Path):
        from ytm_player.config import paths
        from ytm_player.utils.doctor import gather_diagnostics

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        monkeypatch.setattr(paths, "CRASH_DIR", crash_dir)

        report = gather_diagnostics()
        start = report.index("=== Most recent faulthandler trace ===")
        end = report.index("=== Most recent crash file ===")
        section = report[start:end]
        body = section.lower()
        assert "no faulthandler trace" in body or "(empty" in body

    def test_active_hooks_section_lists_all(self):
        from ytm_player.utils.doctor import gather_diagnostics

        report = gather_diagnostics()
        start = report.index("=== Active hooks ===")
        section = report[start:]
        assert "sys.excepthook" in section
        assert "threading.excepthook" in section
        assert "sys.unraisablehook" in section
        assert "faulthandler" in section


class TestCrashStaleness:
    """A stale crash from an older build must not read as a live bug (#89)."""

    def test_flags_crash_from_older_version(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        note = _crash_staleness_note("=== Crash ===\nversion: 1.0.0\ntrace", "1.9.4")
        assert note is not None
        assert "1.0.0" in note
        assert "may already be fixed" in note

    def test_no_note_for_current_version(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        assert _crash_staleness_note("version: 1.9.4\ntrace", "1.9.4") is None

    def test_no_note_for_newer_version(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        assert _crash_staleness_note("version: 2.0.0\ntrace", "1.9.4") is None

    def test_soft_note_when_version_unrecorded(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        note = _crash_staleness_note("=== Crash ===\ntrace only, no version line", "1.9.4")
        assert note is not None
        assert "no version recorded" in note

    def test_no_note_for_unknown_sentinel(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        assert _crash_staleness_note("=== Crash ===\nversion: unknown\ntrace", "1.9.4") is None


class TestMprisStatus:
    """MPRIS section must tell Linux users when dbus-fast is missing (#110)."""

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux-only MPRIS path")
    def test_unavailable_reports_fix(self, monkeypatch):
        import ytm_player.services.mpris as mpris_mod
        from ytm_player.utils.doctor import _mpris_status

        monkeypatch.setattr(mpris_mod, "DBUS_AVAILABLE", False)
        status = _mpris_status()
        assert "UNAVAILABLE" in status
        assert "dbus-fast" in status
        assert "reinstall" in status.lower()

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux-only MPRIS path")
    def test_available_reports_bus_name(self, monkeypatch):
        import ytm_player.services.mpris as mpris_mod
        from ytm_player.utils.doctor import _mpris_status

        monkeypatch.setattr(mpris_mod, "DBUS_AVAILABLE", True)
        status = _mpris_status()
        assert "available" in status
        assert mpris_mod.BUS_NAME in status

    @pytest.mark.skipif(sys.platform == "linux", reason="non-Linux native-integration path")
    def test_non_linux_is_not_applicable(self):
        from ytm_player.utils.doctor import _mpris_status

        assert "n/a" in _mpris_status()

    def test_no_note_for_invalid_version_string(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        # An unparseable version must not crash diagnostics or emit a verdict.
        assert (
            _crash_staleness_note("=== Crash ===\nversion: not-a-version\ntrace", "1.9.4") is None
        )

    def test_ignores_version_line_inside_traceback_body(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        # A version-shaped line buried in the traceback must not be parsed as
        # the crash version — only the metadata header counts (Codex nit #1).
        content = (
            "=== Crash ===\n"
            "Traceback (most recent call last):\n"
            "  File 'x.py', line 1, in <module>\n"
            "version: 0.0.1\n"
        )
        note = _crash_staleness_note(content, "1.9.4")
        assert note is not None
        assert "no version recorded" in note

    def test_ignores_exception_shaped_first_line_in_old_crash(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        # Old unstamped crash whose first body line is itself "Word: value"
        # shaped (an exception line) must not be walked as metadata down to a
        # later version-shaped line (Codex 2nd-pass finding).
        content = "=== Crash ===\nValueError: bad config\nversion: 0.0.1\n"
        note = _crash_staleness_note(content, "1.9.4")
        assert note is not None
        assert "no version recorded" in note

    def test_diagnostics_report_flags_stale_crash(self, monkeypatch, tmp_path: Path):
        from ytm_player.config import paths
        from ytm_player.utils.doctor import gather_diagnostics

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        (crash_dir / "ytm-crash-20200101-000000-000000.log").write_text(
            "=== Crash ===\nversion: 0.0.1\nsome traceback", encoding="utf-8"
        )
        monkeypatch.setattr(paths, "CRASH_DIR", crash_dir)

        report = gather_diagnostics()
        assert "may already be fixed" in report


class TestRecentMpvLines:
    """The mpv-warning tail behind the Recent mpv warnings/errors section."""

    def test_keeps_matching_lines_joined_in_order(self, tmp_path):
        from ytm_player.utils.doctor import _recent_mpv_lines

        log = tmp_path / "ytm.log"
        log.write_text(
            "2026-01-01 [WARNING] ytm_player: regular warning\n"
            "2026-01-01 [WARNING] ytm_player.services.player: mpv[ao]: format mismatch\n"
            "2026-01-01 [ERROR] ytm_player.services.player: mpv[file]: cannot open\n"
        )

        assert _recent_mpv_lines(log) == (
            "2026-01-01 [WARNING] ytm_player.services.player: mpv[ao]: format mismatch\n"
            "2026-01-01 [ERROR] ytm_player.services.player: mpv[file]: cannot open\n"
        )

    def test_missing_file_marker(self, tmp_path):
        from ytm_player.utils.doctor import _recent_mpv_lines

        assert _recent_mpv_lines(tmp_path / "nope.log") == "(log file is empty or missing)"

    def test_unreadable_file_marker(self, tmp_path):
        # A directory passes .exists() but open() raises OSError.
        from ytm_player.utils.doctor import _recent_mpv_lines

        assert _recent_mpv_lines(tmp_path) == "(could not read log file)"

    def test_no_matches_marker(self, tmp_path):
        from ytm_player.utils.doctor import _recent_mpv_lines

        log = tmp_path / "ytm.log"
        log.write_text("only regular warnings\nand errors here\n")

        assert _recent_mpv_lines(log) == "(no mpv warnings/errors logged)"

    def test_tail_slice_keeps_last_n(self, tmp_path):
        from ytm_player.utils.doctor import _recent_mpv_lines

        log = tmp_path / "ytm.log"
        log.write_text("".join(f"mpv[{name}]: line {i}\n" for i, name in enumerate("abcde")))

        assert _recent_mpv_lines(log, n=2) == "mpv[d]: line 3\nmpv[e]: line 4\n"

    def test_zero_n_keeps_everything(self, tmp_path):
        # `matches[-0:]` is `matches[0:]` — a zero count is not an empty tail.
        from ytm_player.utils.doctor import _recent_mpv_lines

        log = tmp_path / "ytm.log"
        log.write_text("mpv[a]: 1\nmpv[b]: 2\n")

        assert _recent_mpv_lines(log, n=0) == "mpv[a]: 1\nmpv[b]: 2\n"

    def test_last_line_without_trailing_newline_is_kept(self, tmp_path):
        from ytm_player.utils.doctor import _recent_mpv_lines

        log = tmp_path / "ytm.log"
        log.write_text("mpv[a]: 1\nmpv[b]: no newline")

        assert _recent_mpv_lines(log) == "mpv[a]: 1\nmpv[b]: no newline"

    def test_substring_match_not_prefix(self):
        from ytm_player.utils.doctor import _mpv_lines

        assert _mpv_lines(["player warning mpv[ao]: mid-line\n"], 20) == (
            "player warning mpv[ao]: mid-line\n"
        )

    def test_negative_n_drops_leading_matches(self):
        from ytm_player.utils.doctor import _mpv_lines

        lines = ["mpv[a]: 1\n", "mpv[b]: 2\n", "mpv[c]: 3\n"]
        assert _mpv_lines(lines, -1) == "mpv[b]: 2\nmpv[c]: 3\n"
        # Beyond the length: the slice is empty (not the no-warnings marker).
        assert _mpv_lines(lines, -99) == ""

    def test_engine_never_mutates_input(self):
        from ytm_player.utils.doctor import _mpv_lines

        lines = ["mpv[a]: 1\n", "plain\n"]
        _mpv_lines(lines, 5)
        assert lines == ["mpv[a]: 1\n", "plain\n"]


class TestCrashStalenessPep440:
    """The PEP 440 ordering behind the staleness verdict — the same
    comparison packaging's ``Version`` performs, on both backends."""

    @staticmethod
    def note(content, current):
        from ytm_player.utils.doctor import _crash_staleness_note

        return _crash_staleness_note(content, current)

    def test_exact_note_wording(self):
        note = self.note("=== Crash ===\nversion: 1.9.4\n\nTraceback", "2.1.0")
        assert note == (
            "⚠ This crash was recorded by ytm-player 1.9.4, older than the "
            "installed 2.1.0 — it may already be fixed. Reproduce on the "
            "current version before reporting."
        )

    def test_pre_release_counts_as_older(self):
        assert self.note("version: 2.1.0rc1\n", "2.1.0") is not None
        assert self.note("version: 2.1.0alpha2\n", "2.1.0") is not None
        assert self.note("version: 2.1.0AlPhA1\n", "2.1.0") is not None

    def test_dev_release_counts_as_older(self):
        # A dev-only release sorts before every pre-release.
        assert self.note("version: 2.1.0.dev1\n", "2.1.0") is not None
        assert self.note("version: 2.1.0.dev1\n", "2.1.0a1") is not None

    def test_post_release_is_not_older(self):
        assert self.note("version: 2.1.0.post1\n", "2.1.0") is None
        assert self.note("version: 1.0.0-1\n", "1.0.0") is None

    def test_epoch_dominates(self):
        assert self.note("version: 1!2.0\n", "2!1.0") is not None
        assert self.note("version: 1!1.0\n", "2.0") is None  # epoch 1 beats 0
        assert self.note("version: 2.0\n", "1!1.0") is not None

    def test_trailing_zeros_compare_equal(self):
        assert self.note("version: 2.1\n", "2.1.0") is None
        assert self.note("version: 2.1.0.0.0\n", "2.1") is None

    def test_local_version_ordering(self):
        # No local sorts before any local; strings sort before numbers.
        assert self.note("version: 2.1.0\n", "2.1.0+local") is not None
        assert self.note("version: 2.1.0+build.1\n", "2.1.0+build.2") is not None
        assert self.note("version: 2.1.0+9\n", "2.1.0+10") is not None
        assert self.note("version: 2.1.0+abc\n", "2.1.0+2") is not None
        assert self.note("version: 2.1.0+local\n", "2.1.0") is None

    def test_v_prefix_and_padding(self):
        assert self.note("version: v1.9.4\n", "2.1.0") is not None
        assert self.note("version: v2.1.0\n", "2.1.0") is None

    def test_write_crash_file_header_shape_is_read(self):
        # The exact header logging._crash_metadata_header stamps.
        content = (
            "=== Main thread crash ===\n"
            "version:   2.0.1\n"
            "time:      2026-01-01T00:00:00\n"
            "python:    3.12.0\n"
            "platform:  Linux 6.1.0-arch1 (x86_64)\n"
            "\n"
            "Traceback body\n"
        )
        note = self.note(content, "2.1.0")
        assert note is not None
        assert "2.0.1" in note

    def test_giant_digit_version_is_a_no_verdict(self):
        # CPython's int(str) refuses >4300-digit numbers with a plain
        # ValueError; both backends treat such a version as unparseable
        # instead of crashing the doctor.
        assert self.note("version: " + "9" * 5000 + "\n", "2.1.0") is None
        assert self.note("version: 1.0+" + "9" * 5000 + "\n", "2.1.0") is None

    def test_unicode_folds_are_rejected(self):
        # packaging's (?a:) ASCII scoping: the long s and the Kelvin sign
        # are not ASCII letters.
        assert self.note("version: 1.0+\u017f\n", "2.1.0") is None  # ſ (long s)
        assert self.note("version: 1.0po\u017ft1\n", "2.1.0") is None
        assert self.note("version: 1.0+\u212a\n", "2.1.0") is None  # K (Kelvin sign)


class TestArgvIsYtm:
    """The running-instance matcher must see real entry points and skip
    bystanders (an editor with the repo open, say)."""

    def test_console_script_launch(self):
        # Console scripts run as `python /…/bin/ytm`: interpreter in argv[0],
        # script path in argv[1]. This is the shape every pip/AUR install has.
        from ytm_player.utils.doctor import _argv_is_ytm

        assert _argv_is_ytm(["/home/u/.venv/bin/python3", "/home/u/.venv/bin/ytm"])

    def test_bare_ytm(self):
        from ytm_player.utils.doctor import _argv_is_ytm

        assert _argv_is_ytm(["/usr/bin/ytm"])

    def test_python_dash_m(self):
        from ytm_player.utils.doctor import _argv_is_ytm

        assert _argv_is_ytm(["/usr/bin/python3", "-m", "ytm_player"])

    def test_editor_on_repo_is_not_ytm(self):
        from ytm_player.utils.doctor import _argv_is_ytm

        assert not _argv_is_ytm(["nvim", "/home/u/AI/ytm-player/src/ytm_player/cli.py"])

    def test_unrelated_process(self):
        from ytm_player.utils.doctor import _argv_is_ytm

        assert not _argv_is_ytm(["/usr/bin/firefox", "--new-window"])


try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestNativeParity:
    """When the Rust extension is built, its output must match the
    pure-Python fallback byte-for-byte on the same inputs (mirrors the
    parity checks the other ported utils rely on)."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize(
        "text",
        [
            "",
            "plain diagnostic line\n",
            "HTTP 403 opening 'https://media.invalid/play?ip=192.0.2.42&sig=x' failed",
            'HTTP://media.invalid/a(b)?unknown=fake-secret and "tcp://[2001:db8::42]:443"',
            "Authorization: Bearer fake-auth\nCookie: SAPISID=fake-cookie",
            "bearer fake-bearer\ntoken: fake-token\nx-goog-pageid: fake-page-id",
            "SAPISID=fake-cookie\nauthorization=fake\nstatus: 403; retry unavailable",
            "Version: 2.1.0\nmpv: available\nconfig: /tmp/ytm/config.toml",
        ],
    )
    def test_redact_native_matches_fallback(self, monkeypatch, text):
        from ytm_player.utils import doctor

        native = _ytm_player_native.redact(text)
        monkeypatch.setattr(doctor, "_native_redact", None)
        fallback = doctor._redact(text)
        assert native == fallback

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    @pytest.mark.parametrize(
        "argv",
        [
            [],
            [""],
            ["/"],
            ["/home/u/.venv/bin/python3", "/home/u/.venv/bin/ytm"],
            ["/usr/bin/ytm"],
            ["/usr/bin/python3", "-m", "ytm_player"],
            ["ytm/", "/usr/bin/ytm.exe"],
            ["nvim", "/home/u/AI/ytm-player/src/ytm_player/cli.py"],
            ["/usr/bin/firefox", "--new-window"],
            ["/usr/bin/python3", "-Im", "ytm_player", "--debug"],
        ],
    )
    def test_argv_is_ytm_native_matches_fallback(self, monkeypatch, argv):
        from ytm_player.utils import doctor

        native = _ytm_player_native.argv_is_ytm(argv)
        monkeypatch.setattr(doctor, "_native_argv_is_ytm", None)
        fallback = doctor._argv_is_ytm(argv)
        assert native == fallback


class TestDoctorReport:
    """The report assembly engine behind ``gather_diagnostics``."""

    @staticmethod
    def _fields(**overrides):
        base = {
            "version": "2.1.0",
            "py_major": 3,
            "py_minor": 12,
            "py_micro": 16,
            "system": "Linux",
            "release": "6.1.0-arch1",
            "machine": "x86_64",
            "mpv_status": "mpv 0.38.0",
            "libmpv_status": "libmpv: OK",
            "config_path": "/home/u/.config/ytm-player/config.toml",
            "theme_path": "/home/u/.config/ytm-player/theme.toml",
            "session_path": "/home/u/.config/ytm-player/session.json",
            "log_path": "/home/u/.config/ytm-player/logs/ytm.log",
            "crash_dir": "/home/u/.config/ytm-player/crashes",
            "process_status": "ytm not running",
            "mpris_status": "MPRIS: available (dbus-fast OK; registers as org.mpris.MediaPlayer2.ytm)",
            "recent_errors": "(none logged)",
            "recent_mpv": "(no mpv warnings/errors logged)",
            "faulthandler": "(no faulthandler trace — file does not exist)",
            "crash": None,
            "crash_note": None,
            "hooks": "sys.excepthook: installed",
        }
        base.update(overrides)
        return base

    def test_full_report_layout(self):
        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(**self._fields())
        expected = "\n".join(
            [
                "=== ytm-player diagnostics ===",
                "Version: 2.1.0",
                "Python:  3.12.16",
                "OS:      Linux 6.1.0-arch1",
                "Machine: x86_64",
                "mpv 0.38.0",
                "libmpv: OK",
                "",
                "=== Paths ===",
                "config:   /home/u/.config/ytm-player/config.toml",
                "theme:    /home/u/.config/ytm-player/theme.toml",
                "session:  /home/u/.config/ytm-player/session.json",
                "log:      /home/u/.config/ytm-player/logs/ytm.log",
                "crashes:  /home/u/.config/ytm-player/crashes",
                "",
                "=== Process status ===",
                "ytm not running",
                "",
                "=== MPRIS / media keys ===",
                "MPRIS: available (dbus-fast OK; registers as org.mpris.MediaPlayer2.ytm)",
                "",
                "=== Recent ERROR/WARNING (last 20) ===",
                "(none logged)",
                "",
                "=== Recent mpv warnings/errors ===",
                "(no mpv warnings/errors logged)",
                "",
                "=== Most recent faulthandler trace ===",
                "(no faulthandler trace — file does not exist)",
                "",
                "=== Most recent crash file ===",
                "(no crash files found)",
                "",
                "=== Active hooks ===",
                "sys.excepthook: installed",
                "",
                "Note: this section reflects the doctor subcommand's own process. "
                "Hooks are installed inside the long-running ytm TUI process — "
                "their effect is visible above (crashes/, faulthandler.log).",
            ]
        )
        assert report == expected

    def test_crash_section_unpacks_the_pair(self):
        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(
            **self._fields(
                crash=("/crashes/c.log", "=== Crash ===\ntraceback body"),
                crash_note="⚠ stale note",
            )
        )
        start = report.index("=== Most recent crash file ===")
        end = report.index("=== Active hooks ===")
        assert report[start:end] == (
            "=== Most recent crash file ===\n"
            "From: /crashes/c.log\n"
            "=== Crash ===\n"
            "traceback body\n"
            "⚠ stale note\n"
            "\n"
        )

    def test_crash_note_none_omits_the_line(self):
        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(**self._fields(crash=("/crashes/c.log", "body"), crash_note=None))
        assert "From: /crashes/c.log\nbody\n\n=== Active hooks ===" in report

    def test_empty_string_crash_note_still_lands(self):
        # `if note is not None` — not a truthiness gate.
        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(**self._fields(crash=("/c.log", "body"), crash_note=""))
        assert "From: /c.log\nbody\n\n\n=== Active hooks ===" in report

    def test_bad_crash_shape_raises_unpack_error(self):
        from ytm_player.utils.doctor import _doctor_report

        with pytest.raises(ValueError, match="too many values to unpack"):
            _doctor_report(**self._fields(crash=("a", "b", "c")))
        with pytest.raises(ValueError, match="not enough values to unpack"):
            _doctor_report(**self._fields(crash=("a",)))

    @pytest.mark.parametrize("falsy", [None, "", 0, 0.0, False, [], {}])
    def test_falsy_recent_errors_get_the_placeholder(self, falsy):
        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(**self._fields(recent_errors=falsy))
        assert "(no warnings or errors logged)" in report

    def test_multiline_probe_values_are_appended_raw(self):
        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(
            **self._fields(
                recent_errors="2026-01-01 [ERROR] one\n2026-01-01 [WARNING] two",
                recent_mpv="mpv[ao]: line\nmpv[file]: another",
            )
        )
        assert (
            "=== Recent ERROR/WARNING (last 20) ===\n2026-01-01 [ERROR] one\n2026-01-01 [WARNING] two"
            in report
        )
        assert "=== Recent mpv warnings/errors ===\nmpv[ao]: line\nmpv[file]: another" in report

    def test_path_objects_render_through_str(self):
        from pathlib import Path

        from ytm_player.utils.doctor import _doctor_report

        report = _doctor_report(
            **self._fields(config_path=Path("/x/config.toml"), crash_dir=Path("/x/crashes"))
        )
        assert "config:   /x/config.toml" in report
        assert "crashes:  /x/crashes" in report

    def test_report_is_redacted(self):
        from ytm_player.utils.doctor import _doctor_report

        url = "https://media.invalid/play?ip=192.0.2.42&sig=fake-signature"
        report = _doctor_report(
            **self._fields(
                recent_mpv=f"[WARNING] mpv[file]: HTTP 403 opening '{url}' failed",
                crash=("/c.log", f"Authorization: Bearer fake-token\n{url}"),
            )
        )
        assert "[URL REDACTED]" in report
        assert "Authorization: [REDACTED]" in report
        for private in ("192.0.2.42", "fake-signature", "fake-token", "media.invalid"):
            assert private not in report

    def test_inputs_never_mutated(self):
        from ytm_player.utils.doctor import _doctor_report

        crash = ("/c.log", "body")
        fields = self._fields(crash=crash, crash_note="note")
        _doctor_report(**fields)
        assert crash == ("/c.log", "body")
        assert fields["version"] == "2.1.0"


# ── Doctor-report native vs fallback parity ────────────────────────────

# The matrix below is shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable (the crash pair travels as a
# two-item list — the reference's unpacking is shape-agnostic).
_DOCTOR_PARITY_CASES = [
    {
        "version": "2.1.0",
        "py_major": 3,
        "py_minor": 12,
        "py_micro": 16,
        "system": "Linux",
        "release": "6.1.0",
        "machine": "x86_64",
        "mpv_status": "mpv 0.38.0",
        "libmpv_status": "libmpv: OK",
        "config_path": "/home/u/.config/ytm-player/config.toml",
        "theme_path": "/home/u/.config/ytm-player/theme.toml",
        "session_path": "/home/u/.config/ytm-player/session.json",
        "log_path": "/home/u/.config/ytm-player/logs/ytm.log",
        "crash_dir": "/home/u/.config/ytm-player/crashes",
        "process_status": "ytm not running",
        "mpris_status": "MPRIS: available",
        "recent_errors": "(none logged)",
        "recent_mpv": "(no mpv warnings/errors logged)",
        "faulthandler": "(no faulthandler trace)",
        "crash": None,
        "crash_note": None,
        "hooks": "sys.excepthook: installed",
    },
    # The crash section with and without a note.
    {
        "version": "2.1.0",
        "py_major": 3,
        "py_minor": 10,
        "py_micro": 1,
        "system": "Darwin",
        "release": "23.1.0",
        "machine": "arm64",
        "mpv_status": "mpv: NOT FOUND in PATH",
        "libmpv_status": "libmpv: NOT LOADABLE (first line)",
        "config_path": "/Users/u/.config/ytm-player/config.toml",
        "theme_path": "/Users/u/.config/ytm-player/theme.toml",
        "session_path": "/Users/u/.config/ytm-player/session.json",
        "log_path": "/Users/u/.config/ytm-player/logs/ytm.log",
        "crash_dir": "/Users/u/.config/ytm-player/crashes",
        "process_status": "ytm running: PID 42 (RSS 120 MB)",
        "mpris_status": "MPRIS: n/a (Darwin uses its native media integration)",
        "recent_errors": "line one\nline two",
        "recent_mpv": "mpv[ao]: format mismatch",
        "faulthandler": "Fatal Python error: Segmentation fault",
        "crash": ["/crashes/ytm-crash-x.log", "=== Crash ===\nversion: 1.0.0\ntrace"],
        "crash_note": "⚠ This crash was recorded by ytm-player 1.0.0, older than the installed 2.1.0",
        "hooks": "sys.excepthook: installed\nfaulthandler: enabled",
    },
    # A falsy recent_errors substitution and an empty-string note.
    {
        "version": "9.9.9",
        "py_major": 3,
        "py_minor": 14,
        "py_micro": 0,
        "system": "Windows",
        "release": "11",
        "machine": "AMD64",
        "mpv_status": "mpv 0.37",
        "libmpv_status": "libmpv: OK",
        "config_path": "C:\\Users\\u\\config.toml",
        "theme_path": "C:\\Users\\u\\theme.toml",
        "session_path": "C:\\Users\\u\\session.json",
        "log_path": "C:\\Users\\u\\ytm.log",
        "crash_dir": "C:\\Users\\u\\crashes",
        "process_status": "(cannot determine — /proc not available on this OS)",
        "mpris_status": "MPRIS: n/a",
        "recent_errors": "",
        "recent_mpv": "",
        "faulthandler": "",
        "crash": ["/c.log", "body with unicode ✓ and\nnewlines"],
        "crash_note": "",
        "hooks": "",
    },
]

_DOCTOR_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _doctor_report

payload = json.load(sys.stdin)
json.dump([_doctor_report(**fields) for fields in payload], sys.stdout)
"""


class TestDoctorReportNativeParity:
    """The report-assembly engine, on the same child-interpreter pattern
    the redact / argv_is_ytm parity tests above use."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_doctor_report_matches_fallback(self):
        from ytm_player.utils.doctor import _doctor_report

        result = subprocess.run(
            [sys.executable, "-c", _DOCTOR_CHILD_PROG],
            input=json.dumps(_DOCTOR_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_doctor_report(**fields) for fields in _DOCTOR_PARITY_CASES]
        for fields, n, f in zip(_DOCTOR_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback reports differ for {fields['system']} "
                f"crash={fields['crash']!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )


# ── mpv-tail native vs fallback parity ────────────────────────────────

# Shared with the child interpreter as JSON, so every value must be
# JSON-round-trippable. (lines, n) pairs; n covers Python's whole [-n:]
# slice range — positive tails, the zero-keeps-everything quirk, and
# negative counts dropping leading matches.
_MPV_PARITY_CASES = [
    ([], 20),
    (["plain line\n"], 20),
    (["mpv[ao]: one\n"], 20),
    (["plain\n", "mpv[a]: 1\n", "plain\n", "mpv[b]: 2\n"], 20),
    (["mpv[a]: 1\n", "mpv[b]: 2\n", "mpv[c]: 3\n"], 1),
    (["mpv[a]: 1\n", "mpv[b]: 2\n"], 0),
    (["mpv[a]: 1\n", "mpv[b]: 2\n", "mpv[c]: 3\n"], -1),
    (["mpv[a]: 1\n", "mpv[b]: 2\n"], -99),
    (["mpv[a]: 1\n"], 99),
    (["mpv[a]: no trailing newline"], 20),
    (["mpv[a]: 1\r\n", "mpv[b]: 2\r\n"], 20),
    (["warning mpv[x]: mid-line match\n"], 20),
    (["", "mpv[a]: between empties\n", ""], 20),
    (["mpv[]: bare bracket pair\n"], 20),
    (["unicode ✓ mpv[ao]: ünïcode ✗\n"], 20),
]

_MPV_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _mpv_lines

payload = json.load(sys.stdin)
json.dump([_mpv_lines(lines, n) for lines, n in payload], sys.stdout)
"""


class TestRecentMpvLinesNativeParity:
    """The mpv-warning tail engine, on the same child-interpreter pattern
    the doctor-report parity test above uses."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_recent_mpv_lines_matches_fallback(self):
        from ytm_player.utils.doctor import _mpv_lines

        result = subprocess.run(
            [sys.executable, "-c", _MPV_CHILD_PROG],
            input=json.dumps(_MPV_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_mpv_lines(lines, n) for lines, n in _MPV_PARITY_CASES]
        for (lines, n), nv, fb in zip(_MPV_PARITY_CASES, native, fallback):
            assert nv == fb, (
                f"native and fallback mpv tails differ for n={n} lines={lines!r}:\n"
                f"native:   {nv!r}\n"
                f"fallback: {fb!r}"
            )


# ── Crash-staleness native vs fallback parity ─────────────────────────

# (content, current_version) pairs, JSON-round-trippable. The matrix
# covers the header scan (banners, known/unknown/unicode keys, blanks,
# exotic separators and line boundaries, values-with-trailing-text) and
# the PEP 440 comparison space (epochs, pre/post/dev ranks, trailing
# zeros, local segments, case-insensitive spellings, invalid and
# digit-limit-overflowing versions).
_CRASH_PARITY_CASES = [
    # Header-scan shapes.
    ("", "2.1.0"),
    ("=== Main thread crash ===", "2.1.0"),
    ("=== Crash ===\nversion:   2.0.0\n\nTraceback body", "2.1.0"),
    ("version: 2.0.0\nno banner at all", "2.1.0"),
    (
        "=== Crash ===\ntime:      2026-01-01T00:00:00\npython:    3.12.0\n"
        "platform:  Linux 6.1.0-arch1 (x86_64)\nversion:   2.0.0\n\nbody",
        "2.1.0",
    ),
    ("=== Crash ===\ntime: 2026-01-01\n\nversion: 2.0.0\nbody", "2.1.0"),
    ("=== Crash ===\nValueError: bad config\nversion: 0.0.1\n", "1.9.4"),
    ("=== Crash ===\nTraceback (most recent call last):\n  version: 0.0.1\n", "1.9.4"),
    ("version:1.0\n", "2.1.0"),  # no whitespace after the colon -> not a header line
    ("Version: 2.0.0\n", "2.1.0"),  # key membership is case-sensitive
    ("versión: 2.0.0\n", "2.1.0"),  # unicode key is not in the metadata set
    ("version:   2.0.0 trailing words here\n", "2.1.0"),  # \S+ takes the first token only
    ("version: \n", "2.1.0"),  # blank value -> not a header line
    ("version:\t2.0.0\n", "2.1.0"),  # tab separator
    ("version:  2.0.0 \n", "2.1.0"),  # trailing whitespace stays out of the value
    ("version:\x1c1.0\n", "2.1.0"),  # \x1c is a splitlines boundary, not a separator
    ("=== Crash ===\r\nversion: 2.0.0\r\n", "2.1.0"),  # CRLF content
    ("=== Crash ===\u2028version: 2.0.0\u2029", "2.1.0"),  # exotic line boundaries
    ("version: 1.0:2\n", "2.1.0"),  # the value may contain colons
    # The unknown sentinel and the missing version.
    ("=== Crash ===\nversion: unknown\ntrace", "2.1.0"),
    ("=== Crash ===\ntrace only, no version line", "1.9.4"),
    # Older versions -> the stale note.
    ("version: 1.0.0\n", "2.1.0"),
    ("version: 2.0.9\n", "2.1.0"),
    ("version: 2.1.0rc1\n", "2.1.0"),
    ("version: 2.1.0a1\n", "2.1.0"),
    ("version: 2.1.0alpha2\n", "2.1.0"),
    ("version: 2.1.0AlPhA1\n", "2.1.0"),
    ("version: 2.1.0b2\n", "2.1.0"),
    ("version: 2.1.0preview3\n", "2.1.0"),
    ("version: 2.1.0.pre1\n", "2.1.0"),
    ("version: 2.1.0_c3\n", "2.1.0"),
    ("version: 2.1.0.dev1\n", "2.1.0"),
    ("version: 2.1.0dev\n", "2.1.0"),
    ("version: v1.9.4\n", "2.1.0"),
    ("version: 1!2.0\n", "2!1.0"),
    ("version: 1.0.0-1\n", "2.0.0"),
    ("version: 1.0a2.post2.dev3\n", "1.0"),
    ("version: 2.1.0+build.1\n", "2.1.0+build.2"),
    ("version: 2.1.0+9\n", "2.1.0+10"),
    ("version: 2.1.0+abc\n", "2.1.0+2"),
    ("version: 2.1.0\n", "2.1.0+local"),
    ("version:  2.0.0  \n", "2.1.0"),
    ("version: 007.0002\n", "7.3"),
    # Equal / newer -> no note.
    ("version: 2.1.0\n", "2.1.0"),
    ("version: 2.1\n", "2.1.0"),
    ("version: 2.1.0.0.0\n", "2.1"),
    ("version: v2.1.0\n", "2.1.0"),
    ("version: 2.1.0.post1\n", "2.1.0"),
    ("version: 2.1.0.post1.dev1\n", "2.1.0.post1"),
    ("version: 3.0\n", "2.1.0"),
    ("version: 2.1.1\n", "2.1.0"),
    ("version: 2.1.0+local\n", "2.1.0"),
    ("version: 1!1.0\n", "2.0"),
    ("version: 2!0.1\n", "2.1.0"),
    # Invalid versions -> no note.
    ("version: not-a-version\n", "2.1.0"),
    ("version: 1..0\n", "2.1.0"),
    ("version: 1.0+\n", "2.1.0"),
    ("version: 1.0++local\n", "2.1.0"),
    ("version: 1.0+\u017f\n", "2.1.0"),  # long s — outside packaging's ASCII local alphabet
    ("version: 1.0po\u017ft1\n", "2.1.0"),
    ("version: 1.0+\u212a\n", "2.1.0"),  # Kelvin sign U+212A
    ("version: ²1.0\n", "2.1.0"),
    ("version: 1.0-4-5\n", "2.1.0"),
    ("version: 2.1.0-c4\n", "2.1.0"),  # implicit post needs digits after the hyphen
    # The int-conversion digit limit (>4300 digits) on either side.
    ("version: " + "9" * 5000 + "\n", "2.1.0"),
    ("version: 1.0+" + "9" * 5000 + "\n", "2.1.0"),
    ("version: 1.0a" + "9" * 5000 + "\n", "2.1.0"),
    ("version: 1.0\n", "2.1.0rc1"),  # a valid current version with a pre-release
    ("version: 1.0\n", "  2.1.0  "),  # whitespace-padded current version
]

_CRASH_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _crash_staleness_note

payload = json.load(sys.stdin)
json.dump([_crash_staleness_note(content, current) for content, current in payload], sys.stdout)
"""


class TestCrashStalenessNativeParity:
    """The crash-staleness verdict engine, on the same child-interpreter
    pattern the doctor-report parity test above uses."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_crash_staleness_note_matches_fallback(self):
        from ytm_player.utils.doctor import _crash_staleness_note

        result = subprocess.run(
            [sys.executable, "-c", _CRASH_CHILD_PROG],
            input=json.dumps(_CRASH_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [
            _crash_staleness_note(content, current) for content, current in _CRASH_PARITY_CASES
        ]
        for (content, current), nv, fb in zip(_CRASH_PARITY_CASES, native, fallback):
            assert nv == fb, (
                f"native and fallback staleness notes differ for "
                f"content={content!r} current={current!r}:\n"
                f"native:   {nv!r}\n"
                f"fallback: {fb!r}"
            )
