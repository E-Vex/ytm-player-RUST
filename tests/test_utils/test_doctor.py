"""Tests for utils.doctor — diagnostic gathering for `ytm doctor`."""

from __future__ import annotations

import json
import re
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


class TestFaulthandlerTail:
    """The `Fatal Python error:` tail behind the Most recent
    faulthandler trace section."""

    def test_last_block_from_marker_to_end(self, tmp_path):
        from ytm_player.utils.doctor import _recent_faulthandler

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        (crash_dir / "faulthandler.log").write_text(
            "Fatal Python error: Segmentation fault\n\n"
            "Current thread 0x0 (most recent call first):\n"
            "  File 'a.py', line 1 in foo\n"
        )

        assert _recent_faulthandler(crash_dir) == (
            "Fatal Python error: Segmentation fault\n\n"
            "Current thread 0x0 (most recent call first):\n"
            "  File 'a.py', line 1 in foo\n"
        )

    def test_last_of_several_blocks_wins(self, tmp_path):
        from ytm_player.utils.doctor import _recent_faulthandler

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        (crash_dir / "faulthandler.log").write_text(
            "Fatal Python error: Segmentation fault\nold block\n"
            "Fatal Python error: Aborted\nnew block\n"
        )

        assert _recent_faulthandler(crash_dir) == "Fatal Python error: Aborted\nnew block\n"

    def test_missing_file_marker(self, tmp_path):
        from ytm_player.utils.doctor import _recent_faulthandler

        assert _recent_faulthandler(tmp_path) == "(no faulthandler trace — file does not exist)"

    def test_unreadable_file_marker(self, tmp_path):
        # A directory passes .exists() but read_text() raises OSError.
        from ytm_player.utils.doctor import _recent_faulthandler

        crash_dir = tmp_path / "crashes"
        (crash_dir / "faulthandler.log").mkdir(parents=True)

        assert _recent_faulthandler(crash_dir) == "(could not read faulthandler.log)"

    def test_empty_file_marker(self, tmp_path):
        from ytm_player.utils.doctor import _recent_faulthandler

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        (crash_dir / "faulthandler.log").write_text("")

        assert _recent_faulthandler(crash_dir) == (
            "(faulthandler.log is empty — no fatal signals captured)"
        )

    def test_long_log_without_marker_is_tailed_through_the_file(self, tmp_path):
        from ytm_player.utils.doctor import _recent_faulthandler

        crash_dir = tmp_path / "crashes"
        crash_dir.mkdir()
        (crash_dir / "faulthandler.log").write_text("z" * 4500)

        assert _recent_faulthandler(crash_dir) == "z" * 4000

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " ",
            "\t\n \r\n\x0b\x0c",
            "\x1c\x1d\x1e\x1f",
            "\u00a0\u1680\u2028\u2029\u3000",
        ],
    )
    def test_whitespace_only_text_is_the_empty_marker(self, text):
        # The strip set is CPython's str.isspace — \x1c-\x1f count where
        # Rust's own is_whitespace would stop at White_Space.
        from ytm_player.utils.doctor import _faulthandler_tail

        assert _faulthandler_tail(text) == (
            "(faulthandler.log is empty — no fatal signals captured)"
        )

    def test_no_marker_short_text_is_returned_whole(self):
        from ytm_player.utils.doctor import _faulthandler_tail

        text = "no fatal errors here\njust noise\n"
        assert _faulthandler_tail(text) == text

    def test_no_marker_text_of_exactly_4000_chars_is_returned_whole(self):
        from ytm_player.utils.doctor import _faulthandler_tail

        text = "y" * 4000
        assert _faulthandler_tail(text) == text

    def test_no_marker_tail_caps_at_4000_chars(self):
        from ytm_player.utils.doctor import _faulthandler_tail

        assert _faulthandler_tail("x" * 4001) == "x" * 4000

    def test_tail_counts_characters_not_bytes(self):
        # text[-4000:] slices code points: a 4001-char multi-byte text
        # loses exactly its first character, not its first bytes.
        from ytm_player.utils.doctor import _faulthandler_tail

        assert _faulthandler_tail("é" * 4001) == "é" * 4000

    def test_marker_wins_over_the_4000_char_tail(self):
        # A marker earlier than the last 4000 characters still opens the
        # slice — the two paths never mix.
        from ytm_player.utils.doctor import _faulthandler_tail

        text = "Fatal Python error: old\n" + "x" * 5000
        assert _faulthandler_tail(text) == text

    def test_marker_only_text(self):
        from ytm_player.utils.doctor import _faulthandler_tail

        assert _faulthandler_tail("Fatal Python error:") == "Fatal Python error:"

    def test_partial_or_lowercase_marker_is_not_a_marker(self):
        from ytm_player.utils.doctor import _faulthandler_tail

        text = "fatal python error: Segmentation fault\n"  # lowercase — not the marker
        assert _faulthandler_tail(text) == text
        text = "Fatal Python error no colon\n"
        assert _faulthandler_tail(text) == text


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

    def test_unicode_folds_follow_the_installed_packaging(self):
        # CPython's re.IGNORECASE folds a handful of non-ASCII letters onto
        # ASCII ones. packaging compiled its PEP 440 grammar with no ASCII
        # scoping through 26.0 — the long s (U+017F), the Kelvin sign
        # (U+212A) and the dotted and dotless i (U+0130/U+0131) were valid
        # local-segment material, and the s of "post" / the i of "preview"
        # accepted their folds — while 26.1+ scopes the grammar's groups
        # (?a:...) and rejects the same spellings. The engine's contract is
        # to agree with whichever packaging release is installed (the same
        # Version() the pure-Python fallback itself calls), so the expected
        # verdict is computed from that Version, not pinned to either side.
        from packaging.version import Version

        def parses(v: str) -> bool:
            try:
                Version(v)
            except ValueError:
                return False
            return True

        folds_admitted = parses("1.0+\u017f")
        assert folds_admitted == parses("1.0+\u212a")
        assert folds_admitted == parses("1.0+\u0130")
        assert folds_admitted == parses("1.0po\u017ft1")
        assert folds_admitted == parses("2.1.0prev\u0131ew1")

        expected = (
            "⚠ This crash was recorded by ytm-player 1.0+\u017f, older than "
            "the installed 2.1.0 — it may already be fixed. Reproduce on "
            "the current version before reporting."
        )
        if folds_admitted:
            assert self.note("version: 1.0+\u017f\n", "2.1.0") == expected  # ſ (long s)
            assert self.note("version: 1.0+\u212a\n", "2.1.0") is not None  # K (Kelvin sign)
            assert self.note("version: 1.0+\u0130\n", "2.1.0") is not None  # İ (dotted capital i)
            assert self.note("version: 1.0+\u0131\n", "2.1.0") is not None  # ı (dotless i)
            assert self.note("version: 1.0po\u017ft1\n", "2.1.0") is not None
            assert self.note("version: 2.1.0prev\u0131ew1\n", "2.1.0") is not None
        else:
            # packaging 26.1+: the fold spellings are InvalidVersion, so
            # every one of these is a no-verdict instead.
            assert self.note("version: 1.0+\u017f\n", "2.1.0") is None
            assert self.note("version: 1.0+\u212a\n", "2.1.0") is None
            assert self.note("version: 1.0+\u0130\n", "2.1.0") is None
            assert self.note("version: 1.0+\u0131\n", "2.1.0") is None
            assert self.note("version: 1.0po\u017ft1\n", "2.1.0") is None
            assert self.note("version: 2.1.0prev\u0131ew1\n", "2.1.0") is None

    def test_exotic_marker_spellings_keep_their_raw_order(self):
        # packaging normalises only the exact marker spellings; on a
        # fold-admitting packaging (<= 26.0) a spelling like "prevıew" or
        # "poſt" survives the lower() untouched and compares as its own
        # string — between b and rc for the pre-release letter, above
        # "post" for the post one — while the plain spellings collapse
        # onto their canonical letters. The alpha/beta pair below is
        # packaging-independent; the fold pairs run only where the
        # installed packaging admits folds (26.1+ rejects the spellings
        # outright, a no-verdict on both backends).
        from packaging.version import Version

        assert self.note("version: 2.1.0alpha1\n", "2.1.0b1") is not None
        assert self.note("version: 2.1.0b1\n", "2.1.0alpha1") is None

        try:
            Version("2.1.0prev\u0131ew1")
        except ValueError:
            return  # installed packaging rejects fold spellings

        assert self.note("version: 2.1.0prev\u0131ew1\n", "2.1.0rc1") is not None
        assert self.note("version: 2.1.0rc1\n", "2.1.0prev\u0131ew1") is None
        assert self.note("version: 1.0-1\n", "1.0-po\u017ft") is not None
        assert self.note("version: 1.0-po\u017ft\n", "1.0-1") is None


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
        # The report renders Path objects through str() — the expectation
        # must spell them the same way str(Path) does on this platform.
        assert f"config:   {Path('/x/config.toml')}" in report
        assert f"crashes:  {Path('/x/crashes')}" in report

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


# ── faulthandler-tail native vs fallback parity ───────────────────────

# Shared with the child interpreter as JSON, so every value must be
# JSON-round-trippable. Texts cover the strip-empty zoo (CPython's
# str.isspace set, including the \x1c-\x1f file separators where Rust's
# own is_whitespace disagrees), the marker paths (single, repeated — the
# last block wins —, partial and case-mangled non-markers, the marker
# with and without surrounding noise), and the 4000-character tail
# boundary (exactly at it, one past it in ASCII, in two-byte and in
# combining characters, the marker before it, and the marker after it).
_FAULTHANDLER_PARITY_CASES = [
    "",
    " ",
    "\t\n \r\n\x0b\x0c",
    "\x1c\x1d\x1e\x1f",
    "\u00a0\u1680\u2028\u2029\u3000",
    "  \n not strip-empty: leading and trailing noise  ",
    "Fatal Python error:",
    "Fatal Python error: Segmentation fault\n",
    "noise before\nFatal Python error: Aborted\nCurrent thread 0x0 (most recent call first):\n",
    "Fatal Python error: A\nblock one\nFatal Python error: B\nblock two\n",
    "Fatal Python error: A\n" + "x" * 5000,
    "x" * 4000 + "Fatal Python error: late\nend",
    "fatal python error: lowercase is not the marker\n",
    "Fatal Python error missing its colon\n",
    "Fatal Python Error: capital E is not the marker either\n",
    "no marker at all, short text",
    "x" * 4000,
    "x" * 4001,
    "x" * 4500,
    "ab" * 2001,
    "é" * 4001,
    "e\u0301" * 4001,
    "ünïcödé ✓ " * 500,
    "line one\r\nFatal Python error: Windows CRLF block\r\nframe\r\n",
    "Fatal Python error: trailing text without a newline",
    "\x00\x01\x02not whitespace, no marker",
]

_FAULTHANDLER_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _faulthandler_tail

payload = json.load(sys.stdin)
json.dump([_faulthandler_tail(text) for text in payload], sys.stdout)
"""


class TestFaulthandlerTailNativeParity:
    """The faulthandler-tail engine, on the same child-interpreter
    pattern the doctor-report parity test above uses."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_faulthandler_tail_matches_fallback(self):
        from ytm_player.utils.doctor import _faulthandler_tail

        result = subprocess.run(
            [sys.executable, "-c", _FAULTHANDLER_CHILD_PROG],
            input=json.dumps(_FAULTHANDLER_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_faulthandler_tail(text) for text in _FAULTHANDLER_PARITY_CASES]
        for text, nv, fb in zip(_FAULTHANDLER_PARITY_CASES, native, fallback):
            assert nv == fb, (
                f"native and fallback faulthandler tails differ for {text[:60]!r}:\n"
                f"native:   {nv[:80]!r}\n"
                f"fallback: {fb[:80]!r}"
            )


# ── Crash-staleness native vs fallback parity ─────────────────────────

# (content, current_version) pairs, JSON-round-trippable. The matrix
# covers the header scan (banners, known/unknown/unicode keys, blanks,
# exotic separators and line boundaries, values-with-trailing-text) and
# the PEP 440 comparison space (epochs, pre/post/dev ranks — the exotic
# Unicode-fold marker spellings included — trailing zeros, local
# segments, case-insensitive spellings, invalid and digit-limit-
# overflowing versions).
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
    # Older via the pre/post letter's raw string: alpha normalises onto a
    # (below b), while an exotic fold spelling like prevıew stays raw and
    # ranks between b and rc; poſt stays raw and ranks above post.
    ("version: 2.1.0alpha1\n", "2.1.0b1"),
    ("version: 2.1.0prev\u0131ew1\n", "2.1.0rc1"),
    ("version: 1.0-1\n", "1.0-po\u017ft"),
    # Older through the Unicode folds re.IGNORECASE admits: the long s, the
    # Kelvin sign and the dotted/dotless i are valid local material, and
    # they spell marker words too (the s of "post", the i of "preview").
    ("version: 1.0+\u017f\n", "2.1.0"),
    ("version: 1.0+\u212a\n", "2.1.0"),
    ("version: 1.0+\u0130\n", "2.1.0"),
    ("version: 1.0+\u0131\n", "2.1.0"),
    ("version: 1.0po\u017ft1\n", "2.1.0"),
    ("version: 2.1.0prev\u0131ew1\n", "2.1.0"),
    ("version: 2.1.0prev\u0130ew1\n", "2.1.0"),
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

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_both_packaging_grammar_shapes_are_reachable(self):
        """The engine's fold flag selects between the two grammars
        packaging has shipped, regardless of which one is installed.

        With ``admit_unicode_folds=True`` (packaging <= 26.0's unscoped
        ``re.IGNORECASE``) the fold spellings parse and order by their raw
        characters; with ``False`` (26.1+ ``(?a:...)`` scoping) the same
        spellings are no-verdicts. Every non-fold verdict is identical
        across both shapes — the grammars differ in exactly three
        positions, none of which an ASCII spelling can reach.
        """
        from ytm_player.utils._ytm_player_native import crash_staleness_note

        fold_cases = [
            ("version: 1.0+\u017f\n", "2.1.0"),  # ſ local
            ("version: 1.0+\u212a\n", "2.1.0"),  # K local
            ("version: 1.0+\u0130\n", "2.1.0"),  # İ local
            ("version: 1.0+\u0131\n", "2.1.0"),  # ı local
            ("version: 1.0po\u017ft1\n", "2.1.0"),  # poſt marker
            ("version: 2.1.0prev\u0131ew1\n", "2.1.0"),  # prevıew marker
            ("version: 2.1.0prev\u0130ew1\n", "2.1.0"),  # prevİew marker
            ("version: 2.1.0prev\u0131ew1\n", "2.1.0rc1"),  # raw ordering
            ("version: 1.0-1\n", "1.0-po\u017ft"),  # raw post ordering
        ]
        for content, current in fold_cases:
            assert crash_staleness_note(content, current, True) is not None, (content, current)
            assert crash_staleness_note(content, current, False) is None, (content, current)

        # The surrounding grammar is shared: ASCII spellings give the same
        # verdict under both shapes — a stale note or none, as pinned.
        shared_cases = [
            ("version: 1.9.4\n", "2.1.0", True),
            ("version: 2.1.0\n", "2.1.0", False),
            ("version: 2.1.0rc1\n", "2.1.0", True),
            ("version: 2.1.0alpha1\n", "2.1.0b1", True),
            ("version: 2.1.0.post1\n", "2.1.0", False),
            ("version: 2.1.0.dev1\n", "2.1.0a1", True),
            ("version: 2.1.0+build.1\n", "2.1.0+build.2", True),
            ("version: not-a-version\n", "2.1.0", False),
            ("version: 1.0++local\n", "2.1.0", False),
        ]
        for content, current, stale in shared_cases:
            a = crash_staleness_note(content, current, True)
            b = crash_staleness_note(content, current, False)
            assert a == b, (content, current, a, b)
            assert (a is not None) is stale, (content, current, a)

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_probe_flag_matches_the_installed_packaging(self):
        """The production wrapper's probe steers the engine to the
        installed packaging's verdict, whichever release that is."""
        from packaging.version import Version

        import ytm_player.utils.doctor as doctor_mod

        try:
            Version("1.0+\u017f")
            admitted = True
        except ValueError:
            admitted = False

        assert doctor_mod._packaging_admits_unicode_folds() is admitted
        assert doctor_mod._PACKAGING_ADMITS_UNICODE_FOLDS is admitted

        note = doctor_mod._crash_staleness_note("version: 1.0+\u017f\n", "2.1.0")
        assert (note is not None) is admitted


# ── The /proc walk behind the Process status section ───────────────────


class _ProcEntry:
    """One synthetic /proc entry: a name, a cmdline byte string and a
    status text, with optional OSError markers on either read."""

    def __init__(self, name, cmdline=b"", status="", cmdline_error=None, status_error=None):
        self.name = name
        self.cmdline = cmdline
        self.status = status
        self.cmdline_error = cmdline_error
        self.status_error = status_error


def _proc_thunks(case):
    """The five /proc thunks for a synthetic case dict, plus the call log.

    Case shape (JSON-round-trippable for the parity matrix):
    ``{"exists": bool, "entries": [{"name": str, "cmdline": [int],
    "status": str, "cmdline_error": "OSError" | None, "status_error":
    "OSError" | None}], "own_pid": int | str, "exists_error":
    "RuntimeError" | None}`` — the entry dicts become `_ProcEntry`
    objects, the shapes the production thunks pass around.
    """
    entries = [
        _ProcEntry(
            e["name"],
            bytes(e.get("cmdline") or []),
            e.get("status") or "",
            e.get("cmdline_error"),
            e.get("status_error"),
        )
        for e in case["entries"]
    ]
    log: list[str] = []

    def proc_exists():
        log.append("exists")
        if case.get("exists_error"):
            raise RuntimeError("proc probe broke")
        return case["exists"]

    def proc_entries():
        log.append("iterdir")
        return iter(entries)

    def read_cmdline(entry):
        log.append(f"cmdline:{entry.name}")
        if entry.cmdline_error:
            raise OSError(f"cmdline gone: {entry.name}")
        return entry.cmdline

    def read_status(entry):
        log.append(f"status:{entry.name}")
        if entry.status_error:
            raise OSError(f"status gone: {entry.name}")
        return entry.status

    def own_pid():
        log.append("getpid")
        return case["own_pid"]

    return proc_exists, proc_entries, read_cmdline, read_status, own_pid, log


_YTM_CONSOLE = [ord(c) for c in "/home/u/.venv/bin/python3"] + [0]
_YTM_CONSOLE += [ord(c) for c in "/home/u/.venv/bin/ytm"] + [0]
_YTM_DASH_M = [ord(c) for c in "/usr/bin/python3"] + [0]
_YTM_DASH_M += [ord(c) for c in "-m"] + [0]
_YTM_DASH_M += [ord(c) for c in "ytm_player"] + [0]
_FIREFOX = [ord(c) for c in "/usr/bin/firefox"] + [0]


class TestRunningStatusText:
    """The /proc walk behind the Process status section — on whichever
    backend is loaded."""

    def _run(self, case):
        from ytm_player.utils.doctor import _running_status_text

        thunks = _proc_thunks(case)
        return _running_status_text(*thunks[:5]), thunks[5]

    def test_missing_proc_is_the_cannot_determine_marker(self):
        text, log = self._run(
            {"exists": False, "entries": [{"name": "1", "cmdline": _YTM_CONSOLE}], "own_pid": 0}
        )
        assert text == "(cannot determine — /proc not available on this OS)"
        # Nothing else is probed — no listing, no reads, no getpid.
        assert log == ["exists"]

    def test_proc_probe_error_propagates(self):
        with pytest.raises(RuntimeError, match="proc probe broke"):
            self._run(
                {
                    "exists": True,
                    "entries": [],
                    "own_pid": 0,
                    "exists_error": "RuntimeError",
                }
            )

    def test_no_ytm_processes(self):
        text, log = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "cpuinfo"},
                    {"name": "1", "cmdline": _FIREFOX},
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm not running"
        # getpid is never consulted when nothing was collected.
        assert "getpid" not in log

    def test_running_ytm_reports_pid_and_rss(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "1234", "cmdline": _YTM_CONSOLE, "status": "VmRSS:\t1536 kB\n"}
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1234 (RSS 1 MB)"

    def test_python_dash_m_matches_too(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {
                        "name": "42",
                        "cmdline": _YTM_DASH_M,
                        "status": "Name:\tytm\nVmRSS:\t1048576 kB\n",
                    }
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 42 (RSS 1024 MB)"

    def test_multiple_pids_join_in_entry_order(self):
        text, log = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "7", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1024 kB"},
                    {"name": "8", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 2048 kB"},
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 7 (RSS 1 MB), PID 8 (RSS 2 MB)"
        # One getpid consultation per collected pid — the reference's
        # comprehension evaluates the filter once per element.
        assert log.count("getpid") == 2

    def test_own_pid_is_excluded(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "5", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1 kB"}],
                "own_pid": 5,
            }
        )
        assert text == "ytm not running"

    def test_non_int_own_pid_is_never_equal(self):
        # pid != own_pid through CPython's != — an int pid is never
        # equal to a string, so the entry survives the filter.
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "5", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1024 kB"}],
                "own_pid": "5",
            }
        )
        assert text == "ytm running: PID 5 (RSS 1 MB)"

    def test_non_digit_names_never_get_their_cmdline_read(self):
        _, log = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "self"},
                    {"name": "cpuinfo"},
                    {"name": "123abc"},
                    {"name": ""},
                ],
                "own_pid": 0,
            }
        )
        assert "iterdir" in log
        assert not any(call.startswith("cmdline:") for call in log)

    def test_superscript_digit_name_passes_the_digit_gate(self):
        # "¹" carries Numeric_Type=Digit — CPython's isdigit accepts it
        # where a plain ASCII check would stop — so its cmdline is read
        # like any digit-named entry (here: empty, so the walk skips on
        # the argv before the name's int() would ever run).
        _, log = self._run(
            {"exists": True, "entries": [{"name": "¹", "cmdline": []}], "own_pid": 0}
        )
        assert "cmdline:¹" in log

    def test_unrelated_cmdline_never_gets_its_status_read(self):
        _, log = self._run(
            {"exists": True, "entries": [{"name": "1", "cmdline": _FIREFOX}], "own_pid": 0}
        )
        assert "cmdline:1" in log
        assert not any(call.startswith("status:") for call in log)

    def test_cmdline_oserror_skips_the_entry(self):
        text, log = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "1", "cmdline": _YTM_CONSOLE, "cmdline_error": "OSError"},
                    {"name": "2", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1024 kB"},
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 2 (RSS 1 MB)"
        assert "status:1" not in log  # the unread entry never reaches the status read

    def test_status_oserror_skips_the_entry(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status_error": "OSError"}],
                "own_pid": 0,
            }
        )
        assert text == "ytm not running"

    def test_cmdline_non_oserror_propagates(self):
        with pytest.raises(RuntimeError, match="cmdline broke"):
            from ytm_player.utils.doctor import _running_status_text

            _running_status_text(
                lambda: True,
                lambda: iter([_ProcEntry("1", b"ytm")]),
                lambda entry: (_ for _ in ()).throw(RuntimeError("cmdline broke")),
                lambda entry: "",
                lambda: 0,
            )

    def test_empty_cmdline_is_skipped(self):
        text, log = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "1", "cmdline": []},
                    {"name": "2", "cmdline": [0, 0, 0]},  # NULs become spaces, then nothing
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm not running"
        assert log.count("cmdline:1") == 1
        assert log.count("cmdline:2") == 1

    def test_missing_vmrss_line_is_zero_mb(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {"name": "1", "cmdline": _YTM_CONSOLE, "status": "Name:\tytm\nVmSize: 1 kB"}
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 0 MB)"

    def test_malformed_vmrss_word_is_zero_mb(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 5,368 kB"}],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 0 MB)"

    def test_bare_vmrss_prefix_is_zero_mb(self):
        # "VmRSS:" alone splits to one word — the IndexError arm.
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS:"}],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 0 MB)"

    def test_first_vmrss_line_wins(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {
                        "name": "1",
                        "cmdline": _YTM_CONSOLE,
                        "status": "VmRSS: 2048 kB\nVmRSS: 999999 kB",
                    }
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 2 MB)"

    def test_vmrss_underscore_spelling_parses(self):
        # int("1_024") is 1024 — CPython's int() acceptance set.
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1_024 kB"}],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 1 MB)"

    def test_vmrss_beyond_i64_renders_exactly(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {
                        "name": "1",
                        "cmdline": _YTM_CONSOLE,
                        "status": "VmRSS: 99999999999999999999999 kB",
                    }
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 97656249999999999999 MB)"

    def test_status_line_boundaries_are_cpythons(self):
        # \x85 is a str.splitlines boundary — the VmRSS line behind it
        # is its own line, exactly like the reference's parse.
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {
                        "name": "1",
                        "cmdline": _YTM_CONSOLE,
                        "status": "Name:\tytm\x85VmRSS: 5120 kB",
                    }
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 5 MB)"

    def test_cmdline_invalid_utf8_decodes_with_replacement(self):
        # 0xff and a truncated four-byte start both become U+FFFD — the
        # match still sees the real argv[0] in front of them.
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {
                        "name": "1",
                        "cmdline": _YTM_CONSOLE[:1] + [0xFF, 0xEF, 0xF0, 0x9F, 0x98],
                        "status": "VmRSS: 1024 kB",
                    }
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm not running"  # argv[0] "/h\uFFFD\uFFFD..." is not a ytm entry point

    def test_cmdline_trailing_garbage_still_matches_the_head(self):
        text, _ = self._run(
            {
                "exists": True,
                "entries": [
                    {
                        "name": "1",
                        "cmdline": _YTM_CONSOLE + [0xFF, 0xEF, 0xF0, 0x9F, 0x98],
                        "status": "VmRSS: 1024 kB",
                    }
                ],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 1 (RSS 1 MB)"

    def test_unicode_decimal_digit_name_parses(self):
        # "٣" (Arabic-Indic three) is a digit isdigit accepts and int
        # parses — the same CPython semantics both backends borrow.
        text, _ = self._run(
            {
                "exists": True,
                "entries": [{"name": "٣", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 2048 kB"}],
                "own_pid": 0,
            }
        )
        assert text == "ytm running: PID 3 (RSS 2 MB)"

    def test_superscript_digit_name_raises_like_the_reference(self):
        # "²" passes isdigit but int rejects it — the reference crashes
        # here, so the engine must too (both through CPython's own int).
        with pytest.raises(ValueError):
            self._run(
                {
                    "exists": True,
                    "entries": [{"name": "²", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1 kB"}],
                    "own_pid": 0,
                }
            )

    def test_entry_without_a_name_attribute_raises(self):
        from ytm_player.utils.doctor import _running_status_text

        with pytest.raises(AttributeError):
            _running_status_text(
                lambda: True,
                lambda: iter([object()]),
                lambda entry: b"",
                lambda entry: "",
                lambda: 0,
            )


# ── The mpv version probe behind the mpv status line ───────────────────


class _Out:
    """A CompletedProcess-shaped stub whose stdout/stderr attribute reads
    can be made to raise (the attribute itself, like a property would)."""

    def __init__(self, stdout=None, stderr=None, stdout_error=None, stderr_error=None):
        self._stdout = stdout
        self._stderr = stderr
        self._stdout_error = stdout_error
        self._stderr_error = stderr_error

    @property
    def stdout(self):
        if self._stdout_error:
            raise {
                "OSError": OSError,
                "RuntimeError": RuntimeError,
                "AttributeError": AttributeError,
            }[self._stdout_error]("stdout broke")
        return self._stdout

    @property
    def stderr(self):
        if self._stderr_error:
            raise {
                "OSError": OSError,
                "RuntimeError": RuntimeError,
                "AttributeError": AttributeError,
            }[self._stderr_error]("stderr broke")
        return self._stderr


def _mpv_thunks(case):
    """The which + run thunks for a synthetic mpv probe case, plus the log.

    Case shape (JSON-round-trippable for the parity matrix):
    ``{"which": str | None, "stdout": str | None, "stderr": str | None,
    "stdout_error" / "stderr_error": str | None, "run_error":
    [kind, arg] | None}``.
    """
    import subprocess as _sp

    log: list[str] = []

    def which_mpv():
        log.append("which")
        return case["which"]

    def run_version(mpv_bin):
        log.append(f"run:{mpv_bin}")
        err = case.get("run_error")
        if err is not None:
            kind, arg = err
            if kind == "FileNotFoundError":
                raise FileNotFoundError(arg)
            if kind == "TimeoutExpired":
                raise _sp.TimeoutExpired(cmd=arg, timeout=3)
            if kind == "ValueError":
                raise ValueError(arg)
        return _Out(
            case.get("stdout"),
            case.get("stderr"),
            case.get("stdout_error"),
            case.get("stderr_error"),
        )

    return which_mpv, run_version, log


class TestMpvVersionText:
    """The mpv version probe — on whichever backend is loaded."""

    def _run(self, case):
        from ytm_player.utils.doctor import _mpv_version_text

        which_mpv, run_version, log = _mpv_thunks(case)
        return _mpv_version_text(which_mpv, run_version), log

    def test_missing_mpv_is_not_found(self):
        text, log = self._run({"which": None, "stdout": "never read"})
        assert text == "mpv: NOT FOUND in PATH"
        # The subprocess probe never runs — the PATH scan already decided.
        assert log == ["which"]

    def test_empty_which_result_is_also_not_found(self):
        text, log = self._run({"which": "", "stdout": "never read"})
        assert text == "mpv: NOT FOUND in PATH"
        assert log == ["which"]

    def test_version_line_comes_from_stdout(self):
        text, log = self._run(
            {"which": "/usr/bin/mpv", "stdout": "mpv 0.38.0 (C) 2000-2026\nmore lines\n"}
        )
        assert text == "mpv: mpv 0.38.0 (C) 2000-2026"
        assert log == ["which", "run:/usr/bin/mpv"]

    def test_stdout_without_trailing_newline(self):
        text, _ = self._run({"which": "/usr/bin/mpv", "stdout": "mpv 0.38.0"})
        assert text == "mpv: mpv 0.38.0"

    def test_stdout_none_falls_back_to_stderr(self):
        text, _ = self._run(
            {"which": "/usr/bin/mpv", "stdout": None, "stderr": "built from source\n"}
        )
        assert text == "mpv: built from source"

    def test_stdout_empty_falls_back_to_stderr(self):
        text, _ = self._run({"which": "/usr/bin/mpv", "stdout": "", "stderr": "warning line\n"})
        assert text == "mpv: warning line"

    def test_both_empty_is_unknown(self):
        text, _ = self._run({"which": "/usr/bin/mpv", "stdout": "", "stderr": ""})
        assert text == "mpv: unknown"

    def test_both_none_is_unknown(self):
        text, _ = self._run({"which": "/usr/bin/mpv", "stdout": None, "stderr": None})
        assert text == "mpv: unknown"

    def test_line_boundaries_are_cpythons(self):
        # \x85 is a str.splitlines boundary — the first line stops there.
        text, _ = self._run({"which": "/usr/bin/mpv", "stdout": "mpv 0.1\u0085hidden tail"})
        assert text == "mpv: mpv 0.1"

    def test_stderr_is_only_read_when_stdout_is_falsy(self):
        # A truthy stdout satisfies the or-chain — the stderr attribute
        # (here: one that raises) is never read.
        text, _ = self._run(
            {"which": "/usr/bin/mpv", "stdout": "mpv 0.38.0", "stderr_error": "RuntimeError"}
        )
        assert text == "mpv: mpv 0.38.0"

    def test_stderr_attribute_error_propagates(self):
        with pytest.raises(AttributeError, match="stderr broke"):
            self._run({"which": "/usr/bin/mpv", "stdout": "", "stderr_error": "AttributeError"})

    def test_stdout_oserror_is_failed_to_execute(self):
        # The attribute read sits inside the reference's try block, so
        # its OSError lands in the failed-to-execute arm.
        text, _ = self._run({"which": "/usr/bin/mpv", "stdout_error": "OSError"})
        assert text == "mpv: failed to execute"

    def test_missing_stdout_attribute_propagates(self):
        from ytm_player.utils.doctor import _mpv_version_text

        class _Bare:
            pass

        with pytest.raises(AttributeError):
            _mpv_version_text(lambda: "/mpv", lambda b: _Bare())

    def test_oserror_from_the_run_is_failed_to_execute(self):
        text, _ = self._run({"which": "/mpv", "run_error": ["FileNotFoundError", "/mpv"]})
        assert text == "mpv: failed to execute"

    def test_subprocess_error_from_the_run_is_failed_to_execute(self):
        text, _ = self._run({"which": "/mpv", "run_error": ["TimeoutExpired", "/mpv --version"]})
        assert text == "mpv: failed to execute"

    def test_other_errors_from_the_run_propagate(self):
        with pytest.raises(ValueError, match="boom"):
            self._run({"which": "/mpv", "run_error": ["ValueError", "boom"]})

    def test_which_probe_runs_exactly_once(self):
        _, log = self._run({"which": "/usr/bin/mpv", "stdout": "mpv 1\n"})
        assert log.count("which") == 1


# ── The libmpv import probe behind the libmpv status line ──────────────


class _OddStr:
    """A message object whose ``__str__`` returns the text — or raises
    the configured error instead (the reference's ``str()`` sits inside
    its try block, so an ``ImportError`` from it is the OK marker)."""

    def __init__(self, text, str_error=None):
        self._text = text
        self._str_error = str_error

    def __str__(self):
        if self._str_error is not None:
            raise self._str_error("str broke")
        return self._text


class TestLibmpvStatusText:
    """The libmpv loadability probe — on whichever backend is loaded."""

    def _run(self, msg_or_error):
        from ytm_player.utils.doctor import _libmpv_status_text

        def import_player():
            if isinstance(msg_or_error, BaseException):
                raise msg_or_error
            return msg_or_error

        return _libmpv_status_text(import_player)

    def test_import_error_is_the_ok_marker(self):
        assert self._run(ImportError("cannot import name '_IMPORT_ERROR_MSG'")) == "libmpv: OK"

    def test_module_import_failure_is_also_ok(self):
        # The module's own import failing raises the same ImportError
        # through the reference's single from-import — same arm.
        assert self._run(ImportError("No module named 'mpv'")) == "libmpv: OK"

    def test_the_last_line_of_the_message_renders(self):
        assert self._run("Cannot load libmpv.\n  Linux: apt install mpv\n\nOriginal: boom") == (
            "libmpv: NOT LOADABLE (Original: boom)"
        )

    def test_single_line_message(self):
        assert self._run("one line") == "libmpv: NOT LOADABLE (one line)"

    def test_no_trailing_newline(self):
        assert self._run("a\nb") == "libmpv: NOT LOADABLE (b)"

    def test_the_last_line_is_stripped(self):
        assert self._run("head\n  \t padded \t ") == "libmpv: NOT LOADABLE (padded)"

    def test_the_strip_is_pythons_whitespace_set(self):
        # \x1c–\x1f are str.isspace() whitespace for CPython (and so
        # stripped by str.strip()) but not Unicode White_Space — a trim
        # that stops at White_Space would leave them behind.
        assert self._run("head\n\x1cpadded\x1f") == "libmpv: NOT LOADABLE (padded)"

    def test_non_breaking_space_is_stripped(self):
        assert self._run("head\n\xa0nbsp\xa0") == "libmpv: NOT LOADABLE (nbsp)"

    def test_empty_message_is_an_index_error(self):
        # splitlines() of "" is [] — the [-1] index raises through,
        # exactly like the reference's (the except clause covers
        # ImportError only).
        with pytest.raises(IndexError, match="list index out of range"):
            self._run("")

    def test_line_boundaries_are_cpythons(self):
        # \x85 and \u2028 are str.splitlines boundaries — the last line
        # stops at them.
        assert self._run("a\x85b") == "libmpv: NOT LOADABLE (b)"
        assert self._run("a\u2028b") == "libmpv: NOT LOADABLE (b)"

    def test_non_import_errors_propagate(self):
        with pytest.raises(RuntimeError, match="probe broke"):
            self._run(RuntimeError("probe broke"))

    def test_str_conversion_runs_inside_the_try(self):
        # A message object whose __str__ raises ImportError lands in the
        # OK arm — the reference's str() sits inside its try block.
        assert self._run(_OddStr("ignored", ImportError)) == "libmpv: OK"

    def test_str_conversion_error_of_other_kinds_propagates(self):
        with pytest.raises(RuntimeError, match="str broke"):
            self._run(_OddStr("ignored", RuntimeError))

    def test_str_conversion_of_an_odd_object_renders_its_text(self):
        assert self._run(_OddStr("a\nb")) == "libmpv: NOT LOADABLE (b)"

    def test_the_probe_runs_exactly_once(self):
        from ytm_player.utils.doctor import _libmpv_status_text

        calls = []

        def import_player():
            calls.append("import")
            return "a\nb"

        _libmpv_status_text(import_player)
        assert calls == ["import"]

    @pytest.mark.skipif(sys.platform == "win32", reason="player module imports on win32 too")
    def test_the_real_player_probe(self):
        # The production thunk shape: the wrapper's import against the
        # real player module — either the OK marker (libmpv loadable) or
        # the NOT LOADABLE line naming the message's last line.
        from ytm_player.utils.doctor import _libmpv_status

        status = _libmpv_status()
        assert status == "libmpv: OK" or status.startswith("libmpv: NOT LOADABLE (")


# ── The MPRIS import probe behind the MPRIS status line ────────────────


def _cpython_unpack2_error(values: object) -> str:
    """The exact ValueError this interpreter raises for a 2-unpack of
    *values* — the spelling drifted across versions (3.14 appends
    ", got N" to the too-many side), so the tests derive it instead of
    hardcoding."""
    try:
        _k, _v = values
    except ValueError as exc:
        return str(exc)
    raise AssertionError(f"2-unpack of {values!r} unexpectedly succeeded")


class TestMprisStatusText:
    """The MPRIS availability probe — on whichever backend is loaded."""

    def _run(self, is_linux, system_name, pair_or_error):
        from ytm_player.utils.doctor import _mpris_status_text

        def import_mpris():
            if isinstance(pair_or_error, BaseException):
                raise pair_or_error
            return pair_or_error

        return _mpris_status_text(is_linux, system_name, import_mpris)

    def test_non_linux_is_na_with_the_system_name(self):
        assert self._run(False, lambda: "Darwin", RuntimeError("never imported")) == (
            "MPRIS: n/a (Darwin uses its native media integration)"
        )

    def test_the_system_probe_never_runs_on_linux(self):
        def system_name():
            raise RuntimeError("platform.system() must not run on Linux")

        assert self._run(True, system_name, ("org.mpris.MediaPlayer2.ytm_player", True)) == (
            "MPRIS: available (dbus-fast OK; registers as org.mpris.MediaPlayer2.ytm_player)"
        )

    def test_available_line_names_the_bus_name(self):
        assert self._run(True, None, ("org.mpris.MediaPlayer2.ytm_player", True)) == (
            "MPRIS: available (dbus-fast OK; registers as org.mpris.MediaPlayer2.ytm_player)"
        )

    def test_unavailable_text_carries_the_fix(self):
        text = self._run(True, None, ("org.mpris.MediaPlayer2.ytm_player", False))
        assert text == (
            "MPRIS: UNAVAILABLE — dbus-fast is missing or failed to load (it's a "
            "Linux core dependency, so this install is broken/partial; an "
            "incompatible dbus-fast build logs the failure to ytm.log — see the "
            "recent-errors section below). Fix: reinstall ytm-player "
            "(pip install --force-reinstall ytm-player), or pip install dbus-fast."
        )

    def test_truthiness_decides_the_verdict(self):
        # The reference's `if DBUS_AVAILABLE:` — Python truthiness, not
        # an is-True check.
        assert "available" in self._run(True, None, ("n", 1))
        assert "available" in self._run(True, None, ("n", "yes"))
        assert "UNAVAILABLE" in self._run(True, None, ("n", 0))
        assert "UNAVAILABLE" in self._run(True, None, ("n", ""))
        assert "UNAVAILABLE" in self._run(True, None, ("n", None))

    def test_bus_name_renders_through_str(self):
        assert self._run(True, None, (_OddStr("bus-of-î"), True)) == (
            "MPRIS: available (dbus-fast OK; registers as bus-of-î)"
        )

    def test_import_error_propagates(self):
        # Unlike the libmpv probe, the reference has no try around the
        # import — an ImportError surfaces, it is not the marker.
        with pytest.raises(ImportError, match="No module named 'dbus_fast'"):
            self._run(True, None, ImportError("No module named 'dbus_fast'"))

    def test_system_probe_error_propagates(self):
        with pytest.raises(RuntimeError, match="system broke"):
            self._run(False, lambda: (_ for _ in ()).throw(RuntimeError("system broke")), None)

    def test_short_pair_is_a_value_error(self):
        with pytest.raises(ValueError, match=re.escape(_cpython_unpack2_error(("name",)))):
            self._run(True, None, ("name",))

    def test_long_pair_is_a_value_error(self):
        with pytest.raises(ValueError, match=re.escape(_cpython_unpack2_error(("a", "b", "c")))):
            self._run(True, None, ("a", "b", "c"))

    def test_non_iterable_pair_is_a_type_error(self):
        with pytest.raises(TypeError, match="cannot unpack non-iterable int object"):
            self._run(True, None, 5)

    def test_the_pair_unpacks_like_cpythons_opcode(self):
        # A generator, an __getitem__ sequence and a 2-char string all
        # unpack through the iterator protocol exactly like
        # UNPACK_SEQUENCE(2).
        assert self._run(True, None, (g for g in ("gen-name", 1))) == (
            "MPRIS: available (dbus-fast OK; registers as gen-name)"
        )
        assert self._run(True, None, "a1") == "MPRIS: available (dbus-fast OK; registers as a)"
        assert "UNAVAILABLE" in self._run(True, None, (g for g in ("n", "")))

    def test_the_system_probe_runs_exactly_once(self):
        calls = []

        def system_name():
            calls.append("system")
            return "Darwin"

        self._run(False, system_name, RuntimeError("never imported"))
        assert calls == ["system"]

    def test_falsy_is_linux_shapes(self):
        # The caller-computed gate follows the reference's own
        # truthiness — any falsy value is the n/a branch.
        assert self._run(0, lambda: "OS/2", RuntimeError("never")) == (
            "MPRIS: n/a (OS/2 uses its native media integration)"
        )
        assert self._run(None, lambda: "Haiku", RuntimeError("never")) == (
            "MPRIS: n/a (Haiku uses its native media integration)"
        )

    def test_the_real_probe_on_this_platform(self):
        # The production thunk shape: the wrapper's platform gate and
        # import against the real mpris module.
        from ytm_player.utils.doctor import _mpris_status

        status = _mpris_status()
        if sys.platform == "linux":
            assert "MPRIS: " in status
        else:
            assert "MPRIS: n/a (" in status


# ── Native vs fallback parity for the two probe engines ────────────────

# Both matrices are shared with the child interpreter as JSON, so every
# value must be JSON-round-trippable (cmdlines as byte-value lists).

_RUNNING_PARITY_CASES = [
    # The missing /proc.
    {"exists": False, "entries": [{"name": "1", "cmdline": _YTM_CONSOLE}], "own_pid": 0},
    # Nothing running.
    {"exists": True, "entries": [], "own_pid": 0},
    {"exists": True, "entries": [{"name": "cpuinfo"}, {"name": "self"}], "own_pid": 0},
    # Non-matches: non-digit names, unrelated cmdlines, empty cmdlines.
    {"exists": True, "entries": [{"name": "123abc"}, {"name": "¹"}, {"name": ""}], "own_pid": 0},
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _FIREFOX, "status": "VmRSS: 1 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": []}, {"name": "2", "cmdline": [0, 0]}],
        "own_pid": 0,
    },
    # The own-pid exclusion (and the getpid call count in the log).
    {
        "exists": True,
        "entries": [{"name": "5", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1 kB"}],
        "own_pid": 5,
    },
    {
        "exists": True,
        "entries": [{"name": "5", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1 kB"}],
        "own_pid": "5",
    },
    # Matches, with the VmRSS zoo.
    {
        "exists": True,
        "entries": [{"name": "1234", "cmdline": _YTM_CONSOLE, "status": "VmRSS:\t1536 kB\n"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "42", "cmdline": _YTM_DASH_M, "status": "VmRSS: 1048576 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "Name:\tytm"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 5,368 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS:"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1_024 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: +2 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: ٩٩٩ kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [
            {"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 99999999999999999999999 kB"}
        ],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [
            {"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 2048 kB\nVmRSS: 7 kB"}
        ],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [
            {"name": "1", "cmdline": _YTM_CONSOLE, "status": "Name:\tytm\u0085VmRSS: 5120 kB"}
        ],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [
            {"name": "1", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 512 kB\u2028VmRSS: 2048 kB"}
        ],
        "own_pid": 0,
    },
    # Several pids, in order, one excluded.
    {
        "exists": True,
        "entries": [
            {"name": "7", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1024 kB"},
            {"name": "8", "cmdline": _FIREFOX, "status": "VmRSS: 1 kB"},
            {"name": "9", "cmdline": _YTM_DASH_M, "status": "VmRSS: 2048 kB"},
        ],
        "own_pid": 9,
    },
    # OSError skips — cmdline and status.
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "cmdline_error": "OSError"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status_error": "OSError"}],
        "own_pid": 0,
    },
    # The lossy-decode zoo on the cmdline bytes.
    {
        "exists": True,
        "entries": [
            {
                "name": "1",
                "cmdline": _YTM_CONSOLE + [0xFF, 0xEF, 0xF0, 0x9F, 0x98],
                "status": "VmRSS: 1024 kB",
            }
        ],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [
            {
                "name": "1",
                "cmdline": _YTM_CONSOLE + [0xC0, 0xAF, 0xED, 0xA0, 0x80],
                "status": "VmRSS: 1024 kB",
            }
        ],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [
            {"name": "1", "cmdline": _YTM_CONSOLE + [0xE0, 0x80], "status": "VmRSS: 1024 kB"}
        ],
        "own_pid": 0,
    },
    # The unicode-name zoo.
    {
        "exists": True,
        "entries": [{"name": "٣", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 2048 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "١٢", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1024 kB"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "²", "cmdline": _YTM_CONSOLE, "status": "VmRSS: 1 kB"}],
        "own_pid": 0,
    },
    # Error shapes that must match: a raising exists probe, a non-OSError
    # from a read, a nameless entry.
    {"exists": True, "entries": [], "own_pid": 0, "exists_error": "RuntimeError"},
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "cmdline_error": "RuntimeError"}],
        "own_pid": 0,
    },
    {
        "exists": True,
        "entries": [{"name": "1", "cmdline": _YTM_CONSOLE, "status_error": "RuntimeError"}],
        "own_pid": 0,
    },
]

_RUNNING_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _running_status_text


class _ProcEntry:
    def __init__(self, name, cmdline, status, cmdline_error, status_error):
        self.name = name
        self.cmdline = cmdline
        self.status = status
        self.cmdline_error = cmdline_error
        self.status_error = status_error


def build_thunks(case):
    entries = [
        _ProcEntry(
            e["name"],
            bytes(e.get("cmdline") or []),
            e.get("status") or "",
            e.get("cmdline_error"),
            e.get("status_error"),
        )
        for e in case["entries"]
    ]
    log = []

    def proc_exists():
        log.append("exists")
        if case.get("exists_error"):
            raise RuntimeError("proc probe broke")
        return case["exists"]

    def proc_entries():
        log.append("iterdir")
        return iter(entries)

    def read_cmdline(entry):
        log.append(f"cmdline:{entry.name}")
        if entry.cmdline_error:
            raise OSError(f"cmdline gone: {entry.name}")
        return entry.cmdline

    def read_status(entry):
        log.append(f"status:{entry.name}")
        if entry.status_error:
            raise OSError(f"status gone: {entry.name}")
        return entry.status

    def own_pid():
        log.append("getpid")
        return case["own_pid"]

    return proc_exists, proc_entries, read_cmdline, read_status, own_pid, log


def outcome(case):
    thunks = build_thunks(case)
    try:
        text = _running_status_text(*thunks[:5])
        return {"ok": text, "log": thunks[5]}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": thunks[5]}


cases = json.load(sys.stdin)
json.dump([outcome(case) for case in cases], sys.stdout)
"""


class TestRunningStatusNativeParity:
    """When the Rust extension is built, the native /proc walk must
    decide exactly like the pure-Python reference — same verdicts, same
    skip sets, same getpid counts, same error shapes — for every case,
    so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_running_status_text_matches_fallback(self):
        from ytm_player.utils.doctor import _running_status_text

        result = subprocess.run(
            [sys.executable, "-c", _RUNNING_CHILD_PROG],
            input=json.dumps(_RUNNING_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = []
        for case in _RUNNING_PARITY_CASES:
            thunks = _proc_thunks(case)
            try:
                native.append({"ok": _running_status_text(*thunks[:5]), "log": thunks[5]})
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                native.append({"error": f"{type(exc).__name__}: {exc}", "log": thunks[5]})
        for case, n, f in zip(_RUNNING_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback /proc walk outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )


_MPV_VERSION_PARITY_CASES = [
    {"which": None, "stdout": "never read"},
    {"which": "", "stdout": "never read"},
    {"which": "/usr/bin/mpv", "stdout": "mpv 0.38.0 (C) 2000-2026\nmore lines\n", "stderr": None},
    {"which": "/usr/bin/mpv", "stdout": "mpv 0.38.0"},
    {"which": "/usr/bin/mpv", "stdout": None, "stderr": "built from source\n"},
    {"which": "/usr/bin/mpv", "stdout": "", "stderr": "warning line\n"},
    {"which": "/usr/bin/mpv", "stdout": "", "stderr": ""},
    {"which": "/usr/bin/mpv", "stdout": None, "stderr": None},
    {"which": "/usr/bin/mpv", "stdout": "mpv 0.1\u0085hidden tail"},
    {"which": "/usr/bin/mpv", "stdout": "first\u2028second", "stderr": None},
    {"which": "/usr/bin/mpv", "stdout": "\nmpv later", "stderr": None},
    # The stderr attribute is only read when stdout is falsy.
    {"which": "/usr/bin/mpv", "stdout": "mpv 0.38.0", "stderr_error": "RuntimeError"},
    {"which": "/usr/bin/mpv", "stdout": "", "stderr_error": "AttributeError"},
    {"which": "/usr/bin/mpv", "stdout_error": "OSError"},
    # Run errors: the two caught arms and the propagating one.
    {"which": "/mpv", "run_error": ["FileNotFoundError", "/mpv"]},
    {"which": "/mpv", "run_error": ["TimeoutExpired", "/mpv --version"]},
    {"which": "/mpv", "run_error": ["ValueError", "boom"]},
    # A weird-but-honest which result.
    {"which": " ", "stdout": "mpv 1\n"},
]

_MPV_VERSION_CHILD_PROG = """
import json
import subprocess
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _mpv_version_text

_ERRORS = {
    "OSError": OSError,
    "RuntimeError": RuntimeError,
    "AttributeError": AttributeError,
}


class _Out:
    def __init__(self, stdout, stderr, stdout_error, stderr_error):
        self._stdout = stdout
        self._stderr = stderr
        self._stdout_error = stdout_error
        self._stderr_error = stderr_error

    @property
    def stdout(self):
        if self._stdout_error:
            raise _ERRORS[self._stdout_error]("stdout broke")
        return self._stdout

    @property
    def stderr(self):
        if self._stderr_error:
            raise _ERRORS[self._stderr_error]("stderr broke")
        return self._stderr


def build_thunks(case):
    log = []

    def which_mpv():
        log.append("which")
        return case["which"]

    def run_version(mpv_bin):
        log.append(f"run:{mpv_bin}")
        err = case.get("run_error")
        if err is not None:
            kind, arg = err
            if kind == "FileNotFoundError":
                raise FileNotFoundError(arg)
            if kind == "TimeoutExpired":
                raise subprocess.TimeoutExpired(cmd=arg, timeout=3)
            if kind == "ValueError":
                raise ValueError(arg)
        return _Out(
            case.get("stdout"),
            case.get("stderr"),
            case.get("stdout_error"),
            case.get("stderr_error"),
        )

    return which_mpv, run_version, log


def outcome(case):
    which_mpv, run_version, log = build_thunks(case)
    try:
        text = _mpv_version_text(which_mpv, run_version)
        return {"ok": text, "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


cases = json.load(sys.stdin)
json.dump([outcome(case) for case in cases], sys.stdout)
"""


class TestMpvVersionNativeParity:
    """When the Rust extension is built, the native mpv probe must
    decide exactly like the pure-Python reference — same version lines,
    same markers, same probe timings, same error shapes — for every
    case, so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_mpv_version_text_matches_fallback(self):
        from ytm_player.utils.doctor import _mpv_version_text

        result = subprocess.run(
            [sys.executable, "-c", _MPV_VERSION_CHILD_PROG],
            input=json.dumps(_MPV_VERSION_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = []
        for case in _MPV_VERSION_PARITY_CASES:
            which_mpv, run_version, log = _mpv_thunks(case)
            try:
                native.append({"ok": _mpv_version_text(which_mpv, run_version), "log": log})
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                native.append({"error": f"{type(exc).__name__}: {exc}", "log": log})
        for case, n, f in zip(_MPV_VERSION_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback mpv probe outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # The probe-timing contract, pinned by the compared call logs:
        # the which probe always runs exactly once, and the subprocess
        # probe only when the which result was truthy.
        for case, n in zip(_MPV_VERSION_PARITY_CASES, native):
            assert n["log"].count("which") == 1
            if not case["which"]:
                assert len(n["log"]) == 1


# ── libmpv import-probe native vs fallback parity ──────────────────────

# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable. Case shape:
# {"msg": str, "raise_exc": str | None, "odd_str": bool,
#  "str_error": str | None} — the thunk returns the message (or an odd
# object whose __str__ renders it / raises the configured error), or
# raises the configured exception.
_LIBMPV_PARITY_CASES = [
    {"msg": "Cannot load libmpv.\n  Linux: apt install\n\nOriginal: boom"},
    {"msg": "one line"},
    {"msg": "a\nb"},  # no trailing newline
    {"msg": ""},  # splitlines() is [] — the IndexError
    {"msg": "\n"},  # splitlines() is [''] — the empty last line
    {"msg": "head\n  \t padded \t "},  # the strip
    {"msg": "head\n\x1cpadded\x1f"},  # CPython whitespace, not White_Space
    {"msg": "head\n\xa0nbsp\xa0"},
    {"msg": "a\x85b"},  # CPython line boundary
    {"msg": "a\u2028b"},  # the other one
    {"msg": "a\rb\rc"},  # \r boundaries
    {"msg": "a\r\nb"},  # the \r\n single boundary
    {"msg": "ünïcode\nlïne ✓"},  # unicode message
    # The thunk raising.
    {"msg": "", "raise_exc": "ImportError"},
    {"msg": "", "raise_exc": "RuntimeError"},
    # An odd message object whose __str__ renders the text.
    {"msg": "x\ny", "odd_str": True},
    # Whose __str__ raises — the ImportError arm is the OK marker (the
    # reference's str() sits inside its try block), the others surface.
    {"msg": "x", "odd_str": True, "str_error": "ImportError"},
    {"msg": "x", "odd_str": True, "str_error": "RuntimeError"},
]

_LIBMPV_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _libmpv_status_text

ERRORS = {"ImportError": ImportError, "RuntimeError": RuntimeError}


class OddStr:
    def __init__(self, text, str_error):
        self._text = text
        self._str_error = str_error

    def __str__(self):
        if self._str_error:
            raise ERRORS[self._str_error]("str broke")
        return self._text


def build(case):
    log = []

    def import_player():
        log.append("import")
        if case.get("raise_exc"):
            raise ERRORS[case["raise_exc"]]("probe broke")
        if case.get("odd_str"):
            return OddStr(case["msg"], case.get("str_error"))
        return case["msg"]

    return import_player, log


def outcome(case):
    import_player, log = build(case)
    try:
        return {"ok": _libmpv_status_text(import_player), "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


cases = json.load(sys.stdin)
json.dump([outcome(case) for case in cases], sys.stdout)
"""


def _libmpv_thunks(case):
    """The import thunk for a synthetic libmpv probe case, plus the log."""

    class _OddStr:
        def __init__(self, text, str_error):
            self._text = text
            self._str_error = str_error

        def __str__(self):
            if self._str_error is not None:
                raise _ERRORS[self._str_error]("str broke")
            return self._text

    log: list[str] = []

    def import_player():
        log.append("import")
        if case.get("raise_exc"):
            raise _ERRORS[case["raise_exc"]]("probe broke")
        if case.get("odd_str"):
            return _OddStr(case["msg"], case.get("str_error"))
        return case["msg"]

    return import_player, log


_ERRORS = {"ImportError": ImportError, "RuntimeError": RuntimeError}


class TestLibmpvStatusNativeParity:
    """When the Rust extension is built, the native libmpv probe must
    decide exactly like the pure-Python reference — same status lines,
    same OK markers, same error shapes — for every case, so the two
    implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_libmpv_status_text_matches_fallback(self):
        from ytm_player.utils.doctor import _libmpv_status_text

        result = subprocess.run(
            [sys.executable, "-c", _LIBMPV_CHILD_PROG],
            input=json.dumps(_LIBMPV_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = []
        for case in _LIBMPV_PARITY_CASES:
            import_player, log = _libmpv_thunks(case)
            try:
                native.append({"ok": _libmpv_status_text(import_player), "log": log})
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                native.append({"error": f"{type(exc).__name__}: {exc}", "log": log})
        for case, n, f in zip(_LIBMPV_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback libmpv probe outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # The probe-timing contract, pinned by the compared call logs:
        # the import thunk always runs exactly once.
        for n in native:
            assert n["log"] == ["import"]


# ── MPRIS import-probe native vs fallback parity ───────────────────────

# Case shape: {"is_linux": bool, "system": str, "bus_name": str,
# "dbus_available": <json value>, "pair_kind": "tuple" | "list" | "gen" |
# "str" | "non_iterable" | "short" | "long", "raise_exc": str | None,
# "system_error": str | None}.
_MPRIS_PARITY_CASES = [
    {"is_linux": False, "system": "Darwin", "bus_name": "n", "dbus_available": True},
    {"is_linux": False, "system": "Windows", "bus_name": "n", "dbus_available": False},
    {"is_linux": False, "system": "OS/2", "bus_name": "n", "dbus_available": True},
    # The system probe's own failure surfaces on the n/a branch.
    {
        "is_linux": False,
        "system": "Darwin",
        "system_error": "RuntimeError",
        "bus_name": "n",
        "dbus_available": True,
    },
    # The Linux branch — the import never even runs off Linux.
    {"is_linux": True, "system": "never read", "bus_name": "org.mpris.ytm", "dbus_available": True},
    {
        "is_linux": True,
        "system": "never read",
        "bus_name": "org.mpris.ytm",
        "dbus_available": False,
    },
    # Truthiness of the DBUS_AVAILABLE element.
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": 1},
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": 0},
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": ""},
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": None},
    # The bus name renders through str().
    {"is_linux": True, "system": "", "bus_name": "ünïcode ✓", "dbus_available": True},
    {"is_linux": True, "system": "", "bus_name": "", "dbus_available": True},
    # The import raising — an ImportError surfaces, it is not the marker.
    {
        "is_linux": True,
        "system": "",
        "bus_name": "n",
        "dbus_available": True,
        "raise_exc": "ImportError",
    },
    {
        "is_linux": True,
        "system": "",
        "bus_name": "n",
        "dbus_available": True,
        "raise_exc": "RuntimeError",
    },
    # The pair's own shape — CPython's UNPACK_SEQUENCE(2) semantics.
    {
        "is_linux": True,
        "system": "",
        "bus_name": "gen-name",
        "dbus_available": 1,
        "pair_kind": "gen",
    },
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": True, "pair_kind": "list"},
    {"is_linux": True, "system": "", "bus_name": "a", "dbus_available": "1", "pair_kind": "str"},
    {
        "is_linux": True,
        "system": "",
        "bus_name": "n",
        "dbus_available": True,
        "pair_kind": "non_iterable",
    },
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": True, "pair_kind": "short"},
    {"is_linux": True, "system": "", "bus_name": "n", "dbus_available": True, "pair_kind": "long"},
]

_MPRIS_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.utils.doctor import _mpris_status_text

ERRORS = {"ImportError": ImportError, "RuntimeError": RuntimeError}


def build_pair(case):
    kind = case.get("pair_kind", "tuple")
    bus_name = case["bus_name"]
    dbus = case["dbus_available"]
    if kind == "tuple":
        return (bus_name, dbus)
    if kind == "list":
        return [bus_name, dbus]
    if kind == "gen":
        return (v for v in (bus_name, dbus))
    if kind == "str":
        return str(bus_name) + str(dbus)
    if kind == "non_iterable":
        return 5
    if kind == "short":
        return (bus_name,)
    if kind == "long":
        return (bus_name, dbus, "extra")
    raise AssertionError(kind)


def build(case):
    log = []

    def system_name():
        log.append("system")
        if case.get("system_error"):
            raise ERRORS[case["system_error"]]("system broke")
        return case["system"]

    def import_mpris():
        log.append("import")
        if case.get("raise_exc"):
            raise ERRORS[case["raise_exc"]]("probe broke")
        return build_pair(case)

    return system_name, import_mpris, log


def outcome(case):
    system_name, import_mpris, log = build(case)
    try:
        text = _mpris_status_text(case["is_linux"], system_name, import_mpris)
        return {"ok": text, "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


cases = json.load(sys.stdin)
json.dump([outcome(case) for case in cases], sys.stdout)
"""


def _mpris_thunks(case):
    """The system + import thunks for a synthetic MPRIS probe case, plus
    the log."""
    log: list[str] = []

    def build_pair():
        kind = case.get("pair_kind", "tuple")
        bus_name = case["bus_name"]
        dbus = case["dbus_available"]
        if kind == "tuple":
            return (bus_name, dbus)
        if kind == "list":
            return [bus_name, dbus]
        if kind == "gen":
            return (v for v in (bus_name, dbus))
        if kind == "str":
            return str(bus_name) + str(dbus)
        if kind == "non_iterable":
            return 5
        if kind == "short":
            return (bus_name,)
        if kind == "long":
            return (bus_name, dbus, "extra")
        raise AssertionError(kind)

    def system_name():
        log.append("system")
        if case.get("system_error"):
            raise _ERRORS[case["system_error"]]("system broke")
        return case["system"]

    def import_mpris():
        log.append("import")
        if case.get("raise_exc"):
            raise _ERRORS[case["raise_exc"]]("probe broke")
        return build_pair()

    return system_name, import_mpris, log


class TestMprisStatusNativeParity:
    """When the Rust extension is built, the native MPRIS probe must
    decide exactly like the pure-Python reference — same verdicts, same
    n/a renders, same probe timings, same unpack error shapes — for
    every case, so the two implementations cannot drift."""

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_mpris_status_text_matches_fallback(self):
        from ytm_player.utils.doctor import _mpris_status_text

        result = subprocess.run(
            [sys.executable, "-c", _MPRIS_CHILD_PROG],
            input=json.dumps(_MPRIS_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = []
        for case in _MPRIS_PARITY_CASES:
            system_name, import_mpris, log = _mpris_thunks(case)
            try:
                native.append(
                    {
                        "ok": _mpris_status_text(case["is_linux"], system_name, import_mpris),
                        "log": log,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
                native.append({"error": f"{type(exc).__name__}: {exc}", "log": log})
        for case, n, f in zip(_MPRIS_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback MPRIS probe outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # The probe-timing contract, pinned by the compared call logs:
        # the system probe never runs on the Linux branch, and the
        # import never runs off it.
        for case, n in zip(_MPRIS_PARITY_CASES, native):
            if case["is_linux"]:
                assert n["log"] == ["import"], f"system probe ran on Linux: {case!r}"
            else:
                assert n["log"] == ["system"] or n["log"] == [], f"import ran off Linux: {case!r}"
