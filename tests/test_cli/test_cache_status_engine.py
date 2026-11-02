"""Tests for the CLI cache-status engines.

``_cache_status_payload`` and ``_cache_status_text`` are the engines
behind the standalone ``ytm cache status`` command: the six-key status
payload assembled from the cache directory and the walk's three
aggregates, and the four aligned human lines echoed without ``--json``.
The pure-Python transcriptions live in ``cli.py``; when the Rust
extension is built the same engines run natively
(``_ytm_player_native.cache_status_payload`` /
``_ytm_player_native.cache_status_text``).

The unit tests below assert the engines' contract on whichever backend
is loaded, plus the command glue (the directory walk, the skipped
subdirectories, the separately-sized database, and the no-directory
zeros). When the extension is built, the parity test additionally
compares the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case —
payloads, texts and key orders — so the two implementations cannot
drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytm_player.cli import _cache_status_payload, _cache_status_text, main

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestCacheStatusPayload:
    """The six-key status payload behind ``ytm cache status --json``."""

    def test_payload_carries_the_six_keys(self):
        payload = _cache_status_payload(Path("/home/u/.cache/ytm/audio"), 3, 1049088, 4096)
        assert payload == {
            "cache_dir": "/home/u/.cache/ytm/audio",
            "file_count": 3,
            "total_bytes": 1049088,
            "total_human": "1.0 MB",
            "db_size_bytes": 4096,
            "db_size_human": "4.0 KB",
        }

    def test_key_order_is_the_json_output_order(self):
        # The key order is the contract the JSON output preserves — it
        # must never silently reorder.
        payload = _cache_status_payload("/x", 0, 0, 0)
        assert list(payload) == [
            "cache_dir",
            "file_count",
            "total_bytes",
            "total_human",
            "db_size_bytes",
            "db_size_human",
        ]

    def test_cache_dir_renders_through_str(self):
        # str(Path) is its path text; a str passes through as itself; a
        # Path subclass's own __str__ steers the rendering.
        assert _cache_status_payload(Path("/a/b"), 0, 0, 0)["cache_dir"] == "/a/b"
        assert _cache_status_payload("/a/b", 0, 0, 0)["cache_dir"] == "/a/b"

        class OddPath:
            def __str__(self) -> str:
                return "odd"

        assert _cache_status_payload(OddPath(), 0, 0, 0)["cache_dir"] == "odd"

    def test_aggregates_pass_through_as_the_objects_they_are(self):
        payload = _cache_status_payload("/x", True, 1024, 0)
        assert payload["file_count"] is True  # the bool object, never 1
        assert payload["total_bytes"] == 1024
        assert payload["db_size_bytes"] == 0

    @pytest.mark.parametrize(
        ("value", "human"),
        [
            (0, "0 B"),
            (1, "1 B"),
            (512, "512 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024**2, "1.0 MB"),
            (1049088, "1.0 MB"),
            (1024**3, "1.0 GB"),
            (1024**4, "1.0 TB"),
            (1024**5, "1.0 PB"),
            (7 * 1024**3 + 123, "7.0 GB"),
        ],
    )
    def test_human_spellings_at_the_unit_boundaries(self, value, human):
        payload = _cache_status_payload("/x", 0, value, value)
        assert payload["total_human"] == human
        assert payload["db_size_human"] == human

    def test_bigint_count_is_never_truncated(self):
        payload = _cache_status_payload("/x", 10**30, 0, 0)
        assert payload["file_count"] == 10**30
        assert type(payload["file_count"]) is int

    def test_fresh_map_every_call(self):
        args = ("/x", 3, 1024, 2048)
        first = _cache_status_payload(*args)
        second = _cache_status_payload(*args)
        assert first is not second
        assert first == second


class TestCacheStatusText:
    """The four aligned human lines behind ``ytm cache status``."""

    def test_exact_layout(self):
        text = _cache_status_text(Path("/home/u/.cache/ytm/audio"), 3, 1049088, 4096)
        assert text == (
            "Cache directory: /home/u/.cache/ytm/audio\n"
            "Cached files:    3\n"
            "Total size:      1.0 MB\n"
            "Database size:   4.0 KB"
        )

    def test_no_trailing_newline(self):
        # The caller's click.echo supplies the final newline, exactly
        # like the four separate echoes did.
        assert not _cache_status_text("/x", 0, 0, 0).endswith("\n")

    def test_count_interpolates_through_str(self):
        assert _cache_status_text("/x", True, 0, 0).splitlines()[1] == "Cached files:    True"

    def test_zero_directory(self):
        assert _cache_status_text("/gone", 0, 0, 0) == (
            "Cache directory: /gone\nCached files:    0\nTotal size:      0 B\nDatabase size:   0 B"
        )


# ── Command glue ────────────────────────────────────────────────────────


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    """A cache directory with two files and a skipped subdirectory, plus a
    separately-sized cache database, both pointed at temporary paths."""
    cache_dir = tmp_path / "audio"
    cache_dir.mkdir()
    (cache_dir / "a.webm").write_bytes(b"x" * 1024)
    (cache_dir / "b.webm").write_bytes(b"y" * 2048)
    (cache_dir / "nested").mkdir()  # directories are never counted
    (cache_dir / "nested" / "c.webm").write_bytes(b"z" * 4096)

    cache_db = tmp_path / "cache.db"
    cache_db.write_bytes(b"d" * 512)

    monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
    monkeypatch.setattr("ytm_player.cli.CACHE_DB", cache_db)
    return cache_dir, cache_db


class TestCacheStatusCommand:
    def test_human_output(self, cache_dirs):
        cache_dir, _ = cache_dirs
        result = CliRunner().invoke(main, ["cache", "status"])
        assert result.exit_code == 0, result.output
        assert result.output == (
            f"Cache directory: {cache_dir}\n"
            "Cached files:    2\n"
            "Total size:      3.0 KB\n"
            "Database size:   512 B\n"
        )

    def test_json_output(self, cache_dirs):
        cache_dir, _ = cache_dirs
        result = CliRunner().invoke(main, ["cache", "status", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload == {
            "cache_dir": str(cache_dir),
            "file_count": 2,
            "total_bytes": 3072,
            "total_human": "3.0 KB",
            "db_size_bytes": 512,
            "db_size_human": "512 B",
        }
        assert list(payload) == [
            "cache_dir",
            "file_count",
            "total_bytes",
            "total_human",
            "db_size_bytes",
            "db_size_human",
        ]

    def test_json_output_is_compact(self, cache_dirs):
        result = CliRunner().invoke(main, ["cache", "status", "--json"])
        assert result.exit_code == 0, result.output
        assert "\n" not in result.output.strip()

    def test_bare_cache_group_invokes_status(self, cache_dirs):
        cache_dir, _ = cache_dirs
        result = CliRunner().invoke(main, ["cache"])
        assert result.exit_code == 0, result.output
        assert result.output.startswith(f"Cache directory: {cache_dir}\n")

    def test_missing_directory_outputs_zeros(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", tmp_path / "absent")
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "status"])
        assert result.exit_code == 0, result.output
        assert result.output == (
            f"Cache directory: {tmp_path / 'absent'}\n"
            "Cached files:    0\n"
            "Total size:      0 B\n"
            "Database size:   0 B\n"
        )

    def test_empty_directory_outputs_zeros(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "status"])
        assert result.exit_code == 0, result.output
        assert "Cached files:    0\n" in result.output

    def test_database_missing_but_directory_populated(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        (cache_dir / "a.webm").write_bytes(b"x" * 1024)
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "status"])
        assert result.exit_code == 0, result.output
        assert "Total size:      1.0 KB\n" in result.output
        assert "Database size:   0 B\n" in result.output

    def test_symlinked_file_is_counted_like_a_file(self, tmp_path, monkeypatch):
        # is_file() follows symlinks — a symlink to a regular file is a
        # cached file like any other.
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        target = tmp_path / "elsewhere.webm"
        target.write_bytes(b"x" * 256)
        (cache_dir / "link.webm").symlink_to(target)
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "status"])
        assert result.exit_code == 0, result.output
        assert "Cached files:    1\n" in result.output
        assert "Total size:      256 B\n" in result.output


# ── Native vs fallback parity ──────────────────────────────────────────

# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable. (cache_dir, file_count,
# total_bytes, db_size) tuples; the sizes stay ints within st_size's
# i64 range — the walk's production shape (a size outside it would
# exercise format_size's own backend contract, locked by its own parity
# tests, not this port's), while the count runs the pass-through zoo.
_CACHE_PARITY_CASES = [
    ("/home/u/.cache/ytm-player/audio", 3, 1049088, 4096),
    ("/x", 0, 0, 0),  # the missing directory
    ("", 0, 0, 0),
    ("weird ñ path ✓", 12, 5 * 1024**3, 1024**5),
    ("/x", 1, 1023, 1024),
    ("/x", 2, 1536, 1024**2),
    ("/x", 99, 7 * 1024**3 + 123, 1024**4),
    ("/x", True, 512, 0),  # the count passes through as the bool object
    ("/x", None, 0, 0),  # a count passes through as None
    ("/x", 10**30, 1024, 2048),  # a bigint count is never truncated
    ("/x", "7", 0, 0),  # a string count passes through / renders as text
]

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _cache_status_payload, _cache_status_text


def payload_outcome(cache_dir, file_count, total_bytes, db_size):
    try:
        return {"ok": _cache_status_payload(cache_dir, file_count, total_bytes, db_size)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def text_outcome(cache_dir, file_count, total_bytes, db_size):
    try:
        return {"ok": _cache_status_text(cache_dir, file_count, total_bytes, db_size)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


cases = json.load(sys.stdin)
json.dump(
    [[payload_outcome(*case), text_outcome(*case)] for case in cases],
    sys.stdout,
)
"""


def _payload_outcome(case: tuple) -> dict:
    try:
        return {"ok": _cache_status_payload(*case)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


def _text_outcome(case: tuple) -> dict:
    try:
        return {"ok": _cache_status_text(*case)}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}"}


class TestNativeParity:
    """When the Rust extension is built, the native engines must decide
    exactly like the pure-Python references — same payloads, same texts,
    same errors — for every case, so the two implementations cannot
    drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_cache_status_engines_match_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_CACHE_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [[_payload_outcome(case), _text_outcome(case)] for case in _CACHE_PARITY_CASES]
        for case, n_pair, f_pair in zip(_CACHE_PARITY_CASES, native, fallback):
            for engine, n, f in (
                ("payload", n_pair[0], f_pair[0]),
                ("text", n_pair[1], f_pair[1]),
            ):
                assert n == f, (
                    f"native and fallback {engine} outcomes differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
        # Key order is part of the contract (the JSON output preserves
        # it) — compare the serialized payloads' key lists too.
        for n_pair, f_pair in zip(native, fallback):
            if "ok" in n_pair[0] and "ok" in f_pair[0]:
                assert list(n_pair[0]["ok"]) == list(f_pair[0]["ok"]), (
                    f"native and fallback payload key orders differ:\n"
                    f"native:   {list(n_pair[0]['ok'])!r}\n"
                    f"fallback: {list(f_pair[0]['ok'])!r}"
                )
