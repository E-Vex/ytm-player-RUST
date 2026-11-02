"""Tests for the CLI cache-clear walk engine.

``_cache_clear_walk`` is the engine behind the standalone ``ytm cache
clear`` command: the directory walk that unlinks the cache directory's
plain files and counts them, with the exists/iterdir/is_file/unlink
operations supplied as thunks the engine invokes exactly where the
reference performed them. The pure-Python transcription lives in
``cli.py``; when the Rust extension is built the same engine runs
natively (``_ytm_player_native.cache_clear_walk``).

The unit tests below assert the engine's contract on whichever backend
is loaded, plus the command glue (the files gone, the subdirectories
and their contents untouched, the separately-unlinked database, and
the confirmation gate). When the extension is built, the parity test
additionally compares the native engine against the pure-Python
reference (run in a child interpreter with the extension import
blocked) case for case — counts, unlink orders and call logs — so the
two implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from click.testing import CliRunner

from ytm_player.cli import _cache_clear_walk, main

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class _RecordingWalk:
    """Thunk quartet over a synthetic entry list, recording every call.

    *entries* are the names the listing yields; *files* the names the
    is_file thunk accepts; the optional error hooks raise instead of
    performing the named operation (``list_error`` on the listing,
    ``is_file_error`` on the named entry's file test, ``unlink_error``
    on the named entry's — or every entry's, when "all" — unlink).
    """

    def __init__(self, entries, files, *, list_error=None, is_file_error=None, unlink_error=None):
        self.entries = list(entries)
        self.files = set(files)
        self.list_error = list_error
        self.is_file_error = is_file_error
        self.unlink_error = unlink_error
        self.log: list[str] = []
        self.unlinked: list[str] = []

    def dir_exists(self):
        self.log.append("exists")
        return True

    def list_entries(self):
        self.log.append("iterdir")
        if self.list_error is not None:
            raise self.list_error
        return iter(self.entries)

    def is_file(self, entry):
        self.log.append(f"is_file:{entry}")
        if entry == self.is_file_error:
            raise RuntimeError(f"file test broke on {entry}")
        return entry in self.files

    def unlink(self, entry):
        self.log.append(f"unlink:{entry}")
        self.unlinked.append(entry)
        if self.unlink_error is not None and self.unlink_error in (entry, "all"):
            raise PermissionError(f"cannot unlink {entry}")


class TestCacheClearWalk:
    """The directory walk behind ``ytm cache clear`` — on whichever
    backend is loaded."""

    def test_unlinks_only_the_files_and_counts_them(self):
        walk = _RecordingWalk(["a.webm", "subdir", "b.webm", "sock"], ["a.webm", "b.webm"])
        removed = _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert removed == 2
        assert walk.unlinked == ["a.webm", "b.webm"]

    def test_empty_directory_removes_nothing(self):
        walk = _RecordingWalk([], [])
        assert _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink) == 0
        assert walk.unlinked == []

    def test_missing_directory_never_lists(self):
        log: list[str] = []

        def dir_exists():
            log.append("exists")
            return False

        def list_entries():
            log.append("iterdir")
            return iter(["a.webm"])

        removed = _cache_clear_walk(dir_exists, list_entries, lambda e: True, lambda e: None)
        assert removed == 0
        # The listing, the file tests and the unlinks are all gated on
        # the existence probe — a missing directory walks nothing.
        assert log == ["exists"]

    def test_unlinks_in_iteration_order(self):
        walk = _RecordingWalk(["c", "a", "b"], ["a", "b", "c"])
        _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert walk.unlinked == ["c", "a", "b"]

    def test_the_call_log_pins_the_operations_order(self):
        walk = _RecordingWalk(["dir", "a.webm"], ["a.webm"])
        _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert walk.log == ["exists", "iterdir", "is_file:dir", "is_file:a.webm", "unlink:a.webm"]

    def test_exists_probe_runs_exactly_once(self):
        walk = _RecordingWalk(["a"], ["a"])
        _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert walk.log.count("exists") == 1
        assert walk.log.count("iterdir") == 1

    def test_unlink_error_propagates_and_keeps_the_earlier_work(self):
        # The reference's unlink raises through — the walk stops, the
        # count is never returned, and the unlinks already performed
        # stay performed.
        walk = _RecordingWalk(["a", "b", "c"], ["a", "b", "c"], unlink_error="b")
        with pytest.raises(PermissionError):
            _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert walk.unlinked == ["a", "b"]  # b's thunk raised after logging it

    def test_listing_error_propagates(self):
        walk = _RecordingWalk(["a"], ["a"], list_error=OSError("unreadable"))
        with pytest.raises(OSError, match="unreadable"):
            _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert walk.unlinked == []

    def test_file_test_error_propagates(self):
        walk = _RecordingWalk(["a"], ["a"], is_file_error="a")
        with pytest.raises(RuntimeError, match="file test broke"):
            _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert walk.unlinked == []

    def test_the_count_is_an_int(self):
        walk = _RecordingWalk(["a"], ["a"])
        removed = _cache_clear_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.unlink)
        assert removed == 1
        assert type(removed) is int

    def test_real_paths_walk_like_the_reference(self):
        # Through the production thunk shape — bound Path methods and
        # the is_file/unlink lambdas the command passes — against a
        # real temporary directory.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "audio"
            cache.mkdir()
            (cache / "a.webm").write_bytes(b"x" * 4)
            (cache / "b.webm").write_bytes(b"y" * 4)
            (cache / "nested").mkdir()
            (cache / "nested" / "c.webm").write_bytes(b"z" * 4)
            outside = Path(td) / "elsewhere.webm"
            outside.write_bytes(b"w" * 4)
            (cache / "link.webm").symlink_to(outside)

            removed = _cache_clear_walk(
                cache.exists, cache.iterdir, lambda f: f.is_file(), lambda f: f.unlink()
            )

            assert removed == 3  # a, b and the symlink — is_file follows symlinks
            assert sorted(p.name for p in cache.iterdir()) == ["nested"]
            assert (cache / "nested" / "c.webm").exists()  # subdirectories are never descended
            assert outside.exists()  # the symlink's target survives its unlink


# ── Command glue ────────────────────────────────────────────────────────


@pytest.fixture
def cache_dirs(tmp_path, monkeypatch):
    """A cache directory with two files and a populated subdirectory,
    plus a cache database, both pointed at temporary paths."""
    cache_dir = tmp_path / "audio"
    cache_dir.mkdir()
    (cache_dir / "a.webm").write_bytes(b"x" * 1024)
    (cache_dir / "b.webm").write_bytes(b"y" * 2048)
    (cache_dir / "nested").mkdir()  # directories are never cleared
    (cache_dir / "nested" / "c.webm").write_bytes(b"z" * 4096)

    cache_db = tmp_path / "cache.db"
    cache_db.write_bytes(b"d" * 512)

    monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
    monkeypatch.setattr("ytm_player.cli.CACHE_DB", cache_db)
    return cache_dir, cache_db


class TestCacheClearCommand:
    def test_clears_files_keeps_subdirs_and_unlinks_the_db(self, cache_dirs):
        cache_dir, cache_db = cache_dirs
        result = CliRunner().invoke(main, ["cache", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert result.output == "Cleared 2 cached file(s).\n"
        assert not (cache_dir / "a.webm").exists()
        assert not (cache_dir / "b.webm").exists()
        assert (cache_dir / "nested").is_dir()  # the subdirectory survives
        assert (cache_dir / "nested" / "c.webm").exists()  # and so does its file
        assert not cache_db.exists()  # the database is separately unlinked

    def test_missing_directory_clears_zero(self, tmp_path, monkeypatch):
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", tmp_path / "absent")
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert result.output == "Cleared 0 cached file(s).\n"

    def test_empty_directory_clears_zero_but_still_unlinks_the_db(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        cache_db = tmp_path / "cache.db"
        cache_db.write_bytes(b"d" * 8)
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", cache_db)
        result = CliRunner().invoke(main, ["cache", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert result.output == "Cleared 0 cached file(s).\n"
        assert not cache_db.exists()

    def test_directory_without_db(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        (cache_dir / "a.webm").write_bytes(b"x")
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert result.output == "Cleared 1 cached file(s).\n"

    def test_symlinked_file_is_cleared_like_a_file(self, tmp_path, monkeypatch):
        # is_file() follows symlinks — a symlink to a regular file is a
        # cached file like any other (and unlinking it never touches the
        # target).
        cache_dir = tmp_path / "audio"
        cache_dir.mkdir()
        target = tmp_path / "elsewhere.webm"
        target.write_bytes(b"x" * 256)
        (cache_dir / "link.webm").symlink_to(target)
        monkeypatch.setattr("ytm_player.cli.CACHE_DIR", cache_dir)
        monkeypatch.setattr("ytm_player.cli.CACHE_DB", tmp_path / "absent.db")
        result = CliRunner().invoke(main, ["cache", "clear", "--yes"])
        assert result.exit_code == 0, result.output
        assert result.output == "Cleared 1 cached file(s).\n"
        assert target.exists()  # the symlink's target survives

    def test_declined_confirmation_clears_nothing(self, cache_dirs):
        cache_dir, cache_db = cache_dirs
        result = CliRunner().invoke(main, ["cache", "clear"], input="n\n")
        assert result.exit_code == 1  # the confirmation aborts before the walk
        assert (cache_dir / "a.webm").exists()
        assert (cache_dir / "b.webm").exists()
        assert cache_db.exists()

    def test_accepted_confirmation_through_the_prompt(self, cache_dirs):
        cache_dir, _ = cache_dirs
        result = CliRunner().invoke(main, ["cache", "clear"], input="y\n")
        assert result.exit_code == 0, result.output
        assert result.output.endswith("Cleared 2 cached file(s).\n")
        assert not (cache_dir / "a.webm").exists()


# ── Native vs fallback parity ──────────────────────────────────────────

# The matrices below are shared with the child interpreter as JSON, so
# every value must be JSON-round-trippable. Case shape:
# {"exists": bool, "entries": [str], "files": [str], plus the optional
# error hooks "exists_error" / "list_error" / "is_file_error" /
# "unlink_error" (+ "unlink_error_on": one entry name, or "all")}.
_CACHE_CLEAR_PARITY_CASES = [
    {"exists": True, "entries": ["a.webm", "b.webm"], "files": ["a.webm", "b.webm"]},
    {"exists": True, "entries": [], "files": []},  # the empty directory
    {"exists": False, "entries": ["a.webm"], "files": ["a.webm"]},  # never listed
    {"exists": True, "entries": ["subdir", "a.webm", "subdir2"], "files": ["a.webm"]},
    {"exists": True, "entries": ["x", "y", "z"], "files": ["x", "y", "z"]},
    {"exists": True, "entries": ["weird ñ ✓.webm"], "files": ["weird ñ ✓.webm"]},
    {"exists": True, "entries": ["weird ñ ✓.webm"], "files": []},
    # error hooks — where each operation's failure surfaces and what it
    # leaves behind.
    {
        "exists": True,
        "entries": ["a"],
        "files": ["a"],
        "unlink_error": "PermissionError",
        "unlink_error_on": "all",
    },
    {
        "exists": True,
        "entries": ["a", "b"],
        "files": ["a", "b"],
        "unlink_error": "OSError",
        "unlink_error_on": "b",
    },
    {"exists": True, "entries": ["a"], "files": ["a"], "is_file_error": "a"},
    {"exists": True, "entries": ["a"], "files": [], "list_error": "OSError"},
    {"exists": True, "entries": ["a"], "files": ["a"], "exists_error": "RuntimeError"},
]

_ERROR_TYPES = {
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
}

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _cache_clear_walk

ERROR_TYPES = {
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
}


def build(case):
    log = []

    def dir_exists():
        log.append("exists")
        if case.get("exists_error"):
            raise ERROR_TYPES[case["exists_error"]]
        return case["exists"]

    def list_entries():
        log.append("iterdir")
        if case.get("list_error"):
            raise ERROR_TYPES[case["list_error"]]
        return iter(case["entries"])

    def is_file(entry):
        log.append(f"is_file:{entry}")
        if case.get("is_file_error") == entry:
            raise RuntimeError(f"file test broke on {entry}")
        return entry in set(case["files"])

    def unlink(entry):
        log.append(f"unlink:{entry}")
        if case.get("unlink_error") and case.get("unlink_error_on") in (entry, "all"):
            raise ERROR_TYPES[case["unlink_error"]]

    return dir_exists, list_entries, is_file, unlink, log


def outcome(case):
    dir_exists, list_entries, is_file, unlink, log = build(case)
    try:
        count = _cache_clear_walk(dir_exists, list_entries, is_file, unlink)
        return {"ok": count, "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


cases = json.load(sys.stdin)
json.dump([outcome(case) for case in cases], sys.stdout)
"""


def _walk_outcome(case: dict) -> dict:
    """The parent-side twin of the child program's outcome builder —
    same thunks, same log, same error shaping, native engine."""
    log: list[str] = []

    def dir_exists():
        log.append("exists")
        if case.get("exists_error"):
            raise _ERROR_TYPES[case["exists_error"]]
        return case["exists"]

    def list_entries():
        log.append("iterdir")
        if case.get("list_error"):
            raise _ERROR_TYPES[case["list_error"]]
        return iter(case["entries"])

    def is_file(entry):
        log.append(f"is_file:{entry}")
        if case.get("is_file_error") == entry:
            raise RuntimeError(f"file test broke on {entry}")
        return entry in set(case["files"])

    def unlink(entry):
        log.append(f"unlink:{entry}")
        if case.get("unlink_error") and case.get("unlink_error_on") in (entry, "all"):
            raise _ERROR_TYPES[case["unlink_error"]]

    try:
        count = _cache_clear_walk(dir_exists, list_entries, is_file, unlink)
        return {"ok": count, "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


class TestNativeParity:
    """When the Rust extension is built, the native engine must walk
    exactly like the pure-Python reference — same counts, same unlink
    sets, same call orders, same error shapes — for every case, so the
    two implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_cache_clear_walk_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_CACHE_CLEAR_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_walk_outcome(case) for case in _CACHE_CLEAR_PARITY_CASES]
        for case, n, f in zip(_CACHE_CLEAR_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback walk outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # The operation-timing contract, pinned by the compared call
        # logs: nothing but the existence probe runs on a missing
        # directory, and every file test precedes its own unlink.
        for case, n in zip(_CACHE_CLEAR_PARITY_CASES, native):
            if not case["exists"] or "exists_error" in case:
                # A missing directory walks nothing; a raising existence
                # probe never reaches the listing either.
                assert n["log"] == ["exists"], f"walk ran without a directory: {case!r}"
            else:
                assert n["log"].count("iterdir") == 1
