"""Tests for the CLI cache-status engines.

``_cache_status_payload`` and ``_cache_status_text`` are the engines
behind the standalone ``ytm cache status`` command: the six-key status
payload assembled from the cache directory and the walk's three
aggregates, and the four aligned human lines echoed without ``--json``.
``_cache_status_walk`` is the directory walk that produces two of those
aggregates — the plain-file count and the byte total — with the
exists/iterdir/is_file/stat operations supplied as thunks the engine
invokes exactly where the reference performed them. The pure-Python
transcriptions live in ``cli.py``; when the Rust extension is built the
same engines run natively
(``_ytm_player_native.cache_status_payload`` /
``_ytm_player_native.cache_status_text`` /
``_ytm_player_native.cache_status_walk``).

The unit tests below assert the engines' contract on whichever backend
is loaded, plus the command glue (the directory walk, the skipped
subdirectories, the separately-sized database, and the no-directory
zeros). When the extension is built, the parity tests additionally
compare the native engines against the pure-Python references (run in
a child interpreter with the extension import blocked) case for case —
payloads, texts, key orders, walk pairs and call logs — so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ytm_player.cli import _cache_status_payload, _cache_status_text, _cache_status_walk, main

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestCacheStatusPayload:
    """The six-key status payload behind ``ytm cache status --json``."""

    def test_payload_carries_the_six_keys(self):
        cache_dir = Path("/home/u/.cache/ytm/audio")
        payload = _cache_status_payload(cache_dir, 3, 1049088, 4096)
        assert payload == {
            "cache_dir": str(cache_dir),
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
        # str(Path) is its path text — platform-spelled, so the expectation
        # mirrors the same str(Path) rendering; a str passes through as
        # itself; a Path subclass's own __str__ steers the rendering.
        assert _cache_status_payload(Path("/a/b"), 0, 0, 0)["cache_dir"] == str(Path("/a/b"))
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
        cache_dir = Path("/home/u/.cache/ytm/audio")
        text = _cache_status_text(cache_dir, 3, 1049088, 4096)
        assert text == (
            f"Cache directory: {cache_dir}\n"
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


class _RecordingWalk:
    """Thunk quartet over a synthetic entry list, recording every call.

    *entries* are the names the listing yields; *files* the names the
    is_file thunk accepts; *sizes* the values the stat_size thunk
    returns for the accepted names (a missing name reads as 0); the
    optional error hooks raise instead of performing the named
    operation (``list_error`` on the listing, ``is_file_error`` on the
    named entry's file test, ``stat_error`` on the named entry's — or
    every entry's, when "all" — size read).
    """

    def __init__(
        self, entries, files, sizes=None, *, list_error=None, is_file_error=None, stat_error=None
    ):
        self.entries = list(entries)
        self.files = set(files)
        self.sizes = dict(sizes or {})
        self.list_error = list_error
        self.is_file_error = is_file_error
        self.stat_error = stat_error
        self.log: list[str] = []

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

    def stat_size(self, entry):
        self.log.append(f"stat:{entry}")
        if self.stat_error is not None and self.stat_error in (entry, "all"):
            raise PermissionError(f"cannot stat {entry}")
        return self.sizes.get(entry, 0)


class TestCacheStatusWalk:
    """The directory walk behind ``ytm cache status`` — on whichever
    backend is loaded."""

    def test_counts_and_sums_only_the_files(self):
        walk = _RecordingWalk(
            ["a.webm", "subdir", "b.webm", "sock"],
            ["a.webm", "b.webm"],
            {"a.webm": 1024, "b.webm": 2048},
        )
        assert _cache_status_walk(
            walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size
        ) == (2, 3072)

    def test_empty_directory_counts_nothing(self):
        walk = _RecordingWalk([], [])
        assert _cache_status_walk(
            walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size
        ) == (0, 0)

    def test_missing_directory_never_lists(self):
        log: list[str] = []

        def dir_exists():
            log.append("exists")
            return False

        def list_entries():
            log.append("iterdir")
            return iter(["a.webm"])

        def stat_size(entry):
            log.append(f"stat:{entry}")
            return 1

        assert _cache_status_walk(dir_exists, list_entries, lambda e: True, stat_size) == (0, 0)
        # The listing, the file tests and the size reads are all gated on
        # the existence probe — a missing directory walks nothing.
        assert log == ["exists"]

    def test_the_call_log_pins_the_operations_order(self):
        walk = _RecordingWalk(["dir", "a.webm"], ["a.webm"], {"a.webm": 512})
        _cache_status_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size)
        # The size read comes only for the entries that pass the file
        # test, right after their own test — subdirectories are never
        # statted.
        assert walk.log == ["exists", "iterdir", "is_file:dir", "is_file:a.webm", "stat:a.webm"]

    def test_exists_and_listing_run_exactly_once(self):
        walk = _RecordingWalk(["a"], ["a"], {"a": 1})
        _cache_status_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size)
        assert walk.log.count("exists") == 1
        assert walk.log.count("iterdir") == 1

    def test_the_pair_is_two_ints_in_count_first_order(self):
        walk = _RecordingWalk(["a", "b"], ["a", "b"], {"a": 3, "b": 4})
        pair = _cache_status_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size)
        assert type(pair) is tuple
        assert len(pair) == 2
        assert pair == (2, 7)
        assert type(pair[0]) is int
        assert type(pair[1]) is int

    def test_bigint_sizes_sum_exactly(self):
        # The totals follow Python's own += — a sum beyond every
        # fixed-width coercion stays exact, never truncated.
        walk = _RecordingWalk(["a", "b"], ["a", "b"], {"a": 10**30, "b": 10**30})
        count, total = _cache_status_walk(
            walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size
        )
        assert count == 2
        assert total == 2 * 10**30
        assert type(total) is int

    def test_float_sizes_follow_the_reference_arithmetic(self):
        # 0 + 0.5 + 1.25 — the int accumulator becomes the float the
        # reference's own += would build.
        walk = _RecordingWalk(["a", "b"], ["a", "b"], {"a": 0.5, "b": 1.25})
        count, total = _cache_status_walk(
            walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size
        )
        assert count == 2
        assert total == 1.75
        assert type(total) is float

    def test_stat_error_propagates_and_keeps_the_earlier_reads(self):
        # The reference's stat raises through — the walk stops, the pair
        # is never returned, and the size reads already performed stay
        # performed.
        walk = _RecordingWalk(["a", "b", "c"], ["a", "b", "c"], {"a": 1, "b": 2}, stat_error="b")
        with pytest.raises(PermissionError):
            _cache_status_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size)
        assert walk.log == [
            "exists",
            "iterdir",
            "is_file:a",
            "stat:a",
            "is_file:b",
            "stat:b",  # b's thunk raised after logging it
        ]

    def test_listing_error_propagates(self):
        walk = _RecordingWalk(["a"], ["a"], {"a": 1}, list_error=OSError("unreadable"))
        with pytest.raises(OSError, match="unreadable"):
            _cache_status_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size)

    def test_file_test_error_propagates(self):
        walk = _RecordingWalk(["a"], ["a"], {"a": 1}, is_file_error="a")
        with pytest.raises(RuntimeError, match="file test broke"):
            _cache_status_walk(walk.dir_exists, walk.list_entries, walk.is_file, walk.stat_size)

    def test_real_paths_walk_like_the_reference(self):
        # Through the production thunk shape — bound Path methods and
        # the is_file/stat lambdas the command passes — against a real
        # temporary directory.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cache = Path(td) / "audio"
            cache.mkdir()
            (cache / "a.webm").write_bytes(b"x" * 4)
            (cache / "b.webm").write_bytes(b"y" * 6)
            (cache / "nested").mkdir()
            (cache / "nested" / "c.webm").write_bytes(b"z" * 8)
            outside = Path(td) / "elsewhere.webm"
            outside.write_bytes(b"w" * 10)
            (cache / "link.webm").symlink_to(outside)

            count, total = _cache_status_walk(
                cache.exists, cache.iterdir, lambda f: f.is_file(), lambda f: f.stat().st_size
            )

            assert count == 3  # a, b and the symlink — is_file follows symlinks
            assert total == 20  # 4 + 6 + the target's 10
            assert (cache / "nested" / "c.webm").exists()  # subdirectories never contribute


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

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_cache_status_walk_matches_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _WALK_CHILD_PROG],
            input=json.dumps(_CACHE_WALK_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = [_walk_outcome(case) for case in _CACHE_WALK_PARITY_CASES]
        for case, n, f in zip(_CACHE_WALK_PARITY_CASES, native, fallback):
            assert n == f, (
                f"native and fallback walk outcomes differ for {case!r}:\n"
                f"native:   {n!r}\n"
                f"fallback: {f!r}"
            )
        # The operation-timing contract, pinned by the compared call
        # logs: nothing but the existence probe runs on a missing
        # directory, and every size read follows its own file test.
        for case, n in zip(_CACHE_WALK_PARITY_CASES, native):
            if not case["exists"] or "exists_error" in case:
                # A missing directory walks nothing; a raising existence
                # probe never reaches the listing either.
                assert n["log"] == ["exists"], f"walk ran without a directory: {case!r}"
            else:
                assert n["log"].count("iterdir") == 1


# The walk matrix, shared with the child interpreter as JSON like the
# payload/text one above. Case shape: {"exists": bool, "entries": [str],
# "files": [str], "sizes": {str: number}, plus the optional error hooks
# "exists_error" / "list_error" / "is_file_error" / "stat_error" (+
# "stat_error_on": one entry name, or "all")}. The sizes run the numeric
# zoo the reference's own += accepts — ints (including bigints beyond
# every fixed-width coercion), floats (the accumulator follows), bools
# (0 + True is 1) — plus the two TypeError arms, so the native engine's
# in-place-add protocol is compared against the reference's exactly.
_CACHE_WALK_PARITY_CASES = [
    {
        "exists": True,
        "entries": ["a.webm", "b.webm"],
        "files": ["a.webm", "b.webm"],
        "sizes": {"a.webm": 1024, "b.webm": 2048},
    },
    {"exists": True, "entries": [], "files": []},  # the empty directory
    {"exists": False, "entries": ["a.webm"], "files": ["a.webm"], "sizes": {"a.webm": 1}},
    {
        "exists": True,
        "entries": ["subdir", "a.webm", "subdir2"],
        "files": ["a.webm"],
        "sizes": {"a.webm": 512},
    },
    {
        "exists": True,
        "entries": ["x", "y", "z"],
        "files": ["x", "y", "z"],
        "sizes": {"x": 1, "y": 2, "z": 3},
    },
    {
        "exists": True,
        "entries": ["weird ñ ✓.webm"],
        "files": ["weird ñ ✓.webm"],
        "sizes": {"weird ñ ✓.webm": 4096},
    },
    {"exists": True, "entries": ["weird ñ ✓.webm"], "files": []},
    # The numeric zoo — the reference's own += decides the totals.
    {
        "exists": True,
        "entries": ["a", "b"],
        "files": ["a", "b"],
        "sizes": {"a": 10**30, "b": 10**30},
    },  # bigint sums stay exact
    {
        "exists": True,
        "entries": ["a", "b"],
        "files": ["a", "b"],
        "sizes": {"a": 0.5, "b": 1.25},
    },  # the int accumulator becomes a float
    {"exists": True, "entries": ["a", "b"], "files": ["a", "b"], "sizes": {"a": 1, "b": 2.5}},
    {"exists": True, "entries": ["a"], "files": ["a"], "sizes": {"a": True}},  # 0 + True is 1
    {"exists": True, "entries": ["a"], "files": ["a"], "sizes": {"a": -5}},  # negatives sum too
    {"exists": True, "entries": ["a"], "files": ["a"], "sizes": {"a": None}},  # TypeError
    {"exists": True, "entries": ["a"], "files": ["a"], "sizes": {"a": "4"}},  # TypeError
    # Error hooks — where each operation's failure surfaces and what it
    # leaves behind.
    {
        "exists": True,
        "entries": ["a", "b"],
        "files": ["a", "b"],
        "sizes": {"a": 1, "b": 2},
        "stat_error": "PermissionError",
        "stat_error_on": "b",
    },
    {
        "exists": True,
        "entries": ["a"],
        "files": ["a"],
        "sizes": {"a": 1},
        "stat_error": "OSError",
        "stat_error_on": "all",
    },
    {"exists": True, "entries": ["a"], "files": ["a"], "sizes": {"a": 1}, "is_file_error": "a"},
    {"exists": True, "entries": ["a"], "files": [], "sizes": {}, "list_error": "OSError"},
    {
        "exists": True,
        "entries": ["a"],
        "files": ["a"],
        "sizes": {"a": 1},
        "exists_error": "RuntimeError",
    },
]

_WALK_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.cli import _cache_status_walk

ERROR_TYPES = {
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
}


def build(case):
    log = []
    sizes = case.get("sizes", {})

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

    def stat_size(entry):
        log.append(f"stat:{entry}")
        if case.get("stat_error") and case.get("stat_error_on") in (entry, "all"):
            raise ERROR_TYPES[case["stat_error"]]
        return sizes.get(entry, 0)

    return dir_exists, list_entries, is_file, stat_size, log


def outcome(case):
    dir_exists, list_entries, is_file, stat_size, log = build(case)
    try:
        count, total = _cache_status_walk(dir_exists, list_entries, is_file, stat_size)
        return {"ok": [count, total], "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


cases = json.load(sys.stdin)
json.dump([outcome(case) for case in cases], sys.stdout)
"""


def _walk_outcome(case: dict) -> dict:
    """The parent-side twin of the child program's outcome builder —
    same thunks, same log, same error shaping, native engine."""
    log: list[str] = []
    sizes = case.get("sizes", {})

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

    def stat_size(entry):
        log.append(f"stat:{entry}")
        if case.get("stat_error") and case.get("stat_error_on") in (entry, "all"):
            raise _ERROR_TYPES[case["stat_error"]]
        return sizes.get(entry, 0)

    try:
        count, total = _cache_status_walk(dir_exists, list_entries, is_file, stat_size)
        return {"ok": [count, total], "log": log}
    except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
        return {"error": f"{type(exc).__name__}: {exc}", "log": log}


_ERROR_TYPES = {
    "OSError": OSError,
    "PermissionError": PermissionError,
    "RuntimeError": RuntimeError,
}
