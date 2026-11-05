"""Tests for the download-service engines.

The two pure cores of ``services/download.py`` run with two
interchangeable backends: the pure-Python transcriptions in the module
and, when the Rust extension is built, the native implementations
bound at the module's tail. ``_base_opts`` is the nine-key yt-dlp
options dict behind ``DownloadService._build_opts`` (before the
configured extras); ``_candidate_names`` is the five-filename probe
order behind ``_download_sync``'s publication walk and ``get_path``'s
lookup.

The unit tests below assert each engine's contract on whichever
backend is loaded. When the extension is built, the parity test
additionally compares the native engines against the pure-Python
references (run in a child interpreter with the extension import
blocked) case for case, so the two implementations cannot drift
apart.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from ytm_player.services.download import _base_opts, _candidate_names

try:
    from ytm_player.utils import _ytm_player_native  # type: ignore[import-not-found]
except ImportError:
    _ytm_player_native = None

_HAVE_NATIVE = _ytm_player_native is not None


class TestBaseOpts:
    """The base yt-dlp options dict a download runs with."""

    def test_keys_in_the_reference_order(self):
        opts = _base_opts("/tmp/stage/vid.%(ext)s")
        assert list(opts) == [
            "format",
            "outtmpl",
            "quiet",
            "no_warnings",
            "noplaylist",
            "extract_flat",
            "writethumbnail",
            "writeinfojson",
            "postprocessors",
        ]

    def test_the_full_dict_matches_the_reference(self):
        output_path = "/tmp/stage/dQw4w9WgXcQ.%(ext)s"
        opts = _base_opts(output_path)
        assert opts == {
            "format": "bestaudio[ext=webm]/bestaudio/best",
            "outtmpl": output_path,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "extract_flat": False,
            "writethumbnail": False,
            "writeinfojson": False,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "opus",
                    "preferredquality": "128",
                }
            ],
        }

    def test_the_output_path_passes_through_as_the_object(self):
        output_path = "staged-template"
        assert _base_opts(output_path)["outtmpl"] is output_path

    def test_every_call_builds_fresh_containers(self):
        a = _base_opts("/tmp/a.%(ext)s")
        b = _base_opts("/tmp/a.%(ext)s")
        assert a is not b
        assert a == b
        assert a["postprocessors"] is not b["postprocessors"]
        assert a["postprocessors"][0] is not b["postprocessors"][0]


class TestCandidateNames:
    """The five-filename probe order."""

    def test_the_probe_order(self):
        assert _candidate_names("dQw4w9WgXcQ") == (
            "dQw4w9WgXcQ.opus",
            "dQw4w9WgXcQ.webm",
            "dQw4w9WgXcQ.m4a",
            "dQw4w9WgXcQ.mp3",
            "dQw4w9WgXcQ.ogg",
        )

    def test_opus_is_probed_first(self):
        # The postprocessor's target extension wins the race.
        assert _candidate_names("vid")[0] == "vid.opus"

    def test_every_video_id_gets_five_names(self):
        for video_id in ("a", "with-dash", "with_underscore", "with.dot"):
            names = _candidate_names(video_id)
            assert len(names) == 5
            assert all(name.startswith(f"{video_id}.") for name in names)

    def test_the_answer_is_a_fresh_tuple_of_fresh_strings(self):
        a = _candidate_names("vid")
        b = _candidate_names("vid")
        assert a == b
        assert a is not b
        assert a[0] is not b[0]


class TestServiceIntegration:
    """The engines wired into the service's own lookups."""

    def test_get_path_probes_the_candidate_order(self, tmp_path):
        from ytm_player.services.download import DownloadService

        service = DownloadService(download_dir=tmp_path)
        # Only the LAST probe extension exists — the walk still finds it.
        (tmp_path / "vid.ogg").write_bytes(b"audio")
        assert service.get_path("vid") == tmp_path / "vid.ogg"
        assert service.is_downloaded("vid")

    def test_get_path_prefers_opus_when_several_exist(self, tmp_path):
        from ytm_player.services.download import DownloadService

        service = DownloadService(download_dir=tmp_path)
        (tmp_path / "vid.webm").write_bytes(b"a")
        (tmp_path / "vid.opus").write_bytes(b"b")
        assert service.get_path("vid") == tmp_path / "vid.opus"


# Each parity case: the engine's arguments as JSON-round-trippable
# lists, keyed by the engine's native name (the Python module-level
# fallback spells the mapping below).
_PARITY_CASES = {
    "download_base_opts": [
        ["/tmp/stage/dQw4w9WgXcQ.%(ext)s"],
        ["staged-template"],
        [""],
    ],
    "download_candidates": [["dQw4w9WgXcQ"], ["a"], ["with-dash"], ["with.dot"], [""]],
}

_PY_NAMES = {
    "download_base_opts": "_base_opts",
    "download_candidates": "_candidate_names",
}

_CHILD_PROG = """
import json
import sys

sys.modules['ytm_player.utils._ytm_player_native'] = None

from ytm_player.services import download

PY_NAMES = {
    "download_base_opts": "_base_opts",
    "download_candidates": "_candidate_names",
}


def outcome(name, args):
    fn = getattr(download, PY_NAMES[name])
    try:
        result = fn(*args)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(result, tuple):
        return {"ok": list(result)}
    return {"ok": result}


payload = json.load(sys.stdin)
out = {name: [outcome(name, args) for args in cases] for name, cases in payload.items()}
json.dump(out, sys.stdout)
"""


def _native_outcomes():
    def outcome(name, args):
        fn = getattr(sys.modules["ytm_player.services.download"], _PY_NAMES[name])
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - parity-pinned error shape
            return {"error": f"{type(exc).__name__}: {exc}"}
        if isinstance(result, tuple):
            return {"ok": list(result)}
        return {"ok": result}

    return {name: [outcome(name, args) for args in cases] for name, cases in _PARITY_CASES.items()}


class TestNativeParity:
    """When the Rust extension is built, the native engines must answer
    exactly like the pure-Python references — same dicts (key order
    included), same probe tuples — for every case, so the two
    implementations cannot drift.
    """

    @pytest.mark.skipif(not _HAVE_NATIVE, reason="native extension not built")
    def test_download_engines_match_fallback(self):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_PROG],
            input=json.dumps(_PARITY_CASES),
            capture_output=True,
            text=True,
            check=True,
        )
        fallback = json.loads(result.stdout)
        native = _native_outcomes()
        assert set(native) == set(fallback) == set(_PARITY_CASES)
        for name, cases in _PARITY_CASES.items():
            for case, n, f in zip(cases, native[name], fallback[name]):
                assert n == f, (
                    f"native and fallback {name} differ for {case!r}:\n"
                    f"native:   {n!r}\n"
                    f"fallback: {f!r}"
                )
        # Key order is part of the opts contract — pin it beyond the
        # dict equality the JSON round trip already compares.
        for case in _PARITY_CASES["download_base_opts"]:
            native_opts = _PY_NAMES["download_base_opts"]
            fn = getattr(sys.modules["ytm_player.services.download"], native_opts)
            assert list(fn(*case)) == [
                "format",
                "outtmpl",
                "quiet",
                "no_warnings",
                "noplaylist",
                "extract_flat",
                "writethumbnail",
                "writeinfojson",
                "postprocessors",
            ]
